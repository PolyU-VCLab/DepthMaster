#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DepthMaster panoramic depth evaluation.

Uses cubemap projection (perspective DepthMaster on 6 faces) to estimate
panoramic depth, then re-projects to ERP for evaluation. The cubemap
projection layout matches the one used during training (FOV=95 deg).
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Make the local `depthmaster/` package and the panoramic `eval/` helpers importable.
# Layout (after install):
#   <repo>/depthmaster/                  <- model + alignment utilities
#   <repo>/eval_panorama/eval.py  <- this file
#   <repo>/eval_panorama/eval/    <- panorama eval helpers
_FILE_DIR = Path(__file__).absolute().parent
_REPO_ROOT = _FILE_DIR.parent
for _p in (str(_REPO_ROOT), str(_FILE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from depthmaster.model import DepthMasterModel
from eval.utils.infer import run_evaluation


# ============================================================
# Rotation matrices used for cubemap sampling (resampling pixels).
# Face order: [Front, Right, Back, Left, Up, Down].
# Note: the Up/Down rotations differ from those in
# _OPENCV_W2C_ROTATIONS_NP. This is because the panorama sampler
# (_e2c_fov95) constructs `local_dir` with x going left->right and
# y going top->bottom. Compared with the OpenCV W2C convention,
# the Up and Down faces therefore need an extra y/z flip. These
# rotations are identical to `_rotations` in
# Mirror_Pano_demo/infer.py.
# ============================================================
_CUBE_ROTATIONS_NP = np.array([
    [[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]],   # Front
    [[ 0,  0, -1], [ 0,  1,  0], [ 1,  0,  0]],   # Right
    [[-1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]],   # Back
    [[ 0,  0,  1], [ 0,  1,  0], [-1,  0,  0]],   # Left
    [[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]],   # Up   (matches infer.py _rotations)
    [[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]],   # Down (matches infer.py _rotations)
], dtype=np.float32)

# ============================================================
# W2C rotations passed to the model (strict OpenCV convention:
# x-right, y-down, z-forward).
# Face order: [Front, Right, Back, Left, Up, Down].
# Identical to ``opencv_rotations`` in Mirror_Pano_demo/infer.py;
# used only when building the W2C input passed to the model.
# ============================================================
_OPENCV_W2C_ROTATIONS_NP = np.array([
    [[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]],   # Front
    [[ 0,  0, -1], [ 0,  1,  0], [ 1,  0,  0]],   # Right
    [[-1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]],   # Back
    [[ 0,  0,  1], [ 0,  1,  0], [-1,  0,  0]],   # Left
    [[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]],   # Up
    [[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]],   # Down
], dtype=np.float32)


# ============================================================
# GPU caches: sampling grids and weights for ERP -> Cubemap and Cubemap -> ERP.
# ============================================================
_e2c_grid_cache = {}   # key=(face_w, fov_deg, erp_h, erp_w, device) -> grid (6,face_w,face_w,2)
_c2e_cache = {}        # key=(face_w, fov_deg, pano_h, pano_w, device) ->
                       #   (grid (6,pano_h,pano_w,2), weight (6,1,pano_h,pano_w), face_idx valid masks)


def _build_e2c_grid(face_w: int, fov_deg: float, erp_h: int, erp_w: int, device):
    """Build the ERP -> 6 cubemap-faces sampling grid (GPU version).

    The projection formula is identical to ``_get_e2p_remap_maps`` in
    Mirror_Pano_demo/infer.py:
      For each cubemap face, the camera-space direction at pixel (i, j) is
        (x, y, z) = ((-x_max..x_max), (y_max..-y_max), 1)   # note the y flip
      It is then rotated by the per-face rotation matrix into the world (ERP)
      frame and projected onto the ERP image:
        u = atan2(X, Z) / (2*pi) + 0.5
        v = 0.5 - atan2(Y, sqrt(X^2 + Z^2)) / pi
    """
    key = (face_w, float(fov_deg), erp_h, erp_w, str(device))
    if key in _e2c_grid_cache:
        return _e2c_grid_cache[key]

    # Local direction vectors on a cubemap face. Construction matches xyz in
    # infer.py: x in [-x_max, x_max], y in [y_max, -y_max] (top-down decreasing),
    # z = 1.
    fov_rad = np.deg2rad(fov_deg)
    x_max = np.tan(fov_rad / 2.0)
    y_max = np.tan(fov_rad / 2.0)

    xs = torch.linspace(-x_max, x_max, face_w, device=device, dtype=torch.float32)
    ys = torch.linspace(y_max, -y_max, face_w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (face_w, face_w)
    ones = torch.ones_like(grid_x)
    local_dirs = torch.stack([grid_x, grid_y, ones], dim=-1)  # (H, W, 3)

    R = torch.from_numpy(_CUBE_ROTATIONS_NP).to(device=device, dtype=torch.float32)  # (6, 3, 3)

    # world_dir = local_dir @ R   (matches `out.dot(Rx).dot(Ry)` in infer.py;
    # _CUBE_ROTATIONS_NP[i] is exactly Rx @ Ry.)
    # Note: applying R from the right to local_dir corresponds to
    # einsum 'hwc,fcd->fhwd', NOT 'hwc,fdc->fhwd' (the latter would be
    # local @ R.T, which is the inverse direction).
    # shape: (6, face_w, face_w, 3)
    world_dirs = torch.einsum('hwc,fcd->fhwd', local_dirs, R)

    X = world_dirs[..., 0]
    Y = world_dirs[..., 1]
    Z = world_dirs[..., 2]

    # ERP projection. Note: OpenCV convention with y-down.
    # In infer.py: u = atan2(x, z), v = atan2(y, sqrt(x^2 + z^2));
    # coor_y = (-v / pi + 0.5) * H - 0.5 (y-down: positive going down).
    lon = torch.atan2(X, Z)                         # (-pi, pi)
    lat = torch.atan2(Y, torch.sqrt(X * X + Z * Z)) # (-pi/2, pi/2)

    u = lon / (2.0 * np.pi) + 0.5            # [0, 1]
    v = -lat / np.pi + 0.5                   # [0, 1]  (y-down)

    # Normalize to grid_sample's [-1, 1] convention.
    # ERP wraps around in longitude. ``F.grid_sample(padding_mode='border')``
    # does NOT wrap, so we keep u in [-1, 1) and bring out-of-range values back
    # via ``fmod`` (here implemented as ``u - floor(u)``).
    u = (u - torch.floor(u))                  # wrap to [0, 1)
    gx = u * 2.0 - 1.0
    gy = v * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)     # (6, face_w, face_w, 2)

    _e2c_grid_cache[key] = grid
    return grid


def _build_c2e_grid_and_weight(face_w: int, fov_deg: float, pano_h: int, pano_w: int, device):
    """Build the Cubemap -> ERP sampling grid plus soft-blending weights (GPU).

    The projection matches ``_cubemap_to_equirect_blend`` in
    Mirror_Pano_demo/infer.py:
      For every ERP pixel, compute its spherical direction (x, y, z) under the
      OpenCV y-down convention:
        theta = (u - 0.5) * 2*pi,  phi = (0.5 - v) * pi
        x = cos(phi) * sin(theta), y = sin(phi), z = cos(phi) * cos(theta)
      For each cubemap face: cam_dir = world_dir @ R_face
        front_mask = (z_cam > 1e-6)
        px = ( x_cam / z_cam / tan_half + 1) / 2 * (face_w - 1)
        py = (-y_cam / z_cam / tan_half + 1) / 2 * (face_w - 1)
        weight w = cos(|x/z/tan_half| * pi/2) * cos(|y/z/tan_half| * pi/2);
        the closer to the face center, the larger w.
    """
    key = (face_w, float(fov_deg), pano_h, pano_w, str(device))
    if key in _c2e_cache:
        return _c2e_cache[key]

    # 1. ERP per-pixel direction (matches _get_erp_directions: OpenCV y-down).
    u = torch.linspace(0.5 / pano_w, 1.0 - 0.5 / pano_w, pano_w, device=device, dtype=torch.float32)
    v = torch.linspace(0.5 / pano_h, 1.0 - 0.5 / pano_h, pano_h, device=device, dtype=torch.float32)
    grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')  # (H, W)
    theta = (grid_u - 0.5) * 2.0 * np.pi
    phi   = (0.5 - grid_v) * np.pi
    cos_phi = torch.cos(phi)
    x = cos_phi * torch.sin(theta)
    y = torch.sin(phi)
    z = cos_phi * torch.cos(theta)
    erp_dirs = torch.stack([x, y, z], dim=-1)              # (H, W, 3)

    # 2. Project onto the 6 faces.
    R = torch.from_numpy(_CUBE_ROTATIONS_NP).to(device=device, dtype=torch.float32)  # (6, 3, 3)
    # cam_dir = erp_dir @ R_face (equivalent to numpy's
    # ``np.einsum('hwc,dc->hwd', dirs, R)``).
    cam_dirs = torch.einsum('hwc,fdc->fhwd', erp_dirs, R)  # (6, H, W, 3)
    x_cam = cam_dirs[..., 0]
    y_cam = cam_dirs[..., 1]
    z_cam = cam_dirs[..., 2]
    front_mask = z_cam > 1e-6

    tan_half = np.tan(np.deg2rad(fov_deg / 2.0))
    safe_z = torch.where(front_mask, z_cam, torch.ones_like(z_cam))
    xz = x_cam / safe_z
    yz = y_cam / safe_z

    # Pixel coordinates (px, py) in [0, face_w - 1] (matches numpy version).
    px = (xz / tan_half + 1.0) / 2.0 * (face_w - 1)
    py = (-yz / tan_half + 1.0) / 2.0 * (face_w - 1)

    in_bounds = front_mask & (px >= 0) & (px <= face_w - 1) & (py >= 0) & (py <= face_w - 1)

    # 3. Soft-blending weight: cos(u * pi / 2) * cos(v * pi / 2).
    u_norm = torch.clamp(torch.where(in_bounds, torch.abs(xz / tan_half), torch.ones_like(xz)), 0.0, 1.0)
    v_norm = torch.clamp(torch.where(in_bounds, torch.abs(yz / tan_half), torch.ones_like(yz)), 0.0, 1.0)
    w = torch.where(
        in_bounds,
        torch.cos(u_norm * (np.pi / 2.0)) * torch.cos(v_norm * (np.pi / 2.0)),
        torch.zeros_like(u_norm),
    )  # (6, H, W)

    # 4. Normalize to grid_sample's [-1, 1] range.
    # px in [0, face_w - 1]  ->  gx = (px / (face_w - 1)) * 2 - 1
    # (matches align_corners=True).
    gx = (px / (face_w - 1)) * 2.0 - 1.0
    gy = (py / (face_w - 1)) * 2.0 - 1.0
    # Out-of-bounds positions are filled with valid coordinates (their
    # contribution is suppressed by w = 0 anyway).
    gx = torch.where(in_bounds, gx, torch.zeros_like(gx))
    gy = torch.where(in_bounds, gy, torch.zeros_like(gy))
    grid = torch.stack([gx, gy], dim=-1)   # (6, H, W, 2)

    result = (grid, w.unsqueeze(1))        # w: (6, 1, H, W)
    _c2e_cache[key] = result
    return result


def erp_to_cubemap_gpu(erp_img: torch.Tensor, face_w: int, fov_deg: float = 95.0,
                       mode: str = 'bilinear') -> torch.Tensor:
    """GPU implementation of ERP -> Cubemap.

    Args:
        erp_img: ``(C, H, W)`` or ``(B, C, H, W)`` float tensor. The value range
            is arbitrary (typically [0, 1] or [0, 255]).
        face_w: Resolution of each cubemap face.
        fov_deg: Cubemap FOV in degrees.
        mode: grid_sample interpolation mode.

    Returns:
        ``(6, C, face_w, face_w)`` or ``(B, 6, C, face_w, face_w)``.
    """
    squeeze_batch = False
    if erp_img.dim() == 3:
        erp_img = erp_img.unsqueeze(0)
        squeeze_batch = True
    B, C, H, W = erp_img.shape
    device = erp_img.device

    grid = _build_e2c_grid(face_w, fov_deg, H, W, device)       # (6, fw, fw, 2)
    grid = grid.to(erp_img.dtype)

    # Reshape across batches and faces. Combine to (B*6, C, H, W).
    erp_rep = erp_img.unsqueeze(1).expand(B, 6, C, H, W).reshape(B * 6, C, H, W)
    grid_rep = grid.unsqueeze(0).expand(B, 6, face_w, face_w, 2).reshape(B * 6, face_w, face_w, 2)

    faces = F.grid_sample(erp_rep, grid_rep, mode=mode, padding_mode='border', align_corners=True)
    faces = faces.reshape(B, 6, C, face_w, face_w)
    if squeeze_batch:
        faces = faces.squeeze(0)
    return faces


def cubemap_to_erp_gpu(cube_faces: torch.Tensor, pano_h: int, pano_w: int,
                       fov_deg: float = 95.0, mode: str = 'bilinear') -> torch.Tensor:
    """GPU implementation of Cubemap -> ERP with soft blending to remove seams.

    Args:
        cube_faces: ``(6, H, W)``, ``(6, C, H, W)`` or ``(B, 6, C, H, W)`` float tensor.
        pano_h, pano_w: Output ERP size.
        fov_deg: Cubemap FOV.

    Returns:
        ``(pano_h, pano_w)`` / ``(C, pano_h, pano_w)`` / ``(B, C, pano_h, pano_w)``,
        matching the dimensionality of the input.
    """
    orig_dim = cube_faces.dim()
    # Normalize the shape to (B, 6, C, H, W).
    if cube_faces.dim() == 3:       # (6, H, W) → (1, 6, 1, H, W)
        x = cube_faces.unsqueeze(1).unsqueeze(0)
    elif cube_faces.dim() == 4:     # (6, C, H, W) → (1, 6, C, H, W)
        x = cube_faces.unsqueeze(0)
    elif cube_faces.dim() == 5:     # (B, 6, C, H, W)
        x = cube_faces
    else:
        raise ValueError(f"Unsupported cube_faces shape: {tuple(cube_faces.shape)}")

    B, _, C, face_h, face_w = x.shape
    assert face_h == face_w, "Cubemap faces must be square."
    device = x.device

    grid, weight = _build_c2e_grid_and_weight(face_w, fov_deg, pano_h, pano_w, device)
    grid = grid.to(x.dtype)                 # (6, pano_h, pano_w, 2)
    weight = weight.to(x.dtype)             # (6, 1, pano_h, pano_w)

    # Merge B x 6 and run grid_sample face by face.
    # For each face f, sample (B, C, face, face) using grid (B, pano_h, pano_w, 2).
    out = x.new_zeros((B, C, pano_h, pano_w))
    w_sum = x.new_zeros((B, 1, pano_h, pano_w))
    for f in range(6):
        face_img = x[:, f]                                         # (B, C, face, face)
        g = grid[f].unsqueeze(0).expand(B, -1, -1, -1)            # (B, pano_h, pano_w, 2)
        sampled = F.grid_sample(face_img, g, mode=mode,
                                padding_mode='border', align_corners=True)  # (B, C, pano_h, pano_w)
        wf = weight[f].unsqueeze(0).expand(B, -1, -1, -1)          # (B, 1, pano_h, pano_w)
        out = out + sampled * wf
        w_sum = w_sum + wf

    safe_w = torch.where(w_sum > 1e-8, w_sum, torch.ones_like(w_sum))
    out = torch.where(w_sum > 1e-8, out / safe_w, torch.zeros_like(out))

    if orig_dim == 3:
        return out.squeeze(0).squeeze(0)   # (pano_h, pano_w)
    if orig_dim == 4:
        return out.squeeze(0)              # (C, pano_h, pano_w)
    return out                             # (B, C, pano_h, pano_w)


# ============================================================
# Model-related helpers (W2C is identical to Mirror_Pano_demo/infer.py).
# ============================================================

def _get_cubemap_K_W2C(fov_deg: float, face_w: int):
    """Return the intrinsics and W2C of the 6 cubemap faces (OpenCV convention).

    Identical to ``_get_cubemap_K_W2C`` in Mirror_Pano_demo/infer.py.
    Face order: [Front, Right, Back, Left, Up, Down].
    """
    f = (face_w - 1) / (2.0 * np.tan(0.5 * np.radians(fov_deg)))
    cx = (face_w - 1) / 2.0
    cy = (face_w - 1) / 2.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32)
    K_normalized = K.copy()
    K_normalized[0, 0] /= face_w
    K_normalized[1, 1] /= face_w
    K_normalized[0, 2] /= face_w
    K_normalized[1, 2] /= face_w

    K_list, W2C_list = [], []
    for R in _OPENCV_W2C_ROTATIONS_NP:
        W2C = np.eye(4, dtype=np.float32)
        W2C[:3, :3] = R
        K_list.append(K_normalized.copy())
        W2C_list.append(W2C)

    return np.stack(K_list), np.stack(W2C_list)


# Global cache for camera parameters.
_cached_camera_params = {}

def _get_camera_params_cached(fov_deg: float, face_w: int, device, dtype):
    """Return cached camera parameter tensors."""
    key = (fov_deg, face_w, str(device), dtype)
    if key not in _cached_camera_params:
        K_np, W2C_np = _get_cubemap_K_W2C(fov_deg=fov_deg, face_w=face_w)
        W2C = torch.from_numpy(W2C_np).to(device=device, dtype=dtype)
        intrinsics = torch.from_numpy(K_np).to(device=device, dtype=dtype)
        _cached_camera_params[key] = (W2C, intrinsics)
    return _cached_camera_params[key]


def load_depthmaster_model(pretrained_path, device=None):
    """Load a DepthMaster model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = DepthMasterModel.from_pretrained(pretrained_path).to(device).eval()

    print(f"[OK] DepthMaster model loaded: {pretrained_path}")
    return model


def depthmaster_predict_fn(model, rgb_int, device, cubemap_size=518, fov_deg=95.0):
    """DepthMaster panorama prediction (GPU-accelerated).

    The projection logic mirrors Mirror_Pano_demo/infer.py:
      1. ERP -> Cubemap   (GPU torch.grid_sample)
      2. Model inference  (W2C and intrinsics identical to Mirror_Pano_demo)
      3. Cubemap -> ERP   (GPU torch.grid_sample with soft-blending weights)

    Args:
        model: DepthMaster model.
        rgb_int: RGB image tensor with shape ``[B, C, H, W]`` and value range
            ``[0, 255]``.
        device: Compute device.
        cubemap_size: Per-face cubemap resolution.
        fov_deg: Cubemap FOV in degrees.

    Returns:
        depth_pred: numpy array with shape ``[H, W]``; predicted range depth in
        meters.
    """
    # rgb_int: [B, C, H, W], range [0, 255]
    rgb = rgb_int.to(device=device, dtype=torch.float32)
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    B, C, H, W = rgb.shape
    assert B == 1, "predict_fn currently supports batch size 1 only."
    rgb01 = rgb / 255.0   # (1, 3, H, W)

    with torch.inference_mode():
        # 1. ERP → Cubemap（GPU）
        faces = erp_to_cubemap_gpu(rgb01, face_w=cubemap_size, fov_deg=fov_deg, mode='bilinear')
        # faces: (1, 6, 3, cubemap_size, cubemap_size)

        # 2. Fetch cached camera parameters.
        W2C, intrinsics = _get_camera_params_cached(fov_deg, cubemap_size, device, torch.float32)

        # 3. Build the model input.
        cubemap_input = faces.to(dtype=model.dtype)                   # (1, 6, 3, H, W)
        W2C_input = W2C.unsqueeze(0).to(dtype=model.dtype)            # (1, 6, 4, 4)
        intrinsics_input = intrinsics.unsqueeze(0).to(dtype=model.dtype)  # (1, 6, 3, 3)

        # 4. Model inference.
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            raw_output = model.forward(
                cubemap_input,
                num_tokens=0,
                camera_type="Panorama",
                W2C=W2C_input,
                intrinsics=intrinsics_input,
            )

        # 5. Extract points and compute range depth.
        points = raw_output.get('pts3d', None)
        metric_scale = raw_output.get('metric_scale', None)

        if points is not None:
            points = points.float()
            if metric_scale is not None:
                points = points * metric_scale.float()[:, None, None, None]
            points = points.squeeze(0)                                # (6, H, W, 3)
            depth_faces = torch.sqrt((points * points).sum(dim=-1))   # (6, H, W)
        else:
            depth_raw = raw_output.get('depth', None)
            if depth_raw is None:
                raise RuntimeError("Model output contains neither pts3d nor depth.")
            depth_raw = depth_raw.float()
            if metric_scale is not None:
                depth_raw = depth_raw * metric_scale.float()[:, None, None]
            depth_faces = depth_raw.squeeze(0)                         # (6, H, W)

        depth_faces = torch.nan_to_num(depth_faces, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

        # 6. Cubemap -> ERP (GPU soft-blending).
        # depth_faces: (6, H, W)  ->  erp_depth: (H, W)
        # Optimization: a cubemap of resolution cubemap_size has a "natural"
        # equivalent ERP resolution of about (cubemap_size * 2, cubemap_size * 4),
        # i.e. cubemap_size pixels per 90 deg around the central direction.
        # Going beyond that is just upsampling that adds no information, so we
        # first project to the natural size and then bilinearly upsample to the
        # target. This significantly cuts grid_sample compute (e.g. for 2D3DS
        # 2048x4096 -> 1036x2072, roughly a 4x speedup).
        cap_h = cubemap_size * 2      # natural polar-angle resolution upper bound (180 / (90 / cubemap_size)).
        cap_w = cubemap_size * 4      # natural azimuth resolution upper bound (360 / (90 / cubemap_size)).
        proj_h = min(H, cap_h)
        proj_w = min(W, cap_w)

        erp_depth = cubemap_to_erp_gpu(depth_faces, pano_h=proj_h, pano_w=proj_w,
                                        fov_deg=fov_deg, mode='bilinear')
        # erp_depth: (proj_h, proj_w)

        # If the projected size is smaller than the GT size, bilinearly upsample.
        if proj_h != H or proj_w != W:
            erp_depth = F.interpolate(
                erp_depth[None, None],                # (1, 1, proj_h, proj_w)
                size=(H, W),
                mode='bilinear',
                align_corners=True,
            )[0, 0]

        erp_depth = erp_depth.clamp_min(1e-6)

    return erp_depth.detach().cpu().numpy().astype(np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='DepthMaster panoramic depth evaluation')
    parser.add_argument('--config', type=str, default='configs/eval_panorama.json',
                        help='Path to the panoramic evaluation config (json).')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Path to a DepthMaster checkpoint (.pt).')
    parser.add_argument('--cubemap_size', type=int, default=518,
                        help='Cubemap face resolution (default 518, multiple of patch_size=14).')
    parser.add_argument('--fov_deg', type=float, default=95.0,
                        help='Cubemap FOV in degrees (default 95, matches training).')
    parser.add_argument('--output_dir', type=str, default='output/eval_panorama',
                        help='Output directory.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the evaluation configuration.
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Load the DepthMaster model.
    model = load_depthmaster_model(pretrained_path=args.pretrained, device=device)

    # Output directory.
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Build the predict_fn closure with the chosen settings.
    cubemap_size = args.cubemap_size
    fov_deg = args.fov_deg

    def predict_fn(model, rgb_int, device):
        return depthmaster_predict_fn(model, rgb_int, device, cubemap_size=cubemap_size, fov_deg=fov_deg)

    # Evaluate each dataset.
    eval_datasets = config['evaluation']['datasets']

    print(f"\n{'='*60}")
    print(f"DepthMaster panoramic depth evaluation (GPU-accelerated projection)")
    print(f"Model:           {args.pretrained}")
    print(f"Cubemap size:    {cubemap_size}x{cubemap_size}, FOV: {fov_deg} deg")
    print(f"Alignment:       {config['evaluation']['alignment']}")
    print(f"Metric eval:     {config['evaluation'].get('metric_depth_eval', False)}")
    print(f"Datasets:        {list(eval_datasets.keys())}")
    print(f"{'='*60}\n")

    all_results = {}

    with torch.no_grad():
        for dataset_name in eval_datasets.keys():
            print(f"\n--- Evaluating dataset: {dataset_name} ---")
            metrics = run_evaluation(
                model=model,
                config=config,
                dataset_name=dataset_name,
                output_dir=output_dir,
                device=device,
                predict_fn=predict_fn
            )
            all_results[dataset_name] = metrics

            print(f"\n{dataset_name} results:")
            grouped = {}
            for metric_name, value in metrics.items():
                if '/' in metric_name:
                    group, name = metric_name.split('/', 1)
                else:
                    group, name = 'default', metric_name
                if group not in grouped:
                    grouped[group] = {}
                grouped[group][name] = value

            for group, group_metrics in grouped.items():
                print(f"  [{group}]")
                for name, value in group_metrics.items():
                    print(f"    {name}: {value:.6f}")

    # Save the aggregated results.
    results_path = os.path.join(output_dir, 'results_summary.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
