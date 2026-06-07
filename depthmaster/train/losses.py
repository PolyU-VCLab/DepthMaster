from typing import *
import math

import torch
import torch.nn.functional as F
import utils3d

from ..utils.geometry_torch import (
    weighted_mean, 
    harmonic_mean, 
    geometric_mean,
    normalized_view_plane_uv,
    angle_diff_vec3
)
from ..utils.alignment import (
    align_points_scale_z_shift, 
    align_points_scale, 
    align_points_scale_xyz_shift,
    align_points_z_shift,
    align_depth_affine,
    align_depth_scale,
    align_affine_lstsq,
)


def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


def affine_invariant_global_loss(
    pred_points: torch.Tensor, 
    gt_points: torch.Tensor, 
    align_resolution: int = 64, 
    beta: float = 0.0, 
    trunc: float = 1.0, 
    sparsity_aware: bool = False
):
    device = pred_points.device

    mask = torch.isfinite(gt_points).all(dim=-1)
    gt_points = torch.where(mask[..., None], gt_points, 1)

    # Align
    pred_points_lr, gt_points_lr, lr_mask = utils3d.pt.masked_nearest_resize(pred_points, gt_points, mask=mask, size=(align_resolution, align_resolution))
    scale, shift = align_points_scale_z_shift(pred_points_lr.flatten(-3, -2), gt_points_lr.flatten(-3, -2), lr_mask.flatten(-2, -1) / gt_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2), trunc=trunc)
    valid = scale > 0
    scale, shift = torch.where(valid, scale, 0), torch.where(valid[..., None], shift, 0)

    pred_points = scale[..., None, None, None] * pred_points + shift[..., None, None, :]

    # Compute loss
    weight = (valid[..., None, None] & mask).float() / gt_points[..., 2].clamp_min(1e-5)
    weight = weight.clamp_max(10.0 * weighted_mean(weight, mask, dim=(-2, -1), keepdim=True))   # In case your data contains extremely small depth values
    loss = _smooth((pred_points - gt_points).abs() * weight[..., None], beta=beta).mean(dim=(-3, -2, -1))

    if sparsity_aware:
        # Reweighting improves performance on sparse depth data. NOTE: this is not used in DepthMaster-1.
        sparsity = mask.float().mean(dim=(-2, -1)) / lr_mask.float().mean(dim=(-2, -1))
        loss = loss / (sparsity + 1e-7)

    err = (pred_points.detach() - gt_points).norm(dim=-1) / gt_points[..., 2]

    # Record any scalar metric
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1.0), mask).item(),
        'delta': weighted_mean((err < 1).float(), mask).item()
    }

    return loss, misc, scale.detach()


def z_aligned_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    align_resolution: int = 64,
    beta: float = 0.0,
    trunc: float = 1.0,
):
    """
    Perspective-only z loss.

    Reuse the same scale + z-shift alignment as affine_invariant_global_loss,
    but only supervise the aligned z channel to emphasize depth geometry while
    keeping the inv_log point parameterization unchanged.
    """
    mask = torch.isfinite(gt_points).all(dim=-1)
    gt_points_safe = torch.where(mask[..., None], gt_points, 1)

    pred_points_lr, gt_points_lr, lr_mask = utils3d.pt.masked_nearest_resize(
        pred_points, gt_points_safe, mask=mask, size=(align_resolution, align_resolution)
    )
    scale, shift = align_points_scale_z_shift(
        pred_points_lr.flatten(-3, -2),
        gt_points_lr.flatten(-3, -2),
        lr_mask.flatten(-2, -1) / gt_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2),
        trunc=trunc,
    )
    valid = scale > 0
    scale = torch.where(valid, scale, torch.zeros_like(scale))
    shift = torch.where(valid[..., None], shift, torch.zeros_like(shift))

    pred_z = scale[..., None, None] * pred_points[..., 2] + shift[..., 2, None, None]
    gt_z = gt_points_safe[..., 2]

    weight = (valid[..., None, None] & mask).float() / gt_z.clamp_min(1e-5)
    weight = weight.clamp_max(10.0 * weighted_mean(weight, mask, dim=(-2, -1), keepdim=True))
    loss = _smooth((pred_z - gt_z).abs() * weight, beta=beta).mean(dim=(-2, -1))

    err = (pred_z.detach() - gt_z).abs() / gt_z.clamp_min(1e-5)
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1.0), mask).item(),
        'delta': weighted_mean((err < 0.25).float(), mask).item(),
    }

    return loss, misc


def z_scale_aligned_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    align_resolution: int = 64,
    beta: float = 0.0,
    trunc: float = 1.0,
):
    """
    Scale-invariant z loss (与 evaluation 协议 depth_scale_invariant 对齐).

    与 z_aligned_loss 的区别：
    - z_aligned_loss: 在 3D 点云上求 (scale, z_shift)，对 z 通道做 affine 对齐
    - z_scale_aligned_loss: 把 z 通道单独拿出来在 1D 上求 *只有* scale 的对齐
      （无 shift），完全镜像 evaluation 的 depth_scale_invariant align_depth_scale。

    这是更严的约束：模型必须自己学会把 z 通道的 origin 学对，不能依赖 shift。
    """
    mask = torch.isfinite(gt_points).all(dim=-1)
    gt_points_safe = torch.where(mask[..., None], gt_points, 1)

    pred_points_lr, gt_points_lr, lr_mask = utils3d.pt.masked_nearest_resize(
        pred_points, gt_points_safe, mask=mask,
        size=(align_resolution, align_resolution)
    )

    # 把 z 通道单独拿出来，flatten 成 1D 后做 scale-only 对齐（沿用 z_aligned 的 unsqueeze 模式）
    pred_z_lr = pred_points_lr[..., 2].flatten(-2).unsqueeze(0)         # (1, B?, N) -> 兼容无 batch
    gt_z_lr   = gt_points_lr[..., 2].flatten(-2).unsqueeze(0)
    weight_lr = (lr_mask.flatten(-2) / gt_z_lr.squeeze(0).clamp_min(1e-2)).unsqueeze(0)

    scale = align_depth_scale(pred_z_lr, gt_z_lr, weight_lr, trunc=trunc)
    scale = scale.squeeze(0)

    valid = scale > 0
    scale = torch.where(valid, scale, torch.zeros_like(scale))

    # 监督 full-res z 通道（无 shift）
    pred_z = scale[..., None, None] * pred_points[..., 2]
    gt_z   = gt_points_safe[..., 2]

    weight = (valid[..., None, None] & mask).float() / gt_z.clamp_min(1e-5)
    weight = weight.clamp_max(10.0 * weighted_mean(weight, mask, dim=(-2, -1), keepdim=True))
    loss = _smooth((pred_z - gt_z).abs() * weight, beta=beta).mean(dim=(-2, -1))

    err = (pred_z.detach() - gt_z).abs() / gt_z.clamp_min(1e-5)
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1.0), mask).item(),
        'delta': weighted_mean((err < 0.25).float(), mask).item(),
    }
    return loss, misc


def camera_consistency_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    beta: float = math.radians(3.0),
    max_angle: float = math.radians(90.0),
    delta_angle: float = math.radians(5.0),
    eps: float = 1e-6,
):
    """
    Perspective-only camera consistency prior.

    The loss is applied on raw predicted points (without any alignment) and
    constrains each point to stay on the camera ray of its pixel. The target ray
    is obtained from the GT point map, so the constraint only depends on the
    pixel ray direction rather than absolute depth.
    """
    gt_mask = torch.isfinite(gt_points).all(dim=-1)
    pred_mask = torch.isfinite(pred_points).all(dim=-1)
    valid = gt_mask & pred_mask & (gt_points.norm(dim=-1) > eps) & (pred_points.norm(dim=-1) > eps)

    gt_points_safe = torch.where(gt_mask[..., None], gt_points, torch.ones_like(gt_points))
    pred_points_safe = torch.where(pred_mask[..., None], pred_points, torch.ones_like(pred_points))

    angle = angle_diff_vec3(pred_points_safe, gt_points_safe).clamp_max(max_angle)
    loss = weighted_mean(_smooth(angle, beta=beta), valid, dim=(-2, -1))

    misc = {
        'truncated_error': weighted_mean(angle.detach(), valid).item(),
        'delta': weighted_mean((angle.detach() < delta_angle).float(), valid).item(),
    }

    return loss, misc


def _build_cubemap_seam_index_pairs(H: int, W: int, device: torch.device):
    """
    [已弃用] 此函数基于旧坐标约定（部分面 det=-1），不再被 cubemap_seam_loss 使用。
    cubemap_seam_loss 现在使用 _build_correspondences（基于 OpenCV 标准坐标系）来建立对应关系。
    保留此函数仅用于向后兼容和测试脚本。
    
    注意：此函数的索引配对逻辑基于旧的 _xyzcube_fov95 坐标约定，
    与当前 OpenCV 标准坐标系不一致。如需使用，请改用 _build_correspondences。
    
    构造 cubemap 6 个面之间 12 条物理 seam 的像素对应表。

    几何约定严格对齐 depthmaster/train/dataset_readers.py::_xyzcube_fov95 的面布局：
      - 面顺序: 0=Front(z=+0.5) 1=Right(x=+0.5) 2=Back(z=-0.5) 3=Left(x=-0.5) 4=Up(y=+0.5) 5=Down(y=-0.5)
      - 每个面内网格: grid[r,c]=(rng[c], -rng[r]) * tan(fov/2), rng=linspace(-0.5,+0.5,W)
      - 即 row r 增大 → 对应的轴 (y 或 z) 数值 *减小*，col c 增大 → 对应的轴 (x 或 z) 数值 *增大*
      - 特别地: Front/Back 面 (r,c) 映射到 (x[c], y[r]); Left/Right 面 (r,c) 映射到 (y[r], z[c]);
        Up/Down 面 (r,c) 映射到 (x[c], z=-rng[r]*s)
      - 注意本工程的 Up/Down 面 *在 pred/gt 张量里是未经 rot90 的 raw 张量*，所以直接按
        _xyzcube_fov95 的布局来推导即可。

    返回: face_a, ra, ca, face_b, rb, cb 六个长度为 N 的 LongTensor。
    每个 k-th 元素给出一对 seam 像素 (face_a[k], ra[k], ca[k]) ↔ (face_b[k], rb[k], cb[k])，
    这两个像素在 3D 世界系里对应立方体棱上的同一个点。

    一共 12 条物理 seam (每条只列一次):
      横向 4 条: Front↔Right, Right↔Back, Back↔Left, Left↔Front (沿 y 对齐，row 对 row)
      Up ↔ 4 侧面 4 条:  Up-Front, Up-Right, Up-Back, Up-Left
      Down ↔ 4 侧面 4 条: Down-Front, Down-Right, Down-Back, Down-Left
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    assert H == W, f"cubemap 要求正方形面, 当前 H={H}, W={W}"
    idx = torch.arange(H, device=device, dtype=torch.long)
    zeros = torch.zeros(H, device=device, dtype=torch.long)
    last = torch.full((H,), H - 1, device=device, dtype=torch.long)
    rev = (H - 1) - idx  # (H,)

    fa_list, ra_list, ca_list = [], [], []
    fb_list, rb_list, cb_list = [], [], []

    def _add(fa, ra, ca, fb, rb, cb):
        N = ra.shape[0]
        fa_list.append(torch.full((N,), fa, device=device, dtype=torch.long))
        fb_list.append(torch.full((N,), fb, device=device, dtype=torch.long))
        ra_list.append(ra); ca_list.append(ca)
        rb_list.append(rb); cb_list.append(cb)

    # ========== 横向 4 条 seam (row 对 row, y 轴共享) ==========
    # Front[r, W-1]  ↔ Right[r, W-1]    (共享 x=+0.5, z=+0.5 棱)
    _add(0, idx, last,  1, idx, last)
    # Right[r, 0]    ↔ Back[r, W-1]     (共享 x=+0.5, z=-0.5 棱)
    _add(1, idx, zeros, 2, idx, last)
    # Back[r, 0]     ↔ Left[r, 0]       (共享 x=-0.5, z=-0.5 棱)
    _add(2, idx, zeros, 3, idx, zeros)
    # Left[r, W-1]   ↔ Front[r, 0]      (共享 x=-0.5, z=+0.5 棱)
    _add(3, idx, last,  0, idx, zeros)

    # ========== Up 面 ↔ 4 侧面 (共享 y=+0.5 棱) ==========
    # Up[0, c]       ↔ Front[0, c]      (z=+0.5 棱, x 对 x)
    _add(4, zeros, idx, 0, zeros, idx)
    # Up[H-1-c, W-1] ↔ Right[0, c]      (x=+0.5 棱, z 对 z, 方向翻转)
    _add(4, rev, last,  1, zeros, idx)
    # Up[H-1, c]     ↔ Back[0, c]       (z=-0.5 棱, x 对 x)
    _add(4, last, idx,  2, zeros, idx)
    # Up[H-1-c, 0]   ↔ Left[0, c]       (x=-0.5 棱, z 对 z, 方向翻转)
    _add(4, rev, zeros, 3, zeros, idx)

    # ========== Down 面 ↔ 4 侧面 (共享 y=-0.5 棱) ==========
    # Down[0, c]     ↔ Front[H-1, c]    (z=+0.5 棱)
    _add(5, zeros, idx, 0, last, idx)
    # Down[H-1-c, W-1] ↔ Right[H-1, c]  (x=+0.5 棱, 方向翻转)
    _add(5, rev, last,  1, last, idx)
    # Down[H-1, c]   ↔ Back[H-1, c]     (z=-0.5 棱)
    _add(5, last, idx,  2, last, idx)
    # Down[H-1-c, 0] ↔ Left[H-1, c]     (x=-0.5 棱, 方向翻转)
    _add(5, rev, zeros, 3, last, idx)

    face_a = torch.cat(fa_list, dim=0)
    ra = torch.cat(ra_list, dim=0)
    ca = torch.cat(ca_list, dim=0)
    face_b = torch.cat(fb_list, dim=0)
    rb = torch.cat(rb_list, dim=0)
    cb = torch.cat(cb_list, dim=0)
    return face_a, ra, ca, face_b, rb, cb


def cubemap_seam_loss(
    pred_points_world: torch.Tensor,
    gt_points_world: torch.Tensor,
    fov_deg: float = 95.0,
    beta: float = 0.0,
    max_correspondences: int = 4096,
):
    """
    Cubemap 接缝 loss：通过单应性投影 + grid_sample 在 overlap 区域找精确对应点，
    约束相邻两面预测的世界系 3D 点一致。

    不需要 scale/shift 对齐——直接约束预测的相对一致性。
    对应关系由 _build_correspondences 通过单应性矩阵精确计算（亚像素精度），
    比旧的边缘像素配对方法精度提升 ~1000x。

    Args:
        pred_points_world: (V=6, H, W, 3) 预测世界系点云
        gt_points_world:   (V=6, H, W, 3) GT 世界系点云
        fov_deg: cubemap 的视场角（度），默认 95.0
        beta: smooth L1 参数
        max_correspondences: 每对面的最大对应点数，超过则随机采样

    Returns:
        loss: 标量 tensor
        misc: dict，包含 seam_loss / truncated_error / delta / num_pairs
    """
    assert pred_points_world.dim() == 4 and pred_points_world.shape[0] == 6, \
        f"pred_points_world 应为 (6,H,W,3), got {pred_points_world.shape}"
    device = pred_points_world.device
    dtype = pred_points_world.dtype
    V, H, W, _ = pred_points_world.shape

    # 构建对应关系（带缓存），face_size 使用点云的空间尺寸
    correspondences = _build_correspondences(fov_deg, H, device)

    if len(correspondences) == 0:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return zero, {'seam_loss': 0.0, 'truncated_error': 0.0, 'delta': 1.0, 'num_pairs': 0}

    all_pa = []
    all_pb = []
    all_mask = []
    all_range_mean = []

    for face_i, face_j, src_coords, tgt_grid in correspondences:
        N = src_coords.shape[0]
        if N == 0:
            continue

        # 随机采样以控制显存
        if N > max_correspondences:
            indices = torch.randperm(N, device=device)[:max_correspondences]
            src_coords_s = src_coords[indices]
            tgt_grid_s = tgt_grid[indices]
            N_s = max_correspondences
        else:
            src_coords_s = src_coords
            tgt_grid_s = tgt_grid
            N_s = N

        # 面 i 的源点: 直接用整数索引取
        pa = pred_points_world[face_i, src_coords_s[:, 0], src_coords_s[:, 1]]  # (N_s, 3)
        ga = gt_points_world[face_i, src_coords_s[:, 0], src_coords_s[:, 1]]    # (N_s, 3)

        # 面 j 的对应点: 用 grid_sample 在点云图上双线性插值
        # pred: (1, 3, H, W)
        pred_j_map = pred_points_world[face_j].permute(2, 0, 1).unsqueeze(0)
        gt_j_map = gt_points_world[face_j].permute(2, 0, 1).unsqueeze(0)

        grid = tgt_grid_s.unsqueeze(0).unsqueeze(0)  # (1, 1, N_s, 2)
        pb = F.grid_sample(
            pred_j_map, grid, mode='bilinear', padding_mode='border', align_corners=True
        ).squeeze(0).squeeze(1).T  # (N_s, 3)
        gb = F.grid_sample(
            gt_j_map, grid, mode='bilinear', padding_mode='border', align_corners=True
        ).squeeze(0).squeeze(1).T  # (N_s, 3)

        # mask: 两侧 GT 均为 finite，且两侧 pred 均为 finite
        mask_a = torch.isfinite(ga).all(dim=-1)
        mask_b = torch.isfinite(gb).all(dim=-1)
        mask_pa = torch.isfinite(pa).all(dim=-1)
        mask_pb = torch.isfinite(pb).all(dim=-1)
        mask = mask_a & mask_b & mask_pa & mask_pb  # (N_s,)

        # 权重用的 range: 两侧 GT 的平均 range
        ga_safe = torch.where(mask_a[..., None], ga, torch.ones_like(ga))
        gb_safe = torch.where(mask_b[..., None], gb, torch.ones_like(gb))
        range_a = ga_safe.norm(dim=-1).clamp_min(1e-5)
        range_b = gb_safe.norm(dim=-1).clamp_min(1e-5)
        range_mean = 0.5 * (range_a + range_b)

        all_pa.append(pa)
        all_pb.append(pb)
        all_mask.append(mask)
        all_range_mean.append(range_mean)

    if len(all_pa) == 0:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return zero, {'seam_loss': 0.0, 'truncated_error': 0.0, 'delta': 1.0, 'num_pairs': 0}

    # 拼接所有对应点对
    pa = torch.cat(all_pa, dim=0)              # (N_total, 3)
    pb = torch.cat(all_pb, dim=0)              # (N_total, 3)
    mask = torch.cat(all_mask, dim=0)          # (N_total,)
    range_mean = torch.cat(all_range_mean, dim=0)  # (N_total,)

    # 权重: 1/range，用两侧 range 的平均
    weight = mask.float() * 2.0 / (range_mean + range_mean)  # = mask / range_mean

    # clamp 掉极端权重，防止 nan
    if mask.any():
        w_mean = (weight * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        weight = weight.clamp_max(10.0 * w_mean.clamp_min(1e-6))

    # L1 on 3D world points
    diff = (pa - pb).abs()  # (N_total, 3)
    # 将 NaN diff 替换为 0（已被 mask 过滤，不影响有效 loss）
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    per_pair_loss = _smooth(diff * weight[..., None], beta=beta).sum(dim=-1)  # (N_total,)

    num_valid = mask.float().sum().clamp_min(1.0)
    loss = per_pair_loss.sum() / num_valid

    # 指标
    with torch.no_grad():
        err = (pa.detach() - pb.detach()).norm(dim=-1) / range_mean.clamp_min(1e-5)
        err = err.masked_fill(~mask, 0.0)
        trunc_err = (err.clamp_max(1.0) * mask.float()).sum() / num_valid
        delta = ((err < 1.0).float() * mask.float()).sum() / num_valid

    misc = {
        'seam_loss': loss.item(),
        'truncated_error': trunc_err.item(),
        'delta': delta.item(),
        'num_pairs': int(num_valid.item()),
    }
    return loss, misc


def affine_invariant_global_loss_panorama(
    pred_points_world: torch.Tensor,
    gt_points_world: torch.Tensor,
    align_resolution: int = 24,
    beta: float = 0.0,
    trunc: float = 1.0,
):
    """
    全景图专用的 global loss。
    Point Head 直接预测世界坐标系点云，6 个面的点云拼接后做全局 align，
    使用 align_points_scale_xyz_shift 对齐（一个 scale + 一个 3D shift），
    然后在世界坐标系下直接计算 loss，权重用 1/range（到原点的距离）。

    Args:
        pred_points_world: (V, H, W, 3) 世界坐标系下的预测点云，V=6
        gt_points_world: (V, H, W, 3) 世界坐标系下的 GT 点云，V=6
        align_resolution: 每个 face 的低分辨率采样尺寸（默认 24，6面共 3456 点）
        beta: smooth L1 的 beta 参数
        trunc: align 的截断参数
    """
    device = pred_points_world.device
    V, H_face, W_face, _ = pred_points_world.shape

    # ---- Step 1: 在世界坐标系下，batch 化 masked_nearest_resize（V 个面一起处理） ----
    mask_world = torch.isfinite(gt_points_world).all(dim=-1)  # (V, H, W)
    gt_points_world_safe = torch.where(mask_world[..., None], gt_points_world, 1)

    # masked_nearest_resize 支持 batch 维度: (V, H, W, 3) -> (V, R, R, 3)
    pred_lr_world, gt_lr_world, mask_lr_world = utils3d.pt.masked_nearest_resize(
        pred_points_world, gt_points_world_safe, mask=mask_world,
        size=(align_resolution, align_resolution)
    )

    # ---- Step 2: 6 个面拼接后做全局 align_points_scale_xyz_shift ----
    # 用 1/range（到原点的距离）做权重，避免世界坐标系下某个轴接近 0 的问题
    gt_lr_range = gt_lr_world.norm(dim=-1)  # (V, R, R)
    scale, shift = align_points_scale_xyz_shift(
        pred_lr_world.reshape(1, -1, 3),   # (1, V*R*R, 3)
        gt_lr_world.reshape(1, -1, 3),     # (1, V*R*R, 3)
        (mask_lr_world.reshape(1, -1) / gt_lr_range.reshape(1, -1).clamp_min(1e-2)),  # (1, V*R*R)
        trunc=trunc
    )
    # scale: (1,), shift: (1, 3)
    valid = scale > 0
    scale = torch.where(valid, scale, torch.zeros_like(scale))
    shift = torch.where(valid[..., None], shift, torch.zeros_like(shift))

    # ---- Step 3: 应用全局 scale + xyz shift，在世界坐标系下直接计算 loss ----
    # pred_aligned = scale * pred + shift
    pred_aligned = scale[..., None, None, None] * pred_points_world + shift[..., None, None, :]  # (V, H, W, 3)

    # 用 1/range 做权重（世界坐标系下 range = 到原点的距离）
    gt_range = gt_points_world_safe.norm(dim=-1)  # (V, H, W)
    weight = (valid[..., None, None] & mask_world).float() / gt_range.clamp_min(1e-5)  # (V, H, W)
    weight = weight.clamp_max(10.0 * weighted_mean(weight, mask_world, dim=(-2, -1), keepdim=True))

    loss_per_face = _smooth((pred_aligned - gt_points_world_safe).abs() * weight[..., None], beta=beta).mean(dim=(-3, -2, -1))  # (V,)
    loss = loss_per_face.mean()

    err = (pred_aligned.detach() - gt_points_world_safe).norm(dim=-1) / gt_range.clamp_min(1e-5)  # (V, H, W)
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1.0), mask_world).item(),
        'delta': weighted_mean((err < 1).float(), mask_world).item(),
        '_shift': shift.detach(),  # (1, 3) 给 cubemap_seam_loss 复用
    }

    return loss, misc, scale.detach()


def monitoring(points: torch.Tensor):
    return {
        'std': points.std().item(),
    }


def compute_anchor_sampling_weight(
    points: torch.Tensor, 
    mask: torch.Tensor, 
    radius_2d: torch.Tensor, 
    radius_3d: torch.Tensor, 
    num_test: int = 64
) -> torch.Tensor:
    # Importance sampling to balance the sampled probability of fine strutures.
    # NOTE: DepthMaster-1 uses uniform random sampling instead of importance sampling.
    #       This is an incremental trick introduced later than the publication of DepthMaster-1 paper.

    height, width = points.shape[-3:-1]

    pixel_i, pixel_j = torch.meshgrid(
        torch.arange(height, device=points.device), 
        torch.arange(width, device=points.device),
        indexing='ij'
    )
    
    test_delta_i = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
    test_delta_j = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
    test_i, test_j = pixel_i[..., None] + test_delta_i, pixel_j[..., None] + test_delta_j                       # [height, width, num_test]
    test_mask = (test_i >= 0) & (test_i < height) & (test_j >= 0) & (test_j < width)                            # [height, width, num_test]
    test_i, test_j = test_i.clamp(0, height - 1), test_j.clamp(0, width - 1)                                    # [height, width, num_test]
    test_mask = test_mask & mask[..., test_i, test_j]                                                           # [..., height, width, num_test]
    test_points = points[..., test_i, test_j, :]                                                                # [..., height, width, num_test, 3]
    test_dist = (test_points - points[..., None, :]).norm(dim=-1)                                               # [..., height, width, num_test]

    weight = 1 / ((test_dist <= radius_3d[..., None]) & test_mask).float().sum(dim=-1).clamp_min(1)
    weight = torch.where(mask, weight, 0)
    weight = weight / weight.sum(dim=(-2, -1), keepdim=True).add(1e-7)                                          # [..., height, width]
    return weight


def affine_invariant_local_loss(
    pred_points: torch.Tensor, 
    gt_points: torch.Tensor, 
    focal: torch.Tensor, 
    global_scale: torch.Tensor, 
    level: Literal[4, 16, 64], 
    align_resolution: int = 32, 
    num_patches: int = 16, 
    beta: float = 0.0, 
    trunc: float = 1.0, 
    sparsity_aware: bool = False
):
    device, dtype = pred_points.device, pred_points.dtype
    *batch_shape, height, width, _ = pred_points.shape
    batch_size = math.prod(batch_shape)

    gt_mask = torch.isfinite(gt_points).all(dim=-1)
    gt_points = torch.where(gt_mask[..., None], gt_points, 1)
    pred_points, gt_points, gt_mask, focal, global_scale = pred_points.reshape(-1, height, width, 3), gt_points.reshape(-1, height, width, 3), gt_mask.reshape(-1, height, width), focal.reshape(-1), global_scale.reshape(-1) if global_scale is not None else None

    # Sample patch anchor points indices [num_total_patches]
    radius_2d = math.ceil(0.5 / level * (height ** 2 + width ** 2) ** 0.5)
    radius_3d = 0.5 / level / focal * gt_points[..., 2]
    anchor_sampling_weights = compute_anchor_sampling_weight(gt_points, gt_mask, radius_2d, radius_3d, num_test=64)
    where_mask = torch.where(gt_mask)
    random_selection = torch.multinomial(anchor_sampling_weights[where_mask], num_patches * batch_size, replacement=True)
    patch_batch_idx, patch_anchor_i, patch_anchor_j = [indices[random_selection] for indices in where_mask]     # [num_total_patches]

    # Get patch indices [num_total_patches, patch_h, patch_w]
    patch_i, patch_j = torch.meshgrid(
        torch.arange(-radius_2d, radius_2d + 1, device=device), 
        torch.arange(-radius_2d, radius_2d + 1, device=device),
        indexing='ij'
    )
    patch_i, patch_j = patch_i + patch_anchor_i[:, None, None], patch_j + patch_anchor_j[:, None, None]
    patch_mask = (patch_i >= 0) & (patch_i < height) & (patch_j >= 0) & (patch_j < width)
    patch_i, patch_j = patch_i.clamp(0, height - 1), patch_j.clamp(0, width - 1)
    
    # Get patch mask and gt patch points
    gt_patch_anchor_points = gt_points[patch_batch_idx, patch_anchor_i, patch_anchor_j]
    gt_patch_radius_3d = 0.5 / level / focal[patch_batch_idx] * gt_patch_anchor_points[:, 2]
    gt_patch_points = gt_points[patch_batch_idx[:, None, None], patch_i, patch_j]
    gt_patch_dist = (gt_patch_points - gt_patch_anchor_points[:, None, None, :]).norm(dim=-1)    
    patch_mask &= gt_mask[patch_batch_idx[:, None, None], patch_i, patch_j]
    patch_mask &= gt_patch_dist <= gt_patch_radius_3d[:, None, None]

    # Pick only non-empty patches
    MINIMUM_POINTS_PER_PATCH = 32
    nonempty = torch.where(patch_mask.sum(dim=(-2, -1)) >= MINIMUM_POINTS_PER_PATCH)
    num_nonempty_patches = nonempty[0].shape[0]
    if num_nonempty_patches == 0:
        return torch.tensor(0.0, dtype=dtype, device=device), {}
    
    # Finalize all patch variables
    patch_batch_idx, patch_i, patch_j = patch_batch_idx[nonempty], patch_i[nonempty], patch_j[nonempty]
    patch_mask = patch_mask[nonempty]                                   # [num_nonempty_patches, patch_h, patch_w]
    gt_patch_points = gt_patch_points[nonempty]                         # [num_nonempty_patches, patch_h, patch_w, 3]
    gt_patch_radius_3d = gt_patch_radius_3d[nonempty]                   # [num_nonempty_patches]
    gt_patch_anchor_points = gt_patch_anchor_points[nonempty]           # [num_nonempty_patches, 3]
    pred_patch_points = pred_points[patch_batch_idx[:, None, None], patch_i, patch_j]
    
    # Align patch points
    pred_patch_points_lr, gt_patch_points_lr, patch_lr_mask = utils3d.pt.masked_nearest_resize(pred_patch_points, gt_patch_points, mask=patch_mask, size=(align_resolution, align_resolution))
    local_scale, local_shift = align_points_scale_xyz_shift(pred_patch_points_lr.flatten(-3, -2), gt_patch_points_lr.flatten(-3, -2), patch_lr_mask.flatten(-2) / gt_patch_radius_3d[:, None].add(1e-7), trunc=trunc)
    if global_scale is not None:
        scale_differ = local_scale / global_scale[patch_batch_idx]
        patch_valid = (scale_differ > 0.1) & (scale_differ < 10.0) & (global_scale > 0)
    else:
        patch_valid = local_scale > 0
    local_scale, local_shift = torch.where(patch_valid, local_scale, 0), torch.where(patch_valid[:, None], local_shift, 0)
    patch_mask &= patch_valid[:, None, None]

    pred_patch_points = local_scale[:, None, None, None] * pred_patch_points + local_shift[:, None, None, :]                   # [num_patches_nonempty, patch_h, patch_w, 3]
    
    # Compute loss
    gt_mean = harmonic_mean(gt_points[..., 2], gt_mask, dim=(-2, -1))
    patch_weight = patch_mask.float() / gt_patch_points[..., 2].clamp_min(0.1 * gt_mean[patch_batch_idx, None, None])          # [num_patches_nonempty, patch_h, patch_w]
    loss = _smooth((pred_patch_points - gt_patch_points).abs() * patch_weight[..., None], beta=beta).mean(dim=(-3, -2, -1))    # [num_patches_nonempty]
    
    if sparsity_aware:
        # Reweighting improves performance on sparse depth data. NOTE: this is not used in DepthMaster-1.
        sparsity = patch_mask.float().mean(dim=(-2, -1)) / patch_lr_mask.float().mean(dim=(-2, -1))
        loss = loss / (sparsity + 1e-7)
    loss = torch.scatter_reduce(torch.zeros(batch_size, dtype=dtype, device=device), dim=0, index=patch_batch_idx, src=loss, reduce='sum') / num_patches
    loss = loss.reshape(batch_shape)
    
    err = (pred_patch_points.detach() - gt_patch_points).norm(dim=-1) / gt_patch_radius_3d[..., None, None]

    # Record any scalar metric
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1), patch_mask).item(),
        'delta': weighted_mean((err < 1).float(), patch_mask).item()
    }

    return loss, misc


def affine_invariant_segment_loss(
    pred_points: torch.Tensor,            # (H, W, 3)
    gt_points: torch.Tensor,              # (H, W, 3)
    segmentation_mask: torch.Tensor,      # (H, W) long, 0 = invalid/background
    segmentation_labels: Dict[str, int],  # {label_name: seg_id}
    align_resolution: int = 16,
    min_seg_pixels: int = 64,
    beta: float = 0.0,
    trunc: float = 1.0,
):
    """
    Per-instance affine-invariant point loss, mirroring the evaluation-time
    `local_points` protocol (depthmaster/test/metrics.py: 285-312):
      - For each instance mask `seg_id`, take pred/gt points within the mask,
        compute diameter = max(extent_xyz) of GT,
        align pred to gt via (scale, 3D shift) using `align_points_scale_xyz_shift`,
        then accumulate L1 error weighted by 1/diameter.
    Differences from the evaluation:
      - Single-image (no batch dim). Wrapper invokes per instance i.
      - Skips instances with < min_seg_pixels valid pixels.
    Args:
        pred_points: (H, W, 3)
        gt_points: (H, W, 3) with NaN at invalid pixels.
        segmentation_mask: (H, W) int tensor; 0 means no instance.
        segmentation_labels: dict mapping label_name -> seg_id (int).
                             May be None or empty -> returns 0 loss.
    Returns:
        loss: scalar tensor. mean over valid instances; 0 if no valid instance.
        misc: dict with 'num_segments', 'truncated_error', 'delta'.
    """
    device, dtype = pred_points.device, pred_points.dtype
    H, W, _ = pred_points.shape

    misc = {'num_segments': 0, 'truncated_error': 0.0, 'delta': 0.0}
    zero = torch.zeros((), dtype=dtype, device=device)

    if segmentation_labels is None or len(segmentation_labels) == 0:
        return zero, misc
    if segmentation_mask is None:
        return zero, misc

    seg = segmentation_mask.to(device).long()
    if seg.shape != (H, W):
        return zero, misc

    gt_finite_mask = torch.isfinite(gt_points).all(dim=-1)
    # Replace NaN to 1 so downstream ops don't propagate NaN; we use mask separately.
    gt_points_safe = torch.where(gt_finite_mask[..., None], gt_points, torch.ones_like(gt_points))

    seg_ids = list(segmentation_labels.values())

    losses_per_seg = []
    err_running, delta_running, weight_running = 0.0, 0.0, 0.0

    for sid in seg_ids:
        if sid is None or int(sid) <= 0:
            continue
        seg_bool = (seg == int(sid)) & gt_finite_mask
        n = int(seg_bool.sum().item())
        if n < min_seg_pixels:
            continue

        # Diameter from gt bbox extent (max of xyz extents)
        ys, xs = torch.where(seg_bool)
        gt_inst = gt_points_safe[ys, xs]  # (N, 3)
        diameter = (gt_inst.amax(dim=0) - gt_inst.amin(dim=0)).amax().clamp_min(1e-6)

        pred_inst = pred_points[ys, xs]   # (N, 3)

        # Low-resolution sampling for alignment stability (cap at align_resolution^2)
        max_n_lr = align_resolution * align_resolution
        if n > max_n_lr:
            idx = torch.randperm(n, device=device)[:max_n_lr]
            pred_lr, gt_lr = pred_inst[idx], gt_inst[idx]
        else:
            pred_lr, gt_lr = pred_inst, gt_inst

        # align_points_scale_xyz_shift expects [B, N, 3] + per-point weights.
        weight_lr = torch.full((1, pred_lr.shape[0]), 1.0 / diameter.detach(), dtype=dtype, device=device)
        scale, shift = align_points_scale_xyz_shift(
            pred_lr.unsqueeze(0), gt_lr.unsqueeze(0), weight_lr, trunc=trunc
        )
        scale, shift = scale.squeeze(0), shift.squeeze(0)
        # Skip degenerate alignment (negative or zero scale)
        if not torch.isfinite(scale) or scale.item() <= 0:
            continue

        pred_aligned = pred_inst * scale + shift  # (N, 3)
        err = (pred_aligned - gt_inst).abs().sum(dim=-1) / diameter  # (N,)
        loss_seg = _smooth(err, beta=beta).mean()
        losses_per_seg.append(loss_seg)

        # Stats (detached)
        with torch.no_grad():
            d_err = err.detach()
            err_running += float(d_err.clamp_max(1.0).mean().item()) * n
            delta_running += float((d_err < 1.0).float().mean().item()) * n
            weight_running += n

    if not losses_per_seg:
        return zero, misc

    loss = torch.stack(losses_per_seg).mean()
    misc['num_segments'] = len(losses_per_seg)
    if weight_running > 0:
        misc['truncated_error'] = err_running / weight_running
        misc['delta'] = delta_running / weight_running
    return loss, misc


def normal_loss(points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    device, dtype = points.device, points.dtype
    height, width = points.shape[-3:-1]

    mask = torch.isfinite(gt_points).all(dim=-1)
    gt_points = torch.where(mask[..., None], gt_points, 1)

    leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
    upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
    leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
    downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
    rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

    gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
    gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
    gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
    gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
    gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

    mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
    mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
    mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
    mask_downxright = mask_leftdown & mask_rightup & mask_leftup
    mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

    MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

    loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

    loss = loss.mean() / (4 * max(points.shape[-3:-1]))

    return loss, {}


def edge_loss(points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    device, dtype = points.device, points.dtype
    height, width = points.shape[-3:-1]

    mask = torch.isfinite(gt_points).all(dim=-1)
    gt_points = torch.where(mask[..., None], gt_points, 1)

    dx = points[..., :-1, :, :] - points[..., 1:, :, :]
    dy = points[..., :, :-1, :] - points[..., :, 1:, :]
    
    gt_dx = gt_points[..., :-1, :, :] - gt_points[..., 1:, :, :]
    gt_dy = gt_points[..., :, :-1, :] - gt_points[..., :, 1:, :]

    mask_dx = mask[..., :-1, :] & mask[..., 1:, :]
    mask_dy = mask[..., :, :-1] & mask[..., :, 1:]

    MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(0.1), math.radians(90), math.radians(3)

    loss_dx = mask_dx * _smooth(angle_diff_vec3(dx, gt_dx).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)
    loss_dy = mask_dy * _smooth(angle_diff_vec3(dy, gt_dy).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)
    loss = (loss_dx.mean(dim=(-2, -1)) + loss_dy.mean(dim=(-2, -1))) / (2 * max(points.shape[-3:-1]))

    return loss, {}


def mask_l2_loss(pred_mask: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
    loss = gt_mask_neg.float() * pred_mask.square() + gt_mask_pos.float() * (1 - pred_mask).square()
    loss = loss.mean(dim=(-2, -1))
    return loss, {}


def mask_bce_loss(pred_mask_prob: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast(device_type='cuda', enabled=False):
        pred_mask_prob = pred_mask_prob.float()
        loss = (gt_mask_pos | gt_mask_neg) * F.binary_cross_entropy(pred_mask_prob, gt_mask_pos.float(), reduction='none')
    loss = loss.mean(dim=(-2, -1))
    return loss, {}


def metric_scale_loss(scale_pred: torch.Tensor, scale_gt: torch.Tensor):
    scale_pred = scale_pred.squeeze()
    scale_gt = scale_gt.squeeze()
    valid = scale_gt > 0
    return torch.where(valid, F.mse_loss(scale_pred.log(), torch.where(valid, scale_gt.log(), 0), reduction='none'), 0), {}


def normal_map_loss(pred_normal: torch.Tensor, gt_normal: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(gt_normal).all(dim=-1)
    gt_normal = torch.where(mask[..., None], gt_normal, 1)

    loss = (mask * utils3d.pt.angle_between(pred_normal, gt_normal).square()).mean(dim=(-2, -1))
    return loss, {}


def depth_affine_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    align_resolution: int = 48,
    beta: float = 0.0,
    trunc: float = 1.0,
):
    """
    Affine-invariant depth loss: aligns predicted depth to GT depth via optimal
    scale and shift (in 1/depth space), then computes weighted L1 loss.

    This mirrors the affine_invariant_global_loss logic but operates on scalar
    depth maps instead of 3D point maps.

    Args:
        pred_depth: (H, W) predicted depth map (positive values)
        gt_depth: (H, W) ground-truth depth map (positive values, inf/nan for invalid)
        align_resolution: resolution for computing alignment parameters
        beta: smooth L1 beta (0 = standard L1)
        trunc: truncation parameter for alignment

    Returns:
        loss: scalar loss
        misc: dict with monitoring metrics
    """
    device = pred_depth.device

    # Valid mask: finite and positive
    mask = torch.isfinite(gt_depth) & (gt_depth > 0)
    gt_safe = torch.where(mask, gt_depth, torch.ones_like(gt_depth))

    # Downsample for alignment
    pred_lr, gt_lr, mask_lr = utils3d.pt.masked_nearest_resize(
        pred_depth, gt_safe, mask=mask, size=(align_resolution, align_resolution)
    )

    # Align using scale and shift: pred_aligned = scale * pred + shift
    # Weight by 1/depth (consistent with point cloud loss)
    weight_lr = mask_lr.float() / gt_lr.clamp_min(1e-2)
    scale, shift = align_depth_affine(
        pred_lr.flatten(-2).unsqueeze(0),
        gt_lr.flatten(-2).unsqueeze(0),
        weight_lr.flatten(-2).unsqueeze(0),
        trunc=trunc,
    )
    valid = scale > 0
    scale = torch.where(valid, scale, torch.zeros_like(scale))
    shift = torch.where(valid, shift, torch.zeros_like(shift))

    # Apply alignment
    pred_aligned = scale.squeeze() * pred_depth + shift.squeeze()

    # Compute loss with 1/depth weighting
    weight = (valid.squeeze() & mask).float() / gt_safe.clamp_min(1e-5)
    weight = weight.clamp_max(10.0 * weighted_mean(weight, mask, dim=(-2, -1), keepdim=True))
    loss = _smooth((pred_aligned - gt_safe).abs() * weight, beta=beta).mean(dim=(-2, -1))

    # Monitoring metrics
    err = (pred_aligned.detach() - gt_safe).abs() / gt_safe.clamp_min(1e-5)
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1.0), mask).item(),
        'delta': weighted_mean((err < 0.25).float(), mask).item(),
    }

    return loss, misc


def disparity_affine_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    align_resolution: int = 48,
    beta: float = 0.0,
):
    """
    Affine-invariant disparity loss, mirroring the evaluation-time
    `disparity_affine_invariant` protocol (depthmaster/test/metrics.py:192-215):
      - Convert pred/gt depth to disparity (1/depth) on valid pixels.
      - Solve (scale, shift) in disparity space via least squares with weight 1.
      - Apply scale, shift to pred_disparity, clamp by min gt disparity to avoid
        extreme outliers near 0, then convert back to depth.
      - Compute weighted L1 in depth space (rel-style: |Dpred - Dgt| / Dgt).
    Args:
        pred_depth: (H, W) predicted depth map (positive values).
        gt_depth: (H, W) ground-truth depth (positive; inf/nan for invalid).
        align_resolution: low-res grid for solving (scale, shift), keeps cost low.
        beta: smooth L1 beta (0 = standard L1).
    Returns:
        loss: scalar tensor.
        misc: dict with monitoring metrics ('truncated_error', 'delta1').
    """
    device = pred_depth.device

    # Valid mask: finite + positive
    mask = torch.isfinite(gt_depth) & (gt_depth > 0)
    gt_safe = torch.where(mask, gt_depth, torch.ones_like(gt_depth))
    pred_safe = pred_depth.clamp_min(1e-6)  # avoid /0 in disparity

    # Low-resolution downsample for solving (scale, shift)
    pred_lr, gt_lr, mask_lr = utils3d.pt.masked_nearest_resize(
        pred_safe, gt_safe, mask=mask, size=(align_resolution, align_resolution)
    )
    pred_disp_lr = 1.0 / pred_lr.clamp_min(1e-6)
    gt_disp_lr = 1.0 / gt_lr.clamp_min(1e-6)

    # Solve in disparity space via lstsq (与评测端 align_affine_lstsq 完全一致)
    weight_lr = mask_lr.float()
    scale, shift = align_affine_lstsq(
        pred_disp_lr.flatten(-2).unsqueeze(0),
        gt_disp_lr.flatten(-2).unsqueeze(0),
        weight_lr.flatten(-2).unsqueeze(0),
    )
    scale, shift = scale.squeeze(), shift.squeeze()
    if not torch.isfinite(scale) or not torch.isfinite(shift):
        return torch.zeros((), device=device, dtype=pred_depth.dtype), {
            'truncated_error': 0.0, 'delta1': 0.0,
        }

    # Apply alignment in disparity space, then convert back to depth.
    pred_disp = 1.0 / pred_safe
    pred_disp_aligned = pred_disp * scale + shift
    # 评测端 clamp: clamp_min by 1 / gt_depth.max() (即最远点 disparity 下界)
    gt_max_depth = gt_safe[mask].max() if mask.any() else gt_safe.max()
    min_disp = 1.0 / gt_max_depth.clamp_min(1e-6)
    pred_disp_aligned = pred_disp_aligned.clamp_min(min_disp)
    pred_depth_aligned = 1.0 / pred_disp_aligned

    # Compute rel-style L1: |Dpred - Dgt| / Dgt, weighted by mask.
    err = (pred_depth_aligned - gt_safe).abs() / gt_safe.clamp_min(1e-5)
    loss = _smooth(err * mask.float(), beta=beta)
    # mean over valid pixels
    n_valid = mask.float().sum().clamp_min(1.0)
    loss = loss.sum() / n_valid

    # Stats (detached)
    with torch.no_grad():
        d_err = err.detach()
        delta1 = ((d_err < 0.25).float() * mask.float()).sum() / n_valid
        misc = {
            'truncated_error': float(((d_err.clamp_max(1.0)) * mask.float()).sum() / n_valid),
            'delta1': float(delta1),
        }

    return loss, misc


def erp_depth_affine_loss(
    pred_erp_depth: torch.Tensor,
    gt_erp_depth: torch.Tensor,
    align_resolution: int = 64,
    beta: float = 0.0,
    trunc: float = 1.0,
):
    """
    在 ERP 全景深度图上计算仿射变换无关的 loss。
    使用 align_depth_affine 对齐预测深度和 GT 深度（消除 scale 和 shift），
    然后计算 L1 loss。

    Args:
        pred_erp_depth: (H, W) 预测的 ERP range depth
        gt_erp_depth: (H, W) GT 的 ERP range depth
        align_resolution: 用于 align 的低分辨率尺寸
        beta: smooth L1 的 beta 参数
        trunc: align 的截断参数

    Returns:
        loss: 标量 loss
        misc: 字典，包含 truncated_error 和 delta 等指标
    """
    device = pred_erp_depth.device
    H, W = pred_erp_depth.shape

    # mask: 有效区域（GT 为有限正数）
    mask = torch.isfinite(gt_erp_depth) & (gt_erp_depth > 0)  # (H, W)
    gt_safe = torch.where(mask, gt_erp_depth, torch.ones_like(gt_erp_depth))

    # 使用 masked_nearest_resize 下采样（和点云 align 一致的做法）
    # align_resolution 对应 ERP 高度，宽度按 2:1 比例
    align_h, align_w = align_resolution, align_resolution * 2
    pred_lr, gt_lr, mask_lr = utils3d.pt.masked_nearest_resize(
        pred_erp_depth, gt_safe, mask=mask,
        size=(align_h, align_w)
    )  # (align_h, align_w), (align_h, align_w), (align_h, align_w)

    # 用 align_depth_affine 求解最优 scale 和 shift
    # weight: 用 1/depth 做权重（和点云 loss 中用 1/z 做权重一致）
    weight_lr = mask_lr.float() / gt_lr.clamp_min(1e-2)  # (align_h, align_w)
    scale, shift = align_depth_affine(
        pred_lr.flatten(-2).unsqueeze(0),   # (1, align_h * align_w)
        gt_lr.flatten(-2).unsqueeze(0),     # (1, align_h * align_w)
        weight_lr.flatten(-2).unsqueeze(0), # (1, align_h * align_w)
        trunc=trunc
    )
    # scale: (1,), shift: (1,)
    valid = scale > 0
    scale = torch.where(valid, scale, torch.zeros_like(scale))
    shift = torch.where(valid, shift, torch.zeros_like(shift))

    # 对齐预测深度
    pred_aligned = scale.squeeze() * pred_erp_depth + shift.squeeze()  # (H, W)

    # 计算 loss
    weight = (valid.squeeze() & mask).float() / gt_safe.clamp_min(1e-5)  # (H, W)
    weight = weight.clamp_max(10.0 * weighted_mean(weight, mask, dim=(-2, -1), keepdim=True))
    loss = _smooth((pred_aligned - gt_safe).abs() * weight, beta=beta).mean(dim=(-2, -1))

    # 计算指标
    err = (pred_aligned.detach() - gt_safe).abs() / gt_safe.clamp_min(1e-5)  # (H, W)
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1.0), mask).item(),
        'delta': weighted_mean((err < 1).float(), mask).item(),
    }

    return loss, misc


def _compute_cubemap_ray_len(face_h: int, face_w: int, fov_deg: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Compute per-pixel ray length for cubemap faces (z-buffer → range depth conversion).

    For a pinhole camera with given FOV, each pixel's ray direction is (x_cam, y_cam, 1),
    and ray_len = ||(x_cam, y_cam, 1)||. range_depth = z_buffer_depth * ray_len.

    Args:
        face_h: cubemap face height
        face_w: cubemap face width
        fov_deg: cubemap face FOV in degrees
        device: torch device
        dtype: torch dtype

    Returns:
        ray_len: (face_h, face_w) per-pixel ray length
    """
    f = 0.5 / math.tan(math.radians(fov_deg) / 2)  # normalized focal length
    cx, cy = 0.5, 0.5

    u = torch.linspace(0, 1, face_w, device=device, dtype=dtype)
    v = torch.linspace(0, 1, face_h, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing='ij')

    x_cam = (uu - cx) / f
    y_cam = (vv - cy) / f
    z_cam = torch.ones_like(x_cam)

    ray_len = torch.sqrt(x_cam ** 2 + y_cam ** 2 + z_cam ** 2)  # (face_h, face_w)
    return ray_len


def panorama_depth_affine_loss(
    pred_depth_per_face: torch.Tensor,
    gt_depth_per_face: torch.Tensor,
    gt_erp_depth: torch.Tensor,
    align_resolution: int = 48,
    beta: float = 0.0,
    trunc: float = 1.0,
    fov_deg: float = 95.0,
    erp_loss_weight: float = 1.0,
    face_loss_weight: float = 1.0,
):
    """
    Panorama depth loss: align in ERP space, compute loss on both ERP and cubemap faces.

    Important: cubemap depth is z-buffer depth, ERP depth is range depth.
    The function converts z-buffer → range depth before ERP projection.

    Strategy:
    1. Convert pred/gt cubemap z-buffer depth → range depth (using ray_len)
    2. Project pred range depth → ERP range depth (using cubemap_to_equirect_torch)
    3. Align pred ERP depth to GT ERP depth via align_depth_affine → get global scale & shift
    4. Apply scale & shift and compute loss on:
       a. ERP space (range depth)
       b. Each cubemap face (range depth)

    Args:
        pred_depth_per_face: (6, H, W) predicted z-buffer depth per cubemap face
        gt_depth_per_face: (6, H, W) GT z-buffer depth per cubemap face
        gt_erp_depth: (any_h, any_w) GT ERP range depth (will be resized to match)
        align_resolution: resolution for alignment (ERP height)
        beta: smooth L1 beta
        trunc: truncation for alignment
        fov_deg: cubemap face FOV in degrees (default 95.0)
        erp_loss_weight: weight for ERP loss component (default 1.0)
        face_loss_weight: weight for cubemap face loss component (default 1.0)

    Returns:
        loss: scalar loss (weighted sum of ERP loss and face loss)
        misc: dict with monitoring metrics
    """
    from depthmaster.train.cubemap_to_equirect import cubemap_to_equirect_torch

    device = pred_depth_per_face.device
    dtype = pred_depth_per_face.dtype

    face_h, face_w = pred_depth_per_face.shape[1], pred_depth_per_face.shape[2]
    erp_h = face_h
    erp_w = face_w * 2

    # Step 0: Compute ray_len and convert z-buffer depth → range depth
    ray_len = _compute_cubemap_ray_len(face_h, face_w, fov_deg, device, dtype)  # (H, W)

    # z-buffer → range depth: range = z * ray_len
    pred_range_per_face = pred_depth_per_face * ray_len.unsqueeze(0)  # (6, H, W)
    gt_range_per_face = gt_depth_per_face * ray_len.unsqueeze(0)      # (6, H, W)

    # Step 1: Project pred range depth → ERP range depth
    # cubemap_to_equirect_torch expects (B, 6, H, W), returns (B, pano_h, pano_w)
    pred_erp_depth = cubemap_to_equirect_torch(
        pred_range_per_face.unsqueeze(0),  # (1, 6, H, W)
        pano_h=erp_h, pano_w=erp_w,
        fov_deg=fov_deg,
    ).squeeze(0)  # (erp_h, erp_w)

    # Step 2: Resize GT ERP depth to match pred ERP resolution
    if gt_erp_depth.shape[0] != erp_h or gt_erp_depth.shape[1] != erp_w:
        # Use nearest interpolation to avoid mixing depth values at boundaries
        gt_for_resize = gt_erp_depth.clone()
        gt_inf_mask = torch.isinf(gt_for_resize)
        gt_nan_mask = torch.isnan(gt_for_resize)
        gt_for_resize[gt_inf_mask] = 1e6
        gt_for_resize[gt_nan_mask] = -1.0
        gt_for_resize = F.interpolate(
            gt_for_resize.unsqueeze(0).unsqueeze(0),
            size=(erp_h, erp_w), mode='nearest',
        ).squeeze(0).squeeze(0)
        # Restore inf/nan
        gt_erp_depth = gt_for_resize
        gt_erp_depth[gt_for_resize >= 1e6 * 0.9] = float('inf')
        gt_erp_depth[gt_for_resize < 0] = float('nan')

    # Valid mask in ERP space
    mask_erp = torch.isfinite(gt_erp_depth) & (gt_erp_depth > 0) & (pred_erp_depth > 0)
    gt_erp_safe = torch.where(mask_erp, gt_erp_depth, torch.ones_like(gt_erp_depth))
    pred_erp_safe = torch.where(mask_erp, pred_erp_depth, torch.zeros_like(pred_erp_depth))

    # Step 3: Affine alignment in ERP space (low-res for efficiency)
    align_h, align_w = align_resolution, align_resolution * 2
    pred_lr, gt_lr, mask_lr = utils3d.pt.masked_nearest_resize(
        pred_erp_safe, gt_erp_safe, mask=mask_erp,
        size=(align_h, align_w)
    )

    # Weight by 1/depth (consistent with point cloud loss)
    weight_lr = mask_lr.float() / gt_lr.clamp_min(1e-2)
    scale, shift = align_depth_affine(
        pred_lr.flatten(-2).unsqueeze(0),
        gt_lr.flatten(-2).unsqueeze(0),
        weight_lr.flatten(-2).unsqueeze(0),
        trunc=trunc,
    )
    # scale: (1,), shift: (1,)
    valid = scale > 0
    scale = torch.where(valid, scale, torch.zeros_like(scale))
    shift = torch.where(valid, shift, torch.zeros_like(shift))

    scale_val = scale.squeeze()
    shift_val = shift.squeeze()
    valid_val = valid.squeeze()

    # Step 4a: ERP loss (range depth space)
    pred_erp_aligned = scale_val * pred_erp_depth + shift_val
    erp_weight = (valid_val & mask_erp).float() / gt_erp_safe.clamp_min(1e-5)
    erp_weight = erp_weight.clamp_max(10.0 * weighted_mean(erp_weight, mask_erp, dim=(-2, -1), keepdim=True))
    erp_loss = _smooth((pred_erp_aligned - gt_erp_safe).abs() * erp_weight, beta=beta).mean()

    # Step 4b: Cubemap face loss (range depth space, with same scale & shift from ERP alignment)
    face_losses = []
    for fi in range(6):
        pred_face = pred_range_per_face[fi]  # (H, W) range depth
        gt_face = gt_range_per_face[fi]      # (H, W) range depth

        # Valid mask for this face
        face_mask = torch.isfinite(gt_face) & (gt_face > 0)
        gt_face_safe = torch.where(face_mask, gt_face, torch.ones_like(gt_face))

        # Align prediction (same scale & shift from ERP alignment)
        pred_aligned = scale_val * pred_face + shift_val

        # Compute loss with 1/depth weighting
        weight = (valid_val & face_mask).float() / gt_face_safe.clamp_min(1e-5)
        weight = weight.clamp_max(10.0 * weighted_mean(weight, face_mask, dim=(-2, -1), keepdim=True))
        face_loss = _smooth((pred_aligned - gt_face_safe).abs() * weight, beta=beta).mean(dim=(-2, -1))
        face_losses.append(face_loss)

    cubemap_face_loss = torch.stack(face_losses).mean()

    # Combined loss
    loss = erp_loss_weight * erp_loss + face_loss_weight * cubemap_face_loss

    # Monitoring metrics (on ERP range depth)
    err = (pred_erp_aligned.detach() - gt_erp_safe).abs() / gt_erp_safe.clamp_min(1e-5)
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1.0), mask_erp).item(),
        'delta': weighted_mean((err < 0.25).float(), mask_erp).item(),
        'erp_loss': erp_loss.item(),
        'face_loss': cubemap_face_loss.item(),
    }

    return loss, misc


def panorama_depth_affine_loss_hard(
    pred_depth_per_face: torch.Tensor,
    gt_depth_per_face: torch.Tensor,
    gt_erp_depth: torch.Tensor,
    align_resolution: int = 48,
    beta: float = 0.0,
    trunc: float = 1.0,
    fov_deg: float = 95.0,
    erp_loss_weight: float = 1.0,
    face_loss_weight: float = 1.0,
):
    """[实验 A] panorama_depth_affine_loss 的硬分配版本。

    与 panorama_depth_affine_loss 的唯一区别：
        cubemap → ERP 投影改用 blend=False（每个 ERP 像素只来自权重最大的那个面，
        不做面间软混合），避免 blending 在 overlap 区域允许两面互相补偿的问题。

    其他逻辑（z-buffer → range、affine alignment、ERP loss + face_loss）完全保留。
    """
    from depthmaster.train.cubemap_to_equirect import cubemap_to_equirect_torch

    device = pred_depth_per_face.device
    dtype = pred_depth_per_face.dtype

    face_h, face_w = pred_depth_per_face.shape[1], pred_depth_per_face.shape[2]
    erp_h = face_h
    erp_w = face_w * 2

    # Step 0: z-buffer → range depth
    ray_len = _compute_cubemap_ray_len(face_h, face_w, fov_deg, device, dtype)  # (H, W)
    pred_range_per_face = pred_depth_per_face * ray_len.unsqueeze(0)  # (6, H, W)
    gt_range_per_face = gt_depth_per_face * ray_len.unsqueeze(0)      # (6, H, W)

    # Step 1: Project pred range → ERP range（**硬分配**，无 blending）
    pred_erp_depth = cubemap_to_equirect_torch(
        pred_range_per_face.unsqueeze(0),
        pano_h=erp_h, pano_w=erp_w,
        fov_deg=fov_deg,
        blend=False,                  # ← 关键改动
    ).squeeze(0)  # (erp_h, erp_w)

    # Step 2: Resize GT ERP depth to match pred ERP resolution
    if gt_erp_depth.shape[0] != erp_h or gt_erp_depth.shape[1] != erp_w:
        gt_for_resize = gt_erp_depth.clone()
        gt_inf_mask = torch.isinf(gt_for_resize)
        gt_nan_mask = torch.isnan(gt_for_resize)
        gt_for_resize[gt_inf_mask] = 1e6
        gt_for_resize[gt_nan_mask] = -1.0
        gt_for_resize = F.interpolate(
            gt_for_resize.unsqueeze(0).unsqueeze(0),
            size=(erp_h, erp_w), mode='nearest',
        ).squeeze(0).squeeze(0)
        gt_erp_depth = gt_for_resize
        gt_erp_depth[gt_for_resize >= 1e6 * 0.9] = float('inf')
        gt_erp_depth[gt_for_resize < 0] = float('nan')

    # Valid mask in ERP space
    mask_erp = torch.isfinite(gt_erp_depth) & (gt_erp_depth > 0) & (pred_erp_depth > 0)
    gt_erp_safe = torch.where(mask_erp, gt_erp_depth, torch.ones_like(gt_erp_depth))
    pred_erp_safe = torch.where(mask_erp, pred_erp_depth, torch.zeros_like(pred_erp_depth))

    # Step 3: Affine alignment in ERP space
    align_h, align_w = align_resolution, align_resolution * 2
    pred_lr, gt_lr, mask_lr = utils3d.pt.masked_nearest_resize(
        pred_erp_safe, gt_erp_safe, mask=mask_erp,
        size=(align_h, align_w),
    )
    weight_lr = mask_lr.float() / gt_lr.clamp_min(1e-2)
    scale, shift = align_depth_affine(
        pred_lr.flatten(-2).unsqueeze(0),
        gt_lr.flatten(-2).unsqueeze(0),
        weight_lr.flatten(-2).unsqueeze(0),
        trunc=trunc,
    )
    valid = scale > 0
    scale = torch.where(valid, scale, torch.zeros_like(scale))
    shift = torch.where(valid, shift, torch.zeros_like(shift))
    scale_val = scale.squeeze()
    shift_val = shift.squeeze()
    valid_val = valid.squeeze()

    # Step 4a: ERP loss
    pred_erp_aligned = scale_val * pred_erp_depth + shift_val
    erp_weight = (valid_val & mask_erp).float() / gt_erp_safe.clamp_min(1e-5)
    erp_weight = erp_weight.clamp_max(10.0 * weighted_mean(erp_weight, mask_erp, dim=(-2, -1), keepdim=True))
    erp_loss = _smooth((pred_erp_aligned - gt_erp_safe).abs() * erp_weight, beta=beta).mean()

    # Step 4b: Cubemap face loss (same scale & shift)
    face_losses = []
    for fi in range(6):
        pred_face = pred_range_per_face[fi]
        gt_face = gt_range_per_face[fi]
        face_mask = torch.isfinite(gt_face) & (gt_face > 0)
        gt_face_safe = torch.where(face_mask, gt_face, torch.ones_like(gt_face))
        pred_aligned = scale_val * pred_face + shift_val
        weight = (valid_val & face_mask).float() / gt_face_safe.clamp_min(1e-5)
        weight = weight.clamp_max(10.0 * weighted_mean(weight, face_mask, dim=(-2, -1), keepdim=True))
        face_loss = _smooth((pred_aligned - gt_face_safe).abs() * weight, beta=beta).mean(dim=(-2, -1))
        face_losses.append(face_loss)
    cubemap_face_loss = torch.stack(face_losses).mean()

    loss = erp_loss_weight * erp_loss + face_loss_weight * cubemap_face_loss

    err = (pred_erp_aligned.detach() - gt_erp_safe).abs() / gt_erp_safe.clamp_min(1e-5)
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1.0), mask_erp).item(),
        'delta': weighted_mean((err < 0.25).float(), mask_erp).item(),
        'erp_loss': erp_loss.item(),
        'face_loss': cubemap_face_loss.item(),
    }

    return loss, misc


def depth_to_points_global_loss(
    pred_depth_zbuf: torch.Tensor,         # (V=6, H, W)
    intrinsics: torch.Tensor,              # (V=6, 3, 3)
    W2C: torch.Tensor,                     # (V=6, 4, 4)
    gt_points_world: torch.Tensor,         # (V=6, H, W, 3)
    align_resolution: int = 24,
    beta: float = 0.0,
    trunc: float = 1.0,
):
    """[实验 B] 把 depth_head 的 z-buffer 反投影到世界点云，再用与 point_head 相同的
    affine_invariant_global_loss_panorama 进行监督。

    与 panorama_depth_affine_loss 的关键区别：
    - 不走 cubemap → ERP 投影路径（避免 blending / 极点稀疏 / 非线性失真）
    - 不在标量 depth 空间做 1D affine alignment
    - 而是先把 zbuf 反投影到世界点云，再在 3D world points 空间做与 points_head 完全
      相同的 affine 对齐 (1 标量 scale + 1 个 3D shift, 4 自由度)

    这样 depth_head 与 points_head 受到结构对称的监督，几何空间一致，且天然鼓励
    overlap 区域反投影一致 (因为 6 个面的世界点云一起拼接做 affine align)。

    Args:
        pred_depth_zbuf: (6, H, W) 预测 z-buffer
        intrinsics:      (6, 3, 3) 相机内参（每个面的 K，与训练数据一致）
        W2C:             (6, 4, 4) world-to-camera 变换
        gt_points_world: (6, H, W, 3) GT 世界点云
        align_resolution: 每个 face 的低分辨率采样尺寸
        beta:  smooth L1 beta
        trunc: align 截断参数

    Returns:
        loss, misc, scale (与 affine_invariant_global_loss_panorama 一致)
    """
    V, H, W = pred_depth_zbuf.shape
    device = pred_depth_zbuf.device
    dtype = pred_depth_zbuf.dtype
    assert V == 6, f"depth_to_points_global_loss 期望 6 个面, got {V}"

    # Step 1: 归一化像素中心 (u, v) ∈ [0, 1]
    # NOTE: cubemap_intrinsics 是归一化坐标系下的内参 (fx ≈ 0.5/tan(fov/2),
    # cx=cy=0.5)，详见 e2c_gpu.py::_build_cubemap_K_W2C。这里必须用归一化像素坐标
    # 才能和 K 的单位一致；如果使用绝对像素坐标 (arange(W)+0.5)，反投影出的 X/Y
    # 会被错误放大约 W 倍，导致 affine alignment 完全失败。
    u = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W  # (W,) ∈ (0, 1)
    v = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H  # (H,) ∈ (0, 1)
    vv, uu = torch.meshgrid(v, u, indexing='ij')                  # (H, W)

    # Step 2: per face 内参解出射线方向系数（归一化 K）
    fx = intrinsics[:, 0, 0].view(V, 1, 1)  # (V, 1, 1)
    fy = intrinsics[:, 1, 1].view(V, 1, 1)
    cx = intrinsics[:, 0, 2].view(V, 1, 1)
    cy = intrinsics[:, 1, 2].view(V, 1, 1)
    # cam-space (x, y, z) = ((u-cx)/fx * z, (v-cy)/fy * z, z),  z = pred_depth_zbuf
    z_cam = pred_depth_zbuf  # (V, H, W)
    x_cam = (uu.unsqueeze(0) - cx) / fx.clamp_min(1e-6) * z_cam  # (V, H, W)
    y_cam = (vv.unsqueeze(0) - cy) / fy.clamp_min(1e-6) * z_cam  # (V, H, W)
    p_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)           # (V, H, W, 3)

    # Step 3: cam → world: p_world = R^T (p_cam - t),  W2C = [R | t]
    R = W2C[:, :3, :3]                            # (V, 3, 3)
    t = W2C[:, :3, 3]                             # (V, 3)
    R_T = R.transpose(-1, -2)                     # (V, 3, 3)
    p_minus_t = p_cam - t.view(V, 1, 1, 3)        # (V, H, W, 3)
    pred_points_world = torch.einsum('vij,vhwj->vhwi', R_T, p_minus_t)  # (V, H, W, 3)

    # Step 4: 复用 point_head 的 panorama global loss
    loss, misc, scale = affine_invariant_global_loss_panorama(
        pred_points_world, gt_points_world,
        align_resolution=align_resolution,
        beta=beta,
        trunc=trunc,
    )
    misc = dict(misc) if isinstance(misc, dict) else {}
    misc['_loss_source'] = 'depth_to_points'

    return loss, misc, scale


# ============================================================================
# Correspondence Consistency Loss (CCL)
# ============================================================================

import numpy as np

# 模块级缓存：以 (fov_deg, face_size) 为 key 缓存对应关系
_ccl_correspondence_cache: Dict[tuple, list] = {}

# 12 对相邻面索引：
# 水平相邻 4 对 + 与 Up 相邻 4 对 + 与 Down 相邻 4 对
# 面索引：0=Front, 1=Right, 2=Back, 3=Left, 4=Up, 5=Down
CUBEMAP_ADJACENT_PAIRS = [
    # 水平相邻
    (0, 1),  # Front - Right
    (1, 2),  # Right - Back
    (2, 3),  # Back - Left
    (3, 0),  # Left - Front
    # 与 Up 相邻
    (0, 4),  # Front - Up
    (1, 4),  # Right - Up
    (2, 4),  # Back - Up
    (3, 4),  # Left - Up
    # 与 Down 相邻
    (0, 5),  # Front - Down
    (1, 5),  # Right - Down
    (2, 5),  # Back - Down
    (3, 5),  # Left - Down
]


def _get_cubemap_rotations() -> List[np.ndarray]:
    """获取 cubemap 6 个面的 W2C 旋转矩阵（OpenCV y-down 世界坐标系）。
    
    严格对齐 dataset_readers.py::_xyzcube_fov95 的 OpenCV 坐标约定：
      - 面顺序: 0=Front(z+) 1=Right(x+) 2=Back(z-) 3=Left(x-) 4=Up(天空,y-) 5=Down(地面,y+)
      - 相机坐标系: x-right, y-down, z-forward (OpenCV 标准)
      - 世界坐标系: x-right, y-down, z-forward (OpenCV 标准)
      - R @ world_point = camera_point
      - Front 面 W2C = I（单位矩阵）
    
    Returns:
        rotations: 6 个 3x3 W2C 旋转矩阵的列表
    """
    return [
        # Front: 看向 z+
        np.array([[ 1,  0,  0],
                  [ 0,  1,  0],
                  [ 0,  0,  1]], dtype=np.float64),
        # Right: 看向 x+
        np.array([[ 0,  0, -1],
                  [ 0,  1,  0],
                  [ 1,  0,  0]], dtype=np.float64),
        # Back: 看向 z-
        np.array([[-1,  0,  0],
                  [ 0,  1,  0],
                  [ 0,  0, -1]], dtype=np.float64),
        # Left: 看向 x-
        np.array([[ 0,  0,  1],
                  [ 0,  1,  0],
                  [-1,  0,  0]], dtype=np.float64),
        # Up: 看向 y-（天空，y-down 中 y 负方向）— 绕 x 轴旋转 -90°
        np.array([[ 1,  0,  0],
                  [ 0,  0,  1],
                  [ 0, -1,  0]], dtype=np.float64),
        # Down: 看向 y+（地面，y-down 中 y 正方向）— 绕 x 轴旋转 +90°
        np.array([[ 1,  0,  0],
                  [ 0,  0, -1],
                  [ 0,  1,  0]], dtype=np.float64),
    ]


def _build_intrinsic_matrix(fov_deg: float, face_size: int) -> np.ndarray:
    """基于 FOV 和面尺寸构建内参矩阵 K（align_corners=True 约定）。
    
    与 _xyzcube_fov95 的坐标约定一致：
      rng = linspace(-0.5, 0.5, face_size)  =>  像素 c 对应 rng[c] = c/(face_size-1) - 0.5
      世界坐标 = rng[c] * tan(fov/2)
    
    对应的 pinhole 内参（align_corners=True）：
      f = (face_size - 1) / (2 * tan(fov/2))
      cx = cy = (face_size - 1) / 2
    
    Args:
        fov_deg: 视场角（度）
        face_size: 面的像素尺寸
    
    Returns:
        K: 3x3 内参矩阵
    """
    f = (face_size - 1) / (2.0 * np.tan(0.5 * np.radians(fov_deg)))
    cx = (face_size - 1) / 2.0
    cy = (face_size - 1) / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float64)
    return K


def _build_correspondences(fov_deg: float, face_size: int, device: torch.device) -> list:
    """为 12 对相邻面建立密集对应关系。
    
    对每对相邻面 (i, j)，通过单应性矩阵 H_ij = K @ R_j^{-1} @ R_i @ K^{-1}
    将面 i 的所有像素坐标投影到面 j，筛选落在有效范围内的对应点。
    
    Args:
        fov_deg: 视场角（度）
        face_size: 特征图尺寸
        device: 目标设备
    
    Returns:
        correspondences: 列表，每个元素为 (face_i, face_j, src_coords, tgt_grid) 元组
            - face_i, face_j: 面索引
            - src_coords: (N, 2) 面 i 中有效源像素的 (row, col) 坐标
            - tgt_grid: (N, 2) 面 j 中对应点的归一化坐标（用于 grid_sample，范围 [-1, 1]）
    """
    cache_key = (fov_deg, face_size)
    if cache_key in _ccl_correspondence_cache:
        # 从缓存中取出并转移到目标设备
        cached = _ccl_correspondence_cache[cache_key]
        return [(fi, fj, sc.to(device), tg.to(device)) for fi, fj, sc, tg in cached]
    
    K = _build_intrinsic_matrix(fov_deg, face_size)
    K_inv = np.linalg.inv(K)
    rotations = _get_cubemap_rotations()
    
    # 生成面 i 的所有像素坐标网格
    rows, cols = np.meshgrid(np.arange(face_size), np.arange(face_size), indexing='ij')
    # 齐次坐标: (3, face_size*face_size)
    pixels_homo = np.stack([cols.ravel(), rows.ravel(), np.ones(face_size * face_size)], axis=0)  # (3, N)
    
    correspondences = []
    for face_i, face_j in CUBEMAP_ADJACENT_PAIRS:
        R_i = rotations[face_i].astype(np.float64)
        R_j = rotations[face_j].astype(np.float64)
        
        # 单应性矩阵: H_ij = K_j @ R_j @ R_i^{-1} @ K_i^{-1}
        # R 是 W2C 矩阵（正交），R^{-1} = R^T
        # 注意: 部分面的 R 包含镜像 (det=-1)，但 R^T 仍是其逆
        H_ij = K @ R_j @ R_i.T @ K_inv  # (3, 3)
        
        # 投影: p_j ~ H_ij @ p_i
        projected = H_ij @ pixels_homo  # (3, N)
        # 处理 z <= 0 的情况（射线在目标面后方，不会产生有效投影）
        z = projected[2:3, :]
        valid_z = z.ravel() > 1e-10  # z > 0 才在目标面前方
        z_safe = np.where(np.abs(z) > 1e-10, z, 1.0)  # 避免除零
        projected = projected / z_safe  # 归一化齐次坐标
        
        proj_col = projected[0, :]  # x 坐标
        proj_row = projected[1, :]  # y 坐标
        
        # 筛选有效对应点：z > 0（在目标面前方）且落在 [0, face_size-1] 范围内
        valid = valid_z.ravel() & \
                (proj_col >= 0) & (proj_col <= face_size - 1) & \
                (proj_row >= 0) & (proj_row <= face_size - 1)
        
        if valid.sum() == 0:
            continue
        
        # 源像素坐标 (row, col)
        src_rows = rows.ravel()[valid]
        src_cols = cols.ravel()[valid]
        src_coords = torch.tensor(np.stack([src_rows, src_cols], axis=1), dtype=torch.long)  # (N, 2)
        
        # 目标像素坐标转换为 grid_sample 的归一化坐标 [-1, 1]
        tgt_col_norm = proj_col[valid] / (face_size - 1) * 2 - 1  # [0, face_size-1] -> [-1, 1]
        tgt_row_norm = proj_row[valid] / (face_size - 1) * 2 - 1
        tgt_grid = torch.tensor(np.stack([tgt_col_norm, tgt_row_norm], axis=1), dtype=torch.float32)  # (N, 2)
        
        correspondences.append((face_i, face_j, src_coords, tgt_grid))
    
    # 缓存到 CPU（设备无关）
    _ccl_correspondence_cache[cache_key] = [
        (fi, fj, sc.cpu(), tg.cpu()) for fi, fj, sc, tg in correspondences
    ]
    
    return [(fi, fj, sc.to(device), tg.to(device)) for fi, fj, sc, tg in correspondences]


def correspondence_consistency_loss(
    intermediate_features: List[torch.Tensor],
    fov_deg: float = 95.0,
    max_correspondences: int = 4096,
) -> Tuple[torch.Tensor, dict]:
    """Correspondence Consistency Loss (CCL)。
    
    基于 cubemap 相邻面的重叠区域建立密集对应关系，
    在多层 encoder 特征上计算特征一致性 loss。
    
    Args:
        intermediate_features: 4 层中间特征列表，每层形状为 (B*6, C, H, W)
        fov_deg: cubemap 的视场角（度），默认 95.0
        max_correspondences: 每对面的最大对应点数，超过则随机采样
    
    Returns:
        loss: 标量 tensor
        misc: 包含调试信息的字典
    """
    device = intermediate_features[0].device
    dtype = intermediate_features[0].dtype
    BV, C, H, W = intermediate_features[0].shape
    assert BV % 6 == 0, f"Batch size * 6 expected, got {BV}"
    B = BV // 6
    face_size = H  # 特征图尺寸（应为 37）
    
    # 建立对应关系（带缓存）
    correspondences = _build_correspondences(fov_deg, face_size, device)
    
    if len(correspondences) == 0:
        return torch.tensor(0.0, device=device, dtype=dtype), {'ccl_loss': 0.0, 'num_correspondences': 0}
    
    num_layers = len(intermediate_features)
    layer_losses = []
    total_correspondences = 0
    
    for layer_idx in range(num_layers):
        feat = intermediate_features[layer_idx]  # (B*6, C, H, W)
        feat_reshaped = feat.view(B, 6, C, H, W)  # (B, 6, C, H, W)
        
        pair_losses = []
        for face_i, face_j, src_coords, tgt_grid in correspondences:
            N = src_coords.shape[0]
            if N == 0:
                continue
            
            # 随机采样以控制显存
            if N > max_correspondences:
                indices = torch.randperm(N, device=device)[:max_correspondences]
                src_coords_sampled = src_coords[indices]
                tgt_grid_sampled = tgt_grid[indices]
                N_sampled = max_correspondences
            else:
                src_coords_sampled = src_coords
                tgt_grid_sampled = tgt_grid
                N_sampled = N
            
            total_correspondences += N_sampled
            
            # 提取面 i 的源特征: (B, C, N_sampled)
            feat_i = feat_reshaped[:, face_i, :, src_coords_sampled[:, 0], src_coords_sampled[:, 1]]  # (B, C, N_sampled)
            
            # 使用 grid_sample 从面 j 提取对应特征
            # grid_sample 需要 (B, H_out, W_out, 2) 的 grid
            grid = tgt_grid_sampled.unsqueeze(0).unsqueeze(0).expand(B, 1, N_sampled, 2)  # (B, 1, N_sampled, 2)
            feat_j_map = feat_reshaped[:, face_j]  # (B, C, H, W)
            feat_j_sampled = F.grid_sample(
                feat_j_map, grid, mode='bilinear', padding_mode='border', align_corners=True
            )  # (B, C, 1, N_sampled)
            feat_j_sampled = feat_j_sampled.squeeze(2)  # (B, C, N_sampled)
            
            # 先对特征做 L2 归一化，再计算 cosine distance = 1 - cos_sim
            # 这样 loss 值范围在 [0, 2]，避免原始特征数值过大导致 loss 爆炸
            feat_i_norm = F.normalize(feat_i, p=2, dim=1, eps=1e-6)      # (B, C, N_sampled)
            feat_j_norm = F.normalize(feat_j_sampled, p=2, dim=1, eps=1e-6)  # (B, C, N_sampled)
            cos_sim = (feat_i_norm * feat_j_norm).sum(dim=1)  # (B, N_sampled)
            pair_loss = (1.0 - cos_sim).mean()
            pair_losses.append(pair_loss)
        
        if len(pair_losses) > 0:
            layer_loss = torch.stack(pair_losses).mean()
            layer_losses.append(layer_loss)
    
    if len(layer_losses) == 0:
        return torch.tensor(0.0, device=device, dtype=dtype), {'ccl_loss': 0.0, 'num_correspondences': 0}
    
    loss = torch.stack(layer_losses).mean()
    
    misc = {
        'ccl_loss': loss.item(),
        'num_correspondences': total_correspondences,
        'num_layers': num_layers,
        'per_layer_loss': [l.item() for l in layer_losses],
    }
    
    return loss, misc
