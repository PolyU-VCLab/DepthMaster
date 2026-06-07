"""
Cubemap → Equirectangular 投影工具。

提供两种模式:
1. 软混合模式 (默认): 对 ERP 每个像素，数学投影到所有 6 个 cubemap 面，
   在 overlap 区域用距离加权混合，消除接缝。
2. 硬分配模式 (legacy): 使用预计算映射文件，每个像素只从一个面采样。

cubemap 面顺序: [Front(0°), Right(90°), Back(180°), Left(270°), Up(90°↑), Down(90°↓)]
与 dataset_readers.py 中 _e2c_fov95 的面顺序一致。
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates
from pathlib import Path

# 预计算文件路径（硬分配模式使用）
_PRE_PANO_DIR = Path(__file__).absolute().parents[2] / 'ref_code' / 'pre_pano'

# ============================================================
# Cubemap 面的相机参数（FOV=95°）
# ============================================================

_FOV_DEG = 95.0
_HALF_FOV_RAD = np.deg2rad(_FOV_DEG / 2.0)
_TAN_HALF_FOV = np.tan(_HALF_FOV_RAD)

def _get_cubemap_rotations():
    """获取 6 个 cubemap 面的 W2C 旋转矩阵（用于 cubemap ↔ ERP 投影）。
    
    注意：这些 W2C 矩阵与 _get_e2p_remap_maps 的正向投影一致，
    隐含 y-up 世界坐标系（ERP 球面坐标自然约定）。
    
    这与 dataset_readers.py::_get_cubemap_K_W2C 中的 y-down W2C 矩阵不同！
    _get_cubemap_K_W2C 的 W2C 矩阵用于 camera pose 和点云（OpenCV 约定），
    而这里的 W2C 矩阵仅用于 cubemap ↔ ERP 的投影/反投影。
    
    坐标系约定:
      - 相机坐标系: x-right, y-down, z-forward (OpenCV 标准)
      - 世界坐标系: x-right, y-up, z-forward (ERP 球面坐标自然约定)
      - 所有 6 个面的 det(R) = +1 (proper rotation)
    
    面顺序: [Front(z+), Right(x+), Back(z-), Left(x-), Up(y+天空), Down(y-地面)]
    
    Returns:
        rotations: (6, 3, 3) numpy array, W2C 旋转矩阵
    """
    return np.array([
        # Front: 看向 z+
        [[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]],
        # Right: 看向 x+
        [[ 0,  0, -1], [ 0,  1,  0], [ 1,  0,  0]],
        # Back: 看向 z-
        [[-1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]],
        # Left: 看向 x-
        [[ 0,  0,  1], [ 0,  1,  0], [-1,  0,  0]],
        # Up: 看向 y+（天空）
        [[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]],
        # Down: 看向 y-（地面）
        [[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]],
    ], dtype=np.float32)  # (6, 3, 3)


# 缓存旋转矩阵（修改 _get_cubemap_rotations 后需要清除缓存）
_cached_rotations = None

def _get_rotations():
    global _cached_rotations
    if _cached_rotations is None:
        _cached_rotations = _get_cubemap_rotations()
    return _cached_rotations


# 缓存 torch 版本的旋转矩阵（修改 _get_cubemap_rotations 后需要清除缓存）
_cached_rotations_torch = {}

def _get_rotations_torch(device, dtype=torch.float32):
    key = (device, dtype)
    if key not in _cached_rotations_torch:
        R = _get_rotations()
        _cached_rotations_torch[key] = torch.from_numpy(R).to(device=device, dtype=dtype)
    return _cached_rotations_torch[key]


# ============================================================
# 缓存 ERP 方向向量
# ============================================================

_cached_erp_dirs = {}

def _get_erp_directions_np(pano_h, pano_w):
    """获取 ERP 每个像素的球面方向向量 (numpy)。"""
    key = (pano_h, pano_w)
    if key not in _cached_erp_dirs:
        # u: [0, 1], v: [0, 1]
        u = np.linspace(0.5 / pano_w, 1.0 - 0.5 / pano_w, pano_w, dtype=np.float32)
        v = np.linspace(0.5 / pano_h, 1.0 - 0.5 / pano_h, pano_h, dtype=np.float32)
        u, v = np.meshgrid(u, v)
        
        # ERP 坐标 → 球面方向
        # theta: 经度 [-pi, pi], phi: 纬度 [pi/2, -pi/2]
        theta = (u - 0.5) * 2 * np.pi   # [0,1] → [-pi, pi]
        phi = (0.5 - v) * np.pi          # [0,1] → [pi/2, -pi/2]
        
        x = np.cos(phi) * np.sin(theta)
        # ERP 方向向量使用 y-up 世界坐标（与 _get_e2p_remap_maps 正向投影一致）：
        #   ERP 顶部（天空，phi=+pi/2）→ y=+1
        #   ERP 底部（地面，phi=-pi/2）→ y=-1
        y = np.sin(phi)
        z = np.cos(phi) * np.cos(theta)
        
        dirs = np.stack([x, y, z], axis=-1)  # (H, W, 3)
        _cached_erp_dirs[key] = dirs
    return _cached_erp_dirs[key]


_cached_erp_dirs_torch = {}

def _get_erp_directions_torch(pano_h, pano_w, device, dtype=torch.float32):
    """获取 ERP 每个像素的球面方向向量 (torch)。"""
    key = (pano_h, pano_w, device, dtype)
    if key not in _cached_erp_dirs_torch:
        dirs_np = _get_erp_directions_np(pano_h, pano_w)
        _cached_erp_dirs_torch[key] = torch.from_numpy(dirs_np).to(device=device, dtype=dtype)
    return _cached_erp_dirs_torch[key]


# ============================================================
# 软混合 Numpy 版本（用于可视化/评估）
# ============================================================

def cubemap_to_equirect_np(
    cubemap: np.ndarray,
    pano_h: int = 512,
    pano_w: int = 1024,
    fov_deg: float = 95.0,
    interpolation: str = 'bilinear',
    blend: bool = True,
) -> np.ndarray:
    """将 6 个 cubemap 面拼回 equirectangular 全景图（numpy 版本）。

    Args:
        cubemap: (6, face_w, face_w) 或 (6, face_w, face_w, C) cubemap 面
                 面顺序: [Front, Right, Back, Left, Up, Down]
        pano_h: 输出全景图高度
        pano_w: 输出全景图宽度
        fov_deg: cubemap 面的 FOV（度）
        interpolation: 'bilinear' 或 'nearest'
        blend: True 使用软混合（消除接缝），False 使用硬分配

    Returns:
        pano: (pano_h, pano_w) 或 (pano_h, pano_w, C) 全景图
    """
    if not blend:
        # 回退到硬分配模式（需要 512x1024）
        assert pano_h == 512 and pano_w == 1024, \
            f"硬分配模式固定输出 512x1024，但收到 {pano_h}x{pano_w}"
        has_channel = cubemap.ndim == 4
        if has_channel:
            C = cubemap.shape[3]
            results = []
            for c in range(C):
                results.append(_cubemap_to_equirect_np_hard(cubemap[..., c], interpolation))
            return np.stack(results, axis=-1)
        else:
            return _cubemap_to_equirect_np_hard(cubemap, interpolation)
    
    # 软混合模式
    has_channel = cubemap.ndim == 4
    if has_channel:
        C = cubemap.shape[3]
        results = []
        for c in range(C):
            results.append(_cubemap_to_equirect_np_blend(cubemap[..., c], pano_h, pano_w, fov_deg, interpolation))
        return np.stack(results, axis=-1)
    else:
        return _cubemap_to_equirect_np_blend(cubemap, pano_h, pano_w, fov_deg, interpolation)


def _cubemap_to_equirect_np_blend(
    cube_faces: np.ndarray,
    pano_h: int,
    pano_w: int,
    fov_deg: float = 95.0,
    interpolation: str = 'bilinear',
) -> np.ndarray:
    """单通道 cubemap → ERP 投影，使用软混合消除接缝（numpy 版本）。

    对 ERP 每个像素，计算它在所有 6 个面上的投影坐标，
    对落在有效区域内的面用距离加权混合。

    Args:
        cube_faces: (6, face_w, face_w) 单通道 cubemap 面
        pano_h: 输出全景图高度
        pano_w: 输出全景图宽度
        fov_deg: cubemap 面的 FOV（度）
        interpolation: 'bilinear' 或 'nearest'

    Returns:
        erp: (pano_h, pano_w) 全景图
    """
    face_w = cube_faces.shape[1]
    rotations = _get_rotations()  # (6, 3, 3)
    dirs = _get_erp_directions_np(pano_h, pano_w)  # (H, W, 3)
    
    half_fov_rad = np.deg2rad(fov_deg / 2.0)
    tan_half_fov = np.tan(half_fov_rad)
    
    # 坐标系说明:
    #   ERP 方向向量使用 y-up 世界坐标（与 _get_e2p_remap_maps 正向投影一致）。
    #   W2C 矩阵将 y-up 世界坐标变换到 y-down 相机坐标。
    #   但正向投影中 cubemap 图像的 r=0（顶部）对应 y_rng=+y_max（天空方向），
    #   即 cubemap 图像的行方向与 OpenCV 相机 y 轴方向相反。
    #   因此反投影时 py 需要对 y_cam 取反:
    #   px = ( X/Z / tan_half_fov + 1) / 2 * (face_w - 1)
    #   py = (-Y/Z / tan_half_fov + 1) / 2 * (face_w - 1)
    
    order = 0 if interpolation == 'nearest' else 1
    
    result = np.zeros((pano_h, pano_w), dtype=np.float64)
    weight_sum = np.zeros((pano_h, pano_w), dtype=np.float64)
    
    for fi in range(6):
        R = rotations[fi]  # (3, 3) W2C 旋转矩阵
        
        # 将世界方向变换到相机坐标系
        # dirs: (H, W, 3), R: (3, 3) → cam_dirs: (H, W, 3)
        cam_dirs = np.einsum('hwc,dc->hwd', dirs, R)  # (H, W, 3)
        
        z_cam = cam_dirs[..., 2]  # (H, W)
        
        # 只处理 z > 0 的像素（在相机前方）
        front_mask = z_cam > 1e-6
        
        x_cam = cam_dirs[..., 0]
        y_cam = cam_dirs[..., 1]
        
        # 归一化平面坐标
        safe_z = np.where(front_mask, z_cam, 1.0)
        xz = x_cam / safe_z   # X/Z
        yz = y_cam / safe_z   # Y/Z
        
        # 像素坐标: px 正常, py 对 y_cam 取反（因为 cubemap 图像行方向与 y_cam 相反）
        px = np.where(front_mask, (xz / tan_half_fov + 1.0) / 2.0 * (face_w - 1), -1.0)
        py = np.where(front_mask, (-yz / tan_half_fov + 1.0) / 2.0 * (face_w - 1), -1.0)
        
        # 有效区域: 像素坐标在 [0, face_w-1] 范围内
        valid_mask = front_mask & (px >= 0) & (px <= face_w - 1) & (py >= 0) & (py <= face_w - 1)
        
        if not np.any(valid_mask):
            continue
        
        # 计算权重: 离面中心越近权重越大
        # 使用归一化平面坐标，权重 = cos(|xz/tan_half_fov| * pi/2) * cos(|yz/tan_half_fov| * pi/2)
        u_norm = np.where(valid_mask, np.abs(xz / tan_half_fov), 1.0)  # [0, 1]
        v_norm = np.where(valid_mask, np.abs(yz / tan_half_fov), 1.0)  # [0, 1]
        u_clamped = np.clip(u_norm, 0, 1)
        v_clamped = np.clip(v_norm, 0, 1)
        w = np.where(valid_mask, np.cos(u_clamped * np.pi / 2) * np.cos(v_clamped * np.pi / 2), 0.0)
        
        # 采样
        if order == 0:
            # 最近邻
            ix = np.clip(np.round(px).astype(int), 0, face_w - 1)
            iy = np.clip(np.round(py).astype(int), 0, face_w - 1)
            sampled = np.where(valid_mask, cube_faces[fi, iy, ix], 0.0)
        else:
            # 双线性插值 - 使用 map_coordinates
            # 只对有效像素采样以提高效率
            valid_indices = np.where(valid_mask)
            sampled = np.zeros((pano_h, pano_w), dtype=np.float64)
            if len(valid_indices[0]) > 0:
                coords = np.array([py[valid_indices], px[valid_indices]])
                vals = map_coordinates(cube_faces[fi], coords, order=1, mode='nearest')
                sampled[valid_indices] = vals
        
        result += sampled * w
        weight_sum += w
    
    # 归一化（避免除零 warning）
    safe_weight = np.where(weight_sum > 1e-8, weight_sum, 1.0)
    result = np.where(weight_sum > 1e-8, result / safe_weight, 0.0)
    return result.astype(np.float32)


# ============================================================
# 软混合 Torch 版本（用于训练中的 loss 计算，支持梯度反传）
# ============================================================

# 缓存投影映射（避免每次重复计算）
_cached_blend_maps_torch = {}
# 硬分配映射缓存
_cached_hard_maps_torch = {}


def _get_hard_assign_maps_torch(pano_h, pano_w, face_w, fov_deg, device, dtype=torch.float32):
    """硬分配版本的投影映射：每个 ERP 像素只来自权重最大的那个面。
    
    复用 _get_blend_maps_torch 的所有几何计算（保证坐标约定一致），
    然后对每个 ERP 像素的 6 个面权重做 argmax，把硬权重设为 1.0（仅最大面）/ 0（其他）。
    
    Returns:
        grids_list: list of 6 个 tensor，每个 (N_i, 2)，N_i = 该面被硬分配到的 ERP 像素数
        masks: (6, pano_h, pano_w) bool，每个面被硬分配的像素掩码 (互斥, 加和 = 全 1)
        weights: 全 1（保留接口）
    """
    key = (pano_h, pano_w, face_w, fov_deg, device, dtype)
    if key in _cached_hard_maps_torch:
        return _cached_hard_maps_torch[key]
    
    # 复用 blend 的几何映射（grids、初始 weights、初始 masks）
    blend_grids, blend_masks, blend_weights = _get_blend_maps_torch(
        pano_h, pano_w, face_w, fov_deg, device, dtype
    )
    # blend_weights: (6, pano_h, pano_w) 已归一化
    # 对每个 ERP 像素，取权重最大的那个面（argmax）
    # 注意：极少数像素可能 6 面权重都为 0（如果 fov < 90°，理论上不会发生 fov=95°）
    weight_sum = blend_weights.sum(dim=0)  # (pano_h, pano_w)
    valid_pixel = weight_sum > 1e-8        # (pano_h, pano_w) 至少有一个面覆盖
    argmax_face = blend_weights.argmax(dim=0)  # (pano_h, pano_w) 每个像素归属面 id
    
    # 重建 hard masks 和 hard grids
    hard_masks_list = []
    hard_grids_list = []
    for fi in range(6):
        # 本面被硬分配的 ERP 像素 = (argmax == fi) & 该面有效（覆盖）& 全局有效
        face_valid = blend_masks[fi]                                # (pano_h, pano_w)
        hard_mask = (argmax_face == fi) & face_valid & valid_pixel  # (pano_h, pano_w)
        hard_masks_list.append(hard_mask)
        
        if not hard_mask.any():
            hard_grids_list.append(torch.zeros((0, 2), device=device, dtype=dtype))
            continue
        
        # 直接复用 blend 的 face_valid 像素 → 归一化坐标的映射；
        # blend_grids[fi] 是 (N_face_valid, 2)，按 face_valid 的 row-major 顺序排列
        # 我们需要从中筛出 hard_mask 对应的子集
        
        # face_valid 在 ERP 上 row-major flatten 后的索引
        face_valid_flat = face_valid.flatten()  # (pano_h * pano_w,)
        hard_mask_flat = hard_mask.flatten()     # (pano_h * pano_w,)
        
        # blend_grids[fi] 中第 k 个对应原 ERP 像素中"face_valid 为 True 的第 k 个像素"
        # 我们需要找：哪些 face_valid 像素同时是 hard_mask 像素
        # 即：在 face_valid 的 True 子序列里，对应位置 hard_mask 是否也 True
        select_in_face_valid = hard_mask_flat[face_valid_flat]  # (N_face_valid,)
        hard_grids_list.append(blend_grids[fi][select_in_face_valid])  # (N_hard_fi, 2)
    
    hard_masks = torch.stack(hard_masks_list)                     # (6, pano_h, pano_w)
    hard_weights = torch.ones(6, pano_h, pano_w,
                               device=device, dtype=dtype)         # 占位，硬分配下每个像素权重就是 1
    
    result = (hard_grids_list, hard_masks, hard_weights)
    _cached_hard_maps_torch[key] = result
    return result

def _get_blend_maps_torch(pano_h, pano_w, face_w, fov_deg, device, dtype=torch.float32):
    """预计算并缓存软混合所需的投影映射。
    
    Returns:
        grids: (6, H_valid, 1, 2) 每个面的有效像素的归一化采样坐标
        masks: (6, pano_h, pano_w) bool, 每个面的有效像素掩码
        weights: (6, pano_h, pano_w) float, 每个面的混合权重
        valid_indices: list of (H_valid,) 每个面有效像素的 flat indices
    """
    key = (pano_h, pano_w, face_w, fov_deg, device, dtype)
    if key in _cached_blend_maps_torch:
        return _cached_blend_maps_torch[key]
    
    rotations = _get_rotations_torch(device, dtype)  # (6, 3, 3)
    dirs = _get_erp_directions_torch(pano_h, pano_w, device, dtype)  # (H, W, 3)
    
    half_fov_rad = np.deg2rad(fov_deg / 2.0)
    tan_half_fov = np.tan(half_fov_rad)
    
    grids_list = []
    masks_list = []
    weights = torch.zeros(6, pano_h, pano_w, device=device, dtype=dtype)
    
    for fi in range(6):
        R = rotations[fi]  # (3, 3)
        cam_dirs = torch.einsum('hwc,dc->hwd', dirs, R)  # (H, W, 3)
        z_cam = cam_dirs[..., 2]
        
        front_mask = z_cam > 1e-6
        x_cam = cam_dirs[..., 0]
        y_cam = cam_dirs[..., 1]
        
        # 归一化平面坐标
        safe_z = torch.where(front_mask, z_cam, torch.ones_like(z_cam))
        xz = x_cam / safe_z   # X/Z
        yz = y_cam / safe_z   # Y/Z
        
        # 像素坐标: px 正常, py 对 y_cam 取反（因为 cubemap 图像行方向与 y_cam 相反）
        px = (xz / tan_half_fov + 1.0) / 2.0 * (face_w - 1)
        py = (-yz / tan_half_fov + 1.0) / 2.0 * (face_w - 1)
        
        valid_mask = front_mask & (px >= 0) & (px <= face_w - 1) & (py >= 0) & (py <= face_w - 1)
        masks_list.append(valid_mask)
        
        # 权重: 使用归一化平面坐标
        u_abs = torch.clamp(torch.abs(xz / tan_half_fov), 0, 1)
        v_abs = torch.clamp(torch.abs(yz / tan_half_fov), 0, 1)
        w = torch.cos(u_abs * np.pi / 2) * torch.cos(v_abs * np.pi / 2)
        w = torch.where(valid_mask, w, torch.zeros_like(w))
        weights[fi] = w
        
        # 归一化采样坐标 [-1, 1] for grid_sample (align_corners=True)
        grid_x = torch.where(valid_mask, 2.0 * px / (face_w - 1) - 1.0, torch.zeros_like(px))
        grid_y = torch.where(valid_mask, 2.0 * py / (face_w - 1) - 1.0, torch.zeros_like(py))
        
        # 提取有效像素的坐标
        valid_grid = torch.stack([grid_x[valid_mask], grid_y[valid_mask]], dim=1)  # (N, 2)
        grids_list.append(valid_grid)
    
    masks = torch.stack(masks_list)  # (6, H, W)
    
    # 归一化权重
    weight_sum = weights.sum(dim=0, keepdim=True).clamp(min=1e-8)  # (1, H, W)
    weights = weights / weight_sum  # (6, H, W)
    
    result = (grids_list, masks, weights)
    _cached_blend_maps_torch[key] = result
    return result


def cubemap_to_equirect_torch(
    cubemap: torch.Tensor,
    pano_h: int = 512,
    pano_w: int = 1024,
    fov_deg: float = 95.0,
    mode: str = 'bilinear',
    blend: bool = True,
) -> torch.Tensor:
    """将 6 个 cubemap 面拼回 equirectangular 全景图（torch 版本，支持 batch 和梯度反传）。

    Args:
        cubemap: (B, 6, H, W) cubemap 面
        pano_h: 输出全景图高度
        pano_w: 输出全景图宽度
        fov_deg: cubemap 面的 FOV（度）
        mode: 'bilinear' 或 'nearest'
        blend: True 使用软混合（消除接缝），False 使用硬分配

    Returns:
        pano: (B, pano_h, pano_w) 全景图
    """
    if not blend:
        # 硬分配版本（任意分辨率，保证坐标约定与 blend 版本一致）
        return _cubemap_to_equirect_torch_hard_v2(
            cubemap, pano_h=pano_h, pano_w=pano_w,
            fov_deg=fov_deg, mode=mode,
        )
    
    device = cubemap.device
    dtype = cubemap.dtype
    B = cubemap.shape[0]
    face_w = cubemap.shape[2]
    
    grids_list, masks, weights = _get_blend_maps_torch(pano_h, pano_w, face_w, fov_deg, device, dtype)
    
    grid_sample_mode = 'nearest' if mode == 'nearest' else 'bilinear'
    
    result = torch.zeros(B, pano_h, pano_w, device=device, dtype=dtype)
    
    for fi in range(6):
        mask = masks[fi]  # (H, W) bool
        w = weights[fi]   # (H, W) float, 已归一化
        
        if not mask.any():
            continue
        
        valid_grid = grids_list[fi]  # (N, 2)
        N = valid_grid.shape[0]
        
        # 创建采样网格
        grid = valid_grid.view(1, N, 1, 2).expand(B, -1, -1, -1)  # (B, N, 1, 2)
        
        # 提取当前面的数据
        face_data = cubemap[:, fi:fi+1]  # (B, 1, face_w, face_w)
        
        # 采样
        sampled = F.grid_sample(
            face_data, grid,
            mode=grid_sample_mode,
            padding_mode='border',
            align_corners=True,
        )  # (B, 1, N, 1)
        
        sampled = sampled.squeeze(-1).squeeze(1)  # (B, N)
        
        # 加权累加到结果
        # result[:, mask] += sampled * w[mask]
        w_valid = w[mask]  # (N,)
        result[:, mask] = result[:, mask] + sampled * w_valid.unsqueeze(0)
    
    return result


def _cubemap_to_equirect_torch_hard_v2(
    cubemap: torch.Tensor,
    pano_h: int = 512,
    pano_w: int = 1024,
    fov_deg: float = 95.0,
    mode: str = 'bilinear',
) -> torch.Tensor:
    """硬分配版本（v2）：每个 ERP 像素只来自权重最大的那个面，无 blending。
    
    - 与 blend=True 版本共享坐标约定（_get_blend_maps_torch 的几何）
    - 不依赖预计算 npy 文件
    - 支持任意分辨率
    
    Args:
        cubemap: (B, 6, H, W) cubemap 面
        pano_h: 输出全景图高度
        pano_w: 输出全景图宽度
        fov_deg: cubemap 面的 FOV（度）
        mode: 'bilinear' 或 'nearest'
    
    Returns:
        pano: (B, pano_h, pano_w)
    """
    device = cubemap.device
    dtype = cubemap.dtype
    B = cubemap.shape[0]
    face_w = cubemap.shape[2]
    
    grids_list, masks, _ = _get_hard_assign_maps_torch(
        pano_h, pano_w, face_w, fov_deg, device, dtype
    )
    
    grid_sample_mode = 'nearest' if mode == 'nearest' else 'bilinear'
    
    result = torch.zeros(B, pano_h, pano_w, device=device, dtype=dtype)
    
    for fi in range(6):
        mask = masks[fi]  # (pano_h, pano_w) bool, 互斥（每个像素仅属一个面）
        if not mask.any():
            continue
        
        valid_grid = grids_list[fi]  # (N, 2)
        N = valid_grid.shape[0]
        if N == 0:
            continue
        
        # 创建采样网格
        grid = valid_grid.view(1, N, 1, 2).expand(B, -1, -1, -1)  # (B, N, 1, 2)
        
        # 提取当前面的数据并采样
        face_data = cubemap[:, fi:fi+1]  # (B, 1, face_w, face_w)
        sampled = F.grid_sample(
            face_data, grid,
            mode=grid_sample_mode,
            padding_mode='border',
            align_corners=True,
        )  # (B, 1, N, 1)
        sampled = sampled.squeeze(-1).squeeze(1)  # (B, N)
        
        # 直接赋值（无加权累加，因为各面 mask 互斥）
        result[:, mask] = sampled
    
    return result



def _load_precomputed_maps():
    """加载预计算的 cubemap → ERP 映射文件。"""
    tp = np.load(str(_PRE_PANO_DIR / 'tp.npy'))        # (512, 1024) int32, 面索引 0-5
    coor_y = np.load(str(_PRE_PANO_DIR / 'coor_y.npy'))  # (512, 1024) float64
    coor_x = np.load(str(_PRE_PANO_DIR / 'coor_x.npy'))  # (512, 1024) float64
    return tp, coor_y, coor_x

_cached_maps = None

def _get_precomputed_maps():
    global _cached_maps
    if _cached_maps is None:
        _cached_maps = _load_precomputed_maps()
    return _cached_maps

_cached_torch_maps = {}

def _get_precomputed_maps_torch(device, dtype=torch.float32):
    key = (device, dtype)
    if key not in _cached_torch_maps:
        tp, coor_y, coor_x = _get_precomputed_maps()
        _cached_torch_maps[key] = (
            torch.from_numpy(tp).to(device=device, dtype=dtype),
            torch.from_numpy(coor_y).to(device=device, dtype=dtype),
            torch.from_numpy(coor_x).to(device=device, dtype=dtype),
        )
    return _cached_torch_maps[key]


def _cubemap_to_equirect_np_hard(
    cube_faces: np.ndarray,
    interpolation: str = 'bilinear',
) -> np.ndarray:
    """单通道 cubemap → ERP 投影（硬分配，numpy 版本）。
    
    注意：预计算映射文件 (tp.npy, coor_y.npy, coor_x.npy) 基于旧坐标约定。
    需要先将正视角 cubemap 面变换回旧坐标约定，再使用预计算映射。
    
    旧坐标约定的 Up/Down 面需要 rot90 处理（这是预计算映射的要求）。
    从正视角（新坐标）到旧坐标的变换:
      Front/Right/Back/Left: flipud（正视角 y-down → Rodrigues y-up）
      Up:    恒等（rot90_ccw 变为 Rodrigues，再 rot90_cw 抵消，净变换=恒等）
      Down:  恒等（rot90_cw 变为 Rodrigues，再 rot90_ccw 抵消，净变换=恒等）
    """
    tp, coor_y, coor_x = _get_precomputed_maps()

    cube_faces = cube_faces.copy()
    # 将正视角（新坐标）变换回旧坐标约定
    cube_faces[0] = cube_faces[0][::-1]   # Front: flipud
    cube_faces[1] = cube_faces[1][::-1]   # Right: flipud
    cube_faces[2] = cube_faces[2][::-1]   # Back:  flipud
    cube_faces[3] = cube_faces[3][::-1]   # Left:  flipud
    # Up/Down: 净变换为恒等，无需操作

    pad_ud = np.zeros((6, 2, cube_faces.shape[2]), dtype=cube_faces.dtype)
    pad_ud[0, 0] = cube_faces[5, 0, :]
    pad_ud[0, 1] = cube_faces[4, -1, :]
    pad_ud[1, 0] = cube_faces[5, :, -1]
    pad_ud[1, 1] = cube_faces[4, ::-1, -1]
    pad_ud[2, 0] = cube_faces[5, -1, ::-1]
    pad_ud[2, 1] = cube_faces[4, 0, ::-1]
    pad_ud[3, 0] = cube_faces[5, ::-1, 0]
    pad_ud[3, 1] = cube_faces[4, :, 0]
    pad_ud[4, 0] = cube_faces[0, 0, :]
    pad_ud[4, 1] = cube_faces[2, 0, ::-1]
    pad_ud[5, 0] = cube_faces[2, -1, ::-1]
    pad_ud[5, 1] = cube_faces[0, -1, :]
    cube_faces = np.concatenate([cube_faces, pad_ud], axis=1)

    pad_lr = np.zeros((6, cube_faces.shape[1], 2), dtype=cube_faces.dtype)
    pad_lr[0, :, 0] = cube_faces[1, :, 0]
    pad_lr[0, :, 1] = cube_faces[3, :, -1]
    pad_lr[1, :, 0] = cube_faces[2, :, 0]
    pad_lr[1, :, 1] = cube_faces[0, :, -1]
    pad_lr[2, :, 0] = cube_faces[3, :, 0]
    pad_lr[2, :, 1] = cube_faces[1, :, -1]
    pad_lr[3, :, 0] = cube_faces[0, :, 0]
    pad_lr[3, :, 1] = cube_faces[2, :, -1]
    pad_lr[4, 1:-1, 0] = cube_faces[1, 0, ::-1]
    pad_lr[4, 1:-1, 1] = cube_faces[3, 0, :]
    pad_lr[5, 1:-1, 0] = cube_faces[1, -2, :]
    pad_lr[5, 1:-1, 1] = cube_faces[3, -2, ::-1]
    cube_faces = np.concatenate([cube_faces, pad_lr], axis=2)

    order = 0 if interpolation == 'nearest' else 1
    return map_coordinates(cube_faces, [tp, coor_y, coor_x], order=order, mode='wrap')


def _cubemap_to_equirect_torch_hard(
    cubemap: torch.Tensor,
    pano_h: int = 512,
    pano_w: int = 1024,
    fov_deg: float = 95.0,
    mode: str = 'bilinear',
) -> torch.Tensor:
    """硬分配版本的 torch cubemap → ERP 投影。"""
    device = cubemap.device
    dtype = cubemap.dtype
    B = cubemap.shape[0]

    tp, coor_y, coor_x = _get_precomputed_maps_torch(device, dtype)

    cube_faces = cube_faces.clone()
    # 将正视角（新坐标）变换回旧坐标约定
    cube_faces[:, 0] = torch.flip(cube_faces[:, 0], [1])   # Front: flipud
    cube_faces[:, 1] = torch.flip(cube_faces[:, 1], [1])   # Right: flipud
    cube_faces[:, 2] = torch.flip(cube_faces[:, 2], [1])   # Back:  flipud
    cube_faces[:, 3] = torch.flip(cube_faces[:, 3], [1])   # Left:  flipud
    # Up/Down: 净变换为恒等，无需操作
    pad_ud = torch.zeros((B, 6, 2, cube_faces.shape[3]), device=device, dtype=dtype)
    pad_ud[:, 0, 0] = cube_faces[:, 5, 0, :]
    pad_ud[:, 0, 1] = cube_faces[:, 4, -1, :]
    pad_ud[:, 1, 0] = cube_faces[:, 5, :, -1]
    pad_ud[:, 1, 1] = torch.flip(cube_faces[:, 4, :, -1], [1])
    pad_ud[:, 2, 0] = torch.flip(cube_faces[:, 5, -1, :], [1])
    pad_ud[:, 2, 1] = torch.flip(cube_faces[:, 4, 0, :], [1])
    pad_ud[:, 3, 0] = torch.flip(cube_faces[:, 5, :, 0], [1])
    pad_ud[:, 3, 1] = cube_faces[:, 4, :, 0]
    pad_ud[:, 4, 0] = cube_faces[:, 0, 0, :]
    pad_ud[:, 4, 1] = torch.flip(cube_faces[:, 2, 0, :], [1])
    pad_ud[:, 5, 0] = torch.flip(cube_faces[:, 2, -1, :], [1])
    pad_ud[:, 5, 1] = cube_faces[:, 0, -1, :]
    cube_faces = torch.cat([cube_faces, pad_ud], dim=2)

    pad_lr = torch.zeros((B, 6, cube_faces.shape[2], 2), device=device, dtype=dtype)
    pad_lr[:, 0, :, 0] = cube_faces[:, 1, :, 0]
    pad_lr[:, 0, :, 1] = cube_faces[:, 3, :, -1]
    pad_lr[:, 1, :, 0] = cube_faces[:, 2, :, 0]
    pad_lr[:, 1, :, 1] = cube_faces[:, 0, :, -1]
    pad_lr[:, 2, :, 0] = cube_faces[:, 3, :, 0]
    pad_lr[:, 2, :, 1] = cube_faces[:, 1, :, -1]
    pad_lr[:, 3, :, 0] = cube_faces[:, 0, :, 0]
    pad_lr[:, 3, :, 1] = cube_faces[:, 2, :, -1]
    pad_lr[:, 4, 1:-1, 0] = torch.flip(cube_faces[:, 1, 0, :], [1])
    pad_lr[:, 4, 1:-1, 1] = cube_faces[:, 3, 0, :]
    pad_lr[:, 5, 1:-1, 0] = cube_faces[:, 1, -2, :]
    pad_lr[:, 5, 1:-1, 1] = torch.flip(cube_faces[:, 3, -2, :], [1])
    cube_faces = torch.cat([cube_faces, pad_lr], dim=3)

    H_padded = cube_faces.shape[2]
    W_padded = cube_faces.shape[3]

    y_normalized = 2.0 * coor_y / (H_padded - 1) - 1.0
    x_normalized = 2.0 * coor_x / (W_padded - 1) - 1.0

    grid_sample_mode = 'nearest' if mode == 'nearest' else 'bilinear'

    result = torch.zeros(B, pano_h, pano_w, device=device, dtype=dtype)

    for fi in range(6):
        mask = (tp == fi)
        if not mask.any():
            continue

        grid_y = y_normalized[mask]
        grid_x = x_normalized[mask]

        N = grid_y.shape[0]
        grid = torch.stack([grid_x, grid_y], dim=1)
        grid = grid.view(1, N, 1, 2).expand(B, -1, -1, -1)

        face_data = cube_faces[:, fi:fi+1]

        sampled = F.grid_sample(
            face_data, grid,
            mode=grid_sample_mode,
            padding_mode='border',
            align_corners=True,
        )

        result[:, mask] = sampled.squeeze(-1).squeeze(1)

    return result
