"""Per-dataset reader functions.

All reader functions share the unified signature::

    reader(instance: dict, read_image_fn) -> Optional[dict]

Returning a dict containing at least ``image``, ``depth``, ``intrinsics``
(perspective) or panorama-specific keys, or ``None`` if the sample failed
to load.
"""

from pathlib import Path
from typing import Optional, Dict, Callable, Tuple, List
from functools import lru_cache
import os

# Enable OpenCV's OpenEXR backend at module import time so that EXR-backed
# panorama readers work regardless of import order.
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import numpy as np
import cv2
from PIL import Image
import utils3d


# ---------- Edge flying-point filter constants ----------
EDGE_MASK_RTOL = 0.1  # relative tolerance for edge flying-point filter


# ---------- 通用工具函数 ----------

def _filter_edge_flying_points(depth: np.ndarray) -> np.ndarray:
    """过滤深度图中的边缘飞点（flying points）。
    
    使用 utils3d.np.depth_map_edge 检测深度不连续的边缘区域，
    将这些区域的深度值设为 nan，使其不参与训练。
    
    Args:
        depth: 深度图，float32，包含有限正值、inf（天空）和 nan（无效）
    
    Returns:
        过滤后的深度图，边缘飞点区域被设为 nan
    """
    valid_mask = np.isfinite(depth) & (depth > 0)
    if not np.any(valid_mask):
        return depth
    edge_mask = utils3d.np.depth_map_edge(depth, rtol=EDGE_MASK_RTOL, mask=valid_mask)
    depth = depth.copy()
    depth[edge_mask] = np.nan
    return depth


def _process_depth(depth: np.ndarray) -> np.ndarray:
    """统一的深度图后处理：将无效值设为 nan，保留 inf（天空/无穷远标记）。
    
    处理后深度图只有三种状态：
      - 有限正值: 有效深度，参与深度回归 loss
      - inf: 天空/无穷远，作为 mask loss 负样本
      - nan: 未知/无效，不参与任何 loss
    
    注意: 深度范围裁剪由 dataloader 的 clamp_max_depth 动态完成，此处不做 clip。
    """
    depth = depth.astype(np.float32)
    depth[depth <= 0] = np.nan
    # 保留 inf（天空标记），仅将 nan 保持为 nan
    depth[np.isnan(depth)] = np.nan
    return depth


def _normalize_intrinsics(K: np.ndarray, H: int, W: int) -> np.ndarray:
    """将像素坐标内参归一化为 DepthMaster 使用的 [0,1] 归一化内参"""
    return np.array([
        [K[0, 0] / W, 0.0,          K[0, 2] / W],
        [0.0,          K[1, 1] / H,  K[1, 2] / H],
        [0.0,          0.0,          1.0         ]
    ], dtype=np.float32)


def _load_depthmaster_depth(depthmaster_depth_path: Path, H: int, W: int) -> Optional[np.ndarray]:
    """加载 depthmaster_depth 并 resize 到与 GT 深度相同的尺寸。
    
    depthmaster_depth 中 <= 0 的值视为天空/无穷远（depthmaster 预测时根据 sky_mask 过滤），设为 inf。
    原始 nan/非有限值保持为 nan（无效/未知区域）。
    返回 float32 数组，形状 (H, W)；如果文件不存在则返回 None。
    """
    depthmaster_depth_path = Path(depthmaster_depth_path)
    if not depthmaster_depth_path.exists():
        return None
    depthmaster_depth = np.load(str(depthmaster_depth_path)).astype(np.float32)
    # 原始 nan 保持为 nan（无效/未知区域）
    nan_mask = np.isnan(depthmaster_depth)
    # depthmaster_depth 中 <= 0 表示天空（depthmaster 预测时已根据 sky_mask 过滤），设为 inf 作为 mask loss 负样本
    sky_mask = (depthmaster_depth <= 0) & ~nan_mask
    depthmaster_depth[sky_mask] = np.inf
    # resize 到与 GT 深度相同的尺寸（先将 inf 临时替换为大值，避免 PIL resize 异常）
    mh, mw = depthmaster_depth.shape[:2]
    if (mh, mw) != (H, W):
        inf_mask = np.isinf(depthmaster_depth)
        nan_mask_final = np.isnan(depthmaster_depth)
        # 临时替换特殊值用于 resize
        temp = depthmaster_depth.copy()
        temp[inf_mask] = 1e6
        temp[nan_mask_final] = 0.0
        temp_pil = Image.fromarray(temp)
        temp_resized = np.array(temp_pil.resize((W, H), Image.NEAREST)).astype(np.float32)
        # resize mask 并恢复特殊值
        inf_mask_resized = np.array(Image.fromarray(inf_mask.astype(np.uint8)).resize((W, H), Image.NEAREST)).astype(bool)
        nan_mask_resized = np.array(Image.fromarray(nan_mask_final.astype(np.uint8)).resize((W, H), Image.NEAREST)).astype(bool)
        temp_resized[inf_mask_resized] = np.inf
        temp_resized[nan_mask_resized] = np.nan
        depthmaster_depth = temp_resized
    return depthmaster_depth


# ---------- 各数据集读取函数 ----------

def load_hypersim(instance: dict, read_image_fn: Callable) -> Optional[Dict]:
    """
    Hypersim 格式
    索引: scene/cam_XX/frame_id  (e.g. ai_001_003/cam_00/000000)
    文件: {parent}/{frame_id}_rgb.png, {frame_id}_depth.npy, {frame_id}_cam.npz
    cam.npz keys: intrinsics (3x3)
    """
    base_path = instance['path']
    parent_dir = base_path.parent
    frame_id = base_path.name

    image = read_image_fn(parent_dir / f"{frame_id}_rgb.png")
    depth = _process_depth(np.load(str(parent_dir / f"{frame_id}_depth.npy")))

    if not np.any(np.isfinite(depth)):
        return None

    cam_data = np.load(str(parent_dir / f"{frame_id}_cam.npz"))
    K = cam_data['intrinsics'].astype(np.float32)
    H, W = image.shape[:2]
    intrinsics = _normalize_intrinsics(K, H, W)

    return {'image': image, 'depth': depth, 'intrinsics': intrinsics}


# ---------- Structured3D 全景图工具函数 ----------

def _xyzcube_fov95(face_w: int) -> np.ndarray:
    """生成 FOV=95° 的 cubemap 6 个面的 xyz 坐标（OpenCV 坐标系约定）。
    
    顺序: [Front, Right, Back, Left, Up, Down]
    
    坐标系约定（OpenCV 标准）:
      - 世界坐标系: x-right, y-down, z-forward（与 OpenCV 相机坐标系一致）
      - 相机坐标系: x-right, y-down, z-forward（OpenCV 标准）
      - Front 面 W2C = I（单位矩阵），即世界坐标 = 相机坐标
      - 所有 6 个面的 W2C 旋转矩阵均为 proper rotation (det=+1)
      - pixel (r, c) -> cam = ((c-cx)/f, (r-cy)/f, 1) -> world = R^T @ cam
    
    W2C 旋转矩阵（y-down 世界坐标系）:
      Front:  [[1,0,0],[0,1,0],[0,0,1]]       看向 z+
      Right:  [[0,0,-1],[0,1,0],[1,0,0]]      看向 x+
      Back:   [[-1,0,0],[0,1,0],[0,0,-1]]     看向 z-
      Left:   [[0,0,1],[0,1,0],[-1,0,0]]      看向 x-
      Up:     [[1,0,0],[0,0,1],[0,-1,0]]      看向 y-（天空）— 绕 x 轴旋转 -90°
      Down:   [[1,0,0],[0,0,-1],[0,1,0]]      看向 y+（地面）— 绕 x 轴旋转 +90°
    """
    out = np.zeros((face_w, face_w * 6, 3), np.float32)
    rng = np.linspace(-0.5, 0.5, num=face_w, dtype=np.float32)
    half_fov_rad = np.deg2rad(95.0 / 2)
    s = np.tan(half_fov_rad)
    
    # OpenCV cam: pixel(r,c) -> cam = (rng[c]*2s, rng[r]*2s, 1)
    # 归一化到主轴=0.5: (rng[c]*s, rng[r]*s, 0.5)
    # world = R^T @ cam
    
    # Front (z=+0.5): R=I, world = (rng[c]*s, rng[r]*s, 0.5)
    for r in range(face_w):
        for c in range(face_w):
            out[r, 0*face_w+c] = [rng[c]*s, rng[r]*s, 0.5]
    
    # Right (x=+0.5): R^T=[[0,0,1],[0,1,0],[-1,0,0]]
    # world = (0.5, rng[r]*s, -rng[c]*s)
    for r in range(face_w):
        for c in range(face_w):
            out[r, 1*face_w+c] = [0.5, rng[r]*s, -rng[c]*s]
    
    # Back (z=-0.5): R^T=[[-1,0,0],[0,1,0],[0,0,-1]]
    # world = (-rng[c]*s, rng[r]*s, -0.5)
    for r in range(face_w):
        for c in range(face_w):
            out[r, 2*face_w+c] = [-rng[c]*s, rng[r]*s, -0.5]
    
    # Left (x=-0.5): R^T=[[0,0,-1],[0,1,0],[1,0,0]]
    # world = (-0.5, rng[r]*s, rng[c]*s)
    for r in range(face_w):
        for c in range(face_w):
            out[r, 3*face_w+c] = [-0.5, rng[r]*s, rng[c]*s]
    
    # Up (y=-0.5, 天空): W2C=Rx(-90°)=[[1,0,0],[0,0,1],[0,-1,0]], C2W=[[1,0,0],[0,0,-1],[0,1,0]]
    # world = C2W @ cam = (rng[c]*s, -0.5, rng[r]*s)
    for r in range(face_w):
        for c in range(face_w):
            out[r, 4*face_w+c] = [rng[c]*s, -0.5, rng[r]*s]
    
    # Down (y=+0.5, 地面): W2C=Rx(+90°)=[[1,0,0],[0,0,-1],[0,1,0]], C2W=[[1,0,0],[0,0,1],[0,-1,0]]
    # world = C2W @ cam = (rng[c]*s, 0.5, -rng[r]*s)
    for r in range(face_w):
        for c in range(face_w):
            out[r, 5*face_w+c] = [rng[c]*s, 0.5, -rng[r]*s]
    
    return out


def _xyz2uv(xyz: np.ndarray) -> np.ndarray:
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(x, z)
    c = np.sqrt(x**2 + z**2)
    v = np.arctan2(y, c)
    return np.concatenate([u, v], axis=-1)


def _uv2coor(uv: np.ndarray, h: int, w: int) -> np.ndarray:
    u, v = np.split(uv, 2, axis=-1)
    coor_x = (u / (2 * np.pi) + 0.5) * w - 0.5
    coor_y = (-v / np.pi + 0.5) * h - 0.5
    return np.concatenate([coor_x, coor_y], axis=-1)


# ---------- 缓存的 e2c remap 映射表 ----------

@lru_cache(maxsize=12)
def _get_e2p_remap_maps(u_deg: float, v_deg: float, face_w: int, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """计算并缓存 ERP → 透视投影的 cv2.remap 映射表。
    
    映射表只依赖于视角参数和图像尺寸，与图像内容无关，
    因此可以安全地缓存复用，避免每次调用都重新计算。
    
    Args:
        u_deg: 水平旋转角度
        v_deg: 垂直旋转角度
        face_w: 输出面的边长
        h: ERP 图像高度
        w: ERP 图像宽度
    
    Returns:
        map_x, map_y: cv2.remap 所需的 float32 映射表，shape=(face_w, face_w)
    """
    h_fov = v_fov = np.deg2rad(95.0)
    
    # 生成透视视角的 xyz 坐标
    out = np.ones((face_w, face_w, 3), np.float32)
    x_max = np.tan(h_fov / 2)
    y_max = np.tan(v_fov / 2)
    x_rng = np.linspace(-x_max, x_max, num=face_w, dtype=np.float32)
    # y_rng 从 +y_max 到 -y_max，使得：
    #   r=0（图像顶部）→ y=+y_max → 经 _xyz2uv 得到正仰角 → ERP 顶部（天空）
    #   r=H-1（图像底部）→ y=-y_max → 经 _xyz2uv 得到负仰角 → ERP 底部（地面）
    # 这样采样出来的图像是正视角（顶部对应天空），与透视图先验一致。
    # 注意：这里的 y_rng 是 ERP 球面采样坐标（用于 _xyz2uv），不是世界坐标系的 y。
    # 世界坐标系使用 OpenCV y-down（_xyzcube_fov95 中 r=0 时 y=-0.546，即向上）。
    # 两者方向一致：cubemap r=0 = 天空。
    y_rng = np.linspace(y_max, -y_max, num=face_w, dtype=np.float32)
    out[..., :2] = np.stack(np.meshgrid(x_rng, y_rng), -1)
    
    u = -u_deg * np.pi / 180
    v = v_deg * np.pi / 180
    
    # 旋转矩阵
    Rx = _rotation_matrix(v, [1, 0, 0])
    Ry = _rotation_matrix(u, [0, 1, 0])
    xyz = out.dot(Rx).dot(Ry)
    
    uv = _xyz2uv(xyz)
    coor_xy = _uv2coor(uv, h, w)
    
    # 提取 map_x, map_y 用于 cv2.remap
    map_x = coor_xy[..., 0].astype(np.float32)
    map_y = coor_xy[..., 1].astype(np.float32)
    
    return map_x, map_y


def _e2c_fov95(e_img: np.ndarray, face_w: int = 512, order: int = 1) -> np.ndarray:
    """将 equirectangular 全景图投影为 6 个 cubemap 面 (FOV=95°)。
    
    使用 cv2.remap 替代 scipy.ndimage.map_coordinates 进行采样，
    并缓存采样坐标映射表以避免重复计算，显著提升性能。
    
    Args:
        e_img: (H, W, C) equirectangular 图像
        face_w: cubemap 每个面的边长
        order: 插值阶数，0=最近邻，1=双线性（默认）。深度图应使用 order=0
    
    Returns:
        cubemap: (6, face_w, face_w, C) 6 个面的图像
    """
    if e_img.shape[-1] == 4:
        e_img = e_img[:, :, :3]
    h, w = e_img.shape[:2]
    C = e_img.shape[2] if len(e_img.shape) == 3 else 1
    
    # cv2.remap 插值模式映射
    interp = cv2.INTER_NEAREST if order == 0 else cv2.INTER_LINEAR
    
    cubemap = np.zeros((6, face_w, face_w, C), dtype=e_img.dtype)
    u_degs = [0, 90, 180, 270, 0, 0]
    # v_deg 含义：绕 x 轴旋转，正值向上仰（看向 ERP y+/天空），负值向下俯（看向 ERP y-/地面）
    #   Up 面（看向天空，ERP y+）：v_deg=+90
    #   Down 面（看向地面，ERP y-）：v_deg=-90
    v_degs = [0, 0, 0, 0, 90, -90]
    
    # 采样坐标系直接使用 OpenCV 标准（y 向下）：
    #   pixel(r,c) -> cam = (x_rng[c], y_rng[r], 1)，r=0 时 y 最大（图像顶部对应天空）
    # 采样出来的图像天然就是正视角，无需任何后处理变换。
    # Up/Down 面的 u_deg 为 0，v_deg 分别为 +90/-90，与 _xyzcube_fov95 一致。
    _face_transforms = [
        lambda img: img.copy(),   # Front: 无变换
        lambda img: img.copy(),   # Right: 无变换
        lambda img: img.copy(),   # Back:  无变换
        lambda img: img.copy(),   # Left:  无变换
        lambda img: img.copy(),   # Up:    无变换
        lambda img: img.copy(),   # Down:  无变换
    ]
    
    for i in range(6):
        # 获取缓存的 remap 映射表（首次调用时计算，后续直接复用）
        map_x, map_y = _get_e2p_remap_maps(u_degs[i], v_degs[i], face_w, h, w)
        # cv2.remap 原生支持多通道，无需逐通道处理
        pers_img = cv2.remap(
            e_img, map_x, map_y,
            interpolation=interp,
            borderMode=cv2.BORDER_WRAP  # ERP 水平方向环绕
        )
        if len(pers_img.shape) == 2:
            pers_img = pers_img[..., None]
        cubemap[i] = _face_transforms[i](pers_img)
    
    return cubemap


def _rotation_matrix(rad: float, ax: list) -> np.ndarray:
    ax = np.array(ax, dtype=np.float64)
    ax = ax / np.sqrt((ax**2).sum())
    R = np.diag([np.cos(rad)] * 3)
    R = R + np.outer(ax, ax) * (1.0 - np.cos(rad))
    ax = ax * np.sin(rad)
    R = R + np.array([[0, -ax[2], ax[1]],
                      [ax[2], 0, -ax[0]],
                      [-ax[1], ax[0], 0]])
    return R.astype(np.float32)


def _get_cubemap_K_W2C(fov_deg: float, face_w: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """获取 cubemap 6 个面的内参和 W2C 矩阵（OpenCV 标准坐标系）。
    
    所有 W2C 旋转矩阵均为 proper rotation (det=+1)，
    与 _xyzcube_fov95 的 OpenCV 坐标约定完全一致。
    
    相机坐标系: x-right, y-down, z-forward (OpenCV 标准)
    世界坐标系: x-right, y-down, z-forward (与 OpenCV 相机坐标系一致，Front 面 W2C=I)
    
    Returns:
        K_list: 6 个归一化内参 (3x3)
        W2C_list: 6 个 W2C 矩阵 (4x4)
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
    
    # OpenCV y-down 世界坐标系下的 cubemap W2C 旋转矩阵
    # Front/Right/Back/Left 面不变（它们不涉及 y 轴翻转）
    # Up/Down 面需要修改：在 y-down 世界坐标中，天空是 y-，地面是 y+
    opencv_rotations = [
        # Front: 看向 z+
        np.array([[ 1,  0,  0],
                  [ 0,  1,  0],
                  [ 0,  0,  1]], dtype=np.float32),
        # Right: 看向 x+
        np.array([[ 0,  0, -1],
                  [ 0,  1,  0],
                  [ 1,  0,  0]], dtype=np.float32),
        # Back: 看向 z-
        np.array([[-1,  0,  0],
                  [ 0,  1,  0],
                  [ 0,  0, -1]], dtype=np.float32),
        # Left: 看向 x-
        np.array([[ 0,  0,  1],
                  [ 0,  1,  0],
                  [-1,  0,  0]], dtype=np.float32),
        # Up: 看向 y-（天空，y-down 中 y 负方向）— 绕 x 轴旋转 -90°
        np.array([[ 1,  0,  0],
                  [ 0,  0,  1],
                  [ 0, -1,  0]], dtype=np.float32),
        # Down: 看向 y+（地面，y-down 中 y 正方向）— 绕 x 轴旋转 +90°
        np.array([[ 1,  0,  0],
                  [ 0,  0, -1],
                  [ 0,  1,  0]], dtype=np.float32),
    ]
    
    K_list = []
    W2C_list = []
    for R in opencv_rotations:
        W2C = np.eye(4, dtype=np.float32)
        W2C[:3, :3] = R
        W2C[:3, 3] = 0  # 全景相机在原点
        
        K_list.append(K_normalized.copy())
        W2C_list.append(W2C)
    
    return K_list, W2C_list


def load_structured3d(instance: dict, read_image_fn: Callable) -> Optional[Dict]:
    """Structured3D 全景图格式

    索引: scene_XXXXX/2D_rendering/XXXXXX/panorama/full
    文件:
      - 图像: {path}/rgb_rawlight.png  (equirectangular, 1024x512, RGBA)
      - 深度: {path}/depth.png  (uint16, 单位毫米, 0=无效)

    处理方式:
      1. 读取全景 RGB 和深度图
      2. 使用 e2c 投影成 6 个 FOV=95° 的 cubemap 面
      3. 根据深度 GT 生成全景点云作为监督信号
      4. 返回 cubemap 图像、内参、W2C 矩阵和全景点云 GT
    """
    # instance['path'] 可能是文件路径（如 .../rgb_rawlight.png）或目录路径
    raw_path = Path(instance['path'])
    if raw_path.is_file() or raw_path.suffix:
        base_path = raw_path.parent  # 文件路径 -> 取父目录
    else:
        base_path = raw_path

    # 读取全景 RGB 图像
    pano_image = read_image_fn(base_path / 'rgb_rawlight.png')
    if pano_image.shape[2] == 4:
        pano_image = pano_image[:, :, :3]

    # 读取全景深度图 (uint16, 毫米)
    depth_pil = Image.open(str(base_path / 'depth.png'))
    pano_depth = np.array(depth_pil).astype(np.float32) / 1000.0  # 毫米 -> 米
    pano_depth[pano_depth <= 0] = np.nan  # 无效区域标记为 nan

    if not np.any(np.isfinite(pano_depth)):
        return None

    # 深度最大值裁剪
    valid_depth = pano_depth[np.isfinite(pano_depth)]
    max_depth = 10.0  # Structured3D 室内场景最大 10m
    pano_depth[pano_depth > max_depth] = np.nan

    # 投影成 6 个 cubemap 面 (FOV=95°)
    face_w = 512
    cubemap_rgb = _e2c_fov95(pano_image, face_w=face_w)  # (6, face_w, face_w, 3)
    
    # 深度图也投影成 cubemap（使用最近邻插值避免深度混合）
    pano_depth_for_proj = pano_depth.copy()
    # 将 nan 临时替换为大值用于投影（e2c 不支持 nan）
    nan_val = 1e6
    pano_depth_for_proj[np.isnan(pano_depth_for_proj)] = nan_val
    pano_depth_3ch = np.stack([pano_depth_for_proj] * 3, axis=-1)  # e2c 需要 3 通道
    cubemap_depth_3ch = _e2c_fov95(pano_depth_3ch, face_w=face_w, order=0)  # (6, face_w, face_w, 3) 深度图使用最近邻插值
    cubemap_depth = cubemap_depth_3ch[:, :, :, 0]  # (6, face_w, face_w)
    # 恢复 nan：大于 max_depth 或接近 nan_val 的值都设为 nan（无效区域）
    cubemap_depth[cubemap_depth >= max_depth] = np.nan

    # 边缘飞点过滤：对每个 cubemap 面的深度图进行边缘过滤
    for face_idx in range(6):
        cubemap_depth[face_idx] = _filter_edge_flying_points(cubemap_depth[face_idx])

    # 获取 cubemap 6 个面的内参和 W2C 矩阵
    K_list, W2C_list = _get_cubemap_K_W2C(fov_deg=95.0, face_w=face_w)
    
    # 从 cubemap 面的 range depth 生成 GT 点云（世界坐标系），
    # 同时将 range depth 转换为 z-buffer depth 用于训练
    cubemap_points_world = []  # 存储每个面的世界坐标点云
    cubemap_z_depth = np.empty_like(cubemap_depth)  # z-buffer depth
    for face_idx in range(6):
        face_range_depth = cubemap_depth[face_idx]  # (face_w, face_w), range depth
        face_K = K_list[face_idx]             # (3, 3) 归一化内参
        face_W2C = W2C_list[face_idx]         # (4, 4)
        
        h_face, w_face = face_range_depth.shape
        u = np.linspace(0, 1, w_face, dtype=np.float32)
        v = np.linspace(0, 1, h_face, dtype=np.float32)
        u, v = np.meshgrid(u, v)
        
        fx, fy = face_K[0, 0], face_K[1, 1]
        cx, cy = face_K[0, 2], face_K[1, 2]
        
        # 相机坐标系下的方向向量（未归一化）
        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy
        z_cam = np.ones_like(x_cam)
        
        ray_dir = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)
        ray_len = np.linalg.norm(ray_dir, axis=-1, keepdims=True)  # (H, W, 1)
        ray_dir_normalized = ray_dir / ray_len  # 归一化方向向量
        
        # 无效区域统一为 nan（包括超过 max_depth 和边缘飞点）
        invalid_mask = ~np.isfinite(face_range_depth)  # nan 区域
        
        # 相机坐标系下的点云 = 归一化方向 * range_depth
        points_cam = ray_dir_normalized * face_range_depth[..., None]  # (H, W, 3)
        points_cam[invalid_mask] = np.nan  # 无效区域设为 nan，不参与任何 loss
        
        # range depth → z-buffer depth: z_depth = range_depth / ray_len
        # 等价于 points_cam[..., 2]，即相机坐标系下的 z 分量
        face_z_depth = face_range_depth / ray_len[..., 0]
        face_z_depth[invalid_mask] = np.nan  # 无效区域设为 nan，不参与任何 loss
        cubemap_z_depth[face_idx] = face_z_depth
        
        # 相机坐标系 -> 世界坐标系: C2W = W2C^{-1}
        C2W = np.linalg.inv(face_W2C)  # (4, 4)
        R_c2w = C2W[:3, :3]  # (3, 3)
        t_c2w = C2W[:3, 3]   # (3,)
        
        # 变换到世界坐标系
        points_world = np.einsum('hwc,dc->hwd', points_cam, R_c2w) + t_c2w[None, None, :]
        points_world[invalid_mask] = np.nan  # 无效区域设为 nan
        cubemap_points_world.append(points_world)
    
    cubemap_points_world = np.stack(cubemap_points_world)  # (6, face_w, face_w, 3)

    return {
        'camera_type': 'Panorama',
        'cubemap_images': cubemap_rgb,           # (6, face_w, face_w, 3) uint8
        'cubemap_depths': cubemap_z_depth,        # (6, face_w, face_w) float32, z-buffer depth
        'cubemap_intrinsics': np.stack(K_list),   # (6, 3, 3) float32
        'cubemap_W2C': np.stack(W2C_list),        # (6, 4, 4) float32
        'cubemap_points_world': cubemap_points_world,  # (6, face_w, face_w, 3) float32, 世界坐标系GT点云
    }


# ---------- GPU E2C 模式的 ERP 读取函数 ----------
# 这些函数只读取原始 ERP 图像和深度，不做 e2c 投影。
# 投影将在 GPU 上由 E2C_GPU 模块完成。

def load_structured3d_erp(instance: dict, read_image_fn: Callable) -> Optional[Dict]:
    """Structured3D GPU E2C 模式：只返回原始 ERP 图像和深度，不做投影。"""
    raw_path = Path(instance['path'])
    if raw_path.is_file() or raw_path.suffix:
        base_path = raw_path.parent
    else:
        base_path = raw_path

    pano_image = read_image_fn(base_path / 'rgb_rawlight.png')
    if pano_image.shape[2] == 4:
        pano_image = pano_image[:, :, :3]

    depth_pil = Image.open(str(base_path / 'depth.png'))
    pano_depth = np.array(depth_pil).astype(np.float32) / 1000.0  # 毫米 -> 米
    pano_depth[pano_depth <= 0] = np.nan

    if not np.any(np.isfinite(pano_depth)):
        return None

    max_depth = 10.0
    pano_depth[pano_depth > max_depth] = np.nan

    return {
        'camera_type': 'Panorama_ERP',
        'erp_image': pano_image,        # (H, W, 3) uint8
        'erp_depth': pano_depth,        # (H, W) float32, range depth, nan=无效
        'max_depth': max_depth,
    }


# ---------- Reader registry ----------

DATASET_READERS = {
    # Perspective dataset
    'hypersim': load_hypersim,
    # Panoramic dataset (cubemap projection performed inside the reader, CPU)
    'structured3d': load_structured3d,
    # Panoramic dataset (raw ERP only; cubemap projection deferred to GPU via E2C_GPU)
    'structured3d_erp': load_structured3d_erp,
}


def get_reader(data_format: str) -> Callable:
    """Return the reader function corresponding to ``data_format``."""
    if data_format not in DATASET_READERS:
        raise ValueError(
            f"Unknown data_format: '{data_format}'. "
            f"Supported formats: {list(DATASET_READERS.keys())}"
        )
    return DATASET_READERS[data_format]
