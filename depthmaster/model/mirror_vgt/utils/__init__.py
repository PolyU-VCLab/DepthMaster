"""Utility modules for Mirror's VGT (migrated from HunyuanWorld-Mirror)."""
from .grid import create_uv_grid, position_grid_to_embed
from .rotation import quat_to_rotmat, rotmat_to_quat
from .camera_utils import (
    camera_params_to_vector,
    extrinsics_to_vector,
    vector_to_extrinsics,
    vector_to_camera_matrices,
)
from .priors import normalize_poses, normalize_depth

__all__ = [
    "create_uv_grid",
    "position_grid_to_embed",
    "quat_to_rotmat",
    "rotmat_to_quat",
    "camera_params_to_vector",
    "extrinsics_to_vector",
    "vector_to_extrinsics",
    "vector_to_camera_matrices",
    "normalize_poses",
    "normalize_depth",
]
