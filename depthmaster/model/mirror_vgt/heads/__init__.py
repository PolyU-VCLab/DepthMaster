"""Dense / camera heads for Mirror's VGT (migrated from HunyuanWorld-Mirror)."""
from .dense_head import DPTHead, custom_interpolate
from .camera_head import CameraHead

__all__ = ["DPTHead", "custom_interpolate", "CameraHead"]
