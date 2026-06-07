"""Mirror VGT subpackage, migrated from the HunyuanWorld-Mirror project.

Provides the core modules required by the DepthMaster model:

* ``VisualGeometryTransformer``      - the alternative full-size backbone.
* ``VisualGeometryTransformerLite``  - the DA3-style lightweight backbone used by DepthMaster.
* ``DPTHead``                        - dense prediction head.
* ``normalize_poses``                - pose normalization helper.
* ``extrinsics_to_vector``           - extrinsics-to-vector conversion.
"""
from .visual_transformer import VisualGeometryTransformer, expand_and_flatten_special_tokens
from .visual_transformer_lite import VisualGeometryTransformerLite
from .heads.dense_head import DPTHead, custom_interpolate
from .heads.camera_head import CameraHead
from .utils.priors import normalize_poses, normalize_depth
from .utils.camera_utils import extrinsics_to_vector, vector_to_extrinsics, vector_to_camera_matrices

__all__ = [
    "VisualGeometryTransformer",
    "VisualGeometryTransformerLite",
    "expand_and_flatten_special_tokens",
    "DPTHead",
    "CameraHead",
    "custom_interpolate",
    "normalize_poses",
    "normalize_depth",
    "extrinsics_to_vector",
    "vector_to_extrinsics",
    "vector_to_camera_matrices",
]
