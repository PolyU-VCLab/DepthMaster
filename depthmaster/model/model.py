"""DepthMaster model: a unified monocular depth estimator built on top of the
HunyuanWorld-Mirror VisualGeometryTransformerLite backbone.

使用 MirrorLite 的 VGT Lite 作为 backbone，DPTHead 作为 Points/Normal Head。
不使用 CameraHead 和 DepthHead。
在第 alt_start 层用 cam_token 替换 DINOv2 的 cls_token（与 mirrorlite_stage1_da3style 一致）。
不使用 register tokens。
ScaleHead 使用 cam_token（位置 0）进行 metric_scale 预测。
"""
import warnings
from typing import *
from numbers import Number
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import utils3d
import utils3d.torch as utils3d_torch
from huggingface_hub import hf_hub_download

# Mirror VGT Lite 模型组件
from .mirror_vgt import (
    VisualGeometryTransformerLite,
    DPTHead as _MirrorDPTHead,
    normalize_poses,
    normalize_depth,
    extrinsics_to_vector,
)
from ..utils.geometry_torch import recover_focal_shift
from .mirror_vgt.layers.block import Block


class _DPTHeadNoConf(_MirrorDPTHead):
    """
    继承 Mirror 的 DPTHead，去掉 conf 通道。

    支持四种模式（通过 head_mode 参数控制）：
    - "pts": output_dim=4, 输出 xyz(3维) + mask_prob(1维, sigmoid)
      返回 (xyz, mask_prob)
    - "xyz_only": output_dim=3, 输出纯 xyz(3维)
      返回 (xyz, None) —— 用于独立 mask head 模式下的 pts_head
    - "mask_only": output_dim=1, 输出 mask_prob(1维, sigmoid)
      返回 (mask_prob, None) —— 用于独立 mask head
    - "norm": output_dim=3, 输出 normals(3维)
      返回 (normals, None)

    关键：重写 activate_head 和 _forward_impl，不再分离 conf 通道。
    """

    def __init__(self, *args, head_mode: str = "norm", activation_mode: str = "inv_log", **kwargs):
        """
        Args:
            head_mode: "pts" / "xyz_only" / "mask_only" / "norm"，控制 activate_head 的行为
            activation_mode: "inv_log" (default, sign-preserving exp(|x|)-1) 或 "exp"
                （DepthMaster v2 风格: z=exp(z_raw), xy=xy_raw*z，强制 xy/z 尺度耦合，仅 head_mode="pts"/"xyz_only" 生效）
                注意 "exp" 模式下 z 严格为正，不支持负 z 输出，仅适合纯透视场景。
        """
        super().__init__(*args, **kwargs)
        self.head_mode = head_mode
        self.activation_mode = activation_mode
        if activation_mode not in ("inv_log", "exp"):
            raise ValueError(f"Unknown activation_mode: {activation_mode}")

    def activate_head(self, out_head: torch.Tensor, activation: str = None):
        """
        处理网络输出，不分离 conf 通道。
        """
        # (B, C, H, W) -> (B, H, W, C)
        feat = out_head.permute(0, 2, 3, 1)

        if self.head_mode == "pts":
            # 4通道: xyz(3) + mask_prob(1, sigmoid)
            xyz = feat[..., :3]
            mask_logits = feat[..., 3]
            xyz_out = self._apply_pts_activation(xyz)
            mask_prob = torch.sigmoid(mask_logits)
            return xyz_out, mask_prob
        elif self.head_mode == "xyz_only":
            # 3通道: 纯 xyz
            xyz_out = self._apply_pts_activation(feat)
            return xyz_out, None
        elif self.head_mode == "mask_only":
            # 1通道: mask_prob (sigmoid)，对外直接暴露概率
            mask_logits = feat[..., 0]
            mask_prob = torch.sigmoid(mask_logits)
            return mask_prob, None
        elif self.head_mode == "norm":
            # 3通道: 全部归一化为单位法线
            normals = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return normals, None
        else:
            raise ValueError(f"Unknown head_mode: {self.head_mode}")

    def _apply_pts_activation(self, xyz: torch.Tensor) -> torch.Tensor:
        """对 xyz 输出施加激活函数。

        - "inv_log" (default): 三通道独立施加 sign(x)*(exp(|x|)-1)，支持负值
        - "exp" (DepthMaster v2 风格): z=exp(z_raw), xy=xy_raw*z，强制 xy/z 尺度耦合
          z 严格为正，xy 跟 z 尺度联动；不适用于全景场景。
        """
        if self.activation_mode == "exp":
            # 防止 exp 溢出：clamp z_raw
            xy_raw = xyz[..., :2]
            z_raw = xyz[..., 2:3].clamp(-3.0, 8.0)
            z = z_raw.exp()
            xy = xy_raw * z
            return torch.cat([xy, z], dim=-1)
        # "inv_log"
        return self._apply_inverse_log_transform(xyz)

    def _forward_impl(
        self,
        token_list,
        images,
        patch_start_idx,
        frame_start=None,
        frame_end=None,
        border_padding: int = 0,
    ):
        """
        重写 _forward_impl。

        - head_mode="pts": 返回 (xyz, mask_logits) — 两个 tensor
        - head_mode="norm": 返回 (normals, None)
        """
        # 切片帧
        if frame_start is not None and frame_end is not None:
            images = images[:, frame_start:frame_end].contiguous()

        B, S, _, H, W = images.shape
        ph = H // self.patch_size
        pw = W // self.patch_size

        # 提取并投影多层特征
        feats = []
        for proj, resize, tokens in zip(self.projects, self.resize_layers, token_list):
            patch_tokens = tokens[:, :, patch_start_idx:]
            if frame_start is not None and frame_end is not None:
                patch_tokens = patch_tokens[:, frame_start:frame_end]

            patch_tokens = patch_tokens.reshape(B * S, -1, patch_tokens.shape[-1])
            patch_tokens = self.norm(patch_tokens)

            feat = patch_tokens.permute(0, 2, 1).reshape(B * S, patch_tokens.shape[-1], ph, pw)
            feat = proj(feat)

            if self.pos_embed:
                feat = self._apply_pos_embed(feat, W, H)
            feat = resize(feat)
            feats.append(feat)

        # 融合多层特征
        from torch.utils.checkpoint import checkpoint as ckpt_fn
        from .mirror_vgt.heads.dense_head import custom_interpolate
        fused = ckpt_fn(self.scratch_forward, feats, use_reentrant=False) if self.gradient_checkpoint else self.scratch_forward(feats)
        fused = custom_interpolate(
            fused,
            size=(
                int(ph * self.patch_size / self.down_ratio),
                int(pw * self.patch_size / self.down_ratio)
            ),
            mode="bilinear",
            align_corners=True,
            border_padding=border_padding,
        )

        if self.pos_embed:
            fused = self._apply_pos_embed(fused, W, H)

        # 生成预测
        out = self.scratch.output_conv2(fused.float().contiguous())
        result_a, result_b = self.activate_head(out)

        # reshape
        result_a = result_a.reshape(B, S, *result_a.shape[1:])
        if result_b is not None:
            result_b = result_b.reshape(B, S, *result_b.shape[1:])

        if self.head_mode == "pts":
            # 返回 (xyz, mask_logits)
            return result_a, result_b
        else:
            # 返回 (normals, None)
            return result_a, None

    def forward(
        self,
        token_list,
        images,
        patch_start_idx,
        frames_chunk_size: int = 8,
        border_padding: int = 0,
    ):
        """
        重写 forward，简化 frame chunking 逻辑以适配无 conf 的返回格式。
        """
        B, S, _, H, W = images.shape

        # 不分块，直接调用 _forward_impl
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(token_list, images, patch_start_idx, border_padding=border_padding)

        # 分块处理
        result_a_chunks = []
        result_b_chunks = []

        for frame_start in range(0, S, frames_chunk_size):
            frame_end = min(frame_start + frames_chunk_size, S)
            a, b = self._forward_impl(
                token_list, images, patch_start_idx, frame_start, frame_end, border_padding=border_padding
            )
            result_a_chunks.append(a)
            if b is not None:
                result_b_chunks.append(b)

        result_a = torch.cat(result_a_chunks, dim=1)
        result_b = torch.cat(result_b_chunks, dim=1) if result_b_chunks else None
        return result_a, result_b
from .modules import MLP


class DepthMasterModel(nn.Module):
    """
DepthMaster: a unified perspective + panoramic monocular depth estimator built on a
MirrorLite (DA3-style) Visual Geometry Transformer backbone.

    包含：
    - visual_geometry_transformer: VisualGeometryTransformerLite (DA3 风格 backbone)
    - pts_head: DPTHead (点云预测，output_dim=5: 3维xyz + 1维conf + 1维mask_logits，与训练端完全一致)
    - norm_head: DPTHead (法线预测)
    - scale_head: MLP (cam_token → metric_scale)

    不使用 CameraHead 和 DepthHead。
    在第 alt_start 层用 cam_token 替换 DINOv2 的 cls_token（与 mirrorlite_stage1_da3style 一致）。
    不使用 register tokens（num_register_tokens=0）。
    cam_token 重新初始化为 [1, 1, 1, C]（不区分 ref/src），从头学习，用于 ScaleHead 做 metric_scale 预测。
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        model_size="large",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=0,  # DA3 风格：去掉 register tokens
        enable_cond=False,
        enable_pts=True,
        enable_norm=True,
        fixed_patch_embed=False,
        sampling_strategy="uniform",
        condition_strategy=["token", "pow3r", "token"],
        rope_base=100.0,
        normalized_rope=True,
        rope_normalize_coords="separate",
        rope_shift_coords=None,
        rope_jitter_coords=None,
        rope_rescale_coords=None,
        # MirrorLite 特有参数
        alt_start=8,
        intermediate_idxs=None,
        # DepthMaster 特有参数
        remap_output: str = 'linear',
        num_tokens_range: List[int] = [1200, 3600],
        # Scale Head 参数
        scale_head: Optional[Dict[str, Any]] = None,
        enable_scale: bool = True,
        # Panorama 模式下 scale head 使用的 cls token 聚合方式："mean"=6 面平均，"front"=仅前视图
        panorama_scale_pooling: Literal["mean", "front"] = "mean",
        # 是否启用 conf 通道（pts3d_conf, normals_conf）
        enable_conf: bool = True,
        # 是否启用独立 mask head（与 v2 风格一致：pts_head 不再预测 mask，新增独立 mask_head 分支）
        # 仅在 enable_conf=False 时生效。enable_conf=True 时此选项被忽略。
        enable_separate_mask_head: bool = False,
        # pts_head xyz 通道激活模式: "inv_log" (默认) 或 "exp" (DepthMaster v2 风格, 强制 xy/z 耦合)
        # 仅在 enable_conf=False 时生效（即 pts_head 是 _DPTHeadNoConf 时）
        pts_activation_mode: str = "inv_log",
        **deprecated_kwargs
    ):
        super().__init__()
        if deprecated_kwargs:
            warnings.warn(f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}")

        # 中间层索引
        if intermediate_idxs is None:
            intermediate_idxs = [11, 15, 19, 23]

        # 模型尺寸配置（与 mirrorlite_stage1_da3style 一致，去掉 register tokens）
        self.model_size = model_size
        if model_size == "large":
            embed_dim = 1024
            depth = 24
            num_heads = 16
            mlp_ratio = 4.0
            num_register_tokens = 0
        elif model_size == "base":
            embed_dim = 768
            depth = 12
            num_heads = 12
            mlp_ratio = 4.0
            num_register_tokens = 0
        elif model_size == "small":
            embed_dim = 384
            depth = 12
            num_heads = 6
            mlp_ratio = 4.0
            num_register_tokens = 0
        elif model_size is None:
            pass
        print(
f"[DepthMaster MirrorLite] model_size: {model_size}, embed_dim: {embed_dim}, "
            f"depth: {depth}, num_heads: {num_heads}, alt_start: {alt_start}, "
            f"intermediate_idxs: {intermediate_idxs}"
        )

        # 存储配置
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_register_tokens = num_register_tokens

        self.enable_pts = enable_pts
        self.enable_norm = enable_norm
        self.enable_cond = enable_cond
        self.alt_start = alt_start
        self.intermediate_idxs = intermediate_idxs

        self.remap_output = remap_output
        self.num_tokens_range = num_tokens_range
        self.enable_scale = enable_scale
        if panorama_scale_pooling not in ("mean", "front"):
            raise ValueError(f"Unknown panorama_scale_pooling: {panorama_scale_pooling}")
        self.panorama_scale_pooling = panorama_scale_pooling
        self.enable_conf = enable_conf
        # 独立 mask head 仅在 enable_conf=False 时生效
        self.enable_separate_mask_head = bool(enable_separate_mask_head) and (not enable_conf)
        self.pts_activation_mode = pts_activation_mode
        self._scale_head_config = scale_head

        # Visual Geometry Transformer Lite（核心 backbone，与 MirrorLite 完全一致）
        self.visual_geometry_transformer = VisualGeometryTransformerLite(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_register_tokens=num_register_tokens,
            block_fn=Block,
            alt_start=alt_start,
            intermediate_idxs=intermediate_idxs,
            normalized_rope=normalized_rope,
            rope_normalize_coords=rope_normalize_coords,
            rope_shift_coords=rope_shift_coords,
            rope_jitter_coords=rope_jitter_coords,
            rope_rescale_coords=rope_rescale_coords,
            enable_cond=enable_cond,
            sampling_strategy=sampling_strategy,
            fixed_patch_embed=fixed_patch_embed,
            condition_strategy=condition_strategy,
        )

        # 初始化预测头（与 MirrorLite 完全一致 + Scale Head）
        self._init_heads(embed_dim, patch_size)

    def _init_heads(self, dim, patch_size):
        """初始化预测头：PointsHead + NormalHead + ScaleHead。"""

        if self.enable_pts:
            if self.enable_conf:
                # output_dim=5, enable_depth_mask=True: 与 mirrorlite_stage1_da3style 完全一致
                # 三段式拆分: attr(3维 xyz) + conf(1维 置信度) + mask(1维 天空mask logits)
                # activation="inv_log+expp1+linear": xyz 用 inv_log, conf 用 expp1, mask 用 linear
                self.pts_head = _MirrorDPTHead(
                    dim_in=2 * dim,
                    output_dim=5,
                    patch_size=patch_size,
                    activation="inv_log+expp1+linear",
                    enable_depth_mask=True,
                )
            else:
                if self.enable_separate_mask_head:
                    # 独立 mask head 模式: pts_head 输出纯 xyz(3通道)，mask 由独立 head 预测
                    self.pts_head = _DPTHeadNoConf(
                        dim_in=2 * dim,
                        output_dim=3,
                        patch_size=patch_size,
                        activation="inv_log",
                        enable_depth_mask=False,
                        head_mode="xyz_only",
                        activation_mode=self.pts_activation_mode,
                    )
                else:
                    # 无 conf 模式: output_dim=4 (xyz_3 + mask_logits_1)
                    # 使用 _DPTHeadNoConf(head_mode="pts")
                    self.pts_head = _DPTHeadNoConf(
                        dim_in=2 * dim,
                        output_dim=4,
                        patch_size=patch_size,
                        activation="inv_log+linear",
                        enable_depth_mask=False,
                        head_mode="pts",
                        activation_mode=self.pts_activation_mode,
                    )

        # 独立 mask head（仅在 enable_separate_mask_head=True 时初始化）
        # 用 _DPTHeadNoConf(head_mode="mask_only")，与 pts_head 完全解耦的独立 DPT trunk
        if self.enable_separate_mask_head:
            self.mask_head = _DPTHeadNoConf(
                dim_in=2 * dim,
                output_dim=1,
                patch_size=patch_size,
                activation="linear",
                enable_depth_mask=False,
                head_mode="mask_only",
            )

        if self.enable_norm:
            if self.enable_conf:
                self.norm_head = _MirrorDPTHead(
                    dim_in=2 * dim,
                    output_dim=4,
                    patch_size=patch_size,
                    activation="norm+expp1",
                )
            else:
                # 无 conf 模式: output_dim=3, 全部通道归一化为法线方向
                # 使用 _DPTHeadNoConf(head_mode="norm")
                self.norm_head = _DPTHeadNoConf(
                    dim_in=2 * dim,
                    output_dim=3,
                    patch_size=patch_size,
                    activation="norm",
                    enable_depth_mask=False,
                    head_mode="norm",
                )
        # Scale Head: MLP(cam_token -> metric_scale)
        # cam_token 重新初始化为 [1, 1, 1, C]（不区分 ref/src），从头学习
        if self.enable_scale:
            dim_in = 2 * dim
            if self._scale_head_config is not None:
                self.scale_head = MLP(**self._scale_head_config)
            else:
                # 默认: [2*embed_dim, 2*embed_dim, 2*embed_dim, 1]
                self.scale_head = MLP(dims=[dim_in, dim_in, dim_in, 1])

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], model_kwargs: Optional[Dict[str, Any]] = None, **hf_kwargs) -> 'DepthMasterModel':
        """
        从 checkpoint 加载模型。
        """
        if Path(pretrained_model_name_or_path).exists():
            checkpoint_path = pretrained_model_name_or_path
        else:
            checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                repo_type="model",
                filename="model.pt",
                **hf_kwargs
            )
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        model_config = checkpoint['model_config']
        if model_kwargs is not None:
            model_config.update(model_kwargs)
        model = cls(**model_config)
        model.load_state_dict(checkpoint['model'], strict=False)

        return model

    def init_weights(self, mirrorlite_pretrained: str = None, dinov2_pretrained: str = None,
                     skip_pts_head: bool = False):
        """
        初始化权重。

        支持两种初始化方式：
        1. mirrorlite_pretrained: 从 MirrorLite 完整模型 checkpoint 加载全部权重
        2. dinov2_pretrained: 仅从 DINOv2 预训练权重初始化 backbone

        Args:
            mirrorlite_pretrained: MirrorLite 完整模型 checkpoint 路径（.pt/.safetensors）
            dinov2_pretrained: DINOv2 预训练权重路径（HuggingFace 格式目录或文件）
            skip_pts_head: 若为 True，则跳过 pts_head.* 权重加载（pts_head 全部从头随机初始化），
                适用于切换 pts_head 输出激活语义（如 inv_log → exp）的实验场景。
        """
        if mirrorlite_pretrained is not None:
            self._load_mirrorlite_weights(mirrorlite_pretrained, skip_pts_head=skip_pts_head)
        elif dinov2_pretrained is not None:
            self._load_dinov2_weights(dinov2_pretrained)
        else:
            print("[DepthMaster] No pretrained weights provided, using random initialization")

    def _load_mirrorlite_weights(self, pretrained_path: str, skip_pts_head: bool = False):
        """
        从 MirrorLite 完整模型 checkpoint 加载全部权重。
        由于模型结构与 WorldMirrorLite 完全一致，可以直接加载。

        Args:
            pretrained_path: MirrorLite checkpoint 路径
        """
        import os
        print(f"[DepthMaster] Loading from MirrorLite pretrained weights: {pretrained_path}")

        if os.path.isdir(pretrained_path):
            safetensors_path = os.path.join(pretrained_path, "model.safetensors")
            bin_path = os.path.join(pretrained_path, "pytorch_model.bin")
            if os.path.isfile(safetensors_path):
                pretrained_path = safetensors_path
            elif os.path.isfile(bin_path):
                pretrained_path = bin_path
            else:
                raise FileNotFoundError(f"在目录 {pretrained_path} 中未找到 model.safetensors 或 pytorch_model.bin")

        if pretrained_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

        # ========== Strip 'model.' 前缀 ==========
        # PyTorch Lightning 的 checkpoint 中，所有参数 key 都带有 'model.' 前缀
        # （因为 LightningModule 中模型是 self.model = ...）
        # 需要去掉这个前缀才能与我们的模型 key 匹配
        stripped_state_dict = {}
        has_model_prefix = any(k.startswith('model.') for k in state_dict.keys())
        if has_model_prefix:
            print("[DepthMaster] Detected 'model.' prefix (PyTorch Lightning format); stripping it")
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    stripped_state_dict[k[len('model.'):]] = v
                else:
                    stripped_state_dict[k] = v
            state_dict = stripped_state_dict

        # ========== 特殊处理 cam_token ==========
        # 不再用 cam_token 替换 cls_token，cam_token 参数虽然存在但不再被使用
        # scale_head 直接使用 DINOv2 cls_token 经过所有层后的输出
        # 删除预训练 cam_token 以避免 unexpected key 警告
        backbone_prefix = 'visual_geometry_transformer.'
        
        cam_token_key = backbone_prefix + 'cam_token'
        if cam_token_key in state_dict:
            print(f"[DepthMaster] cam_token: dropping pretrained weight {state_dict[cam_token_key].shape} (no longer used; scale_head consumes DINOv2 cls_token directly)")
            del state_dict[cam_token_key]
        
        # reg_token: mirrorlite_stage1_da3style 中没有 reg_token（num_register_tokens=0）
        # 如果预训练中意外存在，直接忽略
        reg_token_key = backbone_prefix + 'reg_token'
        if reg_token_key in state_dict:
            print(f"[DepthMaster] reg_token: present in pretrained weights but unused by the model, ignoring")
            del state_dict[reg_token_key]

        # ========== 删除 scale_head 预训练权重 ==========
        # scale_head 是新增的 MLP，预训练中没有对应权重（预训练用的是 cam_head）
        # 即使预训练中有 scale_head 相关 key，也应该从头学习
        scale_head_keys = [k for k in state_dict if k.startswith('scale_head.')]
        for k in scale_head_keys:
            del state_dict[k]
        if scale_head_keys:
            print(f"[DepthMaster] Dropping scale_head pretrained weights ({len(scale_head_keys)} keys), training from scratch")

        # ========== 可选: 跳过 pts_head 权重加载 ==========
        # 用于切换 pts_head 输出激活语义（如 inv_log → exp）的实验场景：
        # 直接从头随机初始化 pts_head，避免预训练 conv 输出语义错位带来的前期 loss 爆涨。
        if skip_pts_head:
            pts_head_keys = [k for k in state_dict if k.startswith('pts_head.')]
            for k in pts_head_keys:
                del state_dict[k]
            if pts_head_keys:
                print(f"[DepthMaster] skip_pts_head=True: dropped pts_head pretrained weights ({len(pts_head_keys)} keys); pts_head will be re-initialized from scratch")

        # ========== 加载权重 ==========
        # 如果 enable_conf=False，需要裁剪 pts_head 和 norm_head 最后一层 Conv 的 conf 通道
        # 加载权重 ==========
        # 如果 enable_conf=False，需要裁剪 pts_head 和 norm_head 最后一层 Conv 的 conf 通道
        if not self.enable_conf:
            # pts_head: 预训练 output_dim=5 (xyz_3 + conf_1 + mask_1)
            #   - enable_separate_mask_head=False → 目标 output_dim=4 (xyz_3 + mask_1)，保留 channels [0,1,2,4]
            #   - enable_separate_mask_head=True  → 目标 output_dim=3 (xyz_3 only)，保留 channels [0,1,2]
            pts_conv_weight_key = 'pts_head.scratch.output_conv2.2.weight'
            pts_conv_bias_key = 'pts_head.scratch.output_conv2.2.bias'
            if pts_conv_weight_key in state_dict:
                w = state_dict[pts_conv_weight_key]
                if w.shape[0] == 5:
                    if self.enable_separate_mask_head:
                        state_dict[pts_conv_weight_key] = w[:3]
                    else:
                        state_dict[pts_conv_weight_key] = torch.cat([w[:3], w[4:5]], dim=0)
                        print(f"[DepthMaster] pts_head Conv weight: {w.shape} -> {state_dict[pts_conv_weight_key].shape}")
            if pts_conv_bias_key in state_dict:
                b = state_dict[pts_conv_bias_key]
                if b.shape[0] == 5:
                    if self.enable_separate_mask_head:
                        state_dict[pts_conv_bias_key] = b[:3]
                    else:
                        state_dict[pts_conv_bias_key] = torch.cat([b[:3], b[4:5]], dim=0)
                        print(f"[DepthMaster] pts_head Conv bias: {b.shape} -> {state_dict[pts_conv_bias_key].shape}")

            # 独立 mask head 模式: 把预训练 pts_head 的整套 backbone trunk（除最后输出层）复制到 mask_head，
            # 让 mask_head 有一个合理的初始化起点；最后一层 Conv (output_conv2.2) 不复制（输出维度不匹配，从头学习）。
            if self.enable_separate_mask_head:
                pts_to_mask_keys = []
                for k in list(state_dict.keys()):
                    if k.startswith('pts_head.') and not k.startswith('pts_head.scratch.output_conv2.2.'):
                        new_k = 'mask_head.' + k[len('pts_head.'):]
                        # 仅在目标模型确实有这个 key 时才复制（避免 unexpected key 警告）
                        if new_k in self.state_dict() and new_k not in state_dict:
                            state_dict[new_k] = state_dict[k].clone()
                            pts_to_mask_keys.append(new_k)
                if pts_to_mask_keys:
                    print(f"[DepthMaster] enable_separate_mask_head: 从 pts_head 复制 {len(pts_to_mask_keys)} 个权重到 mask_head（trunk 共享初始化，输出层从头学习）")            # 最后一层 Conv 的 weight shape: [4, in_ch, kH, kW] → 去掉 index=3 (conf)
            # 最后一层 Conv 的 bias shape: [4] → 去掉 index=3 (conf)
            norm_conv_weight_key = 'norm_head.scratch.output_conv2.2.weight'
            norm_conv_bias_key = 'norm_head.scratch.output_conv2.2.bias'
            if norm_conv_weight_key in state_dict:
                w = state_dict[norm_conv_weight_key]
                if w.shape[0] == 4:
                    state_dict[norm_conv_weight_key] = w[:3]
                    print(f"[DepthMaster] norm_head Conv weight: {w.shape} → {state_dict[norm_conv_weight_key].shape} (去掉 conf 通道)")
            if norm_conv_bias_key in state_dict:
                b = state_dict[norm_conv_bias_key]
                if b.shape[0] == 4:
                    state_dict[norm_conv_bias_key] = b[:3]
                    print(f"[DepthMaster] norm_head Conv bias: {b.shape} → {state_dict[norm_conv_bias_key].shape} (去掉 conf 通道)")

        my_state_dict = self.state_dict()
        loaded_keys = []
        missing_keys = []
        unexpected_keys = []
        shape_mismatch_keys = []

        for key, value in state_dict.items():
            if key in my_state_dict:
                if value.shape == my_state_dict[key].shape:
                    my_state_dict[key] = value
                    loaded_keys.append(key)
                else:
                    shape_mismatch_keys.append(f"{key} (pretrained: {value.shape} vs model: {my_state_dict[key].shape})")
            else:
                unexpected_keys.append(key)

        for key in my_state_dict:
            if key not in state_dict:
                missing_keys.append(key)

        self.load_state_dict(my_state_dict, strict=False)

        print(f"[DepthMaster] MirrorLite 权重加载完成:")
        print(f"  成功加载: {len(loaded_keys)}")
        print(f"  缺失 (保持随机初始化): {len(missing_keys)}")
        print(f"  多余 (忽略): {len(unexpected_keys)}")
        print(f"  形状不匹配 (跳过): {len(shape_mismatch_keys)}")
        if missing_keys:
            print(f"  Missing keys (前10个): {missing_keys[:10]}")
        if unexpected_keys:
            print(f"  Unexpected keys (前10个): {unexpected_keys[:10]}")
        if shape_mismatch_keys:
            print(f"  Shape mismatch keys: {shape_mismatch_keys[:10]}")

    def _load_dinov2_weights(self, dinov2_path: str):
        """
        从 DINOv2 预训练权重初始化 backbone。

        Args:
            dinov2_path: DINOv2 预训练权重路径（HuggingFace 格式目录或 safetensors 文件）
        """
        import os
        print(f"[DepthMaster] 从 DINOv2 预训练权重初始化 backbone: {dinov2_path}")

        if os.path.isdir(dinov2_path):
            safetensors_path = os.path.join(dinov2_path, "model.safetensors")
            bin_path = os.path.join(dinov2_path, "pytorch_model.bin")
            if os.path.isfile(safetensors_path):
                dinov2_path = safetensors_path
            elif os.path.isfile(bin_path):
                dinov2_path = bin_path
            else:
                raise FileNotFoundError(f"在目录 {dinov2_path} 中未找到 model.safetensors 或 pytorch_model.bin")

        if dinov2_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(dinov2_path)
        else:
            state_dict = torch.load(dinov2_path, map_location='cpu', weights_only=True)

        self.visual_geometry_transformer.load_dinov2_weights(state_dict)

    def enable_gradient_checkpointing(self, enable: bool = True):
        """启用/禁用梯度检查点。启用时节省显存但 backward 更慢，禁用时 backward 更快但显存占用更大。"""
        self.visual_geometry_transformer.set_gradient_checkpointing(enable)

    def forward(
        self,
        image: torch.Tensor,
        num_tokens: Union[int, torch.LongTensor] = 0,
        camera_type: str = "pinhole",
        return_intermediate_features: bool = False,
        **extra_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        Args:
            image: 透视模式 (B, 3, H, W)，全景模式 (B, 6, 3, H, W)
            num_tokens: 透视模式下的 token 数量（用于 resize），全景模式下忽略
            camera_type: "pinhole" 或 "Panorama"
            return_intermediate_features: 是否返回中间层特征
            **extra_kwargs: 全景模式下可传入 W2C 和 intrinsics 用于条件嵌入

        Returns:
            预测结果字典
        """
        if camera_type == "Panorama":
            # 全景模式：输入 (B, 6, 3, H, W) → 视为 6 帧多视图
            B, V, C, img_h, img_w = image.shape
            # 将 6 个面视为 6 帧序列
            images = image  # [B, 6, 3, H, W]
            
            # 提取 priors（相机位姿和光线条件）
            priors = None
            cond_flags = [0, 0, 0]  # [pose, depth, ray]
            
            if 'W2C' in extra_kwargs and 'intrinsics' in extra_kwargs:
                W2C = extra_kwargs['W2C']
                intrinsics = extra_kwargs['intrinsics']
                priors = self._extract_priors(W2C, intrinsics, img_h, img_w)
                cond_flags = [1, 0, 1]  # 启用 pose 和 ray 条件，不使用 depth 条件
        else:
            # 透视模式：输入 (B, 3, H, W) → 视为 1 帧
            B, C, img_h, img_w = image.shape
            V = 1

            # 根据 num_tokens 计算目标分辨率并 resize
            if num_tokens > 0:
                aspect_ratio = img_w / img_h
                base_h = (num_tokens / aspect_ratio) ** 0.5
                base_w = (num_tokens * aspect_ratio) ** 0.5
                if isinstance(base_h, torch.Tensor):
                    base_h, base_w = base_h.round().long().item(), base_w.round().long().item()
                else:
                    base_h, base_w = round(base_h), round(base_w)
                target_h = base_h * self.patch_size
                target_w = base_w * self.patch_size
                if target_h != img_h or target_w != img_w:
                    image = F.interpolate(image, (target_h, target_w), mode='bilinear', align_corners=False, antialias=True)

            images = image.unsqueeze(1)  # [B, 1, 3, H', W']
            priors = None
            cond_flags = [0, 0, 0]  # 透视图不使用条件嵌入

        _, S, _, H, W = images.shape

        # 通过 VGT Lite backbone
        if priors is not None:
            token_list, patch_start_idx = self.visual_geometry_transformer(
                images, priors, cond_flags=cond_flags, enable_bf16=False
            )
        else:
            token_list, patch_start_idx = self.visual_geometry_transformer(
                images, enable_bf16=False
            )

        # 预测头
        return_dict = {}

        # Points head: output_dim=5, enable_depth_mask=True
        # 三段式返回 (pts, pts_conf, pts_mask_logits)
        # pts: 3维 xyz, pts_conf: 1维置信度（训练时可选用，推理时忽略）, pts_mask_logits: 1维天空mask
        if self.enable_pts:
            if self.enable_conf:
                pts, pts_conf, pts_mask_logits = self.pts_head(
                    token_list, images=images, patch_start_idx=patch_start_idx,
                )
                return_dict["pts3d"] = pts
                return_dict["pts3d_conf"] = pts_conf
                return_dict["pts3d_mask_logits"] = pts_mask_logits
                return_dict["pts3d_mask"] = pts_mask_logits.sigmoid()
            else:
                if self.enable_separate_mask_head:
                    # 独立 mask head 模式: pts_head 输出纯 xyz (3 通道)
                    pts, _ = self.pts_head(
                        token_list, images=images, patch_start_idx=patch_start_idx,
                    )
                    return_dict["pts3d"] = pts
                    # mask 由独立的 mask_head 预测（与 pts_head 完全解耦的独立 DPT trunk）
                    pts_mask, _ = self.mask_head(
                        token_list, images=images, patch_start_idx=patch_start_idx,
                    )
                    return_dict["pts3d_mask"] = pts_mask
                else:
                    # 无 conf 模式: pts_head 内部已对 mask 通道做 sigmoid，返回 (xyz, mask_prob)
                    # Return probabilities directly (no logits field).
                    pts, pts_mask = self.pts_head(
                        token_list, images=images, patch_start_idx=patch_start_idx,
                    )
                    return_dict["pts3d"] = pts
                    return_dict["pts3d_mask"] = pts_mask

        # Normal head
        if self.enable_norm:
            if self.enable_conf:
                normals, norm_conf = self.norm_head(
                    token_list, images=images, patch_start_idx=patch_start_idx,
                )
                return_dict["normals"] = normals
                return_dict["normals_conf"] = norm_conf
            else:
                # 无 conf 模式: norm_head 使用 _DPTHeadNoConf，返回 (normals, None)
                normals, _ = self.norm_head(
                    token_list, images=images, patch_start_idx=patch_start_idx,
                )
                return_dict["normals"] = normals

        # Scale Head: 从 cls_token（位置 0）预测 metric_scale（follow v2 的做法）
        # 不再用 cam_token 替换 cls_token，直接使用 DINOv2 cls_token 经过所有层后的输出
        if self.enable_scale:
            last_layer_tokens = token_list[-1]  # (B, S, N, 2*embed_dim)
            if camera_type == "Panorama":
                cls_tokens = last_layer_tokens[:, :, 0, :]  # (B, V, 2*embed_dim)
                if self.panorama_scale_pooling == "front":
                    # 全景模式：仅使用 cubemap 前视图（face 0）的 cls_token 预测 metric_scale
                    cls_token = cls_tokens[:, 0, :]  # (B, 2*embed_dim)
                else:
                    # 全景模式：对 6 个面的 cls_token 做 mean pooling（默认，与 v2 一致）
                    cls_token = cls_tokens.mean(dim=1)  # (B, 2*embed_dim)
            else:
                # 透视模式：直接取第 0 帧的 cls_token
                cls_token = last_layer_tokens[:, 0, 0, :]  # (B, 2*embed_dim)
            # follow v2: 直接传入 scale_head，不做 .float() 转换
            metric_scale = self.scale_head(cls_token)
            metric_scale = metric_scale.squeeze(-1).exp()  # (B,)
            return_dict["metric_scale"] = metric_scale

        # 如果是透视模式，squeeze 掉 S 维度
        if camera_type != "Panorama" and V == 1:
            for key in list(return_dict.keys()):
                val = return_dict[key]
                if isinstance(val, torch.Tensor) and val.dim() >= 2 and val.shape[1] == 1:
                    return_dict[key] = val.squeeze(1)

        # 如果需要 resize 回原始分辨率
        if camera_type != "Panorama" and num_tokens > 0:
            for key in ['pts3d', 'normals', 'depth']:
                if key in return_dict:
                    val = return_dict[key]
                    if val.shape[-3] != img_h or val.shape[-2] != img_w:
                        # val: [B, H, W, C] 或 [B, H, W, 1]
                        if val.dim() == 4:
                            val = val.permute(0, 3, 1, 2)  # [B, C, H, W]
                            val = F.interpolate(val, (img_h, img_w), mode='bilinear', align_corners=False)
                            val = val.permute(0, 2, 3, 1)  # [B, H, W, C]
                            return_dict[key] = val
            # mask 相关字段: [B, H, W]（无 channel 维度），需要单独处理 resize
            # 注意: enable_conf=False 时不再有 'pts3d_mask_logits' 字段
            for key in ['pts3d_mask', 'pts3d_mask_logits', 'pts3d_conf']:
                if key in return_dict:
                    val = return_dict[key]
                    if val.dim() == 3 and (val.shape[-2] != img_h or val.shape[-1] != img_w):
                        val = val.unsqueeze(1)  # [B, 1, H, W]
                        val = F.interpolate(val, (img_h, img_w), mode='bilinear', align_corners=False)
                        val = val.squeeze(1)  # [B, H, W]
                        return_dict[key] = val

        if return_intermediate_features:
            # 将 token_list 中的 patch token 切出来并 reshape 成 (B*V, C, ph, pw) 空间 feature map,
            # 语义与 DPTHead 内部使用的特征一致，可直接供 CCL loss 使用。
            # token_list[l]: (B, S, N, 2*embed_dim), N = patch_start_idx + ph*pw
            ph = H // self.patch_size
            pw = W // self.patch_size
            feat_maps = []
            for tokens in token_list:
                # 去掉 cam/reg token，只保留 patch token
                patch_tokens = tokens[:, :, patch_start_idx:, :]          # (B, V, ph*pw, C_tok)
                C_tok = patch_tokens.shape[-1]
                # (B, V, ph*pw, C_tok) -> (B*V, C_tok, ph, pw)
                feat = patch_tokens.reshape(B * S, ph * pw, C_tok)
                feat = feat.permute(0, 2, 1).contiguous().reshape(B * S, C_tok, ph, pw)
                feat_maps.append(feat)
            return_dict['intermediate_features'] = feat_maps

        # Field name remap: unify all output keys for downstream training code.
        # pts3d -> points, normals -> normal, pts3d_mask (sigmoid) -> mask
        if 'pts3d' in return_dict:
            return_dict['points'] = return_dict['pts3d']
        if 'normals' in return_dict:
            return_dict['normal'] = return_dict['normals']
        # pts3d_mask 已经是 sigmoid 后的概率（无 conf 模式由 _DPTHeadNoConf 内部完成；
        # enable_conf 模式由 forward 中 .sigmoid() 显式完成），可直接喂给 mask_bce_loss。
        if 'pts3d_mask' in return_dict:
            return_dict['mask'] = return_dict['pts3d_mask']
        elif 'pts3d_mask_logits' in return_dict:
            return_dict['mask'] = return_dict['pts3d_mask_logits'].sigmoid()

        return return_dict

    def _extract_priors(self, W2C: torch.Tensor, intrinsics: torch.Tensor, img_h: int, img_w: int):
        """
        从 DepthMaster batch 数据构造 VGT 所需的 priors。

        Args:
            W2C: 世界到相机变换矩阵 (B, V, 4, 4)
            intrinsics: 相机内参 (B, V, 3, 3)
            img_h: 图像高度
            img_w: 图像宽度

        Returns:
            (depths, rays, poses) 元组
        """
        B, V = W2C.shape[:2]

        # W2C -> C2W
        R_w2c = W2C[:, :, :3, :3]  # (B, V, 3, 3)
        t_w2c = W2C[:, :, :3, 3]   # (B, V, 3)
        R_c2w = R_w2c.transpose(-1, -2)  # (B, V, 3, 3)
        t_c2w = -torch.einsum('bvij,bvj->bvi', R_c2w, t_w2c)  # (B, V, 3)

        # 构造 C2W extrinsics: (B, V, 3, 4)
        c2w_extrinsics = torch.cat([R_c2w, t_c2w.unsqueeze(-1)], dim=-1)

        # 注意：不调用 normalize_poses！
        # normalize_poses 是为多视图重建设计的，假设相机分布在不同位置。
        # 对于 cubemap，所有 6 个面的相机都在同一位置（原点），translation 全为 0，
        # normalize_poses 会把 translation 退化为全 0.5，完全丢失位置信息。
        # 直接用 extrinsics_to_vector 提取 [t, q]，translation 保持原始值（全 0），
        # pose_embed 至少能从 quaternion 中学到朝向信息。
        cam_params = extrinsics_to_vector(c2w_extrinsics)
        poses = cam_params[:, :, :7]  # (B, V, 7)

        # 从 intrinsics 提取归一化的 ray directions
        fx = intrinsics[:, :, 0, 0] / img_w  # (B, V)
        fy = intrinsics[:, :, 1, 1] / img_h
        cx = intrinsics[:, :, 0, 2] / img_w
        cy = intrinsics[:, :, 1, 2] / img_h
        rays = torch.stack([fx, fy, cx, cy], dim=-1)  # (B, V, 4)

        # depths 设为 None（不使用 depth condition）
        depths = None

        return (depths, rays, poses)

    @torch.inference_mode()
    def infer(
        self,
        image: torch.Tensor,
        num_tokens: int = None,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: bool = True,
        fov_x: Optional[Union[Number, torch.Tensor]] = None,
        use_fp16: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        透视图推理接口。

        Args:
            image: 输入图像 (B, 3, H, W) 或 (3, H, W)
            num_tokens: ViT token 数量，建议 1200~2500
            resolution_level: 分辨率级别 0~9
            force_projection: 是否使用深度图重新计算点云
            apply_mask: 是否应用掩码
            fov_x: 水平 FoV（度），None 则自动推断
            use_fp16: 是否使用混合精度

        Returns:
            {'points', 'depth', 'intrinsics', 'mask', 'normal'}
        """
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        image = image.to(dtype=self.dtype, device=self.device)

        original_height, original_width = image.shape[-2:]
        aspect_ratio = original_width / original_height

        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_fp16 and self.dtype != torch.float16):
            output = self.forward(image, num_tokens=num_tokens)

        # 从 MirrorLite 输出格式提取结果
        # Prefer the 'mask' field (sigmoid probabilities); compatible with enable_conf=True/False.
        points = output.get('pts3d', None)
        normal = output.get('normals', None)
        mask_prob = output.get('mask', output.get('pts3d_mask', None))  # 概率 ∈ [0, 1]
        depth = output.get('depth', None)
        metric_scale = output.get('metric_scale', None)

        # 转为 float32 进行后处理
        points, normal, depth, metric_scale, fov_x = map(
            lambda x: x.float() if isinstance(x, torch.Tensor) else x,
            [points, normal, depth, metric_scale, fov_x]
        )

        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            # Convert mask: probability > 0.5 -> foreground.
            if mask_prob is not None:
                mp = mask_prob.float()
                # 兼容形状 [B, H, W] 或 [B, H, W, 1]
                if mp.dim() == 4 and mp.shape[-1] == 1:
                    mp = mp.squeeze(-1)
                mask_binary = mp > 0.5
            else:
                mask_binary = None

            # 处理 points → depth → intrinsics
            if points is not None:
                # points: [B, H, W, 3]
                if fov_x is None:
                    focal, shift = recover_focal_shift(points, mask_binary)
                else:
                    focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2))
                    if focal.ndim == 0:
                        focal = focal[None].expand(points.shape[0])
                    _, shift = recover_focal_shift(points, mask_binary, focal=focal)
                fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
                intrinsics = utils3d.pt.intrinsics_from_focal_center(fx, fy, torch.tensor(0.5, device=points.device, dtype=points.dtype), torch.tensor(0.5, device=points.device, dtype=points.dtype))
                points[..., 2] += shift[..., None, None]
                if mask_binary is not None:
                    mask_binary &= points[..., 2] > 0
                depth_from_pts = points[..., 2].clone()
            else:
                depth_from_pts, intrinsics = None, None

            if force_projection and depth_from_pts is not None:
                points = utils3d.pt.depth_map_to_point_map(depth_from_pts, intrinsics=intrinsics)

            # Apply metric scale（与 v2 一致：将 affine-invariant 预测转换为 metric 尺度）
            if metric_scale is not None:
                if points is not None:
                    points = points * metric_scale[:, None, None, None]
                if depth_from_pts is not None:
                    depth_from_pts = depth_from_pts * metric_scale[:, None, None]

            if apply_mask and mask_binary is not None:
                points = torch.where(mask_binary[..., None], points, torch.inf) if points is not None else None
                depth_from_pts = torch.where(mask_binary, depth_from_pts, torch.inf) if depth_from_pts is not None else None
                normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal)) if normal is not None else None

        return_dict = {
            'points': points,
            'intrinsics': intrinsics,
            'depth': depth_from_pts,
            'mask': mask_binary,
            'normal': normal,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        if omit_batch_dim:
            return_dict = {k: v.squeeze(0) for k, v in return_dict.items()}

        return return_dict

    @torch.inference_mode()
    def infer_panorama(
        self,
        cubemap_faces: torch.Tensor,
        W2C: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        use_fp16: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        全景图推理接口。

        Args:
            cubemap_faces: cubemap 6 个面的图像 (B, 6, 3, H, W) 或 (6, 3, H, W)
            W2C: 世界到相机变换矩阵 (B, 6, 4, 4) 或 (6, 4, 4)，None 则不使用相机条件
            intrinsics: 相机内参 (B, 6, 3, 3) 或 (6, 3, 3)，None 则不使用相机条件
            use_fp16: 是否使用混合精度

        Returns:
            预测结果字典
        """
        if cubemap_faces.dim() == 4:
            omit_batch_dim = True
            cubemap_faces = cubemap_faces.unsqueeze(0)
            if W2C is not None:
                W2C = W2C.unsqueeze(0)
            if intrinsics is not None:
                intrinsics = intrinsics.unsqueeze(0)
        else:
            omit_batch_dim = False
        
        cubemap_faces = cubemap_faces.to(dtype=self.dtype, device=self.device)
        if W2C is not None:
            W2C = W2C.to(dtype=self.dtype, device=self.device)
        if intrinsics is not None:
            intrinsics = intrinsics.to(dtype=self.dtype, device=self.device)

        B, V, C, img_h, img_w = cubemap_faces.shape

        # 准备 forward 参数
        kwargs = {}
        if W2C is not None and intrinsics is not None:
            kwargs['W2C'] = W2C
            kwargs['intrinsics'] = intrinsics

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_fp16 and self.dtype != torch.float16):
            output = self.forward(cubemap_faces, num_tokens=0, camera_type="Panorama", **kwargs)

        # 从输出格式提取结果
        # Prefer the 'mask' field (sigmoid probabilities); compatible with enable_conf=True/False.
        points = output.get('pts3d', None)
        normal = output.get('normals', None)
        mask_prob = output.get('mask', output.get('pts3d_mask', None))  # 概率 ∈ [0, 1]
        depth = output.get('depth', None)

        # 转为 float32 进行后处理
        points, normal, depth = map(
            lambda x: x.float() if isinstance(x, torch.Tensor) else x,
            [points, normal, depth]
        )

        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            # Convert mask: probability > 0.5 -> foreground.
            if mask_prob is not None:
                mp = mask_prob.float()
                if mp.dim() == 5 and mp.shape[-1] == 1:
                    mp = mp.squeeze(-1)
                mask_binary = mp > 0.5
            else:
                mask_binary = None

            # 应用 mask
            if mask_binary is not None:
                points = torch.where(mask_binary[..., None], points, torch.full_like(points, float('inf'))) if points is not None else None
                normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal)) if normal is not None else None
                depth = torch.where(mask_binary, depth, torch.full_like(depth, float('inf'))) if depth is not None else None

        return_dict = {
            'points': points,
            'normal': normal,
            'mask': mask_binary,
            'depth': depth,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        if omit_batch_dim:
            return_dict = {k: v.squeeze(0) for k, v in return_dict.items()}

        return return_dict
