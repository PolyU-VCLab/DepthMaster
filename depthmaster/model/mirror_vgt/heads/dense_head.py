"""
Original source: HunyuanWorld-Mirror/src/models/heads/dense_head.py
Migrated into DepthMaster (mirror_vgt subpackage).

与 MirrorLite 完全一致的 DPTHead 实现，支持：
- enable_depth_mask: 三段式 activation（attr + conf + mask）
- gradient_checkpoint: 梯度检查点
- FP32 保持: output_conv2 保持 FP32
"""
# inspired by https://github.com/DepthAnything/Depth-Anything-V2
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..layers.mlp import MlpFP32
from ..utils.grid import create_uv_grid, position_grid_to_embed


class DPTHead(nn.Module):
    """
    DPT Head for dense prediction tasks.

    Args:
        dim_in (int): Number of input feature channels.
        patch_size (int): Patch size used by the backbone, default is 14.
        output_dim (int): Number of output channels, default is 4.
        activation (str): Activation function type string, default is "inv_log+expp1".
        features (int): Number of channels used in intermediate feature representations, default is 256.
        out_channels (List[int]): Number of channels for each intermediate multi-scale feature.
        pos_embed (bool): Whether to add positional encoding to the features, default is True.
        down_ratio (int): Downsampling ratio of the output predictions, default is 1.
        is_gsdpt (bool): Whether this is a GS-DPT head.
        enable_depth_mask (bool): Whether to enable depth mask output (三段式 activation).
        gradient_checkpoint (bool): Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "inv_log+expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        pos_embed: bool = True,
        down_ratio: int = 1,
        is_gsdpt: bool = False,
        enable_depth_mask: bool = False,
        gradient_checkpoint: bool = False,
    ) -> None:
        super(DPTHead, self).__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.is_gsdpt = is_gsdpt
        self.enable_depth_mask = enable_depth_mask
        self.gradient_checkpoint = gradient_checkpoint

        self.norm = nn.LayerNorm(dim_in)
        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList([nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels])
        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )
        self.scratch = _make_scratch(out_channels, features, expand=False)

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32

        if self.is_gsdpt:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            conv2_in_channels = head_features_1 // 2
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )
            self.input_merger = nn.Sequential(
                nn.Conv2d(3, conv2_in_channels, 7, 1, 3),
                nn.ReLU()
                )
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
            )
            conv2_in_channels = head_features_1 // 2
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )

    def to(self, *args, **kwargs):
        self.norm = self.norm.to(*args, **kwargs)
        self.projects = self.projects.to(*args, **kwargs)
        self.resize_layers = self.resize_layers.to(*args, **kwargs)
        if self.is_gsdpt:
            self.input_merger = self.input_merger.to(*args, **kwargs)
        for key in ('layer1_rn', 'layer2_rn', 'layer3_rn', 'layer4_rn',
                    'refinenet1', 'refinenet2', 'refinenet3', 'refinenet4',
                    'output_conv1'):
            if not hasattr(self.scratch, key):
                continue
            setattr(self.scratch, key, getattr(self.scratch, key).to(*args, **kwargs))

        # keep output_conv2 in FP32
        args, kwargs = MlpFP32.map_to_args_to_float(args, kwargs)
        self.scratch.output_conv2 = self.scratch.output_conv2.to(*args, **kwargs)

        return self

    def forward(
        self,
        token_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
        border_padding: int = 0,  # 新增：边缘padding参数
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional frame chunking for memory efficiency.

        Args:
            token_list: List of token tensors from transformer, each [B, S, N, C]
            images: Input images [B, S, 3, H, W], range [0, 1]
            patch_start_idx: Starting index of patch tokens
            frames_chunk_size: Number of frames per chunk
            border_padding: 上采样前对边界做 replicate padding 的像素数，
                            用于缓解边界（尤其四角）的异常值问题。
                            0 表示不做额外 padding（与原始行为一致），推荐推理时设为 2~4。

        Returns:
            Tuple of predictions and confidence (and optionally depth_mask)
        """
        B, S, _, H, W = images.shape

        # Process all frames together if chunk size not specified or large enough
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(token_list, images, patch_start_idx, border_padding=border_padding)

        assert frames_chunk_size > 0

        # Process frames in chunks
        preds_chunks = []
        conf_chunks = []
        gs_chunks = []
        depth_mask_chunks = []

        for frame_start in range(0, S, frames_chunk_size):
            frame_end = min(frame_start + frames_chunk_size, S)

            if self.is_gsdpt:
                if self.enable_depth_mask:
                    gs, preds, conf, depth_mask = self._forward_impl(
                        token_list, images, patch_start_idx, frame_start, frame_end, border_padding=border_padding
                    )
                    gs_chunks.append(gs)
                    preds_chunks.append(preds)
                    conf_chunks.append(conf)
                    depth_mask_chunks.append(depth_mask)
                else:
                    gs, preds, conf = self._forward_impl(
                        token_list, images, patch_start_idx, frame_start, frame_end, border_padding=border_padding
                    )
                    gs_chunks.append(gs)
                    preds_chunks.append(preds)
                    conf_chunks.append(conf)
            else:
                if self.enable_depth_mask:
                    preds, conf, depth_mask = self._forward_impl(
                        token_list, images, patch_start_idx, frame_start, frame_end, border_padding=border_padding
                    )
                    preds_chunks.append(preds)
                    conf_chunks.append(conf)
                    depth_mask_chunks.append(depth_mask)
                else:
                    preds, conf = self._forward_impl(
                        token_list, images, patch_start_idx, frame_start, frame_end, border_padding=border_padding
                    )
                    preds_chunks.append(preds)
                    conf_chunks.append(conf)

        # Concatenate chunks along frame dimension
        if self.is_gsdpt:
            if self.enable_depth_mask:
                return (
                    torch.cat(gs_chunks, dim=1),
                    torch.cat(preds_chunks, dim=1),
                    torch.cat(conf_chunks, dim=1),
                    torch.cat(depth_mask_chunks, dim=1),
                )
            return torch.cat(gs_chunks, dim=1), torch.cat(preds_chunks, dim=1), torch.cat(conf_chunks, dim=1)
        else:
            if self.enable_depth_mask:
                return torch.cat(preds_chunks, dim=1), torch.cat(conf_chunks, dim=1), torch.cat(depth_mask_chunks, dim=1)
            else:
                return torch.cat(preds_chunks, dim=1), torch.cat(conf_chunks, dim=1)

    def _forward_impl(
        self,
        token_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frame_start: int = None,
        frame_end: int = None,
        border_padding: int = 0,  # 新增：边缘padding参数
    ) -> torch.Tensor:
        """
        Core forward implementation for DPT head.
        """
        # Slice frames if chunking
        if frame_start is not None and frame_end is not None:
            images = images[:, frame_start:frame_end].contiguous()

        B, S, _, H, W = images.shape
        ph = H // self.patch_size  # patch height
        pw = W // self.patch_size  # patch width

        # Extract and project multi-level features
        feats = []
        for proj, resize, tokens in zip(self.projects, self.resize_layers, token_list):
            # Extract patch tokens
            patch_tokens = tokens[:, :, patch_start_idx:]
            if frame_start is not None and frame_end is not None:
                patch_tokens = patch_tokens[:, frame_start:frame_end]

            # Reshape to [B*S, N_patches, C]
            patch_tokens = patch_tokens.reshape(B * S, -1, patch_tokens.shape[-1])
            patch_tokens = self.norm(patch_tokens)

            # Convert to 2D feature map [B*S, C, ph, pw]
            feat = patch_tokens.permute(0, 2, 1).reshape(B * S, patch_tokens.shape[-1], ph, pw)
            feat = proj(feat)

            if self.pos_embed:
                feat = self._apply_pos_embed(feat, W, H)
            feat = resize(feat)
            feats.append(feat)

        # Fuse multi-level features
        fused = checkpoint(self.scratch_forward, feats, use_reentrant=False) if self.gradient_checkpoint else self.scratch_forward(feats)
        _interpolate_fn = lambda t: custom_interpolate(
            t,
            size=(
                int(ph * self.patch_size / self.down_ratio),
                int(pw * self.patch_size / self.down_ratio)
            ),
            mode="bilinear",
            align_corners=True,
            border_padding=border_padding,  # 传递border_padding参数
        )
        fused = checkpoint(_interpolate_fn, fused, use_reentrant=False) if self.gradient_checkpoint else _interpolate_fn(fused)

        # Apply positional embedding after upsampling
        if self.pos_embed:
            fused = self._apply_pos_embed(fused, W, H)

        # Generate predictions and confidence
        if self.is_gsdpt:
            # GSDPT: output features, predictions, and confidence
            out = self.scratch.output_conv2(fused.float().contiguous())
            if self.enable_depth_mask:
                preds, conf, depth_mask = self.activate_head(out, activation=self.activation)
            else:
                preds, conf = self.activate_head(out, activation=self.activation)
            preds = preds.reshape(B, S, *preds.shape[1:])
            conf = conf.reshape(B, S, *conf.shape[1:])

            # Merge direct image features
            img_flat = images.reshape(B * S, -1, H, W)
            img_feat = self.input_merger(img_flat)
            fused = fused + img_feat
            fused = fused.reshape(B, S, *fused.shape[1:]).float().contiguous()
            if self.enable_depth_mask:
                depth_mask = depth_mask.reshape(B, S, *depth_mask.shape[1:])
                return fused, preds, conf, depth_mask
            return fused, preds, conf
        else:
            # Standard: output predictions and confidence
            out = self.scratch.output_conv2(fused.float().contiguous())
            if self.enable_depth_mask:
                preds, conf, depth_mask = self.activate_head(out, activation=self.activation)
                preds = preds.reshape(B, S, *preds.shape[1:])
                conf = conf.reshape(B, S, *conf.shape[1:])
                depth_mask = depth_mask.reshape(B, S, *depth_mask.shape[1:])
                return preds, conf, depth_mask
            else:
                preds, conf = self.activate_head(out, activation=self.activation)
                preds = preds.reshape(B, S, *preds.shape[1:])
                conf = conf.reshape(B, S, *conf.shape[1:])
                return preds, conf

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """Apply positional embedding to tensor x."""
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed.to(x.dtype)

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through the fusion blocks."""
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out

    def activate_head(self, out_head: torch.Tensor, activation: str = "inv_log+expp1") -> Tuple[torch.Tensor, ...]:
        """
        Process network output to extract attribute, confidence, and optionally depth mask.

        支持两段式 (attr+conf) 和三段式 (attr+conf+mask) activation。
        """
        # Parse activation string
        if self.enable_depth_mask:
            parts = activation.split("+") if "+" in activation else [activation, "expp1", "linear"]
            act_attr, act_conf, act_depth_mask = parts[0], parts[1], parts[2] if len(parts) > 2 else "linear"

            # (B,C,H,W) -> (B,H,W,C)
            feat = out_head.permute(0, 2, 3, 1)
            attr, conf, depth_mask = feat[..., :-2], feat[..., -2], feat[..., -1]
        else:
            act_attr, act_conf = (activation.split("+") if "+" in activation else (activation, "expp1"))

            # (B,C,H,W) -> (B,H,W,C)
            feat = out_head.permute(0, 2, 3, 1)
            attr, conf = feat[..., :-1], feat[..., -1]

        # Attribute activation mapping
        attr_activations = {
            "norm_exp": lambda x: (x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)) * torch.expm1(x.norm(dim=-1, keepdim=True)),
            "norm": lambda x: x / x.norm(dim=-1, keepdim=True),
            "exp": torch.exp,
            "relu": F.relu,
            "inv_log": self._apply_inverse_log_transform,
            "xy_inv_log": lambda x: torch.cat([
                x[..., :2] * self._apply_inverse_log_transform(x[..., 2:]),
                self._apply_inverse_log_transform(x[..., 2:])
            ], dim=-1),
            "sigmoid": torch.sigmoid,
            "linear": lambda x: x
        }

        if act_attr not in attr_activations:
            raise ValueError(f"Unknown attribute activation: {act_attr}")
        attr_out = attr_activations[act_attr](attr)

        # Confidence activation mapping
        conf_activations = {
            "expp1": lambda c: 1 + c.exp(),
            "expp0": torch.exp,
            "sigmoid": torch.sigmoid
        }
        if act_conf not in conf_activations:
            raise ValueError(f"Unknown confidence activation: {act_conf}")
        conf_out = conf_activations[act_conf](conf)

        if self.enable_depth_mask:
            depth_mask_activations = {
                "sigmoid": torch.sigmoid,
                "linear": lambda x: x,
            }
            if act_depth_mask not in depth_mask_activations:
                raise ValueError(f"Unknown depth mask activation: {act_depth_mask}")
            depth_mask_out = depth_mask_activations[act_depth_mask](depth_mask)
            return attr_out, conf_out, depth_mask_out
        else:
            return attr_out, conf_out

    def _apply_inverse_log_transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse logarithm transform: sign(y) * (exp(|y|) - 1)"""
        return torch.sign(input_tensor) * (torch.expm1(torch.abs(input_tensor)))



################################################################################
# DPT Modules
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module with skip connection."""

    def __init__(self, features, activation, bn, groups=1):
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
    border_padding: int = 0,
) -> torch.Tensor:
    """
    Custom interpolation function to handle large tensors by chunking.

    Avoids INT_MAX overflow issues in nn.functional.interpolate when dealing with
    very large input tensors by splitting them into smaller chunks.

    Args:
        border_padding: 上采样前对边界做 replicate padding 的像素数，
                        用于缓解边界（尤其四角）的异常值问题。
                        0 表示不做额外 padding（与原始行为一致），推荐推理时设为 2~4。

    原理（与 v2 ConvStack 中的 border_padding 一致）：
        1. 上采样前对输入做 replicate padding，让边界像素在插值时有合理的上下文
        2. 计算 padding 在上采样后对应的像素数（= border_padding * scale_factor）
        3. 上采样时目标尺寸相应增大，使得 crop 后恰好得到原始目标尺寸
    """
    # 计算原始目标尺寸
    if size is None:
        target_size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    else:
        target_size = size

    # 计算上采样的 scale factor（用于确定 padding 在输出空间的大小）
    orig_h, orig_w = x.shape[-2], x.shape[-1]

    # 边缘 padding 处理
    if border_padding > 0:
        x = F.pad(x, [border_padding] * 4, mode='replicate')

    # padding 后的输入尺寸
    padded_h, padded_w = x.shape[-2], x.shape[-1]

    if border_padding > 0:
        # 计算 scale factor：原始输入 -> 原始目标
        sf_h = target_size[0] / orig_h
        sf_w = target_size[1] / orig_w
        # padding 在输出空间对应的像素数
        pad_out_h = int(round(border_padding * sf_h))
        pad_out_w = int(round(border_padding * sf_w))
        # 上采样的实际目标尺寸 = 原始目标 + 两侧 padding 对应的输出像素
        interp_size = (target_size[0] + 2 * pad_out_h, target_size[1] + 2 * pad_out_w)
    else:
        interp_size = target_size
        pad_out_h = pad_out_w = 0

    INT_MAX = 1610612736
    input_elements = interp_size[0] * interp_size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=interp_size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
    else:
        x = nn.functional.interpolate(x, size=interp_size, mode=mode, align_corners=align_corners)

    # crop 掉 padding 对应的输出区域，恢复到原始目标尺寸
    if border_padding > 0:
        x = x[:, :, pad_out_h:-pad_out_h, pad_out_w:-pad_out_w]

    return x.contiguous()
