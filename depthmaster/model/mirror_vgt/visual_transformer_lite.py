"""
Original source: HunyuanWorld-Mirror/src/models/models/visual_transformer_lite.py
Migrated into DepthMaster (mirror_vgt subpackage).

VisualGeometryTransformerLite — DA3 风格的轻量版 VGT

核心设计：
- 在 DINOv2 ViT-L/14 的 24 层 Block 内部注入交替注意力（Local/Global）
- 前 alt_start 层保持标准 DINOv2 行为（无 QK Norm、无 RoPE）
- 第 alt_start 层起启用 QK Norm + RoPE + 交替注意力
- Block 参数共享：同一个 Block 既做 Local 又做 Global（DA3 方案 A）
- 条件注入保持 Mirror 做法，注入时机提前到第 alt_start 层
"""

import logging
import math
from functools import partial
from typing import Tuple, List, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .layers import PatchEmbed, PatchEmbed_Mlp
from .layers.block import Block

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class VisualGeometryTransformerLite(nn.Module):
    """
    DA3 风格的轻量版 Visual Geometry Transformer。

    与原始 VisualGeometryTransformer 的区别：
    - 去掉独立的 24 层 frame_blocks + 24 层 global_blocks
    - 在 DINOv2 的 24 层 Block 内部注入交替注意力
    - 参数量从 ~745M 降低到 ~405M

    Args:
        img_size: 输入图像尺寸
        patch_size: Patch 尺寸
        embed_dim: 嵌入维度
        depth: Block 层数
        num_heads: 注意力头数
        mlp_ratio: MLP 隐藏层维度比
        num_register_tokens: Register token 数量
        alt_start: 交替注意力开始层（前 alt_start 层保持标准 DINOv2 行为）
        intermediate_idxs: 中间层特征提取的层索引
        qk_norm: 是否启用 QK Norm
        rope_base: RoPE 基频
        normalized_rope: 是否使用 NormalizedRoPE
        rope_normalize_coords: RoPE 坐标归一化方式
        enable_cond: 是否启用条件注入
        condition_strategy: 条件注入策略
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=0,  # DA3 风格：去掉 register tokens
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        alt_start=8,
        intermediate_idxs: List[int] = [11, 15, 19, 23],
        qk_norm=True,
        rope_base=100.0,
        normalized_rope=True,
        rope_normalize_coords="separate",
        rope_shift_coords=None,
        rope_jitter_coords=None,
        rope_rescale_coords=None,
        init_values=1.0,
        enable_cond=False,
        sampling_strategy="uniform",
        fixed_patch_embed=False,
        condition_strategy=["token", "pow3r", "token"],
    ):
        super().__init__()

        # 存储配置参数
        self.enable_cond = enable_cond
        self.sampling_strategy = sampling_strategy
        self.cond_methods = condition_strategy
        self.intermediate_idxs = intermediate_idxs
        self.depth = depth
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.alt_start = alt_start
        self.num_register_tokens = num_register_tokens  # 设为 0，与 DA3 对齐

        # 初始化 Patch Embedding（DINOv2 风格）
        self._init_patch_embedding(img_size, patch_size, embed_dim, init_values)

        # 初始化条件嵌入
        if self.enable_cond:
            self._init_cond_embeddings(embed_dim, img_size, patch_size, num_register_tokens)

        # 初始化 RoPE（仅用于 alt_start 层之后）
        self._init_rotary_position_embedding(
            rope_base, normalized_rope, embed_dim // num_heads,
            rope_normalize_coords, rope_shift_coords,
            rope_jitter_coords, rope_rescale_coords
        )

        # 初始化 Transformer Blocks
        self._init_transformer_blocks(
            block_fn, embed_dim, num_heads, mlp_ratio,
            qkv_bias, proj_bias, ffn_bias, init_values, qk_norm
        )

        # 初始化可学习 token
        self._init_learnable_tokens(embed_dim, num_register_tokens)

        # 计算 patch_start_idx
        # DA3 风格：去掉 register tokens，用 cam_token 替换 cls_token
        if self.enable_cond:
            # cam_token(1) + pose_token(1) + ray_token(1) = 3
            self.patch_start_idx = 1 + 1 + 1
        else:
            # cam_token(1) = 1
            self.patch_start_idx = 1

        # DINOv2 前 alt_start 层的 patch_start_idx（只有 cls_token，无 register tokens）
        self.dino_patch_start_idx = 1

        # 注册归一化常量
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).reshape(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False
        self.use_gradient_checkpointing = True  # 默认开启，可通过 set_gradient_checkpointing(False) 关闭

    def set_gradient_checkpointing(self, enable: bool):
        """设置是否使用 gradient checkpointing。关闭后 backward 更快但显存占用更大。"""
        self.use_gradient_checkpointing = enable
        print(f"[VGT Lite] Gradient checkpointing: {'ON' if enable else 'OFF'}")

    def _init_patch_embedding(self, img_size, patch_size, embed_dim, init_values):
        """初始化 DINOv2 风格的 Patch Embedding，包含 cls_token、pos_embed（去掉 register_tokens，与 DA3 对齐）。"""
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # DINOv2 标准参数（去掉 register_tokens，与 DA3 对齐）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # +1 for cls_token

        # 最终的 LayerNorm（DINOv2 标准）
        self.norm = norm_layer(embed_dim)

        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)

    def _init_cond_embeddings(self, embed_dim, img_size, patch_size, num_reg_tokens):
        """初始化条件嵌入（与 Mirror 完全一致）。"""
        assert self.cond_methods is not None
        assert self.cond_methods[0] == "token"

        # Camera pose embedding
        self.pose_embed = nn.Sequential(
            nn.Linear(7, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True)
        )

        # Depth map embedding
        if self.cond_methods[1] == "pow3r":
            self.depth_embed = PatchEmbed_Mlp(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=1,
                embed_dim=embed_dim
            )
        else:
            raise NotImplementedError

        # Ray direction embedding
        self.ray_embed = nn.Sequential(
            nn.Linear(4, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True)
        )

    def _init_rotary_position_embedding(
        self, rope_base, normalized_rope, head_dim,
        rope_normalize_coords, rope_shift_coords,
        rope_jitter_coords, rope_rescale_coords
    ):
        """初始化 RoPE（与 Mirror 一致，使用 NormalizedRoPE）。"""
        if normalized_rope:
            logger.info("[MirrorLite] 使用 NormalizedRoPE!")
            from .layers.norm_rope import NormalizedRotaryPositionEmbedding2D, PositionGetter
            if head_dim % 4 != 0:
                raise ValueError("RoPE 要求 head_dim 能被 4 整除")
            self.rope = NormalizedRotaryPositionEmbedding2D(
                head_dim=head_dim,
                base=rope_base,
                normalize_coords=rope_normalize_coords,
                shift_coords=rope_shift_coords,
                jitter_coords=rope_jitter_coords,
                rescale_coords=rope_rescale_coords,
            ) if rope_base > 0 else None
            self.pos_getter = PositionGetter() if self.rope is not None else None
        else:
            logger.info("[MirrorLite] 使用标准 RoPE!")
            from .layers.rope import RotaryPositionEmbedding2D, PositionGetter
            self.rope = RotaryPositionEmbedding2D(
                frequency=rope_base,
            ) if rope_base > 0 else None
            self.pos_getter = PositionGetter() if self.rope is not None else None

    def _init_transformer_blocks(
        self, block_fn, embed_dim, num_heads, mlp_ratio,
        qkv_bias, proj_bias, ffn_bias, init_values, qk_norm
    ):
        """
        初始化 24 层 Transformer Block（DA3 方案 A：参数共享）。

        - 前 alt_start 层：标准 DINOv2 Block（无 QK Norm、无 RoPE）
        - 第 alt_start 层起：启用 QK Norm + RoPE 的 Block
        """
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                # 前 alt_start 层不启用 QK Norm 和 RoPE
                qk_norm=qk_norm if i >= self.alt_start else False,
                rope=self.rope if i >= self.alt_start else None,
            )
            for i in range(self.depth)
        ])

    def _init_learnable_tokens(self, embed_dim, num_reg_tokens):
        """Initialize learnable tokens.

        Compared with mirrorlite_stage1_da3style:
        - Training end: cam_token [1, 2, 1, C], dim=1 distinguishes ref (idx=0) / src (idx=1).
        - DepthMaster: cam_token [1, 1, 1, C], no ref/src distinction; all views share one token.

        Reasons:
        - cam_token is repurposed by ``scale_head`` for metric_scale regression
          (it no longer participates in relative-pose regression), so training
          from scratch keeps things clean and avoids pretrained biases.
        - register tokens are not used.
        """
        self.cam_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        nn.init.normal_(self.cam_token, std=1e-6)

    def interpolate_pos_encoding(self, x, w, h):
        """插值位置编码（与 DINOv2 一致）。"""
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        sx = float(w0 + 0.1) / M
        sy = float(h0 + 0.1) / M
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=True,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_dino_tokens(self, images, b, seq_len):
        """
        准备 DINOv2 前 alt_start 层的 token 序列。

        输入: images [B*S, 3, H, W]
        输出: tokens [B*S, 1+num_patches, embed_dim]（无 register tokens，与 DA3 对齐）
        """
        ch, h, w = images.shape[1:]

        # Patch embedding
        patch_tokens = self.patch_embed(images)  # [B*S, num_patches, embed_dim]

        # cls_token + patch_tokens + pos_embed
        cls_tokens = self.cls_token.expand(b * seq_len, -1, -1)
        x = torch.cat([cls_tokens, patch_tokens], dim=1)  # [B*S, 1+num_patches, embed_dim]
        x = x + self.interpolate_pos_encoding(x, w, h)

        # 不插入 register_tokens（与 DA3 对齐）
        return x

    def inject_conditions(self, x, b, seq_len, images, depth_maps, ray_dirs, poses, cond_flags):
        """
        在第 alt_start 层注入条件（DA3 风格）。

        核心操作：
        1. 保留 DINOv2 原始 cls_token（不替换为 cam_token）
        2. 如果 enable_cond，额外插入 pose_token 和 ray_token，融合 depth_tokens
        3. 不使用 register tokens

        输入: x [B, S, N, C]（4D 格式）
        输出: x [B, S, N', C]（N' 可能因条件 token 增加而变大）
        """
        embed_dim = self.embed_dim
        device = x.device
        dtype = x.dtype

        # ===== 保留原始 cls_token =====
        # 不再用 cam_token 替换 cls_token，scale_head 直接使用 DINOv2 cls_token 的输出
        cls_token = x[:, :, :1, :]  # [B, S, 1, C]，保留原始 DINOv2 cls_token

        # 提取 patch_tokens（跳过 cls_token）
        patch_tokens = x[:, :, self.dino_patch_start_idx:, :]  # [B, S, num_patches, C]

        if self.enable_cond:
            # 处理条件输入
            h, w = images.shape[-2:]
            num_patches = patch_tokens.shape[2]

            # Pose token
            use_poses = (cond_flags[0] == 1 and poses is not None)
            if use_poses:
                pose_flat = poses.reshape(b * seq_len, -1)
                pose_tokens = self.pose_embed(pose_flat).unsqueeze(1)  # [B*S, 1, C]
            else:
                pose_tokens = torch.zeros((b * seq_len, 1, embed_dim), device=device, dtype=dtype)
            pose_tokens = pose_tokens.reshape(b, seq_len, 1, embed_dim)

            # Depth tokens（加法融合到 patch_tokens）
            use_depth = (cond_flags[1] == 1 and depth_maps is not None)
            if use_depth:
                depth_flat = depth_maps.reshape(b * seq_len, 1, h, w)
                depth_tokens = self.depth_embed(depth_flat).reshape(b, seq_len, num_patches, embed_dim)
            else:
                depth_tokens = torch.zeros((b, seq_len, num_patches, embed_dim), device=device, dtype=dtype)
            patch_tokens = patch_tokens + depth_tokens

            # Ray token
            use_rays = (cond_flags[2] == 1 and ray_dirs is not None)
            if use_rays:
                ray_flat = ray_dirs.reshape(b * seq_len, -1)
                ray_tokens = self.ray_embed(ray_flat).unsqueeze(1)  # [B*S, 1, C]
            else:
                ray_tokens = torch.zeros((b * seq_len, 1, embed_dim), device=device, dtype=dtype)
            ray_tokens = ray_tokens.reshape(b, seq_len, 1, embed_dim)

            # 拼接: [cls_token, pose, ray, patches]（保留原始 cls_token，无 reg_tokens）
            x = torch.cat([cls_token, pose_tokens, ray_tokens, patch_tokens], dim=2)
        else:
            # 不注入条件，保留原始 cls_token + patch_tokens（无 reg_tokens）
            x = torch.cat([cls_token, patch_tokens], dim=2)

        return x

    def _prepare_rope_positions(self, b, seq_len, h, w, device, dtype):
        """
        准备 RoPE 位置编码。

        返回:
            pos: 正常 2D 坐标位置（用于 Local 注意力）[B, S, N, 2]
            pos_nodiff: 全零坐标（用于 Global 注意力）[B, S, N, 2]
        """
        if self.rope is None:
            return None, None

        # 获取 patch 网格位置
        pos = self.pos_getter(
            b * seq_len,
            h // self.patch_size,
            w // self.patch_size,
            device=device
        )  # [B*S, num_patches, 2]
        pos = pos.reshape(b, seq_len, -1, 2)

        # Global 注意力使用全零位置（与 DA3 一致）
        pos_nodiff = torch.zeros_like(pos)

        # 为特殊 token 添加位置（偏移 +1 避免冲突）
        if self.patch_start_idx > 0:
            pos = pos + 1
            special_pos = torch.zeros(b, seq_len, self.patch_start_idx, 2, device=device, dtype=pos.dtype)
            pos = torch.cat([special_pos, pos], dim=2)

            pos_nodiff = pos_nodiff + 1
            special_pos_nodiff = torch.zeros(b, seq_len, self.patch_start_idx, 2, device=device, dtype=pos_nodiff.dtype)
            pos_nodiff = torch.cat([special_pos_nodiff, pos_nodiff], dim=2)

        return pos, pos_nodiff

    def forward(
        self,
        images: torch.Tensor,
        priors: List | None = None,
        cond_flags: List[int] = [0, 0, 0],
        ctx_frames: int = None,
        enable_bf16=False,
        sp_size: int = 1,
        sp_group=None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        前向传播。

        Args:
            images: [B, S, 3, H, W]，范围 [0, 1]
            priors: (depth_maps, ray_dirs, poses) 或 None
            cond_flags: [pose, depth, rays] 条件标志
            enable_bf16: 是否启用 bf16
            sp_size: Sequence Parallelism 大小
            sp_group: SP 进程组

        Returns:
            (list[torch.Tensor], int): 中间层特征列表和 patch_start_idx
        """
        depth_maps, ray_dirs, poses = priors if priors is not None else (None, None, None)

        b, seq_len, ch, h, w = images.shape
        if ch != 3:
            raise ValueError(f"期望 3 个输入通道，得到 {ch}")

        # =====================================================================
        # Phase 1: DINOv2 前 alt_start 层（标准 DINOv2 行为）
        # =====================================================================
        with torch.amp.autocast('cuda', enabled=(not enable_bf16), dtype=torch.bfloat16):
            # 图像归一化
            images_norm = (images - self._resnet_mean) / self._resnet_std
            images_flat = images_norm.reshape(b * seq_len, ch, h, w)

            # 准备 DINOv2 token 序列
            x = self.prepare_dino_tokens(images_flat, b, seq_len)
            # x: [B*S, 1+num_reg+num_patches, embed_dim]

            # 前 alt_start 层：标准 DINOv2 Local 注意力（无 RoPE）
            for i in range(self.alt_start):
                if self.training and self.use_gradient_checkpointing:
                    x = checkpoint(self.blocks[i], x, use_reentrant=self.use_reentrant)
                else:
                    x = self.blocks[i](x)

        # =====================================================================
        # Phase 2: 条件注入（在第 alt_start 层）
        # =====================================================================
        with torch.amp.autocast('cuda', enabled=(not enable_bf16), dtype=torch.bfloat16):
            # reshape 为 4D: [B, S, N, C]
            _, n_tokens, embed_dim = x.shape
            x = x.reshape(b, seq_len, n_tokens, embed_dim)

            # 注入条件
            x = self.inject_conditions(x, b, seq_len, images_flat, depth_maps, ray_dirs, poses, cond_flags)
            # x: [B, S, N', C]，N' 可能因条件 token 增加

            _, _, patch_count, _ = x.shape

            # 准备 RoPE 位置
            pos, pos_nodiff = self._prepare_rope_positions(b, seq_len, h, w, x.device, x.dtype)

        # =====================================================================
        # Phase 3: 交替注意力层（alt_start 到 depth-1）
        # =====================================================================
        with torch.amp.autocast('cuda', enabled=(not enable_bf16), dtype=torch.bfloat16):
            outputs = self._forward_alternating(
                x, b, seq_len, patch_count, embed_dim, pos, pos_nodiff
            )

        return outputs, self.patch_start_idx

    def _forward_alternating(
        self, x, b, seq_len, patch_count, embed_dim, pos, pos_nodiff
    ):
        """标准（非 SP）交替注意力前向传播。"""
        outputs = []
        local_x = None

        for i in range(self.alt_start, self.depth):
            if i % 2 == 0:
                # 偶数层：Local 注意力 [B*S, N, C]
                x_local = x.reshape(b * seq_len, patch_count, embed_dim)
                pos_local = pos.reshape(b * seq_len, patch_count, 2) if pos is not None else None

                if self.training and self.use_gradient_checkpointing:
                    x_local = checkpoint(
                        self.blocks[i], x_local, pos=pos_local,
                        use_reentrant=self.use_reentrant
                    )
                else:
                    x_local = self.blocks[i](x_local, pos=pos_local)

                x = x_local.reshape(b, seq_len, patch_count, embed_dim)
                local_x = x.clone()
            else:
                # 奇数层：Global 注意力 [B, S*N, C]
                x_global = x.reshape(b, seq_len * patch_count, embed_dim)
                pos_global = pos_nodiff.reshape(b, seq_len * patch_count, 2) if pos_nodiff is not None else None

                if self.training and self.use_gradient_checkpointing:
                    x_global = checkpoint(
                        self.blocks[i], x_global, pos=pos_global,
                        use_reentrant=self.use_reentrant
                    )
                else:
                    x_global = self.blocks[i](x_global, pos=pos_global)

                x = x_global.reshape(b, seq_len, patch_count, embed_dim)

            # 提取中间层特征
            if i in self.intermediate_idxs:
                if local_x is not None:
                    combined_out = torch.cat([local_x, x], dim=-1)  # [B, S, N, 2*C]
                else:
                    # 如果还没有 local_x（不应该发生），使用 x 本身
                    combined_out = torch.cat([x, x], dim=-1)
                outputs.append(combined_out)

        return outputs

    def load_dinov2_weights(self, state_dict: dict, strict: bool = False):
        """
        加载 DINOv2 预训练权重。

        正确映射：
        - patch_embed.proj.weight/bias
        - cls_token
        - pos_embed
        - register_tokens (如果存在)
        - blocks[0-23] 的 qkv/proj/mlp/norm1/norm2/ls1/ls2

        第 alt_start-23 层新增的 q_norm/k_norm 使用默认初始化。

        Args:
            state_dict: DINOv2 预训练权重字典
            strict: 是否严格匹配
        """
        current_state = self.state_dict()
        matched = 0
        skipped = 0
        not_found = 0

        for key in current_state.keys():
            if key in state_dict:
                if current_state[key].shape == state_dict[key].shape:
                    current_state[key] = state_dict[key]
                    matched += 1
                else:
                    logger.warning(
                        f"[DINOv2 权重加载] Shape 不匹配，跳过 '{key}': "
                        f"当前 {current_state[key].shape} vs 预训练 {state_dict[key].shape}"
                    )
                    skipped += 1
            else:
                not_found += 1

        self.load_state_dict(current_state, strict=True)
        logger.info(
            f"[DINOv2 权重加载] 匹配: {matched}, 跳过(shape不匹配): {skipped}, "
            f"未找到(新参数): {not_found}, 总计: {len(current_state)}"
        )


def _expand_and_flatten_special_tokens(token_tensor, b, seq_len):
    """
    处理特殊 token 的展开函数（兼容两种 shape）。

    同时兼容：
    - (1, 1, X, C): simplified path; all views share a single token (used by DepthMaster).
    - (1, 2, X, C): original path; dim=1 distinguishes ref/src frames (kept for backward compat).

    Args:
        token_tensor: [1, 1, X, C] 或 [1, 2, X, C]
        b: batch size
        seq_len: 序列长度

    Returns:
        [B*S, X, C]
    """
    if token_tensor.shape[1] == 1:
        # 简化路径: (1, 1, X, C) -> expand -> (B, S, X, C) -> (B*S, X, C)
        tokens = token_tensor.expand(b, seq_len, *token_tensor.shape[2:])
        return tokens.reshape(b * seq_len, *token_tensor.shape[2:])
    else:
        # 兼容原始 (1, 2, X, C) 路径
        first_frame_tokens = token_tensor[:, 0:1, ...].expand(b, 1, *token_tensor.shape[2:])
        remaining_frame_tokens = token_tensor[:, 1:, ...].expand(b, seq_len - 1, *token_tensor.shape[2:])
        combined_tokens = torch.cat([first_frame_tokens, remaining_frame_tokens], dim=1)
        return combined_tokens.reshape(b * seq_len, *combined_tokens.shape[2:])



