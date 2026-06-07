# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


# ============ Flash Attention 2 (Tri Dao) ============
FLASH_ATTN_ENABLED = os.environ.get("FLASH_ATTN_DISABLED") is None
try:
    if FLASH_ATTN_ENABLED:
        from flash_attn import flash_attn_func

        FLASH_ATTN_AVAILABLE = True
        logger.info("Flash Attention 2 is available")
    else:
        raise ImportError
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# ============ xFormers ============
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Attention)")
    else:
        # warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    # warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, H, N, C // H)

        q, k, v = qkv.unbind(0)      # (B, H, N, C // H)

        x = F.scaled_dot_product_attention(q, k, v, attn_bias)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C) 

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    """
    高效 Attention 实现，优先级：
    1. Flash Attention 2 (flash_attn 包，Tri Dao 实现) — 最快，显存最省
    2. xFormers memory_efficient_attention — 次优
    3. PyTorch F.scaled_dot_product_attention — 兜底（自动选择后端）
    """
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        # 优先使用 Flash Attention 2（不支持 attn_bias，需要 attn_bias 时回退）
        if FLASH_ATTN_AVAILABLE and attn_bias is None:
            return self._forward_flash_attn(x)

        # 其次使用 xFormers
        if XFORMERS_AVAILABLE:
            return self._forward_xformers(x, attn_bias)

        # 兜底：PyTorch SDPA
        if attn_bias is not None:
            raise AssertionError("xFormers or Flash Attention is required for using nested tensors with attn_bias")
        return super().forward(x)

    def _forward_flash_attn(self, x: Tensor) -> Tensor:
        """使用 Flash Attention 2 (Tri Dao) 实现"""
        B, N, C = x.shape
        head_dim = C // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim)  # (B, N, 3, H, D)

        q, k, v = qkv.unbind(2)  # 每个 (B, N, H, D)

        # flash_attn_func 要求输入 dtype 为 fp16/bf16
        input_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.half(), k.half(), v.half()

        # dropout_p 仅在训练时使用
        dropout_p = self.attn_drop.p if self.training else 0.0
        x = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=False)  # (B, N, H, D)

        x = x.to(input_dtype)
        x = x.reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _forward_xformers(self, x: Tensor, attn_bias=None) -> Tensor:
        """使用 xFormers memory_efficient_attention 实现"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
