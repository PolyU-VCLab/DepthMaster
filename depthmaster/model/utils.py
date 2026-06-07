from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

def wrap_module_with_gradient_checkpointing(module: nn.Module):
    from torch.utils.checkpoint import checkpoint
    class _CheckpointingWrapper(module.__class__):
        _restore_cls = module.__class__
        def forward(self, *args, **kwargs):
            return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)
        
    module.__class__ = _CheckpointingWrapper
    return module


def unwrap_module_with_gradient_checkpointing(module: nn.Module):
    module.__class__ = module.__class__._restore_cls


def wrap_dinov2_attention_with_sdpa(module: nn.Module):
    assert torch.__version__ >= '2.0', "SDPA requires PyTorch 2.0 or later"
    class _AttentionWrapper(module.__class__):
        def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, H, N, C // H)

            q, k, v = torch.unbind(qkv, 0)      # (B, H, N, C // H)

            x = F.scaled_dot_product_attention(q, k, v, attn_bias)
            x = x.permute(0, 2, 1, 3).reshape(B, N, C) 

            x = self.proj(x)
            x = self.proj_drop(x)
            return x
    module.__class__ = _AttentionWrapper
    return module


def sync_ddp_hook(state, bucket: torch.distributed.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """DDP 通信 hook：使用 bf16 异步 all-reduce，通信量减半 + 通信/计算重叠。
    
    流程：fp32 梯度 → div(world_size) → 转 bf16 → 异步 all-reduce(bf16) → 转回 fp32
    
    关键优化：
    1. bf16 通信：通信量从 4 bytes/param 降为 2 bytes/param
    2. 异步执行：all-reduce 与后续 bucket 的反向传播计算重叠，
       避免阻塞等待，大幅减少 backward 中的通信等待时间
    """
    group_to_use = torch.distributed.group.WORLD
    world_size = group_to_use.size()
    
    # 获取 fp32 梯度并求平均
    grad_fp32 = bucket.buffer()
    grad_fp32.div_(world_size)
    
    # 转为 bf16 减半通信量
    grad_bf16 = grad_fp32.to(torch.bfloat16)
    
    # 异步 bf16 all-reduce（返回 Future，不阻塞当前 CUDA stream）
    fut = torch.distributed.all_reduce(grad_bf16, group=group_to_use, async_op=True).get_future()
    
    # 注册回调：all-reduce 完成后，将 bf16 结果转回 fp32 写回原 buffer
    def _to_fp32(fut_result):
        result = fut_result.value()[0]  # all_reduce 的 Future.value() 返回 list
        grad_fp32.copy_(result)
        return grad_fp32
    
    return fut.then(_to_fp32)
