"""DepthMaster Joint Training Wrapper.

This module trains the DepthMaster model under the Mirror PyTorch Lightning
framework.

Design:
    * Reuse DepthMaster's TrainDataLoaderPipeline and loss functions as-is.
    * Replace only the outer training loop (Accelerate -> Lightning).
    * No changes to the data and loss code.

Usage:
    The wrapper is instantiated through Hydra configuration files and the
    Lightning Trainer manages the training loop.
"""
import os
import sys
import json
import random
import time
import io
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

# Make the local `depthmaster/` package importable when launching from arbitrary cwd.
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from depthmaster.model import DepthMasterModel
from depthmaster.train.losses import (
    affine_invariant_global_loss,
    affine_invariant_global_loss_panorama,
    affine_invariant_local_loss,
    affine_invariant_segment_loss,
    edge_loss,
    normal_loss,
    mask_l2_loss,
    mask_bce_loss,
    metric_scale_loss,
    normal_map_loss,
    depth_affine_loss,
    panorama_depth_affine_loss,
    panorama_depth_affine_loss_hard,
    depth_to_points_global_loss,
    erp_depth_affine_loss,
    z_aligned_loss,
    z_scale_aligned_loss,
    camera_consistency_loss,
    monitoring,
    correspondence_consistency_loss,
    cubemap_seam_loss,
    disparity_affine_loss,
)
from depthmaster.train.e2c_gpu import E2C_GPU
from depthmaster.train.utils import build_optimizer, build_lr_scheduler
from depthmaster.utils.tools import key_average, flatten_nested_dict
from depthmaster.utils.vis import colorize_depth, colorize_normal
from depthmaster.train.cubemap_to_equirect import cubemap_to_equirect_np, cubemap_to_equirect_torch

import utils3d

from training.utils.logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DepthMasterWrapper(LightningModule):
    """Lightning wrapper for joint training of DepthMaster.

    Wraps a DepthMasterModel and implements ``training_step``,
    ``configure_optimizers`` and friends. The DepthMaster loss functions and
    data pipeline are reused unchanged.
    """

    def __init__(
        self,
        # Model configuration (corresponds to the "model" field in depthmaster_train.json).
        model: Dict[str, Any],
        # Loss configuration (the "loss" field; covers the four label_types A/B/C/D).
        loss: Dict[str, Any],
        # Optimizer configuration (the "optimizer" field).
        optimizer: Dict[str, Any],
        # LR scheduler configuration (the "lr_scheduler" field).
        lr_scheduler: Dict[str, Any],
        # CCL loss configuration.
        ccl_loss: Optional[Dict[str, Any]] = None,
        # Path to Mirror pretrained weights.
        mirror_pretrained: Optional[str] = None,
        # Path to a DepthMaster pretrained model (used by ``from_pretrained``).
        pretrained: Optional[str] = None,
        # If True, drop the pts_head weights from the pretrained checkpoint and
        # re-initialize them from scratch. Useful for ablations that change the
        # pts_head activation semantics (e.g. inv_log -> exp).
        skip_pts_head_pretrained: bool = False,
        # Whether to enable gradient checkpointing.
        enable_gradient_checkpointing: bool = True,
        # Whether to enable EMA.
        enable_ema: bool = True,
        # EMA decay.
        ema_decay: float = 0.999,
        # GPU equirect-to-cubemap projection.
        gpu_e2c: bool = True,
        e2c_face_w: int = 518,
        e2c_fov_deg: float = 95.0,
        # Number of low-resolution warm-up steps.
        low_resolution_training_steps: int = 0,
        # Visualization interval.
        vis_every: int = 0,
        vis_log_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        # Save hyperparameters (large objects are excluded).
        self.save_hyperparameters(ignore=["kwargs"])

        # 存储配置
        self.model_config = model if isinstance(model, dict) else OmegaConf.to_container(model, resolve=True)
        self.loss_config = loss if isinstance(loss, dict) else OmegaConf.to_container(loss, resolve=True)
        self.optimizer_config = optimizer if isinstance(optimizer, dict) else OmegaConf.to_container(optimizer, resolve=True)
        self.lr_scheduler_config = lr_scheduler if isinstance(lr_scheduler, dict) else OmegaConf.to_container(lr_scheduler, resolve=True)
        self.ccl_config = ccl_loss if ccl_loss is None else (ccl_loss if isinstance(ccl_loss, dict) else OmegaConf.to_container(ccl_loss, resolve=True))
        self.mirror_pretrained = mirror_pretrained
        self.pretrained = pretrained
        self._skip_pts_head_pretrained = bool(skip_pts_head_pretrained)
        self._enable_gradient_checkpointing = enable_gradient_checkpointing
        self._enable_ema = enable_ema
        self._ema_decay = ema_decay
        self._gpu_e2c = gpu_e2c
        self._e2c_face_w = e2c_face_w
        self._e2c_fov_deg = e2c_fov_deg
        self._low_resolution_training_steps = low_resolution_training_steps
        self._vis_every = vis_every
        self._vis_log_dir = vis_log_dir

        # Instantiate the DepthMaster model.
        self.model = DepthMasterModel(**self.model_config)
        log.info(f"[DepthMasterWrapper] Model instantiated, parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")

        # Load pretrained weights.
        if self.pretrained is not None:
            log.info(f"[DepthMasterWrapper] Loading from DepthMaster pretrained model: {self.pretrained}")
            pretrained_model = DepthMasterModel.from_pretrained(self.pretrained)
            pretrained_state = pretrained_model.state_dict()

            # ===== 可选：跳过 pts_head 预训练权重（pts_head 从头随机初始化）=====
            # 用于切换 pts_head 激活语义（如 inv_log ↔ exp）的对照实验场景。
            # 注意：这里在 enable_separate_mask_head 处理之前执行，
            # 这样 mask_head 也不会从 pts_head trunk 复制旧权重。
            if self._skip_pts_head_pretrained:
                pts_head_keys = [k for k in pretrained_state if k.startswith('pts_head.')]
                for k in pts_head_keys:
                    del pretrained_state[k]
                if pts_head_keys:
                    log.info(
                        f"[DepthMasterWrapper] skip_pts_head_pretrained=True: 删除 pts_head 预训练权重"
                        f"（{len(pts_head_keys)} 个 key），pts_head 从头随机初始化"
                    )

            # Compatibility with enable_separate_mask_head: the checkpoint's pts_head
            # is typically 4-channel (xyz+mask). When the target model uses a separate
            # mask head, pts_head degenerates to 3 channels (xyz only) and we need to
            #   1) crop pts_head.scratch.output_conv2.2 weight/bias from 4ch -> 3ch,
            #   2) copy the trunk weights (everything except the final output conv)
            #      from pts_head into mask_head, so the mask_head trunk does not
            #      restart from random initialization.
            if getattr(self.model, 'enable_separate_mask_head', False):
                pts_w_key = 'pts_head.scratch.output_conv2.2.weight'
                pts_b_key = 'pts_head.scratch.output_conv2.2.bias'
                # 1) 裁剪 pts_head 最后一层 Conv: 4ch (xyz+mask) → 3ch (xyz)
                if pts_w_key in pretrained_state:
                    w = pretrained_state[pts_w_key]
                    target_w_shape = self.model.state_dict()[pts_w_key].shape
                    if w.shape[0] != target_w_shape[0]:
                        pretrained_state[pts_w_key] = w[:target_w_shape[0]].clone()
                        log.info(f"[DepthMasterWrapper] 裁剪 pts_head Conv weight: {tuple(w.shape)} → {tuple(pretrained_state[pts_w_key].shape)} (只保留 xyz 通道)")
                if pts_b_key in pretrained_state:
                    b = pretrained_state[pts_b_key]
                    target_b_shape = self.model.state_dict()[pts_b_key].shape
                    if b.shape[0] != target_b_shape[0]:
                        pretrained_state[pts_b_key] = b[:target_b_shape[0]].clone()
                        log.info(f"[DepthMasterWrapper] 裁剪 pts_head Conv bias: {tuple(b.shape)} → {tuple(pretrained_state[pts_b_key].shape)} (只保留 xyz 通道)")
                # 2) 把 pts_head 的 trunk 权重（除最后输出层外）复制到 mask_head
                target_state_keys = set(self.model.state_dict().keys())
                copied_keys = []
                for k in list(pretrained_state.keys()):
                    if k.startswith('pts_head.') and not k.startswith('pts_head.scratch.output_conv2.2.'):
                        new_k = 'mask_head.' + k[len('pts_head.'):]
                        if new_k in target_state_keys and new_k not in pretrained_state:
                            pretrained_state[new_k] = pretrained_state[k].clone()
                            copied_keys.append(new_k)
                if copied_keys:
                    log.info(f"[DepthMasterWrapper] enable_separate_mask_head: 从 pts_head 复制 {len(copied_keys)} 个 trunk 权重到 mask_head（最后输出 Conv 1ch 从头学习）")

            missing, unexpected = self.model.load_state_dict(pretrained_state, strict=False)
            if missing:
                log.info(f"  Missing keys: {missing[:10]}")
            if unexpected:
                log.info(f"  Unexpected keys: {unexpected[:10]}")
            del pretrained_model
            del pretrained_state
        elif self.mirror_pretrained is not None:
            log.info(f"[DepthMasterWrapper] Initializing from Mirror pretrained weights: {self.mirror_pretrained}")
            self.model.init_weights(mirrorlite_pretrained=self.mirror_pretrained)
        else:
            log.info("[DepthMasterWrapper] No pretrained weights provided, using random initialization")

        # Configure gradient checkpointing.
        self.model.enable_gradient_checkpointing(self._enable_gradient_checkpointing)
        log.info(f"[DepthMasterWrapper] Gradient Checkpointing: {'ON' if self._enable_gradient_checkpointing else 'OFF'}")

        # Conditional input parameters (pose_embed, depth_embed, ray_embed):
        # - In panorama mode, W2C / intrinsics are passed in and these parameters take part in the computation.
        # - In perspective mode no condition is passed in, so they do not participate.
        # - Because perspective and panorama steps alternate during training, these parameters
        #   only receive gradients on panorama steps; DDP find_unused_parameters=true handles
        #   the perspective steps where they have no gradient.
        # - depth_embed is not used (cond_flags=[1,0,1]) so we freeze it.
        # - visual_geometry_transformer.norm is the final DINOv2 LayerNorm; it is bypassed in
        #   MirrorLite, so we freeze it as well.
        frozen_prefixes = [
            'visual_geometry_transformer.depth_embed.',
            'visual_geometry_transformer.norm.',
        ]
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if any(name.startswith(prefix) for prefix in frozen_prefixes):
                param.requires_grad = False
                frozen_count += 1
        if frozen_count > 0:
            log.info(f"[DepthMasterWrapper] Frozen {frozen_count} unused parameters (depth_embed, vgt.norm)")

        # EMA 模型（仅在 rank 0 上维护）
        self._ema_model = None  # 延迟初始化，在 on_fit_start 中创建

        # GPU E2C 投影器（延迟初始化）
        self._e2c_gpu = None

        # 训练记录
        self._records = []

        # 可视化状态追踪：确保每次可视化周期内透视图和全景图各保存 10 张
        self._vis_pending_persp = False
        self._vis_pending_pano = False
        self._vis_pending_step = -1  # 记录触发可视化的 step
        self._vis_persp_count = 0  # 当前周期已保存的透视图数量
        self._vis_pano_count = 0   # 当前周期已保存的全景图数量
        self._vis_num_samples = 10  # 每种类型保存的样本数

    def on_fit_start(self):
        """训练开始时的初始化。"""
        # 初始化 GPU E2C 投影器
        if self._gpu_e2c:
            self._e2c_gpu = E2C_GPU(
                face_w=self._e2c_face_w,
                fov_deg=self._e2c_fov_deg,
                device=self.device,
            )
            log.info(f"[DepthMasterWrapper] GPU E2C 投影器已初始化: face_w={self._e2c_face_w}, fov_deg={self._e2c_fov_deg}")

        # 初始化 EMA 模型
        if self._enable_ema and self.global_rank == 0:
            ema_avg_fn = lambda averaged, model_param, num_averaged: (
                self._ema_decay * averaged + (1 - self._ema_decay) * model_param
            )
            self._ema_model = torch.optim.swa_utils.AveragedModel(
                self.model, device=self.device, avg_fn=ema_avg_fn
            )
            log.info(f"[DepthMasterWrapper] EMA 模型已初始化 (decay={self._ema_decay})")

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        训练步骤：区分透视图和全景图 batch，分别计算 loss。

        Args:
            batch: 由 DepthMasterDataModule 提供的 batch 数据
            batch_idx: batch 索引

        Returns:
            loss: 标量 loss tensor
        """
        device = self.device
        is_panorama_batch = batch.get('camera_type') in ('Panorama', 'Panorama_ERP')
        label_type = batch['label_type']
        is_metric = batch['is_metric']

        # ====== 确定 num_tokens ======
        if self.global_step <= self._low_resolution_training_steps:
            num_tokens = self.model_config['num_tokens_range'][0]
        else:
            # 使用 global_step 作为种子确保所有 rank 使用相同的 num_tokens
            rng = random.Random(self.global_step)
            num_tokens = rng.randint(*self.model_config['num_tokens_range'])

        # 注意：使用标准 PyTorch DataLoader 多进程模式后，
        # 数据加载在独立子进程中完成，不再有 GIL 竞争问题，无需 barrier

        # DEBUG: 临时用简单 loss 测试 backward 是否能完成
        if is_panorama_batch:
            loss, sub_losses = self._training_step_panorama(batch, num_tokens, label_type, is_metric)
        else:
            loss, sub_losses = self._training_step_perspective(batch, num_tokens, label_type, is_metric)

        # EMA 更新
        if self._enable_ema and self._ema_model is not None and self.global_rank == 0:
            self._ema_model.update_parameters(self.model)

        # 日志记录
        self.log("train/loss", loss.item(), prog_bar=True, sync_dist=False)
        self.log("train/batch_type", 1.0 if is_panorama_batch else 0.0, prog_bar=False)

        # 记录各子 loss 到 TensorBoard，全部子 loss 显示在进度条
        for k, v in sub_losses.items():
            self.log(f"train/{k}", v, prog_bar=True, sync_dist=False)

        # 可视化：在 iteration 0 和每 vis_every 步触发
        # 策略：触发时标记两种类型都需要可视化，累积多个 batch 直到各保存 10 个样本
        if self._vis_every > 0 and self.global_rank == 0:
            # 检查是否到达可视化触发点
            if self.global_step == 0 or (self.global_step % self._vis_every == 0):
                self._vis_pending_persp = True
                self._vis_pending_pano = True
                self._vis_pending_step = self.global_step
                self._vis_persp_count = 0
                self._vis_pano_count = 0

            # 如果有 pending 的可视化需求，且当前 batch 类型匹配，则执行可视化
            if is_panorama_batch and self._vis_pending_pano:
                self._visualize(batch, is_panorama_batch)
            elif not is_panorama_batch and self._vis_pending_persp:
                self._visualize(batch, is_panorama_batch)

        return loss

    def _training_step_panorama(
        self,
        batch: Dict[str, Any],
        num_tokens: int,
        label_type: List[str],
        is_metric: List[bool],
    ) -> torch.Tensor:
        """全景图 batch 的训练步骤。"""
        device = self.device

        # GPU E2C 模式：先在 GPU 上完成 ERP → Cubemap 投影
        if batch.get('camera_type') == 'Panorama_ERP' and self._e2c_gpu is not None:
            erp_images = [img.to(device) for img in batch['erp_image']]
            erp_depths = [dep.to(device) for dep in batch['erp_depth']]
            max_depths = batch['max_depth'].to(device)
            B_pano = len(erp_images)

            with torch.no_grad():
                gpu_result = self._e2c_gpu.process_panorama_batch(erp_images, erp_depths, max_depths)

            image = gpu_result['image']
            cubemap_intrinsics = gpu_result['intrinsics'].unsqueeze(0).expand(B_pano, -1, -1, -1)
            cubemap_W2C = gpu_result['W2C'].unsqueeze(0).expand(B_pano, -1, -1, -1)
            gt_mask_fin = gpu_result['depth_mask_fin']
            gt_mask_inf = gpu_result['depth_mask_inf']
            gt_depth = gpu_result['depth']
            gt_cubemap_points_world = gpu_result['cubemap_points_world']
            # Retain GT ERP depth for panorama_depth_affine_loss (squeeze channel dim)
            gt_erp_depth_list = [dep.squeeze(0) for dep in erp_depths]  # list of (H, W)
            del erp_images, erp_depths, gpu_result
        else:
            image = batch['image'].to(device)
            cubemap_intrinsics = batch['intrinsics'].to(device)
            cubemap_W2C = batch['W2C'].to(device)
            gt_mask_fin = batch['depth_mask_fin'].to(device)
            gt_mask_inf = batch['depth_mask_inf'].to(device)
            gt_depth = batch['depth'].to(device)
            gt_cubemap_points_world = batch['cubemap_points_world'].to(device)
            # GT ERP depth from batch (squeeze channel dim)
            gt_erp_depth_list = [dep.to(device).squeeze(0) for dep in batch['erp_depth']]  # list of (H, W)

        current_batch_size = image.shape[0]

        # Forward: panorama mode.
        ccl_enabled = self.ccl_config is not None and self.ccl_config.get('enabled', False)
        extra_kwargs = {
            'W2C': cubemap_W2C,
            'intrinsics': cubemap_intrinsics,
        }

        output = self.model(image, num_tokens=0, camera_type="Panorama",
                           return_intermediate_features=ccl_enabled, **extra_kwargs)

        pred_points_per_face = output.get('points', None)
        pred_mask = output.get('mask', None)
        pred_metric_scale = output.get('metric_scale', None)
        pred_depth_per_face = output.get('depth', None)

        # 将预测结果转为 float32 用于 loss 计算
        if pred_points_per_face is not None:
            pred_points_per_face = pred_points_per_face.float()
        if pred_mask is not None:
            pred_mask = pred_mask.float()
        if pred_metric_scale is not None:
            pred_metric_scale = pred_metric_scale.float()
        if pred_depth_per_face is not None:
            pred_depth_per_face = pred_depth_per_face.float()

        # 计算 loss（在 float32 下进行）
        loss_list = []
        all_loss_dicts = []
        for i in range(current_batch_size):
            loss_dict, weight_dict = {}, {}

            # 每个面的 loss
            face_pred_points_all = pred_points_per_face[i]
            face_gt_points_all = gt_cubemap_points_world[i]

            # depth_head 通过 depth_to_points_global_loss 得到的 affine scale，
            # 若该 loss 被启用则用于覆盖 metric_scale_loss 的监督源（替代 point head 的 scale）
            gt_metric_scale_depth_pts = None

            for k, v in self.loss_config.get(label_type[i], {}).items():
                if v['function'] == 'normal_loss':
                    weight_dict[f'normal_{k}'] = v['weight']
                    loss_dict[f'normal_{k}'], _ = normal_loss(face_pred_points_all, face_gt_points_all)
                elif v['function'] == 'edge_loss':
                    weight_dict[f'edge_{k}'] = v['weight']
                    loss_val, _ = edge_loss(face_pred_points_all, face_gt_points_all)
                    loss_dict[f'edge_{k}'] = loss_val.mean()
                elif v['function'] == 'mask_bce_loss':
                    if pred_mask is not None:
                        weight_dict[f'mask_bce_{k}'] = v['weight']
                        loss_val, _ = mask_bce_loss(pred_mask[i], gt_mask_fin[i], gt_mask_inf[i])
                        loss_dict[f'mask_bce_{k}'] = loss_val.mean()
                elif v['function'] == 'mask_l2_loss':
                    if pred_mask is not None:
                        weight_dict[f'mask_l2_{k}'] = v['weight']
                        loss_val, _ = mask_l2_loss(pred_mask[i], gt_mask_fin[i], gt_mask_inf[i])
                        loss_dict[f'mask_l2_{k}'] = loss_val.mean()
                elif v['function'] == 'erp_depth_affine_loss':
                    weight_dict[f'erp_{k}'] = v['weight']

                    # cubemap -> erp，分辨率绑定 face size：(H, 2H)
                    pred_range = pred_points_per_face[i].norm(dim=-1)        # (6, H, H)
                    face_h = pred_range.shape[-2]
                    erp_h, erp_w = face_h, face_h * 2
                    pred_erp = cubemap_to_equirect_torch(
                        pred_range[None], pano_h=erp_h, pano_w=erp_w, mode='bilinear'
                    )[0]                                                     # (erp_h, erp_w)

                    # GT 统一 resize 到 (erp_h, erp_w)
                    gt_erp = gt_erp_depth_list[i]                            # (Hg, Wg)
                    if gt_erp.shape[-2:] != (erp_h, erp_w):
                        fin_mask = torch.isfinite(gt_erp) & (gt_erp > 0)
                        inf_mask = torch.isinf(gt_erp) & (gt_erp > 0)

                        # bilinear 下采样 finite 深度（无效处先填 0，避免 NaN 传播），
                        # 用 mask 同样 bilinear 后做归一化，抵消 0 填充偏差
                        gt_fin = torch.where(fin_mask, gt_erp, torch.zeros_like(gt_erp))
                        gt_fin_rs = F.interpolate(
                            gt_fin[None, None], size=(erp_h, erp_w),
                            mode='bilinear', align_corners=False,
                        )[0, 0]
                        w_rs = F.interpolate(
                            fin_mask.float()[None, None], size=(erp_h, erp_w),
                            mode='bilinear', align_corners=False,
                        )[0, 0]
                        gt_fin_rs = gt_fin_rs / w_rs.clamp_min(1e-6)

                        # finite / inf mask 用 nearest 保持语义
                        fin_rs = F.interpolate(
                            fin_mask.float()[None, None], size=(erp_h, erp_w),
                            mode='nearest',
                        )[0, 0] > 0.5
                        inf_rs = F.interpolate(
                            inf_mask.float()[None, None], size=(erp_h, erp_w),
                            mode='nearest',
                        )[0, 0] > 0.5

                        gt_erp_rs = torch.full(
                            (erp_h, erp_w), float('nan'),
                            device=gt_erp.device, dtype=gt_erp.dtype,
                        )
                        gt_erp_rs[fin_rs] = gt_fin_rs[fin_rs]
                        gt_erp_rs[inf_rs] = float('inf')
                        gt_erp = gt_erp_rs

                    assert pred_erp.shape == gt_erp.shape, \
                        f"ERP shape mismatch: pred={pred_erp.shape}, gt={gt_erp.shape}"

                    loss_dict[f'erp_{k}'], _ = erp_depth_affine_loss(
                        pred_erp, gt_erp, **v.get('params', {})
                    )
                elif v['function'] == 'depth_affine_loss':
                    if pred_depth_per_face is not None:
                        weight_dict[f'depth_{k}'] = v['weight']
                        dl, _ = panorama_depth_affine_loss(
                            pred_depth_per_face[i], gt_depth[i],
                            gt_erp_depth_list[i],
                            **v.get('params', {})
                        )
                        loss_dict[f'depth_{k}'] = dl
                elif v['function'] == 'panorama_depth_affine_loss_hard':
                    # [实验 A] 用硬分配 cubemap → ERP 投影计算 depth affine loss
                    if pred_depth_per_face is not None:
                        weight_dict[f'depth_hard_{k}'] = v['weight']
                        dl, _ = panorama_depth_affine_loss_hard(
                            pred_depth_per_face[i], gt_depth[i],
                            gt_erp_depth_list[i],
                            **v.get('params', {})
                        )
                        loss_dict[f'depth_hard_{k}'] = dl
                elif v['function'] == 'depth_to_points_global_loss':
                    # [实验 B] depth_head zbuf 反投影到世界点云后用 affine_invariant_global_loss_panorama 监督
                    if pred_depth_per_face is not None:
                        weight_dict[f'depth_pts_{k}'] = v['weight']
                        dl, _, gt_metric_scale_depth_pts = depth_to_points_global_loss(
                            pred_depth_per_face[i],
                            cubemap_intrinsics[i],
                            cubemap_W2C[i],
                            gt_cubemap_points_world[i],
                            **v.get('params', {})
                        )
                        loss_dict[f'depth_pts_{k}'] = dl

            # 全局点云 loss
            gt_metric_scale_global = None
            for k, v in self.loss_config.get(label_type[i], {}).items():
                if v['function'] == 'affine_invariant_global_loss':
                    weight_dict[f'global_{k}'] = v['weight']
                    loss_dict[f'global_{k}'], _, gt_metric_scale_global = affine_invariant_global_loss_panorama(
                        pred_points_per_face[i], gt_cubemap_points_world[i], **v['params']
                    )
                elif v['function'] == 'cubemap_seam_loss':
                    weight_dict[f'seam_{k}'] = v['weight']
                    loss_dict[f'seam_{k}'], _ = cubemap_seam_loss(
                        pred_points_per_face[i], gt_cubemap_points_world[i], **v.get('params', {})
                    )
                elif v['function'] == 'metric_scale_loss':
                    # 优先使用 depth_head 对齐得到的 scale 作为 metric_scale 监督源（如果 depth_pts loss 已计算）
                    # 仅在配置了 depth_to_points_global_loss 的实验中触发，对其他实验完全无影响
                    gt_metric_scale_for_supervision = (
                        gt_metric_scale_depth_pts
                        if gt_metric_scale_depth_pts is not None
                        else gt_metric_scale_global
                    )
                    if is_metric[i] and pred_metric_scale is not None and gt_metric_scale_for_supervision is not None:
                        weight_dict[f'global_{k}'] = v['weight']
                        loss_dict[f'global_{k}'], _ = metric_scale_loss(pred_metric_scale[i], gt_metric_scale_for_supervision)

            # 汇总 loss
            weight_dict = {'.'.join(k) if isinstance(k, tuple) else k: v for k, v in flatten_nested_dict(weight_dict).items()}
            loss_dict = {'.'.join(k) if isinstance(k, tuple) else k: v for k, v in flatten_nested_dict(loss_dict).items()}
            loss_ = sum([weight_dict[k] * loss_dict[k] for k in loss_dict if k in weight_dict], start=torch.tensor(0.0, device=device))
            loss_list.append(loss_)
            all_loss_dicts.append(loss_dict)

        loss = sum(loss_list) / max(len(loss_list), 1)

        # 统一 loss 路径：确保所有模型输出都参与计算图，避免 DDP 梯度同步死锁
        unused_outputs_sum = torch.tensor(0.0, device=device)
        if pred_points_per_face is not None:
            unused_outputs_sum = unused_outputs_sum + pred_points_per_face.sum() * 0.0
        if pred_mask is not None:
            unused_outputs_sum = unused_outputs_sum + pred_mask.sum() * 0.0
        if pred_metric_scale is not None:
            unused_outputs_sum = unused_outputs_sum + pred_metric_scale.sum() * 0.0
        if pred_depth_per_face is not None:
            unused_outputs_sum = unused_outputs_sum + pred_depth_per_face.sum() * 0.0
        loss = loss + unused_outputs_sum

        # CCL Loss
        if ccl_enabled and output.get('intermediate_features', None) is not None:
            ccl_weight = self.ccl_config.get('weight', 0.1)
            ccl_fov = self.ccl_config.get('fov_deg', 95.0)
            ccl_max_corr = self.ccl_config.get('max_correspondences', 4096)
            ccl_loss_val, _ = correspondence_consistency_loss(
                output['intermediate_features'], fov_deg=ccl_fov, max_correspondences=ccl_max_corr
            )
            loss = loss + ccl_weight * ccl_loss_val
            self.log("train/ccl_loss", ccl_loss_val.item(), prog_bar=False)

        self.log("train/pano_loss", loss.item(), prog_bar=False)

        # 收集各子 loss 的平均值
        avg_sub_losses = {}
        for loss_d in all_loss_dicts:
            for k, v in loss_d.items():
                if k not in avg_sub_losses:
                    avg_sub_losses[k] = []
                avg_sub_losses[k].append(v.item() if hasattr(v, 'item') else float(v))
        avg_sub_losses = {k: sum(vs) / len(vs) for k, vs in avg_sub_losses.items()}

        return loss, avg_sub_losses

    def _training_step_perspective(
        self,
        batch: Dict[str, Any],
        num_tokens: int,
        label_type: List[str],
        is_metric: List[bool],
    ) -> torch.Tensor:
        """透视图 batch 的训练步骤。"""
        device = self.device

        image = batch['image'].to(device)
        gt_depth = batch['depth'].to(device)
        gt_normal = batch['normal'].to(device)
        gt_mask_fin = batch['depth_mask_fin'].to(device)
        gt_mask_inf = batch['depth_mask_inf'].to(device)
        gt_intrinsics = batch['intrinsics'].to(device)
        current_batch_size = image.shape[0]

        gt_points = utils3d.torch.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
        gt_focal = 1 / (1 / gt_intrinsics[..., 0, 0] ** 2 + 1 / gt_intrinsics[..., 1, 1] ** 2) ** 0.5

        # Forward
        output = self.model(image, num_tokens=num_tokens)
        pred_points = output.get('points', None)
        pred_mask = output.get('mask', None)
        pred_normal = output.get('normal', None)
        pred_metric_scale = output.get('metric_scale', None)
        pred_depth = output.get('depth', None)

        # 将预测结果转为 float32 用于 loss 计算（避免 autocast 下 BCE 不安全的问题）
        if pred_points is not None:
            pred_points = pred_points.float()
        if pred_mask is not None:
            pred_mask = pred_mask.float()
        if pred_normal is not None:
            pred_normal = pred_normal.float()
        if pred_metric_scale is not None:
            pred_metric_scale = pred_metric_scale.float()
        if pred_depth is not None:
            pred_depth = pred_depth.float()

        # 计算 loss（逐 instance，在 float32 下进行）
        loss_list = []
        all_loss_dicts = []
        for i in range(current_batch_size):
            gt_metric_scale = None
            loss_dict, weight_dict = {}, {}

            for k, v in self.loss_config.get(label_type[i], {}).items():
                weight_dict[k] = v['weight']
                if v['function'] == 'affine_invariant_global_loss':
                    loss_dict[k], _, gt_metric_scale = affine_invariant_global_loss(
                        pred_points[i], gt_points[i], **v['params']
                    )
                elif v['function'] == 'affine_invariant_local_loss':
                    loss_dict[k], _ = affine_invariant_local_loss(
                        pred_points[i], gt_points[i], gt_focal[i], gt_metric_scale, **v['params']
                    )
                elif v['function'] == 'affine_invariant_segment_loss':
                    seg_mask_list = batch.get('segmentation_mask', None)
                    seg_labels_list = batch.get('segmentation_labels', None)
                    seg_mask_i = seg_mask_list[i] if seg_mask_list is not None else None
                    seg_labels_i = seg_labels_list[i] if seg_labels_list is not None else None
                    if seg_mask_i is not None and seg_labels_i:
                        seg_mask_i = seg_mask_i.to(device)
                        loss_dict[k], _ = affine_invariant_segment_loss(
                            pred_points[i], gt_points[i], seg_mask_i, seg_labels_i, **v.get('params', {})
                        )
                    # else: 静默跳过 (没有 seg 的样本)
                elif v['function'] == 'z_aligned_loss':
                    loss_dict[k], _ = z_aligned_loss(
                        pred_points[i], gt_points[i], **v.get('params', {})
                    )
                elif v['function'] == 'z_scale_aligned_loss':
                    loss_dict[k], _ = z_scale_aligned_loss(
                        pred_points[i], gt_points[i], **v.get('params', {})
                    )
                elif v['function'] == 'camera_consistency_loss':
                    loss_dict[k], _ = camera_consistency_loss(
                        pred_points[i], gt_points[i], **v.get('params', {})
                    )
                elif v['function'] == 'normal_loss':
                    loss_dict[k], _ = normal_loss(pred_points[i], gt_points[i])
                elif v['function'] == 'edge_loss':
                    loss_dict[k], _ = edge_loss(pred_points[i], gt_points[i])
                elif v['function'] == 'normal_map_loss':
                    if pred_normal is not None:
                        loss_dict[k], _ = normal_map_loss(pred_normal[i], gt_normal[i])
                elif v['function'] == 'mask_bce_loss':
                    if pred_mask is not None:
                        loss_dict[k], _ = mask_bce_loss(pred_mask[i], gt_mask_fin[i], gt_mask_inf[i])
                elif v['function'] == 'mask_l2_loss':
                    if pred_mask is not None:
                        loss_dict[k], _ = mask_l2_loss(pred_mask[i], gt_mask_fin[i], gt_mask_inf[i])
                elif v['function'] == 'metric_scale_loss':
                    if is_metric[i] and pred_metric_scale is not None and gt_metric_scale is not None:
                        loss_dict[k], _ = metric_scale_loss(pred_metric_scale[i], gt_metric_scale)
                elif v['function'] == 'depth_affine_loss':
                    if pred_depth is not None:
                        loss_dict[k], _ = depth_affine_loss(pred_depth[i], gt_depth[i], **v.get('params', {}))
                elif v['function'] == 'disparity_affine_loss':
                    if pred_depth is not None:
                        loss_dict[k], _ = disparity_affine_loss(pred_depth[i], gt_depth[i], **v.get('params', {}))
                else:
                    raise ValueError(f'Undefined loss function: {v["function"]}')

            weight_dict = {'.'.join(k) if isinstance(k, tuple) else k: v for k, v in flatten_nested_dict(weight_dict).items()}
            loss_dict = {'.'.join(k) if isinstance(k, tuple) else k: v for k, v in flatten_nested_dict(loss_dict).items()}
            loss_ = sum([weight_dict[k] * loss_dict[k] for k in loss_dict if k in weight_dict], start=torch.tensor(0.0, device=device))
            loss_list.append(loss_)
            all_loss_dicts.append(loss_dict)

        loss = sum(loss_list) / max(len(loss_list), 1)

        # 统一 loss 路径：确保所有模型输出都参与计算图，避免 DDP 梯度同步死锁
        # 对于当前 label_type 未使用到的输出，乘以 0 加入 loss，不影响数值但保证梯度流
        unused_outputs_sum = torch.tensor(0.0, device=device)
        if pred_normal is not None:
            unused_outputs_sum = unused_outputs_sum + pred_normal.sum() * 0.0
        if pred_mask is not None:
            unused_outputs_sum = unused_outputs_sum + pred_mask.sum() * 0.0
        if pred_metric_scale is not None:
            unused_outputs_sum = unused_outputs_sum + pred_metric_scale.sum() * 0.0
        if pred_points is not None:
            unused_outputs_sum = unused_outputs_sum + pred_points.sum() * 0.0
        if pred_depth is not None:
            unused_outputs_sum = unused_outputs_sum + pred_depth.sum() * 0.0
        loss = loss + unused_outputs_sum

        self.log("train/persp_loss", loss.item(), prog_bar=False)

        # 收集各子 loss 的平均值
        avg_sub_losses = {}
        for loss_d in all_loss_dicts:
            for k, v in loss_d.items():
                if k not in avg_sub_losses:
                    avg_sub_losses[k] = []
                avg_sub_losses[k].append(v.item() if hasattr(v, 'item') else float(v))
        avg_sub_losses = {k: sum(vs) / len(vs) for k, vs in avg_sub_losses.items()}

        return loss, avg_sub_losses

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器。

        复用 DepthMaster 的 build_optimizer 和 build_lr_scheduler 函数。
        支持分组学习率：backbone 低学习率，heads 高学习率。
        """
        optimizer = build_optimizer(self.model, self.optimizer_config)
        lr_scheduler = build_lr_scheduler(optimizer, self.lr_scheduler_config)

        # 打印参数组信息
        for i, pg in enumerate(optimizer.param_groups):
            count = sum(p.numel() for p in pg['params'] if p.requires_grad)
            log.info(f"  Optimizer Group {i}: {count / 1e6:.1f}M params, lr={pg.get('lr', 'N/A')}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "train/lr",
            },
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        保存 checkpoint 时的钩子。

        额外保存 DepthMaster 格式的 .pt 文件，便于 DepthMaster 框架直接加载。
        """
        # 在 checkpoint 中保存模型配置
        checkpoint['model_config'] = self.model_config

        # 额外保存 DepthMaster 格式的 checkpoint（仅 rank 0）
        if self.global_rank == 0:
            ckpt_dir = self.trainer.checkpoint_callback.dirpath if self.trainer.checkpoint_callback else None
            if ckpt_dir is not None:
                depthmaster_ckpt_path = Path(ckpt_dir) / f"depthmaster_step_{self.global_step:08d}.pt"
                torch.save({
                    'model_config': self.model_config,
                    'model': self.model.state_dict(),
                }, depthmaster_ckpt_path)
                log.info(f"[DepthMasterWrapper] 已保存 DepthMaster 格式 checkpoint: {depthmaster_ckpt_path}")

                # 保存 EMA 模型
                if self._enable_ema and self._ema_model is not None:
                    ema_ckpt_path = Path(ckpt_dir) / f"depthmaster_step_{self.global_step:08d}_ema.pt"
                    torch.save({
                        'model_config': self.model_config,
                        'model': self._ema_model.module.state_dict(),
                    }, ema_ckpt_path)
                    log.info(f"[DepthMasterWrapper] 已保存 EMA checkpoint: {ema_ckpt_path}")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        加载 checkpoint 时的钩子。

        支持从 DepthMaster 的 .safetensors 或 .pt 格式恢复。
        """
        # Lightning checkpoint 中 state_dict 的 key 带有 'model.' 前缀
        # 这由 Lightning 自动处理，无需额外操作
        state_dict = checkpoint.get('state_dict', {})

        # 如果 checkpoint 中有 EMA 权重，恢复 EMA 模型
        if 'ema_model_state_dict' in checkpoint and self._enable_ema:
            # EMA 模型在 on_fit_start 中初始化，这里先保存状态
            self._pending_ema_state_dict = checkpoint['ema_model_state_dict']

    def on_train_start(self):
        """训练开始后恢复 EMA 状态。"""
        if hasattr(self, '_pending_ema_state_dict') and self._ema_model is not None:
            self._ema_model.module.load_state_dict(self._pending_ema_state_dict, strict=False)
            del self._pending_ema_state_dict
            log.info("[DepthMasterWrapper] EMA 模型状态已恢复")

    @staticmethod
    def _save_ply(path: str, points: np.ndarray, colors: np.ndarray):
        """保存点云为 PLY 文件（binary little-endian 格式）。"""
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
        pts = points[valid].astype(np.float32)
        cols = colors[valid].astype(np.uint8)
        N = pts.shape[0]
        if N == 0:
            return
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {N}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        ).encode('ascii')
        data = np.zeros(N, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        data['x'] = pts[:, 0]
        data['y'] = pts[:, 1]
        data['z'] = pts[:, 2]
        data['red'] = cols[:, 0]
        data['green'] = cols[:, 1]
        data['blue'] = cols[:, 2]
        with open(path, 'wb') as f:
            f.write(header)
            f.write(data.tobytes())

    def _visualize(self, batch: Dict[str, Any], is_panorama_batch: bool):
        """
        可视化当前 batch 的预测结果，保存到 vis_log_dir。
        
        核心策略：复用训练 forward 的计算图中间结果进行可视化，
        避免额外的 model forward pass 导致 OOM（尤其是全景图模式）。
        
        - 透视图：保存 RGB、GT depth、Pred depth、Pred normal、Pred mask、点云 PLY
        - 全景图：保存每个 cubemap 面的 RGB/GT depth/Pred depth、ERP 全景拼接、世界坐标系点云 PLY
        """
        if self._vis_log_dir is None:
            return

        # 使用触发可视化时的 step 作为目录名，确保透视图和全景图保存到同一目录
        vis_step = self._vis_pending_step if self._vis_pending_step >= 0 else self.global_step
        save_dir = Path(self._vis_log_dir) / f'step_{vis_step:08d}'
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            if is_panorama_batch:
                self._visualize_panorama(batch, save_dir)
            else:
                self._visualize_perspective(batch, save_dir)
            log.info(f"[DepthMasterWrapper] 可视化已保存: {save_dir}")
        except Exception as e:
            import traceback
            log.warning(f"[DepthMasterWrapper] 可视化失败: {e}\n{traceback.format_exc()}")

    @torch.no_grad()
    def _visualize_perspective(self, batch: Dict[str, Any], save_dir: Path):
        """透视图 batch 的可视化（使用独立 forward pass，透视图不会 OOM）。"""
        device = self.device
        image = batch['image'].to(device)
        gt_depth = batch['depth'].to(device)
        gt_intrinsics = batch['intrinsics'].to(device)

        # 使用固定的 num_tokens 进行推理
        num_tokens = self.model_config['num_tokens_range'][0]

        # Forward
        self.model.eval()
        output = self.model(image, num_tokens=num_tokens)
        self.model.train()

        pred_points = output.get('points', None)
        pred_mask = output.get('mask', None)
        pred_normal = output.get('normal', None)

        # 转为 numpy（注意：模型输出可能是 bfloat16，需先转 float32）
        image_np = (image.float().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        gt_depth_np = gt_depth.float().cpu().numpy()
        pred_points_np = pred_points.float().cpu().numpy() if pred_points is not None else None
        pred_depth_np = pred_points_np[..., 2] if pred_points_np is not None else None
        pred_mask_np = pred_mask.float().cpu().numpy() if pred_mask is not None else None
        pred_normal_np = pred_normal.float().cpu().numpy() if pred_normal is not None else None

        batch_size = image_np.shape[0]
        # 累积保存：从当前已保存数量开始编号，直到凑够 _vis_num_samples
        remaining = self._vis_num_samples - self._vis_persp_count
        num_to_save = min(batch_size, remaining)
        for i in range(num_to_save):
            idx = self._vis_persp_count + i
            inst_dir = save_dir / f'{idx:04d}_persp'
            inst_dir.mkdir(parents=True, exist_ok=True)

            # RGB
            cv2.imwrite(str(inst_dir / 'image.jpg'),
                        cv2.cvtColor(image_np[i], cv2.COLOR_RGB2BGR))

            # GT depth
            gt_dep_i = gt_depth_np[i]
            gt_mask = np.isfinite(gt_dep_i) & (gt_dep_i > 0)
            cv2.imwrite(str(inst_dir / 'gt_depth.png'),
                        cv2.cvtColor(colorize_depth(gt_dep_i, gt_mask), cv2.COLOR_RGB2BGR))

            # Pred depth
            if pred_depth_np is not None:
                pred_dep_i = pred_depth_np[i]
                pred_valid = np.isfinite(pred_dep_i) & (pred_dep_i > 0)
                # 不乘 mask 的原始 depth
                cv2.imwrite(str(inst_dir / 'pred_depth_raw.png'),
                            cv2.cvtColor(colorize_depth(pred_dep_i, pred_valid), cv2.COLOR_RGB2BGR))
                # 乘 mask 的最终版本
                if pred_mask_np is not None:
                    masked_valid = pred_valid & (pred_mask_np[i] > 0.5)
                    cv2.imwrite(str(inst_dir / 'pred_depth.png'),
                                cv2.cvtColor(colorize_depth(pred_dep_i, masked_valid), cv2.COLOR_RGB2BGR))

            # Pred mask
            if pred_mask_np is not None:
                mask_vis = (pred_mask_np[i].clip(0, 1) * 255).astype(np.uint8)
                cv2.imwrite(str(inst_dir / 'pred_mask.png'), mask_vis)

            # Pred normal
            if pred_normal_np is not None:
                normal_vis = colorize_normal(pred_normal_np[i],
                                            pred_mask_np[i] if pred_mask_np is not None else None)
                cv2.imwrite(str(inst_dir / 'pred_normal.png'),
                            cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR))

            # 点云 PLY（相机坐标系）
            if pred_points_np is not None:
                pts_i = pred_points_np[i].reshape(-1, 3)
                colors_i = image_np[i].reshape(-1, 3)
                self._save_ply(str(inst_dir / 'pred_points.ply'), pts_i, colors_i)

        # 更新累积计数器
        self._vis_persp_count += num_to_save
        if self._vis_persp_count >= self._vis_num_samples:
            self._vis_pending_persp = False

    @torch.no_grad()
    def _visualize_panorama(self, batch: Dict[str, Any], save_dir: Path):
        """
        全景图 batch 的可视化。
        
        核心改进：复用训练 forward 的输出，不再做额外的 model forward pass，
        避免全景图推理时的 OOM 问题。
        """
        device = self.device

        # GPU E2C 模式：先在 GPU 上完成 ERP → Cubemap 投影
        if batch.get('camera_type') == 'Panorama_ERP' and self._e2c_gpu is not None:
            erp_images = [img.to(device) for img in batch['erp_image']]
            erp_depths = [dep.to(device) for dep in batch['erp_depth']]
            max_depths = batch['max_depth'].to(device)
            B_pano = len(erp_images)

            gpu_result = self._e2c_gpu.process_panorama_batch(erp_images, erp_depths, max_depths)

            image = gpu_result['image']
            cubemap_intrinsics = gpu_result['intrinsics'].unsqueeze(0).expand(B_pano, -1, -1, -1)
            cubemap_W2C = gpu_result['W2C'].unsqueeze(0).expand(B_pano, -1, -1, -1)
            gt_depth = gpu_result['depth']
            gt_cubemap_points_world = gpu_result['cubemap_points_world']
            del erp_images, erp_depths, gpu_result
        else:
            image = batch['image'].to(device)
            cubemap_intrinsics = batch['intrinsics'].to(device)
            cubemap_W2C = batch['W2C'].to(device)
            gt_depth = batch['depth'].to(device)
            gt_cubemap_points_world = batch['cubemap_points_world'].to(device)

        B_vis = image.shape[0]
        face_names = ['front', 'right', 'back', 'left', 'up', 'down']

        # Re-use the training forward pass: run another forward with the current batch.
        # Even though training already did a forward, this no_grad pass does not store
        # activations, so its peak memory is far below the training forward+backward.
        extra_kwargs = {
            'W2C': cubemap_W2C,
            'intrinsics': cubemap_intrinsics,
        }

        self.model.eval()
        output = self.model(image, num_tokens=0, camera_type="Panorama", **extra_kwargs)
        self.model.train()

        pred_points_all = output.get('points', None)  # (B, 6, H, W, 3) 世界坐标系
        pred_mask_all = output.get('mask', None)      # (B, 6, H, W)
        pred_depth_all = output.get('depth', None)    # (B, 6, H, W) depth head direct prediction

        # 累积保存：从当前已保存数量开始编号，直到凑够 _vis_num_samples
        remaining_pano = self._vis_num_samples - self._vis_pano_count
        num_to_save_pano = min(B_vis, remaining_pano)
        for i_instance in range(num_to_save_pano):
            idx = self._vis_pano_count + i_instance
            inst_dir = save_dir / f'{idx:04d}_pano'
            inst_dir.mkdir(parents=True, exist_ok=True)

            # 提取当前 instance 的预测
            pred_points_pano = pred_points_all[i_instance] if pred_points_all is not None else None  # (6, H, W, 3)
            pred_mask_pano = pred_mask_all[i_instance] if pred_mask_all is not None else None  # (6, H, W)

            # W2C 矩阵
            W2C_i = cubemap_W2C[i_instance].float().cpu().numpy()  # (6, 4, 4)
            R = W2C_i[:, :3, :3]  # (6, 3, 3)
            t = W2C_i[:, :3, 3]   # (6, 3)

            # 预测世界系点云 → 相机系点云（per-face）
            if pred_points_pano is not None:
                pred_pts_world_np = pred_points_pano.detach().float().cpu().numpy()  # (6, H, W, 3)
                pred_pts_cam = np.einsum('vij,vhwj->vhwi', R, pred_pts_world_np) + t[:, None, None, :]
            else:
                pred_pts_world_np = None
                pred_pts_cam = None

            # GT depth
            gt_depth_i = gt_depth[i_instance].float().cpu().numpy()  # (6, H, W)

            # 保存每个 cubemap 面
            for fi in range(6):
                # RGB
                face_img = (image[i_instance, fi].float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                cv2.imwrite(str(inst_dir / f'face_{fi}_{face_names[fi]}_rgb.jpg'),
                            cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

                # GT depth
                face_gt_dep = gt_depth_i[fi]
                face_gt_mask = np.isfinite(face_gt_dep) & (face_gt_dep > 0)
                cv2.imwrite(str(inst_dir / f'face_{fi}_{face_names[fi]}_gt_depth.png'),
                            cv2.cvtColor(colorize_depth(face_gt_dep, face_gt_mask), cv2.COLOR_RGB2BGR))

                # Pred depth (z-buffer from camera space)
                if pred_pts_cam is not None:
                    face_pred_z = pred_pts_cam[fi, :, :, 2]
                    raw_valid = np.isfinite(face_pred_z) & (face_pred_z > 0)
                    cv2.imwrite(str(inst_dir / f'face_{fi}_{face_names[fi]}_pred_depth_raw.png'),
                                cv2.cvtColor(colorize_depth(face_pred_z, raw_valid), cv2.COLOR_RGB2BGR))

                    # 乘 pred_mask 的版本
                    if pred_mask_pano is not None:
                        face_pred_mask_np = pred_mask_pano[fi].detach().float().cpu().numpy()
                        face_pred_mask = raw_valid & (face_pred_mask_np > 0.5)
                        cv2.imwrite(str(inst_dir / f'face_{fi}_{face_names[fi]}_pred_depth.png'),
                                    cv2.cvtColor(colorize_depth(face_pred_z, face_pred_mask), cv2.COLOR_RGB2BGR))

                # Pred mask 概率图
                if pred_mask_pano is not None:
                    face_mask_prob = pred_mask_pano[fi].detach().float().cpu().numpy().clip(0, 1)
                    cv2.imwrite(str(inst_dir / f'face_{fi}_{face_names[fi]}_pred_mask.png'),
                                (face_mask_prob * 255).astype(np.uint8))

            # 保存世界坐标系点云 PLY（所有 6 面合并）
            if pred_pts_world_np is not None:
                all_pts = pred_pts_world_np.reshape(-1, 3)
                all_colors = np.zeros_like(all_pts, dtype=np.uint8)
                for fi in range(6):
                    face_img = (image[i_instance, fi].float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    H_f, W_f = face_img.shape[:2]
                    all_colors[fi * H_f * W_f:(fi + 1) * H_f * W_f] = face_img.reshape(-1, 3)
                # 只过滤 nan/inf，不用 mask 过滤（训练初期 mask 可能全为 0 导致点云为空）
                valid_pred = np.all(np.isfinite(all_pts), axis=-1)
                self._save_ply(str(inst_dir / 'pred_pointcloud.ply'), all_pts[valid_pred], all_colors[valid_pred])

            # 保存 depth_head 反投影到世界系的点云 PLY（depth-to-points）
            # 与 losses.py::depth_to_points_global_loss 中的反投影几何完全一致：
            #   p_cam = ((u-cx)/fx * z, (v-cy)/fy * z, z),  z = pred_zbuf
            #   p_world = R^T (p_cam - t)
            # 用于直观对比 depth head 的预测与 point head 的预测之间的几何差异。
            if pred_depth_all is not None:
                pred_depth_pano = pred_depth_all[i_instance].detach().float().cpu().numpy()  # (6, H, W)
                K_i = cubemap_intrinsics[i_instance].float().cpu().numpy()  # (6, 3, 3)
                _, H_f, W_f = pred_depth_pano.shape

                # 归一化像素中心坐标（与 depth_to_points_global_loss 一致）
                u_n = (np.arange(W_f, dtype=np.float32) + 0.5) / W_f  # (W,)
                v_n = (np.arange(H_f, dtype=np.float32) + 0.5) / H_f  # (H,)
                vv_n, uu_n = np.meshgrid(v_n, u_n, indexing='ij')      # (H, W)

                pred_pts_world_from_depth = np.zeros((6, H_f, W_f, 3), dtype=np.float32)
                for fi in range(6):
                    fx, fy = K_i[fi, 0, 0], K_i[fi, 1, 1]
                    cx, cy = K_i[fi, 0, 2], K_i[fi, 1, 2]
                    z_cam = pred_depth_pano[fi]                                     # (H, W)
                    x_cam = (uu_n - cx) / max(fx, 1e-6) * z_cam
                    y_cam = (vv_n - cy) / max(fy, 1e-6) * z_cam
                    p_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)                # (H, W, 3)
                    # cam → world: p_world = R^T (p_cam - t)
                    p_world = (p_cam - t[fi][None, None, :]) @ R[fi]                # R^T 应用：(H,W,3) @ (3,3) = (p_cam - t) · R = R^T·(p_cam - t)
                    pred_pts_world_from_depth[fi] = p_world

                depth_pts_flat = pred_pts_world_from_depth.reshape(-1, 3)
                depth_colors_flat = np.zeros_like(depth_pts_flat, dtype=np.uint8)
                for fi in range(6):
                    face_img = (image[i_instance, fi].float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    H_img, W_img = face_img.shape[:2]
                    # 若 face_img 与 depth 分辨率不一致，先 resize 到 depth 的 (H_f, W_f)
                    if (H_img, W_img) != (H_f, W_f):
                        face_img = cv2.resize(face_img, (W_f, H_f), interpolation=cv2.INTER_LINEAR)
                    depth_colors_flat[fi * H_f * W_f:(fi + 1) * H_f * W_f] = face_img.reshape(-1, 3)
                valid_depth_pts = (
                    np.all(np.isfinite(depth_pts_flat), axis=-1)
                    & (pred_depth_pano.reshape(-1) > 0)
                )
                self._save_ply(
                    str(inst_dir / 'pred_pointcloud_depth.ply'),
                    depth_pts_flat[valid_depth_pts],
                    depth_colors_flat[valid_depth_pts],
                )

            # 保存 GT 世界坐标系点云 PLY
            gt_pts_world_i = gt_cubemap_points_world[i_instance].float().cpu().numpy()  # (6, H, W, 3)
            gt_pts_flat = gt_pts_world_i.reshape(-1, 3)
            gt_rgb_flat = np.zeros_like(gt_pts_flat, dtype=np.uint8)
            for fi in range(6):
                face_img = (image[i_instance, fi].float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                H_f, W_f = face_img.shape[:2]
                gt_rgb_flat[fi * H_f * W_f:(fi + 1) * H_f * W_f] = face_img.reshape(-1, 3)
            gt_depth_flat = gt_depth_i.reshape(-1)
            valid_gt = (gt_depth_flat > 0) & np.isfinite(gt_depth_flat) & np.all(np.isfinite(gt_pts_flat), axis=-1)
            self._save_ply(str(inst_dir / 'gt_pointcloud.ply'), gt_pts_flat[valid_gt], gt_rgb_flat[valid_gt])

            # 拼接 ERP 全景图
            if pred_pts_cam is not None:
                H_face, W_face = pred_pts_cam.shape[1], pred_pts_cam.shape[2]
                face_w_orig = 512  # cubemap 标准面尺寸

                # 预测 range depth = ||p_cam||
                pred_range_all = np.linalg.norm(pred_pts_cam, axis=-1)  # (6, H, W)
                pred_invalid = (~np.isfinite(pred_range_all)) | (pred_pts_cam[..., 2] <= 0)
                pred_range_all = np.where(pred_invalid, 0.0, pred_range_all).astype(np.float32)

                # GT range depth
                gt_pts_world_np = gt_cubemap_points_world[i_instance].float().cpu().numpy()  # (6, H, W, 3)
                gt_pts_cam = np.einsum('vij,vhwj->vhwi', R, gt_pts_world_np) + t[:, None, None, :]
                gt_range_all = np.linalg.norm(gt_pts_cam, axis=-1)  # (6, H, W)
                gt_invalid = (~np.isfinite(gt_range_all)) | (gt_pts_cam[..., 2] <= 0)
                gt_range_all = np.where(gt_invalid, 0.0, gt_range_all).astype(np.float32)

                # Resize to standard cubemap face size
                pred_range_faces = np.zeros((6, face_w_orig, face_w_orig), dtype=np.float32)
                gt_range_faces = np.zeros((6, face_w_orig, face_w_orig), dtype=np.float32)
                rgb_faces = np.zeros((6, face_w_orig, face_w_orig, 3), dtype=np.float32)

                for fi in range(6):
                    pred_range_faces[fi] = cv2.resize(pred_range_all[fi], (face_w_orig, face_w_orig), interpolation=cv2.INTER_LINEAR)
                    gt_range_faces[fi] = cv2.resize(gt_range_all[fi], (face_w_orig, face_w_orig), interpolation=cv2.INTER_LINEAR)
                    face_rgb = (image[i_instance, fi].float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.float32)
                    rgb_faces[fi] = cv2.resize(face_rgb, (face_w_orig, face_w_orig), interpolation=cv2.INTER_LINEAR)

                # Cubemap → ERP
                pred_erp_depth = cubemap_to_equirect_np(pred_range_faces)
                pred_erp_mask = pred_erp_depth > 0
                cv2.imwrite(str(inst_dir / 'erp_pred_depth.png'),
                            cv2.cvtColor(colorize_depth(pred_erp_depth, pred_erp_mask), cv2.COLOR_RGB2BGR))

                gt_erp_depth = cubemap_to_equirect_np(gt_range_faces)
                gt_erp_mask = gt_erp_depth > 0
                cv2.imwrite(str(inst_dir / 'erp_gt_depth.png'),
                            cv2.cvtColor(colorize_depth(gt_erp_depth, gt_erp_mask), cv2.COLOR_RGB2BGR))

                erp_rgb = cubemap_to_equirect_np(rgb_faces).astype(np.uint8)
                cv2.imwrite(str(inst_dir / 'erp_rgb.jpg'),
                            cv2.cvtColor(erp_rgb, cv2.COLOR_RGB2BGR))

            # Depth head direct prediction → ERP depth
            # Note: depth head outputs z-buffer depth, need to convert to range depth before ERP projection
            if pred_depth_all is not None:
                pred_depth_faces = pred_depth_all[i_instance].detach().float().cpu().numpy()  # (6, H, W) z-buffer
                face_w_orig = 512
                fov_deg = 95.0
                f_norm = 0.5 / np.tan(np.deg2rad(fov_deg) / 2)
                u_vis = np.linspace(0, 1, face_w_orig, dtype=np.float32)
                v_vis = np.linspace(0, 1, face_w_orig, dtype=np.float32)
                uu_vis, vv_vis = np.meshgrid(u_vis, v_vis)
                ray_len_vis = np.sqrt(((uu_vis - 0.5) / f_norm) ** 2 + ((vv_vis - 0.5) / f_norm) ** 2 + 1.0)
                pred_depth_faces_resized = np.zeros((6, face_w_orig, face_w_orig), dtype=np.float32)
                for fi in range(6):
                    zbuf_resized = cv2.resize(
                        pred_depth_faces[fi], (face_w_orig, face_w_orig), interpolation=cv2.INTER_LINEAR
                    )
                    pred_depth_faces_resized[fi] = zbuf_resized * ray_len_vis  # z-buffer → range depth
                pred_depth_erp = cubemap_to_equirect_np(pred_depth_faces_resized)
                pred_depth_erp_mask = pred_depth_erp > 0
                cv2.imwrite(str(inst_dir / 'erp_pred_depth_head.png'),
                            cv2.cvtColor(colorize_depth(pred_depth_erp, pred_depth_erp_mask), cv2.COLOR_RGB2BGR))

        # 更新累积计数器
        self._vis_pano_count += num_to_save_pano
        if self._vis_pano_count >= self._vis_num_samples:
            self._vis_pending_pano = False

        # 释放
        del output, pred_points_all, pred_mask_all, pred_depth_all
        torch.cuda.empty_cache()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """验证步骤（可选，暂时只计算 loss）。"""
        # 简单的验证：计算 loss 但不反向传播
        with torch.no_grad():
            is_panorama_batch = batch.get('camera_type') in ('Panorama', 'Panorama_ERP')
            label_type = batch.get('label_type', ['A'] * batch['image'].shape[0])
            is_metric = batch.get('is_metric', [True] * batch['image'].shape[0])

            if is_panorama_batch:
                loss, _ = self._training_step_panorama(batch, 0, label_type, is_metric)
            else:
                num_tokens = self.model_config['num_tokens_range'][1]
                loss, _ = self._training_step_perspective(batch, num_tokens, label_type, is_metric)

            self.log("val/loss", loss.item(), prog_bar=True, sync_dist=False)

    @classmethod
    def from_config(cls, config_path: str, **override_kwargs) -> "DepthMasterWrapper":
        """
        Build a Wrapper instance from a DepthMaster JSON config file.

        Args:
            config_path: Path to a DepthMaster JSON config (e.g. ``depthmaster_train.json``).
            **override_kwargs: Override any field defined in the config.

        Returns:
            A ``DepthMasterWrapper`` instance.
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        kwargs = {
            'model': config['model'],
            'loss': config['loss'],
            'optimizer': config['optimizer'],
            'lr_scheduler': config['lr_scheduler'],
            'ccl_loss': config.get('ccl_loss', None),
            'mirror_pretrained': config.get('mirror_pretrained', None),
            'low_resolution_training_steps': config.get('low_resolution_training_steps', 0),
        }
        kwargs.update(override_kwargs)
        return cls(**kwargs)
