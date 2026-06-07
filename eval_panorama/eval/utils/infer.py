# Authors: Jing He, Haodong Li
# Last modified: 2025-10-10

import sys
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from tabulate import tabulate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eval.datasets import (
    BaseDepthDataset,
    get_dataset,
    DatasetMode,
    get_pred_name
)
from eval.utils import (
    MetricTracker,
    metric,
    init_per_sample_csv,
    write_per_sample_csv,
    write_metrics_txt, 
    align_depth_least_square, 
    align_depth_median
)

# Use the DepthMaster alignment functions (scale-invariant and affine-invariant).
# The repository root (which contains the `depthmaster/` package) is expected to be
# on the Python path; eval_panorama/eval.py takes care of this when launched
# via `bash eval_panorama.sh`.
from depthmaster.utils.alignment import align_depth_scale as dm_align_depth_scale
from depthmaster.utils.alignment import align_depth_affine as dm_align_depth_affine
from utils3d.torch import masked_nearest_resize


# ==================== Metric depth metrics (DepthMaster style) ====================
def metric_abs_rel(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor):
    """Absolute Relative Error, no alignment applied."""
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    return (torch.abs(pred_valid - gt_valid) / gt_valid).mean()

def metric_delta1(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor):
    """Delta < 1.25 accuracy, no alignment applied."""
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    return (torch.maximum(gt_valid / pred_valid, pred_valid / gt_valid) < 1.25).float().mean()

def metric_rmse(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor):
    """Root Mean Squared Error, no alignment applied."""
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    return torch.sqrt(((pred_valid - gt_valid) ** 2).mean())

def metric_log_rmse(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor):
    """Log RMSE, no alignment applied."""
    pred_valid = pred[valid_mask].clamp_min(1e-6)
    gt_valid = gt[valid_mask].clamp_min(1e-6)
    return torch.sqrt(((torch.log(pred_valid) - torch.log(gt_valid)) ** 2).mean())


def resize_max_res(
    img: torch.Tensor,
    max_edge_resolution: int,
    resample_method: InterpolationMode = InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`torch.Tensor`):
            Image tensor to be resized. Expected shape: [B, C, H, W]
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
        resample_method (`PIL.Image.Resampling`):
            Resampling method used to resize images.

    Returns:
        `torch.Tensor`: Resized image.
    """
    assert 4 == img.dim(), f"Invalid input shape {img.shape}"

    original_height, original_width = img.shape[-2:]
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = resize(img, (new_height, new_width), resample_method, antialias=True)
    return resized_img


def _save_depth_visualization(depth_pred, depth_gt, valid_mask, sample_idx, rgb_name, vis_dir):
    """Save a side-by-side visualization of predicted and GT depth.

    Args:
        depth_pred: numpy array of predicted depth, shape ``(H, W)``.
        depth_gt:   numpy array of GT depth, shape ``(H, W)``.
        valid_mask: numpy array of the valid mask, shape ``(H, W)``.
        sample_idx: int, sample index.
        rgb_name:   str, RGB filename used as figure title.
        vis_dir:    str, output directory for the visualization.
    """
    # Use the valid region to determine a shared colorbar range.
    valid_gt = depth_gt[valid_mask > 0]
    valid_pred = depth_pred[valid_mask > 0]
    
    if len(valid_gt) == 0:
        return
    
    # Use the 2nd / 98th percentiles of the GT valid region as the colorbar
    # range to avoid being driven by outliers.
    vmin = float(np.percentile(valid_gt, 2))
    vmax = float(np.percentile(valid_gt, 98))
    
    # Mark invalid pixels as NaN so matplotlib renders them as white.
    pred_vis = depth_pred.copy().astype(np.float32)
    gt_vis = depth_gt.copy().astype(np.float32)
    pred_vis[valid_mask == 0] = np.nan
    gt_vis[valid_mask == 0] = np.nan
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Predicted depth.
    im0 = axes[0].imshow(pred_vis, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Pred Depth (range: [{vmin:.2f}, {vmax:.2f}])')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # GT depth.
    im1 = axes[1].imshow(gt_vis, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'GT Depth (range: [{vmin:.2f}, {vmax:.2f}])')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Use the filename as the overall title.
    short_name = os.path.basename(rgb_name) if rgb_name else f"sample_{sample_idx}"
    fig.suptitle(f'Sample {sample_idx}: {short_name}', fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(vis_dir, f'{sample_idx:04d}_{short_name.replace("/", "_").rsplit(".", 1)[0]}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def _align_depth(depth_pred, depth_raw, valid_mask, depth_raw_ts, valid_mask_ts, alignment, dataset_config, device, depth_pred_ts_gpu=None, dm_lr_cache=None):
    """Align the predicted depth to the GT depth.

    Args:
        depth_pred:        numpy array, raw predicted depth.
        depth_raw:         numpy array, GT depth.
        valid_mask:        numpy array, valid mask.
        depth_raw_ts:      torch.Tensor of GT depth (on device).
        valid_mask_ts:     torch.Tensor of valid mask (on device).
        alignment:         str, alignment method.
        dataset_config:    dict, dataset configuration.
        device:            torch.device.
        depth_pred_ts_gpu: optional pre-computed GPU tensor of ``depth_pred``,
            used to avoid repeating the host-to-device transfer.
        dm_lr_cache:       optional pre-computed ``(lr_mask, lr_index)`` cache
            so that ``masked_nearest_resize`` is not called repeatedly.

    Returns:
        aligned_depth: numpy array, the aligned depth.
    """
    depth_aligned = depth_pred.copy()
    
    if "least_square" == alignment:
        depth_aligned = np.clip(depth_aligned, a_min=-1e6, a_max=1e6)
        depth_aligned, _, _ = align_depth_least_square(
            gt_arr=depth_raw,
            pred_arr=depth_aligned,
            valid_mask_arr=valid_mask,
            return_scale_shift=True,
            max_resolution=dataset_config['alignment_max_res'],
        )
    elif "median" == alignment:
        depth_aligned = np.clip(depth_aligned, a_min=-1e6, a_max=1e6)
        depth_aligned, _, _ = align_depth_median(
            gt_arr=depth_raw,
            pred_arr=depth_aligned,
            valid_mask_arr=valid_mask,
            return_scale_shift=True,
        )
    elif "dm_scale" == alignment:
        # DepthMaster scale-invariant alignment: solve only for the optimal
        # scale s such that ``s * pred ~ gt``.
        # Reuse the pre-computed GPU tensor to skip another host-to-device copy.
        pred_ts = depth_pred_ts_gpu if depth_pred_ts_gpu is not None else torch.from_numpy(depth_aligned).float().to(device)
        pred_ts = pred_ts.clamp(-1e6, 1e6)
        gt_ts = depth_raw_ts.float()
        # Reuse the cached down-sampling indices to avoid recomputation.
        if dm_lr_cache is not None:
            lr_mask, lr_index = dm_lr_cache
        else:
            mask_2d = valid_mask_ts.bool()
            lr_mask, lr_index = masked_nearest_resize(mask=mask_2d, size=(64, 64), return_index=True)
        pred_lr_masked = pred_ts[lr_index][lr_mask]
        gt_lr_masked = gt_ts[lr_index][lr_mask]
        weight_lr = 1.0 / gt_lr_masked.clamp_min(1e-6)
        scale = dm_align_depth_scale(
            pred_lr_masked.reshape(1, -1),
            gt_lr_masked.reshape(1, -1),
            weight_lr.reshape(1, -1)
        )
        depth_aligned = depth_aligned * scale.item()
    elif "dm_affine" == alignment:
        # DepthMaster affine-invariant alignment: solve for ``scale`` and ``shift``
        # such that ``scale * pred + shift ~ gt``.
        # Reuse the pre-computed GPU tensor to skip another host-to-device copy.
        pred_ts = depth_pred_ts_gpu if depth_pred_ts_gpu is not None else torch.from_numpy(depth_aligned).float().to(device)
        pred_ts = pred_ts.clamp(-1e6, 1e6)
        gt_ts = depth_raw_ts.float()
        # Reuse the cached down-sampling indices.
        if dm_lr_cache is not None:
            lr_mask, lr_index = dm_lr_cache
        else:
            mask_2d = valid_mask_ts.bool()
            lr_mask, lr_index = masked_nearest_resize(mask=mask_2d, size=(64, 64), return_index=True)
        pred_lr_masked = pred_ts[lr_index][lr_mask]
        gt_lr_masked = gt_ts[lr_index][lr_mask]
        weight_lr = 1.0 / gt_lr_masked.clamp_min(1e-6)
        scale, shift = dm_align_depth_affine(
            pred_lr_masked.reshape(1, -1),
            gt_lr_masked.reshape(1, -1),
            weight_lr.reshape(1, -1)
        )
        depth_aligned = depth_aligned * scale.item() + shift.item()
    elif "metric" == alignment:
        pass
    else:
        raise NotImplementedError(f"Unknown alignment: {alignment}")
    
    return depth_aligned


def run_evaluation(model, config, dataset_name, output_dir, device, predict_fn=None):
    """Run evaluation. Multiple alignment methods can be evaluated in one pass
    so that inference is performed only once.

    Args:
        model:        Model object.
        config:       Configuration dict.
            * ``config['evaluation']['alignment']``: ``str`` or ``list[str]``.
              A single alignment (e.g. ``"dm_scale"``) or a list of
              alignments (e.g. ``["dm_scale", "dm_affine"]``).
            * ``config['evaluation']['metric_depth_eval']``: ``bool``.
              Whether to additionally compute metric-depth metrics (i.e.
              metrics computed without any alignment).
        dataset_name: Dataset name.
        output_dir:   Output directory.
        device:       Compute device.
        predict_fn:   Optional custom prediction function with signature
            ``predict_fn(model, rgb_int, device) -> depth_pred_numpy``.
            If ``None``, fall back to the default DA-2 model inference path.
    """
    eval_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(eval_dir): os.makedirs(eval_dir)

    dataset_config = config['evaluation']['datasets'][dataset_name]
    model_dtype = config.get('spherevit', {}).get('dtype', torch.float32)

    dataset: BaseDepthDataset = get_dataset(dataset_config, dataset_name, 
        base_data_dir=config['evaluation']['datasets_dir'], mode=DatasetMode.EVAL)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, persistent_workers=True)

    metric_funcs = [getattr(metric, _met) for _met in config['evaluation']['metric_names']]
    
    # Resolve the alignment configuration: accept a single string or a list.
    alignment_cfg = config['evaluation']['alignment']
    if isinstance(alignment_cfg, str):
        alignments = [alignment_cfg]
    else:
        alignments = alignment_cfg
    
    use_metric_depth = config['evaluation'].get('metric_depth_eval', False)
    
    # Build the full list of metric names. When multiple alignments are used,
    # each metric is prefixed with the alignment name.
    all_metric_names = []
    for align_name in alignments:
        for m in metric_funcs:
            # When there is a single alignment, drop the prefix to keep
            # backward compatibility with the legacy logic.
            if len(alignments) == 1:
                all_metric_names.append(m.__name__)
            else:
                all_metric_names.append(f"{align_name}/{m.__name__}")
    
    # Metric-depth metrics (no alignment).
    if use_metric_depth:
        all_metric_names += ['metric/abs_rel', 'metric/delta1', 'metric/rmse', 'metric/log_rmse']
    
    metric_tracker = MetricTracker(*all_metric_names)
    metric_tracker.reset()
    
    # ``per_sample_csv`` records the first alignment only.
    per_sample_csv = init_per_sample_csv(eval_dir, alignments[0], metric_funcs)

    # Depth visualization: save predicted-vs-GT depth maps for the first 20 samples.
    vis_dir = os.path.join(eval_dir, 'depth_vis')
    os.makedirs(vis_dir, exist_ok=True)
    max_vis_samples = 20  # Maximum number of samples to visualize.

    total_samples = len(dataloader)
    is_tty = sys.stderr.isatty()
    for sample_idx, data in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}", disable=not is_tty)):
        # GT data
        depth_raw_ts = data["depth_raw_linear"].squeeze()
        valid_mask_ts = data["valid_mask_raw"].squeeze()
        rgb_name = data["rgb_relative_path"][0]

        # Move data to GPU directly (use ``non_blocking`` to overlap copies).
        valid_mask_ts = valid_mask_ts.to(device, non_blocking=True)
        depth_raw_ts = depth_raw_ts.to(device, non_blocking=True)
        
        # Lazily fetch the numpy version (only used by alignment methods).
        depth_raw = depth_raw_ts.cpu().numpy()
        valid_mask = valid_mask_ts.cpu().numpy()

        # Get predictions
        rgb_basename = os.path.basename(rgb_name)
        pred_basename = get_pred_name(
            rgb_basename, dataset.name_mode, suffix=""
        )
        pred_name = os.path.join(os.path.dirname(rgb_name), pred_basename)

        if predict_fn is not None:
            # Use a custom prediction function (e.g. UniK3D, DepthMaster).
            depth_pred = predict_fn(model, data["rgb_int"], device)
        else:
            # Default DA-2 model inference path.
            input_size = data["rgb_int"].shape
            input_rgb = resize_max_res(
                data["rgb_int"],
                max_edge_resolution=1092,
            )

            input_rgb = input_rgb[0].to(device)
            input_rgb = input_rgb / 255.0
            input_rgb = input_rgb.to(model_dtype)

            depth_pred = model(input_rgb.unsqueeze(0))
            depth_pred = depth_pred.unsqueeze(0)
            depth_pred = resize(depth_pred, input_size[-2:], antialias=True)
            depth_pred = depth_pred.squeeze().cpu().numpy()

        # Make sure ``depth_pred`` is a numpy array with the same shape as GT.
        if isinstance(depth_pred, torch.Tensor):
            depth_pred = depth_pred.cpu().numpy()
        if depth_pred.shape != depth_raw.shape:
            # Resize the prediction to the GT size.
            depth_pred_ts = torch.from_numpy(depth_pred).float().unsqueeze(0).unsqueeze(0)
            depth_pred_ts = resize(depth_pred_ts, depth_raw.shape[-2:], antialias=True)
            depth_pred = depth_pred_ts.squeeze().numpy()

        # ==================== Save the depth visualization ====================
        if sample_idx < max_vis_samples:
            _save_depth_visualization(
                depth_pred, depth_raw, valid_mask,
                sample_idx, rgb_name, vis_dir
            )

        # ==================== Metric-depth evaluation (computed BEFORE alignment) ====================
        # Pre-load ``depth_pred`` to GPU once; the alignment code reuses it.
        depth_pred_ts_gpu = torch.from_numpy(depth_pred).float().to(device, non_blocking=True)
        
        if use_metric_depth:
            depth_pred_metric_ts = depth_pred_ts_gpu.clamp_min(1e-6)
            valid_mask_ts_bool = valid_mask_ts.bool()
            
            _m_abs_rel = metric_abs_rel(depth_pred_metric_ts, depth_raw_ts, valid_mask_ts_bool).item()
            _m_delta1 = metric_delta1(depth_pred_metric_ts, depth_raw_ts, valid_mask_ts_bool).item()
            _m_rmse = metric_rmse(depth_pred_metric_ts, depth_raw_ts, valid_mask_ts_bool).item()
            _m_log_rmse = metric_log_rmse(depth_pred_metric_ts, depth_raw_ts, valid_mask_ts_bool).item()
            
            metric_tracker.update('metric/abs_rel', _m_abs_rel)
            metric_tracker.update('metric/delta1', _m_delta1)
            metric_tracker.update('metric/rmse', _m_rmse)
            metric_tracker.update('metric/log_rmse', _m_log_rmse)

        # ==================== Compute metrics for each alignment method ====================
        # Pre-compute ``masked_nearest_resize`` once (shared by ``dm_scale`` and
        # ``dm_affine``).
        dm_lr_cache = None
        if any(a in ('dm_scale', 'dm_affine') for a in alignments):
            mask_2d = valid_mask_ts.bool()
            lr_mask, lr_index = masked_nearest_resize(mask=mask_2d, size=(64, 64), return_index=True)
            dm_lr_cache = (lr_mask, lr_index)
        
        first_aligned_metrics = []  # Cache the first-alignment metrics for the per-sample CSV.
        for idx, align_name in enumerate(alignments):
            # Align (using GPU tensors to accelerate dm_scale / dm_affine).
            depth_aligned = _align_depth(
                depth_pred, depth_raw, valid_mask,
                depth_raw_ts, valid_mask_ts,
                align_name, dataset_config, device,
                depth_pred_ts_gpu=depth_pred_ts_gpu,
                dm_lr_cache=dm_lr_cache
            )
            
            # Clip to dataset min max
            depth_aligned = np.clip(
                depth_aligned, a_min=dataset.min_depth, a_max=dataset.max_depth
            )
            # clip to d > 0 for evaluation
            depth_aligned = np.clip(depth_aligned, a_min=1e-6, a_max=None)

            # Evaluate
            depth_aligned_ts = torch.from_numpy(depth_aligned).float().to(device, non_blocking=True)

            for met_func in metric_funcs:
                _metric_name = met_func.__name__
                _metric = met_func(depth_aligned_ts, depth_raw_ts, valid_mask_ts).item()
                
                if len(alignments) == 1:
                    metric_tracker.update(_metric_name, _metric)
                else:
                    metric_tracker.update(f"{align_name}/{_metric_name}", _metric)
                
                # Cache the first-alignment results for the per-sample CSV.
                if idx == 0:
                    first_aligned_metrics.append(_metric.__str__())

        # ``per_sample_csv`` records only the first alignment (legacy compatibility).
        write_per_sample_csv(per_sample_csv, pred_name, first_aligned_metrics)

        # Print progress every 500 samples.
        if (sample_idx + 1) % 500 == 0 or (sample_idx + 1) == total_samples:
            print(f"  [{dataset_name}] progress: {sample_idx + 1}/{total_samples}", flush=True)

    # -------------------- Save metrics to file --------------------
    eval_text = tabulate(
        [metric_tracker.result().keys(), metric_tracker.result().values()]
    )
    alignment_str = "+".join(alignments) if len(alignments) > 1 else alignments[0]
    write_metrics_txt(eval_dir, alignment_str, eval_text)

    return metric_tracker.result()
