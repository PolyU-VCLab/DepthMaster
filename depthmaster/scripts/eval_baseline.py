import os
import sys
from pathlib import Path
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
import json
from typing import *
import importlib
import importlib.util

import click


@click.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, help='Evaluation script for DepthMaster on perspective benchmarks.')
@click.option('--baseline', 'baseline_code_path', type=click.Path(), required=True, help='Path to the baseline model python code.')
@click.option('--config', 'config_path', type=click.Path(), default='configs/eval/all_benchmarks.json', help='Path to the evaluation configurations. '
    'Defaults to "configs/eval/all_benchmarks.json".')
@click.option('--output', '-o', 'output_path',  type=click.Path(), required=True, help='Path to the output json file.')
@click.option('--oracle', 'oracle_mode', is_flag=True, help='Use oracle mode for evaluation, i.e., use the GT intrinsics input.')
@click.option('--dump_pred', is_flag=True, help='Dump predition results.')
@click.option('--dump_gt', is_flag=True, help='Dump ground truth.')
@click.option('--save_per_sample', is_flag=True, help='Save per-sample detailed metrics to a separate json file.')
@click.pass_context
def main(ctx: click.Context, baseline_code_path: str, config_path: str, oracle_mode: bool, output_path: Union[str, Path], dump_pred: bool, dump_gt: bool, save_per_sample: bool):
    # Lazy import
    import  cv2
    import numpy as np
    from tqdm import tqdm
    import torch
    import torch.nn.functional as F
    import utils3d

    from depthmaster.test.baseline import MGEBaselineInterface
    from depthmaster.test.dataloader import EvalDataLoaderPipeline
    from depthmaster.test.metrics import compute_metrics
    from depthmaster.utils.vis import colorize_depth, colorize_normal, colorize_disparity
    from depthmaster.utils.tools import key_average, flatten_nested_dict, timeit, import_file_as_module
    from depthmaster.utils.io import save_ply
    
    # Load the baseline model
    module = import_file_as_module(baseline_code_path, Path(baseline_code_path).stem)
    baseline_cls: Type[MGEBaselineInterface] = getattr(module, 'Baseline')
    baseline : MGEBaselineInterface = baseline_cls.load.main(ctx.args, standalone_mode=False)

    # Load the evaluation configurations
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    per_sample_metrics = {}  # Stores per-sample detailed metrics.

    # Per-sample output folder. Each evaluation run produces one folder, one
    # json per dataset. Example: output=eval_output/foo.json
    #                     ->     per_sample_dir=eval_output/foo_per_sample/
    per_sample_dir = Path(output_path).with_name(Path(output_path).stem + '_per_sample')
    if save_per_sample:
        per_sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"[per_sample] saving to: {per_sample_dir}")

    # Iterate over the dataset
    for benchmark_name, benchmark_config in tqdm(list(config.items()), desc='Benchmarks'):
        filenames, metrics_list = [], []
        with (
            EvalDataLoaderPipeline(**benchmark_config) as eval_data_pipe,
            tqdm(total=len(eval_data_pipe), desc=benchmark_name, leave=False) as pbar
        ):  
            # Iterate over the samples in the dataset
            for i in range(len(eval_data_pipe)):
                sample = eval_data_pipe.get()
                sample = {k: v.to(baseline.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                image = sample['image']
                gt_intrinsics = sample['intrinsics']

                # Inference
                torch.cuda.synchronize()
                with torch.inference_mode(), timeit('_inference_timer', verbose=False) as timer:
                    if oracle_mode:
                        pred = baseline.infer_for_evaluation(image, gt_intrinsics)
                    else:
                        pred = baseline.infer_for_evaluation(image)
                    torch.cuda.synchronize()

                # Compute metrics
                metrics, misc = compute_metrics(pred, sample, vis=True)
                metrics['inference_time'] = timer.time
                metrics_list.append(metrics)

                # Save the detailed metrics for the current sample.
                if save_per_sample:
                    sample_name = sample['filename']
                    if benchmark_name not in per_sample_metrics:
                        per_sample_metrics[benchmark_name] = {}
                    per_sample_metrics[benchmark_name][sample_name] = metrics

                # Dump results
                dump_path = Path(output_path.replace(".json", f"_dump"), f'{benchmark_name}', sample['filename'].replace('.zip', ''))
                if dump_pred:
                    dump_path.joinpath('pred').mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dump_path / 'pred' / 'image.jpg'), cv2.cvtColor((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    with Path(dump_path, 'pred', 'metrics.json').open('w') as f:
                        json.dump(metrics, f, indent=4)

                    if 'pred_points' in misc:
                        points = misc['pred_points'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'points.exr'), cv2.cvtColor(points.astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                    
                    if 'pred_depth' in misc:
                        depth = misc['pred_depth'].cpu().numpy()
                        if 'mask' in pred:
                            mask = pred['mask'].cpu().numpy()
                            depth = np.where(mask, depth, np.inf)
                        cv2.imwrite(str(dump_path / 'pred' / 'depth.png'), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))

                        # Save the disparity visualization.
                        pred_disp = 1.0 / np.where((depth > 0) & np.isfinite(depth), depth, np.nan)
                        cv2.imwrite(str(dump_path / 'pred' / 'disparity.png'), cv2.cvtColor(colorize_disparity(pred_disp), cv2.COLOR_RGB2BGR))

                    if 'mask' in pred:
                        mask = pred['mask'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'mask.png'), (mask * 255).astype(np.uint8))

                    if 'normal' in pred:
                        normal = pred['normal'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'normal.png'), cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR))

                    # Save the predicted point cloud as a PLY file.
                    if 'pred_points' in misc:
                        try:
                            pred_pts = misc['pred_points'].cpu().numpy()  # (H, W, 3)
                            img_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)  # (H, W, 3)
                            H, W = pred_pts.shape[:2]
                            # Build the validity mask.
                            valid = np.isfinite(pred_pts).all(axis=-1)
                            if 'mask' in pred:
                                valid = valid & pred['mask'].cpu().numpy()
                            vertices = pred_pts[valid]
                            colors = img_np[:H, :W][valid]
                            # Save as a vertex-only PLY (no faces).
                            faces = np.zeros((0, 3), dtype=np.int32)
                            save_ply(str(dump_path / 'pred' / 'pointcloud.ply'), vertices, faces, colors)
                        except Exception as e:
                            print(f"  Failed to save predicted point cloud PLY: {e}")
                
                if dump_gt:
                    dump_path.joinpath('gt').mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dump_path / 'gt' / 'image.jpg'), cv2.cvtColor((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    if 'points' in sample:
                        points = sample['points']
                        cv2.imwrite(str(dump_path / 'gt' / 'points.exr'), cv2.cvtColor(points.cpu().numpy().astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

                    if 'depth' in sample:
                        depth = sample['depth']
                        mask = sample['depth_mask']
                        cv2.imwrite(str(dump_path / 'gt' / 'depth.png'), cv2.cvtColor(colorize_depth(depth.cpu().numpy(), mask=mask.cpu().numpy()), cv2.COLOR_RGB2BGR))

                        # Save the GT disparity visualization.
                        gt_depth_np = depth.cpu().numpy()
                        gt_mask_np = mask.cpu().numpy()
                        gt_disp = 1.0 / np.where((gt_depth_np > 0) & np.isfinite(gt_depth_np) & gt_mask_np, gt_depth_np, np.nan)
                        cv2.imwrite(str(dump_path / 'gt' / 'disparity.png'), cv2.cvtColor(colorize_disparity(gt_disp), cv2.COLOR_RGB2BGR))

                    if 'normal' in sample:
                        normal = sample['normal']
                        cv2.imwrite(str(dump_path / 'gt' / 'normal.png'), cv2.cvtColor(colorize_normal(normal.cpu().numpy()), cv2.COLOR_RGB2BGR))

                    if 'depth_mask' in sample:
                        mask = sample['depth_mask']
                        cv2.imwrite(str(dump_path / 'gt' /'mask.png'), (mask.cpu().numpy() * 255).astype(np.uint8))

                    # Save the GT point cloud as a PLY file.
                    if 'points' in sample and 'depth_mask' in sample:
                        try:
                            gt_pts = sample['points'].cpu().numpy()  # (H, W, 3)
                            img_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                            gt_mask_np = sample['depth_mask'].cpu().numpy()
                            H, W = gt_pts.shape[:2]
                            valid = np.isfinite(gt_pts).all(axis=-1) & gt_mask_np
                            vertices = gt_pts[valid]
                            colors = img_np[:H, :W][valid]
                            faces = np.zeros((0, 3), dtype=np.int32)
                            save_ply(str(dump_path / 'gt' / 'pointcloud.ply'), vertices, faces, colors)
                        except Exception as e:
                            print(f"  Failed to save GT point cloud PLY: {e}")

                # Save intermediate results
                if i % 100 == 0 or i == len(eval_data_pipe) - 1:
                    Path(output_path).write_text(
                        json.dumps({
                            **all_metrics, 
                            benchmark_name: key_average(metrics_list)
                        }, indent=4)
                    )
                pbar.update(1)

            all_metrics[benchmark_name] = key_average(metrics_list)

            # As soon as a benchmark finishes, dump the per-sample json for that dataset.
            if save_per_sample and benchmark_name in per_sample_metrics:
                bench_path = per_sample_dir / f'{benchmark_name}.json'
                bench_path.write_text(json.dumps(per_sample_metrics[benchmark_name], indent=4))

    # Save final results
    all_metrics['mean'] = key_average(list(all_metrics.values()))
    Path(output_path).write_text(json.dumps(all_metrics, indent=4))
    
    # Save the per-sample detailed metrics. Each benchmark already wrote its own
    # json into per_sample_dir/<benchmark>.json above; here we additionally write
    # a single aggregated file for convenience.
    if save_per_sample:
        per_sample_output_path = per_sample_dir / 'all_benchmarks.json'
        per_sample_output_path.write_text(json.dumps(per_sample_metrics, indent=4))
        print(f"Per-sample metrics saved to: {per_sample_dir} (each benchmark + all_benchmarks.json)")


if __name__ == '__main__':
    main()
