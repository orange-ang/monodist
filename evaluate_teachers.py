"""
Teacher accuracy evaluation on the KITTI eigen test split.

Mirrors evaluate_depth.py (median scaling + Garg crop + [1e-3, 80] GT mask),
but evaluates the TEACHER models instead of the student, reproducing the
training-time alignment pipeline:
  - UniDepth: metric output, only clamped (is_relative=False)
  - DA3: DA3METRIC metric output (focal-normalized -> meters), only
    clamped (is_relative=False), exactly as in trainer.py
  - Mean: pixelwise mean of the aligned teacher depths

Usage (run from the Lite-Mono-MonDI root):
    python evaluate_teachers.py \
        --data_path /path/to/kitti_raw \
        --teacher_da3_path /path/to/da3metric_large.safetensors \
        --teacher_unidepth_path /path/to/unidepth_v2_vitl14.pth

Add --disable_median_scaling to report raw metric accuracy (no GT rescaling).
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import datasets
from utils import readlines
from teacher_models import load_teacher_models
from teacher_reprojection import align_teacher_depths


cv2.setNumThreads(0)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

MIN_DEPTH = 1e-3
MAX_DEPTH = 80  # KITTI eigen evaluation range


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate teacher models on KITTI")
    parser.add_argument("--data_path", type=str, required=True,
                        help="path to KITTI raw dataset")
    parser.add_argument("--eval_split", type=str, default="eigen")

    # Dataset resolution (teachers resize internally; only affects the
    # resolution at which aligned depths are stored before GT-resize)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=640)

    # Teacher checkpoints
    parser.add_argument("--teacher_da3_path", type=str, default=None,
                        help="path to Depth-Anything-3 (da3metric-large) "
                             "checkpoint (.safetensors)")
    parser.add_argument("--teacher_da3_model_name", type=str, default="da3metric-large",
                        help="Depth-Anything-3 preset name")
    parser.add_argument("--teacher_unidepth_path", type=str, default=None,
                        help="path to UniDepth pretrained checkpoint")
    parser.add_argument("--teacher_unidepth_backbone", type=str, default="vitl14")
    parser.add_argument("--teacher_unidepth_version", type=str, default="v2",
                        choices=["v1", "v2"])
    # 9 = top band (~560-600k pixels): high-res eval inputs (>~600k pixels)
    # are only kept near-native at the top bands; lower bands DOWNSCALE
    # them, and smaller inputs get UPSCALED. Use 9 for paper-grade eval.
    parser.add_argument("--teacher_unidepth_resolution_level", type=int, default=9)

    # Alignment / evaluation options (match trainer.py defaults)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=100.0)
    parser.add_argument("--disable_median_scaling", action="store_true",
                        help="evaluate raw metric accuracy without GT-based "
                             "median rescaling")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_pred_depths", action="store_true",
                        help="save aligned depth maps to a .npz file")
    return parser.parse_args()


def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Load teachers (same loaders as training) ----
    opt = SimpleNamespace(
        teacher_da3_path=args.teacher_da3_path,
        teacher_da3_model_name=args.teacher_da3_model_name,
        teacher_unidepth_path=args.teacher_unidepth_path,
        teacher_unidepth_backbone=args.teacher_unidepth_backbone,
        teacher_unidepth_version=args.teacher_unidepth_version,
        teacher_unidepth_resolution_level=args.teacher_unidepth_resolution_level,
    )
    ensemble = load_teacher_models(opt, device)
    if ensemble is None:
        raise ValueError("Provide at least one of --teacher_da3_path / "
                         "--teacher_unidepth_path")

    has_da3 = 'DA3' in ensemble.teacher_models
    has_unidepth = 'UniDepth' in ensemble.teacher_models

    # ---- Dataset (same as evaluate_depth.py) ----
    filenames = readlines(os.path.join(splits_dir, args.eval_split,
                                       "test_files.txt"))
    dataset = datasets.KITTIRAWDataset(args.data_path, filenames,
                                       args.height, args.width,
                                       [0], 4, is_train=False)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True, drop_last=False)
    print("-> {} test images".format(len(dataset)))

    # ---- Channels to evaluate ----
    channel_names = []
    if has_unidepth:
        channel_names.append('UniDepth')
    if has_da3:
        channel_names.append('DA3')
    if has_da3 and has_unidepth:
        channel_names.append('Mean')
    errors = {name: [] for name in channel_names}
    ratios = {name: [] for name in channel_names}
    pred_depths_all = {name: [] for name in channel_names}  # per-image [H, W]
    saved_depths = {name: [] for name in channel_names} if args.save_pred_depths else None

    print("-> Predicting with teachers")
    with torch.no_grad():
        for step, data in enumerate(dataloader):
            input_color = data[("color", 0, 0)].to(device)

            # Raw teacher outputs: [N, M, H, W]
            # (DA3: focal-normalized -> meters; UniDepth: meters)
            teacher_depths = ensemble.predict_all(input_color)

            # Align relative teachers exactly as trainer.compute_teacher_outputs:
            # anchor = UniDepth's raw metric channel
            metric_anchor_depth = None
            if any(ensemble.is_relative) and has_unidepth:
                u_idx = ensemble.teacher_names.index('UniDepth')
                metric_anchor_depth = teacher_depths[:, u_idx:u_idx + 1]

            aligned = align_teacher_depths(
                teacher_depths=teacher_depths,
                is_relative=ensemble.is_relative,
                student_depth=None,
                metric_anchor_depth=metric_anchor_depth,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )

            channels = {}
            if has_unidepth:
                u_idx = ensemble.teacher_names.index('UniDepth')
                channels['UniDepth'] = aligned[:, u_idx]
            if has_da3:
                d_idx = ensemble.teacher_names.index('DA3')
                channels['DA3'] = aligned[:, d_idx]
            if has_da3 and has_unidepth:
                channels['Mean'] = aligned.mean(dim=1)

            for name, depth in channels.items():
                d = depth.cpu().numpy()  # [N, H, W]
                if saved_depths is not None:
                    saved_depths[name].append(d)
                for j in range(d.shape[0]):
                    pred_depths_all[name].append(d[j])

            if step % 50 == 0:
                print("   step {}/{}".format(step, len(dataloader)))

    # ---- Per-image metrics against GT (same protocol as evaluate_depth.py) ----
    gt_path = os.path.join(splits_dir, args.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1',
                        allow_pickle=True)["data"]

    print("-> Evaluating (median scaling: {})".format(
        "OFF" if args.disable_median_scaling else "ON"))

    for i in range(len(gt_depths)):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        if args.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]
                            ).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        else:
            mask = gt_depth > 0

        for name in channel_names:
            pred_depth = cv2.resize(pred_depths_all[name][i],
                                    (gt_width, gt_height))
            pred_depth_m = pred_depth[mask]
            gt_depth_m = gt_depth[mask]

            if not args.disable_median_scaling:
                ratio = np.median(gt_depth_m) / np.median(pred_depth_m)
                ratios[name].append(ratio)
                pred_depth_m = pred_depth_m * ratio

            pred_depth_m = np.clip(pred_depth_m, MIN_DEPTH, MAX_DEPTH)
            errors[name].append(compute_errors(gt_depth_m, pred_depth_m))

    # ---- Report ----
    header = ("{:>18} | " + "{:>8} | " * 7).format(
        "model", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
    print("\n" + header)
    print("-" * len(header))
    for name in channel_names:
        mean_errors = np.array(errors[name]).mean(0)
        print(("{:>18} | " + "{:8.3f} | " * 7).format(name, *mean_errors.tolist()))

    if not args.disable_median_scaling:
        print("\nScaling ratios (median | std):")
        for name in channel_names:
            r = np.array(ratios[name])
            med = np.median(r)
            print("  {:>18}: {:0.3f} | {:0.3f}".format(
                name, med, np.std(r / med)))

    if saved_depths is not None:
        out_path = os.path.join(os.path.dirname(__file__),
                                "teacher_depths_{}_split.npz".format(args.eval_split))
        np.savez_compressed(out_path,
                            **{name: np.stack(saved_depths[name])
                               for name in channel_names})
        print("\n-> Saved aligned depth maps to", out_path)


if __name__ == "__main__":
    evaluate(parse_args())
