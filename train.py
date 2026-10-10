"""
Training script for Lite-Mono with Monitored Distillation (MonDI).

This framework combines Lite-Mono's self-supervised depth estimation with
MonDI-style monitored distillation from multiple teacher models
(Depth-Anything-3, UniDepth).

The training dynamically selects the best teacher per-pixel based on
reprojection loss (using student's pose network output), and adaptively
weights between distillation loss and self-supervised photometric loss.

Usage:
    python train.py \
        --data_path /path/to/kitti_data \
        --teacher_da3_path /path/to/da3metric_large.safetensors \
        --teacher_unidepth_path /path/to/unidepth_v2_vitl14.pth \
        --distillation_mode adaptive \
        --w_distillation 1.0 \
        --ensemble_method mondi
"""

from __future__ import absolute_import, division, print_function

from options import LiteMonoOptions
from trainer import Trainer

if __name__ == "__main__":
    options = LiteMonoOptions()
    opts = options.parse()
    trainer = Trainer(opts)
    trainer.train()
