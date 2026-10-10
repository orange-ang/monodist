from __future__ import absolute_import, division, print_function

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.models_pose = {}
        self.parameters_to_train = []
        self.parameters_to_train_pose = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.profile = self.opt.profile

        self.num_scales = len(self.opt.scales)
        assert len(self.opt.distillation_scale_weights) == self.num_scales, (
            "--distillation_scale_weights must have one entry per scale "
            "(got {} for {} scales)".format(
                len(self.opt.distillation_scale_weights), self.num_scales))
        self.frame_ids = len(self.opt.frame_ids)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # ========== Build Student Network (Lite-Mono) ==========
        self.models["encoder"] = networks.LiteMono(
            model=self.opt.model, drop_path_rate=self.opt.drop_path,
            width=self.opt.width, height=self.opt.height)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # Feature-distillation projection heads (train-time only, discarded
        # at inference): ONE single 1x1 conv per feature-exporting teacher,
        # mapping the student's BOTTOM encoder output (stride 8,
        # num_ch_enc[-1]) to that teacher's tap channel dim. Separate heads
        # because the two DINOv2 ViT-L spaces are fine-tuned differently and
        # cannot share a projection. Registered in self.models so save_model
        # persists them (resume with --models_to_load feat_projector_DA3
        # feat_projector_UniDepth).
        from teacher_models import feature_teacher_names
        self.feature_teacher_names = feature_teacher_names(self.opt)
        for name in self.feature_teacher_names:
            key = "feat_projector_" + name
            self.models[key] = nn.Conv2d(
                self.models["encoder"].num_ch_enc[-1],
                self.opt.feature_distillation_channels,
                kernel_size=1)
            self.models[key].to(self.device)
            self.parameters_to_train += list(self.models[key].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models_pose["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)
                self.models_pose["pose_encoder"].to(self.device)
                self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())

                self.models_pose["pose"] = networks.PoseDecoder(
                    self.models_pose["pose_encoder"].num_ch_enc,
                    num_input_features=1, num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models_pose["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models_pose["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models_pose["pose"].to(self.device)
            self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        # ========== Optimizers ==========
        self.model_optimizer = optim.AdamW(
            self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
        if self.use_pose_net:
            self.model_pose_optimizer = optim.AdamW(
                self.parameters_to_train_pose, self.opt.lr[3], weight_decay=self.opt.weight_decay)

        self.model_lr_scheduler = ChainedScheduler(
            self.model_optimizer, T_0=int(self.opt.lr[2]), T_mul=1,
            eta_min=self.opt.lr[1], last_epoch=-1, max_lr=self.opt.lr[0],
            warmup_steps=0, gamma=0.9)
        if self.use_pose_net:
            self.model_pose_lr_scheduler = ChainedScheduler(
                self.model_pose_optimizer, T_0=int(self.opt.lr[5]), T_mul=1,
                eta_min=self.opt.lr[4], last_epoch=-1, max_lr=self.opt.lr[3],
                warmup_steps=0, gamma=0.9)

        if self.opt.load_weights_folder is not None:
            self.load_model()
        if self.opt.mypretrain is not None:
            self.load_pretrain()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # ========== Load Teacher Models ==========
        self.teacher_ensemble = None
        if self.opt.distillation_mode != "self_only":
            from teacher_models import load_teacher_models
            self.teacher_ensemble = load_teacher_models(self.opt, self.device)
            if self.teacher_ensemble is not None:
                print(f"Monitored distillation enabled with {self.teacher_ensemble.n_teachers} teachers")
                print(f"  Distillation mode: {self.opt.distillation_mode}")
                print(f"  Ensemble method: {self.opt.ensemble_method}")
            else:
                print("No teachers available, falling back to self_only mode")
                self.opt.distillation_mode = "self_only"

        # Checked AFTER the fallback above: if the teachers failed to load the
        # mode has already been rewritten to self_only and this must not fire.
        # feature_only pays for a full teacher forward (and, with the
        # reprojection criterion, the pose-driven warps) on every step, so
        # without a feature term it would be self_only at a large extra cost.
        if (self.opt.distillation_mode == "feature_only"
                and self.opt.w_feature_distillation <= 0):
            raise ValueError(
                "--distillation_mode feature_only disables the depth "
                "distillation term, so with --w_feature_distillation <= 0 the "
                "teachers would be loaded and run every step for nothing. "
                "That is self_only at extra cost -- either use "
                "--distillation_mode self_only, or set "
                "--w_feature_distillation > 0")

        if self.opt.w_feature_distillation > 0:
            if self.teacher_ensemble is None:
                raise ValueError(
                    "--w_feature_distillation > 0 requires at least one "
                    "teacher and distillation_mode != self_only")
            if not self.feature_teacher_names:
                raise ValueError(
                    "--w_feature_distillation > 0 but no feature-exporting "
                    "teacher is configured (loaded: {}). Feature export "
                    "needs --teacher_da3_path and/or "
                    "--teacher_unidepth_version v2.".format(
                        self.teacher_ensemble.teacher_names))
            unavailable = [
                n for n in self.feature_teacher_names
                if (n not in self.teacher_ensemble.teacher_models
                    or not self.teacher_ensemble.teacher_models[n]
                    .supports_feature_export)]
            if unavailable:
                raise ValueError(
                    "--w_feature_distillation > 0 but teacher(s) {} cannot "
                    "export backbone features (loaded: {})".format(
                        unavailable, self.teacher_ensemble.teacher_names))
            feat_T = self.opt.feature_distillation_temperature
            print("Feature distillation on: {} (layer {}, {}ch, cosine, "
                  "{} weights)".format(
                      ", ".join(self.feature_teacher_names),
                      self.opt.feature_distillation_layer,
                      self.opt.feature_distillation_channels,
                      "uniform 1/m" if feat_T <= 0
                      else "per-sample soft T={:g}".format(feat_T)))

        # ========== Data ==========
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        # Teacher selection statistics
        self.teacher_selection_stats = None
        if self.teacher_ensemble is not None:
            self.teacher_selection_stats = np.zeros(self.teacher_ensemble.n_teachers)

        self.save_opts()

    def set_train(self):
        for m in self.models.values():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        print("Training")
        self.set_train()
        self.model_lr_scheduler.step()
        if self.use_pose_net:
            self.model_pose_lr_scheduler.step()

        if self.teacher_selection_stats is not None:
            self.teacher_selection_stats = np.zeros(self.teacher_ensemble.n_teachers)

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            if self.use_pose_net:
                self.model_pose_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            if self.use_pose_net:
                self.model_pose_optimizer.step()

            duration = time.time() - before_op_time
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        if self.teacher_selection_stats is not None:
            total = self.teacher_selection_stats.sum() + 1e-8
            print(f"Teacher selection stats (epoch {self.epoch}):")
            for i, name in enumerate(self.teacher_ensemble.teacher_names):
                pct = 100 * self.teacher_selection_stats[i] / total
                print(f"  {name}: {int(self.teacher_selection_stats[i])} ({pct:.1f}%)")

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]
            outputs = self.models["depth"](features[0])
        else:
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        # Student anchor for feature distillation: the BOTTOM encoder
        # output (stride 8, deepest stage). For the shared pose model the
        # encoder runs on all frames at once -> features is a dict.
        if self.opt.w_feature_distillation > 0:
            student_feats = (features[0] if isinstance(features, dict)
                             else features)
            outputs["student_features_bottom"] = student_feats[-1]

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)

        # Teacher inference (before loss computation)
        if self.teacher_ensemble is not None and self.opt.distillation_mode != "self_only":
            self.compute_teacher_outputs(inputs, outputs)

        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    @torch.no_grad()
    def compute_teacher_outputs(self, inputs, outputs):
        """Run teachers and select best per-pixel via reprojection loss."""
        from teacher_reprojection import (
            align_teacher_depths, select_best_teacher)
        image0 = inputs[("color", 0, 0)]
        # Feature distillation: piggyback the backbone-tap export on the
        # same teacher forward passes (each wrapper caches its own tap)
        feat_layer = (
            self.opt.feature_distillation_layer
            if self.opt.w_feature_distillation > 0 else None)
        teacher_depths = self.teacher_ensemble.predict_all(
            image0, feat_layer=feat_layer)  # [N, M, H, W]

        # Align relative-depth teachers to metric scale.
        # Preferred anchor: the first metric teacher's output (UniDepth or
        # DA3) — an independent metric reference that decouples the
        # distillation target from the student (no target-following
        # collapse). Falls back to the student's own depth when no metric
        # teacher is loaded.
        metric_anchor_depth = None
        student_depth_pred = None
        if any(self.teacher_ensemble.is_relative):
            metric_idx = next((i for i, rel in enumerate(self.teacher_ensemble.is_relative)
                               if not rel), None)
            if metric_idx is not None:
                metric_anchor_depth = teacher_depths[:, metric_idx:metric_idx + 1]
            else:
                # Convert student disparity to depth for reference
                _, student_depth_pred = disp_to_depth(
                    outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

        teacher_depths = align_teacher_depths(
            teacher_depths=teacher_depths,
            is_relative=self.teacher_ensemble.is_relative,
            student_depth=student_depth_pred,
            metric_anchor_depth=metric_anchor_depth,
            min_depth=self.opt.min_depth,
            max_depth=self.opt.max_depth
        )
        outputs["teacher_depths"] = teacher_depths

        # Expose DA-3 raw metric depth for logging (teacher_depths above
        # already contains the clamped depth)
        if 'DA3' in self.teacher_ensemble.teacher_models:
            da3 = self.teacher_ensemble.teacher_models['DA3']
            if getattr(da3, 'last_raw_depth', None) is not None:
                outputs["DA3_raw_depth"] = da3.last_raw_depth

        M = self.teacher_ensemble.n_teachers
        # Per-teacher criterion, computed by the mondi/soft branches; the
        # feature soft weights reuse it (computed here only if the chosen
        # ensemble method did not need it).
        reprojection_losses = None

        if self.opt.ensemble_method == "mondi":
            # MonDI: per-pixel argmin over teacher reprojection loss
            reprojection_losses = self._teacher_reprojection_losses(
                teacher_depths, inputs, outputs)
            selected_depth, selected_loss, teacher_idxs = select_best_teacher(
                teacher_depths=teacher_depths,
                reprojection_losses=reprojection_losses)

            outputs["selected_teacher_depth"] = selected_depth
            outputs["selected_teacher_loss"] = selected_loss
            outputs["teacher_idxs"] = teacher_idxs

            # Update stats
            if teacher_idxs is not None and self.teacher_selection_stats is not None:
                for i in range(M):
                    self.teacher_selection_stats[i] += (teacher_idxs == i).sum().item()

        elif self.opt.ensemble_method == "soft":
            # Soft fusion: per-teacher criterion -> softmax weights ->
            # log-domain (geometric) weighted mean. Keeps BOTH teachers'
            # knowledge at every pixel (dual-teacher mondi beats either
            # single-teacher mondi) and avoids salt-and-pepper selection.
            reprojection_losses = self._teacher_reprojection_losses(
                teacher_depths, inputs, outputs)
            stacked = torch.cat(reprojection_losses, dim=1)        # [N,M,H,W]
            if self.opt.w_ensemble_temperature > 0:
                weights = F.softmax(
                    -self.opt.w_ensemble_temperature * stacked, dim=1)
            else:
                weights = torch.full_like(stacked, 1.0 / M)
            log_depths = torch.log(teacher_depths.clamp(min=1e-3))
            fused = torch.exp(
                (weights * log_depths).sum(dim=1, keepdim=True))

            outputs["selected_teacher_depth"] = fused
            # Gate criterion: reprojection loss of the fused teacher map
            outputs["selected_teacher_loss"] = self._map_reprojection_loss(
                fused, inputs, outputs)
            outputs["teacher_idxs"] = None

        elif self.opt.ensemble_method == "mean":
            # Log-domain (geometric) mean of teacher depths
            fused = torch.exp(torch.log(
                teacher_depths.clamp(min=1e-3)).mean(dim=1, keepdim=True))
            outputs["selected_teacher_depth"] = fused
            outputs["selected_teacher_loss"] = self._map_reprojection_loss(
                fused, inputs, outputs)
            outputs["teacher_idxs"] = None

        elif self.opt.ensemble_method == "random":
            r = random.randint(0, M - 1)
            outputs["selected_teacher_depth"] = teacher_depths[:, r:r+1]
            outputs["selected_teacher_loss"] = self._map_reprojection_loss(
                teacher_depths[:, r:r+1], inputs, outputs)
            outputs["teacher_idxs"] = None

        # ========== Feature-distillation targets + soft weights ==========
        if feat_layer is not None:
            feat_names, feat_list = self.teacher_ensemble.collect_features()
            if len(feat_list) > 0:
                outputs["teacher_features"] = feat_list
                outputs["teacher_feature_names"] = feat_names
                outputs["teacher_feature_weights"] = self._feature_soft_weights(
                    feat_names, teacher_depths, inputs, outputs,
                    reprojection_losses)

    @torch.no_grad()
    def _feature_soft_weights(self, feat_names, teacher_depths, inputs,
                              outputs, reprojection_losses):
        """Per-SAMPLE soft weights over the feature-exporting teachers.

        Returns [N, m, 1, 1] -- one scalar per (image, teacher), broadcast
        over the whole feature grid. Deliberately NOT per-pixel; all three
        reasons follow from what the term actually is:

        1. CAPABILITY. The projection head is a 1x1 conv -- a single
           globally shared matrix -- so it cannot express a spatially
           varying target. A per-pixel weight map can only reach it through
           the spatial average of the per-pixel gradients, which is exactly
           what a per-sample weight computes directly.
        2. GRANULARITY. The student anchor is the encoder's bottom output,
           whose receptive field spans the whole image: at 192x640 its 9
           dilated convs already reach ~392 px, and stage 2 ends in an LGFI
           block whose XCA sums over all 1920 positions with no position
           embedding. Putting a per-pixel criterion on that grid needs an
           8x downsample, and bilinear reads <=4 of each 8x8 block's 64
           pixels -- it drops essentially all of a peaked criterion's
           energy.
        3. ATTRIBUTION. The criterion asks whether a teacher's DEPTH
           survives backprojection under the student's pose -- a chain with
           four fallible links (depth head, median scaling, student K,
           student pose) of which the backbone feature is only the first.
           Averaging over the image keeps the systematic part (which
           teacher is better on this scene) and cancels the per-pixel noise
           that a globally shared head could not have used anyway.

        Consequence: |dL| SHRINKS against the per-pixel version, because
        random components cancel at 1/sqrt(HW), so at a given T the weights
        sit closer to 1/m. That is a better signal-to-noise ratio, not a
        lost signal -- raise T to sharpen. With m = 2 the better teacher's
        weight is still exactly sigmoid(T * mean_dL). T <= 0 gives exactly
        uniform 1/m, the no-criterion ablation arm.

        Restricting the softmax to the exporting subset keeps the weights
        normalized over exactly the teachers that contribute a feature term
        (a teacher without a tap must not steal weight mass). The feature
        maps themselves never take part: they carry no metric scale, so
        backproject/project cannot be applied to them, and substituting the
        per-teacher cosine loss would close a self-referential loop (weight
        depends on loss depends on weight).
        """
        if reprojection_losses is None:
            reprojection_losses = self._teacher_reprojection_losses(
                teacher_depths, inputs, outputs)
        idxs = [self.teacher_ensemble.teacher_names.index(n)
                for n in feat_names]
        stacked = torch.cat([reprojection_losses[i] for i in idxs], dim=1)
        T = self.opt.feature_distillation_temperature
        m = len(idxs)
        if T > 0:
            # Average the criterion over its spatial extent FIRST, then
            # softmax over the teacher axis -> [N, m, 1, 1]. No resize is
            # needed downstream, so sum-to-1 holds exactly per sample.
            return F.softmax(
                -T * stacked.mean(dim=[2, 3], keepdim=True), dim=1)
        return torch.full((stacked.shape[0], m, 1, 1), 1.0 / m,
                          dtype=stacked.dtype, device=stacked.device)

    def _teacher_reprojection_losses(self, teacher_depths, inputs, outputs):
        """Per-teacher reprojection losses with the student's (detached)
        pose, min-aggregated over source frames. Ensemble criterion."""
        K = inputs[("K", 0)]
        inv_K = inputs[("inv_K", 0)]
        return [self._map_reprojection_loss(
            teacher_depths[:, i:i+1], inputs, outputs, K=K, inv_K=inv_K)
            for i in range(self.teacher_ensemble.n_teachers)]

    def _map_reprojection_loss(self, depth_map, inputs, outputs,
                               K=None, inv_K=None):
        """Reprojection loss of a teacher depth map (min over source
        frames, detached output). Serves as selection criterion and/or
        gate input — never as a loss term. Min aggregation matches the
        student's self-supervised convention (a pixel explained by any
        one view counts; per-frame mean would double-count structural
        occlusion error and break distill/selfsup complementarity)."""
        from teacher_reprojection import compute_teacher_reprojection_loss
        if K is None:
            K = inputs[("K", 0)]
            inv_K = inputs[("inv_K", 0)]

        frame_rep_losses = []
        for frame_id in self.opt.frame_ids[1:]:
            if frame_id == "s":
                T = inputs["stereo_T"]
            else:
                if ("cam_T_cam", 0, frame_id) not in outputs:
                    continue
                T = outputs[("cam_T_cam", 0, frame_id)]
            rep_loss, _ = compute_teacher_reprojection_loss(
                teacher_depth=depth_map,
                image_source=inputs[("color", frame_id, 0)],
                image_target=inputs[("color", 0, 0)],
                pose_T=T.detach(), K=K, inv_K=inv_K,
                backproject_depth=self.backproject_depth[0],
                project_3d=self.project_3d[0],
                ssim_module=self.ssim if not self.opt.no_ssim else None,
                no_ssim=self.opt.no_ssim)
            frame_rep_losses.append(rep_loss)

        if len(frame_rep_losses) > 0:
            return torch.cat(frame_rep_losses, dim=1).min(
                dim=1, keepdim=True)[0]
        # Fallback: random loss to allow random selection
        return torch.rand(
            self.opt.batch_size, 1, self.opt.height, self.opt.width,
            device=self.device) * 0.1 + 0.05

    def predict_poses(self, inputs, features):
        outputs = {}
        if self.num_pose_frames == 2:
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models_pose["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        else:
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)
                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models_pose["pose_encoder"](pose_inputs)]
            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models_pose["pose"](pose_inputs)
            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])
        return outputs

    def val(self):
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses
        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]
                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            return l1_loss
        ssim_loss = self.ssim(pred, target).mean(1, True)
        return 0.85 * ssim_loss + 0.15 * l1_loss

    def compute_losses(self, inputs, outputs):
        """Compute losses with optional monitored distillation."""
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            # Compute reprojection losses for each source frame
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            # Auto-masking
            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(mask, [self.opt.height, self.opt.width],
                                         mode="bilinear", align_corners=False)
                reprojection_losses *= mask
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            # Per-pixel self-supervised loss
            self_sup_loss = to_optimise
            if self_sup_loss.dim() == 4 and self_sup_loss.shape[1] > 1:
                self_sup_loss = self_sup_loss.mean(1, keepdim=True)

            # ========== Monitored Distillation Loss ==========
            # Multi-scale: applied at every decoder scale, mirroring the
            # multi-scale self-supervised loss (coarse disp is upsampled to
            # the teacher's resolution, as self-sup upsamples disp to full
            # resolution before the photometric comparison).
            # WHITELIST, not "!= self_only": feature_only ALSO loads and runs
            # the teachers (the feature term and its reprojection-criterion
            # weights both need them) but must not touch the depth term. Naming
            # the two modes that do distill depth keeps any future mode from
            # silently inheriting it.
            use_distillation = (
                self.teacher_ensemble is not None and
                self.opt.distillation_mode in ("adaptive", "distill_only") and
                "selected_teacher_depth" in outputs
            )

            if use_distillation:
                from teacher_reprojection import (
                    compute_monitored_distillation_loss,
                    compute_adaptive_combined_loss)
                # Per-scale distillation strength: --w_distillation times
                # the scale-specific multiplier (uniform by default)
                w_distillation = (
                    self.opt.w_distillation *
                    self.opt.distillation_scale_weights[
                        self.opt.scales.index(scale)])
                student_depth = outputs[("depth", 0, scale)]
                sel_teacher_depth = outputs["selected_teacher_depth"]
                sel_teacher_loss = outputs["selected_teacher_loss"]

                # Resize teacher to match student if needed.
                # Log-domain interpolation: bilinear averaging in the depth
                # domain smears values across occlusion boundaries; the
                # geometric-mean behavior of exp(interp(log d)) preserves
                # discontinuities much better.
                if sel_teacher_depth.shape[-2:] != student_depth.shape[-2:]:
                    sel_teacher_depth = torch.exp(F.interpolate(
                        torch.log(sel_teacher_depth.clamp(min=1e-3)),
                        student_depth.shape[-2:],
                        mode='bilinear', align_corners=False))
                    sel_teacher_loss = F.interpolate(
                        sel_teacher_loss, student_depth.shape[-2:],
                        mode='bilinear', align_corners=False)

                # Distillation warmup: over the first epochs the routing
                # criterion (student-pose reprojection) is unreliable, so
                # the factor is applied inside the loss functions as a GATE
                # warmup: q is blended from the conservative valid_mask
                # (full-strength self-supervision, distillation weight 0)
                # toward the routed value; w_teacher = valid_mask - q then
                # fades the per-pixel distillation weight in automatically.
                if self.opt.distillation_warmup_epochs > 0:
                    warmup_factor = min(
                        1.0, self.epoch / float(self.opt.distillation_warmup_epochs))
                else:
                    warmup_factor = 1.0

                # Distillation target: pixel-wise SILog against the teacher
                # pseudo-label (the only depth-domain target; the former
                # reprojection-consistency 'warp' target was removed).

                if self.opt.distillation_mode == "adaptive":
                    # Per-pixel complementary weighting
                    # (1 - q) * distillation + q * self-supervised
                    combined_loss, loss_info = compute_adaptive_combined_loss(
                        student_depth=student_depth,
                        selected_teacher_depth=sel_teacher_depth,
                        teacher_output_loss=sel_teacher_loss,
                        self_supervised_loss=self_sup_loss,
                        w_distillation=w_distillation,
                        w_ensemble_temperature=self.opt.w_ensemble_temperature,
                        min_depth=self.opt.min_depth,
                        max_depth=self.opt.max_depth,
                        silog_lambda=self.opt.distillation_silog_lambda,
                        silog_alpha=self.opt.distillation_silog_alpha,
                        warmup_factor=warmup_factor,
                        gate_threshold=self.opt.distillation_gate_threshold)
                    loss += combined_loss
                    # Log the ACTUAL contribution to loss/{scale}: the
                    # function returns the raw (unweighted) distill_part,
                    # while the loss uses w_distillation * distill_part
                    # (incl. the per-scale multiplier). distill_only logs
                    # the already-weighted scalar; keep both consistent.
                    losses["distill_loss/{}".format(scale)] = (
                        w_distillation * loss_info['distillation_loss'])
                    losses["selfsup_loss/{}".format(scale)] = loss_info['self_supervised_loss']
                    losses["w_teacher/{}".format(scale)] = loss_info['w_teacher_mean']

                elif self.opt.distillation_mode == "distill_only":
                    # Pure distillation: uniform trust over the valid domain
                    # (no reprojection gate / warmup -- those arbitrate
                    # against a self-supervised term that
                    # does not exist in this mode).
                    dist_loss, _ = compute_monitored_distillation_loss(
                        student_depth=student_depth,
                        selected_teacher_depth=sel_teacher_depth,
                        w_distillation=w_distillation,
                        min_depth=self.opt.min_depth,
                        max_depth=self.opt.max_depth,
                        silog_lambda=self.opt.distillation_silog_lambda,
                        silog_alpha=self.opt.distillation_silog_alpha)
                    loss += dist_loss
                    losses["distill_loss/{}".format(scale)] = dist_loss.detach()
                else:
                    loss += self_sup_loss.mean()
            else:
                # Pure self-supervised: self_only, and feature_only where the
                # teachers run but contribute no depth term. Logged so the
                # self-supervision is comparable across arms -- adaptive logs
                # (q * self_sup).mean(), whereas here q is identically 1 and
                # there is no valid_mask, so this is the unattenuated value.
                selfsup = self_sup_loss.mean()
                loss += selfsup
                losses["selfsup_loss/{}".format(scale)] = selfsup.detach()

            # Smoothness loss
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales

        # ========== Feature distillation (orthogonal, once per batch) ==========
        # NOT per-scale: the student anchor is the single bottom encoder
        # output. NOT gated by q / w_teacher: features carry no
        # reprojection-criterion semantics of their own. Faded in with the
        # same warmup as the gate (the projection heads start untrained and
        # early student features are noise, so full strength from epoch 0 is
        # wasted). One cosine term per feature-exporting teacher, combined
        # with the soft reprojection-criterion weights.
        if ("teacher_features" in outputs
                and "student_features_bottom" in outputs):
            from teacher_reprojection import compute_feature_distillation_loss
            feat_names = outputs["teacher_feature_names"]
            projectors = [self.models["feat_projector_" + n]
                          for n in feat_names]
            if self.opt.distillation_warmup_epochs > 0:
                feat_warmup = min(
                    1.0, self.epoch / float(self.opt.distillation_warmup_epochs))
            else:
                feat_warmup = 1.0
            feat_info = {}
            feat_loss = compute_feature_distillation_loss(
                outputs["student_features_bottom"],
                outputs["teacher_features"],
                projectors,
                weights=outputs.get("teacher_feature_weights"),
                info=feat_info)
            feat_term = (self.opt.w_feature_distillation
                         * feat_warmup * feat_loss)
            total_loss = total_loss + feat_term
            losses["feat_distill_loss"] = feat_term.detach()
            # Diagnostics: the raw (unweighted) cosine loss per teacher, so a
            # head that fails to align shows up instead of hiding inside the
            # soft-weighted aggregate.
            for i, name in enumerate(feat_names):
                losses["feat_cos_{}".format(name)] = feat_info['per_teacher'][i]

        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)
        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        lr_pose = self.model_pose_optimizer.state_dict()['param_groups'][0]['lr'] if self.use_pose_net else 0
        print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
                                  lr_pose, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...], self.step)
                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

            # Log individual teacher outputs in the DISP domain: aligned
            # teacher depth converted via the exact inverse of disp_to_depth.
            # Distillation is computed in this space, so these images show
            # exactly what the student is being trained to match.
            if "teacher_depths" in outputs:
                from teacher_reprojection import depth_to_disp_norm
                teacher_depths = outputs["teacher_depths"]  # [N, M, H, W]
                M = teacher_depths.shape[1]
                for t_idx in range(M):
                    t_disp = depth_to_disp_norm(
                        teacher_depths[j, t_idx],
                        self.opt.min_depth, self.opt.max_depth)
                    t_vis = normalize_image(t_disp.unsqueeze(0)).expand(3, -1, -1)  # [3, H, W]
                    teacher_name = self.teacher_ensemble.teacher_names[t_idx]
                    writer.add_image(
                        "teacher_{}_disp/{}".format(teacher_name, j),
                        t_vis, self.step)

            # Log DA-3 raw metric depth (model output before any downstream
            # clamping) in the SAME normalized-disp domain as the UniDepth /
            # selected_teacher logs above, so it can be compared directly
            # against the student's disp output.
            if "DA3_raw_depth" in outputs:
                from teacher_reprojection import depth_to_disp_norm
                da3_disp = depth_to_disp_norm(
                    outputs["DA3_raw_depth"][j, 0],
                    self.opt.min_depth, self.opt.max_depth)
                da3_vis = normalize_image(da3_disp.unsqueeze(0)).expand(3, -1, -1)
                writer.add_image("teacher_DA3_disp/{}".format(j), da3_vis, self.step)

            # Log selected (per-pixel best) teacher output in DISP domain
            # (normalize_image for the same reason as above)
            if "selected_teacher_depth" in outputs:
                from teacher_reprojection import depth_to_disp_norm
                s_disp = depth_to_disp_norm(
                    outputs["selected_teacher_depth"][j],
                    self.opt.min_depth, self.opt.max_depth)
                s_vis = normalize_image(s_disp).expand(3, -1, -1)  # [3, H, W]
                writer.add_image("selected_teacher_disp/{}".format(j), s_vis, self.step)

    def save_opts(self):
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()
        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        for model_name, model in self.models_pose.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            torch.save(model.state_dict(), save_path)

        torch.save(self.model_optimizer.state_dict(),
                   os.path.join(save_folder, "adam.pth"))
        if self.use_pose_net:
            torch.save(self.model_pose_optimizer.state_dict(),
                       os.path.join(save_folder, "adam_pose.pth"))

        if self.teacher_selection_stats is not None:
            np.save(os.path.join(save_folder, "teacher_selection_stats.npy"),
                    self.teacher_selection_stats)

    def load_pretrain(self):
        self.opt.mypretrain = os.path.expanduser(self.opt.mypretrain)
        path = self.opt.mypretrain
        model_dict = self.models["encoder"].state_dict()
        pretrained_dict = torch.load(path)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                          if (k in model_dict and not k.startswith('norm'))}
        model_dict.update(pretrained_dict)
        self.models["encoder"].load_state_dict(model_dict)
        print('mypretrain loaded.')

    def load_model(self):
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            if n in ['pose_encoder', 'pose']:
                model_dict = self.models_pose[n].state_dict()
            else:
                model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            if n in ['pose_encoder', 'pose']:
                self.models_pose[n].load_state_dict(model_dict)
            else:
                self.models[n].load_state_dict(model_dict)

        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        optimizer_pose_load_path = os.path.join(self.opt.load_weights_folder, "adam_pose.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            self.model_optimizer.load_state_dict(torch.load(optimizer_load_path))
            if os.path.isfile(optimizer_pose_load_path):
                self.model_pose_optimizer.load_state_dict(torch.load(optimizer_pose_load_path))
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
