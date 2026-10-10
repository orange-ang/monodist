from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = "/public/experiments/chenziang/litemono/dist1"


class LiteMonoOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Lite-Mono options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default="/public/datasets_neo/kitti_jpg/KITTI_raw")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(file_dir, "node85"))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="lite-mono")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou")
        self.parser.add_argument("--model",
                                 type=str,
                                 help="which model to load",
                                 choices=["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"],
                                 default="lite-mono")
        self.parser.add_argument("--weight_decay",
                                 type=float,
                                 help="weight decay in AdamW",
                                 default=1e-2)
        self.parser.add_argument("--drop_path",
                                 type=float,
                                 help="drop path rate",
                                 default=0.3)
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        self.parser.add_argument("--profile",
                                 type=bool,
                                 help="profile once at the beginning of the training",
                                 default=True)

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--lr",
                                 nargs="+",
                                 type=float,
                                 help="learning rates of DepthNet and PoseNet. "
                                      "Initial learning rate, "
                                      "minimum learning rate, "
                                      "First cycle step size.",
                                 default=[0.0001, 5e-6, 31, 0.0001, 1e-5, 31])
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=30)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--mypretrain",
                                 type=str,
                                 help="if set, use my pretrained encoder")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

      # ========== MONITORED DISTILLATION options ==========
        # Teacher model paths
        self.parser.add_argument("--teacher_da3_path",
                                 type=str, default=None,
                                 help="path to Depth-Anything-3 (da3metric-large) "
                                      "checkpoint (.safetensors)")
        self.parser.add_argument("--teacher_da3_model_name",
                                 type=str, default="da3metric-large",
                                 help="Depth-Anything-3 preset name "
                                      "(da3metric-large outputs metric depth "
                                      "with internal focal scaling)")
        self.parser.add_argument("--teacher_unidepth_path",
                                 type=str, default=None,
                                 help="path to UniDepth pretrained checkpoint")
        self.parser.add_argument("--teacher_unidepth_backbone",
                                 type=str, default="vitl14",
                                 help="UniDepth backbone variant")
        self.parser.add_argument("--teacher_unidepth_version",
                                 type=str, default="v2",
                                 choices=["v1", "v2"],
                                 help="UniDepth version")
        self.parser.add_argument("--teacher_unidepth_resolution_level",
                                 type=int, default=0,
                                 help="UniDepth processing resolution level "
                                      "in [0, 10): slices [pixels_min, "
                                      "pixels_max] into bands and UPSCALES "
                                      "smaller inputs to reach the band. "
                                      "Attention cost grows quadratically, "
                                      "so keep 0 for training (teacher depth "
                                      "is resized to the student resolution "
                                      "anyway); raise only for eval")

        # Distillation loss settings
        self.parser.add_argument("--w_distillation",
                                 type=float, default=0.2,
                                 help="weight of distillation loss (relative to "
                                      "self-supervised). Scales ONLY the "
                                      "distillation term in adaptive mode — "
                                      "the single strength knob "
                                      "(distillation is applied at all "
                                      "scales; per-scale multipliers via "
                                      "--distillation_scale_weights)")
        self.parser.add_argument("--distillation_scale_weights",
                                 nargs="+", type=float,
                                 default=[1.0, 1.0, 1.0],
                                 help="per-scale multipliers of "
                                      "--w_distillation, in the order of "
                                      "--scales (must have one entry per "
                                      "scale)")
        self.parser.add_argument("--w_ensemble_temperature",
                                 type=float, default=8,
                                 help="temperature for adaptive weighting "
                                      "(higher = sharper selection). Drives "
                                      "BOTH the adaptive gate sigmoid and "
                                      "--ensemble_method soft depth fusion, "
                                      "so sweeping it moves two mechanisms at "
                                      "once. It no longer drives the feature "
                                      "distillation weights -- those have "
                                      "their own "
                                      "--feature_distillation_temperature")
        self.parser.add_argument("--distillation_silog_lambda",
                                 type=float, default=0.85,
                                 help="lambda of the SILog distillation loss "
                                      "(weight of the squared-mean term; "
                                      "0 = plain log-L2 keeping the teacher's "
                                      "metric scale, 1 = fully scale "
                                      "invariant, 0.85 = paper default)")
        self.parser.add_argument("--distillation_silog_alpha",
                                 type=float, default=1.0,
                                 help="alpha multiplier of the SILog "
                                      "distillation loss (paper uses 10; "
                                      "multiplies only the depth-domain "
                                      "SILog term; overall strength knob is "
                                      "--w_distillation)")
        self.parser.add_argument("--distillation_warmup_epochs",
                                 type=int, default=4,
                                 help="gate warmup: over this many epochs "
                                      "the routing gate q is linearly handed "
                                      "over from the conservative value "
                                      "valid_mask (full self-supervision, no "
                                      "distillation weight; early student "
                                      "pose is too noisy for reliable "
                                      "routing) to the routed value; 0 "
                                      "disables warmup")
        self.parser.add_argument("--ensemble_method",
                                 type=str, default="mondi",
                                 choices=["mondi", "mean", "random", "soft"],
                                 help="teacher ensemble method: 'mondi' = "
                                      "per-pixel argmin over teacher "
                                      "reprojection loss; 'mean' = log-domain "
                                      "(geometric) mean of teacher depths; "
                                      "'random' = one random teacher per "
                                      "batch; 'soft' = softmax-weighted "
                                      "log-domain fusion (weights from "
                                      "per-teacher reprojection loss, "
                                      "temperature = --w_ensemble_"
                                      "temperature). ALL methods derive the "
                                      "gate criterion from the reprojection "
                                      "loss of the resulting teacher map, so "
                                      "routing stays content-adaptive")
        self.parser.add_argument("--distillation_gate_threshold",
                                 type=float, default=0.10,
                                 help="fixed threshold of the sigmoid routing "
                                      "gate q = sigmoid(T*(thr - rep_loss)). "
                                      "Calibration: 0.15 sits at the half-"
                                      "activation equilibrium (threshold ~ "
                                      "median of the criterion distribution, "
                                      "w_teacher plateaus ~0.48); lowering "
                                      "shifts the balance toward "
                                      "distillation")
        self.parser.add_argument("--distillation_mode",
                                 type=str, default="adaptive",
                                 choices=["adaptive", "distill_only",
                                          "self_only", "feature_only"],
                                 help="adaptive: MonDI-style adaptive weighting; "
                                      "distill_only: only distillation loss; "
                                      "self_only: only self-supervised loss, "
                                      "teachers are not even loaded; "
                                      "feature_only: teachers ARE loaded and run "
                                      "every step, but the DEPTH distillation "
                                      "term is skipped entirely and the "
                                      "photometric self-supervision stays at "
                                      "FULL weight (no q, no valid_mask), so the "
                                      "feature term is the only teacher signal. "
                                      "This is the arm that isolates feature "
                                      "distillation's own contribution. Note "
                                      "that adaptive with --w_distillation 0 is "
                                      "NOT equivalent: there the "
                                      "self-supervision is still multiplied by "
                                      "q (~0.44 at the criterion median) and by "
                                      "valid_mask. Requires "
                                      "--w_feature_distillation > 0")
        # Feature distillation settings: distill the student encoder's
        # BOTTOM output (stride 8, deepest stage) toward a backbone tap of
        # EVERY teacher that can export one (DA3 natively, UniDepth v2 via
        # a pixel-encoder hook), each through its own trainable 1x1
        # projection head. The per-teacher cosine losses are combined with
        # PER-SAMPLE soft weights: the depth-reprojection criterion the gate
        # and depth fusion use, averaged over the image and then softmaxed
        # over teachers at its own temperature. So each image gets ONE weight
        # vector rather than a per-pixel map, and the feature weighting can
        # be sharpened or flattened without moving either of those.
        # Orthogonal additive term — not gated by q, faded in with the
        # gate warmup.
        self.parser.add_argument("--w_feature_distillation",
                                 type=float, default=1,
                                 help="weight of the intermediate-feature "
                                      "distillation term (0 = disabled). "
                                      "Requires at least one feature-"
                                      "exporting teacher (DA3 or UniDepth "
                                      "v2) and distillation_mode != "
                                      "self_only; the projection heads are "
                                      "train-time only (discarded at "
                                      "inference)")
        self.parser.add_argument("--feature_distillation_layer",
                                 type=int, default=23,
                                 help="DINOv2 backbone block index exported "
                                      "as the feature target (0-indexed; "
                                      "both teachers are ViT-L/14 with 24 "
                                      "blocks, so one index drives DA3's "
                                      "export_feat_layers and UniDepth's "
                                      "encoder tap alike). 23 = final block "
                                      "output: matches the student's "
                                      "deepest encoder stage semantically "
                                      "(and for DA3 is also the coarsest-"
                                      "scale branch source of the DPT head)")
        self.parser.add_argument("--feature_distillation_channels",
                                 type=int, default=1024,
                                 help="channel dim of the teacher features "
                                      "(DINOv2 ViT-L = 1024 for both "
                                      "teachers); each projection head maps "
                                      "the student bottom features to this "
                                      "dim")
        self.parser.add_argument("--feature_distillation_temperature",
                                 type=float, default=20.0,
                                 help="temperature of the soft weights that "
                                      "combine the per-teacher feature cosine "
                                      "losses. ")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
