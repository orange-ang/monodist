"""
Teacher model wrappers for monitored distillation.
Supports Depth-Anything-3 (DA3METRIC) and UniDepth as teacher models.
Each teacher is frozen and used in eval mode for inference only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys


def _ensure_init_py(path):
    """Create __init__.py in directory if it doesn't exist (fixes import issues)."""
    os.makedirs(path, exist_ok=True)
    init_file = os.path.join(path, '__init__.py')
    if not os.path.exists(init_file):
        open(init_file, 'w').close()


def _find_external_project(project_name):
    """
    Find an external project directory. Checks local subdirectory first,
    then falls back to sibling directory.
    
    Args:
        project_name: e.g. 'Depth-Anything-3', 'UniDepth'
    
    Returns:
        Absolute path to the project directory
    
    Raises:
        FileNotFoundError if not found in either location
    """
    current_file = os.path.abspath(__file__)
    project_dir = os.path.dirname(current_file)  # Lite-Mono-MonDI/
    parent_dir = os.path.dirname(project_dir)     # parent
    
    # Check subdirectory first (self-contained mode)
    local_path = os.path.join(project_dir, project_name)
    if os.path.exists(local_path):
        return local_path

    # Fallback to sibling directory (exact name first, then -main suffix)
    sibling_path2 = os.path.join(parent_dir, project_name)
    if os.path.exists(sibling_path2):
        return sibling_path2

    sibling_path = os.path.join(parent_dir, project_name + '-main')
    if os.path.exists(sibling_path):
        return sibling_path

    raise FileNotFoundError(
        f"External project '{project_name}' not found.\n"
        f"  Checked: {local_path}\n"
        f"  Checked: {sibling_path2}\n"
        f"  Checked: {sibling_path}\n"
        f"Please place it as a subdirectory of Lite-Mono-MonDI or as a sibling.")


def _add_to_sys_path(path):
    """Add path to sys.path if not already present."""
    abs_path = os.path.abspath(path)
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)
    return abs_path


class TeacherBase(nn.Module):
    """Base class for teacher models.
    
    Attributes:
        relative_depth: If True, output is scale-ambiguous relative depth
                        (needs alignment). If False, output is metric depth.
        supports_feature_export: If True, predict_depth(export_feat_layer=i)
                        additionally caches that DINOv2 backbone block's
                        features in self.last_features as [N, C, h', w']
                        fp32 (no grad), for intermediate-feature
                        distillation.
    """

    supports_feature_export = False

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.relative_depth = False  # Default: metric depth
        self.last_features = None

    def _freeze(self):
        """Freeze all parameters to prevent gradient computation during training."""
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def predict_depth(self, image, export_feat_layer=None):
        """
        Predict depth from an RGB image.

        Args:
            image: torch.Tensor [N, 3, H, W] normalized to [0, 1]
            export_feat_layer: optional DINOv2 backbone block index (see
                supports_feature_export); ignored by teachers that cannot
                export features.

        Returns:
            torch.Tensor [N, 1, H, W] depth (metric or relative depending on model)
        """
        raise NotImplementedError


def _normalize_state_dict_keys(state_dict, model_state_dict):
    """Remap checkpoint keys to the model's naming scheme.

    The official HuggingFace DA3 checkpoints are saved from the
    DepthAnything3 API wrapper (PyTorchModelHubMixin), whose attribute is
    `self.model = <net>`, so every key carries a `model.` prefix that a
    direct create_object() net does not have. Try a set of prefix
    strip/add candidates and return the variant with the most
    key+shape matches.
    """
    def n_matches(sd):
        return sum(1 for k, v in sd.items()
                   if k in model_state_dict
                   and model_state_dict[k].shape == v.shape)

    # Strip common wrappers (DDP, nested wrappers) first
    core = state_dict
    for prefix in ('module.', 'model.model.'):
        if core and all(k.startswith(prefix) for k in core):
            core = {k[len(prefix):]: v for k, v in core.items()}

    candidates = {'as_is': core}
    if core and all(k.startswith('model.') for k in core):
        candidates['strip model.'] = {k[len('model.'):]: v for k, v in core.items()}
    if core and all(k.startswith('da3.') for k in core):
        candidates['strip da3.'] = {k[len('da3.'):]: v for k, v in core.items()}
    candidates['add model.'] = {'model.' + k: v for k, v in core.items()}

    best_name, best_sd, best_n = 'as_is', core, n_matches(core)
    for name, sd in candidates.items():
        n = n_matches(sd)
        if n > best_n:
            best_name, best_sd, best_n = name, sd, n
    if best_name != 'as_is':
        print(f"[DepthAnything3Teacher] checkpoint key remap applied: "
              f"'{best_name}' (matched {best_n}/{len(model_state_dict)})",
              flush=True)
    return best_sd


class DepthAnything3Teacher(TeacherBase):
    """Depth-Anything-3 (da3metric-large) teacher model wrapper.

    da3metric-large is a NestedDepthAnything3Net: it aligns the main
    branch's depth to its own metric branch via least-squares scaling,
    with the metric conversion depth * (focal / 300) driven by the
    network's OWN predicted intrinsics (utils/alignment.py
    apply_metric_scaling). The returned depth is therefore already in
    meters - no external focal length is required.

    The network is built directly from the official registry/config
    (depth_anything_3.cfg) instead of the depth_anything_3.api wrapper,
    because api.py imports utils.export, whose module-level imports pull
    in heavy optional dependencies (moviepy/trimesh/open3d/pycolmap).
    """

    # Native support: forward(..., export_feat_layers=[i]) returns
    # output.aux["feat_layer_i"] on the resized patch grid.
    supports_feature_export = True

    def __init__(self, model_path, model_name='da3metric-large', device='cuda'):
        super().__init__(device)
        # da3metric outputs metric depth in meters (internal focal scaling)
        self.relative_depth = False

        # The DA3 repo is embedded as a subdirectory (self-contained
        # mode; see _find_external_project). Official repo is src-layout:
        # the depth_anything_3 package lives under <repo>/src.
        da_root = _find_external_project('Depth-Anything-3')
        src_dir = os.path.join(da_root, 'src')
        if not os.path.isdir(src_dir):
            src_dir = da_root  # tolerate a flat layout
        _add_to_sys_path(src_dir)

        # The embedded copy is already patched (see Depth-Anything-3/);
        # these calls are idempotent and also handle a fresh external clone.
        # NOTE: utils/export/__init__.py imports moviepy/trimesh/pycolmap
        # at module level - replace it with a lazy version so importing
        # depth_anything_3 does not crash when those are absent.
        pkg = os.path.join(src_dir, 'depth_anything_3')
        for sub in ('', 'utils', os.path.join('utils', 'io'),
                    os.path.join('model', 'dinov2')):
            _ensure_init_py(os.path.join(pkg, sub))
        _write_lazy_export_init(os.path.join(pkg, 'utils', 'export', '__init__.py'))

        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY

        self.model = create_object(load_config(MODEL_REGISTRY[model_name]))

        if model_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(model_path, device='cpu')
        else:
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
        incompat = self.model.load_state_dict(state_dict, strict=False)
        n_model = len(self.model.state_dict())
        n_missing = len(incompat.missing_keys)
        if n_model - n_missing < 0.9 * n_model:
            # Key scheme mismatch (e.g. official HF checkpoints carry a
            # `model.` prefix from the DepthAnything3 API wrapper):
            # retry with automatic key remapping.
            state_dict = _normalize_state_dict_keys(
                state_dict, self.model.state_dict())
            incompat = self.model.load_state_dict(state_dict, strict=False)
            n_missing = len(incompat.missing_keys)
        # Guard against SILENT partial loading (same rationale as UniDepth)
        n_model = len(self.model.state_dict())
        n_missing = len(incompat.missing_keys)
        print(f"[DepthAnything3Teacher] weights loaded: {n_model - n_missing}/{n_model} "
              f"(missing={n_missing}, unexpected={len(incompat.unexpected_keys)})",
              flush=True)
        if n_model - n_missing < 0.9 * n_model:
            print("WARNING: [DepthAnything3Teacher] checkpoint matched <90% of "
                  "model params - check the checkpoint against "
                  "da3metric-large! First missing keys: {}".format(
                      incompat.missing_keys[:5]), flush=True)
        self.model.eval()
        self.model.to(device)
        self._freeze()  # Freeze all parameters

        # DINOv2 patch size is 14. The official pipeline is ASPECT-
        # PRESERVING: longest side <= target, then each dimension is
        # rounded to the nearest multiple of 14 (InputProcessor's
        # upper_bound_resize). Squashing to a square (as an earlier
        # version did) distorts geometry and corrupts the network's
        # self-predicted intrinsics, breaking the focal/300 metric
        # scaling of da3metric-large. self.input_height stores the
        # longest-side target.
        self.input_height = 518
        self.input_width = 518
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        # Cached metric depth (before any downstream clamping) for logging
        self.last_raw_depth = None
        # Cached backbone-tap features [N, C, h', w'] (fp32) exported on
        # demand for feature distillation; None when not requested
        self.last_features = None

    @torch.no_grad()
    def predict_depth(self, image, export_feat_layer=None):
        """
        Args:
            image: [N, 3, H, W] in [0, 1]
            export_feat_layer: optional DA3 backbone block index (0-indexed,
                ViT-L has 24 blocks). When given, the block's token features
                are exported via the native export_feat_layers mechanism and
                cached in self.last_features as [N, C, H'/14, W'/14] fp32
                (the DPT head's input taps are exactly out_layers
                [4, 11, 17, 23]; 23 = final block = coarsest DPT branch).
        Returns:
            depth: [N, 1, H, W] metric depth in meters (the nested metric
                   branch does the focal scaling internally), clamped to
                   [0.1, 100] downstream by align_teacher_depths
        """
        N, C, H, W = image.shape
        # Aspect-preserving resize: longest side -> 518, then round each
        # dimension to the nearest multiple of 14 (official behavior).
        scale = self.input_height / float(max(H, W))
        new_h = max(14, int(round(H * scale / 14.0)) * 14)
        new_w = max(14, int(round(W * scale / 14.0)) * 14)
        input_image = F.interpolate(image, (new_h, new_w),
                                    mode='bilinear', align_corners=False)
        input_image = (input_image - self.imagenet_mean) / self.imagenet_std
        # Network expects (B, N_views, 3, H, W): treat each batch sample as
        # an independent single-view scene.
        input_image = input_image.unsqueeze(1)  # [N, 1, 3, H', W']

        if export_feat_layer is None:
            raw_output = self.model(input_image)
        else:
            raw_output = self.model(
                input_image, export_feat_layers=[export_feat_layer])

        # Cache the exported backbone tap for feature distillation: aux
        # arrives as [N, S=1, H'/14, W'/14, C] tokens on the RESIZED grid
        self.last_features = None
        if export_feat_layer is not None:
            feat = raw_output.aux["feat_layer_{}".format(export_feat_layer)]
            self.last_features = feat.squeeze(1).permute(
                0, 3, 1, 2).float().contiguous().detach()

        depth = raw_output['depth']
        depth = depth.squeeze(0)  # drop the batch dim: [N, 1, H', W']
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.shape[1] != 1:
            depth = depth[:, :1]

        # Log-domain resize: bilinear averaging in the depth domain smears
        # values across occlusion boundaries ("flying pixels"); interpolating
        # log(depth) keeps the average in the geometric-mean sense and
        # preserves discontinuities much better.
        depth = torch.exp(F.interpolate(
            torch.log(depth.float().clamp(min=1e-3, max=1e4)), (H, W),
            mode='bilinear', align_corners=False))

        # Cache for tensorboard logging (before any downstream clamping)
        self.last_raw_depth = depth.detach()

        return depth


def _write_lazy_export_init(init_path):
    """Replace depth_anything_3/utils/export/__init__.py with a lazy-import
    version. The official file imports gs/colmap/glb exporters at module
    level, which drag in moviepy/trimesh/open3d/pycolmap; the distillation
    pipeline never calls export(), so lazy forwarding is safe."""
    lazy_content = (
        "# Lazy-import shim (auto-generated by teacher_models.py):\n"
        "# the official __init__ imports heavy optional deps (moviepy,\n"
        "# trimesh, open3d, pycolmap) at module level.\n"
        "from depth_anything_3.specs import Prediction  # noqa: F401\n"
        "\n"
        "def export(prediction, export_format, export_dir, **kwargs):\n"
        "    from depth_anything_3.utils.export_impl import export as _export\n"
        "    return _export(prediction, export_format, export_dir, **kwargs)\n"
    )
    if os.path.exists(init_path):
        with open(init_path, 'r') as f:
            content = f.read()
        if 'Lazy-import shim' in content:
            return  # already patched
    else:
        content = ''
    # Keep the official dispatcher reachable under a new module name in case
    # something else imports it explicitly.
    impl_path = os.path.join(os.path.dirname(init_path), 'export_impl.py')
    if content and not os.path.exists(impl_path):
        with open(impl_path, 'w') as f:
            f.write(content)
    with open(init_path, 'w') as f:
        f.write(lazy_content)


class UniDepthTeacher(TeacherBase):
    """UniDepth teacher model wrapper.
    Outputs metric depth in meters."""

    def __init__(self, checkpoint_path, device='cuda',
                 backbone='vitl14', version='v2', resolution_level=0):
        super().__init__(device)
        self.relative_depth = False  # UniDepth outputs metric depth

        # Add UniDepth to sys.path
        unidepth_root = _find_external_project('UniDepth')
        _add_to_sys_path(unidepth_root)

        # Ensure __init__.py exists in subpackages
        _ensure_init_py(os.path.join(unidepth_root, 'unidepth'))
        _ensure_init_py(os.path.join(unidepth_root, 'unidepth', 'models'))
        _ensure_init_py(os.path.join(unidepth_root, 'unidepth', 'models', 'unidepthv2'))
        _ensure_init_py(os.path.join(unidepth_root, 'unidepth', 'models', 'backbones'))
        _ensure_init_py(os.path.join(unidepth_root, 'unidepth', 'layers'))
        _ensure_init_py(os.path.join(unidepth_root, 'unidepth', 'ops'))
        _ensure_init_py(os.path.join(unidepth_root, 'unidepth', 'utils'))

        # Load config JSON (required by UniDepthV2.__init__)
        import json
        config_name = f'config_{version}_{backbone}.json'
        config_path = os.path.join(unidepth_root, 'configs', config_name)
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"UniDepth config not found: {config_path}\n"
                f"Available configs: {os.listdir(os.path.join(unidepth_root, 'configs'))}")
        with open(config_path, 'r') as f:
            config = json.load(f)

        if version == 'v2':
            from unidepth.models.unidepthv2.unidepthv2 import UniDepthV2
            self.model = UniDepthV2(config=config)
        else:
            from unidepth.models.unidepthv1.unidepthv1 import UniDepthV1
            self.model = UniDepthV1(config=config)

        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        incompat = self.model.load_state_dict(state_dict, strict=False)
        # Guard against SILENT partial loading (config/checkpoint mismatch
        # leaves a randomly-initialized network whose output is garbage and
        # gets clamped to a constant, silently destroying distillation).
        n_model = len(self.model.state_dict())
        n_missing = len(incompat.missing_keys)
        n_loaded = n_model - n_missing
        print(f"[UniDepthTeacher] weights loaded: {n_loaded}/{n_model} "
              f"(missing={n_missing}, unexpected={len(incompat.unexpected_keys)})",
              flush=True)
        if n_loaded < 0.9 * n_model:
            print("WARNING: [UniDepthTeacher] checkpoint matched <90% of "
                  "model params - check --teacher_unidepth_version / "
                  "--teacher_unidepth_backbone against the checkpoint! "
                  "First missing keys: {}".format(incompat.missing_keys[:5]),
                  flush=True)
        self.model.eval()
        self.model.to(device)
        # Set interpolation mode (UniDepth V2 attribute)
        if hasattr(self.model, 'interpolation_mode'):
            self.model.interpolation_mode = 'bilinear'
        # Pin the processing resolution ([0,10) slices the [pixels_min,
        # pixels_max] range; unset -> full bounds + warning). IMPORTANT:
        # inputs with fewer pixels than the band's lower bound are UPSCALED
        # to reach it, and ViT attention cost grows quadratically with the
        # resulting token count. Default 0 = lowest band (~200-240k pixels,
        # ~1000 tokens): plenty for distilling into a 192x640 student, since
        # the teacher depth is resized down to the student resolution anyway.
        # Higher levels (e.g. 9 -> ~580k pixels, ~3000 tokens, ~8x cost) are
        # only worth it for high-resolution benchmark evaluation.
        if hasattr(self.model, 'shape_constraints'):
            self.model.resolution_level = resolution_level
        self._freeze()  # Freeze all parameters

        # ---- Intermediate-feature export (v2 only) ----
        # UniDepthV2.infer() returns depth/points/rays/confidence but NOT
        # the backbone taps, so capture the pixel encoder's output with a
        # forward hook rather than patching the model source. v1 has a
        # different encoder/tap layout and stays unsupported.
        self.supports_feature_export = (version == 'v2')
        self._feat_layer = None
        self._captured_feat = None
        if self.supports_feature_export:
            from unidepth.models.unidepthv2.unidepthv2 import get_paddings
            self._get_paddings = get_paddings
            self.model.pixel_encoder.register_forward_hook(
                self._capture_encoder_features)

    def _capture_encoder_features(self, module, inputs, output):
        """Forward hook on UniDepth's DINOv2 pixel encoder.

        output = (features, class_tokens), where `features` holds ONE ENTRY
        PER BLOCK (24 for ViT-L: DinoVisionTransformer.forward appends
        unconditionally in the block loop), each [B, h', w', C] on the
        padded+resized grid, cls/register tokens already stripped and the
        final LayerNorm applied (use_norm=true). The decoder only consumes
        the 4 taps picked by slices_encoder_range ([6,12,18,24] via
        stacking_fn="last"), but every block is reachable, so
        --feature_distillation_layer indexes this list directly: 23 = final
        block = tap 24 = the deepest, i.e. exactly what matches the
        student's bottom encoder stage semantically (the decoder seeds its
        coarsest latent from tap 6 instead, but distillation transfers
        representation content, not pipeline role).
        """
        if self._feat_layer is None:
            return
        features = output[0]
        if self._feat_layer >= len(features):
            raise ValueError(
                "--feature_distillation_layer {} exceeds UniDepth's {} "
                "encoder blocks".format(self._feat_layer, len(features)))
        # infer() runs under fp16 autocast -> cast up for the loss
        self._captured_feat = features[self._feat_layer].float()

    def _postprocess_features(self, feat, H, W):
        """[B, h', w', C] on the padded grid -> [B, C, h', w'] covering
        exactly the original H x W content.

        infer() pads the image to satisfy ratio_bounds BEFORE resizing, so
        the encoder grid carries letterbox bands with no counterpart in the
        student's features; resizing that padded grid straight onto the
        student grid would squeeze the scene and silently misalign the two
        maps. The resize is uniform, so the crop box scales linearly from
        padded-image coords to grid coords.
        """
        feat = feat.permute(0, 3, 1, 2).contiguous()      # [B, C, fh, fw]
        ratio_bounds = self.model.shape_constraints["ratio_bounds"]
        (pad_l, pad_r, pad_t, pad_b), (pad_H, pad_W) = self._get_paddings(
            (H, W), ratio_bounds)
        if pad_l == pad_r == pad_t == pad_b == 0:
            return feat
        fh, fw = feat.shape[-2:]
        r0 = int(round(pad_t * fh / float(pad_H)))
        r1 = int(round((pad_H - pad_b) * fh / float(pad_H)))
        c0 = int(round(pad_l * fw / float(pad_W)))
        c1 = int(round((pad_W - pad_r) * fw / float(pad_W)))
        if r1 - r0 < 1 or c1 - c0 < 1:
            # Grid too coarse to crop meaningfully: keep it whole rather
            # than produce an empty map (better slightly misaligned than
            # crashing / dropping the term).
            return feat
        return feat[..., r0:r1, c0:c1]

    @torch.no_grad()
    def predict_depth(self, image, export_feat_layer=None):
        """
        Args:
            image: [N, 3, H, W] in [0, 1]
            export_feat_layer: optional DINOv2 block index; when given, the
                encoder tap is captured by the forward hook and cached in
                self.last_features as [N, C, h', w'] fp32 (padding cropped).
        Returns:
            depth: [N, 1, H, W]
        """
        N, C, H, W = image.shape

        # UniDepthV2.infer expects uint8-like [0,255] input and handles
        # normalization, resizing, and padding internally.
        rgb = image * 255.0  # convert [0,1] -> [0,255]

        # Arm the hook for this call only (it is a no-op otherwise, so
        # plain depth-only inference is untouched).
        self.last_features = None
        self._captured_feat = None
        self._feat_layer = (
            export_feat_layer if self.supports_feature_export else None)
        try:
            # Pass camera=None: UniDepth will estimate its own intrinsics
            # internally. We don't pass the student's K here because:
            # 1) infer() resizes+pads images internally, K would need
            #    adjustment
            # 2) Teacher only produces depth; the actual reprojection uses
            #    student K
            outputs = self.model.infer(rgb, camera=None)
        finally:
            self._feat_layer = None
        if self._captured_feat is not None:
            self.last_features = self._postprocess_features(
                self._captured_feat, H, W).detach()
            self._captured_feat = None

        if isinstance(outputs, dict):
            depth = outputs['depth']  # [N, 1, H', W']
        else:
            depth = outputs

        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.shape[-2:] != (H, W):
            # Log-domain resize (see DepthAnything3Teacher.predict_depth):
            # avoids depth-space smearing across occlusion boundaries.
            depth = torch.exp(F.interpolate(
                torch.log(depth.clamp(min=1e-3, max=1e4)), (H, W),
                mode='bilinear', align_corners=False))
        depth = torch.clamp(depth, min=0.1, max=300.0)
        return depth


class TeacherEnsemble:
    """
    Manages multiple teacher models and computes per-pixel teacher selection
    based on reprojection loss (following the MonDI approach).
    """

    def __init__(self, teacher_models, device='cuda'):
        """
        Args:
            teacher_models: dict of {name: TeacherBase}
            device: torch device
        """
        self.teacher_models = teacher_models
        self.device = device
        self.teacher_names = list(teacher_models.keys())
        self.n_teachers = len(self.teacher_names)
        # Track which teachers output relative (non-metric) depth
        self.is_relative = [teacher_models[n].relative_depth for n in self.teacher_names]

    @torch.no_grad()
    def predict_all(self, image, feat_layer=None):
        """
        Get depth predictions from all teachers.

        Args:
            image: [N, 3, H, W]
            feat_layer: optional DINOv2 backbone block index; EVERY teacher
                with supports_feature_export additionally caches that tap's
                features in its own last_features (piggybacks on the same
                forward pass, no extra teacher inference)

        Returns:
            teacher_depths: [N, M, H, W] stacked teacher depth predictions
        """
        depths = []
        for name in self.teacher_names:
            teacher = self.teacher_models[name]
            if feat_layer is not None and teacher.supports_feature_export:
                depth = teacher.predict_depth(
                    image, export_feat_layer=feat_layer)
            else:
                depth = teacher.predict_depth(image)
            # Ensure shape is [N, 1, H, W]
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            if depth.shape[1] != 1:
                raise ValueError(
                    f"Teacher '{name}' predict_depth returned shape {depth.shape}, "
                    f"expected [N, 1, H, W]")
            depths.append(depth)
        teacher_depths = torch.cat(depths, dim=1)
        return teacher_depths

    def collect_features(self):
        """(names, features) of every teacher that produced a backbone tap
        on the last predict_all call, in teacher_names order — so the
        caller can line them up with the per-teacher reprojection-criterion
        channels when building the soft weights."""
        names, features = [], []
        for name in self.teacher_names:
            feat = getattr(self.teacher_models[name], 'last_features', None)
            if feat is not None:
                names.append(name)
                features.append(feat)
        return names, features


def feature_teacher_names(opt):
    """Names of the teachers that will export intermediate features, derived
    from the OPTIONS alone (mirroring load_teacher_models' conditions and
    insertion order).

    Needed because the projection heads must be built BEFORE the teachers
    are instantiated: they have to exist in time to join the optimizer's
    parameter groups, while load_teacher_models runs after that.
    """
    names = []
    if getattr(opt, 'w_feature_distillation', 0) <= 0:
        return names
    if getattr(opt, 'teacher_da3_path', None):
        names.append('DA3')
    if (getattr(opt, 'teacher_unidepth_path', None)
            and getattr(opt, 'teacher_unidepth_version', 'v2') == 'v2'):
        names.append('UniDepth')
    return names


def load_teacher_models(opt, device):
    """
    Load teacher models based on options.

    Args:
        opt: parsed options
        device: torch device

    Returns:
        TeacherEnsemble instance or None
    """
    teachers = {}

    if opt.teacher_da3_path is not None and opt.teacher_da3_path != '':
        print(f"Loading Depth-Anything-3 teacher from: {opt.teacher_da3_path}")
        teachers['DA3'] = DepthAnything3Teacher(
            model_path=opt.teacher_da3_path,
            model_name=getattr(opt, 'teacher_da3_model_name', 'da3metric-large'),
            device=device)

    if opt.teacher_unidepth_path is not None and opt.teacher_unidepth_path != '':
        print(f"Loading UniDepth teacher from: {opt.teacher_unidepth_path}")
        teachers['UniDepth'] = UniDepthTeacher(
            checkpoint_path=opt.teacher_unidepth_path,
            device=device,
            backbone=opt.teacher_unidepth_backbone,
            version=opt.teacher_unidepth_version,
            resolution_level=getattr(opt, 'teacher_unidepth_resolution_level', 0))

    if len(teachers) == 0:
        print("WARNING: No teacher models loaded. Running in pure self-supervised mode.")
        return None

    print(f"Loaded {len(teachers)} teacher model(s): {list(teachers.keys())}")
    return TeacherEnsemble(teachers, device=device)
