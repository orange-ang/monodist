"""
Teacher reprojection and monitored distillation loss.
Implements the MonDI-style teacher selection via reprojection loss
using student's pose network output.
"""

import torch
import torch.nn.functional as F


def align_teacher_depths(teacher_depths, is_relative,
                         student_depth=None, metric_anchor_depth=None,
                         min_depth=0.1, max_depth=100.0):
    """
    Align relative (scale-ambiguous) teacher depth outputs to metric scale.

    metric_anchor_depth: preferred reference (e.g. UniDepth's metric output).
    Anchoring to an independent metric teacher decouples the distillation
    target from the student (no target-following collapse). Falls back to
    student_depth when no anchor is provided.
    """
    N, M, H, W = teacher_depths.shape

    # Safety check: is_relative must match number of teachers
    if len(is_relative) != M:
        raise ValueError(
            f"is_relative has {len(is_relative)} elements but teacher_depths "
            f"has {M} channels. Check that predict_depth returns [N, 1, H, W].")

    aligned = teacher_depths.clone()

    for i in range(M):
        if not is_relative[i]:
            aligned[:, i:i+1] = torch.clamp(aligned[:, i:i+1], min=min_depth, max=max_depth)
            continue

        teacher_d = teacher_depths[:, i:i+1]

        if metric_anchor_depth is not None:
            aligned_d = _align_to_anchor(teacher_d, metric_anchor_depth,
                                         min_depth, max_depth)
        elif student_depth is not None:
            aligned_d = _align_to_anchor(teacher_d, student_depth,
                                         min_depth, max_depth)
        else:
            # Fallback: normalize to [min_depth, max_depth] when no student reference
            t_min = teacher_d.flatten(2).min(dim=2, keepdim=True)[0].unsqueeze(-1)
            t_max = teacher_d.flatten(2).max(dim=2, keepdim=True)[0].unsqueeze(-1)
            aligned_d = (teacher_d - t_min) / (t_max - t_min + 1e-8) * (max_depth - min_depth) + min_depth

        aligned[:, i:i+1] = torch.clamp(aligned_d, min=min_depth, max=max_depth)

    return aligned


def _align_to_anchor(teacher_d, anchor_depth, min_depth, max_depth):

    valid = ((anchor_depth > min_depth) & (anchor_depth < max_depth)
             & (teacher_d > 1e-6))
    anchor_masked = torch.where(
        valid, anchor_depth,
        torch.full_like(anchor_depth, float('nan')))
    teacher_masked = torch.where(
        valid, teacher_d,
        torch.full_like(teacher_d, float('nan')))

    # Paired spatial medians (nanmedian ignores masked pixels)
    anchor_med = torch.nanmedian(
        anchor_masked.flatten(2), dim=2, keepdim=True)[0].unsqueeze(-1)
    teacher_med = torch.nanmedian(
        teacher_masked.flatten(2), dim=2, keepdim=True)[0].unsqueeze(-1)

    # Degenerate guard: if a sample has no jointly-valid pixel, fall back to
    # the unmasked medians (scale stays finite, no NaN propagation)
    nan_rows = torch.isnan(anchor_med) | torch.isnan(teacher_med)
    if nan_rows.any():
        fallback_anchor = anchor_depth.flatten(2).median(
            dim=2, keepdim=True)[0].unsqueeze(-1)
        fallback_teacher = teacher_d.flatten(2).median(
            dim=2, keepdim=True)[0].unsqueeze(-1)
        anchor_med = torch.where(nan_rows, fallback_anchor, anchor_med)
        teacher_med = torch.where(nan_rows, fallback_teacher, teacher_med)

    # Scale factor to align teacher to the anchor's metric scale
    scale = anchor_med / (teacher_med.abs() + 1e-8)
    scale = scale.clamp(min=1e-3, max=1e3)  # prevent degenerate scaling

    aligned = teacher_d * scale
    return aligned


def _warp_source_image(image_source, depth, pose_T, K, inv_K,
                       backproject_depth, project_3d):
    """Warp the source frame into the target view with the given depth map
    and pose (standard monodepth2 reprojection). Differentiable w.r.t.
    depth: gradients flow through the grid_sample coordinates."""
    cam_points = backproject_depth(depth, inv_K)
    pix_coords = project_3d(cam_points, K, pose_T)
    return F.grid_sample(
        image_source, pix_coords, padding_mode="border", align_corners=True)


def compute_teacher_reprojection_loss(
    teacher_depth, image_source, image_target, pose_T, K, inv_K,
    backproject_depth, project_3d, ssim_module=None, no_ssim=False
):
    """Compute reprojection loss for a teacher's depth using student's pose."""
    warped_image = _warp_source_image(
        image_source, teacher_depth, pose_T, K, inv_K,
        backproject_depth, project_3d)

    abs_diff = torch.abs(image_target - warped_image)
    l1_loss = abs_diff.mean(1, True)

    var = image_target.var(dim=[1, 2, 3], keepdim=True).detach()
    var_adj = 3 * torch.abs(1 - 4 * var) + 3

    if no_ssim or ssim_module is None:
        reprojection_loss = l1_loss
    else:
        pred_new = warped_image * var_adj
        target_new = image_target * var_adj
        ssim_loss = ssim_module(pred_new, target_new).mean(1, True)
        # ssim_loss = ssim_module(warped_image, image_target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss.detach(), warped_image


def select_best_teacher(teacher_depths, reprojection_losses):
    """
    Select the best teacher per-pixel based on minimum reprojection loss.
    Pure per-pixel argmin selection, no spatial filtering.
    """
    N, M, H, W = teacher_depths.shape

    # Stack losses: [N, M, H, W]
    stacked_losses = torch.cat(reprojection_losses, dim=1)

    # Find teacher with minimum loss per pixel
    min_losses, teacher_idxs = torch.min(stacked_losses, dim=1, keepdim=True)  # [N, 1, H, W]

    # Gather the selected teacher's depth
    idx_expanded = teacher_idxs.expand(N, 1, H, W)
    selected_depth = torch.gather(teacher_depths, dim=1, index=idx_expanded)  # [N, 1, H, W]

    return selected_depth, min_losses, teacher_idxs


def depth_to_disp_norm(depth, min_depth, max_depth):
    """
    Convert metric depth to normalized disparity in [0, 1].
    Exact inverse of disp_to_depth (layers.py):
        disp = (1/depth - min_disp) / (max_disp - min_disp)
    with min_disp = 1/max_depth, max_disp = 1/min_depth.
    Teacher depth must already be aligned to the student's metric scale
    (median scaling), so the result is directly comparable to the student's
    native sigmoid disparity output.
    """
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    disp = (1.0 / depth.clamp(min=1e-3) - min_disp) / (max_disp - min_disp)
    return disp.clamp(0, 1)


def _weighted_silog(student_depth, teacher_depth, w_teacher, valid_mask,
                    silog_lambda=0.85, silog_alpha=1.0):

    log_diff = (torch.log(student_depth.clamp(min=1e-3))
                - torch.log(teacher_depth.clamp(min=1e-3)))

    # Weighted first / second moments over the gate-weighted population
    w_sum = w_teacher.flatten(1).sum(dim=1).clamp(min=1e-8)            # [N]
    mean_d = (w_teacher * log_diff).flatten(2).sum(dim=2) / w_sum      # [N]
    mean_d2 = (w_teacher * log_diff.pow(2)).flatten(2).sum(dim=2) / w_sum

    # +1e-8 guards the sqrt derivative: when the variance is exactly 0
    # (e.g. gate warmup f=0 -> w_teacher all zeros), sqrt'(0) = inf and
    # the upstream gradient is exactly 0, so autograd produces inf*0=NaN
    # and the student parameters blow up from the second batch on. The
    # eps keeps the derivative finite (value bias ~1e-4*fit_ratio,
    # negligible). fit_ratio separately fades the term to ~0.
    variance = torch.clamp(mean_d2 - silog_lambda * mean_d.pow(2), min=0.0)
    per_sample = silog_alpha * (variance + 1e-8).sqrt()                # [N]

    # Sample-level adaptation factor: overall distillation strength
    # follows how much of the valid population the gate actually trusts
    n_valid = valid_mask.flatten(1).sum(dim=1).clamp(min=1.0)          # [N]
    fit_ratio = w_sum / n_valid                                        # [N]

    return per_sample * fit_ratio


def compute_monitored_distillation_loss(
    student_depth, selected_teacher_depth,
    w_distillation=1.0,
    min_depth=0.1, max_depth=100.0,
    silog_lambda=0.85, silog_alpha=1.0
):

    valid_mask = ((selected_teacher_depth > min_depth) & (selected_teacher_depth < max_depth)).float()
    w_teacher = valid_mask

    distill_scalar = _weighted_silog(
        student_depth, selected_teacher_depth, w_teacher, valid_mask,
        silog_lambda, silog_alpha)                                     # [N]

    return w_distillation * distill_scalar.mean(), w_teacher


def compute_adaptive_combined_loss(
    student_depth, selected_teacher_depth, teacher_output_loss,
    self_supervised_loss, w_distillation=1.0,
    w_ensemble_temperature=5.0,
    min_depth=0.1, max_depth=100.0,
    silog_lambda=0.85, silog_alpha=1.0,
    warmup_factor=1.0,
    gate_threshold=0.15
):
    """
    Per-pixel complementary weighting with a gate-weighted SILog:

        loss = w_distillation * weighted_silog(w_teacher) + mean(q * selfsup)
        w_teacher = valid_mask - q

    w_distillation scales ONLY the distillation term (the self-supervised
    side stays at unit weight), so it is the single strength knob for
    distillation.

    q = valid_mask * sigmoid(temperature * (gate_threshold - teacher_output_loss))
    (detached, routing only) is the self-supervision-side weight: high
    reprojection loss -> weight toward distillation, low -> toward
    self-supervision. gate_threshold is the sigmoid pivot (calibrated
    against the criterion distribution; at the median it yields the
    half-activation equilibrium, see --distillation_gate_threshold).
    The distillation term is the per-pixel weighted
    SILog in log-depth space (see _weighted_silog): the gate enters the
    weighted statistics themselves and the fit_ratio multiplier scales
    the whole term by the trusted valid-pixel fraction. No extra masking
    is needed on top: w_teacher = 0 already excludes invalid pixels
    (zero contribution, zero gradient).

    warmup_factor is a GATE warmup: the routing criterion (student-pose
    reprojection) is unreliable early in training, so q is blended from
    the conservative valid_mask (full-strength self-supervision on all
    valid pixels, routed value ignored) toward the routed value as
    warmup_factor goes 0 -> 1. This matters because unreliable routing
    early on systematically depresses q (noisy pose -> high reprojection
    loss -> small sigmoid), which would otherwise silently attenuate the
    self-supervised term too. Through the complementary coupling
    w_teacher = valid_mask - q, the per-pixel distillation weight fades
    in automatically (fit_ratio = sum(w)/sum(valid) scales with it), so
    no separate scalar warmup multiplier is needed.
    """
    # Validity mask: excludes the teacher's clamp-saturation pixels
    valid_mask = ((selected_teacher_depth > min_depth) & (selected_teacher_depth < max_depth)).float()

    # Per-pixel gate q (fixed-threshold sigmoid gating) with gate warmup
    # (see docstring); the distillation weight is its complement on valid
    # pixels
    if w_ensemble_temperature > 0:
        q_routed = valid_mask * torch.sigmoid(
            w_ensemble_temperature * (gate_threshold - teacher_output_loss.detach()))
    else:
        q_routed = valid_mask
    q = warmup_factor * q_routed + (1.0 - warmup_factor) * valid_mask
    w_teacher = valid_mask - q

    distill_scalar = _weighted_silog(
        student_depth, selected_teacher_depth, w_teacher, valid_mask,
        silog_lambda, silog_alpha)                                     # [N]

    # The self-supervised term lives at full photometric resolution; bring
    # it onto the gate's grid for the q-weighted mean.
    if self_supervised_loss.shape[-2:] != q.shape[-2:]:
        self_supervised_loss = F.interpolate(
            self_supervised_loss, q.shape[-2:],
            mode='bilinear', align_corners=False)

    # Complementary combination: gate-weighted SILog scalar (distillation
    # side, fit_ratio included) + q-weighted self-supervised mean. The
    # gate warmup (see docstring) already fades the distillation part in
    # at the pixel level via w_teacher / fit_ratio, so no extra scalar is
    # applied here. w_distillation multiplies only the distillation term:
    # it is the strength knob for distillation relative to
    # self-supervision.
    distill_part = distill_scalar.mean()
    selfsup_part = (q * self_supervised_loss).mean()
    total_loss = w_distillation * distill_part + selfsup_part

    loss_dict = {
        'distillation_loss': distill_part.detach(),
        'self_supervised_loss': selfsup_part.detach(),
        'w_teacher_mean': w_teacher.mean().detach(),
        'w_self_mean': q.mean().detach(),
        'total_loss': total_loss
    }

    return total_loss, loss_dict


def _center_channels(feat):
    """Subtract the per-(sample, channel) spatial mean.

    DINOv2 token features and student conv features both carry a large
    per-channel DC offset, and their offsets live in unrelated places: an
    uncentered cosine would be dominated by that shared DC direction and
    report a spuriously high similarity (~0.9) for almost any pair of
    feature maps, leaving a near-zero gradient. Centering first turns the
    measure into a per-pixel correlation of the spatial VARIATION patterns,
    which is the part that actually carries scene structure.
    """
    return feat - feat.mean(dim=[2, 3], keepdim=True)


def compute_feature_distillation_loss(student_features, teacher_features,
                                      projectors, weights=None, info=None):
    """
    Intermediate-feature distillation (--w_feature_distillation > 0).

    student_features: [N, Cs, h, w] — the student encoder's BOTTOM output
        (stride 8, deepest stage; 128ch @ 24x80 for lite-mono at 192x640).
    teacher_features: list of m tensors [N, Ct_i, H'_i, W'_i] — one
        backbone tap per feature-exporting teacher (default block 23 =
        final DINOv2 ViT-L block, matching the student's deepest stage
        semantically), no grad, fp32. Each teacher lives on its own grid
        (DA3 11x37 aspect-preserving, UniDepth cropped back from its
        padding), so each is resized onto the student grid separately.
    projectors: list of m trainable 1x1 convs Cs -> Ct_i (one per teacher:
        the two ViT-L feature spaces are fine-tuned differently and cannot
        share a head). Discarded at inference.
    weights: [N, m, 1, 1] PER-SAMPLE soft weights over teachers (sum to 1
        across m; broadcast over the grid, never resized), or None for a
        uniform average. A full-resolution [N, m, H, W] map is still
        accepted and bilinearly resized onto the student grid.
    info: optional dict filled in-place with 'per_teacher' — the unweighted
        mean cosine loss of each teacher [m], for logging which teacher is
        driving the term (free: reuses the already-computed maps).

    Constraint: per-pixel COSINE similarity over the channel dim, not a
    value-space L1/L2. The student (a 128ch conv pyramid) and the teachers
    (1024ch DINOv2 ViT-L tokens) are structurally too far apart for their
    activations to be comparable in magnitude or offset; cosine only asks
    that the projected student direction agree with the teacher direction
    at each position. What that buys, precisely:
      * invariant to the OVERALL magnitude (the head need not match the
        DINOv2 activation scale) and to any per-channel DC offset (removed
        by _center_channels);
      * deliberately NOT invariant to per-channel re-weighting — cosine runs
        over channels, so the head is expected to learn each teacher's
        channel-importance profile. Z-scoring per channel instead would
        make it invariant, but at the cost of amplifying low-variance noise
        channels to the same footing as the informative ones.
    Loss = 1 - cos in [0, 2], averaged over pixels.

    Soft weighting uses the same per-teacher DEPTH reprojection criterion
    as the depth fusion (--ensemble_method soft), but reduces it to ONE
    scalar per (image, teacher) before the softmax: both teachers then
    contribute everywhere in a fixed proportion for that image instead of
    trading dominance pixel by pixel. See Trainer._feature_soft_weights for
    why the per-pixel version was dropped -- a globally shared 1x1 head
    cannot express a spatially varying target, and the bottom feature's
    receptive field is image-wide while a per-pixel criterion resampled
    onto that grid is an 8x point sample.

    Orthogonal to the depth-domain gate: features carry no reprojection
    criterion semantics of their own, so the term is NOT weighted by
    q / w_teacher.
    """
    if len(teacher_features) != len(projectors):
        raise ValueError(
            "feature distillation got {} teacher feature maps but {} "
            "projection heads".format(len(teacher_features), len(projectors)))

    target_hw = student_features.shape[-2:]
    loss_maps = []
    for teacher_feat, projector in zip(teacher_features, projectors):
        teacher_feat = F.interpolate(
            teacher_feat, target_hw, mode='bilinear', align_corners=False)
        projected = projector(student_features)
        cos = F.cosine_similarity(
            _center_channels(projected), _center_channels(teacher_feat),
            dim=1, eps=1e-6)                                     # [N, h, w]
        loss_maps.append(1.0 - cos)
    stacked = torch.stack(loss_maps, dim=1)                      # [N, m, h, w]

    if info is not None:
        info['per_teacher'] = stacked.detach().mean(dim=[0, 2, 3])   # [m]

    if weights is None:
        return stacked.mean(dim=1).mean()
    if weights.shape[1] != stacked.shape[1]:
        raise ValueError(
            "feature soft weights cover {} teachers but {} feature maps "
            "were supplied".format(weights.shape[1], stacked.shape[1]))
    # Per-sample weights arrive as [N, m, 1, 1] and broadcast against
    # [N, m, h, w] for free: no resize, so no interpolation error, and
    # sum-to-1 holds EXACTLY per sample rather than only to first order.
    # A genuinely per-pixel map (still accepted) is the only case that
    # needs bringing onto the coarse feature grid.
    if (weights.shape[-2:] != target_hw
            and tuple(weights.shape[-2:]) != (1, 1)):
        weights = F.interpolate(
            weights, target_hw, mode='bilinear', align_corners=False)
    return (stacked * weights).sum(dim=1).mean()

