"""Within-modality alignment of same-acquisition, different-anatomy pairs."""

import math

import torch
import torch.nn.functional as F


def within_modality_style_loss(style, paired_style, *, n_views=2, variance_weight=1.0, capture_components=False):
    """Align corresponding styles separately within each view, never across views.

    Both inputs are post-bottleneck, pre-quantization tensors in view-major order:
    [T1 subjects; FLAIR subjects]. Row i in each tensor must share acquisition
    settings, but have independently sampled anatomy. No factor labels are read.
    Use every spatial coordinate; GAP could hide equal-and-opposite spatial leaks.

    Loss = mean-view MSE + variance_weight * mean-view/branch variance hinge.
    Each hinge measures subject variation after centering each spatial coordinate,
    averaged within a channel. Fixed spatial templates cannot satisfy the hinge.
    No covariance penalty forces more independent factors than acquisition has.
    This discourages collapse but has a zero subgradient at exact collapse.
    """
    if (
        n_views != 2
        or style.ndim < 2
        or style.shape != paired_style.shape
        or style.numel() == 0
        or style.shape[0] % n_views
        or style.device != paired_style.device
    ):
        raise ValueError("Within-modality style alignment needs matching nonempty [T1; FLAIR] tensors.")
    if not math.isfinite(variance_weight) or variance_weight < 0:
        raise ValueError("Style alignment variance weight must be finite and nonnegative.")
    batch = style.shape[0] // n_views
    if batch < 2:
        raise ValueError("Within-modality style alignment needs at least two subjects per view.")

    diagnostics, alignments, hinges = {}, [], []
    with torch.autocast(device_type=style.device.type, enabled=False):
        for view, (left, right) in enumerate(zip(style.float().split(batch), paired_style.float().split(batch))):
            alignment = F.mse_loss(left, right)
            alignments.append(alignment)
            diagnostics[f"alignment_mse_v{view}"] = alignment.detach().item()
            # A cyclic mismatch is a diagnostic only, not an added training negative.
            diagnostics[f"alignment_mismatched_mse_v{view}"] = F.mse_loss(
                left.detach(), right.detach().roll(1, 0)
            ).item()
            for name, features in (("anchor", left), ("pair", right)):
                centered = features - features.mean(0, keepdim=True)
                rows = centered.movedim(1, 0).reshape(features.shape[1], -1)
                std = torch.linalg.vector_norm(rows, dim=1) / rows.shape[1] ** 0.5
                hinges.append(F.relu(1.0 - std).mean())
                diagnostics[f"alignment_std_{name}_v{view}"] = std.mean().detach().item()
        alignment = torch.stack(alignments).mean()
        hinge = torch.stack(hinges).mean()
        loss = alignment + variance_weight * hinge
    diagnostics.update(
        alignment_mse=alignment.detach().item(),
        alignment_var_hinge=hinge.detach().item(),
        alignment_var_weighted=(variance_weight * hinge).detach().item(),
    )
    if capture_components:
        loss._loss_components = {"alignment": alignment, "variance": variance_weight * hinge}
    return loss, diagnostics
