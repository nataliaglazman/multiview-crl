"""Style/anatomy independence penalties: supervised (vs ground truth) and label-free (cross-view)."""

import torch
import torch.nn.functional as F


def _rbf_kernel(x):
    # Standardize across subjects, retaining every spatial coordinate. Differentiable
    # scaling prevents lowering the penalty simply by shrinking the style features.
    x = x.flatten(1).float()
    x = x - x.mean(0, keepdim=True)
    x = x / x.square().mean(0, keepdim=True).clamp_min(1e-12).sqrt()
    norms = x.square().sum(1)
    distances = (norms[:, None] + norms[None, :] - 2 * (x @ x.T)).clamp_min(0)
    pairs = torch.triu_indices(len(x), len(x), offset=1, device=x.device)
    bandwidth = distances[pairs[0], pairs[1]].detach().median().clamp_min(1e-6)
    return torch.exp(-distances / (2 * bandwidth))


def _rbf_gram(x):
    kernel = _rbf_kernel(x)
    return kernel - kernel.mean(0, keepdim=True) - kernel.mean(1, keepdim=True) + kernel.mean()


def _unbiased_hsic(k, l):
    """Song et al. (2012), Eq. 5, for two fixed-kernel Gram matrices.

    The biased estimator keeps each kernel's diagonal. With high-dimensional,
    weakly structured features, that shared identity component can dominate the
    centered kernels even for independent representations.
    """
    if k.ndim != 2 or l.shape != k.shape or k.shape[0] != k.shape[1] or k.shape[0] < 4:
        raise ValueError("Unbiased HSIC needs matching square Gram matrices with at least four rows.")
    n = k.shape[0]
    off = 1 - torch.eye(n, device=k.device, dtype=torch.float64)
    # The estimator is invariant to a constant added off the diagonal. Removing the mean first,
    # in float64, avoids the cancellation between its large, nearly equal sums: in float32 that
    # residue divided by a near-zero normalizer returned -3.8e5 for collapsed features.
    k, l = k.double() * off, l.double() * off
    k = (k - k.sum() / (n * (n - 1))) * off
    l = (l - l.sum() / (n * (n - 1))) * off
    # 1^T K L 1 is a dot product of column/row sums, not an O(B^3) matmul.
    row_product = (k.sum(0) * l.sum(1)).sum()
    return ((k * l).sum() + k.sum() * l.sum() / ((n - 1) * (n - 2)) - 2 * row_product / (n - 2)) / (n * (n - 3))


def style_content_hsic_loss(style_features, content, n_views):
    """Mean normalized, biased RBF-HSIC over levels and modalities.

    ``style_features`` contains decoder-bound pre-quantization tensors with batch
    order [view0 subjects, view1 subjects, ...]. ``content`` is the B-by-K ground-truth
    content matrix, not learned content (which could itself omit the leaked factor).
    Full spatial tensors are flattened, never GAP-pooled. Kernels operate across
    subjects separately in each view, so opposite view-specific encodings cannot cancel.

    The centered kernel inner product is normalized by their Frobenius norms. This
    puts nonconstant scores on [0, 1]; finite minibatches have a positive dependence
    floor. Constant features score zero, so this is not an anti-collapse loss.
    Batches below four subjects are skipped and explicitly counted in diagnostics.
    Returns a differentiable scalar and detached diagnostics for training logs.
    """
    if not style_features:
        raise ValueError("Style HSIC requires a nonempty decoder style block (--inject-style-to-decoder).")
    if content.ndim != 2 or content.shape[1] == 0 or n_views < 1:
        raise ValueError("Style HSIC requires ground-truth z_content with shape (subjects, factors) and n_views >= 1.")
    first = next(iter(style_features.values()))
    content = content.detach().to(device=first.device, dtype=torch.float32)
    if not torch.isfinite(content).all():
        raise ValueError("Style HSIC received non-finite ground-truth content factors.")
    batch = content.shape[0]
    for features in style_features.values():
        if features.ndim < 2 or features.shape[0] != batch * n_views or features.numel() == 0:
            raise ValueError("Style HSIC features must contain n_views * subjects rows and nonempty style dimensions.")
    if batch < 4:
        return first.sum() * 0, {"Style/hsic_skipped_small_batch": 1.0}

    terms, diagnostics = [], {"Style/hsic_skipped_small_batch": 0.0}
    with torch.autocast(device_type=first.device.type, enabled=False):
        target_kernel = _rbf_gram(content)
        target_norm = target_kernel.square().sum()
        for level, features in style_features.items():
            for view, feature in enumerate(features.split(batch, dim=0)):
                kernel = _rbf_gram(feature)
                denominator = (kernel.square().sum() * target_norm).clamp_min(1e-12).sqrt()
                term = (kernel * target_kernel).sum() / denominator
                terms.append(term)
                diagnostics[f"Style/hsic_L{level}_v{view}"] = term.detach().item()
        loss = torch.stack(terms).mean()
    diagnostics["Style/hsic"] = loss.detach().item()
    return loss, diagnostics


def style_independence_loss(style, *, variance_weight=1.0):
    """Label-free style penalty: debiased RBF-CKA between the two views' style codes.

    Independent acquisition latents motivate this penalty on synthetic paired views.
    It targets dependence shared by BOTH styles, not all anatomical leakage: anatomy
    stored in only one view, or independent anatomical factors split between the two
    styles, can evade it. Low finite-batch HSIC is not an independence certificate.
    Coordinatewise offsets/nonzero rescalings cannot change the dependence score
    above the numerical variance floor; general affine mixing can change it.

    The Song estimator removes the diagonal contribution that can dominate the
    biased normalized score in high dimensions. Normalization, data-adaptive kernel
    bandwidth, and clamping mean the final training loss is NOT an unbiased HSIC
    estimator. The signed diagnostic can be negative; the penalty clamps at zero.

    ``style`` is one level's post-bottleneck, PRE-quantization tensor in view-major
    order, before decoder detachment/dropout. It is not the final quantized decoder
    input. Dependence and the variance hinge act on this SAME tensor. The hinge
    uses each channel's RMS variation across subjects and spatial coordinates after
    centering EACH coordinate across subjects. A fixed spatial template therefore
    cannot satisfy it, while varying spatial patterns need not survive GAP. It
    discourages collapse but does not guarantee useful acquisition information.

    Returns the loss and a dict of detached components; batches below four subjects contribute
    only the hinge.
    """
    if style.ndim < 2 or style.numel() == 0 or style.shape[0] % 2:
        raise ValueError("Style independence needs a nonempty [view0; view1] style tensor with an even batch.")
    if not 0 <= float(variance_weight) < float("inf"):
        raise ValueError("Style independence variance weight must be finite and nonnegative.")
    batch = style.shape[0] // 2
    with torch.autocast(device_type=style.device.type, enabled=False):
        style = style.float()
        hinge = style.sum() * 0
        diagnostics = {"xview_hsic_skipped_small_batch": float(batch < 4)}
        for view, features in enumerate(style.split(batch, dim=0)):
            centered = features - features.mean(0, keepdim=True)
            channel_rows = centered.movedim(1, 0).reshape(features.shape[1], -1)
            # vector_norm has a finite zero subgradient at exact collapse.
            std = torch.linalg.vector_norm(channel_rows, dim=1) / channel_rows.shape[1] ** 0.5
            diagnostics[f"subject_std_v{view}"] = std.mean().detach().item()
            if batch >= 2:
                hinge = hinge + F.relu(1.0 - std).mean()
        loss = variance_weight * hinge
        diagnostics.update(var_hinge=hinge.detach().item(), var_hinge_weighted=loss.detach().item())
        if batch < 4:
            return loss, diagnostics
        k0, k1 = _rbf_kernel(style[:batch]), _rbf_kernel(style[batch:])
        h00, h11 = _unbiased_hsic(k0, k0), _unbiased_hsic(k1, k1)
        # A view without variation shares nothing: define 0 rather than dividing noise by ~0.
        valid = (h00 > 1e-12) & (h11 > 1e-12)
        ratio = _unbiased_hsic(k0, k1) / (h00 * h11).clamp_min(1e-24).sqrt()
        dependence = torch.where(valid, ratio, torch.zeros_like(ratio)).float()
        penalty = dependence.clamp_min(0)
    diagnostics.update(
        xview_hsic=dependence.detach().item(),
        xview_hsic_penalty=penalty.detach().item(),
        xview_hsic_degenerate=float(not bool(valid)),
    )
    return loss + penalty, diagnostics
