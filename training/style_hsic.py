"""Supervised style/anatomy independence penalty; no renderer changes required."""

import torch


def _rbf_gram(x):
    # Standardize across subjects, retaining every spatial coordinate. Differentiable
    # scaling prevents lowering the penalty simply by shrinking the style features.
    x = x.flatten(1).float()
    x = x - x.mean(0, keepdim=True)
    x = x / x.square().mean(0, keepdim=True).clamp_min(1e-12).sqrt()
    norms = x.square().sum(1)
    distances = (norms[:, None] + norms[None, :] - 2 * (x @ x.T)).clamp_min(0)
    pairs = torch.triu_indices(len(x), len(x), offset=1, device=x.device)
    bandwidth = distances[pairs[0], pairs[1]].detach().median().clamp_min(1e-6)
    kernel = torch.exp(-distances / (2 * bandwidth))
    return kernel - kernel.mean(0, keepdim=True) - kernel.mean(1, keepdim=True) + kernel.mean()


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
