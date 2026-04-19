"""Cross-reconstruction evaluation for content/style separation quality.

Measures how well content and style are disentangled by:
1. Content invariance: content representations of the same subject across T1/T2
   should be similar (style should not leak into content).
2. Style invariance: style representations of different subjects within the same
   modality should be similar (content should not leak into style).
3. Linear probe leakage: a linear probe on content should NOT predict modality,
   and a linear probe on style should NOT predict subject identity.

All metrics are computed *per level* for every level in ``args.content_style_levels``
(the levels where a content/style mask is applied).  The composite
``separation_score`` aggregates across levels so that the sweep objective
reflects content/style quality at the coarse levels (L1, L2) as well as L0.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


def _mask_to_indices(mask):
    """Convert a soft/hard content mask tensor to (v0_idx, v1_idx) channel lists.

    Accepts either a single ``(1, C)`` tensor (shared mask) or a
    ``(mask_v0, mask_v1)`` tuple (per-view masks).
    """
    if isinstance(mask, tuple):
        m_v0, m_v1 = mask
        idx_v0 = torch.where(m_v0.bool())[-1].tolist()
        idx_v1 = torch.where(m_v1.bool())[-1].tolist()
    else:
        idx = torch.where(mask.bool())[-1].tolist()
        idx_v0 = idx_v1 = idx
    return idx_v0, idx_v1


@torch.no_grad()
def _collect_representations(vqvae_model, dataloader, args, device, max_batches=200):
    """Encode all samples and collect content/style representations per level.

    Returns:
        dict mapping level index → {content_v0, content_v1, style_v0, style_v1}
        arrays of shape ``(N, n_content_channels)`` / ``(N, n_style_channels)``.
    """
    was_training = vqvae_model.training
    vqvae_model.eval()
    try:
        # Unwrap DataParallel / MoCo
        raw = vqvae_model
        if hasattr(raw, "online"):
            raw = raw.online
        if hasattr(raw, "module"):
            raw = raw.module

        levels = getattr(args, "content_style_levels", [0]) or [0]

        # Accumulate per-level lists
        per_level = {lvl: {"c_v0": [], "c_v1": [], "s_v0": [], "s_v1": []} for lvl in levels}

        for batch_idx, data in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            samples = data["image"]
            n_views = len(samples)
            if n_views < 2:
                continue
            images = torch.cat(samples, 0).to(device, non_blocking=True)
            B = samples[0].shape[0]

            (
                _recon,
                _diffs,
                encoder_features,
                _estimated_content_indices,
                _decoder_outputs,
                _id_outputs,
                soft_content_masks,
                _style_id_outputs,
            ) = vqvae_model(images, return_recon=False, pool_only=True, n_views=2)

            for lvl in levels:
                if lvl >= len(encoder_features):
                    continue

                hz = encoder_features[lvl]  # (2B, hidden_channels)
                hz_v0, hz_v1 = hz[:B], hz[B:]

                # Resolve content/style indices for this level.
                if soft_content_masks is not None and lvl in soft_content_masks:
                    c_idx_v0, c_idx_v1 = _mask_to_indices(soft_content_masks[lvl])
                else:
                    # Fall back to static config-based split (uniform across views).
                    c_idx_v0 = c_idx_v1 = list(range(len(args.content_indices[0])))

                all_channels = set(range(hz.shape[1]))
                s_idx_v0 = sorted(all_channels - set(c_idx_v0))
                s_idx_v1 = sorted(all_channels - set(c_idx_v1))

                per_level[lvl]["c_v0"].append(hz_v0[:, c_idx_v0].cpu().numpy())
                per_level[lvl]["c_v1"].append(hz_v1[:, c_idx_v1].cpu().numpy())
                per_level[lvl]["s_v0"].append(hz_v0[:, s_idx_v0].cpu().numpy())
                per_level[lvl]["s_v1"].append(hz_v1[:, s_idx_v1].cpu().numpy())

        out = {}
        for lvl, bufs in per_level.items():
            if not bufs["c_v0"]:
                continue
            out[lvl] = {
                "content_v0": np.concatenate(bufs["c_v0"], axis=0),
                "content_v1": np.concatenate(bufs["c_v1"], axis=0),
                "style_v0": np.concatenate(bufs["s_v0"], axis=0),
                "style_v1": np.concatenate(bufs["s_v1"], axis=0),
            }
        return out
    finally:
        vqvae_model.train(was_training)


def _cosine_sim(a, b):
    """Row-wise cosine similarity between two (N, D) arrays."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a_norm * b_norm).sum(axis=1)


def _metrics_for_level(reps_lvl):
    """Compute the full set of content/style separation metrics for one level.

    Returns a dict with un-suffixed keys; the caller is responsible for
    suffixing them with ``_L{i}``.
    """
    m = {}
    N = reps_lvl["content_v0"].shape[0]

    # --- 1. Content invariance across views ---
    content_sim = _cosine_sim(reps_lvl["content_v0"], reps_lvl["content_v1"])
    m["content/cross_view_cosine_mean"] = float(np.mean(content_sim))
    m["content/cross_view_cosine_std"] = float(np.std(content_sim))

    # --- 2. Style invariance within modality ---
    perm = np.random.permutation(N)
    style_intra_sim_v0 = _cosine_sim(reps_lvl["style_v0"], reps_lvl["style_v0"][perm])
    m["style/intra_view_cosine_mean_v0"] = float(np.mean(style_intra_sim_v0))

    style_cross_sim = _cosine_sim(reps_lvl["style_v0"], reps_lvl["style_v1"])
    m["style/cross_view_cosine_mean"] = float(np.mean(style_cross_sim))

    # --- 3. Linear probe: modality from content (should be chance = 0.5) ---
    content_all = np.concatenate([reps_lvl["content_v0"], reps_lvl["content_v1"]], axis=0)
    modality_labels = np.array([0] * N + [1] * N)

    content_scaled = make_pipeline(Normalizer(norm="l2"), StandardScaler()).fit_transform(content_all)
    try:
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        scores = cross_val_score(clf, content_scaled, modality_labels, cv=5, scoring="accuracy")
        content_modality_acc = float(np.mean(scores))
    except Exception:
        content_modality_acc = 0.5
    m["content/modality_probe_acc"] = content_modality_acc
    m["content/modality_invariance"] = 1.0 - abs(content_modality_acc - 0.5) * 2

    # --- 4. Linear probe: subject identity from style (should be chance) ---
    subject_ids = np.arange(N)
    style_scaled_v0 = make_pipeline(Normalizer(norm="l2"), StandardScaler()).fit_transform(reps_lvl["style_v0"])
    try:
        ridge = Ridge(alpha=1.0)
        scores = cross_val_score(ridge, style_scaled_v0, subject_ids, cv=5, scoring="r2")
        style_subject_r2 = float(np.mean(scores))
    except Exception:
        style_subject_r2 = 0.0
    m["style/subject_probe_r2"] = style_subject_r2
    m["style/subject_invariance"] = 1.0 - max(0.0, style_subject_r2)

    # --- 5. Linear probe: modality from style (should be high ~1.0) ---
    style_all = np.concatenate([reps_lvl["style_v0"], reps_lvl["style_v1"]], axis=0)
    style_all_scaled = make_pipeline(Normalizer(norm="l2"), StandardScaler()).fit_transform(style_all)
    try:
        clf2 = LogisticRegression(max_iter=1000, solver="lbfgs")
        scores2 = cross_val_score(clf2, style_all_scaled, modality_labels, cv=5, scoring="accuracy")
        style_modality_acc = float(np.mean(scores2))
    except Exception:
        style_modality_acc = 0.5
    m["style/modality_probe_acc"] = style_modality_acc

    # Per-level composite (same formula as the old global separation_score)
    m["separation_score"] = (m["content/modality_invariance"] + m["style/subject_invariance"]) / 2.0
    return m


def evaluate_content_style_separation(vqvae_model, dataloader, args, device, max_batches=200):
    """Run all content/style separation metrics for every masked level.

    For each level ``lvl`` in ``args.content_style_levels`` the full metric
    suite is computed and emitted with a ``_L{lvl}`` suffix
    (e.g. ``content/modality_invariance_L2``).  The composite
    ``separation_score`` is the mean of the per-level separation scores;
    ``separation_score_min`` is the worst level, useful for filtering runs
    where a coarse level is modality-contaminated while L0 looks fine.
    """
    reps = _collect_representations(vqvae_model, dataloader, args, device, max_batches)
    metrics = {}

    per_level_sep = {}
    for lvl, reps_lvl in reps.items():
        level_metrics = _metrics_for_level(reps_lvl)
        for k, v in level_metrics.items():
            metrics[f"{k}_L{lvl}"] = v
        per_level_sep[lvl] = level_metrics["separation_score"]

        # Per-level convenience aliases (L{i}/* namespace for easy sweep
        # filtering and W&B dashboard pinning).
        if lvl in (1, 2):
            prefix = f"L{lvl}"
            metrics[f"{prefix}/content_modality_invariance"] = level_metrics["content/modality_invariance"]
            metrics[f"{prefix}/style_subject_invariance"] = level_metrics["style/subject_invariance"]
            metrics[f"{prefix}/content_modality_probe_acc"] = level_metrics["content/modality_probe_acc"]
            metrics[f"{prefix}/style_subject_probe_r2"] = level_metrics["style/subject_probe_r2"]
            metrics[f"{prefix}/content_cross_view_cosine"] = level_metrics["content/cross_view_cosine_mean"]
            metrics[f"{prefix}/separation_score"] = level_metrics["separation_score"]

    # Backward-compatible unsuffixed keys mirror the finest available level
    # (usually L0), so existing dashboards that track e.g.
    # ``content/modality_invariance`` keep working.
    if reps:
        finest = min(reps.keys())
        for k in (
            "content/cross_view_cosine_mean",
            "content/cross_view_cosine_std",
            "style/intra_view_cosine_mean_v0",
            "style/cross_view_cosine_mean",
            "content/modality_probe_acc",
            "content/modality_invariance",
            "style/subject_probe_r2",
            "style/subject_invariance",
            "style/modality_probe_acc",
        ):
            suffixed = f"{k}_L{finest}"
            if suffixed in metrics:
                metrics[k] = metrics[suffixed]

    # --- Composite separation score aggregated across levels ---
    if per_level_sep:
        scores = list(per_level_sep.values())
        metrics["separation_score"] = float(np.mean(scores))
        metrics["separation_score_min"] = float(np.min(scores))
        metrics["separation_score_max"] = float(np.max(scores))
    else:
        metrics["separation_score"] = 0.0
        metrics["separation_score_min"] = 0.0
        metrics["separation_score_max"] = 0.0

    return metrics
