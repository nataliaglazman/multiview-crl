"""Cross-reconstruction evaluation for content/style separation quality.

Measures how well content and style are disentangled by:
1. Content invariance: content representations of the same subject across T1/T2
   should be similar (style should not leak into content).
2. Style non-identifiability: given ``style_v0[i]``, its nearest neighbour in
   ``style_v1`` should NOT be subject ``i`` above chance (subject identity
   should not leak into style).
3. Linear probe leakage: a linear probe on content should NOT predict modality,
   and a linear probe on style SHOULD predict modality (style carries modality).

All metrics are computed *per level* for every level in ``args.content_style_levels``
(the levels where a content/style mask is applied).  The composite
``separation_score`` aggregates across levels so that the sweep objective
reflects content/style quality at the coarse levels (L1, L2) as well as L0.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

# Module-level cache of validation input batches. The val set is fixed across
# training, but NFS reads + torch.load of the .pt cache dominated the periodic
# eval cost. We pull the batches once into CPU memory and reuse them on every
# subsequent call. Keyed by (id(dataset), max_batches) so a different dataset
# or batch budget transparently rebuilds the cache.
_BATCH_CACHE = {"key": None, "batches": None}


def _get_cached_batches(dataloader, max_batches):
    key = (id(dataloader.dataset), max_batches)
    if _BATCH_CACHE["key"] == key and _BATCH_CACHE["batches"] is not None:
        return _BATCH_CACHE["batches"]
    cached = []
    for batch_idx, data in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        entry = {"image": [s.detach().cpu() for s in data["image"]]}
        if "label" in data:
            lbl = data["label"]
            entry["label"] = lbl.detach().cpu() if torch.is_tensor(lbl) else np.asarray(lbl)
        cached.append(entry)
    _BATCH_CACHE["key"] = key
    _BATCH_CACHE["batches"] = cached
    return cached


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
        labels_buf = []

        batches = _get_cached_batches(dataloader, max_batches)

        for data in batches:
            samples = data["image"]
            n_views = len(samples)
            if n_views < 2:
                continue
            images = torch.cat(samples, 0).to(device, non_blocking=True)
            B = samples[0].shape[0]
            if "label" in data:
                lbl = data["label"]
                if torch.is_tensor(lbl):
                    labels_buf.append(lbl.detach().cpu().numpy())
                else:
                    labels_buf.append(np.asarray(lbl))

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
        labels_arr = np.concatenate(labels_buf, axis=0) if labels_buf else None
        for lvl, bufs in per_level.items():
            if not bufs["c_v0"]:
                continue
            per_lvl = {
                "content_v0": np.concatenate(bufs["c_v0"], axis=0),
                "content_v1": np.concatenate(bufs["c_v1"], axis=0),
                "style_v0": np.concatenate(bufs["s_v0"], axis=0),
                "style_v1": np.concatenate(bufs["s_v1"], axis=0),
            }
            if labels_arr is not None:
                # Labels are per-subject; align to the encoded N (labels may be
                # longer if more batches were iterated than stored — slice down).
                n = per_lvl["content_v0"].shape[0]
                per_lvl["labels"] = labels_arr[:n]
            out[lvl] = per_lvl
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
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    style_intra_sim_v0 = _cosine_sim(reps_lvl["style_v0"], reps_lvl["style_v0"][perm])
    m["style/intra_view_cosine_mean_v0"] = float(np.mean(style_intra_sim_v0))

    style_cross_sim = _cosine_sim(reps_lvl["style_v0"], reps_lvl["style_v1"])
    m["style/cross_view_cosine_mean"] = float(np.mean(style_cross_sim))

    # --- 3. Linear probe: modality from content (should be chance = 0.5) ---
    content_all = np.concatenate([reps_lvl["content_v0"], reps_lvl["content_v1"]], axis=0)
    modality_labels = np.array([0] * N + [1] * N)

    content_scaled = make_pipeline(Normalizer(norm="l2"), StandardScaler()).fit_transform(content_all)
    try:
        clf = LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=-1)
        scores = cross_val_score(clf, content_scaled, modality_labels, cv=3, scoring="accuracy", n_jobs=-1)
        content_modality_acc = float(np.mean(scores))
    except Exception:
        content_modality_acc = 0.5
    m["content/modality_probe_acc"] = content_modality_acc
    m["content/modality_invariance"] = 1.0 - abs(content_modality_acc - 0.5) * 2

    # --- 4. Subject identifiability from style via paired retrieval ---
    # For each i, rank subjects in view 1 by cosine similarity to style_v0[i].
    # If style carries subject identity, the true match (j == i) ranks first.
    # Chance top-1 = 1/N; chance mean-rank = (N-1)/2.
    sv0 = reps_lvl["style_v0"]
    sv1 = reps_lvl["style_v1"]
    sv0_n = sv0 / (np.linalg.norm(sv0, axis=1, keepdims=True) + 1e-8)
    sv1_n = sv1 / (np.linalg.norm(sv1, axis=1, keepdims=True) + 1e-8)
    sim_matrix = sv0_n @ sv1_n.T  # (N, N)
    # Rank of the diagonal entry (true match) within each row, descending.
    order = np.argsort(-sim_matrix, axis=1)
    ranks = np.argmax(order == np.arange(N)[:, None], axis=1)
    top1 = float(np.mean(ranks == 0))
    mean_rank = float(np.mean(ranks))
    m["style/subject_retrieval_top1"] = top1
    m["style/subject_retrieval_mean_rank"] = mean_rank
    # Normalise top-1 against chance into an invariance score in [0, 1]:
    # 1.0 when top-1 == chance (no identity leakage), 0.0 when top-1 == 1.
    chance = 1.0 / max(N, 2)
    m["style/subject_invariance"] = float(max(0.0, 1.0 - (top1 - chance) / (1.0 - chance)))

    # --- 5. Linear probe: modality from style (should be high ~1.0) ---
    style_all = np.concatenate([reps_lvl["style_v0"], reps_lvl["style_v1"]], axis=0)
    style_all_scaled = make_pipeline(Normalizer(norm="l2"), StandardScaler()).fit_transform(style_all)
    try:
        clf2 = LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=-1)
        scores2 = cross_val_score(clf2, style_all_scaled, modality_labels, cv=3, scoring="accuracy", n_jobs=-1)
        style_modality_acc = float(np.mean(scores2))
    except Exception:
        style_modality_acc = 0.5
    m["style/modality_probe_acc"] = style_modality_acc

    # Per-level composite (same formula as the old global separation_score)
    m["separation_score"] = (m["content/modality_invariance"] + m["style/subject_invariance"]) / 2.0

    # --- 6. Anatomy-probe floor: content should be predictive of diagnosis.
    # High content/modality_invariance is only meaningful if content still
    # carries anatomical signal. diagnosis_info = 0 at chance, 1 at perfect.
    labels = reps_lvl.get("labels")
    if labels is not None and len(np.unique(labels[labels >= 0])) >= 2:
        mask_valid = labels >= 0
        content_all = np.concatenate([reps_lvl["content_v0"][mask_valid], reps_lvl["content_v1"][mask_valid]], axis=0)
        y_all = np.concatenate([labels[mask_valid], labels[mask_valid]], axis=0)
        counts = np.bincount(y_all)
        chance = float(counts.max()) / float(len(y_all))
        try:
            content_scaled = make_pipeline(Normalizer(norm="l2"), StandardScaler()).fit_transform(content_all)
            clf_d = LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=-1)
            scores_d = cross_val_score(clf_d, content_scaled, y_all, cv=3, scoring="accuracy", n_jobs=-1)
            diag_acc = float(np.mean(scores_d))
        except Exception:
            diag_acc = chance
        m["content/diagnosis_probe_acc"] = diag_acc
        m["content/diagnosis_probe_chance"] = chance
        m["content/diagnosis_info"] = float(max(0.0, (diag_acc - chance) / max(1e-6, 1.0 - chance)))
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
    torch.manual_seed(0)
    np.random.seed(0)
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
            metrics[f"{prefix}/style_subject_retrieval_top1"] = level_metrics["style/subject_retrieval_top1"]
            metrics[f"{prefix}/content_cross_view_cosine"] = level_metrics["content/cross_view_cosine_mean"]
            metrics[f"{prefix}/separation_score"] = level_metrics["separation_score"]
            if "content/diagnosis_info" in level_metrics:
                metrics[f"{prefix}/content_diagnosis_info"] = level_metrics["content/diagnosis_info"]
                metrics[f"{prefix}/content_diagnosis_probe_acc"] = level_metrics["content/diagnosis_probe_acc"]

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
            "style/subject_retrieval_top1",
            "style/subject_retrieval_mean_rank",
            "style/subject_invariance",
            "style/modality_probe_acc",
            "content/diagnosis_probe_acc",
            "content/diagnosis_probe_chance",
            "content/diagnosis_info",
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

    # Floor-gated separation: penalise runs whose content carries no diagnosis
    # information — otherwise a collapsed encoder wins by default (content
    # invariant because content is empty). The finest level (L0) is the one
    # the decoder sees, so its anatomy signal is what matters.
    floor = float(getattr(args, "separation_floor_diagnosis_info", 0.1))
    finest = min(reps.keys()) if reps else None
    diag_info_finest = metrics.get(f"content/diagnosis_info_L{finest}") if finest is not None else None
    if diag_info_finest is not None:
        gate = min(1.0, diag_info_finest / max(1e-6, floor))
        metrics["separation_score_gated"] = metrics["separation_score"] * gate
        metrics["anatomy_floor_gate"] = gate
    else:
        # No labels available — gating is a no-op.
        metrics["separation_score_gated"] = metrics["separation_score"]
        metrics["anatomy_floor_gate"] = 1.0

    return metrics
