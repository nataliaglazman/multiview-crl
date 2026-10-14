"""Alignment readouts for encoder-only ADNI training, using the offline probe protocol."""

import numpy as np

from eval.adni.encoder_modality_gap import diagnose, normalize_rows

PROBE_SEED = 1729


def _hit_probability(tied, positives, slots):
    """Probability of at least one positive when a cutoff intersects an exact tie."""
    draws = min(max(int(slots), 0), int(tied))
    if draws == 0:
        return 0.0
    if draws > tied - positives:
        return 1.0
    return float(1 - np.prod([(tied - positives - j) / (tied - j) for j in range(draws)]))


def retrieval_metrics(similarity, subjects):
    """Bidirectional subject retrieval; other visits are positives and ties get fractional credit."""
    same_subject = subjects[:, None] == subjects[None, :]
    n = len(subjects)
    metrics = {}
    for view, scores in enumerate((similarity, similarity.T)):
        hits = {1: [], 5: []}
        chance = {1: [], 5: []}
        ranks = []
        for row, positive in zip(scores, same_subject):
            best = row[positive].max()
            tied = np.isclose(row, best, rtol=0, atol=1e-12)
            ahead = int(((row > best) & ~tied).sum())
            count, positive_count = int(tied.sum()), int((tied & positive).sum())
            ranks.append(ahead + (count + 1) / (positive_count + 1))
            for k in hits:
                hits[k].append(_hit_probability(count, positive_count, k - ahead))
                chance[k].append(_hit_probability(n, int(positive.sum()), k))
        for k in hits:
            metrics[f"val/content_retrieval_top{k}_v{view}"] = float(np.mean(hits[k]))
            metrics[f"val/content_retrieval_chance_top{k}"] = float(np.mean(chance[k]))
        metrics[f"val/content_retrieval_mean_rank_v{view}"] = float(np.mean(ranks))
    for metric in ("top1", "top5", "mean_rank"):
        metrics[f"val/content_retrieval_{metric}"] = float(
            np.mean([metrics[f"val/content_retrieval_{metric}_v{view}"] for view in (0, 1)])
        )
    metrics["val/content_retrieval_chance"] = metrics["val/content_retrieval_chance_top1"]
    return metrics


def alignment_metrics(representations):
    """Measure subject matching, modality separation and spread on the same frozen vectors.

    Modality scores call the checkpoint diagnostic directly: normalization, folds,
    scaling, numerical tolerance and classifiers cannot drift between the two paths.
    """
    a = np.asarray(representations["content_v0"], dtype=np.float64)
    b = np.asarray(representations["content_v1"], dtype=np.float64)
    subjects = np.asarray(representations["subjects"], dtype=str)
    folds = min(3, len(np.unique(subjects)))
    report, _ = diagnose(a, b, subjects, n_splits=folds, seed=PROBE_SEED)
    geometry = report["geometry_normalized"]
    metrics = {
        "alignment/pairs": len(a),
        "alignment/subjects": len(np.unique(subjects)),
        "alignment/content_channels": a.shape[1],
        "alignment/probe_folds": folds,
        "alignment/probe_seed": PROBE_SEED,
        "content/centroid_distance": geometry["centroid_distance"],
        "content/centroid_distance_over_within_rms": geometry["centroid_distance_over_within_rms"],
        "content/paired_squared_distance": geometry["paired_squared_distance_mean"],
        "content/mean_offset_fraction": geometry["mean_offset_fraction_of_pair_squared_distance"],
        "content/paired_cosine_after_centering": report["paired_cosine_after_fold_centering"],
        "content/cross_view_cosine_std": geometry["paired_cosine_std"],
        "content/paired_cosine_valid_pairs": geometry["paired_cosine_valid_pairs"],
    }
    for condition in ("normalized", "mean_centered"):
        for probe in ("linear", "rbf"):
            metrics[f"content/modality_{condition}_{probe}_acc"] = report["probes"][f"{condition}_{probe}"]["accuracy"]
    metrics["content/modality_probe_acc"] = metrics["content/modality_normalized_linear_acc"]

    a, b = normalize_rows(a), normalize_rows(b)
    similarity = np.clip(a @ b.T, -1, 1)
    valid_a, valid_b = np.linalg.norm(a, axis=1) > 0, np.linalg.norm(b, axis=1) > 0
    valid_pair = valid_a & valid_b
    unmatched = (subjects[:, None] != subjects[None, :]) & valid_a[:, None] & valid_b[None, :]
    paired = float(np.diag(similarity)[valid_pair].mean()) if valid_pair.any() else float("nan")
    unpaired = float(similarity[unmatched].mean()) if unmatched.any() else float("nan")
    metrics.update(
        {
            "content/cross_view_cosine_mean": paired,
            "content/cross_view_cosine_unpaired": unpaired,
            "content/alignment_gap": paired - unpaired,
        }
    )
    metrics.update(retrieval_metrics(similarity, subjects))
    for view, values in enumerate((a, b)):
        centered = values - values.mean(0)
        singular = np.linalg.svd(centered, compute_uv=False)
        singular = singular[singular > 1e-12]
        weights = singular / singular.sum() if len(singular) else singular
        rank = float(np.exp(-(weights * np.log(weights)).sum())) if len(weights) else 0.0
        metrics[f"content/normalized_eff_rank_v{view}"] = rank
        metrics[f"content/normalized_rank_fraction_v{view}"] = rank / values.shape[1]
        metrics[f"content/normalized_spread_v{view}"] = float(np.sqrt(np.square(centered).sum(1).mean()))
        metrics[f"content/zero_vectors_v{view}"] = int((np.linalg.norm(values, axis=1) == 0).sum())
    return {key: float(value) if value is not None else float("nan") for key, value in metrics.items()}


def print_alignment_report(metrics, floor=None, previous=None):
    """Print interpretable groups with reference levels and signed temporal changes."""
    floor, previous = floor or {}, previous or {}
    print(
        f"    {metrics['alignment/pairs']:.0f} validation pairs | {metrics['alignment/subjects']:.0f} subjects | "
        f"{metrics['alignment/content_channels']:.0f} content channels | "
        f"{metrics['alignment/probe_folds']:.0f} subject folds, seed {metrics['alignment/probe_seed']:.0f}",
        flush=True,
    )
    print("    ↑ higher; ↓ lower; → toward reference; changes are descriptive, not significance tests", flush=True)
    header = f"      {'metric':<42}{'value':>9}{'reference':>11}"
    if floor:
        header += f"{'initial':>9}{'Δ initial':>11}"
    if previous:
        header += f"{'Δ previous':>12}"

    def section(title, rows):
        print(f"    --- {title} ---", flush=True)
        print(header, flush=True)
        for key, label, reference in rows:
            if key not in metrics:
                continue
            value = metrics[key]
            shown = f"{value:.3f}" if np.isfinite(value) else "n/a"
            ref = f"{reference:.3f}" if reference is not None and np.isfinite(reference) else "—"
            line = f"      {label:<42}{shown:>9}{ref:>11}"
            if floor:
                initial = floor.get(key, float("nan"))
                line += (
                    f"{initial:>9.3f}{value - initial:>+11.3f}"
                    if np.isfinite(value + initial)
                    else f"{'—':>9}{'—':>11}"
                )
            if previous:
                change = value - previous.get(key, float("nan"))
                line += f"{change:>+12.3f}" if np.isfinite(change) else f"{'—':>12}"
            print(line, flush=True)

    section(
        "subject alignment (retrieval reference = random matching)",
        [
            (
                "val/content_retrieval_top1",
                "↑ retrieval top-1, both directions",
                metrics["val/content_retrieval_chance"],
            ),
            (
                "val/content_retrieval_top5",
                "↑ retrieval top-5, both directions",
                metrics["val/content_retrieval_chance_top5"],
            ),
            ("content/cross_view_cosine_mean", "paired cosine (read with unpaired)", None),
            ("content/cross_view_cosine_unpaired", "different-subject cosine", None),
            ("content/alignment_gap", "↑ paired minus different-subject cosine", 0.0),
            ("content/centroid_distance_over_within_rms", "↓ modality mean gap / within-view spread", 0.0),
        ],
    )
    section(
        "modality leakage (reference = chance; uncentered normalized content)",
        [
            ("content/modality_normalized_linear_acc", "→ modality probe, linear", 0.5),
            ("content/modality_normalized_rbf_acc", "→ modality probe, RBF", 0.5),
            ("content/modality_mean_centered_rbf_acc", "RBF after modality centering [diagnostic]", 0.5),
        ],
    )
    print("      Centering uses known modality; its score does not measure learned invariance.", flush=True)
    section(
        "representation health (normalized content; rank alone is not quality)",
        [
            ("content/normalized_eff_rank_v0", "effective rank, T1", None),
            ("content/normalized_eff_rank_v1", "effective rank, FLAIR", None),
            ("content/normalized_spread_v0", "within-view spread, T1", None),
            ("content/normalized_spread_v1", "within-view spread, FLAIR", None),
        ],
    )
    if any(metrics[f"content/normalized_spread_v{v}"] <= 1e-8 for v in (0, 1)):
        print(
            "      WARNING: nearly constant content in at least one view; chance modality accuracy can reflect collapse.",
            flush=True,
        )
    section(
        "label information (reference = balanced chance)",
        [
            (
                f"content/{target}_probe_balanced_acc_v{view}",
                f"↑ {target} balanced acc, {name} content",
                metrics.get(f"content/{target}_probe_balanced_chance"),
            )
            for target in ("diagnosis", "gender", "race")
            for view, name in enumerate(("T1", "FLAIR"))
        ],
    )
    for target in ("diagnosis", "gender", "race"):
        prefix = f"content/{target}_probe"
        if f"{prefix}_n" not in metrics:
            continue
        stats = {
            key: metrics[f"{prefix}_{key}"]
            for key in (
                "n",
                "subjects",
                "classes",
                "folds",
                "missing",
                "rare",
                "conflicting",
                "chance",
                "balanced_chance",
            )
        }
        print(
            f"      {target}: {stats['n']:.0f} pairs / {stats['subjects']:.0f} subjects, "
            f"{stats['classes']:.0f} classes, {stats['folds']:.0f} folds; "
            f"excluded missing={stats['missing']:.0f}, rare={stats['rare']:.0f}, conflicting={stats['conflicting']:.0f}; "
            f"majority={stats['chance']:.3f}, balanced chance={stats['balanced_chance']:.3f}"
            + (" (probe unavailable: insufficient labeled subjects)" if not stats["folds"] else ""),
            flush=True,
        )

    section("training objective", [("val/loss", "↓ held-out training objective", None)])
