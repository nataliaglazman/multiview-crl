"""Per-channel linear probes for disease and modality prediction.

For every encoder channel at each requested level, global-average-pools the
spatial feature map to a scalar and fits logistic regression for:
  (a) disease classification (same task options as disease_classifier.py)
  (b) modality classification (T1 vs T2; requires --views both)

Outputs ``<run-dir>/channel_probe_metrics.json`` containing per-channel scores
and aggregated content-vs-style summaries.  The ``channels`` list is designed
for easy loading into a DataFrame / scatter plot::

    df = pd.DataFrame(metrics["channels"])
    df.plot.scatter(x="modality_bacc", y="disease_bacc",
                    c=df.mask_type.map({"content": "tab:blue", "style": "tab:red"}))

Usage:
    python -m eval.channel_probe \\
        --run-dir runs/<model_id> \\
        --feature-levels 0 1 \\
        --task ad_cn \\
        [--views both]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from typing import Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from eval.disease_classifier import (
    FeatureCache,
    _build_datasets,
    _filter_cache,
    _load_encoder,
    _load_run_args,
    _MemmapRowView,
    _precompute_features,
    _resolve_task,
    _resolve_views,
)

logger = logging.getLogger("channel_probe")


# ---------------------------------------------------------------------------
# GAP extraction
# ---------------------------------------------------------------------------


def _compute_gap(cache: FeatureCache, level: int, batch_size: int = 64) -> np.ndarray:
    """Global average pool over spatial dims → ``(N, C)`` float32 array."""
    feat = cache.feats[level]
    N = len(feat) if isinstance(feat, _MemmapRowView) else feat.shape[0]
    C = feat.shape[1]

    gap = np.empty((N, C), dtype=np.float32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        if isinstance(feat, _MemmapRowView):
            rows = feat.row_idx[start:end]
            block = np.ascontiguousarray(feat.mm[rows])
        elif isinstance(feat, np.memmap):
            block = np.ascontiguousarray(feat[start:end])
        else:
            block = feat[start:end].numpy()
        gap[start:end] = block.mean(axis=(2, 3, 4))
    return gap


def _channel_mask_type(cache: FeatureCache, level: int, channel: int, view_idxs: list[int]) -> str:
    """Return ``'content'``, ``'style'``, or ``'none'`` (no mask at this level)."""
    masks = cache.content_masks.get(level, {})
    if not masks:
        return "none"
    mask = masks.get(view_idxs[0])
    if mask is None:
        return "none"
    return "content" if mask[channel].item() else "style"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run-dir",
        required=True,
        help="Directory containing settings.json + vqvae_model.pt",
    )
    p.add_argument("--feature-levels", type=int, nargs="+", default=[0])
    p.add_argument("--task", default="ad_cn", choices=["all", "ad_cn"])
    p.add_argument("--views", default="both", choices=["t1", "t2", "both"])
    p.add_argument("--extract-batch-size", type=int, default=4)
    p.add_argument("--cache-dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument(
        "--cache-mode",
        default="memmap",
        choices=["memmap", "memory"],
    )
    p.add_argument("--feature-cache-dir", default=None)
    p.add_argument("--keep-cache", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-name", default="channel_probe")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cli):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    device = torch.device(cli.device)

    logger.info(f"Loading run from {cli.run_dir}")
    run_args = _load_run_args(cli.run_dir)

    logger.info("Building VQ-VAE encoder")
    encoder = _load_encoder(cli.run_dir, run_args, device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    train_ds, val_ds = _build_datasets(run_args)
    view_idxs = _resolve_views(cli.views)
    label_remap, task_name = _resolve_task(cli.task, train_ds, val_ds)
    logger.info(f"Task: {cli.task} ({task_name}) | views: {cli.views}")

    cache_dtype = torch.float16 if cli.cache_dtype == "fp16" else torch.float32
    cache_dir: Optional[str]
    if cli.cache_mode == "memory":
        cache_dir = None
    else:
        cache_dir = cli.feature_cache_dir or os.path.join(cli.run_dir, ".channel_probe_cache")
        os.makedirs(cache_dir, exist_ok=True)

    levels = sorted(set(cli.feature_levels))

    logger.info("Precomputing features (single encoder pass) ...")
    train_cache = _precompute_features(
        encoder,
        train_ds,
        levels,
        view_idxs,
        device,
        batch_size=cli.extract_batch_size,
        workers=cli.workers,
        dtype=cache_dtype,
        cache_dir=cache_dir,
        cache_tag="train",
    )
    val_cache = _precompute_features(
        encoder,
        val_ds,
        levels,
        view_idxs,
        device,
        batch_size=cli.extract_batch_size,
        workers=cli.workers,
        dtype=cache_dtype,
        cache_dir=cache_dir,
        cache_tag="val",
    )

    # Disease-filtered caches (e.g. drop MCI for ad_cn).
    def _apply_task(cache: FeatureCache) -> FeatureCache:
        valid = torch.tensor([int(l.item()) in label_remap for l in cache.labels])
        cache = _filter_cache(cache, valid)
        cache.labels = torch.tensor([label_remap[int(l.item())] for l in cache.labels], dtype=torch.long)
        return cache

    train_disease = _apply_task(train_cache)
    val_disease = _apply_task(val_cache)
    logger.info(f"After task filter: train={len(train_disease)} val={len(val_disease)}")

    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    can_probe_modality = cli.views == "both"

    # ------------------------------------------------------------------
    # Per-channel probing
    # ------------------------------------------------------------------
    channels: list[dict] = []

    for level in levels:
        logger.info(f"\n=== Level {level} ===")

        gap_train_d = _compute_gap(train_disease, level)
        gap_val_d = _compute_gap(val_disease, level)
        y_train_d = train_disease.labels.numpy()
        y_val_d = val_disease.labels.numpy()

        if can_probe_modality:
            gap_train_m = _compute_gap(train_cache, level)
            gap_val_m = _compute_gap(val_cache, level)
            y_train_m = train_cache.view_ids.numpy()
            y_val_m = val_cache.view_ids.numpy()

        C = gap_train_d.shape[1]
        logger.info(f"  {C} channels")

        for c in range(C):
            mask_type = _channel_mask_type(train_cache, level, c, view_idxs)

            # --- disease probe ---
            X_tr, X_va = gap_train_d[:, c : c + 1], gap_val_d[:, c : c + 1]
            if X_tr.std() < 1e-9:
                disease_bacc = float("nan")
            else:
                lr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=cli.seed)
                lr.fit(X_tr, y_train_d)
                disease_bacc = float(balanced_accuracy_score(y_val_d, lr.predict(X_va)))

            # --- modality probe ---
            modality_bacc = None
            if can_probe_modality:
                X_tr_m = gap_train_m[:, c : c + 1]
                X_va_m = gap_val_m[:, c : c + 1]
                if X_tr_m.std() < 1e-9:
                    modality_bacc = float("nan")
                else:
                    lr_m = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=cli.seed)
                    lr_m.fit(X_tr_m, y_train_m)
                    modality_bacc = float(balanced_accuracy_score(y_val_m, lr_m.predict(X_va_m)))

            rec = {
                "level": level,
                "channel": c,
                "mask_type": mask_type,
                "disease_bacc": disease_bacc,
            }
            if modality_bacc is not None:
                rec["modality_bacc"] = modality_bacc
            channels.append(rec)

        # Per-level summary table.
        for mt in ("content", "style", "none"):
            subset = [r for r in channels if r["level"] == level and r["mask_type"] == mt]
            if not subset:
                continue
            d = [r["disease_bacc"] for r in subset if _finite(r["disease_bacc"])]
            line = f"  {mt:>7s} ({len(subset):3d} ch): disease bACC={_safe_mean(d):.3f}"
            if can_probe_modality:
                m = [r["modality_bacc"] for r in subset if _finite(r.get("modality_bacc"))]
                line += f"  modality bACC={_safe_mean(m):.3f}"
            logger.info(line)

    # ------------------------------------------------------------------
    # Aggregated summary
    # ------------------------------------------------------------------
    aggregated: dict[str, dict] = {}
    for level in levels:
        for mt in ("content", "style", "none"):
            subset = [r for r in channels if r["level"] == level and r["mask_type"] == mt]
            if not subset:
                continue
            key = f"L{level}_{mt}"
            d = [r["disease_bacc"] for r in subset if _finite(r["disease_bacc"])]
            agg: dict = {
                "n_channels": len(subset),
                "disease_bacc_mean": _safe_mean(d),
                "disease_bacc_std": _safe_std(d),
                "disease_bacc_max": float(np.max(d)) if d else None,
            }
            if can_probe_modality:
                m = [r["modality_bacc"] for r in subset if _finite(r.get("modality_bacc"))]
                agg["modality_bacc_mean"] = _safe_mean(m)
                agg["modality_bacc_std"] = _safe_std(m)
                agg["modality_bacc_max"] = float(np.max(m)) if m else None
            aggregated[key] = agg

    metrics_path = os.path.join(cli.run_dir, f"{cli.out_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {"channels": channels, "aggregated": aggregated, "config": vars(cli)},
            f,
            indent=2,
        )
    logger.info(f"\nMetrics written to {metrics_path}")

    # ------------------------------------------------------------------
    # Cleanup memmap files
    # ------------------------------------------------------------------
    if cache_dir is not None and not cli.keep_cache:
        del train_cache, val_cache, train_disease, val_disease
        gc.collect()
        for fn in os.listdir(cache_dir):
            if fn.startswith("feats_") and fn.endswith(".dat"):
                try:
                    os.remove(os.path.join(cache_dir, fn))
                except OSError as e:
                    logger.warning(f"  Could not remove {fn}: {e}")
        try:
            os.rmdir(cache_dir)
        except OSError:
            pass
        logger.info(f"Cache cleaned ({cache_dir})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finite(x) -> bool:
    return x is not None and np.isfinite(x)


def _safe_mean(xs: list) -> Optional[float]:
    return float(np.mean(xs)) if xs else None


def _safe_std(xs: list) -> Optional[float]:
    return float(np.std(xs)) if xs else None


if __name__ == "__main__":
    main(parse_args())
