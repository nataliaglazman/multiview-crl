#!/usr/bin/env python
"""Final DCI comparison across models that share architecture / channel allocation.

This is the *direct model-comparison* protocol distilled from the methodology
discussion.  It is for ranking models that differ in the objective (loss type,
contrastive on/off, seed) but **not** in content/style channel counts — if those
differ, the null floor alone is not enough and you must equalize capacity first.

Protocol
--------
* **One frozen synthetic test set**, built once and shared across every model
  (same seed + N), so differences are about the model, not the data draw.
* **Several poolings** (default gap, stats, 2x2x2).  Each ground-truth factor is
  scored under the pooling that can physically expose it (global→gap, scale→stats,
  spatial→patch); the mapping is fixed up-front (``FACTOR_POOLING``) so there is no
  max-over-poolings selection bias.
* **Cross-validated ridge probes** with multiple seeds → mean ± std error bars
  (from ``eval.identifiability_metrics``), not a single-split GBT.
* **Null floor via label permutation** → every informativeness number is reported
  as a GAP (real − null).  The GAP cancels each model's channel-count/shape
  advantage, which is what makes models directly comparable.
* **0-contrastive baseline** → scored *all-channels only* by default (its
  content/style split carries no meaning without the objective): it appears in the
  all-channels capacity table and as ``-`` placeholders in the per-block tables.
  Roughly-equal capacity (model vs baseline) shows the objective *organizes* the
  information rather than adding it.  Pass ``--baseline-per-block`` to score its
  split too (then Δ model − baseline is reported).
* **All-channels capacity** → every factor predicted from content+style together
  (GAP, per factor at its assigned pooling).  The one apples-to-apples axis: same
  total width across models, no assumption of a split.
* **Optional split-free DCI** (``--with-dci``) → GAP disentanglement/completeness
  on the all-channels rep, so even a no-split vanilla baseline gets a real
  component-wise disentanglement number.  GBT-based, so off by default (slower).
* **Ranks on the theory-aligned headline metrics** — content→style leakage,
  block-MCC, view-invariance — never the capacity-bound content→content diagonal.

Usage
-----
    python -m eval.run_dci_compare \
        --run-dirs runs/infonce runs/moco runs/barlow \
        --baseline runs/no_contrastive \
        --num-samples 2000 --level 0 --out dci_compare_out

The torch-dependent parts (model load + encoder forward) are lazily imported so
the scoring/ranking logic stays unit-testable on plain numpy (see ``__main__``).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os

import numpy as np
from joblib import Parallel, delayed

from eval.identifiability_metrics import block_mcc, cv_probe_r2, view_invariance

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Each factor is scored under the pooling that can physically expose it.  Fixed
# in advance so the headline number is not a max over noisy pooling estimates.
FACTOR_POOLING = {
    # global morphometry — present in the channel mean
    "brain_size": "gap",
    "ventricle_size": "gap",
    "cortical_thickness": "gap",
    "temporal_atrophy": "gap",
    "lr_asymmetry": "gap",
    "sulcal_widening": "gap",
    # localized — needs spatial layout
    "lesion_x": "patch",
    "lesion_y": "patch",
    "lesion_z": "patch",
    # intensity / scale / noise — needs variance statistics
    "gain": "stats",
    "bias": "gap",
    "noise_sigma": "stats",
}

# Index of the content / style array inside a level_data tuple from
# eval.dci._extract_synthetic_representations: (content, style, content_v2, style_v2, info)
_CONTENT, _STYLE = 0, 1
_CONTENT_V2, _STYLE_V2 = 2, 3


# --------------------------------------------------------------------------- #
# Pooling parsing
# --------------------------------------------------------------------------- #


def parse_poolings(spec):
    """``"gap,stats,2x2x2"`` → ``[("gap","gap"), ("stats","stats"), ("patch",(2,2,2))]``.

    Returns a list of ``(key, value)`` where ``key`` is the bucket used by
    ``FACTOR_POOLING`` and ``value`` is what ``_extract_synthetic_representations``
    expects (``"gap"``/``"stats"`` or a ``(D,H,W)`` tuple).
    """
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok in ("gap", "stats"):
            out.append((tok, tok))
        elif "x" in tok:
            out.append(("patch", tuple(int(x) for x in tok.split("x"))))
        else:
            raise ValueError(f"unrecognized pooling token: {tok!r} (use gap, stats, or e.g. 2x2x2)")
    return out


def _resolve_key(want, avail):
    """Map a desired pooling bucket to one that was actually computed."""
    if want in avail:
        return want
    for fb in ("stats", "gap", "patch"):
        if fb in avail:
            return fb
    return next(iter(avail))


# --------------------------------------------------------------------------- #
# Null floors (label permutation) for the CV probes
# --------------------------------------------------------------------------- #


def _null_cv_r2(X, y, n_null, seeds, rng):
    vals = [cv_probe_r2(X, y[rng.permutation(len(y))], seeds=seeds)["mean"] for _ in range(n_null)]
    return float(np.mean(vals)) if vals else float("nan")


def _null_block_mcc(X, Z, n_null, seeds, rng):
    vals = [block_mcc(X, Z[rng.permutation(len(Z))], seeds=seeds)["mean"] for _ in range(n_null)]
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# Scoring (pure numpy — no torch, unit-testable)
# --------------------------------------------------------------------------- #


def _block_array(reprs, key, level, block_idx):
    ld = reprs.get(key)
    if ld is None or level not in ld:
        return None
    # A tuple/list of indices means "concatenate these blocks" — used to build the
    # split-free all-channels array (content+style) for the capacity probe.
    if isinstance(block_idx, (tuple, list)):
        parts = [ld[level][i] for i in block_idx]
        parts = [p for p in parts if p is not None and p.shape[1] > 0]
        return np.hstack(parts) if parts else None
    return ld[level][block_idx]


def _probe_cell(X, y, seeds, perm_indices):
    """One (factor, pooling) cell: real CV-R² + permutation null floor."""
    r = cv_probe_r2(X, y, seeds=seeds)
    null_vals = [cv_probe_r2(X, y[pi], seeds=seeds)["mean"] for pi in perm_indices]
    nl = float(np.mean(null_vals)) if null_vals else float("nan")
    return {"real": r["mean"], "std": r["std"], "null": nl, "gap": r["mean"] - nl}


def _score_block(reprs, level, block_idx, factors, names, avail, n_null, seeds, rng, n_jobs=1):
    """Per-factor informativeness for one repr→factor block, under every pooling.

    ``block_idx`` selects the content (0) or style (1) array; ``factors``/``names``
    select which ground-truth columns to predict.  For each factor the GAP
    (real − null) test-R² is computed under *every* available pooling and kept in
    ``by_pooling``; the headline ``gap`` is that factor's assigned pooling
    (``FACTOR_POOLING``).  Returns ``{"mean_gap", "per_factor"}``.
    """
    pools = [k for k in ("gap", "stats", "patch") if k in avail]

    # Collect work items; pre-draw permutation indices so the shared rng is
    # consumed deterministically regardless of worker ordering.
    work = []
    for j, name in enumerate(names):
        for pk in pools:
            X = _block_array(reprs, pk, level, block_idx)
            if X is None or X.shape[1] == 0:
                continue
            perm_indices = [rng.permutation(factors.shape[0]) for _ in range(n_null)]
            work.append((name, j, pk, X, perm_indices))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_probe_cell)(X, factors[:, j], seeds, pi) for _name, j, _pk, X, pi in work
    )

    per = {}
    for (name, _j, pk, _X, _pi), cell in zip(work, results):
        per.setdefault(name, {})
        per[name][pk] = cell

    for name in list(per):
        by_pooling = per[name]
        assigned = _resolve_key(FACTOR_POOLING.get(name, "stats"), avail)
        head = by_pooling.get(
            assigned,
            {
                "gap": float("nan"),
                "real": float("nan"),
                "null": float("nan"),
                "std": float("nan"),
            },
        )
        per[name] = {
            "gap": head["gap"],
            "real": head["real"],
            "null": head["null"],
            "std": head["std"],
            "pooling": assigned,
            "by_pooling": by_pooling,
        }
    gaps = [v["gap"] for v in per.values() if np.isfinite(v["gap"])]
    return {
        "mean_gap": float(np.mean(gaps)) if gaps else float("nan"),
        "per_factor": per,
    }


def _has_v2(reprs, level):
    """Check whether any pooling produced non-empty view-2 features at *level*."""
    for ld in reprs.values():
        if level in ld:
            arr = ld[level][_CONTENT_V2]
            if arr is not None and arr.shape[1] > 0:
                return True
    return False


_DCI_KEYS = ("dci_d", "dci_d_null", "dci_d_gap", "dci_c", "dci_c_null", "dci_c_gap")


def _score_dci(reprs, level, block_idx, gt_content, gt_style, avail, n_null, rng, train_ratio=0.8):
    """GAP DCI disentanglement + completeness on one representation block.

    ``block_idx`` is normally ``(content_idx, style_idx)`` so DCI is measured on the
    full (all-channels) representation — the split-free disentanglement axis that is
    defined even for a vanilla, no-split baseline.  Reuses the GBT importance +
    label-permutation null floor from ``eval.dci`` (the same path as
    ``compute_dci_synthetic``), at the richest available pooling.  Returns real /
    null / gap for both D and C; ``gap = real - null`` cancels the shape advantage of
    a wider latent so models with different channel counts stay comparable.
    """
    nan = float("nan")
    blank = {k: nan for k in _DCI_KEYS}
    mkey = _resolve_key("stats", avail)
    X = _block_array(reprs, mkey, level, block_idx)
    if X is None or X.shape[1] == 0 or X.shape[0] < 20:
        return {**blank, "dci_pooling": None}

    F = np.hstack([gt_content, gt_style]) if gt_style is not None and gt_style.shape[1] else gt_content
    split = int(X.shape[0] * train_ratio)
    factor_types = ["continuous"] * F.shape[1]

    from eval.dci import _compute_dci, _null_permuted_dci  # pulls torch in lazily

    try:
        real, _ = _compute_dci(X[:split].T, F[:split].T, X[split:].T, F[split:].T, factor_types)
        rd, rc = real["disentanglement"], real["completeness"]
    except Exception as e:
        logger.warning("DCI (real) failed: %s", e)
        return {**blank, "dci_pooling": mkey}

    null = _null_permuted_dci(X, F, split, factor_types, n_null, rng)
    nd, nc = null.get("disentanglement", nan), null.get("completeness", nan)
    return {
        "dci_d": rd,
        "dci_d_null": nd,
        "dci_d_gap": rd - nd,
        "dci_c": rc,
        "dci_c_null": nc,
        "dci_c_gap": rc - nc,
        "dci_pooling": mkey,
    }


def _score_one_encoder(
    reprs,
    level,
    content_idx,
    style_idx,
    gt_content,
    gt_style,
    info,
    avail,
    n_null,
    seeds,
    rng,
    n_jobs,
    all_only=False,
    with_dci=False,
):
    """Score the four split blocks (cc, cs, ss, sc) + block-MCC for one encoder's
    features, plus the split-free ``all``-channels capacity (full representation →
    every factor).  ``all_only`` skips the split blocks and returns only that
    capacity — for a baseline with no meaningful content/style distinction.
    ``with_dci`` adds GAP DCI disentanglement/completeness on the all-channels rep."""
    cnames, snames = info["content_names"], info["style_names"]
    has_split = info["has_split"]

    # All-channels capacity: every factor predicted from content+style together.
    # Computed for every model so it is the one apples-to-apples axis (same total
    # width across models) whether or not the split means anything.
    allb = _score_block(
        reprs,
        level,
        (content_idx, style_idx),
        np.hstack([gt_content, gt_style]),
        cnames + snames,
        avail,
        n_null,
        seeds,
        rng,
        n_jobs=n_jobs,
    )
    allb["content_names"] = list(cnames)
    allb["style_names"] = list(snames)

    # Split-free DCI (component-wise) on the all-channels rep — the disentanglement
    # axis that is defined even for a no-split baseline.  Off by default (GBT cost).
    dci = (
        _score_dci(
            reprs,
            level,
            (content_idx, style_idx),
            gt_content,
            gt_style,
            avail,
            n_null,
            rng,
        )
        if with_dci
        else {**{k: float("nan") for k in _DCI_KEYS}, "dci_pooling": None}
    )

    if all_only:
        nan = float("nan")
        return {
            "leak_c2s": nan,
            "info_c2c": nan,
            "suff_s2s": nan,
            "leak_s2c": nan,
            "mcc_cc": nan,
            "mcc_cc_null": nan,
            "mcc_cs": nan,
            "mcc_cs_null": nan,
            "separation": nan,
            "info_all": allb["mean_gap"],
            **dci,
            "detail": {
                "content2content": None,
                "content2style": None,
                "style2style": None,
                "style2content": None,
                "all": allb,
            },
        }

    cc = _score_block(
        reprs,
        level,
        content_idx,
        gt_content,
        cnames,
        avail,
        n_null,
        seeds,
        rng,
        n_jobs=n_jobs,
    )
    cs = _score_block(
        reprs,
        level,
        content_idx,
        gt_style,
        snames,
        avail,
        n_null,
        seeds,
        rng,
        n_jobs=n_jobs,
    )
    ss = (
        _score_block(
            reprs,
            level,
            style_idx,
            gt_style,
            snames,
            avail,
            n_null,
            seeds,
            rng,
            n_jobs=n_jobs,
        )
        if has_split
        else None
    )
    sc = (
        _score_block(
            reprs,
            level,
            style_idx,
            gt_content,
            cnames,
            avail,
            n_null,
            seeds,
            rng,
            n_jobs=n_jobs,
        )
        if has_split
        else None
    )

    mkey = _resolve_key("stats", avail)
    Cc = _block_array(reprs, mkey, level, content_idx)
    mcc_cc = block_mcc(Cc, gt_content, seeds=seeds)["mean"] if Cc is not None and Cc.shape[1] else float("nan")
    mcc_cc_null = (
        _null_block_mcc(Cc, gt_content, n_null, seeds, rng) if Cc is not None and Cc.shape[1] else float("nan")
    )
    mcc_cs = block_mcc(Cc, gt_style, seeds=seeds)["mean"] if Cc is not None and Cc.shape[1] else float("nan")
    mcc_cs_null = _null_block_mcc(Cc, gt_style, n_null, seeds, rng) if Cc is not None and Cc.shape[1] else float("nan")

    sep = mcc_cc - mcc_cs if np.isfinite(mcc_cc) and np.isfinite(mcc_cs) else float("nan")

    return {
        "leak_c2s": cs["mean_gap"],
        "info_c2c": cc["mean_gap"],
        "suff_s2s": ss["mean_gap"] if ss else float("nan"),
        "leak_s2c": sc["mean_gap"] if sc else float("nan"),
        "mcc_cc": mcc_cc,
        "mcc_cc_null": mcc_cc_null,
        "mcc_cs": mcc_cs,
        "mcc_cs_null": mcc_cs_null,
        "separation": sep,
        "info_all": allb["mean_gap"],
        **dci,
        "detail": {
            "content2content": cc,
            "content2style": cs,
            "style2style": ss,
            "style2content": sc,
            "all": allb,
        },
    }


def score_reprs(
    reprs,
    gt_content,
    gt_style,
    info,
    level,
    n_null=3,
    seeds=(0, 1, 2),
    n_jobs=1,
    per_encoder=False,
    all_only=False,
    with_dci=False,
):
    """Turn extracted representations into one comparison row (torch-free).

    ``reprs`` maps a pooling key to a ``level_data`` dict (the output of
    ``_extract_synthetic_representations``).  ``info`` is that level's factor_info.

    When ``per_encoder`` is True and view-2 features exist, the four score blocks
    and block-MCC are computed separately for each encoder and reported with a
    ``_v2`` suffix.  ``with_dci`` adds the split-free GAP DCI scores.
    """
    avail = set(reprs.keys())
    rng = np.random.RandomState(0)

    enc1 = _score_one_encoder(
        reprs,
        level,
        _CONTENT,
        _STYLE,
        gt_content,
        gt_style,
        info,
        avail,
        n_null,
        seeds,
        rng,
        n_jobs,
        all_only=all_only,
        with_dci=with_dci,
    )

    # View-invariance (uses both encoders).  Skipped for an all-channels-only
    # baseline: with no content block, "can content predict the view" is moot.
    mkey = _resolve_key("stats", avail)
    Cc = _block_array(reprs, mkey, level, _CONTENT)
    Sc = _block_array(reprs, mkey, level, _STYLE)
    Cc2 = reprs[mkey][level][_CONTENT_V2] if mkey in reprs and level in reprs[mkey] else None
    Sc2 = reprs[mkey][level][_STYLE_V2] if mkey in reprs and level in reprs[mkey] else None
    if not all_only and Cc is not None and Cc2 is not None:
        s1 = Sc if Sc is not None and Sc.shape[1] else Cc[:, :0]
        s2 = Sc2 if Sc2 is not None and Sc2.shape[1] else Cc2[:, :0]
        vi = view_invariance(Cc, Cc2, s1, s2, seeds=seeds)
        content_view, style_view, chance = (
            vi["content_acc"],
            vi.get("style_acc", float("nan")),
            vi["chance"],
        )
    else:
        content_view = style_view = chance = float("nan")

    row = {
        "n_content_channels": info["n_content_channels"],
        "n_style_channels": info["n_style_channels"],
        **enc1,
        "content_view": content_view,
        "style_view": style_view,
        "view_chance": chance,
    }

    if per_encoder and _has_v2(reprs, level):
        enc2 = _score_one_encoder(
            reprs,
            level,
            _CONTENT_V2,
            _STYLE_V2,
            gt_content,
            gt_style,
            info,
            avail,
            n_null,
            seeds,
            rng,
            n_jobs,
            all_only=all_only,
            with_dci=with_dci,
        )
        for k, v in enc2.items():
            if k == "detail":
                row["detail_v2"] = v
            else:
                row[k + "_v2"] = v

    return row


# --------------------------------------------------------------------------- #
# Per-model driver (torch — lazily imported)
# --------------------------------------------------------------------------- #


def _resolve_checkpoint(run_dir, name):
    """Prefer ``<run_dir>/<name>``; fall back to vqvae_model.pt if it is missing.

    Lets a comparison mix runs that have a best-by-loss checkpoint with runs that
    only have a latest one, instead of dropping the latter.  Returns the preferred
    path unchanged when neither exists so the loader can raise a clear error.
    """
    preferred = os.path.join(run_dir, name)
    if os.path.exists(preferred):
        return preferred
    fallback = os.path.join(run_dir, "vqvae_model.pt")
    if name != "vqvae_model.pt" and os.path.exists(fallback):
        logger.warning("%s not found in %s — falling back to vqvae_model.pt", name, run_dir)
        return fallback
    return preferred


def evaluate_model(
    name,
    run_dir,
    dataset,
    poolings,
    level,
    n_null,
    seeds,
    batch_size,
    num_workers,
    device,
    checkpoint=None,
    n_jobs=1,
    per_encoder=False,
    all_only=False,
    with_dci=False,
):
    """Load a model, extract representations under each pooling, return a row."""
    from eval.dci import _extract_synthetic_representations
    from eval.run_dci_synthetic import load_model_from_run_dir

    logger.info("=== evaluating %s (%s) ===", name, run_dir)
    model, _args, device = load_model_from_run_dir(run_dir, checkpoint, device)

    reprs, gt_content, gt_style, info = {}, None, None, None
    for key, value in poolings:
        level_data, gc, gsv1, _gsv2 = _extract_synthetic_representations(
            model, dataset, device, batch_size, num_workers, pooling=value
        )
        reprs[key] = level_data
        if gt_content is None:
            gt_content, gt_style = gc, gsv1
        if info is None and level in level_data:
            info = level_data[level][4]

    del model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if info is None:
        raise RuntimeError(f"level {level} not found in encoder outputs for {name}")

    row = score_reprs(
        reprs,
        gt_content,
        gt_style,
        info,
        level,
        n_null=n_null,
        seeds=seeds,
        n_jobs=n_jobs,
        per_encoder=per_encoder,
        all_only=all_only,
        with_dci=with_dci,
    )
    row["name"] = name
    row["run_dir"] = run_dir
    row["checkpoint"] = os.path.basename(checkpoint) if checkpoint else "vqvae_model.pt"
    return row


# --------------------------------------------------------------------------- #
# Interpretive layer — derived human-readable scores (all 0..1, higher = better)
# --------------------------------------------------------------------------- #
#
# The raw protocol metrics each point a different direction (leak↓, mcc↑,
# view≈0.5) and sit on different scales, so reading "the whole picture" means
# holding the theory in your head.  These derived scores re-express the same
# numbers as four intuitive questions, every one oriented so 1.0 = ideal:
#
#   content_anatomy  — does content capture the shared (anatomy) factors?
#   content_purity   — does content stay free of modality/style?
#   style_modality   — does style capture the modality signal?
#   style_purity     — does style stay free of anatomy?
#
# Their mean is a single 0..1 "disentanglement" health score (→ letter grade).
# All are clipped to [0,1] and degrade to NaN (not a crash) when an input is
# missing, so a no-style-split model still scores its content half.

_EPS = 1e-6


def _clip01(x):
    if x is None or not np.isfinite(x):
        return float("nan")
    return float(min(1.0, max(0.0, x)))


def _headroom(real, null):
    """Fraction of the achievable ``[null, 1]`` band that ``real`` reaches.

    Normalising by ``1 - null`` removes each block's shape/chance advantage, so
    0 = no better than the permutation floor, 1 = perfect readout.
    """
    if real is None or null is None or not (np.isfinite(real) and np.isfinite(null)):
        return float("nan")
    denom = 1.0 - null
    if denom <= _EPS:
        return float("nan")
    return _clip01((real - null) / denom)


def _leak_rejection(leak_gap, signal_gap):
    """1 = no leak; 0 = leak as large as the block's legitimate signal."""
    if leak_gap is None or not np.isfinite(leak_gap):
        return float("nan")
    leak = max(0.0, leak_gap)
    sig = max(
        signal_gap if (signal_gap is not None and np.isfinite(signal_gap)) else 0.0,
        _EPS,
    )
    return _clip01(1.0 - leak / sig)


def _view_invariance_score(content_view):
    """1 = content predicts the view at chance (0.5); 0 = predicts it perfectly."""
    if content_view is None or not np.isfinite(content_view):
        return float("nan")
    return _clip01(1.0 - (content_view - 0.5) / 0.5)


def _view_specificity_score(style_view):
    """1 = style predicts the view perfectly; 0 = no better than chance."""
    if style_view is None or not np.isfinite(style_view):
        return float("nan")
    return _clip01((style_view - 0.5) / 0.5)


def _mean_finite(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _grade(score):
    if score is None or not np.isfinite(score):
        return "?"
    for thresh, letter in ((0.80, "A"), (0.65, "B"), (0.50, "C"), (0.35, "D")):
        if score >= thresh:
            return letter
    return "F"


def derive_scores(row, suffix=""):
    """Map raw protocol metrics to interpretable 0..1 sub-scores + an overall.

    ``suffix`` selects an encoder's metrics (``""`` = enc1, ``"_v2"`` = enc2).
    The view-identity probes are model-level (shared across encoders), so they
    only feed the enc1 scores; enc2 purity/modality fall back to the
    informativeness-based signals alone.  Returns keys with ``suffix`` appended.
    """
    s = suffix

    content_anatomy = _headroom(row.get("mcc_cc" + s), row.get("mcc_cc_null" + s))

    purity_info = _leak_rejection(row.get("leak_c2s" + s), row.get("info_c2c" + s))
    purity_view = _view_invariance_score(row.get("content_view")) if not s else float("nan")
    content_purity = _mean_finite([purity_info, purity_view])

    suff = _clip01(row.get("suff_s2s" + s))
    spec_view = _view_specificity_score(row.get("style_view")) if not s else float("nan")
    style_modality = _mean_finite([suff, spec_view])

    style_purity = _leak_rejection(row.get("leak_s2c" + s), row.get("suff_s2s" + s))

    overall = _mean_finite([content_anatomy, content_purity, style_modality, style_purity])
    out = {
        "content_anatomy": content_anatomy,
        "content_purity": content_purity,
        "style_modality": style_modality,
        "style_purity": style_purity,
        "disentanglement": overall,
        "grade": _grade(overall),
    }
    return {k + s: v for k, v in out.items()}


def attach_scores(rows):
    """Add derived scores in-place to every row (enc1, and enc2 when present)."""
    for r in rows:
        r.update(derive_scores(r, ""))
        if "separation_v2" in r:
            r.update(derive_scores(r, "_v2"))
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

_HEADLINE_COLS = [
    "separation",
    "leak_c2s",
    "mcc_cc",
    "mcc_cs",
    "content_view",
    "style_view",
    "suff_s2s",
]

_SUBSCORES = [
    ("content_anatomy", "Content: captures anatomy"),
    ("content_purity", "Content: rejects modality"),
    ("style_modality", "Style:   captures modality"),
    ("style_purity", "Style:   rejects anatomy"),
]


def _bar(score, width=20):
    if score is None or not np.isfinite(score):
        return "[" + "?" * width + "]"
    filled = int(round(_clip01(score) * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _verdict_lines(row):
    """A few plain-English findings, each tagged [+] good / [~] caution / [!] problem."""
    out = []
    ca = row.get("content_anatomy", float("nan"))
    leak = row.get("leak_c2s", float("nan"))
    cv = row.get("content_view", float("nan"))
    sv = row.get("style_view", float("nan"))
    sep = row.get("separation", float("nan"))

    if np.isfinite(ca):
        if ca >= 0.60:
            out.append("[+] content captures the shared anatomy factors")
        elif ca >= 0.35:
            out.append("[~] content captures anatomy only partially")
        else:
            out.append("[!] content barely captures the shared factors")
    if np.isfinite(leak):
        if leak <= 0.05:
            out.append("[+] negligible style leakage into content")
        elif leak <= 0.15:
            out.append("[~] some style information leaks into content")
        else:
            out.append("[!] substantial style leakage into content")
    if np.isfinite(cv):
        if cv <= 0.55:
            out.append("[+] a linear probe cannot read modality from content")
        elif cv <= 0.70:
            out.append("[~] content weakly predicts the modality/view")
        else:
            out.append("[!] content strongly predicts the modality (view leak)")
    if np.isfinite(sv):
        if sv >= 0.80:
            out.append("[+] style carries the modality signal")
        elif sv >= 0.60:
            out.append("[~] style only partly captures the modality")
        else:
            out.append("[!] style fails to capture the modality")
    if np.isfinite(sep) and sep < 0.10:
        out.append("[!] weak content/style separation overall")
    return out


def _disent_of(r):
    v = r.get("disentanglement", float("nan"))
    return v if (v is not None and np.isfinite(v)) else -1.0


def print_scorecard(rows, baseline_name=None):
    """Headline readable view: per-model sub-score bars, grade, and verdicts.

    Ranks by the overall disentanglement health score (↓).  This is the layer to
    read first; ``print_table`` below has the underlying protocol numbers.
    """
    ranked = sorted(rows, key=_disent_of, reverse=True)
    w = 76
    print()
    print("=" * w)
    print("  MODEL SCORECARD   (every score 0-100%, higher = better)")
    print("=" * w)
    for r in ranked:
        tag = "   <- baseline" if r["name"] == baseline_name else ""
        overall = r.get("disentanglement", float("nan"))
        opct = f"{overall * 100:3.0f}%" if np.isfinite(overall) else " ? "
        print()
        print(f"  {r['name']}{tag}")
        print(f"      {'OVERALL (disentanglement)':26s} {_bar(overall)} {opct}  grade {r.get('grade', '?')}")
        print(f"      {'-' * (26 + 22 + 11)}")
        for key, label in _SUBSCORES:
            sc = r.get(key, float("nan"))
            pct = f"{sc * 100:3.0f}%" if np.isfinite(sc) else " ? "
            print(f"      {label:26s} {_bar(sc)} {pct}")
        for line in _verdict_lines(r):
            print(f"        {line}")
    print("=" * w)
    print(
        "  anatomy  : content informative about shared factors (block-MCC above null)\n"
        "  rejects  : content can't predict the view + low style leakage\n"
        "  modality : style predicts the view & is sufficient for style factors\n"
        "  Read this first; the protocol table below has the raw numbers.\n"
    )


_COL_W = 10  # width per metric column


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-".center(_COL_W)
    return f"{v:.{nd}f}".center(_COL_W)


def _model_width(rows):
    return max(len(r["name"]) for r in rows) + 2 if rows else 18


def _print_metric_row(label, mw, r, suffix="", tag=""):
    """Print one data row.  *suffix* selects v2 keys when non-empty."""
    s = suffix
    chans = f"{r['n_content_channels']}/{r['n_style_channels']}"
    print(
        f"  {label:<{mw}s} {chans:>6s}"
        f"  {_fmt(r.get('separation' + s))}"
        f"  {_fmt(r.get('leak_c2s' + s))}"
        f"  {_fmt(r.get('mcc_cc' + s))}"
        f"  {_fmt(r.get('mcc_cs' + s))}"
        f"  {_fmt(r.get('content_view') if not s else float('nan'))}"
        f"  {_fmt(r.get('style_view') if not s else float('nan'))}"
        f"  {_fmt(r.get('suff_s2s' + s))}"
        f"{tag}"
    )


def print_table(rows, baseline_name=None):
    """Ranked stdout table.  Ranks by separation↑ (tie-break leakage↓)."""
    base = next((r for r in rows if r["name"] == baseline_name), None)
    ranked = sorted(
        rows,
        key=lambda r: (
            -(r["separation"] if np.isfinite(r["separation"]) else -9),
            r["leak_c2s"],
        ),
    )
    has_v2 = any("separation_v2" in r for r in rows)
    mw = _model_width(rows) + (6 if has_v2 else 0)  # extra room for enc label
    w = mw + 8 + 7 * (_COL_W + 2) + 2
    hdr_label = "model" if not has_v2 else "model / encoder"

    print()
    print("=" * w)
    print("  DCI MODEL COMPARISON   (GAP = real - null)")
    print("=" * w)
    print(
        f"  {hdr_label:<{mw}s} {'c/s':>6s}"
        f"  {'SEP':^{_COL_W}s}"
        f"  {'leak c>s':^{_COL_W}s}"
        f"  {'mcc c>c':^{_COL_W}s}"
        f"  {'mcc c>s':^{_COL_W}s}"
        f"  {'c>view':^{_COL_W}s}"
        f"  {'s>view':^{_COL_W}s}"
        f"  {'suff s>s':^{_COL_W}s}"
    )
    print(
        f"  {'':>{mw}s} {'':>6s}"
        f"  {'higher':^{_COL_W}s}"
        f"  {'lower':^{_COL_W}s}"
        f"  {'higher':^{_COL_W}s}"
        f"  {'lower':^{_COL_W}s}"
        f"  {'~0.5':^{_COL_W}s}"
        f"  {'higher':^{_COL_W}s}"
        f"  {'higher':^{_COL_W}s}"
    )
    print("  " + "-" * (w - 2))

    for r in ranked:
        tag = "  *base" if r["name"] == baseline_name else ""
        if has_v2:
            _print_metric_row(r["name"] + " enc1", mw, r, suffix="", tag=tag)
            if "separation_v2" in r:
                _print_metric_row(r["name"] + " enc2", mw, r, suffix="_v2")
            print()
        else:
            _print_metric_row(r["name"], mw, r, tag=tag)

    if base is not None and np.isfinite(base.get("separation", float("nan"))):
        print("  " + "-" * (w - 2))
        print(f"  Delta vs baseline '{baseline_name}'")
        for r in ranked:
            if r["name"] == baseline_name:
                continue
            d = {
                k: r.get(k, float("nan")) - base.get(k, float("nan"))
                for k in ("separation", "leak_c2s", "mcc_cc", "content_view")
            }
            print(
                f"  {r['name']:<{mw}s} {'':>6s}"
                f"  {_fmt(d['separation'])}"
                f"  {_fmt(d['leak_c2s'])}"
                f"  {_fmt(d['mcc_cc'])}"
                f"  {'':^{_COL_W}s}"
                f"  {_fmt(d['content_view'])}"
            )

    print("=" * w)
    print(
        "  SEP = mcc(c>c) - mcc(c>s)   |   leak c>s near 0 = content is style-invariant\n"
        "  info_c2c is capacity-bound (high even at 0 contrastive) -- not shown; see JSON.\n"
    )


# block key -> (predicted_from, factor_type, relationship, want).  These spell
# out what each repr→factor block means so the CSV is self-describing: a reader
# can filter `relationship == leakage and gap > 0.1` without knowing the protocol.
_BLOCK_SEMANTICS = {
    "content2content": ("content", "content", "signal", "high"),
    "content2style": ("content", "style", "leakage", "low"),
    "style2style": ("style", "style", "signal", "high"),
    "style2content": ("style", "content", "leakage", "low"),
}


def _round(x, nd=4):
    return round(float(x), nd) if x is not None and np.isfinite(x) else None


def iter_per_latent_rows(rows):
    """Long-format rows: one per (model, encoder, factor, predicted_from, pooling).

    Columns are de-jargonised — ``relationship`` (signal vs leakage), ``want``
    (high/low), and ``is_assigned`` (the headline pooling) — so the file is
    filterable in a spreadsheet without consulting the protocol docstring.
    """
    for r in rows:
        for detail_key, enc_label in [("detail", "enc1"), ("detail_v2", "enc2")]:
            detail = r.get(detail_key, {})
            if not detail:
                continue
            for bkey, (pred_from, ftype, rel, want) in _BLOCK_SEMANTICS.items():
                blk = detail.get(bkey)
                if not blk:
                    continue
                for fname, fd in blk["per_factor"].items():
                    assigned = fd.get("pooling")
                    for pool, pd in fd.get("by_pooling", {}).items():
                        yield {
                            "model": r["name"],
                            "encoder": enc_label,
                            "factor": fname,
                            "factor_type": ftype,
                            "predicted_from": pred_from,
                            "relationship": rel,
                            "want": want,
                            "pooling": pool,
                            "is_assigned": pool == assigned,
                            "real_r2": _round(pd.get("real")),
                            "null_r2": _round(pd.get("null")),
                            "gap": _round(pd.get("gap")),
                            "real_std": _round(pd.get("std")),
                        }

            allb = detail.get("all")
            if allb:
                cset = set(allb.get("content_names", []))
                for fname, fd in allb["per_factor"].items():
                    assigned = fd.get("pooling")
                    for pool, pd in fd.get("by_pooling", {}).items():
                        yield {
                            "model": r["name"],
                            "encoder": enc_label,
                            "factor": fname,
                            "factor_type": "content" if fname in cset else "style",
                            "predicted_from": "all",
                            "relationship": "capacity",
                            "want": "high",
                            "pooling": pool,
                            "is_assigned": pool == assigned,
                            "real_r2": _round(pd.get("real")),
                            "null_r2": _round(pd.get("null")),
                            "gap": _round(pd.get("gap")),
                            "real_std": _round(pd.get("std")),
                        }


def _num(x, nd=3):
    return f"{x:.{nd}f}" if x is not None and np.isfinite(x) else "  -  "


def _minibar(gap, width=10):
    """Inline ``[####......]`` bar for a GAP in [0,1]; negatives render empty."""
    if gap is None or not np.isfinite(gap):
        return " " * (width + 2)
    filled = int(round(min(1.0, max(0.0, gap)) * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def print_per_latent(rows):
    """Per-model, per-factor breakdown: for every ground-truth factor, the GAP
    test-R² of predicting it from its OWN block (signal, want high) next to the
    OTHER block (leak, want low), at the factor's assigned pooling.

    Putting signal and leak side by side is the readable disentanglement story:
    a content factor should light up the signal bar and leave the leak bar empty,
    and vice-versa for style.  The full per-pooling sweep lives in the CSV.
    """
    print()
    print("  PER-LATENT BREAKDOWN   (GAP = real - null at each factor's assigned pooling)")
    print("  SIGNAL = predicted from its own block (want high)   LEAK = from the other block (want low)")

    _NUMW, _CELLW = 5, 18  # "0.650" and "0.650 [##########]"

    def _cell(g):
        return f"{_num(g):>{_NUMW}s} {_minibar(g)}"

    for r in rows:
        for detail_key, enc_label in [("detail", ""), ("detail_v2", "  [enc2]")]:
            detail = r.get(detail_key)
            if not detail:
                continue
            cc = detail.get("content2content")  # content factors from content (signal)
            cs = detail.get("content2style")  # style   factors from content (leak)
            ss = detail.get("style2style")  # style   factors from style   (signal)
            sc = detail.get("style2content")  # content factors from style   (leak)
            if not cc:
                continue

            names = list(cc["per_factor"]) + (list(ss["per_factor"]) if ss else [])
            fw = max([len(n) for n in names] + [14])
            sep_w = fw + 2 + 6 + 2 + _CELLW + 4 + _CELLW

            def _emit(kind, signal_pf, leak_pf, sig_src, leak_src):
                print(
                    f"    {kind + ' factor':<{fw}s}  {'pool':<6s}  "
                    f"{'SIGNAL: ' + sig_src:<{_CELLW}s}    {'LEAK: ' + leak_src:<{_CELLW}s}"
                )
                print("    " + "-" * sep_w)
                for fname, fd in signal_pf.items():
                    pool = fd.get("pooling") or "?"
                    sgap = fd.get("gap")
                    lgap = (leak_pf.get(fname, {}) or {}).get("gap") if leak_pf else None
                    leak_cell = _cell(lgap) if leak_pf is not None else f"{'n/a':>{_NUMW}s}"
                    print(f"    {fname:<{fw}s}  {pool:<6s}  {_cell(sgap)}    {leak_cell}")

            print()
            print(f"  {r['name']}{enc_label}  ({r.get('checkpoint', '?')})")
            _emit(
                "content",
                cc["per_factor"],
                sc["per_factor"] if sc else None,
                "from content",
                "from style",
            )
            if ss:
                print()
                _emit(
                    "style",
                    ss["per_factor"],
                    cs["per_factor"] if cs else None,
                    "from style",
                    "from content",
                )
    print()


def print_capacity_table(rows, baseline_name=None):
    """All-channels capacity: GAP test-R² of predicting each factor from the FULL
    representation (content+style together), with no split.

    This is the apples-to-apples axis — every model gets a number here (including a
    baseline scored all-channels-only), because it assumes nothing about a
    content/style boundary.  Roughly-equal capacity across models is the point: it
    shows the objective *organizes* the information rather than adding it.
    """
    have = [r for r in rows if np.isfinite(r.get("info_all", float("nan")))]
    if not have:
        return
    ranked = sorted(have, key=lambda r: r.get("info_all", float("-inf")), reverse=True)
    print()
    print("  ALL-CHANNELS CAPACITY   (GAP = real - null; full representation -> each factor)")
    print("  higher = more factor information present somewhere in the representation")
    for r in ranked:
        allb = (r.get("detail") or {}).get("all")
        mean = r.get("info_all", float("nan"))
        tag = "   <- baseline (all-channels only)" if r["name"] == baseline_name else ""
        chans = r.get("n_content_channels", 0) + r.get("n_style_channels", 0)
        print()
        print(f"  {r['name']}{tag}   {chans}ch   mean {_num(mean)} {_minibar(mean)}")
        dgap = r.get("dci_d_gap", float("nan"))
        if np.isfinite(dgap):
            cgap = r.get("dci_c_gap", float("nan"))
            print(
                f"      {'DCI (gap)':14s} D {_num(dgap)} {_minibar(dgap)}   "
                f"C {_num(cgap)} {_minibar(cgap)}   (disentangle / complete; split-free)"
            )
        if allb:
            cset = set(allb.get("content_names", []))
            fw = max([len(n) for n in allb["per_factor"]] + [16])
            for fname, fd in allb["per_factor"].items():
                kind = "content" if fname in cset else "style"
                g = fd.get("gap")
                pool = fd.get("pooling") or "?"
                print(f"      {kind:7s} {fname:<{fw}s} {pool:<6s} {_num(g)} {_minibar(g)}")
    print()


def write_outputs(rows, out_dir, baseline_name=None):
    os.makedirs(out_dir, exist_ok=True)
    attach_scores(rows)  # idempotent — ensure derived columns exist even if called directly
    _score_cols = [
        "grade",
        "disentanglement",
        "content_anatomy",
        "content_purity",
        "style_modality",
        "style_purity",
    ]
    _enc1_cols = [
        "info_all",
        "dci_d",
        "dci_d_null",
        "dci_d_gap",
        "dci_c",
        "dci_c_null",
        "dci_c_gap",
        "separation",
        "leak_c2s",
        "info_c2c",
        "suff_s2s",
        "leak_s2c",
        "mcc_cc",
        "mcc_cc_null",
        "mcc_cs",
        "mcc_cs_null",
    ]
    has_v2 = any("separation_v2" in r for r in rows)
    flat_cols = [
        "name",
        "run_dir",
        "checkpoint",
        "n_content_channels",
        "n_style_channels",
        *_score_cols,
        *_enc1_cols,
        "content_view",
        "style_view",
        "view_chance",
        *([c + "_v2" for c in (_score_cols + _enc1_cols)] if has_v2 else []),
        "num_samples",
        "poolings",
    ]
    csv_path = os.path.join(out_dir, "dci_compare.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in flat_cols})

    per_latent_path = os.path.join(out_dir, "dci_compare_per_latent.csv")
    pl_cols = [
        "model",
        "encoder",
        "factor",
        "factor_type",
        "predicted_from",
        "relationship",
        "want",
        "pooling",
        "is_assigned",
        "real_r2",
        "null_r2",
        "gap",
        "real_std",
    ]
    # Sort so a factor's signal and leak rows sit together, assigned pooling first.
    pl_rows = sorted(
        iter_per_latent_rows(rows),
        key=lambda d: (
            d["model"],
            d["encoder"],
            d["factor_type"],
            d["factor"],
            d["relationship"],
            not d["is_assigned"],
        ),
    )
    with open(per_latent_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pl_cols)
        w.writeheader()
        for prow in pl_rows:
            w.writerow(prow)

    json_path = os.path.join(out_dir, "dci_compare.json")
    with open(json_path, "w") as f:
        json.dump({"baseline": baseline_name, "models": rows}, f, indent=2, default=float)
    logger.info("Wrote %s, %s, %s", csv_path, per_latent_path, json_path)


def _load_existing_rows(out_dir):
    """Load previously-saved model rows from dci_compare.json, for incremental runs."""
    path = os.path.join(out_dir, "dci_compare.json")
    if not os.path.exists(path):
        return [], None
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("models", []), data.get("baseline")
    except Exception as e:
        logger.warning("Could not read existing %s (%s) — starting fresh.", path, e)
        return [], None


def _merge_rows(existing, new):
    """Merge model rows by name: new wins, existing order preserved, genuinely-new appended."""
    by_name = {r["name"]: r for r in existing}
    order = [r["name"] for r in existing]
    for r in new:
        if r["name"] not in by_name:
            order.append(r["name"])
        by_name[r["name"]] = r
    return [by_name[n] for n in order]


def _settings_of(row):
    """The eval settings that must match for rows to be comparable in one leaderboard."""
    return (
        row.get("num_samples"),
        row.get("poolings"),
        row.get("level"),
        row.get("n_null"),
        row.get("seeds"),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(description="Final DCI comparison across models (shared architecture).")
    p.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Run directories to compare (settings.json each).",
    )
    p.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Labels (default: basename of each run-dir).",
    )
    p.add_argument(
        "--baseline",
        default=None,
        help="Run-dir to anchor Δ (e.g. the 0-contrastive model).",
    )
    p.add_argument(
        "--checkpoint-name",
        default="vqvae_model.pt",
        help="Checkpoint filename loaded from every run-dir. Use vqvae_best.pt for the best-by-loss copy "
        "(recommended for a final comparison; same choice for all models keeps it fair).",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Frozen test-set size, shared across models.",
    )
    p.add_argument(
        "--poolings",
        default="gap,stats,2x2x2",
        help="Comma list: gap, stats, and/or DxHxW (e.g. 2x2x2).",
    )
    p.add_argument("--level", type=int, default=0, help="Encoder level to compare on.")
    p.add_argument("--seeds", default="0,1,2", help="Probe CV seeds.")
    p.add_argument("--n-null", type=int, default=3, help="Permutations for the null floor.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel probe jobs (-1 = all cores, 1 = sequential).",
    )
    p.add_argument(
        "--per-encoder",
        action="store_true",
        help="Score each encoder separately (for separate_encoders models).",
    )
    p.add_argument(
        "--baseline-per-block",
        action="store_true",
        help="Also score the --baseline run on the content/style split (leakage, "
        "separation, view-invariance).  Default: the baseline is scored all-channels "
        "only, since with no contrastive objective its split is not meaningful — it "
        "appears only in the capacity table, with placeholders in the per-block tables.",
    )
    p.add_argument(
        "--with-dci",
        action="store_true",
        help="Also compute split-free GAP DCI disentanglement/completeness on the "
        "all-channels representation for every model (incl. a no-split vanilla "
        "baseline).  Uses GBT importance — noticeably slower; off by default.",
    )
    p.add_argument("--out", default="dci_compare_out", help="Output directory.")
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Overwrite any existing results in --out instead of merging the new models into them.",
    )
    cli = p.parse_args()

    poolings = parse_poolings(cli.poolings)
    seeds = tuple(int(s) for s in cli.seeds.split(","))

    # Assemble the model list; the baseline (if separate) is evaluated too.
    specs = list(cli.run_dirs)
    names = cli.names or [os.path.basename(os.path.normpath(d)) for d in specs]
    if len(names) != len(specs):
        p.error("--names must have one entry per --run-dirs")
    baseline_name = None
    if cli.baseline:
        if cli.baseline in specs:
            baseline_name = names[specs.index(cli.baseline)]
        else:
            baseline_name = os.path.basename(os.path.normpath(cli.baseline))
            specs = [cli.baseline] + specs
            names = [baseline_name] + names

    # One frozen test set, built from the first run's settings, reused for all.
    import torch

    from eval.run_dci_synthetic import build_synthetic_test_set, load_run_args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_args = load_run_args(specs[0])
    dataset = build_synthetic_test_set(ref_args, cli.num_samples)
    logger.info(
        "Frozen test set: %d samples, shared across %d model(s).",
        cli.num_samples,
        len(specs),
    )

    rows = []
    for name, run_dir in zip(names, specs):
        try:
            rows.append(
                evaluate_model(
                    name,
                    run_dir,
                    dataset,
                    poolings,
                    cli.level,
                    cli.n_null,
                    seeds,
                    cli.batch_size,
                    cli.num_workers,
                    device,
                    checkpoint=_resolve_checkpoint(run_dir, cli.checkpoint_name),
                    n_jobs=cli.n_jobs,
                    per_encoder=cli.per_encoder,
                    with_dci=cli.with_dci,
                    all_only=(baseline_name is not None and name == baseline_name and not cli.baseline_per_block),
                )
            )
        except Exception as e:
            logger.error("Skipping %s (%s): %s", name, run_dir, e)

    if not rows:
        logger.error("No models evaluated successfully.")
        return

    # Stamp eval settings on each row (provenance + the merge-comparability check).
    meta = {
        "num_samples": cli.num_samples,
        "poolings": cli.poolings,
        "level": cli.level,
        "n_null": cli.n_null,
        "seeds": cli.seeds,
    }
    for r in rows:
        r.update(meta)

    existing, existing_baseline = ([], None) if cli.fresh else _load_existing_rows(cli.out)
    if existing:
        if any(_settings_of(e) != _settings_of(rows[0]) for e in existing):
            logger.warning(
                "Existing results used different eval settings (num-samples/poolings/level/n-null/seeds); "
                "the merged leaderboard would mix them. Use --fresh to start clean, or match the settings."
            )
        logger.info(
            "Merging %d new with %d existing model(s) in %s.",
            len(rows),
            len(existing),
            cli.out,
        )
    merged = _merge_rows(existing, rows)
    if baseline_name is None:
        baseline_name = existing_baseline

    attach_scores(merged)
    print_capacity_table(merged, baseline_name)
    print_scorecard(merged, baseline_name)
    print_table(merged, baseline_name)
    print_per_latent(merged)
    write_outputs(merged, cli.out, baseline_name)


if __name__ == "__main__":
    main()
