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
* **Optional 0-contrastive baseline anchor** → Δ (model − baseline) isolates what
  the objective *earned* on top of the architecture.
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
):
    """Score all four blocks (cc, cs, ss, sc) + block-MCC for one encoder's features."""
    cnames, snames = info["content_names"], info["style_names"]
    has_split = info["has_split"]

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
        "detail": {
            "content2content": cc,
            "content2style": cs,
            "style2style": ss,
            "style2content": sc,
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
):
    """Turn extracted representations into one comparison row (torch-free).

    ``reprs`` maps a pooling key to a ``level_data`` dict (the output of
    ``_extract_synthetic_representations``).  ``info`` is that level's factor_info.

    When ``per_encoder`` is True and view-2 features exist, the four score blocks
    and block-MCC are computed separately for each encoder and reported with a
    ``_v2`` suffix.
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
    )

    # View-invariance (always uses both encoders).
    mkey = _resolve_key("stats", avail)
    Cc = _block_array(reprs, mkey, level, _CONTENT)
    Sc = _block_array(reprs, mkey, level, _STYLE)
    Cc2 = reprs[mkey][level][_CONTENT_V2] if mkey in reprs and level in reprs[mkey] else None
    Sc2 = reprs[mkey][level][_STYLE_V2] if mkey in reprs and level in reprs[mkey] else None
    if Cc is not None and Cc2 is not None:
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

    if base is not None:
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


_BLOCK_LABELS = {
    "content2content": "content→content",
    "content2style": "content→style",
    "style2style": "style→style",
    "style2content": "style→content",
}


def iter_per_latent_rows(rows):
    """Long-format rows: one per (model, encoder, block, factor, pooling) informativeness cell."""
    for r in rows:
        for detail_key, enc_label in [("detail", "enc1"), ("detail_v2", "enc2")]:
            detail = r.get(detail_key, {})
            if not detail:
                continue
            for bkey, blabel in _BLOCK_LABELS.items():
                blk = detail.get(bkey)
                if not blk:
                    continue
                for fname, fd in blk["per_factor"].items():
                    assigned = fd.get("pooling")
                    for pool, pd in fd.get("by_pooling", {}).items():
                        yield {
                            "model": r["name"],
                            "encoder": enc_label,
                            "block": blabel,
                            "factor": fname,
                            "pooling": pool,
                            "assigned": pool == assigned,
                            "real_r2": pd["real"],
                            "real_std": pd["std"],
                            "null_r2": pd["null"],
                            "gap": pd["gap"],
                        }


def print_per_latent(rows):
    """Per-model per-latent informativeness grid: GAP test-R² of the matched block,
    one column per pooling.  ``*`` marks each factor's assigned (headline) pooling,
    so you can see whether a factor is better exposed by a non-default pooling.
    """
    print()
    print("  PER-LATENT INFORMATIVENESS  (GAP test-R2; * = assigned pooling)")
    print()
    for r in rows:
        for detail_key, enc_label in [("detail", ""), ("detail_v2", " enc2")]:
            detail = r.get(detail_key)
            if not detail:
                continue
            cc = detail.get("content2content")
            ss = detail.get("style2style")
            if not cc:
                continue
            pools = [
                k
                for k in ("gap", "stats", "patch")
                if any(k in fd.get("by_pooling", {}) for fd in cc["per_factor"].values())
            ]
            header = f"  {r['name']}{enc_label}  ({r.get('checkpoint', '?')})"
            print(header)
            print("    " + f"{'factor':24s}" + "".join(f"{p:>10s}" for p in pools))
            print("    " + "-" * (24 + 10 * len(pools)))

            def _emit(block):
                for fname, fd in block["per_factor"].items():
                    assigned = fd.get("pooling")
                    cells = ""
                    for p in pools:
                        g = fd.get("by_pooling", {}).get(p, {}).get("gap")
                        txt = f"{g:.3f}" if g is not None and np.isfinite(g) else "-"
                        cells += f"{'  ' + txt + '*' if p == assigned else '  ' + txt:>10s}"
                    print(f"    {fname:24s}{cells}")

            _emit(cc)
            if ss:
                print(f"    {'-- style --':24s}")
                _emit(ss)
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
        "block",
        "factor",
        "pooling",
        "assigned",
        "real_r2",
        "real_std",
        "null_r2",
        "gap",
    ]
    with open(per_latent_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pl_cols)
        w.writeheader()
        for prow in iter_per_latent_rows(rows):
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
    print_scorecard(merged, baseline_name)
    print_table(merged, baseline_name)
    print_per_latent(merged)
    write_outputs(merged, cli.out, baseline_name)


if __name__ == "__main__":
    main()
