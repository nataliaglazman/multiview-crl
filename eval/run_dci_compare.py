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
* **0-contrastive baseline** → scored as one *all-content* block by default (its
  content/style split carries no meaning without the objective, so all channels are
  treated as content): its content-side metrics (separation, leakage, block-MCC,
  content→view) and capacity are filled and directly comparable to the models, with
  the style-side N/A.  Pass ``--baseline-per-block`` to score its own (arbitrary)
  split instead.  The ``HEAD-TO-HEAD vs BASELINE`` block at the top of the report is
  the headline model-vs-baseline view; roughly-equal capacity there shows the
  objective *organizes* the information rather than adding it.
* **All-channels capacity** → every factor predicted from content+style together
  (GAP, per factor at its assigned pooling).  The one apples-to-apples axis: same
  total width across models, no assumption of a split.
* **Optional split-free DCI** (``--with-dci``) → GAP disentanglement/completeness
  in three scopes: ``dci`` (all-channels × all-factors, stats pooling — defined even
  for a no-split vanilla baseline), ``dci_content`` (content-channels × content-factors)
  and ``dci_patch`` (all-channels at the PATCH pooling, so spatially-localized factors
  like lesions count).  GBT-based, so off by default (slower).
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
import re
import time

import numpy as np
from joblib import Parallel, delayed

from eval.identifiability_metrics import block_mcc, cv_probe_r2, cv_probe_r2_multi, view_invariance

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Each factor is scored under the pooling that can physically expose it.  Fixed
# in advance so the headline number is not a max over noisy pooling estimates.
FACTOR_POOLING = {
    # global morphometry — present in the channel mean.  Routed to gap, not patch: at
    # patch pooling an untrained encoder already reads these at R2 0.78-0.92 (measured:
    # brain_size floor 0.897, lr_asymmetry 0.923), leaving under 0.07 of headroom, so two
    # models cannot be told apart there.  Use --factor-pooling patch for a one-off
    # all-on-one-rung view instead of editing this table.
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

# --probe-dim sentinel: reduce only the blocks that are actually in the p>>n regime.
# Lives here rather than in content_rank_pca (which already imports from this module) so
# the two scripts cannot drift on the rule.
PROBE_DIM_AUTO = "auto"


def _auto_probe_dim(n, d):
    """``--probe-dim auto``: the width to reduce a (n, d) block to, or 0 for "leave alone".

    Only blocks in the p>>n regime are touched. The patch block is the one that gets
    there: at patch_grid 8^3 x 44 channels it is 22528 features against N~2000 samples,
    where a ridge probe on a signal-free target returns a NEGATIVE R^2 — measured, and
    it inverted the sign of the lesion dims in the scaling-2-vs-4 comparison until the
    reduction was applied. ``gap`` (44) and ``stats`` (176) are well-conditioned at that
    N and are left at full width, so this does not silently reshape the readouts that
    were never overfitting.

    Measured on this generator (untrained encoder, patch 8^3, N=200, iid factors): with
    the labels SHUFFLED, the unreduced probe returns mean R^2 -0.270 while every reduced
    width returns ~-0.04. The bias is roughly constant, so it is invisible on a strong
    factor (brain_size reads 0.991 either way) and decisive on a weak one (lesion_y
    -0.173 unreduced, +0.047 at 64). That asymmetry is why this cannot be left to the
    caller: it silently penalises exactly the factors under investigation.
    """
    budget = max(1, n // 4)
    return min(64, budget) if d > budget else 0


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


def _null_cv_r2(X, y, n_null, seeds, rng, kind="ridge"):
    vals = [cv_probe_r2(X, y[rng.permutation(len(y))], seeds=seeds, kind=kind)["mean"] for _ in range(n_null)]
    return float(np.mean(vals)) if vals else float("nan")


def _null_block_mcc(X, Z, n_null, seeds, rng, kind="ridge"):
    vals = [block_mcc(X, Z[rng.permutation(len(Z))], seeds=seeds, kind=kind)["mean"] for _ in range(n_null)]
    return float(np.mean(vals)) if vals else float("nan")


def mcc_by_pooling(reprs, level, gt_content, seeds=(0, 1, 2), kind="ridge", per_factor_pooling="patch", names=()):
    """Block-MCC of the content block against ``gt_content`` at EVERY available pooling.

    ``_score_one_encoder`` hardcodes stats pooling for its headline ``mcc_cc``, and stats
    is ``[mean, std, amax, amin]`` over space — all permutation-invariant over voxels, so
    ``lesion_x/y/z`` (positions) are structurally unreadable there.  That silently drops 3
    of the 9 content factors, and they are the ones a reconstruction model recovers best:
    measured, baseline vs contrastive flips from stats +0.011 to patch -0.043.  Reporting
    the whole ladder makes the reversal visible instead of only showing the rung that
    happens to flatter the contrastive arm.

    The ladder is an unweighted MEAN over factors, which hides a sign split: measured on
    one run, brain_size ROSE while cortical_thickness, lr_asymmetry, temporal_atrophy and
    sulcal_widening all FELL, and the average reported a uniform decline.  ``block_mcc``
    computes the per-factor breakdown on its way to that mean, so it is emitted for
    ``per_factor_pooling`` at no extra cost.

    Returns ``{"mcc_cc_pool_<key>": mean, ...}`` plus, for ``per_factor_pooling``, the
    ``mcc_cc_factor_*`` / ``mcc_cc_assignment_identity`` keys.  Read every one of these as
    a gap over an untrained twin (``--floor``): at patch pooling a random projection of
    this architecture already scores ~0.86, so the absolute value is nearly all floor.
    """
    out = {}
    if gt_content is None:
        return out
    for key in reprs:
        blk = _block_array(reprs, key, level, _CONTENT)
        if blk is None or not blk.shape[1]:
            continue
        res = block_mcc(blk, gt_content, seeds=seeds, kind=kind)
        out[f"mcc_cc_pool_{key}"] = res["mean"]
        if key != per_factor_pooling:
            continue
        pf, sd, dg = res.get("per_factor"), res.get("per_factor_std"), res.get("per_factor_diag")
        if pf is None:
            continue
        for j in range(len(pf)):
            fname = names[j] if j < len(names) else f"factor{j}"
            out[f"mcc_cc_factor_{fname}"] = float(pf[j])
            out[f"mcc_cc_factor_std_{fname}"] = float(sd[j])
            out[f"mcc_cc_factor_diag_{fname}"] = float(dg[j])
        out["mcc_cc_assignment_identity"] = res.get("assignment_identity", float("nan"))
    return out


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


def _score_block(
    reprs,
    level,
    block_idx,
    factors,
    names,
    avail,
    n_null,
    seeds,
    rng,
    n_jobs=1,
    kind="ridge",
    factor_pooling="assigned",
):
    """Per-factor informativeness for one repr→factor block, under every pooling.

    ``block_idx`` selects the content (0) or style (1) array; ``factors``/``names``
    select which ground-truth columns to predict.  For each factor the GAP
    (real − null) test-R² is computed under *every* available pooling and kept in
    ``by_pooling``; the headline ``gap`` is that factor's assigned pooling
    (``FACTOR_POOLING``).  Returns ``{"mean_gap", "per_factor"}``.

    All of a pooling's factors and their permutation nulls share one ``X``, so they
    are scored in a single batched probe per pooling (``cv_probe_r2_multi``) that
    reuses one StandardScaler + ridge decomposition across every target instead of
    re-decomposing ``X`` once per factor and per permutation.
    """
    pools = [k for k in ("gap", "stats", "patch") if k in avail]
    pool_X = [(pk, _block_array(reprs, pk, level, block_idx)) for pk in pools]
    pool_X = [(pk, X) for pk, X in pool_X if X is not None and X.shape[1] > 0]

    K = len(names)
    n = factors.shape[0]
    if not names or not pool_X:
        return {"mean_gap": float("nan"), "per_factor": {}}

    # Pre-draw the null permutations in the same (factor, pooling) order the per-cell
    # implementation used, so the structural null floor is unchanged and the shared
    # rng is consumed deterministically regardless of worker ordering.
    perms = {}
    for j in range(K):
        for pk, _X in pool_X:
            perms[(pk, j)] = [rng.permutation(n) for _ in range(n_null)]

    def _targets(pk):
        # [K real columns | per-factor null columns] — one batched target set per X.
        cols = [factors[:, j] for j in range(K)]
        for j in range(K):
            cols.extend(factors[pi, j] for pi in perms[(pk, j)])
        return np.column_stack(cols)

    tasks = [(pk, X, _targets(pk)) for pk, X in pool_X]
    results = Parallel(n_jobs=n_jobs)(delayed(cv_probe_r2_multi)(X, Y, seeds=seeds, kind=kind) for _pk, X, Y in tasks)

    per = {name: {} for name in names}
    for (pk, _X, _Y), res in zip(tasks, results):
        means, stds = res["mean"], res["std"]
        for j, name in enumerate(names):
            real, std = float(means[j]), float(stds[j])
            start = K + j * n_null
            nl = float(np.mean(means[start : start + n_null])) if n_null > 0 else float("nan")
            per[name][pk] = {"real": real, "std": std, "null": nl, "gap": real - nl}
    per = {name: by_pool for name, by_pool in per.items() if by_pool}

    for name in list(per):
        by_pooling = per[name]
        # ``factor_pooling`` overrides FACTOR_POOLING's per-factor routing and reads every
        # factor at one rung.  The routing is the default because it keeps each factor off
        # its ceiling as well as on a pooling that can express it: at patch an untrained
        # encoder reads the morphometry factors at 0.78-0.92, leaving no headroom to
        # compare models in.  The override is for asking "how does this factor read HERE".
        want = FACTOR_POOLING.get(name, "stats") if factor_pooling == "assigned" else factor_pooling
        assigned = _resolve_key(want, avail)
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


_DCI_BASE = ("d", "d_null", "d_gap", "c", "c_null", "c_gap")
# Three scopes: "dci" = all-channels × all-factors at the richest (stats) pooling
# (baseline-comparable, defined even for a no-split model); "dci_content" =
# content-channels × content-factors (the model-internal "is content disentangled"
# number, undefined for a no-split baseline); "dci_patch" = all-channels × all-factors
# at the PATCH pooling (spatial-grid codes, so localized factors like lesions count).
_DCI_KEYS = tuple(f"dci_{s}" for s in _DCI_BASE)
_DCI_CONTENT_KEYS = tuple(f"dci_content_{s}" for s in _DCI_BASE)
_DCI_PATCH_KEYS = tuple(f"dci_patch_{s}" for s in _DCI_BASE)


def _blank_dci(prefix):
    return {
        **{f"{prefix}_{s}": float("nan") for s in _DCI_BASE},
        f"{prefix}_pooling": None,
        f"{prefix}_n_codes": float("nan"),
    }


def mean_std_structs(structs):
    """Elementwise ``(mean, std)`` over a list of identically-shaped nested dicts.

    Used to collapse several untrained-floor scorings — one per init seed — into a single
    floor plus its across-seed spread.  That spread is the point: an unseeded floor is one
    draw of a random projection, and every ``learned = trained - floor`` number inherits
    its noise silently.  Measured on this project, three near-equivalent lesion axes drew
    floors spanning 0.16 of block-MCC, which is the same order as the effects being
    claimed over them.

    Numeric leaves are averaged over the finite values; dicts recurse; anything else
    (pooling names, factor-name lists, None) is taken from the first struct unchanged
    since it is metadata, not a measurement.  With one struct the std is 0.0.
    """
    if not structs:
        return {}, {}
    first = structs[0]
    if isinstance(first, dict):
        mean, std = {}, {}
        for k in first:
            vals = [s[k] for s in structs if isinstance(s, dict) and k in s]
            mean[k], std[k] = mean_std_structs(vals)
        return mean, std
    if isinstance(first, bool) or not isinstance(first, (int, float, np.floating, np.integer)):
        return first, first
    finite = [float(v) for v in structs if np.isfinite(_as_float(v))]
    if not finite:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite))


def _as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _select_codes(X, max_codes):
    """Cap a DCI block's width by SELECTING the highest-variance codes, never rotating.

    DCI is basis-dependent by construction: ``disentanglement`` asks "is this CODE
    dedicated to few factors", which is a statement about one specific feature, and
    ``completeness`` normalises by the code count.  So the block handed to DCI must be
    made of the model's own features.  Projecting onto principal components — what
    ``_reduce_reprs`` does for the probes — replaces every code with a variance-ordered
    MIXTURE of channels, and then D/C describe the PCA basis rather than the encoder.
    Selection keeps each retained code a real ``(channel, patch)`` feature, so the score
    still refers to the representation.

    Returns ``(X, n_codes)``.  ``max_codes <= 0`` disables the cap.
    """
    if not max_codes or X.shape[1] <= max_codes:
        return X, X.shape[1]
    keep = np.argsort(-X.var(axis=0), kind="stable")[:max_codes]
    keep.sort()  # original column order, so a code index still maps back to a channel
    return X[:, keep], int(max_codes)


def _score_dci(
    reprs,
    level,
    block_idx,
    gt_content,
    gt_style,
    avail,
    n_null,
    rng,
    key_prefix="dci",
    pooling_key=None,
    train_ratio=0.8,
    max_codes=0,
    n_jobs=1,
):
    """GAP DCI disentanglement + completeness on one representation block.

    ``block_idx`` selects the codes: ``(content_idx, style_idx)`` for the all-channels
    rep, or ``content_idx`` alone for content-only.  ``gt_content``/``gt_style`` pick
    the factor columns (pass ``gt_style=None`` for content-only).  ``pooling_key``
    picks which pooling's features are the codes (default: the richest, ``stats``;
    pass ``"patch"`` for the spatial-grid rep that exposes localized factors).
    Reuses the GBT importance + label-permutation null floor from ``eval.dci`` (the
    same path as ``compute_dci_synthetic``).  Returns real / null / gap for D and C
    under ``key_prefix``; ``gap = real - null`` cancels the shape advantage of a
    wider latent so different channel counts stay comparable.

    Callers must pass the UNREDUCED ``reprs`` — see ``_select_codes`` for why a
    PCA-projected block cannot carry a D/C score.  ``max_codes`` bounds the GBT cost by
    selecting codes instead, and the width actually used is returned as ``_n_codes`` so
    two rows can be checked for commensurability before they are compared.
    """
    nan = float("nan")
    blank = _blank_dci(key_prefix)
    mkey = pooling_key or _resolve_key("stats", avail)
    X = _block_array(reprs, mkey, level, block_idx)
    if X is None or X.shape[1] == 0 or X.shape[0] < 20:
        return blank
    X, n_codes = _select_codes(X, max_codes)

    F = np.hstack([gt_content, gt_style]) if gt_style is not None and gt_style.shape[1] else gt_content
    split = int(X.shape[0] * train_ratio)
    factor_types = ["continuous"] * F.shape[1]

    from eval.dci import _compute_dci, _null_permuted_dci  # pulls torch in lazily

    try:
        real, _ = _compute_dci(X[:split].T, F[:split].T, X[split:].T, F[split:].T, factor_types, n_jobs=n_jobs)
        rd, rc = real["disentanglement"], real["completeness"]
    except Exception as e:
        logger.warning("DCI (real) failed: %s", e)
        return {**blank, f"{key_prefix}_pooling": mkey, f"{key_prefix}_n_codes": n_codes}

    null = _null_permuted_dci(X, F, split, factor_types, n_null, rng, n_jobs=n_jobs)
    nd, nc = null.get("disentanglement", nan), null.get("completeness", nan)
    return {
        f"{key_prefix}_d": rd,
        f"{key_prefix}_d_null": nd,
        f"{key_prefix}_d_gap": rd - nd,
        f"{key_prefix}_c": rc,
        f"{key_prefix}_c_null": nc,
        f"{key_prefix}_c_gap": rc - nc,
        f"{key_prefix}_pooling": mkey,
        f"{key_prefix}_n_codes": n_codes,
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
    probe_kind="ridge",
    dci_reprs=None,
    dci_max_codes=0,
    factor_pooling="assigned",
):
    """Score the four split blocks (cc, cs, ss, sc) + block-MCC for one encoder's
    features, plus the split-free ``all``-channels capacity (full representation →
    every factor).  ``all_only`` skips the split blocks and returns only that
    capacity — for a baseline with no meaningful content/style distinction.
    ``with_dci`` adds GAP DCI disentanglement/completeness on the all-channels rep.

    ``dci_reprs`` is the pre-``_reduce_reprs`` representation, used for the DCI scopes
    only: the probes want a well-conditioned block, DCI wants the model's own basis, and
    those are different requirements.  Defaults to ``reprs`` for callers that never
    reduced."""
    cnames, snames = info["content_names"], info["style_names"]
    has_split = info["has_split"]
    # DCI reads the unreduced block; every probe/MCC below keeps reading `reprs`.
    dreprs = reprs if dci_reprs is None else dci_reprs

    # Bind the probe kind and the factor routing once so every block, null and MCC in this
    # row is scored by the same estimator on the same rung — a row mixing linear and
    # nonlinear probes, or mixing routings, is not internally comparable.
    def _sblock(*a, **kw):
        return _score_block(*a, kind=probe_kind, factor_pooling=factor_pooling, **kw)

    # All-channels capacity: every factor predicted from content+style together.
    # Computed for every model so it is the one apples-to-apples axis (same total
    # width across models) whether or not the split means anything.
    allb = _sblock(
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

    # Component-wise DCI, three scopes (off by default — GBT cost):
    #   dci_*          — all-channels × all-factors at stats pooling: defined even for
    #                    a no-split baseline, so it is the apples-to-apples baseline axis.
    #   dci_content_*  — content-channels × content-factors: the model-internal
    #                    "is content disentangled" number; NaN for a no-split baseline.
    #   dci_patch_*    — all-channels × all-factors at the PATCH pooling: spatial-grid
    #                    codes, so localized factors (lesions) contribute; NaN if no
    #                    patch pooling was requested.
    #
    # All three read `dreprs`, so the three scopes are always in the SAME basis (the
    # encoder's own channels) and stay commensurable with each other.  Before this they
    # read the probe-reduced blocks, which under `--probe-dim auto` reduces whichever
    # scope happens to be in the p>>n regime and leaves the others alone: at N=2000 that
    # put `dci_patch` on 64 principal components while `dci`/`dci_content` stayed on raw
    # channels, and the three rows printed together were not comparable.
    if with_dci:
        dci = _score_dci(
            dreprs,
            level,
            (content_idx, style_idx),
            gt_content,
            gt_style,
            avail,
            n_null,
            rng,
            key_prefix="dci",
            max_codes=dci_max_codes,
            n_jobs=n_jobs,
        )
        if not all_only and has_split:
            dci.update(
                _score_dci(
                    dreprs,
                    level,
                    content_idx,
                    gt_content,
                    None,
                    avail,
                    n_null,
                    rng,
                    key_prefix="dci_content",
                    max_codes=dci_max_codes,
                    n_jobs=n_jobs,
                )
            )
        else:
            dci.update(_blank_dci("dci_content"))
        if "patch" in avail:
            dci.update(
                _score_dci(
                    dreprs,
                    level,
                    (content_idx, style_idx),
                    gt_content,
                    gt_style,
                    avail,
                    n_null,
                    rng,
                    key_prefix="dci_patch",
                    pooling_key="patch",
                    max_codes=dci_max_codes,
                    n_jobs=n_jobs,
                )
            )
        else:
            dci.update(_blank_dci("dci_patch"))
    else:
        dci = {
            **_blank_dci("dci"),
            **_blank_dci("dci_content"),
            **_blank_dci("dci_patch"),
        }

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

    cc = _sblock(
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
    cs = _sblock(
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
        _sblock(
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
        _sblock(
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
    mcc_cc = (
        block_mcc(Cc, gt_content, seeds=seeds, kind=probe_kind)["mean"]
        if Cc is not None and Cc.shape[1]
        else float("nan")
    )
    mcc_cc_null = (
        _null_block_mcc(Cc, gt_content, n_null, seeds, rng, kind=probe_kind)
        if Cc is not None and Cc.shape[1]
        else float("nan")
    )
    mcc_cs = (
        block_mcc(Cc, gt_style, seeds=seeds, kind=probe_kind)["mean"]
        if Cc is not None and Cc.shape[1]
        else float("nan")
    )
    mcc_cs_null = (
        _null_block_mcc(Cc, gt_style, n_null, seeds, rng, kind=probe_kind)
        if Cc is not None and Cc.shape[1]
        else float("nan")
    )

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


def _reduce_reprs(reprs, level, probe_dim, seed=0):
    """Return a copy of *reprs* with each block at *level* PCA-reduced to
    ``min(probe_dim, width)`` components, so every model's probes see the same
    feature capacity regardless of its channel count.

    ``probe_dim=PROBE_DIM_AUTO`` resolves per block via ``_auto_probe_dim``: only the
    p>>n blocks are touched, and a well-conditioned one is left at full width. An
    explicit integer still reduces EVERY block, which is the setting to use when
    equalising a channel-count confound across models of different width — that is a
    different goal from fixing the conditioning, and only the integer serves it.

    View-1 and view-2 of a block share one PCA basis (fit on view 1) so the
    view-invariance probe is not confounded by mismatched bases.  PCA is
    unsupervised and fit on all samples — a transductive but label-free step.  The
    ``info`` dict is left untouched, so the tables still report the model's real
    channel count.  Blocks already at or below ``probe_dim`` are left unchanged.

    PCA keeps top-VARIANCE directions, not top-signal ones, so an over-aggressive width
    deletes weak factors outright (measured, same setup as ``_auto_probe_dim``:
    sulcal_widening reads 0.866 at 64 and 0.073 at 16). Treat the width as a reported
    parameter, and never compare two models scored at different values.
    """
    from sklearn.decomposition import PCA

    def _width(A):
        if probe_dim == PROBE_DIM_AUTO:
            return _auto_probe_dim(A.shape[0], A.shape[1])
        return probe_dim

    def _fit(A):
        if A is None or A.shape[1] == 0:
            return None
        k = _width(A)
        if k <= 0 or A.shape[1] <= k:
            return None
        return PCA(n_components=min(k, A.shape[0]), random_state=seed).fit(A)

    def _apply(pca, A):
        return pca.transform(A) if (pca is not None and A is not None) else A

    out = {}
    for key, ld in reprs.items():
        if level not in ld:
            out[key] = ld
            continue
        content, style, content_v2, style_v2, info = ld[level]
        pca_c, pca_s = _fit(content), _fit(style)  # fit on view-1, reuse for view-2
        new_ld = dict(ld)
        new_ld[level] = (
            _apply(pca_c, content),
            _apply(pca_s, style),
            _apply(pca_c, content_v2),
            _apply(pca_s, style_v2),
            info,
        )
        out[key] = new_ld
    return out


def _collapse_to_all_content(reprs, info, level):
    """Merge style channels into content at *level* so the model is scored as one
    all-content block with no style — the "baseline = full representation used as
    content" treatment.

    The content/style split metrics then run on the full representation, and the
    style-side blocks (ss, sc, style→view) fall out as NaN via the ``has_split``
    gate.  The all-channels capacity, ``dci`` and ``dci_patch`` are unchanged —
    they concatenate content+style regardless.  Returns ``(new_reprs, new_info)``.
    """

    def _merge(c, s):
        if c is None:
            return None
        return c if (s is None or s.shape[1] == 0) else np.hstack([c, s])

    out = {}
    for key, ld in reprs.items():
        if level not in ld:
            out[key] = ld
            continue
        content, style, content_v2, style_v2, inner = ld[level]
        merged = _merge(content, style)
        new_inner = dict(inner)
        new_inner["n_content_channels"] = merged.shape[1] if merged is not None else 0
        new_inner["n_style_channels"] = 0
        new_inner["has_split"] = False
        new_ld = dict(ld)
        new_ld[level] = (merged, None, _merge(content_v2, style_v2), None, new_inner)
        out[key] = new_ld

    new_info = dict(info)
    new_info["n_content_channels"] = info.get("n_content_channels", 0) + info.get("n_style_channels", 0)
    new_info["n_style_channels"] = 0
    new_info["has_split"] = False
    return out, new_info


def _effective_rank(X):
    """Roy–Vetterli effective rank: ``exp(entropy of the normalized singular values)``.

    A continuous count, in ``[1, n_features]``, of the dimensions a block actually
    uses.  Far below the nominal width means the representation is (partly) collapsed
    — a failure mode contrastive/VICReg/BT objectives are prone to that the probes can
    miss, and the thing the nominal channel count overstates.  Computed on the centered
    features.
    """
    if X is None or np.ndim(X) != 2 or X.shape[1] == 0 or X.shape[0] < 2:
        return float("nan")
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    sv = sv[sv > 1e-12]
    if sv.size == 0:
        return float("nan")
    p = sv / sv.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


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
    probe_dim=0,
    gt_style_v2=None,
    probe_kind="ridge",
    dci_max_codes=0,
    factor_pooling="assigned",
):
    """Turn extracted representations into one comparison row (torch-free).

    ``reprs`` maps a pooling key to a ``level_data`` dict (the output of
    ``_extract_synthetic_representations``).  ``info`` is that level's factor_info.

    When ``per_encoder`` is True and view-2 features exist, the four score blocks
    and block-MCC are computed separately for each encoder and reported with a
    ``_v2`` suffix.  ``with_dci`` adds the split-free GAP DCI scores.  ``probe_dim``
    (>0) PCA-reduces every block to that many components first, so informativeness and
    MCC are compared at equal capacity across models of different width.  DCI is scored
    on the UNREDUCED blocks regardless (``_select_codes``) and bounded by
    ``dci_max_codes`` instead, because D/C are basis-dependent and a PCA-projected block
    describes the PCA basis rather than the encoder.
    """
    # Effective rank (collapse diagnostic) on the ORIGINAL representation, before any
    # probe_dim PCA — how many dims a block actually uses vs its nominal width.  Use
    # the gap pooling (one value per channel) so it reads as "effective channels".
    _rk = "gap" if "gap" in reprs else _resolve_key("stats", set(reprs.keys()))
    ranks = {
        "content_rank": _effective_rank(_block_array(reprs, _rk, level, _CONTENT)),
        "style_rank": _effective_rank(_block_array(reprs, _rk, level, _STYLE)),
        "all_rank": _effective_rank(_block_array(reprs, _rk, level, (_CONTENT, _STYLE))),
    }
    if per_encoder and _has_v2(reprs, level):
        ranks["content_rank_v2"] = _effective_rank(_block_array(reprs, _rk, level, _CONTENT_V2))
        ranks["style_rank_v2"] = _effective_rank(_block_array(reprs, _rk, level, _STYLE_V2))
        ranks["all_rank_v2"] = _effective_rank(_block_array(reprs, _rk, level, (_CONTENT_V2, _STYLE_V2)))

    # Keep a handle on the unreduced blocks: the probes want conditioning, DCI wants the
    # encoder's own basis, and `_reduce_reprs` can only serve the first.
    dci_reprs = reprs
    if probe_dim == PROBE_DIM_AUTO or (isinstance(probe_dim, int) and probe_dim > 0):
        reprs = _reduce_reprs(reprs, level, probe_dim)
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
        probe_kind=probe_kind,
        dci_reprs=dci_reprs,
        dci_max_codes=dci_max_codes,
        factor_pooling=factor_pooling,
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
        **ranks,
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
            # Encoder 2 sees view-2 style (z_style_v2), which is drawn independently of
            # view-1's — so it MUST be scored against gsv2, not gsv1. Falls back to
            # gt_style for callers that don't supply gt_style_v2.
            gt_style_v2 if gt_style_v2 is not None else gt_style,
            info,
            avail,
            n_null,
            seeds,
            rng,
            n_jobs,
            all_only=all_only,
            with_dci=with_dci,
            probe_kind=probe_kind,
            dci_reprs=dci_reprs,
            dci_max_codes=dci_max_codes,
            factor_pooling=factor_pooling,
        )
        for k, v in enc2.items():
            if k == "detail":
                row["detail_v2"] = v
            else:
                row[k + "_v2"] = v

    # SCM-induced leakage floor: how well the GROUND-TRUTH content predicts style.
    # Under a causal graph (synthetic_causal) style genuinely depends on content, so a
    # nonzero leak_c2s is expected and does NOT by itself mean the encoder leaked — it
    # must be read against this reference rather than against zero. Yao et al. (ICLR
    # 2024, Tab. 6) do the same, reporting content->style R2 of 0.32-0.71 on their SCM.
    # Pure ground truth, so it is model-independent and identical across compared runs.
    if gt_content is not None and gt_style is not None and gt_content.shape[1] and gt_style.shape[1]:
        _ref = cv_probe_r2_multi(gt_content, gt_style, seeds=seeds, kind=probe_kind)["mean"]
        row["leak_c2s_ref"] = float(np.mean(_ref))
        row["leak_c2s_excess"] = row.get("leak_c2s", float("nan")) - row["leak_c2s_ref"]

    return row


# --------------------------------------------------------------------------- #
# Per-model driver (torch — lazily imported)
# --------------------------------------------------------------------------- #


def _resolve_checkpoint(run_dir, name=None):
    """Prefer ``<run_dir>/<name>``; fall back to vqvae_model.pt if it is missing.

    Lets a comparison mix runs that have a best-by-loss checkpoint with runs that
    only have a latest one, instead of dropping the latter.  Returns the preferred
    path unchanged when neither exists so the loader can raise a clear error.

    ``name=None`` means "the default checkpoint" — callers whose own CLI defaults the
    filename to None would otherwise land in ``os.path.join`` with a TypeError that says
    nothing about checkpoints.
    """
    name = name or "vqvae_model.pt"
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
    all_content=False,
    with_dci=False,
    probe_dim=0,
    probe_kind="ridge",
    squash_content=False,
    random_init=False,
    dci_max_codes=0,
    factor_pooling="assigned",
    init_seed=None,
):
    """Load a model, extract representations under each pooling, return a row.

    ``random_init=True`` skips the checkpoint and scores this run's exact architecture
    UNTRAINED — the zero point every other row should be read as a gap over. It has to be
    produced by this function rather than by ``run_dci_synthetic --random-init`` so the
    floor and the trained rows share the pooling, probe width, null count, factor
    distribution and FACTOR_POOLING routing; a floor measured on any other axis is not
    subtractable, which is the failure mode that has repeatedly produced phantom results
    on this project.
    """
    from eval.dci import _extract_synthetic_representations
    from eval.run_dci_synthetic import load_model_from_run_dir

    logger.info("=== evaluating %s (%s)%s ===", name, run_dir, " [UNTRAINED FLOOR]" if random_init else "")
    t0 = time.perf_counter()
    model, _args, device = load_model_from_run_dir(run_dir, checkpoint, device, random_init=random_init, seed=init_seed)
    t_load = time.perf_counter() - t0

    reprs, gt_content, gt_style, gt_style_v2, info = {}, None, None, None, None
    t0 = time.perf_counter()
    for key, value in poolings:
        level_data, gc, gsv1, gsv2 = _extract_synthetic_representations(
            model, dataset, device, batch_size, num_workers, pooling=value
        )
        if squash_content:
            # Post-squash view for a 'bounded'-mode model: the training loss shaped
            # tanh(content), so probing raw content reads a DIFFERENT parameterisation
            # than the one optimised. Applying tanh here matches them. Diagnostic only,
            # meant for a single bounded run — it will (harmlessly but pointlessly)
            # squash any other arm in the same call, so compare against the same
            # models' un-squashed run rather than mixing squashed and raw in one table.
            # Content lives at tuple positions 0 (view 1) and 2 (view 2); style is
            # untouched because bounded mode squashes only the content block.
            for _lvl, _ld in level_data.items():
                _ld = list(_ld)
                for _ci in (0, 2):
                    if _ld[_ci] is not None:
                        _ld[_ci] = np.tanh(_ld[_ci])
                level_data[_lvl] = tuple(_ld)
        reprs[key] = level_data
        if gt_content is None:
            gt_content, gt_style, gt_style_v2 = gc, gsv1, gsv2
        if info is None and level in level_data:
            info = level_data[level][4]
    t_extract = time.perf_counter() - t0

    del model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if info is None:
        raise RuntimeError(f"level {level} not found in encoder outputs for {name}")

    if all_content:
        reprs, info = _collapse_to_all_content(reprs, info, level)

    t0 = time.perf_counter()
    row = score_reprs(
        reprs,
        gt_content,
        gt_style,
        info,
        level,
        gt_style_v2=gt_style_v2,
        n_null=n_null,
        seeds=seeds,
        n_jobs=n_jobs,
        per_encoder=per_encoder,
        all_only=all_only,
        with_dci=with_dci,
        probe_dim=probe_dim,
        probe_kind=probe_kind,
        dci_max_codes=dci_max_codes,
        factor_pooling=factor_pooling,
    )
    t_score = time.perf_counter() - t0
    logger.info(
        "timing [%s]: load %.1fs | extract %.1fs (%d poolings) | score %.1fs",
        name,
        t_load,
        t_extract,
        len(poolings),
        t_score,
    )
    row["name"] = name
    row["run_dir"] = run_dir
    row["checkpoint"] = os.path.basename(checkpoint) if checkpoint else "vqvae_model.pt"
    row["baseline_all_content"] = all_content
    return row


def score_encoder_live(
    model,
    dataset,
    device,
    level=0,
    poolings=(("gap", "gap"), ("stats", "stats")),
    n_null=3,
    seeds=(0, 1),
    batch_size=32,
    num_workers=0,
    per_encoder=False,
    probe_dim=0,
    max_samples=None,
    per_factor_pooling="patch",
    n_jobs=1,
):
    """In-training counterpart of ``evaluate_model`` for a live (in-memory) encoder.

    Extracts representations under each pooling and scores them with the *same*
    ``score_reprs`` + ``attach_scores`` path the offline comparison uses, so the
    returned health composite tracks the ``run_dci_compare`` ranking by
    construction.  Use it to drive in-training checkpoint selection on synthetic
    runs (where ground-truth content/style factors are available).

    ``poolings`` is a list of ``(key, value)`` pairs: ``value`` is passed to
    ``_extract_synthetic_representations`` (``"gap"`` / ``"stats"`` / a ``(D,H,W)``
    patch tuple) and ``key`` is how that pooling is addressed in the score
    (``"gap"`` / ``"stats"`` / ``"patch"``).  ``gap`` + ``stats`` is enough for the
    composite; add ``patch`` only when spatially-localized factors matter.

    ``max_samples`` (when set) caps the dataset to its first N items so the
    periodic selection cost stays bounded regardless of val-set size.

    Note: ``_extract_synthetic_representations`` calls ``model.eval()`` and does
    not restore train mode — the caller must save/restore it.

    Returns one row dict carrying the raw protocol metrics plus the derived
    scores: ``disentanglement`` (= overall_score), ``content_anatomy``,
    ``content_purity``, ``style_modality``, ``style_purity``, ``grade``.
    """
    from eval.dci import _extract_synthetic_representations

    if max_samples and hasattr(dataset, "__len__") and len(dataset) > max_samples:
        from torch.utils.data import Subset

        dataset = Subset(dataset, range(max_samples))

    reprs, gt_content, gt_style, gt_style_v2, info = {}, None, None, None, None
    for key, value in poolings:
        level_data, gc, gsv1, gsv2 = _extract_synthetic_representations(
            model, dataset, device, batch_size, num_workers, pooling=value
        )
        reprs[key] = level_data
        if gt_content is None:
            gt_content, gt_style, gt_style_v2 = gc, gsv1, gsv2
        if info is None and level in level_data:
            info = level_data[level][4]
    if info is None:
        raise RuntimeError(f"level {level} not found in encoder outputs")

    row = score_reprs(
        reprs,
        gt_content,
        gt_style,
        info,
        level,
        gt_style_v2=gt_style_v2,
        n_null=n_null,
        seeds=seeds,
        per_encoder=per_encoder,
        probe_dim=probe_dim,
        n_jobs=n_jobs,
    )
    attach_scores([row])

    # Block-MCC at every pooling, not just the stats rung `mcc_cc` uses — see
    # `mcc_by_pooling` for why the stats rung alone is misleading.  Log-only by
    # construction: nothing here feeds `attach_scores`, `overall_score` or the selection
    # gate, so existing runs stay comparable.
    row.update(
        mcc_by_pooling(
            reprs,
            level,
            gt_content,
            seeds=seeds,
            per_factor_pooling=per_factor_pooling,
            names=(info or {}).get("content_names") or [],
        )
    )
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


_N_SECTIONS = 6


def _section(n, title, subtitle=""):
    """Slim numbered rule before each section.

    Deliberately NOT a banner box: every section below already prints its own header
    with context the divider should not duplicate (which baseline, which mode).  This
    only makes a long report navigable — "look at section 3" beats "scroll down a bit".
    """
    head = f"── {n}/{_N_SECTIONS} ─ {title} "
    print("\n" + head + "─" * max(4, 96 - len(head)))
    if subtitle:
        print(f"   {subtitle}")


def print_reading_guide(rows, baseline_name=None):
    """The four rules that decide whether any number below means anything.

    Printed first and kept short on purpose: every one of these corresponds to a mistake
    that has actually been made on this project and cost an investigation.
    """
    w = 96
    has_floor = bool(_floor_map(rows))
    pdim = next((r.get("probe_dim") for r in rows if r.get("probe_dim")), 0)
    print()
    print("=" * w)
    print("  HOW TO READ THIS REPORT")
    print("=" * w)
    print("  1. GAP, not absolute.  Every informativeness number is real - null, where null is the")
    print("     same probe on permuted labels.  A raw R2 or MCC includes whatever the block's SHAPE")
    print("     yields from noise, which is large for wide blocks.")
    if has_floor:
        print("  2. FLOOR delta.  The trailing signed number is this row minus its untrained twin.")
        print("     THAT is the learned part.  At patch pooling an untrained encoder alone scores")
        print("     >0.8 on most factors and ~0.86 block-MCC, so the absolute is nearly all floor.")
    else:
        print("  2. NO FLOOR MEASURED.  Without --floor you cannot tell a learned number from what")
        print("     a random projection of this architecture scores. At patch pooling that floor is")
        print("     ~0.86 block-MCC. Re-run with --floor before reporting anything below.")
    print("  3. Compare LIKE with LIKE.  Never difference two numbers from different poolings,")
    print("     probe widths, --causal modes or DCI code counts.  Cross-axis differencing is what")
    print("     produced the phantom scaling-rate gap and the 0.695-vs-0.001 ventricle reading.")
    print("  4. MCC is not invariant to re-gauging.  An exactly INVERTIBLE map costs up to 0.38 of")
    print("     block-MCC, and block identifiability permits exactly such maps.  A fall in MCC is")
    print("     therefore 'harder to read linearly', not necessarily 'lost'.")
    if pdim == PROBE_DIM_AUTO:
        print("  NOTE  --probe-dim auto: blocks with d > N/4 are PCA-reduced to min(64, N/4) for the")
        print("        PROBES only.  DCI is scored on the unreduced block (its scores are")
        print("        basis-dependent); effective rank likewise.")
    elif pdim:
        print(f"  NOTE  --probe-dim {pdim}: every probe block PCA-reduced to {pdim} components.")
        print("        DCI and effective rank are scored on the unreduced block.")
    if baseline_name:
        print(f"  Baseline row: {baseline_name}")


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
                            "is_headline_pooling": pool == assigned,
                            "real_r2": _round(pd.get("real")),
                            "null_r2": _round(pd.get("null")),
                            "gap_r2": _round(pd.get("gap")),
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
                            "is_headline_pooling": pool == assigned,
                            "real_r2": _round(pd.get("real")),
                            "null_r2": _round(pd.get("null")),
                            "gap_r2": _round(pd.get("gap")),
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


# Suffix --floor appends to a run's name for its untrained twin. One definition, shared
# by the row builder and the readers below.
_FLOOR_SUFFIX = "-floor"


def _dnum(x, nd=3):
    """Signed delta, so a floor-subtracted column reads at a glance."""
    return f"{x:+.{nd}f}" if x is not None and np.isfinite(x) else "   -  "


def _floor_map(rows):
    """``{model name: its --floor twin row}``, empty when --floor was not passed."""
    by_name = {r.get("name"): r for r in rows}
    return {
        n: by_name[n + _FLOOR_SUFFIX] for n in by_name if not n.endswith(_FLOOR_SUFFIX) and n + _FLOOR_SUFFIX in by_name
    }


def _floor_gap(floor_row, block_key, fname, detail_key="detail"):
    """That factor's GAP in the floor twin's ``block_key``, or NaN if absent.

    ``detail_key`` must track the row being printed: under --per-encoder, encoder 2's
    numbers have to be differenced against encoder 2's floor, not encoder 1's.
    """
    if floor_row is None:
        return float("nan")
    block = (floor_row.get(detail_key) or {}).get(block_key)
    if not block:
        return float("nan")
    return _fl((block.get("per_factor", {}).get(fname) or {}).get("gap"))


# A factor beaten this far at another pooling is very likely mis-assigned rather than
# unrecovered. Deliberately blunt: the point is to prompt a look at FACTOR_POOLING, NOT
# to rescore on the max — scoring on the best-of-N pooling is what the fixed assignment
# exists to prevent.
_ROUTING_MARGIN = 0.15


def _routing_note(fd):
    """Flag a factor whose assigned pooling is far from the best one available.

    Caught sulcal_widening, which FACTOR_POOLING files under "present in the channel
    mean" while the generator renders it as sin(12x)sin(12y)sin(12z) — zero-mean by
    construction, so it cannot be in a channel average. Measured on an untrained
    encoder: gap -0.08, stats -0.09, patch +0.87. Read at gap it looks like a factor no
    model recovers; it is a factor no model was ever asked about.
    """
    by_pool = (fd or {}).get("by_pooling") or {}
    assigned, g = fd.get("pooling"), _fl(fd.get("gap"))
    best_k, best_g = None, float("-inf")
    for k, v in by_pool.items():
        gk = _fl(v.get("gap"))
        if np.isfinite(gk) and gk > best_g:
            best_k, best_g = k, gk
    if best_k is None or best_k == assigned or not np.isfinite(g):
        return ""
    return f"   <- {best_g:.2f} at {best_k}; check FACTOR_POOLING" if best_g - g > _ROUTING_MARGIN else ""


def print_per_latent(rows):
    """Per-model, per-factor breakdown: for every ground-truth factor, the GAP
    test-R² of predicting it from its OWN block (signal, want high) next to the
    OTHER block (leak, want low), at the factor's assigned pooling.

    Putting signal and leak side by side is the readable disentanglement story:
    a content factor should light up the signal bar and leave the leak bar empty,
    and vice-versa for style.  The full per-pooling sweep lives in the CSV.
    """
    floors = _floor_map(rows)
    print()
    print("  PER-LATENT BREAKDOWN   (GAP = real - null at each factor's assigned pooling)")
    print("  SIGNAL = predicted from its own block (want high)   LEAK = from the other block (want low)")
    if floors:
        print("  Signed number after each cell = minus the --floor twin. That is the learned part.")

    _NUMW, _CELLW = 5, 18  # "0.650" and "0.650 [##########]"

    def _cell(g):
        return f"{_num(g):>{_NUMW}s} {_minibar(g)}"

    for r in rows:
        fl = floors.get(r["name"])
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

            def _emit(kind, signal_pf, leak_pf, sig_src, leak_src, sig_key, leak_key):
                dh = "  Δflr " if fl else ""
                print(
                    f"    {kind + ' factor':<{fw}s}  {'pool':<6s}  "
                    f"{'SIGNAL: ' + sig_src:<{_CELLW}s}{dh}    {'LEAK: ' + leak_src:<{_CELLW}s}{dh}"
                )
                print("    " + "-" * (sep_w + (14 if fl else 0)))
                for fname, fd in signal_pf.items():
                    pool = fd.get("pooling") or "?"
                    sgap = fd.get("gap")
                    lgap = (leak_pf.get(fname, {}) or {}).get("gap") if leak_pf else None
                    leak_cell = _cell(lgap) if leak_pf is not None else f"{'n/a':>{_NUMW}s}"
                    ds = f"  {_dnum(_fl(sgap) - _floor_gap(fl, sig_key, fname, detail_key))}" if fl else ""
                    dl = ""
                    if fl:
                        dl = (
                            f"  {_dnum(_fl(lgap) - _floor_gap(fl, leak_key, fname, detail_key))}"
                            if leak_pf is not None
                            else ""
                        )
                    print(f"    {fname:<{fw}s}  {pool:<6s}  {_cell(sgap)}{ds}    {leak_cell}{dl}")

            print()
            print(f"  {r['name']}{enc_label}  ({r.get('checkpoint', '?')})")
            _emit(
                "content",
                cc["per_factor"],
                sc["per_factor"] if sc else None,
                "from content",
                "from style",
                "content2content",
                "style2content",
            )
            if ss:
                print()
                _emit(
                    "style",
                    ss["per_factor"],
                    cs["per_factor"] if cs else None,
                    "from style",
                    "from content",
                    "style2style",
                    "content2style",
                )
    print()


# Head-to-head vs baseline.  direction drives the verdict mark: "up" = higher beats
# the baseline, "down" = lower beats it, "half" = closer to 0.5 beats it, "eq" =
# should roughly match the baseline (a capacity sanity check, not a payoff).
_VS_CAPACITY = [
    ("info_all", "info_all", "eq"),
    ("dci_d_gap", "DCI all", "eq"),
    ("dci_patch_d_gap", "DCI patch", "eq"),
]
# Only the channel-count-fair, theory-aligned metrics belong here.  Raw mcc_cc /
# info_c2c are capacity-bound (an all-content baseline reads content from ALL
# channels, so they inflate for it) — separation = mcc_cc - mcc_cs is the fair
# version and is what carries the block-MCC story.
_VS_ORG = [
    ("separation", "separation", "up"),
    ("leak_c2s", "leak c>s", "down"),
    ("content_view", "c>view (~.5)", "half"),
]

# Per-encoder metrics with an enc2 (_v2) twin worth consolidating in the per-encoder
# summary.  (content_view/style_view are model-level — no _v2 — so they are omitted.)
_ENC_METRICS = [
    ("separation", "separation"),
    ("leak_c2s", "leak c>s"),
    ("suff_s2s", "suff s>s"),
    ("leak_s2c", "leak s>c"),
    ("info_all", "info_all"),
    ("content_rank", "content rank"),
    ("style_rank", "style rank"),
    # DCI gaps per encoder (--with-dci only; the caller skips rows where both encoders
    # are NaN, so these cost nothing when DCI is off).  Gaps rather than raw D/C because
    # only the gap is comparable — the real - null decomposition is in the capacity
    # table and the CSV.
    ("dci_d_gap", "D gap all"),
    ("dci_c_gap", "C gap all"),
    ("dci_content_d_gap", "D gap cont"),
    ("dci_content_c_gap", "C gap cont"),
    ("dci_patch_d_gap", "D gap patch"),
    ("dci_patch_c_gap", "C gap patch"),
]


def _fl(x):
    """Coerce to float; NaN on None / non-numeric."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _enc_mean(r, key):
    """Mean of a metric across the available encoders (enc1 + enc2 ``_v2``).

    For a single-encoder run (no ``_v2``) this is just enc1, so callers can use it
    unconditionally; NaN encoders are dropped.
    """
    vals = [v for v in (_fl(r.get(key)), _fl(r.get(key + "_v2"))) if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _vs_verdict(mv, bv, direction, tol=0.05):
    """Verdict mark for one model-vs-baseline cell (ASCII, matches the scorecard)."""
    if not (np.isfinite(mv) and np.isfinite(bv)):
        return ""
    d = mv - bv
    if direction == "eq":
        return "[~] match" if abs(d) <= tol else "[!] diverges"
    if direction == "half":
        return "[+] better" if abs(mv - 0.5) < abs(bv - 0.5) else "[!] worse"
    if abs(d) <= 1e-9:
        return "[ ] same"
    better = (d > 0) if direction == "up" else (d < 0)
    return "[+] better" if better else "[!] worse"


def _vs_line(label, mv, bv, direction):
    d = mv - bv if (np.isfinite(mv) and np.isfinite(bv)) else float("nan")
    dtxt = f"{d:+.3f}" if np.isfinite(d) else "  -  "
    return (
        f"        {label:<14s} {_num(mv):>7s}   base {_num(bv):>7s}   "
        f"Δ {dtxt:>7s}   {_vs_verdict(mv, bv, direction)}"
    )


def print_comparison(rows, baseline_name=None):
    """Headline model-vs-baseline view — the first thing to read.

    For each model it puts the apples-to-apples CAPACITY metrics (which should
    roughly match the baseline — the objective organizes information, it does not
    add it) next to the ORGANIZATION metrics (which should beat the baseline — that
    is the disentanglement payoff), with Δ and a verdict mark.  Degrades gracefully:
    a metric the baseline was not scored on shows ``-`` instead of silently dropping
    the whole comparison.
    """
    w = 76
    print()
    print("=" * w)
    base = next((r for r in rows if r["name"] == baseline_name), None)
    models = [r for r in rows if r["name"] != baseline_name]
    if base is None or not models:
        print("  HEAD-TO-HEAD vs BASELINE")
        print("=" * w)
        if base is None:
            print("  No --baseline set — nothing to compare against.  Re-run with")
            print("  --baseline <run-dir> to anchor the comparison (the baseline is scored")
            print("  as one all-content block by default, so its metrics line up here).")
        else:
            print(f"  Only the baseline '{baseline_name}' was evaluated — add model run-dirs to compare.")
        print("=" * w)
        print()
        return

    mode = (
        "all content"
        if base.get("baseline_all_content")
        else ("own split" if np.isfinite(_enc_mean(base, "separation")) else "capacity only")
    )
    enc_note = "   · values = mean over both encoders" if any("separation_v2" in r for r in rows) else ""
    print(f"  HEAD-TO-HEAD vs BASELINE '{baseline_name}'   (baseline scored: {mode}){enc_note}")
    print("=" * w)
    print("  CAPACITY      should ~ match the baseline (objective organizes, doesn't add, info)")
    print("  ORGANIZATION  should beat it: separation up, leak down, c>view -> 0.5")

    ranked = sorted(
        models,
        key=lambda r: -(_enc_mean(r, "separation") if np.isfinite(_enc_mean(r, "separation")) else -9),
    )
    for r in ranked:
        print()
        print(f"  {r['name']}")
        cap = [
            (k, lab, dirn)
            for (k, lab, dirn) in _VS_CAPACITY
            if np.isfinite(_enc_mean(r, k)) or np.isfinite(_enc_mean(base, k))
        ]
        if cap:
            print("      capacity")
            for k, lab, dirn in cap:
                print(_vs_line(lab, _enc_mean(r, k), _enc_mean(base, k), dirn))
        print("      organization")
        for k, lab, dirn in _VS_ORG:
            print(_vs_line(lab, _enc_mean(r, k), _enc_mean(base, k), dirn))
    print("=" * w)
    print("  [+] beats baseline   [~] matches (capacity)   [!] worse / diverges\n")


def print_encoder_summary(rows):
    """Consolidate the two encoders (--per-encoder) into one compact per-model block.

    With ``separate_encoders`` every metric is scored twice (enc1 + enc2/``_v2``) and
    otherwise ends up split across the tables.  This collapses each model to enc1,
    enc2, their mean, and the agreement ``|enc1 - enc2|`` (small = the two encoders
    behave the same) in one place.  Skipped entirely for single-encoder runs.
    """
    per_enc = [r for r in rows if "separation_v2" in r]
    if not per_enc:
        return
    w = 76
    print()
    print("=" * w)
    print("  PER-ENCODER SUMMARY   (separate_encoders: enc1 vs enc2)")
    print("=" * w)
    print("  agree = |enc1 - enc2|;  small = the two encoders behave consistently.")
    ranked = sorted(
        per_enc,
        key=lambda r: -(_enc_mean(r, "separation") if np.isfinite(_enc_mean(r, "separation")) else -9),
    )
    for r in ranked:
        print()
        print(f"  {r['name']}")
        print(f"      {'metric':<12s} {'enc1':>7s} {'enc2':>7s} {'mean':>7s} {'agree':>7s}")
        for k, label in _ENC_METRICS:
            v1, v2 = _fl(r.get(k)), _fl(r.get(k + "_v2"))
            if not (np.isfinite(v1) or np.isfinite(v2)):
                continue
            agree = abs(v1 - v2) if (np.isfinite(v1) and np.isfinite(v2)) else float("nan")
            print(f"      {label:<12s} {_num(v1):>7s} {_num(v2):>7s} {_num(_enc_mean(r, k)):>7s} {_num(agree):>7s}")
    print("=" * w)
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
    pdim = next((r.get("probe_dim") for r in have if r.get("probe_dim")), 0)
    floors = _floor_map(rows)
    print()
    print("  ALL-CHANNELS CAPACITY   (GAP = real - null; full representation -> each factor)")
    print("  higher = more factor information present somewhere in the representation")
    if floors:
        print(
            "  TRAILING SIGNED NUMBER = this row minus its --floor twin. Read THAT, not the "
            "absolute:\n  at patch pooling an untrained encoder alone scores >0.8 on most factors."
        )
    if pdim == PROBE_DIM_AUTO:
        print("  NOTE: --probe-dim auto — p>>n blocks (d > N/4) reduced to min(64, N/4) PCs; others full width")
    elif pdim:
        print(f"  NOTE: scored on top-{pdim} PCs per block (--probe-dim) — capacity equalized across widths")
    for r in ranked:
        if r["name"] == baseline_name:
            if r.get("baseline_all_content"):
                _bmode = "all content"
            elif np.isfinite(r.get("separation", float("nan"))):
                _bmode = "own split"
            else:
                _bmode = "all-channels only"
            tag = f"   <- baseline ({_bmode})"
        else:
            tag = ""
        chans = r.get("n_content_channels", 0) + r.get("n_style_channels", 0)
        # When --per-encoder produced a second encoder, show its capacity too — the
        # baseline (all-channels) is scored on both encoders, not just the first.
        has_v2 = np.isfinite(r.get("info_all_v2", float("nan")))

        fl = floors.get(r["name"])

        def _emit_enc(suffix, enc_label):
            allb = (r.get("detail_v2" if suffix else "detail") or {}).get("all")
            mean = r.get("info_all" + suffix, float("nan"))
            head = f"    {enc_label}  " if enc_label else "    "
            dmean = f"  {_dnum(_fl(mean) - _fl(fl.get('info_all' + suffix)))}" if fl else ""
            print(f"{head}mean {_num(mean)} {_minibar(mean)}{dmean}")
            # D/C as "real - null = gap" per scope.  The gap alone is not readable: raw
            # D/C are shape artifacts (D is normalized by the factor count, C by the code
            # count), so a model can look disentangled purely from its dimensions, and a
            # null that saturates near 1 or collapses to 0 makes the gap meaningless in a
            # way that is only visible next to the null.
            if np.isfinite(r.get("dci_d_gap" + suffix, float("nan"))):
                for prefix, label, note in (
                    ("dci", "DCI all", "all chan x all factors"),
                    ("dci_content", "DCI content", "content chan x content factors"),
                    ("dci_patch", "DCI patch", "patch-grid cells x all factors"),
                ):
                    if not np.isfinite(r.get(f"{prefix}_d_gap{suffix}", float("nan"))):
                        continue
                    cells = []
                    for m in ("d", "c"):
                        real = r.get(f"{prefix}_{m}{suffix}", float("nan"))
                        null = r.get(f"{prefix}_{m}_null{suffix}", float("nan"))
                        gap = r.get(f"{prefix}_{m}_gap{suffix}", float("nan"))
                        d = f" {_dnum(_fl(gap) - _fl(fl.get(f'{prefix}_{m}_gap{suffix}')))}" if fl else ""
                        cells.append(f"{m.upper()} {_num(real)} -{_num(null)} ={_num(gap)}{d}")
                    ncodes = _fl(r.get(f"{prefix}_n_codes{suffix}"))
                    cw = f"{int(ncodes):>6d}" if np.isfinite(ncodes) else "     ?"
                    print(f"      {label:14s} {cells[0]}   {cells[1]}   {cw} codes  ({note})")
                    nulls = [_fl(r.get(f"{prefix}_{m}_null{suffix}")) for m in ("d", "c")]
                    if any(np.isfinite(v) and v > 0.9 for v in nulls):
                        print(
                            f"      {'':14s} ^ null saturated (>0.9) — the gap cannot go positive here; "
                            "raise --n-null or reconsider the pooling"
                        )
                # C is normalised by log(n_codes) and D by log(n_factors), so two scopes
                # with different code counts are on different scales even after the null
                # subtraction: 512 near-duplicate patch cells dilute one channel's
                # importance by construction.  Compare a scope against the SAME scope in
                # another model, never one scope against another.
                _ncs = {
                    _fl(r.get(f"{p}_n_codes{suffix}"))
                    for p in ("dci", "dci_content", "dci_patch")
                    if np.isfinite(_fl(r.get(f"{p}_n_codes{suffix}")))
                }
                if len(_ncs) > 1:
                    print(
                        f"      {'':14s} rows above differ in code count — read each DOWN the model "
                        "column, not ACROSS scopes"
                    )
            if allb:
                cset = set(allb.get("content_names", []))
                fw = max([len(n) for n in allb["per_factor"]] + [16])
                fl_all = ((fl or {}).get("detail_v2" if suffix else "detail") or {}).get("all") if fl else None
                for fname, fd in allb["per_factor"].items():
                    kind = "content" if fname in cset else "style"
                    g = fd.get("gap")
                    pool = fd.get("pooling") or "?"
                    d = ""
                    if fl_all:
                        fg = _fl((fl_all.get("per_factor", {}).get(fname) or {}).get("gap"))
                        d = f"  {_dnum(_fl(g) - fg)}"
                    print(f"      {kind:7s} {fname:<{fw}s} {pool:<6s} {_num(g)} {_minibar(g)}{d}{_routing_note(fd)}")

        print()
        print(f"  {r['name']}{tag}   {chans}ch")
        _cr, _sr = _fl(r.get("content_rank")), _fl(r.get("style_rank"))
        if np.isfinite(_cr) or np.isfinite(_sr):
            _nc, _ns = r.get("n_content_channels", 0), r.get("n_style_channels", 0)
            _crs = f"{_cr:.1f}" if np.isfinite(_cr) else "-"
            _srs = f"{_sr:.1f}" if np.isfinite(_sr) else "-"
            print(f"      eff. rank  content {_crs}/{_nc}   style {_srs}/{_ns}   (low vs nominal => collapse)")
        if has_v2:
            _emit_enc("", "encoder 1")
            _emit_enc("_v2", "encoder 2")
        else:
            _emit_enc("", "")
    print()


_CSV_RENAME = {
    "name": "model",
    "n_content_channels": "content_channels",
    "n_style_channels": "style_channels",
    "disentanglement": "overall_score",
    "info_all": "informativeness",
    "content_rank": "content_eff_rank",
    "style_rank": "style_eff_rank",
    "all_rank": "total_eff_rank",
    "dci_d_gap": "dci_disent_gap",
    "dci_content_d_gap": "dci_disent_gap_content",
    "dci_patch_d_gap": "dci_disent_gap_patch",
    "leak_c2s": "content_to_style_leak",
    "content_view": "content_view_acc",
    "style_view": "style_view_acc",
    "suff_s2s": "style_sufficiency",
    "leak_s2c": "style_to_content_leak",
    "mcc_cc": "mcc_content_to_content",
    "mcc_cs": "mcc_content_to_style",
    "mcc_cc_null": "mcc_content_to_content_null",
    "mcc_cs_null": "mcc_content_to_style_null",
    "dci_c_gap": "dci_complete_gap",
    "dci_content_c_gap": "dci_complete_gap_content",
    "dci_patch_c_gap": "dci_complete_gap_patch",
    # Raw D/C and their null floors, next to the gaps.  A gap on its own is not
    # interpretable: raw D/C are shape artifacts (D is normalized by the factor count,
    # C by the code count) and a null that saturates near 1 or collapses to 0 makes the
    # gap meaningless — both are only visible with real and null in the table.
    "dci_d": "dci_disent",
    "dci_d_null": "dci_disent_null",
    "dci_content_d": "dci_disent_content",
    "dci_content_d_null": "dci_disent_null_content",
    "dci_patch_d": "dci_disent_patch",
    "dci_patch_d_null": "dci_disent_null_patch",
    "dci_c": "dci_complete",
    "dci_c_null": "dci_complete_null",
    "dci_content_c": "dci_complete_content",
    "dci_content_c_null": "dci_complete_null_content",
    "dci_patch_c": "dci_complete_patch",
    "dci_patch_c_null": "dci_complete_null_patch",
    # Width the scope was actually scored at. Not cosmetic: C is normalised by
    # log(n_codes), so two rows with different counts are on different scales.
    "dci_n_codes": "dci_n_codes",
    "dci_content_n_codes": "dci_n_codes_content",
    "dci_patch_n_codes": "dci_n_codes_patch",
}
_CSV_TO_INTERNAL = {v: k for k, v in _CSV_RENAME.items()}
for _old, _new in list(_CSV_RENAME.items()):
    _CSV_TO_INTERNAL[_new + "_v2"] = _old + "_v2"

_CSV_LEGEND = """\
# DCI MODEL COMPARISON — one row per model
# Use comment='#' when loading with pandas:  pd.read_csv("dci_compare.csv", comment='#')
#
# IDENTITY: model, role, checkpoint, content_channels, style_channels
#
# HEALTH SCORES (derived 0–1 composites, higher = better):
#   grade              letter grade (A+ to F) from overall_score
#   overall_score      mean of the four sub-scores below
#   content_anatomy    content captures shared anatomy factors
#   content_purity     content rejects modality / view information
#   style_modality     style captures modality / view information
#   style_purity       style rejects anatomy information
#
# CAPACITY:
#   informativeness          test R² predicting all factors from all channels  (higher = better)
#   content/style_eff_rank   effective rank of representation (low vs nominal → collapse)
#   dci_disent_gap*          DCI Disentanglement score (real − null)
#
# ORGANIZATION (raw protocol metrics):
#   separation               mcc(content→content) − mcc(content→style)    higher = better
#   content_to_style_leak    content→style R² gap (null-subtracted)       lower = better, ~0 = clean
#   content_view_acc         probe: predict view from content             ~0.5 = view-invariant
#   style_view_acc           probe: predict view from style               higher = better
#   style_sufficiency        style→style R² gap                           higher = better
#   style_to_content_leak    style→content R² gap                         lower = better
#
# DELTA: delta_* columns = this model minus the baseline
#
# RAW DETAIL:
#   mcc_content_to_*         block MCC scores and their null-permuted floors
#   view_chance              chance-level view classification accuracy
#   dci_complete_gap*        DCI Completeness score (real − null)
#
# DCI RAW (read these WITH the gaps — a gap alone is not interpretable):
#   dci_disent, dci_complete            real D / C
#   dci_disent_null, dci_complete_null  label-permutation floor for the same shape
#   *_content, *_patch                  content-block and patch-pooled scopes
#   dci_n_codes*                        width each scope was scored at
#   Raw D/C are shape artifacts: D is normalized by the factor count and C by the code
#   count, so both sit high when factors are few or codes many. Compare gaps, and treat
#   a scope whose null exceeds ~0.9 (or collapses to ~0) as unreportable.
#   DCI ignores --probe-dim by design and is scored on the encoder's own basis, capped by
#   --dci-max-codes (highest-variance SELECTION, never a rotation): D asks "is this CODE
#   dedicated to few factors", so a PCA-projected block would describe the PCA basis.
#   Compare a scope only against the SAME scope in another row, and only when
#   dci_n_codes matches — C is normalized by log(n_codes), so different widths are
#   different scales. At patch pooling one channel becomes many near-duplicate cells,
#   which dilutes C and inflates D by construction; that is the pooling, not the model.
#   With --per-encoder every DCI column also appears with a _v2 suffix (encoder 2).
#
# META: num_samples, poolings, probe_dim, run_dir
#   generator   "current" or "legacy_pre_7ac56a3" (--old-generator). Rows with different
#               values are NOT comparable on lesion_*, temporal_atrophy, sulcal_widening.
#
"""


def write_outputs(rows, out_dir, baseline_name=None):
    os.makedirs(out_dir, exist_ok=True)
    attach_scores(rows)  # idempotent — ensure derived columns exist even if called directly

    base_row = next((r for r in rows if baseline_name and r.get("name") == baseline_name), None)
    has_v2 = any("separation_v2" in r for r in rows)

    id_cols = ["model", "role", "checkpoint", "content_channels", "style_channels"]
    health_cols = ["grade", "overall_score", "content_anatomy", "content_purity", "style_modality", "style_purity"]
    capacity_cols = [
        "informativeness",
        "content_eff_rank",
        "style_eff_rank",
        "total_eff_rank",
        "dci_disent_gap",
        "dci_disent_gap_content",
        "dci_disent_gap_patch",
    ]
    org_cols = [
        "separation",
        "content_to_style_leak",
        "content_view_acc",
        "style_view_acc",
        "style_sufficiency",
        "style_to_content_leak",
    ]
    delta_cols = [
        "delta_separation",
        "delta_content_to_style_leak",
        "delta_content_view_acc",
        "delta_informativeness",
    ]
    detail_cols = [
        "mcc_content_to_content",
        "mcc_content_to_style",
        "mcc_content_to_content_null",
        "mcc_content_to_style_null",
        "view_chance",
        "dci_complete_gap",
        "dci_complete_gap_content",
        "dci_complete_gap_patch",
    ]
    meta_cols = ["num_samples", "poolings", "probe_dim", "factor_pooling", "generator", "run_dir"]
    v2_cols = (
        [
            "grade_v2",
            "overall_score_v2",
            "informativeness_v2",
            "content_eff_rank_v2",
            "style_eff_rank_v2",
            "separation_v2",
            "content_to_style_leak_v2",
            "style_sufficiency_v2",
            "style_to_content_leak_v2",
        ]
        if has_v2
        else []
    )

    def _dci_cols(suffix="", include_gap=True):
        """DCI column names for one encoder: real, null and (optionally) gap for both
        metrics across the three scopes.  Generated rather than listed so the two
        encoders cannot drift apart."""
        cols = []
        for metric in ("disent", "complete"):
            for scope in ("", "_content", "_patch"):
                cols.append(f"dci_{metric}{scope}{suffix}")
                cols.append(f"dci_{metric}_null{scope}{suffix}")
                if include_gap:
                    cols.append(f"dci_{metric}_gap{scope}{suffix}")
        # One code count per scope, not per metric — it is a property of the block.
        for scope in ("", "_content", "_patch"):
            cols.append(f"dci_n_codes{scope}{suffix}")
        return cols

    # Encoder 1's gap columns already live in capacity_cols / detail_cols, so only its
    # real+null are added here.  Encoder 2 had NO dci columns at all, so it gets the set.
    dci_cols = _dci_cols("", include_gap=False)
    dci_v2_cols = _dci_cols("_v2", include_gap=True) if has_v2 else []

    flat_cols = (
        id_cols
        + health_cols
        + capacity_cols
        + org_cols
        + delta_cols
        + detail_cols
        + dci_cols
        + meta_cols
        + v2_cols
        + dci_v2_cols
    )
    assert len(set(flat_cols)) == len(flat_cols), "duplicate CSV columns"

    def _cell(v):
        if isinstance(v, float):
            return round(v, 4) if np.isfinite(v) else ""
        return v

    def _delta(r, k):
        if base_row is None or r is base_row:
            return ""
        mv, bv = r.get(k), base_row.get(k)
        if not (isinstance(mv, (int, float)) and isinstance(bv, (int, float)) and np.isfinite(mv) and np.isfinite(bv)):
            return ""
        return round(mv - bv, 4)

    def _row_out(r):
        is_base = bool(baseline_name) and r.get("name") == baseline_name
        deltas = {
            "delta_separation": _delta(r, "separation"),
            "delta_content_to_style_leak": _delta(r, "leak_c2s"),
            "delta_content_view_acc": _delta(r, "content_view"),
            "delta_informativeness": _delta(r, "info_all"),
        }
        out = {}
        for c in flat_cols:
            if c == "role":
                out[c] = "baseline" if is_base else "model"
            elif c in deltas:
                out[c] = deltas[c]
            else:
                internal = _CSV_TO_INTERNAL.get(c, c)
                out[c] = _cell(r.get(internal))
        return out

    def _sort_key(r):
        s = r.get("separation")
        s = s if isinstance(s, (int, float)) and np.isfinite(s) else -9.0
        is_base = bool(baseline_name) and r.get("name") == baseline_name
        return (0 if is_base else 1, -s)

    csv_path = os.path.join(out_dir, "dci_compare.csv")
    with open(csv_path, "w", newline="") as f:
        f.write(_CSV_LEGEND)
        w = csv.DictWriter(f, fieldnames=flat_cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=_sort_key):
            w.writerow(_row_out(r))

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
        "is_headline_pooling",
        "real_r2",
        "null_r2",
        "gap_r2",
        "real_std",
    ]
    pl_rows = sorted(
        iter_per_latent_rows(rows),
        key=lambda d: (
            d["model"],
            d["encoder"],
            d["factor_type"],
            d["factor"],
            d["relationship"],
            not d["is_headline_pooling"],
        ),
    )
    _PL_LEGEND = """\
# DCI PER-LATENT BREAKDOWN — one row per (model, encoder, factor, pooling)
# Use comment='#' when loading with pandas:  pd.read_csv("dci_compare_per_latent.csv", comment='#')
#
# IDENTITY:
#   model                model name
#   encoder              enc1 or enc2 (when two encoders are evaluated)
#   factor               ground-truth factor name (e.g. brain_size, gain)
#   factor_type          content or style — which group the factor belongs to
#
# PROBE SETUP:
#   predicted_from       which representation block is used as input (content / style / all)
#   relationship         signal = factor predicted from its own block (want high)
#                        leakage = factor predicted from the wrong block (want low)
#                        capacity = factor predicted from all channels
#   want                 desired direction for gap_r2: "high" for signal/capacity, "low" for leakage
#   pooling              spatial pooling method used (gap, stats, patch)
#   is_headline_pooling  True = this pooling was selected as the headline (best-of) for this factor
#
# RESULTS:
#   real_r2              test R² from the GBT probe
#   null_r2              test R² under row-permuted factors (structural floor for this block shape)
#   gap_r2               real_r2 − null_r2 (the non-structural signal)
#   real_std              std of real_r2 across repeats
#
"""
    with open(per_latent_path, "w", newline="") as f:
        f.write(_PL_LEGEND)
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
        row.get("probe_dim"),
        row.get("factor_pooling"),
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
        "(recommended for a final comparison; same choice for all models keeps it fair). Also the "
        "checkpoint used for a --baseline directory that is not itself listed in --run-dirs.",
    )
    p.add_argument(
        "--checkpoint-names",
        nargs="*",
        default=None,
        help="Per-run checkpoint filenames, one per --run-dirs (or exactly one, reused for all). "
        "Overrides --checkpoint-name. Repeat a run directory to compare two of its own checkpoints: "
        "--run-dirs RUN RUN --checkpoint-names vqvae_model.pt vqvae_model_late.pt. Default labels are "
        "disambiguated with the checkpoint stem when a directory repeats.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Frozen test-set size, shared across models.",
    )
    p.add_argument(
        "--causal",
        choices=("match", "iid"),
        default="match",
        help="Factor distribution of the frozen test set, for runs trained with --synthetic-causal. "
        "'match' (default) reproduces the training SCM; 'iid' forces independent factors. Not "
        "cosmetic: under a random graph the dependent factors are largely determined by their parents, "
        "so scoring them i.i.d. reads as lost information — and that gap WIDENS the better the model "
        "fits the training distribution, which looks exactly like degradation over training. The "
        "default was 'iid-with-a-warning' until 20 Aug 2026; it produced a phantom 0.2-mean-R² gap "
        "between two runs that differed only in scaling rate, so numbers printed before that date are "
        "not comparable to these unless they passed --causal explicitly. Pass 'iid' when you WANT the "
        "factors decorrelated so per-factor attribution is unambiguous.",
    )
    p.add_argument(
        "--poolings",
        default="gap,stats,2x2x2",
        help="Comma list: gap, stats, and/or DxHxW (e.g. 2x2x2).",
    )
    p.add_argument("--level", type=int, default=0, help="Encoder level to compare on.")
    p.add_argument("--seeds", default="0,1,2", help="Probe CV seeds.")
    p.add_argument("--n-null", type=int, default=3, help="Permutations for the null floor.")
    p.add_argument(
        "--floor",
        action="store_true",
        help="Also score an UNTRAINED twin of every run (row '<name>-floor'), on the same "
        "pooling, probe width, null count and factor distribution. The label-permutation null "
        "already in every row asks 'is there any signal at all'; this asks the different and "
        "usually harder question 'is there more signal than a random projection of this same "
        "architecture'. At patch pooling an untrained encoder reads six of nine content factors "
        "above R^2 0.8, so an absolute per-factor number there is close to meaningless without "
        "it. Roughly doubles runtime.",
    )
    p.add_argument(
        "--floor-seeds",
        type=int,
        default=3,
        help="How many untrained draws to average the floor over (>=1). The untrained weights "
        "ARE the measurement, and they were unseeded: one draw of a random projection, different "
        "every invocation, whose noise went silently into every reported delta. Measured here, "
        "three near-equivalent lesion axes drew floors spanning 0.16 of block-MCC — the same order "
        "as the effects claimed over them. Seeds are 0..N-1, so a floor is now reproducible, and "
        "the across-seed std is reported so a delta can be read against it. Costs one extra "
        "extraction per seed; use 1 for a fast, still-reproducible floor.",
    )
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
        help="Score the --baseline run on its own (arbitrary) content/style split — every "
        "metric filled, content AND style side.  Default (without this flag): the baseline is "
        "scored as one all-content block (style merged into content), so the content-side "
        "metrics + capacity are comparable to the models and the style-side is N/A.",
    )
    p.add_argument(
        "--with-dci",
        action="store_true",
        help="Also compute split-free GAP DCI disentanglement/completeness on the "
        "all-channels representation for every model (incl. a no-split vanilla "
        "baseline).  Uses GBT importance — noticeably slower; off by default.",
    )
    p.add_argument(
        "--factor-pooling",
        default="assigned",
        choices=("assigned", "gap", "stats", "patch"),
        help="Which pooling each factor is scored at. 'assigned' (default) uses FACTOR_POOLING: "
        "each factor read at the coarsest pooling that can physically expose it, fixed in advance "
        "so the headline is never a max over poolings. That routing also keeps factors off their "
        "ceiling — at patch an untrained encoder reads the morphometry factors at R2 0.78-0.92, "
        "leaving under 0.07 of headroom to tell two models apart. Naming a pooling scores EVERY "
        "factor there; use it for a deliberate one-rung view rather than editing FACTOR_POOLING, "
        "which changes the routing for every script that imports it and leaves no record in the CSV.",
    )
    p.add_argument(
        "--dci-max-codes",
        type=int,
        default=4096,
        help="Cap the DCI blocks at this many codes, keeping the highest-variance ones "
        "(0 = no cap).  DCI is NOT subject to --probe-dim: D asks 'is this CODE dedicated "
        "to few factors' and C normalises by the code count, so both are statements about "
        "one specific basis, and a PCA projection replaces every code with a mixture of "
        "channels — the score then describes the PCA basis, not the encoder.  Selection "
        "keeps each retained code a real (channel, patch) feature while bounding the GBT "
        "cost, which at 8x8x8 would otherwise be ~22528 features per fit.",
    )
    p.add_argument(
        "--probe-dim",
        default=PROBE_DIM_AUTO,
        help="PCA width for every probe block. 'auto' (default) reduces ONLY blocks in the "
        "p>>n regime (d > N/4) to min(64, N/4): at --poolings 8x8x8 the patch block is ~22528 "
        "features, where a ridge probe returns R^2 -0.27 on a SHUFFLED target and drags the weak "
        "factors negative, while gap (44) and stats (176) are well-conditioned and stay at full "
        "width. An explicit integer reduces EVERY block instead, which is what you want to "
        "equalise a channel-count confound when sweeping width — set it <= the smallest content "
        "width in the comparison. 0 disables reduction entirely and reproduces pre-2026 numbers; "
        "it is not a safe default at patch pooling. The width is a reported parameter, not "
        "preprocessing: two models must be scored at the same value.",
    )
    p.add_argument(
        "--probe-kind",
        default="ridge",
        choices=("ridge", "kernel", "mlp"),
        help="Estimator for every informativeness / block-MCC probe. 'ridge' (default) measures "
        "LINEAR accessibility only — a gain there can be re-formatting rather than added "
        "identifiable content. 'kernel' is RBF kernel ridge, the probe Yao et al. evaluate with; "
        "'mlp' was the most sensitive to purely nonlinear structure in testing. Run ridge and a "
        "nonlinear kind and compare: linear-only movement means re-formatting, both moving means "
        "the nonlinear ceiling actually rose.",
    )
    p.add_argument(
        "--squash-content",
        action="store_true",
        help="Apply tanh to the extracted content block before probing — the post-squash view of a "
        "'bounded'-mode (Thm 3.2) model, whose loss shaped tanh(content) while probes otherwise read "
        "raw content. Diagnostic: run one bounded model with and without this flag and compare the "
        "RIDGE score. A pre/post gap means the tanh is doing linear reformatting (and, if the model "
        "saturated, that pre-squash is no longer diffeomorphism-equivalent). Applies to every arm in "
        "the call, so use it on a single bounded run, not a mixed table.",
    )
    p.add_argument(
        "--old-generator",
        action="store_true",
        help="Render the frozen test set with the PRE-7ac56a3 render_structure (eval.legacy_renderer). "
        "Use when the checkpoints were trained before 2026-07-05: they never saw the current "
        "rendering, so scoring lesion_*/temporal_atrophy/sulcal_widening on it is invalid rather "
        "than merely unflattering. Applies to every model in the run, and those factors' numbers "
        "are not comparable with current-generator runs. brain_size / ventricle_size / "
        "cortical_thickness / lr_asymmetry are unaffected by the fix and stay comparable.",
    )
    p.add_argument("--out", default="dci_compare_out", help="Output directory.")
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Overwrite any existing results in --out instead of merging the new models into them.",
    )
    cli = p.parse_args()

    if cli.probe_dim != PROBE_DIM_AUTO:
        try:
            cli.probe_dim = int(cli.probe_dim)
        except (TypeError, ValueError):
            p.error(f"--probe-dim must be an integer or '{PROBE_DIM_AUTO}' (got {cli.probe_dim!r})")
        if cli.probe_dim < 0:
            p.error(f"--probe-dim must be >= 0 or '{PROBE_DIM_AUTO}' (got {cli.probe_dim})")

    poolings = parse_poolings(cli.poolings)
    seeds = tuple(int(s) for s in cli.seeds.split(","))

    # Assemble the model list; the baseline (if separate) is evaluated too.
    specs = list(cli.run_dirs)
    ckpts = list(cli.checkpoint_names) if cli.checkpoint_names else [cli.checkpoint_name]
    if len(ckpts) == 1:
        ckpts = ckpts * len(specs)
    if len(ckpts) != len(specs):
        p.error("--checkpoint-names must have one entry per --run-dirs (or exactly one)")
    if cli.names:
        names = cli.names
    else:
        # Repeating a run-dir to compare its own checkpoints would collide on the basename,
        # so fold the checkpoint stem into the default label in that case only.
        _repeats = len(set(specs)) != len(specs)
        names = [
            os.path.basename(os.path.normpath(d)) + (f":{os.path.splitext(c)[0]}" if _repeats else "")
            for d, c in zip(specs, ckpts)
        ]
    if len(names) != len(specs):
        p.error("--names must have one entry per --run-dirs")
    baseline_name = None
    if cli.baseline:
        if cli.baseline in specs:
            if specs.count(cli.baseline) > 1:
                p.error(
                    f"--baseline {cli.baseline} appears more than once in --run-dirs, so which "
                    "checkpoint anchors Δ is ambiguous. Point --baseline at a separate run "
                    "directory, or drop it."
                )
            baseline_name = names[specs.index(cli.baseline)]
        else:
            baseline_name = os.path.basename(os.path.normpath(cli.baseline))
            specs = [cli.baseline] + specs
            names = [baseline_name] + names
            ckpts = [cli.checkpoint_name] + ckpts

    # One frozen test set, built from the first run's settings, reused for all.
    import torch

    from eval.run_dci_synthetic import build_synthetic_test_set, load_run_args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_args = load_run_args(specs[0])
    dataset = build_synthetic_test_set(ref_args, cli.num_samples, causal=cli.causal == "match")
    logger.info(
        "Frozen test set: %d samples, shared across %d model(s).",
        cli.num_samples,
        len(specs),
    )
    if cli.old_generator:
        # Applied to the single frozen dataset, so EVERY model in the comparison sees the
        # legacy rendering — mixing vintages within one table would be worse than either
        # vintage alone.
        from eval.legacy_renderer import FIXED_FACTORS, use_legacy_renderer

        use_legacy_renderer(dataset)
        logger.warning(
            "--old-generator: scoring on the pre-7ac56a3 renderer. %s are rendered the OLD way, "
            "so these numbers are NOT comparable with current-generator runs for those factors.",
            ", ".join(FIXED_FACTORS),
        )

    # --floor appends an untrained twin of each run, scored by the identical path, so the
    # zero point sits in the same table on the same axis instead of being carried over from
    # a separate run_dci_synthetic invocation with its own defaults.
    # `base` is the row this one is scored alongside: a floor row must inherit its twin's
    # baseline status, or the baseline's floor would be scored on a different content/style
    # split than the baseline and stop being subtractable from it.
    #
    # Each floor is scored `--floor-seeds` times with a fixed torch seed per draw and then
    # averaged into one `<name>-floor` row (see `mean_std_structs`). Unseeded, the floor was
    # a single random projection redrawn on every invocation, so the zero point everything
    # is subtracted from carried noise nobody could reproduce or bound.
    n_floor_seeds = max(1, cli.floor_seeds)
    plan = [(n, n, d, c, False, None) for n, d, c in zip(names, specs, ckpts)]
    if cli.floor:
        plan += [
            (f"{n}{_FLOOR_SUFFIX}-s{seed}", n, d, c, True, seed)
            for seed in range(n_floor_seeds)
            for n, d, c in zip(names, specs, ckpts)
        ]

    rows = []
    for name, base, run_dir, ckpt_name, random_init, init_seed in plan:
        is_baseline = baseline_name is not None and base == baseline_name
        # Baseline default = all-content: style is merged into content (one content
        # block, no style), so its content-side metrics + capacity line up with the
        # models.  --baseline-per-block instead scores its own (arbitrary) split.
        all_content = is_baseline and not cli.baseline_per_block
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
                    checkpoint=_resolve_checkpoint(run_dir, ckpt_name),
                    n_jobs=cli.n_jobs,
                    per_encoder=cli.per_encoder,
                    with_dci=cli.with_dci,
                    probe_dim=cli.probe_dim,
                    probe_kind=cli.probe_kind,
                    dci_max_codes=cli.dci_max_codes,
                    factor_pooling=cli.factor_pooling,
                    init_seed=init_seed,
                    squash_content=cli.squash_content,
                    all_content=all_content,
                    random_init=random_init,
                )
            )
        except Exception as e:
            logger.error("Skipping %s (%s): %s", name, run_dir, e)

    if not rows:
        logger.error("No models evaluated successfully.")
        return

    # Collapse the per-seed floor draws into one `<name>-floor` row. The across-seed std is
    # kept on the row as `floor_std` (same nested shape) so a delta can be read against the
    # floor's own spread rather than treated as exact.
    if cli.floor:
        keep, by_base = [], {}
        for r in rows:
            m = re.match(rf"^(?P<base>.+){re.escape(_FLOOR_SUFFIX)}-s\d+$", r["name"])
            if m:
                by_base.setdefault(m.group("base"), []).append(r)
            else:
                keep.append(r)
        for base, draws in by_base.items():
            avg, std = mean_std_structs(draws)
            avg["name"] = base + _FLOOR_SUFFIX
            avg["floor_seeds"] = len(draws)
            avg["floor_std"] = std
            keep.append(avg)
            spread = _as_float(std.get("info_all"))
            logger.info(
                "floor for %s: averaged over %d untrained draws (info_all across-seed std %.4f)",
                base,
                len(draws),
                spread,
            )
        rows = keep

    # Stamp eval settings on each row (provenance + the merge-comparability check).
    meta = {
        "num_samples": cli.num_samples,
        "poolings": cli.poolings,
        "level": cli.level,
        "n_null": cli.n_null,
        "seeds": cli.seeds,
        "probe_dim": cli.probe_dim,
        "factor_pooling": cli.factor_pooling,
        # Generator vintage is provenance, not a setting: a legacy-renderer row and a
        # current-renderer row are different experiments on the fix-affected factors, and
        # nothing else in the output would reveal the difference.
        "generator": "legacy_pre_7ac56a3" if cli.old_generator else "current",
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
    print_reading_guide(merged, baseline_name)
    _section(1, "HEAD-TO-HEAD vs BASELINE", "the headline: does the objective organize better?")
    print_comparison(merged, baseline_name)
    _section(2, "ENCODER SUMMARY", "per-encoder split, when --per-encoder gave each view its own")
    print_encoder_summary(merged)
    _section(3, "CAPACITY + DCI", "how much factor information is present, and how it is spread")
    print_capacity_table(merged, baseline_name)
    _section(4, "SCORECARD", "derived 0-100% health scores — a summary, never the evidence")
    print_scorecard(merged, baseline_name)
    _section(5, "RAW METRICS", "the protocol numbers the scores above are derived from")
    print_table(merged, baseline_name)
    _section(6, "PER-LATENT BREAKDOWN", "per factor: signal vs leak, at its assigned pooling")
    print_per_latent(merged)
    write_outputs(merged, cli.out, baseline_name)
    print(f"\n  Wrote CSV + JSON to {cli.out}/\n")


if __name__ == "__main__":
    main()
