#!/usr/bin/env python
"""Does spatial averaging make the two views AGREE more?  If so, alignment rewards smearing.

The claim under test: a cross-view alignment objective has a gradient pointing toward
spatially smeared representations, because averaging suppresses whatever differs between
the views while content survives.  If true, an alignment term does not merely fail to
create locality — it actively destroys it, which is what the measured lesion gap looks
like (recon-only 0.775/0.804/0.522 vs +contrastive 0.695/0.727/0.286 for lesion x/y/z).

This is a statement about the OBJECTIVE and the DATA, not about a trained encoder, so the
decisive form needs no model and no training.  Pool both views to a grid, correlate them
cell-by-cell across samples, and sweep the grid.  Rising correlation as the grid coarsens
IS the gradient the objective feels.

    correlation rises as cells get bigger  ->  smearing is rewarded; the mechanism is real
    correlation flat across scales         ->  the mechanism is dead, look elsewhere

Which difference drives it (``--decompose``)
--------------------------------------------
The two views differ in four ways, and they behave very differently under averaging:

    modality   T1 vs FLAIR LUT — a deterministic function of tissue label. Spatially
               structured, and NOT obviously suppressed by averaging.
    style      gain and bias — GLOBAL scalars. Averaging cannot touch them.
    seed       the bias field (``_seeded_noise(scale=4)``, smooth) plus per-voxel noise.
               Only this component is expected to average down, and only partly, since
               the scale-4 field is spatially correlated.
    all        the real pairing, all three at once.

Re-rendering pairs that differ in exactly one component isolates which one (if any) makes
alignment scale-dependent.  A mechanism that lives entirely in ``seed`` is a statement
about render noise; one that lives in ``modality`` is a statement about the benchmark's
view design.

Built-in validity check: the ``identical`` regime (same style, same seed, same LUT) must
give correlation 1.000 at every scale.  If it does not, the rendering path here has
diverged from the dataset's and nothing else in the output means anything.

Validated on planted pairs (200 samples, 32^3, ``cellwise_corr`` on exact block-means):

    planted view difference          16^3    8^3    4^3    2^3    1^3
    identical                       1.000  1.000  1.000  1.000  1.000
    per-voxel (white) noise         0.942  0.992  0.999  0.999  0.999   <- RISES
    per-sample global gain          0.899  0.899  0.899  0.894  0.904   <- flat
    smooth scale-4 field            0.702  0.702  0.702  0.681  0.687   <- flat

**Read that calibration before interpreting the real output.**  Only a *white-noise* view
difference is suppressed by averaging.  A difference that is spatially correlated (the
scale-4 bias field) or global (gain/bias) is NOT suppressed at any grid — pooling below its
correlation length changes nothing, and pooling above it just turns it into a global scalar.
So "alignment rewards smearing" can only be true to the extent that the view difference is
white, and the ``--decompose`` panels are what say whether it is.

Usage
-----
    python -m eval.alignment_scale_test --run-dir results/synthetic/RUN --decompose
    python -m eval.alignment_scale_test --run-dir results/synthetic/RUN --with-encoder
"""

from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

GRIDS = (16, 8, 4, 2, 1)


def cellwise_corr(a, b, keep):
    """Mean across-sample correlation between paired cells, over kept cells only.

    Cells that are background in every sample have ~zero variance and would return a
    meaningless (or NaN) correlation, so ``keep`` restricts to cells with real signal.
    """
    a = a[:, keep].astype(np.float64)
    b = b[:, keep].astype(np.float64)
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    na, nb = np.sqrt((a**2).sum(0)), np.sqrt((b**2).sum(0))
    ok = (na > 1e-9) & (nb > 1e-9)
    if not ok.any():
        return float("nan"), 0
    r = (a[:, ok] * b[:, ok]).sum(0) / (na[ok] * nb[ok])
    return float(np.mean(r)), int(ok.sum())


def pool_pairs(v1, v2, mask, grid):
    """Pool both views (and the mask) to ``grid`` and flatten. Returns (N, g^3) each."""
    g = (grid, grid, grid)
    p1 = F.adaptive_avg_pool3d(v1, g).flatten(1).numpy()
    p2 = F.adaptive_avg_pool3d(v2, g).flatten(1).numpy()
    frac = F.adaptive_avg_pool3d(mask, g).flatten(1).numpy().mean(0)
    return p1, p2, frac


def sweep(v1, v2, mask, grids, fg_thresh):
    """Cross-view correlation at every grid. Returns a list of row dicts."""
    rows = []
    for g in grids:
        p1, p2, frac = pool_pairs(v1, v2, mask, g)
        keep = frac >= fg_thresh if g > 1 else np.ones(1, dtype=bool)
        if not keep.any():
            keep = np.ones_like(frac, dtype=bool)
        r, n = cellwise_corr(p1, p2, keep)
        rows.append({"grid": g, "cells": int(g**3), "kept": n, "corr": r})
    return rows


def _print_sweep(title, rows, note=""):
    print("\n" + "=" * 66)
    print(f"  {title}")
    if note:
        print(f"  {note}")
    print("=" * 66)
    print(f"  {'grid':>6}{'cells':>8}{'kept':>7}{'cross-view corr':>18}")
    print("  " + "-" * 62)
    for r in rows:
        bar = "#" * int(max(0.0, min(1.0, r["corr"])) * 20)
        print(f"  {r['grid']:>4}^3{r['cells']:>8}{r['kept']:>7}{r['corr']:>18.4f}  {bar}")
    finite = [r["corr"] for r in rows if np.isfinite(r["corr"])]
    if len(finite) >= 2:
        d = finite[-1] - finite[0]
        verdict = (
            "REWARDS SMEARING (agreement rises as cells get bigger)"
            if d > 0.02
            else ("flat — no scale pressure" if abs(d) <= 0.02 else "penalises smearing (agreement FALLS)")
        )
        print(f"\n  finest {finite[0]:.4f} -> coarsest {finite[-1]:.4f}   delta {d:+.4f}   {verdict}")


def build_pairs(ds, n, regime, device):
    """Re-render ``n`` sample pairs whose two views differ ONLY in ``regime``.

    Anatomy still varies across samples (needed for an across-sample correlation); what is
    controlled is what differs WITHIN each pair.  Mirrors the dataset's own call sequence
    in ``_pseudo_mri_item`` so ``regime='all'`` reproduces the real pairing.
    """
    inner = ds._inner if hasattr(ds, "_inner") else ds
    v1s, v2s, masks = [], [], []
    for i in range(n):
        lat = inner._pseudo_mri_item(i)[2]
        tissue, lesion = inner.renderer.render_structure(
            lat["z_content"], lat["z_deformation"], lat["z_fissure"], device=device, clean=inner.clean_content
        )
        seed = inner.seed * 1000003 + i
        s1, s2 = lat["z_style_v1"], lat["z_style_v2"]
        # Each regime turns ON exactly one source of view difference.
        mod1, mod2 = ("T1", "T1")
        sty1, sty2 = (s1, s1)
        sd1, sd2 = (seed * 2, seed * 2)
        if regime in ("modality", "all"):
            mod2 = "FLAIR"
        if regime in ("style", "all"):
            sty2 = s2
        if regime in ("seed", "all"):
            sd2 = seed * 2 + 1
        v1s.append(inner.renderer.render_modality(tissue, lesion, sty1, mod1, view_seed=sd1, device=device))
        v2s.append(inner.renderer.render_modality(tissue, lesion, sty2, mod2, view_seed=sd2, device=device))
        masks.append((tissue > 0).float())

    def _stack(xs):
        return torch.stack([x if x.dim() == 4 else x[None] for x in xs]).float().cpu()

    return _stack(v1s), _stack(v2s), _stack(masks)


def encoder_sweep(run_dir, checkpoint_name, dataset, grids, level, batch_size, num_workers, untrained):
    """Same question through an encoder's CONTENT channels: BT's diagonal per grid."""
    from eval.dci import _extract_synthetic_representations
    from eval.run_dci_compare import _CONTENT, _CONTENT_V2
    from eval.run_dci_synthetic import load_model_from_run_dir

    ckpt = os.path.join(run_dir, checkpoint_name)
    model, _args, device = load_model_from_run_dir(run_dir, ckpt if os.path.exists(ckpt) else None, None)
    if untrained:
        # Re-initialise every parameter: if the pressure is present at random init it is a
        # property of the objective and the data, not something training produced.
        for m in model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    rows = []
    for g in grids:
        pooling = "gap" if g == 1 else (g, g, g)
        ld, _gc, _a, _b = _extract_synthetic_representations(
            model, dataset, device, batch_size, num_workers, pooling=pooling
        )
        if level not in ld:
            continue
        c1, c2 = ld[level][_CONTENT], ld[level][_CONTENT_V2]
        if c1 is None or c2 is None:
            continue
        if g > 1:
            # Patch extraction flattens patch-major: (N, P*C) -> (N, C, P).
            P = g**3
            C = c1.shape[1] // P
            c1 = c1.reshape(len(c1), P, C).transpose(0, 2, 1).reshape(len(c1), -1)
            c2 = c2.reshape(len(c2), P, C).transpose(0, 2, 1).reshape(len(c2), -1)
        r, n = cellwise_corr(c1, c2, np.ones(c1.shape[1], dtype=bool))
        rows.append({"grid": g, "cells": c1.shape[1], "kept": n, "corr": r})
    del model
    return rows


def main():
    p = argparse.ArgumentParser(description="Does spatial averaging increase cross-view agreement?")
    p.add_argument("--run-dir", required=True, help="Run directory (its settings define the dataset).")
    p.add_argument("--checkpoint-name", default="vqvae_model.pt")
    p.add_argument("--num-samples", type=int, default=400, help="Sample pairs (400 is plenty for a correlation).")
    p.add_argument("--causal", choices=("match", "iid"), default="match")
    p.add_argument("--grids", default="16,8,4,2,1", help="Pooling grids, finest first. 1 = whole volume.")
    p.add_argument("--fg-thresh", type=float, default=0.05, help="Min foreground fraction for a cell to count.")
    p.add_argument("--decompose", action="store_true", help="Also sweep each source of view difference alone.")
    p.add_argument("--with-encoder", action="store_true", help="Also sweep the encoder's content channels.")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", default="alignment_scale_out")
    cli = p.parse_args()

    grids = tuple(int(g) for g in cli.grids.split(","))

    from eval.run_dci_synthetic import build_synthetic_test_set, load_run_args

    ref_args = load_run_args(cli.run_dir)
    dataset = build_synthetic_test_set(ref_args, cli.num_samples, causal=cli.causal == "match")

    # ---- A: the real view pairing, straight from the dataset -----------------
    v1, v2, mk = [], [], []
    for i in range(min(cli.num_samples, len(dataset))):
        s = dataset[i]
        imgs = s["image"]
        v1.append(imgs[0] if imgs[0].dim() == 4 else imgs[0][None])
        v2.append(imgs[1] if imgs[1].dim() == 4 else imgs[1][None])
        m = s.get("mask")
        m = m[0] if isinstance(m, (list, tuple)) else m
        m = torch.as_tensor(m).float()
        mk.append(m if m.dim() == 4 else m[None])
    v1, v2, mk = torch.stack(v1).float(), torch.stack(v2).float(), torch.stack(mk).float()

    all_rows = []
    rows = sweep(v1, v2, mk, grids, cli.fg_thresh)
    for r in rows:
        r["regime"] = "dataset"
    all_rows.extend(rows)
    _print_sweep("A. RAW DATA — the real view pairing (no model involved)", rows)

    # ---- B: which difference makes it scale-dependent ------------------------
    if cli.decompose:
        for regime in ("identical", "modality", "style", "seed", "all"):
            a, b, m = build_pairs(dataset, min(cli.num_samples, len(dataset)), regime, torch.device("cpu"))
            rws = sweep(a, b, m, grids, cli.fg_thresh)
            for r in rws:
                r["regime"] = regime
            all_rows.extend(rws)
            note = {
                "identical": "VALIDITY CHECK — must be 1.0000 flat, or the render path here is wrong",
                "modality": "T1 vs FLAIR LUT only (structured, tied to tissue label)",
                "style": "gain/bias only (GLOBAL scalars — averaging cannot suppress these)",
                "seed": "bias field + render noise only (the component that CAN average down)",
                "all": "all three — should reproduce panel A",
            }[regime]
            _print_sweep(f"B. CONTROLLED PAIRS — differ by: {regime}", rws, note)

    # ---- C: through the encoder ---------------------------------------------
    if cli.with_encoder:
        for untrained in (True, False):
            rws = encoder_sweep(
                cli.run_dir,
                cli.checkpoint_name,
                dataset,
                grids,
                cli.level,
                cli.batch_size,
                cli.num_workers,
                untrained,
            )
            tag = "untrained (random init)" if untrained else "trained"
            for r in rws:
                r["regime"] = f"encoder_{'untrained' if untrained else 'trained'}"
            all_rows.extend(rws)
            _print_sweep(
                f"C. ENCODER CONTENT CHANNELS — {tag}",
                rws,
                "the same pressure at random init = a property of the objective, not of training",
            )

    os.makedirs(cli.out, exist_ok=True)
    path = os.path.join(cli.out, "alignment_scale.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["regime", "grid", "cells", "kept", "corr"])
        for r in all_rows:
            w.writerow([r.get("regime"), r["grid"], r["cells"], r["kept"], r["corr"]])
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
