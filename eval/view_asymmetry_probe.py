#!/usr/bin/env python
"""How much does each VIEW know about each factor?  The ceiling a view-invariant code faces.

A cross-view objective forces the content code to agree across views, so it can only encode
what BOTH views support.  If one view carries a factor much more weakly than the other, an
aligned representation is capped near the *weaker* view — not because the objective is
malfunctioning, but because that is what view-invariance costs.

Reconstruction has no such constraint: it can encode a factor sharply from whichever view
shows it and reconstruct each view from what that view actually contains.  So the recon
baseline is bounded by the BETTER view while the contrastive arm is bounded by the WORSE.

That predicts the measured lesion gap without any appeal to the objective misbehaving:

    lesion   recon-only baseline   +contrastive
    x                     0.775          0.695
    y                     0.804          0.727
    z                     0.522          0.286

This measures the two ceilings directly, from raw pooled voxels — no model, no checkpoint.

Why the lesion is the factor to watch.  ``render_structure`` places it INSIDE white matter
(``lesion_mask = ... & mask_wm``), so the contrast that matters is lesion-against-WM:

                WM (surround)   lesion   contrast   polarity
    T1                   0.80     0.40       0.40   lesion DARKER
    FLAIR                0.40     1.00       0.60   lesion BRIGHTER

Two things follow, and they point in different directions:

* **Amplitude asymmetry is modest.**  FLAIR's contrast is only 1.5x T1's, and both are huge
  against a post-blur noise floor near 0.01.  So the strong form of the ceiling argument —
  "T1 barely sees the lesion" — is probably FALSE, and this probe is expected to show both
  views recovering lesions well.  (An earlier version of this docstring compared the lesion
  to grey matter, which is not the surrounding tissue; that was wrong.)
* **Polarity INVERTS between views.**  The lesion is darker than its surround in T1 and
  brighter in FLAIR.  A view-invariant code therefore cannot encode signed local contrast —
  it has to encode something like |anomaly|, a nonlinear rectification.  That is a real
  extra demand the reconstruction baseline never faces, and it is the fine-scale version of
  the whole-volume contrast inversion that makes raw T1/FLAIR voxels ANTI-correlated
  (measured at -0.21 by ``eval/alignment_scale_test.py``).

So run this to settle the amplitude question, not because the answer is expected to be yes.

Reading the output
------------------
* **T1 ~ contrastive and FLAIR ~ baseline** on the lesions => the contrastive model is
  performing AT the weaker view's ceiling. The objective is paying the correct price for
  view-invariance on a benchmark whose views are very unequal about lesions, and there is
  no locality bug to fix.
* **T1 ~ FLAIR** => the views are equally informative, the ceiling argument fails, and the
  contrastive model's deficit is a genuine failure to use available information.

Built-in control: ``brain_size`` is geometric, not a contrast property, so both views should
recover it about equally.  A large T1/FLAIR gap there means the extraction is wrong, not the
benchmark asymmetric.

Usage
-----
    python -m eval.view_asymmetry_probe --run-dir results/synthetic/RUN --grids 16,8,4
"""

from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np

from eval.identifiability_metrics import cv_probe_r2

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# z_content index order, per eval/legacy_renderer.py (lesion_xyz = z_content[2:5]).
FACTORS = [
    "brain_size",
    "ventricle_size",
    "lesion_x",
    "lesion_y",
    "lesion_z",
    "cortical_thickness",
    "temporal_atrophy",
    "lr_asymmetry",
    "sulcal_widening",
]

# The published per-factor numbers this test is meant to explain (patch pooling, causal
# match, 2000 samples). Printed alongside so the ceilings can be read against them.
MODEL_REF = {"lesion_x": (0.775, 0.695), "lesion_y": (0.804, 0.727), "lesion_z": (0.522, 0.286)}


def extract_views(dataset, grid, batch_size=32, num_workers=0):
    """Pooled raw features for BOTH views plus z_content, aligned by sample order."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    f1, f2, gtc = [], [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"]
            if not isinstance(img, (list, tuple)) or len(img) < 2:
                raise RuntimeError("Expected two views per sample; got a single-view batch.")
            for src, dst in ((img[0], f1), (img[1], f2)):
                x = src.float()
                p = F.adaptive_avg_pool3d(x, (grid, grid, grid)) if x.dim() == 5 else x
                dst.append(p.reshape(p.shape[0], -1).cpu().numpy())
            gtc.append(batch["gt_latents"]["z_content"].numpy())
    return np.concatenate(f1), np.concatenate(f2), np.concatenate(gtc)


def probe_all(X, Z, seeds):
    """Per-factor CV R² of predicting each z_content column from X."""
    return [cv_probe_r2(X, Z[:, j], seeds=seeds)["mean"] for j in range(Z.shape[1])]


def main():
    p = argparse.ArgumentParser(description="Per-view factor ceilings from raw voxels.")
    p.add_argument("--run-dir", required=True, help="Run directory (its settings define the dataset).")
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--causal", choices=("match", "iid"), default="match")
    p.add_argument("--grids", default="16,8,4", help="Pooling grids to sweep (a lesion is local: 8-16 is the range).")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", default="view_asymmetry_out")
    cli = p.parse_args()

    seeds = tuple(int(s) for s in cli.seeds.split(","))
    grids = tuple(int(g) for g in cli.grids.split(","))

    from eval.run_dci_synthetic import build_synthetic_test_set, load_run_args

    ref_args = load_run_args(cli.run_dir)
    dataset = build_synthetic_test_set(ref_args, cli.num_samples, causal=cli.causal == "match")

    rows = []
    for grid in grids:
        logger.info("grid %d^3 ...", grid)
        X1, X2, Z = extract_views(dataset, grid, cli.batch_size, cli.num_workers)
        names = FACTORS[: Z.shape[1]]
        res = {
            "T1": probe_all(X1, Z, seeds),
            "FLAIR": probe_all(X2, Z, seeds),
            "both": probe_all(np.hstack([X1, X2]), Z, seeds),
        }

        print("\n" + "=" * 84)
        print(f"  PER-VIEW FACTOR CEILING — raw voxels pooled to {grid}^3 ({X1.shape[1]} features per view)")
        print("  a view-invariant code is capped near the WEAKER view; reconstruction is not")
        print("=" * 84)
        hdr = f"  {'factor':<20}{'T1':>9}{'FLAIR':>9}{'both':>9}{'FLAIR-T1':>10}   reference (base / contrastive)"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for j, nm in enumerate(names):
            t1, fl, bo = res["T1"][j], res["FLAIR"][j], res["both"][j]
            ref = MODEL_REF.get(nm)
            ref_s = f"{ref[0]:.3f} / {ref[1]:.3f}" if ref else ""
            flag = "  <-" if ref else ""
            print(f"  {nm:<20}{t1:>9.3f}{fl:>9.3f}{bo:>9.3f}{fl - t1:>10.3f}   {ref_s}{flag}")
            rows.append({"grid": grid, "factor": nm, "t1": t1, "flair": fl, "both": bo})

        les = [j for j, nm in enumerate(names) if nm.startswith("lesion")]
        if les:
            mt1 = float(np.mean([res["T1"][j] for j in les]))
            mfl = float(np.mean([res["FLAIR"][j] for j in les]))
            mbs = float(np.mean([MODEL_REF[names[j]][0] for j in les if names[j] in MODEL_REF]))
            mct = float(np.mean([MODEL_REF[names[j]][1] for j in les if names[j] in MODEL_REF]))
            print(f"\n  lesion mean   T1 {mt1:.3f}   FLAIR {mfl:.3f}   |   baseline {mbs:.3f}   contrastive {mct:.3f}")
            if mfl - mt1 < 0.05:
                print("  => VIEWS ARE EQUALLY INFORMATIVE about lesions. The weaker-view ceiling cannot")
                print("     explain the contrastive deficit; it is a genuine failure to use what is there.")
            elif abs(mct - mt1) < abs(mct - mfl):
                print("  => CONTRASTIVE SITS AT THE T1 (WEAKER) CEILING. Consistent with view-invariance")
                print("     costing the stronger view's extra lesion information — not a locality bug.")
            else:
                print("  => Views are unequal, but the contrastive model is NOT at the weaker ceiling.")
                print("     The asymmetry is real yet does not by itself account for the deficit.")

        # brain_size is geometric, so a large per-view gap there indicts the extraction.
        if "brain_size" in names:
            b = names.index("brain_size")
            gap = abs(res["FLAIR"][b] - res["T1"][b])
            if gap > 0.15:
                print(f"\n  [!] CONTROL FAILED: brain_size differs by {gap:.3f} between views. It is geometric,")
                print("      not contrast-dependent, so this points at the extraction rather than the data.")

    os.makedirs(cli.out, exist_ok=True)
    path = os.path.join(cli.out, "view_asymmetry.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grid", "factor", "t1", "flair", "both"])
        for r in rows:
            w.writerow([r["grid"], r["factor"], r["t1"], r["flair"], r["both"]])
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
