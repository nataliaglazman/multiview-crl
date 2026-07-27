"""Does the model localize ventricle_size at the ventricle's actual location (centre)?

The coarse background_leak readout averages over ~260 in-brain positions; the
ventricle is 1-2 feature voxels at the centre, so that average DILUTES any local
signal toward zero (core|bg ~ 0.06 may be an artefact of dilution, not absence).

This probes the CENTRAL voxels specifically -- no averaging over the whole brain --
and asks whether they carry ventricle_size BEYOND the global level that background
already provides:

    R2_bg              far-background feature -> ventricle   (the global gauge)
    R2_centre          central-voxel feature  -> ventricle   (raw)
    R2_centre | bg     central AFTER removing the bg-predictable part -> ventricle

  centre|bg >> 0  -> the model DOES localize the ventricle at its real location
                     (region-averaging was hiding it; the global story is incomplete).
  centre|bg ~ 0   -> the centre carries nothing past the global gauge -> global shortcut,
                     properly measured this time.

For contrast it also prints the diluted whole-core number (core|bg) so the effect
of averaging is visible side by side.

Usage:
  python -m eval.central_ventricle_probe --run-dir results/synthetic/<run> --num-samples 800
  python -m eval.central_ventricle_probe --run-dir ... --grid 32   # finer, if native supports it
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.linear_model import LinearRegression

from eval.dci import CONTENT_FACTOR_NAMES
from eval.identifiability_metrics import cv_probe_r2

logger = logging.getLogger(__name__)
FACTORS = ("ventricle_size", "brain_size", "lr_asymmetry")


def resid(a, b):
    return a - LinearRegression().fit(b, a).predict(b)


def probe(Xf, y):
    return float("nan") if Xf is None else cv_probe_r2(Xf, y, kind="ridge")["mean"]


def region_readout(F, sel):
    """Mean feature over selected positions -> (N, C).  F: (N, C, P)."""
    return None if sel.sum() == 0 else F[:, :, sel].mean(axis=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--num-samples", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from torch.utils.data import DataLoader

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, args, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint, device)
    model.eval()
    g = cli.grid
    grid = (g, g, g)

    ds = build_synthetic_test_set(args, cli.num_samples)
    loader = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=0)
    feats, cov_acc, Z = [], [], []
    for batch in loader:
        imgs = batch["image"]
        nv = len(imgs)
        x = torch.cat(imgs, 0).to(device)
        with torch.no_grad():
            out = model(x, pool_only=True, n_views=nv, patch_grid=grid)
            f = (out[2] if isinstance(out, tuple) else out)[cli.level]
            f = f.reshape(nv, -1, *f.shape[1:])[0]  # (B, C, P)
        feats.append(f.cpu().numpy().astype(np.float32))
        m = batch.get("mask")
        mm = torch.cat(m, 0).to(device).float()
        cov_acc.append(Fn.adaptive_avg_pool3d(mm, grid).flatten(1)[: f.shape[0]].cpu().numpy())
        v = batch["gt_latents"]["z_content"]
        Z.append(np.asarray(v).reshape(v.shape[0], -1))
    F = np.concatenate(feats, 0)
    cov = np.concatenate(cov_acc, 0).mean(0)  # (P,) mean coverage
    Z = np.concatenate(Z, 0)
    N, C, P = F.shape

    # distance of each position from the volume centre
    idx = np.arange(P)
    d, h, w = np.unravel_index(idx, (g, g, g))
    c = (g - 1) / 2.0
    dist_c = np.sqrt((d - c) ** 2 + (h - c) ** 2 + (w - c) ** 2)

    centre = dist_c <= 1.5  # the ventricle location (few voxels)
    core = cov >= 0.9  # the whole brain (what background_leak averaged over)
    bg = cov <= 0.1

    print(f"\nrun: {cli.run_dir} | level {cli.level} | grid {list(grid)} | N={N} C={C} P={P}")
    print(
        f"positions: centre(<=1.5vox)={int(centre.sum())}  core(>=0.9)={int(core.sum())}  bg(<=0.1)={int(bg.sum())}\n"
    )

    f_bg = region_readout(F, bg)
    f_core = region_readout(F, core)
    f_ctr = region_readout(F, centre)

    print(f"{'factor':<16}{'R2_bg':>9}{'R2_core':>9}{'R2_centre':>11}{'core|bg':>9}{'CENTRE|bg':>11}")
    print("-" * 65)
    for fname in FACTORS:
        y = Z[:, CONTENT_FACTOR_NAMES.index(fname)]
        r_bg = probe(f_bg, y)
        r_core = probe(f_core, y)
        r_ctr = probe(f_ctr, y)
        r_core_bg = probe(resid(f_core, f_bg), y)
        r_ctr_bg = probe(resid(f_ctr, f_bg), y)
        print(f"{fname:<16}{r_bg:>9.3f}{r_core:>9.3f}{r_ctr:>11.3f}{r_core_bg:>9.3f}{r_ctr_bg:>11.3f}")

    # concentric central shells for ventricle: does the signal peak AT the centre?
    print("\nventricle_size by distance-from-centre shell (centre|bg residualised):")
    y = Z[:, CONTENT_FACTOR_NAMES.index("ventricle_size")]
    print(f"  {'shell (vox)':<14}{'n':>6}{'R2':>9}{'R2|bg':>9}")
    for lo, hi in [(0, 1.5), (1.5, 3), (3, 5), (5, 8)]:
        sel = (dist_c >= lo) & (dist_c < hi)
        rr = probe(region_readout(F, sel), y)
        rr_bg = probe(resid(region_readout(F, sel), f_bg), y) if sel.sum() else float("nan")
        print(f"  [{lo},{hi})".ljust(16) + f"{int(sel.sum()):>6}{rr:>9.3f}{rr_bg:>9.3f}")

    print("\nverdict:")
    y = Z[:, CONTENT_FACTOR_NAMES.index("ventricle_size")]
    v_ctr_bg = probe(resid(f_ctr, f_bg), y)
    v_core_bg = probe(resid(f_core, f_bg), y)
    if v_ctr_bg > max(0.05, 2 * v_core_bg):
        print(f"  CENTRE|bg = {v_ctr_bg:.3f} >> core|bg = {v_core_bg:.3f}: the ventricle IS locally encoded at the")
        print("  centre -- region-averaging over the whole brain diluted it. The 'purely global' read was wrong.")
    elif v_ctr_bg > 0.05:
        print(f"  CENTRE|bg = {v_ctr_bg:.3f}: some local encoding at the centre, modest. Both global + local present.")
    else:
        print(f"  CENTRE|bg = {v_ctr_bg:.3f} ~ 0: even the central voxels carry nothing past the global gauge.")
        print("  Global shortcut confirmed with the dilution objection controlled for.")


if __name__ == "__main__":
    main()
