"""How far can a CORNER feature actually see? Empirical receptive field + feature variance.

Motivating puzzle: for the LayerNorm baseline, background positions 7-8 voxels from
any brain tissue predict brain_size at R^2 0.77, and the profile RISES with distance.
Input there is exactly zero (brain-masked) and LayerNorm is per-voxel, so there is no
obvious global-coupling operation. Two candidate resolutions, both measured here:

  1. RECEPTIVE FIELD. If the encoder RF is large enough to span the volume, every
     output position -- corners included -- genuinely sees the whole brain. The flat
     profile is then expected, and the RISE with distance is just "far positions have
     no local anatomy competing as probe noise". Measured by backprop: d(corner
     feature)/d(input), then the spatial extent of |grad| and whether it overlaps the
     brain region.

  2. FEATURE VARIANCE. If far-background features were constant across samples they
     could not predict anything (R^2 would be ~0), so a high R^2 there REQUIRES real
     across-sample variance. Reported for far-bg vs core positions as a sanity check
     on the probe itself.

Verdict logic:
  RF reaches the brain  -> no exotic mechanism needed; the corner signal is ordinary
                           (if very long-range) convolutional reach.
  RF does NOT reach     -> a genuine non-local route survives LayerNorm; that is a
                           real finding and needs a different explanation.

Usage:
  python -m eval.receptive_field_test --run-dir results/synthetic/<run>
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--num-samples", type=int, default=64, help="Samples for the variance check.")
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
    res = int(getattr(args, "synthetic_res", 64))
    print(f"\nrun: {cli.run_dir}")
    print(f"input {res}^3 | scaling_rates={getattr(args, 'vqvae_scaling_rates', '?')} | probe grid {list(grid)}")

    # ---- 1. Empirical receptive field of a CORNER output position ----------------
    x = torch.zeros(1, 1, res, res, res, device=device, requires_grad=True)
    out = model(x, pool_only=True, n_views=1, patch_grid=grid)
    f = (out[2] if isinstance(out, tuple) else out)[cli.level]  # (1, C, P)
    P = f.shape[-1]
    gg = round(P ** (1 / 3))
    corner_flat = 0  # position (0,0,0) in the pooled grid
    f[0, :, corner_flat].sum().backward()
    gr = x.grad.detach().abs()[0, 0].cpu().numpy()  # (res,res,res)

    thr = gr.max() * 1e-3
    nz = np.argwhere(gr > thr)
    if len(nz) == 0:
        print("\n  RF: corner feature has ZERO gradient w.r.t. the input (dead position).")
        return
    lo, hi = nz.min(0), nz.max(0)
    extent = hi - lo + 1
    # corner output position corresponds to input corner (0,0,0)
    reach = float(np.abs(nz - np.array([0, 0, 0])).max())
    centre = (res - 1) / 2.0
    dist_to_centre = float(np.sqrt(((nz - centre) ** 2).sum(1)).min())
    brain_radius_vox = 0.65 * (res / 2)  # gm radius ~0.65 in normalised coords

    print("\n1) empirical receptive field of the CORNER output voxel (input voxels):")
    print(f"   nonzero-gradient bbox: {lo.tolist()} -> {hi.tolist()}   extent {extent.tolist()}")
    print(f"   max reach from the corner            : {reach:.1f} voxels")
    print(f"   closest RF voxel to volume centre    : {dist_to_centre:.1f} voxels")
    print(f"   brain surface is at radius           : {brain_radius_vox:.1f} voxels from centre")
    sees_brain = dist_to_centre <= brain_radius_vox
    print(f"   => corner RF {'REACHES' if sees_brain else 'does NOT reach'} the brain")

    # ---- 2. Across-sample variance of far-background vs core features ------------
    ds = build_synthetic_test_set(args, cli.num_samples)
    loader = DataLoader(ds, batch_size=min(32, cli.num_samples), shuffle=False, num_workers=0)
    feats, covs = [], []
    import torch.nn.functional as Fn

    for batch in loader:
        imgs = batch["image"]
        nv = len(imgs)
        xb = torch.cat(imgs, 0).to(device)
        with torch.no_grad():
            o = model(xb, pool_only=True, n_views=nv, patch_grid=grid)
            ff = (o[2] if isinstance(o, tuple) else o)[cli.level]
            ff = ff.reshape(nv, -1, *ff.shape[1:])[0]
        feats.append(ff.cpu().numpy())
        mm = torch.cat(batch["mask"], 0).to(device).float()
        covs.append(Fn.adaptive_avg_pool3d(mm, grid).flatten(1)[: ff.shape[0]].cpu().numpy())
    F = np.concatenate(feats, 0)
    cov = np.concatenate(covs, 0).mean(0)

    d, h, w = np.unravel_index(np.arange(P), (gg, gg, gg))
    c = (gg - 1) / 2.0
    dc = np.sqrt((d - c) ** 2 + (h - c) ** 2 + (w - c) ** 2)
    far_bg = (cov <= 0.1) & (dc >= dc.max() - 2)  # extreme corners
    core = cov >= 0.9

    def spread(sel):
        sd = F[:, :, sel].std(axis=0)  # (C, n_sel) across samples
        return float(sd.mean())

    print(f"\n2) across-sample feature variability (N={F.shape[0]}):")
    print(f"   core positions        (n={int(core.sum()):5d}) mean std = {spread(core):.5f}")
    print(f"   extreme-corner bg     (n={int(far_bg.sum()):5d}) mean std = {spread(far_bg):.5f}")
    print("   (corner std ~0 would make any corner R^2 impossible -> probe artefact;")
    print("    substantial corner std means information genuinely arrives there.)")

    print("\nverdict:")
    if sees_brain:
        print("  The corner RF REACHES the brain -> corner features legitimately see brain tissue.")
        print("  A flat/rising R^2-vs-distance profile is then EXPECTED (all positions see the whole")
        print("  brain; far ones have no competing local anatomy) and needs no global-coupling story.")
    else:
        print("  The corner RF does NOT reach the brain, yet corner features vary across samples and")
        print("  predict brain_size -> a genuine non-local route survives LayerNorm. Real finding;")
        print("  none of the mechanisms tested so far (GroupNorm, padding, RF) explains it.")


if __name__ == "__main__":
    main()
