"""Visualise the encoder receptive field over the brain, with a magnitude falloff profile.

`receptive_field_test` reports the nonzero-gradient BBOX, which is ambiguous: a bbox
spanning the volume can mean a genuinely global RF, or a numerically tiny tail that any
threshold counts as "nonzero". Only the gradient MAGNITUDE distinguishes them, so this
plots it.

Panels:
  A  |d feature / d input| slices through the source voxel, log colour scale, with the
     brain outline overlaid -- shows what the feature actually looks at.
  B  radial falloff: mean |grad| vs distance from the source (log y). A hard RF shows a
     cliff; a long-range-but-real RF decays smoothly; numerical noise sits flat near zero.
  C  cumulative gradient MASS vs radius -> the effective RF ("90% of what this feature
     responds to lies within R voxels"). This is the number to quote, not the bbox.

Both a CORNER and a CENTRE output voxel are computed, so you can see whether corner
features genuinely integrate brain tissue or only pick up a negligible tail.

Usage:
  python -m eval.plot_receptive_field --run-dir results/synthetic/<run>
  python -m eval.plot_receptive_field --run-dir ... --out rf.png
"""
from __future__ import annotations

import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


def grad_map(model, res, grid, level, flat_idx, device):
    """|d(feature at flat_idx)/d(input)| as a (res,res,res) array."""
    x = torch.zeros(1, 1, res, res, res, device=device, requires_grad=True)
    out = model(x, pool_only=True, n_views=1, patch_grid=grid)
    f = (out[2] if isinstance(out, tuple) else out)[level]
    f[0, :, flat_idx].sum().backward()
    return x.grad.detach().abs()[0, 0].cpu().numpy()


def radial_profile(gr, src, nbins=48):
    """(centres, mean |grad| per bin, cumulative mass fraction) vs distance from src."""
    res = gr.shape[0]
    zz, yy, xx = np.meshgrid(*[np.arange(res)] * 3, indexing="ij")
    d = np.sqrt((zz - src[0]) ** 2 + (yy - src[1]) ** 2 + (xx - src[2]) ** 2)
    edges = np.linspace(0, d.max(), nbins + 1)
    idx = np.digitize(d.ravel(), edges) - 1
    g = gr.ravel()
    mean = np.array([g[idx == i].mean() if (idx == i).any() else np.nan for i in range(nbins)])
    mass = np.array([g[idx == i].sum() if (idx == i).any() else 0.0 for i in range(nbins)])
    cum = np.cumsum(mass) / max(mass.sum(), 1e-12)
    return 0.5 * (edges[:-1] + edges[1:]), mean, cum


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, args, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint, device)
    model.eval()
    res = int(getattr(args, "synthetic_res", 64))
    g = cli.grid
    grid = (g, g, g)

    # a real brain mask for the outline
    ds = build_synthetic_test_set(args, 1)
    item = ds[0]
    mask = np.asarray(item["mask"][0]).squeeze()
    img = np.asarray(item["image"][0]).squeeze()

    scale = res / g  # input voxels per feature voxel
    sources = {
        "corner (0,0,0)": (0, np.array([0.5, 0.5, 0.5]) * scale),
        f"centre ({g//2},{g//2},{g//2})": (
            (g // 2) * g * g + (g // 2) * g + (g // 2),
            (np.array([g // 2, g // 2, g // 2]) + 0.5) * scale,
        ),
    }

    fig = plt.figure(figsize=(15, 7.5))
    print(f"\nrun: {cli.run_dir}\ninput {res}^3 | grid {list(grid)} | {scale:.0f} input voxels per feature voxel\n")

    for row, (name, (flat_idx, src)) in enumerate(sources.items()):
        gr = grad_map(model, res, grid, cli.level, flat_idx, device)
        gmax = gr.max()
        sl = int(round(src[0]))
        sl = min(max(sl, 0), res - 1)

        # A: gradient slice with brain outline
        axA = fig.add_subplot(2, 3, row * 3 + 1)
        vis = np.log10(np.maximum(gr[sl], gmax * 1e-8) / gmax)
        im = axA.imshow(vis, cmap="magma", vmin=-6, vmax=0)
        axA.contour(mask[sl], levels=[0.5], colors="cyan", linewidths=1.2)
        axA.plot(src[2], src[1], "w+", markersize=12, markeredgewidth=2)
        axA.set_title(f"{name}\n|grad| (log10, slice z={sl}); cyan = brain", fontsize=9)
        axA.axis("off")
        plt.colorbar(im, ax=axA, fraction=0.046, label="log10(|grad| / max)")

        # B: radial falloff
        r, mean, cum = radial_profile(gr, src)
        axB = fig.add_subplot(2, 3, row * 3 + 2)
        axB.semilogy(r, np.maximum(mean / gmax, 1e-12), lw=1.8)
        brain_r = 0.65 * (res / 2)
        d_centre = float(np.linalg.norm(src - np.array([(res - 1) / 2.0] * 3)))
        axB.axvline(max(d_centre - brain_r, 0), color="cyan", ls="--", lw=1, label="nearest brain")
        axB.axvline(d_centre + brain_r, color="cyan", ls=":", lw=1, label="far brain edge")
        axB.set_xlabel("distance from source (input voxels)")
        axB.set_ylabel("mean |grad| / max")
        axB.set_title("radial falloff", fontsize=9)
        axB.legend(fontsize=7)
        axB.grid(alpha=0.3)

        # C: cumulative gradient mass -> effective RF
        axC = fig.add_subplot(2, 3, row * 3 + 3)
        axC.plot(r, cum, lw=1.8)
        for q, cstyle in ((0.5, "-."), (0.9, "--")):
            rq = float(np.interp(q, cum, r))
            axC.axhline(q, color="grey", ls=cstyle, lw=0.8)
            axC.annotate(f"{int(q*100)}%: r={rq:.0f}", (rq, q), fontsize=8, xytext=(4, -12), textcoords="offset points")
        axC.axvline(max(d_centre - brain_r, 0), color="cyan", ls="--", lw=1)
        axC.set_xlabel("radius (input voxels)")
        axC.set_ylabel("cumulative |grad| mass")
        axC.set_title("effective receptive field", fontsize=9)
        axC.grid(alpha=0.3)

        # numbers
        inside = gr[mask > 0.5].sum() / max(gr.sum(), 1e-12)
        r50 = float(np.interp(0.5, cum, r))
        r90 = float(np.interp(0.9, cum, r))
        print(f"{name}:")
        print(f"   effective RF: 50% of gradient mass within r={r50:.1f}, 90% within r={r90:.1f} input voxels")
        print(f"   fraction of total gradient mass falling INSIDE the brain mask: {inside:.4f}")
        print(f"   distance from this source to the nearest brain voxel: {max(d_centre - brain_r, 0):.1f}\n")

    out = cli.out or os.path.join(cli.run_dir, "receptive_field.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved -> {out}")
    print("\nHow to read it: if the corner's 90%-mass radius is much SMALLER than its distance to")
    print("the brain, and 'fraction inside brain mask' is ~0, then the whole-volume bbox was a")
    print("numerical tail -- the corner does NOT meaningfully see the brain, and the flat R^2")
    print("profile still needs an explanation. If a real share of the mass lands inside the brain,")
    print("the long-range RF is genuine and explains the flat profile.")


if __name__ == "__main__":
    main()
