#!/usr/bin/env python
"""Look at T1 and FLAIR reconstructions next to their originals -- is the ventricle there?

Picks samples spanning the ventricle_size range and shows, per view, the original, the
reconstruction, and an intensity profile straight through the ventricle.  The profile is
the decisive panel: the ventricle is a dark cavity, so it reads as a DIP, and the question
is whether the reconstruction's dip tracks the ground-truth ventricle size or stays flat.

Factors are drawn i.i.d. (`causal=False`).  That matters here for a concrete reason: under
the run's SCM ventricle_size correlates ~0.8 with brain_size, so the "large ventricle"
sample would also be the large-brain sample and you could not tell which one the
reconstruction is responding to.

Images share one grayscale window per view across original and reconstruction -- per-panel
autoscaling would hide exactly the difference being looked for.

Usage:
  python -m eval.plot_reconstruction --run-dir results/synthetic/<run>
  python -m eval.plot_reconstruction --run-dir ... --factor-index 1 --n-samples 4
  python -m eval.plot_reconstruction --self-test        # torch-free, checks the layout
"""

from __future__ import annotations

import argparse
import csv
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

VIEW_LABEL = ("T1", "FLAIR")
# Sequential ramp: the series IS a magnitude (small -> large ventricle), so one hue
# stepped by lightness, never categorical hues. Original vs recon is linestyle, not colour.
RAMP = ("#9dc3f0", "#2a78d6", "#0b3d7a")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"


def profile(vol):
    """Intensity along x through the volume centre -- crosses both ventricle lobes."""
    a = np.asarray(vol).squeeze()
    return a[:, a.shape[1] // 2, a.shape[2] // 2]


def mid_slice(vol):
    a = np.asarray(vol).squeeze()
    return a[:, :, a.shape[2] // 2].T


def dip_area(prof, centre_frac=0.30, ring=(0.35, 0.60)):
    """Integrated intensity deficit below the surrounding tissue, across the central window.

    NOT the dip's depth: depth is fixed by the CSF-vs-WM contrast and is the same whatever
    the cavity's size, so it cannot track ventricle_size at all.  Size changes the dip's
    WIDTH, so the area under the surround level is the quantity that moves.  Integrating
    also handles the septum split, which puts two lobes either side of a thin WM bridge.
    """
    p = np.asarray(prof, dtype=float)
    n = len(p)
    # Normalised distance from the profile's centre: 0 at the middle, 1 at either end.
    # Fractions rather than voxel counts so this is resolution-independent -- and the ring
    # must stay INSIDE the brain (edge at ~0.65 of the half-width), or the surround level
    # is read off background and every area collapses to zero.
    pos = np.abs(np.arange(n) - (n - 1) / 2.0) / (n / 2.0)
    core = pos <= centre_frac
    surround = (pos >= ring[0]) & (pos <= ring[1])
    if not core.any() or not surround.any():
        return float("nan")
    base = float(np.median(p[surround]))
    return float(np.clip(base - p[core], 0, None).sum())


def build_figure(samples, out_png, out_csv, factor_name="ventricle_size"):
    """samples: list of dicts with keys value, orig (2,D,H,W), recon (2,D,H,W)."""
    n = len(samples)
    colours = [RAMP[min(int(i * len(RAMP) / max(n, 1)), len(RAMP) - 1)] for i in range(n)]

    fig, axes = plt.subplots(n + 1, 4, figsize=(13.0, 2.9 * n + 3.4), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")

    rows = []
    for v in range(2):
        allvals = np.concatenate([[s["orig"][v].ravel(), s["recon"][v].ravel()] for s in samples], axis=None)
        lo, hi = float(np.percentile(allvals, 1)), float(np.percentile(allvals, 99.5))
        for i, s in enumerate(samples):
            for k, key in enumerate(("orig", "recon")):
                ax = axes[i][2 * v + k]
                ax.imshow(mid_slice(s[key][v]), cmap="gray", vmin=lo, vmax=hi, origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor("#d9d8d2")
                if i == 0:
                    ax.set_title(
                        f"{VIEW_LABEL[v]} {'original' if key == 'orig' else 'reconstruction'}",
                        fontsize=11.5,
                        fontweight="bold",
                        color=INK,
                        pad=6,
                    )
                if 2 * v + k == 0:
                    ax.text(
                        -0.07,
                        0.5,
                        f"{factor_name}\n{s['value']:+.2f}",
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=10,
                        color=INK,
                    )

    # Bottom row: the profiles, one panel per view, spanning two columns each.
    for v in range(2):
        for k in range(2):
            axes[n][2 * v + k].remove()
        ax = fig.add_subplot(n + 1, 2, (n * 2) + v + 1)
        xs = np.arange(len(profile(samples[0]["orig"][v])))
        for i, s in enumerate(samples):
            po, pr = profile(s["orig"][v]), profile(s["recon"][v])
            # Only the originals carry a legend entry: linestyle already encodes
            # original-vs-recon and the x-label says so, so labelling both doubles the
            # legend for no information and it then covers the curves.
            ax.plot(xs, po, color=colours[i], lw=2.0, label=f"{s['value']:+.2f}")
            ax.plot(xs, pr, color=colours[i], lw=2.0, ls="--")
            rows.append(
                {
                    "view": VIEW_LABEL[v],
                    factor_name: round(float(s["value"]), 4),
                    "dip_area_original": round(dip_area(po), 4),
                    "dip_area_recon": round(dip_area(pr), 4),
                }
            )
        ax.set_title(f"{VIEW_LABEL[v]}: profile through the ventricle", fontsize=11.5, fontweight="bold", color=INK)
        ax.set_xlabel("x (voxels)  — solid = original, dashed = reconstruction", fontsize=10, color=INK2)
        ax.set_ylabel("intensity", fontsize=10, color=INK2)
        ax.grid(True, color="#e8e7e1", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_edgecolor("#d9d8d2")
        ax.tick_params(colors=MUTED, labelsize=9)
        top = max(float(np.nanmax(profile(s["orig"][v]))) for s in samples)
        bot = min(float(np.nanmin(profile(s["orig"][v]))) for s in samples)
        ax.set_ylim(bot - 0.05 * (top - bot), top + 0.45 * (top - bot))  # headroom for the legend
        leg = ax.legend(
            frameon=False,
            fontsize=9,
            ncol=min(len(samples), 4),
            labelcolor=INK2,
            loc="upper center",
            columnspacing=1.4,
            handlelength=1.6,
        )
        leg.set_title(factor_name, prop={"size": 9})
        leg.get_title().set_color(MUTED)

    fig.suptitle(
        f"Reconstruction vs original across the {factor_name} range",
        fontsize=13.5,
        fontweight="bold",
        color=INK,
        y=0.995,
    )
    fig.text(
        0.5,
        0.004,
        "Each view shares one grayscale window across original and reconstruction. "
        "If the dashed curve's cavity does not widen with the factor, the reconstruction is not carrying it.",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )
    fig.tight_layout(rect=[0.01, 0.02, 1, 0.97])
    fig.savefig(out_png, dpi=160, facecolor=fig.get_facecolor())

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["view", factor_name, "dip_area_original", "dip_area_recon"])
        w.writeheader()
        w.writerows(rows)
    return rows


def _self_test():
    """Planted volumes: the original's cavity tracks the factor, the recon's does not."""
    res = 32
    g = np.linspace(-1, 1, res)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    d = np.sqrt(X**2 + Y**2 + Z**2)

    def vol(vent):
        v = np.where(d < 0.65, 0.8, 0.0)
        return np.where(d < vent, 0.1, v)

    samples = []
    for value in (-0.8, 0.0, 0.8):
        vent = 0.20 + value * 0.08
        orig = np.stack([vol(vent), vol(vent)])
        recon = np.stack([vol(0.20), vol(0.20)])  # frozen: ignores the factor
        samples.append({"value": value, "orig": orig, "recon": recon})

    out = "/tmp/_plot_recon_selftest"
    rows = build_figure(samples, f"{out}.png", f"{out}.csv")
    orig_dips = [r["dip_area_original"] for r in rows if r["view"] == "T1"]
    recon_dips = [r["dip_area_recon"] for r in rows if r["view"] == "T1"]
    assert max(orig_dips) - min(orig_dips) > 1e-6, f"planted variation not detected: {orig_dips}"
    assert len(set(recon_dips)) == 1, f"frozen recon should give a constant area, got {recon_dips}"
    print(f"original cavity areas vary {orig_dips}; frozen-recon areas constant {recon_dips}")
    print(f"self-test OK: wrote {out}.png")


def main():
    ap = argparse.ArgumentParser(description="Plot T1/FLAIR reconstructions against originals.")
    ap.add_argument("--run-dir", help="Training run dir with settings.json.")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--factor-index", type=int, default=1, help="GT z_content index to span (1 = ventricle_size).")
    ap.add_argument("--factor-name", default="ventricle_size")
    ap.add_argument("--n-samples", type=int, default=3, help="Samples shown, spanning the factor's range.")
    ap.add_argument("--scan", type=int, default=48, help="Candidates to draw before picking the spread.")
    ap.add_argument("--out", default="reconstruction.png")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.self_test:
        _self_test()
        return
    if not args.run_dir:
        ap.error("--run-dir is required (or pass --self-test)")

    import torch
    from torch.utils.data import DataLoader

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    model, run_args, device = load_model_from_run_dir(args.run_dir, args.checkpoint, None)
    model.eval()
    inner = model.module if hasattr(model, "module") else model

    ds = build_synthetic_test_set(run_args, args.scan, causal=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    imgs, masks, vals = [], [], []
    for batch in loader:
        v1, v2 = batch["image"]
        for b in range(v1.shape[0]):
            imgs.append((v1[b], v2[b]))
            m = batch.get("mask")
            masks.append(m[0][b] if m is not None else None)
            vals.append(float(batch["gt_latents"]["z_content"][b, args.factor_index]))
    order = np.argsort(vals)
    picks = [int(order[i]) for i in np.linspace(0, len(order) - 1, args.n_samples).round().astype(int)]
    logger.info("picked %s at %s = %s", picks, args.factor_name, [round(vals[p], 3) for p in picks])

    # View-major batching: separate encoders and per-view codebooks require all of view 0
    # then all of view 1, matching eval/lesion_reconstruction.reconstruct.
    x = torch.cat([torch.stack([imgs[p][v] for p in picks]) for v in range(2)]).to(device)
    fwd_mask = None
    if getattr(inner, "latent_mask", False) and masks[picks[0]] is not None:
        fwd_mask = torch.cat([torch.stack([masks[p] for p in picks])] * 2).to(device)
    with torch.no_grad():
        y = model(x, return_recon=True, pool_only=True, n_views=2, subsets=[(0, 1)], mask=fwd_mask)[0]
    if y is None or y.shape != x.shape:
        raise SystemExit("No reconstruction returned — was the run trained with a decoder?")
    recon = y.detach().cpu().numpy().reshape(2, len(picks), *y.shape[1:])[:, :, 0]
    orig = x.detach().cpu().numpy().reshape(2, len(picks), *x.shape[1:])[:, :, 0]

    samples = [
        {"value": vals[p], "orig": np.stack([orig[0][i], orig[1][i]]), "recon": np.stack([recon[0][i], recon[1][i]])}
        for i, p in enumerate(picks)
    ]
    out_csv = args.out.rsplit(".", 1)[0] + ".csv"
    rows = build_figure(samples, args.out, out_csv, factor_name=args.factor_name)
    print(f"\nwrote {args.out} and {out_csv}\n")
    print(f"  {'view':<8}{args.factor_name:>16}{'cavity area':>15}{'recon area':>12}")
    print("  " + "-" * 51)
    for r in rows:
        print(
            f"  {r['view']:<8}{r[args.factor_name]:>16.3f}{r['dip_area_original']:>15.3f}{r['dip_area_recon']:>12.3f}"
        )
    print("\n  A recon area that stays flat while the original's grows = the ventricle is not encoded.")


if __name__ == "__main__":
    main()
