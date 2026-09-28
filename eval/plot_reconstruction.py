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

`--swap` adds a third panel per view: the OTHER view's content decoded with THIS view's
style, from `VQVAE.forward(cross_recon=True)` -- content codes held at their own values,
only the final decoder re-run.  It should look like this view's original.  If it looks like
the other view instead, the decoder is ignoring style and the modality rides in content.
`style gain` puts a number on it: 1 = swapping the style alone turns the content view's
reconstruction into this view's, 0 = the style changes nothing.

Usage:
  python -m eval.plot_reconstruction --run-dir results/synthetic/<run>
  python -m eval.plot_reconstruction --run-dir ... --factor-index 1 --n-samples 4
  python -m eval.plot_reconstruction --run-dir ... --swap      # + cross-style decode per view
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


def swaps_by_target(cross, n):
    """Index forward(cross_recon=True)'s extra output by the view each decode should render.

    forward stacks [decode(content_v0, style_v1); decode(content_v1, style_v0)], so its FIRST
    half carries view 1's style and targets view 1.  Exchanging the halves gives
    swap[v] = decode(content_{1-v}, style_v), which is shown and scored against view v.
    """
    a = np.asarray(cross)
    return a.reshape(2, n, *a.shape[1:])[::-1]


def masked_mae(a, b, mask=None):
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    return float(d[mask].mean() if mask is not None else d.mean())


def style_gain(swap, recon_target, recon_donor, mask=None):
    """How much of the view difference swapping in the target's style reproduces, on its own.

    Projects decode(c_donor, s_target) - decode(c_donor, s_donor), the change from swapping
    only the style, onto decode(c_target, s_target) - decode(c_donor, s_donor), the whole
    difference between the two views' reconstructions.  1 = style alone carries the view
    difference; 0 = the decoder ignores style.  Scored on decoder outputs, so reconstruction
    error does not enter it, and within the brain mask, since the recon loss leaves the
    background unconstrained.  NaN when the two reconstructions do not differ.
    """
    arrays = [np.asarray(a, dtype=float) for a in (swap, recon_target, recon_donor)]
    if mask is not None:
        arrays = [a[mask] for a in arrays]
    swap, recon_target, recon_donor = (a.ravel() for a in arrays)
    effect, gap = swap - recon_donor, recon_target - recon_donor
    energy = float(gap @ gap)
    return float(effect @ gap / energy) if energy > 1e-12 else float("nan")


def build_figure(samples, out_png, out_csv, factor_name="ventricle_size"):
    """samples: list of dicts with keys value, orig (2,D,H,W), recon (2,D,H,W).

    Optional, for --swap: swap (2,D,H,W), where swap[v] is the decode that should render view v
    (see swaps_by_target), and mask (D,H,W), the brain voxels its metrics are scored on.
    """
    n = len(samples)
    colours = [RAMP[min(int(i * len(RAMP) / max(n, 1)), len(RAMP) - 1)] for i in range(n)]
    keys = ("orig", "recon", "swap") if "swap" in samples[0] else ("orig", "recon")
    ncol = len(keys)
    titles = {"orig": "{0} original", "recon": "{0} reconstruction", "swap": "{1} content + {0} style"}

    swap_stats = {}
    if "swap" in keys:
        for v in range(2):
            for i, s in enumerate(samples):
                mask = s.get("mask")
                swap_stats[v, i] = {
                    "mae_recon": masked_mae(s["recon"][v], s["orig"][v], mask),
                    "mae_swap": masked_mae(s["swap"][v], s["orig"][v], mask),
                    # What the swap scores if style changes nothing: the content view's own recon.
                    "mae_if_style_ignored": masked_mae(s["recon"][1 - v], s["orig"][v], mask),
                    "style_gain": style_gain(s["swap"][v], s["recon"][v], s["recon"][1 - v], mask),
                }

    fig, axes = plt.subplots(n + 1, 2 * ncol, figsize=(6.5 * ncol, 2.9 * n + 3.4), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")

    rows = []
    for v in range(2):
        # The swap is left out of the window's range: it is judged against this view's
        # original, so it is drawn in that original's window rather than allowed to move it.
        allvals = np.concatenate([[s["orig"][v].ravel(), s["recon"][v].ravel()] for s in samples], axis=None)
        lo, hi = float(np.percentile(allvals, 1)), float(np.percentile(allvals, 99.5))
        for i, s in enumerate(samples):
            for k, key in enumerate(keys):
                ax = axes[i][ncol * v + k]
                ax.imshow(mid_slice(s[key][v]), cmap="gray", vmin=lo, vmax=hi, origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor("#d9d8d2")
                if i == 0:
                    ax.set_title(
                        titles[key].format(VIEW_LABEL[v], VIEW_LABEL[1 - v]),
                        fontsize=11.5,
                        fontweight="bold",
                        color=INK,
                        pad=6,
                    )
                if key == "swap":
                    ax.set_xlabel(f"style gain {swap_stats[v, i]['style_gain']:+.2f}", fontsize=10, color=INK2)
                if v == 0 and k == 0:
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

    # Bottom row: the profiles, one panel per view, spanning all of that view's columns.
    for v in range(2):
        for k in range(ncol):
            axes[n][ncol * v + k].remove()
        ax = fig.add_subplot(n + 1, 2, (n * 2) + v + 1)
        xs = np.arange(len(profile(samples[0]["orig"][v])))
        for i, s in enumerate(samples):
            po, pr = profile(s["orig"][v]), profile(s["recon"][v])
            # Only the originals carry a legend entry: linestyle already encodes
            # original-vs-recon and the x-label says so, so labelling both doubles the
            # legend for no information and it then covers the curves.
            ax.plot(xs, po, color=colours[i], lw=2.0, label=f"{s['value']:+.2f}")
            ax.plot(xs, pr, color=colours[i], lw=2.0, ls="--")
            row = {
                "view": VIEW_LABEL[v],
                factor_name: round(float(s["value"]), 4),
                "dip_area_original": round(dip_area(po), 4),
                "dip_area_recon": round(dip_area(pr), 4),
            }
            if "swap" in keys:
                ps = profile(s["swap"][v])
                ax.plot(xs, ps, color=colours[i], lw=2.0, ls=":")
                row["dip_area_swap"] = round(dip_area(ps), 4)
                row.update({name: round(value, 4) for name, value in swap_stats[v, i].items()})
            rows.append(row)
        linestyles = "solid = original, dashed = reconstruction" + (", dotted = style swap" if "swap" in keys else "")
        ax.set_title(f"{VIEW_LABEL[v]}: profile through the ventricle", fontsize=11.5, fontweight="bold", color=INK)
        ax.set_xlabel(f"x (voxels)  — {linestyles}", fontsize=10, color=INK2)
        ax.set_ylabel("intensity", fontsize=10, color=INK2)
        ax.grid(True, color="#e8e7e1", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_edgecolor("#d9d8d2")
        ax.tick_params(colors=MUTED, labelsize=9)
        curves = [profile(s["orig"][v]) for s in samples]
        if "swap" in keys:
            # A swap that kept the other view's contrast can sit outside this view's range.
            curves += [profile(s["swap"][v]) for s in samples]
        top = max(float(np.nanmax(c)) for c in curves)
        bot = min(float(np.nanmin(c)) for c in curves)
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
        f"{'Reconstruction and cross-style swap' if 'swap' in keys else 'Reconstruction'} "
        f"vs original across the {factor_name} range",
        fontsize=13.5,
        fontweight="bold",
        color=INK,
        y=0.995,
    )
    if "swap" in keys:
        note = (
            "Each view shares one grayscale window across its panels. The style swap should match that view's "
            "original (style gain 1); gain 0 means the decoder ignores style and the modality rides in content."
        )
    else:
        note = (
            "Each view shares one grayscale window across original and reconstruction. "
            "If the dashed curve's cavity does not widen with the factor, the reconstruction is not carrying it."
        )
    fig.text(0.5, 0.004, note, ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=[0.01, 0.02, 1, 0.97])
    fig.savefig(out_png, dpi=160, facecolor=fig.get_facecolor())

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
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

    # forward's cross output, labelled 10 * content view + style view, in forward's row order.
    cross = np.concatenate([np.full((2, 4), 10 * 0 + 1.0), np.full((2, 4), 10 * 1 + 0.0)])
    swap = swaps_by_target(cross, 2)
    assert (swap[0] == 10).all() and (swap[1] == 1).all(), "swap[v] must pair view v's style with the other content"

    t1, flair = vol(0.20), 0.5 * vol(0.20)
    for planted, want in ((t1, 1.0), (flair, 0.0), ((t1 + flair) / 2, 0.5)):
        got = style_gain(planted, t1, flair)
        assert abs(got - want) < 1e-9, f"style gain {got} for a planted {want}"

    # Style ignored: each swap comes back as its content view's own reconstruction.
    brain = d < 0.65
    swap_samples = []
    for value in (-0.8, 0.0, 0.8):
        both = np.stack([vol(0.20 + value * 0.08), 0.5 * vol(0.20 + value * 0.08)])
        swap_samples.append({"value": value, "orig": both, "recon": both, "swap": both[::-1], "mask": brain})
    rows = build_figure(swap_samples, f"{out}_swap.png", f"{out}_swap.csv")
    assert all(abs(r["style_gain"]) < 1e-9 for r in rows), [r["style_gain"] for r in rows]
    assert all(r["mae_swap"] == r["mae_if_style_ignored"] > r["mae_recon"] == 0 for r in rows), rows
    print("swap order, style gain 1/0/0.5 and the style-ignored panel all recovered")
    print(f"self-test OK: wrote {out}.png and {out}_swap.png")


def main():
    ap = argparse.ArgumentParser(description="Plot T1/FLAIR reconstructions against originals.")
    ap.add_argument("--run-dir", help="Training run dir with settings.json.")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--factor-index", type=int, default=1, help="GT z_content index to span (1 = ventricle_size).")
    ap.add_argument("--factor-name", default="ventricle_size")
    ap.add_argument("--n-samples", type=int, default=3, help="Samples shown, spanning the factor's range.")
    ap.add_argument("--scan", type=int, default=48, help="Candidates to draw before picking the spread.")
    ap.add_argument("--out", default="reconstruction.png")
    ap.add_argument(
        "--swap",
        action="store_true",
        help="Add a panel per view: the other view's content decoded with this view's style "
        "(VQVAE.forward cross_recon=True). Needs style injected into the decoder at level 0.",
    )
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
        out = model(
            x, return_recon=True, pool_only=True, n_views=2, subsets=[(0, 1)], mask=fwd_mask, cross_recon=args.swap
        )
    # cross_recon appends the swapped decode after the usual eight-tuple.
    out, cross = out if args.swap else (out, None)
    y = out[0]
    if y is None or y.shape != x.shape:
        raise SystemExit("No reconstruction returned — was the run trained with a decoder?")
    recon = y.detach().cpu().numpy().reshape(2, len(picks), *y.shape[1:])[:, :, 0]
    orig = x.detach().cpu().numpy().reshape(2, len(picks), *x.shape[1:])[:, :, 0]

    samples = [
        {"value": vals[p], "orig": np.stack([orig[0][i], orig[1][i]]), "recon": np.stack([recon[0][i], recon[1][i]])}
        for i, p in enumerate(picks)
    ]
    if cross is not None:
        swap = swaps_by_target(cross.detach().cpu().numpy()[:, 0], len(picks))
        if masks[picks[0]] is None:
            logger.warning("No brain mask in the batch: swap metrics include the unconstrained background.")
        for i, (s, p) in enumerate(zip(samples, picks)):
            s["swap"] = swap[:, i]
            if masks[p] is not None:
                s["mask"] = masks[p].numpy().reshape(orig.shape[2:]) > 0
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
    if cross is None:
        return
    print("\n  Style swap: the other view's content decoded with this view's style; MAEs against this view's original.")
    print(
        f"  {'view':<8}{args.factor_name:>16}{'recon MAE':>12}{'swap MAE':>11}{'style ignored':>15}{'style gain':>12}"
    )
    print("  " + "-" * 74)
    for r in rows:
        print(
            f"  {r['view']:<8}{r[args.factor_name]:>16.3f}{r['mae_recon']:>12.4f}{r['mae_swap']:>11.4f}"
            f"{r['mae_if_style_ignored']:>15.4f}{r['style_gain']:>12.2f}"
        )
    print("\n  'style ignored' = the MAE the swap would score if style changed nothing (the content view's own recon).")
    print("  Swap MAE near recon MAE, gain ~1: style carries the modality.")
    print("  Swap MAE near 'style ignored', gain ~0: the decoder ignores style and the modality rides in content.")


if __name__ == "__main__":
    main()
