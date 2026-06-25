#!/usr/bin/env python
"""Figure: are content latents identified *locally* (spatially) or as a *global channel feature*?

Tests the "local, not global-channel" hypothesis by plotting recovery and usable
dimensionality against pooling LOCALITY, using the CSVs that ``eval.content_rank_pca``
already writes (``content_rank_pca.csv`` + ``content_rank_per_factor.csv``):

    gap   = global per-channel mean          (most global — 1 value / channel)
    stats = per-channel mean/std/max/min     (semi-global — 4 / channel)
    patch = local patch grid, e.g. 4^3       (most local — 64 / channel)

Three panels:
  A  effective rank vs locality   — if the usable dimensionality is far higher locally
                                     than globally, the content lives in spatial structure,
                                     not the global channel mean. (The cleanest evidence.)
  B  aggregate recovery R² vs locality.
  C  per-factor R² × pooling heatmap (one model), rows sorted by locality gain
     (patch − gap) so the most *locally* identifiable factors sit at the top.

Torch-free — only pandas/matplotlib/numpy — so it runs anywhere the CSVs are.

Usage:
    python -m eval.plot_local_vs_global --in content_rank_out
    python -m eval.plot_local_vs_global --in content_rank_out --model <name> --out fig.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pooling keys ordered most-global -> most-local
LOCALITY = ["gap", "stats", "patch"]
XLABEL = {"gap": "gap\n(global mean)", "stats": "stats\n(per-channel)", "patch": "patch\n(local)"}


def _present(seq, order=LOCALITY):
    s = set(seq)
    return [p for p in order if p in s]


def main():
    ap = argparse.ArgumentParser(description="Local-vs-global identifiability figure.")
    ap.add_argument("--in", dest="indir", default="content_rank_out", help="Dir holding the content_rank_pca CSVs.")
    ap.add_argument("--model", default=None, help="Model for the per-factor heatmap (default: first in the CSV).")
    ap.add_argument("--out", default=None, help="Output PNG (default: <in>/local_vs_global.png).")
    args = ap.parse_args()

    rk = pd.read_csv(os.path.join(args.indir, "content_rank_pca.csv"))
    pf = pd.read_csv(os.path.join(args.indir, "content_rank_per_factor.csv"))

    # eff_rank / r2_full are constant across the per-k rows — collapse to one per (model, pooling).
    summ = rk.drop_duplicates(subset=["model", "pooling"])[["model", "pooling", "eff_rank", "r2_full"]]
    models = list(dict.fromkeys(summ["model"]))
    pools = _present(summ["pooling"])
    if not pools:
        raise SystemExit("No gap/stats/patch poolings in content_rank_pca.csv.")
    x = np.arange(len(pools))

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 5))

    # ---- A: effective rank vs locality ------------------------------------------------
    for m in models:
        s = summ[summ["model"] == m].set_index("pooling").reindex(pools)
        axA.plot(x, s["eff_rank"].values, marker="o", label=m)
    axA.set_xticks(x)
    axA.set_xticklabels([XLABEL.get(p, p) for p in pools])
    axA.set_ylabel("effective rank (usable dims)")
    axA.set_title("A. Usable dimensionality vs locality\nrises sharply => content is local")
    axA.grid(alpha=0.3)
    axA.legend(fontsize=7)

    # ---- B: aggregate recovery vs locality --------------------------------------------
    for m in models:
        s = summ[summ["model"] == m].set_index("pooling").reindex(pools)
        axB.plot(x, s["r2_full"].values, marker="o", label=m)
    axB.set_xticks(x)
    axB.set_xticklabels([XLABEL.get(p, p) for p in pools])
    axB.set_ylabel("content recovery R² (full block)")
    axB.set_title("B. Recovery vs locality")
    axB.grid(alpha=0.3)
    axB.legend(fontsize=7)

    # ---- C: per-factor R² x pooling heatmap, rows sorted by locality gain --------------
    model = args.model or models[0]
    pm = pf[pf["model"] == model]
    cpools = _present(pm["pooling"])
    piv = pm.pivot_table(index="factor", columns="pooling", values="r2_linear").reindex(columns=cpools)
    if "patch" in cpools and "gap" in cpools:
        gain = (piv["patch"] - piv["gap"]).fillna(-np.inf)
        piv = piv.loc[gain.sort_values(ascending=False).index]
    data = piv.values.astype(float)
    vmax = max(0.1, float(np.nanmax(data)))
    im = axC.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    axC.set_xticks(range(len(cpools)))
    axC.set_xticklabels([XLABEL.get(p, p) for p in cpools])
    axC.set_yticks(range(len(piv.index)))
    axC.set_yticklabels(piv.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                axC.text(
                    j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.5 * vmax else "black", fontsize=7
                )
    axC.set_title(f"C. Per-factor R² x locality\n{model} (sorted by patch-gap gain)")
    fig.colorbar(im, ax=axC, fraction=0.046, label="R²")

    fig.suptitle("Local vs global identifiability of content factors", y=1.02, fontsize=13)
    fig.tight_layout()
    out = args.out or os.path.join(args.indir, "local_vs_global.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
