"""Collect every run into results/summary.csv, aggregate over seeds, and write the E1 tables and figures.

    PYTHONPATH=. python -m experiments.gl_ident.summarize

Never re-scores: reads metrics.json / metrics_untrained.json / config.yaml from results/runs/*. Each figure
ships a .csv twin with the numbers it plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from experiments.gl_ident.config import load_config  # noqa: E402
from experiments.gl_ident.evaluate import headline  # noqa: E402

RESULTS = Path(__file__).parent / "results"
PLOTS = RESULTS / "plots"
CONDITION = [
    "name",
    "kind",
    "trained",
    "snr",
    "noise",
    "lambda_nu",
    "n_g",
    "d_hat",
    "sphere_extra_dim",
    "window_offsets",
    "objective",
    "padding",
]

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)


def collect() -> pd.DataFrame:
    rows = []
    for path in sorted((RESULTS / "runs").glob("*/metrics.json")):
        run = path.parent
        cfg = load_config(run / "config.yaml")
        base = dict(
            run=run.name,
            name=cfg.name,
            kind=cfg.model.kind,
            seed=cfg.train.seed,
            snr=cfg.generator.snr,
            noise=cfg.generator.noise,
            lambda_nu=cfg.generator.lambda_nu,
            n_g=cfg.model.n_g,
            d_hat=cfg.model.d_hat or 0,
            sphere_extra_dim=cfg.model.sphere_extra_dim,
            window_offsets=cfg.model.window_offsets if cfg.model.kind == "global" else "",
            objective=cfg.loss.objective,
            padding=cfg.model.padding,
        )
        metrics = json.loads(path.read_text())
        info = metrics.get("info", {})
        rows.append(
            {
                **base,
                "trained": True,
                **headline(metrics, cfg.model.kind),
                "best_step": info.get("best_step"),
                "best_val_loss": info.get("best_val_loss"),
                "receptive_field_sites": info.get("receptive_field_sites"),
            }
        )
        untrained = run / "metrics_untrained.json"
        if untrained.exists():
            rows.append({**base, "trained": False, **headline(json.loads(untrained.read_text()), cfg.model.kind)})
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in df.columns if c not in CONDITION + ["run", "seed"] and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(CONDITION, dropna=False)[metrics]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).join(g.size().rename("n_seeds"))
    return out.reset_index()


def fmt(series: pd.Series, digits=3) -> str:
    vals = series.dropna()
    if len(vals) == 0:
        return "-"
    if len(vals) == 1:
        return f"{vals.iloc[0]:.{digits}f} (n=1)"
    return f"{vals.mean():.{digits}f} ± {vals.std(ddof=1):.{digits}f} (n={len(vals)})"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9)


def e1_local_figure(df: pd.DataFrame):
    local = df[df.kind == "local"]
    R = int(local.R.dropna().iloc[0])
    radii = sorted(
        int(c[4:-4]) for c in local.columns if c.startswith("M5_r") and c.endswith("_mlp") and c[4:-4].isdigit()
    )
    entities = [
        ("trained (E1, 3 seeds)", local[(local.name == "e1") & local.trained]),
        ("untrained floor (seed 0)", local[(local.name == "e1") & ~local.trained]),
        ("literal code size, d_hat = 8 on the sphere (seed 0)", local[(local.name == "e1lit") & local.trained]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    twin = []
    for ax, metric, title in (
        (axes[0], "M5", "M5: R²( ĉ(u) ← [c|B_r(u), γ] )"),
        (axes[1], "M7", "M7: R²( c(u) ← ĉ|B_r(u) )"),
    ):
        _style(ax)
        for colour, (label, rows) in zip(SERIES, entities):
            if rows.empty:
                continue
            for reg, ls in (("mlp", "-"), ("ridge", "--")):
                cols = [f"{metric}_r{r}_{reg}" for r in radii]
                mean, sd = rows[cols].mean().values, rows[cols].std(ddof=1).fillna(0).values
                ax.errorbar(
                    radii,
                    mean,
                    yerr=sd if len(rows) > 1 else None,
                    color=colour,
                    ls=ls,
                    lw=2,
                    marker="o",
                    ms=5,
                    capsize=3,
                    label=f"{label}, {reg.upper()}",
                )
                twin += [
                    dict(metric=metric, entity=label, regressor=reg, r=r, mean=m, sd=s, n=len(rows))
                    for r, m, s in zip(radii, mean, sd)
                ]
        ax.axvline(R, color=MUTED, ls=":", lw=1.5)
        ax.text(R + 0.05, 0.02, f"R = {R}", color=INK2, fontsize=9, transform=ax.get_xaxis_transform())
        ax.set_xticks(radii)
        ax.set_xlabel("neighbourhood radius r (coarse sites)", color=INK2, fontsize=9)
        ax.set_ylabel("R² (test half)", color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    lo = min(0.0, min(t["mean"] for t in twin) - 0.05)
    for ax in axes:
        ax.set_ylim(lo, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(PLOTS / "e1_local_curves.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    pd.DataFrame(twin).to_csv(PLOTS / "e1_local_curves.csv", index=False)


def e1_global_figure(df: pd.DataFrame):
    rows = df[df.kind.isin(["global", "pooled"])]
    entities = [
        ("window g (disjoint), trained", rows[(rows.name == "e1") & (rows.kind == "global") & rows.trained], 0, "o"),
        ("window g, untrained floor", rows[(rows.name == "e1") & (rows.kind == "global") & ~rows.trained], 0, "x"),
        ("window g, literal code size", rows[(rows.name == "e1lit") & (rows.kind == "global") & rows.trained], 2, "s"),
        ("pooled g_pool, trained", rows[(rows.name == "e1") & (rows.kind == "pooled") & rows.trained], 1, "o"),
        ("pooled g_pool, untrained floor", rows[(rows.name == "e1") & (rows.kind == "pooled") & ~rows.trained], 1, "x"),
    ]
    panels = [
        ("M1_mlp", "M1  R²(γ ← ĝ)"),
        ("M2_pure_mlp", "M2  R²(ĝ ← γ)"),
        ("M2_gap_mlp", "M2 purity gap"),
        ("M2_resid_share_mlp", "residual leak share (added)"),
        ("M3_mlp", "M3  cross-view R²"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 3.6), sharey=True, facecolor=SURFACE)
    twin = []
    for ax, (col, title) in zip(axes, panels):
        _style(ax)
        for y, (label, sub, slot, marker) in enumerate(entities):
            if sub.empty:
                continue
            vals = sub[col].dropna()
            mean, sd = vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0.0)
            ax.errorbar(
                mean,
                y,
                xerr=sd or None,
                color=SERIES[slot],
                marker=marker,
                ms=8,
                lw=2,
                capsize=3,
                mfc=SERIES[slot] if marker != "x" else "none",
            )
            twin.append(dict(metric=col, entity=label, mean=mean, sd=sd, n=len(vals)))
        ax.axvline(0, color="#c3c2b7", lw=1)
        values = [t["mean"] for t in twin if t["metric"] == col]
        if col in ("M1_mlp", "M2_pure_mlp", "M3_mlp"):
            ax.set_xlim(min(0.0, min(values)) - 0.05, 1.05)
        else:
            ax.set_xlim(min(-0.05, min(values) - 0.03), max(0.1, max(values) + 0.03))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("MLP regressor", color=MUTED, fontsize=8)
    axes[0].set_yticks(range(len(entities)))
    axes[0].set_yticklabels([e[0] for e in entities], color=INK2, fontsize=9)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(PLOTS / "e1_global.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    pd.DataFrame(twin).to_csv(PLOTS / "e1_global.csv", index=False)


def e1_hypotheses(df: pd.DataFrame) -> list[str]:
    """Spec section 7 criteria that E1 can decide, per seed and on the seed mean (MLP regressor = primary)."""
    lines = [
        "## Hypotheses decidable from E1 (MLP primary)",
        "",
        "| ID | criterion | per seed | mean ± sd | verdict |",
        "|---|---|---|---|---|",
    ]

    def row(hid, criterion, values, ok):
        per_seed = ", ".join(f"{v:.3f}" for v in values)
        verdict = "pass" if ok is True else "fail" if ok is False else "exploratory"
        lines.append(f"| {hid} | {criterion} | {per_seed} | {fmt(pd.Series(values))} | {verdict} |")

    g = df[(df.name == "e1") & (df.kind == "global") & df.trained].sort_values("seed")
    p = df[(df.name == "e1") & (df.kind == "pooled") & df.trained].sort_values("seed")
    loc = df[(df.name == "e1") & (df.kind == "local") & df.trained].sort_values("seed")
    if not g.empty:
        h1 = (g.M1_mlp >= 0.9) & (g.M2_pure_mlp >= 0.9) & (g.M2_gap_mlp <= 0.05)
        row("H1", "M1 >= 0.9", g.M1_mlp.tolist(), None)
        row("H1", "R²(ĝ ← γ) >= 0.9", g.M2_pure_mlp.tolist(), None)
        row("H1", "purity gap <= 0.05", g.M2_gap_mlp.tolist(), None)
        row("H1", "all three, every seed", h1.astype(float).tolist(), bool(h1.all()))
    if not g.empty and not p.empty:
        row("H2", "pooled purity gap (compare with H1's)", p.M2_gap_mlp.tolist(), None)
        row("H2", "pooled residual leak share (added)", p.M2_resid_share_mlp.tolist(), None)
        row("H2", "window residual leak share (added)", g.M2_resid_share_mlp.tolist(), None)
    if not loc.empty:
        R = int(loc.R.iloc[0])
        r_cols = [c for c in loc.columns if c.startswith("M5_r") and c.endswith("_mlp") and c[4:-4].isdigit()]
        radii = sorted(int(c[4:-4]) for c in r_cols)
        m5 = loc[[f"M5_r{r}_mlp" for r in radii if r <= R]].max(axis=1)
        m7 = loc[[f"M7_r{r}_mlp" for r in radii if r <= R + 2]].max(axis=1)
        row("H4", f"max over r <= {R} of M5 >= 0.9", m5.tolist(), bool((m5 >= 0.9).all()))
        row("H5", "M6 <= 0.05", loc.M6_mlp.tolist(), bool((loc.M6_mlp <= 0.05).all()))
        row("H6", f"max over r <= {R + 2} of M7 >= 0.9", m7.tolist(), bool((m7 >= 0.9).all()))
        row("H7", f"M8: M5(r={R}) - M5(r=0)", (loc[f"M5_r{R}_mlp"] - loc.M5_r0_mlp).tolist(), None)
        row("H8 (SNR = inf part)", "M9 mean C_ii", loc.M9_mean.tolist(), None)
        row("H10 (lambda = 0 part)", "M10", loc.M10_mlp.tolist(), None)
    if not g.empty:
        row("H10 (lambda = 0 part)", "M4, window ĝ", g.M4_mlp.tolist(), None)
    if not p.empty:
        row("H10 (lambda = 0 part)", "M4, pooled ĝ", p.M4_mlp.tolist(), None)
    return lines + [""]


def e1_tables(df: pd.DataFrame) -> str:
    """Markdown tables of every E1 number the report quotes (mean ± sd over seeds)."""
    lines = ["# E1 tables (generated by summarize.py; do not edit)", ""]
    glob = df[df.kind.isin(["global", "pooled"])]
    cols = ["M1", "M2_pure", "M2_full", "M2_gap", "M2_resid_share", "M3", "M4"]
    for reg in ("mlp", "ridge"):
        lines += [
            f"## Global metrics, {reg.upper()} regressor",
            "",
            "| model | " + " | ".join(cols) + " | C_ii mean | C_ii min |",
            "|---" * (len(cols) + 3) + "|",
        ]
        for (name, kind, trained), sub in glob.groupby(["name", "kind", "trained"], sort=False):
            cells = [fmt(sub[f"{c}_{reg}"]) for c in cols] + [fmt(sub.C_ii_global_mean), fmt(sub.C_ii_global_min)]
            lines.append(f"| {name} {kind} {'trained' if trained else 'untrained'} | " + " | ".join(cells) + " |")
        lines.append("")
    local = df[df.kind == "local"]
    if not local.empty:
        R = int(local.R.dropna().iloc[0])
        radii = sorted(
            int(c[4:-4]) for c in local.columns if c.startswith("M5_r") and c.endswith("_mlp") and c[4:-4].isdigit()
        )
        for metric in ("M5", "M7"):
            for reg in ("mlp", "ridge"):
                lines += [
                    f"## {metric} by radius, {reg.upper()} (R = {R})",
                    "",
                    "| model | " + " | ".join(f"r={r}" for r in radii) + " |",
                    "|---" * (len(radii) + 1) + "|",
                ]
                for (name, trained), sub in local.groupby(["name", "trained"], sort=False):
                    cells = [fmt(sub[f"{metric}_r{r}_{reg}"]) for r in radii]
                    lines.append(f"| {name} {'trained' if trained else 'untrained'} | " + " | ".join(cells) + " |")
                lines.append("")
        lines += [
            "## Other local metrics",
            "",
            "| model | M6 MLP | M6 ridge | M8 MLP (M5 r=R minus r=0) | M9 mean | M9 min | M10 MLP | M10 ridge |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for (name, trained), sub in local.groupby(["name", "trained"], sort=False):
            m8 = sub[f"M5_r{R}_mlp"] - sub["M5_r0_mlp"]
            cells = [
                fmt(sub.M6_mlp),
                fmt(sub.M6_ridge),
                fmt(m8),
                fmt(sub.M9_mean),
                fmt(sub.M9_min),
                fmt(sub.M10_mlp),
                fmt(sub.M10_ridge),
            ]
            lines.append(f"| {name} {'trained' if trained else 'untrained'} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += e1_hypotheses(df)
    runs = df[df.trained]
    lines += ["## Training", "", "| run | best step | best val loss | receptive field (sites) |", "|---|---|---|---|"]
    for _, r in runs.iterrows():
        rf = "" if pd.isna(r.get("receptive_field_sites")) else int(r.receptive_field_sites)
        lines.append(f"| {r.run} | {int(r.best_step)} | {r.best_val_loss:.4f} | {rf} |")
    return "\n".join(lines) + "\n"


def export_splits(cfg_path=Path(__file__).parent / "configs" / "e1.yaml"):
    """Store the fixed val/test latents (common to every condition) and a digest of the E1 test images."""
    import hashlib

    import torch

    from experiments.gl_ident.generator import SyntheticGenerator

    out = RESULTS / "data"
    if (out / "test_latents.pt").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    gen = SyntheticGenerator(load_config(cfg_path).generator)
    digests = {}
    for split in ("val", "test"):
        lat = gen.split(split)
        torch.save({k: v.cpu() for k, v in lat.items()}, out / f"{split}_latents.pt")
        x1, x2 = gen.split_images(split, lat)
        digests[split] = hashlib.sha256(torch.cat([x1, x2]).numpy().tobytes()).hexdigest()
    (out / "e1_image_sha256.json").write_text(json.dumps(digests, indent=1))


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    export_splits()
    df = collect()
    df.to_csv(RESULTS / "summary.csv", index=False)
    aggregate(df).to_csv(RESULTS / "summary_by_condition.csv", index=False)
    e1 = df[df.name.isin(["e1", "e1lit"])]
    if not e1.empty:
        (RESULTS / "e1_tables.md").write_text(e1_tables(e1))
        if (e1.kind == "local").any():
            e1_local_figure(e1)
        if e1.kind.isin(["global", "pooled"]).any():
            e1_global_figure(e1)
    print(f"{len(df)} rows -> {RESULTS / 'summary.csv'}")


if __name__ == "__main__":
    main()
