"""Apply the pre-registered two-sided elbow rule to a content-dimensionality sweep.

Companion to ``scripts/sweep_content_dim_synthetic.sh``. Pulls runs from W&B by
group (one group per regime×arm cell, e.g. ``cdim-causal-deploy``), aggregates by
``content_size`` across seeds, and reports the elbow.

Pre-registered rule:
    k* = min content_size such that, on the mean over seeds:
        content->content/informativeness_test  >= complete_frac * max   (lower fence: completeness)
        content->view/acc                       <= view_ceiling          (upper fence: style leakage)
    Headline identifiability number at k*: content->content/block_mcc.

The lower fence is a *completeness* signal (does the content block recover all
recoverable shared factors). NB: do NOT use block_mcc for the lower fence — its
Hungarian match averages over min(k, n_factors) pairs, so it reads high even at
small k by greedily picking the few easiest factors. block_mcc is the headline
*subspace-quality* number at the chosen k, not the rising/elbow signal.

Example:
    python scripts/analyze_content_dim.py \\
        --wandb-project multiview-crl-content-dim --group cdim-causal-deploy --plot
"""

import argparse
import math
import sys

import pandas as pd

try:
    import wandb
except ImportError:
    sys.exit("wandb is required. pip install wandb")


def _metric_keys(level: int | None):
    """W&B summary keys for the decision metrics (single-level → no L{n}/ prefix)."""
    p = f"L{level}/" if level is not None else ""
    base = "dci_synthetic/"
    return {
        "complete": f"{base}{p}content->content/informativeness_test",
        "view": f"{base}{p}content->view/acc",
        "leak_style": f"{base}{p}content->style/informativeness_test",
        "block_mcc": f"{base}{p}content->content/block_mcc",
    }


def fetch_runs(project: str, group: str, entity: str | None) -> pd.DataFrame:
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"group": group, "state": "finished"})
    rows = []
    for run in runs:
        rows.append({"run_name": run.name, "_config": dict(run.config), "_summary": dict(run.summary)})
    return pd.DataFrame(rows)


def extract_metrics(df: pd.DataFrame, keys: dict) -> pd.DataFrame:
    out = []
    for _, row in df.iterrows():
        cfg, summ = row["_config"], row["_summary"]
        cap = cfg.get("content_size")
        if cap is None:
            continue
        out.append(
            {
                "run_name": row["run_name"],
                "content_size": int(cap),
                "seed": cfg.get("seed", -1),
                "complete": summ.get(keys["complete"]),
                "view": summ.get(keys["view"]),
                "leak_style": summ.get(keys["leak_style"]),
                "block_mcc": summ.get(keys["block_mcc"]),
            }
        )
    return pd.DataFrame(out)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("content_size").agg(
        n_seeds=("seed", "count"),
        complete_mean=("complete", "mean"),
        complete_std=("complete", "std"),
        view_mean=("view", "mean"),
        view_std=("view", "std"),
        leak_style_mean=("leak_style", "mean"),
        block_mcc_mean=("block_mcc", "mean"),
        block_mcc_std=("block_mcc", "std"),
    )
    return g.sort_index()


def apply_decision_rule(agg: pd.DataFrame, complete_frac: float, view_ceiling: float, noise_threshold: float) -> dict:
    if agg.empty or agg["complete_mean"].isna().all():
        return {"winner": None, "reason": "completeness metric missing from all runs", "detail": agg}

    max_complete = agg["complete_mean"].max()
    complete_ok = agg["complete_mean"] >= complete_frac * max_complete
    # If the view probe is unavailable, the upper fence can't be applied — say so.
    if agg["view_mean"].isna().all():
        view_ok = pd.Series(True, index=agg.index)
        view_note = "(content->view unavailable — upper fence NOT applied)"
    else:
        view_ok = agg["view_mean"] <= view_ceiling
        view_note = f"(view ceiling = {view_ceiling})"

    valid = complete_ok & view_ok
    valid_caps = agg.index[valid].tolist()
    winner = min(valid_caps) if valid_caps else None

    noise_warning = None
    if winner is not None:
        std_here = agg.loc[winner, "complete_std"]
        if std_here is not None and not math.isnan(std_here) and std_here > noise_threshold:
            noise_warning = f"std(completeness) at k*={std_here:.3f} > {noise_threshold} — add more seeds"

    return {
        "winner": winner,
        "valid_capacities": valid_caps,
        "complete_ok": complete_ok,
        "view_ok": view_ok,
        "max_complete": max_complete,
        "view_note": view_note,
        "noise_warning": noise_warning,
        "detail": agg,
    }


def print_result(res: dict, group: str) -> None:
    print("=" * 76)
    print(f"Content-dimensionality sweep — group: {group}")
    print("=" * 76)
    agg = res["detail"]
    cols = ["n_seeds", "complete_mean", "complete_std", "view_mean", "leak_style_mean", "block_mcc_mean"]
    print("\nPer-capacity aggregates (mean ± std across seeds):\n")
    print(agg[cols].round(4).to_string())
    print()
    print("Decision-rule masks (lower fence = completeness, upper fence = view leakage):")
    for cap in agg.index:
        c = "Y" if res["complete_ok"].get(cap, False) else "N"
        v = "Y" if res["view_ok"].get(cap, False) else "N"
        print(f"  k={cap:>4}: completeness={c}  view-invariant={v}")
    print()
    print(res.get("view_note", ""))
    if res["winner"] is None:
        print("*** NO content_size satisfies both fences. Check the sweep before trusting an elbow. ***")
    else:
        k = res["winner"]
        mcc = agg.loc[k, "block_mcc_mean"]
        print(f"*** k* = {k}  (smallest content_size inside both fences) ***")
        print(f"    headline block-MCC(content→z_content) at k* = {mcc:.3f}")
        if res.get("noise_warning"):
            print(f"    WARNING: {res['noise_warning']}")
    print()


def maybe_plot(agg: pd.DataFrame, winner, group: str, path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping --plot", file=sys.stderr)
        return

    k = agg.index.values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.errorbar(
        k, agg["complete_mean"], yerr=agg["complete_std"], marker="o", label="completeness (content→content I)"
    )
    ax1.plot(k, agg["block_mcc_mean"], marker="s", color="seagreen", label="block-MCC (content→z_content)")
    ax1.set_xlabel("content_size k")
    ax1.set_ylabel("score")
    ax1.set_title("Lower fence: content recovery")
    ax1.legend(fontsize=8)
    ax2.errorbar(k, agg["view_mean"], yerr=agg["view_std"], marker="o", color="firebrick", label="content→view acc")
    ax2.axhline(0.5, color="k", lw=0.8, ls="--", label="chance")
    ax2.set_xlabel("content_size k")
    ax2.set_ylabel("accuracy")
    ax2.set_title("Upper fence: style leakage")
    ax2.legend(fontsize=8)
    if winner is not None:
        for ax in (ax1, ax2):
            ax.axvline(winner, color="goldenrod", lw=1.5, ls=":")
    fig.suptitle(group)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Saved plot to {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wandb-project", required=True)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--group", required=True, help="W&B group, e.g. cdim-causal-deploy")
    p.add_argument("--level", type=int, default=None, help="Encoder level (omit for single-level synthetic runs)")
    p.add_argument("--complete-frac", type=float, default=0.9, help="Lower fence: fraction of max completeness")
    p.add_argument("--view-ceiling", type=float, default=0.55, help="Upper fence: max content→view accuracy")
    p.add_argument("--noise-threshold", type=float, default=0.05)
    p.add_argument("--output-csv", default=None)
    p.add_argument("--plot", default=None, nargs="?", const="content_dim_sweep.png", help="Save a PNG (optional path)")
    args = p.parse_args()

    runs_df = fetch_runs(args.wandb_project, args.group, args.wandb_entity)
    if runs_df.empty:
        sys.exit(f"No finished runs in project={args.wandb_project} group={args.group}.")
    print(f"Fetched {len(runs_df)} finished runs.")

    metrics = extract_metrics(runs_df, _metric_keys(args.level))
    if metrics.empty:
        sys.exit("No runs carry a content_size config value. Check the sweep.")
    agg = aggregate(metrics)
    res = apply_decision_rule(agg, args.complete_frac, args.view_ceiling, args.noise_threshold)
    print_result(res, args.group)

    if args.output_csv:
        agg.to_csv(args.output_csv)
        print(f"Saved aggregate table to {args.output_csv}")
    if args.plot:
        maybe_plot(agg, res["winner"], args.group, args.plot)


if __name__ == "__main__":
    main()
