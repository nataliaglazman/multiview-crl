"""Apply the pre-registered decision rule to a content/style capacity sweep.

Pulls runs from W&B by group (or tag prefix), aggregates per capacity value
across seeds, applies the three-mask decision rule, and prints the winner.

Pre-registered rule (see scripts/phase1_L0_sweep.sh):
    C* = min C such that, on the mean over seeds,
        diag_info_L{lvl}                   >= 0.9 * max(diag_info_L{lvl})
        content/modality_probe_acc_L{lvl}  <= 0.55
        val/recon                          <= min(val/recon) * (1 + recon_tolerance)

    (Reconstruction is gated on val/recon — lower is better — because the
    training loop does not log PSNR. Use --recon-tolerance to tune slack.)

Example:
    python scripts/analyze_capacity_sweep.py \\
        --wandb-project multiview-crl-capacity --group phase1-L0 \\
        --param content_ratios_L0 --level 0
"""

import argparse
import math
import sys

import pandas as pd

try:
    import wandb
except ImportError:
    sys.exit("wandb is required. pip install wandb")


def fetch_runs(project: str, group: str, entity: str | None) -> pd.DataFrame:
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"group": group, "state": "finished"})
    rows = []
    for run in runs:
        cfg = dict(run.config)
        summary = dict(run.summary)
        # Flatten ``content_ratios`` list → per-level scalars for grouping.
        cr = cfg.get("content_ratios")
        if isinstance(cr, (list, tuple)):
            for i, v in enumerate(cr):
                cfg[f"content_ratios_L{i}"] = float(v)
        rows.append({"run_id": run.id, "run_name": run.name, "_config": cfg, "_summary": summary})
    return pd.DataFrame(rows)


def extract_metrics(df: pd.DataFrame, param: str, level: int) -> pd.DataFrame:
    """Pull the capacity parameter and the three decision metrics + separation + seed."""
    diag_key = f"content/diagnosis_info_L{level}"
    mod_key = f"content/modality_probe_acc_L{level}"
    out = []
    for _, row in df.iterrows():
        cfg = row["_config"]
        summ = row["_summary"]
        c_val = cfg.get(param)
        if c_val is None:
            continue
        out.append(
            {
                "run_name": row["run_name"],
                "capacity": float(c_val),
                "seed": cfg.get("seed", -1),
                "diag_info": summ.get(diag_key),
                "modality_acc": summ.get(mod_key),
                "val_recon": summ.get("val/recon"),
                "separation_gated": summ.get("separation_score_gated"),
                "separation_raw": summ.get("separation_score"),
            }
        )
    return pd.DataFrame(out)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std across seeds at each capacity."""
    g = df.groupby("capacity").agg(
        n_seeds=("seed", "count"),
        diag_info_mean=("diag_info", "mean"),
        diag_info_std=("diag_info", "std"),
        modality_acc_mean=("modality_acc", "mean"),
        modality_acc_std=("modality_acc", "std"),
        val_recon_mean=("val_recon", "mean"),
        val_recon_std=("val_recon", "std"),
        sep_gated_mean=("separation_gated", "mean"),
        sep_raw_mean=("separation_raw", "mean"),
    )
    return g.sort_index()


def apply_decision_rule(
    agg: pd.DataFrame,
    diag_fraction: float = 0.9,
    modality_ceiling: float = 0.55,
    recon_tolerance: float = 0.1,
    noise_threshold: float = 0.05,
) -> dict:
    """Return the chosen capacity plus the mask that produced it."""
    if agg.empty:
        return {"winner": None, "reason": "no runs", "detail": agg}

    # Handle missing metrics defensively: if any mask is all-NaN, the rule
    # can't be applied — report that instead of picking a winner by accident.
    if agg["diag_info_mean"].isna().all():
        return {"winner": None, "reason": "diag_info missing from all runs", "detail": agg}

    max_diag = agg["diag_info_mean"].max()
    min_recon = agg["val_recon_mean"].min() if not agg["val_recon_mean"].isna().all() else None

    anatomy_ok = agg["diag_info_mean"] >= diag_fraction * max_diag
    invariance_ok = agg["modality_acc_mean"] <= modality_ceiling
    if min_recon is None or math.isnan(min_recon):
        recon_ok = pd.Series(True, index=agg.index)  # val/recon unavailable — skip gate
        recon_note = "(val/recon unavailable — not gated on reconstruction)"
    else:
        recon_ok = agg["val_recon_mean"] <= min_recon * (1.0 + recon_tolerance)
        recon_note = f"(min val/recon={min_recon:.4f}, tolerance={recon_tolerance:.0%})"

    valid = anatomy_ok & invariance_ok & recon_ok
    valid_capacities = agg.index[valid].tolist()
    winner = min(valid_capacities) if valid_capacities else None

    # Noise check at the winner.
    noise_warning = None
    if winner is not None:
        std_here = agg.loc[winner, "diag_info_std"]
        if std_here is not None and not math.isnan(std_here) and std_here > noise_threshold:
            noise_warning = f"std(diag_info) at C*={std_here:.3f} > {noise_threshold} — add more seeds"

    return {
        "winner": winner,
        "valid_capacities": valid_capacities,
        "anatomy_ok": anatomy_ok,
        "invariance_ok": invariance_ok,
        "recon_ok": recon_ok,
        "max_diag": max_diag,
        "recon_note": recon_note,
        "noise_warning": noise_warning,
        "detail": agg,
    }


def print_result(res: dict, param: str, level: int) -> None:
    print("=" * 72)
    print(f"Capacity sweep analysis  — parameter: {param}  level: L{level}")
    print("=" * 72)
    agg = res["detail"]
    print("\nPer-capacity aggregates (mean ± std across seeds):\n")
    cols = [
        "n_seeds",
        "diag_info_mean",
        "diag_info_std",
        "modality_acc_mean",
        "psnr_mean",
        "sep_gated_mean",
    ]
    print(agg[cols].round(4).to_string())
    print()
    print("Decision-rule masks:")
    for cap in agg.index:
        flags = (
            f"anatomy={'Y' if res['anatomy_ok'].get(cap, False) else 'N'}  "
            f"invariance={'Y' if res['invariance_ok'].get(cap, False) else 'N'}  "
            f"recon={'Y' if res['recon_ok'].get(cap, False) else 'N'}"
        )
        print(f"  C={cap:>6}: {flags}")
    print()
    print(res.get("psnr_note", ""))
    if res["winner"] is None:
        print("*** NO capacity satisfies all three rules. Fix the pipeline before sweeping further. ***")
    else:
        print(f"*** C* = {res['winner']}  (smallest capacity satisfying all three rules) ***")
        if res.get("noise_warning"):
            print(f"    WARNING: {res['noise_warning']}")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wandb-project", required=True)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--group", required=True, help="W&B group used by the sweep script")
    p.add_argument("--param", default="content_ratios_L0", help="Config key that encodes the capacity axis")
    p.add_argument("--level", type=int, default=0, help="Level whose metrics to evaluate (L0, L1, L2)")
    p.add_argument("--diag-fraction", type=float, default=0.9)
    p.add_argument("--modality-ceiling", type=float, default=0.55)
    p.add_argument("--psnr-tolerance-db", type=float, default=1.0)
    p.add_argument("--noise-threshold", type=float, default=0.05)
    p.add_argument("--output-csv", default=None)
    args = p.parse_args()

    runs_df = fetch_runs(args.wandb_project, args.group, args.wandb_entity)
    if runs_df.empty:
        sys.exit(f"No finished runs found in project={args.wandb_project} group={args.group}.")
    print(f"Fetched {len(runs_df)} finished runs.")

    metrics = extract_metrics(runs_df, args.param, args.level)
    if metrics.empty:
        sys.exit(f"No runs carry config key {args.param!r}. Check --param.")
    agg = aggregate(metrics)
    result = apply_decision_rule(
        agg,
        diag_fraction=args.diag_fraction,
        modality_ceiling=args.modality_ceiling,
        psnr_tolerance_db=args.psnr_tolerance_db,
        noise_threshold=args.noise_threshold,
    )
    print_result(result, args.param, args.level)

    if args.output_csv:
        agg.to_csv(args.output_csv)
        print(f"Saved aggregate table to {args.output_csv}")


if __name__ == "__main__":
    main()
