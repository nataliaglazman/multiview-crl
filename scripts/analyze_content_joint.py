"""Joint content-size sweep analysis: per-level marginal effects via OLS.

Pulls runs from the joint sweep (3D content-size space + seeds), fits
per-level OLS regressions of the form

    probe_metric_L{l} ~ C_L0 + C_L1 + C_L2 + seed_dummies

and reports each level's marginal effect, partial-regression scatter, and the
pre-registered decision rule applied per level.

Decision rule (matches analyze_capacity_sweep.py, applied to the *fitted*
marginals so each per-level conclusion uses information from all runs):
    For each level l, find the smallest C_L{l} such that, holding other
    levels at their mean,
        diag_info_L{l}        >= 0.9 * max(predicted diag_info)
        modality_probe_L{l}   <= 0.55
        val/recon             <= min(val/recon) * (1 + recon_tolerance)

Usage:
    python scripts/analyze_content_joint.py \\
        --wandb-project multiview-crl-content-joint \\
        --sweep-id <id> \\
        [--entity natalia] [--out-dir analysis/content_joint]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import wandb
except ImportError:
    sys.exit("wandb is required. pip install wandb")

try:
    import statsmodels.api as sm
except ImportError:
    sys.exit("statsmodels is required. pip install statsmodels")


HIDDEN_CHANNELS = 32  # locked in sweep_content_joint.yaml


def fetch_runs(project: str, sweep_id: str, entity: str | None) -> pd.DataFrame:
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    sweep_path = f"{path}/{sweep_id}" if "/" not in sweep_id else sweep_id
    sweep = api.sweep(sweep_path)
    rows = []
    for run in sweep.runs:
        if run.state != "finished":
            continue
        cfg = dict(run.config)
        summary = dict(run.summary)
        cr = cfg.get("content_ratios")
        if not isinstance(cr, (list, tuple)) or len(cr) != 3:
            continue
        row = {
            "run_id": run.id,
            "run_name": run.name,
            "seed": int(cfg.get("seed", -1)),
            "C_L0": float(cr[0]) * HIDDEN_CHANNELS,
            "C_L1": float(cr[1]) * HIDDEN_CHANNELS,
            "C_L2": float(cr[2]) * HIDDEN_CHANNELS,
            "val_recon": summary.get("val/recon"),
            "separation_score": summary.get("separation_score"),
            "separation_score_gated": summary.get("separation_score_gated"),
        }
        for l in (0, 1, 2):
            row[f"diag_info_L{l}"] = summary.get(f"content/diagnosis_info_L{l}")
            row[f"modality_acc_L{l}"] = summary.get(f"content/modality_probe_acc_L{l}")
            row[f"anatomy_acc_L{l}"] = summary.get(f"content/anatomy_probe_acc_L{l}")
        rows.append(row)
    return pd.DataFrame(rows)


def fit_marginal(df: pd.DataFrame, target: str) -> sm.regression.linear_model.RegressionResultsWrapper | None:
    """OLS: target ~ C_L0 + C_L1 + C_L2 + C(seed). Returns fitted model or None."""
    sub = df.dropna(subset=[target, "C_L0", "C_L1", "C_L2", "seed"]).copy()
    if len(sub) < 6:
        return None
    X = sub[["C_L0", "C_L1", "C_L2"]].astype(float)
    seed_dummies = pd.get_dummies(sub["seed"].astype(int), prefix="seed", drop_first=True).astype(float)
    X = pd.concat([X, seed_dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = sub[target].astype(float)
    return sm.OLS(y, X).fit()


def predict_at(model, c_vals: np.ndarray, level: int, df: pd.DataFrame) -> np.ndarray:
    """Predict target as we vary C_L{level} with the other two at their means and seed=mean."""
    other_levels = [l for l in (0, 1, 2) if l != level]
    means = {f"C_L{l}": df[f"C_L{l}"].mean() for l in other_levels}
    n = len(c_vals)
    X = pd.DataFrame({"const": np.ones(n)})
    for l in (0, 1, 2):
        X[f"C_L{l}"] = c_vals if l == level else means[f"C_L{l}"]
    # seed dummies all 0 → reference seed; effect is shifted but slope is unaffected
    for col in model.params.index:
        if col.startswith("seed_"):
            X[col] = 0.0
    X = X[model.params.index]
    return model.predict(X).to_numpy()


def per_level_decision(
    df: pd.DataFrame,
    level: int,
    grid: np.ndarray,
    diag_fraction: float = 0.9,
    modality_ceiling: float = 0.55,
    recon_tolerance: float = 0.1,
) -> dict:
    diag_model = fit_marginal(df, f"diag_info_L{level}")
    mod_model = fit_marginal(df, f"modality_acc_L{level}")
    recon_model = fit_marginal(df, "val_recon")

    if diag_model is None or mod_model is None:
        return {"level": level, "winner": None, "reason": "missing metrics"}

    diag_pred = predict_at(diag_model, grid, level, df)
    mod_pred = predict_at(mod_model, grid, level, df)
    recon_pred = predict_at(recon_model, grid, level, df) if recon_model is not None else None

    anatomy_ok = diag_pred >= diag_fraction * np.nanmax(diag_pred)
    invariance_ok = mod_pred <= modality_ceiling
    if recon_pred is not None:
        recon_ok = recon_pred <= np.nanmin(recon_pred) * (1.0 + recon_tolerance)
    else:
        recon_ok = np.ones_like(grid, dtype=bool)

    valid = anatomy_ok & invariance_ok & recon_ok
    winner = float(grid[valid].min()) if valid.any() else None

    return {
        "level": level,
        "winner": winner,
        "grid": grid,
        "diag_pred": diag_pred,
        "mod_pred": mod_pred,
        "recon_pred": recon_pred,
        "anatomy_ok": anatomy_ok,
        "invariance_ok": invariance_ok,
        "recon_ok": recon_ok,
        "diag_model": diag_model,
        "mod_model": mod_model,
        "recon_model": recon_model,
    }


def print_level_result(res: dict) -> None:
    l = res["level"]
    print("=" * 72)
    print(f"Level L{l}  — marginal effect of C_L{l} (others held at mean)")
    print("=" * 72)
    if res.get("winner") is None and "reason" in res:
        print(f"SKIPPED: {res['reason']}")
        return

    diag = res["diag_model"]
    mod = res["mod_model"]
    print(f"\nOLS coefficients on C_L{l}:")
    print(
        f"  diag_info_L{l}      "
        f"β={diag.params[f'C_L{l}']:+.4f}  "
        f"SE={diag.bse[f'C_L{l}']:.4f}  "
        f"p={diag.pvalues[f'C_L{l}']:.3g}"
    )
    print(
        f"  modality_acc_L{l}   "
        f"β={mod.params[f'C_L{l}']:+.4f}  "
        f"SE={mod.bse[f'C_L{l}']:.4f}  "
        f"p={mod.pvalues[f'C_L{l}']:.3g}"
    )
    if res.get("recon_model") is not None:
        rec = res["recon_model"]
        print(
            f"  val/recon          "
            f"β={rec.params[f'C_L{l}']:+.4f}  "
            f"SE={rec.bse[f'C_L{l}']:.4f}  "
            f"p={rec.pvalues[f'C_L{l}']:.3g}"
        )

    print("\nPredicted marginals (others at mean):")
    print(f"  {'C':>6}  {'diag_info':>10}  {'mod_acc':>8}  {'val_recon':>10}  flags")
    for i, c in enumerate(res["grid"]):
        flags = (
            f"anatomy={'Y' if res['anatomy_ok'][i] else 'N'} "
            f"inv={'Y' if res['invariance_ok'][i] else 'N'} "
            f"recon={'Y' if res['recon_ok'][i] else 'N'}"
        )
        recon_v = f"{res['recon_pred'][i]:.4f}" if res["recon_pred"] is not None else "  n/a"
        print(f"  {c:>6.1f}  {res['diag_pred'][i]:>10.4f}  " f"{res['mod_pred'][i]:>8.4f}  {recon_v:>10}  {flags}")

    if res["winner"] is None:
        print(f"\n*** L{l}: NO C value satisfies all three rules along this marginal. ***")
    else:
        print(f"\n*** L{l}: C* = {res['winner']:.1f} channels (smallest passing all rules) ***")
    print()


def maybe_plot(df: pd.DataFrame, results: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex="col")
    metrics = [("diag_info", "Diagnosis info"), ("modality_acc", "Modality probe acc"), ("val_recon", "val/recon")]
    for col, res in enumerate(results):
        l = res["level"]
        x_obs = df[f"C_L{l}"]
        for row, (key, label) in enumerate(metrics):
            ax = axes[row, col]
            if key == "val_recon":
                y_obs = df["val_recon"]
                y_pred = res["recon_pred"]
            else:
                y_obs = df[f"{key}_L{l}"]
                y_pred = res["diag_pred"] if key == "diag_info" else res["mod_pred"]
            ax.scatter(x_obs, y_obs, alpha=0.5, s=20, label="runs")
            if y_pred is not None:
                ax.plot(res["grid"], y_pred, "r-", label="OLS marginal")
            if res.get("winner") is not None and row == 0:
                ax.axvline(res["winner"], color="g", linestyle="--", alpha=0.6, label=f"C*={res['winner']:.0f}")
            if row == 0:
                ax.set_title(f"Level L{l}")
            if col == 0:
                ax.set_ylabel(label)
            if row == 2:
                ax.set_xlabel(f"C_L{l} (channels)")
            ax.legend(fontsize=7)
    fig.suptitle("Joint content-size sweep — per-level marginals", y=1.0)
    fig.tight_layout()
    out_path = out_dir / "marginals.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wandb-project", required=True)
    p.add_argument("--sweep-id", required=True, help="Sweep ID (e.g. abc123) or full path")
    p.add_argument("--entity", default=None)
    p.add_argument("--out-dir", default="analysis/content_joint")
    p.add_argument("--diag-fraction", type=float, default=0.9)
    p.add_argument("--modality-ceiling", type=float, default=0.55)
    p.add_argument("--recon-tolerance", type=float, default=0.1)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    print(f"Fetching runs from sweep {args.sweep_id} in {args.wandb_project}...")
    df = fetch_runs(args.wandb_project, args.sweep_id, args.entity)
    print(f"Fetched {len(df)} finished runs.")
    if df.empty:
        sys.exit("No finished runs found — check sweep ID and entity.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "runs.csv", index=False)
    print(f"Saved runs to {out_dir / 'runs.csv'}")

    grid = np.array([4, 8, 13, 16, 20], dtype=float)
    results = []
    for l in (0, 1, 2):
        res = per_level_decision(
            df,
            level=l,
            grid=grid,
            diag_fraction=args.diag_fraction,
            modality_ceiling=args.modality_ceiling,
            recon_tolerance=args.recon_tolerance,
        )
        print_level_result(res)
        results.append(res)

    summary = pd.DataFrame([{"level": r["level"], "C_star": r.get("winner")} for r in results])
    summary.to_csv(out_dir / "winners.csv", index=False)
    print(f"\nSummary:\n{summary.to_string(index=False)}")
    print(f"Saved winners to {out_dir / 'winners.csv'}")

    if not args.no_plot:
        maybe_plot(df, results, out_dir)


if __name__ == "__main__":
    main()
