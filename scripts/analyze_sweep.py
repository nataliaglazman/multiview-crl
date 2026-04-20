"""Analyze a W&B sweep and identify the best configurations for content/style separation.

Usage:
    python scripts/analyze_sweep.py --sweep-id <entity/project/sweep_id> [--top-k 5]
"""

import argparse

import pandas as pd
import wandb

SWEEP_PARAMS = [
    "scale_contrastive_loss",
    "scale_recon_loss",
    "vqvae_hidden_channels",
    "content_size",
    "mask_mode",
    "contrastive_loss_type",
    "tau",
    "lr",
    "patch_contrastive",
    "bt_lambda",
]

METRICS = [
    "separation_score",
    "content/modality_invariance",
    "content/modality_probe_acc",
    "content/cross_view_cosine_mean",
    "style/subject_invariance",
    "style/subject_retrieval_top1",
    "style/subject_retrieval_mean_rank",
    "style/modality_probe_acc",
    "style/intra_view_cosine_mean_v0",
    "style/cross_view_cosine_mean",
]


def fetch_sweep_runs(sweep_id: str) -> pd.DataFrame:
    """Fetch all completed runs from a W&B sweep."""
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    rows = []
    for run in sweep.runs:
        if run.state != "finished":
            continue
        row = {"run_id": run.id, "run_name": run.name}
        for p in SWEEP_PARAMS:
            row[p] = run.config.get(p)
        for m in METRICS:
            row[m] = run.summary.get(m)
        # Also grab final training losses
        row["final_loss_total"] = run.summary.get("loss/total")
        row["final_loss_contrastive"] = run.summary.get("loss/contrastive")
        row["final_loss_recon"] = run.summary.get("loss/recon")
        rows.append(row)
    return pd.DataFrame(rows)


def print_top_configs(df: pd.DataFrame, top_k: int = 5):
    """Print the top-K configurations ranked by separation_score."""
    df_valid = df.dropna(subset=["separation_score"]).sort_values("separation_score", ascending=False)

    print(f"\n{'='*80}")
    print(f"TOP {top_k} CONFIGURATIONS BY SEPARATION SCORE")
    print(f"{'='*80}\n")

    for rank, (_, row) in enumerate(df_valid.head(top_k).iterrows(), 1):
        print(f"--- Rank {rank} (run: {row['run_name']}) ---")
        print(f"  Separation score:        {row['separation_score']:.4f}")
        print(f"  Content modality inv.:   {row.get('content/modality_invariance', 'N/A')}")
        print(f"  Style subject inv.:      {row.get('style/subject_invariance', 'N/A')}")
        print(f"  Style modality probe:    {row.get('style/modality_probe_acc', 'N/A')}")
        print(f"  Content cross-view cos:  {row.get('content/cross_view_cosine_mean', 'N/A')}")
        print(f"  Hyperparameters:")
        for p in SWEEP_PARAMS:
            val = row.get(p)
            if val is not None:
                print(f"    {p}: {val}")
        print()

    return df_valid


def print_param_analysis(df: pd.DataFrame):
    """Print correlation between each parameter and the separation score."""
    df_valid = df.dropna(subset=["separation_score"])

    print(f"\n{'='*80}")
    print("PARAMETER IMPACT ANALYSIS")
    print(f"{'='*80}\n")

    # Categorical parameters: show mean separation_score per value
    for p in SWEEP_PARAMS:
        if df_valid[p].dtype == object or df_valid[p].nunique() <= 5:
            grouped = df_valid.groupby(p)["separation_score"].agg(["mean", "std", "count"])
            print(f"\n{p}:")
            for val, row in grouped.iterrows():
                print(f"  {val:>20s}: {row['mean']:.4f} +/- {row['std']:.4f}  (n={row['count']:.0f})")
        else:
            corr = df_valid[[p, "separation_score"]].corr().iloc[0, 1]
            print(f"\n{p}: correlation with separation_score = {corr:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze W&B sweep results")
    parser.add_argument("--sweep-id", type=str, required=True, help="W&B sweep ID (entity/project/sweep_id)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top configs to show")
    parser.add_argument("--output", type=str, default=None, help="Save full results to CSV")
    args = parser.parse_args()

    print(f"Fetching runs from sweep: {args.sweep_id}")
    df = fetch_sweep_runs(args.sweep_id)
    print(f"Found {len(df)} completed runs")

    if len(df) == 0:
        print("No completed runs found. Check the sweep ID and try again.")
        return

    df_ranked = print_top_configs(df, args.top_k)
    print_param_analysis(df)

    if args.output:
        df_ranked.to_csv(args.output, index=False)
        print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
