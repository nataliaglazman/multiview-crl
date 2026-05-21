#!/usr/bin/env python
"""Standalone DCI evaluation on a saved VQ-VAE checkpoint + synthetic data.

Usage:
    python -m eval.run_dci_synthetic --run-dir path/to/run_directory

    # or point directly at a checkpoint (settings.json must be in same dir):
    python -m eval.run_dci_synthetic --run-dir path/to/run_directory \
        --checkpoint path/to/vqvae_best.pt

The script reads settings.json from the run directory to reconstruct the
exact model architecture, loads the checkpoint, generates a synthetic test
set, and prints DCI scores for every content/style combination.

Examples:
    python -m eval.run_dci_synthetic --run-dir runs/my_run
    python -m eval.run_dci_synthetic --run-dir runs/my_run --num-samples 500
    python -m eval.run_dci_synthetic --run-dir runs/my_run --checkpoint runs/my_run/vqvae_best.pt
"""

import argparse
import csv
import json
import logging
import os
import sys

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _namespace_from_dict(d):
    """Convert a flat dict to an argparse.Namespace."""
    return argparse.Namespace(**d)


def main():
    parser = argparse.ArgumentParser(description="Standalone DCI evaluation for synthetic data")
    parser.add_argument("--run-dir", type=str, required=True, help="Training run directory containing settings.json")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to .pt file (default: <run-dir>/vqvae_model.pt)"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Where to write results (default: run-dir)")
    parser.add_argument("--num-samples", type=int, default=None, help="Override number of test samples")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--pooling",
        type=str,
        default="gap",
        help='Pooling mode: "gap", "stats", or a patch grid like "2,2,2"',
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="0",
        help='Comma-separated encoder levels to evaluate, e.g. "0,1,2"',
    )
    cli = parser.parse_args()

    if cli.pooling in ("gap", "stats"):
        pooling = cli.pooling
    else:
        pooling = tuple(int(x) for x in cli.pooling.split(","))
    levels = [int(x) for x in cli.levels.split(",")]

    # ── Load settings.json ────────────────────────────────────────────────
    settings_path = os.path.join(cli.run_dir, "settings.json")
    if not os.path.exists(settings_path):
        logger.error("settings.json not found in %s", cli.run_dir)
        sys.exit(1)

    with open(settings_path) as f:
        settings = json.load(f)
    args = _namespace_from_dict(settings)
    logger.info("Loaded settings from: %s", settings_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build model from saved settings ───────────────────────────────────
    import models.vqvae as vqvae

    logger.info(
        "Building VQVAE: hidden=%d, levels=%d, embed=%d",
        args.vqvae_hidden_channels,
        args.vqvae_nb_levels,
        args.vqvae_embed_dim,
    )
    model = vqvae.VQVAE(
        in_channels=1,
        hidden_channels=args.vqvae_hidden_channels,
        res_channels=args.vqvae_res_channels,
        nb_res_layers=2,
        nb_levels=args.vqvae_nb_levels,
        embed_dim=args.vqvae_embed_dim,
        nb_entries=args.vqvae_nb_entries,
        scaling_rates=args.vqvae_scaling_rates,
        content_size=len(args.content_indices[0]),
        style_size=len(args.style_indices),
        inject_style_to_decoder=getattr(args, "inject_style_to_decoder", False),
        content_style_levels=getattr(args, "content_style_levels", [0]),
        content_ratios=getattr(args, "content_ratios", None),
        separate_encoders=getattr(args, "separate_encoders", False),
        mask_mode=getattr(args, "mask_mode", "onthefly"),
        quantize_style=getattr(args, "quantize_style", False),
        style_embed_dim=getattr(args, "style_embed_dim", None),
        style_nb_entries=getattr(args, "style_nb_entries", None),
        style_injection_mode=getattr(args, "style_injection_mode", "concat"),
        use_content_projection=getattr(args, "use_content_projection", False),
        narrow_encoder_input=getattr(args, "narrow_encoder_input", False),
        top_level_recon_only=getattr(args, "top_level_recon_only", False),
        pass_full_to_next_level=getattr(args, "pass_full_to_next_level", False),
        skip_decoder_concat_levels=getattr(args, "skip_decoder_concat_levels", None),
        style_dropout_prob=getattr(args, "style_dropout_prob", 0.0),
        detach_style_injection=getattr(args, "detach_style_injection", False),
    )

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt_path = cli.checkpoint or os.path.join(cli.run_dir, "vqvae_model.pt")
    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    state_dict = ckpt.get("encoders", ckpt)
    cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()

    step = ckpt.get("step", -1)
    logger.info("Checkpoint step: %s", step)
    logger.info("Model params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    # ── Build synthetic dataset ───────────────────────────────────────────
    from data.datasets import SyntheticBrainDataset

    n_samples = cli.num_samples or getattr(args, "synthetic_num_test", 200)
    res = getattr(args, "synthetic_res", 64)
    spatial_size = getattr(args, "spatial_size", (res, res, res))

    logger.info("Generating %d synthetic test samples at resolution %s...", n_samples, spatial_size)
    test_dataset = SyntheticBrainDataset(
        mode="test",
        spatial_size=spatial_size,
        synthetic_mode=getattr(args, "synthetic_mode", "pseudo_mri"),
        synthetic_seed=getattr(args, "synthetic_seed", 42),
        synthetic_num_samples=n_samples,
        synthetic_n_content=getattr(args, "synthetic_n_content", 9),
        synthetic_n_style=getattr(args, "synthetic_n_style", 3),
        synthetic_n_deformation_grid=getattr(args, "synthetic_n_deformation_grid", 4),
        synthetic_n_fissure_grid=getattr(args, "synthetic_n_fissure_grid", 8),
        synthetic_hierarchical_content=getattr(args, "synthetic_hierarchical_content", False),
    )

    # ── Run DCI ───────────────────────────────────────────────────────────
    from eval.dci import compute_dci_synthetic

    logger.info("Computing DCI metrics (pooling=%s, levels=%s)...", pooling, levels)
    results = compute_dci_synthetic(
        encoder=model,
        dataset=test_dataset,
        device=device,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        pooling=pooling,
        levels=levels,
    )

    # ── Print results ─────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("DCI Synthetic Evaluation Results")
    print("=" * 65)

    sections = ["content→content", "content→style", "style→style", "style→content"]

    for lvl in levels:
        prefix = f"L{lvl}/" if len(levels) > 1 else ""
        fi_key = f"{prefix}factor_info"
        fi = results.get(fi_key)
        if fi is None:
            print(f"\n  Level {lvl}: no data")
            continue

        print(f"\n  Level {lvl}  (pooling={fi['pooling']})")
        print(f"  Content channels: {fi['n_content_channels']}  |  Style channels: {fi['n_style_channels']}")
        print(f"  Content factors:  {fi['content_names']}")
        print(f"  Style factors:    {fi['style_names']}")
        print()

        for section in sections:
            full_section = f"{prefix}{section}"
            detail = results.get(f"{full_section}/detail")
            if detail is None:
                continue

            d_score = results.get(f"{full_section}/disentanglement", float("nan"))
            c_score = results.get(f"{full_section}/completeness", float("nan"))
            i_train = results.get(f"{full_section}/informativeness_train", float("nan"))
            i_test = results.get(f"{full_section}/informativeness_test", float("nan"))

            print(f"  {section}")
            print(f"    {'':30s} {'Train R²':>10s} {'Test R²':>10s} {'Complet.':>10s}")
            print(f"    {'─' * 62}")
            names = detail["factor_names"]
            for j, name in enumerate(names):
                tr = detail["per_factor_train"][j]
                te = detail["per_factor_test"][j]
                co = detail["per_factor_completeness"][j]
                print(f"    {name:30s} {tr:10.4f} {te:10.4f} {co:10.4f}")
            print(f"    {'─' * 62}")
            print(f"    {'MEAN':30s} {i_train:10.4f} {i_test:10.4f} {c_score:10.4f}")
            print(f"    Disentanglement: {d_score:.4f}")
            print()

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = cli.output_dir or cli.run_dir
    os.makedirs(out_dir, exist_ok=True)

    from eval.dci import DCI_CSV_COLUMNS, dci_results_to_rows

    rows = dci_results_to_rows(results)
    csv_path = os.path.join(out_dir, "dci_synthetic.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DCI_CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    json_path = os.path.join(out_dir, "dci_synthetic.json")
    serialisable = {}
    for k, v in results.items():
        if isinstance(v, (int, float, np.floating)):
            serialisable[k] = float(v)
        elif isinstance(v, dict) and "importance_matrix" not in v:
            serialisable[k] = v
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    logger.info("Results saved to: %s", csv_path)
    logger.info("Results saved to: %s", json_path)

    # ── Importance heatmaps ───────────────────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if isinstance(pooling, (list, tuple)):
            n_repeats = int(np.prod(pooling))
        elif pooling == "stats":
            n_repeats = 4
        else:
            n_repeats = 1

        for lvl in levels:
            prefix = f"L{lvl}/" if len(levels) > 1 else ""
            fi = results.get(f"{prefix}factor_info")
            if fi is None:
                continue

            heatmap_sections = []
            for section in sections:
                detail = results.get(f"{prefix}{section}/detail")
                if detail is not None:
                    heatmap_sections.append((section, detail))
            if not heatmap_sections:
                continue

            fig, axes = plt.subplots(1, len(heatmap_sections), figsize=(5 * len(heatmap_sections), 6), squeeze=False)
            for ax, (section, detail) in zip(axes[0], heatmap_sections):
                im = detail["importance_matrix"]
                names = detail["factor_names"]
                n_codes, n_factors = im.shape

                if n_repeats > 1 and n_codes >= n_repeats:
                    n_ch = n_codes // n_repeats
                    agg = im.reshape(n_repeats, n_ch, n_factors).sum(axis=0)
                else:
                    agg = im

                ax.imshow(agg.T, aspect="auto", cmap="viridis")
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names, fontsize=8)
                xlabel = f"channel (aggregated from {n_codes} features)" if n_repeats > 1 else "channel"
                ax.set_xlabel(xlabel, fontsize=8)
                ax.set_title(section, fontsize=10)

            fig.suptitle(f"Importance matrix — Level {lvl} ({fi['pooling']})", fontsize=12)
            fig.tight_layout()
            heatmap_path = os.path.join(out_dir, f"dci_importance_L{lvl}.png")
            fig.savefig(heatmap_path, dpi=150)
            plt.close(fig)
            logger.info("Heatmap saved to: %s", heatmap_path)

            np.savez_compressed(
                os.path.join(out_dir, f"dci_importance_L{lvl}.npz"),
                **{f"{s}/importance_matrix": d["importance_matrix"] for s, d in heatmap_sections},
                **{f"{s}/factor_names": d["factor_names"] for s, d in heatmap_sections},
            )
    except ImportError:
        logger.warning("matplotlib not available — skipping heatmap export")


if __name__ == "__main__":
    main()
