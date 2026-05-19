#!/usr/bin/env python
"""Standalone DCI evaluation on a saved VQ-VAE checkpoint + synthetic data.

Usage:
    python -m eval.run_dci_synthetic --checkpoint path/to/vqvae_model.pt [training flags...]

The script re-creates the VQVAE from the same CLI flags used during training,
loads the checkpoint weights, generates a synthetic test set, and prints DCI
scores for every content/style combination.

Any flag accepted by the training script can be passed here — the script uses
the same parse_args() so the model architecture matches the checkpoint.  Only
the flags that affect model construction and synthetic data matter; training-
only flags (lr, iterations, etc.) are ignored.

Examples:
    # Minimal — uses defaults for all arch flags:
    python -m eval.run_dci_synthetic --checkpoint runs/my_run/vqvae_model.pt

    # Override synthetic data size and architecture flags to match training:
    python -m eval.run_dci_synthetic \
        --checkpoint runs/my_run/vqvae_best.pt \
        --vqvae-hidden-channels 128 \
        --vqvae-nb-levels 3 \
        --content-size 32 \
        --synthetic-n-content 9 \
        --synthetic-n-style 3 \
        --synthetic-num-test 500 \
        --batch-size 16
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

import models.vqvae as vqvae
from data.datasets import SyntheticBrainDataset
from eval.dci import compute_dci_synthetic
from utils.config import parse_args, update_args

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    # ── Parse args ────────────────────────────────────────────────────────
    # Inject --dataset_name synthetic so parse_args sets up DATASETCLASS etc.
    # Also inject --evaluate so the arg-postprocessing path runs.
    if "--dataset_name" not in sys.argv:
        sys.argv.extend(["--dataset_name", "synthetic"])
    if "--evaluate" not in sys.argv:
        sys.argv.append("--evaluate")

    # Add --checkpoint to the parser before parse_args runs.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file")
    pre_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write dci_synthetic.csv (default: same dir as checkpoint)",
    )
    pre_parser.add_argument("--num-samples", type=int, default=None, help="Override number of test samples")
    pre_parser.add_argument("--batch-size-eval", type=int, default=32)
    pre_parser.add_argument("--num-workers-eval", type=int, default=0)
    pre_known, remaining = pre_parser.parse_known_args()

    # Temporarily replace sys.argv so parse_args sees only the remaining flags.
    # parse_args() returns the ArgumentParser; .parse_args() gives the namespace;
    # update_args() fills in dataset-specific fields (content_indices, etc.).
    orig_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining
    try:
        parser = parse_args()
        args = parser.parse_args()
        args = update_args(args)
    finally:
        sys.argv = orig_argv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # ── Build model ───────────────────────────────────────────────────────
    logger.info("Building VQVAE with the same architecture flags...")
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
    ckpt_path = pre_known.checkpoint
    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    state_dict = ckpt.get("encoders", ckpt)
    # Strip "module." prefix from DataParallel checkpoints
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.removeprefix("module.")] = v
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()

    step = ckpt.get("step", -1)
    logger.info("Checkpoint step: %s", step)
    logger.info("Model params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    # ── Build synthetic dataset ───────────────────────────────────────────
    n_samples = pre_known.num_samples or getattr(args, "synthetic_num_test", 200)
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
    logger.info("Computing DCI metrics...")
    results = compute_dci_synthetic(
        encoder=model,
        dataset=test_dataset,
        device=device,
        batch_size=pre_known.batch_size_eval,
        num_workers=pre_known.num_workers_eval,
    )

    # ── Print results ─────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("DCI Synthetic Evaluation Results")
    print("=" * 65)

    fi = results["factor_info"]
    print(f"  Content channels: {fi['n_content_channels']}  |  Style channels: {fi['n_style_channels']}")
    print(f"  Content factors:  {fi['content_names']}")
    print(f"  Style factors:    {fi['style_names']}")
    print()

    for section in ["content→content", "content→style", "style→style", "style→content"]:
        header = f"  {section}"
        metrics = {k.split("/")[1]: v for k, v in results.items() if k.startswith(section + "/")}
        if not metrics:
            continue
        print(header)
        for mk, mv in metrics.items():
            print(f"    {mk:<25s} {mv:.4f}")
        print()

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = pre_known.output_dir or os.path.dirname(ckpt_path)
    os.makedirs(out_dir, exist_ok=True)

    flat = {k: v for k, v in results.items() if k != "factor_info"}
    csv_path = os.path.join(out_dir, "dci_synthetic.csv")
    import csv

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat.keys())
        w.writeheader()
        w.writerow(flat)

    json_path = os.path.join(out_dir, "dci_synthetic.json")
    with open(json_path, "w") as f:
        json.dump({k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in results.items()}, f, indent=2)

    logger.info("Results saved to: %s", csv_path)
    logger.info("Results saved to: %s", json_path)


if __name__ == "__main__":
    main()
