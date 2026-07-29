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


def load_run_args(run_dir):
    """Load a run's settings.json into an argparse.Namespace (no model build)."""
    settings_path = os.path.join(run_dir, "settings.json")
    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"settings.json not found in {run_dir}")
    with open(settings_path) as f:
        return _namespace_from_dict(json.load(f))


def load_model_from_run_dir(run_dir, checkpoint=None, device=None):
    """Rebuild the VQVAE from a run's settings.json and load its checkpoint.

    Returns ``(model, args, device)``.  Shared by the single-run eval and the
    model-comparison pipeline so model construction lives in exactly one place.
    """
    import models.vqvae as vqvae

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = load_run_args(run_dir)
    logger.info("Loaded settings from: %s", os.path.join(run_dir, "settings.json"))
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
        nb_res_layers=getattr(args, "vqvae_nb_res_layers", 2),
        nb_levels=args.vqvae_nb_levels,
        embed_dim=args.vqvae_embed_dim,
        nb_entries=args.vqvae_nb_entries,
        scaling_rates=args.vqvae_scaling_rates,
        use_checkpoint=False,
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
        cb_ema_decay=getattr(args, "cb_ema_decay", 0.999),
        cb_reset_every=getattr(args, "cb_reset_every", 100),
        cb_reset_threshold=getattr(args, "cb_reset_threshold", 1.0),
        use_content_projection=getattr(args, "use_content_projection", False),
        narrow_encoder_input=getattr(args, "narrow_encoder_input", False),
        top_level_recon_only=getattr(args, "top_level_recon_only", False),
        pass_full_to_next_level=getattr(args, "pass_full_to_next_level", False),
        skip_decoder_concat_levels=getattr(args, "skip_decoder_concat_levels", None),
        style_dropout_prob=getattr(args, "style_dropout_prob", 0.0),
        detach_style_injection=getattr(args, "detach_style_injection", False),
        style_spatial_size=getattr(args, "style_spatial_size", 0),
        final_recon_norm=not getattr(args, "no_final_recon_norm", False),
        # These were missing, and their absence was SILENT. norm_type in particular:
        # ChannelLayerNorm3d stores its affine params under a nested `.norm.` prefix while
        # GroupNorm stores them flat, so rebuilding a `--norm-type layer` run with the
        # GroupNorm default made all 28 encoder-norm tensors mismatch by NAME. With
        # strict=False below they were dropped without error, and the evaluated model was
        # a hybrid that never existed in training: trained convolutions plus freshly
        # initialised GroupNorms -- which re-introduces exactly the whole-volume statistic
        # the run was configured to remove.
        norm_type=getattr(args, "norm_type", "group"),
        decoder_norm_type=getattr(args, "decoder_norm_type", None),
        separate_content_codebooks=getattr(args, "separate_content_codebooks", False),
        separate_style_codebooks=getattr(args, "separate_style_codebooks", False),
        latent_mask=getattr(args, "latent_mask", False),
        latent_mask_thresh=getattr(args, "latent_mask_thresh", 0.0),
    )

    ckpt_path = checkpoint or os.path.join(run_dir, "vqvae_model.pt")
    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("encoders", ckpt)
    cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}
    incompat = model.load_state_dict(cleaned, strict=False)
    # strict=False is kept on purpose (checkpoints legitimately carry extra state such as
    # MoCo queues), but it must never be silent: a name mismatch here means the rebuilt
    # architecture disagrees with the trained one and the eval is running a different model.
    if incompat.missing_keys or incompat.unexpected_keys:
        logger.warning(
            "ARCHITECTURE MISMATCH: %d parameter(s) missing (left at INIT), %d unexpected (DISCARDED). "
            "The rebuilt model does not match the checkpoint -- results from it are not this run's.",
            len(incompat.missing_keys),
            len(incompat.unexpected_keys),
        )
        for k in incompat.missing_keys[:5]:
            logger.warning("    missing    (using fresh init): %s", k)
        for k in incompat.unexpected_keys[:5]:
            logger.warning("    unexpected (dropped from ckpt): %s", k)
        if any(".norm." in k for k in incompat.unexpected_keys) or any(
            k.endswith((".weight", ".bias")) and ".norm." not in k for k in incompat.missing_keys
        ):
            logger.warning(
                "    Looks like a norm_type mismatch (layer <-> group). Check settings.json norm_type "
                "against how this model was built."
            )
    model.to(device)
    model.eval()
    logger.info(
        "Checkpoint step: %s | params: %s",
        ckpt.get("step", -1),
        f"{sum(p.numel() for p in model.parameters()):,}",
    )
    return model, args, device


def build_synthetic_test_set(args, num_samples=None, cache=True, causal=None):
    """Build the synthetic test dataset from a run's settings (shared helper).

    ``cache=True`` (default) renders each volume once into RAM and reuses it on
    every later access.  This matters because the frozen test set is iterated
    once per pooling and once per model, so without caching the procedural
    generator re-renders the whole set 3*N times.  Costs ~num_samples*4*res**3
    floats of RAM; pass ``cache=False`` if that is too much.

    ``causal`` controls whether the eval set reproduces the training-time SCM:

    * ``None`` (default) — i.i.d. factors, the long-standing behaviour. Kept as the default
      so existing numbers stay comparable, but a WARNING is emitted when the run itself was
      trained with an SCM, because the two distributions are not interchangeable: under a
      random graph ventricle_size and brain_size correlate at ~0.8, so a "ventricle" probe
      largely reads brain_size, while at i.i.d. it does not.
    * ``True`` — forward the run's SCM, matching the training distribution.
    * ``False`` — force i.i.d. deliberately and silently (the honest choice when you want
      factors decorrelated so per-factor attribution is unambiguous).
    """
    from data.datasets import SyntheticBrainDataset

    n_samples = num_samples or getattr(args, "synthetic_num_test", 200)
    res = getattr(args, "synthetic_res", 64)
    spatial_size = getattr(args, "spatial_size", None) or (res, res, res)
    trained_causal = bool(getattr(args, "synthetic_causal", False))
    if causal is None and trained_causal:
        logger.warning(
            "This run was TRAINED with synthetic_causal=True, but the eval set is being built with "
            "i.i.d. factors. Factor correlations differ between train and eval, which can reorder "
            "per-factor results. Pass causal=True to match training, or causal=False to silence this."
        )
    use_causal = trained_causal if causal else False
    logger.info(
        "Generating %d synthetic test samples at resolution %s (causal=%s)...", n_samples, spatial_size, use_causal
    )
    return SyntheticBrainDataset(
        mode="test",
        spatial_size=spatial_size,
        cache=cache,
        synthetic_mode=getattr(args, "synthetic_mode", "pseudo_mri"),
        synthetic_seed=getattr(args, "synthetic_seed", 42),
        synthetic_num_samples=n_samples,
        synthetic_n_content=getattr(args, "synthetic_n_content", 9),
        synthetic_n_style=getattr(args, "synthetic_n_style", 3),
        synthetic_style_scale=getattr(args, "synthetic_style_scale", 1.0),
        synthetic_content_scale=getattr(args, "synthetic_content_scale", 1.0),
        synthetic_n_deformation_grid=getattr(args, "synthetic_n_deformation_grid", 4),
        synthetic_n_fissure_grid=getattr(args, "synthetic_n_fissure_grid", 8),
        synthetic_hierarchical_content=getattr(args, "synthetic_hierarchical_content", False),
        synthetic_normalize=getattr(args, "synthetic_normalize", "per_sample"),
        synthetic_clean_content=getattr(args, "synthetic_clean_content", False),
        synthetic_causal=use_causal,
        synthetic_causal_graph=getattr(args, "synthetic_causal_graph", "chain"),
        synthetic_causal_edge_prob=getattr(args, "synthetic_causal_edge_prob", 0.5),
        synthetic_causal_noise_scale=getattr(args, "synthetic_causal_noise_scale", 0.4),
        synthetic_causal_nonlinearity=getattr(args, "synthetic_causal_nonlinearity", "leaky_relu"),
    )


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
    parser.add_argument(
        "--null-permute",
        action="store_true",
        help="Also compute a row-permuted null floor per block — the D/C/I a block's "
        "shape yields from noise alone. Real minus null is the non-structural signal.",
    )
    parser.add_argument("--n-null", type=int, default=5, help="Permutations to average for the null floor")
    cli = parser.parse_args()

    if cli.pooling in ("gap", "stats"):
        pooling = cli.pooling
    else:
        pooling = tuple(int(x) for x in cli.pooling.split(","))
    levels = [int(x) for x in cli.levels.split(",")]

    # ── Load model + build synthetic dataset (shared helpers) ─────────────
    if not os.path.exists(os.path.join(cli.run_dir, "settings.json")):
        logger.error("settings.json not found in %s", cli.run_dir)
        sys.exit(1)
    model, args, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint)
    test_dataset = build_synthetic_test_set(args, cli.num_samples)

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
        null_permute=cli.null_permute,
        n_null=cli.n_null,
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

            n_train = results.get(f"{full_section}/null/informativeness_train")
            if n_train is not None:
                n_test = results.get(f"{full_section}/null/informativeness_test", float("nan"))
                n_comp = results.get(f"{full_section}/null/completeness", float("nan"))
                n_dis = results.get(f"{full_section}/null/disentanglement", float("nan"))
                print(f"    {'NULL (shape floor)':30s} {n_train:10.4f} {n_test:10.4f} {n_comp:10.4f}")
                print(
                    f"    {'GAP (real − null)':30s} {i_train - n_train:10.4f} {i_test - n_test:10.4f} {c_score - n_comp:10.4f}"
                )
                print(f"    Disentanglement: {d_score:.4f}   (null {n_dis:.4f}, gap {d_score - n_dis:+.4f})")
            else:
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
