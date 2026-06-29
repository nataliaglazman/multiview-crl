"""Identifiability analysis for a trained MultiviewVAE checkpoint.

Loads ``results/<model-id>/{settings.json, model.pt}`` (produced by
``training.main_vae_synthetic``), rebuilds the model, and scores it on the
synthetic **test** split with the same trusted metric functions used elsewhere
in the repo.  Reports three families, ordered by how much the theory supports
them (see ``eval/identifiability_metrics`` docstring):

  Tier 1 — block identifiability (the actual claim):
      * content->content block-MCC  (high = content subspace recovered up to an
        invertible map) and content->style block-MCC (low = no style leakage).
      * content informativeness (ridge / GBT test-R^2).
      * content->view probe accuracy (~0.5 = content is view-invariant).

  Tier 3 — component-wise diagnostics (NOT guaranteed by the theory):
      * per-channel ("normal") MCC — strict one-channel<->one-factor matching.
      * DCI disentanglement / completeness.
      * factor-correlation ceiling — the dependence DCI-D cannot exceed.

A contrastive multi-view model is expected to be *block*-identifiable, not
axis-aligned, so a high block-MCC with a modest per-channel MCC is the success
case, not a failure.

Usage:
    python -m eval.analyze_vae_identifiability --run-dir results/vae_synthetic
    python -m eval.analyze_vae_identifiability --run-dir results/vae_synth_c9 --null-permute
"""

import argparse
import json
import os

import numpy as np
import torch

import eval.dci as dci
from data.datasets import SyntheticBrainDataset
from eval.identifiability_metrics import block_mcc, factor_correlation_ceiling, per_channel_mcc
from models.vae import MultiviewVAE


def load_model(run_dir, device):
    with open(os.path.join(run_dir, "settings.json")) as fp:
        cfg = json.load(fp)
    model = MultiviewVAE(
        in_channels=1,
        hidden_channels=cfg.get("hidden_channels", 64),
        res_channels=cfg.get("res_channels", 32),
        nb_res_layers=cfg.get("nb_res_layers", 2),
        downscale_factor=cfg.get("downscale_factor", 4),
        latent_channels=cfg.get("latent_channels", 16),
        content_channels=cfg.get("content_channels", 9),
        beta_kl=cfg.get("beta_kl", 0.0),
        separate_encoders=not cfg.get("no_separate_encoders", False),
    ).to(device)
    state = torch.load(os.path.join(run_dir, "model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def build_test_dataset(cfg, num_samples):
    return SyntheticBrainDataset(
        mode="test",
        spatial_size=(cfg.get("res", 32),) * 3,
        cache=True,
        synthetic_mode=cfg.get("synthetic_mode", "pseudo_mri"),
        synthetic_seed=cfg.get("seed", 42),
        synthetic_num_samples=num_samples,
        synthetic_n_content=cfg.get("n_content", 9),
        synthetic_n_style=cfg.get("n_style", 3),
        synthetic_normalize=cfg.get("synthetic_normalize", "per_sample"),
        synthetic_causal=cfg.get("synthetic_causal", False),
    )


def _fmt(v):
    return "  nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def run_analysis(run_dir, num_test_samples=400, batch_size=32, num_workers=0, null_permute=False, device=None):
    """Load a run, score it on the test split, and return ``(cfg, metrics)``.

    ``metrics`` is a flat dict: the trusted ``compute_dci_synthetic`` outputs
    (DCI, block-MCC, content->view leakage, ridge informativeness) plus the
    component-wise per-channel MCC and the factor-correlation ceiling.  Shared by
    the single-run CLI below and ``eval.sweep_vae_identifiability``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(run_dir, device)
    dataset = build_test_dataset(cfg, num_test_samples)
    per_encoder = not cfg.get("no_separate_encoders", False)

    results = dci.compute_dci_synthetic(
        encoder=model,
        dataset=dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        pooling="gap",
        per_encoder=per_encoder,
        null_permute=null_permute,
        n_null=5,
    )
    metrics = dict(dci.flatten_dci_results(results))

    # Per-channel ("normal") MCC needs the raw channels — re-extract once.
    level_data, gt_content, gt_style_v1, _gt_style_v2 = dci._extract_synthetic_representations(
        model, dataset, device, batch_size=batch_size, num_workers=num_workers, pooling="gap"
    )
    content_repr, style_repr, _content_v2, _style_v2, _info = level_data[0]
    metrics["per_channel_mcc/content->content"] = float(per_channel_mcc(content_repr, gt_content))
    if style_repr is not None:
        metrics["per_channel_mcc/style->style"] = float(per_channel_mcc(style_repr, gt_style_v1))
    metrics["factor_correlation_ceiling/content"] = float(factor_correlation_ceiling(gt_content))
    return cfg, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="results/<model-id> directory with settings.json + model.pt")
    p.add_argument("--num-test-samples", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--null-permute", action="store_true", help="Also compute row-permuted null floors for DCI")
    p.add_argument("--no-cuda", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model, cfg = load_model(args.run_dir, device)
    dataset = build_test_dataset(cfg, args.num_test_samples)
    print(f"run-dir: {args.run_dir}")
    print(
        f"latent_channels={cfg.get('latent_channels')}  content_channels={cfg.get('content_channels')}  "
        f"n_content={cfg.get('n_content')}  n_style={cfg.get('n_style')}"
    )

    # --- Trusted protocol: DCI + block-MCC + content->view leakage ---
    results = dci.compute_dci_synthetic(
        encoder=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pooling="gap",
        per_encoder=not cfg.get("no_separate_encoders", False),
        null_permute=args.null_permute,
        n_null=5,
    )
    flat = dci.flatten_dci_results(results)

    def g(key):
        return flat.get(key, float("nan"))

    print("\n=== Tier 1: block identifiability (the claim) ===")
    print(f"{'block':32s} {'D':>6} {'C':>6} {'info_test':>10} {'info_ridge':>11} {'block_MCC':>10}")
    for block in ("content->content", "content->style", "style->style", "style->content"):
        print(
            f"{block:32s} "
            f"{_fmt(g(block + '/disentanglement')):>6} "
            f"{_fmt(g(block + '/completeness')):>6} "
            f"{_fmt(g(block + '/informativeness_test')):>10} "
            f"{_fmt(g(block + '/informativeness_ridge')):>11} "
            f"{_fmt(g(block + '/block_mcc')):>10}"
        )
    print(
        f"\ncontent->view/acc (chance 0.5): {_fmt(g('content->view/acc'))}   "
        f"style->view/acc: {_fmt(g('style->view/acc'))}"
    )

    # --- Tier 3: per-channel ("normal") MCC + factor-correlation ceiling ---
    # Re-extract the raw representations so per_channel_mcc sees individual channels.
    level_data, gt_content, gt_style_v1, gt_style_v2 = dci._extract_synthetic_representations(
        model, dataset, device, batch_size=args.batch_size, num_workers=args.num_workers, pooling="gap"
    )
    content_repr, style_repr, content_v2, style_v2, _info = level_data[0]

    print("\n=== Tier 3: component-wise diagnostics (NOT theory-guaranteed) ===")
    print(f"per-channel MCC  content->content : {_fmt(per_channel_mcc(content_repr, gt_content))}")
    if style_repr is not None:
        print(f"per-channel MCC  style->style     : {_fmt(per_channel_mcc(style_repr, gt_style_v1))}")
    # Side-by-side block-MCC on the SAME extraction (sanity check vs the table above).
    print(f"block-MCC (recomputed) content    : {_fmt(block_mcc(content_repr, gt_content)['mean'])}")
    print(
        f"factor-correlation ceiling (content): {_fmt(factor_correlation_ceiling(gt_content))}"
        "   (DCI-D cannot exceed ~1 - this)"
    )

    out_path = os.path.join(args.run_dir, "identifiability_analysis.json")
    summary = {k: float(v) for k, v in flat.items() if not (isinstance(v, float) and np.isnan(v))}
    summary["per_channel_mcc/content->content"] = float(per_channel_mcc(content_repr, gt_content))
    if style_repr is not None:
        summary["per_channel_mcc/style->style"] = float(per_channel_mcc(style_repr, gt_style_v1))
    summary["factor_correlation_ceiling/content"] = float(factor_correlation_ceiling(gt_content))
    with open(out_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nfull summary -> {out_path}")


if __name__ == "__main__":
    main()
