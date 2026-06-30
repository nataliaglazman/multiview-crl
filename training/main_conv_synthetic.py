"""Encoder-only multi-view contrastive learning on the 3D synthetic data.

The faithful image-track method of Yao et al. (ICLR 2024): per-view 3D conv
encoders (``models.multiview_encoder.MultiviewConvEncoder``) trained by the
content-alignment − entropy contrastive loss (InfoNCE or Barlow Twins) with
**no decoder and no reconstruction**. Sibling to ``training.main_numerical`` (the
MLP/abstract-latent version) and ``training.main_vae_synthetic`` (the recon
baseline) — this one keeps the same VQ-VAE conv backbone as the latter but drops
the decoder, so the two scripts form a controlled recon vs. no-recon comparison.

Identifiability is scored with ``eval.dci.compute_dci_synthetic`` (per-latent
RidgeCV/GBT R² + block-MCC + content→view leakage): content latents → high,
independent style → ~chance is the block-identification signal.

Example:
    python -m training.main_conv_synthetic --model-id conv_c9 \
        --latent-dim 16 --content-channels 9 --n-content 9 --n-style 3 \
        --train-steps 50000 --eval-every 2000 --contrastive-loss-type infonce
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import eval.dci as dci
import training.losses as losses
from data.datasets import SyntheticBrainDataset
from models.multiview_encoder import MultiviewConvEncoder


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--model-id", type=str, default="conv_synthetic")

    # Model.
    p.add_argument("--latent-dim", type=int, default=16, help="Total encoding size (content + style)")
    p.add_argument("--content-channels", type=int, default=9, help="Content units (set to the true n_content)")
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--res-channels", type=int, default=32)
    p.add_argument("--nb-res-layers", type=int, default=2)
    p.add_argument("--downscale-factor", type=int, default=4, help="Spatial downscale (power of 2)")
    p.add_argument("--no-separate-encoders", action="store_true", help="Share one encoder across both views")

    # Contrastive loss (content alignment − entropy).
    p.add_argument("--contrastive-loss-type", type=str, default="infonce", choices=["infonce", "barlow_twins"])
    p.add_argument("--tau", type=float, default=1.0, help="InfoNCE temperature")
    p.add_argument("--bt-lambda", type=float, default=0.005, help="Barlow Twins off-diagonal weight")
    p.add_argument(
        "--cross-view-negs-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="InfoNCE negatives only from the other view (paper aligns across views)",
    )

    # Optimisation.
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64, help="Per-view batch size")
    p.add_argument("--train-steps", type=int, default=50000)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=2.0, help="Max grad 2-norm (paper uses 2); 0 disables")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-cuda", action="store_true")

    # Evaluation pooling — GAP is the paper-faithful default; patch probes whether
    # content survives at spatial resolution (see groupnorm-caps-gap-pooled-mcc).
    p.add_argument("--eval-pooling", type=str, default="gap", choices=["gap", "patch"])
    p.add_argument("--eval-patch-grid", type=int, nargs=3, default=[4, 5, 4])

    # Synthetic dataset (forwarded in full to train AND val so the distributions match).
    p.add_argument("--res", type=int, default=32, help="Cubic resolution (power of 2)")
    p.add_argument("--n-content", type=int, default=9, help="True shared content factors")
    p.add_argument("--n-style", type=int, default=3, help="Per-view style factors")
    p.add_argument("--num-train-samples", type=int, default=2000)
    p.add_argument("--num-val-samples", type=int, default=400)
    p.add_argument("--synthetic-mode", type=str, default="pseudo_mri")
    p.add_argument(
        "--synthetic-normalize", type=str, default="per_sample", choices=["per_sample", "shared", "fixed_reference"]
    )
    p.add_argument(
        "--synthetic-clean-content",
        action="store_true",
        help="Zero the unlabeled deformation/fissure nuisance so the named content factors dominate",
    )
    p.add_argument("--synthetic-style-scale", type=float, default=1.0)
    p.add_argument("--synthetic-content-scale", type=float, default=1.0)
    p.add_argument("--synthetic-n-deformation-grid", type=int, default=4)
    p.add_argument("--synthetic-n-fissure-grid", type=int, default=8)
    p.add_argument("--synthetic-hierarchical-content", action="store_true")
    p.add_argument("--synthetic-causal", action="store_true")
    return p.parse_args()


def make_dataset(args, mode, num_samples):
    """Single factory for train/val/test so the generative distribution is identical."""
    return SyntheticBrainDataset(
        mode=mode,
        spatial_size=(args.res, args.res, args.res),
        cache=True,
        synthetic_mode=args.synthetic_mode,
        synthetic_seed=args.seed,
        synthetic_num_samples=num_samples,
        synthetic_n_content=args.n_content,
        synthetic_n_style=args.n_style,
        synthetic_style_scale=args.synthetic_style_scale,
        synthetic_content_scale=args.synthetic_content_scale,
        synthetic_n_deformation_grid=args.synthetic_n_deformation_grid,
        synthetic_n_fissure_grid=args.synthetic_n_fissure_grid,
        synthetic_hierarchical_content=args.synthetic_hierarchical_content,
        synthetic_normalize=args.synthetic_normalize,
        synthetic_causal=args.synthetic_causal,
        synthetic_clean_content=args.synthetic_clean_content,
    )


def contrastive_loss(pooled, args, sim_metric, criterion):
    """Content-alignment − entropy on the pooled content block across the two views."""
    b = pooled.shape[0] // 2
    hz = torch.stack([pooled[:b], pooled[b:]], dim=0)  # (2, B, C)
    content_indices = [list(range(args.content_channels))]
    if args.contrastive_loss_type == "barlow_twins":
        loss = losses.barlow_twins_loss(
            hz, estimated_content_indices=content_indices, subsets=[(0, 1)], lambd=args.bt_lambda
        )
    else:
        loss = losses.infonce_loss(
            hz,
            sim_metric=sim_metric,
            criterion=criterion,
            tau=args.tau,
            projector=(lambda t: t),
            estimated_content_indices=content_indices,
            subsets=[(0, 1)],
            cross_view_negs_only=args.cross_view_negs_only,
        )
    return loss.squeeze()


def evaluate(model, val_dataset, device, args, save_dir, step, writer=None):
    print(f"  [eval] synthetic DCI @ step {step} ...", flush=True)
    pooling = "gap" if args.eval_pooling == "gap" else tuple(args.eval_patch_grid)
    results = dci.compute_dci_synthetic(
        encoder=model,
        dataset=val_dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=0,
        pooling=pooling,
        per_encoder=not args.no_separate_encoders,
    )
    flat = dci.flatten_dci_results(results)

    def show(key):
        for k in flat:
            if k.endswith(key) and np.isfinite(flat[k]):
                print(f"      {k:60s} {flat[k]:.3f}", flush=True)

    print("    --- identifiability summary ---", flush=True)
    show("content->content/block_mcc")
    show("content->content/informativeness_ridge")
    show("content->style/block_mcc")
    show("content->view/acc")

    if writer is not None:
        for k, v in flat.items():
            if np.isfinite(v):
                writer.add_scalar(f"dci_synthetic/{k}", v, step)

    with open(os.path.join(save_dir, f"dci_step{step}.json"), "w") as fp:
        json.dump({k: float(v) for k, v in flat.items()}, fp, indent=2)
    return flat


def main():
    args = parse_args()
    save_dir = os.path.join(args.out_dir, args.model_id)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "settings.json"), "w") as fp:
        json.dump(vars(args), fp, indent=2)

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"device: {device}", flush=True)
    for k, v in vars(args).items():
        print(f"\t{k}: {v}", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = make_dataset(args, "train", args.num_train_samples)
    val_dataset = make_dataset(args, "val", args.num_val_samples)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )

    model = MultiviewConvEncoder(
        in_channels=1,
        hidden_channels=args.hidden_channels,
        res_channels=args.res_channels,
        nb_res_layers=args.nb_res_layers,
        downscale_factor=args.downscale_factor,
        latent_dim=args.latent_dim,
        content_channels=args.content_channels,
        separate_encoders=not args.no_separate_encoders,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sim_metric = torch.nn.CosineSimilarity(dim=-1)
    criterion = torch.nn.CrossEntropyLoss()

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(os.path.join(save_dir, "tensorboard"))
    except Exception:
        writer = None

    step = 0
    running = {"loss": 0.0, "n": 0}
    model.train()
    while step < args.train_steps:
        for batch in train_loader:
            if step >= args.train_steps:
                break
            x = torch.cat(batch["image"], dim=0).to(device)  # (2B, 1, res, res, res)

            _, _, feats, _, _, _, _, _ = model(x, pool_only=True, n_views=2)
            pooled = feats[0]  # (2B, latent_dim)
            loss = contrastive_loss(pooled, args, sim_metric, criterion)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running["loss"] += loss.item()
            running["n"] += 1
            step += 1

            if step % args.log_every == 0:
                n = max(running["n"], 1)
                print(f"step {step:6d} | contrastive {running['loss']/n:.4f}", flush=True)
                if writer is not None:
                    writer.add_scalar("train/contrastive", running["loss"] / n, step)
                running = {"loss": 0.0, "n": 0}

            if step % args.eval_every == 0 or step == args.train_steps:
                model.eval()
                evaluate(model, val_dataset, device, args, save_dir, step, writer)
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
                model.train()

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    print(f"done. checkpoints + DCI logs in {save_dir}", flush=True)


if __name__ == "__main__":
    main()
