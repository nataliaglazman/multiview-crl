"""Encoder-only multi-view contrastive learning on the 3D synthetic data.

3D encoders trained with InfoNCE or Barlow Twins, without reconstruction.
The default ``conv`` architecture uses the VQ-VAE convolutional backbone with
an affine readout. ``--encoder-architecture resnet18`` uses a 3D adaptation of
the upstream image encoder: ResNet-18 -> GAP -> Linear -> LeakyReLU -> Linear.
This architecture option does not change the loss, data, or view-sharing policy.

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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--model-id", type=str, default="conv_synthetic")

    # Model.
    p.add_argument(
        "--encoder-architecture",
        choices=("conv", "resnet18"),
        default="conv",
        help="conv: original VQ-VAE backbone; resnet18: upstream ResNet-18 architecture adapted to 3D",
    )
    p.add_argument(
        "--encoder-head-hidden",
        type=int,
        default=100,
        help="ResNet readout width: GAP -> Linear(512, width) -> LeakyReLU -> Linear(width, latent_dim)",
    )
    p.add_argument("--latent-dim", type=int, default=16, help="Total encoding size (content + style)")
    p.add_argument("--content-channels", type=int, default=9, help="Content units (set to the true n_content)")
    p.add_argument("--hidden-channels", type=int, default=64, help="conv architecture only")
    p.add_argument("--res-channels", type=int, default=32, help="conv architecture only")
    p.add_argument("--nb-res-layers", type=int, default=2, help="conv architecture only")
    p.add_argument("--downscale-factor", type=int, default=4, help="conv downscale (power of 2); ResNet uses 32")
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
    p.add_argument(
        "--contrastive-proj-dim",
        type=int,
        default=0,
        help="If > 0, insert an MLP head between the pooled content block and the "
        "contrastive loss. The loss is computed on the head's output while the DCI "
        "probes keep reading the pre-head encoding (the SimCLR/MoCo recipe — the "
        "loss-facing space over-compresses toward view-invariance and loses "
        "linear-probe info). 0 (default) disables the head: the loss acts directly "
        "on the representation being scored.",
    )
    p.add_argument(
        "--contrastive-proj-hidden",
        type=int,
        default=256,
        help="Hidden width of the projection head MLP (Linear -> ReLU -> Linear). "
        "Only used when --contrastive-proj-dim > 0.",
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
    p.add_argument(
        "--floor-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one eval before training and keep it as the untrained floor, so every "
        "later per-factor score prints its delta against it. Costs one extra eval.",
    )
    p.add_argument("--eval-pooling", type=str, default="gap", choices=["gap", "patch"])
    p.add_argument("--eval-patch-grid", type=int, nargs=3, default=[4, 5, 4])

    # Synthetic dataset (forwarded in full to train AND val so the distributions match).
    p.add_argument("--res", type=int, default=32, help="Cubic resolution (power of 2)")
    p.add_argument("--n-content", type=int, default=9, help="True shared content factors")
    p.add_argument("--n-style", type=int, default=3, help="Per-view style factors")
    p.add_argument("--num-train-samples", type=int, default=2000)
    p.add_argument("--num-val-samples", type=int, default=400)
    p.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hold rendered volumes in RAM (default). The cache is (samples x 2 views x res^3 x 4B) "
        "per split once full -- 23.5 GB for 1000 train + 400 val at res 128, which the OOM killer "
        "takes out on the first eval. Pass --no-cache at high res to re-render each sample instead: "
        "slower per step, constant memory.",
    )
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
    p.add_argument(
        "--synthetic-lesion-placement",
        choices=("legacy", "wm_interior"),
        default="legacy",
        help="wm_interior places a full fixed-radius sphere inside final WM labels; legacy reproduces old runs",
    )
    p.add_argument("--synthetic-n-deformation-grid", type=int, default=4)
    p.add_argument("--synthetic-n-fissure-grid", type=int, default=8)
    p.add_argument("--synthetic-hierarchical-content", action="store_true")
    p.add_argument("--synthetic-causal", action="store_true")
    args = p.parse_args(argv)
    if args.encoder_architecture == "resnet18":
        if args.encoder_head_hidden <= 0:
            p.error("--encoder-head-hidden must be positive")
        spatial_size = (args.res + 31) // 32
        if args.eval_pooling == "patch" and any(g < 1 or g > spatial_size for g in args.eval_patch_grid):
            p.error(f"ResNet at --res {args.res} has a {spatial_size}^3 map; --eval-patch-grid must fit it")
    return args


def cache_gb(res, num_samples):
    """RAM the dataset's in-memory cache will hold once every sample has been rendered."""
    return num_samples * 2 * (res**3) * 4 / 1e9


def make_dataset(args, mode, num_samples):
    """Single factory for train/val/test so the generative distribution is identical."""
    if args.cache:
        gb = cache_gb(args.res, num_samples)
        # The cache fills lazily, so an over-large one survives training and is killed
        # later, on the first eval that starts filling the val split's share of it.
        if gb > 4.0:
            print(
                f"  WARNING: {mode} cache will grow to {gb:.1f} GB in RAM "
                f"({num_samples} samples x 2 views x {args.res}^3 x 4B). "
                f"Pass --no-cache to re-render instead of caching.",
                flush=True,
            )
    return SyntheticBrainDataset(
        mode=mode,
        spatial_size=(args.res, args.res, args.res),
        cache=args.cache,
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
        synthetic_lesion_placement=getattr(args, "synthetic_lesion_placement", "legacy"),
    )


def contrastive_loss(pooled, model, args, sim_metric, criterion):
    """Content-alignment − entropy on the pooled content block across the two views.

    The content block is selected first and then projected, so with a head every output
    dimension is part of the loss-facing space and counts as content downstream. The head
    is applied here rather than handed to the losses' ``projector`` argument, which
    ``infonce_base_loss`` accepts but never calls. Without a head ``project`` is the
    identity and this is the same tensor the loss selected internally before.
    """
    b = pooled.shape[0] // 2
    hz = torch.stack([pooled[:b], pooled[b:]], dim=0)  # (2, B, C)
    hz = model.project(hz[..., : args.content_channels])
    content_indices = [list(range(hz.shape[-1]))]
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
            estimated_content_indices=content_indices,
            subsets=[(0, 1)],
            cross_view_negs_only=args.cross_view_negs_only,
        )
    return loss.squeeze()


def effective_rank(feat):
    """Participation ratio of the covariance spectrum: 1 = collapsed to a line, C = isotropic.

    The quantity InfoNCE quietly destroys when alignment is the only pressure on the
    representation. GAP already starts this near 1 on a random encoder, so a run whose
    rank never climbs is discarding information rather than aligning content, and every
    identifiability number it reports will sit at the untrained floor.
    """
    x = feat.detach().float()
    x = x - x.mean(dim=0, keepdim=True)
    ev = torch.linalg.eigvalsh(torch.cov(x.T)).clamp(min=0)
    total = ev.sum()
    return float(total**2 / ev.pow(2).sum()) if total > 0 else 0.0


def per_factor_scores(results):
    """Per-factor ridge R² and block-MCC, keyed by block then factor name.

    Both are already computed inside ``compute_dci_synthetic``; this just reads them off
    the block detail dicts. Returned in a plain-dict shape so a step-0 call can be kept
    as the untrained floor and subtracted from every later eval.
    """
    out = {}
    for block in ("content→content", "content→style"):
        detail = results.get(f"{block}/detail")
        if not isinstance(detail, dict):
            continue
        names = detail.get("factor_names") or []
        ridge, mcc, mcc_std = (detail.get(k) for k in ("per_factor_ridge", "per_factor_mcc", "per_factor_mcc_std"))

        def at(arr, j):
            return float(arr[j]) if arr is not None and j < len(arr) else float("nan")

        out[block] = {
            nm: {"ridge": at(ridge, j), "mcc": at(mcc, j), "mcc_std": at(mcc_std, j)} for j, nm in enumerate(names)
        }
    return out


def print_per_factor(scores, floor=None, writer=None, step=0):
    """One row per ground-truth factor: which factors the representation actually carries.

    The mean over factors hides exactly the case worth checking — a block that recovers
    two coarse global factors well and every localised one at chance reads as a decent
    average. Floor-subtracted where a step-0 eval is available, because raw per-factor R²
    has the same untrained-floor problem as the block means.
    """
    for block, rows in scores.items():
        if not rows:
            continue
        has_floor = bool(floor) and block in floor
        print(f"    --- per-factor recovery: {block} ---", flush=True)
        head = f"      {'factor':<20s}{'ridge R²':>9s}{'MCC':>8s}{'±':>7s}"
        print(head + (f"{'floor':>9s}{'Δ vs floor':>12s}" if has_floor else ""), flush=True)
        for nm, v in rows.items():
            line = f"      {nm:<20s}{v['ridge']:>9.3f}{v['mcc']:>8.3f}{v['mcc_std']:>7.3f}"
            if has_floor and nm in floor[block]:
                fl = floor[block][nm]["ridge"]
                line += f"{fl:>9.3f}{v['ridge'] - fl:>+12.3f}"
            print(line, flush=True)
            if writer is not None:
                tag = block.replace("→", "_to_")
                writer.add_scalar(f"per_factor/{tag}/{nm}/ridge_r2", v["ridge"], step)
                writer.add_scalar(f"per_factor/{tag}/{nm}/mcc", v["mcc"], step)


def evaluate(model, val_dataset, device, args, save_dir, step, writer=None, floor=None):
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

    scores = per_factor_scores(results)
    print_per_factor(scores, floor=floor, writer=writer, step=step)

    if writer is not None:
        for k, v in flat.items():
            if np.isfinite(v):
                writer.add_scalar(f"dci_synthetic/{k}", v, step)

    payload = {k: float(v) for k, v in flat.items()}
    payload["per_factor"] = scores  # nested, so the existing flat keys stay at top level
    with open(os.path.join(save_dir, f"dci_step{step}.json"), "w") as fp:
        json.dump(payload, fp, indent=2)
    return flat, scores


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
        proj_dim=args.contrastive_proj_dim,
        proj_hidden=args.contrastive_proj_hidden,
        encoder_architecture=args.encoder_architecture,
        encoder_head_hidden=args.encoder_head_hidden,
    ).to(device)
    if args.encoder_architecture == "resnet18":
        print(
            f"encoder: 3D ResNet-18, stride 32, GAP -> 512 -> {args.encoder_head_hidden} -> {args.latent_dim}; "
            f"{'separate' if model.separate_encoders else 'shared'} view backbone(s). "
            "The conv-only width, residual-layer and downscale flags are inactive.",
            flush=True,
        )
    if model.projector is not None:
        print(
            f"projection head: {args.content_channels} -> {args.contrastive_proj_hidden} -> "
            f"{args.contrastive_proj_dim} (loss runs here; probes read the {args.latent_dim}-d encoding)",
            flush=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sim_metric = torch.nn.CosineSimilarity(dim=-1)
    criterion = torch.nn.CrossEntropyLoss()

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(os.path.join(save_dir, "tensorboard"))
    except Exception:
        writer = None

    # Step-0 eval doubles as the untrained floor: same architecture, same seed, no training.
    # Per-factor R² needs it more than the block means do -- a localised factor can read
    # 0.2 from a random encoder, so the raw number alone cannot say whether it was learned.
    floor = None
    if args.floor_eval:
        model.eval()
        _, floor = evaluate(model, val_dataset, device, args, save_dir, 0, writer)
        model.train()

    step = 0
    running = {"loss": 0.0, "rank": 0.0, "n": 0}
    model.train()
    while step < args.train_steps:
        for batch in train_loader:
            if step >= args.train_steps:
                break
            x = torch.cat(batch["image"], dim=0).to(device)  # (2B, 1, res, res, res)

            _, _, feats, _, _, _, _, _ = model(x, pool_only=True, n_views=2)
            pooled = feats[0]  # (2B, latent_dim)
            loss = contrastive_loss(pooled, model, args, sim_metric, criterion)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running["loss"] += loss.item()
            running["rank"] += effective_rank(pooled[: pooled.shape[0] // 2, : args.content_channels])
            running["n"] += 1
            step += 1

            if step % args.log_every == 0:
                n = max(running["n"], 1)
                print(
                    f"step {step:6d} | contrastive {running['loss']/n:.4f} "
                    f"| content eff_rank {running['rank']/n:.2f}/{args.content_channels}",
                    flush=True,
                )
                if writer is not None:
                    writer.add_scalar("train/contrastive", running["loss"] / n, step)
                    writer.add_scalar("train/content_eff_rank", running["rank"] / n, step)
                running = {"loss": 0.0, "rank": 0.0, "n": 0}

            if step % args.eval_every == 0 or step == args.train_steps:
                model.eval()
                evaluate(model, val_dataset, device, args, save_dir, step, writer, floor=floor)
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
                model.train()

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    print(f"done. checkpoints + DCI logs in {save_dir}", flush=True)


if __name__ == "__main__":
    main()
