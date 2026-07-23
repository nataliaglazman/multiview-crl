"""Train the classic multiview AE (dense-vector bottleneck) on synthetic brain data.

Sibling of ``training.main_vae_synthetic`` — same experiment, same trusted metric
(``eval.dci.compute_dci_synthetic``), only the model differs:

  * ``models.ae.MultiviewAE`` — two encoders + a *dense* ``Linear`` bottleneck to a
    global latent vector (no spatial grid → no pooling degree of freedom), split by a
    fixed mask into a content / style block.  Deterministic: no ``mu``/``logvar``,
    no KL.  A *classic* AE, not variational.

Loss is per-view L1 reconstruction plus cross-view InfoNCE on the pooled content
block.  The baseline ablation is just ``--contrastive-weight 0`` (recon-only), exactly
as for the VAE.

``--encoder-only`` removes the decoder and the reconstruction term, leaving the
align-minus-entropy objective of Yao et al. (ICLR 2024, eq. 3.1/3.3) on this same
backbone.  Together the three settings form a controlled ablation of the one axis
their §5.2 isolates (they beat the Ada-GVAE autoencoder 0.42 vs 0.11 DCI and
attribute it to "avoiding reconstructing the image"):

  recon-only (--contrastive-weight 0)  |  recon + contrastive  |  contrastive-only

The scientific point vs the VAE: with ``latent_dim`` (``--latent-channels``) set near
the true generative dimensionality (``n_content + n_style``), the dense bottleneck is
forced to *compress* to the factors instead of leaving them redundant across ~512
spatial locations — so identifiability is read straight off the latent vector, with no
GAP/patch confound.

Example:
    python -m training.main_ae_synthetic --content-channels 9 --n-content 9 \
        --latent-channels 16 --train-steps 20000 --eval-every 2000
    # baseline ablation (recon-only):
    python -m training.main_ae_synthetic --model-id ae_synthetic_baseline \
        --contrastive-weight 0 --content-channels 9 --n-content 9
    # paper-faithful ablation (encoder-only, no decoder):
    python -m training.main_ae_synthetic --model-id ae_synthetic_enconly \
        --encoder-only --content-channels 9 --n-content 9
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
from models.ae import MultiviewAE


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--model-id", type=str, default="ae_synthetic")

    # Model / latent dimensionality (the knob under test).
    p.add_argument("--latent-channels", type=int, default=16, help="Total latent-vector size (content + style)")
    p.add_argument("--content-channels", type=int, default=9, help="Content units — set to the true n_content")
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--res-channels", type=int, default=32)
    p.add_argument("--nb-res-layers", type=int, default=2)
    p.add_argument("--downscale-factor", type=int, default=4, help="Spatial downscale (power of 2)")
    p.add_argument("--no-separate-encoders", action="store_true", help="Share one encoder across both views")

    # Loss weights.
    p.add_argument("--recon-weight", type=float, default=1.0)
    p.add_argument("--contrastive-weight", type=float, default=1.0, help="0 → recon-only baseline ablation")
    p.add_argument(
        "--encoder-only",
        action="store_true",
        help="Drop the decoder and the recon term → Yao et al. align−entropy objective",
    )
    p.add_argument(
        "--entropy-scope",
        type=str,
        default=None,
        choices=["content", "full"],
        help="Where the InfoNCE denominator (entropy) lives. 'content': Thm 3.2, both terms in "
        "the |C|-dim content space. 'full': eq. (3.3), alignment on content but entropy over the "
        "whole encoding. Defaults to 'full' under --encoder-only, else 'content'.",
    )
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument(
        "--cross-view-negs-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Contrastive negatives only from the other view (recommended with separate encoders)",
    )
    p.add_argument("--use-mask", action="store_true", help="Restrict the reconstruction loss to brain voxels")

    # Optimisation.
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16, help="Per-view batch size")
    p.add_argument("--train-steps", type=int, default=20000)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-cuda", action="store_true")

    # Synthetic dataset.
    p.add_argument("--res", type=int, default=32, help="Cubic resolution (power of 2 for clean recon)")
    p.add_argument("--n-content", type=int, default=9, help="True shared content factors")
    p.add_argument("--n-style", type=int, default=3, help="Per-view style factors")
    p.add_argument("--num-train-samples", type=int, default=2000)
    p.add_argument("--num-val-samples", type=int, default=200)
    p.add_argument("--synthetic-mode", type=str, default="pseudo_mri")
    p.add_argument(
        "--synthetic-normalize",
        type=str,
        default="fixed_reference",
        choices=["per_sample", "shared", "fixed_reference"],
    )
    p.add_argument("--synthetic-causal", action="store_true", help="Sample content from a causal SCM")
    return p.parse_args()


def make_dataset(args, mode, num_samples):
    return SyntheticBrainDataset(
        mode=mode,
        spatial_size=(args.res, args.res, args.res),
        cache=True,
        synthetic_mode=args.synthetic_mode,
        synthetic_seed=args.seed,
        synthetic_num_samples=num_samples,
        synthetic_n_content=args.n_content,
        synthetic_n_style=args.n_style,
        synthetic_style_scale=getattr(args, "synthetic_style_scale", 1.0),
        synthetic_content_scale=getattr(args, "synthetic_content_scale", 1.0),
        synthetic_normalize=args.synthetic_normalize,
        synthetic_causal=args.synthetic_causal,
    )


def reconstruction_loss(recon, x, mask=None):
    if mask is not None:
        diff = (recon - x).abs() * mask
        return diff.sum() / mask.sum().clamp_min(1.0)
    return torch.nn.functional.l1_loss(recon, x)


def contrastive_loss(pooled, content_channels, sim_metric, criterion, args):
    """Content alignment − entropy on the content block across the two views.

    ``entropy_scope="content"`` is plain InfoNCE restricted to the content block: both
    the numerator and the denominator live in the |C|-dim content space, matching Thm 3.2
    where the content encoders map straight onto (0,1)^|C|.

    ``"full"`` is eq. (3.3) instead — alignment on the *selected* block ``phi ⊘ r_k(x_k)``,
    entropy over the whole ``|S_k|``-dim encoding.  Without a decoder that distinction is
    load-bearing rather than cosmetic: under ``"content"`` nothing in the objective ever
    touches the style units, so they would sit at their random initialisation and every
    style-block metric would be scoring noise.
    """
    b = pooled.shape[0] // 2
    hz = torch.stack([pooled[:b], pooled[b:]], dim=0)  # (2, B, C)
    if args.entropy_scope == "full":
        loss = losses.split_infonce_loss(
            hz_align=hz[..., :content_channels],
            hz_entropy=hz,
            tau=args.tau,
            cross_view_negs_only=args.cross_view_negs_only,
        )
    else:
        loss = losses.infonce_loss(
            hz,
            sim_metric=sim_metric,
            criterion=criterion,
            tau=args.tau,
            projector=(lambda t: t),
            estimated_content_indices=[list(range(content_channels))],
            subsets=[(0, 1)],
            cross_view_negs_only=args.cross_view_negs_only,
        )
    return loss.squeeze()


def evaluate(model, val_dataset, device, args, save_dir, step, writer=None):
    print(f"  [eval] synthetic DCI @ step {step} ...", flush=True)
    results = dci.compute_dci_synthetic(
        encoder=model,
        dataset=val_dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=0,
        pooling="gap",
        per_encoder=not args.no_separate_encoders,
    )
    flat = dci.flatten_dci_results(results)

    def show(key):
        for k in flat:
            if k.endswith(key) and np.isfinite(flat[k]):
                print(f"      {k:60s} {flat[k]:.3f}", flush=True)

    print("    --- identifiability summary ---", flush=True)
    show("content->content/informativeness_ridge")
    show("content->content/block_mcc")
    show("content->style/informativeness_ridge")
    show("style->style/informativeness_ridge")
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

    # Resolve the coupled flags before anything reads them — settings.json must record
    # what actually ran, since eval.analyze_vae_identifiability rebuilds the model from it.
    if args.encoder_only:
        assert args.contrastive_weight > 0, "--encoder-only with --contrastive-weight 0 leaves no loss at all"
        args.recon_weight = 0.0
    if args.entropy_scope is None:
        args.entropy_scope = "full" if args.encoder_only else "content"

    save_dir = os.path.join(args.out_dir, args.model_id)
    os.makedirs(save_dir, exist_ok=True)
    # ``model_type`` lets eval.analyze_vae_identifiability.load_model rebuild the right class.
    with open(os.path.join(save_dir, "settings.json"), "w") as fp:
        json.dump({**vars(args), "model_type": "ae"}, fp, indent=2)

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"device: {device}", flush=True)
    print("Arguments:", flush=True)
    for k, v in vars(args).items():
        print(f"\t{k}: {v}", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = make_dataset(args, "train", args.num_train_samples)
    val_dataset = make_dataset(args, "val", args.num_val_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = MultiviewAE(
        in_channels=1,
        hidden_channels=args.hidden_channels,
        res_channels=args.res_channels,
        nb_res_layers=args.nb_res_layers,
        downscale_factor=args.downscale_factor,
        latent_dim=args.latent_channels,
        content_channels=args.content_channels,
        separate_encoders=not args.no_separate_encoders,
        res=args.res,
        encoder_only=args.encoder_only,
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
    running = {"loss": 0.0, "recon": 0.0, "contrast": 0.0, "n": 0}
    model.train()
    while step < args.train_steps:
        for batch in train_loader:
            if step >= args.train_steps:
                break
            x = torch.cat(batch["image"], dim=0).to(device)  # (2B, 1, res, res, res)
            mask = torch.cat(batch["mask"], dim=0).to(device) if args.use_mask else None

            recon, _diffs, feats, _, _, _, _, _ = model(
                x, return_recon=not args.encoder_only, pool_only=True, n_views=2
            )
            pooled = feats[0]  # (2B, latent_dim)

            contrast_l = contrastive_loss(pooled, args.content_channels, sim_metric, criterion, args)
            if args.encoder_only:
                recon_l = torch.zeros((), device=device)
                loss = args.contrastive_weight * contrast_l
            else:
                recon_l = reconstruction_loss(recon, x, mask)
                loss = args.recon_weight * recon_l + args.contrastive_weight * contrast_l

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running["loss"] += loss.item()
            running["recon"] += recon_l.item()
            running["contrast"] += contrast_l.item()
            running["n"] += 1
            step += 1

            if step % args.log_every == 0:
                n = max(running["n"], 1)
                print(
                    f"step {step:6d} | loss {running['loss']/n:.4f} "
                    f"| recon {running['recon']/n:.4f} "
                    f"| contrast {running['contrast']/n:.4f}",
                    flush=True,
                )
                if writer is not None:
                    for key in ("loss", "recon", "contrast"):
                        writer.add_scalar(f"train/{key}", running[key] / n, step)
                running = {"loss": 0.0, "recon": 0.0, "contrast": 0.0, "n": 0}

            if step % args.eval_every == 0 or step == args.train_steps:
                model.eval()
                evaluate(model, val_dataset, device, args, save_dir, step, writer)
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
                model.train()

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    print(f"done. checkpoints + DCI logs in {save_dir}", flush=True)


if __name__ == "__main__":
    main()
