"""Train 3D DenseNet heads on frozen VQ-VAE spatial features and compare.

Loads a trained run by reading ``<run-dir>/settings.json`` (to rebuild the
encoder identically) and ``<run-dir>/vqvae_model.pt`` (the weights).  The
encoder is frozen and run in eval mode; the deterministic top-k content mask
splits each level's feature map ``(B, hidden_channels, D, H, W)`` into content
and style slices.  One MONAI ``DenseNet121`` (3D) is trained per requested
``(level, kind)`` combination — letting you compare disease-classification
performance across content vs style vs all channels at one or more levels.

Usage:
    python -m eval.disease_classifier \\
        --run-dir runs/<model_id> \\
        --features content style all \\
        --feature-levels 0 1 \\
        --classifier-epochs 50 --classifier-batch-size 8 \\
        [--views t1|t2|both] [--unfreeze-encoder]

Outputs (per (level, kind) tag):
    <run-dir>/disease_classifier_<tag>_best.pt    — best DenseNet head
    <run-dir>/disease_classifier_metrics.json     — single combined metrics file
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from data import datasets
from models import vqvae as vqvae_mod

logger = logging.getLogger("disease_classifier")


# ---------------------------------------------------------------------------
# Settings restoration
# ---------------------------------------------------------------------------


def _load_run_args(run_dir: str) -> SimpleNamespace:
    settings_path = os.path.join(run_dir, "settings.json")
    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"No settings.json in {run_dir}")
    with open(settings_path) as f:
        d = json.load(f)
    args = SimpleNamespace(**d)
    # Re-attach DATASETCLASS (stripped at save time).
    name = args.dataset_name
    if name in ("adni", "ADNI_registered", "ADNI_stripped", "ADNI_stripped_masks", "custom"):
        args.DATASETCLASS = datasets.MyCustomDataset
    elif name == "synthetic":
        args.DATASETCLASS = datasets.SyntheticBrainDataset
    else:
        raise ValueError(f"disease_classifier only supports custom/ADNI/synthetic datasets, got {name}")
    return args


# ---------------------------------------------------------------------------
# Encoder construction (mirrors training/main_multimodal.py:1189 — keep in sync
# if VQVAE constructor signature changes).
# ---------------------------------------------------------------------------


def _build_vqvae(args: SimpleNamespace) -> vqvae_mod.VQVAE:
    return vqvae_mod.VQVAE(
        in_channels=1,
        hidden_channels=args.vqvae_hidden_channels,
        res_channels=args.vqvae_res_channels,
        nb_res_layers=2,
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
    )


def _load_encoder(run_dir: str, args: SimpleNamespace, device) -> vqvae_mod.VQVAE:
    ckpt_path = os.path.join(run_dir, "vqvae_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No vqvae_model.pt in {run_dir}")
    model = _build_vqvae(args)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["encoders"]

    # Strip DataParallel "module." and torch.compile "_orig_mod." prefixes.
    def _strip(k: str) -> str:
        for pfx in ("module.", "_orig_mod."):
            while k.startswith(pfx):
                k = k[len(pfx) :]
        return k

    state = {_strip(k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"  Encoder missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        logger.warning(f"  Encoder unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def _build_datasets(args: SimpleNamespace):
    common = dict(
        data_dir=args.datapath,
        change_lists=args.change_lists,
        spacing=getattr(args, "image_spacing", 2.0),
        crop_margin=getattr(args, "crop_margin", 0),
        spatial_size=getattr(args, "spatial_size", None),
        labels_path=getattr(args, "labels_path", None),
        masks_dir=getattr(args, "masks_dir", None),
        asymmetric_aug=False,  # off for probing — eliminate aug-induced noise
        shared_brain_mask=getattr(args, "shared_brain_mask", False),
        cache=getattr(args, "cache_dataset", False),
        cache_dir=getattr(args, "cache_dir", None),
    )
    train_ds = args.DATASETCLASS(mode="train", **common)
    val_ds = args.DATASETCLASS(mode="val", **common)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Feature extraction (frozen path)
# ---------------------------------------------------------------------------


@dataclass
class FeatureSpec:
    level: int = 0
    kind: str = "content"  # "content" | "style" | "all"

    @property
    def label(self) -> str:
        return f"{self.kind}_L{self.level}"


def _extract_features(encoder: vqvae_mod.VQVAE, img: torch.Tensor, view_idx: int, spec: FeatureSpec) -> torch.Tensor:
    """Run the encoder on one view, return ``(B, k, D, H, W)`` feature map.

    ``spec.kind`` selects which channels of the level-``spec.level`` map to keep:
    ``content`` (Gumbel-mask-selected), ``style`` (complement), or ``all``.
    """
    out = encoder(img, return_recon=False, pool_only=False, view_idx=view_idx)
    encoder_features = out[2]
    soft_masks = out[6]
    feat = encoder_features[spec.level]  # (B, hidden_channels, D, H, W)
    if spec.kind == "all":
        return feat
    mask = soft_masks.get(spec.level)
    if mask is None:
        # No content/style split was configured for this level — fall back to all.
        return feat
    if isinstance(mask, tuple):
        mask = mask[view_idx]
    content_idx = torch.where(mask.bool())[-1]
    if spec.kind == "content":
        return feat[:, content_idx]
    if spec.kind == "style":
        style_idx = torch.where(~mask.bool())[-1]
        return feat[:, style_idx]
    raise ValueError(f"Unknown FeatureSpec.kind={spec.kind!r}")


# ---------------------------------------------------------------------------
# Classifier model
# ---------------------------------------------------------------------------


def _build_classifier(in_channels: int, num_classes: int) -> nn.Module:
    return DenseNet121(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        dropout_prob=0.2,
    )


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def _resolve_views(name: str) -> list[int]:
    name = name.lower()
    if name == "t1":
        return [0]
    if name == "t2":
        return [1]
    if name == "both":
        return [0, 1]
    raise ValueError(f"--views must be t1|t2|both, got {name}")


def _batch_to_features(
    batch, encoder, spec: FeatureSpec, view_idxs: list[int], device, encoder_grad: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a dataloader batch into ``(features, labels)`` ready for the head.

    Stacks views along the batch dimension (label is replicated per view)."""
    images = batch["image"]  # list of (B, 1, D, H, W) — one tensor per view
    labels = batch["label"].to(device).long()

    feats_list, lbls_list = [], []
    ctx = torch.enable_grad() if encoder_grad else torch.no_grad()
    with ctx:
        for v in view_idxs:
            img = images[v].to(device, non_blocking=True)
            f = _extract_features(encoder, img, view_idx=v, spec=spec)
            feats_list.append(f)
            lbls_list.append(labels)
    return torch.cat(feats_list, dim=0), torch.cat(lbls_list, dim=0)


def _epoch(
    encoder,
    head,
    loader,
    opt,
    spec,
    view_idxs,
    device,
    train: bool,
    encoder_grad: bool,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    head.train(train)
    if encoder_grad:
        encoder.train(train)
    else:
        encoder.eval()

    losses = []
    all_pred, all_true = [], []
    for batch in loader:
        feats, lbls = _batch_to_features(batch, encoder, spec, view_idxs, device, encoder_grad)
        # Drop samples without a valid label (-1 sentinel from dataset).
        keep = lbls >= 0
        if keep.sum() == 0:
            continue
        feats, lbls = feats[keep], lbls[keep]

        logits = head(feats)
        loss = F.cross_entropy(logits, lbls)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        losses.append(float(loss.detach()))
        all_pred.append(logits.argmax(1).detach().cpu().numpy())
        all_true.append(lbls.detach().cpu().numpy())

    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    y_true = np.concatenate(all_true) if all_true else np.array([])
    bacc = balanced_accuracy_score(y_true, y_pred) if len(y_true) else float("nan")
    return float(np.mean(losses) if losses else float("nan")), bacc, y_true, y_pred


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Directory containing settings.json + vqvae_model.pt")
    p.add_argument("--classifier-epochs", type=int, default=50)
    p.add_argument("--classifier-lr", type=float, default=1e-4)
    p.add_argument("--classifier-batch-size", type=int, default=8)
    p.add_argument("--classifier-weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--feature-levels",
        type=int,
        nargs="+",
        default=[0],
        help="Encoder level(s) to probe. One classifier is trained per (level, kind).",
    )
    p.add_argument(
        "--features",
        nargs="+",
        default=["content", "style"],
        choices=["content", "style", "all"],
        help="Which channel slice(s) to compare. Trains one classifier per choice.",
    )
    p.add_argument("--views", default="both", choices=["t1", "t2", "both"])
    p.add_argument("--unfreeze-encoder", action="store_true")
    p.add_argument("--encoder-lr", type=float, default=1e-5, help="LR for encoder params if --unfreeze-encoder")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-name", default="disease_classifier", help="Filename prefix for outputs in --run-dir")
    return p.parse_args()


def main(cli):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    device = torch.device(cli.device)

    logger.info(f"Loading settings from {cli.run_dir}")
    run_args = _load_run_args(cli.run_dir)

    logger.info("Building VQ-VAE encoder")
    encoder = _load_encoder(cli.run_dir, run_args, device)
    if not cli.unfreeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
    logger.info(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    logger.info("Building datasets")
    train_ds, val_ds = _build_datasets(run_args)
    logger.info(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    view_idxs = _resolve_views(cli.views)

    # Discover label set from the training dataset's items.
    train_labels = np.array([it["label"] for it in train_ds.items if it.get("label", -1) >= 0])
    num_classes = int(train_labels.max()) + 1 if len(train_labels) else 0
    if num_classes < 2:
        raise RuntimeError(f"Need ≥ 2 classes; found {num_classes} in {run_args.labels_path}")
    logger.info(f"  num_classes={num_classes}")

    # Loaders are shared across all (level, kind) runs.
    train_loader = DataLoader(
        train_ds,
        batch_size=cli.classifier_batch_size,
        shuffle=True,
        num_workers=cli.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cli.classifier_batch_size,
        shuffle=False,
        num_workers=cli.workers,
        pin_memory=True,
    )

    # Probe once per level to discover channel counts (cheap — single batch).
    probe_loader = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0)
    probe_batch = next(iter(probe_loader))

    comparison: dict[str, dict] = {}
    for level in cli.feature_levels:
        for kind in cli.features:
            spec = FeatureSpec(level=level, kind=kind)
            tag = spec.label
            logger.info("")
            logger.info(f"=== Probing {tag} ===")

            # Reseed per run so each classifier sees the same init / shuffle order.
            torch.manual_seed(cli.seed)
            np.random.seed(cli.seed)

            with torch.no_grad():
                feats, _ = _batch_to_features(probe_batch, encoder, spec, view_idxs[:1], device, encoder_grad=False)
            in_channels = feats.shape[1]
            feat_shape = tuple(feats.shape[2:])
            if in_channels == 0:
                logger.warning(f"  {tag}: 0 channels selected — skipping")
                continue
            logger.info(f"  Feature shape: ({in_channels}, {feat_shape})")

            head = _build_classifier(in_channels=in_channels, num_classes=num_classes).to(device)
            logger.info(f"  Head params: {sum(p.numel() for p in head.parameters()):,}")

            param_groups = [{"params": head.parameters(), "lr": cli.classifier_lr}]
            if cli.unfreeze_encoder:
                param_groups.append({"params": encoder.parameters(), "lr": cli.encoder_lr})
            opt = torch.optim.AdamW(param_groups, weight_decay=cli.classifier_weight_decay)

            best_bacc = -1.0
            best_report: dict = {}
            best_cm: list = []
            best_epoch = 0
            history: list = []
            best_path = os.path.join(cli.run_dir, f"{cli.out_name}_{tag}_best.pt")

            for epoch in range(1, cli.classifier_epochs + 1):
                tr_loss, tr_bacc, _, _ = _epoch(
                    encoder,
                    head,
                    train_loader,
                    opt,
                    spec,
                    view_idxs,
                    device,
                    train=True,
                    encoder_grad=cli.unfreeze_encoder,
                )
                va_loss, va_bacc, y_true, y_pred = _epoch(
                    encoder,
                    head,
                    val_loader,
                    opt,
                    spec,
                    view_idxs,
                    device,
                    train=False,
                    encoder_grad=False,
                )
                logger.info(
                    f"  [{tag}] epoch {epoch:3d}/{cli.classifier_epochs} | "
                    f"train loss={tr_loss:.4f} bacc={tr_bacc:.3f} | "
                    f"val loss={va_loss:.4f} bacc={va_bacc:.3f}"
                )
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": tr_loss,
                        "train_bacc": tr_bacc,
                        "val_loss": va_loss,
                        "val_bacc": va_bacc,
                    }
                )

                if va_bacc > best_bacc:
                    best_bacc = va_bacc
                    best_epoch = epoch
                    torch.save(
                        {
                            "head_state_dict": head.state_dict(),
                            "epoch": epoch,
                            "val_bacc": va_bacc,
                            "in_channels": in_channels,
                            "num_classes": num_classes,
                            "feature_spec": vars(spec),
                            "views": cli.views,
                        },
                        best_path,
                    )
                    best_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    best_cm = confusion_matrix(y_true, y_pred).tolist()

            logger.info(f"  [{tag}] best val bACC = {best_bacc:.4f} @ epoch {best_epoch} → {best_path}")
            comparison[tag] = {
                "level": level,
                "kind": kind,
                "in_channels": in_channels,
                "feat_shape": list(feat_shape),
                "best_epoch": best_epoch,
                "best_val_bacc": best_bacc,
                "best_classification_report": best_report,
                "best_confusion_matrix": best_cm,
                "history": history,
                "ckpt_path": best_path,
            }

            del head, opt
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Final summary
    logger.info("")
    logger.info("=== Comparison summary (best val balanced accuracy) ===")
    for tag, res in sorted(comparison.items(), key=lambda kv: -kv[1]["best_val_bacc"]):
        logger.info(f"  {tag:<14} bACC={res['best_val_bacc']:.4f}  channels={res['in_channels']}")

    metrics_path = os.path.join(cli.run_dir, f"{cli.out_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "comparison": comparison,
                "summary": {tag: res["best_val_bacc"] for tag, res in comparison.items()},
                "config": vars(cli),
            },
            f,
            indent=2,
        )
    logger.info(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main(parse_args())
