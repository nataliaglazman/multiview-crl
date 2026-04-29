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
        --task ad_cn \\
        --classifier-epochs 50 --classifier-batch-size 8 \\
        [--views t1|t2|both]

Speed: the encoder runs ONCE over train + val and the resulting spatial maps
are held in CPU RAM (fp16 by default).  All (level, kind) heads then train on
those cached tensors, so encoder cost is paid only once regardless of how many
combinations are compared.  Memory budget per cached level ≈
``N_rows × hidden_channels × D × H × W × 2 bytes`` (fp16).  At level 0 with
default ADNI shapes this is ~30 MB per row — keep an eye on it.

Tasks:
  --task all    — multi-class over every Group label found in the labels CSV
  --task ad_cn  — binary AD vs CN (CN→0, AD→1); MCI rows are dropped

Outputs (per (level, kind) tag):
    <run-dir>/disease_classifier_<task>__<tag>_best.pt — best DenseNet head
    <run-dir>/disease_classifier_metrics.json          — single combined metrics file
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


# ---------------------------------------------------------------------------
# Feature precomputation — encoder runs ONCE per dataset, results held in RAM.
# This is the main speedup: every (level, kind) head trains on cached tensors
# instead of re-running the encoder each epoch.
# ---------------------------------------------------------------------------


@dataclass
class FeatureCache:
    """All-channel features per requested level, plus labels and per-view masks."""

    feats: dict[int, torch.Tensor]  # level → (N, C, D, H, W) fp16/fp32 on CPU
    labels: torch.Tensor  # (N,) long
    view_ids: torch.Tensor  # (N,) long, 0=T1, 1=T2
    content_masks: dict[int, dict[int, torch.Tensor]]  # level → view_idx → bool (C,)

    def slice_channels(self, level: int, kind: str, view: int) -> torch.Tensor:
        """Return the channel-index tensor selecting `kind` from level `level`."""
        if kind == "all":
            C = self.feats[level].shape[1]
            return torch.arange(C)
        mask = self.content_masks.get(level, {}).get(view)
        if mask is None:
            C = self.feats[level].shape[1]
            return torch.arange(C)
        if kind == "content":
            return torch.where(mask)[0]
        if kind == "style":
            return torch.where(~mask)[0]
        raise ValueError(kind)


@torch.no_grad()
def _precompute_features(
    encoder,
    dataset,
    levels: list[int],
    view_idxs: list[int],
    device,
    batch_size: int,
    workers: int,
    dtype: torch.dtype,
) -> FeatureCache:
    """One pass of the encoder; caches per-level spatial maps to CPU memory."""
    encoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    feats_by_level: dict[int, list[torch.Tensor]] = {l: [] for l in levels}
    labels_list: list[torch.Tensor] = []
    view_list: list[torch.Tensor] = []
    masks: dict[int, dict[int, torch.Tensor]] = {l: {} for l in levels}

    n_done = 0
    for batch in loader:
        labels = batch["label"].long()
        for v in view_idxs:
            img = batch["image"][v].to(device, non_blocking=True)
            out = encoder(img, return_recon=False, pool_only=False, view_idx=v)
            encoder_features, soft_masks = out[2], out[6]
            for L in levels:
                feats_by_level[L].append(encoder_features[L].to(dtype).cpu())
                if L not in masks or v not in masks[L]:
                    m = soft_masks.get(L)
                    if m is not None:
                        if isinstance(m, tuple):
                            m = m[v]
                        masks[L][v] = m.bool().cpu()
            labels_list.append(labels)
            view_list.append(torch.full_like(labels, v))
        n_done += img.size(0)
        if n_done % (10 * batch_size) < batch_size:
            logger.info(f"  precomputed {n_done}/{len(dataset)}")

    feats = {L: torch.cat(feats_by_level[L], 0) for L in levels}
    labels_all = torch.cat(labels_list, 0)
    views_all = torch.cat(view_list, 0)
    return FeatureCache(feats=feats, labels=labels_all, view_ids=views_all, content_masks=masks)


def _filter_cache(cache: FeatureCache, valid_mask: torch.Tensor) -> FeatureCache:
    """Apply a row-mask to all per-row tensors in the cache (in-place new)."""
    return FeatureCache(
        feats={L: t[valid_mask] for L, t in cache.feats.items()},
        labels=cache.labels[valid_mask],
        view_ids=cache.view_ids[valid_mask],
        content_masks=cache.content_masks,
    )


# ---------------------------------------------------------------------------
# Cached-feature epoch
# ---------------------------------------------------------------------------


def _epoch_cached(
    head,
    feats_full: torch.Tensor,
    labels: torch.Tensor,
    ch_idx: torch.Tensor,
    opt,
    device,
    batch_size: int,
    train: bool,
    use_amp: bool,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    head.train(train)
    N = feats_full.shape[0]
    order = torch.randperm(N) if train else torch.arange(N)

    losses, preds, trues = [], [], []
    autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda")

    for start in range(0, N, batch_size):
        idx = order[start : start + batch_size]
        # Slice channels lazily (cheap on CPU), then move to GPU + cast.
        x = feats_full[idx][:, ch_idx].to(device, non_blocking=True).float()
        y = labels[idx].to(device, non_blocking=True)

        with autocast_ctx:
            logits = head(x)
            loss = F.cross_entropy(logits, y)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        losses.append(float(loss.detach()))
        preds.append(logits.argmax(1).detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

    y_pred = np.concatenate(preds) if preds else np.array([])
    y_true = np.concatenate(trues) if trues else np.array([])
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
    p.add_argument(
        "--task",
        default="all",
        choices=["all", "ad_cn"],
        help="`all`: multi-class over every Group label found in the CSV. "
        "`ad_cn`: binary AD vs CN, MCI rows are dropped.",
    )
    p.add_argument(
        "--unfreeze-encoder",
        action="store_true",
        help="(Disabled in cached mode.) End-to-end fine-tuning is not supported when "
        "features are cached; this flag is kept for back-compat and is currently a no-op.",
    )
    p.add_argument(
        "--cache-dtype",
        default="fp16",
        choices=["fp16", "fp32"],
        help="Dtype for the on-CPU feature cache. fp16 halves memory; cast back to fp32 for the head.",
    )
    p.add_argument("--no-amp", action="store_true", help="Disable autocast in the head training loop.")
    p.add_argument("--extract-batch-size", type=int, default=4, help="Batch size for the one-time encoder pass.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-name", default="disease_classifier", help="Filename prefix for outputs in --run-dir")
    return p.parse_args()


def _resolve_task(task: str, train_ds, val_ds) -> tuple[dict, str]:
    """Return (label_remap: orig_int → new_int, task_label_name).

    ``orig_int`` keys NOT in the remap are dropped from training/eval.
    """
    label_map = getattr(train_ds, "label_map", None) or getattr(val_ds, "label_map", None)
    if label_map is None:
        # Fallback: assume identity over the integer labels found in the data.
        ints = sorted({it["label"] for it in train_ds.items if it.get("label", -1) >= 0})
        label_map = {str(i): i for i in ints}

    if task == "all":
        # Identity: every original label maps to itself.
        remap = {v: v for v in label_map.values()}
        return remap, ",".join(label_map.keys())

    if task == "ad_cn":
        # Case-insensitive match: any group whose name starts with AD or CN.
        ad_int = next((v for k, v in label_map.items() if k.upper().startswith("AD")), None)
        cn_int = next((v for k, v in label_map.items() if k.upper().startswith("CN")), None)
        if ad_int is None or cn_int is None:
            raise RuntimeError(
                f"--task ad_cn requires AD and CN groups in the labels CSV; " f"found {list(label_map.keys())}"
            )
        # CN → 0, AD → 1. All other groups dropped.
        return {cn_int: 0, ad_int: 1}, "CN_vs_AD"

    raise ValueError(task)


def main(cli):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    device = torch.device(cli.device)

    logger.info(f"Loading settings from {cli.run_dir}")
    run_args = _load_run_args(cli.run_dir)

    logger.info("Building VQ-VAE encoder")
    encoder = _load_encoder(cli.run_dir, run_args, device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    if cli.unfreeze_encoder:
        logger.warning("  --unfreeze-encoder is a no-op in cached mode (encoder stays frozen).")
    logger.info(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    logger.info("Building datasets")
    train_ds, val_ds = _build_datasets(run_args)
    logger.info(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    view_idxs = _resolve_views(cli.views)
    label_remap, task_name = _resolve_task(cli.task, train_ds, val_ds)
    logger.info(f"  Task: {cli.task} ({task_name}) | label_remap={label_remap}")

    cache_dtype = torch.float16 if cli.cache_dtype == "fp16" else torch.float32

    # ---- One-time encoder pass over train+val ----
    logger.info("Precomputing features (single encoder pass) ...")
    levels = sorted(set(cli.feature_levels))
    train_cache = _precompute_features(
        encoder,
        train_ds,
        levels,
        view_idxs,
        device,
        batch_size=cli.extract_batch_size,
        workers=cli.workers,
        dtype=cache_dtype,
    )
    val_cache = _precompute_features(
        encoder,
        val_ds,
        levels,
        view_idxs,
        device,
        batch_size=cli.extract_batch_size,
        workers=cli.workers,
        dtype=cache_dtype,
    )

    # Apply task filter + label remap.
    def _apply_task(cache: FeatureCache) -> FeatureCache:
        valid = torch.tensor([int(l.item()) in label_remap for l in cache.labels])
        cache = _filter_cache(cache, valid)
        # Remap to contiguous task labels.
        new_labels = torch.tensor([label_remap[int(l.item())] for l in cache.labels], dtype=torch.long)
        cache.labels = new_labels
        return cache

    train_cache = _apply_task(train_cache)
    val_cache = _apply_task(val_cache)
    num_classes = int(max(train_cache.labels.max().item(), val_cache.labels.max().item())) + 1
    logger.info(
        f"  After task filter: train rows={len(train_cache.labels)} "
        f"val rows={len(val_cache.labels)} num_classes={num_classes}"
    )
    if len(train_cache.labels) == 0 or len(val_cache.labels) == 0:
        raise RuntimeError("Empty train or val set after task filter.")

    # Encoder no longer needed — free GPU memory for the head.
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    use_amp = not cli.no_amp

    comparison: dict[str, dict] = {}
    for level in levels:
        for kind in cli.features:
            spec = FeatureSpec(level=level, kind=kind)
            tag = f"{task_name}__{spec.label}"
            logger.info("")
            logger.info(f"=== Probing {tag} ===")

            torch.manual_seed(cli.seed)
            np.random.seed(cli.seed)

            # Channel selection per view. If T1 and T2 use different content
            # channels (separate_encoders + per-view masks), we can't share a
            # single channel index across rows; in that case route per view by
            # building a virtual channel-aligned tensor.
            v0_idx = train_cache.slice_channels(level, kind, view=view_idxs[0])
            in_channels = len(v0_idx)
            if in_channels == 0:
                logger.warning(f"  {tag}: 0 channels selected — skipping")
                continue

            # Build per-row channel-sliced tensors. Per-view masks may differ;
            # we handle by gathering separately and concatenating row-aligned.
            def _slice_cache(cache: FeatureCache) -> torch.Tensor:
                feat = cache.feats[level]
                out = torch.empty((feat.shape[0], in_channels, *feat.shape[2:]), dtype=feat.dtype)
                for v in view_idxs:
                    rows = cache.view_ids == v
                    ch = cache.slice_channels(level, kind, view=v)
                    if len(ch) != in_channels:
                        raise RuntimeError(
                            f"Per-view channel-count mismatch at level {level} "
                            f"({len(ch)} vs {in_channels}); content channel allocation "
                            "differs between views."
                        )
                    out[rows] = feat[rows][:, ch]
                return out

            train_feats = _slice_cache(train_cache)
            val_feats = _slice_cache(val_cache)
            feat_shape = tuple(train_feats.shape[2:])
            logger.info(
                f"  Feature shape per sample: ({in_channels}, {feat_shape})  "
                f"train rows={len(train_feats)} val rows={len(val_feats)}"
            )

            head = _build_classifier(in_channels=in_channels, num_classes=num_classes).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=cli.classifier_lr, weight_decay=cli.classifier_weight_decay)
            logger.info(f"  Head params: {sum(p.numel() for p in head.parameters()):,}")

            best_bacc = -1.0
            best_report: dict = {}
            best_cm: list = []
            best_epoch = 0
            history: list = []
            best_path = os.path.join(cli.run_dir, f"{cli.out_name}_{tag}_best.pt")

            for epoch in range(1, cli.classifier_epochs + 1):
                tr_loss, tr_bacc, _, _ = _epoch_cached(
                    head,
                    train_feats,
                    train_cache.labels,
                    torch.arange(in_channels),
                    opt,
                    device,
                    cli.classifier_batch_size,
                    train=True,
                    use_amp=use_amp,
                )
                with torch.no_grad():
                    va_loss, va_bacc, y_true, y_pred = _epoch_cached(
                        head,
                        val_feats,
                        val_cache.labels,
                        torch.arange(in_channels),
                        opt,
                        device,
                        cli.classifier_batch_size,
                        train=False,
                        use_amp=use_amp,
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
                            "task": cli.task,
                        },
                        best_path,
                    )
                    best_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    best_cm = confusion_matrix(y_true, y_pred).tolist()

            logger.info(f"  [{tag}] best val bACC = {best_bacc:.4f} @ epoch {best_epoch} → {best_path}")
            comparison[tag] = {
                "task": cli.task,
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

            del head, opt, train_feats, val_feats
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
