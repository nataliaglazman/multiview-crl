# isort: skip_file
# Multiview Contrastive Representation Learning — main training script.
#
# High-level structure
# --------------------
# config.py          parse_args / update_args / compute_gt_idx
# logging_setup.py   setup_logging
# checkpointing.py   save_checkpoint / load_checkpoint / save_emergency_checkpoint
# visualisation.py   save_decoded_images / save_vqvae_decoded_images
# evaluation.py      val_step / get_data / eval_step
# main_multimodal.py train_step + main (this file)

import collections
import csv
import json
import math
import os
import random
import traceback
import uuid
import warnings
from datetime import datetime

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import models.vae as vae
import models.vqvae as vqvae
import utils.utils as utils
from data.infinite_iterator import InfiniteIterator
from eval.evaluation import eval_step, get_data
from training.losses import BaselineLoss, barlow_twins_loss, infonce_loss, moco_loss, patch_infonce_loss, vicreg_loss
from models.encoders import TextEncoder2D
from utils.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    save_emergency_checkpoint,
)
from utils.config import parse_args, update_args
from utils.logging_setup import setup_logging
from utils.visualisation import save_decoded_images, save_vqvae_decoded_images

device_ids = [0]


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_step(
    data,
    encoders,
    decoders,
    loss_func,
    optimizer,
    params,
    args,
    scaler=None,
    recon_loss_fn=None,
    accumulation_step=0,
    total_accumulation_steps=1,
    moco_loss_func=None,
    step=0,
    force_compute_recon=None,
    patch_loss_func=None,
):
    """
    Perform a single forward + (optionally) backward pass.

    Args:
        data: Batch dictionary from the DataLoader.
        encoders: List of encoder models (or a single MoCoEncoder-wrapped VQVAE).
        decoders: List of decoder models (empty for VQ-VAE mode).
        loss_func: InfoNCE loss callable ``(hz, content_indices, subsets) -> loss``.
        optimizer: Optimizer (``None`` during validation).
        params: Iterable of parameters to clip gradients for.
        args: Parsed argument namespace.
        scaler: ``GradScaler`` for AMP (``None`` disables AMP).
        recon_loss_fn: Reconstruction loss instance (instantiated on first call if ``None``).
        accumulation_step: Index within the current accumulation window (0-based).
        total_accumulation_steps: Total number of mini-steps per optimizer update.
        moco_loss_func: MoCo loss callable (``None`` → standard in-batch InfoNCE).
        force_compute_recon: If not None, overrides the skip_recon_ratio dice roll
            (used to keep memory consistent across gradient accumulation micro-steps).

    Returns:
        tuple: ``(total_loss, contrastive_loss, recon_loss, vq_loss, estimated_content_indices)``
    """
    _diag = {}  # MoCo stale-queue diagnostics (populated below when applicable)

    if optimizer is not None and accumulation_step == 0:
        # set_to_none=True frees gradient tensors rather than zeroing them (~1× param memory saved)
        optimizer.zero_grad(set_to_none=True)

    if recon_loss_fn is None:
        recon_loss_fn = BaselineLoss().to(next(encoders[0].parameters()).device)

    use_amp = scaler is not None
    device = next(encoders[0].parameters()).device

    with autocast("cuda", enabled=use_amp):
        samples = data["image"]
        n_views = len(samples)
        images = torch.concat(samples, 0).to(device, non_blocking=True)  # (n_views * B, 1, D, H, W)
        input_shape = images.shape[2:]

        # ------------------------------------------------------------------
        # VQ-VAE-2 path
        # ------------------------------------------------------------------
        if args.encoder_type == "vqvae":
            vqvae_model = encoders[0]

            if force_compute_recon is not None:
                compute_recon = force_compute_recon
            else:
                skip_recon_ratio = getattr(args, "skip_recon_ratio", 0.0)
                compute_recon = (skip_recon_ratio == 0.0) or (torch.rand(1).item() > skip_recon_ratio)

            _patch_grid = tuple(args.patch_grid) if getattr(args, "patch_contrastive", False) else None

            (
                recon,
                diffs,
                encoder_outputs,
                estimated_content_indices,
                _,
                _,
                fwd_soft_content_masks,
                _,  # style_id_outputs
            ) = vqvae_model(
                images,
                return_recon=compute_recon,
                pool_only=True,
                n_views=n_views,
                subsets=args.subsets,
                patch_grid=_patch_grid,
            )

            # Compute momentum-encoder key embeddings BEFORE deleting images.
            # During mask warmup, disable MoCo so in-batch InfoNCE is used
            # instead — this lets the learned mask stabilise before stale
            # queue negatives can corrupt the contrastive signal.
            use_moco = getattr(args, "use_moco", False)
            _mask_warmup_steps = getattr(args, "mask_warmup_steps", 0)
            _in_mask_warmup = _mask_warmup_steps > 0 and step <= _mask_warmup_steps
            if _in_mask_warmup:
                use_moco = False
            if use_moco:
                from models.vqvae import MoCoEncoder

                assert isinstance(
                    vqvae_model, MoCoEncoder
                ), "MoCo requested but encoders[0] is not a MoCoEncoder instance."
                with torch.no_grad():
                    key_outputs = vqvae_model.encode_keys(images, n_views=n_views, patch_grid=_patch_grid)

            _recon_start = getattr(args, "recon_loss_start_step", 0)
            _recon_active = step >= _recon_start or getattr(args, "_resumed_past_recon_start", False)
            if compute_recon and recon is not None and _recon_active:
                # Safety net: the model now interpolates internally, but guard
                # against size mismatch in case decode_codes or an older
                # checkpoint path bypasses it.
                if recon.shape[2:] != input_shape:
                    recon = F.interpolate(recon, size=input_shape, mode="trilinear", align_corners=False)
                recon_loss = (
                    recon_loss_fn(
                        {"reconstruction": [recon], "quantization_losses": diffs},
                        images,
                    )
                    * args.scale_recon_loss
                )
                del recon, images
            else:
                recon_loss = torch.zeros(1, device=device)
                del images
                if recon is not None:
                    del recon

            vq_loss = sum(diffs) * args.vq_commitment_weight
            del diffs

            total_contrastive_loss = torch.zeros(1, device=device)
            level_losses = []
            default_content_ratio = len(args.content_indices[0]) / (
                len(args.content_indices[0]) + len(args.style_indices)
            )

            # Unwrap DataParallel / MoCoEncoder to reach the bare VQVAE so we
            # can read channel_logits (fix #1 / #4).
            _raw_vqvae = vqvae_model.online if hasattr(vqvae_model, "online") else vqvae_model
            _raw_vqvae = _raw_vqvae.module if hasattr(_raw_vqvae, "module") else _raw_vqvae

            # Per-level content channel counts from the model (set by --content-ratios)
            _content_ch_per_level = getattr(_raw_vqvae, "content_channels_per_level", {})

            for level_idx, enc_pooled in enumerate(encoder_outputs):
                # Global pool: enc_pooled is (2B, C) → hz_level (n_views, B, C)
                # Patch pool:  enc_pooled is (2B, C, P) → hz_level (n_views, B, C, P)
                hz_level = enc_pooled.reshape(n_views, -1, *enc_pooled.shape[1:])
                _is_patch = hz_level.ndim == 4  # has patch dimension
                n_channels = hz_level.shape[2] if _is_patch else hz_level.shape[-1]
                # Use per-level content_channels if available, otherwise fall back to ratio
                if level_idx in _content_ch_per_level:
                    content_size = _content_ch_per_level[level_idx]
                else:
                    content_size = max(1, int(default_content_ratio * n_channels))

                soft_content_mask = None

                if level_idx in fwd_soft_content_masks:
                    mask_or_tuple = fwd_soft_content_masks[level_idx]

                    if isinstance(mask_or_tuple, tuple):
                        mask_v0, mask_v1 = mask_or_tuple
                        idx_v0 = torch.where(mask_v0.bool())[-1]
                        idx_v1 = torch.where(mask_v1.bool())[-1]
                        k_content = int(mask_v0.sum().item())

                        # Pre-mask and extract k-dim content per view
                        if _is_patch:
                            # hz_level: (n_views, B, C, P), mask: (1, C)
                            hz_v0_content = (hz_level[0] * mask_v0.unsqueeze(-1))[:, idx_v0, :]  # (B, k, P)
                            hz_v1_content = (hz_level[1] * mask_v1.unsqueeze(-1))[:, idx_v1, :]  # (B, k, P)
                        else:
                            hz_v0_content = (hz_level[0] * mask_v0)[:, idx_v0]  # (B, k)
                            hz_v1_content = (hz_level[1] * mask_v1)[:, idx_v1]  # (B, k)
                        hz_content = torch.stack([hz_v0_content, hz_v1_content], dim=0)

                        # All k dims are now content (already selected)
                        level_content_indices = [list(range(k_content))] * len(args.subsets)
                        # Only set estimated_content_indices on the first masked
                        # level so later levels don't overwrite it (fix #6).
                        if estimated_content_indices is None:
                            estimated_content_indices = [idx_v0.tolist()]  # view-0 for backward compat

                        if use_moco:
                            assert not _is_patch, (
                                "Per-view mask MoCo path does not support patch-contrastive yet. "
                                "Use --mask-mode fixed or --mask-mode learned (without per-view masks) instead."
                            )
                            key_pooled = key_outputs[level_idx]
                            k_level = key_pooled.reshape(n_views, -1, *key_pooled.shape[1:])
                            # Pre-mask momentum keys the same way
                            k_v0_content = (k_level[0] * mask_v0.detach())[:, idx_v0]
                            k_v1_content = (k_level[1] * mask_v1.detach())[:, idx_v1]

                            q_snap_v0 = vqvae_model.queues[level_idx].clone().detach()
                            q_snap_v1 = vqvae_model.queues_v1[level_idx].clone().detach()
                            _norm_eps = 1e-6  # avoid NaN when masked features have zero norm
                            queue_v0 = F.normalize(q_snap_v0[idx_v0, :], dim=0, eps=_norm_eps)
                            queue_v1 = F.normalize(q_snap_v1[idx_v1, :], dim=0, eps=_norm_eps)
                            q_v0 = F.normalize(hz_v0_content, dim=-1, eps=_norm_eps)
                            q_v1 = F.normalize(hz_v1_content, dim=-1, eps=_norm_eps)
                            k_v0_n = F.normalize(k_v0_content, dim=-1, eps=_norm_eps)
                            k_v1_n = F.normalize(k_v1_content, dim=-1, eps=_norm_eps)
                            _tau = args.tau
                            B_moco = q_v0.shape[0]
                            _targets = torch.zeros(B_moco, dtype=torch.long, device=device)

                            if getattr(args, "cross_view_negs_only", False):
                                neg_queue_for_v0, neg_queue_for_v1 = queue_v1, queue_v0
                            else:
                                neg_queue_for_v0, neg_queue_for_v1 = queue_v0, queue_v1
                            # view-0 query → view-1 key positive
                            pos_01 = (q_v0 * k_v1_n).sum(dim=-1, keepdim=True)
                            logits_01 = torch.cat([pos_01, q_v0 @ neg_queue_for_v0], dim=1) / _tau
                            # view-1 query → view-0 key positive
                            pos_10 = (q_v1 * k_v0_n).sum(dim=-1, keepdim=True)
                            logits_10 = torch.cat([pos_10, q_v1 @ neg_queue_for_v1], dim=1) / _tau
                            level_loss = F.cross_entropy(logits_01, _targets) + F.cross_entropy(logits_10, _targets)

                            # --- Contrastive diagnostics for per-view path ---
                            with torch.no_grad():
                                _pv_correct = (logits_01.argmax(dim=1) == 0).sum().item() + (
                                    logits_10.argmax(dim=1) == 0
                                ).sum().item()
                                _pv_total = logits_01.shape[0] + logits_10.shape[0]
                                level_loss._contrastive_diag = {
                                    "top1_acc": _pv_correct / max(_pv_total, 1),
                                    "pos_sim_mean": torch.cat([pos_01.squeeze(-1), pos_10.squeeze(-1)]).mean().item(),
                                    "pos_sim_std": torch.cat([pos_01.squeeze(-1), pos_10.squeeze(-1)]).std().item(),
                                    "neg_sim_mean": torch.cat([q_v0 @ neg_queue_for_v0, q_v1 @ neg_queue_for_v1])
                                    .mean()
                                    .item(),
                                    "neg_sim_std": torch.cat([q_v0 @ neg_queue_for_v0, q_v1 @ neg_queue_for_v1])
                                    .std()
                                    .item(),
                                }

                            # --- Stale-queue diagnostic (cheap, no grad) ---
                            if level_idx == 0 and optimizer is not None:
                                with torch.no_grad():
                                    # 1. Positive vs negative similarity gap
                                    #    Healthy: pos >> mean(neg).  Stale queue: gap shrinks.
                                    neg_sim_v0 = (q_v0 @ neg_queue_for_v0).mean().item()
                                    neg_sim_v1 = (q_v1 @ neg_queue_for_v1).mean().item()
                                    pos_sim = (
                                        (q_v0 * k_v1_n).sum(-1).mean().item() + (q_v1 * k_v0_n).sum(-1).mean().item()
                                    ) / 2
                                    # 2. Queue feature norm BEFORE L2-norm (detects dead channels)
                                    raw_norm_v0 = q_snap_v0[idx_v0, :].norm(dim=0).mean().item()
                                    raw_norm_v1 = q_snap_v1[idx_v1, :].norm(dim=0).mean().item()
                                    _diag = {
                                        "MoCo/pos_sim": pos_sim,
                                        "MoCo/neg_sim_v0": neg_sim_v0,
                                        "MoCo/neg_sim_v1": neg_sim_v1,
                                        "MoCo/pos_neg_gap": pos_sim - (neg_sim_v0 + neg_sim_v1) / 2,
                                        "MoCo/queue_raw_norm_v0": raw_norm_v0,
                                        "MoCo/queue_raw_norm_v1": raw_norm_v1,
                                    }
                        else:
                            _lf = patch_loss_func if _is_patch else loss_func
                            level_loss = _lf(
                                hz_content,
                                level_content_indices,
                                args.subsets,
                                soft_content_mask=None,
                            )
                    else:
                        # --- Shared mask (original path) ---
                        # This level has a learnable Gumbel mask — reuse the same
                        # mask the forward pass sampled for the codebook.  Gradients
                        # from the contrastive loss flow back to channel_logits.
                        soft_content_mask = mask_or_tuple
                        content_masks = [soft_content_mask] * len(args.subsets)
                        _level_ci = [torch.where(m.bool())[-1].tolist() for m in content_masks]
                        level_content_indices = _level_ci
                        if estimated_content_indices is None:
                            estimated_content_indices = _level_ci

                        if use_moco:
                            key_pooled = key_outputs[level_idx]
                            k_level = key_pooled.reshape(n_views, -1, *key_pooled.shape[1:])
                            queue_snapshot = vqvae_model.queues[level_idx].clone().detach()
                            _qv1 = (
                                vqvae_model.queues_v1[level_idx].clone().detach()
                                if hasattr(vqvae_model, "queues_v1")
                                else None
                            )
                            # Patch MoCo: flatten (n_views, B, C, P) → (n_views, B*P, C)
                            # so each patch becomes an independent query/key in the queue.
                            # Positives: same subject + same patch position across views.
                            # Negatives: queue entries from all subjects × all patches.
                            _hz_moco = (
                                hz_level.permute(0, 1, 3, 2).reshape(n_views, -1, hz_level.shape[2])
                                if _is_patch
                                else hz_level
                            )
                            _k_moco = (
                                k_level.permute(0, 1, 3, 2).reshape(n_views, -1, k_level.shape[2])
                                if _is_patch
                                else k_level
                            )

                            level_loss = moco_loss_func(
                                _hz_moco,
                                _k_moco,
                                queue_snapshot,
                                level_content_indices,
                                args.subsets,
                                soft_content_mask=soft_content_mask,
                                queue_v1=_qv1,
                            )
                        else:
                            _lf = patch_loss_func if _is_patch else loss_func
                            level_loss = _lf(
                                hz_level,
                                level_content_indices,
                                args.subsets,
                                soft_content_mask=soft_content_mask,
                            )
                else:
                    # Fallback: no channel_logits configured, use batch statistics.
                    # For patch mode, average over the patch dim to get per-channel logits.
                    _hz_for_logits = hz_level.mean(dim=-1) if _is_patch else hz_level
                    avg_logits = _hz_for_logits.mean(dim=[0, 1], keepdim=False).unsqueeze(0)
                    if len(args.subsets) > 1 and content_size > 0:
                        content_masks = utils.smart_gumbel_softmax_mask(
                            avg_logits=avg_logits,
                            content_sizes=[content_size],
                            subsets=args.subsets,
                        )
                    else:
                        content_masks = utils.gumbel_softmax_mask(
                            avg_logits=avg_logits,
                            content_sizes=[content_size],
                            subsets=args.subsets,
                        )

                    _level_ci = [torch.where(m.bool())[-1].tolist() for m in content_masks]
                    level_content_indices = _level_ci
                    if estimated_content_indices is None:
                        estimated_content_indices = _level_ci

                    if use_moco:
                        key_pooled = key_outputs[level_idx]
                        k_level = key_pooled.reshape(n_views, -1, *key_pooled.shape[1:])
                        queue_snapshot = vqvae_model.queues[level_idx].clone().detach()
                        _qv1 = (
                            vqvae_model.queues_v1[level_idx].clone().detach()
                            if hasattr(vqvae_model, "queues_v1")
                            else None
                        )
                        # Patch MoCo: flatten (n_views, B, C, P) → (n_views, B*P, C)
                        _hz_moco = (
                            hz_level.permute(0, 1, 3, 2).reshape(n_views, -1, hz_level.shape[2])
                            if _is_patch
                            else hz_level
                        )
                        _k_moco = (
                            k_level.permute(0, 1, 3, 2).reshape(n_views, -1, k_level.shape[2]) if _is_patch else k_level
                        )

                        level_loss = moco_loss_func(
                            _hz_moco,
                            _k_moco,
                            queue_snapshot,
                            level_content_indices,
                            args.subsets,
                            soft_content_mask=soft_content_mask,
                            queue_v1=_qv1,
                        )

                        # --- Stale-queue diagnostic for shared-mask / onthefly path ---
                        if level_idx == 0 and optimizer is not None and accumulation_step == 0:
                            with torch.no_grad():
                                _ci = level_content_indices[0]
                                _q = F.normalize(_hz_moco[0, :, _ci], dim=-1)
                                _k = F.normalize(_k_moco[1, :, _ci], dim=-1)
                                _queue_neg = F.normalize(queue_snapshot[_ci, :], dim=0)
                                _pos = (_q * _k).sum(-1).mean().item()
                                _neg = (_q @ _queue_neg).mean().item()
                                _diag = {
                                    "MoCo/pos_sim": _pos,
                                    "MoCo/neg_sim_v0": _neg,
                                    "MoCo/pos_neg_gap": _pos - _neg,
                                    "MoCo/queue_raw_norm": queue_snapshot.norm(dim=0).mean().item(),
                                }
                    else:
                        _lf = patch_loss_func if _is_patch else loss_func
                        level_loss = _lf(
                            hz_level,
                            level_content_indices,
                            args.subsets,
                            soft_content_mask=soft_content_mask,
                        )

                level_losses.append(level_loss.item())
                # Collect contrastive diagnostics (top-1 acc, sim distributions)
                if hasattr(level_loss, "_contrastive_diag"):
                    for _dk, _dv in level_loss._contrastive_diag.items():
                        _diag[f"Contrastive/{_dk}_L{level_idx}"] = _dv
                _lvl_weights = getattr(args, "contrastive_level_weights", None)
                _lvl_w = _lvl_weights[level_idx] if _lvl_weights and level_idx < len(_lvl_weights) else 1.0
                total_contrastive_loss = total_contrastive_loss + level_loss * args.scale_contrastive_loss * _lvl_w

            # Enqueue all levels in one call after the loss loop.
            if use_moco and optimizer is not None:
                with torch.no_grad():
                    _keys = []
                    for _lvl_idx, k in enumerate(key_outputs):
                        if _patch_grid is not None:
                            _k_flat = k.detach().permute(0, 2, 1).reshape(-1, k.shape[1])
                        else:
                            _k_flat = k.detach()
                        _keys.append(_k_flat)
                    vqvae_model.enqueue(_keys, n_views=n_views)

            contrastive_loss = total_contrastive_loss
            total_loss = contrastive_loss + recon_loss + vq_loss
            recon_loss_value = recon_loss.item()
            vq_loss_value = vq_loss.item()
            contrastive_loss_value = contrastive_loss.item()
            # NOTE: estimated_content_indices was already set to the dynamically
            # computed indices from channel_logits inside the level loop above
            # (line: estimated_content_indices = [torch.where(m.bool())[-1].tolist()
            # for m in content_masks]).  Do NOT overwrite it here with
            # args.content_indices — that would replace the learned channel
            # selection with the static config-based indices and break evaluation
            # in get_data() for any run that uses channel_logits.

        # ------------------------------------------------------------------
        # VAE path
        # ------------------------------------------------------------------
        else:
            hz = []
            for m_midx, m in enumerate(args.modalities):
                samples = data[m]
                hz_m = encoders[m_midx](torch.concat(samples, 0))
                hz += [hz_m]
            hz = torch.concat(hz, 0)
            hz_flat = hz.view(hz.size(0), -1)

            decoded_images = decoders[0](hz_flat)
            ground_truth_images = torch.concat(data["image"], 0).to(decoded_images.device)
            recon_loss = (
                recon_loss_fn(
                    {"reconstruction": [decoded_images], "quantization_losses": []},
                    ground_truth_images,
                )
                * args.scale_recon_loss
            )

            avg_logits = hz_flat.mean(0)[None]
            if "content_indices" not in data:
                data["content_indices"] = args.content_indices
            content_size = [len(content) for content in data["content_indices"]]

            if args.selection in ["ground_truth", "concat"]:
                estimated_content_indices = args.content_indices
            else:
                if args.subsets[-1] == list(range(args.n_views)) and content_size[-1] > 0:
                    content_masks = utils.smart_gumbel_softmax_mask(
                        avg_logits=avg_logits,
                        content_sizes=content_size,
                        subsets=args.subsets,
                    )
                else:
                    content_masks = utils.gumbel_softmax_mask(
                        avg_logits=avg_logits,
                        content_sizes=content_size,
                        subsets=args.subsets,
                    )
                estimated_content_indices = [torch.where(c_mask)[-1].tolist() for c_mask in content_masks]

            contrastive_loss = loss_func(
                hz_flat.reshape(n_views, -1, hz_flat.shape[-1]),
                estimated_content_indices,
                args.subsets,
            )
            total_loss = contrastive_loss + recon_loss
            recon_loss_value = recon_loss.item()
            vq_loss_value = 0.0
            contrastive_loss_value = contrastive_loss.item()
            level_losses = [contrastive_loss_value]

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    if optimizer is not None:
        scaled_loss = total_loss / total_accumulation_steps

        # Guard against NaN: skip the entire backward + step to avoid
        # corrupting model parameters and optimizer state.
        if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
            print(
                f"  ⚠ NaN/Inf detected in loss (contrastive={contrastive_loss_value:.4f}, "
                f"recon={recon_loss_value:.4f}, vq={vq_loss_value:.4f}). "
                f"Skipping backward pass for this step.",
                flush=True,
            )
            optimizer.zero_grad(set_to_none=True)
            return (
                0.0,
                contrastive_loss_value,
                recon_loss_value,
                vq_loss_value,
                estimated_content_indices,
                level_losses,
                _diag,
            )

        if use_amp:
            scaler.scale(scaled_loss).backward()
            if accumulation_step == total_accumulation_steps - 1:
                scaler.unscale_(optimizer)
                clip_grad_norm_(params, max_norm=2.0, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
        else:
            scaled_loss.backward()
            if accumulation_step == total_accumulation_steps - 1:
                clip_grad_norm_(params, max_norm=2.0, norm_type=2)
                optimizer.step()

        # MoCo momentum update: must happen AFTER optimizer.step() so the
        # momentum encoder trails the online encoder by one step.
        # During mask warmup, we still update the momentum encoder (even
        # though the queue is disabled) so it's warmed up when MoCo begins.
        _moco_requested = getattr(args, "use_moco", False)
        if _moco_requested and accumulation_step == total_accumulation_steps - 1:
            from models.vqvae import MoCoEncoder

            if isinstance(vqvae_model, MoCoEncoder):
                vqvae_model.momentum_update()

    return (
        total_loss.item(),
        contrastive_loss_value,
        recon_loss_value,
        vq_loss_value,
        estimated_content_indices,
        level_losses,
        _diag,
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Periodic validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def _run_validation(
    val_loader,
    encoders,
    decoders,
    loss_func,
    args,
    recon_loss_fn,
    moco_loss_func,
    device,
    max_batches=20,
):
    """Run a short validation pass and return averaged (total, contrastive, recon, vq) losses."""
    # Temporarily switch to eval mode
    was_training = {}
    for i, enc in enumerate(encoders):
        was_training[f"enc_{i}"] = enc.training
        enc.eval()
    for i, dec in enumerate(decoders):
        was_training[f"dec_{i}"] = dec.training
        dec.eval()

    totals, cons, recs, vqs = [], [], [], []

    for batch_idx, data in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        try:
            total_loss, contrastive_loss, recon_loss, vq_loss, _, _, _ = train_step(
                data,
                encoders,
                decoders,
                loss_func,
                optimizer=None,  # no backward
                params=[],
                args=args,
                scaler=None,
                recon_loss_fn=recon_loss_fn,
                moco_loss_func=moco_loss_func,
                step=getattr(args, "recon_loss_start_step", 0),  # ensure recon is always active in val
            )
            totals.append(total_loss)
            cons.append(contrastive_loss)
            recs.append(recon_loss)
            vqs.append(vq_loss)
        except RuntimeError:
            # Skip OOM or shape errors on val batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    # Restore training mode
    for i, enc in enumerate(encoders):
        if was_training[f"enc_{i}"]:
            enc.train()
    for i, dec in enumerate(decoders):
        if was_training[f"dec_{i}"]:
            dec.train()

    if not totals:
        return 0.0, 0.0, 0.0, 0.0
    return (
        np.mean(totals),
        np.mean(cons),
        np.mean(recs),
        np.mean(vqs),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    # CUDA memory settings — must be applied before any allocation
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.cuda.empty_cache()

    # Resolve paths
    if args.dataset_name != "mpi3d":
        args.datapath = os.path.join(args.dataroot, args.dataset_name)
    else:
        args.datapath = os.path.join(
            args.dataroot,
            f"{args.dataset_name}/real3d_complicated_shapes_ordered.npz",
        )
    args.model_dir = os.path.join(args.model_dir, args.dataset_name)
    if args.model_id is None:
        setattr(args, "model_id", uuid.uuid4())
    args.save_dir = os.path.join(args.model_dir, args.model_id)
    os.makedirs(args.save_dir, exist_ok=True)

    # Logging
    logger = setup_logging(args.save_dir)
    logger.info("=" * 60)
    logger.info("MULTIVIEW CONTRASTIVE REPRESENTATION LEARNING")
    logger.info("=" * 60)
    logger.info(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mode: {'EVALUATION' if args.evaluate else 'TRAINING'}")
    logger.info("")
    logger.info("[PATHS]")
    logger.info(f"  Data root:  {args.dataroot}")
    logger.info(f"  Data path:  {args.datapath}")
    logger.info(f"  Save dir:   {args.save_dir}")
    logger.info(f"  Model ID:   {args.model_id}")

    # Optionally reload saved args (evaluation only)
    if args.evaluate and args.load_args:
        with open(os.path.join(args.save_dir, "settings.json"), "r") as fp:
            loaded_args = json.load(fp)
        for arg in ["encoding_size", "hidden_size"]:
            setattr(args, arg, loaded_args[arg])

    args = update_args(args)

    logger.info("")
    logger.info("[CONFIGURATION]")
    logger.info(f"  Dataset:       {args.dataset_name}")
    logger.info(f"  Modalities:    {args.modalities}")
    logger.info(f"  Num views:     {args.n_views}")
    logger.info(f"  Encoding size: {args.encoding_size}")
    logger.info(f"  Batch size:    {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Temperature:   {args.tau}")
    logger.info(f"  Train steps:   {args.train_steps}")
    logger.info(f"  Subsets:       {args.subsets}")

    # Print all args for backwards compatibility
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    # Reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        logger.info(f"  Seed: {args.seed}")

    # Persist settings
    if not args.evaluate:
        settings_dict = {k: v for k, v in args.__dict__.items() if k != "DATASETCLASS"}
        settings_path = os.path.join(args.save_dir, "settings.json")
        with open(settings_path, "w") as f:
            json.dump(settings_dict, f, indent=4)
        logger.info(f"  Settings saved to: {settings_path}")

    # Device
    logger.info("")
    logger.info("[DEVICE]")
    if torch.cuda.is_available() and not args.no_cuda:
        device = f"cuda:{device_ids[0]}"
        logger.info(f"  GPU: {device} — {torch.cuda.get_device_name(device_ids[0])}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(device_ids[0]).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        warnings.warn("CUDA not available or --no-cuda set; running on CPU.")
        logger.warning("  Using CPU.")

    # Loss functions
    sim_metric = torch.nn.CosineSimilarity(dim=-1)
    criterion = torch.nn.CrossEntropyLoss()

    _cross_view_negs = getattr(args, "cross_view_negs_only", False)
    _contrastive_type = getattr(args, "contrastive_loss_type", "infonce")

    if _contrastive_type == "barlow_twins":
        _bt_lambda = getattr(args, "bt_lambda", 0.005)
        logger.info(f"[LOSS] Barlow Twins (λ={_bt_lambda})")

        def loss_func(z_rec_tuple, estimated_content_indices, subsets, soft_content_mask=None):
            return barlow_twins_loss(
                z_rec_tuple,
                estimated_content_indices=estimated_content_indices,
                subsets=subsets,
                soft_content_mask=soft_content_mask,
                lambd=_bt_lambda,
            )

        patch_loss_func = loss_func  # BT works identically on pooled or patch features

    elif _contrastive_type == "vicreg":
        _sim_c = getattr(args, "vicreg_sim_coeff", 25.0)
        _std_c = getattr(args, "vicreg_std_coeff", 25.0)
        _cov_c = getattr(args, "vicreg_cov_coeff", 1.0)
        logger.info(f"[LOSS] VICReg (sim={_sim_c}, std={_std_c}, cov={_cov_c})")

        def loss_func(z_rec_tuple, estimated_content_indices, subsets, soft_content_mask=None):
            return vicreg_loss(
                z_rec_tuple,
                estimated_content_indices=estimated_content_indices,
                subsets=subsets,
                soft_content_mask=soft_content_mask,
                sim_coeff=_sim_c,
                std_coeff=_std_c,
                cov_coeff=_cov_c,
            )

        patch_loss_func = loss_func

    else:
        logger.info("[LOSS] InfoNCE")

        def loss_func(z_rec_tuple, estimated_content_indices, subsets, soft_content_mask=None):
            return infonce_loss(
                z_rec_tuple,
                sim_metric=sim_metric,
                criterion=criterion,
                tau=args.tau,
                projector=(lambda x: x),
                estimated_content_indices=estimated_content_indices,
                subsets=subsets,
                soft_content_mask=soft_content_mask,
                cross_view_negs_only=_cross_view_negs,
            )

        def patch_loss_func(z_rec_tuple, estimated_content_indices, subsets, soft_content_mask=None):
            return patch_infonce_loss(
                z_rec_tuple,
                sim_metric=sim_metric,
                criterion=criterion,
                tau=args.tau,
                estimated_content_indices=estimated_content_indices,
                subsets=subsets,
                soft_content_mask=soft_content_mask,
                cross_view_negs_only=_cross_view_negs,
            )

    def moco_loss_func(q, k, queue, estimated_content_indices, subsets, soft_content_mask=None, queue_v1=None):
        return moco_loss(
            q,
            k,
            queue,
            sim_metric=sim_metric,
            tau=args.tau,
            estimated_content_indices=estimated_content_indices,
            subsets=subsets,
            soft_content_mask=soft_content_mask,
            queue_v1=queue_v1,
            cross_view_negs_only=_cross_view_negs,
        )

    # Augmentations / transforms
    if HAS_FAISS:
        faiss.omp_set_num_threads(args.faiss_omp_threads)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                args.DATASETCLASS.mean_per_channel,
                args.DATASETCLASS.std_per_channel,
            ),
        ]
    )

    dataset_kwargs = {
        "transform": transform,
        "labels_path": getattr(args, "labels_path", None),
    }
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": args.workers,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
    }

    # Datasets
    logger.info("")
    logger.info("[DATASETS]")
    train_dataset = args.DATASETCLASS(
        data_dir=args.datapath,
        mode="train",
        change_lists=args.change_lists,
        spacing=getattr(args, "image_spacing", 2.0),
        crop_margin=getattr(args, "crop_margin", 0),
        spatial_size=getattr(args, "spatial_size", None),
        cache=getattr(args, "cache_dataset", False),
        cache_dir=getattr(args, "cache_dir", None),
        **dataset_kwargs,
    )
    logger.info(f"  Train: {len(train_dataset)} samples from {args.datapath}")

    if args.dataset_name == "multimodal3di":
        dataset_kwargs["vocab_filepath"] = train_dataset.vocab_filepath
    if args.dataset_name == "mpi3d":
        dataset_kwargs["collate_random_pair"] = True
        train_dataset.collate_random_pair = True
        if args.collate_random_pair:
            dataloader_kwargs["collate_fn"] = train_dataset.__collate_fn__random_pair__

    # Always create val_dataset for periodic validation during training or final check
    val_every = getattr(args, "val_every", 0)
    need_val_dataset = args.evaluate or val_every > 0 or getattr(args, "eval_separation_at_end", True)
    if need_val_dataset:
        val_dataset = args.DATASETCLASS(
            data_dir=args.datapath,
            mode="val",
            change_lists=args.change_lists,
            spacing=getattr(args, "image_spacing", 2.0),
            crop_margin=getattr(args, "crop_margin", 0),
            spatial_size=getattr(args, "spatial_size", None),
            **dataset_kwargs,
        )
        val_kwargs = dict(dataloader_kwargs)
        val_kwargs["shuffle"] = False
        val_kwargs.pop("collate_fn", None)
        val_loader = DataLoader(val_dataset, **val_kwargs)
    else:
        val_dataset = None
        val_loader = None

    if args.evaluate:
        test_dataset = args.DATASETCLASS(
            data_dir=args.datapath,
            mode="test",
            change_lists=args.change_lists,
            spacing=getattr(args, "image_spacing", 2.0),
            crop_margin=getattr(args, "crop_margin", 0),
            spatial_size=getattr(args, "spatial_size", None),
            **dataset_kwargs,
        )
    else:
        train_loader = DataLoader(train_dataset, **dataloader_kwargs)
        train_iterator = InfiniteIterator(train_loader)

    print(f"Train dataset size: {len(train_dataset)} samples.")

    # Model
    logger.info("")
    logger.info("[MODEL]")
    if args.encoder_type == "vqvae":
        use_checkpoint = getattr(args, "gradient_checkpointing", False)
        _entries_arg = args.vqvae_nb_entries
        _entries_log = _entries_arg[0] if isinstance(_entries_arg, list) and len(_entries_arg) == 1 else _entries_arg
        logger.info(
            f"  VQ-VAE-2 | levels={args.vqvae_nb_levels} "
            f"hidden={args.vqvae_hidden_channels} embed={args.vqvae_embed_dim} "
            f"entries={_entries_log} grad_ckpt={use_checkpoint}"
        )
        vqvae_model = vqvae.VQVAE(
            in_channels=1,
            hidden_channels=args.vqvae_hidden_channels,
            res_channels=args.vqvae_res_channels,
            nb_res_layers=2,
            nb_levels=args.vqvae_nb_levels,
            embed_dim=args.vqvae_embed_dim,
            nb_entries=args.vqvae_nb_entries,
            scaling_rates=args.vqvae_scaling_rates,
            use_checkpoint=use_checkpoint,
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
        )
        if getattr(args, "compile_model", False):
            logger.info("  Compiling VQ-VAE-2 with torch.compile (this may take a minute)...")
            vqvae_model = torch.compile(vqvae_model)
        vqvae_model = torch.nn.DataParallel(vqvae_model, device_ids=device_ids)
        vqvae_model.to(device)
        logger.info(f"  Parameters: {sum(p.numel() for p in vqvae_model.parameters()):,}")
        cs_levels = getattr(args, "content_style_levels", [0])
        cs_ratios = getattr(args, "content_ratios", None)
        logger.info(f"  Content/style mask levels: {cs_levels}")
        if cs_ratios is not None:
            logger.info(f"  Per-level content ratios: {dict(zip(cs_levels, cs_ratios))}")
        if getattr(args, "separate_encoders", False):
            logger.info("  Separate encoders: ENABLED (one encoder stack per view)")
        mask_mode = getattr(args, "mask_mode", "onthefly")
        _mask_desc = {
            "onthefly": " (on-the-fly from avg activations, shared across views)",
            "learned": " (learnable nn.Parameter per level)",
            "fixed": " (static first-K-channels = content, no Gumbel noise)",
        }
        logger.info(f"  Mask mode: {mask_mode}" + _mask_desc.get(mask_mode, ""))
        if getattr(args, "quantize_style", False):
            _se = getattr(args, "style_embed_dim", None) or args.vqvae_embed_dim
            _sn = getattr(args, "style_nb_entries", None) or args.vqvae_nb_entries
            if isinstance(_sn, list) and len(_sn) == 1:
                _sn = _sn[0]
            logger.info(f"  Style quantization: ENABLED (embed_dim={_se}, nb_entries={_sn})")
        _skip_levels = getattr(args, "skip_decoder_concat_levels", None)
        if _skip_levels:
            logger.info(
                f"  Final-decoder concat: SKIPPING levels {sorted(_skip_levels)} "
                f"(their codes will be zeroed in the level-0 decoder input)"
            )

        encoders = [vqvae_model]
        decoders = []

        if getattr(args, "use_moco", False):
            from models.vqvae import MoCoEncoder

            moco_model = MoCoEncoder(
                vqvae_model,
                queue_size=args.moco_queue_size,
                momentum=args.moco_momentum,
                nb_levels=args.vqvae_nb_levels,
            )
            moco_model.to(device)
            encoders = [moco_model]
            logger.info(f"  MoCo: queue_size={args.moco_queue_size} momentum={args.moco_momentum}")

        total_params = sum(p.numel() for p in vqvae_model.parameters())

    else:
        logger.info("  VAE encoder + decoder")
        encoder_img = torch.nn.DataParallel(vae.Encoder(), device_ids=device_ids).to(device)
        encoders = [encoder_img]

        if "text" in args.modalities:
            encoder_txt = TextEncoder2D(
                input_size=train_dataset.vocab_size,
                output_size=args.encoding_size,
                sequence_length=train_dataset.max_sequence_length,
            )
            encoder_txt = torch.nn.DataParallel(encoder_txt, device_ids=device_ids).to(device)
            encoders += [encoder_txt]

        # Compute spatial size from spacing/crop settings to match the encoder's output.
        # Must mirror the logic in utils.utils.transforms() exactly.
        _custom_spatial = getattr(args, "spatial_size", None)
        crop_margin = getattr(args, "crop_margin", 0)
        if _custom_spatial is not None:
            spatial_size = tuple(_custom_spatial)
            if crop_margin > 0:
                spatial_size = tuple(s - 2 * crop_margin for s in spatial_size)
        else:
            spacing = getattr(args, "image_spacing", 2.0)
            if spacing == 1.0:
                spatial_size = (182, 218, 182)
            elif spacing == 2.0:
                spatial_size = (91, 109, 91)
            else:
                spatial_size = tuple(int(s / spacing) for s in (182, 218, 182))
            if crop_margin > 0:
                spatial_size = tuple(s - 2 * crop_margin for s in spatial_size)
        decoder = torch.nn.DataParallel(
            vae.Decoder(latent_dim=512, spatial_size=spatial_size), device_ids=device_ids
        ).to(device)
        decoders = [decoder]
        total_params = sum(sum(p.numel() for p in m.parameters()) for m in encoders + decoders)

    logger.info(f"  Total trainable parameters: {total_params:,}")

    # Load pretrained weights (evaluation mode)
    if args.evaluate:
        logger.info("")
        logger.info("[LOADING PRETRAINED MODELS]")
        if args.encoder_type == "vqvae":
            path = os.path.join(args.save_dir, "vqvae_model.pt")
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            encoders[0].load_state_dict(checkpoint["encoders"])
            if getattr(args, "use_moco", False) and "moco_queues" in checkpoint:
                from models.vqvae import MoCoEncoder

                if isinstance(encoders[0], MoCoEncoder):
                    for lvl, q_cpu in enumerate(checkpoint["moco_queues"]):
                        encoders[0]._get_queue(lvl).copy_(q_cpu.to(device))
                    encoders[0].queue_ptrs.copy_(torch.tensor(checkpoint["moco_queue_ptrs"], dtype=torch.long))
            logger.info(f"  Loaded VQ-VAE-2 from {path}")
        else:
            for m_idx, m in enumerate(args.modalities):
                path = os.path.join(args.save_dir, f"encoder_{m}.pt")
                encoders[m_idx].load_state_dict(torch.load(path, map_location=device, weights_only=False))
                logger.info(f"  Loaded encoder_{m} from {path}")

    # Optimizer — separate param groups so weight decay skips biases & norms.
    # Mask parameters (channel_logits) get their own group with a scaled LR
    # so the content/style mask evolves slowly relative to the encoder,
    # reducing MoCo queue staleness when --mask-mode is learned/learned_split.
    _wd = getattr(args, "weight_decay", 0.01)
    _mask_lr_scale = getattr(args, "mask_lr_scale", 1.0)
    # Collect param ids for mask logits so we can route them to a dedicated group.
    _mask_param_ids = set()
    for module in encoders + decoders:
        for name, param in module.named_parameters():
            if "channel_logits" in name or "split_gate_logits" in name:
                _mask_param_ids.add(id(param))
    decay_params = []
    no_decay_params = []
    mask_params = []
    for module in encoders + decoders:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in _mask_param_ids:
                mask_params.append(param)
                continue
            # Skip weight decay for biases, LayerNorm/GroupNorm weights, and
            # ReZero alpha scalars — these should not be regularised.
            if name.endswith(".bias") or "norm" in name.lower() or name.endswith(".alpha"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    param_groups = [
        {"params": decay_params, "weight_decay": _wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if mask_params:
        param_groups.append({"params": mask_params, "weight_decay": 0.0, "lr": args.lr * _mask_lr_scale})
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, fused=use_fused)
    params = decay_params + no_decay_params + mask_params  # flat list for gradient clipping
    logger.info("")
    logger.info(
        f"[OPTIMIZER] AdamW (fused={use_fused}) lr={args.lr} wd={_wd} "
        f"params={sum(p.numel() for p in params):,} "
        f"(decay={len(decay_params)}, no_decay={len(no_decay_params)})"
    )
    if mask_params:
        logger.info(
            f"[OPTIMIZER] Mask param group: {len(mask_params)} params, "
            f"lr={args.lr * _mask_lr_scale:.2e} (scale={_mask_lr_scale})"
        )

    # LR schedule: linear warmup then cosine annealing (or constant)
    warmup_steps = getattr(args, "warmup_steps", 0)
    lr_schedule = getattr(args, "lr_schedule", "constant")
    lr_min = getattr(args, "lr_min", 0.0)
    lr_min_ratio = lr_min / args.lr if args.lr > 0 else 0.0
    total_steps = args.train_steps

    def _lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return (current_step + 1) / warmup_steps
        if lr_schedule == "constant":
            return 1.0
        # Cosine annealing
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr_min_ratio + (1.0 - lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    logger.info(
        f"[LR SCHEDULE] {lr_schedule} | warmup={warmup_steps} steps | " f"lr_min={lr_min} | total={total_steps} steps"
    )

    _mask_warmup = getattr(args, "mask_warmup_steps", 0)
    if _mask_warmup > 0 and getattr(args, "use_moco", False):
        logger.info(
            f"[MASK WARMUP] First {_mask_warmup} steps use in-batch InfoNCE (no MoCo queue) "
            f"to let the learned mask stabilise."
        )

    scaler = GradScaler("cuda") if args.use_amp else None
    if args.use_amp:
        logger.info("  Mixed precision: enabled (AMP)")

    recon_loss_fn = BaselineLoss().to(device)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    file_name = os.path.join(args.save_dir, "Training.csv")

    # W&B initialization
    _use_wandb = getattr(args, "use_wandb", False) and HAS_WANDB
    if _use_wandb:
        wandb_config = {k: v for k, v in vars(args).items() if k != "DATASETCLASS"}
        wandb_dir = os.environ.get("WANDB_DIR", args.save_dir)
        wandb.init(
            project=getattr(args, "wandb_project", "multiview-crl-sweep"),
            entity=getattr(args, "wandb_entity", None),
            config=wandb_config,
            name=str(args.model_id),
            dir=wandb_dir,
            settings=wandb.Settings(init_timeout=300),
        )
        logger.info("[WANDB] Logging enabled")
    elif getattr(args, "use_wandb", False) and not HAS_WANDB:
        logger.warning("[WANDB] --use-wandb set but wandb not installed. Skipping.")

    if not args.evaluate:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tensorboard"))
        logger.info("")
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)
        logger.info(
            f"  Steps: {args.train_steps}  |  Log every: {args.log_steps}  "
            f"|  Checkpoint every: {args.checkpoint_steps}"
        )

        step = 1
        loss_values = collections.deque(maxlen=args.log_steps)
        contrastive_losses = collections.deque(maxlen=args.log_steps)
        recon_losses = collections.deque(maxlen=args.log_steps)
        vq_losses = collections.deque(maxlen=args.log_steps)

        loss_deques = {
            "loss": loss_values,
            "contrastive_loss": contrastive_losses,
            "recon_loss": recon_losses,
            "vq_loss": vq_losses,
        }
        step = load_checkpoint(
            args, encoders, decoders, optimizer, device, loss_deques, scheduler=scheduler, scaler=scaler
        )

        # If we successfully resumed from a checkpoint, the model is already
        # warm — skip the recon_loss_start_step delay.
        _recon_start = getattr(args, "recon_loss_start_step", 0)
        args._resumed_past_recon_start = step > 1
        if args._resumed_past_recon_start and _recon_start > 0:
            logger.info(
                f"  Resumed at step {step}: recon loss active immediately "
                f"(skipping --recon-loss-start-step {_recon_start} warmup)."
            )

        # Restore best loss from best checkpoint if resuming
        best_total_loss = float("inf")
        best_ckpt_path = os.path.join(
            args.save_dir, "vqvae_best.pt" if args.encoder_type == "vqvae" else "checkpoint_best.pt"
        )
        if getattr(args, "resume_training", False) and os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            best_total_loss = best_ckpt.get("loss", float("inf"))
            logger.info(f"  Restored best loss: {best_total_loss:.4f} from {best_ckpt_path}")
            del best_ckpt

        # Early stopping state
        _es_patience = getattr(args, "early_stopping_patience", 0)
        _es_min_delta = getattr(args, "early_stopping_min_delta", 0.0)
        _es_best = float("inf")
        _es_wait = 0
        _es_triggered = False
        if _es_patience > 0:
            logger.info(
                f"  Early stopping enabled: patience={_es_patience} checkpoint intervals, "
                f"min_delta={_es_min_delta:.6f}, "
                f"monitoring={'validation loss' if val_every > 0 else 'rolling training loss'}"
            )

        oom_count = 0
        MAX_OOM_RETRIES = 5

        try:
            while step <= args.train_steps:
                try:
                    accum_steps = getattr(args, "gradient_accumulation_steps", 1)
                    accum_total = accum_contrastive = accum_recon = accum_vq = 0.0

                    # Roll skip_recon dice ONCE per accumulation window so all
                    # micro-steps have the same memory profile (avoids worst-case
                    # where all micro-steps happen to compute recon simultaneously).
                    _skip_ratio = getattr(args, "skip_recon_ratio", 0.0)
                    _window_compute_recon = (_skip_ratio == 0.0) or (torch.rand(1).item() > _skip_ratio)

                    accum_level_losses = None
                    for accum_idx in range(accum_steps):
                        data = next(train_iterator)
                        (
                            total_loss,
                            contrastive_loss,
                            recon_loss,
                            vq_loss,
                            _,
                            step_level_losses,
                            step_moco_diag,
                        ) = train_step(
                            data,
                            encoders,
                            decoders,
                            loss_func,
                            optimizer,
                            params,
                            args=args,
                            scaler=scaler,
                            recon_loss_fn=recon_loss_fn,
                            accumulation_step=accum_idx,
                            total_accumulation_steps=accum_steps,
                            moco_loss_func=moco_loss_func,
                            step=step,
                            force_compute_recon=_window_compute_recon,
                            patch_loss_func=patch_loss_func,
                        )
                        accum_total += total_loss / accum_steps
                        accum_contrastive += contrastive_loss / accum_steps
                        accum_recon += recon_loss / accum_steps
                        accum_vq += vq_loss / accum_steps
                        if accum_level_losses is None:
                            accum_level_losses = [v / accum_steps for v in step_level_losses]
                        else:
                            accum_level_losses = [
                                a + v / accum_steps for a, v in zip(accum_level_losses, step_level_losses)
                            ]

                    scheduler.step()

                    # Flush MoCo queues at the end of mask warmup so stale
                    # embeddings from the warmup phase don't pollute the queue.
                    _mask_warmup_steps = getattr(args, "mask_warmup_steps", 0)
                    if _mask_warmup_steps > 0 and step == _mask_warmup_steps and getattr(args, "use_moco", False):
                        from models.vqvae import MoCoEncoder

                        if isinstance(encoders[0], MoCoEncoder):
                            for lvl in range(encoders[0].nb_levels):
                                encoders[0]._get_queue(lvl).normal_()
                                F.normalize(encoders[0]._get_queue(lvl), dim=0, out=encoders[0]._get_queue(lvl))
                                if encoders[0]._separate_queues:
                                    encoders[0]._get_queue(lvl, view=1).normal_()
                                    F.normalize(
                                        encoders[0]._get_queue(lvl, view=1),
                                        dim=0,
                                        out=encoders[0]._get_queue(lvl, view=1),
                                    )
                            encoders[0].queue_ptrs.zero_()
                            if encoders[0]._separate_queues:
                                encoders[0].queue_v1_ptrs.zero_()
                            logger.info(
                                f"  [MASK WARMUP] Step {step}: mask warmup complete — "
                                f"MoCo queues flushed, switching to MoCo contrastive."
                            )

                    oom_count = 0
                    loss_values.append(accum_total)
                    contrastive_losses.append(accum_contrastive)
                    recon_losses.append(accum_recon)
                    vq_losses.append(accum_vq)

                    # Resolve underlying VQVAE module (unwrap MoCo / DataParallel)
                    _raw = encoders[0]
                    if hasattr(_raw, "online"):
                        _raw = _raw.online
                    if hasattr(_raw, "module"):
                        _raw = _raw.module

                    _acc_str = ""
                    if step_moco_diag:
                        if _contrastive_type == "barlow_twins":
                            # Show on/off-diagonal loss per level
                            _bt_parts = []
                            for _li in range(args.vqvae_nb_levels):
                                _on = step_moco_diag.get(f"Contrastive/on_diag_loss_L{_li}", None)
                                _off = step_moco_diag.get(f"Contrastive/off_diag_loss_L{_li}", None)
                                if _on is not None:
                                    _bt_parts.append(f"L{_li}: on={_on:.3f} off={_off:.3f}")
                            if _bt_parts:
                                _acc_str = f" | BT({', '.join(_bt_parts)})"
                        elif _contrastive_type == "vicreg":
                            # Show sim/var/cov loss per level
                            _vr_parts = []
                            for _li in range(args.vqvae_nb_levels):
                                _sim = step_moco_diag.get(f"Contrastive/sim_loss_L{_li}", None)
                                _var = step_moco_diag.get(f"Contrastive/var_loss_L{_li}", None)
                                _cov = step_moco_diag.get(f"Contrastive/cov_loss_L{_li}", None)
                                if _sim is not None:
                                    _vr_parts.append(f"L{_li}: sim={_sim:.3f} var={_var:.3f} cov={_cov:.3f}")
                            if _vr_parts:
                                _acc_str = f" | VICReg({', '.join(_vr_parts)})"
                        else:
                            _acc_parts = []
                            for _li in range(args.vqvae_nb_levels):
                                _ak = f"Contrastive/top1_acc_L{_li}"
                                if _ak in step_moco_diag:
                                    _acc_parts.append(f"L{_li}={step_moco_diag[_ak]:.1%}")
                            if _acc_parts:
                                _acc_str = f" | Top1Acc: {', '.join(_acc_parts)}"
                    _cb_parts = []
                    for _cb_lvl, _cb in enumerate(_raw.codebooks):
                        _alive = (_cb.cluster_size > 1.0).sum().item()
                        _cb_parts.append(f"L{_cb_lvl}={_alive:.0f}/{_cb.n_embed}")
                    _cb_str = f" | CB: {', '.join(_cb_parts)}" if _cb_parts else ""
                    print(
                        f"Step {step}: Total={accum_total:.4f} | "
                        f"Contrastive={accum_contrastive:.4f} | "
                        f"Recon={accum_recon:.4f} | VQ={accum_vq:.4f}{_acc_str}{_cb_str}",
                        flush=True,
                    )

                    if step % args.log_steps == 0:
                        tb_writer.add_scalar("Loss/Total", accum_total, step)
                        tb_writer.add_scalar("Loss/Contrastive", accum_contrastive, step)
                        tb_writer.add_scalar("Loss/Recon", accum_recon, step)
                        tb_writer.add_scalar("Loss/VQ", accum_vq, step)
                        tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], step)

                        # Per-level contrastive losses
                        if accum_level_losses:
                            for _li, _lv in enumerate(accum_level_losses):
                                tb_writer.add_scalar(f"Loss/Contrastive_L{_li}", _lv, step)

                        # Log Gumbel mask diagnostics per level (skip for fixed mode — no logits)
                        if hasattr(_raw, "channel_logits") and getattr(_raw, "mask_mode", "onthefly") != "fixed":
                            for lvl_key, logits_param in _raw.channel_logits.items():
                                probs = torch.softmax(logits_param.detach(), dim=0)
                                entropy = -(probs * probs.log().clamp(min=-100)).sum().item()
                                max_entropy = np.log(probs.numel())
                                tb_writer.add_scalar(f"Mask/Entropy_L{lvl_key}", entropy, step)
                                tb_writer.add_scalar(f"Mask/NormEntropy_L{lvl_key}", entropy / max_entropy, step)
                                # How spread out the logits are (higher = more decisive)
                                tb_writer.add_scalar(
                                    f"Mask/LogitStd_L{lvl_key}", logits_param.detach().std().item(), step
                                )
                                # Top-k vs bottom-k gap: mean of selected minus mean of not selected
                                k_lvl = _raw.content_channels_per_level.get(int(lvl_key), _raw.content_channels)
                                sorted_logits = logits_param.detach().sort(descending=True).values
                                top_mean = sorted_logits[:k_lvl].mean().item()
                                bot_mean = sorted_logits[k_lvl:].mean().item()
                                tb_writer.add_scalar(f"Mask/TopBotGap_L{lvl_key}", top_mean - bot_mean, step)

                        # Log learned_split gate diagnostics (effective content size)
                        if hasattr(_raw, "split_gate_logits"):
                            for lvl_key, gate_param in _raw.split_gate_logits.items():
                                gate_probs = torch.sigmoid(gate_param.detach())
                                n_content = (gate_probs > 0.5).sum().item()
                                n_total = gate_param.numel()
                                tb_writer.add_scalar(f"Split/ContentSize_L{lvl_key}", n_content, step)
                                tb_writer.add_scalar(f"Split/ContentRatio_L{lvl_key}", n_content / n_total, step)
                                # Mean gate probability (how confident the gates are)
                                tb_writer.add_scalar(f"Split/GateMean_L{lvl_key}", gate_probs.mean().item(), step)
                                # Gate entropy: low = confident split, high = uncertain
                                gate_ent = (
                                    -(
                                        gate_probs * gate_probs.clamp(min=1e-7).log()
                                        + (1 - gate_probs) * (1 - gate_probs).clamp(min=1e-7).log()
                                    )
                                    .mean()
                                    .item()
                                )
                                tb_writer.add_scalar(f"Split/GateEntropy_L{lvl_key}", gate_ent, step)

                        # Log codebook utilization per level
                        for _cb_lvl, _cb in enumerate(_raw.codebooks):
                            _alive = (_cb.cluster_size > 1.0).sum().item()
                            _total = _cb.n_embed
                            tb_writer.add_scalar(f"Codebook/Active_L{_cb_lvl}", _alive, step)
                            tb_writer.add_scalar(f"Codebook/Utilization_L{_cb_lvl}", _alive / _total, step)
                        # Style codebook utilization (if active)
                        if hasattr(_raw, "style_codebooks") and _raw.style_codebooks:
                            for _sc_key, _sc_cb in _raw.style_codebooks.items():
                                _s_alive = (_sc_cb.cluster_size > 1.0).sum().item()
                                _s_total = _sc_cb.n_embed
                                tb_writer.add_scalar(f"Codebook/StyleActive_L{_sc_key}", _s_alive, step)
                                tb_writer.add_scalar(f"Codebook/StyleUtil_L{_sc_key}", _s_alive / _s_total, step)

                        # Log MoCo stale-queue diagnostics
                        if step_moco_diag:
                            for diag_key, diag_val in step_moco_diag.items():
                                tb_writer.add_scalar(diag_key, diag_val, step)

                        # W&B step logging
                        if _use_wandb:
                            wandb_log = {
                                "loss/total": accum_total,
                                "loss/contrastive": accum_contrastive,
                                "loss/recon": accum_recon,
                                "loss/vq": accum_vq,
                                "lr": optimizer.param_groups[0]["lr"],
                            }
                            if accum_level_losses:
                                for _li, _lv in enumerate(accum_level_losses):
                                    wandb_log[f"loss/contrastive_L{_li}"] = _lv
                            if step_moco_diag:
                                for diag_key, diag_val in step_moco_diag.items():
                                    wandb_log[diag_key.replace("/", "/")] = diag_val
                            for _cb_lvl, _cb in enumerate(_raw.codebooks):
                                _alive = (_cb.cluster_size > 1.0).sum().item()
                                wandb_log[f"codebook/active_L{_cb_lvl}"] = _alive
                                wandb_log[f"codebook/utilization_L{_cb_lvl}"] = _alive / _cb.n_embed
                                wandb_log[f"codebook/fwd_count_L{_cb_lvl}"] = getattr(
                                    _cb, "_fwd_count", torch.tensor(0)
                                ).item()
                                wandb_log[f"codebook/finite_L{_cb_lvl}"] = int(getattr(_cb, "_last_finite", True))
                                wandb_log[f"codebook/dead_L{_cb_lvl}"] = (
                                    (_cb.cluster_size < getattr(_cb, "reset_threshold", 1.0)).sum().item()
                                )
                            if hasattr(_raw, "style_codebooks") and _raw.style_codebooks:
                                for _sc_key, _sc_cb in _raw.style_codebooks.items():
                                    _s_alive = (_sc_cb.cluster_size > 1.0).sum().item()
                                    wandb_log[f"codebook/style_active_L{_sc_key}"] = _s_alive
                                    wandb_log[f"codebook/style_util_L{_sc_key}"] = _s_alive / _sc_cb.n_embed
                            wandb.log(wandb_log, step=step)

                        if step % args.log_steps == 1 or step == args.train_steps:
                            with open(file_name, "a+") as f:
                                csv.writer(f).writerow(
                                    [
                                        "Step",
                                        step,
                                        "Total",
                                        f"{np.mean(loss_values):.3f}",
                                        "Contrastive",
                                        f"{np.mean(contrastive_losses):.3f}",
                                        "Recon",
                                        f"{np.mean(recon_losses):.3f}",
                                        "VQ",
                                        f"{np.mean(vq_losses):.3f}",
                                    ]
                                )
                            tb_writer.flush()

                    if (step % 200 == 0 or step == 1) and args.encoder_type != "vqvae":
                        save_decoded_images(encoders, decoders, data, args, step)
                    if (step % 200 == 0 or step == 1) and args.encoder_type == "vqvae":
                        save_vqvae_decoded_images(encoders[0], data, args, step)

                    if step % args.checkpoint_steps == 1 or step == args.train_steps or step == args.log_steps * 2:
                        # Check if rolling average loss is a new best
                        rolling_loss = np.mean(loss_values) if len(loss_values) == loss_values.maxlen else None
                        new_best = None
                        if rolling_loss is not None and rolling_loss < best_total_loss:
                            best_total_loss = rolling_loss
                            new_best = rolling_loss

                        save_checkpoint(
                            args,
                            step,
                            encoders,
                            decoders,
                            optimizer,
                            total_loss,
                            contrastive_loss,
                            recon_loss,
                            vq_loss,
                            scheduler=scheduler,
                            best_loss=new_best,
                            scaler=scaler,
                        )

                        # Periodic separation score evaluation
                        if step % 2000 == 1 or step == args.train_steps:
                            if getattr(args, "eval_separation_periodic", True) and args.encoder_type == "vqvae":
                                try:
                                    from eval.cross_reconstruction import evaluate_content_style_separation

                                    logger.info(
                                        f"  [EVALUATION] Running periodic content/style separation metrics (step {step})..."
                                    )
                                    # Use a smaller max_batches for speed during training (80 batches = 320 samples with batch_size=4)
                                    cs_metrics = evaluate_content_style_separation(
                                        encoders[0],
                                        val_loader
                                        or DataLoader(val_dataset, **{**dataloader_kwargs, "shuffle": False}),
                                        args,
                                        device,
                                        max_batches=80,
                                    )
                                    if _use_wandb:
                                        wandb.log(cs_metrics, step=step)
                                except Exception as e:
                                    logger.warning(f"  [WARNING] Periodic separation evaluation failed: {e}")

                    # --- Periodic validation ---
                    if val_every > 0 and val_loader is not None and step % val_every == 0:
                        val_total, val_con, val_rec, val_vq = _run_validation(
                            val_loader,
                            encoders,
                            decoders,
                            loss_func,
                            args,
                            recon_loss_fn,
                            moco_loss_func,
                            device,
                        )
                        tb_writer.add_scalar("Val/Total", val_total, step)
                        tb_writer.add_scalar("Val/Contrastive", val_con, step)
                        tb_writer.add_scalar("Val/Recon", val_rec, step)
                        tb_writer.add_scalar("Val/VQ", val_vq, step)
                        tb_writer.flush()
                        if _use_wandb:
                            wandb.log(
                                {
                                    "val/total": val_total,
                                    "val/contrastive": val_con,
                                    "val/recon": val_rec,
                                    "val/vq": val_vq,
                                },
                                step=step,
                            )
                        logger.info(
                            f"  [Val @ step {step}] Total={val_total:.4f} | "
                            f"Contrastive={val_con:.4f} | Recon={val_rec:.4f} | "
                            f"VQ={val_vq:.4f}"
                        )

                    # --- Early stopping check ---
                    # Runs at checkpoint intervals (same cadence as best-model tracking).
                    # Monitors val loss when --val-every is set, otherwise rolling training loss.
                    if _es_patience > 0 and (step % args.checkpoint_steps == 1 or step == args.train_steps):
                        # Pick the metric to monitor
                        if val_every > 0 and val_loader is not None and step % val_every == 0:
                            _es_metric = val_total
                            _es_source = "val"
                        elif rolling_loss is not None:
                            _es_metric = rolling_loss
                            _es_source = "rolling_train"
                        else:
                            _es_metric = None

                        if _es_metric is not None:
                            if _es_metric < _es_best - _es_min_delta:
                                _es_best = _es_metric
                                _es_wait = 0
                            else:
                                _es_wait += 1
                                logger.info(
                                    f"  [Early stopping] No improvement in {_es_source} loss "
                                    f"({_es_metric:.4f} vs best {_es_best:.4f}, "
                                    f"delta={_es_min_delta:.6f}). "
                                    f"Patience: {_es_wait}/{_es_patience}"
                                )
                                if _es_wait >= _es_patience:
                                    logger.info(
                                        f"  [Early stopping] Patience exhausted at step {step}. "
                                        f"Best {_es_source} loss: {_es_best:.4f}. Stopping."
                                    )
                                    _es_triggered = True
                                    break

                    # Periodic CUDA cache cleanup to reduce fragmentation-induced OOM
                    if step % 20 == 0 and torch.cuda.is_available():
                        import gc

                        gc.collect()
                        torch.cuda.empty_cache()

                    step += 1

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        oom_count += 1
                        if torch.cuda.is_available():
                            alloc = torch.cuda.memory_allocated() / 1024**3
                            reserv = torch.cuda.memory_reserved() / 1024**3
                            peak = torch.cuda.max_memory_allocated() / 1024**3
                            logger.error(
                                f"[OOM] Step {step}: allocated={alloc:.2f}GB "
                                f"reserved={reserv:.2f}GB peak={peak:.2f}GB "
                                f"({oom_count}/{MAX_OOM_RETRIES})"
                            )
                        torch.cuda.empty_cache()
                        import gc

                        gc.collect()
                        if optimizer is not None:
                            optimizer.zero_grad(set_to_none=True)
                        if oom_count >= MAX_OOM_RETRIES:
                            logger.error(f"[OOM] {MAX_OOM_RETRIES} consecutive OOMs — aborting.")
                            save_emergency_checkpoint(
                                args,
                                step,
                                encoders,
                                decoders,
                                optimizer,
                                reason=f"oom_x{oom_count}",
                                scheduler=scheduler,
                            )
                            raise
                        logger.warning(f"[OOM] Skipping step {step}, continuing...")
                        step += 1
                    else:
                        logger.error(f"[ERROR] Step {step}: {e}\n{traceback.format_exc()}")
                        save_emergency_checkpoint(
                            args,
                            step,
                            encoders,
                            decoders,
                            optimizer,
                            reason=f"runtime_error_step{step}",
                            scheduler=scheduler,
                        )
                        raise

                except Exception as e:
                    logger.error(f"[ERROR] Step {step}: {type(e).__name__} — {e}\n{traceback.format_exc()}")
                    save_emergency_checkpoint(
                        args,
                        step,
                        encoders,
                        decoders,
                        optimizer,
                        reason=f"{type(e).__name__}_step{step}",
                        scheduler=scheduler,
                    )
                    raise

        except KeyboardInterrupt:
            logger.warning(f"\n[INTERRUPTED] Training stopped at step {step}")
            save_emergency_checkpoint(
                args,
                step,
                encoders,
                decoders,
                optimizer,
                reason=f"keyboard_interrupt_step{step}",
                scheduler=scheduler,
            )
            tb_writer.close()
            return

        logger.info("")
        logger.info("=" * 60)
        if _es_triggered:
            logger.info(f"TRAINING STOPPED EARLY (step {step})")
        else:
            logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        if loss_values:
            logger.info(f"  Final total loss:       {loss_values[-1]:.4f}")
            logger.info(f"  Final contrastive loss: {contrastive_losses[-1]:.4f}")
            logger.info(f"  Final recon loss:       {recon_losses[-1]:.4f}")
            logger.info(f"  Final VQ loss:          {vq_losses[-1]:.4f}")
            logger.info(f"  Rolling avg total (last {args.log_steps}): {np.mean(loss_values):.4f}")
        if _es_triggered:
            logger.info(f"  Early stopping best monitored loss: {_es_best:.4f}")
        logger.info(f"  Models saved to: {args.save_dir}")

        # Compute separation score at the very end of training so sweeps have it
        if getattr(args, "eval_separation_at_end", True) and args.encoder_type == "vqvae":
            # First, reload the BEST model weights instead of using the final step's weights
            best_ckpt_path = os.path.join(args.save_dir, "vqvae_best.pt")
            if os.path.exists(best_ckpt_path):
                logger.info(
                    f"  [EVALUATION] Reloading BEST checkpoint from {best_ckpt_path} for final separation metrics..."
                )
                try:
                    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                    encoders[0].load_state_dict(checkpoint["encoders"])
                except Exception as e:
                    logger.warning(f"  Failed to load best checkpoint, using final weights instead: {e}")

            try:
                from eval.cross_reconstruction import evaluate_content_style_separation

                logger.info("[EVALUATION] Running final content/style separation metrics...")
                cs_metrics = evaluate_content_style_separation(
                    encoders[0],
                    val_loader or DataLoader(val_dataset, **{**dataloader_kwargs, "shuffle": False}),
                    args,
                    device,
                )
                for k, v in cs_metrics.items():
                    logger.info(f"  {k}: {v:.4f}")
                if _use_wandb:
                    wandb.log(cs_metrics)
                    # For sweeps, make sure it's pushed to summary so the agent easily captures it
                    wandb.summary.update(cs_metrics)
                # Save to CSV
                cs_path = os.path.join(args.save_dir, "cross_recon_metrics_train_end.csv")
                import pandas as pd

                pd.DataFrame([cs_metrics]).to_csv(cs_path, index=False)
                logger.info(f"  Cross-recon metrics saved to: {cs_path}")
            except Exception as e:
                logger.error(f"[ERROR] Final content/style separation evaluation failed: {e}")

        tb_writer.close()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    if args.evaluate:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STARTING EVALUATION")
        logger.info("=" * 60)

        logger.info("[EVALUATION] Collecting validation encodings...")
        val_dict = get_data(
            val_dataset,
            encoders,
            decoders,
            loss_func,
            dataloader_kwargs,
            args=args,
            num_samples=args.val_size,
            recon_loss_fn=recon_loss_fn,
            moco_loss_func=moco_loss_func,
        )
        logger.info("[EVALUATION] Collecting test encodings...")
        test_dict = get_data(
            test_dataset,
            encoders,
            decoders,
            loss_func,
            dataloader_kwargs,
            args=args,
            num_samples=args.test_size,
            recon_loss_fn=recon_loss_fn,
            moco_loss_func=moco_loss_func,
        )

        logger.info(f"  Val loss:  {np.mean(val_dict['loss_values']):.4f}")
        logger.info(f"  Test loss: {np.mean(test_dict['loss_values']):.4f}")
        print(f"<Val Loss>: {np.mean(val_dict['loss_values']):.4f}")
        print(f"<Test Loss>: {np.mean(test_dict['loss_values']):.4f}")

        if args.encoding_size == 1:
            for split in (val_dict, test_dict):
                for m in args.modalities:
                    split[f"hz_{m}"] = split[f"hz_{m}"].reshape(-1, 1)

        for m in args.modalities:
            sc = StandardScaler()
            val_dict[f"hz_{m}"] = sc.fit_transform(val_dict[f"hz_{m}"])
            test_dict[f"hz_{m}"] = sc.transform(test_dict[f"hz_{m}"])
            for s in args.subsets:
                sc = StandardScaler()
                val_dict[f"hz_{m}_subsets"][s] = sc.fit_transform(val_dict[f"hz_{m}_subsets"][s])
                test_dict[f"hz_{m}_subsets"][s] = sc.transform(test_dict[f"hz_{m}_subsets"][s])

        results = []
        for m_idx, m in enumerate(args.modalities):
            factors_m = args.DATASETCLASS.FACTORS[m]
            discrete_factors_m = args.DATASETCLASS.DISCRETE_FACTORS[m]

            if args.eval_dci:
                import eval.dci as dci

                def repr_fn(samples):
                    with torch.no_grad():
                        if m == "image" and args.dataset_name == "mpi3d":
                            samples = torch.stack([transform(i) for i in samples], dim=0)
                        return encoders[m_idx](samples).cpu().numpy()

                dci_score = dci.compute_dci(
                    ground_truth_data=val_dataset,
                    representation_function=repr_fn,
                    num_train=10000,
                    num_test=5000,
                    random_state=np.random.RandomState(),
                    factor_types=["discrete" if ix in discrete_factors_m else "continuous" for ix in factors_m],
                )
                with open(os.path.join(args.save_dir, f"dci_{m}.csv"), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=dci_score.keys())
                    w.writeheader()
                    w.writerow(dci_score)
                continue

            for ix, factor_name in factors_m.items():
                for s in args.subsets:
                    data_eval = [
                        val_dict[f"hz_{m}_subsets"][s],
                        val_dict[f"labels_{m}"][factor_name],
                        test_dict[f"hz_{m}_subsets"][s],
                        test_dict[f"labels_{m}"][factor_name],
                    ]
                    results.append(eval_step(ix, s, m, factor_name, discrete_factors_m, data_eval, args))
                if args.eval_style and len(args.style_indices) > 0:
                    data_eval = [
                        val_dict[f"hz_{m}"][..., args.style_indices],
                        val_dict[f"labels_{m}"][factor_name],
                        test_dict[f"hz_{m}"][..., args.style_indices],
                        test_dict[f"labels_{m}"][factor_name],
                    ]
                    results.append(eval_step(ix, -1, m, factor_name, discrete_factors_m, data_eval, args))

        columns = [
            "subset",
            "ix",
            "modality",
            "factor_name",
            "factor_type",
            "r2_linreg",
            "r2_krreg",
            "acc_logreg",
            "acc_mlp",
        ]
        df_results = pd.DataFrame(results, columns=columns)
        results_path = os.path.join(args.save_dir, "results.csv")
        df_results.to_csv(results_path)
        logger.info(f"  Results saved to: {results_path}")
        print(df_results.to_string())

        # Cross-reconstruction evaluation for content/style separation
        if args.encoder_type == "vqvae" and hasattr(args, "content_indices"):
            try:
                from eval.cross_reconstruction import evaluate_content_style_separation

                logger.info("[EVALUATION] Running content/style separation metrics...")
                cs_metrics = evaluate_content_style_separation(
                    encoders[0],
                    val_loader or DataLoader(val_dataset, **{**dataloader_kwargs, "shuffle": False}),
                    args,
                    device,
                )
                for k, v in cs_metrics.items():
                    logger.info(f"  {k}: {v:.4f}")
                if _use_wandb:
                    wandb.log(cs_metrics)
                    wandb.summary.update(cs_metrics)
                # Save to CSV
                cs_path = os.path.join(args.save_dir, "cross_recon_metrics.csv")
                pd.DataFrame([cs_metrics]).to_csv(cs_path, index=False)
                logger.info(f"  Cross-recon metrics saved to: {cs_path}")
            except Exception as e:
                logger.warning(f"  Cross-reconstruction evaluation failed: {e}")

        # Log evaluation results to W&B
        if _use_wandb and len(results) > 0:
            for row in results:
                subset, ix, modality, factor_name, factor_type, r2_lin, r2_kr, acc_log, acc_mlp = row
                prefix = f"eval/{modality}/{factor_name}/subset_{subset}"
                wandb.summary[f"{prefix}/r2_linreg"] = r2_lin
                wandb.summary[f"{prefix}/r2_krreg"] = r2_kr
                wandb.summary[f"{prefix}/acc_logreg"] = acc_log
                wandb.summary[f"{prefix}/acc_mlp"] = acc_mlp

    if _use_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args().parse_args()
    main(args)
