# isort: skip_file
# Multiview Contrastive Representation Learning — main training script.
#
# High-level structure
# --------------------
# config.py          parse_args / update_args
# logging_setup.py   setup_logging
# checkpointing.py   save_checkpoint / load_checkpoint / save_emergency_checkpoint
# visualisation.py   save_vqvae_decoded_images
# evaluation.py      val_step / get_data / eval_step
# main_multimodal.py train_step + main (this file)

import collections
import csv
import faulthandler
import json
import math
import os
import random
import signal
import sys
import traceback
import uuid
import warnings
from datetime import datetime

faulthandler.enable()
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import numpy as np
import torch
import torch.multiprocessing as _torch_mp
import torch.nn.functional as F

# NFS workaround: the default "file_descriptor" sharing strategy hits FD limits
# and stalls when DataLoader workers hand off large tensors over NFS. Switching
# to "file_system" routes the handoff via shm-style temp files and is the
# standard fix for NFS-backed clusters.
_torch_mp.set_sharing_strategy("file_system")
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import models.vqvae as vqvae
import utils.utils as utils
from data.infinite_iterator import InfiniteIterator, ResumableSampler
from eval.evaluation import eval_step, get_data
from training.losses import (
    BaselineLoss,
    JukeboxPerceptualLoss,
    barlow_twins_loss,
    content_modality_adv_loss,
    content_patch_modality_adv_loss,
    infonce_loss,
    moco_loss,
    patch_infonce_loss,
    split_infonce_loss,
    style_infonce_loss,
    style_modality_ce_loss,
    vicreg_loss,
)
from utils.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    save_emergency_checkpoint,
)
from utils.config import parse_args, update_args
from utils.logging_setup import setup_logging
from utils.visualisation import save_vqvae_decoded_images

device_ids = [0]


def _project_contrastive_content(head, hz_c, is_patch):
    """Apply an MLP projection head to content-selected features for the contrastive loss.

    ``hz_c`` is the content block already sliced to its content channels:
    ``(n_views, B, k)`` (pooled) or ``(n_views, B, k, P)`` (patch). Returns the
    projected features ``(n_views, B, d)`` / ``(n_views, B, d, P)``. The head is a
    plain ``Linear -> ReLU -> Linear`` MLP, so for the patch case we move the
    channel axis last, project, and move it back. Eval/probes never call this — they
    read the pre-head encoder features — which is the whole point of the head.
    """
    if is_patch:
        return head(hz_c.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).contiguous()
    return head(hz_c)


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
    discriminator=None,
    disc_optimizer=None,
    disc_scaler=None,
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
    _gan_recon = None
    _gan_real = None
    adv_loss_value = 0.0

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
        if getattr(args, "channels_last", False):
            images = images.to(memory_format=torch.channels_last_3d)
        input_shape = images.shape[2:]

        # Brain masks are plumbed alongside the images so the reconstruction
        # loss can restrict its support to brain voxels.
        mask_samples = data.get("mask")
        if mask_samples is not None:
            masks = torch.concat(mask_samples, 0).to(device, non_blocking=True).float()
        else:
            masks = None

        # ------------------------------------------------------------------
        # VQ-VAE-2 path
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # VQ-VAE-2 forward pass
        # ------------------------------------------------------------------
        vqvae_model = encoders[0]

        if getattr(args, "contrastive_only", False):
            # Encoder-only ablation: never decode. return_recon=False makes the
            # VQVAE skip the codebook+decoder loop entirely (skip_codebook), so
            # there is no VQ commitment loss and no reconstruction loss. Overrides
            # force_compute_recon so the validation and accumulation-window
            # callers honour it too.
            compute_recon = False
        elif force_compute_recon is not None:
            compute_recon = force_compute_recon
        else:
            skip_recon_ratio = getattr(args, "skip_recon_ratio", 0.0)
            compute_recon = (skip_recon_ratio == 0.0) or (torch.rand(1).item() > skip_recon_ratio)

        if getattr(args, "patch_contrastive", False):
            _pgpl = getattr(args, "patch_grid_per_level", None)
            # Per-level override (list of tuples) when provided, else single shared tuple.
            _patch_grid = _pgpl if _pgpl is not None else tuple(args.patch_grid)
        else:
            _patch_grid = None

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

            assert isinstance(vqvae_model, MoCoEncoder), "MoCo requested but encoders[0] is not a MoCoEncoder instance."
            with torch.no_grad():
                key_outputs = vqvae_model.encode_keys(images, n_views=n_views, patch_grid=_patch_grid)

        _recon_start = getattr(args, "recon_loss_start_step", 0)
        _recon_active = step >= _recon_start or getattr(args, "_resumed_past_recon_start", False)
        _gan_recon = None
        _gan_real = None
        if compute_recon and recon is not None and _recon_active:
            # Safety net: the model now interpolates internally, but guard
            # against size mismatch in case decode_codes or an older
            # checkpoint path bypasses it.
            if recon.shape[2:] != input_shape:
                recon = F.interpolate(recon, size=input_shape, mode="trilinear", align_corners=False)
            recon_loss = (
                recon_loss_fn(
                    {
                        "reconstruction": [recon],
                        "quantization_losses": diffs,
                        "mask": masks,
                    },
                    images,
                )
                * args.scale_recon_loss
            )
            # Stash for GAN update before freeing (references keep tensors alive)
            _use_gan = discriminator is not None and step >= getattr(args, "gan_start_step", 0)
            if _use_gan:
                _gan_recon = recon
                _gan_real = images
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
        default_content_ratio = len(args.content_indices[0]) / (len(args.content_indices[0]) + len(args.style_indices))

        # Unwrap DataParallel / MoCoEncoder to reach the bare VQVAE so we
        # can read channel_logits (fix #1 / #4).
        _raw_vqvae = vqvae_model.online if hasattr(vqvae_model, "online") else vqvae_model
        _raw_vqvae = _raw_vqvae.module if hasattr(_raw_vqvae, "module") else _raw_vqvae

        # Per-level content channel counts from the model (set by --content-ratios)
        _content_ch_per_level = getattr(_raw_vqvae, "content_channels_per_level", {})

        # Optional contrastive projection head(s): a per-level MLP applied to the
        # content features *before* the contrastive loss (None when disabled).
        # Eval/probes keep reading the pre-head encoder features, so the head only
        # shapes the loss-facing space.
        _proj_heads = getattr(_raw_vqvae, "_contrastive_proj_heads", None)
        _proj_mode = getattr(args, "contrastive_proj_mode", "head")

        def _proj_head_for(_lvl):
            if _proj_heads is None:
                return None
            return _proj_heads[f"L{_lvl}"] if f"L{_lvl}" in _proj_heads else None

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

            # Foreground patch masking: drop always-background patch positions so the
            # patch-contrastive loss isn't dominated by dead background patches (which
            # inject noise — actively harmful for InfoNCE, diluting for BT; see
            # patch-infonce-false-negatives). Position-level, recomputed per batch from
            # the brain mask: keep a position if any sample has >= thresh brain there.
            # Slicing hz_level here flows to every downstream patch path (content, style,
            # loss). Gated to the in-batch path; MoCo patch keys are not sliced in v1.
            if _is_patch and getattr(args, "patch_foreground_mask", False) and masks is not None and not use_moco:
                _pg = _patch_grid[level_idx] if isinstance(_patch_grid, list) else _patch_grid
                _fg_thr = getattr(args, "patch_foreground_thresh", 0.05)
                with torch.no_grad():
                    _frac = F.adaptive_avg_pool3d(masks, tuple(_pg)).flatten(1)  # (n_views*B, P)
                    _keep_pos = (_frac >= _fg_thr).any(dim=0)  # (P,)
                    if not bool(_keep_pos.any()):
                        _keep_pos = torch.ones_like(_keep_pos)  # never drop every patch
                hz_level = hz_level[..., _keep_pos]

            soft_content_mask = None
            _style_hz_v0 = _style_hz_v1 = None

            if level_idx in fwd_soft_content_masks:
                mask_or_tuple = fwd_soft_content_masks[level_idx]

                if isinstance(mask_or_tuple, tuple):
                    mask_v0, mask_v1 = mask_or_tuple
                    idx_v0 = torch.where(mask_v0.bool())[-1]
                    idx_v1 = torch.where(mask_v1.bool())[-1]
                    k_content = int(mask_v0.sum())

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
                    _k_range = torch.arange(k_content, device=hz_level.device)
                    level_content_indices = [_k_range] * len(args.subsets)
                    # Only set estimated_content_indices on the first masked
                    # level so later levels don't overwrite it (fix #6).
                    if estimated_content_indices is None:
                        estimated_content_indices = [idx_v0]  # view-0 for backward compat

                    # Style indices: complement of the content mask (no GPU→CPU sync)
                    _style_idx_v0 = torch.where(~mask_v0.bool())[-1]
                    _style_idx_v1 = torch.where(~mask_v1.bool())[-1]
                    if (
                        _style_idx_v0.numel() > 0
                        and _style_idx_v1.numel() > 0
                        and _style_idx_v0.numel() == _style_idx_v1.numel()
                    ):
                        if _is_patch:
                            _style_hz_v0 = hz_level[0][:, _style_idx_v0, :].mean(-1)
                            _style_hz_v1 = hz_level[1][:, _style_idx_v1, :].mean(-1)
                        else:
                            _style_hz_v0 = hz_level[0][:, _style_idx_v0]
                            _style_hz_v1 = hz_level[1][:, _style_idx_v1]

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

                        q_snap_v0 = vqvae_model.queues[level_idx].detach()
                        q_snap_v1 = vqvae_model.queues_v1[level_idx].detach()
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
                        # Cache the negative-similarity matmuls — these dominate the
                        # contrastive cost (B × queue_size) and were previously
                        # recomputed up to 4× per step for diagnostics.
                        neg_sim_v0 = q_v0 @ neg_queue_for_v0
                        neg_sim_v1 = q_v1 @ neg_queue_for_v1
                        # view-0 query → view-1 key positive
                        pos_01 = (q_v0 * k_v1_n).sum(dim=-1, keepdim=True)
                        logits_01 = torch.cat([pos_01, neg_sim_v0], dim=1) / _tau
                        # view-1 query → view-0 key positive
                        pos_10 = (q_v1 * k_v0_n).sum(dim=-1, keepdim=True)
                        logits_10 = torch.cat([pos_10, neg_sim_v1], dim=1) / _tau
                        level_loss = F.cross_entropy(logits_01, _targets) + F.cross_entropy(logits_10, _targets)

                        # --- Contrastive diagnostics for per-view path ---
                        # top1_acc is consumed every step in the train-loop printout;
                        # the sim summaries are TB-only, so defer the extra .item() syncs
                        # and reductions to log-step boundaries.
                        _is_log_step = step is None or (step % args.log_steps == 0)
                        with torch.no_grad():
                            _pv_correct = (logits_01.argmax(dim=1) == 0).sum().item() + (
                                logits_10.argmax(dim=1) == 0
                            ).sum().item()
                            _pv_total = logits_01.shape[0] + logits_10.shape[0]
                            _diag_dict = {"top1_acc": _pv_correct / max(_pv_total, 1)}
                            if _is_log_step:
                                _pos_cat = torch.cat([pos_01.squeeze(-1), pos_10.squeeze(-1)])
                                _neg_cat = torch.cat([neg_sim_v0, neg_sim_v1])
                                _diag_dict.update(
                                    {
                                        "pos_sim_mean": _pos_cat.mean().item(),
                                        "pos_sim_std": _pos_cat.std().item(),
                                        "neg_sim_mean": _neg_cat.mean().item(),
                                        "neg_sim_std": _neg_cat.std().item(),
                                    }
                                )
                            level_loss._contrastive_diag = _diag_dict

                        # --- Stale-queue diagnostic (cheap, no grad) ---
                        # TB-only — gate to log-step to skip the extra .item() syncs.
                        if level_idx == 0 and optimizer is not None and _is_log_step:
                            with torch.no_grad():
                                # 1. Positive vs negative similarity gap
                                #    Healthy: pos >> mean(neg).  Stale queue: gap shrinks.
                                _neg_v0 = neg_sim_v0.mean().item()
                                _neg_v1 = neg_sim_v1.mean().item()
                                pos_sim = (
                                    (q_v0 * k_v1_n).sum(-1).mean().item() + (q_v1 * k_v0_n).sum(-1).mean().item()
                                ) / 2
                                # 2. Queue feature norm BEFORE L2-norm (detects dead channels)
                                raw_norm_v0 = q_snap_v0[idx_v0, :].norm(dim=0).mean().item()
                                raw_norm_v1 = q_snap_v1[idx_v1, :].norm(dim=0).mean().item()
                                _diag = {
                                    "MoCo/pos_sim": pos_sim,
                                    "MoCo/neg_sim_v0": _neg_v0,
                                    "MoCo/neg_sim_v1": _neg_v1,
                                    "MoCo/pos_neg_gap": pos_sim - (_neg_v0 + _neg_v1) / 2,
                                    "MoCo/queue_raw_norm_v0": raw_norm_v0,
                                    "MoCo/queue_raw_norm_v1": raw_norm_v1,
                                }
                    else:
                        _lf = patch_loss_func if _is_patch else loss_func
                        _ph = _proj_head_for(level_idx)
                        if _ph is not None:
                            # hz_content is already sliced to k content channels.
                            _hz_proj = _project_contrastive_content(_ph, hz_content, _is_patch)
                            _proj_ci = [torch.arange(_hz_proj.shape[2], device=_hz_proj.device)] * len(args.subsets)
                            level_loss = _lf(_hz_proj, _proj_ci, args.subsets, soft_content_mask=None)
                        else:
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
                    _level_ci = [torch.where(m.bool())[-1] for m in content_masks]
                    level_content_indices = _level_ci
                    if estimated_content_indices is None:
                        estimated_content_indices = _level_ci

                    _s_idx = torch.where(~soft_content_mask.bool())[-1]
                    if _s_idx.numel() > 0:
                        if _is_patch:
                            _style_hz_v0 = hz_level[0][:, _s_idx, :].mean(-1)
                            _style_hz_v1 = hz_level[1][:, _s_idx, :].mean(-1)
                        else:
                            _style_hz_v0 = hz_level[0][:, _s_idx]
                            _style_hz_v1 = hz_level[1][:, _s_idx]

                    if use_moco:
                        key_pooled = key_outputs[level_idx]
                        k_level = key_pooled.reshape(n_views, -1, *key_pooled.shape[1:])
                        # Queue is mutated only by enqueue() after the loss loop, so
                        # detach() (a view) is safe — no need for the extra clone.
                        queue_snapshot = vqvae_model.queues[level_idx].detach()
                        _qv1 = vqvae_model.queues_v1[level_idx].detach() if hasattr(vqvae_model, "queues_v1") else None
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
                    else:
                        _lf = patch_loss_func if _is_patch else loss_func
                        _ph = _proj_head_for(level_idx)
                        if _ph is not None:
                            # Slice to content channels — keep the differentiable
                            # mask multiply so gradients still reach learned logits —
                            # then project before the loss.
                            _c_idx = level_content_indices[0]
                            if _is_patch:
                                _hz_c = (hz_level * soft_content_mask.unsqueeze(-1))[:, :, _c_idx, :]
                            else:
                                _hz_c = (hz_level * soft_content_mask)[:, :, _c_idx]
                            _hz_proj = _project_contrastive_content(_ph, _hz_c, _is_patch)
                            if _proj_mode == "entropy":
                                # Yao eq. (3.3): alignment on the RAW block, entropy on the
                                # projection. Reduces exactly to the branch of infonce_base_loss
                                # it replaces when t is the identity, so mode is the only variable.
                                level_loss = split_infonce_loss(
                                    _hz_c,
                                    _hz_proj,
                                    tau=args.tau,
                                    tau_entropy=getattr(args, "tau_entropy", None),
                                    cross_view_negs_only=getattr(args, "cross_view_negs_only", False),
                                )
                            else:
                                _proj_ci = [torch.arange(_hz_proj.shape[2], device=_hz_proj.device)] * len(args.subsets)
                                level_loss = _lf(_hz_proj, _proj_ci, args.subsets, soft_content_mask=None)
                        else:
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

                _level_ci = [torch.where(m.bool())[-1] for m in content_masks]
                level_content_indices = _level_ci
                if estimated_content_indices is None:
                    estimated_content_indices = _level_ci

                if use_moco:
                    key_pooled = key_outputs[level_idx]
                    k_level = key_pooled.reshape(n_views, -1, *key_pooled.shape[1:])
                    # Queue is mutated only by enqueue() after the loss loop.
                    queue_snapshot = vqvae_model.queues[level_idx].detach()
                    _qv1 = vqvae_model.queues_v1[level_idx].detach() if hasattr(vqvae_model, "queues_v1") else None
                    # Patch MoCo: flatten (n_views, B, C, P) → (n_views, B*P, C)
                    _hz_moco = (
                        hz_level.permute(0, 1, 3, 2).reshape(n_views, -1, hz_level.shape[2]) if _is_patch else hz_level
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
                    # TB-only — gate to log-step to skip the extra .item() syncs.
                    if (
                        level_idx == 0
                        and optimizer is not None
                        and accumulation_step == 0
                        and (step is None or step % args.log_steps == 0)
                    ):
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
                    _ph = _proj_head_for(level_idx)
                    if _ph is not None:
                        # Fallback path: select content via the gumbel mask (keeping
                        # the multiply for gradient flow), then project.
                        _mask0 = content_masks[0]
                        _c_idx = level_content_indices[0]
                        if _is_patch:
                            _hz_c = (hz_level * _mask0.unsqueeze(-1))[:, :, _c_idx, :]
                        else:
                            _hz_c = (hz_level * _mask0)[:, :, _c_idx]
                        _hz_proj = _project_contrastive_content(_ph, _hz_c, _is_patch)
                        _proj_ci = [torch.arange(_hz_proj.shape[2], device=_hz_proj.device)] * len(args.subsets)
                        level_loss = _lf(_hz_proj, _proj_ci, args.subsets, soft_content_mask=None)
                    else:
                        level_loss = _lf(
                            hz_level,
                            level_content_indices,
                            args.subsets,
                            soft_content_mask=soft_content_mask,
                        )

            _style_cl_scale = getattr(args, "scale_style_contrastive_loss", 0.0)
            if _style_cl_scale > 0.0 and _style_hz_v0 is not None:
                _style_loss = style_infonce_loss(_style_hz_v0, _style_hz_v1, tau=args.tau)
                total_contrastive_loss = total_contrastive_loss + _style_loss * _style_cl_scale
                _diag[f"Style/infonce_L{level_idx}"] = _style_loss.item()

            # --- Auxiliary modality heads (decouple invariance from capacity) ---
            # Content path: gradient-reversal → content becomes linearly
            #   modality-invariant regardless of style dim.
            # Style path: CE → style must be linearly modality-sufficient,
            #   preventing collapse of modality-correlated demographic signal.
            _adv_scale = getattr(args, "scale_content_modality_adv", 0.0)
            _patch_adv_scale = getattr(args, "scale_content_patch_modality_adv", 0.0)
            _suf_scale = getattr(args, "scale_style_modality_ce", 0.0)
            if (_adv_scale > 0.0 or _patch_adv_scale > 0.0 or _suf_scale > 0.0) and _style_hz_v0 is not None:
                # Pull content features for this level (shared-mask path).
                _ci = level_content_indices[0] if level_content_indices else None
                if _ci:
                    if _is_patch:
                        _content_hz_v0_patch = hz_level[0][:, _ci, :]  # (B, k, P)
                        _content_hz_v1_patch = hz_level[1][:, _ci, :]
                        _content_hz_v0 = _content_hz_v0_patch.mean(-1)  # (B, k)
                        _content_hz_v1 = _content_hz_v1_patch.mean(-1)
                    else:
                        _content_hz_v0 = hz_level[0][:, _ci]
                        _content_hz_v1 = hz_level[1][:, _ci]

                    # Lazy-init heads on first call, store on the raw model.
                    _heads = getattr(_raw_vqvae, "_aux_modality_heads", None)
                    if _heads is None:
                        _heads = torch.nn.ModuleDict()
                        _raw_vqvae._aux_modality_heads = _heads
                    _ck = f"content_L{level_idx}"
                    _cpk = f"content_patch_L{level_idx}"
                    _sk = f"style_L{level_idx}"
                    if _ck not in _heads:
                        _heads[_ck] = torch.nn.Linear(_content_hz_v0.shape[-1], 2).to(device)
                    if _cpk not in _heads and _is_patch:
                        _heads[_cpk] = torch.nn.Linear(_content_hz_v0.shape[-1], 2).to(device)
                    if _sk not in _heads:
                        _heads[_sk] = torch.nn.Linear(_style_hz_v0.shape[-1], 2).to(device)

                    if _adv_scale > 0.0:
                        _lam = getattr(args, "content_modality_adv_lambda", 1.0)
                        _adv_loss, _adv_acc = content_modality_adv_loss(
                            _content_hz_v0, _content_hz_v1, _heads[_ck], lambd=_lam
                        )
                        total_contrastive_loss = total_contrastive_loss + _adv_loss * _adv_scale
                        _diag[f"ModAdv/loss_L{level_idx}"] = _adv_loss.item()
                        _diag[f"ModAdv/acc_L{level_idx}"] = _adv_acc  # ~0.5 = invariant

                    if _patch_adv_scale > 0.0 and _is_patch:
                        _lam = getattr(args, "content_modality_adv_lambda", 1.0)
                        _padv_loss, _padv_acc = content_patch_modality_adv_loss(
                            _content_hz_v0_patch,
                            _content_hz_v1_patch,
                            _heads[_cpk],
                            lambd=_lam,
                        )
                        total_contrastive_loss = total_contrastive_loss + _padv_loss * _patch_adv_scale
                        _diag[f"ModAdvPatch/loss_L{level_idx}"] = _padv_loss.item()
                        _diag[f"ModAdvPatch/acc_L{level_idx}"] = _padv_acc  # ~0.5 = invariant

                    if _suf_scale > 0.0:
                        _suf_loss, _suf_acc = style_modality_ce_loss(_style_hz_v0, _style_hz_v1, _heads[_sk])
                        total_contrastive_loss = total_contrastive_loss + _suf_loss * _suf_scale
                        _diag[f"ModSuf/loss_L{level_idx}"] = _suf_loss.item()
                        _diag[f"ModSuf/acc_L{level_idx}"] = _suf_acc  # ~1.0 = sufficient

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

        # Generator adversarial loss: fool the discriminator into predicting
        # the reconstruction as real (hinge: -mean(D(fake))).
        adv_loss_value = 0.0
        if _gan_recon is not None and optimizer is not None:
            g_adv = -discriminator(_gan_recon).mean() * args.scale_adv_loss
            total_loss = total_loss + g_adv
            adv_loss_value = g_adv.item()

        recon_loss_value = recon_loss.item()
        vq_loss_value = vq_loss.item()
        contrastive_loss_value = contrastive_loss.item()
        # NOTE: estimated_content_indices was already set to the dynamically
        # computed indices from channel_logits inside the level loop above
        # (line: estimated_content_indices = [torch.where(m.bool())[-1]
        # for m in content_masks]).  Do NOT overwrite it here with
        # args.content_indices — that would replace the learned channel
        # selection with the static config-based indices and break evaluation
        # in get_data() for any run that uses channel_logits.

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

        # ------------------------------------------------------------------
        # Discriminator update (hinge loss; runs after G step so D sees the
        # updated generator, keeping training dynamics stable).
        # ------------------------------------------------------------------
        if _gan_recon is not None and disc_optimizer is not None and accumulation_step == total_accumulation_steps - 1:
            disc_optimizer.zero_grad(set_to_none=True)
            _use_disc_amp = disc_scaler is not None
            with autocast("cuda", enabled=_use_disc_amp):
                d_real = discriminator(_gan_real)
                d_fake = discriminator(_gan_recon.detach())
                # Hinge loss: push real > +1, push fake < -1
                d_loss = (F.relu(1.0 - d_real) + F.relu(1.0 + d_fake)).mean()
            if _use_disc_amp:
                disc_scaler.scale(d_loss).backward()
                disc_scaler.step(disc_optimizer)
                disc_scaler.update()
            else:
                d_loss.backward()
                disc_optimizer.step()
            _diag["GAN/D_loss"] = d_loss.item()
            _diag["GAN/D_real"] = d_real.mean().item()
            _diag["GAN/D_fake"] = d_fake.mean().item()

    if adv_loss_value != 0.0:
        _diag["GAN/G_adv_loss"] = adv_loss_value

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
# Deterministic data loading
# ---------------------------------------------------------------------------


def _seed_dataset_transforms(dataset, seed):
    """Seed any MONAI ``Compose`` pipelines a dataset carries (no-op if absent).

    ``Compose.set_random_state`` propagates a distinct, derived seed to each
    child ``Rand*`` transform, so the whole augmentation pipeline becomes
    reproducible from a single seed.
    """
    for attr in ("monai_transform", "_aug_transform"):
        t = getattr(dataset, attr, None)
        if t is not None and hasattr(t, "set_random_state"):
            t.set_random_state(seed=seed)


def _seed_worker(worker_id):
    """DataLoader ``worker_init_fn`` for reproducible augmentation.

    PyTorch seeds each worker's torch/python RNG from the main seed but leaves
    numpy and MONAI's per-transform ``RandomState`` untouched. Re-seed all three,
    decorrelated per worker via ``torch.initial_seed()`` (= base_seed + worker_id).
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    info = torch.utils.data.get_worker_info()
    if info is not None:
        _seed_dataset_transforms(info.dataset, worker_seed)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_vqvae(args) -> vqvae.VQVAE:
    """Construct the bare ``VQVAE`` from a parsed args namespace.

    Single source of truth for the args→model mapping so training (``main``)
    and offline tools (e.g. ``eval/phase0_extract.py``) build an identical
    architecture from the same config. Returns the unwrapped module — callers
    add DataParallel / MoCo / aux heads as needed.
    """
    use_checkpoint = getattr(args, "gradient_checkpointing", False)
    return vqvae.VQVAE(
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
        separate_content_codebooks=getattr(args, "separate_content_codebooks", False),
        mask_mode=getattr(args, "mask_mode", "onthefly"),
        quantize_style=getattr(args, "quantize_style", False),
        separate_style_codebooks=getattr(args, "separate_style_codebooks", False),
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
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    # CUDA memory settings — must be applied before any allocation
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if getattr(args, "deterministic", False):
            # cuBLAS needs a fixed workspace for deterministic matmuls; the env
            # var must be set before the first CUDA matmul. cudnn.benchmark stays
            # off (set in the reproducibility block below).
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        else:
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.cuda.empty_cache()

    # Resolve paths
    args.datapath = os.path.join(args.dataroot, args.dataset_name)
    if args.model_id is None:
        setattr(args, "model_id", str(uuid.uuid4()))
    args.save_dir = os.path.join(args.model_dir, args.dataset_name, args.model_id)
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

    args = update_args(args)

    logger.info("")
    logger.info("[CONFIGURATION]")
    logger.info(f"  Dataset:       {args.dataset_name}")
    logger.info(f"  Batch size:    {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Temperature:   {args.tau}")
    logger.info(f"  Train steps:   {args.train_steps}")

    # Print all args for backwards compatibility
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    # Reproducibility
    if getattr(args, "deterministic", False):
        seed = args.seed if args.seed is not None else 42
        try:
            from monai.utils import set_determinism

            # Seeds torch/numpy/python RNG and every MONAI transform's RandomState.
            set_determinism(seed=seed)
        except Exception as e:  # MONAI layout/version mismatch — seed the basics.
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            logger.warning(f"  monai.set_determinism unavailable ({e}); seeded torch/numpy/random only")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # warn_only: 3D conv-transpose / trilinear-interp backward have no
        # deterministic CUDA kernel; warn instead of crashing the run.
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info(f"  Seed: {seed} (deterministic mode: cudnn.benchmark off, deterministic algorithms on)")
    elif args.seed is not None:
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

    def moco_loss_func(
        q,
        k,
        queue,
        estimated_content_indices,
        subsets,
        soft_content_mask=None,
        queue_v1=None,
    ):
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

    dataset_kwargs = {
        "labels_path": getattr(args, "labels_path", None),
        "masks_dir": getattr(args, "masks_dir", None),
        "asymmetric_aug": getattr(args, "asymmetric_aug", False),
        "shared_brain_mask": getattr(args, "shared_brain_mask", False),
    }
    if args.dataset_name == "synthetic":
        dataset_kwargs.update(
            {
                "synthetic_mode": getattr(args, "synthetic_mode", "pseudo_mri"),
                "synthetic_seed": getattr(args, "synthetic_seed", 42),
                "synthetic_n_content": getattr(args, "synthetic_n_content", 5),
                "synthetic_n_style": getattr(args, "synthetic_n_style", 3),
                "synthetic_style_scale": getattr(args, "synthetic_style_scale", 1.0),
                "synthetic_content_scale": getattr(args, "synthetic_content_scale", 1.0),
                "synthetic_normalize": getattr(args, "synthetic_normalize", "per_sample"),
                "synthetic_hierarchical_content": getattr(args, "synthetic_hierarchical_content", False),
                "synthetic_causal": getattr(args, "synthetic_causal", False),
                "synthetic_causal_graph": getattr(args, "synthetic_causal_graph", "chain"),
                "synthetic_causal_edge_prob": getattr(args, "synthetic_causal_edge_prob", 0.5),
                "synthetic_causal_noise_scale": getattr(args, "synthetic_causal_noise_scale", 0.4),
                "synthetic_causal_nonlinearity": getattr(args, "synthetic_causal_nonlinearity", "leaky_relu"),
                "synthetic_clean_content": getattr(args, "synthetic_clean_content", False),
                "synthetic_identifiable_ventricle": getattr(args, "synthetic_identifiable_ventricle", False),
                "synthetic_num_samples_per_mode": {
                    "train": getattr(args, "synthetic_num_train", 1000),
                    "val": getattr(args, "synthetic_num_val", 100),
                    "test": getattr(args, "synthetic_num_test", 200),
                },
            }
        )
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": args.workers,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
    }
    if getattr(args, "deterministic", False):
        # Reproducible augmentation: seed numpy + MONAI transforms in each worker.
        dataloader_kwargs["worker_init_fn"] = _seed_worker

    # Datasets
    logger.info("")
    logger.info("[DATASETS]")
    train_dataset = args.DATASETCLASS(
        data_dir=args.datapath,
        mode="train",
        spacing=getattr(args, "image_spacing", 2.0),
        crop_margin=getattr(args, "crop_margin", 0),
        spatial_size=getattr(args, "spatial_size", None),
        cache=getattr(args, "cache_dataset", False),
        cache_dir=getattr(args, "cache_dir", None),
        **dataset_kwargs,
    )
    logger.info(f"  Train: {len(train_dataset)} samples from {args.datapath}")

    # Always create val_dataset for periodic validation during training or final check
    val_every = getattr(args, "val_every", 0)
    need_val_dataset = args.evaluate or val_every > 0 or getattr(args, "eval_separation_at_end", True)
    if need_val_dataset:
        val_dataset = args.DATASETCLASS(
            data_dir=args.datapath,
            mode="val",
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

    train_sampler = None
    if args.evaluate:
        test_dataset = args.DATASETCLASS(
            data_dir=args.datapath,
            mode="test",
            spacing=getattr(args, "image_spacing", 2.0),
            crop_margin=getattr(args, "crop_margin", 0),
            spatial_size=getattr(args, "spatial_size", None),
            **dataset_kwargs,
        )
    else:
        if getattr(args, "no_resumable_sampler", False):
            # Pre-Apr-25 behaviour: DataLoader's built-in RandomSampler draws
            # from the global torch RNG, so augmentations / Gumbel noise /
            # dropout see the same random stream as before.
            train_loader = DataLoader(train_dataset, **dataloader_kwargs)
            train_iterator = InfiniteIterator(train_loader)
        else:
            # Use a resumable sampler so mid-epoch resume continues the same
            # shuffle order — otherwise EMA codebook stats and any contrastive
            # queues see a different sample stream after resume and transiently
            # drift. Defer InfiniteIterator construction until after load_checkpoint
            # restores the sampler offset.
            train_dl_kwargs = {**dataloader_kwargs}
            train_dl_kwargs.pop("shuffle", None)
            train_sampler = ResumableSampler(train_dataset, seed=getattr(args, "seed", 0) or 0)
            train_loader = DataLoader(train_dataset, sampler=train_sampler, **train_dl_kwargs)
            train_iterator = None  # built after load_checkpoint below

    print(f"Train dataset size: {len(train_dataset)} samples.")

    # Model
    logger.info("")
    logger.info("[MODEL]")
    use_checkpoint = getattr(args, "gradient_checkpointing", False)
    _entries_arg = args.vqvae_nb_entries
    _entries_log = _entries_arg[0] if isinstance(_entries_arg, list) and len(_entries_arg) == 1 else _entries_arg
    logger.info(
        f"  VQ-VAE-2 | levels={args.vqvae_nb_levels} "
        f"hidden={args.vqvae_hidden_channels} embed={args.vqvae_embed_dim} "
        f"entries={_entries_log} grad_ckpt={use_checkpoint}"
    )
    vqvae_model = build_vqvae(args)
    if getattr(args, "channels_last", False):
        vqvae_model = vqvae_model.to(memory_format=torch.channels_last_3d)
        logger.info("  Memory format: channels_last_3d")
    _adv_on = getattr(args, "scale_content_modality_adv", 0.0) > 0.0
    _patch_adv_on = getattr(args, "scale_content_patch_modality_adv", 0.0) > 0.0
    _suf_on = getattr(args, "scale_style_modality_ce", 0.0) > 0.0
    if _adv_on or _patch_adv_on or _suf_on:
        _heads = torch.nn.ModuleDict()
        _hid = args.vqvae_hidden_channels
        for _lvl in getattr(args, "content_style_levels", [0]):
            _cc = vqvae_model.content_channels_per_level.get(_lvl)
            if _cc is None:
                continue
            _sc = _hid - _cc
            _heads[f"content_L{_lvl}"] = torch.nn.Linear(_cc, 2)
            if _patch_adv_on:
                _heads[f"content_patch_L{_lvl}"] = torch.nn.Linear(_cc, 2)
            _heads[f"style_L{_lvl}"] = torch.nn.Linear(_sc, 2)
        vqvae_model._aux_modality_heads = _heads
        logger.info(
            f"  Aux modality heads: adv={_adv_on} patch_adv={_patch_adv_on} suf={_suf_on} levels={list(_heads.keys())}"
        )

    # Optional contrastive projection head(s): an MLP per content/style level placed
    # between the pooled content features and the contrastive loss. The loss is
    # computed on the head's output; eval/probes keep reading the pre-head encoder
    # features (SimCLR/MoCo/BYOL recipe — the loss-facing space over-compresses toward
    # view-invariance and loses linear-probe info). Built eagerly here (before the
    # optimizer + DataParallel wrap) so its params land in the optimizer and the
    # checkpoint state_dict, mirroring the aux-modality heads above.
    _proj_dim = getattr(args, "contrastive_proj_dim", 0)
    _proj_mode = getattr(args, "contrastive_proj_mode", "head")
    if _proj_dim > 0:
        _proj_hidden = getattr(args, "contrastive_proj_hidden", 256)
        _proj_heads = torch.nn.ModuleDict()
        for _lvl in getattr(args, "content_style_levels", [0]):
            _cc = vqvae_model.content_channels_per_level.get(_lvl)
            if _cc is None:
                continue
            if _proj_mode == "entropy":
                # Yao et al. Defn 3.6: t_k maps the view-specific latent space onto a
                # hyper unit-cube of the SAME dimension |S_k| — hence _cc out and the
                # bounded activation, not a SimCLR-style compression to _proj_dim.
                # Tanh gives (-1,1)^k rather than the paper's (0,1)^k: an affine
                # reparameterisation, so the max-entropy-implies-uniform argument is
                # untouched, but the entropy term is estimated with cosine similarity
                # and a (0,1) codomain confines every vector to the positive orthant,
                # squeezing pairwise cosine into ~[0.75, 1]. Centering restores the
                # full range and costs nothing theoretically.
                _proj_heads[f"L{_lvl}"] = torch.nn.Sequential(
                    torch.nn.Linear(_cc, _proj_hidden),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(_proj_hidden, _cc),
                    torch.nn.Tanh(),
                )
            else:
                _proj_heads[f"L{_lvl}"] = torch.nn.Sequential(
                    torch.nn.Linear(_cc, _proj_hidden),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(_proj_hidden, _proj_dim),
                )
        vqvae_model._contrastive_proj_heads = _proj_heads
        if _proj_mode == "entropy":
            logger.info(
                f"  Contrastive projection (Yao eq. 3.3, ENTROPY term only): dim-preserving "
                f"-> tanh cube (-1,1)^k, hidden={_proj_hidden} levels={list(_proj_heads.keys())}; "
                f"alignment stays on the raw block. --contrastive-proj-dim={_proj_dim} ignored for width."
            )
        else:
            logger.info(
                f"  Contrastive projection head (SimCLR, WHOLE loss): dim={_proj_dim} hidden={_proj_hidden} "
                f"levels={list(_proj_heads.keys())} (loss-facing only; probes read pre-head features)"
            )

    if getattr(args, "compile_model", False):
        logger.warning(
            "  --compile-model ignored: torch.compile is incompatible with "
            "3D VQ-VAE (Triton misaligned-address on odd spatial dims, "
            "aot_eager OOM from graph materialisation). Running eager."
        )

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
    if getattr(args, "separate_content_codebooks", False):
        logger.info(
            "  Separate content codebooks: ENABLED (one content codebook per view; "
            "decoders shared) — ablation, weakens content identifiability"
        )
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
        if getattr(args, "separate_style_codebooks", False):
            logger.info("  Separate style codebooks: ENABLED (one style codebook per view; content shared)")
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
    logger.info(f"  Total trainable parameters: {total_params:,}")

    # Load pretrained weights (evaluation mode)
    if args.evaluate:
        logger.info("")
        logger.info("[LOADING PRETRAINED MODELS]")
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
    # Fused AdamW is fastest but requires params, grads, exp_avg, and
    # exp_avg_sq to share the exact same dtype, device AND memory layout.
    # channels_last_3d changes the layout of 5D conv weights while 1D/2D
    # params stay contiguous — PyTorch < 2.4 fused kernels can't handle the
    # mixed layouts within a param group.  Fall back to the foreach backend
    # (nearly as fast, no layout constraint) when channels_last is active.
    _channels_last = getattr(args, "channels_last", False)
    use_fused = torch.cuda.is_available() and not _channels_last
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, fused=use_fused)
    params = decay_params + no_decay_params + mask_params  # flat list for gradient clipping
    logger.info("")
    logger.info(
        f"[OPTIMIZER] AdamW (fused={use_fused}{', channels_last→foreach' if _channels_last else ''}) "
        f"lr={args.lr} wd={_wd} "
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

    recon_loss_fn = getattr(args, "recon_loss_fn", "BaselineLoss")
    if recon_loss_fn == "JukeboxPerceptualLoss":
        pixel_loss_type = getattr(args, "jukebox_pixel_loss_type", "mse")
        recon_loss_fn = JukeboxPerceptualLoss(dimensions=3, pixel_loss_type=pixel_loss_type).to(device)
        logger.info(f"  Reconstruction loss: Jukebox Perceptual (2.5D LPIPS + FFT + pixel[{pixel_loss_type}])")
    else:
        recon_loss_fn = BaselineLoss().to(device)

    # ------------------------------------------------------------------
    # GAN discriminator (optional, vqvae path only)
    # ------------------------------------------------------------------
    discriminator = None
    disc_optimizer = None
    disc_scaler = None
    if getattr(args, "use_gan", False):
        from models.discriminator import PatchDiscriminator3D

        _disc_base_ch = getattr(args, "disc_base_channels", 32)
        discriminator = PatchDiscriminator3D(in_channels=1, base_channels=_disc_base_ch).to(device)
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=getattr(args, "disc_lr", 4e-4),
            betas=(0.5, 0.9),
            weight_decay=0.0,
        )
        disc_scaler = GradScaler("cuda") if args.use_amp else None
        _disc_params = sum(p.numel() for p in discriminator.parameters())
        logger.info(
            f"[GAN] PatchDiscriminator3D enabled | base_ch={_disc_base_ch} | "
            f"params={_disc_params:,} | disc_lr={getattr(args, 'disc_lr', 4e-4):.2e} | "
            f"adv_weight={args.scale_adv_loss} | starts at step {getattr(args, 'gan_start_step', 0)}"
        )

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
            group=getattr(args, "wandb_group", None),
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
            args,
            encoders,
            decoders,
            optimizer,
            device,
            loss_deques,
            scheduler=scheduler,
            scaler=scaler,
            train_sampler=train_sampler,
        )

        # Build the train iterator AFTER the sampler has been (potentially)
        # restored, so the first epoch starts from the correct mid-epoch offset.
        train_iterator = InfiniteIterator(train_loader)

        # If we successfully resumed from a checkpoint, the model is already
        # warm — skip the recon_loss_start_step delay.
        _recon_start = getattr(args, "recon_loss_start_step", 0)
        args._resumed_past_recon_start = step > 1
        if args._resumed_past_recon_start and _recon_start > 0:
            logger.info(
                f"  Resumed at step {step}: recon loss active immediately "
                f"(skipping --recon-loss-start-step {_recon_start} warmup)."
            )

        # Restore best-model tracking state from best checkpoint if resuming.
        # For VQ-VAE: best is chosen by separation_score (higher is better).
        # For other encoders: best is chosen by rolling training loss (lower is better).
        best_total_loss = float("inf")
        best_separation_score = float("-inf")
        best_ckpt_path = os.path.join(args.save_dir, "vqvae_best.pt")
        # Fallback: if the dedicated best file is missing, read the bookkeeping
        # from the rolling checkpoint (which now mirrors best_metric_*).
        if getattr(args, "resume_training", False) and not os.path.exists(best_ckpt_path):
            _rolling_path = os.path.join(args.save_dir, "vqvae_model.pt")
            if os.path.exists(_rolling_path):
                best_ckpt_path = _rolling_path

        if getattr(args, "resume_training", False) and os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            _prev_name = best_ckpt.get("best_metric_name")
            _prev_value = best_ckpt.get("best_metric_value")
            _expected_sel_name = (
                "synthetic_overall_score"
                if (args.dataset_name == "synthetic" and getattr(args, "select_by_synthetic_dci", True))
                else "separation_score"
            )
            if _prev_name in ("separation_score", "synthetic_overall_score") and _prev_value is not None:
                # Both are higher-is-better selection scores tracked by the same
                # variable, but they are on different scales — only restore when the
                # stored metric matches what this run selects by, else start fresh.
                if _prev_name == _expected_sel_name:
                    best_separation_score = float(_prev_value)
                    logger.info(f"  Restored best {_prev_name}: {best_separation_score:.4f} from {best_ckpt_path}")
                else:
                    logger.info(
                        f"  Prior best used '{_prev_name}' but this run selects by "
                        f"'{_expected_sel_name}'; starting selection fresh."
                    )
            elif _prev_name == "rolling_loss" and _prev_value is not None:
                best_total_loss = float(_prev_value)
                logger.info(f"  Restored best rolling_loss: {best_total_loss:.4f} from {best_ckpt_path}")
            else:
                # Legacy checkpoint without best_metric_* fields — fall back to stored total loss.
                best_total_loss = best_ckpt.get("loss", float("inf"))
                logger.info(
                    f"  Restored legacy best loss: {best_total_loss:.4f} from {best_ckpt_path} "
                    f"(will be replaced once a separation score is computed)"
                )
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
                            discriminator=discriminator,
                            disc_optimizer=disc_optimizer,
                            disc_scaler=disc_scaler,
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
                                F.normalize(
                                    encoders[0]._get_queue(lvl),
                                    dim=0,
                                    out=encoders[0]._get_queue(lvl),
                                )
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
                    # True per-batch utilization (from actual quantizer assignments).
                    # id_outputs is appended coarsest-first inside VQVAE.forward, so
                    # id_outputs[i] corresponds to codebook level (nb_levels - 1 - i).
                    _last_id_outputs = getattr(_raw, "_last_id_outputs", None)
                    _last_style_id_outputs = getattr(_raw, "_last_style_id_outputs", None)
                    _cb_true = {}  # level → (n_unique, perplexity)
                    if _last_id_outputs:
                        with torch.no_grad():
                            for _i, _ids in enumerate(_last_id_outputs):
                                if _ids is None:
                                    continue
                                _cb_lvl = _raw.nb_levels - 1 - _i
                                _flat = _ids.reshape(-1)
                                _n_total = _raw.codebooks[_cb_lvl].n_embed
                                _counts = torch.bincount(_flat, minlength=_n_total).float()
                                _p = _counts / _counts.sum().clamp(min=1.0)
                                _entropy = -(_p * _p.clamp(min=1e-12).log()).sum().item()
                                _cb_true[_cb_lvl] = (
                                    _flat.unique().numel(),
                                    float(np.exp(_entropy)),
                                )
                    _cb_parts = []
                    for _cb_lvl, _cb in enumerate(_raw.codebooks):
                        _alive_ema = (_cb.cluster_size > 1.0).sum().item()
                        if _cb_lvl in _cb_true:
                            _u, _ppl = _cb_true[_cb_lvl]
                            _cb_parts.append(f"L{_cb_lvl}={_u}/{_cb.n_embed}(ppl={_ppl:.1f},ema={_alive_ema:.0f})")
                        else:
                            _cb_parts.append(f"L{_cb_lvl}=ema{_alive_ema:.0f}/{_cb.n_embed}")
                    _cb_str = f" | CB: {', '.join(_cb_parts)}" if _cb_parts else ""
                    _gan_str = ""
                    if step_moco_diag.get("GAN/G_adv_loss") is not None:
                        _gan_str = (
                            f" | G_adv={step_moco_diag['GAN/G_adv_loss']:.4f}"
                            f" D={step_moco_diag.get('GAN/D_loss', 0.0):.4f}"
                        )
                    _style_parts = [
                        f"L{_li}={step_moco_diag[f'Style/infonce_L{_li}']:.4f}"
                        for _li in range(args.vqvae_nb_levels)
                        if f"Style/infonce_L{_li}" in step_moco_diag
                    ]
                    _style_str = f" | StyleNCE: {', '.join(_style_parts)}" if _style_parts else ""
                    print(
                        f"Step {step}: Total={accum_total:.4f} | "
                        f"Contrastive={accum_contrastive:.4f} | "
                        f"Recon={accum_recon:.4f} | VQ={accum_vq:.4f}{_acc_str}{_cb_str}{_gan_str}{_style_str}",
                        flush=True,
                    )

                    if step % args.log_steps == 0:
                        tb_writer.add_scalar("Loss/Total", accum_total, step)
                        tb_writer.add_scalar("Loss/Contrastive", accum_contrastive, step)
                        tb_writer.add_scalar("Loss/Recon", accum_recon, step)
                        tb_writer.add_scalar("Loss/VQ", accum_vq, step)

                        # GAN diagnostics
                        for _gan_key in (
                            "GAN/G_adv_loss",
                            "GAN/D_loss",
                            "GAN/D_real",
                            "GAN/D_fake",
                        ):
                            if _gan_key in step_moco_diag:
                                tb_writer.add_scalar(_gan_key, step_moco_diag[_gan_key], step)
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
                                tb_writer.add_scalar(
                                    f"Mask/NormEntropy_L{lvl_key}",
                                    entropy / max_entropy,
                                    step,
                                )
                                # How spread out the logits are (higher = more decisive)
                                tb_writer.add_scalar(
                                    f"Mask/LogitStd_L{lvl_key}",
                                    logits_param.detach().std().item(),
                                    step,
                                )
                                # Top-k vs bottom-k gap: mean of selected minus mean of not selected
                                k_lvl = _raw.content_channels_per_level.get(int(lvl_key), _raw.content_channels)
                                sorted_logits = logits_param.detach().sort(descending=True).values
                                top_mean = sorted_logits[:k_lvl].mean().item()
                                bot_mean = sorted_logits[k_lvl:].mean().item()
                                tb_writer.add_scalar(
                                    f"Mask/TopBotGap_L{lvl_key}",
                                    top_mean - bot_mean,
                                    step,
                                )

                        # Log learned_split gate diagnostics (effective content size)
                        if hasattr(_raw, "split_gate_logits"):
                            for lvl_key, gate_param in _raw.split_gate_logits.items():
                                gate_probs = torch.sigmoid(gate_param.detach())
                                n_content = (gate_probs > 0.5).sum().item()
                                n_total = gate_param.numel()
                                tb_writer.add_scalar(f"Split/ContentSize_L{lvl_key}", n_content, step)
                                tb_writer.add_scalar(
                                    f"Split/ContentRatio_L{lvl_key}",
                                    n_content / n_total,
                                    step,
                                )
                                # Mean gate probability (how confident the gates are)
                                tb_writer.add_scalar(
                                    f"Split/GateMean_L{lvl_key}",
                                    gate_probs.mean().item(),
                                    step,
                                )
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

                        # Log codebook utilization per level.
                        # `Active_L*` / `Utilization_L*` are EMA-based ("ever recently used")
                        # and overestimate when the encoder has collapsed onto a small
                        # subset of codes. `UniqueIdx_L*`, `UtilizationTrue_L*`, and
                        # `Perplexity_L*` are computed from the actual quantizer
                        # assignments on the current batch and are the honest signal.
                        _style_true = {}
                        if hasattr(_raw, "style_codebooks") and _raw.style_codebooks and _last_style_id_outputs:
                            with torch.no_grad():
                                for _sc_key_int, _ids in _last_style_id_outputs.items():
                                    if _ids is None:
                                        continue
                                    _sc_key = str(_sc_key_int)
                                    if _sc_key not in _raw.style_codebooks:
                                        continue
                                    _flat = _ids.reshape(-1)
                                    _n_total = _raw.style_codebooks[_sc_key].n_embed
                                    _counts = torch.bincount(_flat, minlength=_n_total).float()
                                    _p = _counts / _counts.sum().clamp(min=1.0)
                                    _entropy = -(_p * _p.clamp(min=1e-12).log()).sum().item()
                                    _style_true[_sc_key] = (
                                        _flat.unique().numel(),
                                        float(np.exp(_entropy)),
                                    )
                        for _cb_lvl, _cb in enumerate(_raw.codebooks):
                            _alive = (_cb.cluster_size > 1.0).sum().item()
                            _total = _cb.n_embed
                            tb_writer.add_scalar(f"Codebook/Active_L{_cb_lvl}", _alive, step)
                            tb_writer.add_scalar(
                                f"Codebook/Utilization_L{_cb_lvl}",
                                _alive / _total,
                                step,
                            )
                            if _cb_lvl in _cb_true:
                                _u, _ppl = _cb_true[_cb_lvl]
                                tb_writer.add_scalar(f"Codebook/UniqueIdx_L{_cb_lvl}", _u, step)
                                tb_writer.add_scalar(
                                    f"Codebook/UtilizationTrue_L{_cb_lvl}",
                                    _u / _total,
                                    step,
                                )
                                tb_writer.add_scalar(f"Codebook/Perplexity_L{_cb_lvl}", _ppl, step)
                                tb_writer.add_scalar(
                                    f"Codebook/PerplexityRatio_L{_cb_lvl}",
                                    _ppl / _total,
                                    step,
                                )
                        # Style codebook utilization (if active)
                        if hasattr(_raw, "style_codebooks") and _raw.style_codebooks:
                            for _sc_key, _sc_cb in _raw.style_codebooks.items():
                                _s_alive = (_sc_cb.cluster_size > 1.0).sum().item()
                                _s_total = _sc_cb.n_embed
                                tb_writer.add_scalar(f"Codebook/StyleActive_L{_sc_key}", _s_alive, step)
                                tb_writer.add_scalar(
                                    f"Codebook/StyleUtil_L{_sc_key}",
                                    _s_alive / _s_total,
                                    step,
                                )
                                if _sc_key in _style_true:
                                    _u, _ppl = _style_true[_sc_key]
                                    tb_writer.add_scalar(f"Codebook/StyleUniqueIdx_L{_sc_key}", _u, step)
                                    tb_writer.add_scalar(
                                        f"Codebook/StyleUtilTrue_L{_sc_key}",
                                        _u / _s_total,
                                        step,
                                    )
                                    tb_writer.add_scalar(
                                        f"Codebook/StylePerplexity_L{_sc_key}",
                                        _ppl,
                                        step,
                                    )
                                    tb_writer.add_scalar(
                                        f"Codebook/StylePerplexityRatio_L{_sc_key}",
                                        _ppl / _s_total,
                                        step,
                                    )

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
                                if _cb_lvl in _cb_true:
                                    _u, _ppl = _cb_true[_cb_lvl]
                                    wandb_log[f"codebook/unique_idx_L{_cb_lvl}"] = _u
                                    wandb_log[f"codebook/utilization_true_L{_cb_lvl}"] = _u / _cb.n_embed
                                    wandb_log[f"codebook/perplexity_L{_cb_lvl}"] = _ppl
                                    wandb_log[f"codebook/perplexity_ratio_L{_cb_lvl}"] = _ppl / _cb.n_embed
                            if hasattr(_raw, "style_codebooks") and _raw.style_codebooks:
                                for _sc_key, _sc_cb in _raw.style_codebooks.items():
                                    _s_alive = (_sc_cb.cluster_size > 1.0).sum().item()
                                    wandb_log[f"codebook/style_active_L{_sc_key}"] = _s_alive
                                    wandb_log[f"codebook/style_util_L{_sc_key}"] = _s_alive / _sc_cb.n_embed
                                    if _sc_key in _style_true:
                                        _u, _ppl = _style_true[_sc_key]
                                        wandb_log[f"codebook/style_unique_idx_L{_sc_key}"] = _u
                                        wandb_log[f"codebook/style_utilization_true_L{_sc_key}"] = _u / _sc_cb.n_embed
                                        wandb_log[f"codebook/style_perplexity_L{_sc_key}"] = _ppl
                                        wandb_log[f"codebook/style_perplexity_ratio_L{_sc_key}"] = _ppl / _sc_cb.n_embed
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

                    if (step % 200 == 0 or step == 1) and not getattr(args, "contrastive_only", False):
                        save_vqvae_decoded_images(encoders[0], data, args, step)

                    # Periodic synthetic DCI — track identifiability (content vs
                    # ground-truth factor recovery) DURING training instead of only
                    # at the end of the run. Synthetic-only: requires the gt_latents
                    # emitted by SyntheticBrainDataset.
                    _dci_every = getattr(args, "dci_every", 0)
                    if (
                        _dci_every > 0
                        and args.dataset_name == "synthetic"
                        and val_dataset is not None
                        and (step % _dci_every == 1 or step == args.train_steps)
                    ):
                        # compute_dci_synthetic calls encoder.eval() and does NOT
                        # restore train mode; save/restore it here, otherwise codebook
                        # EMA and Gumbel sampling stay disabled for the rest of training.
                        _dci_was_training = encoders[0].training
                        try:
                            import eval.dci as dci

                            logger.info(f"  [EVALUATION] Periodic synthetic DCI (step {step})...")
                            # Gap only, deliberately: this path scores via GBT
                            # (compute_importance_gbt), which refits per factor per
                            # encoder and is orders of magnitude slower on the ~22k
                            # codes patch pooling produces — it blocks the training
                            # loop. Localized factors (lesion_*) are still covered at
                            # patch by the selection composite below, which probes
                            # with RidgeCV instead, and by the end-of-training DCI.
                            _dci_poolings = [("gap", "gap")]
                            for _dci_label, _dci_pooling in _dci_poolings:
                                _dci_synth = dci.compute_dci_synthetic(
                                    encoder=encoders[0],
                                    dataset=val_dataset,
                                    device=device,
                                    batch_size=dataloader_kwargs.get("batch_size", 32),
                                    num_workers=0,
                                    pooling=_dci_pooling,
                                    per_encoder=getattr(args, "separate_encoders", False),
                                )
                                _dci_flat = dci.flatten_dci_results(_dci_synth)
                                for _dci_k, _dci_v in _dci_flat.items():
                                    if not np.isnan(_dci_v):
                                        tb_writer.add_scalar(f"dci_synthetic/{_dci_label}/{_dci_k}", _dci_v, step)
                                if _use_wandb:
                                    wandb.log(
                                        {
                                            f"dci_synthetic/{_dci_label}/{_dci_k}": _dci_v
                                            for _dci_k, _dci_v in _dci_flat.items()
                                            if not np.isnan(_dci_v)
                                        },
                                        step=step,
                                    )
                        except Exception as e:
                            logger.warning(f"  [WARNING] Periodic synthetic DCI failed: {e}")
                        finally:
                            encoders[0].train(_dci_was_training)

                    if step % args.checkpoint_steps == 1 or step == args.train_steps or step == args.log_steps * 2:
                        # Periodic separation score evaluation — run BEFORE saving the
                        # checkpoint so that if the process is killed mid-eval, the
                        # prior checkpoint is preserved and the eval will be retried
                        # on resume (instead of being silently skipped because step
                        # has already advanced past the `step % 2000 == 1` trigger).
                        separation_score = None
                        selection_name = "separation_score"
                        if step % 2000 == 1 or step == args.train_steps:
                            # (A) Synthetic GT selection: select on the SAME health
                            # composite (overall_score) as eval.run_dci_compare so the
                            # chosen checkpoint is the one that protocol would rank best
                            # (single source of truth). ADNI runs have no GT factors and
                            # fall through to the cross-reconstruction proxy in (B).
                            if (
                                args.dataset_name == "synthetic"
                                and val_dataset is not None
                                and getattr(args, "select_by_synthetic_dci", True)
                            ):
                                # score_encoder_live → _extract_synthetic_representations
                                # calls encoder.eval() without restoring train mode.
                                _sel_was_training = encoders[0].training
                                try:
                                    from eval.run_dci_compare import score_encoder_live

                                    # Patch pooling is included whenever a grid exists,
                                    # independent of the training objective: without it
                                    # FACTOR_POOLING's lesion_* entries fall through
                                    # _resolve_key to "stats", which is permutation-
                                    # invariant over voxels and cannot expose position —
                                    # scoring them near-null while looking like a real
                                    # measurement in the composite.
                                    _sel_pool = [("gap", "gap"), ("stats", "stats")]
                                    if getattr(args, "patch_grid", None):
                                        _sel_pool.append(("patch", tuple(args.patch_grid)))
                                    logger.info(f"  [EVALUATION] Synthetic GT selection composite (step {step})...")
                                    _sel_row = score_encoder_live(
                                        encoders[0],
                                        val_dataset,
                                        device,
                                        level=getattr(args, "selection_dci_level", 0),
                                        poolings=_sel_pool,
                                        n_null=getattr(args, "selection_dci_n_null", 3),
                                        seeds=tuple(range(getattr(args, "selection_dci_n_seeds", 2))),
                                        batch_size=dataloader_kwargs.get("batch_size", 32),
                                        num_workers=0,
                                        per_encoder=getattr(args, "separate_encoders", False),
                                        max_samples=getattr(args, "selection_dci_max_samples", 2000) or None,
                                    )
                                    separation_score = _sel_row.get("disentanglement")  # overall_score
                                    selection_name = "synthetic_overall_score"
                                    _mcc_cc = _sel_row.get("mcc_cc", float("nan"))
                                    _mcc_null = _sel_row.get("mcc_cc_null", float("nan"))
                                    _mcc_gap = (
                                        _mcc_cc - _mcc_null
                                        if np.isfinite(_mcc_cc) and np.isfinite(_mcc_null)
                                        else float("nan")
                                    )
                                    _sel_log = {
                                        "selection/overall_score": _sel_row.get("disentanglement"),
                                        "selection/content_anatomy": _sel_row.get("content_anatomy"),
                                        "selection/content_purity": _sel_row.get("content_purity"),
                                        "selection/style_modality": _sel_row.get("style_modality"),
                                        "selection/style_purity": _sel_row.get("style_purity"),
                                        "selection/separation": _sel_row.get("separation"),
                                        "selection/content_to_style_leak": _sel_row.get("leak_c2s"),
                                        "selection/mcc_cc_gap": _mcc_gap,
                                        "selection/content_view_acc": _sel_row.get("content_view"),
                                    }
                                    for _sel_k, _sel_v in _sel_log.items():
                                        if _sel_v is not None and np.isfinite(_sel_v):
                                            tb_writer.add_scalar(_sel_k, _sel_v, step)
                                    if _use_wandb:
                                        wandb.log(
                                            {
                                                _sel_k: _sel_v
                                                for _sel_k, _sel_v in _sel_log.items()
                                                if _sel_v is not None and np.isfinite(_sel_v)
                                            },
                                            step=step,
                                        )
                                    logger.info(
                                        f"  [SELECTION] synthetic overall_score={separation_score:.4f} "
                                        f"grade={_sel_row.get('grade')} at step {step}"
                                    )
                                except Exception as e:
                                    logger.warning(f"  [WARNING] Synthetic GT selection failed: {e}")
                                finally:
                                    encoders[0].train(_sel_was_training)

                            # (B) Cross-reconstruction separation proxy — used for selection
                            # only when the synthetic GT composite is unavailable (ADNI runs,
                            # --no-select-by-synthetic-dci, or a failed synthetic eval).
                            if (
                                separation_score is None
                                and getattr(args, "eval_separation_periodic", True)
                                and not getattr(args, "contrastive_only", False)
                            ):
                                try:
                                    from eval.cross_reconstruction import (
                                        evaluate_content_style_separation,
                                    )

                                    logger.info(
                                        f"  [EVALUATION] Running periodic content/style separation metrics (step {step})..."
                                    )
                                    # Single-process eval loader. Multi-worker eval was
                                    # hanging on NFS (workers stuck on torch.load of
                                    # cached .pt files, never returning a batch). The
                                    # 200-batch loop is short enough that synchronous
                                    # loading is acceptable, and it surfaces any worker
                                    # exception in the main thread instead of swallowing it.
                                    eval_loader_kwargs = {
                                        **dataloader_kwargs,
                                        "shuffle": False,
                                        "num_workers": 0,
                                        "persistent_workers": False,
                                        "prefetch_factor": None,
                                        "pin_memory": False,
                                    }
                                    eval_loader = DataLoader(val_dataset, **eval_loader_kwargs)
                                    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
                                    try:
                                        cs_metrics = evaluate_content_style_separation(
                                            encoders[0],
                                            eval_loader,
                                            args,
                                            device,
                                            max_batches=50,
                                        )
                                    finally:
                                        faulthandler.cancel_dump_traceback_later()
                                    for _cs_k, _cs_v in cs_metrics.items():
                                        tb_writer.add_scalar(_cs_k, _cs_v, step)
                                    if _use_wandb:
                                        wandb.log(cs_metrics, step=step)
                                    # If the user opts into gating, selection uses the
                                    # anatomy-floor-penalised score so a collapsed
                                    # encoder with "invariant" but content-free features
                                    # cannot win. Falls back to raw score if labels are
                                    # unavailable (gate = 1.0 in that case).
                                    if getattr(args, "select_by_gated_score", False):
                                        separation_score = cs_metrics.get(
                                            "separation_score_gated",
                                            cs_metrics.get("separation_score"),
                                        )
                                    else:
                                        separation_score = cs_metrics.get("separation_score")
                                except Exception as e:
                                    logger.warning(f"  [WARNING] Periodic separation evaluation failed: {e}")

                        # Best-checkpoint selection.
                        # VQ-VAE: pick the checkpoint with the highest separation_score
                        # (content/style disentanglement), which is what the training
                        # objective ultimately targets. Only updated on steps where the
                        # separation eval ran (every 2000 steps).
                        rolling_loss = np.mean(loss_values) if len(loss_values) == loss_values.maxlen else None
                        new_best = None
                        best_metric_name = "total_loss"
                        if separation_score is not None and separation_score > best_separation_score:
                            best_separation_score = separation_score
                            new_best = separation_score
                            best_metric_name = selection_name
                            logger.info(f"  [BEST] New best {selection_name}={separation_score:.4f} at step {step}")

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
                            best_metric_name=best_metric_name,
                            scaler=scaler,
                            train_sampler=train_sampler,
                        )

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
                                train_sampler=train_sampler,
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
                            train_sampler=train_sampler,
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
                        train_sampler=train_sampler,
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
                train_sampler=train_sampler,
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

        # Compute separation score at the very end of training so sweeps have it.
        # contrastive_only disables the decoder, so cross-reconstruction metrics
        # are not meaningful and the block is skipped entirely.
        if getattr(args, "contrastive_only", False):
            logger.info("[EVALUATION] contrastive_only: decoder disabled, skipping final cross-reconstruction metrics.")
        if getattr(args, "eval_separation_at_end", True) and not getattr(args, "contrastive_only", False):
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
                    tb_writer.add_scalar(k, v, step)
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

        for m in args.modalities:
            sc = StandardScaler()
            val_dict[f"hz_{m}"] = sc.fit_transform(val_dict[f"hz_{m}"])
            test_dict[f"hz_{m}"] = sc.transform(test_dict[f"hz_{m}"])
            for s in args.subsets:
                sc = StandardScaler()
                val_dict[f"hz_{m}_subsets"][s] = sc.fit_transform(val_dict[f"hz_{m}_subsets"][s])
                test_dict[f"hz_{m}_subsets"][s] = sc.transform(test_dict[f"hz_{m}_subsets"][s])

        if args.dataset_name == "synthetic" and args.eval_dci:
            import eval.dci as dci

            logger.info("[EVALUATION] Computing DCI metrics on synthetic GT factors...")
            # Same split as the periodic DCI above: gap keeps the historical
            # filename, patch is written alongside it so localized factors
            # (lesion_*) are recorded under a pooling that can expose them.
            _final_poolings = [("gap", "gap", "dci_synthetic.csv")]
            if getattr(args, "patch_grid", None):
                _final_poolings.append(("patch", tuple(args.patch_grid), "dci_synthetic_patch.csv"))
            for _f_label, _f_pooling, _f_name in _final_poolings:
                dci_synth = dci.compute_dci_synthetic(
                    encoder=encoders[0],
                    dataset=test_dataset,
                    device=args.device,
                    batch_size=dataloader_kwargs.get("batch_size", 32),
                    num_workers=dataloader_kwargs.get("num_workers", 0),
                    pooling=_f_pooling,
                    per_encoder=getattr(args, "separate_encoders", False),
                )
                dci_synth_path = os.path.join(args.save_dir, _f_name)
                rows = dci.dci_results_to_rows(dci_synth)
                with open(dci_synth_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=dci.DCI_CSV_COLUMNS)
                    w.writeheader()
                    w.writerows(rows)
                logger.info(f"  Synthetic DCI ({_f_label}) saved to: {dci_synth_path}")

                flat = dci.flatten_dci_results(dci_synth)
                if tb_writer is not None:
                    for k, v in flat.items():
                        if not np.isnan(v):
                            tb_writer.add_scalar(f"dci_synthetic/{_f_label}/{k}", v, global_step=args.iterations)
                if _use_wandb:
                    wandb.log({f"dci_synthetic/{_f_label}/{k}": v for k, v in flat.items() if not np.isnan(v)})

        results = []
        for m_idx, m in enumerate(args.modalities):
            factors_m = args.DATASETCLASS.FACTORS[m]
            discrete_factors_m = args.DATASETCLASS.DISCRETE_FACTORS[m]

            if args.eval_dci:
                import eval.dci as dci

                def repr_fn(samples):
                    with torch.no_grad():
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
                    results.append(eval_step(ix, s, m, factor_name, discrete_factors_m, data_eval))
                if args.eval_style and len(args.style_indices) > 0:
                    data_eval = [
                        val_dict[f"hz_{m}"][..., args.style_indices],
                        val_dict[f"labels_{m}"][factor_name],
                        test_dict[f"hz_{m}"][..., args.style_indices],
                        test_dict[f"labels_{m}"][factor_name],
                    ]
                    results.append(eval_step(ix, -1, m, factor_name, discrete_factors_m, data_eval))

        columns = [
            "subset",
            "ix",
            "modality",
            "factor_name",
            "factor_type",
            "r2_linreg",
            "r2_mlpreg",
            "acc_logreg",
            "acc_mlp",
        ]
        df_results = pd.DataFrame(results, columns=columns)
        results_path = os.path.join(args.save_dir, "results.csv")
        df_results.to_csv(results_path)
        logger.info(f"  Results saved to: {results_path}")
        print(df_results.to_string())

        # Cross-reconstruction evaluation for content/style separation
        # (skipped under contrastive_only — the decoder is never trained).
        if hasattr(args, "content_indices") and not getattr(args, "contrastive_only", False):
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
                (
                    subset,
                    ix,
                    modality,
                    factor_name,
                    factor_type,
                    r2_lin,
                    r2_mlp,
                    acc_log,
                    acc_mlp,
                ) = row
                prefix = f"eval/{modality}/{factor_name}/subset_{subset}"
                wandb.summary[f"{prefix}/r2_linreg"] = r2_lin
                wandb.summary[f"{prefix}/r2_mlpreg"] = r2_mlp
                wandb.summary[f"{prefix}/acc_logreg"] = acc_log
                wandb.summary[f"{prefix}/acc_mlp"] = acc_mlp

    if _use_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args().parse_args()
    main(args)
