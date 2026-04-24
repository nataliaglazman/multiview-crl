"""Checkpoint save and load helpers for multiview-CRL training."""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger("multiview_crl")


def _capture_rng_state() -> dict:
    """Snapshot all RNGs so a resumed run reproduces the same augmentation /
    Gumbel / shuffling sequence as the original."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    """Restore RNGs previously captured by ``_capture_rng_state``. Missing keys
    are tolerated so old checkpoints still load."""
    if not isinstance(state, dict):
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        try:
            torch.set_rng_state(_as_byte_tensor(state["torch"]))
        except Exception as e:
            logger.warning(f"  Could not restore torch RNG state: {e}")
    if "cuda" in state and torch.cuda.is_available():
        try:
            cuda_state = state["cuda"]
            if isinstance(cuda_state, (list, tuple)):
                cuda_state = [_as_byte_tensor(s) for s in cuda_state]
            else:
                cuda_state = _as_byte_tensor(cuda_state)
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception as e:
            logger.warning(f"  Could not restore CUDA RNG state: {e}")


def _as_byte_tensor(t) -> torch.Tensor:
    """Coerce a tensor loaded from checkpoint into the CPU ByteTensor that
    ``torch.set_rng_state`` requires."""
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    return t.detach().cpu().to(torch.uint8).contiguous()


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_checkpoint(
    args,
    step: int,
    encoders: list,
    decoders: list,
    optimizer: torch.optim.Optimizer,
    total_loss,
    contrastive_loss,
    recon_loss,
    vq_loss,
    scheduler=None,
    best_loss=None,
    best_metric_name="total_loss",
    scaler=None,
) -> None:
    """
    Save a training checkpoint to ``args.save_dir``.

    For VQ-VAE mode a single ``vqvae_model.pt`` is written.
    For VAE mode a ``checkpoint.pt`` plus per-modality ``encoder_<m>.pt`` files are written.
    MoCo queue state is included automatically when ``args.use_moco`` is True.

    Args:
        args: Parsed argument namespace.
        step: Current training step.
        encoders: List of encoder (or MoCoEncoder-wrapped) models.
        decoders: List of decoder models.
        optimizer: The optimizer whose state should be saved.
        total_loss: Scalar total loss value at this step.
        contrastive_loss: Scalar contrastive loss value.
        recon_loss: Scalar reconstruction loss value.
        vq_loss: Scalar VQ commitment loss value.
        scheduler: Optional LR scheduler whose state should be saved.
        best_loss: When not None, also saves a ``*_best.pt`` copy of the checkpoint.
    """
    if args.encoder_type == "vqvae":
        checkpoint_path = os.path.join(args.save_dir, "vqvae_model.pt")
        checkpoint = {
            "encoders": encoders[0].state_dict(),
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "rng_state": _capture_rng_state(),
            # Mirror the best-metric bookkeeping into the rolling checkpoint so
            # that resuming from the latest file (even when vqvae_best.pt is
            # missing or stale) preserves the selector's history.
            "best_metric_name": best_metric_name,
            "best_metric_value": float(best_loss) if best_loss is not None else None,
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        if getattr(args, "use_moco", False):
            from models.vqvae import MoCoEncoder

            if isinstance(encoders[0], MoCoEncoder):
                checkpoint["moco_queues"] = [q.cpu() for q in encoders[0].queues]
                checkpoint["moco_queue_ptrs"] = encoders[0].queue_ptrs.tolist()
                # Save view-1 queues when separate encoders are active
                if encoders[0]._separate_queues:
                    checkpoint["moco_queues_v1"] = [q.cpu() for q in encoders[0].queues_v1]
                    checkpoint["moco_queue_v1_ptrs"] = encoders[0].queue_v1_ptrs.tolist()
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"[CHECKPOINT] Step {step}: Saved VQ-VAE-2 to {checkpoint_path}")

    else:
        checkpoint_path = os.path.join(args.save_dir, "checkpoint.pt")
        checkpoint = {
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),
            "decoder": decoders[0].state_dict(),
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "rng_state": _capture_rng_state(),
            "best_metric_name": best_metric_name,
            "best_metric_value": float(best_loss) if best_loss is not None else None,
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        for m_idx, m in enumerate(args.modalities):
            checkpoint[f"encoder_{m}"] = encoders[m_idx].state_dict()
            encoder_path = os.path.join(args.save_dir, f"encoder_{m}.pt")
            torch.save(encoders[m_idx].state_dict(), encoder_path)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"[CHECKPOINT] Step {step}: Saved checkpoint to {args.save_dir}")

    # ── Best-model tracking ─────────────────────────────────────────────
    if best_loss is not None:
        suffix = "vqvae_best.pt" if args.encoder_type == "vqvae" else "checkpoint_best.pt"
        best_path = os.path.join(args.save_dir, suffix)
        checkpoint["best_metric_name"] = best_metric_name
        checkpoint["best_metric_value"] = float(best_loss)
        torch.save(checkpoint, best_path)
        logger.info(f"[CHECKPOINT] Step {step}: New best model " f"({best_metric_name}={best_loss:.4f}) → {best_path}")

    if args.save_all_checkpoints:
        m_idx = len(args.modalities) - 1
        m = args.modalities[m_idx]
        versioned_path = os.path.join(args.save_dir, f"encoder_{m}_{step:07d}.pt")
        torch.save(encoders[m_idx].state_dict(), versioned_path)
        logger.info(f"[CHECKPOINT] Step {step}: Saved versioned checkpoint to {versioned_path}")


def save_emergency_checkpoint(
    args,
    step: int,
    encoders: list,
    decoders: list,
    optimizer: torch.optim.Optimizer,
    reason: str = "unknown",
    scheduler=None,
) -> None:
    """
    Best-effort checkpoint written on unexpected interruption (OOM, crash, KeyboardInterrupt).

    Args:
        args: Parsed argument namespace.
        step: Current training step.
        encoders: List of encoder models.
        decoders: List of decoder models.
        optimizer: The optimizer.
        reason: Short description of why this checkpoint was triggered.
    """
    try:
        emergency_path = os.path.join(args.save_dir, "emergency_checkpoint.pt")
        if args.encoder_type == "vqvae":
            ckpt = {
                "encoders": encoders[0].state_dict(),
                "step": step,
                "optimizer_state_dict": optimizer.state_dict(),
                "reason": reason,
            }
        else:
            ckpt = {
                "step": step,
                "optimizer_state_dict": optimizer.state_dict(),
                "decoder": decoders[0].state_dict(),
                "reason": reason,
            }
            for m_idx, m in enumerate(args.modalities):
                ckpt[f"encoder_{m}"] = encoders[m_idx].state_dict()
        if scheduler is not None:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(ckpt, emergency_path)
        logger.warning(f"[EMERGENCY] Saved emergency checkpoint to {emergency_path} (reason: {reason})")
    except Exception as save_err:
        logger.error(f"[EMERGENCY] Failed to save emergency checkpoint: {save_err}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def _state_dicts_compatible(model: torch.nn.Module, saved_state_dict: dict) -> bool:
    """
    Return True when *saved_state_dict* is compatible with *model*.

    Compatibility requires:
    - Identical parameter names (keys).
    - Identical tensor shapes for every parameter.

    Args:
        model: The instantiated model to compare against.
        saved_state_dict: The ``state_dict`` loaded from disk.

    Returns:
        bool: ``True`` if every key and shape matches, ``False`` otherwise.
    """
    model_sd = model.state_dict()
    if set(model_sd.keys()) != set(saved_state_dict.keys()):
        return False
    for key in model_sd:
        if model_sd[key].shape != saved_state_dict[key].shape:
            return False
    return True


def _try_load_state_dict(
    model: torch.nn.Module,
    saved_state_dict: dict,
    label: str = "model",
) -> bool:
    """
    Load *saved_state_dict* into *model*, tolerating minor differences.

    Strategy:
    1. If keys & shapes match exactly → ``strict=True`` load.
    2. Otherwise, load only the parameters whose keys exist in both the model
       and the checkpoint **and** whose shapes match.  Missing and unexpected
       keys are logged as warnings, and shape-mismatched keys are skipped.

    Returns ``True`` if at least some weights were loaded, ``False`` if nothing
    could be loaded (e.g. zero overlapping keys).
    """
    if _state_dicts_compatible(model, saved_state_dict):
        model.load_state_dict(saved_state_dict, strict=True)
        return True

    model_sd = model.state_dict()
    saved_keys = set(saved_state_dict.keys())
    model_keys = set(model_sd.keys())

    missing_from_ckpt = model_keys - saved_keys
    unexpected_in_ckpt = saved_keys - model_keys
    shared_keys = model_keys & saved_keys

    shape_mismatched = []
    loadable = {}
    for key in shared_keys:
        if model_sd[key].shape == saved_state_dict[key].shape:
            loadable[key] = saved_state_dict[key]
        else:
            shape_mismatched.append(
                f"    {key}: checkpoint {tuple(saved_state_dict[key].shape)} " f"vs model {tuple(model_sd[key].shape)}"
            )

    if not loadable:
        logger.warning(f"  [{label}] No compatible weights found in checkpoint — cannot resume.")
        return False

    # Report differences
    n_total = len(model_sd)
    n_loaded = len(loadable)
    if missing_from_ckpt or unexpected_in_ckpt or shape_mismatched:
        logger.warning(
            f"  [{label}] Architecture partially changed — loading {n_loaded}/{n_total} " f"compatible parameters."
        )
        if missing_from_ckpt:
            logger.warning(
                f"  [{label}] {len(missing_from_ckpt)} new parameter(s) not in checkpoint "
                f"(will use random init):\n" + "\n".join(f"    {k}" for k in sorted(missing_from_ckpt))
            )
        if unexpected_in_ckpt:
            logger.warning(
                f"  [{label}] {len(unexpected_in_ckpt)} checkpoint parameter(s) no longer "
                f"in model (ignored):\n" + "\n".join(f"    {k}" for k in sorted(unexpected_in_ckpt))
            )
        if shape_mismatched:
            logger.warning(
                f"  [{label}] {len(shape_mismatched)} parameter(s) with shape mismatch "
                f"(skipped):\n" + "\n".join(shape_mismatched)
            )

    model.load_state_dict(loadable, strict=False)
    return True


def load_checkpoint(
    args,
    encoders: list,
    decoders: list,
    optimizer: torch.optim.Optimizer,
    device,
    loss_deques: dict,
    scheduler=None,
    scaler=None,
) -> int:
    """
    Restore training state from the most recent checkpoint, if one exists
    and ``--resume-training`` was passed.

    Args:
        args: Parsed argument namespace.  Must have ``resume_training`` attribute.
        encoders: List of encoder models (weights updated in-place).
        decoders: List of decoder models (weights updated in-place).
        optimizer: Optimizer (state updated in-place).
        device: Target device for tensor restoration.
        loss_deques: Dict mapping loss name → ``collections.deque`` to pre-fill
                     with the last saved value.  Expected keys:
                     ``'loss'``, ``'contrastive_loss'``, ``'recon_loss'``, ``'vq_loss'``.
        scheduler: Optional LR scheduler to restore state into.

    Returns:
        int: The step to resume from (``saved_step + 1``), or ``1`` if no
             compatible checkpoint is found or resuming was not requested.
    """
    if not getattr(args, "resume_training", False):
        logger.info("  --resume-training not set, starting fresh.")
        return 1

    if not os.path.exists(args.save_dir):
        return 1

    if args.encoder_type == "vqvae":
        checkpoint_path = os.path.join(args.save_dir, "vqvae_model.pt")
        if not os.path.exists(checkpoint_path):
            logger.info("  No VQ-VAE checkpoint found, starting fresh training.")
            return 1

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if not _try_load_state_dict(encoders[0], checkpoint["encoders"], label="VQ-VAE"):
            logger.warning(
                "  VQ-VAE checkpoint found but model architecture is completely incompatible "
                f"(checkpoint: {checkpoint_path}). Starting fresh training."
            )
            return 1

        logger.info(f"  Auto-resuming VQ-VAE training from checkpoint: {checkpoint_path}")
        # Try to restore optimizer state; skip if it doesn't match (e.g.
        # param count changed) — the optimizer will reinitialize cleanly.
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (ValueError, KeyError) as exc:
            logger.warning(f"  Optimizer state could not be restored ({exc}); using fresh optimizer.")
        step = checkpoint["step"] + 1

        if scheduler is not None:
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("  LR scheduler state restored from checkpoint.")
            else:
                # Old checkpoint without scheduler state — fast-forward to current step
                scheduler.last_epoch = step - 2
                scheduler.step()
                logger.info(f"  LR scheduler fast-forwarded to step {step - 1} (no saved state).")

        for key, deque in loss_deques.items():
            deque.append(checkpoint.get(key, 0))

        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logger.info("  AMP GradScaler state restored from checkpoint.")

        if "rng_state" in checkpoint:
            _restore_rng_state(checkpoint["rng_state"])
            logger.info("  RNG state (torch/cuda/numpy/python) restored from checkpoint.")

        if getattr(args, "use_moco", False) and "moco_queues" in checkpoint:
            from models.vqvae import MoCoEncoder

            if isinstance(encoders[0], MoCoEncoder):
                _restored = True
                for lvl, q_cpu in enumerate(checkpoint["moco_queues"]):
                    target_queue = encoders[0]._get_queue(lvl)
                    if target_queue.shape != q_cpu.shape:
                        logger.warning(
                            f"  MoCo queue shape mismatch at level {lvl}: "
                            f"checkpoint {q_cpu.shape} vs model {target_queue.shape}. "
                            f"Queues will be re-initialized (not restored)."
                        )
                        _restored = False
                        break
                    target_queue.copy_(q_cpu.to(device))
                if _restored:
                    encoders[0].queue_ptrs.copy_(torch.tensor(checkpoint["moco_queue_ptrs"], dtype=torch.long))
                    logger.info("  MoCo view-0 queue state restored from checkpoint.")

                # Restore view-1 queues (separate encoders)
                if _restored and encoders[0]._separate_queues and "moco_queues_v1" in checkpoint:
                    _v1_ok = True
                    for lvl, q_cpu in enumerate(checkpoint["moco_queues_v1"]):
                        target_queue = encoders[0]._get_queue(lvl, view=1)
                        if target_queue.shape != q_cpu.shape:
                            logger.warning(
                                f"  MoCo view-1 queue shape mismatch at level {lvl}. "
                                f"View-1 queues will be re-initialized."
                            )
                            _v1_ok = False
                            break
                        target_queue.copy_(q_cpu.to(device))
                    if _v1_ok:
                        encoders[0].queue_v1_ptrs.copy_(
                            torch.tensor(checkpoint["moco_queue_v1_ptrs"], dtype=torch.long)
                        )
                        logger.info("  MoCo view-1 queue state restored from checkpoint.")
                elif _restored and encoders[0]._separate_queues:
                    logger.warning(
                        "  Checkpoint has no view-1 queue data (old format). "
                        "View-1 queues start from random initialization."
                    )

        logger.info(f"  Checkpoint loaded successfully! Resuming from step {step}")
        logger.info(f"  Previous loss: {checkpoint.get('loss', 'N/A')}")
        return step

    else:
        checkpoint_path = os.path.join(args.save_dir, "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            logger.info("  No VAE checkpoint found, starting fresh training.")
            return 1

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Try to load all encoders and the decoder, tolerating minor differences.
        any_failed = False
        for m_idx, m in enumerate(args.modalities):
            key = f"encoder_{m}"
            if key not in checkpoint:
                logger.warning(f"  VAE checkpoint has no entry for encoder '{m}' — starting fresh training.")
                any_failed = True
                break
            if not _try_load_state_dict(encoders[m_idx], checkpoint[key], label=f"encoder-{m}"):
                logger.warning(
                    f"  VAE encoder '{m}' is completely incompatible with checkpoint — " f"starting fresh training."
                )
                any_failed = True
                break
        if not any_failed and not _try_load_state_dict(decoders[0], checkpoint["decoder"], label="decoder"):
            logger.warning("  VAE decoder is completely incompatible with checkpoint — " f"starting fresh training.")
            any_failed = True
        if any_failed:
            return 1

        logger.info(f"  Auto-resuming VAE training from checkpoint: {checkpoint_path}")
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (ValueError, KeyError) as exc:
            logger.warning(f"  Optimizer state could not be restored ({exc}); using fresh optimizer.")
        step = checkpoint["step"] + 1

        if scheduler is not None:
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("  LR scheduler state restored from checkpoint.")
            else:
                scheduler.last_epoch = step - 2
                scheduler.step()
                logger.info(f"  LR scheduler fast-forwarded to step {step - 1} (no saved state).")

        for key, deque in loss_deques.items():
            deque.append(checkpoint.get(key, 0))

        if "rng_state" in checkpoint:
            _restore_rng_state(checkpoint["rng_state"])
            logger.info("  RNG state (torch/cuda/numpy/python) restored from checkpoint.")

        logger.info(f"  Checkpoint loaded successfully! Resuming from step {step}")
        logger.info(f"  Previous loss: {checkpoint.get('loss', 'N/A')}")
        return step
