"""Checkpoint save and load helpers for multiview-CRL training."""

import logging
import os

import torch

logger = logging.getLogger("multiview_crl")


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
        }
        if getattr(args, "use_moco", False):
            from models.vqvae import MoCoEncoder

            if isinstance(encoders[0], MoCoEncoder):
                checkpoint["moco_queues"] = [q.cpu() for q in encoders[0].queues]
                checkpoint["moco_queue_ptrs"] = list(encoders[0].queue_ptrs)
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
        }
        for m_idx, m in enumerate(args.modalities):
            checkpoint[f"encoder_{m}"] = encoders[m_idx].state_dict()
            encoder_path = os.path.join(args.save_dir, f"encoder_{m}.pt")
            torch.save(encoders[m_idx].state_dict(), encoder_path)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"[CHECKPOINT] Step {step}: Saved checkpoint to {args.save_dir}")

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


def load_checkpoint(
    args,
    encoders: list,
    decoders: list,
    optimizer: torch.optim.Optimizer,
    device,
    loss_deques: dict,
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

        if not _state_dicts_compatible(encoders[0], checkpoint["encoders"]):
            logger.warning(
                "  VQ-VAE checkpoint found but model architecture does not match "
                f"(checkpoint: {checkpoint_path}). Starting fresh training."
            )
            return 1

        logger.info(f"  Auto-resuming VQ-VAE training from checkpoint: {checkpoint_path}")
        encoders[0].load_state_dict(checkpoint["encoders"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint["step"] + 1

        for key, deque in loss_deques.items():
            deque.append(checkpoint.get(key, 0))

        if getattr(args, "use_moco", False) and "moco_queues" in checkpoint:
            from models.vqvae import MoCoEncoder

            if isinstance(encoders[0], MoCoEncoder):
                for lvl, q_cpu in enumerate(checkpoint["moco_queues"]):
                    encoders[0]._get_queue(lvl).copy_(q_cpu.to(device))
                encoders[0].queue_ptrs = list(checkpoint["moco_queue_ptrs"])
                logger.info("  MoCo queue state restored from checkpoint.")

        logger.info(f"  Checkpoint loaded successfully! Resuming from step {step}")
        logger.info(f"  Previous loss: {checkpoint.get('loss', 'N/A')}")
        return step

    else:
        checkpoint_path = os.path.join(args.save_dir, "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            logger.info("  No VAE checkpoint found, starting fresh training.")
            return 1

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Verify all encoders and the decoder are compatible before loading anything.
        for m_idx, m in enumerate(args.modalities):
            key = f"encoder_{m}"
            if key not in checkpoint or not _state_dicts_compatible(encoders[m_idx], checkpoint[key]):
                logger.warning(
                    f"  VAE checkpoint found but encoder '{m}' architecture does not match "
                    f"(checkpoint: {checkpoint_path}). Starting fresh training."
                )
                return 1
        if not _state_dicts_compatible(decoders[0], checkpoint["decoder"]):
            logger.warning(
                "  VAE checkpoint found but decoder architecture does not match "
                f"(checkpoint: {checkpoint_path}). Starting fresh training."
            )
            return 1

        logger.info(f"  Auto-resuming VAE training from checkpoint: {checkpoint_path}")
        for m_idx, m in enumerate(args.modalities):
            encoders[m_idx].load_state_dict(checkpoint[f"encoder_{m}"])
        decoders[0].load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint["step"] + 1

        for key, deque in loss_deques.items():
            deque.append(checkpoint.get(key, 0))

        logger.info(f"  Checkpoint loaded successfully! Resuming from step {step}")
        logger.info(f"  Previous loss: {checkpoint.get('loss', 'N/A')}")
        return step
