"""Visualisation helpers — save decoded NIfTI images during training."""

import os

import numpy as np
import torch


def save_decoded_images(encoders, decoders, data, args, step: int) -> None:
    """
    Encode then decode the first sample and write original + reconstruction as NIfTI files.

    Only used in VAE mode (encoder + separate decoder).

    Args:
        encoders: List of encoder models.
        decoders: List of decoder models.
        data: Current batch dictionary (must contain key ``"image"``).
        args: Parsed argument namespace (needs ``args.save_dir``).
        step: Current training step (used in the output filename).
    """
    import nibabel as nib

    with torch.no_grad():
        samples = data["image"]
        img = samples[0][0:1]  # (1, 1, D, H, W) — first sample from first view

        hz = encoders[0](img)
        hz_flat = hz.view(hz.size(0), -1)
        decoded = decoders[0](hz_flat)

        decoded_np = decoded.squeeze().cpu().numpy()
        original_np = img.squeeze().cpu().numpy()

        affine_2mm = np.diag([2.0, 2.0, 2.0, 1.0])
        save_dir = os.path.join(args.save_dir, "decoded_images")
        os.makedirs(save_dir, exist_ok=True)

        nib.save(
            nib.Nifti1Image(original_np, affine=affine_2mm),
            f"{save_dir}/step_{step:05d}_original.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(decoded_np, affine=affine_2mm),
            f"{save_dir}/step_{step:05d}_decoded.nii.gz",
        )
        print(f"[SAVED] Decoded images at step {step} to {save_dir}/", flush=True)


def save_vqvae_decoded_images(vqvae_model, data, args, step: int) -> None:
    """
    Run the first sample through the VQ-VAE-2 and write original + reconstruction as NIfTI files.

    Args:
        vqvae_model: The VQ-VAE-2 (or MoCoEncoder-wrapped) model.
        data: Current batch dictionary (must contain key ``"image"``).
        args: Parsed argument namespace (needs ``args.save_dir``).
        step: Current training step (used in the output filename).
    """
    import nibabel as nib

    with torch.no_grad():
        samples = data["image"]
        img = samples[0][0:1]  # (1, 1, D, H, W)

        recon, diffs, _, _, _, _, _ = vqvae_model(img)

        decoded_np = recon.squeeze().cpu().numpy()
        original_np = img.squeeze().cpu().numpy()

        affine_2mm = np.diag([2.0, 2.0, 2.0, 1.0])
        save_dir = os.path.join(args.save_dir, "decoded_images")
        os.makedirs(save_dir, exist_ok=True)

        nib.save(
            nib.Nifti1Image(original_np, affine=affine_2mm),
            f"{save_dir}/step_{step:05d}_original.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(decoded_np, affine=affine_2mm),
            f"{save_dir}/step_{step:05d}_decoded.nii.gz",
        )
        vq_losses_str = ", ".join([f"L{i}:{d.item():.4f}" for i, d in enumerate(diffs)])
        print(
            f"[SAVED] VQ-VAE decoded at step {step} | VQ losses: {vq_losses_str}",
            flush=True,
        )
