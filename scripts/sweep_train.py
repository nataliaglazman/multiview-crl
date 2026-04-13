"""Wrapper script for W&B sweep agent.

Translates W&B sweep config values into command-line arguments for
training/main_multimodal.py, handling:
  - Boolean flags (patch_contrastive, etc.)
  - The content_size < vqvae_hidden_channels constraint
  - Setting vqvae_embed_dim = vqvae_hidden_channels
"""

import sys

import wandb


def main():
    run = wandb.init()
    config = dict(run.config)

    # --- Constraint: content_size must be < hidden_channels ---
    hidden = config.get("vqvae_hidden_channels", 64)
    content = config.get("content_size", 16)
    if content >= hidden:
        print(f"SKIP: content_size ({content}) >= vqvae_hidden_channels ({hidden}). " f"Marking run as failed.")
        wandb.log({"separation_score": 0.0})
        wandb.finish(exit_code=1)
        return

    # Build command-line args
    argv = [
        sys.executable,
        "training/main_multimodal.py",
        "--scale-contrastive-loss",
        str(config["scale_contrastive_loss"]),
        "--scale-recon-loss",
        str(config["scale_recon_loss"]),
        "--vqvae-hidden-channels",
        str(hidden),
        "--vqvae-embed-dim",
        str(hidden),  # always equal
        "--content-size",
        str(content),
        "--mask-mode",
        config["mask_mode"],
        "--contrastive-loss-type",
        config["contrastive_loss_type"],
        "--tau",
        str(config["tau"]),
        "--lr",
        str(config["lr"]),
        "--bt-lambda",
        str(config.get("bt_lambda", 0.005)),
        # Fixed params
        "--encoder-type",
        "vqvae",
        "--dataset_name",
        "ADNI_registered",
        "--batch-size",
        str(config.get("batch_size", 2)),
        "--gradient-accumulation-steps",
        str(config.get("gradient_accumulation_steps", 8)),
        "--train-steps",
        str(config.get("train_steps", 50000)),
        "--vqvae-nb-levels",
        "3",
        "--content-style-levels",
        "0",
        "1",
        "2",
        # Boolean flags (always on)
        "--separate-encoders",
        "--quantize-style",
        "--use-wandb",
        "--evaluate",
        "--use-amp",
    ]

    # Conditional boolean flags
    if config.get("patch_contrastive", False):
        argv.append("--patch-contrastive")

    # Pass dataroot/labels-path if set in environment
    import os

    if os.environ.get("DATAROOT"):
        argv.extend(["--dataroot", os.environ["DATAROOT"]])
    if os.environ.get("LABELS_PATH"):
        argv.extend(["--labels-path", os.environ["LABELS_PATH"]])

    wandb.finish()  # Close the sweep-agent run; main_multimodal.py will init its own

    import subprocess

    print(f"Running: {' '.join(argv)}")
    result = subprocess.run(argv)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
