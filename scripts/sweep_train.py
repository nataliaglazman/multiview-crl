"""Wrapper script for W&B sweep agent.

Translates W&B sweep config values into command-line arguments for
training/main_multimodal.py, handling boolean flags, lists, and dynamic arguments.
"""

import os
import sys

import wandb

# Add project root to python path so we can import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.main_multimodal import main as run_main
from utils.config import parse_args


def main():
    run = wandb.init()
    config = dict(run.config)

    # Build command-line args for argparse
    cmd_args = [
        "--model-id",
        f"sweep-{run.id}",
    ]

    # Ensure vqvae_embed_dim == vqvae_hidden_channels if we are sweeping over hidden_channels
    if "vqvae_hidden_channels" in config:
        config["vqvae_embed_dim"] = config["vqvae_hidden_channels"]

    # Dynamically inject all parameters from the wandb config
    for key, value in config.items():
        if key.startswith("_"):
            continue  # Skip internal wandb keys

        # Convert underscores to hyphens for most arguments,
        # except for a few anomalies in argparse definitions (like dataset_name)
        if key == "dataset_name":
            arg_name = "--dataset_name"
        else:
            arg_name = f"--{key.replace('_', '-')}"

        if isinstance(value, bool):
            if value:
                cmd_args.append(arg_name)
        elif isinstance(value, (list, tuple)):
            cmd_args.append(arg_name)
            cmd_args.extend(str(v) for v in value)
        else:
            cmd_args.append(arg_name)
            cmd_args.append(str(value))

    # Pass dataroot/labels-path if set in environment overrides
    if os.environ.get("DATAROOT") and "dataroot" not in config:
        cmd_args.extend(["--dataroot", os.environ["DATAROOT"]])
    if os.environ.get("LABELS_PATH") and "labels_path" not in config:
        cmd_args.extend(["--labels-path", os.environ["LABELS_PATH"]])

    print(f"Parsed args for run_main: {' '.join(cmd_args)}")

    # Parse the arguments natively exactly as main_multimodal.py expects them
    parser = parse_args()
    args = parser.parse_args(cmd_args)

    # Run the main training loop in the SAME process.
    # This prevents WandB daemon conflicts that happen when spawning subprocesses
    # and dropping connection locks, which fixes metrics not logging correctly.
    run_main(args)

    # Close the sweep-agent run politely after training finishes
    wandb.finish()
