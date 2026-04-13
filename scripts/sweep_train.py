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
    sys.stdout.flush()

    # Parse the arguments natively exactly as main_multimodal.py expects them
    parser = parse_args()

    # We must properly format list arguments. wandb passes strings like "[4, 5, 4]"
    # but argparse with nargs='+' expects multiple string arguments like "4" "5" "4"
    cleaned_args = []
    import ast

    for arg in cmd_args:
        if arg.startswith("[") and arg.endswith("]"):
            try:
                # Safely parse stringified lists like "[4, 5, 4]" into python lists
                parsed_list = ast.literal_eval(arg)
                cleaned_args.extend(str(x) for x in parsed_list)
            except Exception:
                cleaned_args.append(arg)
        else:
            cleaned_args.append(arg)

    print(f"Cleaned args for argparse: {' '.join(cleaned_args)}")
    sys.stdout.flush()

    try:
        args = parser.parse_args(cleaned_args)
    except SystemExit as e:
        print(f"Argparse failed and called sys.exit({e.code}). Arguments passed were:")
        print(cleaned_args)
        wandb.log({"separation_score": 0.0})
        wandb.finish(exit_code=1)
        sys.exit(1)

    try:
        # W&B has a known issue where it hijacks standard logging configurations.
        # We need to make sure the root logger doesn't suppress all python errors.
        import logging

        logging.getLogger().setLevel(logging.INFO)

        # Run the main training loop in the SAME process.
        # This prevents WandB daemon conflicts that happen when spawning subprocesses
        # and dropping connection locks, which fixes metrics not logging correctly.
        run_main(args)
    except Exception as e:
        import sys
        import traceback

        print(f"CRASH OCCURRED: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        wandb.log({"separation_score": 0.0})
        wandb.finish(exit_code=1)
        sys.exit(1)

    # Close the sweep-agent run politely after training finishes
    wandb.finish()
