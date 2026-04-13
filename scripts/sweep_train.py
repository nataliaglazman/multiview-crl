"""Wrapper script for W&B sweep agent.

Translates W&B sweep config values into command-line arguments for
training/main_multimodal.py, handling boolean flags, lists, and dynamic arguments.
"""

import os
import subprocess
import sys

import wandb


def main():
    run = wandb.init()
    config = dict(run.config)

    # Build command-line args
    argv = [
        sys.executable,
        "training/main_multimodal.py",
        "--model-id",
        f"sweep-{run.id}",  # Ensure each sweep run gets a unique directory
    ]

    # Enforce vqvae_embed_dim == vqvae_hidden_channels if we are sweeping over hidden_channels
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
                argv.append(arg_name)
        elif isinstance(value, (list, tuple)):
            argv.append(arg_name)
            argv.extend(str(v) for v in value)
        else:
            argv.append(arg_name)
            argv.append(str(value))

    # Pass dataroot/labels-path if set in environment overrides
    if os.environ.get("DATAROOT") and "dataroot" not in config:
        argv.extend(["--dataroot", os.environ["DATAROOT"]])
    if os.environ.get("LABELS_PATH") and "labels_path" not in config:
        argv.extend(["--labels-path", os.environ["LABELS_PATH"]])

    wandb.finish()  # Close the sweep-agent run; main_multimodal.py will init its own

    # Ensure PYTHONPATH is set so local modules are found
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    print(f"Running: {' '.join(argv)}")
    result = subprocess.run(argv, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
