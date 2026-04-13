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

    # Ensure each sweep run gets a unique directory
    argv = [
        sys.executable,
        "training/main_multimodal.py",
        "--model-id",
        f"sweep-{run.id}",
    ]

    # Handle the constraint: content_size must be strictly less than hidden_channels
    if "vqvae_hidden_channels" in config:
        config["vqvae_embed_dim"] = config["vqvae_hidden_channels"]
        hidden = config["vqvae_hidden_channels"]

        # Only apply constraint if we are still sweeping over 'content_size'
        if "content_size" in config:
            content = config["content_size"]
            if content >= hidden:
                print(f"SKIP: content_size ({content}) >= vqvae_hidden_channels ({hidden}). " f"Marking run as failed.")
                wandb.log({"separation_score": 0.0})
                wandb.finish(exit_code=1)
                return

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
