#!/usr/bin/env python3
"""Generate the Conv + MLP global/patch InfoNCE SLURM experiment; never submit."""

import argparse
import math
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import generate_encoder_ablation_slurm as slurm


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-loss-weight", type=float, default=1.0)
    parser.add_argument("--train-patch-grid", type=int, nargs=3, default=[8, 8, 8])
    parser.add_argument("--separate-spatial-readout", action="store_true")
    parser.add_argument("--cluster-config", type=Path, default=ROOT / "experiments/cluster/slurm_bio.yaml")
    parser.add_argument("--results-dir", default="/scratch/users/k24058220/encoder_patch_slurm_bio")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/generated")
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    config = slurm.ablations.comparison.read_json(ROOT / "experiments/encoder_comparison.json")
    slurm.ablations.comparison.validate_config(config)
    if args.seed not in config["seeds"]:
        parser.error("Choose a seed from the comparison recipe")
    if not math.isfinite(args.patch_loss_weight) or args.patch_loss_weight <= 0:
        parser.error("Patch loss weight must be finite and positive")
    spatial = config["shared"]["res"] // config["shared"]["downscale_factor"]
    if any(g < 1 or g > spatial for g in args.train_patch_grid):
        parser.error(f"Patch grid must fit the {spatial}^3 Conv map")
    grid = "x".join(map(str, args.train_patch_grid))
    name = f"conv_mlp_patch_g{grid}_w{args.patch_loss_weight:g}"
    tag = "_separate_spatial" if args.separate_spatial_readout else ""
    name += tag
    options = slurm.ablations.training_options(
        config, {"reference": "conv", "overrides": {"conv_readout": "mlp"}}, name, args.seed, args.results_dir
    )
    options.update(
        patch_loss_weight=args.patch_loss_weight, train_patch_grid=args.train_patch_grid, spatial_recovery_eval=True
    )
    if args.separate_spatial_readout:
        options["separate_spatial_readout"] = True
    command = shlex.join(["python", "scripts/generate_conv_patch_slurm.py", *arguments])
    script = slurm.render_script(
        config, options, slurm.load_resources(args.cluster_config), f"encoder-conv-patch{tag}-s{args.seed}", command
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"encoder_conv_mlp_patch{tag}_s{args.seed}.slurm_bio.sh"
    path.write_text(script)
    path.chmod(0o755)
    print(path)


if __name__ == "__main__":
    main()
