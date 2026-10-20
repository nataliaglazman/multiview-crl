"""Train raw/centered/standardized global codes using an existing run's saved recipe.

Runs sequentially from fresh initialization, with the reference's data, initialization,
batching, and optimization settings. No checkpoint is resumed. Existing outputs are refused.
Preview without training using --dry-run. Run from the repository root with python -m.
"""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from scripts.launch import REPO_ROOT, encoder_local_command
from training import encoder_args

MODES = ("standardized", "centered", "raw")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-run", type=Path, required=True)
    p.add_argument("--modes", choices=MODES, nargs="+", default=["centered", "raw"])
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--out-dir", type=Path, help="Default: next to the reference run")
    p.add_argument("--seed", type=int, help="Override initialization and loader seeds; keep the saved data seed")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if len(set(args.modes)) != len(args.modes):
        p.error("Duplicate modes")
    if args.seed is not None and args.seed < 0:
        p.error("seed must be nonnegative")
    return args


def plan_runs(args):
    reference = args.reference_run.resolve()
    saved = json.loads((reference / "settings.json").read_text())
    # Unknown settings are errors: silently dropping one would weaken the matched comparison.
    cfg = vars(encoder_args.parse_args(encoder_args.cli_arguments(saved)))
    if cfg["contrastive_loss_type"] != "infonce" or cfg["mask_mode"] != "fixed":
        raise ValueError("This ablation requires global InfoNCE with a fixed content mask")
    output = (args.out_dir or reference.parent).resolve()
    planned = []
    for mode in args.modes:
        settings = dict(cfg)
        settings.update(
            standardize_code=mode == "standardized",
            center_code=mode == "centered",
            require_new_run=True,
            out_dir=str(output),
            model_id=f"{reference.name}_code_{mode}" + (f"_seed{args.seed}" if args.seed is not None else ""),
        )
        if args.device is not None:
            settings.update(device=args.device, no_cuda=args.device == "cpu")
        if args.seed is not None:
            settings.update(seed=args.seed, model_seed=args.seed, loader_seed=args.seed + 10000)
        settings = vars(encoder_args.parse_args(encoder_args.cli_arguments(settings)))
        destination = output / settings["model_id"]
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}; choose a new --out-dir")
        planned.append((mode, settings))
    return cfg, planned


def main(argv=None):
    args = parse_args(argv)
    reference, planned = plan_runs(args)
    print(
        f"Saved recipe: lesion target={reference['synthetic_lesion_target']}, "
        f"matching={reference['stratify_batches']} sampler={reference['stratify_sampler']} "
        f"K={reference['stratify_clusters']} neighbours={reference['stratify_neighbours']}, "
        f"batch={reference['batch_size']}, steps={reference['train_steps']}, "
        f"data seed={reference['data_seed']}",
        flush=True,
    )
    for mode, settings in planned:
        command, env = encoder_local_command(encoder_args.cli_arguments(settings))
        print(f"{mode}: {settings['out_dir']}/{settings['model_id']}", flush=True)
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            env["PYTHONUNBUFFERED"] = "1"
            subprocess.run(command, env=env, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
