#!/usr/bin/env python3
"""Controlled residual-detector comparison; plan/train/evaluate/summarize/all.

Uses the existing Conv-MLP recipe, independent global/residual gradient clipping,
the same training data/PCA reference/batch order, and final-step checkpoints.
No labels or interventions enter a training loss. Only the audit selects heads.
"""

import argparse
import csv
import os
import shlex
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import compare_encoders as comparison  # noqa: E402

ARMS = ("frozen_shared", "shared", "separate", "separate_conv")


def execute(command, log, threads):
    # Reuse the local launcher's optional-dependency workaround. LPIPS is never
    # constructed by encoder-only training; the stand-in raises if that changes.
    from scripts.run_encoder_mps import lpips_stand_in

    directory = lpips_stand_in()
    previous = os.environ.get("PYTHONPATH")
    try:
        if directory is not None:
            os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (str(directory), previous)))
        comparison.execute(command, log, threads)
    finally:
        if previous is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous
        if directory is not None:
            shutil.rmtree(directory)


def training_options(args, config, arm, seed):
    options = comparison.run_options(config, args.output_dir.resolve(), seed, "conv")
    options.update(
        model_id=f"{arm}_s{seed}",
        device=args.device,
        conv_readout="mlp",
        norm_type="layer",
        synthetic_lesion_target="position",
        synthetic_lesion_intensity="styled",
        lesion_keypoints=4,
        lesion_input="residual",
        lesion_temperature=0.03,
        lesion_normative_components=20,
        lesion_normative_subjects=300,
        lesion_head_init="positive",
        lesion_pairing="cross_modal",
        lesion_detector="shared" if arm == "frozen_shared" else arm,
        lesion_branch_frozen=arm == "frozen_shared",
        lesion_localization_eval=True,
        lesion_loss_weight=1.0,
        lesion_decorrelation_weight=0.0,
        patch_loss_weight=0.0,
        global_pool="gap",
        floor_eval=True,
        best_metric="none",
        train_steps=args.train_steps,
        eval_every=args.eval_every,
    )
    return options


def audit_command(args, options):
    run = Path(options["out_dir"]) / options["model_id"]
    return [
        sys.executable,
        "-m",
        "eval.lesion.lesion_detector_audit",
        "--run-dir",
        str(run),
        "--out-dir",
        str(run / "evaluation/native_lesion"),
        "--device",
        args.device,
        "--num-samples",
        str(args.test_samples),
        "--validation-samples",
        str(args.validation_samples),
        "--movement-subjects",
        str(args.movement_subjects),
        "--batch-size",
        str(args.audit_batch_size),
    ]


def manifest(args, config):
    # Exclude the action and arm/seed subset: separate jobs may execute the same
    # immutable plan. All output-producing options and source files are fixed.
    settings = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ("action", "arms", "seeds")
    }
    return dict(
        settings=settings,
        config=config,
        sources={
            **comparison.source_hashes(),
            str(Path(__file__).relative_to(ROOT)): comparison.file_hash(__file__),
        },
    )


def summarize(args):
    from collections import defaultdict

    rows, paired, normative = [], defaultdict(list), defaultdict(list)
    for seed in args.seeds:
        for arm in args.arms:
            run = args.output_dir / "runs" / f"{arm}_s{seed}"
            progress = comparison.read_json(run / "training_progress.json")
            if progress["status"] != "complete":
                raise ValueError(f"Incomplete training: {run}")
            paired[seed].append(progress.get("training_input_sha256"))
            report = comparison.read_json(run / "evaluation/native_lesion/report.json")
            if report["status"] != "complete":
                raise ValueError(f"Incomplete audit: {run}")
            normative[seed].append(report["initial"]["normative_sha256"])
            for state in ("initial", "trained"):
                checkpoint = "model_init.pt" if state == "initial" else "model.pt"
                if report[state]["checkpoint_sha256"] != comparison.file_hash(run / checkpoint):
                    raise ValueError(f"Checkpoint changed since audit: {run / checkpoint}")
            for loc in report["localization"]:
                move = next(m for m in report["movement"] if all(m[k] == loc[k] for k in ("arm", "view", "method")))
                rows.append(
                    dict(
                        experiment=arm,
                        seed=seed,
                        **loc,
                        movement_skill=move["movement_skill"],
                        movement_rmse_vox=move["movement_rmse_vox"],
                        delta_movement_skill_vs_initial=move["delta_movement_skill_vs_initial"],
                    )
                )
    for seed, hashes in paired.items():
        if None in hashes or len(set(hashes)) != 1:
            raise ValueError(f"Training input digests do not match for seed {seed}")
        if None in normative[seed] or len(set(normative[seed])) != 1:
            raise ValueError(f"PCA reference models do not match for seed {seed}")
    path = args.output_dir / "summary.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Trained, validation-selected native head: mean voxel error / hit <=3 / movement skill")
    for row in rows:
        if row["arm"] == "trained" and row["method"] == "selected_validation_head":
            skill = row["movement_skill"] if row["movement_skill"] is not None else float("nan")
            print(
                f"  {row['experiment']:14s} s{row['seed']} {row['view']:5s} "
                f"{row['mean_error_vox']:.3f} / {row['hit_within_3_vox']:.3f} / {skill:+.3f}"
            )
    print(f"Saved {path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "train", "evaluate", "summarize", "all"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "experiments/encoder_comparison.json")
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--train-steps", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--test-samples", type=int, default=400)
    p.add_argument("--validation-samples", type=int, default=200)
    p.add_argument("--movement-subjects", type=int, default=64)
    p.add_argument("--audit-batch-size", type=int, default=4)
    args = p.parse_args(argv)
    if (
        min(
            args.train_steps,
            args.eval_every,
            args.test_samples,
            args.validation_samples,
            args.movement_subjects,
            args.audit_batch_size,
        )
        < 1
    ):
        p.error("Step/sample counts must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.arms)) != len(args.arms):
        p.error("Seeds and arms must be unique")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = comparison.read_json(args.config)
    comparison.validate_config(config)
    if any(seed not in config["seeds"] for seed in args.seeds):
        raise ValueError("Select seeds from the comparison recipe")
    jobs = [training_options(args, config, arm, seed) for seed in args.seeds for arm in args.arms]
    if args.action == "plan":
        for options in jobs:
            print(shlex.join(comparison.training_command(options)))
            print(shlex.join(audit_command(args, options)))
        return
    expected = manifest(args, config)
    path = args.output_dir / "lesion_comparison_manifest.json"
    if path.exists():
        if comparison.read_json(path) != expected:
            raise ValueError("Comparison settings/source differ; use a new --output-dir")
    else:
        if args.action not in ("train", "all"):
            raise ValueError("No training manifest found")
        comparison.write_json(path, expected)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for options in jobs:
        run = Path(options["out_dir"]) / options["model_id"]
        if args.action in ("train", "all"):
            if run.exists():
                comparison.validate_run(run, options, require_receipt=False)
                print(f"Verified completed run; reusing {run}", flush=True)
            else:
                execute(
                    comparison.training_command(options),
                    args.output_dir / "logs" / f"{run.name}.log",
                    options["cpu_threads"],
                )
                comparison.validate_run(run, options, require_receipt=False)
        if args.action in ("evaluate", "all"):
            _, hashes = comparison.validate_run(run, options, require_receipt=False)
            report = run / "evaluation/native_lesion/report.json"
            if report.exists() and comparison.read_json(report).get("status") == "complete":
                saved = comparison.read_json(report)
                if any(
                    saved[state]["checkpoint_sha256"] != hashes[checkpoint]
                    for state, checkpoint in (("initial", "model_init.pt"), ("trained", "model.pt"))
                ):
                    raise ValueError(f"Checkpoint changed since audit: {run}")
                print(f"Reusing completed audit {report}", flush=True)
            else:
                execute(
                    audit_command(args, options),
                    args.output_dir / "logs" / f"{run.name}_audit.log",
                    options["cpu_threads"],
                )
    if args.action in ("summarize", "all"):
        summarize(args)


if __name__ == "__main__":
    main()
