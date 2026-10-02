#!/usr/bin/env python
"""Plan, train, evaluate and summarize paired native conv/ResNet experiments.

The launcher uses only the standard library. Child processes use the same Python
interpreter and require the project's normal training environment. No GPU jobs
are started by 'plan'. See training/ENCODER_COMPARISON.md.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("conv", "resnet18")
BOOLEAN_OPTIONAL = {"cache", "floor_eval", "cross_view_negs_only"}
STORE_TRUE = {
    "no_separate_encoders",
    "synthetic_clean_content",
    "synthetic_causal",
    "synthetic_hierarchical_content",
    "no_cuda",
    "require_new_run",
    "deterministic",
    "deterministic_warn_only",
    "hash_training_inputs",
}
RESERVED = {
    "seed",
    "data_seed",
    "model_seed",
    "loader_seed",
    "encoder_architecture",
    "out_dir",
    "model_id",
    "require_new_run",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    paths = [Path(__file__).resolve()]
    for directory in ("data", "models", "training", "eval", "utils"):
        paths.extend((ROOT / directory).rglob("*.py"))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(paths)}


def cli_arguments(options):
    arguments = []
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if key in BOOLEAN_OPTIONAL:
                arguments.append(flag if value else "--no-" + key.replace("_", "-"))
            elif key in STORE_TRUE:
                if value:
                    arguments.append(flag)
            else:
                raise ValueError(f"Unknown boolean training option: {key}")
        elif value is not None:
            arguments.append(flag)
            arguments.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return arguments


def validate_config(config):
    shared, evaluation = config["shared"], config["evaluation"]
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Provide unique, nonempty paired seeds")
    if RESERVED & shared.keys():
        raise ValueError(f"Shared settings cannot override pairing fields: {sorted(RESERVED & shared.keys())}")
    if (
        shared["contrastive_loss_type"] != "infonce"
        or shared["eval_pooling"] != "gap"
        or shared["best_metric"] != "none"
    ):
        raise ValueError("This protocol requires InfoNCE, GAP evaluation and fixed-step selection (best_metric=none)")
    if shared["contrastive_proj_dim"] != 0:
        raise ValueError("This protocol compares the actual content vectors; contrastive_proj_dim must be 0")
    if shared.get("cpu_threads", 0) < 1 or not shared.get("hash_training_inputs", False):
        raise ValueError("Pin positive cpu_threads and enable hash_training_inputs for verified pairing")
    if shared.get("deterministic_warn_only", False) and not shared.get("deterministic", False):
        raise ValueError("deterministic_warn_only requires deterministic=true")
    if not 2 <= shared["batch_size"] <= shared["num_train_samples"]:
        raise ValueError("Training needs a full batch of at least two subjects")
    if not shared["batch_size"] <= evaluation["num_samples"] <= shared["num_train_samples"]:
        raise ValueError("Evaluation num_samples must be >= training batch_size and <= num_train_samples")
    if evaluation["num_samples"] < 20 or evaluation["probe_samples"] < 20:
        raise ValueError("Need at least 20 evaluation/probe subjects")
    if evaluation["probe_samples"] != shared["num_val_samples"]:
        raise ValueError(
            "Keep probe_samples equal to num_val_samples so validation normalization uses the same reference"
        )
    native = min(shared["res"] // shared["downscale_factor"], (shared["res"] + 31) // 32)
    if not 1 <= evaluation["patch_grid"] <= native:
        raise ValueError(f"The common patch grid must fit BOTH backbones (maximum {native})")
    if not 0 < shared["content_channels"] <= shared["latent_dim"]:
        raise ValueError("Invalid content/latent dimensions")
    if min(shared["train_steps"], shared["eval_every"], evaluation["batch_size"], evaluation["lesion_shuffles"]) < 1:
        raise ValueError("Steps, batch size and shuffle count must be positive")
    cli_arguments(shared)


def run_options(config, output, seed, architecture):
    return {
        **config["shared"],
        "seed": seed,
        "data_seed": config["data_seed"],
        "model_seed": seed,
        "loader_seed": seed + 10000,
        "encoder_architecture": architecture,
        "require_new_run": True,
        "out_dir": str(output / "runs"),
        "model_id": f"{architecture}_s{seed}",
    }


def training_command(options):
    return [sys.executable, "-m", "training.main_conv_synthetic", *cli_arguments(options)]


def ensure_manifest(output, config, create=False):
    expected = {"schema_version": 1, "config": config, "sources": source_hashes()}
    path = output / "comparison_manifest.json"
    if not path.exists():
        if not create:
            raise ValueError("No comparison manifest; train this experiment first")
        output.mkdir(parents=True, exist_ok=True)
        # Publish a complete file atomically, including when two jobs start together.
        with tempfile.NamedTemporaryFile(mode="w", dir=output, delete=False) as stream:
            json.dump(expected, stream, indent=2)
            temporary = Path(stream.name)
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        finally:
            temporary.unlink()
    if read_json(path) != expected:
        raise ValueError("Configuration or source code differs from the saved experiment. Use a new --output-dir")


def execute(command, log, cpu_threads=1):
    print("\n" + shlex.join(command), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS=str(cpu_threads),
        OPENBLAS_NUM_THREADS=str(cpu_threads),
        MKL_NUM_THREADS=str(cpu_threads),
    )
    with log.open("a") as stream:
        stream.write("\n$ " + shlex.join(command) + "\n")
        with subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            if process.wait():
                raise RuntimeError(f"Command failed; see {log}")


def validate_run(run, options, require_receipt=True):
    settings = read_json(run / "settings.json")
    differences = [key for key, value in options.items() if settings.get(key) != value]
    if differences:
        raise ValueError(f"{run.name}: saved settings differ in {differences}")
    progress = read_json(run / "training_progress.json")
    if progress["status"] != "complete" or progress["step"] != options["train_steps"]:
        raise ValueError(f"{run.name}: training is incomplete or at a different step")
    hashes = {name: file_hash(run / name) for name in ("model.pt", "model_init.pt")}
    if require_receipt and read_json(run / "comparison_receipt.json") != hashes:
        raise ValueError(f"{run.name}: checkpoint files changed after training")
    return progress, hashes


def evaluate_run(run, config, hashes, tag, skip_completed):
    directory = run / tag
    receipt = directory / "complete.json"
    expected = {"checkpoint_hashes": hashes, "evaluation": config["evaluation"]}
    if receipt.exists() and skip_completed:
        if read_json(receipt) != expected:
            raise ValueError(f"{run.name}: evaluation provenance differs")
        print(f"Skipping completed evaluation: {directory}")
        return
    if directory.exists():
        raise ValueError(
            f"Evaluation directory exists: {directory}. Use --skip-completed or a new --evaluation-tag after a failed run"
        )
    directory.mkdir(parents=True)
    e = config["evaluation"]
    base = ["--run-dir", str(run), "--checkpoint", "model.pt", "--batch-size", str(e["batch_size"])]
    device = ["--no-cuda"] if config["shared"].get("no_cuda", False) else []
    audit = [
        sys.executable,
        "-m",
        "eval.encoder.encoder_generalization_audit",
        *base,
        *device,
        "--num-samples",
        str(e["num_samples"]),
        "--probe-samples",
        str(e["probe_samples"]),
        "--retrieval-draws",
        str(e["retrieval_draws"]),
        "--seed",
        str(e["seed"]),
        "--skip-bn-recalibration",
        "--out-dir",
        str(directory / "global_path"),
    ]
    threads = config["shared"]["cpu_threads"]
    execute(audit, directory / "global_path.log", threads)
    score = [
        sys.executable,
        "-m",
        "eval.protocol.score_checkpoint",
        *base,
        *device,
        "--num-samples",
        str(config["shared"]["num_val_samples"]),
        "--no-graph",
        "--no-dci",
    ]
    execute(
        [
            *score,
            "--pooling",
            "gap",
            "--lesion-analysis",
            "--lesion-grids",
            "1",
            str(e["patch_grid"]),
            "--lesion-shuffles",
            str(e["lesion_shuffles"]),
            "--lesion-seed",
            str(e["seed"]),
            "--out",
            str(directory / "validation_gap_lesions.json"),
        ],
        directory / "validation_gap_lesions.log",
        threads,
    )
    execute(
        [
            *score,
            "--pooling",
            "patch",
            "--patch-grid",
            *([str(e["patch_grid"])] * 3),
            "--out",
            str(directory / "validation_patch.json"),
        ],
        directory / "validation_patch.log",
        threads,
    )
    if {name: file_hash(run / name) for name in hashes} != hashes:
        raise ValueError("Checkpoint changed during evaluation")
    write_json(receipt, expected)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output, config, seeds, tag):
    scores, runtime, indexed = [], [], {}
    for seed in seeds:
        pair = {}
        for architecture in ARCHITECTURES:
            options = run_options(config, output, seed, architecture)
            run = Path(options["out_dir"]) / options["model_id"]
            progress, hashes = validate_run(run, options)
            receipt = read_json(run / tag / "complete.json")
            if receipt != {"checkpoint_hashes": hashes, "evaluation": config["evaluation"]}:
                raise ValueError(f"{run.name}: evaluation receipt differs")
            report = read_json(run / tag / "global_path" / "report.json")
            if report["status"] != "complete" or report["checkpoint_sha256"] != hashes["model.pt"]:
                raise ValueError(f"{run.name}: incomplete or mismatched test audit")
            pair[architecture] = (progress, report)
            runtime.append({"seed": seed, "architecture": architecture, **progress})
            for row in report["probes"]:
                if row["arm"] == "original" and row["stage"] == "content" and row["condition"] == "observed":
                    score = {
                        "seed": seed,
                        "architecture": architecture,
                        "view": row["view"],
                        "target": row["target"],
                        "probe": row["probe"],
                        "test_r2": row["test_r2"],
                    }
                    scores.append(score)
                    indexed[(seed, architecture, row["view"], row["target"], row["probe"])] = row["test_r2"]
        a, b = pair["conv"], pair["resnet18"]
        if a[0]["batch_order_sha256"] != b[0]["batch_order_sha256"]:
            raise ValueError(f"Seed {seed}: training subject order differed")
        if not a[0]["training_input_sha256"] or a[0]["training_input_sha256"] != b[0]["training_input_sha256"]:
            raise ValueError(f"Seed {seed}: actual training images differed")
        if any(
            a[0][key] != b[0][key]
            for key in (
                "torch_version",
                "numpy_version",
                "cpu_threads",
                "optimizer_defaults",
                "deterministic_algorithms",
                "deterministic_warn_only",
            )
        ):
            raise ValueError(f"Seed {seed}: runtime or optimizer defaults differed")
        if a[1]["cohorts"] != b[1]["cohorts"] or a[1]["probe_split"] != b[1]["probe_split"]:
            raise ValueError(f"Seed {seed}: evaluation subjects or probe splits differed")
        for split in ("train", "val", "test"):
            if a[1]["state"]["original"][split]["input_sha256"] != b[1]["state"]["original"][split]["input_sha256"]:
                raise ValueError(f"Seed {seed}: {split} images differed")
    groups = sorted({(r["view"], r["target"], r["probe"]) for r in scores})
    summary = []
    for view, target, probe in groups:
        values = {arch: [indexed[(seed, arch, view, target, probe)] for seed in seeds] for arch in ARCHITECTURES}
        valid = [
            (a, b)
            for a, b in zip(values["conv"], values["resnet18"])
            if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
        ]
        deltas = [b - a for a, b in valid]
        summary.append(
            {
                "view": view,
                "target": target,
                "probe": probe,
                "paired_seeds": len(valid),
                "conv_mean_r2": statistics.mean(a for a, _ in valid) if valid else None,
                "resnet18_mean_r2": statistics.mean(b for _, b in valid) if valid else None,
                "resnet_minus_conv_mean": statistics.mean(deltas) if deltas else None,
                "paired_difference_sd": statistics.stdev(deltas) if len(deltas) > 1 else None,
            }
        )
    if not scores:
        raise ValueError("No observed global-content test scores found")
    destination = output / f"summary_{tag}"
    destination.mkdir(exist_ok=True)
    write_csv(destination / "test_scores.csv", scores)
    write_csv(destination / "paired_comparison.csv", summary)
    write_json(destination / "training_costs.json", runtime)
    write_json(
        destination / "verification.json",
        {
            "paired_seeds": seeds,
            "data_seed": config["data_seed"],
            "batch_order_and_evaluation_images_match": True,
            "scope": "Native architecture comparison; parameters, normalization, stride and readout are not equalized.",
        },
    )
    print(f"Verified paired batches and images. Results: {destination}")
    for row in summary:
        if row["probe"] == "ridge" and row["conv_mean_r2"] is not None:
            print(
                f"{row['view']:5} {row['target']:20} conv={row['conv_mean_r2']:+.3f} "
                f"resnet={row['resnet18_mean_r2']:+.3f} delta={row['resnet_minus_conv_mean']:+.3f}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "train", "evaluate", "summarize"))
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/encoder_comparison.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/encoder_comparison")
    parser.add_argument("--only-seed", type=int)
    parser.add_argument("--only-architecture", choices=ARCHITECTURES)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--evaluation-tag", default="evaluation")
    args = parser.parse_args(argv)
    config, output = read_json(args.config), args.output_dir.resolve()
    validate_config(config)
    if args.only_seed is not None and args.only_seed not in config["seeds"]:
        parser.error("--only-seed must occur in the configuration")
    if not args.evaluation_tag or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.evaluation_tag
    ):
        parser.error("--evaluation-tag must contain only letters, digits, underscores or hyphens")
    seeds = [args.only_seed] if args.only_seed is not None else config["seeds"]
    architectures = [args.only_architecture] if args.only_architecture else ARCHITECTURES
    if args.action != "plan":
        ensure_manifest(output, config, create=args.action == "train")
    if args.action == "summarize":
        if args.only_architecture:
            parser.error("Summarization requires both architectures")
        summarize(output, config, seeds, args.evaluation_tag)
        return
    if args.action == "evaluate":
        # Validate every selected checkpoint before launching any costly audits.
        # A successful first arm must not hide an incomplete paired training run.
        ready = []
        for seed in seeds:
            for architecture in architectures:
                options = run_options(config, output, seed, architecture)
                run = Path(options["out_dir"]) / options["model_id"]
                try:
                    _, hashes = validate_run(run, options)
                except (ValueError, FileNotFoundError, KeyError) as error:
                    raise ValueError(f"Evaluation not started: {error}") from error
                ready.append((run, hashes))
        for run, hashes in ready:
            evaluate_run(run, config, hashes, args.evaluation_tag, args.skip_completed)
        return
    for seed in seeds:
        for architecture in architectures:
            options = run_options(config, output, seed, architecture)
            command = training_command(options)
            run = Path(options["out_dir"]) / options["model_id"]
            if args.action == "plan":
                print(shlex.join(command))
            elif args.action == "train":
                if run.exists():
                    if not args.skip_completed:
                        raise ValueError(f"Run exists: {run}. Use --skip-completed for verified completed runs")
                    validate_run(run, options)
                    print(f"Skipping completed run: {run}")
                    continue
                execute(command, output / "logs" / f"{run.name}.log", config["shared"]["cpu_threads"])
                _, hashes = validate_run(run, options, require_receipt=False)
                write_json(run / "comparison_receipt.json", hashes)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as error:
        raise SystemExit(str(error)) from error
