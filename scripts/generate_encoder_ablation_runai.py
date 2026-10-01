#!/usr/bin/env python3
"""Generate encoder ablation Run:ai scripts without submitting jobs.

Uses the matched encoder recipe and existing Run:ai cluster resource settings.
See training/ENCODER_ABLATIONS_RUNAI.md. Requires PyYAML; does not import torch.
"""

import argparse
import shlex
import sys
from pathlib import Path, PurePosixPath

# Support both `python scripts/...py` and module imports from repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import compare_encoders as comparison  # noqa: E402
from scripts import launch  # noqa: E402

MODEL_DEFAULTS = {"conv_readout": "linear", "resnet_norm": "batch", "resnet_output_stride": 32}


def training_options(config, variant, name, seed, results_dir):
    overrides = variant["overrides"]
    reference = variant["reference"]
    if reference not in comparison.ARCHITECTURES:
        raise ValueError(f"Unknown reference encoder: {reference}")
    if overrides.keys() - MODEL_DEFAULTS.keys():
        raise ValueError("Variants may change only conv_readout, resnet_norm or resnet_output_stride")
    if any(config["shared"].get(key, value) != value for key, value in MODEL_DEFAULTS.items()):
        raise ValueError("The shared recipe must retain baseline architecture options")
    options = comparison.run_options(config, PurePosixPath(results_dir), seed, reference)
    options.update(MODEL_DEFAULTS)
    options.update(overrides)
    options["model_id"] = f"{name}_s{seed}"
    if options["conv_readout"] not in ("linear", "mlp") or options["resnet_norm"] not in ("batch", "group"):
        raise ValueError("Invalid readout or normalization")
    if options["resnet_output_stride"] not in (8, 16, 32):
        raise ValueError("ResNet output stride must be 8, 16 or 32")
    if reference == "conv" and (options["resnet_norm"] != "batch" or options["resnet_output_stride"] != 32):
        raise ValueError("ResNet options cannot change a conv run")
    if reference == "resnet18" and options["conv_readout"] != "linear":
        raise ValueError("conv_readout cannot change a ResNet run")
    return options


def render_script(config, options, runai, job_name, recipe_label, regeneration):
    repo = str(runai["repo_path"])
    threads = str(config["shared"]["cpu_threads"])
    environment = {str(k): str(v) for k, v in (runai.get("environment") or {}).items()}
    environment.update(
        PYTHONPATH=repo,
        PYTHONUNBUFFERED="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        **{name: threads for name in launch.THREAD_ENV_VARS},
    )
    if any(not name.isidentifier() or not name.isascii() for name in environment):
        raise ValueError("Invalid environment variable name in cluster config")
    if any("\n" in value or "\r" in value for value in [repo, *environment.values()]):
        raise ValueError("Newlines are not supported in folded container commands")
    cli = comparison.cli_arguments(options)
    if any("\n" in value or "\r" in value for value in cli):
        raise ValueError("Newlines are not supported in training arguments")
    # Preserve shell quoting within each flag group before folding the heredoc.
    groups = []
    for value in cli:
        if value.startswith("--"):
            groups.append([])
        groups[-1].append(value)
    container = [
        "set -euo pipefail ;",
        f"cd {shlex.quote(repo)} ;",
        "export " + " ".join(shlex.quote(f"{key}={value}") for key, value in environment.items()) + " ;",
        "python -c "
        + shlex.quote("import torch; assert torch.cuda.is_available(), 'Run:ai job has no usable CUDA device'")
        + " ;",
        "python -m unittest tests.test_encoder_runtime -v ;",
        "python -m training.main_conv_synthetic",
        *("    " + shlex.join(group) for group in groups),
    ]
    flags = [
        f"    {flag}" + (" " + shlex.quote(value) if value is not None else "") + " \\\n"
        for flag, value in launch._runai_submit_flags(runai)
    ]
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            f"# Auto-generated encoder ablation; recipe: {recipe_label}",
            f"# Re-generate with: {regeneration}",
            f"# Model: {options['model_id']}; data seed {options['data_seed']}; model seed {options['model_seed']}",
            "# Preview without submitting: bash this-script.runai.sh --dry-run",
            "set -euo pipefail",
            "",
            "# Fold to one line, matching the existing generated Run:ai scripts.",
            "TRAIN_CMD=$(tr '\\n' ' ' <<'TRAIN_EOF'",
            *container,
            "TRAIN_EOF",
            ")",
            "",
            'if [[ "$#" -gt 1 ]]; then echo "Usage: $0 [--dry-run]" >&2; exit 2; fi',
            'case "${1:-}" in',
            "    --dry-run) printf '%s\\n' \"$TRAIN_CMD\"; exit 0 ;;",
            '    "") ;;',
            '    *) echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;',
            "esac",
            "",
            f"runai training standard submit {shlex.quote(job_name)} \\",
            "".join(flags).rstrip("\n"),
            '    --command -- bash -c "${TRAIN_CMD}"',
            "",
        ]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/encoder_comparison.json")
    parser.add_argument("--variants", type=Path, default=ROOT / "experiments/encoder_ablations/variants.yaml")
    parser.add_argument("--cluster-config", type=Path, default=ROOT / "experiments/cluster/runai.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/generated")
    parser.add_argument("--repo-path", help="Override the mounted repository path inside the container")
    parser.add_argument(
        "--results-dir", default="results/encoder_ablations", help="Absolute or relative to mounted repo"
    )
    parser.add_argument("--job-prefix", default="encoder-ablation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--include-controls", action="store_true", help="Also generate native Conv and ResNet controls")
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    regeneration = shlex.join(["python", "scripts/generate_encoder_ablation_runai.py", *arguments])
    config = comparison.read_json(args.config)
    comparison.validate_config(config)
    if config["shared"].get("no_cuda", False):
        parser.error("Run:ai GPU scripts require no_cuda=false")
    if len(set(args.seeds)) != len(args.seeds) or any(seed not in config["seeds"] for seed in args.seeds):
        parser.error("Provide unique seeds from the comparison recipe")
    variants = launch.load_yaml(args.variants)
    if args.include_controls:
        variants = {
            **{arch: {"reference": arch, "overrides": {}} for arch in comparison.ARCHITECTURES},
            **variants,
        }
    runai = launch.load_yaml(args.cluster_config)["_runai"].copy()
    if args.repo_path:
        runai["repo_path"] = args.repo_path
    if not PurePosixPath(runai["repo_path"]).is_absolute():
        parser.error("The container repository path must be absolute")
    results = PurePosixPath(args.results_dir)
    if not results.is_absolute():
        results = PurePosixPath(runai["repo_path"]) / results
    # Validate everything before writing any scripts.
    scripts = []
    for seed in args.seeds:
        for name, variant in variants.items():
            if not name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in name):
                parser.error("Variant names must contain only lowercase letters, digits and underscores")
            job = f"{args.job_prefix}-{name.replace('_', '-')}-s{seed}"
            if len(job) > 63 or job[0] == "-" or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in job):
                parser.error("Run:ai job names must be lowercase DNS labels of at most 63 characters")
            options = training_options(config, variant, name, seed, results)
            script = render_script(config, options, runai, job, args.config.name, regeneration)
            scripts.append((f"encoder_{name}_s{seed}.runai.sh", script))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in scripts:
        path = args.output_dir / filename
        path.write_text(content)
        path.chmod(0o755)
        print(path)
    print(f"Generated {len(scripts)} scripts. No jobs submitted.")


if __name__ == "__main__":
    main()
