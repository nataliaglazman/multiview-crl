#!/usr/bin/env python3
"""Generate slurm_bio scripts for the matched encoder ablations; never submit jobs.

Uses the same training-option builder as the Run:ai generator and reads the
existing slurm_bio cluster YAML. Requires PyYAML, not PyTorch.
"""

import argparse
import shlex
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import generate_encoder_ablation_runai as ablations  # noqa: E402


def load_resources(path):
    config = ablations.launch.load_yaml(path)
    # The checked-in slurm_bio.yaml uses _slurm, not _slurm_bio.
    resources = config.get("_slurm_bio", config.get("_slurm"))
    if not isinstance(resources, dict) or not resources:
        raise ValueError("Cluster config must contain _slurm_bio or _slurm resources")
    return resources.copy()


def render_script(config, options, resources, job_name, regeneration, repo_path=None):
    directives = {
        "job-name": job_name,
        "output": "/scratch/users/%u/%x-%j.out",
        "error": "/scratch/users/%u/%x-%j.err",
        "partition": resources["partition"],
        "gres": resources["gres"],
        "nodes": resources.get("nodes", 1),
        "ntasks": 1,
        "mem": resources["mem"],
        "cpus-per-task": resources["cpus_per_task"],
        "time": resources["time"],
    }
    if resources.get("constraint"):
        directives["constraint"] = resources["constraint"]
    if resources.get("account"):
        directives["account"] = resources["account"]
    if resources.get("qos"):
        directives["qos"] = resources["qos"]
    conda_env = str(resources.get("conda_env", "multiview-env"))
    if not conda_env or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in conda_env
    ):
        raise ValueError("conda_env must be an environment name; use ENCODER_PYTHON for a custom prefix")
    module = str(resources.get("module", "anaconda3/2022.10-gcc-13.2.0"))
    cli = ablations.comparison.cli_arguments({key: value for key, value in options.items() if key != "out_dir"})
    strings = [*map(str, directives.values()), module, options["out_dir"], repo_path or "", *cli]
    if any("\n" in value or "\r" in value for value in strings):
        raise ValueError("Newlines are not supported in job resources, paths or training arguments")
    groups = []
    for value in cli:
        if value.startswith("--"):
            groups.append([])
        groups[-1].append(value)
    repo_default = (
        [f"DEFAULT_REPO={shlex.quote(repo_path)}", 'REPO="${ENCODER_REPO:-$DEFAULT_REPO}"']
        if repo_path
        else ['REPO="${ENCODER_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"']
    )
    return "\n".join(
        [
            "#!/bin/bash -l",
            "# Auto-generated matched encoder ablation for slurm_bio.",
            f"# Re-generate with: {regeneration}",
            "# Submit from repository root: sbatch experiments/generated/<name>.slurm_bio.sh",
            *(f"#SBATCH --{key}={shlex.quote(str(value))}" for key, value in directives.items()),
            "",
            "set -euo pipefail",
            "",
            *repo_default,
            f"CONDA_ENV_NAME={shlex.quote(conda_env)}",
            'PYTHON="${ENCODER_PYTHON:-${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python}"',
            f"OUT_DIR={shlex.quote(options['out_dir'])}",
            'if [[ "$OUT_DIR" != /* ]]; then OUT_DIR="$REPO/$OUT_DIR"; fi',
            "TRAIN_ARGS=(",
            *("    " + shlex.join(group) for group in groups),
            '    --out-dir "$OUT_DIR"',
            ")",
            "",
            'if [[ "$#" -gt 1 ]]; then echo "Usage: $0 [--dry-run]" >&2; exit 2; fi',
            'case "${1:-}" in',
            "    --dry-run)",
            '        printf \'%q \' "$PYTHON" -m training.main_conv_synthetic "${TRAIN_ARGS[@]}"',
            "        printf '\\n'",
            "        exit 0 ;;",
            '    "") ;;',
            '    *) echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;',
            "esac",
            'if [[ -z "${SLURM_JOB_ID:-}" ]]; then',
            '    echo "Submit with sbatch from the repository root; use --dry-run to preview locally." >&2',
            "    exit 2",
            "fi",
            'cd "$REPO"',
            "if [[ ! -f training/main_conv_synthetic.py ]]; then",
            '    echo "Repository missing at $REPO. Submit from its root or set ENCODER_REPO." >&2',
            "    exit 1",
            "fi",
            "",
            "# Reuse the prepared environment; concurrent jobs never install/remove packages.",
            f"module load {shlex.quote(module)}",
            'if [[ ! -x "$PYTHON" ]]; then',
            '    echo "Python not found at $PYTHON. Prepare $CONDA_ENV_NAME or set ENCODER_PYTHON before sbatch." >&2',
            "    exit 1",
            "fi",
            'export PYTHONPATH="$REPO"',
            "export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8",
            *(f"export {name}={config['shared']['cpu_threads']}" for name in ablations.launch.THREAD_ENV_VARS),
            'echo "Node: $(hostname)  Job: $SLURM_JOB_ID  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"',
            '"$PYTHON" -c '
            + shlex.quote(
                "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda}'); "
                "assert torch.cuda.is_available(), 'Allocated job has no usable CUDA device'"
            ),
            '"$PYTHON" -m unittest tests.test_encoder_runtime -v',
            "",
            'exec "$PYTHON" -m training.main_conv_synthetic "${TRAIN_ARGS[@]}"',
            "",
        ]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/encoder_comparison.json")
    parser.add_argument("--variants", type=Path, default=ROOT / "experiments/encoder_ablations/variants.yaml")
    parser.add_argument("--cluster-config", type=Path, default=ROOT / "experiments/cluster/slurm_bio.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/generated")
    parser.add_argument("--results-dir", default="results/encoder_ablations_slurm_bio")
    parser.add_argument("--repo-path", help="Cluster repository path; default: directory where sbatch is invoked")
    parser.add_argument("--job-prefix", default="encoder-ablation-bio")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--include-controls", action="store_true")
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    regeneration = shlex.join(["python", "scripts/generate_encoder_ablation_slurm.py", *arguments])
    config = ablations.comparison.read_json(args.config)
    ablations.comparison.validate_config(config)
    if config["shared"].get("no_cuda", False):
        parser.error("SLURM GPU scripts require no_cuda=false")
    if len(set(args.seeds)) != len(args.seeds) or any(seed not in config["seeds"] for seed in args.seeds):
        parser.error("Provide unique seeds from the comparison recipe")
    if args.repo_path and not PurePosixPath(args.repo_path).is_absolute():
        parser.error("--repo-path must be absolute on the cluster")
    resources = load_resources(args.cluster_config)
    variants = ablations.launch.load_yaml(args.variants)
    if args.include_controls:
        variants = {
            **{arch: {"reference": arch, "overrides": {}} for arch in ablations.comparison.ARCHITECTURES},
            **variants,
        }
    scripts = []
    for seed in args.seeds:
        for name, variant in variants.items():
            if not name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in name):
                parser.error("Variant names must contain only lowercase letters, digits and underscores")
            job = f"{args.job_prefix}-{name.replace('_', '-')}-s{seed}"
            if not args.job_prefix or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in job):
                parser.error("Job names must use lowercase letters, digits and hyphens")
            options = ablations.training_options(config, variant, name, seed, args.results_dir)
            script = render_script(config, options, resources, job, regeneration, args.repo_path)
            scripts.append((f"encoder_{name}_s{seed}.slurm_bio.sh", script))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in scripts:
        path = args.output_dir / filename
        path.write_text(content)
        path.chmod(0o755)
        print(path)
    print(f"Generated {len(scripts)} slurm_bio scripts. No jobs submitted.")


if __name__ == "__main__":
    main()
