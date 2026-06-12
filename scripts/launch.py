#!/usr/bin/env python3
"""Launch an experiment from a YAML config on RunAI or SLURM.

Usage:
    # Dry run — print resolved config + command, don't submit:
    python scripts/launch.py experiments/ablation_baseline.yaml --cluster runai --dry-run

    # Submit to RunAI (needs Python on cluster):
    python scripts/launch.py experiments/ablation_baseline.yaml --cluster runai

    # Generate bash scripts for ALL experiments (no Python needed on cluster):
    python scripts/launch.py --generate --cluster runai
    # Then on cluster:  bash experiments/generated/ablation_baseline.runai.sh

    # Override any parameter from CLI:
    python scripts/launch.py experiments/ablation_baseline.yaml --cluster runai --set lr=5e-4 train_steps=50000

    # Re-launch from a previous run's settings.json:
    python scripts/launch.py --from-config results/my-run/settings.json --cluster slurm --dry-run
    python scripts/launch.py --from-config results/my-run/settings.json --cluster slurm --set train_steps=100000

    # Save a settings.json as an experiment YAML (for version control):
    python scripts/launch.py --from-config results/my-run/settings.json --save-yaml experiments/rerun.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS_PATH = REPO_ROOT / "experiments" / "defaults.yaml"
CLUSTER_DIR = REPO_ROOT / "experiments" / "cluster"
GENERATED_DIR = REPO_ROOT / "experiments" / "generated"

# Boolean flags in argparse that are store_true (no value, just the flag name).
# Maintained here so the launcher knows to emit `--flag` instead of `--flag True`.
_STORE_TRUE_FLAGS = {
    "evaluate",
    "no_cuda",
    "use_amp",
    "save_all_checkpoints",
    "resume_training",
    "use_gan",
    "separate_encoders",
    "quantize_style",
    "cross_view_negs_only",
    "patch_contrastive",
    "gradient_checkpointing",
    "compile_model",
    "channels_last",
    "cache_dataset",
    "inject_style_to_decoder",
    "use_moco",
    "eval_dci",
    "eval_style",
    "grid_search_eval",
    "use_content_projection",
    "narrow_encoder_input",
    "top_level_recon_only",
    "use_wandb",
    "no_resumable_sampler",
    "shared_brain_mask",
    "deterministic",
    "asymmetric_aug",
    "pass_full_to_next_level",
    "select_by_gated_score",
    "detach_style_injection",
    "no_final_recon_norm",
    "synthetic_hierarchical_content",
    "synthetic_causal",
}

# Keys added at runtime by update_args() or the training loop — not part of the
# experiment config and must be stripped when reconstructing from settings.json.
_RUNTIME_KEYS = {
    "DATASETCLASS",
    "modalities",
    "n_views",
    "subsets",
    "content_indices",
    "style_indices",
    "save_dir",
}


def _get_valid_config_keys() -> set:
    """Return the set of argparse dest names accepted by main_multimodal.

    Parsed once from ``utils/config.py`` via regex so we don't need to import
    torch just to introspect the parser.  Also includes launcher-only keys
    (``tag``, ``model_dir``) and underscore-prefixed internal keys.
    """
    import re

    config_py = REPO_ROOT / "utils" / "config.py"
    source = config_py.read_text()
    flags = set(re.findall(r'"--([a-z][a-z0-9-]*)"', source))
    dests = {f.replace("-", "_") for f in flags}
    # Launcher-only keys that config_to_cli_args handles specially.
    dests.add("tag")
    return dests


def load_from_config(path: Path, strip_cluster: bool = False) -> dict:
    """Read a settings.json or config YAML from a previous run and return a clean config.

    Handles both:
    - ``settings.json`` saved by the training loop (``args.__dict__`` as JSON)
    - ``config_<ts>.yaml`` snapshots saved by launch.py (includes ``_provenance``)

    Runtime-derived keys (added by ``update_args`` / the training loop) are stripped
    so the result can be fed back into ``config_to_cli_args`` or saved as an
    experiment YAML.  Keys that are not recognised by the current argparse
    definition (e.g. renamed or removed flags from older code) are dropped with
    a warning so the re-launched job doesn't crash on ``unrecognized arguments``.

    Args:
        path: Path to settings.json or config_*.yaml.
        strip_cluster: If True, also remove cluster-specific resource keys
            (``_slurm``, ``_runai``) so the output is a pure training config.

    Returns:
        Flat config dict ready for ``config_to_cli_args``.
    """
    path = Path(path)
    if not path.exists():
        print(f"Error: config file not found: {path}", file=sys.stderr)
        sys.exit(1)

    if path.suffix == ".json":
        with open(path) as f:
            config = json.load(f)
    else:
        config = load_yaml(path)

    # Strip runtime keys that aren't part of the experiment config.
    for key in _RUNTIME_KEYS:
        config.pop(key, None)

    # Strip provenance metadata (from config YAML snapshots).
    config.pop("_provenance", None)

    if strip_cluster:
        config.pop("_slurm", None)
        config.pop("_runai", None)

    # settings.json stores the argparse key "model_id"; the launcher uses "tag".
    # Normalise to "tag" so config_to_cli_args emits --model-id correctly.
    if "model_id" in config and "tag" not in config:
        config["tag"] = config.pop("model_id")
    elif "model_id" in config and "tag" in config:
        config.pop("model_id")

    # patch_grid_per_level is stored as a list of [D,H,W] lists after
    # update_args converts it; flatten back to the CLI-style flat int list.
    pgpl = config.get("patch_grid_per_level")
    if pgpl is not None and len(pgpl) > 0 and isinstance(pgpl[0], (list, tuple)):
        config["patch_grid_per_level"] = [v for triple in pgpl for v in triple]

    # Drop keys that the current argparse doesn't recognise (renamed/removed
    # flags from older runs) so the training script doesn't crash on
    # "unrecognized arguments".
    valid_keys = _get_valid_config_keys()
    stale = [k for k in config if not k.startswith("_") and k not in valid_keys]
    if stale:
        print(f"Warning: dropping unrecognised keys from {path.name}: {stale}", file=sys.stderr)
        for k in stale:
            del config[k]

    return config


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        return result.returncode != 0
    except Exception:
        return False


def resolve_config(experiment_path: Path, cluster_name: str | None, cli_overrides: dict) -> dict:
    """Merge defaults <- base (if any) <- cluster <- experiment <- CLI overrides.

    If the experiment YAML contains a ``_base_`` key (e.g. ``_base_: synthetic_defaults``),
    that file is loaded on top of defaults.yaml before the experiment overrides.
    This lets synthetic experiments inherit from ``synthetic_defaults.yaml`` instead of
    repeating all the synthetic-specific settings.
    """
    config = load_yaml(DEFAULTS_PATH)

    # Check the experiment for a _base_ reference before merging cluster/experiment.
    experiment_cfg = load_yaml(experiment_path)
    base_name = experiment_cfg.pop("_base_", None)
    base_cfg = {}
    if base_name is not None:
        base_path = REPO_ROOT / "experiments" / f"{base_name}.yaml"
        if not base_path.exists():
            print(f"Error: _base_ config not found: {base_path}", file=sys.stderr)
            sys.exit(1)
        base_cfg = load_yaml(base_path)
        base_cfg.pop("_base_", None)  # Don't chain further.
        config.update(base_cfg)

    if cluster_name and cluster_name != "local":
        cluster_path = CLUSTER_DIR / f"{cluster_name}.yaml"
        if not cluster_path.exists():
            print(f"Error: cluster config not found: {cluster_path}", file=sys.stderr)
            sys.exit(1)
        cluster_cfg = load_yaml(cluster_path)
        config.update(cluster_cfg)

    # Re-apply null keys from the base so they aren't overridden by cluster paths.
    # E.g. synthetic_defaults.yaml sets labels_path: null to suppress the ADNI
    # labels path that the cluster config provides.
    for key, value in base_cfg.items():
        if value is None:
            config[key] = None

    config.update(experiment_cfg)

    # CLI --set overrides take highest priority.
    config.update(cli_overrides)

    return config


def config_to_cli_args(config: dict) -> list[str]:
    """Convert a flat config dict into argparse-style CLI arguments.

    Keys with a value of ``None`` are skipped (use this to unset a default).
    """
    args = []
    for key, value in sorted(config.items()):
        if key.startswith("_"):
            continue
        if value is None:
            continue
        if key == "tag":
            args.extend(["--model-id", str(value)])
            continue

        cli_flag = f"--{key.replace('_', '-')}"

        if key in _STORE_TRUE_FLAGS:
            if value:
                args.append(cli_flag)
            continue

        if isinstance(value, list):
            args.append(cli_flag)
            args.extend(str(v) for v in value)
        elif isinstance(value, bool):
            # Non-store-true booleans (shouldn't happen, but be safe).
            if value:
                args.append(cli_flag)
        else:
            args.append(cli_flag)
            args.append(str(value))

    return args


def _group_cli_args(cli_args: list[str]) -> list[str]:
    """Group CLI args into logical lines: --flag value [value ...] per line."""
    arg_lines = []
    i = 0
    while i < len(cli_args):
        if cli_args[i].startswith("--"):
            chunk = [cli_args[i]]
            i += 1
            while i < len(cli_args) and not cli_args[i].startswith("--"):
                chunk.append(cli_args[i])
                i += 1
            arg_lines.append(" ".join(chunk))
        else:
            arg_lines.append(cli_args[i])
            i += 1
    return arg_lines


def save_resolved_config(config: dict, experiment_path: Path) -> dict:
    """Add provenance metadata to the config for the snapshot saved alongside results."""
    snapshot = copy.deepcopy(config)
    snapshot["_provenance"] = {
        "experiment_file": str(experiment_path),
        "resolved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
    }
    return snapshot


def build_training_script(config: dict, tag: str, cluster_name: str, experiment_path: Path) -> str:
    """Build a self-contained bash script that runs the training command.

    For RunAI: wraps in `runai submit`.
    For SLURM: adds #SBATCH headers.
    For local: plain python command.
    """
    cli_args = config_to_cli_args(config)
    arg_lines = _group_cli_args(cli_args)
    sha = git_sha()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        exp_label = experiment_path.relative_to(REPO_ROOT)
    except ValueError:
        exp_label = experiment_path

    header = [
        "#!/usr/bin/env bash",
        f"# Auto-generated from: {exp_label}",
        f"# Generated at: {ts}",
        f"# Git SHA: {sha}",
        "# Re-generate with: python scripts/launch.py --generate --cluster " + cluster_name,
        "",
        "set -euo pipefail",
        "",
    ]

    if cluster_name == "runai":
        runai = config.get("_runai", {})
        repo_path = runai.get("repo_path", "/nfs/home/nglazman/crl-2/multiview-crl")

        # Build the training command as a single string for bash -c.
        train_parts = ["python -m training.main_multimodal \\"]
        for j, al in enumerate(arg_lines):
            suffix = " \\" if j < len(arg_lines) - 1 else ""
            train_parts.append(f"    {al}{suffix}")
        train_cmd = "\n".join(train_parts)

        lines = header + [
            "# --- Training command ---",
            f"TRAIN_CMD=$(cat <<'TRAIN_EOF'",
            f"cd {repo_path} && PYTHONPATH={repo_path} \\",
            train_cmd,
            "TRAIN_EOF",
            ")",
            "",
            "# --- RunAI submission ---",
            f"runai submit {tag} \\",
            f'    --project {runai.get("project", "nglazman")} \\',
            f'    --image {runai.get("image", "")} \\',
            f"    --run-as-user \\",
            f"    --large-shm \\",
            f'    --node-type {runai.get("node_type", "A100")} \\',
            f'    --gpu {runai.get("gpu", 1)} \\',
            f'    --cpu {runai.get("cpu", 16)} \\',
            f'    --cpu-limit {runai.get("cpu_limit", 32)} \\',
            f'    --memory {runai.get("memory", "64G")} \\',
            f'    --memory-limit {runai.get("memory_limit", "128G")} \\',
            f'    --volume {runai.get("volume", "/nfs:/nfs")} \\',
            f'    --command -- bash -c "${{TRAIN_CMD}}"',
        ]

    elif cluster_name == "slurm":
        slurm = config.get("_slurm", {})
        conda_env = slurm.get("conda_env", "multiview-env")
        req_txt = slurm.get("requirements_txt", "docker/requirements.txt")
        try:
            exp_rel = experiment_path.relative_to(REPO_ROOT)
        except ValueError:
            exp_rel = experiment_path

        lines = [
            "#!/bin/bash -l",
            f"# Auto-generated from: {exp_rel}",
            f"# Generated at: {ts}",
            f"# Git SHA: {sha}",
            "# Re-generate with: python scripts/launch.py --generate --cluster slurm",
            f"#SBATCH --job-name={tag}",
            "#SBATCH --output=/scratch/users/%u/%j.out",
            f"#SBATCH --error={tag}-%j.err",
            f"#SBATCH --partition={slurm.get('partition', 'gpu')}",
            f"#SBATCH --gres={slurm.get('gres', 'gpu:1')}",
            f"#SBATCH --nodes={slurm.get('nodes', 1)}",
            f"#SBATCH --mem={slurm.get('mem', '64G')}",
            f"#SBATCH --cpus-per-task={slurm.get('cpus_per_task', 8)}",
            f"#SBATCH --time={slurm.get('time', '24:00:00')}",
            *([f"#SBATCH --constraint={slurm['constraint']}"] if "constraint" in slurm else []),
            "",
            "# -- Software & Environment Setup --",
            "module load anaconda3/2022.10-gcc-13.2.0",
            "",
            f'CONDA_ENV_NAME="{conda_env}"',
            'PYTHON="${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python"',
            "",
            "export PYTHONNOUSERSITE=1",
            "",
            "# Automatically repair/build the environment if numpy or torch are missing",
            'if ! "$PYTHON" -c "import torch; import numpy" 2>/dev/null; then',
            "    echo \"Environment '${CONDA_ENV_NAME}' missing or broken -- rebuilding cleanly...\"",
            '    conda env remove -n "${CONDA_ENV_NAME}" --yes 2>/dev/null || true',
            '    conda create -n "${CONDA_ENV_NAME}" python=3.10 -y',
            "",
            '    "$PYTHON" -m pip install --upgrade pip',
            '    "$PYTHON" -m pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121',
            '    "$PYTHON" -m pip install numpy',
            '    "$PYTHON" -m pip install scikit-learn',
            '    "$PYTHON" -m pip install tensorboard pandas matplotlib',
            "fi",
            "",
            f'if [ -f "${{SLURM_SUBMIT_DIR}}/{req_txt}" ]; then',
            f'    "$PYTHON" -m pip install -r "${{SLURM_SUBMIT_DIR}}/{req_txt}" || echo "Requirements sync skipped a broken package."',
            "fi",
            'echo "Environment setup complete."',
            "",
            "# -- Working directory --",
            'cd "${SLURM_SUBMIT_DIR}"',
            'export PYTHONPATH="${SLURM_SUBMIT_DIR}"',
            'source "$(conda info --base)/etc/profile.d/conda.sh"',
            'conda activate "${CONDA_ENV_NAME}"',
            "",
            "# -- Training --",
            '"$PYTHON" -m training.main_multimodal \\',
        ]
        for j, al in enumerate(arg_lines):
            suffix = " \\" if j < len(arg_lines) - 1 else ""
            lines.append(f"    {al}{suffix}")

    else:
        # Local.
        lines = header + [
            "python -m training.main_multimodal \\",
        ]
        for j, al in enumerate(arg_lines):
            suffix = " \\" if j < len(arg_lines) - 1 else ""
            lines.append(f"    {al}{suffix}")

    return "\n".join(lines) + "\n"


def build_runai_command(config: dict, tag: str) -> list[str]:
    runai = config.get("_runai", {})
    repo_path = runai.get("repo_path", "/nfs/home/nglazman/crl-2/multiview-crl")
    cli_args = config_to_cli_args(config)
    train_cmd = f"cd {repo_path} && PYTHONPATH={repo_path} python -m training.main_multimodal {' '.join(cli_args)}"

    cmd = [
        "runai",
        "submit",
        tag,
        "--project",
        runai.get("project", "nglazman"),
        "--image",
        runai.get("image", ""),
        "--run-as-user",
        "--large-shm",
        "--node-type",
        runai.get("node_type", "A100"),
        "--gpu",
        str(runai.get("gpu", 1)),
        "--cpu",
        str(runai.get("cpu", 16)),
        "--cpu-limit",
        str(runai.get("cpu_limit", 32)),
        "--memory",
        runai.get("memory", "64G"),
        "--memory-limit",
        runai.get("memory_limit", "128G"),
        "--volume",
        runai.get("volume", "/nfs:/nfs"),
        "--command",
        "--",
        "bash",
        "-c",
        train_cmd,
    ]
    return cmd


def build_local_command(config: dict) -> str:
    cli_args = config_to_cli_args(config)
    return f"python -m training.main_multimodal {' '.join(cli_args)}"


def parse_cli_overrides(override_strs: list[str]) -> dict:
    """Parse 'key=value' pairs from --set arguments."""
    overrides = {}
    for s in override_strs:
        if "=" not in s:
            print(f"Error: --set value must be key=value, got: {s}", file=sys.stderr)
            sys.exit(1)
        key, val = s.split("=", 1)
        try:
            import ast

            parsed = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            parsed = val
        overrides[key] = parsed
    return overrides


def find_experiment_yamls() -> list[Path]:
    """Find all experiment YAML files (excluding cluster configs and defaults)."""
    exp_dir = REPO_ROOT / "experiments"
    skip = {"defaults.yaml", "cluster"}
    yamls = []
    for p in sorted(exp_dir.glob("*.yaml")):
        if p.name not in skip:
            yamls.append(p)
    return yamls


def generate_all(cluster_name: str):
    """Generate bash scripts for all experiment YAMLs."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    experiments = find_experiment_yamls()
    if not experiments:
        print("No experiment YAMLs found in experiments/", file=sys.stderr)
        sys.exit(1)

    generated = []
    for exp_path in experiments:
        config = resolve_config(exp_path, cluster_name, {})
        tag = config.get("tag", exp_path.stem)
        script = build_training_script(config, tag, cluster_name, exp_path)
        out_name = f"{exp_path.stem}.{cluster_name}.sh"
        out_path = GENERATED_DIR / out_name
        with open(out_path, "w") as f:
            f.write(script)
        os.chmod(out_path, 0o755)
        generated.append(out_name)
        print(f"  {out_path}")

    print(f"\nGenerated {len(generated)} scripts in experiments/generated/")
    print(f"On cluster: bash experiments/generated/<name>.{cluster_name}.sh")


def main():
    parser = argparse.ArgumentParser(
        description="Launch a multiview-CRL experiment from a YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "experiment",
        type=Path,
        nargs="?",
        default=None,
        help="Path to experiment YAML (omit when using --generate)",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="local",
        help="Cluster name (matches experiments/cluster/<name>.yaml), or 'local'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config and command without submitting",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate bash scripts for all experiment YAMLs (no Python needed on cluster)",
    )
    parser.add_argument(
        "--from-config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Re-launch from a previous run's settings.json or config YAML snapshot. "
        "Replaces the experiment YAML; --cluster and --set still apply on top.",
    )
    parser.add_argument(
        "--save-yaml",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save the resolved config as an experiment YAML (for version control). "
        "Strips runtime and cluster keys so the output is a clean experiment file. "
        "Use with --from-config to convert a settings.json into a reusable YAML.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="Override config values (e.g. --set lr=5e-4 train_steps=50000)",
    )
    args = parser.parse_args()

    # --generate mode: batch-generate scripts for all experiments.
    if args.generate:
        generate_all(args.cluster)
        return

    cli_overrides = parse_cli_overrides(args.overrides)

    # --from-config mode: reconstruct config from a previous run's settings.json
    # or config YAML, instead of resolving from an experiment YAML.
    if args.from_config is not None:
        config = load_from_config(args.from_config)

        # Merge cluster resource config (_slurm / _runai) if a cluster is specified,
        # so the submission wrapper knows how to submit the job.  Training-level
        # keys from the cluster YAML (dataroot, labels_path, …) are NOT merged —
        # the settings.json already has the fully resolved paths from the original run.
        # Use --set to override paths for a different cluster.
        if args.cluster and args.cluster != "local":
            cluster_path = CLUSTER_DIR / f"{args.cluster}.yaml"
            if cluster_path.exists():
                cluster_cfg = load_yaml(cluster_path)
                for key, value in cluster_cfg.items():
                    if key.startswith("_"):
                        config[key] = value

        # CLI --set overrides take highest priority.
        config.update(cli_overrides)

        # --save-yaml: write a clean experiment YAML and exit.
        if args.save_yaml is not None:
            save_config = load_from_config(args.from_config, strip_cluster=True)
            save_config.update(cli_overrides)
            args.save_yaml.parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_yaml, "w") as f:
                yaml.dump(save_config, f, default_flow_style=False, sort_keys=False)
            print(f"Experiment YAML saved to: {args.save_yaml}")
            return

        tag = config.get("tag", args.from_config.stem)
        experiment_source = args.from_config
    else:
        if args.experiment is None:
            parser.error("experiment path is required (or use --generate / --from-config)")

        if not args.experiment.exists():
            print(f"Error: experiment file not found: {args.experiment}", file=sys.stderr)
            sys.exit(1)

        config = resolve_config(args.experiment, args.cluster, cli_overrides)
        tag = config.get("tag", args.experiment.stem)
        experiment_source = args.experiment

    # -- Provenance snapshot --
    snapshot = save_resolved_config(config, experiment_source)

    if args.dry_run:
        print("=" * 60)
        print("RESOLVED CONFIG")
        print("=" * 60)
        print(yaml.dump(snapshot, default_flow_style=False, sort_keys=False))
        print("=" * 60)

    if args.cluster == "local":
        cmd_str = build_local_command(config)
        if args.dry_run:
            print(f"COMMAND:\n  {cmd_str}")
        else:
            print(f"Running locally: {cmd_str}")
            os.execvp(
                "python",
                ["python", "-m", "training.main_multimodal"] + config_to_cli_args(config),
            )

    elif args.cluster == "runai" or (CLUSTER_DIR / f"{args.cluster}.yaml").exists() and "_runai" in config:
        cmd = build_runai_command(config, tag)
        if args.dry_run:
            print(f"RUNAI COMMAND:\n  {' '.join(cmd)}")
        else:
            # Save snapshot to results dir before submitting.
            results_dir = REPO_ROOT / "results" / tag
            results_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = results_dir / f"config_{ts}.yaml"
            with open(snapshot_path, "w") as f:
                yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
            print(f"Config snapshot saved to: {snapshot_path}")
            print(f"Submitting to RunAI: {tag}")
            subprocess.run(cmd, check=True)

    elif args.cluster == "slurm" or "_slurm" in config:
        script = build_training_script(config, tag, "slurm", experiment_source)
        if args.dry_run:
            print(f"SLURM SCRIPT:\n{script}")
        else:
            results_dir = REPO_ROOT / "results" / tag
            results_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = results_dir / f"config_{ts}.yaml"
            with open(snapshot_path, "w") as f:
                yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
            script_path = results_dir / f"submit_{ts}.sh"
            with open(script_path, "w") as f:
                f.write(script)
            print(f"Config snapshot saved to: {snapshot_path}")
            print(f"Submitting to SLURM: {tag}")
            subprocess.run(["sbatch", str(script_path)], check=True)
    else:
        print(
            f"Error: unknown cluster '{args.cluster}'. Expected 'local', 'runai', or 'slurm'.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
