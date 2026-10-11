#!/usr/bin/env python3
"""Launch an experiment from a YAML config on RunAI or SLURM.

Usage:
    # Dry run — print resolved config + command, don't submit:
    python scripts/launch.py experiments/synthetic_causal.yaml --cluster runai --dry-run

    # Submit to RunAI (needs Python on cluster):
    python scripts/launch.py experiments/synthetic_causal.yaml --cluster runai

    # Generate bash scripts for ALL experiments (no Python needed on cluster):
    python scripts/launch.py --generate --cluster runai
    # Then on cluster:  bash experiments/generated/synthetic_causal.runai.sh

    # Override any parameter from CLI:
    python scripts/launch.py experiments/synthetic_causal.yaml --cluster runai --set lr=5e-4 train_steps=50000

    # Re-launch from a previous run's settings.json:
    python scripts/launch.py --from-config results/my-run/settings.json --cluster slurm --dry-run
    python scripts/launch.py --from-config results/my-run/settings.json --cluster slurm --set train_steps=100000

    # Re-launch on a different partition and drop the original node constraint:
    python scripts/launch.py --from-config results/my-run/settings.json --cluster slurm \
        --partition cpu --no-constraint --set resume_training=True

    # Re-launch on a specific partition and node constraint:
    python scripts/launch.py --from-config results/my-run/settings.json --cluster slurm \
        --partition gpu --constraint "h200|l40s" --set resume_training=True

    # Save a settings.json as an experiment YAML (for version control):
    python scripts/launch.py --from-config results/my-run/settings.json --save-yaml experiments/rerun.yaml

Encoder-only experiments (training/main_conv_synthetic.py) live in experiments/encoder/ and start
from experiments/encoder/defaults.yaml instead of experiments/defaults.yaml:

    python scripts/launch.py experiments/encoder/conv_mlp.yaml --cluster slurm --set seed=142 --dry-run
    python scripts/launch.py experiments/encoder/adni.yaml --cluster local --set device=mps dataroot=...
    # One script for a Python-less submit host (written to experiments/generated/encoder/):
    python scripts/launch.py experiments/encoder/adni.yaml --cluster runai --generate --set mask_mode=learned
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DEFAULTS_PATH = EXPERIMENTS_DIR / "defaults.yaml"
CLUSTER_DIR = EXPERIMENTS_DIR / "cluster"
GENERATED_DIR = EXPERIMENTS_DIR / "generated"
# Track defaults and bases: never launched or generated on their own.
BASE_CONFIG_NAMES = {"defaults.yaml", "synthetic_defaults.yaml"}

VQ_ENTRYPOINT = "training.main_multimodal"
ENCODER_ENTRYPOINT = "training.main_conv_synthetic"
# Where encoder launches keep their resolved-config snapshot: never the run directory itself,
# which the trainer creates and, under require_new_run, refuses to find already present.
ENCODER_LAUNCH_DIR = REPO_ROOT / "results" / "launches"

# Boolean flags in argparse that are store_true (no value, just the flag name).
# Maintained here so the launcher knows to emit `--flag` instead of `--flag True`.
_STORE_TRUE_FLAGS = {
    "evaluate",
    "no_cuda",
    "use_amp",
    "save_all_checkpoints",
    "resume_training",
    "freeze_encoder",
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
    "contrastive_only",
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

    if is_encoder(config) or "encoder_architecture" in config:
        return _load_encoder_settings(config, path, strip_cluster)

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


def _load_encoder_settings(config: dict, path: Path, strip_cluster: bool) -> dict:
    """``load_from_config`` for an encoder-only run's settings.json or config snapshot.

    settings.json is the trainer's whole namespace, so every key is an option; keys a newer
    trainer no longer has are dropped with a warning, as for the VQ-VAE.
    """
    from training import encoder_args

    config = {key: value for key, value in config.items() if key != "_provenance"}
    if strip_cluster:
        for key in ("_slurm", "_runai"):
            config.pop(key, None)
    if "model_id" in config:
        config.setdefault("tag", config["model_id"])
        del config["model_id"]
    config["_entrypoint"] = ENCODER_ENTRYPOINT
    known = {action.dest for action in encoder_args.build_parser()._actions}
    stale = [key for key in config if not key.startswith("_") and key != "tag" and key not in known]
    if stale:
        print(f"Warning: dropping unrecognised keys from {path.name}: {stale}", file=sys.stderr)
        for key in stale:
            del config[key]
    return config


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def is_encoder(config: dict) -> bool:
    """Whether ``config`` runs the encoder-only trainer rather than the VQ-VAE one."""
    return config.get("_entrypoint", VQ_ENTRYPOINT) == ENCODER_ENTRYPOINT


def defaults_path_for(experiment_path: Path) -> Path:
    """The track defaults an experiment starts from: the defaults.yaml in its own directory.

    experiments/<name>.yaml get experiments/defaults.yaml (VQ-VAE) and experiments/encoder/<name>.yaml
    get experiments/encoder/defaults.yaml (encoder-only). A YAML elsewhere falls back to the VQ-VAE's.
    """
    local = Path(experiment_path).resolve().parent / "defaults.yaml"
    return local if local.exists() else DEFAULTS_PATH


def format_tag(config: dict) -> None:
    """Fill ``{key}`` references in the tag from the resolved config, e.g. ``conv_mlp_s{seed}``.

    So ``--set seed=142`` names the run after its seed without a second override.
    """
    tag = config.get("tag")
    if not isinstance(tag, str) or "{" not in tag:
        return
    try:
        config["tag"] = tag.format(**{key: value for key, value in config.items() if not key.startswith("_")})
    except (KeyError, IndexError, ValueError) as error:
        print(f"Error: tag {tag!r} refers to a setting the config does not have ({error})", file=sys.stderr)
        sys.exit(1)


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
    """Merge defaults <- base (if any) <- cluster <- experiment <- its _cluster block <- CLI overrides.

    The defaults are the track's: the defaults.yaml in the experiment's own directory
    (``defaults_path_for``). If the experiment YAML contains a ``_base_`` key (e.g.
    ``_base_: synthetic_defaults``), that file is loaded on top of the defaults before the
    experiment overrides. This lets synthetic experiments inherit from ``synthetic_defaults.yaml``
    instead of repeating all the synthetic-specific settings. An experiment's ``_cluster`` block
    (and its base's) maps a cluster name to settings that apply only there (e.g. a per-cluster
    labels file).
    """
    config = load_yaml(defaults_path_for(experiment_path))

    # Check the experiment for a _base_ reference before merging cluster/experiment.
    experiment_cfg = load_yaml(experiment_path)
    cluster_overrides = experiment_cfg.pop("_cluster", None) or {}
    base_name = experiment_cfg.pop("_base_", None)
    base_cfg = {}
    if base_name is not None:
        base_path = REPO_ROOT / "experiments" / f"{base_name}.yaml"
        if not base_path.exists():
            print(f"Error: _base_ config not found: {base_path}", file=sys.stderr)
            sys.exit(1)
        base_cfg = load_yaml(base_path)
        base_cfg.pop("_base_", None)  # Don't chain further.
        base_cluster = base_cfg.pop("_cluster", None) or {}
        cluster_overrides = {
            name: {**(base_cluster.get(name) or {}), **(cluster_overrides.get(name) or {})}
            for name in {*base_cluster, *cluster_overrides}
        }
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
    config.update(cluster_overrides.get(cluster_name or "local") or {})

    # CLI --set overrides take highest priority.
    config.update(cli_overrides)
    format_tag(config)

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


def _runai_host_path(runai: dict) -> tuple[str, str]:
    """Split the docker-style ``volume: "host:container"`` into RunAI's two keys.

    The v1 CLI took ``--volume /nfs:/nfs``; the v2 CLI wants
    ``--host-path path=/nfs,mount=/nfs``.  Passing the un-split ``/nfs:/nfs``
    to either key silently produces a job with nothing mounted, so the repo
    path is missing inside the container.
    """
    volume = str(runai.get("volume", "/nfs:/nfs"))
    host, _, container = volume.partition(":")
    return host, (container or host)


def _runai_container_command(config: dict, repo_path: str, mount_path: str) -> str:
    """Build the body of the container command, one statement per line.

    Lines are folded into a single line before submission (see
    ``build_training_script``), so every statement ends in an explicit ``;``
    and no backslash line-continuations are used.
    """
    arg_lines = _group_cli_args(config_to_cli_args(config))
    lines = [
        f"cd {repo_path} || {{ echo ERROR: {repo_path} is missing inside the container - check the "
        f"--host-path mount of {mount_path} >&2 ; exit 1 ; }} ;",
        f"export PYTHONPATH={repo_path} ;",
        "python -m training.main_multimodal",
    ]
    lines.extend(f"    {al}" for al in arg_lines)
    return "\n".join(lines)


WANDB_KEY_HINT = "export WANDB_API_KEY before submitting - get it from https://wandb.ai/authorize"


THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _thread_env(n_cpus) -> dict[str, str]:
    """Cap native thread pools at the job's CPU allocation.

    OpenMP and OpenBLAS size their pools from the *host* core count, not the
    job's cgroup quota.  On a shared A100 node that means the sklearn probes in
    the evaluation step spawn ~100 threads against a handful of guaranteed
    cores and spend their time in scheduler contention — and how bad it gets
    depends on which node the job landed on.  None of these are set by
    default, so pin them all to what the job actually asked for.
    """
    return {name: str(max(1, int(n_cpus))) for name in THREAD_ENV_VARS}


def _runai_env_flags(config: dict, runai: dict, literal: bool) -> list[tuple[str, str]]:
    """``--environment`` flags for the job: thread caps, the cluster's own map, W&B auth.

    The container has no ``~/.netrc``, so ``--use-wandb`` needs ``WANDB_API_KEY``
    passed in or ``wandb.init`` raises "No API key configured".  The key is read
    from the submitting shell, never baked into a file: ``literal=False`` emits
    the unexpanded ``${WANDB_API_KEY:?...}`` for the generated bash script,
    ``literal=True`` resolves it now for direct ``subprocess`` submission.
    """
    env = _thread_env(runai.get("cpu", 16))
    # The cluster YAML's own map wins, so a run can still override the caps.
    env.update({str(k): str(v) for k, v in (runai.get("environment") or {}).items()})

    if config.get("use_wandb"):
        if literal:
            key = os.environ.get("WANDB_API_KEY", "")
            if not key:
                print(f"Error: use_wandb is set but WANDB_API_KEY is unset. {WANDB_KEY_HINT}", file=sys.stderr)
                sys.exit(1)
            env["WANDB_API_KEY"] = key
        else:
            env["WANDB_API_KEY"] = f"${{WANDB_API_KEY:?{WANDB_KEY_HINT}}}"

    return [("--environment", f"{name}={value}") for name, value in env.items()]


def _runai_submit_flags(runai: dict) -> list[tuple[str, str | None]]:
    """RunAI v2 CLI (``runai training standard submit``) resource flags."""
    host_path, mount_path = _runai_host_path(runai)
    return [
        ("--project", runai.get("project", "nglazman")),
        ("--image", runai.get("image", "")),
        ("--run-as-user", None),
        ("--large-shm", None),
        ("--node-type", runai.get("node_type", "A100")),
        ("--gpu-devices-request", str(runai.get("gpu", 1))),
        ("--cpu-core-request", str(runai.get("cpu", 16))),
        ("--cpu-core-limit", str(runai.get("cpu_limit", 32))),
        ("--cpu-memory-request", str(runai.get("memory", "64G"))),
        ("--cpu-memory-limit", str(runai.get("memory_limit", "128G"))),
        # `readwrite` is opt-in: without it the v2 CLI mounts read-only and the
        # first os.makedirs(save_dir) dies with "Errno 30: Read-only file system".
        # The v1 `--volume` was read-write by default.
        ("--host-path", f"path={host_path},mount={mount_path},readwrite"),
    ]


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
        _, mount_path = _runai_host_path(runai)

        # `tr` folds the heredoc to a single line: backslash line-continuations do
        # not survive the trip through `runai ... --command -- bash -c "$TRAIN_CMD"`
        # (the newlines arrive, the trailing `\` do not), which leaves the container
        # running each `--flag value` line as its own command.
        flag_lines = [
            f"    {flag}{'' if value is None else ' ' + value} \\" for flag, value in _runai_submit_flags(runai)
        ]
        # Quoted: the values carry `${...}` expansions and spaces.
        flag_lines += [f'    {flag} "{value}" \\' for flag, value in _runai_env_flags(config, runai, literal=False)]
        lines = header + [
            "# --- Training command (folded to one line: see note in scripts/launch.py) ---",
            "TRAIN_CMD=$(tr '\\n' ' ' <<'TRAIN_EOF'",
            _runai_container_command(config, repo_path, mount_path),
            "TRAIN_EOF",
            ")",
            "",
            "# --- RunAI submission ---",
            f"runai training standard submit {tag} \\",
            *flag_lines,
            f'    --command -- bash -c "${{TRAIN_CMD}}"',
        ]

    elif "slurm" in cluster_name:
        if cluster_name == "slurm_bio":
            slurm = config.get("_slurm_bio", {})
        else:
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
            f"#SBATCH --partition={slurm.get('partition', 'biomed_a100_gpu')}",
            f"#SBATCH --gres={slurm.get('gres', 'gpu:1')}",
            f"#SBATCH --nodes={slurm.get('nodes', 1)}",
            f"#SBATCH --mem={slurm.get('mem', '64G')}",
            f"#SBATCH --cpus-per-task={slurm.get('cpus_per_task', 8)}",
            f"#SBATCH --time={slurm.get('time', '24:00:00')}",
            *([f"#SBATCH --constraint={slurm.get('constraint', 'a100_80g')}"] if slurm.get("constraint") else []),
            "",
            "# -- Software & Environment Setup --",
            "module load anaconda3/2022.10-gcc-13.2.0",
            "",
            f'CONDA_ENV_NAME="{conda_env}"',
            'PYTHON="${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python"',
            "",
            "export PYTHONNOUSERSITE=1",
            *[f"export {name}={value}" for name, value in _thread_env(slurm.get("cpus_per_task", 8)).items()],
            "",
            "# Automatically repair/build the environment if numpy or torch are missing",
            "if ! \"$PYTHON\" -c \"import importlib.util; raise SystemExit(0 if importlib.util.find_spec('torch') and importlib.util.find_spec('numpy') else 1)\" 2>/dev/null; then",
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
            "# -- GPU preflight --",
            'echo "Node: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"',
            'nvidia-smi || echo "WARNING: nvidia-smi unavailable on $(hostname)"',
            *(
                []
                if any("--no-cuda" in al for al in arg_lines)
                else [
                    # A silent CPU fallback burns the whole walltime doing nothing;
                    # fail in seconds instead so the node can be reported.
                    'if ! "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then',
                    '    echo "ERROR: GPU allocated but torch cannot use it on $(hostname). Aborting."',
                    '    "$PYTHON" -c "import torch; print(f\'torch {torch.__version__} cuda={torch.version.cuda}\')"',
                    "    exit 1",
                    "fi",
                ]
            ),
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


def build_runai_command(config: dict, tag: str, resolve_secrets: bool = True) -> list[str]:
    """Direct-submit form of the generated script — same v2 CLI, same container command.

    ``resolve_secrets=False`` leaves ``WANDB_API_KEY`` as its unexpanded
    placeholder so ``--dry-run`` can print the command without echoing the key.
    """
    runai = config.get("_runai", {})
    repo_path = runai.get("repo_path", "/nfs/home/nglazman/crl-2/multiview-crl")
    _, mount_path = _runai_host_path(runai)
    train_cmd = " ".join(_runai_container_command(config, repo_path, mount_path).split())

    cmd = ["runai", "training", "standard", "submit", tag]
    for flag, value in _runai_submit_flags(runai) + _runai_env_flags(config, runai, literal=resolve_secrets):
        cmd.append(flag)
        if value is not None:
            cmd.append(value)
    cmd += ["--command", "--", "bash", "-c", train_cmd]
    return cmd


def build_local_command(config: dict) -> str:
    cli_args = config_to_cli_args(config)
    return f"python -m training.main_multimodal {' '.join(cli_args)}"


# ---------------------------------------------------------------------------
# Encoder-only track (training/main_conv_synthetic.py)
# ---------------------------------------------------------------------------


def encoder_cli_args(config: dict) -> list[str]:
    """The encoder trainer's arguments for ``config``, refused here exactly as the trainer would.

    ``training.encoder_args`` is the trainer's own parser without torch, so a misspelt key, a
    wrong value or an invalid combination stops the launch before anything is written or
    submitted, rather than when the job starts on the cluster.
    """
    from training import encoder_args

    options = {key: value for key, value in sorted(config.items()) if not key.startswith("_") and key != "tag"}
    if "model_id" in options:
        print("Error: name encoder runs with tag, not model_id", file=sys.stderr)
        sys.exit(1)
    options["model_id"] = str(config["tag"])
    try:
        arguments = encoder_args.cli_arguments(options)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    messages = io.StringIO()
    try:
        with contextlib.redirect_stderr(messages):
            encoder_args.parse_args(arguments)
    except SystemExit:
        # argparse prints its whole usage first; the reason is the last line.
        reason = (messages.getvalue().strip().splitlines() or ["invalid configuration"])[-1].split("error: ", 1)[-1]
        print(f"Error: {config['tag']}: the trainer refuses this configuration: {reason}", file=sys.stderr)
        sys.exit(2)
    return arguments


def _encoder_arg_lines(arguments: list[str]) -> list[str]:
    """One shell-quoted ``--flag value...`` per line."""
    groups = []
    for value in arguments:
        if value.startswith("--"):
            groups.append([])
        groups[-1].append(value)
    return [shlex.join(group) for group in groups]


def _encoder_env(config: dict) -> dict[str, str]:
    """Environment of an encoder job: one native thread per process unless cpu_threads says otherwise.

    The main process feeds the GPU and DataLoader workers augment on the CPU, so a thread pool
    sized from the host's core count only adds contention; the synthetic recipe pins cpu_threads
    for its paired rendering.
    """
    threads = str(config.get("cpu_threads") or 1)
    env = {"PYTHONUNBUFFERED": "1", **{name: threads for name in THREAD_ENV_VARS}}
    if config.get("deterministic"):
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def runai_job_name(tag: str) -> str:
    """Run:ai job names allow lowercase letters, digits and hyphens: the tag with '_' -> '-'."""
    name = str(tag).replace("_", "-").lower()
    if not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", name) or len(name) > 63:
        print(f"Error: tag {tag!r} gives Run:ai job name {name!r}, which is not a valid name", file=sys.stderr)
        sys.exit(1)
    return name


def _encoder_container_command(config: dict, arguments: list[str]) -> str:
    """Body of an encoder Run:ai job, one statement per line; folded to one line before submission.

    The environment is exported inside the command, so it reaches the job whatever the Run:ai
    CLI version calls its environment flag, and a job without a usable GPU stops at once.
    """
    runai = config.get("_runai", {})
    repo = runai.get("repo_path", "/nfs/home/nglazman/crl-2/multiview-crl")
    _, mount_path = _runai_host_path(runai)
    env = {str(k): str(v) for k, v in (runai.get("environment") or {}).items()}
    env.update(PYTHONPATH=repo, **_encoder_env(config))
    return "\n".join(
        [
            f"cd {repo} || {{ echo ERROR: {repo} is missing inside the container - check the "
            f"--host-path mount of {mount_path} >&2 ; exit 1 ; }} ;",
            "export " + " ".join(f"{name}={shlex.quote(value)}" for name, value in env.items()) + " ;",
            "python -c 'import torch; assert torch.cuda.is_available(), \"Run:ai job has no usable CUDA device\"' ;",
            f"python -m {ENCODER_ENTRYPOINT}",
            *(f"    {line}" for line in _encoder_arg_lines(arguments)),
        ]
    )


def build_encoder_runai_command(config: dict) -> list[str]:
    """``runai training standard submit`` for an encoder run, as one argument list."""
    command = ["runai", "training", "standard", "submit", runai_job_name(config["tag"])]
    for flag, value in _runai_submit_flags(config.get("_runai", {})):
        command += [flag] if value is None else [flag, value]
    body = " ".join(_encoder_container_command(config, encoder_cli_args(config)).split())
    return command + ["--command", "--", "bash", "-c", body]


def build_encoder_script(config: dict, cluster_name: str, experiment_path: Path, regenerate: str) -> str:
    """A self-contained bash script for one encoder run on ``cluster_name``.

    Run:ai and SLURM scripts take ``--dry-run``, which prints the command without submitting.
    Unlike the VQ-VAE's SLURM script, the SLURM one never installs packages: concurrent jobs
    would race on one conda environment, so a missing environment stops the job instead.
    No timestamp or git SHA is written, so regenerating an unchanged experiment changes nothing.
    """
    arguments = encoder_cli_args(config)
    tag = str(config["tag"])
    try:
        source = experiment_path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        source = experiment_path
    header = [
        f"# Auto-generated from {source} (encoder-only: {ENCODER_ENTRYPOINT}); edit the YAML, not this file.",
        f"# Re-generate with: {regenerate}",
    ]
    usage = [
        'if [[ "$#" -gt 1 ]]; then echo "Usage: $0 [--dry-run]" >&2; exit 2; fi',
        'case "${1:-}" in',
    ]

    if cluster_name == "runai":
        runai = config.get("_runai", {})
        flags = [
            f"    {flag}{'' if value is None else ' ' + shlex.quote(value)} \\"
            for flag, value in _runai_submit_flags(runai)
        ]
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                *header,
                "# Preview without submitting: bash <this script> --dry-run",
                "set -euo pipefail",
                "",
                "# Folded to one line: backslash continuations do not survive the trip through runai ... bash -c.",
                "TRAIN_CMD=$(tr '\\n' ' ' <<'TRAIN_EOF'",
                _encoder_container_command(config, arguments),
                "TRAIN_EOF",
                ")",
                "",
                *usage,
                "    --dry-run) printf '%s\\n' \"$TRAIN_CMD\"; exit 0 ;;",
                '    "") ;;',
                '    *) echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;',
                "esac",
                "",
                f"runai training standard submit {runai_job_name(tag)} \\",
                *flags,
                '    --command -- bash -c "${TRAIN_CMD}"',
                "",
            ]
        )

    if "slurm" in cluster_name:
        slurm = config.get("_slurm", {})
        directives = {
            "job-name": tag,
            "output": "/scratch/users/%u/%x-%j.out",
            "error": "/scratch/users/%u/%x-%j.err",
            "partition": slurm.get("partition", "gpu"),
            "gres": slurm.get("gres", "gpu:1"),
            "nodes": slurm.get("nodes", 1),
            "ntasks": 1,
            "mem": slurm.get("mem", "64G"),
            "cpus-per-task": slurm.get("cpus_per_task", 8),
            "time": slurm.get("time", "48:00:00"),
        }
        if slurm.get("constraint"):
            directives["constraint"] = slurm["constraint"]
        conda_env = str(slurm.get("conda_env", "multiview-env"))
        module = str(slurm.get("module", "anaconda3/2022.10-gcc-13.2.0"))
        exports = " ".join(f"{name}={shlex.quote(value)}" for name, value in _encoder_env(config).items())
        check = "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda}'); "
        check += "assert torch.cuda.is_available(), 'Allocated job has no usable CUDA device'"
        return "\n".join(
            [
                "#!/bin/bash -l",
                *header,
                "# Submit from the repository root: sbatch <this script>. Preview: bash <this script> --dry-run",
                # Not read by a shell, so written unquoted (`a100|h200|l40s` is one constraint).
                *(f"#SBATCH --{key}={value}" for key, value in directives.items()),
                "",
                "set -euo pipefail",
                "",
                "# sbatch runs a spooled copy of this script, so the repository is the submit directory.",
                'REPO="${ENCODER_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"',
                f"CONDA_ENV_NAME={shlex.quote(conda_env)}",
                'PYTHON="${ENCODER_PYTHON:-${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python}"',
                "TRAIN_ARGS=(",
                *(f"    {line}" for line in _encoder_arg_lines(arguments)),
                ")",
                "",
                *usage,
                "    --dry-run)",
                f'        printf \'%q \' "$PYTHON" -m {ENCODER_ENTRYPOINT} "${{TRAIN_ARGS[@]}}"',
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
                f"if [[ ! -f {ENCODER_ENTRYPOINT.replace('.', '/')}.py ]]; then",
                '    echo "Repository missing at $REPO. Submit from its root or set ENCODER_REPO." >&2',
                "    exit 1",
                "fi",
                "",
                "# Reuse the prepared environment; concurrent jobs never install or remove packages.",
                f"module load {shlex.quote(module)}",
                'if [[ ! -x "$PYTHON" ]]; then',
                '    echo "Python not found at $PYTHON. Prepare $CONDA_ENV_NAME or set ENCODER_PYTHON before sbatch." >&2',
                "    exit 1",
                "fi",
                f'export PYTHONPATH="$REPO" PYTHONNOUSERSITE=1 {exports}',
                'echo "Node: $(hostname)  Job: $SLURM_JOB_ID  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"',
                f'"$PYTHON" -c {shlex.quote(check)}',
                f'exec "$PYTHON" -m {ENCODER_ENTRYPOINT} "${{TRAIN_ARGS[@]}}"',
                "",
            ]
        )

    # Local: the active environment's Python, from the repository root; extra arguments go to the trainer.
    exports = " ".join(f"{name}={shlex.quote(value)}" for name, value in _encoder_env(config).items())
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            *header,
            "# Run from the repository root (or set ENCODER_REPO); extra arguments are passed to the trainer.",
            "set -euo pipefail",
            "",
            'REPO="${ENCODER_REPO:-$PWD}"',
            f"if [[ ! -f \"$REPO/{ENCODER_ENTRYPOINT.replace('.', '/')}.py\" ]]; then",
            '    echo "Run from the repository root or set ENCODER_REPO." >&2',
            "    exit 1",
            "fi",
            'cd "$REPO"',
            f'export PYTHONPATH="$REPO" {exports}',
            f"exec python -m {ENCODER_ENTRYPOINT} \\",
            *(f"    {line} \\" for line in _encoder_arg_lines(arguments)),
            '    "$@"',
            "",
        ]
    )


def lpips_stand_in() -> Path | None:
    """A directory holding a stand-in ``lpips`` package, or None when lpips is installed.

    ``training.losses`` imports LPIPS at module scope for the reconstruction losses, which the
    encoder-only trainer never builds; the stand-in raises if anything ever does. The directory
    is fixed, so a run started with ``exec`` needs no cleanup.
    """
    if importlib.util.find_spec("lpips") is not None:
        return None
    directory = Path(tempfile.gettempdir()) / "multiview-crl-lpips-stand-in"
    (directory / "lpips").mkdir(parents=True, exist_ok=True)
    (directory / "lpips" / "__init__.py").write_text(
        "class LPIPS:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        raise ImportError('lpips is not installed; encoder-only training should never build LPIPS')\n"
    )
    return directory


def encoder_local_launch(config: dict) -> tuple[list[str], dict[str, str]]:
    """The command and environment of a local encoder run with this interpreter.

    MPS gets PyTorch's CPU fallback (read once, when torch is imported); on macOS the run is wrapped
    in ``caffeinate -i`` so the machine does not sleep under it; without lpips installed, the
    stand-in above is put on PYTHONPATH.
    """
    command = [sys.executable, "-m", ENCODER_ENTRYPOINT, *encoder_cli_args(config)]
    env = dict(os.environ, **_encoder_env(config))
    paths = [str(REPO_ROOT)]
    stand_in = lpips_stand_in()
    if stand_in is not None:
        paths.insert(0, str(stand_in))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [*paths, os.environ.get("PYTHONPATH")]))
    if config.get("device") == "mps":
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        command = ["caffeinate", "-i", *command]
    return command, env


def save_encoder_launch(config: dict, snapshot: dict, script: str | None = None) -> Path:
    """Write the resolved config (and the submitted script) under results/launches/<tag>/."""
    directory = ENCODER_LAUNCH_DIR / str(config["tag"])
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(directory / f"config_{stamp}.yaml", "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
    print(f"Config snapshot saved to: {directory / f'config_{stamp}.yaml'}")
    if script is None:
        return directory
    path = directory / f"submit_{stamp}.sh"
    path.write_text(script)
    return path


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


def find_experiment_yamls(directory: Path | None = None) -> list[Path]:
    """Experiment YAMLs in ``directory`` (default: experiments/ and experiments/encoder/), without bases."""
    directories = [EXPERIMENTS_DIR, EXPERIMENTS_DIR / "encoder"] if directory is None else [Path(directory)]
    return [p for d in directories for p in sorted(d.glob("*.yaml")) if p.name not in BASE_CONFIG_NAMES]


def generated_path(experiment_path: Path, name: str, cluster_name: str) -> Path:
    """experiments/<sub>/<x>.yaml -> experiments/generated/<sub>/<name>.<cluster>.sh."""
    try:
        sub = experiment_path.resolve().parent.relative_to(EXPERIMENTS_DIR)
    except ValueError:
        sub = Path()
    return GENERATED_DIR / sub / f"{name}.{cluster_name}.sh"


def write_generated(config: dict, experiment_path: Path, cluster_name: str, name: str, regenerate: str) -> Path:
    """Write one experiment's script for ``cluster_name`` and return its path."""
    if is_encoder(config):
        script = build_encoder_script(config, cluster_name, experiment_path, regenerate)
    else:
        script = build_training_script(config, config.get("tag", experiment_path.stem), cluster_name, experiment_path)
    out_path = generated_path(experiment_path, name, cluster_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script)
    os.chmod(out_path, 0o755)
    print(f"  {out_path}")
    return out_path


def generate_all(cluster_name: str, directory: Path | None = None):
    """Generate bash scripts for every experiment YAML (or every one in ``directory``)."""
    experiments = find_experiment_yamls(directory)
    if not experiments:
        print(f"No experiment YAMLs found in {directory or 'experiments/'}", file=sys.stderr)
        sys.exit(1)

    for exp_path in experiments:
        config = resolve_config(exp_path, cluster_name, {})
        rel = exp_path.resolve().relative_to(REPO_ROOT)
        regenerate = f"python scripts/launch.py {rel} --cluster {cluster_name} --generate"
        write_generated(config, exp_path, cluster_name, exp_path.stem, regenerate)

    where = generated_path(experiments[0], "<name>", cluster_name)
    if where.is_relative_to(REPO_ROOT):
        where = where.relative_to(REPO_ROOT)
    print(f"\nGenerated {len(experiments)} scripts in experiments/generated/")
    print(f"On cluster: bash {where}")


def launch_encoder(config: dict, cluster_name: str, experiment_source: Path, snapshot: dict, dry_run: bool):
    """Run, submit or preview an encoder-only experiment."""
    if cluster_name == "local":
        command, env = encoder_local_launch(config)
        if dry_run:
            print(f"COMMAND:\n  {shlex.join(command)}")
            return
        print(f"Running locally: {shlex.join(command)}", flush=True)
        os.chdir(REPO_ROOT)
        os.execvpe(command[0], command, env)
    elif cluster_name == "runai" or "_runai" in config:
        command = build_encoder_runai_command(config)
        if dry_run:
            print(f"RUNAI COMMAND:\n  {shlex.join(command)}")
            return
        save_encoder_launch(config, snapshot)
        print(f"Submitting to Run:ai: {command[4]}")
        subprocess.run(command, check=True)
    elif "slurm" in cluster_name or "_slurm" in config:
        script = build_encoder_script(config, cluster_name, experiment_source, shlex.join(["python", *sys.argv]))
        if dry_run:
            print(f"SLURM SCRIPT:\n{script}")
            return
        script_path = save_encoder_launch(config, snapshot, script)
        print(f"Submitting to SLURM: {config['tag']}")
        # From the repository root, which the script takes as its working directory.
        subprocess.run(["sbatch", str(script_path)], check=True, cwd=REPO_ROOT)
    else:
        print(
            f"Error: unknown cluster '{cluster_name}'. Expected 'local', 'runai', 'slurm' or 'slurm_bio'.",
            file=sys.stderr,
        )
        sys.exit(1)


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
        help="Path to experiment YAML. With --generate: omit it for every experiment, or give a directory "
        "(e.g. experiments/encoder) for every YAML in it",
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
        help="Write bash scripts into experiments/generated/ instead of launching, for hosts without Python "
        "(no Python needed on cluster): for every experiment, a directory's, or one YAML with --set applied",
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
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        metavar="NAME",
        help="Override the SLURM partition (sets _slurm.partition). Useful when "
        "re-launching a job on a different queue.",
    )
    parser.add_argument(
        "--no-constraint",
        action="store_true",
        help="Drop the SLURM --constraint header (clears _slurm.constraint). Useful "
        "when re-launching on a partition where the original node constraint no longer applies.",
    )
    parser.add_argument(
        "--constraint",
        type=str,
        default=None,
        metavar="EXPR",
        help="Override the SLURM node constraint (sets _slurm.constraint), e.g. "
        "'a100' or 'h200|l40s'. Takes precedence over --no-constraint.",
    )
    args = parser.parse_args()
    cli_overrides = parse_cli_overrides(args.overrides)

    # --generate mode: batch-generate scripts for all experiments, or for every YAML in a directory.
    if args.generate and (args.experiment is None or args.experiment.is_dir()):
        if cli_overrides:
            parser.error("--set applies to one experiment: pass its YAML with --generate")
        generate_all(args.cluster, args.experiment)
        return

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

    # -- SLURM resource overrides (apply to the nested _slurm block, which
    # --set cannot reach since it only updates top-level keys). --
    if args.partition is not None or args.no_constraint or args.constraint is not None:
        slurm_cfg = config.setdefault("_slurm", {})
        if args.partition is not None:
            slurm_cfg["partition"] = args.partition
        if args.no_constraint:
            slurm_cfg["constraint"] = None
        if args.constraint is not None:
            slurm_cfg["constraint"] = args.constraint

    # --generate with one experiment: write its script, named after the tag when --set changed it.
    if args.generate:
        if args.from_config is not None:
            parser.error("--generate takes an experiment YAML, not --from-config")
        name = config.get("tag", args.experiment.stem) if cli_overrides else args.experiment.stem
        write_generated(config, args.experiment, args.cluster, str(name), shlex.join(["python", *sys.argv]))
        return

    # -- Provenance snapshot --
    snapshot = save_resolved_config(config, experiment_source)

    if args.dry_run:
        print("=" * 60)
        print("RESOLVED CONFIG")
        print("=" * 60)
        print(yaml.dump(snapshot, default_flow_style=False, sort_keys=False))
        print("=" * 60)

    if is_encoder(config):
        launch_encoder(config, args.cluster, experiment_source, snapshot, args.dry_run)
    elif args.cluster == "local":
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
        cmd = build_runai_command(config, tag, resolve_secrets=not args.dry_run)
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
