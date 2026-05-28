# multiview-crl

Multiview contrastive representation learning on paired T1/T2 brain MRI (ADNI). Learns content (shared anatomy) vs style (modality contrast) representations. PhD research code.

**Full details:** `METHODOLOGY_REPORT.md` (63KB). Read it only when you need specifics — changelog at the top captures recent design decisions.

## Layout

- `training/main_multimodal.py` — primary training entrypoint (VQ-VAE-2, InfoNCE/MoCo/BT/VICReg, Gumbel content mask, optional style quantization). ~2300 lines.
- `training/main_numerical.py` — small numerical theory-validation experiments (separate, simpler).
- `training/losses.py` — contrastive + recon losses (InfoNCE, MoCo, Barlow Twins, VICReg, patch-InfoNCE, LPIPS-based `BaselineLoss`).
- `models/vqvae.py` — hierarchical 3D VQ-VAE-2 (content/style split, Gumbel mask, style codebooks). Primary model.
- `models/encoders.py` — MLP helpers for numerical experiments.
- `models/discriminator.py` — optional 3D PatchGAN discriminator (behind `--use-gan`).
- `data/datasets.py` — `MyCustomDataset` (ADNI) + `SyntheticBrainDataset`, NIfTI loading, MONAI preprocessing, SHA-256 fingerprinted disk cache.
- `data/infinite_iterator.py` — wraps DataLoader for infinite iteration.
- `eval/evaluation.py` — `val_step`, `get_data`, `eval_step` (linear/kernel/MLP probes, R²/accuracy).
- `eval/cross_reconstruction.py`, `eval/dci.py` — disentanglement metrics.
- `eval/view_latents.ipynb`, `eval/dino.ipynb` — analysis notebooks.
- `utils/config.py` — `parse_args`, `update_args`. CLI surface lives here. Datasets: ADNI, synthetic, custom only.
- `utils/checkpointing.py` — save/load/emergency checkpoints, auto-resume, architecture compat check.
- `utils/visualisation.py` — decoded-image TB logging.
- `utils/logging_setup.py` — logging config.
- `utils/utils.py` — MONAI transforms (`CreateBrainMaskd`, `ApplyBrainMaskd`), `load_data`, `TBSummaryTypes`.
- `utils/helper.py` — `HelperModule`, `get_parameter_count` (used by vqvae.py).
- `experiments/defaults.yaml` — base config (all shared flags). Experiment YAMLs override only what differs.
- `experiments/cluster/{runai,slurm}.yaml` — cluster-specific paths and job resource configs.
- `experiments/*.yaml` — per-experiment configs (e.g. `ablation_baseline.yaml`).
- `scripts/launch.py` — reads experiment YAML, merges defaults+cluster+overrides, submits to RunAI/SLURM/local. Saves timestamped resolved config snapshot with git SHA to the run's output directory.
- `scripts/sweep_config.yaml` + `sweep_train.py` — W&B Bayesian sweep wrapper (handles bool flags + constraints).
- `scripts/launch_sweep.sh`, `sweep_runai.sh`, `analyze_sweep.py` — RunAI sweep launchers and analysis.
- `docker/` — CUDA 12.1 / Python 3.12 container, training scripts for RunAI cluster.
- `data/` (dir of code) vs `/data/natalia/ADNI_registered/` (actual dataset on cluster).

## Key facts

- 3D volumes, target shape `(91, 109, 91)` at 2mm isotropic.
- VQ-VAE-2 is the only encoder type. 3 levels, content channels via learned/fixed Gumbel mask at level 0 (finest), separate style codebook per level (optional).
- `VQVAE.forward()` returns an 8-tuple; callers assume that signature.
- Persistent `.pt` cache with SHA-256 fingerprint over `(spacing, crop_margin, paths)`; NFS-safe atomic writes.
- W&B + TensorBoard logging. Contrastive diagnostics (top-1 acc, pos/neg sim) logged per level.
- Pre-commit: black/isort (flake8 disabled). `pyproject.toml` has isort config only.

## Conventions

- Imports are first-party package style (`import training.losses`, `import models.vqvae`), not relative.
- Arg parsing is centralized in `utils/config.py`; add new flags there.
- When touching `VQVAE.forward` tuple, update: training loop, `visualisation.py`, eval notebook.
- Don't add docstrings/comments to code you aren't changing.

## Commands

- Launch experiment: `python scripts/launch.py experiments/<name>.yaml --cluster runai` (or `--cluster slurm`, `--cluster local`).
- Dry run (show resolved config + command): `python scripts/launch.py experiments/<name>.yaml --cluster runai --dry-run`.
- Override at launch: `python scripts/launch.py experiments/<name>.yaml --cluster runai --set lr=5e-4 train_steps=50000`.
- Direct train: `python -m training.main_multimodal --dataroot ... --dataset-name ADNI_stripped ...`
- Sweep: `wandb sweep scripts/sweep_config.yaml` then `./scripts/launch_sweep.sh`.
- Docker: `./docker/run_docker.sh`.
