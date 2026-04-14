# multiview-crl

Multiview contrastive representation learning on paired T1/T2 brain MRI (ADNI). Learns content (shared anatomy) vs style (modality contrast) representations. PhD research code.

**Full details:** `METHODOLOGY_REPORT.md` (63KB). Read it only when you need specifics — changelog at the top captures recent design decisions.

## Layout

- `training/main_multimodal.py` — primary training entrypoint (VAE/VQ-VAE, InfoNCE/MoCo/BT/VICReg, Gumbel content mask, optional style quantization). Large file (~1900 lines).
- `training/main_numerical.py` — small numerical theory-validation experiments (separate, simpler).
- `training/losses.py` — contrastive + recon losses (InfoNCE, MoCo, Barlow Twins, VICReg, patch-InfoNCE, LPIPS-based `BaselineLoss`).
- `training/trainer.py` — older `VQVAETrainer` class (not used by `main_multimodal`).
- `models/vqvae.py` — hierarchical 3D VQ-VAE-2 (content/style split, Gumbel mask, style codebooks). Primary model.
- `models/vae.py` — simpler 3D VAE baseline.
- `models/encoders.py` — MLP helpers, `TextEncoder2D`.
- `models/pixelsnail.py` — autoregressive prior (unused in main path).
- `data/datasets.py` — `MultiviewDataset` + NIfTI loading, MONAI preprocessing, SHA-256 fingerprinted disk cache.
- `data/infinite_iterator.py` — wraps DataLoader for infinite iteration.
- `eval/evaluation.py` — `val_step`, `get_data`, `eval_step` (linear/kernel/MLP probes, R²/accuracy).
- `eval/cross_reconstruction.py`, `eval/dci.py` — disentanglement metrics.
- `eval/view_latents.ipynb`, `eval/dino.ipynb` — analysis notebooks.
- `utils/config.py` — `parse_args`, arg post-processing, GT index computation. CLI surface lives here.
- `utils/checkpointing.py` — save/load/emergency checkpoints, auto-resume, architecture compat check.
- `utils/visualisation.py` — decoded-image TB logging.
- `utils/logging_setup.py` — logging config.
- `utils/utils.py` — MONAI transforms (`CreateBrainMaskd`, `ApplyBrainMaskd`), `load_data`, `TBSummaryTypes`.
- `utils/{spaces,latent_spaces,invertible_network_utils,helper}.py` — numerical-experiment support.
- `scripts/sweep_config.yaml` + `sweep_train.py` — W&B Bayesian sweep wrapper (handles bool flags + constraints).
- `scripts/launch_sweep.sh`, `sweep_runai.sh`, `analyze_sweep.py` — SLURM/RunAI sweep launchers and analysis.
- `docker/` — CUDA 12.1 / Python 3.12 container, training scripts for RunAI cluster.
- `data/` (dir of code) vs `/data/natalia/ADNI_registered/` (actual dataset on cluster).

## Key facts

- 3D volumes, target shape `(91, 109, 91)` at 2mm isotropic.
- VQ-VAE default: 3 levels, content channels via learned/fixed Gumbel mask at level 0 (finest), separate style codebook per level (optional).
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

- Train: `python -m training.main_multimodal --dataroot ... --dataset-name ADNI_stripped --encoder-type vqvae ...` (see `METHODOLOGY_REPORT.md` for full arg list).
- Sweep: `wandb sweep scripts/sweep_config.yaml` then `./scripts/launch_sweep.sh`.
- Docker: `./docker/run_docker.sh`.
