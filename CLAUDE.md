# multiview-crl

Multiview contrastive representation learning on paired T1/T2 brain MRI (ADNI). Learns content (shared anatomy) vs style (modality contrast) representations. PhD research code.

**Full details:** `METHODOLOGY_REPORT.md` (63KB). Read it only when you need specifics — changelog at the top captures recent design decisions.

## Layout

- `training/main_multimodal.py` — primary training entrypoint (VQ-VAE-2, InfoNCE/MoCo/BT/VICReg, Gumbel content mask, optional style quantization). ~2300 lines.
- `training/main_numerical.py` — small numerical theory-validation experiments (separate, simpler).
- `training/losses.py` — contrastive + recon losses (InfoNCE, MoCo, Barlow Twins, VICReg, patch-InfoNCE, LPIPS-based `BaselineLoss`, `cross_reconstruction_loss` for the style-swapped decode).
- `models/vqvae.py` — hierarchical 3D VQ-VAE-2 (content/style split, Gumbel mask, style codebooks). Primary model.
- `models/encoders.py` — MLP helpers for numerical experiments.
- `models/discriminator.py` — optional 3D PatchGAN discriminator (behind `--use-gan`).
- `data/datasets.py` — `MyCustomDataset` (ADNI) + `SyntheticBrainDataset`, NIfTI loading, MONAI preprocessing, SHA-256 fingerprinted disk cache.
- `data/infinite_iterator.py` — wraps DataLoader for infinite iteration.
- `eval/evaluation.py` — `val_step`, `get_data`, `eval_step` (linear/kernel/MLP probes, R²/accuracy).
- `eval/cross_reconstruction.py`, `eval/dci.py` — disentanglement metrics.
- `eval/run_dci_compare.py` — full cross-model protocol (R²/MCC/DCI, nulls, `--floor`, CSV). The source of truth for the metric rules; other eval scripts import them from here rather than re-deriving.
- `eval/identifiability_report.py` — one-page readable R²/MCC/DCI report for a single run vs its untrained floor. Per-factor R² at each factor's assigned pooling, plus an R² ladder showing every factor at every rung (gap/stats/patch). Thin layer over the above; `--self-test` runs torch-free.
- `eval/export_vq_bundle.py` — VQ features → the shared bundle `.npz` (the format `dinov3_embed_synthetic` writes), so VQ and DINO can be scored by one function.
- `eval/compare_bundles.py` — scores N bundles through one protocol into one table; verifies row identity, matches probe width, keeps each floor with its own bundle. `--with-graph` adds PC causal discovery per representation scored against the true SCM adjacency, with a ground-truth ceiling row; `--graph-repeats` puts a paired resampling error bar on it.
- `eval/plot_compare_bundles.py` — PNG figures from `compare_bundles --out` JSON: per-factor recovery with each model's own floor, what training bought, PC recovery vs the true adjacency with resampling error bars, partial-vs-raw R². Never re-scores; each figure ships a `.csv` twin.
- `eval/plot_pairing.py` — the two-arm pairing figures from the same `compare.json`: a dumbbell per factor (both arms, each with its own floor tick) and the difference alone anchored at zero, hatched where the arms' seed spreads cross. Warns when `gap` and `delta_floor` disagree on the winner. Never re-scores; CSV twins; `--self-test`.
- `eval/bundle_identity.py` — factor/generator digests that prove two bundles describe the same evaluation rows.
- `eval/COMPARING_3DINO_VQVAE.md` — what the matched DINO/VQ comparison equalises and what it cannot. Read before quoting a cross-model number.
- `eval/plot_identifiability.py` — PNG figures from `identifiability_report --out` JSON (one or two models). Never re-scores; each figure ships a `.csv` twin.
- `eval/latent_causal_discovery.py` — PC on the content CHANNELS directly (`--reduce gap|pc1|std|max`), no supervised readout, so no label builds the graph. Labels only name nodes afterwards via one Hungarian match on `|corr|`, scored against a random-assignment null plus `--floor`/`--ceiling`. Torch-free `--self-test`.
- `eval/plot_causal_recovery.py` — PNG figures from `run_causal_recovery`'s JSON: recovered/missed/spurious edge map, alpha sweep, per-factor partial-vs-raw R², CPDAG orientation. Never re-runs PC; each figure ships a `.csv` twin.
- `eval/plot_causal_graph.py` — PNG of the ground-truth content SCM (node-link + adjacency matrix) straight from `build_content_scm`, no run or scoring involved. Each figure ships a `.csv` twin; `--self-test` runs torch-free.
- `training/finetune_dino.py` — fine-tune a DINO backbone on the paired synthetic views. `--objective {infonce,barlow,vicreg}` varies the loss; `--pairing {cross_modal,within_modality}` varies what the positive pair is (the real T1/FLAIR pair, or one modality augmented twice) — that is the axis that isolates the cross-modal pairing from training on the data at all. Cross-view retrieval diagnostics are logged for every arm and are never part of a negative-free loss.
- `eval/dinov3_embed_synthetic.py` — run the synthetic views through a pretrained DINOv3 (HF `transformers`), slice 3D→2D, save embeddings + GT latents + SCM adjacency to `.npz`. `--random-init` gives the untrained floor.
- `eval/dinov3_identifiability.py` — score those embeddings: per-factor R²/MCC vs permutation null, floor and voxel baselines, plus PC graph recovery via `run_causal_recovery.evaluate_arrays`. Torch-free; `--self-test`.
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
- `scripts/compare_dino_objectives.sh` — end-to-end DINO objective ablation: verifies each fine-tune run's recorded objective, extracts bundles + floors, scores the content block and the full embedding separately, plots both.
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
