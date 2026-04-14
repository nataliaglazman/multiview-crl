# Multiview Contrastive Representation Learning on ADNI Brain MRI

## Technical Report for Supervisor Review

**Project:** `multiview-crl`
**Dataset:** ADNI (Alzheimer's Disease Neuroimaging Initiative) - Registered T1 and T2 MRI scans
**Date:** February 2026
**Last Updated:** 14 April 2026

### Changelog

| Date | Changes |
|------|--------|
| 14 Apr 2026 | **W&B sweep infrastructure:** new Bayesian hyperparameter sweep via `scripts/sweep_config.yaml` + `scripts/sweep_train.py` (W&B agent wrapper that handles bool flags and coerces `vqvae_embed_dim` to equal `vqvae_hidden_channels`), launched with `scripts/launch_sweep.sh` (creates sweep, fans out Run:AI agents via `scripts/sweep_runai.sh`). `scripts/analyze_sweep.py` ranks finished runs by `separation_score`. Main training script now accepts `--use-wandb`, `--wandb-project`, `--wandb-entity` and mirrors all per-step scalars + summary metrics to W&B when enabled. Sweep objective is a new composite **`separation_score`** computed in `eval/cross_reconstruction.py` as the mean of (1 − |content→modality accuracy − 0.5|·2) and (1 − max(0, style→subject R²)); logged at the end of training and pushed to `wandb.summary`. **Contrastive loss objectives:** `--contrastive-loss-type {infonce, barlow_twins, vicreg}` (InfoNCE is default). Barlow Twins adds redundancy-reduction via the cross-correlation matrix (`--bt-lambda`, default 0.005); VICReg uses variance-invariance-covariance (`--vicreg-sim-coeff` 25, `--vicreg-std-coeff` 25, `--vicreg-cov-coeff` 1). Both are negative-free and work across patch and pooled modes; MoCo is auto-disabled when selected. **Patch-level (dense) InfoNCE:** `--patch-contrastive` with configurable `--patch-grid` (default `4 5 4`, ≈80 patches) aligns corresponding spatial patches across views instead of globally-pooled vectors. The encoder now adaptive-pools to the patch grid and the loss computes InfoNCE per patch position (or folds patches into the batch for BT/VICReg). **Multi-level content/style masking:** `--content-style-levels` (default `[0]`, can be `0 1 2`) selects which encoder levels receive the Gumbel content/style mask. `--content-ratios` lets each level use its own content ratio (e.g. `0.5 0.3 0.2` for a content-heavy pyramid). Each masked level produces its own style tensor for (a) contrastive loss, (b) decoder injection, and (c) optional style quantization. **Four mask modes** (`--mask-mode`): `onthefly` (default — logits = per-channel mean activation, shared across views, no learnable params; matches the original Yao et al. 2024 repo), `learned` (persistent `channel_logits` nn.Parameter; per-view variants when `--separate-encoders`), `fixed` (deterministic first-K-content split — no Gumbel noise, avoids MoCo queue staleness), and `learned_split` (per-channel sigmoid gates; the content/style size is no longer fixed — it emerges from training; incompatible with `--inject-style-to-decoder`). `--mask-warmup-steps` runs in-batch InfoNCE for the first N steps before starting the MoCo queue so the learned mask can stabilise; `--mask-lr-scale` slows mask evolution relative to the encoder. **Per-view encoders:** `--separate-encoders` gives each modality its own encoder stack (`encoders_v1`) while codebooks, decoders, and masks remain shared (consistent with view-specific identifiability theory, Yao et al. 2024). In this mode the forward pass splits the batch, encodes each half through its stack, and re-concatenates before the codebook loop. Learned mask mode extends automatically to per-view logits (`channel_logits_v1`). **Cross-view-negatives-only InfoNCE:** `--cross-view-negs-only` restricts the InfoNCE denominator to negatives drawn from the other view, forcing the model to align across modalities rather than within-view instance discrimination. Recommended with `--separate-encoders`. **Spatial FiLM style injection:** `--style-injection-mode {concat, film}`. `concat` is the legacy behaviour (style appended to the penultimate decoder feature map). `film` adds a `SpatialFiLM` module after each decoder stage — style is 1×1×1-convolved into per-location scale/shift maps that modulate every resolution, giving style access at every decoder layer. **Content propagation modes:** for levels with masking, four strategies for what the *next* encoder level sees (checked in priority order): `--pass-full-to-next-level` (unchanged full tensor — masking only affects the codebook/contrastive paths), `--narrow-encoder-input` (slice to content channels — cheapest; next encoder expects narrower input), `--use-content-projection` (slice + learned 1×1 conv back to `hidden_channels`), or fallback zero-masking. **SplitGroupNorm:** new module that group-normalises content and style channel groups independently so modality-specific style statistics can't bias content activations before the mask (stored in `content_norms`, applied between each encoder stage). **Codebook-collapse prevention:** explicit `--cb-ema-decay` (default 0.999), `--cb-reset-every` (default 100), `--cb-reset-threshold` (default 1.0) replace dead codebook entries (EMA cluster_size below threshold) with noisy copies of live ones every N forward passes per codebook. Codebook utilisation is now logged per level to TB/W&B (`codebook/active_L{i}`, `codebook/utilization_L{i}`, plus style-codebook variants when `--quantize-style`). **Top-level-recon-only ablation:** `--top-level-recon-only` zeros out encoder contributions at all but the coarsest level before the codebook, making reconstruction depend only on the top-level embedding while the contrastive loss still sees all levels. **Direct content-size override:** `--content-size N` bypasses the `--content-dim / --total-dim` ratio and sets content channels directly (propagates to all `content-style-levels`). **Custom spatial sizing:** `--spatial-size D H W` overrides the `image-spacing`/`crop-margin` derived shape. Matching RunAI defaults use `image-spacing 1.0` with `spatial-size 150 180 150`. **LR schedule + warmup:** `--warmup-steps` (default 1000), `--lr-schedule {cosine, constant}`, `--lr-min`. **Weight decay:** `--weight-decay` (default 0.01, applied to all params except biases, norms, ReZero alphas). **Early stopping:** `--early-stopping-patience` / `--early-stopping-min-delta` halt training if the monitored rolling avg (validation loss when `--val-every > 0`, otherwise training loss) fails to improve. **ADNI_stripped dataset variant:** new choice in `--dataset_name` for the skull-stripped ADNI preprocessing. **Changelog sweep defaults** now live in `scripts/sweep_config.yaml` — current Tier-1 search covers `scale_contrastive_loss`, `scale_recon_loss`, `vqvae_hidden_channels∈{32,48,64}` (locked equal to `vqvae_embed_dim`), `mask_mode∈{fixed,learned}`, `tau`, `content_ratios` (five preset pyramids), and Tier-2 `lr`, `bt_lambda`. Fixed sweep parameters include `patch_contrastive`, `content_style_levels=[0,1,2]`, `cross_view_negs_only`, `separate_encoders`, `pass_full_to_next_level`, `recon_loss_start_step=2000`, and `contrastive_level_weights=[1,1,1]`. **NaN-loss hardening:** losses.py now guards against zero-variance activations in Barlow Twins and division-by-zero in VICReg; contrastive path skips the loss instead of propagating NaN when a level has no valid positives. **Resume robustness:** `utils/checkpointing.py` now reloads optimizer/step state alongside queues, tolerates mismatched bool-flag shapes, and re-seeds dataloader iteration so resumed runs reproduce mid-epoch positions. |
| 31 Mar 2026 | **Style quantization (Option A):** added independent per-level style codebooks (`--quantize-style`) that vector-quantize style channels before decoder injection, giving style its own discrete bottleneck; configurable via `--style-embed-dim` and `--style-nb-entries` (default to main codebook settings). **Contrastive diagnostics:** both `moco_infonce_loss` and `infonce_base_loss` now compute and return top-1 accuracy, positive/negative cosine similarity distributions (mean, std) as detached side-outputs; logged to TensorBoard per level (`Contrastive/top1_acc_L{i}`, `Contrastive/pos_sim_mean_L{i}`, etc.) and printed to console. **8-tuple forward return:** `VQVAE.forward()` now returns `style_id_outputs` (dict: level → style codebook indices) as the 8th element; updated all callers (training loop, visualisation, eval notebook). |
| 24 Mar 2026 | **Content/style mask architecture overhaul:** moved Gumbel mask from embed_dim (32) back to hidden_channels (64) to prevent modality leakage through the `conv_in` projection; removed `content_proj` round-trip — level-0 codebook now receives only content channels directly. **Contrastive loss gradient fix:** removed `.detach()` from `channel_logits` in the contrastive path and switched from non-differentiable integer-index selection (`hz[..., indices]`) to differentiable soft masking (`hz * mask`) so Gumbel straight-through gradients flow from contrastive loss back to `channel_logits`. **Shared Gumbel mask:** forward pass now returns the soft content mask as a 7th output; the contrastive loss reuses this same mask instead of drawing an independent Gumbel sample, eliminating conflicting channel selections between reconstruction and contrastive objectives. **Persistent disk caching:** added `--cache-dir` for NFS-safe per-sample `.pt` caching with SHA-256 fingerprint invalidation, atomic writes, corruption detection (validates file size on startup, auto-deletes corrupt files at load time), and `.tmp` cleanup. **Periodic validation:** added `--val-every N` to run a short no-grad validation pass every N training steps, logging `Val/Total`, `Val/Contrastive`, `Val/Recon`, `Val/VQ` to TensorBoard. **Codebook indexing fix:** content-channels-aware codebook is now at index 0 (finest level, where the mask applies), not index `nb_levels-1`. Updated notebook (3 cells) and `visualisation.py` for 7-tuple return signature. |
| 20 Mar 2026 | Added best-model checkpointing (`vqvae_best.pt`) with rolling-average loss tracking; fixed `decode_codes` 2D→3D permute bug (`permute(0,3,1,2)` → `permute(0,4,1,2,3)` for 3D volumes); added codebook analysis to evaluation notebook (Sections 11–16): codebook usage histograms by diagnosis, mutual information & chi-squared discriminativeness, PCA/t-SNE of codebook usage, code replacement & reconstruction with CN vs AD comparison; added NIfTI export of reconstructed volumes with correct post-transform affine; fixed t-SNE content-only filtering at level 0; fixed `last_id_outputs` leaked loop variable bug |
| 25 Feb 2026 | Added style injection to decoder (`--inject-style-to-decoder`): style channels from encoder level-0 concatenated to penultimate decoder feature map before final conv; added `--content-dim` / `--total-dim` CLI args replacing hardcoded 256/256 content-style split; improved checkpointing with auto-resume and architecture compatibility check; added Docker/RunAI cluster scripts (`docker/`); fixed Gumbel mask bool-cast bug in `models/vqvae.py`; updated `eval/view_latents.ipynb` imports to match project package structure; disabled flake8 pre-commit hook |
| 23 Feb 2026 | Added MoCo contrastive training scheme (momentum encoder, per-level queues, `moco_infonce_loss`); refactored `main_multimodal.py` into six focused modules (`utils/config.py`, `utils/logging_setup.py`, `utils/checkpointing.py`, `utils/visualisation.py`, `eval/evaluation.py`, `training/main_multimodal.py`); updated file structure diagram; updated all CLI argument tables with MoCo args; resolved all pre-commit hook failures (flake8, isort/black conflict, check-docstring-first); added `pyproject.toml` with isort profile |
| 19 Feb 2026 | Removed unused `use_depthwise` parameter from residual blocks; added gradient checkpointing to `ResidualStack`; moved TensorBoard writer inside training block, now logs per-step scalars + learning rate; documented gradient accumulation, skip-recon-ratio, image spacing/cropping options; updated `BaselineLoss` documentation (L1 + FFT + LPIPS); added `view_latents.ipynb` to file structure; updated default `--vqvae-nb-entries` to 384; added checkpoint compatibility notes |
| 11 Feb 2026 | Initial report |

---

## 1. Overview

This project implements **Multiview Contrastive Representation Learning** for 3D brain MRI data, using paired T1-weighted and T2-weighted scans as two complementary views of the same subject. The goal is to learn disentangled representations that separate **content** (shared anatomical information) from **style** (modality-specific contrast differences).

### 1.1 Problem Formulation

Given paired observations $(x^{(1)}, x^{(2)})$ representing T1 and T2 MRI scans of the same subject, we aim to learn an encoder $f_\theta$ that maps each observation to a latent representation $z = f_\theta(x)$, where:

- **Content dimensions** $z_c$: Capture shared information (brain anatomy) that is consistent across both views
- **Style dimensions** $z_s$: Capture view-specific information (T1 vs T2 contrast characteristics)

### 1.2 Encoder Architecture

A hierarchical 3D **VQ-VAE-2** encoder (`models/vqvae.py`) is the sole architecture. Discrete codebook representations at three scales provide multi-resolution features for content/style disentanglement.

---

## 2. Data Pipeline

### 2.1 Input Data

| Property | Value |
|----------|-------|
| **Dataset** | ADNI (Alzheimer's Disease Neuroimaging Initiative) |
| **Data Location** | `/data/natalia/ADNI_registered/` |
| **Views** | 2 (T1-weighted and T2-weighted MRI) |
| **Original Resolution** | Variable (native scanner resolution) |
| **Target Resolution** | 2mm isotropic |
| **Target Spatial Size** | $(91, 109, 91)$ voxels |

### 2.2 Preprocessing Pipeline (MONAI Transforms)

The preprocessing is implemented in `utils/utils.py` using MONAI transforms:

```
1. LoadImaged           → Load NIfTI files
2. EnsureChannelFirstd  → Ensure channel-first format (C, H, W, D)
3. CreateBrainMaskd     → Create binary brain mask (threshold=50)
4. Spacingd             → Resample to 2mm isotropic (mode: bilinear)
5. ResizeWithPadOrCrop  → Crop/pad to (91, 109, 91)
6. NormalizeIntensityd  → Z-score normalization (per-volume)
7. ApplyBrainMaskd      → Zero out background using resampled mask
8. ToTensord            → Convert to PyTorch tensors
```

**Key Design Decisions:**
- **2mm resolution** reduces memory footprint while preserving anatomically relevant features
- **Brain masking** eliminates spurious non-zero background values caused by interpolation
- **Z-score normalization** standardizes intensity distributions across subjects

### 2.3 Persistent Disk Caching (`--cache-dir`)

Preprocessing brain MRI volumes (loading NIfTI, resampling, brain masking) is I/O-bound and takes ~30+ minutes for the full ADNI dataset. The persistent caching system pre-processes each volume once and saves it as a `.pt` file for instant loading on subsequent runs.

**Mechanism:**
1. A **SHA-256 fingerprint** is computed from `(spacing, crop_margin, sorted file paths)`. The cache directory is `<cache_dir>/preprocessed_<fingerprint>/`.
2. Each sample is saved as `<idx>.pt` containing plain tensors (`image_t1`, `image_t2`, `label`).
3. **Atomic writes**: saves to `.tmp` then `os.replace()` to prevent half-written files.
4. **Resumable**: on startup, only missing samples are processed (supports interrupted runs).
5. **NFS-safe**: `os.makedirs()` in worker processes handles NFS visibility delays.

**Corruption resilience:**
- Leftover `.tmp` files are cleaned up on startup
- Files smaller than 1 KB are treated as corrupted and re-generated
- At load time, if `torch.load` fails with a corruption error, the file is auto-deleted and a clear error message instructs to restart (the next run regenerates just that sample)

**Memory efficiency:**
- The persistent path does **not** retain loaded tensors in `self._cache` — each `__getitem__` loads from disk and releases. The OS page cache keeps hot files in memory transparently, avoiding Python-heap duplication that previously caused OOM after ~200 steps.

```bash
--cache-dir /data/natalia/cache  # Enable persistent disk caching
```

---

## 3. Model Architecture

## 3.1 VQ-VAE-2 Architecture (`models/vqvae.py`)

The VQ-VAE-2 is a **hierarchical Vector Quantized Variational Autoencoder** adapted for 3D brain MRI. It uses discrete codebook representations at multiple scales, enabling multi-resolution feature learning.

### 3.1.1 Architecture Overview

```
Input Image (91, 109, 91)
         │
         ▼
   ┌─────────────┐
   │  Encoder 0  │  (4× downscale)
   └─────────────┘
         │ (64 channels, ~23×27×23)
         ▼
   ┌─────────────┐
   │  Encoder 1  │  (2× downscale)
   └─────────────┘
         │ (64 channels, ~12×14×12)
         ▼
   ┌─────────────┐
   │  Encoder 2  │  (2× downscale)
   └─────────────┘
         │ (64 channels, ~6×7×6)
         ▼
   ┌─────────────┐
   │ Codebook 2  │  (Top level - most abstract)
   └─────────────┘
         │ (32 embed_dim)
         ▼
   ┌─────────────┐
   │  Decoder 2  │  + Upscaled codes
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │ Codebook 1  │  (Conditioned on level 2)
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │  Decoder 1  │  + Upscaled codes
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │ Codebook 0  │  (Bottom level - most detailed)
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │  Decoder 0  │  (Final reconstruction)
   └─────────────┘
         │
         ▼
Reconstructed Image (91, 109, 91)
```

### 3.1.2 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--vqvae-hidden-channels` | 64 | Hidden layer channels |
| `--vqvae-res-channels` | 32 | Residual block channels |
| `--vqvae-nb-levels` | 3 | Number of hierarchical levels |
| `--vqvae-embed-dim` | 32 | Codebook embedding dimension |
| `--vqvae-nb-entries` | 384 | Codebook size (number of codes) |
| `--vqvae-scaling-rates` | [2, 2, 2] | Downscale factor per level |
| `--vq-commitment-weight` | 0.25 | VQ commitment loss weight |
| `--gradient-checkpointing` | False | Trade compute for memory in residual blocks |
| `--skip-recon-ratio` | 0.0 | Fraction of steps to skip decoder (0–1) |
| `--gradient-accumulation-steps` | 1 | Accumulate gradients over N mini-batches |
| `--content-dim` | 128 | Content dimensions (ratio `content_dim/total_dim` determines `content_channels` on `hidden_channels`) |
| `--total-dim` | 512 | Total latent dims (`content_dim + style_dim`) |
| `--inject-style-to-decoder` | False | Feed style channels from encoder level-0 into the final decoder layer |
| `--quantize-style` | False | Vector-quantize style channels through independent per-level codebooks |
| `--style-embed-dim` | None | Style codebook embedding dim (defaults to `embed_dim`) |
| `--style-nb-entries` | None | Style codebook size (defaults to `nb_entries`) |

### 3.1.3 Encoder Details (Per Level)

Each encoder uses strided 3D convolutions for downsampling. Output sizes depend on input volume shape and `--vqvae-scaling-rates`.

| Level | Input Channels | Downscale | Output Channels | Output Size (approx) |
|-------|---------------|-----------|-----------------|---------------------|
| 0 | 1 (image) | 2× | 64 | (45, 54, 45) |
| 1 | 64 | 2× | 64 | (22, 27, 22) |
| 2 | 64 | 2× | 64 | (11, 13, 11) |

**Internal Encoder Structure:**
```
For encoder with 2× downscale:
      Conv3d(in, 32, k=4, s=2)  → BatchNorm → ReLU   (2× down)
      Conv3d(32, 64, k=3, s=1)  → BatchNorm          (refine)
      ResidualStack(64, 32, 2)                        (2 ReZero blocks)
```

**Gradient Checkpointing:** The `ResidualStack` supports optional gradient checkpointing (`use_checkpoint=True` by default). During training, the entire stack is wrapped with `torch.utils.checkpoint.checkpoint()`, trading recomputation for memory savings. This is controlled via `--gradient-checkpointing` at the command line, which passes through `VQVAE → Encoder/Decoder → ResidualStack`.

### 3.1.4 Vector Quantization Layer

The `CodeLayer` implements EMA (Exponential Moving Average) codebook updates. Each codebook has a `conv_in` projection (1×1×1 Conv3d) that maps from the input channels to `embed_dim`, followed by nearest-neighbour lookup in the codebook:

```python
# Discrete bottleneck
z_e = conv_in(encoder_output)           # (B, hidden_channels, ...) → (B, embed_dim, ...)
z_q = nearest_codebook_entry(z_e)       # (B, embed_dim, D, H, W)
commitment_loss = ||z_e - sg[z_q]||²    # Straight-through gradient
```

**Codebook input channels vary by level:**

| Level | Input to `conv_in` | In Channels | Description |
|-------|-------------------|-------------|-------------|
| 0 (finest) | `content_channels + embed_dim` | ~86 | Content-only encoder output + decoder conditioning |
| 1 (middle) | `hidden_channels + embed_dim` | ~96 | Full encoder output + decoder conditioning |
| 2 (coarsest) | `hidden_channels` | 64 | Full encoder output only (no decoder above) |

Level 0 receives only the content channels (selected by the Gumbel mask) rather than all `hidden_channels`. This ensures the finest-level codebook encodes only content information.

**Key Properties:**
- **Codebook Size:** 384 discrete codes (default)
- **Embedding Dimension:** 32
- **Update Method:** EMA (no codebook gradient, more stable than loss-based)
- **Decay Rate:** 0.99

### 3.1.5 Decoder Details

Each decoder receives concatenated codes from current and all higher levels:

| Level | Input Channels | Upscale | Output |
|-------|---------------|---------|--------|
| 2 (top) | 32 (1 code) | 2× | 32 (embed_dim) |
| 1 | 64 (2 codes) | 2× | 32 (embed_dim) |
| 0 (bottom) | 96 (3 codes) | 4× | 1 (image) |

### 3.1.6 Style Injection to Decoder (`--inject-style-to-decoder`)

By default, only the **content channels** (selected by the Gumbel mask) are passed forward to the level-0 codebook and decoder chain. The **style channels** (the complement within `hidden_channels`) are discarded after level-0, so the decoder has no signal about modality-specific features.

When `--inject-style-to-decoder` is enabled, the style channels are instead captured and fed back into the bottom decoder (`decoders[0]`) before the final output conv:

```
Encoder level-0 output  (B, hidden_channels, D, H, W)
         │
         ├─ content_idx channels ──► level-0 codebook ──► decoder chain
         │
         └─ style_idx channels ──────────────────────────────────────────┐
                                                                         ▼
                                                      decoders[0] penultimate feat
                                                      trilinear upsample to match
                                                      cat([feat, style], dim=1)
                                                                         ▼
                                                           final_conv ──► output
```

**Implementation details:**
- `Decoder` is built with `style_channels = hidden_channels − content_channels` when injection is active; a separate `self.final_conv` (replacing the original merged last layer) accepts the concatenated feature map.
- If the spatial size of the style map differs from the penultimate decoder feature map, trilinear interpolation is applied automatically.
- Has **no effect** unless `channel_logits` is active (i.e. `content_size > 0` and `style_size > 0`).

**Motivation:** Giving the decoder access to style (modality-specific) channels should improve reconstruction quality for T2 scans, whose contrast differs markedly from T1, without weakening the contrastive signal on content dimensions.

### 3.1.7 Style Quantization (`--quantize-style`)

By default, style channels are injected into the decoder as raw encoder activations (continuous). When `--quantize-style` is enabled, each masked level gets an **independent style codebook** that vector-quantizes the style channels before decoder injection. This is "Option A" — fully independent codebooks with no cross-level conditioning.

```
Encoder level-l output  (B, hidden_channels, D, H, W)
         │
         ├─ content_idx channels ──► content codebook ──► decoder body
         │
         └─ style_idx channels ──► style codebook ──► quantized ──► decoder final layer
```

**Architecture details:**
- One `CodeLayer` per masked level, with `in_channels = style_channels_per_level[lvl]`
- Embedding dimension and codebook size are configurable independently from the main codebooks via `--style-embed-dim` and `--style-nb-entries` (default to the main `--vqvae-embed-dim` and `--vqvae-nb-entries`)
- The decoder's `style_channels` input becomes `style_embed_dim` (the style codebook's embedding dimension) rather than the raw `hidden_channels - content_channels`
- Style commitment losses are appended to `diffs` and flow through `vq_commitment_weight` automatically

**Inference (`decode_codes`):** accepts a `style_codes` dict (level → LongTensor of style codebook indices) that are decoded through the style codebooks before being passed to the decoder. This enables style transfer by swapping style codes between subjects.

**Design rationale (Option A vs alternatives):**
- **Option A (implemented):** Fully independent style codebooks per level. No cross-level conditioning. Maximally disentangled — style codes cannot encode content through hierarchical dependencies. Simplest implementation.
- **Option B (not implemented):** Style codebooks with their own top-down chain. More expressive (coarse→fine style hierarchy) but requires style upscalers and risks content leakage through the style conditioning path.
- **Option C (not implemented):** Style conditioned on content. Better reconstruction but weaker disentanglement since style becomes content-dependent.

```bash
--inject-style-to-decoder \
--quantize-style \
--style-embed-dim 32 \    # defaults to --vqvae-embed-dim
--style-nb-entries 256    # defaults to --vqvae-nb-entries
```

### 3.1.8 Total Parameters

With default configuration:
- **Total VQ-VAE-2 Parameters:** ~2.9M

### 3.1.9 Reconstruction Loss (BaselineLoss)

The reconstruction loss (`BaselineLoss` in `training/losses.py`) combines three complementary terms:

$$\mathcal{L}_{recon} = \underbrace{\mathcal{L}_{pixel}}_{\lambda_p=1.0} + \underbrace{\mathcal{L}_{FFT}}_{\lambda_f=1.0} + \underbrace{\mathcal{L}_{LPIPS}}_{\lambda_{perc}=0.002}$$

| Component | Implementation | Purpose |
|-----------|---------------|----------|
| **Pixel loss** | L1 distance | Voxel-level accuracy |
| **Frequency loss** | MSE on FFT magnitudes (ortho-normalized) | Penalize missing high-frequency detail |
| **Perceptual loss** | LPIPS (SqueezeNet backbone, 512 random axial slices) | Structural/feature-level similarity |

VQ commitment losses per codebook level are added *on top* of this reconstruction loss and logged separately.

---

## 3.2 Hierarchical Contrastive Learning

A key innovation in this implementation is **hierarchical contrastive learning** applied across all encoder levels.

### 3.2.1 Motivation

Different encoder levels capture different types of information:
- **Level 0 (bottom):** Fine-grained local features (textures, edges)
- **Level 1 (middle):** Mid-level structures (gyri patterns, regional anatomy)
- **Level 2 (top):** Global/abstract features (overall brain shape, ventricle size)

By applying contrastive loss at each level, we enforce content/style disentanglement across all scales.

### 3.2.2 Content Selection Strategy

Content masking is controlled by two flags:
- `--content-style-levels` — which encoder levels receive a mask (default `[0]`, i.e. finest only). Set to `0 1 2` to mask at every level.
- `--mask-mode` — how the mask is computed. Four modes are implemented (see Section 6 for full details): `onthefly` (default), `learned`, `fixed`, `learned_split`.

At each masked level the mask is applied BEFORE the codebook projection so it operates on the raw `hidden_channels` encoder activations (typically 64) rather than the compressed embed_dim space, where the `conv_in` projection would mix all channels and leak modality information.

**Critical design: shared Gumbel mask.** When the mask is drawn (non-`fixed` modes), it is sampled ONCE per forward pass and used for both (a) selecting content channels for that level's codebook and (b) soft-masking pooled features for the contrastive loss. `VQVAE.forward()` returns the per-level soft masks as `soft_content_masks` (7th tuple element); the training loop reuses them verbatim, so both losses agree on which channels are content on every step. Previously, two independent Gumbel samples were drawn, causing the reconstruction and contrastive objectives to work on different channel subsets and preventing convergence.

```python
# In VQVAE.forward():
soft_mask = topk_gumbel_softmax(k=content_channels, logits=channel_logits)
content_enc_out = (enc_out * soft_mask)[:, content_idx, ...]  # → codebook

# Returned as 7th output (8th is style_id_outputs), reused for contrastive loss:
# In training loop (no second Gumbel sample):
soft_content_mask = fwd_soft_content_mask  # reuse from forward pass
level_loss = moco_loss(..., soft_content_mask=soft_content_mask)
```

**Differentiable masking for contrastive loss.** The contrastive loss uses `hz * soft_mask` (element-wise multiplication) instead of `hz[..., content_indices]` (integer indexing). This preserves the Gumbel straight-through gradient path from the contrastive loss back to `channel_logits`, so both reconstruction and contrastive objectives jointly guide which channels should be content vs style.

```python
# Level 0: Learnable content/style split via Gumbel-Softmax
content_ratio = content_dim / total_dim  # e.g. 0.75
n_channels = 64                          # hidden_channels
content_size = int(0.75 * 64) = 48       # content channels

# Higher levels: Use same ratio, batch-statistics mask
level_1_content = 48 dims
level_2_content = 48 dims
```

### 3.2.3 Training Loop

```python
# Forward pass returns shared mask (8-tuple)
recon, diffs, encoder_pools, est_indices, _, _, fwd_soft_content_mask, style_ids = \
    vqvae_model(images, return_recon=True, pool_only=True)

for level_idx, enc_pooled in enumerate(encoder_pools):
    # enc_pooled: (B, hidden_channels) — already spatially pooled
    hz = enc_pooled.reshape(n_views, batch_size, hidden_channels)

    if level_idx == 0 and channel_logits is not None:
        # Reuse the same mask from the forward pass
        soft_content_mask = fwd_soft_content_mask  # (1, 64)
    else:
        # Batch-statistics Gumbel mask for higher levels
        soft_content_mask = None  # falls back to integer indices

    # Contrastive loss: soft masking (differentiable) or index selection
    if use_moco:
        level_loss = moco_loss(hz, keys, queue, content_indices,
                               soft_content_mask=soft_content_mask)
    else:
        level_loss = infonce_loss(hz, content_indices,
                                  soft_content_mask=soft_content_mask)

    total_contrastive_loss += level_loss

total_loss = contrastive_loss + recon_loss * scale + vq_commitment_loss
```

### 3.2.4 Loss Components

$$\mathcal{L}_{total} = \sum_{l=0}^{2} \mathcal{L}_{contrastive}^{(l)} + \lambda_r \cdot \mathcal{L}_{recon} + \lambda_c \cdot \sum_{l=0}^{2} \mathcal{L}_{VQ}^{(l)}$$

Where:
- $\mathcal{L}_{contrastive}^{(l)}$: InfoNCE loss at encoder level $l$ (or MoCo InfoNCE if `--use-moco` is enabled)
- $\mathcal{L}_{recon}$: Reconstruction loss (BaselineLoss: L1 + FFT + LPIPS)
- $\mathcal{L}_{VQ}^{(l)}$: VQ commitment loss at level $l$
- $\lambda_r = 0.00001$ (scale_recon_loss)
- $\lambda_c = 0.25$ (vq_commitment_weight)

---

## 4. Loss Functions

### 4.1 Total Loss

$$\mathcal{L}_{total} = \sum_{l=0}^{2} \mathcal{L}_{contrastive}^{(l)} + \lambda_r \cdot \mathcal{L}_{recon} + \lambda_c \cdot \sum_{l=0}^{2} \mathcal{L}_{VQ}^{(l)}$$

Where $\lambda_r$ = `scale_recon_loss` (default: 0.00001), $\lambda_c$ = `vq_commitment_weight` (default: 0.25)

### 4.2 Contrastive Objective Selection (`--contrastive-loss-type`)

The contrastive loss family is chosen at runtime:

| Value | Function | Negatives? | Notes |
|-------|----------|-----------|-------|
| `infonce` (default) | `infonce_loss` / `patch_infonce_loss` / `moco_infonce_loss` | Yes (in-batch or MoCo queue) | Works with `--use-moco`, `--cross-view-negs-only`, `--patch-contrastive`. |
| `barlow_twins` | `barlow_twins_loss` | No | Push diagonal of cross-correlation matrix to 1 and off-diagonals to 0. Works at any batch size. `--bt-lambda` (default 0.005) weights the off-diagonal term. |
| `vicreg` | `vicreg_loss` | No | Variance–Invariance–Covariance — MSE across paired views + per-dim variance hinge (std ≥ 1) + off-diagonal covariance penalty. More stable than BT at very small batches. Coefficients `--vicreg-{sim,std,cov}-coeff`. |

MoCo is auto-disabled when a negative-free objective is selected. All three objectives accept the same `soft_content_mask` argument and fold patch dimensions into the batch when `--patch-contrastive` is set.

### 4.2.1 Standard Contrastive Loss (InfoNCE)

Implemented in `training/losses.py` as `infonce_loss`. Used by default when `--use-moco` is not set.

$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(z^{(1)}_c, z^{(2)}_c) / \tau)}{\sum_{k} \exp(\text{sim}(z^{(1)}_c, z^{(k)}_c) / \tau)}$$

**Key Properties:**
- **Similarity Metric:** Cosine similarity
- **Temperature ($\tau$):** 1.0 (default)
- **Content-Only Similarity:** Only `content_indices` dimensions are used for similarity computation
- **Positive Pairs:** Same subject, different views (T1, T2)
- **Negative Pairs:** Different subjects within the batch (or only the *other* view when `--cross-view-negs-only` is set — forces alignment across modalities rather than within-view instance discrimination)

**Limitation:** With small batches (batch size 2–4), the number of in-batch negatives is very small, which weakens the contrastive signal. MoCo addresses this (see Section 4.3). Barlow Twins and VICReg side-step the problem entirely by discarding negatives.

### 4.2.2 Patch-Level (Dense) InfoNCE (`--patch-contrastive`)

With `--patch-contrastive`, the encoder adaptively pools each level's spatial map to a grid of size `--patch-grid D H W` (default `4 5 4`, ≈80 patches). `patch_infonce_loss` then computes InfoNCE *per patch position* (positives are the same patch location across views; negatives are other subjects at the same patch) and averages over positions. This preserves spatial correspondence: patches at the same anatomical location should align, providing a richer signal than global pooling. For Barlow Twins / VICReg the patch axis is folded into the batch dimension so both objectives operate on the richer sample set.

### 4.2.3 Barlow Twins and VICReg

Both objectives operate on content-masked per-level features.

- **Barlow Twins** computes the cross-correlation matrix $C = Z_1^\top Z_2 / B$ between batch-normalised content features of the two views, then minimises $\sum_i (C_{ii}-1)^2 + \lambda \sum_{i\neq j} C_{ij}^2$. Encourages invariance on the diagonal and decorrelation off-diagonal. `top1_acc` is not meaningful and is reported as 0.
- **VICReg** combines (i) MSE between paired views (invariance), (ii) `relu(1 − std(z_i))` averaged per dimension (variance hinge to prevent representation collapse), and (iii) a normalised off-diagonal covariance penalty (decorrelation). Unlike Barlow Twins, the variance is enforced directly rather than inferred from the correlation matrix, so the loss behaves better at batch sizes 2–4.

### 4.3 MoCo Contrastive Loss (`--use-moco`)

Implemented in `training/losses.py` as `moco_infonce_loss` and `moco_loss`. Enabled via `--use-moco`. This extends the standard InfoNCE with a large external **queue of negatives** produced by a slowly-updated **momentum encoder**, decoupling the number of negatives from the batch size.

#### 4.3.1 Momentum Encoder (`MoCoEncoder` in `models/vqvae.py`)

A `MoCoEncoder` wraps the online VQ-VAE-2 and maintains:

- **Momentum encoder stack** — a frozen copy of the encoder stack (no decoder, no codebooks), updated via EMA:
  $$\theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q$$
  where $m$ is `--moco-momentum` (default 0.999).

- **Per-level FIFO queues** — one circular buffer of keys per encoder level, each of shape $(C, Q)$ where $Q$ is `--moco-queue-size` (default 4096).

The key design choices:
- The momentum encoder mirrors **only the encoder stack**, not the decoder or codebooks — avoiding the memory cost of duplicating the full VQ-VAE-2.
- Keys are encoded **without gradients** (`torch.no_grad()`) and stored as L2-normalised vectors.
- Queues are stored as non-parameter buffers so they are saved/restored in checkpoints automatically.

#### 4.3.2 MoCo Training Loop

```python
# 1. Encode queries with online encoder (forward pass, gradients flow)
online_outputs = vqvae_model(images)       # (B, C, D, H, W) per level

# 2. Encode keys with momentum encoder (no gradients)
key_outputs = vqvae_model.encode_keys(images)  # [(B, C), ...] per level

# 3. Per-level MoCo loss
for level_idx in range(nb_levels):
    hz_level = online_enc_features[level_idx]     # (n_views*B, C)
    k_level  = key_outputs[level_idx]             # (B, C)
    queue    = vqvae_model.queues[level_idx]      # (C, Q) snapshot

    level_loss = moco_loss(hz_level, k_level, queue,
                           content_indices, args.subsets)
    total_contrastive_loss += level_loss

# 4. Enqueue new keys (all levels in one call, after the loss loop)
vqvae_model.enqueue([k.detach() for k in key_outputs])

# 5. EMA update of momentum encoder happens inside vqvae_model.forward()
```

#### 4.3.3 MoCo InfoNCE Loss

For each view-pair $(q, k^+)$:

$$\mathcal{L}_{MoCo} = -\log \frac{\exp(q \cdot k^+ / \tau)}{\exp(q \cdot k^+ / \tau) + \sum_{k^- \in \mathcal{Q}} \exp(q \cdot k^- / \tau)}$$

where $\mathcal{Q}$ is the queue of past keys (negatives). The positive key $k^+$ is placed at index 0; all queue entries are treated as negatives. The loss is symmetric (computed for both view directions) and averaged over content dimensions.

#### 4.3.4 MoCo vs Standard InfoNCE

| Property | Standard InfoNCE | MoCo |
|----------|-----------------|------|
| **Negatives** | In-batch only (batch_size − 1) | Queue + in-batch (up to 4096) |
| **Memory overhead** | None | Queue buffers + momentum encoder |
| **Gradient through negatives** | Yes (costly) | No (queue is detached) |
| **Consistency of negatives** | Varies per batch | Maintained by EMA encoder |
| **Best for** | Large batches | Small batches (≤4 common in 3D MRI) |

#### 4.3.5 Queue Cold-Start

For the first `queue_size / batch_size` steps, the queue contains random L2-normalised noise rather than real encoder features. During this warm-up period the contrastive loss is meaningful but not yet fully representative. Mitigation: start with a smaller `--moco-queue-size` (e.g. 256–512) and increase as training stabilises.

### 4.4 Reconstruction Loss (BaselineLoss)

Implemented in `training/losses.py` as `BaselineLoss`:

$$\mathcal{L}_{recon} = \mathcal{L}_{L1}(\hat{x}, x) + \mathcal{L}_{FFT}(\hat{x}, x) + 0.002 \cdot \mathcal{L}_{LPIPS}(\hat{x}, x)$$

Where:
- $\hat{x} = g_\phi(z)$ is the decoded image
- $\mathcal{L}_{L1}$: Mean absolute pixel error
- $\mathcal{L}_{FFT}$: MSE on ortho-normalized FFT magnitudes
- $\mathcal{L}_{LPIPS}$: Perceptual loss using SqueezeNet on 512 random 2D slices

### 4.5 VQ Commitment Loss

Each codebook level has a commitment loss:

$$\mathcal{L}_{VQ}^{(l)} = ||z_e^{(l)} - \text{sg}[z_q^{(l)}]||^2$$

Where:
- $z_e^{(l)}$ is the encoder output at level $l$
- $z_q^{(l)}$ is the quantized codebook entry
- $\text{sg}[\cdot]$ is the stop-gradient operator

The codebook is updated via EMA (Exponential Moving Average) rather than gradient descent.

When `--quantize-style` is active, style codebook commitment losses are also included in $\sum \mathcal{L}_{VQ}$ (appended to `diffs` alongside the content codebook losses).

---

## 5. Training Configuration

### 5.1 Optimization

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW (fused when CUDA available) |
| **Learning Rate** | 1e-5 |
| **Gradient Clipping** | max_norm=2.0, L2 norm |
| **Mixed Precision (AMP)** | Enabled (fp16) via `--use-amp` |
| **Batch Size** | Configurable (default 2, effective = batch_size × n_views) |
| **Gradient Accumulation** | `--gradient-accumulation-steps N` (effective batch = batch_size × N) |
| **Skip Reconstruction** | `--skip-recon-ratio R` — fraction of steps that skip the decoder (saves memory) |

### 5.1.1 Image Preprocessing Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--image-spacing` | 2.0 | Isotropic voxel spacing in mm |
| `--crop-margin` | 0 | Voxels to crop from each edge |

These are passed to the dataset class and affect the spatial dimensions of the input volumes.

### 5.2 Training Loop

```
For each step:
    1. Load batch: {T1: (B, 1, 91, 109, 91), T2: (B, 1, 91, 109, 91)}
    2. Concatenate views: (2B, 1, 91, 109, 91)
    3. Encode: z = encoder(images) → per-level features
    4. [MoCo only] Encode keys with momentum encoder (no_grad)
    5. Decode: x_hat = decoder(z) → (2B, 1, 91, 109, 91)  [if not skipped]
    6. Compute contrastive loss (InfoNCE or MoCo) per encoder level
    7. [MoCo only] Enqueue new keys into per-level queues
    8. Compute reconstruction loss on (x_hat, images)
    9. Backpropagate total_loss with AMP scaling
   10. Clip gradients and update parameters
   11. [MoCo] EMA update of momentum encoder happens inside forward()
```

### 5.3 Logging and Checkpoints

| Event | Frequency | Destination |
|-------|-----------|-------------|
| Loss + top-1 acc printing (console) | Every step | stdout |
| TensorBoard scalars | Every step | `{save_dir}/tensorboard/` |
| CSV logging (smoothed avg) | Every `--log-steps` steps (default 100) | `{save_dir}/Training.csv` |
| Decoded image saving | Every 200 steps | `{save_dir}/decoded_images/` |
| Model checkpointing | Every `--checkpoint-steps` steps (default 1000) | `{save_dir}/vqvae_model.pt` |
| Best-model checkpoint | When rolling avg loss improves | `{save_dir}/vqvae_best.pt` |

**Best-Model Tracking:**

The checkpointing system tracks the best model via a rolling average of the total loss. At each checkpoint step, if `len(loss_values) == maxlen` (the rolling window is full), the current rolling average is compared against the historical best. If improved, the checkpoint is saved to `vqvae_best.pt`. When resuming training, the best loss is restored from the existing best checkpoint if present.

```python
# In main_multimodal.py:
rolling_loss = np.mean(loss_values) if len(loss_values) == loss_values.maxlen else None
if rolling_loss is not None and rolling_loss < best_total_loss:
    best_total_loss = rolling_loss
    save_checkpoint(..., best_loss=rolling_loss)  # → vqvae_best.pt
```

**TensorBoard Metrics (per step):**
- `Loss/Total` — total training loss
- `Loss/Contrastive` — InfoNCE or MoCo contrastive loss
- `Loss/Contrastive_L{i}` — per-level contrastive loss
- `Loss/Recon` — reconstruction loss (BaselineLoss)
- `Loss/VQ` — VQ commitment loss (includes style codebook commitment if `--quantize-style`)
- `LR` — current learning rate
- `Contrastive/top1_acc_L{i}` — fraction of times the positive is ranked #1 at level i
- `Contrastive/pos_sim_mean_L{i}` / `pos_sim_std_L{i}` — cosine similarity distribution for positive pairs
- `Contrastive/neg_sim_mean_L{i}` / `neg_sim_std_L{i}` — cosine similarity distribution for negatives

**Validation Metrics (every `--val-every` steps, if > 0):**
- `Val/Total` — validation total loss (no-grad, eval mode)
- `Val/Contrastive` — validation contrastive loss
- `Val/Recon` — validation reconstruction loss
- `Val/VQ` — validation VQ commitment loss

Validation runs a short pass (up to 20 batches) over the validation split with the model in `eval()` mode and no gradient computation. This provides an overfitting signal without significant training slowdown.

**Checkpoint Contents:**
- Model weights (`vqvae_model.pt`; best model in `vqvae_best.pt`)
- Optimizer state
- Current step and last loss values (total, contrastive, recon, VQ)
- If `--use-moco`: per-level queue tensors and queue pointers (explicit `moco_queues` / `moco_queue_ptrs` keys)
- Best-model checkpoint includes `best_loss` field for resume tracking

**Auto-Resume with Architecture Compatibility Check:**

`load_checkpoint` now automatically resumes from the checkpoint in `args.save_dir` if one exists, regardless of whether `--resume-training` was passed. Before loading, it runs `_state_dicts_compatible()` which verifies that every parameter name **and** tensor shape in the checkpoint matches the current model. If there is a mismatch (e.g. architecture was changed between runs), it logs a warning and starts fresh rather than crashing:

```
[CHECKPOINT] VQ-VAE checkpoint found but model architecture does not match
             (checkpoint: results/.../vqvae_model.pt). Starting fresh training.
```

This makes it safe to rerun a training command after changing architecture flags — the old checkpoint is preserved on disk but not loaded.

**Viewing TensorBoard from a headless cluster:**
```bash
# On the cluster:
tensorboard --logdir results/ADNI_registered/<model_id>/tensorboard --port 6006 --bind_all

# On your local machine:
ssh -N -L 6006:localhost:6006 <user>@<cluster-hostname>
# Then open http://localhost:6006
```

---

## 6. Content-Style Disentanglement

### 6.1 Theoretical Foundation

The method is based on the principle that:
- **Content** (shared information) should be similar across views of the same subject
- **Style** (view-specific information) should vary between T1 and T2

By applying contrastive loss **only on content dimensions**, the model is encouraged to:
1. Push content dimensions to encode shared anatomical features
2. Leave style dimensions free to capture modality-specific information

### 6.2 Multi-Level Mask Configuration

The content/style split is configured via a combination of `--content-dim`, `--total-dim`, `--content-style-levels`, `--content-ratios`, and `--content-size`:

```bash
--content-dim 128 --total-dim 512             # global ratio 0.25
--content-style-levels 0 1 2                  # apply at every level
--content-ratios      0.5 0.3 0.2             # per-level override (optional)
# or
--content-size 48 --vqvae-hidden-channels 64  # directly set 48 content / 16 style
```

The number of content channels at each masked level is `round(ratio * hidden_channels)`. Levels not listed in `--content-style-levels` pass their encoder output through unmasked.

### 6.3 Mask Modes (`--mask-mode`)

| Mode | Learnable? | Gumbel? | Per-view? | When to use |
|------|------------|---------|-----------|-------------|
| `onthefly` (default) | No | Top-k Gumbel-Softmax on mean-activation logits | Shared (logits averaged over batch+views) | Matches Yao et al. 2024; no learnable mask params. |
| `learned` | Yes (`channel_logits` nn.Parameter per level) | Top-k Gumbel-Softmax | Per-view when `--separate-encoders` (via `channel_logits_v1`) | Strongest separation signal, but can drift → stale MoCo queue. |
| `fixed` | No | No — deterministic first-K split | Shared | Avoids Gumbel noise and queue staleness entirely; good baseline for sweeps. |
| `learned_split` | Yes (per-channel sigmoid gates) | Straight-through hard sigmoid | Shared | Content size is NOT fixed — it emerges from training. Incompatible with `--inject-style-to-decoder` (style size varies per step). |

For learned modes, `--mask-warmup-steps N` runs the first N steps with in-batch InfoNCE (MoCo queue frozen) so the mask can stabilise before the queue is filled, and `--mask-lr-scale` slows mask evolution relative to the encoder.

### 6.4 Content Propagation Between Levels

When content is masked at level $l$, the next encoder level can receive one of four inputs (checked in priority order):

1. `--pass-full-to-next-level` → pass the FULL unmasked tensor forward (mask only affects codebook/contrastive paths).
2. `--narrow-encoder-input` → slice to content channels (narrower conv input, no extra params).
3. `--use-content-projection` → slice + learned 1×1×1 conv back to `hidden_channels` (`ContentProjection` module).
4. Fallback → zero-mask style channels in place.

The choice is a trade-off between preserving modality information for deeper features (option 1) and enforcing bottleneck-like information removal between levels (options 2–4). Sweep runs currently favour option 1 with `--content-style-levels 0 1 2`.

### 6.5 SplitGroupNorm

A per-level `SplitGroupNorm` (registered in `content_norms`) re-normalises content and style channel groups independently between encoder stages, so modality-specific style statistics cannot bias content activations before the mask separates them. Applied in both the shared-encoder and `--separate-encoders` paths.

### 6.6 Cross-Reconstruction Evaluation (`separation_score`)

`eval/cross_reconstruction.py::evaluate_content_style_separation` is run at the end of training (and pushed to W&B summary when `--use-wandb`). It collects pooled level-0 content/style features for every subject and computes:

| Metric | Desired value |
|--------|---------------|
| `content/cross_view_cosine_mean` | high (same subject's T1 and T2 content should match) |
| `content/modality_probe_acc` (LR on content → predict T1 vs T2) | ≈ 0.5 (chance) |
| `content/modality_invariance` = 1 − 2·|acc − 0.5| | → 1.0 |
| `style/modality_probe_acc` (LR on style → predict T1 vs T2) | → 1.0 |
| `style/subject_probe_r2` (Ridge on style → predict subject index) | ≈ 0 |
| `style/subject_invariance` = 1 − max(0, r²) | → 1.0 |
| **`separation_score`** = mean(content/modality_invariance, style/subject_invariance) | → 1.0 |

`separation_score` is the default optimisation target for the W&B Bayesian sweep (see Section 8.4).

### 6.7 Implementation Summary

A learnable parameter `channel_logits` (size `hidden_channels`, per level when masked) is trained with Gumbel-Softmax to select which specific channels are content vs style. The selection is:
1. **Differentiable** — Gumbel straight-through estimator allows gradients from both reconstruction and contrastive losses to flow back to `channel_logits`
2. **Shared** — a single Gumbel sample is drawn per step and used for both the codebook (reconstruction) and the contrastive loss
3. **Applied on hidden_channels** — the mask operates on the raw 64-dim encoder output, not on the compressed 32-dim codebook embedding, because the `conv_in` projection would mix all channels and leak modality information

```python
# In VQVAE.forward():
soft_mask = topk_gumbel_softmax(k=content_channels, logits=channel_logits)
content_enc_out = (enc_l0 * soft_mask)[:, content_idx, :]  # → level-0 codebook

# In contrastive loss (differentiable masking):
hz_content = hz * soft_mask  # NOT hz[..., content_indices]
sim = cosine_similarity(hz_content_view1, hz_content_view2)
```

### 6.8 Expected Outcomes

After training:
- Content channels should encode: brain structure, ventricle size, atrophy patterns
- Style channels should encode: T1 vs T2 contrast, intensity characteristics
- `separation_score` (Section 6.6) → 1.0 as the two probes approach chance on the adversarial axes and saturation on the desired ones.

---

## 7. File Structure

```
multiview-crl/
├── pyproject.toml              # isort / black configuration
├── .pre-commit-config.yaml     # Pre-commit hooks (black, isort, autoflake, …; flake8 disabled)
├── METHODOLOGY_REPORT.md       # This documentation
├── docker/
│   ├── dockerfile              # Docker image definition
│   ├── run_docker.sh           # RunAI submit command (A100, 1 GPU)
│   ├── run_training.sh         # Training invocation inside the container
│   ├── run_container.sh        # Interactive container launch
│   ├── run_interactive.sh      # Shell into running container
│   └── run_tensorboard.sh      # TensorBoard port-forward helper
├── data/
│   ├── datasets.py             # Dataset classes (MyCustomDataset for ADNI)
│   └── infinite_iterator.py    # Infinite data loader wrapper
├── eval/
│   ├── dci.py                  # DCI disentanglement metric
│   ├── evaluation.py           # val_step / get_data / eval_step
│   ├── cross_reconstruction.py # evaluate_content_style_separation → separation_score (W&B sweep target)
│   ├── dino.ipynb              # Self-supervised pretraining exploration
│   └── view_latents.ipynb      # Load checkpoint, extract & visualize latents
├── models/
│   ├── encoders.py             # Additional encoder architectures (text)
│   ├── pixelsnail.py           # PixelSNAIL autoregressive prior
│   └── vqvae.py                # VQ-VAE-2 (SpatialFiLM, SplitGroupNorm, ContentProjection) + MoCoEncoder
├── scripts/
│   ├── sweep_config.yaml       # W&B Bayesian sweep definition (targets separation_score)
│   ├── sweep_train.py          # W&B agent wrapper (bool-flag + list handling, in-process run_main)
│   ├── launch_sweep.sh         # Create sweep + fan out Run:AI agents
│   ├── sweep_runai.sh          # Single Run:AI submission for one sweep agent
│   └── analyze_sweep.py        # Rank sweep runs by separation_score
├── training/
│   ├── losses.py               # InfoNCE, patch-InfoNCE, MoCo InfoNCE, Barlow Twins, VICReg, BaselineLoss
│   ├── main_multimodal.py      # train_step + main (VQ-VAE-2); W&B + TB logging
│   ├── main_numerical.py       # Numerical experiment training script
│   └── trainer.py              # PixelSNAIL prior trainer
└── utils/
    ├── checkpointing.py        # save/load checkpoint, emergency checkpoint
    ├── config.py               # parse_args / update_args / compute_gt_idx
    ├── helper.py               # HelperModule base class
    ├── latent_spaces.py        # Latent space utilities
    ├── logging_setup.py        # setup_logging (file + console handlers)
    ├── spaces.py               # Space definitions
    ├── utils.py                # MONAI transforms, Gumbel-Softmax, utilities
    └── visualisation.py        # save_decoded_images / save_vqvae_decoded_images

results/
└── ADNI_registered/
    └── {vqvae_model_id}/
        ├── settings.json           # Training configuration
        ├── Training.csv            # Loss history (CSV)
        ├── vqvae_model.pt          # Full VQ-VAE-2 checkpoint — latest (incl. MoCo queues if used)
        ├── vqvae_best.pt           # Best VQ-VAE-2 checkpoint (by rolling avg total loss)
        ├── tensorboard/            # TensorBoard event files
        └── decoded_images/
            ├── step_00001_original.nii.gz
            └── step_00001_decoded.nii.gz
```

---

## 8. Running the Code

### 8.1 Training Commands

**Current production run (A100, via RunAI — `docker/run_training.sh`):**
```bash
python training/main_multimodal.py \
    --dataroot /nfs/home/nglazman \
    --dataset_name ADNI_registered \
    --encoder-type vqvae \
    --use-moco \
    --moco-queue-size 4096 \
    --moco-momentum 0.999 \
    --vqvae-nb-levels 3 \
    --vqvae-scaling-rates 2 2 2 \
    --vqvae-hidden-channels 48 \
    --vqvae-embed-dim 24 \
    --content-dim 384 \
    --total-dim 512 \
    --inject-style-to-decoder \
    --lr 0.00001 \
    --train-steps 50000 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --gradient-checkpointing \
    --skip-recon-ratio 0.3 \
    --image-spacing 1.0 \
    --crop-margin 10 \
    --use-amp \
    --resume-training \
    --model-id vqvae-384-128
```
*(Submitted via `docker/run_docker.sh` using RunAI — see Section 8.3)*

**VQ-VAE-2 with standard InfoNCE:**
```bash
python training/main_multimodal.py \
    --dataroot /data/natalia \
    --dataset_name ADNI_registered \
    --encoder-type vqvae \
    --vqvae-nb-levels 3 \
    --vqvae-scaling-rates 4 2 2 \
    --content-dim 256 --total-dim 512 \
    --train-steps 10000 \
    --batch-size 4 \
    --use-amp \
    --workers 4 \
    --model-id vqvae_experiment
```

**Resume Training:**

Training now auto-resumes whenever a compatible checkpoint exists in the save directory, so `--resume-training` is technically optional. Pass it explicitly for clarity:
```bash
python training/main_multimodal.py \
    --dataroot /data/natalia \
    --dataset_name ADNI_registered \
    --encoder-type vqvae \
    --model-id vqvae_experiment \
    --resume-training \
    --train-steps 20000
```

### 8.3 RunAI Cluster Submission

Jobs are submitted to a NVIDIA A100 GPU via RunAI using the scripts in `docker/`:

```bash
# Submit training job (runs docker/run_training.sh inside the container)
bash docker/run_docker.sh
```

`run_docker.sh` calls `runai submit` with:
- Image: `aicregistry:5000/nglazman:multiview-crl-vqvae-latest`
- Node type: A100, 1 GPU, 16–32 CPU cores, 64–128 GB RAM
- NFS mount: `/nfs` for data and code access

### 8.2 Key Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataroot` | `/data/natalia/` | Root data directory |
| `--dataset_name` | `ADNI_registered` | Dataset name |
| `--encoder-type` | `vqvae` | Fixed to `vqvae` (hierarchical VQ-VAE-2) |
| `--batch-size` | 2 | Samples per view per batch |
| `--lr` | 1e-5 | Learning rate |
| `--train-steps` | 300001 | Total training steps |
| `--tau` | 1.0 | Temperature for InfoNCE / MoCo |
| `--scale-recon-loss` | 1.0 | Weight for reconstruction loss |
| `--scale-contrastive-loss` | 1.0 | Weight for contrastive loss |
| `--use-amp` | False | Enable mixed precision |
| `--model-id` | Auto | Experiment identifier |
| `--resume-training` | False | Resume from last checkpoint |
| `--image-spacing` | 2.0 | Isotropic voxel spacing (mm) |
| `--crop-margin` | 0 | Voxels to crop from each edge |
| `--gradient-accumulation-steps` | 1 | Accumulate N mini-batches before optimizer step |
| `--skip-recon-ratio` | 0.0 | Fraction of steps that skip decoder (0–1) |
| `--weight-decay` | 0.01 | AdamW weight decay (biases, norms, ReZero alphas are exempted) |
| `--warmup-steps` | 1000 | Linear LR warmup steps (0 disables) |
| `--lr-schedule` | `cosine` | Post-warmup schedule: `cosine` or `constant` |
| `--lr-min` | 0.0 | Minimum LR for cosine annealing |
| `--recon-loss-start-step` | 0 | Step at which to start applying the reconstruction loss |
| `--spatial-size D H W` | None | Explicit input spatial size after resampling (overrides `--image-spacing` / `--crop-margin` derived shape) |
| `--early-stopping-patience` | 0 | Stop if rolling monitored loss fails to improve for N checkpoint intervals (0 disables). Uses Val/Total when `--val-every > 0`, else training loss |
| `--early-stopping-min-delta` | 0.0 | Minimum improvement to count as progress |
| `--dataset_name` | `ADNI_registered` | Also supports `ADNI_stripped` (skull-stripped variant) |

**Contrastive Objective Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--contrastive-loss-type` | `infonce` | One of `infonce`, `barlow_twins`, `vicreg` |
| `--bt-lambda` | 0.005 | Barlow Twins off-diagonal weight |
| `--vicreg-sim-coeff` | 25.0 | VICReg invariance (MSE) weight |
| `--vicreg-std-coeff` | 25.0 | VICReg variance (hinge) weight |
| `--vicreg-cov-coeff` | 1.0 | VICReg covariance (decorrelation) weight |
| `--patch-contrastive` | False | Use dense patch-level alignment instead of global pooling |
| `--patch-grid D H W` | `4 5 4` | Patch grid for `--patch-contrastive` |
| `--cross-view-negs-only` | False | Restrict InfoNCE negatives to the other view (recommended with `--separate-encoders`) |
| `--contrastive-level-weights w0 w1 …` | None | Per-level weight for the contrastive loss (default: uniform 1.0) |

**VQ-VAE-2 Specific Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--vqvae-hidden-channels` | 64 | Hidden layer channels |
| `--vqvae-res-channels` | 32 | Residual block channels |
| `--vqvae-nb-levels` | 3 | Number of hierarchical levels |
| `--vqvae-embed-dim` | 32 | Codebook embedding dimension |
| `--vqvae-nb-entries` | 384 | Codebook size |
| `--vqvae-scaling-rates` | [2, 2, 2] | Downscale factor per level |
| `--vq-commitment-weight` | 0.25 | VQ commitment loss weight |
| `--gradient-checkpointing` | False | Trade compute for memory in residual blocks |
| `--content-dim` | 128 | Content dimensions (determines `content_channels` ratio on `hidden_channels`) |
| `--total-dim` | 512 | Total dims (`content_dim + style_dim`); ratio `content_dim/total_dim` sets channel split |
| `--inject-style-to-decoder` | False | Feed style channels from masked encoder levels into each decoder. With `--style-injection-mode concat` style is appended to the penultimate feature map; with `film` a `SpatialFiLM` modulator is applied at every decoder stage. |
| `--style-injection-mode` | `concat` | `concat` (legacy) or `film` (Spatial FiLM at every decoder stage) |
| `--quantize-style` | False | Quantize style channels through independent per-level codebooks (requires `--inject-style-to-decoder`) |
| `--style-embed-dim` | None | Embedding dimension for style codebooks (defaults to `--vqvae-embed-dim`) |
| `--style-nb-entries` | None | Codebook size for style codebooks (defaults to `--vqvae-nb-entries`) |
| `--cache-dir` | None | Directory for persistent preprocessed `.pt` cache (NFS-safe, fingerprinted) |
| `--val-every` | 0 | Run validation every N steps (0 disables periodic validation) |
| `--cb-ema-decay` | 0.999 | EMA momentum for codebook running averages |
| `--cb-reset-every` | 100 | Reset dead codebook entries every N forward passes per codebook (0 disables) |
| `--cb-reset-threshold` | 1.0 | EMA cluster_size below this value marks a codebook entry as dead |
| `--top-level-recon-only` | False | Zero out encoder outputs at non-top levels before the codebook (reconstruction depends only on coarsest level; contrastive still uses all levels) |

**Content/Style Masking Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--content-style-levels L …` | `[0]` | Encoder levels that receive the content/style Gumbel mask. Use `0 1 2` for all three levels. |
| `--content-ratios r0 r1 …` | None | Per-level content ratio (fraction of `hidden_channels` that are content), one per entry in `--content-style-levels`. Example: `--content-style-levels 0 1 2 --content-ratios 0.5 0.3 0.2`. |
| `--content-size N` | None | Directly set the number of content channels out of `--vqvae-hidden-channels` (overrides the `--content-dim/--total-dim` ratio; applies uniformly across `--content-style-levels`). |
| `--mask-mode` | `onthefly` | One of `onthefly` (logits = mean activation per channel, shared across views), `learned` (persistent nn.Parameter per level; per-view when `--separate-encoders`), `fixed` (first K channels = content, no Gumbel noise), `learned_split` (per-channel sigmoid gates, content size emerges from training). |
| `--mask-warmup-steps` | 0 | When `learned`/`learned_split` + MoCo, disable the queue for N steps and use in-batch InfoNCE so the mask can stabilise; queue is flushed afterwards. |
| `--mask-lr-scale` | 1.0 | LR multiplier for channel_logits (set <1 to slow mask evolution). |
| `--separate-encoders` | False | Per-view encoder stacks (one VQVAE encoder per modality). Codebooks, decoders, and masks remain shared. Consistent with view-specific identifiability theory (Yao et al., 2024). |
| `--narrow-encoder-input` | False | After masking, slice to content channels before feeding the next encoder level (narrower input — no extra params). |
| `--use-content-projection` | False | Slice to content channels and pass through a learned 1×1×1 conv back to `hidden_channels` before the next encoder level. |
| `--pass-full-to-next-level` | False | Pass the FULL encoder output (content + style) to the next level — masking only affects the codebook and contrastive paths. Incompatible with `--narrow-encoder-input` and `--use-content-projection`. |

**MoCo Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-moco` | False | Enable MoCo contrastive training (replaces standard InfoNCE; ignored with `barlow_twins`/`vicreg`) |
| `--moco-queue-size` | 4096 | Number of negative keys stored per encoder level |
| `--moco-momentum` | 0.999 | EMA momentum for momentum encoder updates |

**Weights & Biases Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-wandb` | False | Mirror all TB scalars + summary metrics to W&B |
| `--wandb-project` | `multiview-crl-sweep` | W&B project name |
| `--wandb-entity` | None | W&B team/entity (uses default if unset) |

### 8.4 W&B Bayesian Sweep

Hyperparameter search is orchestrated by W&B Bayesian sweeps, optimising the composite `separation_score` (Section 6.6).

**Files:**
- `scripts/sweep_config.yaml` — sweep definition (method, metric, parameter grid/distributions, fixed args).
- `scripts/sweep_train.py` — W&B agent entrypoint. Calls `wandb.init()`, reconstructs the CLI argv from `wandb.config` (handles bool flags and stringified list values like `"[4, 5, 4]"`), forces `vqvae_embed_dim == vqvae_hidden_channels` when both are in the grid, and calls `run_main(args)` in the same process (avoids daemon-conflict issues that drop W&B metrics when spawning a subprocess).
- `scripts/launch_sweep.sh` — creates the sweep via `wandb sweep …` and fans out Run:AI agents with `scripts/sweep_runai.sh`.
- `scripts/analyze_sweep.py` — ranks runs by `separation_score` and prints top-K configurations + per-parameter marginals.

**Current sweep space (`scripts/sweep_config.yaml`):**

| Tier | Parameter | Distribution / values |
|------|-----------|-----------------------|
| 1 | `scale_contrastive_loss` | log-uniform 0.1 – 10 |
| 1 | `scale_recon_loss` | log-uniform 0.1 – 10 |
| 1 | `vqvae_hidden_channels` (= `vqvae_embed_dim`) | {32, 48, 64} |
| 1 | `mask_mode` | {`fixed`, `learned`} |
| 1 | `tau` | log-uniform 0.05 – 1.0 |
| 1 | `content_ratios` | 5 preset pyramids, e.g. `[0.5, 0.3, 0.2]`, `[0.7, 0.7, 0.7]`, `[0.3, 0.5, 0.7]` |
| 2 | `lr` | log-uniform 5e-4 – 2e-3 |
| 2 | `bt_lambda` | log-uniform 1e-3 – 5e-2 |

Fixed sweep parameters include `patch_contrastive=true` with `patch_grid=[4, 5, 4]`, `content_style_levels=[0, 1, 2]`, `cross_view_negs_only=true`, `separate_encoders=true`, `pass_full_to_next_level=true`, `gradient_checkpointing=true`, `skip_recon_ratio=0.5`, `recon_loss_start_step=2000`, `image_spacing=1.0`, `spatial_size=[150, 180, 150]`, `train_steps=20000`, `resume_training=true`, `use_wandb=true`, and a cache directory on NFS.

```bash
# Launch a fresh sweep with 15 Run:AI agents
./scripts/launch_sweep.sh --num-agents 15 --wandb-project multiview-crl-sweep-new

# Analyse a completed sweep
python scripts/analyze_sweep.py <sweep-id>
```

---

## 9. Current Training Status

Based on recent VQ-VAE-2 training runs:

| Step | Total Loss | Contrastive | Recon + VQ |
|------|------------|-------------|------------|
| 1 | 9.64 | 2.76 | 6.89 |
| 25 | 7.11 | 0.86 | 6.26 |
| 50 | 7.16 | 1.77 | 5.39 |

**Observations:**
- Contrastive loss fluctuates due to Gumbel-Softmax exploration
- VQ commitment losses per level: L0≈1.5, L1≈0.3, L2≈0.2 (top level most stable)

---

## 10. Checkpoint Compatibility

### 10.1 VQ-VAE-2 State Dict Keys

The `ResidualStack` module stores its layers as `self.stack` (an `nn.Sequential`). Checkpoint state dict keys follow the pattern:

```
module.encoders.0.layers.3.stack.0.alpha
module.encoders.0.layers.3.stack.0.layers.0.weight
...
```

Older checkpoints may use `.blocks.` instead of `.stack.` (from a prior refactor that used `nn.ModuleList`). The `eval/view_latents.ipynb` notebook handles this automatically by converting `.blocks.` → `.stack.` when loading:

```python
for key, val in state_dict.items():
    new_key = key.replace('.blocks.', '.stack.')
    new_state_dict[new_key] = val
```

### 10.2 DataParallel Prefix

All checkpoints are saved from a `DataParallel`-wrapped model, so keys start with `module.`. The notebook wraps the model in `DataParallel` before loading to match.

### 10.3 MoCo Queue State

When `--use-moco` is active, the per-level queues and queue pointers are saved as explicit `moco_queues` / `moco_queue_ptrs` keys in `vqvae_model.pt` and restored by `load_checkpoint`. The restore path checks for the presence of these keys before attempting restoration (so non-MoCo checkpoints can be loaded into a MoCo-enabled model without error — queues start cold).

### 10.4 Gumbel Mask Bool-Cast Fix

A bug was fixed where `content_mask` (output of `topk_gumbel_softmax`) was passed directly to `torch.where()` without an explicit `.bool()` cast. On some PyTorch versions this produced incorrect index selection. The fix:

```python
# Before (buggy):
content_idx = torch.where(content_mask)[-1].tolist()
style_idx   = torch.where(~content_mask)[-1].tolist()

# After (correct):
content_mask_bool = content_mask.bool()
content_idx = torch.where(content_mask_bool)[-1].tolist()
style_idx   = torch.where(~content_mask_bool)[-1].tolist()
```

This affects all checkpoints trained after commit `d968f53` (24 Feb 2026). Checkpoints trained before this fix may have had unreliable content/style channel splits.

### 10.5 `decode_codes` 3D Volume Fix

The `decode_codes` method in `models/vqvae.py` (used for decoding from raw codebook indices without a forward pass) contained a 2D→3D adaptation bug. The `embed_code` call on 3D spatial indices `(B, D, H, W)` returns a 5D tensor `(B, D, H, W, embed_dim)`, but the permute call used 4D indices:

```python
# Before (buggy — line 581):
code_q = codebook.embed_code(cs[l]).permute(0, 3, 1, 2)  # 4D permute on 5D tensor

# After (correct):
code_q = codebook.embed_code(cs[l]).permute(0, 4, 1, 2, 3)  # channels-first for 3D
```

This only affects the `decode_codes` path (used in evaluation for code replacement experiments). The standard `forward()` path was unaffected because it uses `quantize()` which handles the permutation correctly.

---

## 11. Latent Visualization (`eval/view_latents.ipynb`)

The notebook extracts and visualizes latent representations from trained VQ-VAE-2 models. It is organized into 16 sections:

### 11.1 Core Analysis (Sections 1–10)

1. **Configuration & imports** — checkpoint path (supports `vqvae_best.pt` or `vqvae_model.pt`), data paths, device setup
2. **Load checkpoint** — handles `.blocks.` → `.stack.` key conversion for backward compatibility
3. **Load ADNI data** — supports both original resolution and 2mm downsampled modes
4. **Extract encoder features** — global average pooling of each hierarchical level
5. **PCA** — content-only filtering at level 0 (uses Gumbel mask `content_idx`), coloured by diagnosis (AD, MCI, CN) and modality (T1, T2)
6. **t-SNE** — same content-only filtering as PCA to prevent style channels from dominating the embedding
7. **Paired distance analysis** — compare T1 vs T2 latent distances per subject
8. **Diagnostic prediction** — 5-fold CV logistic regression and random forest per feature set (content-only, style-only, all, per-level)
9. **Reconstruction visualization** — mid-sagittal slice comparison of originals vs reconstructed
10. **Content/style embedding statistics** — norm distributions, correlation matrices

### 11.2 Codebook Analysis (Sections 11–16)

11. **Codebook index extraction** — extracts discrete codebook indices at all levels for every subject (T1 + T2); stores per-subject normalized usage histograms; keeps one T1 example per diagnosis class (CN, AD, MCI) for code-replacement demo; captures MONAI MetaTensor affine for NIfTI export
12. **Codebook usage by diagnosis** — per-level bar charts showing mean code frequency by class (CN, MCI, AD) and stacked heatmaps; identifies class-specific code utilisation patterns
13. **Mutual information & chi-squared discriminativeness** — for each code at each level, computes MI(code_present, diagnosis_label) and chi-squared test of independence; ranks codes by discriminative power; prints top-10 most discriminative codes per level
14. **PCA & t-SNE of codebook usage histograms** — treats each subject's codebook histogram as a feature vector; produces per-level and combined (concatenated across levels) embeddings; coloured by diagnosis and modality
15. **Code replacement & reconstruction (CN vs AD)** — replaces the most common code at a target level with the least common; decodes back to image space using `decode_codes`; compares the effect on a healthy (CN) vs diseased (AD) subject side by side; prints embedding diagnostics (L2 distance, cosine similarity between swapped codes); shared intensity scales for direct visual comparison
16. **NIfTI export** — saves original input, reconstruction, code-modified reconstruction, and absolute difference map as `.nii.gz` files for each class; uses post-transform affine from MONAI MetaTensor (or constructs fallback from preprocessing params); files can be viewed in FSLeyes / ITK-SNAP / 3D Slicer

### 11.3 Import Structure

The notebook lives in `eval/` but imports from the project root packages. The first cell adds the project root to `sys.path` and uses fully-qualified module paths:

```python
import sys, os
sys.path.insert(0, os.path.abspath('..'))   # project root

import models.vqvae as vqvae                 # was: import vqvae
import utils.utils as utils                  # was: import utils
from utils.utils import load_data, CreateBrainMaskd, ApplyBrainMaskd
```

The transforms cell maps `RESAMPLE_MODE` to a `spacing` float (`1.0` or `2.0`) consistent with `--image-spacing` used during training.

---

## 12. Future Work and Considerations

### 12.1 Preventing Modality Leakage into Content

Current observation: content latents at deeper levels may still carry some modality (T1 vs T2) information despite the contrastive objective. Approaches under consideration:

1. **Adversarial modality prediction (gradient reversal)** — Add an MLP head that predicts modality from content features, trained with a gradient reversal layer (GRL). The encoder learns to make content features that fool the modality discriminator. Most established approach (Ganin et al., DANN).

2. **HSIC regularization** — Add a kernel-based penalty (Hilbert-Schmidt Independence Criterion) that directly measures and penalizes statistical dependence between content features and modality labels. Simpler than adversarial training (no minimax instability).

3. **Tighter information bottleneck at `content_proj`** — Reduce output dimension, add noise injection (variational information bottleneck), or apply channel-wise dropout at the content projection layer.

4. **Modality-specific normalization at level 0** — Use separate Instance Norm / Batch Norm statistics for T1 and T2 at the first encoder level, absorbing modality-specific intensity distributions before features enter `content_proj`.

### 12.2 Identifiability Evaluation

Further evaluations to demonstrate that learned latents are identifiable:

1. **DCI (Disentanglement, Completeness, Informativeness) scores** — already partially implemented in `eval/dci.py`
2. **Downstream classification** — Alzheimer's vs. healthy using content-only features (5-fold CV logistic regression and random forest already in notebook)
3. **Cross-view reconstruction** — encode T1, decode with T2 style channels (and vice versa)
4. **Style transfer experiments** — apply T1 style to T2 content to verify separation
5. **Position-wise codebook pair analysis** — for each spatial position at each level, identify the most common code pairs across subjects and assess whether they carry class-discriminative information

### 12.3 Codebook Analysis (Completed)

The following analyses have been implemented in `eval/view_latents.ipynb` (Sections 11–16):
- Codebook usage histograms by diagnosis class
- Mutual information and chi-squared tests for code discriminativeness
- PCA/t-SNE of codebook usage histograms
- Code replacement & reconstruction with CN vs AD comparison
- NIfTI export for interactive 3D inspection

### 12.4 MoCo Tuning

- Experiment with different queue sizes (256 → 8192) and momentum values (0.99 → 0.9999)
- Implement a warm-up phase for the queue to reduce cold-start noise
- Compare MoCo vs standard InfoNCE on downstream classification tasks

### 12.5 VQ-VAE-2 Improvements

- Increase codebook size for more expressivity
- Add PixelSNAIL prior for generative sampling
- Experiment with different scaling rates

### 12.6 Hyperparameter Tuning

- VQ commitment weight optimization
- Content/style ratio exploration
- Temperature scheduling for contrastive loss

### 12.7 Memory Optimization (available now)

- Gradient checkpointing (`--gradient-checkpointing`)
- Gradient accumulation (`--gradient-accumulation-steps N`)
- Skip reconstruction (`--skip-recon-ratio R`)
- MoCo removes the need for large batches, directly reducing GPU memory pressure

---

## 13. Summary

This implementation provides a framework for learning disentangled representations from paired multimodal brain MRI data.

### Architecture:
- **Hierarchical 3D VQ-VAE-2** with 3 levels (~2.9M params total)
- **Discrete codebook representations** (384 codes × 32 dims per level) with optional **independent style codebooks** (`--quantize-style`)
- **Hierarchical contrastive loss** at all encoder levels with three objectives — InfoNCE (with optional MoCo queue, cross-view-only negatives, and dense patch alignment), Barlow Twins, or VICReg
- **MoCo option** (`--use-moco`): momentum encoder + per-level queues (4096 negatives per level) for effective contrastive learning with small batches
- **Multi-level content/style masking** — any combination of encoder levels can carry a Gumbel mask, with per-level content ratios; four mask modes (`onthefly`, `learned`, `fixed`, `learned_split`) trade off learnability vs. queue stability
- **Per-view encoders** (`--separate-encoders`) consistent with view-specific identifiability theory (Yao et al., 2024); shared codebooks/decoders/masks
- **Style injection** (`--inject-style-to-decoder`) with `concat` or `film` modes; spatial FiLM modulates every decoder stage
- **Codebook-collapse prevention** via EMA decay + periodic dead-entry reset
- **EMA codebook updates** for stable training
- **Auto-resume** with architecture compatibility check — safe to rerun after changing architecture flags
- **Best-model tracking** via rolling average total loss (`vqvae_best.pt`)

### Evaluation:
- **Cross-reconstruction probes** — `separation_score` composite metric (content → modality invariance + style → subject invariance)
- **Codebook analysis** — usage histograms, MI/chi-squared discriminativeness, PCA/t-SNE of codebook usage per diagnosis class
- **Code replacement** — swap codebook entries and compare reconstructions between CN and AD subjects
- **NIfTI export** — save reconstructed volumes with correct affine for 3D inspection in FSLeyes / ITK-SNAP
- **Downstream classification** — logistic regression and random forest on content-only, style-only, and combined features

### Hyperparameter Search:
- **W&B Bayesian sweep** (`scripts/sweep_config.yaml` + `scripts/sweep_train.py` + `scripts/launch_sweep.sh`) optimising `separation_score` across contrastive/reconstruction scales, `vqvae_hidden_channels`, mask mode, temperature, content ratios, LR, and Barlow Twins λ. Run:AI agents are spawned via `scripts/sweep_runai.sh`; `scripts/analyze_sweep.py` ranks finished runs.

The model learns to encode shared anatomical information in content dimensions while allowing style dimensions to capture modality-specific (T1 vs T2) characteristics.
