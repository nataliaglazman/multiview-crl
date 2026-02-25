# Multiview Contrastive Representation Learning on ADNI Brain MRI

## Technical Report for Supervisor Review

**Project:** `multiview-crl`
**Dataset:** ADNI (Alzheimer's Disease Neuroimaging Initiative) - Registered T1 and T2 MRI scans
**Date:** February 2026
**Last Updated:** 25 February 2026

### Changelog

| Date | Changes |
|------|--------|
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

### 1.2 Two Encoder Architectures

The project supports two encoder architectures:

| Architecture | Description | Use Case |
|-------------|-------------|----------|
| **VAE** | Simple 3D CNN encoder with separate decoder | Baseline, faster training |
| **VQ-VAE-2** | Hierarchical Vector Quantized VAE with 3 levels | Discrete representations, better disentanglement |

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

---

## 3. Model Architectures

### 3.1 Architecture Selection

The encoder type is selected via command-line argument:

```bash
--encoder-type vae      # Use simple VAE encoder/decoder
--encoder-type vqvae    # Use hierarchical VQ-VAE-2 (default)
```

---

## 3.2 VAE Architecture (`models/vae.py`)

#### 3.2.1 Encoder

A 3D convolutional neural network with ResNet-style skip connections:

| Layer | Input Channels | Output Channels | Output Size | Description |
|-------|---------------|-----------------|-------------|-------------|
| Input | 1 | - | $(91, 109, 91)$ | Single-channel 3D MRI |
| Conv1 + ResBlock | 1 | 32 | $(46, 55, 46)$ | 3×3×3 conv + MaxPool |
| Conv2 + ResBlock | 32 | 64 | $(23, 28, 23)$ | 3×3×3 conv + MaxPool |
| Conv3 + ResBlock | 64 | 128 | $(12, 14, 12)$ | 3×3×3 conv + MaxPool |
| Conv4 + ResBlock | 128 | 256 | $(6, 7, 6)$ | 3×3×3 conv + MaxPool |
| Conv5 + ResBlock | 256 | 512 | $(3, 4, 3)$ | 3×3×3 conv + MaxPool |
| AdaptiveAvgPool | 512 | 512 | $(1, 1, 1)$ | Global average pooling |
| Flatten | - | - | **512** | Final latent vector |

**Architecture Details:**
- **Normalization:** GroupNorm (16 groups) for stable training with small batch sizes
- **Activation:** ReLU for conv blocks, residual connections for gradient flow
- **Total Encoder Parameters:** ~15.5M

#### 3.2.2 Decoder

A symmetric decoder using transposed convolutions for sharper reconstructions:

| Layer | Input Channels | Output Channels | Output Size | Description |
|-------|---------------|-----------------|-------------|-------------|
| Linear | 512 | 512×2×2×2 | $(512, 2, 2, 2)$ | Project to spatial |
| TransConv1 + ResBlock | 512 | 512 | $(4, 4, 4)$ | Upsample ×2 |
| TransConv2 + ResBlock | 512 | 256 | $(8, 8, 8)$ | Upsample ×2 |
| TransConv3 + ResBlock | 256 | 128 | $(16, 16, 16)$ | Upsample ×2 |
| TransConv4 + ResBlock | 128 | 64 | $(32, 32, 32)$ | Upsample ×2 |
| TransConv5 + ResBlock | 64 | 32 | $(64, 64, 64)$ | Upsample ×2 |
| TransConv6 + ResBlock | 32 | 16 | $(128, 128, 128)$ | Upsample ×2 |
| Refine Block | 16 | 1 | $(128, 128, 128)$ | 3 conv layers |
| Trilinear Upsample | 1 | 1 | $(91, 109, 91)$ | Final resize |

**Architecture Details:**
- **Transposed Convolutions:** 4×4×4 kernels, stride 2 for learned upsampling
- **Refinement Block:** 3 conv layers (16→16→8→1) for sharper output
- **Final Activation:** None (unbounded output for normalized data)
- **Total Decoder Parameters:** ~27.8M

#### 3.2.3 VAE Latent Space Structure

| Property | Value | Description |
|----------|-------|-------------|
| **Total Latent Dimensions** | 512 | Fixed encoder output |
| **Content Dimensions** | 256 (indices 0-255) | Shared between T1/T2 |
| **Style Dimensions** | 256 (indices 256-511) | View-specific |

---

## 3.3 VQ-VAE-2 Architecture (`models/vqvae.py`)

The VQ-VAE-2 is a **hierarchical Vector Quantized Variational Autoencoder** adapted for 3D brain MRI. It uses discrete codebook representations at multiple scales, enabling multi-resolution feature learning.

### 3.3.1 Architecture Overview

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

### 3.3.2 Configuration Parameters

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
| `--content-dim` | 128 | Content dimensions for `content_proj` (must match training when loading checkpoint) |
| `--total-dim` | 512 | Total latent dims (`content_dim + style_dim`) for `content_proj` |
| `--inject-style-to-decoder` | False | Feed style channels from encoder level-0 into the final decoder layer |

### 3.3.3 Encoder Details (Per Level)

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

### 3.3.4 Vector Quantization Layer

The `CodeLayer` implements EMA (Exponential Moving Average) codebook updates:

```python
# Discrete bottleneck
z_e = encoder_output                    # (B, 64, D, H, W)
z_q = nearest_codebook_entry(z_e)       # (B, 32, D, H, W)
commitment_loss = ||z_e - sg[z_q]||²    # Straight-through gradient
```

**Key Properties:**
- **Codebook Size:** 384 discrete codes (default)
- **Embedding Dimension:** 32
- **Update Method:** EMA (no codebook gradient, more stable than loss-based)
- **Decay Rate:** 0.99

### 3.3.5 Decoder Details

Each decoder receives concatenated codes from current and all higher levels:

| Level | Input Channels | Upscale | Output |
|-------|---------------|---------|--------|
| 2 (top) | 32 (1 code) | 2× | 32 (embed_dim) |
| 1 | 64 (2 codes) | 2× | 32 (embed_dim) |
| 0 (bottom) | 96 (3 codes) | 4× | 1 (image) |

### 3.3.6 Style Injection to Decoder (`--inject-style-to-decoder`)

By default, only the **content channels** (selected by the Gumbel mask) are passed forward through `content_proj` to deeper encoders and the decoder chain. The **style channels** (the complement within `hidden_channels`) are discarded after level-0, so the decoder has no signal about modality-specific features.

When `--inject-style-to-decoder` is enabled, the style channels are instead captured and fed back into the bottom decoder (`decoders[0]`) before the final output conv:

```
Encoder level-0 output  (B, hidden_channels, D, H, W)
         │
         ├─ content_idx channels ──► content_proj ──► deeper encoders / codebooks
         │
         └─ style_idx channels ──► detach ──────────────────────────────────┐
                                                                             ▼
                                                          decoders[0] penultimate feat
                                                          trilinear upsample to match
                                                          cat([feat, style], dim=1)
                                                                             ▼
                                                               final_conv ──► output
```

**Implementation details:**
- `Decoder` is built with `style_channels = hidden_channels − content_channels` when injection is active; a separate `self.final_conv` (replacing the original merged last layer) accepts the concatenated feature map.
- The style tensor is **detached** before concatenation so the reconstruction gradient does not flow back through the encoder's style channels a second time (the contrastive loss already provides that signal).
- If the spatial size of the style map differs from the penultimate decoder feature map, trilinear interpolation is applied automatically.
- Has **no effect** unless `content_proj` is active (i.e. `--content-dim` and `--total-dim` must be set).

**Motivation:** Giving the decoder access to style (modality-specific) channels should improve reconstruction quality for T2 scans, whose contrast differs markedly from T1, without weakening the contrastive signal on content dimensions.

### 3.3.7 Total Parameters

With default configuration:
- **Total VQ-VAE-2 Parameters:** ~2.9M (much smaller than VAE)

### 3.3.8 Reconstruction Loss (BaselineLoss)

The reconstruction loss (`BaselineLoss` in `training/losses.py`) combines three complementary terms:

$$\mathcal{L}_{recon} = \underbrace{\mathcal{L}_{pixel}}_{\lambda_p=1.0} + \underbrace{\mathcal{L}_{FFT}}_{\lambda_f=1.0} + \underbrace{\mathcal{L}_{LPIPS}}_{\lambda_{perc}=0.002}$$

| Component | Implementation | Purpose |
|-----------|---------------|----------|
| **Pixel loss** | L1 distance | Voxel-level accuracy |
| **Frequency loss** | MSE on FFT magnitudes (ortho-normalized) | Penalize missing high-frequency detail |
| **Perceptual loss** | LPIPS (SqueezeNet backbone, 512 random axial slices) | Structural/feature-level similarity |

For VQ-VAE, VQ commitment losses per codebook level are added *on top* of this reconstruction loss and logged separately.

---

## 3.4 Hierarchical Contrastive Learning (VQ-VAE-2)

A key innovation in this implementation is **hierarchical contrastive learning** applied across all encoder levels.

### 3.4.1 Motivation

Different encoder levels capture different types of information:
- **Level 0 (bottom):** Fine-grained local features (textures, edges)
- **Level 1 (middle):** Mid-level structures (gyri patterns, regional anatomy)
- **Level 2 (top):** Global/abstract features (overall brain shape, ventricle size)

By applying contrastive loss at each level, we enforce content/style disentanglement across all scales.

### 3.4.2 Content Selection Strategy

Content indices are selected **only at Level 0**, then propagated proportionally to higher levels:

```python
# Level 0: Learn content/style split via Gumbel-Softmax
content_ratio = 256/512 = 0.5  # From args.content_indices
n_channels = 64               # Encoder output channels
content_size = int(0.5 * 64) = 32  # 50% of channels as content

# Use Gumbel-Softmax to dynamically select which 32 dims are content
content_masks = gumbel_softmax_mask(logits, content_size)

# Higher levels: Use same 50% ratio
level_1_content = 32 dims
level_2_content = 32 dims
```

### 3.4.3 Training Loop for VQ-VAE-2

```python
for level_idx, encoder_output in enumerate(encoder_outputs):
    # Global average pool: (B, 64, D, H, W) → (B, 64)
    hz = encoder_output.mean(dim=[2,3,4])

    # Reshape for contrastive: (n_views, batch, channels)
    hz = hz.reshape(2, batch_size, 64)

    if level_idx == 0:
        # Select content indices via Gumbel-Softmax
        content_indices = gumbel_select(hz, k=32)
        content_ratio = 32/64
    else:
        # Proportional content for higher levels
        content_size = int(content_ratio * n_channels)
        content_indices = range(content_size)

    # Contrastive loss on content dims only
    level_loss = infonce_loss(hz, content_indices)
    total_contrastive_loss += level_loss

# Total loss
total_loss = contrastive_loss + recon_loss * scale + vq_commitment_loss
```

### 3.4.4 VQ-VAE-2 Loss Components

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

**For VAE mode:**
$$\mathcal{L}_{total} = \mathcal{L}_{contrastive} + \lambda \cdot \mathcal{L}_{reconstruction}$$

**For VQ-VAE-2 mode:**
$$\mathcal{L}_{total} = \sum_{l=0}^{2} \mathcal{L}_{contrastive}^{(l)} + \lambda_r \cdot \mathcal{L}_{recon} + \lambda_c \cdot \sum_{l=0}^{2} \mathcal{L}_{VQ}^{(l)}$$

Where $\lambda_r$ = `scale_recon_loss` (default: 0.00001), $\lambda_c$ = `vq_commitment_weight` (default: 0.25)

### 4.2 Standard Contrastive Loss (InfoNCE)

Implemented in `training/losses.py` as `infonce_loss`. Used by default when `--use-moco` is not set.

$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(z^{(1)}_c, z^{(2)}_c) / \tau)}{\sum_{k} \exp(\text{sim}(z^{(1)}_c, z^{(k)}_c) / \tau)}$$

**Key Properties:**
- **Similarity Metric:** Cosine similarity
- **Temperature ($\tau$):** 1.0 (default)
- **Content-Only Similarity:** Only `content_indices` dimensions are used for similarity computation
- **Positive Pairs:** Same subject, different views (T1, T2)
- **Negative Pairs:** Different subjects within the batch

**Limitation:** With small batches (batch size 2–4), the number of in-batch negatives is very small, which weakens the contrastive signal. MoCo addresses this (see Section 4.3).

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

For VQ-VAE-2 mode, each codebook level has a commitment loss:

$$\mathcal{L}_{VQ}^{(l)} = ||z_e^{(l)} - \text{sg}[z_q^{(l)}]||^2$$

Where:
- $z_e^{(l)}$ is the encoder output at level $l$
- $z_q^{(l)}$ is the quantized codebook entry
- $\text{sg}[\cdot]$ is the stop-gradient operator

The codebook is updated via EMA (Exponential Moving Average) rather than gradient descent.

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
| Loss printing (console) | Every step | stdout |
| TensorBoard scalars | Every step | `{save_dir}/tensorboard/` |
| CSV logging (smoothed avg) | Every `--log-steps` steps (default 100) | `{save_dir}/Training.csv` |
| Decoded image saving | Every 200 steps | `{save_dir}/decoded_images/` |
| Model checkpointing | Every `--checkpoint-steps` steps (default 1000) | `{save_dir}/vqvae_model.pt` |

**TensorBoard Metrics (per step):**
- `Loss/Total` — total training loss
- `Loss/Contrastive` — InfoNCE or MoCo contrastive loss
- `Loss/Recon` — reconstruction loss (BaselineLoss)
- `Loss/VQ` — VQ commitment loss
- `LR` — current learning rate

**Checkpoint Contents:**
- Model weights (`vqvae_model.pt` or `checkpoint.pt`)
- Optimizer state
- Current step and last loss values (total, contrastive, recon, VQ)
- If `--use-moco`: per-level queue tensors and queue pointers (explicit `moco_queues` / `moco_queue_ptrs` keys)

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

### 6.2 Implementation

The content/style split is configured via `--content-dim` and `--total-dim` (replacing the earlier hardcoded 256/256 values):

```bash
--content-dim 384   # first 384 dims are content
--total-dim   512   # dims 384–511 are style
```

```python
# Set in update_args() for ADNI/custom datasets:
content_indices = list(range(args.content_dim))           # [0, …, content_dim-1]
style_indices   = list(range(args.content_dim, args.total_dim))  # [content_dim, …, total_dim-1]

# Contrastive similarity computed ONLY on content
sim = cosine_similarity(z1[content_indices], z2[content_indices])
```

The `content_channels` in the VQ-VAE-2 is derived proportionally:
```python
content_channels = round(content_dim / total_dim * hidden_channels)
```
So with `--content-dim 384 --total-dim 512 --vqvae-hidden-channels 48`, `content_channels = round(0.75 × 48) = 36`.

### 6.3 Expected Outcomes

After training:
- `z[0:content_dim]` (content) should encode: brain structure, ventricle size, atrophy patterns
- `z[content_dim:total_dim]` (style) should encode: T1 vs T2 contrast, intensity characteristics

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
│   └── view_latents.ipynb      # Load checkpoint, extract & visualize latents
├── models/
│   ├── encoders.py             # Additional encoder architectures (text)
│   ├── pixelsnail.py           # PixelSNAIL autoregressive prior
│   ├── vae.py                  # VAE Encoder and Decoder architectures
│   └── vqvae.py                # VQ-VAE-2 + MoCoEncoder
├── training/
│   ├── losses.py               # InfoNCE, MoCo InfoNCE, and BaselineLoss
│   ├── main_multimodal.py      # train_step + main (VQ-VAE-2 / VAE)
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
    ├── {vae_model_id}/
    │   ├── settings.json           # Training configuration
    │   ├── Training.csv            # Loss history (CSV)
    │   ├── checkpoint.pt           # Full checkpoint (for resume)
    │   ├── encoder_image.pt        # Encoder weights only
    │   ├── tensorboard/            # TensorBoard event files
    │   └── decoded_images/
    │       ├── step_00001_original.nii.gz
    │       └── step_00001_decoded.nii.gz
    └── {vqvae_model_id}/
        ├── settings.json           # Training configuration
        ├── Training.csv            # Loss history (CSV)
        ├── vqvae_model.pt          # Full VQ-VAE-2 checkpoint (incl. MoCo queues if used)
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

**VAE (Baseline):**
```bash
python training/main_multimodal.py \
    --dataroot /data/natalia \
    --dataset_name ADNI_registered \
    --encoder-type vae \
    --train-steps 5000 \
    --batch-size 4 \
    --use-amp \
    --scale-recon-loss 0.00001
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
| `--encoder-type` | `vqvae` | Architecture: `vae` or `vqvae` |
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
| `--content-dim` | 128 | Content dimensions for `content_proj` split |
| `--total-dim` | 512 | Total dims (`content_dim + style_dim`) for `content_proj` |
| `--inject-style-to-decoder` | False | Feed style channels from encoder level-0 into the bottom decoder's final layer |

**MoCo Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-moco` | False | Enable MoCo contrastive training (replaces standard InfoNCE) |
| `--moco-queue-size` | 4096 | Number of negative keys stored per encoder level |
| `--moco-momentum` | 0.999 | EMA momentum for momentum encoder updates |

---

## 9. Current Training Status

### 9.1 VQ-VAE-2 Training Results

Based on recent VQ-VAE-2 training runs:

| Step | Total Loss | Contrastive | Recon + VQ |
|------|------------|-------------|------------|
| 1 | 9.64 | 2.76 | 6.89 |
| 25 | 7.11 | 0.86 | 6.26 |
| 50 | 7.16 | 1.77 | 5.39 |

**Observations:**
- VQ-VAE-2 achieves lower reconstruction loss faster (~5.4 vs ~50+ for VAE)
- Contrastive loss fluctuates due to Gumbel-Softmax exploration
- VQ commitment losses per level: L0≈1.5, L1≈0.3, L2≈0.2 (top level most stable)

### 9.2 VAE Training Results

Based on `results/ADNI_registered/2/Training.csv`:

| Step | Total Loss | Contrastive | Reconstruction |
|------|------------|-------------|----------------|
| 1 | 103.285 | 1.946 | 101.340 |
| 501 | 68.358 | 1.645 | 66.714 |
| 1001 | 53.411 | 1.515 | 51.896 |

**Observations:**
- VAE has higher reconstruction loss due to bottleneck through 512-dim vector
- Contrastive loss is stable and decreasing

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

---

## 11. Latent Visualization (`eval/view_latents.ipynb`)

The notebook extracts and visualizes latent representations from trained VQ-VAE-2 models:

1. **Load checkpoint** — handles `.blocks.` → `.stack.` key conversion for backward compatibility
2. **Load ADNI data** — supports both original resolution and 2mm downsampled modes
3. **Extract encoder features** — global average pooling of each hierarchical level
4. **Dimensionality reduction** — PCA and t-SNE on pooled features
5. **Visualization** — scatter plots colored by diagnostic label (AD, MCI, CN)
6. **Paired distance analysis** — compare T1 vs T2 latent distances per subject
7. **Diagnostic prediction** — 5-fold CV logistic regression and random forest per feature set
8. **Reconstruction visualization** — mid-sagittal slice comparison of originals vs reconstructed

**Import structure (updated 25 Feb 2026):**

The notebook lives in `eval/` but imports from the project root packages. The first cell now adds the project root to `sys.path` and uses fully-qualified module paths:

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

1. **MoCo Tuning:**
   - Experiment with different queue sizes (256 → 8192) and momentum values (0.99 → 0.9999)
   - Implement a warm-up phase for the queue to reduce cold-start noise
   - Compare MoCo vs standard InfoNCE on downstream classification tasks

2. **VQ-VAE-2 Improvements:**
   - Increase codebook size for more expressivity
   - Add PixelSNAIL prior for generative sampling
   - Experiment with different scaling rates

3. **Evaluation Metrics:** Implement:
   - DCI (Disentanglement, Completeness, Informativeness) scores
   - Downstream classification (e.g., Alzheimer's vs. healthy)
   - Cross-view reconstruction (encode T1, decode T2)

4. **Content/Style Analysis:**
   - Visualize learned content vs style dimensions
   - Perform style transfer experiments (apply T1 style to T2 content)

5. **Hyperparameter Tuning:**
   - VQ commitment weight optimization
   - Content/style ratio exploration
   - Temperature scheduling for contrastive loss

6. **Memory Optimization (available now):**
   - Gradient checkpointing (`--gradient-checkpointing`)
   - Gradient accumulation (`--gradient-accumulation-steps N`)
   - Skip reconstruction (`--skip-recon-ratio R`)
   - MoCo removes the need for large batches, directly reducing GPU memory pressure

---

## 13. Summary

This implementation provides a framework for learning disentangled representations from paired multimodal brain MRI data using two architectures:

### VAE Mode:
- **Shared 3D CNN encoder** (15.5M params) + separate decoder (27.8M params)
- **512-dimensional latent space** split into 256 content + 256 style dimensions
- **InfoNCE contrastive loss** applied only to content dimensions

### VQ-VAE-2 Mode (Recommended):
- **Hierarchical 3D VQ-VAE-2** with 3 levels (~2.9M params total)
- **Discrete codebook representations** (384 codes × 32 dims per level)
- **Hierarchical contrastive loss** at all encoder levels — either standard InfoNCE or MoCo
- **MoCo option** (`--use-moco`): momentum encoder + per-level queues (4096 negatives per level) for effective contrastive learning with small batches
- **Content/style split** controlled via `--content-dim` / `--total-dim` (no longer hardcoded)
- **Style injection** (`--inject-style-to-decoder`): style channels fed back into the bottom decoder for better modality-specific reconstruction
- **Content selection at Level 0** propagated proportionally to higher levels
- **EMA codebook updates** for stable training
- **Auto-resume** with architecture compatibility check — safe to rerun after changing architecture flags

Both models learn to encode shared anatomical information in content dimensions while allowing style dimensions to capture modality-specific (T1 vs T2) characteristics.
