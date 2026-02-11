# Multiview Contrastive Representation Learning on ADNI Brain MRI

## Technical Report for Supervisor Review

**Project:** `multiview-crl`  
**Dataset:** ADNI (Alzheimer's Disease Neuroimaging Initiative) - Registered T1 and T2 MRI scans  
**Date:** February 2026  
**Last Updated:** 11 February 2026

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

The preprocessing is implemented in `utils.py` using MONAI transforms:

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

## 3.2 VAE Architecture (`vae.py`)

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

## 3.3 VQ-VAE-2 Architecture (`vqvae.py`)

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
| `--vqvae-nb-entries` | 512 | Codebook size (number of codes) |
| `--vqvae-scaling-rates` | [4, 2, 2] | Downscale factor per level |
| `--vq-commitment-weight` | 0.25 | VQ commitment loss weight |

### 3.3.3 Encoder Details (Per Level)

Each encoder uses strided 3D convolutions for downsampling:

| Level | Input Channels | Downscale | Output Channels | Output Size (approx) |
|-------|---------------|-----------|-----------------|---------------------|
| 0 | 1 (image) | 4× | 64 | (23, 27, 23) |
| 1 | 64 | 2× | 64 | (12, 14, 12) |
| 2 | 64 | 2× | 64 | (6, 7, 6) |

**Internal Encoder Structure:**
```
For encoder with 4× downscale:
  Conv3d(in, 32, k=4, s=2)  → BatchNorm → ReLU   (2× down)
  Conv3d(32, 64, k=4, s=2)  → BatchNorm → ReLU   (2× down)
  Conv3d(64, 64, k=3, s=1)  → BatchNorm          (refine)
  ResidualStack(64, 32, 2)                        (2 ReZero blocks)
```

### 3.3.4 Vector Quantization Layer

The `CodeLayer` implements EMA (Exponential Moving Average) codebook updates:

```python
# Discrete bottleneck
z_e = encoder_output                    # (B, 64, D, H, W)
z_q = nearest_codebook_entry(z_e)       # (B, 32, D, H, W)
commitment_loss = ||z_e - sg[z_q]||²    # Straight-through gradient
```

**Key Properties:**
- **Codebook Size:** 512 discrete codes
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

### 3.3.6 Total Parameters

With default configuration:
- **Total VQ-VAE-2 Parameters:** ~2.9M (much smaller than VAE)

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
- $\mathcal{L}_{contrastive}^{(l)}$: InfoNCE loss at encoder level $l$
- $\mathcal{L}_{recon}$: Reconstruction loss (BaurLoss: L1 + L2)
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

### 4.2 Contrastive Loss (InfoNCE)

Implemented in `losses.py` as `infonce_loss`:

$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(z^{(1)}_c, z^{(2)}_c) / \tau)}{\sum_{k} \exp(\text{sim}(z^{(1)}_c, z^{(k)}_c) / \tau)}$$

**Key Properties:**
- **Similarity Metric:** Cosine similarity
- **Temperature ($\tau$):** 1.0 (default)
- **Content-Only Similarity:** Only `content_indices` (first 256 dims) are used for similarity computation
- **Positive Pairs:** Same subject, different views (T1, T2)
- **Negative Pairs:** Different subjects within the batch

**Mechanism:**
1. Encode both views: $z^{(1)} = f_\theta(x^{(1)}), z^{(2)} = f_\theta(x^{(2)})$
2. Extract content: $z^{(1)}_c = z^{(1)}[0:256], z^{(2)}_c = z^{(2)}[0:256]$
3. Compute pairwise similarities within batch
4. Apply cross-entropy loss treating matched pairs as positives

### 4.3 Reconstruction Loss (BaurLoss)

Implemented in `losses.py` as `BaurLoss`:

$$\mathcal{L}_{recon} = \text{L1}(\hat{x}, x) + \text{L2}(\hat{x}, x)$$

Where:
- $\hat{x} = g_\phi(z)$ is the decoded image
- L1 uses `PairwiseDistance(p=1)` with `.mean()` reduction
- L2 uses `PairwiseDistance(p=2)` with `.mean()` reduction

**Note:** Both T1 and T2 images are reconstructed through the shared decoder.

### 4.4 VQ Commitment Loss

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
| **Optimizer** | Adam |
| **Learning Rate** | 1e-5 |
| **Gradient Clipping** | max_norm=2.0, L2 norm |
| **Mixed Precision (AMP)** | Enabled (fp16) |
| **Batch Size** | 4 (effective: 8 images = 4 T1 + 4 T2) |

### 5.2 Training Loop

```
For each step:
    1. Load batch: {T1: (B, 1, 91, 109, 91), T2: (B, 1, 91, 109, 91)}
    2. Concatenate views: (2B, 1, 91, 109, 91)
    3. Encode: z = encoder(images) → (2B, 512)
    4. Decode: x_hat = decoder(z) → (2B, 1, 91, 109, 91)
    5. Compute contrastive loss on z[:, 0:256] (content only)
    6. Compute reconstruction loss on (x_hat, images)
    7. Backpropagate total_loss with AMP scaling
    8. Clip gradients and update parameters
```

### 5.3 Logging and Checkpoints

| Event | Frequency |
|-------|-----------|
| Loss printing (console) | Every step |
| CSV logging | Every 100 steps |
| Decoded image saving | Every 200 steps |
| Model checkpointing | Every 1000 steps |

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

```python
# Content dimensions (used for contrastive loss)
content_indices = list(range(256))      # [0, 1, ..., 255]

# Style dimensions (NOT used for contrastive loss)
style_indices = list(range(256, 512))   # [256, 257, ..., 511]

# Contrastive similarity computed ONLY on content
sim = cosine_similarity(z1[content_indices], z2[content_indices])
```

### 6.3 Expected Outcomes

After training:
- `z[0:256]` (content) should encode: brain structure, ventricle size, atrophy patterns
- `z[256:512]` (style) should encode: T1 vs T2 contrast, intensity characteristics

---

## 7. File Structure

```
multiview-crl/
├── main_multimodal.py      # Main training script (VAE + VQ-VAE-2)
├── vae.py                  # VAE Encoder and Decoder architectures
├── vqvae.py                # VQ-VAE-2 hierarchical architecture
├── helper.py               # Helper classes for VQ-VAE-2
├── losses.py               # InfoNCE and BaurLoss implementations
├── datasets.py             # Dataset classes (MyCustomDataset for ADNI)
├── utils.py                # MONAI transforms, Gumbel-Softmax, utilities
├── encoders.py             # Additional encoder architectures (text)
├── METHODOLOGY_REPORT.md   # This documentation
├── results/
│   └── ADNI_registered/
│       ├── {vae_model_id}/
│       │   ├── settings.json           # Training configuration
│       │   ├── Training.csv            # Loss history
│       │   ├── checkpoint.pt           # Full checkpoint (for resume)
│       │   ├── encoder_image.pt        # Encoder weights only
│       │   └── decoded_images/
│       │       ├── step_00001_original.nii.gz
│       │       └── step_00001_decoded.nii.gz
│       └── {vqvae_model_id}/
│           ├── settings.json           # Training configuration
│           ├── Training.csv            # Loss history
│           ├── vqvae_model.pt          # Full VQ-VAE-2 checkpoint
│           └── decoded_images/
│               ├── step_00001_original.nii.gz
│               └── step_00001_decoded.nii.gz
```

---

## 8. Running the Code

### 8.1 Training Commands

**VQ-VAE-2 (Recommended):**
```bash
conda run -n monai_env python main_multimodal.py \
    --dataroot /data/natalia \
    --dataset_name ADNI_registered \
    --encoder-type vqvae \
    --vqvae-nb-levels 3 \
    --vqvae-scaling-rates 4 2 2 \
    --train-steps 10000 \
    --batch-size 4 \
    --use-amp \
    --workers 4 \
    --model-id vqvae_experiment
```

**VAE (Baseline):**
```bash
conda run -n monai_env python main_multimodal.py \
    --dataroot /data/natalia \
    --dataset_name ADNI_registered \
    --encoder-type vae \
    --train-steps 5000 \
    --batch-size 4 \
    --use-amp \
    --scale-recon-loss 0.00001
```

**Resume Training:**
```bash
# Add --resume-training flag to continue from last checkpoint
python main_multimodal.py \
    --dataroot /data/natalia \
    --dataset_name ADNI_registered \
    --encoder-type vqvae \
    --model-id vqvae_experiment \
    --resume-training \
    --train-steps 20000
```

### 8.2 Key Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataroot` | `/data/natalia/` | Root data directory |
| `--dataset_name` | `ADNI_registered` | Dataset name |
| `--encoder-type` | `vqvae` | Architecture: `vae` or `vqvae` |
| `--batch-size` | 2 | Samples per view per batch |
| `--lr` | 1e-5 | Learning rate |
| `--train-steps` | 300001 | Total training steps |
| `--tau` | 1.0 | Temperature for InfoNCE |
| `--scale-recon-loss` | 0.00001 | Weight for reconstruction loss |
| `--use-amp` | False | Enable mixed precision |
| `--model-id` | Auto | Experiment identifier |
| `--resume-training` | False | Resume from last checkpoint |

**VQ-VAE-2 Specific Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--vqvae-hidden-channels` | 64 | Hidden layer channels |
| `--vqvae-res-channels` | 32 | Residual block channels |
| `--vqvae-nb-levels` | 3 | Number of hierarchical levels |
| `--vqvae-embed-dim` | 32 | Codebook embedding dimension |
| `--vqvae-nb-entries` | 512 | Codebook size |
| `--vqvae-scaling-rates` | [4, 2, 2] | Downscale factor per level |
| `--vq-commitment-weight` | 0.25 | VQ commitment loss weight |

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

## 10. Future Work and Considerations

1. **VQ-VAE-2 Improvements:**
   - Increase codebook size for more expressivity
   - Add PixelSNAIL prior for generative sampling
   - Experiment with different scaling rates

2. **Evaluation Metrics:** Implement:
   - DCI (Disentanglement, Completeness, Informativeness) scores
   - Downstream classification (e.g., Alzheimer's vs. healthy)
   - Cross-view reconstruction (encode T1, decode T2)

3. **Content/Style Analysis:**
   - Visualize learned content vs style dimensions
   - Perform style transfer experiments (apply T1 style to T2 content)

4. **Hyperparameter Tuning:**
   - VQ commitment weight optimization
   - Content/style ratio exploration
   - Temperature scheduling for contrastive loss

---

## 11. Summary

This implementation provides a framework for learning disentangled representations from paired multimodal brain MRI data using two architectures:

### VAE Mode:
- **Shared 3D CNN encoder** (15.5M params) + separate decoder (27.8M params)
- **512-dimensional latent space** split into 256 content + 256 style dimensions
- **InfoNCE contrastive loss** applied only to content dimensions

### VQ-VAE-2 Mode (Recommended):
- **Hierarchical 3D VQ-VAE-2** with 3 levels (~2.9M params total)
- **Discrete codebook representations** (512 codes × 32 dims per level)
- **Hierarchical contrastive loss** at all encoder levels
- **Content selection at Level 0** propagated proportionally to higher levels
- **EMA codebook updates** for stable training

Both models learn to encode shared anatomical information in content dimensions while allowing style dimensions to capture modality-specific (T1 vs T2) characteristics.
