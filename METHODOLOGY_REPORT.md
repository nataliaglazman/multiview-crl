# Multiview Contrastive Representation Learning on ADNI Brain MRI

## Technical Report for Supervisor Review

**Project:** `multiview-crl`  
**Dataset:** ADNI (Alzheimer's Disease Neuroimaging Initiative) - Registered T1 and T2 MRI scans  
**Date:** February 2026

---

## 1. Overview

This project implements **Multiview Contrastive Representation Learning** for 3D brain MRI data, using paired T1-weighted and T2-weighted scans as two complementary views of the same subject. The goal is to learn disentangled representations that separate **content** (shared anatomical information) from **style** (modality-specific contrast differences).

### 1.1 Problem Formulation

Given paired observations $(x^{(1)}, x^{(2)})$ representing T1 and T2 MRI scans of the same subject, we aim to learn an encoder $f_\theta$ that maps each observation to a latent representation $z = f_\theta(x)$, where:

- **Content dimensions** $z_c$: Capture shared information (brain anatomy) that is consistent across both views
- **Style dimensions** $z_s$: Capture view-specific information (T1 vs T2 contrast characteristics)

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

## 3. Model Architecture

### 3.1 Encoder (`vae.py` - `Encoder` class)

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

### 3.2 Decoder (`vae.py` - `Decoder` class)

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

### 3.3 Latent Space Structure

| Property | Value | Description |
|----------|-------|-------------|
| **Total Latent Dimensions** | 512 | Fixed encoder output |
| **Content Dimensions** | 256 (indices 0-255) | Shared between T1/T2 |
| **Style Dimensions** | 256 (indices 256-511) | View-specific |

---

## 4. Loss Functions

### 4.1 Total Loss

$$\mathcal{L}_{total} = \mathcal{L}_{contrastive} + \lambda \cdot \mathcal{L}_{reconstruction}$$

Where $\lambda$ = `scale_recon_loss` (default: 0.00001)

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
├── main_multimodal.py      # Main training script
├── vae.py                  # Encoder and Decoder architectures
├── losses.py               # InfoNCE and BaurLoss implementations
├── datasets.py             # Dataset classes (MyCustomDataset for ADNI)
├── utils.py                # MONAI transforms, brain masking, utilities
├── encoders.py             # Additional encoder architectures (text)
├── results/
│   └── ADNI_registered/
│       └── {model_id}/
│           ├── settings.json           # Training configuration
│           ├── Training.csv            # Loss history
│           ├── encoder_image.pt        # Saved encoder weights
│           └── decoded_images/
│               ├── step_00001_original.nii.gz
│               ├── step_00001_decoded.nii.gz
│               └── ...
```

---

## 8. Running the Code

### 8.1 Training Command

```bash
conda run -n monai_env python main_multimodal.py \
    --dataroot /data/natalia \
    --dataset_name ADNI_registered \
    --train-steps 5000 \
    --batch-size 4 \
    --use-amp \
    --scale-recon-loss 0.00001
```

### 8.2 Key Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataroot` | `/data/natalia/` | Root data directory |
| `--dataset_name` | `ADNI_registered` | Dataset name |
| `--batch-size` | 2 | Samples per view per batch |
| `--lr` | 1e-5 | Learning rate |
| `--train-steps` | 300001 | Total training steps |
| `--tau` | 1.0 | Temperature for InfoNCE |
| `--scale-recon-loss` | 0.00001 | Weight for reconstruction loss |
| `--use-amp` | False | Enable mixed precision |
| `--model-id` | Auto | Experiment identifier |

---

## 9. Current Training Status

Based on `results/ADNI_registered/2/Training.csv`:

| Step | Total Loss | Contrastive | Reconstruction |
|------|------------|-------------|----------------|
| 1 | 103.285 | 1.946 | 101.340 |
| 501 | 68.358 | 1.645 | 66.714 |
| 1001 | 53.411 | 1.515 | 51.896 |

**Observations:**
- Contrastive loss is decreasing (1.946 → 1.515), indicating the model is learning to distinguish subjects
- Reconstruction loss is decreasing (101.340 → 51.896), indicating improved reconstruction quality
- The reconstruction loss dominates due to the BaurLoss implementation using pairwise distances

---

## 10. Future Work and Considerations

1. **Reconstruction Loss Scaling:** The current `scale_recon_loss=0.00001` may be too small; consider increasing to improve reconstruction quality

2. **Evaluation Metrics:** Implement:
   - DCI (Disentanglement, Completeness, Informativeness) scores
   - Downstream classification (e.g., Alzheimer's vs. healthy)
   - Cross-view reconstruction (encode T1, decode T2)

3. **Style Supervision:** Consider adding a style classification loss to encourage view-specific encoding

4. **Decoder Activation:** Currently no final activation; may need to add activation matching the input normalization range

---

## 11. Summary

This implementation provides a framework for learning disentangled representations from paired multimodal brain MRI data using:

- **Shared 3D CNN encoder** processing both T1 and T2 modalities
- **512-dimensional latent space** split into 256 content + 256 style dimensions
- **InfoNCE contrastive loss** applied only to content dimensions
- **Reconstruction loss** ensuring the latent code preserves image information
- **MONAI preprocessing** with 2mm resampling and brain masking

The model learns to encode shared anatomical information in the content dimensions while allowing style dimensions to capture modality-specific characteristics.
