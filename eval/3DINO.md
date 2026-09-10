# Full-volume 3DINO for synthetic MRI

Use `--backbone 3dino` in the existing DINO embedding and paired-view InfoNCE
fine-tuning commands. The adapter imports the unmodified
[AICONSlab/3DINO implementation](https://github.com/AICONSlab/3DINO). The default
`--backbone dinov3` retains the existing 2D slice pipeline.

## Setup

In the existing training environment (PyTorch, MONAI, NumPy, nibabel, etc.):

```bash
git clone https://github.com/AICONSlab/3DINO.git ../3DINO
git -C ../3DINO checkout 85bd4435c1b2ada41cd34cd15cad17c4d3c88d89
python -m pip install omegaconf
```

This is the upstream revision inspected and tested for this integration. The
adapter records the actual checkout revision and checkpoint path in its outputs.
It imports only the model and configuration modules, so the upstream repository's
entire pretraining environment is not required. xFormers is optional for our
single-tensor forward path. Set `XFORMERS_DISABLED=1` before launching Python to
use upstream's ordinary attention fallback if xFormers is incompatible with your
device or PyTorch installation.

Obtain the pretrained checkpoint from
[AICONSlab/3DINO-ViT](https://huggingface.co/AICONSlab/3DINO-ViT), which requires
requesting access/accepting the authors' conditions on Hugging Face. Pass its
local path with `--three-dino-weights`. There is no automatic weight download or
silent random initialization. `--random-init --model-seed 0` explicitly requests
an untrained architecture baseline instead.

The upstream code and weights have their own CC BY-NC-ND 4.0 terms; consult the
[authors' license and usage conditions](https://github.com/AICONSlab/3DINO#license).
This repository's adapter does not redistribute their source or pretrained weights.

## Extract pretrained embeddings

```bash
python -m eval.dinov3_embed_synthetic \
  --backbone 3dino --three-dino-repo ../3DINO \
  --three-dino-weights /path/to/downloaded_checkpoint.pth \
  --run-dir results/synthetic/YOUR_RUN \
  --out results/3dino/pretrained.npz \
  --num-samples 500 --volume-batch 2 --device cuda

python -m eval.dinov3_identifiability \
  --embeddings results/3dino/pretrained.npz
```

The existing identifiability and causal recovery scorer accepts the NPZ directly.
Both modalities, latent labels, causal adjacency, and optional voxel baseline
retain the existing schema and dataset order.

## Fine-tune on paired synthetic modalities

```bash
python -m training.finetune_dino \
  --backbone 3dino --three-dino-repo ../3DINO \
  --three-dino-weights /path/to/downloaded_checkpoint.pth \
  --run-dir results/synthetic/YOUR_RUN \
  --output-dir results/3dino_finetuned \
  --num-samples 1000 --epochs 20 --batch-size 2 \
  --device cuda --dtype bfloat16 --gradient-checkpointing
```

This optimizes the existing symmetric cross-modality InfoNCE objective using a
shared, trainable 3DINO backbone and the existing optional projector. It is
paired-view fine-tuning, not a reimplementation of upstream's DINO/iBOT
self-distillation pretraining. Choose the largest subject batch your GPU supports;
the other subjects supply contrastive negatives. `--batch-size` controls subjects,
and `--plane-batch-size` applies only to the 2D backend. Use `--dtype float32` on
CPU, or `float16` when a CUDA device does not support bfloat16.

Checkpoints are saved under `encoder/model.pt` with architecture/provenance in
`encoder/config.json`. The existing `training_state.pt`, `preprocessing.json`,
`settings.json`, and training metrics are also written. As in the current 2D
trainer, the latest checkpoint is replaced after each epoch; historical epochs
are not automatically archived, and optimizer resume is not implemented.

Extract held-out embeddings with the saved preprocessing and generator settings:

```bash
python -m eval.dinov3_embed_synthetic \
  --backbone 3dino --three-dino-repo ../3DINO \
  --three-dino-weights results/3dino_finetuned/encoder \
  --run-dir results/3dino_finetuned \
  --preprocessing results/3dino_finetuned/preprocessing.json \
  --out results/3dino_finetuned/test_embeddings.npz \
  --num-samples 500 --volume-batch 2 --device cuda
```

Training uses the generator's train split; extraction uses its test split. Saved
dataset window bounds and fixed-reference generator normalization are restored
for evaluation. Encoder weights load strictly: missing or incompatible backbone
parameters fail rather than producing measurements on partially initialized weights.
Official `teacher` checkpoints with `module.backbone.` prefixes and extra
pretraining heads are supported, as are bare backbone state dictionaries.

## What enters the model

The authors' [basic-use notebook](https://github.com/AICONSlab/3DINO/blob/main/notebooks/basic_model_use.ipynb)
uses single-channel volumes, 112³ input, and intensities in [-1, 1]. Our defaults:

1. Accept genuine `(B, 1, X, Y, Z)` volumes; 2D inputs are rejected.
2. Compute each volume's 0.05th and 99.95th intensity percentiles and map/clamp
   them to [-1, 1]. Constant volumes map to zero.
3. Resize the complete volume to 112³ by trilinear interpolation, retaining the
   generator's axis order. There is no slice selection, RGB replication, center
   cropping, 8-bit quantization, or ImageNet normalization.
4. Feed the volume through the official ViT-Large 3D backbone, with 16³ voxel
   patches (343 patch tokens at 112³).

`--volume-size` can change input size to another positive multiple of 16; the
upstream model interpolates its learned positional embeddings. The pretrained
position-embedding parameter shape is built from the original 112³ configuration,
independently of inference size. Cubic resizing is appropriate for the cubic
synthetic data here; this adapter does not perform physical-spacing/orientation
standardization for clinical scans.

Token pooling supports `cls` (default, 1024 features, matching the authors' basic
example), `mean` (1024), `cls_mean` (2048), and `grid` (3D adaptive patch-grid
pooling; `--grid-size 2` produces 8192 features). All modes operate on the full
volume. Slice flags `--axes`, `--slices`, `--slice-agg`, and `--image-size` are
specific to the 2D backend and do not affect 3DINO. Metadata explicitly records
`backbone: 3dino`, `slots: [volume]`, volume size, pooling and normalization.

Per-volume percentile normalization removes affine gain/bias differences before
encoding, so it limits recovery of those style factors. For that experiment use
`--window dataset`, which estimates one fixed window from training subjects and
applies it to every volume. This is an explicit departure from the per-volume
normalization in the authors' example. Always reuse `--preprocessing` for
fine-tuned evaluation and compare models with matched data/window/pooling choices.

## Validation

```bash
THREE_DINO_REPO=../3DINO XFORMERS_DISABLED=1 \
  python -m unittest discover -s tests -p test_three_dino.py -v
```

The integration tests instantiate the actual upstream 3D transformer at reduced
width/depth for CPU execution. They check 3D token layout, pooling, preprocessing,
backbone gradients with activation checkpointing, strict teacher/export loading,
train/eval feature agreement, and a complete synthetic training → export →
held-out extraction pass. They do not download the gated pretrained weights or
establish pretrained ViT-Large performance or GPU memory requirements.
