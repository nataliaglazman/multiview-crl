#!/usr/bin/env bash
# Synthetic-data baseline #2: pseudo-MRI mode.
# Continuous content/style with brain-like appearance. Run AFTER primitives —
# tests whether the model's conv inductive biases hold up on image-shaped data
# that resembles real T1/FLAIR. Smaller GT factor count than primitives
# (5 content + 3 style) so use tighter --content-dim / --total-dim.
#
# Differences from run_synthetic.sh (primitives variant):
#   - --synthetic-mode pseudo_mri
#   - --content-style-levels 0          (content here is global, not hierarchical)
#   - --content-dim 16 / --total-dim 32 (only 5 GT content dims — don't oversize)
#   - exposes --synthetic-n-content / --synthetic-n-style (pseudo_mri-only)

set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
STEPS=${STEPS:-5000}
RES=${RES:-32}
TAG=${TAG:-synthetic-pseudo-mri}

PYTHONPATH=${REPO} \
python ${REPO}/training/main_multimodal.py \
    --dataset_name synthetic \
    --synthetic-mode pseudo_mri \
    --synthetic-res ${RES} \
    --synthetic-num-train 2000 \
    --synthetic-num-val 200 \
    --synthetic-num-test 400 \
    --synthetic-seed 42 \
    --synthetic-n-content 5 \
    --synthetic-n-style 3 \
    --model-id "${TAG}" \
    --encoder-type vqvae \
    --vqvae-nb-levels 3 \
    --vqvae-hidden-channels 32 \
    --vqvae-embed-dim 32 \
    --vqvae-nb-entries 256 \
    --vqvae-scaling-rates 2 2 2 \
    --vq-commitment-weight 0.25 \
    --content-style-levels 0 \
    --content-dim 16 \
    --total-dim 32 \
    --mask-mode fixed \
    --separate-encoders \
    --pass-full-to-next-level \
    --contrastive-loss-type infonce \
    --cross-view-negs-only \
    --scale-contrastive-loss 1.0 \
    --scale-style-contrastive-loss 0.0 \
    --scale-content-modality-adv 0.0 \
    --scale-style-modality-ce 0.0 \
    --scale-recon-loss 1.0 \
    --scale-adv-loss 0.0 \
    --moco-queue-size 0 \
    --tau 0.1 \
    --lr 1e-3 \
    --batch-size 16 \
    --train-steps ${STEPS} \
    --use-amp \
    --workers 4
