#!/usr/bin/env bash
# Synthetic-data baseline. Drops the ADNI-specific bits (dataroot, labels,
# cache, brain-mask threshold tuning) and uses the in-memory pseudo-MRI
# generator instead. Good for sanity-checking model changes end-to-end
# without needing the cluster filesystem.
#
# Knobs vs. ablation_baseline.sh:
#   - --dataset_name synthetic            (replaces ADNI_stripped_masks)
#   - --synthetic-mode pseudo_mri         (continuous content/style; brain-like)
#   - --synthetic-res 32                  (32^3 — divisible by 8, fits VQ-VAE)
#   - --spatial-size auto-set from res    (no manual override needed)
#   - no --labels-path, --cache-dir, --select-by-gated-score
#     (synthetic has no diagnosis labels — selection by val loss instead)

set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
STEPS=${STEPS:-5000}
RES=${RES:-32}
TAG=${TAG:-synthetic-baseline}

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
