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

REPO=${REPO:-"/nfs/home/nglazman/crl-2/multiview-crl"}
STEPS=${STEPS:-200000}
RES=${RES:-64}
TAG=${TAG:-synthetic-baseline-psuedo-mri-patch-batch-dropout-correction-style-cont-causal-correct-norm}

PYTHONPATH=${REPO} \
python ${REPO}/training/main_multimodal.py \
    --dataset_name synthetic \
    --synthetic-mode pseudo_mri \
    --patch-contrastive --patch-grid-per-level 4 4 4 4 4 4 4 4 4 \
    --synthetic-causal \
    --synthetic-causal-graph chain \
    --synthetic-res ${RES} \
    --synthetic-num-train 2000 \
    --quantize-style \
    --synthetic-num-val 200 \
    --synthetic-num-test 400 \
    --synthetic-seed 42 \
    --synthetic-n-content 9 \
    --synthetic-n-style 3 \
    --model-id "${TAG}" \
    --encoder-type vqvae \
    --vqvae-nb-levels 3 \
    --vqvae-hidden-channels 32 \
    --vqvae-embed-dim 32 \
    --vqvae-nb-entries 128 64 32 \
    --vqvae-scaling-rates 2 2 2 \
    --vq-commitment-weight 0.25 \
    --content-style-levels 0 1 2 \
    --content-dim 16 \
    --total-dim 32 \
    --mask-mode fixed \
    --separate-encoders \
    --pass-full-to-next-level \
    --contrastive-loss-type infonce \
    --cross-view-negs-only \
    --scale-contrastive-loss 10.0 \
    --scale-style-contrastive-loss 2.0 \
    --scale-content-modality-adv 0.0 \
    --scale-style-modality-ce 0.0 \
    --scale-recon-loss 1.0 \
    --scale-adv-loss 0.0 \
    --moco-queue-size 0 \
    --tau 0.2 \
    --resume-training \
    --lr 1e-3 \
    --batch-size 32 \
    --train-steps ${STEPS} \
    --content-ratios 0.95 0.95 0.95 \
    --inject-style-to-decoder \
    --recon-loss-start-step 0 \
    --workers 4 \
    --style-dropout-prob 0.2 \
