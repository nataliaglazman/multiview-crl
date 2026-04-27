#!/usr/bin/env bash
# Clean single-experiment baseline. Everything optional is OFF.
# This is the reference run; add ONE component at a time and compare.
#
# Components intentionally disabled here (add back one-at-a-time):
#   - style InfoNCE          (--scale-style-contrastive-loss 0)
#   - content GRL head       (--scale-content-modality-adv 0)
#   - style-CE head          (--scale-style-modality-ce 0)
#   - GAN                    (--scale-adv-loss 0, no --use-gan)
#   - style quantization     (no --quantize-style)
#   - patch InfoNCE          (no --patch-contrastive)
#   - multi-level masking    (--content-style-levels 0 only)
#   - MoCo queue             (--moco-queue-size 0)
#   - asymmetric aug         (no --asymmetric-aug)
#   - learned mask           (--mask-mode fixed)
#
# What's on: separate encoders + cross-view-negs InfoNCE at L0 + recon.
# Best checkpoint selected by separation_score_gated so a collapsed encoder
# cannot win. Requires labels in the val loader (ADNI "Group" column).

set -euo pipefail

REPO=${REPO:-/nfs/home/nglazman/crl-2/multiview-crl}
DATAROOT=${DATAROOT:-/nfs/home/nglazman/data}
LABELS=${LABELS:-/nfs/home/nglazman/nmpevqvae/labels_cleaned_3class.csv}
CACHE=${CACHE:-/nfs/home/nglazman/cache/multiview}
STEPS=${STEPS:-90000}
TAG=${TAG:-ablation-baseline-levels-lr-4}

PYTHONPATH=${REPO} \
python ${REPO}/training/main_multimodal.py \
    --dataroot ${DATAROOT} \
    --dataset_name ADNI_stripped_masks \
    --labels-path ${LABELS} \
    --cache-dataset \
    --cb-reset-threshold 5 \
    --cache-dir ${CACHE} \
    --model-id "${TAG}" \
    --encoder-type vqvae \
    --vqvae-nb-levels 3 \
    --vqvae-hidden-channels 32 \
    --vqvae-embed-dim 32 \
    --vqvae-nb-entries 256 \
    --vqvae-scaling-rates 2 2 2 \
    --vq-commitment-weight 0.25 \
    --content-style-levels 0 1 2 \
    --content-ratios 0.5 0.5 0.5 \
    --content-dim 128 \
    --total-dim 512 \
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
    --lr 0.006 \
    --batch-size 4 \
    --train-steps ${STEPS} \
    --spatial-size 150 180 150 \
    --image-spacing 1.0 \
    --use-amp \
    --workers 8 \
    --gradient-checkpointing \
    --select-by-gated-score \
    --separation-floor-diagnosis-info 0.1 \
    --quantize-style \
    --inject-style-to-decoder \
    --resume-training

