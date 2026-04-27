#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl \
python /nfs/home/nglazman/crl-2/multiview-crl/training/main_multimodal.py \
    --dataroot /nfs/home/nglazman/data \
    --dataset_name ADNI_stripped \
    --encoder-type vqvae \
    --model-id multiview-03-content-lr-001-patches-change-weighting-fix-resuming-FIX \
    --use-moco \
    --patch-contrastive --patch-grid 4 5 4 \
    --scale-recon-loss 1 \
    --labels-path /nfs/home/nglazman/nmpevqvae/labels_cleaned_3class.csv \
    --moco-queue-size 8192 \
    --moco-momentum 0.99 \
    --mask-mode fixed \
    --vqvae-nb-levels 3 \
    --inject-style-to-decoder \
    --style-injection-mode film \
    --lr 0.001 \
    --content-dim 128 \
    --total-dim 512 \
    --vqvae-scaling-rates 2 2 2 \
    --train-steps 50000 \
    --vqvae-hidden-channels 64 \
    --vqvae-embed-dim 38 \
    --batch-size 2 \
    --use-amp \
    --resume-training \
    --workers 8 \
    --recon-loss-start-step 2000 \
    --tau 0.07 \
    --scale-contrastive-loss 1.0 \
    --content-style-levels 0 \
    --cross-view-negs-only \
    --content-ratios 0.3 \
    --separate-encoders \
    --gradient-accumulation-steps 4 \
    --gradient-checkpointing \
    --skip-recon-ratio 0.5 \
    --image-spacing 1.0 \
    --scale-recon-loss 1.0 \
    --crop-margin 12 \
    --cache-dataset \
    --vq-commitment-weight 0.25 \
    --cache-dir /nfs/home/nglazman/cache/multiview \
    --contrastive-level-weights 3.0 0.5 0.5

