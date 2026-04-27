#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl \
python /nfs/home/nglazman/crl-2/multiview-crl/training/main_multimodal.py \
    --dataroot /nfs/home/nglazman/data \
    --dataset_name ADNI_stripped \
    --encoder-type vqvae \
    --model-id multiview-05-content-lr-002-all-levels-corrected-learned \
    --scale-recon-loss 1 \
    --patch-contrastive --patch-grid 4 5 4 \
    --labels-path /nfs/home/nglazman/nmpevqvae/labels_cleaned_3class.csv \
    --moco-queue-size 8192 \
    --moco-momentum 0.99 \
    --mask-mode learned \
    --vqvae-nb-levels 3 \
    --lr 0.002 \
    --content-dim 128 \
    --total-dim 512 \
    --vqvae-scaling-rates 2 2 2 \
    --train-steps 70000 \
    --vqvae-hidden-channels 38 \
    --vqvae-embed-dim 38 \
    --batch-size 3 \
    --spatial-size 150 180 150 \
    --use-amp \
    --resume-training \
    --workers 8 \
    --recon-loss-start-step 2000 \
    --tau 0.07 \
    --scale-contrastive-loss 10.0 \
    --content-style-levels 0 1 2 \
    --cross-view-negs-only \
    --content-ratios 0.5 0.5 0.5 \
    --separate-encoders \
    --gradient-accumulation-steps 4 \
    --gradient-checkpointing \
    --skip-recon-ratio 0.5 \
    --image-spacing 1.0 \
    --cache-dataset \
    --vq-commitment-weight 0.25 \
    --cache-dir /nfs/home/nglazman/cache/multiview \
    --contrastive-level-weights 1.0 1.0 1.0 \
    --pass-full-to-next-level \

