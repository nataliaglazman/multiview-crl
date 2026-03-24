#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl \
python /nfs/home/nglazman/crl-2/multiview-crl/training/main_multimodal.py \
    --dataroot /nfs/home/nglazman \
    --dataset_name ADNI_registered \
    --encoder-type vqvae \
    --use-moco \
    --moco-queue-size 4096 \
    --moco-momentum 0.999 \
    --vqvae-nb-levels 3 \
    --lr 0.0003 \
    --content-dim 384 \
    --total-dim 512 \
    --vqvae-scaling-rates 2 2 2 \
    --train-steps 50000 \
    --vqvae-hidden-channels 64 \
    --vqvae-embed-dim 38 \
    --batch-size 2 \
    --use-amp \
    --resume-training \
    --inject-style-to-decoder \
    --workers 8 \
    --tau 0.07 \
    --scale-contrastive-loss 0.03 \
    --model-id multiview-gradient-corrected \
    --gradient-accumulation-steps 4 \
    --gradient-checkpointing \
    --skip-recon-ratio 0.3 \
    --image-spacing 1.0 \
    --crop-margin 10 \
    --cache-dataset \
    --cache-dir /nfs/home/nglazman/cache/multiview \
