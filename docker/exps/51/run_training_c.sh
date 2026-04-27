#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl \
python /nfs/home/nglazman/crl-2/multiview-crl/training/main_multimodal.py \
    --dataroot /nfs/home/nglazman/data \
    --dataset_name ADNI_stripped_masks \
    --encoder-type vqvae \
    --model-id multiview-32-ch-04-04-04-02-tau-quantize-style-bs-4-2-1-1-patch-change-fml-speed \
    --scale-recon-loss 10 \
    --compile-model --channels-last \
    --quantize-style \
    --inject-style-to-decoder \
    --patch-contrastive --patch-grid 6 8 6 \
    --labels-path /nfs/home/nglazman/nmpevqvae/labels_cleaned_3class.csv \
    --moco-queue-size 8192 \
    --vqvae-nb-entries 256 128 64  \
    --moco-momentum 0.99 \
    --mask-mode fixed \
    --vqvae-nb-levels 3 \
    --lr 0.001 \
    --content-dim 128 \
    --total-dim 512 \
    --vqvae-scaling-rates 2 2 2 \
    --train-steps 80000 \
    --vqvae-hidden-channels 32 \
    --vqvae-embed-dim 32 \
    --batch-size 4 \
    --spatial-size 150 180 150 \
    --use-amp \
    --resume-training \
    --workers 8 \
    --recon-loss-start-step 2000 \
    --tau 0.8 \
    --scale-contrastive-loss 10.0 \
    --content-style-levels 0 1 2 \
    --cross-view-negs-only \
    --scale-style-contrastive-loss 10 \
    --content-ratios 0.4 0.4 0.4 \
    --separate-encoders \
    --gradient-accumulation-steps 4 \
    --gradient-checkpointing \
    --skip-recon-ratio 0.5 \
    --image-spacing 1.0 \
    --cache-dataset \
    --vq-commitment-weight 0.25 \
    --cache-dir /nfs/home/nglazman/cache/multiview \
    --contrastive-level-weights 2.0 1.0 1.0 \
    --pass-full-to-next-level \


