#!/bin/bash
# Auto-generated from: /Users/nataliaglazman/Desktop/PhD/projects/multiview-crl/experiments/ablation_moco_patches.yaml
# Generated at: 2026-05-28T08:55:08Z
# Git SHA: 1c237d3
#SBATCH --job-name=ablation-moco-patches
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/home/nglazman/crl-2/multiview-crl/results/ablation-moco-patches/slurm_%j.log

set -euo pipefail
cd /nfs/home/nglazman/crl-2/multiview-crl
export PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl

python -m training.main_multimodal \
    --batch-size 2 \
    --cache-dataset \
    --cache-dir /nfs/home/nglazman/cache/multiview \
    --content-dim 128 \
    --content-style-levels 0 \
    --contrastive-level-weights 3.0 0.5 0.5 \
    --contrastive-loss-type infonce \
    --crop-margin 12 \
    --cross-view-negs-only \
    --dataroot /nfs/home/nglazman/data \
    --dataset-name ADNI_stripped_masks \
    --gradient-accumulation-steps 4 \
    --gradient-checkpointing \
    --image-spacing 1.0 \
    --inject-style-to-decoder \
    --labels-path /nfs/home/nglazman/nmpevqvae/labels_cleaned_3class.csv \
    --lr 0.001 \
    --mask-mode fixed \
    --masks-dir /nfs/home/nglazman/data/ADNI_stripped_masks \
    --moco-momentum 0.99 \
    --moco-queue-size 8192 \
    --pass-full-to-next-level \
    --patch-contrastive \
    --patch-grid 4 5 4 \
    --recon-loss-start-step 2000 \
    --resume-training \
    --scale-adv-loss 0.0 \
    --scale-content-modality-adv 0.0 \
    --scale-contrastive-loss 1.0 \
    --scale-recon-loss 1.0 \
    --scale-style-contrastive-loss 0.0 \
    --scale-style-modality-ce 0.0 \
    --select-by-gated-score \
    --separate-encoders \
    --separation-floor-diagnosis-info 0.1 \
    --skip-recon-ratio 0.5 \
    --spatial-size 150 180 150 \
    --style-injection-mode film \
    --model-id ablation-moco-patches \
    --tau 0.07 \
    --total-dim 512 \
    --train-steps 50000 \
    --use-amp \
    --use-moco \
    --use-wandb \
    --vq-commitment-weight 0.25 \
    --vqvae-embed-dim 38 \
    --vqvae-hidden-channels 64 \
    --vqvae-nb-entries 256 \
    --vqvae-nb-levels 3 \
    --vqvae-scaling-rates 2 2 2 \
    --workers 8
