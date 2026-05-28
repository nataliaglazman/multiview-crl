#!/bin/bash
# Auto-generated from: /Users/nataliaglazman/Desktop/PhD/projects/multiview-crl/experiments/synthetic_defaults.yaml
# Generated at: 2026-05-28T08:55:08Z
# Git SHA: 1c237d3
#SBATCH --job-name=synthetic_defaults
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/home/nglazman/crl-2/multiview-crl/results/synthetic_defaults/slurm_%j.log

set -euo pipefail
cd /nfs/home/nglazman/crl-2/multiview-crl
export PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl

python -m training.main_multimodal \
    --batch-size 32 \
    --content-dim 128 \
    --content-ratios 0.95 \
    --content-size 9 \
    --content-style-levels 0 \
    --contrastive-loss-type infonce \
    --cross-view-negs-only \
    --dataroot /nfs/home/nglazman/data \
    --dataset-name synthetic \
    --image-spacing 1.0 \
    --lr 0.001 \
    --mask-mode fixed \
    --moco-queue-size 0 \
    --pass-full-to-next-level \
    --patch-contrastive \
    --quantize-style \
    --recon-loss-start-step 0 \
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
    --synthetic-mode pseudo_mri \
    --synthetic-num-test 400 \
    --synthetic-num-train 2000 \
    --synthetic-num-val 200 \
    --synthetic-res 64 \
    --tau 0.1 \
    --total-dim 512 \
    --train-steps 200000 \
    --use-amp \
    --use-wandb \
    --vq-commitment-weight 0.25 \
    --vqvae-embed-dim 48 \
    --vqvae-hidden-channels 48 \
    --vqvae-nb-entries 256 \
    --vqvae-nb-levels 1 \
    --vqvae-scaling-rates 2 \
    --workers 8
