#!/bin/bash
# Auto-generated from: /Users/nataliaglazman/Desktop/PhD/projects/multiview-crl/experiments/ablation_baseline.yaml
# Generated at: 2026-05-28T08:55:08Z
# Git SHA: 1c237d3
#SBATCH --job-name=ablation-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/home/nglazman/crl-2/multiview-crl/results/ablation-baseline/slurm_%j.log

set -euo pipefail
cd /nfs/home/nglazman/crl-2/multiview-crl
export PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl

python -m training.main_multimodal \
    --batch-size 4 \
    --cache-dataset \
    --cache-dir /nfs/home/nglazman/cache/multiview \
    --content-dim 128 \
    --content-style-levels 0 \
    --contrastive-loss-type infonce \
    --cross-view-negs-only \
    --dataroot /nfs/home/nglazman/data \
    --dataset-name ADNI_stripped_masks \
    --gradient-checkpointing \
    --image-spacing 1.0 \
    --labels-path /nfs/home/nglazman/nmpevqvae/labels_cleaned_3class.csv \
    --lr 0.001 \
    --mask-mode fixed \
    --masks-dir /nfs/home/nglazman/data/ADNI_stripped_masks \
    --moco-queue-size 0 \
    --pass-full-to-next-level \
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
    --spatial-size 150 180 150 \
    --model-id ablation-baseline \
    --tau 0.1 \
    --total-dim 512 \
    --train-steps 20000 \
    --use-amp \
    --use-wandb \
    --vq-commitment-weight 0.25 \
    --vqvae-embed-dim 32 \
    --vqvae-hidden-channels 32 \
    --vqvae-nb-entries 256 \
    --vqvae-nb-levels 3 \
    --vqvae-scaling-rates 2 2 2 \
    --workers 8
