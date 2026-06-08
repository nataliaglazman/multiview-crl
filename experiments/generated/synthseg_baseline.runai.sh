#!/usr/bin/env bash
# Auto-generated from: experiments/synthseg_baseline.yaml
# Generated at: 2026-06-08T09:22:24Z
# Git SHA: 4aac7ae
# Re-generate with: python scripts/launch.py --generate --cluster runai

set -euo pipefail

REPO="/nfs/home/nglazman/crl-2/multiview-crl"

# --- Training command ---
TRAIN_CMD=$(cat <<'TRAIN_EOF'
cd ${REPO} && PYTHONPATH=${REPO} \
python -m training.main_multimodal \
    --batch-size 4 \
    --cache-dataset \
    --cache-dir /nfs/home/nglazman/cache/multiview \
    --channels-last \
    --content-dim 128 \
    --content-style-levels 0 \
    --contrastive-loss-type infonce \
    --cross-view-negs-only \
    --dataroot /data/natalia/ADNI_synthseg \
    --dataset-name ADNI_stripped_masks \
    --gradient-checkpointing \
    --image-spacing 1.0 \
    --labels-path /data/natalia/ADNI_synthseg/labels.csv \
    --lr 0.001 \
    --mask-mode fixed \
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
    --model-id synthseg-baseline \
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
TRAIN_EOF
)

# --- RunAI submission ---
runai submit synthseg-baseline \
    --project nglazman \
    --image aicregistry:5000/nglazman:multiview-crl-vqvae-final \
    --run-as-user \
    --large-shm \
    --node-type A100 \
    --gpu 1 \
    --cpu 16 \
    --cpu-limit 32 \
    --memory 64G \
    --memory-limit 128G \
    --volume /nfs:/nfs \
    --command -- bash -c "${TRAIN_CMD}"
