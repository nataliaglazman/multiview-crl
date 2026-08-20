#!/usr/bin/env bash
# Auto-generated from: experiments/ablation_baseline.yaml
# Generated at: 2026-08-20T15:52:43Z
# Git SHA: c5deaf4
# Re-generate with: python scripts/launch.py --generate --cluster runai

set -euo pipefail

# --- Training command ---
TRAIN_CMD=$(cat <<'TRAIN_EOF'
cd /nfs/home/nglazman/crl-2/multiview-crl && PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl \
python -m training.main_multimodal \
    --batch-size 4 \
    --cache-dataset \
    --cache-dir /nfs/home/nglazman/cache/multiview \
    --channels-last \
    --checkpoint-steps 500 \
    --content-dim 128 \
    --content-style-levels 0 \
    --contrastive-loss-type infonce \
    --cross-view-negs-only \
    --dataroot /nfs/home/nglazman/data \
    --dataset-name ADNI_stripped_masks \
    --deterministic \
    --gradient-checkpointing \
    --image-spacing 1.0 \
    --labels-path /nfs/home/nglazman/nmpevqvae/labels_cleaned_3class.csv \
    --log-steps 50 \
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
TRAIN_EOF
)

# --- RunAI submission ---
runai training standard submit ablation-baseline \
    --project nglazman \
    --image aicregistry:5000/nglazman:multiview-crl \
    --run-as-user \
    --large-shm \
    --node-type A100 \
    --gpu-devices-request 1 \
    --cpu-core-request 16 \
    --cpu-core-limit 32 \
    --cpu-memory-request 64G \
    --cpu-memory-limit 128G \
    --host-path path=/nfs:/nfs,mount=/nfs:/nfs \
    --command -- bash -c "${TRAIN_CMD}"
