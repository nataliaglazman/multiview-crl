#!/usr/bin/env bash
# Auto-generated from: experiments/ablation_moco_patches.yaml
# Generated at: 2026-08-26T14:24:49Z
# Git SHA: 77548f5
# Re-generate with: python scripts/launch.py --generate --cluster runai

set -euo pipefail

# --- Training command (folded to one line: see note in scripts/launch.py) ---
TRAIN_CMD=$(tr '\n' ' ' <<'TRAIN_EOF'
cd /nfs/home/nglazman/crl-2/multiview-crl || { echo ERROR: /nfs/home/nglazman/crl-2/multiview-crl is missing inside the container - check the --host-path mount of /nfs >&2 ; exit 1 ; } ;
export PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl ;
python -m training.main_multimodal
    --batch-size 2
    --cache-dataset
    --cache-dir /nfs/home/nglazman/cache/multiview
    --channels-last
    --checkpoint-steps 500
    --content-dim 128
    --content-style-levels 0
    --contrastive-level-weights 3.0 0.5 0.5
    --contrastive-loss-type infonce
    --crop-margin 12
    --cross-view-negs-only
    --dataroot /nfs/home/nglazman/data
    --dataset-name ADNI_stripped_masks
    --deterministic
    --gradient-accumulation-steps 4
    --gradient-checkpointing
    --image-spacing 1.0
    --inject-style-to-decoder
    --labels-path /nfs/home/nglazman/nmpevqvae/labels_cleaned_3class.csv
    --log-steps 50
    --lr 0.001
    --mask-mode fixed
    --masks-dir /nfs/home/nglazman/data/ADNI_stripped_masks
    --moco-momentum 0.99
    --moco-queue-size 8192
    --pass-full-to-next-level
    --patch-contrastive
    --patch-grid 4 5 4
    --recon-loss-start-step 2000
    --resume-training
    --scale-adv-loss 0.0
    --scale-content-modality-adv 0.0
    --scale-contrastive-loss 1.0
    --scale-recon-loss 1.0
    --scale-style-contrastive-loss 0.0
    --scale-style-modality-ce 0.0
    --select-by-gated-score
    --separate-encoders
    --separation-floor-diagnosis-info 0.1
    --skip-recon-ratio 0.5
    --spatial-size 150 180 150
    --style-injection-mode film
    --model-id ablation-moco-patches
    --tau 0.07
    --total-dim 512
    --train-steps 50000
    --use-amp
    --use-moco
    --use-wandb
    --vq-commitment-weight 0.25
    --vqvae-embed-dim 38
    --vqvae-hidden-channels 64
    --vqvae-nb-entries 256
    --vqvae-nb-levels 3
    --vqvae-scaling-rates 2 2 2
    --workers 8
TRAIN_EOF
)

# --- RunAI submission ---
runai training standard submit ablation-moco-patches \
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
    --host-path path=/nfs,mount=/nfs,readwrite \
    --environment "WANDB_DIR=/tmp" \
    --environment "WANDB_API_KEY=${WANDB_API_KEY:?export WANDB_API_KEY before submitting - get it from https://wandb.ai/authorize}" \
    --command -- bash -c "${TRAIN_CMD}"
