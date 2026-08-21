#!/usr/bin/env bash
# Auto-generated from: experiments/synthetic_causal_groupnorm_control.yaml
# Generated at: 2026-08-21T14:56:46Z
# Git SHA: fac6124
# Re-generate with: python scripts/launch.py --generate --cluster runai

set -euo pipefail

# --- Training command (folded to one line: see note in scripts/launch.py) ---
TRAIN_CMD=$(tr '\n' ' ' <<'TRAIN_EOF'
cd /nfs/home/nglazman/crl-2/multiview-crl || { echo ERROR: /nfs/home/nglazman/crl-2/multiview-crl is missing inside the container - check the --host-path mount of /nfs >&2 ; exit 1 ; } ;
export PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl ;
python -m training.main_multimodal
    --batch-size 32
    --channels-last
    --checkpoint-steps 1000
    --content-dim 128
    --content-ratios 0.95
    --content-size 48
    --content-style-levels 0
    --contrastive-loss-type infonce
    --cross-view-negs-only
    --dataroot /nfs/home/nglazman/data
    --dataset-name synthetic
    --dci-every 2000
    --decoder-norm-type group
    --deterministic
    --eval-dci
    --image-spacing 1.0
    --inject-style-to-decoder
    --log-steps 50
    --lr 0.001
    --mask-mode fixed
    --moco-queue-size 0
    --no-final-recon-norm
    --norm-type group
    --pass-full-to-next-level
    --patch-contrastive
    --patch-grid 8 8 8
    --quantize-style
    --recon-loss-start-step 0
    --resume-training
    --scale-adv-loss 0.0
    --scale-content-modality-adv 0.0
    --scale-contrastive-loss 0
    --scale-recon-loss 1
    --scale-style-contrastive-loss 0.0
    --scale-style-modality-ce 0.0
    --select-by-gated-score
    --separate-content-codebooks
    --separate-encoders
    --separate-style-codebooks
    --separation-floor-diagnosis-info 0.1
    --style-injection-mode film
    --synthetic-causal
    --synthetic-causal-edge-prob 0.5
    --synthetic-causal-graph random
    --synthetic-mode pseudo_mri
    --synthetic-normalize fixed_reference
    --synthetic-num-test 400
    --synthetic-num-train 2000
    --synthetic-num-val 1500
    --synthetic-res 128
    --model-id synthetic-causal-groupnorm-control
    --tau 0.07
    --total-dim 512
    --train-steps 300000
    --use-amp
    --use-wandb
    --vq-commitment-weight 0.25
    --vqvae-embed-dim 48
    --vqvae-hidden-channels 48
    --vqvae-nb-entries 256
    --vqvae-nb-levels 1
    --vqvae-scaling-rates 4
    --workers 8
TRAIN_EOF
)

# --- RunAI submission ---
runai training standard submit synthetic-causal-groupnorm-control \
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
