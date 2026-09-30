#!/usr/bin/env bash
# Auto-generated from: experiments/adni_real.yaml
# Generated at: 2026-09-30T11:01:11Z
# Git SHA: d636e7b
# Re-generate with: python scripts/launch.py --generate --cluster runai

set -euo pipefail

# --- Training command (folded to one line: see note in scripts/launch.py) ---
TRAIN_CMD=$(tr '\n' ' ' <<'TRAIN_EOF'
cd /nfs/home/nglazman/crl-2/multiview-crl || { echo ERROR: /nfs/home/nglazman/crl-2/multiview-crl is missing inside the container - check the --host-path mount of /nfs >&2 ; exit 1 ; } ;
export PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl ;
python -m training.main_multimodal
    --batch-size 16
    --bt-corr-ema 0.99
    --bt-gap-lambda 6
    --bt-gap-sim-coeff 0.05
    --bt-gap-std-coeff 0.05
    --bt-gap-weight 1
    --bt-lambda 6
    --bt-patch-weight 1
    --bt-sim-coeff 0.0114
    --bt-std-coeff 0.227
    --cache-dataset
    --cache-dir /nfs/home/nglazman/cache/multiview
    --channels-last
    --checkpoint-steps 1000
    --content-dim 128
    --content-size 40
    --content-style-levels 0
    --contrastive-loss-type barlow_twins
    --cross-recon-start-step 0
    --cross-recon-style-source other_subject
    --cross-view-negs-only
    --dataroot /nfs/home/nglazman/data
    --dataset-name ADNI_stripped_masks
    --decoder-norm-type group
    --deterministic
    --grad-clip-norm 100
    --gradient-checkpointing
    --image-spacing 2.0
    --inject-style-to-decoder
    --labels-path /nfs/home/nglazman/nmpevqvae/labels_cleaned_3class.csv
    --log-steps 50
    --lr 0.001
    --mask-mode fixed
    --masks-dir /nfs/home/nglazman/data/ADNI_stripped_masks
    --moco-queue-size 0
    --no-final-recon-norm
    --norm-type layer
    --pass-full-to-next-level
    --patch-center-mode position
    --patch-contrastive
    --patch-foreground-mask
    --patch-foreground-thresh 0.05
    --patch-grid 8 8 8
    --quantize-style
    --recon-loss-start-step 0
    --resume-training
    --scale-adv-loss 0.0
    --scale-content-modality-adv 0.0
    --scale-contrastive-loss 100
    --scale-cross-recon-loss 0.0
    --scale-recon-loss 16
    --scale-style-contrastive-loss 0.0
    --scale-style-hsic-loss 0.0
    --scale-style-modality-ce 0.0
    --select-by-gated-score
    --separate-style-codebooks
    --separation-floor-diagnosis-info 0.1
    --single-count-commitment
    --spatial-size 96 112 96
    --split-seed 0
    --style-alignment-var-weight 1.0
    --style-contrastive-mode cosine
    --style-independence-var-weight 1.0
    --style-injection-mode input
    --model-id adni-real-bt-patch
    --tau 0.1
    --test-frac 0.1
    --total-dim 512
    --train-steps 20000
    --use-amp
    --use-wandb
    --val-frac 0.2
    --vq-commitment-weight 0.25
    --vqvae-embed-dim 48
    --vqvae-hidden-channels 48
    --vqvae-nb-entries 256
    --vqvae-nb-levels 1
    --vqvae-nb-res-layers 2
    --vqvae-scaling-rates 4
    --workers 8
TRAIN_EOF
)

# --- RunAI submission ---
runai training standard submit adni-real-bt-patch \
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
    --environment "OMP_NUM_THREADS=16" \
    --environment "OPENBLAS_NUM_THREADS=16" \
    --environment "MKL_NUM_THREADS=16" \
    --environment "NUMEXPR_NUM_THREADS=16" \
    --environment "WANDB_DIR=/tmp" \
    --environment "WANDB_API_KEY=${WANDB_API_KEY:?export WANDB_API_KEY before submitting - get it from https://wandb.ai/authorize}" \
    --command -- bash -c "${TRAIN_CMD}"
