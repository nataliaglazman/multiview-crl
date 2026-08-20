#!/usr/bin/env bash
# Auto-generated from: experiments/synthetic_causal.yaml
<<<<<<< HEAD
# Generated at: 2026-08-20T15:10:39Z
# Git SHA: 612e1b5
=======
# Generated at: 2026-08-20T15:52:43Z
# Git SHA: c5deaf4
>>>>>>> ff6501635d8f8ce7ffb67dab76cc0f6111d7bfdf
# Re-generate with: python scripts/launch.py --generate --cluster runai

set -euo pipefail

<<<<<<< HEAD
# --- Training command (folded to one line: see note in scripts/launch.py) ---
TRAIN_CMD=$(tr '\n' ' ' <<'TRAIN_EOF'
cd /nfs/home/nglazman/crl-2/multiview-crl || { echo ERROR: /nfs/home/nglazman/crl-2/multiview-crl is missing inside the container - check the --host-path mount of /nfs >&2 ; exit 1 ; } ;
export PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl ;
python -m training.main_multimodal
    --batch-size 128
    --bt-gap-lambda 0.01
    --bt-gap-weight 1
    --bt-patch-weight 1
    --bt-sim-coeff 0.5
    --bt-sim-normalize
    --bt-std-coeff 10
    --channels-last
    --checkpoint-steps 1000
    --content-dim 128
    --content-ratios 0.95
    --content-size 44
    --content-style-levels 0
    --contrastive-loss-type barlow_twins
    --cross-view-negs-only
    --dataroot /nfs/home/nglazman/data
    --dataset-name synthetic
    --dci-every 2000
    --decoder-norm-type group
    --deterministic
    --eval-dci
    --grad-clip-norm 100
    --image-spacing 1.0
    --inject-style-to-decoder
    --log-steps 50
    --lr 0.001
    --mask-mode fixed
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
    --scale-contrastive-loss 1
    --scale-recon-loss 1
    --scale-style-contrastive-loss 0.0
    --scale-style-modality-ce 0.0
    --select-by-gated-score
    --separate-encoders
    --separate-style-codebooks
    --separation-floor-diagnosis-info 0.1
    --single-count-commitment
    --style-injection-mode input
    --synthetic-causal
    --synthetic-causal-edge-prob 0.5
    --synthetic-causal-graph random
    --synthetic-clean-content
    --synthetic-identifiable-ventricle
    --synthetic-mode pseudo_mri
    --synthetic-normalize fixed_reference
    --synthetic-num-test 400
    --synthetic-num-train 2000
    --synthetic-num-val 1500
    --synthetic-res 64
    --model-id synthetic-causal-clean-content-commitment
    --tau 0.1
    --total-dim 512
    --train-steps 200000
    --use-amp
    --use-wandb
    --vq-commitment-weight 0.25
    --vqvae-embed-dim 48
    --vqvae-hidden-channels 48
    --vqvae-nb-entries 256
    --vqvae-nb-levels 1
    --vqvae-nb-res-layers 2
    --vqvae-scaling-rates 4
=======
# --- Training command ---
TRAIN_CMD=$(cat <<'TRAIN_EOF'
cd /nfs/home/nglazman/crl-2/multiview-crl && PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl \
python -m training.main_multimodal \
    --batch-size 128 \
    --bt-gap-lambda 0.001 \
    --bt-gap-weight 1 \
    --bt-patch-weight 1 \
    --bt-sim-coeff 0.5 \
    --bt-sim-normalize \
    --bt-std-coeff 10 \
    --channels-last \
    --checkpoint-steps 1000 \
    --content-dim 128 \
    --content-ratios 0.95 \
    --content-size 44 \
    --content-style-levels 0 \
    --contrastive-loss-type barlow_twins \
    --cross-view-negs-only \
    --dataroot /nfs/home/nglazman/data \
    --dataset-name synthetic \
    --dci-every 2000 \
    --decoder-norm-type group \
    --deterministic \
    --eval-dci \
    --grad-clip-norm 100 \
    --image-spacing 1.0 \
    --inject-style-to-decoder \
    --log-steps 50 \
    --lr 0.001 \
    --mask-mode fixed \
    --moco-queue-size 0 \
    --no-final-recon-norm \
    --norm-type layer \
    --pass-full-to-next-level \
    --patch-center-mode position \
    --patch-contrastive \
    --patch-foreground-mask \
    --patch-foreground-thresh 0.05 \
    --patch-grid 8 8 8 \
    --quantize-style \
    --recon-loss-start-step 0 \
    --resume-training \
    --scale-adv-loss 0.0 \
    --scale-content-modality-adv 0.0 \
    --scale-contrastive-loss 1 \
    --scale-recon-loss 1 \
    --scale-style-contrastive-loss 0.0 \
    --scale-style-modality-ce 0.0 \
    --select-by-gated-score \
    --separate-encoders \
    --separate-style-codebooks \
    --separation-floor-diagnosis-info 0.1 \
    --single-count-commitment \
    --style-injection-mode input \
    --synthetic-causal \
    --synthetic-causal-edge-prob 0.5 \
    --synthetic-causal-graph random \
    --synthetic-clean-content \
    --synthetic-identifiable-ventricle \
    --synthetic-mode pseudo_mri \
    --synthetic-normalize fixed_reference \
    --synthetic-num-test 400 \
    --synthetic-num-train 2000 \
    --synthetic-num-val 1500 \
    --synthetic-res 64 \
    --model-id synthetic-causal-clean-content-commitment-bt-gap-0001 \
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
    --vqvae-nb-res-layers 2 \
    --vqvae-scaling-rates 4 \
>>>>>>> ff6501635d8f8ce7ffb67dab76cc0f6111d7bfdf
    --workers 8
TRAIN_EOF
)

# --- RunAI submission ---
runai training standard submit synthetic-causal-clean-content-commitment-bt-gap-0001 \
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
