#!/usr/bin/env bash
# Auto-generated from experiments/encoder/adni.yaml (encoder-only: training.main_conv_synthetic); edit the YAML, not this file.
# Re-generate with: python scripts/launch.py experiments/encoder/adni.yaml --cluster runai --generate
# Preview without submitting: bash <this script> --dry-run
set -euo pipefail

# Folded to one line: backslash continuations do not survive the trip through runai ... bash -c.
TRAIN_CMD=$(tr '\n' ' ' <<'TRAIN_EOF'
cd /nfs/home/nglazman/crl-2/multiview-crl || { echo ERROR: /nfs/home/nglazman/crl-2/multiview-crl is missing inside the container - check the --host-path mount of /nfs >&2 ; exit 1 ; } ;
export WANDB_DIR=/tmp PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ;
python -c 'import torch; assert torch.cuda.is_available(), "Run:ai job has no usable CUDA device"' ;
python -m training.main_conv_synthetic
    --asymmetric-aug
    --batch-size 32
    --best-metric val_loss
    --cache-dir /nfs/home/nglazman/cache/multiview
    --content-channels 9
    --contrastive-loss-type infonce
    --contrastive-proj-dim 0
    --conv-readout mlp
    --cross-view-negs-only
    --dataroot /nfs/home/nglazman/data
    --dataset-name ADNI_stripped_masks
    --deterministic
    --deterministic-warn-only
    --downscale-factor 4
    --encoder-architecture conv
    --encoder-head-hidden 100
    --eval-every 500
    --floor-eval
    --grad-clip 2.0
    --hidden-channels 64
    --image-spacing 2.0
    --labels-path /nfs/home/nglazman/nmpevqvae/labels_cleaned_3class_demog.csv
    --latent-dim 12
    --lr 0.0001
    --masks-dir /nfs/home/nglazman/data/ADNI_stripped_masks
    --nb-res-layers 2
    --norm-type group
    --num-workers 8
    --out-dir results/encoder_adni/runs
    --require-new-run
    --res-channels 32
    --seed 42
    --spatial-size 96 112 96
    --split-seed 0
    --tau 0.1
    --test-frac 0.1
    --train-steps 10000
    --val-frac 0.2
    --model-id encoder_adni_conv_mlp_s42
TRAIN_EOF
)

if [[ "$#" -gt 1 ]]; then echo "Usage: $0 [--dry-run]" >&2; exit 2; fi
case "${1:-}" in
    --dry-run) printf '%s\n' "$TRAIN_CMD"; exit 0 ;;
    "") ;;
    *) echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

runai training standard submit encoder-adni-conv-mlp-s42 \
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
    --command -- bash -c "${TRAIN_CMD}"
