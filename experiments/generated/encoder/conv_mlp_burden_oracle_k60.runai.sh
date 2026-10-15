#!/usr/bin/env bash
# Auto-generated from experiments/encoder/conv_mlp_burden_oracle_k60.yaml (encoder-only: training.main_conv_synthetic); edit the YAML, not this file.
# Re-generate with: python scripts/launch.py experiments/encoder/conv_mlp_burden_oracle_k60.yaml --cluster runai --generate
# Preview without submitting: bash <this script> --dry-run
set -euo pipefail

# Folded to one line: backslash continuations do not survive the trip through runai ... bash -c.
TRAIN_CMD=$(tr '\n' ' ' <<'TRAIN_EOF'
cd /nfs/home/nglazman/crl-2/multiview-crl || { echo ERROR: /nfs/home/nglazman/crl-2/multiview-crl is missing inside the container - check the --host-path mount of /nfs >&2 ; exit 1 ; } ;
export WANDB_DIR=/tmp PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ;
python -c 'import torch; assert torch.cuda.is_available(), "Run:ai job has no usable CUDA device"' ;
python -m training.main_conv_synthetic
    --batch-size 32
    --best-metric none
    --no-cache
    --content-channels 9
    --contrastive-loss-type infonce
    --contrastive-proj-dim 0
    --conv-readout mlp
    --cpu-threads 1
    --cross-view-negs-only
    --data-seed 42
    --deterministic
    --deterministic-warn-only
    --downscale-factor 4
    --encoder-architecture conv
    --encoder-head-hidden 100
    --eval-every 2000
    --eval-pooling gap
    --floor-eval
    --grad-clip 2.0
    --hash-training-inputs
    --hidden-channels 64
    --latent-dim 12
    --lr 0.0001
    --n-content 9
    --n-style 3
    --nb-res-layers 2
    --num-train-samples 2000
    --num-val-samples 400
    --num-workers 0
    --out-dir results/encoder/runs
    --require-new-run
    --res 64
    --res-channels 32
    --seed 42
    --standardize-code
    --stratify-batches oracle
    --stratify-clusters 60
    --synthetic-clean-content
    --synthetic-content-scale 1.0
    --synthetic-lesion-intensity styled
    --synthetic-lesion-placement wm_interior
    --synthetic-lesion-t1-value 0.0
    --synthetic-lesion-target burden
    --synthetic-mode pseudo_mri
    --synthetic-normalize fixed_reference
    --synthetic-style-scale 1.0
    --tau 0.1
    --train-steps 10000
    --model-id conv_mlp_burden_oracle_k60_s42
TRAIN_EOF
)

if [[ "$#" -gt 1 ]]; then echo "Usage: $0 [--dry-run]" >&2; exit 2; fi
case "${1:-}" in
    --dry-run) printf '%s\n' "$TRAIN_CMD"; exit 0 ;;
    "") ;;
    *) echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

runai training standard submit conv-mlp-burden-oracle-k60-s42 \
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
