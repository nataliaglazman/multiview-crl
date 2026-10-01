#!/usr/bin/env bash
# Auto-generated encoder ablation; recipe: encoder_comparison.json
# Re-generate with: python scripts/generate_encoder_ablation_runai.py --include-controls
# Model: resnet_stride8_s42; data seed 42; model seed 42
# Preview without submitting: bash this-script.runai.sh --dry-run
set -euo pipefail

# Fold to one line, matching the existing generated Run:ai scripts.
TRAIN_CMD=$(tr '\n' ' ' <<'TRAIN_EOF'
set -euo pipefail ;
cd /nfs/home/nglazman/crl-2/multiview-crl ;
export WANDB_DIR=/tmp PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ;
python -c 'import torch; assert torch.cuda.is_available(), '"'"'Run:ai job has no usable CUDA device'"'"'' ;
python -m unittest tests.test_encoder_runtime -v ;
python -m training.main_conv_synthetic
    --res 64
    --latent-dim 12
    --content-channels 9
    --n-content 9
    --n-style 3
    --hidden-channels 64
    --res-channels 32
    --nb-res-layers 2
    --downscale-factor 4
    --encoder-head-hidden 100
    --synthetic-mode pseudo_mri
    --synthetic-normalize fixed_reference
    --synthetic-clean-content
    --synthetic-lesion-placement wm_interior
    --synthetic-style-scale 1.0
    --synthetic-content-scale 1.0
    --num-train-samples 2000
    --num-val-samples 400
    --batch-size 32
    --num-workers 0
    --no-cache
    --contrastive-loss-type infonce
    --tau 0.1
    --cross-view-negs-only
    --contrastive-proj-dim 0
    --lr 0.0001
    --grad-clip 2.0
    --train-steps 10000
    --eval-every 2000
    --eval-pooling gap
    --floor-eval
    --best-metric none
    --deterministic
    --deterministic-warn-only
    --cpu-threads 1
    --hash-training-inputs
    --seed 42
    --data-seed 42
    --model-seed 42
    --loader-seed 10042
    --encoder-architecture resnet18
    --require-new-run
    --out-dir /nfs/home/nglazman/crl-2/multiview-crl/results/encoder_ablations/runs
    --model-id resnet_stride8_s42
    --conv-readout linear
    --resnet-norm batch
    --resnet-output-stride 8
TRAIN_EOF
)

if [[ "$#" -gt 1 ]]; then echo "Usage: $0 [--dry-run]" >&2; exit 2; fi
case "${1:-}" in
    --dry-run) printf '%s\n' "$TRAIN_CMD"; exit 0 ;;
    "") ;;
    *) echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

runai training standard submit encoder-ablation-resnet-stride8-s42 \
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
