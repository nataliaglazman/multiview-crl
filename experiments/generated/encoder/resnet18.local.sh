#!/usr/bin/env bash
# Auto-generated from experiments/encoder/resnet18.yaml (encoder-only: training.main_conv_synthetic); edit the YAML, not this file.
# Re-generate with: python scripts/launch.py experiments/encoder/resnet18.yaml --cluster local --generate
# Preview: bash <this script> --dry-run. Extra trainer flags override the recipe.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${ENCODER_REPO:-$(cd -- "$SCRIPT_DIR"/../../.. && pwd)}"
PYTHON="${ENCODER_PYTHON:-python}"
if [[ ! -f "$REPO/training/main_conv_synthetic.py" ]]; then
    echo "Repository missing at $REPO; set ENCODER_REPO to its root." >&2
    exit 1
fi
cd "$REPO"
exec "$PYTHON" -m scripts.run_encoder_local \
    --batch-size 32 \
    --best-metric none \
    --no-cache \
    --content-channels 9 \
    --contrastive-loss-type infonce \
    --contrastive-proj-dim 0 \
    --conv-readout linear \
    --cpu-threads 1 \
    --cross-view-negs-only \
    --data-seed 42 \
    --deterministic \
    --deterministic-warn-only \
    --downscale-factor 4 \
    --encoder-architecture resnet18 \
    --encoder-head-hidden 100 \
    --eval-every 2000 \
    --eval-pooling gap \
    --floor-eval \
    --grad-clip 2.0 \
    --hash-training-inputs \
    --hidden-channels 64 \
    --latent-dim 12 \
    --lr 0.0001 \
    --n-content 9 \
    --n-style 3 \
    --nb-res-layers 2 \
    --num-train-samples 2000 \
    --num-val-samples 400 \
    --num-workers 0 \
    --out-dir results/encoder/runs \
    --require-new-run \
    --res 64 \
    --res-channels 32 \
    --seed 42 \
    --synthetic-clean-content \
    --synthetic-content-scale 1.0 \
    --synthetic-lesion-placement wm_interior \
    --synthetic-mode pseudo_mri \
    --synthetic-normalize fixed_reference \
    --synthetic-style-scale 1.0 \
    --tau 0.1 \
    --train-steps 10000 \
    --model-id resnet18_s42 \
    "$@"
