#!/usr/bin/env bash
# Auto-generated from experiments/encoder/adni_patch.yaml (encoder-only: training.main_conv_synthetic); edit the YAML, not this file.
# Re-generate with: python scripts/launch.py experiments/encoder/adni_patch.yaml --cluster local --generate
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
    --asymmetric-aug \
    --batch-size 32 \
    --best-metric val_loss \
    --content-channels 9 \
    --contrastive-loss-type infonce \
    --contrastive-proj-dim 0 \
    --conv-readout mlp \
    --cross-view-negs-only \
    --dataset-name ADNI_stripped_masks \
    --deterministic \
    --deterministic-warn-only \
    --downscale-factor 4 \
    --encoder-architecture conv \
    --encoder-head-hidden 100 \
    --eval-every 500 \
    --floor-eval \
    --grad-clip 2.0 \
    --hidden-channels 64 \
    --image-spacing 2.0 \
    --latent-dim 12 \
    --lr 0.0001 \
    --nb-res-layers 2 \
    --norm-type group \
    --num-workers 8 \
    --out-dir results/encoder_adni/runs \
    --patch-foreground-mask \
    --patch-loss-weight 1.0 \
    --require-new-run \
    --res-channels 32 \
    --seed 42 \
    --spatial-size 96 112 96 \
    --split-seed 0 \
    --tau 0.1 \
    --test-frac 0.1 \
    --train-patch-grid 6 7 6 \
    --train-steps 10000 \
    --val-frac 0.2 \
    --model-id encoder_adni_patch_s42 \
    "$@"
