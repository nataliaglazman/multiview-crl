#!/usr/bin/env bash
# Auto-generated from experiments/encoder/pointwise.yaml (encoder-only: training.main_conv_synthetic); edit the YAML, not this file.
# Re-generate with: python scripts/launch.py experiments/encoder/pointwise.yaml --cluster local --generate
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
    --activation gelu \
    --batch-size 32 \
    --best-metric none \
    --no-cache \
    --content-bound sigmoid \
    --content-channels 10 \
    --contrastive-loss-type infonce \
    --contrastive-proj-dim 0 \
    --conv-readout linear \
    --cpu-threads 1 \
    --cross-view-negs-only \
    --data-seed 42 \
    --deterministic \
    --deterministic-warn-only \
    --downscale-factor 4 \
    --encoder-architecture conv \
    --encoder-head-hidden 100 \
    --eval-every 1000 \
    --eval-pooling gap \
    --floor-eval \
    --global-loss-weight 0.0 \
    --grad-clip 2.0 \
    --hash-training-inputs \
    --hidden-channels 64 \
    --interior-only \
    --latent-dim 13 \
    --lr 0.0001 \
    --n-content 9 \
    --n-style 3 \
    --nb-res-layers 0 \
    --norm-type layer \
    --num-train-samples 2000 \
    --num-val-samples 400 \
    --num-workers 0 \
    --out-dir results/encoder/runs \
    --patch-foreground-mask \
    --pointwise-loss-weight 1.0 \
    --require-new-run \
    --res 64 \
    --res-channels 32 \
    --seed 42 \
    --spatial-recovery-eval \
    --synthetic-clean-content \
    --synthetic-content-scale 1.0 \
    --synthetic-identifiable-ventricle \
    --synthetic-lesion-count 1 \
    --synthetic-lesion-intensity styled \
    --synthetic-lesion-placement wm_interior \
    --synthetic-lesion-target burden \
    --synthetic-mode pseudo_mri \
    --synthetic-normalize fixed_reference \
    --synthetic-style-scale 1.0 \
    --synthetic-sulcal-mode atrophy \
    --tau 0.1 \
    --train-patch-grid 16 16 16 \
    --train-steps 20000 \
    --model-id pointwise_k10_s42 \
    "$@"
