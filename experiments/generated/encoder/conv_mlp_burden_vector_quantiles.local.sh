#!/usr/bin/env bash
# Auto-generated from experiments/encoder/conv_mlp_burden_vector_quantiles.yaml (encoder-only: training.main_conv_synthetic); edit the YAML, not this file.
# Re-generate with: python scripts/launch.py experiments/encoder/conv_mlp_burden_vector_quantiles.yaml --cluster local --generate
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
    --content-channels 14 \
    --contrastive-loss-type infonce \
    --contrastive-proj-dim 0 \
    --conv-readout mlp \
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
    --global-pool gap \
    --grad-clip 2.0 \
    --hash-training-inputs \
    --hidden-channels 64 \
    --latent-dim 18 \
    --lr 0.0001 \
    --mask-mode fixed \
    --n-content 9 \
    --n-style 3 \
    --nb-res-layers 2 \
    --norm-type group \
    --num-train-samples 2000 \
    --num-val-samples 400 \
    --num-workers 0 \
    --out-dir results/encoder/runs \
    --patch-foreground-mask \
    --patch-foreground-thresh 0.05 \
    --patch-loss-type infonce \
    --patch-loss-weight 1.0 \
    --patch-similarity cosine \
    --require-new-run \
    --res 64 \
    --res-channels 32 \
    --seed 42 \
    --no-separate-spatial-readout \
    --standardize-code \
    --standardize-patch-code \
    --stratify-balance 0.0 \
    --stratify-batches oracle \
    --stratify-neighbours 50 \
    --stratify-sampler neighbourhood \
    --synthetic-clean-content \
    --synthetic-content-scale 1.0 \
    --synthetic-identifiable-ventricle \
    --synthetic-lesion-count 1 \
    --synthetic-lesion-intensity styled \
    --synthetic-lesion-placement wm_interior \
    --synthetic-lesion-radius 0.1 \
    --synthetic-lesion-t1-value 0.0 \
    --synthetic-lesion-target burden \
    --synthetic-mode pseudo_mri \
    --synthetic-normalize fixed_reference \
    --synthetic-style-scale 1.0 \
    --synthetic-sulcal-mode atrophy \
    --synthetic-temporal-sigma 0.18 \
    --tau 0.1 \
    --train-patch-grid 16 16 16 \
    --train-steps 20000 \
    --vector-prediction-components 8 \
    --vector-prediction-deviation-strength 1.0 \
    --vector-prediction-grid 16 16 16 \
    --vector-prediction-hidden 128 \
    --vector-prediction-target quantile \
    --vector-prediction-template-subjects 300 \
    --vector-prediction-weight 1.0 \
    --vector-prediction-weight-cap 4.0 \
    --model-id conv_mlp_burden_vector_quantiles_s42 \
    "$@"
