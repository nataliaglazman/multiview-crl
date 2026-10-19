#!/usr/bin/env bash
# Auto-generated from experiments/encoder/conv_mlp_gumbel.yaml (encoder-only: training.main_conv_synthetic); edit the YAML, not this file.
# Re-generate with: python scripts/launch.py experiments/encoder/conv_mlp_gumbel.yaml --cluster local --generate
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
    --attention-pool-frequencies 4 \
    --attention-pool-heads 4 \
    --batch-size 32 \
    --best-metric none \
    --bt-lambda 0.005 \
    --no-cache \
    --content-channels 9 \
    --contrastive-loss-type infonce \
    --contrastive-proj-dim 0 \
    --contrastive-proj-hidden 256 \
    --conv-readout mlp \
    --cpu-threads 1 \
    --cross-view-negs-only \
    --data-seed 42 \
    --dataset-name synthetic \
    --deterministic \
    --deterministic-warn-only \
    --device cuda \
    --downscale-factor 4 \
    --encoder-architecture conv \
    --encoder-head-hidden 100 \
    --eval-every 500 \
    --eval-patch-grid 4 5 4 \
    --eval-pooling gap \
    --floor-eval \
    --global-pool gap \
    --grad-clip 2.0 \
    --hash-training-inputs \
    --hidden-channels 64 \
    --image-spacing 2.0 \
    --latent-dim 12 \
    --lesion-input features \
    --lesion-keypoints 0 \
    --loader-seed 10042 \
    --log-every 100 \
    --lr 0.0001 \
    --mask-mode fixed \
    --model-seed 42 \
    --n-content 9 \
    --n-style 3 \
    --nb-res-layers 2 \
    --norm-type group \
    --num-train-samples 2000 \
    --num-val-samples 400 \
    --num-workers 0 \
    --out-dir results/encoder/runs \
    --patch-foreground-thresh 0.05 \
    --patch-loss-weight 0.0 \
    --require-new-run \
    --res 64 \
    --res-channels 32 \
    --resnet-norm batch \
    --resnet-output-stride 32 \
    --seed 42 \
    --no-separate-spatial-readout \
    --spatial-recovery-batch-size 4 \
    --spatial-recovery-eval \
    --spatial-recovery-grids 1 8 \
    --no-spatial-recovery-native \
    --spatial-recovery-seed 1729 \
    --spatial-recovery-test-samples 400 \
    --split-seed 0 \
    --synthetic-causal-edge-prob 0.5 \
    --synthetic-causal-graph random \
    --synthetic-causal-noise-scale 0.4 \
    --synthetic-causal-nonlinearity leaky_relu \
    --synthetic-clean-content \
    --synthetic-content-scale 1.0 \
    --synthetic-identifiable-ventricle \
    --synthetic-lesion-intensity styled \
    --synthetic-lesion-placement wm_interior \
    --synthetic-lesion-radius 0.1 \
    --synthetic-lesion-t1-value 0.4 \
    --synthetic-lesion-target position \
    --synthetic-mode pseudo_mri \
    --synthetic-n-deformation-grid 4 \
    --synthetic-n-fissure-grid 8 \
    --synthetic-normalize fixed_reference \
    --synthetic-style-scale 1.0 \
    --synthetic-temporal-sigma 0.05 \
    --synthetic-sulcal-mode atrophy \
    --tau 0.1 \
    --test-frac 0.0 \
    --train-patch-grid 8 8 8 \
    --train-steps 20000 \
    --val-frac 0.0 \
    --standardize-code \
    --model-id conv_mlp_gumbel_temp05 \
    "$@"

    # --lesion-loss-weight 0 \
    # --lesion-norm none \
    # --lesion-normative-components 20 \
    # --lesion-normative-subjects 300 \
    # --lesion-pairing cross_modal \
    # --lesion-proj-dim 8 \
    # --lesion-temperature 0.03 \
    #     --lesion-decorrelation-weight 0.0 \
    #     --lesion-frame brain \
    #     --lesion-detector shared \
    # --lesion-head-init random \
    #     --synthetic-causal \
        # --causal-eval \
        #     --synthetic-lesion-count 1 \

    # --mask-estimator relaxed \
    # --mask-init-gap 1.0 \
    # --mask-lr-scale 30 \
    #     