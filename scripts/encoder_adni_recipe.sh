# Encoder-only ADNI recipe shared by scripts/run_encoder_adni_slurm.sh and
# scripts/run_encoder_adni_runai.sh, so both clusters train the same thing.
# Sourced, not run: the caller sets its cluster's paths first (DATAROOT, LABELS_PATH,
# MASKS_DIR, CACHE_DIR, OUT_DIR). Every knob below can be overridden from the
# environment at submit time. Flag trade-offs: training/ENCODER_ADNI.md.

: "${DATAROOT:?}" "${LABELS_PATH:?}" "${MASKS_DIR:?}" "${CACHE_DIR:?}" "${OUT_DIR:?}"

# ---- Data: the preprocessing and split of experiments/adni_real.yaml ----
DATASET_NAME=${DATASET_NAME:-ADNI_stripped_masks}
SPACING=${SPACING:-2.0}
SPATIAL_SIZE=${SPATIAL_SIZE:-"96 112 96"}
VAL_FRAC=${VAL_FRAC:-0.2}
TEST_FRAC=${TEST_FRAC:-0.1}
SPLIT_SEED=${SPLIT_SEED:-0}
ASYMMETRIC_AUG=${ASYMMETRIC_AUG:-1}
# Part of the recipe, not just throughput: each worker draws its own augmentation
# stream, so a different count trains on different augmentations.
WORKERS=${WORKERS:-8}

# ---- Model and objective (experiments/encoder_comparison.json, except where noted) ----
MODEL_ID=${MODEL_ID:-encoder_adni_conv_s42}
ARCH=${ARCH:-conv}
LATENT_DIM=${LATENT_DIM:-12}
CONTENT_CHANNELS=${CONTENT_CHANNELS:-9}
BATCH_SIZE=${BATCH_SIZE:-32}
TAU=${TAU:-0.1}
LR=${LR:-1e-4}
STEPS=${STEPS:-10000}
EVAL_EVERY=${EVAL_EVERY:-500}   # ~14 epochs of ~1,170 training subjects at batch 32
PATCH_WEIGHT=${PATCH_WEIGHT:-0}
PATCH_GRID=${PATCH_GRID:-"6 7 6"}
SEED=${SEED:-42}
EXTRA_ARGS=${EXTRA_ARGS:-}

TRAIN_ARGS=(
    --dataset-name "$DATASET_NAME"
    --dataroot "$DATAROOT"
    --labels-path "$LABELS_PATH"
    --masks-dir "$MASKS_DIR"
    --cache-dir "$CACHE_DIR"
    --image-spacing "$SPACING"
    --spatial-size $SPATIAL_SIZE
    --val-frac "$VAL_FRAC"
    --test-frac "$TEST_FRAC"
    --split-seed "$SPLIT_SEED"
    --encoder-architecture "$ARCH"
    --latent-dim "$LATENT_DIM"
    --content-channels "$CONTENT_CHANNELS"
    --hidden-channels 64
    --res-channels 32
    --nb-res-layers 2
    --downscale-factor 4
    --encoder-head-hidden 100
    --contrastive-loss-type infonce
    --tau "$TAU"
    --cross-view-negs-only
    --contrastive-proj-dim 0
    --lr "$LR"
    --grad-clip 2.0
    --batch-size "$BATCH_SIZE"
    --train-steps "$STEPS"
    --eval-every "$EVAL_EVERY"
    --floor-eval
    --best-metric val_loss
    --num-workers "$WORKERS"
    --deterministic
    --deterministic-warn-only
    --seed "$SEED"
    --out-dir "$OUT_DIR"
    --model-id "$MODEL_ID"
    --require-new-run
)
if [[ "$ASYMMETRIC_AUG" == "1" ]]; then
    TRAIN_ARGS+=(--asymmetric-aug)
fi
if [[ "$PATCH_WEIGHT" != "0" ]]; then
    TRAIN_ARGS+=(--patch-loss-weight "$PATCH_WEIGHT" --train-patch-grid $PATCH_GRID --patch-foreground-mask)
fi
if [[ -n "$EXTRA_ARGS" ]]; then
    read -r -a EXTRA <<< "$EXTRA_ARGS"
    TRAIN_ARGS+=("${EXTRA[@]}")
fi
