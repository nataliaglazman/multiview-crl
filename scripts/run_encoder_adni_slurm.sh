#!/bin/bash -l
# Encoder-only multi-view contrastive learning on real ADNI T1/FLAIR pairs
# (training/main_conv_synthetic.py with --dataset-name). Flags and their trade-offs:
# training/ENCODER_ADNI.md.
#
# Check the data first (same paths, size and split; read-only, login node, training env):
#     python scripts/preflight_adni.py experiments/adni_real.yaml --cluster slurm --sample 8
# Preview, then submit from the repository root:
#     bash scripts/run_encoder_adni_slurm.sh --dry-run
#     sbatch scripts/run_encoder_adni_slurm.sh
#
# Variants (override via environment at submit time):
#     # augmentation as the VQ-VAE ADNI run (shared intensity shift; see the doc first):
#     MODEL_ID=encoder_adni_sharedaug ASYMMETRIC_AUG=0 sbatch scripts/run_encoder_adni_slurm.sh
#     # add patch InfoNCE on cubic 4x4x4-cell bins of the 24x28x24 map, brain positions only:
#     MODEL_ID=encoder_adni_patch PATCH_WEIGHT=1 sbatch scripts/run_encoder_adni_slurm.sh
#     # content size sweep:
#     for c in 9 16 32; do MODEL_ID=encoder_adni_c$c CONTENT_CHANNELS=$c LATENT_DIM=$((c + 3)) \
#         sbatch scripts/run_encoder_adni_slurm.sh; done
#
# Output: $OUT_DIR/$MODEL_ID/ (settings.json, split.json, separation_step*.json,
# model.pt, model_best.pt by held-out loss, TensorBoard).

#SBATCH --job-name=encoder-adni
#SBATCH --output=/scratch/users/%u/%x-%j.out
#SBATCH --error=/scratch/users/%u/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
# Excludes only the B200 node (sm_100), which torch 2.3.1+cu121 cannot run on.
#SBATCH --constraint=a100|h200|l40s

set -euo pipefail

# ---- Data (experiments/cluster/slurm.yaml) ----
DATAROOT=${DATAROOT:-/scratch/users/k24058220}
DATASET_NAME=${DATASET_NAME:-ADNI_stripped_masks}
LABELS_PATH=${LABELS_PATH:-/users/k24058220/multiview-crl/labels_cleaned_3class.csv}
MASKS_DIR=${MASKS_DIR:-/scratch/users/k24058220/ADNI_stripped_masks}
# Same spacing, size and masks as experiments/adni_real.yaml, so its preprocessed cache is reused.
CACHE_DIR=${CACHE_DIR:-/scratch/users/k24058220/cache/multiview}
SPACING=${SPACING:-2.0}
SPATIAL_SIZE=${SPATIAL_SIZE:-"96 112 96"}
VAL_FRAC=${VAL_FRAC:-0.2}
TEST_FRAC=${TEST_FRAC:-0.1}
SPLIT_SEED=${SPLIT_SEED:-0}
ASYMMETRIC_AUG=${ASYMMETRIC_AUG:-1}

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
WORKERS=${WORKERS:-8}
EXTRA_ARGS=${EXTRA_ARGS:-}

OUT_DIR=${OUT_DIR:-/scratch/users/k24058220/encoder_adni/runs}
REPO="${ENCODER_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONDA_ENV_NAME=multiview-env
PYTHON="${ENCODER_PYTHON:-${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python}"

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

if [[ "$#" -gt 1 ]]; then echo "Usage: $0 [--dry-run]" >&2; exit 2; fi
case "${1:-}" in
    --dry-run)
        printf '%q ' "$PYTHON" -m training.main_conv_synthetic "${TRAIN_ARGS[@]}"
        printf '\n'
        exit 0 ;;
    "") ;;
    *) echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit with sbatch from the repository root; use --dry-run to preview locally." >&2
    exit 2
fi
cd "$REPO"
if [[ ! -f training/main_conv_synthetic.py ]]; then
    echo "Repository missing at $REPO. Submit from its root or set ENCODER_REPO." >&2
    exit 1
fi

# Reuse the prepared environment; concurrent jobs never install/remove packages.
module load anaconda3/2022.10-gcc-13.2.0
if [[ ! -x "$PYTHON" ]]; then
    echo "Python not found at $PYTHON. Prepare $CONDA_ENV_NAME or set ENCODER_PYTHON before sbatch." >&2
    exit 1
fi
export PYTHONPATH="$REPO"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
# One thread per process: the main process feeds the GPU, the workers augment on the CPU.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "Node: $(hostname)  Job: $SLURM_JOB_ID  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON" -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda}"); assert torch.cuda.is_available(), "Allocated job has no usable CUDA device"'

exec "$PYTHON" -m training.main_conv_synthetic "${TRAIN_ARGS[@]}"
