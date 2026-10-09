#!/bin/bash -l
# Encoder-only multi-view contrastive learning on real ADNI T1/FLAIR pairs
# (training/main_conv_synthetic.py with --dataset-name) on CREATE SLURM. The recipe
# and its knobs live in scripts/encoder_adni_recipe.sh, shared with
# scripts/run_encoder_adni_runai.sh. Flags and their trade-offs: training/ENCODER_ADNI.md.
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

# sbatch runs a spooled copy of this script, so the repository is the submit directory.
REPO="${ENCODER_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONDA_ENV_NAME=multiview-env
PYTHON="${ENCODER_PYTHON:-${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python}"

# ---- Paths (experiments/cluster/slurm.yaml) ----
DATAROOT=${DATAROOT:-/scratch/users/k24058220}
LABELS_PATH=${LABELS_PATH:-/users/k24058220/multiview-crl/labels_cleaned_3class_demog.csv}
MASKS_DIR=${MASKS_DIR:-/scratch/users/k24058220/ADNI_stripped_masks}
# Same spacing, size and masks as experiments/adni_real.yaml, so its preprocessed cache is reused.
CACHE_DIR=${CACHE_DIR:-/scratch/users/k24058220/cache/multiview}
OUT_DIR=${OUT_DIR:-/scratch/users/k24058220/encoder_adni/runs}

if [[ ! -f "$REPO/scripts/encoder_adni_recipe.sh" ]]; then
    echo "No $REPO/scripts/encoder_adni_recipe.sh: run from the repository root or set ENCODER_REPO." >&2
    exit 1
fi
source "$REPO/scripts/encoder_adni_recipe.sh"

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
