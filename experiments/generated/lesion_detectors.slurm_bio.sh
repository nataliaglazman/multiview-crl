#!/bin/bash -l
# Each array task runs all four arms for one seed and verifies matched inputs/PCA.
#SBATCH --job-name=lesion-detectors
#SBATCH --output=/scratch/users/%u/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/%x-%A_%a.err
#SBATCH --array=0-2
#SBATCH --partition=biomed_a100_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
set -euo pipefail
REPO="${ENCODER_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY="${ENCODER_PYTHON:-$HOME/.conda/envs/multiview-env/bin/python}"
OUTPUT="${LESION_DETECTOR_RESULTS:-/scratch/users/k24058220/lesion_detectors}"
CACHE="${LESION_DETECTOR_CACHE:-/scratch/users/k24058220/cache/lesion_detectors}"
TASK="${SLURM_ARRAY_TASK_ID:-${LESION_DETECTOR_TASK_ID:-0}}"
if [[ ! "$TASK" =~ ^[0-2]$ ]]; then echo "Task index must be 0..2" >&2; exit 2; fi
SEEDS=(42 142 242)
STAMP="${SLURM_ARRAY_JOB_ID:-preview}"
ACTION="all"
if [[ "${1:-}" == "--dry-run" ]]; then ACTION="plan"; shift; fi
cd "$REPO"
if [[ "$ACTION" != "plan" ]]; then
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then echo "Submit with sbatch, or use --dry-run." >&2; exit 2; fi
  module load anaconda3/2022.10-gcc-13.2.0
  mkdir -p "$OUTPUT" "$CACHE/matplotlib" "$CACHE/tmp"
fi
export PYTHONPATH="$REPO" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export XDG_CACHE_HOME="$CACHE" MPLCONFIGDIR="$CACHE/matplotlib" TMPDIR="$CACHE/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$PY" scripts/compare_lesion_detectors.py "$ACTION" \
  --output-dir "$OUTPUT/${STAMP}_s${SEEDS[$TASK]}" --device cuda --seeds "${SEEDS[$TASK]}" "$@"
