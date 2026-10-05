#!/bin/bash -l
# Four source/view combinations; each job runs all three matched training arms.
#SBATCH --job-name=scalar-readout
#SBATCH --output=/scratch/users/%u/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/%x-%A_%a.err
#SBATCH --array=0-3
#SBATCH --partition=biomed_a100_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
set -euo pipefail
REPO="${ENCODER_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN="${ENCODER_REFERENCE_RUN:-$REPO/results/encoder_ablations_slurm_bio/runs/conv_mlp_s42}"
PY="${ENCODER_PYTHON:-$HOME/.conda/envs/multiview-env/bin/python}"
OUTPUT="${SCALAR_RESULTS:-/scratch/users/k24058220/scalar_readout}"
CACHE="${SCALAR_CACHE:-/scratch/users/k24058220/cache/scalar_readout}"
TASK="${SLURM_ARRAY_TASK_ID:-${SCALAR_TASK_ID:-0}}"
if [[ ! "$TASK" =~ ^[0-3]$ ]]; then echo "Task index must be 0..3" >&2; exit 2; fi
SOURCES=(backbone backbone image image)
VIEWS=(t1 flair t1 flair)
STAMP="${SLURM_ARRAY_JOB_ID:-preview}_${TASK}"
COMMAND=("$PY" -m training.scalar_readout_experiment --run-dir "$RUN"
  --out-dir "$OUTPUT/${SOURCES[$TASK]}_${VIEWS[$TASK]}_$STAMP"
  --source "${SOURCES[$TASK]}" --view "${VIEWS[$TASK]}" --device cuda --cache-dir "$CACHE")
if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  printf '%q ' "${COMMAND[@]}" "$@"
  printf '\n'
  exit 0
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then echo "Submit with sbatch, or use --dry-run." >&2; exit 2; fi
cd "$REPO"
module load anaconda3/2022.10-gcc-13.2.0
test -x "$PY"
export PYTHONPATH="$REPO" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "${COMMAND[@]}" "$@"
