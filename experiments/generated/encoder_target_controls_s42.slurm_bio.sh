#!/usr/bin/env bash
# Standalone local launcher; no Slurm submission is required.

set -euo pipefail
REPO="${ENCODER_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${ENCODER_PYTHON:-python3}"
DEVICE="${ENCODER_DEVICE:-cuda}"
RUNS="${ENCODER_RUNS:-$REPO/results/encoder_ablations_slurm_bio/runs}"
OUTPUT="${ENCODER_FOLLOWUPS:-/scratch/users/k24058220/encoder_followups_slurm_bio}"
NAMES=(t1 flair)
TASK_ID="${ENCODER_TASK_ID:-0}"
if [[ "${1:-}" == "t1" || "${1:-}" == "flair" ]]; then
  TASK_ID=0
  if [[ "$1" == "flair" ]]; then
    TASK_ID=1
  fi
  shift
fi
if [[ ! "$TASK_ID" =~ ^[0-1]$ ]]; then echo "Invalid task index: $TASK_ID" >&2; exit 2; fi
NAME="${NAMES[$TASK_ID]}"
STAMP="${ENCODER_STAMP:-local_$(date +%Y%m%d_%H%M%S)_${TASK_ID}}"
REFERENCE="${ENCODER_REFERENCE_RUN:-$RUNS/conv_mlp_s42}"
COMMAND=("$PYTHON" -m training.encoder_target_control --run-dir "$REFERENCE" --out-dir "$OUTPUT/supervised/${NAME}_${STAMP}" --view "$NAME" --device "$DEVICE" --steps 2000 --batch-size 8 --width 24 --grid 8 --test-samples 400 --seed 42)
if [[ "$#" -gt 1 ]]; then echo "Usage: $0 [t1|flair] [--dry-run]" >&2; exit 2; fi
case "${1:-}" in
  --dry-run) printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0 ;;
  "") ;;
  *) echo "Usage: $0 [t1|flair] [--dry-run]" >&2; exit 2 ;;
esac
cd "$REPO"
test -f eval/encoder/encoder_spatial_target_audit.py && test -f training/encoder_target_control.py
if ! PYTHON="$(command -v "$PYTHON")"; then echo "Python missing: ${ENCODER_PYTHON:-python3}" >&2; exit 1; fi
export PYTHONPATH="$REPO" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "Host: $(hostname) Task: $TASK_ID ($NAME)"
if [[ "$DEVICE" == "cuda" ]]; then
  "$PYTHON" -c 'import torch; print(torch.__version__); assert torch.cuda.is_available(), '"'"'No usable CUDA device'"'"''
else
  "$PYTHON" -c 'import torch; print(torch.__version__)'
fi
exec "${COMMAND[@]}"
