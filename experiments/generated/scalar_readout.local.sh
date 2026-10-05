#!/usr/bin/env bash
# Local CPU/CUDA/MPS: Python comes from the active environment unless overridden.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${ENCODER_REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO"
if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash $0 RUN_DIR [extra experiment arguments]" >&2
  exit 2
fi
RUN="$1"
shift
PY="${ENCODER_PYTHON:-python}"
OUTPUT="${SCALAR_OUTPUT:-$REPO/results/scalar_readout_$(date +%Y%m%d_%H%M%S)}"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
COMMAND=("$PY" -m training.scalar_readout_experiment --run-dir "$RUN" --out-dir "$OUTPUT"
  --view "${SCALAR_VIEW:-t1}" --source "${SCALAR_SOURCE:-backbone}" --device "${SCALAR_DEVICE:-auto}")
if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  printf '%q ' "${COMMAND[@]}" "$@"
  printf '\n'
  exit 0
fi
exec "${COMMAND[@]}" "$@"
