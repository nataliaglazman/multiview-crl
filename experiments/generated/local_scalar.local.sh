#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${ENCODER_REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO"
if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash $0 RUN_DIR [--dry-run] [experiment arguments]" >&2
  exit 2
fi
RUN="$1"
shift
PY="${ENCODER_PYTHON:-python}"
OUTPUT="${LOCAL_SCALAR_OUTPUT:-$REPO/results/local_scalar_$(date +%Y%m%d_%H%M%S)}"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# MPS can use the existing CPU fallback for unsupported pooling operations.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
COMMAND=("$PY" -m training.local_scalar_experiment --run-dir "$RUN" --out-dir "$OUTPUT"
  --device "${LOCAL_SCALAR_DEVICE:-auto}")
if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  printf '%q ' "${COMMAND[@]}" "$@"
  printf '\n'
  exit 0
fi
exec "${COMMAND[@]}" "$@"
