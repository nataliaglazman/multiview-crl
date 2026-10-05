#!/usr/bin/env bash
# Local NVIDIA GPU: use Python from the active environment (e.g. monai_env).
# Same Conv + MLP global/patch recipe as the MPS and SLURM launchers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${ENCODER_REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO"
PY="${ENCODER_PYTHON:-python}"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$PY" scripts/run_encoder_mps.py --device cuda --variant conv_mlp --seed 42 \
    --patch-loss-weight 0 --train-patch-grid 8 8 8 --spatial-recovery-eval \
    --results-dir "${ENCODER_PATCH_RESULTS:-$REPO/results/encoder_patch_cuda}" "$@"
