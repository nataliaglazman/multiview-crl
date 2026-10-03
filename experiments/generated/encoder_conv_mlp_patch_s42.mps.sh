#!/usr/bin/env bash
# Conv + MLP: existing global InfoNCE plus unit-weight 8^3 patch InfoNCE.
# --dry-run previews; --check tests a disposable training step without starting a run.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${ENCODER_REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO"
PY="${ENCODER_PYTHON:-python}"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
run=("$PY" scripts/run_encoder_mps.py --variant conv_mlp --seed 42
     --patch-loss-weight 1 --train-patch-grid 8 8 8 --spatial-recovery-eval
     --results-dir "${ENCODER_PATCH_RESULTS:-$REPO/results/encoder_patch_mps}" "$@")
if command -v caffeinate >/dev/null 2>&1; then run=(caffeinate -i "${run[@]}"); fi
exec "${run[@]}"
