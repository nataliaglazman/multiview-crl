#!/usr/bin/env bash
# Apple-silicon (MPS) counterpart of encoder_resnet_stride8_s42.runai.sh: same recipe,
# variant and seeds; only --device, --out-dir and --model-id differ.
# Run: bash experiments/generated/encoder_resnet_stride8_s42.mps.sh
#   About 10 s/step at batch 32 on an M4 Pro (24 GB), so roughly 28 h for the 10k steps.
#   Output: results/encoder_ablations_mps/runs/resnet_stride8_s42_mps (+ logs/<model-id>.log)
# Preview the command:              ... --dry-run
# One disposable MPS step, then exit: ... --check
# Shorter or smaller runs get their own model id: ... --train-steps 200 --eval-every 100
# Python: $ENCODER_PYTHON, else the adni-analysis conda env, else python on PATH.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO"
if [[ -n "${ENCODER_PYTHON:-}" ]]; then
    PY="$ENCODER_PYTHON"
elif [[ -x /opt/miniconda3/envs/adni-analysis/bin/python ]]; then
    PY=/opt/miniconda3/envs/adni-analysis/bin/python
else
    PY=python
fi
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1
# Read once, when torch is imported. MaxPool3d has no MPS kernel in torch 2.6 and runs on CPU.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run=("$PY" scripts/run_encoder_mps.py --variant resnet_stride8 --seed 42 "$@")
# Keep the Mac from idle-sleeping for as long as the run lasts.
if command -v caffeinate >/dev/null 2>&1; then run=(caffeinate -i "${run[@]}"); fi
exec "${run[@]}"
