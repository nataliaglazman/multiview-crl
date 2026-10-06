#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${ENCODER_REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO"
PY="${ENCODER_PYTHON:-python}"
OUTPUT="${LESION_DETECTOR_OUTPUT:-$REPO/results/lesion_detectors_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${LESION_DETECTOR_DEVICE:-cuda}"
ACTION="all"
if [[ "${1:-}" == "--dry-run" ]]; then ACTION="plan"; shift; fi
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
exec "$PY" scripts/compare_lesion_detectors.py "$ACTION" --output-dir "$OUTPUT" --device "$DEVICE" "$@"
