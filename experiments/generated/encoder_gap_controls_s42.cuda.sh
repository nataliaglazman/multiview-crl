#!/usr/bin/env bash
# Local NVIDIA GPU: use Python from the active environment (e.g. monai_env).
# Supervised GAP control for signed sulcal (training/ENCODER_TARGET_FOLLOWUPS.md, Experiment 3).
# Same six arms as encoder_gap_controls_s42.slurm_bio.sh, run one after another.
# Finished arms are skipped, so re-running after an interruption continues from the next arm.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${ENCODER_REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO"
PY="${ENCODER_PYTHON:-python}"
REFERENCE="${ENCODER_REFERENCE_RUN:-$REPO/results/encoder_ablations_slurm_bio/runs/conv_mlp_s42}"
OUTPUT="${ENCODER_GAP_RESULTS:-$REPO/results/gap_control_cuda}"
STEPS="${GAP_CONTROL_STEPS:-6000}"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [[ ! -f "$REFERENCE/settings.json" ]]; then
    echo "Missing $REFERENCE/settings.json (results/ is gitignored; copy it from the Mac or set ENCODER_REFERENCE_RUN)" >&2
    exit 1
fi
"$PY" -c "import torch; assert torch.cuda.is_available(), 'No usable CUDA device'; print(torch.__version__, torch.cuda.get_device_name(0))"
mkdir -p "$OUTPUT"

for VIEW in t1 flair; do
    for ARM in gap grid8 gap_shuffled; do
        case "$ARM" in
            gap) FLAGS=(--grid 1) ;;
            grid8) FLAGS=(--grid 8) ;;
            gap_shuffled) FLAGS=(--grid 1 --shuffle-targets) ;;
        esac
        OUT="$OUTPUT/${VIEW}_${ARM}"
        if [[ -f "$OUT/report.json" ]] && grep -q '"status": "complete"' "$OUT/report.json"; then
            echo "Skipping finished arm: $OUT"
            continue
        fi
        if [[ -e "$OUT" ]]; then
            echo "Unfinished output $OUT exists; delete it to rerun this arm" >&2
            exit 1
        fi
        echo "=== $VIEW / $ARM ==="
        "$PY" -m training.encoder_target_control --run-dir "$REFERENCE" --out-dir "$OUT" --view "$VIEW" \
            --device cuda --steps "$STEPS" --batch-size 8 --width 24 --readout-channels 24 --lesion-weight 0 \
            --magnitude-head --eval-every 500 --test-samples 400 --seed 42 "${FLAGS[@]}" "$@"
        # The rendered banks are only a cache (~3-4 GB per arm); the scores live in report.json.
        if [[ -z "${KEEP_GAP_DATA:-}" ]]; then rm -rf "$OUT/data"; fi
    done
done

echo "Test sulcal scores (signed / magnitude head / sign accuracy):"
"$PY" - "$OUTPUT" <<'EOF'
import json, sys
from pathlib import Path

for path in sorted(Path(sys.argv[1]).glob("*/report.json")):
    report = json.loads(path.read_text())
    if report.get("status") != "complete":
        continue
    rows = {row["target"]: row["r2"] for row in report["test"]["factors"]}
    print(
        f"{path.parent.name:20s} signed R2={rows['sulcal_amplitude']:+.3f}  "
        f"magnitude-head R2={rows['sulcal_magnitude_head']:+.3f}  "
        f"sign acc={report['test']['sulcal_sign_accuracy']:.3f}"
    )
EOF
