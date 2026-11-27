#!/usr/bin/env bash
# One warm.py run inside the Run:ai training image, one job per run (25 Nov; stage 71's A100 replicate):
#   runai_warm.sh NAME BANK STEPS EVERY [warm.py options...]
# The run goes to runs/NAME and its log to logs/NAME.log in the cluster checkout, as run_pc.sh's arm() writes them, so
# a run copied back to the PC reads like one trained there. A finished run (its log ends in DONE) is left alone.
# THREADS (default 8) sets the CPU threads; the job requests 16 cores. WARM (default warm.py) names the training
# script under out/burden_feasibility_20261102/, for a version the checkout does not track (stage 77: stage77/warm.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
D=out/burden_feasibility_20261102
NAME=$1 BANK=$2 STEPS=$3 EVERY=$4
shift 4
mkdir -p "$D/logs" "$D/runs"
if grep -qE '^(TEST|DONE)' "$D/logs/$NAME.log" 2> /dev/null; then
    echo "$NAME is already done"
    exit 0
fi
export PYTHONPATH=$D:out/regional_pilot_20261028/shim:.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${THREADS:-8} OPENBLAS_NUM_THREADS=${THREADS:-8} MKL_NUM_THREADS=${THREADS:-8}
rm -rf "${D:?}/runs/$NAME"
WARM=${WARM:-warm.py}
echo "$(date '+%F %T') $NAME on $(hostname): $WARM $BANK cold $STEPS $EVERY $*" > "$D/logs/$NAME.log"
python "$D/$WARM" "$BANK" "$D/runs/$NAME" cold "$STEPS" "$EVERY" "$@" >> "$D/logs/$NAME.log" 2>&1
