#!/bin/bash
# Train arms side by side past the pilot's 2000 steps on any machine (regional.py picks CUDA, then MPS, then CPU), into
# long/ so the 2000-step rows here stay as they are; then add every checkpoint (one per 500 steps) to long/lesion.json.
# Needs only this repo's code, the bank and shim/ (an lpips stand-in: the trainer imports lpips but never uses it).
# From the repo root, with PY a Python that has torch, monai, nibabel and scikit-learn (default: python on PATH):
#   nohup bash out/regional_pilot_20261028/run_long.sh BANK_DIR STEPS ARM... > long.log 2>&1 &
set -eu
BANK=$1
STEPS=$2
shift 2
ARMS=("$@")
HERE=out/regional_pilot_20261028
OUT=$HERE/long
PY=${PY:-python}
export PYTHONPATH=$HERE/shim:. PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONWARNINGS=ignore
mkdir -p $OUT
PIDS=()
for ARM in "${ARMS[@]}"; do
    $PY $HERE/regional.py $BANK $OUT $ARM $STEPS 500 > $OUT/$ARM.log 2>&1 &
    PIDS+=($!)
    echo "$(date) started $ARM ($!); progress in $OUT/$ARM.log"
done
for i in "${!ARMS[@]}"; do
    wait ${PIDS[$i]} || { echo "$(date) ${ARMS[$i]} stopped before step $STEPS; see $OUT/${ARMS[$i]}.log"; exit 1; }
done
echo "$(date) all at step $STEPS"
for ARM in "${ARMS[@]}"; do
    $PY $HERE/lesion.py $BANK $OUT/lesion.json $(seq -f "$OUT/${ARM}_s%g.pt" 500 500 $STEPS)
done
echo "$(date) done"
