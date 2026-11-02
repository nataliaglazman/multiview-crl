#!/bin/bash
# Wait for the first four arms (PIDs as arguments), then train the fine-grid arms and re-score the first four's
# step-2000 checkpoints with the 4^3/8^3 readouts.
#   nohup caffeinate -i bash out/regional_pilot_20261028/queue.sh BANK_DIR SHIM PID... > out/regional_pilot_20261028/queue.log 2>&1 &
set -u
BANK=$1
SHIM=$2
shift 2
OUT=out/regional_pilot_20261028
PY=/opt/miniconda3/envs/adni-analysis/bin/python
export PYTHONPATH=$SHIM:. PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONWARNINGS=ignore
for pid in "$@"; do
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done
echo "$(date) first four done; starting fine-grid arms"
for ARM in region4 region8 region16 region16dot; do
    $PY $OUT/regional.py $BANK $OUT $ARM 2000 500 > $OUT/$ARM.log 2>&1 &
done
for ARM in gap region2 region3 gap_region2; do
    $PY $OUT/regional.py eval $BANK $OUT $ARM 2000 > $OUT/${ARM}_post.log 2>&1
done
echo "$(date) re-scoring done"
wait
echo "$(date) fine-grid arms done"
