#!/bin/bash
# Train the given arms side by side to step 2000. Then rebuild the step-2000 factor table with every arm
# (compare_2000.txt) and add the arms' rows to the lesion probe at steps 500 and 2000 (lesion_s500.json,
# lesion_s2000.json). run_gap16.sh is this for gap_region16 and gap_region16_double.
#   nohup caffeinate -i -s bash out/regional_pilot_20261028/run_arms.sh BANK_DIR SHIM ARM... > out/regional_pilot_20261028/LOG 2>&1 &
set -u
BANK=$1
SHIM=$2
shift 2
ARMS=("$@")
OUT=out/regional_pilot_20261028
PY=/opt/miniconda3/envs/adni-analysis/bin/python
export PYTHONPATH=$SHIM:. PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONWARNINGS=ignore
PIDS=()
for ARM in "${ARMS[@]}"; do
    $PY $OUT/regional.py $BANK $OUT $ARM 2000 500 > $OUT/$ARM.log 2>&1 &
    PIDS+=($!)
    echo "$(date) started $ARM ($!)"
done

# Wait on each checkpoint file; the PID (a child of this script) only says whether the run died first.
for i in "${!ARMS[@]}"; do
    ARM=${ARMS[$i]}
    until [ -f $OUT/${ARM}_s2000.pt ]; do
        if ! kill -0 ${PIDS[$i]} 2>/dev/null && [ ! -f $OUT/${ARM}_s2000.pt ]; then
            echo "$(date) $ARM stopped before step 2000; see $OUT/$ARM.log"
            exit 1
        fi
        sleep 60
    done
done
sleep 30
echo "$(date) all at step 2000"
$PY $OUT/compare.py 2000 > $OUT/compare_2000.txt
$PY $OUT/lesion.py $BANK $OUT/lesion_s500.json $(for ARM in "${ARMS[@]}"; do echo $OUT/${ARM}_s500.pt; done)
$PY $OUT/lesion.py $BANK $OUT/lesion_s2000.json $(for ARM in "${ARMS[@]}"; do echo $OUT/${ARM}_s2000.pt; done)
echo "$(date) done"
