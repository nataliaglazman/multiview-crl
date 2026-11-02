#!/bin/bash
# Train gap_region16 and gap_region16_double side by side to step 2000. Then rebuild the step-2000 factor table with
# every arm (compare_2000.txt) and add both arms' rows to the lesion probe at steps 500 and 2000 (lesion_s500.json,
# lesion_s2000.json, next to region16 and the rest).
#   nohup caffeinate -i -s bash out/regional_pilot_20261028/run_gap16.sh BANK_DIR SHIM > out/regional_pilot_20261028/run_gap16.log 2>&1 &
set -u
BANK=$1
OUT=out/regional_pilot_20261028
PY=/opt/miniconda3/envs/adni-analysis/bin/python
export PYTHONPATH=$2:. PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONWARNINGS=ignore
$PY $OUT/regional.py $BANK $OUT gap_region16 2000 500 > $OUT/gap_region16.log 2>&1 &
PID_POSITION=$!
$PY $OUT/regional.py $BANK $OUT gap_region16_double 2000 500 > $OUT/gap_region16_double.log 2>&1 &
PID_DOUBLE=$!
echo "$(date) started gap_region16 ($PID_POSITION) and gap_region16_double ($PID_DOUBLE)"

# Wait on the checkpoint file; the PID (a child of this script) only says whether the run died first.
wait_for() {
    until [ -f $OUT/$1_s2000.pt ]; do
        if ! kill -0 $2 2>/dev/null && [ ! -f $OUT/$1_s2000.pt ]; then
            echo "$(date) $1 stopped before step 2000; see $OUT/$1.log"
            exit 1
        fi
        sleep 60
    done
}
wait_for gap_region16 $PID_POSITION
wait_for gap_region16_double $PID_DOUBLE
sleep 30
echo "$(date) both at step 2000"
$PY $OUT/compare.py 2000 > $OUT/compare_2000.txt
$PY $OUT/lesion.py $BANK $OUT/lesion_s500.json $OUT/gap_region16_s500.pt $OUT/gap_region16_double_s500.pt
$PY $OUT/lesion.py $BANK $OUT/lesion_s2000.json $OUT/gap_region16_s2000.pt $OUT/gap_region16_double_s2000.pt
echo "$(date) done"
