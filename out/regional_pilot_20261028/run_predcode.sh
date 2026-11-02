#!/bin/bash
# Train predcode_sepadam to step 4000, then its lesion, cell-code and prediction probes on every checkpoint, into their own
# json files so a run finishing at the same time can't overwrite rows in the shared ones.
#   nohup caffeinate -i -s bash out/regional_pilot_20261028/run_predcode.sh BANK_DIR SHIM > out/regional_pilot_20261028/run_predcode.log 2>&1 &
set -u
BANK=$1
SHIM=$2
OUT=out/regional_pilot_20261028
ARM=predcode_sepadam
PY=/opt/miniconda3/envs/adni-analysis/bin/python
export PYTHONPATH=$SHIM:. PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONWARNINGS=ignore
echo "$(date) started $ARM"
$PY $OUT/regional.py $BANK $OUT $ARM 4000 500 > $OUT/$ARM.log 2>&1 || { echo "$(date) $ARM stopped; see $OUT/$ARM.log"; exit 1; }
echo "$(date) at step 4000"
CHECKPOINTS=$(for STEP in 500 1000 1500 2000 2500 3000 3500 4000; do echo $OUT/${ARM}_s$STEP.pt; done)
$PY $OUT/lesion.py $BANK $OUT/lesion_predcode.json $CHECKPOINTS || { echo "$(date) lesion probe stopped"; exit 1; }
$PY $OUT/cellcode.py $BANK $OUT/cellcode_predcode.json $CHECKPOINTS || { echo "$(date) cell-code probe stopped"; exit 1; }
$PY $OUT/xpred_cells.py $BANK $OUT/xpred_cells_predcode.json $CHECKPOINTS || { echo "$(date) prediction probe stopped"; exit 1; }
echo "$(date) done"
