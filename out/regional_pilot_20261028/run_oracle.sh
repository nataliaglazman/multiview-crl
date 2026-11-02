#!/bin/bash
# Train region16_oracle to step 2000, then its lesion and centred-map probes on the four checkpoints, into their own
# json files so a run finishing at the same time can't overwrite rows in the shared ones.
#   nohup caffeinate -i -s bash out/regional_pilot_20261028/run_oracle.sh BANK_DIR SHIM > out/regional_pilot_20261028/run_oracle.log 2>&1 &
set -u
BANK=$1
SHIM=$2
OUT=out/regional_pilot_20261028
ARM=region16_oracle
PY=/opt/miniconda3/envs/adni-analysis/bin/python
export PYTHONPATH=$SHIM:. PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONWARNINGS=ignore
echo "$(date) started $ARM"
$PY $OUT/regional.py $BANK $OUT $ARM 2000 500 > $OUT/$ARM.log 2>&1 || { echo "$(date) $ARM stopped; see $OUT/$ARM.log"; exit 1; }
echo "$(date) at step 2000"
CHECKPOINTS=$(for STEP in 500 1000 1500 2000; do echo $OUT/${ARM}_s$STEP.pt; done)
$PY $OUT/lesion.py $BANK $OUT/lesion_oracle.json $CHECKPOINTS || { echo "$(date) lesion probe stopped"; exit 1; }
$PY $OUT/cellcode.py $BANK $OUT/cellcode_oracle.json $CHECKPOINTS || { echo "$(date) cell-code probe stopped"; exit 1; }
echo "$(date) done"
