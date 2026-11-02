#!/bin/bash
# When the fine-grid arms have saved step 2000: the factor table (compare_2000.txt) and the lesion probe on their
# step-2000 checkpoints next to init and the xpred pilot's base arm at step 2000 (lesion_s2000.json).
#   nohup caffeinate -i bash out/regional_pilot_20261028/finish.sh BANK_DIR SHIM > out/regional_pilot_20261028/finish.log 2>&1 &
set -u
BANK=$1
OUT=out/regional_pilot_20261028
PY=/opt/miniconda3/envs/adni-analysis/bin/python
export PYTHONPATH=$2:. PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONWARNINGS=ignore
for ARM in region4 region8 region16 region16dot; do
    until [ -f $OUT/${ARM}_s2000.pt ]; do sleep 60; done
done
sleep 30
echo "$(date) all four at step 2000"
$PY $OUT/compare.py 2000 > $OUT/compare_2000.txt
$PY $OUT/lesion.py $BANK $OUT/lesion_s2000.json init out/xpred_pilot_20261028/base_s2000.pt \
    $OUT/region4_s2000.pt $OUT/region8_s2000.pt $OUT/region16_s2000.pt $OUT/region16dot_s2000.pt
echo "$(date) done"
