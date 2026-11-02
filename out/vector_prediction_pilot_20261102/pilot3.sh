#!/bin/bash
cd /Users/nataliaglazman/Desktop/PhD/projects/multiview-crl
export PYTHONPATH=/private/tmp/claude-502/-Users-nataliaglazman-Desktop-PhD-projects-multiview-crl/f4952e56-224b-49a5-8e79-1f811ac600a9/scratchpad/shim:. KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1
PY=/opt/miniconda3/envs/adni-analysis/bin/python
COMMON="$(cat out/vector_prediction_pilot_20261102/flags.txt) --device mps --train-steps 2000 --eval-every 500 --log-every 50 --out-dir out/vector_prediction_pilot_20261102/runs"
Q="--vector-prediction-target quantile --vector-prediction-components 8"
room() { sleep 240; until [ "$(memory_pressure 2>/dev/null | tail -1 | grep -oE '[0-9]+')" -ge 25 ]; do sleep 60; done; }
caffeinate -i -s $PY -m training.main_conv_synthetic $COMMON $Q --vector-prediction-weight 0.1 --model-id quantile_w01 > out/vector_prediction_pilot_20261102/pilot_quantile_w01.log 2>&1 &
room
caffeinate -i -s $PY -m training.main_conv_synthetic $COMMON --vector-prediction-weight 0 --model-id control > out/vector_prediction_pilot_20261102/pilot_control.log 2>&1 &
room
caffeinate -i -s $PY -m training.main_conv_synthetic $COMMON $Q --vector-prediction-weight 1 --model-id quantile_w1 > out/vector_prediction_pilot_20261102/pilot_quantile_w1.log 2>&1 &
wait
echo training done
$PY -m eval.encoder.burden_recovery --run-dirs out/vector_prediction_pilot_20261102/runs/control out/vector_prediction_pilot_20261102/runs/quantile_w01 out/vector_prediction_pilot_20261102/runs/quantile_w1 --split test --num-samples 400 --device mps --out out/vector_prediction_pilot_20261102/pilot_burden.json > out/vector_prediction_pilot_20261102/pilot_burden.log 2>&1
echo eval exit $?
