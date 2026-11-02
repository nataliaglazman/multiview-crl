#!/bin/bash
cd /Users/nataliaglazman/Desktop/PhD/projects/multiview-crl
export PYTHONPATH=/private/tmp/claude-502/-Users-nataliaglazman-Desktop-PhD-projects-multiview-crl/f4952e56-224b-49a5-8e79-1f811ac600a9/scratchpad/shim:. KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1
PY=/opt/miniconda3/envs/adni-analysis/bin/python
COMMON="$(cat /private/tmp/claude-502/-Users-nataliaglazman-Desktop-PhD-projects-multiview-crl/f4952e56-224b-49a5-8e79-1f811ac600a9/scratchpad/flags.txt) --device mps --train-steps 200 --eval-every 200 --log-every 25 --out-dir /private/tmp/claude-502/-Users-nataliaglazman-Desktop-PhD-projects-multiview-crl/f4952e56-224b-49a5-8e79-1f811ac600a9/scratchpad/e2e"
$PY -m training.main_conv_synthetic $COMMON --vector-prediction-weight 0 --model-id control > /private/tmp/claude-502/-Users-nataliaglazman-Desktop-PhD-projects-multiview-crl/f4952e56-224b-49a5-8e79-1f811ac600a9/scratchpad/e2e_control.log 2>&1
echo control exit $?
$PY -m training.main_conv_synthetic $COMMON --vector-prediction-weight 1 --model-id treatment > /private/tmp/claude-502/-Users-nataliaglazman-Desktop-PhD-projects-multiview-crl/f4952e56-224b-49a5-8e79-1f811ac600a9/scratchpad/e2e_treatment.log 2>&1
echo treatment exit $?
