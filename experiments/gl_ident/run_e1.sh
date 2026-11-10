#!/usr/bin/env bash
# E1 on this machine (local runner): local / global / pooled x seeds 0-2, three streams in parallel.
# Seed 0 also scores the untrained (step-0) model as a floor. "e1lit" repeats seed 0 of local and global
# with the literal spec code size (no extra unit for the cosine sphere) to measure what that unit buys.
#   bash experiments/gl_ident/run_e1.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/opt/miniconda3/envs/HCD_env/bin/python}
CFG=experiments/gl_ident/configs/e1.yaml
LOGS=experiments/gl_ident/results/logs
mkdir -p "$LOGS"

run() { PYTHONPATH=. "$PY" -W ignore -m experiments.gl_ident.train --config "$CFG" "$@"; }

stream() {
    local kind=$1
    for seed in 0 1 2; do
        if [[ $seed == 0 ]]; then
            run --set model.kind="$kind" train.seed=$seed --eval-untrained
        else
            run --set model.kind="$kind" train.seed=$seed
        fi
    done
    if [[ $kind != pooled ]]; then
        run --set model.kind="$kind" train.seed=0 model.sphere_extra_dim=False name=e1lit
    fi
}

stream pooled >"$LOGS/e1_pooled.log" 2>&1 &
stream local >"$LOGS/e1_local.log" 2>&1 &
stream global >"$LOGS/e1_global.log" 2>&1 &
wait
echo "E1 done"
