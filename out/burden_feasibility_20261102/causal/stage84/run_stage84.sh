#!/bin/bash
# Stage 84 (rules in HANDOFF section 9, written before panel_check.py and any result): A, cross-seed completeness
# (panel_check.py completeness, CPU); B, the five 10-unit parents charted at 6 and 5 units (chart_check.py, GPU),
# then the width rule (panel_check.py width, with stage 82b's 7- and 8-unit certificates). No training.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=out/burden_feasibility_20261102
S=$D/causal/stage84
PY=${PY:-$HOME/miniconda3/envs/monai_env/bin/python}
export PYTHONPATH=$D:out/regional_pilot_20261028/shim:.
P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_abl_codebatch_predict1_c10_20k
mkdir -p "$S"
(
    export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
    for u in 6 5; do
        echo "$(date +%H:%M) charts at $u units"
        "$PY" "$D/chart_check.py" "$S/chart_u$u.json" --units "$u" --certificate --runs "w${u}_s42=$P" \
            "w${u}_s2=${P}_s2" "w${u}_s3=${P}_s3" "w${u}_s4=${P}_s4" "w${u}_s5=${P}_s5" > "$S/chart_u$u.txt" 2>&1
    done
) > "$S/charts.log" 2>&1 &
CHARTS=$!
echo "$(date +%H:%M) A: cross-seed completeness into $S/panel.txt"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    "$PY" "$D/panel_check.py" completeness "$S/panel.json" > "$S/panel.txt" 2> "$S/panel.log"
echo "$(date +%H:%M) A done; waiting for B's charts"
wait $CHARTS
"$PY" "$D/panel_check.py" width "$S/width.json" | tee "$S/width.txt"
echo "$(date +%H:%M) stage 84 done"
