#!/bin/bash
# Stage 85's scoring on the PC (rules in HANDOFF section 9), once all twenty runs log DONE on the cluster: FETCH=1 first
# copies their runs and logs back. Then bundles, pool_check.py detect (holds and rho1), panel_check.py panel (cross-seed
# scores), every wide code charted at |C| - 1, |C| and |C| + 1 units with certificates (chart_check.py, two processes
# at a time on the GPU), style_audit.py on the drawn bank's codes, and panel_check.py pipeline (rules E1-E3).
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=out/burden_feasibility_20261102
S=$D/causal/stage85
PY=${PY:-$HOME/miniconda3/envs/monai_env/bin/python}
export PYTHONPATH=$D:out/regional_pilot_20261028/shim:. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
R=ssl_uni_jitter_nopatch_restail_t104_causal_nofield
REMOTE=/nfs/home/nglazman/crl-2/multiview-crl/$D
declare -A PREFIX=([acb]=${R}_fixedpos_abl_codebatch_predict1_c10_20k_a100 [auni]=${R}_fixedpos_predict1_c10_20k_a100
                   [dcb]=${R}_abl_codebatch_predict1_c13_20k_a100)
declare -A LABEL=([acb]=c10_acb [auni]=c10_auni [dcb]=c13_dcb)
declare -A SEEDS=([acb]="36 37 38 39 40" [auni]="36 37 38 39 40" [dcb]="46 47 48 49 50 51 52 53 54 55")
declare -A WIDTHS=([acb]="6 7 8" [auni]="6 7 8" [dcb]="9 10 11")
ARMS="acb auni dcb"
PAIRS=()
for arm in $ARMS; do
    for seed in ${SEEDS[$arm]}; do
        n=${PREFIX[$arm]}_s$seed
        if [ "${FETCH:-0}" = 1 ] && [ ! -f "$D/runs/$n/model.pt" ]; then
            ssh -o BatchMode=yes aicheadnode "grep -qx DONE $REMOTE/logs/$n.log" || { echo "$n has not finished"; exit 1; }
            scp -rq -o BatchMode=yes "aicheadnode:$REMOTE/runs/$n" "$D/runs/"
            scp -q -o BatchMode=yes "aicheadnode:$REMOTE/logs/$n.log" "$D/logs/"
        fi
        grep -qx DONE "$D/logs/$n.log" || { echo "$n is not finished or not copied back"; exit 1; }
        if [ ! "$D/causal/$n.npz" -nt "$D/runs/$n/model.pt" ]; then
            echo "$(date +%H:%M) bundle $n"
            nice -n 10 "$PY" "$D/causal_bundle.py" "$D/runs/$n" "$D/causal/$n.npz" > "$S/bundle_${LABEL[$arm]}_s$seed.log" 2>&1
        fi
        PAIRS+=("${LABEL[$arm]}_s$seed=$n")
    done
done
echo "$(date +%H:%M) pool_check detect"
"$PY" "$D/pool_check.py" detect "$S/detect.json" "${PAIRS[@]}" --subsets 3 > "$S/detect.txt" 2>&1
echo "$(date +%H:%M) panel_check panel"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    "$PY" "$D/panel_check.py" panel "$S/panel.json" --detect "$S/detect.json" > "$S/panel.txt" 2> "$S/panel.log"
chart() {  # ARM WIDTH: every wide code of the arm charted at WIDTH units
    local arm=$1 u=$2 runs=()
    for seed in ${SEEDS[$arm]}; do runs+=("u${u}_${LABEL[$arm]}_s$seed=${PREFIX[$arm]}_s$seed"); done
    OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 "$PY" "$D/chart_check.py" "$S/chart_${LABEL[$arm]}_u$u.json" \
        --units "$u" --certificate --runs "${runs[@]}" > "$S/chart_${LABEL[$arm]}_u$u.txt" 2>&1
}
echo "$(date +%H:%M) charts"
i=0
for arm in dcb acb auni; do
    for u in ${WIDTHS[$arm]}; do
        chart "$arm" "$u" &
        i=$((i + 1))
        if [ $((i % 2)) = 0 ]; then wait; fi
    done
done
wait
echo "$(date +%H:%M) style_audit (drawn bank)"
DRAWN=()
for seed in ${SEEDS[dcb]}; do DRAWN+=("$D/runs/${PREFIX[dcb]}_s$seed"); done
"$PY" "$D/style_audit.py" "$S/style_audit.json" "${DRAWN[@]}" > "$S/style_audit.txt" 2>&1
echo "$(date +%H:%M) pipeline and rules"
CHARTS=()
for arm in $ARMS; do for u in ${WIDTHS[$arm]}; do CHARTS+=("$S/chart_${LABEL[$arm]}_u$u.json"); done; done
"$PY" "$D/panel_check.py" pipeline "$S/pipeline.json" --detect "$S/detect.json" --panel "$S/panel.json" \
    --charts "${CHARTS[@]}" | tee "$S/pipeline.txt"
echo "$(date +%H:%M) stage 85 scored"
