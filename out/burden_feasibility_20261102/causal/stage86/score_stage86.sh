#!/bin/bash
# Stage 86's scoring on the PC (rules in HANDOFF section 9), once both runs log DONE on the cluster: FETCH=1 first
# copies their runs and logs back. Then a pseudo-run per kept checkpoint (continue_check.py points), bundles,
# pool_check.py detect, and continue_check.py summary (the rules).
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=out/burden_feasibility_20261102
S=$D/causal/stage86
PY=${PY:-$HOME/miniconda3/envs/monai_env/bin/python}
export PYTHONPATH=$D:out/regional_pilot_20261028/shim:. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_abl_codebatch_predict1_c8_20k_plus40k_plus60k_a100
REMOTE=/nfs/home/nglazman/crl-2/multiview-crl/$D
RUNS=()
for n in "$P" "${P}_s4"; do
    if [ "${FETCH:-0}" = 1 ] && [ ! -f "$D/runs/$n/model.pt" ]; then
        ssh -o BatchMode=yes aicheadnode "grep -qx DONE $REMOTE/logs/$n.log" || { echo "$n has not finished"; exit 1; }
        scp -rq -o BatchMode=yes "aicheadnode:$REMOTE/runs/$n" "$D/runs/"
        scp -q -o BatchMode=yes "aicheadnode:$REMOTE/logs/$n.log" "$D/logs/"
    fi
    grep -qx DONE "$D/logs/$n.log" || { echo "$n is not finished or not copied back"; exit 1; }
    RUNS+=("$D/runs/$n")
done
read -r -a PAIRS <<< "$("$PY" "$D/continue_check.py" points "${RUNS[@]}")"
for pair in "${PAIRS[@]}"; do
    run=${pair#*=}
    name=$(basename "$run")
    if [ -f "$D/causal/$name.npz" ] && [ "$D/causal/$name.npz" -nt "$run/model.pt" ]; then continue; fi
    echo "$(date +%H:%M) bundle $name"
    nice -n 10 "$PY" "$D/causal_bundle.py" "$run" "$D/causal/$name.npz" > "$S/bundle_$name.log" 2>&1
done
echo "$(date +%H:%M) pool_check detect into $S/detect.txt"
nice -n 10 "$PY" "$D/pool_check.py" detect "$S/detect.json" "${PAIRS[@]}" > "$S/detect.txt" 2>&1
"$PY" "$D/continue_check.py" summary "$S/detect.json" | tee "$S/summary.txt"
