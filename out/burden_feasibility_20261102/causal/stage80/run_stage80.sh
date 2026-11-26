#!/bin/bash
# Stage 80 (head_refit.py's docstring has the rules): fresh heads on the 15 frozen recipe backbones, control and suff,
# one at a time, then bundles, pool_check.py detect and the rule summary. Finished heads are skipped, so a rerun
# continues. Run from anywhere: bash out/burden_feasibility_20261102/causal/stage80/run_stage80.sh
# SEEDS (default all 15) picks the backbones; TRAIN_ONLY=1 stops after training (parallel workers take disjoint
# SEEDS); TAG names a partial scoring's files (detect$TAG, summary$TAG), so it never touches the full one.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=out/burden_feasibility_20261102
S=$D/causal/stage80
PY=${PY:-$HOME/miniconda3/envs/monai_env/bin/python}
export PYTHONPATH=$D:out/regional_pilot_20261028/shim:. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_abl_codebatch_predict1_c8_20k
mkdir -p "$S/runs" "$S/logs"
source_of() { case $1 in 42) echo $P ;; [2-5]) echo ${P}_s$1 ;; *) echo ${P}_a100_s$1 ;; esac; }
PAIRS=()
for seed in ${SEEDS:-42 2 3 4 5 6 7 8 9 10 11 12 13 14 15}; do
    src=$(source_of "$seed")
    PAIRS+=("c8_src_s$seed=$D/runs/$src")
    for arm in control suff; do
        name=${src}_refit80_$arm
        label=c8_$([ $arm = control ] && echo ctrl || echo suff)_s$seed
        PAIRS+=("$label=$S/runs/$name")
        if [ -f "$S/runs/$name/model.pt" ]; then echo "skip $name"; continue; fi
        echo "$(date +%H:%M) train $label"
        nice -n 10 "$PY" "$D/head_refit.py" train "$D/runs/$src" "$S/runs/$name" $arm > "$S/logs/$name.log" 2>&1 \
            || { echo "failed: $S/logs/$name.log"; exit 1; }
        tail -2 "$S/logs/$name.log" | head -1 | cut -c1-160
    done
done
if [ "${TRAIN_ONLY:-0}" = 1 ]; then echo "$(date +%H:%M) trained: seeds ${SEEDS:-all}"; exit 0; fi
TAG=${TAG:-}
for pair in "${PAIRS[@]}"; do
    run=${pair#*=}
    name=$(basename "$run")
    [ -f "$D/causal/$name.npz" ] && [ "$D/causal/$name.npz" -nt "$run/model.pt" ] && continue
    echo "$(date +%H:%M) bundle $name"
    nice -n 10 "$PY" "$D/causal_bundle.py" "$run" "$D/causal/$name.npz" > "$S/logs/$name.bundle.log" 2>&1 \
        || { echo "bundle failed: $S/logs/$name.bundle.log"; exit 1; }
done
echo "$(date +%H:%M) pool_check detect into $S/detect$TAG.txt"
nice -n 10 "$PY" "$D/pool_check.py" detect "$S/detect$TAG.json" "${PAIRS[@]}" --subsets 3 > "$S/detect$TAG.txt" 2>&1 \
    || { echo "detect failed: $S/detect$TAG.txt"; exit 1; }
"$PY" "$D/head_refit.py" summary "$S/detect$TAG.json" > "$S/summary$TAG.txt"
cat "$S/summary$TAG.txt"
echo "$(date +%H:%M) stage 80 done"
