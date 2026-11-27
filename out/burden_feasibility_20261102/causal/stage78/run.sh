#!/usr/bin/env bash
# Stage 78 (HANDOFF section 9): the fixed bank's 8-unit codes read with the gauge's stored averages and with the gauge
# recomputed over the clean training images. Run from the repo root: bash out/burden_feasibility_20261102/causal/stage78/run.sh a|b
#   a: stage 76 part B's ten runs (seeds 6-15)   b: stage 75's 45 codes, then the summary of both parts
# Kept out of run_pc.sh because the PC's run_pc.sh and warm.py held an unresolved stash-pop conflict (26 Nov 11:01):
# LAUNCH imports a clean copy of HEAD's warm.py first (~/stage78_shim/run.py); with warm.py whole, LAUNCH="" works.
set -euo pipefail
D=out/burden_feasibility_20261102
OUT=$D/causal/stage78
PY=${PY:-$HOME/miniconda3/envs/monai_env/bin/python}
LAUNCH=${LAUNCH-$HOME/stage78_shim/run.py}
export PYTHONPATH=$HOME/stage78_shim:$D:out/regional_pilot_20261028/shim:.
export OMP_NUM_THREADS=${THREADS:-4} MKL_NUM_THREADS=${THREADS:-4} OPENBLAS_NUM_THREADS=${THREADS:-4}
P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos
py() { "$PY" $LAUNCH "$@"; }
mkdir -p "$OUT"

A=() RA=()
for seed in 6 7 8 9 10 11 12 13 14 15; do
    A+=(c8_ae500_s$seed=${P}_abl_codebatch_predict1_c8_20k_a100_s$seed)
    RA+=($D/runs/${P}_abl_codebatch_predict1_c8_20k_a100_s$seed)
done
B=() RB=()
for arm in e500:_abl_codebatch_predict1_c8_20k e100:_abl_codebatch100_predict1_c8_20k \
    cont:_abl_codebatch_predict1_c8_20k_plus10k red:_abl_from10red_predict1_c8_10k pca:_abl_from10pca_predict1_c8_10k \
    harm:_abl_codebatch_predict1_harm1_c8_20k tau1:_abl_codebatch_predict1_tau1_c8_20k \
    proj64:_abl_codebatch_predict1_proj64_c8_20k aproj64:_abl_codebatch_predict1_proj64_c8_20k_a100; do
    for seed in 42 2 3 4 5; do
        name=$P${arm#*:}
        [ "$seed" = 42 ] || name=${name}_s$seed
        B+=(c8_${arm%%:*}_s$seed=$name)
        RB+=($D/runs/$name)
    done
done

PIDS=()
part() { # part NAME DETECT_OPTIONS...: both gauges' detect reports, in the background
    local name=$1; shift
    echo "$(date +%H:%M) detect $name, stored and recomputed"
    py $D/pool_check.py detect $OUT/${name}_stored.json "$@" > $OUT/${name}_stored.txt 2>&1 &
    PIDS+=($!)
    py $D/pool_check.py detect $OUT/${name}_recomputed.json "$@" --gauge recomputed > $OUT/${name}_recomputed.txt 2>&1 &
    PIDS+=($!)
}
finish() { for pid in "${PIDS[@]}"; do wait "$pid" || { echo "a detect failed, see $OUT/*_stored.txt"; exit 1; }; done; }

case ${1:-} in
a)
    echo "$(date +%H:%M) recomputed exports, part A"
    py $D/pool_check.py export --gauge recomputed "${RA[@]}" > $OUT/export_a.log 2>&1
    part a "${A[@]}" --subsets 2 3 4 5 --rule-k 3
    echo "$(date +%H:%M) neighbourhood statistics, part A"
    py $D/gauge_check.py neighbourhood $OUT/neighbourhood.json "${RA[@]}" > $OUT/neighbourhood_a.log 2>&1
    finish
    echo "$(date +%H:%M) summary, part A"
    py $D/gauge_check.py summary $OUT/summary_a.json --part A $OUT/a_stored.json $OUT/a_recomputed.json \
        --recorded $D/causal/stage76/new_seeds.json --neighbourhood $OUT/neighbourhood.json > $OUT/summary_a.txt 2>&1
    ;;
b)
    echo "$(date +%H:%M) recomputed exports, part B"
    py $D/pool_check.py export --gauge recomputed "${RB[@]}" > $OUT/export_b.log 2>&1
    part b "${B[@]}" --stages stage64 stage65 stage60 stage70 stage71 stage71a --pairs-needed 0.75
    echo "$(date +%H:%M) neighbourhood statistics, part B"
    py $D/gauge_check.py neighbourhood $OUT/neighbourhood.json "${RB[@]}" > $OUT/neighbourhood_b.log 2>&1
    finish
    echo "$(date +%H:%M) summary, both parts"
    py $D/gauge_check.py summary $OUT/gauge_check.json --part A $OUT/a_stored.json $OUT/a_recomputed.json \
        --part B $OUT/b_stored.json $OUT/b_recomputed.json \
        --recorded $D/causal/stage76/new_seeds.json $D/causal/stage75/pool_check.json \
        --neighbourhood $OUT/neighbourhood.json > $OUT/gauge_check.txt 2>&1
    ;;
*)
    echo "usage: bash $0 a|b" >&2
    exit 2
    ;;
esac
echo "$(date +%H:%M) stage 78 part ${1} done"
