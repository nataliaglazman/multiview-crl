#!/bin/bash
# Stage 60's reports run in parallel instead of one after another. The stage's own bash (pid $STAGE_PID, paused
# with SIGSTOP so it cannot start the sequential reports) keeps its five arms training; this script waits for them,
# then does what the stage block does after `finish`, with run_pc.sh's own helper functions, and ends the paused bash.
#   STAGE_PID=986574 bash run_pc_60_reports.sh        (DRY=1 lists the commands without waiting or running)
set -u
cd /home/ng24/projects/multiview-crl || exit 1
D=out/burden_feasibility_20261102
PY=${PY:-python}
DRY=${DRY:-0}
export PYTHONPATH=$D:out/regional_pilot_20261028/shim:.
export BANK_WORKERS=${BANK_WORKERS:-8}
STAGE_PID=${STAGE_PID:?set STAGE_PID to the paused stage 60 bash}

# the helpers exactly as run_pc.sh defines them
eval "$(sed -n -e '/^stage_report() {/,/^}/p' -e '/^causal() {/,/^}/p' -e '/^causal_tables() {/,/^}/p' \
    -e '/^needs_done() {/,/^}/p' "$D/run_pc.sh")"
for f in stage_report causal causal_tables needs_done; do declare -F "$f" > /dev/null || { echo "missing $f"; exit 1; }; done

end_stage() { # end the paused stage bash (SIGKILL acts on a stopped process without letting it run on)
    [ "$DRY" = 1 ] && return
    kill -0 "$STAGE_PID" 2> /dev/null && kill -9 "$STAGE_PID"
}
fail() { echo "$(date +%H:%M) $*"; end_stage; echo "$(date +%H:%M) $* (parallel reports; see $D/run_pc_60_reports.log)" >> "$D/run_pc_60.log"; exit 1; }

# the stage 60 block's runs and labels
P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos
OUT=$D/causal/stage60
sfx() { [ "$1" = 42 ] && echo "" || echo "_s$1"; }
NEW=() REF=() NEWN=() REFN=()
for seed in 42 2 3 4 5; do
    x=$(sfx "$seed")
    NEW+=(c8_harm_s$seed=${P}_abl_codebatch_predict1_harm1_c8_20k$x)
    REF+=(c8_e500_s$seed=${P}_abl_codebatch_predict1_c8_20k$x c10_e500_s$seed=${P}_abl_codebatch_predict1_c10_20k$x)
done
for pair in "${NEW[@]}"; do NEWN+=("${pair#*=}"); done
for pair in "${REF[@]}"; do REFN+=("${pair#*=}"); done
RUNS=()
for name in "${NEWN[@]}" "${REFN[@]}"; do RUNS+=("$D/runs/$name"); done
OWN=("$PY" "$D/own_directions.py" "$OUT/own_directions.json")
for pair in "${NEW[@]}" "${REF[@]}"; do OWN+=("${pair%%=*}=$D/causal/${pair#*=}.npz"); done
HOLDS=("$PY" "$D/holds.py" "$OUT" "${NEW[@]}" "${REF[@]}" --compare c8_harm:c8_e500)

if [ "$DRY" = 0 ]; then
    [ "$(ps -o stat= -p "$STAGE_PID" 2> /dev/null | cut -c1)" = T ] || { echo "pid $STAGE_PID is not a paused process"; exit 1; }
    echo "$(date +%H:%M) waiting for the five arms"
    for name in "${NEWN[@]}"; do  # as run_pc.sh's await: stop if an arm ends without DONE
        until grep -qx DONE "$D/logs/$name.log" 2> /dev/null; do
            if ! pgrep -f -- "$D/runs/$name " > /dev/null; then
                sleep 5
                grep -qx DONE "$D/logs/$name.log" 2> /dev/null && break
                fail "NOT FINISHED: $name (see $D/logs/$name.log)"
            fi
            sleep 60
        done
    done
    "$PY" "$D/summarize.py" > "$D/summary.txt" 2>&1
    echo "$(date +%H:%M) summary updated: $D/summary.txt"
    ( needs_done 60 "${NEWN[@]}" ) || fail "Stage 60's runs are not all finished"
fi

echo "$(date +%H:%M) reports in parallel: four audits, and the causal tables then own_directions"
pids=() what=()
( stage_report "$OUT" own_variation "${RUNS[@]}" --boosted ) & pids+=($!) what+=(own_variation)
( stage_report "$OUT" precision_audit "${RUNS[@]}" ) & pids+=($!) what+=(precision_audit)
( stage_report "$OUT" code_audit "${RUNS[@]}" ) & pids+=($!) what+=(code_audit)
( stage_report "$OUT" style_audit "${RUNS[@]}" ) & pids+=($!) what+=(style_audit)
(
    causal_tables "$OUT" "${NEW[@]}" "${REF[@]}" || exit 1
    if [ "$DRY" = 1 ]; then echo "${OWN[*]} > $OUT/own_directions.txt"; exit 0; fi
    echo "$(date +%H:%M) own_directions into $OUT"
    "${OWN[@]}" > "$OUT/own_directions.txt" 2>&1 || { echo "Stage 60's own_directions failed, see $OUT/own_directions.txt"; exit 1; }
) & pids+=($!) what+=("causal tables and own_directions")
failed=()
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || failed+=("${what[$i]}")
    [ "$DRY" = 1 ] || echo "$(date +%H:%M) done: ${what[$i]}"
done
[ ${#failed[@]} -eq 0 ] || fail "Stage 60 reports failed: ${failed[*]}"

if [ "$DRY" = 1 ]; then echo "${HOLDS[*]} > $OUT/holds.txt"; exit 0; fi
"${HOLDS[@]}" > "$OUT/holds.txt" 2>&1 || fail "Stage 60's holds.py failed, see $OUT/holds.txt"
end_stage
{
    echo "$(date +%H:%M) reports run in parallel by run_pc_60_reports.sh after the arms finished (log: run_pc_60_reports.log)"
    echo "Stage 60 reports in $OUT: holds.txt first, then own_directions.txt"
    echo "$(date +%H:%M) all requested stages done"
} >> "$D/run_pc_60.log"
echo "$(date +%H:%M) Stage 60 reports in $OUT: holds.txt first, then own_directions.txt"
