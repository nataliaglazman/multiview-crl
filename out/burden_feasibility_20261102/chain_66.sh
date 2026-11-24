#!/bin/bash
# Start stage 66 (PARALLEL=3) once stage 65's process (pid $1) exits cleanly, and only if run_pc.sh's stage 66 block
# is the criterion check (the 23 Nov merge left a spliced block that ran stage 62's reports instead).
cd "$(dirname "$0")/../.." || exit 1
D=out/burden_feasibility_20261102
while kill -0 "$1" 2>/dev/null; do sleep 60; done
if ! tail -1 $D/run_pc_65.log | grep -q "all requested stages done"; then
    echo "$(date +%H:%M) stage 65 did not end cleanly; stage 66 not started"; exit 1
fi
block=$(sed -n '/^if wants 66; then/,/^fi$/p' $D/run_pc.sh)
if ! grep -q criterion_check.py <<< "$block" || grep -q own_directions <<< "$block" || ! bash -n $D/run_pc.sh; then
    echo "$(date +%H:%M) run_pc.sh's stage 66 block is not the criterion check (or the script fails bash -n); not started"; exit 1
fi
echo "$(date +%H:%M) stage 65 done; starting stage 66"
PARALLEL=3 exec bash $D/run_pc.sh 66 > $D/run_pc_66.log 2>&1
