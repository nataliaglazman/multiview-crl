#!/bin/bash
# Start stage 65 once stage 64's process (pid given) exits, if stage 64 ended cleanly.
cd /home/ng24/projects/multiview-crl || exit 1
export PATH=/home/ng24/miniconda3/envs/monai_env/bin:$PATH
while kill -0 "$1" 2>/dev/null; do sleep 60; done
if tail -1 out/burden_feasibility_20261102/run_pc_64.log | grep -q "all requested stages done"; then
    echo "$(date +%H:%M) stage 64 done; starting stage 65"
    exec bash out/burden_feasibility_20261102/run_pc.sh 65 > out/burden_feasibility_20261102/run_pc_65.log 2>&1
else
    echo "$(date +%H:%M) stage 64 did not end cleanly; stage 65 not started"
fi
