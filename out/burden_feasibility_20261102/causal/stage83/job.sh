#!/bin/bash
# Stage 83 inside one Run:ai A100 (copied to out/burden_feasibility_20261102/stage83/job.sh on the cluster): the frozen
# recipe at 8 units with a 16-wide projector, one warm.py run per seed given, side by side, each through runai_warm.sh
# with this folder's warm.py (WARM, as stage 77) and 3 CPU threads (five runs share the job's 16 cores).
#   bash out/burden_feasibility_20261102/stage83/job.sh SEED...
cd "$(dirname "$0")/../../.."
D=out/burden_feasibility_20261102
P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_abl_codebatch_predict1_proj64out16_c8_20k_a100
echo "$(date '+%F %T') stage 83 job on $(hostname): seeds $*"
for seed in "$@"; do
    THREADS=3 WARM=stage83/warm.py bash $D/runai_warm.sh ${P}_s$seed $D/bank_causal_nofield_fixedpos 20000 2000 \
        --pool tail --t1key t1_04 --code-batches 500 --code-neighbours 31 --residual --jitter 2 --rotate 3 \
        --patch-weight 0 --content-units 8 --contrastive-loss-type infonce --predict-weight 1 \
        --projector 64 --projector-out 16 --seed $seed &
done
wait
for seed in "$@"; do echo "s$seed: $(tail -1 $D/logs/${P}_s$seed.log)"; done
