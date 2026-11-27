#!/bin/bash
# Stage 86 inside one Run:ai A100 (copied to out/burden_feasibility_20261102/stage86/job.sh on the cluster; rules in
# HANDOFF section 9): stage 81's two runs continued 60,000 steps from their final models, side by side, with stage 81's
# flags and seeds, through runai_warm.sh (the checkout's tracked warm.py) with 8 CPU threads each.
cd "$(dirname "$0")/../../.."
D=out/burden_feasibility_20261102
P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_abl_codebatch_predict1_c8_20k
FLAGS="--pool tail --t1key t1_04 --code-batches 500 --code-neighbours 31 --residual --jitter 2 --rotate 3 --patch-weight 0"
FLAGS="$FLAGS --content-units 8 --contrastive-loss-type infonce --predict-weight 1 --keep-checkpoints"
echo "$(date '+%F %T') stage 86 job on $(hostname)"
for seed in 42 4; do
    x=$([ $seed = 42 ] && echo "" || echo "_s$seed")
    THREADS=8 bash $D/runai_warm.sh "${P}_plus40k_plus60k_a100$x" $D/bank_causal_nofield_fixedpos 60000 10000 $FLAGS \
        --seed $seed --init-run "$D/runs/${P}_plus40k_a100$x/model.pt" &
done
wait
for x in "" _s4; do echo "${x:-s42}: $(tail -1 "$D/logs/${P}_plus40k_plus60k_a100$x.log")"; done
