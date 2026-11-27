#!/bin/bash
# Stage 85 inside one Run:ai A100 (copied to out/burden_feasibility_20261102/stage85/job.sh on the cluster; rules in
# HANDOFF section 9): one arm's seeds side by side, each a warm.py run through runai_warm.sh with the checkout's tracked
# warm.py and 3 CPU threads (five runs share the job's 16 cores).
#   bash out/burden_feasibility_20261102/stage85/job.sh acb|auni|dcb SEED...
cd "$(dirname "$0")/../../.."
D=out/burden_feasibility_20261102
R=ssl_uni_jitter_nopatch_restail_t104_causal_nofield
FLAGS="--pool tail --t1key t1_04 --residual --jitter 2 --rotate 3 --patch-weight 0 --contrastive-loss-type infonce"
FLAGS="$FLAGS --predict-weight 1"
ARM=$1
shift
case $ARM in
    acb)
        BANK=$D/bank_causal_nofield_fixedpos
        P=${R}_fixedpos_abl_codebatch_predict1_c10_20k_a100
        FLAGS="$FLAGS --content-units 10 --code-batches 500 --code-neighbours 31"
        ;;
    auni)
        BANK=$D/bank_causal_nofield_fixedpos
        P=${R}_fixedpos_predict1_c10_20k_a100
        FLAGS="$FLAGS --content-units 10 --uniform"
        ;;
    dcb)
        BANK=$D/bank_causal_nofield
        P=${R}_abl_codebatch_predict1_c13_20k_a100
        FLAGS="$FLAGS --content-units 13 --code-batches 500 --code-neighbours 31"
        ;;
    *)
        echo "unknown arm $ARM"
        exit 1
        ;;
esac
echo "$(date '+%F %T') stage 85 job on $(hostname): arm $ARM, seeds $*"
for seed in "$@"; do
    THREADS=3 bash $D/runai_warm.sh "${P}_s$seed" "$BANK" 20000 2000 $FLAGS --seed "$seed" &
done
wait
for seed in "$@"; do echo "s$seed: $(tail -1 "$D/logs/${P}_s$seed.log")"; done
