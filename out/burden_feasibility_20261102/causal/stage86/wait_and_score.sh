#!/bin/bash
# Stage 86: wait (polling every 10 minutes) until both continued runs log DONE on the cluster and stage 85's watcher has
# finished (its scoring is heavy), then score (score_stage86.sh with FETCH=1). Stops without scoring if the job fails.
cd "$(dirname "$0")/../../../.."
D=out/burden_feasibility_20261102
REMOTE=/nfs/home/nglazman/crl-2/multiview-crl/$D
P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_abl_codebatch_predict1_c8_20k_plus40k_plus60k_a100
while true; do
    if ssh -o BatchMode=yes -o ConnectTimeout=20 aicheadnode "runai training list --project nglazman 2>/dev/null | grep s86 | grep -q Failed"; then
        echo "$(date '+%d %H:%M') the stage 86 job failed; not scoring"
        exit 1
    fi
    n=$(ssh -o BatchMode=yes -o ConnectTimeout=20 aicheadnode "cd $REMOTE && c=0; for x in $P ${P}_s4; do grep -qx DONE logs/\$x.log 2>/dev/null && c=\$((c+1)); done; echo \$c" 2>/dev/null || echo "?")
    busy=$(pgrep -f "stage85/wait_and_score.sh" > /dev/null && echo "stage 85 still pending or scoring" || echo "stage 85 done")
    echo "$(date '+%d %H:%M') runs done: $n of 2; $busy"
    if [ "$n" = 2 ] && [ "$busy" = "stage 85 done" ]; then break; fi
    sleep 600
done
echo "$(date '+%d %H:%M') scoring stage 86"
FETCH=1 bash $D/causal/stage86/score_stage86.sh > $D/causal/stage86/score.log 2>&1
echo "$(date '+%d %H:%M') score_stage86.sh exit $?"
tail -30 $D/causal/stage86/score.log
