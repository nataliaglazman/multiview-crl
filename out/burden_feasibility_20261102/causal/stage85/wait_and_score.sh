#!/bin/bash
# Stage 85: wait (polling every 10 minutes) until stage 81's scoring is over and all twenty runs log DONE on the
# cluster, then score (score_stage85.sh with FETCH=1). Stops without scoring if a stage 85 job shows Failed.
cd "$(dirname "$0")/../../../.."
D=out/burden_feasibility_20261102
REMOTE=/nfs/home/nglazman/crl-2/multiview-crl/$D
R=ssl_uni_jitter_nopatch_restail_t104_causal_nofield
NAMES=""
for s in 36 37 38 39 40; do NAMES="$NAMES ${R}_fixedpos_abl_codebatch_predict1_c10_20k_a100_s$s ${R}_fixedpos_predict1_c10_20k_a100_s$s"; done
for s in 46 47 48 49 50 51 52 53 54 55; do NAMES="$NAMES ${R}_abl_codebatch_predict1_c13_20k_a100_s$s"; done
while true; do
    if ssh -o BatchMode=yes -o ConnectTimeout=20 aicheadnode "runai training list --project nglazman 2>/dev/null | grep s85 | grep -q Failed"; then
        echo "$(date '+%d %H:%M') a stage 85 job failed; not scoring"
        exit 1
    fi
    done_count=$(ssh -o BatchMode=yes -o ConnectTimeout=20 aicheadnode "cd $REMOTE && n=0; for x in $NAMES; do grep -qx DONE logs/\$x.log 2>/dev/null && n=\$((n+1)); done; echo \$n" 2>/dev/null || echo "?")
    echo "$(date '+%d %H:%M') runs done: $done_count of 20"
    if [ "$done_count" = 20 ] && grep -q "stage 81 exit" $D/score_81_83.log; then break; fi
    sleep 600
done
echo "$(date '+%d %H:%M') scoring stage 85"
FETCH=1 bash $D/causal/stage85/score_stage85.sh > $D/causal/stage85/score.log 2>&1
echo "$(date '+%d %H:%M') score_stage85.sh exit $?"
tail -25 $D/causal/stage85/score.log
