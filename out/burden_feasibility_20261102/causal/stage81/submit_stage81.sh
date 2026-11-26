#!/bin/bash
# Stage 81 (long_check.py's docstring has the rules): continue stage 64's two failing 8-unit codes for 40,000 steps on
# Run:ai, one A100 job per seed (runai_warm.sh; Run:ai settings as runai_submit.sh's). Run on the PC; it copies the
# 20,000-step checkpoints to the cluster under stage81/start/ (the cluster's runs/ holds stage 74's runs of the same
# names) and submits over ssh. DRY=1 prints the commands.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=out/burden_feasibility_20261102
HOST=${HOST:-aicheadnode}
REPO=/nfs/home/nglazman/crl-2/multiview-crl
P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_abl_codebatch_predict1_c8_20k
FLAGS="--pool tail --t1key t1_04 --code-batches 500 --code-neighbours 31 --residual --jitter 2 --rotate 3 --patch-weight 0"
FLAGS="$FLAGS --content-units 8 --contrastive-loss-type infonce --predict-weight 1 --keep-checkpoints"
run() { if [ "${DRY:-0}" = 1 ]; then echo "+ $*"; else "$@"; fi; }
for seed in 42 4; do
    x=$([ $seed = 42 ] && echo "" || echo _s$seed)
    src=$D/runs/$P$x/model.pt
    start=$D/stage81/start/c8_e500_s${seed}_pc.pt
    name=${P}_plus40k_a100$x
    [ ! -e "$D/runs/$name" ] || { echo "$name exists on the PC"; exit 1; }
    ssh -o BatchMode=yes "$HOST" "test ! -e $REPO/$D/runs/$name" || { echo "$name exists on the cluster"; exit 1; }
    run ssh -o BatchMode=yes "$HOST" "mkdir -p $REPO/$D/stage81/start"
    run scp -q -o BatchMode=yes "$src" "$HOST:$REPO/$start"
    local_sum=$(sha256sum "$src" | cut -c1-64)
    if [ "${DRY:-0}" != 1 ]; then
        remote_sum=$(ssh -o BatchMode=yes "$HOST" "sha256sum $REPO/$start" | cut -c1-64)
        [ "$local_sum" = "$remote_sum" ] || { echo "checksum differs for $start"; exit 1; }
        echo "copied $start ($local_sum)"
    fi
    cmd="cd $REPO && bash $D/runai_warm.sh $name $D/bank_causal_nofield_fixedpos 40000 5000 $FLAGS --seed $seed --init-run $start"
    run ssh -o BatchMode=yes "$HOST" "runai training standard submit burden-s81-long-s$seed --project nglazman \
        --image aicregistry:5000/nglazman:multiview-crl --run-as-user --large-shm --node-type A100 --gpu-devices-request 1 \
        --cpu-core-request 16 --cpu-core-limit 32 --cpu-memory-request 64G --cpu-memory-limit 128G \
        --host-path path=/nfs,mount=/nfs,readwrite --command -- bash -c \"$cmd\""
done
