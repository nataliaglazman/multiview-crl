#!/bin/bash
# Stage 86 (rules in HANDOFF section 9, written before submission): one Run:ai A100 job continuing stage 81's two runs
# (job.sh). Run on the PC: checks the names are new on both machines and that the cluster's start models are the PC's
# (sha256), copies job.sh to the cluster's untracked stage86/ folder (checksum checked) and submits. DRY=1 prints.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=out/burden_feasibility_20261102
HOST=${HOST:-aicheadnode}
REPO=/nfs/home/nglazman/crl-2/multiview-crl
P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_abl_codebatch_predict1_c8_20k
run() { if [ "${DRY:-0}" = 1 ]; then echo "+ $*"; else "$@"; fi; }
! ls "$D/runs" "$D/logs" | grep -q plus60k || { echo "stage 86's runs or logs already exist on the PC"; exit 1; }
ssh -o BatchMode=yes "$HOST" "cd $REPO/$D && ! ls runs logs | grep -q plus60k" \
    || { echo "stage 86's runs or logs already exist on the cluster"; exit 1; }
for x in "" _s4; do
    here=$(sha256sum "$D/runs/${P}_plus40k_a100$x/model.pt" | cut -c1-64)
    there=$(ssh -o BatchMode=yes "$HOST" "sha256sum $REPO/$D/runs/${P}_plus40k_a100$x/model.pt" | cut -c1-64)
    [ "$here" = "$there" ] || { echo "the start model differs for ${x:-seed 42}"; exit 1; }
    echo "start ${x:-s42}: $here"
done
run ssh -o BatchMode=yes "$HOST" "mkdir -p $REPO/$D/stage86"
run scp -q -o BatchMode=yes "$D/causal/stage86/job.sh" "$HOST:$REPO/$D/stage86/job.sh"
if [ "${DRY:-0}" != 1 ]; then
    here=$(sha256sum "$D/causal/stage86/job.sh" | cut -c1-64)
    there=$(ssh -o BatchMode=yes "$HOST" "sha256sum $REPO/$D/stage86/job.sh" | cut -c1-64)
    [ "$here" = "$there" ] || { echo "checksum differs for job.sh"; exit 1; }
    echo "copied job.sh ($here)"
fi
cmd="cd $REPO && bash $D/stage86/job.sh"
run ssh -o BatchMode=yes "$HOST" "runai training standard submit burden-s86-cont --project nglazman \
    --image aicregistry:5000/nglazman:multiview-crl --run-as-user --large-shm --node-type A100 --gpu-devices-request 1 \
    --cpu-core-request 16 --cpu-core-limit 32 --cpu-memory-request 64G --cpu-memory-limit 128G \
    --host-path path=/nfs,mount=/nfs,readwrite --command -- bash -c \"$cmd\""
