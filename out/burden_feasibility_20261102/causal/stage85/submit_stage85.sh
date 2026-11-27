#!/bin/bash
# Stage 85 (rules in HANDOFF section 9, written before submission): four Run:ai A100 jobs of five seeds each (job.sh):
# F10cb and F10uni at seeds 36-40 on the fixed-position bank, D13cb at seeds 46-55 on the drawn bank. Run on the PC:
# it copies job.sh to the cluster's untracked stage85/ folder (checksum checked) and submits over ssh; the jobs use the
# checkout's tracked warm.py. DRY=1 prints the commands.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=out/burden_feasibility_20261102
HOST=${HOST:-aicheadnode}
REPO=/nfs/home/nglazman/crl-2/multiview-crl
NEW='c10_20k_a100_s(3[6-9]|40)|c13_20k'
run() { if [ "${DRY:-0}" = 1 ]; then echo "+ $*"; else "$@"; fi; }
! ls "$D/runs" "$D/logs" | grep -qE "$NEW" || { echo "stage 85's runs or logs already exist on the PC"; exit 1; }
ssh -o BatchMode=yes "$HOST" "cd $REPO/$D && ! ls runs logs | grep -qE '$NEW'" \
    || { echo "stage 85's runs or logs already exist on the cluster"; exit 1; }
run ssh -o BatchMode=yes "$HOST" "mkdir -p $REPO/$D/stage85"
run scp -q -o BatchMode=yes "$D/causal/stage85/job.sh" "$HOST:$REPO/$D/stage85/job.sh"
if [ "${DRY:-0}" != 1 ]; then
    here=$(sha256sum "$D/causal/stage85/job.sh" | cut -c1-64)
    there=$(ssh -o BatchMode=yes "$HOST" "sha256sum $REPO/$D/stage85/job.sh" | cut -c1-64)
    [ "$here" = "$there" ] || { echo "checksum differs for job.sh"; exit 1; }
    echo "copied job.sh ($here)"
fi
for part in "acb acb 36 37 38 39 40" "auni auni 36 37 38 39 40" "dcb-a dcb 46 47 48 49 50" "dcb-b dcb 51 52 53 54 55"; do
    read -r tag arm seeds <<< "$part"
    cmd="cd $REPO && bash $D/stage85/job.sh $arm $seeds"
    run ssh -o BatchMode=yes "$HOST" "runai training standard submit burden-s85-$tag --project nglazman \
        --image aicregistry:5000/nglazman:multiview-crl --run-as-user --large-shm --node-type A100 --gpu-devices-request 1 \
        --cpu-core-request 16 --cpu-core-limit 32 --cpu-memory-request 64G --cpu-memory-limit 128G \
        --host-path path=/nfs,mount=/nfs,readwrite --command -- bash -c \"$cmd\""
done
