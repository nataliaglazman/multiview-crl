#!/bin/bash
# Stage 83 (rules written 26 Nov before any of its results; the user asked for it): the one-objective form of stage 82.
# Does a projector wider than the code let an 8-unit code hold all seven own parts in one training run?
#
# Why. Stage 82: an 8-unit nonlinear chart of a 10-unit code holds wherever its parent holds (4 of 4), and its linear
# chart never does (0 of 4); the curves InfoNCE wants can live in the chart's decoder. Inside one network the same
# shape is code (8 units) -> MLP -> 16 outputs -> InfoNCE: the loss's entropy acts where there is room, the MLP makes
# the curves, and everything the 16 outputs hold must pass through the 8 units. Stage 71's projector had its output at
# 8 units, so the loss still had to bend within 8 (holds 2 of 5 on the PC, 1 of 5 on A100s).
#
# Arm: the frozen recipe at 8 units (code batches every 500, 31 neighbours, residual, jitter 2 / 3 degrees, prediction
# weight 1 on the code, tails, InfoNCE tau .1) plus warm.py --projector 64 --projector-out 16 (the InfoNCE on all 16
# outputs; the first 8 start as the code, the rest as random linear mixes of it; evals and exports read the code);
# 20,000 steps; seeds 26-35, new to this bank; A100s, five seeds per job (job.sh), this PC's warm.py copied to an
# untracked stage83/ folder on the cluster (WARM, as stage 77). Runs ..._abl_codebatch_predict1_proj64out16_c8_20k_a100_sN.
# Scored as stage 76 part B (causal_bundle.py; pool_check.py detect: holds and rho1) plus style_audit.py
# (score_stage83.sh, once all ten log DONE and are copied back).
#
# Rules (fixed before any result):
#   Works if at least 9 of the 10 codes hold and no code reads its own view's gain, bias or noise level above .10
#   (style_audit, ridge or boosted). Fails if 6 or fewer hold. Otherwise unresolved.
#   Reference: the frozen recipe held 5 of 10 on A100s at seeds 6-15 (stage 76 part B).
#   Described only: rho1 by hold (these codes are not trained on rho1's quantity, so item 65's caution does not
#   apply); how far the head bends (warm.py's logged linear R2 of the head's output from the code); the own parts
#   against stage 76's ten.
# Expectation, written before: 7-9 of 10 hold; the head bends (linear R2 well under 1).
#
# Run on the PC: DRY=1 prints the commands. Copies warm.py and job.sh to the cluster (checksums checked), links
# stage83/reference to ../reference (warm.py reads reference/flags.txt beside itself), and submits two jobs.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=out/burden_feasibility_20261102
HOST=${HOST:-aicheadnode}
REPO=/nfs/home/nglazman/crl-2/multiview-crl
P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_abl_codebatch_predict1_proj64out16_c8_20k_a100
run() { if [ "${DRY:-0}" = 1 ]; then echo "+ $*"; else "$@"; fi; }
for seed in $(seq 26 35); do
    [ ! -e "$D/runs/${P}_s$seed" ] || { echo "${P}_s$seed exists on the PC"; exit 1; }
done
ssh -o BatchMode=yes "$HOST" "cd $REPO/$D && ! ls runs logs | grep -q proj64out16" \
    || { echo "proj64out16 runs or logs already exist on the cluster"; exit 1; }
run ssh -o BatchMode=yes "$HOST" "mkdir -p $REPO/$D/stage83 && ln -sfn ../reference $REPO/$D/stage83/reference"
run scp -q -o BatchMode=yes "$D/warm.py" "$HOST:$REPO/$D/stage83/warm.py"
run scp -q -o BatchMode=yes "$D/causal/stage83/job.sh" "$HOST:$REPO/$D/stage83/job.sh"
if [ "${DRY:-0}" != 1 ]; then
    for f in warm.py causal/stage83/job.sh; do
        here=$(sha256sum "$D/$f" | cut -c1-64)
        there=$(ssh -o BatchMode=yes "$HOST" "sha256sum $REPO/$D/stage83/$(basename $f)" | cut -c1-64)
        [ "$here" = "$there" ] || { echo "checksum differs for $f"; exit 1; }
        echo "copied $f ($here)"
    done
fi
for part in "a 26 27 28 29 30" "b 31 32 33 34 35"; do
    read -r tag seeds <<< "$part"
    cmd="cd $REPO && bash $D/stage83/job.sh $seeds"
    run ssh -o BatchMode=yes "$HOST" "runai training standard submit burden-s83-wproj-$tag --project nglazman \
        --image aicregistry:5000/nglazman:multiview-crl --run-as-user --large-shm --node-type A100 --gpu-devices-request 1 \
        --cpu-core-request 16 --cpu-core-limit 32 --cpu-memory-request 64G --cpu-memory-limit 128G \
        --host-path path=/nfs,mount=/nfs,readwrite --command -- bash -c \"$cmd\""
done
