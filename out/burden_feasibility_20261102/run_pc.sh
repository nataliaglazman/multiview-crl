#!/bin/bash
# Burden feasibility on a CUDA machine. Run from anywhere inside the repository, on the branch that holds this folder.
#
#   bash out/burden_feasibility_20261102/run_pc.sh            all stages, in order
#   bash out/burden_feasibility_20261102/run_pc.sh 5 6        only those stages
#
# Stage 1 Can the encoder learn it from labels? Dense-label detectors, then the exact vector against pooling changes.
# Stage 2 Warm starts from the detector, tail pooling at T1 0.4, labels on temporal sigma 0.09 as a positive control.
# Stage 3 The real test: self-supervised, no labels, end to end. GAP control against tail pooling.
# Stage 4 Is it size? The same recipes with 2x and 4x lesion volume.
# Stage 5 T1 lesion value 0.4: each view's normative residual as a second input channel, no labels, with and without
#         tail pooling, and with labels through the same input as a ceiling.
# Stage 6 Realism: the same at T1 0.4 with the nuisance fields on, so anatomy is not unusually low-dimensional.
# Stage 7 Two follow-ups. Gauge control: GAP through the tail arms' BatchNorm gauge and head with no tails, with and
#         without labels, to separate the tails from the gauge. Queue: QUEUE earlier subjects as extra negatives from
#         a momentum encoder, to test whether the 32-subject batch is why each arm keeps only some factors.
# Stage 8 What else drives the trade-off. An audit of every finished self-supervised code, with no training: what it
#         holds, the lesion centre included (never scored before), which shared directions no known factor explains,
#         and how well it tells subjects apart in uniform and in anatomy-matched sets. Then the batch rule, which every
#         arm so far shared: the oracle neighbourhoods match exactly the six anatomy factors we score. Uniform batches,
#         uniform with the queue, and neighbourhoods rebuilt every 500 steps from the model's own code (no labels).
# Stage 9 The tail arms' BatchNorm gauge uses batch statistics in training and running ones at test, and neighbourhood
#         batches make the two differ. First the audit adds a mode check to every finished run (no training): test
#         retrieval with each statistic, and retrieval on training subjects. Then --gauge running, one normalization
#         in both: a 6,000-step control against ssl_restail_t104, and 20,000-step code-batch arms (alone, with the
#         queue, and with the queue and the nuisance fields on), to see whether anatomy keeps returning while burden
#         holds.
# Stage 10 Levers against feature suppression (InfoNCE stops paying once the batch is told apart), each alone on the
#         best label-free arm so far (residual + tail, code batches, BatchNorm gauge, 6,000 steps), to compare with
#         ssl_code_restail_t104: a Barlow Twins term on the code (it pays until every unit carries shared signal),
#         group dropout before the head (the code must identify subjects from GAP alone or the tails alone), and
#         per-view jitter (no voxel-exact match between the views to memorise).
# Stage 11 Why jitter works. Separate shifts of the two views also misalign the 16^3 patch term, so the patch term off
#         without jitter and with it. Then jitter with group dropout (the only lever that moved temporal) and jitter
#         with a second seed. 6,000 steps each on the stage 10 arm, against ssl_code_restail_t104 and
#         ssl_code_jitter_restail_t104.
# Stage 12 The best arm, jitter with the patch term off: 20,000 steps (does temporal arrive with training, and does
#         memorisation creep back on a fixed set of subjects?), and the jitter dose at 6,000 steps, half (1 voxel,
#         1.5 degrees) and double (4 voxels, 6 degrees), against ssl_code_jitter_nopatch_restail_t104.
# Stage 13 Why jitter works. Independent jitter does two things: the two views of a pair stop matching voxel for voxel,
#         and a training subject never looks the same twice. One transform per subject for both views (--shared-jitter)
#         keeps the second and removes the first. That at 4 voxels, 6 degrees, and the independent 4-voxel arm with a
#         second seed, 6,000 steps each, against ssl_code_jitter4_nopatch_restail_t104. Also the independent 4-voxel
#         arm at 20,000 steps, against ssl_code_jitter_nopatch_restail_t104_20k.
#
# PARALLEL arms share the GPU (default 3; each takes roughly 2-4 GB, check nvidia-smi). PY picks the interpreter.
# Finished arms are skipped, and so are arms still running outside this script (started by hand), so rerunning after a
# crash or an out-of-memory error continues where it stopped.
# DRY=1 prints the commands without running them.

set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
D=out/burden_feasibility_20261102
PY=${PY:-python}
PARALLEL=${PARALLEL:-3}
BATCH=${BATCH:-32}
DRY=${DRY:-0}
STAGES=${*:-1 2 3 4 5 6 7 8 9 10 11 12 13}
QUEUE=${QUEUE:-1024}
export PYTHONPATH=$D:out/regional_pilot_20261028/shim:.
export BANK_WORKERS=${BANK_WORKERS:-8}
mkdir -p "$D/logs" "$D/runs"

if [ "$DRY" = 0 ]; then
    "$PY" -c "import torch, sklearn, scipy, monai; assert torch.cuda.is_available(), 'CUDA is not available'; print('GPU', torch.cuda.get_device_name(0))" || exit 1
fi

wants() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

bank() { # bank DIR [bank.py options]
    local dir=$1; shift
    if [ "$DRY" = 1 ]; then echo "$PY $D/bank.py $dir $*"; return; fi
    if grep -q "BANK DONE" "$dir.log" 2>/dev/null; then return; fi
    echo "$(date +%H:%M) render $dir"
    "$PY" "$D/bank.py" "$dir" "$@" > "$dir.log" 2>&1 || { echo "bank failed, see $dir.log"; exit 1; }
}

residual_bank() { # residual_bank BANK KEY...
    local dir=$1; shift
    if [ "$DRY" = 1 ]; then echo "$PY $D/residual_bank.py $dir $*"; return; fi
    if grep -q "RESIDUAL DONE" "$dir.residual.log" 2>/dev/null; then return; fi
    echo "$(date +%H:%M) residuals for $dir"
    "$PY" "$D/residual_bank.py" "$dir" "$@" > "$dir.residual.log" 2>&1 || { echo "residual bank failed, see $dir.residual.log"; exit 1; }
}

arm() { # arm NAME COMMAND...
    local name=$1; shift
    if [ "$DRY" = 1 ]; then echo "$*"; return; fi
    if grep -qE '^(TEST|DONE)' "$D/logs/$name.log" 2>/dev/null; then echo "skip $name"; return; fi
    if pgrep -f -- "$D/runs/$name " > /dev/null; then echo "skip $name, already running"; return; fi
    rm -rf "${D:?}/runs/$name"
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
    echo "$(date +%H:%M) start $name"
    "$@" > "$D/logs/$name.log" 2>&1 &
}

finish() { # wait for this stage's arms, then report any that did not finish
    [ "$DRY" = 1 ] && return
    wait
    for name in "$@"; do
        grep -qE '^(TEST|DONE)' "$D/logs/$name.log" 2>/dev/null || echo "NOT FINISHED: $name (see $D/logs/$name.log)"
    done
    "$PY" "$D/summarize.py" > "$D/summary.txt" 2>&1
    echo "$(date +%H:%M) summary updated: $D/summary.txt"
}

sup() { # sup NAME BANK ARM T1KEY STEPS [options]
    local name=$1 where=$2 kind=$3 key=$4 steps=$5; shift 5
    arm "$name" "$PY" "$D/sup.py" "$where" "$D/runs/$name" "$kind" "$key" "$steps" 500 --batch "$BATCH" "$@"
}

ssl() { # ssl NAME BANK POOL T1KEY STEPS [options]; EVERY=N before the call sets the eval interval (default 1000)
    local name=$1 where=$2 pool=$3 key=$4 steps=$5; shift 5
    arm "$name" "$PY" "$D/warm.py" "$where" "$D/runs/$name" cold "$steps" "${EVERY:-1000}" --pool "$pool" --t1key "$key" "$@"
}

audit() { # audit every finished self-supervised run; runs already audited at the same step are reused from audit.json
    if [ "$DRY" = 1 ]; then echo "$PY $D/code_audit.py $D/audit.json $D/runs/ssl_*"; return; fi
    echo "$(date +%H:%M) audit of finished self-supervised codes"
    "$PY" "$D/code_audit.py" "$D/audit.json" "$D"/runs/ssl_* > "$D/audit.txt" 2>&1 || echo "audit failed, see $D/audit.txt"
}

MAIN=$D/bank
bank "$MAIN"

if wants 1; then
    sup dense_t100 "$MAIN" dense t1_00 3000
    sup dense_t104 "$MAIN" dense t1_04 3000
    sup vector_t100 "$MAIN" vector t1_00 6000
    sup tail_t100 "$MAIN" tail t1_00 6000
    sup cellsum_t100 "$MAIN" cellsum t1_00 6000
    sup normtail_t100 "$MAIN" normtail t1_00 6000
    finish dense_t100 dense_t104 vector_t100 tail_t100 cellsum_t100 normtail_t100
    if [ "$DRY" = 0 ]; then
        "$PY" "$D/dense_eval.py" "$MAIN" "$D/runs/dense_t100" t1_00 400 > "$D/logs/dense_eval_t100.log" 2>&1
        "$PY" "$D/dense_eval.py" "$MAIN" "$D/runs/dense_t104" t1_04 400 > "$D/logs/dense_eval_t104.log" 2>&1
        "$PY" "$D/summarize.py" > "$D/summary.txt" 2>&1
    fi
fi

if wants 2; then
    bank "$D/bank_temporal009" --t1-values 0.0 --temporal-sigma 0.09
    sup warmvector_t100 "$MAIN" vector t1_00 2000 --init "$D/runs/dense_t100/model.pt"
    sup warmcell_t100 "$MAIN" cellsum t1_00 2000 --init "$D/runs/dense_t100/model.pt"
    sup tail_t104 "$MAIN" tail t1_04 6000
    sup vector_temporal009 "$D/bank_temporal009" vector t1_00 6000 --target temporal
    finish warmvector_t100 warmcell_t100 tail_t104 vector_temporal009
fi

if wants 3; then
    ssl ssl_gap_t100 "$MAIN" gap t1_00 6000
    ssl ssl_tail_t100 "$MAIN" tail t1_00 6000
    ssl ssl_tail_t104 "$MAIN" tail t1_04 6000
    finish ssl_gap_t100 ssl_tail_t100 ssl_tail_t104
fi

if wants 4; then
    bank "$D/bank_r0126" --t1-values 0.0 --radius 0.126
    bank "$D/bank_r0159" --t1-values 0.0 --radius 0.159
    sup vector_r0126 "$D/bank_r0126" vector t1_00 6000
    sup vector_r0159 "$D/bank_r0159" vector t1_00 6000
    ssl ssl_gap_r0126 "$D/bank_r0126" gap t1_00 6000
    ssl ssl_gap_r0159 "$D/bank_r0159" gap t1_00 6000
    finish vector_r0126 vector_r0159 ssl_gap_r0126 ssl_gap_r0159
fi

if wants 5; then
    residual_bank "$MAIN" t1_04 t1_00 flair
    sup normtail_res_t104 "$MAIN" normtail t1_04 6000 --residual
    ssl ssl_restail_t104 "$MAIN" tail t1_04 6000 --residual
    ssl ssl_resgap_t104 "$MAIN" gap t1_04 6000 --residual
    ssl ssl_restail_t100 "$MAIN" tail t1_00 6000 --residual
    finish normtail_res_t104 ssl_restail_t104 ssl_resgap_t104 ssl_restail_t100
fi

if wants 6; then
    NOISY=$D/bank_nuisance
    bank "$NOISY" --nuisance
    if [ "$DRY" = 1 ]; then
        echo "$PY $D/residual.py $NOISY $NOISY/residual.json"
    elif [ ! -f "$NOISY/residual.json" ]; then
        echo "$(date +%H:%M) fixed-filter residual read, nuisance fields on"
        "$PY" "$D/residual.py" "$NOISY" "$NOISY/residual.json" > "$D/logs/residual_nuisance.log" 2>&1
    fi
    residual_bank "$NOISY" t1_04 flair
    ssl ssl_restail_t104_nuisance "$NOISY" tail t1_04 6000 --residual
    ssl ssl_gap_t104_nuisance "$NOISY" gap t1_04 6000
    finish ssl_restail_t104_nuisance ssl_gap_t104_nuisance
fi

if wants 7; then
    NOISY=$D/bank_nuisance
    bank "$NOISY" --nuisance
    residual_bank "$MAIN" t1_04 t1_00 flair
    residual_bank "$NOISY" t1_04 flair
    sup gapnorm_t100 "$MAIN" gapnorm t1_00 6000
    sup gapnorm_res_t104 "$MAIN" gapnorm t1_04 6000 --residual
    ssl ssl_gapnorm_t100 "$MAIN" gapnorm t1_00 6000
    ssl ssl_resgapnorm_t104 "$MAIN" gapnorm t1_04 6000 --residual
    ssl ssl_queue_gap_t100 "$MAIN" gap t1_00 6000 --queue "$QUEUE"
    ssl ssl_queue_restail_t104 "$MAIN" tail t1_04 6000 --residual --queue "$QUEUE"
    ssl ssl_queue_gap_t104_nuisance "$NOISY" gap t1_04 6000 --queue "$QUEUE"
    ssl ssl_queue_restail_t104_nuisance "$NOISY" tail t1_04 6000 --residual --queue "$QUEUE"
    finish gapnorm_t100 gapnorm_res_t104 ssl_gapnorm_t100 ssl_resgapnorm_t104 ssl_queue_gap_t100 \
        ssl_queue_restail_t104 ssl_queue_gap_t104_nuisance ssl_queue_restail_t104_nuisance
fi

if wants 8; then
    residual_bank "$MAIN" t1_04 t1_00 flair
    ssl ssl_uni_restail_t104 "$MAIN" tail t1_04 6000 --residual --uniform
    ssl ssl_uni_queue_restail_t104 "$MAIN" tail t1_04 6000 --residual --uniform --queue "$QUEUE"
    ssl ssl_code_restail_t104 "$MAIN" tail t1_04 6000 --residual --code-batches 500
    audit
    ssl ssl_uni_gap_t100 "$MAIN" gap t1_00 6000 --uniform
    ssl ssl_uni_tail_t100 "$MAIN" tail t1_00 6000 --uniform
    finish ssl_uni_restail_t104 ssl_uni_queue_restail_t104 ssl_code_restail_t104 ssl_uni_gap_t100 ssl_uni_tail_t100
    audit
    [ "$DRY" = 0 ] && "$PY" "$D/summarize.py" > "$D/summary.txt" 2>&1
fi

if wants 9; then
    NOISY=$D/bank_nuisance
    bank "$NOISY" --nuisance
    residual_bank "$MAIN" t1_04 t1_00 flair
    residual_bank "$NOISY" t1_04 flair
    audit
    ssl ssl_run_restail_t104 "$MAIN" tail t1_04 6000 --residual --gauge running
    EVERY=2000 ssl ssl_code_run_restail_t104_20k "$MAIN" tail t1_04 20000 --residual --code-batches 500 --gauge running
    EVERY=2000 ssl ssl_code_run_queue_restail_t104_20k "$MAIN" tail t1_04 20000 --residual --code-batches 500 \
        --gauge running --queue "$QUEUE"
    EVERY=2000 ssl ssl_code_run_queue_restail_t104_nuisance_20k "$NOISY" tail t1_04 20000 --residual \
        --code-batches 500 --gauge running --queue "$QUEUE"
    finish ssl_run_restail_t104 ssl_code_run_restail_t104_20k ssl_code_run_queue_restail_t104_20k \
        ssl_code_run_queue_restail_t104_nuisance_20k
    audit
    [ "$DRY" = 0 ] && "$PY" "$D/summarize.py" > "$D/summary.txt" 2>&1
fi

if wants 10; then
    residual_bank "$MAIN" t1_04 t1_00 flair
    ssl ssl_code_decor_restail_t104 "$MAIN" tail t1_04 6000 --residual --code-batches 500 --decorrelate 0.1
    ssl ssl_code_drop_restail_t104 "$MAIN" tail t1_04 6000 --residual --code-batches 500 --group-dropout 0.33
    ssl ssl_code_jitter_restail_t104 "$MAIN" tail t1_04 6000 --residual --code-batches 500 --jitter 2 --rotate 3
    finish ssl_code_decor_restail_t104 ssl_code_drop_restail_t104 ssl_code_jitter_restail_t104
    audit
    [ "$DRY" = 0 ] && "$PY" "$D/summarize.py" > "$D/summary.txt" 2>&1
fi

if wants 11; then
    residual_bank "$MAIN" t1_04 t1_00 flair
    ssl ssl_code_nopatch_restail_t104 "$MAIN" tail t1_04 6000 --residual --code-batches 500 --patch-weight 0
    ssl ssl_code_jitter_nopatch_restail_t104 "$MAIN" tail t1_04 6000 --residual --code-batches 500 --jitter 2 \
        --rotate 3 --patch-weight 0
    ssl ssl_code_jitter_drop_restail_t104 "$MAIN" tail t1_04 6000 --residual --code-batches 500 --jitter 2 \
        --rotate 3 --group-dropout 0.33
    ssl ssl_code_jitter_restail_t104_s2 "$MAIN" tail t1_04 6000 --residual --code-batches 500 --jitter 2 --rotate 3 \
        --seed 2
    finish ssl_code_nopatch_restail_t104 ssl_code_jitter_nopatch_restail_t104 ssl_code_jitter_drop_restail_t104 \
        ssl_code_jitter_restail_t104_s2
    audit
    [ "$DRY" = 0 ] && "$PY" "$D/summarize.py" > "$D/summary.txt" 2>&1
fi

if wants 12; then
    residual_bank "$MAIN" t1_04 t1_00 flair
    EVERY=2000 ssl ssl_code_jitter_nopatch_restail_t104_20k "$MAIN" tail t1_04 20000 --residual --code-batches 500 \
        --jitter 2 --rotate 3 --patch-weight 0
    ssl ssl_code_jitter1_nopatch_restail_t104 "$MAIN" tail t1_04 6000 --residual --code-batches 500 --jitter 1 \
        --rotate 1.5 --patch-weight 0
    ssl ssl_code_jitter4_nopatch_restail_t104 "$MAIN" tail t1_04 6000 --residual --code-batches 500 --jitter 4 \
        --rotate 6 --patch-weight 0
    finish ssl_code_jitter_nopatch_restail_t104_20k ssl_code_jitter1_nopatch_restail_t104 \
        ssl_code_jitter4_nopatch_restail_t104
    audit
    [ "$DRY" = 0 ] && "$PY" "$D/summarize.py" > "$D/summary.txt" 2>&1
fi

if wants 13; then
    residual_bank "$MAIN" t1_04 t1_00 flair
    EVERY=2000 ssl ssl_code_jitter4_nopatch_restail_t104_20k "$MAIN" tail t1_04 20000 --residual --code-batches 500 \
        --jitter 4 --rotate 6 --patch-weight 0
    ssl ssl_code_jitter4_shared_nopatch_restail_t104 "$MAIN" tail t1_04 6000 --residual --code-batches 500 --jitter 4 \
        --rotate 6 --shared-jitter --patch-weight 0
    ssl ssl_code_jitter4_nopatch_restail_t104_s2 "$MAIN" tail t1_04 6000 --residual --code-batches 500 --jitter 4 \
        --rotate 6 --patch-weight 0 --seed 2
    finish ssl_code_jitter4_nopatch_restail_t104_20k ssl_code_jitter4_shared_nopatch_restail_t104 \
        ssl_code_jitter4_nopatch_restail_t104_s2
    audit
    [ "$DRY" = 0 ] && "$PY" "$D/summarize.py" > "$D/summary.txt" 2>&1
fi

echo "$(date +%H:%M) all requested stages done"
