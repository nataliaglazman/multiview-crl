#!/bin/bash
# Burden feasibility on a CUDA machine. Run from anywhere inside the repository, on the branch that holds this folder.
#
#   bash out/burden_feasibility_20261102/run_pc.sh            default stages, in order
#   bash out/burden_feasibility_20261102/run_pc.sh 5 6        only those stages
#   bash out/burden_feasibility_20261102/run_pc.sh 22         opt-in tabular causal benchmark on stage 19 bundles
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
# Stage 14 After the 20,000-step run, where temporal arrived but burden left the full 14-unit code. Room: the same run
#         with a 24-unit code (TailCode's head widened, the backbone and its init unchanged), against
#         ssl_code_jitter_nopatch_restail_t104_20k. Realism: the best recipe with the nuisance fields on, with and
#         without jitter, 6,000 steps each; before jitter, burden all but vanished from every arm on that bank.
# Stage 15 Causal discovery. A bank whose content latents come from an SCM over the rendered factors (bank.py --causal:
#         brain -> ventricle -> burden -> temporal <- thickness <- brain, temporal -> sulcal, lr on its own), the best
#         recipe on it at 6,000 and 20,000 steps, then each code and its untrained twin as bundles for the repo's
#         label-free discovery (eval.causal.latent_causal_discovery) and decoded discovery (compare_bundles
#         --with-graph), with PC on the true factors as the ceiling. Results in causal/. Needs causal-learn. Then how
#         much of each factor's own variation (its SCM noise, which no parent can fake) the code, TailCode's pooled
#         features and the backbone GAP hold, against the untrained twin (own_variation.py, causal/own_variation.txt).
# Stage 16 Where burden's own variation is lost. On stage 15's bank the backbone holds burden's own SCM noise (~0.3)
#         and the 14-unit code drops it as ventricle and temporal arrive. Room: the 20,000-step run with a 24-unit code
#         (--content-units 24), then label-free and decoded discovery against the 14-unit run (causal/c24_*) and its own
#         variation. Past the head: decoded discovery on the backbone GAP of stage 15's two runs (causal/gap_*), no
#         training, started beside the arm. Needs stage 15's runs and causal-learn.
# Stage 17 Harder negatives on the causal bank. First remainder_price.py prices burden's remainder against temporal's
#         under each batch rule on the bank's true latents, no training (causal/remainder_price_r*.txt): seen as well
#         as temporal, the recipe's code batches already price it near temporal; tighter neighbourhoods and
#         hard-negative weighting help only when it is seen less well. Then the best recipe at 6,000 steps with
#         31-subject neighbourhoods (--code-neighbours 31), with negatives weighted by exp(5 cosine) (--hard-negatives
#         5), and with both, against ssl_code_jitter_nopatch_restail_t104_causal: own variation of each and decoded and
#         label-free discovery (causal/hard_*).
# Stage 18 Do 31-subject neighbourhoods hold up? In stage 17 (6,000 steps, one seed) every child's own variation in
#         the code rose, burden's from .10/.11 to .15/.19, with the roots unchanged. The same at 20,000 steps, where
#         the recipe's code had dropped burden's own variation to .04 (against ..._causal_20k); and a seed replicate
#         at 6,000 steps with its own recipe control (both --seed 2). Own variation of each, and discovery on the
#         20,000-step pair (causal/n31_*).
# Stage 19 Two more fundamental changes, on the causal bank. CCA readout, no training: a regularised CCA between each
#         run's T1 and FLAIR features (TailCode's pooled features, and the backbone GAP), fitted on training subjects,
#         whose top 14 or 24 canonical variates are the code. CCA ranks shared directions by how well the views agree,
#         not by how much they help tell subjects apart, so it may keep the remainders the InfoNCE head drops
#         (causal/cca.txt; decoded discovery on the 24-unit CCA codes in causal/cca_decoded_*). Partial observability
#         (Yao et al. 2024), with nothing built around any factor: a third view, each subject's T1 restricted to a
#         random half of the volume along a random axis, through the T1 backbone and pooled over the half it shows, aligned
#         with FLAIR on the first 8 code units. Those can hold only what every half shows (global content); the other 6
#         align only the whole T1 and FLAIR, so they hold regional content, whatever it is, which global content cannot
#         crowd out. The best recipe with it at 6,000 and 20,000 steps, against
#         ssl_code_jitter_nopatch_restail_t104_causal(_20k): own variation, CCA readout, discovery.
# Stage 20 Does the 20,000-step result for 31-subject neighbourhoods replicate? In stage 18 (seed 42) they kept burden's
#         own variation in the code at .19/.18 against the recipe's .04/.06, but at 6,000 steps the recipe's own burden
#         varied from .10 to .26 between seeds. The recipe and 31-subject neighbourhoods at 20,000 steps, both
#         --seed 2: own variation of each and discovery on the pair (causal/n31s2_*).
# Stage 21 Does cross-view correlation preserve more factors in the code than subject matching? Barlow Twins and
#         InfoNCE, each with 24 units, uniform batches and 20,000 steps on the causal bank. Score each factor's own
#         variation, agreement between matching code units, and CCA readouts from both backbones (causal/stage21/).
# Stage 23 Can units paid only for what the main code lacks keep burden? The recipe's 14 units on uniform batches plus
#         10 remainder units, trained only to tell apart each subject's 31 nearest neighbours on the 14 (rebuilt every
#         500 steps): a conditional InfoNCE, so they gain only from shared content the 14 do not hold. 20,000 steps on
#         the causal bank, seed 42. Control: stage 21's 24-unit InfoNCE arm (same width, head, init, batches and
#         jitter); reference: the 31-neighbour code batches (stage 18). Own variation, from the whole code and from each
#         part, and the audit (causal/stage23/).
# Stage 24 Stage 23 with the remainder neighbourhoods built on both views' backbone GAP (--remainder-space gap) in place
#         of the main units. Stage 23's main units dropped brain size, so its neighbourhoods did not match on it and
#         its remainder units learned brain size. The GAP holds the anatomy and the lesion's position, so neighbours
#         match on both, which is where precision_audit.py's pricing found a 20-voxel burden reading pays. Against
#         stage 23 and stage 21's 24-unit arm: own variation, the audit and the precision audit (causal/stage24/).
#
# PARALLEL arms share the GPU (default 3; each takes roughly 2-4 GB, check nvidia-smi). PY picks the interpreter.
# Finished arms are skipped, and so are arms still running outside this script (started by hand), so rerunning after a
# crash or an out-of-memory error continues where it stopped.
# DRY=1 prints the commands without running them.
# Stage 25 (opt-in) Replace stage 21's global average with 2x2x2 regional averages and keep the tails.
# Compare volume precision and the two 24-unit InfoNCE codes (causal/stage25/).

set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
D=out/burden_feasibility_20261102
PY=${PY:-python}
PARALLEL=${PARALLEL:-3}
BATCH=${BATCH:-32}
DRY=${DRY:-0}
STAGES=${*:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 23 24}
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

causal() { # [TAG=T] [FEATURES=code|gap|pooled] causal NAME...: each run's features and its untrained twin as bundles,
    # then decoded discovery, and label-free discovery too on the code. TAG prefixes the reports; a bundle newer than
    # its run's checkpoint is reused.
    local out=$D/causal features=${FEATURES:-code} prefix=${TAG:+${TAG}_} suffix="" bundles="" floors="" name view base
    [ "$features" = code ] || suffix=.$features
    for name in "$@"; do
        bundles="$bundles $name=$out/$name$suffix.npz"
        floors="$floors $name=$out/$name$suffix.floor.npz"
    done
    if [ "$DRY" = 1 ]; then
        for name in "$@"; do
            echo "$PY $D/causal_bundle.py $D/runs/$name $out/$name$suffix.npz --features $features [--untrained]"
        done
        if [ "$features" = code ]; then
            echo "$PY -m eval.causal.latent_causal_discovery --bundles$bundles --bundle-floors$floors --ceiling" \
                "--out $out/${prefix}label_free_view*.json"
        fi
        echo "$PY -m eval.protocol.compare_bundles --bundles$bundles --floors$floors --with-graph" \
            "--out $out/${prefix}decoded_view*.json"
        return
    fi
    if ! "$PY" -c "import causallearn" 2> /dev/null; then
        echo "causal discovery needs causal-learn: $PY -m pip install causal-learn"
        return
    fi
    mkdir -p "$out"
    echo "$(date +%H:%M) causal discovery on the $features of $*"
    for name in "$@"; do
        base=$out/$name$suffix
        if [ "$base.npz" -nt "$D/runs/$name/model.pt" ] && [ "$base.floor.npz" -nt "$D/runs/$name/model.pt" ]; then
            continue
        fi
        "$PY" "$D/causal_bundle.py" "$D/runs/$name" "$base.npz" --features "$features" > "$base.log" 2>&1 \
            && "$PY" "$D/causal_bundle.py" "$D/runs/$name" "$base.floor.npz" --features "$features" --untrained \
                >> "$base.log" 2>&1 \
            || { echo "bundle failed, see $base.log"; return; }
    done
    for view in 1 2; do
        if [ "$features" = code ]; then
            "$PY" -m eval.causal.latent_causal_discovery --bundles $bundles --bundle-floors $floors \
                --bundle-view "$view" --ceiling --mixing-null 30 --out "$out/${prefix}label_free_view$view.json" \
                > "$out/${prefix}label_free_view$view.txt" 2>&1
        fi
        "$PY" -m eval.protocol.compare_bundles --bundles $bundles --floors $floors --view "$view" --with-graph \
            --holdout-readout --graph-repeats 20 --out "$out/${prefix}decoded_view$view.json" \
            > "$out/${prefix}decoded_view$view.log" 2>&1
    done
    echo "$(date +%H:%M) causal discovery done, see $out"
}

own_variation() { # own_variation NAME...: each factor's own variation (SCM noise) in the code, pooled features and GAP
    local runs="" name
    for name in "$@"; do runs="$runs $D/runs/$name"; done
    if [ "$DRY" = 1 ]; then echo "$PY $D/own_variation.py $D/causal/own_variation.json$runs"; return; fi
    mkdir -p "$D/causal"
    echo "$(date +%H:%M) own variation of each factor in $*"
    "$PY" "$D/own_variation.py" "$D/causal/own_variation.json" $runs > "$D/causal/own_variation.txt" 2>&1 \
        || echo "own-variation check failed, see $D/causal/own_variation.txt"
}

cca_readout() { # cca_readout NAME...: CCA codes from each run's pooled features and GAP, scored like its own code
    local runs="" name
    for name in "$@"; do runs="$runs $D/runs/$name"; done
    if [ "$DRY" = 1 ]; then echo "$PY $D/cca_readout.py $D/causal/cca.json$runs --bundles $D/causal/cca"; return; fi
    mkdir -p "$D/causal"
    echo "$(date +%H:%M) CCA readout of $*"
    "$PY" "$D/cca_readout.py" "$D/causal/cca.json" $runs --bundles "$D/causal/cca" > "$D/causal/cca.txt" 2>&1 \
        || echo "CCA readout failed, see $D/causal/cca.txt"
}

cca_discovery() { # cca_discovery NAME...: decoded discovery on each run's 24-unit CCA codes, from pooled and GAP
    local out=$D/causal bundles="" floors="" name source view
    for name in "$@"; do
        for source in pooled gap; do
            bundles="$bundles $name.$source=$out/cca/$name.cca24_$source.npz"
            floors="$floors $name.$source=$out/cca/$name.cca24_$source.floor.npz"
        done
    done
    if [ "$DRY" = 1 ]; then
        echo "$PY -m eval.protocol.compare_bundles --bundles$bundles --floors$floors --with-graph"
        return
    fi
    if ! "$PY" -c "import causallearn" 2> /dev/null; then
        echo "causal discovery needs causal-learn: $PY -m pip install causal-learn"
        return
    fi
    for view in 1 2; do
        "$PY" -m eval.protocol.compare_bundles --bundles $bundles --floors $floors --view "$view" --with-graph \
            --holdout-readout --graph-repeats 20 --out "$out/cca_decoded_view$view.json" \
            > "$out/cca_decoded_view$view.log" 2>&1
    done
}

MAIN=$D/bank
for stage in {1..14}; do
    if wants "$stage"; then
        bank "$MAIN"
        break
    fi
done

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

if wants 14; then
    NOISY=$D/bank_nuisance
    bank "$NOISY" --nuisance
    residual_bank "$MAIN" t1_04 t1_00 flair
    residual_bank "$NOISY" t1_04 flair
    EVERY=2000 ssl ssl_code_jitter_nopatch_restail_t104_c24_20k "$MAIN" tail t1_04 20000 --residual --code-batches 500 \
        --jitter 2 --rotate 3 --patch-weight 0 --content-units 24
    ssl ssl_code_jitter_nopatch_restail_t104_nuisance "$NOISY" tail t1_04 6000 --residual --code-batches 500 \
        --jitter 2 --rotate 3 --patch-weight 0
    ssl ssl_code_nopatch_restail_t104_nuisance "$NOISY" tail t1_04 6000 --residual --code-batches 500 --patch-weight 0
    finish ssl_code_jitter_nopatch_restail_t104_c24_20k ssl_code_jitter_nopatch_restail_t104_nuisance \
        ssl_code_nopatch_restail_t104_nuisance
    audit
    [ "$DRY" = 0 ] && "$PY" "$D/summarize.py" > "$D/summary.txt" 2>&1
fi

if wants 22; then
    NAME=${TABULAR_RUN:-ssl_code_jitter_nopatch_restail_t104_causal_halves8}
    RUN=$D/runs/$NAME
    OUT=$D/causal/tabular/$NAME
    CCA=$D/causal/cca/$NAME.cca24_gap
    if [ "$DRY" = 0 ]; then
        if [ ! -f "$RUN/model.pt" ] || [ ! "$CCA.npz" -nt "$RUN/model.pt" ] || \
            [ ! "$CCA.floor.npz" -nt "$RUN/model.pt" ]; then
            echo "Stage 22 needs the finished run and current CCA bundles from stage 19 for $NAME."
            exit 1
        fi
        mkdir -p "$OUT" || exit 1
    fi
    for feature in code gap; do
        base=$OUT/$feature
        if [ "$DRY" = 1 ]; then
            echo "$PY $D/causal_bundle.py $RUN $base.npz --features $feature"
            echo "$PY $D/causal_bundle.py $RUN $base.floor.npz --features $feature --untrained"
        elif [ ! "$base.npz" -nt "$RUN/model.pt" ] || [ ! "$base.floor.npz" -nt "$RUN/model.pt" ]; then
            "$PY" "$D/causal_bundle.py" "$RUN" "$base.npz" --features "$feature" > "$base.log" 2>&1 && \
                "$PY" "$D/causal_bundle.py" "$RUN" "$base.floor.npz" --features "$feature" --untrained >> "$base.log" 2>&1 || \
                { echo "Bundle export failed, see $base.log"; exit 1; }
        fi
    done
    CMD=("$PY" -m eval.causal.tabular_benchmark --view each
        --bundles "cca24=$CCA.npz" "code=$OUT/code.npz" "gap=$OUT/gap.npz"
        --floors "cca24=$CCA.floor.npz" "code=$OUT/code.floor.npz" "gap=$OUT/gap.floor.npz"
        --out "$OUT/report.json" --tables-dir "$OUT/tables")
    if [ "$DRY" = 1 ]; then
        echo "${CMD[*]} > $OUT/report.log"
    else
        "${CMD[@]}" > "$OUT/report.log" 2>&1 || { echo "Stage 22 failed, see $OUT/report.log"; exit 1; }
        cat "$OUT/report.log"
    fi
fi

if wants 15; then
    CAUSAL=$D/bank_causal
    bank "$CAUSAL" --t1-values 0.4 --causal
    residual_bank "$CAUSAL" t1_04 flair
    ssl ssl_code_jitter_nopatch_restail_t104_causal "$CAUSAL" tail t1_04 6000 --residual --code-batches 500 --jitter 2 \
        --rotate 3 --patch-weight 0
    EVERY=2000 ssl ssl_code_jitter_nopatch_restail_t104_causal_20k "$CAUSAL" tail t1_04 20000 --residual \
        --code-batches 500 --jitter 2 --rotate 3 --patch-weight 0
    finish ssl_code_jitter_nopatch_restail_t104_causal ssl_code_jitter_nopatch_restail_t104_causal_20k
    audit
    causal ssl_code_jitter_nopatch_restail_t104_causal ssl_code_jitter_nopatch_restail_t104_causal_20k
    own_variation ssl_code_jitter_nopatch_restail_t104_causal ssl_code_jitter_nopatch_restail_t104_causal_20k
    [ "$DRY" = 0 ] && "$PY" "$D/summarize.py" > "$D/summary.txt" 2>&1
fi

if wants 16; then
    CAUSAL=$D/bank_causal
    bank "$CAUSAL" --t1-values 0.4 --causal
    residual_bank "$CAUSAL" t1_04 flair
    EVERY=2000 ssl ssl_code_jitter_nopatch_restail_t104_causal_c24_20k "$CAUSAL" tail t1_04 20000 --residual \
        --code-batches 500 --jitter 2 --rotate 3 --patch-weight 0 --content-units 24
    TAG=gap FEATURES=gap causal ssl_code_jitter_nopatch_restail_t104_causal \
        ssl_code_jitter_nopatch_restail_t104_causal_20k
    finish ssl_code_jitter_nopatch_restail_t104_causal_c24_20k
    audit
    TAG=c24 causal ssl_code_jitter_nopatch_restail_t104_causal_20k ssl_code_jitter_nopatch_restail_t104_causal_c24_20k
    own_variation ssl_code_jitter_nopatch_restail_t104_causal_c24_20k
    [ "$DRY" = 0 ] && "$PY" "$D/summarize.py" > "$D/summary.txt" 2>&1
fi

if wants 17; then
    CAUSAL=$D/bank_causal
    bank "$CAUSAL" --t1-values 0.4 --causal
    residual_bank "$CAUSAL" t1_04 flair
    RECIPE="--residual --code-batches 500 --jitter 2 --rotate 3 --patch-weight 0"
    ssl ssl_code_jitter_nopatch_restail_t104_causal_n31 "$CAUSAL" tail t1_04 6000 $RECIPE --code-neighbours 31
    ssl ssl_code_jitter_nopatch_restail_t104_causal_hard5 "$CAUSAL" tail t1_04 6000 $RECIPE --hard-negatives 5
    ssl ssl_code_jitter_nopatch_restail_t104_causal_n31_hard5 "$CAUSAL" tail t1_04 6000 $RECIPE --code-neighbours 31 \
        --hard-negatives 5
    for r in 0.75 0.5; do
        if [ "$DRY" = 1 ]; then
            echo "$PY $D/remainder_price.py $D/causal/remainder_price_r$r.json --bank $CAUSAL --r-burden $r"
            continue
        fi
        mkdir -p "$D/causal"
        "$PY" "$D/remainder_price.py" "$D/causal/remainder_price_r$r.json" --bank "$CAUSAL" --r-burden "$r" \
            > "$D/causal/remainder_price_r$r.txt" 2>&1 || echo "pricing failed, see $D/causal/remainder_price_r$r.txt"
    done
    finish ssl_code_jitter_nopatch_restail_t104_causal_n31 ssl_code_jitter_nopatch_restail_t104_causal_hard5 \
        ssl_code_jitter_nopatch_restail_t104_causal_n31_hard5
    audit
    own_variation ssl_code_jitter_nopatch_restail_t104_causal_n31 ssl_code_jitter_nopatch_restail_t104_causal_hard5 \
        ssl_code_jitter_nopatch_restail_t104_causal_n31_hard5
    TAG=hard causal ssl_code_jitter_nopatch_restail_t104_causal ssl_code_jitter_nopatch_restail_t104_causal_n31 \
        ssl_code_jitter_nopatch_restail_t104_causal_hard5 ssl_code_jitter_nopatch_restail_t104_causal_n31_hard5
    [ "$DRY" = 0 ] && "$PY" "$D/summarize.py" > "$D/summary.txt" 2>&1
fi

if wants 18; then
    CAUSAL=$D/bank_causal
    bank "$CAUSAL" --t1-values 0.4 --causal
    residual_bank "$CAUSAL" t1_04 flair
    RECIPE="--residual --code-batches 500 --jitter 2 --rotate 3 --patch-weight 0"
    EVERY=2000 ssl ssl_code_jitter_nopatch_restail_t104_causal_n31_20k "$CAUSAL" tail t1_04 20000 $RECIPE \
        --code-neighbours 31
    ssl ssl_code_jitter_nopatch_restail_t104_causal_s2 "$CAUSAL" tail t1_04 6000 $RECIPE --seed 2
    ssl ssl_code_jitter_nopatch_restail_t104_causal_n31_s2 "$CAUSAL" tail t1_04 6000 $RECIPE --code-neighbours 31 \
        --seed 2
    finish ssl_code_jitter_nopatch_restail_t104_causal_n31_20k ssl_code_jitter_nopatch_restail_t104_causal_s2 \
        ssl_code_jitter_nopatch_restail_t104_causal_n31_s2
    audit
    own_variation ssl_code_jitter_nopatch_restail_t104_causal_n31_20k ssl_code_jitter_nopatch_restail_t104_causal_s2 \
        ssl_code_jitter_nopatch_restail_t104_causal_n31_s2
    TAG=n31 causal ssl_code_jitter_nopatch_restail_t104_causal_20k ssl_code_jitter_nopatch_restail_t104_causal_n31_20k
    [ "$DRY" = 0 ] && "$PY" "$D/summarize.py" > "$D/summary.txt" 2>&1
fi

if wants 19; then
    CAUSAL=$D/bank_causal
    bank "$CAUSAL" --t1-values 0.4 --causal
    residual_bank "$CAUSAL" t1_04 flair
    RECIPE="--residual --code-batches 500 --jitter 2 --rotate 3 --patch-weight 0"
    HALVES="--half-views xyz --half-units 8"
    ssl ssl_code_jitter_nopatch_restail_t104_causal_halves8 "$CAUSAL" tail t1_04 6000 $RECIPE $HALVES
    EVERY=2000 ssl ssl_code_jitter_nopatch_restail_t104_causal_halves8_20k "$CAUSAL" tail t1_04 20000 $RECIPE $HALVES
    cca_readout ssl_code_jitter_nopatch_restail_t104_causal ssl_code_jitter_nopatch_restail_t104_causal_20k
    cca_discovery ssl_code_jitter_nopatch_restail_t104_causal ssl_code_jitter_nopatch_restail_t104_causal_20k
    finish ssl_code_jitter_nopatch_restail_t104_causal_halves8 ssl_code_jitter_nopatch_restail_t104_causal_halves8_20k
    audit
    own_variation ssl_code_jitter_nopatch_restail_t104_causal_halves8 \
        ssl_code_jitter_nopatch_restail_t104_causal_halves8_20k
    cca_readout ssl_code_jitter_nopatch_restail_t104_causal_halves8 \
        ssl_code_jitter_nopatch_restail_t104_causal_halves8_20k
    TAG=halves causal ssl_code_jitter_nopatch_restail_t104_causal ssl_code_jitter_nopatch_restail_t104_causal_halves8
    TAG=halves20k causal ssl_code_jitter_nopatch_restail_t104_causal_20k \
        ssl_code_jitter_nopatch_restail_t104_causal_halves8_20k
    [ "$DRY" = 0 ] && "$PY" "$D/summarize.py" > "$D/summary.txt" 2>&1
fi

if wants 20; then
    CAUSAL=$D/bank_causal
    bank "$CAUSAL" --t1-values 0.4 --causal
    residual_bank "$CAUSAL" t1_04 flair
    RECIPE="--residual --code-batches 500 --jitter 2 --rotate 3 --patch-weight 0"
    EVERY=2000 ssl ssl_code_jitter_nopatch_restail_t104_causal_20k_s2 "$CAUSAL" tail t1_04 20000 $RECIPE --seed 2
    EVERY=2000 ssl ssl_code_jitter_nopatch_restail_t104_causal_n31_20k_s2 "$CAUSAL" tail t1_04 20000 $RECIPE \
        --code-neighbours 31 --seed 2
    finish ssl_code_jitter_nopatch_restail_t104_causal_20k_s2 ssl_code_jitter_nopatch_restail_t104_causal_n31_20k_s2
    audit
    own_variation ssl_code_jitter_nopatch_restail_t104_causal_20k_s2 \
        ssl_code_jitter_nopatch_restail_t104_causal_n31_20k_s2
    TAG=n31s2 causal ssl_code_jitter_nopatch_restail_t104_causal_20k_s2 \
        ssl_code_jitter_nopatch_restail_t104_causal_n31_20k_s2
    [ "$DRY" = 0 ] && "$PY" "$D/summarize.py" > "$D/summary.txt" 2>&1
fi

if wants 21; then
    CAUSAL=$D/bank_causal
    bank "$CAUSAL" --t1-values 0.4 --causal
    residual_bank "$CAUSAL" t1_04 flair
    NCE=ssl_uni_jitter_nopatch_restail_t104_causal_c24_20k
    BT=ssl_uni_jitter_nopatch_restail_t104_causal_bt_c24_20k
    RECIPE=(--uniform --residual --jitter 2 --rotate 3 --patch-weight 0 --content-units 24 --seed 42)
    EVERY=2000 ssl "$NCE" "$CAUSAL" tail t1_04 20000 "${RECIPE[@]}" --contrastive-loss-type infonce
    EVERY=2000 ssl "$BT" "$CAUSAL" tail t1_04 20000 "${RECIPE[@]}" --contrastive-loss-type barlow_twins
    finish "$NCE" "$BT"

    OUT=$D/causal/stage21
    if [ "$DRY" = 0 ]; then
        if ! grep -qx DONE "$D/logs/$NCE.log" || ! grep -qx DONE "$D/logs/$BT.log"; then
            echo "Rerun stage 21 after both arms finish to generate the reports."
            exit 1
        fi
        mkdir -p "$OUT" || exit 1
    fi
    for report in own_variation code_audit cca_readout; do
        if [ "$DRY" = 1 ]; then
            echo "$PY $D/$report.py $OUT/$report.json $D/runs/$NCE $D/runs/$BT > $OUT/$report.txt"
        else
            "$PY" "$D/$report.py" "$OUT/$report.json" "$D/runs/$NCE" "$D/runs/$BT" > "$OUT/$report.txt" 2>&1 \
                || { echo "$report failed, see $OUT/$report.txt"; exit 1; }
        fi
    done
    [ "$DRY" = 0 ] && echo "Stage 21 reports in $OUT"
fi

if wants 23; then
    CAUSAL=$D/bank_causal
    bank "$CAUSAL" --t1-values 0.4 --causal
    residual_bank "$CAUSAL" t1_04 flair
    REM=ssl_uni_jitter_nopatch_restail_t104_causal_rem10_20k
    NCE=ssl_uni_jitter_nopatch_restail_t104_causal_c24_20k
    N31=ssl_code_jitter_nopatch_restail_t104_causal_n31_20k
    EVERY=2000 ssl "$REM" "$CAUSAL" tail t1_04 20000 --uniform --residual --jitter 2 --rotate 3 --patch-weight 0 \
        --seed 42 --remainder-units 10 --remainder-neighbours 31 --remainder-every 500
    finish "$REM"

    OUT=$D/causal/stage23
    if [ "$DRY" = 0 ]; then
        for name in "$REM" "$NCE"; do
            if ! grep -qx DONE "$D/logs/$name.log"; then
                echo "Stage 23 needs $name finished (stage 21 trains $NCE); rerun stage 23 after it is."
                exit 1
            fi
        done
        mkdir -p "$OUT" || exit 1
    fi
    for report in own_variation code_audit; do
        if [ "$DRY" = 1 ]; then
            echo "$PY $D/$report.py $OUT/$report.json $D/runs/$REM $D/runs/$NCE $D/runs/$N31 > $OUT/$report.txt"
        else
            "$PY" "$D/$report.py" "$OUT/$report.json" "$D/runs/$REM" "$D/runs/$NCE" "$D/runs/$N31" \
                > "$OUT/$report.txt" 2>&1 || { echo "$report failed, see $OUT/$report.txt"; exit 1; }
        fi
    done
    [ "$DRY" = 0 ] && echo "Stage 23 reports in $OUT"
fi

if wants 24; then
    CAUSAL=$D/bank_causal
    bank "$CAUSAL" --t1-values 0.4 --causal
    residual_bank "$CAUSAL" t1_04 flair
    GAPREM=ssl_uni_jitter_nopatch_restail_t104_causal_rem10gap_20k
    REM=ssl_uni_jitter_nopatch_restail_t104_causal_rem10_20k
    NCE=ssl_uni_jitter_nopatch_restail_t104_causal_c24_20k
    EVERY=2000 ssl "$GAPREM" "$CAUSAL" tail t1_04 20000 --uniform --residual --jitter 2 --rotate 3 --patch-weight 0 \
        --seed 42 --remainder-units 10 --remainder-neighbours 31 --remainder-every 500 --remainder-space gap
    finish "$GAPREM"

    OUT=$D/causal/stage24
    if [ "$DRY" = 0 ]; then
        for name in "$GAPREM" "$REM" "$NCE"; do
            if ! grep -qx DONE "$D/logs/$name.log"; then
                echo "Stage 24 needs $name finished (stage 23 trains $REM, stage 21 $NCE); rerun stage 24 after it is."
                exit 1
            fi
        done
        mkdir -p "$OUT" "$D/causal/precision" || exit 1
    fi
    for report in own_variation code_audit; do
        if [ "$DRY" = 1 ]; then
            echo "$PY $D/$report.py $OUT/$report.json $D/runs/$GAPREM $D/runs/$REM $D/runs/$NCE > $OUT/$report.txt"
        else
            "$PY" "$D/$report.py" "$OUT/$report.json" "$D/runs/$GAPREM" "$D/runs/$REM" "$D/runs/$NCE" \
                > "$OUT/$report.txt" 2>&1 || { echo "$report failed, see $OUT/$report.txt"; exit 1; }
        fi
    done
    if [ "$DRY" = 1 ]; then
        echo "$PY $D/precision_audit.py $D/causal/precision/precision.json $D/runs/$GAPREM $D/runs/$REM" \
            "$D/runs/$NCE > $OUT/precision.txt"
    else
        "$PY" "$D/precision_audit.py" "$D/causal/precision/precision.json" "$D/runs/$GAPREM" "$D/runs/$REM" \
            "$D/runs/$NCE" > "$OUT/precision.txt" 2>&1 \
            || { echo "precision audit failed, see $OUT/precision.txt"; exit 1; }
    fi
    [ "$DRY" = 0 ] && echo "Stage 24 reports in $OUT"
fi

if wants 25; then
    CAUSAL=$D/bank_causal
    bank "$CAUSAL" --t1-values 0.4 --causal
    residual_bank "$CAUSAL" t1_04 flair
    NCE=ssl_uni_jitter_nopatch_restail_t104_causal_c24_20k
    REGIONAL=ssl_uni_jitter_nopatch_regionaltail_t104_causal_c24_20k
    RECIPE=(--uniform --residual --jitter 2 --rotate 3 --patch-weight 0 --content-units 24 --seed 42)
    EVERY=2000 ssl "$NCE" "$CAUSAL" tail t1_04 20000 "${RECIPE[@]}" --contrastive-loss-type infonce
    EVERY=2000 ssl "$REGIONAL" "$CAUSAL" regional_tail t1_04 20000 "${RECIPE[@]}" --contrastive-loss-type infonce
    finish "$NCE" "$REGIONAL"

    OUT=$D/causal/stage25
    if [ "$DRY" = 0 ]; then
        for name in "$NCE" "$REGIONAL"; do
            if ! grep -qx DONE "$D/logs/$name.log" || [ ! -f "$D/runs/$name/model.pt" ]; then
                echo "Stage 25 needs $name finished. Rerun stage 25 to generate the reports."
                exit 1
            fi
        done
        mkdir -p "$OUT" || exit 1
    fi
    for report in precision_audit own_variation code_audit; do
        if [ "$DRY" = 1 ]; then
            echo "$PY $D/$report.py $OUT/$report.json $D/runs/$NCE $D/runs/$REGIONAL > $OUT/$report.txt"
        else
            "$PY" "$D/$report.py" "$OUT/$report.json" "$D/runs/$NCE" "$D/runs/$REGIONAL" > "$OUT/$report.txt" 2>&1 \
                || { echo "$report failed, see $OUT/$report.txt"; exit 1; }
        fi
    done
    [ "$DRY" = 0 ] && echo "Stage 25 reports in $OUT"
fi

echo "$(date +%H:%M) all requested stages done"
