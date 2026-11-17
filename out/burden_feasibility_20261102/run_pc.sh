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
# Stage 26 (opt-in) The precision ceiling, a diagnostic with labels: stage 21's 24-unit arm with the InfoNCE replaced
# by a regression of the code on lesion volume (warm.py --supervise volume), 10k steps. Near 10 voxels in the precision
# audit means the architecture can read volume and InfoNCE never asks; near 20 means the architecture caps it
# (causal/stage26/).
# Stage 27 (opt-in) Why the T1 backbone stays near 20 voxels even with labels, on stage 26's two runs, no training: the
# precision audit on test scans moved by each run's training jitter (1 draw: is the raw scan the problem; 8 draws
# averaged: does the reading depend on where the lesion falls on the 4-voxel grid) and on training subjects
# (memorisation), beside stage 26's raw-scan audit (causal/stage27/compare.txt).
# Stage 28 (opt-in) Is the renderer's multiplicative field what keeps T1 burden unread? It varies every ~4 voxels, about
# a lesion's size, and costs T1 about three times what it costs FLAIR. A causal bank with the field off (bank.py
# --field off: the same subjects, lesions, styles and noise draws), then stage 26's labelled ceiling and stage 21's
# 24-unit InfoNCE arm on it, unchanged. The precision audit on test and training subjects, own variation and the audit,
# beside stage 21's arm on the field-on bank (causal/stage28/).
# Stage 29 (opt-in) After stage 28 the backbone reads lesion volume to ~11 voxels but the 24-unit code only to 18-20:
# the head is the bottleneck. Three head levers on bank_causal_nofield against stage 28's 24-unit InfoNCE arm (same
# bank, seed, batches and jitter): 48 units (is the code full?), 10 remainder units on GAP neighbourhoods (stage 24's
# arm, paid only for what the 14 main units lack) and Barlow Twins with 24 units (per-unit agreement, which does not
# saturate). And gain: stage 28's arm on a bank that also renders every scan at gain 1 (bank.py --gain off: the same
# subjects, lesions, bias, noise and field draws). The precision audit, own variation and the audit (causal/stage29/).
# Stage 30 (opt-in) Preserve the field-off backbone's information through its 24-unit head. Freeze stage 28's
# backbone and pooling statistics, continue identical source heads with InfoNCE alone or with cross-view prediction
# of the other view's 64 global-average features (weights 0.1 and 1). Linear predictors read only the exported code.
# Fit target whitening and initial-code scaling on training subjects only. Cache eight independent jitter draws per
# training subject once, then share them across the three 10k-step arms. This uses a finite augmentation cache, not
# fresh image jitter every step. Reports against the source run go to causal/stage30/; the price table is InfoNCE only.
# Stage 31 (opt-in) Per-unit critic weights (warm.py --critic-weights): one learnable weight per code unit inside the
# InfoNCE cosine, so a unit whose views agree poorly can count for less instead of costing the loss. First, no
# training: stage 29's precision audit re-priced with the critic refitted to each of its five codes
# (causal/stage31/price_stage29.txt). Then three arms on bank_causal_nofield: stage 29's remainder arm with weights on
# the remainder term, and on both terms (against stage 29's remainder arm), and stage 28's 24-unit arm with weights on
# its main term (against that arm; a saturated code, where the weights should change little). The precision audit, own
# variation and the audit of the three and their two controls (causal/stage31/).
# Stage 32 (opt-in) Stage 30's weight-1 head reads lesion volume to 14.9 voxels in both views (its InfoNCE control
# 23.9 / 22.9, the frozen backbone's GAP 11.0) and keeps the anatomy. Two head-only follow-ups, as in stage 30. Weights
# 3 and 10 on stage 30's cached features and batches: does the code close on the backbone? Weight 1 and its InfoNCE
# control on stage 29's gain-off backbone (GAP 8.7 / 8.1), with its own cache of eight jitter draws: does the code
# follow a sharper backbone? The precision audit, own variation and the audit of the four new heads, stage 30's control
# and weight-1 head, and both source runs (causal/stage32/).
# Stage 33 (opt-in) Stages 30 and 32's cross-view prediction trained end to end (warm.py --predict-weight): InfoNCE plus
# each view's code predicting the other view's whitened backbone GAP, taken with no gradient from a momentum copy of the
# backbone on the unjittered scans, so the backbone learns from the term too. 20,000 steps from scratch with stage 28's
# recipe: weights 1 and 3 on bank_causal_nofield (against stage 28's 24-unit InfoNCE arm) and weight 1 on the gain-off
# bank (against stage 29's gain-off arm). The precision audit, own variation and the audit, beside the two controls
# and stage 32's best frozen-backbone heads when they are there (causal/stage33/).
# Stage 34 (opt-in) Two checks before stage 33's prediction term joins the recipe. Style: does any code carry its own
# view's gain, bias or noise level (style_audit.py, no training)? Banks don't store those latents, so they are redrawn
# from the generator's seeds and checked against the stored content latents. It runs first, on stage 33's runs, its
# controls and stage 32's two heads, while the seed check trains. Seed: stage 33's two weight-1 arms again with seed 2.
# Then the precision audit, own variation and the audit of both seeds and both InfoNCE controls, and the style audit
# with the seed-2 runs added (causal/stage34/).
# Stage 35 (opt-in) Stage 33's recipe at the width of the ground truth: 10 content units, one per latent the causal
# bank renders into both views (brain, ventricle, burden, thickness, temporal, lr, sulcal, and the lesion's x, y, z;
# z_content[3:5] are unused with the burden target, and clean content draws no deformation or fissure field).
# Prediction weight 1 and its InfoNCE control on bank_causal_nofield, and weight 1 on the gain-off bank, 20,000 steps,
# seed 42. The four reports, beside the 24-unit counterparts whose scores are reused from stage 34 (causal/stage35/).
# Stage 36 (opt-in) Stage 35's 10-unit prediction code read volume ~5 voxels worse than 24 units: its linear map
# predicts only 10 directions of the other view's whitened GAP, which spreads the shared latents over more. Each map
# becomes the linear map plus a one-hidden-layer MLP of width 128 (warm.py --predict-hidden), through which a 10-unit
# code holding the latents in any invertible form can predict all of them. First, no training (while the arms train):
# bottleneck_check.py fits fresh 10- and 24-unit bottlenecks to stages 33 and 35's frozen features through either map,
# with no InfoNCE. Arms on bank_causal_nofield, 20,000 steps, seed 42: 10 units with the MLP map, the same with critic
# weights on the InfoNCE (a 10-unit cosine charges a less-agreeing unit more), and 24 units with the MLP map (the map
# alone). The four reports, own variation also by boosting, beside the linear-map arms (causal/stage36/).
# Stage 37 (opt-in) Sharper views for the 10-unit code. No change to the objective kept burden in 10 units; what moved
# it was how exactly each view determines burden (gain off: 17.3 / 17.6 -> 14.9 / 14.3 voxels). With the field off, a
# normative residual with 32 PCA directions removed instead of 16 lets the fixed image filter read burden's own part to
# 7.4 / 5.2 voxels instead of 9.0 / 6.6 (normative_components.py, read on a local copy of the bank). The residuals go
# into banks of their own, bank_causal_nofield_k32 and bank_causal_nofield_nogain_k32, whose other arrays are links
# (residual_bank.py --components 32 --source). Arms, 20,000 steps, seed 42, prediction weight 1: stage 35's 10-unit
# arm on each of the two, and stage 33's 24-unit arm on the first. The four reports, beside the 16-component arms
# whose scores are reused from stage 35 (causal/stage37/). Then, no training: normative_components.py on both source
# banks.

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
        elif [ ! "$base.npz" -nt "$RUN/model.pt" ] || [ ! "$base.floor.npz" -nt "$RUN/model.pt" ] || \
            ! "$PY" -c "import sys, numpy; sys.exit('z_own' not in numpy.load(sys.argv[1]).files)" "$base.npz"; then
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

if wants 26; then
    CAUSAL=$D/bank_causal
    bank "$CAUSAL" --t1-values 0.4 --causal
    residual_bank "$CAUSAL" t1_04 flair
    NCE=ssl_uni_jitter_nopatch_restail_t104_causal_c24_20k
    CEILING=ceiling_uni_jitter_nopatch_restail_t104_causal_c24_10k
    ssl "$CEILING" "$CAUSAL" tail t1_04 10000 --uniform --residual --jitter 2 --rotate 3 --patch-weight 0 \
        --content-units 24 --seed 42 --supervise volume
    finish "$CEILING"

    OUT=$D/causal/stage26
    if [ "$DRY" = 0 ]; then
        for name in "$CEILING" "$NCE"; do
            if ! grep -qx DONE "$D/logs/$name.log" || [ ! -f "$D/runs/$name/model.pt" ]; then
                echo "Stage 26 needs $name finished (stage 21 trains $NCE). Rerun stage 26 after it is."
                exit 1
            fi
        done
        mkdir -p "$OUT" || exit 1
    fi
    for report in precision_audit own_variation code_audit; do
        if [ "$DRY" = 1 ]; then
            echo "$PY $D/$report.py $OUT/$report.json $D/runs/$CEILING $D/runs/$NCE > $OUT/$report.txt"
        else
            "$PY" "$D/$report.py" "$OUT/$report.json" "$D/runs/$CEILING" "$D/runs/$NCE" > "$OUT/$report.txt" 2>&1 \
                || { echo "$report failed, see $OUT/$report.txt"; exit 1; }
        fi
    done
    [ "$DRY" = 0 ] && echo "Stage 26 reports in $OUT"
fi

if wants 27; then
    NCE=ssl_uni_jitter_nopatch_restail_t104_causal_c24_20k
    CEILING=ceiling_uni_jitter_nopatch_restail_t104_causal_c24_10k
    RAW=$D/causal/stage26/precision_audit.json
    OUT=$D/causal/stage27
    if [ "$DRY" = 0 ]; then
        if [ ! -f "$RAW" ]; then
            echo "Stage 27 needs stage 26's precision audit ($RAW). Run stage 26 first."
            exit 1
        fi
        mkdir -p "$OUT" || exit 1
    fi
    for check in jitter1 jitter8 train; do
        case $check in
            jitter1) option="--jitter-draws 1" ;;
            jitter8) option="--jitter-draws 8" ;;
            train) option="--split train" ;;
        esac
        if [ "$DRY" = 1 ]; then
            echo "$PY $D/precision_audit.py $OUT/precision_$check.json $D/runs/$CEILING $D/runs/$NCE $option > $OUT/precision_$check.txt"
        else
            # shellcheck disable=SC2086
            "$PY" "$D/precision_audit.py" "$OUT/precision_$check.json" "$D/runs/$CEILING" "$D/runs/$NCE" $option \
                > "$OUT/precision_$check.txt" 2>&1 || { echo "precision_audit $option failed, see $OUT/precision_$check.txt"; exit 1; }
        fi
    done
    COMPARE="raw=$RAW jitter1=$OUT/precision_jitter1.json jitter8=$OUT/precision_jitter8.json train=$OUT/precision_train.json"
    if [ "$DRY" = 1 ]; then
        echo "$PY $D/precision_compare.py $COMPARE > $OUT/compare.txt"
    else
        # shellcheck disable=SC2086
        "$PY" "$D/precision_compare.py" $COMPARE > "$OUT/compare.txt" 2>&1 \
            || { echo "precision_compare failed, see $OUT/compare.txt"; exit 1; }
        echo "Stage 27 reports in $OUT (compare.txt has all four side by side)"
    fi
fi

if wants 28; then
    NOFIELD=$D/bank_causal_nofield
    bank "$NOFIELD" --t1-values 0.4 --causal --field off
    residual_bank "$NOFIELD" t1_04 flair
    NCE=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_c24_20k
    CEILING=ceiling_uni_jitter_nopatch_restail_t104_causal_nofield_c24_10k
    WITH_FIELD=ssl_uni_jitter_nopatch_restail_t104_causal_c24_20k
    RECIPE=(--uniform --residual --jitter 2 --rotate 3 --patch-weight 0 --content-units 24 --seed 42)
    EVERY=2000 ssl "$NCE" "$NOFIELD" tail t1_04 20000 "${RECIPE[@]}" --contrastive-loss-type infonce
    ssl "$CEILING" "$NOFIELD" tail t1_04 10000 "${RECIPE[@]}" --supervise volume
    finish "$NCE" "$CEILING"

    OUT=$D/causal/stage28
    if [ "$DRY" = 0 ]; then
        for name in "$CEILING" "$NCE"; do
            if ! grep -qx DONE "$D/logs/$name.log" || [ ! -f "$D/runs/$name/model.pt" ]; then
                echo "Stage 28 needs $name finished. Rerun stage 28 after it is."
                exit 1
            fi
        done
        mkdir -p "$OUT" || exit 1
    fi
    for report in precision_audit precision_train own_variation code_audit; do
        script=$report option="" runs="$D/runs/$CEILING $D/runs/$NCE"
        case $report in
            precision_train) script=precision_audit option="--split train" ;;
            own_variation | code_audit) runs="$runs $D/runs/$WITH_FIELD" ;;
        esac
        if [ "$DRY" = 1 ]; then
            echo "$PY $D/$script.py $OUT/$report.json $runs $option > $OUT/$report.txt"
        else
            # shellcheck disable=SC2086
            "$PY" "$D/$script.py" "$OUT/$report.json" $runs $option > "$OUT/$report.txt" 2>&1 \
                || { echo "$report failed, see $OUT/$report.txt"; exit 1; }
        fi
    done
    COMPARE="raw=$OUT/precision_audit.json train=$OUT/precision_train.json"
    if [ "$DRY" = 1 ]; then
        echo "$PY $D/precision_compare.py $COMPARE > $OUT/compare.txt"
    else
        # shellcheck disable=SC2086
        "$PY" "$D/precision_compare.py" $COMPARE > "$OUT/compare.txt" 2>&1 \
            || { echo "precision_compare failed, see $OUT/compare.txt"; exit 1; }
        echo "Stage 28 reports in $OUT"
    fi
fi

if wants 29; then
    NOFIELD=$D/bank_causal_nofield
    NOGAIN=$D/bank_causal_nofield_nogain
    bank "$NOFIELD" --t1-values 0.4 --causal --field off
    residual_bank "$NOFIELD" t1_04 flair
    bank "$NOGAIN" --t1-values 0.4 --causal --field off --gain off
    residual_bank "$NOGAIN" t1_04 flair
    NCE=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_c24_20k
    C48=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_c48_20k
    GAPREM=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_rem10gap_20k
    BT=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_bt_c24_20k
    GAINLESS=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_nogain_c24_20k
    BASE=(--uniform --residual --jitter 2 --rotate 3 --patch-weight 0 --seed 42)
    EVERY=2000 ssl "$NCE" "$NOFIELD" tail t1_04 20000 "${BASE[@]}" --content-units 24 --contrastive-loss-type infonce
    EVERY=2000 ssl "$C48" "$NOFIELD" tail t1_04 20000 "${BASE[@]}" --content-units 48
    EVERY=2000 ssl "$GAPREM" "$NOFIELD" tail t1_04 20000 "${BASE[@]}" --remainder-units 10 --remainder-neighbours 31 \
        --remainder-every 500 --remainder-space gap
    EVERY=2000 ssl "$BT" "$NOFIELD" tail t1_04 20000 "${BASE[@]}" --content-units 24 --contrastive-loss-type barlow_twins
    EVERY=2000 ssl "$GAINLESS" "$NOGAIN" tail t1_04 20000 "${BASE[@]}" --content-units 24 --contrastive-loss-type infonce
    finish "$NCE" "$C48" "$GAPREM" "$BT" "$GAINLESS"

    OUT=$D/causal/stage29
    RUNS="$D/runs/$NCE $D/runs/$C48 $D/runs/$GAPREM $D/runs/$BT $D/runs/$GAINLESS"
    if [ "$DRY" = 0 ]; then
        for name in "$NCE" "$C48" "$GAPREM" "$BT" "$GAINLESS"; do
            if ! grep -qx DONE "$D/logs/$name.log" || [ ! -f "$D/runs/$name/model.pt" ]; then
                echo "Stage 29 needs $name finished. Rerun stage 29 after it is."
                exit 1
            fi
        done
        mkdir -p "$OUT" || exit 1
    fi
    for report in precision_audit own_variation code_audit; do
        if [ "$DRY" = 1 ]; then
            echo "$PY $D/$report.py $OUT/$report.json $RUNS > $OUT/$report.txt"
        else
            # shellcheck disable=SC2086
            "$PY" "$D/$report.py" "$OUT/$report.json" $RUNS > "$OUT/$report.txt" 2>&1 \
                || { echo "$report failed, see $OUT/$report.txt"; exit 1; }
        fi
    done
    [ "$DRY" = 0 ] && echo "Stage 29 reports in $OUT"
fi

if wants 30; then
    SOURCE=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_c24_20k
    CONTROL=head_nofield_c24_nce_10k
    PREDICT01=head_nofield_c24_predict01_10k
    PREDICT1=head_nofield_c24_predict1_10k
    OUT=$D/causal/stage30
    CACHE=$OUT/features.pt
    BASE=(--cache "$CACHE" --draws 8 --batch "$BATCH" --seed 42 --whiten-ridge 0.1)
    if [ "$DRY" = 0 ]; then
        if ! grep -qx DONE "$D/logs/$SOURCE.log" || [ ! -f "$D/runs/$SOURCE/model.pt" ] || \
            [ ! -f "$D/runs/$SOURCE/report.json" ]; then
            echo "Stage 30 needs stage 28's $SOURCE finished. Run stage 28 first."
            exit 1
        fi
        mkdir -p "$OUT" || exit 1
    fi
    if [ "$DRY" = 1 ]; then
        echo "$PY $D/frozen_prediction.py $D/runs/$SOURCE $D/runs/$CONTROL 10000 2000 ${BASE[*]} --prepare-only > $OUT/features.log"
    else
        "$PY" "$D/frozen_prediction.py" "$D/runs/$SOURCE" "$D/runs/$CONTROL" 10000 2000 "${BASE[@]}" \
            --prepare-only > "$OUT/features.log" 2>&1 \
            || { echo "Feature preparation failed, see $OUT/features.log"; exit 1; }
    fi
    arm "$CONTROL" "$PY" "$D/frozen_prediction.py" "$D/runs/$SOURCE" "$D/runs/$CONTROL" 10000 2000 \
        "${BASE[@]}" --prediction-weight 0
    arm "$PREDICT01" "$PY" "$D/frozen_prediction.py" "$D/runs/$SOURCE" "$D/runs/$PREDICT01" 10000 2000 \
        "${BASE[@]}" --prediction-weight 0.1
    arm "$PREDICT1" "$PY" "$D/frozen_prediction.py" "$D/runs/$SOURCE" "$D/runs/$PREDICT1" 10000 2000 \
        "${BASE[@]}" --prediction-weight 1
    finish "$CONTROL" "$PREDICT01" "$PREDICT1"
    RUNS=("$D/runs/$SOURCE" "$D/runs/$CONTROL" "$D/runs/$PREDICT01" "$D/runs/$PREDICT1")
    if [ "$DRY" = 0 ]; then
        for name in "$CONTROL" "$PREDICT01" "$PREDICT1"; do
            if ! grep -qx DONE "$D/logs/$name.log" || [ ! -f "$D/runs/$name/model.pt" ] || \
                [ ! -f "$D/runs/$name/report.json" ]; then
                echo "Stage 30 needs $name finished. Rerun stage 30 after it is."
                exit 1
            fi
        done
    fi
    for report in precision_audit own_variation code_audit; do
        if [ "$DRY" = 1 ]; then
            echo "$PY $D/$report.py $OUT/$report.json ${RUNS[*]} > $OUT/$report.txt"
        else
            "$PY" "$D/$report.py" "$OUT/$report.json" "${RUNS[@]}" > "$OUT/$report.txt" 2>&1 \
                || { echo "$report failed, see $OUT/$report.txt"; exit 1; }
        fi
    done
    [ "$DRY" = 0 ] && echo "Stage 30 reports in $OUT"
fi

if wants 31; then
    NOFIELD=$D/bank_causal_nofield
    bank "$NOFIELD" --t1-values 0.4 --causal --field off
    residual_bank "$NOFIELD" t1_04 flair
    P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield
    NCE=${P}_c24_20k
    GAPREM=${P}_rem10gap_20k
    CRITMAIN=${P}_critmain_c24_20k
    CRITREM=${P}_rem10gap_critrem_20k
    CRITBOTH=${P}_rem10gap_critboth_20k
    STAGE29="$D/runs/$NCE $D/runs/${P}_c48_20k $D/runs/$GAPREM $D/runs/${P}_bt_c24_20k $D/runs/${P}_nogain_c24_20k"
    OUT=$D/causal/stage31
    if [ "$DRY" = 1 ]; then
        echo "$PY $D/precision_audit.py $D/causal/stage29/precision_audit.json $STAGE29 > $OUT/price_stage29.txt"
    else
        mkdir -p "$OUT" || exit 1
        echo "$(date +%H:%M) stage 29's codes re-priced with the critic refitted (no training)"
        # shellcheck disable=SC2086
        "$PY" "$D/precision_audit.py" "$D/causal/stage29/precision_audit.json" $STAGE29 > "$OUT/price_stage29.txt" 2>&1 \
            || echo "the re-price failed, see $OUT/price_stage29.txt; training goes on"
    fi
    BASE=(--uniform --residual --jitter 2 --rotate 3 --patch-weight 0 --seed 42)
    REM=(--remainder-units 10 --remainder-neighbours 31 --remainder-every 500 --remainder-space gap)
    EVERY=2000 ssl "$CRITMAIN" "$NOFIELD" tail t1_04 20000 "${BASE[@]}" --content-units 24 --contrastive-loss-type infonce \
        --critic-weights main
    EVERY=2000 ssl "$CRITREM" "$NOFIELD" tail t1_04 20000 "${BASE[@]}" "${REM[@]}" --critic-weights remainder
    EVERY=2000 ssl "$CRITBOTH" "$NOFIELD" tail t1_04 20000 "${BASE[@]}" "${REM[@]}" --critic-weights both
    finish "$CRITMAIN" "$CRITREM" "$CRITBOTH"

    RUNS="$D/runs/$CRITMAIN $D/runs/$NCE $D/runs/$CRITREM $D/runs/$CRITBOTH $D/runs/$GAPREM"
    if [ "$DRY" = 0 ]; then
        for name in "$CRITMAIN" "$NCE" "$CRITREM" "$CRITBOTH" "$GAPREM"; do
            if ! grep -qx DONE "$D/logs/$name.log" || [ ! -f "$D/runs/$name/model.pt" ]; then
                echo "Stage 31 needs $name finished (stage 28 trains $NCE, stage 29 $GAPREM). Rerun stage 31 after it is."
                exit 1
            fi
        done
    fi
    for report in precision_audit own_variation code_audit; do
        if [ "$DRY" = 1 ]; then
            echo "$PY $D/$report.py $OUT/$report.json $RUNS > $OUT/$report.txt"
        else
            # shellcheck disable=SC2086
            "$PY" "$D/$report.py" "$OUT/$report.json" $RUNS > "$OUT/$report.txt" 2>&1 \
                || { echo "$report failed, see $OUT/$report.txt"; exit 1; }
        fi
    done
    [ "$DRY" = 0 ] && echo "Stage 31 reports in $OUT"
fi

if wants 32; then
    NOFIELD=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_c24_20k
    NOGAIN=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_nogain_c24_20k
    CONTROL=head_nofield_c24_nce_10k
    PREDICT1=head_nofield_c24_predict1_10k
    PREDICT3=head_nofield_c24_predict3_10k
    PREDICT10=head_nofield_c24_predict10_10k
    NOGAIN_CONTROL=head_nogain_c24_nce_10k
    NOGAIN_PREDICT1=head_nogain_c24_predict1_10k
    OUT=$D/causal/stage32
    COMMON=(--draws 8 --batch "$BATCH" --seed 42 --whiten-ridge 0.1)
    NOFIELD_BASE=(--cache "$D/causal/stage30/features.pt" "${COMMON[@]}")
    NOGAIN_BASE=(--cache "$OUT/features_nogain.pt" "${COMMON[@]}")
    if [ "$DRY" = 0 ]; then
        for name in "$NOFIELD" "$NOGAIN" "$CONTROL" "$PREDICT1"; do
            if ! grep -qx DONE "$D/logs/$name.log" || [ ! -f "$D/runs/$name/model.pt" ] || \
                [ ! -f "$D/runs/$name/report.json" ]; then
                echo "Stage 32 needs $name finished (stages 28, 29 and 30 train these). Run those first."
                exit 1
            fi
        done
        mkdir -p "$OUT" || exit 1
    fi
    head_cache() { # head_cache SOURCE LOG ARGS...: check or build a feature cache before the parallel arms
        local source=$1 log=$2; shift 2
        if [ "$DRY" = 1 ]; then
            echo "$PY $D/frozen_prediction.py $D/runs/$source $OUT/unused 10000 2000 $* --prepare-only > $log"
            return
        fi
        "$PY" "$D/frozen_prediction.py" "$D/runs/$source" "$OUT/unused" 10000 2000 "$@" --prepare-only > "$log" 2>&1 \
            || { echo "Feature preparation failed, see $log"; exit 1; }
    }
    # Stage 30's cache is only read, and its settings must match, so the new field-off arms see stage 30's jitter draws
    # and batches. Preparation stops before it would create the output folder it is given.
    head_cache "$NOFIELD" "$OUT/features_nofield.log" "${NOFIELD_BASE[@]}"
    head_cache "$NOGAIN" "$OUT/features_nogain.log" "${NOGAIN_BASE[@]}"
    arm "$PREDICT3" "$PY" "$D/frozen_prediction.py" "$D/runs/$NOFIELD" "$D/runs/$PREDICT3" 10000 2000 \
        "${NOFIELD_BASE[@]}" --prediction-weight 3
    arm "$PREDICT10" "$PY" "$D/frozen_prediction.py" "$D/runs/$NOFIELD" "$D/runs/$PREDICT10" 10000 2000 \
        "${NOFIELD_BASE[@]}" --prediction-weight 10
    arm "$NOGAIN_CONTROL" "$PY" "$D/frozen_prediction.py" "$D/runs/$NOGAIN" "$D/runs/$NOGAIN_CONTROL" 10000 2000 \
        "${NOGAIN_BASE[@]}" --prediction-weight 0
    arm "$NOGAIN_PREDICT1" "$PY" "$D/frozen_prediction.py" "$D/runs/$NOGAIN" "$D/runs/$NOGAIN_PREDICT1" 10000 2000 \
        "${NOGAIN_BASE[@]}" --prediction-weight 1
    finish "$PREDICT3" "$PREDICT10" "$NOGAIN_CONTROL" "$NOGAIN_PREDICT1"
    RUNS=("$D/runs/$NOFIELD" "$D/runs/$CONTROL" "$D/runs/$PREDICT1" "$D/runs/$PREDICT3" "$D/runs/$PREDICT10"
        "$D/runs/$NOGAIN" "$D/runs/$NOGAIN_CONTROL" "$D/runs/$NOGAIN_PREDICT1")
    if [ "$DRY" = 0 ]; then
        for name in "$PREDICT3" "$PREDICT10" "$NOGAIN_CONTROL" "$NOGAIN_PREDICT1"; do
            if ! grep -qx DONE "$D/logs/$name.log" || [ ! -f "$D/runs/$name/model.pt" ] || \
                [ ! -f "$D/runs/$name/report.json" ]; then
                echo "Stage 32 needs $name finished. Rerun stage 32 after it is."
                exit 1
            fi
        done
    fi
    for report in precision_audit own_variation code_audit; do
        if [ "$DRY" = 1 ]; then
            echo "$PY $D/$report.py $OUT/$report.json ${RUNS[*]} > $OUT/$report.txt"
        else
            "$PY" "$D/$report.py" "$OUT/$report.json" "${RUNS[@]}" > "$OUT/$report.txt" 2>&1 \
                || { echo "$report failed, see $OUT/$report.txt"; exit 1; }
        fi
    done
    [ "$DRY" = 0 ] && echo "Stage 32 reports in $OUT"
fi

if wants 33; then
    NOFIELD=$D/bank_causal_nofield
    NOGAIN=$D/bank_causal_nofield_nogain
    bank "$NOFIELD" --t1-values 0.4 --causal --field off
    residual_bank "$NOFIELD" t1_04 flair
    bank "$NOGAIN" --t1-values 0.4 --causal --field off --gain off
    residual_bank "$NOGAIN" t1_04 flair
    P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield
    NCE=${P}_c24_20k
    GAINLESS=${P}_nogain_c24_20k
    PREDICT1=${P}_predict1_c24_20k
    PREDICT3=${P}_predict3_c24_20k
    GAINLESS_PREDICT1=${P}_nogain_predict1_c24_20k
    BASE=(--uniform --residual --jitter 2 --rotate 3 --patch-weight 0 --seed 42 --content-units 24
        --contrastive-loss-type infonce)
    EVERY=2000 ssl "$PREDICT1" "$NOFIELD" tail t1_04 20000 "${BASE[@]}" --predict-weight 1
    EVERY=2000 ssl "$PREDICT3" "$NOFIELD" tail t1_04 20000 "${BASE[@]}" --predict-weight 3
    EVERY=2000 ssl "$GAINLESS_PREDICT1" "$NOGAIN" tail t1_04 20000 "${BASE[@]}" --predict-weight 1
    finish "$PREDICT1" "$PREDICT3" "$GAINLESS_PREDICT1"

    OUT=$D/causal/stage33
    RUNS=("$D/runs/$PREDICT1" "$D/runs/$PREDICT3" "$D/runs/$NCE" "$D/runs/$GAINLESS_PREDICT1" "$D/runs/$GAINLESS")
    if [ "$DRY" = 0 ]; then
        for name in "$PREDICT1" "$PREDICT3" "$NCE" "$GAINLESS_PREDICT1" "$GAINLESS"; do
            if ! grep -qx DONE "$D/logs/$name.log" || [ ! -f "$D/runs/$name/model.pt" ]; then
                echo "Stage 33 needs $name finished (stage 28 trains $NCE, stage 29 $GAINLESS)." \
                    "Rerun stage 33 after it is."
                exit 1
            fi
        done
        mkdir -p "$OUT" || exit 1
    fi
    for name in head_nofield_c24_predict3_10k head_nogain_c24_predict1_10k; do  # stage 32's heads, for reference
        if [ "$DRY" = 1 ] || grep -qx DONE "$D/logs/$name.log" 2>/dev/null; then RUNS+=("$D/runs/$name"); fi
    done
    for report in precision_audit own_variation code_audit; do
        if [ "$DRY" = 1 ]; then
            echo "$PY $D/$report.py $OUT/$report.json ${RUNS[*]} > $OUT/$report.txt"
        else
            "$PY" "$D/$report.py" "$OUT/$report.json" "${RUNS[@]}" > "$OUT/$report.txt" 2>&1 \
                || { echo "$report failed, see $OUT/$report.txt"; exit 1; }
        fi
    done
    [ "$DRY" = 0 ] && echo "Stage 33 reports in $OUT"
fi

stage_report() { # stage_report OUT REPORT RUN...: REPORT.py's table for the runs into OUT/REPORT.txt
    local out=$1 report=$2; shift 2
    if [ "$DRY" = 1 ]; then echo "$PY $D/$report.py $out/$report.json $* > $out/$report.txt"; return; fi
    echo "$(date +%H:%M) $report into $out"
    "$PY" "$D/$report.py" "$out/$report.json" "$@" > "$out/$report.txt" 2>&1 \
        || { echo "$report failed, see $out/$report.txt"; exit 1; }
}

if wants 34; then
    NOFIELD=$D/bank_causal_nofield
    NOGAIN=$D/bank_causal_nofield_nogain
    bank "$NOFIELD" --t1-values 0.4 --causal --field off
    residual_bank "$NOFIELD" t1_04 flair
    bank "$NOGAIN" --t1-values 0.4 --causal --field off --gain off
    residual_bank "$NOGAIN" t1_04 flair
    P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield
    NCE=${P}_c24_20k
    GAINLESS=${P}_nogain_c24_20k
    PREDICT1=${P}_predict1_c24_20k
    PREDICT3=${P}_predict3_c24_20k
    GAINLESS_PREDICT1=${P}_nogain_predict1_c24_20k
    OUT=$D/causal/stage34
    if [ "$DRY" = 0 ]; then
        for name in "$NCE" "$GAINLESS" "$PREDICT1" "$PREDICT3" "$GAINLESS_PREDICT1"; do
            if ! grep -qx DONE "$D/logs/$name.log" || [ ! -f "$D/runs/$name/model.pt" ]; then
                echo "Stage 34 needs $name finished (stages 28, 29 and 33 train it). Rerun stage 34 after it is."
                exit 1
            fi
        done
        mkdir -p "$OUT" || exit 1
        # Start from stage 33's assessments, so the reports score only the seed-2 runs (reused at the same step).
        for file in "$D"/causal/stage33/{precision_audit,own_variation,code_audit}.json "$D"/causal/stage33/image_*; do
            if [ -e "$file" ] && [ ! -e "$OUT/${file##*/}" ]; then cp "$file" "$OUT/"; fi
        done
    fi
    BASE=(--uniform --residual --jitter 2 --rotate 3 --patch-weight 0 --seed 2 --content-units 24
        --contrastive-loss-type infonce)
    EVERY=2000 ssl "${PREDICT1}_s2" "$NOFIELD" tail t1_04 20000 "${BASE[@]}" --predict-weight 1
    EVERY=2000 ssl "${GAINLESS_PREDICT1}_s2" "$NOGAIN" tail t1_04 20000 "${BASE[@]}" --predict-weight 1

    STYLED=("$D/runs/$PREDICT1" "$D/runs/$PREDICT3" "$D/runs/$NCE" "$D/runs/$GAINLESS_PREDICT1" "$D/runs/$GAINLESS")
    for name in head_nofield_c24_predict3_10k head_nogain_c24_predict1_10k; do  # stage 32's heads
        if [ "$DRY" = 1 ] || grep -qx DONE "$D/logs/$name.log" 2>/dev/null; then STYLED+=("$D/runs/$name"); fi
    done
    stage_report "$OUT" style_audit "${STYLED[@]}"  # while the seed-2 arms train
    finish "${PREDICT1}_s2" "${GAINLESS_PREDICT1}_s2"
    if [ "$DRY" = 0 ]; then
        for name in "${PREDICT1}_s2" "${GAINLESS_PREDICT1}_s2"; do
            grep -qx DONE "$D/logs/$name.log" || { echo "Stage 34's style audit is in $OUT; rerun for $name."; exit 1; }
        done
    fi
    SEEDS=("$D/runs/$PREDICT1" "$D/runs/${PREDICT1}_s2" "$D/runs/$NCE"
        "$D/runs/$GAINLESS_PREDICT1" "$D/runs/${GAINLESS_PREDICT1}_s2" "$D/runs/$GAINLESS")
    for report in precision_audit own_variation code_audit; do stage_report "$OUT" "$report" "${SEEDS[@]}"; done
    stage_report "$OUT" style_audit "${STYLED[@]}" "$D/runs/${PREDICT1}_s2" "$D/runs/${GAINLESS_PREDICT1}_s2"
    [ "$DRY" = 0 ] && echo "Stage 34 reports in $OUT"
fi

reuse_reports() { # reuse_reports FROM OUT NAME...: FROM's four report entries for these runs, and its feature caches
    local from=$1 out=$2; shift 2
    [ "$DRY" = 1 ] && return
    "$PY" - "$from" "$out" "$@" <<'EOF'
import json
import shutil
import sys
from pathlib import Path

src, dst, names = Path(sys.argv[1]), Path(sys.argv[2]), set(sys.argv[3:])
for report in ("precision_audit", "own_variation", "code_audit", "style_audit"):
    source, target = src / f"{report}.json", dst / f"{report}.json"
    if source.exists() and not target.exists():
        kept = {name: row for name, row in json.loads(source.read_text()).items() if name in names}
        target.write_text(json.dumps(kept, indent=1))
for cache in [*src.glob("image_*.npy"), *src.glob("style_*.npz")]:
    if not (dst / cache.name).exists():
        shutil.copy(cache, dst)
EOF
}

if wants 35; then
    NOFIELD=$D/bank_causal_nofield
    NOGAIN=$D/bank_causal_nofield_nogain
    bank "$NOFIELD" --t1-values 0.4 --causal --field off
    residual_bank "$NOFIELD" t1_04 flair
    bank "$NOGAIN" --t1-values 0.4 --causal --field off --gain off
    residual_bank "$NOGAIN" t1_04 flair
    P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield
    MATCHED=("${P}_predict1_c10_20k" "${P}_c10_20k" "${P}_nogain_predict1_c10_20k")
    WIDE=("${P}_predict1_c24_20k" "${P}_c24_20k" "${P}_nogain_predict1_c24_20k")  # stages 33 and 28, 24 units
    OUT=$D/causal/stage35
    if [ "$DRY" = 0 ]; then
        mkdir -p "$OUT" || exit 1
        reuse_reports "$D/causal/stage34" "$OUT" "${WIDE[@]}"
    fi
    BASE=(--uniform --residual --jitter 2 --rotate 3 --patch-weight 0 --seed 42 --content-units 10
        --contrastive-loss-type infonce)
    EVERY=2000 ssl "${MATCHED[0]}" "$NOFIELD" tail t1_04 20000 "${BASE[@]}" --predict-weight 1
    EVERY=2000 ssl "${MATCHED[1]}" "$NOFIELD" tail t1_04 20000 "${BASE[@]}"
    EVERY=2000 ssl "${MATCHED[2]}" "$NOGAIN" tail t1_04 20000 "${BASE[@]}" --predict-weight 1
    finish "${MATCHED[@]}"
    RUNS=()
    for name in "${MATCHED[@]}"; do
        if [ "$DRY" = 0 ] && ! grep -qx DONE "$D/logs/$name.log"; then echo "Rerun stage 35 for $name."; exit 1; fi
        RUNS+=("$D/runs/$name")
    done
    for name in "${WIDE[@]}"; do
        if [ "$DRY" = 1 ] || grep -qx DONE "$D/logs/$name.log" 2>/dev/null; then RUNS+=("$D/runs/$name"); fi
    done
    for report in precision_audit own_variation code_audit style_audit; do
        stage_report "$OUT" "$report" "${RUNS[@]}"
    done
    [ "$DRY" = 0 ] && echo "Stage 35 reports in $OUT"
fi

if wants 36; then
    NOFIELD=$D/bank_causal_nofield
    bank "$NOFIELD" --t1-values 0.4 --causal --field off
    residual_bank "$NOFIELD" t1_04 flair
    P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield
    MLP=("${P}_predict1_mlp128_c10_20k" "${P}_predict1_mlp128_critmain_c10_20k" "${P}_predict1_mlp128_c24_20k")
    LINEAR=("${P}_predict1_c10_20k" "${P}_predict1_c24_20k" "${P}_c10_20k")  # stages 35 and 33, linear maps
    CHECKED=("${P}_predict1_c24_20k" "${P}_predict1_c10_20k" "${P}_nogain_predict1_c24_20k"
        "${P}_nogain_predict1_c10_20k")  # the no-training check's runs (stages 33 and 35)
    OUT=$D/causal/stage36
    if [ "$DRY" = 0 ]; then
        for name in "${LINEAR[@]}"; do
            if ! grep -qx DONE "$D/logs/$name.log" || [ ! -f "$D/runs/$name/model.pt" ]; then
                echo "Stage 36 needs $name finished (stages 33 and 35 train it). Rerun stage 36 after it is."
                exit 1
            fi
        done
        mkdir -p "$OUT" || exit 1
        reuse_reports "$D/causal/stage35" "$OUT" "${LINEAR[@]}"
    fi
    BASE=(--uniform --residual --jitter 2 --rotate 3 --patch-weight 0 --seed 42 --contrastive-loss-type infonce
        --predict-weight 1 --predict-hidden 128)
    EVERY=2000 ssl "${MLP[0]}" "$NOFIELD" tail t1_04 20000 "${BASE[@]}" --content-units 10
    EVERY=2000 ssl "${MLP[1]}" "$NOFIELD" tail t1_04 20000 "${BASE[@]}" --content-units 10 --critic-weights main
    EVERY=2000 ssl "${MLP[2]}" "$NOFIELD" tail t1_04 20000 "${BASE[@]}" --content-units 24
    RUNS=()
    for name in "${CHECKED[@]}"; do
        if [ "$DRY" = 1 ] || grep -qx DONE "$D/logs/$name.log" 2>/dev/null; then RUNS+=("$D/runs/$name"); fi
    done
    if [ "$DRY" = 1 ]; then
        echo "$PY $D/bottleneck_check.py $OUT/bottleneck_check.json ${RUNS[*]} > $OUT/bottleneck_check.txt"
    else
        echo "$(date +%H:%M) no-training bottleneck check into $OUT, while the arms train"
        "$PY" "$D/bottleneck_check.py" "$OUT/bottleneck_check.json" "${RUNS[@]}" > "$OUT/bottleneck_check.txt" 2>&1 \
            || echo "the bottleneck check failed, see $OUT/bottleneck_check.txt; training goes on"
    fi
    finish "${MLP[@]}"
    RUNS=()
    for name in "${MLP[@]}"; do
        if [ "$DRY" = 0 ] && ! grep -qx DONE "$D/logs/$name.log"; then
            echo "Stage 36's check is in $OUT; rerun stage 36 for $name."
            exit 1
        fi
        RUNS+=("$D/runs/$name")
    done
    for name in "${LINEAR[@]}"; do RUNS+=("$D/runs/$name"); done
    stage_report "$OUT" precision_audit "${RUNS[@]}"
    stage_report "$OUT" own_variation "${RUNS[@]}" --boosted
    stage_report "$OUT" code_audit "${RUNS[@]}"
    stage_report "$OUT" style_audit "${RUNS[@]}"
    [ "$DRY" = 0 ] && echo "Stage 36 reports in $OUT"
fi

if wants 37; then
    NOFIELD=$D/bank_causal_nofield
    NOGAIN=$D/bank_causal_nofield_nogain
    bank "$NOFIELD" --t1-values 0.4 --causal --field off
    residual_bank "$NOFIELD" t1_04 flair
    bank "$NOGAIN" --t1-values 0.4 --causal --field off --gain off
    residual_bank "$NOGAIN" t1_04 flair
    K32=${NOFIELD}_k32
    K32_NOGAIN=${NOGAIN}_k32
    residual_bank "$K32" t1_04 flair --components 32 --source "$NOFIELD"
    residual_bank "$K32_NOGAIN" t1_04 flair --components 32 --source "$NOGAIN"
    P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield
    SHARP=("${P}_k32_predict1_c10_20k" "${P}_nogain_k32_predict1_c10_20k" "${P}_k32_predict1_c24_20k")
    BEFORE=("${P}_predict1_c10_20k" "${P}_nogain_predict1_c10_20k" "${P}_predict1_c24_20k")  # 16 components
    OUT=$D/causal/stage37
    if [ "$DRY" = 0 ]; then
        mkdir -p "$OUT" || exit 1
        reuse_reports "$D/causal/stage35" "$OUT" "${BEFORE[@]}"
    fi
    BASE=(--uniform --residual --jitter 2 --rotate 3 --patch-weight 0 --seed 42 --contrastive-loss-type infonce
        --predict-weight 1)
    EVERY=2000 ssl "${SHARP[0]}" "$K32" tail t1_04 20000 "${BASE[@]}" --content-units 10
    EVERY=2000 ssl "${SHARP[1]}" "$K32_NOGAIN" tail t1_04 20000 "${BASE[@]}" --content-units 10
    EVERY=2000 ssl "${SHARP[2]}" "$K32" tail t1_04 20000 "${BASE[@]}" --content-units 24
    finish "${SHARP[@]}"
    RUNS=()
    for name in "${SHARP[@]}"; do
        if [ "$DRY" = 0 ] && ! grep -qx DONE "$D/logs/$name.log"; then echo "Rerun stage 37 for $name."; exit 1; fi
        RUNS+=("$D/runs/$name")
    done
    for name in "${BEFORE[@]}"; do
        if [ "$DRY" = 1 ] || grep -qx DONE "$D/logs/$name.log" 2>/dev/null; then RUNS+=("$D/runs/$name"); fi
    done
    for report in precision_audit own_variation code_audit style_audit; do
        stage_report "$OUT" "$report" "${RUNS[@]}"
    done
    [ "$DRY" = 0 ] && echo "Stage 37 reports in $OUT"
    for source in "$NOFIELD" "$NOGAIN"; do  # no training; after the reports, so it can never hold them up
        report=$OUT/normative_components_${source##*/}
        if [ "$DRY" = 1 ]; then
            echo "$PY $D/normative_components.py $source $report.json > $report.txt"
        elif [ ! -f "$report.json" ]; then
            echo "$(date +%H:%M) normative components of ${source##*/} into $OUT"
            "$PY" "$D/normative_components.py" "$source" "$report.json" > "$report.txt" 2>&1 \
                || echo "the normative-components check failed, see $report.txt"
        fi
    done
fi

echo "$(date +%H:%M) all requested stages done"
