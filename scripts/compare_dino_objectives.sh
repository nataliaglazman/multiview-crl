#!/usr/bin/env bash
# Compare DINO fine-tuning objectives end to end: extract -> score -> plot.
#
#   bash scripts/compare_dino_objectives.sh
#   ARMS="infonce barlow vicreg" NUM_SAMPLES=2000 bash scripts/compare_dino_objectives.sh
#   FRESH=1 bash scripts/compare_dino_objectives.sh        # re-use of an output dir is refused otherwise
#
# Produces TWO comparisons, because they are not the same question:
#
#   content/  the objective ablation. Every fine-tuned arm carries a content/style
#             partition, so this scores the block the loss actually acted on. The
#             pretrained baseline has no partition and is therefore NOT in this table.
#   all/      the full embedding, which every arm has -- so the pretrained baseline can
#             sit beside the fine-tuned ones here. Answers "did fine-tuning help at all",
#             where content/ answers "which objective".
#
# Every bundle is built from ONE --run-dir with ONE --num-samples so they describe the
# same evaluation rows; compare_bundles verifies that from the stored factor digests and
# refuses to build a table if they disagree.
set -euo pipefail

RUN_DIR=${RUN_DIR:-results/synthetic/synthetic-clean-content-causal-ident-vent-12-4-2}
DINO_REPO=${DINO_REPO:-../3DINO}
PRETRAINED=${PRETRAINED:-../3DINO/3dino_vit_weights.pth}
# An arm is either a bare name, whose run directory is <ARM_DIR>_<name>, or an explicit
# name=dir pair. The pair form exists because the naming convention only fits arms that
# were launched together: a run added later (a pairing control, a re-run on another
# backbone) lives wherever --output-dir put it, and renaming a finished run to satisfy a
# script is how a comparison ends up pointing at the wrong checkpoint.
#   ARMS="infonce within=results/dino_within" bash scripts/compare_dino_objectives.sh
ARMS=${ARMS:-"infonce barlow"}
ARM_DIR=${ARM_DIR:-results/dino_new}          # <ARM_DIR>_<arm> for arms given as a bare name
OUT=${OUT:-results/dino_objective_comparison}
NUM_SAMPLES=${NUM_SAMPLES:-2000}
DEVICE=${DEVICE:-cuda}
VOLUME_BATCH=${VOLUME_BATCH:-2}
GRAPH_REPEATS=${GRAPH_REPEATS:-20}
ALPHA=${ALPHA:-0.05}
VIEW=${VIEW:-1}                               # which modality's embedding is scored
WITH_PRETRAINED=${WITH_PRETRAINED:-1}
FRESH=${FRESH:-0}

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

arm_name () { printf '%s' "${1%%=*}"; }
arm_dir  () { case "$1" in *=*) printf '%s' "${1#*=}";; *) printf '%s' "${ARM_DIR}_${1}";; esac; }

if [ -e "$OUT" ] && [ "$FRESH" != "1" ]; then
  echo "ERROR: $OUT already exists. Re-run with FRESH=1 to replace it, or set OUT=..." >&2
  echo "       (the extraction stage refuses a non-empty --output-dir, so a half-finished" >&2
  echo "        run cannot be silently mixed with a new one.)" >&2
  exit 1
fi
[ "$FRESH" = "1" ] && rm -rf "$OUT"

# --- preflight -------------------------------------------------------------------
# Catch the failure that is invisible later: a run whose recorded objective is not the
# one its directory name claims. Two arms that both trained InfoNCE under different
# names would produce a clean-looking table comparing a model with itself.
say "Checking each fine-tune run"
BACKBONES=$(mktemp)
RECORDED=$(mktemp)
trap 'rm -f "$BACKBONES" "$RECORDED"' EXIT
for SPEC in $ARMS; do
  ARM=$(arm_name "$SPEC"); DIR=$(arm_dir "$SPEC")
  for NEED in encoder preprocessing.json settings.json training_config.json; do
    [ -e "$DIR/$NEED" ] || { echo "ERROR: $DIR/$NEED is missing -- did that arm finish?" >&2; exit 1; }
  done
  python - "$DIR" "$ARM" "$BACKBONES" "$RECORDED" <<'PY'
import json, sys
from training.finetune_dino import OBJECTIVES
run, arm = sys.argv[1], sys.argv[2]
recorded = json.load(open(f"{run}/training_config.json")).get("objective")
# Only check the name against the arm when the arm IS a bare loss name. An arm can also be
# a pairing variant ("within" running infonce on augmented single-modality views), whose
# recorded name is legitimately not OBJECTIVES[arm]; the invariant that actually protects
# the comparison is checked below -- no two arms may record the SAME objective.
if arm in OBJECTIVES and recorded != OBJECTIVES[arm]:
    raise SystemExit(f"ERROR: {run} recorded objective={recorded!r}, expected {OBJECTIVES[arm]!r} for arm {arm!r}")
open(sys.argv[4], "a").write(f"{arm}\t{recorded}\n")
# The backbone decides how this arm is extracted, and it is only knowable from the
# recorded preprocessing: --three-dino-repo/--three-dino-weights without --backbone 3dino
# used to run the 2D slice encoder and never open the local checkpoint, and such a run
# trains and logs indistinguishably.
pre = json.load(open(f"{run}/preprocessing.json"))
epochs = sum(1 for _ in open(f"{run}/metrics.jsonl"))
print(f"  {arm:8s} {run}")
print(f"           objective={recorded}  backbone={pre['backbone']}  token_pool={pre.get('token_pool')}  epochs_logged={epochs}")
open(sys.argv[3], "a").write(pre["backbone"] + "\n")
PY
done
# Every arm must share a backbone. A 2D-slice arm beside a full-volume one is not an
# objective ablation -- it is a different encoder reading different inputs.
# The invariant: two arms that recorded the same objective are the same experiment under
# two names, and the table would compare a model with itself while looking perfectly clean.
if [ "$(cut -f2 "$RECORDED" | sort -u | wc -l)" -ne "$(wc -l < "$RECORDED")" ]; then
  echo "ERROR: two arms recorded the same objective -- they are the same experiment:" >&2
  sed 's/^/       /' "$RECORDED" >&2
  exit 1
fi

BACKEND=$(sort -u "$BACKBONES")
if [ "$(printf '%s' "$BACKEND" | wc -l)" -gt 0 ]; then
  echo "ERROR: the arms were trained with different backbones:" >&2
  sort -u "$BACKBONES" | sed 's/^/       /' >&2
  echo "       Compare arms that share one encoder, or re-run the odd one out." >&2
  exit 1
fi
echo "  -> all arms use backbone=$BACKEND"

# Only now: a preflight failure should leave nothing behind, or the retry would trip the
# already-exists guard above and demand FRESH=1 for a run that never started.
mkdir -p "$OUT"

# --- extract ---------------------------------------------------------------------
# One helper per backbone, because they load an encoder differently: 3DINO takes a repo
# plus a local checkpoint and the pipeline extracts trained+floor in one call, while the
# 2D path is an ordinary HF load where a fine-tuned run's `encoder/` directory IS the
# model id. Both write the same two files, so everything downstream is identical.
extract_arm () {            # $1 = weights/model, $2 = preprocessing.json, $3 = dest dir
  local MODEL="$1" PRE="$2" DEST="$3"
  if [ "$BACKEND" = "3dino" ]; then
    python -m eval.run_3dino_identifiability \
      --three-dino-repo "$DINO_REPO" --three-dino-weights "$MODEL" \
      --preprocessing "$PRE" --run-dir "$RUN_DIR" --output-dir "$DEST" \
      --num-samples "$NUM_SAMPLES" --causal match \
      --volume-batch "$VOLUME_BATCH" --device "$DEVICE" \
      --with-floor --no-graph
  else
    mkdir -p "$DEST"
    # --preprocessing restores the pooling/window the arm was fine-tuned under, and
    # embedding_partition.json next to the weights gives the content/style arrays. The
    # floor is the same architecture unloaded, so it inherits both.
    python -m eval.dinov3_embed_synthetic \
      --model-id "$MODEL" --preprocessing "$PRE" --run-dir "$RUN_DIR" \
      --out "$DEST/embeddings.npz" \
      --num-samples "$NUM_SAMPLES" --causal match --views 1 2 --device "$DEVICE"
    python -m eval.dinov3_embed_synthetic \
      --model-id "$MODEL" --preprocessing "$PRE" --run-dir "$RUN_DIR" \
      --out "$DEST/random_init.npz" --random-init --model-seed 0 \
      --num-samples "$NUM_SAMPLES" --causal match --views 1 2 --device "$DEVICE"
  fi
}

BUNDLES_CONTENT=()
BUNDLES_ALL=()
FLOORS_CONTENT=()
FLOORS_ALL=()
for SPEC in $ARMS; do
  ARM=$(arm_name "$SPEC"); DIR=$(arm_dir "$SPEC")
  say "Extracting $ARM from $DIR (+ its untrained floor)"
  extract_arm "$DIR/encoder" "$DIR/preprocessing.json" "$OUT/extract/$ARM"
  BUNDLES_CONTENT+=("$ARM=$OUT/extract/$ARM/embeddings.npz")
  FLOORS_CONTENT+=("$ARM=$OUT/extract/$ARM/random_init.npz")
  BUNDLES_ALL+=("$ARM=$OUT/extract/$ARM/embeddings.npz")
  FLOORS_ALL+=("$ARM=$OUT/extract/$ARM/random_init.npz")
done

if [ "$WITH_PRETRAINED" = "1" ]; then
  FIRST_DIR=$(arm_dir "$(echo "$ARMS" | awk '{print $1}')")
  FIRST_PRE="$FIRST_DIR/preprocessing.json"
  if [ "$BACKEND" = "3dino" ]; then
    BASE="$PRETRAINED"
  else
    # Whatever this arm was fine-tuned FROM, so the baseline is its starting point rather
    # than some other checkpoint that happens to share an architecture.
    BASE=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['model_id'])" \
             "$FIRST_DIR/training_config.json")
  fi
  say "Extracting the pretrained baseline ($BASE) on the SAME preprocessing"
  # Forced onto a fine-tuned arm's preprocessing.json: without this the baseline is
  # measured through a different window/pooling and the difference is preprocessing as
  # much as training. It has no partition, so it can only join the all-block table.
  extract_arm "$BASE" "$FIRST_PRE" "$OUT/extract/pretrained"
  BUNDLES_ALL=("pretrained=$OUT/extract/pretrained/embeddings.npz" "${BUNDLES_ALL[@]}")
  FLOORS_ALL=("pretrained=$OUT/extract/pretrained/random_init.npz" "${FLOORS_ALL[@]}")
fi

# --- score -----------------------------------------------------------------------
compare () {              # $1 = output dir name, $2 = --representation value, then bundles/floors
  local NAME="$1" BLOCK="$2"; shift 2
  mkdir -p "$OUT/$NAME"
  # --representation is what makes these two tables different questions. Without it both
  # scored the whole embedding and content/ was a copy of all/ minus the baseline row.
  python -m eval.compare_bundles "$@" \
    --view "$VIEW" --representation "$BLOCK" \
    --equal-width --with-graph --holdout-readout \
    --graph-repeats "$GRAPH_REPEATS" \
    --alphas "$ALPHA" --diagnostic-alpha "$ALPHA" \
    --out "$OUT/$NAME/compare.json" --csv "$OUT/$NAME/compare.csv"
  python -m eval.plot_compare_bundles --json "$OUT/$NAME/compare.json" --out "$OUT/$NAME/figures"
}

say "Scoring the content block (objective ablation)"
compare content content --bundles "${BUNDLES_CONTENT[@]}" --floors "${FLOORS_CONTENT[@]}"

say "Scoring the full embedding (with the pretrained baseline)"
compare all all --bundles "${BUNDLES_ALL[@]}" --floors "${FLOORS_ALL[@]}"

say "Done"
echo "  tables   $OUT/{content,all}/compare.txt"
echo "  figures  $OUT/{content,all}/figures/"
echo "  csv      $OUT/{content,all}/compare.csv  (+ compare_graph.csv)"
