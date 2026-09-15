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
ARMS=${ARMS:-"infonce barlow"}
ARM_DIR=${ARM_DIR:-results/dino_new}          # <ARM_DIR>_<arm> is each fine-tune run
OUT=${OUT:-results/dino_objective_comparison}
NUM_SAMPLES=${NUM_SAMPLES:-2000}
DEVICE=${DEVICE:-cuda}
VOLUME_BATCH=${VOLUME_BATCH:-2}
GRAPH_REPEATS=${GRAPH_REPEATS:-20}
ALPHA=${ALPHA:-0.05}
WITH_PRETRAINED=${WITH_PRETRAINED:-1}
FRESH=${FRESH:-0}

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

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
for ARM in $ARMS; do
  DIR="${ARM_DIR}_${ARM}"
  for NEED in encoder preprocessing.json settings.json training_config.json; do
    [ -e "$DIR/$NEED" ] || { echo "ERROR: $DIR/$NEED is missing -- did that arm finish?" >&2; exit 1; }
  done
  python - "$DIR" "$ARM" <<'PY'
import json, sys
from training.finetune_dino import OBJECTIVES
run, arm = sys.argv[1], sys.argv[2]
recorded = json.load(open(f"{run}/training_config.json")).get("objective")
expected = OBJECTIVES[arm]
if recorded != expected:
    raise SystemExit(f"ERROR: {run} recorded objective={recorded!r}, expected {expected!r} for arm {arm!r}")
# The backbone too. Passing --three-dino-repo/--three-dino-weights without --backbone
# 3dino used to run the 2D slice encoder and ignore the local checkpoint entirely; those
# runs train and log normally, so the recorded preprocessing is the only way to tell.
pre = json.load(open(f"{run}/preprocessing.json"))
if pre.get("backbone") != "3dino":
    raise SystemExit(
        f"ERROR: {run} was trained with backbone={pre.get('backbone')!r}, not '3dino' -- the 2D "
        f"slice encoder ran and the 3DINO weights were never loaded. Re-run that arm with "
        f"--backbone 3dino."
    )
epochs = sum(1 for _ in open(f"{run}/metrics.jsonl"))
print(f"  {arm:8s} objective={recorded}  backbone={pre['backbone']}  token_pool={pre.get('token_pool')}  epochs_logged={epochs}")
PY
done
# Only now: a preflight failure should leave nothing behind, or the retry would trip the
# already-exists guard above and demand FRESH=1 for a run that never started.
mkdir -p "$OUT"

# --- extract ---------------------------------------------------------------------
# The fine-tuned arms: preprocessing.json and embedding_partition.json are found next to
# the encoder, so the saved preprocessing wins over any pooling flag -- which is what
# makes the arms comparable to each other.
BUNDLES_CONTENT=()
BUNDLES_ALL=()
FLOORS_CONTENT=()
FLOORS_ALL=()
for ARM in $ARMS; do
  say "Extracting $ARM (+ its untrained floor)"
  python -m eval.run_3dino_identifiability \
    --three-dino-repo "$DINO_REPO" \
    --three-dino-weights "${ARM_DIR}_${ARM}/encoder" \
    --run-dir "$RUN_DIR" \
    --output-dir "$OUT/extract/$ARM" \
    --num-samples "$NUM_SAMPLES" --causal match \
    --volume-batch "$VOLUME_BATCH" --device "$DEVICE" \
    --with-floor --no-graph
  BUNDLES_CONTENT+=("$ARM=$OUT/extract/$ARM/embeddings.npz")
  FLOORS_CONTENT+=("$ARM=$OUT/extract/$ARM/random_init.npz")
  BUNDLES_ALL+=("$ARM=$OUT/extract/$ARM/embeddings.npz")
  FLOORS_ALL+=("$ARM=$OUT/extract/$ARM/random_init.npz")
done

if [ "$WITH_PRETRAINED" = "1" ]; then
  FIRST_ARM=$(echo "$ARMS" | awk '{print $1}')
  say "Extracting the pretrained baseline on the SAME preprocessing"
  # Forced onto a fine-tuned arm's preprocessing.json: without this the baseline is
  # measured through a different window/pooling and the difference is preprocessing as
  # much as training. It has no partition, so it can only join the all-block table.
  python -m eval.run_3dino_identifiability \
    --three-dino-repo "$DINO_REPO" \
    --three-dino-weights "$PRETRAINED" \
    --run-dir "$RUN_DIR" \
    --preprocessing "${ARM_DIR}_${FIRST_ARM}/preprocessing.json" \
    --output-dir "$OUT/extract/pretrained" \
    --num-samples "$NUM_SAMPLES" --causal match \
    --volume-batch "$VOLUME_BATCH" --device "$DEVICE" \
    --with-floor --no-graph
  BUNDLES_ALL=("pretrained=$OUT/extract/pretrained/embeddings.npz" "${BUNDLES_ALL[@]}")
  FLOORS_ALL=("pretrained=$OUT/extract/pretrained/random_init.npz" "${FLOORS_ALL[@]}")
fi

# --- score -----------------------------------------------------------------------
compare () {              # $1 = view name, $2 = --view value, then bundles/floors
  local NAME="$1" VIEW="$2"; shift 2
  mkdir -p "$OUT/$NAME"
  python -m eval.compare_bundles "$@" \
    --view "$VIEW" \
    --equal-width --with-graph --holdout-readout \
    --graph-repeats "$GRAPH_REPEATS" \
    --alphas "$ALPHA" --diagnostic-alpha "$ALPHA" \
    --out "$OUT/$NAME/compare.json" --csv "$OUT/$NAME/compare.csv"
  python -m eval.plot_compare_bundles --json "$OUT/$NAME/compare.json" --out "$OUT/$NAME/figures"
}

say "Scoring the content block (objective ablation)"
compare content 1 --bundles "${BUNDLES_CONTENT[@]}" --floors "${FLOORS_CONTENT[@]}"

say "Scoring the full embedding (with the pretrained baseline)"
compare all 1 --bundles "${BUNDLES_ALL[@]}" --floors "${FLOORS_ALL[@]}"

say "Done"
echo "  tables   $OUT/{content,all}/compare.txt"
echo "  figures  $OUT/{content,all}/figures/"
echo "  csv      $OUT/{content,all}/compare.csv  (+ compare_graph.csv)"
