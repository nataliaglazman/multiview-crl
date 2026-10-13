#!/usr/bin/env bash
# A/B/B' comparison of the feature-competition experiment with batch-standardized InfoNCE
# (experiments/FEATURE_COMPETITION_PROTOCOL.md, section 1).
#
# Arm A holds brain size fixed, B shares one brain-size draw between T1 and FLAIR, and B' draws it separately
# for each view. Subjects, lesions, seeds and batch order are the same in every arm. In the arm-A pilot, plain
# InfoNCE never left chance (near-identical codes, a cold start), while --standardize-code learned the lesion
# from about step 3,000. So every arm here uses --standardize-code. Each arm is scored as soon as it finishes;
# summary/curves_arms_batch-standardized.png overlays the arms.
#
# Mac, one seed (about 12 h on an M4 Pro):   bash scripts/feature_competition_comparison.sh
# CUDA PC, three seeds:                       DEVICE=cuda PYTHON=python SEEDS="42 142 242" bash scripts/feature_competition_comparison.sh
# SUPERVISED=1 also trains the same-architecture supervised GAP control for every arm and view.
# Rerunning skips finished runs and resumes unfinished ones. Resuming needs the hashed source files unchanged.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/opt/miniconda3/envs/adni-analysis/bin/python}"
BANK="${BANK:-$REPO/results/feature_competition/abbp}"
DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-42}"
ARMS="${ARMS:-B Bp A}"
STEPS="${STEPS:-10000}"
SUPERVISED="${SUPERVISED:-0}"

# Keep a Mac awake for the whole comparison.
if [[ "$(uname)" == "Darwin" && -z "${COMPARISON_CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
  export COMPARISON_CAFFEINATED=1
  exec caffeinate -i "$BASH" "${BASH_SOURCE[0]}" "$@"
fi

mkdir -p "$(dirname "$BANK")"
exec > >(tee -a "$BANK.log") 2>&1
echo "== feature-competition A/B/B' comparison: $(date), bank $BANK, arms $ARMS, seeds $SEEDS, device $DEVICE"

# training.losses imports LPIPS at module scope; nothing in this experiment builds it.
if ! "$PYTHON" -c "import lpips" 2>/dev/null; then
  STAND_IN="${TMPDIR:-/tmp}/multiview-crl-lpips-stand-in"
  mkdir -p "$STAND_IN/lpips"
  printf '%s\n' "class LPIPS:" "    def __init__(self, *args, **kwargs):" \
    "        raise ImportError('lpips is not installed; this experiment never builds LPIPS')" \
    >"$STAND_IN/lpips/__init__.py"
  export PYTHONPATH="$STAND_IN:$REPO${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
fi
[[ "$DEVICE" == mps ]] && export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"
fc() { "$PYTHON" -m training.feature_competition "$@" --bank "$BANK"; }
# True when this run folder has reached the requested step. A finished run is skipped rather than resumed,
# because resuming re-checks the code hashes it recorded, and later code edits fail that check.
complete() {
  "$PYTHON" -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["step"] < int(sys.argv[2]))' \
    "$BANK/runs/$1/progress.json" "$2" 2>/dev/null
}

[[ -f "$BANK/manifest.json" ]] || fc prepare --arms A B Bp
[[ -f "$BANK/audit/report.json" ]] || fc audit

for seed in $SEEDS; do
  for arm in $ARMS; do
    ssl=(--kind ssl --standardize-code --arms "$arm" --seeds "$seed" --device "$DEVICE")
    # Dense early checkpoints: in the pilot the lesion appeared between steps 3,000 and 6,000.
    complete "ssl-std_${arm}_s${seed}_paired" "$STEPS" ||
      fc train "${ssl[@]}" --steps "$STEPS" --save-every 1000 --save-steps 250 500 1500 2500 3500 4500 --resume
    fc evaluate "${ssl[@]}" --checkpoint all --batch-size 32
    fc curves
  done
done

if [[ "$SUPERVISED" == 1 ]]; then
  for seed in $SEEDS; do
    for arm in $ARMS; do
      gap=(--kind supervised_gap --arms "$arm" --seeds "$seed" --device "$DEVICE")
      complete "supervised_gap_${arm}_s${seed}_t1" "$STEPS" && complete "supervised_gap_${arm}_s${seed}_flair" "$STEPS" ||
        fc train "${gap[@]}" --steps "$STEPS" --save-every 1000 --save-steps 500 --resume
      fc evaluate "${gap[@]}" --checkpoint all --batch-size 32
    done
  done
fi

fc summarize
fc curves
echo "== done: $(date). Figures: $BANK/summary/curves_arms_batch-standardized.png and recovery.png; numbers in report.json"
