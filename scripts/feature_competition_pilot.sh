#!/usr/bin/env bash
# Arm-A pilot of the feature-competition experiment on a Mac (experiments/FEATURE_COMPETITION_PROTOCOL.md,
# "Arm-A pilot").
#
# Arm A holds brain size fixed, so the lesion is the only thing a subject's T1 and FLAIR share. The pilot asks:
#   1. Can the SSL encoder's own architecture (its GAP code) carry lesion location when trained with labels?
#   2. Does a spatial heatmap localizer find it, i.e. does the image hold it?
#   3. Does the self-supervised encoder learn it with no competitor? Its held-out InfoNCE is a lesion meter:
#      any drop below chance (2 ln 32) can only come from lesion information.
# If 3 fails, an A/B/B' comparison cannot show suppression: there is nothing to suppress.
#
# About 5.5 h on an M4 Pro (measured: SSL 1.33 s/step, GAP control 0.17, heatmap 0.34); keep the Mac plugged
# in. Every step skips or resumes when its output exists, so rerunning continues where it stopped (Ctrl-C is
# safe). If `prepare` itself was interrupted, delete the half-written bank directory first.
#
#   bash scripts/feature_competition_pilot.sh
#   BANK=/elsewhere/pilot_A SSL_STEPS=5000 bash scripts/feature_competition_pilot.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/opt/miniconda3/envs/adni-analysis/bin/python}"
BANK="${BANK:-$REPO/results/feature_competition/pilot_A}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"
SSL_STEPS="${SSL_STEPS:-10000}"
SUPERVISED_STEPS="${SUPERVISED_STEPS:-5000}"

# Keep the Mac awake for the whole pilot.
if [[ "$(uname)" == "Darwin" && -z "${PILOT_CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
  export PILOT_CAFFEINATED=1
  exec caffeinate -i "$BASH" "${BASH_SOURCE[0]}" "$@"
fi

mkdir -p "$(dirname "$BANK")"
exec > >(tee -a "$BANK.log") 2>&1
echo "== feature-competition arm-A pilot: $(date), bank $BANK, device $DEVICE"

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
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"
fc() { "$PYTHON" -m training.feature_competition "$@" --bank "$BANK"; }
# True when every listed run of this kind has reached the requested step. A finished run is skipped rather
# than resumed, because resuming re-checks the code hashes it recorded, and later code edits fail that check.
complete() {
  local kind=$1 steps=$2 view
  for view in "${@:3}"; do
    "$PYTHON" -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["step"] < int(sys.argv[2]))' \
      "$BANK/runs/${kind}_A_s${SEED}_${view}/progress.json" "$steps" 2>/dev/null || return 1
  done
}

[[ -f "$BANK/manifest.json" ]] || fc prepare --arms A
[[ -f "$BANK/audit/report.json" ]] || fc audit

run=(--arms A --seeds "$SEED" --device "$DEVICE")
# 1. Same architecture with labels (~30 min): can the GAP code carry location at all? If not, the SSL arm
#    below cannot learn it either, and the readout needs changing before any A/B/B' comparison.
complete supervised_gap "$SUPERVISED_STEPS" t1 flair ||
  fc train --kind supervised_gap "${run[@]}" --steps "$SUPERVISED_STEPS" --save-every 1000 --save-steps 250 500 --resume
fc evaluate --kind supervised_gap "${run[@]}" --checkpoint all --batch-size 32
fc curves --arms A
# 2. Self-supervised, no competitor (~4 h).
complete ssl "$SSL_STEPS" paired ||
  fc train --kind ssl "${run[@]}" --steps "$SSL_STEPS" --save-every 1000 --save-steps 250 500 1500 --resume
fc evaluate --kind ssl "${run[@]}" --checkpoint all --batch-size 32
fc curves --arms A
# 3. Heatmap localizer with labels (~1 h; its pooling runs on the CPU under MPS): does the image hold location?
complete supervised "$SUPERVISED_STEPS" t1 flair ||
  fc train --kind supervised "${run[@]}" --steps "$SUPERVISED_STEPS" --save-every 1000 --save-steps 250 500 --resume
fc evaluate --kind supervised "${run[@]}" --checkpoint all --batch-size 32
fc curves --arms A
echo "== done: $(date). Figure: $BANK/summary/curves_A.png; tables above and in $BANK.log"
