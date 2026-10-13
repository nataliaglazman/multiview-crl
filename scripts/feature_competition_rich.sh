#!/usr/bin/env bash
# Rich-competitor test of the feature-competition experiment (experiments/FEATURE_COMPETITION_PROTOCOL.md,
# "Rich competitor"): does lesion learning drop once several shared anatomy factors can tell subjects apart?
#
# Arms, all with batch-standardized InfoNCE on one bank with identical lesions:
#   A   anatomy fixed (no competitor)
#   B   brain size shared by T1 and FLAIR
#   R   brain size, cortical thickness and temporal atrophy, all shared
#   R'  the same three factors drawn separately for each view (same image variation, useless for matching)
# Three is the most this lesion geometry allows: with more factors no lesion position stays in white matter for
# every anatomy. Every evaluation also scores held-out matching with and without the lesion (lesion reliance).
#
# CUDA PC, one seed:    DEVICE=cuda PYTHON=python bash scripts/feature_competition_rich.sh
# three seeds:          DEVICE=cuda PYTHON=python SEEDS="42 142 242" bash scripts/feature_competition_rich.sh
# Rerunning skips finished runs and resumes unfinished ones.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BANK="${BANK:-$REPO/results/feature_competition/rich}"
export BANK_ARMS="${BANK_ARMS:-A B R Rp}"
export ARMS="${ARMS:-R Rp B A}"
export COMPETITORS="${COMPETITORS:-brain_size cortical_thickness temporal_atrophy}"
exec bash "$REPO/scripts/feature_competition_comparison.sh" "$@"
