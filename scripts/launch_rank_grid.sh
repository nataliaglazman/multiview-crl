#!/bin/bash
# Submit the rank-collapse grid: 3 x 2 over the two coefficients the measured force budget
# says carry it. Everything else is held fixed by experiments/synthetic_causal_clean_content.yaml,
# so the gon=1.0 / pw=1.0 cell is the control.
#
#   bt_gap_on_coeff   attacks btgap_on, the largest single rank-destroying force (-1.381)
#   bt_patch_weight   attacks the whole patch block (-1.010, with nothing positive in it)
#
# 12k steps is enough: the collapse was already complete by 18k on the reference run, and
# selection/* is logged every 2000 steps by --dci-every.
#
# Usage:
#   ./scripts/launch_rank_grid.sh                       # runai, 12k steps
#   ./scripts/launch_rank_grid.sh --cluster slurm
#   ./scripts/launch_rank_grid.sh --steps 20000
#   ./scripts/launch_rank_grid.sh --dry-run             # print the resolved commands only
#
# Then:
#   python -m eval.grid_report results/synthetic/rank-grid-*

set -euo pipefail

CLUSTER="runai"
STEPS=12000
EXPERIMENT="experiments/synthetic_causal_clean_content.yaml"
GON_VALUES="1.0 0.5 0.25"
PW_VALUES="1.0 0.5"
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster) CLUSTER="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        --gon) GON_VALUES="$2"; shift 2 ;;
        --pw) PW_VALUES="$2"; shift 2 ;;
        *) PASSTHROUGH+=("$1"); shift ;;
    esac
done

cd "$(dirname "$0")/.."

for gon in ${GON_VALUES}; do
    for pw in ${PW_VALUES}; do
        # RunAI workload names accept lowercase alphanumeric and hyphen ONLY, so the decimal
        # point becomes 'p': 0.25 -> 0p25. Checked rather than assumed, because a bad name
        # fails at submit time with a message that does not name the offending character.
        tag="rank-grid-gon${gon//./p}-pw${pw//./p}"
        if [[ ! "${tag}" =~ ^[a-z0-9-]+$ ]]; then
            echo "Invalid workload name '${tag}': RunAI allows lowercase alphanumeric and hyphen only." >&2
            exit 1
        fi
        echo "=== ${tag} (bt_gap_on_coeff=${gon}, bt_patch_weight=${pw}, ${STEPS} steps)"
        python scripts/launch.py "${EXPERIMENT}" \
            --cluster "${CLUSTER}" \
            --set "bt_gap_on_coeff=${gon}" "bt_patch_weight=${pw}" \
                  "train_steps=${STEPS}" "tag=${tag}" \
            ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
    done
done
