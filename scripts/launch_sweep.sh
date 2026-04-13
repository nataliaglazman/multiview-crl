#!/bin/bash
# Launch a W&B sweep and submit agents as Run:AI jobs.
#
# Usage:
#   ./scripts/launch_sweep.sh [--num-agents 50] [--wandb-project multiview-crl-sweep]

set -euo pipefail

NUM_AGENTS=10
WANDB_PROJECT="multiview-crl-sweep"
SWEEP_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-agents) NUM_AGENTS="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --sweep-id) SWEEP_ID="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${SWEEP_ID}" ]]; then
    echo "Creating W&B sweep..."
    RAW_OUTPUT=$(wandb sweep --project "${WANDB_PROJECT}" scripts/sweep_config.yaml 2>&1 || true)
    SWEEP_ID=$(echo "$RAW_OUTPUT" | awk '/Run sweep agent with:/ {print $NF}')

    if [[ -z "${SWEEP_ID}" ]]; then
        echo "ERROR: Failed to parse sweep ID."
        echo "--- RAW WANDB OUTPUT ---"
        echo "$RAW_OUTPUT"
        echo "------------------------"
        echo "Run 'wandb sweep scripts/sweep_config.yaml' manually to debug."
        exit 1
    fi
fi

echo "Sweep ID: ${SWEEP_ID}"
echo "Submitting ${NUM_AGENTS} Run:AI jobs..."

for i in $(seq 0 $((NUM_AGENTS - 1))); do
    ./scripts/sweep_runai.sh "${SWEEP_ID}" "${i}"
done

echo ""
echo "Done. ${NUM_AGENTS} agents submitted."
echo "Monitor at: https://wandb.ai/${SWEEP_ID}"
