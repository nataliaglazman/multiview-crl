#!/bin/bash
# Submit a single W&B sweep agent as a Run:AI job.
#
# Usage:
#   ./scripts/sweep_runai.sh <SWEEP_ID> <AGENT_INDEX>
#
# Example:
#   ./scripts/sweep_runai.sh natalia/multiview-crl-sweep/abc123 0
#
# Configure the variables below to match your Run:AI setup.

set -euo pipefail

SWEEP_ID="${1:?Error: pass the W&B sweep ID as first argument}"
AGENT_IDX="${2:?Error: pass the agent index as second argument}"

# ---- Configure these for your setup ----
IMAGE="aicregistry:5000/nglazman:multiview-crl-vqvae-final"          # e.g. registry.example.com/multiview-crl:latest
PROJECT="nglazman"       # e.g. natalia
GPU=1
CPU=16
MEMORY="64Gi"
PVC_NAME="/nfs:/nfs"          # PVC with data + code
PVC_MOUNT="/home/nglazman/crl-2"               # Mount point inside container
WANDB_API_KEY="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"
# -----------------------------------------

JOB_NAME="sweep-${AGENT_IDX}"

runai submit "${JOB_NAME}" \
    --project "${PROJECT}" \
    --image "${IMAGE}" \
    --run-as-user \
    --large-shm \
    --node-type A100 \
    --gpu "${GPU}" \
    --cpu "${CPU}" \
    --memory "${MEMORY}" \
    --memory-limit 128G \
    --volume "/nfs:${PVC_MOUNT}" \
    --environment "WANDB_API_KEY=${WANDB_API_KEY}" \
    --command -- bash -c "cd ${PVC_MOUNT}/multiview-crl && wandb agent --count 1 ${SWEEP_ID}"

echo "Submitted Run:AI job: ${JOB_NAME}"
