#!/usr/bin/env bash
# Phase 1 (synthetic three-level benchmark): submit one training run to Run:ai.
#
#   bash mlmv/scripts/runai/phase1.sh mlmv/config/runs/phase1/p1_exact.yaml
#   bash mlmv/scripts/runai/phase1.sh --dry-run mlmv/config/runs/phase1/p1_exact.yaml   # print, do not submit
#   for c in mlmv/config/runs/phase1/*.yaml; do bash mlmv/scripts/runai/phase1.sh "$c"; done   # the whole phase
#
# The only argument is the config path (relative to the repo root). The run writes TensorBoard logs, the
# scalars log, checkpoints and the leakage matrix under results/mlmv/<run.name>/ in the cluster checkout.
# Image, project, resources and mounts copy experiments/generated/encoder/*.runai.sh.
set -euo pipefail

DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; shift; fi
if [[ "$#" -ne 1 ]]; then echo "Usage: $0 [--dry-run] <config.yaml>" >&2; exit 2; fi
CONFIG=$1
REPO=/nfs/home/nglazman/crl-2/multiview-crl
# Run:ai job names: lowercase letters, digits and '-', at most 40 characters.
JOB=$(basename "$CONFIG" .yaml | tr '[:upper:]_.' '[:lower:]--' | cut -c1-35)
JOB="mlmv-${JOB}"

TRAIN_CMD="cd ${REPO} || { echo ERROR: ${REPO} is missing inside the container - check the --host-path mount of /nfs >&2 ; exit 1 ; } ; \
test -f ${CONFIG} || { echo ERROR: config ${CONFIG} not found in ${REPO} >&2 ; exit 1 ; } ; \
export PYTHONPATH=${REPO} PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ; \
python -c 'import torch; assert torch.cuda.is_available(), \"Run:ai job has no usable CUDA device\"' ; \
python -m mlmv.train.train_levels --config ${CONFIG}"

if [[ $DRY == 1 ]]; then printf 'job: %s\n%s\n' "$JOB" "$TRAIN_CMD"; exit 0; fi

runai training standard submit "$JOB" \
    --project nglazman \
    --image aicregistry:5000/nglazman:multiview-crl \
    --run-as-user \
    --large-shm \
    --node-type A100 \
    --gpu-devices-request 1 \
    --cpu-core-request 16 \
    --cpu-core-limit 32 \
    --cpu-memory-request 64G \
    --cpu-memory-limit 128G \
    --host-path path=/nfs,mount=/nfs,readwrite \
    --command -- bash -c "${TRAIN_CMD}"
