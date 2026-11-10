#!/usr/bin/env bash
# Submit one gl_ident training run to Run:ai. The submit flags, image and paths copy
# experiments/generated/encoder/*.runai.sh; only the training command differs. Not used for E1 (run locally).
#   bash experiments/gl_ident/runai_submit.sh <job-name> --config experiments/gl_ident/configs/e1.yaml --set model.kind=local train.seed=0
#   bash experiments/gl_ident/runai_submit.sh --dry-run <job-name> ...   # print the command only
set -euo pipefail

DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; shift; fi
if [[ "$#" -lt 2 ]]; then echo "Usage: $0 [--dry-run] <job-name> <train args...>" >&2; exit 2; fi
JOB=$1; shift
ARGS=$(printf ' %q' "$@")

TRAIN_CMD="cd /nfs/home/nglazman/crl-2/multiview-crl || { echo ERROR: /nfs/home/nglazman/crl-2/multiview-crl is missing inside the container - check the --host-path mount of /nfs >&2 ; exit 1 ; } ; \
export PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ; \
python -c 'import torch; assert torch.cuda.is_available(), \"Run:ai job has no usable CUDA device\"' ; \
python -W ignore -m experiments.gl_ident.train${ARGS}"

if [[ $DRY == 1 ]]; then printf '%s\n' "$TRAIN_CMD"; exit 0; fi

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
