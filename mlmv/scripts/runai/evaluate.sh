#!/usr/bin/env bash
# Score a saved checkpoint with the leakage evaluation on Run:ai (mlmv.eval.evaluate_run).
#
#   bash mlmv/scripts/runai/evaluate.sh results/mlmv/p1_exact                       # final.pt
#   bash mlmv/scripts/runai/evaluate.sh results/mlmv/p1_exact_batchstd step_050000.pt
#   bash mlmv/scripts/runai/evaluate.sh --dry-run results/mlmv/p1_exact             # print, do not submit
#
# Arguments: the run directory (relative to the repo root) and optionally a checkpoint in its checkpoints/
# folder. Results go to <run-dir>/eval/<checkpoint name>/. MLMV_REPO overrides the checkout, as in phase1.sh.
set -euo pipefail

DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; shift; fi
if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then echo "Usage: $0 [--dry-run] <run-dir> [checkpoint]" >&2; exit 2; fi
RUN_DIR=${1%/}
CKPT=${2:-final.pt}
REPO=${MLMV_REPO:-/nfs/home/nglazman/crl-2/multiview-crl}
# Run:ai job names: lowercase letters, digits and '-', at most 40 characters. The checkpoint part is kept whole
# (step_050000 -> s050000) and the run name is shortened to fit, so two checkpoints never share a job name.
RUN_TAG=$(basename "$RUN_DIR" | tr '[:upper:]_.' '[:lower:]--')
CKPT_TAG=$(printf '%s' "${CKPT%.pt}" | sed 's/^step_/s/' | tr '[:upper:]_.' '[:lower:]--')
MAX=$((29 - ${#CKPT_TAG}))
JOB="mlmv-eval-${RUN_TAG:0:$MAX}-${CKPT_TAG}"

TRAIN_CMD="cd ${REPO} || { echo ERROR: ${REPO} is missing inside the container - check the --host-path mount of /nfs >&2 ; exit 1 ; } ; \
test -d ${RUN_DIR} || { echo ERROR: run directory ${RUN_DIR} not found in ${REPO} >&2 ; exit 1 ; } ; \
export PYTHONPATH=${REPO} PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ; \
python -c 'import torch; assert torch.cuda.is_available(), \"Run:ai job has no usable CUDA device\"' ; \
python -m mlmv.eval.evaluate_run --run-dir ${RUN_DIR} --checkpoint ${CKPT}"

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
