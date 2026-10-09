#!/usr/bin/env bash
# Frozen ADNI checkpoint diagnostic. Submit from the authenticated Run:ai host.
# Sync this checkout to REPO_PATH first. Extra arguments go directly to the probe.
#   bash scripts/run_encoder_adni_modality_gap_runai.sh --dry-run
#   bash scripts/run_encoder_adni_modality_gap_runai.sh --checkpoint model_init.pt
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

REPO_PATH=${REPO_PATH:-/nfs/home/nglazman/crl-2/multiview-crl}
MODEL_ID=${MODEL_ID:-encoder_adni_conv_mlp_layernorm_s42}
RUN_DIR=${RUN_DIR:-$REPO_PATH/results/encoder_adni/runs/$MODEL_ID}
CHECKPOINT=${CHECKPOINT:-model_best.pt}
BATCH_SIZE=${BATCH_SIZE:-4}
JOB_NAME=${JOB_NAME:-adni-modality-gap-$(date +%s)}
if [[ ! "$JOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ || ${#JOB_NAME} -gt 63 ]]; then
    echo "Run:ai JOB_NAME must contain at most 63 lowercase letters, digits and hyphens, starting/ending alphanumeric." >&2
    exit 2
fi

PROBE_ARGS=(--run-dir "$RUN_DIR" --checkpoint "$CHECKPOINT" --device cuda --batch-size "$BATCH_SIZE" "$@")
REPO_Q=$(printf '%q' "$REPO_PATH")
ARGS_Q=$(printf '%q ' "${PROBE_ARGS[@]}")
# Keep the entire bash -c payload on one line, with no continuation backslashes
# whose meaning could change if a submission client folds embedded newlines.
PROBE_CMD="set -euo pipefail ; cd $REPO_Q ; export PYTHONPATH=$REPO_Q PYTHONUNBUFFERED=1"
PROBE_CMD+=" CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
PROBE_CMD+=" NUMEXPR_NUM_THREADS=1 ; exec python -m eval.adni.encoder_modality_gap ${ARGS_Q% }"

RUNAI_CMD=(
    runai training standard submit "$JOB_NAME"
    --project "${RUNAI_PROJECT:-nglazman}"
    --image "${RUNAI_IMAGE:-aicregistry:5000/nglazman:multiview-crl}"
    --run-as-user
    --large-shm
    --node-type "${RUNAI_NODE_TYPE:-A100}"
    --gpu-devices-request 1
    --cpu-core-request "${RUNAI_CPU:-4}"
    --cpu-core-limit "${RUNAI_CPU_LIMIT:-8}"
    --cpu-memory-request "${RUNAI_MEMORY:-32G}"
    --cpu-memory-limit "${RUNAI_MEMORY_LIMIT:-64G}"
    --host-path path=/nfs,mount=/nfs,readwrite
    --command -- bash -c "$PROBE_CMD"
)

if [[ "$DRY_RUN" == "1" ]]; then
    printf '# container command\n%s\n# submit command\n' "$PROBE_CMD"
    printf '%q ' "${RUNAI_CMD[@]}"
    printf '\n'
    exit 0
fi
printf 'Job: %s\n' "$JOB_NAME"
exec "${RUNAI_CMD[@]}"
