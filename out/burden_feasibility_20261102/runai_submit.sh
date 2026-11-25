#!/usr/bin/env bash
# Submit run_pc.sh stages (default: 68) as one Run:ai job, from the authenticated Run:ai host. The job runs
# runai_job.sh in the training image on the cluster checkout (REPO_PATH), so pull this commit there first.
#   bash out/burden_feasibility_20261102/runai_submit.sh --dry-run [STAGE...]
#   bash out/burden_feasibility_20261102/runai_submit.sh [STAGE...]
# Set any of ADNI_DATA, ADNI_LABELS, ADNI_MASKS, ADNI_SPLIT, WHEELS and PYTHONUSERBASE (stage 68) or WIDTH_SEEDS,
# WIDTH_UNITS, WIDTH_PARALLEL and WIDTH_WAIT_BANK (stage 74) to pass them to the job.
# Resources: RUNAI_PROJECT, RUNAI_IMAGE, RUNAI_NODE_TYPE, RUNAI_CPU, RUNAI_CPU_LIMIT, RUNAI_MEMORY, RUNAI_MEMORY_LIMIT.
# Progress: out/burden_feasibility_20261102/adni/runai.log in the checkout, then the stage's own logs.
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
STAGES=("${@:-68}")
for stage in "${STAGES[@]}"; do
    if [[ ! "$stage" =~ ^[0-9]+$ ]]; then
        echo "Stages are run_pc.sh stage numbers, got '$stage'." >&2
        exit 2
    fi
done

REPO_PATH=${REPO_PATH:-/nfs/home/nglazman/crl-2/multiview-crl}
JOB_NAME=${JOB_NAME:-burden-s$(IFS=-; echo "${STAGES[*]}")-$(date +%m%d-%H%M)}
if [[ ! "$JOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ || ${#JOB_NAME} -gt 63 ]]; then
    echo "Run:ai JOB_NAME must contain at most 63 lowercase letters, digits and hyphens, starting/ending alphanumeric." >&2
    exit 2
fi

PASS=""
for name in ADNI_DATA ADNI_LABELS ADNI_MASKS ADNI_SPLIT WHEELS PYTHONUSERBASE WIDTH_SEEDS WIDTH_UNITS WIDTH_PARALLEL WIDTH_WAIT_BANK; do
    if [[ -n "${!name:-}" ]]; then
        PASS+="$name=$(printf '%q' "${!name}") "
    fi
done
# One line with no double quote: Run:ai cuts a bash -c payload at an inner double quote.
JOB_CMD="cd $(printf '%q' "$REPO_PATH") && ${PASS}bash out/burden_feasibility_20261102/runai_job.sh ${STAGES[*]}"
if [[ "$JOB_CMD" == *'"'* || "$JOB_CMD" == *$'\n'* ]]; then
    echo "The job command must be one line with no double quote; check the paths passed in." >&2
    exit 2
fi

RUNAI_CMD=(
    runai training standard submit "$JOB_NAME"
    --project "${RUNAI_PROJECT:-nglazman}"
    --image "${RUNAI_IMAGE:-aicregistry:5000/nglazman:multiview-crl}"
    --run-as-user
    --large-shm
    --node-type "${RUNAI_NODE_TYPE:-A100}"
    --gpu-devices-request 1
    --cpu-core-request "${RUNAI_CPU:-16}"
    --cpu-core-limit "${RUNAI_CPU_LIMIT:-32}"
    --cpu-memory-request "${RUNAI_MEMORY:-64G}"
    --cpu-memory-limit "${RUNAI_MEMORY_LIMIT:-128G}"
    --host-path path=/nfs,mount=/nfs,readwrite
    --command -- bash -c "$JOB_CMD"
)

if [[ "$DRY_RUN" == "1" ]]; then
    printf '# container command\n%s\n# submit command\n' "$JOB_CMD"
    printf '%q ' "${RUNAI_CMD[@]}"
    printf '\n'
    exit 0
fi
printf 'Job: %s\n' "$JOB_NAME"
exec "${RUNAI_CMD[@]}"
