#!/usr/bin/env bash
# Encoder-only multi-view contrastive learning on real ADNI T1/FLAIR pairs
# (training/main_conv_synthetic.py with --dataset-name) as one Run:ai job. The recipe
# and its knobs live in scripts/encoder_adni_recipe.sh, shared with
# scripts/run_encoder_adni_slurm.sh; paths and resources come from
# experiments/cluster/runai.yaml. Flags and their trade-offs: training/ENCODER_ADNI.md.
#
# The job runs the checkout at $REPO_PATH on /nfs, so sync the code there first.
# Preview, then submit from a machine with the authenticated Run:ai CLI:
#     bash scripts/run_encoder_adni_runai.sh --dry-run
#     bash scripts/run_encoder_adni_runai.sh
#
# Variants (override via environment), e.g.:
#     MODEL_ID=encoder_adni_patch PATCH_WEIGHT=1 bash scripts/run_encoder_adni_runai.sh
# The job name is MODEL_ID with '_' -> '-'; a name already in use is rejected by Run:ai.
#
# Output: $OUT_DIR/$MODEL_ID/ (settings.json, split.json, separation_step*.json,
# model.pt, model_best.pt by held-out loss, TensorBoard).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Paths (experiments/cluster/runai.yaml) ----
REPO_PATH=${REPO_PATH:-/nfs/home/nglazman/crl-2/multiview-crl}
DATAROOT=${DATAROOT:-/nfs/home/nglazman/data}
LABELS_PATH=${LABELS_PATH:-/nfs/home/nglazman/nmpevqvae/labels_cleaned_3class_demog.csv}
MASKS_DIR=${MASKS_DIR:-/nfs/home/nglazman/data/ADNI_stripped_masks}
# Same spacing, size and masks as experiments/adni_real.yaml, so its preprocessed cache is reused.
CACHE_DIR=${CACHE_DIR:-/nfs/home/nglazman/cache/multiview}
OUT_DIR=${OUT_DIR:-$REPO_PATH/results/encoder_adni/runs}

# ---- Resources (experiments/cluster/runai.yaml) ----
RUNAI_PROJECT=${RUNAI_PROJECT:-nglazman}
RUNAI_IMAGE=${RUNAI_IMAGE:-aicregistry:5000/nglazman:multiview-crl}
RUNAI_NODE_TYPE=${RUNAI_NODE_TYPE:-A100}
RUNAI_CPU=${RUNAI_CPU:-16}
RUNAI_CPU_LIMIT=${RUNAI_CPU_LIMIT:-32}
RUNAI_MEMORY=${RUNAI_MEMORY:-64G}
RUNAI_MEMORY_LIMIT=${RUNAI_MEMORY_LIMIT:-128G}

source "$HERE/encoder_adni_recipe.sh"

JOB_NAME=${JOB_NAME:-$(printf '%s' "${MODEL_ID//_/-}" | tr 'A-Z' 'a-z')}
if [[ ! "$JOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
    echo "Run:ai job name '$JOB_NAME' must be lowercase letters, digits and hyphens; set JOB_NAME." >&2
    exit 2
fi

# One line, as the generated Run:ai scripts fold theirs. One thread per process: the main
# process feeds the GPU, the workers augment on the CPU.
REPO_Q=$(printf '%q' "$REPO_PATH")
TRAIN_CMD="set -euo pipefail ; cd $REPO_Q ; export PYTHONPATH=$REPO_Q PYTHONUNBUFFERED=1"
TRAIN_CMD+=" CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
TRAIN_CMD+=" NUMEXPR_NUM_THREADS=1 ; python -c 'import torch; assert torch.cuda.is_available()' ;"
ARGS_Q=$(printf '%q ' "${TRAIN_ARGS[@]}")
TRAIN_CMD+=" python -m training.main_conv_synthetic ${ARGS_Q% }"

# Flags in launch.py's order. --large-shm: DataLoader workers hand batches over in /dev/shm.
# readwrite: without it the v2 CLI mounts /nfs read-only and the run directory cannot be made.
RUNAI_CMD=(
    runai training standard submit "$JOB_NAME"
    --project "$RUNAI_PROJECT"
    --image "$RUNAI_IMAGE"
    --run-as-user
    --large-shm
    --node-type "$RUNAI_NODE_TYPE"
    --gpu-devices-request 1
    --cpu-core-request "$RUNAI_CPU"
    --cpu-core-limit "$RUNAI_CPU_LIMIT"
    --cpu-memory-request "$RUNAI_MEMORY"
    --cpu-memory-limit "$RUNAI_MEMORY_LIMIT"
    --host-path path=/nfs,mount=/nfs,readwrite
    --command -- bash -c "$TRAIN_CMD"
)

if [[ "$#" -gt 1 ]]; then echo "Usage: $0 [--dry-run]" >&2; exit 2; fi
case "${1:-}" in
    --dry-run)
        printf '# container command\n%s\n# submit command\n' "$TRAIN_CMD"
        printf '%q ' "${RUNAI_CMD[@]}"
        printf '\n'
        exit 0 ;;
    "") ;;
    *) echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;
esac
exec "${RUNAI_CMD[@]}"
