#!/bin/bash -l
# Sweep identifiability analysis across trained MultiviewVAE runs.
# Run AFTER the training jobs (scripts/run_vae_synthetic_slurm.sh) have finished.
#
# Submit from the repo root:
#     sbatch scripts/run_vae_analysis_slurm.sh
#
# Override what to analyse / how:
#     RESULTS_GLOB='results/vae_synth_c*' OUT=results/vae_sweep.csv sbatch scripts/run_vae_analysis_slurm.sh
#     NULL_PERMUTE=1 sbatch scripts/run_vae_analysis_slurm.sh        # add DCI null floors (slower)
#
# Output: a printed table in the .out file + a CSV at $OUT, plus a per-run
# results/<run>/identifiability_analysis.json from each run_analysis call.

#SBATCH --job-name=vae-analysis
#SBATCH --output=vae-analysis-%j.out
#SBATCH --error=vae-analysis-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

# ---- Analysis knobs (override via environment at submit time) ----
RESULTS_GLOB=${RESULTS_GLOB:-results/vae_synth_c*}
OUT=${OUT:-results/vae_identifiability_sweep.csv}
NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES:-400}
BATCH_SIZE=${BATCH_SIZE:-32}

# Set NULL_PERMUTE=1 to add row-permuted DCI null floors (slower).
NULL_FLAG=""
if [ "${NULL_PERMUTE:-0}" = "1" ]; then
    NULL_FLAG="--null-permute"
fi

# ---- Software & environment ----
module load anaconda3/2022.10-gcc-13.2.0

CONDA_ENV_NAME="multiview-env"
PYTHON="${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python"
export PYTHONNOUSERSITE=1

if ! "$PYTHON" -c "import torch, numpy, sklearn" 2>/dev/null; then
    echo "Environment '${CONDA_ENV_NAME}' missing or broken -- rebuild it on a login node first." >&2
    exit 1
fi

# ---- Working directory ----
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

# ---- Analyse ----
"$PYTHON" -m eval.sweep_vae_identifiability \
    --results-glob "${RESULTS_GLOB}" \
    --num-test-samples "${NUM_TEST_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers 8 \
    --out "${OUT}" \
    ${NULL_FLAG}
