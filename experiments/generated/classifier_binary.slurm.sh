#!/bin/bash -l
# SLURM equivalent of: runai submit --name classifier-binary-8
# Runs DCI evaluation on a synthetic causal run checkpoint.
#SBATCH --job-name=classifier-binary
#SBATCH --output=/scratch/users/%u/classifier-binary-%j.out
#SBATCH --error=classifier-binary-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# -- Software & Environment Setup --
module load anaconda3/2022.10-gcc-13.2.0

CONDA_ENV_NAME="multiview-env"
PYTHON="${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python"

export PYTHONNOUSERSITE=1

if ! "$PYTHON" -c "import torch; import numpy" 2>/dev/null; then
    echo "Environment '${CONDA_ENV_NAME}' missing or broken -- rebuilding cleanly..."
    conda env remove -n "${CONDA_ENV_NAME}" --yes 2>/dev/null || true
    conda create -n "${CONDA_ENV_NAME}" python=3.10 -y

    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    "$PYTHON" -m pip install numpy
    "$PYTHON" -m pip install scikit-learn
    "$PYTHON" -m pip install tensorboard pandas matplotlib
fi

if [ -f "${SLURM_SUBMIT_DIR}/docker/requirements.txt" ]; then
    "$PYTHON" -m pip install -r "${SLURM_SUBMIT_DIR}/docker/requirements.txt" || echo "Requirements sync skipped a broken package."
fi
echo "Environment setup complete."

# -- Working directory --
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

# -- Paths (edit these for your run) --
RUN_DIR="/scratch/users/k24058220/multiview-crl/results/synthetic/synthetic-create-causal-random"
CHECKPOINT="${RUN_DIR}/vqvae_best.pt"

# -- Evaluation --
"$PYTHON" -m eval.run_dci_synthetic \
    --run-dir "${RUN_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --pooling 4,4,4 \
    --levels 0 \
    --num-samples 400
