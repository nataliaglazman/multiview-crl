#!/bin/bash -l
# Train the simple multiview VAE (models/vae.py) on synthetic data and measure
# identifiability via eval.dci.compute_dci_synthetic.
#
# Submit from the repo root:
#     sbatch scripts/run_vae_synthetic_slurm.sh
#
# Override knobs at submit time, e.g. to sweep the latent dimensionality:
#     CONTENT_CHANNELS=6  MODEL_ID=vae_synth_c6  sbatch scripts/run_vae_synthetic_slurm.sh
#     CONTENT_CHANNELS=9  MODEL_ID=vae_synth_c9  sbatch scripts/run_vae_synthetic_slurm.sh
#     CONTENT_CHANNELS=12 MODEL_ID=vae_synth_c12 sbatch scripts/run_vae_synthetic_slurm.sh
#
# No W&B / no internet needed: logs go to results/<MODEL_ID>/ (TensorBoard + DCI JSON).

#SBATCH --job-name=vae-synthetic
#SBATCH --output=vae-synthetic-%j.out
#SBATCH --error=vae-synthetic-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

# ---- Experiment knobs (override via environment at submit time) ----
MODEL_ID=${MODEL_ID:-ae_normal}
CONTENT_CHANNELS=${CONTENT_CHANNELS:-9}   # the "number of dimensions" under test
LATENT_CHANNELS=${LATENT_CHANNELS:-16}    # total latent = content + style
N_CONTENT=${N_CONTENT:-9}                 # true shared content factors
N_STYLE=${N_STYLE:-3}                     # per-view style factors
RES=${RES:-64}                            # cubic resolution (power of 2)
STEPS=${STEPS:-20000}
EVAL_EVERY=${EVAL_EVERY:-2000}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-3}
BETA_KL=${BETA_KL:-1e-4}

# ---- Software & environment ----
module load anaconda3/2022.10-gcc-13.2.0

CONDA_ENV_NAME="multiview-env"
PYTHON="${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python"
export PYTHONNOUSERSITE=1

# Rebuild the env only if torch/numpy are missing (needs an internet-capable
# node — pre-build multiview-env beforehand on CREATE login nodes).
if ! "$PYTHON" -c "import torch, numpy" 2>/dev/null; then
    echo "Environment '${CONDA_ENV_NAME}' missing or broken -- rebuilding..."
    conda create -n "${CONDA_ENV_NAME}" python=3.10 -y
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    "$PYTHON" -m pip install numpy scikit-learn tensorboard pandas matplotlib
fi
if [ -f "${SLURM_SUBMIT_DIR}/docker/requirements.txt" ]; then
    "$PYTHON" -m pip install -r "${SLURM_SUBMIT_DIR}/docker/requirements.txt" || echo "Requirements sync skipped a broken package."
fi
echo "Environment setup complete."

# ---- Working directory ----
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

# ---- Train ----
"$PYTHON" -m training.main_ae_synthetic \
    --model-id "${MODEL_ID}" \
    --content-channels "${CONTENT_CHANNELS}" \
    --latent-channels "${LATENT_CHANNELS}" \
    --n-content "${N_CONTENT}" \
    --n-style "${N_STYLE}" \
    --res "${RES}" \
    --train-steps "${STEPS}" \
    --eval-every "${EVAL_EVERY}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --num-workers 8 \
    --recon-weight 1 \
    --contrastive-weight 1
