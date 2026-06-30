#!/bin/bash -l
# Encoder-only multi-view contrastive learning on the 3D synthetic data
# (training/main_conv_synthetic.py — Yao et al. ICLR 2024 image track, in 3D):
# per-view conv encoders, content-alignment − entropy loss, NO decoder.
# Identifiability is scored with eval.dci.compute_dci_synthetic (R^2 + block-MCC).
#
# Submit from the repo root:
#     sbatch scripts/run_conv_synthetic_slurm.sh
#
# Common variants (override via environment at submit time):
#     # sweep the content size (find the elbow at the true n_content):
#     for c in 6 9 12; do MODEL_ID=conv_c$c CONTENT_CHANNELS=$c sbatch scripts/run_conv_synthetic_slurm.sh; done
#     # Barlow Twins instead of InfoNCE (the paper's image-track loss):
#     MODEL_ID=conv_bt LOSS_TYPE=barlow_twins sbatch scripts/run_conv_synthetic_slurm.sh
#     # clean content (named factors dominate) + style-preserving normalization:
#     MODEL_ID=conv_clean CLEAN_CONTENT=1 NORMALIZE=fixed_reference sbatch scripts/run_conv_synthetic_slurm.sh
#     # probe at patch resolution instead of GAP (tests the GAP ceiling):
#     MODEL_ID=conv_patch EVAL_POOLING=patch sbatch scripts/run_conv_synthetic_slurm.sh
#
# Output: results/<MODEL_ID>/ (settings.json, model.pt, dci_step*.json, TensorBoard).

#SBATCH --job-name=mvcrl-conv
#SBATCH --output=mvcrl-conv-%j.out
#SBATCH --error=mvcrl-conv-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

# ---- Experiment knobs (override via environment at submit time) ----
MODEL_ID=${MODEL_ID:-conv_synthetic}
LATENT_DIM=${LATENT_DIM:-16}              # total encoding size (content + style)
CONTENT_CHANNELS=${CONTENT_CHANNELS:-9}   # the "number of content dimensions" under test
N_CONTENT=${N_CONTENT:-9}                 # true shared content factors
N_STYLE=${N_STYLE:-3}
RES=${RES:-64}                            # cubic resolution (power of 2)
STEPS=${STEPS:-50000}
EVAL_EVERY=${EVAL_EVERY:-2000}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-4}
LOSS_TYPE=${LOSS_TYPE:-infonce}           # infonce | barlow_twins
EVAL_POOLING=${EVAL_POOLING:-gap}         # gap | patch
NORMALIZE=${NORMALIZE:-per_sample}        # per_sample | shared | fixed_reference
CLEAN_CONTENT=${CLEAN_CONTENT:-0}         # 1 -> zero the deformation/fissure nuisance
EXTRA_ARGS=${EXTRA_ARGS:-}

CLEAN_FLAG=""
if [ "${CLEAN_CONTENT}" = "1" ]; then
    CLEAN_FLAG="--synthetic-clean-content"
fi

# ---- Software & environment ----
module load anaconda3/2022.10-gcc-13.2.0

CONDA_ENV_NAME="multiview-env"
PYTHON="${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python"
export PYTHONNOUSERSITE=1

if ! "$PYTHON" -c "import torch, numpy, sklearn" 2>/dev/null; then
    echo "Environment '${CONDA_ENV_NAME}' missing or broken -- rebuilding..."
    conda create -n "${CONDA_ENV_NAME}" python=3.10 -y
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    "$PYTHON" -m pip install numpy scikit-learn scipy pyyaml tensorboard pandas matplotlib monai nibabel
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

# ---- Train (encoder-only contrastive) ----
"$PYTHON" -m training.main_conv_synthetic \
    --model-id "${MODEL_ID}" \
    --latent-dim "${LATENT_DIM}" \
    --content-channels "${CONTENT_CHANNELS}" \
    --n-content "${N_CONTENT}" \
    --n-style "${N_STYLE}" \
    --res "${RES}" \
    --train-steps "${STEPS}" \
    --eval-every "${EVAL_EVERY}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --contrastive-loss-type "${LOSS_TYPE}" \
    --eval-pooling "${EVAL_POOLING}" \
    --synthetic-normalize "${NORMALIZE}" \
    --num-workers 8 \
    ${CLEAN_FLAG} \
    ${EXTRA_ARGS}
