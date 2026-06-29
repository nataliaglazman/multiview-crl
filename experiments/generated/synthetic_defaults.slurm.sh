#!/bin/bash -l
# Auto-generated from: experiments/synthetic_defaults.yaml
# Generated at: 2026-06-29T13:45:59Z
# Git SHA: 1a81090
# Re-generate with: python scripts/launch.py --generate --cluster slurm
#SBATCH --job-name=synthetic_defaults
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --error=synthetic_defaults-%j.err
#SBATCH --partition=biomed_a100_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --constraint=a100_80g

# -- Software & Environment Setup --
module load anaconda3/2022.10-gcc-13.2.0

CONDA_ENV_NAME="multiview-env"
PYTHON="${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python"

export PYTHONNOUSERSITE=1

# Automatically repair/build the environment if numpy or torch are missing
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

# -- Training --
"$PYTHON" -m training.main_multimodal \
    --batch-size 32 \
    --channels-last \
    --checkpoint-steps 500 \
    --content-dim 128 \
    --content-ratios 0.95 \
    --content-style-levels 0 \
    --contrastive-loss-type infonce \
    --cross-view-negs-only \
    --dataroot /scratch/users/k24058220 \
    --dataset-name synthetic \
    --dci-every 2000 \
    --deterministic \
    --eval-dci \
    --image-spacing 1.0 \
    --lr 0.001 \
    --mask-mode fixed \
    --moco-queue-size 0 \
    --pass-full-to-next-level \
    --patch-contrastive \
    --quantize-style \
    --recon-loss-start-step 0 \
    --resume-training \
    --scale-adv-loss 0.0 \
    --scale-content-modality-adv 0.0 \
    --scale-contrastive-loss 1.0 \
    --scale-recon-loss 0.0 \
    --scale-style-contrastive-loss 0.0 \
    --scale-style-modality-ce 0.0 \
    --select-by-gated-score \
    --separate-encoders \
    --separation-floor-diagnosis-info 0.1 \
    --synthetic-mode pseudo_mri \
    --synthetic-num-test 400 \
    --synthetic-num-train 2000 \
    --synthetic-num-val 200 \
    --synthetic-res 64 \
    --tau 0.1 \
    --total-dim 512 \
    --train-steps 200000 \
    --use-amp \
    --use-wandb \
    --vq-commitment-weight 0.25 \
    --vqvae-embed-dim 48 \
    --vqvae-hidden-channels 48 \
    --vqvae-nb-entries 256 \
    --vqvae-nb-levels 1 \
    --vqvae-scaling-rates 2 \
    --workers 8
