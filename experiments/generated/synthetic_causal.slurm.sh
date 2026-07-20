#!/bin/bash -l
# Auto-generated from: experiments/synthetic_causal.yaml
# Generated at: 2026-07-20T10:26:19Z
# Git SHA: a137ed5
# Re-generate with: python scripts/launch.py --generate --cluster slurm
#SBATCH --job-name=synthetic-causal-projection-entropy
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --error=synthetic-causal-projection-entropy-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

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
    --batch-size 64 \
    --channels-last \
    --checkpoint-steps 1000 \
    --content-dim 128 \
    --content-ratios 0.95 \
    --content-size 44 \
    --content-style-levels 0 \
    --contrastive-loss-type infonce \
    --contrastive-proj-dim 16 \
    --contrastive-proj-hidden 128 \
    --contrastive-proj-mode entropy \
    --cross-view-negs-only \
    --dataroot /scratch/users/k24058220 \
    --dataset-name synthetic \
    --dci-every 2000 \
    --deterministic \
    --eval-dci \
    --image-spacing 1.0 \
    --inject-style-to-decoder \
    --lr 0.001 \
    --mask-mode fixed \
    --moco-queue-size 0 \
    --no-final-recon-norm \
    --pass-full-to-next-level \
    --patch-grid 8 8 8 \
    --quantize-style \
    --recon-loss-start-step 0 \
    --resume-training \
    --scale-adv-loss 0.0 \
    --scale-content-modality-adv 0.0 \
    --scale-contrastive-loss 1 \
    --scale-recon-loss 1 \
    --scale-style-contrastive-loss 0.0 \
    --scale-style-modality-ce 0.0 \
    --select-by-gated-score \
    --separate-encoders \
    --separate-style-codebooks \
    --separation-floor-diagnosis-info 0.1 \
    --style-injection-mode input \
    --synthetic-causal \
    --synthetic-causal-edge-prob 0.5 \
    --synthetic-causal-graph random \
    --synthetic-mode pseudo_mri \
    --synthetic-normalize fixed_reference \
    --synthetic-num-test 400 \
    --synthetic-num-train 2000 \
    --synthetic-num-val 1500 \
    --synthetic-res 64 \
    --model-id synthetic-causal-projection-entropy \
    --tau 0.07 \
    --total-dim 512 \
    --train-steps 300000 \
    --use-amp \
    --use-wandb \
    --vq-commitment-weight 0.25 \
    --vqvae-embed-dim 48 \
    --vqvae-hidden-channels 48 \
    --vqvae-nb-entries 256 \
    --vqvae-nb-levels 1 \
    --vqvae-scaling-rates 4 \
    --workers 8
