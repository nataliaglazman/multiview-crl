#!/bin/bash -l
# Auto-generated from: experiments/synthetic_causal.yaml
# Generated at: 2026-08-11T12:43:34Z
# Git SHA: 3c46722
# Re-generate with: python scripts/launch.py --generate --cluster slurm
#SBATCH --job-name=synthetic-causal-clean-content-lambda-01-mse-25-10-recon-fixed
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --error=synthetic-causal-clean-content-lambda-01-mse-25-10-recon-fixed-%j.err
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
if ! "$PYTHON" -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('torch') and importlib.util.find_spec('numpy') else 1)" 2>/dev/null; then
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

# -- GPU preflight --
echo "Node: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || echo "WARNING: nvidia-smi unavailable on $(hostname)"
if ! "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: GPU allocated but torch cannot use it on $(hostname). Aborting."
    "$PYTHON" -c "import torch; print(f'torch {torch.__version__} cuda={torch.version.cuda}')"
    exit 1
fi

# -- Training --
"$PYTHON" -m training.main_multimodal \
    --batch-size 128 \
    --bt-gap-lambda 0.01 \
    --bt-gap-weight 1 \
    --bt-lambda 1 \
    --bt-patch-weight 1 \
    --bt-sim-coeff 0.025 \
    --bt-sim-normalize \
    --bt-std-coeff 10 \
    --channels-last \
    --checkpoint-steps 1000 \
    --content-dim 128 \
    --content-ratios 0.95 \
    --content-size 44 \
    --content-style-levels 0 \
    --contrastive-loss-type barlow_twins \
    --cross-view-negs-only \
    --dataroot /scratch/users/k24058220 \
    --dataset-name synthetic \
    --dci-every 2000 \
    --decoder-norm-type group \
    --deterministic \
    --eval-dci \
    --image-spacing 1.0 \
    --inject-style-to-decoder \
    --log-steps 50 \
    --lr 0.001 \
    --mask-mode fixed \
    --moco-queue-size 0 \
    --no-final-recon-norm \
    --norm-type layer \
    --pass-full-to-next-level \
    --patch-center-mode position \
    --patch-contrastive \
    --patch-foreground-mask \
    --patch-foreground-thresh 0.05 \
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
    --synthetic-clean-content \
    --synthetic-mode pseudo_mri \
    --synthetic-normalize fixed_reference \
    --synthetic-num-test 400 \
    --synthetic-num-train 2000 \
    --synthetic-num-val 1500 \
    --synthetic-res 64 \
    --model-id synthetic-causal-clean-content-lambda-01-mse-25-10-recon-fixed \
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
    --vqvae-scaling-rates 4 \
    --workers 8
