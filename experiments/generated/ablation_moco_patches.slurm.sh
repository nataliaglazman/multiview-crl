#!/bin/bash -l
# Auto-generated from: experiments/ablation_moco_patches.yaml
# Generated at: 2026-08-06T10:42:02Z
# Git SHA: b5efc3d
# Re-generate with: python scripts/launch.py --generate --cluster slurm
#SBATCH --job-name=ablation-moco-patches
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --error=ablation-moco-patches-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --constraint=a100|h200|l40s

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
    --batch-size 2 \
    --cache-dataset \
    --cache-dir /scratch/users/k24058220/cache/multiview \
    --channels-last \
    --checkpoint-steps 500 \
    --content-dim 128 \
    --content-style-levels 0 \
    --contrastive-level-weights 3.0 0.5 0.5 \
    --contrastive-loss-type infonce \
    --crop-margin 12 \
    --cross-view-negs-only \
    --dataroot /scratch/users/k24058220 \
    --dataset-name ADNI_stripped_masks \
    --deterministic \
    --gradient-accumulation-steps 4 \
    --gradient-checkpointing \
    --image-spacing 1.0 \
    --inject-style-to-decoder \
    --labels-path /users/k24058220/multiview-crl/labels_cleaned_3class.csv \
    --log-steps 50 \
    --lr 0.001 \
    --mask-mode fixed \
    --masks-dir /scratch/users/k24058220/ADNI_stripped_masks \
    --moco-momentum 0.99 \
    --moco-queue-size 8192 \
    --pass-full-to-next-level \
    --patch-contrastive \
    --patch-grid 4 5 4 \
    --recon-loss-start-step 2000 \
    --resume-training \
    --scale-adv-loss 0.0 \
    --scale-content-modality-adv 0.0 \
    --scale-contrastive-loss 1.0 \
    --scale-recon-loss 1.0 \
    --scale-style-contrastive-loss 0.0 \
    --scale-style-modality-ce 0.0 \
    --select-by-gated-score \
    --separate-encoders \
    --separation-floor-diagnosis-info 0.1 \
    --skip-recon-ratio 0.5 \
    --spatial-size 150 180 150 \
    --style-injection-mode film \
    --model-id ablation-moco-patches \
    --tau 0.07 \
    --total-dim 512 \
    --train-steps 50000 \
    --use-amp \
    --use-moco \
    --use-wandb \
    --vq-commitment-weight 0.25 \
    --vqvae-embed-dim 38 \
    --vqvae-hidden-channels 64 \
    --vqvae-nb-entries 256 \
    --vqvae-nb-levels 3 \
    --vqvae-scaling-rates 2 2 2 \
    --workers 8
