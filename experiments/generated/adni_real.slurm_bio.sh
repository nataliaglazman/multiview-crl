#!/bin/bash -l
# Auto-generated from: experiments/adni_real.yaml
# Generated at: 2026-09-28T08:13:56Z
# Git SHA: 7e027c5
# Re-generate with: python scripts/launch.py --generate --cluster slurm
#SBATCH --job-name=adni-real-bt-patch
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --error=adni-real-bt-patch-%j.err
#SBATCH --partition=biomed_a100_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

# -- Software & Environment Setup --
module load anaconda3/2022.10-gcc-13.2.0

CONDA_ENV_NAME="multiview-env"
PYTHON="${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

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
    --batch-size 16 \
    --bt-corr-ema 0.99 \
    --bt-gap-lambda 6 \
    --bt-gap-sim-coeff 0.05 \
    --bt-gap-std-coeff 0.05 \
    --bt-gap-weight 1 \
    --bt-lambda 6 \
    --bt-patch-weight 1 \
    --bt-sim-coeff 0.0114 \
    --bt-std-coeff 0.227 \
    --cache-dataset \
    --cache-dir /scratch/users/k24058220/cache/multiview \
    --channels-last \
    --checkpoint-steps 1000 \
    --content-dim 128 \
    --content-size 40 \
    --content-style-levels 0 \
    --contrastive-loss-type barlow_twins \
    --cross-recon-start-step 0 \
    --cross-view-negs-only \
    --dataroot /scratch/users/k24058220 \
    --dataset-name ADNI_stripped_masks \
    --decoder-norm-type group \
    --deterministic \
    --grad-clip-norm 100 \
    --gradient-checkpointing \
    --image-spacing 2.0 \
    --inject-style-to-decoder \
    --labels-path /users/k24058220/multiview-crl/labels_cleaned_3class.csv \
    --log-steps 50 \
    --lr 0.001 \
    --mask-mode fixed \
    --masks-dir /scratch/users/k24058220/ADNI_stripped_masks \
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
    --scale-contrastive-loss 100 \
    --scale-cross-recon-loss 0.0 \
    --scale-recon-loss 16 \
    --scale-style-contrastive-loss 0.0 \
    --scale-style-hsic-loss 0.0 \
    --scale-style-modality-ce 0.0 \
    --select-by-gated-score \
    --separate-style-codebooks \
    --separation-floor-diagnosis-info 0.1 \
    --single-count-commitment \
    --spatial-size 96 112 96 \
    --split-seed 0 \
    --style-alignment-var-weight 1.0 \
    --style-contrastive-mode cosine \
    --style-independence-var-weight 1.0 \
    --style-injection-mode input \
    --model-id adni-real-bt-patch \
    --tau 0.1 \
    --test-frac 0.1 \
    --total-dim 512 \
    --train-steps 20000 \
    --use-amp \
    --use-wandb \
    --val-frac 0.2 \
    --vq-commitment-weight 0.25 \
    --vqvae-embed-dim 48 \
    --vqvae-hidden-channels 48 \
    --vqvae-nb-entries 256 \
    --vqvae-nb-levels 1 \
    --vqvae-nb-res-layers 2 \
    --vqvae-scaling-rates 4 \
    --workers 8
