#!/bin/bash -l
# Auto-generated from experiments/encoder/adni_patch.yaml (encoder-only: training.main_conv_synthetic); edit the YAML, not this file.
# Re-generate with: python scripts/launch.py experiments/encoder/adni_patch.yaml --cluster slurm_bio --generate
# Submit from the repository root: sbatch <this script>. Preview: bash <this script> --dry-run
#SBATCH --job-name=encoder_adni_patch_s42
#SBATCH --output=/scratch/users/%u/%x-%j.out
#SBATCH --error=/scratch/users/%u/%x-%j.err
#SBATCH --partition=biomed_a100_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --constraint=a100_80g

set -euo pipefail

# sbatch runs a spooled copy of this script, so the repository is the submit directory.
REPO="${ENCODER_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONDA_ENV_NAME=multiview-env
PYTHON="${ENCODER_PYTHON:-${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python}"
TRAIN_ARGS=(
    --asymmetric-aug
    --batch-size 32
    --best-metric val_loss
    --cache-dir /scratch/users/k24058220/cache/multiview
    --content-channels 9
    --contrastive-loss-type infonce
    --contrastive-proj-dim 0
    --conv-readout mlp
    --cross-view-negs-only
    --dataroot /scratch/users/k24058220
    --dataset-name ADNI_stripped_masks
    --deterministic
    --deterministic-warn-only
    --downscale-factor 4
    --encoder-architecture conv
    --encoder-head-hidden 100
    --eval-every 500
    --floor-eval
    --grad-clip 2.0
    --hidden-channels 64
    --image-spacing 2.0
    --labels-path /users/k24058220/multiview-crl/labels_cleaned_3class_demog.csv
    --latent-dim 12
    --lr 0.0001
    --masks-dir /scratch/users/k24058220/ADNI_stripped_masks
    --nb-res-layers 2
    --norm-type group
    --num-workers 8
    --out-dir /scratch/users/k24058220/encoder_adni/runs
    --patch-foreground-mask
    --patch-loss-weight 1.0
    --require-new-run
    --res-channels 32
    --seed 42
    --spatial-size 96 112 96
    --split-seed 0
    --tau 0.1
    --test-frac 0.1
    --train-patch-grid 6 7 6
    --train-steps 10000
    --val-frac 0.2
    --model-id encoder_adni_patch_s42
)

if [[ "$#" -gt 1 ]]; then echo "Usage: $0 [--dry-run]" >&2; exit 2; fi
case "${1:-}" in
    --dry-run)
        printf '%q ' "$PYTHON" -m training.main_conv_synthetic "${TRAIN_ARGS[@]}"
        printf '\n'
        exit 0 ;;
    "") ;;
    *) echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit with sbatch from the repository root; use --dry-run to preview locally." >&2
    exit 2
fi
cd "$REPO"
if [[ ! -f training/main_conv_synthetic.py ]]; then
    echo "Repository missing at $REPO. Submit from its root or set ENCODER_REPO." >&2
    exit 1
fi

# Reuse the prepared environment; concurrent jobs never install or remove packages.
module load anaconda3/2022.10-gcc-13.2.0
if [[ ! -x "$PYTHON" ]]; then
    echo "Python not found at $PYTHON. Prepare $CONDA_ENV_NAME or set ENCODER_PYTHON before sbatch." >&2
    exit 1
fi
export PYTHONPATH="$REPO" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "Node: $(hostname)  Job: $SLURM_JOB_ID  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON" -c 'import torch; print(f'"'"'torch={torch.__version__} cuda={torch.version.cuda}'"'"'); assert torch.cuda.is_available(), '"'"'Allocated job has no usable CUDA device'"'"''
exec "$PYTHON" -m training.main_conv_synthetic "${TRAIN_ARGS[@]}"
