#!/bin/bash -l
# Auto-generated from experiments/encoder/conv_mlp_burden_oracle_k10.yaml (encoder-only: training.main_conv_synthetic); edit the YAML, not this file.
# Re-generate with: python scripts/launch.py experiments/encoder/conv_mlp_burden_oracle_k10.yaml --cluster slurm_bio --generate
# Submit from the repository root: sbatch <this script>. Preview: bash <this script> --dry-run
#SBATCH --job-name=conv_mlp_burden_oracle_k10_s42
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
    --batch-size 32
    --best-metric none
    --no-cache
    --content-channels 9
    --contrastive-loss-type infonce
    --contrastive-proj-dim 0
    --conv-readout mlp
    --cpu-threads 1
    --cross-view-negs-only
    --data-seed 42
    --deterministic
    --deterministic-warn-only
    --downscale-factor 4
    --encoder-architecture conv
    --encoder-head-hidden 100
    --eval-every 2000
    --eval-pooling gap
    --floor-eval
    --grad-clip 2.0
    --hash-training-inputs
    --hidden-channels 64
    --latent-dim 12
    --lr 0.0001
    --n-content 9
    --n-style 3
    --nb-res-layers 2
    --num-train-samples 2000
    --num-val-samples 400
    --num-workers 0
    --out-dir results/encoder/runs
    --require-new-run
    --res 64
    --res-channels 32
    --seed 42
    --standardize-code
    --stratify-batches oracle
    --stratify-clusters 10
    --synthetic-clean-content
    --synthetic-content-scale 1.0
    --synthetic-lesion-intensity styled
    --synthetic-lesion-placement wm_interior
    --synthetic-lesion-t1-value 0.0
    --synthetic-lesion-target burden
    --synthetic-mode pseudo_mri
    --synthetic-normalize fixed_reference
    --synthetic-style-scale 1.0
    --tau 0.1
    --train-steps 10000
    --model-id conv_mlp_burden_oracle_k10_s42
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
