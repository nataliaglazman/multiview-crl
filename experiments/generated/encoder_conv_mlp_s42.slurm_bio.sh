#!/bin/bash -l
# Auto-generated matched encoder ablation for slurm_bio.
# Re-generate with: python scripts/generate_encoder_ablation_slurm.py --include-controls
# Submit from repository root: sbatch experiments/generated/<name>.slurm_bio.sh
#SBATCH --job-name=encoder-ablation-bio-conv-mlp-s42
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

REPO="${ENCODER_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONDA_ENV_NAME=multiview-env
PYTHON="${ENCODER_PYTHON:-${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python}"
OUT_DIR=results/encoder_ablations_slurm_bio/runs
if [[ "$OUT_DIR" != /* ]]; then OUT_DIR="$REPO/$OUT_DIR"; fi
TRAIN_ARGS=(
    --res 64
    --latent-dim 12
    --content-channels 9
    --n-content 9
    --n-style 3
    --hidden-channels 64
    --res-channels 32
    --nb-res-layers 2
    --downscale-factor 4
    --encoder-head-hidden 100
    --synthetic-mode pseudo_mri
    --synthetic-normalize fixed_reference
    --synthetic-clean-content
    --synthetic-lesion-placement wm_interior
    --synthetic-style-scale 1.0
    --synthetic-content-scale 1.0
    --num-train-samples 2000
    --num-val-samples 400
    --batch-size 32
    --num-workers 0
    --no-cache
    --contrastive-loss-type infonce
    --tau 0.1
    --cross-view-negs-only
    --contrastive-proj-dim 0
    --lr 0.0001
    --grad-clip 2.0
    --train-steps 10000
    --eval-every 2000
    --eval-pooling gap
    --floor-eval
    --best-metric none
    --deterministic
    --deterministic-warn-only
    --cpu-threads 1
    --hash-training-inputs
    --seed 42
    --data-seed 42
    --model-seed 42
    --loader-seed 10042
    --encoder-architecture conv
    --require-new-run
    --model-id conv_mlp_s42
    --conv-readout mlp
    --resnet-norm batch
    --resnet-output-stride 32
    --out-dir "$OUT_DIR"
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

# Reuse the prepared environment; concurrent jobs never install/remove packages.
module load anaconda3/2022.10-gcc-13.2.0
if [[ ! -x "$PYTHON" ]]; then
    echo "Python not found at $PYTHON. Prepare $CONDA_ENV_NAME or set ENCODER_PYTHON before sbatch." >&2
    exit 1
fi
export PYTHONPATH="$REPO"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
echo "Node: $(hostname)  Job: $SLURM_JOB_ID  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON" -c 'import torch; print(f'"'"'torch={torch.__version__} cuda={torch.version.cuda}'"'"'); assert torch.cuda.is_available(), '"'"'Allocated job has no usable CUDA device'"'"''
"$PYTHON" -m unittest tests.test_encoder_runtime -v

exec "$PYTHON" -m training.main_conv_synthetic "${TRAIN_ARGS[@]}"
