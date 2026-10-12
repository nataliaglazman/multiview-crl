#!/bin/bash -l
# Auto-generated from experiments/encoder/conv_mlp_gumbel.yaml (encoder-only: training.main_conv_synthetic); edit the YAML, not this file.
# Re-generate with: python scripts/launch.py experiments/encoder/conv_mlp_gumbel.yaml --cluster slurm --generate
# Submit from the repository root: sbatch <this script>. Preview: bash <this script> --dry-run
#SBATCH --job-name=conv_mlp_gumbel_s42
#SBATCH --output=/scratch/users/%u/%x-%j.out
#SBATCH --error=/scratch/users/%u/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --constraint=a100|h200|l40s

set -euo pipefail

# sbatch runs a spooled copy of this script, so the repository is the submit directory.
REPO="${ENCODER_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONDA_ENV_NAME=multiview-env
PYTHON="${ENCODER_PYTHON:-${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python}"
TRAIN_ARGS=(
    --attention-pool-frequencies 4
    --attention-pool-heads 4
    --batch-size 32
    --best-metric none
    --bt-lambda 0.005
    --no-cache
    --content-channels 9
    --contrastive-loss-type infonce
    --contrastive-proj-dim 0
    --contrastive-proj-hidden 256
    --conv-readout mlp
    --cpu-threads 1
    --cross-view-negs-only
    --data-seed 42
    --dataset-name synthetic
    --deterministic
    --deterministic-warn-only
    --device cuda
    --downscale-factor 4
    --encoder-architecture conv
    --encoder-head-hidden 100
    --eval-every 2000
    --eval-patch-grid 4 5 4
    --eval-pooling gap
    --floor-eval
    --global-pool gap
    --grad-clip 2.0
    --hash-training-inputs
    --hidden-channels 64
    --image-spacing 2.0
    --latent-dim 12
    --lesion-decorrelation-weight 0.0
    --lesion-detector shared
    --lesion-frame brain
    --lesion-head-init random
    --lesion-input residual
    --lesion-keypoints 4
    --lesion-loss-weight 1.0
    --lesion-norm none
    --lesion-normative-components 20
    --lesion-normative-subjects 300
    --lesion-pairing cross_modal
    --lesion-proj-dim 8
    --lesion-temperature 0.03
    --loader-seed 10042
    --log-every 100
    --lr 0.0001
    --mask-estimator relaxed
    --mask-init-gap 4.0
    --mask-lr-scale 10
    --mask-mode learned
    --model-seed 42
    --n-content 9
    --n-style 3
    --nb-res-layers 2
    --norm-type group
    --num-train-samples 2000
    --num-val-samples 400
    --num-workers 0
    --out-dir results/encoder/runs
    --patch-foreground-thresh 0.05
    --patch-loss-weight 0.0
    --require-new-run
    --res 64
    --res-channels 32
    --resnet-norm batch
    --resnet-output-stride 32
    --seed 42
    --no-separate-spatial-readout
    --spatial-recovery-batch-size 4
    --spatial-recovery-eval
    --spatial-recovery-grids 1 8
    --no-spatial-recovery-native
    --spatial-recovery-seed 1729
    --spatial-recovery-test-samples 400
    --split-seed 0
    --synthetic-causal
    --synthetic-causal-edge-prob 0.5
    --synthetic-causal-graph random
    --synthetic-causal-noise-scale 0.4
    --synthetic-causal-nonlinearity leaky_relu
    --synthetic-clean-content
    --synthetic-content-scale 1.0
    --synthetic-identifiable-ventricle
    --synthetic-lesion-count 4
    --synthetic-lesion-intensity styled
    --synthetic-lesion-placement wm_interior
    --synthetic-lesion-radius 0.1
    --synthetic-lesion-t1-value 0.4
    --synthetic-lesion-target position
    --synthetic-mode pseudo_mri
    --synthetic-n-deformation-grid 4
    --synthetic-n-fissure-grid 8
    --synthetic-normalize fixed_reference
    --synthetic-style-scale 1.0
    --synthetic-sulcal-mode atrophy
    --tau 0.1
    --test-frac 0.0
    --train-patch-grid 8 8 8
    --train-steps 10000
    --val-frac 0.0
    --model-id conv_mlp_gumbel_s42
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
