#!/bin/bash -l
# Faithful encoder-only multi-view CRL numerical experiment
# (training/main_numerical.py — Yao et al. ICLR 2024, "Multi-View Causal
# Representation Learning with Partial Observability", §5.1 / App D.1).
#
# This is the paper's method, NOT the VQ-VAE/recon track: per-view MLP encoders +
# random fixed invertible-MLP mixing functions (x_k = f_k(z_Sk)), partial
# observability via --S-k, content = subset intersections, content selectors
# (Gumbel-Softmax), loss = content-alignment − entropy (InfoNCE). No decoder.
#
# Submit from the repo root:
#     sbatch scripts/run_numerical_slurm.sh
#
# Common variants (override via environment at submit time):
#     # Fig 2 — independent latents (default, Sigma_z = I):
#     MODEL_ID=num_independent sbatch scripts/run_numerical_slurm.sh
#     # Fig 4 — causally dependent latents (Wishart Sigma_z over the first dims):
#     MODEL_ID=num_dependent N_DEPENDENT_DIMS=6 sbatch scripts/run_numerical_slurm.sh
#     # explicit causal SCM (repo extension):
#     MODEL_ID=num_causal CAUSAL=1 CAUSAL_GRAPH=chain sbatch scripts/run_numerical_slurm.sh
#     # paper uses 3 seeds — launch three jobs:
#     for s in 0 1 2; do MODEL_ID=num_independent_s$s SEED=$s sbatch scripts/run_numerical_slurm.sh; done
#     # selector ablation (oracle indices instead of learned Gumbel):
#     MODEL_ID=num_oracle SELECTION=ground_truth sbatch scripts/run_numerical_slurm.sh
#
# Output: results/numerical/<MODEL_ID>/ (Training.csv, Evaluation.csv, model.pt,
# mixing_fns.pt, and Sigma_z.csv / scm.pt). Evaluation.csv holds the per-latent
# linear + nonlinear (RBF kernel-ridge) R^2 — content latents -> ~1, independent
# style latents -> ~0 is the block-identification signal.

#SBATCH --job-name=mvcrl-numerical
#SBATCH --output=mvcrl-numerical-%j.out
#SBATCH --error=mvcrl-numerical-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

# ---- Experiment knobs (override via environment at submit time) ----
MODEL_ID=${MODEL_ID:-num_independent}
N_VIEWS=${N_VIEWS:-4}                     # default --S-k is Example 2.1 (N=6, K=4); set --S-k via EXTRA_ARGS if you change this
SELECTION=${SELECTION:-gumbel_softmax}    # gumbel_softmax | ground_truth | concat
N_STEPS=${N_STEPS:-100001}
BATCH_SIZE=${BATCH_SIZE:-4096}
LR=${LR:-1e-4}
SEED=${SEED:-42}
N_DEPENDENT_DIMS=${N_DEPENDENT_DIMS:-0}   # >0 -> Wishart Sigma_z (dependent latents, Fig 4)
ENCODING_SIZE=${ENCODING_SIZE:-20}        # only used when SELECTION=concat
RUN_EVAL=${RUN_EVAL:-1}                   # after training, run an --evaluate pass (adds nonlinear kernel-ridge R^2)
EXTRA_ARGS=${EXTRA_ARGS:-}                # e.g. EXTRA_ARGS="--grid-search-eval" or a custom --S-k ...

# Optional explicit causal SCM (repo extension). Mutually exclusive with N_DEPENDENT_DIMS>0.
CAUSAL=${CAUSAL:-0}
CAUSAL_GRAPH=${CAUSAL_GRAPH:-chain}       # chain | full | random
CAUSAL_FLAG=""
if [ "${CAUSAL}" = "1" ]; then
    CAUSAL_FLAG="--causal --causal-graph ${CAUSAL_GRAPH}"
fi

# ---- Software & environment ----
module load anaconda3/2022.10-gcc-13.2.0

CONDA_ENV_NAME="multiview-env"
PYTHON="${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python"
export PYTHONNOUSERSITE=1

# Rebuild the env only if core deps are missing (needs an internet-capable node —
# pre-build multiview-env beforehand on login nodes).
if ! "$PYTHON" -c "import torch, numpy, sklearn, scipy, yaml" 2>/dev/null; then
    echo "Environment '${CONDA_ENV_NAME}' missing or broken -- rebuilding..."
    conda create -n "${CONDA_ENV_NAME}" python=3.10 -y
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    "$PYTHON" -m pip install numpy scikit-learn scipy pyyaml tensorboard pandas matplotlib
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

# ---- Train ----
"$PYTHON" -m training.main_numerical \
    --model-id "${MODEL_ID}" \
    --n-views "${N_VIEWS}" \
    --selection "${SELECTION}" \
    --n-steps "${N_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --n_dependent_dims "${N_DEPENDENT_DIMS}" \
    --encoding-size "${ENCODING_SIZE}" \
    ${CAUSAL_FLAG} \
    ${EXTRA_ARGS}

# ---- Evaluate (reloads model.pt + mixing_fns.pt; adds nonlinear kernel-ridge R^2) ----
if [ "${RUN_EVAL}" = "1" ]; then
    "$PYTHON" -m training.main_numerical \
        --model-id "${MODEL_ID}" \
        --n-views "${N_VIEWS}" \
        --selection "${SELECTION}" \
        --seed "${SEED}" \
        --n_dependent_dims "${N_DEPENDENT_DIMS}" \
        --encoding-size "${ENCODING_SIZE}" \
        ${CAUSAL_FLAG} \
        ${EXTRA_ARGS} \
        --evaluate
fi
