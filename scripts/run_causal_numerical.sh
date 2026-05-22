#!/usr/bin/env bash
# Numerical experiments with causal content latents.
#
# The default S_k only shares 2 dims across all views — not enough
# for a meaningful causal graph. The configs below widen the shared
# content block to 4 dims so we can test chain / full / random DAGs.
#
# S_k layout (10 latent dims, 4 views, 6 dims each):
#   content (shared by all views): {0,1,2,3}
#   style view 0: {4,5}    style view 1: {6,7}
#   style view 2: {8,9}    style view 3: {4,6}  (partial overlap)
#
# Run a single experiment:   EXPT=chain  bash scripts/run_causal_numerical.sh
# Run all experiments:        bash scripts/run_causal_numerical.sh all

set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
EXPT=${1:-${EXPT:-chain}}

COMMON=(
    --n-views 4
    --S-k 0 1 2 3 4 5  0 1 2 3 6 7  0 1 2 3 8 9  0 1 2 3 4 6
    --latent-dim 10
    --encoding-size 20
    --n-mixing-layer 3
    --batch-size 4096
    --n-steps 100001
    --n-log-steps 500
    --lr 1e-4
    --seed 42
    --selection gumbel_softmax
)

run_expt() {
    local tag=$1; shift
    echo "========== ${tag} =========="
    PYTHONPATH=${REPO} python -m training.main_numerical \
        "${COMMON[@]}" \
        --model-id "${tag}" \
        "$@"
}

# ── 0. Independent baseline (no causal, no Wishart) ────────────
run_baseline() {
    run_expt "causal_baseline"
}

# ── 1. Chain:  z0 → z1 → z2 → z3 ──────────────────────────────
run_chain() {
    run_expt "causal_chain" \
        --causal \
        --causal-graph chain \
        --causal-noise-scale 0.4 \
        --causal-nonlinearity leaky_relu
}

# ── 2. Fully connected DAG ─────────────────────────────────────
run_full() {
    run_expt "causal_full" \
        --causal \
        --causal-graph full \
        --causal-noise-scale 0.4 \
        --causal-nonlinearity leaky_relu
}

# ── 3. Random Erdős–Rényi DAG (p=0.5) ─────────────────────────
run_random() {
    run_expt "causal_random_p05" \
        --causal \
        --causal-graph random \
        --causal-edge-prob 0.5 \
        --causal-noise-scale 0.4 \
        --causal-nonlinearity leaky_relu
}

# ── 4. Noise-scale sweep on chain (identifiability vs. SNR) ───
run_noise_sweep() {
    for ns in 0.1 0.4 0.8; do
        run_expt "causal_chain_ns${ns}" \
            --causal \
            --causal-graph chain \
            --causal-noise-scale ${ns} \
            --causal-nonlinearity leaky_relu
    done
}

# ── 5. Linear structural equations (ablation) ─────────────────
run_linear() {
    run_expt "causal_chain_linear" \
        --causal \
        --causal-graph chain \
        --causal-noise-scale 0.4 \
        --causal-nonlinearity linear
}

# ── dispatch ───────────────────────────────────────────────────
case "${EXPT}" in
    baseline)     run_baseline ;;
    chain)        run_chain ;;
    full)         run_full ;;
    random)       run_random ;;
    noise_sweep)  run_noise_sweep ;;
    linear)       run_linear ;;
    all)
        run_baseline
        run_chain
        run_full
        run_random
        run_noise_sweep
        run_linear
        ;;
    *)  echo "Unknown experiment: ${EXPT}"; exit 1 ;;
esac
