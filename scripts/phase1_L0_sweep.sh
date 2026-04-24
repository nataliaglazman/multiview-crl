#!/usr/bin/env bash
# Phase 1: L0 content-size sweep. Pre-registered protocol:
#   - hidden_channels = 32
#   - C_L0 ∈ {2, 4, 6, 8, 12, 16, 20, 24}
#   - 3 seeds per point → 24 runs
#   - no masking at L1/L2 (content_style_levels = [0])
#   - Everything else fixed (loss scales, tau, LR, queue, batch size)
#
# Decision rule (applied by scripts/analyze_capacity_sweep.py):
#   C*_L0 = min C such that
#     diag_info_L0  ≥ 0.9 · max(diag_info_L0)
#     modality_acc  ≤ 0.55
#     PSNR          ≥ max(PSNR) − 1.0 dB
#
# Launch: each loop iteration runs sequentially. On a cluster, copy/paste
# blocks across GPUs or fan out through your launcher (sweep_runai.sh style).

set -euo pipefail

REPO=${REPO:-/nfs/home/nglazman/crl-2/multiview-crl}
DATAROOT=${DATAROOT:-/nfs/home/nglazman/data}
LABELS=${LABELS:-/nfs/home/nglazman/nmpevqvae/labels_cleaned_3class.csv}
CACHE=${CACHE:-/nfs/home/nglazman/cache/multiview}
STEPS=${STEPS:-30000}
WANDB_PROJECT=${WANDB_PROJECT:-multiview-crl-capacity}
GROUP=${GROUP:-phase1-L0}

HIDDEN=32
C_L0_VALUES=(2 4 6 8 12 16 20 24)
SEEDS=(0 1 2)

for SEED in "${SEEDS[@]}"; do
    for C0 in "${C_L0_VALUES[@]}"; do
        RATIO_L0=$(python -c "print(${C0}/${HIDDEN})")
        TAG="phase1-cL0-${C0}-s${SEED}"

        echo "=============================================================="
        echo "Run: ${TAG} | C_L0=${C0}/${HIDDEN} (ratio=${RATIO_L0}) | seed=${SEED}"
        echo "=============================================================="

        PYTHONPATH=${REPO} \
        python ${REPO}/training/main_multimodal.py \
            --dataroot ${DATAROOT} \
            --dataset_name ADNI_stripped_masks \
            --labels-path ${LABELS} \
            --cache-dataset \
            --cache-dir ${CACHE} \
            --model-id "${TAG}" \
            --seed ${SEED} \
            --use-wandb \
            --wandb-project ${WANDB_PROJECT} \
            --wandb-group ${GROUP} \
            --encoder-type vqvae \
            --vqvae-nb-levels 3 \
            --vqvae-hidden-channels ${HIDDEN} \
            --vqvae-embed-dim 32 \
            --vqvae-nb-entries 256 \
            --vqvae-scaling-rates 2 2 2 \
            --vq-commitment-weight 0.25 \
            --content-style-levels 0 \
            --content-ratios ${RATIO_L0} \
            --mask-mode fixed \
            --separate-encoders \
            --pass-full-to-next-level \
            --contrastive-loss-type infonce \
            --cross-view-negs-only \
            --moco-queue-size 8192 \
            --moco-momentum 0.99 \
            --scale-contrastive-loss 1.0 \
            --scale-style-contrastive-loss 0.0 \
            --scale-content-modality-adv 0.0 \
            --scale-style-modality-ce 0.0 \
            --scale-recon-loss 1.0 \
            --scale-adv-loss 0.0 \
            --tau 0.1 \
            --lr 1e-3 \
            --batch-size 4 \
            --train-steps ${STEPS} \
            --spatial-size 150 180 150 \
            --image-spacing 1.0 \
            --use-amp \
            --workers 8 \
            --gradient-checkpointing \
            --separation-floor-diagnosis-info 0.1
    done
done

echo ""
echo "Sweep complete. Analyze with:"
echo "  python scripts/analyze_capacity_sweep.py \\"
echo "    --wandb-project ${WANDB_PROJECT} --group ${GROUP} --param content_ratios_L0"
