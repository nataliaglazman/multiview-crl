#!/usr/bin/env bash
# Phase 2: sweep absolute content channels at L0 while L1/L2 are fixed.
# Selection rule: smallest C_content_L0 where anatomy probe plateaus AND
# modality probe stays at chance.
#
# hidden_channels = 32 (baseline). Content_L0 ∈ {4, 8, 13, 16, 20} channels.
# L1 = 10 ch (0.3125 ≈ 0.3), L2 = 10 ch (0.3125 ≈ 0.3) — matches baseline.
#
# Run sequentially on one GPU, or copy/paste individual blocks across GPUs.

set -euo pipefail

REPO=/nfs/home/nglazman/crl-2/multiview-crl
HIDDEN=32
RATIO_L1=0.3125   # 10 / 32
RATIO_L2=0.3125   # 10 / 32
SWEEP_STEPS=20000 # enough for probes to stabilize; bump if they're still moving

# C_content_L0 values to try (absolute channels out of 32)
for C0 in 4 8 13 16 20; do
    RATIO_L0=$(python -c "print(${C0}/${HIDDEN})")
    TAG="phase2-cL0-${C0}"

    echo "=============================================================="
    echo "Run: ${TAG} | ratios = ${RATIO_L0} ${RATIO_L1} ${RATIO_L2}"
    echo "=============================================================="

    PYTHONPATH=${REPO} \
    python ${REPO}/training/main_multimodal.py \
        --dataroot /nfs/home/nglazman/data \
        --dataset_name ADNI_stripped_masks \
        --encoder-type vqvae \
        --model-id "${TAG}" \
        --scale-recon-loss 10 \
        --quantize-style \
        --patch-contrastive --patch-grid 4 5 4 \
        --labels-path /nfs/home/nglazman/nmpevqvae/labels_cleaned_3class.csv \
        --moco-queue-size 8192 \
        --vqvae-nb-entries 256 128 64 \
        --moco-momentum 0.99 \
        --mask-mode fixed \
        --vqvae-nb-levels 3 \
        --lr 0.001 \
        --content-dim 128 \
        --total-dim 512 \
        --vqvae-scaling-rates 2 2 2 \
        --train-steps ${SWEEP_STEPS} \
        --vqvae-hidden-channels ${HIDDEN} \
        --vqvae-embed-dim 32 \
        --batch-size 4 \
        --spatial-size 150 180 150 \
        --use-amp \
        --workers 8 \
        --recon-loss-start-step 2000 \
        --tau 0.8 \
        --scale-contrastive-loss 10.0 \
        --content-style-levels 0 1 2 \
        --cross-view-negs-only \
        --scale-style-contrastive-loss 10 \
        --content-ratios ${RATIO_L0} ${RATIO_L1} ${RATIO_L2} \
        --separate-encoders \
        --gradient-accumulation-steps 4 \
        --gradient-checkpointing \
        --skip-recon-ratio 0.5 \
        --image-spacing 1.0 \
        --cache-dataset \
        --vq-commitment-weight 0.25 \
        --cache-dir /nfs/home/nglazman/cache/multiview \
        --contrastive-level-weights 1.0 1.0 1.0 \
        --pass-full-to-next-level
done

echo "Done. Compare runs in W&B by tag prefix 'phase2-cL0-'."
echo "Pick the smallest C0 where content_anatomy_probe_L0 plateaus AND"
echo "content_modality_probe_L0 is at chance."
