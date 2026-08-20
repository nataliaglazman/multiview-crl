set -euo pipefail

REPO=${REPO:-"/nfs/home/nglazman/crl-2/multiview-crl"}
STEPS=${STEPS:-200000}
RES=${RES:-64}
TAG=${TAG:-synthetic-baseline-psuedo-mri-simplify-recon-increase-proportions-real}

PYTHONPATH=${REPO} \
python ${REPO}/training/main_multimodal.py \
    --dataset_name synthetic \
    --synthetic-mode pseudo_mri \
    --vqvae-nb-levels 1 \
    --content-ratios 0.95 \
    --vqvae-hidden-channels 48 --vqvae-embed-dim 48 \
    --content-style-levels 0 --content-size 9 \
    --mask-mode fixed \
    --model-id "${TAG}" \
    --synthetic-res ${RES} \
    --synthetic-num-train 2000 \
    --synthetic-num-val 200 \
    --synthetic-num-test 400 \
    --vqvae-scaling-rates 2 \
    --scale-contrastive-loss 1.0 --scale-recon-loss 1.0 \
    --contrastive-loss-type infonce --tau 0.1 \
    --batch-size 32 --train-steps ${STEPS} --lr 1e-3 \
    --separate-encoders \
    --resume-training \
    --quantize-style \
    --patch-contrastive \
    --pass-full-to-next-level
