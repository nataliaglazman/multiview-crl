set -euo pipefail

REPO=${REPO:-"/nfs/home/nglazman/crl-2/multiview-crl"}
STEPS=${STEPS:-100000}
RES=${RES:-64}
TAG=${TAG:-synthetic-baseline-psuedo-mri-simplify}

PYTHONPATH=${REPO} \
python ${REPO}/training/main_multimodal.py \
    --dataset_name synthetic \
    --synthetic-mode pseudo_mri \
    --vqvae-nb-levels 1 \
    --vqvae-hidden-channels 32 --vqvae-embed-dim 32 \
    --content-style-levels 0 --content-size 16 \
    --mask-mode fixed \
    --model-id "${TAG}" \
    --synthetic-res ${RES} \
    --synthetic-num-train 2000 \
    --synthetic-num-val 200 \
    --synthetic-num-test 400 \
    --vqvae-scaling-rates 2 \
    --scale-contrastive-loss 1.0 --scale-recon-loss 0.0 \
    --contrastive-loss-type infonce --tau 0.1 \
    --batch-size 32 --train-steps ${STEPS} --lr 1e-3
