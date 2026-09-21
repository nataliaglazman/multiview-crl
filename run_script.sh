
#!/usr/bin/env bash
# Auto-generated from: experiments/synthetic_causal.yaml
# Generated at: 2026-09-15T09:40:00Z
# Git SHA: 7eaca16
# Re-generate with: python scripts/launch.py --generate --cluster runai

set -euo pipefail

# --- Training command (folded to one line: see note in scripts/launch.py) ---
TRAIN_CMD=$(tr '\n' ' ' <<'TRAIN_EOF'
cd /nfs/home/nglazman/crl-2/multiview-crl || { echo ERROR: /nfs/home/nglazman/crl-2/multiview-crl is missing inside the container - check the --host-path mount of /nfs >&2 ; exit 1 ; } ;
export PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl ;
python -m training.main_conv_synthetic \
  --model-id dummy_infonce --res 128 --downscale-factor 8 \
  --latent-dim 12 --content-channels 9 --n-content 9 --n-style 3 \
  --batch-size 32 --hidden-channels 64 \
  --no-cache \
  --synthetic-normalize fixed_reference --synthetic-clean-content \
  --num-train-samples 1000 --num-val-samples 200 \
  --train-steps 50000 --eval-every 2000 --synthetic-causal --synthetic-causal-graph random --synthetic-causal-edge-prob 0.5
TRAIN_EOF
)

# --- RunAI submission ---
runai training standard submit infonce-dummy \
    --project nglazman \
    --image aicregistry:5000/nglazman:multiview-crl \
    --run-as-user \
    --large-shm \
    --node-type A100 \
    --gpu-devices-request 1 \
    --cpu-core-request 16 \
    --cpu-core-limit 32 \
    --cpu-memory-request 64G \
    --cpu-memory-limit 128G \
    --host-path path=/nfs,mount=/nfs,readwrite \
    --environment "WANDB_DIR=/tmp" \
    --environment "WANDB_API_KEY=${WANDB_API_KEY:?export WANDB_API_KEY before submitting - get it from https://wandb.ai/authorize}" \
    --command -- bash -c "${TRAIN_CMD}"
