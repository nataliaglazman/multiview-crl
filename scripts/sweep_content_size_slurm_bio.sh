#!/usr/bin/env bash
# Content-size sweep on the brain encoder (Daunhawer et al., ICLR 2023, App. B.2): one sbatch job per
# (content size k, seed) of experiments/encoder/conv_mlp_content_size.yaml on biomed_a100_gpu. Each run
# writes val/loss (held-out InfoNCE) and per-factor R² to dci_step*.json every 2000 steps.
#
# On the CREATE login node, from the repository root:
#   DRY_RUN=1 bash scripts/sweep_content_size_slurm_bio.sh     # resolve and check every point, submit nothing
#   bash scripts/sweep_content_size_slurm_bio.sh               # submit 8 sizes x 3 seeds = 24 jobs
#   K_VALUES="4 9" SEEDS=42 bash scripts/sweep_content_size_slurm_bio.sh
# When they finish:
#   python scripts/analyze_content_size.py 'results/encoder/runs/conv_mlp_csize_k*_s*' --out results/content_size_brain
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
K_VALUES="${K_VALUES:-2 4 6 8 9 10 12 16}"
SEEDS="${SEEDS:-42 142 242}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_SET="${EXTRA_SET:-}"

cd "$REPO"
flags=()
[[ "$DRY_RUN" == 1 ]] && flags+=(--dry-run)

n=0
for k in $K_VALUES; do
  for seed in $SEEDS; do
    echo "== content_channels=$k seed=$seed"
    # shellcheck disable=SC2086
    "$PYTHON" scripts/launch.py experiments/encoder/conv_mlp_content_size.yaml --cluster slurm_bio \
      --set content_channels="$k" seed="$seed" $EXTRA_SET "${flags[@]}"
    n=$((n + 1))
  done
done
echo "== $n points $([[ "$DRY_RUN" == 1 ]] && echo checked || echo submitted)"
