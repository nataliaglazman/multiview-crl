#!/usr/bin/env bash
# Content-dimensionality sweep on synthetic pseudo-MRI (Yao-style block identifiability).
#
# Goal: find the content-channel count k = --content-size whose value matches the
# true recoverable shared dimension n_c* of the data — the elbow between
# under-allocation (content can't hold all shared factors) and over-allocation
# (spare capacity absorbs view-specific style, breaking "content is clean of style").
#
# Pre-registered TWO-SIDED decision rule (applied by scripts/analyze_content_dim.py):
#   k* = min content_size such that, on the mean over seeds:
#     dci_synthetic/content->content/informativeness_test  >= 0.9 * max   (completeness, lower fence)
#     dci_synthetic/content->view/acc                       <= 0.55        (style leakage, upper fence)
#   Headline number at k*: dci_synthetic/content->content/block_mcc.
#
# Design choices (see the project evaluation notes):
#   * Single knob: --content-size (broadcasts to content_ratios across the level).
#   * mask_mode=fixed: deterministic first-k split — no Gumbel noise, fully controlled k.
#   * ALL auxiliary modality losses OFF (content/style adversaries, style-contrastive,
#     GAN) so content->view leakage reflects the capacity, not an adversary forcing
#     it to chance.
#
# Two arms (the reconstruction confound is itself a result):
#   deploy — recon on  (scale_recon_loss=1, inject_style, quantize_style): brackets k*.
#   theory — recon off (scale_recon_loss=0, decoder skipped): pure contrastive plateau.
# Two regimes:
#   causal       — dependent content factors (the block-identifiability testbed).
#   independent  — control: true dim is unambiguous; the elbow should land at n_c*.
#
# Cluster fan-out: one `scripts/launch.py ... --cluster <C>` submission per point.
#
# !!! launch.py parses --set with ast.literal_eval, so booleans MUST be Python-cased
#     (True/False). Lowercase true/false would be read as truthy STRINGS and a
#     store-true flag like inject_style_to_decoder could not be turned off. !!!
#
# Usage (coarse pass to locate the elbow, then widen SEEDS for the fine pass):
#   ./scripts/sweep_content_dim_synthetic.sh
#   SEEDS="0 1 2" K_VALUES="5 6 7 8 9 10 12" ./scripts/sweep_content_dim_synthetic.sh
#   CLUSTER=slurm REGIMES=causal ARMS=theory ./scripts/sweep_content_dim_synthetic.sh

set -euo pipefail

CLUSTER=${CLUSTER:-runai}
EXP=${EXP:-experiments/synthetic_causal.yaml}
WANDB_PROJECT=${WANDB_PROJECT:-multiview-crl-content-dim}
STEPS=${STEPS:-60000}
DCI_EVERY=${DCI_EVERY:-2000}

# Grid. Coarse default (1 seed) locates the elbow; rerun with SEEDS="0 1 2" and a
# finer K_VALUES band around the elbow for the headline numbers. hidden_channels=48
# in synthetic_defaults, so content_size must stay in [1, 47].
K_VALUES=${K_VALUES:-"2 4 6 8 12 16 24 32"}
SEEDS=${SEEDS:-"0"}
REGIMES=${REGIMES:-"causal independent"}
ARMS=${ARMS:-"deploy theory"}

# Overrides shared by every run: single knob + deterministic mask + all auxiliary
# modality / style-contrastive / GAN losses disabled.
COMMON_OFF="mask_mode=fixed scale_content_modality_adv=0.0 scale_content_patch_modality_adv=0.0 scale_style_modality_ce=0.0 scale_style_contrastive_loss=0.0 scale_adv_loss=0.0"

submit () {
  local regime="$1" arm="$2" k="$3" seed="$4"
  local group="cdim-${regime}-${arm}"
  local tag="${group}-k${k}-s${seed}"

  local regime_ov
  case "${regime}" in
    causal)      regime_ov="synthetic_causal=True synthetic_causal_graph=random" ;;
    independent) regime_ov="synthetic_causal=False synthetic_hierarchical_content=False" ;;
    *) echo "unknown regime: ${regime}" >&2; exit 1 ;;
  esac

  local arm_ov
  case "${arm}" in
    deploy) arm_ov="scale_recon_loss=1.0 inject_style_to_decoder=True quantize_style=True skip_recon_ratio=0.0" ;;
    theory) arm_ov="scale_recon_loss=0.0 inject_style_to_decoder=False quantize_style=False skip_recon_ratio=1.0" ;;
    *) echo "unknown arm: ${arm}" >&2; exit 1 ;;
  esac

  echo "── submit ${tag}  (cluster=${CLUSTER})"
  python scripts/launch.py "${EXP}" --cluster "${CLUSTER}" --set \
    content_size="${k}" seed="${seed}" \
    ${COMMON_OFF} ${regime_ov} ${arm_ov} \
    train_steps="${STEPS}" dci_every="${DCI_EVERY}" eval_dci=True \
    use_wandb=True wandb_project="${WANDB_PROJECT}" wandb_group="${group}" tag="${tag}"
}

n=0
for regime in ${REGIMES}; do
  for arm in ${ARMS}; do
    for k in ${K_VALUES}; do
      for seed in ${SEEDS}; do
        submit "${regime}" "${arm}" "${k}" "${seed}"
        n=$((n + 1))
      done
    done
  done
done

echo ""
echo "Submitted ${n} runs to project '${WANDB_PROJECT}'."
echo "Analyse each (regime, arm) cell once the runs finish, e.g.:"
for regime in ${REGIMES}; do
  for arm in ${ARMS}; do
    echo "  python scripts/analyze_content_dim.py --wandb-project ${WANDB_PROJECT} --group cdim-${regime}-${arm}"
  done
done
