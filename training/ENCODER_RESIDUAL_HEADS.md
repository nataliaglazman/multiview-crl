# Separate image and residual heads

`experiments/encoder/conv_mlp_burden_residual_heads_auxiliary.yaml` is the corrected
1,000-step pilot. It keeps the image InfoNCE objective and adds a weight-0.1 joint
auxiliary term, with independently normalized alignment for the two paths. Sparse
residual evidence has its own encoder and tail readout; the original image code
retains its contrastive training signal. This is an experiment, with no promised
burden-recovery improvement. Inspect the short pilot before launching a longer run.

The original `conv_mlp_burden_residual_heads.yaml` remains unchanged for
reproducibility. Its global InfoNCE weight is zero and its alignment uses detached
population gauges. Existing image-only recipes keep their behavior:
`--residual-head-units` defaults to `0`.

## Representation

For each modality, the original image follows the existing Conv encoder and GAP to
MLP readout. Independently, a fixed population residual follows a small Conv encoder,
tail pooling and its own head:

```text
raw image ----------------> existing encoder -> GAP -> image head -> a_k
          \-> fixed PCA residual -> small encoder -> tails -> head -> b_k

export: [original content, original style, residual code]
shared content: [a_k, b_k]
auxiliary: sigmoid MLP([a_k, b_k]) -> q_k
```

The original output indices retain their meaning. The residual code is appended
after all original units and included in the content mask; it is not appended to the
private style block. With the recipe, the total output width is 26: 14 original
content units, 4 original style units, then 8 residual content units. Shared content
has width 22. The auxiliary projector has 10 outputs and is not the exported code.

The residual model first standardizes each image over its nonzero brain voxels.
For each modality it fits a population-brain support (at least 50% training
occupancy), mean image and leading PCs. It subtracts their reconstruction and
divides by one scalar residual RMS per modality, estimated from that same training
cohort. The first `residual_head_subjects` training subjects supply the fit; no
separate validation or test calibration set is used. The PCA transform stays fixed
during optimization. Its mean, basis, scales, brain support and fitted state are
checkpoint buffers, so evaluation uses the fitted training transform. Validation/test
images and latent labels do not fit it.

Tail pooling records positive and negative excursions above thresholds 2, 3 and 4
using smooth softplus tails and log compression. It uses cells with at least 90%
mean brain occupancy in both modalities. A global feature mean is included too.
Its per-position running population statistics belong to the residual pathway and
are initialized on the training fit cohort. The two residual encoders have separate
parameters and share their residual readout head. Neither PCA nor tail pooling
identifies lesions by construction: residuals can contain anatomy, style and noise,
and the removed population directions can contain lesion information.

## Corrected objective

The pilot uses the existing image-content InfoNCE as its primary objective. Its
auxiliary combines image-block alignment, residual-block alignment and joint
uniformity:

```text
total = image_InfoNCE + 0.1 * auxiliary

auxiliary = image_alignment + residual_alignment
              + mean_k MMD²(q_k, Uniform((0, 1)^joint_dim))
```

`global_loss_weight: 1` multiplies image InfoNCE;
`residual_head_auxiliary_weight: 0.1` multiplies the complete auxiliary. The three
internal weights remain 1. The primary InfoNCE reads the original image content,
so the additional residual units cannot satisfy it on the image path's behalf.

Alignment uses `residual_head_alignment_mode: batch`. For each view and content
coordinate, it subtracts the current batch mean and divides by
`sqrt(batch_variance + epsilon)`. Both statistics remain on the gradient path.
Squared cross-view differences are then averaged separately in the image and
residual blocks. This removes the direct incentive to shrink both views together
when their variance is appreciably above epsilon. It does not guarantee that a
branch stays informative: constant codes are stationary, and epsilon matters near
zero variance. The loss also leaves each view's raw mean and scale unconstrained.

The legacy `population` mode instead computes alignment using detached running
means and scales. For a fixed gauge, shrinking a mismatched pair of codes can lower
that alignment loss. A running gauge does not remove the instantaneous gradient
incentive. Keep this mode for reproducing the original experiment rather than
assuming that its lower alignment loss demonstrates improved sharing.

The shared joint projector always uses the detached population code gauge,
including in `batch` alignment mode. Its per-subject input is therefore independent
of the other subjects in an evaluation batch. Population statistics are calibrated
on training data, updated during training, and fixed during evaluation. The
projector reads both content blocks, passes them through a shared sigmoid MLP, and
uses a biased, nonnegative multiscale RBF MMD against fresh uniform samples. The
kernel compares joint vectors rather than separate coordinate marginals. Low
finite-sample MMD is not a certificate of intrinsic dimension or retained factors.

Both recipes disable patch loss and use uniform subject batches
(`stratify_batches: none`). Oracle anatomy neighbourhood batches change the
distribution seen by the population objective; they are not the default for this
experiment. The training objective uses no lesion labels. Synthetic latent labels
are used for evaluation probes only.

To preview the corrected short pilot:

```bash
python scripts/launch.py experiments/encoder/conv_mlp_burden_residual_heads_auxiliary.yaml \
  --cluster local --dry-run
```

Remove `--dry-run` to run 1,000 steps, with evaluation at steps 0, 250, 500, 750 and
1,000; ordinary logging every 50 steps; and gradient diagnostics every 100 steps.
The same YAML supports the `slurm`, `slurm_bio` and `runai` backends. Generated
scripts use the suffixes `.local.sh`, `.slurm.sh`, `.slurm_bio.sh` and `.runai.sh` in
`experiments/generated/encoder/`.

For a from-scratch NVIDIA run, activate the Python environment containing PyTorch
and run this from the repository root. No image checkpoint path is needed:

```bash
RUN_ID="residual_heads_auxiliary_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p logs
nohup env PYTHONUNBUFFERED=1 ENCODER_PYTHON="$(command -v python)" CUDA_VISIBLE_DEVICES=0 \
  bash experiments/generated/encoder/conv_mlp_burden_residual_heads_auxiliary.local.sh \
  --device cuda --model-id "$RUN_ID" \
  >"logs/${RUN_ID}.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"logs/${RUN_ID}.pid"
```

After that run has finished, run the matched image-only objective control
sequentially, with its own run ID:

```bash
CONTROL_ID="residual_heads_image_control_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p logs
nohup env PYTHONUNBUFFERED=1 ENCODER_PYTHON="$(command -v python)" CUDA_VISIBLE_DEVICES=0 \
  bash experiments/generated/encoder/conv_mlp_burden_residual_heads_auxiliary.local.sh \
  --device cuda --model-id "$CONTROL_ID" --residual-head-auxiliary-weight 0 \
  >"logs/${CONTROL_ID}.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"logs/${CONTROL_ID}.pid"
```

This control keeps the same raw initialization, generator, sampler, seeds and
schedule. A zero auxiliary weight freezes the independent residual/projector
parameters and their running buffers after initial training-only calibration;
they remain available for the initial-state comparison while the image path trains
on InfoNCE. This also prevents AdamW weight decay from moving an unused branch.

The ordinary trainer retains `model_init.pt` and the latest `model.pt`; each
`dci_step*.json` is an evaluation report, not a numbered weight checkpoint. Copy a
checkpoint before further training overwrites it if a particular endpoint must be
retained, including an existing step-4,000 run.

## Protecting an existing anatomy representation

The intended first experiment initializes the complete image pathway from a
compatible anatomy checkpoint and freezes its parameters and normalization
statistics. The residual branch and joint projector remain trainable. The frozen
image encoder stays in evaluation mode when the enclosing model enters training
mode, preventing changes through BatchNorm or other running state.

Freezing requires `--residual-head-image-init` to name a real raw-encoder run directory
or checkpoint with its sibling `settings.json`. The load is strict and checks the
original architecture/settings, including activation, normalization, readout and
output dimensions. Choose a source whose image representation you have actually
measured. A path to an arbitrary run is not evidence of good anatomy recovery.
Source details are recorded with the new run's provenance.

The recipe defaults to training both paths so it resolves without an invented
checkpoint path. For the protected experiment, override it with your actual source:

```bash
python scripts/launch.py experiments/encoder/conv_mlp_burden_residual_heads_auxiliary.yaml \
  --cluster local --dry-run \
  --set residual_head_freeze_image=True \
        residual_head_image_init=/actual/compatible/anatomy/run \
        tag=conv_mlp_burden_residual_heads_auxiliary_protected_s42
```

Remove `--dry-run` to train. Use `--cluster slurm_bio`, `slurm`, or `runai` with a
source path that exists on that backend. No checkpoint path is supplied by the YAML.

The matched trainable comparison uses the same initialization, data and settings,
but allows the image pathway to move:

```bash
python scripts/launch.py experiments/encoder/conv_mlp_burden_residual_heads_auxiliary.yaml \
  --cluster local --dry-run \
  --set residual_head_freeze_image=False \
        residual_head_image_init=/actual/compatible/anatomy/run \
        tag=conv_mlp_burden_residual_heads_auxiliary_trainable_s42
```

At trainer level, the corresponding flag is `--no-residual-head-freeze-image`.
This warm-started comparison and the frozen run start from the same image code;
their step-0 evaluation is a warm-start reference, not an untrained floor. Training
from scratch is a separate comparison, obtained by leaving the source unset.

## What this preserves of the theory

The combined representation remains the identification target. Keeping the original
image pathway avoids forcing all information through a noninvertible PCA residual.
The separate heads reserve capacity; their names do not imply an identifiable
anatomy/lesion decomposition. No independence or decorrelation penalty is imposed
between the exported blocks.

The conditional argument is: if both blocks align across views and a smooth
`q = P([a, b])` reaches the required population optimum at the true shared dimension,
then identification of shared content from `q` implies that `[a, b]` retains that
content. An overcomplete exported code need not itself be an invertible coordinate
system. The auxiliary projector's dimension must be justified from the generator,
not selected from the number of anatomy names or the exported content width.

For this clean-content, single-lesion generator the count is 10: six active anatomy
scalars, burden, and the lesion's three placement variables. Changing the generator,
enabling shared fields, adding lesions, or changing which factors vary requires
revisiting `residual_head_joint_dim`. Ten is a recipe-specific choice.

Population MMD with a characteristic RBF kernel vanishes exactly when its two
distributions agree. Matching to a uniform target therefore supplies a route to
the uniform joint distribution used in the ideal argument. Finite batches, a small
projector, running gauges and optimization do not certify that optimum. The existing
raw architecture also retains its legacy activation for exact checkpoint reuse;
smoothness, image-derived support masks and other theorem assumptions need separate
justification. A frozen image pathway can make the joint optimum
unattainable if it has already discarded required shared content. None of these
experiments establishes Yao's guarantee for the trained model.

The corrected additive objective and differentiable batch alignment are practical
optimization changes. They do not make this finite experiment an implementation
of the exact population alignment-minus-entropy theorem. In particular,
batch-standardized agreement does not enforce equality of the raw code means and
scales across views, and low empirical MMD does not establish a population optimum.
The theory-facing statement remains conditional on identification from the joint
auxiliary representation and the original generative, dimensionality and
attainability assumptions.

## Check the pilot before a longer run

The corrected recipe is a diagnostic budget, not an instruction to extend training
automatically. Compare its saved evaluation reports before choosing a
long-run budget:

- **Anatomy retention:** compare each anatomy factor in the image block and the
  combined block with a matched image-only InfoNCE run. Match the raw initialization,
  generator, subject split, sampler, seeds, batch size and update count. A lower
  total loss does not compensate for consistently worse anatomy recovery.
- **Added lesion information:** compare burden and actual rendered-voxel recovery
  against each block's saved step-0 state and the matched image-only control at the
  same endpoint, in both views. Look for an improvement carried by the residual or
  combined code over both references, rather than a single favorable probe row.
  Keep lesion placement and burden distinct. Failing either comparison is evidence
  against extending this configuration for burden recovery.
- **Branch health:** `image_variance` and `residual_variance` measure variance after
  population standardization. Inspect the corresponding effective ranks, and use
  `image_raw_variance`, `residual_raw_variance` and minimum raw code SD for raw-scale
  health. Also inspect projected-code variance and rank. A falling raw scale can be
  hidden by population standardization; a growing projector norm can hide weak
  inputs. These are collapse diagnostics, not estimates of intrinsic shared
  dimension or proofs of information preservation.
- **Gradient pressure:** inspect `residual_head_gradients.jsonl`. Compare
  `image_primary/image` with `auxiliary/image` within the image parameter group,
  and `residual_alignment/residual` with `uniformity/residual` within the residual
  group. These values already include the configured objective weights. A weight
  of 0.1 does not bound auxiliary gradients at one tenth of primary gradients.
  Sustained auxiliary dominance together with anatomy loss is a reason to revise
  the weight or freeze the image path before increasing the budget.

Gradient RMS is normalized by the parameter count of its group, including unused
entries. Compare objective terms within the same parameter group; ratios between
different groups have different meanings. The diagnostic uses extra gradient
computations every 100 steps and does not replace the optimizer gradients.

Record the measured pilot result separately. Do not infer lesion recovery from
alignment, rank, MMD or gradient diagnostics alone, and do not use repeated
validation monitoring as an untouched final test.

The local paired pilot's status, protocol and outcomes are recorded in
[`out/residual_heads_auxiliary_pilot_20261103/README.md`](../out/residual_heads_auxiliary_pilot_20261103/README.md).
That driver uses 500 steps, batch size 16, full 64³ images cached as float16 then
cast to float32, and matched uniform subject subsets. It retains weights at steps
0, 250 and 500. The committed recipe instead uses 1,000 steps, batch size 32,
online float32 rendering and epoch-shuffled batches. Read the pilot as evidence
for its recorded protocol rather than as a completed run of the exact YAML.

## Reading the experiment

Evaluate the image block, residual block and their concatenation separately. For
each modality, inspect anatomy-factor, lesion burden and lesion-placement recovery,
and compare cross-view transfer. The frozen run should reproduce the source image
code exactly; check the code itself, not just an aggregate probe score. A smaller
training loss does not establish successful lesion recovery.

Use held-out subjects for these probes and compare them with the same step-0 state.
Read the separate alignment and joint-uniformity diagnostics together: good image
alignment can coexist with an uninformative residual block, and low empirical MMD
alone does not demonstrate recovery of the actual shared factors. Fit the normative
transform once on the training split and restore its saved buffers for every
held-out evaluation.

Training records `residual_heads_image_alignment`,
`residual_heads_residual_alignment`, `residual_heads_uniformity`,
`residual_heads_joint_effective_rank`, `residual_heads_joint_variance` and
`residual_heads_projected_variance`, alongside the raw and weighted auxiliary
losses (`residual_heads_auxiliary_raw`, `residual_heads_auxiliary_weighted`) and
`global_weighted`. Additional branch diagnostics include image/residual variance
after population standardization (`image_variance`, `residual_variance`), their
effective ranks, raw exported-code variance (`image_raw_variance`,
`residual_raw_variance`), `code_sd_min`, `projected_effective_rank` and
`projector_weight_norm`, all under the `residual_heads_` prefix.

The trainer also runs `eval.encoder.residual_head_recovery` for this model. Its
`views.t1` and `views.flair` reports each contain `image`, `residual` and `combined`
readouts with per-factor `r2` scores and their feature widths. These are supervised
evaluation probes: each outer training fold fits its own StandardScaler and RidgeCV;
scores use only out-of-fold predictions. There are five deterministic subject folds
(four for a four-subject smoke test), with regularization selected inside each
training fold. Constant factors have `null` scores, and cohorts smaller than four
are explicitly marked insufficient. Neither model buffers nor training random
streams are changed by the probes.

The factor names follow the generator's lesion target. When the synthetic renderer
is available, an additional `rendered_burden_voxels` target measures the lesion mask
actually rendered; it is separate from the generator's burden control. Position
targets retain their position names. Repeated validation diagnostics are not an
untouched final test set.

To regenerate the global checkpoint report and its saved-initial-state comparison:

```bash
python -m eval.protocol.score_checkpoint \
  --run-dir /actual/residual_heads/run --no-graph
```

Read `residual_heads` and `residual_heads_initial` in `score_report.json` for the
three content readouts. The latter restores `model_init.pt`, including the fitted
training residual transform and the actual image warm start.

Key controls are the protected run, the same warm start with both paths trainable,
and the original image-only checkpoint. Keep seeds, generator, batch size and
evaluation subjects fixed. The new code is enabled only by a positive
`residual_head_units`; no existing result is silently changed.
