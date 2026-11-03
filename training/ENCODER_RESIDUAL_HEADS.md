# Separate image and residual heads

`experiments/encoder/conv_mlp_burden_residual_heads.yaml` adds a second image-to-code
path to the encoder-only synthetic trainer. The purpose is to let sparse residual
evidence have its own encoder and tail readout while retaining the existing image
path. It is an experiment, with no measured burden-recovery improvement yet.
Existing recipes keep their behavior: `--residual-head-units` defaults to `0`.

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

## Objective

The content blocks are standardized separately with running population means and
scales, rather than statistics from the current batch. Let their standardized values
be `a_k` and `b_k`. The new objective is

```text
image_weight * mean_squared_alignment(a_1, a_2)
  + residual_weight * mean_squared_alignment(b_1, b_2)
  + uniformity_weight * mean_k MMD²(q_k, Uniform((0, 1)^joint_dim))
```

The projector is shared across modalities and reads the concatenated content blocks.
The MMD estimate is the biased, nonnegative multiscale RBF estimate, using uniform
target samples in the same dimension. It measures a joint distribution, not a sum
of per-coordinate penalties. Both heads duplicating the same factor cannot supply
the missing dimensions of a genuinely full-dimensional target distribution.

Global InfoNCE is still evaluated for comparison, but contributes no gradient when
the new feature is enabled. The supplied recipe also disables patch loss and uses
uniform subject batches (`stratify_batches: none`). Oracle anatomy neighbourhood
batches change the distribution seen by the population objective; they are not the
default for this experiment. The implementation does not add lesion labels to the
training objective. Synthetic latent labels are used for evaluation probes only.

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
python scripts/launch.py experiments/encoder/conv_mlp_burden_residual_heads.yaml \
  --cluster local --dry-run \
  --set residual_head_freeze_image=True \
        residual_head_image_init=/actual/compatible/anatomy/run \
        tag=conv_mlp_burden_residual_heads_protected_s42
```

Remove `--dry-run` to train. Use `--cluster slurm_bio`, `slurm`, or `runai` with a
source path that exists on that backend. No checkpoint path is supplied by the YAML.

The matched trainable comparison uses the same initialization, data and settings,
but allows the image pathway to move:

```bash
python scripts/launch.py experiments/encoder/conv_mlp_burden_residual_heads.yaml \
  --cluster local --dry-run \
  --set residual_head_freeze_image=False \
        residual_head_image_init=/actual/compatible/anatomy/run \
        tag=conv_mlp_burden_residual_heads_trainable_s42
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
`residual_heads_projected_variance`, alongside the total `residual_heads` loss.

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
