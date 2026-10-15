# Batch standardization and residual anatomy

This frozen-checkpoint probe tests whether `--standardize-code` increases the
global InfoNCE objective's reliance on anatomy still varying inside a cluster.
It does not train a model or change a checkpoint.

```bash
python -m eval.encoder.code_standardization_probe \
  --run-dir "$RUN" --checkpoints model_init.pt model.pt \
  --batches 8 --shuffles 8 --device cuda
```

`--checkpoints` defaults to `model.pt`. Use `--device cpu` on a CPU host.
The default factors include brain size, cortical thickness, all other active
anatomy factors, and lesion burden or position. `--factors brain_size
cortical_thickness lesion_burden` can restrict the printed factors; partial
associations still adjust for every other active factor. Check the run's factor
names before choosing them. Outputs default to `$RUN/standardization_probe/`;
use `--out-dir` for another location. Nonempty output directories require
`--overwrite`.

The dataset is restored with the trainer's factory and options. Stratified runs
require their original `stratification.json`: the probe uses those saved labels
and the trainer's cluster batch sampler, including its dropped small clusters
and remainders. Unstratified runs use the trainer's seeded shuffled loader. The
first requested batches of that schedule are replayed at each checkpoint; these
are not claimed to be the historical batches at the checkpoint's training step.
Subject indices and image hashes must agree between checkpoints.

Training mode is the default for the encoder forward. Model buffers, module modes,
and parameter gradients are preserved. `--model-mode eval` explicitly uses the
checkpoint's evaluation behavior instead. The projection head and content selection
are shared with training. Both standardization toggles are checked against the
trainer's actual global InfoNCE on every batch. Learned masks, non-InfoNCE losses,
real data, and within-modality augmented pairing are rejected. Other training loss
terms are outside this probe's scope.

## Three conditions

- **raw**: the selected, projected content code, before batch normalization.
- **centered**: subtract each view's batch mean, without changing dimension scales.
- **standardized**: the exact training transform, `(code - mean) / (std + 1e-6)`;
  PyTorch's sample standard deviation, independently for each view and dimension.

All conditions then use the training cosine similarity, temperature, and negative
set. Raw-to-centered isolates removal of offsets. Centered-to-standardized isolates
rescaling. Increasing raw feature amplitude is not sufficient evidence: cosine
normalization can cancel that change. The probe measures the actual loss as well.

## Factor removal

Within a batch, center factor `y` and form its unit subject-space direction `q`.
For each view's pre-standardization code matrix `H`, the associated linear
component is `E = q q^T H`. Remove `E` in both views and reevaluate InfoNCE.

The **marginal** direction uses `y` directly. The **unique** direction first
projects `y` off all other active factor columns. This avoids crediting an
anatomy proxy to a correlated lesion variable in the linear model. Collinear
factors or insufficient residual degrees of freedom produce unavailable rows.
These are in-batch linear decompositions, not held-out decoding scores, biological
interventions, or evidence of nonlinear conditional independence.

`delta_loss_recomputed` is the loss after removal minus the original loss, with
batch statistics recomputed. `delta_loss_fixed_stats` holds the original batch
mean and SD fixed instead. Comparing them exposes effects caused by changing
normalization statistics during removal. A positive delta means the component
helps the frozen code solve instance discrimination. `first_order_removal` is
`-gradient(loss, H) dot E`, including differentiation through standardization.
It describes a small move, not a full parameter update; a full removal can have
an effect even when this derivative is zero.

Shuffled-factor controls project onto shuffled subject directions using the same
codes, controls, and permutations for all conditions/checkpoints. The partial null
shuffles the residual and projects it off the controls again. `excess_*` subtracts
the mean shuffled-control effect. These are descriptive controls, not calibrated
p-values. Marginal/unique components do not form an additive factor decomposition.

## Read the outputs

- `report.json`: checkpoint/settings hashes, exact batches, loss parity, summaries.
- `baselines.csv`: loss, cross-view retrieval, cosine margin, pre-transform gradient norm.
- `channels.csv`: per-view mean, SD, and multiplier `1 / (SD + 1e-6)`.
- `batch_spread.csv`: each factor's within-batch SD relative to the training-bank SD.
- `factors.csv`: associated signal amplitude and fraction of centered code variance,
  marginal/unique removals, derivatives, shuffled controls; one row per batch.
- `scaling_effect.csv`: paired standardized-minus-centered differences in
  null-adjusted removal effects. Positive means greater reliance after scaling.
- `codes.npz`: compact projected codes, factor values, and subject indices for replay;
  `checkpoint_0`, `checkpoint_1`, etc. follow the requested checkpoint order.

Evidence for the proposed mechanism is residual anatomy variation in the batches,
an increased share of loss-facing code variance associated with that anatomy, and
a consistently larger null-adjusted loss penalty when its component is removed
after standardization than after centering alone. Compare anatomy against lesions
and trained against initial checkpoints; standardization may amplify both.
Means and SDs describe the sampled batches, not independent-sample confidence
intervals. Repeated epochs can revisit subjects.

Per-dimension affine scaling does not create information or improve an ideal
scale-invariant linear probe within a batch. This diagnostic tests loss geometry
and dependence at fixed weights. It cannot establish whether training without
standardization would ultimately learn more lesion information; that requires
a matched training experiment. It can support or refute the simpler explanation
that the current objective becomes more dependent on residual anatomy when
standardization is applied.
