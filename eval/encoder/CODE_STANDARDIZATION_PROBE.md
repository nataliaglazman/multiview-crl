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
and remainders, or for `--stratify-sampler neighbourhood` the saved neighbour
pools and the trainer's neighbourhood sampler. Unstratified runs use the trainer's seeded shuffled loader. The
first requested batches of that schedule are replayed at each checkpoint; these
are not claimed to be the historical batches at the checkpoint's training step.
Subject indices and image hashes must agree between checkpoints.

Training mode is the default for the encoder forward. Model buffers, module modes,
and parameter gradients are preserved. `--model-mode eval` explicitly uses the
checkpoint's evaluation behavior instead. The projection head and content selection
are shared with training. All three code transformations are checked against the
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

## Matched training follow-up

Compare the existing standardized run with two fresh runs, retaining the saved
data generator, lesion target, clustering, model/loader seeds, learning rate,
temperature, batch size, and step budget:

| Arm | Trainer flags | What it tests |
| --- | --- | --- |
| Standardized | `--standardize-code --no-center-code` | Existing reference |
| Centered | `--no-standardize-code --center-code` | Remove SD scaling while retaining centering |
| Raw | `--no-standardize-code --no-center-code` | Remove both transforms |

From the repository root, after syncing the updated trainer and launcher to the
training machine:

```bash
RUN=results/encoder/runs/conv_mlp_burden_oracle_k60_s42

python -m scripts.run_encoder_code_ablation \
  --reference-run "$RUN" --modes centered raw --device cuda --dry-run

nohup python -u -m scripts.run_encoder_code_ablation \
  --reference-run "$RUN" --modes centered raw --device cuda \
  > code_ablation_s42.log 2>&1 &
```

This runs sequentially, starting from fresh initialization. It writes siblings
`${RUN}_code_centered` and `${RUN}_code_raw`, refuses existing outputs, and stops
on failure. It reads `settings.json`, not the current experiment YAML or the
folder name, and prints the saved lesion target first. A folder named `burden`
can therefore still produce position-target runs if that is what it actually
trained. Unknown saved settings are refused rather than silently omitted.

If the original run was trained with different code or dependencies, include
`--modes standardized centered raw` to produce a fresh standardized control too.
Saved settings/seeds cannot reproduce older generator or model implementations;
compare `stratification.json`, initialization tensors, and training input hashes
when reusing an older control.

The primary outcome is held-out per-factor lesion ridge R² relative to the
untrained floor, in both encoders, at the same training step. The trainer already
logs these in `dci_step*.json` when using the synthetic recipe. Compare brain size,
cortical thickness, and other anatomy factors as secondary outcomes. A lower
training loss alone does not establish improved lesion recovery; the objectives
have different geometries. Loss of anatomy without a lesion gain is not the
desired result either.

Rerun the frozen-checkpoint probe with automatic factor selection so it uses the
actual target. A separate output directory preserves the first probe:

```bash
for DIR in "$RUN" "${RUN}_code_centered" "${RUN}_code_raw"; do
  python -m eval.encoder.code_standardization_probe \
    --run-dir "$DIR" --checkpoints model_init.pt model.pt \
    --batches 16 --shuffles 16 --device cuda \
    --out-dir "$DIR/standardization_probe_followup" || break
done
```

For replication, use all three modes and `--seed 43` (then `--seed 44`). This
changes the model seed and loader order together, keeps the saved data seed, and
adds `_seed43` or `_seed44` to each output name. Start with the seed-42 comparison
before spending compute on more K values or other model changes.
