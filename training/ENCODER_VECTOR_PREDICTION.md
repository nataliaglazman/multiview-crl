# Cross-view prediction through the exported content vector

This is an experimental auxiliary objective for `training.main_conv_synthetic`. It
tests whether requiring subject-specific image evidence to pass through the exported
content vector helps retain small lesions. Existing runs keep their original behavior:
`--vector-prediction-weight` defaults to `0`.

## Why this bottleneck

The earlier cross-view prediction pilot improved lesion detection in the backbone
without producing useful global-code burden recovery. Its predictor read the full
backbone map, so it could satisfy its objective without putting the evidence into the
14 exported content units. This experiment puts the prediction loss directly on that
vector. Better reconstruction or a lower auxiliary loss alone does not establish
better lesion recovery.

## What the loss pays for

**Read this first: on the recipe below, the spatial target does not pay for burden.**
*Pay* is how much the loss falls when the decoder's input gains a piece of information.
It was measured by training this recipe's decoder on true latents instead of learned
codes, with the exact weighted loss on the recipe's 2,000 training and 400 validation
subjects, so no encoder limits it. Percentages are of the loss of a decoder given only
the anatomy factors: 0.081 for the T1 target and 0.138 for the FLAIR target, held out.

| Decoder input | Train, T1 / FLAIR | Held out, T1 / FLAIR |
|---|---|---|
| anatomy + burden | +0.35 / +0.39% | −0.94 / −1.40% |
| anatomy + burden + true lesion xyz | +0.64 / +0.92% | −1.51 / −3.12% |
| anatomy + burden + 64 Fourier features of xyz | +10.0 / +11.4% | −0.74 / −4.35% |
| the same, 1024 hidden units | +45.3 / +54.2% | −8.25 / −25.6% |
| anatomy + the true lesion map, one fitted amplitude | | +20.7 / +16.7% |

A perfect lesion description is worth 17–21% of the loss, but a global vector collects
that only by decoding *where* the lesion is. Burden alone buys nothing: a lesion can
sit anywhere, so the mean target given its burden is nearly flat. Richer position codes
lower the training loss by memorizing subjects and raise the held-out loss. So expect
this objective to reward anatomy in the vector, and expect no burden gain from it. The
matched pair is still a valid test of that expectation.

A **position-free** target does pay. Here the decoder predicts 17 quantiles of the other
view's residual over its foreground cells, which say how much is abnormal and not where.
Pay is measured the same way, anatomy + burden against anatomy alone, on that target.

| Target | Train, T1 / FLAIR | Held out, T1 / FLAIR | Burden R² from the target alone |
|---|---|---|---|
| Spatial 16³ grid (implemented) | +0.35 / +0.39% | −0.94 / −1.40% | |
| Quantiles of the template residual | +5.7 / +12.2% | +6.6 / +10.2% | 0.34 / 0.25 |
| Quantiles after removing 16 population PCs | +20.4 / +13.4% | +19.3 / +11.2% | 0.64 / 0.51 |

Scripts, logs and the flags they used are in `out/vector_prediction_pilot_20261102/`:
`oracle_pay2.py`, `position_free.py`, `position_free2.py`.

`--vector-prediction-target quantile` implements this, as the third arm. The design was
checked by the same test before it was built (`position_free2.py`, T1 / FLAIR):

| Target | Cells | Held-out pay | Burden R² | Smallest third, Spearman |
|---|---|---|---|---|
| Half-brain cells, 16 PCs from 300 subjects | 556 | +15.3 / +12.4% | 0.69 / 0.56 | 0.37 / 0.45 |
| Interior, no PCs | 304 | +14.9 / +21.2% | 0.35 / 0.37 | 0.07 / 0.13 |
| Interior, 16 PCs from 2,000 subjects | 304 | +21.9 / +21.7% | 0.66 / 0.57 | 0.40 / 0.49 |
| Interior, 16 PCs from 300 subjects | 304 | +18.6 / +22.4% | 0.62 / 0.57 | 0.42 / 0.51 |
| **Interior, 8 PCs from 300 subjects (implemented)** | 304 | +20.9 / +24.6% | 0.71 / 0.59 | 0.45 / 0.54 |

Each target's own loss is its baseline, so pay is comparable across rows only roughly.
Interior means at least 90% mean brain occupancy, which keeps the tails from being
brain-edge anatomy. The number of PCs was chosen on these 400 validation subjects, so
that row is slightly flattered. All of this is oracle evidence: it shows the target
rewards burden in the vector, not that an encoder will learn to put it there.

The fit uses only the template's training subjects and no labels: per view, the interior
cells, the 8 largest directions of their residuals, and each quantile's mean and SD. The
decoder then predicts the other view's 17 standardized quantiles with plain MSE, so
`--vector-prediction-deviation-strength` and `--vector-prediction-weight-cap` do not act.

The term is far stronger than the spatial one, because each quantile summarizes every
interior cell (`gradbalance3.py`, the same two states as the table further down):

| State | Global InfoNCE | Patch InfoNCE | Quantile, weight 1 |
|---|---|---|---|
| Initialization | 0.66 | 0.030 | 0.021 |
| Trained, decoder fitted | 0.0085 | 0.0040 | 0.085 |

## Objective and information path

Each modality has a small decoder whose **only input is its exported content vector**.
With the supplied recipe this means the first 14 units of the 18-unit global code. T1
content predicts a grid derived from FLAIR, and FLAIR content predicts a grid derived
from T1. The decoder has no access to backbone maps, patch features, image values,
masks, lesion positions, lesion masks, burden values or other generator latents. Its
output is auxiliary; the exported representation is still the content vector.

The decoder reads those units in a **running population gauge**: per view, each unit
minus its running mean over its running SD, both slow averages with no gradient. It is
one fixed affine map of the subject's own vector. The raw units cannot be read directly:
InfoNCE sees them batch-standardized, which leaves their scale free, and their
between-subject SD is about 0.008 at initialization and 0.0045 at step 2,000. Reading
them raw left the decoder a constant input and the term effectively off. Batch
statistics are not used because the oracle neighbourhood sampler's batches share their
anatomy: subtracting a batch's mean would remove the anatomy the decoder has to predict.
The gauge's spread is taken about the running mean, so it spans batches.

Encoder gradient RMS per term, weight 1, one AdamW (`gradbalance.py`, `rawscale.py`,
`gradbalance2.py`). The trained row is one checkpoint: the pilot `gap_region16` step
2,000 encoder with a decoder fitted to its codes, on this recipe's subjects.

| State | Global InfoNCE | Patch InfoNCE | Prediction, raw | Prediction, gauge |
|---|---|---|---|---|
| Initialization | 0.67 | 0.030 | 0.000019 | 0.0013 |
| Trained, decoder fitted | 0.0085 | 0.0040 | not measured | 0.0063 |

So the term is about 520× below the global term at initialization and level with it
once the global loss has converged. This is why the term shares the contrastive AdamW,
though the cell-code pilot gave its predictor its own: that pilot's reason was dilution,
which is gone at the trained state. Between those two states it was not measured.

The target is the other view's image averaged onto `--vector-prediction-grid`,
standardized over its foreground cells, minus that modality's fixed population
template. Cell means divide pooled foreground intensity by pooled foreground
occupancy, so background values do not dilute partially occupied boundary cells.
The per-image z-score uses only cells with at least 50% foreground occupancy.
Templates are estimated from the first
`min(vector_prediction_template_subjects, len(training_dataset))` **training
subjects only**, using the same image-derived target transform. The template at
each position is the foreground-occupancy-weighted mean over qualifying training
cells; never-observed positions remain zero. Templates stay fixed during
optimization. Validation and test subjects do not fit the templates. This
subtraction removes the common image template from the prediction task; it does
not remove all subject anatomy.

The loss is foreground-normalized MSE, with detached, capped weights derived from
the target's deviation from its template. `--vector-prediction-deviation-strength`
controls the extra emphasis on deviations, and `--vector-prediction-weight-cap`
bounds it before weights are normalized to mean one over each subject's kept
cells. Each subject and prediction direction contributes equally. Image targets,
templates, foreground masks and deviation weights are detached: gradients update
the decoders and input content vectors only. Foreground masks select and normalize
the loss; they are never decoder inputs.

The total objective in this recipe is global InfoNCE + patch InfoNCE +
`vector_prediction_weight` × vector prediction. The auxiliary term uses paired image
observations and foreground support, with no lesion labels. The existing **oracle
anatomy sampler does use ground-truth anatomy latents** to choose neighbourhoods;
that is inherited from the control and is separate from the prediction objective.

The implementation supports paired cross-modal inputs, a fixed content mask, and
Conv MLP or dense readouts. The main recipe uses the Conv MLP. The dense readout's
existing incompatibility with patch loss still applies: a separate dense experiment
must disable patch training and match its own control.

## Matched recipes

- `experiments/encoder/conv_mlp_burden_nativegrid.yaml`: weight-zero control.
- `experiments/encoder/conv_mlp_burden_vector_prediction.yaml`: same recipe, prediction
  weight `1`. This is an initial experimental weight, not a tuned or validated choice.
- `experiments/encoder/conv_mlp_burden_vector_quantiles.yaml`: the third arm, the
  intervention with `vector_prediction_target: quantile`.

Both inherit `encoder/synthetic_defaults` and retain the supplied run's baseline:

- Separate T1/FLAIR Conv encoders, GroupNorm, two residual layers, stride four, 64
  backbone channels and the GAP → MLP readout with hidden width 100.
- 18 output units, first 14 fixed as content, standardized global code, global
  cosine InfoNCE at temperature 0.1, cross-view negatives, no contrastive projector.
- Unit-weight cosine patch InfoNCE on the native 16³ grid, standardized patch codes,
  shared global/patch readout, foreground threshold 0.05.
- Oracle anatomy neighbourhood batches with 50 neighbours and balance 0.
- 64³ paired images, 2,000 train/400 validation subjects, independent Gaussian
  content, clean content, fixed-reference normalization, styled lesions, identifiable
  ventricles, sulcal atrophy, temporal width 0.18.
- One WM-contained burden lesion, maximum radius 0.1 and T1 value 0.0, with no
  minimum-radius floor. Tiny lesions remain part of the data distribution.
- Batch 32, AdamW at 1e-4, clipping at 2, 20,000 updates, validation every 1,000,
  initial floor evaluation and final-step selection. Data seed stays 42; model and
  loader seeds follow the run seed.

The intervention adds a grid of 16³, decoder hidden width 128, 300 training template
subjects, deviation strength 1 and weight cap 4. The two YAMLs differ only in their
tag and the prediction weight. No old recipe or default is changed.

`vector_prediction_template.json` records template source, training indices,
subjects per view, grid, foreground threshold, position coverage and template
hash. The template and the gauge are also saved with the model. A warm start that
already has a fitted template restores it rather than fitting validation or test
data. The gauge moves only in training; before the first training step, the step-0
floor falls back on its batch's statistics.

Training logs report `vector_prediction`, its weighted contribution
`vector_prediction_weighted`, both directions (`vector_prediction_t1_to_flair`,
`vector_prediction_flair_to_t1`), and the corresponding unweighted MSE diagnostics.
They also record the normalized weight range and foreground fraction. Check both
weighted and unweighted prediction loss when changing the deviation weighting.

## Launch and compare

From the repository root, preview the resolved command without training:

```bash
python scripts/launch.py experiments/encoder/conv_mlp_burden_vector_prediction.yaml \
  --cluster local --dry-run
```

Launch the matched pair on the same backend and hardware:

```bash
python scripts/launch.py experiments/encoder/conv_mlp_burden_nativegrid.yaml --cluster slurm_bio
python scripts/launch.py experiments/encoder/conv_mlp_burden_vector_prediction.yaml --cluster slurm_bio
```

Replace `slurm_bio` with `slurm`, `runai` or `local` as appropriate. For local Apple
Silicon training, add `--set device=mps` to both commands. Training requires the
repository's existing environment; the launcher does not install dependencies.

The intervention YAML can also produce its own exact weight-zero control. Give it a
new tag so it cannot overwrite or reuse the intervention run:

```bash
python scripts/launch.py experiments/encoder/conv_mlp_burden_vector_prediction.yaml \
  --cluster slurm_bio \
  --set vector_prediction_weight=0 'tag=conv_mlp_burden_vector_prediction_w0_s{seed}'
```

For a Python-less submit host, generate a script on a machine with Python, sync the
repository and the generated script, then preview before submission:

```bash
python scripts/launch.py experiments/encoder/conv_mlp_burden_vector_prediction.yaml \
  --cluster runai --generate
bash experiments/generated/encoder/conv_mlp_burden_vector_prediction.runai.sh --dry-run
```

Repeat the matched pair with additional seeds, keeping the data seed and all other
settings equal. For example, add `--set seed=142` to both YAML launch commands; tags
and model/loader seeds follow that override.

## What counts as success

`eval.encoder.burden_recovery` makes the comparison. Give it the control first:

```bash
python -m eval.encoder.burden_recovery \
  --run-dirs results/encoder/runs/conv_mlp_burden_nativegrid_s42 \
             results/encoder/runs/conv_mlp_burden_vector_prediction_s42 \
  --split test --num-samples 1000 --out burden_s42.json
```

It reads burden from the final vector, the regional codes and the backbone map, per
view, with a linear probe, an RBF kernel probe and, for the two spatial readouts, a
position-free probe: each channel's quantiles over the brain's interior cells, then
ridge. The same probe on the raw image is printed as its positive control. Scores are
held out, on fresh test subjects, over all subjects and over lesions below 10, 20 and
30 voxels. Each run is scored against its own `model_init.pt`, and the intervention
minus the control gets a paired bootstrap interval. It refuses a pair unless the
settings outside the prediction options, the step, the batch-order and input hashes and
the initial encoder all match, so a mismatch cannot pass unnoticed. Inside a size bin
read Spearman: the bins restrict the range, which drags R² down in every arm.

The probe and its cell set matter. On the cell-code pilot's `predcode_sepadam` step
4,000 checkpoint (T1 lesion 0.4, one seed, 400 validation subjects, no control arm),
interior cells give the map Spearman 0.53 with burden in T1 (R² 0.24, untrained 0.02)
and 0.39 in FLAIR (untrained 0.42), against 0.39 and 0.63 for the raw image. Adding
brain-edge cells takes every one of these to about 0, and takes the raw FLAIR image to
0.24, because edge anatomy owns the tails. A small network per cell, summed over the
brain and trained on 320 subjects, failed the positive control and was removed. The
smallest third of lesions stays near 0.04–0.08, so "the map holds burden" is true of
larger lesions only. Logs: `set_control3.log` in the pilot folder.

A 200-step run of the matched pair on this recipe, scored on 400 test subjects
(`burden_e2e.log`, one seed), shows where the burden sits. Spearman, T1 / FLAIR:

| Readout and probe | Untrained | Control | Intervention | Raw image |
|---|---|---|---|---|
| Map, quantile | 0.31 / 0.32 | 0.72 / 0.63 | 0.63 / 0.54 | 0.66 / 0.63 |
| Regional codes, quantile | 0.13 / −0.07 | 0.64 / 0.65 | 0.61 / 0.54 | |
| Final vector, ridge | −0.09 / −0.02 | −0.05 / 0.01 | 0.02 / −0.01 | |
| Final vector, kernel | 0.01 / 0.12 | −0.02 / −0.01 | 0.00 / 0.05 | |

So the 14-unit regional codes hold what the map holds, and the global vector holds none
of it. This run checks the wiring and proves the arms match: its held-out prediction
loss only reaches about the template's. It says nothing yet about the trained
intervention, including whether its small deficit on the map and regional codes lasts.

Success is the **final vector** gaining burden over the control and over its own floor,
in both views and across seeds. A gain in only the map or the regional codes is a
different result: the plan's aim is that the exported vector cannot discard the lesion.
Check that other rendered factors stay useful (`dci_step*.json`). `unused_3` and
`unused_4` are unrendered in burden mode and should not be treated as failed factors.

Repeat across model seeds before concluding that an effect is reliable. Validation
trajectories guide selection and are not an independent test.

Backbone detection, cell-code detection, final-vector burden recovery and reduced
prediction loss answer different questions. `eval.encoder.regional_lesion_detection`
scores detection on one-lesion burden runs too; its aggregate AUROC can conceal failure
on the smallest lesions.

The target is still a coarse image grid. Small lesions can be diluted by rendering
noise, blur and cell averaging; the weighted loss can still be dominated by anatomy
and acquisition residuals. Template subtraction does not identify lesions, and the
capped deviation weighting has not established that it focuses on lesions.

Anatomy-conditioned residual targets are **not implemented**. Changing the minimum lesion
radius changes the size task; keep the original small-lesion distribution for the primary
test.

## Local-patch distribution target: experimental fourth arm

`experiments/encoder/conv_mlp_burden_vector_patch_distribution.yaml` adds
`--vector-prediction-target patch_distribution`. It retains the quantile arm's
encoder, anatomy sampler, batch size 32, 20,000-step schedule, one-lesion burden
prior and contrastive objectives. The encoder map and contrastive patch grid stay
at **16³**. Only the image-derived prediction target uses a **32³** grid, so this
change does not add a high-resolution feature path into the exported vector.

The target represents a distribution of local residual patches rather than their
absolute positions. Its template, eight population PCA directions, brain-interior
support and target standardization are fitted from the first 300 training
subjects only, or all training subjects when fewer are available. After removing
the fitted population directions, it extracts size-three residual patches at
scales 1 and 2. Only patches fully inside brain support contribute. Eight fixed
projection directions per scale, fixed with local seed 1729, summarize each patch;
one direction retains its central voxel. Nineteen quantiles of each projection,
including the minimum and maximum around the existing 17 quantiles, give a
**304-coordinate target per view**. The endpoints keep a one-voxel event from
falling beyond every sampled quantile on a fine grid, but may amplify noise. The
existing single-voxel quantile target retains its original 17 quantiles. The
decoder predicts the other view's
standardized target using the same 14 exported content units and population gauge
as the other prediction arms.

The loss is **tail-weighted standardized projected-quantile MSE**. Comparing
projected distributions is inspired by sliced Wasserstein-2 distance; finite
projections, selected quantiles, training-based coordinate standardization and
tail weighting make this an approximation rather than exact optimal transport.
There are no lesion labels, lesion positions, adaptive lesion crops or spatial
features supplied to the decoder. Foreground support and all fitted transforms
belong to the detached target path. The inherited oracle anatomy sampler remains
separate from this label-free auxiliary target.

Scales mean nonoverlapping block averaging by 1 or 2, followed by valid 3³
convolution with each fixed projection. Patch support intersects the fixed
population interior and the subject's anatomical brain mask. An explicit mask is
required: treating `image != 0` as foreground would exclude the zero-valued T1
lesions. Decoded quantiles are sorted in the target modality's physical residual
units before standardization, so each predicted curve is monotone. Checkpoints
record the projection filters and descriptor settings and reject a different
scale order or projection seed even when the output dimensions match.

Local neighborhoods can distinguish patterns that a distribution of single-voxel
intensities treats as equivalent. That is a design motivation, **not evidence of
successful small-lesion learning**. No lesion-recovery result is established for
this arm. Its recipe weight **0.1** is an exploratory starting value, not a
validated choice; the trainer's general default prediction weight remains zero.

Preview the new full recipe without training:

```bash
python scripts/launch.py experiments/encoder/conv_mlp_burden_vector_patch_distribution.yaml \
  --cluster slurm_bio --dry-run
```

Start with a matched 2,000-step pilot, changing only the auxiliary weight and run
tag. These commands launch training when deliberately run; remove nothing from
the matched encoder, data or contrastive recipe:

```bash
python scripts/launch.py experiments/encoder/conv_mlp_burden_vector_patch_distribution.yaml \
  --cluster slurm_bio --set train_steps=2000 vector_prediction_weight=0 \
  'tag=conv_mlp_burden_patch_distribution_pilot_w0_s{seed}'

python scripts/launch.py experiments/encoder/conv_mlp_burden_vector_patch_distribution.yaml \
  --cluster slurm_bio --set train_steps=2000 vector_prediction_weight=0.1 \
  'tag=conv_mlp_burden_patch_distribution_pilot_w01_s{seed}'
```

Use `--dry-run` on either command to preview it. Change the backend equally in
both arms; for a local MPS run, add `device=mps` to each command's `--set` values.
For another paired seed, add `seed=142` to both commands. Data seed stays 42.

Compare the completed pilots on fresh test subjects:

```bash
python -m eval.encoder.burden_recovery \
  --run-dirs results/encoder/runs/conv_mlp_burden_patch_distribution_pilot_w0_s42 \
             results/encoder/runs/conv_mlp_burden_patch_distribution_pilot_w01_s42 \
  --split test --num-samples 1000 --out burden_patch_distribution_pilot_s42.json
```

Judge the final content vector against the weight-zero control and each run's
initial floor, separately by modality and lesion-size bin. Inspect paired
uncertainty intervals, Spearman within the smallest bins, and preservation of
other rendered factors. Improved prediction loss or backbone detection alone
does not meet the exported-vector objective. A 2,000-step pilot is preliminary;
repeat useful findings across seeds and the full schedule.

The 32³ target and local patch extraction increase target construction, template
and PCA fitting costs. PCA may remove lesion-related variation along with common
anatomy, and a fit from 300 subjects may generalize imperfectly. Boundary support,
noise, blur, finite projections and extreme quantiles can still obscure tiny
lesions. Pooling patch distributions discards absolute location and can also
discard distinctions between arrangements of similar patches. The higher target
resolution does not recover information already lost by the encoder. These are
limitations to measure with matched controls, not reasons to claim that the new
objective succeeds or fails before it is tested.
