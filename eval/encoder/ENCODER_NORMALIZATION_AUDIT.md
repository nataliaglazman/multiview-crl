# Frozen GroupNorm versus channel-LayerNorm recovery audit

This tests whether a factor is recoverable from a spatial map but becomes hard
to recover after pooling or the learned global readout. It also probes the input,
output, and mean/scale statistics of one normalization layer. Encoder weights
and buffers stay frozen; ground-truth labels train the diagnostic probes only.

Run from the repository root. Each run directory needs `settings.json` and its
original checkpoint. Use checkpoints at the same training step, with the same
data recipe, architecture, pooling, objective, seed, and training budget where
possible. GN and LN models trained under different recipes cannot isolate the
effect of normalization, even when their evaluation images match.

```bash
GN_RUN="/path/to/groupnorm/run"
LN_RUN="/path/to/layernorm/run"
OUT="results/normalization_audit_$(date +%Y%m%d_%H%M%S)"

python -m eval.encoder.encoder_normalization_audit \
  --run "gn=$GN_RUN" \
  --run "ln=$LN_RUN" \
  --checkpoint model.pt \
  --device cuda \
  --batch-size 4 \
  --test-samples 400 \
  --out-dir "$OUT"
```

Use `--device cpu` on a laptop without a supported accelerator. For Apple
Silicon, set `PYTORCH_ENABLE_MPS_FALLBACK=1` before starting Python and use
`--device mps`; unsupported operations can fall back to CPU. The probe fitting
itself runs on CPU for every device choice.

One `--run` is also valid. To evaluate original initialization, run the same
command with `--checkpoint model_init.pt` and a new output directory. The script
never substitutes a fresh random initialization for a missing saved checkpoint.

Full native maps are the default. They can require several GB of temporary disk
space and substantial CPU time for the spatial probes. Set `OUT` to a scratch
directory if needed. Feature banks are deleted after scoring each run; only
reports and predictions remain. `--keep-features` retains the large banks.
For a cheaper preliminary pass, add `--spatial-grid 4` (which must fit both
models' maps). This averages spatial features before probing, so it is a
coarsened-map test rather than a full-native-map result. GAP and final-code
features are unchanged by this option. Kernels are built in column chunks rather
than loading complete spatial banks into RAM.

## What is probed

| Stage | Features available to the probe |
|---|---|
| `backbone_spatial` | Backbone map, flattened with spatial positions retained |
| `backbone_gap` | One spatial average per backbone channel |
| `global_content` | The actual forward's first `content_channels` outputs, usually nine |
| `global_all` | The global content and style units together |
| `lesion_branch` | Appended lesion keypoint coordinates, if present; kept separate |
| `norm_pre_spatial`, `norm_post_spatial` | Maps immediately before/after the selected normalization, before a following activation |
| `norm_pre_gap`, `norm_post_gap` | Channel averages of those same tensors |
| `norm_mean` | Mean subtracted by that normalization |
| `norm_scale` | Denominator `sqrt(population_variance + epsilon)` used by that normalization |
| `norm_statistics` | Mean and scale together |
| `norm_post_plus_statistics` | Post-normalization map plus its mean and scale |
| `actual_pool` | Actual attention-pooled features, only for attention models |

For GN, statistics are one pair per group, reducing over channels within the
group and all spatial positions. For this repository's `ChannelLayerNorm3d`,
statistics are two spatial maps, reducing over channels separately at each
voxel. This is **channel-only LayerNorm**, not LayerNorm over the whole volume.
The audit records the actual classes, groups, epsilon, layer paths, and shapes.

The default tap is the main-path normalization directly before the residual
stack. Residual blocks still lie between this tap and the final backbone map.
To investigate an earlier layer, rerun with a new output directory and, for
example, `--norm-layer layers.0.1` for the first downsampling norm. Paths are
relative to each view's encoder. Hook outputs are copied before in-place ReLU
can change them. Only the conv GN/channel-LN architectures are supported here.

The global features come from the real forward, not an average of patch-head
outputs. Separate patch heads are not used. An attention model gets a separate
`actual_pool` row; its GAP row is a diagnostic alternative, not its real readout
input. A residual-input lesion branch also reads image information through its
own path, so its row is not evidence that the backbone retained that information.

## Evaluation and outputs

- Fit each probe on 75% of the original validation cohort; tune on the other
  25%; score on a separately generated test cohort. The original validation
  count is retained because changing it can change dataset normalization.
- Fit feature and target scaling on probe-fit subjects only. Tune ridge and
  RBF kernel-ridge hyperparameters using validation loss only.
- Shuffle training and tuning labels independently for the null controls;
  score those controls against the same unshuffled test truth.
- Match image bytes, target bytes, subject IDs, and split seeds across runs.
  A mismatch fails the comparison. These checks do not establish that training
  budgets or checkpoint-selection policies were matched.
- Predict all nine latent controls, physical lesion centroid coordinates, and
  signed/absolute sulcal amplitudes. Centroids use the renderer's coordinates,
  not voxel indices or millimetres. Burden runs name latent 2 `lesion_burden` and
  label the unused controls 3–4 explicitly; their centroid is the union-of-lesions
  support centroid, not three generative location factors. New renderer settings
  the historical dataset factory cannot restore are rejected.

Files:

- `summary.csv`: one row per run/view/stage/probe/control, with all target R²s.
- `probes.csv`: detailed scores, feature dimensions, selected hyperparameters.
- `contrasts.csv`: candidate-minus-reference R² differences, both between stages
  and between each later run and the first run. Includes paired 95% bootstrap
  intervals over test subjects. These are conditional on the fitted models and
  probes, **not** uncertainty over training seeds or probe fits, and are not
  corrected for multiple comparisons.
- `NAME_predictions.npz`: held-out predictions, truth, and target names, allowing
  later analysis without copying spatial feature banks.
- `report.json`: settings, hashes, exact splits, tensor metadata, and all scores.

## Reading the result

1. **Spatial recovery is good, GAP recovery is poor:** supports a mismatch
   between the spatial representation and averaging. Check native maps and both
   probe families before concluding that pooling destroys the information.
2. **GAP recovery is good, global-content recovery is poor:** implicates the
   learned readout/content selection. Recovery in `global_all` can reveal that
   the signal is accessible in units assigned to style.
3. **Recovery drops from norm input to norm output, while the statistics predict
   the factor:** suggests the normalization statistics matter for accessibility.
   The joint post-plus-statistics control checks whether making those statistics
   available helps. Recoverability in statistics alone does not show that their
   information was absent from the normalized map.
4. **LN and GN differ already before this normalization:** earlier layers or
   learned representations differ; this tap cannot attribute that difference to
   this one normalization operation. Probe an earlier layer to locate it.

Spatial banks have many more dimensions than GAP vectors. Matched samples and
tuning make comparisons useful, but do not equalize statistical difficulty.
Returning means/scales can require a multiplicative reconstruction that finite
ridge/RBF probes do not discover. Failure is not proof of information loss.
These are joint-feature probes: a good score from nine final units does **not**
show that one scalar individually identifies a factor. The audit also does not
measure heatmap localization quality or establish theoretical identifiability.

```bash
python -m unittest tests.test_encoder_normalization_audit -v
```
