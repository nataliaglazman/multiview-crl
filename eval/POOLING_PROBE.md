# Frozen pooling comparison

Run once on each contrastive/baseline checkpoint. No encoder optimization, decoder
passes, or checkpoint writes. Synthetic labels fit diagnostic readouts only;
they do not select regions, channels, or pooling weights.

```bash
python -m eval.pooling_probe \
  --run-dir results/synthetic/synthetic-clean-content-causal-ident-vent-12-4-3 \
  --num-samples 512 --batch-size 8 \
  --grids 1 8 --tail-fraction 0.25 --causal iid
```

For a quick smoke run, use `--num-samples 128`. Keep the same arguments for the
baseline run, changing only `--run-dir`. Dataset settings come from each run;
matching split seeds does **not** make differently configured renderers identical.
`--causal iid` removes SCM/hierarchical factor correlations for this diagnostic;
this may be outside the training distribution. Repeat with `--causal match` to
retain the run's distribution, where a factor can be predicted through its parents.

## What is compared

Each batch is encoded once. The script taps native encoder maps **before
content_norm**, at the stage used by the model's contrastive pooling, applies the
model's configured latent mask if any, and verifies GAP against the actual forward
return. It uses the actual stable content-channel selection. No projector,
quantization, or reconstruction is evaluated. `--block style` tests style instead.

At each requested grid, every channel/region produces:

| Method | Descriptor |
|---|---|
| mean | Mean of all sites |
| max | Largest activation |
| upper | Mean of the largest `ceil(fraction * sites)` activations |
| lower | Mean of the smallest `ceil(fraction * sites)` activations |
| mean_tails | Concatenated mean, upper and lower descriptors |

Regions stay in a fixed spatial order. Grid 1 is global pooling; grid 8 retains
512 locations. For a 16³ native map and grid 8, each region has eight sites:
fraction 0.25 selects two, whereas 0.05 selects one and is exactly max pooling.
Non-divisible grids fail explicitly instead of using overlapping adaptive bins.
All regions are included; the training loss's foreground-patch filter is not
applied. Thus grid-1 mean is ordinary whole-map GAP, not necessarily the training
loss's foreground-filtered GAP. No anatomical region is privileged by this test.

Means/tails are computed **before** spatial averaging. Signed activation tails
have no predetermined relationship to bright/dark input tissue. Tied activations
produce arbitrary top-k site selections; inspect the saved feature values too.

## Probes and controls

- One disjoint 60/20/20 subject split is shared by all modalities/methods.
- Ridge and RBF kernel ridge use train-only feature and target standardization,
  no PCA, and validation-only selection of regularization/bandwidth per factor.
  They are reported separately; test scores never choose the probe.
- A shuffled-label control independently permutes train and validation labels and
  predicts the original test targets. Even null fitting never uses test labels.
- Targets: ventricle size, lesion x/y/z, brain size and cortical thickness. Lesion
  coordinates are latent factors, **not** a lesion-presence/size metric or physical
  centroid. Sphere lesions only; field lesions are rejected.
- Equal-width max/upper/lower comparisons distinguish pooling changes from the
  combined descriptor's threefold expansion. That expansion is reported explicitly.
- Paired bootstrap intervals (500 resamples) quantify changes versus mean on the
  same test subjects, conditional on these fitted probes. They do not include
  training/split uncertainty or adjust for multiple comparisons.

A strong FLAIR improvement for tails at the **same grid** supports a readout
limitation of mean pooling on this checkpoint. Compare grid 1 and grid 8 separately:
a regional improvement could come from preserving position. If only T1 improves,
the cross-view imbalance remains. Negative R² is allowed; it means worse than the
test-set mean predictor. Neither a weak probe nor a stronger tail probe establishes
absence of information or proves a new training objective will move it into content.

## Outputs

A new `pooling_probe_TIMESTAMP` directory is created inside the run, or at the
non-existing path supplied by `--output-dir`:

- `scores.csv`: individual held-out factor R², shuffled controls, feature counts,
  validation-selected hyperparameters, paired deltas and confidence intervals.
- `summary.json`: scores, all run/diagnostic settings, exact native sizes, selected
  channels, actual tail counts and interpretation limits.
- `predictions.npz`: held-out predictions, truth and all subject split indices.
- `examples.npz`: first four subjects' normalized inputs, native selected feature
  maps, and upper/lower selection density per voxel (averaged across channels).
  Keys such as `flair_g8_upper_selection` show where extremes came from without
  using anatomy labels. Densities are at native resolution, not input resolution.
  Set `--examples 0` to skip this file.

Large pooled feature matrices use temporary disk storage and are removed afterward.
Probe kernels use O(N²) RAM; the dense kernel solves grow roughly as O(N³).
Start at 512 subjects, rather than immediately scaling to thousands. Temporary
storage uses the system temp directory (`TMPDIR` can choose a larger volume).

```bash
python -m unittest discover -s tests -p 'test_pooling_probe.py' -v
```
