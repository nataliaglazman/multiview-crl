# Early normalization and lesion movement

This frozen-checkpoint test asks where lesion location becomes hard to recover,
and whether recomputing normalization statistics suppresses the lesion response.
It evaluates saved initialization and the trained model, in T1 and FLAIR.
It prints the actual normalization class independently of the user-supplied run
label, so a run named `gn` with LayerNorm is explicitly identified as LayerNorm.

From the repository root, replacing the paths:

```bash
python -m eval.encoder.encoder_lesion_norm_audit \
  --run "run_a=/path/to/first/run" \
  --run "run_b=/path/to/second/run" \
  --checkpoint model.pt \
  --device cuda \
  --batch-size 1 \
  --num-samples 64 \
  --norm-layers early pre_residual \
  --spatial-grid 8 \
  --out-dir "results/lesion_norm_$(date +%Y%m%d_%H%M%S)"
```

The run directories must contain `settings.json`, `model.pt`, and the original
`model_init.pt`. `--skip-initial` explicitly omits initialization; the audit never
manufactures a replacement. One `--run` is also valid. Use matching training steps
and recipes when comparing GN/LN; the audit verifies matching *evaluation images*
but cannot establish matched training history.

On M1, prefix Python with `PYTORCH_ENABLE_MPS_FALLBACK=1` and use `--device mps`.
`--device cpu` also works. Probe fitting always runs on CPU. Put `--out-dir` on
scratch when temporary disk usage matters. One layer/checkpoint is processed at
a time, and all feature banks are removed after scoring or a failed run.

The default grid of 8 reduces the cost of the early 32³ feature maps. It does
average spatial features before fitting probes. For an unpooled check, use
`--spatial-grid 0`; native feature banks can require many GB of temporary disk.
**The normalization-response diagnostics always use native tensors**, regardless
of the probe grid. A negative result with grid 8 should be checked at native
resolution before attributing it to normalization. Do not upsample a smaller map:
the requested grid must fit each selected layer.

## What the test holds fixed

The original validation cohort is split 75/25 for probe fitting/tuning. A separate
test cohort (default IDs 1000–1063) receives -/+0.5 changes to each of the three raw
lesion controls. Anatomy, style, noise draws, and the original image-normalization
affine are fixed across each pair. The renderer checks that the tissue map stays
unchanged and image differences lie within the changed lesion/blur support.

Physical movement can affect more than the nominal axis because lesion placement
is conditional on white matter. The score uses the full actual centroid movement.
Quantized/saturated controls that do not move the lesion are retained and counted,
not redrawn. All axes for one subject stay together in bootstrap draws.

The current protocol supports the conv encoder with GN/channel-only LN,
`pseudo_mri`, nine factors, spherical lesions, position targets, and
`wm_interior` placement. It does not reinterpret burden latents as positions.
Unsupported renderer configurations are rejected by the shared dataset protocol.
Labels fit diagnostic probes only. There is no encoder optimization, checkpoint
selection, or substitution of a different normalization into the trained model.

## Taps and probes

`early` selects the downsampling normalization layers: normally `layers.0.1` and
`layers.1.1`. `pre_residual` selects the main-path norm just before the residual
stack. Explicit paths relative to the encoder are also accepted, for example
`--norm-layers layers.0.1`.

Each tap provides pre/post spatial maps, their channel-wise GAP vectors, the mean
and denominator used by normalization, the two statistics together, and the
normalized map plus its statistics. Outputs are copied before in-place ReLU.
For LN, mean and scale are spatial maps; for GN, they are global group vectors.
The denominator is `sqrt(population_variance + epsilon)`.

Ridge and RBF kernel-ridge probes are standardized and fitted only on probe-fit
subjects, with hyperparameters selected on probe-tuning subjects. They predict
physical centroid coordinates for the held-out A/B endpoints. Shuffled controls
permute fit/tune labels independently; intervention labels never select a probe.

The main score is **movement skill**:

`1 - sum ||predicted displacement - actual displacement||² / sum ||actual displacement||²`

- 1: follows movement perfectly.
- 0: equivalent to predicting no movement.
- Negative: worse than predicting no movement.

Also inspect movement gain (1 ideal, 0 no following), displacement error in voxels,
and the subject-bootstrap interval. High endpoint R² alone can reflect anatomy
correlations; movement skill tests tracking with anatomy held fixed. All scores
use joint features, not one-unit identifiability. The pooled mean over axes can
hide differences; individual intervention axes are included in the CSV.

## Direct normalization response

For a tapped layer with paired inputs `h_A`, `h_B`, compare:

1. Actual response: `Norm(h_B) - Norm(h_A)`, with each input's own statistics.
2. Fixed-statistics response: apply A's mean/denominator and the learned affine
   parameters to both inputs. Their difference is `gamma * (h_B-h_A) / scale_A`.

`adaptive_to_fixed_response` is the ratio of the two native RMS responses. A small
ratio means recomputing statistics suppresses this finite feature change relative
to the fixed-statistics control. It can exceed 1 when normalization amplifies the
change. Undefined ratios are recorded as null when the denominator is zero.
**This is a layer-local calculation: the counterfactual is not passed through the
rest of the network, and response suppression is not proof of information loss.**

The input difference is also decomposed within each normalization domain into a
constant shift, a component parallel to A's centered activation vector, and an
orthogonal component. The corresponding energy fractions describe sensitivity
to shift/scale normalization. They are not fractions of lesion information;
epsilon, affine parameters, nonlinear readout, and finite movement all matter.

`sensitivity.csv` additionally reports feature response relative to variation
across probe-fit subjects, at the chosen probe grid. Identical-input replay gives
a numerical response floor. Raw pre/post RMS values have different units; do not
read their uncalibrated ratio as retention of information.

## Outputs and interpretation

- `summary.csv`: movement skill, gain, errors, conditional subject-bootstrap
  intervals, and endpoint R², by actual norm/run/checkpoint/layer/view/stage/probe.
- `normalization_response.csv`: native response and shift/scale diagnostics for
  each pair. Filter to `moved=True` for movement summaries.
- `sensitivity.csv`: reference-calibrated response and replay floor.
- `probe_parameters.csv`: fitted hyperparameters and endpoint scores, including
  shuffled controls.
- `pairs.csv`: actual movements and IDs, including pairs that did not move.
- `*_predictions.npz`: endpoint predictions and truth for later paired analysis.
- `report.json`: settings, hashes, tensor shapes/classes, cohorts, and all scores.

| Observation | Supported interpretation |
|---|---|
| Good pre-norm movement skill, poor post-norm skill; native response strongly suppressed | This normalization is a candidate bottleneck for the measured representation. Check native probes and both probe families. |
| Post-norm skill recovers when statistics are supplied | Making normalization statistics available helps this probe recover movement. |
| Small native response but good movement skill | The signal became smaller while remaining decodable. |
| Poor skill before and after this norm | Investigate earlier layers; this tap cannot establish where the failure originated. |
| Initialization tracks movements, trained checkpoint does not | Training reduced accessibility of this cue under these probes. |
| RBF tracks movements where ridge fails | The location signal remains accessible to a nonlinear probe. |

Do not infer absence from a failed finite probe. Small lesions, many spatial
dimensions, coarse grids, and limited fit subjects all affect difficulty.
Bootstrap intervals condition on the checkpoint and fitted probe and do not cover
training-seed uncertainty. These results concern synthetic lesion interventions,
not a guarantee about identifiability or real-data lesion segmentation.

```bash
python -m unittest tests.test_encoder_lesion_norm_audit tests.test_encoder_normalization_audit -v
```

## Normalization alternatives worth testing

These are proposed follow-ups, not implemented training changes or established
improvements on this dataset. Keep the loss, readout, data, steps, and seeds matched.

1. **Keep early LN statistics available.** Feed the per-voxel mean and scale into
   later layers alongside normalized features, or use a separate path around the
   first normalization. This directly tests the discarded-statistics hypothesis.
   It is inspired by [Positional Normalization, Li et al., NeurIPS 2019](https://arxiv.org/abs/1907.04312),
   which reinjects moments to preserve structure in generative models. Its results
   do not establish improved lesion identifiability. Carry statistics from the
   early layers: the existing late tap already has poor lesion recovery.
2. **No activation normalization in the first block, with weight normalization
   as an optimization alternative.** [Salimans and Kingma, NeurIPS 2016](https://arxiv.org/abs/1602.07868)
   reparameterize weight magnitude/direction rather than centering/scaling each
   input activation. That avoids explicitly quotienting out each voxel's contrast
   at that layer, but can retain scanner/style variation and needs a controlled
   training comparison. Removing every normalization at once would change too
   much; use GN elsewhere initially.
3. **Filter Response Normalization (FRN).** [Singh and Krishnan, CVPR 2020](https://arxiv.org/abs/1911.09737)
   normalize each channel using a spatial RMS, without subtracting its mean, and
   pair this with a learned-threshold activation. It is a candidate for preserving
   local deviations relative to the rest of the map. It still removes channel
   scale, couples spatial positions, and changing the activation adds a confound.
4. **Channel RMSNorm as a diagnostic control.** [Zhang and Sennrich, NeurIPS 2019](https://arxiv.org/abs/1910.07467)
   remove mean subtraction while keeping RMS rescaling. This separates centering
   from scaling, but it remains approximately invariant to positive local scaling
   when used across channels at each voxel. It is not an automatic fix for the
   proposed lesion-contrast mechanism.

First compare GN, LN, and one targeted intervention chosen from the early-layer
results. Preserving lesion contrast does not by itself solve the separate sulcal
spatial-to-GAP bottleneck; those factors still need an appropriate scalar readout.
