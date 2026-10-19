# Frozen VQ-VAE spatial bottleneck

This experiment learns a continuous vector from the full spatial content map of
a saved VQ-VAE. A shared convolution/flatten/MLP readout produces `z`; a shared
map decoder receives **only z**. Neither the source encoder nor its codebooks
are optimized. No anatomical labels or interventions enter training or selection.

The objective is

    reconstruction_weight * mean(((R(z) - H) / scale)^2)
  + alignment_weight      * mean((z_view1 - z_view2)^2)
  + variance_weight       * mean(relu(1 - sqrt(Var_subject(z) + 1e-4)))

Variance is computed separately in each view. There is no decorrelation penalty
on the code: anatomical variables can be correlated. This loss is a practical
preservation objective, not an implementation of a causal identifiability theorem.

## Run

From the repository root, in the normal PyTorch/MONAI environment:

~~~sh
python -m training.spatial_bottleneck_experiment \
  --run-dir /absolute/path/to/synthetic-vq-run \
  --checkpoint vqvae_model.pt \
  --out-dir /absolute/path/to/new-bottleneck-run \
  --level 0 --latent-dim 16 --width 8 --hidden 128 \
  --train-samples 256 --val-samples 64 --test-samples 64 \
  --batch-size 16 --steps 2000 --device cuda
~~~

The source run needs its saved `settings.json` and checkpoint. The output
directory must be new. Repeat with different `--latent-dim` values and separate
output directories for a capacity sweep. Choose dimensions using validation;
do not select a dimension from its test results.

The CLI currently supports synthetic checkpoints. It reproduces saved generator
options and uses distinct train/validation/test sample seeds with the original
SCM seed. The readout's training subjects can overlap the source encoder's
training subjects; validation/test are held out from readout optimization.
For ADNI, `extract_content_maps` accepts the existing paired-volume DataLoader
contract; the caller must provide subject-disjoint splits and fixed preprocessing.
ADNI dataset/launcher integration is not part of this draft.

## Representation and normalization

- Maps are continuous, native-resolution outputs of `VQVAE.forward`, after
  content normalization where present and before codebook projection/quantization.
  These differ from the pre-content-normalization features used by the trainer's
  pooled contrastive path. The report records the extraction point.
- Only the selected content channels are retained, with separate saved indices
  for each view where the source has per-view selectors. Batch-dependent
  `onthefly` masks are rejected. Fixed and deterministic evaluation-time learned
  masks are supported. Both views must have the same nonzero content width.
- A common spatial template and per-channel residual scale are fitted on the
  **training subjects only**, across both views. The readout sees standardized
  deviations; the decoder restores the template and scale. This fixed template
  is not a subject-dependent bypass. Validation/test never refit these buffers.
- There is no spatial averaging in the readout. Its fully connected layers grow
  with the native map size; `--width` and `--hidden` control parameter cost.
  `--max-map-gib` caps the combined in-memory map banks (default 1 GiB), not the
  model/optimizer memory. Small extraction batches reduce device memory usage.
- The strict checkpoint check permits unused contrastive projector weights but
  rejects other architecture mismatches. Runs with `split_encoder_norm` are
  rejected until the shared VQ checkpoint loader restores that option.

## Outputs

- `bottleneck.pt`: validation-selected readout/decoder, normalization buffers,
  architecture, source checkpoint hash and content-channel indices. It excludes
  the frozen source model. Reload with `load_bottleneck(path, device)`; `encode`
  accepts unnormalized selected maps and `decode` returns maps in original units.
- `report.json`: configuration, split identities/seeds, validation history,
  selected step (0 means initialization beat trained candidates), and test metrics.
- `test_outputs.pt`: paired small vectors, raw GAP vectors for the same subjects,
  subject IDs and a spatial map of normalized reconstruction error.

Metrics include reconstruction in raw and normalized units, per-view errors,
error relative to the fixed training-template predictor, matched/mismatched
cross-view code errors, and code standard deviations across subjects. The
original map disagreement is also reported: a frozen encoder may leave modality
differences that conflict with vector agreement and exact map reconstruction.
Only validation selects checkpoints; test metrics are computed after selection.

Inspect spatial errors as well as averages: low aggregate error can conceal a
lost lesion. These outputs assess map compression, not individual causal-factor
recovery; `eval.protocol.spatial_bottleneck_eval` (below) does that. A dimension
small enough to represent selected scalar factors may be insufficient for
additional shared spatial fields in the generator.

## Factor recovery

~~~sh
python -m eval.protocol.spatial_bottleneck_eval \
  --bottleneck-dirs /path/to/bottleneck_d16 /path/to/bottleneck_d16_s1 /path/to/bottleneck_d32 \
  --out-dir /path/to/new-eval-dir --test-samples 512 --device cuda
~~~

This regenerates the bottleneck's own subjects from the source run's settings,
collects their factor labels and asks, per factor, how well each representation
predicts it on held-out test subjects:

| Representation | Role |
|---|---|
| `map` | full content map (the bottleneck's normalisation): what was available before compression |
| `patch2x2x2` | channel means on a coarse grid: the existing local readout |
| `gap` | channel means: the existing global readout |
| `pca{d}` | top-d principal components of the map, at each bottleneck's size: linear compression |
| `z_init:RUN` | the readout at its initialisation (same seed and normalisation): the floor |
| `z:RUN` | the trained code |

- Standardisation, PCA and probe weights are fitted on training subjects only.
  Each factor's ridge penalty / RBF width is chosen by validation R², and test
  labels are used once. Both a linear and an RBF kernel probe are reported;
  the full map is probed exactly through its Gram matrix.
- Before scoring, the regenerated test maps must reproduce the saved test codes
  and GAP vectors, and the source checkpoint and settings must match their
  recorded hashes; otherwise the script stops. Use `--run-dir` if the source run
  moved.
- `--test-samples` / `--val-samples` add further held-out subjects from the same
  split seeds. The readout is frozen, so this only costs rendering, and with
  64 test subjects per-factor intervals are wide. Under `fixed_reference`
  normalisation a split trained with fewer than 64 subjects cannot be extended
  (the parity check fails and says so).
- Several bottleneck directories trained on the same source and splits are
  evaluated together. Different seeds at one size show seed variability;
  different sizes add a "latent size chosen by validation" line.
- Targets, in four blocks:
  - **content:** the content latents.
  - **lesion centre:** `centroid_x/y/z`, where the lesion actually sits
    (re-rendered by `eval.encoder.encoder_target_protocol.sample_targets`).
    Under `wm_interior` placement the lesion latents are quantiles through each
    subject's own white matter, so a probe must redo a subject-specific count
    to read them; read the centre rows for whether a representation knows
    where the lesion is. The block is omitted, with the reason printed, when a
    subject has no rendered lesion or the generator has no single sphere lesion.
  - **shared style:** a style factor identical in both views (gain and bias
    under `synthetic_shared_gain_bias`). This is shared information, so a
    cross-view code is expected to carry it.
  - **per-view style:** leakage into the code, so lower is better.
- The header gives each code's reconstruction error over the training template
  on the evaluated test subjects, next to the same ratio for the map's top-d PCA
  projection: the linear reference at the same size.
- Outputs: `factor_recovery.json` (everything, including each factor's chosen
  hyperparameters and the parity record), `factor_recovery.csv` (one row per
  representation, probe and factor, with per-view R²) and `comparisons.csv`
  (`z` minus each baseline, with a paired bootstrap interval over test subjects
  and the share of draws at or below zero).

Read `z - gap` and `z - patch` for local factors (lesion centre, temporal
atrophy, sulcal widening), `z - pca{d}` for whether the learned readout beats
linear compression at the same size, `z - map` for what compression lost, and
`z - z_init` for what training the readout bought. The bootstrap holds probe
fits fixed; use several bottleneck seeds for training variability. Under
`--synthetic-causal` a factor can be predicted through correlated anatomy, so a
lesion R² is not on its own evidence that the lesion's position is encoded; that
needs an intervention test.

## Validation

~~~sh
python -m unittest discover -s tests -p test_spatial_bottleneck.py -v
~~~

Tests cover spatial sensitivity when GAP is constant, training-only normalization,
variance across subjects, decoder isolation, fixed and per-view learned masks,
frozen encoder/codebook state, a learned toy spatial compression task, and a real
small VQ-VAE checkpoint-to-report run including reload and output protection.
The factor evaluator's tests check kernel ridge and PCA against sklearn on
explicit features (including PCA's reconstruction ratio), that test labels never
move a fit or a selection, that a GAP-invisible spatial factor is read from the
map and PCA but not GAP, and a checkpoint-to-factor-report run with extended test
subjects, shared gain/bias, per-subject lesion centres matching `sample_targets`,
a tampered test output and mismatched splits.
