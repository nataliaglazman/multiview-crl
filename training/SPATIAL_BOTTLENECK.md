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
recovery. Factor probes/intervention tests remain separate evaluations, and a
dimension small enough to represent selected scalar factors may be insufficient
for additional shared spatial fields in the generator.

## Validation

~~~sh
python -m unittest discover -s tests -p test_spatial_bottleneck.py -v
~~~

Tests cover spatial sensitivity when GAP is constant, training-only normalization,
variance across subjects, decoder isolation, fixed and per-view learned masks,
frozen encoder/codebook state, a learned toy spatial compression task, and a real
small VQ-VAE checkpoint-to-report run including reload and output protection.
