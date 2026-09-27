# Ventricle alignment and balanced mismatch losses

```bash
python -m eval.ventricle_alignment \
  --run-dir results/synthetic/synthetic-clean-content-causal-ident-vent-12-4-3 \
  --num-samples 64 --batch-size 8 --loss-batch-size 64 \
  --eps 0.25 --causal match
```

The diagnostic freezes the checkpoint and measures how content features change
when ventricular size changes. It does not retrain the model or introduce
anatomical supervision into training. It supports the same Barlow Twins encoder
and standard projection-head path as `eval.lesion_alignment`.

## Feature responses

Each subject has two endpoints, with `z_content[1]` decreased/increased by `eps`.
The increment is in latent coordinates, not millimeters or voxels. Other content
latents, style, deformation fields, rendering noise, foreground, and the original
normalization affine are fixed. For causal datasets this is a coordinate
intervention, not propagation through the causal graph. Endpoints may be outside
the training distribution.

The console reports the same metrics as the lesion alignment test, with
`delta_h = high - low` and `high-cos` replacing `on-cos`:

- `native_probe`: native maps after `content_norms`.
- `native_alignment_source`: maps before `content_norms`; checked against the
  actual forward pooling output.
- `pooled_content`: pooled content after the training foreground filter.
- `loss_patch` or `loss_global`: after the optional trained projection head.
- `loss_gap`: the configured companion pooling of retained projected patches.
  If `bt_gap_pooling=stats`, this uses the actual statistics pooling despite the
  historical `loss_gap` name.

Positive response cosine means the intervention moves both views in similar
feature directions. Inspect absolute RMS as well: negligible responses do not
establish shared encoding. Similar feature vectors at the high endpoint do not
establish alignment of the ventricular response. Raw cosine is not the BT loss.
The wrong-subject control compares a subject's T1 response to the next subject's
FLAIR response within the logical subject batch; it is undefined for singletons.

WM-interior lesion placement can depend on ventricular geometry. If the rendered
lesion changes when ventricles change, the pair is excluded from feature/loss
summaries and recorded in `interventions.csv`. Nonlocal input changes and pairs
without a measurable input change in both views are also excluded. There is no
silent substitution of a different lesion placement or exclusion of lesions
from the images. The console reports the valid and confounded counts.

## Balanced loss comparison

For N subjects, each condition has 2N rows per modality:

| Condition | T1 rows | FLAIR rows |
| --- | --- | --- |
| Matched | low, high | low, high |
| Ventricle mismatch | low, high | high, low, within each subject |
| Subject mismatch | low, high | low, high, cyclically shifted between subjects |

Every condition contains exactly the same feature vectors in each modality.
Ventricle mismatch changes only the pairing of the two endpoints, leaving other
subject factors paired. Subject mismatch is a broader sensitivity control,
not a scale-matched intervention or a significance threshold.

`delta_ventricle = loss(ventricle mismatch) - loss(matched)`:

- Positive: this objective penalizes the wrong ventricle pairing in this batch.
- Near zero: it scarcely distinguishes the pairings, or its components cancel.
  Check response magnitudes and individual loss components before interpreting.
- Negative: the weighted total favors the mismatched pairing in this batch.
  This is not proof that a training update will erase ventricular information.

The code calls the actual `training.losses.barlow_twins_loss`, preserving saved
centering, patch statistics, per-entry normalization, similarity/variance
coefficients, GAP overrides, applicable whitening, and arm/level/overall weights.
Only the requested level is measured. `pairing_losses.csv` includes every
condition's raw terms and their paired differences, as well as unweighted and
weighted totals. Terms can cancel; changes in the total are not additive
anatomical attribution. Disabled arms have zero weighted loss.

Two correlation modes are reported when the checkpoint uses an EMA:

1. `instantaneous`: recompute correlations from this batch.
2. `ema_matched_reference`: one update from a hypothetical settled history equal
   to this batch's matched correlation. Each condition receives a fresh clone of
   the same reference. This illustrates EMA attenuation but does not recover the
   historical training state, which is not checkpointed. MSE and variance terms
   are not EMA-averaged.

A mismatch penalty does **not** prove an incentive to preserve the factor: both
views might discard it and remain matched. This is a sensitivity diagnostic, not
a factor-retention test, gradient attribution, or a causal explanation of the
training trajectory.

## Memory, batch size, and reproducibility

`--batch-size` (alias `--encode-batch`) controls the subjects in each GPU encoding
chunk. `--loss-batch-size` controls endpoint rows per modality for the loss. It
defaults to the saved training batch size and must be even and at least 4.
For 64 loss rows, the diagnostic uses 32 subjects with two endpoints each.
The foreground mask is computed over the entire logical subject batch and held
fixed across all encoding chunks and endpoint conditions. Content channel
selection is checked across chunks and endpoints. Evaluation mode freezes BN.

Excluded subjects are omitted before constructing batches; the last batch can
be smaller, and the actual loss-row count is saved and printed. A final single
subject still gets response metrics but no pairing loss or wrong-subject score.
The paired endpoints are dependent, so matching the row count does not reproduce
the distribution of 64 independent training subjects. The printed loss means
weight batches by subject count and are not confidence intervals. Larger
`--num-samples` provides more independent subject batches without increasing GPU
chunk size. No patch is treated as an independent statistical replicate.

The checkpoint is checked against loaded state. Every registered parameter and
buffer is hashed before and after evaluation; a mutation raises an error. CUDA
math flags used for reproducibility are restored afterward. No checkpoint is
written. Output directories are timestamped by default.

## Saved files

- `summary.json`: settings, checkpoint/state provenance, stage summaries,
  validity counts, per-batch loss terms, and limitations.
- `samples.csv`: per-subject stage responses and absolute magnitudes.
- `pairing_losses.csv`: matched, ventricle-mismatched, and subject-mismatched
  losses, components, weights, and deltas for both correlation modes.
- `interventions.csv`: every requested subject, latent endpoints, input signal,
  validity/exclusion reason, selected channels, and batch membership.
- `channels.csv`: per-channel response metrics at the loss inputs.

If there are no eligible interventions, the report records that fact rather
than treating zero response as successful alignment; empty tables are omitted.
