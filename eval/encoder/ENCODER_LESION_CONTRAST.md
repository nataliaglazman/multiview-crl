# Lesion contrast, probe errors and NIfTI examples

This follow-up tests whether lower rendered lesion contrast is associated with
larger T1 physical-location errors. It reuses the predictions from
`encoder_lesion_intervention`; it does not load an encoder, fit another probe, or
train anything. It runs on CPU, including on a laptop. The default also renders
compressed NIfTI examples from the low- and high-contrast T1 groups.

## Run on the NVIDIA PC

From the repository root in `monai_env`, after copying the new source:

```bash
AUDIT="results/encoder_patch_cuda_baseline/runs/conv_mlp_s42_cuda/evaluation/lesion_moves_20261005_093801"
OUT="${AUDIT}_contrast_$(date +%Y%m%d_%H%M%S)"

python -m eval.encoder.encoder_lesion_contrast \
  --audit-dir "$AUDIT" \
  --out-dir "$OUT" \
  --nifti-per-group 3
```

`OUT` must be a new directory. The source audit is read-only. Use any other
completed movement audit as `AUDIT` to test another model/checkpoint.
NIfTI export uses `nibabel`, already used elsewhere in this repository.
If it is missing from your environment, install it with `python -m pip install nibabel`.

The minimum inputs are `report.json`, `pairs.csv`, `trained_predictions.npz`, and
`initial_predictions.npz` if that arm was included. No checkpoint or feature banks
are needed. These are the only files to copy to a laptop for this test. The saved
settings, subject IDs, intervention axes and normalization reconstruct the images.
Exact image SHA256 must match the original audit before the analysis or NIfTI
exports proceed. If replay fails, use the renderer and Python/PyTorch environment
that produced the source audit; the program refuses to pair changed images with
old predictions. The saved CPU thread count is honored when present.

## Contrast definition

For each moved-lesion endpoint, the renderer produces a reference with the lesion
removed. Anatomy, tissue intensities, acquisition parameters, random noise draws,
bias field and the original normalization affine are held fixed. This is a
measurement reference only: it is never passed to a probe or encoder.

Primary contrast is:

```text
abs(mean(actual image - lesion-free reference, over lesion voxels))
```

It measures the actual lesion effect after bias field, magnitude noise, smoothing
and normalization, at the exact lesion location. The signed value is also saved
(usually negative for T1 and positive for FLAIR). The no-lesion reference restores
white matter at that position. A check rejects any difference outside the lesion
plus its one-voxel blur boundary.

A secondary `local_wm_contrast` compares the visible lesion mean with a local
healthy-WM shell. The shell excludes the blur boundary and must contain at least
eight voxels; otherwise that metric is missing. This measure can be affected by
nearby tissue boundaries, unlike the matched reference. Neither measure is a
calibrated contrast-to-noise ratio.

## Reading the analysis

- **`associations.csv`**: subject-level Spearman correlation between contrast and
  error. Negative means higher contrast goes with lower error. `partial_spearman`
  adjusts the ranks for mean nonzero movement distance and the renderer's preblur
  noise sigma after normalization. Read the bootstrap intervals, not only the sign.
- **`contrast_groups.csv`**: low/middle/high subject contrast thirds and their
  movement skill, gain, movement error and endpoint error. `own_view` uses each
  modality's contrast groups; `t1_matched` compares T1 and FLAIR on exactly the
  same subjects, grouped by T1 contrast.
- **`paired_views.csv`**: one matched T1/FLAIR comparison per subject and probe.
  Positive `t1_minus_flair_endpoint_error_vox` means T1's error was larger.
- **`subject_errors.csv`**: per-subject contrast, errors and movement counts.
- **`image_contrasts.csv`**: individual endpoint contrasts, acquisition parameters,
  normalization, mask sizes and image hashes.
- **`report.json`**: protocol, verification, settings, group thresholds and results.

All axes and endpoints from one subject are averaged before correlation. The
bootstrap resamples whole subjects, not individual endpoints. Group membership
and image export selection use contrast only, never prediction errors. Constant
contrasts are not arbitrarily divided into low/high groups.

`endpoint_error_vox` in the subject table is mean Euclidean position error over
all that subject's endpoints. `relative_movement_error` is mean displacement-error
norm divided by true displacement norm over its nonzero moves. Group
`all_endpoint_error_vox` includes no-movement pairs too; the inherited movement
metrics and `endpoint_error_vox` in the group table use moved pairs, matching the
original audit. Subjects with no nonzero moves cannot enter the adjusted
associations, but remain in the image exports and endpoint-error tables.

The result would support a contrast explanation if low-contrast T1 subjects have
higher errors, the association persists after adjustment, and the T1/FLAIR error
gap is smaller in high-contrast T1 subjects. If even high-contrast T1 subjects have
poor recovery, investigate the representation/readout and probe next. These are
exploratory associations with fixed probes/checkpoints, not proof that contrast
causes the entire gap. Anatomy and acquisition still covary. Initial-checkpoint
and shuffled-probe results are retained when present in the source audit.

This diagnoses **physical centroid predictions at the controlled move endpoints**.
It does not recompute raw `lesion_x/y/z` control R², the observational held-out
location score, or an identifiability guarantee. Controlled endpoint pairs may
differ from the observational training distribution.

## Compare low/high T1 images in a NIfTI viewer

By default, `nifti/` contains the three lowest-contrast subjects in the low third
and the three highest-contrast subjects in the high third (fewer if a group is
small). Each subject uses the first intervention axis in the source audit,
usually x, with both endpoints:

```text
low_t1_subject<ID>_x_a_t1.nii.gz
low_t1_subject<ID>_x_b_t1.nii.gz
low_t1_subject<ID>_x_a_flair.nii.gz
low_t1_subject<ID>_x_b_flair.nii.gz
low_t1_subject<ID>_x_a_lesion_mask.nii.gz
low_t1_subject<ID>_x_b_lesion_mask.nii.gz
low_t1_subject<ID>_x_no_lesion_t1.nii.gz
low_t1_subject<ID>_x_no_lesion_flair.nii.gz
low_t1_subject<ID>_x_white_matter_mask.nii.gz
```

High-contrast examples use the `high_t1_` prefix. `a` is the -eps endpoint and `b`
the +eps endpoint, not different contrast conditions within one subject. T1 and
FLAIR at the same endpoint share the same lesion mask. Subject group assignment
uses mean contrast over all its axes/endpoints; individual endpoint contrasts can
vary slightly.

Open a low and a high T1 image side by side and overlay their lesion masks.
`nifti/examples.csv` lists each file, its measured contrast and its zero-based
i,j,k lesion centroid so you can navigate to the correct slices. Compare the
matched FLAIR files at those same positions.

**Use the same brightness/window settings for all T1 examples.** The exporter
preserves the exact normalized inputs without intensity rescaling; it writes
common per-modality display windows to `display_windows.json` and the NIfTI
`cal_min/max` fields. Viewers that auto-window each image separately can obscure
the difference. FLAIR has its own common window. The lesion-free references help
show what each location would look like without the lesion.

These are synthetic index-space volumes: the affine is identity, spacing is one
voxel with unknown spatial units, and array axes are preserved. There is no
claimed physical millimetre scale or anatomical RAS orientation.

Increase `--nifti-per-group` for more examples; set it to `0` for tables only.
The `.nii.gz` files are compressed; the default exports six subjects rather than
the complete cohort. Different subjects still have different anatomy and
acquisition, so the examples are illustrative, not a controlled high/low-contrast
intervention in the same anatomy.

## Verification

```bash
python -m unittest tests.test_encoder_lesion_contrast -v
```

Tests run the actual movement audit first, remove its checkpoints, and replay its
images/predictions through this analysis. They check exact NIfTI values and masks,
shared/per-sample normalization, source integrity and mismatched-cohort rejection,
subject-level grouping, known renderer contrast changes, and a planted confound
for the adjusted correlation.
