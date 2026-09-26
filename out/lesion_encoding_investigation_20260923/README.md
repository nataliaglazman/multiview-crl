# Lesion encoding investigation

This is a generator-only diagnostic of the local code, using 128 validation
subjects and the supplied dense-model configuration: seed 42, resolution 64,
9 independent content factors, 3 style factors, clean content, fixed-reference
normalization, default sphere lesions and default (not identifiable) ventricles.
No trained checkpoint was available or loaded. No model or renderer was modified.

Run from the repository with the project's Python environment:

```bash
python out/lesion_encoding_investigation_20260923/reproduce.py
```

`visibility.json` records every subject and the aggregates. The script rewrites
that file. The runtime uses the current local renderer, so results are conditional
on its matching the remote training revision.

## Observations

- Lesions existed in all 128 images; minimum 126 voxels, median 131 voxels.
- Median lesion fraction was 0.355% of brain foreground, about 0.050% of the
  64-cubed image. Nominal diameter is 6.3 input voxels or 1.575 spacings on the
  stride-4 feature grid. Feature responses can cover more sites than this geometry.
- In 84/128 subjects, lesion support overlapped tissue labelled CSF, including
  the fissure. Median CSF-labelled fraction across all lesions was 14.6%.
  The geometric `mask_wm` used to constrain lesion support is not equivalent to
  `tissue_map == 2`: CSF and fissure labels overwrite tissue within that region.
- A fixed polarity-aware difference-of-Gaussians detector searching the natural
  masked image localized within one lesion radius in 4.7% of T1 and 65.6% of FLAIR
  samples. Restricting the search to known WM labels gave 8.6% and 90.6% respectively.
  These are the performance of one fixed detector, not limits on CNN localization.
- On the coupled lesion-on minus lesion-off image difference, an absolute-response
  detector localized all lesions within one radius in both modalities. Median
  errors were 0.71 voxels T1 and 0.57 voxels FLAIR. This counterfactual control proves
  a localized image effect exists, not that localization from one image is easy.
- A cross-validated ridge readout from the true rendered centroid to the three
  latent coordinates achieved R² 0.920, 0.870, 0.877. The nonlinear mapping from
  latent to physical position does not by itself explain near-zero model probes.

## Interpretation and limits

The reported lesion factors describe location, not lesion presence or size. A
constant-size blob that translates across an equivariant feature map can leave
GAP, global max and global quantiles unchanged. Anatomical context, boundaries,
striding and learned position-sensitive channels can break this invariance, so
this argument is a hypothesis about the learned encoder, not a proof of failure.

T1 also presents a harder localization task: lesion intensity 0.4 is below nominal
WM 0.8 but above nominal CSF 0.1, so support crossing their boundary has mixed
contrast. FLAIR lesion intensity 1.0 exceeds both WM 0.4 and CSF 0.1. Styles, noise
and smoothing modify these contrasts further. Lesions overwrite the tissue
intensity after its gain/bias modulation, so the two are not styled identically.

The previous VQ-VAE spatial FLAIR readouts already recovered lesion coordinates;
"lesions are never encoded" is therefore too broad. The next checkpoint-specific
test should compare full backbone maps, the nine projected content maps, GAP and
normalized GAP on identical held-out subjects, targeting both physical centroids
and latent coordinates. Matched lesion-position interventions would distinguish
insensitivity from a response that existing probes cannot decode. Existing
VQ-VAE-specific evaluators need a loader/forward adapter for this conv model.

All anatomical labels here are used for diagnostic evaluation only. This work
does not add supervised training or establish that InfoNCE caused the loss.
