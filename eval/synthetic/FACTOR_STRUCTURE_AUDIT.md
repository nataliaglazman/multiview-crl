# Factor response geometry and latent spatial covariance

This CPU audit asks two separate questions:

1. Do the nine content controls produce spatially distinct image changes, and can
   the lesion/sulcal changes be explained by combinations of the other changes?
2. What spatial covariance do the actual ground-truth fields have, and does the
   generator meet the assumptions of Hälvä et al.'s spatial nonlinear ICA model?

It uses saved generator settings only. It does not load a checkpoint, train an
encoder, fit a prediction probe, or require repeated real acquisitions. Labels and
generator access are used for this synthetic diagnostic, not a training loss.

## Run

From the repository root, in an environment with this project's torch, scipy,
matplotlib and nibabel dependencies:

```bash
RUN="$PWD/results/encoder_patch_cuda_baseline/runs/conv_mlp_s42_cuda"
OUT="$RUN/evaluation/factor_structure_$(date +%Y%m%d_%H%M%S)"

python -m eval.synthetic.factor_structure_audit \
  --run-dir "$RUN" --out-dir "$OUT" \
  --num-samples 32 --subject-offset 1000 \
  --eps 0.1 0.25 0.5 \
  --kernel-samples 256 --kernel-grid 8 --kernel-bootstrap 100 \
  --nifti-subjects 1
```

For a quick check use `--num-samples 4 --eps 0.5 --kernel-samples 32
--kernel-bootstrap 20`. For kernels alone add `--kernels-only`; for image
responses alone add `--skip-kernels`. `--nifti-subjects 0` omits volumetric exports.
Output must be a new directory. `--cpu-threads` defaults to 1.

An existing encoder audit report can replace the run directory:

```bash
python -m eval.synthetic.factor_structure_audit \
  --audit-report /path/to/report.json \
  --out-dir /path/to/new_output \
  --kernels-only --kernel-samples 512
```

The source report must contain `settings`. To run on a laptop, copy just the
run's `settings.json` into a directory and use that as `--run-dir`, or copy an
existing report. No checkpoint or large feature bank needs transferring.

## Finite response maps

For each subject, modality, factor k and step h, the audit renders

```text
D_k = X(z + h e_k) - X(z - h e_k)
J_k(h) = D_k / (2h)
```

All other raw controls, deformation/fissure fields, acquisition draws and the
original subject's normalization affine are fixed. The same random draws remove
independent-noise fluctuations from the paired comparison; they do not remove
the renderer's noise nonlinearity. The endpoint's foreground mask is recomputed
so growth at the boundary is not cropped away. With per-sample/shared normalization
this deliberately isolates the factor under a fixed baseline affine, rather than
recomputing normalization after every intervention.

The **native lesion-placement rule** is retained. Changing brain anatomy can
change where a fixed lesion control places the sphere in white matter. Therefore
`lesion_displacement_vox` is recorded for every factor, including anatomical ones.
These are total generator responses, including that coupling; they are not
responses with the physical lesion pinned in place.

Quantization or saturation can produce no detectable response, especially at the
smallest step. These cases are counted as `zero_response`, not as successful
separation. If an intervention leaves no room for the lesion, it is recorded as
`render_failed`; the subject is never redrawn. Projections requiring a failed
reference factor are undefined. Check valid counts before comparing medians.

The default reference block is brain size, ventricle size, cortical thickness,
temporal atrophy and left–right asymmetry. This is called **anatomy**, not a proven
global partition: several of those mechanisms are regional. Lesion coordinates
and sulcal widening are tested both separately and together.

For each subject and step, the audit forms normalized response columns. If G
contains the anatomy columns, then

```text
P_G = projection onto span(G)
residual fraction = ||(I - P_G) D_k||² / ||D_k||²
```

This is uncentered Euclidean image energy, not probe R². It is invariant to
rescaling a nonzero column, separating response shape from response magnitude.
It does not whiten by a noise covariance and is not a detection-SNR estimate.

- `residual_energy_vs_anatomy` near 1: most of this measured response lies outside
  the anatomy response space at this subject and step.
- Near 0: combinations of anatomy responses explain most of it to this finite
  approximation.
- `residual_energy_vs_all_others` additionally compares against all eight other
  content responses. For an anatomy factor, the anatomy reference excludes itself.
- `min_principal_angle_deg` near 0: the two active response spaces contain nearly
  aligned directions; near 90 means their measured directions are orthogonal.
- `target_rank` and `residual_target_rank` must be read with the requested target
  dimension. A missing or redundant target direction cannot be rescued by a large
  angle between the remaining directions.

Projection/angles are computed **within each subject** before summary. Averaging
signed response maps over subjects first could cancel or displace lesion effects.
Reference and target columns use a relative SVD threshold (`--svd-rtol`, default
1e-5). Directions with maximum endpoint difference <= `--response-atol` (1e-6)
are treated as unresolved. These tolerances and perturbation steps are explicit
parts of the diagnostic, not a theoretical rank certificate.

Spatial extent is measured by the fraction of full-volume voxels containing 90%
of squared response energy and the effective number of regions carrying energy.
For region energies E_r, effective regions = `(sum E_r)² / sum(E_r²)`.
Regions use a regular 4³ grid by default. `gap_survival` is `|sum D| / sum |D|`:
it is small when positive/negative changes cancel under whole-image averaging.
No single extent metric establishes biological locality or identifies a block.

Outputs:

```text
response_summary.png   median spatial extent and residual-energy fractions
response_summary.csv   medians, quartiles, valid counts, failures and zero responses
responses.csv          per-subject factor effects, extent, projections and lesion movement
blocks.csv             principal angles and ranks for the four block comparisons
cosines.csv            pairwise signed response cosines, per subject
nifti/                 baseline images, signed differences and target residual maps
```

NIfTI maps retain array axes, an identity affine and unknown spatial units; they
do not imply anatomical orientation or physical millimetre spacing. Differences
are `D_k`, not `J_k`; divide by `2h` to compare derivative amplitudes. Residual maps
have the anatomy projection removed. Use a signed colour map centered on zero.

## Covariance kernels

Each ground-truth spatial field supplies one realization per subject. The audit
estimates the full ensemble covariance at locations u and v:

```text
K(u,v) = sum_n [(s_n(u)-mean_n s_n(u)) (s_n(v)-mean_n s_n(v))] / (N-1)
```

The centering is across subjects at each location. The audit does **not** subtract
each subject's spatial mean or divide by its spatial standard deviation. Such
operations would change the process being measured.

Deformation and fissure fields, plus the lesion field when enabled, use the
renderer’s exact interpolation before sampling the same voxel locations. Their
native-grid covariance matrices are also saved. Default iid 4³ and 8³ grids can
have different covariance after interpolation because their physical spatial
scales differ. This is distinct from sampling fields from explicitly different
GP kernels on the same native lattice.

The nine named content controls are **scalars per subject**. Their 9x9 covariance
and correlation matrices are reported separately. They are not treated as nine
spatial processes, and no artificial constant-in-space kernels are constructed
by broadcasting them. The sulcal scalar drives a spatial pattern in the renderer;
the covariance of that derived pattern would be a different object.

`kernel_profiles.csv` averages K(u,u+h) over positions for x/y/z axial lags and
their mean. These are descriptive lag profiles; no stationarity or squared-
exponential form is assumed. Normalized profiles divide by mean point variance.
Their intervals resample whole subjects, re-estimate ensemble means, and are
pointwise exploratory intervals, not simultaneous confidence bands.

Kernel comparisons remove overall variance using `K / mean(diag K)` before
computing relative Frobenius distance. The output also gives split-half distances
within each component to illustrate finite-sample variability. These distances
are not formal equality tests. Cross-component correlations are diagnostics, not
proof of statistical independence. With N subjects, empirical covariance rank is
at most N-1 even if the population covariance has full rank.

Outputs:

```text
kernel_diagnostics.png          scalar correlations and spatial kernel comparisons
scalar_content_covariance.csv  ordinary scalar covariance/correlation
latent_covariances.npz          full native/common-grid covariance matrices and metadata
kernel_profiles.csv            covariance vs distance, with subject-bootstrap intervals
kernel_comparisons.csv          kernel distances and cross-component correlations
kernel_profile_differences.csv paired differences of normalized lag profiles
kernel_report.json             protocol, component activity and applicability assessment
halva_checks.csv               explicit model-assumption checks
```

For deformation/fissure the effective pre-threshold field covariance is their
structural multiplier squared times the raw covariance. In `clean_content=True`
runs both multipliers are zero: nonzero, different **raw** kernels do not imply
recoverable fields in the observed images.

## Does Hälvä's theorem apply?

[Hälvä et al. (AISTATS 2024)](https://proceedings.mlr.press/v238/halva24a.html)
study independent spatial components observed through injective pointwise mixing
with an observation-noise model. Their GP special case requires distinct kernels;
that criterion alone does not establish the full model assumptions.

The current renderer has hard tissue thresholds, spatial masking and image
smoothing, along with magnitude noise. The clean-content setting also removes
two latent fields from the images. Moreover, the implemented `sample_gp_field`
standardizes each realization before returning it, so its `gp`/`tp` option names
alone do not certify the paper's process distributions. WM-fit rejection can
also alter the accepted joint latent distribution. The audit reports these
limitations explicitly and never declares the theorem established from a kernel
plot. Spatial priors can still be useful empirical modelling choices.

## Tests

```bash
python -m unittest tests.test_factor_structure_audit -v
```

Tests include known confounded/orthogonal/rank-deficient blocks, zero responses,
native renderer replay, exact exported NIfTI values, retained placement failures,
settings restoration including GP options, correct ensemble covariance, differing
spatial dependence, and explicit reporting of inactive fields.
