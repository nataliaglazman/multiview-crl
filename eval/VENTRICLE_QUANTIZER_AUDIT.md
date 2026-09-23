# Ventricle quantizer audit

Three frozen-checkpoint diagnostics: projection conditioning, sensitivity of real
code assignments/quantized tensors to ventricular interventions, and spatial
localization of those responses. No model training, optimizer, or checkpoint writes.
Synthetic anatomy is used for evaluation only.

```bash
python -m eval.ventricle_quantizer_audit \
  --run-dir results/synthetic/synthetic-clean-content-causal-ident-vent-12-4-3 \
  --num-samples 64 --batch-size 4 --eps 0.25 --causal iid
```

Repeat on the baseline with the same arguments. For a response-size sweep, use
`--eps 0.1 0.25 0.5`; this triples paired inference work. Eight subjects are enough
for a smoke run, not a stable scientific conclusion. This test does not fit probes.

## What the tests measure

**Projection:** SVD of each view's actual content and (when quantized) style
`CodeLayer.conv_in`, including all embedding coordinates. Shared codebooks produce
identical per-view SVDs. Saves singular values, effective rank at float32 tolerance,
float64 rank, input nullity and condition number when full column rank. A rank-deficient
map has no finite injective condition number (saved as null). Bias does not affect rank.
The real paired input/projection tensors must reproduce the affine; a truncated
pseudoinverse then measures how accurately the pre-projection response is recoverable.
Full rank supports invertibility, not necessarily easy finite-sample regression.

**Intervention:** `z_content[1] = original_value +/- eps`. All other realized latent
values, style, deformation, fissure, lesion and rendering noise seeds stay fixed.
This does not propagate an intervention through SCM descendants. Original rendering
is replayed exactly to verify seed conventions. The original observation supplies
one fixed normalization affine per view and a fixed brain mask for both variants.
Non-affine normalization is rejected. Thus the test avoids global differences caused
by independently re-normalizing each intervention. The pair can be off-distribution,
especially for per-sample normalization or saturated latent/radius settings.

`--causal iid` samples independent base factors, matching the default prior probe
tests; `--causal match` retains the run's sampling distribution for base subjects.
Settings and actual z1 endpoints are saved. Subjects with no measurable rendered
change (voxelization, saturation, overlap with lesion, etc.) remain in the CSV but
are excluded from response summaries. Unchanged lesion load is checked explicitly.

One low, high and repeated-low real forward per batch captures the native projected
features, exact quantized decoder tensors and **actual forward code assignments**.
No nearest-code approximation or lookup reconstruction of the decoder tensor.
Continuous style is skipped because it has no code assignments. Model architecture
support is the same as `content_path_probe`: one level, hard fixed/learned masks,
style injection, shared/separate encoders and codebooks. Sphere-lesion datasets only.

Per subject/view/block/region:

- RMS of projected and quantized finite differences, changed-code fraction.
- Quantized/projected response norm ratio, directional gain, cosine and relative error.
  These compare responses in the **same embedding space**, not image space.
- Endpoint quantization error, plus its ratio to the continuous intervention response.
- Projection response norm gain and pseudoinverse reconstruction error.
- Repeat-low response/assignment error. A feature response is flagged resolved only
  above `max(1e-8, 10 * repeat RMS)`. Undefined ratios/cosines remain missing. A real
  projected response with zero quantized response has ratio 0, error 1 and no cosine.

Discrete responses may jump or amplify; a norm ratio above one does **not** imply
improved information, and a zero local response does not imply global information loss.
Inspect `resolved_quant` and replay-code changes before interpreting quantized responses;
the printed `resolved` column counts resolved **projected** responses.

**Spatial regions:** identify changed tissue voxels, dilate by one input voxel to
cover the renderer's 3³ blur, and verify no input difference occurs outside that support.
Map support to feature-map bins with max pooling. Report:

- `all`: entire map;
- `affected_bins`: bins intersecting changed-tissue/blur support;
- `neighborhood`: affected bins dilated by `--halo` native sites (default 1);
- `outside`: complement of that neighborhood, including background feature sites.

Save response-energy fractions in each region as well as RMS (to distinguish area
from per-site response strength). These are **spatial-bin regions, not receptive-field
boundaries**. Convolutions and normalization can legitimately propagate responses
outside them. A spatially global style tensor cannot be localized this way. Pooled
code-usage counts include both endpoints of measurable pairs; unobserved codes are
not a claim of dead training codes. Regions overlap except neighborhood/outside.

## Outputs and interpretation

`ventricle_quantizer_audit_TIMESTAMP/` contains:

- `summary.json`: SVDs, descriptive mean/median responses with valid counts, endpoint
  assignment histograms/perplexity, run settings and registered-state preservation check.
- `responses.csv`: all paired metrics, replay errors, validity flags and z1 endpoints.
- `examples.npz`: first four subjects' input differences/support, projected/quantized
  response-norm maps, changed-code maps and affected/neighborhood masks at native
  resolution. Keys identify perturbation, subject, view, block and array type.

No automatic diagnosis or significance verdict. Baseline/contrastive comparisons
must match renderer settings/resolution and consider different training durations.
The same dataset seed is used across eps values, so a sweep reuses subjects and
must not be treated as independent replicates.

- Reversible projection response with weaker probe recovery suggests investigating
  conditioning/readout rather than declaring anatomical information destroyed.
- Measurable projected response with few code changes and weak quantized response
  supports an assignment-resolution bottleneck for this intervention size.
- Adequate quantized response with style-dominated reconstruction points toward a
  separate decoder-use question; this audit does not perform decoder swaps.

```bash
python -m unittest discover -s tests -p 'test_ventricle_quantizer_audit.py' -v
```
