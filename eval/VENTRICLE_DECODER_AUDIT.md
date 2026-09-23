# Ventricle decoder audit

This frozen-checkpoint diagnostic asks whether changing ventricular size reaches
the reconstruction through content, style, or both. Synthetic anatomy is used only
to generate and evaluate interventions; no labels or supervision enter training.

```bash
python -m eval.ventricle_decoder_audit \
  --run-dir results/synthetic/synthetic-clean-content-causal-sp-s-1-cont \
  --num-samples 64 --batch-size 4 --eps 0.25 --causal match
```

Run the same command with the older run directory for comparison. Use matching
sample count, epsilon, causal mode and renderer settings. Different training steps
or simultaneous changes to style quantization confound attribution to spatial size.
`--eps 0.125 0.25 0.5` optionally checks sensitivity to intervention magnitude;
`--examples 0` disables image artifacts. No retraining is required.

### Reconstruction panels and NIfTI export

Each saved subject now also gets a `*_reconstructions.png` panel showing natural
T1 and FLAIR inputs next to their reconstructions in three array planes. Input and
reconstruction share an intensity scale within each modality and use matched slices.

Add `--save-nifti` to export the saved examples to a `nifti/` subdirectory.
`--examples` controls how many subjects are saved (default 2; use 64 to save all
subjects of a 64-subject audit). Export includes both modalities' natural inputs
and reconstructions, intervention inputs, all four decoder combinations, affected
masks and signed response volumes. Files are compressed `.nii.gz`.

You can export an **already completed audit without running inference again**:

```bash
python -m eval.ventricle_decoder_audit \
  --from-examples results/synthetic/synthetic-clean-content-causal-sp-s-1-cont/ventricle_decoder_audit_20260921_130818_473311/examples.npz \
  --save-nifti
```

This writes new panels and NIfTIs under a timestamped `exports_*` directory beside
the archive. Only subjects already saved in that archive can be exported; the
export-only mode uses all of them. `--output-dir` can select a new destination.

NIfTI export requires `nibabel` (`python -m pip install nibabel` if missing).
Arrays are saved without transposing, clipping, or intensity rescaling. These
synthetic volumes have no physical image geometry: the affine is identity, voxel
spacing is one in unknown units, and no anatomical orientation is asserted. All
volumes from a subject overlay in the same synthetic coordinate system. See
`nifti/README.txt` for the filename and coordinate conventions.

## Intervention and swaps

For each test subject, the renderer generates `z_content[1] - eps` and
`z_content[1] + eps`. All other realized latents, lesions, appearance, and noise
seeds stay fixed. The original render is reproduced exactly before intervention.
The natural sample's normalization affine and foreground mask stay fixed. This
is a direct intervention, not propagation through causal descendants; variants
may be off-distribution, especially for per-sample normalization.

Each endpoint is encoded normally. The audit captures the **actual content and
style tensors passed to the decoder**, including quantization or continuous style
and the configured spatial bottleneck. It decodes:

| Name | Content | Style |
|---|---|---|
| ll | Low ventricular latent | Low ventricular latent |
| lh | Low | High |
| hl | High | Low |
| hh | High | High |

All swaps stay within the same subject and modality. Replaying ll and hh must
match their forward reconstructions under a per-subject RMS tolerance. A repeated
low-endpoint forward also measures numerical repeatability. Both errors are
reported regionally, not only averaged across a batch. This supports one VQ level
with injected style and fixed or hard learned masks. Strict checkpoint loading
rejects missing/unexpected weights; registered state is hashed before and after.

## Read the output in this order

1. **Natural and endpoint reconstruction:** inspect masked MAE/RMSE and saved
   images. Natural reconstruction uses the unperturbed subject. The affected
   region is tissue changed by this intervention, expanded for renderer blur;
   it is not the entire ventricle or a segmentation-derived volume measurement.
2. **Joint response fidelity:** compare `hh - ll` with the rendered input change.
   Identity gives gain 1, cosine 1, relative error 0. No response gives gain 0,
   relative error 1 and undefined cosine. High cosine alone can hide attenuation.
3. **Pathway response:** content mean is `((hl-ll) + (hh-lh))/2`; style mean is
   `((lh-ll) + (hh-hl))/2`. These average the two contexts and sum to the joint
   response. Look for a faithful joint response carried mainly by content.
4. **Interactions and spill:** both individual context effects are saved.
   `hh-hl-lh+ll` measures interaction. Outside the changed support, examine absolute
   response RMS and the fraction of brain-wide joint response energy. Input
   change there is zero, so gains and relative errors are intentionally undefined.

`resolved_input` only means input RMS exceeds 1e-8 and ten times the larger replay
or repeated-forward error. It does **not** certify reconstruction quality. No
arbitrary quality threshold filters out poorly reconstructed subjects. Normalized
metrics are omitted for unresolved inputs; absolute RMS and nonempty-region
reconstruction errors remain available. Each summary metric carries its own count.

Gains and response energy fractions are not information-retention fractions.
Mean pathway effects may conceal opposite effects in the two contexts; inspect
the context-specific columns and interaction. This audit measures the decoder's
response to ventricular interventions, not the causal contribution of a training
loss, anatomical volume accuracy, or successful appearance disentanglement.

## Files

A timestamped `ventricle_decoder_audit_*` directory contains:

- `summary.json`: configuration, checkpoint step, state hash, mean/median metrics
  and 95% subject-bootstrap intervals (500 draws). These intervals do not include
  training-seed uncertainty.
- `responses.csv`: one row per subject, epsilon, modality and region (`brain`,
  `affected`, `outside`), including numerical controls and all context effects.
- `reconstruction.csv`: natural, low and high reconstruction MAE/RMSE for each
  region. Natural rows repeat across epsilons because affected regions differ;
  do not count those repeats as independent subjects.
- `examples.npz`: complete input, reconstruction, four swap and response volumes.
- PNG panels: natural input/reconstruction, paired inputs, all four decoder
  combinations and input/joint/content/style changes, at a slice with maximal
  affected support. Images share one scale; changes share a symmetric color scale.

The original checkpoint is never written. This test supports continuous global
style as well as spatial quantized style. Dataset construction currently shares
the existing pseudo-MRI sphere-lesion diagnostic helper.

Tests:

```bash
python -m unittest discover -s tests -p 'test_ventricle_decoder_audit.py'
```
