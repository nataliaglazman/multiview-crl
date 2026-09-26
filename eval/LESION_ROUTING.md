# Where does a VQVAE carry lesion location?

Run this on a **VQVAE reconstruction checkpoint**, not the encoder-only ResNet run:

```bash
python -m eval.lesion_routing \
  --run-dir results/synthetic/synthetic-clean-content-causal-sp-s-1-cont \
  --num-samples 64 --batch-size 2 --eps 0.5 \
  --axes x y z --causal match --examples 2 --save-nifti
```

Replace the run directory with the checkpoint whose T1 reconstructions show lesions.
Default checkpoint: `vqvae_model.pt`; use `--checkpoint vqvae_best.pt` if needed.
For a faster first check, use `--num-samples 16 --axes x --examples 1`.
No optimizer, fitted probe, or model training is involved. The test verifies that the
loaded model matches the saved checkpoint and that registered parameters/buffers do
not change. It never writes a checkpoint. Synthetic factors are used only to generate
controlled evaluation pairs and score their effects.

## What changes

For each subject and each requested axis, A/B change just that lesion latent by
minus/plus `eps`. Other anatomical factors, styles, deformation fields, noise seeds,
brain mask, and the original normalization affine stay fixed. The anatomical tissue
map must be identical; image changes must stay within the changed lesion support plus
the renderer's one-voxel blur. Actual lesion centroids and voxel counts are saved:
latent controls need not equal physical coordinates, and placement can couple axes.

`--causal match` matches the **starting subjects** to the saved settings. Moving one
coordinate holds all other factors fixed, including causal descendants. This is an
image-factor isolation test, not a propagated SCM intervention. Perturbed images and
hybrid codes can be outside the training distribution. `--causal iid` changes the
starting distribution too; do not interpret it as a matched-distribution comparison.
Field-lesion generators are rejected because their lesion factors are not x/y/z.
Explicit placement settings unsupported by the local dataset class are also rejected
instead of silently using another generator. For checkpoints made with the old
renderer, `--old-generator` has the same meaning as in `eval.ventricle_routing`.

## What is decoded

The test captures full spatial content tensors after quantization and actual injected
style tensors after their bottleneck/quantization. It reuses the checked replay and
swap implementation in `eval.ventricle_routing`. It preserves view-major batching,
per-view codebook provenance, decoder batch shape, and forward's actual floating-point
quantized values. Replaying both endpoints must reproduce the model forward; the
remaining numerical error must be small relative to the local intervention.

| Reconstruction | Content donor | Style donor |
|---|---|---|
| AA | A | A |
| BA | B | A |
| AB | A | B |
| BB | B | B |

All content levels are swapped together. In multi-level models, fine content codes
may already depend on coarser style-conditioned decoder outputs, so routing concerns
the tensors supplied to the decoder, not independent encoder mechanisms.

## Read the numbers in this order

1. **Coverage and joint fidelity.** Both lesions must be nonempty, the input movement
   measurable, and endpoint replay resolved. Joint gain is the projection of `BB-AA`
   onto `input_B-input_A` within changed-lesion support dilated by one voxel. Ideal
   gain/cosine/error are `1/1/0`. Poor joint fidelity means the reconstruction does not
   reliably track this movement, so a content/style assignment is inconclusive.
   Natural endpoint ROI errors and the fraction of joint response energy inside the
   affected region are saved too. Inspect the endpoint reconstructions visually.
2. **Conditional routing.** Content effects are `BA-AA` and `BB-AB`; style effects
   are `AB-AA` and `BB-BA`. The printed content/style gains average the two contexts.
   Strong content gain with little style effect supports decoder use of content;
   the reverse supports style. These are response gains, not percentages of mutual
   information or coordinate-probe R².
3. **Interaction.** `BB-BA-AB+AA` measures dependence on donor context. A large RMS
   relative to the input movement makes a single-path interpretation misleading.
   Mean content/style gains sum to joint gain by construction; that equality does
   not establish independent or additive representations. Check both conditional
   gains in `responses.csv`.
4. **Spatial versus pooled sensitivity.** Native and GAP RMS compare `latent_B-
   latent_A` before/after spatial averaging. Large native response with small GAP
   response demonstrates cancellation under pooling. It does not demonstrate that
   a particular finite-sample coordinate probe can recover the location. Native
   response alone also does not prove that the decoder uses it; read the swaps.

If both blocks failed coordinate probes but the joint reconstruction follows the
movement, lesion information is accessible to the decoder through their full spatial
tensors. Swaps distinguish their contribution and interaction. Failure of these
probes is not evidence that the information exists in neither block.

## Files

A new `lesion_routing_<timestamp>` directory contains:

- `summary.json`: settings, checkpoint step/state digest, protocol, per-axis/view
  means, medians, counts, and subject-bootstrap 95% intervals for routing metrics.
- `responses.csv`: every subject, including empty/unresolved cases, actual centroids,
  both donor contexts, numerical controls, latent responses and code changes.
- T1 and FLAIR PNGs for the first `--examples` subjects per axis, showing inputs,
  all four reconstructions, and signed input/reconstruction changes. Both endpoint
  lesion planes are shown when they differ.
- With `--save-nifti`, the same example volumes and endpoint lesion masks. Affines
  are identity: coordinates are synthetic voxel indices, not patient orientation
  or a claim about physical voxel spacing.

The terminal prints **paired means**, as does the updated `eval.ventricle_routing`
command (see `VENTRICLE_ROUTING.md` for the matching size intervention). Older
ventricle reports printed medians. Compare like summaries when comparing runs.
`resolved` means numerically resolved, not that
reconstruction fidelity passed a biological or anatomical threshold.
