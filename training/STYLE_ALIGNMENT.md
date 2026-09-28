# Within-modality style alignment

`style_contrastive_mode: within_modality` adds a synthetic same-acquisition,
different-anatomy alignment objective. It aligns T1 with T1 and FLAIR with FLAIR;
it never pulls all subjects of a modality to one embedding and never aligns T1
style to FLAIR style. Existing modes and default runs are unchanged.

## Pair construction and assumptions

For each anchor, the dataset selects a different subject's anatomy from the
**same split**, including content factors and deformation/fissure/lesion fields.
It renders that anatomy using the anchor's separate T1 and FLAIR acquisition
settings. Gain, bias, noise sigma, and the renderer's random multiplicative bias
field are shared with the respective anchor view. Noise realizations are redrawn
independently, so an identical noise pattern is not a positive-pair shortcut.

Training partners/noise are resampled on every access, even with cached anchors.
Validation/test pairs are repeatable for each index. No new pairs are generated
when the overall loss weight is zero. The dataset retains its original images,
ground-truth evaluation labels, split sizes, and content-alignment pairs.

The loss reads only embeddings and pair correspondence: no segmentation masks,
ventricular sizes, lesion coordinates, or acquisition values are targets. However,
the correspondence **uses controlled generator information** about which factors
remain fixed. This is additional self-supervision, not a claim of identification
from arbitrary observational images. It is currently restricted to synthetic
`pseudo_mri` with `fixed_reference` normalization. Subject-specific normalization
would change acquisition appearance with anatomy and invalidate the pairs.

## Objective and training path

For each level, let `a_v` be anchor styles and `b_v` paired styles in modality `v`.

```
alignment = mean_v mean((a_v - b_v)^2)
variance = mean_(v, branch) mean_channels relu(1 - subject_std)
loss_level = alignment + style_alignment_var_weight * variance
weighted_loss = scale_style_contrastive_loss * sum_levels(loss_level)
```

Both terms act on the **same post-bottleneck, pre-quantization** style tensor
before decoder dropout/detachment. Full spatial tensors are compared without GAP
pooling. Subject standard deviation removes each coordinate's batch mean before
averaging squared deviations within each channel; a fixed spatial template cannot
satisfy it. Views are kept separate throughout. There is no covariance penalty
forcing acquisition to occupy more independent dimensions than it has.

The paired anatomy receives an additional encoder-only forward with gradients
through both branches. Its own brain mask is supplied for configurations that
mask latent background. There is no extra decode, quantization loss, codebook EMA
update, or cross-modal content loss for this branch. Encoder computation and
activation memory increase. Training BatchNorm sees the extra forward, as with
other augmentation passes. A fixed content/style mask is required so both passes
compare the same channels. At least two subjects per view are required.

This implementation adds **style alignment**, not swapped reconstruction. The
ordinary reconstruction objective still needs to preserve information in content.
The variance hinge discourages collapse but has zero subgradient at exact constant
features. Low alignment loss alone can reflect collapse; monitor variance too.
Pre-quantization alignment does not certify post-quantization agreement or routing.

## Configuration

Add to a synthetic VQ-VAE YAML:

```yaml
style_contrastive_mode: within_modality
scale_style_contrastive_loss: 1.0  # Illustrative starting scale, NOT calibrated for your run.
style_alignment_var_weight: 1.0
scale_style_hsic_loss: 0.0        # Keep the separate supervised loss off.
synthetic_normalize: fixed_reference
inject_style_to_decoder: true
mask_mode: fixed
style_spatial_size: 1            # Recommended from the existing routing experiment; not required.
```

The corresponding CLI flags are `--style-contrastive-mode within_modality`,
`--scale-style-contrastive-loss`, and `--style-alignment-var-weight`.
`style_independence_var_weight` belongs only to the separate independence mode.
Modes are exclusive; this does not also apply legacy cosine repulsion or HSIC.

Calibrate the overall weight against encoder gradient magnitudes, then use a short
ablation before a full run. Reconstruction/content weights are not changed here.
In particular this does not repair a content BT objective that rewards ventricular
mismatches through its off-diagonal term.

## Diagnostics and checks

- `Style/alignment_mse_v{0,1}_L*`: correct-pair MSE, separately for T1/FLAIR.
- `Style/alignment_mismatched_mse_v{0,1}_L*`: cyclic wrong-pair MSE, diagnostic only.
- `Style/alignment_std_{anchor,pair}_v{0,1}_L*`: subject variation on each branch.
- `Style/alignment_var_hinge_L*`, `Style/alignment_var_weighted_L*`: collapse control.
- `Style/within_modality_L*`, `Style/within_modality_weighted`: total and weighted loss.

Successful training should retain acquisition variation while making correct-pair
MSE smaller than wrong-pair MSE. Validate ventricular/lesion routing, reconstruction
fidelity, and acquisition recovery, including post-quantization codes. Factor
labels may be used in these evaluations without entering this training objective.

```bash
python -m unittest discover -s tests -p 'test_style_alignment.py' -v
```

Tests cover one-view anatomical leakage invisible to GAP, collapse, independent
modality styles, gradients, actual renderer pairing/normalization/noise, cache and
split behavior, disabled defaults, and the real VQ-VAE training path. No trained
improvement is claimed by these implementation tests.
