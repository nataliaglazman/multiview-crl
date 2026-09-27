# Label-free cross-view style independence

The experimental `style_contrastive_mode: independence` penalizes dependence
between paired T1 and FLAIR style representations across subjects. It uses no
anatomy labels, masks as targets, or ground-truth latent factors. The legacy
`cosine` mode remains the default and the overall style weight remains zero.

Configuration keys:

```yaml
style_contrastive_mode: independence
style_independence_var_weight: 1.0
scale_style_contrastive_loss: 0.0  # Choose/calibrate a positive weight to enable.
scale_style_hsic_loss: 0.0        # Keep the separate supervised loss disabled.
inject_style_to_decoder: true
```

The implementation requires synthetic data, exactly two paired views, a nonempty
style block, and a configured training batch of at least four subjects. The
pseudo-MRI generator samples independent per-view acquisition latents, which
motivates this experiment. Prefer `synthetic_normalize: fixed_reference`:
`shared` normalization uses T1's subject-specific statistics for BOTH modalities
and can introduce legitimate cross-view acquisition dependence. Real paired
scans can share scanner/site effects, so synthetic-only validation is enforced.

## What is measured

The loss receives the full spatial style tensor **after the configured spatial
bottleneck and before style quantization, decoder detachment, or style dropout**.
It does not measure the final quantized decoder input. Pre-quantization scoring
keeps encoder gradients available even when reconstruction/codebooks are skipped.
The supervised HSIC loss uses the same existing model output but remains separate.

For each view, every coordinate is centered/scaled across subjects. RBF Gram
matrices use a detached median squared-distance bandwidth. Dependence is the
normalized ratio of Song et al.'s diagonal-removed HSIC estimator, computed in
float64 after float32 kernel construction:

`dependence = HSIC_u(K0, K1) / sqrt(HSIC_u(K0, K0) * HSIC_u(K1, K1))`

The numerator follows [Song et al. (2012), Eq. 5](https://www.jmlr.org/papers/volume13/song12a/song12a.pdf).
Normalization, data-adaptive kernels, and the positive-part clamp mean the final
loss is not itself an unbiased estimator of population HSIC. A finite-batch
signed score can be negative. Degenerate kernel normalizers are explicitly logged
and return zero dependence, not evidence of successful disentanglement.

Both dependence and anti-collapse use the **same** post-bottleneck style tensor.
For each channel, the variance term takes the RMS of deviations across subjects
and spatial coordinates, after removing each coordinate's across-subject mean.
A subject-invariant spatial pattern cannot satisfy it. A spatial pattern that
varies across subjects can satisfy it even when its GAP is zero. The per-view
hinge is `mean(relu(1 - channel_std))`; the two views' hinges are summed.

Per-level loss:

`max(dependence, 0) + style_independence_var_weight * variance_hinge`

These losses are summed across available style levels and multiplied once by
`scale_style_contrastive_loss`, independently of `scale_contrastive_loss`.
Changing the variance weight separates its effect from dependence minimization.
The hinge discourages collapse; it does not guarantee meaningful style, prevent
quantizer collapse, or provide a nonzero gradient at exact constant features.
Runtime batches below four subjects skip dependence; batches of two or three
still receive the hinge. Single-subject batches skip both and are logged.

## What it cannot guarantee

- Anatomy encoded in only one view's style can remain independent of the other
  view's style. Independent anatomical factors can also be split between views.
- Small/local signals can be difficult to detect in high-dimensional kernels.
  A low score is not proof that every anatomical factor is absent from style.
- Minimizing dependence can discard information instead of moving it to content.
  Reconstruction and content-path quality still matter.
- Variance can come from nuisance/noise. Preserved variance is not preserved
  acquisition information.
- Coordinatewise translation/rescaling invariance does not mean invariance to
  every affine transformation mixing channels.

The previously implemented anti-collapse hinge used earlier pooled encoder maps,
while dependence used post-bottleneck style. A constant post-bottleneck style
could therefore receive zero loss when the earlier maps varied. The revised
version measures both terms on the same representation and tests this failure.

## Diagnostics and evaluation

- `Style/xview_hsic_L*`: signed normalized dependence.
- `Style/xview_hsic_penalty_L*`: positive-part dependence penalty.
- `Style/xview_hsic_degenerate_L*`: degenerate normalization flag.
- `Style/var_hinge_L*` and `Style/var_hinge_weighted_L*`: unweighted and internally
  weighted anti-collapse terms.
- `Style/subject_std_v{0,1}_L*`: average channel variation at the measured tensor.
- `Style/independence_L*`: combined per-level loss.
- `Style/independence_weighted`: actual total contribution to the training loss.
- `Style/xview_hsic_skipped_small_batch_L*`: dependence not scored in this batch.

Choose a weight from measured gradient scales on an existing checkpoint, not by
equating raw HSIC to raw reconstruction loss. Before a long run, compare the
style score against subject-shuffled controls and inspect pre/post-quantization
style-factor recovery. A short ablation should assess held-out ventricular/lesion
routing, reconstruction fidelity, and content recovery alongside style recovery.
Ground-truth latents are appropriate for evaluation; they are not supplied to
this training objective.

CPU tests use the shipped VQVAE/training functions and independently check the
HSIC formula and gradients, constant/style-template controls, nonlinear shared
signals, the one-view leakage limitation, weight application, and skipped
reconstruction. No trained performance improvement is claimed from these tests.

```bash
python -m unittest discover -s tests -p 'test_style_hsic.py' -v
```
