# Optional style–anatomy independence penalty

Enable on a synthetic run with ground-truth `gt_latents.z_content`:

```yaml
scale_style_hsic_loss: 1.0
scale_style_contrastive_loss: 0.0
inject_style_to_decoder: true
```

Equivalent CLI: `--scale-style-hsic-loss 1 --scale-style-contrastive-loss 0 --inject-style-to-decoder`.
The HSIC weight defaults to zero. The example weight is a starting point, not calibrated
against reconstruction gradients. Both style losses are independently configurable;
setting the old style contrastive weight to zero replaces its cross-view repulsion.

The penalty is normalized, biased RBF-HSIC between the **entire ground-truth content
vector** and each modality's **full spatial style tensor**. It averages over views and
levels, then adds `scale_style_hsic_loss * hsic` directly to total loss, independently
of `scale_contrastive_loss` and reconstruction scheduling. Each spatial coordinate is
standardized across subjects before constructing a Gaussian kernel with detached
median squared-distance bandwidth. Standardization remains differentiable so simple
rescaling cannot reduce the penalty (above the numerical variance floor).

Style is collected after the configured spatial bottleneck and before quantization,
decoder detachment or dropout. Thus the loss reaches the style encoder even on steps
that skip decoding, and does not add codebook updates on those steps. It covers the
decoder-bound representation, including spatial patterns that cancel under GAP. The
normal model call retains its eight-tuple return and checkpoint format; training opts
into `(normal_output, style_features)` only when this flag is enabled.

Requires a nonempty style block, style injection, and at least four subjects per
training batch. Smaller final validation batches return zero with an explicit skip
diagnostic. Ground-truth content targets are detached; missing or non-finite targets
raise an error. Kernel calculations use float32 with autocast disabled.

TensorBoard/W&B diagnostics:

- `Style/hsic_L{level}_v{view}`: individual dependence scores.
- `Style/hsic`: mean unweighted penalty.
- `Style/hsic_weighted`: actual contribution to total loss.
- `Style/hsic_skipped_small_batch`: one when fewer than four subjects were supplied.

This is a **supervised diagnostic ablation**, not an unsupervised identifiability
guarantee. It changes neither the renderer nor data generation. A finite-sample biased
HSIC score has a positive independence floor; constant style also minimizes it.
Check held-out nonlinear style-to-content probes, style-factor recovery, and ventricle
reconstruction alongside content-to-ventricle decoding. A lower HSIC score alone does
not establish that anatomy moved into content or that useful style was preserved.

CPU verification (requires PyTorch):

```sh
python -m unittest discover -s tests -p test_style_hsic.py -v
```
