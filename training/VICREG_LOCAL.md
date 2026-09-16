# Registered local/global VICReg

Select `contrastive_loss_type: vicregl`. This is a VICRegL-inspired adaptation for
registered volumes, not a reproduction of the original paper. It pairs identical
patch positions, uses independent local/global projection heads, and needs no
anatomical labels, segmentation targets or feature-nearest-neighbour matching.
The reconstruction, quantization and style pathways are unchanged.

## What changes

The incoming content tensor is `(views, subjects, channels, positions)` from the
existing encoder patch pooling. Current folded VICReg treats subject-position
pairs as its statistical samples. This objective instead computes a separate
covariance matrix and variance vector across subjects **at every position**.
Covariance penalties are squared before averaging over positions, so correlations
of opposite signs at different positions cannot cancel one another.

For each registered view pair and eligible position, the local terms are:

- Mean squared difference of the raw projected embeddings.
- Mean `relu(1 - sqrt(sample_variance + 1e-4))`, averaged over the two views.
- Sum of squared off-diagonal sample covariances divided by the channel count,
  averaged over the two views.

The three scalars are averaged over eligible positions. Variance and covariance
subtract each position's across-subject mean; the alignment inputs are **not**
batch-centered, whitened, or L2-normalized. Spatial anatomy shared identically by
all subjects cannot satisfy the local variance requirement.

With foreground masking, training still selects the batch union of foreground
positions. This loss additionally uses per-subject foreground validity: a local
pair participates only where both modalities meet the threshold. A position needs
at least two valid subjects. Empty/ineligible positions contribute no loss and
coverage is logged. With masking disabled, every subject-position is eligible.

The global arm performs foreground-weighted GAP on the selected content features
**before** its own head, producing one vector per subject. It applies the same
three terms across subjects. It is independent of `bt_gap_pooling`, which is not
used by this objective. No BT correlation EMA is used. All loss statistics use
FP32 even with AMP; projection MLPs follow the outer autocast setting.

## Projection heads

Each arm has a shared-across-modalities `Linear → ReLU → Linear` MLP, with no final
normalization or BatchNorm. Local and global heads have separate parameters. The
local head acts pointwise and does not mix positions. Both receive gradients, as
does the encoder; reconstruction and probes continue reading pre-head features.
Heads are attached before the optimizer/parallel wrapper, included in checkpoints,
and restored by the shared synthetic evaluation loader. Like the existing generic
contrastive heads, their parameters are excluded from weight decay.

The heads give the two objectives independent loss-facing feature spaces. They
may reduce conflict with reconstruction, but could also discard a weak factor.
Set `vicregl_no_projectors: true` (CLI `--vicregl-no-projectors`) for the direct
content-feature ablation. This installs identity heads with no parameters. Output
dimension and hidden-width flags then have no effect on feature width.

## Configuration

```yaml
contrastive_loss_type: vicregl
patch_contrastive: true
patch_grid: [8, 8, 8]
patch_foreground_mask: true
patch_center_mode: none
contrastive_proj_dim: 0
contrastive_proj_mode: head
use_moco: false

vicregl_local_weight: 1.0
vicregl_global_weight: 0.25
vicregl_local_dim: 16
vicregl_global_dim: 16
vicregl_hidden: 64
vicregl_no_projectors: false
vicreg_sim_coeff: 25.0
vicreg_std_coeff: 25.0
vicreg_cov_coeff: 1.0
scale_contrastive_loss: 1.0
```

These are starting values, not coefficients calibrated against reconstruction.
Do not automatically carry over BT's `scale_contrastive_loss: 100`. Total loss is
the existing reconstruction/VQ terms plus `scale_contrastive_loss` times
`local_weight*local_loss + global_weight*global_loss`, with the existing level
weight. Both VICReg arms use the `vicreg_*_coeff` flags; all `bt_*` loss flags are
irrelevant. The variance coefficient must be positive. An arm can have zero
weight, but both cannot be disabled. Disabled arms do not create learned heads.

The initial implementation requires one VQ level with a fixed content/style split,
at least two subjects per batch, registered corresponding grids, no MoCo, and no
generic contrastive projector. Non-`none` `patch_center_mode` is rejected to avoid
silently changing alignment semantics. Multi-step accumulation does not increase
the number of subjects used for a covariance estimate. The training loader already
drops incomplete batches; direct callers must supply at least two subjects.

`experiments/synthetic_causal_vicregl.yaml` supplies a 5,000-step example. Review its
resolved settings before using it as a matched comparison; synthetic defaults
also supply encoder/style flags. To preserve the exact architecture and generator
of an existing experiment, apply the block above to a copy of that experiment and
give it a new run name. Start fresh rather than resuming an incompatible BT state.

```bash
python scripts/launch.py experiments/synthetic_causal_vicregl.yaml --cluster local --dry-run
```

For the heads-off ablation use a separate run name and
`--set vicregl_no_projectors=True tag=synthetic-causal-vicregl-noheads-12-4`.

## Diagnostics and limits

TensorBoard keys under `Contrastive/` include
`vicregl_local_sim_L0`, `vicregl_local_var_L0`, `vicregl_local_cov_L0`, and the
corresponding `global` keys. Each arm reports mean std, eligible fraction, loss
and weighted loss. The weighted arm values include the arm weight; the outer
`scale_contrastive_loss` is applied by the trainer. Check coverage before interpreting
low loss, and watch for covariance dominating or persistently collapsed features.

This changes the regularization units, not the semantic factors the model must
retain. The MSE term still averages over positions: rare structures can remain
weak, and unrelated subject variation at the same position can satisfy variance.
It does not make weak input contrast observable, undo encoder downsampling,
guarantee content use by the decoder, or prevent style leakage. Compare both
content probes and decoder routing while checking joint reconstruction fidelity.
Keep grid size and style capacity fixed in the first loss comparison.

Tests cover subject/position statistics, covariance cancellation, masks and empty
support, offsets, constant-feature gradients, AMP, separate and disabled heads,
configuration checks, a real VQVAE optimization step without labels, and strict
checkpoint round trips:

```bash
python -m unittest discover -s tests -p 'test_vicreg_local.py' -v
```
