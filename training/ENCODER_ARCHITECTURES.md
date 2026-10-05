# Encoder-only architecture comparison

`training.main_conv_synthetic` defaults to the existing `conv` architecture.
Add these flags to an existing command, using a new model ID:

```sh
--model-id infonce_resnet18_3d \
--encoder-architecture resnet18 \
--encoder-head-hidden 100 \
--contrastive-proj-dim 0 \
--eval-pooling gap
```

The new encoder follows the image encoder in
[upstream main_multimodal.py](https://github.com/CausalLearningAI/multiview-crl/blob/main/main_multimodal.py#L437-L442),
with 2D operators replaced by 3D operators and one input channel:

```text
Conv 7, stride 2 -> BatchNorm -> ReLU -> MaxPool 3, stride 2
BasicBlock stages: [2, 2, 2, 2], widths [64, 128, 256, 512]
stage strides: [1, 2, 2, 2]
GAP -> Linear(512, 100) -> LeakyReLU(0.01) -> Linear(100, latent_dim)
```

It uses ordinary ResNet residual additions, BatchNorm with running statistics,
and torchvision's default convolution/BatchNorm initialization. Weights are
randomly initialized; no pretrained images or labels are used. This is a
volumetric adaptation, not the original 2D model or torchvision's video ResNet.

`--hidden-channels`, `--res-channels`, `--nb-res-layers`, and `--downscale-factor`
only configure `conv`. ResNet has fixed widths and stride 32: a 64³ input yields
a 512-channel 2³ feature map. Its compute and memory requirements are much larger
than the old backbone. BatchNorm in training needs more than one value per
channel: with 32³ inputs, use at least two subjects per separate view backbone.

The architecture flag preserves the current separate-view-backbone setting.
Add `--no-separate-encoders` to also match upstream's use of one image encoder
for all image views. The dense readout is shared in either case. Keep
`--contrastive-proj-dim 0` to apply the loss directly to the scored content
encoding, as upstream does; this optional loss-only projector is separate from
the ResNet readout, which always runs. The content/style split, temperature,
negative sampling, optimizer, and data settings retain their existing behavior.
Changing the architecture alone is not a reproduction of the entire upstream
training recipe.

To use the same random causal graph configuration as the earlier VQ-VAE YAML,
add these dataset flags with either encoder architecture:

```sh
--synthetic-causal \
--synthetic-causal-graph random \
--synthetic-causal-edge-prob 0.5
```

Graph choices are `chain` (default), `full`, and `random`. The edge probability
defaults to 0.5 and only affects `random`; each permitted edge from an earlier
content index to a later index is sampled independently. The run's `--seed`
determines the graph and mechanism weights, which are shared across train,
validation and test splits. Use the same seed and content-factor count when
matching a previous run's graph. These settings are recorded in `settings.json`
and restored by `eval.protocol.score_checkpoint`, including its lesion analysis. Omitting
`--synthetic-causal` keeps the SCM disabled regardless of the graph flags.

Settings are saved automatically. `eval.protocol.score_checkpoint` reconstructs both the
trained model and its untrained comparison from those settings. Old settings
without the new fields still load the original architecture strictly.

```sh
python -m eval.protocol.score_checkpoint --run-dir results/infonce_resnet18_3d --no-graph
```

Training always pools the backbone features **before** the nonlinear readout:
by averaging, or with `--global-pool attention` (below) by attention pooling.
For optional patch diagnostics, the readout is applied after averaging each
spatial bin. At input resolution 64, `--pooling patch --patch-grid 2 2 2` is the
finest available grid; larger grids raise an error rather than duplicating
features. These patch vectors are diagnostic encodings, not parts whose average
must equal the global encoding. The same distinction applies to unpooled spatial
outputs: applying a nonlinear head and then averaging is generally different
from averaging first.

To diagnose poor factor recovery without retraining, run the
[encoder generalization audit](../eval/encoder/ENCODER_GENERALIZATION_AUDIT.md). It combines
training/test cross-view retrieval, ridge/RBF probes through the readout, and
training-only BatchNorm recalibration on a disposable model copy.

## Attention pooling

`--global-pool attention` replaces the GAP in front of the global readout (either
architecture, linear or MLP readout) with multi-head attention pooling,
`models/attention_pool.py`. GAP of translation-equivariant features cancels a
structure that only moves: deleting a fixed-size lesion in one place and adding it
elsewhere leaves the average unchanged. It also dilutes compact signals with the
rest of the volume. Attention pooling weights positions instead:

```text
x_p    = h_p + W_pos phi(p)                 phi: fixed Fourier features of the cell centre
a_m(p) = softmax_p(q_m . LayerNorm(x_p))    one learned query per head
out_m  = sum_p a_m(p) x_p[channels of m]    each head pools its own channel slice
```

The positional term enters the values as well as the keys. A head that locks onto a
structure therefore reports where it is (a soft argmax), which content-only
attention cannot. The output has the backbone's width, so the readout that follows
is unchanged.

`q` and `W_pos` start at zero. The pool is then exact GAP up to float rounding, and
building it draws no random numbers. Every other weight, and the untrained floor,
therefore match the GAP run with the same seed, so learned deltas of the two arms
are directly comparable. Each evaluation prints every head's effective positions
as a fraction of the map, per view: 1 is GAP, 1/N is a single position. The step-0
row is 1 by construction. `model.attention_maps(x, n_views=2)` returns the
`(B, heads, d, h, w)` maps for plotting.

```sh
--global-pool attention \
--attention-pool-heads 4 \
--attention-pool-frequencies 4
```

- `--attention-pool-heads` must divide the backbone channels (`--hidden-channels`
  for conv, 512 for ResNet).
- `--attention-pool-frequencies 0` removes the positional encoding. Attention is
  then permutation-invariant over positions, like GAP. Use it as the matched
  control for whether position is what helps.
- The backbone map needs more than one position. ResNet at stride 32 and 64³ has
  only 2³; stride 8 gives 8³, and conv at downscale 4 gives 16³.
- Patch readouts, patch training and the unpooled map still average each bin. Patch
  outputs are diagnostic encodings, not parts of the global one.
- The pool starts where GAP is. At conv widths with one channel per GroupNorm group
  (e.g. 8, 16 or 32), the GAP vector is ~0 for every subject at initialization. The
  global loss's gradient is then float noise in either arm. Width 64 is not affected.
- Saved settings restore the pool in `eval.protocol.score_checkpoint`. The
  generalization audit's backbone stage reads the pool's output. Locally,
  `scripts/run_encoder_mps.py --global-pool attention` adds it to a recipe variant.
  The Run:ai/SLURM variant files have no attention variant yet.
