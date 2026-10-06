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

## Lesion branch

`--lesion-keypoints K` adds a branch that reports *where* things are as scalars,
`models/keypoint_pool.py`. Position-capable pooling is not enough on its own: in toy
runs, once an easy shared factor (brain size, a global gain) separates the subjects,
InfoNCE never learns the lesion, even when the pooled features could carry it. This is
feature suppression (Chen, Luo & Li, NeurIPS 2021). The branch gives the lesion its own
units and its own loss:

```text
a_k(p) = softmax_p(w_k . h_p + b_k [+ log brain occupancy])   one 1x1-conv detector per head
c_k    = sum_p a_k(p) p                                        expected (x, y, z) of head k
code   = [content (content_channels) | style | c_1 ... c_K]   3K content units after latent_dim
loss   = InfoNCE(content) + w * InfoNCE(lesion_projector(c))
```

- **Own InfoNCE.** The coordinates train through their own InfoNCE, via `lesion_projector`
  (3K -> `--lesion-proj-dim`). Probes read the raw coordinates. Mixed into the content
  readout instead, they were ignored in the toy. Before the projector, each coordinate is
  standardized over the batch (all subjects, both views). Untrained heads spread over most
  of the brain, so with a LayerNorm backbone the coordinates differed between subjects by
  ~7e-3. Every projected code then pointed almost the same way (mean cosine 0.9996), and
  the branch's InfoNCE sat at exactly chance.
- **`--lesion-frame brain` (default).** Heads only look inside the brain (weights
  proportional to exp(logit) times the brain occupancy from the input's nonzero support).
  Coordinates are relative to that brain's centroid and per-axis spread. A head on the
  boundary, or spread evenly over the brain, reports the same coordinate whatever the
  brain size. The generator places lesions relative to the white-matter extent, so
  lesion_x/y/z live in this frame. `grid` keeps absolute coordinates over the whole map.
- **`--lesion-norm layer`.** A per-voxel LayerNorm before the logits removes a per-voxel
  gain. In the toy it helped partly against a pure gain and hurt against a geometric
  factor, so it is off by default.
- **`--lesion-temperature T`.** Divides the keypoint logits (not the brain-occupancy term), so
  T < 1 makes the heads sharp. At the default T = 1 every head on the real recipe stayed spread
  over the whole brain (~16% of the map) through training. Its coordinate was then the brain's
  tissue centroid, which brain size, asymmetry and thickness all move: InfoNCE got subject
  identity without any head localizing anything. At initialization (LayerNorm backbone, ~818
  brain cells), each head covers 699 effective cells at T = 1, 412 at 0.05, 68 at 0.02 and 9 at 0.01.
- **`--lesion-pairing within_modality`.** The branch's own InfoNCE pairs two copies of the
  FLAIR view, each with its own random gain (0.7–1.3 about the brain mean), offset (±0.2
  brain SD) and noise (≤0.05 brain SD) inside the brain. Both go through the FLAIR encoder;
  the content block keeps its T1/FLAIR pairs. The reason: the lesion is darker than white
  matter in T1 (0.4 vs 0.8) and brighter in FLAIR (1.0 vs 0.4), and the two encoders start
  identical. So cross-modal alignment pushes the lesion's T1 and FLAIR codes apart and learns
  to suppress it. On a patch-loss checkpoint, cross-view patch retrieval at a subject's lesion
  position was 0.088, against 0.146 at the same positions in subjects without a lesion there.
  It costs one more FLAIR encoder pass per step.
- **`--lesion-input residual`.** The heads read each view's residual against a low-rank
  normative model (`models/normative_residual.py`) instead of the backbone map.
  - Before step 0, a per-view PCA is fitted on the first `--lesion-normative-subjects` (300)
    training subjects. Two thirds give the mean and `--lesion-normative-components` (20) modes;
    one third gives the per-voxel residual SD.
  - The residual z is split into brighter- and darker-than-normal channels and averaged to the
    feature grid.
  - The branch then shares nothing with the backbone. The content InfoNCE trains the encoders;
    the lesion InfoNCE trains only the heads' K × 2 weights and the projector.
  - Global shape is explained away before the heads see anything, and the residual's magnitude
    agrees across views where the lesion's sign does not.
  - Untrained, it already reads lesion x/y/z at about 0.45–0.7 R². Read the step-0 row before
    crediting training. Needs cross-modal pairs.
- **`--lesion-head-init`** (residual input only) sets the signs of the heads' initial weights:
  - `positive` (default): every head attends to anomalies.
  - `random`: plain random signs.
  - `negative`: every head avoids anomalies. The lesion floor is about 0–0.25, so this is the
    control for whether training finds the lesion by itself.
  - Each evaluation prints every head's logit per unit residual z [brighter, darker].
- **Use `--norm-type layer` for the backbone.** Even untrained, the branch already encodes
  brain size, because GroupNorm writes each subject's global statistics into every voxel
  and so rescales the keypoint logits like a gain. Measured on the conv_mlp recipe (240
  validation subjects, mean of 3 init seeds, linear R² T1/FLAIR):

  | backbone | branch norm | frame | brain_size |
  |---|---|---|---|
  | group | none | brain | 0.62 / 0.70 |
  | group | layer | brain | 0.50 / 0.56 |
  | group | none | grid | 0.71 / 0.76 |
  | layer | none | brain | 0.33 / 0.34 |
  | layer | none | grid | 0.69 / 0.71 |

  lesion_x/y/z start near 0 everywhere. The brain frame only helps once GroupNorm's gain
  path is gone. The ~0.33 that remains is shape change with brain size that a
  centroid-and-spread frame cannot remove.
- **Initialization.** The branch is built last, so every other weight matches the run
  without it. Its logits start small and random: zero would put every subject at the
  same point, and the branch's InfoNCE would have no gradient.
- **Reading a run.** Each evaluation prints the branch alone: linear R² of every content
  factor from the raw coordinates (T1 and FLAIR), plus each head's effective positions.
  lesion_x/y/z means the heads found the lesion. brain_size or ventricle_size means a
  factor that moves where attention lands captured them. Compare against the step-0 rows.
  - A head spread over ~16% of the map covers the whole brain. Its coordinate is then the
    brain's tissue centroid, which every global shape factor moves. That was the outcome of
    every real run so far.
  - Sharp heads (`--lesion-temperature 0.02`) locked onto anatomical edges instead.
- **content→view near 1 is mostly a constant per-view offset.** With `--cross-view-negs-only`,
  a shift shared by every vector of one view cancels in the softmax, so the loss never removes
  it. Subtract each view's mean before reading the view probe; its floor is ~0.36, not 0.5.
- **Backend check.** At low `--lesion-temperature`, CUDA's TF32 convolutions shift the keypoint
  coordinates by ~2e-3 against CPU. `run_encoder_mps.compare_backend` therefore scales their
  tolerance by 1/temperature, and skips the lesion loss, which is random under within-modality
  pairing.
- **Scoring.** The content mask marks the lesion units, so `compute_dci_synthetic`,
  `score_checkpoint.encode_blocks` and `checkpoint_lesion_analysis` count them as
  content. Patch and unpooled outputs have none. Readers that slice the global code by
  position, such as `encoder_generalization_audit`'s `[:, :content_channels]`, do not see them.

```sh
--lesion-keypoints 4 [--lesion-frame brain] [--lesion-norm none] [--lesion-proj-dim 8] [--lesion-loss-weight 1]
[--lesion-temperature 1] [--lesion-decorrelation-weight 0] [--lesion-pairing cross_modal]
[--lesion-input features] [--lesion-normative-components 20] [--lesion-normative-subjects 300]
[--lesion-head-init positive]
```

Locally: `scripts/run_encoder_mps.py --variant conv_mlp --lesion-keypoints 4` (run ID
gains `_lesionkp4`). Saved settings restore the branch in `eval.protocol.score_checkpoint`.
Results so far, the diagnoses behind them, and open threads are in `LESION_BRANCH_HANDOFF.md`
at the repo root.

## Normalization

`--norm-type` sets the Conv encoder's normalization. ResNet keeps `--resnet-norm`.

- `group` (default, and every run so far): GroupNorm in every block, with the
  largest group count up to 32 that divides the width. At the default widths that is
  one channel per group in the first downsampling block and inside each ReZero block
  (an instance norm), and two per group elsewhere. Each sample's statistics are
  pooled over all positions and written back into every cell. Background cells
  therefore carry whole-volume information such as brain size and cortical
  thickness, and the norm divides out a uniform scaling of its input.
- `layer`: `ChannelLayerNorm3d`, a LayerNorm over the channels at each voxel. No
  statistic is shared between positions. A cell whose receptive field misses the
  brain is identical for every subject, so the background stops carrying global
  factors. The VQ-VAE recipe already uses it; there, switching from GroupNorm took
  the coupling between brain and background features from 1.377 to 0.

```sh
--norm-type layer
```

- Norm layers draw no random numbers. A `layer` run's convolutions and head start
  identical to the `group` run with the same seed, so the two arms compare directly.
- Saved settings restore the norm in `eval.protocol.score_checkpoint`; runs saved
  before the flag existed load as `group`. The two norms' parameter names differ,
  so a checkpoint cannot load into the wrong one silently.
- `scripts/run_encoder_mps.py --norm-type layer` (and the wrappers) add
  `_layernorm` to the run ID. `python scripts/generate_conv_patch_slurm.py
  --norm-type layer` writes `encoder_conv_mlp_patch_layernorm_s42.slurm_bio.sh`.
- `eval.encoder.encoder_spatial_maps` reports each factor's readout in brain, edge
  and background cells, which is where the difference between the arms shows.
