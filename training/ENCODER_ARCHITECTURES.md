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
and restored by `eval.score_checkpoint`, including its lesion analysis. Omitting
`--synthetic-causal` keeps the SCM disabled regardless of the graph flags.

Settings are saved automatically. `eval.score_checkpoint` reconstructs both the
trained model and its untrained comparison from those settings. Old settings
without the new fields still load the original architecture strictly.

```sh
python -m eval.score_checkpoint --run-dir results/infonce_resnet18_3d --no-graph
```

Training always averages the backbone features **before** the nonlinear readout.
For optional patch diagnostics, the readout is applied after averaging each
spatial bin. At input resolution 64, `--pooling patch --patch-grid 2 2 2` is the
finest available grid; larger grids raise an error rather than duplicating
features. These patch vectors are diagnostic encodings, not parts whose average
must equal the global encoding. The same distinction applies to unpooled spatial
outputs: applying a nonlinear head and then averaging is generally different
from averaging first.
