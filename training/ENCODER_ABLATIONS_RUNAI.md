# Encoder architecture ablations on Run:ai

These scripts follow `experiments/generated/*.runai.sh`: one job per script,
`runai training standard submit`, the existing image/project/A100 resources,
the read-write `/nfs` mount, and a folded container command. They call
`training.main_conv_synthetic`, the encoder-only trainer. They train from scratch
and perform the usual periodic validation; they do not run the full post-training
audit or submit any other jobs.

## Experiments

| Generated file (seed 42) | Architectural intervention | Compare against |
|---|---|---|
| `encoder_conv_mlp_s42.runai.sh` | Conv backbone unchanged; replace linear readout with GAP → Linear(64,100) → LeakyReLU → Linear(100,12) | Native Conv |
| `encoder_resnet_groupnorm_s42.runai.sh` | Replace all 20 BatchNorm layers per backbone, including shortcuts, with 32-group GroupNorm | Native ResNet18 |
| `encoder_resnet_stride8_s42.runai.sh` | Change the first-block strides in layer3 and layer4 from 2 to 1; final map is 8³ rather than 2³ | Native ResNet18 |

The stride-8 variant retains the 7³ stride-2 stem convolution and stride-2
max-pooling. It preserves kernel sizes, channels, depth and BatchNorm. It adds no
dilation, so receptive fields and computation also change. This tests late-stage
downsampling, not the separate question of whether the stem already loses lesion
information. Expect greater activation memory than the original ResNet.

The Conv MLP is the **representation readout**, not an additional contrastive
projection head. InfoNCE still uses its first nine output coordinates. Patch
diagnostics apply each MLP after averaging each spatial bin, just as for ResNet.

Optional controls are already generated:
`encoder_conv_s42.runai.sh` and `encoder_resnet18_s42.runai.sh`.
Each variant should be compared to its own control, rather than interpreting
differences between two variants as a single architectural effect.

## Shared recipe

The generator reads `experiments/encoder_comparison.json`; interventions are in
`experiments/encoder_ablations/variants.yaml`. Current scripts embed:

- Synthetic paired T1/FLAIR at 64³; 2,000 training and 400 validation subjects.
- Independent content factors, clean-content mode, WM-contained sphere lesions,
  and fixed-reference input normalization.
- Separate view backbones; 12 output units, nine content units, no extra projector.
- Batch size 32 per view, InfoNCE temperature 0.1, AdamW at 1e-4, clipping at 2,
  10,000 updates, validation every 2,000 updates, an untrained floor, final-step
  selection (`best_metric=none`).
- Data seed 42, model seed 42 and loader seed 10042 for the supplied files.
  Input and subject-order hashes are recorded in `training_progress.json`.
- Deterministic algorithms with warning fallback for unsupported CUDA operations;
  one CPU numerical thread, identically across runs. Exact weight reproduction is
  not guaranteed by warning mode.

Resource defaults come from `experiments/cluster/runai.yaml`: project `nglazman`,
image `aicregistry:5000/nglazman:multiview-crl`, one A100, 16 requested CPU cores,
32 CPU limit, 64G requested RAM and 128G RAM limit. Numerical threads remain pinned
to one despite this allocation to preserve the paired rendering protocol.

## Submit

First sync the updated **repository code and generated scripts** to the mounted
repository at `/nfs/home/nglazman/crl-2/multiview-crl`. Copying only the shell files
is insufficient: the trainer, model, saved-checkpoint loader and audits now
support the new options. The mounted source is used inside the existing image.

Preview commands without submitting:

```bash
bash experiments/generated/encoder_conv_mlp_s42.runai.sh --dry-run
bash experiments/generated/encoder_resnet_groupnorm_s42.runai.sh --dry-run
bash experiments/generated/encoder_resnet_stride8_s42.runai.sh --dry-run
```

Submit the three interventions from a machine with the authenticated Run:ai CLI:

```bash
bash experiments/generated/encoder_conv_mlp_s42.runai.sh
bash experiments/generated/encoder_resnet_groupnorm_s42.runai.sh
bash experiments/generated/encoder_resnet_stride8_s42.runai.sh
```

These are three independent submissions, each requesting one GPU. The scheduler
may queue them. On startup, each container checks CUDA availability and runs the
determinism/max-pooling regression tests before starting training. W&B is not
used, so no W&B API key is needed. Environment variables are exported inside the
container command, avoiding CLI-version differences between `--environment`
(an environment asset in newer versions) and `--environment-variable`.
See the [Run:ai CLI reference](https://run-ai-docs.nvidia.com/self-hosted/reference/cli/runai/runai_training_standard_submit).

For controls trained under the same code and cluster environment:

```bash
bash experiments/generated/encoder_conv_s42.runai.sh
bash experiments/generated/encoder_resnet18_s42.runai.sh
```

Results are stored under the mounted repository:

```text
results/encoder_ablations/runs/
  conv_s42/                 # optional control
  resnet18_s42/             # optional control
  conv_mlp_s42/
  resnet_groupnorm_s42/
  resnet_stride8_s42/
```

Each run saves settings, initial and final checkpoints, validation scores,
training progress and TensorBoard events. Existing run directories are refused;
these jobs do not implement optimizer resume.

## Regenerate or repeat

Generation needs Python and PyYAML, but does not load PyTorch or submit jobs:

```bash
python scripts/generate_encoder_ablation_runai.py --include-controls
```

For all three configured seeds (15 scripts including controls):

```bash
python scripts/generate_encoder_ablation_runai.py \
  --seeds 42 142 242 --include-controls
```

Each seed changes initialization and shuffling; the dataset seed remains 42.
Regenerating writes shell scripts only. It does not restart jobs. Existing
scripts for other seeds remain in place, so submit explicitly chosen files.

For a fresh attempt, use new result directories **and** workload names:

```bash
python scripts/generate_encoder_ablation_runai.py \
  --include-controls \
  --results-dir results/encoder_ablations_v2 \
  --job-prefix encoder-ablation-v2
```

`--repo-path` overrides the container repository location. `--cluster-config`
can select another resource config. To change the training recipe, edit a copy
of the comparison JSON, pass `--config`, regenerate, and change it for controls
and interventions together. If stride 8 cannot fit, reduce batch size for every
arm in a fresh experiment. Keep the original logs.

## Evaluate completed checkpoints

Run these inside the project training environment with GPU access after each
training job completes (example: Conv MLP):

```bash
RUN=results/encoder_ablations/runs/conv_mlp_s42
python -m eval.encoder_generalization_audit \
  --run-dir "$RUN" --checkpoint model.pt \
  --num-samples 400 --probe-samples 400 --batch-size 4 \
  --retrieval-draws 8 --seed 1729 --skip-bn-recalibration \
  --out-dir "$RUN/evaluation/global_path"

python -m eval.score_checkpoint \
  --run-dir "$RUN" --checkpoint model.pt --num-samples 400 --batch-size 4 \
  --pooling gap --no-graph --no-dci \
  --lesion-analysis --lesion-grids 1 2 --lesion-shuffles 3 --lesion-seed 1729 \
  --out "$RUN/evaluation/validation_gap_lesions.json"

python -m eval.score_checkpoint \
  --run-dir "$RUN" --checkpoint model.pt --num-samples 400 --batch-size 4 \
  --pooling patch --patch-grid 2 2 2 --no-graph --no-dci \
  --out "$RUN/evaluation/validation_patch.json"
```

Use the same settings for all controls and variants. Keep the common 2³ patch
grid for the matched comparison; larger grids are additional diagnostics.
The global audit keeps encoders frozen, fits probes on 300 validation subjects,
tunes on 100 and scores on 400 separate test subjects. Compare final content
test R² per factor and per view; inspect backbone/hidden stages to locate drops.

These standalone jobs do not create `compare_encoders.py` manifests/receipts;
use the audit commands above rather than that launcher's evaluate/summarize
actions. Check `training_progress.json` for completion at step 10,000 and matching
batch/input hashes across compared runs. Keep code, software and GPU type fixed.

## Local checks

```bash
python -m unittest tests.test_encoder_ablations tests.test_encoder_ablation_runai \
  tests.test_resnet_encoder tests.test_encoder_pairing \
  tests.test_checkpoint_lesion_analysis tests.test_encoder_generalization_audit
```

Tests cover real forward/backward passes, normalization and stride interventions,
old checkpoint behavior, saved-model replay, global and patch extraction, and
shell submission argument preservation with a fake local Run:ai executable.
They do not measure A100 memory use or establish compatibility with a particular
installed cluster CLI version.
