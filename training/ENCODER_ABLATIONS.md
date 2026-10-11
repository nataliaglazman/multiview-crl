# Encoder architecture ablations

Three single-option interventions on the encoder-only trainer (`training.main_conv_synthetic`),
each with its own control. Every arm is an experiment YAML in `experiments/encoder/` and is
launched with `scripts/launch.py` on any cluster. The arms train from scratch with the usual
periodic validation. The post-training audit is separate (see below).

## Experiments

| Experiment | Architectural intervention | Compare against |
|---|---|---|
| `conv_mlp` | Conv backbone unchanged; the linear readout becomes GAP → Linear(64,100) → LeakyReLU → Linear(100,12) | `conv` |
| `resnet_groupnorm` | All 20 BatchNorm layers per backbone, including shortcuts, become 32-group GroupNorm | `resnet18` |
| `resnet_stride8` | The first-block strides in layer3 and layer4 go from 2 to 1; the final map is 8³ rather than 2³ | `resnet18` |

The stride-8 variant keeps the 7³ stride-2 stem convolution and the stride-2 max-pooling, the
kernel sizes, channels, depth and BatchNorm. It adds no dilation, so receptive fields and
computation change too. It tests late-stage downsampling, not whether the stem already loses
lesion information. Expect more activation memory than the original ResNet.

The Conv MLP is the **representation readout**, not an extra contrastive projection head.
InfoNCE still reads its first nine output units. Patch diagnostics apply each MLP after
averaging each spatial bin, as for ResNet.

Compare each variant with its own control, not two variants with each other. A test checks
that each variant's resolved command differs from its control's only in the intervention
(`tests/test_encoder_launch.py`).

## Shared recipe

`experiments/encoder/defaults.yaml` (model, objective, optimisation) and
`experiments/encoder/synthetic_defaults.yaml` (data, evaluation) hold the recipe of
`experiments/encoder_comparison.json`, which `scripts/compare_encoders.py` reads. A test keeps
the two equal.

- Synthetic paired T1/FLAIR at 64³; 2,000 training and 400 validation subjects.
- Independent content factors, clean-content mode, WM-contained sphere lesions, and
  fixed-reference input normalisation.
- Separate view backbones; 12 output units, nine content units, no extra projector.
- Batch 32 per view, InfoNCE temperature 0.1, AdamW at 1e-4, clipping at 2, 10,000 updates,
  validation every 2,000 updates, an untrained floor, final-step selection (`best_metric: none`).
- Seed 42 and data seed 42; the trainer derives model seed 42 and loader seed 10042. Input
  and subject-order hashes are recorded in `training_progress.json`.
- Deterministic algorithms with warning fallback for unsupported CUDA operations, and one
  numerical thread in every run. Warning mode does not guarantee exact weight reproduction.

## Launch

Preview first: `--dry-run` prints the resolved config and the exact command or script.

```bash
python scripts/launch.py experiments/encoder/conv_mlp.yaml --cluster slurm_bio --dry-run
```

```bash
for name in conv_mlp resnet_groupnorm resnet_stride8 conv resnet18; do
    python scripts/launch.py experiments/encoder/$name.yaml --cluster slurm_bio
done
```

`--cluster` takes `slurm` (CREATE `gpu` partition), `slurm_bio` (`biomed_a100_gpu`), `runai` or
`local`. Resources come from `experiments/cluster/<cluster>.yaml`. The SLURM scripts reuse the
prepared `multiview-env` and never install packages, so concurrent jobs cannot alter each
other's environment. `ENCODER_PYTHON` points a job at another interpreter.

On Run:ai, the submit host has no Python packages, so submit the generated scripts. Sync the
repository, including `experiments/generated/encoder/`, to the mounted checkout first:

```bash
bash experiments/generated/encoder/conv_mlp.runai.sh --dry-run
bash experiments/generated/encoder/conv_mlp.runai.sh
```

Each container checks CUDA before training. Its environment is exported inside the container
command, which avoids CLI-version differences between `--environment` and
`--environment-variable`.

### Other seeds

The YAML tags read `<name>_s{seed}`, so a seed override renames the run. The data seed stays
42, so a seed changes only the initialisation and the batch order:

```bash
for seed in 142 242; do
    python scripts/launch.py experiments/encoder/conv_mlp.yaml --cluster slurm_bio --set seed=$seed
done
```

On Run:ai, generate the script on a machine with Python, then sync and submit it:

```bash
python scripts/launch.py experiments/encoder/conv_mlp.yaml --cluster runai --generate --set seed=142
```

That writes `experiments/generated/encoder/conv_mlp_s142.runai.sh`. For a fresh attempt, give
the runs a new output root with `--set out_dir=results/encoder_v2/runs`. Run directories that
already exist are refused, and the jobs do not resume optimiser state. To change the recipe,
edit the YAMLs, which changes controls and interventions together. If stride 8 does not fit,
reduce the batch for every arm in a fresh experiment.

## Outputs

Runs land in `results/encoder/runs/<tag>` of the checkout the job runs in: `conv_mlp_s42`,
`resnet_groupnorm_s42`, and so on. The runs from before the YAML launcher are in
`results/encoder_ablations/runs` (Run:ai) and `results/encoder_ablations_slurm_bio/runs`
(slurm_bio). Each run saves settings, initial and final checkpoints, validation scores,
training progress and TensorBoard events. A submission also saves its resolved config, with
the git SHA, under `results/launches/<tag>/`.

## Evaluate completed checkpoints

Run these in the training environment with GPU access after each job completes (example: Conv MLP):

```bash
RUN=results/encoder/runs/conv_mlp_s42
python -m eval.encoder.encoder_generalization_audit \
  --run-dir "$RUN" --checkpoint model.pt \
  --num-samples 400 --probe-samples 400 --batch-size 4 \
  --retrieval-draws 8 --seed 1729 --skip-bn-recalibration \
  --out-dir "$RUN/evaluation/global_path"

python -m eval.protocol.score_checkpoint \
  --run-dir "$RUN" --checkpoint model.pt --num-samples 400 --batch-size 4 \
  --pooling gap --no-graph --no-dci \
  --lesion-analysis --lesion-grids 1 2 --lesion-shuffles 3 --lesion-seed 1729 \
  --out "$RUN/evaluation/validation_gap_lesions.json"

python -m eval.protocol.score_checkpoint \
  --run-dir "$RUN" --checkpoint model.pt --num-samples 400 --batch-size 4 \
  --pooling patch --patch-grid 2 2 2 --no-graph --no-dci \
  --out "$RUN/evaluation/validation_patch.json"
```

Use the same settings for all controls and variants. Keep the common 2³ patch grid for the
matched comparison; larger grids are extra diagnostics. The global audit keeps the encoders
frozen, fits probes on 300 validation subjects, tunes on 100 and scores 400 separate test
subjects. Compare the final content test R² per factor and per view; inspect the
backbone/hidden stages to locate drops.

These jobs do not create `compare_encoders.py` manifests or receipts, so use the audit commands
above rather than that tool's evaluate/summarize actions. Check `training_progress.json` for
completion at step 10,000 and for matching batch/input hashes across compared runs. Keep code,
software and GPU type fixed.
