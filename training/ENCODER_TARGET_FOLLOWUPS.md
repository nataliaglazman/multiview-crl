# Frozen spatial probes and supervised observability controls

These follow-ups address the three seed-42 SLURM encoder ablations. They answer:

1. Are lesion/sulcal targets recoverable from frozen spatial features even when
   they are poorly recovered from the global content vector?
2. Can a small network explicitly trained for these targets recover them from
   each individual synthetic image view?

The second experiment is **supervised from scratch**, not fine-tuning of the
contrastive encoders and not an unsupervised identifiability result. It is a
positive observability control. Success establishes that a learner can extract
the target under this protocol. Failure does not prove the target is impossible;
optimization, sample size, and this control architecture can still limit it.

## Submit on slurm_bio

Copy the new Python files and generated scripts to the same cluster checkout
that holds the completed runs. Both arrays use the existing `multiview-env`
environment and the same A100 partition/resources as the training scripts.
No packages are installed, no original checkpoint is overwritten, and no
comparison manifest is rewritten.

From the repository root on the cluster:

```bash
# Read-only previews of task 0 in each array; no SLURM allocation required.
bash experiments/generated/encoder_spatial_probes_s42.slurm_bio.sh --dry-run
bash experiments/generated/encoder_target_controls_s42.slurm_bio.sh --dry-run

# Three frozen-audit tasks, one per existing encoder checkpoint.
sbatch experiments/generated/encoder_spatial_probes_s42.slurm_bio.sh

# Two fresh supervised tasks, one T1 and one FLAIR.
sbatch experiments/generated/encoder_target_controls_s42.slurm_bio.sh
```

These commands request five independent one-GPU tasks in total. They can run in
either order; the supervised control does not depend on audit completion. To
limit simultaneous GPUs, submit with `sbatch --array=0-2%1 ...spatial_probes...`
or `sbatch --array=0-1%1 ...target_controls...`, using the full filenames above.

Defaults:

| Setting | Value |
|---|---|
| Frozen array 0 / 1 / 2 | `conv_mlp_s42` / `resnet_groupnorm_s42` / `resnet_stride8_s42` |
| Supervised array 0 / 1 | `t1` / `flair` |
| Original runs | `results/encoder_ablations_slurm_bio/runs/` |
| New results | `/scratch/users/k24058220/encoder_followups_slurm_bio/` |
| Supervised data recipe | `conv_mlp_s42/settings.json` (same data settings in all three supplied logs) |
| Python | `$HOME/.conda/envs/multiview-env/bin/python` |
| Resources per task | One A100, 8 CPUs, 64G host RAM, 48h allocation |

Override `ENCODER_REPO`, `ENCODER_PYTHON`, `ENCODER_RUNS`, `ENCODER_FOLLOWUPS`, or
`ENCODER_REFERENCE_RUN` before submission if the locations differ. For example:

```bash
export ENCODER_RUNS="$PWD/results/encoder_ablations_slurm_bio/runs"
export ENCODER_PYTHON="$HOME/.conda/envs/multiview-env/bin/python"
export ENCODER_FOLLOWUPS="/scratch/users/k24058220/encoder_followups_slurm_bio"
ENCODER_TASK_ID=2 bash experiments/generated/encoder_spatial_probes_s42.slurm_bio.sh --dry-run
```

The scratch output root contains `spatial/` for frozen probes and `supervised/`
for fresh controls, including all feature/image caches, reports, predictions,
and new control checkpoints. Output directories are created automatically.
Existing checkpoints are still read from `ENCODER_RUNS`; existing results are
not moved. An exported `ENCODER_FOLLOWUPS` overrides the scratch default, so
update it if your shell still has the old location. Changing the scripts affects
new launches only, not jobs already submitted or running.

Every output directory includes job and array-task identifiers. Existing output
directories are refused; these scripts do not resume interrupted jobs. Logs go
to `/scratch/users/$USER/` with array job/task IDs in their filenames. That scratch
directory must exist before submission.

The frozen scripts require `settings.json`, `model.pt`, and `model_init.pt` in
each original run. The supervised jobs require only the reference `settings.json`.
Adding Python evaluators changes the repository source inventory; these standalone
follow-ups deliberately have their own provenance records and do not use the old
`compare_encoders.py` manifest gate. Preserve the original manifests unchanged.

## Experiment 1: frozen features

Entrypoint: `python -m eval.encoder.encoder_spatial_target_audit`.

- Evaluate **both T1 and FLAIR**, with models strictly in evaluation mode and
  gradients disabled. Check both registered model state and checkpoint-file hashes.
- Compare GAP, a 2×2×2 grid, and the complete native spatial grid: 16³ for Conv,
  2³ for GroupNorm ResNet, 8³ for stride-8 ResNet. Duplicate grids are evaluated once.
- Probe backbone features, projected content coordinates, and remaining style
  coordinates. The actual global MLP hidden output is included at grid 1.
- Spatial MLP outputs apply the readout independently after averaging each bin;
  they are diagnostic features, not the global vector optimized during training.
- Fit **ridge and RBF kernel** readouts. Select regularization/bandwidth using
  probe-tuning subjects only. Compare trained and saved initial weights, plus
  separately shuffled fit/tune labels; test labels never select probes.
- Use the original 400 validation subjects: 300 for probe fitting and 100 for
  hyperparameter selection. Evaluate on 400 independent test subjects. All
  feature and target standardization is fitted on the 300 fitting subjects only.
- Score all nine original latent factors, three physical centroid coordinates,
  signed sulcal corrugation amplitude, and its absolute magnitude.

The full native maps are wide. Features are stored as float32 `.npy` memory maps,
and linear kernels use feature chunks. Allow roughly 4–5 GB of output disk for
each Conv/stride-8 audit with initial controls; exact use depends on architecture.
High-dimensional native probes also consume more CPU time than coarse grids.
Feature counts are recorded: this is a within-model readout diagnostic, not a
dimension-matched comparison between architectures.

For a cheaper first pass, omit `--include-native` in direct commands, or regenerate
both shell scripts with `python scripts/generate_encoder_followups_slurm.py --no-native`.

Inspect `summary.csv` first. It reports mean xyz scores for raw lesion controls
and physical centroids, plus separate raw/signed/magnitude sulcal scores. The
complete per-factor results are in `probes.csv` and `report.json`; predictions
are retained in `trained_predictions.npz` and `initial_predictions.npz`.

## Experiment 2: supervised controls

Entrypoint: `python -m training.encoder_target_control`.

Each image view gets an independent, freshly initialized small 3D CNN:

- Full 64³ image input; no ground-truth crop, mask, or coordinates enter the network.
- Stride-2 feature map, with a heatmap head supervised by rendered lesion occupancy.
  Softmax over space and a coordinate expectation produce physical centroids.
- A separate spatial readout pools features to an 8³ grid and flattens it before
  regression. It predicts the three original lesion controls, the original sulcal
  latent, and the renderer's signed sulcal amplitude.
- Sulcal magnitude is evaluated as the absolute predicted amplitude; sign accuracy
  is also reported. A network that predicts roughness but misses sign should not
  be reported as recovering the signed latent.
- Loss is lesion-distribution cross-entropy plus the mean squared standardized
  regression error, each with weight 1. Target scales come from training labels only.
- Use the original 2,000 training and 400 validation subjects and a separate 400
  test subjects. Float32 rendered inputs/targets are cached in each NEW output
  directory. Each supervised task uses roughly 3–4 GB of disk with these defaults.
- AdamW, learning rate 1e-3, cosine decay, batch 8, width 24, 2,000 updates, model
  seed 42. Validation is logged every 500 updates. The final step is prespecified;
  test scores are computed once after training, not used to choose a checkpoint.
- Optional `--shuffle-targets` permutes paired training labels/support maps as a
  negative-control run. Use a separate new output directory for it.

The control preserves the original generator and its split-specific fixed-reference
image normalization. These numbers use physical centroids in renderer coordinates;
localization errors also appear in voxels and as the fraction within one lesion
radius. The renderer quantizes lesion placement, so physical localization and
recovery of the continuous generator coordinates are distinct endpoints.

Outputs include `report.json`, `test_scores.csv`, `test_predictions.npz`, cached
data, and a new control `model.pt` containing model weights and target scaling.
This file is separate from the original encoder checkpoint.

## Experiment 3: supervised GAP control for signed sulcal

Question: does the GAP score for `sulcal_widening` stay at the floor because GAP
*cannot* carry the sign of a zero-mean corrugation, or only because no
objective asks for it? Flipping the sign of z8 shifts `sin(12x)sin(12y)sin(12z)`
by half a period. The spatial mean of a translation-equivariant feature map is
unchanged by that shift, so GAP is sign-blind up to the leaks that break
equivariance: the brain mask, the volume boundary, and zero padding. A network
trained end to end, with the sign as its explicit target, is the strongest
attempt to exploit those leaks.

The three arms use `training.encoder_target_control` with matched features,
budget and seed. Every arm uses `--lesion-weight 0` (features trained for
regression only), `--readout-channels 24` (so the GAP arm pools 24 nonlinear
channels, not 4) and `--magnitude-head`:

| Arm | Readout | Role |
|---|---|---|
| `gap` | `--grid 1` | The test |
| `grid8` | `--grid 8` | Positive control: the same network can read the sign when position survives pooling |
| `gap_shuffled` | `--grid 1 --shuffle-targets` | Floor |

```bash
python scripts/generate_encoder_followups_slurm.py   # writes encoder_gap_controls_s42.slurm_bio.sh (6 tasks: t1/flair × 3 arms)
sbatch experiments/generated/encoder_gap_controls_s42.slurm_bio.sh
```

The default is 6,000 steps (`--gap-control-steps`), longer than the 8³ control,
because pooled readouts can sit on a plateau before escaping. Read
`history[*].validation` in `report.json`: a GAP sulcal R² still rising at the
final step makes the result inconclusive, not negative.

The reference run `conv_mlp_s42` uses `synthetic_clean_content=True` and
`synthetic_causal=False`. This removes both other explanations for a low GAP
score: the nuisance deformation field competing at the same spatial frequency,
and sulcal signal inherited from its parents in the causal graph. A
nuisance-on or SCM reference run answers a different question.

| `gap` result | Interpretation |
|---|---|
| `sulcal_amplitude` ≈ shuffled, sign accuracy ≈ 0.5, `sulcal_magnitude_head` well above shuffled | GAP is sign-blind by construction; the magnitude head shows the arm trained and sees the corrugation |
| Signed and magnitude both ≈ shuffled | Inconclusive: the arm may simply have failed to train. Check `grid8` and the loss history |
| Signed clearly above shuffled | Equivariance leaks carry the sign; a flat GAP under a contrastive objective was the objective's choice, not a limit of pooling |
| `grid8` signed ≈ shuffled | The control network failed; the GAP arm says nothing |

The derived `sulcal_magnitude` row (|signed prediction|) is kept for continuity.
Under GAP it reads low whenever the sign head collapses to ~0, so judge
magnitude by `sulcal_magnitude_head`. Lesion centroid rows are meaningless at
`--lesion-weight 0`, but the regressed `lesion_x/y/z` give a free GAP
replication of the lesion-position cancellation.

## How to interpret the pair of experiments

| Finding | Supported interpretation |
|---|---|
| Spatial features outperform GAP on held-out targets | The frozen spatial representation retains useful information absent or less accessible after pooling |
| Backbone probes outperform the final content vector | The readout/content restriction reduces accessibility to these probes |
| RBF succeeds where ridge fails | Some information is present but not readily linearly decoded |
| Physical centroids succeed but original lesion latents fail | Separate localization from the anatomy-dependent/quantized generator parameterization |
| Sulcal magnitude succeeds but signed amplitude fails | Inspect sign/phase sensitivity rather than claiming complete sulcal recovery |
| Supervised control succeeds but frozen probes fail | Images support prediction; the learned representation or finite probes limit recovery |
| Supervised control also fails | Check optimization, model capacity, rendering visibility and target definition; do not conclude non-identifiability |

Compare observed rows against both initial and shuffled controls. One model seed
does not establish robustness. The two supervised jobs share a data recipe rather
than comparing the native Conv and ResNet architectures. Neither experiment proves
that target information is present in every representation or fully identifiable.

## Local verification

```bash
python -m unittest tests.test_encoder_target_followups -v
```

Tests include real CPU forwards through Conv and both ResNet variants, complete
small synthetic frozen/supervised workflows, checkpoint immutability, native-grid
extraction, physical-coordinate conversion, both-head gradient updates, absence
of test access during optimization, invariance of fitted probe predictions to test
label changes, and previews of all eleven cluster tasks (including that the GAP-control arms differ only in pooling and label order). Full experiments require
the original cluster files and have not been run by these checks.
