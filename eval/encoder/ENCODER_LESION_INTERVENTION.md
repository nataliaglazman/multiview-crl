# Frozen encoder lesion-movement test

This audit tests whether an encoder responds to the lesion itself and whether
its position probe follows a controlled move. It supports the Conv/ResNet
encoder-only checkpoints, including the separate spatial readout. It does not
retrain the encoder or select a checkpoint. Labels fit diagnostic probes only.

## Run on the NVIDIA PC

After copying the updated `eval/encoder/encoder_lesion_intervention.py` and other
current source to the PC, activate `monai_env` and run from the repository root:

```bash
RUN="$PWD/results/encoder_patch_cuda/runs/conv_mlp_s42_cuda_patch8x8x8_w1_separate_spatial"
OUT="$RUN/evaluation/lesion_moves_$(date +%Y%m%d_%H%M%S)"

python -m eval.encoder.encoder_lesion_intervention \
  --run-dir "$RUN" --checkpoint model.pt \
  --out-dir "$OUT" --device cuda \
  --num-samples 64 --subject-offset 1000 \
  --axes x y z --eps 0.5 --grids 1 8 --batch-size 2
```

Use the actual run directory if it has batch/step/lesion-radius suffixes. Use a
finished checkpoint or a stable checkpoint copy: the audit checks file hashes
before/after loading and at completion and refuses a moving source. A named
copy in the run directory can be selected with `--checkpoint NAME.pt`.

The default also evaluates `model_init.pt`, with separately fitted probes, on
exactly the same validation images and intervention pairs. Use `--skip-initial`
only when that saved checkpoint is unavailable or for a quicker first check.
It never substitutes a newly randomized model for the saved initialization.

For a quick test, use `--num-samples 8 --axes x`. This reduces intervention work,
but still uses the run's complete original validation cohort for probe fitting.
For MPS use `--device mps` with `PYTORCH_ENABLE_MPS_FALLBACK=1`; CPU is supported
with `--device cpu`. `--batch-size` counts pairs, so two pairs mean four images
per view during intervention extraction. Lower it to 1 if memory is tight.

Add `--include-native` to compare the native map with the pooled 8³ map; this
increases temporary disk and RAM use. Grids must fit the actual feature map:
a stride-32 ResNet at resolution 64 needs `--grids 1 2` instead of `1 8`.

On SLURM, invoke the same command inside an allocated GPU job with `RUN` and
`OUT` under `/scratch/users/k24058220/...`. Temporary feature banks are created
under `OUT` and removed on success or failure. The command itself does not submit
a job. The output directory must be new.

## What is held fixed

Each test subject gets three pairs by default. A pair changes one raw lesion
control to its original value minus/plus `eps`. All other content factors,
deformation/fissure fields, acquisition parameters, random noise draws, and the
original image-normalization affine stay fixed. The renderer verifies identical
anatomical tissue maps and foreground masks, and checks that image changes are
confined to the changed lesion support plus its 3-voxel smoothing boundary.

The audit requires `wm_interior` sphere placement: the complete lesion remains
inside white matter. Conditional placement means changing one latent control can
move more than one Cartesian coordinate. Scores therefore use the full actual
3-D mask-centroid displacement. Voxelization or saturated controls may produce
no movement; these pairs remain in the output, are counted, and are excluded from
movement metrics. They are never silently redrawn or replaced by larger moves.

The test subject IDs default to 1000–1063, separate from the original validation
split and the usual first 400 monitoring subjects. The test normalizer uses the
unchanged first 64 observations of its split; reducing the intervention count
does not change this reference. Per-sample/shared normalization, when present
in the saved run, is frozen to each original subject before moving the lesion.

## Features and probes

Both T1 and FLAIR are evaluated independently:

- `backbone`: features before the content readout, pooled to the requested grid.
- `projected`, grid 1: the actual global content vector.
- `projected`, larger grids: the active spatial content readout, including the
  separate head when enabled. Spatial order is retained when flattening.

Ridge and RBF probes fit on 75% of the run's original validation subjects and
select regularization/bandwidth on the remaining 25%. Feature/target scaling
uses only the fitting subset. These probes predict physical centroids in renderer
coordinates `[-1,1]`; voxel errors use `(resolution - 1) / 2` for conversion.
No moved-lesion labels enter fitting or tuning. Shuffled-label controls permute
the observational fitting and tuning labels separately.

For each representation, the audit also re-encodes identical endpoint images
in the same batch arrangement to measure numerical replay variation. The model
stays in evaluation/inference mode; parameter and buffer hashes must be unchanged.

## Reading the output

Let `d = centroid_B - centroid_A` and `p = prediction_B - prediction_A`.

| Metric | Meaning |
|---|---|
| `movement_skill` | `1 - sum(||p-d||²) / sum(||d||²)`: 1 is perfect, 0 is predicting no movement, negative is worse. This is not ordinary centered R². |
| `movement_gain` | `sum(p·d) / sum(||d||²)`: 1 follows the true displacement, 0 has no component along it, negative follows the opposite direction. Read with skill/error because orthogonal errors can coexist with gain 1. |
| `movement_rmse_vox` | RMS Euclidean error of predicted displacement, in voxels. |
| `endpoint_mean_r2` | Mean per-coordinate position R² across endpoints. This can be high from anatomy alone, even when movement skill is zero. |
| `feature_delta_rms` | RMS feature change between the two lesion positions; its units depend on the representation. |
| `delta_to_subject_variation` | Feature-change RMS divided by the RMS feature standard deviation across probe-fit subjects. A calibrated sensitivity measure, not an information score. |
| `replay_delta_rms` | RMS feature difference when identical endpoint inputs are encoded again. Compare against lesion-induced response to assess numerical scale. |

The console shows observed-label scores pooled over intervention axes. CSV/JSON
also include each axis, shuffled controls, and initialization. The movement-skill
interval resamples subjects with all their axes together; it describes evaluation
subject variation for fixed probes/checkpoints, not variation across training seeds.

- Backbone response and tracking, followed by weak head tracking, locate a
  recoverability gap near the readout.
- Feature response without successful tracking means sensitivity is present but
  these fitted probes do not reliably decode displacement.
- Good endpoint R² with near-zero movement skill is consistent with anatomical
  proxies or poor transfer of the observational probe to controlled moves.
- Responses close to numerical replay levels suggest weak sensitivity in the
  tested representation; this alone does not prove that all lesion information
  has disappeared.

These interventions isolate an image factor. For causal/hierarchical data, other
factors are held fixed rather than propagated through a generative causal graph;
the resulting pairs need not follow the original joint distribution.

## Small files to keep or copy back

```text
report.json                protocol, cohort hashes, metrics and settings
summary.csv                movement metrics by stage, probe, view and axis
sensitivity.csv            per-pair feature responses and replay controls
probe_parameters.csv       selected probe hyperparameters and endpoint R²
pairs.csv                  subject IDs, controls, true centroids and input changes
trained_predictions.npz    endpoint predictions and ground truth
initial_predictions.npz    corresponding initialization control (unless skipped)
```

Checkpoints and rendered images are not copied into this output. Large feature
arrays are removed automatically. `report.json` records failures if an audit
cannot complete.

## Verification

```bash
python -m unittest tests.test_encoder_lesion_intervention -v
```

Tests include fixed-acquisition rendering, actual new-head routing, unchanged
encoder state/gradients, no-movement quantization cases, artificial true-tracking
and anatomy-only predictors, fitting/test separation, checkpoint replay and cleanup.
