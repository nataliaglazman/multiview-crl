# Frozen lesion and sulcal scalar experiment

This implements the experiment in `LESION_SULCAL_DECISION.md`. It trains small
readouts on a **frozen, already trained global-only encoder**. It does not update
that encoder or its global head. Oracle arms are explicitly supervised controls;
all other arms receive images, frozen features and augmentations only.

## First run on the NVIDIA PC

From the repository root in `monai_env`, set `RUN` to the trained **global-only**
Conv run containing `settings.json` and `model.pt`, then run:

```bash
RUN=results/encoder_patch_cuda_baseline/runs/conv_mlp_s42_cuda
python -m unittest tests.test_local_scalar_experiment -v
LOCAL_SCALAR_DEVICE=cuda bash experiments/generated/local_scalar.local.sh "$RUN"
```

The default experiment uses the checkpoint's lesion-intensity setting and evaluates
that same setting. It runs every arm below for 2,000 steps with 512 training,
256 validation/probe and 400 test subjects, batch 32, four keypoint heads, grid 8.
Validation and test subjects are generated **after** all readout training.
Probe labels never select checkpoints or train the unsupervised readouts.

Output defaults to a new `results/local_scalar_<timestamp>` directory. Override
with `LOCAL_SCALAR_OUTPUT=/path/to/new/output`. Existing directories are refused;
there is no resume feature. `ENCODER_PYTHON` can select the Python executable.
`RUN` must be a global-only checkpoint; branch and patch-trained checkpoints are
rejected to keep this comparison controlled.

For a quick integration check, add these arguments after `"$RUN"`:

```bash
--steps 5 --train-samples 16 --probe-samples 16 --test-samples 16 \
--train-pair-subjects 1 --test-pair-subjects 2 --batch-size 8 \
--eval-eps .5 --bootstrap 10
```

Five steps are a software check, not evidence of recovery.

## Matched arms

| Name | Training signal |
|---|---|
| `initial` | Untrained geometric/signed readout; always evaluated |
| `hybrid_oracle` | Actual training centroids and signed amplitude, plus actual finite-factor endpoint responses |
| `free_oracle` | Earlier free spatial MLP architecture; supervised on the same four local targets |
| `infonce` | Local code identifies matching subjects across T1/FLAIR |
| `decorrelated` | Same InfoNCE plus within-view local/global correlation penalty |
| `barlow` | Barlow Twins on the same local code and projector |
| `residual` | Restricted local reconstruction, geometric equivariance, appearance consistency and a variance floor |
| `original_global` | Frozen source global code; always evaluated |

All hybrid readout arms start from identical weights and use the same
observational minibatch sequence. `free_oracle` has a different architecture and
parameter count; counts are saved. It reuses `ScalarReadout(geometric=False)` but
optimizes its three centroid outputs and amplitude only, unlike the earlier
nine-target experiment. The other five outputs are not evaluated here.

Choose a subset with, for example:

```bash
--arms infonce decorrelated barlow residual
```

With no oracle selected, no training intervention pairs are generated. The
readout emits 3K brain-frame coordinates plus one signed scalar (13 numbers for
K=4). Physical coordinates are derived using the input support's centroid and
spread. The brain frame is **not** the renderer's WM-quantile coordinate system.
Here "physical" means the actual rendered centroid in normalized image
coordinates; voxel errors use the input grid, not scanner-space millimetres.
The scalar is an ordered spatial projection with unrestricted signed weights;
it is not a spatial average or a magnitude-only statistic.

InfoNCE and Barlow share an identical linear projector, with output dimension
`min(16, 3K+1)` so linear expansion does not impose an impossible rank target on
Barlow. Coordinates are standardized within each modality before projection.
Barlow then standardizes projected outputs per modality and uses the standard
sum of diagonal squared errors plus `--barlow-lambda` (default .005) times the
sum of off-diagonal squared correlations. Raw loss magnitudes across objectives
are not comparable. Small batches give poor correlation estimates; the default
batch of 32 exceeds the default projector dimension of 13.

Barlow reduces redundancy within the projected local code, not between global
and local blocks. It can still learn anatomy or struggle with weak T1 lesions;
it is an empirical ablation, not an identifiability guarantee.
[Original paper](https://proceedings.mlr.press/v139/zbontar21a.html).

## What the residual arm actually predicts

1. Extract each modality with its saved frozen encoder and exact global readout.
   Augmented images are **re-encoded**, not simulated by warping frozen features.
2. Fit channel means/scales on training maps only. A fixed saved orthogonal
   projection supplies signed target channels (default up to 64). This preserves
   the whole 64-channel Conv feature space; when the source has more channels it
   compresses them. Use `--target-channels 512` for a full 512-channel ResNet
   target, or a smaller number as an explicitly recorded target-capacity ablation.
3. Fit a per-view quadratic ridge predictor from global code to every target
   channel/location. It uses all linear and unique quadratic terms, with fixed
   ridge strength `--ridge 10`. Freeze it before readout training. It is a limited
   anatomy predictor, not a claim that every anatomical effect is removed.
4. Subtract that prediction to obtain fixed signed residual feature maps.
5. Reconstruct residuals only through the local outputs. Each keypoint places a
   Gaussian blob of fixed width (`--blob-width-vox 3`, in input voxels), with
   learned per-view/channel strengths shared across subjects. The amplitude
   multiplies one learned signed template per view. There is no decoder bias,
   global-code path, image-dependent blob strength, feature skip or label input.
6. Average foreground-weighted reconstruction errors over full-resolution
   residuals, signed high-pass residuals and a grid at half the resolution.
   Each band/channel/view uses its training residual RMS (minimum .05) for scale.
   Foreground occupancy is from nonzero input support, **not lesion masks**.
7. Use original and mild gain/bias-augmented images to predict the original
   residual. Known reflections enforce physical-coordinate equivariance only;
   they impose no sulcal-amplitude reflection invariance. The high-pass target
   retains sign; it is not squared energy.

The predictor and normalization are fitted before every arm and reused exactly.
Checksums verify that the frozen encoder and target generator remain unchanged.
The training decoder is discarded at inference; inference remains encoder-only.

The residual may contain noise and unmodelled anatomy. Localized blobs may choose
another structure, while the template may fit another distributed effect. Use
held-out intervention recovery to decide whether these biases help. Whole-volume
loss reduction or decorrelation alone is not success.

## Visibility and modality controls

Use styled lesions for the visibility control, and evaluate both renderings:

```bash
LOCAL_SCALAR_DEVICE=cuda bash experiments/generated/local_scalar.local.sh "$RUN" \
  --train-intensity styled --eval-intensities styled fixed
```

All arms still use the **same frozen checkpoint**, even when the training images
use a different intensity rule. Styled rendering is not an improvement on the
original low-contrast distribution. The two evaluations are saved separately.
Each rendering has its own validation probes/head selection, so calibrated probe
scores describe recoverability under that rendering, not zero-shot transfer of
a single calibration. Native per-head coordinate metrics are uncalibrated.

For a FLAIR-only learning-signal control:

```bash
LOCAL_SCALAR_DEVICE=cuda bash experiments/generated/local_scalar.local.sh "$RUN" \
  --views flair --arms infonce decorrelated barlow residual
```

With one modality, InfoNCE/Barlow compare original versus appearance-augmented
versions of the same scan. Paired mode compares T1 and FLAIR. No repeated
acquisitions are required. Held-out recovery is computed only for selected views.

The decorrelation penalty is the **mean** squared within-view correlation across
local/global dimension pairs and modalities, with global codes detached. Default
`--decorrelation-weight .1`; try a prespecified sweep such as 0, .1 and 1.
This differs from the old branch's summed, pooled-view penalty: old/new lambda=1
are not equivalent. The backbone is actually frozen here. Independent raw
controls still need not produce independent physical coordinates.

## M1 and SLURM Bio

MPS is opt-in (`auto` means CUDA if available, otherwise CPU):

```bash
LOCAL_SCALAR_DEVICE=mps bash experiments/generated/local_scalar.local.sh "$RUN"
```

The local launcher enables PyTorch's MPS CPU fallback. CPU double precision is
used only while fitting the ridge predictor and doing probes; GPU training uses
float32. `--determinism warn` is the default because some CUDA/MPS operations lack
strict deterministic support. `--determinism strict` may legitimately fail.

For SLURM, first select the actual source checkpoint and submit from the repo:

```bash
export ENCODER_REFERENCE_RUN=/scratch/users/k24058220/PATH/TO/conv_mlp_s42
sbatch experiments/generated/local_scalar.slurm_bio.sh
```

The array has six jobs: fixed/styled training x readout seeds 42/142/242. Each job
runs all six trained arms and evaluates fixed and styled images. These seeds
vary readout initialization, augmentations and minibatches; all jobs retain the
checkpoint's data seed and frozen encoder. This is not a three-encoder-seed test.
For a smaller first submission use `sbatch --array=0 ...`.

Cluster results, temporary banks, plotting cache and scheduler logs all go to
`/scratch/users/k24058220`. Override `LOCAL_SCALAR_RESULTS`, `LOCAL_SCALAR_CACHE`,
`ENCODER_REFERENCE_RUN`, `ENCODER_REPO`, or `ENCODER_PYTHON` as needed. The script
uses the existing Bio A100 partition/module/environment conventions.

Preview either launcher without importing PyTorch or starting training:

```bash
bash experiments/generated/local_scalar.local.sh "$RUN" --dry-run
bash experiments/generated/local_scalar.slurm_bio.sh --dry-run
```

## Read the results

`report.json` records source hashes, settings, seeds, split identities, initial
readout hashes, parameter counts, training losses, and train/validation/test
reconstruction errors. Read train versus held-out global-predictor error to spot
memorization. Test labels never tune the predictor or losses.

Each `evaluation_fixed/` or `evaluation_styled/` contains:

- `recovery.csv`: raw controls, actual centroid, brain-frame centroid and signed
  amplitude. `joint_ridge` uses the entire local block; `best_single_scalar` uses
  one validation-chosen unit per target (no one-to-one constraint);
  `amplitude_scalar` uses the designated signed scalar; `native_head_N` uses exact
  geometric outputs. `selected_native_head` selects one head on validation only.
  Shuffled controls repeat calibration with permuted validation targets.
- `localization.csv`: native per-head mean/median Euclidean error in input voxels.
- `movement.csv`: actual lesion movement tracking for native selected heads and
  joint/shuffled probes, pooled over lesion-control axes, with subject-bootstrap
  confidence intervals. Skill 1 means perfect; 0 means predicting no movement.
- `intervention_recovery.csv`: signed response skills to all nine factor
  interventions, including anatomy-only changes. Endpoints use their actual
  rendered centroids; no incorrect diagonal response target is assumed.
- `response_matrix.csv`: signed mean and RMS raw-code response per raw factor,
  normalized by validation code SD, plus constant-unit/failed/zero-image counts.
- `*_calibration.json`, `*_codes.npz`, `truth.npz`: probe choices and compact
  predictions, including exact per-view brain-frame targets and physical targets.
- `*_error_maps.npz`: normalized reconstruction error and occupancy by spatial
  band. `recovery.png` gives a compact comparison; joint centroid recovery does
  not mean a native keypoint located the lesion.

Contrast groups use measured matched lesion-versus-lesion-free image differences,
with quartile cutoffs fitted on validation subjects only. These ground-truth
lesion regions are used in **evaluation only**. Low/high groups with too few
subjects can be absent; compare counts alongside scores.

Oracle training interventions use only training subjects. Test interventions use
a separate test subject range (`test_samples + 1000` onward), all nine factors,
and `--eval-eps .1 .25 .5`. Invalid placements are recorded without redraws.
Voxel-quantized zero-image changes are kept and counted.

`residual_target.pt` stores all fitted normalization/projection/prediction buffers;
restore with `ResidualTarget.from_state_dict(...)`. Each trained-arm `.pt` stores
readout state, optional decoder state, dimensions, arguments and source hashes.
Banks are temporary float32 memmaps, automatically removed on success or failure;
`--cache-dir` sets their parent. With the default Conv, expect roughly a few GB
of temporary disk for all cohorts, while retained reports/checkpoints are much
smaller. No NIfTI volumes or full feature banks are retained.

Start by checking oracle capacity, then native location and intervention scores.
Only a promising frozen-readout result warrants encoder fine-tuning. A failed
finite-budget oracle or SSL run is not an impossibility proof.
