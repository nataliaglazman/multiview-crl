# Scalar readout capacity and learning-signal comparison

This experiment tests whether local information can be retained in nine scalar
outputs, and whether an annotation-free objective learns to retain it. It leaves
the source checkpoint and existing training entry points unchanged.

| Arm | Readout | Training information |
|---|---|---|
| `free_oracle` | Spatial CNN + flattened MLP → nine scalars | Original targets and finite factor interventions |
| `geometric_oracle` | One centroid distribution → three coordinates; six attention-weighted scalars | Same targets/interventions and loss as free oracle |
| `geometric_ssl` | Identical architecture and initialization to geometric oracle | Ordinary images, known reflections, mild intensity augmentations; no factor labels/interventions |

The untrained geometric readout and original checkpoint's nine global content
values are evaluated as controls. The original global control is available for
`--source backbone` only.

## Run on the NVIDIA PC

Activate `monai_env`, update these new source files, then run from the repo root:

```bash
RUN="results/encoder_patch_cuda_baseline/runs/conv_mlp_s42_cuda"

python -m training.scalar_readout_experiment \
  --run-dir "$RUN" \
  --out-dir "results/scalar_readout_t1_$(date +%Y%m%d_%H%M%S)" \
  --source backbone --view t1 --device cuda
```

Repeat with `--view flair`. `--checkpoint model.pt` is the default. A new output
directory is mandatory; there is no checkpoint selection using test results.

To test the image-input control, repeat with `--source image` in a new output
directory. This uses saved generator settings but loads no source checkpoint.
Its small CNN is trained from scratch as part of the readout; it is not a frozen
identity-map baseline. The backbone and image experiments therefore answer
different capacity questions and are not parameter-matched architectures.

Defaults: 2,000 steps per arm, batch 8, seed 42; 512 training subjects, 256
validation subjects for post-training probes, 400 test subjects. There are 128
training intervention subjects at ±0.5, and 32 independent test intervention
subjects at ±0.1/0.25/0.5. Training interventions use the first 128 training
subjects; test interventions use test-split IDs starting at `test_samples+1000`.

The default backbone feature grid is 8³. It must fit and divide the native feature
map: for stride-32 ResNet at resolution 64 use `--grid 2`. No feature upsampling
pretends to recover a finer spatial map. The image-input stem has stride 4, so
`4*grid` must divide the image resolution. Descriptor grid 16 must also divide it.

Use `--cache-dir /scratch/users/k24058220/cache/scalar_readout` on the cluster or
another spacious temporary directory. Feature/image banks are created uniquely
and deleted at completion or failure. Image-source caches can occupy several GB;
reports and scalar predictions are small. Readout checkpoints include the SSL
auxiliary decoder, whose default final layer is approximately 4 MB.

## Laptop and cluster launchers

```bash
# Active Python environment; MPS is explicit, otherwise auto selects CUDA/CPU.
SCALAR_DEVICE=mps bash experiments/generated/scalar_readout.local.sh "$RUN"

# Preview the command without starting training.
bash experiments/generated/scalar_readout.local.sh "$RUN" --dry-run --steps 100

# SLURM: output, temporary banks and scheduler logs go to scratch.
export ENCODER_REFERENCE_RUN="/path/to/your/conv_mlp_s42"
bash experiments/generated/scalar_readout.slurm_bio.sh --dry-run
sbatch experiments/generated/scalar_readout.slurm_bio.sh
```

The SLURM array has four jobs: backbone/T1, backbone/FLAIR, image/T1, image/FLAIR.
Each runs the three arms sequentially. `SCALAR_RESULTS`, `SCALAR_CACHE`,
`ENCODER_REPO` and `ENCODER_PYTHON` override paths. Nothing is submitted by Python.
The MPS/CUDA launchers are provided, but accelerator execution must be verified
on that host; CPU tests do not certify accelerator support.

For a short workflow check (not scientifically interpretable training):

```bash
python -m training.scalar_readout_experiment \
  --run-dir "$RUN" --out-dir results/scalar_readout_smoke \
  --view t1 --device cpu --steps 5 --batch-size 4 \
  --train-samples 16 --probe-samples 16 --test-samples 16 \
  --train-pair-subjects 2 --test-pair-subjects 2 --eval-eps 0.5
```

## What the scalars mean

The output order is:

```text
brain_size, ventricle_size, centroid_x, centroid_y, centroid_z,
cortical_thickness, temporal_atrophy, lr_asymmetry, sulcal_amplitude
```

For the oracle arms these are the explicitly trained targets. For SSL, only the
three coordinate outputs have an architectural coordinate meaning; the other six
units have no imposed factor assignment. The data do not automatically make the
coordinate head a lesion detector. Validation-only matching tests whether the
scalars acquired the intended meanings.

Centroids are physical index coordinates in the renderer's [-1,1] frame. They
are NOT the original lesion_x/y/z controls, which select positions relative to
available white matter. Their expected responses can change when anatomy changes.
The signed sulcal-amplitude target includes the renderer's squash and scale. Raw
control recovery, including all nine original content factors, is also reported.

The geometric head learns one spatial probability distribution and returns its
three coordinate expectations. It learns six separate weighted scalar summaries
for the other outputs. There is no MLP after this geometric readout. A multi-modal
location distribution can still average several structures; a coordinate output
does not itself prove localization of one lesion.

## Oracle objective

Training-only target standard deviations balance the nine units. The loss is:

```text
standardized baseline MSE
  + 0.5 * standardized endpoint MSE
  + response_weight * standardized paired-response MSE
```

Each intervention changes one raw content control, retaining the other raw
latents, acquisition random draws, baseline normalization affine and native
white-matter placement rule. The paired response target is the actual change in
the semantic targets. It is not artificially diagonal: moving brain boundaries
can also move the physical lesion. Invalid placements are recorded and excluded;
they are not redrawn. Quantized pairs with no image change are retained, so the
audit reveals the resulting inability to follow the latent change.

These are supervised positive controls, not examples of unsupervised discovery.
Matching finite differences plus target levels tests a concrete readout's capacity;
failure within 2,000 steps is not proof of an information-theoretic limitation.

## Annotation-free objective

`train_ssl` receives only the six explicitly named unlabelled arrays. It cannot
access targets, factor IDs, intervention images, validation observations or test
observations. `--arms geometric_ssl` skips training-intervention generation.
Reflected images are passed through the frozen encoder again, rather than simply
flipping its feature maps. One reproducible non-identity reflection and one mild
photometric augmentation are cached per training subject.

The SSL loss combines:

- Coordinate equivariance under the known reflection.
- Nine-scalar consistency under mild gain/bias/noise changes.
- An anti-collapse standard-deviation floor (a heuristic, not identifiability).
- Reconstruction through the nine scalars only: local mean intensity and
  high-pass energy on a 16³ descriptor lattice. Energy is computed before spatial
  binning. The two descriptor channels are standardized using unlabelled training
  images. Original/reflected images reconstruct their matching descriptors;
  photometrically augmented inputs reconstruct the original descriptors.

The reconstruction decoder is an auxiliary training head and can be discarded
at inference. It sees no spatial feature map, source image, or per-subject context
in addition to the nine scalars. It can still allocate information to unintended
scalars; intervention matrices are intended to detect that outcome.

This objective is a testable proposal, not a claim that self-supervision must find
lesions. Easy anatomical landmarks can satisfy equivariance; high-pass energy
alone does not identify the sign of corrugation; reconstruction can prefer anatomy
or noise. No invariance of all scalars to reflection is imposed, since reflection
can change signed anatomical factors.

## Read the results

`recovery.csv` and `scalar_recovery.png` compare independent test observations:

- `scalar`: one-to-one unit assignment and one-variable affine calibration fitted
  on validation observations. A target never receives a mixture of code units.
- `vector_ridge`: all nine units, with ridge strength selected on an internal
  validation split and then refitted on validation. Good vector scores but poor
  scalar scores suggest distributed information.
- `shuffled_scalar`: validation target rows shuffled before matching/calibration.
- `direct`: native target-unit predictions, oracle arms only.
- `geometric_coordinates`: the three native centroid outputs, with no label-based
  calibration or matching, including for the SSL and initial geometric heads.

`response_matrix.csv/.png` records finite responses to each of nine raw factors,
normalized by each code unit's validation SD. It reports signed mean AND RMS so
opposing subject responses do not disappear in an average. Read constant-unit,
zero-image and valid/failed counts. A bright or diagonal matrix alone is not a
recovery score and can reflect noise sensitivity or tiny denominators.

`intervention_recovery.csv` applies the SAME frozen validation calibration to
both endpoints. For each target and intervention it reports:

```text
response_skill = 1 - sum((predicted_delta - true_delta)^2) / sum(true_delta^2)
```

Skill 1 is perfect; 0 matches no predicted change. Where the true response is
zero, skill is undefined and predicted-delta RMS measures unwanted sensitivity.
No test intervention selects a unit, calibration, hyperparameter or checkpoint.

Other outputs include per-pair failure/zero flags, scalar codes, truth arrays,
calibration weights, trained readout checkpoints and a provenance report. The
report verifies frozen encoder parameters/buffers and original files unchanged.

Interpret contrasts cautiously:

| Outcome | Supported interpretation |
|---|---|
| Image oracle good, backbone oracle poor | This trained image readout succeeds where this frozen representation/readout combination does not |
| Free oracle good, geometric oracle poor | The tested geometric readout or its optimization is restrictive |
| Geometric oracle good, identical SSL architecture poor | Capacity exists; this SSL learning signal does not recover it |
| Vector recovery good, scalar recovery poor | Information is linearly recoverable jointly, not from the tested individual affine readouts |
| All controls poor | Investigate optimization, target quantization, SNR and readout capacity; no impossibility conclusion |

Affine scalar-probe failure does not rule out nonlinear one-variable recovery.
Finite response matrices do not prove global invertibility or identifiability.
Repeat runs with several `--seed` values; this varies initialization, optimization,
augmentation and probe split, while the saved generator seed fixes the cohort.

## Validation

```bash
python -m unittest tests.test_scalar_readout_experiment -v
```

Tests check geometric coordinates/reflections, a nine-scalar-only reconstruction
path, oracle response targets with physical coupling, scalar-vs-rotated-vector
recovery, response cancellation, retained placement failures, full frozen-backbone
and image-source runs, SSL-only intervention isolation, source integrity and
temporary-bank cleanup.
