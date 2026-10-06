# Residual lesion detectors and native localisation

This experiment tests whether sharing T1/FLAIR residual detectors erodes useful
lesion location information. Training is label-free. Synthetic masks and lesion
interventions are used **only for evaluation**; validation labels select a head
for one explicitly labelled diagnostic, not for training/checkpoint selection.

## Four matched arms

| Arm | Residual readout |
|---|---|
| `frozen_shared` | Existing shared 1x1 heads and projector frozen; global encoder trains |
| `shared` | Existing shared 1x1 heads and projector train |
| `separate` | Independent T1/FLAIR 1x1 heads, initially identical; shared coordinate projector |
| `separate_conv` | Independent T1/FLAIR Conv3d(2,16,3), GELU, Conv3d(16,16,3), GELU, then 1x1 heads |

All use the Conv-MLP backbone with per-voxel LayerNorm, styled lesions, 64³ inputs,
four heads at 16³, brain-relative coordinates, temperature 0.03, batch 32 and
cross-modal InfoNCE. Global and branch objectives remain separate. No patch loss
or decorrelation is added. Every arm uses the same data seed, subject order,
per-view PCA fit (200 training subjects), and residual SD calibration (100 more).
The comparison verifies matching input and normative-buffer hashes.

Simple separate heads copy the shared initial weights without consuming RNG.
Convolutional detectors start randomly, with identical copies across modalities;
their initialization preserves the global/projector weights and RNG stream.
They are compared with **their own initial checkpoint**, not just another arm's
floor. Sign-based `--lesion-head-init` ablations apply only to simple heads: the
conv head reads learned channels, not brighter/darker residual channels.

Residual and global gradients are clipped independently. Previously a single
gradient norm could couple their updates despite disjoint weights. Therefore
rerun the shared control with this code instead of treating an older run as the
matched control. Existing checkpoint layouts still load with shared defaults.

## Run

From the repository, in the normal PyTorch environment, on the CUDA PC:

```bash
bash experiments/generated/lesion_detectors.local.sh --dry-run
bash experiments/generated/lesion_detectors.local.sh
```

By default this runs all four arms for seed 42, for 2,000 steps, evaluates every
1,000 steps and at initialization, then runs the final held-out audit. It creates
a new timestamped directory under `results/lesion_detectors_*`.

On M1:

```bash
LESION_DETECTOR_DEVICE=mps bash experiments/generated/lesion_detectors.local.sh
```

For a full three-seed confirmation:

```bash
bash experiments/generated/lesion_detectors.local.sh \
  --seeds 42 142 242 --train-steps 10000 --eval-every 2000
```

On SLURM Bio, each array task runs all four arms for one seed. Outputs, logs,
temporary files and caches go to scratch:

```bash
sbatch experiments/generated/lesion_detectors.slurm_bio.sh
# Full-length confirmation:
sbatch experiments/generated/lesion_detectors.slurm_bio.sh --train-steps 10000 --eval-every 2000
```

`ENCODER_PYTHON` overrides the Python executable; `LESION_DETECTOR_OUTPUT` sets
the local output directory. SLURM uses `LESION_DETECTOR_RESULTS` as its root.
The Python driver also supports `plan`, `train`, `evaluate`, `summarize`, `all`:

```bash
python scripts/compare_lesion_detectors.py all --device cuda --output-dir results/my_detector_comparison
```

Repeat the same command to reuse completed, verified runs. Incomplete training
is not resumed automatically. Source/settings changes require a new directory.
Use exactly the same experiment options with separate driver actions.

For a single arm through the existing local launcher:

```bash
python scripts/run_encoder_mps.py --device cuda --variant conv_mlp \
  --lesion-keypoints 4 --lesion-input residual --lesion-detector separate \
  --lesion-temperature 0.03 --norm-type layer --synthetic-lesion-intensity styled \
  --lesion-localization-eval --train-steps 2000 --eval-every 1000
```

Use `--lesion-detector separate_conv` for convolutional capacity, or
`--lesion-detector shared --lesion-branch-frozen` for the frozen control.
The single-run launcher logs validation localisation; the four-arm driver also
automatically runs the held-out movement audit.

## Read the outputs

- Existing DCI tables remain: raw lesion-control R² and other factor recovery.
- Every training evaluation adds **all heads' native validation mean voxel error
  and hit rate within three voxels** to the log and `dci_step*.json`. No fitted
  coordinate probe or per-subject best-head selection is involved.
- `runs/<arm>_s<seed>/evaluation/native_lesion/localization.csv`: initial/trained
  localisation on 400 test subjects, per view/head, plus one validation-selected
  head. The head is selected using 200 separate validation subjects, once per
  checkpoint/view. This uses labels diagnostically, not as an unsupervised
  deployment selection rule. Different checkpoints may select different heads;
  retain the per-head rows when interpreting improvement.
- `movement.csv`: 64 further test subjects, each with x/y/z controls moved by
  ±0.5 while anatomy/acquisition/normalization stay fixed. Skill is
  `1 - sum(displacement error²)/sum(true displacement²)`: 1 perfect, 0 no movement.
  Scores use the **actual 3D centroid displacement**, not the nominal control
  axis. Quantized no-movement pairs and placement failures are counted, never
  redrawn. Confidence intervals bootstrap subjects, keeping their axes together.
- Two fixed full-resolution controls: smoothed unsigned residual peak, and
  dark-T1/bright-FLAIR peak. The latter explicitly uses a modality-polarity prior.
  Their grid differs from the 16³ head grid; this is a useful reference, not a
  capacity-matched head. Smoothing sigma is fixed at one input voxel.
- `predictions.npz`: compact positions, validation selections' inputs and movement
  endpoints for reanalysis; no image volumes or feature banks.
- Root `summary.csv`: all arms/initial states, direct metrics and changes versus
  initialization. Lower voxel error, higher hit rate and higher movement skill
  are better. A declining InfoNCE loss alone is not success.

Standalone audit of a compatible existing residual-branch run:

```bash
python -m eval.lesion.lesion_detector_audit --run-dir "$RUN" \
  --device cuda --out-dir "$RUN/evaluation/native_lesion"
```

This requires `model_init.pt` with the already-fitted PCA buffers and `model.pt`.
It rejects changed PCA buffers and reports both states. The audit uses only the
single-lesion `wm_interior` recipe, not lesion burden. Voxel coordinates follow
input array axes; distances are voxels, not clinical millimetres. These synthetic
diagnostics assess localisation and recovery, not an identifiability theorem.

## Implementation validation

The 24 existing lesion-branch tests and 9 new detector/audit tests pass, including
actual short CPU training, checkpoint replay, independent modality gradients,
frozen-head preservation, voxel-bin geometry and validation-only head selection.
A separate four-arm, two-step CPU smoke run completed training, initial/final
audits and aggregation. Its final global encoder weights matched exactly across
arms, the frozen detector stayed unchanged, and the PCA/input hashes matched.
Those tiny runs validate the pipeline, not scientific recovery performance.
CUDA/MPS execution was unavailable in the test host.

Broader checks retain the pre-existing failure in
`test_encoder_mps_runner.test_cuda_patch_launcher_matches_mps_and_has_separate_outputs`
(the older CUDA/MPS patch wrappers use different patch-loss weights). The eight
existing checkpoint-analysis tests pass with normal warning handling; this Mac's
NumPy/sklearn matmul RuntimeWarnings make four fail when warnings are promoted
to errors. The new detector tests pass with RuntimeWarnings treated as errors.
