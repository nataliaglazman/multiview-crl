# Conv encoder: global plus patch InfoNCE

This experiment keeps the existing Conv + MLP model, data recipe, nine content
coordinates, global readout, and global InfoNCE loss. It adds a loss on an 8×8×8
grid of spatial content vectors:

`total = global_InfoNCE + patch_loss_weight * patch_InfoNCE`

The starting weight is **1.0**, a prespecified first experiment rather than a
tuned optimum. Training is from scratch with the same seed-42 initialization
and data/loader seeds as the original comparison. Existing checkpoints are not
overwritten or used as starting weights.

## What the patch loss sees

- The 64³ image produces a 16³ Conv feature map. Pool it to an 8³ grid, then
  apply the existing readout to each bin; select its first nine content channels.
- Pair the same subject and spatial position across T1/FLAIR. Negatives are
  other subjects at that position. Average InfoNCE over positions.
- Compute the original global vector separately: average backbone features
  first, then apply the MLP. Averaging patch MLP outputs would change this vector.
- Both branches reuse one backbone pass and the same readout parameters. No
  additional trainable parameters are introduced. An optional existing contrastive
  projector is shared across branches and acts along the channel axis.
- Use every grid position. No lesion masks, brain masks, factor values, or
  label-based crop selection enter the loss. The input pair must be spatially aligned.

The new launchers retain `--best-metric none`: factor labels are used only for
evaluation, not optimization or checkpoint selection. The saved final checkpoint
is the prespecified endpoint. This is a recoverability experiment, not a guarantee
of unsupervised identifiability.

## Local MPS run

From the repository root, in the local PyTorch environment:

```bash
# Preview only; no imports of PyTorch, allocation, or training output.
bash experiments/generated/encoder_conv_mlp_patch_s42.mps.sh --dry-run

# Check one disposable update on MPS, including both losses and CPU agreement.
bash experiments/generated/encoder_conv_mlp_patch_s42.mps.sh --check

# Train the full experiment only after the check succeeds.
bash experiments/generated/encoder_conv_mlp_patch_s42.mps.sh
```

Set `ENCODER_PYTHON` to choose Python, or activate the intended environment.
Set `ENCODER_PATCH_RESULTS` to change the local output root. Defaults:

```text
results/encoder_patch_mps/
  runs/conv_mlp_s42_mps_patch8x8x8_w1/
  logs/conv_mlp_s42_mps_patch8x8x8_w1.log
```

The launcher uses batch 32 per view and 10,000 steps to match the original
recipe. Smaller batches are available if needed, but change the number of
negatives and subjects seen. Compare both objectives at the same batch size:

```bash
# Matched smaller-batch control and patch experiment, in distinct directories.
bash experiments/generated/encoder_conv_mlp_patch_s42.mps.sh --batch-size 8 --patch-loss-weight 0
bash experiments/generated/encoder_conv_mlp_patch_s42.mps.sh --batch-size 8
```

Batch/step overrides and active patch settings appear in the default run ID;
existing directories are refused. `--train-steps 200 --eval-every 100` creates a
shorter run with a distinct ID. No long training job is launched by the tests.

## Local NVIDIA GPU / Linux PC

Copy the updated source and CUDA launcher to the PC first. From the repository
root on the laptop, this transfers code/configuration only, including the current
`eval/encoder`, `eval/protocol`, and `eval/metrics` package layout. It excludes
checkpoints, datasets, result directories, and Python caches:

```bash
rsync -avR \
  --exclude '__pycache__/' --exclude 'results/' --exclude 'runs/' \
  --include '*/' --include '*.py' --include '*.sh' \
  --include '*.json' --include '*.yaml' --include '*.yml' --exclude '*' \
  models/ training/ scripts/ eval/ data/ utils/ experiments/ \
  ng24@sie114-u-pc:~/projects/multiview-crl/
```

On the PC:

```bash
cd ~/projects/multiview-crl
conda activate monai_env
bash experiments/generated/encoder_conv_mlp_patch_s42.cuda.sh --dry-run
bash experiments/generated/encoder_conv_mlp_patch_s42.cuda.sh --check
bash experiments/generated/encoder_conv_mlp_patch_s42.cuda.sh
```

The script requires CUDA; it does not silently run on CPU. It uses the active
environment's Python and the same seed, data, batch 32 per view, 10,000 steps,
temperature, and unit-weight 8³ patch loss as the other launchers. Despite its
legacy filename, `scripts/run_encoder_mps.py` now supports `--device cuda` too.
The existing deterministic warning mode is retained.

```text
results/encoder_patch_cuda/
  runs/conv_mlp_s42_cuda_patch8x8x8_w1/
  logs/conv_mlp_s42_cuda_patch8x8x8_w1.log
```

The full command also runs the disposable backend check before training.
Use `CUDA_VISIBLE_DEVICES=0` to choose a GPU, `ENCODER_PYTHON` to choose Python,
or `ENCODER_PATCH_RESULTS` to change the output root. If batch 32 does not fit,
pass `--batch-size 8` to both `--check` and training, and use the same batch size
for the global-only control. As on MPS, this creates a distinct run ID.
For the spatial audit below, set `RUN` to the CUDA run and use `--device cuda`.

## Spatial recovery during training

The patch CUDA, MPS, and generated SLURM launchers now enable
`--spatial-recovery-eval`. At initialization (when `--floor-eval` is enabled)
and every `--eval-every` steps, this adds a frozen probe evaluation alongside
the unchanged global DCI evaluation. Direct trainer invocations keep it off
unless explicitly requested. A global-only control can enable the same flag.

The default reads backbone and content features at GAP and the training grid
(1³ and 8³ for these launchers), for **both T1 and FLAIR**. `projected` in reports
means the content readout before the optional contrastive projector. Feature
maps are flattened with spatial order retained; patches are not averaged back
into one vector before probing. The optional `--spatial-recovery-native` also
reads the full native map. This adds CPU work and temporary disk usage.

For each representation, ridge and nonlinear RBF probes are fitted on 75% of the
original validation subjects and tuned on the other 25%. They are scored on a
separate diagnostic cohort (400 subjects by default). This cohort is monitored
repeatedly, so it is not an untouched final test set. Shuffled controls permute
fit/tuning targets, never the scoring targets. All nine latent factors, physical
lesion centroids, and signed/absolute rendered sulcal amplitudes are scored.

Console output focuses on lesion coordinates, physical centroids, sulcal latent
and amplitude R², with amplitude's initialization delta and shuffled score.
Full per-factor scores, shuffled controls, and deltas from initialization are
saved here, and logged to TensorBoard:

```text
<run>/spatial_recovery/step_00000000/
<run>/spatial_recovery/step_00002000/
  report.json
  probes.csv
  summary.csv
  trained_predictions.npz
```

The initial folder has `initial_predictions.npz`. With `--no-floor-eval`,
initialization deltas are unavailable. Large feature arrays are temporary and
removed after each evaluation, including on probe failures. They live under the
run directory, so cluster scratch settings apply to them too.

The monitor restores Python/NumPy/PyTorch random states and module modes, checks
encoder weights/buffers are unchanged, and never updates gradients or selects an
encoder checkpoint. Labels supervise only the diagnostic probes.

Useful overrides on the CUDA/MPS launcher:

```bash
# A new run directory is needed if this experiment was already started.
bash experiments/generated/encoder_conv_mlp_patch_s42.cuda.sh \
  --results-dir results/encoder_patch_cuda_spatial \
  --spatial-recovery-native --spatial-recovery-batch-size 2

# Disable the additional evaluation; training objective is unchanged.
bash experiments/generated/encoder_conv_mlp_patch_s42.cuda.sh --no-spatial-recovery-eval
```

`--spatial-recovery-grids 1 8` fixes the grids explicitly, useful when matching a
global-only control. `--spatial-recovery-test-samples` changes diagnostic cohort
size; keep it and `--spatial-recovery-seed` matched across compared runs.

An already-running Python process will not acquire this evaluation from a file
update. There is no need to retrain an existing model: use the offline audit
below on a completed run, or on a separate stable checkpoint copy. The audit
rejects checkpoint files that change while it is running.

## SLURM run

```bash
bash experiments/generated/encoder_conv_mlp_patch_s42.slurm_bio.sh --dry-run
sbatch experiments/generated/encoder_conv_mlp_patch_s42.slurm_bio.sh
```

New cluster results default to:

```text
/scratch/users/k24058220/encoder_patch_slurm_bio/runs/conv_mlp_patch_g8x8x8_w1_s42/
```

SLURM stdout/stderr also go to the user's scratch directory. The script retains
the cluster resource/environment settings used by the original encoder scripts.
Regenerate only this new launcher, optionally changing its output root:

```bash
python scripts/generate_conv_patch_slurm.py \
  --results-dir /scratch/users/k24058220/encoder_patch_slurm_bio
```

## Trainer flags and diagnostics

For a direct `python -m training.main_conv_synthetic` command, add
`--patch-loss-weight 1 --train-patch-grid 8 8 8 --best-metric none` to the original
Conv arguments, with a new model ID/output directory. The default weight is zero,
which preserves global-only forward, loss, and gradients. A positive weight
currently requires InfoNCE and a grid that fits the actual backbone map.

`--eval-pooling` and `--eval-patch-grid` remain evaluation-only options. The new
training flags are saved in `settings.json`; console/TensorBoard logs and
`training_progress.json` record global, patch, weighted patch, and total losses.
Loss magnitude alone does not establish factor recovery. All-position averaging
can still dilute a sparse lesion signal, and spatial training can improve maps
without improving the final global content vector.

After training, run the same spatial audit on the new checkpoint:

```bash
RUN="$PWD/results/encoder_patch_mps/runs/conv_mlp_s42_mps_patch8x8x8_w1"
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -m eval.encoder.encoder_spatial_target_audit \
  --run-dir "$RUN" \
  --out-dir "$RUN/evaluation/spatial_$(date +%Y%m%d_%H%M%S)" \
  --device mps --batch-size 1 --test-samples 400 \
  --grids 1 8 --include-native --discard-features --seed 1729
```

Update `RUN` for a smaller-batch/shorter run or the cluster checkpoint. The
standalone audit accepts the unchanged checkpoint format; original comparison
manifests should not be rewritten after source changes. Compare trained versus
initial weights, both views, shuffled controls, and spatial versus global stages.
`--discard-features` keeps CSV/JSON/predictions and removes large feature banks;
omit it if those arrays are needed for further analysis. The audit restores the
saved lesion radius, including non-default radii.

For the existing default PC run, after training finishes:

```bash
RUN="$PWD/results/encoder_patch_cuda/runs/conv_mlp_s42_cuda_patch8x8x8_w1"
python -m eval.encoder.encoder_spatial_target_audit \
  --run-dir "$RUN" \
  --out-dir "$RUN/evaluation/spatial_$(date +%Y%m%d_%H%M%S)" \
  --device cuda --batch-size 2 --test-samples 400 \
  --grids 1 8 --include-native --discard-features --seed 1729
```

## Verification

```bash
python -m unittest tests.test_spatial_recovery_monitor tests.test_encoder_patch_training tests.test_encoder_mps_runner -v
```

Tests check actual local gradients into both encoders/readouts, content-only
selection, same-position subject comparisons, exact global-only regression,
one backbone pass per view, real short CPU training/checkpoint replay, and both
launchers. GPU-specific tests require their accelerator; use `--check` on the
laptop to validate its installed MPS backend before a long run.
Monitor tests additionally compare real training with evaluation enabled/disabled:
weights, input hashes, and batch order remain identical. They check RNG/mode/gradient
preservation, held-out controls, initialization pairing, and temporary-file cleanup.
