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
- By default, both branches reuse one backbone pass and the same readout parameters. No
  additional trainable parameters are introduced. An optional existing contrastive
  projector is shared across branches and acts along the channel axis.
- Use every grid position. No lesion masks, brain masks, factor values, or
  label-based crop selection enter the loss. The input pair must be spatially aligned.

The new launchers retain `--best-metric none`: factor labels are used only for
evaluation, not optimization or checkpoint selection. The saved final checkpoint
is the prespecified endpoint. This is a recoverability experiment, not a guarantee
of unsupervised identifiability.

## Separate spatial readout experiment

Add `--separate-spatial-readout` to test whether tying the two readouts contributes
to the loss of lesion recoverability between the backbone and content map:

```text
                         GAP -> global MLP (64 -> 100 -> 12)
                        /                   first 9 -> global InfoNCE
view-specific backbone
                        \                   all 9 -> patch InfoNCE
                         pool 8³ -> spatial MLP (64 -> 100 -> 9)
```

Both readouts are shared across T1/FLAIR. Each objective directly updates its own
readout; both objectives update the same view-specific backbones. The spatial
head copies the global head's hidden layer and first nine output rows at
initialization, using independent parameters and no extra random draws. The
global head and backbone initial weights therefore match the shared-head control
at the same model seed. This is a new training run from scratch, not a continuation
of an existing checkpoint.

Nine is retained to match the current patch code's width, not to assign one channel
to each factor. A flattened 8³ code still contains 4,608 content values. The new
head adds **7,409 parameters** for the default Conv MLP; this tests untying the
readouts at fixed representation width, not a parameter-matched architecture.
The initial implementation requires an MLP readout, positive patch weight and
`--contrastive-proj-dim 0`, so no shared loss projector reconnects the heads.
The default shared-head model and its checkpoint keys are preserved.

The experiment is `experiments/encoder/conv_mlp_patch_separate_spatial.yaml`. On the NVIDIA PC
(after syncing the source as below), on SLURM, or on the Mac:

```bash
python scripts/launch.py experiments/encoder/conv_mlp_patch_separate_spatial.yaml --cluster local --set device=cuda
python scripts/launch.py experiments/encoder/conv_mlp_patch_separate_spatial.yaml --cluster slurm_bio
```

Its run is `results/encoder/runs/conv_mlp_patch_separate_spatial_s42/`, next to the
shared-head control's `conv_mlp_patch_s42/`.

Keep every override matched to the shared-head patch control: batch size, training
steps, lesion radius, seeds, patch weight/grid and evaluation settings. For example, if
the control used `--set synthetic_lesion_radius=0.14`, pass it to this run too, with a tag
that says so (`tag=conv_mlp_patch_separate_spatial_lr0.14_s{seed}`). Changing the batch size
for one arm only changes the contrastive negatives as well as memory use.

Spatial recovery is enabled by both patch experiments. In reports, **`projected` at grid
1 reads the global head; at larger grids it reads the separate spatial head**.
The CSV's `readout_head` field identifies the mapping. Spatial style probes are
absent because this head has only content outputs. Global DCI still reads the
unchanged global pathway. Model calls with an explicit patch grid, or unpooled
outputs, also use the active spatial head; their output/mask width is nine.

Compare physical lesion centroid and lesion-control R² in the spatial readout
against the shared-head patch run, at matched steps and on the same cohorts.
Check the backbone scores too: improved readout recovery with similar backbone
recovery would support the shared-readout bottleneck hypothesis. A change in both
means the backbone also adapted. Track sulcal recovery and initialization/shuffled
controls, and repeat across seeds before treating differences as reliable.

## Lesion intensity that follows acquisition style

By default the renderer blends the lesion in at a fixed intensity after the style
gain/bias have set the tissue intensities, so T1 lesion contrast ranges from 0.06
to 0.74 and nearly vanishes in low-gain scans
([details](../eval/encoder/ENCODER_LESION_CONTRAST.md)). With
`--synthetic-lesion-intensity styled` the lesion goes through the same gain/bias
map as the tissues: T1 contrast becomes `0.4*gain` (0.28 to 0.52) and FLAIR
`0.6*gain` (0.42 to 0.78). In the raw render only the lesion and its one-voxel
blur ring change. The `fixed_reference` normalization constants include lesion
voxels, so they shift by about 0.2%. This changes the training data and needs a
new run from scratch. The VQ-VAE trainer has the same flag.

Add the flag to the experiment that trained the fixed-intensity control, with a tag of
its own so the two never share a directory:

```bash
python scripts/launch.py experiments/encoder/conv_mlp_patch.yaml --cluster slurm_bio \
  --set synthetic_lesion_intensity=styled tag=conv_mlp_patch_lesionstyled_s{seed}
```

`experiments/encoder/conv_mlp_patch_lesion_corrected.yaml` is the styled patch run that was
trained for 20k steps.

The setting is saved in `settings.json` and restored by `score_checkpoint`, the
training-time spatial monitor and every `eval/encoder` audit. Runs saved before
the flag existed evaluate as `fixed`. Compare against the fixed-intensity run at
matched seeds, batch size, steps and lesion radius, with the lesion-movement and
contrast audits.

## Identifiable ventricle

The VQ-VAE recipe (`experiments/synthetic_causal.yaml`) renders the ventricle with
`synthetic_identifiable_ventricle: true`, but encoder-only runs before this flag
always used the default ventricle. So ventricle numbers compared across the two
model families describe different ventricles. `--synthetic-identifiable-ventricle`
gives the encoder-only trainer the VQ-VAE ventricle:

| | default | `--synthetic-identifiable-ventricle` |
|---|---|---|
| radius | 0.10–0.20 | 0.12–0.28 |
| shape | read off the deformed radius, so L–R asymmetry, temporal atrophy and the sulcal corrugation reshape it | read off the undeformed radius; no other factor moves it |
| fissure | ventricle CSF (same label and intensity) | its own label, at intensity 0.3 in both views |

This changes the training data and needs a new run, under a tag of its own (`--set` reads
booleans as Python literals, so `True`):

```bash
python scripts/launch.py experiments/encoder/conv_mlp_patch.yaml --cluster slurm_bio \
  --set synthetic_identifiable_ventricle=True tag=conv_mlp_patch_identvent_s{seed}
```

The setting is saved in `settings.json` and restored by `score_checkpoint`, the
training-time spatial monitor and the `eval/encoder` audits, which used to refuse
it. Runs saved before the flag existed evaluate with the default ventricle.

## Sulcal widening as atrophy

By default `sulcal_widening` (`z_content[8]`) is the signed depth of a fixed
zero-mean corrugation, `sin(12x)sin(12y)sin(12z)`, of the cortical surfaces. Its
image change cancels under spatial averaging, so a GAP readout cannot see it: an
untrained encoder's GAP reads it at R² −0.03. It also moves the brain outline and
the white matter, and the ventricle unless the ventricle flag above is set.

`--synthetic-sulcal-mode atrophy` makes the factor what sulcal widening is in
atrophy: more CSF in the sulci. The corrugation is switched off, and grey matter
inside fixed radial clefts becomes CSF:

- A cleft is every direction within an angular half-width of the zero set of a
  gyroid (frequency 9) on the direction sphere. All clefts share one width, run
  radially and scale with the brain.
- The half-width is `0.031 + 0.019·tanh(z8)` rad, 0.5 to 2 voxels across at the
  pial surface at resolution 64. Sulcal CSF goes from 7% to 28% of the cortex,
  linear in `tanh(z8)` and independent of brain size.
- Clefts reach 0.8 of the way from the pial surface to the white matter, so grey
  matter lines their floor.
- Only grey-matter labels change. The brain outline, white matter, ventricles,
  fissure and lesion are identical at every `z8`. Wider clefts darken both views
  (CSF is darker than grey matter in T1 and FLAIR), so every change has one sign.

On 400 validation subjects of the clean encoder recipe, an untrained encoder's GAP
reads the factor at R² 0.73 instead of −0.03. The other factors read the same as
before: brain size 0.94, thickness 0.86 → 0.85, ventricle 0.36 → 0.38. Caveats:

- Tissue volumes alone no longer identify the ventricle (R² 0.90 → 0.00). Four
  factors now move three volumes, and only location separates sulcal from
  ventricular CSF. A conv encoder sees location, but a probe that pools without
  spatial context mixes the two.
- It adds a factor that is easy to read at GAP. Easy global factors suppress the
  lesion under InfoNCE, so lesion recovery may get harder.
- Every subject has the same cleft pattern, the white matter does not fold, and
  clefts narrower than a voxel render as dotted lines.

In this mode the `sulcal_amplitude` and `sulcal_magnitude` evaluation targets both
hold the cleft half-width; `sulcal_widening` is `z8` as before. Run it under a tag of
its own:

```bash
python scripts/launch.py experiments/encoder/conv_mlp_patch.yaml --cluster slurm_bio \
  --set synthetic_sulcal_mode=atrophy tag=conv_mlp_patch_sulcalatrophy_s{seed}
```

The two flags are independent and can be combined. Settings are restored the same
way as above, and runs saved before the flag existed evaluate as `corrugation`.
Only the encoder-only trainer has this flag. Compare against the corrugation run
at matched seeds, batch size and steps.

## Foreground-masked patch loss

By default the patch InfoNCE averages over every position of the training grid.
At resolution 64 about 77% of a subject's 8³ grid is background, and those bins
still differ between subjects: receptive fields reach the brain edge and GroupNorm
writes whole-volume statistics into every position. Most of the patch loss can
therefore be met with global information such as brain size, and a lesion touches
1–2 of 512 bins.

`--patch-foreground-mask` uses the VQ-VAE trainer's rule and flag names. Each
batch, the brain mask is pooled to `--train-patch-grid`, and a position stays in
the patch loss if any sample has at least `--patch-foreground-thresh` brain there
(default 0.05). All subjects keep the same positions, and every position is kept
if none qualifies. With 32 subjects per batch at resolution 64 this keeps 180–200
of 512 positions; `--patch-foreground-thresh 0.5` keeps about 140. The mask comes
from the input itself, never from generator factors. The global loss is
unchanged, and the first batch's kept count is printed at the start of training.

It requires a positive `--patch-loss-weight`, which the patch experiments have:

```bash
python scripts/launch.py experiments/encoder/conv_mlp_patch.yaml --cluster slurm_bio \
  --set patch_foreground_mask=True tag=conv_mlp_patch_fgmask_s{seed}
```

Compare against the unmasked
patch run at matched seeds, batch size, steps, grid and weight. This changes
training only, so the evaluation tools need no setting from it.

## Local runs (Mac MPS or a Linux PC with CUDA)

From the repository root, in the local PyTorch environment, `--cluster local` runs the
trainer with the Python that runs the launcher:

```bash
# Preview only: the resolved config and command; nothing is imported or written.
python scripts/launch.py experiments/encoder/conv_mlp_patch.yaml --cluster local --set device=mps --dry-run
python scripts/launch.py experiments/encoder/conv_mlp_patch.yaml --cluster local --set device=mps
```

The launcher sets `PYTORCH_ENABLE_MPS_FALLBACK=1` for MPS, keeps a Mac awake under
`caffeinate`, pins one numerical thread, and supplies a stand-in for `lpips` when it is not
installed (the encoder trainer imports it but never uses it). The run lands in
`results/encoder/runs/conv_mlp_patch_s42/`.

The recipe uses batch 32 per view and 10,000 steps. Smaller batches change the number of
negatives and subjects seen, so compare both objectives at the same batch size, each under
its own tag:

```bash
python scripts/launch.py experiments/encoder/conv_mlp.yaml --cluster local --set device=mps batch_size=8 tag=conv_mlp_b8_s{seed}
python scripts/launch.py experiments/encoder/conv_mlp_patch.yaml --cluster local --set device=mps batch_size=8 tag=conv_mlp_patch_b8_s{seed}
```

Existing run directories are refused. `--set train_steps=200 eval_every=100 tag=...` makes a
short run.

On the NVIDIA PC, copy the code and configuration first. From the repository root on the
laptop, this excludes checkpoints, datasets, results and Python caches:

```bash
rsync -avR \
  --exclude '__pycache__/' --exclude 'results/' --exclude 'runs/' \
  --include '*/' --include '*.py' --include '*.sh' \
  --include '*.json' --include '*.yaml' --include '*.yml' --exclude '*' \
  models/ training/ scripts/ eval/ data/ utils/ experiments/ \
  ng24@sie114-u-pc:~/projects/multiview-crl/
```

Then, on the PC:

```bash
cd ~/projects/multiview-crl
conda activate monai_env
python scripts/launch.py experiments/encoder/conv_mlp_patch.yaml --cluster local --set device=cuda
```

`--set device=cuda` fails at once without a usable GPU instead of falling back to CPU. Use
`CUDA_VISIBLE_DEVICES=0` to choose a GPU. For the spatial audit below, set `RUN` to the PC's
run and use `--device cuda`.

## Spatial recovery during training

The patch experiments (`conv_mlp_patch*.yaml`) enable `--spatial-recovery-eval`. At initialization (when `--floor-eval` is enabled)
and every `--eval-every` steps, this adds a frozen probe evaluation alongside
the unchanged global DCI evaluation. Direct trainer invocations keep it off
unless explicitly requested. A global-only control can enable the same flag.

The default reads backbone and content features at GAP and the training grid
(1³ and 8³ for these experiments), for **both T1 and FLAIR**. `projected` in reports
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

Useful overrides:

```bash
# A new run directory is needed if this experiment was already started.
python scripts/launch.py experiments/encoder/conv_mlp_patch.yaml --cluster local --set device=cuda \
  out_dir=results/encoder_spatial/runs spatial_recovery_native=True spatial_recovery_batch_size=2

# Disable the additional evaluation; the training objective is unchanged.
python scripts/launch.py experiments/encoder/conv_mlp_patch.yaml --cluster local --set device=cuda \
  spatial_recovery_eval=False
```

`--set 'spatial_recovery_grids=[1,8]'` fixes the grids explicitly, useful when matching a
global-only control. `--spatial-recovery-test-samples` changes diagnostic cohort
size; keep it and `--spatial-recovery-seed` matched across compared runs.

An already-running Python process will not acquire this evaluation from a file
update. There is no need to retrain an existing model: use the offline audit
below on a completed run, or on a separate stable checkpoint copy. The audit
rejects checkpoint files that change while it is running.

## SLURM run

```bash
python scripts/launch.py experiments/encoder/conv_mlp_patch.yaml --cluster slurm_bio --dry-run
python scripts/launch.py experiments/encoder/conv_mlp_patch.yaml --cluster slurm_bio
```

The run lands in the checkout's `results/encoder/runs/conv_mlp_patch_s42/`. Earlier patch runs
are under `/scratch/users/k24058220/encoder_patch_slurm_bio/runs/`;
`--set out_dir=/scratch/users/k24058220/encoder_patch_slurm_bio/runs` keeps new ones there.
SLURM stdout/stderr go to the user's scratch directory. Resources come from
`experiments/cluster/slurm_bio.yaml`, as for every encoder job.

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
RUN="$PWD/results/encoder/runs/conv_mlp_patch_s42"
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

On the PC, after training finishes:

```bash
RUN="$PWD/results/encoder/runs/conv_mlp_patch_s42"
python -m eval.encoder.encoder_spatial_target_audit \
  --run-dir "$RUN" \
  --out-dir "$RUN/evaluation/spatial_$(date +%Y%m%d_%H%M%S)" \
  --device cuda --batch-size 2 --test-samples 400 \
  --grids 1 8 --include-native --discard-features --seed 1729
```

## Verification

To distinguish lesion sensitivity from predictions based on anatomy, run the
[frozen lesion-movement test](../eval/encoder/ENCODER_LESION_INTERVENTION.md).
It moves only the lesion with acquisition/anatomy fixed and tests the backbone,
global code and active spatial readout without retraining the encoder.

```bash
python -m unittest tests.test_separate_spatial_readout tests.test_spatial_recovery_monitor tests.test_encoder_patch_training tests.test_encoder_launch -v
```

Tests check actual local gradients into both encoders/readouts, content-only
selection, same-position subject comparisons, exact global-only regression,
one backbone pass per view, real short CPU training/checkpoint replay, and that each
patch experiment differs from `conv_mlp` only in its patch settings. A short local run
(`--set train_steps=200 eval_every=100 tag=...`) checks an accelerator before a long one.
Monitor tests additionally compare real training with evaluation enabled/disabled:
weights, input hashes, and batch order remain identical. They check RNG/mode/gradient
preservation, held-out controls, initialization pairing, and temporary-file cleanup.
Separate-head tests check copied initialization and RNG state, isolated head
gradients, active-head probe routing, strict checkpoint replay, and matched inputs
through real short training.
