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

## Negative-free patch loss (Barlow Twins or VICReg)

Patch InfoNCE solves a separate softmax at every position: at position p, subject i's
FLAIR vector has to pick out subject i's T1 vector from the other subjects' T1 vectors
at p. Every position is asked to tell subjects apart, including one that looks the
same in every subject. Cosine similarity ignores scale, so such a region is pushed as
hard as an informative one, and the only differences it has left to amplify are
nuisance, noise and contrast. `--patch-loss-type barlow_twins|vicreg` replaces it with
a loss that has no negatives. The global loss keeps `--contrastive-loss-type` (InfoNCE
or Barlow Twins).

**What it computes.** The content patches form a tensor of shape
(view, subject, channel, position). Each view's mean over the batch's subjects is
subtracted at every (channel, position). That removes the anatomy all subjects share
there, which both views agree on and which would otherwise satisfy the alignment for
free. Variances, correlations and covariances then pool the (subject, position) rows.
So a channel's variance is its across-subject variance averaged over positions:

- `barlow_twins`: `barlow_twins_loss` with `center_mode="position"` and
  `patch_stat="fold"`, using `--bt-lambda`. It pushes the 9×9 cross-view correlation
  of the content channels towards the identity.
- `vicreg`: `losses.pooled_patch_vicreg_terms`, weighted by `--vicreg-sim-coeff`,
  `--vicreg-std-coeff` and `--vicreg-cov-coeff` (25/25/1). The invariance term is the
  MSE of the uncentred paired features, so a per-view offset at a position still
  costs. The variance hinge and the covariance use the pooled, centred statistics. The
  term weights follow the official VICReg code, which halves the hinge;
  `losses.vicreg_loss` does not.

**Why pooling, not per-position statistics.** After centring, a region that is alike in
every subject has rows near zero. It adds almost nothing to the pooled statistics, and
its gradient scales with its own spread. View-specific noise in it lowers the
cross-view correlation, so the loss shrinks that noise. The per-position alternatives
bring the problem back:

- `patch_stat="per_position"` divides each position by its own across-subject spread.
- VICRegL's per-position hinge (`training/vicreg_local.py`) requires every position to
  vary across subjects.

`tests/test_encoder_patch_noncontrastive.py` checks this on a toy. Rich positions carry
a shared subject signal. Flat positions carry only δ·(shared nuisance + view noise).
The table gives each loss's push on the flat nuisance, relative to its push on the
rich signal:

| patch loss | δ = 0.1 | δ = 0.01 | push on flat view noise |
|---|---|---|---|
| InfoNCE | +29 | +29 | down |
| Barlow Twins (pooled) | +0.011 | +0.0001 | down |
| VICReg (pooled) | +0.010 | +0.0001 | down |

In a scratch run of the same toy, Barlow Twins with per-position standardisation
pushed the flat nuisance 3.3× harder than the rich signal at any δ. VICRegL's
per-position hinge pushed the view noise up.

**Costs.** The rule that protects flat regions also down-weights a genuine local signal
of small variance, such as a lesion in 1–2 bins. So this targets regional style
leakage, not the recovery of hard local factors. Barlow Twins' correlation cannot see
a per-view constant offset; VICReg's MSE can. Both need `--mask-mode fixed`, and
`--patch-foreground-mask`, `--separate-spatial-readout` and the projector work as
for InfoNCE.

**Scale at step 0.** These are means over two batches of 32 at the
`conv_mlp_patch` recipe, seed 42:

- global InfoNCE: 6.91 (chance is 2 ln 32 = 6.93)
- patch InfoNCE: 7.40
- patch Barlow Twins: 7.92 (`on_diag` 7.92 of a maximum of 9)
- patch VICReg: 23.4, almost all variance hinge (pooled std 0.084)

So weight 1 matches the InfoNCE arm for Barlow Twins. For VICReg, weight 1 starts at
3.4× the global term, until the readout scales its patch outputs to unit spread.

```bash
python scripts/launch.py experiments/encoder/conv_mlp_patch.yaml --cluster slurm_bio \
  --set patch_loss_type=barlow_twins 'tag=conv_mlp_patch_bt_s{seed}'
python scripts/launch.py experiments/encoder/conv_mlp_patch.yaml --cluster slurm_bio \
  --set patch_loss_type=vicreg 'tag=conv_mlp_patch_vicreg_s{seed}'
# With the untied spatial head, for the readout interaction:
python scripts/launch.py experiments/encoder/conv_mlp_patch_separate_spatial.yaml --cluster slurm_bio \
  --set patch_loss_type=barlow_twins 'tag=conv_mlp_patch_separate_spatial_bt_s{seed}'
```

The logs add the unweighted parts: `on_diag` and `off_diag` for Barlow Twins, and
`sim`, `var`, `cov` and the pooled `std` for VICReg. They appear on the console line, in
TensorBoard as `train/patch_<type>_<part>`, in `last_loss_terms` and as `val/patch_<part>`
on real data. `training_progress.json` records `patch_loss_type`. InfoNCE runs keep
their old tags.

## Standardised patch InfoNCE

`--standardize-patch-code` is the patch analogue of `--standardize-code`. Before the
per-position similarity, each view's patch code is centred over the batch at every
position. Each unit is then divided by its spread pooled over subjects and positions,
not by its spread at each position. Dividing per position would promote a unit that is
flat at a position to unit-variance noise there, and would give every position the
same size. It needs a positive `--patch-loss-weight`, `--patch-loss-type infonce` and
`--mask-mode fixed`.

`--patch-similarity` picks the similarity that follows:

- `cosine` (default) L2-normalises every (view, subject, position) vector over units,
  as `losses._patch_infonce_base` always has. After centring, the vectors at a position
  alike in every subject are small, and the normalisation scales them back to unit
  length. So the standardisation then only reweights units within a position: a flat
  position keeps its full say and a rarely varying cell gains none. The VQ trainer's
  `--patch-center-weight` exists for the same reason.
- `dot` takes the dot product of the standardised code divided by the number of content
  units, so `--tau` keeps its meaning for a cell of typical size. Each cell keeps its
  size: a position alike in every subject drops out, and a cell that differs from the
  rest of the batch at its position counts in proportion to how much it differs. It
  needs `--standardize-patch-code`; on an unscaled code the encoder could lower the loss
  by growing its features.

**Measured.** The share of the squared gradient on the patch code, on one batch with the
foreground mask on. "Flattest cells" are the 10% of (subject, position) cells with the
smallest standardised residual. The last column is the patch term's encoder gradient
norm over the global term's.

On the patch8 recipe (12 content units, the neighbourhood sampler's first batch, 149 of
512 positions kept, untrained; lesion cells are 2.5% of the cells):

| patch similarity | flattest cells | lesion cells | patch / global gradient |
|---|---|---|---|
| raw cosine (default) | 0.23 | 0.004 | 0.06 |
| standardised cosine | 0.30 | 0.008 | 0.08 |
| standardised dot | 0.009 | 0.095 | 0.15 |

On the 9-unit run `conv_mlp_s42_mps_patch8x8x8_w1` (a random batch of 32, 201 positions
kept, untrained / trained; lesion cells are 2.2%):

| patch similarity | flattest cells | lesion cells | patch / global gradient |
|---|---|---|---|
| raw cosine (default) | 0.16 / 0.38 | 0.006 / 0.013 | 9.8 / 0.19 |
| standardised cosine | 0.35 / 0.36 | 0.006 / 0.016 | 15 / 0.07 |
| standardised cosine, positions weighted by mean residual size* | 0.15 / 0.14 | 0.025 / 0.043 | 16 / 0.07 |
| standardised dot | 0.007 / 0.012 | 0.060 / 0.078 | 28 / 0.21 |

\* Not a trainer option.

Only the rows that keep each cell's size move gradient onto the rare cells, and the
pooled scale is what lets them. These are first-order pushes at one point, not what
training does over many steps.

At step 0 the dot arm's patch push is 2.5× the cosine arm's in the patch8 recipe (0.15
vs 0.06 of the global gradient); a patch weight of 0.39 matches the two at step 0. On
the trained 9-unit run the two ratios were alike (0.21 vs 0.19). The arm matched to an
existing patch run changes only the similarity and the standardisation it needs; the
second command is a control for the stronger early push:

```bash
python scripts/launch.py --from-config results/encoder/runs/patch8/settings.json --cluster local \
  --set standardize_patch_code=True patch_similarity=dot tag=patch8_patchdot
python scripts/launch.py --from-config results/encoder/runs/patch8/settings.json --cluster local \
  --set standardize_patch_code=True patch_similarity=dot patch_loss_weight=0.39 tag=patch8_patchdot_w039
```

The logged `patch_infonce` values are not comparable across similarities: at step 0 the
patch8 recipe gives 8.0 under the cosine and 15.5 under the dot product (chance is
2 ln 32 = 6.93). Compare the arms with the regional readouts instead:

```bash
python -m eval.encoder.regional_lesion_detection \
  --run-dirs results/encoder/runs/patch8 results/encoder/runs/patch8_patchdot
python -m eval.encoder.regional_probe \
  --run-dirs results/encoder/runs/patch8 results/encoder/runs/patch8_patchdot --out regional_patchdot.json
```

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
