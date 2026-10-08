# Encoder-only training on ADNI

`training/main_conv_synthetic.py --dataset-name ADNI_stripped_masks` trains the
encoder-only models on paired T1/FLAIR volumes. These are the conv or ResNet-18
backbones, with InfoNCE or Barlow Twins and the optional patch loss. The model and
objective are unchanged; only the data, preprocessing and evaluation differ. Submit
with `scripts/run_encoder_adni_slurm.sh` (`--dry-run` previews the command). Run
`scripts/preflight_adni.py experiments/adni_real.yaml --cluster slurm --sample 8`
first: it checks the same paths, size and split.

## What differs from a synthetic run

- **Data.** `data.datasets.MyCustomDataset` is built with the same arguments as
  `training/main_multimodal.py`. The flag names match `utils/config.py`, so the
  cluster YAML paths apply as written. With the same spacing, size, masks and labels,
  both trainers fingerprint to one preprocessing cache.
- **Evaluation.** Real data has no generator factors. DCI, per-factor R², the lesion
  branch and the spatial-recovery monitor are unavailable, and the parser refuses
  them. `eval.protocol.score_checkpoint.make_dataset` refuses a real run, so no
  synthetic scoring script can score it on the wrong data. Each evaluation writes
  `separation_step<N>.json` and prints these readouts against the step-0 floor:
  - the training objective on the held-out subjects;
  - T1↔FLAIR subject retrieval from content (chance 1/N);
  - the `eval.metrics.cross_reconstruction` probes (diagnosis, modality, effective
    rank, style retrieval). Their definitions match the VQ-VAE ADNI runs.
- **Selection.** `--best-metric val_loss` (the real-data default) keeps the lowest
  held-out objective. `none` keeps the last step only.
- **Files.** `split.json` lists the train/val/test subjects. Training never loads the
  test split.

## Flag considerations, roughly in order of impact

### Subject split: `--val-frac`, `--test-frac`, `--split-seed` (required)

The labels CSV is read in full whatever the mode. Without `--val-frac` the
validation subjects are the training subjects, so the trainer refuses to run.
Splits are subject-level and stratified by `Group`. If every subject in
`labels_cleaned_3class.csv` resolves on disk, `0.2 / 0.1` gives:

| split | subjects | CN / MCI / AD |
| --- | --- | --- |
| train | 1,166 | 661 / 400 / 105 |
| val | 291 | 165 / 100 / 26 |
| test | 163 | 92 / 56 / 15 |

Keep `--split-seed` fixed across any runs you compare. AD is 9% of each split, so
diagnosis probe accuracy is dominated by CN vs MCI. Majority-group chance is 0.567.

### Augmentation: `--asymmetric-aug` (recommended), `--shared-brain-mask`

The default training augmentation (`utils.utils.transforms` and the cached-path copy
in `MyCustomDataset`) has two properties that matter for a cross-view loss.
Both were measured on a fake tree, and both hold with and without `--cache`:

1. **Shared intensity shift.** `RandShiftIntensityd` (p=0.2) draws one offset for
   both views and adds it to every voxel, background included. Those pairs then share
   a non-anatomical offset that InfoNCE can match on.
2. **Unmasked edges.** A bilinear `RandAffined` without re-masking leaves the image
   non-zero outside its mask, up to 0.6 on the z-scored scale. Without `--cache` the
   masks are not transformed at all.

`--asymmetric-aug` draws the intensity augmentation independently per view and
re-applies the mask afterwards, which fixes both. The launcher turns it on. The
spatial affine stays shared, which keeps the views registered for the patch loss.
That affine's pose (small rotation and shear, up to 5% scaling) is still a shared
nuisance. Content can absorb it, and it confounds any brain-size readout. This flag also differs from
`experiments/adni_real.yaml`, so an encoder-vs-VQ comparison then differs in
augmentation too. It does not touch the preprocessing cache, which is built before
augmentation.

`--shared-brain-mask` intersects the T1 and FLAIR masks. The outline difference it
removes is view-specific, not shared, so it is not a shortcut. It does change the
cache fingerprint, which means a fresh preprocessing pass. Leave it off unless the
content modality probe shows boundary leakage.

### Volume size: `--image-spacing`, `--spatial-size` (required)

`2.0` mm with `96 112 96` matches `adni_real.yaml`, so it reuses that run's cache if
one was built. The conv encoder at stride 4 then gives a **24×28×24** map.

- Every grid must fit that map per axis; the parser checks this.
  `--train-patch-grid 8 8 8` fits, but 28/8 makes uneven, overlapping adaptive-pool
  bins. `6 7 6` gives cubic 4×4×4-cell bins, and the launcher uses it with
  `PATCH_WEIGHT=1`.
- ResNet-18 at stride 32 rounds up to a 3×4×3 map: too coarse for patch work.
  Stride 8 gives 12×14×12.
- Cost: 1.03 M voxels per view is about 4× a synthetic `--res 64` volume, in
  compute and activation memory. 1 mm (`150 180 150`) is about 15×, so try it only
  once the 2 mm recipe holds up.

### Normalisation (no flag)

The ADNI pipeline z-scores each scan over its brain voxels. That is the synthetic
`per_sample` mode, not the `fixed_reference` mode of the encoder synthetic recipe.
Per-scan z-scoring makes post-normalisation tissue intensity depend on anatomy
(tissue fractions). This gives a global code a route by which intensity carries
anatomy. A synthetic→ADNI comparison therefore changes normalisation as well as data.
WM-referenced normalisation (WhiteStripe) would be the fix; it is not implemented.

### Batch, steps and selection: `--batch-size`, `--train-steps`, `--eval-every`, `--tau`

- 32 per view gives InfoNCE 31 negatives. Within a batch, a few global factors
  (brain size, ventricles) already tell 32 subjects apart, so the loss can saturate
  on easy factors. Larger batches help but are memory-bound. The trainer has no AMP
  or gradient checkpointing.
- About 1,170 training subjects make 36 steps per epoch, so 10k steps is about 275
  epochs. Overfitting is the expected failure mode. Watch the train-vs-held-out loss
  gap, and keep `val_loss` selection (or a prespecified `none` endpoint). The
  launcher evaluates every 500 steps.
- Use `--tau 0.1` as in the recipe. The trainer's default of 1.0 learns much more
  slowly; on fake data it sat at chance for about 100 steps.

### Content size: `--content-channels`, `--latent-dim`

Real data has no true `n_content`, and the shared T1/FLAIR content is far larger
than 9 factors. `12 / 9` mirrors the synthetic recipe for comparability. A sweep is
a natural first experiment (launcher header: 9 / 16 / 32), judged on held-out
retrieval, the per-view diagnosis probes and effective rank, each against its floor.
The loss reads only the content block, so **the style units receive no gradient**.
They are an untrained readout of the trained backbone. The style rows, and anything
that mixes them in (`separation_score`), describe that readout, not a learned style
code, so they are not a selection target here.

### Architecture: `--encoder-architecture`, `--norm-type`, separate encoders

The conv backbone (stride 4) is the default and the one with a usable map at 2 mm.
GroupNorm pools statistics over all positions, background included, and the brain
fills only part of the volume. `--norm-type layer` normalises each voxel on its own;
keep it matched to the synthetic run you compare against. Separate per-view encoders
(the default) follow the paper.

### Cache, workers, threads: `--cache`, `--cache-dir`, `--num-workers`, `--hash-training-inputs`

- `--cache` (on by default) with `--cache-dir` writes 16.5 MB per subject: two
  views and two masks, float32. That is about 24 GB on disk for train plus val.
  Without `--cache-dir` the same 24 GB sits in RAM. Evaluation also holds the
  validation images in RAM, another 2.4 GB.
- Augmentation runs on the CPU (an affine on four volumes per subject), so use
  `--num-workers 8` on the cluster. Each worker reseeds its augmentation from the
  loader's generator, so runs still replay. On macOS keep 0: workers are spawned, and
  each receives a pickled copy of the dataset, RAM cache included.
- Do not pin `--cpu-threads 1` with `--num-workers 0`. Skip
  `--hash-training-inputs`: it hashes about 260 MB per step at batch 32. The batch
  order is still hashed.

### Determinism: `--deterministic`, `--deterministic-warn-only`, seeds

`--data-seed` seeds the augmentation and `--loader-seed` the order; worker streams
derive from both. CUDA adaptive-pooling backward has no deterministic kernel, so keep
`--deterministic-warn-only` as the recipe does.

## Reading the evaluation

- **Read the Δ column.** An untrained encoder already reads diagnosis and modality
  from raw anatomy and contrast.
- **Prefer the per-view diagnosis probes** (`_v0`, `_v1`). The pooled probe stacks
  both views, and its folds put a subject's T1 in one fold and its FLAIR in another.
- **Retrieval assumes one scan per subject**, which holds for this CSV. Chance is
  about 1/291.
- **Style rows**: see the content-size section above.

## Not wired yet

- **Factor-level targets.** `data_local/freesurfer.csv` (UCSF FreeSurfer, per visit)
  or ADNIMERGE would supply them: Ventricles, Hippocampus, ICV. They are the real-data
  counterpart of per-factor R², needed before claiming what the content encodes.
- **Test-split scoring.** `split.json` already lists the test subjects.
- **Per-view spatial augmentation** for global-only runs, to remove the shared-pose
  nuisance.
- **WM-referenced intensity normalisation.**
