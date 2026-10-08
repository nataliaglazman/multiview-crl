# Encoder-only training on ADNI

`training/main_conv_synthetic.py --dataset-name ADNI_stripped_masks` trains the
encoder-only models on paired T1/FLAIR volumes. These are the conv or ResNet-18
backbones, with InfoNCE or Barlow Twins and the optional patch loss. The model and
objective are unchanged; only the data, preprocessing and evaluation differ.

## Launching

The recipe lives in `scripts/encoder_adni_recipe.sh`, and one wrapper per cluster
sources it. The wrappers add only their cluster's paths (from
`experiments/cluster/<cluster>.yaml`) and submission. Each recipe knob (`MODEL_ID`,
`READOUT`, `NORM_TYPE`, `PATCH_WEIGHT`, `CONTENT_CHANNELS`, ...) is an environment
variable, overridable at submit time. `tests/test_encoder_adni.py` checks that the two clusters' commands
differ only in the five path flags.

| Cluster | Preview | Submit |
| --- | --- | --- |
| CREATE (SLURM) | `bash scripts/run_encoder_adni_slurm.sh --dry-run` | `sbatch scripts/run_encoder_adni_slurm.sh` |
| Run:ai | `bash scripts/run_encoder_adni_runai.sh --dry-run` | `bash scripts/run_encoder_adni_runai.sh` |

The Run:ai job runs the checkout at `/nfs/home/nglazman/crl-2/multiview-crl` inside
the training image. Sync the code there first. Runs land in that checkout's
`results/encoder_adni/runs/<MODEL_ID>`, and the job is named after `MODEL_ID`.

Run the data preflight first; it checks the same paths, size and split. On CREATE,
run it in the training env on the login node:
`python scripts/preflight_adni.py experiments/adni_real.yaml --cluster slurm --sample 8`.
On Run:ai the host Python has none of the dependencies, so run it as a job in the
training image and read the log from the host afterwards:

```bash
runai training standard submit preflight-adni-$(date +%H%M) \
    --project nglazman --image aicregistry:5000/nglazman:multiview-crl --run-as-user \
    --node-type A100 --gpu-devices-request 1 --cpu-core-request 16 --cpu-core-limit 32 \
    --cpu-memory-request 64G --cpu-memory-limit 128G --host-path path=/nfs,mount=/nfs,readwrite \
    --command -- bash -c "cd /nfs/home/nglazman/crl-2/multiview-crl && PYTHONPATH=. python scripts/preflight_adni.py experiments/adni_real.yaml --cluster runai --sample 8 > /nfs/home/nglazman/preflight_adni.log 2>&1"
```

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
re-applies the mask afterwards, which fixes both. The recipe turns it on. The
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
  bins. `6 7 6` gives cubic 4×4×4-cell bins, and the recipe uses it with
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
  on easy factors. The conv model has the GPU memory for 64 or 128 per view (see
  Architecture), which would make negatives harder. That changes the recipe relative
  to the synthetic runs, so treat it as its own arm.
- About 1,170 training subjects make 36 steps per epoch, so 10k steps is about 275
  epochs. Overfitting is the expected failure mode. Watch the train-vs-held-out loss
  gap, and keep `val_loss` selection (or a prespecified `none` endpoint). The
  recipe evaluates every 500 steps.
- Use `--tau 0.1` as in the recipe. The trainer's default of 1.0 learns much more
  slowly; on fake data it sat at chance for about 100 steps.

### Content size: `--content-channels`, `--latent-dim`

Real data has no true `n_content`, and the shared T1/FLAIR content is far larger
than 9 factors. `12 / 9` mirrors the synthetic recipe for comparability. A sweep is
a natural first experiment (9 / 16 / 32; the SLURM wrapper's header has the loop).
Judge it on held-out retrieval, the per-view diagnosis probes and effective rank,
each against its floor.
The loss reads only the content block, so **the style units receive no gradient**.
They are an untrained readout of the trained backbone. The style rows, and anything
that mixes them in (`separation_score`), describe that readout, not a learned style
code, so they are not a selection target here.

### Architecture: `--encoder-architecture`, `--conv-readout`, `--norm-type`

The recipe's default model is `conv_mlp_s42`, the reference model of the synthetic
encoder-only experiments (patch training, target follow-ups, lesion branch). That is
the conv backbone at stride 4, GAP, then a Linear(64, 100) → LeakyReLU → Linear(100, 12)
readout (`READOUT=mlp`), with GroupNorm and separate per-view encoders. The trainer's
own default is the linear readout, under which `--encoder-head-hidden` does nothing.
The conv backbone is also the only one with a usable map at 2 mm.

GroupNorm (`NORM_TYPE=group`) is what every encoder-only synthetic run used, so it
keeps ADNI comparable with them. It pools each scan's statistics over all positions,
background included, and writes them back into every cell. Background cells therefore
carry whole-volume quantities such as brain size. `NORM_TYPE=layer` normalises each
voxel across channels instead. It is the VQ-VAE ADNI run's encoder norm, and it starts
from the same weights at the same seed, so a `layer` run is a paired arm.

Measured at 96×112×96, the conv model saves about 5 GiB of activations at batch 32
per view, under either norm and readout. ResNet-18 saves 15.5 GiB (stride 32) to
18.7 GiB (stride 8). GPU memory is therefore not what limits the batch size.

### Cache, workers, threads: `--cache`, `--cache-dir`, `--num-workers`, `--hash-training-inputs`

- `--cache` (on by default) with `--cache-dir` writes 16.5 MB per subject: two
  views and two masks, float32. That is about 24 GB on disk for train plus val.
  Without `--cache-dir` the same 24 GB sits in RAM. Evaluation also holds the
  validation images in RAM, another 2.4 GB.
- Augmentation runs on the CPU (an affine on four volumes per subject). Measured
  single-threaded at 96×112×96 it takes about 0.06 s per subject, so 8 workers need
  about 0.2 s per step and keep up; with 0 workers it would be about 2 s per step.
  Each worker reseeds its augmentation from the loader's generator, so runs still
  replay.
- The worker count is part of the recipe, not just a throughput setting. Batches go
  to workers in turn and each worker has its own augmentation stream, so a different
  count trains on different augmentations. The recipe keeps 8 on both clusters,
  although Run:ai allocates 16 cores.
- On macOS keep 0 workers: they are spawned, and each receives a pickled copy of the
  dataset, RAM cache included.
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
