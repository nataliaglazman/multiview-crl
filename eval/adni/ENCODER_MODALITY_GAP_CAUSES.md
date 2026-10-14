# Encoder-only ADNI modality gap: candidate causes

[`encoder_modality_gap`](ENCODER_MODALITY_GAP.md) measures that T1 and FLAIR content
separate. `encoder_modality_gap_causes` tests why. Each check targets one candidate
cause and reports the signature that cause would leave:

| Candidate cause | Check | Signature if it is the cause |
| --- | --- | --- |
| The two view-specific encoders generalize differently to new subjects | `splits` | Gap and probes much smaller on training subjects than on held-out ones |
| Modality-specific inputs that barely vary across subjects (each view's own brain mask, edge contrast, slice thickness) | `offset`, `subjects` | Offset along directions subjects barely vary in; per-subject offset tracks mask or blur descriptors |
| Information only one modality carries (FLAIR lesions, T1 grey/white contrast) | `shape`, `ladder`, `subjects` | A few directions with a FLAIR/T1 variance ratio far from the rest; a residual that whitening removes; residuals tracking WMH or protocol |
| A weak loss signal | `loss` | In-batch top-1 near 1, and a small offset cost at the trained batch size |

## Run

On Run:ai, the existing launcher runs it with `MODULE` set. It encodes the validation
and training subjects (about 1,460 pairs) with augmentation off, so give it a finished
checkpoint and the run's preprocessing cache:

```bash
MODULE=eval.adni.encoder_modality_gap_causes bash scripts/run_encoder_adni_modality_gap_runai.sh --dry-run
MODULE=eval.adni.encoder_modality_gap_causes bash scripts/run_encoder_adni_modality_gap_runai.sh
```

`MODEL_ID`, `RUN_DIR`, `CHECKPOINT` and the other launcher overrides work as for the
diagnostic. For the untrained floor, add `--checkpoint model_init.pt`; every check
that reads below 'trained' should be compared with it.

Re-analyse saved features anywhere with NumPy, SciPy, scikit-learn and threadpoolctl.
This path loads neither PyTorch nor MRI, so it is the place to add covariates:

```bash
python -m eval.adni.encoder_modality_gap_causes \
    --features val=<report>/features_val.npz train=<report>/features_train.npz \
    --covariates covariates.csv --subject-column Subject
```

The diagnostic's own `features.npz` is accepted too, as the validation split without
descriptors. So are the `alignment_step<N>.npz` files the trainer writes at every
evaluation. They hold validation content only, with no run settings, so the loss
check assumes the recipe (tau 0.1, cross-view negatives, batch 32); pass `--tau` if
the run differs. That gives the offset, shape, ladder and loss checks at any step,
including step 0 as the floor, without a GPU:

```bash
for step in 0 10000; do
    python -m eval.adni.encoder_modality_gap_causes \
        --features results/encoder_adni/runs/<MODEL_ID>/alignment_step$step.npz \
        --out-dir causes_step$step
done
```

The training-vs-held-out comparison and the descriptors need `--run-dir`. The test
split is never read.

## The checks

All checks read the content block the diagnostic reads (the evaluated content units,
before any projector), L2-normalized unless stated. The **swap null** exchanges T1 and
FLAIR within a random half of the pairs: the distribution a statistic would have if the
modality label were arbitrary.

### offset: where the mean offset lies

- **Radial share**: the fraction of the offset's squared length along the two
  modalities' average mean. Near 1, the two means point the same way and differ in
  length, so one modality's codes are more concentrated. Near 0, one cloud is shifted
  sideways relative to the other.
- **Accuracy the offset alone gives**: the best linear accuracy the offset would give
  if each modality were a Gaussian with the pooled within-modality covariance. It is
  compared with the same-length offset pointed in 4,000 random directions. An offset
  that beats most random directions runs along directions subjects barely vary in,
  which makes it a near-constant modality tag rather than a bias on a large anatomical
  axis. The low-variance-half share says the same thing geometrically.
- Read this section only when the centroid distance is well above its swap-null p95.

### shape: variance ratios per direction

Each modality's covariance about its own mean, compared along the directions that
decorrelate both at once (generalized eigenvalues: FLAIR variance over T1 variance).
They do not change under any invertible linear map applied to both modalities, so the
channels' scales do not matter.

- The raw offset is removed first (both modalities given the pooled raw mean, then
  L2-normalized). Normalizing two clouds that sit in different places compresses each
  along its own mean direction, so a pure offset would otherwise read as a shape
  difference.
- Ratios are divided by their median. On unit vectors each modality's total spread is
  close to 1, so a modality that varies less along a few directions varies more along
  all the others; relative ratios keep that from flagging every remaining direction.
- Each rank is compared with its own swap-null envelope, because sample generalized
  eigenvalues spread away from 1 even when the covariances are equal. Directions
  outside it are ones the two encoders read at a different gain. The shape distance
  `sqrt(sum(log(ratio)^2))` is the omnibus test.

### ladder: probes after each removal

The diagnostic's folds, scaler and probes (logistic regression and RBF SVM, C=1) on
four versions of the codes. Every removed statistic is fitted on the training folds.

| Rung | Removes | High accuracy means |
| --- | --- | --- |
| `normalized` | nothing (= diagnostic) | modality is decodable |
| `mean_centered` | each modality's normalized-space mean (= diagnostic) | something beyond a constant normalized offset |
| `raw_offset_removed` | the raw offset, then L2 normalization | something beyond what a constant encoder bias could change |
| `whitened` | each modality's mean and sample covariance | structure beyond mean and covariance, e.g. subgroups |

- **Read every rung against its own swap-null p95, not against 0.5.** Fitting a
  modality-specific transform on training folds can push the null below 0.5.
- `raw_offset_removed` well below `mean_centered`: part of the centred residual is a raw
  offset that L2 normalization made subject-dependent (it shrinks with each code's norm).
- `mean_centered` high and `whitened` near its null: the residual is a covariance
  difference, which the `shape` check then localizes.
- The two linear rungs after a mean removal sit at 0.5 by construction.
- `normalized` and `mean_centered` reproduce the diagnostic's numbers exactly; a test checks this.

### loss: what the offset costs the run's objective

The trainer's global InfoNCE, mirrored in NumPy and checked against
`training.losses.infonce_loss`. Its temperature, negatives and `--standardize-code` come
from the run's settings. It is evaluated on 500 random batches of these pairs at the
run's batch size and four times it.

- **as is** vs **offset removed** (the raw offset, the most a constant per-modality
  encoder bias could change) vs **fully aligned** (the pair average fed as both views:
  the loss's floor for this spread of subjects).
- **In-batch top-1 near 1** means batches are saturated: once the true partner wins the
  softmax, every gradient is small, the one that would remove the offset included.
- **Offset share**: the fraction of the loss above that floor the offset accounts for.
- Barlow Twins, `--standardize-code` and projection-head runs are blind to a per-view
  offset by construction or are not reproducible from these features; the check says so.
- These are held-out, unaugmented pairs; training saw augmented training pairs. With a
  learned mask, training multiplies every unit by a relaxed mask; this uses the
  evaluated top-k units.

### subjects: per-subject residuals against covariates

Three per-subject measures:

- `residual`: the T1-FLAIR difference left after removing the mean offset.
- `offset_along_gap`: this subject's T1-FLAIR difference along the mean offset (its mean
  is the centroid distance).
- `modality_evidence`: how far the held-out mean-centred RBF probe puts this subject's
  two scans on their correct sides.

They are tested against:

- **Extraction descriptors** (`--run-dir` only): brain-mask volume per view in mL, the
  FLAIR/T1 mask-volume ratio and the masks' Dice overlap, and per view a **blur
  ratio**. The blur ratio is the blurriest image axis's mean absolute step between
  neighbouring voxels (one voxel inside the mask edge), over the mean of the other two
  axes' steps, plus which axis it was. It is about 1 for an isotropic scan and lower
  along the slice axis of a thick-slice acquisition such as ADNI-GO/2's 2D FLAIR. If
  registration tilted the slice axis off the grid, the effect spreads over two axes.
- **Site**, parsed from ADNI PTIDs (`<site>_S_<RID>`).
- **`--covariates`**: any CSV keyed by `--subject-column`, e.g. WMH volume (UC Davis
  tables), FLAIR acquisition type, scanner vendor, diagnosis. Columns with more than
  ten distinct numbers are numeric (Spearman); others are categories (Kruskal-Wallis
  over groups of at least `--min-group` subjects, effect epsilon-squared).

Benjamini-Hochberg q-values run over all tests of a split, and q < 0.1 is printed.
These are correlations: a mask-ratio association says the offset tracks the mask
mismatch, not that the mask causes it. The arm `--set shared_brain_mask=True` is the
interventional test.

### splits: training vs held-out subjects

Validation and training subjects through the same checks. The training split is
encoded with the validation preprocessing (`disable_augmentation`), so both are clean
images. Statistics that depend on n are size-matched: the probes and retrieval on the
larger split use `--matched-draws` random subsets of the smaller split's size. The
probe swap null runs once, on the smaller split, and the matched draws are read
against it. A gap present on held-out subjects but much smaller on training subjects
means the encoders align the pairs they were trained on and diverge on new ones.

## Outputs

Each run creates `modality_gap_causes_<timestamp>/` beside the run (or `--out-dir`):

- `report.json`: every check per split, the side-by-side comparison, settings (with any
  recipe values assumed), protocol and source checkpoint SHA-256.
- `features_<split>.npz`: raw content vectors, subject IDs and descriptors (`desc_*`),
  readable by both this tool and the diagnostic.
- `subjects_<split>.csv`: per-subject measures next to every covariate, for plotting.

## Verification

```bash
python -m unittest discover -s tests -p test_encoder_modality_gap_causes.py -v
python -m unittest discover -s tests -p test_encoder_modality_gap_runai.py -v
```

The tests plant one cause at a time and check that its check fires and the others do not:
- a low- vs high-variance offset;
- a concentration difference;
- a per-direction gain;
- an offset that would fake a shape difference if normalized first;
- a raw offset distorted by L2 normalization;
- an offset the loss prices, and a standardized code that cannot see it;
- planted numeric and categorical covariates.

They also check that the loss mirror matches the trainer's InfoNCE in both negative
modes and with `--standardize-code`. One test encodes a fake ADNI tree end to end: the
subject lists match `split.json`, augmentation is off (repeat runs are identical) and
the descriptors are written.
