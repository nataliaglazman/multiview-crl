# Locate the lesion-information bottleneck in a frozen encoder

Run the diagnostic on the same checkpoint as your earlier patch probe:

```sh
python -m eval.score_checkpoint \
  --run-dir results/dummy_infonce_wm_interior \
  --checkpoint model_best.pt \
  --lesion-analysis-only \
  --lesion-grids 1 4 \
  --num-samples 400 \
  --batch-size 8
```

To append it to the usual recovery/graph report instead, use `--lesion-analysis`.
The regular `--pooling` / `--patch-grid` flags configure the usual report;
`--lesion-grids` configures this additional analysis independently.

## Comparisons

For each view, compare full backbone features with projected content channels at
identical mean-pooling grids. For the original run these are 64 and 9 channels,
respectively. Grid 1 is the actual GAP representation and is always included.
Every image is encoded once per trained/untrained model; all grids come from
those captured tensors. No ground-truth masks are used to crop or weight encoder
features.

Two target families are fitted separately per axis:

- `latent_x/y/z`: the generating controls `z_content[2:5]`.
- `centroid_x/y/z`: the centroid of the re-rendered lesion support, before
  intensity rendering and blur, in the renderer's normalized coordinates [-1, 1].

These are different targets under `wm_interior`: its controls select quantiles
of anatomy-dependent admissible white-matter centres. The saved geometry-only
reference fits true centroids to latent controls. It helps check whether this
inverse mapping is linearly accessible, but is **not an upper bound**, since it
does not include the surrounding anatomy.

The original convolutional encoder is pooled after its pointwise projection.
The optional ResNet encoder is pooled before its nonlinear head, exactly as in
training. ResNet patch vectors apply the head after each bin's averaging; their
average need not equal the global encoding. At resolution 64, ResNet has only
2³ spatial features. Omitting `--lesion-grids` chooses GAP and grid 4, or the
native size when smaller. Explicit grids larger than the native map fail with
an explanatory error. For the original stride-4 encoder, a follow-up with
`--lesion-grids 1 4 8` checks finer spatial features at greater memory/CPU cost.

## Probe and controls

The probe uses the scorer's existing ridge protocol: five outer folds at seeds
0, 1 and 2. Feature standardization is fitted on each training fold, and RidgeCV
selects regularization per target inside that training fold. The outer test
fold is not used for fitting or selection. All stages, views and controls use
identical subject splits. Targets are batched to reuse the feature decomposition;
adding shuffled targets cannot change a real target's regularization choice.

Both trained and seeded untrained encoders receive identical images, verified
by a digest. Three jointly shuffled six-target copies are included by default;
set `--lesion-shuffles` or `--lesion-seed` to configure them. These are null
references, not permutation p-values. `--no-floor` explicitly omits the
untrained encoder but keeps shuffled controls. The untrained model is rebuilt
from the run's settings and seed, not loaded from a saved step-zero checkpoint.

Absent lesions are excluded consistently across all comparisons and listed in
the report. At least 20 valid subjects are required. A coordinate constant in
any test fold has undefined R², reported as `n/a` / JSON `null`, rather than a
spurious perfect score. Negative R² is retained. `seed_std` is spread across
CV-seed means, not a confidence interval.

This is a frozen diagnostic on the scorer's validation split. If `model_best.pt`
was chosen using that same split, it is not an untouched final test set. Larger
backbone feature sets and different grids have different statistical demands;
a lower ridge score alone does not establish absence of information. No encoder
training uses the diagnostic targets. Parameters and buffers are hashed before
and after extraction; checkpoints are never changed.

## Saved output

The terminal prints mean xyz R² for each target family, alongside the untrained
and shuffled references. Without `--out`, a timestamped `score_lesion_*.json`
is saved in the run directory. Use `--out PATH.json` to choose the report path.
The JSON contains the complete analysis under `lesion_analysis`, including
per-axis results, sample IDs, target permutations and state/input digests.
Two CSV files with the same prefix are also saved:

- `_lesion_scores.csv`: per-axis trained/untrained R², all null repeats, and
  differences versus the untrained arm, matching backbone, and GAP.
- `_lesion_targets.csv`: subject IDs, lesion sizes, validity, generating controls
  and physical centroids.

`delta_backbone` is projected minus backbone R² at the same grid;
`delta_gap` is spatial minus GAP R² at the same stage. These are matched score
differences, without confidence intervals.

Interpret the paired comparisons:

- Backbone recovery exceeding projected recovery suggests a readout bottleneck.
- Spatial recovery exceeding GAP recovery suggests averaging loses accessible
  location information.
- Centroid recovery exceeding latent recovery suggests the generator's inverse
  mapping contributes to the original poor latent scores.
- Poor scores everywhere leave representation quality, linear-probe limits and
  sample size unresolved; they do not prove that the input image lacks the lesion.
