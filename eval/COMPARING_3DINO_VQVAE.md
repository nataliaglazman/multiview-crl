# Comparing 3DINO and VQ-VAE evaluation

Code audit: 10 September 2026. This compares the current source, not a particular
saved experiment. Historical results also depend on the code revision and flags
used to produce them.

The 3DINO graph panel already calls `run_causal_recovery.evaluate_arrays`.
Its identifiability table uses the same core metric module as
`identifiability_report.py`, but the two reports do not yet form a fully matched
benchmark. The notebook `identifiability_report.ipynb` is a third protocol.

## Differences that affect a comparison

| Item | 3DINO pipeline | VQ-VAE evaluations | How to align |
| --- | --- | --- | --- |
| Samples | 500 by default, test split | Python report: 2000, test. Causal script: saved `synthetic_num_test`, test. Notebook: 500, validation. | Explicitly set the same N and split; check labels and adjacency, not just directory names. |
| Generator | Shared `build_synthetic_test_set` when `--run-dir` is supplied | Python report and causal script share this helper. Notebook manually constructs the dataset and omits newer renderer settings. | Use the Python entry points with the same run settings. Match `--causal match`; IID/shuffled reports are a separate experiment. |
| Representation | Full final-layer embedding, default CLS token | Selected encoder level and content channels; pre-codebook features, not quantized codes | Report full-representation recovery and content-block recovery separately. DINO has no learned content/style partition. |
| Pooling | One pooling for every factor: CLS, token mean, or token grid | Python report routes morphometry to GAP, lesion coordinates to patches, and some style factors to statistics. Default patch grid is 8³. Causal script uses 4³ for every factor. | Use one explicit VQ pooling for the matched comparison. Pair token mean with GAP, or token grid with a matching spatial grid. Keep CLS as a separate baseline. |
| Feature dimension | CLS has 1024 features; grid pooling is wider | Depends on level, content-channel count and pooling | Specify probe/readout dimensions and inspect effective widths. `auto` is not guaranteed to produce equal widths. |
| Input preprocessing | Additional percentile clipping, mapping to [-1,1], and resizing to 112³ by default | Generator-normalized volumes at their generated resolution | Record these differences. Compare preprocessing ablations separately; preserve a fine-tuned model's saved preprocessing. |
| Random baseline | Optional, one draw by default (`--with-floor`, seed 0) | Python report defaults to three untrained draws; causal script has no untrained baseline | For an immediate single-draw match use `--floor-seeds 1` on VQ. For uncertainty, extend DINO to multiple draws. Never subtract one architecture's floor from the other. |
| Metrics provided | Content/style decodability, block MCC, graph diagnostics | Python report additionally measures content/style leakage and sufficiency, view invariance, optional DCI and nonlinear parent adjustment | Missing metrics are not zero. DINO style decodability does not establish content/style disentanglement. |

Sources: `dinov3_embed_synthetic._generator_args/build_dataset`,
`run_dci_synthetic.build_synthetic_test_set`, `models.three_dino.prepare_volumes`,
`identifiability_report.main/score_run`, `run_dci_compare.FACTOR_POOLING`,
`run_causal_recovery.extract_content/evaluate_run`, and notebook setup cells.

## The score columns are not interchangeable

For the Python VQ report and DINO's factor table:

| Quantity | VQ report JSON | DINO report JSON |
| --- | --- | --- |
| CV factor R² | `per_factor[factor].r2_raw` | `content[factor].real` |
| CV R² minus shuffled-label null | `per_factor[factor].r2` | `content[factor].gap` |
| Gap minus untrained architecture gap | Derived from `res` and `floor` | `content[factor].delta_floor` |

Choose the same VQ `by_pooling` entry before comparing these columns.
The notebook's `*_raw` column instead means the **raw-image baseline**, not the
unadjusted encoder R². Its `*_delta` subtracts that image baseline.

There are also different meanings of partial R²:

- `run_causal_recovery` and the DINO graph panel: linear parent regression fit
  on all samples, then residual decoding with Ridge(alpha=1), a single 70/30
  split (seed 0), train-only scaling, and the original unreduced feature block.
  No permutation null or untrained-floor subtraction is applied to this score.
- VQ Python report's legacy partial table: linear residuals, five-fold RidgeCV
  across seeds 0/1/2, the report's selected pooling/PCA, a permutation-null
  subtraction, and floor subtraction in the printed partial column when a
  floor is available.
- VQ `--parent-adjustment nonlinear`: a separate cross-fitted nonlinear
  nuisance model and fold-local feature preprocessing. DINO does not implement
  this panel.
- The notebook's current partial table: CV residual R² through
  `identifiability_metrics.partial_r2_vs_parents`, without the Python report's
  null/floor corrections or automatic PCA.

The pipeline summary's `partial_r2` comes from the **graph** panel. Compare it
with `run_causal_recovery.partial_r2_mean`, not the VQ report's printed partial.
These legacy scores measure residual-target decodability; they are not the
incremental test R² of adding an embedding to a parent-only predictor.

## A confirmed nonlinear-probe implementation mismatch

`identifiability_report.per_factor_scores` fits each real and shuffled target
separately with `cv_probe_r2`. DINO's `factor_recovery` batches all of them through
`cv_probe_r2_multi`.

For ridge, `alpha_per_target=True` preserves independent alpha selection, so the
paths agree. For kernel probes, one GridSearchCV selects a shared alpha across
all real and null targets. For MLP, a shared network is trained on those targets
jointly. Neither is equivalent to independent single-target probes; changing
the number of null targets can therefore change the real score itself.

An identical-array check used seed 19, 120 rows, 12 independent Gaussian
features, targets `X[:,0] + .05*noise` and `.2*sin(3*X[:,1]) + noise`, and one
permuted copy of each target. Three-fold CV, seed 0:

| Probe | First target, separate fit | First target, batched fit | Absolute difference |
| --- | ---: | ---: | ---: |
| Ridge | 0.997512 | 0.997512 | < 5e-15 |
| Kernel | 0.852159 | 0.709798 | 0.142360 |
| MLP | 0.936418 | 0.524527 | 0.411891 |

These are implementation checks on synthetic arrays, not model performance
estimates. Use ridge for current cross-script comparisons. A shared scorer
should use separate kernel/MLP fits for each real/null target, while retaining
the equivalent batched optimization for ridge.

## PCA and graph controls

Both Python factor reports fit legacy PCA before outer CV, without labels.
This matches between scripts but is transductive. Explicit `--probe-dim K`
aligns feature budgets only if both blocks contain at least K dimensions.
`--probe-dim 0` removes this reduction. The nonlinear VQ parent panel uses
train-fold-only PCA and should not be mixed with those legacy scores.

`--probe-dim` does **not** control graph scoring. `--readout-dim` controls the
supervised decoder whose predicted factors enter PC, and does **not** change
the raw/partial R² probes. Setting these flags to the same number does not
make all three probes identical.

For comparable graphs use identical labels, factor ordering, adjacency, N,
`--readout-dim`, `--holdout-readout`, independence test, and alpha. Both scripts
default to in-sample graph readouts and choose the headline alpha by true-graph
F1. Use `--alphas 0.05 --diagnostic-alpha 0.05` for one prespecified alpha.
With `--holdout-readout`, only 30% of N reaches PC (600 of 2000).

Headline SHD is skeleton SHD (FP+FN); incident per-factor SHDs count each edge
error twice across factors. DINO enables CPDAG orientation diagnostics by
default; the VQ causal CLI requires `--orientation`. This does not change
headline skeleton scores. VQ's optional `--factor-rescue` is not exposed by the
DINO pipeline. DINO drops constant factors before graph scoring, so verify the
reported factor set before comparing a degenerate run.

For capped KCI use the same causal-learn build on both sides. The DINO pipeline
rejects versions whose PC function cannot honor `max_k`; the VQ causal script
currently lacks that capability check. Local causal-learn 0.1.4.0 silently
ignored the cap; 0.1.4.8 honored it in the preceding pipeline tests.

## Conditional mismatches inside the VQ evaluators

These depend on checkpoint configuration and should be checked before treating
the VQ scripts as a single reference implementation:

1. **Batch-dependent content selection.** `run_causal_recovery.extract_content`
   freezes output channel indices from sample 0. `dci._extract_synthetic_representations`
   reads each batch's mask. With `mask_mode=onthefly`, different batches can
   select different physical channels for the same probe column. Their default
   batch sizes also differ (8 versus 32). Freezing extracted indices alone does
   not freeze masks used internally between encoder levels.
2. **Normalization location.** VQ GAP/patch pools are formed before optional
   `content_norms`, whereas the stats path reads `encoder_outputs` after those
   norms. The causal script and notebook hook the raw encoder outputs.
   With SplitGroupNorm enabled, this makes a pooling comparison also a
   normalization comparison.
3. **Foreground masking.** Neither VQ extraction path passes the dataset's
   foreground mask to forward. For `latent_mask=True` checkpoints, the model
   warns that evaluation does not match training. A corrected hook-based path
   must also deliberately choose pre- or post-mask features.

The causal and Python identifiability scripts otherwise both examine
pre-codebook encoder features. Calling these scores “VQ code identifiability”
would overstate what is measured.

## Practical comparison using existing flags

This spatial protocol brings the existing scripts closer. Substitute the run,
checkpoint, and repository paths. Choose the same explicit content level for
both VQ commands. It still compares VQ content channels with the full DINO
representation; it does not resolve the conditional issues above.

```bash
python -m eval.identifiability_report \
  --run-dir results/synthetic/YOUR_VQ_RUN --checkpoint vqvae_model.pt \
  --num-samples 2000 --causal match --level 0 \
  --poolings 4x4x4 --factor-pooling patch \
  --probe-kind ridge --probe-dim 64 --seeds 0,1,2 --n-null 3 \
  --floor-seeds 1 --out results/vq_identifiability_matched.json

python -m eval.run_causal_recovery \
  --run-dirs results/synthetic/YOUR_VQ_RUN --checkpoint vqvae_model.pt \
  --num-samples 2000 --level 0 --pooling 4,4,4 \
  --readout-dim 64 --holdout-readout \
  --alphas 0.05 --diagnostic-alpha 0.05 --indep-test fisherz \
  --output-dir results/vq_causal_matched

python -m eval.run_3dino_identifiability \
  --three-dino-repo ../3DINO --three-dino-weights /path/to/pretrained.pth \
  --run-dir results/synthetic/YOUR_VQ_RUN \
  --num-samples 2000 --eval-views 1 --token-pool grid --grid-size 4 \
  --probe-kind ridge --probe-dim 64 --seeds 0 1 2 --n-null 3 \
  --with-floor --floor-seed 0 --readout-dim 64 --holdout-readout \
  --alphas 0.05 --diagnostic-alpha 0.05 --indep-test fisherz \
  --no-orientation --volume-batch 2 --device cuda \
  --output-dir results/3dino_spatial_matched
```

Spatial DINO extraction must be repeated if existing NPZs only contain CLS.
A 4³ grid has 65,536 features for ViT-Large and costs substantially more RAM
and probe time than CLS. A second, cheaper global protocol uses VQ GAP and
DINO `--token-pool mean`; choose K at or below the narrower block's width.
A fine-tuned encoder's saved preprocessing overrides pooling flags; inspect
the resulting metadata instead of assuming the command changed its pooling.

For a fully direct benchmark, add a shared feature export and scoring path:
one fixed evaluation dataset, stored factor/image hashes and row IDs; VQ full
and content-only features plus DINO features; identical CV splits, null
permutations, train-only preprocessing, per-target probes, parent adjustment,
and baseline seeds. Score all representations through those same functions.
This is the remaining implementation work; the commands above do not claim
to provide it.
