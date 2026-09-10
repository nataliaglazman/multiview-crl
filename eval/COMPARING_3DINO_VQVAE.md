# Comparing 3DINO and VQ-VAE evaluation

Last revised 10 September 2026. This describes the current source, not a particular saved
experiment; historical results also depend on the code revision and flags used to produce
them.

There are now two ways to put a DINO number next to a VQ-VAE number, and they are not
equally trustworthy.

* **Matched** — export both representations as feature bundles and score them through one
  function (`eval.export_vq_bundle` → `eval.compare_bundles`). The CV splits, permutation
  nulls, PCA rule, probe and floor handling are the same objects doing the same work, and
  row identity is verified rather than assumed. Prefer this.
* **Side by side** — run `eval.identifiability_report` and `eval.run_3dino_identifiability`
  separately and read the two reports together. Everything below about protocol differences
  applies, and each difference is yours to align by hand.

## The matched protocol

```bash
# 1. The VQ-VAE run, as a bundle. --block all is the block comparable to a DINO
#    embedding; --block content is the learned partition DINO has no counterpart for.
python -m eval.export_vq_bundle --run-dir results/synthetic/YOUR_VQ_RUN \
  --checkpoint vqvae_model.pt --level 0 --pooling gap --block all \
  --num-samples 2000 --raw-grid 4,4,4 --out results/bundles/vq_all.npz

python -m eval.export_vq_bundle --run-dir results/synthetic/YOUR_VQ_RUN \
  --checkpoint vqvae_model.pt --level 0 --pooling gap --block content \
  --num-samples 2000 --raw-grid 4,4,4 --out results/bundles/vq_content.npz

# 2. Its untrained twin, for the floor.
python -m eval.export_vq_bundle --run-dir results/synthetic/YOUR_VQ_RUN \
  --random-init --model-seed 0 --level 0 --pooling gap --block all \
  --num-samples 2000 --out results/bundles/vq_all_floor.npz

# 3. The DINO embeddings, from the SAME run directory and the same --num-samples.
python -m eval.run_3dino_identifiability \
  --three-dino-repo ../3DINO --three-dino-weights /path/to/pretrained.pth \
  --run-dir results/synthetic/YOUR_VQ_RUN --num-samples 2000 \
  --token-pool mean --with-floor --output-dir results/3dino_matched

# 4. One table, one protocol.
python -m eval.compare_bundles \
  --bundles vq_all=results/bundles/vq_all.npz \
            vq_content=results/bundles/vq_content.npz \
            dino=results/3dino_matched/embeddings.npz \
  --floors  vq_all=results/bundles/vq_all_floor.npz \
            dino=results/3dino_matched/random_init.npz \
  --equal-width --with-graph --alphas 0.05 --diagnostic-alpha 0.05 \
  --probe-kind ridge --seeds 0 1 2 --n-null 3 \
  --out results/matched/compare.json --csv results/matched/compare.csv
```

`--run-dir` on both sides is what makes step 1 and step 3 render the same brains: both call
`run_dci_synthetic.build_synthetic_test_set` on that run's `settings.json`. Step 4 then
proves it rather than trusting it — see **Row identity** below.

Pair the poolings deliberately. VQ `--pooling gap` goes with DINO `--token-pool mean`;
VQ `--pooling 4,4,4` goes with `--token-pool grid --grid-size 4`. Keep `--token-pool cls`
as a separate baseline rather than pairing it with anything. A 4³ grid is 65,536 features
for ViT-Large and costs far more RAM and probe time than CLS, and spatial DINO extraction
must be repeated if existing NPZs contain only CLS.

## What the matched path guarantees

**One probe.** Every bundle is scored by `dinov3_identifiability.score` with one options
object. `KFold(shuffle=True, random_state=seed).split(X)` partitions row indices and never
looks at feature values, so bundle *k*'s fold *i* holds the same rows as bundle *j*'s; the
permutation nulls are redrawn from a fixed `--null-seed` per bundle, so both are differenced
against the same permutation. Scoring a bundle alone and scoring it beside another give
identical numbers, which is asserted in `tests/test_matched_bundles.py`.

**Row identity, verified.** Every bundle carries a SHA-256 digest of its ground-truth factor
arrays (`eval/bundle_identity.py`), and `compare_bundles` refuses to build a table from
bundles whose digests disagree. Equal `--num-samples` is not evidence of the same rows: two
draws of one generator at different seeds give 2000 rows each and share none of them.
Bundles written before the digest existed still compare, because the digest is a pure
function of arrays they already carry. `--allow-row-mismatch` downgrades the refusal to a
warning; the output is then not a comparison.

**Equal probe capacity.** `--probe-dim auto` picks a width per block, so two bundles can
reach the table with different capacity and part of the difference between them is that.
The effective width is printed per bundle, a mismatch is called out in the report, and
`--equal-width` pins every bundle — floors included — to the narrowest block's width.

**Floors that stay with their own architecture.** Each bundle's Δfloor is computed against
the floor supplied for that bundle. One architecture's floor is never subtracted from
another's, which would not be a baseline correction at all.

**A frozen content mask.** `export_vq_bundle` freezes the Gumbel mask's channel selection to
the first batch by default. With `mask_mode=onthefly` the mask is redrawn on every forward,
so without this a stacked feature column does not describe one physical channel. The
extractor now warns whenever the selection actually moves, on every code path.
`identifiability_report --freeze-content-mask` opts the report into the same behaviour;
it is off there by default because turning it on moves numbers for exactly those
checkpoints, and the reports already published were produced without it.

## Causal discovery across representations

`--with-graph` runs the PC panel on every bundle and scores each recovered skeleton
**against the true SCM adjacency** the generator used. The section carries:

- one row per representation, plus one per untrained floor;
- a `truth (ceiling)` row — the identical panel on the ground-truth factors themselves,
  computed once because it depends only on the factors every bundle shares. It is a
  finite-sample reference, not a strict upper bound: a representation can beat it by
  decoding factors into columns PC happens to find easier to separate;
- **two alpha selections**. The prespecified `--diagnostic-alpha` row is the one a
  head-to-head reads off, because every source is tested at the same threshold. The
  best-F1 row is each source at its own most flattering alpha, selected by looking at the
  answer — the alpha column is part of that result. Quote the prespecified row and use
  `--alphas 0.05 --diagnostic-alpha 0.05` so the sweep cannot drift between models;
- **partial R² per factor**, which is where a graph difference usually comes from: a
  representation that reads a factor only through its SCM parents scores high raw and near
  zero partial, and PC then sees a column that is mostly the parent;
- **orientation vs the true CPDAG** at the diagnostic alpha. The CPDAG, not the DAG,
  because PC identifies a Markov equivalence class — on the default `chain` SCM that class
  is entirely undirected, and scoring against the DAG would charge every estimate for two
  edges no observational method can orient.

`--equal-width` matters more here than for the probes. `--probe-dim` does not touch the
graph; the readout has its own rule (`run_causal_recovery.readout_width`) that caps at each
block's feature count, so a 48-channel VQ block and an 18,432-dim embedding otherwise get
readouts of 48 and 64. `--equal-width` pins both to what the narrowest block can reach and
the report prints the widths so a mismatch is visible. The ceiling is exempt by
construction.

`compare_graph.csv` (written next to `--csv`, taking its stem) holds one row per source
and alpha, so the whole sweep can be replotted without re-running PC.

Two limits that no flag removes. PC runs on a *supervised readout* of each representation's
decoded factors, so this measures how well the SCM survives that representation, not causal
discovery from raw features. And the default readout is in-sample — it decodes the rows it
was fit on, with the true labels — so pass `--holdout-readout` for a held-out graph at the
cost of 70% of the rows. See `eval/CAUSAL_EVALUATION.md`.

## What it still does not equalise

These are properties of the representations and the pipelines, not of the scorer, and no
flag removes them. Report them; do not net them out.

| Item | 3DINO | VQ-VAE | Status |
| --- | --- | --- | --- |
| Representation | Full final-layer embedding | Pre-codebook encoder features at one level | Different objects. `--block all` is the comparable one; `--block content` answers a question DINO cannot be asked. Neither is "VQ code identifiability" — the codes are not what is probed. |
| Input preprocessing | Percentile clipping, mapping to [-1,1], resize to 112³ | Generator-normalized volumes at their generated resolution | Recorded in each bundle's `meta`, never reconciled. Compare preprocessing ablations separately; a fine-tuned model's saved preprocessing overrides pooling flags. |
| Normalization location | n/a | GAP/patch pools are formed *before* optional `content_norms`; the stats path reads encoder outputs *after* them | Recorded as `meta.normalization_location`. With SplitGroupNorm enabled, a pooling comparison is also a normalization comparison. |
| Foreground masking | n/a | Neither VQ extraction path passes the dataset's foreground mask to `forward` | Recorded as `meta.foreground_mask: false`. For `latent_mask=True` checkpoints the model warns that evaluation does not match training. Unresolved. |
| Metrics | Content/style decodability, block-MCC, graph diagnostics | Additionally leakage, sufficiency, view invariance, optional DCI, nonlinear parent adjustment | Missing metrics are not zero. DINO style decodability does not establish content/style disentanglement. |
| PCA | Fitted before outer CV, without labels | Same | Matches between scripts, and is transductive in both. `--probe-dim 0` removes it. The nonlinear VQ parent panel uses train-fold-only PCA and must not be mixed with these scores. |

## Fixed: the batched-probe mismatch

`identifiability_report.per_factor_scores` fits each real and shuffled target separately with
`cv_probe_r2`; `dinov3_identifiability.factor_recovery` and `run_dci_compare._score_block`
batch them through `cv_probe_r2_multi`. For ridge these agreed. For kernel and MLP they did
not: one `GridSearchCV` selected a shared alpha across all real and null targets, and one MLP
was trained on them jointly, so a target's score depended on which other targets rode along —
and adding permutation nulls therefore moved the real score.

Measured on 120×12 Gaussian features, targets `X[:,0] + .05*noise` and
`.2*sin(3*X[:,1]) + noise` plus one permuted copy of each, 3-fold CV at seed 0:

| Probe | Separate fit | Batched fit (before) | Batched fit (now) |
| --- | ---: | ---: | ---: |
| Ridge | 0.997512 | 0.997512 | 0.997512 |
| Kernel | 0.852159 | 0.709798 | 0.852159 |
| MLP | 0.936418 | 0.524527 | 0.936418 |

`cv_probe_r2_multi` now routes only the kinds in `BATCHABLE_PROBES` (ridge, whose
`alpha_per_target=True` fit is column-wise identical to a single-target fit) through the
multi-output path, and fits every other kind one target at a time, reusing the per-fold
scaler. Column *j* equals `cv_probe_r2` on column *j* to 1e-12 for all three kinds; the
equality and the `n_null`-invariance are asserted in `tests/test_identifiability_metrics.py`.

## The score columns are not interchangeable

For the VQ Python report and DINO's factor table:

| Quantity | VQ report JSON | DINO / `compare_bundles` JSON |
| --- | --- | --- |
| CV factor R² | `per_factor[factor].r2_raw` | `content[factor].real` |
| CV R² minus shuffled-label null | `per_factor[factor].r2` | `content[factor].gap` |
| Gap minus untrained architecture gap | derived from `res` and `floor` | `content[factor].delta_floor` |

Choose the same VQ `by_pooling` entry before comparing these columns. The notebook
`identifiability_report.ipynb` is a third protocol: its `*_raw` column means the **raw-image
baseline**, not the unadjusted encoder R², and its `*_delta` subtracts that image baseline.

Partial R² also means different things in different places:

- `run_causal_recovery` and the DINO graph panel: linear parent regression on all samples,
  then residual decoding with Ridge(alpha=1), a single 70/30 split (seed 0), train-only
  scaling, the original unreduced feature block. No permutation null, no floor subtraction.
- The VQ report's legacy partial table: linear residuals, five-fold RidgeCV across seeds
  0/1/2, the report's selected pooling/PCA, a permutation-null subtraction, and floor
  subtraction in the printed partial column when a floor is available.
- `--parent-adjustment nonlinear`: a separate cross-fitted nuisance model with fold-local
  preprocessing. DINO does not implement this panel.

The pipeline summary's `partial_r2` comes from the **graph** panel; compare it with
`run_causal_recovery.partial_r2_mean`, not the VQ report's printed partial. These are
residual-target decodability, not the incremental test R² of adding an embedding to a
parent-only predictor.

## Graph controls

`--probe-dim` does **not** control graph scoring. `--readout-dim` controls the supervised
decoder whose predicted factors enter PC, and does **not** change the raw/partial R² probes.
Setting these to the same number does not make all three probes identical.

For comparable graphs use identical labels, factor ordering, adjacency, N, `--readout-dim`,
`--holdout-readout`, independence test and alpha. Both scripts default to in-sample graph
readouts and pick the headline alpha by true-graph F1, so use
`--alphas 0.05 --diagnostic-alpha 0.05` for one prespecified alpha. With
`--holdout-readout`, only 30% of N reaches PC (600 of 2000).

Headline SHD is skeleton SHD (FP+FN); incident per-factor SHDs count each edge error twice
across factors. DINO enables CPDAG orientation diagnostics by default; the VQ causal CLI
requires `--orientation`. This does not change headline skeleton scores. VQ's optional
`--factor-rescue` is not exposed by the DINO pipeline. DINO drops constant factors before
graph scoring, so verify the reported factor set before comparing a degenerate run.

For capped KCI use the same causal-learn build on both sides. The DINO pipeline rejects
versions whose PC function cannot honor `max_k`; the VQ causal script lacks that capability
check. Local causal-learn 0.1.4.0 silently ignored the cap; 0.1.4.8 honored it.

## Side-by-side protocol, if you are not using bundles

This brings the standalone scripts closer without resolving the items above. Choose the
same explicit content level for both VQ commands.

```bash
python -m eval.identifiability_report \
  --run-dir results/synthetic/YOUR_VQ_RUN --checkpoint vqvae_model.pt \
  --num-samples 2000 --causal match --level 0 \
  --poolings 4x4x4 --factor-pooling patch --freeze-content-mask \
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

`--floor-seeds 1` matches DINO's single untrained draw. For uncertainty, raise it on the VQ
side and extend DINO to multiple draws. Never subtract one architecture's floor from the
other's.
