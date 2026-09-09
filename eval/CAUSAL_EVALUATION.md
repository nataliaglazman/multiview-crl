# Optional causal evaluation diagnostics

## Batch notebook graph recovery

`eval/run_causal_recovery.py` runs the notebook's section 7i over a list of
VQVAE result directories. Each directory must contain `settings.json` and
`vqvae_model.pt` (or the filename passed with `--checkpoint`). Run it in the
training environment with the additional `causal-learn` package installed:

```bash
python -m pip install causal-learn
python -m eval.run_causal_recovery \
  --run-dirs results/run_a results/run_b \
  --num-samples 500 --output-dir results/causal_recovery

# Quoted globs are supported. Each matching path should be an individual run.
python -m eval.run_causal_recovery --run-dirs 'results/experiment/*' \
  --checkpoint vqvae_best.pt --device cuda

# Alternatively, one run path or glob per line; blank lines and # comments ignored.
python -m eval.run_causal_recovery --runs-file runs.txt
```

Paths inside `runs.txt` are relative to that file; command-line paths are relative
to the working directory. Duplicate directories are evaluated once. Non-causal
runs are skipped; errors are recorded and processing continues. An error in any
run produces exit code 1 after saving the reports. Explicit checkpoint filenames
are required to exist in each run; there is no fallback to a different checkpoint.

The script regenerates each run's matched test distribution and uses its actual
SCM adjacency. It captures raw view-1 encoder maps, fixes the content channel mask
using sample 0 as in notebook section 4, and defaults to `--pooling 4,4,4`.
`--pooling gap` is also available. The default encoder level is the first
`content_style_levels` entry (otherwise 0); override it with `--level`.

`causal_recovery.csv` contains one summary row per run: raw and residual R² means,
best alpha, skeleton precision/recall/F1, false positives/negatives, skeleton SHD
(missing plus extra edges), and `exact_match`. The JSON additionally stores
per-factor scores/parents, the true DAG, every alpha's estimated skeleton and
metrics, generator settings, and protocol metadata. Both files are updated after
each run. By default they are saved under `results/causal_recovery`.

A compact comparison table is also printed at the end and saved as
`causal_recovery_summary.txt`, with columns for directory name, F1, precision,
recall, SHD, mean partial R², and status. Skipped or failed runs show unavailable
metrics. The CSV includes both `directory_name` and the full `run_dir`.
To produce this table from an existing JSON report without rerunning evaluation:

```bash
python -m eval.run_causal_recovery --from-json results/causal_recovery/causal_recovery.json
```

This regenerates the reports alongside the input JSON; use `--output-dir` to
write them elsewhere.

### Locate factors that degrade during training

Every evaluation and JSON replay now also writes `causal_recovery_factors.txt`
and `causal_recovery_factors.csv`. The text report ranks each run's factors from
largest to smallest decline in partial R² relative to a reference. The CSV keeps
one row per run and factor, including raw values, deltas, and graph neighbors.

For reports you have already generated, no checkpoint evaluation is needed:

```bash
python -m eval.run_causal_recovery \
  --from-json results/causal_recovery/causal_recovery.json \
  --reference-run early_checkpoint
```

`--reference-run` accepts a unique directory basename or the exact saved directory
path. It defaults to the first run with factor scores. Supply directories in
training order when comparing checkpoints; the script preserves input order and
does not infer steps from names. Glob matches use lexicographic order, so use an
explicit list or a runs file when names such as `step_2` and `step_10` would sort
incorrectly. Comparisons to a reference are not tests for monotonic degradation.

The diagnostics include:

- **Raw R² and partial R²**, and their changes from the reference, per named factor.
  Both falling supports loss of linearly decodable information about that factor.
  Partial R² falling while raw R² remains high suggests the probe increasingly
  relies on parent-related information. Nonlinear parent effects remain a caveat.
- **Drop %**: the factor's share of the summed decreases in partial R². Improving
  factors are excluded from this denominator, so improvements cannot hide losses.
  CSV `mean_partial_delta_contribution` is the signed factor delta divided by the
  number of factors; these contributions sum to the change in the mean partial R².
  Neither quantity attributes the change in graph F1 to that factor.
- **Incident graph F1/precision/recall/SHD**, missing and false neighbors, and the
  change in incident SHD. Graph comparisons use `--diagnostic-alpha 0.05` by
  default, fixed across runs. Each incorrect undirected edge is incident on TWO
  factors, so summing incident SHD gives twice global SHD. Shared errors alone
  cannot establish which endpoint's representation is responsible.
- On **fresh evaluations**, **PC R²** is an additional held-out RidgeCV decoding
  score after PCA/scaling at the graph readout's dimension. This diagnostic fits
  preprocessing on training samples only. It can expose a loss after compression
  even when the full-feature probe remains strong. It is distinct from the
  original in-sample predictions fed to PC. JSON and CSV also record each decoded
  factor's variance relative to its true variance and its correlation with truth.

To add a more direct graph sensitivity check during a fresh evaluation:

```bash
python -m eval.run_causal_recovery \
  --run-dirs results/early_checkpoint results/middle_checkpoint results/late_checkpoint \
  --num-samples 500 --factor-rescue --diagnostic-alpha 0.05
```

**Factor rescue** replaces one column of the decoded factors with its true values,
keeps all other decoded columns, and reruns PC at the fixed diagnostic alpha.
Positive `rescue_shd_reduction` / `rescue_f1_gain` mean that this replacement
improved the global graph. This is an oracle sensitivity check, not a causal
intervention or proof of the training mechanism. Repairs may fail to help when
multiple decoded factors degrade together; the individual gains are not additive.
It adds one PC run per factor. Full repaired graphs and failures are stored in
JSON. Old JSON lacks the decoded samples, so rescue and PC R² cannot be computed
with `--from-json`; unavailable entries remain blank rather than being estimated.

The fixed diagnostic alpha is evaluated even if it is outside `--alphas`, but is
excluded from headline best-alpha selection unless explicitly in that sweep.
Old JSON that lacks the fixed alpha shows unavailable graph diagnostics; choose
an alpha present in the saved sweep or reevaluate. Known differences in ground
truth DAG, sample count, level, pooling, or saved synthetic settings suppress
reference deltas and are labelled `incompatible`. Missing metadata is labelled
`unverified`. This check cannot establish that different directories belong to
one training trajectory; compare matched data, preprocessing, architecture and
checkpoint lineage. Scores have sampling/probe variability; small changes alone
do not establish collapse or its cause.

The calculations intentionally preserve section 7i: parent regressions use all
samples; Ridge probes use a fixed 70/30 train/test split; scaled/PCA features feed
supervised RidgeCV factor predictions on the same samples used to fit them; PC
uses Fisher-Z (see `--indep-test`) with alphas `0.01 0.05 0.1 0.2`, retaining the last
alpha in a tie.
Use `--alphas 0.05` to report a single prespecified alpha. These metrics describe
**undirected skeleton recovery**, not recovery of causal directions. The default
best F1 is selected against the known truth, and the graph readout is in-sample,
so it is an optimistic diagnostic rather than held-out causal discovery evidence.
Linear residualization need not remove nonlinear parent effects. The residual
score is the notebook's "partial R²", not a nested-model partial R² statistic.
For an empty true and estimated graph, F1 remains 0 as in the notebook, while
`exact_match` is true and SHD is 0. Constant decoded factors or numerical PC
failures retain the raw/partial factor scores and record errors per alpha. If no
requested alpha can be scored, the run has `partial` status, graph scores are
unavailable, and the batch returns exit code 1 after writing all reports.

## Identifiability report diagnostics

The entry point is `eval/identifiability_report.py`. Existing defaults remain
`--causal match --parent-adjustment legacy`; neither option changes training or the
generative model.

Run the two diagnostics separately on each checkpoint:

```bash
python -m eval.identifiability_report --run-dir RUN \
  --causal match --parent-adjustment nonlinear --out matched_nonlinear.json

python -m eval.identifiability_report --run-dir RUN \
  --causal shuffled --shuffle-seed 0 --out shuffled_marginals.json
```

Keep sample count, generator seed/settings, pooling, probe options and floor seeds
the same across models. `--from-json PATH` replays all new tables without fitting.

## Nonlinear parent adjustment

`--parent-adjustment nonlinear` replaces the legacy partial column with table 1c.
For each content factor with observed SCM parents, a histogram gradient boosting
regressor estimates its conditional mean from those parents. Each outer CV test
fold is held out from all nuisance fitting. Within the outer training set, three
inner folds produce out-of-fold parent predictions, so training residuals do not
come from in-sample fitted values. Parentless targets are left unchanged.

The encoder stays frozen. PCA and scaling for table 1c are fitted on each outer
training fold only. The probe predicts the parent residual from content and,
unless `--no-leakage` is set, from style. Targets, splits and nuisance fits are
reused across poolings and untrained twins. `--probe-kind`, `--probe-dim`, `--seeds`,
`--n-null`, `--factor-pooling` and `--n-jobs` also apply to this section.

The table and JSON contain:

- `parent_r2`: held-out R² of the parent predictor, to assess nuisance fit quality.
- `full_r2_raw`: original-target decoding with the same fold-local preprocessing.
- `r2_raw`: parent-residual decoding R².
- `r2_null` and `r2`: within-split permutation baseline and residual R² minus that
  baseline. Test targets never move into probe training targets.
- Learned gap and across-twin floor spread in the printed table. JSON retains all
  scores under `parent_adjusted.content` and `parent_adjusted.style`, including
  each pooling. `r2_std` is the residual score's spread across CV seeds, not a
  confidence interval.

This additional section uses its own full-target reference and untrained floor;
the other tables and overall verdict keep their existing scoring pipeline. A
large style residual score suggests that style carries content information beyond
what the fitted parent predictor explains. The nuisance model is approximate:
remaining nonlinear dependence, heteroscedasticity and finite-sample errors can
still affect these scores. Residual R² is not an estimate of exogenous noise
recovery or proof of independence. Full and residual targets have different
variances, so subtracting their R² does not estimate a mediated fraction.

This requires `--causal match`, at least 20 samples, and an observed-parent SCM.
Without that graph, the report marks adjustment unavailable; hierarchical latent
confounding alone does not supply such a graph. Use substantially more than the
minimum samples for useful nonlinear fits. This mode adds probe and nuisance
fitting cost. The nuisance model uses 150 boosting iterations, 15 leaves, minimum
leaf size 10 and L2 penalty 1; it is fixed rather than selected using test labels.

## Marginal-preserving independence stress test

`--causal shuffled` first draws the same latent population as `--causal match`.
It independently permutes each named `z_content` column across subjects, then
re-renders both views with the existing renderer. The sorted values in every
content column remain exactly identical to the source sample. Style factors,
deformation/fissure/lesion fields, rendering seeds, renderer options and
normalization stay attached to each subject. Fixed-reference normalization uses
the original matched population's constants.

This is a finite-sample approximation to the product of empirical content
marginals. It breaks systematic content dependencies, including associations
with fixed subject-level nuisance variables; empirical correlations need not be
exactly zero. Preserving marginals still changes the joint image distribution,
so a score drop can reflect sensitivity to unfamiliar factor combinations.
Compare this diagnostic with the matched analysis when assessing disentanglement.
It is not a statistical independence test with a p-value.

The mode supports `pseudo_mri`. It collects source latent draws with one rendering
pass, then renders the shuffled dataset and caches those images for the checkpoint
and floor runs. It does not cache the original images. JSON records the shuffle
seed, source causal/hierarchical settings, content/permutation hashes and mean
absolute pairwise correlations before/after under `evaluation_distribution`.
Those correlations are a sanity check, not a nonlinear independence test. Repeat
with additional shuffle seeds to assess sensitivity. The source SCM is not used
to residualize the shuffled labels, which no longer follow its joint law.

`--causal iid` retains the older causal-off behavior; it does not preserve the
matched factor marginals and may still use a configured hierarchical sampler.

## Edge directions

`run_causal_recovery --orientation` additionally scores each recovered graph's edge
DIRECTIONS. The comparison target is the true DAG's **CPDAG** (via causal-learn's
`dag2cpdag`), not the DAG: PC identifies a Markov equivalence class, and the default
`chain` SCM's class is entirely undirected, so scoring arrows against the DAG would
charge the estimate for edges no observational method can orient. The headline metrics
are unchanged and remain skeleton-only; this adds one console line and, per alpha in the
JSON, `correct_directed`, `reversed`, `undirected_in_estimate` (PC declined to orient an
edge the truth's class does orient — a weaker failure than a reversal),
`directed_in_estimate`, `bidirected_in_estimate`, `both_undirected`, `cpdag_shd` and
`cpdag_exact_match`, plus the estimated and true CPDAG matrices. `cpdag_shd` counts node
pairs whose edge type differs at all, so it is comparable to `skeleton_shd` but strictly
harder. Without the flag the outputs are byte-identical to before.

## Conditional-independence test

`--indep-test {fisherz,kci}` on both `run_causal_recovery` and
`eval/dinov3_identifiability`. Default `fisherz`, so existing outputs are unchanged; the
choice is recorded per run as `indep_test` in the JSON, the CSV and the protocol block.

Fisher-Z is a partial-correlation test, so it sees only the **linear** part of a
dependence. This generator's mechanisms are `leaky_relu` of a weighted parent sum
(`--synthetic-causal-nonlinearity tanh` is smoother still), so Fisher-Z is misspecified
for it — and PC only ever returns a CPDAG, which together is why the orientation ceiling
reads SHD 17 with 8 reversals *on the ground-truth factors*. A purely nonlinear edge is
invisible to it: on `y = x²` with symmetric `x`, the linear correlation is zero and
Fisher-Z reports independence, while KCI recovers the edge (`tests/`
`test_dinov3_identifiability.py::IndepTestTests`).

KCI is nonparametric, and the cost is not a constant factor. Measured here: a 3-factor
chain takes 0.13 / 0.37 / 1.52 s at 200 / 400 / 800 rows, but **9 factors at 500 rows did
not finish one alpha in 30 minutes**, where Fisher-Z is instant. The blow-up is in the
number and size of conditioning sets, which grows with the factor count, and PC re-runs
the whole search once per alpha — so pass a single `--alphas` value with it.

`--max-cond-set N` caps PC's conditioning-set size (its `max_k`) and is what makes KCI
usable at that width: 9 factors at 300 rows went from not finishing to **9.7 s at
`--max-cond-set 2`**, recovering the same 14 edges as `1`. It is an approximation — pairs
that only separate on a larger conditioning set keep their edge, so the skeleton can gain
edges but never lose them, which shows up as lower precision rather than lower recall.
`evaluate_arrays` warns before spending the time when KCI is uncapped at five or more
factors. causal-learn's KCI defaults to the gamma approximation (`approx=True`), which is
deterministic, so no seeding is needed.

A workable starting point at 9 factors:

```bash
python -m eval.run_causal_recovery --run-dirs RUN --num-samples 500 --pooling gap \
  --indep-test kci --max-cond-set 2 --alphas 0.05 --orientation
```

Note that KCI fixes the *test*, not the estimator: PC still returns a Markov equivalence
class. For a generator that is a nonlinear additive-noise model the DAG itself is
identifiable, which needs an ANM-family method rather than a constraint-based one.

## The decoded-factor readout

PC does not see the representation. It sees `n_content` supervised reconstructions of the
true factors: the features are standardised, PCA-reduced, and a `RidgeCV` per factor
decodes it from that basis. Two flags control that step, on both
`run_causal_recovery` and `eval/dinov3_identifiability`; both default to the original
behaviour, so existing outputs are unchanged.

`--readout-dim N` pins the PCA width. The default rule is
`min(64, max(n_content, N_samples/5))`, capped at the block's own width — so a 48-channel
encoder block gets a 48-dim readout while an 18432-dim embedding gets 64, and part of any
difference in the recovered graph is that gap rather than the representation. Pin it to
the narrower of two models to compare them at equal readout capacity; a block narrower
than the request keeps its own width. The effective value is reported as
`graph_readout_dim`, alongside `readout_mode` and `graph_samples`.

`--holdout-readout` fits the readout on the 70/30 train split and runs PC on the held-out
rows only, instead of decoding the same rows it was fit on with the true labels. This is
what removes the panel's in-sample optimism — the `PC R²` column then reports exactly the
decoding quality of the columns PC was handed, because both use the same split. It costs
70% of the rows, so pair it with `--num-samples 2000` or more; PC on fewer than
`20 * n_content` rows logs a warning and fewer than 20 is an error. A single split rather
than cross-fitting is deliberate: a cross-fitted row's decoding depends on every other
row's label, and Fisher-Z assumes the rows are independent draws.

Neither flag touches the raw/partial R² columns, which keep their own full-width probe.

## DINOv3 embeddings as a reference representation

Two scripts run the same two questions on a pretrained 2-D vision encoder instead of a
trained VQ-VAE, so "what does a general-purpose foundation model already recover here"
has an answer on this generator's own terms. They need `transformers` and `causal-learn`
in the environment, and DINOv3 weights are gated on the Hub (accept the licence and
`hf auth login`, or pass a local snapshot to `--model-id`).

```bash
python -m pip install transformers causal-learn

# 1. embed. --run-dir takes the generator settings from a training run, so the
#    embeddings are scored on exactly that run's distribution.
python -m eval.dinov3_embed_synthetic --out results/dinov3/emb.npz --num-samples 500
python -m eval.dinov3_embed_synthetic --out results/dinov3/emb_floor.npz --random-init \
  --num-samples 500                      # untrained twin, same architecture and seed

# 2. score
python -m eval.dinov3_identifiability --embeddings results/dinov3/emb.npz \
  --floor results/dinov3/emb_floor.npz --out results/dinov3/report.json
```

The volume is reduced to `--slices` evenly spaced planes per axis in `--axes`, embedded
independently and concatenated (`--slice-agg mean` averages them instead and throws away
which plane a feature came from, which is most of what locates `lesion_x/y/z`). Two
choices are not cosmetic and are recorded in the output's `meta`:

- `--window per_slice` (the recipe in `eval/dino.ipynb`) maps every plane onto the same
  range, which is exactly the affine map style applies. Style recovery is then bounded by
  the windowing rather than by the encoder — the same trap `--synthetic-normalize
  per_sample` sets. The default `dataset` estimates one window from a pilot of volumes.
- `--token-pool` defaults to `cls_mean`. Mean pooling over patch tokens is
  permutation-invariant, so in-plane position is not in it; `--token-pool grid` keeps a
  `--grid-size` average of the patch map. The patch-token prefix (CLS plus DINOv3's four
  register tokens) is inferred from the sequence length rather than trusted from the
  config, so `mean` never silently averages register tokens in with the patches.

The report has four sections. Tables 1 and 2 are per-factor cross-validated probe R² with
a permutation null and block-MCC, computed by `eval.identifiability_metrics` and batched
as in `run_dci_compare._score_block`; `gap = real − null` is the reportable column,
`Δfloor` subtracts the untrained twin and `Δvox` compares against downsampled voxels.
Table 3 is `run_causal_recovery.evaluate_arrays` unchanged, with two extra rows: `truth`
runs the identical panel on the ground-truth factors (the ceiling — PC at this sample
size cannot beat that row) and `floor` runs it on the untrained twin. Every caveat above
about alpha selection and the in-sample graph readout applies to it unchanged. Table 3's
raw/partial R² use that panel's full-width single-split Ridge probe, which is biased low
when the embeddings are wider than the sample count; table 1's null-corrected gap is the
factor-recovery number to quote.

`--self-test` on either script runs its logic on planted arrays with no model, no GPU and
(for the scoring script) no torch.
