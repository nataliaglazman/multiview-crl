# Optional causal evaluation diagnostics

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
