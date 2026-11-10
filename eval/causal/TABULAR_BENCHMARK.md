# A tabular causal benchmark for frozen MRI vectors

This benchmark adds synthetic tabular variables to an existing bank of synthetic MRI subjects. It uses the bank's true image factors and known causal graph, and keeps every bundle on the same subjects in the same order. All varying image factors enter through the same rule. Train the image encoder before running the benchmark and use a bundle exported on held-out subjects.

## Simulated tasks

Three observed tabular covariates precede treatment. Two are related and affect both treatment and outcome. The third predicts outcome. The standardized image factors receive seeded weights of equal magnitude to form a hidden image score. This score and the two tabular confounders drive both treatment and outcome. A confounder is a common cause of treatment and outcome. The probability of treatment stays between 0.1 and 0.9 by default.

**Effect estimation.** Treatment is binary and the outcome is continuous. The simulator records both potential outcomes, the outcomes under treatment and under control, for every subject. Their mean difference is the known average treatment effect (ATE), 1 by default. `--heterogeneity` lets the effect vary by the image score while preserving the mean. `--effect 0` makes a null-effect task when heterogeneity is zero. `--confounding 0` makes a randomized-treatment control.

Every adjusted arm uses the same cross-fitted augmented inverse probability weighting estimator (AIPW). Each fold learns treatment probabilities and outcome predictions from the other folds. The image arm adds the frozen vector directly to the observed tabular variables. True MRI factors and counterfactual outcomes are reserved for simulation, the oracle comparison, and scoring. The default nuisance models are logistic regression and ridge regression fitted separately by treatment arm. `--learner rbf` adds smooth nonlinear features fitted within each fold. [Cross-fitting and ATE estimation](https://doi.org/10.1111/ectj.12097)

**Causal discovery.** A companion task draws continuous exposure and outcome columns with a constant effect. Each task seed picks one image factor, which drives exposure together with the first tabular covariate. A different image factor drives outcome together with the second and third tabular covariates. Each edge gets a seeded weight between 0.5 and 1 in magnitude. The first tabular covariate still confounds exposure and outcome through its edge to the second. No variable is a direct parent of both exposure and outcome. When one is, its edge into exposure nearly cancels once PC conditions on outcome. The effect task's equations made that cancellation exact, and the true factors recovered about a quarter of the MRI-to-tabular edges. A single ridge readout decodes all image factors jointly, with a shared penalty fitted on one half of the subjects. The PC algorithm uses those decoded factors and the continuous tabular data on the other half. The score measures recovery of the graph's undirected edges, including the edges between MRI factors and tabular variables. The true-factor arm runs PC on the same held-out rows. Fisher-Z tests partial correlation and is approximate for this bank's non-Gaussian image factors. This is a supervised graph recovery task. [Fisher-Z and PC](https://www.jmlr.org/papers/v8/kalisch07a.html)

## Run

With stage 19's completed half-view checkpoint and canonical correlation analysis (CCA) bundles on the PC, run from the repo root.

```bash
bash out/burden_feasibility_20261102/run_pc.sh 22
```

The stage compares the 24-unit CCA readout of backbone global average pooling (GAP), the original code, and the full GAP vector. Each has a matched untrained version. T1 and FLAIR are scored separately. `TABULAR_RUN` selects a different finished stage 19 run with current CCA bundles. Results go to `out/burden_feasibility_20261102/causal/tabular/$TABULAR_RUN/`, with the selected run's name in the directory. Use `DRY=1` to print the commands.

For any exported bundle in the shared `.npz` format, run the scorer directly.

```bash
python -m eval.causal.tabular_benchmark \
  --bundles dense=/path/to/dense.npz \
  --floors dense=/path/to/dense.floor.npz \
  --view each --tables-dir results/tabular/tables --out results/tabular/report.json
```

Discovery uses `causal-learn` and a known `causal_adj` in the bundle. `--effects-only` runs without either. Use `--view 1` or `--view 2` for a single modality, or `--view both` to concatenate the two vectors. The report records feature widths, so comparisons between differently sized vectors include the effect of width.

`observed_<seed>.csv` holds the three tabular columns, treatment, and factual outcome. Join `row` to the corresponding zero-based embedding row. `truth_<seed>.npz` separately stores the image factors, potential outcomes, true assignment probabilities, both graphs, and the two continuous discovery columns. Graph matrices use parent rows and child columns. Their node order is the image factors, the three tabular covariates, exposure, then outcome.

## Read the report

- For effect estimation, compare mean absolute ATE error with tabular-only adjustment and with the same representation at initialization. A negative paired error difference favors the trained vector. The true-factor arm measures finite-sample estimator error when the confounders are available. The unadjusted mean difference shows the scale of confounding.
- Check fitted assignment probabilities and weighted effective sample sizes. The nominal 95% intervals depend on sufficient adjustment and accurate nuisance estimates. The ten task seeds reuse the same MRI cohort, so error and coverage summaries describe those simulated tasks.
- For discovery, read overall edge F1 and MRI-to-tabular edge F1 with held-out factor prediction scores. F1 combines edge precision and recall. MRI-to-tabular F1 is the column that separates representations. In a Gaussian simulation with the bank's factor correlations, true factors score about 0.9 and decoded factors with R² 0.6 score about 0.75. Overall F1 moves little. The graph changes with the task seed, and each entry in `runs` stores its own `discovery_truth`. The graph and effect panels address separate questions. Low ATE error indicates useful adjustment for the simulated task. A clinical effect estimate would require its own treatment definition, timing, and adjustment assumptions.
