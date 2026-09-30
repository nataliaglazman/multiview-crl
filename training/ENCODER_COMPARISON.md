# Matched encoder comparison

This compares the existing `conv` and `resnet18` architecture configurations under
one training recipe. It holds data, batch order, objective, optimization, output
dimension and evaluation constant. It does **not** equalize parameter counts,
normalization, native spatial resolution, receptive fields, or readout family.
Results therefore concern the complete architectures, not depth alone.

## Run

Activate the environment used for training and run these from the repository root:

```bash
# Prints the six training commands without creating files or starting training.
python scripts/compare_encoders.py plan

# Sequential: two architectures x three initialization/shuffle seeds.
python scripts/compare_encoders.py train

# Frozen global test probes, then GAP/patch/lesion validation diagnostics.
python scripts/compare_encoders.py evaluate

# Verifies pairing before producing the final per-factor comparison.
python scripts/compare_encoders.py summarize
```

Configuration: `experiments/encoder_comparison.json`. Defaults are resolution 64,
2,000 training subjects, 400 validation subjects, batch size 32 **per view**,
10,000 updates, nine content units, 12 total units, independent content factors,
clean content, WM-contained sphere lesions, fixed-reference input normalization,
InfoNCE at temperature 0.1, AdamW at 1e-4, and separate view backbones. AdamW's
resolved defaults are recorded and checked. These are an explicit starting recipe,
not a reconstruction of either earlier run or a claim of optimal hyperparameters.

The fixed data seed is 42. Model seeds are 42, 142 and 242; loader seeds add 10,000.
Both architectures within each pair see identical shuffled subject batches.
Across pairs the dataset stays fixed while initialization and order vary. Thus
the reported variation concerns initialization/order, not generalization across
independently sampled datasets. To vary datasets, use separate configurations
and output roots with different `data_seed` values.

Edit a **copy** of the config for another recipe, then use the same options for
every phase:

```bash
python scripts/compare_encoders.py plan \
  --config experiments/my_encoder_comparison.json \
  --output-dir results/my_encoder_comparison
```

For one pair first, append `--only-seed 42` to each command. To launch independent
jobs on a cluster, choose a seed and architecture explicitly:

```bash
python scripts/compare_encoders.py train --only-seed 42 --only-architecture conv
python scripts/compare_encoders.py train --only-seed 42 --only-architecture resnet18
```

Use separate GPU allocations for concurrent jobs. The launcher itself neither
submits cluster jobs nor assigns devices; it inherits `CUDA_VISIBLE_DEVICES`.
The same selectors work for evaluation. Summarization requires both architectures
for every selected seed. The same interpreter that runs the launcher runs its
children; `plan` alone needs only Python's standard library.

## Reproducibility and safeguards

- The training entry point accepts `--data-seed`, `--model-seed` and
  `--loader-seed`. Dataset creation can reset the global RNG, so the model is
  explicitly seeded **after** datasets are built. The loader has a private
  generator independent of initialization and evaluation. Defaults are the
  original `--seed` for data/model and `--seed + 10000` for the loader.
- New training runs save `model_init.pt`, the actual initial weights, alongside
  the usual `model.pt`. Evaluation restores the saved data/model seeds. Old
  settings without these fields continue using their original `seed` fallback;
  old checkpoints are unchanged. Retraining now uses the corrected RNG handling,
  so it does not reproduce the old implicit RNG sequence.
- `training_progress.json` records completion, exact step, subject exposures,
  streaming SHA-256 hashes of batch subject indices and actual input images, parameter count, architectural
  differences, optimizer defaults, PyTorch version and elapsed time including
  evaluations. Elapsed time is descriptive, not a compute-matched benchmark.
- Deterministic PyTorch operations are enabled. Unsupported operations fail
  explicitly. Set `deterministic: false` for **both** arms in a new configuration
  if the environment cannot support them; do not silently change one arm.
- CPU rendering and numerical-library thread counts are pinned to one by default
  in both training and evaluation. Saved deterministic settings are restored by
  evaluation as well. Exact input hashes deliberately reject even floating-point
  rendering differences between runs; do not bypass this check to report a pair.
- `comparison_manifest.json` freezes the recipe and Python source hashes for
  data/model/training/evaluation/utilities. Changed code or settings require a new
  output root. Record the same environment and hardware for both architectures;
  PyTorch and optimizer defaults are checked automatically.
- Run directories cannot be overwritten. `--skip-completed` skips only completed
  runs with matching settings and checkpoint hashes. It is not optimizer resume.
  Failed partial training needs a new output root/run experiment; keep its logs.
- For an interrupted evaluation, choose a new `--evaluation-tag retry1` for
  `evaluate` and `summarize`. Completed evaluation runs can use `--skip-completed`.
- Keep training batch size the same for both architectures. If ResNet cannot fit,
  lower it for both in a fresh experiment. Encoding batch size during evaluation
  is separate and does not change the retrieval candidate pool.

## Evaluation and outputs

The primary endpoint is the final checkpoint at the prespecified common step.
`best_metric: none` prevents selecting one model at an especially favorable step.
Training still logs validation curves every 2,000 steps; preserve the same schedule
for both. More training or hyperparameter tuning should use an equal, prespecified
budget for each architecture, with test results kept out of selection.

For each run, `evaluation/global_path/` uses the existing generalization audit:
ridge and RBF hyperparameters are fit/tuned on a 75/25 split of the 400 validation
subjects, then evaluated on 400 separate test subjects. Both views are evaluated.
The original checkpoint is primary; BatchNorm recalibration is disabled in this
suite and remains available separately through `eval.encoder_generalization_audit`.
The renderer's existing split-specific fixed-reference normalization is retained
identically in both arms; this suite does not change that preprocessing protocol.

`validation_gap_lesions.json` and `validation_patch.json` are **validation**
diagnostics. They use each architecture's matched random initialization as a
baseline and the same 2x2x2 grid (72 content features). The lesion report also
includes backbone/style probes, physical centroids and shuffled targets. Backbone
dimensions still differ: 64 vs 512 channels by default. Treat their ridge results
as diagnostics, not a capacity-matched probe contest. ResNet patch outputs apply
its nonlinear head to individual spatial bins; they are not the trained global
vector. The common grid is checked against both native maps before launch.

Summarization refuses unequal training batch/input hashes, changed checkpoints,
different input-image hashes, mismatched subject IDs/probe splits, or incomplete
runs. Under `summary_evaluation/` it writes:

- `test_scores.csv`: individual seed/view/factor ridge and RBF test R².
- `paired_comparison.csv`: per-factor means and ResNet-minus-conv paired
  differences, including sample SD across seeds. One seed has no SD estimate.
- `training_costs.json`: training metadata and actual architecture differences.
- `verification.json`: the pairing checks and experiment scope.

Initial validation baselines are in each run's `dci_step0.json`; diagnostic
baseline-adjusted scores are in the validation reports. Do not subtract those
validation baselines from the separately fitted test scores. The suite reports
raw held-out test recovery as the primary endpoint. Always examine lesions and
sulcal widening individually, together with modality leakage, rather than using
only the average across factors.

## Checks

```bash
python -m unittest tests.test_encoder_comparison
python -m unittest tests.test_encoder_pairing tests.test_conv_causal_config \
  tests.test_resnet_encoder tests.test_encoder_generalization_audit \
  tests.test_checkpoint_lesion_analysis
```

The second command requires the normal training dependencies. It includes real
CPU optimization steps through both backbones, seed/checkpoint round trips and
existing diagnostic regression tests. It is not a full training experiment.
