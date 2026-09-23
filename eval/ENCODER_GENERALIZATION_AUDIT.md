# Encoder generalization and BatchNorm audit

Run all three diagnostics on an encoder-only checkpoint, without retraining:

```sh
python -m eval.encoder_generalization_audit \
  --run-dir results/dummy_infonce_wm_interior_causal_no_sep \
  --checkpoint model.pt \
  --num-samples 400 --probe-samples 400 \
  --batch-size 4 \
  --retrieval-batch-size 64 --retrieval-draws 8 \
  --bn-samples 512 --bn-batch-size 64
```

Run this in the same environment as training. Both `resnet18` and `conv` are
supported; BatchNorm recalibration is explicitly not applicable to a backbone
without BatchNorm. The audit currently requires an InfoNCE run. No labels enter
encoder updates; labels are used only to fit and evaluate diagnostic readouts.

`model.pt` is the latest saved checkpoint, not necessarily step 2000. Select a
saved snapshot of the step you want to diagnose. The audit loads the file once
and records its SHA-256. It never writes any model checkpoint. Avoid launching
while training is in the middle of writing that file.

## 1. Training versus held-out cross-view retrieval

The audit samples 400 distinct indices from the original training set and uses
400 subjects from the generator's separate **test** split. All settings,
including lesion placement, causal graph, edge probability and graph seed,
match the training entry point. The graph is shared across splits; subjects,
styles and noise use different split seeds. `--num-samples` must not exceed the
original number of training subjects.

The model stays in evaluation mode. The audit reports both directions of
T1/FLAIR retrieval with:

- All 400 opposite-view candidates, including the matching subject.
- Eight independently sampled groups of 64 subjects, matching the training
  batch/negative-pool size. Draws and pool sizes match between arms and cohorts.

Top-1/top-5 accuracy, reciprocal rank, mean rank, positive/negative cosine and
InfoNCE are saved. Ties get expected scores under random tie-breaking, so a
collapsed embedding scores at chance. The loss uses the saved temperature,
negative policy and optional projector. It matches the training code's
reductions: the cross-only loss is a **sum** of directional means; the
same-plus-cross-negative loss is their **mean**. Compare train/test within the
same candidate-pool size. Accuracy's chance level is 1/candidate count.

`--batch-size` controls only encoding memory. It does **not** change the negative
pool. Evaluation-mode losses need not reproduce training-mode log values.
Large training-versus-test gaps support poor generalization; these metrics alone
do not identify which shortcut was used.

## 2. Ridge and nonlinear recovery along the actual global path

For each view, capture one real forward through:

1. Backbone features after global average pooling (512 dimensions for ResNet).
2. The nonlinear readout's hidden activation (100 dimensions by default).
3. The final content vector (9 dimensions in this run).
4. The L2-normalized content vector and its scalar norm, as additional controls.

The hidden readout exists even when `contrastive_proj_dim=0`; that flag only
disables the extra loss-only projector. Retrieval uses the actual loss space.
The older convolutional encoder has no nonlinear hidden-readout stage.

The 400 **validation** subjects are split into 300 probe-fit and 100 probe-tuning
subjects. Feature/target scaling is fitted on probe-fit subjects only. Ridge
regularization and RBF kernel-ridge regularization/bandwidth are selected per
factor on probe-tuning subjects. The separate 400 **test** subjects are used
only for final R². Negative R² is retained; undefined constant-target scores are
saved as JSON null. One shuffled-label control is fitted/tuned separately with
the same protocol; it is a diagnostic reference, not a significance test.

These scores use a different split protocol from training-time cross-validation,
so compare stages and arms *within this report*. Validation subjects may have
participated in earlier checkpoint selection. The test cohort is independent
of that selection unless you have separately used the same test split before.
Subject indices and model-training versus probe-training roles are recorded.

Strong backbone scores and weak final scores locate a loss of accessibility in
the readout. RBF recovery above ridge suggests nonlinear accessibility. Neither
probe's failure proves absence of information. With causal factors, prediction
can exploit correlated anatomy rather than the named factor's direct image
effect; this audit does not replace a controlled renderer intervention.

## 3. Train-only BatchNorm recalibration

Make a disposable copy of the loaded model. Reset its BatchNorm running means,
variances and batch counters; use cumulative averaging over one pass through a
random subset of **original training images only**, with no gradients. Only
BatchNorm modules enter training mode. No test or validation images enter this
step. Shared/separate view routing matches the actual model.

The default batch size equals the training batch size (64 subjects per view,
128 images through a shared backbone). Full equal-sized batches are used; a
partial final batch is dropped and the exact used subject count is reported.
If memory requires a smaller `--bn-batch-size`, record that change: batch size
can affect the calibration. This is a standard cumulative batch-statistics
diagnostic, not exact population normalization of every layer.

Return the copy to evaluation mode and repeat retrieval and all probes on the
**same** subjects. Readouts are refitted separately for each normalization arm,
using identical probe-fit/tuning/test splits and search grids. This measures
accessibility after recalibration, not transfer of a fixed old readout.

Checks assert that model weights and non-BatchNorm buffers do not change in the
copy, that the original model's entire registered state is unchanged, and that
both arms see identical images and targets. Improvement supports a contribution
from running statistics. No improvement does not exclude other training-time
BatchNorm effects or architectural differences.

## Outputs and next decisions

A new `encoder_audit_<timestamp>/` directory contains:

- `report.json`: complete protocol, cohort IDs, state hashes, calibration changes
  and all measurements.
- `retrieval.csv`: every cohort, arm, direction and negative-pool draw.
- `probes.csv`: every factor, stage, view, readout and shuffled control, plus
  changes from the original arm and from the backbone stage.
- `ranks.csv`: evaluation-mode effective rank; high rank is not evidence for
  semantic factor recovery.
- `original_features.npz`, `bn_recalibrated_features.npz`: all pooled feature
  banks and targets, saved before fitting probes.
- Corresponding `*_predictions.npz` files: test predictions and true targets.
- `checkpoint_source.json`: source checkpoint and loaded snapshot hash.

Features and completed results are saved incrementally. The normal invocation
performs new extraction; retained feature banks also permit separate downstream
analysis without rendering again. Intermediate reports say `status: running`;
only a finished audit says `complete`.

| Finding | Next step |
| --- | --- |
| Good training retrieval, poor test retrieval | Test more/fresh training pairs or a smaller backbone. |
| Recalibration improves held-out retrieval and recovery | Investigate BatchNorm running statistics and train/eval mismatch. |
| Backbone recovery is good but final content recovery is poor | Isolate the nonlinear readout/compression in a controlled ablation. |
| Final RBF recovery is good while ridge is poor | Information remains nonlinearly accessible; linear scores understate it. |
| Test retrieval is good but both probes are weak for some factors | Investigate feature selection with matched-data ablations and factor interventions. |

For a cheaper wiring check, use `--num-samples 128 --probe-samples 200
--bn-samples 128` and leave the retrieval/calibration batch size at 64. Use the
larger cohort before drawing conclusions about weak recovery.
