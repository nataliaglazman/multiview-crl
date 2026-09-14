# Ventricle checkpoint diagnostic

Run this before another training ablation. It measures ventricular recoverability
and the local effect of GAP redundancy/MSE descent, using a saved model. It never
saves a modified model or optimizer. Anatomical labels are used for diagnostic
readouts and evaluation only; neither update direction uses them.

```bash
python -m eval.ventricle_checkpoint_test \
  --run-dir results/synthetic/synthetic-clean-content-causal-ident-vent-12-4-3
```

Defaults: 256 probe-fit subjects, 128 disjoint test subjects, four gradient batches
at the training batch size, 16 further routing subjects, and two temporary step
sizes per active direction. There are five main feature-extraction passes (baseline
plus four trials), two held-out replay passes, and smaller gradient/routing passes.
This is more expensive than a single probe but does not run a training loop.
Decoder forward passes are necessary to observe the actual quantized tensors.

For an initial smoke run:

```bash
python -m eval.ventricle_checkpoint_test \
  --run-dir results/synthetic/synthetic-clean-content-causal-ident-vent-12-4-3 \
  --fit-samples 128 --test-samples 64 --grad-batches 2 \
  --routing-samples 8 --relative-steps 1e-5
```

A smoke run cannot establish that ventricular information is absent. Increase the
fit/test subjects if probe performance or confidence intervals are inconclusive.
`--routing-samples 0` skips routing. `--cache-images` trades potentially several GB
of host RAM for less repeated rendering. `--encode-batch` controls inference memory;
`--grad-batch-size` defaults to the training size because changing it changes the
loss statistics. `--checkpoint` accepts a run-relative filename or absolute path.

## What is measured

* **Decoder content:** hooks capture the actual quantized output of the content
  codebook. All embedding coordinates are retained; the embedding dimension need
  not equal the number of encoder channels selected as content.
* **Decoder style:** the actual decoder-bound style tensor, after any configured
  bottleneck/quantization. Eval mode disables training-only dropout.
* **Readouts:** separate ridge and RBF kernel-ridge ventricular-size probes for
  each view/block, on GAP and a spatial grid (default 4 cubed). This grid belongs
  only to the readout, independently of the training alignment grid (8 cubed in
  the motivating run). No PCA is applied. Global style tensors are never enlarged
  into duplicate spatial features.
* **Probe fitting:** all scaling, target centering and hyperparameter selection use
  fit subjects only, with three inner CV folds. No held-out labels enter fitting.
  Trial probes either stay fixed or are refitted on the same fit subjects with the
  baseline-selected hyperparameters. Fixed-only degradation can indicate a change
  of coordinates rather than less recoverable information.
* **Routing:** cached ventricular A/B interventions, content/style swaps and endpoint
  replay checks from `ventricle_routing.py`. Each trial is compared on the same
  subjects that have measurable input changes and resolved endpoints in BOTH runs.

Gradient samples use the run's matched SCM on the validation split. Probe fitting,
testing and routing use nonoverlapping indices in the test split. The default probe
distribution is IID (both causal and hierarchical content coupling disabled); it
reduces recovery via correlated anatomical factors but may be outside training
support. `--probe-causal match` provides a complementary matched-distribution check,
whose R² can include recovery via other factors. Generator resolution and renderer
settings come from the run; labels are z_content[1], not a segmentation target.

## Local interventions

Only encoder parameters (including encoder-side content normalization/projections)
are perturbed. Codebooks, decoder parameters, masks and registered buffers are
frozen. Registered tensors and requires-grad/training flags are restored even if
the diagnostic raises an exception. A no-step replay and final restored replay
check features and discrete content assignments.

The two directions isolate **GAP off-diagonal redundancy** and **GAP feature MSE**.
Features reproduce the training forward's content selection and per-batch
foreground patch filtering, then average those retained patches. The script
honors normalized BT terms, GAP-specific coefficient inheritance, outer/layer
weights and the optional detached variance denominator in MSE. It neither includes
the patch term nor the diagonal/variance/reconstruction terms in these directions.

The historical correlation EMA is not saved in the model checkpoint. The diagnostic
therefore estimates a stationary reference from the frozen gradient batches. With
decay m, it evaluates `m * reference.detach() + (1-m) * current_correlation`.
The reference stays fixed for every gradient and trial, reproducing the settled
EMA's current-batch derivative. This is explicitly a **stationary surrogate**, not
an exact replay of the old optimizer/EMA trajectory. Without EMA, the isolated GAP
loss is evaluated directly. Numerical tests compare values and gradients against
the shipped BT implementation, including EMA and normalization.

Each step is:

```
theta_trial = theta_original - relative_step * ||theta_original|| * g / ||g||
```

Default relative steps are 1e-5 and 1e-4. These deliberately compare directions at
equal parameter-displacement norms. **They do not compare actual training influence**:
the reports also include weighted gradient norms and equivalent raw-gradient step
sizes. Scaling a positive coefficient changes the reported gradient norm but not
its normalized direction. AdamW momentum, adaptive scaling, weight decay, global
clipping, decoder adaptation and future optimization are outside this test.

The source loss is remeasured after each step. A trial that does not reduce its
source loss is flagged as inconclusive. Gradient norms and cosines to the mean are
reported per batch; four batches are a small sample, not a historical estimate.

## Reading the output

* RBF much better than ridge: ventricular size is more accessible to this nonlinear
  readout. Neither probe provides an upper bound on encoded information.
* Positive delta R²: better held-out recovery. Negative: worse. The paired bootstrap
  resamples test subjects and conditions on probe-fit data and the chosen gradient;
  it does not cover all training/probe selection uncertainty or multiple testing.
* Fixed probes worsen but refitted probes recover: evidence of a coordinate change.
* Refitted probes and content routing worsen consistently across small GAP descent
  steps: local evidence that this direction opposes useful ventricular content.
  It does not prove that the loss caused the original training outcome.
* Zero code-change fraction and unchanged post-VQ metrics: potentially no quantizer
  boundary crossing, not proof of zero encoder influence.
* Read joint reconstruction fidelity before interpreting content/style routing gains.
  Small routing sample counts are a smoke check; increase them for a stable estimate.
* Compare changes with the no-step replay and across step sizes. Very small numerical
  changes or inconsistent signs do not justify another training recommendation.

Outputs go to a new timestamped directory inside the run (or `--out`, which must
not already exist):

* `summary.json`: configuration, baseline R², gradient statistics, null replay,
  trial source losses, code changes, complete routing rows and paired summaries.
* `probe_deltas.csv`: fixed/refitted probe changes and paired confidence intervals.
* `baseline_features.npz`: decoder feature blocks and targets for reuse; first
  `fit_samples` rows are fit subjects, the remainder are held out.
* `predictions.npz`: held-out targets and baseline/trial predictions.

Partial results are saved after each completed trial. `restoration_verified: true`
appears only after the final restored replay passes. The checkpoint is read-only
throughout, including on failures.

The initial implementation requires a fixed split, one VQ level, injected style,
patch Barlow Twins, and no loss-facing projection/MoCo. It fails on incompatible
settings or missing/unexpected checkpoint keys instead of approximating silently.

## Tests

```bash
python -m unittest discover -s tests -p test_ventricle_checkpoint_test.py -v
```

Tests cover BT value/gradient parity, coefficient inheritance, a planted nonlinear
signal, coordinate-change versus refit behavior, paired statistics, exact quantized
features with separate codebooks, label-free directions, descent checks, cleanup on
exceptions, and a complete run with actual VQVAE code and checkpoint restoration.
