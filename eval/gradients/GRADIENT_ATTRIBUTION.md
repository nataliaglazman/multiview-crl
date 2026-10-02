# Gradient attribution against the real training objective

`eval.gradients.gradient_attribution` asks which loss terms move the **encoder** at a checkpoint,
how hard, and whether they pull against each other. Earlier versions rebuilt the loss
by hand (`_losses_at`) and drifted from training: no GAP arm, no correlation EMA, no
style or HSIC terms, and the wrong `bt_lambda` default. This version does not rebuild
the loss. It runs `train_step` itself, with no optimizer, and reads every weighted term
of `total_loss` from that one forward through `train_step(..., loss_observer=...)`.

```bash
python -m eval.gradients.gradient_attribution --run-dir results/synthetic/RUN                          # balance (default)
python -m eval.gradients.gradient_attribution --run-dir results/synthetic/RUN --target decode          # + i.i.d. decoding steps
python -m eval.gradients.gradient_attribution --run-dir results/synthetic/RUN --target mcc             # + block-MCC steps too
python -m eval.gradients.gradient_attribution --run-dir results/synthetic/RUN --target reconstruction  # pixel-MAE experiment
```

`--target reconstruction` is the separate similarity-vs-pixel-MAE experiment in
`eval/gradients/reconstruction_attribution.py`, unchanged. Everything below is about the other targets.

## Why the numbers are the trained objective

- **Barlow Twins is built in one place.** `training/bt_objective.py` builds the plain,
  patch and GAP arms for training *and* for this audit. It maps every argument the
  pre-refactor closures in `main_multimodal.py` used, and
  `tests/test_loss_gradient_audit.py` checks that it matches them bit for bit: values,
  gradients and diagnostics, over several steps of EMA state.
- **Every batch passes two parity checks before it is used.** The observed terms must sum
  to `train_step`'s total (relative error 2e-5), and their encoder gradients must sum to
  the total's encoder gradient (within 1e-3 of the terms' summed gradient norms). A term
  added to `total_loss` without passing through `observe()` fails the run. It cannot
  silently drop out of the tables. The measured errors go to `parity.csv`. On
  `ident-vent-hsic` they were ~1e-7 for values and ~1e-5 for gradients, which is float32
  re-association through the encoder's conv reductions.
- **Training is untouched.** With `loss_observer=None` the component capture is off, and
  an SGD step is bit-identical with and without the observer (tested).

## Components and groups

Each row is one weighted term, carrying every coefficient training applies to it:

| component | what it is |
|---|---|
| `content/L0/{patch,gap}/{on_diag,off_diag,sim,variance}` | BT terms × arm weight × `scale_contrastive_loss` × level weight. Zero-coefficient terms are omitted. |
| `reconstruction/{pixel,perceptual}` | `BaselineLoss` terms × `scale_recon_loss` |
| `reconstruction/commitment_L0` | the VQ commitment cost `BaselineLoss` adds itself, × `scale_recon_loss` (absent under `--single-count-commitment`) |
| `commitment/total` | the same cost added again as Loss/VQ, × `vq_commitment_weight` |
| `hsic/total`, `style/L0/...`, `cross_reconstruction/total` | when active |

The **group** table sums components per batch into `content`, `style`, `reconstruction`,
`commitment` (both counts), `hsic` and `cross_reconstruction`. It is the old tool's
contrastive / recon / vq split, with the double-counted commitment put in one place.

Columns:

- `loss`: the weighted term, averaged over batches.
- `rms |g|`: RMS over batches of the per-batch encoder-gradient norm, i.e. the force one
  step feels.
- `|E g|`: the bias-corrected expected gradient norm, sqrt(|mean|² − tr(Cov)/B). A
  negative value means the mean is not resolved from zero at this many batches. Near zero
  does **not** prove a term is harmless.
- `share`: mean over batches of g_k·g/|g|². Across components it sums to 1. It can be
  negative or exceed 1, so it is a signed projection, not an importance percentage.
- `b->SNR1`: how many batches must be averaged before the mean outweighs the noise.

Pairwise cosines are per batch, reported with the fraction of batches that are negative.
The console prints the group matrix and the strongest component conflicts. `cosines.csv`
has every pair.

## The correlation EMA

With `--bt-corr-ema m`, training differentiates `m·C_ema + (1−m)·C_batch`, and only the
current batch carries gradient. The checkpoint does not store `C_ema`. So
`--ema-mode reference` (the default) estimates it as the mean instantaneous correlation
over **separate** reference batches at this checkpoint, then freezes it. Measuring a
batch never advances it. It is a stationary estimate, not the training history.

The reference must be about as precise as training's own EMA. The EMA averages
(1+m)/(1−m) batches, which is 199 at m=0.99. A reference built from fewer batches carries
more sampling noise, and that noise enters the off-diagonal multiplier directly and
inflates the off-diagonal gradient. The default is therefore that window. The report
prints, for each arm, the reference's off-diagonal noise next to the training EMA's, so
you can see the cost of `--ema-reference-batches N`. With 3 batches on `ident-vent-hsic`
the noise was 66× the EMA's. Reference batches are encoder-only forwards, but rendering
25k synthetic subjects is slow single-threaded (~78 ms each at 64³), hence `--workers`.

`--ema-mode instantaneous` drops the EMA, which is a different derivative. A cold EMA
would be identical to it, because the bias correction at t=1 cancels exactly, so it is
not offered.

## i.i.d. decoding of a factor (every target; `--decode-factors`)

This asks whether each term's descent direction makes a factor more or less decodable.
The default factor is `ventricle_size`; pass several names, or pass the flag with no
names to skip. The probe is `run_dci_compare`'s reading of a GAP-assigned factor:
pooled pre-norm content channels, StandardScaler, the repo's `RidgeCV` and k-fold CV.
Each view is scored separately (T1 and FLAIR) because their encoders are separate. Only
GAP-assigned factors (the morphometry ones) are accepted.

The `--decode-samples` test subjects (default 512) are rendered once with factors drawn
**i.i.d.**, whatever `--causal` says. Under the training SCM, ventricle_size correlates
~0.8 with brain_size, so a matched probe would read brain_size in disguise.

**First-order rates** (the per-term table in every report). Fit the probe once, then take
the gradient of its objective J = ‖y − S(X)w − b‖² + α‖w‖². The envelope theorem makes
this exact: the derivative of the optimally *refit* ridge equals the derivative with the
fitted (w, b) held fixed. A change a refit probe would absorb, such as rescaling a
channel, therefore counts as nothing. For each term,
`dR2/deta = g_k·∇J / SS_tot` is the change in penalised in-sample R² per unit of a raw
descent step on that term; positive helps decoding. The rates are additive across
components and sum to the total's. `hurts` is the fraction of batches whose step lowers R².

**Blind at R² ≈ 0.** A signal that is not yet there enters R² quadratically, so its first
derivative vanishes. The report warns when a factor's CV R² is below 0.05. In that case
read the finite steps instead: on `ident-vent-hsic`, ventricle_size read −0.04 (T1) and
−0.02 (FLAIR) from GAP content at N=256.

## Finite steps: `--target decode` and `--target mcc`

This takes a temporary unit step along each **group's** mean gradient,
θ' = θ − η·g/|g|, and re-measures every metric: the CV R² of each decode factor per view,
plus patch block-MCC under `mcc`. The same step is repeated along matched-random
directions that have the same per-tensor norms. A negative delta means descending that
loss lowers the metric. The **excess over random** is the attribution; the raw delta also
carries generic perturbation sensitivity. Compare the excess with block-MCC's per-seed sd
(~0.001). The linearity R² checks that the η sweep is small enough for a finite
difference to mean a derivative.

Each decode metric also gets a `fixed-alpha fit` row, which is the penalised in-sample R²
re-measured at the checkpoint's α, next to its `first-order` prediction. Agreement between
the two validates the rate table. CV R² can still move differently: `RidgeCV` re-selects α
on every step, which makes it jump discontinuously, and at small N it is noisy.
Parameters and buffers are restored after every trial, and on error. `steps.csv` and
`steps_summary.csv` hold the curves.

## Not reproduced

- **Eval mode.** The codebook EMA and resets are frozen, and style dropout is off. The run
  reports a caveat when `style_dropout_prob > 0`.
- **FP32**, not AMP.
- **Raw gradients.** No AdamW momentum or preconditioning, so `--precondition` is refused.
  There is no global clip factor either: norms are encoder-only.
- **One checkpoint.** It is not training history. The balance of forces changes over
  training, so run it at several checkpoints before generalising.
- **Batch size.** It defaults to training's, and should stay there. BT correlations,
  variance hinges and HSIC all depend on B.

Refused rather than approximated: non-BT objectives, non-synthetic data, more than one
VQ level, learned or on-the-fly masks, non-`head` projection modes, MoCo, frozen
encoders, modality-adversarial heads, an active GAN generator term, and
`skip_recon_ratio > 0`.

## Outputs

These go to `RUN/gradient_attribution_<timestamp>/`, or to `--out`, with one subdirectory
per checkpoint when several are given:

- `components.csv`, `groups.csv`, `cosines.csv`, `group_cosines.csv`, `modules.csv`
  (norms per encoder module)
- `batches.csv`, `parity.csv`
- `decoding.csv`: rate, cosine and `hurts` per (factor, view, component or group)
- `steps.csv`, `steps_summary.csv` (finite-step targets only)
- `summary.json`: settings, arguments, EMA reference report, the decoding baseline and
  fitted probes (α, fit R²), caveats, and the mean
  `train_step` diagnostics. Those are the same `Contrastive/*` and `Style/*` keys
  TensorBoard logs, so the audited forward can be checked against the run's curves at
  the checkpoint step.
