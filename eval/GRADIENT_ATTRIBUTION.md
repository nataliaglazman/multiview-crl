# Gradient attribution against the real training objective

`eval.gradient_attribution` asks which loss terms move the **encoder** at a checkpoint,
how hard, and whether they pull against each other. Earlier versions rebuilt the loss
by hand (`_losses_at`) and drifted from training: no GAP arm, no correlation EMA, no
style or HSIC terms, and the wrong `bt_lambda` default. This version does not rebuild
the loss. It runs `train_step` itself, with no optimizer, and reads every weighted term
of `total_loss` from that one forward through `train_step(..., loss_observer=...)`.

```bash
python -m eval.gradient_attribution --run-dir results/synthetic/RUN                          # balance (default)
python -m eval.gradient_attribution --run-dir results/synthetic/RUN --target mcc             # + block-MCC steps
python -m eval.gradient_attribution --run-dir results/synthetic/RUN --target reconstruction  # pixel-MAE experiment
```

`--target reconstruction` is the separate similarity-vs-pixel-MAE experiment in
`eval/reconstruction_attribution.py`, unchanged. Everything below is about `balance` and `mcc`.

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

## `--target mcc`

This takes a temporary unit step along each **group's** mean gradient,
θ' = θ − η·g/|g|, and re-measures patch block-MCC. The same step is repeated along
matched-random directions that have the same per-tensor norms. A negative delta means
descending that loss lowers block-MCC. The **excess over random** is the attribution;
the raw delta also carries generic perturbation sensitivity. Compare the excess with
block-MCC's per-seed sd (~0.001). The linearity R² checks that the η sweep is small
enough for a finite difference to mean a derivative. Parameters and buffers are restored
after every trial, and on error. `mcc_steps.csv` and `mcc_summary.csv` hold the curves.

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
- `summary.json`: settings, arguments, EMA reference report, caveats, and the mean
  `train_step` diagnostics. Those are the same `Contrastive/*` and `Style/*` keys
  TensorBoard logs, so the audited forward can be checked against the run's curves at
  the checkpoint step.
