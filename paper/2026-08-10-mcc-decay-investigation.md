# The patch-MCC decay: what it is, what it is not

Investigation of 2026-08-09/10. Subject run:
`results/synthetic/synthetic-causal-clean-content-mse-real-lambda-01-normalized`
(Barlow Twins, patch+GAP, `content_size 44` of `hidden 48`, `clean_content`, 89k steps).
Comparators: `synthetic-causal-baseline-44-final` (recon-only) and a frozen-encoder arm.

**Starting question.** `selection/mcc_by_pool/patch` peaks at step 2001 and decays for the
remaining 86k steps, while the recon-only baseline rises monotonically. Why, and can it be
prevented without giving up reconstruction?

---

## 1. Headline findings

**The decay is real but small, and it sits on a large floor.** Peak 0.8994 @2001, final
0.8508 @88001, per-seed sd 0.0009–0.0016, so the drop is ~30 sigma. But an untrained
encoder already scores ~0.86 on the same probe: the trained model ends **+0.008** above
random initialisation and peaked at **+0.039**. The permutation null (0.08) is not the
relevant baseline; random init is.

**The metric is dominated by globally-available information.** At the final checkpoint,
background positions score 0.8574 against foreground's 0.8760, and 32 "pure brain"
positions (1408 features) match 148 strictly-covered positions (6512 features) to within
the error bar. Patch pooling is therefore not delivering what it was introduced for
(exposing `lesion_x/y/z`).

**Reconstruction causes the decay; the contrastive objective does not.** Measured directly
by stepping along each loss's gradient and re-measuring block-MCC, against a random
direction matched to the same per-tensor energy profile. At the final checkpoint,
excess over the matched-random control:

| direction | eta=0.05 | eta=0.2 | eta=0.8 |
|---|---|---|---|
| recon | −0.0040 | **−0.0087** | −0.1158 |
| contrastive | −0.0000 | +0.0017 | −0.0038 |
| vq | −0.0002 | −0.0035 | +0.0023 |

Recon's raw dMCC is linear in eta (R^2 0.957–0.999); the contrastive rows are flagged
non-linear (R^2 0.475) and straddle zero. Quote the eta=0.2 column; eta=0.8 is outside the
local regime.

**The objective is inactive for 98% of training — but not because it converges unusually
fast.** All four Barlow Twins terms complete >=93% of their descent before the MCC peak
(post-peak shares: on_diag 0.2%, off_diag 5.8%, sim 6.6%, var 0.0%), and `on_diag` is
90%-saturated by step 450. With 2000 training samples at batch 128 that is 15.6 steps per
epoch, so **step 2001 is epoch 128 and step 88001 is epoch 5632**. Convergence at 128
epochs is ordinary; the budget is ~44x longer than the objective needs.

**Consequence for the comparison.** The contrastive arm spends 98% of its training as a
reconstruction-only model with a 2k contrastive warm-start. Contrastive-vs-baseline as run
is therefore not a test of contrastive learning, and the **+29% reconstruction cost**
measured against the recon-only baseline is the cost of the *architecture* (content/style
split, 4 quantised style channels, SplitGroupNorm), not of an active objective. It should
not be reported as the objective's cost.

**Freezing the encoder trades reconstruction for identifiability, structurally.** Frozen at
the peak with the decoder trained out for 88k steps: masked L1 0.0450/0.0539 against the
joint run's 0.0250/0.0282 (+80%/+91%), with block-MCC pinned at 0.8859. The deficit is not
intensity calibration — a per-sample optimal gain+offset removes only 8% of it, and a
per-sample optimal *pointwise* map removes only 9%. The frozen representation encodes less
anatomy.

---

## 2. Hypotheses tested and rejected

Seven mechanisms were proposed from the configuration and theory. All were rejected by
measurement, not argument. Recording them because the negatives constrain the conclusion.

| hypothesis | how it died |
|---|---|
| Background dilution of the probe (training masks background, the probe does not) | `block_mcc` standardises every feature, so background magnitude is invisible: flat to 4 dp across a 67x sweep |
| `bt_lambda` decorrelation destroys the multi-channel co-activation local factors need | Measured the other way: a fully channel-private encoding scores **higher** than a redundant one (0.873 vs 0.829) |
| Conditioning / rank collapse | `content_rank` **rose** 18.7 -> 39.7; post-peak correlation with MCC +0.045 (p=0.77) |
| Style bottleneck drives the decay | `content_view_acc` is pinned at 1.0000 at both checkpoints — flat, so it cannot explain a change |
| The benchmark is too easy for the objective | `view_difficulty`: only 74% of view 2 is pointwise-explainable, contrast map is non-monotone (0.45), edge agreement 0.544. A genuine task |
| The frozen-run recon deficit is global intensity calibration | 92% of the error survives an optimal per-sample gain+offset; fitted gains are 1.01–1.03 |
| ...or a per-subject nonlinear intensity remap | 91% survives an optimal per-sample pointwise map (control: the joint run reads 99–102%, i.e. nothing to fit) |

---

## 3. Measurement problems found

Several conclusions in this project rest on instruments that were reading the wrong thing.

**`--freeze-encoder` was a silent no-op.** `vqvae_model` is wrapped in
`torch.nn.DataParallel` (`main_multimodal.py:1633`) before the freeze block (:1728), and
DataParallel does not proxy arbitrary attribute access, so every
`getattr(vqvae_model, "encoders")` returned `None`. Nothing was frozen; the run still
logged `[FREEZE ENCODER]`. One 88k run was voided. Fixed by unwrapping `.module`/`.online`
first, and the block now raises rather than proceeding when nothing matches.

**`eval/bt_term_balance.py` measured the wrong `sim`.** It called `barlow_twins_loss`
without `sim_normalize`, which defaults to `False`, while the run optimises the normalised
form. Every suggested `bt_sim_coeff` was off by a factor of `2*feat_std^2` — **5674x at
patch pooling**, turning a correct ~0.35 into 6e-5. Fixed.

**`content_view_acc`'s floor is 0.355, not 0.5.** `cv_probe_acc` runs StratifiedKFold on
the stacked view pair, so a subject's two rows land in different folds; the classifier
learns subject identity and inverts the paired test row. Measured on statistically
identical views: shipped 0.3550, subject-grouped 0.4975, unpaired control 0.4987. It does
not inflate a genuine positive (0.9062 vs 0.9019 with a real offset). Being linear, it can
also only ever detect a mean shift — a purely structural view difference reads 0.2631.
Deliberately **not** patched, since every recorded result depends on it.

**Pooling modes read different tensors** (fixed in a parallel session). `gap`/`patch` pool
before `SplitGroupNorm`; `stats` was read after it. `mcc_cc` and `content_anatomy` are
stats-pooled, so they were scoring a different representation from `mcc_by_pool/gap`.
Now unified on `norm_source="prenorm"`; **all stats-derived numbers for existing runs have
moved and need re-scoring.**

**VQ is counted twice.** `BaselineLoss` adds `quantization_losses` into its own return
(`losses.py:1611`) and `main_multimodal` then adds `sum(diffs) * vq_commitment_weight` on
top. The effective commitment weight exceeds the configured 0.25.

**`Loss-MAE-Reconstruction` never reaches TensorBoard.** `BaselineLoss` records it but
`get_summaries()` is never called, so there is no training-side pixel-L1 curve. `Loss/Recon`
is a different quantity (it includes the perceptual term and the VQ losses).

---

## 4. What the view structure actually is

`eval/view_difficulty.py` (data only, no model), 200 test subjects:

| map from view 1 -> view 2 | held-out R^2 |
|---|---|
| linear, per subject | 0.3751 |
| pointwise, shared across all subjects | 0.1040 |
| pointwise, per subject | 0.7391 |

Per-subject gain over a shared map: **+0.6350**. Shared-map monotonicity 0.45 (the contrast
*reorders* tissue intensities, as a real T1/T2 change does). Edge agreement between views
0.5440.

So the views differ by a **per-subject, nonlinear, non-monotone intensity remapping**, and
63.5% of view-2 variance requires the per-subject part. That is what a style code has to
carry; `style_sufficiency` is 0.38 and the style codebook is 44% utilised (perplexity ratio
0.438 with all 256 entries active). The gap between requirement and delivery is large.

Separately: the pre-norm content differs between views by a **pure constant offset and
nothing else** — a linear-alignment probe reads 0.5022, the floor. The strong nonlinear
view separability seen at `stats` pooling (0.9460) is manufactured by SplitGroupNorm's
per-sample statistics. Internal control from the same run: the stats `mean` group scores
0.779 while the gap block scores 0.5075 — the identical statistic on either side of the norm.

---

## 5. Objective diagnostics

At the final checkpoint (`bt_term_balance`, batch 128, unweighted):

| pooling | d | rows | on_diag | off_diag | noise floor | sim/(2 std^2) | 1-corr | feat_std |
|---|---|---|---|---|---|---|---|---|
| patch | 44 | 65536 | 0.0076 | 0.0370 | 0.029 | 0.022 | 0.013 | 53.26 |
| GAP | 44 | 128 | 0.0538 | 17.999 | 14.78 | 2.113 | 0.029 | 1.405 |

- The patch `off_diag` is **78% sampling noise**; the GAP one is **82%**. Neither is a
  useful optimisation target as configured.
- At GAP the **offset dominates the view disagreement by 73x**. That is what the `sim`
  distance term exists to remove and what the standardised correlations are blind to — and
  `bt_sim_coeff` was 2e-5, contributing ~2.4e-7 of the loss. Effectively off.
- `feat_std_mean` ran 0.2937 -> 66.9396. Nothing bounds the scale: `on_diag`/`off_diag` are
  correlations, `sim` was off, and the patch variance hinge is **structurally blind to
  subject collapse** — its `std(dim=0)` runs over folded (subject, position) rows, so
  spatial variance alone satisfies it (documented at `main_multimodal.py:1305`). Only the
  GAP hinge acts on subjects.

**A retry at `bt_sim_coeff 0.25` collapsed the run** (style perplexity and utilisation
down, reconstruction ~10x worse). Mechanism: the sim denominator is detached, so its
effective gradient scales as `sim_coeff / var` while the hinge's scales as `std_coeff`.
Below `std ~ sqrt(sim_coeff/std_coeff)` = 0.5 the collapse is self-reinforcing, and the run
had to traverse that on its way down from 53. Because the encoder trunk is shared, the
collapse propagated to style and to reconstruction. The fix is to raise the hinge
(`bt_std_coeff 10`, one-sided so it costs nothing above std 1) rather than lower `sim`.

---

## 6. Tooling produced

| file | purpose |
|---|---|
| `eval/patch_mcc_decay.py` | `--calibrate` (what the metric can detect), `curves`, `strata`, `geometry`, `viewgap` |
| `eval/gradient_attribution.py` | per-loss gradients on encoder params; dMCC per step with matched-random controls |
| `eval/view_difficulty.py` | how hard cross-view alignment is, from the data alone |
| notebook Section 23 | reconstruction quality, with affine and pointwise correction columns |
| `--freeze-encoder`, `--init-from-checkpoint` | two-phase training; `selection/encoder_l2` logs whether the freeze held |
| `experiments/synthetic_causal_frozen_encoder.yaml` | phase-2 config |

`--calibrate` is the reusable part: it establishes, on synthetic data of the same shape at
the real operating point, which perturbations block-MCC can even detect. Background noise
magnitude x67: 0.0000. Channel decorrelation: +0.0447 (helps). An **exactly invertible**
anisotropic map: −0.2566 at cond 1e2, −0.3812 at cond 1e4. Signal x0.5: −0.3869. Run it
before attributing any movement in that curve to a mechanism.

---

## 7. Open question and the next experiment

Four surviving observations point one way: recon (not contrastive) drives the decay; the
objective is done at epoch 128 of 5632; the metric is ~95% floor; and the 2k representation
is structurally poorer than the 88k one. All are consistent with a single deflating reading
— **the 2k "peak" is close to an untrained encoder, and block-MCC is largely measuring
proximity to a random projection.** A random projection preserves linear decodability of
everything, which would explain the 0.86 floor, background scoring as well as brain, and
the score falling as the encoder specialises for reconstruction.

**Decisive test, not yet run:** freeze a *randomly initialised* encoder and train the
decoder out for 88k steps.

- Random reaches ~0.045 masked L1 and ~0.86 block-MCC -> the 2k checkpoint is no better
  than a random projection on either axis, and the peak-vs-final story is a floor artifact.
- Random is clearly worse on both -> the 2k representation is real and the
  identifiability/reconstruction trade-off is genuine and structural.

Everything else (style capacity, `bt_sim_coeff` with the hinge rail, the freeze-point Pareto
sweep, a localised-factor probe with usable dynamic range) is worth doing but should wait on
this, since a null result would change what they mean.

---

## 8. Caveats to carry into any writeup

- Quote block-MCC against the **random-init value (~0.86)**, never the permutation null
  (0.08). The latter makes 0.87 look like a strong result.
- Report the contrastive arm's +29% recon cost and the freeze arm's +80% separately. They
  have different causes and merging them misattributes the freeze penalty to the objective.
- The gradient-attribution eta=0.8 column is superlinear and must not be quoted as a
  derivative. The contrastive rows there are the least reliable in the table: the
  batch-to-batch sd of that gradient is 87–104% of its mean.
- All stats-pooled metrics (`mcc_cc`, `content_anatomy`, `selection/overall_score`) moved
  when the pooling fix landed. Re-score before comparing any run to another.
- Everything here is one training seed. The ±0.001 error bars are probe-seed spread only
  and say nothing about run-to-run variance.
