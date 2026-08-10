# The patch-MCC decay: what it is, what it is not

Investigation of 2026-08-09/10. Reference run:
`results/synthetic/synthetic-causal-clean-content-mse-real-lambda-01-normalized`
(Barlow Twins, patch+GAP, `content_size 44` of `hidden 48`, `clean_content`, 89k steps).
Comparators: `synthetic-causal-baseline-44-final` (recon-only, 34k), a frozen-encoder arm,
a frozen-random arm, and a short contrastive-only arm.

**Starting question.** `selection/mcc_by_pool/patch` peaks at step 2001 and decays for the
remaining 86k steps, while the recon-only baseline rises monotonically. Why, and can it be
prevented without giving up reconstruction?

---

## 1. The four arms, side by side

| arm | patch-MCC | masked L1 | content→style leak |
|---|---|---|---|
| recon-only baseline (34k) | 0.8929 @24k, **holds** 0.8926 | **0.0194** | 0.5716 |
| joint contrastive (89k) | 0.8994 @2k → **0.8508** | 0.0250 | **0.0316** |
| contrastive-only (4k only) | 0.9021 @2k → 0.8969 @4k | n/a | 0.1889 |
| frozen encoder @peak | 0.8859 (pinned) | 0.0450 | — |
| frozen random encoder | below all trained arms | — | — |

**The joint run at 1:1 is Pareto-dominated by the recon-only baseline** on both MCC and
reconstruction. Its only win is content purity, and that win is 18x. That is the single
most important number to carry forward: *what the contrastive objective currently buys is
purity, not identifiability.*

The decay is real (per-seed sd 0.0009-0.0016, so ~30 sigma) but sits on a large floor: a
randomly-initialised encoder already scores ~0.86, so the trained model ends **+0.008**
above random and peaked at **+0.039**.

---

## 2. Established mechanism

**Recon's effect on identifiability is configuration-dependent, sign and all.** Excess over
a matched-random direction at eta=0.2:

| configuration | recon excess | recon↔vq cosine | ‖g_recon‖ |
|---|---|---|---|
| baseline (recon-only, 34k) | **+0.0048** | **+0.346** | 0.150 |
| contrastive @ peak (2k) | +0.0002 | −0.144 | 0.763 |
| contrastive @ final (89k) | **−0.0087** | **−0.532** | 0.272 |

Reconstruction *improves* block-MCC along its own trajectory and *degrades* it from the
configuration the contrastive objective produces. Reconstruction and vector quantisation
move from agreement to conflict over the same span.

**But the decay tracks the contrastive objective's presence, not recon's.** Recon-only does
not decay; both arms containing a contrastive term do. Those two facts are not yet
reconciled, and reconciling them is the open scientific question.

**The decay is a real loss, not a re-gauging.** At GAP pooling, best -> final: ridge
0.8309 -> 0.7992 (−0.0317), kernel 0.8458 -> 0.8251 (−0.0207). A nonlinear readout recovers
35%; **65% survives every probe tried**. A linear map from the late block into the early
basis reaches 0.7899 against the true early 0.8233 — it cannot manufacture what is not
there. The map R^2 asymmetry (late->early 0.780, early->late 0.504) says the late block
holds *new* information while having lost factor-decodable structure: a trade, not decay.

**The decay is brain-localised**, best -> final gap: pure brain −0.0469, fg strict −0.0324,
fg train-rule −0.0284, **background −0.0053**.

---

## 3. Hypotheses tested and rejected

Ten mechanisms, all argued from configuration or theory, all killed by measurement.

| hypothesis | how it died |
|---|---|
| Background dilution of the probe | `block_mcc` standardises every feature; flat to 4 dp across a 67x background-noise sweep |
| `bt_lambda` decorrelation destroys local co-activation | Measured the other way: fully channel-private scores **higher** (0.873 vs 0.829) |
| Conditioning / rank collapse | `content_rank` **rose** 18.7 -> 39.7; post-peak correlation with MCC +0.045 (p=0.77) |
| Style bottleneck drives the decay | `content_view_acc` pinned at 1.0000 at both checkpoints — flat cannot explain a change |
| The benchmark is too easy | Only 74% of view 2 is pointwise-explainable; contrast map non-monotone (0.45); edge agreement 0.544 |
| Frozen-arm recon deficit is global intensity calibration | 92% of the error survives an optimal per-sample gain+offset; fitted gains 1.01-1.03 |
| ...or a per-subject nonlinear intensity remap | 91% survives an optimal per-sample pointwise map (control: joint run reads 99-102%) |
| The joint run relaxes toward recon's solution | It would then land at 0.8926; it lands at 0.8508, *below* the baseline |
| block-MCC measures proximity to a random projection | The frozen-random arm scores below both trained arms |
| `off_diag` noise-fitting random-walks the encoder | Gradient SNR: contrastive EXPECTED 11.11, signal share 0.654 — a genuine descent direction |

---

## 4. Measurement problems found

Five instruments were reading the wrong thing. Any earlier result touching these needs
re-checking.

**`--freeze-encoder` was a silent no-op.** `vqvae_model` is wrapped in
`torch.nn.DataParallel` (`main_multimodal.py:1633`) before the freeze block (:1728), and
DataParallel does not proxy attribute access, so every `getattr(..., "encoders")` returned
`None`. **One 88k run voided.** Fixed by unwrapping first; the block now raises rather than
proceeding, and `selection/encoder_l2` logs whether the freeze held.

**`bt_term_balance` measured the wrong `sim`.** It called `barlow_twins_loss` without
`sim_normalize`, which defaults False, while the run optimises the normalised form. Every
suggested `bt_sim_coeff` was off by `2*feat_std^2` — **5674x at patch**. Fixed.

**`content_view_acc`'s floor is 0.355, not 0.5.** `cv_probe_acc` runs StratifiedKFold on the
stacked view pair, so a subject's two rows land in different folds; the classifier learns
subject identity and inverts the paired row. Measured on identical views: shipped 0.3550,
subject-grouped 0.4975, unpaired control 0.4987. Being linear it also only ever detects a
mean shift (a purely structural difference reads 0.2631). Deliberately **not patched** —
every recorded result depends on it.

**Poolings read different tensors** (fixed in a parallel session): `gap`/`patch` pool before
`SplitGroupNorm`, `stats` was read after. **All stats-derived numbers for existing runs have
moved and need re-scoring.**

**VQ is counted twice.** `BaselineLoss` adds `quantization_losses` into its return
(`losses.py:1611`) and `main_multimodal` adds `sum(diffs) * vq_commitment_weight` on top.
`scale_recon_loss` therefore also scales VQ.

**`Loss-MAE-Reconstruction` never reaches TensorBoard** — `get_summaries()` is never called.
`Loss/Recon` is a different quantity (it includes perceptual + VQ).

**The matched-random control has a blind spot.** It subtracts generic perturbation
sensitivity, which is correct — but that is exactly the signature a noise-driven mechanism
would produce, so a loss whose gradient is near-random shows zero excess by construction.
Worth stating whenever the dMCC table is quoted.

---

## 5. The objective, measured

At the final checkpoint (`bt_term_balance`, batch 128, unweighted):

| pooling | d | rows | on_diag | off_diag | noise floor | sim/(2 std^2) | 1-corr | feat_std |
|---|---|---|---|---|---|---|---|---|
| patch | 44 | 65536 | 0.0076 | 0.0370 | 0.029 | 0.022 | 0.013 | 53.26 |
| GAP | 44 | 128 | 0.0538 | 17.999 | 14.78 | 2.113 | 0.029 | 1.405 |

- `off_diag` is **78% sampling noise at patch, 82% at GAP**.
- At GAP the **offset dominates the view disagreement 73:1**. That is what the `sim` distance
  term exists to remove and what the standardised correlations are structurally blind to —
  and `bt_sim_coeff` was 2e-5, contributing 2.4e-7 of the loss.
- Normalised `sim` puts **99% of its weight at GAP** (2.113 vs 0.0215); unnormalised puts
  94% at patch (122.157 vs 8.349), where there is no offset. Normalisation is what aims the
  term at the problem.

**The objective's activity is contingent on reconstruction.** Both runs start at
`feat_std` ~0.28 with `var_loss` ~1.46. In the joint run recon drives `feat_std` to **66.9**
and the one-sided hinge goes dormant (`var_loss` -> 0) — all four terms genuinely converge.
In contrastive-only nothing pushes the scale up, `feat_std` falls to **0.239**, and the
hinge stays engaged and *losing* (`var_loss` 1.477 -> 1.512, `sim` 0.0999 -> 0.1217).

**Why the objective converges by epoch 128.** Not because the data is easy. At d=44 it
imposes 44 diagonal + 1892 off-diagonal constraints on two *independently parameterised*
encoders (~600k params), of which 1892 are 78-82% noise, the hinge is dormant, and `sim` is
off. With `separate_encoders`, correlating two independent networks never requires
invariance. `on_diag` is a *correlation* — satisfied by a per-channel linear relationship,
strictly weaker than the a.s. equality Yao's Lemma C.2 requires.

**`sim` attempts so far both failed.** 0.25 with `bt_std_coeff` 1 collapsed the run (style
perplexity and utilisation down, recon ~10x worse); 0.25 with `bt_std_coeff` 10 explodes
reconstruction at ~10k. The principled anchor **0.025** (matching GAP `on_diag`) is untested.

---

## 6. The data

`eval/view_difficulty.py` (no model, 200 test subjects):

| map view 1 -> view 2 | held-out R^2 |
|---|---|
| linear, per subject | 0.3751 |
| pointwise, shared across all | 0.1040 |
| pointwise, per subject | 0.7391 |

Per-subject gain **+0.6350**; shared-map monotonicity 0.45 (contrast *reorders* tissue
intensities); edge agreement 0.544. The views differ by a **per-subject, nonlinear,
non-monotone intensity remap**, and 63.5% of view-2 variance needs the per-subject part.
That is what a style code must carry; `style_sufficiency` is 0.38 with the style codebook
44% utilised (perplexity ratio 0.438, all 256 entries active).

Pre-norm content differs between views by a **pure constant offset and nothing else**
(linear-alignment probe 0.5022, the floor). The nonlinear view separability at `stats`
(0.9460) is manufactured by SplitGroupNorm's per-sample statistics. `content_view_acc` is
1.0000 in the **recon-only baseline too**, so it is architectural, not an objective failure.

---

## 7. Tooling produced

| file | purpose |
|---|---|
| `eval/patch_mcc_decay.py` | `--calibrate`, `--summarise`, and `curves` / `strata` / `geometry` / `viewgap` |
| `eval/gradient_attribution.py` | per-loss gradients, matched-random controls, dMCC sweep, `--snr` decomposition |
| `eval/view_difficulty.py` | alignment difficulty from the data alone |
| `eval/reformat_vs_loss.py` | probe ladder + cross-checkpoint linear recoverability |
| notebook Section 23 | reconstruction quality with affine and pointwise correction columns |
| `--freeze-encoder`, `--init-from-checkpoint`, `selection/encoder_l2` | two-phase training with a freeze that fails loudly |

`--calibrate` is the reusable part: what block-MCC can detect, at the real operating point.
Background noise magnitude x67: **0.0000**. Channel decorrelation: **+0.0447** (helps). An
*exactly invertible* anisotropic map: **−0.2566** at cond 1e2, **−0.3812** at 1e4. Signal
x0.5: **−0.3869**. Run it before attributing any movement in that curve to a mechanism.

---

## 8. Where it stands and what to run

**The open question** is why the decay tracks the contrastive objective's presence while
recon carries the local attribution. Ten mechanisms have died; the pattern each time was
that the effect was smaller than the noise, or the run was too short, or a control was
missing.

**Next: six runs, seed as the only variable, 34k steps, no config changes.**

Baseline x3 (`scale_contrastive_loss=0`) and reference-config x3. The result at n=1 —
baseline beats contrastive on MCC and reconstruction, contrastive wins 18x on purity — is
the gate on everything else, and altering any coefficient severs the link to every number
above. Read with `patch_mcc_decay --summarise`; **if the between-config difference does not
exceed the within-config seed spread, there is no effect to report.**

Deferred until that reports: the `bt_sim_coeff 0.025` run (judged on the GAP offset, not
MCC), the recon-weight Pareto sweep, a long contrastive-only arm, and the freeze-point
sweep.

**Still unbuilt, and the largest measurement gap:** a probe restricted to where the
localised factors actually are. `lesion_x/y/z` are what patch pooling was introduced to
expose, and background scores 0.8574 against foreground's 0.8760.

---

## 9. Caveats for any writeup

- Quote block-MCC against the **random-init value (~0.86)**, never the permutation null
  (0.08). The latter makes 0.87 look like a strong result.
- Report the contrastive arm's **+29%** reconstruction cost (vs the recon-only baseline) and
  the freeze arm's **+80%** separately. Different causes; merging them misattributes the
  freeze penalty to the objective.
- The contrastive arm spends **98% of training as a reconstruction-only model** with a 2k
  warm-start, so the +29% is the cost of the *architecture*, not of an active objective.
- The gradient-attribution eta=0.8 column is superlinear and must not be quoted as a
  derivative. The contrastive rows there are least reliable: batch-to-batch sd is 87-104%
  of the mean.
- All stats-pooled metrics moved when the pooling fix landed. Re-score before comparing.
- **Two random initialisations differ by 0.023** in step-1 patch-MCC — larger than most
  config effects measured here (0.005-0.011). Everything is n=1 and the error bars quoted
  are probe-seed spread only.
