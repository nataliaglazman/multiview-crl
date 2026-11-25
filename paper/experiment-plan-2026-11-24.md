# Experiment plan for the paper: small signals at the theory's own width

24 November 2026, revised the same day. The method now uses matched width: the user chose it after checking that Yao
et al.'s theorem does not cover the earlier 24-unit code. A plan, not results. Numbers quoted as existing evidence come
from `out/burden_feasibility_20261102/HANDOFF.md` (items cited), `paper/burden-explanation.tex` or
`paper/prediction-identifiability.md`. They are one seed unless stated, and T1 / FLAIR where two are given.

## Summary

- **The paper.** At the width Yao et al.'s theorem asks for, one unit per shared latent, multiview training should
  identify all of the shared content. At 2,000 subjects standard training does not: it keeps the factors that tell
  subjects apart and misses the small ones. Generic, label-free additions let a matched-width code hold every factor's
  own part: the normative-residual input, independent per-view jitter, rectified-tail pooling, a cross-view prediction
  term and code-conditioned negatives. The code finds that solution in about half of seeds, and then its causal
  estimates come close to a 10-unit reference. Format as agreed on 18 Nov: one frozen recipe, three claims, an ADNI
  validation, and reading rules fixed before each result.
- **The main bank changes.** On the drawn-position bank the lesion's random position is a near-unique fingerprint
  that takes about half of a matched-width code (item 39). The main bank becomes the fixed-position bank: 7 shared
  latents, so 8 units. The drawn bank (10 latents, 11 units) becomes the test of that crowding.
- **What exists.** The frozen recipe at 8 units on five seeds (it holds all seven own parts in 3 of 5), its 9- and
  10-unit counterparts, leave-one-out ablations at 10 units, and causal tables for every code.
- **What is missing.** The theory's standard arm at matched width, the theorem-exact form of the recipe, enough seeds
  for hold rates, ablations at 8 units, a signal-size sweep, a label-free way to set the width on ADNI, and every ADNI
  number.
- **Headline figures.** Hold rate and own-part recovery at matched width, standard against recipe. A dose–response
  curve of the small factor against its size and its visibility in T1.
- **Minimum set (P0).** About 100 training arms on the PC (about 4 days of continuous GPU time) and about 18 ADNI runs.
  Section 9 lists the decisions for you.

## 1. Terms

- **View.** One modality's scan of a subject: T1 or FLAIR.
- **Content and style.** Content: the latents both views depend on (anatomy and the lesion). Style: the latents drawn
  separately for each view (intensity gain, bias, noise level, and the pose that jitter adds).
- **Backbone, head, code.** The backbone is a per-view convolutional encoder that gives a 64-channel 16³ feature map.
  The head is a shared MLP from the pooled features to the code. The code has `d` units; `d` is its *width*.
- **Matched width.** A width equal to the number of shared latents `|C|`, as in Yao et al.'s Theorem 3.2.
  `warm.py`'s InfoNCE compares codes by cosine, which ignores a code's length, so the matched width is `|C| + 1`.
  That is 8 units on the fixed-position bank (seven factors) and 11 on the drawn-position bank (seven factors and the
  lesion's x, y and z).
- **Own part** `ε_k` of factor `k`: its exogenous noise in the bank's structural causal model (SCM), independent of
  its parents by construction.
- **Recovery.** The 5-fold ridge R² of `ε_k` from a representation, on test subjects, averaged over the two views.
- **Alignment and completeness.** Yao et al.'s block identifiability asks that the code hold "all and only" the
  content. *Alignment* (the two views' codes agree) gives "only": style differs between the views, so an aligned code
  cannot depend on it. *Completeness* is "all": different contents give different codes, so every factor, own parts
  included, can be read back from the code. Alignment alone does not give it: a code holding only brain size aligns
  perfectly. Here completeness is graded and measured by own-part recovery. It is not DCI's "completeness" (each
  factor captured by few units), which is about how the code is arranged, not what it holds.
- **Holds, hold rate, matches** (`holds.py`, rules fixed before stage 64). A code *holds* all seven own parts when two
  values are each ≥ .40 in both views: the last value of its own-part spectrum (how well its weakest own direction is
  read) and the worst direction of the ventricle–burden plane. The *hold rate* is the share of seeds that hold. A code
  *matches* the 10-unit reference when every own part, the volume error, the ATE bias and the decoded-graph F1 are
  within set margins of it.
- **Curves.** Code directions spent on bent copies of factors the code already holds. The burden section calls them
  harmonics.
- **Availability** `A_k`: recovery from the head input. **Kept fraction** `κ_k`: recovery from the code divided by
  `A_k`.
- **Entry rank** `e_k`: the number of leading T1–FLAIR canonical directions of the head input needed to recover
  `ε_k` at half of `A_k`.
- **Small signal.** A shared factor whose own part enters late (large `e_k`). Three things make one: small extent (a
  lesion), faint visibility in one view (lesions in T1), or a parent that explains most of it (ventricle explains
  ~70% of burden).
- **Weak mean.** The mean recovery over the four weak factors: ventricle, burden, thickness and sulcal widening. The
  strong factors are brain size, left–right asymmetry (lr) and temporal atrophy.
- **Floor**: the same network untrained (the untrained twin). **Ceilings**: the code trained on lesion-volume labels,
  a fixed blob filter on the normative residual, and the true factors.
- **Anchor**: a measured factor (FreeSurfer volume, WMH volume) used as a causal variable. **Block node**: the code
  minus its fit on the anchors.
- **Causal tasks** (`eval.causal.tabular_benchmark`):
  1. *Confounding removed*, or *ATE bias*: how much of the image-driven bias in a treatment-effect estimate adjusting
     for the representation removes, or leaves (tabular-only bias .60).
  2. *Edge F1* of PC on decoded factors plus simulated tabular variables, split into edges out of image factors and
     tabular causes into them. The second kind exists only through own parts.
  3. *Decoded-graph F1*: the decoded factors' graph against the true SCM (truth reads .85).
- **Standard multiview objective**: InfoNCE, Barlow Twins or VICReg between the two views' codes, on globally pooled
  features.

## 2. Framing

### 2.1 Recommended: "identifiable in theory, missed in practice"

Working titles: *Identifiable but missed: small shared signals in multiview representation learning*; *Multiview
representations miss small signals at the theory's own width, and how to keep them*.

| claim | statement | evidence in hand |
|---|---|---|
| 1. Failure in the theory's own setting | At matched width, standard multiview training keeps the factors that tell subjects apart and misses the small ones. Entry rank predicts which go. Curves of strong factors take directions first, and one distinctive shared feature, the drawn lesion's position, takes about half the code. | 10–11-unit codes on the drawn bank keep brain, lr and temporal and drop the own parts of ventricle (.29–.38), burden (.27–.38), thickness (.17–.20) and sulcal (.07–.10) (Table `tab:burden-rank`). Entry rank predicts them at Spearman .82–.96. Without the residual input, burden is 0 in code and backbone (items 41, 42). |
| 2. Fix | The additions let a matched-width code hold all seven own parts, with no style. It does so in about half of seeds, and the codes that hold match the 10-unit reference. | The 8-unit recipe holds in 3 of 5 seeds from scratch, 2 of them matching (item 51). The seed-2 holder: own parts .56–.94, volume error 12.6 / 11.9 voxels (item 50). At 10 units, residual, jitter and prediction are each needed, and code batches are better on everything (item 41). No code carries style. |
| 3. Consequence | Holding the own parts is what the causal estimates need. | Holders: ATE bias .07–.09, decoded graph .82–.84, image–tabular F1 .95–.99 (items 50–51). Failing codes: ATE bias .12–.16, decoded graph .70–.78. ADNI: none yet. |

**Theory.** Yao et al.'s Theorem 3.2 is the target: alignment plus entropy at matched width identifies the content
block.
- *Inside the theorem:* the residual input (a fixed transform of each view), separate encoders, per-view jitter (it
  adds a view-specific pose latent) and the pooling.
- *Keeps the theorem's conclusion when its map is flexible and its target fixed:* the prediction term. A
  block-identifying code also minimises it, so adding it does not change the minimisers (`prediction-identifiability.md`,
  consequence 8; HANDOFF section 8). The proposition there holds at any width from `|C|` up, matched width included.
- *Outside the theorem:* code-conditioned negatives, because the entropy step assumes negatives drawn from the whole
  population. Also the linear prediction map: at narrow width a linear map rewards curves (stage 35).

The theorem-exact arm (Tm, section 4) drops both. The paper's subject is the gap between the theorem's optimum and
what training reaches with 2,000 subjects.

**What the paper must not claim.**
- That matched width makes the units the causal factors. The theorem identifies the block: label-free discovery on
  the units stays at the random-mixture baseline at 10 and 11 units (items 34, 43). Causal variables come from anchors.
- That the recipe identifies reliably. It holds in about half of seeds, and cross-view top-1 cannot tell which seeds
  (item 51).

### 2.2 Other framings

| framing | lead claim | status |
|---|---|---|
| A. Spare width | A 24-unit code holds the small signals reliably, and the width-free prediction proposition justifies it. | Superseded by the width decision. Its results become the spare-width comparison (E4) and the supplement (`burden-explanation.tex`). |
| B. Evaluation | Factor R² hides what causal inference needs; own-part recovery measures it across representation families. | Possible as a short paper from E1 and E6 alone. |
| C. The July draft | Contrastive learning added to reconstruction re-gauges content rather than identifying it. | Keep separate (EVIDENCE.md blocking items 1–7). |

Choose the venue before P1 work. A medical venue (MICCAI, MIDL) needs ADNI as the headline. An ML venue needs E5 in
full. Check current deadlines.

## 3. Ground rules

1. **Freeze the recipe now.** Stages 56 and 64's 8-unit arm on the fixed-position bank:
   ```text
   bank.py --t1-values 0.4 --causal --field off --lesion-position fixed   (+ residual_bank.py ... t1_04 flair)
   warm.py BANK RUN cold 20000 2000 --pool tail --t1key t1_04 --code-batches 500 --code-neighbours 31 --residual
       --jitter 2 --rotate 3 --patch-weight 0 --content-units 8 --contrastive-loss-type infonce --predict-weight 1
       --seed S
   ```
   On other banks only `--content-units` changes, to `|C| + 1`. Stage 53's result goes to the supplement whatever it
   shows. Stage 60 (the harmonic penalty at 8 units, five seeds, running) is read by its own pre-set rules. Whether it
   joins the recipe is a decision (section 9); joining means re-running the recipe's seeds with it.
2. **Development and held-out banks.** The recipe was developed on the drawn- and fixed-position banks (stages
   33–65). Every claim of generality is tested on banks never used for tuning (E7), with the recipe unchanged.
3. **Seeds and hold rates.** At matched width a seed either holds or loses one own direction, so report hold rates
   beside means. Use ten seeds for the recipe and the theorem-exact arm, five for S1 and S5, three for other arms.
4. **Reading rules before launch.** Each experiment below has one. Write it into the stage header and HANDOFF section
   10 before the stage runs, as for stages 56, 60, 64 and 66.
5. **Fair baselines.** Same backbone, data, steps (20,000), batch (32), augmentation and evaluation. The baselines get
   a tuning budget the recipe did not: temperature τ ∈ {.05, .1, .2} (Chen et al. 2021's lever against feature
   suppression), picked by validation top-1 among all validation subjects. Add one row with the τ that scores best on
   the primary endpoint itself (labels), so the baseline's best possible setting is shown.
6. **Lesion-specific tools stay diagnostics and ceilings**: blob filter, labelled code, spike-ins.
7. **Every reported number gets a line in a new EVIDENCE.md section**, with its status.

## 4. Arms

Fixed-position bank, 8 units, unless stated.

| id | arm | input | pooling | objective | negatives | prediction map | status |
|---|---|---|---|---|---|---|---|
| S1 | standard multiview CRL, the theorem's setting | scan | GAP + gauge (`gapnorm`) | InfoNCE | uniform | — | new |
| S3 | Barlow Twins | scan | `gapnorm` | Barlow Twins | uniform | — | new |
| S5 | standard objective, our input | scan + residual | tails | InfoNCE | code batches | — | new at 8 units (stage 50 ran it at 10) |
| R | the recipe | scan + residual | tails | InfoNCE + prediction, weight 1 | code batches | linear | 5 seeds (stages 56, 64) |
| Tm | the theorem-exact recipe | scan + residual | tails | InfoNCE + prediction, weight 1 | uniform | MLP (`--predict-hidden 128`) | new |
| R−x | leave-one-out | | | | | | new at 8 units (stage 50 ran them at 10) |
| W, C | prediction targets: the own view's GAP; the other view's code | | | | | | build |
| R9, R10 | one and two units past matched width | | | | | | 4 and 7 codes (stages 50, 52, 59, 63–65) |
| R24 | the earlier 24-unit recipe | | | | | | drawn bank only (stages 33–34) |
| refs | floor, backbone GAP, labelled code (`--supervise volume`), blob filter, true factors | | | | | | exist |

S1 and S3 get the recipe's per-view jitter. Independent per-view augmentation is standard practice, and without it a
code memorises (cross-view top-1 among 1,000 test subjects .39, item 41). E4 shows jitter's own effect.

## 5. Experiments

Priority: **P0** needed for any version; **P1** strengthens it; **P2** optional. "Arms" counts new 20,000-step
training runs.

### E0. Housekeeping and freeze (P0, no training, about a day)

- **Copy the unsaved burden analyses into the repo today.** EVIDENCE.md B2–B5, B11 and B12 rest on scripts that exist
  only in a session scratchpad, which a reboot clears:
  `/private/tmp/claude-502/-Users-nataliaglazman-Desktop-PhD-projects-multiview-crl/77e32499-5c1e-4c8e-9d77-1bf404f016d1/scratchpad/`
  (`rank_rule/{rank_check,analyse,rank_dirs}.py` with their JSONs, `shared_error/{shared_error,summarise}.py` with
  JSONs, `squash_check.py`). Destination: `out/burden_feasibility_20261102/`. E2b needs `rank_check.py`.
- Read stage 60 by its pre-set rules when it finishes.
- Run stage 67's `export` once, so every no-training analysis reads `causal/codes/<run>.npz`.
- Write the frozen recipe and the reading rules of E1–E8 into HANDOFF section 10 and start the new EVIDENCE.md
  section.
- Build `warm.py --tau`. Smoke-test S1 (scan-only input, `gapnorm`, 8 units) and Tm (`--uniform --predict-hidden 128`
  at 8 units) on the local bank.

### E1. Main comparison at matched width (claims 1–2; P0; about 31 arms)

- **Question.** In the theorem's own setting, does standard training miss the small signals that the recipe keeps?
- **Design.** Fixed-position bank, 8 units. First a τ grid for S1 at seed 42 (3 arms). Then S1 at five seeds, S3 at
  three, S5 at five, R at ten (seeds 42, 2, 3, 4 and 5 exist) and Tm at ten.
- **Reports per run.** `own_variation.py`, `precision_audit.py`, `code_audit.py`, `style_audit.py`,
  `own_directions.py` (where each direction goes) and `holds.py` (holds and matches against the 10-unit reference).
- **Primary endpoints.** Hold rate and the weak mean.
- **Reading rule.** R's hold rate exceeds S1's by at least .4. R's weak mean exceeds S1's and S5's by ≥ .10, paired
  by seed. R's own-view style R² is ≤ .05 for each style factor in every seed. Tm is reported the same way. If its hold
  rate is within .1 of R's, the paper can state its method fully inside the theorem.
- **Expectation.** S1 holds in no seed, since without the residual input burden reads 0 (item 41). R holds in about
  half.

### E2. Why matched-width codes miss small signals (claim 1; P0; no training except E2c)

- **E2a Backbone or head.** `A_k` and `κ_k` for every E1 code. Reading rule for "the objective discards it": S5 has
  burden availability ≥ .5 and burden kept fraction ≤ .7.
- **E2b Entry rank against width.** Entry ranks measured once on a reference head input predict which own parts a
  code of each width keeps. Existing: five narrow codes on the drawn bank, Spearman .82–.96. Add the fixed bank's 8-,
  9- and 10-unit codes. Reading rule: Spearman ≥ .7 for every code.
- **E2c Fingerprint crowding** (3–4 arms). The drawn bank at its matched width (11 units) against the fixed bank at
  8: R and S1 at seeds 42, 2 and 3 on the drawn bank (R at seed 42 exists, stage 38). Reading rule: on the drawn bank
  at least 40% of the code's directions go to the lesion's position, and it holds fewer weak own parts than on the
  fixed bank (existing, one seed: anatomy / lesion .34 / .54 per direction against .81 / .19, item 39).
- **E2d Where the directions go.** `own_directions.py` on every E1 code: directions on own parts, on curves, and the
  one lost. Existing pattern: 6 own directions and 2 curves at 8 units; a failing code loses the minor axis of the most
  correlated pair, the ventricle–burden difference (items 46, 51).
- **E2e InfoNCE's price for a small signal.** The precision audit's price block on S5 and R. It comes with E1's
  reports.

### E3. Signal-size dose–response at matched width (claims 1–2; P0 for E3a–c, P1 for E3d; about 30 arms)

- **Question.** As the shared signal gets smaller or fainter, where does each method lose it?
- **Banks.** Each is a new fixed-position render plus its residual, otherwise identical to the main bank.
  - **E3a, lesion size.** `--radius` .06, .08 and .12 (main bank .10), so lesion volume ×.22, ×.51 and ×1.73. Check
    first that the fixed site still holds the sphere at .12, and that the renderer's redraw rate stays low.
  - **E3b, visibility in T1.** T1 lesion value 0.0 and 0.6 (main bank 0.4). Realised contrast against white matter is
    about −.45 at 0.0 and −.22 at 0.4, so 0.6 should give about −.11. Needs `bank.py`'s `T1_KEYS` to accept 0.6.
    Calibrate the axis on MICCAI WMH 2017, whose manual masks the lab holds (`h1:/nfs/project/AMIGO/WMH`; check which
    agreement covers it): measure real T1 lesion-to-white-matter contrast-to-noise and mark it on the plot.
  - **E3c, size of the own part.** The independent fixed-position bank (`bank_nofield_fixedpos`, stage 58; own part =
    the whole factor) against the causal one. Exists.
  - **E3d, a small signal that is not a lesion** (P1). Focal temporal atrophy at two `--temporal-sigma` values below
    the default.
- **Arms per level.** R and S1 at two seeds, S5 at one. Ceiling per level: the labelled code (one seed). On the fixed
  bank the blob filter is no ceiling, because the residual partly absorbs a fixed lesion (item 39).
- **Figure.** Panel 1: the small factor's own-part recovery against the knob. Panel 2: its entry rank against the
  knob. Panel 3: the other factors, to show what is traded.
- **Reading rule.** At every level where the labelled ceiling recovers the small factor's own part at ≥ .5, R − S1 ≥
  .10 on it. Expected shape: the factor leaves a matched-width code once its entry rank passes the code's reach, and R
  keeps it to smaller sizes than S5.

### E4. Which additions carry the recipe at matched width (claim 2; P0; about 23 arms)

- **Leave-one-out at 8 units, three seeds each:** −residual, −jitter, −tails, uniform negatives, MLP map. (Tm is
  uniform negatives and MLP map together.)
- **What the prediction term predicts,** three seeds each:
  - **W, within-view:** predict the own view's whitened GAP. The theory says reconstruction-like targets pay for style
    (`prediction-identifiability.md`, consequence 2). Reading rule: the cross-view target is what matters if W's
    own-view gain R² exceeds R's by ≥ .05 or its weak mean falls ≥ .10 below R's.
  - **C, the other view's code** (a BYOL/JEPA-like latent target). Its target moves with the code. Reviewers will ask
    for it; report it as it comes.
- **Spare width, for comparison:** 24 units on the fixed bank (two seeds), beside the existing 9- and 10-unit codes.
  Existing: 9 units hold 3 of 4 at every 500 and 10 units 6 of 7, so the ninth unit goes to curves (item 51).
- **Reading rule per component.** "Needed" if removing it lowers the weak mean by ≥ .10, or raises burden's volume
  error by ≥ 2 voxels, in at least 2 of 3 seeds; otherwise "optional". Report each factor's trade, not only the mean:
  at 10 units the residual cost ventricle (.71 without it, .49 with it) while it brought burden from 0 (item 41).

### E5. Theory at matched width (P0 for E5a and E5c; Tm runs in E1)

- **E5a Toy** (`training/prediction_identifiability_numerical.py`; no images, minutes per run). At matched width
  (`k = 5`, and 6 for the cosine) no objective read the weak dimension above .09 within 8,000 steps (the note's E5).
  Run five seeds at four times the steps: does any objective reach the weak dimension at matched width? Keep `k` 10
  and 20 as the spare-width comparison. Either answer is reportable: if one does, the theorem's optimum is reachable;
  if none does, matched width loses a weak factor even in the toy, a finite-training effect.
- **E5c Target condition on the bank.** Render eight style draws per subject (`style_audit.py`'s redraw), average the
  prediction target, and read each factor's own part from the average. Reading rule: ≥ .8 for every factor.
- **E5d Linear against flexible map.** The theory says a linear map at narrow width pays for curves. E4's MLP-map
  arm tests it. Reading rule: the MLP map spends fewer directions on curves (`own_directions.py`: at most 1 against
  the linear map's 2).

### E-W. Setting the width on real data (P0 for ADNI; about 12 PC arms and 6 ADNI runs)

- **Problem.** Matched width needs `|C|`. ADNI does not give it.
- **Candidate label-free rules, fixed before any result:**
  1. the elbow of the held-out loss over a width sweep (Daunhawer et al.'s practice, per the 20 Nov note);
  2. TwoNN intrinsic dimension of the widest code, corrected by the bank's calibration offset (seven factors read 6.4
     on the fixed bank, item 46);
  3. the smallest width whose held-out cross-view retrieval is within a set margin of the widest code's.
- **Validation.** The recipe at widths 4, 6, 8, 10, 12 and 16 on three banks of known size: fixed causal (`|C|` 7),
  fixed independent (7) and drawn causal (10). One seed per width, reusing the widths that exist. A rule passes if it
  lands within one unit of `|C| + 1` on all three banks.
- **Then ADNI.** The same sweep, one seed per width; the passing rule sets ADNI's width. If no rule passes, ADNI's
  width is a stated choice reported with the results at ± 2 units, and the paper says matched width is defined only on
  the banks.

### E6. Causal consequences (claim 3; P0 for E6a–b, P1 for E6c–f; no training)

- **E6a Tabular benchmark** on every E1, E3 and E4 code, each backbone GAP, the floors and the oracle, ten task seeds:
  ATE bias (linear and `rbf` learners), edge F1 in both directions, decoded-graph F1 with resampling. Split codes into
  holders and non-holders. Reading rule: R's mean ATE bias is at least .10 below S1's in every seed pair, and holders'
  ATE bias is below non-holders' (existing: .07–.09 against .12–.16).
- **E6b Own part against R² as the predictor of causal utility,** pooled over every code and factor. Reading rule:
  own-part recovery predicts the per-factor edges found with a Spearman at least .2 higher than raw R² does (stage 22:
  a code reading thickness at R² .79 found 0 of 5 thickness edges).
- **E6c Confounding through one small signal.** The hidden image score built from one factor's own part. Needs a
  small `tabular_benchmark` option.
- **E6d Individual effects** (stage 55, built, not run): effect modifiers set by own parts; error of the individual
  effects (PEHE) and policy regret.
- **E6e A second view cannot fix what both omit** (item 37): the proximal arms on the E1 codes.
- **E6f Readouts under a changed distribution:** fit factor readouts on the fixed causal bank's codes and test them on
  the fixed independent bank's subjects, encoded by the same networks. A code that reads a child through its parent
  fails there.

### E7. Banks not used for tuning, recipe unchanged (claim 2's generality; P1; about 24 arms)

- **(a) Lesions placed by the anatomy.** Several small lesions per subject, placed by the anatomy (for instance
  against the ventricles, as periventricular WMH are), burden as their total volume. No position latents, so `|C|`
  stays 7. This checks the fixed bank's quirks: its lesion sits at the ventricle's edge, which merges ventricle and
  burden for a narrow code (item 46), and its residual partly absorbs the fixed lesion. Needs generator work.
- **(b) Another SCM:** new edge weights and one different graph (needs `bank.py --scm-seed`).
- **(c) Stress banks:** gain off and field on, fixed position.
- **Arms.** R and S1 at three seeds, S5 at two, per bank.
- **Reading rule.** R − S1 ≥ .10 on the weak mean, and R's hold rate above S1's, in every bank.

### E8. ADNI (claim 3 on real scans; P0 for E8a, b and d; P1 for E8c and e; P2 for E8f and g)

- **E8a Training.** At the E-W width: R, S5 and S1, and Tm if it passes E1. Three runs per arm, one per fold of a
  3-fold split stratified by diagnosis and ADNI phase, each with its own seed, so every subject gets an out-of-sample
  code. Stage 68 as built (24 units) runs as the spare-width reference. Read first, as stage 68 says: validation top-1
  and its train–validation gap, the modality probe against October's .96, effective rank. Needs a fold option in
  `adni_bank.py` or `warm.py --real`.
- **E8b Anchors**, labels used only to score. FreeSurfer 7 (log supratentorial volume or ICV, lateral ventricles,
  hippocampus, entorhinal thickness, mean cortical thickness) for 1,543 of 1,620 subjects; UC Davis WMH (log) for 941.
  Own-part proxies: each anchor's residual after a cross-fitted flexible fit on the other anchors, age and sex. Score:
  5-fold ridge R² of each anchor and proxy from each code, within each held-out fold, against the untrained floor.
  Reading rule: R − S5 ≥ .05 on the WMH proxy in all three folds, and on the mean proxy. Needs a scorer that matches
  scans to anchor rows through the manifest's image IDs and dates.
- **E8c Anchor reliability** (P1). A second WMH measure, FreeSurfer's T1 white-matter hypointensity volume, bounds the
  R² any code can reach against the UC Davis measure.
- **E8d Semi-synthetic causal tasks on real scans.** The tabular benchmark with anchors as the image factors and their
  proxies as own parts; real age, sex, education and APOE4 as tabular columns; simulated treatment and outcome
  (general, and confounding through the WMH own part); simulated tabular causes of the anchors' own parts. Truth is
  known by construction. Reading rule: R − S5 ≥ 10 points of confounding removed in every fold. Report adjustment for
  the anchors themselves as the oracle row. Needs an ADNI bundle writer in `causal_bundle.py`'s format.
- **E8e Spike-in lesions on real scans** (P1). WMH-like lesions of known volume rendered into both views of real
  pairs: bright in FLAIR, iso- to hypointense in T1 at E3b's calibrated contrast, partial volume, placed in white
  matter. Train R and S5 on the spike-in bank (two seeds) at E3a's size levels and read the spike-in volume's own part.
- **E8f Known associations** (P2): code-derived WMH against age and vascular risk; code-derived hippocampus against
  diagnosis and APOE4; both beside the anchors' estimates.
- **E8g Acquisition** (P2): leave one ADNI phase out; decode site from the code.

### E9. Beyond ADNI (P2)

- A second cohort with T1 and FLAIR (UK Biobank, OASIS-3, or a lab-held one such as EPAD or SABRE): anchor readouts
  fitted on ADNI codes, applied without refitting. The width comes from the E-W rule on that cohort.
- Other representation families through the bundle format: a shared-plus-private multimodal autoencoder on the same
  backbone; frozen DINOv3 slices (`eval/dino/`, needs a bank adapter).
- Subsampling a randomised trial (A4), if access comes through.

## 6. What the paper shows

| # | figure or table | from |
|---|---|---|
| 1 | Schematic: the theorem's matched-width optimum against what training reaches, and where a small signal is lost | E2a |
| 2 | Hold rate and weak mean at matched width for S1, S3, S5, R and Tm | E1 |
| 3 | Where a matched-width code's directions go; fingerprint crowding on the drawn bank | E2c, E2d |
| 4 | Dose–response, with E3d as an inset | E3 |
| 5 | Ablation at 8 units, and the width curve 8 → 9 → 10 → 24 | E4 |
| 6 | Causal tasks for holders, non-holders and the standard arm; own part against R² as the predictor of edges found | E6a, E6b |
| 7 | ADNI: the width rule, the anchors and the WMH proxy, semi-synthetic confounding | E-W, E8b, E8d |
| 8 | Theory box: Theorem 3.2, the added-term argument, and the toy at matched width | E5 |

Supplement: `burden-explanation.tex` as "what spare width buys" (the 24-unit recipe on the drawn bank and burden's
three-step shortfall), the compression start (item 52), rebuild intervals (items 49–51), stage 60, the held-out banks
(E7), style audits, precision ladders and the τ grid.

## 7. Order and compute

PC throughput is about 1.5 arms an hour (20,000 steps take about 2 hours, three in parallel), plus about 1.5 hours of
reports per stage. New stages take the next free numbers in `run_pc.sh` (69 onwards at the time of writing).

| step | what | where | size |
|---|---|---|---|
| 1 | E0, and read stage 60 | Mac, PC | about a day |
| 2 | E1 | PC | about 31 arms |
| 3 | E-W sweep on the banks | PC | about 12 arms |
| 4 | E-W sweep on ADNI, then E8a | Run:ai | 6, then 9–12 runs |
| 5 | E4 and E2c | PC | about 27 arms |
| 6 | E3a–c | PC | about 30 arms, 5 banks |
| 7 | E2, E5 and E6 reports | PC, Mac | no image training |
| 8 | E8b and E8d builds, then scoring | Mac, Run:ai | no new training |
| 9 | P1: E3d, E7, E8c, E8e | PC, Run:ai | about 40 arms |

P0 comes to about 100 arms: about 67 hours of training plus the reports, so about 4 days of continuous PC time, or
two to three weeks with reading between stages. ADNI needs about 18 runs. Matched width costs more than the earlier
plan because hold rates need seeds.

## 8. Builds

Each one is smoke-tested on the local bank and dry-run before the command is handed over (HANDOFF section 2). Tm needs
no build: `--uniform --predict-hidden 128` without `--code-batches`.

| build | for | size |
|---|---|---|
| `warm.py --tau` | E1 τ grid | small |
| `warm.py` within-view and code targets | E4 | small |
| `warm.py` VICReg | an optional S4 | small |
| Width-rule script over a width sweep (elbow, TwoNN with offset, retrieval margin) | E-W | small |
| `bank.py` T1 value 0.6; radius variants at the fixed site | E3 | small |
| `bank.py` lesions placed by the anatomy; `--scm-seed` | E7 | medium; small |
| `tabular_benchmark` confounding through one factor's own part | E6c | small |
| ADNI folds, anchor matching and own-part proxies, bundle writer | E8a, b, d | medium |
| Spike-in renderer for real pairs | E8e | medium to large |
| Copy the scratchpad analyses into the repo | E0, E2b | trivial |

## 9. Decisions for you

1. **Main bank.** Recommended: the fixed-position bank at 8 units, with the drawn bank as the crowding test (E2c)
   and the anatomically placed bank (E7a) as the realism check.
2. **Code batches in the recipe.** Recommended yes: they are what makes matched width hold (item 41), but they sit
   outside the theorem's negative sampling. Tm reports the theorem-exact form beside them.
3. **Stage 60's harmonic penalty.** If it passes its rules: add it to the recipe (and re-run R's seeds), or report it
   as a finding only.
4. **ADNI's width.** Recommended: whichever E-W rule passes on the three banks; the elbow of the held-out loss is the
   expected candidate.
5. **Baselines get per-view jitter.** Recommended yes.
6. **ADNI causal evaluation, second cohort, venue.** Recommended: semi-synthetic tasks (E8d) at P0, spike-ins (E8e)
   at P1; a second cohort only if time allows.

## 10. Risks and fallbacks

- **Half the seeds fail, and no label-free signal picks the ones that hold** (top-1 overlaps, item 51). Report hold
  rates. Stage 60 may raise them. The clean-compression rule (item 52: every removed direction has held-out R² near 1)
  is an untested label-free selector: fix it before testing it on new seeds (P1).
- **The fixed bank's quirks.** Its lesion sits at the ventricle's edge, the residual partly learns the fixed lesion as
  normal variation (blob filter 21.4 / 15.3 voxels against 9.0 / 6.6), and burden reads more easily at a fixed site.
  E7a's bank checks the claims without them.
- **No width rule passes on the banks.** Then ADNI's width is a stated choice with a sensitivity analysis, and the
  matched-width claim stays synthetic. Real anatomy may have many shared latents, so the estimated width may come out
  large; that is a finding, not a failure.
- **Tm holds less often than R.** Then the method sits partly outside the theorem (code batches, linear map), and the
  paper says so.
- **ADNI shows no gain on WMH.** Check WMH's entry rank on the ADNI features and its T1 visibility (E3b's
  calibration). If real WMH is mostly FLAIR-only content, a shared code should not hold it (Daunhawer et al. 2023), and
  the real-data claim narrows to small shared structures (hippocampus, entorhinal cortex).
- **"Why not use spare width?"** E4's width curve answers what spare width buys. The paper's position: the theorem's
  width is the principled choice, and its finite-sample cost is part of the finding.
- **Novelty.** Cross-view prediction resembles BYOL and JEPA; the normative residual resembles pseudo-healthy anomaly
  detection; rectified pooling resembles top-k or generalised-mean pooling. The contribution is the account of what
  matched-width training misses and why, the additions that keep it, and the causal evaluation. Before writing, search
  for self-supervised small-lesion work and multimodal brain MRI pretraining.
