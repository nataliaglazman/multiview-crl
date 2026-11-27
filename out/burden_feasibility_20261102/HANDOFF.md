# Burden feasibility: handoff (rewritten 27 Nov 2026)

This file holds what matters now: the goal, how to run things, the findings the paper rests on, what is running and
what comes next. The full history to 27 Nov, merged from both sessions' copies, is in `HANDOFF_archive_20261127.md`.
Its section 7 (items 1–70) and section 9 (each stage's rules, written before its results) are the record behind
everything here. Item and stage numbers below, and "HANDOFF section 9" in scripts' docstrings, refer to that archive.
Repo-wide conventions are in `CLAUDE.md` at the root.

## 1. Where things stand

- **The paper: "identifiable in theory, missed in practice".** Yao et al.'s multiview theorem says a contrastive code
  as wide as the shared content (matched width) is a one-to-one function of that content. On a synthetic paired
  T1/FLAIR brain-MRI bank with a known causal model, the matched-width code (8 units for 7 factors) keeps all seven
  factors in only about half of training runs, and no change to training fixes that.
- **The method that works: identify wide, chart narrow.** Train slightly wider (10 units), keep the run whose code
  best predicts the other runs' codes, and compress it with a non-linear chart to the content's own width (7 units).
  Every step has a label-free check. Shown on one bank so far.
- **Running (27 Nov):**
  - stage 85: the method on fresh seeds, on a second bank, and without code batches (four Run:ai jobs);
  - stage 86: two failing 8-unit codes trained a further 60,000 steps (one Run:ai job).
  - Watchers on the PC score both when their runs finish (section 6).
- **Paper draft:** the theory section is `paper/theory-matched-width-draft.tex` (four `\todo` notes for the user).
- **Not committed:** everything since `79ba8bc` in this checkout (stages 75–77, 84–86, `pool_check.py`,
  `panel_check.py`, `continue_check.py`, `warm.py --sufficiency-weight`, tests). The other session's work (stages
  78–83, items 63–66) is on `fork/temporal009-attention-full-width` at `7fab9e7`. Merge and commit only when the user
  asks; the archive already holds both sides' text.

## 2. How we work with the user

- **Rules before results.** Every stage gets written rules (what counts as works and fails) and an expectation in
  this file before any of its results exist. A change after that is recorded with its reason, and only before results.
  Read results against the rules; report misses plainly.
- **Ask before:** any new Run:ai submission, any commit or push, anything destructive. Commit only when asked; name
  files explicitly in `git add` (the tree holds the user's unrelated uncommitted work); end messages with the
  attribution line.
- **Agents run on the PC directly** (`ng24@sie114-u-pc`, RTX 3090, repo `~/projects/multiview-crl`) and reach Run:ai
  over `ssh aicheadnode`.
- **Never edit `run_pc.sh` in place while a stage runs:** bash reads it by offset. Edit a copy and `mv` it over, or
  append below the running block.
- **Run heavy reports one at a time** (`style_audit.py`, bundles, audits): 16 CPUs, and each uses many threads.
- **Another Claude session may be working in the same tree** (it writes under `causal/stage78`–`83`). Do not touch
  its files; agree stage numbers before using one (the next free number is 87).
- **Writing style:** define every term at first use; plain words and short sentences; numbers as T1 / FLAIR.
- **Keep this file current:** a stage's rules before its results, its reading as a finding after.

## 3. Setup

### Data

- **`bank.py`** renders synthetic subjects: 64³ volumes, 2,000 / 400 / 1,000 train / validation / test subjects.
  Each subject has 7 anatomical factors drawn from a known causal model (an SCM, structural causal model):
  ```text
  brain → ventricle → burden → temporal ← thickness ← brain;  temporal → sulcal;  lr (left–right) alone
  ```
  Each child is `leaky_relu(weighted parents) + 0.4 × its own noise`. A factor's **own part** is that noise: the
  variation its causes do not explain. Each view (T1, FLAIR) has its own style (gain, bias, noise level), drawn
  independently, so style is not shared content.
- **`bank_causal_nofield_fixedpos`, the main bank:** the multiplicative intensity field off, the lesion at a fixed
  site. The shared content is the 7 factors (|C| = 7). Matched width under the cosine is |C| + 1 = 8 units.
- **`bank_causal_nofield`, the drawn bank:** the same with the lesion's position drawn, so |C| = 10 (the factors and
  the lesion's x, y, z); matched width 11. Its position is a near-unique fingerprint that takes about half a narrow
  code (item 39), which is why the fixed bank became the main one.
- **`residual_bank.py`** adds each view's normative residual (the scan minus the training mean and its first 16
  principal components).
- **ADNI:** `adni_bank.py` writes paired ADNI scans in the same layout for `warm.py --real` (stage 68, Run:ai).

### The model and the frozen recipe

- **`warm.py BANK RUN cold STEPS EVERY [options]`** trains one encoder pair.
- **The frozen recipe** (every matched-width and wide run since stage 64):
  ```text
  --pool tail --t1key t1_04 --residual --jitter 2 --rotate 3 --patch-weight 0 --contrastive-loss-type infonce
  --predict-weight 1 --code-batches 500 --code-neighbours 31 --content-units N --seed S;  20,000 steps, evals every 2,000
  ```
  - Input: the scan and its normative residual as two channels. Separate T1 and FLAIR conv backbones give a 64-channel
    16³ map.
  - TailCode pooling (`--pool tail`): the map's average (GAP) plus rectified tails over the interior cells, 448
    features, a per-view BatchNorm gauge, then a head shared by the views; the code is its first N outputs.
  - InfoNCE (temperature .1, cosine on codes standardized per batch, the other view's codes as negatives).
  - Code batches: every 500 steps each batch is rebuilt as an anchor and 31 of its nearest subjects in the current
    code (no labels), so the loss pays for what the code does not yet separate.
  - Independent jitter: each view gets its own random 2-voxel / 3° move every step.
  - The prediction term: each view's code predicts the other view's whitened GAP (from a momentum copy) through a
    linear map.
  - Each of the residual, the jitter and the prediction term is needed; the tails keep a narrow code linearly
    readable; code batches beat uniform batches on everything (items 41, 42, 44).

### Holding all seven factors: the rule

- A code **holds** when, on the 1,000 test subjects and in both views, the own-parts spectrum's last value (the R² of
  the least-recovered direction of the seven own parts) and the ventricle–burden worst direction are each at least
  .40 (`holds.py`, `own_directions.py`, via `pool_check.py`). "Last value" near 0 means one own-part direction is lost.
- Raw factor R² misleads on the causal bank (a code can read burden .73 from its causes alone); read own parts.

### Tools in use

| tool | what it does |
|---|---|
| `causal_bundle.py RUN OUT.npz` | a run's test codes, latents and own parts (needed by every score below) |
| `pool_check.py export / detect` | exports codes and GAP (`causal/pool/RUN.npz`); `detect OUT.json LABEL=RUN...`: holds, last values, worst direction and ρ1 per run, by cell |
| `panel_check.py` | cross-seed completeness (stage 84 A), the chart-width rule (84 B), and stage 85's `panel` and `pipeline` |
| `chart_check.py` | the non-linear chart of a wide code (`--units U`, `--certificate`: held-out rebuild error with a bootstrap bound) |
| `continue_check.py`, `long_check.py` | trajectories of continued runs (stages 86, 81) |
| `style_audit.py OUT.json RUN...` | each view's gain, bias and noise read from the code against a null |
| `gap_content.py` | factors read from the code and from the backbone's GAP (stage 77) |
| `eval.causal.tabular_benchmark`, `block_tabular.py` | downstream: effect estimation (AIPW, linear and rbf learners) and causal discovery with tabular variables |

### Running things

- **Python and paths:** `/home/ng24/miniconda3/envs/monai_env/bin/python`, from the repo root, with
  `PYTHONPATH=out/burden_feasibility_20261102:out/regional_pilot_20261028/shim:.`. Tests: `python -m unittest
  tests.test_NAME` (no pytest). Results stay on the PC (`runs/`, `logs/`, `causal/`; `*.json` is git-ignored).
- **`run_pc.sh STAGE...`** runs numbered PC stages (`DRY=1` prints, `PARALLEL`, `PY`); finished arms are skipped.
  Newer stages keep their scripts in `causal/stageNN/` instead.
- **Run:ai.** Host `aicheadnode` (ssh, BatchMode). Checkout `/nfs/home/nglazman/crl-2/multiview-crl` at `3b83c41`,
  whose tracked `warm.py` is the one every A100 recipe run used. Untracked there: `runai_warm.sh` (this checkout's
  version: one run per call, `WARM` and `THREADS` variables) and job folders `stage77/`, `stage83/`, `stage85/`,
  `stage86/`.
  - Submit as `submit_stage85.sh` does: `runai training standard submit NAME --project nglazman --image
    aicregistry:5000/nglazman:multiview-crl --run-as-user --large-shm --node-type A100 --gpu-devices-request 1
    --cpu-core-request 16 --cpu-core-limit 32 --cpu-memory-request 64G --cpu-memory-limit 128G --host-path
    path=/nfs,mount=/nfs,readwrite --command -- bash -c "..."`. Check names are new on both machines first.
  - Timing: one 20,000-step run alone about 2 hours; five side by side in one job about 3.5–4 hours.
  - Copy finished runs back with `scp -r` after checking the log ends in `DONE`.
  - **Name hazard:** the cluster's `runs/` holds stage 74's 8- and 10-unit runs at seeds 42, 3, 4 under exactly the
    names of this PC's stage 64–65 runs. Never copy those back.
  - The same seed can hold on one GPU type and fail on another (stage 71a): compare arms on one GPU type.

## 4. Findings the paper rests on

1. **Matched width holds about half the time.** The frozen recipe at 8 units on the main bank holds in 3 of 5 seeds
   on the PC (stage 64) and 5 of 10 on A100s (seeds 6–15, stage 76 B); about half of 8- and 9-unit codes hold
   (item 51). Which seeds hold is a lottery, and the GPU type changes it (items 50, 51, 58, 61).
2. **What is lost, and to what.** A failing code loses one own-part direction outright, usually in the
   ventricle–burden plane, and spends that unit on a curve: a non-linear function of factors it already holds.
   Holders spend one unit on curves, failures two (items 46, 56, 58). Wider codes put every unit past the seventh into
   curves (joint functions of several factors, boosted R² about .75), never style or new content, and their intrinsic
   dimension is the factors' (TwoNN 6.36 for 16-unit codes against 6.37 for the true factors) (items 63, 64). No code
   carries style at any width (items 23, 63).
3. **No change to training makes 8 units reliable.** One bank, 5–10 seeds each: a faster code-batch rebuild (items
   49, 50), prediction weight 3 (49), a compressed 10-unit start (52), a harmonic penalty (54, worse), temperature 1
   (56, worse), a projector head at the code's width (58), a wider 16-output projector (69: 2 of 10), a sufficiency
   term on what the views share beyond the code (67: 0 of 10; it also degraded the backbone), and a fresh head on a
   frozen backbone (65: 0 of 3).
4. **Longer training helps slowly and not always.** 10,000 more steps repaired no seed (item 52). 40,000 more (three
   times the recipe) repaired seed 42, whose lost direction rose steadily from .01 to .44 and held at the last
   checkpoint, while seed 4's stayed at .00–.03 as its gap moved between burden and left–right (item 70). Stage 86
   continues both to 120,000 steps.
5. **ρ1, a failure detector at matched width, with limits.** ρ1 is the top held-out canonical correlation between the
   two views' backbone features once each is regressed on its own code (`pool_check.py detect`). At 8 units failures
   read higher than holders in 69 of 69 within-cell pairs, and the lowest ρ1 among three runs held whenever one did
   (items 59, 61). It fails at 9–10 units (7 of 10 pairs, item 60). It is measured against the run's own backbone,
   so training on it lowers it without fixing the code (items 65, 67): use it only to rank runs of one recipe.
6. **Identify wide, chart narrow works (stage 82, item 66).** A non-linear chart (an MLP autoencoder, shared by the
   views, of the wide code's whitened shared canonical directions, fitted on training subjects) brings a holding 9- or
   10-unit code to 8 units, or 7, without loss: it holds wherever its parent holds (8 of 8 at 8 units, 4 of 4 at 7),
   own parts within .05. The best linear compression never holds. Wider parents chart worse (12 units 2 of 3, 16
   units 0 of 2). The held-out rebuild error's worst direction is at most .017 (8 units) and .033 (7 units).
   Downstream, the charts match their parents: effect estimates with the rbf learner within .02, discovery of image–
   tabular edges F1 .88–.94 against .88–.96 (the true own parts .98).
7. **Label-free checks for both steps (stage 84, item 68).**
   - Which wide code is complete: score each run by how much of the other runs' shared content its code predicts
     (worst direction, against the other run's own other view as the ceiling). The complete code scores higher in 10
     of 10 pairs at 9–10 units (ρ1: 7) and 68 of 69 at 8; among any three runs the top score is complete in 194 of 196
     triples that contain a complete one. It needs a complete run among those compared.
   - The chart's width: the smallest width at which the chart rebuilds the wide code within .10 (worst direction, 95%
     bound) is 7 = |C| for every complete 10-unit code (6 units: .17–.31); the one incomplete code reads 6.
8. **What two views cannot give.** They identify the shared block, not its factors: the units are mixtures at every
   width, label-free discovery on them is at the mixing baseline, and the causal graph needs anchors (measured
   factors) (items 33–36, 40, 43). The paper's causal claims run on the block plus anchors.
9. **ADNI so far.** The 24-unit recipe trained on ADNI (stage 68). Its evaluated code dipped mid-training because
   evaluation read TailCode's gauge through stored running statistics; recomputed on clean training images, the dip
   is gone (stage 73). On the synthetic bank the stored statistics change one hold verdict of 55 (stage 78). The
   method of findings 6–7 has not run on ADNI.

## 5. Theory (draft: `paper/theory-matched-width-draft.tex`)

- Yao et al.'s result is two certificates. Alignment makes the code depend on content only, at any width. Maximum
  entropy at matched width makes it one-to-one, through a lemma that needs equal dimensions and an exact uniform
  density.
- With 2,000 subjects the entropy term sees uniformity only at the batch's resolution, so a curve of well-read factors
  can stand in for the weakest factor's dimension, and alignment prefers the curve: finding 2.
- The construction keeps alignment (in a slightly wide code, where curves get their own units) and replaces entropy
  by reconstruction. If the wide code is one-to-one and the chart has a left inverse on its image, the chart's code
  is one-to-one at width |C| (two-line proof). The rebuild error bounds, per direction, what the chart leaves
  undetermined, on held-out subjects.
- Not covered by the theory: that the wide code is complete (measured; checked by finding 7), and code batches and the
  linear prediction term (they only matter for the entropy step, which the construction does not use).
- Likely reviewer objections, and what answers them: one bank and few seeds (stage 85); dependence on code batches
  (stage 85, F10uni); a bespoke hold threshold (report threshold sweeps and standard metrics); no ADNI ground truth
  (anchors); a thin theorem (frame the contribution as diagnosis, a certified construction and a label-free width).

## 6. Running now (rules fixed before the runs; full text in the archive's section 9)

### Stage 85: the method on fresh seeds, a second bank, and without code batches

- **Arms** (Run:ai, five seeds side by side per A100 job; the checkout's tracked `warm.py`; submitted 27 Nov 10:08):
  - F10cb: main bank, 10 units, the frozen recipe, seeds 36–40 (`..._fixedpos_abl_codebatch_predict1_c10_20k_a100_sN`,
    labels `c10_acb_sN`);
  - F10uni: the same with `--uniform` in place of code batches, seeds 36–40 (`..._fixedpos_predict1_c10_20k_a100_sN`,
    `c10_auni_sN`);
  - D13cb: the drawn bank, 13 units (|C| + 3), the frozen recipe, seeds 46–55 (`..._causal_nofield_abl_codebatch_
    predict1_c13_20k_a100_sN`, `c13_dcb_sN`).
- **Scoring:** holds per wide code; cross-seed scores within each arm; every wide code charted at |C| − 1, |C| and
  |C| + 1 units with certificates; then, per triple of an arm's seeds, keep the top cross-seed score, chart it at its
  rule width (the smallest of the three whose worst-direction bound is at most .10, else the widest), and count a
  success if that chart holds.
- **Rules:** E1 (F10cb) works if at least 8 of the 10 triples succeed, fails at 5 or fewer. E2 (D13cb) works at 80%
  of the 120 triples or more, fails at 50% or less. E3: the wide code rests on code batches if F10uni holds in at
  least 2 fewer seeds than F10cb.
- **Expectation:** F10cb holds in 3–5 and E1 works; F10uni holds in 0–2 (rests on code batches); D13cb holds in 4–8,
  E2 unresolved or works, the width rule gives 10.
- **Files:** `causal/stage85/` (`job.sh`, `submit_stage85.sh`, `score_stage85.sh`, `wait_and_score.sh`). The watcher
  (`wait.log`) polls every 10 minutes and runs `FETCH=1 score_stage85.sh` (about four hours on the PC) when all
  twenty runs log `DONE`. Results: `pipeline.txt`, `detect.txt`, `panel.txt`, `chart_*.txt`, `style_audit.txt`.

### Stage 86: slow learning or a stable trade?

- **Runs:** stage 81's two codes continued from their 60,000-step models to 120,000 steps (same flags and seeds,
  `--init-run`, a checkpoint every 10,000), both in one A100 job (`burden-s86-cont`, submitted 27 Nov 10:36).
- **Rules:** seed 4 is repaired if it holds at some checkpoint and every later one; a stable trade if its last value
  stays under .10 in both views at every checkpoint; otherwise unresolved. Seed 42 is stable if it holds at every
  checkpoint; lost if it fails at 120,000. Seed 4 repaired means both failures were slow learning; a stable trade
  means at least one matched-width failure is not.
- **Expectation:** seed 4 a stable trade (its ventricle–burden direction crosses .40 by about 80,000, its gap does
  not close); seed 42 stable.
- **Files:** `causal/stage86/`, `continue_check.py`. Its watcher scores it after stage 85's scoring finishes.

### Not scored: stage 79 (the other session's)

- Stage 74's width sweep repeated on the drawn bank (do the label-free width rules land on 11?). Its three Run:ai jobs
  (`burden-s79-width-*`) have completed; rules in the archive's section 9.

## 7. Next steps (proposed)

1. Read stages 85 and 86 against their rules; write them as findings here.
2. If E1 and E2 work, the method stands on two banks: write the experiments section (hold rates of the 8-unit recipe
   against the pipeline, the chart table, the cross-seed pick, the width rule). If E3 says the method rests on code
   batches, say so and present code batches as conditioned negatives.
3. ADNI: train three wide runs, pick by the cross-seed score, chart at the width the rebuild error gives, and check the
   code against FreeSurfer anchors. Rules first; the gauge recomputed before any export (stage 73).
4. The paper: resolve the theory draft's `\todo` notes; threshold sweeps and standard metrics for the hold rule.
5. Housekeeping: score stage 79 (the other session's); merge the two branches and commit when the user asks.

## 8. Recent stages at a glance

| stage | question | verdict | item |
|---|---|---|---|
| 64 | 8, 9, 10 units at five seeds | about half hold at 8 and 9 units | 51 |
| 65 | compressed 10-unit start; 10,000 more steps | no more holds | 52 |
| 66, 67 | label-free signals to choose runs (top-1; measured anchors) | top-1 passes narrowly; anchors choose well | 53, 55 |
| 68, 73 | the recipe on ADNI; why its evaluated code dipped | the gauge's stored statistics | archive §10 |
| 60, 70, 71, 71a | harmonic penalty; temperature; projector head | none helps; two hurt | 54, 56, 58 |
| 72 | a suggested local loss | would not target the lost factor | 57 |
| 74 | width sweep, fixed bank | spare units are curves; the loss's elbow overshoots | 63, 64 |
| 75, 76 | ρ1 as a detector; merging runs | ρ1 works at 8 units, not at 9–10; merging fails | 59–61 |
| 77 | a sufficiency term | 0 of 10, worse | 67 |
| 78 | stored versus recomputed gauge on synthetic codes | one verdict of 55 moves | archive §9 |
| 80 | a fresh head on a frozen failing backbone | 0 of 3 | 65 |
| 81 | 40,000 more steps on two failures | one repaired, slowly | 70 |
| 82 | identify wide, chart narrow | works (8 of 8; 7 units 4 of 4) | 66 |
| 83 | a 16-output projector | 2 of 10 | 69 |
| 84 | label-free completeness; chart width | both work | 68 |
| 85, 86 | the method on new seeds and a second bank; 120,000 steps | running | – |

## 9. Tried and dropped (do not retry without a new reason)

- For burden's precision on the drawn bank (items 1–32): more negatives (`--queue`), a running gauge, a Barlow Twins
  term on the code, group dropout, hard-negative weighting, the patch loss under jitter, wider heads, remainder units,
  critic weights, tail or pooled-vector prediction targets, a sharper 32-component residual for narrow codes.
- Lesion-specific tools (a lesion-blind third view, lesion filters): rejected by the user, who wants generic levers.
- More subjects: not available on ADNI.
- Two views as proxies for each other (proximal two-stage least squares, item 37); own parts by variance
  heterogeneity across age or sex (item 38); the theory's bridge (stage 46b, item 38): none removes the gap.
- At 8 units: every lever in finding 3.
