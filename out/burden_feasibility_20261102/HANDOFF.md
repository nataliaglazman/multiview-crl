# Burden feasibility: handoff (10 Nov 2026)

Everything for this line of work lives in this folder. `run_pc.sh` runs it in numbered stages, and its header
describes every stage. This file says what the work is for, how it is run, what has been found, what is still open
and what to do next. Repo-wide conventions are in `CLAUDE.md` at the root. They apply to every agent.

## 1. The goal

- **The model.** The encoder-only multiview contrastive model on paired synthetic T1/FLAIR brain MRI.
  - Each view has its own backbone.
  - A shared head turns the pooled features into a 14-unit content code.
  - InfoNCE aligns the two views' codes.
- **The target.** Get the burden (total volume) of one small white-matter lesion into the code, with no labels and
  without losing the anatomy factors.
- **The hard case.** T1 lesion contrast 0.4: faint in T1, bright in FLAIR.
- **Constraints set by the user.** These are not negotiable:
  - **Generic levers only.** A method must find any hard feature, not just the lesion. Lesion-specific tools (blob
    filters, lesion branches, views built around the lesion) are diagnostics at most. A lesion-blind third view was
    built and then rejected for this reason ("i want it to be lesion-agnostic").
  - **One shared route.** Every factor is read through one shared pooling route, never per-factor heads.
  - **Real-data levers only.** The work has to transfer to ADNI, where the subject count is fixed. "More subjects"
    is a diagnostic, never the proposal. Augmentation-type levers are fine.

## 2. How we work with the user

- **The user runs everything on their PC and pastes results back.**
  - The PC is an RTX 3090 at `ng24@sie114-u-pc`, with the repo at `~/projects/multiview-crl` and conda env
    `monai_env`. The repo's `.venv` has also been used there.
  - Never ssh or connect to remote hosts without asking first.
- **Our job is to prepare a stage, check it locally, and hand over commands.**
  - Add the stage to `run_pc.sh` with any code it needs, smoke-test it on a small local bank, and dry-run it.
  - Then give the user one commit command, one pull and one run command. Analyse what they paste back.
- **Commits.** Commit or push only when the user asks. Normally you give the command and the user runs it.
  - Name files explicitly in `git add`. The working tree holds the user's other uncommitted work, which must never
    go into these commits: `eval/protocol/score_checkpoint.py`, `models/multiview_encoder.py`,
    `models/residual_heads.py`, `training/main_conv_synthetic.py`, `training/encoder_args.py`, several tests, ….
  - End commit messages with `Co-Authored-By:` your attribution line.
- **Never `git pull` on the PC while `run_pc.sh` is running.** Bash reads the script as it goes, so changing it under
  a running stage breaks that stage.
- **Writing style.**
  - Define every term at first use.
  - Plain words, short sentences.
  - Succinct: the user has asked to drop unnecessary comments.
  - Quote numbers as T1 / FLAIR.

## 3. Running things

### On the PC

```bash
PARALLEL=3 nohup bash out/burden_feasibility_20261102/run_pc.sh 20 > out/burden_feasibility_20261102/run_pc_20.log 2>&1 &
```

- **Stages.** Pass stage numbers as arguments. With none, the default stages in `run_pc.sh` run.
- **Skipping.** Finished arms (their log ends in `TEST` or `DONE`) are skipped. So are arms still running outside the
  script. Banks and residuals with a done-marker are not rebuilt. A rerun after a crash therefore continues.
- **Environment variables:**
  - `DRY=1` prints the commands without running them.
  - `PY` sets the interpreter (default `python`).
  - `PARALLEL` (default 3), `BATCH`, `QUEUE`, `BANK_WORKERS`.
- **Per-call variables:**
  - `EVERY=N` before `ssl` sets that arm's eval interval.
  - `TAG`/`FEATURES` before `causal` set the report prefix and the exported features.
- **Helpers:**
  - `bank`, `residual_bank`.
  - `arm`, `ssl` (warm.py arms), `sup` (labelled arms), `finish`.
  - `audit`, `causal`, `own_variation`, `cca_readout`, `cca_discovery`.
- **One-off tools** need `PYTHONPATH=out/burden_feasibility_20261102:out/regional_pilot_20261028/shim:.` and must be
  run from the repo root.
- **Where results live.** Results stay on the PC: `runs/`, `logs/`, `causal/`, `audit.json`/`audit.txt`,
  `summary.txt`. `*.json` is git-ignored.

### Locally (Mac)

- **Python:** `/opt/miniconda3/envs/adni-analysis/bin/python` with
  `PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=out/burden_feasibility_20261102:out/regional_pilot_20261028/shim:.`.
  There is no pytest: run test files by path.
- **Lint.**
  - black: `~/.cache/pre-commit/repo2fpit8sz/py_env-python3/bin/black --line-length 120`.
  - isort: `~/.cache/pre-commit/repolaqflkce/py_env-python3/bin/isort`.
  - pyflakes: `~/.cache/pre-commit/repomjdnwzxs/py_env-python3/bin/python -m pyflakes`.
- **Small local bank for smoke tests.** It renders in about a minute with 8 workers:
  ```bash
  python bank.py SCRATCH/bank_causal --t1-values 0.4 --causal --sizes 208 32 160
  python residual_bank.py SCRATCH/bank_causal t1_04 flair
  ```
  - `residual_bank.py` needs at least 200 training subjects.
  - 2-step `warm.py` runs on it (`--test-subjects 160`) check any new option.
  - Check that a default run's `report.json` history is unchanged by your edits.
- **Shell.** zsh needs `${=VAR}` to word-split. macOS has no `timeout`.

## 4. Data

- **`bank.py`** renders synthetic subjects into memory-mapped arrays: 64³ volumes and a 2,000 / 400 / 1,000
  train / val / test split. Its settings come from `reference/flags.txt`.
  - Arrays per split: T1 at each lesion value (`t1_00`, `t1_04`), `flair`, the content latents `zc` (9),
    the lesion volume `vox`, and the 16³ lesion load `cells`.
  - The 7 rendered factors are brain size, ventricle, burden, thickness, temporal atrophy, left–right asymmetry
    (`lr`) and sulcal. `zc[3:5]` are unused with the burden target.
- **Banks on the PC:**
  - `bank`: independent factors, T1 at 0.0 and 0.4.
  - `bank_nuisance`: the shared nuisance deformations switched on.
  - `bank_causal` (`--causal`): the factors follow a known causal graph (an SCM, structural causal model). Each
    child is `leaky_relu(weighted parents) + 0.4 × its own noise`. The structure and weights are in
    `bank.causal_scm`.
    ```text
    brain → ventricle → burden → temporal ← thickness ← brain;  temporal → sulcal;  lr alone
    ```
    - Ventricle explains about 70% of burden's variance (corr .84), so burden's own variation has sd .43 of a
      total of .76.
    - The burden → temporal edge is weak (weight .2).
- **`residual_bank.py`** writes each view's normative residual: the scan z-scored over its brain, minus the
  training mean, minus its first 16 principal components, scaled to unit SD. Each modality uses only its own basis.

## 5. The model and the current recipe

- **`warm.py`** trains one arm. Usage: `warm.py BANK RUN cold STEPS EVERY [options]`.
- **Input.** The scan plus its residual, as two channels (`--residual`). The residual alone would break the
  theory's injectivity assumption.
- **Backbones.** Separate T1 and FLAIR conv encoders give a 64-channel 16³ map.
- **TailCode pooling** (`--pool tail`):
  - The GAP over all 4,096 cells, plus "tails" over the 321 interior cells.
  - Tails are `log1p(100 · mean ReLU(±z − t))` at t = 2, 3, 4, where z uses running per-cell, per-view statistics.
  - That gives 448 features, then a per-view BatchNorm gauge, then a shared head 448 → 100 → 18. The code is the
    first 14 outputs.
- **Loss.** InfoNCE: τ 0.1, cosine on codes standardized per batch, the other view's codes as negatives, both
  directions summed.
- **Current recipe:**
  ```text
  --pool tail --t1key t1_04 --residual --code-batches 500 --jitter 2 --rotate 3 --patch-weight 0
  ```
  - Code batches: every 500 steps each batch is rebuilt as an anchor plus 31 of its 50 nearest subjects in the
    model's own code. No labels are used.
  - Independent per-view jitter: each view gets its own random 2-voxel / 3° rigid move every step.
  - Patch loss off.
- **Other options:**
  - `--code-neighbours N`: the neighbourhood size; N = 31 looks better (stage 18).
  - `--content-units N`: a wider code.
  - `--pool regional_tail` replaces the global average with eight fixed regional averages and keeps the tails.
  - `--contrastive-loss-type barlow_twins`: use cross-view correlation as the global loss. Set `--patch-weight 0`.
  - `--half-views xyz --half-units K`: the lesion-agnostic partial-observability arm (stage 19).
  - `--remainder-units K` (with `--remainder-neighbours`, `--remainder-every`, `--remainder-weight`): K more code units
    trained only by InfoNCE on batches of each subject's nearest neighbours on the main units (stage 23).
  - `--hard-negatives G`, `--seed S`.
  - Older levers: `--queue`, `--gauge running`, `--group-dropout`, `--decorrelate`, `--shared-jitter`, `--uniform`.
  - Each option refuses combinations it can't handle.
- **`sup.py`** trains the same backbone with lesion labels. It was used early as a ceiling.

## 6. Evaluation tools and how to read them

| tool | what it reports |
|---|---|
| `summarize.py` | each arm's training history, compact enough to paste |
| `code_audit.py` | factor R² from the code and backbone GAP (ridge, boosted), lesion centre x/y/z, the same rows for the untrained twin (the floor), canonical correlations, direct correlations between matching code units, effective rank (how many distinct directions the code uses), and retrieval on train and test subjects |
| `own_variation.py` | causal bank only: each factor's **own variation** (its SCM noise) read from the code, the pooled features and the GAP, trained vs untrained twin, with reference columns |
| `precision_audit.py` | the lesion-volume reading error in voxels at each stage (image filter on the residual, backbone map, GAP, pooled features, code), with the anatomy part handed to the readout; error by lesion size, error shared by the views, and the loss's reward for a sharper reading, also with lesion position and/or anatomy removed from the code |
| `cca_readout.py` | CCA codes (top 14 or 24 canonical directions between T1 and FLAIR features, fitted on training subjects) from the pooled features and GAP, scored the same way; writes bundles |
| `causal_bundle.py` + `causal()` | label-free (`eval.causal.latent_causal_discovery`) and decoded (`eval.protocol.compare_bundles --with-graph`) discovery against the true graph |
| `remainder_price.py`, `batch_price.py` | InfoNCE's reward for adding a unit, priced on true latents with no training |

**Reading rules learned the hard way:**
- **Raw burden R² is uninformative on the causal bank.** A code that knows every other factor but not the lesion
  scores .73.
  - Use own variation instead. Its "no own" reference for burden is .01.
  - The straight-line partial R² (the factor minus a linear fit on its parents) can be faked by reading a parent
    non-linearly: .12 for burden from its parents alone.
- **Seed noise.** About .10 on a single factor at 6k. Burden's own variation in the recipe at 6k was .10 (seed 42)
  and .26 (seed 2).
- **Read memorisation from the train–test top-1 gap.** Top-1 is compressed near 1.
- **Label-free discovery:** compare F1 against the mixing null (random mixtures of the units), not the naming null.
- **Decoded discovery is noisy** at 300 rows; read the resampled F1.
  - compare_bundles warns when widths differ. Don't pass `--equal-width` for a wider code: it would cut the code
    down to the narrower width.

## 7. Findings, in order

1. **Pooling (stages 1–4).**
   - A plain mean over the map can't read one small lesion: leakage over ~1,500 cells swamps it.
   - Rectified tails can. With labels: T1 .95, FLAIR .84.
   - Label-free tails at T1 0.0 give .51 / .44; at T1 0.4 they give 0.
2. **Residual input (stages 5–6).**
   - At T1 0.4 the lesion is nearly invisible in the raw scan (fixed-filter R² .10) and clear in the residual (.94).
   - Residual + tails, label-free: .55 / .55, but anatomy is lost.
   - With the nuisance fields on, burden reads 0.
3. **What doesn't fix it (stages 7–9).**
   - The gauge isn't the factor.
   - A queue keeps burden (.72) but anatomy stays near 0.
   - Code batches keep burden (.56) with some anatomy.
   - The residual arms memorise the 2,000 training subjects: training retrieval ~1.00 against .58–.81 on new ones.
4. **Jitter (stages 10–12).**
   - Independent per-view jitter of 2 voxels or more removes the memorisation. It gives burden .73 / .76 with most
     anatomy (ventricle .81, lr .88).
   - 1 voxel is too little. Patch loss off is better.
   - Mechanism: matching fine voxel-aligned detail stops working, so the code has to use features that survive a
     small move.
5. **Long training (stage 12).**
   - At 20k, temporal arrives (.74), but the code's burden falls from .73 to .46 while the pooled features keep
     .81 / .83.
   - The 14-unit code is full.
6. **Causal bank (stages 15–16).**
   - Label-free discovery: F1 .53 against .45 ± .16 for random mixtures. The units are an arbitrary rotation of the
     content block, as the theory predicts for two views that share all content.
   - Decoded discovery from the 14-unit code is no better than an untrained network (.54–.59 vs .60).
   - From the backbone GAP it matches PC on the true factors (.84 vs .85).
   - Own variation: the backbone holds burden's ~.3, the code .10 at 6k falling to .04 at 20k.
   - Why the code drops it:
     - InfoNCE only pays for information that helps tell subjects apart beyond what the code holds.
     - Ventricle predicts ~70% of burden, so only a small, faint remainder is new.
7. **Harder negatives (stage 17).**
   - 31-neighbour batches raise every child's own variation at 6k.
   - Hard-negative weighting starves the broad factors (lr .89 → .56). Dropped.
8. **31 neighbours at 20k (stage 18, seed 42).**
   - The code's burden own variation is .19 / .18 against .04 / .06.
   - At 6k the seed-2 pair replicates the gains for thickness, ventricle and temporal, not for burden.
9. **CCA readout (stage 19).**
   - Top 24 CCA directions on the backbone GAP, fitted on training subjects, no retraining.
   - Holds burden's own variation at ~.3 on every backbone (recipe .29 / .33, half views .29 / .36,
     31 neighbours .33 / .37, 24 units .32 / .29), and beats the InfoNCE code on every factor.
   - Decoded discovery on it matches the full GAP (.84 / .81 at 6k).
   - At 14 units, CCA loses burden too: the shared block is wider than 14.
10. **Half views (stage 19).**
    - The lesion-agnostic partial-observability arm: the first 8 units are also aligned between FLAIR and a T1
      masked to a random half of the brain.
    - It gives the best backbone so far (thickness .65 / .60 in the GAP), but the code's burden is 0.
11. **A 24-unit InfoNCE code (stage 16, scored 10 Nov).**
    - Burden .09 / .10 only.
    - The extra units went to the lesion's **position** (code x / y / z .60 / .63 / .70), a near-unique fingerprint
      (all-1,000 test top-1 .947), and to anatomy remainders (thickness .51 vs .20).
    - Loss .035: the task is solved.
    - **InfoNCE ranks shared content by identification value, and burden's remainder comes last. CCA ranks by
      cross-view agreement, where it is within the top 24.**
12. **48 units (run by hand, 10 Nov).** Width bought anatomy (brain .18 → .80 in the uniform code) and lesion position
    (x / y / z .65 / .92 / .86), not burden's own part (.24 / .27 against the backbone's .31 / .36).
13. **Precision (10 Nov, `precision_audit.py`).**
    - The backbone reads lesion volume to ~20 voxels in every run on both banks; the image allows ~10. Burden's own
      part spreads ±20 voxels on the causal bank, so the backbone recovers about half of it.
    - At ~20 voxels the reading agrees across views at r ~.5, and the loss charges a unit that agrees less than the
      code's others. It pays only when the negatives already match lesion position and anatomy: stage 23's set-up.
    - Lesion position alone is not the blocker.
    - Big lesions are read worst by the network, but not by the image filter.
14. **Remainder units (stage 23, read 11 Nov).**
    - Conditional negatives teach the remainder units whatever the main units lack.
    - With 14 main units on uniform batches, that was brain size, not burden.
    - The code's anatomy became the best on the causal bank, but burden's own part fell to .19 / .17. The backbone
      stayed at ~20 voxels.

## 8. Theory status (Yao et al. 2024; von Kügelgen et al. 2021)

- **Inside the theory:** scan + residual input, separate encoders, independent jitter (a view-specific pose
  latent), patch loss off, the InfoNCE objective.
- **Not covered:** code batches (any neighbourhood rule), hard negatives.
- **Silent:** finite subjects (memorisation), and code width versus the size of the shared block. The block is wider
  than 24 here once the lesion's position is counted.
- **The CCA readout is not covered as used.**
  - It is fitted after training, it is linear, and it keeps weakly shared directions. Directions 15–24 agree at only
    r .45–.72, so part of each is view-specific (style or noise, unmeasured).
  - Use it as a diagnostic of how wide the shared block is.
  - The theory-backed counterpart is latent correlation maximisation trained end to end (Lyu et al., ICLR 2022): align
    the two views' codes under a whitening constraint, which identifies the shared block. Barlow Twins is a close
    relative.
- **Label-free factor-level discovery** needs views that share different subsets of factors, or interventions.
  Two views that share everything identify only the block.

## 9. State on 11 Nov

- **Stage 23 was read on 11 Nov.**
  - The remainder units learned brain size and anatomy, not burden.
  - The whole code has the best anatomy on the causal bank. Burden's own part is .19 / .17, and the backbone's
    precision is unchanged.
  - See section 10, item 3.
- **New since 10 Nov:** the 48-unit run (section 10, item 4) and the precision audit (item 5).
  - The backbone reads lesion volume to ~20 voxels everywhere.
  - At that precision the loss charges a burden unit unless the negatives already match the code's anatomy and lesion
    position.
- Stage 20 is in commit `c2c3f55`. Its seed-2 results have yet to be reported.
- **Results that exist on the PC but were never reported:**
  - Stage 13: shared-jitter control, 4-voxel seed, 4-voxel 20k. The shared-jitter arm is confounded: a pose shared by
    both views becomes content InfoNCE can match on.
  - Stage 14: 24 units on the independent bank, nuisance-on arms.
  - The half-view run's audit, i.e. whether its regional units hold the lesion's position. See `summary.txt` and
    `audit.txt`.
- **Best readout so far:** CCA on the backbone GAP (24 directions); the backbones that read best this way are the
  24-unit run and half views. The user does not want CCA after training as the method, so it stays a diagnostic of
  what the backbone holds.
- **Stage 21:** the user reports that Barlow Twins did not help (numbers not recorded here).
- `code_audit.py` prints an untrained-twin floor under every factor table, lesion x/y/z included.
- Stage 25 compares regional pooling with stage 21's 24-unit InfoNCE control on the same subjects, seed and jitter.
  It scores the code that enters InfoNCE. In `causal/stage25/`, compare volume error and agreement between views in
  `precision_audit.txt`, burden's own part in `own_variation.txt`, and position and anatomy in `code_audit.txt`.
  The baseline is about 20 voxels of error in the backbone and 22–23 in the code.

## 10. Recent stages and next steps

1. **Read stage 20:** does the 31-neighbour burden at 20k (.19 / .18) replicate with seed 2?
2. **Stage 21 compares Barlow Twins with InfoNCE** at 24 units and 20,000 steps on `bank_causal`. Both use the
   same seed (42), uniform 32-subject batches, scan and residual input, tail pooling, independent 2-voxel / 3°
   jitter and no patch loss. Barlow Twins uses the repo's default redundancy weight of 0.005 and per-batch
   correlations. This is a test of the loss on the fixed subject set.
   - `causal/stage21/own_variation.txt`: compare held-out burden and anatomy from the same exported code.
   - `causal/stage21/code_audit.txt`: check whether matching units agree across views and how many distinct
     directions the code uses.
   - `causal/stage21/cca_readout.txt`: training-fit CCA benchmark for both backbones.
3. **Stage 23: remainder units with conditional negatives (run 10 Nov, read 11 Nov). It brought anatomy back, not
   burden: the remainder units learned brain size, because the main units had dropped it. The success test below is
   not met.**
   - **Result** (`causal/stage23/`, `causal/precision/precision.txt`):
     - The 14 main units (uniform batches) kept lesion position (y / z .83 / .78), lr and temporal, but dropped brain
       size (own part .17), as the uniform 24-unit control did (.18).
     - The remainder neighbourhoods are built on the main units, so they matched lesion position but not brain size.
       The remainder units therefore learned brain size (.82), ventricle and thickness.
     - Burden's own part: main units .08, remainder units .05, whole code .19 / .17. The control has .28 / .32 and
       31 neighbours .19 / .18.
     - Precision audit: code error 31 / 34 voxels (control 23 / 22). The backbone is unchanged at 19–22.
     - Whole code: the best anatomy on the causal bank. Own parts: brain .88, ventricle .66 / .61, temporal
       .75 / .77, lr .85 / .86 (control .18, .51, .15, .52).
     - Little memorisation: code-32 top-1 .999 on training subjects, .972 on test subjects.
     - This fits the pricing. With only position matched, a 20-voxel burden reading still costs (−.015 on the run's
       own neighbour sets); with anatomy matched too, it pays (+.15).
   - **Next: stage 24 (built 11 Nov, not run):** the same arm with the remainder neighbourhoods built on both views'
     backbone GAP (`warm.py --remainder-space gap`) instead of the main units. Label-free.
     - Run: `ssl_uni_jitter_nopatch_restail_t104_causal_rem10gap_20k`.
     - Reports: own_variation, the audit and the precision audit against stage 23 and the control, in
       `causal/stage24/`.
     - The GAP holds anatomy and lesion position, so neighbours match on both.
     - Risk: the GAP also holds ~.3 of burden's own part, plus the anatomy remainders, which would compete.
     - Expected cost: brain size may leave the code, since the main units drop it and the remainder units can no
       longer use it. If stage 24 works, add a third block: main units, then remainder units on the main units, then a
       block on GAP neighbourhoods.
     - Smoke check: the default path (`--remainder-space main`) repeats stage 23's 50-step smoke run exactly.
   - **Why:** once a batch is told apart, InfoNCE stops paying for new shared information. The pricing in item 5 adds
     that a burden reading at today's ~20-voxel precision pays only when the negatives already match the code's lesion
     position and anatomy, and the remainder units are scored only on such negatives.
   - **Arm:** the recipe's 14 units on uniform batches, plus 10 units trained only by InfoNCE on batches of each
     subject's 31 nearest neighbours on the 14 (rebuilt every 500 steps). They gain only from shared content the 14
     lack: Sordoni et al. 2021's decomposed estimator, or MCL within one model. 20,000 steps on `bank_causal`, seed 42.
   - **Control:** stage 21's 24-unit InfoNCE arm: same width, head shape, init (the step-0 evals are identical),
     batches and jitter. **Reference:** the 31-neighbour code batches (stage 18).
   - **Success (revised 11 Nov):**
     - The remainder units hold burden's own part near the backbone's level, the main units keep the anatomy, and
       train–test retrieval shows no new memorisation.
     - The real win would be the backbone's reading error falling below ~20 voxels in `precision_audit.py`. No run so
       far has.
     - Beating the control on the whole code is not the test: the control's code already reads burden to 22–23
       voxels, close to its backbone's ~20.
4. **48-unit InfoNCE, run by hand 10 Nov** (`ssl_uni_jitter_nopatch_restail_t104_causal_c48_20k`: stage 21's
   InfoNCE arm with `--content-units 48`; reports `causal/stage21/{own_variation,code_audit}_c48.txt`). Width bought
   anatomy, not burden. The backbone is within ~.02 of the 24-unit run's and is now burden's ceiling.
   - Code, T1 / FLAIR, 24 → 48 units: brain .18 / .44 → .80 / .81, ventricle .24 / .34 → .78 / .79, thickness
     .18 / .29 → .74 / .74, lr .52 / .56 → .73 / .74; burden beyond ventricle (partial R²) .31 / .35 → .25 / .29.
   - Burden's own part: code .24 / .27, head input .32 / .34, backbone GAP .31 / .36, untrained twin .01 / .10.
   - Lesion x / y / z from the code .65 / .92 / .86 (T1), floor ~0. The two most-shared directions are the lesion.
   - Still missing: thickness's and temporal's own parts (.18 and .22, at their no-own references), which code
     batches keep. Next arm: the same with `--code-batches 500 --code-neighbours 31` in place of `--uniform`.
5. **Why burden's own part stops where it does: the precision audit (10 Nov).**
   - **Tool:** `precision_audit.py OUT_JSON RUN...` gives the lesion-volume reading error in voxels at each stage:
     the fixed blob filter on the normative residual, the backbone's map read position-free, GAP, pooled features,
     code.
     - Each readout is handed the volume the other true latents predict, so only the lesion reading is scored,
       against the untrained twin.
     - It splits the error by lesion size (thirds of the anatomy-predicted volume) and measures the error the views
       share.
     - It prices a sharper reading in the run's loss, on the code as trained and with lesion position, anatomy or
       both removed.
     - Readings with a known error of 2–30 voxels come back within ~10%. Results are in
       `causal/precision/precision.txt` on the PC.
   - **The target is small.** The rendered volume is an exact function of the burden latent, so a perfect reader
     would recover the own part. On the causal bank the other factors explain ~.58 of the volume's variance, leaving
     an own part of ~20 voxels (SD) out of ~86. On the independent bank it is the whole ±41 voxels.
   - **The backbone reads volume to ~19–22 voxels** in every run on both banks, at 6k and 20k. The fixed filter on the
     residual reads it to 8–12, so half the available precision is lost inside the backbone. The same error
     recovers ~.80 of the own part on the independent bank but ~.47–.52 on the causal bank. own_variation.py's
     latent target, read from the features alone, gives ~.3 for the GAP.
   - **The head keeps or loses it by batch rule.** Code error:
     - uniform 24 / 48 units 22–25 voxels;
     - 31 neighbours 34–37;
     - the 50-neighbour recipe 46–71;
     - independent bank 23–25 at 6k, 38–42 at 20k.
   - **Pricing: the blocker is cross-view agreement, not lesion position.**
     - On the causal bank, a burden unit read to 20 voxels earns ≤0 in every code and batch rule. It costs −.04 to
       −.08 nats in the code-batch codes and ~0 in the saturated uniform codes.
     - The reward crosses zero at ~12–17 voxels, where the reading's cross-view r is ~.6–.75; at 20 voxels it is .5.
       A unit that agrees less than the code's others costs more than it adds, saturated or not. On the independent
       bank 20 voxels gives r .81 and pays everywhere.
     - Removing lesion position (Chen et al. 2021's "easy feature") does not help. e20 on uniform sets goes:
       recipe −.09 → −.15, 31 neighbours −.11 → −.21, 24 units −.013 → −.027, 48 units −.002 → −.004. The removal
       also strips ~40% of the code's anatomy, because lesion placement depends on anatomy.
     - A 20-voxel reading pays only with position AND anatomy removed AND neighbour sets (+.13 recipe,
       +.17 31 neighbours): stage 23's situation.
     - The uniform 24 / 48-unit codes are indifferent (every reward within ±.03). They tell subjects apart even with
       position and anatomy removed (loss .64 / .30), so they hold other shared content the factors don't explain.
     - Sharpening always pays (20→10 positive in every row). But at 20 voxels no unit is paid to carry burden, so no
       gradient reaches the backbone's precision. This fits 20k being no sharper than 6k.
   - **Other leads:**
     - The largest third of lesions (mean 116 voxels) is read at 27–50 voxels by every network stage but at 8–12 by
       the image filter, so the network compresses big lesions. This is only visible on the causal bank.
     - The code's burden errors are shared across views (r .36–.73, backbone .18–.35).
6. **Later:** check the nuisance-on runs before porting to ADNI (`training/main_conv_synthetic.py`): residual input,
   tail pooling, per-view jitter with the patch loss off, code batches, and the CCA readout.
   - Score white-matter hyperintensity (WMH) burden by its own part: the residual after a flexible fit on ventricle
     volume and age.
   - Otherwise atrophy passes for WMH.

## 11. Tried and dropped

| lever | why it was dropped |
|---|---|
| more negatives (`--queue`) | memorised the most; anatomy near 0 |
| running gauge | about 2–2.5× slower, no gain |
| Barlow Twins term on the code (`--decorrelate` 0.1) | no effect |
| group dropout | traded burden for temporal |
| hard-negative weighting | starved lr and temporal |
| patch loss under jitter | can't train once the views are misaligned; off is better |
| lesion-blind third view | lesion-specific; rejected by the user |
| more subjects | not available on ADNI |
