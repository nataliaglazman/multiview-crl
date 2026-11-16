# Burden feasibility: handoff (13 Nov 2026)

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
  - `bank_causal_nofield` (`--causal --field off`, stage 28): the causal bank's subjects, lesions, styles and noise
    draws, with the renderer's multiplicative field off (section 7, items 17–18).
  - `bank_causal_nofield_nogain` (`--causal --field off --gain off`, stage 29): the same, with every scan rendered at
    gain 1, so tissue and lesion contrast no longer vary from scan to scan. The bias and the noise level still vary.
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
  - `--critic-weights main|remainder|both` (with `--critic-lr`, default 1e-3): one learnable weight per code unit
    inside that term's InfoNCE cosine, starting at 1, so a unit whose views agree poorly can count for less (stage 31).
  - `--predict-weight W`: each view's code predicts the other view's whitened GAP from a momentum copy (stage 33).
    `--predict-hidden H` adds a one-hidden-layer MLP of width H to each linear map, starting at zero (stage 36).
  - `--hard-negatives G`, `--seed S`.
  - Older levers: `--queue`, `--gauge running`, `--group-dropout`, `--decorrelate`, `--shared-jitter`, `--uniform`.
  - Each option refuses combinations it can't handle.
- **`sup.py`** trains the same backbone with lesion labels. It was used early as a ceiling.

## 6. Evaluation tools and how to read them

| tool | what it reports |
|---|---|
| `summarize.py` | each arm's training history, compact enough to paste |
| `code_audit.py` | factor R² from the code and backbone GAP (ridge, boosted), lesion centre x/y/z, the same rows for the untrained twin (the floor), canonical correlations, direct correlations between matching code units, effective rank (how many distinct directions the code uses), and retrieval on train and test subjects |
| `own_variation.py` | causal bank only: each factor's **own variation** (its SCM noise) read from the code, the pooled features and the GAP, trained vs untrained twin, with reference columns. `--boosted` adds the code read by gradient boosting beside the true factors read the same way: ridge reads only what a code holds linearly |
| `bottleneck_check.py` | no training: fresh 10- and 24-unit bottlenecks fitted on a run's frozen features to predict the other view's whitened GAP through a linear or an MLP map, read like a code (volume error, alone, anatomy and position by ridge and boosting). Stage 36 |
| `style_audit.py` | each view's gain, bias and noise level read from the code, pooled features and GAP, trained vs untrained twin, against the other view's (drawn independently: the null). Banks don't store style, so it is redrawn from the generator's seeds and must reproduce the stored content latents. `code_audit.py` scores no style |
| `precision_audit.py` | the lesion-volume reading error in voxels at each stage (image filter on the residual, backbone map, GAP, pooled features, code), with the anatomy part handed to the readout; error by lesion size, error shared by the views, and the loss's reward for a sharper reading, also with lesion position and/or anatomy removed from the code. `--jitter-draws K` reads test scans moved by the run's training jitter (mean over K moves), `--split train` reads training subjects, and `precision_compare.py` puts conditions side by side. Two "weighted" price rows refit the critic to each code (one weight per unit inside the cosine; as trained, and with position and anatomy removed); rerunning the audit on a stored JSON adds them without redoing the rest |
| `cca_readout.py` | CCA codes (top 14 or 24 canonical directions between T1 and FLAIR features, fitted on training subjects) from the pooled features and GAP, scored the same way; writes bundles |
| `causal_bundle.py` + `causal()` | label-free (`eval.causal.latent_causal_discovery`) and decoded (`eval.protocol.compare_bundles --with-graph`) discovery against the true graph |
| `remainder_price.py`, `batch_price.py` | InfoNCE's reward for adding a unit, priced on true latents with no training |
| `eval.causal.tabular_benchmark` | synthetic tabular tasks on the bank's test subjects: the share of image-driven confounding a vector removes (AIPW), and PC's recovery of edges out of the image factors (MRI→tab) and into them (tab→MRI), with orientation. Takes `causal_bundle.py` bundles, which carry each factor's own noise (`z_own`). Section 7, item 24 |

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
15. **Supervised ceiling (stage 26, read 11 Nov).** The recipe network trained on lesion-volume labels instead of
    InfoNCE (`warm.py --supervise volume`, a diagnostic, 10k steps), against stage 21's 24-unit InfoNCE control.
    Reading error in voxels, InfoNCE → labels (`causal/stage26/precision_audit.txt`):
    - FLAIR: map 20 → 9, code 22 → 6, as sharp as the image filter (8). The FLAIR backbone can be precise; InfoNCE
      never asks it to be.
    - T1: map 21 → 20, code 23 → 17, against the image filter's 11. On small and middle lesions the map is unchanged;
      labels only stop it compressing big ones (largest third 50 → 27).
    - The loss pays only for what both views agree on, so T1 sets the ceiling. At the labelled precision the views'
      readings agree at r .74, like a 12-voxel reading in both. On the control's price table that earns ~+.003 nats
      on uniform sets, ~+.02 on neighbour sets, and ~+.08 only on neighbour sets with position and anatomy removed.
    - The control code's burden errors are shared by the views (partial r .51, labelled .07), so part of its
      agreement is common error.
    - Ignore the supervised run's price block: its code was never trained to match the views (loss 12.3).
    - What the scans show (example scans rendered 11 Nov from a 160-test-subject local copy of the bank): per voxel
      the lesion stands out in both residuals, by 4.4 SD in T1 and 8.8 in FLAIR (medians). Around it the residual is
      smooth leftover structure (neighbouring voxels correlate at .86), not voxel noise, so the training jitter's
      resampling barely changes it (SD −7%). A raw-scan mismatch between training and the audit is unlikely.
    - Volume goes with the cube of the radius. At the median lesion (radius 2.7 voxels), 0.1 voxel of radius is 9
      voxels of volume, so reading burden's own part (SD 20) means placing the lesion's edge to ~0.1–0.2 voxel, on a
      backbone that works in 4-voxel cells.
16. **Why T1 stays near 20 voxels (stage 27, read 11 Nov, `causal/stage27/compare.txt`).** Stage 26's two runs read
    again, no training: on test scans moved by the training jitter (one move, and the mean over 8) and on training
    subjects.
    - Not jitter, not grid pose: one move reads slightly worse than the raw scans, and the mean over 8 moves gains under
      1.5 voxels.
    - Overfitting. With labels, training subjects are read far more sharply than test subjects: T1 map 10 vs 20, code
      4.6 vs 17 (FLAIR code 3 vs 6). The fixed image filter reads both at ~11, so the training subjects are not easier.
    - So stage 26's "the architecture caps T1" was wrong: the T1 backbone can hold a sharp reading, but what it learns
      from 2,000 subjects does not transfer to new ones.
    - InfoNCE stays at 18–19 voxels even on training subjects (code agreement .89 there, .72 on test): its first limit
      is still that nothing pays for precision.
17. **What makes the T1 residual patchy: the renderer's multiplicative field (11 Nov, local bank, no training).**
    - `render_modality` multiplies each view by 1 + 0.15 × noise drawn on a 16³ grid, so the field varies every ~4
      voxels, about a lesion's size; a real bias field varies over centimetres. It also scales the lesion's own
      contrast. T1 pays about three times what FLAIR does: ±15% of bright white matter (0.8) against a lesion 0.4
      darker, versus ±0.06 against 0.6 brighter in FLAIR.
    - The same 160 test subjects re-rendered with the field off (same anatomy, lesions, styles and noise draws):
      residual SD away from the lesion 1.08 → .71; lesion contrast / noise T1 4.4 → 8.4 (FLAIR 8.7 → 13.7); the blob
      filter's own-part error T1 15 → 10 voxels (FLAIR 10 → 8.6; readouts fitted on 160 subjects run high). What is
      left is thin rings at tissue boundaries, anatomy the 16 PCs miss.
    - A smooth field of the same strength barely helps (T1 blob filter 15 → 14), because it still scales each lesion's
      contrast; on real scans that is what N4 bias correction removes.
    - Every burden lesion is centred on a voxel, its volume lives in the partial-volume load of ~3 boundary shells,
      and the renderer box-blurs each scan (3³) after the noise.
    - Stage 28 tested whether the field is what keeps T1 burden unread (item 18).
    - Figure: `out/field_preview_20261111/render.py` writes `1_field_on_off.png`, one subject's scans, field and
      residual with the field on and off. Over 40 test subjects of the local independent bank, the lesion's own
      residual barely changes (T1 3.7 → 3.9 SD) while the spread around it halves (.98 → .47).
18. **The field was the cap (stage 28, read 12 Nov, `causal/stage28/compare.txt`).**
    - **Set-up.** Stage 26's labelled ceiling and stage 21's 24-unit InfoNCE arm, unchanged, on `bank_causal_nofield`.
    - **Results.** Reading error in voxels on new subjects, T1 / FLAIR, field on → off. The own part has SD 19.9 in
      both banks.
      - Image filter: 10.7 / 7.8 → 9.0 / 6.6.
      - Labels: map 19.6 / 8.9 → 8.3 / 7.3, code 16.9 / 6.2 → 5.9 / 4.7. T1 now reads like FLAIR, and better than
        the image filter.
      - Most of item 16's overfitting came with the field. On training subjects the labelled code reads 3.6, against
        5.9 on new ones (field on: 4.6 against 16.9).
      - InfoNCE backbone: pooled 19.5 / 19.1 → 11.1 / 11.2, GAP 11.0 / 11.0, map 21.2 / 20.1 → 14.9 / 15.9. There is
        no train–test gap (pooled 10.9 / 10.4 on training subjects). The views' readings agree at r .86–.87.
      - InfoNCE code: only 23.2 / 22.3 → 18.0 / 19.7 (agreement .72 → .80).
    - **The head is now the bottleneck.** With the field on, code and backbone were both near 20 voxels. Now the code
      loses most of the backbone's gain.
    - **Big lesions.** In the largest third of new subjects the InfoNCE code's error is 47 / 42 voxels (training
      subjects 29 / 37, map 29 / 28); the labelled code reads them to 5.5 / 3.9.
      - InfoNCE's features compress big lesions. It starts in the backbone, and the head makes it worse.
    - **Shared error.** The InfoNCE views share about half their reading error: on training subjects, shared-error r is
      .52 at the GAP and .58 at the code, against .14–.22 with labels. A cross-view loss pays for a shared error as it
      pays for burden.
    - **Own variation (read 13 Nov, `own_variation.txt`).** Burden's own part, T1 / FLAIR, field on → off: code
      .28 / .32 → .35 / .34; pooled .34 / .37 → .50 / .49; GAP .35 / .34 → .57 / .55. The second success test (code
      above .28 / .32) is passed only marginally: the backbone gained far more than the code.
      - The labelled code holds .41 / .48 although it reads volume to 5.9 / 4.7 voxels. Own variation also needs the
        parent (ventricle, .17 / .21 in that code) read precisely, so it understates a code that holds volume alone.
    - **Code audit (`code_audit.txt`).** The field-off code holds more anatomy than the field-on one. Own part, T1 /
      FLAIR: brain .48 / .32 (field on .18 / .44), temporal .73 / .71 (.15 / .18), thickness .34 / .43 (.11 / .14).
      Retrieval among all 1,000 test subjects is .93.
    - **Price (`precision_audit.txt`).** A burden unit read to ~10 voxels, about the backbone's precision, earns
      +.003 nats in uniform sets and +.017 in neighbour sets (+.024 with position and anatomy removed). At 20 voxels
      it costs −.011. The code is saturated (loss .10 on uniform sets).
19. **Two local checks of next-step levers (11 Nov, no training).**
    - **Per-unit weights in the critic** (the similarity inside InfoNCE).
      - Set-up: the run's InfoNCE on codes built from true latents (6 anatomy factors at cross-view r .9, the lesion
        centroid at .95), plus one unit holding burden's own part (SD 20) read with 20 voxels of error.
      - With a learned weight per unit inside the cosine, applied after the per-batch standardization, that unit stops
        costing the loss. On 9 units, uniform sets go −.12 → +.01 nats and neighbour sets +.02 → +.09.
      - A saturated 24-unit code pays ~0 either way. So weights help only on unsaturated terms: remainder units and
        neighbour sets.
    - **A repeat scan as a third view.**
      - Set-up: 600 local test subjects re-rendered with a second T1 and a second FLAIR (fresh style, field and noise).
      - With the fixed blob filter, agreement on a 20-voxel own part is T1–FLAIR .80, FLAIR–FLAIR .85 and T1–T1 .83.
        Part of the filter's error repeats within a modality (error r .33–.44).
      - Pairing T1 with FLAIR is not what blocks burden.
20. **Head levers on the field-off bank don't move burden (stages 29 and 31, read 13 Nov).** All 20k steps, seed 42,
    against stage 28's 24-unit InfoNCE arm. Code error in voxels, then burden's own part in the code, T1 / FLAIR:
    - 24 units (control): 18.0 / 19.7, .35 / .34. Backbone GAP 11.0 / 11.0, own part .57 / .55.
    - 48 units: 17.8 / 19.1, .36 / .35. Slightly more anatomy, nothing for burden.
    - Barlow Twins, 24 units: 15.4 / 16.1, the sharpest code (largest third 27 / 30 against 47 / 42), but .35 / .35.
      It loses temporal (.01), thickness, lr and sulcal: the code fills with lesion position (.94–.96) and brain.
    - 10 remainder units on GAP neighbourhoods: 19.0 / 18.7, .35 / .35, with the best code anatomy so far (own part:
      brain .90, lr .83, temporal .76, ventricle .38). Largest third 32 / 31.
    - Gain off (`bank_causal_nofield_nogain`): the image filter (8.0 / 6.2) and the backbone GAP (8.7 / 8.1, own part
      .66 / .65, the best backbone so far) sharpen, but the code stays at 17.5 / 18.2, .37 / .33.
    - Critic weights: on the main term 18.2 / 19.7, .33 / .31; on the remainder term 17.4 / 17.8, .35 / .32; on both
      17.8 / 18.6, .37 / .34. The two remainder arms keep the remainder arm's anatomy and read the largest third to
      26–28; the main-term arm is like the control.
    - **Burden's own part in the code sits at .31–.37 in all eight label-free arms,** while their backbones hold
      .51–.66. Width, Barlow Twins, remainder conditioning and critic weights don't move it.
    - **Price (stage 31's re-price of stage 29's codes).** Weights remove the cost of a 20-voxel burden unit (−.011 →
      .000) but add no reward. At ~10 voxels a burden unit earns +.000 to +.003 nats in uniform sets (+.015 in the
      Barlow Twins code, which InfoNCE has not saturated). Only with position and anatomy removed exactly, in neighbour
      sets, does it earn more (+.07 in the remainder code): a regime the remainder term's GAP neighbourhoods don't
      reach.
    - **What the code carries instead.** In the remainder codes the views' burden readings agree at .93, but their
      errors correlate at .85: a shared stand-in built from anatomy and lesion position, not burden's own part.
    - **Reading:** the blocker is the saturated InfoNCE at the head (retrieval ~.99 in sets of 32), not the backbone.
      Stage 30 changes the head's objective.
21. **Cross-view prediction at the head keeps what the backbone holds (stages 30 and 32; precision audit read 13 Nov,
    own variation and the audit not yet).** Backbone frozen; the head continued 10k updates with InfoNCE alone, or
    plus weight × each view's code predicting the other view's whitened GAP. Code volume error, T1 / FLAIR:
    - Field-off backbone (GAP 11.0 / 11.0): InfoNCE alone 23.9 / 22.9 (worse than its 18.0 / 19.7 start); weight 0.1
      21.0 / 20.4; weight 1 14.9 / 14.9; weight 3 14.0 / 14.6; weight 10 14.1 / 14.5. The weight saturates by 3, and
      the largest third stays at 27–29 (GAP 16).
    - Gain-off backbone (GAP 8.7 / 8.1): InfoNCE alone 17.8 / 17.6; weight 1 10.9 / 10.9, the best label-free code so
      far (largest third 15 / 17). Burden's own part read from the code with no help from the other factors ("alone")
      is .75 / .75, against the backbone's .82 / .83; every earlier code sat at .38–.53.
    - Every prediction head keeps the anatomy (mean R² .78–.82) and lesion position (.90–.91), as well as the
      remainder arms (.75–.76). The InfoNCE loss rises with the weight (.05 → .10). The head loses ~2–4 voxels against
      its backbone either way, so the gain-off code is better mainly because its backbone is.
    - InfoNCE alone sheds burden, position and anatomy as it keeps training while its loss halves: a saturated InfoNCE
      discards what it doesn't need.
    - **Not settled.** Barlow Twins also read ~15 voxels and held only .35 of burden's own variation, so
      `causal/stage32/own_variation.txt` (it covers stage 30's heads too) decides. Gain off is generator-side: gain
      scales tissue and lesion contrast by up to ±30% under a fixed noise level, and the scan channel is normalised
      with one fixed reference (only the residual channel is z-scored per scan). Per-scan white-matter normalisation
      (WhiteStripe) is the real-data counterpart but can't undo the change in contrast-to-noise, so test it before
      treating gain off as deployable.
22. **Training the prediction term end to end lifts burden's own part in the code (stage 33, read 15 Nov).** 20k
    steps from scratch, seed 42: InfoNCE plus weight × each view's code predicting the other view's whitened GAP,
    taken from a momentum copy of the backbone (section 9). Code volume error, T1 / FLAIR, against each bank's
    InfoNCE control and stage 32's frozen-backbone head:
    - Field off: control 18.0 / 19.7, head (weight 3) 14.0 / 14.6, weight 1 12.7 / 13.2, weight 3 12.8 / 12.8. The
      backbone GAP goes from 11.0 / 11.0 to 10.1 / 10.9 at weight 1 and 10.9 / 12.2 at weight 3.
    - Gain off: control 17.5 / 18.2, head (weight 1) 10.9 / 10.9, weight 1 8.9 / 9.2, the best label-free code so
      far. The backbone GAP goes from 8.7 / 8.1 to 7.6 / 7.2.
    - The weight-1 codes lose 1.3–2.6 voxels against their own backbone's GAP (heads 2.2–3.6, controls 7–10).
    - Big lesions: the largest third reads 18 / 20 (field off) and 10 / 11 (gain off), against 47 / 42 and 35 / 34 in
      the controls and 27 / 29 and 15 / 17 in the heads. The gain-off code reads small, middle and large lesions
      about equally (8.8 / 8.8 / 9.9 in T1).
    - The weight-1 codes keep the most anatomy of any code so far (mean R² .87–.91; controls .37–.52, heads
      .78–.82) and lesion position (.89–.91; controls .36–.45).
    - "Alone": field off .61 / .61 at weight 1 (control .40 / .38, head .53 / .52); gain off .76 / .75 (control
      .53 / .52, head .75 / .75). With gain off the end-to-end code beats the head on e but not alone.
    - Weight 3 reads burden as weight 1 does, here and in own variation. It trades lesion position (mean R² .76 /
      .77 against .89) and the FLAIR backbone (12.2 voxels) for a little ventricle and sulcal (own part .76 / .74 and
      .65 / .68, against .71 / .68 and .52 / .60).
    - Price: a 10-voxel burden unit earns InfoNCE −.006 to +.004 nats as trained (either set rule, with or without
      critic weights). Only with position and anatomy removed exactly does it earn more (up to +.08, field off), as
      before. The prediction term is what holds burden, and the table doesn't price it.
    - Gain costs the image filter 1.0 / 0.4 voxels (9.0 / 6.6 on, 8.0 / 6.2 off), the backbone GAP 2.3–3.7 and the
      weight-1 code 3.8 / 4.0. The residual keeps most of the information under gain; the network loses it.
    - **Own variation: burden's own part in the code, T1 / FLAIR.** Field off: .48 / .47 at weight 1, .49 / .47 at
      weight 3 (control .35 / .34, frozen head .41 / .40, backbone GAP .60 / .59). Gain off: .60 / .58 (control
      .37 / .33, frozen head .58 / .57, GAP .69 / .67). The first label-free codes above the .31–.37 band. On the
      field-off bank the weight-1 code matches the volume-supervised code (.41 / .48, item 18), which loses
      ventricle. Reference: "no own" .01; a linear readout of the true factors .89.
    - Every other factor rises too. Weight 1, field off, code: brain .48 / .32 → .95 / .94, ventricle .24 / .24 →
      .71 / .68, thickness .34 / .43 → .63 / .64, sulcal .12 / .15 → .52 / .60, lr .59 / .57 → .89 / .90. The
      backbone GAP gains as well (T1: ventricle .65 → .80, thickness .66 → .74, sulcal .44 → .66). With gain off the
      code reaches ventricle .82 / .79, thickness .74 / .75, sulcal .71 / .75.
    - With gain off the frozen head already held burden (.58 / .57). Training end to end adds anatomy there (T1:
      thickness .50 → .74, sulcal .32 → .71, ventricle .66 → .82), not burden.
    - **Code audit.** The weight-1 codes read every factor and the lesion centre within .05 of their own GAP by
      ridge (sulcal .10 on the field-off bank), and better by ridge than by gradient boosting, the reverse of the
      InfoNCE controls: the linear maps make the code linear in the factors. Cross-view top-1 among all 1,000 test
      subjects rises (.932 → .966 field off, .982 → .998 gain off), with train and test retrieval equal.
    - **Style and a second seed:** item 23. No code carries style. A second seed reads burden as sharply; its own
      variation is not read yet.
23. **No code carries style, and a second seed reads burden as sharply (stage 34; style and precision audits read
    16 Nov, own variation and the audit not yet).**
    - **Style** (`causal/stage34/style_audit.txt`): no code reads its own view's gain, bias or noise level. Ridge R²
      runs from −.01 to +.04 in every run, against −.01 to .00 for the other view's style (the null), and gradient
      boosting reads below 0. The most is a trace of gain in the FLAIR code of the gain-on weight-1 arms (.04 with
      seed 42, .03 with seed 2; InfoNCE control .00).
    - The head strips style; the backbone keeps it. The untrained twin's code reads gain at .39–.44 (bias at .44–.60
      on the gain-off bank), while every trained backbone's GAP still reads gain at .55–.75, bias at .25–.87 and the
      noise level at .26–.49.
    - On the gain-off bank, gain reads −.01 everywhere: the built-in null. The redraw reproduced both banks' stored
      content latents (`--causal`).
    - **Seed 2, code volume error T1 / FLAIR:** field off 13.1 / 12.6 (seed 42: 12.7 / 13.2), gain off 9.5 / 10.0
      (8.9 / 9.2). GAP 10.8 / 10.5 and 8.3 / 7.7, still at least as sharp as the InfoNCE controls (11.0 / 11.0,
      8.7 / 8.1). "Alone" .60 / .62 and .77 / .76 (seed 42: .61 / .61, .76 / .75). Largest third 19.7 / 18.2 and
      10.4 / 12.0. Anatomy .85–.89, lesion position .90–.91. Seeds differ by 0.4–0.8 voxels; the lead over the
      frozen heads holds at 0.9–2.0 voxels.
    - **Not settled:** seed 2's own variation (`causal/stage34/own_variation.txt`).
24. **Tabular causal benchmark: the end-to-end prediction code matches its own backbone (read 15–16 Nov,
    `causal/tabular/<run>/`).**
    - **What it is** (`eval/causal/TABULAR_BENCHMARK.md`). Synthetic tabular columns added to the 1,000 causal-bank
      test subjects from their true factors, over 10 task seeds.
      - Effect task: a hidden image score (all 7 factors, equal ± weights) and two tabular covariates confound a
        binary treatment. Read the share of the tabular-only bias (0.597) that adding the vector removes
        (cross-fitted AIPW).
      - Discovery task: one factor drives a continuous exposure and another the outcome (20 MRI→tab edges). Two
        tabular causes each drive one factor (20 tab→MRI edges). Each cause is a noisy copy of its target's own noise
        (`z_own`), so it reaches the image only through the target's own variation. A ridge readout fitted on the
        true labels decodes the factors; PC (Fisher-Z, α .05) runs on 500 held-out subjects.
      - One edge moves an F1 by ~.03–.07. PC can orient at most 15 of the 20 causes (5 target roots), so the
        oracle's orientation of .70 is the ceiling.
    - **Results, code, T1 / FLAIR** (confounding removed; MRI→tab F1; tab→MRI F1):
      - halves8 (field on, InfoNCE, 6k): 50 / 50%; .13 / .20; .37 / .50.
      - Field off, InfoNCE 24 units: 47 / 43%; .52 / .45; .51 / .66. + remainder units: 67 / 66%; .75 / .77; .51 /
        .65.
      - Frozen-backbone prediction heads: field off weight 3, 69 / 72%; .73 / .77; .72 / .77. Gain off weight 1,
        74 / 77%; .83 / .93; .83 / .88.
      - **End to end, weight 1 (stages 33–34), seed 42 · seed 2:** gain on 79 / 83% · 80 / 80%; .85 / .87 both;
        .93 / .91 · .89 / .88. Gain off 90 / 89% · 88 / 87%; .88 / .83 · .90 / .90; .93 / .96 · .94 / .96.
        Orientation .60–.70.
      - Their backbone GAPs: gain on 85 / 90% · 82 / 81%, gain off 89 / 94% · 91 / 91%; MRI→tab .85–.92, tab→MRI
        .88–.96. Oracle: ~100%, .95, 1.00.
    - **Reading.**
      - Discovery follows own variation, not readout R²: a factor's edges are mostly found at own variation ≳.3 and
        mostly missed at ≲.15.
      - The effect task needs every factor, so it punishes heads that drop anatomy. The InfoNCE heads lose brain size
        (R² .41 / .27 field off, ~0 in the frozen-backbone InfoNCE controls), and the prediction term restores it.
      - Before training end to end, the false incoming edges came almost all from sulcal, the least-read factor (48
        of 52 in the field-off InfoNCE codes). Its decoded value was built from its ancestors and inherited their
        dependencies.
      - The end-to-end codes now match their own GAP on every panel (within ±.05, sometimes above) in both seeds.
        The remaining distance to the oracle sits in the backbone. Gain costs 8–10 points of confounding removed.
    - **Caveats.** The readout uses labels, so this measures what the vector holds, not label-free discovery. Only
      the bank's 7 factors count, and the 10 tasks share one cohort.
    - **Running it** on any warm.py run with a TailCode head: `causal_bundle.py RUN OUT/code.npz --features code`,
      and the same for `gap` and for each `--untrained` floor. Then run
      `python -m eval.causal.tabular_benchmark --view each --bundles code=… gap=… --floors code=… gap=…
      --out OUT/report.json`. `run_pc.sh 22` does this with a CCA arm, but only for runs with stage 19 CCA bundles
      (`TABULAR_RUN`). The one-table summary, per-factor breakdown and false-edge scripts are kept in `~/` on the PC
      (`tabular_summary.py`, `tabular_breakdown.py`, `tabular_spurious.py`). halves8's `report_11nov.json` predates
      the tabular causes.
25. **Ten content units read burden worse than 24 (stage 35; precision audit read 16 Nov, own variation, the audit and
    the style audit not yet).** Stage 33's recipe with `--content-units 10`, one unit per shared latent (section 9),
    20k steps, seed 42. Code volume error, T1 / FLAIR, 24 → 10 units:
    - InfoNCE control: 18.0 / 19.7 → 18.1 / 18.4; "alone" .40 / .38 → .36 / .35. The theory's own setting at the
      matched width changes nothing for burden. The narrower code is less saturated (uniform loss .10 → .16), and the
      extra pressure went to lesion position (mean R² .39 / .37 → .47 / .44), not burden. A burden unit read to 10
      voxels still costs it (−.008 nats in uniform sets).
    - Prediction weight 1, field off: 12.7 / 13.2 → 17.3 / 17.6, within a voxel of the 10-unit control; "alone"
      .61 / .61 → .42 / .39. Gain off: 8.9 / 9.2 → 14.9 / 14.3; "alone" .76 / .75 → .56 / .60.
    - Largest third: control 47 / 42 → 61 / 40, weight 1 18 / 20 → 29 / 32, gain off 10 / 11 → 26 / 26.
    - Anatomy and lesion position left in the weight-1 codes (mean ridge R²) fall from .87–.91 to .61–.74. The
      end-to-end backbone gain is gone: GAP 11.1 / 11.9 (24 units 10.1 / 10.9, control 11.0 / 11.0); gain off
      8.9 / 8.0 (24 units 7.6 / 7.2, control 8.7 / 8.1).
    - Likely why: the width matches the code, not the target. Each code predicts the other view's 64 whitened GAP
      features linearly, and the GAP spreads the 10 latents over more than 10 linear directions (CCA lost burden at
      14, item 9). Ten units predict only a 10-direction slice, and burden's own part is not in it. The precision
      readout takes the better of ridge and boosting, so the 10 units don't hold burden non-linearly either.
    - Price: within each run, a burden unit pays only below ~6–8 voxels at 10 units (cross-view r ~.86–.92), against
      ~8–12 at 24. One unit is a bigger share of a narrow cosine. Critic weights remove the cost, as before.
    - With a smooth function of position and anatomy removed, the 10-unit codes tell 32 subjects apart far less well
      (uniform loss 1.2 / 2.3 / 1.6 against .37 / .98 / .35; control, weight 1, gain off). They hold less beyond the
      latents, as a matched width should, but burden went too. The views share more of the code's burden error
      (r .64–.75 against .57–.68).
    - **Reading:** keep 24 units with prediction weight 1. An 11-unit arm (the cosine's lost degree of freedom,
      section 9) would only close the theory's formal case: the 24-unit InfoNCE code had far more room and read burden
      the same.

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

## 9. State on 16 Nov

- **Stage 36 (built 16 Nov, not run): the next step.** Can a 10-unit code match the 24-unit one when its map to the
  other view is non-linear?
  - **Why:** stage 35's 10-unit prediction code read volume ~5 voxels worse than at 24 units (section 7, item 25).
    Its linear map predicts at most 10 directions of the other view's whitened GAP, which spreads the shared latents
    over more than 10 (CCA lost burden at 14, item 9). Through a non-linear map, a code that holds the 10 latents in
    any invertible form can predict all of them, and at the optimum it must hold every latent the other view's GAP
    varies with: the theory's matched-width claim, reached through prediction rather than InfoNCE's entropy term.
  - **Option:** `warm.py --predict-hidden H`. Each map becomes the linear map plus a one-hidden-layer MLP of width H
    (GELU), its output layer starting at zero, so the first step equals the linear map's. The default, 0, leaves runs
    unchanged.
  - **First, no training** (`bottleneck_check.py`, run while the arms train): on the frozen features of stage 33's and
    35's weight-1 runs (both banks), fresh bottlenecks shaped like TailCode's head are fitted on the training subjects
    to predict the other view's whitened GAP, through the linear or the MLP map, at 10 and 24 units, three seeds each,
    with no InfoNCE. Each is read on the test subjects like a code. Read it first: if the MLP bottleneck at 10 units
    does not read volume near the linear one at 24, the arms are unlikely to help. The linear one at 24 should read
    about as the run's own 24-unit code does; if not, the fit sets the numbers, not the map.
  - **Arms** on `bank_causal_nofield`, 20k steps, seed 42, stage 33's recipe with `--predict-hidden 128`:
    - 10 units (`ssl_uni_jitter_nopatch_restail_t104_causal_nofield_predict1_mlp128_c10_20k`);
    - the same with `--critic-weights main` (`..._predict1_mlp128_critmain_c10_20k`): a 10-unit cosine charges a
      less-agreeing unit more (item 25), and refitted weights removed that charge in the price table;
    - 24 units (`..._predict1_mlp128_c24_20k`): the map's effect alone.
  - **Reports** in `causal/stage36/`: the precision audit, own variation with `--boosted`, the code audit and the
    style audit, beside stage 35's 10-unit weight-1 arm and InfoNCE control and stage 33's 24-unit weight-1 arm. Their
    entries are reused from stage 35, except own variation, which rescores them for the boosted column.
  - **Success:** the 10-unit MLP code within ~1 voxel of the 24-unit linear arm (12.7 / 13.2), "alone" ≳ .55,
    burden's own part by boosting near the 24-unit arm's, and anatomy and lesion position ≳ .85 by boosting in the
    code audit. Read ridge beside boosting: the linear maps made the 24-unit code linear in the factors (item 22),
    which the tabular benchmark's ridge readout and PC's Fisher-Z rely on, and an MLP map may not.
  - **Gain off** follows if the field-off arm works.
  - **Checked locally:**
    - Unit tests: `tests/test_warm_prediction.py` (the default maps are the old linear maps; the MLP map starts as its
      linear map; one unit predicts two directions only through it; the options) and `tests/test_bottleneck_check.py`
      (two units drop a weak shared latent through a linear map and keep it through the MLP map).
    - A two-step default run is bit-identical before and after the change: history, weights and maps.
    - Two-step runs of the three arms on a 208 / 32 / 160 field-off causal bank finish and save their maps (and critic
      weights). `own_variation.py --boosted` and `bottleneck_check.py` run on such a run; the check took ~10 minutes
      on the Mac's CPU. Shell syntax and the stage 36 dry run pass.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 36`. It needs stages 33 and 35 finished, and reuses stage
    35's report caches, so start it after stage 35's reports are written.
- **Stage 35 (run 16 Nov; precision audit read in section 7, item 25: 10 units read burden worse than 24; own
  variation, the audit and the style audit not yet).** Stage 33's recipe at the width of the ground truth.
  - **Why:** the theory identifies the shared block when the code is as wide as the content the views share (Yao et
    al. 2024). The 24-unit code is wider, a case the theory is silent on (section 8).
  - **The count is 10:** the causal bank renders seven SCM factors into both views (brain, ventricle, burden,
    thickness, temporal, lr, sulcal) and the lesion's centre (x, y, z, drawn by `z_lesion`). `z_content` has nine
    slots, but slots 3 and 4 are unused with the burden target, and clean content draws no deformation or fissure
    field. Gain, bias, noise and jitter are per view, so they are style, not content.
  - **Two caveats:**
    - The cosine InfoNCE sees only the code's direction: 9 degrees of freedom in 10 units. The prediction term reads
      the standardized code, all 10. If 10 units fall short, an 11-unit arm separates that one lost dimension.
    - Ten latents need not fit ten linear directions. Linear CCA put the shared block above 24 directions at the GAP
      (item 9, section 8), so read the code audit's boosted rows beside its ridge rows.
  - **Arms:** 20k steps, seed 42, stage 33's recipe with `--content-units 10`. Prediction weight 1 and its InfoNCE
    control on `bank_causal_nofield`, and prediction weight 1 on the gain-off bank:
    `ssl_uni_jitter_nopatch_restail_t104_causal_nofield_{predict1_c10,c10,nogain_predict1_c10}_20k`.
  - **Reports** in `causal/stage35/`: the precision audit, own variation, code audit and style audit of the three,
    beside their 24-unit counterparts (stage 33's two weight-1 arms, stage 28's control). The counterparts' scores
    are copied from stage 34's reports, so only the new runs are scored.
  - **Reading:**
    - Does the 10-unit code still hold every factor (code audit, ridge and boosted) and burden's own part (.48 / .47
      with gain on, .60 / .58 with gain off, at 24 units)?
    - Is every shared code direction a function of the 10 latents (the code audit's "both" column near 1)?
    - Does InfoNCE alone, the theory's own setting, do better at the matched width than at 24 units (.35 / .34)?
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 35`. It needs stages 28 and 33 finished for the
    counterparts.
- **Stage 34 (run 16 Nov; style and precision audits read in section 7, item 23; own variation and audit not
  yet).** Two checks before stage 33's prediction term joins the recipe.
  - **Style** (`style_audit.py`, no training): does a code carry its own view's gain, bias or noise level? Each view
    draws its own (the generator's shared gain-bias option is off), so neither loss term rewards them, but nothing
    has measured them: `code_audit.py` scores only the content factors and the lesion centre.
    - Banks don't store the style latents. The script redraws them per test subject from the generator's own seeds,
      with the generator set up as `bank.py` sets it up, and refuses to score unless the redrawn content latents
      equal the stored ones. `bank.py` records none of its options, so it tries those that change the draw
      (`--causal`, `--nuisance`) and names the one that reproduces the bank.
    - Targets, as the renderer applies them: clamp(z0) for gain, clamp(z1) for bias, |z2| for the noise level.
      Read by own_variation's ridge (and by gradient boosting on the code) from the code, pooled features and GAP,
      against the other view's style (drawn independently: the null) and against the untrained twin.
    - On the gain-off bank, gain is drawn but never rendered, so its column is a null too.
  - **Seed:** stage 33's two weight-1 arms again with seed 2
    (`ssl_uni_jitter_nopatch_restail_t104_causal_nofield_predict1_c24_20k_s2` and
    `..._nofield_nogain_predict1_c24_20k_s2`). The InfoNCE controls stay single-seed: eight InfoNCE arms sit at
    .31–.37.
  - **Order:** the style audit runs first, on stage 33's five runs and stage 32's two heads, while the seed-2 arms
    train. Then come the precision audit, own variation and the audit of both seeds and both controls, and the style
    audit again with the seed-2 runs added. The three reports start from copies of stage 33's caches, so only the
    seed-2 runs are scored anew; they also print stage 33's other runs. Reports in `causal/stage34/`.
  - **Pass:** each prediction code reads its own view's gain and bias no better than its InfoNCE control's code
    does, and near the null; seed 2 within ~.10 of seed 42 on burden's own part in the code (.48 / .47, .60 / .58),
    and above .31–.37. If even the GAP can't read the noise level, its column says nothing.
  - **Checked locally:** six unit tests (`tests/test_style_audit.py`): the targets; the redraw keeps the setting that
    reproduces the bank and refuses a bank no setting reproduces, or any subject that differs; the cache; the
    own-view reading against the null. On stage 33's two-step smoke runs, the redraw reproduces the repo's `bank`
    (no options) and the small causal field-off bank (`--causal`). At init, gain reads at ~.7 from the GAP and
    pooled features and ~.5 from the code, bias at ~.3 and ~.1, and the noise level not at all (≤ 0), with nulls
    at or below 0. Locally, gradient boosting can hang with torch loaded (OpenMP), so the unit test stands ridge in
    for it; on the PC, `code_audit.py` has always run it.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 34`. It needs stages 28, 29 and 33 finished.
- **Stage 33 (run and read 15 Nov, section 7, item 22: passes all four success tests at weight 1).** Stages 30 and
  32's prediction term trained end to end, `warm.py --predict-weight W`.
  - **Why:** the frozen-backbone heads keep all but ~2–4 voxels of what their backbone holds (section 7, item 21), so
    the backbone is now the limit. Training it with the term may sharpen it, and one training run keeps the method a
    training objective. Section 8 names latent correlation maximisation trained end to end as the theory-backed
    counterpart of the CCA readout; this term is close to it.
  - **Term:** InfoNCE plus W × the mean squared error of each view's linear map from its code units (standardized over
    the batch, as the InfoNCE sees them) to the other view's backbone GAP. The targets come with no gradient from a
    momentum copy of the backbone (momentum 0.996) on the unjittered scans, whitened by running statistics (momentum
    0.01; ridge 0.1 on the correlation matrix, as in stage 30). The copy keeps the backbone from making its own
    targets easy to predict. The maps are dropped after training; the checkpoint keeps them and the whitening under
    `predict`. Each step costs one extra backbone pass without gradient.
  - **Arms**, 20k steps from scratch, seed 42, stage 28's recipe (`--uniform --residual --jitter 2 --rotate 3
    --patch-weight 0 --content-units 24`): weights 1 and 3 on `bank_causal_nofield`
    (`ssl_uni_jitter_nopatch_restail_t104_causal_nofield_predict1_c24_20k` and `..._predict3_c24_20k`, against stage
    28's `..._nofield_c24_20k`), and weight 1 on the gain-off bank (`..._nofield_nogain_predict1_c24_20k`, against
    stage 29's `..._nofield_nogain_c24_20k`). No remainder units: the prediction heads already hold anatomy as well.
  - **Reports:** `causal/stage33/{precision_audit,own_variation,code_audit}.txt`: the three arms, both controls, and
    stage 32's `head_nofield_c24_predict3_10k` and `head_nogain_c24_predict1_10k` for reference. Each eval in a run's
    `report.json` also logs `target_spread`, about how many directions the targets spread over (the participation
    ratio of their correlation eigenvalues; ~2.5 of 64 at init). A fall toward 1 would mean the backbone's GAP is
    collapsing onto a few directions.
  - **Success:** the backbone GAP at least as sharp as its InfoNCE control (11.0, or 8.7 / 8.1 with gain off); the
    code sharper than the frozen-backbone heads (14.0 / 14.6, or 10.9); burden's own variation in the code above
    .31–.37; anatomy holding. If the backbone gets worse, keep the two-step version (stages 30, 32).
  - **Checked locally** (200-subject field-off bank, two steps on the CPU): weight 0 reproduces the control's weights
    exactly; weight 1 trains, logs the term and saves the maps; own variation reads the run. The precision audit and
    the code audit need more test subjects than that bank has (they fail there for the control too), so they were
    run on two-step runs on the repo's full-size `bank`, where both read the control and weight 1. Six unit tests
    (`tests/test_warm_prediction.py`): the running whitening starts equal to stage 30's fit and follows new batches by
    its momentum, each code predicts the other view, the target gets no gradient, the momentum copy moves toward the
    online backbone, and the options are checked.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 33`. It needs stages 28 and 29's controls finished and
    renders nothing new.
- **Stage 32 (run 13 Nov; precision audit read in section 7, item 21; own variation and audit not yet).** Two
  head-only follow-ups to stage 30, with its script (`frozen_prediction.py`) unchanged.
  - **Why:** stage 30's weight-1 head reads lesion volume to 14.9 voxels in both views, against 23.9 / 22.9 for its
    InfoNCE control and 11.0 for the frozen backbone's GAP, and keeps the anatomy. The error fell steadily with the
    weight (0, 0.1, 1), so it may fall further. The gain-off backbone reads volume more precisely (GAP 8.7 / 8.1).
  - **Arms**, 10,000 head updates each, seed 42:
    - weights 3 and 10 on stage 30's backbone, cache and batches (`head_nofield_c24_predict3_10k`,
      `head_nofield_c24_predict10_10k`), against stage 30's control and weight-1 head;
    - weight 1 and its InfoNCE control on stage 29's gain-off backbone (`head_nogain_c24_predict1_10k`,
      `head_nogain_c24_nce_10k`; source `ssl_uni_jitter_nopatch_restail_t104_causal_nofield_nogain_c24_20k`), with
      their own cache of eight jitter draws (`causal/stage32/features_nogain.pt`).
  - **Reports:** `causal/stage32/{precision_audit,own_variation,code_audit}.txt`: the four new heads beside stage
    30's control and weight-1 head and both source runs. The price table prices InfoNCE only.
  - **Success:** the code's volume error falls toward its backbone's GAP (11.0, or 8.7 / 8.1 with gain off), and
    burden's own variation in the code rises above stages 29 and 31's .31–.37, with anatomy holding. At weight 10
    InfoNCE has little say, so also watch the as-trained InfoNCE loss and retrieval in the audit.
  - **Checked locally:** shell syntax and the stage 32 dry run. The trainer is unchanged; there is no local gain-off
    source run to smoke-test it on.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 32`. It reads stage 30's cache and needs stage 30's
    control and weight-1 head, and stage 29's gain-off run, finished.
- **Stage 31 was read on 13 Nov** (section 7, item 20). Critic weights change nothing for burden (code .31–.37,
  17–20 voxels). As built: per-unit critic weights, `warm.py --critic-weights` (section 7, item 19).
  - **First, no training:** stage 29's precision audit is re-priced with the critic refitted to each of its five
    codes (`causal/stage31/price_stage29.txt`). Read it first: if the weighted rows don't make a burden unit read to
    ~11 voxels pay, the training arms are unlikely to help.
  - **Arms** on `bank_causal_nofield`, 20k steps, seed 42:
    - stage 29's remainder arm with weights on the remainder term, and on both terms (control: stage 29's remainder
      arm);
    - stage 28's 24-unit arm with weights on its main term (control: that arm). That code is saturated, so the weights
      should change little.
  - **Reports** go to `causal/stage31/`: `precision_audit.txt`, `own_variation.txt` and `code_audit.txt`. Each run's
    `report.json` history logs the weights (`critic_weights`); watch for them collapsing onto a few units.
  - **Checked locally:** at weight 1 the loss equals the recipe's; the default arms give identical reports before and
    after the change; two-step runs of the three arms, and the reports, finish.
  - **Numbering:** stage 30 (frozen-backbone prediction) was built at the same time in another session, so this one
    is 31, with its own `causal/stage31/`.
- **Stage 30 (run 13 Nov; precision audit read, own variation and audit not yet).** Frozen-backbone feature
  prediction, `frozen_prediction.py`.
  - **Precision audit (`causal/stage30/precision_audit.txt`).** Code volume error, T1 / FLAIR, with the frozen GAP at
    11.0 / 11.0: start (stage 28's head) 18.0 / 19.7; InfoNCE alone 23.9 / 22.9; weight 0.1 21.0 / 20.4; weight 1
    14.9 / 14.9 (largest third 26 / 28; lesion position .90 and anatomy .78 still in the code). More InfoNCE alone
    sheds burden, position and anatomy while its loss halves. Barlow Twins also read ~15 voxels (section 7, item 20)
    and held only .35 of burden's own variation, so own variation decides.
  - **Why:** stages 29 and 31 show the head drops burden's own part because the saturated InfoNCE stops paying for it
    (section 7, item 20). A prediction error keeps paying for any feature the code fails to reproduce. Whitening gives
    every direction of the other view's features equal weight, including the faint one that carries burden's own
    part. Predicting the other view, not the same one, limits the code to what both views share (style differs
    between the scans), with no labels. Freezing the backbone isolates the head: can it keep what the backbone holds?
  - **Loss:** InfoNCE plus weight × the mean squared error of each view's linear decoder (T1 code → whitened FLAIR GAP,
    FLAIR code → whitened T1 GAP). Only the head and the two decoders train; the decoders are dropped after training.
  - **Source:** stage 28's finished `ssl_uni_jitter_nopatch_restail_t104_causal_nofield_c24_20k`.
    No new bank is rendered. All arms continue its identical shared 24-unit head; the backbone, pooling statistics
    and gauges stay frozen.
  - **Arms:** InfoNCE alone (`head_nofield_c24_nce_10k`), plus prediction at weight 0.1
    (`head_nofield_c24_predict01_10k`) and weight 1 (`head_nofield_c24_predict1_10k`). Each runs 10,000 head updates,
    seed 42, with reports every 2,000.
  - **Prediction:** the T1 code predicts the frozen FLAIR backbone's 64 global-average features, and vice versa.
    Two linear predictors read only the exported code, with no feature bypass or factor labels. They are used only
    during training; every factor is still read from the same code.
  - **Scaling:** target whitening decorrelates and rescales feature directions, with ridge 0.1 to limit amplification
    of weak directions. It and the fixed initial-code mean/SD use training subjects only. Test features are cached
    for reporting and never fit these statistics.
  - **Matched input:** `causal/stage30/features.pt` stores eight independent source-jitter draws per training subject
    as frozen pooled head inputs. Targets are the unaugmented global-average features. Preparation runs once before
    the parallel arms; all three use the same cache and batches. This is a finite set of eight augmentations per
    subject, unlike stage 28's fresh jitter every step, so compare the prediction arms with their matched control.
  - **Reports:** `causal/stage30/{precision_audit,own_variation,code_audit}.txt`, including the source run.
    The precision price table measures only the InfoNCE component, not the new total loss. The reported untrained
    twin is a fully random network; the source run is the starting point for these head continuations.
  - **Success:** the code's equivalent volume error moves toward the frozen backbone's ~11 voxels, burden's own
    variation improves over the matched control, and anatomy holds. A lower prediction loss alone is insufficient.
  - **Checked locally:** six focused tests; two-step runs of all three arms on the small field-off bank with two
    cached jitter draws; identical initial heads and batch sequences; unchanged backbone and pooling state; saved
    codes reproduced by the existing loader; all three reports finished. Shell syntax and stage-30 dry run pass.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 30` after stage 28 has finished.
- **Stage 29 was read on 13 Nov** (section 7, item 20). No head lever moves burden's own part in the code; remainder
  units give the best code anatomy, and gain off the best backbone. As built:
  - **Arms.** Three head levers on `bank_causal_nofield`, against stage 28's 24-unit InfoNCE arm: 48 units, 10
    remainder units on GAP neighbourhoods, and Barlow Twins with 24 units. Plus stage 28's arm on
    `bank_causal_nofield_nogain`, which tests gain alone.
  - **Reports** go to `causal/stage29/`: `precision_audit.txt` (with the price table), `own_variation.txt` and
    `code_audit.txt`.
  - **Success:**
    - The code's error falls from 18–20 voxels toward the backbone's ~11.
    - Burden's own part beats the 24-unit arm, and anatomy holds.
    - The largest third of lesions is read better than now (47 / 42).
  - **Checked locally:**
    - The default bank recipe is bit-identical to the stored bank.
    - `--gain off` keeps every latent and lesion, and cuts the scan-to-scan spread in contrast (CV of each scan's
      brain SD) from .23 to .06 in T1.
    - Two-step runs of all four arms, and the three reports, finish.
- **Stage 28 was read on 12 Nov** (section 7, item 18). Without the renderer's field:
  - With labels, T1 reads new subjects to 5.9 voxels (field on 16.9).
  - The InfoNCE backbone reads to ~11 (field on ~20).
  - The 24-unit code reads only to 18–20, so the head is now the bottleneck.
  - The first success test (T1 behaves as FLAIR) is met. The second, the code's burden own part above .28 / .32, is
    met only marginally (.35 / .34, against .57 / .55 in the backbone; read 13 Nov).
- **Stage 28 overturns parts of stages 26–27.** Those were read with the field on. The ~20-voxel backbone and the T1
  overfitting both came mostly from the field.
- **Stage 26 was read on 11 Nov** (section 7, item 15). With labels the FLAIR backbone reads lesion volume to ~9
  voxels, the T1 backbone still to ~20. T1, not the objective, caps burden in this network, so stage 24's real win
  (the backbone below ~20 voxels) is unlikely on T1. (Field on; see stage 28.)
- **Stage 27 was read on 11 Nov** (section 7, item 16). The T1 cap is overfitting, not capacity, jitter or grid pose:
  with labels the backbone reads training subjects' T1 volume to ~10 voxels and new subjects to ~20. (Field on; most
  of the overfitting went with the field in stage 28.)
- **Stage 23 was read on 11 Nov.**
  - The remainder units learned brain size and anatomy, not burden.
  - The whole code has the best anatomy on the causal bank. Burden's own part is .19 / .17, and the backbone's
    precision is unchanged.
  - See section 10, item 3.
- **New since 10 Nov:** the 48-unit run (section 10, item 4) and the precision audit (item 5).
  - With the field on, the backbone reads lesion volume to ~20 voxels everywhere.
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

**Next, after stage 34's style and precision audits (16 Nov).** The numbered items below were written with the field
on.
- **Read `causal/stage34/own_variation.txt` and `code_audit.txt`.** Stage 33 passed all four success tests at weight
  1 (section 7, item 22): burden's own part in the code .48 / .47 with gain on and .60 / .58 with gain off, against
  .31–.37 before. Stage 34 found no style in any code and seed 2 as sharp as seed 42 (item 23). If seed 2's own part
  also holds, the prediction term at weight 1 is ready to join the recipe.
- **Read the rest of stage 35** (`causal/stage35/`; precision audit read, item 25): `own_variation.txt`, and
  `code_audit.txt`'s boosted rows and "both" column (do the 10 units hold anatomy and position non-linearly, and
  nothing else?). 10 units read burden worse than 24, so the recipe keeps 24.
- **Run stage 36** (`run_pc.sh 36`, built 16 Nov, section 9): a non-linear prediction map at 10 units. Read
  `causal/stage36/bottleneck_check.txt` first; it is ready while the arms train.
- **Then test white-matter normalisation** of the scan channel on the gain-on bank with prediction weight 1 at 24
  units, the real-data version of gain off. Gain costs burden's own part in the code .60 → .48, and costs the code ~4
  voxels against ≤1 for the image filter (item 22).
- Stage 32's `own_variation.txt` and `code_audit.txt` are unread, but stage 33's reports cover its two heads that
  matter (field-off weight 3, gain-off weight 1); the rest are stage 30's weaker heads.
- **Decide with the user whether `bank_causal_nofield` becomes the main burden bank,** with the field-on bank kept as a
  stress test. For: a real bias field varies over centimetres and N4 removes it, so the 16³ field has no counterpart
  in corrected scans. The same question for gain off, whose real counterpart is intensity normalisation (WhiteStripe).
- **Tabular benchmark (read 16 Nov, item 24).** The end-to-end weight-1 codes match their own GAP on confounding
  and graph recovery in both seeds; with gain off they remove 87–90% of the confounding. That is one more reason for
  prediction weight 1 to join the recipe. Score the white-matter-normalisation run the same way.
- **Done on 16 Nov:** stage 34 ran; its style and precision audits are read (item 23). Stage 35 ran; its precision
  audit is read (item 25).
- **Done on 15 Nov:** stage 33 ran and its three reports are read (item 22). Weight 3 reads burden no better than
  weight 1 and gives up lesion position for a little ventricle and sulcal.
- **Done on 13 Nov:** stage 28's remaining reports (item 18) and the head levers of stages 29 and 31 (item 20: width,
  remainder units, Barlow Twins, gain off, critic weights). None moved burden's own part in the code. Stages 30 and 32
  ran and their precision audits are read (item 21).
- **Big-lesion compression** (item 18). Labels remove it, so nothing in InfoNCE pays to keep big lesions apart.
  Remainder units and critic weights cut the code's error on the largest third from 47 / 42 to 26–33; prediction
  trained end to end cuts it to 18 / 20, and 10 / 11 with gain off (item 22).

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
| head levers for burden on the field-off bank: 48 units, Barlow Twins, remainder units, critic weights (stages 29, 31) | burden's own part in the code unchanged (.31–.37); remainder units stay for anatomy |
