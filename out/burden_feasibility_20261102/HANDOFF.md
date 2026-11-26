# Burden feasibility: handoff (13 Nov 2026)

Everything for this line of work lives in this folder. `run_pc.sh` runs it in numbered stages, and its header
describes every stage. This file says what the work is for, how it is run, what has been found, what is still open
and what to do next. Repo-wide conventions are in `CLAUDE.md` at the root. They apply to every agent.

## 1. The goal

- **The model.** The encoder-only multiview contrastive model on paired synthetic T1/FLAIR brain MRI.
  - Each view has its own backbone.
  - A shared head turns the pooled features into the content code: 14 units at first, 24 in the plan adopted on 18
    Nov (section 10).
  - InfoNCE aligns the two views' codes.
- **The target.** Get the burden (total volume) of one small white-matter lesion into the code, with no labels and
  without losing the anatomy factors.
  - Under the 18 Nov plan this tests whether a label-free code is complete: whether it holds a weak, focal factor at
    all. The causal variables come from anchors, not from the code's units.
- **The hard case.** T1 lesion contrast 0.4: faint in T1, bright in FLAIR.
- **Constraints set by the user.** These are not negotiable:
  - **Generic levers only.** A method must find any hard feature, not just the lesion. Lesion-specific tools (blob
    filters, lesion branches, views built around the lesion) are diagnostics at most. A lesion-blind third view was
    built and then rejected for this reason ("i want it to be lesion-agnostic").
  - **One shared route.** Every factor is read through one shared pooling route, never per-factor heads.
  - **Real-data levers only.** The work has to transfer to ADNI, where the subject count is fixed. "More subjects"
    is a diagnostic, never the proposal. Augmentation-type levers are fine.
  - **Width and causal variables (18 Nov; replaces matched width).** The code is a 24-unit embedding of the shared
    anatomy, its width set by a label-free agreement rule. The causal variables come from anchors (measurements), not
    from the code's units (section 10, the plan).
    - Matched width (16 Nov: 10 units, one per shared latent, to fit Yao's theory) is dropped. The user's reason was
      that the causal reasoning needs the right number of units. Stages 41–44 showed the units are mixtures at every
      width, so the number belongs to the causal variables; matched width costs every causal task; and ADNI has no
      finite shared block to match. The 10-unit results stay as a finite-sample finding.
  - **Judge a 10-unit lever on every weak factor (17 Nov).** The narrow code drops the own parts of ventricle,
    thickness and sulcal as well as burden's (item 30). The user declined white-matter normalisation because it
    targets the lesion.

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
  - Working on the PC itself (as on 18 Nov): edit a copy of `run_pc.sh` and `mv` it over the original. The running
    bash keeps the old file open (`ls -l /proc/PID/fd/255` then shows it as deleted), so the stage is unaffected.
    Never edit `run_pc.sh` in place while a stage runs.
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

### On Run:ai (ADNI, stages 68 and 73)

The ADNI scans live on the cluster's NFS. From the Run:ai host (`h2`), once the checkout at
`/nfs/home/nglazman/crl-2/multiview-crl` holds this branch:

```bash
bash out/burden_feasibility_20261102/runai_submit.sh --dry-run
```

- **`runai_submit.sh [STAGE...]`** (default 68) submits one A100 job that runs `runai_job.sh` in the training image.
  Without `--dry-run` it submits. Set `ADNI_*`, `WHEELS` or `PYTHONUSERBASE` to pass them to the job.
- **`runai_job.sh`** covers what the image lacks:
  - no git: `run_pc.sh` gets a stand-in for its one git call (`rev-parse --show-toplevel`);
  - no SimpleITK: it is installed once into `PYTHONUSERBASE` on /nfs (default `/nfs/home/nglazman/.pyuser/py312`),
    from a wheel in `WHEELS` if one is there, else from PyPI. If the pod has no internet, the job stops and prints
    the `pip download` line to run where pip works.
  - The ADNI paths default to `experiments/cluster/runai.yaml`'s. The October run's split.json is the split check
    when it exists in the checkout.
- **Progress:** `adni/runai.log`, then `adni/bank_n4_64.log`, `adni/bank_n4_64.residual.log` and
  `adni/logs/<run>.log`. A resubmit skips finished steps, but restarts an unfinished training run from step 0.
- **Stage 73** (once stage 68 has finished): `bash out/burden_feasibility_20261102/runai_submit.sh 73`, after a
  `--dry-run`. It needs no ADNI paths and no SimpleITK. Its three runs share the A100; expect about 3 hours (one
  20,000-step run alone took 2). Ran 25 Nov 12:32–16:04 (0.58–0.63 s/step with two training runs on one A100); read
  26 Nov, section 10.
- **Stage 79** (built 26 Nov, not submitted): stage 74's sweep on the drawn causal bank. Commit and push, pull on the
  cluster checkout, then one job per seed and a final scoring job, each after a `--dry-run`:
  `WIDTH_SEEDS=42 JOB_NAME=burden-s79-width-s42 bash out/burden_feasibility_20261102/runai_submit.sh 79`; then the same
  with `WIDTH_SEEDS=3` and `WIDTH_SEEDS=4`, each with `WIDTH_WAIT_BANK=1` (the first job renders
  `bank_causal_nofield` and its residuals unless that bank is copied to the cluster first, as the fixed bank was);
  when all 27 runs log DONE, `JOB_NAME=burden-s79-score bash out/burden_feasibility_20261102/runai_submit.sh 79`.
  Nine runs a job, five at a time: about 4–5 hours a job at stage 74's pace, plus the render.

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
  - `bank_causal_nofield_k32`, `bank_causal_nofield_nogain_k32` (stage 37): the two banks above with 32 principal
    components removed from the residual instead of 16. Only the residuals are new; every other array is a link.
  - `bank_causal_nofield_fixedpos` (`--causal --field off --lesion-position fixed`, stage 47; a diagnostic): every
    lesion at the white-matter voxel nearest `bank.FIXED_CENTRE` that can hold the full-size sphere, in place of its
    drawn position. The drawn position is still drawn, so the subjects, anatomy, lesion sizes and style match
    `bank_causal_nofield` (checked bit for bit on small banks). The centre moves only with the anatomy: total SD 2.7
    voxels against 12.1, and 56% of subjects get the voxel itself. The shared block is then the 7 SCM factors.
- **`adni_bank.py`** (24 Nov) writes ADNI in the same layout, with no latents, for `warm.py --real`:
  `{split}_t1.npy` and `{split}_flair.npy` (64³ float16), `{split}_ids.npy`, `manifest.csv` (split, row, QC,
  exclusion, and the image ID, series ID and date where a file name carries them) and `bank_meta.json`.
  - Each scan: N4 inside its brain mask (`--bias-correction n4`, needs SimpleITK), 3 mm block means over brain voxels
    only, the centre 64³ box (the MNI brain fits), z-scored over the brain.
  - The split is the encoder ADNI recipe's (seed 0, 0.2 / 0.1; `--check-split` compares with a run's split.json; the
    October run's has 1,157 / 289 / 160). Pairs are then dropped within their split by the rule fixed on 24 Nov,
    before any ADNI result of this recipe: T1/FLAIR mask Dice < .9 or a mask < 300 mL. `--qc-only` stops after the
    manifest. Tests: `tests/test_adni_bank.py`.
  - The `ADNI_stripped_masks` tree looks uncorrected for bias: in `mri_preproc/scripts_preproc`, skull stripping reads
    `ADNI_registered`, and `bias_correction.py` (FSL FAST) wrote FLAIR only, into a separate tree. Hence N4 here.
- **`residual_bank.py`** writes each view's normative residual: the scan z-scored over its brain, minus the
  training mean, minus its first 16 principal components, scaled to unit SD. Each modality uses only its own basis.
  - `--components K` removes K components (at most 64, what the stored basis holds).
  - `--source OTHER_BANK` first links OTHER_BANK's arrays and basis into the bank, so a residual with another K is a
    bank of its own and the source is left as it is.

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
    `--predict-target grid|quantile` replaces the GAP target with the other view's map as eight octant averages, or
    as each channel's quantiles (`PREDICT_QUANTILES`) over the interior cells (stage 53; tests:
    `tests/test_warm_prediction_target.py`).
  - `--entropy-units N`, the bridge (stage 46b): the recipe's InfoNCE on the first N code units and alignment alone,
    (1 − cosine) / τ, on the rest. Stage 46 ran an earlier, degenerate form (item 37). `bridge_view.py` writes those
    N units out as a run of their own.
  - `--hard-negatives G`, `--seed S`.
  - `--real` (24 Nov): an `adni_bank.py` bank (`--t1key t1`). Training is unchanged. Each eval reads only images, on
    validation subjects (the test split stays unread): cross-view top-1 among all of them, on as many training
    subjects (the gap is memorisation) and in sets of 32, the held-out loss, effective rank, matching-unit r, the
    backbone GAP's top-1 and `eval.adni.encoder_modality_gap`'s modality probes. The step-0 row is the floor. Needs
    `--uniform` or `--code-batches`. `summarize.py` prints these runs from `adni/runs/`.
  - `--real` readouts (25 Nov, `gauge_readouts`; TailCode pooling only). The evaluated code gauges clean images with
    the gauge's running statistics; the loss's code gauges jittered images with each batch's own (BatchNorm gauge) or
    running ones (`--gauge running`). Each readout undoes one difference: `batch_clean` and `batch_jittered` (each
    evaluation set gauged by its own statistics, top-1 in the sets of 32; BatchNorm gauge only; `batch_jittered` is
    the loss's code), `running_jittered`, and `recalibrated` (a copy of the pool whose gauge is recomputed from all
    clean training images, the network frozen). The last two also give top-1 among all and the modality probes. None
    changes the run: `tests/test_warm_readouts.py`. The jitter repeats exactly at every eval.
  - Stage 73 (read 26 Nov, section 10): read `recalibrated`. The plain row reads the code through the gauge's stored
    statistics, which can be far off (top-1 .831 plain against .953 recalibrated on the same checkpoint).
  - `--keep-checkpoints` keeps `model_STEP.pt` at every eval. `--init-run RUN/model.pt` with 0 steps evaluates a
    finished run once (its row is labelled with the checkpoint in `summarize.py`).
  - Older levers: `--queue`, `--gauge running`, `--group-dropout`, `--decorrelate`, `--shared-jitter`, `--uniform`.
  - Each option refuses combinations it can't handle.
- **`sup.py`** trains the same backbone with lesion labels. It was used early as a ceiling.

## 6. Evaluation tools and how to read them

| tool | what it reports |
|---|---|
| `summarize.py` | each arm's training history, compact enough to paste |
| `code_audit.py` | factor R² from the code and backbone GAP (ridge, boosted), lesion centre x/y/z, the same rows for the untrained twin (the floor), canonical correlations, direct correlations between matching code units, effective rank (how many distinct directions the code uses), and retrieval on train and test subjects |
| `own_variation.py` | causal bank only: each factor's **own variation** (its SCM noise) read from the code, the pooled features and the GAP, trained vs untrained twin, with reference columns. `--boosted` adds the code read by gradient boosting beside the true factors read the same way: ridge reads only what a code holds linearly |
| `bottleneck_check.py` | no training: fresh 10- and 24-unit bottlenecks fitted on a run's frozen features to predict the other view's whitened GAP through a linear or an MLP map, read like a code (volume error, alone, anatomy and position by ridge and boosting, each factor too). `--targets gap tails pooled` also predicts the other view's tail features or its whole pooled vector; `--heads 512 512x2` tries wider or deeper heads than TailCode's (one hidden layer of 100). Stages 36, 39 and 40 |
| `style_audit.py` | each view's gain, bias and noise level read from the code, pooled features and GAP, trained vs untrained twin, against the other view's (drawn independently: the null). Banks don't store style, so it is redrawn from the generator's seeds and must reproduce the stored content latents. `code_audit.py` scores no style |
| `width_sweep.py` | stage 74: `measure` (GPU) each run's held-out InfoNCE on test subjects and own-part R² from its exported code; `score` (no torch) the elbow of the mean loss over widths against \|C\| + 1, own-part saturation width, figure (skipped without matplotlib). Item 63 |
| `spare_units.py` | no torch: where a code's whitened directions go as width grows (own parts linearly, curves of the factors by boosting, shared outside the factors, view-private), and which factors the directions past the seventh are curves of. Item 63 |
| `width_rules.py` | no torch: E-W's rules 2 (TwoNN of the widest code plus the bank's offset from its true latents) and 3 (smallest width with top-1 among all within 5% of the range below the widest's) on a width sweep's exports, with the rules' numbers in its docstring. `--self-test`. Item 64 |
| `precision_audit.py` | the lesion-volume reading error in voxels at each stage (image filter on the residual, backbone map, GAP, pooled features, code), with the anatomy part handed to the readout; error by lesion size, error shared by the views, and the loss's reward for a sharper reading, also with lesion position and/or anatomy removed from the code. `--jitter-draws K` reads test scans moved by the run's training jitter (mean over K moves), `--split train` reads training subjects, and `precision_compare.py` puts conditions side by side. Two "weighted" price rows refit the critic to each code (one weight per unit inside the cosine; as trained, and with position and anatomy removed); rerunning the audit on a stored JSON adds them without redoing the rest |
| `cca_readout.py` | CCA codes (top 14 or 24 canonical directions between T1 and FLAIR features, fitted on training subjects) from the pooled features and GAP, scored the same way; writes bundles |
| `causal_bundle.py` + `causal()` | label-free (`eval.causal.latent_causal_discovery`) and decoded (`eval.protocol.compare_bundles --with-graph`) discovery against the true graph |
| `ica_simulation.py` | no images, no training: the bank's SCM mixed into a 24-unit code; whether FastICA recovers the own parts (by noise distribution, link and harmonic strength) and whether DirectLiNGAM finds the order from the factors and from a mixed code. Item 33 |
| `heterogeneity_ica.py` | no images, no training: item 33's set-up with each own part's variance changing across the levels of an observed variable; separation by joint diagonalisation of the levels' covariances, against FastICA, PCA, random rotations and a supervised ceiling, and whether the anchors' loadings on the separated components place each own part. Item 38 |
| `adni_heterogeneity.py` | no images: on ADNIMERGE's FreeSurfer volumes (baseline visits, per FreeSurfer version), how much their variances change across age bins or sex and how stably that separates components, against shuffled labels. Item 38 |
| `factoriser.py` | no training: the concealed-factor benchmark of `paper/factor-discovery-framework.tex` on a run's frozen code and its untrained twin. Anchored slots (ridge readouts of the `--anchors`), a label-free count of the novel shared factors (cross-view agreement of the code's residuals, greedy non-linear deflation, a redundancy check), their naming against the concealed factors' own parts and the lesion centre (CCA order, ICA, random rotations), and PC on the anchors and the components against what PC can find among the anchors. `--anchor-source measured` subtracts the anchors' measured values instead of the readouts. A trace shows each direction the count examined with its correlation to each concealed target, and each target's R² from the dropped directions. Stages 42 and 44 |
| `bridge_view.py` | no training: the first N code units of a `--entropy-units` run (the bridge) written out as a run of their own, with TailCode's head cut to those outputs and `--content-units N` in its report, so every audit reads them unchanged. Stage 46 |
| `rotated_bundle.py` | no training, no labels: a causal bundle's codes rotated into their cross-view canonical directions (CCA between the views, the pairs beyond a shuffled-pairing null kept), written as a bundle of its own for label-free discovery. Tests: `tests/test_rotated_bundle.py`. Stage 51 |
| `footprints.py` | no training, and no labels in the footprints or the rotation: where in the image each direction of a run's code reads. Per 16³ cell, the units × units Gram (summed products) of the code's input gradient, each voxel weighted by the subject's deviation from the mean image (`--weight sensitivity`: the plain gradient), on 300 training subjects. Then the rotation of the code whose directions read from the fewest places: the joint diagonalisation (Jacobi angles) of the cell Grams whitened by their sum, pooled over subjects and per subject. Labels only score: names (one Hungarian match on absolute correlation with the factors and the lesion centre; MCC against the code's units, random rotations and the twin), readout footprints, model-free image maps (each factor's own part against the scan's intensity), the divergence from the code's mean footprint, and brain-interior and residual-channel shares. Figures, maps, and bundles of the code and each rotation for label-free discovery. Tests: `tests/test_footprints.py`. Stage 54 |
| `intrinsic_dimension.py` | no training, and no labels in the estimate: TwoNN (with decimation to N/2, N/4, N/8) and Levina–Bickel on each code, the code on the unit sphere, the backbone GAP and the twin's, beside the true latents, random non-linear embeddings of them with reading noise, and Gaussian noise of each width. Tests: `tests/test_intrinsic_dimension.py`. Stage 54 |
| `criterion_check.py` | no training: label-free signals of every finished run (canonical correlations held out, their mutual information and gain over the twin, retrieval, memorisation, neighbour agreement, spectrum, TwoNN, the GAP's, final loss) against its own parts; per bank, Spearman with a bootstrap interval and the share of a random pick's regret removed, with a pass rule fixed beforehand. `--self-test`. Stage 66 |
| `anchor_check.py` | no training: `export` writes each run's test codes, latents, lesion volumes and own parts to `causal/codes/<run>.npz` (GPU, once); `score` (no torch) picks runs within each bank by how well their codes read noisy measured anchors (raw, or adjusted for the other anchors) and judges the pick on the hidden factors' own parts, beside the exact own parts and stage 66's label-free signals. `--self-test`. Stage 67 |
| `adni_bank.py` | no training: ADNI's paired scans as a bank for `warm.py --real`, with per-subject QC, the exclusion rule and scan identifiers in `manifest.csv` (section 4). Stage 68 |
| `remainder_price.py`, `batch_price.py` | InfoNCE's reward for adding a unit, priced on true latents with no training |
| `normative_components.py` | no training, no network: the image filter's reading of burden's own part from the residual with 0 to 128 principal components removed (exact PCA), per view, with the leftover spread, the lesion's contrast over it, and the views' agreement. Stage 37 |
| `eval.causal.tabular_benchmark` | synthetic tabular tasks on the bank's test subjects: the share of image-driven confounding a vector removes (AIPW), and PC's recovery of edges out of the image factors (MRI→tab) and into them (tab→MRI), with orientation. Takes `causal_bundle.py` bundles, which carry each factor's own noise (`z_own`). `--proximal` (with `--view each`) adds both views in one AIPW and proximal two-stage least squares with one view instrumenting the other. `--counterfactual` adds tasks whose effect varies with the hidden image score (and, with `--counterfactual-modifiers score own`, one task per factor whose effect varies with its own part), and scores each arm's per-subject effects (DR- and T-learner): R² of the individual effects, PEHE, policy regret and the counterfactual outcome's error. Section 7, items 24 and 37; stage 55 |

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
      8.9 / 8.0 (24 units 7.6 / 7.2, control 8.7 / 8.1). Yet a fresh 24-unit bottleneck reads burden from the 10-unit
      backbones nearly as well as from the 24-unit ones (item 26): the head drops it.
    - Proposed why: the width matches the code, not the target. Each code predicts the other view's 64 whitened GAP
      features linearly, and the GAP spreads the 10 latents over more than 10 linear directions (CCA lost burden at
      14, item 9). Ten units predict only a 10-direction slice, and burden's own part is not in it. The precision
      readout takes the better of ridge and boosting, so the 10 units don't hold burden non-linearly either.
      **Stage 36's check rules this out (item 26): a non-linear map does not help.**
    - Price: within each run, a burden unit pays only below ~6–8 voxels at 10 units (cross-view r ~.86–.92), against
      ~8–12 at 24. One unit is a bigger share of a narrow cosine. Critic weights remove the cost, as before.
    - With a smooth function of position and anatomy removed, the 10-unit codes tell 32 subjects apart far less well
      (uniform loss 1.2 / 2.3 / 1.6 against .37 / .98 / .35; control, weight 1, gain off). They hold less beyond the
      latents, as a matched width should, but burden went too. The views share more of the code's burden error
      (r .64–.75 against .57–.68).
    - **Reading:** keep 24 units with prediction weight 1. An 11-unit arm (the cosine's lost degree of freedom,
      section 9) would only close the theory's formal case: the 24-unit InfoNCE code had far more room and read burden
      the same.
26. **A non-linear map does not let 10 units hold burden (stage 36's no-training check, read 16 Nov,
    `causal/stage36/bottleneck_check.txt`).** Fresh bottlenecks shaped like TailCode's head, trained only to predict
    the other view's whitened GAP from the frozen features of stage 33's and 35's weight-1 runs, three seeds each, read
    on test subjects. Volume error, T1 / FLAIR: the run's own code; linear map at 10 units; MLP map at 10; linear at
    24; MLP at 24.
    - Field off, 24-unit backbone: 12.7 / 13.2; 24.7 / 31.2; 22.0 / 26.4; 13.6 / 13.6; 14.4 / 14.6.
    - Field off, 10-unit backbone: 17.3 / 17.6; 16.7 / 16.3; 18.1 / 18.7; 13.6 / 13.8; 14.8 / 15.4.
    - Gain off, 24-unit backbone: 8.9 / 9.2; 18.0 / 25.2; 20.0 / 22.9; 9.3 / 9.4; 10.7 / 10.7.
    - Gain off, 10-unit backbone: 14.9 / 14.3; 16.3 / 15.0; 15.9 / 16.3; 10.6 / 11.1; 10.6 / 11.0.
    - The linear 24-unit bottleneck reads like each run's own 24-unit code, so the fit is sound.
    - At 10 units the MLP map reads burden no better than the linear one (within 5 voxels either way, no consistent
      sign) and 5–13 voxels worse than 24 units, on every backbone. What it buys is lesion position (boosted R²
      .92–.94 against .74–.89) and a lower prediction loss on validation subjects, lower even than the linear map at 24
      units (.14–.25 against .17–.28). The prediction loss values the non-linear structure of position and anatomy
      above burden's own part, and spends a narrow code on it: the width binds, not the map.
    - At 24 units the MLP map reads burden 0–1.6 voxels worse than the linear one.
    - The 10-unit backbones still hold burden: a fresh 24-unit bottleneck reads them about as well as the 24-unit
      backbones (field off 13.6 / 13.8 against 13.6 / 13.6; gain off 10.6 / 11.1 against 9.3 / 9.4). The 10-unit head
      drops it.
    - The check has no InfoNCE, and the runs' own 10-unit codes read about as the linear 10-unit bottleneck does. So
      InfoNCE's charge (the critic-weight arm's target) is not what keeps burden out of 10 units either.
    - **Reading:** the gate fails, so stage 36's arms are unlikely to help: stop them (section 9). Keep 24 units.
      Every objective tried ranks burden's own part last, so it needs spare width; no change to the map will fit it
      into 10 units. One cheap lever is left, the target's weighting (section 10), and the theory note is in
      section 8.

27. **At 10 units the lever is how sharply the views show burden, not the objective (16 Nov; a simulation and one
    local no-training check).** The user then wanted 10 units, the matched width (dropped 18 Nov, section 1).
    - **Simulation.** `bottleneck_check.py`'s fit on made-up features: nine sharp shared latents, one faint one (58%
      explained by a parent and read with noise in each view), and harmonics of the sharp ones in the targets. A
      harmonic is a non-linear function of a sharp latent, such as a place field over position. Tuned until it showed
      item 26's pattern: 10 units drop the faint latent through either map, 24 linear units keep it. Two seeds. The
      score is the R² of the faint latent's own part read from the code; the full features give ~.67. The scripts
      were one-offs in a session scratchpad and are not in the repo.
      - **Fails at 10 units (≤ .02 unless said):** whitening ridge .001, section 10's lever; a bigger map; a penalty
        on units that the other units predict; InfoNCE alone with 31 or 1,999 negatives, cosine or distance
        similarity, one weight per unit, or 11 cosine units; training-only extra units tied to the 10 (the faint
        latent leaks into them); a wide code narrowed to 10 by rebuilding it (kept in 1 of 4 runs).
      - **Why.** The faint latent is worth ~3% of the prediction loss. A hand-built code of the ten true latents
        scores a worse loss (.175) than the learned code without the faint one (.170): the map cannot rebuild every
        harmonic from 2,000 subjects, so the tenth unit carries harmonics. Ten times the subjects bring part of it
        back (.21–.32).
      - **Width.** The MLP map needs 11–12 units there, the linear map more than 18. The real check (item 26) ran
        only 10 and 24.
      - **What moves it at 10 units:** how exactly each view determines the latent. Own part 0 as above; .38 when it
        is read as sharply as the anatomy; .65 of a possible .98 when it is also independent of its parent.
      - **Partial observability** (Yao's theorem 3.8: a residual-only pair of views with a 4–5-unit block of its own
        inside the 10): .04–.18 only. Position's harmonics outbid the faint latent even in the small block.
    - **Burden is not unrecoverable.** The labelled code reads its own part to 5.9 / 4.7 voxels (item 18), 91% / 94%
      of an SD of 19.9; the fixed image filter is not the ideal reader. The theory assumes each view determines the
      content exactly. Burden's own part meets that less well than any other latent, so it is the one dropped at the
      matched width.
    - **A richer normative model sharpens the residual** (`normative_components.py` on a Mac copy of
      `bank_causal_nofield`, rendered with the same options, 1,000 test subjects, exact PCA). Image-filter error in
      voxels, T1 / FLAIR, by principal components removed:
      - 0: 10.2 / 8.0. 16, the residual so far: 9.0 / 6.6, the PC's numbers. 32: 7.4 / 5.2. 64: 6.7 / 5.9.
        128: 7.4 / 7.2. 512: 9.3 / 9.3.
      - The views' readings agree at r .93 with 32 or 64, against .91 with 16.
      - Past ~64 the components remove the lesion itself: its contrast over the leftover falls.
      - With the field on it does not help: on the independent `bank`, T1 reads 14.8 at 16 and worse past 64 (300
        test subjects). The field, not leftover anatomy, limits that bank.
      - The filter gains 1.6 / 1.4 voxels at 32. Gain off gave it 1.0 / 0.4 and moved the 10-unit code by ~2.5–3
        (item 25). Whether the network follows is stage 37's question.
    - **Reading:** stage 37 (section 9). Benchmark-side alternatives if it falls short: a bank where burden's own
      part is larger (the SCM's own-noise weight is 0.4 for every child; on an independent field-off bank an
      11-voxel reading recovers ~93% of ±41 voxels), or several lesions per subject, which adds three position
      latents per lesion to the matched width.
28. **A sharper residual reaches the backbone and the 24-unit code, not the 10-unit code (stage 37; precision audit
    read 17 Nov, the other reports not yet).** 32 principal components removed instead of 16, prediction weight 1,
    20k steps, seed 42. Volume error in voxels, T1 / FLAIR, 16 → 32 components:
    - **Field off, 10 units.** Image filter 9.0 / 6.6 → 7.4 / 5.2 (as item 27 predicted). Backbone GAP 11.1 / 11.9 →
      9.7 / 9.4, pooled 11.4 / 11.4 → 9.9 / 10.0. Code 17.3 / 17.6 → 17.0 / 17.3: unchanged. "Alone" .42 / .39 →
      .52 / .52. Largest third 29 / 32 → 33 / 41. Position and anatomy left in the code (mean ridge R²) .70 / .62 →
      .60 / .54.
    - **Gain off, 10 units.** Image 8.0 / 6.2 → 6.5 / 4.8. GAP 8.9 / 8.0 → 9.1 / 7.9: no gain. Code 14.9 / 14.3 →
      16.6 / 16.7: worse. "Alone" .56 / .60 → .52 / .53. Position and anatomy .74 / .61 → .65 / .55. The 10-unit seed
      spread is unmeasured (24 units: 0.4–0.8 voxels, item 23), so part of this may be seed.
    - **Field off, 24 units.** GAP 10.1 / 10.9 → 9.5 / 9.4. Code 12.7 / 13.2 → 12.4 / 10.8, the best field-off code
      so far. "Alone" .61 / .61 → .72 / .76. Largest third 18 / 20 → 16 / 14. Position held (.89 → .90); anatomy fell
      (.87 → .72).
    - **The 10-unit code stays 6–9 voxels behind its own GAP in all four 10-unit runs** (code minus GAP: field off
      +6.2 / +5.7 → +7.3 / +7.9, gain off +6.0 / +6.3 → +7.5 / +8.8). The 24-unit codes stay 1.4–2.9 behind. Against
      the image filter the four 10-unit codes give no curve: image T1 9.0, 8.0, 7.4, 6.5 → code 17.3, 14.9, 17.0,
      16.6.
    - **Price.** A burden unit read to 10 voxels still costs the 10-unit codes' InfoNCE (−.010 to −.012 nats on uniform
      sets), as before; +.002 in the 24-unit code.
    - **Reading.** Item 27's lever fails at 10 units: within the sharpness reachable here (image 6.5–9, GAP 8–12
      voxels) the 10-unit code keeps a coarse reading of ~15–17 voxels, own R² ~.57–.66, whatever its backbone holds.
      The simulation needed far sharper views (the latent as sharp as the anatomy) before 10 units kept it. At 24
      units the sharper residual helps burden and costs anatomy; own variation will say which factors.
29. **The 32-component residual removes focal anatomy, and a 10-unit code spends half its directions on the lesion's
    position (stage 37's own variation and code audit, read 17 Nov; the 16-component 10-unit arms of stage 35 are read
    here for the first time).** T1 / FLAIR.
    - **Burden's own part in the code**, 16 → 32 components (backbone GAP in brackets): field off, 10 units .33 / .34
      (.54 / .53) → .40 / .39 (.64 / .62); gain off, 10 units .44 / .46 → .43 / .44 (GAP .64 both); field off, 24
      units .48 / .47 (.60 / .59) → .54 / .57 (.64 / .62).
    - **What the 32 components cost.** Temporal atrophy's own part: 24 units .84 / .81 → .38 / .39 (GAP .85 / .83 →
      .66 / .65); 10 units, field off .78 / .78 → .08 / .07, gain off .80 / .79 → .07 / .07. Thickness at 24 units
      .63 / .64 → .25 / .27; sulcal .52 / .60 → .11 / .11. Temporal atrophy is a compact regional change, so the
      larger normative model absorbs it and the residual stops showing it. The 32 components sharpen the lesion by
      removing other focal factors: the opposite of a generic lever. The residual stays at 16.
    - **10 units drop more than burden.** Own parts in stage 35's 16-component 10-unit code, field off: brain
      .66 / .64, ventricle .29 / .30, burden .33 / .34, thickness .20 / .19, temporal .78 / .78, lr .79 / .80, sulcal
      .07 / .07. At 24 units (stage 33): .95 / .94, .71 / .68, .48 / .47, .63 / .64, .84 / .81, .89 / .90, .52 / .60.
      With gain off, 10 units: ventricle .42 / .40, burden .44 / .46, thickness .12 / .14, sulcal .12 / .11.
    - **Code audit of the four 10-unit codes.**
      - Every unit is shared: matching units agree at r .94–.99, the effective rank is 9.3–9.8 of 10, and all 10
        directions have cross-view r ≥ .9.
      - Each direction is mostly a function of the true latents (boosted R² from anatomy and lesion together .70–.96,
        mean .88–.89).
      - The lesion block (centre and burden, 4 latents) dominates. A direction's mean boosted R² is .53–.61 from the
        lesion latents and .26–.34 from the anatomy; 4–6 of the 10 have anatomy R² ≤ .13.
      - The codes are non-linear in the factors: boosting reads more than ridge (field off, 16 components: lesion x
        .61 → .79, brain .66 → .82; 32 components: x .36 → .71). The 24-unit codes are the reverse.
      - Cross-view top-1 among all 1,000 test subjects: 10 units .974–.993, 24 units .937–.966. The narrower code tells
        subjects apart better while holding less.
    - **Reading.** The matched-width code is full of shared, latent-determined content, as the theory asks. But about
      half of it goes to the lesion's position, a near-unique fingerprint, and the rest cannot hold the anatomy
      children's own parts and burden's. Linear readouts (the tabular benchmark, PC's Fisher-Z) under-read these codes'
      position and raw factors; the own parts are not hidden non-linearly (item 30, own variation by boosting).
30. **A 10-unit code reads the lesion's volume to 16–19 voxels on both banks; 11 cosine units gain a little (stage 38's
    precision audit, own variation and code audit, read 17 Nov; the style audit not yet).** Volume error in
    voxels, T1 / FLAIR; the code's loss against the run's own pooled features in brackets.
    - **Causal bank, field off.** 10 units (stage 35): 17.3 / 17.6 (+5.9 / +6.2), "alone" .42 / .39, largest third
      29 / 32. 11 units: 16.4 / 15.1 (+5.6 / +4.2), "alone" .48 / .50, largest third 24 / 25. 24 units (stage 33):
      12.7 / 13.2 (+2.5 / +3.0). At 11 units the code keeps a little more anatomy (mean ridge R² .62 → .67) and a
      little less lesion position (.70 → .67).
    - **Seeds.** A second 10-unit seed reads 16.8 / 16.7, "alone" .39 / .39, largest third 36 / 33: the 10-unit seeds
      differ by 0.5 / 0.9 voxels. 11 units beats both 10-unit seeds in FLAIR (by 1.6–2.5 voxels), in "alone"
      (.48 / .50 against .39–.42) and on the largest lesions (24 / 25 against 29–36 / 32–33). In T1 it is 0.4–0.9
      better, about the seed spread. One 11-unit seed only.
    - **Independent bank, field off.** Burden's own part is its whole spread (SD 41.3); the other latents explain none
      of it. Image filter 11.0 / 8.2 and pooled features 11.0 / 10.0 in both runs. 10 units: 19.5 / 18.9
      (+8.5 / +8.9), own R² .82 / .83. 24 units: 12.5 / 11.9 (+1.5 / +1.4), own R² .92 / .92. Lesion position and
      anatomy left in the code (mean ridge R²): .56 and .69 at 10 units, .78 and .85 at 24.
    - **Reading.** Stage 38's second question is answered: the 10-unit limit is the precision of the lesion reading,
      not how faint burden's own part is. On both banks a 10-unit code reads the volume to 16–19 voxels and a 24-unit
      code to 12–13, while their backbones read 10–11. The causal bank's small own part (SD 20) makes that limit
      costly: the same error is own R² ~.57 there and .82 on the independent bank.
    - **Reading, 11 units.** The cosine's lost dimension (section 8) costs a little. 11 cosine units, the theory's
      formal matched width, read burden 1–2 voxels better than 10 and stay ~3 behind 24.
    - **Own variation, causal bank** (each factor's own part in the code by ridge, T1 / FLAIR; 10 units seed 42,
      seed 2, 11 units, 24 units):
      - brain .66 / .64, .78 / .79, .78 / .80, .95 / .94;
      - ventricle .29 / .30, .30 / .28, .38 / .38, .71 / .68;
      - burden .33 / .34, .27 / .27, .38 / .38, .48 / .47;
      - thickness .20 / .19, .20 / .20, .18 / .16, .63 / .64;
      - temporal .78 / .78, .71 / .72, .79 / .78, .84 / .81;
      - lr .79 / .80, .68 / .65, .79 / .82, .89 / .90;
      - sulcal .07 / .07, .07 / .06, .11 / .10, .52 / .60.
    - Boosting reads the own parts no better than ridge (burden .29–.31 at 10 units, .36–.37 at 11, .41–.43 at 24;
      the true factors read the same way reach .95–1.00). The narrow codes do not hide the own parts non-linearly.
    - **Reading, own variation.** At 10 and 11 units the code keeps brain, lr and temporal atrophy and drops the own
      parts of ventricle, thickness and sulcal as well as burden's; 24 units keeps them all at .47–.95. The 11th unit
      lifts ventricle (.29 → .38, boosted .20–.24 → .36–.39, beyond the 10-unit seeds' spread) and burden a little
      (.27–.34 → .38), not thickness. Matched width at this sample size keeps the strongest factors and drops the
      weaker own parts, not burden's alone.
    - **Code audit.** Every 10- and 11-unit code is fully used (effective rank 9.3–10.5), every direction is shared
      (cross-view r ≥ .93), and each is mostly a function of the true latents (mean boosted R² .87–.89).
      - On the independent bank the 10-unit code's directions split about evenly between the lesion (mean boosted R²
        .42) and the anatomy (.45); on the causal bank .54–.56 and .32–.34. The 24-unit code there splits .39 / .36.
        The lesion block there includes a large burden factor, and position reads lower than on the causal bank
        (T1 ridge x / y / z .54 / .50 / .64, boosted .72 / .65 / .73).
      - 11 units keeps the causal bank's split (.56 / .32). The extra unit raises ventricle (T1 ridge .57–.70 →
        .74) and identification (top-1 among 1,000 test subjects .978 → .990), not the anatomy's share.
      - The two 10-unit seeds keep different details: brain .66 against .78, ventricle .57 against .70, lr .79
        against .68, lesion x .61 against .35 (T1 ridge). At matched width the choice among competing factors is
        close to a tie, and the seed decides it.
    - My interpretation: on the independent bank the narrow code loses more against its backbone (+8.5 voxels
      against +6), though burden is larger there. Its ten latents are independent, so none can be read partly from
      another, and the narrow code has less to share.
31. **Predicting the other view's tail features does not help a 10-unit code (stage 39's no-training check, read 17
    Nov).** Fresh bottlenecks on the frozen features of stage 33's and 35's weight-1 runs, three seeds each, read on
    test subjects. Volume error in voxels, T1 / FLAIR, with the GAP / tails / pooled target:
    - Field off, 10-unit backbone. MLP map at 10 units: 18.1 / 18.7, 23.2 / 22.3, 23.1 / 20.8. Linear map at 24
      units: 13.6 / 13.8, 20.0 / 21.6, 16.9 / 18.0.
    - Field off, 24-unit backbone. MLP 10: 22.0 / 26.4, 26.0 / 23.2, 22.4 / 25.6. Linear 24: 13.6 / 13.6,
      20.0 / 19.8, 17.2 / 18.6.
    - Gain off, 10-unit backbone. MLP 10: 15.9 / 16.3, 17.5 / 19.5, 18.4 / 18.7. Linear 24: 10.6 / 11.1,
      17.0 / 17.0, 14.6 / 14.1.
    - Gain off, 24-unit backbone. MLP 10: 20.0 / 22.9, 20.5 / 21.3, 19.9 / 22.7. Linear 24: 9.3 / 9.4,
      18.6 / 19.5, 15.9 / 17.9.
    - The tails read burden worse than the GAP at both widths and lose the anatomy: mean boosted R² .23–.52 against
      .70–.85 with the GAP; temporal −.04 to .36, lr .19–.64, sulcal below 0. The pooled target sits between them.
      Lesion position stays: the MLP map at 10 units reads it at boosted .91–.95 with the tails, as with the GAP.
    - The targets' spread: the 10-unit runs' GAP spreads over ~7 directions, the 24-unit runs' over ~12, and the
      tails over 6–7 in every run. Training end to end at 10 units narrows the backbone's GAP (item 25's lost backbone
      gain).
    - My interpretation: the tails are not position-free. TailCode z-scores each cell with that cell's own running
      statistics, so how far a lesion's cells exceed a threshold depends on where it sits, and the tails carry
      position as much as volume. The same per-cell normalization removes the global anatomy the GAP carries.
    - **Reading:** the gate fails; the tails target is dropped.
32. **A larger head does not let 10 units match 24 (stage 40's no-training check, read 17 Nov).** Fresh bottlenecks
    on the same four runs, GAP target, three seeds each, with TailCode's head (one hidden layer of 100) or a larger
    one (one or two hidden layers of 512). Volume error in voxels, T1 / FLAIR, heads 100 / 512 / 512x2:
    - Field off, 10-unit backbone. Linear map at 10 units: 16.7 / 16.3, 16.4 / 16.0, 16.0 / 16.0. MLP map at 10:
      18.1 / 18.7, 17.8 / 18.1, 19.1 / 18.3. Linear map at 24: 13.6 / 13.8, 13.5 / 13.4, 14.5 / 14.4.
    - Gain off, 10-unit backbone. Linear 10: 16.3 / 15.0, 13.1 / 15.0, 13.1 / 14.1. MLP 10: 15.9 / 16.3,
      15.6 / 17.2, 17.3 / 17.9. Linear 24: 10.6 / 11.1, 10.4 / 11.1, 11.0 / 11.9.
    - Field off, 24-unit backbone. Linear 10: 24.7 / 31.2, 20.0 / 19.2, 19.8 / 18.8. Linear 24: 13.6 / 13.6,
      13.4 / 14.1, 13.6 / 14.9.
    - Gain off, 24-unit backbone. Linear 10: 18.0 / 25.2, 24.8 / 20.5, 19.1 / 19.2. Linear 24: 9.3 / 9.4,
      9.2 / 9.6, 10.2 / 10.4.
    - On the 10-unit backbones a larger head gains 0.3–0.7 voxels with the field off, and up to 3.2 in T1 with gain
      off, through the linear map only; the MLP map gains nothing, and at 24 units nothing changes. On the same
      features 10 units stays 2.5–3 voxels behind 24 (field off 16.0 / 16.0 against 13.5 / 13.4; gain off
      13.1 / 14.1 against 10.4 / 11.1).
    - On the 24-unit backbones the larger heads mostly remove failed seeds (TailCode's head at 10 units had seed
      spreads up to 16–19 voxels there) and stay at 19–25 voxels.
    - **Reading:** the gate fails; the head's capacity is not what limits the narrow code. With stages 35–39 this
      ends the search for a 10-unit lever (section 8).
33. **ICA recovers the own parts from a near-linear code only under non-Gaussian noise, and does not give the causal
    order (`ica_simulation.py`, no images, run 17 Nov).** The bank's SCM sampled directly (graph, weights, noise scale
    0.4); the seven factors and the lesion centre standardized and mixed into a 24-unit code with reading noise .3;
    five seeds. FastICA to ten components, matched to the true sources by one Hungarian match. Mean R² of the seven own
    parts, as ICA / PCA / random-rotation null (95th percentile) / supervised ceiling:
    - Gaussian noise, as the bank draws it: .35–.36 / .38–.41 / .28–.29 (.46–.48) / .78–.80. No better than PCA.
    - Non-Gaussian noise with the bank's leaky-ReLU link: Laplace .61, uniform .80, skewed .69, against a ceiling of
      .78–.81. With a linear link: .69–.72 against .74. The lesion centre (uniform, so non-Gaussian in every
      condition): .94–.95.
    - Harmonic directions in the code break it: six squares and products at weight .3 give .30–.64; at weight 1, about
      the null (.17–.26).
    - DirectLiNGAM finds the causal order from the factors themselves (non-Gaussian, linear link: skeleton and
      directed F1 .98–1.00; leaky ReLU .78–.87 and .55–.87; Gaussian .71–.79 and .16–.31) but not from a random
      orthogonal mix of them (.36–.46 and .16–.26 for every noise). A code identified only up to its block leaves the
      factors' coordinates unknown: ICA returns the own parts, not the variables LiNGAM needs.
    - **Reading:** an ICA factoriser could give label-free own parts if the exogenous variation is non-Gaussian (real
      lesion load is skewed) and the code is near-linear in the factors (the 24-unit prediction code reads them
      linearly; the narrow codes hold harmonics). The bank's Gaussian noise cannot test it on images; a non-Gaussian
      noise option in the generator would. The graph among the factors still needs anchors or another constraint on
      the mixing (the partial anchors of `paper/factor-discovery-framework.tex`).
34. **Matched width does not make the units causal variables, and every causal task is worse at matched width
    (stage 41, read 17 Nov).** Causal bank, field off, prediction weight 1. View 1 (T1) unless said.
    - **Label-free discovery** (PC on each code's own units, named by one Hungarian match; F1 at each row's best
      alpha, optimistic). 10 units, seed 42: .33, seed 2: .20; 11 units: .40; 24 units: .00. Random mixtures of the
      same units: .33–.39 (± .12–.19). Truth: .83. Every code sits at or below the random-mixture baseline, and below
      its own untrained floor (.20–.56). No unit is a factor: the mean matched |r| is .40–.46 at every width
      (10 units, seed 42: brain .47, ventricle .38, burden .26, thickness .39).
    - **Decoded discovery** (PC on the seven factors read from each code by a held-out supervised readout).
      - Skeleton F1 over 20 resamplings: 10 units .69 ± .08 (seed 42) and .87 ± .07 (seed 2); 11 units .75 ± .07;
        24 units .84 ± .04; truth .85 ± .03. Orientation against the true CPDAG, SHD: 10 units 5 and 5, 11 units 7,
        24 units 3, truth 3.
      - Mean factor R² over the permutation null: .63, .67, .68 and .87.
      - The signal each edge rests on (linear-residual partial R²), 10 units / 11 / 24: ventricle .37–.38 / .44 /
        .76, burden .28–.32 / .35 / .58, thickness .19–.25 / .13 / .72, sulcal .06–.07 / .10 / .58. Brain, temporal
        and lr are .68–.96 everywhere.
      - The skeleton differences between widths are within the resampling noise (6 true edges); the partial R² is
        the sensitive measure.
    - **View 2 (FLAIR)** repeats this. Decoded skeleton F1 over resamplings: 10 units .69 ± .08 and .80 ± .06, 11 units
      .80 ± .07, 24 units .85 ± .05 (truth .85 ± .03); CPDAG SHD 5 and 7, 6, and 3 (truth 3); the same weak partial R²
      at 10 and 11 units (ventricle .36–.48, burden .30–.37, thickness .14–.27, sulcal .05–.08). Label-free, the
      narrow codes are again at or below their random-mixture baseline (.29–.40 against .35–.40). The 24-unit code
      scores .75 against .34 ± .20, but .00 in view 1. Only 7 of its 24 channels are named, most recovered edges fall
      outside the score, and the best channel for each factor still correlates only .34–.54 with it. So its
      label-free F1 is too unstable to read, and no channel is a factor at any width.
    - **Tabular benchmark** (10 tasks, item 24's measures), T1 / FLAIR:
      - Confounding removed (of the tabular-only bias .597): 10 units 58 / 57% (seed 42) and 57 / 56% (seed 2); 11
        units 58 / 60%; 24 units 79 / 83%. The backbone GAP of the 10-unit run 78 / 74%, of the 24-unit run 85 / 90%.
        Oracle 100%.
      - MRI–tab F1: 10 units .51 / .53 and .43 / .46; 11 units .55 / .50; 24 units .90 / .89; the backbone GAPs
        .81–.90; oracle .98.
      - tab→MRI F1 (a tabular cause reaches the image only through a factor's own part): 10 units .45 / .53 and
        .46 / .43; 11 units .51 / .37; 24 units .93 / .91; oracle 1.00. Oriented: 10 units .10–.35, 11 units
        .05–.25, 24 units .65–.70 (the oracle's ceiling .70).
      - The 24-unit numbers repeat item 24's for the same run. The 10-unit runs' own backbones remove 74–78% of the
        confounding and find the MRI–tab edges at .81–.89, so the narrow head, not the backbone, loses them.
    - **Reading.** Matched width does not do what the causal argument needs. The narrow code's units are mixtures,
      like the wide code's, and label-free discovery on them is at chance at every width, as two full views predict
      (section 8). With a supervised readout, the 24-unit code matches the ground truth's skeleton and orientation;
      the 10-unit codes depend on the seed (.69 or .87) and carry weak signals for four of the seven factors. Every
      causal task is markedly worse at 10 or 11 units than at 24: confounding removed 56–60% against 79–83%, tabular
      edges found at .43–.55 against .89–.90, orientation .05–.35 against .65–.70. The number of causal variables
      belongs in the factoriser, not in the encoder width (section 10).
35. **The label-free count does not isolate the weak own parts of a trained code (stage 42, read 17 Nov).** Causal
    bank, field off, prediction weight 1; `factoriser.py` (section 9). T1 / FLAIR. "Full code": a supervised readout
    of the target from the whole code; "block": from the counted components only (both for scoring only).
    - **Anchors** (step 1). 24 units: brain .95 / .94, ventricle .88 / .87, thickness .91 / .91, temporal .91 / .90;
      burden as an anchor .79 / .79. 10 units: .53–.81. Untrained twins: .11–.63.
    - **Count** (step 2; the truth is 6 in both anchor splits). Every untrained twin counts 0, so the count comes from
      training.
      - 24 units, default anchors: 6, but not the right 6. 18 directions taken, 12 dropped. Block against full code:
        lr .78 / .80 against .89 / .90, lesion centre .70–.79 against .88–.90, burden's own part .04 / .03 against
        .48 / .47, sulcal's .10 / .10 against .52 / .60. Two components match no target (ICA |r| .01 and .30).
      - 24 units, burden an anchor (thickness concealed): 9, three of them unmatched. Every target is diluted:
        thickness's own part .33 / .36 against .63 / .64, sulcal's .32 / .37 against .52 / .60, lr .63 / .64, lesion
        centre .57–.70.
      - 10 units: 6 and 5, forced by the width. Ten units minus four slots leave six directions ("no variance left").
        The code itself holds little of the weak own parts (burden .33 / .34, thickness .20 / .19, sulcal .07 / .07),
        and the block less (burden .19 / .18, thickness and sulcal .00); lr and the centre stay near the full code.
    - **Why the rule fails here.** On the synthetic codes the redundancy R² of counted and dropped directions stood
      apart (counted ≤ .18, dropped ≥ .29). On the 24-unit code it runs continuously across the .2 cut (counted
      .00–.19, dropped .20–.49), and the dropped directions agree at .55–.96. The likely source is what the linear
      slots miss of the anchors: the code holds the anchors more fully than a ridge readout captures (slot R²
      .79–.95). That remainder is anatomy, so both views share it, but the other view's slots predict it only in
      part. The weak own parts, which the SCM ties to anchors (burden to ventricle, thickness to brain, sulcal to
      temporal), leave with these leftovers or spread across them. The worse slot set (burden at .79) gave the worse
      count (9).
    - **Graph** (step 4). The components connect to nothing (edges touching them at or below chance), as section 8
      says. The edges among the 24-unit slots match PC on the true anchors in both splits; their F1 (.70 / .69 and
      .68 / .62, against .75 and .84) is lower only by chance edges to the components. The 10-unit slots and every
      untrained twin find edges the true anchors miss at 300 rows: the induced temporal–ventricle (.90–1.00, default
      anchors) and the weak burden–temporal (.25–1.00, burden an anchor). Slots that mix factors make edges, so their
      F1 can exceed the true anchors' own (10 units, default: .83 / .90 against .75). Read the edges against the
      true-anchor ceiling, not F1 alone.
    - **Reading.** The factoriser counts the shared content a code adds to its anchors' readouts, and in a trained
      code that includes partly redundant anatomy the redundancy test cannot place. It keeps the strong novel factors
      (lr, the lesion centre) at 70–90% of the full code and loses or dilutes the weak own parts, burden's among them.
      Untested: residualizing on the anchors' measured values (ADNI has them for every subject) instead of the code's
      readouts of them (section 10).
36. **The weak own parts are lost at the redundancy cut, not through the anchors' readouts (stage 44, read 18 Nov).**
    Stage 42's codes and anchor splits again, with the anchors' measured values in place of the code's readouts, and
    both arms traced: each direction's |r| with each target, and each target's R² from the dropped directions. T1 /
    FLAIR.
    - **The readouts arm reproduces stage 42 exactly.** Where the own parts went, 24 units:
      - Default anchors: burden's own part is .04 / .03 in the block and .21 / .23 in the dropped directions (full
        code .48 / .47). Its clearest direction agreed at .79 and was dropped at R² .34 (|r| .33 / .35). Sulcal's is
        .10 / .10 in the block and .39 / .45 dropped (full .52 / .60), mostly in one direction at .86, dropped at R² .26
        (|r| .53 / .55).
      - Burden an anchor: thickness's own part was counted, in one direction at .92 with R² .10 (|r| .48 / .50; block
        .33 / .36 of .63 / .64).
    - **Measured anchors do not keep them.**
      - 24 units, default: 7 components. Burden .03 / .03 in the block and .26 / .28 dropped (its direction at .78,
        dropped at R² .24); sulcal .18 / .17 and .31 / .36. lr and the lesion centre come out a little better than with
        readouts (lr .84 / .86, centre .69–.82).
      - 24 units, burden an anchor: 10 components, four unmatched. Thickness's own part is now dropped (its direction
        at .94, R² .21, |r| .51 / .54; block .02 / .04, dropped .34 / .34).
      - 10 units, default: burden's own part sits in one direction at .91 with |r| .50 / .52, dropped at R² .22 (block
        .04 / .05, dropped .31 / .38).
      - 10 units, burden an anchor: the one arm where every target comes out. 8 components, two unmatched; thickness's
        own part .25 / .24 and sulcal's .19 / .16, above the code's own linear readout (.20 / .19 and .07 / .07).
        Subtracting the exact parents uncovers own parts a linear readout cannot reach. The measured values supply
        that gain, so it is supervised, not label-free.
    - **Floors.** Readouts: 0 components in all four. Measured: 1 in all four (agreement .51–.63, R² .08–.12). In the
      24-unit floor that component correlates with burden's own part (.26 / .35): an untrained network sees the lesion,
      and once ventricle is subtracted exactly, burden's own part shows.
    - **Graph.** With measured anchors the edges among the anchors are the true anchors' own; the components add only
      chance edges (F1 .55–.69).
    - **Reading.**
      - The readouts are not the cause: measured anchors lose the same own parts, and thickness's as well.
      - Agreement is not the cause either: the lost own parts sit in directions agreeing at .78–.94, and the count
        stopped at directions that carry no target.
      - The cause is the redundancy cut acting on mixed directions. Each weak own part shares its direction with
        content the other view's anchors and counted factors partly predict (R² .20–.34 when dropped). The own part
        explains only 10–30% of that direction (|r| .33–.55), so the whole direction goes. Sharper anchors predict the
        other share better, which is why measured anchors drop thickness too.
      - The 24-unit code holds much shared content that is none of the targets: 10 of the 12 directions dropped in
        the default readouts arm correlate with no target at |r| ≥ .3, yet agree at .55–.94. Subtracting only the
        predictable part of a dropped direction would keep that content too and inflate the count.
      - Even all the shared directions together hold only about half of burden's own part at 24 units (block plus
        dropped, .25–.31 of .48); the rest lies in directions agreeing below .5. Sulcal's is mostly there (.49–.55).
      - So the label-free count cannot split the weak own parts out of this code's novel block, as section 8 expects
        (two views identify the block). A factor-level split needs more structure (step 3b) or anchors for those
        factors.
37. **A second view cannot close the code's adjustment shortfall: the confounding the code leaves, both views leave
    (proximal check, run 18 Nov, `causal/proximal/report.txt`; no training).** The tabular benchmark's effect task (item
    24) on stage 41's bundles with `--proximal`. Besides each view's vector, it adjusts for both views' vectors in one
    AIPW ("both"), and estimates the effect by proximal two-stage least squares (P2SLS; Miao et al. 2018, Tchetgen
    Tchetgen et al. 2020): one view's vector instruments the other's in a linear outcome model. Proximal inference
    treats the two views as two measurements of the anatomy whose errors are independent given it, so it removes the
    confounding that one view's reading error hides. Both vectors are first cut to the canonical pairs the views share
    beyond a shuffled-pairing null.
    - **Known answer** (no images): three latents, each view a noisy 6-unit mixture of them, true effect 1. One view
      leaves a bias of +.12 to +.19, both views in one AIPW +.01 to +.05, P2SLS −.01 to −.03; it finds the 3 shared
      directions every time (`tests/test_tabular_benchmark.py`).
    - **Confounding removed** (share of the tabular-only bias .597, 10 tasks). One view T1 / FLAIR; both views;
      proximal T1→FLAIR / FLAIR→T1:
      - 10 units, seed 42: 58% / 57%; 61%; 58% / 58%. Seed 2: 57% / 56%; 60%; 59% / 59%.
      - 11 units: 58% / 60%; 63%; 61% / 60%.
      - 24 units: 79% / 83%; 83%; 86% / 84%.
      - Backbone GAP of the 10-unit run: 78% / 74%; 77%; 80% / 80%. Of the 24-unit run: 85% / 90%; 93%; 92% / 87%.
      - True factors: bias −.001.
    - Every trained code unit is a shared direction beyond the null (10 of 10, 11 of 11, 24 of 24), so the cut removed
      nothing from the codes; the GAP keeps 33–35 of 64. Untrained floors share 4–11.
    - **Reading.** P2SLS gains at most 3 points over both views in one AIPW, and loses up to 6 on the GAP, where on the
      known answer it removed the whole leftover. So the leftover is not reading error independent between the views.
      Both views' codes omit the same content: they are trained to agree, and both drop the weak own parts (item 30).
      Proxies cannot restore what neither holds, so the adjustment ceiling is what the representation holds.
    - The 24-unit head drops about 10 points the backbone keeps (83% against 93% with both views' GAP). For adjustment
      (the plan's block node, section 10), both views' backbone GAP removes the most.
    - Proxies stay right for two measurements with independent errors, such as ADNI's FreeSurfer anchors against the
      code's readouts, if their errors are independent (both start from T1).
38. **Own parts separate by variance heterogeneity in simulation, but ADNI's age and sex do not supply it (run 18
    Nov; no images, no training).** ICA fails on the bank's Gaussian own parts (item 33). Gaussian sources become
    separable when their variances change, each differently, across the levels of an observed variable that does not
    descend from them: joint diagonalisation of the levels' covariances (second-order nonstationary separation, the
    linear case of time-contrastive learning and iVAE).
    - **Simulation** (`heterogeneity_ica.py`): item 33's set-up, with each own part's SD changing across 5 levels by
      exp(strength × a standard normal draw); the lesion centre does not change. Own parts' mean R², against a
      supervised ceiling of .80: strength 0, .27 (random rotations .29); .25, .72; .5, .76; 1, .77. FastICA .42–.61.
      - With harmonics in the code (weight .3): .57–.64, ventricle and thickness lowest (.24–.39).
      - Half the own parts varying: .46–.49. Two levels: .48–.54.
    - **Placement:** with the four anchors regressed on the separated components, burden's component loads on temporal,
      its only descendant among them, in every seed, but also on ventricle in 2 of 5. Imperfect separation leaves
      spurious loadings: precision .51–.64 (.47–.53 with harmonics) against 1.00 with the true noises.
    - **ADNI** (`adni_heterogeneity.py` on ADNIMERGE; baseline visits with all seven FreeSurfer volumes, one FreeSurfer
      version at a time, 565–716 subjects each): strength across 5 age bins .09–.15, against .08–.15 with age shuffled;
      by sex .06–.11 against .04–.07. No component is more stable under bootstrap than with the labels shuffled.
    - **Reading.** The method is sound, but ADNI's age and sex change the anatomy's variances about as much as chance,
      and the simulation needs ≳ .25. Diagnosis would be stronger, but it descends from the anatomy, so splitting on it
      couples the sources. Dropped unless a variable with a real, larger variance change turns up; check any candidate
      with `adni_heterogeneity.py` first.

37. **Stage 46's bridge collapsed: its loss had a degenerate minimum (stage 46, read 18 Nov,
    `causal/stage46/`).** The loss, not the bridge idea, failed.
    - **Result.** Own parts from the 24-unit code, T1 / FLAIR: ventricle −.01 / .00, burden −.01 / −.03, thickness
      −.16 / .04, temporal −.12 / −.01, lr .00 / .01, sulcal −.02 / −.02 (stage 33: .71 / .68, .48 / .47, .63 / .64,
      .84 / .81, .89 / .90, .52 / .60). Only brain size stays (.79 / .90). Its first 10 units hold nothing. Volume
      error >99 voxels (stage 33: 12.7 / 13.2). Style came in: gain by boosting .41 / .59, against about 0 in every
      other trained code. The backbone suffered too (GAP: lr .81 / .71, ventricle .41 / .18). The recipe's InfoNCE on
      the code ended at 4.21 (chance 6.93; stage 33: 0.04), and the bridge term at −11.9.
    - **Why.** Stage 46's form put the whole code's cosine over a log-sum-exp of the block's cosines that included the
      positive pair. The two do not cancel as in InfoNCE, so the loss has no floor at zero. Minimised over free codes
      (32 subjects, τ 0.1) it reaches about −15: the block's two views disagree (cosine −.33), the block shrinks to 5%
      of each subject's code, and a few outlier subjects carry its batch spread. Nothing then pays the code to tell
      subjects apart.
    - **Fix (18 Nov).** `bridge_infonce` is now the recipe's InfoNCE on the block plus alignment alone,
      (1 − cosine) / τ, on the other units. Every term rewards agreement and the loss is never negative. Over free
      codes it settles at 0 with the block's views agreeing (cosine 1.00). Alignment on the block plus alignment on
      the rest is alignment on everything, so section 8's argument holds. In a 400-step check of stage 46b's arm the
      recipe's InfoNCE on the code fell to 2.43 (stage 36's MLP arm 1.88, stage 46 4.76) and the code read volume at
      R² .71 / .69.
    - Stage 46's run keeps its name and its old loss; `--entropy-units` now means the fixed form.

38. **The theory's bridge holds the weak factors no better than a matched-width code (stage 46b, read 18 Nov,
    `causal/stage46b/`).** One seed. T1 / FLAIR.
    - **Own parts in the code.** The bridge's 24 units (InfoNCE on the first 11, alignment alone on the other 13,
      MLP map), then stage 36's 24-unit MLP arm (the same with InfoNCE on all 24), then stage 38's 11-unit code:
      ventricle .37 / .35, .74 / .73, .38 / .38; burden .38 / .37, .48 / .47, .38 / .38; thickness .34 / .37,
      .61 / .61, .18 / .16; temporal .64 / .64, .84 / .83, .79 / .78; sulcal .15 / .13, .56 / .55, .11 / .10. Brain
      .92 / .92 and lr .81 / .80 stay.
    - Volume error 17.3 / 17.1 voxels (largest third 37.5 / 38.0) against 11.9 / 12.2 (17.5 / 20.0) for stage 36's
      arm and 16.4 / 15.1 for the 11-unit code. The recipe's InfoNCE on the code ends at 0.50 (stage 36's arm 0.04);
      top-1 among 1,000 test subjects .950 (.986).
    - **The first 11 units** hold less than a standalone 11-unit code (ventricle .28, burden .25, temporal .55, brain
      .24 against .38, .38, .79, .78): brain size moved into the other 13 units. Effective rank 10.6 of 11.
    - **The 13 aligned-only units collapse.** The whole code's effective rank is 3.5 / 3.3 of 24 (stage 36's arm
      20.1 / 20.3): those units hold a few copied directions. Neither code carries style.
    - **The map is not the cause:** stage 36's MLP arm and stage 33's linear arm match at 24 units.
    - **Reading.** Alignment keeps the extra units content-only, as the theory says, but nothing pays them to hold
      anything new, so they collapse onto the easiest shared directions. At the theory's optimum they are redundant,
      and here they are. What buys the recipe's weak factors at 24 units is the entropy term acting on every unit,
      the part the theory does not cover (an aligned 24-unit code of 10 latents has no finite entropy). The formal
      bridge cannot carry the recipe's numbers (section 8).

39. **The lesion's position crowded the weak factors out of the narrow code (stage 47, read 18 Nov,
    `causal/stage47/`).** One seed per arm. The fixed-position bank holds the same subjects and anatomy as the
    drawn-position one, so the own parts are on one scale. T1 / FLAIR.
    - **Own parts in the code** (ridge): 10 units on the fixed bank, 8 units on the fixed bank, 10 units on the drawn
      bank (stage 35), 24 units on the drawn bank (stage 33). Ventricle .49 / .45, .49 / .48, .29 / .30, .71 / .68;
      burden .46 / .45, .41 / .40, .33 / .34, .48 / .47; thickness .59 / .65, .51 / .50, .20 / .19, .63 / .64;
      sulcal .50 / .54, .33 / .36, .07 / .07, .52 / .60; temporal .75 / .78, .72 / .73, .78 / .78, .84 / .81; lr
      .76 / .77, .74 / .76, .79 / .80, .89 / .90. Brain reads .44 / .49 and .57 / .56 by ridge in the fixed-bank
      codes but .83 and .86 by boosting (drawn: .82 at 10 units, .85 at 24): narrow codes hold it non-linearly.
    - **What the code holds** (code audit). Per direction, anatomy / lesion .81 / .19 at 10 units and .82 / .22 at 8,
      against .34 / .54 for 10 units on the drawn bank. Effective rank 9.3–9.4 of 10 and 7.5–7.7 of 8. Cross-view
      top-1 among 1,000 test subjects .71–.72 against .978: the drawn-bank code's near-perfect identification came
      from the lesion's position. Neither code carries style.
    - **The backbone moves too.** GAP own parts in the fixed-bank 10-unit run: thickness .75 / .76, sulcal .69 / .74,
      ventricle .81 / .73. Drawn bank, 10 units: .50 / .50, .36 / .56, .63 / .62; 24 units: .74 / .72, .66 / .75,
      .80 / .78. Narrow training on the drawn bank cost the backbone the weak factors as well as the head.
    - **Volume precision** (code): 11.6 / 11.7 voxels at 10 units and 12.3 / 12.4 at 8 (largest third 11.6 / 10.6
      and 13.3 / 12.4), against 17.3 / 17.6 (29.4 / 32.0) for 10 units and 12.7 / 13.2 for 24 on the drawn bank.
    - **Caveat on burden.** With the lesion always in about one place, the backbone reads it more easily (GAP 8–9.5
      voxels against 10–12), and the normative residual learns it as normal variation: the fixed image filter reads
      21.4 / 15.3 voxels against 9.0 / 6.6, so it is no ceiling on this bank. Burden's gain is partly easier reading.
      The anatomy gains are not: the images differ only within 8 voxels of a lesion, and the untrained floors are
      close.
    - **Reading.** A narrow code is fragile: one distinctive shared feature, here the randomly placed sphere's
      position, takes about half of it and crowds out the weaker factors, in the backbone too. Without it the theory's
      matched width (8 cosine units for 7 latents) keeps every factor's own part at .33–.51, and 10 units reach
      24 units' levels on thickness, sulcal and burden; ventricle stays below (.49 against .71). Spare width still
      helps (8 → 10 units: sulcal .33 → .50, thickness .51 → .59 / .65). The matched-width finding of items 26–32 is
      mostly this benchmark's fingerprint; ADNI has no randomly placed sphere.

40. **Without the position, a narrow code is a good block for discovery beyond the anchors, and needs a non-linear
    adjustment for confounding (stage 48, read 18 Nov, `causal/stage48/`).** One seed per code. T1 / FLAIR. Codes: 8
    and 10 units on the fixed bank; 10, 11 and 24 units on the drawn bank.
    - **Effect estimation, bias** (tabular-only .60, oracle .00), linear learner then `--learner rbf`
      (`tabular_rbf.log`, effects only): fixed 8 units .24 / .24 → .17 / .16; fixed 10 units .25 / .22 → .16 / .14;
      drawn 10 units .25 / .26 → .22 / .23; 11 units .25 / .24 → .22 / .21; 24 units .12 / .10 → .12 / .10. The
      fixed-bank codes hold most of the confounding content, but non-linearly (brain .44 by ridge, .83 by boosting);
      the drawn narrow codes lack content (crowded out), so a flexible learner barely helps them.
    - **Discovery with tabular variables.** Image–tabular edges F1: fixed 10 units .78 / .79, fixed 8 .76 / .64,
      drawn 10 .51 / .53, 24 units .90 / .89 (oracle .98). Tabular causes of image factors, which need the own parts:
      .87 / .87, .75 / .68, .45 / .53, .93 / .91.
    - **Discovery among the image factors** (decoded, resampled): every narrow code .69–.81, 24 units .84 / .85
      (truth .85); no narrow-code difference is resolved. Label-free PC on the units stays at the random-mixture
      baseline at every width (fixed 8 units .27 against .44).
    - **Concealed-factor benchmark, measured anchors.** Default split (burden, lr, sulcal concealed): the block keeps
      burden's own part .57 / .56 (fixed 10) and .54 / .55 (fixed 8), sulcal's .54 / .54 and .59 / .58; drawn 10 units
      .04 / .05 and .00, 24 units .03 / .03 and .18 / .17, their blocks taken by the lesion's position. Single
      directions match single concealed factors (fixed 10: lr .85, burden .73, sulcal .45–.52), named well above
      random rotations (.41). Burden split (thickness, lr, sulcal concealed): thickness .42 / .46 and sulcal
      .54 / .56 in the fixed 10-unit block; the 8-unit block keeps thickness (.37) but drops a sulcal direction at
      the cut (.17); drawn 10 units .25 and .19, 24 units .02 / .04 (thickness dropped at the cut) and .25. PC still
      places the components nowhere (their edges at chance), as section 8 expects.
    - **Reading.** Stage 44's loss of the weak own parts at the redundancy cut was mostly the lesion's position: with
      it fixed, the plan's block discovery (steps 3–4) finds the concealed weak factors as separate, nameable
      directions, even at matched width. Linear confounding adjustment still favours the wide code; with a flexible
      learner the fixed-bank narrow codes close most of the gap (.15 against .11). Burden's numbers carry stage 47's
      caveat (easier reading at a fixed place); sulcal's and thickness's do not.

41. **On the fixed-position bank the 10-unit code needs the residual, jitter and prediction, keeps its factors
    linearly readable with the tails, and is best with code batches (stage 50, read 19 Nov, `causal/stage50/`).**
    Leave-one-out from stage 47's 10-unit arm; one seed per ablation. T1 / FLAIR unless one number.
    - **Seed spread** (the reference at seeds 42 and 2), own parts by ridge: brain .44 / .68, ventricle .49 / .63,
      sulcal .50 / .64; burden, thickness, temporal and lr within .09. Novel-block burden .57 against .30. Narrow codes
      are seed-sensitive; the rule's thresholds were brain .24, ventricle .16, sulcal .14, the rest .10.
    - **Residual: needed.** Without it burden's own part is 0 in code and backbone alike, temporal .75 → .34, sulcal
      .50 → .32 (ventricle rises .49 → .71). ATE bias .34 / .35 (rbf .23 / .22) against .25 / .22 (.16 / .14). The
      novel block loses burden (0) and most of sulcal (.21).
    - **Jitter: needed.** Without it the code memorises (cross-view top-1 among 1,000 test subjects .39): burden 0,
      ventricle .13, sulcal .31 (brain rises to .83); the backbone loses burden and ventricle too.
    - **Prediction: needed.** Without it burden .25, thickness .43, sulcal .26, lr .58; volume error 18.3 / 16.6
      against 11.6 / 11.7; decoded image-graph F1 .49 / .53 against .70 / .74 (seed 2 .81 / .79). Its novel-block
      burden (.29) sits at seed 2's level, so that one measure does not separate it.
    - **Tails: they keep the narrow code linear** (corrected 19 Nov; first read as "not needed"). Own parts, volume
      error and the block stay within the seed spread (burden .46 → .58), and boosting reads the factors as before
      (brain .82, thickness .80, temporal .82, lr .84). But the factors themselves read far worse by ridge: ventricle
      .21 / .24 against .47 / .55 and .61 / .60 (seeds 42 and 2), thickness .34 / .40 against .45 / .63 and .64 /
      .59, brain .34 / .32 against .44 / .49 and .68 / .64, the lesion volume .48 / .47 against .70 / .72 and .76 /
      .79. Linear tools pay: ATE bias .28 / .28 against .25 / .22 and .20 / .20 (rbf .19 / .18 against .16 / .14 and
      .14 / .14). The rule fixed beforehand covered own parts, volume error and the block, not raw linear
      readability, so it missed this. At 24 units the tails make no such difference (item 42).
    - **Code batches instead of uniform: better on everything.** Own parts brain .95, ventricle .85 / .81, burden
      .60 / .58, thickness .75 / .76, temporal .85, lr .91 / .93, sulcal .72 / .77, all beyond the seed spread
      (temporal at its edge); volume error 11.3 / 11.5 (unchanged). ATE bias .09 / .07 (rbf .07 / .06), below the
      drawn 24-unit code's .12 / .10; image–tabular edges F1 .94 / .92 (oracle .98); decoded image-graph F1 .84 / .83
      (truth .85). The label-free count beside measured anchors keeps exactly three components, one per concealed
      factor (lr .86, burden .72–.73, sulcal .73). No style; cross-view top-1 .74 among 1,000; mild memorisation
      (code32 sets: train .97, test .91). Variance concentrates in a few directions (effective rank 2.2 of 10),
      which linear readouts do not mind.
    - **Why code batches now help when they hurt before** (item 5; stage 43, section 9): code neighbourhoods match on
      whatever dominates the code. On the drawn bank that was the lesion's position, so negatives shared position
      but not anatomy; here they share the anatomy, so InfoNCE must use the weak factors. Item 5's pricing predicted
      this: a weak reading pays only when the negatives already match on the strong content.
    - **Label-free discovery on the units** stays at the random-mixture baseline (one view of the code-batch arm
      above it, the other below).
    - **Reading.** Residual, jitter and prediction carry the recipe; the tails keep the factors linearly readable, so
      a narrow code keeps them. Code batches turn the narrow code into the best causal block so far, but on one seed
      and on a bank without a fingerprint; a second seed is the first check (stage 52), and stage 49 shows what they
      do where the fingerprint exists.

42. **The 24-unit recipe needs the residual, jitter and prediction, and neither the tails nor uniform batches
    (stage 49, read 19 Nov, `causal/stage49/`).** Leave-one-out from stage 33's recipe on the drawn-position bank; one
    seed per ablation; T1 / FLAIR. The reference seeds agree closely (own parts within .08 except ventricle .71 /
    .57; volume 12.7 / 13.1 voxels; ATE bias .12 / .10 and .12 / .12).
    - **Prediction: needed, most of all.** Without it (stage 28's arm) ventricle .24, thickness .34 / .43, sulcal
      .12 / .15, lr .59 / .57, brain .48 / .32, burden .35 / .34; volume error 18.0 / 19.7; ATE bias .32 / .34;
      image–tabular edges F1 .51 / .59; decoded image-graph F1 .56 / .71 (recipe .84 / .85).
    - **Residual: needed.** Without it burden 0 in code and backbone, temporal .50 / .48 (thickness and ventricle
      rise to .77 and .85); ATE bias .15 / .16; image–tabular F1 .73 / .68; image graph .67 / .71.
    - **Jitter: needed.** Without it burden .22 / .20, ventricle .41, thickness .39, lr .68, temporal .63 / .66;
      volume error 27.7 / 28.7; ATE bias .20 / .21.
    - **Tails: not needed at 24 units.** Without them (`gapnorm`) burden .54 / .50, sulcal .67 / .69 (+.15), the
      rest unchanged; the factors read by ridge as before (lesion volume .90 / .88 against .88 / .87); volume error
      10.9 / 12.2; ATE bias .10 / .11; image–tabular F1 .90 / .93; image graph .85 / .85. With the residual input
      and the prediction term, a plain average through the same gauge and head reads the randomly placed lesion as
      well as the tails do; item 1's need for them predates both.
    - **Code batches: no difference at 24 units here** (every measure within the seed spread: ATE bias .11 / .10,
      image graph .84 / .84, volume 13.3 / 12.2), against their large gain at 10 units on the fixed bank (item 41).
    - **Reading.** The recipe's load-bearing parts are the residual input, jitter and the prediction term. At 24
      units the tails can go; at 10 units they keep the factors linearly readable (item 41), so keep them for a narrow
      code. Code batches neither help nor hurt a wide code with the fingerprint present.

43. **Label-free PC on canonical directions is empty by construction (stage 51, read 19 Nov, `causal/stage51/`).**
    Every CCA-rotated code, the fixed 8-unit code reduced to 7 directions included, gave PC no edges (F1 0, SHD 6,
    every true edge missed), in both views. CCA whitens each view, so its canonical variates are mutually
    uncorrelated, and PC's Fisher-Z tests then find every pair independent. The graph lives in the factors'
    correlations, which any whitening rotation (CCA, ICA, PCA-whitened) removes; the components that match the
    concealed factors are their own parts, which are independent by definition (item 33's point for ICA). The
    rotation did align components with factors a little better than random rotations on the fixed bank (MCC .49 /
    .51 against the rotations' .39–.43 and the raw units' .40 / .46) and not on the drawn bank (c10 .44, c24 .45 to
    .46). Label-free graph discovery from two views stays out of reach, as section 8 says; the graph needs anchors.
    The test's design was at fault (no one checked that the rotation decorrelates), not the data.

44. **Code batches replicate at seed 2 (stage 52, read 19 Nov, `causal/stage52/`).** 10 units, fixed-position bank.
    T1 / FLAIR; code batches at seed 42, then seed 2.
    - **Own parts:** brain .95 / .95, .94 / .94; ventricle .85 / .81, .78 / .77; burden .60 / .58, .60 / .62;
      thickness .75 / .76, .76 / .77; temporal .85 / .84, .86 / .85; lr .91 / .93, .92 / .92; sulcal .72 / .77,
      .72 / .77. The code-batch seeds agree within .07; the uniform-batch seeds differ by up to .24 (brain .44 / .68).
    - **Precision:** volume error 11.3 / 11.5 and 10.4 / 10.0 voxels; the volume reads linearly at .88 and .90.
    - **Causal tables:** ATE bias .09 / .07 and .09 / .07 (rbf .07 / .06 and .07 / .05; uniform seeds .25 / .22 and
      .20 / .20, rbf .16 / .14 and .14 / .14). Image–tabular edges F1 .94 / .92 and .95 / .93; tabular causes of image
      factors .94 / .94 and .96 / .92. Decoded image graph .84 / .83 and .85 / .84 (truth .85).
    - **The block beyond measured anchors does not replicate for burden.** Seed 42 keeps three components (lr, burden,
      sulcal; burden .58); seed 2 keeps two (lr .94, sulcal .69–.71) and drops burden's directions at the redundancy
      cut (block .24 / .23): they carry R² .23–.30 from the other view's anchors and counted factors, above the .2
      threshold. The uniform seeds split the same way (.57 against .30): item 36's mechanism, seed-fragile either way.
    - **Reading.** On a bank without a fingerprint, code batches make the 10-unit code a strong and stable causal block
      on every measure except the label-free count's burden. They are label-free and generic. Whether they help on
      ADNI depends on what dominates the code there: with the lesion's position present they were neutral at 24 units
      (item 42) and mixed at 10 (stage 43, section 9).

45. **Footprints separate the factors that have a place, not the brain-wide ones (stage 54's footprints, read 19
    Nov, `causal/stage54/footprints.txt`).** Eight codes; Jacobians on 300 training subjects, weighted by variation;
    T1 / FLAIR. The image maps (per 16³ cell, how closely the scan's intensity tracks a factor's own part) are focal
    for the ventricle (66–76 effective cells), temporal atrophy (396–406 / 650–661) and, on the fixed bank only,
    burden (272 / 809). Brain size, thickness, asymmetry (lr) and sulcal widening cover 600–1,000 cells, nearly the
    whole brain.
    - **The rotation's rule** (its MCC above the units' and the random rotations' mean by more than their SD, and
      above the twin's, in both views):
      - Passes on the 24-unit codes at both seeds, with both rotations (pooled .65 / .68 and .72 / .66 against
        random .59 / .59 and .57 / .58, SD at most .02).
      - Passes on the fixed bank's uniform-batch 10-unit codes with the pooled rotation only (.44 / .50 and .51 /
        .51 against .38 / .43 and .46 / .47).
      - Fails on the code-batch arm in FLAIR at both seeds (.65 against .65; .65 against .63 with SD .04; T1
        passes, .69 and .67 against .63), on the drawn bank's 10-unit code, and on the 8-unit code.
      - Every trained code beats its twin (.20–.39) by far, so the twin is not the binding comparison. In the
        whitened frame random rotations already beat the 24-unit codes' units (.57–.59 against .40–.44): there the
        whitening by gradient energy does most of the aligning. The code-batch arm's units are the best aligned of
        any code (.57–.62).
    - **What the rotation finds,** by the second rule's first half (a component's specific footprint correlates
      positively with its factor's image map in both views):
      - Temporal atrophy, .37–.49 / .21–.34, in 7 of 8 codes, with a footprint like the temporal readout's (cosine
        .77–.98, except .45 in one view). The exception is the code-batch arm at seed 42, .02 / .14 (its readout
        .36 / .19).
      - Burden at the lesion's fixed site, in every fixed-bank code: .44–.81 / .45–.47, cosine with burden's
        readout .85–1.00. The component tracks burden itself only .27–.75 / .33–.67 (code batches .75 / .54 and
        .44 / .33): the site's directions also carry the lesion centre, and footprints cannot choose among
        directions read from one place.
      - Not the ventricle. The readouts read it at the ventricles (code batches .84 / .69 and .83 / .68; the
        24-unit code at seed 42 .54 / .66), but the component named after it is positive in both views only once
        (.66 / .08, the fixed 10-unit code at seed 42, its T1 footprint unlike the readout's). Names come from the
        raw factor, which shares variance with its parents, so a component that reads those elsewhere can take the
        name. Whether a ventricle-local component carries another name is for the `components` figures.
      - Not the brain-wide factors: their components' correlation with the image map is at most .25, mostly under
        .15.
      - The rule's other half (above the twin's matching component) needs `footprints.json`; the printout lacks the
        twin's image correlations. Not checked yet, so the small positives prove nothing.
    - **Code batches widen where the code reads.** The arm's mean footprint covers 391–437 cells against 271–292
      for uniform batches on the same bank; its interior share is .51–.57 against .60–.65 (the rest on the brain's
      boundary), and its residual-channel share .43–.53 against .63–.68. It reads more of the scan, where the
      brain-wide factors show, and reads those factors far better (R² .79–.95 against .43–.77).
    - **The per-subject rotation adds nothing.** It beats the pooled one in 2 of 16 views (by .06 and .01) and reads
      the drawn lesion's coordinates no better. On the drawn bank every direction's per-subject footprint, random
      ones included, covers 3–7 cells (19–71 on the fixed bank): the variation weighting lets each subject's most
      unusual cells dominate, most likely its own lesion.
    - **Reading.** Step 3b gives identity only to a factor with a place of its own. Without labels it finds a
      temporal component in most codes and a lesion-site component on the fixed bank, each reading where the image
      shows its factor. It cannot split factors that act across the whole brain, or several read from one place,
      and those are most of the block. On the best 10-unit code the rotation is no better than random rotations in
      FLAIR. The plan stands: anchors supply the factors' identity. Discovery on the rotations
      (`label_free_view{1,2}.txt`) and the intrinsic dimension are not in this item.

46. **Eight units give up ventricle's and burden's own parts (stage 56, read 20 Nov, `causal/stage56/`).** The
    code-batch recipe at 8 units, seed 42, fixed-position bank, against the 10-unit code-batch seeds 42 and 2 (items
    41, 44) and stage 47's uniform-batch 8-unit code (item 39). T1 / FLAIR.
    - **Own parts** (ridge), 8 units against the lower 10-unit seed: brain .94 / .94 (.94 / .94); ventricle .43 / .41
      (.78 / .77); burden .30 / .32 (.60 / .58); thickness .71 / .73 (.75 / .76); temporal .85 / .85 (.85 / .84); lr
      .84 / .84 (.91 / .92); sulcal .68 / .73 (.72 / .77). Boosting agrees: ventricle .45 / .42 against .73 / .67,
      burden .23 / .27 against .51 / .45. Summed over the seven factors the own parts come to 4.8 / 4.8 at 8 units,
      5.6 / 5.6 at 10 and 3.8 / 3.8 for the uniform 8-unit code.
    - **Precision:** volume error 15.5 / 14.7 voxels, against 11.3 / 11.5 at 10 units and 12.3 / 12.4 for the uniform
      8-unit code. The backbone still reads it (pooled 11.4 / 11.1, GAP 10.2 / 8.5), so the head drops it.
    - **Causal tables:** ATE bias .12 / .12 against .09 / .07 (rbf .11 / .10 against .07 / .06); the uniform 8-unit
      code .24 / .24 (rbf .17 / .16). Image–tabular edges F1 .86 / .83 against .94 / .92 and .95 / .93. Decoded image
      graph .76 / .77 against .84 / .83 and .85 / .84 (truth .85), about one more false edge.
    - **Concealed-factor benchmark** (measured anchors): F1 against the anchors' truth .67 / .70 against .73 / .73
      and .73 / .75; four novel components against three and two, with 1.0 / .6 false edges to them per graph
      against at most .2.
    - **Intrinsic dimension:** TwoNN 5.6 / 5.6 against 5.5 for its width's noise-free calibration row; at N/8 it falls
      .8 / .5, the calibration .6.
    - **Against the rule fixed before the results:**
      - Eight units do not keep what ten hold. Ventricle's own part falls .35 / .36 (allowed .15), burden's .30 / .26
        (allowed .10), and the volume error rises 4.2 / 3.2 voxels (allowed 2). The decoded graph falls .08 / .07
        (allowed .05), not resolved by resampling (± .06 / .05). Brain, thickness, temporal, lr and sulcal stay within
        .08, and the ATE bias within .05 (.03–.04 with both learners).
      - Code batches help at 8 units on four factors, past stage 50's thresholds: brain +.37 / +.38, thickness +.20 /
        +.23, temporal +.13 / +.12 and sulcal +.35 / +.37 over the uniform 8-unit code (ridge; by boosting brain gains
        only .10 / .11, as the uniform code holds brain non-linearly, item 39). Not lr (+.10 / +.08), and they lose
        ventricle (−.06 / −.07), burden (−.11 / −.08) and volume precision (3.2 / 2.3 voxels). The ATE bias halves.
      - The intrinsic dimension passes by the rule's letter, yet two own parts read .3–.4: at this N the estimator
        does not tell a weakly held direction from a full one. It cannot judge this question.
    - **Expectation, written before:** half right. Burden went and sulcal held, but the ventricle, strong at 10
      units, went too. The arm beats the uniform 8-unit code on four factors, not all seven.
    - **Label-free discovery without the code's length** (`latent_causal_discovery --sphere`, added 20 Nov, run
      locally on the stage's bundles): each unit standardized and each code divided by its length, the code as the
      cosine sees it. No unit can be dropped instead: once standardized, every unit takes both signs.
      - With uniform batches the length holds almost nothing: removing it costs any own part at most .03. With code
        batches it holds brain size: brain's own part falls from .94–.95 to .79–.84, lr's by .05–.10. Each batch is
        standardized around its own neighbourhood, so the population length is not the loss's blind number there.
      - Removing it moves single rows' F1 by up to .40 either way (uniform 8 units, T1: .27 → .67; 10 units seed 42,
        FLAIR: .71 → .57), and the untrained twins move as much. With the length, one row cleared its mixing null by
        two SDs (that FLAIR row, .71 against .41 ± .12), while seed 2's twin read .75 in the same view; without it
        none does (largest 1.6 SD). Item 41 stands: the units' graph sits at the mixing baseline, and width is not
        the reason.
    - **Why these two** (checked 20 Nov on the stage's bundles). The 8-unit codes keep one blend of ventricle's and
      burden's own parts, in about equal weights, and almost none of their difference: R² .68 and .05 with code
      batches, .71 and .18–.20 with uniform batches. The 10-unit codes keep both (.80–.85 and .58–.60). The two own
      parts are uncorrelated (r −.02), so the blend comes from the image: the fixed lesion sits at the ventricle's
      edge. Its centre is 12 voxels from the middle of the volume and its edge 2–5 voxels from the ventricle wall
      (`FIXED_CENTRE` in `bank.py`; the ventricle is a sphere of radius .12–.28 about the middle). At the largest
      ventricles the spot can no longer hold the lesion, and it moves to the nearest spot that can. Ventricle growth
      and lesion growth change the same patch, and a narrow code reads it as one number. The two are also the most
      correlated pair of raw factors (r .83, through the edge ventricle → burden), so the edge is the other suspect;
      a far site does not exist on this generator (below).
    - **What 10 units add** (20 Nov, same bundles, T1). Reading all seven own parts at once, the 10-unit codes hold
      every direction of that space (R² .96 down to .58–.59); both 8-unit codes miss exactly one (.01 with code
      batches, −.01 with uniform), the ventricle–burden difference. For that pair neither lever works alone: own
      parts ventricle / burden .49 / .41 at 8 uniform units, .49 / .46 at 10 (item 39), .43 / .30 at 8 with code
      batches, .78–.85 / .60 at 10 with them. Code batches create the demand: neighbours already agree on what the
      code holds, so only finer detail tells them apart (top-1 in the training sets .95–.96, against .99 with uniform
      batches, where the strong factors already settle the task). Width gives the room.
    - **No far site exists** (20 Nov, `lesion_site.py`, 100 training subjects). At the fixed site the lesion's edge is
      1.7 voxels from the ventricle wall (median; 10th–90th percentile 0.6–3.0) and 0.8 from the midline fissure or
      the cortex on its other side. The white matter between the ventricle (radius .12–.28) and the cortex (.40–.60)
      is 4–15 voxels thick and the lesion 6.3 across: of the 9,270 voxels that hold it in at least 10% of subjects,
      none keeps its edge 2 voxels clear of the ventricle in 90% of them. A far-site arm needs a smaller lesion or
      ventricle, which changes what is compared.
    - **The lesion's shift** (20 Nov, `lesion_shift.py`, the 1,000 test subjects re-rendered, rows checked against
      the bundles). 54% of lesions sit within half a voxel of the site; the rest slide sideways (SD 2.4 voxels left–
      right, 0.9 and 0.6 on the other axes). Brain size drives it, not the ventricle: one factor at a time, boosted R²
      of the shift's length .36 for brain, −.04 for ventricle (small brains have too little white matter there). The
      10-unit codes read the length at R² .59–.65 linearly (twins .22–.36) and .70–.73 boosted (seven true factors
      .77), and linearly better than the true factors (+.12 to +.18), so they hold it along directions of their own.
      The 8-unit codes do not (+.00 and −.01; uniform −.04 to .00): the shift did not take the direction the
      ventricle–burden pair lost, and holding it is one of the things the spare width does.
    - **The shift is not a latent of its own.** Placement is deterministic, so the anatomy fixes the shift: it is a
      switch at about the median brain size (smallest fifth of brains 2% on site, mean shift 4.0 voxels; fourth
      fifth 99% on site, 0.0). It adds no dimension (fixed-bank calibration, TwoNN: seven factors 6.4, with the
      lesion centre 5.8), and boosting reads it as well from the seven factors as from the code (.77 against .70;
      both .81). It still costs a direction: a switch is not a straight-line function of brain size, so a code that
      reads it linearly needs a direction for it, as it would for an extra latent.
    - **Ruled out, and what is left** (20 Nov). Not style: no trained code holds gain, bias or noise (R² at most .05).
      Not the encoder: the 8-unit run's backbone GAP holds both own parts (ventricle plus burden R² 1.41 / 1.24, above
      the 1 a single direction allows), so the 8-unit head drops one. Not curvature: ridge and boosting agree on the
      code. In raw terms the code keeps the pair's sum (R² .88) and drops its difference (.19), the minor axis: at
      raw r .83 the difference carries 9% of the sum's variance, the cheapest direction to lose for a loss that tells
      subjects apart. The prediction term whitens its targets, so it values that direction like any other, but a
      linear map from 8 units predicts at most 8. Left: the correlation and the shared patch (stage 58 separates
      them, with its caveat), and the training path: both 8-unit arms ran seed 42, so they started from the same
      weights.
    - **Where the directions go** (20 Nov; every direction of the code counted alike, T1). The 8-unit code-batch code
      spends 6 of its 8 directions on the own parts (R² .94–.62) and 2 on curved functions of factors it already
      holds (straight-line R² from the own parts .03 and .00; boosted from the seven factors .77 and .97). One bends
      brain size into a second direction (brain alone, boosted .78), the other is a joint function involving temporal
      atrophy. The 10-unit codes spend 7 of 10 on the own parts and 3 on curves (both seeds); the uniform 8-unit code
      6 and 2. So the correlation and the shared patch decide which own direction goes; the curves decide that one
      must. The likely reason the loss pays for a second direction of brain size first: the cosine InfoNCE pushes
      subjects to spread over the sphere (Wang and Isola 2020), and bending the strongest factor spreads them most.
    - **Reading.** At the matched width the code merged the two factors the image shows in one place, ventricle and
      the fixed lesion, keeping their sum and losing their difference. The two units 10 has past the matched width
      are not spare: summed over the factors they hold about .8 of one factor's own variation, nearly all of it
      ventricle and burden, plus 4 voxels of volume precision. Code batches change which factors a narrow code keeps;
      they do not replace width. This backs the 18 Nov width decision (section 1) and section 8's point that weaker
      own parts need spare width. One seed, but the own-part losses (.26–.36) are four to five times the code-batch
      seed spread (.07).

47. **Without the edges, 8 units hold ventricle and burden apart and drop lr instead (stage 58, read 21 Nov,
    `causal/stage58/`).** Stage 56's 8-unit code-batch arm on `bank_nofield_fixedpos` (same image, lesion site and
    generator, no SCM), seed 42, against stage 56's causal 8-unit code and stage 50's 10-unit code. On the independent
    bank the own parts are the factors themselves. T1 / FLAIR.
    - **Ventricle–burden plane:** worst direction .75 / .75 (causal 8 units .05 / .06), blend .00 / .00 (.24 / .27).
      Ventricle .86 / .86, burden .75 / .75.
    - **The spectrum still misses one direction:** .94 .88 .86 .78 .75 .55 .22 (T1; FLAIR last .18). The weakest
      direction is lr alone (+.96 / +.95), and lr's own R² is .26 / .24 (.84 / .84 on the causal bank). Temporal .65 /
      .65, the next weakest (.85 there). Brain .93, thickness .88–.89, sulcal .70–.73.
    - **Where the directions go:** 8, with 6 on the factors and 2 on curves, as on the causal bank. One curve is
      partly lr (factor R² .23 / .20; boosted from the seven factors .90, from lr alone .14), the other bends brain size
      (boosted from the seven .94, brain alone .34–.36).
    - **Precision:** volume error 20.3 / 19.8 voxels on an SD of 41.3 (own R² .80 / .81). Not comparable in voxels with
      the causal bank (SD 19.9 for burden's own part). No style in the code (R² at most .04).
    - **Against the rule fixed before the results:** the edges made the blend. Worst direction .40 or more and blend
      .10 or less in both views (.75 and .00). The rule's caveat applies: without the SCM burden's own signal is about
      twice as wide (±41 against ±20 voxels), so the larger signal could hold the pair apart. "Held apart" is weaker
      evidence for the edges than "still blended" would have been for the image.
    - **Expectation, written before:** none firm. The result fits the brain–thickness case (r .80, not blended).
    - **Reading.** 8 units miss one direction on either bank. Which one goes follows from what is cheapest for the
      loss. On the causal bank it is the minor axis of the most correlated pair (ventricle–burden, r .83: their
      difference carries 9% of the sum's variance, item 46). On the independent bank it is lr, the faintest factor.
      The shared patch at the ventricle's edge was not enough to blend the pair without the correlation. The curves
      take 2 directions on both banks, so the image, not the edges, decides how many directions curves cost.

48. **At 9 units the curves take the extra unit (stage 59, read 21 Nov, `causal/stage59/`).** Stage 56's code-batch
    arm with `--content-units 9`, seed 42, fixed-position causal bank, against the 8-unit code (stage 56) and the
    10-unit codes at seeds 42 and 2 (stages 50, 52). Trained 20,000 steps cleanly (final loss .255). T1 / FLAIR.
    - **Where the directions go:** 9, with 6 on the own parts and 3 on curves (8 units: 6 and 2; 10 units: 7 and 3).
      Spectrum .95 .90 .82 .74 .64 .60 .04 (FLAIR last .02). The curves: one partly lr (own R² .05 / .04; boosted from
      the seven factors .80 / .82, lr alone .52 / .55), one bends brain size (brain alone .65 / .57), one a joint
      function (boosted .91 / .92, no single factor above .17).
    - **The missing direction now blends three factors:** ventricle +.61, burden −.56, lr +.47 (FLAIR +.70, −.48,
      +.47). The ventricle–burden worst direction rises .05 → .25 / .06 → .23 (blend .19 / .21), and lr falls.
    - **Own parts** (ridge), 9 units against 8: brain .94 / .94 (.94 / .94); ventricle .55 / .44 (.43 / .41); burden
      .45 / .49 (.30 / .32); thickness .69 / .70 (.71 / .73); temporal .76 / .80 (.85 / .85); lr .68 / .69 (.84 /
      .84); sulcal .67 / .74 (.68 / .73). The 10-unit codes: ventricle .78–.85, burden .58–.62, lr .91–.93.
    - **Precision:** volume error 11.9 / 11.5 voxels, near the 10-unit codes (11.3 / 11.5 and 10.4 / 10.0) and well
      under the 8-unit code (15.5 / 14.7). No style (R² at most .00).
    - **Causal tables:** ATE bias .16 / .15 (rbf .14 / .14), worse than 8 units (.12 / .12) and 10 (.09 / .07,
      .09 / .07). Image–tabular edges F1 .90 / .92 (8 units .86 / .83; 10 units .94 / .92, .95 / .93). Decoded image
      graph (20 resamples of 800 rows) .70 / .70 against .76 / .77 at 8 units (difference not resolved) and .84 / .83,
      .85 / .84 at 10 (resolved; truth .85).
    - **Label-free discovery** (PC on the channels, view 1): F1 .59 (null .35, learned +.24, SHD 7, MCC .61), the same
      as 8 units (.59, +.24); both 10-unit codes sit at or below their nulls (learned −.05 and −.01).
    - **Against the rule fixed before the results:** the curves grow with room. 3 directions on curves and the
      spectrum's last value below .15 (.04 / .02) in both views. The count does not set the width (that needed .40 on
      the last value and on the pair's worst direction; .04 and .25).
    - **Expectation, written before:** right. The curves went from 2 to 3 and 9 units still miss one own direction.
    - **Reading.** Extra width goes to curves first, so no fixed margin over the matched width is safe: 8 → 9 units
      bought a curve, and only 9 → 10 bought the seventh own direction. The 9-unit code spreads its loss over three
      factors where the 8-unit code lost one pair, which helps volume precision but not the causal outputs (ATE bias
      and the decoded graph are no better than at 8 units). This motivates stage 60's penalty on curves at 8 units, not
      more width. One seed.

49. **Rebuilding the code neighbourhoods every 100 steps nearly closes the 8-unit gap; prediction weight 3 does not
    (stage 62, read 21 Nov, `causal/stage62/`).** Stage 56's 8-unit code-batch arm with one change each, seed 42,
    fixed-position causal bank, against stage 56's 8-unit code and stage 50's 10-unit code. T1 / FLAIR.
    - **Rebuild every 100 steps** (`..._abl_codebatch100_predict1_c8_20k`):
      - Spectrum .95 .91 .87 .76 .74 .58 .45 (FLAIR last .36; stage 56 .01 / .01, 10 units .59 / .56). Ventricle–
        burden worst direction .50 / .42 (.05 / .06; 10 units .59 / .57), blend .04 / .09 (.24 / .27). The weakest
        direction mixes ventricle +.65, burden −.60, sulcal +.42.
      - Directions: 6 above .5, one at .46 / .37 (the seventh own direction, partly held), one curve bending brain size
        (brain alone .83). Stage 56 had two curves at .03 and .00.
      - Own parts (ridge) against stage 56 and the 10-unit code: ventricle .62 / .57 (.43 / .41; .85 / .81), burden
        .54 / .50 (.30 / .32; .60 / .58), brain .94, thickness .76 / .76, temporal .85 / .86, lr .90 / .90, sulcal
        .67 / .66 (.68 / .73).
      - Volume error 12.6 / 13.0 voxels (15.5 / 14.7; 10 units 11.3 / 11.5). No style (R² at most .00). Cross-view
        top-1 .981 / .890 / .770 for uniform 32 / neighbour 32 / all 1,000 (stage 56 .981 / .896 / .794).
      - Causal tables: ATE bias .10 / .10, rbf .08 / .08 (.12 / .12; 10 units .09 / .07). Image–tabular edges F1 .92
        / .89 (.86 / .83; .94 / .92). Decoded image graph .86 / .85 (.76 / .77; .84 / .83; truth .85).
      - Label-free discovery (view 1): F1 .71 (null .33, learned +.39, SHD 4, MCC .67), the best of any code so far
        (stage 56 .59, +.24; 10 units .29, −.05).
      - Final neighbourhood spread anatomy .48, burden .55 (stage 56 .51, .60): rebuilt neighbourhoods stay more
        alike in the current code.
    - **Prediction weight 3** (`..._abl_codebatch_predict3_c8_20k`):
      - Spectrum's last value −.02 / −.01, two curves at .00 (one bends brain size, .59 / .39). The missing direction
        moves: ventricle +.68, sulcal +.59, burden −.40. Ventricle–sulcal worst .10 / .11.
      - Own parts: burden .51 / .52 (+.21), lr .91 / .91, but sulcal .41 / .42 (−.27 / −.31) and ventricle .36 /
        .34. Volume error 11.6 / 10.7.
      - Cross-view top-1 falls to .930 / .724 / .478. ATE bias .14 / .12 (rbf .11 / .10), decoded graph .74 / .75,
        label-free F1 .46 (learned +.15).
    - **Against the rule fixed before the results:**
      - Prediction weight 3 does nothing: 2 curves and the last value below .15 in both views. It trades which own
        direction goes (sulcal for burden) and costs discrimination.
      - Rebuild every 100 is "anything else", nearest to closing the gap. T1 passes both thresholds (last value .45,
        pair's worst .50); FLAIR passes the pair (.42) and misses the last value (.36 against .40). By the .5 cut-off
        the code still counts 2 directions under .5, but one of them is the seventh own direction at .46 / .37, so
        only one is a curve. Every other own part stays within .10 of stage 56's (sulcal FLAIR −.07).
    - **Expectation, written before:** wrong for the rebuild. It predicted little change, because the spread settles by
      the first rebuild. The spread does settle, but the fresher neighbourhoods still freed a direction.
    - **Reading.** How closely the negatives track the current code matters, not only that they are neighbours. With
      neighbourhoods 500 steps stale, a batch's subjects are alike in a code the network has since changed. Rebuilt
      every 100 steps, they are alike in what the code already holds, so the loss pays only for what it lacks. At 8
      units this cut one curve and gave the seventh own direction .45 / .36. The causal outputs reach the 10-unit
      code's, and label-free discovery beats it, while ventricle (.62 against .85) and volume error (1.3–1.5 voxels)
      still trail. One seed, and the gains on ventricle and burden (.16–.24) are two to three times the 10-unit
      seed spread (.07). Next: replicate at seed 2, and try rebuilding more often (every 20–50 steps), at 8 and at 9
      units.

50. **Stage 62's rebuild gain does not replicate; at 8 units the seed decides whether the seventh direction is held,
    and the 9-unit code rebuilt every 100 steps matches the 10-unit code (stage 63, read 22 Nov, `causal/stage63/`).**
    Stage 56's 8-unit code-batch recipe, four arms, fixed-position causal bank, against stage 62's rebuild-100 code,
    stages 56 and 59's codes and the 10-unit codes (stage 50 seed 42, stage 52 seed 2). T1 / FLAIR.
    - **Seed-2 control, 8 units rebuilt every 500** (`..._abl_codebatch_predict1_c8_20k_s2`):
      - Spectrum .95 .91 .89 .84 .74 .66 .56 (FLAIR last .56): all seven own directions held, no curve among them
        (the eighth direction bends brain size, own R² .00). Weakest direction is burden alone (+.99). Ventricle–burden
        worst .56 / .56, blend .00 / .01.
      - Own parts: brain .94, ventricle .82 / .79, burden .56 / .57, thickness .80 / .80, temporal .85 / .86, lr .90
        / .91, sulcal .67 / .71. Every one within .07 of the 10-unit code.
      - Volume error 12.6 / 11.9 voxels (10 units 11.3 / 11.5). ATE bias .079 / .072, rbf .070 / .061. Decoded
        graph .835 / .838. Image–tabular F1 .95 / .96. Cross-view top-1 .991 / .919 / .834 (10 units .972 / .875 /
        .740). Label-free F1 .40 (learned +.06).
    - **Seed 2 rebuilt every 100** (`..._abl_codebatch100_predict1_c8_20k_s2`):
      - Spectrum's last value .12 / .11: the seventh direction is lost, as in stage 56. It mixes burden +.82,
        ventricle −.40, lr −.35. Ventricle–burden worst .21 / .22. One curve at .13 (lr alone .28), one bends brain
        size.
      - Own parts: ventricle .69 / .64, burden .29 / .33, lr .66 / .72, thickness .69 / .72, sulcal .64 / .75.
      - Volume error 17.9 / 15.7. ATE bias .148 / .126 (rbf .128 / .114). Decoded graph .841 / .811. Top-1 .967 /
        .825 / .653. Effective rank 2.3. Label-free F1 .35 (learned −.01).
    - **8 units rebuilt every 25, seed 42** (`..._abl_codebatch25_predict1_c8_20k`):
      - Spectrum's last value −.02 / −.02 (stage 62's every 100: .45 / .36). Two curves at .00; the missing
        direction mixes burden +.78, lr −.54. Burden–lr worst .06 / .06.
      - Own parts: ventricle .81 / .77, burden .24 / .27, lr .52 / .49. Volume error 22.1 / 19.6, the worst of the
        fixed-bank code-batch codes. ATE bias .164 / .160. Decoded graph .785 / .799. Top-1 .968 / .824 / .671
        (every 100: .770).
    - **9 units rebuilt every 100, seed 42** (`..._abl_codebatch100_predict1_c9_20k`):
      - Spectrum .95 .89 .86 .83 .74 .67 .62 (FLAIR last .62): all seven held. Weakest direction burden (+.92 /
        +.97). Ventricle–burden worst .63 / .63, blend .00. Two directions at own R² .00: one bends brain size, one
        a curve the boosted factors read at .76 / .73 (stage 59's 9-unit code: last .04 / .02, 3 curves).
      - Own parts: brain .94, ventricle .80 / .78, burden .63 / .63, thickness .78 / .80, temporal .86 / .87, lr .84
        / .87, sulcal .73 / .77.
      - Volume error 10.7 / 10.3 voxels, better than the 10-unit code's 11.3 / 11.5. ATE bias .086 / .078, rbf
        .075 / .073. Decoded graph .828 / .821. Image–tabular F1 .96 / .99. Top-1 .990 / .924 / .840. Label-free F1
        .33 (learned +.00).
    - No arm holds style (code R² of gain, bias and noise at most .02).
    - **Against the rules fixed before the results:**
      - **Replication: void.** The seed-2 control's last value is .56 / .56, above the .40 at which the rule voids
        the comparison: stage 62's gain was seed 42's luck. The seed-2 rebuild-100 arm also falls well below its
        own control (last value −.44, ventricle −.13 / −.15, burden −.27 / −.24), so the lever does not even help
        at seed 2.
      - **Every 25 against every 100: hurts.** The last value falls .47 / .38; top-1 over all 1,000 falls .770 to
        .671.
      - **9 units rebuilt every 100: holds all seven** (last value .62 / .62, pair's worst .63 / .63).
      - **The goal, matching the 10-unit code:** the 9-unit rebuild-100 code matches on every criterion: own parts
        within .07 (lr −.07 / −.06 the largest gap), volume error 0.6 / 1.2 voxels better, ATE bias .000 / +.009,
        decoded graph −.007 / −.013. The seed-2 8-unit control misses by one: own parts within .07, ATE bias and
        decoded graph match, but T1 volume error is 1.3 voxels worse (limit 1; FLAIR 0.4). The other two arms miss
        by much (burden .24–.33). The closest arm is the 9-unit rebuild-100 code.
    - **Expectation, written before:** wrong on replication (the control held the direction, the lever lost it) and
      on every 25 (hurts, not saturates). Right that the 9-unit arm holds all seven and comes closest; wrong that
      ventricle trails most (lr does, at .84 / .87).
    - **Reading.** At 8 units, whether the code holds the seventh own direction varies by seed: seed 42 loses it at
      every 500 and at every 25, keeps it partly at every 100; seed 2 keeps it at every 500 and loses it at every
      100. A rebuild interval changes the training trajectory the way a new seed does, so one seed per cell cannot
      separate the interval from luck, and the four 8-unit codes give no sign that one interval is better. The
      seed-2 control shows that 8 units can hold all seven own parts with this recipe at full precision except
      1.3 voxels of T1 volume, so the matched width suffices in practice, as Yao's theory says, but the optimiser
      finds that solution only sometimes. The 9-unit result is one seed too, and stage 59's 9-unit code at every 500
      missed the direction, so whether 9 units reach it more reliably needs seeds. The codes that hold all seven also
      have the highest cross-view top-1 over all 1,000 subjects (.834, .840), above both 10-unit codes; top-1 may
      flag a held seventh direction without labels. Label-free discovery does not follow: stage 62's F1 .71 is not
      repeated by any code that holds all seven (.33–.40). Next: count how often each width holds all seven, over
      more seeds (8 and 9 units at every 500 and every 100), before calling any lever.

51. **Seed count: about half of 8- and 9-unit codes hold all seven own parts; the extra unit buys no reliability,
    rebuilding every 100 steps costs it, and top-1 does not flag a held code (stage 64, read 23 Nov,
    `causal/stage64/holds.txt`).** Stage 56's code-batch recipe at 8 and 9 units, rebuilt every 500 and every 100,
    seeds 3, 4 and 5, scored beside the earlier codes of each cell: 5 codes per 8-unit cell, 4 per 9-unit cell,
    and the 10-unit codes at seeds 42 and 2. T1 / FLAIR.
    - **Per cell, holds / matches the 10-unit code (stage 50, seed 42):**
      - 8 units, every 500: holds 3/5 (seeds 2, 3, 5), matches 2/5 (3, 5; seed 2 misses T1 volume, 12.6).
      - 8 units, every 100: holds 2/5 (3, 4), matches 0/5 (both holders miss on ventricle, .74 / .67 and .70 FLAIR;
        seed 42 holds T1 only, last .45 / .36).
      - 9 units, every 500: holds 3/4 (3, 4, 5), matches 2/4 (4, 5; seed 3 misses sulcal FLAIR, .67).
      - 9 units, every 100: holds 1/4 (seed 42, stage 63's code), matches 1/4.
      - 10 units: holds 2/2, matches 2/2.
    - **How codes fail.** Every failing code keeps 6 directions on the own parts and loses the seventh (last value
      .01–.45). In 7 of 9 the lost direction blends ventricle and burden (often with lr): the familiar failure. In
      two (8 units every 100 seed 5, 9 units every 100 seed 5) it blends sulcal, thickness and temporal instead, and
      ventricle–burden holds (.54–.58). The 9-unit codes spend both spare units on directions with own R² .00 (curves
      and a bent copy of brain size), whether they hold or not: the ninth unit goes to curves, not to the seventh
      own part. Holders are good codes: 4 of 6 holders at every 500 match the 10-unit code; the misses are single
      criteria near the limit (T1 volume 12.6, sulcal FLAIR .67). Failures miss on many (ATE bias .12–.16, decoded
      graph .70–.78 in most).
    - **Against the rules fixed before the results:**
      - **Width: the same.** Pooled hold rate 9 units 4/8 against 8 units 5/10, difference .00 (Fisher p 1.00).
        Match rate 3/8 against 2/10, +.17: unresolved (p .61).
      - **Rebuild interval: every 500 more reliable.** Holds 6/9 against 3/9, matches 4/9 against 1/9, both −.33
        for every 100 (p .35, .29: the threshold is met, significance is not claimed).
      - **Cells:** 9 units every 500 is reliable by the rule (3 of 4). 8 units every 500 (3/5), 8 units every 100
        (2/5) and 9 units every 100 (1/4) are seed lotteries. No cell is a failure.
      - **Top-1 as a flag: no.** Holders and the rest overlap at both widths (6 codes on the wrong side each; 8
        units: holders .698–.854, the rest .653–.794; 9 units: holders .746–.858, the rest .762–.835). Item 50's
        hint does not survive the count.
      - **Highest top-1 per width holds at both widths** (8 units every 500 seed 3, .854, holds and matches; 9
        units every 500 seed 3, .858, holds, misses sulcal FLAIR). So "train several seeds, keep the highest top-1"
        passes its rule here. Given that top-1 does not separate holders from the rest, this rests on the top code
        of each width and is weak evidence for a recipe.
    - **Expectation, written before:** 8 units held in 5 of 10, above the 2–4 expected; 9 units in 4 of 8, in the
      expected range, but width reads "the same", not "unresolved or more reliable". Wrong that the interval makes
      no difference (every 100 is worse) and that top-1 flags holding. Right that the highest-top-1 code holds at
      both widths.
    - **Reading.** At the matched width the recipe holds all seven own parts in about half of seeds, and a code
      that holds mostly matches the 10-unit code: Yao's matched width is enough in practice when the optimiser
      finds the solution, and it finds it about one time in two. One extra unit does not change that, because the
      ninth unit goes to curves. Stage 62's and stage 63's rebuild-100 gains were seed luck (every 100 holds in 3 of
      9 codes, every 500 in 6 of 9). The interval to use is every 500. Top-1 cannot tell a held code from a failed
      one without labels. Label-free fixes have to act in training or in the start, not in selection: stage 65
      (8 units started from a 10-unit code compressed without labels) tests the start.

52. **A compressed 10-unit start does not make 8 units reliable: it holds as often as from scratch (3/5), and longer
    training repairs no seed (stage 65, read 24 Nov, `causal/stage65/holds.txt`).** Each 10-unit parent (stage 56's
    recipe, 20,000 steps, seeds 42, 2, 3, 4, 5) compressed to 8 units without labels, by redundancy (`red`) or by
    PCA (`pca`), then 10,000 steps at 8 units; stage 64's from-scratch 8-unit codes (every 500) continued 10,000
    steps (`cont`); beside the from-scratch codes (`e500`) and the parents. T1 / FLAIR.
    - **Per arm, holds / matches the 10-unit code (stage 50, seed 42):**
      - 10-unit parents: holds 4/5 (seed 4 fails, last .04 / .07), matches 3/5 (seed 3 misses lr .66 / .70 and
        FLAIR ATE bias .103).
      - red: holds 3/5 (42, 2, 5), matches 3/5 (the same three).
      - from scratch (e500): holds 3/5 (2, 3, 5), matches 2/5 (3, 5; seed 2 misses T1 volume, 12.6).
      - cont: holds 3/5 (2, 3, 5: the same seeds as from scratch), matches 2/5 (3, 5; seed 2 misses sulcal .60 /
        .65).
      - pca: holds 2/5 (5, 42), matches 1/5 (42; seed 5 misses ventricle, temporal and ATE bias .14 / .15).
    - **Against the rules fixed before the results:**
      - **Compression makes 8 units reliable: no.** Red holds 3/5 against from scratch 3/5 (difference 0, Fisher
        p 1.00); matches 3/5 against 2/5 (+1). Both differences are at most 1: no better. Paired by seed, each arm
        alone holds in one seed (red at 42, from scratch at 3).
      - **The selection matters: no, by the rule.** Red 3/5 against PCA 2/5 holds, within 1: the warm start carries
        it, not the choice of directions. Matches differ by 2 (3/5 against 1/5). The rule is on holds, so this is
        not a verdict.
      - **Extra training alone: no.** Cont holds in exactly the seeds from scratch holds (3/5 against 3/5; paired,
        0 and 0). Seed 42's last value rises from .01 to .23 / .17 but stays below .40. A seed that lost the
        direction stays lost.
      - **Parents:** seed 4's parent does not hold. Over the four parents that hold, red holds 3/4 and matches 3/4.
      - **Secondary, red's own parts against its parent's (within .10, both views):** within at seeds 42, 2 and 5.
        Seed 3 loses lr (.66 -> .26, .70 -> .30): the parent's weakest lr of the five, cut by the compression and not
        regained. Seed 4 drops temporal T1 (.78 -> .67) from a parent that already fails.
    - **The pre-recorded compression caveat lines up with the failures.** The compression logs (written 23 Nov 21:20,
      before any result) gave the second removed direction held-out R² .997, .990 and .986 at seeds 42, 2 and 5,
      and .848 and .650 at seeds 3 and 4. The three clean cuts held and matched (3/3). Seed 3's unclean cut took lr.
      Seed 4's came from a parent that does not hold. This pattern was seen after the results, on 5 seeds: it is a
      hypothesis, not a finding.
    - **How codes fail.** Seed 4 fails in all five arms here, the 10-unit parent included, each time with the
      ventricle–burden blend (worst .14–.34). Red seed 3 and PCA seeds 2 and 3 keep ventricle–burden (worst
      .47–.64) and lose lr instead (last .10–.18). PCA's start drops low-variance own parts (brain, thickness,
      temporal at seed 2), as the pre-flight showed, and fine-tuning does not get them all back.
    - **Top-1 as a flag: still no.** Holders .792–.900 against the rest .700–.879 at 8 units (16 codes on the wrong
      side). The highest top-1 code (cont seed 3, .900) holds and matches, as in item 51.
    - **Expectation, written before:** right that the parents hold (4 of 5, the stated fallback), that PCA holds in
      1–3 (2), that the continued arm is within 1 of from scratch (identical), and that red matches in at least 3
      (3). Wrong that red holds in 4 or 5 (3).
    - **Reading.** A compressed start keeps what its parent holds, provided the cut is clean, and fine-tuning does
      not repair what the start lacks. So the lottery moves; it does not go away. An 8-unit code now depends on a
      10-unit seed that holds (4/5 here) and on a compression that removes only curves. Where both happen (seeds 42,
      2, 5) the 8-unit code matches the 10-unit reference on every criterion. The open question is whether
      "every removed direction has held-out R² near 1" is a label-free accept rule. The compression log already
      records it, so the rule costs no labels. It has to be fixed before being tested on new seeds; five seeds
      chosen after the fact do not establish it. Longer training is not a lever (cont).

53. **One label-free signal passes the rule for choosing runs: cross-view top-1 over all validation subjects, and
    only narrowly (stage 66, read 24 Nov, `causal/stage66/criterion.txt`).** `criterion_check.py` on every
    finished warm.py run: 149 scored, 17 supervised runs skipped. Truth is each run's own parts on test subjects; the
    signals use training and validation subjects only. Grouping "bank" (7 banks of at least 4 runs) is the rule's;
    "bank_width" (11 groups) is shown beside it.
    - **Pass rule (docstring, fixed before any result):** under "bank", for both own_mean and own_weakest, the
      Spearman interval lies above 0 and picking the signal's best run removes at least half of a random pick's
      regret.
    - **top1_all passes, just:** own_mean rho +.41 [.22, .56], regret .53, right sign in 7/7 banks; own_weakest +.31
      [.14, .47], regret .51, 5/7. Both regrets are within .03 of the .5 threshold. Under "bank_width" its own_weakest
      regret falls to .44.
    - **Close but failing:** gauss_mi_gain (rho +.48 / +.35, regret .47 / .31), gap_gauss_mi_gain (.43 / .33),
      top1_32 (+.47 / +.37 but regret .30 / .16; it passes under "bank_width", which is not the rule's grouping).
      The spectrum signals (RankMe, effective rank) run the wrong way on own_mean (rho -.18, -.24). Final loss,
      memorisation, neighbour agreement and alpha-ReQ fail.
    - **Burden:** no signal tracks burden's own part (every rho .19 or below, most intervals include 0).
    - **Against items 51 and 52:** within one 8-unit cell, top-1 does not separate codes that hold from those that
      fail (16 codes on the wrong side in stage 65). Across recipes in a bank it ranks runs better than chance. It
      can choose a recipe, but not a seed within one, and it says nothing about burden.
    - No expectation was written before this stage.
    - More breakdowns (per bank, per factor, seed spread, and choosing by the measured factors' exact own
      parts) are in the status section's stage 66 entry, from the other session's reading.

54. **The harmonic penalty makes 8 units less reliable, keeps the curves and costs discrimination (stage 60, read
    24 Nov, `causal/stage60/holds.txt`).** Stage 64's from-scratch 8-unit arm (every 500) plus `--harmonic-weight 1`
    at seeds 42, 2, 3, 4, 5, beside the from-scratch 8-unit and 10-unit codes of the same seeds. T1 / FLAIR.
    - **Holds / matches the 10-unit code (stage 50, seed 42):** penalty 1/5 (seed 2) / 0/5 (seed 2 misses T1 volume
      error, 13.6, as its from-scratch twin did at 12.6); from scratch 3/5 (2, 3, 5) / 2/5 (3, 5); 10 units 4/5 /
      3/5.
    - **Against the rules fixed before the results:**
      - **Reliable: no.** Holds 1/5 against 3/5 (Fisher p .52), matches 0/5 against 2/5 (p .44). Both are 2 below
        from scratch, outside the "no better" band of 1. The rules named no "worse" outcome; this is the mirror of
        the "better" threshold, and five seeds do not make it significant.
      - **Paired by seed:** no rescue. Seeds 42 and 4 still lose the seventh direction to the ventricle–burden
        blend (last -.01 / -.01 and -.02 / -.02, worst .00 / .01 and .05 / .06). Two holders are lost: seed 3
        drops sulcal (own R² .28 / .30, last -.01 / -.01; the weakest direction mixes sulcal, ventricle and
        thickness), and seed 5 narrowly (last .35 / .41, worst .49 / .49). Seed 2 holds, lower than its twin
        (last .47 / .50 against .56 / .56).
      - **Against 10 units (described only):** 1/5 and 0/5 against 4/5 and 3/5.
      - **Mechanism: fails.** Every code keeps a curve in both views (5/5; the rule's failure is 4/5). Counts T1 |
        FLAIR at seeds 42, 2, 3, 4, 5: 1|2, 2|1, 1|2, 2|2, 2|3, against from scratch 2|2, 1|1, 1|1, 2|2, 1|1. Each
        code keeps a direction at own R² .00 that boosting from the seven factors predicts at .84–.98, mostly from
        brain size alone (.59–.83; split over two directions in FLAIR at seeds 42 and 4). Some of the extra curves
        are weak own parts just under .5 (seed 2's T1 burden direction at .48, seed 5's at .36–.50); the rule
        counts them.
      - **Cost: yes.** Cross-view top-1 over all 1,000 subjects falls at every seed, by .084 to .318, mean .207
        (the rule's cost is a mean drop above .05).
      - **Style (read beside):** none in the code in any arm (linear R² at most .05, boosted below 0), except
        seed 3's penalty code, which reads T1 gain at .13 linearly (boosted -.05).
    - **The penalty is not being paid** (seen after the results, in the training logs). In all five arms it fell
      from .12–.21 in the first 1,000 steps to .017–.024 at step 20,000. The critic's full fit ended within .01 of
      its linear one (R² -.07 to +.05). So the online critic stopped finding the non-linear redundancy that
      boosting from the true factors finds at .84–.98. The rule's gloss for a failure, "the loss keeps them and
      pays the penalty", does not apply: the code keeps them and the critic does not see them. Either the encoder
      escapes the critic, or the curves cannot be read from the other units at all, only from the factors. A
      critic fitted afresh on the saved codes, with no training, would tell which.
    - **Other measures:** volume error is worse at seeds 2, 3 and 5 (seed 5 15.0 / 14.1 against 11.0 / 9.5) and
      better at seed 42. Seeds 3 and 5 about double their ATE bias (.15 / .14 and .16 / .15 against .07–.08), and
      seed 3's decoded graph falls to .65 / .71 (.85 / .84 from scratch).
    - **Expectation, written before:** wrong. The five-seed expectation said the penalty would remove most curves,
      rescue seed 42 most likely, hold in 3 or 4 of 5 and cost up to .05 of top-1. It removed none, rescued none,
      held in 1 of 5 and cost .21. The 20 Nov expectation's alternative, that the code keeps the curves or loses
      discrimination, happened in both halves, but without paying the penalty.
    - **Reading.** Charging what a small non-linear critic gets beyond a linear one, at weight 1, does not stop an
      8-unit code spending directions on curves. The critic is satisfied while the curves stay, and the code loses
      discrimination and two of three holders. Not a lever in this form. Whether any penalty on units predicting
      units could work turns on the offline check above.

55. **Measured anchors choose runs, noisy or not, where retrieval could not (stage 67, read 25 Nov,
    `causal/stage67/anchor.txt`).** `anchor_check.py`: 154 runs exported (17 trained with labels skipped), 149 scored
    (those in stage 66's `criterion.json` at the same step). Truth: the hidden factors' own parts on test subjects,
    mean of the views; it equals stage 66's to 5e-5. Each score picks runs within each bank.
    - **Pass rule (docstring, fixed before any result):** split 'adni' (lr and sulcal hidden; brain, ventricle,
      thickness and temporal measured as their latents, burden as the lesion volume), anchors adjusted for each
      other, reliability .9, grouping 'bank'. Usable if the bootstrap 95% interval of rho lies above 0 and the pick
      removes at least half of a random pick's shortfall; preferred over retrieval if it also removes more than
      top1_32.
    - **Usable: yes.** rho +.77 [.64, .84], shortfall removed .89.
    - **Preferred over retrieval: yes.** top1_32 removes .40 on the same runs. The rule named top1_32, which failed
      stage 66; top1_all, which passed it, removes .66 here. The anchors beat both.
    - **Noise costs nothing at these levels.** Adjusted anchors at reliability 1, .9, .75 and .5: rho .75–.77,
      removed .85–.89. Raw anchors do as well (.80–.82, .85–.86), and so do the measured factors' exact own parts
      (.80, .89). Adjusting for the other anchors is not needed to choose runs here.
    - **Other hidden sets** (r = .9; every split hiding 1, 2 or 3 factors, averaged): rho .64 / .71 / .75, removed
      .73 / .79 / .80, against .19 / .25 / .27 for top1_32 and .37 / .46 / .50 for top1_all. Under 'bank_width'
      (11 groups, not the rule's grouping): rho .76, removed .81, against top1_32's .48, .69.
    - **By bank** (rule cell, rho / removed; `causal/stage67/posthoc/by_bank_adni.txt`, which reproduces the pooled
      values exactly; `by_bank_no_wmh.txt` for burden hidden):
      - Fixed-position bank (47 runs: the 8-, 9- and 10-unit arms): .68 / .89. Retrieval removes -.43 there, worse
        than a random pick.
      - Current recipe's bank (32): .82 / 1.00; retrieval does well too (.91–.92).
      - Drawn bank at T1 gain .4 (24): .88 / 1.00. bank_causal (19): .69 / .92, where retrieval removes -1.48.
        T1 gain 0 (9): .93 / 1.00. No-gain (6): 1.00 / 1.00.
      - The one failure is the 4-run nuisance bank (.36 / -.10; retrieval 1.00), whose hidden own parts are all low
        (.08–.34).
    - **Burden hidden ('no_wmh'): fails.** rho .27, removed .11; the exact own parts do no better (.29, .14).
      - It works in the prediction-recipe banks: fixed-position .57 / .65, current recipe .66 / 1.00, no-gain
        .93 / .88.
      - It reverses in the older banks: bank_causal -.55 / -.40, T1 gain 0 -.75 / -.54, nuisance -.80 / -.32; T1
        gain .4 .20 / .13. Stage 66 found burden's own part trading against the others there.
      - ADNI measures the WMH volume, so the 'adni' split is the one that applies. A factor ADNI does not measure
        is chosen for only if its quality moves with the measured ones'.
    - **Within one cell (looked at after the reading, so post hoc).** The rule's score ranked the seeds of each
      cell (units, arm) of stages 64, 65 and 60: 43 codes in 9 cells (`causal/stage67/posthoc/within_cell.txt`;
      the scripts are beside it).
      - The anchors' top seed holds in 9 of 9 cells; a random seed would in about 4.6.
      - A holder outscores a failing code of its cell in 42 of 44 pairs.
      - Stage 60's penalty cell was not in stage 66's set, so the score was never checked on it. Its one holder,
        seed 2, comes first.
      - Both misses are in the PCA cell. Its narrow holder (seed 5, last .46) scores below seed 3, which lost lr
        (hidden), and seed 2, by .005.
      - The top seed matches the 10-unit code in 4 of 9 cells; a random seed would in about 3. Among codes that
        hold, the scores are mostly within .01, so they do not order the match criteria.
      - The cells share seeds (seed 4 fails in 7 of 9), so these are not nine independent tests.
      - Top-1 did not separate holders within a cell (items 51–53). The 8-unit failures sit in ventricle and
        burden, which are measured.
    - **What the check leaves out.** The noise is independent and Gaussian, with one draw per bank. FreeSurfer's
      errors may follow scanner and contrast, which is what style holds. The latents stand in for FreeSurfer's
      volumes, which are noisy functions of the anatomy rather than the generator's own factors. The check ranks
      runs within a bank; it gives no threshold for a single run.
    - No expectation was written before this stage.
    - **Reading.**
      - To choose among runs on ADNI, use the measured anchors. The run whose code best reads them (adjusted or
        raw) is close to the best on the factors ADNI does not measure, nearly as close as choosing by the exact
        own parts, and noise down to reliability .5 does not change that.
      - Within one 8-unit cell the same score finds the seeds that hold, post hoc; top-1 could not.
      - The user declined (25 Nov) using the anchors to choose seeds, so stage 69 does not score its children this
        way.

56. **A higher temperature makes 8 units worse: no code holds, and half of every code goes to curves (stage 70,
    read 25 Nov, `causal/stage70/holds.txt`).** Stage 64's from-scratch 8-unit arm (every 500) with the global
    InfoNCE at τ 1, scaled by 10 (`warm.py --tau 1`), seeds 42, 2, 3, 4, 5, beside the from-scratch 8-unit and
    10-unit codes of the same seeds. T1 / FLAIR.
    - **Holds / matches the 10-unit code (stage 50, seed 42):** τ 1 0/5 / 0/5; from scratch 3/5 / 2/5; 10 units 4/5 /
      3/5.
    - **Against the rules fixed before the results:**
      - **Reliable: no; worse.** Holds 0/5 against 3/5 (Fisher p .17), matches 0/5 against 2/5 (p .44): 3 and 2
        fewer, past the "worse" threshold of 2.
      - **Paired by seed:** no rescue (42 and 4 still fail), and all three holders are lost (2, 3, 5).
      - **Mechanism: fails.** Every code spends 4 directions on curves in each view, seed 4 five (from scratch 1 or
        2). Each keeps 4 own directions (seed 4: 3); the spectrum's last value is −.04 to −.02 and the ventricle–
        burden worst direction −.02 to −.01.
      - **Cost: yes.** Cross-view top-1 over all 1,000 subjects falls at every seed, by .058 to .575, mean .277.
        Style: none (linear R² at most .00).
      - **Described:** volume error 52–226 voxels against 9.5–15.5. Seed 4's code reads no volume at all (own R²
        below 0), so the audit gives no error; `holds.py` crashed on it at the end of the stage and now counts it as
        a miss, the rule unchanged, rerun by hand. ATE bias .19–.26 against .07–.16. Decoded graph .47–.70 against
        .76–.85.
    - **What goes:** burden's own part at every seed (R² .05 or less), sulcal (.12–.20; seed 4 −.01), thickness
      (.13–.43) and ventricle at three seeds (.00–.32). Brain (.90–.92), temporal (.80–.87) and lr (.72–.92) stay.
      The step-14,000 evals already showed it (stage 70 entry, "Interim").
    - **Expectation, written before:** wrong. It predicted fewer curves and most likely 4 of 5 holding; curves
      doubled and none held. The stated risk, that the weak own parts lose room, happened in full, and to more
      factors than burden and lr.
    - **Reading.**
      - The number of curves is the width less the own parts held, at every width and arm: 8 units 7+1 or 6+2, 9
        units 7+2 or 6+3, 10 units 7+3, τ 1 4+4 or 3+5. Counting curves measures what was lost, not why.
      - τ 1 lands where InfoNCE alone did. Stage 62's check found 4 own directions and 6 curves at 10 units with
        uniform batches and no prediction term; with prediction 5 and 5; with code batches too 7 and 3.
      - So the low temperature is what lets the code batches work. A soft loss barely separates neighbours that
        agree on what the code holds, so the weak factors stop paying, and bent copies of the strong ones take their
        units.
      - The 1/τ argument was wrong about what the temperature does. The isotropy term left at τ 1 is met by bending
        as well as by new factors.
      - Not a lever. Section 11's hard-negative weighting (sharper) starved lr and temporal, so both directions of
        the temperature are worse than the recipe's 0.1.
    - **Why 8 units fail (synthesis, 25 Nov, from items 46–56 and stage 62's check):**
      - InfoNCE reads the code through a fixed cosine on standardized units, so every unit counts alike, both in
        spreading subjects and in the views' agreement, and no unit can sit idle (item 46).
      - A unit holding a bent copy of a strong, cleanly seen factor (brain size, read at .94 in both views) spreads
        subjects and costs no agreement. A unit holding the weakest own direction (the ventricle–burden difference,
        9% of the pair's variance, burden the factor the views determine least exactly) costs agreement. It earns
        only where the batch is not already told apart, and InfoNCE stops paying once it is (warm.py's prediction
        docstring).
      - InfoNCE alone therefore prefers bending (4 own directions of 7 at 10 units). Code batches, the prediction
        term and the low temperature make information pay. At 8 units, 7 degrees of freedom under the cosine (section
        8), the two pulls are nearly even, so the seed decides. A 9th unit buys a curve (item 51), more steps change
        nothing (item 52), and every lever that adds weight on one side only moves which factor goes (items 49, 54,
        56; section 11).
      - So the curves are InfoNCE's preferred use of the code's capacity, not a side effect to charge (stage 60) or
        soften away (stage 70). A fix has to stop that competition for the code's units. The direct one: a projector
        head, so the bending and the weighting of noisy units happen after the code. If the head's output stays at
        the matched width (8 cosine units), the theory applies to the whole encoder, and its guarantee, an output
        that is a bijection of the content, needs the 8-unit code to carry all of it. Stage 71 tests it.

57. **A seventh shared local direction does not mark holding, so the suggested local loss would not target the
    missing factor (stage 72, read 25 Nov, `causal/stage72/local_directions.txt`).** `local_directions.py` on the 48
    fixed-bank codes with hold verdicts, no training.
    - **Against the rule fixed before the results: fail.**
      - Own neighbourhoods: AUC of s7 for the 35 8-unit codes, holders against failures, .718 at k = 32 and .667 at
        k = 64. The pass needs .80 at both. It is .656 at k = 128 (described only).
      - The signal is real shared variation: every holder's s7 exceeds all 5 of its shuffled values (14 of 14). The
        nulls are at most .007, against s7 of .10–.25.
      - Fixed neighbourhoods (labels) fail the same way: .680, .673, and .670 at 128.
      - So the outcome is "both fail". The weakest local shared direction does not mark holding at these sizes, and
        the loss is not to be built in this form.
    - **Why it fails.** In trained codes the seventh local direction is weak whether the code holds or not.
      - Profiles at k = 32: holder c8_e500_s2 .68 .62 .56 .49 .44 .35, then .10 .02; failure c8_e500_s42 .80 .64 .53
        .45 .38 .27, then .10 .03.
      - A holder's seventh own part is readable across the population (ridge R² .5–.6), but it adds little shared
        variation among neighbours: about what a failure's curve leaves.
      - The calibration's 7 + 1 code has a seventh direction like its others (.21 at k = 32, .34 at 128). No trained
        code looks like that.
    - **What it rewards instead.**
      - The compressed starts (pca, red) read high s7 whether they hold or not (.20–.25; pca's holder–failure pairs
        are all on the wrong side, 6 of 6). So do all five 10-unit codes, seed 4's failure included (.25–.32).
      - Those failures keep a strong seventh shared direction that is not an own part. The likeliest candidate is the
        lesion's shift, which the 10-unit parents hold along directions of their own (item 46); not checked.
      - That is the shared artifact the suggestion itself warned about.
    - **What it does track.**
      - Spearman over the 8-unit codes: +.72 with top-1 over 1,000 and +.59 with the spectrum's last value. It
        measures how well neighbours are told apart, as top-1 does, more than whether the seventh own part is held.
      - The temperature-1 codes, which keep 4 own directions, read s7 near 0 (.002–.006). So the measure does see a
        code that has lost several local directions, and curves do add almost no local shared direction there.
    - **Within cells** (k = 32, holder–failure pairs on the wrong side): e500 2/6, e100 0/6, cont 2/6, pca 6/6, red
      2/6, harm 1/4; 9 units e500 0/3, e100 1/3. Pooled at 9 units the AUC is .75. As a label-free way to recognize
      holders it is no better than top-1.
    - **Expectation, written before:** a pass, narrowly at k = 32. Wrong. It failed at both sizes and with labels
      too; the synthetic tests had already made k = 32 look harder.
    - **Reading.** Half of the premise holds: curves add almost no local shared direction. But a held weak factor
      adds little more, and a shared artifact adds more. Rewarding the seventh local direction would therefore not
      pick out the missing factor. Not a training target in this form.

58. **A projector head does not make 8 units reliable: it never bends, the curves stay, and holds fall to 2/5
    (stage 71, read 25 Nov, `causal/stage71/holds.txt`, `own_directions.txt`).** The PC's five, paired by seed with
    the from-scratch 8-unit codes; the A100 replicate is read separately (stage 71a).
    - **Against the rules fixed before the results: no better.** Holds 2/5 (seeds 3, 5) against 3/5, and matches 1/5
      (seed 3) against 2/5: within 1 both times, so "no better", not "worse" (that needs 1/5 or none). Paired by seed
      it rescues neither failure (42, 4) and loses seed 2 (last value .32 / .29, ventricle–burden worst .43 / .40).
    - **Mechanism: fails.** Holders spend one direction on curves and failures two, exactly as from scratch: 2 of 5
      codes have at most one curve per view (from scratch 3 of 5), and 3 codes (42, 2, 4) still spend two. Every code's
      eighth direction is a curve of brain size (boosted .96–.99 from the seven factors, brain alone .77–.86). The
      second curve in the failures reads .72–.79 by boosting.
    - **The head stayed linear:** linearity .979–.994 at step 20,000. The arm tested a learned linear reweighting of
      the units, not bending after the code.
    - **Which direction goes still moves with the seed.** Seed 4 now holds the ventricle–burden plane (worst .52 /
      .50, from scratch .22 / .19) but loses a mix of sulcal, lr, thickness and temporal (last value .00). Seed 42 still
      loses ventricle–burden (worst .18 / .19). Seed 2 loses part of ventricle, burden and sulcal.
    - **Costs: none by the rules.** Cross-view top-1 changes by −.01 on average (paired: +.02, −.05, .00, −.03,
      +.01). Style at most .02 in any code and view.
    - **Described:** ATE bias is higher in the holders (.09–.11 against .07–.08 from scratch); decoded graph within
      .01 of from scratch except seed 4 (.73 / .71 against .78) and seed 42 (.81 against .76–.77).
    - **Expectation, written before:** curves leave, the head bends, holds rise to about 4/5. All three wrong. The
      named failure ("without helping": holds near 3/5 with fewer curves) did not happen either: the curves did not
      fall. The head left the tie between a curve and the weakest own direction where it was.
    - **Reading.** Seventh training change at 8 units without effect on the hold rate (rebuild interval, prediction
      weight, compressed start, longer training, harmonic penalty, temperature, projector head). Not a lever.
    - **A100 replicate** (stage 71a, `causal/stage71a/holds.txt`, read 25 Nov 18:05; described only, as fixed
      before): holds 1/5 (seed 2) and matches 1/5 (seed 2). The ten projector codes pooled: holds 3/10, matches
      2/10, against 3/5 and 2/5 from scratch. The holder spends one direction on curves and the four failures two
      (their seventh direction .00–.26). Linearity .977–.994, top-1 −.01 on average, style at most .03.
      - **No seed held on both GPUs.** The PC held at seeds 3 and 5, the A100 at seed 2. The same seed and flags on
        another GPU flipped the outcome at three of five seeds (2, 3, 5). Whether a run holds is settled along the
        trajectory, not fixed by the seed, so it behaves like a coin flip per run: a reason to pool or choose among
        runs (the cross-seed check, the merge) rather than to look for good seeds.

59. **The backbone's shared leftover past the code tells failing codes from holders without labels (AUC .97, every
    within-cell pair right); merging runs and repairing a code by the leftover both fail (stage 75, read 25 Nov,
    `causal/stage75/pool_check.txt`).** 45 fixed-bank 8-unit codes in nine cells, no training.
    - **Against the rules fixed before the results:**
      - **Merge: fails.** The merged code holds in 4 of the 8 cells with a holder (e500 narrowly at .42 / .44, red,
        pca, proj64), against the rule's 7; 4 or fewer is "fails".
      - **Detector: works.** ρ1, the held-out correlation of the top direction the two views' leftovers share, is
        higher in every failing code than in every holder of its cell (44 of 44 pairs), AUC .97 over all 45 codes.
        Holders read .19–.56 (median .41), failures .45–.86 (median .61), tau1's .75–.87.
      - **Repair: fails.** 4 of 23 failing codes hold after the swap (e100_s42, harm_s5, proj64_s2, aproj64_s42);
        no holder stops holding (0 of 17).
    - **So the lowest ρ1 in each cell is a holder, in all 8 cells with one** (a consequence of the 44 pairs, not a
      separate test). Top-1 put 7–9 codes on the wrong side on the same codes (items 51, 58).
    - **Why the detector works, in the theory's terms.** The views are independent given the content, because
      style is drawn per view. A code that holds all the content therefore leaves nothing the views share once each
      view's backbone features are regressed on it. A code that lost a direction leaves that direction shared. ρ1 is
      a test of the views' conditional independence given the code: it checks the "all" half of block
      identifiability directly. Holders' floor (.2–.5) is the finite-sample map plus the weak parts they hold only
      partly.
    - **Why the repair fails.** The top shared leftover direction is not the lost own part. In failing codes it reads
      the best single own part at r² .02–.24 only; the lost direction shares it with other content the code does not
      hold. The swap raised the ventricle–burden worst direction in some codes (e500_s42 .05 → .32, e500_s4 .22 →
      .44) but rarely to a hold. The removed direction was a curve every time (boosted from the factors .89–.99).
    - **Why the merge fails.** As the expectation's named miss: the merged code spends two directions on curves in
      the failing cells (e500, cont, harm, aproj64), so the bottleneck meets the same tie. Its restarts disagree
      (2–5 of 5 hold per cell), so merging has a lottery of its own.
    - **Described:**
      - Failing runs merged alone hold in 2 of 9 cells (pca, proj64): sometimes the failures' union is enough.
      - Random subsets: 2 runs hold in 5 of 27 merges, 3 runs in 14 of 27. All ten projector runs merged hold
        (.61 / .59). Same-seed PC and A100 pairs hold at seeds 2 and 5, not at 3, 4, 42.
      - Against `c10_e500_s42`, holding merges still miss sulcal (pca, proj64) or ventricle (e500).
    - **Expectation, written before:** merge 7–8 of 8 (wrong: 4); detector AUC .75–.85 (wrong, better: .97, 44 of
      44); repair about half of failures (wrong: 4 of 23).
    - **Reading.** A label-free way to tell which runs hold now exists, and it follows from the theory's own
      assumption (views independent given the content). With it, the frozen recipe plus several runs and the
      lowest ρ1 gives a holding code in every cell here. Not yet shown on runs the rule never saw: new seeds, other
      banks and widths, and ADNI, where every code will leave some shared content and only the ranking within a
      recipe can be read.

60. **At 9 and 10 units the leftover detector does not separate holders from failures (stage 76 part A, read 25 Nov
    23:40, `causal/stage76/existing.txt`).** The fixed bank's 13 codes with verdicts, no training.
    - **Against the rule fixed before the results: fails.** AUC .55 (failures above holders; the rule fails it at .65
      or below) and 7 of 10 within-cell pairs right (it needed 8). Holds recomputed from the bundles match the stages'
      verdicts for all 13.
    - **Per cell** (ρ1 by rank, h holds, f fails):
      - 10 units every 500: .35h .36h .43h .46f .56h, 3 of 4 pairs. The failure is seed 4, which fails at every
        width; it sits between the holders.
      - 9 units every 500: .24h .38h .52h .64f, 3 of 3.
      - 9 units every 100: .20f .31f .31h .55f, 1 of 3. The lowest is seed 4, a near-holder (FLAIR's last value
        .399 against the .40 cut). Seed 5 (.31) lost a thickness–temporal–sulcal mix, not the ventricle–burden
        plane, and leaves little shared leftover for it.
    - The lowest ρ1 per cell holds in 2 of 3 cells (both every-500 cells).
    - **Described, other banks and recipes** (single seeds, all failing by the hold rule): ρ1 .47–.77, all within
      stage 75's failure range (.45 and up); no holder to compare.
    - **Expectation, written before:** holds up, AUC about .9. Wrong.
    - **Reading.** The clean split at 8 units (item 59) does not carry to 9 and 10 units in this form. Pooled across
      cells, ρ1's level differs with width and recipe, and within cells three of the ten pairs are wrong: a
      near-holder, a code that lost a cortical mix the leftover barely shows, and seed 4 at 10 units. Whether the rule
      works at 8 units on runs it never saw is part B's question (item 61).

61. **On ten new 8-unit runs, the lowest ρ1 always picks a holder (stage 76 part B, read 26 Nov 03:45,
    `causal/stage76/new_seeds.txt`).** The frozen recipe at seeds 6–15, each trained as its own Run:ai job on one A100
    (25 Nov 23:00 to 26 Nov 03:35; flags identical to the original from-scratch runs, apart from options added since,
    all at their off defaults), copied back, bundled and scored.
    - **Hold rate 5 of 10** (seeds 6, 10, 11, 12, 13 hold; 7, 8, 9, 14, 15 fail), against 3 of 5 from scratch on the
      PC.
    - **Against the rules fixed before the results: both work.**
      - Detector: AUC 1.00, 25 of 25 holder–failure pairs right. Holders read .32–.44, failures .46–.68.
      - Selection: in all 110 of the 120 subsets of three runs that contain a holder, the lowest ρ1 holds (100%; a
        random pick 55%).
    - **Described:** K = 2, 4 and 5 also pick a holder in every subset that has one (35 of 35, 205 of 205, 251 of
      251). Stage 75's ranges as a fixed cut: every holder at or under .56 and every failure at or over .45, but a
      cut at .50 misclassifies seed 15 (a failure at .46). The thinnest margin is .02: holder seed 12 at .44 against
      failure seed 15 at .46, whose ventricle–burden worst direction reads .39 / .33.
    - **Expectation, written before:** 4–6 of 10 hold (right, 5); AUC about .9 (better, 1.00); the pick holds in
      about 95% of subsets with a holder (better, 100%).
    - **Reading.**
      - At the matched width the rule now holds on 69 of 69 within-cell pairs: 44 in stage 75, 25 here on runs it
        never saw. At 9 and 10 units it does not (item 60). The paper's subject is the matched width.
      - So "train several runs of the frozen recipe and keep the one with the lowest ρ1" gives a holding 8-unit code
        whenever any run holds, with no labels: at a 50% hold rate, 3 runs suffice 7 times in 8 and 5 runs 31 in 32.
      - Read ρ1 as a ranking among runs of one recipe, not as a fixed cut: its level moves with recipe and width.
      - Its justification is the theory's own assumption: the two views are independent given the content, so a
        complete code leaves nothing shared in the backbone (item 59).

62. **The new toy does not reproduce the width-dependent failure, so it cannot test the sufficiency term (stage 77
    step 1, calibration read 26 Nov 10:21, `causal/stage77/calibration.txt`).** Baseline only (InfoNCE plus linear
    prediction), 9 settings × 2 widths × 5 seeds.
    - **Against the rule: no setting meets all three conditions; step 1 stops.**
      - At parent share .5 and view noise .3 the backbone holds the faint own part (.68) and both widths hold: 12
        units 5 of 5 (burden own .62), 8 units 5 of 5 (.56). No narrow-width failure.
      - Every harder setting fails at both widths (12 units hold 0 of 5). The faint own part's availability in the
        backbone falls to .50 and below, so the failures are of seeing, not of the narrow code's competition.
      - The width gap exists but sits below the hold line: at share .7 and noise .3, 12 units read burden's own part
        .34 and 8 units .19 (availability .50); at share .5 and noise .6, .26 against .04 (availability .42).
    - **Why the toy differs from the bank.** On the bank the backbone reads burden's own part at .58–.60 while
      failing 8-unit codes read .30, and 10 units hold in 4 of 5. The toy has no code batches, which item 41 found
      make matched width hold, and its MLP backbone sees the faint part less well than the bank's convolutional one
      at the same difficulty. A finer grid between the two regimes (share .5–.7, noise .3–.6) might find a setting
      that meets the rule; that would be a new calibration, recorded before any sufficiency run.
    - **No sufficiency run was made**, so nothing here argues for or against the term.

63. **The content-size elbow overshoots |C| + 1: the held-out loss keeps falling to 16 units, paid for by curved
    functions of the seven factors, not by style or new content (stage 74, read 26 Nov 11:15,
    `causal/stage74/width_sweep.txt`, `style_audit.txt`, `spare_units.txt`).** Daunhawer et al.'s App. B.2 sweep on
    the fixed-position bank (7 shared latents, so |C| + 1 = 8): stage 56's recipe at widths 4, 5, 6, 7, 8, 9, 10, 12
    and 16 × seeds 42, 3, 4, 27 runs on Run:ai A100s.
    - **Against the rule fixed before the results (E-W): FAIL.** The elbow (smallest width within 5% of the total
      drop above the minimum of the mean held-out InfoNCE) is 16; per seed 16, 12, 16.
    - **The labelled reference lands on the theory.** Mean own-part R² rises to .78 at 8 units and stops (k_own 8);
      the weakest factor (burden) reads .56 at 8, .61 at 16.
      - The loss is not monotone: 7 units (2.01) sit above 6 (1.87) and 10 (1.91) above 9 (1.68).
      - Its seed SD at 9 and 12 (.38, .41) is as large as the steps between widths.
      - Below 8, the loss flattens at 5–6 units while burden, sulcal and ventricle are still missing (they arrive at
        7–8).

      | units | held-out loss | own mean | weakest |
      |---|---|---|---|
      | 4 | 4.57 ± 1.41 | .33 | .02 |
      | 6 | 1.87 ± .20 | .59 | .08 |
      | 8 | 1.71 ± .14 | .78 | .56 |
      | 10 | 1.91 ± .19 | .73 | .56 |
      | 12 | 1.32 ± .41 | .77 | .62 |
      | 16 | .76 ± .11 | .75 | .61 |
    - **No style at any width.** Each view's gain, bias and noise read 0.00 from the trained code at every width,
      linearly and by boosting. The untrained twin's code reads gain .16 at 4 units, rising to .41 at 16. The pooled
      features still hold gain (~.8); none reaches the code.
    - **Where the directions go** (`spare_units.py`: each code whitened so every direction counts alike, each
      direction's variance split by 5-fold fits; mean over 3 seeds × 2 views):

      | units | own parts (linear) | curves of the factors | shared outside the factors | view-private |
      |---|---|---|---|---|
      | 4 | 2.33 | .86 | .01 | .80 |
      | 8 | 5.48 | 1.74 | .13 | .66 |
      | 10 | 5.04 | 3.59 | .23 | 1.13 |
      | 12 | 5.32 | 4.72 | .40 | 1.57 |
      | 16 | 5.17 | 7.50 | .76 | 2.58 |
      - The own parts stop at about 5.3 directions from 8 units on.
      - Every direction past the seventh reads 0 from the own parts and about .75 by boosting from the seven factors.
        The 8-unit code's eighth direction is already all curve (.97).
      - **They are not single-factor harmonics** (unlike stage 56's bent copy of brain size, item 46). Over the 84
        directions past the seventh in the 12- and 16-unit codes, by boosting:
        - one factor reads a median .07 (quartiles −.01–.17);
        - the best pair .30 (.17–.43), most often brain + lr (34), brain + temporal (16), thickness + temporal (10);
        - all seven .75 (.70–.79).
        So each is a joint non-linear function of several factors.
      - Shared content outside the factors stays small (.76 of 16 directions). The fixed bank's residual lesion-position
        jitter (SD 2.7 voxels) is a candidate. Its latent is not in the exported codes, so this is untested.
    - **Reading.**
      - InfoNCE rewards telling subjects apart, and a curved function of known factors separates subjects that the
        linear directions leave close. So the loss cannot tell a new factor from finer resolution of known ones, and
        a label-free width elbow overshoots |C| + 1 on images.
      - This agrees with Yao's guarantee: the code is an invertible, not a linear, function of the content block.
      - The spare directions are multi-factor interactions, so a single-factor harmonic penalty (item 54) would not
        catch them.
      - Matched width stays the paper's width (24 Nov); this sweep is evidence for fixing it from the theory rather
        than by the loss.
    - **Running it.**
      - The jobs: `runai_submit.sh 74`, one job per seed (`WIDTH_SEEDS=...`), then one default job to score all.
      - Two fixes were needed after training. `21e8a25`: `needs_done` demanded causal-learn, which the image lacks
        (`NEEDS_CAUSAL=0` for stage 74). `482d1ce`: score skips the figure without matplotlib. Draw the figure
        locally with `width_sweep.py score ... --figure`.
      - `spare_units.py` ran locally on the 27 exports copied from the cluster's `causal/codes/`. The exports now sit
        in `causal/stage74/codes/` (gitignored; copied from a session scratchpad 26 Nov).

64. **E-W's other two width rules both land on |C| + 1 = 8 on the fixed causal bank: the intrinsic dimension
    cleanly, retrieval only on the mean (stage 74's codes, no training; read 26 Nov, `causal/stage74/width_rules.txt`,
    `width_rules.py`).** The plan named these rules without numbers; `width_rules.py`'s docstring fixed them before
    either was computed on any stage 74 code. A rule passes on a bank if it lands within one unit of |C| + 1.
    - **Rule 2, TwoNN of the widest code: PASS on this bank.** The 16-unit codes read 6.36 (SD .32 over 3 seeds × 2
      views), the bank's seven true factors on the same 1,000 test subjects 6.37 (stage 56's calibration row, repeated
      exactly). Offset 7 − 6.37 = .63, so |C| reads 7.00 and the width 8; every seed alone gives 8.
      - TwoNN rises with width and flattens from 9 units: 3.46, 4.39, 4.60, 5.48, 5.77 (8 units), 6.01, 6.07, 6.16,
        6.36. On the sphere the same within .1 from 6 units on, so the code's length adds no dimension here.
      - So the curves that fill units 8–16 (item 63) add no manifold dimension: they are functions of the same seven
        factors, which is what TwoNN counts and what the loss cannot tell apart.
    - **Rule 3, retrieval within 5% of the 4→16 range below the 16-unit top-1: PASS on the mean, not per seed.**
      Mean top-1 among all 1,000: .169, .612, .693, .638, .771 (8), .775, .764, .783, .734 (16). The margin (.028)
      gives 8. Per seed it gives 7, 6, 6 (two of three outside the tolerance), and an absolute margin of .05 gives 6.
      Top-1 is not monotone (16 units below 12; 7 below 6, SD .23 at 7), so the rule rests on a noisy plateau.
    - **What this does not settle.**
      - One bank of three. The plan's pass needs the fixed independent bank (|C| 7) and the drawn causal bank
        (|C| 10, target 11) too.
      - Rule 2's offset is the bank's own, from its true latents. ADNI has none, so the offset would be borrowed. It
        comes from the latents' distribution, not from the estimator alone: on the same subjects Gaussian noise reads
        7.64 at 8 units and 10.05 at 10 (stage 56's calibration rows), yet the seven factors read 6.37. Whether a
        borrowed offset holds at another |C| and SCM is untested; the drawn bank's sweep (|C| 10) is the test that
        matters for rule 2.
      - Both rules read the code through the gauge's stored statistics (stage 78's question). TailCode's head is
        non-linear, so recomputing the gauge is not a per-unit rescaling and could move both.

## 8. Theory status (Yao et al. 2024; von Kügelgen et al. 2021)

- **Inside the theory:** scan + residual input, separate encoders, independent jitter (a view-specific pose
  latent), patch loss off, the InfoNCE objective.
- **Not covered:** code batches (any neighbourhood rule), hard negatives.
- **Silent:** finite subjects (memorisation), and code width versus the size of the shared block.
- **The claim adopted on 18 Nov (section 10's plan).**
  - The code is an embedding of the shared anatomy: it depends on the shared content only, and the content can be
    recovered from it. This is what `paper/factor-discovery-framework.tex` targets (its identification section).
  - Alignment makes a code content-only at any width. At 24 units the content is recoverable in practice: every
    factor is readable, and stage 34 found no style.
  - A formal version on the synthetic bank is the bridge (section 10): align all units and put the entropy term on a
    learned projection as wide as the shared block. At the optimum the projection is Yao's bijection, and the whole
    code an embedding. Under warm.py's cosine that block is 11 units, and the prediction map must be flexible: stage
    46b. Stage 46 (10 units, linear map) is outside the theory on both counts (the 10 / 24 discrepancy, below).
    Stage 46b ran: the formal version holds the weak factors no better than matched width (item 38). It backs the
    claim's form, not the recipe's numbers.
  - Factors and their graph need identifying information from outside the two views. The framework says so (a block
    does not determine its internal factors), and stages 41–44 show it. Here that information comes from anchors.
- **Matched width costs precision at finite data (stages 35–40).** The shared block is 10 latents (seven SCM factors
  and the lesion's x, y, z; checked in the generator, items 28–30), spread over more than 24 linear directions of the
  features (linear CCA, item 9).
  - **Stage 47 qualifies this (item 39).** Most of the loss below is the lesion's position, a near-unique
    fingerprint that takes about half a narrow code. With the position fixed, 8 cosine units (the matched width for
    7 latents) keep every factor's own part at .33–.51, and 10 units approach 24 units' levels.
  - A code as wide as the block (10 units, or 11 under the cosine) holds every latent, coarsely. It reads the
    lesion's volume to 15–19 voxels against 12–13 at 24 units on both banks, and drops the own parts of ventricle,
    thickness and sulcal as well as burden's (items 30, 32).
  - Nothing tried closes the gap: InfoNCE variants (simulation), a linear or an MLP map (item 26), sharper inputs
    (items 28–29), other prediction targets (item 31) and a larger head (item 32). 11 cosine units recover 1–2 voxels
    and some of ventricle (item 30).
  - Which weaker factors a matched-width code keeps is close to a tie that the seed decides (item 30, code audit).
  - Matched width does not make the units the causal variables: label-free discovery on the code's own units is at
    the random-mixture baseline at 10, 11 and 24 units (item 34), as block identifiability predicts.
  - The theory's matched-width result is for the ideal, infinite-data objective. With 2,000 subjects these objectives
    keep the strongest factors and the coarse readings, and the weaker own parts need spare width.
  - Worth a sentence in the paper.
  - **When an added term keeps the theory's result** (my argument, 17 Nov). If one encoder minimises both Yao's
    objective and an added term, the minimisers of the sum are the encoders that minimise both, so they stay
    block-identifying. Cross-view prediction qualifies when its map is flexible: the other view's features depend on
    the first view only through the content, so a code holding the content invertibly predicts them as well as any
    code can. With a linear map it does not qualify: a 10-unit code predicts more of the target linearly by holding
    harmonics of the sharp latents, as stage 35 showed. The argument holds at a fixed target; warm.py's target is a
    momentum copy that moves.
  - **Which factor is dropped follows from an assumption (item 27).** The theory takes each view to determine the
    content exactly. Burden's own part meets that least well (the labelled code recovers 91% / 94%, the label-free
    backbone ~70%). A testable form: at the matched width, a latent is kept in proportion to how exactly the views
    determine it. Stages 35 and 37 give four 10-unit points for that curve.
  - **Inside the theory:** any fixed, label-free transform of one view from its own modality, so a residual with
    more components removed is covered, as the 16-component one is.
  - **The 10-unit arms are one dimension short for InfoNCE.** warm.py's global InfoNCE is cosine (hard-coded), which
    sees only the code's direction: 9 degrees of freedom in 10 units. The prediction term reads all 10. Under a
    cosine, the theory's matched width for 10 latents is 11 units, or 10 with a distance similarity (Zimmermann et al.
    2021 identify sphere-valued latents with the cosine). Stage 38's 11-unit arm is that formal case: it recovers 1–2
    voxels and some of ventricle (item 30), so matched width falls short there too.
  - **The generator's shared block is exactly the 10 latents** (checked 17 Nov in
    `eval/synthetic/synthetic_dataset.py`). With clean content the deformation and fissure fields are zeroed and the
    structure is a deterministic function of `z_content` and `z_lesion`. Gain, bias, noise and the field are drawn
    per view. So the 10-unit code is not short of hidden shared content. The error the views' burden readings share
    (.29–.40 at the image filter) must then be a function of the latents, likely leftover anatomy in the residual.
- **The 10 / 24 discrepancy (18 Nov).** Yao's theorem fixes the block it identifies (the 10 latents) and, for its
  entropy step, the width of the code that holds it. It says nothing about extra units, and at no width are the units
  the variables (item 34). So the paper states the gap rather than removing it.
  - **Stage 46 is not the formal case.** It differs from the theory twice: 10 cosine units hold 9 degrees of freedom
    (the bullet above), and its prediction map is linear, which the added-term argument excludes. Stage 46b (section
    9) fixes both: the entropy on 11 units, and the MLP map. Both keep the moving momentum target.
  - **Stage 46's loss was also degenerate and the run collapsed (item 37).** The theory's split into alignment plus
    entropy holds only in the large-batch limit, and that finite-batch form did not approximate it. The bridge is now
    InfoNCE on the block plus alignment alone on the rest, which keeps Yao's objective exactly on the block.
  - **How the paper states it.**
    1. Theorem (Yao): with 11 cosine units, alignment plus entropy gives a bijection (a one-to-one map) of the 10
       shared latents.
    2. The recipe's 24-unit code: alignment makes it content-only at any width; that it holds the content is
       measured, not proven. The theory-covered bridge (alignment on all 24, the entropy on 11, a flexible map) is an
       embedding at the optimum but holds the weak factors no better than matched width (stage 46b, item 38).
       Everything the plan uses at the block level (the block node, adjustment for confounding, independence tests)
       depends only on what the code determines. Linear tools at finite data (Fisher-Z, ridge) favour the wider
       code, which is easier to read linearly.
    3. Finding: a narrow code is fragile. One distinctive shared feature (here the lesion's position) can take most
       of it and crowd out the weaker factors (items 29, 39); without such a feature, matched width keeps every
       factor (item 39). Spare width protects against it, but only when InfoNCE's entropy acts on the extra units
       (item 38). At the theory's optimum those units are redundant, so their value is finite-sample only. Say so,
       and do not claim that matched width fails in general.
  - **What 24 measures.** The label-free count is the number of linear canonical directions the views share above
    r .5. A curved 10-dimensional block spans more than 10 linear directions, so 24 is the code's width rule, not an
    estimate of the block's size. On ADNI there is no finite block, so the bridge (which needs its size) is
    synthetic-only. There the claim is alignment (content-only at any width) plus measured recoverability: the
    anchors read from the code, and no style.
  - **A width rule for the prediction term (my argument, 18 Nov, unchecked).** Assume the views are independent given
    the content and the target is fixed. At width d, the linear prediction term is minimised by the d directions of
    the other view's whitened GAP that this view predicts best (reduced-rank regression). These are a function of the
    content only. The code is an embedding once those directions separate the content. That number is at least 10,
    and it grows with curvature (the harmonics of item 27) and with how noisily a factor is read. It explains why 10
    units drop burden and what the label-free count approximates, and it needs no block size, so it carries to ADNI.
    It explains the recipe but does not guarantee it, because InfoNCE's optimum and this one differ.
  - **Optional check, no training:** the intrinsic dimension of the 24-unit code (for example TwoNN), calibrated on
    the true latents passed through a random non-linear embedding at 1,000 subjects. An embedding of the block should
    read about 10. Weak factors look like noise at small scales, so this supports the claim and does not decide it.
- **The CCA readout is not covered as used.**
  - It is fitted after training, it is linear, and it keeps weakly shared directions. Directions 15–24 agree at only
    r .45–.72, so part of each is view-specific (style or noise, unmeasured).
  - Use it as a diagnostic of how wide the shared block is.
  - The theory-backed counterpart is latent correlation maximisation trained end to end (Lyu et al., ICLR 2022): align
    the two views' codes under a whitening constraint, which identifies the shared block. Barlow Twins is a close
    relative.
- **Label-free factor-level discovery** needs views that share different subsets of factors, or interventions.
  Two views that share everything identify only the block. Stages 42 and 44 show it on the trained codes (item 36):
  a label-free count beside the anchors keeps the strong novel factors and loses the weak own parts in directions
  mixed with other shared content.
- **Anchors identify a concealed factor's information, not its place in the graph** (found building stage 42, 17
  Nov). A new shared direction found beside the anchors is identified only up to adding a function of the anchors:
  the code reads P and P + h(anchors) equally well, and both agree across views. The label-free choice, the residual
  of the anchors, is uncorrelated with every anchor, so PC connects it to nothing, whether the true edge runs into P
  or out of it. A concealed mediator then shows only as an anchor–anchor edge the reference graph lacks (ventricle –
  temporal through burden). Placing P needs directions taken in the whole code under some further structure (the
  design's step 3b: spatial footprints), or labels.

## 9. State on 17 Nov

- **Stage 46 (ran 18 Nov; collapsed, item 37: its loss was degenerate): the bridge.** The plan's step
  1 (section 10): does a 24-unit code trained the theory's way hold what stage 33's code holds?
  - **Arm:** stage 33's 24-unit prediction arm (uniform batches, prediction weight 1, seed 42, 20,000 steps,
    `bank_causal_nofield`) with `--entropy-units 10`: InfoNCE's alignment on all 24 units, its entropy only on the
    first 10.
  - **Why this is the theory's version** (section 8). Alignment makes every unit content-only at any width. The
    entropy on a block as wide as the shared latents (10) makes that block a bijection of them, so the whole code is
    an embedding. With the entropy on all 24 aligned units, as in the recipe, the theory's argument does not apply: an
    aligned 24-unit code of 10 latents has no finite entropy.
    - Correction (18 Nov): stage 46 is not quite that version. Under the cosine its 10 units hold 9 degrees of
      freedom, and its prediction map is linear (section 8, the 10 / 24 discrepancy). Stage 46b is the formal arm.
  - **Reports** (`causal/stage46/`): own variation (with boosting), the precision audit, the code audit and the style
    audit on the bridge's 24-unit code, its first 10 units (`bridge_view.py`), stage 33's 24-unit code, and the
    10-unit codes of stages 35 and 38 (seeds 42 and 2; their entries reused from stage 43).
  - **Reading:**
    - The bridge passes if its 24-unit code holds the factors as stage 33's does (own parts within seed noise, about
      .10; volume precision within a voxel or two) and carries no style. The paper can then state the formal claim
      with the recipe's numbers.
    - Its first 10 units are a matched-width code trained with the 24-unit code's support. If they hold the weak own
      parts better than the standalone 10-unit codes (T1: ventricle .29, thickness .20, sulcal .07, burden .33), the
      bridge also answers the 10-unit question in a form the theory covers (stage 45).
  - **Checked locally:**
    - `tests/test_warm_bridge.py`, 4 tests: with the entropy on every unit the loss is the recipe's; it equals
      alignment on all units plus the entropy on the block; units past the block get only the alignment's gradient;
      the view exports exactly the first units. Stage 45's 14 tests (`tests/test_warm_prediction.py`) still pass.
    - A 2-step bridge arm on the Mac's copy of `bank_causal_nofield` trains and saves, and its log states the split.
      `bridge_view.py` writes its first 10 units, and `code_audit.py` reads both runs (24 and 10 units) in about 12 min.
      A disallowed combination (`--entropy-units 24` on a 24-unit code) stops with its message.
    - `bash -n` passes, and `DRY=1 bash out/burden_feasibility_20261102/run_pc.sh 46` prints the arm, the view and the
      four reports.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 46` (one training arm, then the reports).
- **Stage 79 (built 26 Nov, rules written before any of its results; the user asked for it): do E-W's width rules
  find the drawn causal bank's width?**
  - **Why.** Item 64: on the fixed causal bank (|C| 7) rule 2 (TwoNN + offset) gives 8 at every seed and rule 3
    (retrieval) 8 on the mean only; rule 1 (loss elbow) fails. The plan needs three banks. The drawn bank has more
    shared latents (seven factors and the lesion's x, y, z; |C| 10, target 11) and a different latent distribution,
    so it also tests whether rule 2's offset, which ADNI would borrow, holds at another |C|.
  - **Arms.** Stage 74's recipe unchanged (code batches every 500, 31 neighbours, residual, jitter 2 / 3°,
    prediction weight 1, tails, InfoNCE, 20,000 steps) on `bank_causal_nofield`, widths 4, 6, 8, 9, 10, 11, 12, 14 and
    16, seeds 42, 3, 4: 27 runs (`run_pc.sh 79`; Run:ai in section 3).
  - **Reports** (`causal/stage79/`): `width_sweep.py` (`--latents 10`), `width_rules.py --latents 10 --bank`, the
    style audit. width_sweep's own-part reference reads the seven factors only, not the lesion centre, so its k_own is
    no reference for 11 here; read the rules against the target.
  - **Rules** (as item 64; each passes on this bank if within one unit of 11):
    - rule 1: the loss elbow (`width_sweep.py`, 5% of the drop);
    - rule 2: round(TwoNN of the 16-unit codes + (10 − TwoNN of the bank's ten true latents on the same subjects)) +
      1;
    - rule 3: the smallest width whose mean top-1 among all 1,000 is within 5% of the 4→16 range below the 16-unit
      code's.
    - **Offset transfer** (what ADNI needs): rule 2 again with the fixed bank's offset (.63) in place of this bank's.
      It transfers if that also lands within one unit of 11.
    - A rule that passes here and in item 64 still needs the fixed independent bank before it sets ADNI's width.
  - **Expectation, written before any result.** Rule 1 overshoots again (the curves of item 63 do not depend on the
    bank). Rule 2 passes with this bank's offset; the offset is larger than .63, because the lesion centre is a
    bounded, clustered variable on which TwoNN underreads, so the borrowed offset lands at 10 or below and is marginal.
    Rule 3 is again noisy per seed. On the drawn bank the lesion's position takes about half of a narrow code (item 39),
    so codes below 11 units should lose weak factors first and top-1 may keep rising past 11.
- **Stage 78 (no training; rules written 26 Nov before any of its results; the user asked for it): do TailCode's
  stored gauge statistics change any synthetic verdict?**
  - **Why.**
    - The gauge is TailCode's per-view BatchNorm (no learned scale or shift) between the pooled features and the head.
      Training standardizes each batch by that batch's own statistics. Every evaluation and export reads the code in
      eval mode, through the gauge's stored running averages (momentum .1), which the loss never uses.
    - On ADNI the stored averages were far off (stage 73, section 10): top-1 .831 stored against .953 with the gauge
      recomputed, on the same checkpoint.
    - Every synthetic hold verdict, own part and ρ1 so far was read through the stored averages (`causal_bundle.py`,
      `code_audit.load_run`, `pool_check.py export`).
  - **Codes.** A: stage 76 part B's ten runs (`..._abl_codebatch_predict1_c8_20k_a100_sN`, seeds 6–15, labels
    `c8_ae500_sN`). B, after A: stage 75's 45 codes (nine cells × seeds 42, 2, 3, 4, 5, its labels and verdicts).
  - **Two exports per code,** from the same checkpoint, for the same subjects (2,000 training and 1,000 test, both
    views):
    - stored: as in every stage so far;
    - recomputed: each view's gauge mean and variance recomputed over the bank's 2,000 clean training images
      (`warm.recalibrated_pool`), the network frozen and the run's files untouched. Label-free, and no test image is
      used.
    - The recomputed export is a new option, off by default. With it off, the export must repeat the existing one
      (largest difference at most 1e-4); checked on one run before anything else.
  - **Read on both, exactly as stages 75–76:** holds by `own_directions.py`'s analysis and `holds.py`'s rule (the
    own-parts spectrum's last value and the ventricle–burden worst direction each at least .40 in both views, test
    subjects); ρ1 by `pool_check.py detect` (same map, CCA and settings).
  - **Also per run and view:**
    - how far the stored averages are from the recomputed ones, over the 448 pooled features: the mean absolute
      offset (stored mean minus recomputed mean, in recomputed SDs) and the mean log variance ratio (log of stored over
      recomputed variance; below 0, the stored spread is narrower), the gauge's 1e-5 added to each variance;
    - cross-view top-1 among all 1,000 test subjects under each gauge (each view standardized over the test set,
      cosine, both directions averaged, as `holds.py` reads it).
  - **Rules.**
    - **Verdicts.** A code's verdict moves if it holds under one export and not the other. Its margin is the smallest of
      its four hold values (last value and ventricle–burden worst, T1 and FLAIR).
      - No effect: no verdict moves and no margin moves by .05 or more. The recorded hold counts stand.
      - Values move: no verdict moves, but some margin moves by .05 or more. The counts stand; the values are restated.
      - Verdicts move: at least one verdict moves. Each stage's hold count is restated under the recomputed gauge,
        with the flips each way.
    - **ρ1.** Stages 75's and 76's detector and selection rules are re-scored with the recomputed holds and ρ1, at the
      same thresholds:
      - A: stands if ρ1's AUC (failures above holders) is at least .80 and the lowest ρ1 among three runs holds in at
        least 90% of the 120 subsets that contain a holder. Fails at AUC .65 or below, or under 70%. Between:
        unresolved. Judged only with at least two holders and two failures.
      - B: stands if the AUC over the 45 codes is at least .80 and at least 3/4 of within-cell holder–failure pairs are
        right. Fails at AUC .65 or below.
      - Both: the 8-unit count of within-cell pairs right (69 of 69 stored).
    - **Described:** ρ1 stored against recomputed (rank correlation over codes, the largest change); top-1's change;
      the distances; and, for the mechanism, each run's stored averages against the average within-batch statistics
      of code-neighbourhood batches (each anchor with its 31 nearest in the stored code, as these runs'
      `--code-neighbours 31` builds them; clean training images), which is what training's batches look like.
  - **Expectation, written before any result.**
    - The recomputed gauge will not help here as it did on ADNI. These runs all train on code batches, and a
      neighbourhood varies less than the population. So the stored variances should sit below the recomputed ones
      (mean log ratio below 0), and the head was trained on features standardized within neighbourhoods.
    - Stage 64's code audits point the same way (the mode check, 8-unit codes). Within uniform sets of 32, top-1 reads
      .983–.991 with the stored statistics and .957–.975 with each set's own (close to the population's). Within
      code-neighbourhood sets it is the reverse: .918–.940 against .945–.965.
    - So: top-1 among all 1,000 the same or lower under the recomputed gauge; no verdict moves, except perhaps a code
      whose margin is within .05 of the cut; ρ1's order nearly unchanged (rank correlation above .8), so both ρ1 rules
      stand.
  - **Cost:** on the PC's GPU about a minute a run for both exports (about an hour for 55), then CPU only: holds and ρ1
    about a minute a code and export.
- **Stage 77 (rules written 26 Nov before any of its results; the user asked for it, starting with the toy): does a
  sufficiency term make the matched-width code keep all seven own parts in one run?**
  - **Why.**
    - Items 59–61: ρ1, the canonical correlation the two views' features still share once a flexible map from the
      code has explained what it can, tells holders from failures at 8 units in 69 of 69 within-cell pairs. Picking
      the lowest ρ1 among several runs works but needs the several runs.
    - Train on the quantity instead. Code batches (negatives alike in the current code, so the loss pays only for
      what the code lacks) and the cross-view prediction term (whitened before the code's part is removed, so a weak
      direction is diluted, and with a linear map that pays for curves) both approximate it. A term computed after
      the code's part is removed does not dilute the weak direction, and curves earn nothing from it.
    - In the theory's terms: alignment makes the code computable from either view alone (a Gács–Körner common
      part: "only"); sufficiency makes the views independent given it (a Wyner common variable: "all"). Together at
      matched width they ask for exactly block identifiability.
  - **The term.** Targets are each view's whitened momentum-teacher features (as the prediction term's). A small MLP
    per view (the explainer) predicts its own view's targets from that view's code, trained by squared error with the
    code detached, so the explainer cannot pay the code for style. The residuals (targets minus the explainer's
    output, the explainer's weights frozen for this part) are whitened by running statistics, and the loss is the
    sum of squared cross-view correlations between the two views' whitened residuals (the squared canonical
    correlations). Its gradient reaches the encoder only through the code.
  - **Step 1, the toy** (`sufficiency_toy.py`, no images, CPU). The toy the paper plan cites for E5a
    (`training/prediction_identifiability_numerical.py`) is not on this machine or in git, so a new one is built for
    this question.
    - Model: seven shared latents; six strong and independent; the seventh (faint) gets a parent share and its own
      part. Each view sees the latents through its own fixed non-linear map, with three style latents of its own,
      noise, and its own noise on the faint latent. 2,000 training and 1,000 test subjects. Per view an MLP encoder
      (a 64-wide hidden layer as the backbone, then an 8-unit code). Scored by ridge R² of each own part from each
      view's code on test subjects; a code holds if every own part reads at least .40 in both views.
    - Baseline, the recipe's objective in small: InfoNCE (cosine on per-unit batch-standardized codes, τ .1, batch
      32, negatives from the batch) plus linear cross-view prediction (weight 1) of the other view's whitened teacher
      features.
    - **Calibration, on the baseline alone, before any sufficiency run:** the faint latent's parent share and view
      noise are chosen from a small grid on seeds 0–4 so that (a) the teacher features hold the faint own part
      (ridge R² at least .5), (b) a 12-unit baseline code holds it (at least .40 in both views) in at least 4 of 5
      seeds, and (c) the 8-unit baseline loses it in at least 3 of 5. If no setting in the grid meets all three, the
      toy is uninformative and step 1 stops there.
    - The grid: parent share .5, .7, .85 crossed with view noise .3, .6, 1.0 on the faint latent. Of the settings
      that meet (a)–(c), the one where the 8-unit baseline fails most often is used; ties go to the parent share
      nearest the bank's .7 (ventricle explains about 70% of burden), then to the lower noise. Availability (a) is
      read from the 8-unit baseline's trained backbone, averaged over views and seeds.
    - Arms at seeds 0–9 in the chosen setting: the baseline; the baseline plus sufficiency (weight 1); and lean,
      InfoNCE plus sufficiency without the prediction term.
    - **Rule:** sufficiency passes the toy if baseline plus sufficiency holds in at least 8 of 10 seeds and in at
      least 3 more than the baseline. It fails if it holds in at most 1 more than the baseline. Between: unresolved.
      A cost if the strong own parts fall more than .05 below the baseline on average.
    - **Described:** the lean arm; ρ1 at the end of training in each arm; seeds 5–9 alone (never used to calibrate).
    - **Built 26 Nov:** `sufficiency_toy.py` (`calibrate`, then `arms --share S --noise N`; outputs in
      `causal/stage77/`). Self-test: on fixed codes with fitted explainers, the term's running squared canonical
      correlation reads .87 when the code lacks a part both views' targets share and .002 when it holds it. A run of
      8,000 steps and its scoring take about 35 s on one CPU thread.
    - **Calibration read 26 Nov 10:21 (item 62): no setting meets (a)–(c), so the toy is uninformative and step 1
      stops, as the rule says.** No sufficiency run was made.
  - **Step 2, built 26 Nov (the user chose to go straight to it and agreed to the Run:ai jobs once the tests pass):**
    - `warm.py --sufficiency-weight W [--sufficiency-hidden H]` (classes `RunningMoments`, `Sufficiency`). The local
      `warm.py` is now the other session's (`1bf7c96`) plus this option, so the later merge is trivial and the cluster's
      other modules match. Off by default; with it off every code path and random draw is as before.
    - The term as built: on the contrastive batch, with the prediction term's whitened teacher GAP as targets (the
      teacher is built even without `--predict-weight`). Explainers 8 → 128 → 128 → 64 per view, trained in the main
      optimizer by squared error with the code detached. Residuals (explainers frozen) are whitened by running moments
      that start at mean 0 and covariance I, then each channel is standardized over the batch inside the graph. The
      running cross-view matrix starts at zero, and the term is sum(running × batch matrix).
    - Two faults found by smoke runs and fixed before any result. First, moments seeded from the first batch (32
      subjects, 64 channels) were rank-deficient, and the running values reached 240. Second, without the per-batch
      standardization the term is linear in an unnormalized matrix, which the encoder could drive down without bound
      by inflating residuals faster than the running whitening follows; the values grew to 41 by step 300.
    - With both fixes the values stay bounded but are not exact canonical correlations early on: the top value reads
      4.5–4.6 at steps 200–400 and 1.2 by step 600, while the lagging whitening leaves channels correlated within a
      view. The fixed point (zero) means the views' residuals are uncorrelated.
    - Tests, `tests/test_warm_sufficiency.py`, 6: options refused with a message and nothing written; the term reads a
      shared part the code lacks (> .6) and not one it holds (< .15); a curve does not stand in for the missing part;
      gradients reach the code from the term and the explainers only from their fit; values stay at or under 1 when
      the first batches are rank-deficient; the moments whiten. The other warm suites pass (39), and so do the other
      session's 7 readout tests run against this `warm.py`.
    - Smoke, 600 steps of the recipe plus sufficiency on the fixed bank: InfoNCE and prediction track the recipe
      step for step (11.33 against 11.44 at step 100, 7.50 against 7.12 at 500); about 15% slower. The lean arm runs;
      `code_audit.load_run` loads both checkpoints (the term is saved under its own key).
  - **Step 3, the runs** (rules above; seeds 16–25, none used on this bank; one A100 job each, `runai_warm.sh` with
    `WARM=stage77/warm.py`: this `warm.py` copied into an untracked `stage77/` folder on the cluster, so the other
    session's checkout and its tracked `warm.py` are left alone):
    - (a) `..._abl_codebatch_predict1_suff1_c8_20k_a100_sN`, labels `c8_asuff_sN`: the frozen recipe's flags plus
      `--sufficiency-weight 1`.
    - (b) `..._uni_suff1_c8_20k_a100_sN`, labels `c8_alean_sN`: `--pool tail --t1key t1_04 --uniform --residual
      --jitter 2 --rotate 3 --patch-weight 0 --content-units 8 --contrastive-loss-type infonce
      --sufficiency-weight 1`, 20,000 steps, evals every 2,000.
    - Scored as stage 76 part B (`run_pc.sh 77b`: bundles, `pool_check.py detect` for holds and ρ1), plus
      `style_audit.py` on both arms for rule (b)'s style bound.
    - **Submitted 26 Nov 10:49** as `burden-s77-suff-sN` and `burden-s77-lean-sN`. The first pods crashed at start:
      `warm.py` reads `reference/flags.txt` beside itself, which `stage77/` lacked. A symlink `stage77/reference ->
      ../reference` (10:51) fixed it, and Run:ai restarted the crashed pods on their own (each workload allows several
      attempts; a restart rewrites its log). No training step ran before the fix.
  - **Steps 2 and 3, only if the toy passes and the user agrees to the Run:ai jobs:**
    - `warm.py --sufficiency-weight W` with unit tests: a complete synthetic code scores about 0 and an incomplete
      one high; curves do not lower the score; no gradient reaches the explainer's weights from the term, nor the
      code from the explainer's own fit.
    - Ten new seeds of each of two arms on Run:ai, scored like stage 76 part B:
      - (a) the frozen recipe plus sufficiency (weight 1): works if at least 9 of 10 hold (the recipe: 5 of 10 at
        seeds 6–15); fails at 6 or fewer.
      - (b) lean: residual input, jitter, tails, InfoNCE on uniform batches and sufficiency, with no code batches and
        no prediction term: works if it holds at least as often as the recipe (5 of 10 or more) with no style
        above .10; then it is the candidate for the paper's recipe.
  - **Expectation, written before:** calibration succeeds (item 27's simulation showed this pattern at 10 units).
    Sufficiency holds in 8 or more of 10 toy seeds; lean holds about as often but may cost the strong parts some
    precision without the prediction term.
- **Stage 76 (rules written 25 Nov before any of its results; the user asked for both parts, the Run:ai jobs
  included; part A read 25 Nov, item 60: fails at 9 and 10 units; part B read 26 Nov, item 61: both rules work on
  ten new 8-unit runs): does the leftover detector hold on codes it never saw, and does "keep the run with the
  lowest ρ1" pick holders among new runs?**
  - **Why.** In stage 75 (item 59), ρ1 told failing codes from holders in all 44 within-cell pairs, but on the 45
    codes that motivated it. Two tests on codes it never saw.
  - **Numbering.** The other session pushed its own stages 73 (why stage 68's evaluated code dipped) and 74 (a
    content-size sweep) on 25 Nov. The merge and leftover stage built here as 73 is renumbered 75, and this one is 76.
  - **ρ1** exactly as stage 75 (`pool_check.py detect`: same export, map, CCA and settings). Holds are recomputed from
    each run's bundle with `own_directions.py`'s analysis and `holds.py`'s rule, and checked against a stage's verdict
    where one exists.
  - **A. Codes it never saw, no training:** the fixed bank's 9- and 10-unit codes with verdicts. 9 units rebuilt every
    500 (seeds 42, 3, 4, 5; 3 hold) and every 100 (42, 3, 4, 5; 1 holds), stage 64; 10 units every 500 (42, 2, 3, 4,
    5; 4 hold), stages 64–65. 13 codes, 8 holders, 5 failures, 10 within-cell pairs.
    - **Rule:** holds up if the AUC (failures above holders) is at least .80 and at least 8 of the 10 within-cell pairs
      are on the right side. Fails if the AUC is at most .65. Between: unresolved.
    - **Described** (single seeds, no pairs): the fixed bank's uniform-batch 8- and 10-unit codes, the drawn bank's
      10- and 11-unit codes, the independent fixed bank's 8-unit code. On the drawn bank the lesion's position is
      shared content that the hold rule does not count, so a code can hold yet leave position behind.
  - **B. New runs:** the frozen recipe (stages 56 and 64's 8-unit arm; flags as in
    `runs/..._abl_codebatch_predict1_c8_20k_s2/report.json`: code batches of 31 neighbours rebuilt every 500,
    residual, jitter 2, rotate 3, tails, prediction weight 1, patch weight 0, 20,000 steps, evals every 2,000) at ten
    seeds never used on this bank, 6–15. Each is its own Run:ai job on one A100 (`runai_warm.sh`; names
    `..._abl_codebatch_predict1_c8_20k_a100_sN`, labels `c8_ae500_sN`), then copied back, bundled
    (`causal_bundle.py`) and read by `pool_check.py detect`. The cluster checkout is at the other session's `1bf7c96`;
    its `warm.py` changes touch only `--real` readouts, a refactor of TailCode's pooling that keeps its arithmetic,
    and `--keep-checkpoints`.
    - **Detector rule:** works if the AUC is at least .80; fails at .65 or below; between: unresolved.
    - **Selection rule** (the way it would be used: train K runs, keep the lowest ρ1): over all 120 subsets of 3 of
      the 10 runs, the pick holds in at least 90% of the subsets that contain a holder: works. Below 70%: fails.
      Between: unresolved.
    - Both rules are judged only with at least 2 holders and 2 failures among the ten; otherwise described.
    - **Described:** a random pick's rate in the same subsets; K = 2, 4 and 5; the ten runs' hold rate (from scratch
      on the PC: 3/5); stage 75's ranges as a fixed cut (holders .19–.56, failures .45–.86).
  - **Expectation, written before:** A holds up, AUC about .9: a spare unit holds a curve, which the map removes. B: 4
    to 6 of 10 hold, AUC about .9, and the pick holds in about 95% of subsets with a holder.
  - **Cost:** A: export 19 runs (about 5 minutes on the GPU), then about a minute a code. B: ten A100 jobs in
    parallel, about 2–3 hours at stage 71a's pace (0.32–0.52 s a step), then about 20 minutes here.
- **Stage 75 (no training; rules written 25 Nov before any code or result; the user asked for both checks; read 25
  Nov, item 59: merge fails, detector works, repair fails; built as 73, renumbered because the other session's
  stages 73 and 74 took those numbers): can the runs' codes be merged into one 8-unit code that holds, and does the
  backbone's leftover past the code carry what a failing code lost?**
  - **Why.**
    - At 8 units about half of runs hold, and seven training changes did not move that (item 58). Whether a run holds
      is settled along its trajectory: the same seed on two GPUs flipped the outcome at three of five seeds.
    - Failing codes lose different directions (ventricle–burden, lr, sulcal); holders keep all seven. So the union of
      a cell's runs holds everything even where most runs fail.
    - The backbone still holds what a failing code lost. Seed 42's from-scratch run: GAP reads ventricle's own part
      .81 / .66 and burden's .60 / .58; its code .43 / .41 and .30 / .32 (stage 56's own variation).
    - Both checks use only the block claim's logic: what the two views share is content, and style is drawn per
      view. No label is read except to score.
  - **Codes:** the fixed bank's 45 finished 8-unit codes with hold verdicts, nine cells of five seeds (42, 2, 3, 4,
    5): e500, e100 (stage 64), cont, red, pca (65), harm (60), tau1 (70), proj (71), aproj (71a). 17 hold; every cell
    but tau1 has a holder. Labels and verdicts come from those stages' `holds.txt`.
  - **Export** (`pool_check.py export`, GPU, reused while newer than the checkpoint): each run's code and backbone GAP
    (64 channels) for its 2,000 training and 1,000 test subjects, both views, into `causal/pool/`. Every fit below uses
    training subjects only (400 of them held out for early stopping where a network is fitted); every score is read
    on the 1,000 test subjects: `own_directions.py`'s analysis per view, and holds as `holds.py` (the spectrum's last
    value and the ventricle–burden worst direction each at least .40 in both views).
  - **A. Merge.**
    - Per cell, each run's code is standardized and whitened on its training subjects, both views stacked
      (`compress_code.whiten`), and the five runs' whitened codes are stacked: 40 dimensions per view.
    - A linear map to 8 units, shared by both views, feeds an MLP (8 → 128 → 128 → 40) that predicts the other view's
      stacked codes. Five restarts; the one with the lowest validation loss is kept (label-free). The merged code is
      the linear map's output on test subjects.
    - Predicting the other view keeps the merged code to shared content. Curves need no unit, because the MLP can
      rebuild them from the factors the 8 units hold.
    - **Rule:** the merge works if the merged code holds in at least 7 of the 8 cells with a holder. It fails if it
      holds in at most 4 of 8 (one run drawn at random holds in 17 of 40 here, so about 3.4 of 8). Between:
      unresolved.
    - **Described:** own parts against the cell's best holder and against `c10_e500_s42`; directions spent on curves;
      the tau1 cell (no holder); each cell's failing runs merged alone (does a direction no single run holds come
      back?); random subsets of 2 and 3 runs per cell (hold rate against the number of runs, for ADNI); the ten
      projector runs together; the same-seed PC and A100 pairs; agreement among the restarts.
  - **B. Leftover past the code.**
    - Per code and view: a map from the code to the standardized GAP, least squares on the code and then RBF kernel
      ridge on what that leaves (settings by 5-fold CV on training subjects). The leftover is the GAP minus that map,
      out of fold on training subjects, so what the code holds, straight or curved, leaves with it.
    - Changed before any real result: in the synthetic test a kernel ridge map alone shrank the linear part, and the
      shrunk content is shared by the views, so a code holding all seven read ρ1 .82. With least squares first that
      code reads .03, and a code that lost one part .98, its top leftover direction reading the lost part at .95.
    - Regularized CCA between the two views' leftovers (ridge .1 of the mean variance), fitted on training subjects.
      ρ1 is the held-out correlation of the top shared leftover direction on test subjects.
    - **Detector rule:** ρ1 should be higher in failing codes. It works if the AUC over the 45 codes (failures against
      holders) is at least .80 and at least 3/4 of within-cell holder–failure pairs are on the right side. It fails if
      the AUC is at most .65. Between: unresolved. (Top-1 put 7–9 codes on the wrong side, item 58.)
    - **Repair:** remove from the whitened code its most predictable direction (`compress_code.redundancy`, one step,
      8 → 7, fitted on training codes) and add each view's top shared leftover direction: an 8-unit code.
    - **Repair rule**, over the eight cells other than tau1 (its codes keep only 3–4 own directions, so one swap
      cannot fix them; described only): the repair works if at least 16 of the 23 failing codes hold afterwards and
      at most 1 of the 17 holders stops holding. It fails if at most 7 of 23 failing codes hold. Between: unresolved.
    - **Described:** what the top leftover direction reads (each own part by ridge, test subjects), ρ2, the GAP's own
      CCA beside the leftover's, and what the redundancy step removed (boosted from the factors, as
      `own_directions.py` reads a curve).
  - **Expectation, written before any result:**
    - Merge: holds in 7 or 8 of 8. Holders' codes carry all seven linearly, and the MLP rebuilds the curves. The
      likeliest miss is item 27's finite-sample effect, the MLP failing to rebuild a curve so that a unit is spent on
      it; that shows as a merged code with two curves. Failing runs merged alone hold in about half of the cells,
      because most failures lose the same ventricle–burden direction.
    - Detector: AUC around .75–.85. Holders leave some shared leftover too (their weakest own part reads about .5 in
      the code and higher in the backbone), so the separation is partial.
    - Repair: about half of failing codes hold. The leftover should carry burden's own part where the backbone does
      (.58–.60), but one direction may mix ventricle's and burden's leftovers. Holders keep holding, since the
      removed direction is brain's curve.
  - **Cost:** the export about 20 s a run on the GPU (about 15 minutes). Then CPU only, run alone: merges take
    minutes a cell; kernel ridge, CCA and one redundancy step take under a minute a code; the scoring about 20 s a
    code and view. Under two hours in all.
  - **Checked locally (25 Nov):** `tests/test_pool_check.py`, 9 tests on synthetic codes (seven independent own
    parts, codes that hold some of them behind a mixing the views share, curves of brain, view noise; backbone-like
    features with every part, a curve and per-view style):
    - The merge keeps what holders hold (every part R² > .8), recovers parts no single run holds from failures that
      lose different ones (> .7), and cannot invent a part no run holds (< .1).
    - The leftover of a code that lost a part shares it across views (ρ1 > .8, top direction reading it > .7), at
      least .2 above a code that holds all; style alone shares nothing (|ρ1| < .2).
    - The repair removes a curve (R² > .8 from the curves) and then holds every part (> .6); on a holder it removes
      brain's curve and keeps every part (> .8). Holds reads the ventricle–burden plane; names and verdict parsing.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 75` (export, then `pool_check.py run` into
    `causal/stage75/`).
- **Stage 72 (no training; built 25 Nov, the user asked for it): can a seventh shared local direction be measured
  without labels?** A suggestion from another assistant (Codex), passed on by the user on 25 Nov: train the 8 units to
  keep seven local changes that both scans agree on, in place of InfoNCE. It proposed checking first that the signal
  is measurable, on the saved codes. Stage 72 is that check, with three changes from item 56's discussion: each
  code's own neighbourhoods at the batch's size, held-out scoring within each neighbourhood, and no pooling across
  neighbourhoods.
  - **Why.** Among similar subjects a curve moves with the factor it bends, so it adds no local direction. An
    independent factor that both views see does add one. A code that holds all seven own parts should then show seven
    local directions the views share, and a code that lost one should show six plus a curve's leftover. Pooling
    across neighbourhoods would undo this, since a curve's local slope changes from one neighbourhood to the next.
  - **Codes:** the 48 fixed-position causal-bank codes with a hold verdict (stages 64, 65, 60, 70): 35 at 8 units
    (14 hold; cells e500, e100, cont, pca, red, harm, tau1), 8 at 9 units (4 hold), 5 at 10 units (4 hold). Each
    bundle's 1,000 test subjects, both views.
  - **Statistic** (`local_directions.py`, new):
    - Each code is whitened by the covariance of its two views' mean code. Around each of the 1,000 subjects, a
      neighbourhood of k subjects (the subject and its k − 1 nearest).
    - Each neighbourhood is split in two at random, one draw per subject, the same for every code. On one half, the
      shared directions: eigenvectors of the symmetrized T1–FLAIR cross-covariance of the deviations from that half's
      mean, ranked by eigenvalue. Independent view noise does not enter a cross-covariance. On the other half, the
      shared covariance along each direction (the mean product of the two views' deviations) and its correlation.
    - Both ways round, averaged over neighbourhoods. **s7** is the seventh direction's held-out shared covariance (7
      content factors, at every width); r7 its correlation.
    - Neighbourhoods: **own**, the code's own (both views' codes side by side, each unit standardized, by
      `data.cluster_batches.nearest_neighbours`, training's rule); **fixed**, the seven true factors standardized,
      the same for every code. Fixed is a diagnostic: it uses labels.
    - Sizes: k = 32, the training batch (an anchor and 31 neighbours), and k = 64.
    - Null: the same with the T1–FLAIR pairs shuffled within each neighbourhood, 5 shuffles per code.
    - Calibration rows, described only: the bank's true own parts plus independent view noise (reliability .8 per
      unit), as a 7 + 1 code (the seven parts and brain's own part squared) and a 6 + 2 code (ventricle and burden
      replaced by their sum, plus temporal's own part squared).
  - **Pass rule, fixed before the results** (own neighbourhoods, the measure a training loss could use):
    1. Over the 35 8-unit codes, the AUC of s7 for holders against failures is at least .80 at k = 32.
    2. It is at least .80 at k = 64 as well.
    3. The signal is real shared variation: in at least 13 of the 14 holding 8-unit codes, s7 at k = 32 exceeds the
       largest of its 5 shuffled values.
  - **What each outcome means:**
    - All three pass: the seventh direction is measurable without labels at the batch's size. Building the loss is
      worth it, as a term added beside InfoNCE, after stage 71 reads. A pass does not show that training on it adds
      the missing factor: holders are defined by holding seven parts, so some separation is built in.
    - Own fails but fixed passes (the same rule on fixed neighbourhoods): the signal exists, but the code's own
      neighbourhoods cannot see it, so a loss on code batches would not either.
    - Both fail: the weakest local shared direction does not mark holding at these sizes. The proposed loss would
      reward something else, such as curves or shared artifacts. Do not build it.
  - **Described:** per cell, holder–failure pairs on the wrong side (as item 51's top-1 check); the 9- and 10-unit
    codes; s7's Spearman with the spectrum's last value and with top-1 over 1,000; the whole profile s1–s8 for one
    holder and one failure; the calibration rows.
  - **Expectation, written before the results:** a pass, but not by much at k = 32. My estimate is that a weak factor's
    local shared variance is about 2–3 times a curve's leftover, which noise at 16 subjects a half could blur. Most
    likely AUC .80–.90 with own neighbourhoods at both sizes, higher with fixed ones. The calibration separates
    clearly. The shuffle null sits near 0 for every code.
  - **Cost:** CPU only, minutes; it does not touch the GPU, which stage 71 is using.
  - **Checked on synthetic codes, before any trained code was scored (25 Nov):**
    - `tests/test_local_directions.py`, 7 tests. The synthetic codes are seven factors plus one curve against six plus
      two, at reliability .8 per unit and N 1,000.
    - At k = 32 the held-out profile flattens, with 16 subjects a half for 8 directions: s7 .21 against .15, only 1.4
      times. At k = 128 it is .34 against .17, and with reliability .96 at k = 32 it is .29 against .11 (own
      neighbourhoods).
    - The shuffle null and pure view noise read near 0.
    - Fixed neighbourhoods separate less than the code's own (1.7–1.9 times where own reach 2.0–2.7): they leave the
      curve's coordinates free inside each neighbourhood.
    - Added after these tests: k = 128, described only. The pass rule is unchanged (k = 32 and 64). The tests also
      make k = 32 look harder than the expectation above assumed, and a loss on 32-subject batches would meet the
      same noise.
- **Stage 71 (built 25 Nov after item 56; the user asked to try it; read 25 Nov, item 58: no better, holds 2/5):
  does a projector head make 8 units reliable?**
  - **Why.**
    - The cosine reads the code directly, so every standardized unit counts alike, and a curve costs one of the
      code's units. A head between the code and the loss can bend factors and weight units itself, so the InfoNCE no
      longer competes with the information for the code's units (item 56's synthesis).
    - With the head's output at the code's width (8 cosine units: 7 degrees of freedom for 7 latents), code and head
      are one matched-width encoder. Yao's guarantee is an output that is a bijection of the content. The output is
      computed from the code, so the guarantee needs the 8-unit code to carry all of the content.
    - This corrects the status note of 25 Nov ("asked 25 Nov, no anchors") that a projector leaves Yao's setup. Only a
      head wider than the matched width would.
  - **Arm** (`warm.py --projector 64`, new):
    - The from-scratch 8-unit arm of stages 64, 60 and 70, at seeds 42, 2, 3, 4, 5: code batches of 31 neighbours
      rebuilt every 500 steps, residual input, jitter, prediction weight 1, tails, 20,000 steps, fixed-position
      causal bank. The global InfoNCE reads the code through one head shared by both views.
    - The head takes each view's code in its own running population gauge (mean and second moment over training
      batches, momentum .01). A batch gauge would lose each subject's absolute place under neighbourhood batches,
      and the code's own curves are absolute.
    - Then a linear map that starts as the identity, plus a one-hidden-layer MLP of width 64 (GELU) whose output
      layer starts at zero, at the code's 8 units. The loss standardizes that output over the batch and takes cosines
      as it does the code's, so at the first step it equals the recipe's.
    - The head is shared, so the two views' codes must still agree for their outputs to agree. The prediction term
      still reads the code directly through its linear map. Evals and exports read the code, never the head.
    - Width 64 is one value, set before any run and not tuned. Every other flag matches the from-scratch codes of the
      same seeds (checked against stage 70's recorded arguments). Names `..._abl_codebatch_predict1_proj64_c8_20k
      [_sN]`. `PARALLEL=5`.
    - Each eval logs the head's linearity: the mean over its output units of the 5-fold ridge R² of the head's
      output from the code, on test subjects. It stays at 1 while the head stays linear.
  - **Scored** (`causal/stage71/`, `holds.txt` first) beside the from-scratch 8-unit and 10-unit codes of the same
    five seeds, whose report entries are reused from stage 70:
    - precision, code and style audits; both discoveries; both tabular benchmarks; `own_directions.py`; and
      `holds.py --compare c8_proj64:c8_e500`. Labels `c8_proj64_sN`, `c8_e500_sN` and `c10_e500_sN`.
    - Reports run one at a time.
  - **Baseline:** from scratch holds 3/5 (seeds 2, 3, 5) and matches 2/5 (3, 5). Its holders spend 1 direction on
    curves in each view and its failures 2. The 10-unit codes hold 4/5 and match 3/5; seed 4 fails even at 10 units.
  - **Reading rules, fixed before the results** (stages 60 and 70's, so the levers read alike):
    - **The head makes 8 units reliable** if the arm holds in at least 4 of 5 codes and in at least 2 more than from
      scratch (3 of 5), so 5 of 5. It is no better if they differ by at most 1, and worse if it holds in 2 or more
      fewer (1 of 5 or none). The same thresholds apply to the match count (from scratch 2 of 5, so at least 4 of 5).
      Five seeds cannot make any of these significant; the rule is a threshold read with the seeds paired.
    - **Paired by seed:** name the from-scratch failures it rescues (seeds 42, 4) and the holders it loses (2, 3, 5).
    - **Against 10 units** (4 of 5 hold, 3 of 5 match at these seeds): described only.
    - **Mechanism** (`own_directions.txt`; a curve as in stage 60: a direction with own R² under .5 that boosting
      from the seven factors predicts at .5 or more, counted in each view):
      - The head takes the curves if at least 4 of 5 codes have at most one curve in each view (from scratch: 3 of
        5).
      - It takes them without helping if that happens but the hold count stays within 1 of from scratch. Then the
        competition for units was not what decided, and the weakest direction's own worth to the loss is the problem
        (item 56's named way to be wrong).
      - It fails if 2 or more codes still spend two or more directions on curves in a view.
      - Described: what each code's eighth direction holds (a curve, the lesion's shift, a weak own part, or style),
        and the head's final linearity (1 means it never bent).
    - **Costs:**
      - Cross-view top-1 over all 1,000 subjects, paired by seed, is a cost if the mean drop exceeds .05.
      - Style is a cost if any code reads a style variable at linear R² above .10 in either view. The from-scratch
        codes read at most .05. This is the arm's main risk: a head can ignore a unit, and nothing then pays the code
        to fill it with content.
      - Volume error, ATE bias and the decoded graph are described against from scratch.
  - **Expectation, written before the results:**
    - Curves leave the code: most codes at one curve per view or none, and the head bends (linearity well under 1).
    - Holds rise, most likely to 4 of 5, which the rule reads as no better. 5 of 5 needs seed 4, whose 10-unit code
      also fails, so its failure may not be about units alone.
    - Top-1 holds or rises, since holders read higher top-1 at 8 units. Style stays under .10, because a shared head
      still needs the views' codes to agree.
    - How it can fail: the head's output faces the same tie inside the loss, so the code holds the weakest direction
      only if the prediction term pays for it. Holds then stay near 3 of 5 with fewer curves ("without helping").
  - **Cost:** 100,000 steps, five at a time: about 3.7 hours at stage 70's pace (its training took 3 h 42 min). Then
    about 1.5 hours of reports, one at a time. Stage 68, the other session's region bank, waits for the GPU again.
  - **Checked locally (25 Nov):**
    - `tests/test_warm_projector.py`, 7 tests. Unsupported options are refused and nothing is written. The head is
      announced, and without it nothing is said. At the start the loss sees the recipe's code: the batch-standardized
      output equals the batch-standardized code. Every part of the head gets a gradient, the zero output layer
      included. The gauge tracks the population, and eval leaves it unchanged. The linearity measure reads 1 for a
      linear head and less for a bent one.
    - The other warm.py suites still pass (3 temperature, 5 bridge, 7 harmonic, 14 prediction, 3 prediction target).
    - A 100-step run on the fixed bank trains end to end. Its InfoNCE is 12.17 / 11.27 at steps 50 / 100, against
      the original run's 12.23 / 11.44, and its linearity 1.000 at step 100. `code_audit.load_run` loads its
      checkpoint; the head is saved under its own key, which the audits ignore.
    - `bash -n` passes. The dry run lists stage 70's flags with `--projector 64` in place of `--tau 1` for the five
      arms, then the reports.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 71` under `monai_env` (log `run_pc_71.log`). Started 25
    Nov 10:53, all five arms at once. Each logs "projector head 8 -> 64 -> 8". At step 50 the trained term is within
    .015 of the recipe's InfoNCE on the code at every seed (12.16–13.27), as the identity start requires.
  - **Interim, step 2,000** (25 Nov 11:17; not the verdict): the trained term is .11–.30 below the recipe's InfoNCE
    on the code at every seed, but the head's linearity is still 1.000 in both views. So far it reweights or mixes
    units through its linear part and does not bend. If it stays linear, the arm tests a learned reweighting, not
    bending after the code.
  - **Interim, step 12,000** (25 Nov 13:09; the PC's five; not the verdict, which reads own parts at step 20,000):
    - The head has barely bent: linearity .995–.999.
    - Seeds 2, 3 and 5, which hold from scratch, track their twins: sulcal .61 / .71 / .57 against .61 / .75 / .63,
      ventricle .86 against .86–.89.
    - Seeds 42 and 4, which fail from scratch, lag on sulcal (.42 and .33 against .74 and .69) and temporal (.74
      against .89 both). Ventricle runs slightly ahead (.83 and .84 against .79 and .82). So far that looks like
      trading which direction goes rather than rescue. Sulcal was still rising in the twins to step 20,000.
  - **Interim, step 18,000** (25 Nov 14:12; factor readouts, mean of the views; not the verdict):
    - PC: linearity .983–.995, bending a little since step 12,000. Seeds 3 and 5 track their twins (sulcal .78 / .73
      against .79 / .76). Seed 2 trails on sulcal (.65 against .75). Seed 42: sulcal .68 against .77, ventricle .85
      against .82. Seed 4: sulcal .39 against .75, temporal .77 against .90, lr .64 against .57.
    - A100 (latest eval): the same seed left its PC run by step 8,000 (seed 4's sulcal .46 against .19, seed 2's .56
      against .39), as a new draw would. Seed 3 finished with ventricle .84, against .92 for its PC run and its twin.
      Seed 42 stalls on sulcal (.36 at step 14,000). Seed 2 reads sulcal .82 at step 18,000, above both. Seed 4 at
      step 12,000 reads sulcal .71 and lr .79, ahead of its PC run (.33, .70).
    - Sulcal is the factor these codes lose most often, where the from-scratch failures lost ventricle–burden (42)
      and lr (4). The jobs ran at 0.32–0.50 s/step, slower than the timing job's 0.30, likely on shared nodes.
  - **A100 replicate** (25 Nov, the user's request; fixed before any of its results):
    - The same five seeds and flags as five separate Run:ai jobs, one A100 each, alongside the PC (`runai_warm.sh`).
      Names `..._abl_codebatch_predict1_proj64_c8_20k_a100[_sN]`, labels `c8_aproj64_sN`. (Fixed first as
      `c8_proj64a_sN`, renamed before any result because `holds.py` only parses an arm of letters then digits.)
    - The same seed reproduces on the same GPU, but a different GPU changes the trajectory the way a new seed does
      (timing job). So these are five more draws of the arm, not copies of the PC's.
    - How they are read: the stage's verdict stays on the PC's five, paired with the PC's from-scratch codes. The
      replicate's hold and match counts are reported beside them, and the ten codes pooled, described only. Its
      curve counts and costs are read as for the PC's arm, also described.
    - Its runs are copied back for the same reports once both sets finish: `run_pc.sh 71a` (no training; needs the
      runs and logs in `runs/` and `logs/`), into `causal/stage71a/`, after stage 71's reports and not beside them.
      It reuses stage 71's audits for the fifteen PC codes and bundles, so only the five A100 codes are audited anew.
      `holds.py` compares each set with from scratch and the two sets with each other; the ten pooled are the sum of
      the two per-cell counts. `bash -n` and the dry run pass.
- **Stage 70 (built 25 Nov; rules and the factor of 10 agreed by the user; started 25 Nov 05:48; read 25 Nov, item
  56: worse, holds 0/5): does a higher
  InfoNCE temperature make 8 units reliable?** On 25 Nov the user declined the anchors as a seed rule, asked for
  another way to make seeds hold, and chose this lever.
  - **Why.**
    - In the five from-scratch 8-unit seeds, the codes that hold spend one direction on curves and those that fail
      spend two (stage 60's baseline). Item 46 traced the curves to the cosine InfoNCE's push to spread subjects
      over the sphere: bending the strongest factor spreads them most. Stage 60 charged the curves after they formed
      and failed (item 54). The temperature sets the push itself.
    - Expanded in 1/τ, InfoNCE is alignment, a centring term, and an isotropy term on the code's second moment
      (weight 1/τ²). Full uniformity, the part that pays for bends, enters only at higher orders (1/τ³ and up). At
      τ 0.1 those orders weigh heavily. At τ 1 mostly the isotropy term is left, which a straight code meets and
      which values the ventricle–burden difference like any other direction.
    - Section 11 points the same way: hard-negative weighting, which sharpens the loss, starved lr and temporal.
    - The theory's optimum (aligned and uniform) does not depend on τ, so the arm stays inside Yao's setup.
  - **Arm** (`warm.py --tau 1`, new):
    - The from-scratch 8-unit arm of stages 56, 64 and 60 (code batches of 31 neighbours rebuilt every 500 steps,
      residual input, jitter, prediction weight 1, tails, 20,000 steps, fixed-position causal bank), with the global
      InfoNCE at τ 1 in place of 0.1, at seeds 42, 2, 3, 4 and 5.
    - The term is multiplied by τ over the recipe's (10), SupCon's convention, so its gradient keeps the recipe's
      scale against the prediction term. The arm changes the loss's shape, not its weight. Without the factor it
      would also make the prediction term about ten times heavier.
    - τ 1 is the general trainer's default (`training/encoder_args.py`), not a tuned value.
    - Every other flag matches the from-scratch codes of the same seeds (checked against their `report.json`). Names
      `..._abl_codebatch_predict1_tau1_c8_20k[_sN]`. `PARALLEL=5`.
  - **Scored** (`causal/stage70/`, `holds.txt` first) beside the from-scratch 8-unit and 10-unit codes of the same
    five seeds, whose report entries are reused from stage 60:
    - precision, code and style audits; both discoveries; both tabular benchmarks; `own_directions.py`; and
      `holds.py --compare c8_tau1:c8_e500`. Labels `c8_tau1_sN`, `c8_e500_sN` and `c10_e500_sN`.
    - The own variation audit is left out: no rule reads it. Reports run one at a time.
  - **Baseline** (stage 60's, recorded before its results): from-scratch 8-unit codes spend 1 direction on curves in
    each view where they hold (seeds 2, 3, 5) and 2 where they fail (4, 42). The 10-unit codes spend 3 or 4.
  - **Reading rules, fixed before the results.** Holds and matches as in stage 64 (reference stage 50, seed 42). The
    rules are stage 60's, so the two levers read alike.
    - **The temperature makes 8 units reliable** if the arm holds in at least 4 of 5 codes and in at least 2 more
      than from scratch (3 of 5), so 5 of 5. It is no better if they differ by at most 1, and worse if it holds in 2
      or more fewer (1 of 5 or none). The same thresholds apply to the match count (from scratch 2 of 5, so at least
      4 of 5). Five seeds cannot make any of these significant; the rule is a threshold read with the seeds paired.
    - **Paired by seed:** name the from-scratch failures it rescues (seeds 42, 4) and the holders it loses (2, 3, 5).
    - **Against 10 units** (4 of 5 hold, 3 of 5 match at these seeds): described only.
    - **Mechanism** (`own_directions.txt`; a curve as in stage 60: a direction with own R² under .5 that boosting
      from the seven factors predicts at .5 or more, counted in each view):
      - The temperature cuts the curves if at least 4 of 5 codes have at most one curve in each view (from scratch:
        3 of 5).
      - It cuts them without helping if that happens but the hold count stays within 1 of from scratch: the curves
        were not what decided which own part went.
      - It fails if 2 or more codes still spend two or more directions on curves in a view.
    - **Costs:**
      - Cross-view top-1 over all 1,000 subjects, paired by seed, is a cost if the mean drop exceeds .05.
      - Style is a cost if any code reads a style variable at linear R² above .10 in either view. The from-scratch
        codes read at most .05.
      - Volume error, ATE bias and the decoded graph are described against from scratch.
  - **Expectation, written before the results:**
    - Fewer curves, most codes at one per view.
    - Seed 42, item 46's crowding case, is the likeliest rescue. Seed 4 is the hard case: its 10-unit code fails with
      four units on curves, so a shortage of units is not all of its failure.
    - Most likely 4 of 5 hold, which the rule reads as no better; 5 of 5 needs both rescues and no loss.
    - A softer loss tells near neighbours apart less sharply, so top-1 falls, probably by more than .05.
    - The risk on the other side: with less spreading pressure the weak own parts (burden, lr) get less room, and the
      arm holds no more often.
  - **Cost:** 100,000 steps, five at a time: about 4.5 hours at stage 65's pace. Then about 1.5–2 hours of reports,
    one at a time. Stage 68, the other session's region bank, also needs the GPU, so the order is the user's call.
  - **Checked locally (25 Nov):**
    - `tests/test_warm_tau.py`, 3 tests: unsupported options are refused and nothing is written; the term is scaled
      by τ over the recipe's (10 at τ 1); without the option nothing is said or scaled. The other warm.py suites still
      pass (7 harmonic, 22 bridge and prediction).
    - A 40-step run on the fixed bank trains end to end on the GPU and logs "global InfoNCE temperature 1.0 scaled
      by 10.0".
    - `bash -n` passes. The dry run lists stage 56's flags plus `--tau 1` for the five arms, then the reports.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 70` under `monai_env` (log `run_pc_70.log`). Started
    25 Nov 05:48, all five arms at once (GPU at 100%, 15 GB). Each logs "global InfoNCE temperature 1.0 scaled by
    10.0". At step 50 the global term (both directions, unscaled) is 6.66–6.80 against 6.93 at chance.
  - **Interim, step 14,000 of 20,000** (read 25 Nov 08:40 from the runs' `report.json` evals; the verdict reads the
    final codes): the softer loss is dropping weak factors, not straightening them.
    - Sulcal's linear readout (mean of the two views) is .21–.42, against .69–.77 for the same seeds at τ 0.1 at the
      same step. It has gained under .10 at every seed since step 8,000.
    - Ventricle is .66–.84 against .81–.89, flat since step 8,000 except at seed 3 (.76 to .84).
    - Brain sits .02–.04 under the reference and thickness up to .09 under. Temporal is at or above it except at
      seed 5 (.76). Lr is above it at seed 4 (.90 against .60) and below it at seed 5 (.65 against .90). Cross-view
      code r is .96–.995.
    - This is the risk the expectation named, but it lands on sulcal and ventricle, not burden and lr.
    - The spectrum's last value cannot exceed any single own part's R². So wherever sulcal's own part stays under
      .40, the code fails the hold rule whatever its curve count.
- **Stage 69 (proposed 25 Nov; rules for the user to check; not built, nothing trained): is a clean compression a
  label-free accept rule for 8 units?** On 25 Nov, after item 54, the user agreed to test stage 65's after-the-fact
  pattern on new seeds, with the rules written first.
  - **Why.** At 8 units the seed decides whether the seventh own part is held, about half the time. No training lever
    changed that: rebuild interval, prediction weight, compressed start, longer training, harmonic penalty (items
    50–54). In stage 65, the three redundancy compressions whose removed directions were predicted at held-out R²
    .986–.997 held and matched. The two at .848 and .650 did not: one came from a parent that fails, one cut lr. That
    was seen after the results, on 5 seeds. If it holds on new seeds, it certifies an 8-unit code without labels,
    which top-1 cannot do within a recipe (items 51–53).
  - **Arms**, stage 65's exactly, at new seeds 6–15 (none used on this bank before):
    - 10-unit parents, `..._abl_codebatch_predict1_c10_20k_sN`: stage 56's recipe (code batches of 31 neighbours
      rebuilt every 500 steps, residual input, jitter, prediction weight 1, tails, 20,000 steps, fixed-position
      causal bank).
    - Each parent compressed to 8 units by redundancy (`compress_code.py --units 8 --method redundancy`, fitted on
      its training codes without labels, as in stage 65), then 10,000 steps at 8 units,
      `..._abl_from10red_predict1_c8_10k_sN`. Every child is trained, accepted or not, so the rule can be judged.
  - **The accept rule, fixed before any result:** accept a child if both removed directions have held-out R² of at
    least .98 in its compression log (the value written in the status section on 24 Nov). Right after each
    compression the stage reads the log and labels the child `c8_acc_sN` or `c8_rej_sN`. It writes the list to its
    log before any report runs. Parents are labelled `c10_e500_sN`.
  - **Scored** (`causal/stage69/`, `holds.txt` first): precision and code audits, both discoveries, both tabular
    benchmarks, `own_directions.py` and `holds.py --compare c8_acc:c8_rej`. Holds and matches as in stage 64
    (reference stage 50, seed 42). The own variation and style audits are left out: no rule reads them, and dropping
    them saves about 2 hours. Reports run one at a time.
  - **Reading rules, fixed before the results:**
    - **The rule works** if accepted codes hold all seven own parts in at least 80% (for example 4 of 5 or 5 of 6),
      and in a share at least .4 above the rejected codes'.
    - **It fails** if accepted codes hold in at most 60%, the from-scratch 8-unit rate (3 of 5 at every 500, stages
      64–65). Between the two: unresolved.
    - **The cut does not matter** if accepted codes hold in at least 80% but rejected ones come within .4. Then read
      the redundancy arm's overall rate instead.
    - The same thresholds apply to the match count.
    - The rule is judged only with at least 4 children accepted and 2 rejected. Otherwise the counts are only
      described. If every child is accepted, the redundancy arm's hold rate is read against stage 65's 3 of 5.
    - **Why rejections happen** (described): for each rejected child, whether its parent holds, and which own part
      fell more than .10 below the parent's. For each accepted child that fails, whether its parent held.
    - **Yield** (described): codes accepted and holding per seed, beside the from-scratch 8-unit hold rate. The
      rule's value is the certificate (how often an accepted code holds), not the yield.
    - **Pooled with stage 65** (described only, marked as including the seeds that suggested the rule): accepted
      against rejected over all 15 seeds.
  - **Expectation, written before the results:**
    - The rule accepts 5–7 of 10.
    - A clean cut keeps what the parent holds (stage 65), and parents hold in about 4 of 5, so accepted codes hold
      in about 80%: the rule passes narrowly or lands between.
    - The likeliest accepted failure is a parent that fails yet cuts cleanly.
    - Rejected codes hold in at most half.
  - **Cost:** 300,000 steps, five arms at a time: about 13 hours at stage 65's pace (210,000 steps in 9.3 hours).
    Then about 3 hours of reports, one at a time. Five seeds would halve the cost, but would leave about 3 accepted
    and 2 rejected codes, too few to judge the rule.
  - **Option, added 25 Nov after item 55, declined by the user the same day (no anchors for choosing seeds): the
    measured-anchor score as a second rule on the same codes.** Kept for the record; not part of stage 69. After the fact, item 55 found that stage 67's score ranks the seeds of a cell
    by whether they hold (the top seed holds in 9 of 9 cells; holders come first in 42 of 44 pairs).
    - **Score:** stage 67's rule score for every child and parent: split 'adni', anchors adjusted for each other,
      reliability .9, mean of 10 draws. It uses `anchor_check.py`'s functions on codes the stage exports.
    - **Reading rules, fixed now:**
      - The score works if, among the 10 children, the holder scores higher in at least .8 of (holder, failing
        child) pairs. It fails at .6 or below (chance is .5). It is judged only with at least 2 holders and 2
        failures.
      - Described only: the top-scoring child of seeds 6–10 and of seeds 11–15 (two picks are too few to judge),
        and the parents' pairs.
    - **Expectation:** works. Its likeliest miss is a child that loses lr or sulcal, which the anchors do not
      measure, as in item 55's two misses.
  - **To build once the rules are agreed:** the stage block, with a dry run. It covers parents, compressions,
    labels from the logs, children with `--init-run`, reports and `holds.py`. Stage 68, the other session's region
    bank, also needs the GPU, so the order is the user's call.
- **Stage 65 (built 23 Nov; read 24 Nov, item 52): an 8-unit code started from a compressed 10-unit
  code.** The user asked for a way to stop 8-unit seeds failing. Stage 64's early own parts (23 Nov, before its
  verdicts) suggested that about half of from-scratch 8-unit seeds hold. Failing codes spend the eighth unit on a
  curve, and the seventh own part has no unit left. The recipe can reach the 10-unit solution at 8 units (seeds 2,
  3, 5), so this is an optimisation problem. Both scored 10-unit codes hold all seven, so start from one.
  - **Compression, without labels** (`compress_code.py RUN OUT.pt --units 8 --method redundancy|pca`):
    - The run's code on its 2,000 training subjects, both views stacked, is standardized per unit (as the loss does)
      and whitened, so no direction predicts another linearly.
    - `redundancy` removes, one at a time, the direction an MLP predicts best from everything orthogonal to it (R² on
      held-out subjects). That is a curve; an own part, independent of the rest, is unpredictable.
    - `pca` drops the lowest-variance directions: the naive control.
    - The kept directions, orthonormal in the whitened space, become the new head's last layer. Network and TailCode
      are otherwise the run's. `warm.py --init-run OUT.pt` (new) starts a run from it with a fresh optimizer and
      fresh prediction maps.
  - **Pre-flight, 23 Nov, no training** (compression fitted label-free on each 10-unit code's test codes; own parts
    then read with labels only as a check, ridge, T1 / FLAIR):
    - Seed 42: the two directions removed are predicted at held-out R² .999 and .996. Redundancy keeps every own part
      within .02 except lr (.91 / .93 -> .83 / .87). PCA loses thickness (.75 -> .35), burden (.60 -> .48 / .42),
      ventricle (.85 -> .75) and others.
    - Seed 2: removed at .999 and .986. Redundancy keeps every own part within .04. PCA loses ventricle (.78 -> .49 /
      .37), thickness (.76 -> .44 / .54) and others.
    - Smoke test (fitted on the seed-42 run's training codes, R² .999 / .997): `--init-run` loads it, the run's GAP
      readout is unchanged (.924 / .930), and the 8-unit code reads lesion volume at R² .80 / .86 at step 0 and .86 /
      .88 by step 100.
  - **Arms**, all with stage 56's recipe (code batches of 31 neighbours rebuilt every 500 steps, residual input,
    jitter, prediction weight 1, tails, fixed-position causal bank), seeds 42, 2, 3, 4 and 5:
    - `..._abl_codebatch_predict1_c10_20k_s{3,4,5}`: the three missing 10-unit parents, 20,000 steps.
    - `..._abl_from10red_predict1_c8_10k[_sN]`: each parent compressed by redundancy, then 10,000 steps at 8 units.
    - `..._abl_from10pca_predict1_c8_10k[_sN]`: the same from the PCA compression.
    - `..._abl_codebatch_predict1_c8_20k_plus10k[_sN]`: stage 64's from-scratch 8-unit codes (every 500) continued
      10,000 steps through `--init-run` with their own checkpoints. The compressed arms see 30,000 steps in all; this
      separates the extra training from the start.
    - Compressions go to `compressed/`. `PARALLEL=5`; the parents start first; each later parent's children wait for
      it (`await`).
  - **Scored** (`causal/stage65/`, `holds.txt` first) beside stage 64's from-scratch 8-unit codes (every 500, the same
    five seeds) and the 10-unit codes at seeds 42 and 2: the four reports, both discoveries, both tabular benchmarks,
    `own_directions.py` and `holds.py --compare c8_red:c8_e500 c8_red:c8_pca c8_red:c8_cont c8_cont:c8_e500`. Labels
    are `c8_{red,pca,cont,e500}_sN` and `c10_e500_sN`. `holds.py` now takes named arms and `--compare` with paired-by-
    seed counts; stage 64's output is unchanged apart from wording and a line for the 10-unit cell.
  - **Reading rules, fixed before the results** (holds and matches as in stage 64; reference stage 50, seed 42):
    - **Compression makes 8 units reliable** if the redundancy arm holds in at least 4 of 5 codes and in at least 2
      more than the from-scratch arm. It is no better if they differ by at most 1. Between: unresolved. The same
      thresholds apply to the match count.
    - **The selection matters** if redundancy holds in at least 2 more of 5 than PCA. If they are within 1, the warm
      start carries it, not the choice of directions.
    - **Extra training alone** repairs failing seeds if the continued arm holds in at least 2 more than from-scratch.
      Then read the redundancy arm against the continued arm, not against from-scratch.
    - **Parents:** a parent that does not hold is named, and the redundancy arm's rate is also given over the parents
      that hold. Secondary: each redundancy code's own parts against its parent's (from `own_directions.txt`), within
      .10 in both views or named.
  - **Expectation, written before the results:**
    - The parents hold (5 of 5, perhaps 4).
    - Redundancy holds in 4 or 5 of 5 and matches in at least 3: the fine-tune keeps what the start already holds.
    - PCA holds in 1 to 3: its start loses own parts, and fine-tuning recovers some, as from scratch.
    - The continued arm is no better than from-scratch (within 1): a seed that lost the direction stays lost.
  - **Checked locally:** `bash -n`; the dry run lists the three parents, ten compressions, fifteen 10,000-step arms
    with `--init-run`, the reports over 20 codes and `holds.py` with the four comparisons. The stage was added below
    stage 64's block while stage 64 ran: inserting text above a running block shifts bash's read offset (it happened
    for a minute on 23 Nov and was moved before stage 64's block ended).
  - **Compression logs, recorded 23 Nov 21:20, before any stage 65 result** (`compressed/*_redundancy8.log`, held-out
    R² of each removed direction, first / second): seed 42 .999 / .997, seed 2 .999 / .990, seed 3 .999 / .848,
    seed 4 .999 / .650, seed 5 .999 / .986. At seeds 3 and 4 the second removed direction is not a clean curve; the
    cut may take part of an own part. Read those two seeds' red arms with that in mind; the rules stay as written.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 65` under `monai_env`, chained to start when stage 64's
    process exits. About 7–8 hours of training (three 20,000-step parents and fifteen 10,000-step arms, five at a
    time), then about 3 hours of reports.
- **Stage 64 (built 22 Nov; read 23 Nov, item 51): how often each width holds all seven own parts.** Item 50 found that at 8
  units the seed decides, so one seed per cell names no lever. The user asked for the seed count.
  - **Arms** (stage 56's recipe: code batches of 31 neighbours, residual input, jitter, prediction weight 1, tails,
    20,000 steps, fixed-position causal bank): 8 and 9 units, neighbourhoods rebuilt every 500 and every 100 steps,
    seeds 3, 4 and 5. Twelve arms, four at a time (`PARALLEL=4`), one seed's four cells per batch, so a stage stopped
    early leaves whole seeds. Names `..._abl_codebatch{,100}_predict1_c{8,9}_20k_s{3,4,5}`.
  - **Scored beside** the earlier codes of the four cells (8 units every 500: seeds 42 and 2; every 100: 42 and 2;
    9 units every 500 and every 100: seed 42) and the 10-unit codes at seeds 42 and 2. So 5 codes per 8-unit cell, 4
    per 9-unit cell. The four reports, both discoveries, both tabular benchmarks, `own_directions.py`, and the new
    `holds.py` (`causal/stage64/holds.txt`, read it first): per-code verdicts, per-cell and pooled counts, and top-1
    against holding. Labels `c{units}_e{every}_s{seed}`. Checked on stage 63's reports relabelled: it reproduces
    item 50's readings.
  - **Reading rules, fixed before the results:**
    - **Holds** all seven own parts: last spectrum value and ventricle–burden worst direction each .40 or more, in both
      views (stage 63's 9-unit rule). **Matches** the 10-unit code (stage 50, seed 42): every own part at least its
      value less .10, volume error at most its value plus 1 voxel, |ATE bias| at most its value plus .03, decoded
      graph at least its value less .05, in both views. One-sided: beating the reference never fails (as item 50
      read it).
    - **Width.** 9 units hold more reliably than 8 if the pooled hold rate (8 codes) beats 8 units' (10 codes) by .30
      or more; the same if the rates differ by less than .15; between, unresolved. The same thresholds for the match
      rate. No significance claim at these counts; give Fisher's exact p beside.
    - **Rebuild interval.** Every 100 against every 500, pooled over both widths (9 codes each), the same thresholds.
    - **A cell is reliable** if it holds in at least 4 of its 5 (or 3 of 4) codes; a seed lottery if 1 to 3 of 5 (1
      to 2 of 4); a failure if none.
    - **Top-1 as a label-free flag** (cross-view top-1 over all 1,000, per width): it flags holding if at most one
      code sits on the wrong side of the holders/rest split; otherwise not. Separately, if the highest-top-1 code of
      each width holds, "train several seeds, keep the highest top-1" is a label-free way to get the matched-width
      code; read whether it also matches.
  - **Expectation, written before the results:** 8 units hold in 2 to 4 of 10 codes (a seed lottery in both cells);
    9 units in 4 to 6 of 8, so width reads as unresolved or more reliable; the interval makes no difference (within
    .15). Top-1 flags holding with at most one exception, and the highest-top-1 code holds in both widths.
  - **Checked locally:** `bash -n`; the dry run lists the twelve arms (seed, width and interval as above, recipe
    flags unchanged), the reports, 20 bundles, both discoveries, both tabular benchmarks, `own_directions.py` and
    `holds.py`.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 64` under `monai_env` (started 22 Nov 23:22; three
    batches of about 3.3 hours, then about 2 hours of reports).
- **Stage 63 (built 21 Nov; read 22 Nov, item 50): stage 62's fresher neighbourhoods, replicated and pushed.** The user asked for
  all three follow-ups to item 49, with matching the 10-unit code at the matched width (Yao's theory) as the goal.
  - **Arms** (stage 56's recipe: code batches of 31 neighbours, residual input, jitter, prediction weight 1, tails,
    20,000 steps, fixed-position causal bank; four arms share the GPU, `PARALLEL=4`):
    - `..._abl_codebatch100_predict1_c8_20k_s2`: 8 units rebuilt every 100 steps, seed 2 (stage 62's arm, new seed).
    - `..._abl_codebatch_predict1_c8_20k_s2`: its control, 8 units rebuilt every 500 steps, seed 2. Added because no
      seed-2 8-unit code exists; without it a seed-2 result cannot be told from seed variation.
    - `..._abl_codebatch25_predict1_c8_20k`: 8 units rebuilt every 25 steps, seed 42.
    - `..._abl_codebatch100_predict1_c9_20k`: 9 units rebuilt every 100 steps, seed 42.
  - **Reports** (`causal/stage63/`): the four reports, both discoveries, both tabular benchmarks and
    `own_directions.py`, beside stage 62's rebuild-100 code, stages 56 and 59's codes and the 10-unit codes at seeds
    42 and 2.
  - **Reading rules, fixed before the results** (`own_directions.txt` and the reports, both views):
    - **Replication (seed 2, against its own control).** The lever replicates if the rebuild-100 code's last spectrum
      value and ventricle–burden worst direction each beat the seed-2 control's by .25 or more (stage 62: +.44 / +.35
      and +.45 / +.36), and ventricle's and burden's own parts each by .15 or more. It fails if either gain is under
      .10. Between: partial, named. If the seed-2 control already holds the seventh direction (last value .40 or
      more), stage 62's gain was seed 42's luck, and the comparison is void.
    - **Rebuilding every 25 steps (against stage 62's every 100, seed 42).** More often helps if the last value rises
      .10 or more in both views; it saturates if it moves less than .10 either way; it hurts if it falls .10 or more.
      Read cross-view top-1 beside it: neighbourhoods rebuilt this often could turn the batches into near-duplicates.
    - **9 units rebuilt every 100 (against stage 59's 9-unit code and the 10-unit codes).** It holds all seven own
      parts if the last value and the pair's worst direction are each .40 or more in both views.
    - **The goal, applied to every arm: matches the 10-unit code** (stage 50, seed 42) if every own part is within .10
      of it in both views, the volume error within 1 voxel, the ATE bias within .03 and the decoded image graph within
      .05. Also name the arm closest to it.
  - **Expectation, written before the results:** the seed-2 arm replicates, perhaps with a smaller gain. Every 25 steps
    saturates: the gain in stage 62 came from removing staleness, and 100 steps already removes most of it. The 9-unit
    arm holds all seven own parts, since stage 59's 9-unit code had 3 curves and fresher neighbourhoods cut one in
    stage 62; it comes closest to matching the 10-unit code, with ventricle still the factor most likely to trail.
  - **Checked locally:** `bash -n`; the dry run lists the four arms with the recipe's flags and one change each, the
    reports, the bundles, both discoveries, both tabular benchmarks and `own_directions.py`.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 63` under `monai_env` (four training arms at once, then the
    reports; about 5–6 hours).
- **Stage 62 (built 21 Nov; read 21 Nov, item 49): the curve-cutting levers pushed further at 8 units.** The user asked to try
  them, after the matched width (8 or 9 units) failed to match 10 (items 46, 48), with Yao's theory as the reason to
  want it.
  - **Why** (checked 21 Nov, `own_directions.py` on existing fixed-bank 10-unit bundles, no training; cut-off .5). The
    curves come mostly from InfoNCE and the other terms cut them: InfoNCE alone (`abl_nopredict_c10`) spends 4 of 10
    directions on the own parts and 6 on curves; with the prediction term (`predict1_c10`) 5 and 5; with code batches
    as well (stage 50) 7 and 3. The uniform 8-unit code (`predict1_c8`) is 5 and 3 at this cut-off (item 46 counts 6
    and 2; one direction sits at .29 / .34). If those levers cut curves, pushing them may free the 8-unit code's
    missing direction. Stronger entropy pressure was set aside for the same reason: InfoNCE alone makes the most curves.
  - **Arms** (stage 56's: code batches, 8 units, residual input, jitter, tails, seed 42, 20,000 steps, fixed-position
    causal bank), one lever each:
    - `..._abl_codebatch_predict3_c8_20k`: prediction weight 3 in place of 1.
    - `..._abl_codebatch100_predict1_c8_20k`: code neighbourhoods rebuilt every 100 steps in place of 500.
    - A tighter neighbourhood is not available: `--code-neighbours` must be at least the batch less one (31, the
      recipe's), so each batch is already one anchor's whole neighbourhood. A smaller one needs a smaller batch, which
      also halves the negatives.
  - **Reports** (`causal/stage62/`): the four reports, both discoveries, both tabular benchmarks and
    `own_directions.py`, beside stage 56's 8-unit code and stage 50's 10-unit code.
  - **Reading rule, fixed before the results** (per arm, `own_directions.txt`, both views; stage 56's code: 6 own
    directions and 2 curves, spectrum's last value .01 / .01, ventricle–burden worst direction .05 / .06):
    - The lever closes the gap if the code holds all seven own parts as directions (the spectrum's last value .40 or
      more, and the pair's worst direction .40 or more) with at most 1 direction on curves.
    - It cuts curves without helping if the curves fall to 1 or none but the last value stays below .15.
    - It does nothing if 2 directions stay on curves and the last value stays below .15.
    - Anything else: name it. Costs, read beside: every other own part within .10 of stage 56's 8-unit code, the
      volume error, no style, cross-view top-1.
  - **Expectation, written before the results:** a small effect at best. Prediction weight 0 → 1 removed one curve at
    10 units, but at width d the prediction term keeps the d target directions best predicted, which can be harmonics
    of sharp factors (section 8's width rule), so weight 3 may not remove another. The neighbourhoods' spread settles by
    the first rebuild (anatomy .78 → .65–.67 in stage 59's log), so rebuilding five times as often should change
    little. If neither closes the gap, stage 60's penalty and the label-free compression of a 10-unit code remain.
  - **Checked locally:** `bash -n`; the dry run lists both arms with stage 56's flags and one change each, the
    reports, the bundles, both discoveries, both tabular benchmarks and `own_directions.py`.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 62` under `monai_env` (two training arms in parallel,
    then the reports; about 2–2.5 hours).
- **Stage 60 (built 20 Nov as one seed, five seeds since 24 Nov; read 24 Nov, item 54): a non-linear
  redundancy penalty at 8 units.** The user asked how to stop the curves of item 46, then for this arm. On 24 Nov,
  after items 51–53, the user agreed to run it over five seeds.
  - **Why.** The 8-unit code spends 2 of its 8 directions on harmonics, curved copies of factors it already holds,
    leaving 6 for the seven own parts (item 46). Item 27's levers all failed in simulation, and none charged harmonics
    directly; its "penalty on units the other units predict" is not recorded as non-linear. Charging only what a
    non-linear predictor gets beyond a linear one leaves linear dependence between factors alone.
  - **Penalty** (`warm.py --harmonic-weight W`, new). Each view's code is whitened by running statistics, so nothing
    is linearly predictable, and read in eight fixed orthonormal bases (the units and seven random rotations), so a
    bent copy cannot hide by rotating. In each basis a critic predicts every coordinate from the others through a
    linear map and through that map plus a one-hidden-layer MLP of 64, trained on the detached code by its own
    optimizer. The penalty is the MLP's gain over the linear map, floored at zero and averaged over coordinates, bases
    and views. It runs on a separate uniform batch, since a neighbourhood batch barely varies in brain size, so a step
    takes about twice as long.
  - **Arms (24 Nov; they replace the single seed-42 arm, since items 51–52 showed one 8-unit seed cannot be read):**
    the from-scratch 8-unit arm of stages 56 and 64 (code batches of 31 neighbours rebuilt every 500 steps, residual
    input, jitter, prediction weight 1, tails, 20,000 steps, fixed-position causal bank) plus `--harmonic-weight 1`
    (the prediction term's weight; the penalty is at most 1), at seeds 42, 2, 3, 4 and 5. Every other flag matches the
    from-scratch codes of the same seeds (checked against their `report.json`). Names
    `..._abl_codebatch_predict1_harm1_c8_20k[_sN]`. `PARALLEL=5`.
  - **Scored** (`causal/stage60/`, `holds.txt` first) beside the from-scratch 8-unit codes and the 10-unit codes of the
    same five seeds (their report entries reused from stage 65): the four reports, both discoveries, both tabular
    benchmarks, `own_directions.py` and `holds.py --compare c8_harm:c8_e500`. Labels `c8_harm_sN`, `c8_e500_sN` and
    `c10_e500_sN`.
  - **Baseline, recorded 24 Nov before any stage 60 result** (`causal/stage65/own_directions.json`): every from-scratch
    8-unit code spends its spare directions on curves (own R² under .5, predicted from the seven factors by boosting
    at .73–.99), in both views: 1 in the three codes that hold (seeds 2, 3, 5), 2 in the two that fail (4, 42). The
    10-unit codes spend 3 (seeds 42, 2) or 4 (seed 4, which fails with units to spare).
  - **Reading rules, fixed before the results** (holds and matches as in stage 64; reference stage 50, seed 42). They
    replace the single arm's rule of 20 Nov, whose curve outcomes carry over as the mechanism rule:
    - **The penalty makes 8 units reliable** if the harmonic arm holds in at least 4 of 5 codes and in at least 2 more
      than from scratch (3 of 5 at these seeds), so 5 of 5. It is no better if they differ by at most 1. The same
      thresholds apply to the match count (from scratch 2 of 5, so at least 4 of 5).
    - **Paired by seed:** name the from-scratch failures it rescues (seeds 4, 42) and the holders it loses (2, 3, 5).
    - **Against 10 units** (4 of 5 hold, 3 of 5 match at these seeds): described only; five seeds cannot rank them.
    - **Mechanism** (`own_directions.txt`; a curve is a direction with own R² under .5 that boosting from the seven
      factors predicts at .5 or more, counted in each view):
      - The penalty removes curves if at least 4 of 5 codes have none in either view; then name what the eighth
        direction holds.
      - It removes curves without helping if they go but the hold count stays within 1 of from scratch: the curves
        were not what crowded the seventh own part out.
      - It fails if at least 4 of 5 codes keep a curve: the loss keeps them and pays the penalty.
    - **Costs:** cross-view top-1 over all 1,000 subjects, paired by seed, is a cost if the mean drop exceeds .05.
      Style in the code (style audit) is read beside. The penalty also charges the SCM's own bends (each child is a
      leaky ReLU of its parents), so the code may move toward the own parts themselves; that is a change of basis,
      not a loss.
  - **Expectation, written 20 Nov before the results:** uncertain. In item 27's simulation every lever that changed
    what the loss values failed at matched width; this one charges the curves directly. If the loss needs them to
    tell subjects apart, the code may keep them and pay the penalty, or lose discrimination (cross-view top-1).
  - **Expectation for five seeds, written 24 Nov before the results:** the penalty removes most curves. Seed 42, the
    crowding case of item 46, is the likeliest rescue. Seed 4 is the hard case: its 10-unit code fails with four
    units on curves, so its failure is not a shortage of units, and the penalty rescues it only if charging the
    curves turns the loss toward the missing direction. Most likely 3 or 4 of 5 hold, which reads "no better"; 5 of
    5 needs both failures rescued and no holder lost. Top-1 falls by up to about .05.
  - **Checked locally:** `tests/test_warm_harmonic.py`, 7 tests. The critic starts as its linear map, no coordinate
    predicts itself, and whitening leaves no linear correlation. Trained on planted codes, the penalty reads .002 for
    independent units, .002 for linear dependence, .53 for a bent copy and .54 for a bent copy rotated across two
    units (.03 with the units alone, which is why the eight bases). Its gradient reaches the code and never the
    critic, and unsupported options are refused. A 12-step run on a tiny local bank trains end to end (the critic
    trains and is saved, and the reports load the run), and a run without the penalty still does. `bash -n`; the dry
    run lists stage 56's flags plus `--harmonic-weight 1`.
    On 24 Nov: the 7 tests pass here; a 200-step smoke run on the fixed bank trains end to end, at about 3 GB of GPU
    memory and 1.15 s a step alone, against 0.28 late in stage 56's from-scratch run: about 4 times slower, not twice.
    The five-seed dry run lists five arms whose flags differ from the from-scratch codes' only by the penalty.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 60` under `monai_env`: five 20,000-step arms at once,
    then about 1.5 hours of reports (the reference codes' entries are reused). Started 24 Nov 09:46; with all five
    running each takes 1.13–1.15 s a step (GPU at 99%, 15 GB), so about 6.5 hours of training.
    Reports in parallel (the user's request, 24 Nov 13:36): the stage's bash is paused (SIGSTOP) and
    `run_pc_60_reports.sh` waits for the arms, then runs the same 25 report commands (checked against the stage's dry
    run) with the four audits and the causal tables side by side, and ends the paused bash (log
    `run_pc_60_reports.log`): about 40 minutes of reports instead of 1.5 hours.
    That backfired. The arms finished at 15:53, but each audit uses every CPU (16 here, about 27 threads each), so
    side by side they thrashed: 30–70 minutes a code against 2–5 run alone in stages 64–65. Run one after another,
    the reports would have ended about 17:25. The causal tables and `own_directions` finished at 17:08. From 18:12 the
    audits ran one at a time (precision, code, style, own variation; the waiting ones paused with SIGSTOP by a scratch
    script). Alone, each took 1–15 minutes for its remaining codes. All finished at 18:44, when `holds.txt` was
    written (identical to an early `holds.py` run at 18:28) and the stage ended. On this machine, run the audits one
    after another.
- **Stage 61 (built 20 Nov; not run): counterfactual images from tabular data.** The user asked to find the causal
  relationships from tabular data to the image latents, change the tabular data and see how the decoding does.
  - **Setup** (`counterfactual_images.py`). A tabular cause of one image factor, by the tabular benchmark's rule (T =
    rho u + sqrt(1 - rho^2) e, u the factor's own noise, rho .7), is raised by 3 SD for every subject. Read as T -> u,
    that raises u by rho x 3 and keeps each subject's e. The true counterfactual latents are the bank's SCM run
    forward from the subject's own noise, so the target and its descendants move: for ventricle, ventricle +1.09 SD,
    burden +.95, temporal +.16 and sulcal -.07. The true counterfactual images are the generator's render of those
    latents with every other draw the subject's own, normalized and given a residual as the bank's were. The factual
    re-render must reproduce the bank's stored images, or the script stops.
  - **Learned** on the training subjects: PC on the cause beside the factors read from the code names its direct
    effects; the code's slope on the cause is its total effect (the cause is exogenous), so each test subject's
    counterfactual code is its code plus slope x 3; per view, a decoder from the code to both channels, and the same
    decoder from the seven true factors. Decoders are saved in the stage folder and reused.
  - **Arms**, on 200 test subjects, per view and channel, each scored as the share of the true image change it
    reproduces over the brain, with the per-subject cosine and the size ratio: the code moved by the estimated effect;
    the decoder fed the codes of the true factual and counterfactual images (the decoder alone); the true-factor
    decoder fed the true latents (the oracle). Also the code's own change against the true counterfactual's code, and
    each factor's change read from the counterfactual code against its true change.
  - **Targets:** ventricle (burden, temporal and sulcal follow), temporal atrophy (focal; sulcal follows), thickness
    (brain-wide; temporal and sulcal follow). Stage 50's 10-unit code-batch code.
  - **Reading rule, fixed before the results** (scan channel, both views):
    - The code supports counterfactual images for a target if its estimated arm reproduces at least half the change
      the oracle reproduces, and the factors read from the counterfactual code move the target and each descendant the
      true way by at least half the true size, and no other factor by more than .2 SD.
    - The effect estimate is the bottleneck if the true-codes arm beats the estimated arm by .2 or more; the decoder
      is, if the oracle beats the true-codes arm by .2 or more.
    - PC should name the target, not its descendants, as the cause's direct effect.
  - **Expectation, written before the results:** one shift of the code for everyone gets the average change but not
    each subject's own (a change in ventricle moves the lesion differently in each brain), so the estimated arm falls
    below the true-codes arm, most for ventricle.
  - **Checked locally:** `bash -n`; the dry run; the SCM's inverse reproduces the bundles' own noise exactly and runs
    forward to the stored latents; the factual re-render matched a 290-subject local bank to 0.0000.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 61` (no encoder training; four decoders, about 1.5 h).
- **Stage 59 (built 20 Nov; read 21 Nov, item 48): the code-batch recipe at 9 units.** The user asked for it as a new stage, with
  stage 58 already running.
  - **Why.** Counting every direction alike, the 8-unit code spends 6 of its 8 directions on the own parts and 2 on
    curved copies of factors it already holds; the 10-unit codes spend 7 of 10 and 3 (item 46). If the curves take a
    fixed 2, nine units leave 7 directions for the seven own parts.
  - **Arm:** stage 56's arm with `--content-units 9` (code batches, residual input, jitter, prediction weight 1,
    tails, seed 42, 20,000 steps, fixed-position causal bank). Run name:
    `ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_abl_codebatch_predict1_c9_20k`.
  - **Reports** (`causal/stage59/`): the four reports, both discoveries and both tabular benchmarks, and
    `own_directions.py`, which now also counts where each code's directions go. Beside the 8-unit code (stage 56) and
    the 10-unit codes at seeds 42 and 2 (stages 50, 52).
  - **Reading rule, fixed before the results** (`own_directions.txt`, both views):
    - The count sets the width if the 9-unit code holds all seven own parts as directions (the spectrum's last value
      .40 or more, and the ventricle–burden pair's worst direction .40 or more) while still spending 2 directions on
      curves (2 directions under .5).
    - The curves grow with room if the 9-unit code spends 3 directions on them and still misses one own direction (the
      spectrum's last value below .15). Then extra width goes to curves first, and no fixed margin over the matched
      width is safe.
    - Anything else: name it, with the own parts and the precision audit beside the 8- and 10-unit codes.
  - **Expectation, written before the results:** the curves went from 2 directions at 8 units to 3 at 10, so they may
    grow with room, and then 9 units still miss one own direction. If they stay at 2, all seven fit.
  - **Checked locally:** `bash -n`; the dry run lists the arm (stage 56's flags with 9 units), the reports, the
    bundles, both discoveries, both tabular benchmarks and `own_directions.py`. The report's new count reproduces item
    46 on stage 56's bundles (8 units 6 and 2, 10 units 7 and 3, both views).
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 59` after stage 58 has finished (one training arm, then
    the reports).
- **Stage 58 (built 20 Nov; read 21 Nov, item 47): the 8-unit code on an independent bank.** The user asked for a non-causal
  variant of the 8-unit model.
  - **Why.** At 8 units the code keeps one blend of ventricle's and burden's own parts and drops the other (item 46).
    Two causes fit. The two change one patch of image (the fixed lesion sits at the ventricle's edge), or the edge
    ventricle → burden makes them the most correlated raw factors (r .83), so a narrow code keeps their shared axis.
    Without the SCM the factors are independent, while the lesion's site and the image stay.
  - **Arm:** stage 56's (code batches, 8 units, residual input, jitter, prediction weight 1, tails, seed 42, 20,000
    steps) on `bank_nofield_fixedpos`, the fixed-position bank rendered without `--causal`. Run name:
    `ssl_uni_jitter_nopatch_restail_t104_nofield_fixedpos_abl_codebatch_predict1_c8_20k`.
  - **Reports** (`causal/stage58/`): the precision, code and style audits, and the new `own_directions.py`. Per code
    and view it gives R² along each direction of the seven own parts' space, the weakest direction's mix, and every
    pair's best and worst directions, beside stage 56's causal 8-unit code and stage 50's 10-unit code. On the
    independent bank the own parts are the factors; own variation needs the SCM and is not run there.
    `causal_bundle.py` now exports an independent bank's codes, with no graph.
  - **Reading rule, fixed before the results** (stage 56's code: the pair's worst direction .05 / .06, blend .24 /
    .27, T1 / FLAIR):
    - The edges made the blend if, on the independent bank, the pair's worst direction reaches .40 and its blend falls
      to .10 or less in both views.
    - The image made it if the worst direction stays below .15 and the blend at .20 or more in both views.
    - Anything else is mixed. Either way, name the spectrum's weakest direction: if the code still misses one (last
      value below .15), its mix says what 8 units drop when the pair is free.
    - Caveat: without the SCM each factor's whole spread is its own part, so burden's own signal is twice as wide
      (±41 voxels against ±20) and the other children's own parts grow too. A pair held apart there may owe it to the
      larger signals, so "held apart" is weaker evidence for the edges than "still blended" is for the image.
  - **Expectation, written before the results:** no firm one. A pair as correlated (brain–thickness, r .80) was not
    blended, which points to the image; the blend's equal weights fit either.
  - **Checked locally:** `bash -n`; the dry run lists the bank (no `--causal`), the residuals, the arm with stage 56's
    flags, the audits, the bundles and the report. `causal_bundle.save` reproduces stage 56's bundle exactly (arrays
    and both identity digests) on the causal path and writes no graph on the independent one. `own_directions.py`
    reproduces item 46's numbers on stage 56's bundles.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 58` (renders a bank, then one training arm).
- **Stage 56 (built 20 Nov; read 20 Nov, item 46): the best 10-unit recipe at 8 units.** The user asked for the
  code-batch recipe with 8 units instead of 10.
  - **Why.** Under the cosine, 8 units are the matched width for the fixed bank's 7 latents (section 8): the
    contrastive term sees the code's direction only, 7 free numbers, one per latent, with none spare. Code batches
    made the 10-unit code the best causal block so far (items 41, 44). Stage 47's 8-unit code, with uniform batches,
    held the weak factors less well than its 10-unit code (sulcal .33 / .36 against .50 / .54, thickness .51 / .50
    against .59 / .65; item 39). Do code batches carry to the matched width, and what did the two spare units buy?
  - **Arm:** stage 50's code-batch arm with `--content-units 8`: fixed bank, residual input, jitter, prediction
    weight 1, tails, and batches of an anchor and its 31 nearest neighbours in the model's own code, rebuilt every
    500 steps; seed 42, 20,000 steps. Run name:
    `ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_abl_codebatch_predict1_c8_20k`.
  - **Reports** (`causal/stage56/`): the four reports, the intrinsic dimension, discovery (decoded and label-free),
    the tabular benchmark (linear and rbf) and the concealed-factor benchmark with measured anchors, beside the
    10-unit code-batch arm at seeds 42 and 2 and stage 47's uniform 8-unit code (their report entries reused).
  - **Reading rule, fixed before the results** (the 10-unit code-batch seeds agree within .07, item 44):
    - Eight units keep what ten hold if every own part stays within .10 of the lower 10-unit seed (.15 for
      ventricle), the volume error within 2 voxels of the worse seed, and the ATE bias (linear and rbf) and the
      decoded image graph's F1 within .05. Otherwise, name what is lost.
    - Code batches help at 8 units if the arm beats stage 47's uniform 8-unit code by more than stage 50's
      thresholds on own parts (brain .24, ventricle .16, sulcal .14, the rest .10).
    - The code fills the 7-latent block if its TwoNN reads near the calibration row of its width and holds flat
      under decimation (in stage 54 the 10-unit code-batch arms fell by about 1 at N/8).
  - **Expectation, written before the results:** with no spare direction, the weakest own parts go first (items 36,
    39), so sulcal or burden falls below the 10-unit code while the strong factors hold, and the arm still beats the
    uniform 8-unit code.
  - **Checked locally:** `bash -n` passes; the dry run lists the arm (stage 52's flags with `--content-units 8` and
    seed 42), the four reports, the intrinsic dimension, the bundle exports, both discoveries, both tabular
    benchmarks and the factoriser.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 56` (one training arm, then the reports).
- **Stage 55 (built 19 Nov; not run; no training): individual effects and counterfactual outcomes.** The user asked
  for counterfactuals with the tabular data.
  - **Why.** The ATE (item 24) is an average. A counterfactual asks what one subject's outcome would have been under
    the treatment they did not get, so it needs the subject's own effect. The simulator stores both potential outcomes,
    so individual effects can be scored exactly. It is the completeness test in causal terms: a code that drops a
    factor's own part cannot personalize an effect that depends on it.
  - **Tasks** (`eval.causal.tabular_benchmark --counterfactual --counterfactual-modifiers score own`). Per task seed,
    the effect is 1 + a standardized modifier: the hidden image score (all seven factors), or one factor's own part
    (one task per factor). The tasks of a seed share treatment and the outcome under control (`simulate_tabular`'s new
    `modifier`), so they differ only in who benefits.
  - **Estimators**, cross-fitted, each subject's effect from models that never saw it. T-learner: the difference of
    the AIPW's two outcome models. DR-learner (Kennedy 2023): each subject's AIPW score regressed on the covariates,
    with the ridge penalty chosen by leave-one-out (with ridge 1 the noisy scores were overfitted: R² −28 for a noisy
    24-unit code at 160 subjects).
  - **Scores.** R² of the individual effects (1 − PEHE² / their variance; tabular-only near 0 or below); PEHE; policy
    regret (the effect lost by treating exactly those with a positive estimate); and the counterfactual outcome's
    error. Both potential outcomes share their noise here, so the counterfactual that keeps the subject's own noise
    (observed outcome minus or plus the estimated effect) has the PEHE as its error; the other arm's model alone
    leaves that noise in (`cf mean`). Rows: tabular only, the true factors, the true own parts, each code and its
    untrained floor, T1 and FLAIR.
  - **Codes:** stage 54's eight, the backbone GAP of the 24-unit recipe and of the fixed bank's code-batch arm, and
    stage 53's grid and quantile arms once trained. Linear and rbf learners.
  - **Reading rule, fixed before the results.** In a simulation at 1,000 subjects the true factors read about .92 by
    the DR-learner and a 24-unit linear code holding the factors with reading noise about .55. A code holds what
    individual effects need if its R² beats its floor by more than .1 in both views. A factor's own part is usable for
    personalization if its task's R² clears .1 above the floor. Compare each code's own-part R² with its own variation
    (`own_variation.py`): this panel reads the same content through a noisier, confounded estimator, so expect it
    lower.
  - **Checked locally.** The refactored `estimate_ate` and `simulate_tabular` give bit-identical outputs to before (a
    saved fixed-input check). `tests/test_tabular_benchmark.py`: the 8 existing tests and 3 new ones (a modifier
    changes only the effect; the true factors recover individual effects at R² > .9 by both learners and the tabular
    columns at < .1; the CLI panel on a toy cohort, including the abducted counterfactual's error equalling the PEHE
    and the errors for a missing `z_own` or zero heterogeneity). The panel ran end to end on the stage 54 smoke
    bundles (160 subjects, so its numbers are noise there). `bash -n` passes; the dry run lists the bundle exports
    (reused when newer than the run) and both benchmark calls.
  - **Run** after stage 53 finishes: `bash out/burden_feasibility_20261102/run_pc.sh 55` (or `54 55`). Reports:
    `causal/stage55/counterfactual.log` and `counterfactual_rbf.log`, with JSONs beside them.
- **Stage 54 (built 19 Nov; footprints read 19 Nov, item 45; no training): where each code reads in the image, and
  how many dimensions it fills.** Section 10's label-free options at the user's request: step 3b (spatial footprints)
  and section 8's optional intrinsic-dimension check.
  - **Why.** Two views identify the shared anatomy only as a block (section 8). Locality is structure that can pick
    one remix of it. If each factor is read from its own region, the directions that read single factors read from few
    places, and mixtures from the union of those places (the structural sparsity of Zheng et al. 2022, in a linear
    form). The intrinsic dimension tests the embedding claim without labels: an embedding of the block fills a manifold
    of about 10 dimensions on the drawn bank and 7 on the fixed bank, whatever the code's width.
  - **Footprints** (`footprints.py`). Per 16³ cell and input channel, the units × units Gram (summed products) of each
    code unit's input gradient, averaged over 300 training subjects. Each voxel's gradient is weighted by that
    subject's deviation from the mean image, so a direction w's footprint w'G_c w is, to first order, how much of its
    subject-to-subject variation comes from cell c.
    - The weighting is needed. With the plain gradient (`--weight sensitivity`), a code pooled over the whole brain
      gives every direction nearly the same brain-wide map: in the smoke test every pair of footprints had cosine .99,
      while the image maps were distinct.
    - Each direction is also read against the code's mean footprint (the average over whitened directions, the same
      for every rotation): its specific footprint (where it reads more than the code as a whole) and its KL divergence
      from the mean (0 for a direction that reads like the whole code).
  - **The rotation.** Whitened by the total Gram, every unit direction's footprint sums to 1. Summed footprint sparsity
    over an orthonormal set of directions then equals the diagonal energy of the rotated cell Grams. So the sparsest
    rotation is the cell Grams' joint diagonalisation (Jacobi angles, the identity and 5 random starts).
    - Footprints from disjoint voxels and a linear code make the Grams exactly jointly diagonal; the tests recover the
      factors then.
    - Unlike stage 51's CCA, the whitening is by gradient energy, not the code's covariance, so the components stay
      correlated across subjects and PC is not empty by construction.
    - Two versions. Pooled: the Grams averaged over subjects. Subject: each subject's own Grams, through their fourth
      moment, which also counts as local a direction that reads one place per subject at a place that moves (the
      drawn lesion).
  - **Scoring; labels only score.**
    - Names: one Hungarian match on absolute correlation with the seven factors and the lesion centre. MCC is the
      mean matched |r| of the seven factors, against the code's units, 30 random rotations in the same frame, and the
      untrained twin's rotation.
    - Readout footprints: each target's ridge readout, taken as a direction of the code.
    - Image maps, model-free: per cell, the squared correlation across training subjects between the scan's intensity
      and each factor's own part (so its parents' regions drop out), or each lesion coordinate.
    - Each footprint's correlation with its target's image map and cosine with its readout footprint (both on specific
      footprints), and its shares in the brain interior, on the boundary, outside, and through the residual channel.
    - Figures per run, model and view in `causal/stage54/footprints/`.
  - **Intrinsic dimension** (`intrinsic_dimension.py`). TwoNN at N, N/2, N/4 and N/8 subjects, and Levina–Bickel at
    10 and 20 neighbours, on each code, the code on the unit sphere (as the cosine sees it), the backbone GAP and the
    twin's. Calibrated per bank on the true latents (7, and 7 plus the lesion centre), on those 10 latents through
    random leaky-ReLU maps into each code width with reading noise 0, .1 and .3, and on Gaussian noise of each width.
  - **Discovery:** each code, its two footprint rotations and their floors through `latent_causal_discovery` with the
    mixing null, as in stage 51.
  - **Codes.** Drawn bank: the 24-unit recipe at seeds 42 and 2 (stages 33–34) and the 10-unit code (stage 35). Fixed
    bank: the 10-unit code at seeds 42 and 2 (stages 47 and 50), the code-batch arm at seeds 42 and 2 (stages 50 and
    52), and the 8-unit code (stage 47).
  - **Reading rules, fixed before the results:**
    - The rotation separates factors if, in both views, its MCC beats the code's units and the random rotations' mean
      by more than the rotations' SD, and beats the twin's rotation by that much too.
    - A component reads its factor where the image shows it if its correlation with the factor's image map is
      positive in both views and above the twin's matching component's. Read the figures with the numbers.
    - The rotation helps graph discovery if its F1 beats the raw code's and the mixing null's mean by more than the
      null's SD, in both views (stage 51's rule).
    - The code supports the embedding claim if its TwoNN holds roughly flat under decimation, reads near the
      calibration row of its width at a matching noise level, and sits clearly below Gaussian noise of that width and
      away from the twin. (Corrected after the results; the verdicts do not change. The reference is the calibration
      row, not the block's size: at 1,000 subjects the estimators read the drawn bank's 10 latents as about 8 and the
      fixed bank's 7 factors as about 5.5–6. Noise and directions of small spread make the estimate fall as N falls.)
  - **Checked locally.**
    - `tests/test_footprints.py`, 8 tests: the Jacobi angles recover a planted basis and never lower the objective;
      the Jacobian of a toy linear code; both rotations unmix factors read from disjoint cells; the per-subject count;
      a footprint every direction shares has KL 0 and a rotation-free mean.
    - `tests/test_intrinsic_dimension.py`, 8 tests: TwoNN and Levina–Bickel on subspaces of known dimension, a swiss
      roll, decimation, duplicates, the calibration embedding, and noise raising the small-scale estimate.
    - Both tools ran end to end on 2-step 24- and 10-unit codes of a 160-test-subject bank (16 Jacobian subjects;
      footprints about 4.5 min for both runs on the Mac), and a rerun reused both. TwoNN underreads at 160 subjects
      (the 7 true factors read 6.2), which is what the calibration rows are for.
    - Writing the bundles there needed `causal_bundle.own_noise`'s parent check relaxed: sulcal's own noise
      correlated .23 with its parent at 160 subjects, and the check is sized for 1,000. The stage uses the check
      unchanged. The bundles read in `latent_causal_discovery`.
    - `bash -n` passes; the dry run lists both reports and both discovery calls.
  - **Cost on the PC (estimate):** Jacobians on 300 subjects, both views, up to 24 backward passes per batch, for each
    run and its twin; the joint diagonalisation about 20 s per call at 24 units (8 calls per run); discovery on 24
    bundles and 24 floors per view with a 30-draw mixing null.
  - **Run** after stage 53 finishes (never pull while `run_pc.sh` runs): `bash out/burden_feasibility_20261102/run_pc.sh
    54`. Reports in `causal/stage54/`: `intrinsic_dimension.txt`, `footprints.txt` (figures in `footprints/`),
    `label_free_view1.txt` and `label_free_view2.txt`.
- **Stage 53 (built 19 Nov; running since 10:58): other prediction targets for the 24-unit recipe.** The user asked
  whether predicting the other view's map does better than predicting its GAP.
  - **Arms** (stage 33's recipe, `bank_causal_nofield`, seed 42, 20,000 steps): `--predict-target grid`, the other
    view's backbone map as its eight octant averages (a coarse map, 512 features); `--predict-target quantile`, each
    channel's quantiles .5, .75, .9, .97, .99 over the interior cells (position-free, 320 features). Against the
    GAP recipe at seeds 42 and 2.
  - **Expectation, written before the results:** the vector-prediction study (`training/ENCODER_VECTOR_PREDICTION.md`)
    found a spatial target pays for knowing where the lesion is, not its size, while a position-free distribution
    of the other view's map pays for burden. So the grid target should reward anatomy and position, and the quantile
    target is the one that might sharpen burden.
  - **Reading rule, fixed before the results** (the GAP recipe's seeds agree closely: own parts within .08 except
    ventricle .71 / .57; volume 12.7 / 13.1 voxels; ATE bias .10–.12; image graph .84–.86): a target does better if
    an own part rises by more than .10 (.15 for ventricle) or the volume error falls by more than 2 voxels without
    another factor falling as much, or the ATE bias falls by more than .03 or the image-graph F1 rises by more than
    .03. Raw linear readability of the factors is read too (item 41's lesson).
  - **Checked:** 3 tests (the GAP target unchanged; the grid target is the octant averages; the quantile target is
    unchanged when a lesion moves and the grid target is not); all 22 `warm.py` tests pass; a 2-step arm of each
    target trains and saves (initial target spread: grid about 7 directions, quantile about 4); dry run.
- **Stage 52 (ran 19 Nov; read, item 44: replicates): a second seed of the code-batch arm.** Stage 50's code-batch
  arm (10 units, fixed bank) at seed 2, then the four reports, discovery, the tabular benchmark (linear and rbf) and
  the concealed-factor benchmark on both code-batch seeds beside the reference at seeds 42 and 2 (`causal/stage52/`).
  Item 41's code-batch gain stands if seed 2 also clears the reference's seed spread.
- **Stage 51 (ran 18 Nov; read, item 43: empty by construction): label-free discovery on the codes' cross-view
  canonical directions.** The user's choice among three label-free options (the others: the heterogeneity separation,
  which needs an observed variable that changes the factors' variances, absent from the bank; and ADNI's
  heterogeneity check).
  - **Why it might work:** canonical directions are unique where their correlations differ, and stage 48's trace on
    the fixed-bank 10-unit code showed single directions matching single concealed factors.
  - **Why it might not:** the trained codes' canonical correlations are bunched near 1 (fixed 10 units .98, .97,
    .97, .97, .96, .95, .94, .92, .90, .81; drawn 10 units .99–.94), so the top directions are near-ties and may
    stay mixed. The untrained twins keep 3–6 pairs beyond chance.
  - **Set-up:** `rotated_bundle.py --all-pairs` (a pure rotation; the untrained twins keep only 3–6 pairs beyond
    chance, fewer than the 7 factors PC names, which stopped the first launch) on stage 48's bundles and floors
    (fixed 10 and 8 units, drawn 10 and 24), then `latent_causal_discovery` on raw and rotated codes in one table per
    view, with 30 random rotations of each block (`causal/stage51/`).
  - **The fixed 8-unit code reduced to 7 components** (the user's request): its 7 most-agreeing canonical directions
    (`--components 7`, dropping r .84), one per latent of the fixed bank, so every component is named and every edge
    scored. Read against the raw 8-unit code, its 8-pair rotation and random rotations of the 7.
  - **Reading rule, fixed before the results:** the rotation helps if a rotated code's F1 beats both its raw code's
    and the random-rotation mean by more than the rotations' SD, with the per-factor match |r| higher too.
  - **Checked:** `tests/test_rotated_bundle.py` (2 tests: factors shared at different strengths come out as
    separate components, each at its reading ceiling; the rotated bundle keeps its rows and identity); dry run.
- **Stage 50 (ran 18–19 Nov; read, item 41): which of the recipe's parts the 10-unit code needs on the
  fixed-position bank.** Stage 49 ablates the 24-unit recipe; the user
  wanted the 10-unit code's version. Stage 47's 10-unit improvement itself came from one change, the fixed position.
  - **Arms** (`bank_causal_nofield_fixedpos`, 10 units, 20,000 steps): stage 47's 10-unit arm with seed 2, then, at
    seed 42, no residual, no jitter, code batches, no tails (`gapnorm`) and no prediction.
  - **Reports** (`causal/stage50/`): the four reports beside stage 47's 8- and 10-unit codes; discovery and the
    tabular benchmark, now with the rbf learner as well (`causal_tables` runs both, item 40); the concealed-factor
    benchmark with measured anchors, default split.
  - **Reading rule, fixed before the results:** removing a part matters if an own part moves by more than the two
    reference seeds differ on that factor and by at least .10, the volume error by more than 2 voxels, or burden's or
    sulcal's share of the novel block by more than .10. Narrow codes split close ties by seed (item 30), so the seed
    spread decides.
  - **Checked:** `bash -n`; the dry run lists the six arms and every report. The arm configurations match stage 49's,
    smoke-tested there, at 10 units.
- **Stage 49 (ran 18–19 Nov; read, item 42): which of the recipe's parts carry it.** Leave-one-out arms from stage
  33's recipe (24 units, prediction weight 1, uniform batches, residual input, tail pooling, 2-voxel / 3° jitter,
  `bank_causal_nofield`, 20,000 steps, seed 42), one part removed each:
  - no residual (the scan alone), no jitter, code batches (31 of 50 neighbours, rebuilt every 500 steps) in place of
    uniform ones, and no tails (`--pool gapnorm`: the average through the same gauge and head; prediction stays on);
  - already trained: no prediction (stage 28), 10 units (stage 35), and the recipe's second seed (stage 34);
  - known, not rerun: the linear and MLP maps match at 24 units (item 38); the field and gain are bank changes
    (items 18, 20–22), and the lesion's position is stage 47's (item 39).
  - **Reports** (`causal/stage49/`): the four reports on all eight codes, then stage 48's discovery and tabular
    benchmark on them.
  - **Reading rule, fixed before the results** (from the recipe's two seeds: own parts within .04 except ventricle
    .71 / .57 and sulcal .52 / .60; volume 12.7 / 13.1 voxels): removing a part matters if an own part moves by more
    than .10 (.15 for ventricle) or the volume error by more than 2 voxels. One seed per arm.
  - **Checked:** a 2-step arm of each new configuration trains and saves on a small bank; `--pool gap` refuses
    `--content-units`, so the tails are removed with `gapnorm`, which keeps the gauge and head; dry run as expected.
- **Stage 48 (ran 18 Nov, no training; read, item 40): stage 47's codes as causal blocks.** Stage 41's discovery
  (label-free PC on the units against random mixtures; PC on factors decoded from the code) and tabular benchmark
  (effect estimation and discovery with tabular variables), then stage 44's concealed-factor benchmark with
  measured anchors (both splits), on 8 and 10 units on the fixed bank and 10, 11 and 24 units on the drawn bank. Both
  banks hold the same test subjects and factors, so the bundles pass the identity check and share one table
  (`causal/stage48/`). New helpers in `run_pc.sh`: `causal_tables` (discovery plus the tabular benchmark for
  labelled runs) and `needs_done`.
- **Stage 47 (ran 18 Nov; read, item 39: yes, it does): does the lesion's position crowd the weak factors
  out of a narrow code?** About half of a 10-unit code goes to the position, a near-unique fingerprint (item 29).
  Stage 43's code-neighbour batches, whose negatives already share it, moved the code from position to anatomy (T1,
  boosted: x / y / z .79 / .84 / .84 → .74 / .62 / .69; brain .82 → .94, ventricle .68 → .81, thickness .75 → .88;
  per direction, lesion / anatomy .54 / .34 → .48 / .40). But those batches change more than position. This removes
  only the position.
  - **Bank:** `bank_causal_nofield_fixedpos` (section 4).
  - **Arms** (prediction weight 1, uniform batches, 20,000 steps, seed 42): 10 units (stage 35's recipe) and 8 units
    (the matched width for the 7 latents under the cosine), as
    `ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_predict1_c{8,10}_20k`. No 24-unit arm (the user
    dropped it, 18 Nov): stage 33's 24-unit code on the drawn-position bank is the reference. The subjects and their
    anatomy are the same, so the own parts are on one scale, but a 24-unit code on the fixed bank might read the weak
    factors a little better still, so that reference is a slightly low bar.
  - **Reports** (`causal/stage47/`): the four reports on the two arms and, on the drawn-position bank, stages 35, 38
    and 33's 10-, 11- and 24-unit codes (reused from stage 38).
  - **Reading:**
    - The own parts to watch are ventricle, thickness and sulcal, and burden's (drawn-position bank, T1: 10 units
      .29, .20, .07, .33; 24 units .71, .63, .52, .48).
    - If the 10-unit code on the fixed bank comes close to the 24-unit code, the position was crowding them out. The
      matched-width finding is then partly the benchmark's: ADNI has no randomly placed sphere. Phrase it as "a
      near-unique shared feature crowds weak factors out of a narrow code"; real scans have their own such features
      (head size and shape, positioning, scanner).
    - The 8-unit arm asks whether matched width drops the weak own parts without such a feature. If it does, the
      claim is general.
    - Lesion x, y, z in the code audit are near-constant on this bank, so their R² means little.
  - **Checked on the PC:**
    - `tests/test_bank_position.py` (4 tests): the lesion lands at the voxel nearest the target that fits, found by an
      independent brute-force search; the drawn placement is ignored; the default leaves the renderer alone.
    - Small banks (64 / 16 / 48 subjects): the default path reproduces `bank_causal_nofield`'s first subjects bit
      for bit (images, latents, volumes, lesion cells). The fixed bank keeps every content latent and lesion volume
      (within 0.0002 voxels); every image difference lies within 8 voxels of a lesion. The centre's SD falls from
      2.0 / 1.5 / 1.5 to 0.3–0.7 / 0.1–0.5 / 0.2–0.3 (16³ cells).
    - On a 200 / 16 / 120 fixed bank: the residual, a 2-step 8-unit arm and all four audits run.
    - `bash -n` passes, and `DRY=1 bash out/burden_feasibility_20261102/run_pc.sh 47` prints the bank, the residual,
      the two arms and the four reports.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 47`. The bank takes about half an hour, then the two
    arms train together.
- **Stage 46b (ran 18 Nov; read, item 38: no better than matched width): the bridge as the theory states it.**
  Section 8 (the 10 / 24 discrepancy) gives the reasons.
  - **Arm:** stage 36's 24-unit MLP-map arm (uniform batches, prediction weight 1, `--predict-hidden 128`, seed 42,
    20,000 steps, `bank_causal_nofield`) with `--entropy-units 11`, in the fixed form (item 37): InfoNCE on the first
    11 units, alignment alone on the other 13:
    `ssl_uni_jitter_nopatch_restail_t104_causal_nofield_predict1_mlp128_c24_entropy11_20k`, and its first 11 units
    as `…_first11`.
  - **Reports** (`causal/stage46b/`): the four reports on the bridge's 24-unit code and its first 11 units, stage 36's
    24-unit MLP-map code (the same arm with the entropy on all 24 units), stage 33's 24-unit code, stage 38's 11-unit
    code and stage 36's 10-unit MLP-map code. Stage 46's bridge and its first 10 units join once stage 46 is done:
    rerun 46b then. Finished arms and scored runs are reused, so the rerun only scores what is new.
  - **Reading:**
    - The bridge passes if its 24-unit code holds the factors as stage 36's 24-unit MLP-map code does (the one
      difference is the entropy block) and close to stage 33's (own parts within about .10, volume within a voxel or
      two), with no style. The paper can then state the formal claim. Item 26 expects this: at 24 units the MLP map
      reads burden only 0–1.6 voxels worse than the linear one.
    - Its first 11 units against stage 38's 11-unit code: does a block the theory covers hold the weak own parts
      better with 24 units' support? Item 27's simulation expects not (the faint latent leaks into the extra units).
      That would not affect the claim.
  - **Checked on the PC:**
    - `bash -n` passes, and `DRY=1 bash out/burden_feasibility_20261102/run_pc.sh 46b` prints the arm, the view and
      the four reports.
    - A 2-step arm in a scratch directory trains and saves; its log states "maps linear + MLP 128".
      `bridge_view.py` writes its first 11 units as an 11-unit run.
    - With the fixed loss: `tests/test_warm_bridge.py` (5 tests, one minimising the loss over free codes) and
      `tests/test_warm_prediction.py` pass; a 400-step arm states "InfoNCE on the first 11 units, alignment alone on
      the other 13" and trains like stage 36's MLP arm (item 37).
    - `reuse_reports` now adds entries to an existing report file instead of skipping it (an entry already there
      wins), so 46b draws on stages 36, 38 and 46. Checked into a scratch directory.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 46b`. It can run beside stage 46: the GPU had 3 of 24 GB
    in use.
- **Stage 45 (built by Codex 18 Nov, not yet run): two more 10-unit arms on neighbour batches.** Codex's reading of
  stage 43 (not yet recorded in section 7): the neighbour arm raised thickness's own part from .20 / .19 to .48 / .49
  and its backbone reading from .50 / .50 to .68 / .67. Sulcal stays low (.16 / .16 against .52 / .60 at 24 units),
  ventricle rises only modestly, burden's volume precision worsens, and temporal and lr fall against the matching seed.
  - **Arms:** the neighbour arm again with seed 2; and neighbour batches for InfoNCE with a separate uniform batch for
    prediction (`--predict-batches uniform`, seed 42). Prediction weight 1, 20,000 steps, field-off bank. The second
    tests whether the anatomy gain survives without the burden-precision cost.
  - The plan (section 10) stops the 10-unit search; Codex argues stage 43 reopens it. The user decides.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 45`.
- **Stage 44 (run and read 18 Nov, section 7, item 36; no training): why stage 42's count lost the weak own parts.**
  `factoriser.py` on stage 42's two codes (24 and 10 units) and both anchor splits.
  - **Why:** item 35. The suspected fault is the anchors' readouts. The code holds the anchors more fully than a ridge
    readout captures (R² .79–.95), and the shared anatomy the readouts miss may carry the weak own parts away.
  - **Arms:**
    - `--anchor-source measured`: step 2 subtracts each view's code on the anchors' measured values for every
      subject, and PC uses them as the anchor nodes. The anchors are then observed for every subject, not labels on
      training subjects only; ADNI has FreeSurfer measures for every scan.
    - `--anchor-source readouts`: stage 42 again, with the same numbers, now traced.
  - **The trace** (both arms, labels for scoring only): each direction the count examined (counted, dropped, or the
    one it stopped at) with its |r| with each concealed target, and each target's R² from the dropped directions,
    which is what the count subtracted of it as redundant.
  - **Reading:**
    - If the measured arm counts near 6 with the own parts near the full code, the readouts were the fault.
    - If burden's R² sits in the dropped directions in both arms, the fault is the known limit: a canonical pair
      that mixes a factor with a leftover. The candidate fix is to subtract only a dropped direction's redundant
      part (its fit on the anchors and counted factors), so the factor stays in the code. Not built.
    - If burden appears only in the direction the count stopped at (agreement below .5), the code does not share it
      across views well enough to be counted without labels.
  - **Checked locally:**
    - 17 tests pass, including measured anchors on a bank-like code, the trace, and both arms end to end.
    - On 20 bank-like codes (10 seeds each with burden read at error .25 and .35), in both arms, the count is right
      in 39 of the 40 runs. Measured anchors are not uniformly better there. At error .35 they miscount one code (5)
      and lose burden on it (block .00, dropped-direction R² .62). The readouts lose part of burden on two seeds but
      count it. On these codes the readouts are near-perfect (R² .98–.99), so they cannot show the suspected fault;
      the real code can.
    - A code that also holds every factor through a saturating tanh overcounts (7–9) in both arms but keeps every
      target. It is not the real failure.
    - Stage 41's 2-step 10- and 24-unit runs, measured anchors, default split: the report and the trace print, in
      about 3 min. Unlike the readouts, measured anchors count 1 on both 2-step codes and both untrained twins
      (agreement .77–.84). Subtracting the measured values leaves each code's own error in reading the anchors, and
      that error is shared across views when it comes from other shared content. So in this arm the floor need not
      count 0: read the trained count against it.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 44` (causal-learn needed, as for stage 42). Reports:
    `causal/stage44/measured_default.txt`, `measured_burden.txt`, `readouts_default.txt` and `readouts_burden.txt`.
- **Stage 43 (built 17 Nov, not run): what a 10-unit code keeps beyond the lesion's position.**
  - **Why:** at 10 units the code keeps brain, temporal and lr but drops the own parts of ventricle, thickness and
    sulcal as well as burden's (item 30, T1: .29, .20, .07 and .33, against .71, .63, .52 and .48 at 24 units). About
    half its directions go to the lesion's position (item 29). Both arms change what competes with those factors for
    the ten units; neither is aimed at the lesion.
  - **Arms**, both 10 units on `bank_causal_nofield`, 20,000 steps, seed 42, stage 35's recipe otherwise:
    - `ssl_uni_jitter_nopatch_restail_t104_causal_nofield_predict3_c10_20k`: prediction weight 3. At 24 units weight 3
      traded lesion position (mean R² .89 → .76 / .77) for the own parts of ventricle (.71 / .68 → .76 / .74) and
      sulcal (.52 / .60 → .65 / .68) (item 22). At 10 units position is what crowds those factors out.
    - `ssl_code_jitter_nopatch_restail_t104_causal_nofield_n31_predict1_c10_20k`: code-neighbour batches with
      prediction weight 1. Every 500 steps each batch is rebuilt as a subject and its 31 nearest neighbours in the
      model's own 10-unit code, both views, no labels (`--code-batches 500 --code-neighbours 31`). Every prediction arm
      so far drew uniform batches. Neighbours in the code already match on what it holds (lesion position and the
      strong factors), so InfoNCE has to tell them apart by what it lacks. Before the prediction term, code batches
      kept thickness's and temporal's own parts that uniform batches lost (section 10, item 4), and conditional
      negatives taught stage 23's remainder units the anatomy the main units lacked. Not covered by the theory
      (section 8).
  - **References** (report entries reused from stage 38): the 10-unit weight-1 codes, seeds 42 (stage 35) and 2
    (stage 38), and the 24-unit weight-1 code (stage 33).
  - **Reports** in `causal/stage43/`: `own_variation.txt` (with boosting; the main readout), `code_audit.txt` (how the
    directions split between the lesion and the anatomy), `precision_audit.txt` and `style_audit.txt`.
  - **Success**, in both views: an arm lifts at least two of the own parts of ventricle, thickness and sulcal by .05 or
    more over the better 10-unit seed (the seeds agree within .02 on these: ventricle .28–.30, thickness .19–.20,
    sulcal .06–.07), keeps burden's own part at .27 or above (the lower seed), and keeps brain, temporal and lr at or
    above the lower seed (.64, .71 and .65).
  - **Prediction:** weight 3 lifts ventricle and sulcal and gives up lesion position, perhaps with a weaker FLAIR
    backbone (12.2 voxels at 24 units). Code batches lift thickness and ventricle and may cost burden's precision, as
    they did in the InfoNCE-only codes (section 10, item 5). If both help, the next arm combines them.
  - **Checked locally:** two-step runs of both arms on the Mac copy of `bank_causal_nofield` (1,000 test subjects)
    train, evaluate and save; the code-batch arm rebuilds its neighbourhoods on the 10-unit code. `bash -n` passes, and
    `DRY=1 bash out/burden_feasibility_20261102/run_pc.sh 43` prints both arms and the four reports.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 43`, in section 3's `nohup` form (two 20k-step arms). It
    needs the runs of stages 33, 35 and 38 finished.
- **Stage 42 (run and read 17 Nov, section 7, item 35; no training): the concealed-factor benchmark on the frozen 24-
  and 10-unit codes.** `factoriser.py` runs the factoriser of `paper/factor-discovery-framework.tex` on
  our synthetic bank, in the order of its "Controlled discovery of omitted factors" section.
  - **Why:** stage 41 (item 34). The causal variables have to be made from the code by a factoriser with anchors; the
    code's own units are mixtures at every width.
  - **Input:** stage 33's 24-unit and stage 35's 10-unit prediction codes on `bank_causal_nofield` (field off, weight
    1), both views, on the 2,000 / 400 / 1,000 split. Each run's untrained twin is its floor.
  - **Anchors** (observed variables, labels on training subjects only): brain, ventricle, thickness and temporal, the
    factors FreeSurfer measures (brain volume, ventricular volume, cortical thickness, hippocampal volume).
    - **Concealed:** burden, lr and sulcal, and the lesion centre (shared content outside the causal graph).
    - **Second split** (`burden.txt`): burden as an anchor (ADNI has WMH volumes), thickness concealed.
    - `--anchor-noise F` adds noise to the training labels (F times each anchor's SD), to mimic FreeSurfer's error.
      Off in the stage.
  - **Step 1, anchored slots:** a ridge readout of each anchor from each view's code.
  - **Step 2, the novel block, a label-free count.** Each view's code minus its least-squares fit on the slots and
    their squares and products. The two views' residuals are compared by CCA (canonical correlation analysis: the
    pairs of directions, one per view, that correlate most). Greedy deflation: take the top pair while its agreement
    (the correlation of the two views' projections on validation subjects) beats .5 and a null that shuffles the
    pairing. Count it unless it is redundant: a degree-3 polynomial of the other view's slots and counted factors
    predicts it at R² ≥ .2. Either way, subtract it with its squares and products, and repeat. Expected: 6 (burden's,
    lr's and sulcal's own parts and the lesion centre's x, y and z).
    - **The subtraction must be exact** (found building the stage). After a ridge fit, at any penalty, a remnant of
      each taken direction stays in the code, and nothing else is left along it. CCA ignores scale, so it finds the
      remnant again, as shared as the direction itself: a linear readout of the next residual recovers the taken
      direction at R² .96, at penalty 1e-8 as at 10. Dropping the remnant as redundant dragged real factors out with
      it: burden's own part on one synthetic code, a latent on a toy. Least squares leaves nothing.
    - **Squares are clipped.** With exact fits, a few held-out subjects beyond the training range were extrapolated
      quadratically, and the error compounded over steps: the test residual grew a thousandfold in four steps. Squares
      and products now use the inputs clipped to their training range; the linear terms stay exact.
    - A direction dropped as redundant is left out of later redundancy tests, so it cannot make a real factor look
      redundant.
  - **Step 3, coordinates inside the block:** the CCA order and FastICA (3a), each component named after one concealed
    target by one Hungarian match on |corr| (labels only score), against random rotations of the block. ICA should
    separate the uniform lesion-centre coordinates but not the Gaussian own parts (item 33). 3b (spatial footprints:
    each direction's input-gradient footprint per 16³ cell, rotated to make the footprints sparse; Zheng et al. 2022) is
    not built.
  - **Step 4, the graph:** PC on the slots and the components (Fisher-Z, 300 rows × 20 resamplings). It is scored
    against what PC can find among the anchors with the concealed factors latent, not against the SCM. The components
    are residuals of the anchors, so they connect to nothing (section 8). Default anchors: brain–thickness,
    brain–ventricle, temporal–thickness, and temporal–ventricle, induced through the concealed burden. Ceilings: PC on
    the true anchors, and on all seven true factors against the SCM.
  - **Checked locally:**
    - `tests/test_factoriser.py`, 15 tests in about 9 s. They cover the count on toy codes (squares not counted, an
      anchor's cube flagged, a weak factor counted, nothing counted without shared content), the exact subtraction, the
      clipped squares, the graph truth, and the whole benchmark end to end, report included, on a code built like the
      bank's with a known answer.
    - The counting rule on 60 synthetic codes. 30 are built like the bank's (its SCM, the lesion centre with the
      squares and product of its x and y, six novel latents), at burden reading error .25 and .35, 15 seeds each. 30
      are toys.
      - Count right on all 30 bank-like codes and on 28 of 30 toys; 2 leftovers were counted, at agreement .52 and .73.
      - Burden's own part in the block: .59–.73 at error .25, against .67–.73 from the whole code. At error .35:
        .50–.60 against .56–.63, except on two seeds (.31 and .40), where burden shared a canonical pair with a
        dropped leftover. The report prints both readings for every concealed target, so a loss shows.
      - Not kept: the first rule (top pair, ridge subtraction) lost burden on 1 of 5 seeds. Testing the shared
        subspace's most redundant direction first overcounted bank-like codes (7 or 8) before the fixes. After them it
        keeps a weakly read burden better (.57–.64 at error .35, where this rule has its two losses), but loses
        sulcal's own part on 6 of 17 bank-like codes (.61–.71 against .91–.93) and overcounts 2 of 17 bank-like codes
        and 2 of 15 toys. A rule that separates a factor from a leftover of equal agreement is still open.
    - The 2-step 10- and 24-unit runs of stage 41's smoke test (a local copy of `bank_causal_nofield` with a small
      validation split), both anchor splits: loading, calibration, slots, the count, PC and the report all run, in
      4–6 min each on the Mac. A 2-step code holds no novel shared factor: 0 components, trained and floor.
    - `bash -n` passes, and `DRY=1 bash out/burden_feasibility_20261102/run_pc.sh 42` prints both factoriser commands.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 42`. It needs stages 33 and 35's runs finished and
    causal-learn (`python -m pip install causal-learn`).
  - **Reports:** `causal/stage42/default.txt` (default anchors) and `burden.txt` (burden an anchor, thickness
    concealed), with JSONs beside them. A rerun reuses a run already in the JSON at the same step and settings.
  - **Success:**
    - Anchors read at R² ≳ .9.
    - The count close to 6 on the trained code (the floor's count is printed beside it).
    - The block holds each concealed factor's own part about as well as the whole code does (the code's own
      variation: burden .48, sulcal .52–.60, lr .89).
    - PC recovers the anchors' graph and adds no more edges to the components than chance (printed beside them).
  - **What it cannot identify** (section 8): where a concealed factor sits in the graph. Its component is a residual
    of the anchors, so every edge between it and an anchor is absorbed, into it or out of it. A concealed mediator
    shows only as an induced anchor–anchor edge. Placing it needs 3b's directions, taken in the whole code, or labels.
  - **Decisions open:** step 2 counts with cross-view CCA on the frozen code. The user rejected CCA fitted after
    training as the burden readout (section 11 and memory); whether it may count factors inside the factoriser is
    theirs to say. Also open: anchor noise, and building 3b.
- **Stage 41 (run and read 17 Nov, section 7, item 34; no training): does the causal analysis need a code as wide
  as the shared block?**
  - **Why:** two full views identify the content block, not its coordinates. So even a 10-unit code's units are
    mixtures of the latents (item 6; item 30's code audit), and the narrow code drops the own parts PC needs (items
    24, 30). This stage measures what width does to the causal analysis.
  - **Codes:** the causal bank's 10-unit codes (stage 35, seed 42; stage 38, seed 2), the 11-unit code (stage 38) and
    the 24-unit code (stage 33); field off, prediction weight 1.
  - **Label-free discovery** (`eval.causal.latent_causal_discovery`): PC on each code's own units, named by one
    Hungarian match, scored against the true graph and against 30 random mixtures of the units. This is the test of
    whether matched width makes the units the causal variables.
  - **Decoded discovery** (`eval.protocol.compare_bundles --with-graph`): PC on the seven factors read from each code
    with labels, 20 resampling repeats.
  - **Tabular benchmark** (item 24) on the four codes and on the backbone GAP of the 10- and 24-unit runs: the
    confounding removed, and the MRI→tab and tab→MRI F1.
  - **Reports** in `causal/stage41/`: `discovery_label_free_view{1,2}.txt`, `discovery_decoded_view{1,2}.log` and
    `tabular.log`. The code bundles go to `causal/<run>.npz` and are reused when newer than the run's checkpoint.
  - **Prediction:** label-free F1 at the mixing null at every width; decoded discovery and the tabular benchmark
    better at 24 units, with thickness's and sulcal's edges missed at 10 and 11. If the 10-unit codes come out
    better, matched width is worth its precision cost for the causal analysis.
  - **Checked locally:** on two-step 10- and 24-unit runs on a Mac copy of `bank_causal_nofield` (1,000 test
    subjects) every step runs: code and GAP bundles with their untrained floors, label-free discovery with the mixing
    null, decoded discovery and the tabular benchmark on four bundles. Shell syntax and the dry run pass.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 41`. It needs causal-learn
    (`python -m pip install causal-learn`).
- **Stage 40 (run and read 17 Nov, section 7, item 32: the gate fails; no training): is the head's capacity what
  limits a 10-unit code?**
  - **Why:** every check so far kept TailCode's head at features → 100 → K and varied only the map to the target
    (items 26, 31). A 10-unit code has to turn the pooled features into about one direction per latent, a harder
    non-linear job than a 24-unit code, which can keep position's harmonics in separate units. One hidden layer of
    100 may not manage it.
  - **Check:** `bottleneck_check.py --heads 100 512 512x2` on the four runs of stages 36 and 39, GAP target, linear
    and MLP maps, 10 and 24 units, three seeds each. "512" is one hidden layer of 512 units, "512x2" two. TailCode's
    own head ("100") is reused from stage 36's JSON, so 96 bottlenecks are new.
  - **Reading:**
    - A larger head works if its 10-unit bottleneck reads volume near the 24-unit one with TailCode's head (field off
      13.6 / 13.6 and 13.6 / 13.8; gain off 9.3 / 9.4 and 10.6 / 11.1) while keeping lesion position and anatomy.
      Then the narrow code is limited by the head, and a training arm with that head at 10 or 11 units follows
      (`warm.py` would need a head-width option; not built).
    - If the 24-unit bottleneck gains as much, the head limits both widths and the gap between them stays.
    - If nothing moves, the objective's ranking limits the narrow code, and item 30's precision result stands as the
      finite-sample cost of matched width.
  - **Run:** a one-off, like stage 39 (`run_pc.sh` unchanged). From the repo root:
    ```bash
    D=out/burden_feasibility_20261102; P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield; mkdir -p $D/causal/stage40 && { [ -f $D/causal/stage40/bottleneck_check.json ] || cp $D/causal/stage36/bottleneck_check.json $D/causal/stage40/; } && PYTHONPATH=$D:out/regional_pilot_20261028/shim:. OMP_NUM_THREADS=4 nohup python $D/bottleneck_check.py $D/causal/stage40/bottleneck_check.json $D/runs/${P}_predict1_c24_20k $D/runs/${P}_predict1_c10_20k $D/runs/${P}_nogain_predict1_c24_20k $D/runs/${P}_nogain_predict1_c10_20k --heads 100 512 512x2 > $D/causal/stage40/bottleneck_check.txt 2>&1 &
    ```
    Drop `OMP_NUM_THREADS=4` once stage 38 has finished.
  - **Checked locally:**
    - Fourteen unit tests (`tests/test_bottleneck_check.py`). Head names parse ("100", "512x2") and bad ones are
      refused. TailCode's head is built exactly as stage 36 built it (the same weights from the same seed), so the
      reused rows stay comparable. A deeper head has the right layers. A new head fits only its own bottlenecks and
      keeps the rest. Keys and the table carry the head.
    - On a two-step run on a Mac copy of `bank_causal_nofield`, the 512 and 512x2 heads fit, save, reuse the saved
      rows of TailCode's head and print.
- **Stage 39 (run and read 17 Nov, section 7, item 31: the gate fails, the tails target is dropped; no training):
  what if the 10-unit code predicted the other view's tail features?**
  - **Why:** item 29. A 10-unit code spends about half its directions on the lesion's position. The GAP the
    prediction term targets is full of position structure. TailCode's tail features are means over the interior
    cells, so they carry the lesion's volume and focal anatomy and little of its position. With a flexible map the
    argument in section 8 still holds for any target that depends on the other view only through the content.
  - **Check:** `bottleneck_check.py --targets gap tails pooled` on the four runs of stage 36's check: stage 33's two
    24-unit weight-1 arms and stage 35's two 10-unit ones, field off and gain off. Fresh bottlenecks at 10 and 24
    units predict the other view's GAP (64 features), its tails (384) or its whole pooled vector (448, the GAP and
    the tails), through the linear or the MLP map, three seeds each, with no InfoNCE. Each is read like a code, with
    each anatomy factor and lesion coordinate by boosting too.
  - **Reuse:** the JSON starts as a copy of stage 36's, so its GAP rows and each run's own code are reused and only
    the tails and pooled rows are fitted (96 bottlenecks). The reused rows have no per-factor values. On the same
    features the new script's GAP rows equal the old script's exactly (checked locally).
  - **Reading:**
    - A target works if its 10-unit bottleneck reads volume near the 24-unit GAP bottleneck (field off 13.6 / 13.6
      on the 24-unit backbone, 13.6 / 13.8 on the 10-unit one; gain off 9.3 / 9.4 and 10.6 / 11.1) while keeping the
      lesion position (boosted ≳ .85) and each anatomy factor at least as well as the GAP target's 10-unit rows.
    - The tails alone may drop the global anatomy (brain size, lr) that the GAP carries; the pooled target says
      whether adding the GAP back keeps both.
    - If a target passes, the next step is a training arm with that target (`warm.py` would need a
      `--predict-target` option; not built).
  - **Run:** not in `run_pc.sh`, so it can start while stage 38 trains: the commit changes neither `run_pc.sh` nor
    any script stage 38 calls. From the repo root:
    ```bash
    D=out/burden_feasibility_20261102; P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield; mkdir -p $D/causal/stage39 && { [ -f $D/causal/stage39/bottleneck_check.json ] || cp $D/causal/stage36/bottleneck_check.json $D/causal/stage39/; } && PYTHONPATH=$D:out/regional_pilot_20261028/shim:. OMP_NUM_THREADS=4 nohup python $D/bottleneck_check.py $D/causal/stage39/bottleneck_check.json $D/runs/${P}_predict1_c24_20k $D/runs/${P}_predict1_c10_20k $D/runs/${P}_nogain_predict1_c24_20k $D/runs/${P}_nogain_predict1_c10_20k --targets gap tails pooled > $D/causal/stage39/bottleneck_check.txt 2>&1 &
    ```
    `OMP_NUM_THREADS=4` keeps the boosted readouts from crowding stage 38's arms. Expect several hours. The JSON is
    saved after every bottleneck, so the same command resumes after a crash; the table prints at the end.
  - **Checked locally:**
    - Ten unit tests (`tests/test_bottleneck_check.py`): the targets slice the pooled vector; the GAP keys keep stage
      36's names; a row at the same step keeps its bottlenecks, fits only the missing ones and saves after each; a row
      from another step is refitted; a run without tails is refused; stage 36's rows and the new ones both print.
    - On a two-step run on a Mac copy of `bank_causal_nofield`, the new script's GAP rows equal the old script's
      exactly (428 numbers), and the tails and pooled targets fit, save, resume from the saved rows and print.
    - No script stage 38 runs imports `bottleneck_check.py`.
- **Stage 38 (run 17 Nov; precision audit, own variation and code audit read in section 7, item 30; the style audit
  not yet).** The theory's own matched width, and where matched width holds.
  - **Why** (section 8, item 29):
    - warm.py's InfoNCE is cosine, so 10 units give it 9 degrees of freedom. The formal matched width for 10 latents
      is 11 cosine units.
    - On the causal bank, burden's own part is the latent the views determine least well (item 27). The independent
      bank tests a 10-unit code when burden is a whole factor.
  - **Arms**, stage 35's recipe (prediction weight 1, 16 components, 20k steps):
    - 11 units, field off, seed 42: `ssl_uni_jitter_nopatch_restail_t104_causal_nofield_predict1_c11_20k`;
    - 10 units, field off, seed 2: `..._causal_nofield_predict1_c10_20k_s2`, the 10-unit seed spread, unmeasured so
      far. It starts when a slot frees, so the stage takes about one arm longer;
    - 10 and 24 units on `bank_nofield` (the independent bank with the field off, `bank.py --t1-values 0.4 --field
      off`, rendered while the first arm trains): `ssl_uni_jitter_nopatch_restail_t104_nofield_predict1_{c10,c24}_20k`.
      Burden's own part there is its whole ±41-voxel spread, against ±20 on the causal bank.
  - **Controls:** stage 35's 10-unit arm (17.3 / 17.6) and stage 33's 24-unit arm (12.7 / 13.2), reused from stage
    37's reports.
  - **Reports** in `causal/stage38/`: the precision audit, the code audit and the style audit of all six, and own
    variation with `--boosted` of the four causal-bank runs (it needs the causal graph).
  - **Reading:**
    - Does 11 units beat 10 by more than the two 10-unit seeds differ?
    - On the independent bank, 10 against 24 units in voxels. Both near their backbone: matched width works when the
      factor is large, and the causal bank's failure is about how faint burden's own part is. 10 units near 17
      voxels again: the limit is the precision of the lesion reading itself. Compare voxels within a bank: the same
      error is own R² .85 there and .58 here.
    - Does boosting read more of each own part from the 10- and 11-unit codes?
  - **Checked locally:** shell syntax and the dry run. A small independent field-off bank (208 / 32 / 160) renders with
    no causal graph. Two-step runs of the 10- and 24-unit arms on it, and of the 11-unit arm on a Mac copy of
    `bank_causal_nofield`, finish. Own variation with `--boosted` reads the 11-unit run. The style audit reproduces the
    independent bank with no options and scores its 10-unit run.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 38`. The new bank needs about 7 GB.
- **Stage 37 (run 16–17 Nov; read in section 7, items 28–29: the sharper backbone does not reach the 10-unit code,
  and the 32 components remove temporal atrophy, thickness and sulcal).** Does a sharper residual let the 10-unit code
  keep burden?
  - **Why:** item 27. No change to the objective kept a faint latent in 10 units; sharper views did. With 32
    components removed instead of 16, the image filter reads burden's own part to 7.4 / 5.2 voxels instead of
    9.0 / 6.6.
  - **Banks:** `bank_causal_nofield_k32` and `bank_causal_nofield_nogain_k32`, built by
    `residual_bank.py --components 32 --source BANK`. It reuses the source's stored basis, so the 16-component banks
    and every finished run are untouched. The links are relative, so the folders can move together.
  - **Arms**, 20k steps, seed 42, prediction weight 1, the recipes of stages 35 and 33 on the new banks:
    - 10 units, field off: `ssl_uni_jitter_nopatch_restail_t104_causal_nofield_k32_predict1_c10_20k`;
    - 10 units, gain off: `..._nofield_nogain_k32_predict1_c10_20k`;
    - 24 units, field off: `..._nofield_k32_predict1_c24_20k`.
  - **Controls** with 16 components: stage 35's two 10-unit weight-1 arms and stage 33's 24-unit arm. Their scores
    are reused from stage 35's reports.
  - **After the reports, no training:** `normative_components.py` on both source banks, into
    `causal/stage37/normative_components_<bank>.txt`. The field-off table should repeat item 27's. The gain-off one
    is new: check that 32 is a good choice there too.
    - The first PC run (16 Nov) stalled in this check, which then ran while the arms trained and before the reports:
      the arms finished, but the reports never started. Its workers were forked after the parent's PCA. They are now
      started fresh (spawn), each with one BLAS thread, and the check runs last. Rerunning stage 37 skips the
      finished arms and banks.
  - **Reports** in `causal/stage37/`: the precision audit, own variation, the code audit and the style audit.
  - **Success:**
    - The image-filter row of the field-off k32 arms near 7.4 / 5.2.
    - The backbone GAP sharper than the 10-unit controls' (11.1 / 11.9; gain off 8.9 / 8.0).
    - The 10-unit code below 17.3 / 17.6 (gain off 14.9 / 14.3), "alone" above .42 / .39 (.56 / .60), with anatomy
      and lesion position holding (.61–.74).
    - Read the four 10-unit arms as a curve: the code's error against the image-filter row of its bank.
  - **Checked locally:**
    - Unit tests (`tests/test_residual_bank.py`): a linked bank gets its own residuals through relative links and
      leaves the source alone; more components leave less of the scan; more components than the basis holds, or a
      bank as its own source, are refused.
    - The default is unchanged: the repo `bank`'s FLAIR residual, rebuilt through a linked copy with 16 components,
      equals the stored one bit for bit on all 3,400 subjects.
    - On a Mac render of `bank_causal_nofield` (2,000 / 32 / 1,000), the image filter reads the stored residuals to
      9.0 / 6.6 with 16 components, the PC's numbers, and to 7.4 / 5.2 with 32 through the linked bank.
    - Two-step runs of the 10- and 24-unit arms finish on the linked bank, and own variation reads the 10-unit one.
      `normative_components.py` runs on 200 test subjects. Shell syntax and the stage 37 dry run pass.
    - Not run locally: the gain-off bank, and the precision, code and style audits on a linked bank. They take a
      bank folder like any other.
  - **Run:** `bash out/burden_feasibility_20261102/run_pc.sh 37`. Stop stage 36's arms first (below). It reuses
    stage 35's report caches, so start it after stage 35's reports are written.
- **Stage 36 (started 16 Nov; its no-training check read in section 7, item 26: an MLP map does not let 10 units hold
  burden, so the arms are unlikely to help).** Can a 10-unit code match the 24-unit one when its map to the other view
  is non-linear?
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
  - **Gain off** would have followed if the field-off arm worked; dropped with the arms.
  - **Stop the arms** (recommended 16 Nov, after the check): the 10-unit arms are very unlikely to reach the 24-unit
    level, and at 24 units the MLP map read burden 0–1.6 voxels worse than the linear one. On the PC,
    `pgrep -af predict1_mlp128` lists them and `pkill -f predict1_mlp128` stops them. Stage 36 then exits by itself,
    without its four reports.
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

**Ran 25 Nov, read 26 Nov: stage 73, why stage 68's evaluated code dipped. Answer: the gauge's stored statistics.
Recomputed on clean training images, the code matches as well as the loss's own and the dip is gone** (built as stage
70; renumbered when merged with the temperature arm's stage 70). Every `--real` eval now carries the
readouts of section 5, each undoing one difference between the evaluated code and the loss's. Stage 73 first reads
them on stage 68's 20,000-step checkpoint with no training (`adni_predict1_c24_20k_at20k`; its plain row must repeat
stage 68's last one, .831, or the checkpoint did not load as trained). Then stage 68's recipe at seed 1 (does the dip
recur?) and at seed 42 with `--gauge running` (one normalization in training and at test), 20,000 steps each, keeping
every eval's checkpoint. Reading, at @32: `batch_jittered` is the loss's code and the plain row the evaluated one;
the gap between them is what needs explaining. `batch_clean` closing it blames the gauge's statistics,
`running_jittered` closing it blames the clean images. If `recalibrated` closes it too (top-1 up, centred probe
down), recomputing the gauge before export fixes the code without retraining. Run it on Run:ai (section 3).
- **The check holds.** The 20,000-step checkpoint's plain row repeats stage 68's last row exactly (.831, @32 .951,
  centred probe .86), so the checkpoint loaded as trained.
- **The stored statistics are the cause; the jitter is not.** Stage 68's checkpoint, validation subjects:

  | readout | top-1 among all 275 | @32 | centred probe |
  |---|---|---|---|
  | plain (stored statistics, clean images) | .831 | .951 | .86 |
  | `batch_clean` | – | .984 | – |
  | `batch_jittered` (the loss's code) | – | .983 | – |
  | `running_jittered` | .773 | .938 | .87 |
  | `recalibrated` | .953 | .985 | .52 |

  - With batch statistics, clean or jittered images change @32 by at most .02 at any step of either batch-gauge run.
  - Recomputing the gauge closes the whole gap to the loss's code. The views' codes can then no longer be told apart
    (centred probe .52; chance .5).
  - Recomputing does not lift an untrained code (step 0: .029 and .038 against plain .016 and .042).
- **Seed 1 repeats the dip, and recomputing removes it.** Top-1 among all at steps 2,000 to 20,000:
  - plain: .458 .269 .287 .362 .493 .480 .478 .842 .815 .896;
  - recalibrated: .571 .738 .829 .878 .893 .913 .918 .956 .947 .949.
  - Recalibrated @32 rises .863 → .986, within .013 of `batch_clean` at every step. Its centred probe reads .51–.56
    from step 4,000 on; plain reads .85–.99.
  - At 20,000: training subjects .993 (plain), effective rank 18.1 / 16.7, backbone GAP top-1 .958.
- **The running gauge is not a fix; it is a worse recipe.** Seed 42 with `--gauge running`, where the loss sees the
  same statistics as the test. At 20,000:
  - top-1 .635 (recalibrated .527), @32 .894, effective rank 6.2 / 6.2 of 24;
  - backbone GAP top-1 .222 (batch-gauge runs .935 / .958); training InfoNCE .68 (batch gauge .03–.04);
  - it dips too: .645 at 16,000, .218 at 18,000, .635 at 20,000.
  - On the synthetic bank it was slower with no gain. Dropped.
- **Why the stored statistics go wrong (not tested).**
  - They are a running average over roughly the last 10–20 training batches (BatchNorm momentum .1).
  - The loss never uses them: in training each batch is standardized by its own statistics. So nothing in training
    keeps them right.
  - They are wrong for jittered images too (`running_jittered` @32 .938 against `batch_jittered` .983), so the gap is
    not clean images against jittered ones.
  - AdamW runs at a constant learning rate, so the late recovery is not a learning-rate decay.
  - The test: compare stored and recomputed means and variances, feature by feature, on seed 1's kept checkpoints
    (`model_STEP.pt`).
- **What changes.**
  - Stage 68's code, read with the gauge recomputed: top-1 among all .953 (seed 1 .949), @32 .985 / .986. Quote these
    for the recipe on ADNI, not .831 / .896.
  - Stage 68's reading of the centred probe (below: "the two views' evaluated codes pulling apart in shape") was the
    stored statistics, not the network.
  - The plain rows' effective rank, matching-unit r, held-out loss and modality probes all read the stored
    statistics. None is measured on the recomputed code yet.
  - Anchor scoring and any ADNI export should recompute the gauge first. That stays label-free: clean training images
    only.
  - **Synthetic banks, untested.** Every TailCode arm uses the same gauge. Bundles, holds, ρ1 and own parts are read
    from codes gauged by the stored statistics. The frozen recipe's code batches are also rebuilt from that code
    (`features()` in eval mode, every 500 steps), so there the stored statistics steer training as well.

**Ran 24 Nov: stage 68, the recipe's first ADNI run** (the plan's step 3). `adni_bank.py` exports the
scans (section 4), then the residuals, then stage 33's 24-unit prediction arm with `--real`, 20,000 steps, seed 42.
Set `ADNI_DATA` (the folder of subject folders) and `ADNI_LABELS`, optionally `ADNI_MASKS` and `ADNI_SPLIT`. Bank,
runs and logs live in `adni/`, so stages 66 and 67 never meet a run without latents. It runs on Run:ai, where the
scans are (`runai_submit.sh`, section 3). First job `burden-s68-1124-1206` (commit 1ec5f0c): the split matched
October's split.json. 14 label subjects have no FLAIR (one no T1 either). The rule excluded 54 of 1,606 (36 / 14 / 4),
all for Dice; the 6 small masks are among them. Training plus validation is 50, as October's 50 bad-mask subjects. Kept
1,121 / 275 / 156. 0.36 s/step on an A100 (20,000 steps ≈ 2 h). The 1 mm grid is 181 × 217 × 181, so the 64³ box at
3 mm crops 12 mm off each end in y; it cuts at most 0.9 / 1.4 mL of brain (one FLAIR scan over 1 mL, of 1,552).
Validation subjects, step 0 → 20,000: cross-view top-1 among all 275 .042 → .831 (training subjects
.938, a memorisation gap of .11, as at 6,000 steps); top-1 in sets of 32 .178 → .951; backbone GAP top-1 .109 → .935;
effective rank 6.7 / 6.4 → 16.3 / 15.9 of 24 units; matching-unit r .36 → .76; L2-normalised modality probe .82 → .93
(October .96). The mean-centred RBF, which a constant per-view offset cannot raise, is .864 at 20,000 (October .70),
so the views' codes differ by more than an offset, and training put it there: .555 untrained, .996 at 2,000 steps. It
moves against matching: .96 / .94 / .98 in the dip, .89 and .86 as matching recovered (r −.76 with top-1 from 6,000
steps on; the offset-sensitive probe does not track, r .22). So the dip is the two views' evaluated codes pulling apart
in shape. **Corrected 26 Nov (stage 73, above): the gauge's stored statistics, not the network. With the gauge
recomputed, the centred probe reads .52 and top-1 .953.**
**Not stable:** from 6,000 to 16,000 steps the code's top-1 fell .489 → .278 (training subjects .595 → .362) while
the training loss fell .175 → .053 and backbone GAP top-1 rose .775 → .900; it jumped back to .713 and .831 by 20,000.
Not overfitting (training subjects fell as far) and not the backbone (GroupNorm, the same in training and at test).
The loss's code and the evaluated code differ in two ways only: TailCode's BatchNorm gauge (batch statistics in
training, running ones at test) and the jitter (evaluation images are clean). Only the 20,000-step checkpoint exists
(`model.pt` is overwritten at each evaluation). Stage 73 (above) separates the two: it was the gauge's stored
statistics. Then anchor scoring, not built:
FreeSurfer 7 from `data_local/freesurfer.csv` (1,543 / 1,620 subjects) and UC Davis WMH (941, almost no ADNI4), matched
to scans through the manifest's IDs and dates.

**Status on 25 Nov (stages 46–70; read this first).**
- **The bridge:** its first loss collapsed (item 37); the fixed form, the theory's own version, holds the weak factors
  no better than matched width (item 38). The recipe's 24-unit advantage lies outside Yao's theorem.
- **The lesion's position was the benchmark's fingerprint:** it took about half a narrow code and crowded out the weak
  factors (item 39). Without it a narrow code is a good causal block, and the label-free count beside measured anchors
  finds the concealed factors as separate directions (item 40), though burden's is seed-fragile (item 44).
- **What carries the recipe** (items 41–42): the residual input, jitter and the prediction term, at both widths. The
  tails are optional at 24 units and keep a 10-unit code linearly readable.
- **Code batches:** a large gain at 10 units on the fixed bank, replicated at seed 2 (items 41, 44); neutral at 24 units
  with the position present (item 42).
- **Effect estimation:** narrow codes hold strong factors non-linearly, so the tabular benchmark now also runs the rbf
  learner by default (`causal_tables`, item 40).
- **Label-free graph discovery from two views** stays out of reach; rotating the code into canonical directions gives
  PC no edges by construction (item 43). The graph needs anchors.
- **Running:** stage 53, the map (`grid`) and position-free (`quantile`) prediction targets at 24 units (section 9).
- **Stage 54's footprints are read (item 45):** without labels, locality finds the factors that have a place
  (temporal atrophy in most codes, burden at the fixed lesion site) and none of the brain-wide ones. The rotation
  passes its rule on the 24-unit codes and fails on the code-batch arm in FLAIR. Its discovery reports and the
  intrinsic dimension are not recorded here yet.
- **Built, not run:** stage 55, no training: individual effects and counterfactual outcomes in the tabular benchmark
  (`--counterfactual`), on the same codes (section 9). Run it after stage 53, with stage 54 or alone.
- **Stage 56 is read (item 46):** at 8 units, the cosine's matched width, the code-batch recipe gives up most of
  ventricle's and burden's own parts (.43 and .30 against .78 and .60) and 4 voxels of volume precision; the other
  five factors and the ATE hold. The two units past the matched width are not spare. The new
  `latent_causal_discovery --sphere` removes a code's length before PC; the units' graph stays at the mixing baseline.
- **Stopped after T1 (20 Nov 16:19, user's request; started 13:40):** stage 57, no network training. Saved in
  `causal/stage57/recon_{drawn,fixed}.json`: T1 for truth, truth + style, the 10-unit code and its twin on both banks
  (the drawn bank's 24-unit rows and every FLAIR row never ran), and the fixed bank's T1 figures. A rerun redoes every
  decoder. The first figures drew every value at or below 0 as black, which is most of a centred brain; scan rows now
  use each true brain's 1st-99th percentile, and `recon_check.py --redraw NPZ` redraws a saved figure without
  training. The stage as built: a fresh 3D decoder learned from
  each frozen code to its view's scan and residual (`recon_check.py`). The 10-unit code (stage 35), the 24-unit code
  (stage 33) and the fixed bank's 10-unit code-batch code (stage 50), each against its untrained twin (floor), the
  bank's 10 rendered shared latents, and those plus the view's own style redrawn as `style_audit.py` does (the
  ceiling: style is most of the scan's variation, ridge R2 .16 -> .67 for T1 on a small local bank, and untrained
  twins carry gain). Read R2 of scan and residual, the residual's R2 over the lesion cells, and the lesion volume read
  off the rebuilt residual (e, voxels). The residual is the column that tests content; the scan mostly tests style.
  Twenty decoders in all (twelve on the drawn bank, eight on the fixed), up to 150 epochs each at about 30 s an epoch
  on a shared GPU, so it takes most of a working day.
- **Stage 58 is read (item 47):** on the independent bank 8 units hold ventricle and burden apart (worst direction
  .75, blend .00) and drop lr instead (own R² .26). The edges made the blend, with the rule's caveat. 8 units miss one
  direction on either bank; curves take 2 on both.
- **Stage 59 is read (item 48):** at 9 units the curves take the extra unit (6 own directions, 3 curves, last spectrum
  value .04 / .02), as expected. The missing direction blends ventricle, burden and lr. Volume precision nears the
  10-unit codes; ATE bias and the decoded graph are no better than at 8 units.
- **Stage 63 is read (item 50):** stage 62's rebuild-100 gain does not replicate: the seed-2 8-unit control
  (every 500) holds all seven own parts (last value .56) and the seed-2 rebuild-100 arm loses the seventh (.12). At
  8 units the seed decides. Rebuilding every 25 steps hurts. 9 units rebuilt every 100 (seed 42) hold all seven and
  match the 10-unit code on every criterion of the goal; the seed-2 8-unit control misses only T1 volume error, by
  0.3 voxels. One seed each; count hold rates over seeds next.
- **Stage 64 is read (item 51):** about half of codes hold all seven own parts at 8 units (5/10) and at 9 units
  (4/8): the extra unit goes to curves and buys no reliability. Rebuilding every 100 steps costs it (3/9 against 6/9
  at every 500). Codes that hold mostly match the 10-unit code. Top-1 does not flag a held code (6 on the wrong side
  per width), though the highest-top-1 code of each width holds.
- **Stage 65 is read (item 52):** a 10-unit code compressed to 8 units without labels (redundancy) and fine-tuned
  holds in 3/5, the same as from scratch; PCA 2/5; 10,000 more from-scratch steps repair no seed. Where the parent
  holds and both removed directions are clean curves (held-out R² ≥ .98; seeds 42, 2, 5), the 8-unit code matches
  the 10-unit reference. That label-free accept rule was seen after the results and is untested.
- **Stage 62 is read (item 49):** at 8 units, code neighbourhoods rebuilt every 100 steps instead of 500 give the
  seventh own direction .45 / .36 (stage 56 .01) and leave one curve. Ventricle .62, burden .54. The causal outputs
  match the 10-unit code (decoded graph .86 / .85), and label-free discovery is the best yet (F1 .71). Prediction
  weight 3 does nothing and costs discrimination. One seed; replicate next.
- **Stage 60 is read (item 54):** the harmonic penalty (`warm.py --harmonic-weight 1`, a critic charging units the
  others predict non-linearly, i.e. curves) on the from-scratch 8-unit arm at seeds 42, 2, 3, 4 and 5 holds in 1/5
  against 3/5 from scratch, and matches in 0/5 against 2/5. It rescues neither failure, loses seeds 3 and 5, keeps
  a curve in every code (brain size's in all five) and costs .21 of cross-view top-1. The penalty fell to about .02
  while the curves stayed: the critic stopped seeing them. Not a lever in this form. A critic fitted afresh on the
  saved codes would say whether any penalty on units predicting units can see them.
- **Built, not run:** stage 61, no encoder training: counterfactual images from tabular data. A tabular cause of an
  image factor is raised, the code moves by its estimated effect, a decoder rebuilds the image, and the result is
  scored against the generator's own counterfactual render of each subject (section 9).
- **Stage 66 is read (24 Nov; item 53):** can a run be chosen without labels? `criterion_check.py` on 149 finished runs:
  label-free signals from training and validation subjects against each run's own parts on test subjects
  (`causal/stage66/`; tests: `tests/test_criterion_check.py`).
  - Only cross-view retrieval passed the rule fixed beforehand, narrowly: top-1 among all validation subjects, rho .41
    (mean own part) / .31 (weakest factor), removing 53% / 51% of a random pick's shortfall against a 50% bar.
  - In the current recipe's bank (field-off causal, 24 units) retrieval removes 96% / 92%. It fails where an ablation
    keeps the easy anatomy (fixed-position 10 units: no residual, top-1 .91, own parts .43) or holds the factors
    non-linearly (no tails: ridge .57, boosting .69). The mutual-information signals pick the Barlow Twins arms
    (own parts .26-.27): that loss maximises cross-view correlation itself.
  - The signals follow ventricle, thickness, temporal, sulcal and lr (rho .3-.58). Burden follows the other six own
    parts in the prediction-recipe banks (rho .46-.90) but traded against them in the older banks (-.54 to -.80).
  - Seed spread: SD .04 on the mean own part, .07 on the weakest; recipe means differ by SD .16 / .20.
  - Choosing by the measured factors' exact own parts carries over to hidden factors (hide 1-3: rho .64-.74, 70-81%
    of the shortfall removed; lr and sulcal hidden: .78, 79%), far better than retrieval (.38-.48, 46-57%). Hiding
    burden alone: .27, 12%. Stage 67 repeats this with noisy measurements, as ADNI gives them.
- **Proposed (25 Nov), not built:** stage 69 tests stage 65's clean-compression pattern as a label-free accept rule
  on ten new seeds (6–15). The rules are in section 9, for the user to check before anything is built or trained.
  The user declined a second rule from item 55 that would have scored the same children by measured anchors.
- **Asked 25 Nov, no anchors:** another way to make 8-unit seeds hold. In the five from-scratch seeds, the codes that
  hold spend one direction on curves (1|1 at seeds 2, 3, 5) and those that fail spend two (2|2 at 42, 4; item 54).
  Item 46 traced the curves to the cosine InfoNCE's push to spread subjects over the sphere. Two untried levers aim
  at that push:
  - Raise the temperature from τ 0.1 to 1 (the trainer's own default). Expanded in 1/τ, InfoNCE is alignment plus
    an isotropy term on the code's second moment, with the higher moments (full uniformity, which bends a factor
    into a second direction) entering at higher orders. A high temperature leaves mostly the isotropy term, which a
    straight code meets and which values the ventricle–burden difference like any other direction. Consistent with
    section 11: hard-negative weighting, which sharpens the loss, starved lr and temporal. Inside Yao's setup.
  - A projector head between the 8 units and the loss (`--contrastive-proj-dim` exists in the general trainer, not
    yet in `warm.py`), so the head can make the bends. This leaves Yao's setup, since the loss no longer acts on the
    matched-width code, and the spare unit could take up style.
  - The user chose the temperature and agreed its rules and the factor of 10.
- **Stage 70 is read (item 56): temperature 1 makes 8 units worse.** Holds 0/5 and matches 0/5, against 3/5 and 2/5
  from scratch. Every code spends 4–5 directions on curves and loses burden, and top-1 falls .28 on average. Not a
  lever. Training ended 09:30 and the GPU is free, so stage 68 can start.
  - Item 56's synthesis answers the user's question (25 Nov) of why 8 units fail. InfoNCE prefers bending strong
    factors to holding weak ones, and at 8 units the recipe's counter-pressure only about evens it. Its proposed fix,
    a matched-width projector head, is stage 71 (below).
- **Stage 72 is read (item 57): fail.** The suggested local loss (seven local changes both scans agree on, from
  another assistant via the user) would not target the missing factor. A held weak factor adds no more local shared
  variation than a failure's curve leaves, and a shared artifact adds more. Do not build it in this form.
- **Stage 71 is read (item 58): a projector head makes 8 units no better.** Holds 2/5 and matches 1/5 against 3/5
  and 2/5 from scratch; it rescues neither failure, loses seed 2, keeps two curves in every failing code, and never
  bends (linearity .98–.99). No cost to top-1 or style. A matched-width head keeps Yao's setup; this corrects the
  bullet above that says a projector leaves it.
- **Stage 75 is read (item 59): the shared leftover detects failing codes; merge and repair fail.** ρ1 (what the
  two views' backbone features still share once regressed on the code) is higher in every failing code than in
  every holder of its cell (44 of 44 pairs, AUC .97), so the lowest ρ1 per cell holds in 8 of 8 cells. Merging a
  cell's runs holds in 4 of 8 cells and the repair fixes 4 of 23 failures: both fail their rules. Next, if the user
  wants it: test ρ1 as a seed rule on runs it never saw (new seeds; other banks and widths), then ADNI.
- **Stage 77 (section 9):** step 1, the toy, stopped (item 62: its calibration found no setting where 12 units hold
  and 8 fail). The user chose to go straight on. Step 2 is built (`warm.py --sufficiency-weight`, 6 tests, smoke runs).
  **Running:** step 3, twenty Run:ai jobs submitted 26 Nov 10:49 (`burden-s77-suff-s16…25`, `burden-s77-lean-s16…25`).
  When all log DONE: copy `runs/` and `logs/` back (`..._suff1_c8_20k_a100_sN`), then `run_pc.sh 77b`.
- **Stage 76 is read (the user asked for both parts; rules written before any of its results):**
  - A (item 60): at 9 and 10 units the detector fails its rule (AUC .55, 7 of 10 within-cell pairs).
  - B (item 61): on ten new 8-unit runs (seeds 6–15, A100s) 5 hold, and both rules work: AUC 1.00 (25 of 25
    pairs), and the lowest ρ1 among any three runs holds whenever one of them does (110 of 110). At the matched
    width the rule is now 69 of 69 within-cell pairs, 25 of them on runs it never saw.
  - The ten Run:ai jobs `burden-s76-a100-s6` … `-s15` (one A100 each, `runai_warm.sh`) ran 25 Nov 23:00 to 26 Nov
    03:35; their runs and logs are copied back (`..._abl_codebatch_predict1_c8_20k_a100_sN`), and `run_pc.sh 76b`
    scored them (log `run_pc_76b.log`).
- **The other session (25 Nov):** it pushed `c8cfef3` (its stage 74, a content-size sweep with Run:ai submission)
  and `1bf7c96` (its stage 73, why stage 68's evaluated code dipped) on top of `d3f7d74`; the cluster checkout is at
  `1bf7c96`, and its jobs `burden-s74-width-s3/s4/s42` show as Failed on Run:ai (not looked into here; later: they
  trained all 27 runs and failed only at the causal-learn check, fixed in `21e8a25` and `482d1ce`; stage 74 is read
  as item 63). This
  checkout's branch has `3bea16d` (not pushed) and today's uncommitted work (stages 71a, 75, 76; items 58–59), so
  the next commit needs a merge with `fork/temporal009-attention-full-width`: both sides edit `HANDOFF.md`,
  `run_pc.sh` and `warm.py`. My stages were renumbered 75 and 76 so that none collides with the other session's.
- **Stage 71a is read (item 58, A100 replicate):** holds 1/5 (seed 2), 3/10 pooled with the PC's five; no seed held
  on both GPUs. Reports ran 17:04–18:05 (`run_pc_71a.log`, `causal/stage71a/`). The main checkout was fast-forwarded to
  `sync-25nov` (3bea16d) once stage 71 exited and the scratch worktree was removed; HANDOFF and `run_pc.sh` carry
  uncommitted edits since (stage 71a, items 58, interims).
- **Code and Run:ai, 25 Nov (the user asked to commit and to set up Run:ai for these stages):**
  - Committed `4896045` (stages 70–72). The remote branch had 4 commits from another machine (stage 68's ADNI export,
    `warm.py --real`, `runai_submit.sh` and `runai_job.sh`, the paper plan). They were merged in a scratch worktree, so
    the running stage 71 never saw a rewritten `run_pc.sh`: `d3f7d74`, pushed.
  - The merge kept both sides of each conflict. `tests/test_warm_prediction.py`'s fake bank gained `__len__`, which
    `draw_prediction_batch` has used since `--real`; three of its tests failed on the incoming branch without it.
  - The local checkout stays at `4896045` until stage 71's `run_pc.sh` exits. Then fast-forward it to `sync-25nov`
    (the merge, plus this note) and remove the scratch worktree.
  - The cluster checkout (`/nfs/home/nglazman/crl-2/multiview-crl`) is at `d3f7d74`. Its `summary.txt` from the
    stage 68 job is kept in `git stash` (`stash@{0}`). `bank_causal_nofield_fixedpos` was copied there (6.8 GB, 24
    files, sizes checked).
  - Stage 68 ran there, despite the "built, not run" paragraph at the top of this section: Run:ai lists
    `burden-s68-1124-1206` as completed on 24 Nov. Not read here.
  - `runai_submit.sh STAGE` runs a whole `run_pc.sh` stage as one A100 job, all of its arms on one GPU. One GPU per
    seed needs a seed subset per job and one report job at the end; not built.
  - Timing job `burden-timing-a100-1125`, submitted 11:54 on node dgxa100-a. It runs stage 71's recipe for 600 steps,
    one arm alone and then five at once (`timing/timing_a100.sh`, log `timing/timing.log` on the cluster; runs in
    `timing/runs`, apart from `runs/`). Results (A100-SXM4-40GB; steady rate between steps 300 and 500):
    - One seed alone: 0.295 s a step (0.39 averaged over the first 600 steps, start-up and one neighbourhood rebuild
      included). A 20,000-step run comes to about 2 hours.
    - Five seeds on the one GPU: 0.335–0.35 s a step each, only about 15% slower. All five together come to about
      2.1 hours, against about 3.7 on the PC's 3090 at 0.65–0.66 s a step.
    - One seed leaves an A100 mostly idle, so its pace is set off the GPU. One job per stage (`runai_submit.sh` as it
      is) is nearly as fast as one job per seed and uses a fifth of the GPUs. An A100 may well take more than five
      seeds; not measured.
    - The same seed reproduced on the same GPU: the alone run and five-at-once seed 42 log identical losses at step
      600 (10.102 / 10.215 / .399). So the PC's runs are probably reproducible on the PC too, and a move to A100s
      changes trajectories the way a new seed would.
- **Stage 67 is read (item 55):** measured anchors choose runs (`causal/stage67/`; `anchor_check.py`, no training).
  - With lr and sulcal hidden, as on ADNI, the run whose code best reads the measured anchors comes close to the
    best on the hidden factors. Anchors adjusted for each other at reliability .9: rho .77 [.64, .84], 89% of a
    random pick's shortfall removed, against 40% for top1_32 and 66% for top1_all.
  - Noise down to reliability .5 costs nothing, and raw anchors do as well.
  - It works in the fixed-position bank, where retrieval is worse than random. With burden hidden it fails outside
    the prediction-recipe banks.
  - After the fact, the same score finds the 8-unit seeds that hold within a cell, which top-1 could not: the top
    seed holds in 9 of 9 cells, and holders come first in 42 of 44 pairs. They match in 4 of 9.
- **Paper (discussed with the user 18 Nov):** one frozen recipe, three claims, an ADNI validation, and reading rules
  fixed before each result. The user is wary of anything that looks like fine-tuning; keep new levers to what a claim
  needs.

**The plan (adopted by the user on 18 Nov).** It replaces matched width and the 10-unit lever search. Theory and
stages 41–44 say two views identify the shared anatomy as a block, not its factors (section 8). Measurements are the
identifying information available here.
1. **Representation.** Stage 33's 24-unit prediction code, claimed as an embedding of the shared anatomy.
   - The width comes from a label-free rule, the same on ADNI: count the frozen backbone's cross-view canonical
     directions that agree above about .5. On stage 19's CCA that is just under 24.
   - For a formal guarantee on the synthetic bank, train one arm of the bridge (align all units, the entropy term on a
     block as wide as the shared latents) and show it matches the current recipe. Under the cosine the block is 11
     units and the prediction map must be flexible: stage 46b (section 9). Stage 46 (10 units, linear map) was not
     the formal case and collapsed under an earlier, degenerate form of the loss (item 37). Stage 46b holds the weak
     factors no better than matched width (item 38), so the recipe's 24-unit numbers rest on measurement, not on the
     bridge.
2. **Causal variables from anchors.**
   - Synthetic: all seven factors anchored, as in stage 41's decoded discovery (24 units: skeleton F1 .84 against the
     truth's .85, the same 3 orientation errors).
   - ADNI: the measured values are the nodes (FreeSurfer whole brain, ventricles, cortical thickness and hippocampus;
     the UC Davis WMH volume for burden). The code's readout of each serves as a second, independent measurement,
     because error in a single noisy measurement creates false edges (the 10-unit readouts in stage 42).
3. **One block node for the rest:** the code minus its fit on the anchors. Use it to adjust for confounding and to test
   links to outside variables such as cognition and diagnosis. Do not split it, and expect no edges between it and the
   anchors: those are absorbed (section 8).
4. **Discovery at the block level.** Stages 42 and 44 are the controlled evidence. Trained codes find shared content
   beyond the anchors and their untrained twins do not (readout anchors); strong omitted factors come out as
   directions; weak ones stay in the block. The framework names this an informative outcome.
5. **Graph:** FCI (partial ancestral graphs) wherever hidden confounding is plausible, as it is once anything is
   concealed.
- **Stopped:** the 10-unit lever search (stage 43 may finish; the 10-unit result becomes a finite-sample finding for
  the paper: matched width keeps the strong factors and drops the weak ones) and tuning the label-free count (the loss
  is mixing, item 36).
- **Burden's role:** the completeness test of the representation. At 24 units its own part is .48 / .47, and .60 / .58
  with gain off (item 22).
- **Two different routes checked 18 Nov (items 37–38); neither changes the plan.**
  - Treating the two views as proxies (proximal causal inference) leaves the confounding where one view or both views
    leave it: the code's shortfall is content both views omit, so only a more complete representation reduces it.
  - For the block node's adjustment (item 3 above), both views' backbone GAP removes 93% of the image-driven
    confounding against the 24-unit code's 83%. Worth weighing against the code's smaller width.
  - Separating own parts by variance heterogeneity works in simulation, but ADNI's age and sex give no usable variance
    change.
- **Later, if individual factors are wanted without labels:** treat scans from different ADNI visits as extra views.
  Pairs across visits share the stable factors (head size, asymmetry) but not the changing ones (atrophy, WMH growth),
  so Yao's partial-observability result splits the block into stable and changing parts. Check protocol changes
  between ADNI phases first.
- **Benchmark realism:** the sphere lesion's position takes about half of a narrow code (item 29) and has no ADNI
  counterpart. Stage 47 confirmed that it crowds the weak factors out (item 39). A diffuse WMH-like burden would
  bring the bank closer to ADNI. This checks the benchmark, not the method.

**Next, after stage 34's style and precision audits (16 Nov).** The numbered items below were written with the field
on.
- **Read `causal/stage34/own_variation.txt` and `code_audit.txt`.** Stage 33 passed all four success tests at weight
  1 (section 7, item 22): burden's own part in the code .48 / .47 with gain on and .60 / .58 with gain off, against
  .31–.37 before. Stage 34 found no style in any code and seed 2 as sharp as seed 42 (item 23). If seed 2's own part
  also holds, the prediction term at weight 1 is ready to join the recipe.
- **Read the rest of stage 35** (`causal/stage35/`; precision audit read, item 25): `own_variation.txt`, and
  `code_audit.txt`'s boosted rows and "both" column (do the 10 units hold anatomy and position non-linearly, and
  nothing else?). 10 units read burden worse than 24, so the recipe keeps 24.
- **Stage 41 is read** (item 34).
- **Stage 42 is read** (item 35). On the 24-unit code the label-free count keeps lr and the lesion centre and loses or
  dilutes the weak own parts (burden .04 / .03 in the block against .48 / .47 in the full code). At 10 units the
  width fixes the count. Step 3b (spatial footprints) stays the only route here to placing a concealed factor in the
  graph.
- **Stage 44 is read** (item 36). The anchors' readouts did not cause stage 42's loss: the weak own parts go out with
  dropped directions they share with other, partly predictable content, whichever anchors are used. Decided 18 Nov:
  stop the label-free split at the block (the plan above). Step 3b (spatial footprints) was built on 19 Nov as stage
  54 (section 9), at the user's request, with the intrinsic-dimension check. Its footprints, read 19 Nov (item 45),
  separate only the factors that have a place. Burden and the lesion centre both act where the lesion is, so their
  footprints overlap, as the item confirms.
- **Implementation proposal (17 Nov; adopted 18 Nov as the plan above).** Set the number of units at the
  causal variables, not at the encoder, as `paper/factor-discovery-framework.tex` already separates them:
  - representation: the 24-unit prediction code, claimed as an embedding of the content block;
  - a factoriser from the frozen code to K slots, one per causal variable, with K and slot sizes chosen on validation
    data and identifying information from anchors (synthetic: supplied factors with others concealed; ADNI:
    FreeSurfer regional volumes). ICA can supply the own parts label-free when the noise is non-Gaussian and the
    code near-linear (item 33), but not the graph: that still needs the anchors;
  - graph estimation on the slots and the observed variables.
  - Optional theory bridge (derived in another session, not a published result): InfoNCE on a learned K-unit
    projection, alignment and prediction on all 24 units. At the optimum the projection is a bijection of the content
    (Yao's argument) and the 24-unit code an embedding.
- **The search for a 10-unit lever is over (17 Nov, items 26–32; section 8).** A matched-width code holds every
  latent coarsely and needs spare width for the burden's precision and the weaker own parts. Open, if wanted:
  - a second 11-unit seed, so the claim that 11 cosine units beat 10 rests on two seeds each;
  - a write-up of stages 35–40 for the paper (section 8's matched-width note).
  - Reopened for the anatomy children (17 Nov): stage 43 (section 9) tests two levers judged on every factor the
    narrow code drops, not on burden alone.
- **Stage 38 is read** (item 30) but for `style_audit.txt` (do the new codes carry style?). The plan above (18 Nov)
  adopts 24 units, so the 10-unit goal in the bullets below is history.
- **Stage 37 is read** (items 28–29). Its `normative_components_*.txt` (the gain-off table) is unread; the
  32-component residual is dropped either way.
- **Candidates beyond stage 38 (17 Nov, not built).** Item 29: a 10-unit code spends about half its directions on the
  lesion's position. Ways to change what the scarce units hold without changing the theory's target:
  - The prediction term predicts the other view's tail features instead of its GAP: failed (stage 39, item 31).
  - A second InfoNCE term on the same 10 units, on batches of code neighbours (subjects close in the model's own code,
    so mostly matched on position). Weak: in neighbour sets as the code builds them (the price tables' "as trained,
    neighbour" rows) a 10-voxel burden unit still costs −.03 nats. It earns +.30 only with position and anatomy
    removed exactly, which finite neighbourhoods in a 10-unit code don't do. Not covered by the theory (section 8),
    and neighbour batches hurt burden before (item 5 below).
  - A larger head (encoder side): stage 40 (section 9) is its no-training check.
  - Benchmark side: several lesions per subject. Burden's range grows with the count, but each lesion adds three
    position latents to the block.
- **Stage 36's check is read (item 26):** a non-linear map does not let 10 units hold burden, so the recipe keeps 24
  units. Stop the stage's arms (section 9).
  - The whitening-ridge lever (rerun `bottleneck_check.py` at ridge 0.01 and 0.001) fails in the simulation of
    item 27 and costs position and anatomy there. It has not been run on real features; the expectation is low.
  - Untested on real features: the width the MLP map needs. The simulation says 11–12 units; the check ran 10 and 24.
- **White-matter normalisation is dropped (17 Nov):** the user declined it because it targets the lesion (section 1).
  It was planned as the real-data version of gain off, which also raised the anatomy children's own parts at 24 units
  (item 22).
- Stage 32's `own_variation.txt` and `code_audit.txt` are unread, but stage 33's reports cover its two heads that
  matter (field-off weight 3, gain-off weight 1); the rest are stage 30's weaker heads.
- **Decide with the user whether `bank_causal_nofield` becomes the main burden bank,** with the field-on bank kept as a
  stress test. For: a real bias field varies over centimetres and N4 removes it, so the 16³ field has no counterpart
  in corrected scans. The same question for gain off, whose real counterpart is intensity normalisation (WhiteStripe).
- **Tabular benchmark (read 16 Nov, item 24).** The end-to-end weight-1 codes match their own GAP on confounding
  and graph recovery in both seeds; with gain off they remove 87–90% of the confounding. That is one more reason for
  prediction weight 1 to join the recipe.
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
| a sharper residual (32 components) for the 10-unit code (stage 37) | the backbone sharpens with the field off, but the 10-unit code does not (17.0 / 17.3; gain off 16.6 / 16.7, worse), and every code loses temporal atrophy, thickness and sulcal (item 29): the extra components absorb focal anatomy, so it is not generic. At 24 units it helps burden (12.4 / 10.8, own part .54 / .57) at that cost |
| a larger head, one or two hidden layers of 512 (stage 40, no training) | at 10 units 0.3–3.2 voxels better through the linear map, nothing through the MLP map; 10 units stays 2.5–3 voxels behind 24 on the same features |
| predicting the other view's tail features or whole pooled vector (stage 39, no training) | burden worse than with the GAP target at 10 and 24 units, and the anatomy lost; the tails carry position through TailCode's per-cell statistics |
| a non-linear prediction map as the way to 10 units (stage 36) | 10 units read burden 5–13 voxels worse than 24, under InfoNCE and under prediction through either map. 10 units stays the goal; stage 37 sharpens the views instead |
| the two views as proxies: proximal two-stage least squares, one view instrumenting the other (item 37, no training) | removes no more confounding than adjusting for both views at once (24 units 84–86% against 83%; 10 units 58–59% against 60–61%): what one code misses, the other misses too |
| separating own parts by variance heterogeneity across age or sex (item 38, no training) | works in simulation from strength .25 (own parts .72–.77 of a .80 ceiling), but ADNI's age and sex give .06–.15, the same as shuffled labels |
