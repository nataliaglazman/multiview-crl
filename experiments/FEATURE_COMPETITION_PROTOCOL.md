# Draft: does brain-size variation suppress lesion-location learning?

Status: protocol and initial runner, 12 October 2026. Run instructions are in
`experiments/FEATURE_COMPETITION_RUN.md`; entry point:
`python -m training.feature_competition`. Only short software checks have been
run locally, not the full scientific experiment. Added later on 12 October: the
B′ arm (`Bp`, section 1), the same-architecture supervised control and the arm-A
pilot (section 0). The pilot was started on the local Mac with
`scripts/feature_competition_pilot.sh` the same day.

Implementation calibration: the initial runner uses brain-size amplitude 0.5
times the original coefficient and one-voxel movement pairs. At resolution 64,
the full original amplitude leaves no centres with a complete fixed-radius
lesion plus a white-matter blur neighbourhood. The reduced range gives 376
admissible centres. Four-voxel x-moves are unavailable in that set, whereas
one-voxel moves exist on every axis. These choices are fixed before scientific
training and are recorded in each data-bank manifest.

## Question and scope

At unchanged lesion visibility, does introducing between-subject brain-size
variation reduce the lesion-location information learned by the encoder? Does
increasing lesion visibility reduce that effect?

The first experiment uses **lesion location** as the target and **brain size** as
the competitor. It tests a controlled synthetic mechanism. It does not establish
that variance alone causes suppression, or that the same mechanism explains all
failures on the full synthetic distribution or acquired MRI.

Use independently sampled factors, with the causal and hierarchical generators
disabled. For this initial isolation experiment, fix the other anatomical
controls at their neutral values in every condition. Retain the same acquisition
style distribution in every condition. A subsequent replication can restore the
other anatomical factors; do not combine those two populations into one result.

## 0. Arm-A pilot: is there anything to suppress?

Run this before section 1. In arm A brain size is fixed and every other anatomical
control is neutral, so the lesion is the only thing a subject's T1 and FLAIR share.
Style, bias field and noise are drawn per view. A suppression test needs a
condition in which the target is learned. If arm A never learns lesion location,
A and B both sit at the floor and the section-1 grid cannot show suppression.

The pilot uses one seed (42) on an A-only bank (`prepare --arms A`). Rendering is
deterministic, so its images are exactly arm A's images in a full bank.

| Run | Question it answers |
| --- | --- |
| `supervised_gap`: the SSL encoder's architecture (backbone, GAP, MLP head, its 9 content units) plus a linear head to the three coordinates, trained with labels, one model per view, 5,000 steps | Can this global code carry lesion location at all? |
| `ssl`: the section-1 recipe, 10,000 steps | Does the self-supervised encoder learn location when nothing competes? |
| `supervised`: the heatmap localizer of section 2, one model per view, 5,000 steps | Does the image hold the location? |

**Held-out InfoNCE as a lesion meter.** At every checkpoint the trainer computes
the InfoNCE loss and the cross-view top-1 retrieval rate on the probe split, in
fixed batches of the training size. Chance is 2 ln 32 ≈ 6.93 and 1/32. In arm A,
a sustained drop below chance can only come from lesion information. The check
uses held-out subjects because a fixed training bank could be memorized through
its noise pairs. Checkpoints are at steps 0, 250, 500, 1,000 and 1,500, then
every 1,000. Each is evaluated with the section-1 probes (`evaluate --checkpoint
all`), and `curves` plots the learning curves.

Proposed criteria, written down on 12 October before any pilot result:

- **`supervised_gap` works** for a view if its test lesion R² (common regime,
  mean over x/y/z) is at least 0.5. It **fails** below 0.1.
- **`ssl` learns** location if both probe families on the global code exceed
  the step-0 floor by at least 0.2 in at least one view, and held-out InfoNCE is at
  least 0.1 nat below chance. It **fails** if the code stays within 0.05 of the
  floor and InfoNCE within 0.05 of chance.
- Anything in between is unresolved. Extend every affected run with `--resume`
  by the same number of steps.

| Outcome | Reading | Next |
| --- | --- | --- |
| `supervised_gap` fails in both views | The GAP code cannot carry location: a pooling blind spot | Switch the readout to `global_pool="attention"` (exists in `MultiviewConvEncoder`) and repeat the pilot |
| `supervised_gap` works only for FLAIR | Location is not really shared across views | Raise T1 contrast (arm C) or test same-modality pairs |
| Both controls work and `ssl` learns location in A | The comparison is testable | Section 1: A, B and B′ first, then C and D |
| Both controls work and `ssl` fails even in A | Nothing competes in A, so the failure is not competition | The objective does not find this target; competition cannot explain its absence |

## 1. Matched training comparison

### Five conditions

| Arm | Lesion contrast | Brain-size variation |
| --- | --- | --- |
| A | Reference, multiplier 1 | Absent: effective brain-size control fixed at 0 |
| B | Reference, multiplier 1 | Present: independent brain-size draws at half the original amplitude, one draw shared by T1 and FLAIR |
| B′ (`Bp`) | Reference, multiplier 1 | Present, but T1 and FLAIR disagree: T1 uses B's draw, FLAIR an independent draw from the same distribution |
| C | Increased, candidate multiplier 1.5 | Absent: same setting as A |
| D | Increased, same multiplier as C | Present: same draws as B |

**Why B′.** If B learns less than A, two mechanisms could be responsible:

1. **Competition through the loss.** Brain size is shared by the two views, so
   it solves part of the matching task, and the pressure to find the lesion drops.
2. **Input interference.** Brain-size variation changes how the lesion's
   surroundings look. One example is the lesion's position relative to the moving
   tissue boundaries, which a GAP encoder may use as an implicit position code.
   This has nothing to do with the loss.

B′ keeps the second mechanism and removes the first. Each B′ image carries the same
brain-size variation as in B; its T1 is B's T1 exactly. Because the two views
disagree, brain size cannot help match them, and alignment pushes it out of the code.

| Lesion recovery if the truth is… | A | B | B′ |
| --- | --- | --- | --- |
| Competition through the loss | learned | lower | ≈ A |
| Input interference | learned | lower | ≈ B |
| No effect | learned | ≈ A | ≈ A |
| Nothing learned (section 0 failed) | floor | floor | floor |

B′ also asks the encoder to discard brain size. If B′ lands between A and B, run
an exactly stratified control: every subject in a batch shares one brain size, so
brain size cannot separate negatives and the code is not asked to drop it. The
manipulation check reads brain size from the global code: it should be readable
in B and not in B′, where each view is scored against its own draw.

"Reference" and "increased" are the operational treatments. Call them low/high
SNR only after the visibility audit below confirms that description in each view.
The varying competitor is not assumed to be easy: report its recovery in B/D as
a manipulation check.

Draw a common subject bank first. Each subject has the same lesion position,
acquisition style, bias field and noise seeds in A–D and B′. B and D share the same
brain-size draw; A and C replace only its effective value with zero. B′'s FLAIR
draw is taken after all other draws, so adding it changed no existing value. The two
modalities retain their own distinct acquisition/noise seeds; matching seeds
across experimental arms must not make T1 and FLAIR share noise.

Use the reference clean-content prior: an independent standard-normal brain-size
draw, squashed by tanh and multiplied by 0.5 for the initial controlled runner.
Use the existing coefficient (nominal WM radius 0.5 plus 0.1 times that effective
control, hence a maximum radius shift of 0.05 here). Record both the raw draw and its
effective value. Do not use `synthetic_content_scale=0` to remove the competitor:
that changes multiple factors. Freeze or scale brain size alone.

### Lesion geometry must be independent of the competitor

The current `wm_interior` placement maps position controls through the available
white matter. Reusing its three raw controls after changing brain size does not
hold physical location fixed.

For this experiment, construct one fixed set of admissible lesion centres across
the entire prespecified brain-size range, with all other anatomy at its fixed
values. Every centre must accommodate the same complete sphere in every arm.
Sample locations from that set independently of brain size and reuse the exact
lesion mask across A–D. Use the same set for training, probe fitting and testing.

This changes the location-sampling distribution relative to the original recipe;
apply that change to all four arms and record it. Validate containment across the
allowed anatomy range, not just the observed training draws. If the set is empty
or too restricted, revise the allowed range before training and repeat the audit.
Do not silently shrink lesions, move them, or reject subjects separately by arm.

### Change visibility through contrast, with geometry and noise fixed

Keep lesion radius, mask, position distribution, blur and acquisition-noise law
fixed. Start with the reference `fixed` lesion-intensity rule in all four arms;
do not switch `fixed` to `styled` as the high-visibility treatment.

Add an experimental contrast multiplier before bias-field multiplication, Rician
noise and blur. If `B_v` is the lesion-free tissue intensity, `L_v` the reference
lesion intensity and `M` the common binary lesion mask, use:

```text
I_v(k) = B_v + k * M * (L_v - B_v)
```

Here `k=1` reproduces the reference intensity rule. Use candidate `k=1.5` for
C/D, subject to the calibration in section 2. Apply it in both modalities and
report their visibility separately. Verify that intensities remain valid before
Rician rendering, that contrast polarity is preserved and that no clipping is
introduced. A contrast multiplier is not a guaranteed proportional SNR change.

### Training recipe and replication

Use `experiments/encoder/conv_mlp.yaml` as the model/objective reference:

| Setting | Proposed value |
| --- | --- |
| Model | Existing Conv backbone with MLP global readout; train from scratch |
| Objective | Paired T1/FLAIR InfoNCE, temperature 0.1, with the code batch-standardized per view before the cosine (`--standardize-code`). Plain InfoNCE stayed at chance for 8,000 steps in the arm-A pilot (a cold start: near-identical codes), so it cannot serve as the competitor-free baseline; the standardized loss learned the lesion from about step 3,000 |
| Resolution / lesion radius | 64 cubed / 0.1 in renderer coordinates |
| Generator baseline | Clean content; style/content scales 1; T1 lesion value 0.4, FLAIR value 1.0 |
| Batch size / learning rate | 32 / 0.0001 |
| Training budget | 10,000 steps per arm |
| Training subjects | 2,000, with matched subject IDs across arms |
| Probe/validation subjects | 400 new subjects, shared across arms |
| Final test subjects | 1,000 new subjects, shared across arms |
| Training seeds | 42, 142, 242; identical initialization and batch order within each four-arm replicate |
| Data seed | 42 for the shared bank; distinct recorded streams for the separate subject splits |
| Checkpoints | Step 0 and every 2,000 steps; primary endpoint is step 10,000. For learning curves add early steps, the same in every arm (the pilot uses `--save-every 1000 --save-steps 250 500 1500`) |

Keep encoder width, content allocation, normalization layers, augmentations,
optimizer and all other loss settings identical. Add no lesion branch or new
loss for this comparison. Save the fully resolved recipe and code version.
The five arms across three seeds require 15 SSL runs. A, B and B′ answer the
main question; C and D add the visibility interaction.

Partition subjects before rendering, with distinct random streams for training,
calibration, probe fitting and testing. Store split hashes; do not rely on split
names alone to guarantee independence. Repeated acquisitions and counterfactual
versions of one subject must remain in the same split.

### Endpoint and comparison

The primary target is the **rendered lesion centroid in input voxel coordinates**.
Report per-axis and mean coordinate R-squared, plus mean/median Euclidean error
in voxels. These are not scanner-space millimetres. Do not average inactive
anatomical factors into a headline MCC or recovery score.

Freeze each encoder at evaluation. Fit the same ridge and small nonlinear probe
families using the 400 probe subjects, with hyperparameters selected by folds
within those subjects. Preprocessing is fitted inside each fold. Report both
probe families; do not select whichever favors the hypothesis on the test set.
Evaluate spatial maps as well as the global code, because GAP can limit
localization. The initial runner pools the native 16-cubed backbone map to an
8-cubed grid for probe cost; `evaluate --spatial-grid 16` retains the native grid
at greater memory and compute cost. Poor recovery at grid 8 does not establish
absence from the native map. Avoid automatic high-variance PCA truncation. Fix descriptor
construction and probe budgets before reading test outcomes.

The primary comparison evaluates A and B on **identical held-out reference-
contrast images with brain size fixed at zero**. Fit their probes on the same
independent calibration distribution. Evaluate C and D analogously at increased
contrast. Thus each within-row comparison uses the same test images and targets;
it measures the effect of the training distribution on recovery in that common
environment. Zero is within the varying arm's brain-size support.

Also report each arm on its own training distribution using fresh subjects and
appropriately fitted probes. Testing a constant-brain-size model across varying
brain sizes is a separate generalization result, not the primary suppression
estimate. Do not mix these evaluation regimes in one number.

For recovery score `R` (higher is better), prespecify:

```text
Competition penalty at reference visibility: P_ref   = R_A - R_B
Competition penalty at higher visibility:    P_high  = R_C - R_D
Visibility interaction:                      I       = P_ref - P_high
Input interference:                          P_input = R_A - R_Bp
Competition through the loss:                P_loss  = R_Bp - R_B   (P_ref = P_input + P_loss)
```

Positive `P_ref` supports a competition effect if the controls in section 2 pass.
Positive `P_loss` with `P_input` near zero locates it in the objective; positive
`P_input` means brain-size variation alone degrades learning for this encoder.
Positive `I` supports a larger penalty at weaker visibility. Show all seed-level
paired differences and learning curves. Subject bootstrap intervals describe
test-sampling uncertainty conditional on the trained models; three training
seeds provide only a preliminary estimate of training variability. Do not treat
thousands of test subjects as thousands of independent training replicates.

If learning curves remain unsettled, the conclusion is about learning within
10,000 steps. Any extension must use the same additional budget in every arm.

## 2. Establish unchanged input visibility and supervised recoverability

Complete this audit before interpreting the SSL comparison. Use a separate pilot
bank, proposed size 128 subjects, with no access to final test outcomes.

### Geometry, acquisition and preprocessing checks

For every matched subject, verify and save:

- Identical lesion-mask hash, voxel count, radius and centroid across A–D.
- Identical style draws, bias-field realization and noise draws across arms for
  each modality. Only brain size and the specified contrast multiplier may vary.
- Full lesion containment in white matter for every allowed competitor value.
- Unchanged lesion-free tissue values within the lesion support when toggling
  the competitor. Inspect the blur halo for effects from nearby boundaries.
- Identical target distributions and absence of accidental target/competitor
  association in the saved bank. Include a brain-size-only coordinate predictor
  as a check for a location shortcut.

Use **one saved normalization affine for the whole experiment**, estimated from
the reference-contrast B training bank and reused in A–D, all splits and all
counterfactuals. Keep the saved reference fixed if a training run is repeated.
The current `fixed_reference` option estimates constants per dataset instance;
selecting that option independently in four datasets is insufficient.

Inspect images and intervention maps after the exact preprocessing seen by the
encoder. Brain masks may change with brain size, which is part of the intended
competitor signal, but must not clip the lesion or change its local scaling.

### Measure visibility of location, not only lesion presence

Create controlled movement pairs: place the same lesion at two valid centres
separated by one input voxel along one axis, holding anatomy and acquisition
fixed. Cover x/y/z equally and reuse the exact endpoint masks across arms.
Record availability of valid pairs; never silently substitute a different
displacement. Use lesion on/off pairs only as a supplementary appearance check.

For each modality, compare A versus B and C versus D using the same movement.
Use common noise for paired endpoint differences to isolate the intervention.
Separately redraw acquisition noise while holding anatomy, bias field, style and
each endpoint fixed. Do not count deterministic bias-field variation as freshly
drawn acquisition noise.

On a subset of 32 pilot subjects, use 32 independent noise realizations per
endpoint. Define a common ROI as the union of the endpoint lesion masks plus the
one-voxel halo of the renderer's 3-by-3-by-3 blur. Report:

```text
signal = RMS over ROI of (mean_noise X_end2 - mean_noise X_end1)
noise  = sqrt(mean over ROI of the mean endpoint noise variances)
visibility proxy = signal / noise
```

Calculate this after normalization, with the same ROI across arms. This is a
local image-effect-to-noise proxy for a specified movement. It is not a scanner
SNR, an optimal detector, or an information-theoretic bound; spatial noise
correlation and finite-repeat estimation affect its interpretation.

Plot paired distributions, by modality, rather than one pooled average. Proposed
matching criterion: the 95% paired-subject bootstrap interval for the median
log visibility ratio (B/A and D/C separately) lies within log(0.9) to log(1.1).
Also inspect the weakest-visibility quartile and individual signal/noise terms.
The 10% margin is a design choice to record before the audit, not a universal
threshold. Failure to detect a difference is not evidence of equivalence.

C/A and D/B must show a consistent increase in measured visibility in each
modality before interpreting a low/high-SNR interaction. If the candidate 1.5
multiplier is unsuitable, revise it using only pilot data, record the choice and
freeze it before main training. Never tune it to maximize the SSL result.

### Supervised image controls

Train a target-only supervised image localizer independently for each arm and
each modality. Use the same spatial architecture, training budget and tuning
procedure everywhere. A heatmap predictor with a coordinate readout is a
suitable first control; give it only the image at inference. Masks may provide
supervised training targets but must not be inference inputs or define an oracle
crop. The labels supervise these controls only, not SSL training.

Also train the SSL encoder's own architecture with labels (`--kind supervised_gap`).
This is the identical backbone, GAP and MLP head with its 9 content units, plus a
linear head to the three standardized coordinates, one model per view. Seeded
like the SSL model, it starts from the same backbone and head weights. The heatmap
localizer shows that the image holds the location; this control shows whether the
code the SSL loss acts on can carry it. If B's same-architecture control is worse
than A's, part of an SSL A–B gap is input interference for this encoder. B′ tests
the same thing without labels.

Use the same subject splits as the corresponding SSL experiment, with early
stopping/tuning restricted to validation subjects. Evaluate on both the common
fixed-brain-size test environment and each arm's own distribution. Repeat across
three seeds for the completed comparison: four arms by two views by three seeds
is 24 supervised fits. Run one seed per arm/view first as a feasibility screen.

Proposed practical criteria, fixed before viewing results:

- Recoverability: reference-contrast median localization error at most the
  lesion radius in voxels (`0.1 * 63 / 2 = 3.15` at resolution 64), with coordinate
  R-squared above the shuffled-label/constant-prediction controls.
- Matching: paired A/B and C/D mean-error differences equivalent within one
  input voxel, assessed with paired subject intervals and shown per training
  seed and modality. If intervals are too wide, matching is unresolved.

These are provisional scientific tolerances, not established clinical cutoffs.
A successful localizer demonstrates recoverability under that procedure. Failure
does not prove absence of image information. In particular, the existing
`eval/diagnostics/supervised_ceiling.py` is an empirical baseline despite its
name and introductory claims. Its GAP-based, multiple-factor regressor is not
automatically the right positive control for a spatial localization question.

### Interpretation rules

| Result | Supported interpretation |
| --- | --- |
| Visibility and supervised recovery match; B learns less location information than A | Evidence that competitor variation impairs learning in this controlled setup |
| That penalty is smaller in C/D | Evidence consistent with greater vulnerability at weaker visibility |
| B has worse input visibility or supervised recovery | Input difficulty remains a confound; do not attribute the SSL gap solely to suppression |
| Target is recoverable spatially but not from the global code | Local information survives; pooling/readout is a plausible limitation |
| A and B are equally poor | Competition is not established; inspect visibility, architecture and optimization |
| A and B are similar with narrow uncertainty | No material competition effect detected under this recipe and training budget |
| B < A and B′ ≈ A | The penalty acts through the loss: competition |
| B < A and B′ ≈ B | Brain-size variation degrades the input for this encoder: interference, not competition through the loss |
| B′ between A and B | Mixed, or B′'s invariance demand costs something; run the batch-stratified control |

## Implementation boundary and deliverables

The dedicated runner implements a matched subject-bank builder, common lesion
placement, explicit contrast scaling, shared normalization, movement/noise audit,
SSL training, supervised spatial controls, held-out probes and paired reporting.
The existing encoder and InfoNCE function are reused. The renderer's new contrast
argument defaults to 1; its original arithmetic path is unchanged at that value.

The runner draws an independent without-replacement minibatch each step, matched
across arms, rather than using the original trainer's shuffled epoch iterator.
The supervised control is a spatial CNN trained on Gaussian coordinate-target
heatmaps, with image-only inference. It uses a fixed step budget and final weights;
there is no supervised early stopping. The nonlinear frozen-feature probe is RBF
kernel ridge, selected alongside linear ridge by three-fold calibration-only CV.
No PCA is applied. The saved training settings record model sizes and code hashes.

Input auditing checks lesion-core pixel identity and lesion on/off response
identity in the response halo; outer-halo background values need not be identical.
The movement-noise audit measures the latter's noise contribution. Its visibility,
contrast-increase and axis-coverage gates must pass before training starts.

Additions for B′ and the pilot (12 October):

- **Bank and audit.** `prepare --arms` renders a subset of arms; the default is all five.
  The audit covers only the bank's arms. For A alone there is nothing to match, so
  its gate reduces to geometry, reproducibility and axis coverage. The audit
  report also gives median movement visibility per arm and view.
- **Training.** Each checkpoint records the held-out loss: InfoNCE with top-1 for
  SSL, the training loss on the probe split for the supervised controls.
  `--save-steps` adds checkpoints.
- **Evaluation.**
  - `evaluate --checkpoint all` scores every saved step. It skips an evaluation
    already done with the same evaluator code, grid and views.
  - `curves` writes `summary/curves.csv`, one figure per arm and a table per run.
  - `summarize` adds the A–B′ comparisons and the `P_input`/`P_loss` decomposition.
- **Ridge grid.** The ridge probe's penalty grid now runs to 1e6. The spatial
  descriptor has 32,768 standardized features for 400 calibration subjects, where
  the original maximum of 100 barely regularizes. This was fixed before any result
  was read.

Save the resolved protocol, subject manifests, normalization constants, pilot
images, visibility table, checkpoints and per-subject predictions. The final
report should show the four-arm recovery plot with all seeds, the paired
visibility/control results and the competition penalties with their uncertainty.

Start with the arm-A pilot (section 0). If it passes, run the visibility audit and
feasibility controls on a full bank, then the A/B/B′ SSL runs. Complete C/D to test
the visibility interaction. A toy or shortened smoke run verifies software only; it
is not evidence for suppression.
