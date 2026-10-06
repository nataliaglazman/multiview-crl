# Decision after reviewing the lesion branch

Reviewed 2026-10-06. This is an experiment decision, not evidence that the proposed
fix already works. Training code and existing runs were left unchanged.

Implementation added subsequently: `training/local_scalar_experiment.py`.
See `training/LOCAL_SCALAR_EXPERIMENT.md` for runnable commands and the exact
objectives, including the additional matched Barlow Twins arm.

## Decision

Use different scalar readouts for lesion position and signed sulcal amplitude.
Start from a frozen, trained global-only encoder and test whether the new local
readouts explain spatial variation that its global code does not explain.
Keep decorrelation as a controlled ablation; do not treat a larger penalty or
HSIC as the default next fix.

The immediate priorities are:

1. Repair the interpretation and isolation of the lesion experiment: evaluate
   physical location as well as raw controls, and distinguish updating a local
   head from updating the shared encoder.
2. Give sulcal amplitude an explicitly spatially ordered, signed readout.
3. Test a local prediction objective through these scalar bottlenecks, against
   the existing branch InfoNCE objective, using identical frozen features.

## Evidence available

- `LESION_BRANCH_HANDOFF.md` reports successful lesion learning in toys without
  an easy global factor and failure when that factor is present. Those toy
  experiments were not rerun for this review.
- In `conv_mlp_s42_mps_t2000_lesionkp4_layernorm_v4`, the step-1000 branch-only
  brain-size R² is 0.882 / 0.864 (T1 / FLAIR). Lesion x/y/z are
  0.047 / -0.030 / -0.070 for T1 and -0.004 / -0.024 / -0.033 for FLAIR.
  Its lesion InfoNCE is 1.6195. Thus decreasing this objective is not evidence
  that the heads found lesions.
- At review time the fixed-intensity decorrelation log ends at step 400. The
  styled-intensity logs, with and without decorrelation, end at step 200. All
  three have only step-0 evaluations. Their trained effectiveness is unknown.
  These log endpoints do not establish whether the processes are still running.
- In the full scalar experiment pasted by the user, the free oracle achieves
  sulcal-amplitude one-scalar R² 0.968; geometric oracle achieves 0.001.
  Both oracle readouts achieve mean centroid R² about 0.38; geometric SSL is
  near zero. These are user-provided results, not a locally rerun experiment.
  The free readout can access ordered spatial locations; the geometric
  amplitude aggregation has no explicit positional input. This supports testing
  a signed spatial readout, although architecture, optimization and parameter
  count have not yet been disentangled.

## Corrections to the handoff's interpretation

**The datasets called "real-data evidence" in the handoff are synthetic MRI
experiments using the actual training pipeline, not acquired patient scans.**

**GAP's position invariance is conditional.** It holds for a translated feature
pattern with translation-equivariant processing and a pooling domain that does
not introduce boundary changes. Anatomy interactions, padding and positional
features can violate those assumptions. The useful finding here is the measured
weakness of the existing readout, not a universal impossibility result for GAP.

**The brain frame is not the renderer's latent coordinate system.**
`_sphere_in_white_matter` uses sequential conditional quantiles over eroded WM
voxels. `KeypointPool3d` uses the input support's centroid and per-axis spread.
These transformations are different. A good physical lesion detector can fail
a linear probe for the raw controls. Quantization also makes exact inversion of
continuous controls impossible at finite resolution.

**The current branch report is a joint probe, not scalar identifiability.**
Four heads output 12 coordinates. `lesion_branch_report` fits each raw target
from all 12, rather than measuring a designated coordinate's accuracy.

**Detaching the content vector does not freeze the shared representation.**
`lesion_decorrelation` stops its gradient through the global-readout path, but
the coordinate path still updates the backbone. A CPU backward check confirmed
zero gradient on the global readout and nonzero gradients on the backbone and
keypoint logits. The existing test checks the readout, not backbone isolation.

**Pooling modalities can confound the decorrelation statistic.** The function
centres and scales T1 and FLAIR together. A constructed example with zero
within-view covariance but shared modality offsets gives a pooled penalty of
99.5707, versus about 2.2e-14 when averaged within views. This establishes a
possible confound, not its measured contribution in the training runs.

**Independent raw factors do not imply independent geometric outputs.**
The anatomy-dependent WM placement means physical lesion position, its allowable
range and brain-relative coordinates may depend on anatomy even when the raw
sampling controls are independent. Linear decorrelation is weaker than
independence, but either penalty can discourage legitimate relationships.
Stronger HSIC therefore needs a target-specific justification.

## Controlled experiment to implement

### Readouts and initialization

- Use the same pretrained global-only checkpoint for every readout arm. Freeze
  the encoder parameters and buffers and the global head during the first stage.
  The previous oracle results make a frozen-readout test informative before more
  end-to-end training.
- Lesion readout: retain the existing K=4 keypoint heads for the first comparison,
  with no added heads or simultaneous normalization change. Export brain-frame
  coordinates and their conversion back to image coordinates. Evaluate each
  head separately, with head selection made on validation subjects only.
- Sulcal readout: an ordered spatial projection to one signed scalar. A small
  channel projection followed by a linear map over the flattened grid is a
  transparent first version. Signed spatial weights can distinguish corrugation
  phase; no softmax/nonnegative restriction on those weights. Include the
  previous free oracle as a positive control rather than presuming the simpler
  linear head is sufficient.
- Do not hard-code the synthetic sine pattern into the label-free arm. That is
  an optional renderer-informed positive control and must be labelled as such.

### Learning signal

Use a training-only decoder to make the local scalar outputs explain spatial
detail, instead of relying only on subject discrimination.

1. Fit a predictor `Dg(g(X), p, view)` from the frozen global code and position
   to signed spatial features `F(X, p)` on training subjects, then freeze it.
   Use train-only feature scaling. Measure held-out prediction too, since a
   predictor that memorizes training subjects could remove the training signal.
2. Form the fixed residual `r(X, p) = F(X, p) - Dg(g(X), p, view)`.
3. Predict this residual through the local scalars. Restrict the lesion decoder
   to localized contributions centred on the predicted keypoints. Give the
   amplitude scalar a signed spatial template. Anatomy conditioning may modulate
   these contributions, but there must be no unrestricted global-only residual
   path or dense feature skip that can bypass the local scalars.
4. Add known-transform coordinate equivariance and mild appearance consistency.
   These are two augmentations of one image, not repeated acquisitions. Avoid
   augmentations that erase the lesion. Signed amplitude is not automatically
   invariant to reflections: the renderer's sine-product pattern changes phase.
5. Use scale-normalized spatial reconstruction losses and report error by scale
   and location. A whole-volume average can still let tiny lesions be ignored.
   Keep signed descriptors; high-pass energy alone can discard phase/sign.

This is a hypothesis: the residual can contain anatomy errors, style or noise,
and an unsupervised keypoint can track another structure. A frozen predictor,
restricted local decoder and intervention evaluation make that failure visible;
they do not prove identifiability. A decoder used only during training leaves
an encoder-only inference model.

### Ablations and stopping decisions

On identical frozen features and train/validation/test subjects, compare:

| Arm | Purpose |
|---|---|
| Initial readouts | Measure architectural recovery before learning |
| Oracle readouts | Synthetic labelled capacity control; no unsupervised claim |
| Local InfoNCE | Reproduce subject-discrimination failure on frozen features |
| Local InfoNCE + within-view decorrelation | Test the handoff's proposal without modality mixing |
| Scalar-bottleneck residual prediction + equivariance | Test an explicit reason to retain local detail |

If testing decorrelation, average per-entry squared correlations within each
view. Its numerical weight then has a different meaning from the old sum over
all entries: record that definition and do not compare lambda=1 as though the
loss were unchanged. Prefer a small prespecified weight sweep over escalating
to nonlinear independence. Compare head variance and recovery to catch collapse.

Use styled lesions as a visibility control and retain fixed-intensity held-out
evaluation. Styled changes the renderer's task distribution; it is not evidence
of improved recovery at the original low contrast. Stratify T1 recovery by actual
image contrast. A FLAIR-only readout experiment can separate a weak-view alignment
problem from a readout-learning problem without needing repeated acquisitions.

Advance to encoder fine-tuning only when a frozen readout beats initialization
and shuffled controls in held-out intervention tracking. If the oracle fails,
first revise the readout/feature scale or verify image observability. If the
oracle succeeds and SSL fails, change the learning signal rather than merely
increasing encoder size. Confirm a promising result with seeds 42, 142 and 242.

### Required evaluation

- Lesions: raw controls, actual physical centroid, brain-frame centroid; native
  coordinate error in voxels; validation-selected individual head; joint probe
  reported separately; lesion-only movement skill with subject bootstrap CIs.
- Sulcal: raw factor and signed physical amplitude separately; one-scalar R²;
  signed finite-intervention response with acquisition draws fixed.
- Both: initial/shuffled controls, response to anatomy-only interventions,
  per-modality and contrast-stratified scores. Lower anatomical probe R² alone
  is not success: it can result from collapse or lost useful information.
- Match sulcal and lesion controls to the actual renderer: anatomy-only
  interventions can physically move WM-constrained lesions even at fixed raw
  lesion controls. Use the rendered endpoint truth, not a forced diagonal
  response matrix in geometric coordinates.

Before reusing the scalar experiment on a brain-frame branch checkpoint, update
`training/scalar_readout_data.py:Extractor`: `_global_code(h)` now needs the input
image when that branch is enabled. Existing global-only checkpoints are unaffected.

## Verification and references

All 15 existing `tests/test_lesion_branch.py` tests passed on CPU during review.
Two additional temporary checks verified the gradient path and modality-offset
counterexample above. No new full training run was launched.

Feature suppression is consistent with, but not uniquely established by, these
results: [Chen, Luo & Li, NeurIPS 2021](https://papers.nips.cc/paper/2021/file/628f16b29939d1b060af49f66ae0f7f8-Paper.pdf).
Unsupervised landmark learning through a constrained generative bottleneck has
precedent in [Jakab et al., NeurIPS 2018](https://robots.ox.ac.uk/~vgg/research/unsupervised_landmarks/).
The residual objective above is a proposed adaptation, not a result established
by either paper for this renderer or for clinical lesions.
