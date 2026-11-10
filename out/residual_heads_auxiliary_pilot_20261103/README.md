# Corrected residual-head pilot, 3 November 2026

The revised model keeps its independent image and residual encoders. Its primary
objective is the original image-content InfoNCE. The auxiliary is weighted by 0.1:

```text
loss = image_InfoNCE + 0.1 * (image_alignment + residual_alignment + joint_MMD)
```

The alignment terms differentiate their current per-view batch mean and variance.
This removes the artificial common-scale contraction gradient away from epsilon;
it does not rule out constant solutions. The joint projector continues to use
training population statistics, fixed at evaluation. Finite MMD does not establish
Yao's population identifiability guarantee.

## Experiment

`run.py` runs two matched arms: image-only InfoNCE and the corrected auxiliary
configuration. Both begin with identical image parameters, use AdamW at 1e-4 and
the same sequence of uniform batches, and train for 500 steps at batch size 16.
Inputs are full 64³ T1/FLAIR pairs. The image code has 14 content coordinates; the
independent residual code has 8; the joint sigmoid projector has 10 outputs.

The existing bank has 2,000 training subjects and 400 validation subjects. It was
rendered with the clean-content burden generator: one WM-interior lesion, radius
0.1, styled intensity, T1 lesion value 0, identifiable ventricles, sulcal atrophy,
temporal sigma 0.18, and data seed 42. Cached images are float16, converted to
float32 for training. Training subjects 0 and 1445 and validation subject 0 were
checked against the current online generator: both images match exactly after
float16 conversion, and content latents match exactly.

PCA and initial tail/code statistics use only the first 300 training subjects.
Latent labels and rendered lesion volumes do not enter optimization. At steps
0, 250 and 500, fixed representations are scored on validation subjects using
five outer subject folds, StandardScaler and RidgeCV. Both scaler and regression
are fitted only on the outer training folds. These are repeatedly monitored
validation scores, not an untouched final test.

This local MPS diagnostic differs from the runnable NVIDIA recipe, which uses
1,000 steps, batch size 32, online float32 images and shuffled training epochs.
The local arms share all these pilot choices with each other. Several objective
settings change together relative to the failed run, so this comparison does not
isolate normalization as the cause of any change in recovery.

## Results

**The pilot did not pass the burden-recovery and anatomy-retention checks.**

The implementation repair is in place, and both arms completed 500 steps without nonfinite losses. The held-out linear probes show no convincing added burden recovery in both views, and several anatomy scores fall below the matched control. Do not extend it to 20,000 steps on this evidence.

### Image-block anatomy recovery at step 500

Held-out ridge R²; deltas subtract the matched image-only endpoint. No uncertainty interval is estimated from this one seed.

| Factor | T1 control | T1 auxiliary | Δ | FLAIR control | FLAIR auxiliary | Δ |
|---|---:|---:|---:|---:|---:|---:|
| brain_size | 0.900 | 0.898 | -0.002 | 0.895 | 0.885 | -0.010 |
| ventricle_size | 0.786 | 0.764 | -0.022 | 0.632 | 0.373 | -0.259 |
| cortical_thickness | 0.833 | 0.827 | -0.006 | 0.855 | 0.842 | -0.013 |
| temporal_atrophy | 0.105 | 0.250 | +0.145 | 0.115 | 0.262 | +0.147 |
| lr_asymmetry | 0.177 | 0.223 | +0.046 | 0.054 | 0.099 | +0.045 |
| sulcal_widening | 0.538 | 0.307 | -0.230 | 0.735 | 0.472 | -0.263 |
| Six-factor mean | 0.556 | 0.545 | -0.011 | 0.548 | 0.489 | -0.059 |

The auxiliary trades recovery between factors: temporal atrophy improves, while sulcal widening and FLAIR ventricle recovery are appreciably lower. The aggregate mean alone hides that tradeoff.

Concatenation does not rescue the main losses: combined-code FLAIR ventricle R² is
0.410 versus the control's 0.632. Combined sulcal R² is 0.323 in T1 and 0.455 in
FLAIR, versus 0.538 and 0.735 in the control. The residual block primarily recovers
brain size (0.568 in T1 and 0.508 in FLAIR), rather than supplying the desired
burden signal.

### Lesion recovery at step 500

| Code | T1 burden latent | T1 rendered volume | FLAIR burden latent | FLAIR rendered volume |
|---|---:|---:|---:|---:|
| Image-only control | -0.013 | -0.009 | -0.051 | -0.047 |
| Auxiliary image | -0.010 | -0.001 | -0.006 | -0.006 |
| Auxiliary residual | -0.001 | -0.002 | 0.013 | 0.012 |
| Auxiliary combined | -0.011 | -0.035 | -0.020 | -0.008 |

Residual and combined step-0 burden scores were also near zero (all within −0.020 to +0.003). The small positive FLAIR residual score at step 500 is not corroborated in T1 or in the combined code. This does not satisfy the added-lesion-information criterion.

### Scale and covariance diagnostics

| View / block | Initial raw variance | Step-500 raw variance | Initial participation ratio | Step-500 participation ratio |
|---|---:|---:|---:|---:|
| t1 / image | 0.000083 | 0.000600 | 2.012 | 1.084 |
| t1 / residual | 0.000755 | 0.043397 | 4.931 | 1.013 |
| flair / image | 0.000064 | 0.000402 | 2.597 | 1.051 |
| flair / residual | 0.000973 | 0.073957 | 4.160 | 1.008 |

Residual variance grows substantially while its covariance concentrates in one direction. This is different from all codes shrinking toward a constant. Participation ratio is not intrinsic dimension: the image block recovers several factors despite its low ratio, so probes remain essential. The auxiliary projector’s within-view participation ratio is 1.011 at step 500.

### Weighted gradient snapshots

| Step | Image primary RMS | Auxiliary image RMS | Auxiliary / primary | Residual alignment RMS | Residual uniformity RMS |
|---|---:|---:|---:|---:|---:|
| 100 | 0.03719 | 0.00114 | 3.1% | 0.00596 | 0.00070 |
| 200 | 0.03176 | 0.00072 | 2.3% | 0.00593 | 0.00078 |
| 300 | 0.02583 | 0.00031 | 1.2% | 0.00512 | 0.00054 |
| 400 | 0.03016 | 0.00056 | 1.9% | 0.00329 | 0.00053 |
| 500 | 0.08871 | 0.00129 | 1.5% | 0.01160 | 0.00150 |

These are snapshots on individual training batches, not population bounds. The primary image gradient dominates the auxiliary at these checkpoints, yet anatomy retention is mixed. A small coefficient or gradient ratio does not ensure preservation of every factor.

### Decision

Keep this as a completed negative pilot. The normalization-gradient regression is fixed, but the lesion-learning problem remains unresolved. Retain the matched checkpoints and compare further residual objectives under a protected, measured image representation before another long run. This pilot does not prove that longer training can never help, and does not isolate which objective change caused each difference. No longer run was launched.

## Artifacts and reproduction

From the repository root, using a Python environment with the training dependencies:

```bash
python out/residual_heads_auxiliary_pilot_20261103/run.py \
  --out out/residual_heads_auxiliary_pilot_20261103/repeat --device mps
```

The existing bank is required; its renderer is
`out/burden_feasibility_20261102/bank.py`. The pilot writes settings, fit provenance,
per-step losses and weighted gradient RMS, held-out recovery scores, and both
arms' initial, step-250 and step-500 weights under `results/`. Existing output
directories are refused.

The ordinary trainer's new short recipe is
`experiments/encoder/conv_mlp_burden_residual_heads_auxiliary.yaml`. It keeps the
legacy recipe intact and has generated launchers for local, RunAI and both SLURM
backends. `training/ENCODER_RESIDUAL_HEADS.md` documents its settings and checks.

For a matched image-only control through that same recipe, set
`--residual-head-auxiliary-weight 0` and give it a distinct `--model-id`. The
residual module's parameters and running buffers stay frozen in this control.
Compare the image block, since its appended residual code is untrained.

## Verification

The repair has 34 passing residual-head tests. An additional 110 existing tests
passed across vector prediction, lesion branches, normative bundles, pointwise
training, checkpoint restoration and checkpoint style scoring. All four new
launchers pass shell syntax and generator-equality checks; all recipe/backend
combinations pass parser validation. A broad existing script-current check still
has 69 unrelated stale generated-script comparisons; those scripts were left
unchanged.

Both final checkpoints restore strictly through the scoring model builder and
match their saved step-500 snapshots exactly. The step-250 checkpoints also pass
an evaluation-batch-companion check, with unchanged buffers and output agreement
within numerical tolerance.

The NVIDIA process is outside this workspace. Its old checkpoint was not copied
or its process stopped by this local task. The ordinary trainer overwrites
`model.pt` at evaluation, so retaining a particular old step requires copying the
checkpoint on that machine while that step is still available.
