# Supervised heatmap controls: both views completed

Both targets are learnable from these synthetic images under explicit supervision. Lesion localization is excellent on FLAIR and strong on most T1 subjects, with a failure tail on T1. Sulcal amplitude is recovered very well in both views. This substantially narrows the interpretation of the earlier frozen-probe failures.

![Held-out localization and sulcal recovery](supervised_controls.png)

## Protocol and verification

Source: `results/encoder_followups_local/20261002_160002/supervised/{t1,flair}`. Both reports are complete: 2,000 updates, batch 8, MPS/PyTorch 2.6.0, seed 42, a fresh 310,634-parameter network for each view, final-step checkpoint selection. Training uses 2,000 labeled subjects; validation and independent test each have 400 subjects. The inputs are full single-view images; lesion masks are training targets, not model inputs. Coordinates come from the expected position of a predicted spatial softmax heatmap; the other targets come from a separate spatial regression head.

All saved test R², RMSE, training-mean baseline R², centroid error metrics, and sulcal sign accuracies were independently recomputed from predictions. Test targets are identical to the frozen audits, validation/test input hashes match, and evaluation source hashes agree. No models were retrained and original outputs were not modified during this analysis.

## Lesion localization compared with the frozen Conv probe

| View | Method | Mean centroid R² | Median error (vox) | Mean error (vox) | Within 3.15 vox |
| --- | --- | --- | --- | --- | --- |
| T1 | Frozen Conv native ridge | 0.1000 | 9.409 | 9.843 | 4.50% |
| T1 | Supervised heatmap | 0.9476 | 0.284 | 0.971 | 93.75% |
| FLAIR | Frozen Conv native ridge | 0.3001 | 7.359 | 8.279 | 10.75% |
| FLAIR | Supervised heatmap | 0.9993 | 0.217 | 0.249 | 100.00% |

FLAIR has median error 0.217 voxels; 398/400 subjects are within one voxel and all 400 are within one lesion radius. The maximum error is 2.378 voxels. T1 has median error 0.284 voxels and 361/400 within one voxel, but 25/400 fall outside one lesion radius. Its 95th-percentile error is 5.558 voxels and maximum is 18.416 voxels. Report this tail rather than relying on the median alone. These are physical-centroid scores, not Dice/IoU segmentation scores; this audit has not measured predicted mask overlap or heatmap calibration.

## Original latent factors and physical targets

| Target | T1 test R² | FLAIR test R² |
| --- | --- | --- |
| lesion_x | 0.8323 | 0.9164 |
| lesion_y | 0.7813 | 0.8874 |
| lesion_z | 0.5985 | 0.6856 |
| sulcal_widening | 0.9152 | 0.9382 |
| sulcal_amplitude | 0.9782 | 0.9831 |
| centroid_x | 0.9390 | 0.9995 |
| centroid_y | 0.9466 | 0.9994 |
| centroid_z | 0.9572 | 0.9991 |
| sulcal_magnitude | 0.9049 | 0.9225 |

Physical localization and recovery of original lesion generator coordinates are different endpoints. The original lesion z latent remains harder (T1 0.598; FLAIR 0.686), even though physical z centroid R² is 0.957/0.999. This is consistent with anatomy-dependent, discretized placement and/or a harder regression task. These results do not establish an irreducible ceiling on latent recovery.

Signed sulcal amplitude R² is 0.978 (T1) and 0.983 (FLAIR); original sulcal latent R² is 0.915/0.938. Sulcal sign accuracy is 97.75%/97.00%. Magnitude, obtained by taking the absolute predicted signed amplitude, has R² 0.905/0.923. Thus success includes both signed structure and magnitude, rather than magnitude alone.

## Learning trajectory

| View | Step | Validation median error (vox) | Within one radius | Signed amplitude R² |
| --- | --- | --- | --- | --- |
| T1 | 500 | 0.406 | 90.75% | 0.932 |
| T1 | 1000 | 0.332 | 91.50% | 0.966 |
| T1 | 1500 | 0.320 | 92.00% | 0.982 |
| T1 | 2000 | 0.295 | 93.00% | 0.983 |
| FLAIR | 500 | 0.409 | 99.75% | 0.929 |
| FLAIR | 1000 | 0.269 | 99.75% | 0.971 |
| FLAIR | 1500 | 0.230 | 100.00% | 0.980 |
| FLAIR | 2000 | 0.214 | 100.00% | 0.980 |

The validation trajectories generally improve through the fixed final step, with test performance broadly consistent with final validation. T1's final logged minibatch loss is higher than the preceding snapshot, but its validation localization and raw sulcal latent score improve; one minibatch loss is not evidence that training diverged. The checkpoints were selected by prespecified final step, not by test performance.

## What the combined experiments establish

1. A learner can infer physical lesion location and sulcal factors from these images on held-out subjects. The earlier low probe scores cannot be explained simply by the targets being invisible in this synthetic dataset.
2. Sulcal decoding already works well from frozen Conv spatial features. Global averaging and the learned global content readout are important practical bottlenecks in the audited encoders.
3. Lesion localization becomes much stronger with a high-resolution, target-supervised network. The contrastive representation/probe pipeline is limiting practical recovery, but this comparison does not isolate which component is responsible.
4. This is supervised recoverability evidence, not a demonstration that the contrastive encoders identify the factors, disentangle their coordinates, or recover a causal model. Strong target decoding under supervision does not establish uniqueness of an unsupervised latent representation.

## Comparison limits and next experiment

The controls change architecture, spatial resolution, loss, probe class, and label budget together: 2,000 supervised training subjects versus 300 labeled fitting subjects for the frozen ridge/RBF probes. They establish observability under this learning protocol, not an architecture ranking or proof that the frozen features contain no lesion information. No shuffled-label supervised training control was run. Results use one training seed and one synthetic test cohort; they do not establish performance on real MRI.

The most informative next experiment is to use a spatial localization head on each original encoder with a matched labeled training cohort and comparable decoder capacity: first freeze the encoder, then allow fine-tuning. A successful frozen head would implicate probe/readout limitations; a gain only after fine-tuning would show that adapting the representation helps under that decoder and training budget. For a practical supervised model, the successful high-resolution heatmap branch is the strongest starting point for lesions, with a spatial sulcal head alongside it. If retaining an unsupervised objective is essential, use these controls as benchmarks and test a spatial learning objective separately. Inspect the 25 T1 localization failures and confirm chosen changes on another seed and fresh test cohort.

Reproduce: `python reports/encoder_followups_local_20261002_160002/analyze_controls.py`.
