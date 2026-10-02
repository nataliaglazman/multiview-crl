# Local frozen encoder follow-ups: interpretation

Source: `results/encoder_followups_local/20261002_160002/spatial/`. All three audits completed on MPS with PyTorch 2.6.0.

Both supervised controls are now available; see [the supervised analysis](supervised_analysis.md) for their results and the combined interpretation.

Verified 6,048 per-factor test R² values against saved predictions. Models and initial/trained arms share identical input hashes, target arrays, probe split, and evaluation source hashes. Checkpoints were recorded as unchanged. The protocol fits on 300 original validation subjects, tunes on 100, and tests on 400 separate subjects. Both views, saved initial weights, and shuffled-label probes are present.

## Main finding

Sulcal amplitude is highly recoverable from the Conv spatial representation and substantially recoverable from the ResNet backbones, but poorly recovered from the final nine-dimensional content vectors. Lesion localization remains weak: Conv FLAIR contains some usable spatial signal; trained ResNet features give near-baseline results with the tested probes. These are statements about finite-probe recoverability, not proof of identifiability or complete information loss.

## Where sulcal recovery drops

Held-out R² for the renderer's signed sulcal amplitude, using ridge throughout (no per-cell selection between ridge and RBF). GAP is global average pooling. Native backbone grids are 16³ / 2³ / 8³ for Conv / GroupNorm ResNet / stride-8 ResNet. Hidden is the actual global MLP hidden output; final content is the actual nine-dimensional output.

| Model | View | Native backbone | GAP backbone | MLP hidden | Final content |
| --- | --- | --- | --- | --- | --- |
| Conv + MLP | T1 | 0.980 | -0.152 | -0.266 | -0.002 |
| Conv + MLP | FLAIR | 0.978 | 0.001 | 0.001 | -0.004 |
| ResNet GroupNorm | T1 | -0.364 | 0.727 | 0.188 | 0.030 |
| ResNet GroupNorm | FLAIR | 0.783 | 0.689 | 0.105 | 0.027 |
| ResNet stride 8 | T1 | 0.830 | 0.024 | 0.025 | 0.017 |
| ResNet stride 8 | FLAIR | 0.826 | 0.020 | 0.019 | 0.023 |

Conv loses accessibility primarily at spatial averaging: native R² ≈0.98 becomes ≈0 or negative at GAP. Stride-8 ResNet shows a similar drop (≈0.83 to ≈0.02). GroupNorm ResNet retains a strong pooled backbone signal (≈0.69–0.73) but its global MLP/content output suppresses accessibility (≈0.03). These comparisons change feature dimension and fitted probes; a lower score does not establish an irreversible mathematical loss.

The GroupNorm T1 native ridge result is a failure of this fitted readout on held-out data, not evidence that its map has no sulcal signal: the matched native RBF probe reaches 0.725. One test subject (index 298) contributes 40% of native ridge amplitude squared error; its predicted amplitude is 0.559 despite truth 0.0114. The underlying amplitude range is approximately ±0.06. Keep that result visible; investigate conditioning/out-of-distribution features before claiming the native map is worse than GAP.

The original raw sulcal latent is also recovered by native Conv ridge (T1 0.892; FLAIR 0.901). The higher amplitude scores partly reflect predicting the bounded rendered quantity rather than the unsquashed latent. This is not merely a magnitude/sign ambiguity.

## What training added

| Model | View | Initial native amplitude | Trained native amplitude | Shuffled trained native |
| --- | --- | --- | --- | --- |
| Conv + MLP | T1 | 0.957 | 0.980 | -0.008 |
| Conv + MLP | FLAIR | 0.948 | 0.978 | -0.005 |
| ResNet GroupNorm | T1 | 0.799 | -0.364 | -0.000 |
| ResNet GroupNorm | FLAIR | 0.764 | 0.783 | -0.076 |
| ResNet stride 8 | T1 | 0.886 | 0.830 | -0.004 |
| ResNet stride 8 | FLAIR | 0.854 | 0.826 | -0.004 |

Initial Conv features already recover sulcal amplitude extremely well (≈0.95); training adds a modest improvement. This establishes strong observability through random spatial features, not that contrastive learning discovered a uniquely identifiable factor. Stride-8 native sulcal recovery is lower after training. Sulcal signal also appears in spatial channels labelled style, so the channel names alone do not establish separation of factors.

## Lesion recovery

Centroid columns average the three physical-coordinate R² values. Raw lesion columns average the original three generator-latent R² values. They are different targets because placement depends on anatomy and rasterization.

| Model | View | Native centroid | GAP centroid | Final content centroid | Native raw lesion |
| --- | --- | --- | --- | --- | --- |
| Conv + MLP | T1 | 0.100 | 0.030 | 0.004 | 0.060 |
| Conv + MLP | FLAIR | 0.300 | 0.059 | 0.010 | 0.202 |
| ResNet GroupNorm | T1 | 0.008 | 0.008 | -0.012 | -0.014 |
| ResNet GroupNorm | FLAIR | 0.007 | 0.009 | -0.012 | -0.014 |
| ResNet stride 8 | T1 | 0.006 | 0.009 | -0.008 | -0.015 |
| ResNet stride 8 | FLAIR | 0.006 | 0.008 | -0.012 | -0.015 |

Conv FLAIR native centroid R² is 0.300: x/y/z = 0.457/0.353/0.090. Original lesion-latent x/y/z = 0.282/0.309/0.013. Recovery is partial and weakest along z. Its native ridge median physical error is 7.36 voxels, and 10.75% of subjects are within one 3.15-voxel lesion radius. T1 is weaker (centroid mean R² 0.100, median error 9.41 voxels). These are not yet accurate lesion localizers.

The trained ResNets remain near baseline even when spatial position is retained; increasing final map resolution alone did not rescue lesions under this recipe. For stride-8 FLAIR, the same spatial projected ridge probe changes from initial centroid R² 0.357 to trained −0.013 (raw lesion 0.210 to −0.015). This suggests training made lesion location much less accessible to that readout. Spatial projected features apply the MLP independently per bin; they are diagnostic representations, not the globally pooled vector used during training.

Conv training effects depend on the readout: its native FLAIR centroid ridge improves from 0.022 to 0.300, but its 2³ backbone centroid ridge changes from 0.336 to 0.291. Do not claim all lesion representations improved. Centroid prediction can also exploit anatomy-dependent placement; a direct localization control remains useful.

## Broad anatomy still present before the ResNet readout

T1 ridge test R², using the same global backbone and final content stages:

| Model | Factor | GAP backbone | Final content |
| --- | --- | --- | --- |
| Conv + MLP | brain_size | 0.925 | 0.912 |
| Conv + MLP | ventricle_size | 0.842 | 0.819 |
| Conv + MLP | cortical_thickness | 0.868 | 0.851 |
| Conv + MLP | temporal_atrophy | 0.792 | 0.774 |
| Conv + MLP | lr_asymmetry | 0.885 | 0.881 |
| ResNet GroupNorm | brain_size | 0.780 | 0.011 |
| ResNet GroupNorm | ventricle_size | 0.051 | -0.009 |
| ResNet GroupNorm | cortical_thickness | 0.651 | 0.108 |
| ResNet GroupNorm | temporal_atrophy | 0.805 | 0.490 |
| ResNet GroupNorm | lr_asymmetry | 0.927 | 0.117 |
| ResNet stride 8 | brain_size | 0.662 | 0.124 |
| ResNet stride 8 | ventricle_size | -0.003 | 0.006 |
| ResNet stride 8 | cortical_thickness | 0.299 | 0.121 |
| ResNet stride 8 | temporal_atrophy | 0.810 | 0.542 |
| ResNet stride 8 | lr_asymmetry | 0.961 | 0.166 |

The poor final ResNet content scores do not imply that its entire backbone failed. For example, asymmetry is recovered at 0.927/0.961 before the GroupNorm/stride-8 heads but only 0.117/0.166 afterward. Conv final content retains its strong broad-anatomy scores.

## Additional magnitude diagnostic (post hoc, no refitting)

Each original probe predicts magnitude separately from signed amplitude. Failure of that direct magnitude readout does not prove magnitude is absent if signed amplitude can be predicted. Taking the absolute value of saved native Conv ridge amplitude predictions gives magnitude R² 0.907 (T1) and 0.901 (FLAIR), compared with 0.603/0.572 for the separately fitted magnitude ridge. These derived scores use the already fitted signed predictor, with no test-label-based tuning; treat this as an exploratory diagnostic.

## Next decisions supported by these results

1. For sulcal decoding, start with the already successful frozen Conv spatial features and ridge probe. A usable endpoint exists without encoder retraining. To change the representation itself, test a spatially aware content readout or explicit sulcal objective; merely widening the final global vector does not address the observed pooling failure.
2. The supervised controls have now completed; consult [their analysis](supervised_analysis.md). They strongly improve lesion localization on both views. To isolate the remaining bottleneck, compare supervised spatial heads on frozen versus fine-tuned original encoders with a matched labeled cohort.
3. If the supervised lesion control succeeds, test early/high-resolution encoder features and a lesion-aware objective before another generic stride comparison. If it fails, inspect optimization, rendering contrast and target definition before concluding non-observability.
4. Confirm selected findings with additional training seeds and a fresh evaluation cohort. These are exploratory comparisons across many readouts, with only one model seed. Conditional test-set uncertainty would not replace training-seed uncertainty.

Native backbone feature counts are 262,144 (Conv), 4,096 (GroupNorm ResNet), and 262,144 (stride-8 ResNet); pooled backbone counts are 64/512/512, and final content has 9 coordinates. Equal native dimensionality for Conv and stride-8 does not equalize their spatial layout, normalization, or other architecture choices. The two ResNet variants also differ in normalization, so these runs do not isolate stride as a causal variable.

Reproduce this summary from repository root: `python reports/encoder_followups_local_20261002_160002/analyze.py`. Only report JSON and saved prediction NPZ files are read; original artifacts and encoder weights are never modified.
