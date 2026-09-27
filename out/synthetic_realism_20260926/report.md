# Synthetic T1/FLAIR realism audit — 26 September 2026

The renderer captures several qualitative MRI contrast relationships. It is a controlled anatomical phantom with hand-selected intensities, rather than a physically calibrated MRI simulator. Its asymmetry of factor visibility is partly plausible and partly a consequence of implementation choices. The latter should be checked before interpreting differences between trained encoders as properties of real MRI.

## Scope and evidence

Inspected the current renderer and generated 64 test subjects at 64³ for each of two saved local configurations:

- `settings.json`: noncausal, clean content, `wm_interior`, fixed-reference normalization, style scale 1; model ID `dummy_infonce_wm_interior_non_res_non_causal`.
- `synthetic-clean-content-causal-sp-s-1/settings.json`: causal random graph, identifiable ventricle, clean content, fixed-reference normalization, style scale 1. This file does not set lesion placement, so the current renderer uses `legacy`.

These are locally reproduced inputs, not reconstructions or a verification of the remote training datasets. The configurations differ in multiple respects, so their comparison is not an isolated placement ablation. No model was loaded or trained; the renderer and training configuration files were not edited.

[Reproducible script](/Users/nataliaglazman/Desktop/PhD/projects/multiview-crl/out/synthetic_realism_20260926/analyze.py), [current configuration measurements](/Users/nataliaglazman/Desktop/PhD/projects/multiview-crl/out/synthetic_realism_20260926/wm_interior_noncausal.json), [older configuration measurements](/Users/nataliaglazman/Desktop/PhD/projects/multiview-crl/out/synthetic_realism_20260926/vqvae_identifiable_ventricle.json). JSON files retain settings, effective flags, normalization constants and subject-level values. `source_hashes.json` records the renderer and wrapper source hashes.

## Tissue contrast

The nominal lookup table, before gain, bias, noise and blur, is:

| Tissue | T1 | FLAIR |
|---|---:|---:|
| CSF | 0.1 | 0.1 |
| White matter | 0.8 | 0.4 |
| Gray matter | 0.5 | 0.8 |
| Lesion | 0.4 | 1.0 |
| Special fissure label, identifiable-ventricle mode only | 0.3 | 0.3 |

These are arbitrary intensity units, not relaxation parameters. White matter brighter than gray matter on T1, CSF suppression on T2-FLAIR, and white-matter lesions brighter on FLAIR are qualitatively appropriate. Conventional FLAIR combines CSF suppression with T2 weighting; see the original [Hajnal et al. study](https://pubmed.ncbi.nlm.nih.gov/1430427/).

The contrast magnitudes are not calibrated. At neutral style, the WM–CSF difference is 0.7 on T1 and 0.3 on FLAIR. Thus the nominal FLAIR difference is only 43% of the T1 difference; this ratio is imposed by the renderer, not established as a clinical constant. In the current `wm_interior` sample, median measured interior WM–CSF differences were 0.699 and 0.264. Only 45/64 subjects had usable CSF after full 3×3×3 erosion and exclusion of the lesion neighborhood; these interior measurements deliberately avoid partial-volume boundaries and do not measure ventricular detection accuracy.

The nominal FLAIR GM/WM ratio is 2.0. This is a particularly strong separation. One experimental clinical-reference TSE-FLAIR protocol had a GM/WM ratio of 1.14, with markedly different ratios under other acquisitions. This is evidence that the synthetic ratio needs protocol-specific calibration, not a universal replacement value: [Demir et al., 2022](https://pubmed.ncbi.nlm.nih.gov/34985151/).

## Lesions: plausible asymmetry, artificial dependence on style

For vascular white-matter hyperintensities, conspicuity on FLAIR with weak or absent T1 hypointensity is plausible. T1 appearance depends on tissue damage and acquisition. The appropriate target depends on whether these represent vascular WMH, MS lesions, or an abstract lesion; the generator currently specifies no disease-specific model. [STRIVE definitions](https://pmc.ncbi.nlm.nih.gov/articles/PMC3714437/) describe vascular WMH as T2 hyperintense and T1 isointense or hypointense, without a CSF-like cavity.

There is a separate implementation issue: `render_modality` applies gain and bias to normal tissue, then replaces lesion voxels with a fixed intensity. Lesions still receive the subsequent smooth multiplicative bias field, noise and blur, but bypass the sampled global gain and additive bias.

At style scale 1, nominal T1 WM spans 0.46–1.14 while a pure lesion stays at 0.4. Its WM contrast can therefore span 0.06–0.74. FLAIR WM spans 0.18–0.62 while lesion intensity stays at 1.0, leaving contrast 0.38–0.82. These ranges precede the magnitude operation, bias field and blur. Consequently the low-contrast tail is much more severe on T1. A scanner-wide gain should also scale lesion signal; different lesion biology should be modeled explicitly rather than generated accidentally by this ordering.

Measured mean absolute lesion-removal responses within the true lesion mask were:

| Local configuration | T1 median [10th, 90th percentiles] | FLAIR median [10th, 90th percentiles] |
|---|---:|---:|
| Current `wm_interior` | 0.261 [0.114, 0.396] | 0.359 [0.275, 0.469] |
| Older identifiable-ventricle / legacy placement | 0.125 [0.058, 0.285] | 0.457 [0.330, 0.561] |

Responses are in raw units, computed with anatomy and noise fixed while removing the lesion. They measure injected image signal, **not** SNR, CNR, visibility to an observer, or a bound on a learned detector. Blur is included. Fixed-reference normalization divides these differences by the corresponding stored scale. The old and new configurations differ in anatomy and causal sampling as well as placement; the table cannot isolate which change accounts for their difference.

The default lesion diameter at 64³ is approximately 6.3 voxels. The 3³ averaging filter blurs a substantial fraction of this small structure. The current brain contains only roughly 40 voxels across at nominal geometry, so anatomy and lesions are very coarsely sampled.

## Placement and anatomy

Current `wm_interior` placement kept all 64 lesions entirely within final WM labels. With the saved older configuration, 58/64 lesions partly overlapped non-WM labels; the median outside-WM fraction was 50%. The older algorithm samples inside the geometric WM envelope without excluding ventricles or fissure labels. This can create an anomalous bright spot in CSF on FLAIR, while T1 lesion contributions can change sign across affected tissue classes and partly cancel under blur. New runs should set `synthetic_lesion_placement: wm_interior` explicitly if the intended object is a noncavitating WM lesion.

Even the corrected placement remains an idealized single sphere of fixed radius. The brain is made from deformed spherical shells, ventricles from split spherical cavities, and the factor named sulcal widening changes a sinusoidal corrugation of tissue boundaries. It does not directly widen anatomically realistic CSF-filled sulci. In identifiable-ventricle mode, the fissure receives a separate fixed intensity specifically to distinguish it from ventricular CSF. This is useful for a controlled identifiability experiment but is not a model of fluid relaxation physics.

## Acquisition and normalization

The Rician-style magnitude construction is a defensible basic ingredient; see [Gudbjartsson and Patz, 1995](https://pmc.ncbi.nlm.nih.gov/articles/PMC2254141/). However, the same box filter is applied after magnitude noise in both views, spatially correlating the noise. There are no tissue T1/T2/proton-density maps, TR/TE/TI settings, acquisition-dependent point-spread functions, or explicit physical voxel sizes. Consequently a numerical comparison with a clinical noise level or lesion diameter in millimeters is not yet possible.

Perfect registration and a shared tissue map are useful experimental controls, but simplify real paired acquisitions. Fixed-reference centering/scaling is also a legitimate model preprocessing choice; setting all background voxels back to zero after centering changes the outer tissue/background contrast. The figures show skull-stripped **raw** intensities to distinguish the renderer's contrast from this preprocessing.

For a more realistic simulator, [BrainWeb](https://brainweb.bic.mni.mcgill.ca/) illustrates the distinction between a realistic anatomical phantom, sequence simulation, and separately controlled noise, nonuniformity and slice thickness. Its standard precomputed volumes are T1/T2/PD, so they should not be described as ready-made T1/FLAIR pairs.

## Priorities for this project

1. Make lesion and normal-tissue signals receive the same global gain/bias: first blend their base signals, then apply the shared affine intensity transformation. Keep a renderer-version switch to reproduce old checkpoints. This is a proposed change, not applied by this audit.
2. Explicitly use `wm_interior` in all new WM-lesion experiments and check the saved settings of each old run.
3. Calibrate tissue contrast and lesion contrast distributions to a specified T1 and T2-FLAIR protocol, ideally with paired reference scans processed the same way. Use tissue-relative contrasts and noise/resolution estimates rather than raw intensity matching alone.
4. Keep a controlled benchmark where each factor is observable in each view, and separately evaluate a more realistic benchmark with unequal visibility. Realistic T1-invisible lesions cannot be required to have an accurately recoverable T1-specific location encoding. This limitation differs from choosing whether visible shared anatomy is routed into content or style.

These measurements support unequal input visibility as one contributor to the learning results. They do not establish that visibility alone caused style routing. The earlier success after restricting spatial style capacity is separate evidence about the model's available reconstruction paths.

![Current wm_interior examples](/Users/nataliaglazman/Desktop/PhD/projects/multiview-crl/out/synthetic_realism_20260926/wm_interior_noncausal_examples.png)

![Older local configuration examples](/Users/nataliaglazman/Desktop/PhD/projects/multiview-crl/out/synthetic_realism_20260926/vqvae_identifiable_ventricle_examples.png)

Examples correspond to approximately the 10th, 50th and 90th percentiles of measured T1 lesion response among the 64 subjects. They are explicitly selected illustrations, not a random visual sample. Paired views use the same slice, and all panels use the same raw display window (0–1.15); intensity clipping in bright regions is a display choice.
