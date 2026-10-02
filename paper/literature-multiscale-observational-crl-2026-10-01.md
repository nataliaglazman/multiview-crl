# Multiscale MRI representations without intervention training data

Literature search and implementation proposals, 1 October 2026. This is a targeted review, not a systematic review or a novelty claim. The proposals below have not been implemented or tested. Repository observations refer to the current working tree.

## Recommendation

Start with a controlled two-level representation experiment on the existing synthetic generator. Preserve a fine spatial map, add a coarser map, and evaluate both spatial preservation and subject-level factor recovery. Then test whether spatial covariance assumptions add useful identification beyond architecture alone.

Three claims must remain separate:

1. The architecture contains features at multiple resolutions.
2. The features recover distinct biological factors or spatial sources.
3. The recovered variables support identification of a causal graph.

The first is enforceable by architecture. The second can sometimes be established from observational data under additional assumptions. The third requires further causal assumptions and cannot be inferred from the first two alone.

“Scale” also needs an operational definition: grid resolution, spatial correlation length, anatomical support of a factor's effects, and level of causal abstraction are different properties. A global size factor can move sharp tissue boundaries. A scalar burden can require fine spatial evidence. A local lesion can contribute low-frequency image information. Do not assign every biological factor to a Fourier band.

## Most relevant literature

### Spatial dependence: the strongest direct connection to coarseness

**Hälvä, So, Turner and Hyvärinen, AISTATS 2024: [Identifiable Feature Learning for Spatial Data with Nonlinear ICA](https://proceedings.mlr.press/v238/halva24a.html).** [Authors' implementation](https://github.com/cambridge-mlg/tp-nica).

Independent latent spatial processes are observed through a smooth, injective pointwise nonlinear mixing function. Spatial dependence supplies identifying information without intervention data. In the GP limit, distinct covariance kernels are required; the TP model permits broader cases under its remaining assumptions. This concerns independent sources, not arbitrary dependent causal variables. Its likelihood, dimensionality, regularity and mixing assumptions matter: adding a smoothness penalty to an MRI encoder does not inherit the theorem.

This is the best starting point for asking whether different correlation lengths can make spatial sources distinguishable. It does not imply that a lesion, cortical thickness and brain size must correspond to three stationary processes.

### Observational grouping: a route to causal variables

**Morioka and Hyvärinen, ICML 2024: [Causal Representation Learning Made Identifiable by Grouping of Observational Variables](https://proceedings.mlr.press/v235/morioka24a.html), G-CaRL.** [Authors' implementation](https://github.com/hmorioka/GCaRL).

The model divides observations into groups, each with its own latent set and mixing function. Latents can interact across groups. Identification requires invertibility, sufficiently informative cross-group connections and specific interaction conditions. Estimation distinguishes real group tuples from independently shuffled tuples using a structured score. An unrestricted discriminator is not an equivalent estimator. Representation identification and edge orientation have separate conditions.

Anatomical regions suggest a possible application, but the assumptions must be checked. Shared global anatomy affects several regions, and overlapping image pyramids do not create disjoint latent sets. Use a controlled grouped benchmark before claiming this theory for full MRI.

### Multimodal sparsity: relevant, but distinguish modalities from image contrasts

**Sun et al., ICLR 2025: [Causal Representation Learning from Multimodal Biomedical Observations](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2ec91b31ac4aa578d99309d4f00c81ad-Abstract-Conference.html).** [Full text](https://arxiv.org/html/2411.06518v3).

This develops component identification using structural sparsity of causal connections between modality-specific latent sets. It requires more than a generic sparsity penalty: invertibility, informative cross-modality influence and a suitable sparsest structure are substantive assumptions. Shared factors receive additional treatment beyond the core setup. T1 and FLAIR primarily measure overlapping anatomy; their acquisition channels should not be treated as biological causes of one another. The useful transfer is the idea of constrained latent connectivity, subject to a defensible measurement model.

### Partial observability: useful for shared content, with a block-level limit

**Yao et al., ICLR 2024: [Multi-View Causal Representation Learning with Partial Observability](https://arxiv.org/abs/2311.04056).** [Authors' implementation](https://github.com/CausalLearningAI/multiview-crl).

Different views observe subsets of potentially causally related latents. Shared information can be identified up to a smooth bijection, with more detailed recovery possible for suitable visibility patterns. A jointly identified block may still mix individual factors. Cropping or downsampling is not automatically a valid latent-subset observation: it may retain weak traces of many variables or remove invertibility. This is an existing conceptual foundation for the project, rather than a new multiscale guarantee.

### Structural sparsity without auxiliary variables

**Zheng, Ng and Zhang, NeurIPS 2022: [On the Identifiability of Nonlinear ICA: Sparsity and Beyond](https://arxiv.org/abs/2206.07751).**

Specific sparsity conditions on the mixing process can identify independent sources without auxiliary variables. This motivates testing restricted spatial influence or decoder connectivity. Generic L1 regularization does not establish those conditions, and independent-source recovery is not identical to recovering causally dependent biological variables. A decoder with local pathways is an experimental hypothesis unless the resulting model is shown to satisfy an applicable theorem.

### Practical local and global learning

**Bardes, Ponce and LeCun, NeurIPS 2022: [VICRegL: Self-Supervised Learning of Local Visual Features](https://arxiv.org/abs/2210.01571).** [Authors' implementation](https://github.com/facebookresearch/VICRegL).

Separate local and global objectives support spatial and image-level downstream tasks. This is empirical representation learning, not causal identification. The repository already has an adaptation for registered MRI, so merely adding local and global heads would duplicate existing functionality. The immediate extension is applying suitable objectives at multiple actual encoder levels and checking which information survives the bottlenecks.

**Yu et al., Medical Image Analysis 2024: [DrasCLR](https://pubmed.ncbi.nlm.nih.gov/38086236/).** [Authors' implementation](https://github.com/batmanlab/DrasCLR).

This lung CT method combines anatomical location information with local and neighborhood contrastive learning. It is useful precedent for learning subject-specific pathology within aligned anatomy. A possible MRI adaptation compares corresponding locations and neighborhoods across views and subjects. Registration errors, false negatives and scanner shortcuts need controls. Its empirical CT results do not establish brain MRI performance or causal identifiability.

**Razavi, van den Oord and Vinyals, NeurIPS 2019: [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446).**

Hierarchical quantized representations provide an architectural precedent already reflected in this repository. Different levels offer different capacities and receptive fields; the construction does not guarantee one biological variable per level or channel.

### A later option if longitudinal data become central

**Li, Fu et al., NeurIPS 2025: [Towards Identifiability of Hierarchical Temporal Causal Representation Learning](https://proceedings.neurips.cc/paper_files/paper/2025/file/1c6decac1477fbcc2cbf12d314ce0133-Paper-Conference.pdf), CHiLD.** [Authors' implementation](https://github.com/MinghaoFu/CHiLD).

Hierarchical temporal structure provides another source of identifying information. Its conditions include suitable conditional independence and variation; simply having three visits is insufficient. This is relevant to a future longitudinal ADNI study, but the current cross-sectional synthetic experiment does not supply the required temporal structure. Hierarchical temporal levels also need not correspond to spatial resolutions.

## Proposal A: two spatial levels with separate evaluation

This is the smallest useful implementation and the first experiment I recommend. It addresses information preservation, not a new identification theorem.

For each MRI view, retain a fine content map and a coarse content map. Apply registered cross-view alignment separately at each level, with subject-wise anti-collapse statistics. Form a subject representation from both maps when needed. Do not require fine and coarse representations to be independent: causal relationships can create dependence, and some information should be available at several scales.

```text
paired MRI view
    -> fine content map, 16 x 16 x 16 -> spatial readout
    -> coarse content map, 4 x 4 x 4  -> regional/global readout

both maps -> optional subject representation -> factor/graph evaluation
```

The arrows describe computation, not a causal graph. A subject representation intended to retain location should use a spatially aware readout, such as fixed regional pooling followed by an MLP, and be compared against GAP. Concatenating two GAP vectors does not automatically recover lost coordinates. A burden summary should be allowed to aggregate evidence from the fine map.

Concrete repository work:

- `models/vqvae.py` already supports multiple levels and per-level patch grids. Starting from the current 64-cubed, stride-4 encoder, use two levels with scaling rates `[4, 4]`, yielding nominal grids 16-cubed and 4-cubed. This preserves the original first-stage resolution. Test `[2, 4]` separately if needed.
- Set content/style separation at both levels and use each level's actual width. Account for decoder paths and codebooks when matching capacity.
- Extend `training/vicreg_local.py`: `validate_vicregl_args` currently rejects more than one level, and `attach_vicregl_heads` creates only `L0`. Construct a head for each selected content level before optimizer/checkpoint initialization.
- Reuse per-level grids and `contrastive_level_weights` in `training/main_multimodal.py`. Register new losses through the existing loss-audit mechanism.
- Evaluate native feature maps before and after quantization. Alignment heads can discard information that remains in their inputs, so evaluate both locations when interpreting a failure.

For continuously varying ground-truth factors, a finite quantized code cannot provide an exact invertible representation over the entire continuum. Apply smooth identification arguments to the appropriate continuous representation and assess the approximation introduced by quantization separately.

A two-level BT experiment can reuse more of the existing plumbing; a two-level VICRegL experiment needs the head changes above. Keep the loss family fixed when testing the architectural change.

The current one-level configuration pools a nominal 16-cubed map to 8-cubed for patch alignment. Include a one-level 16-cubed alignment control before attributing an improvement to hierarchy. The existing constructed-feature audit in `out/patch_alignment_screen_20260927/README.md` motivates this control, but is not evidence of trained encoder performance. Finer grids can still dilute small lesions in a spatially averaged loss.

An optional next ablation is a DrasCLR-inspired neighborhood objective at two physical extents. Treat this as a separate change. Same-position negatives can distinguish subject variation from an atlas template, but do not prove that the variation is lesion-specific.

## Proposal B: give spatial fields distinct covariance structure

This is the most direct research idea about coarseness without intervention data. Start with dedicated spatial-source channels; do not force all anatomy factors into stationary smooth fields.

For a cheap prototype, measure each channel's autocorrelation over a few physical-distance bins and encourage different target correlation lengths. For example, use a target `k_s(d) = exp(-d^2 / (2 ell_s^2))`, with ordered lengths `ell_fine < ell_coarse`. The distances must be measured consistently across grids.

For this loss only, subtract the across-subject mean feature at each registered position before estimating covariance:

```text
R[n, c, u] = F[n, c, u] - mean_over_subjects(F[:, c, u])
```

This reduces the chance of measuring a fixed anatomical template. Do not instead normalize away each subject's spatial mean or total amplitude: those can carry burden. Include foreground pair masks, stable variance normalization, and subject-wise anti-collapse terms. The covariance regularizer is a proposed approximation, not TP-NICA's likelihood or a proven estimator.

A stronger study would first reproduce the spatial-source model using the authors' implementation. Build a small compatible toy with independent fields and a smooth injective pointwise mixing map. Use enough observed channels for the latent dimension; three unrestricted source values cannot be recovered pointwise from two scalar observations through an injective map. Compare distinct-kernel GP sources, repeated-kernel GP sources and TP sources under the paper's assumptions. Then separately test robustness to MRI rendering departures.

Repository details that matter:

- `eval/synthetic/synthetic_dataset.py` already has GP/TP-inspired fields and distinct/repeated kernel options. Its Gaussian-filtered sampler centers and normalizes each realization. It is therefore not literally the unconditioned GP assumed by a standard GP likelihood. A theorem-aligned control needs a suitable sampler or an explicit analysis of the modified distribution.
- Hard tissue overwrites and masks can remove source information. Two MRI contrasts do not automatically satisfy pointwise invertibility for all latent fields.
- With `lesion_mode="field"`, the three lesion-position scalars are inactive and must be removed from that experiment's scalar scoring.
- `synthetic_clean_content` can zero some shared fields. Field-recovery experiments need explicit nonzero-source checks.
- Verify lesion support against the final tissue segmentation; the existing field-mode envelope is not equivalent to the sphere mode's interior-WM placement.
- Reuse `eval/diagnostics/probe_fields.py` to score source recovery. Matching a target correlation length alone is not success.

If independent spatial disturbances feed a causal mechanism, recovering those disturbances still does not identify that mechanism or its endogenous variables. State exactly which object the experiment recovers.

## Proposal C: anatomical groups with a structured observational objective

This is the more ambitious causal project. First reproduce G-CaRL on a small setting satisfying its model. Then create a controlled regional benchmark in which each disjoint region has its own latent block, with causal dependencies across blocks and sufficiently informative observations within each group.

Use regional encoders and the structured group-tuple objective from the reference method. Real examples contain all groups from one subject; negative examples shuffle groups across subjects. This constructs a learning signal from observations and does not require biological intervention data.

Before moving to MRI, test deliberate violations: a global factor affecting several groups, weak cross-group connectivity, overlapping groups and unobserved shared nuisance. These are likely to matter in the intended application. A shared global anatomy block may be a useful extension, but adding it does not automatically preserve the original guarantee. Do not describe a generic region-wise contrastive loss as a reproduction of the identifiable method.

This route has a stronger connection to causal variable recovery than an image pyramid, but substantially more modeling and validation work. It is not the first experiment needed to diagnose the current lesion issue.

## Minimal experiment sequence and success criteria

Keep the existing generator fixed for the initial architecture comparison, including its nine scalar factors. Changing generator, objective and architecture together would prevent attributing an improvement.

| Experiment | Change | Main question |
|---|---|---|
| A0 | Existing one-level model, 8-cubed alignment | Reference |
| A1 | One level, native 16-cubed alignment | Is pooling the main bottleneck? |
| A2 | Two levels, same loss family, per-level alignment | Does extra scale structure help beyond A1? |
| A3 | A2 plus a specified spatial prior | Does the proposed prior improve source/factor recovery? |
| B | Separate compatible spatial-source toy | Does the theoretical mechanism work when its assumptions hold? |

Match subject splits, optimization budget and seeds; report parameters and bottleneck capacity. Equal channels do not imply equal capacity when the number of spatial code positions changes. Prioritize A0–A2 before a broad sweep.

Report the two results the user requested separately:

- **Subject-level recovery:** per-factor scores from the stated subject readout, with anatomy and lesion-coordinate summaries separated. Keep the nine-factor average as a secondary historical comparison. A graph method's actual input must be evaluated directly.
- **Spatial preservation:** held-out lesion localization/map decoding from native maps at every level and their combination, with matched probe capacity and subject splits. Report original latent coordinates and physical centroid separately where they differ. Use dense-map metrics when the generator supports lesion maps.

A scale-by-target score matrix will reveal specialization without assuming it. Test pre- and post-quantization representations, an untrained baseline and label-shuffle controls. Check subject variation at fixed anatomical positions to detect atlas shortcuts. Correlated factors can support indirect decoding, so localization and conditional predictive checks are useful alongside scalar scores; neither alone establishes causal identification.

For causal discovery, distinguish direct unsupervised representation input from supervised factor readouts. The current Ridge-then-PC path evaluates graph recovery after supervised decoding. Report that as such. Also use a conditional independence test appropriate to the nonlinear generator, and distinguish identifiable equivalence classes from a fully oriented DAG.

## Applying this to a more realistic lesion factor later

After the architecture comparison, a separate generator study could represent pathology using burden plus a shared spatial lesion field, optionally with a regional distribution parameter. Burden is a scalar target; the map remains shared content even if it is outside scalar scoring. Do not call it modality-specific nuisance if both contrasts are generated from it.

If burden is defined as an aggregate of the map, changing burden does not specify which local lesions change. A causal model needs an explicit mechanism for that relationship. Likewise, subtracting a field's mean does not hold a nonlinear sigmoid's total mass fixed: a burden-calibrated renderer needs an explicit constraint or calibration step.

Finally, verify the real acquisition pair before making translation claims. The synthetic task uses T1/FLAIR, while repository descriptions of the real pipeline include T1/T2. Shared-factor assumptions should reflect what each actual contrast can reveal. None of the proposed objectives can guarantee recovery of information absent from its input.
