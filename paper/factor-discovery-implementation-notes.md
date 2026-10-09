# Implementation notes for the factor-discovery framework

9 October 2026. Companion to `factor-discovery-framework.tex`. The manuscript is a proposed framework, not a report of completed experiments or an established identifiability theorem.

## Confirmed setting

- Paired ADNI T1/T2 MRI.
- Age, sex, other demographic information, regional volumes, diagnosis, and cognitive scores.
- Exact variable names, measurement dates, missingness, and available acquisition metadata remain to be audited.
- Longitudinal observations and external cohorts are optional extensions, not assumed available.

## Existing components and proposed additions

| Component | Status and starting point |
|---|---|
| Paired representation training | Existing `training/main_multimodal.py` and `models/vqvae.py`. |
| Reconstruction and encoder-only comparisons | Existing variants and evaluation infrastructure. Match their capacity, information access, and readout flexibility. |
| Content/private spatial codes | Existing. Anatomical leakage and actual decoder reliance remain empirical questions. |
| Synthetic anatomical factors and graph | Existing `eval/synthetic/synthetic_dataset.py`. Some renderer factors may be discretised or inactive in particular configurations. |
| Factor and graph diagnostics | Existing `eval/causal/` infrastructure. Supervised factor readouts are diagnostic baselines, not unsupervised factor discovery. |
| Candidate factoriser and progressive splitting | Proposed. Preserve the full content map, use a fixed capacity range, and retain unresolved blocks. |
| Concealed-factor benchmark | Proposed. Keep selected factors active in the generator while withholding their labels from training, selection, and graph priors. |
| Structural intervention solver | Proposed addition for this framework. Preserve the original disturbances and recompute descendants. `eval/protocol/interventional_identifiability.py::_perturb` currently changes a coordinate while holding the other realised factors fixed; retain this as a separate sensitivity diagnostic. |
| Expanded graph and abstraction evaluation | Proposed. Specify state, context, and intervention maps before final testing. |

## Recommended implementation order

1. Audit the data and label inventory; assign each variable a measurement and temporal role.
2. Define subject-level partitions and reserve at least one target family for independent validation. Keep repeat visits together.
3. Implement and verify factual SCM replay, surgical interventions, descendant propagation, and unchanged non-descendants in the simulator.
4. Define concealed-factor and nuisance-only controls with predeclared evaluation metrics.
5. Reuse frozen checkpoints to establish which shared anatomical information is preserved before and after quantisation.
6. Fit a Peng-style linear residual baseline and proposed factorisations on matched data. Use stable linear algebra and select regularisation within training/validation partitions.
7. Compare graph estimators on fixed representations before allowing graph-based feedback to alter the encoder.
8. Evaluate the optional joint generative refinement and intervention-supervised extension as separate information-access conditions.

## Decisions needed before submission

1. **Observed-variable inventory:** which regional volumes and cognitive scores, measured when, and whether diagnosis incorporates those scores.
2. **Reference graph:** which edges are established background constraints, which are tentative, and which are merely associations in the original model.
3. **Discovery target:** common anatomy visible in both modalities versus a separate treatment of modality-specific pathology.
4. **Additional identifying information:** distinct visibility patterns, justified mechanism restrictions, longitudinal variation, or interventions. Two full views of the same block do not by themselves identify its internal coordinates.
5. **Independent endpoint:** which anatomical or cognitive targets remain unused during representation and factor selection.

## Build

From the repository root, compile the source twice with `pdflatex`. Use a temporary build directory for `.aux`, `.log`, and `.out` files. The supplied PDF was compiled and all pages were visually inspected. Its two-column article layout is conference-style but is not an official venue template.

## Scope of claims

The framework can propose factors omitted from a reference graph. A reproducible candidate, a refined measurement of a known phenotype, a biologically novel phenotype, and an identified causal variable are different claims. Report them separately. No synthetic or real-data results were generated while writing this draft.
