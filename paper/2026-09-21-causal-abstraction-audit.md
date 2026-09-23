# Causal abstraction feasibility in the current two-view implementation

Inspected 21 September 2026 at commit `451cf00c13342506d0012ad816e9037da7545ca5`. This is a code and simulator audit, with small CPU checks. No training, checkpoint updates, or fresh checkpoint performance evaluation was performed. The existing dirty nested worktree was left untouched.

## Decision

The most feasible next experiment is **approximate causal abstraction into vector-valued blocks, evaluated with actual SCM interventions**. The simulator supplies the ground-truth mechanisms, explicit rendering inputs, and reproducible noise needed for this. The current learned content tensor is a candidate measurement of the shared state; its sufficiency and nuisance exclusion are not established guarantees.

A smaller, lossy macro model is a separate research objective. The random SCM does not impose the mechanism restrictions that would make arbitrary anatomical summaries causally sufficient. An exact demonstration is best started in the existing numerical invertible-mixing track, with a deliberately modular SCM and explicit intervention map, before moving to the voxel renderer.

## What is actually implemented

### Generative process

`eval/synthetic_dataset.py:11` constructs an upper-triangular DAG. `sample_content_from_scm`, at line 46, uses independent standard Gaussian disturbances, unit noise scale at roots, and otherwise

\[
z_j=\operatorname{LeakyReLU}_{0.2}\left(\sum_{i\in\mathrm{pa}(j)}w_{ij}z_i\right)+0.4u_j
\]

under the inspected saved settings. The nine coordinates are brain size, ventricle size, lesion x/y/z, cortical thickness, temporal atrophy, left-right asymmetry, and sulcal widening. Their randomly assigned directions are synthetic causal ground truth, not assertions about biological causation.

`render_pseudo_mri` (`eval/synthetic_dataset.py:864`) generates one tissue/lesion structure and renders it as T1 and FLAIR with separate style draws and noise seeds. Thus the two modalities measure the same shared state; they are not two causally ordered latent blocks. The simulator does not specify a causal graph between image voxels.

All six inspected top-level saved settings use nine content factors, three style parameters per view, a random SCM with edge probability 0.5 and seed 42, clean content, the identifiable-ventricle variant, and fixed-reference normalization. In clean sphere mode, `CLEAN_NUISANCE_SCALE=0` removes the deformation and fissure fields from the image. Older notes counting hundreds of active shared field coordinates do not describe these runs. Field-lesion mode is a separate case.

`data/datasets.py:669` varies sample seeds across splits while keeping `scm_seed=synthetic_seed`. This is the correct separation of subject variation from the data-generating SCM. Fixed-reference normalization uses constants per dataset object (`data/datasets.py:772`); a new interventional protocol should freeze the observational constants across all intervention regimes, rather than estimate them anew per regime.

### Learned representation

The primary model is `models/vqvae.py`, trained through `training/main_multimodal.py`. `utils/config.py:1917` fixes two views and one common subset `(0,1)`. The channel mask separates content from style; it does not partition content into anatomical causal modules (`models/vqvae.py:1259`).

The active `experiments/synthetic_causal.yaml` uses nine content channels out of twelve, one encoder level, 64-cubed images, scale-four encoding, and 8-cubed patch pooling. `content_size` overrides inherited ratios in `utils/config.py:1890`. The inspected saved runs instead use twelve content channels out of sixteen. Neither number is a count of identified scalar causal variables: each channel is a spatial map.

Contrastive features come from continuous encoder outputs before VQ quantization (`models/vqvae.py:1378`). The decoder uses quantized content and a separate, potentially spatial style route (`models/vqvae.py:1534`, `:1644`). A finite codebook assignment cannot be a diffeomorphic representation of a continuously varying latent state. Continuous pre-quantization maps are the appropriate starting point for an approximate block-identification claim.

Barlow Twins/VICReg losses encourage alignment and noncollapse, but do not certify preservation of all nine factors. In VICRegL, the local and global objectives act after separate nonlinear heads (`training/vicreg_local.py:125`); their agreement does not establish agreement or sufficiency of every pre-head feature. The existing `paper/feature-map-formulation.md:110` already correctly distinguishes exclusion from preservation.

## Existing guarantees versus missing guarantees

| Statement | Status |
|---|---|
| Paired modalities share the same rendered anatomical structure | Implemented by construction. |
| The specified micro factor SCM is acyclic with independent disturbances | Implemented by construction for the inspected Gaussian-noise configuration. |
| Current clean sphere runs have nine active named anatomical factors | Implemented; shared deformation/fissure effects are zeroed. This does not establish that every factor is observable at every value. |
| One content channel represents one scalar factor | Not established; not implied by the architecture or loss. |
| The whole content tensor is an invertible, nuisance-free representation of all nine continuous factors | Not established; exact recovery is additionally obstructed by the hard renderer. |
| Several identified anatomical blocks exist inside content | Not established; additional block-selective information is needed. |
| An anatomical pooling or summary is an exact causal abstraction | No state map, intervention map, or commuting-condition test currently establishes this. |
| The learned graph is recovered without factor labels | Not what the current PC evaluation measures. |
| Group-level acyclicity or faithfulness follows automatically from the micro DAG | False in general; check separately for the chosen partition. |

## Three concrete findings from fresh simulator checks

Executed with `/opt/miniconda3/envs/adni-analysis/bin/python`, PyTorch 2.6.0, two CPU threads. Machine-readable results are in `results/causal_abstraction_audit_20260921/simulator_checks.json`.

### 1. Current perturbations are renderer sensitivity interventions

`eval/interventional_identifiability.py:332` changes one coordinate in `z_content` and leaves every other coordinate fixed, then re-renders. It does not solve the intervened SCM. This is useful for asking whether the image/encoder responds to a factor while controlling the other rendering inputs. It is not the ordinary total effect of `do(z_j=a)` through the factor DAG.

For the actual seed-42 SCM (19 edges), I reconstructed the sampler from its explicitly drawn disturbances and obtained bit-identical factual factors. On one fixed context, increasing brain size by one unit and solving the SCM changed six other factors: ventricle size, lesion z, cortical thickness, temporal atrophy, asymmetry, and sulcal widening. The existing `_perturb` operation changed only brain size.

Keep the existing diagnostic. Add a distinct evaluator that accepts disturbances and a map of intervention targets to values, replaces the targeted equations, and recomputes all other equations in topological order. Holding disturbances fixed gives paired counterfactuals; fresh disturbances give population interventional distributions. Hold the rendering seeds and styles fixed for paired effects, and separately test robustness to resampled rendering nuisance.

### 2. Natural anatomical grouping creates a cycle

For the same seed-42 DAG, take:

- shape = {brain size, cortical thickness, temporal atrophy, asymmetry, sulcal widening};
- ventricle = {ventricle size};
- lesion = {lesion x, lesion y, lesion z}.

There is shape -> ventricle via brain size -> ventricle size, and ventricle -> shape via ventricle size -> asymmetry/sulcal widening. There are also ventricle -> lesion and lesion -> shape edges. A standard group DAG/PC interpretation is therefore invalid for this partition. This does not rule out a more general cyclic group model; it rules out silently treating the grouping as an acyclic SCM.

A controlled acyclic partition is instead:

\[
B_1=(z_0,z_1),\quad B_2=(z_2,z_3,z_4),\quad B_3=(z_5,z_6,z_7,z_8).
\]

It respects the generator's topological order. The current seed gives all three forward block edges: B1 -> B2, B1 -> B3, B2 -> B3. Consequently this example has no block conditional-independence restrictions to orient those edges observationally. It is useful for testing intervention consistency, but a sparse designed block DAG is a better discovery benchmark. Treat this partition as an oracle-designed experimental choice, not an inferred anatomical decomposition.

### 3. The hard renderer is not injective in the continuous factors

`render_structure` uses hard tissue inequalities (`eval/synthetic_dataset.py:374`) and a hard sphere lesion (`:430`). Tanh squashing does not remove this discretization. In one clean 64-cubed sample, changing ventricle size by 0.001 produced bit-identical T1 and FLAIR volumes under the same styles and rendering seed. Brain size and cortical thickness changes of 0.0001 also produced identical pairs. The latent differences were verified to be nonzero.

This is a structural obstacle to exact continuous-factor recovery, not evidence that useful approximate recovery is impossible. The finite grid makes the tissue/lesion maps locally constant away from threshold crossings; adding the same noise or more draws of noise downstream does not distinguish the underlying configurations when the rendered structure is identical.

## Why graph scores are not yet abstraction evidence

`eval/run_causal_recovery.py:295` standardizes/PCA-reduces the representation, fits a RidgeCV prediction of each true scalar factor, and passes those predictions to PC (`:313`, `:325`, `:347`). Its default is in-sample prediction. `--holdout-readout` makes the evaluation out of sample, but it remains a supervised factor readout. The best-F1 alpha is selected using the true graph (`:381`).

These results assess whether labels can turn a representation into useful factor measurements. They do not identify macro variables without labels, validate a micro-to-macro intervention map, or establish exact causal equivalence. For a new abstraction result, retain this as a supervised baseline, use a held-out readout, fix hyperparameters without the final test DAG, and evaluate intervention predictions separately. Fisher-Z is only a linear conditional-independence diagnostic for this nonlinear generator; nonparametric tests still require appropriate assumptions and enough independent subjects. Spatial patches are not additional independent subjects.

An existing VICRegL audit at step 34,001 found that reconstructed ventricle and lesion responses were carried almost entirely by the style route. I verified that both the current checkpoint and settings hashes match that audit's provenance. This is existing evidence, not a fresh performance measurement. It concerns decoder reliance on 64 IID-factor subjects and hybrid-code interventions; it does not prove the content tensor contains no information. It does mean reconstruction fidelity cannot currently serve as evidence of content-only sufficiency. See `synthetic-clean-content-causal-vicregl-rescaled/codex_audit_20260916/REPORT.md:9`.

## Feasibility of the proposed approaches

| Approach | Feasibility now | Additional assumptions or data |
|---|---|---|
| Treat all shared content as one vector-valued node, with an observed treatment/outcome | Feasible as a new synthetic task, once content adequacy is measured. | The current pseudo-MRI task has no separate treatment/outcome SCM. Add it explicitly; specify temporal order/confounding. One shared block alone gives no internal graph. |
| Discover edges between several vector-valued blocks | Best first extension. Exact at the oracle factor level for an acyclic quotient and whole-block interventions; approximate on current image representations. | Choose/design an acyclic partition; establish block-specific sufficiency and exclusion; add SCM interventions. Observational graph recovery additionally needs appropriate group-level Markov/faithfulness assumptions. |
| Compress blocks into scalar or low-dimensional causal summaries | Testable, but no generic exact guarantee for the random SCM. | Specify retained outcomes and intervention family; learn/prove mechanism closure on summary level; include multiple micro realizations of the same macro value. |
| Learn summaries through intervention consistency | Practical with the simulator, once proper SCM interventions are added. | Known or constrained intervention correspondence, information-preservation targets, diverse regimes, and held-out implementations/values. A small training loss is not proof of exactness over all interventions. |
| Establish an exact nonlinear-abstraction theorem with the current hard images/VQ assignments | Unsupported. | Use a generator/measurement model satisfying the required assumptions, or explicitly target approximate/discrete abstraction. The existing numerical mixing track is a better starting point for exact invertible measurements. |

The random SCM gives a concrete obstacle to aggressive compression. The weight matrix from the three lesion coordinates to the mechanisms for cortical thickness, temporal atrophy, asymmetry, and sulcal widening has rank three at seed 42. Since the leaky-ReLU is invertible, preserving all four child mechanisms while their other parents can be fixed requires preserving three independent linear combinations of the lesion coordinates. A smooth scalar lesion summary cannot do that. This is a statement about preserving those individual child mechanisms; it does not exclude a lossy abstraction that also discards downstream detail or restricts interventions.

For a positive lossy example, add a designed block SCM whose cross-block effects factor through known summaries, such as an upstream sum. Neither `random`, `chain`, nor `full` by itself imposes this. The existing `training/main_numerical.py:172` and `utils/invertible_network_utils.py:12` already supply an invertible-mixing scaffold. It still needs the macro mechanisms, intervention API, and appropriate regularity conditions; its current default leaky-ReLU mixing is only piecewise smooth.

## Recommended experiment sequence

1. **Oracle intervention layer.** Add explicit exogenous contexts and a surgical SCM solver, keeping the renderer-sensitivity API separate. Verify factual replay, target clamping, propagation, and unchanged non-descendants. Freeze graph, mechanism parameters, and normalization across regimes. Save target/value and context identifiers.
2. **Oracle block abstraction.** Start with an acyclic block partition or a sparse designed block DAG. Keep all coordinates within each block. Solve the block mechanisms and check that whole-block interventions commute before introducing the encoder.
3. **Frozen representation test.** On the existing checkpoints, fit held-out block readouts from continuous spatial content maps. Compare content-only, full encoder maps, and oracle factors. This is explicitly supervised and tests whether the information is available; failure at this stage does not establish a failure of causal abstraction itself.
4. **Block learning with additional supervision.** Keep two modalities. Generate controlled pairs sharing a selected factor block while varying the complement, or use known block-target labels. These are additional invariance relations; ordinary T1/FLAIR pairing does not supply them. Renderer-controlled pairs can be used for representation training, but distinguish them from SCM-propagated interventions used for causal evaluation. Descendants of a genuine intervention are not required to stay invariant.
5. **Lossy summary test.** Only then reduce block dimensions. Hold a proposed macro value fixed while changing the corresponding micro realization; evaluate retained downstream distributions under matched interventions. Include held-out values, joint interventions, alternate implementations, and a deliberately invalid-summary control. Matching means alone is insufficient.

The supported research claim after successful experiments would be that a two-view representation supports an approximately intervention-consistent abstraction for a specified intervention family and set of retained outcomes. A stronger exact claim requires a separate structural proof and assumptions satisfied by the chosen simulator and representation target.
