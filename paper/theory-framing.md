# Theory framing: two identifiability anchors, and what our construction satisfies

Working note for the paper's theory section. Every theorem statement below is taken from the
source text; every claim about our own setup is tagged with where it was measured or derived.
Conditions we **violate** are marked ✗ and must appear in the paper — they are the difference
between a defensible framing and a referee finding them for us.

Sources:

- **Yao et al., ICLR 2024** — multiview block identifiability (content vs style, across views).
- **Hälvä, So, Turner, Hyvärinen, AISTATS 2024** (PMLR 238), *Identifiable Feature Learning
  for Spatial Data with Nonlinear ICA* — component-wise identifiability for latent **fields**
  over a spatial index set.

The two are complementary and neither subsumes the other: Yao operates *across views* and
delivers a block; Hälvä operates *over space* within a single view and delivers components.

---

## 1. Our generative model

Subject `n` carries latents `z = (c, s_1, s_2)`:

- `c_lab ∈ R^9` — global scalar content factors (brain_size, ventricle_size, lesion_x/y/z,
  cortical_thickness, temporal_atrophy, lr_asymmetry, sulcal_widening)
- `c_field` — two **spatial fields**, `z_deformation` (4³ lattice) and `z_fissure` (8³ lattice),
  upsampled into the volume
- `s_k ∈ R^3` — view-specific style (gain, bias, noise_sigma)

Views `x_k = f_k(c, s_k)`, `k ∈ {1,2}`, each `f_k` injective.

**Scoping statement for the paper.** `c = (c_lab, c_field)` has ~594 dimensions of which 9 are
labelled. Every metric we report probes a 9-dimensional projection of `c`. This is the source
of the R² ceiling and must be stated before any results table.

---

## 2. Representation and its field decomposition

Encoders `g_k : X_k → R^{m × |Ω|}` on the latent grid `Ω` (`|Ω| = 4096` at 16³). Write
`ĉ_k(n,u)` for subject `n` at site `u`. Decompose

    ĉ_k(n,u) = T_k(u) + r_k(n,u),        T_k(u) = E_n[ĉ_k(n,u)]

`patch_center_mode: position` subtracts `T_k` exactly, so **the contrastive objective acts on
`r` alone**.

Measured (2k → 89k, reference run):

| component | share at 2k | share at 89k |
|---|---|---|
| `T` positional template | 66.7% | 61.3% |
| `S` subject signal, shared across views | 30.8% | 37.4% |
| `E` view-specific residual | 2.42% | 1.35% |

So a majority of representation variance is in `T`, which the contrastive objective never
sees. It is shaped by reconstruction alone. State this — it is the formal version of the
finding that the encoder gauge is 68–72% of the cross-view map residual and invisible to BT.

---

## 3. Anchor A — Yao: across-view block identifiability

> **Definition (block identifiability).** `ĉ` is block-identified if there is an invertible
> `h` with `ĉ = h(c)` a.s.

Decomposes into exactly our two metric families:

- **exclusion** — `ĉ ⊥ s_k`: `leak_c2s`
- **sufficiency** — `c` recoverable from `ĉ`: `info_c2c`, `block_mcc`

### Proposition 1 (our alignment surrogate is sufficient) — provable, two lines

Under `patch_stat="fold"` with position centering, channel `i` is standardised over the joint
index `(n,u)`, and `r` is already zero-mean across `n` at each `(u,i)`. Hence `C_ii = 1` forces
equality of the jointly-standardised fields, i.e.

    r_2(n,u) = diag(a) r_1(n,u)   for all n, u,   a_i = σ_{2,i}/σ_{1,i}

— **one positive scalar per channel, constant over space and subjects.** Defining
`g̃_2 = diag(a)^{-1} g_2` gives exact pointwise equality of the two content fields, and `g̃_2`
differs from `g_2` by an invertible map, which block identifiability already quotients.

**Consequence.** The standing objection "correlation is weaker than the a.s. equality Yao's
Lemma C.2 requires" **does not bite for block identifiability** — the gap is exactly a diagonal
gauge inside the theorem's own equivalence class, and with separate encoders the re-gauging is
free.

ε-version: `C_ii = 1 − ε_i` gives `E[(r̃_2 − r̃_1)²] = 2ε_i`. Measured `ρ = 0.965`, so `ε = 0.035`
— the condition holds to within 3.5% of content variance.

### `patch_stat` is a choice of equivalence class

Under `per_position`, statistics are per `(i,u)` across subjects, so `C^(u)_ii = 1` gives
`r_2(n,u) = diag(a(u)) r_1(n,u)` — a gauge free to vary with position. **Strictly weaker** than
`fold`. Worth stating; the current docstring frames the two as near-equivalent on statistical
grounds, which is true of their sampling behaviour but not of what they impose.

### Proposition 2 (our entropy surrogate cannot identify components) — provable

Let `L_off(Z) = Σ_{i≠j} Corr(Z_i, Z_j)²`. If `Corr(Z) = I` then `Corr(ZR) = RᵀR = I` for every
orthogonal `R`. **The minimiser set is closed under O(m)**, so no second-order decorrelation
criterion selects a basis.

Verified numerically: independent Laplace sources vs a random rotation of them —
`L_off` 0.0002 → 0.0003 (unchanged), 4th-order dependence 0.0001 → 0.9095. For Gaussian
sources the rotation is undetectable at *both* orders.

Corollaries to state:

- Component-wise identifiability is unattainable under this objective at any coefficient.
- At the optimum `E[L_off] = m(m−1)/B`. At `m=44, B=128` that is **14.78** against a measured
  17.999 — 82% of the term is sampling error. A statement about the estimator, not the model.

---

## 4. Anchor B — Hälvä et al. (AISTATS 2024): component-wise, over space

### Their model (§3), stated exactly

`N` statistically independent latent components `s^(i) = (s_l^(i))_{l ∈ L}`, `i = 1..N`, where
`L` is an indexing set of **arbitrary dimension** (2D or 3D for spatial data). Each component
is a **t-Process** over `L`:

    s^(i) ~ TP_{ν^(i)}(h^(i), κ^(i))

Observations are generated by a nonlinear injective mixing `f : R^N → R^M`, `M ≥ N`, that
**operates at each index of L**:

    x_l = f(s_l) + ε_l,     l ∈ L

with `ε_l ∈ R^M` i.i.d. across the `M` dimensions and w.r.t. the components. gp-NICA is the
`ν → ∞` limit.

**Proposition 1 (TP as infinite mixture, Yu et al. 2007).** `TP_ν(h,κ)` is sampled by drawing
`τ ~ Gamma(ν/2, ν/2)` then `φ ~ GP(h, (1/τ)κ)`. — This is exactly what
`sample_gp_field(prior='tp')` implements; our code is faithful.

### Theorem 1 (identifiability of tp-NICA)

> Assuming that the assumptions (A1), (A2) and (A3) of Theorem 1 in Hälvä et al. (2021) apply,
> then the tp-NICA model is identifiable such that `p(x; f) = p(x; f̂) ⟹ f⁻¹ ∼ f̂⁻¹`, where `∼`
> denotes **equivalence up to permutation and coordinate-wise bijective transformation** of the
> elements of the de-mixing function.

⚠ **(A1)–(A3) are inherited from Hälvä et al. (2021) NeurIPS and are not stated in this
paper** (its Appendix B is supplementary, not in the PDF we have). They must be checked
against the 2021 paper before we cite Theorem 1 as applying to us.

### Theorem 2 (necessity of distinct covariance kernels, GP case)

> Assume a model otherwise defined as in §3 except with GP components `s^(i) ~ GP(h^(i), κ^(i))`.
> Then `p(x; f) = p(x; f̂) ⟹ f⁻¹ ∼ f̂⁻¹` **if and only if** the covariance kernels of the
> different components are unique, `κ^(i) ≠ κ^(j)` for all `i ≠ j`.

Noise-free case (they note this is the more general one). Proof extends Belouchrani et al.
(1997) for linear ICA.

**This is the symmetry-breaking Proposition 2 says our objective lacks**, and it names two
independent routes: *distinct spatial kernels* (second-order but **spatially** structured —
our rotation-invariance proof concerns the marginal correlation matrix, not the spatial
autocovariance) and *non-Gaussianity* (the t-process; the same mechanism as a `Corr(z²)` term,
but with a theorem).

---

## 5. Mapping our construction onto Hälvä — including what fails

| condition | status |
|---|---|
| `L` a 2D/3D spatial index set | ✓ our latent grid Ω and image domain |
| components are fields over `L` with their own kernels | ✓ for `z_deformation` / `z_fissure` |
| components statistically independent | ✓ fields; **✗ `c_lab` is SCM-coupled** (`causal_graph: random`, ventricle/brain_size ≈ 0.8) |
| mixing `f` applied **pointwise at each `l`** | **✗** our renderer is not pointwise — `brain_size` moves the whole boundary; the CNN encoder has a 13-voxel RF plus 4× downsampling |
| every latent is a field | **✗** the 9 labelled factors are **global scalars**, which the model has no slot for |
| distinct kernels (GP case) | ✓ by construction when `field_kernels="distinct"`; ✗ under `"repeated"` |

### The sharp consequence for `c_lab`

A global scalar is the degenerate limit of a field that is constant over `L`, whose kernel is
`κ(l,m) = σ²` for all `l,m`. **Any two global scalars therefore have identical kernels**, so
Theorem 2's *if and only if* fails immediately for the GP case. Under this framework,
component-wise identifiability of our nine labelled factors **is not available** — not as a
limitation of our estimator, but as a property of the latent structure.

This is consistent with everything we measured (poor per-channel/DCI, good block-level
separation) and it converts a negative result into a prediction of the theory.

### Hälvä's own empirical caveat — calibrate our predictions to it

From §6.1: with **repeated** kernels, tp-NICA scores 0.05–0.15 MCC above gp-NICA "as theory
predicts" — but "**in contrast to theory, the gp-NICA model is not completely unidentifiable**
and as the number of mixing layers increases the two models' performance converges", because
a specific kernel and a specific MLP mixing function impose inductive biases absent from the
theory's arbitrary-`f` assumption. Performance also degrades with mixing depth.

So the preregistered prediction for our control is **tp > gp under repeated kernels**, *not*
gp at chance. Our encoder is far deeper than their 1–4 mixing layers, so expect the gap to be
at the small end.

---

## 6. Metric consequence — we are using the wrong correlation

Hälvä §6.1, on evaluating against an identifiability result that is up to permutation and
coordinate-wise bijection:

> the ground-truth and the estimated components were matched using a **linear sum assignment**
> algorithm prior to computing the MCC. Second, since each component is identifiable up to a
> bijective transformation, **a Pearson correlation is not appropriate and we instead use
> Spearman rank correlation** to compute the MCC.

Our `block_mcc` uses `linear_sum_assignment` ✓ but `_abs_corr_matrix` = **|Pearson|** ✗. Under
any equivalence class that permits coordinate-wise bijections, Pearson under-reads a correctly
recovered component that has been monotonically reparameterised.

**Action:** add a `corr="spearman"` option to `block_mcc` and report both. If the Spearman
variant is materially higher, part of what we have been calling degradation is a
reparameterisation the theory explicitly permits.

---

## 7. The control experiment Theorem 2 licenses

Our generator already implements the grid (`field_kernels ∈ {distinct, repeated}` ×
`field_prior ∈ {iid, gp, tp}`), but **`synthetic_field_prior` / `_kernels` / `_lengthscales`
are not exposed in `utils/config.py` or any experiment YAML**, so every run to date took the
`iid` branch (white noise at two different lattice resolutions — a de facto but uncontrolled
kernel difference).

| | repeated kernels | distinct kernels |
|---|---|---|
| `gp` (Gaussian) | **not identifiable** (Thm 2), expect partial recovery per §6.1 | identifiable |
| `tp` (Student-t) | identifiable (Thm 1) | identifiable |

Three cells, one theorem making the calls in advance, and component-wise metrics that finally
have a reason to be reported.

---

## 8. What to claim, and what not to

**Claim.** (i) Block-level content/style separation, with Proposition 1 showing our correlation
surrogate is faithful to Yao up to the theorem's own gauge, and the 18× purity result as
evidence. (ii) Proposition 2 as a *negative* structural result about second-order surrogates.
(iii) Hälvä's Theorem 2 as the reason component-wise claims are unavailable for global scalar
factors, with the kernel control as the test.

**Do not claim.** That either theorem applies to our estimator. Yao's is proven for an
encoder-only align−entropy objective; Hälvä's for a likelihood-based tp-NICA fit with pointwise
mixing. We run Barlow Twins + reconstruction + VQ, and we violate Hälvä's pointwise-mixing and
independence assumptions outright. The honest statement is that we test whether the
identifiability *conclusions* survive a different estimator and a violated assumption set — and
say which assumptions, explicitly.

**Do not report** per-channel DCI-D as a result without the cross-seed stability check (same
channel, same factor, two seeds). Given Proposition 2, expect it to fail; a clean negative
there is itself a sentence worth writing.

---

# 9. Pointwise block identifiability: the formulation that makes patches work

Two failed attempts, then the fix.

**Flattening** the field into `z ∈ R^N` keeps Yao's theorem valid verbatim but the conclusion
is vacuous spatially: an invertible `h` on `R^{dim(c)}` may permute sites arbitrarily.

**Sites-as-views** (`S_{k,u}` = latents affecting site `u`) breaks on compactly-supported
factors. Either `lesion_x/y/z ∈ S_{k,u}` and `f_{k,u}` is constant in them whenever the lesion
lies outside `u`'s receptive field — not injective, so not a diffeomorphism (Asm 2.1(i)) — or
they are excluded for distant sites and `S_{k,u}` becomes a function of `z`, which the fixed
index-set structure forbids.

The diagnosis: **the failure is not compact support, it is representing a spatial object by a
scalar coordinate.** Fix the representation and the index-set problem disappears entirely,
because there are no per-site index sets left.

## 9.1 Setting

Content splits into

- `γ ∈ R^{n_g}` — global scalars (brain_size, lr_asymmetry, …)
- `c : Ω → R^{n_l}` — a **local content field** (z_deformation, z_fissure, and lesion load
  represented as a field rather than a position triple)

For radius `ρ`, write the **local chart** `c_u := (c(u'))_{u' ∈ N_ρ(u)}`, a point in
`R^{n_l·|N_ρ|}`. Views are `x_k`, style `s_k`.

## 9.2 Assumptions

**(L1) Local stationary mixing (on the residual).** There is a single map `f_k`, *independent
of `u`*, with

        x_k|_{N_ρ(u)} = f_k(c_u, γ, s_k)

a diffeomorphism onto its image.

*Status:* the `u`-independence is spatial stationarity, which our anatomy violates (ventricle
at centre, temporal lobes at |x|≈0.30). **But `patch_center_mode: position` removes
`T_k(u) = E_n[ĉ_k(n,u)]` — the subject-independent template — before the objective sees
anything.** Stationarity is required only of the *subject-varying residual*, i.e. of the local
mechanism by which anatomy varies between subjects, which is far more plausible than
stationarity of the anatomy itself. This retro-justifies a design choice made on other grounds.

**(L2) Regularity.** `Z` open, simply connected, `p_z > 0` a.e. — as Yao Asm 2.1(ii).

## 9.3 Target

> **Definition 9.1 (pointwise, site-shared block identifiability).** `ĉ : Ω → R^m`
> pointwise-block-identifies `(c, γ)` if there exists an invertible `h_0`, **the same at every
> site**, with
>
>        ĉ(u) = h_0(c_u, γ)     for all u ∈ Ω, a.s.

Strictly between the two failed options: stronger than flattened block identifiability (which
permits site permutation), weaker than component-wise (which needs Hälvä). It is exactly what
a weight-shared convolutional encoder can deliver, and `dim(ĉ(u)) = n_l·|N_ρ| + n_g`, which is
the size constraint to check against `content_size`.

## 9.4 The fold is the correct statistic, not a weaker one

Restating Proposition 1 in this frame. Under `fold` + position centering, channel `i` is
standardised over the joint index `(n,u)`, and `C_ii = 1` forces

        r_2(n,u) = diag(a) r_1(n,u)   for all n, u

— one scalar per channel, **constant over space**. That is precisely the site-shared gauge
Definition 9.1 asks for.

Two consequences worth stating:

1. Under `per_position`, `C^{(u)}_ii = 1` gives `r_2(n,u) = diag(a(u)) r_1(n,u)` — a gauge free
   to vary with site, which does **not** satisfy Definition 9.1. So `fold` is the correct
   choice here, reversing the earlier reading that it was the weaker statistic.
2. The **folded entropy** is likewise the right object: since `h_0` is shared across sites, the
   `(n,u)` population is exactly the sample on which its invertibility should be assessed.
   Folding is not a statistical convenience; it is the estimator matched to a site-shared map.

## 9.5 Why spatial claims become legitimate

> **Proposition 9.2 (support recovery).** Suppose `ĉ(u) = h_0(c_u, γ)` with `h_0` invertible and
> `u`-independent. Let `A ⊆ Ω` be a region on which a content component `c^{(j)}` is constant.
> Then for every `u` with `N_ρ(u) ⊆ A`, `ĉ(u)` does not vary with `c^{(j)}`; and wherever
> `c^{(j)}` varies within `N_ρ(u)`, `ĉ(u)` does vary, by invertibility of `h_0`.
>
> Hence `{u : ĉ(u) depends on c^{(j)}}` recovers `supp(c^{(j)})` **dilated by `ρ`**.

This is the spatial claim flattened block identifiability could not support. Localisation is a
*consequence* of Definition 9.1, not an extra assumption.

> **Corollary 9.3 (resolution bound).** Localisation precision is bounded below by the
> receptive-field radius `ρ`. No objective can do better.

*Numerically, for us:* conv RF 13 voxels, downsampling 4×, so `ρ ≈ 3.25` latent cells; the
lesion has radius 3.2 voxels ≈ 0.8 latent cells. **The lesion fits entirely inside one
receptive field**, so Corollary 9.3 caps localisation at ±ρ regardless of the objective — which
is the theoretical counterpart of the measured `lesion_z = 0.699` and of the finding that
raising the probe grid to 16³ did not help. The lever is `ρ`
(`vqvae_scaling_rates: [4] → [2]`), not the loss.

## 9.6 What this costs, and what it does not need

**Needs:** the lesion represented as a field (a load map) rather than a position triple. This
is a *modelling* choice in the generator — no distributional assumption is required, in
particular no GP/TP and no kernel condition, because Definition 9.1 is block-level.

**Does not need:** Hälvä. That enters only one layer up, if you additionally want to separate
the *components* inside `h_0`'s argument, which requires distinct kernels (Thm 2) or
non-Gaussianity (Thm 1).

## 9.7 The resulting two-layer statement

| layer | claim | conditions | our status |
|---|---|---|---|
| 1 — pointwise block (§9) | `ĉ(u) = h_0(c_u, γ)`, support recovered up to `ρ` | (L1) local stationary mixing on the residual, (L2) regularity | alignment half measured at ρ=0.965; entropy half is the folded uniformity term |
| 2 — component-wise (Hälvä) | separation inside `h_0`'s argument | distinct kernels (Thm 2) or non-Gaussianity (Thm 1) | fields ✓ (distinct length scales); global scalars ✗ (identical constant kernels) |

Yao supplies the across-view content/style split (§3); §9 supplies the within-view spatial
structure; Hälvä supplies the within-block component structure. Each layer has its own
conditions and its own control.

## 9.8 Conditions to state as violated or unverified

- **(L1) `u`-independence** holds only for the subject-varying residual, and only approximately.
  Position centering is the mitigation; say so.
- **`dim(ĉ(u)) = n_l·|N_ρ| + n_g`** — compute this for the generator and compare against
  `content_size: 44`. Def 9.1's `h_0` cannot be invertible if they disagree.
- **Lesion-as-field** requires a generator change; until then Prop 9.2 does not apply to the
  lesion and the compact-support gap stands.
- The entropy half of the objective is currently a second-order surrogate (Prop 2), not a
  uniformity estimate, so Definition 9.1's invertibility is not actually being enforced.
