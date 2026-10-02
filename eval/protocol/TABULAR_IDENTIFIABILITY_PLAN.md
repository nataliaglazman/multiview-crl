# Component-wise identifiability from sparse tabular readouts — synthetic plan

The question this answers: **does adding a sparse tabular head move the representation from
block identifiability to component-wise identifiability?**

If it does on synthetic, where ground truth exists, the same head is the route to
component-wise identifiability on ADNI — where the tabular variable also *names* each latent,
which is the thing no other route on the table supplies.

## Why this and not more image views

T1 and FLAIR are two contrasts of the same co-registered object. They share essentially all
anatomical content, which is exactly the setting von Kügelgen et al. proved gives content as a
**block**. No number of image contrasts changes that.

A tabular variable is different in kind: it is a *sparse* readout of a *few* latents.
Hippocampal volume reads medial-temporal atrophy, not ventricle size. With images alone any
rotation of the content block reconstructs equally well; requiring that each tabular variable
be predicted by a small number of coordinates makes a rotation smear that dependence and
violate the sparsity. **The rotation stops being free.** That is the whole mechanism, and it
is what this plan tests.

## The pre-registered prediction

Two metrics that already exist, and they must move differently:

| metric | where | prediction | if violated |
|---|---|---|---|
| `per_channel_mcc` | `identifiability_metrics.py:539` | **rises** | mechanism does not work here |
| `block_mcc` | `identifiability_metrics.py:279` | **flat** | capacity confound, not identifiability |

`per_channel_mcc` fits no readout — it is `|corr|` between raw channels and sources under a
Hungarian match, i.e. literally "is one channel one factor". `block_mcc` permits the invertible
transform, so it measures what we already have. **Both rising means the tabular head simply
added capacity or information, not that it pinned the basis.** That contrast is the experiment.

## Generator changes

New flags on the synthetic generator, all defaulting to off so existing runs are unchanged:

- `--synthetic-n-biomarkers K` — number of tabular variables (0 = off)
- `--synthetic-biomarker-parents P` — content factors each biomarker reads (default 2)
- `--synthetic-biomarker-noise S` — additive noise on each biomarker (default 0.3)
- `--synthetic-biomarker-missing F` — fraction of entries dropped per variable (default 0.0)

Each biomarker `k` gets a random support `S_k` of size `P` drawn once from the content factors,
and `b_k = f_k(z[S_k]) + S * eps`, with `f_k` the same `leaky_relu` mechanism family the SCM
already uses so the tabular arm is not accidentally easier than the causal one.

Returned in `gt_latents["biomarkers"]` alongside `gt_latents["biomarker_support"]` — the true
bipartite support matrix, which is what makes support recovery scorable.

**K < n_content on purpose.** The realistic ADNI case is ~5-10 useful tabular variables against
9+ anatomical factors, with some factors having no direct readout at all. A one-to-one
biomarker per factor is near-direct supervision and would prove nothing.

## Model changes

A tabular head reading the pooled content vector, which already exists as one row per subject
at `training/main_multimodal.py:1471-1474`:

```
t̂_k = g_k(M_k ⊙ s)                       per biomarker, M ∈ [0,1]^{d×K}
L  += β · Σ_k ℓ(t_k, t̂_k) · observed_k    per-variable missingness mask
    + γ · ‖M‖₁                            the identifiability pressure
```

Flags: `--tabular-head`, `--tabular-weight` (β), `--tabular-sparsity` (γ).

`M` at convergence is the learned bipartite support — the naming table, and the thing scored
against `biomarker_support`.

## Arms

Three arms plus a floor each, three seeds per arm:

| arm | tabular head | sparsity γ | isolates |
|---|---|---|---|
| **A** baseline | no | — | where you are now |
| **B** tabular, no sparsity | yes | 0 | does the *information* help? |
| **C** tabular + sparsity | yes | > 0 | does the *sparsity* pin the basis? |

**Arm B is the one that makes the claim.** Without it, a rise in `per_channel_mcc` is
explained equally well by "the model got extra information about the factors" as by "the
sparsity removed the rotation". C-minus-B is the effect the mechanism predicts; C-minus-A is
not.

## Metrics

Floor-subtracted throughout, three init seeds, per the discipline in `run_dci_compare --floor`:

1. **`per_channel_mcc`** — headline.
2. **`block_mcc`** — control; should not move.
3. **`max_alias`** from `interventional_identifiability.py` — an independent, probe-free
   component-wise measure. Two metrics that could disagree, agreeing, is the evidence.
4. **Support recovery** — precision/recall of learned `M` against `biomarker_support`. The
   direct test that the sparsity found the *right* structure rather than just a sparse one.
5. Reconstruction and the existing contrastive diagnostics, to confirm nothing was traded away.

## Gates

Stop at the first failure rather than running the whole grid.

- **Gate 1 — does it work at all?** Arm C beats arm A on `per_channel_mcc` by more than the
  across-seed spread, at the easy end (K = n_content, P = 1, noise 0.1, no missingness). If not,
  the mechanism does not work in this architecture and nothing downstream matters.
- **Gate 2 — is it the sparsity?** Arm C beats arm B. If C ≈ B, the tabular *information*
  helped and the sparsity did nothing; the honest report is "weak supervision helps", which is a
  much smaller claim and does not transfer to ADNI as an identifiability argument.
- **Gate 3 — is it identifiability?** `block_mcc` stays flat while `per_channel_mcc` rises. If
  both rise, it is capacity.
- **Gate 4 — does it survive realism?** Degrade toward ADNI: `K < n_content`, `P = 2-3`,
  noise 0.3, missingness 0.4. The effect should shrink but survive.

## Sweep order

Easy end first, and only degrade once the gates pass. The point of starting easy is to
distinguish "the mechanism is wrong" from "the setting is too hard", which a single mid-range
configuration cannot.

1. `K = n_content`, `P = 1`, noise 0.1, missing 0.0  → gates 1-3
2. sweep `γ` (the sparsity weight) at that setting → find the usable window
3. `K = 6`, `P = 2`, noise 0.3, missing 0.0 → gate 4, partial
4. `K = 6`, `P = 2`, noise 0.3, missing 0.4 → gate 4, ADNI-realistic
5. sweep `P` ∈ {1, 2, 3} at fixed K → how much cross-modal density the mechanism tolerates

Step 5 matters for ADNI: it tells you whether real biomarkers, which read many factors at once,
are sparse *enough*. That is the assumption the whole route rests on and it is the one you
cannot check on real data.

## Then, and only then, ADNI

Two independent naming signals become available, and their agreement is the result:

1. the tabular head — which latent predicts hippocampal volume
2. spatial locality (`interventional_identifiability.locality`) — which latent localises to the
   hippocampus in an atlas

Agreement between them is convergent validation of component-wise identifiability with no
factor labels. Disagreement is also informative and must be reported.

**State plainly that this is weak supervision.** The defensible claim is that identifiability
comes from the sparse *structure*, not from the values — the model is never told which latent
is the hippocampus, it discovers it. That distinction is real but it is not "unsupervised", and
claiming otherwise will not survive review.

## Two caveats to carry into the writeup

**FreeSurfer volumes are not an independent modality.** They are a deterministic function of
the same T1. They still work as sparse readouts that pin the basis, but that is closer to
distilling a segmentation prior than to multimodal identifiability. CSF and PET biomarkers are
genuinely independent measurements and are what carries the argument; check the
T1 + FLAIR + CSF intersection before committing, since it is much smaller than the T1 cohort.

**Downstream variables carry little sparsity signal.** MMSE and ADAS-Cog depend on many factors
through long chains, so their support is dense and they contribute almost nothing here. The
volumetric and biomarker variables do the work.
