# Fully local encoder and pooled pointwise objective

Recipe: `experiments/encoder/pointwise.yaml`. Theory: `paper/theory-framing.md`, section 9. A **cell** is one position
of the 16³ backbone map at `--res 64`. A **row** is one subject at one cell.

| Requirement | Flag |
|---|---|
| No normalisation pooled over voxels | `--norm-type layer` (across channels at each voxel) |
| Bounded content outputs | `--content-bound sigmoid`, the head's last module, so eval scripts calling the head read it too |
| Modest, known receptive field | `--nb-res-layers 0`: 18 voxels (50 with two residual blocks) |
| Smooth activations | `--activation gelu` or `silu` |
| Alignment at the same position | `--pointwise-align-input raw` or `residual` |
| Joint entropy on raw content features | `--pointwise-entropy-input raw`; `residual` is the comparison arm |
| Statistics pooled over subjects and positions | `--pointwise-rows` rows drawn across both each step |
| Negatives from other subjects and positions | `--pointwise-negatives pooled` (all but the row itself) or `strict` (both differ) |
| Loss only where the receptive field stays inside | `--interior-only`, for the patch and pointwise losses |
| No per-subject statistic | `--global-loss-weight 0` keeps the global loss logged but out of the objective |

## Objective

`total = global_weight * global + patch_weight * patch + pointwise_weight * (align + entropy)`

`losses.pointwise_alignment_entropy` follows `split_infonce_loss`: two terms that may read different inputs, the
positive left out of the entropy sum. Identical rows give align 0 and entropy log(M - 1).

## Why the entropy kernel wraps

This departs from a plain reading of "a box to fill". A repulsive kernel on a sphere ends uniform by symmetry; a box has
faces, and repulsion piles mass on them. The code stays in (0, 1)^C, but `--pointwise-kernel torus` (the default)
identifies opposite faces for the entropy term alone. It is a product of von Mises kernels, whose Fourier coefficients
are positive, so uniform is a strict local minimum at any tau. Alignment never wraps. `box` is the plain Gaussian.

Toy encoder, fresh batches, 10 latents, 10 outputs, no style. Old box tau values, so divide tau by 2 pi² (19.7) to
compare; `--pointwise-tau` now uses one scale for both kernels. KS 0.01 and outer-1% 0.02 are a true uniform sample.

| kernel | tau | KS from uniform | values in outer 1% | R² of latents |
|---|---|---|---|---|
| box | 0.001 | 0.085 | 0.003 | 0.77 |
| box | 0.01 | 0.228 | 0.146 | 0.64 |
| box | 0.03 | 0.404 | 0.464 | 0.48 |
| torus | 0.03 | 0.078 | 0.001 | 0.80 |
| torus | 0.1 | 0.040 | 0.003 | 0.70 |
| torus | 0.3 | 0.031 | 0.002 | 0.54 |

With 3 private style latents per view and 4096 rows (torus): tau 0.02 gives KS 0.234 and R² 0.82, tau 0.03 gives 0.198
and 0.80, tau 0.05 gives 0.143 and 0.73, tau 0.1 gives 0.086 and 0.66. A wider kernel fills the box better and recovers
less, so 0.03 is a default from a toy, not a tuned value. Wrapped alignment was no better than plain.

Two more settings from the same toy, at tau 0.03. Rows: 1024 instead of 4096 lowers latent R² from 0.80 to 0.74 and
raises style R² from 0.12 to 0.16, so `--pointwise-rows` is worth its cost. Negatives: adding same-view rows
(`--no-cross-view-negs-only`) halves style R² to 0.06 with latent R² unchanged, at about 4.5× the compute, so it is an arm
rather than the default.

## Interior only depends on depth

At `--res 64`, stride 4, with channel norm. Under GroupNorm no cell escapes padding, and the run prints a note.

| Kept by the interior rule | 2 residual blocks (50 voxels) | 0 residual blocks (18 voxels) |
|---|---|---|
| Cells reading no padding | 64 of 4096 | 1728 of 4096 |
| Foreground-mask cells | 4.5% | 92% |
| Brain voxels | 10% | 99.8% |
| Lesions whose best cell survives | 21% | 100% |

## Content channels

`python -m eval.encoder.encoder_receptive_field --run-dir <run> --out reach.json` reports the hard reach, the effective
reach and the formula's count on the run's generator. It replaces `probe/jacobian_spread.py` here, which needs a decoder.
On the recipe's generator `--synthetic-clean-content` zeroes both shared fields, so the count is 10: 7 scalars and the
lesion's 3 placement numbers. With the fields on it is 91 + 10 at 18 voxels and 576 + 10 at 50, an upper bound. The recipe
header has the channel ablation.

## Reading a run

- `cell eff_rank`: channels the rows really use, and in brackets the same with the template removed. The bracket far
  below `--content-channels` means the subjects vary along too few directions.
- `view r`: the views' mean unit-by-unit correlation on the template residual. High means the code is shared content.
- `saturated`: share of values outside [0.01, 0.99]. High, the code is turning binary.
- `subject share`: share of the code's variance between subjects at a position. Near 0, a template fills the box.

The rank and `view r` are label-free, and they are what separated the two arms below.

## First results: raw against residual entropy

One seed, the recipe on the burden generator (`out/pointwise_pilot_20261102/`). The raw arm ran 2000 steps. The residual
arm was stopped at step 1000 to save battery, so compare the arms there. The probes are the regional pilot's, on the
xpred bank's 400 validation subjects (`score.py`, `code_health.py`). The first two rows are its deeper GroupNorm encoder
at step 2000 with 14 units, so their floor differs and they are for orientation only.

Lesion in the cell code, T1 / FLAIR. Transfer fits the probe on the other view, so it reads what the views share.

| state | code AUROC | code top cell | transfer AUROC | transfer top cell |
|---|---|---|---|---|
| region16, step 2000 | 0.668 / 0.739 | 5% / 10% | 0.725 / 0.775 | 12% / 15% |
| sepadam, step 2000 | 0.688 / 0.717 | 10% / 9% | 0.747 / 0.759 | 18% / 18% |
| recipe, untrained | 0.670 / 0.679 | 9% / 11% | 0.412 / 0.381 | 4% / 3% |
| raw entropy, step 1000 | 0.738 / 0.724 | 10% / 11% | 0.715 / 0.717 | **21% / 26%** |
| raw entropy, step 2000 | 0.727 / 0.725 | 12% / 14% | 0.718 / 0.716 | 20% / 24% |
| residual entropy, step 1000 | **0.842 / 0.911** | **25% / 49%** | 0.714 / 0.819 | 3% / 2% |

What the code is, T1 / FLAIR where it differs by view:

| state | rank on residuals, of 10 | view r | lesion direction cosine across views | saturated |
|---|---|---|---|---|
| untrained | 6.6 / 6.3 | 0.10 | -0.43 | 0% |
| raw entropy, step 1000 | 8.8 / 8.9 | 0.96 | 0.97 | 0.1% |
| raw entropy, step 2000 | 8.6 / 8.7 | 0.97 | 0.995 | 0.3% |
| residual entropy, step 1000 | 2.7 / 2.9 | 0.65 | 0.36 | 7% |

Held-out R² of each factor from the 10-unit code at 16³, T1 / FLAIR:

| state | brain | ventricle | thickness | temporal | asymmetry | sulcal | centroid x | y | z | burden |
|---|---|---|---|---|---|---|---|---|---|---|
| untrained | .85 / .79 | .81 / .41 | .70 / .65 | .65 / .65 | .88 / .89 | .38 / .72 | .27 / .26 | .00 / .00 | -.03 / -.12 | -.01 / -.01 |
| raw, step 1000 | .92 / .89 | .91 / .89 | .86 / .86 | .86 / .85 | .91 / .91 | .65 / .80 | .40 / .49 | .14 / .39 | .13 / .31 | -.21 / -.02 |
| raw, step 2000 | .93 / .91 | .91 / .89 | .87 / .86 | .86 / .85 | .91 / .91 | .65 / .80 | .41 / .49 | .20 / .43 | .15 / .32 | -.02 / -.02 |
| residual, step 1000 | .89 / .87 | .87 / .76 | .80 / .80 | .84 / .85 | .91 / .92 | .26 / .61 | .26 / .48 | -.07 / .34 | .01 / .16 | -.02 / -.02 |

Reading: raw entropy does what the theory asks. The code stays wide, the views agree on it, and the lesion response
that starts with opposite signs in the two views ends in one direction, so a probe fitted on one view finds the lesion
in the other more often than any regional arm. Residual entropy collapses to under three directions. Its within-view
lesion score is high but does not transfer, so it is a view-specific signal rather than shared content. A template did
not win on raw features: `subject share` rose from 0.22 to 0.46 by step 900. Burden stays at zero in every arm.

Between steps 1000 and 2000 the raw loss kept falling but the probes barely moved, so more of the same training is
unlikely to help: the lesion stays in the shared code at about the size of 0.8 residual SDs, and the code's AUROC stays
near 0.72. The trained run's effective reach is 17 voxels of its hard 18, with no operation outside the box.

## Limits

- Raw entropy pooled over positions can in principle be filled by a subject-independent template
  (`paper/feature-map-formulation.md`, section 6), and `theory-framing.md` 9.2 and 9.4 still describe centring first.
  It did not happen in the one seed above; `subject share` is the check on another generator or on real data.
- A small receptive field stops a deep cell seeing global factors (`feature-map-formulation.md`, section 8), and nearly
  all content on this generator is global.
- The lesion is a sphere, not a field, so `theory-framing.md` Proposition 9.2 does not apply to it.
- `encoder_receptive_field` and pointwise training on real data are not validated.
