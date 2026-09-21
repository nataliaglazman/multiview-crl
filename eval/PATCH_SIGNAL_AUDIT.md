# Checkpoint-free patch signal audit

This separates **input averaging**, **competition with unrelated features**, and
**BT's incentive to retain a factor**. It does not test a learned encoder or claim
to predict the outcome of training. No checkpoint or model is loaded, and there
are no optimizer steps or supervised training losses.

Start with the current 8³ patch grid:

```bash
python -m eval.patch_signal_audit \
  --settings results/synthetic/synthetic-clean-content-causal-ident-vent-12-4/settings.json \
  --num-samples 128 --batch-size 64 --grids 8 --causal iid
```

`--settings` reads **only the JSON** for renderer and BT settings. It is optional;
without it the script uses 64³ clean, identifiable-ventricle synthetic images,
fixed-reference normalization, position centering, fold statistics, normalized BT
terms, lambda=6, sim=0.0114, std=0.227, correlation EMA=0.99, and patch/GAP weights=1.
The controlled feature width is independently set by `--channels` (default 12).
No projection head, channel mask learning, reconstruction, or quantizer is simulated.

An optional grid/amplitude sweep uses the same rendered batches:

```bash
python -m eval.patch_signal_audit \
  --settings results/synthetic/synthetic-clean-content-causal-ident-vent-12-4/settings.json \
  --num-samples 128 --batch-size 64 --grids 4 8 16 --strengths 1 4 --causal iid
```

Grid means **number of bins per axis**, not patch width. At 64³, grid 8 averages
8×8×8 input voxels per bin. These are input-space averages, not the receptive
fields of the trained encoder. The grid must divide the rendered resolution.

## 1. Real input interventions

- Ventricle: change only `z_content[1]` by −/+`--eps` (default 0.25).
- Lesion: remove the rendered lesion, with everything else fixed. This tests the
  lesion's appearance/removal, not localization readout accuracy or a location-only move.

The existing rendering helpers preserve the original normalization affine and
noise realization. Only the changed tissue support, dilated by one voxel for the
renderer blur and intersected with foreground, enters the difference image. This
excludes numerical normalization residuals elsewhere.

`geometry.csv` reports each subject, modality and grid:

- `pooled_energy_retained`: squared energy after averaging and lifting the bins
  back to image resolution, divided by original difference energy. This uses
  bin-volume weighting; comparing unweighted vector norms would be misleading.
- `masked_energy_retained`: the same calculation after the training foreground
  rule. A position is kept if **any subject** has enough foreground in that bin;
  the empty-mask fallback also matches training.
- `mask_keeps_pooled_energy`: how much already-pooled signal the mask preserves.
- `affected_kept_fraction`: fraction of retained bins touched by this subject's
  intervention, not the fraction touched by anyone in the batch.
- `signed_to_absolute_pool_energy`: signed-average energy divided by the energy
  obtained by averaging absolute differences. A low ratio indicates cancellation
  of positive and negative changes within bins. The absolute difference is only
  a diagnostic reference; it is not fed to the loss as a proposed feature.

Energy retention is a spatial smoothing measure, **not mutual information or
decodability**. Nonzero signal can remain predictable after large energy loss.
Invisible interventions retain their rows with undefined energy ratios.

## 2. The actual BT loss on controlled features

The script imports `training.losses.barlow_twins_loss`, with explicit channel
indices. It reuses position centering, fold/per-position statistics, per-entry
normalization, sim/variance terms, and the GAP or `stats_pool` companion. The
existing raw/normalized sim and global whitening options are passed through.

Each modality is tested **separately**: its signed, pooled intervention supplies a
spatial template, divided by that subject's lesion-on/ventricle-high foreground
standard deviation. A random subject-level ± sign provides controlled variation.
The template is copied into both ideal aligned views. This intentionally avoids
treating raw T1/FLAIR intensities as aligned embeddings. It cannot measure actual
cross-modal representability or alignment conflict.

Unrelated features are identical across the two views, with expected variance 1:
1/4 from a subject-wide component and 3/4 from subject-by-position variation.
The target enters channel 0:

- `mixed`: channel 0 also contains unrelated variation.
- `dedicated`: channel 0 contains only the target; the other channels are unchanged.

`--strengths` scales the target without rescaling nuisance. These are explicit
feature assumptions, not estimates of a learned feature distribution.

For each layout, compare matched targets with (a) a nonzero cyclic subject shuffle
of the target in the second view, holding nuisance fixed; and (b) target deletion
from **both** views. The loss is also compared on the training foreground mask and
the union of intervention-affected positions in the batch. The latter is an
**oracle diagnostic subset**, not a label-free training mask. It preserves subject
rows but changes the positions used to estimate BT statistics; the difference is
not an additive per-patch attribution. For moving lesions the union may be large.

`loss.csv` contains:

- `mismatch_delta`: L(shuffled target) − L(matched target), with all instantaneous
  batch statistics recomputed. A positive value penalizes this mismatch.
- `both_removed_delta`: L(no target in either view) − L(matched target). A value
  near zero means this controlled objective scarcely distinguishes retention from
  deletion. A negative value means deletion lowers this particular oracle loss.
- Separate on-diagonal, off-diagonal, MSE and variance-hinge values and deltas.
  Inspect these if the total changes little: terms can cancel each other.
- `mismatch_directional_gradient`: derivative at r=1 of
  L(common+target, common+target+r×(shuffled−target)). Positive means a small
  decrease in this explicit mismatch would decrease the instantaneous loss.
- `training_arm_multiplier` and `weighted_mismatch_delta`: the configured arm
  weight times `scale_contrastive_loss`. An arm with zero weight has no training
  effect even though its diagnostic is still calculated. No reconstruction
  gradient is available in this checkpoint-free test.

BT runs in float32 as in training. Tiny differences between large totals may round
to zero; inspect component deltas and directional gradients before calling a loss
insensitive. A nonpositive total delta is not automatically a bug: covariance and
variance criteria can offset the alignment penalty.

## 3. Direct dilution control

`density_control.csv` fixes 512 positions and target amplitude 1 in active bins,
then varies occupied positions: **1, 8, 64, 512**. Nuisance, subject signs and
shuffle are held fixed. Both layouts and both pooling arms are evaluated.

This isolates dilution from anatomical pooling. With raw MSE its mismatch penalty
scales with occupied fraction; GAP's raw MSE penalty scales with the squared
fraction for this uniform target. Correlation normalization can largely protect
the target when it has a dedicated channel. It does not necessarily protect a
small target mixed with dominant nuisance. The actual weighted total also includes
the covariance and variance terms, so do not assume it follows the MSE curve.

## EMA and interpretation

The main console values are **instantaneous**, even if saved settings enable EMA.
With EMA, CSV/JSON additionally report a one-step mismatch delta and derivative
from a hypothetical converged **matched** correlation history. Every counterfactual
starts from the same cloned history. This distinguishes a current loss response
from EMA gradient attenuation without loading a historical checkpoint. It is not
an estimate of the actual training EMA or an independent stationary mismatch run.

Aggregate JSON contains batch means/minima/maxima, not confidence intervals. The
reported batch mean is unweighted if the final batch is smaller. Repeat `--seed`
to vary the controlled nuisance/sign/shuffle; renderer subjects use the saved
synthetic seed and selected split. No patch is treated as an independent subject.

Useful patterns to look for:

1. Low input energy retention: pooling smooths the intervention before any loss.
2. Much stronger mismatch penalty on the affected subset or dense control: loss
   sensitivity depends on spatial occupancy under these controlled features.
3. Dedicated strong, mixed weak: competition within channels matters; patch
   count alone does not explain the behaviour.
4. Mismatch penalized but deletion from both views cheap: alignment can enforce
   agreement without requiring retention of this particular anatomical factor.

These results can motivate a loss-design experiment, but cannot establish that
BT caused the observed content/style routing in a trained model.

Outputs are saved to a new timestamped `results/patch_signal_audit_*` directory,
or a fresh `--out-dir`. Existing directories are rejected. Verification:

```bash
python -m unittest discover -s tests -p 'test_patch_signal_audit.py' -v
```
