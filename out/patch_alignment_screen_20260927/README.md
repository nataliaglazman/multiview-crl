# Controlled patch-alignment screen, 27 September 2026

This is a checkpoint-free diagnostic, not an evaluation of the trained encoder.
It used `eval.patch_signal_audit` with the local
`synthetic-clean-content-causal-sp-s-1/settings.json`, 64 subjects, batch size 64,
grids 8 and 16, seed 0, and `--causal match`. The configuration has position
centering, folded patch statistics, lambda 6, sim coefficient 0.0114, std
coefficient 0.227, correlation EMA 0.99, and patch/GAP weights both 1.
The renderer uses the legacy lesion placement for this configuration.

No checkpoint was loaded and no training parameters were changed. The local
environment lacks `lpips`, an unrelated eager import of `training.losses`.
Execution therefore used the existing `tests/test_patch_signal_audit.py`
`source_losses()` helper to load the original BT function and its dependencies
from source, without modifying their implementation.

## Findings

At grid 8, shuffling a planted target in one of two ideal aligned feature views
changed the instantaneous unweighted patch loss by approximately:

| Target template | Mixed with other variation in its channel | Dedicated target channel |
| --- | ---: | ---: |
| Ventricle, T1 | -0.0000170 | +0.0708 |
| Ventricle, FLAIR | -0.0000080 | +0.0701 |
| Lesion, T1 | +0.0000035 | +0.0679 |
| Lesion, FLAIR | +0.0000042 | +0.0781 |

These are constructed feature distributions with one controlled random seed,
not measured encoder features. Each modality's template was copied into both
ideal views separately; this does not simulate the actual T1/FLAIR discrepancy.
Small negative differences must not be interpreted as evidence that the trained
encoder is being encouraged to erase the factor. Inspect the component deltas
and directional gradients in `loss.csv`, including the EMA reference columns.

The dedicated-channel control shows why sparsity alone is not enough to explain
failure: correlation normalization can protect sparse signals. Competition with
stronger variation within the same channel is a plausible additional mechanism.

Raw input intervention squared-energy retention after average pooling and lifting:

| Intervention | Grid 8 | Grid 16 |
| --- | ---: | ---: |
| Ventricle, T1 | 31.0% | 50.8% |
| Ventricle, FLAIR | 31.0% | 50.8% |
| Lesion on/off, T1 | 9.6% | 39.8% |
| Lesion on/off, FLAIR | 20.4% | 57.5% |

This is input-space smoothing, not the trained encoder's pooling operation,
mutual information, or a decodability score. Finer pooling did not consistently
make the mixed-channel loss sensitive in this screen.

## Stronger test on a real checkpoint (proposed, not run here)

1. Generate paired ventricle low/high endpoints and lesion on/off endpoints,
   fixing the remaining factors, style, noise, and normalization. Test lesion
   location separately with a fixed-size lesion move. For WM-constrained
   placement, verify that changing ventricles does not also move the lesion.
2. Extract intervention responses at the native alignment source, actual
   pooled/projected loss input, and quantized decoder input. Measure response
   magnitude and cross-view response agreement, with wrong-subject controls.
3. Compare matched endpoints `(T1_low, FLAIR_low)` and
   `(T1_high, FLAIR_high)` against within-subject mismatches
   `(T1_low, FLAIR_high)` and `(T1_high, FLAIR_low)`. Both conditions contain
   exactly the same endpoint multiset in each view. Use 32 subjects times two
   endpoints for a logical loss batch of 64; encoder chunk size can be smaller.
4. Report patch and GAP losses separately, including component deltas and
   configured weights. Use identical cloned correlation histories for EMA
   comparisons. A warmed reference cannot recreate the unrecorded training
   history. Repeat batches and estimate uncertainty over subjects.
5. Separately test loss sensitivity to attenuating the factor response in both
   views. Mismatch detection alone does not establish an incentive to preserve
   the factor. Feature-space attenuation is a counterfactual diagnostic, not
   necessarily an achievable encoder update.
6. If needed, take a few temporary patch-only, GAP-only, and reconstruction-only
   encoder updates on separate model copies, then reassess held-out responses.
   Determine whether alignment strengthens the weak view or suppresses the
   strong view. Restore all parameters and buffers between arms.

Ground-truth factors/masks are used only to construct and evaluate diagnostics;
this protocol does not require supervised anatomy losses in training.
