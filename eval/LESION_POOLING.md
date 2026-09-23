# Frozen lesion-location pooling comparison

```bash
python -m eval.lesion_pooling \
  --run-dir results/synthetic/synthetic-clean-content-causal-ident-vent-12-4 \
  --train-samples 256 --val-samples 128 --test-samples 128 \
  --encode-batch 8 --causal iid

python -m eval.lesion_pooling \
  --run-dir results/synthetic/synthetic-clean-content-causal-baseline-12-4 \
  --train-samples 256 --val-samples 128 --test-samples 128 \
  --encode-batch 8 --causal iid
```

This asks whether **lesion location** is recoverable from existing content encoder
patches in T1 and FLAIR. It compares GAP, mean/std, mean/std/max/min, regional
mean/std, and flattened spatial patches. The default regional grid is 2×2×2.
It does not retrain the model, update codebooks, or use anatomical labels to select
features/regions. Labels train diagnostic readouts only.

The sphere generator's `z_content[2:5]` controls location. Lesion radius is fixed
by `synthetic_lesion_radius`, not an independently varying sample factor. The
script fits two three-coordinate targets separately:

- **latent:** the three original content factors;
- **physical:** centroid of the actual rendered lesion support, in voxel indices
  along the tensor's three axes (not scanner RAS coordinates).

Physical position also depends on surrounding anatomy and clipping to white
matter, so these targets are not interchangeable. Field-lesion mode is rejected:
its `z_content[2:5]` does not control lesions. Samples with empty rendered support
are rejected rather than silently scored against an undefined location. This is
not a lesion-presence classifier or a segmentation test.

## Why include flattened patches?

A fixed-size lesion moving between positions can leave the mean, standard
deviation, minimum and maximum unchanged. Poor global-statistic readout can
therefore coexist with a clear spatial lesion response. Regional moments retain
coarse location; `patch_flat` retains the original position of every kept patch.
Deleted foreground positions are zero-filled in their original coordinates,
never concatenated into a shorter, misregistered vector. For 12 content channels
on an 8³ grid, widths are 12, 24, 48, 192 and 6144 respectively.

All descriptors use **the same forward content patches** and training foreground
rule. The saved training grid, masks, threshold, normalization, renderer settings
and resolution are used. Encoding microbatches are independent of foreground mask
batch size. Train, validation and test have separate mask groups; the final group
in each partition can be smaller than the saved training batch size. Retained
positions and regional counts are saved in metadata. Regions use fixed coordinates
without access to lesion labels. Empty regions have zero descriptors; singleton
regions have zero std. These conventions match the ventricular test.

Neither regional nor flattened patch readout recovers detail that was already
averaged away **inside** a patch. A failure of `patch_flat` is not proof that a
native encoder map contains no lesion signal. The existing `eval.lesion_probe`
tests native maps and content/style/joint blocks if that follow-up is needed.

## Readout protocol

Each subject is rendered/encoded once for all five descriptors. The source IDs
are shuffled deterministically, then split into train/validation/test. The default
is 256/128/128 subjects. This uses the existing `eval.lesion_probe` dual ridge/RBF
implementation, not the ventricular script's CV protocol:

- Feature scaling fits on train only, with constant columns excluded.
- Float64 Gram matrices avoid the float32 ridge-conditioning issue and a large
  feature-by-feature solve. No PCA is used.
- Target scaling fits on train. Validation selects regularization and bandwidth
  separately for physical and latent targets. Final predictions use the selected
  training fit; there is no refit on validation or selection using test scores.
- A separately fitted shuffled-label control uses the same procedure.
- No variable descriptor columns produces a training-mean predictor, not a missing row.

Outputs include per-axis and mean held-out R²; physical median position error in
voxels; shuffled-label scores; and paired bootstrap improvements over GAP,
mean/std and flattened patches. Intervals condition on this fitted model, split
and mask realization, and are not confidence over training seeds. A negative R²
means performance worse than predicting the held-out target mean. Compare the
same probe/target across poolings; do not pick a winning method by its test score
and treat that as an independently validated selection.

Interpretation:

- Good `patch_flat`, poor summaries: pooling discards useful location information.
- Regional moments improve over GAP: coarse location helps at this patch scale.
- Good physical but poor latent readout: observable lesion location is more
  accessible than the original generator coordinates under this protocol.
- All methods near shuffled performance: the tested content patches/readouts
  have not recovered location. This does not prove the lesion is absent from the
  inputs, native features, style or decoder output.

`--causal iid` removes SCM factor correlations but can be outside the training
distribution. `--causal match` retains the saved SCM. For arm comparisons, match
generator settings, checkpoint age, grid, filtering and sample budget. Matching
run names or parameter counts alone is insufficient.

## Reuse expensive extraction

`features.npz` is saved before fitting in a new `lesion_pooling_<timestamp>`
directory. It includes flattened patches and the original partitions. To rerun
readouts without rendering, encoding or loading a checkpoint:

```bash
python -m eval.lesion_pooling --features /path/to/lesion_pooling_TIMESTAMP/features.npz
```

Extraction options are ignored in cache mode. `--seed` then affects the shuffled
control/bootstrap, not the cached subject split. A ventricular feature cache
cannot be reused because it does not contain lesion targets or flattened patches.

This shares extraction with `eval.ventricle_pooling` and requires fixed content
selection. Unsupported projection/MoCo/split-normalization settings fail explicitly,
and checkpoint loading is strict. Trained GPU checkpoints must be evaluated on
the machine where they are available.

```bash
python -m unittest discover -s tests -p 'test_lesion_pooling.py' -v
```
