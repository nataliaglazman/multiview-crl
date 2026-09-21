# Frozen ventricular pooling comparison

This diagnostic asks whether a different summary makes ventricular size more
decodable from the **existing content encoder patches**, separately for T1 and
FLAIR. It does not update the encoder, decoder, codebooks or checkpoint. Synthetic
anatomical targets train diagnostic readouts only, not the representation model.

```bash
python -m eval.ventricle_pooling \
  --run-dir results/synthetic/synthetic-clean-content-causal-ident-vent-12-4-3 \
  --fit-samples 384 --test-samples 128 \
  --encode-batch 8 --causal iid
```

The saved training patch grid (8×8×8 for this run), content channel selection,
foreground threshold, renderer resolution, normalization and generator settings
are used. `--causal iid` deliberately removes SCM factor correlations, so brain
size cannot stand in for ventricular size through the SCM. This can be outside
the training distribution. Repeat with `--causal match` for in-distribution
readouts, where correlated factors can explain some apparent recovery.

Default descriptors for 12 content channels:

| Descriptor | Width | Definition |
|---|---:|---|
| `gap` | 12 | Spatial mean |
| `mean_std` | 24 | Mean and unbiased spatial standard deviation |
| `stats` | 48 | Mean, standard deviation, maximum, minimum, in that order |
| `regional_mean_std` | 192 | Mean/std in each of eight fixed, equal 2×2×2 spatial cells |

All four operate on the same retained patches, before patch centering, at the
encoder level selected by `--level` (default 0). These are unquantized content
features returned by `model(..., pool_only=True, return_recon=False)`, not
decoder-bound codes. Statistics over an 8³ patch grid generally differ from
statistics over a full-resolution encoder map, even with the same formulas.

Foreground positions are kept if **any view/subject in a logical batch** meets
the saved foreground threshold, exactly as in the training operation. This uses
the saved training batch size, independently of `--encode-batch`. Only small
patch maps accumulate between microbatches, not full-resolution volumes. Fit and
test subjects never share logical batches; a final partial batch is retained and
its size recorded. All four descriptors use the same batch mask. No population
mask is fitted across the entire evaluation set.

Regions refer to coordinates in the original patch lattice, not contiguous
slices of the compressed foreground list. They use no ventricular segmentation
or anatomical target. Empty regions produce zeros; singleton regions have zero
std. Region counts and all retained positions are recorded per batch, but counts
are not supplied as extra probe features. Changes in foreground support can
still affect every descriptor. `--regions 2 2 2` can be changed when extracting;
each regional dimension must divide the corresponding patch-grid dimension.

## Readouts and results

Ridge and RBF kernel ridge are tuned separately for each descriptor/view, on the
same fit subjects and CV folds. Feature/target scaling is fitted inside each CV
training fold. Final readouts are refitted on all fit subjects and scored on a
disjoint held-out set. The renderer seed comes from settings; `--seed` controls
probe CV and bootstrap sampling, not the renderer or SCM.

The console and `pooling_scores.csv` report held-out R², paired ΔR² against GAP,
and paired ΔR² against mean/std, with 95% bootstrap intervals. The JSON also
contains absolute R² intervals, fit CV scores and chosen hyperparameters.
Intervals condition on the fitted probes, split and extracted mask groups;
they are not uncertainty over training seeds or a full nested-bootstrap analysis.
Descriptor widths differ: comparisons measure practical readout performance at
this sample budget, not a capacity-matched information bound. A negative R²
means the readout is worse than predicting the held-out target mean.

- **`stats` beats `mean_std`:** extrema add recoverable signal under this readout
  protocol. An interval spanning zero leaves that gain unresolved; it does not
  establish equivalence.
- **`regional_mean_std` beats global summaries:** retaining coarse location helps.
- **RBF beats ridge:** some information needs a nonlinear readout.
- **All FLAIR scores remain poor:** these descriptors/readouts have not recovered
  the signal. This is not proof that the full feature map contains no information.

None of these outcomes proves that training with the winning pooling will improve
content encoding or decoder routing. Use this result to choose the next training
ablation, and use the separate routing diagnostic to test decoder reliance.

Outputs are created in a new `ventricle_pooling_<timestamp>` directory inside the
run. `features.npz` is saved **before probe fitting** to preserve expensive
rendering/encoding work if fitting is interrupted. To rerun only the readouts:

```bash
python -m eval.ventricle_pooling \
  --features /path/to/ventricle_pooling_TIMESTAMP/features.npz \
  --probes ridge rbf --seed 1
```

This reuses the original subjects, partition, grid and regions; extraction flags
are ignored in cache mode. To change these, extract a new cache. `--probes ridge`
is a faster first pass. CPU linear-algebra threads default to four; adjust with
`--threads`. Existing output directories are never overwritten.

Run the command separately for a baseline checkpoint to compare arms. Use the
same synthetic settings, distribution, sample counts and extraction settings;
`--grid 8 8 8 --mask-batch-size 64` can explicitly match these aspects. The saved
foreground-filter setting still comes from each run. If it differs between arms,
their comparison also changes masking; do not attribute the difference to the
contrastive objective alone. Within each run the four descriptors always share
identical subjects and foreground masks.

The implementation currently requires a fixed channel split and raw encoder
patches. It rejects projection/bounded/entropy heads, MoCo and split-encoder-norm
configurations the shared loader cannot faithfully rebuild. Checkpoint loading
is strict, so missing or extra state fails rather than silently evaluating a
partially initialized model. Model evaluation uses eval mode and stable replay
math, not historical training batches, dropout or mixed-precision training.

```bash
python -m unittest discover -s tests -p 'test_ventricle_pooling.py' -v
```
