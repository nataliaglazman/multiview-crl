# Normative content bundles

`export_normative_bundle` fits a label-free content representation from the synthetic
generator recorded in a VQ-VAE or encoder-only run's `settings.json`. No checkpoint is
needed. The exporter uses the existing `NormativeResidual` model and shared `.npz` writer.

```bash
python -m eval.protocol.export_normative_bundle \
  --run-dir results/synthetic/RUN \
  --fit-samples 1200 --components 20 --num-samples 2000 \
  --out results/bundles/normative.npz
```

The first two thirds of the training subjects fit each view's PCA modes; the remaining
third calibrates residual SD. CCA on the training mode scores finds cross-view directions.
With `--shared-dims auto`, keep pairs above the 95th percentile of the largest canonical
correlation obtained by shuffling the FLAIR rows and refitting CCA. This width is not an
estimate of the number of causal factors: nonlinear anatomy can need several modes.

The averaged canonical pairs define a shared block. Spatial ICA treats the voxel loading
maps as samples and chooses a non-orthogonal channel basis, allowing the resulting subject
scores to correlate. `--rotation none` and `--rotation promax` provide alternatives.
Three additional channels locate the smoothed unsigned residual peak in the subject's
brain frame. `--lesion none` omits them. All fitted transforms use training images only;
factor labels are saved for evaluation and never enter the transforms.

The default `--block per_view` exports separate T1 and FLAIR features using the same
rotation, with each view's own residual peak. `--block unified` exports only view 1,
using both images: averaged anatomy channels and the peak of the summed residual
magnitudes. That setting has more input information than a single-view encoder.

## Scoring

Run PC directly on the exported channels, with a fixed alpha and a random-mixing reference:

```bash
python -m eval.causal.latent_causal_discovery \
  --bundles normative=results/bundles/normative.npz \
  --alphas 0.05 --ceiling --mixing-null 30 \
  --out results/bundles/normative_graph.json
```

Bundle columns go straight to PC; `--reduce` and `--pooling` apply only to run directories.
`--bundle-view 2` selects FLAIR. Multiple `label=path` inputs, `--run-dirs`, and
`--bundle-floors label=path` can be combined. Bundles and their floors must have matching
row identities; `--allow-row-mismatch` explicitly relaxes this check. If the evaluated
factor rows differ, the single truth ceiling describes only the first scored row.

The table includes channel MCC (Hungarian-matched absolute correlation, without a
readout), graph F1/SHD, and the random naming null. The mixing null reruns PC and matching
after random Gaussian, non-orthogonal mixing of standardized channels. For normative
bundles, only `meta.rotated_channels` are mixed; the lesion coordinates stay fixed.
Other bundles mix all columns unless they supply that metadata. Fewer than two selected
mixable channels give no mixing null. Floors and the truth ceiling are not mixed.

With multiple `--alphas`, each row and each mixing draw selects its best F1 against the
truth. These summaries are optimistic. Use one prespecified alpha for the main comparison;
JSON retains every `alpha_sweep` and the null's `f1_by_alpha`, plus individual best-F1
and MCC draws and their random seed.

For block-level probes, use the ordinary shared scorer:

```bash
python -m eval.protocol.compare_bundles \
  --bundles normative=results/bundles/normative.npz vq=results/bundles/vq.npz
```

Export the same split, row count, generator and causal setting for both models. The
normative default test split matches `export_vq_bundle`; `--split val` matches the
encoder-only checkpoint scorer. `--causal iid` changes the scored factor distribution
and omits the true adjacency, so graph scoring skips that bundle. Fitting still uses the
run's original training distribution. There is no untrained normative twin.

`compare_bundles --with-graph` fits a supervised factor readout before PC. Use
`latent_causal_discovery` for the label-free channel graph instead.

## Limits and validation

Spatial ICA assumes spatially independent loading maps; it does not establish causal
identifiability. Overlapping effects (such as brain size and cortical thickness) may
remain mixed. Promax can change factor correlations even when channels correlate strongly
with individual factors. Dense true graphs and shared lesion channels can make the
naming null easy to beat, so inspect the mixing null and channel MCC alongside graph F1.

The exporter currently restores synthetic generators. It does not fit an ADNI normative
model; real WMH evaluation needs an appropriate training population and residual burden
measure rather than a single lesion peak. Fitting holds compact training images and uses
the normative model's full SVD; memory and time grow with subjects and resolution.

The torch-free self-test covers shared-dimension selection, spatial ICA on correlated
planted factors, the Promax caveat, and residual coordinates:

```bash
python -m eval.protocol.export_normative_bundle --self-test
python -m eval.causal.latent_causal_discovery --self-test
python -m unittest discover -s tests -p test_normative_bundle.py
python -m unittest discover -s tests -p test_latent_causal_bundles.py
```

Regression tests also cover label independence, training/evaluation separation, bundle
round trips, generator row identity, fixed lesion columns under mixing, channel MCC
agreement, floors, view selection, and rejection of mismatched rows.
