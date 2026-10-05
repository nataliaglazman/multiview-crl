# Final-stage spatial maps of an encoder-only model

This analysis shows the encoder-only model's last spatial maps, before they are
averaged into the global content vector. It answers three questions:

1. What do the final maps look like?
2. Where does each content factor change them, and how much of that change
   survives global average pooling (GAP)?
3. From which positions can each factor be read out?

It reads a finished run directory, scores the trained checkpoint and the saved
`model_init.pt` on identical images, and never trains or selects a checkpoint.
Weights and source files are hash-checked.

## Stages

- `backbone`: the encoder's last feature map (16³ × 64 for the Conv recipe at
  resolution 64).
- `projected`: the active spatial content head applied to each bin of the
  analysis grid (9 content channels). The grid defaults to the run's training
  patch grid, which is how training and the spatial-recovery monitor use the
  head. With `--separate-spatial-readout` this is the separate head; otherwise
  the shared head.
- `global`: the content vector the model actually outputs (GAP, then the global
  head). This is what the DCI table probes.

## Run

On the NVIDIA PC, from the repository root in `monai_env`:

```bash
RUN="$PWD/results/encoder_patch_cuda_baseline/runs/conv_mlp_s42_cuda"
python -m eval.encoder.encoder_spatial_maps \
  --run-dir "$RUN" --out-dir "$RUN/evaluation/spatial_maps_$(date +%Y%m%d_%H%M%S)" \
  --device cuda
```

Locally use `--device mps` with `PYTORCH_ENABLE_MPS_FALLBACK=1` exported first,
or `--device cpu`. The output directory must be new. By default the analysis
encodes the run's validation cohort and 400 test subjects per arm. The
decodability grid defaults to the run's training patch grid (8 if it had none).
Add `--skip-initial` if `model_init.pt` is missing.

## Outputs

| File | Contents |
|---|---|
| `gallery_<arm>.png` | For the first test subjects, both views: input slice, backbone activity, its top three principal components, and the nine content-channel maps (one shared colour scale per channel). |
| `responses_<view>.png` | One row per factor: where changing it alters the input, the backbone and the content map (subject mean, maximum over z), trained beside initial on one colour scale. |
| `responses.csv` | Per arm, view, factor and stage: local response, coherent response, `gap_survival`, and for `global` the shift of the content vector. |
| `decodability_<arm>.png` | Held-out R² from each grid cell alone (maximum over z) for both views and stages, with the GAP R² in each title. |
| `decodability.csv`, `decodability_maps.npz` | Per-cell R² for all 14 targets, best cell, median cell and GAP R². Map keys are `arm/view/stage/target`. |
| `report.json` | Settings, arguments, checkpoint and image hashes, notes and all rows. |

## Reading it

Each intervention changes one raw content control by −/+`--delta` for a test
subject. The anatomy otherwise, the acquisition draws and the subject's
normalization affine stay fixed. Non-lesion factors keep the subject's own lesion
in place. Lesion coordinates move only the lesion, as in the lesion-move audit.
Causal descendants are not propagated.

`gap_survival` is `|mean over cells of the change| / mean over cells of |change|`:

- near 1: every position moves the same way, so the change reaches the GAP
  vector (a volume change such as ventricle size);
- near 0: changes at different positions cancel, so GAP cannot see the factor
  however strongly the map responds (the sulcal corrugation, a lesion moving).

The `input` row applies the same ratio to the image itself. A factor whose input
survival is near zero is GAP-blind before any encoder is involved. Compare it with
the backbone row to see whether the encoder makes the change more or less
coherent than the image. The `global` shift shows how far the content vector
actually moves, in validation-SD units.

Spatial response magnitudes are in units of each channel's spread over the
gallery subjects' unperturbed cells. Compare factors within a stage, not across
architectures.

Per-cell probes fit ridge on one cell's features, choose the penalty on a 75/25
split of validation subjects, refit on all of them and score held-out test
subjects. A factor readable from many cells is spread across the map; one
readable only near its own location is local. A best-cell R² far above the GAP
R² means the information is in the map but lost by averaging. These are
finite-probe readouts on fixed checkpoints, not an identifiability guarantee.

## Verification

```bash
python -m unittest tests.test_encoder_spatial_maps -v
```

Tests run the real command on a tiny checkpoint and check every output, hash
and table size. They compare the per-cell ridge with scikit-learn and check that
it finds a planted informative cell. They check survival on coherent and
cancelling fields, and that interventions replay the dataset image and change
only their own factor's tissue.
