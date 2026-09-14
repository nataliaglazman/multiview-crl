# Ventricle content/style swap test

Run from the repository root in the training environment:

```sh
python -m eval.ventricle_routing \
  --run-dir /path/to/run \
  --num-samples 64 --batch-size 2 --eps 0.25
```

The run directory needs `settings.json` and `vqvae_model.pt`. Use `--checkpoint`
for a different checkpoint, `--device cpu` for CPU inference, and `--out-dir`
to choose the destination. Each forward contains four volumes per subject
(two ventricle states × two modalities). Checkpoints are never written.

The diagnostic uses the saved synthetic settings and requires an injected,
nonempty style pathway. Base anatomies follow the run's distribution by default
(`--causal match`); `--causal iid` is an optional distribution-shift control.
Use `--old-generator` for checkpoints trained with the pre-7ac56a3 renderer;
match the generator version to training before interpreting the result.

For each subject, A/B change only `z_content[1]` by minus/plus `eps`. Other
content factors, deformation/fissure/lesion fields, style latents, and acquisition
noise seeds stay fixed. Both states use the original sample's normalization
affine, preventing normalization from creating a global cue. In `per_sample`
or `shared` runs this deliberately differs from independently re-normalizing A/B.
Both states share one forward, so on-the-fly channel masks cannot differ between
donors; those masks may still depend on the other subjects in the batch.

The four reconstructions are `AA`, `BA`, `AB`, `BB`, with **content donor first**.
Swaps are within modality and exchange all decoder-bound quantized content tensors
while holding injected style fixed. The replay captures actual quantizer outputs:
forward's straight-through arithmetic can round differently from an ID lookup.
All decoder calls keep the original batch shape and tensor memory format, avoiding
batch-size changes in GPU convolution kernels. During the test, TF32 is disabled,
cuDNN benchmarking is off and deterministic cuDNN algorithms are requested; the
caller's backend flags are restored afterwards.

Each `AA` and `BB` endpoint must reproduce its forward reconstruction to RMS error
at most `1e-6 + 1e-4 * reference_RMS`, checked separately for each subject/state/view.
Larger errors stop the test with diagnostics. Isolated pointwise errors near zero
no longer abort an otherwise accurate replay. The maximum absolute error and RMS
error are both saved.

For each subject/view, the sum of the two endpoint error norms inside the ventricular
ROI is divided by the input intervention norm. This bounds those endpoint errors'
contribution to the projected difference gain. If this ratio exceeds **0.01**, the
row has `valid_routing=false`: its raw scores remain in the CSV, but it is excluded
from summaries of gains, response ratios, and joint fidelity. Invisible input
changes are also unresolved. Summary coverage reports both `n_valid_input` and
`n_valid_routing`. This is a numerical resolution criterion, not a significance
test or a bound on all errors in the hybrid reconstructions.

Separate modality codebooks and multiple
levels are supported. At multiple levels, fine content codes may already include
conditioning from coarser style-dependent reconstructions.

## Read the output

`summary.json`, `samples.csv`, and FLAIR slice panels in `examples.png` are saved
under `ventricle_routing_match_eps0.25/` in the run directory by default.

The ROI is the actual tissue-label change dilated by one voxel to include the
renderer's blur. A response gain is its projection onto the rendered A→B image
change in that ROI, divided by the input change's squared norm. Identity gain is
1; no response is 0; negative gains mean the response reverses direction.

| Metric | Meaning |
|---|---|
| `joint_gain`, `joint_cosine`, `joint_relative_error` | Fidelity of `BB − AA` to the input change; ideal values are 1, 1, 0 |
| `content_at_style_a_gain` / `content_at_style_b_gain` | `BA − AA` / `BB − AB`: changing content in each fixed style context |
| `style_at_content_a_gain` / `style_at_content_b_gain` | `AB − AA` / `BB − BA`: changing style in each fixed content context |
| `content_mean_gain`, `style_mean_gain` | Averages over the two donor contexts; add to joint gain per subject |
| `interaction_rms_ratio` | Strength of `BB − BA − AB + AA`, relative to the input change |
| `aa_roi_mae`, `bb_roi_mae` | Absolute endpoint reconstruction quality near the ventricle |
| `endpoint_replay_rms`, `endpoint_replay_max_abs` | Replay discrepancy from the original forward |
| `endpoint_error_to_input_ratio` | Local endpoint error relative to the ventricle intervention; must be ≤0.01 for routing summaries |

With good joint fidelity, high style gain and low content gain support decoder
reliance on style for the ventricle change. If joint fidelity is poor, weak pathway
effects are inconclusive. Strong interactions indicate context-dependent reliance;
the averages are not unique causal shares. Summary values are medians, which need
not preserve the per-subject addition identity. Inspect rows and example panels.

Input-invisible interventions (e.g. squash saturation, subvoxel changes, lesion
occlusion) produce null gains and are counted in coverage. Native content and
pre/post-quantization style response RMS, quantized content response RMS, and
content code-change fractions help locate lost sensitivity. Their scale and width
differ, so RMS values are not comparable information scores across blocks.

This tests decoder reliance under intervention, not the historical reason that
training chose a pathway. Hybrids can be off the learned joint manifold. Repeat
with `--eps 0.125` and multiple base anatomies before drawing a routing conclusion.

## Verification

```sh
python -m unittest discover -s tests -p test_ventricle_routing.py -v
```

Controls cover known content/style/mixed routes, nonlinear interactions, a constant
decoder, invisible input changes, real rendering with frozen normalization and
noise, partial batches, CLI output, and real VQ-VAE endpoint round trips with
multiple levels and separate modality codebooks. Regression controls reproduce
straight-through cancellation and a batch-dependent decoder, while retaining the
endpoint validity check. Further controls cover sparse CUDA-sized replay drift,
unresolved local signals, substantial replay errors and backend-flag restoration.
Model parameters and buffers are checked for changes,
and temporary capture hooks are removed even on failure. Tests omit unrelated ADNI imports when loading model/dataset
code, allowing these controls to run without MONAI or pandas.
