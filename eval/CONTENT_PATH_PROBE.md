# Frozen content/style pathway audit

Follow-up to `eval.pooling_probe`: find where ventricular size and lesion-location
readout changes between the spatial encoder maps and the decoder's actual inputs.
Uses the same target definitions, dataset builder, split and probe fitting code.
This is diagnostic probe fitting, not supervised representation training.

```bash
python -m eval.content_path_probe \
  --run-dir results/synthetic/synthetic-clean-content-causal-ident-vent-12-4-3 \
  --num-samples 512 --batch-size 8 --grids 8 --causal iid
```

Repeat with the baseline `--run-dir` and identical arguments. Start with
`--num-samples 128` for a smoke run. `--grids 1 8` includes global readouts as well.
Default split seed is 0, matching `pooling_probe`. IID removes SCM/hierarchical
factor correlations but can differ from training. `--causal match` retains the
run's training distribution, where predicting a correlated factor does not
establish direct representation of that factor. Cross-run comparisons require
matching renderer settings, sample splits and consideration of checkpoint age.

## Captured tensors

Content:

1. `pre_norm`: encoder output after the model's configured latent mask, before
   `content_norm`. This matches the previous pooling test's tap.
2. `post_norm`: output of the actual normalization module, selected content channels.
3. `pre_quant`: output of the content codebook's `conv_in` projection. Including
   this stage separates projection effects from quantization effects.
4. `decoder_input`: the actual quantized tensor passed to the decoder, using **all
   embedding coordinates**. No code-ID reconstruction or encoder-channel slicing.

Style:

1. `pre_norm` and `post_norm`: complementary encoder channels at the same stages.
2. `bottleneck`: actual style after spatial bottlenecking.
3. `pre_quant`: style projection, when style quantization is enabled.
4. `decoder_input`: actual style argument supplied to the decoder. This is continuous
   when style is not quantized, and quantized otherwise. It is captured before any
   interpolation/FiLM processing internal to the decoder.

One real reconstruction forward per batch captures all stages. The pre-norm GAP
is checked against forward's own pooled output. Post-norm style must match the
bottleneck input, and quantized outputs must exactly match decoder arguments.
Registered parameters and buffers are hashed before and after extraction to
verify that codebook EMA state and weights did not change. Hooks/wrappers are
removed even after a failure. Checkpoints are loaded strictly and never written.

Currently supports **one VQ level**, content/style split at level 0, style injection,
and stable hard `fixed` or `learned` masks. Shared or separate encoders and codebooks
are supported. Unsupported architecture settings fail rather than approximating
the decoder input. The shared checkpoint loader must reconstruct the architecture
exactly; strict state loading rejects missing or unexpected tensors.

## Readouts and interpretation

Regional **mean pooling only**, retaining spatial order. No new anatomical mask or
foreground-patch selection is introduced. Smaller style maps are never upsampled
to manufacture features: a 1³ bottleneck yields a global descriptor even when grid
8 was requested. Both requested and actual grids, native shapes and feature counts
are reported. Reduced style resolution is itself a bottleneck to interpret, not a
matched-resolution comparison. Non-divisible grids fail explicitly.

All stages use the same disjoint 60/20/20 train/validation/test subjects. Independent
ridge and RBF probes are fitted at each stage, with train-only feature/target
standardization and validation-only hyperparameter selection. No PCA. Shuffled
controls never use test labels for fitting or selection. Targets are ventricle size,
lesion x/y/z, brain size and cortical thickness. Lesion coordinates are **latent
locations**, not physical centroids, lesion presence or size; sphere lesions only.

Look for changes from:

- `pre_norm → post_norm`: normalization and its effect on probe accessibility.
- `post_norm → pre_quant`: codebook projection, including changed feature dimension.
- `pre_quant → decoder_input`: discrete quantization in the same embedding space.
- Content remaining strong at `decoder_input`: motivates decoder swaps to test
  whether reconstruction actually uses the information available in content.

Strong style probes do not prove that information moved from content: both may
encode it. A drop in finite-sample R² is not proof of information destruction.
Differences also reflect descriptor geometry, resolution and probe capacity.
This test assesses accessibility, **not** decoder reliance or causality of the
training loss. No training change is automatically recommended by the script.

## Outputs

A fresh `content_path_probe_TIMESTAMP` directory contains:

- `scores.csv`: individual factor R², shuffled controls, selected hyperparameters,
  feature counts, changes versus the preceding stage, and paired 95% bootstrap
  intervals. Pre-norm uses itself as reference and has zero delta.
- `summary.json`: the same scores, run/dataset settings, checkpoint step, tensor
  shapes, actual pooling grids, selected encoder channels and state-preservation check.
- `predictions.npz`: test predictions for each view/block/stage/probe, all target
  values and subject split indices, allowing paired follow-up comparisons.

Intervals use 500 subject resamples, conditional on the fitted probes; they exclude
training/split uncertainty and do not correct multiple comparisons. Kernels use
O(N²) RAM and dense solves roughly O(N³) work. Temporary feature arrays are disk-backed
and removed afterward. This runs decoders, so it needs more GPU memory than the
original pooling test; lower `--batch-size` if necessary.

```bash
python -m unittest discover -s tests -p 'test_content_path_probe.py' -v
```
