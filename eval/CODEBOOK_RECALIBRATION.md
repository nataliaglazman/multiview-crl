# Content-codebook recalibration experiment

Test whether a conservative content-codebook refit improves held-out ventricular
response fidelity while retaining reconstruction quality. No anatomical supervision
is used for fitting. Only the content embedding buffers of a **disposable model copy**
change; the original checkpoint and model state remain untouched.

```bash
python -m eval.codebook_recalibration \
  --run-dir results/synthetic/synthetic-clean-content-causal-ident-vent-12-4-3 \
  --fit-samples 256 --num-samples 64 --batch-size 4 \
  --sites-per-subject 256 --iterations 10 --eps 0.25 --causal match
```

Use the same command on the baseline, changing only `--run-dir`. A quick smoke run:
`--fit-samples 16 --num-samples 8 --iterations 3`. This experiment runs several
forwards per held-out subject; it does not resume training. Lower batch size for
GPU memory; `--chunk-size` bounds the centroid-assignment distance matrix.

## Calibration

- Train and test renderer splits have different seeds; overlap is checked.
- Default `--causal match` uses the run's observational distribution for both
  calibration and held-out base subjects. `--causal iid` deliberately changes both
  to independent factors. Previous audits defaulted to IID: these numbers are
  comparable **within this experiment**, not automatically identical to old reports.
- For `fixed_reference`, normalization statistics are estimated from the training
  split only and shared with the held-out split and both model conditions. This is
  another reason old standalone test-split-normalized reports can differ.
- The frozen encoder/projection supplies native pre-quantization content features
  from ordinary training images. Uniformly sample up to 256 spatial sites **per
  subject per view** by default, across the entire map. No ventricle masks, factor
  labels, interventions or reconstruction losses select the sites or fit centers.
- A shared content codebook receives equal contributions from T1 and FLAIR;
  separate content codebooks receive their respective view's points.
- Run fixed-count Lloyd centroid updates, initialized with the existing code vectors.
  Entries with no assigned training points retain their previous vector. Entries
  are never randomly reinitialized or deliberately permuted. Assignment boundaries
  and decoder meaning can still change when centers move.
- Select the lowest **training** distortion iterate, including the original as
  iteration zero. Held-out images and metrics are not accessed until fitting ends.
  If nothing improves, retaining the original is a valid reported outcome.

This is a batch codebook-fit diagnostic, not a recreation of historical EMA updates.
Only `.embed` changes. EMA counts/averages, projection, encoder, decoder, style
codebooks, normalization parameters and all other state stay unchanged. The candidate
runs in eval mode and must not be resumed as a training checkpoint with stale EMA state.

## Held-out evaluation

Both conditions see exactly the same held-out subjects and preprocessing:

1. **Natural reconstruction:** per-subject/view foreground MAE and RMSE, on raw
   unclamped predictions. Also save prediction-change MAE. This is pixel fidelity,
   not the full training objective or a lesion/ventricle-specific reconstruction score.
2. **Ventricle intervention:** reuse `ventricle_quantizer_audit` on both conditions,
   with the same z1 +/- eps pairs, fixed style/noise/normalization, and repeated-low
   numerical replay. Report quantization error, response cosine/relative error/norm
   ratio, changed assignments, regional responses and code usage for content and
   quantized style. Continuous style has no code-assignment audit.
3. Verify that input and pre-quantization response magnitudes agree between conditions.
   Hash original state before/after and candidate state before/after evaluation.
   Explicitly check that the only changed state keys are content embedding buffers.

All comparison deltas are **recalibrated minus original**. Confidence intervals are
paired subject bootstraps (500 draws), conditional on the fitted codebook. They exclude
calibration-subject/initialization uncertainty and multiple-comparison correction.
Undefined cosines (zero response) remain missing, with paired valid counts reported.
Check numerical replay errors in the individual condition CSVs before interpreting
small quantized responses. Norm ratios are not information-retention fractions.

## Reading the result

- Lower training distortion alone is expected and insufficient.
- Better held-out quantizer fidelity **and** stable/better reconstruction support
  a codebook-fit contribution to the original problem.
- Better quantizer fidelity but worse reconstruction means the frozen decoder may
  rely on the original code geometry; it does not justify deploying the new centers.
- No improvement does not rule out all codebook fixes: the test keeps cardinality,
  initialization, empty entries, uniform sampling and frozen projection fixed.
- This is not a causal test of the original contrastive objective, nor a proof that
  reconstruction now uses content instead of style. No decoder swaps or automatic
  adoption/deployment are performed.

## Files

Creates a new `codebook_recalibration_TIMESTAMP/` directory:

- `summary.json`: settings, train/test seeds, train-only normalization, centroid fit
  trace and selected iteration, changed keys, state checks, held-out paired summaries.
- `reconstruction.csv`: original/recalibrated pixel errors for each held-out subject.
- `original_responses.csv`, `recalibrated_responses.csv`: complete intervention metrics.
- `response_comparison.csv`: paired changes and intervals by view, region and metric.
- `diagnostic_centers.npz`: original and selected vectors, shaped [entries, embedding
  dimension]. These are diagnostic artifacts, **not a model checkpoint**.
- Optional example maps from each condition when `--examples` is positive.

Supports the single-level hard fixed/learned split architectures accepted by
`content_path_probe`, including separate encoders/content codebooks/style codebooks.
Original model checkpoints are never overwritten; no recalibrated checkpoint is exported.

```bash
python -m unittest discover -s tests -p 'test_codebook_recalibration.py' -v
```
