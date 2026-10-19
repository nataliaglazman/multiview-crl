# Frozen VQ-VAE spatial bottleneck

This experiment learns a continuous vector from the full spatial content map of
a saved VQ-VAE. A shared convolution/flatten/MLP readout produces `z`; a shared
map decoder receives **only z**. Neither the source encoder nor its codebooks
are optimized. No anatomical labels or interventions enter training or selection.

The objective is

    reconstruction_weight * mean(((R(z) - H) / scale)^2)
  + alignment_weight      * mean((z_view1 - z_view2)^2)
  + variance_weight       * mean(relu(1 - sqrt(Var_subject(z) + 1e-4)))

Variance is computed separately in each view. There is no decorrelation penalty
on the code: anatomical variables can be correlated. This loss is a practical
preservation objective, not an implementation of a causal identifiability theorem.

### Reconstruction weighting

The plain mean counts every map entry equally, so a small local feature such as
a lesion (a handful of the map's positions) barely moves the loss, while a
global factor changes every position. `--recon-weighting` multiplies the
squared error by label-free weights, renormalised to mean 1 so the term keeps
its size against alignment and variance:

- `none` (default): the plain mean above, unchanged.
- `position`: each map entry's error is put in units of that entry's own spread
  around the template across training subjects and both views, so entries that
  rarely vary (deep white matter) weigh more and highly variable ones less. The
  spread is floored at `--position-floor` (default 0.25) times its channel's
  scale, which caps a weight at 1/floor² before renormalising; otherwise
  near-constant entries such as background would dominate. Fitted once on the
  training maps.
- `deviation`: each position gets weight `1 + --deviation-strength * d`
  (default strength 1), where `d` is the RMS over channels of that position's
  standardised departure from the template in the map being reconstructed,
  renormalised to mean 1 within each map. Positions where a subject departs
  from the average (a lesion, but also ventricle and brain edges) weigh more.

The weights shape training and checkpoint selection, which uses the same
weighted objective on validation maps. Every reported metric, and everything
`spatial_bottleneck_eval` computes, stays unweighted, so runs with different
weightings remain comparable. The weights are not stored in `bottleneck.pt`;
`report.json` records the choice under `training.recon_weighting` (for
`position`, also the fraction of entries held at the floor and the weight range).

## Run

From the repository root, in the normal PyTorch/MONAI environment:

~~~sh
python -m training.spatial_bottleneck_experiment \
  --run-dir /absolute/path/to/synthetic-vq-run \
  --checkpoint vqvae_model.pt \
  --out-dir /absolute/path/to/new-bottleneck-run \
  --level 0 --latent-dim 16 --width 8 --hidden 128 \
  --train-samples 256 --val-samples 64 --test-samples 64 \
  --batch-size 16 --steps 2000 --device cuda
~~~

The source run needs its saved `settings.json` and checkpoint. The output
directory must be new. Repeat with different `--latent-dim` values and separate
output directories for a capacity sweep. Choose dimensions using validation;
do not select a dimension from its test results.

The CLI currently supports synthetic checkpoints. It reproduces saved generator
options and uses distinct train/validation/test sample seeds with the original
SCM seed. The readout's training subjects can overlap the source encoder's
training subjects; validation/test are held out from readout optimization.
For ADNI, `extract_content_maps` accepts the existing paired-volume DataLoader
contract; the caller must provide subject-disjoint splits and fixed preprocessing.
ADNI dataset/launcher integration is not part of this draft.

## Representation and normalization

- Maps are continuous, native-resolution outputs of `VQVAE.forward`, after
  content normalization where present and before codebook projection/quantization.
  These differ from the pre-content-normalization features used by the trainer's
  pooled contrastive path. The report records the extraction point.
- Only the selected content channels are retained, with separate saved indices
  for each view where the source has per-view selectors. Batch-dependent
  `onthefly` masks are rejected. Fixed and deterministic evaluation-time learned
  masks are supported. Both views must have the same nonzero content width.
- A common spatial template and per-channel residual scale are fitted on the
  **training subjects only**, across both views. The readout sees standardized
  deviations; the decoder restores the template and scale. This fixed template
  is not a subject-dependent bypass. Validation/test never refit these buffers.
- There is no spatial averaging in the readout. Its fully connected layers grow
  with the native map size; `--width` and `--hidden` control parameter cost.
  `--max-map-gib` caps the combined in-memory map banks (default 1 GiB), not the
  model/optimizer memory. Small extraction batches reduce device memory usage.
- The strict checkpoint check permits unused contrastive projector weights but
  rejects other architecture mismatches. Runs with `split_encoder_norm` are
  rejected until the shared VQ checkpoint loader restores that option.

## Outputs

- `bottleneck.pt`: validation-selected readout/decoder, normalization buffers,
  architecture, source checkpoint hash and content-channel indices. It excludes
  the frozen source model. Reload with `load_bottleneck(path, device)`; `encode`
  accepts unnormalized selected maps and `decode` returns maps in original units.
- `report.json`: configuration, split identities/seeds, validation history,
  selected step (0 means initialization beat trained candidates), and test metrics.
- `test_outputs.pt`: paired small vectors, raw GAP vectors for the same subjects,
  subject IDs and a spatial map of normalized reconstruction error.

Metrics include reconstruction in raw and normalized units, per-view errors,
error relative to the fixed training-template predictor, matched/mismatched
cross-view code errors, and code standard deviations across subjects. The
original map disagreement is also reported: a frozen encoder may leave modality
differences that conflict with vector agreement and exact map reconstruction.
Only validation selects checkpoints; test metrics are computed after selection.

Inspect spatial errors as well as averages: low aggregate error can conceal a
lost lesion. These outputs assess map compression, not individual causal-factor
recovery; `eval.protocol.spatial_bottleneck_eval` (below) does that. A dimension
small enough to represent selected scalar factors may be insufficient for
additional shared spatial fields in the generator.

## Factor recovery

~~~sh
python -m eval.protocol.spatial_bottleneck_eval \
  --bottleneck-dirs /path/to/bottleneck_d16 /path/to/bottleneck_d16_s1 /path/to/bottleneck_d32 \
  --out-dir /path/to/new-eval-dir --test-samples 512 --device cuda
~~~

This regenerates the bottleneck's own subjects from the source run's settings,
collects their factor labels and asks, per factor, how well each representation
predicts it on held-out test subjects:

| Representation | Role |
|---|---|
| `map` | full content map (the bottleneck's normalisation): what was available before compression |
| `patch2x2x2` | channel means on a coarse grid: the existing local readout |
| `gap` | channel means: the existing global readout |
| `pca{d}` | top-d principal components of the map, at each bottleneck's size: linear compression |
| `z_init:RUN` | the readout at its initialisation (same seed and normalisation): the floor |
| `z:RUN` | the trained code |

- Standardisation, PCA and probe weights are fitted on training subjects only.
  Each factor's ridge penalty / RBF width is chosen by validation R², and test
  labels are used once. Both a linear and an RBF kernel probe are reported;
  the full map is probed exactly through its Gram matrix.
- Before scoring, the regenerated test maps must reproduce the saved test codes
  and GAP vectors, and the source checkpoint and settings must match their
  recorded hashes; otherwise the script stops. Use `--run-dir` if the source run
  moved.
- `--test-samples` / `--val-samples` add further held-out subjects from the same
  split seeds. The readout is frozen, so this only costs rendering, and with
  64 test subjects per-factor intervals are wide. Under `fixed_reference`
  normalisation a split trained with fewer than 64 subjects cannot be extended
  (the parity check fails and says so).
- Several bottleneck directories trained on the same source and splits are
  evaluated together. Different seeds at one size show seed variability;
  different sizes add a "latent size chosen by validation" line.
- Targets, in up to five blocks:
  - **content:** the content latents (burden mode's two unrendered latents are dropped).
  - **lesion volume:** the rendered lesion's partial-volume voxel count, when it
    varies (it is constant with one fixed-size lesion).
  - **lesion centre:** `centroid_x/y/z`, where the lesion actually sits, re-rendered
    from the latents (the same centroid as
    `eval.encoder.encoder_target_protocol.sample_targets`).
    Under `wm_interior` placement the lesion latents are quantiles through each
    subject's own white matter, so a probe must redo a subject-specific count
    to read them; read the centre rows for whether a representation knows
    where the lesion is. The lesion blocks are omitted, with the reason printed,
    when a subject has no rendered lesion or the generator has no single sphere lesion.
  - **shared style:** a style factor identical in both views (gain and bias
    under `synthetic_shared_gain_bias`). This is shared information, so a
    cross-view code is expected to carry it.
  - **per-view style:** leakage into the code, so lower is better.
- The header gives each code's reconstruction error over the training template
  on the evaluated test subjects, next to the same ratio for the map's top-d PCA
  projection: the linear reference at the same size.
- `--burden-images` scores every representation on images with one lesion per
  subject whose volume varies as well as its position (the generator's burden
  mode with one lesion; anatomy and style come from the same latents). The frozen
  encoder and bottleneck are applied unchanged, which is out of distribution for
  them: they were trained on one lesion size. Lesion volume and centre are read
  from the same images. A further table scores the centre on the larger and the
  smaller half of the test lesions, because a faint lesion's centre is hard to
  read whatever the representation. Parity is still checked on the bottleneck's
  original subjects, rendered as in training.
- Outputs: `factor_recovery.json` (everything, including each factor's chosen
  hyperparameters and the parity record), `factor_recovery.csv` (one row per
  representation, probe and factor, with per-view R²) and `comparisons.csv`
  (`z` minus each baseline, with a paired bootstrap interval over test subjects
  and the share of draws at or below zero). `--save-predictions` adds
  `test_predictions.csv`: every test subject's prediction per representation,
  probe, view and target, indexed by its position in the test split (the index
  the lesion audits use). `eval.lesion.lesion_miss_location --predictions` reads
  it to compare decoding on the lesions a reconstruction keeps and loses.

Read `z - gap` and `z - patch` for local factors (lesion centre, temporal
atrophy, sulcal widening), `z - pca{d}` for whether the learned readout beats
linear compression at the same size, `z - map` for what compression lost, and
`z - z_init` for what training the readout bought. The bootstrap holds probe
fits fixed; use several bottleneck seeds for training variability. Under
`--synthetic-causal` a factor can be predicted through correlated anatomy, so a
lesion R² is not on its own evidence that the lesion's position is encoded; that
needs the intervention test below.

## Factor interventions

~~~sh
python -m eval.protocol.spatial_bottleneck_interventions \
  --bottleneck-dirs /path/to/bottleneck_d16 /path/to/bottleneck_d32 \
  --out-dir /path/to/new-intervention-dir --num-subjects 128 --eps 0.5 --device cuda
~~~

This asks whether a code moves when only one factor moves. Each of the
bottleneck's test subjects is rendered at `z_j - eps` and at `z_j + eps` with
every other latent held: the other content factors, the deformation and fissure
fields, the lesion draws, both style vectors and the render seed, so the bias
field and the noise are identical in both renders. The frozen maps then go
through the representations of the table above (`map`, `patch2x2x2`, `gap`,
`pca{d}`, `z_init:RUN`, `z:RUN`), in the units their probes use, and the
response is the difference between the two renders.

- **Subjects.** The first `--num-subjects` of the regenerated test split
  (default: the bottleneck's test count; more extends the split, as
  `--test-samples` does above). The parity checks above run first, and each
  subject must re-render bit for bit from its latents.
- **Targets.** Every content factor. The style factors are a leakage check: each
  moves in both views. Gain and bias are identical in both views under
  `synthetic_shared_gain_bias`, so a cross-view code is expected to carry them.
  The deformation and fissure fields give the nuisance null. Cost is about two
  renders per subject and target; `--factors` restricts the list.
- **Two nulls, biased in opposite directions.** The render null (the same latents
  with another render seed) is what cross-view agreement teaches a code to
  ignore, so it flatters `z`. The nuisance fields are shared by both views, so a
  cross-view code keeps them, which flatters the frozen readouts. Under
  `--synthetic-clean-content` the renderer zeroes those fields, so there is no
  nuisance null; read the scale-free metrics.

Per representation and factor (tables average the two views):

| Metric | Reading |
|---|---|
| `snr_render`, `snr_nuisance` | mean response norm over the null's, with a paired bootstrap interval over subjects. Both renders share the render seed, so 0 is no response and 1 is as large as redrawing the noise |
| `snr_render_per_voxel` | lesion coordinates only: `snr_render` per voxel the rendered lesion centre actually moved |
| `pop_snr`, `consistency` | whether subjects share a response direction: a code with a lesion coordinate does, a local change in the map does not |
| `recall`, `id_acc`, `mistaken for` | nearest-centroid accuracy at naming which content factor moved from one subject's response (chance 1/J), and the factor a factor's moves are most often taken for |
| `alias` | \|cos\| between the mean responses of two content factors, against the floor two unrelated directions reach at this N |

A subject whose images an intervention leaves bit-identical was not measured for
that factor. Its response is set to zero, the per-sample SNRs average over the
measured subjects only, and `active_frac` gives the share that moved.

With a single sphere lesion and nine content factors, every content and nuisance
intervention also records the rendered lesion centre at both endpoints
(`sample_targets`). The header gives each intervention's mean centre shift per
axis, its mean displacement in voxels and how many subjects' lesions crossed the
midline. Under `wm_interior` the lesion latents are not coordinates. They are
conditional quantiles (x, then y given x, then z given x and y) through each
subject's own white matter, so:

- a nudge can leave the centre on its voxel: an unchanged image, not a blind code;
- a lesion near the median x quantile can jump hemispheres;
- moving `lesion_x` also shifts y and z, and an anatomy intervention can move the
  lesion, so some confusion among these comes from the generator, not the code.

Read the lesion blocks first. The code encodes the lesion if its `snr_render`
(or `snr_render_per_voxel`) is clearly above `z_init`'s (what an untrained
readout already passes) and `gap`'s (expected near zero), next to `map` (the
ceiling), and if its recall is well above chance with low aliasing to the
anatomy factors. A code that barely moves when only the lesion moves, yet
predicts the lesion well in the factor evaluator, got that R² through correlated
anatomy. Interventions that change no image (`unused_*` under the burden target,
the nuisance fields under clean content) are listed and left out of
identification and aliasing.

Outputs: `interventions.json` (everything, including the subjects, their render
seeds and the parity record), `interventions.csv` (one row per representation
and factor), `alias.csv`, `confusion.csv` and `lesion_moves.csv` (per tracked
intervention and subject: the centre at both endpoints, the displacement in
voxels, a midline-crossing flag and whether the image changed). Full-map
responses are held in memory; `--max-response-gib` caps them.

~~~sh
python tests/test_spatial_bottleneck_interventions.py
~~~

The tests check that each intervention moves one latent and holds the rest,
that the readouts match the factor evaluator's features (PCA through its Gram
matrix), that a GAP-invisible spatial change moves the map, the patch grid and
a code but reads exactly zero in GAP, that SNRs average over measured subjects
and scale by the lesion's displacement, that a factor the renderer ignores gives
exactly zero response everywhere, and that a checkpoint-to-report run uses the
factor evaluator's subjects, writes consistent lesion endpoints, and stops on a
tampered test output or a renderer that no longer reproduces a subject.

## Validation

~~~sh
python -m unittest discover -s tests -p test_spatial_bottleneck.py -v
~~~

Tests cover spatial sensitivity when GAP is constant, training-only normalization,
variance across subjects, decoder isolation, fixed and per-view learned masks,
frozen encoder/codebook state, a learned toy spatial compression task, and a real
small VQ-VAE checkpoint-to-report run including reload and output protection.
Reconstruction weighting is tested for: weights with mean 1, the floor and the
(scale / spread)² ratio; deviation weights peaking where a map departs; the
default path matching weights of one exactly; the weighted validation objective
used for selection while reported metrics stay unweighted; flag validation; and
position- and deviation-weighted runs that record their choice, reload, and pass
the factor evaluator's parity check.
The factor evaluator's tests check kernel ridge and PCA against sklearn on
explicit features (including PCA's reconstruction ratio), that test labels never
move a fit or a selection, that a GAP-invisible spatial factor is read from the
map and PCA but not GAP, and a checkpoint-to-factor-report run with extended test
subjects, shared gain/bias, per-subject lesion centres matching `sample_targets`,
a burden-mode run (rendered volume tracking the burden latent, the centre split
by lesion size, parity on the original subjects), a tampered test output and
mismatched splits.
