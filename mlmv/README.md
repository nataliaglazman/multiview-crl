# mlmv: multi-level multiview encoder (G / L / S)

Implements the spec "Multi-level multiview encoder for T1/FLAIR" (12 Nov 2026). Status: the synthetic path
(Phase 1 with the Phase 2 model, sampler and losses) is built and tested on CPU, with two renderers (below);
nothing has run at full scale yet. Phase 0 (ADNI metadata, splits, atlas regions), Phase 3 (scanner removal),
Phase 4 (evaluation suite) and the Phase 5 export are not built.

## Two synthetic renderers

The same latents (g, r_R, s, w, site, style, lesions) can be rendered two ways (`synth.renderer`):

- `lattice` (default, `config/runs/phase1/`): a 64^3 cube of Voronoi regions; a fixed random network fills
  each 4^3 cell with its own pattern. Injective by design, so the theory's assumptions hold, but the images are
  a voxel-level texture. This is the first rung: a failure here is a failure of the method.
- `brain` (`config/runs/phase1_brain/`): a fixed procedural brain (`synth/brain_template.py`: tissue classes,
  13 regions on the spec's ADNI scheme) on the ADNI grid, 96 x 112 x 96 at 2 mm. The latents are upsampled
  smoothly and change each tissue's T1 and FLAIR intensity; lesions are white-matter spheres, bright in FLAIR
  and dark in T1; each site adds a smooth bias field shared by both modalities and its own Rician noise; the
  background is zero. Ground truth stays exact, but smoothing, partial volume and noise hide part of each
  factor, so read the leakage matrix against its raw-image reference row as well as the thresholds. A real
  template (the ADNI template's SynthSeg map) can replace the phantom through the same arrays.

At rho = 2.5 on the ADNI grid, erosion and the split band are wide compared with small lobes: in the brain
template the temporal lobes, deep grey and brainstem cannot be split and merge into neighbours, leaving 7
regions (frontal, parietal and occipital on each side, one cerebellum) and 39% of brain positions for L and G.
ADNI's lobes are of the same size, so the same will happen there; see `geometry/eligibility.csv` of a run.

## Run

```bash
# one Phase 1 run on Run:ai (the config path is the only argument; --dry-run prints the command)
bash mlmv/scripts/runai/phase1.sh mlmv/config/runs/phase1/p1_exact.yaml
# locally, with overrides
PYTHONPATH=. python -m mlmv.train.train_levels --config mlmv/config/runs/phase1/p1_exact.yaml --set run.device=mps
# re-score a saved checkpoint (final.pt, last.pt or a kept step_<N>.pt) into <run-dir>/eval/<checkpoint>/
bash mlmv/scripts/runai/evaluate.sh results/mlmv/p1_exact_batchstd step_050000.pt
PYTHONPATH=. python -m mlmv.eval.evaluate_run --run-dir results/mlmv/p1_exact --checkpoint final.pt
# Gate 1 tables from finished runs
PYTHONPATH=. python -m mlmv.eval.phase1_summary --runs results/mlmv/p1_* --out results/mlmv/phase1_summary.md
# tests (CPU, about a minute; the repo's tests/ is shadowed in some envs, so run files by path)
for t in tests/test_mlmv_*.py; do PYTHONPATH=. python "$t"; done
```

Each run writes `results/mlmv/<run.name>/`: `config.yaml` (resolved, with the commit), `geometry/` (`rho.json`,
`region_masks.npz`, `eligibility.csv`), `scalars.jsonl` and `tb/`, `checkpoints/`, and `eval/` (`leakage.csv`,
`checks.csv`, `counterfactual.csv`, `leakage.json`).

## Layout

| Path | What |
|---|---|
| `config/defaults.yaml`, `config/__init__.py` | every default; YAML -> dataclasses, strict keys, `--set a.b=v` |
| `config/runs/phase1/*.yaml` | the 12 Phase 1 runs (variants, f_L curve, centring and objective ablations) |
| `config/runs/phase1_brain/*.yaml` | 5 Phase 1 runs on the brain renderer (exact, centring off, noisy, FLAIR-only lesion, site confounded) |
| `config/runs/phase1_fixes/*.yaml` | 3 cube runs testing fixes for the late collapse and G's set dependence (batch statistics, no expander, both) |
| `data/geometry.py` | erosion, eligibility and merging, half-splits, G region sets, majority downsampling |
| `data/sampler.py` | per-step S positions, L half-splits, G region sets (explicit `torch.Generator`) |
| `synth/three_level.py` | the three-level generator and both renderers (extends `experiments/gl_ident/generator.py`) |
| `synth/brain_template.py` | the fixed procedural brain: tissue classes and 13 regions, built in millimetres |
| `models/encoder.py` | stems, shared trunk, block head, per-position norm, template |
| `models/heads.py`, `models/receptive_field.py` | expanders; theoretical and effective rho |
| `losses/barlow.py`, `losses/infonce.py`, `losses/levels.py` | BT with context centring; InfoNCE with context negatives; per-level pairs |
| `train/train_levels.py` | training loop, logging, rho measurement, checkpoints, final leakage eval |
| `eval/leakage.py`, `eval/phase1_summary.py` | leakage matrix, raw-image reference row, counterfactual; Gate 1 tables |
| `eval/site_probe.py`, `eval/consistency.py`, `eval/metrics.py` | training-time site probe and G CKA; probes |

## Decisions that differ from the spec

- **Trunk reach.** Only the first trunk conv is 3^3 (`model.trunk_spatial_convs: 1`); the rest are 1^3. The
  spec's four blocks of 3^3 convs give a radius of 9.5 feature voxels against `max_rf_radius: 3`. Now 2.5.
- **Normalisation.** Per-position group norm over channels in the trunk (`model.norm: channel`), no norm in the
  stems (`model.stem_norm: none`). GroupNorm's whole-volume statistics make every output depend on every input
  voxel. A per-position norm right after the first conv removes each patch's intensity scale, and with one
  channel per group it outputs exactly zero, so groups need at least 4 channels.
- **Distances are Chebyshev** (`geometry.metric`). The theoretical receptive field is a box, so only an
  L-infinity separation above 2 rho guarantees no shared input (it also implies the Euclidean one). Erosion uses
  a cube and the half-split band is scaled by `||n||_1`; `euclidean` is available.
- **Erosion is against other regions, not background.** Unlabelled voxels take the nearest region's label first,
  so the brain surface is not eroded away (it would remove the cortex from every lobe at 4 mm).
- **Eligibility is a share of split planes** (`geometry.eligible_fraction: 0.5`). A region is eligible when half
  of 256 random planes leave 8 positions a side; the sampler redraws a plane that does not. Requiring every
  plane loses elongated regions (in the brain template, both occipital lobes) to planes across their short axis.
- **Running standardisation** is an EMA (momentum 0.99, bias-corrected) of each step's centred statistics;
  `loss.buffer.size` sets the FIFO used for the effective-rank log.
- **G batch size.** `sampler.G.splits_per_subject: 8` region-set splits per subject per step (64 rows), since one
  split gives Barlow Twins 8 rows; `loss.corr_ema` optionally adds a running cross-correlation.
- **Synthetic decoder.** A 3^3 conv on the lattice, LeakyReLU, a 1^3 conv and a x4 pixel shuffle (a stride-4,
  kernel-3 transposed conv leaves gaps between blocks). It reaches one lattice cell.
- **Synthetic regions.** For a cube number of regions (8), Voronoi seeds are jittered on a 2^3 grid; otherwise
  random with a minimum spacing. At 64^3 and rho 2.5 all 8 regions are eligible.
- **InfoNCE** uses its context negatives in place of centring (`loss.centring` acts on Barlow Twins).
- **Site in the leakage matrix** is R^2 on the 2-d site embedding (what enters the renderer), plus balanced
  accuracy.
- **Under test: the first full runs degraded late.** In p1_exact (cube) and p1b_exact (brain) the gradient norm
  grew 10–150x as the learning rate decayed and every level's match fell in the last part of training; G's
  cross-split CKA fell to 0.54 (cube, from 55k) and to about 0.2 (brain, from 10k) while G's match stayed high.
  Suspects: the running standardisation (a dimension whose running variance collapses makes the loss steep) and
  the expander (the loss never sees block variance the expander gives little weight). `loss.standardise: batch`
  and `model.expander: []` test them (`config/runs/phase1_fixes/`); both depart from the spec. Runs now log each
  block's template and deviation RMS and the loss's smallest and mean variance, and keep `step_<N>.pt` model
  checkpoints every 10k steps (`run.keep_every`).
- **Data generation runs on the training device.** Only each subject's random draws happen on the CPU, in a
  fixed order; smoothing, lesion maps, rendering and the brain renderer's Rician noise run on the GPU (0.012 s
  of CPU work per 4-subject step, down from 0.19 s on one thread). The noise comes from a device generator, so
  a subject's noise differs between CPU and GPU runs; latents and noise-free images match to float rounding.
- **Leakage readouts stay where L and G are trained.** L and G are only trained on the eroded region interiors,
  so their values elsewhere are never shaped by a loss (the masks rebuilt at step 5,000 add such a border shell,
  and L's match dips for one step). With the run's geometry, subject codes pool over the union of the eroded
  regions, region codes over each generator region's eroded part (regions with nothing left fall back to their
  full extent and are listed in `leakage.json`), and per-cell probes use only cells fully inside; every block,
  the raw-image reference and the counterfactual use the same readout. Runs scored before this change used
  whole regions; re-score them with `mlmv.eval.evaluate_run`.
- **Raw-image reference.** Each leakage matrix has an extra row (block `raw`, modality `t1+flair`): the same
  probes on simple summaries of both images at each readout (cell: sub-cell and neighbouring cell means; region:
  mean, sd and percentiles; subject: all regions). A reference, not a bound: a trained block can beat it.
