# mlmv: multi-level multiview encoder (G / L / S)

Implements the spec "Multi-level multiview encoder for T1/FLAIR" (12 Nov 2026). Status: the synthetic path
(Phase 1 with the Phase 2 model, sampler and losses) is built and tested on CPU; nothing has run at full
scale yet. Phase 0 (ADNI metadata, splits, atlas regions), Phase 3 (scanner removal), Phase 4 (evaluation
suite) and the Phase 5 export are not built.

## Run

```bash
# one Phase 1 run on Run:ai (the config path is the only argument; --dry-run prints the command)
bash mlmv/scripts/runai/phase1.sh mlmv/config/runs/phase1/p1_exact.yaml
# locally, with overrides
PYTHONPATH=. python -m mlmv.train.train_levels --config mlmv/config/runs/phase1/p1_exact.yaml --set run.device=mps
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
| `data/geometry.py` | erosion, eligibility and merging, half-splits, G region sets, majority downsampling |
| `data/sampler.py` | per-step S positions, L half-splits, G region sets (explicit `torch.Generator`) |
| `synth/three_level.py` | the three-level generator (extends `experiments/gl_ident/generator.py`) |
| `models/encoder.py` | stems, shared trunk, block head, per-position norm, template |
| `models/heads.py`, `models/receptive_field.py` | expanders; theoretical and effective rho |
| `losses/barlow.py`, `losses/infonce.py`, `losses/levels.py` | BT with context centring; InfoNCE with context negatives; per-level pairs |
| `train/train_levels.py` | training loop, logging, rho measurement, checkpoints, final leakage eval |
| `eval/leakage.py`, `eval/phase1_summary.py` | leakage matrix and counterfactual; Gate 1 tables |
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
