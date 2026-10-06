# Lesion branch: findings and handoff (2026-10-06)

Handoff for continuing the work. Nothing here is committed. All numbers are linear (ridge)
R² unless stated. "T1 / FLAIR" means the probe read view 1 or view 2.

## The question

In the encoder-only multiview contrastive model (`training/main_conv_synthetic.py`), can a
*fine, local* factor be encoded as a scalar in the global code, rather than only being decodable
from a spatial feature map? The test case is the synthetic lesion's position, lesion_x/y/z: one
fixed-size lesion per subject whose only varying property is where it sits.

## Background (established before this session)

- **GAP cannot see a moving lesion.** GAP of translation-equivariant features is invariant to
  where a fixed-size blob sits. Measured: moving the lesion shifts the loss-facing vector by 0.026,
  against 0.28 for ventricle size. InfoNCE *would* reward a lesion code (−29..−43% loss).
- **Each kind of local factor needs its own pooling:**
  - Amounts (volume, burden): GAP is fine.
  - Fixed-site signed amplitude (sulcal pattern): a learned spatial template, i.e. a matched filter.
  - Variable-site position (the lesion): coordinate-valued pooling, i.e. spatial soft-argmax.
    For a template in Gaussian noise with a flat prior, soft-argmax is the posterior-mean position.
- **`--global-pool attention`** (`models/attention_pool.py`) already existed as the
  position-capable arm. Its Fourier positions enter keys and values, so it contains soft-argmax.

## Main conclusion so far

Position-capable pooling is necessary but not sufficient. With an easy, shared, global factor
(brain size) present, InfoNCE never learns the lesion. This is **feature suppression** (Chen, Luo &
Li, NeurIPS 2021). A dedicated lesion branch with its own loss is **captured** by brain size
instead, and on real data it stays captured even after the architectural fixes below. The live
candidate fix is a **decorrelation penalty** that charges the branch for duplicating what the
content block already encodes. It worked in toys; the real run is in progress.

## Code added (uncommitted)

| File | What |
|---|---|
| `models/keypoint_pool.py` (new) | `KeypointPool3d`. Per head: 1x1-conv logits, softmax over positions, expected (x, y, z). `frame="brain"` confines attention to the brain (logits + log occupancy) and expresses coordinates relative to the brain's centroid and per-axis spread. `norm="layer"` is an optional per-voxel LayerNorm. Logits start small and random (zero init gives every subject the same coordinates and no gradient). |
| `models/multiview_encoder.py` | `lesion_keypoints`, `lesion_norm`, `lesion_frame`, `lesion_proj_dim`. Built LAST, so every other weight equals the model without it. The 3K coordinates are appended AFTER the style units of the global code and marked as content in `content_mask`; patch and unpooled outputs have none. `project_lesion` standardizes the coordinates over the batch, then applies a linear projector. `lesion_maps()` returns the head maps. Content indices are integer arithmetic (no device sync). |
| `training/main_conv_synthetic.py` | Flags `--lesion-keypoints K` (0 = off), `--lesion-norm {none,layer}`, `--lesion-frame {brain,grid}`, `--lesion-proj-dim`, `--lesion-loss-weight`, `--lesion-decorrelation-weight` (default 0 = off). Adds `lesion_contrastive_loss` (own InfoNCE on the projected coordinates) and `lesion_decorrelation` (sum of squared correlations between standardized coordinates and the DETACHED content block). Every eval prints "lesion branch alone" (linear R² per content factor, T1/FLAIR) and each head's effective positions, also saved in `dci_step*.json` under `lesion_branch`. Also carries the user's own uncommitted change: the foreground-positions print in the patch-objective message. |
| `eval/protocol/score_checkpoint.py` | `build_model` restores the branch. `encode_blocks` counts lesion units as content, not style. |
| `eval/lesion/checkpoint_lesion_analysis.py` | Same content/style fix in `batch_features`. |
| `scripts/run_encoder_mps.py` | Passes the lesion flags through; run ID gains `_lesionkp<K>`, `_gridframe`, `_layerlnorm`, `_lw..`, `_dc..`. |
| `tests/test_lesion_branch.py` (new) | 15 tests: equivariance, brain frame, init invariance, scoring layout, trainer path, runner. |
| `training/ENCODER_ARCHITECTURES.md` | New "Lesion branch" section. |

Bugs fixed along the way:
- `attention_spread` used `.double().cpu()`. MPS has no float64, so the attention-pool arm
  would crash at its first local eval. Now `.cpu().double()`.
- The brain frame originally let heads attend over the whole grid. A near-uniform head then reports
  the grid centre, and that point's brain-relative position is a function of brain size.
- The runner's backend check passes raw options without parser defaults, so the trainer now uses
  `getattr` for the loss weight.

## Toy evidence

Fixed translation-equivariant features on a 16³ grid: a registered "anatomy" field plus an easy
shared factor `s`, a lesion blob, and noise. Two views per subject, InfoNCE B=64, τ=0.1.

**Pooling (supervised regression of the lesion centre, 1500 steps):**
- GAP: 0.00.
- Attention without positional encoding: 0.00.
- Shipped attention pool (zero init): 0 until about step 1000, then 0.94.
- Soft-argmax: 0 until about step 750, then 0.97.

**InfoNCE:**
- **No easy factor:** soft-argmax finds the lesion (0.54 at step 250, 0.93 at step 1500).
- **Easy factor present:** nothing learns the lesion in 2000 steps (≈ 0), although the loss is not
  saturated (top-1 0.69). A random-init keypoint block that already carried the lesion (0.67) was
  ignored by the 8-d embedding. Giving that block its own InfoNCE made it *lose* the lesion
  (0.67 → 0.09): the easy factor moves where the attention lands, so the block encodes it instead.

**Label-free fixes that failed:**
- Lower temperature (0.03).
- Implicit feature modification (ε 0.1).
- Negatives reweighted by the model's own global-code similarity.
- A global-to-local predictor (position-conditioned, same-subject negatives). It stalls at the
  position-only optimum; an oracle check (true lesion centre given to the predictor) shows the
  objective itself is fine. The cause is a cold start.

**What works.** Toy v2: features zero outside the brain and the lesion placed relative to the brain,
as in the real generator. Final lesion R² read from the branch, seeds 0 / 1 / 2:

| Arm | Gain-type easy factor | Scaling-type easy factor (brain-size-like) |
|---|---|---|
| Plain branch | 0.01 / −0.01 / 0.12 | 0.04 / 0.10 / 0.30 |
| Per-voxel LayerNorm | 0.76 / 0.77 / 0.74 | 0.05 / 0.00 / 0.50 |
| Brain frame | 0.95 / 0.66 / 0.88 | 0.82 / 0.67 / 0.74 |
| Augmentation (easy factor redrawn per view, branch only) | **0.95 / 0.96 / 0.95** | **0.87 / 0.88 / 0.85** |
| Plain + decorrelation (λ = 1) | 0.90 / 0.90 / 0.90 | 0.79 / 0.64 / 0.87 |

LayerNorm handles gain-type factors and the brain frame handles scaling. Augmentation is best but
needs a transform that changes the easy factor without moving the lesion. Decorrelation needs no
knowledge of the factor type.

The toy scripts live in a session scratchpad that may be cleaned up:
`/private/tmp/claude-502/-Users-nataliaglazman-Desktop-PhD-projects-multiview-crl/61cf4bd9-2a90-4070-a1a1-495d9c5cface/scratchpad/`
(`pool_saddle.py`, `alternatives.py`, `alt_ln.py`, `alt_v2.py`, `global_local.py`, `init_leak.py`).

## Real-data evidence

Setup: the `conv_mlp` recipe (`experiments/encoder_comparison.json`): res 64, 16³ backbone map,
batch 32, τ 0.1, cross-view negatives only, fixed_reference, clean content, wm_interior lesions,
**lesion intensity fixed (default)**. Local MPS, 2000-step runs, eval every 1000. Runs are in
`results/encoder_ablations_mps/runs/`, logs in `results/encoder_ablations_mps/logs/`.

**1. Untrained, the branch already encodes brain size.** 240 validation subjects, mean of 3 init
seeds, brain_size R² T1 / FLAIR; lesion_x/y/z ≈ 0 throughout:

| Backbone | Branch norm | Frame | brain_size |
|---|---|---|---|
| group | none | brain | 0.62 / 0.70 |
| group | layer | brain | 0.50 / 0.56 |
| group | none | grid | 0.71 / 0.76 |
| group | layer | grid | 0.59 / 0.66 |
| layer | none | brain | **0.33 / 0.34** |
| layer | none | grid | 0.69 / 0.71 |

GroupNorm writes each subject's global statistics into every voxel and acts like a gain on the
keypoint logits. The remaining ~0.33 is structural. The generator makes brain_size an **additive
radius shift** (`radii_wm = 0.5 + size_shift`) while cortical thickness and ventricle radius stay
fixed in absolute units. So relative anatomy changes with brain size, and no frame or global scale
augmentation can remove it.

**2. GroupNorm backbone, brain frame** (`..._lesionkp4_v3`, stopped at about step 1100):
- Lesion branch at step 1000: brain_size **0.84 / 0.83** (0.71 / 0.74 at step 0); lesion_x/y/z ≤ 0.16.
- Its InfoNCE fell from 6.8 to 2.1, almost entirely by sharpening brain size.
- Content block at step 1000: lesion_x/y/z 0.10 / 0.03 / 0.06; brain_size 0.90; content→view acc 0.999.

**3. LayerNorm backbone, without projector standardization** (`..._layernorm_v3`, stopped). Cold
start. The coordinates differ between subjects by 6.9e-3 (GroupNorm: 2.1e-2), and the projected
codes have mean cosine 0.9996. The branch's InfoNCE sat at exactly 2·ln 32 = 6.9313. Standardizing
the coordinates before the projector fixed it.

**4. LayerNorm backbone + brain frame + standardization** (`..._layernorm_v4`, stopped at step 1000).
Still **captured**: brain_size **0.88 / 0.86**, lesion_x/y/z ≤ 0.05, lesion InfoNCE 1.62.

**5. In progress: the same plus `--lesion-decorrelation-weight 1`**
(`conv_mlp_s42_mps_t2000_lesionkp4_layernorm_dc1`). Step 0 matched run 4. Read its step-1000 and
step-2000 "lesion branch alone" blocks in
`results/encoder_ablations_mps/logs/conv_mlp_s42_mps_t2000_lesionkp4_layernorm_dc1.log`.
Success means lesion_x/y/z clearly above step 0 while brain_size in the branch drops.

**6. Decorrelation and styled intensity, step 1000 (LayerNorm backbone, brain frame).** None find the lesion.

| Lesion branch R² (T1 / FLAIR) | fixed | styled | fixed + decor | styled + decor |
|---|---|---|---|---|
| brain_size | 0.88 / 0.86 | 0.88 / 0.87 | 0.44 / 0.72 | 0.55 / 0.80 |
| lr_asymmetry | 0.90 / 0.90 | 0.90 / 0.90 | 0.85 / 0.87 | 0.87 / 0.88 |
| cortical_thickness | 0.83 / 0.87 | 0.83 / 0.88 | 0.65 / 0.80 | 0.69 / 0.85 |
| temporal_atrophy | 0.73 / 0.73 | 0.73 / 0.75 | 0.15 / 0.25 | 0.51 / 0.57 |
| lesion x / y / z | ≤ 0.05 | ≤ 0.04 | ≤ 0.12 | ≤ 0.02 |
| head spread | ≈0.16 | ≈0.16 | ≈0.16 | ≈0.16 |

**Root cause: the heads never focus.** In every run, each head's effective positions stay at
0.13–0.16 of the map, i.e. uniform over the brain, at step 0 and at step 1000. The coordinates are
therefore whole-brain tissue moments, which brain size, asymmetry, thickness and temporal atrophy
all move. That gives InfoNCE subject identity without localizing anything. The lesion is about 1
of ~650 brain cells, and nothing rewards a head for sharpening onto it.
- Styled intensity changes nothing at step 1000.
- Decorrelation removes some brain size, but the branch moves to the other global shape factors.

Next levers:
- Force sharp heads: an entropy penalty or a low, fixed softmax temperature.
- Give the backbone a local discovery signal: `--patch-loss-weight` with `--patch-foreground-mask`.
  In deep white matter, the lesion is the main thing that differs between subjects at a fixed position.
- Normative-anomaly logits: attend where a subject's features deviate most from the population mean
  at that position.

**7. Sharp heads (temperature 0.02), with and without the patch loss** (styled, LayerNorm, brain
frame, step 1000). The heads sharpen (0.2–4% of the map), but they lock onto anatomical edges.

| Lesion branch (T1 / FLAIR) | sharp | sharp + patch loss |
|---|---|---|
| lr_asymmetry | 0.92 / 0.91 | 0.92 / 0.89 |
| cortical_thickness | 0.81 / 0.85 | 0.80 / 0.83 |
| lesion x / y / z | ≤ 0.01 | ≤ −0.01 |

**8. Why the patch loss doesn't help: the lesion is anti-aligned across views.** Cross-view patch
retrieval among 32 subjects, on the patch-loss checkpoint:

| Positions | Retrieval |
|---|---|
| All lesion-free positions | 0.361 |
| Deep-white-matter lesion positions, subject without a lesion there | 0.146 |
| Same positions, subject's lesion there | **0.088** |

The lesion has opposite contrast in the two views (T1 0.4 vs white matter 0.8; FLAIR 1.0 vs 0.4;
styled mode too), and the two encoders start identical. So the lesion pushes the T1 and FLAIR codes
of the same patch apart, and alignment suppresses it. Patch codes are only ~40% explained by the
global code (median R² 0.41), so global identity is not the main issue. Published dense SSL works
because its two views are same-modality augmentations, in which a lesion is consistent. **Next test:
same-modality pairs (two FLAIR renders with different noise/bias draws) for the lesion branch and/or
patch loss.** `training/finetune_dino.py` has `--pairing within_modality`; the encoder trainer does
not yet.

**9. In progress: same-modality pairs for the branch.**
- New option: `--lesion-pairing within_modality`. The branch's InfoNCE pairs two intensity-augmented
  copies of FLAIR through the FLAIR encoder; the content block keeps T1/FLAIR.
- Two arms are queued behind the sharp-head arms. Launcher: scratchpad `launch_within.sh`, which waits
  on PIDs 43986/43987. Both use sharp heads (0.02), the LayerNorm backbone, the brain frame and styled
  intensity:
  - `conv_mlp_s42_mps_t2000_lesionkp4_temp0.02_lpwithin_layernorm_lesionstyled`
  - `conv_mlp_s42_mps_t2000_lesionkp4_dc1_temp0.02_lpwithin_layernorm_lesionstyled` (+ decorrelation 1,
    against capture by anatomical edges)
- Read their step-1000 "lesion branch alone" blocks. The FLAIR column is the one this pairing trains.

## Caveats

- **Lesion contrast in T1.** All real runs used the default `--synthetic-lesion-intensity fixed`.
  The lesion is inserted after gain/bias, so T1 lesion contrast ranges 0.06–0.74. The lesion is
  partly FLAIR-only, and InfoNCE alignment penalizes coding it. Re-running with `--synthetic-lesion-intensity styled` changed nothing at step 1000 (see 6).
- **Decorrelation is linear and assumes independent factors.** Under `--synthetic-causal`, lesion
  position correlates with other content factors, and the penalty would strip real lesion signal.
  The current recipe has independent factors.
- **Single seeds and short runs on real data.** Every real number is one seed at ≤ 2000 steps; the
  toy results have 3 seeds.

## How to run

```bash
# local run (MPS); the run ID is derived from the flags
python scripts/run_encoder_mps.py --variant conv_mlp --lesion-keypoints 4 --norm-type layer --lesion-decorrelation-weight 1 --train-steps 2000 --eval-every 1000
```

Tests: use `/opt/miniconda3/envs/adni-analysis/bin/python`. A site-packages `tests` package shadows
the repo's and `lpips` is missing, so first make a shim directory containing:
- `tests/__init__.py` with `import os; __path__ = [os.path.join(os.getcwd(), "tests")]`
- `lpips/__init__.py` with an `LPIPS` class that raises

Then, from the repo root:

```bash
PYTHONPATH=<shim>:. /opt/miniconda3/envs/adni-analysis/bin/python tests/test_lesion_branch.py
```

All encoder tests pass except two that also fail on clean HEAD (launcher drift, unrelated):
`test_encoder_mps_runner.test_cuda_patch_launcher_matches_mps_and_has_separate_outputs` and
`test_separate_spatial_readout.test_launchers_change_only_readout_and_run_id`.

## Suggested next steps

1. Read the decorrelation run (above).
2. Re-run the best config with `--synthetic-lesion-intensity styled`, with and without decorrelation.
3. **If decorrelation works:**
   - Test it alone: GroupNorm backbone and grid frame (the toy says it needs neither).
   - Run 3 seeds and the full 10k-step recipe, then add Run:ai/SLURM variants.
4. **If it fails:**
   - Try a nonlinear version (HSIC between the branch and the content block).
   - Try a larger λ, or more heads (K = 8).
   - On synthetic data only, try the toy's winner: an oracle per-view brain-size re-render for the
     branch, i.e. render the branch's two views with independent brain_size and the same lesion.
5. Housekeeping:
   - These run folders are from crashed or stopped attempts and can be deleted:
     `..._lesionkp4` (crashed: MPS float64), `..._v2` (flawed frame), `..._v3`,
     `..._layernorm_v3`, `..._layernorm_v4` (stopped).
   - Commit when ready. The code follows the repo conventions: pre-commit black/isort pass and
     pyflakes is clean.
