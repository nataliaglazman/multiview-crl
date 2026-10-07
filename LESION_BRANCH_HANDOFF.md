# Lesion branch: findings and handoff (updated 2026-10-06)

All code below is committed (through `4ff4bdb`). That includes:
- the residual input: `models/normative_residual.py`, `--lesion-input residual` and its
  normative-model flags, `--lesion-head-init`, and the head-weight lines in each evaluation;
- the lesion-burden generator option: `--synthetic-lesion-target burden` and
  `--synthetic-lesion-count`.

Real-data numbers come from local MPS runs of the `conv_mlp` recipe: 2,000 steps (the recipe has
10,000), one seed each. Toy numbers have 3 seeds where stated. R² is linear (ridge), held out.
"T1 / FLAIR" means the probe read the T1 or the FLAIR view.

## The question

In the encoder-only multiview contrastive model (`training/main_conv_synthetic.py`), can a fine,
local factor be encoded as a scalar in the global code, rather than only being decodable from a
spatial feature map? The test case is the synthetic lesion's position (lesion_x/y/z). Every subject
has one fixed-size lesion, and only its position varies.

## The model and how data flows through it

This is the residual-input configuration (`--lesion-input residual`), the only one that puts the
lesion in the code. Two paths are joined into one 24-unit code, and no gradient passes between them:

```text
each view (T1 or FLAIR), 64³
 ├─ global path ─ that view's encoder → 64 ch × 16³ → spatial mean → MLP 64→100→12
 │                → units 0–8 content, 9–11 style
 └─ lesion path ─ frozen normative model → residual z → [brighter, darker] → mean-pooled to 16³
                  → 4 heads (1×1×1 conv, ÷ temperature, softmax inside the brain)
                  → each head's mean (x, y, z), minus the brain centroid, ÷ per-axis brain spread
                  → units 12–23 (content)
```

1. **Data.**
   - Each synthetic subject has 9 labelled anatomy factors plus unlabelled shared deformation.
   - One tissue map is rendered with two contrast tables (T1, FLAIR). Each view gets its own gain,
     bias and noise.
   - The lesion is one small blob in deep white matter: darker than white matter in T1, brighter
     in FLAIR.
   - A batch is 32 subjects, so 64 volumes at 64³.
2. **Global path.**
   - Each view has its own encoder; FLAIR's starts as a copy of T1's.
   - Two stride-2 convs (1→32→64 channels) with LayerNorm, a 3³ conv and two residual blocks give
     a 64-channel 16³ map.
   - The map is averaged over space, then passed through an MLP 64→100→12.
3. **Lesion path.**
   - Before training, a per-view normative model is fitted on the first 300 training subjects,
     then frozen. The mean and 20 principal modes come from 200 subjects; the per-voxel residual
     SD comes from the other 100.
   - For each volume: subtract the mean and its best fit of the modes, then divide by the SD. The
     result is the residual z, what normal anatomy cannot explain.
   - Each head scores every brain cell as w_bright · bright + w_dark · dark, divides by the
     temperature, applies a softmax and reports the attention-weighted mean position.
4. **Losses.**
   - Global InfoNCE on units 0–8. Cosine similarity, τ 0.1; the negatives are other subjects'
     other view. Both directions are summed, so chance is 2·ln 32 = 6.93.
   - Lesion InfoNCE on units 12–23. Each coordinate is z-scored over the batch, then mapped by a
     Linear 12→8; the InfoNCE is the same.
   - Total = sum of the two; Adam, lr 1e-4.
5. **Gradients.**
   - The lesion path reads the raw volumes, not the backbone, so the two paths share nothing.
   - The global loss trains the encoders and the MLP.
   - The lesion loss trains only the heads' K × 2 weights and the projector. The head biases
     cancel in the softmax.
6. **Evaluation.** Step 0, then every 1,000 steps, on 400 validation subjects:
   - Probes from the 21 content units to the 9 factors.
   - Content→style and content→view.
   - The branch alone, per view, with head spread and (for residual input) the head weights.

With `--lesion-input features` (findings 3–8) the heads read the backbone's 64-channel map instead,
so the lesion loss trains the backbone too.

## Findings, in the order they were established

1. **Global average pooling cannot see a lesion that only moves.** Pooling translation-equivariant
   features is invariant to where a fixed-size blob sits. Moving the lesion shifts the loss-facing
   vector by 0.026, against 0.28 for ventricle size. Position needs coordinate-valued pooling
   (spatial soft-argmax). For a template in Gaussian noise with a flat prior, soft-argmax is the
   posterior-mean position.
2. **Position-capable pooling is necessary but not sufficient.** In toys, an easy shared global
   factor stops InfoNCE from learning the lesion even when the pooling could express it. This is
   feature suppression (Chen, Luo & Li 2021).
3. **On real data the lesion branch reads whole-brain shape, not the lesion.**
   - Every head stays spread over the entire brain (~16% of the map) throughout training. Its
     coordinate is therefore the brain's tissue centroid.
   - Brain size, left-right asymmetry, cortical thickness and temporal atrophy all move that
     centroid, so InfoNCE gets subject identity without anything being localised.
   - The lesion is about 1 of ~650 brain cells.
4. **The untrained branch already encodes brain size, mostly through GroupNorm.** GroupNorm writes
   each subject's global statistics into every voxel, which rescales the keypoint logits like a
   gain. A LayerNorm backbone plus the brain frame halves this, from about 0.7 to 0.33. The rest is
   structural: the generator makes brain size an *additive radius shift* while cortical thickness
   and ventricle radius stay fixed in absolute units. Relative anatomy therefore changes with brain
   size, and no frame or global rescaling removes it.
5. **Sharp heads (temperature 0.02) lock onto anatomical edges.** The heads do sharpen, to 0.2–4% of
   the map. But edges are high-contrast, present in every subject, and their position moves with
   thickness and asymmetry, so they capture the heads before any head finds the lesion.
6. **Decorrelation removes brain size from the branch but not the problem.** The penalty is the
   squared correlation with the detached content block. The branch moves on to the other global
   shape factors.
7. **Styled lesion intensity changes nothing** at step 1000: within 0.01 of fixed intensity on every
   factor.
8. **The lesion is anti-aligned across views.**
   - The lesion is darker than white matter in T1 (0.4 vs 0.8) and brighter in FLAIR (1.0 vs 0.4),
     in styled mode too. The two encoders start as identical copies.
   - So the lesion pushes the T1 and FLAIR codes of the same patch *apart*. Cross-view patch
     retrieval is 0.088 at a subject's own lesion position, against 0.146 at the same positions
     when the lesion is elsewhere.
   - Alignment's cheapest fix is to suppress the lesion. The patch loss cannot help.
   - Published dense SSL avoids this because its two views are same-modality augmentations.
9. **Why the ventricle is fine although it is fainter in FLAIR.**
   - CSF is darker than white matter in *both* views: 0.1 vs 0.8 in T1 and 0.1 vs 0.4 in FLAIR.
     The sign is the same; only the strength differs.
   - Ventricle size is an *amount* at a fixed place. A bigger ventricle lowers the mean intensity of
     both views, so every pooled statistic sees it, and even an untrained encoder reads it.
   - Lesion position has no global footprint (the untrained baseline is about 0), so it must be
     learned, and that is exactly where suppression and anti-alignment block it.
10. **A label-free anomaly map finds the lesion without any training.** A voxelwise normative
    z-score (population mean and spread per position, from 200 training subjects), taken as "FLAIR
    brighter than normal", peaks within 3 voxels of the lesion in **99 of 100** subjects. The peak
    position gives lesion x/y/z R² 0.74 / 0.83 / 0.44; lesion_z's ceiling is about 0.57 even with
    perfect localisation.
11. **content→view = 1 is a constant per-view offset, not a failed objective.**
    - Subtract each view's mean and the view probe falls to 0.38. This probe's floor is ~0.36, not
      0.5, because its folds split a subject's two views.
    - With `--cross-view-negs-only`, a shift applied to every FLAIR vector cancels in the softmax,
      so the loss cannot see it.
    - The objective *is* being optimised: cross-view retrieval at batch 32 rose from 0.09 to 0.34
      (chance 0.03) by step 2000. The runs are short, not converged.
12. **A residual input puts the lesion in the code, but before training, not because of it.**
    - The heads read each view's residual against a per-view PCA normative model (k = 20),
      not the backbone map.
    - Global shape is explained away first, so nothing captures the branch. Brain size,
      ventricle and cortical thickness stay at about 0.
    - Untrained, the branch already reads lesion x / y / z at 0.49 / 0.48 / 0.39 (T1) and
      0.68 / 0.68 / 0.35 (FLAIR). At step 1000 the values are 0.45 / 0.46 / 0.39 and 0.61 / 0.60 / 0.35.
    - In the content block: 0.46 / 0.55 / 0.41, against 0.48 / 0.57 / 0.42 untrained.
      - That is several times any earlier run (best 0.10 / 0.03 / 0.06).
      - But the change vs the untrained floor is about −0.02.
    - A perfect localiser (the true lesion centroid in the brain frame, linear readout) gives
      0.80 / 0.87 / 0.66. The branch reaches about 55–75% of that.
13. **On the residual, training learns to drop polarity and nothing more.**
    - Each head is a 1×1×1 weighting of the brighter and darker channels.
    - The heads started mixed (one bright-only, one dark-leaning). By step 1000 all four weight the
      two channels about equally, because only an unsigned detector finds the lesion in both views.
    - FLAIR loses its best bright-only detector (0.68 → 0.61), and the four heads become redundant.
    - With 8 learnable weights on a fixed map, the branch cannot do better than the map.
14. **The objective carries the lesion across views, but it cannot find it from nothing.**
    The head-init control and a 30,000-step replay of the branch alone show this (tables under
    "Real-data results").
    - **From plain random signs, training gets to the same solution as the hand-set positive start.**
      - It reaches the same loss (3.54 vs 3.57) and the same lesion R²: T1 0.50 / 0.55 / 0.36,
        FLAIR 0.66 / 0.63 / 0.33.
      - The T1 part is learned: T1 lesion R² goes from about 0 to 0.50 / 0.55 / 0.36.
      - Heads that started bright-only already found the FLAIR lesion. Cross-view InfoNCE then
        flipped their darker-than-normal weight to positive (−3.6 → +2.0, −1.1 → +2.0,
        −0.4 → +5.0), so they also find the T1 lesion, which is dark.
      - So the hand-set positive init is not needed. With 4 heads, all of them start negative on
        both channels with probability 1/256.
      - **Confirmed by a real run** (`..._lresid_hinitrandom_...`, 2,000 steps, the full model):
        - Heads go from [−2.8 −1.1] [+1.1 −3.6] [+1.2 −1.1] [+4.5 −0.4] to
          [−0.5 −0.6] [+2.1 −0.1] [+2.2 +1.6] [+3.1 +3.9], the same as the replay.
        - Branch T1: −0.09 / 0.14 / −0.04 → 0.48 / 0.52 / 0.38. FLAIR: 0.68 / 0.68 / 0.39 → 0.61 / 0.62 / 0.36.
        - Content block: 0.47 / 0.59 / 0.40. **Δ vs floor +0.47 / +0.46 / +0.42**, the first large
          learned lesion gain in the content block. Lesion InfoNCE 4.17 (positive init 4.61).
    - **From negative weights, training never reaches the lesion.**
      - The weights shrink towards zero but do not cross it, even after 30,000 steps. They end at
        [−0.1 −0.1] [−0.7 −0.8] [−0.1 −0.2] [−0.6 −0.7].
      - Lesion R² stops at about T1 0.01 / 0.29 / 0.14, FLAIR 0.07 / 0.33 / 0.17, and the loss
        plateaus at 5.56 (chance 6.93).
    - **Why it stalls:** coordinates are z-scored over the batch before the projector, so the loss
      cannot see a head's scale. Near zero weights, a head's coordinate is ±(the anomaly map's
      dipole), and its sign only flips with the weights' sign. So zero is a flat ridge that
      gradient descent approaches but cannot cross.
    - **What this means:** the normative residual is necessary, because it makes the lesion the
      dominant anomaly. The contrastive objective adds the polarity-invariant detector, provided
      some head starts on the lesion in at least one view.
15. **Lesion burden (2,000 steps): not learned yet, and training first removes it.**
    - In the 9-unit content code, `lesion_burden` R² is −0.02 at steps 0, 1000 and 2000. So is
      `ventricle_size` (−0.02 to −0.03). The code still carries only brain size (0.70) and
      thickness (0.33), with effective rank ~2.
    - In the encoder's 64-channel features, burden is 0.06 (T1) / 0.14 (FLAIR) untrained, and 0.00
      after 2,000 steps. Training removes it.
    - At first this looked like finding 8 (the lesion is dark in T1 and bright in FLAIR). That
      reading is wrong for the global code.
      - Grey matter also flips sign (T1 0.5 < WM 0.8; FLAIR 0.8 > WM 0.4), and thickness is learned
        to 0.83.
      - Arm B learns burden with the sign still flipped.
      - The separate encoders absorb a sign flip. Polarity only bites where weights are shared
        across views (the keypoint heads), and in patch alignment from identical initial encoders.
      - What blocks burden is T1 visibility (finding 16).
    - In the full encoder-only recipe the ventricle reaches GAP R² 0.83 only by step ~8,000. So
      2,000 steps cannot say whether burden recovers the same way.
16. **Lesion burden on the full recipe (10,000 steps, GroupNorm): the encoder carries it, the
    content code drops it.**
    - In the content code, `lesion_burden` stays at ~0 (−0.04 to 0.04) at every evaluation. Over
      the same run, ventricle goes 0.18 → 0.84, temporal atrophy −0.03 → 0.80, asymmetry → 0.90,
      thickness → 0.79, brain size → 0.89, and style leak → ~0.
    - In the encoder's 64 GAP channels, burden rises with training: T1 0.04 → 0.16, FLAIR 0.22 → 0.38.
    - The 9-unit code keeps only 0.03 (T1) / 0.11 (FLAIR) of it. The ventricle passes through
      intact (0.85 in both).
    - Two candidate causes, not yet separated:
      1. **Cross-view inconsistency.** Burden is much weaker in T1. The T1 lesion (0.4) sits
         between CSF (0.1) and GM (0.5), so it looks like ordinary tissue. The FLAIR lesion (1.0)
         is the brightest thing in the image. Contrastive alignment keeps only what both views
         agree on.
      2. **A saturated loss.** InfoNCE is at 0.51 by step 10,000 (chance 6.93). Five factors
         already identify subjects at batch 32, so nothing pushes burden in. This is feature
         suppression.

## Real-data results

Setup: `conv_mlp` recipe (`experiments/encoder_comparison.json`): res 64, 16³ backbone map, batch
32, τ 0.1, cross-view negatives only, fixed_reference normalisation, clean content, wm_interior
lesions, 4 keypoint heads. Brain frame unless noted. Runs are in
`results/encoder_ablations_mps/runs/`, logs in `results/encoder_ablations_mps/logs/`.

**Untrained branch, brain_size R² (240 validation subjects, mean of 3 init seeds).** lesion_x/y/z
are about 0 in every configuration.

| Backbone | Branch norm | Frame | brain_size |
|---|---|---|---|
| group | none | brain | 0.62 / 0.70 |
| group | layer | brain | 0.50 / 0.56 |
| group | none | grid | 0.71 / 0.76 |
| group | layer | grid | 0.59 / 0.66 |
| layer | none | brain | **0.33 / 0.34** |
| layer | none | grid | 0.69 / 0.71 |

**What the lesion branch alone encodes (step 1000 unless noted).** Spread = each head's effective
positions as a fraction of the map; the brain is about 0.16.

| Run (model ID suffix after `conv_mlp_s42_mps_t2000_`) | brain_size | lr_asym | cortical | temporal | lesion x/y/z | spread |
|---|---|---|---|---|---|---|
| GroupNorm (`lesionkp4_v3`) | 0.84 / 0.83 | 0.86 / 0.84 | 0.79 / 0.83 | 0.43 / 0.44 | ≤ 0.16 | 0.13–0.14 |
| LayerNorm (`lesionkp4_layernorm_v4`) | 0.88 / 0.86 | 0.90 / 0.90 | 0.83 / 0.87 | 0.73 / 0.73 | ≤ 0.05 | 0.16 |
| + decorrelation 1 (`lesionkp4_layernorm_dc1`) | 0.44 / 0.72 | 0.85 / 0.87 | 0.65 / 0.80 | 0.15 / 0.25 | ≤ 0.12 | 0.16 |
| styled (`styled_lesionkp4_layernorm`) | 0.88 / 0.87 | 0.90 / 0.90 | 0.83 / 0.88 | 0.73 / 0.75 | ≤ 0.04 | 0.16 |
| styled + decorrelation (`styled_lesionkp4_layernorm_dc1`) | 0.55 / 0.80 | 0.87 / 0.88 | 0.69 / 0.85 | 0.51 / 0.57 | ≤ 0.02 | 0.16 |
| sharp heads, styled (`lesionkp4_temp0.02_layernorm_lesionstyled`), step 2000 | 0.45 / 0.80 | 0.91 / 0.90 | 0.80 / 0.85 | 0.21 / 0.08 | ≤ 0.01 | 0.006–0.03 |
| sharp + patch loss (`patch8x8x8_w1_fgmask_lesionkp4_temp0.02_layernorm_lesionstyled`), step 2000 | 0.73 / 0.69 | 0.90 / 0.88 | 0.81 / 0.81 | 0.52 / 0.39 | ≈ 0 | 0.01–0.05 |

The LayerNorm runs use the batch-standardised projector. Without it, a LayerNorm backbone has a cold
start: the coordinates differ between subjects by only 7e-3, and the branch's InfoNCE sat at
exactly 2·ln 32 = 6.9313.

**Cross-view patch retrieval among 32 subjects** (sharp + patch-loss checkpoint, step 1000):

| Positions | Retrieval |
|---|---|
| All lesion-free positions | 0.361 |
| Deep-white-matter lesion positions, subject's lesion elsewhere | 0.146 |
| Same positions, subject's lesion there | **0.088** |

Patch codes are only ~40% explained by the global code (median R² 0.41), and retrieval from the
global code alone is 0.27. Global identity is not what defeats the patch loss.

**Normative anomaly map, no training** (200 reference and 100 test subjects; peak of the smoothed
z-map inside the brain):

| Score | Hit (≤3 voxels) | Lesion x / y / z R² from the peak |
|---|---|---|
| FLAIR brighter than normal | **0.99** | **0.74 / 0.83 / 0.44** |
| T1 darker than normal | 0.61 | 0.47 / 0.39 / 0.14 |
| FLAIR, either direction | 0.70 | 0.47 / 0.35 / −0.26 |
| T1, either direction | 0.38 | 0.03 / 0.17 / 0.01 |

Direction matters per view: normal boundary shifts create deviations of both signs.

**Explaining away first fixes that** (per-view PCA on 200 training subjects, residual noise
calibrated on 100 held-out ones, peak of the smoothed residual; hit within 3 voxels):

| Components removed (k) | FLAIR bright | T1 dark | Both views, no polarity |
|---|---|---|---|
| 0 (plain normative map) | 0.81 | 0.31 | 0.52 |
| 5 | 0.97 | 0.87 | 0.93 |
| 20 | 0.96 | 0.92 | **0.95** (R² 0.68 / 0.76 / 0.27) |
| 100 | 0.96 | 0.94 | 0.93 |

A few population modes absorb the global shape factors. Lesions at random positions never become a
component, so they stay in the residual, and the residual's magnitude agrees across views. (The
k = 0 row is setup-sensitive: an earlier normative check with in-brain-only noise estimates gave
0.99 for FLAIR-bright.)

**The view probe** (content block, sharp-head run, 200 validation subjects):

| | untrained | step 2000 |
|---|---|---|
| View probe, raw | 0.83 | 1.00 |
| View probe, each view mean-centred | 0.41 | 0.38 |
| Cross-view retrieval, batch 32 (chance 0.03) | 0.09 | 0.34 |

**Residual-input branch** (`lesionkp4_temp0.03_lresid_layernorm_lesionstyled`: LayerNorm backbone,
temperature 0.03, k = 20, positive head init; 400 validation subjects):

| | Step 0 | Step 1000 |
|---|---|---|
| Branch alone, lesion x / y / z, T1 | 0.49 / 0.48 / 0.39 | 0.45 / 0.46 / 0.39 |
| Branch alone, lesion x / y / z, FLAIR | 0.68 / 0.68 / 0.35 | 0.61 / 0.60 / 0.35 |
| Branch alone, brain_size and cortical | ≈ 0 | ≈ 0 |
| Branch alone, lr_asymmetry (T1 / FLAIR) | 0.25 / 0.22 | 0.24 / 0.12 |
| Content block (21 units), lesion x / y / z | 0.48 / 0.57 / 0.42 | 0.46 / 0.55 / 0.41 |
| Head spread (fraction of the map) | 0.000–0.12 | 0.009–0.08 |

Lesion placement explains at most ~0.1 of the lr_asymmetry reading. The rest is probably heads
landing on leftover boundary residuals when they miss the lesion; T1 detection is the weaker one.
Head weights, logit per unit residual z [brighter, darker]:
- Initial: [+2.8 +1.1] [+1.1 +3.6] [+1.2 +1.1] [+4.5 +0.4].
- Step 1000: [+1.9 +2.8] [+2.4 +3.0] [+1.6 +1.7] [+2.3 +2.6].

**Perfect-localisation ceiling** (true lesion centroid, 400 validation subjects):

| Readout | lesion x / y / z |
|---|---|
| Raw voxel coordinates, linear | 0.71 / 0.85 / 0.65 |
| Brain frame, linear | 0.80 / 0.87 / 0.66 |
| Brain frame, cubic | 0.91 / 0.94 / 0.71 |

**Content-block lesion recovery across runs, step 1000** (ridge R², and change vs each run's
untrained floor):

| Run | lesion x / y / z | Δ vs floor |
|---|---|---|
| GroupNorm (`lesionkp4_v3`) | 0.10 / 0.03 / 0.06 | +0.07 / +0.04 / +0.08 |
| LayerNorm (`lesionkp4_layernorm_v4`) | −0.02 / −0.02 / −0.01 | +0.01 / +0.00 / +0.01 |
| styled LayerNorm (`styled_lesionkp4_layernorm`) | 0.00 / 0.02 / −0.01 | +0.03 / +0.04 / +0.00 |
| sharp heads, and sharp + patch loss | ≈ 0 | ≈ +0.01 |
| **residual input** | **0.46 / 0.55 / 0.41** | −0.02 / −0.01 / −0.02 |

**Untrained floor by `--lesion-head-init`** (residual input, temperature 0.03, one normative model,
3 draws of the head weights; ranges over the draws):

| Init | lesion x / y / z, T1 | lesion x / y / z, FLAIR | lr_asymmetry T1 / FLAIR | spread |
|---|---|---|---|---|
| positive | −0.05–0.45 / 0.27–0.45 / 0.07–0.35 | 0.63–0.67 / 0.67–0.69 / 0.36–0.38 | 0.32–0.36 / 0.15–0.24 | 0.07–0.08 |
| random | −0.04–0.18 / 0.26–0.29 / 0.05–0.15 | 0.51–0.66 / 0.53–0.69 / 0.34–0.38 | 0.40–0.47 / 0.13–0.25 | 0.09–0.11 |
| negative | ≈ 0 / 0.08–0.14 / ≈ 0 | 0.05–0.07 / 0.19–0.25 / ≈ 0 | 0.48–0.49 / 0.18–0.28 | 0.15 |

- The T1 floor under positive init depends on whether some head starts with a large darker weight.
- Negative heads avoid every anomaly and spread over the whole brain. Their coordinates keep only
  a little lesion signal, through the hole it leaves, plus asymmetry.
- Script: scratchpad `head_init_floor.py`.

**Branch-only replay, 30,000 steps** (scratchpad `branch_alone.py`):
- With residual input, the lesion InfoNCE trains only the heads and the projector, so the replay
  trains those two alone on the same objective (batch 32, τ 0.1, cross-view negatives, AdamW
  lr 1e-4, 2,000 training subjects).
- It reproduces the real run: the same initial heads and step-0 R² for every factor, step-100 loss
  14.3 vs 14.6, and negative-init heads at step 2500 of [−0.6 −0.7] [−1.6 −2.0] [−0.6 −0.8] [−1.6 −2.0],
  against the real run's [−0.7 −0.8] [−1.7 −2.0] [−0.7 −0.8] [−1.7 −2.0] at step 2000.
- The real run also clips the gradient norm of all parameters at 2, which the replay leaves out.
- Lesion x / y / z R² in the table; heads are logits per unit residual z [brighter, darker].

| Init | Step 0, T1 / FLAIR | Step 30,000, T1 / FLAIR | Loss | Heads at 30,000 |
|---|---|---|---|---|
| positive | 0.48 / 0.48 / 0.39 · 0.68 / 0.68 / 0.35 | 0.49 / 0.55 / 0.36 · 0.67 / 0.63 / 0.33 | 3.57 | [+1.9 +2.6] [+4.5 +5.3] [+0.1 +0.1] [+2.2 +2.9] |
| random (the real draws' signs) | −0.09 / 0.14 / −0.04 · 0.68 / 0.68 / 0.39 | **0.50 / 0.55 / 0.36** · 0.66 / 0.63 / 0.33 | 3.54 | [−0.1 −0.2] [+1.4 +2.0] [+1.4 +2.0] [+4.3 +5.0] |
| negative | −0.01 / 0.07 / 0.00 · 0.03 / 0.22 / 0.05 | 0.01 / 0.29 / 0.14 · 0.07 / 0.33 / 0.17 | 5.56 | [−0.1 −0.1] [−0.7 −0.8] [−0.1 −0.2] [−0.6 −0.7] |

The random-init heads started at [−2.8 −1.1] [+1.1 −3.6] [+1.2 −1.1] [+4.5 −0.4].

**Linear vs kernel-ridge readout of the lesion units** (scratchpad `krr_probe.py`):
- 1,000 held-out subjects (val 400 + test 600), 5 random 70/30 splits, standardised features.
- RBF kernel-ridge with (alpha, gamma) chosen by 3-fold CV inside train. The ± is the SD over splits.

| Lesion units | lesion x / y / z, linear | lesion x / y / z, kernel | brain_size, kernel | lr_asym, kernel |
|---|---|---|---|---|
| random init, step 2000, T1 | 0.46 / 0.61 / 0.38 | **0.46 / 0.61 / 0.38** | 0.21 | 0.44 |
| random init, step 2000, FLAIR | 0.65 / 0.69 / 0.49 | **0.69 / 0.72 / 0.52** | 0.18 | 0.24 |
| random init, untrained, T1 | 0.02 / 0.15 / 0.00 | 0.01 / 0.13 / 0.00 | 0.25 | 0.51 |
| random init, untrained, FLAIR | 0.71 / 0.74 / 0.51 | 0.73 / 0.76 / 0.53 | 0.09 | 0.17 |
| positive init, step 2000, T1 | 0.41 / 0.61 / 0.38 | 0.44 / 0.62 / 0.40 | 0.04 | 0.20 |
| positive init, step 2000, FLAIR | 0.66 / 0.66 / 0.47 | 0.66 / 0.66 / 0.50 | 0.06 | 0.23 |
| negative init, step 2000, T1 | 0.01 / 0.29 / 0.11 | 0.02 / 0.19 / 0.06 | 0.25 | 0.41 |
| negative init, step 2000, FLAIR | 0.06 / 0.40 / 0.21 | 0.03 / 0.25 / 0.15 | 0.23 | 0.28 |
| CEILING: true centroid, brain frame | 0.82 / 0.88 / 0.69 | **0.94 / 0.94 / 0.74** | | |
| CEILING: centroid + 6 anatomy latents | 0.83 / 0.88 / 0.69 | 0.92 / 0.92 / 0.69 | | |

- **Kernel ≈ linear on the learned lesion units.** The gap to the ceiling is missing information,
  probably missed detections, not a nonlinear distortion.
  - T1 reaches 49 / 65 / 51% of the kernel ceiling, FLAIR 73 / 77 / 70%.
  - So this is approximate, not exact, identifiability. A full invertible function of position
    would reach the ceiling under a flexible readout.
- **The training gain survives the flexible probe.** T1 goes from 0.01 / 0.13 / 0.00 to
  0.46 / 0.61 / 0.38.
- **The block is not clean.** The random-init units carry brain size only nonlinearly (0.21 kernel,
  0.00 linear), plus asymmetry 0.24–0.44.
  - The source is the heads that never found the lesion: head 1 stayed near-uniform. The
    negative-init run, where every head stalled, shows the same 0.25 / 0.41.
  - Positive init, where every head found the lesion, leaks less (brain size 0.04–0.06).
  - Fixes: drop or penalise heads that stay diffuse, or add the decorrelation term.

**Lesion burden, 2,000 steps** (`conv_mlp_s42_mps_t2000_layernorm_lesionstyled_burden`):

| Readout | Factor | Step 0 | Step 2000 |
|---|---|---|---|
| Content block (9 units) | lesion_burden | −0.02 | −0.02 |
| Content block (9 units) | ventricle_size | −0.02 | −0.03 |
| Backbone GAP (64 channels), T1 / FLAIR | lesion_burden | 0.06 / 0.14 | −0.01 / 0.00 |
| Backbone GAP (64 channels), T1 / FLAIR | ventricle | 0.04 / 0.02 | −0.04 / −0.01 |
| Backbone GAP (64 channels), T1 / FLAIR | brain_size | 0.77 / 0.74 | 0.67 / 0.68 |

Script: scratchpad `burden_gap_probe.py`.

## Toy evidence

Fixed translation-equivariant features on a 16³ grid: a registered anatomy field, an easy shared
factor `s`, a lesion blob and noise. Two views per subject, InfoNCE with B = 64 and τ = 0.1.

**Pooling, supervised regression of the lesion centre:**
- GAP: 0.
- Attention without positional encoding: 0.
- The shipped attention pool: 0.94, after a plateau of about 1,000 steps.
- Soft-argmax: 0.97.

**InfoNCE:**
- Without `s`, soft-argmax finds the lesion (0.93).
- With `s`, nothing learns the lesion, even though the loss is not saturated.

**Label-free fixes that failed against `s`:** lower temperature, implicit feature modification,
negatives reweighted by the model's own global-code similarity, and a global-to-local predictor. The
predictor stalls from a cold start; its objective is fine, as an oracle check shows.

**What works.** Toy v2: features zero outside the brain, lesion placed relative to the brain. Lesion
R² from the branch, seeds 0 / 1 / 2:

| Arm | Gain-type easy factor | Scaling-type easy factor |
|---|---|---|
| Plain branch | 0.01 / −0.01 / 0.12 | 0.04 / 0.10 / 0.30 |
| Per-voxel LayerNorm | 0.76 / 0.77 / 0.74 | 0.05 / 0.00 / 0.50 |
| Brain frame | 0.95 / 0.66 / 0.88 | 0.82 / 0.67 / 0.74 |
| Easy factor redrawn per view (branch only) | **0.95 / 0.96 / 0.95** | **0.87 / 0.88 / 0.85** |
| Plain + decorrelation | 0.90 / 0.90 / 0.90 | 0.79 / 0.64 / 0.87 |

In toys every fix works. On real data none of the architectural ones did, because real brain size is
neither a pure gain nor a pure scaling, and because of the lesion's cross-view polarity (finding 8),
which the toys did not have. Toy scripts are in the session scratchpad and may be cleaned up:
`pool_saddle.py`, `alternatives.py`, `alt_ln.py`, `alt_v2.py`, `global_local.py`, `init_leak.py`,
`patch_check.py`, `anomaly_check.py`, `view_check.py`.

## Code

| File | What |
|---|---|
| `models/keypoint_pool.py` | `KeypointPool3d`: per head, 1x1-conv logits divided by `temperature`, softmax over positions, expected (x, y, z). `frame="brain"` confines attention to the brain (logits + log occupancy) and gives coordinates relative to the brain's centroid and per-axis spread. Optional per-voxel LayerNorm. Logits start small and random. |
| `models/normative_residual.py` | `NormativeResidual`: per-view mean, k principal modes and residual SD, fitted once from training subjects and held in buffers, so checkpoints restore it. `forward` returns one view's residual z, zero outside the brain. |
| `models/multiview_encoder.py` | Branch built last, so all other weights match the run without it. 3K coordinates appended after the style units and marked as content. `project_lesion` standardises over the batch before the projector. `lesion_code(x, view_idx)` and `lesion_maps()`. `lesion_input="residual"` reads `[relu(z), relu(−z)]` pooled to the feature grid (`_lesion_inputs`, `fit_normative`); `lesion_head_init` sets the heads' initial signs. |
| `training/main_conv_synthetic.py` | `--lesion-keypoints`, `--lesion-norm`, `--lesion-frame`, `--lesion-proj-dim`, `--lesion-loss-weight`, `--lesion-temperature`, `--lesion-decorrelation-weight`, `--lesion-pairing {cross_modal,within_modality}`; `--lesion-input {features,residual}`, `--lesion-normative-components`, `--lesion-normative-subjects`, `--lesion-head-init {positive,random,negative}`. Own InfoNCE for the branch; `augment_intensity` for same-modality pairs. The normative model is fitted on the first training subjects before step 0. Every evaluation prints "lesion branch alone" (R² per factor, T1/FLAIR), head spread and, for residual input, head weights, also saved under `lesion_branch` in `dci_step*.json`. |
| `eval/protocol/score_checkpoint.py`, `eval/lesion/checkpoint_lesion_analysis.py` | Restore the branch; count lesion units as content, not style. |
| `scripts/run_encoder_mps.py` | Passes all lesion flags through and adds run-ID suffixes (`_lresid`, `_nk{k}`, `_hinit{init}` for the residual options). `compare_backend` scales the keypoint tolerance by 1/temperature and skips the random lesion loss: CUDA's TF32 convolutions differ from CPU by ~2e-3 at temperature 0.02. |
| `tests/test_lesion_branch.py` | 24 tests. |
| `eval/synthetic/synthetic_dataset.py`, `data/datasets.py` | `lesion_target` / `lesion_count` (`synthetic_lesion_target` / `synthetic_lesion_count` on the wrapper): `PseudoMRIRenderer._burden_in_white_matter`, and a `z_lesion` draw of placement quantiles in burden mode. |
| `eval/metrics/dci.py` | `content_factor_names(n, lesion_target)` and `dataset_lesion_target(dataset)`, so burden runs report `lesion_burden` / `unused_3` / `unused_4`. |
| `tests/test_lesion_burden.py` | 5 tests: volume linear in the burden, white-matter containment, positions independent of the burden, other latents unchanged, CLI, runner and scorer. |

Bugs fixed along the way:
- `attention_spread` crashed on MPS (float64); this also affected the attention-pool arm.
- The brain frame originally leaked brain size through uniform attention.
- The backend check was missing parser defaults and was too strict for sharp heads on CUDA.

## Literature (short)

No paper was found that makes a focal-lesion factor identifiable from paired-MRI contrastive
learning. The closest work supports the findings above:
- Daunhawer et al., ICLR 2023: multimodal contrastive learning identifies *shared* factors only.
- FactorCL (Liang et al., NeurIPS 2023) and CoMM (ICLR 2025): ways to capture modality-unique
  information.
- Koutsouvelis et al., 2025, arXiv 2511.11311: modality-invariant brain-MRI pretraining; lesion
  segmentation needs modality-specific features.
- Pignedoli et al., 2026, arXiv 2606.16756: treating QSM/FLAIR asymmetrically helps MS lesions.

## Open threads and next steps

1. **Same-modality pairing** (`--lesion-pairing within_modality`; tests finding 8) is running on the
   user's CUDA PC, with and without decorrelation. Read the FLAIR column of "lesion branch alone".
   - Expectation: brain shape captures it too, since shape is shared within a modality as well.
   - If it does not, that is a learned route with no normative model, and it comes first.
   - If it fails, the remaining options are:
     - unique-information objectives (FactorCL/CoMM);
     - a nonlinear (HSIC) decorrelation;
     - on synthetic data only, an oracle per-view brain-size re-render for the branch.
2. **Head-init control: done (finding 14), including the real random-init run.** The objective
   carries the lesion from one view to the other but cannot find it from nothing. Open: 3 seeds, and
   a nonlinear (kernel-ridge) probe on the lesion units, because identifiability is only up to an
   invertible map.
   - Why not a near-zero init: with near-uniform attention, the coordinates still carry the dipole
     of the anomaly map. At tiny scales `holdout_r2`'s 1e-4 std floor hides it, so R² would rise
     with the weights' scale alone.
3. **Give the residual branch capacity** (not implemented).
   - Put a small learnable conv stack (two 3³ layers, ~16 channels) between the residual and the
     heads, and score it against its own untrained floor.
   - Headroom: the perfect-localisation ceiling is 0.80 / 0.87 / 0.66, against 0.45 / 0.46 / 0.39
     in T1.
   - Risk: it may learn leftover boundary residuals instead (the 0.24 lr_asymmetry already is
     that), so keep the decorrelation term ready.
4. **Writing it up.** Lesion position is now a split result.
   - Three things block it: pooling cannot see position, global shape suppresses the lesion, and
     its polarity flips between T1 and FLAIR.
   - A label-free normative residual gets around all three. It makes the lesion findable.
   - The cross-modal objective then learns the polarity-invariant detector from a random start
     (finding 14). It cannot find a lesion that no head sees in either view.
   - Use a burden factor instead (see "Lesion burden" below). It is an amount, like ventricle size,
     which already works, and it is what ADNI measures.
5. **A learned explain-away** would replace the PCA with a decoder from the model's own global code.
   It is not implemented, and it diverges from the encoder-only paper.
   - A weaker variant adds each view's own normative score to the feature-reading heads' logits.
   - That is a modality-specific inductive bias, and it should be reported as one.
6. **Report the mean-centred content→view probe** alongside the raw one, or remove the per-view
   offset in the model (finding 11).
7. **Statistics:** confirm any positive result with 3 seeds and the full 10,000-step recipe.

## Lesion burden (`--synthetic-lesion-target burden`)

Lesion position has no ADNI counterpart, and pooling cannot carry it. Burden is an amount, like
ventricle size, which already works, and it is what ADNI measures (the UC Davis WMH volumes).

- **What changes.**
  - `z_content[2]` sets the total volume of `--synthetic-lesion-count` (default 4) spheres. Each
    radius is `--synthetic-lesion-radius` × ((b + 1) / 2)^(1/3), where b is the squashed latent
    (tanh with clean content). The total volume is therefore linear in (b + 1) / 2, from 0 to four
    full spheres.
  - The edge is a one-voxel partial-volume ramp, so the volume is continuous, not voxel steps.
  - `z_content[3:5]` are still drawn but render nothing. Reports name the three dims
    `lesion_burden`, `unused_3` and `unused_4`. Aggregates such as block_mcc average the two
    unused ones in.
- **Positions are a nuisance.**
  - Each subject gets a `z_lesion` of 4 × 3 placement quantiles, drawn after every other latent.
  - The spheres are placed with the same conditional-quantile scheme as `wm_interior`, among
    centres where a full-size sphere fits in final white matter, at least two full radii apart.
  - So positions never depend on the burden, and spheres never overlap.
- **Same subjects as position mode.** Every other latent is drawn in the same order, so each
  subject keeps its anatomy and style; only the lesion differs. On 400 validation subjects, 399 are
  identical and one was redrawn for lack of room.
- **Checks** (scratchpad `burden_check.py`, 400 validation subjects, res 64):
  - Total lesion load is 4–530 voxels (four full spheres = 524).
  - Its correlation with (b + 1) / 2 is 0.9998, and it is monotone.
  - Tests: `tests/test_lesion_burden.py` (5).
- **Constraints.** Needs `--synthetic-lesion-placement wm_interior`, and the lesion branch is not
  allowed with it.
- **Where the option is restored.**
  - Restored: the trainer's own evaluation, `eval.protocol.score_checkpoint`, and
    `run_dci_synthetic.build_synthetic_test_set`. The last one means
    `eval.diagnostics.supervised_ceiling` too, which now takes `--view 1|2` (T1 or FLAIR) and
    labels the burden dims correctly.
  - Not yet restored: `main_multimodal`/`utils/config.py`, `radial_factor_profile`,
    `plot_scaling_maps`, `view_difficulty`, `generator_defects`. Those would rebuild
    position-mode data.
- **`--downscale-factor`** (runner, conv only, power of 2; run ID gains `_ds<k>`) sets the
  backbone's downsampling. At 2, a 4–6-voxel lesion spans 2–3 cells instead of ~1.
- **First run (2,000 steps, LayerNorm), finding 15:** `lesion_burden` and `ventricle_size` both
  read about 0. Training removes the burden signal the untrained FLAIR features carried (0.14 →
  0.00).
- **Full recipe (10,000 steps, GroupNorm), finding 16:** `conv_mlp_s42_mps_lesionstyled_burden`.
  The ventricle reaches 0.84; burden stays at 0.00 in the code, with 0.16 / 0.38 in the encoder's
  GAP channels.
- **T1 lesion intensity control: `--synthetic-lesion-t1-value`** (default 0.4, so the data is
  unchanged).
  - Measured T1 contrast with white matter (rendered, scratchpad `t1_contrast.py`, 127 subjects):

    | T1 value | 0.4 (default) | 0.0 | 1.2 | 1.3 | 1.37 | 1.4 | 1.6 |
    |---|---|---|---|---|---|---|---|
    | T1 contrast | −0.22 | −0.45 | +0.33 | +0.40 | ≈ +0.45 | +0.47 | +0.61 |

    FLAIR is +0.41. The magnitude image (√ of the noisy signal) and the final 3×3×3 blur lift a
    dark lesion off zero, so the same LUT distance gives less contrast below WM than above it.
  - **Arm A** (`conv_mlp_s42_mps_lesionstyled_burden_t1les1.37`): T1 lesion 1.37, contrast
    +0.45. Bright, above every tissue, same sign as FLAIR.
  - **Arm B** (`conv_mlp_s42_mps_lesionstyled_burden_t1les0`): T1 lesion 0.0, contrast −0.45.
    Dark, opposite sign. Its rendered value still overlaps the darker tissues.
  - Both run the full recipe (10,000 steps, GroupNorm, styled) on the user's CUDA PC. Local copies
    were stopped after the step-0 evaluation, where burden was −0.02 in both. Their partial
    folders in `results/encoder_ablations_mps/` can be deleted.
  - **Reading them:**
    - B learns burden: weak T1 contrast was the problem.
    - Only A learns it: the sign, or how distinct the lesion is, matters. Those two stay
      confounded, because the dark side of WM is crowded with other tissues.
    - Neither learns it: suspect saturation.
  - Saturation is tested with a larger batch or harder negatives (not run; needs the PC).
  - **Arm B at step 2,000 (CUDA), vs the 0.4 baseline at step 2,000 (MPS):**

    | | lesion_burden | ventricle | brain | thickness | temporal | lr_asym | gain leak | loss | eff rank |
    |---|---|---|---|---|---|---|---|---|---|
    | 0.4 baseline | 0.00 | 0.39 | 0.90 | 0.84 | 0.14 | 0.87 | 0.03 | 1.74 | 3.40 |
    | B (0.0) | **0.20** | 0.46 | 0.90 | 0.83 | 0.15 | 0.82 | **0.24** | 1.77 | 3.45 |

    - Everything else tracks within ~0.05, so the burden gain is not a hardware artefact.
    - Burden is learned with the sign still opposite to FLAIR. The baseline's failure was T1
      contrast (too faint, close to GM), not polarity.
    - The extra gain leak fits the arithmetic: T1 lesion contrast = −0.8·gain, so a burden
      feature carries burden × gain.
- Script for the encoder-channel probe: scratchpad `burden_gap_probe.py <run>`.

## Caveats

- Real-data numbers are single-seed, ≤2,000 steps.
- Decorrelation is linear and assumes independent factors. Under `--synthetic-causal`, the lesion
  correlates with other content factors and the penalty would strip real signal.
- The anomaly result relies on the generator's lesion being FLAIR-hyperintense. On ADNI the
  normative model must account for age and atrophy.
- The residual branch's untrained floor already includes the fitted normative model. So a change
  of about 0 vs floor means training added nothing on top of a label-free prior; it does not mean
  the lesion was missing.
- 20 PCA modes explain the synthetic anatomy. Real anatomy varies in many more ways, and ADNI's
  WMH are many periventricular lesions, not one blob. The residual input will need a stronger
  normative model there.

## How to run

```bash
python scripts/run_encoder_mps.py --variant conv_mlp --lesion-keypoints 4 --norm-type layer --synthetic-lesion-intensity styled --lesion-temperature 0.02 --lesion-pairing within_modality --train-steps 2000 --eval-every 1000
```

Residual input; add `--lesion-head-init negative` for the control:

```bash
python scripts/run_encoder_mps.py --variant conv_mlp --lesion-keypoints 4 --lesion-input residual --lesion-temperature 0.03 --norm-type layer --synthetic-lesion-intensity styled --train-steps 2000 --eval-every 1000
```

The normative model is fitted from the first 300 training subjects before step 0, which takes a few
seconds. It is saved with the checkpoint.

On CUDA, pass the same lesion flags through the CUDA wrapper. It forwards extra arguments to
`run_encoder_mps.py --device cuda`.

Tests: use `/opt/miniconda3/envs/adni-analysis/bin/python`. A site-packages `tests` package shadows
the repo's and `lpips` is missing, so make a shim directory containing:
- `tests/__init__.py` with `import os; __path__ = [os.path.join(os.getcwd(), "tests")]`
- `lpips/__init__.py` with an `LPIPS` class that raises

Then, from the repo root:

```bash
PYTHONPATH=<shim>:. /opt/miniconda3/envs/adni-analysis/bin/python tests/test_lesion_branch.py
```

Two launcher tests fail on clean HEAD as well (unrelated drift):
`test_encoder_mps_runner.test_cuda_patch_launcher_matches_mps_and_has_separate_outputs` and
`test_separate_spatial_readout.test_launchers_change_only_readout_and_run_id`.

## Housekeeping

These local run folders in `results/encoder_ablations_mps/runs/` are incomplete and can be deleted:
- `..._lesionkp4`: crashed (MPS float64).
- `..._v2`: flawed brain frame.
- `..._v3` and `..._layernorm_v3`: stopped.
- `..._layernorm_v4`: stopped at step 1000.
- `..._layernorm_dc1`, `..._styled_lesionkp4_layernorm` and `..._styled_lesionkp4_layernorm_dc1`:
  stopped by the 2-hour tool limit at about step 1700.
- The two `..._lpwithin_...` folders: launched by mistake and stopped at step 0.
