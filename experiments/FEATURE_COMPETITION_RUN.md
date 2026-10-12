# Run the feature-competition experiment

Run from the repository root in the existing MRI training environment, for
example `conda activate monai_env` on the NVIDIA PC. Python needs PyTorch, MONAI,
NumPy, SciPy, scikit-learn, nibabel, matplotlib, threadpoolctl and the existing
training-loss dependencies (including LPIPS).

The commands below use CUDA. For the Mac, replace `--device cuda` with
`--device mps`; CPU is also supported. The completed local software check used
CPU/PyTorch 2.6. The arm-A pilot script below has also run end to end on MPS
(PyTorch 2.6, M4 Pro); CUDA execution has not been verified.

## Arm-A pilot on the Mac (protocol section 0)

Run this first. It asks whether anything can be suppressed at all, i.e. whether
arm A learns lesion location with no competitor:

```bash
bash scripts/feature_competition_pilot.sh
```

The script builds an A-only bank in `results/feature_competition/pilot_A` (6.6 GiB)
and audits it. It then trains, evaluates at every checkpoint and plots, in order:

1. The same-architecture supervised control (`supervised_gap`): about 30 min.
2. The SSL encoder for 10,000 steps: about 4 h.
3. The heatmap localizer (`supervised`): about 1 h.

It uses the `adni-analysis` interpreter, the MPS fallback variable, an LPIPS
stand-in and `caffeinate`. Every step resumes or skips when its output exists, so
rerunning continues after Ctrl-C. Output is logged to
`results/feature_competition/pilot_A.log`. Read the tables there or
`summary/curves_A.png`, against the decision table in protocol section 0.

## Prepare once and verify the input intervention

Choose a new output directory on a disk with at least 40 GiB free. The default
float32 data bank, all five arms, occupies about 33 GiB; checkpoints and reports
need extra space. Data are read through memory maps, so the entire bank need not
fit in RAM. `prepare --arms A B Bp` renders a subset instead.

```bash
conda activate monai_env
export COMPETITION_BANK=results/feature_competition_v1

python -m training.feature_competition prepare --bank "$COMPETITION_BANK"
python -m training.feature_competition audit --bank "$COMPETITION_BANK"
```

Preparation refuses to overwrite an existing bank. It creates all five arms (A, B,
B′ as `Bp`, C, D), 2,000 training subjects, 400 probe subjects, 1,000 test subjects
and a separate 128-subject pilot bank. All conditions reuse the same training-only normalization
reference, subject draws and lesion masks. Only competitor variation and lesion
contrast differ. The manifest records the actual intervention settings.

Inspect `audit/matched_images.png` and `audit/report.json`. The latter must have
`input_gate_passed: true`. The audit uses 32 subjects with 32 fresh noise draws per
movement endpoint, covers all three axes and checks B/A and D/C visibility within
10%, as well as increased visibility in C/A and D/B. It also verifies checksums
and re-renders cached examples. A repeat audit can explicitly replace its own
report with `audit --bank "$COMPETITION_BANK" --overwrite`.

## First A/B/B′ run, one training seed

Run this after the arm-A pilot passes. It is a useful first scientific screen at the
full 10,000-step budget: three SSL fits and six supervised image controls. Add
`--kind supervised_gap` runs, the SSL encoder's own architecture trained with labels,
to check that each arm's global code can carry location. Supervised controls see
coordinate labels; SSL models do not. B′ separates competition through the loss
(B′ ≈ A) from input interference (B′ ≈ B); see protocol section 1.

```bash
python -m training.feature_competition train --bank "$COMPETITION_BANK" \
  --kind supervised --arms A B Bp --seeds 42 --device cuda

python -m training.feature_competition train --bank "$COMPETITION_BANK" \
  --kind ssl --arms A B Bp --seeds 42 --device cuda

python -m training.feature_competition evaluate --bank "$COMPETITION_BANK" \
  --kind supervised --arms A B Bp --seeds 42 --device cuda

python -m training.feature_competition evaluate --bank "$COMPETITION_BANK" \
  --kind ssl --arms A B Bp --seeds 42 --checkpoint step_000000.pt --device cuda

python -m training.feature_competition evaluate --bank "$COMPETITION_BANK" \
  --kind ssl --arms A B Bp --seeds 42 --device cuda

python -m training.feature_competition summarize --bank "$COMPETITION_BANK"
```

The supervision controls must actually recover location before their comparison
can establish matched recoverability. A failed control is inconclusive, not an
information-theoretic ceiling. One seed is a screen, not a stable population
estimate. The full replication below completes the planned comparison.

## Complete all five arms and three seeds

These commands also work directly after preparation/audit if you skip the A/B
screen. `--resume` continues matching unfinished runs and skips matching completed
runs. It creates missing runs and refuses incompatible settings, code or banks.

```bash
python -m training.feature_competition train --bank "$COMPETITION_BANK" \
  --kind supervised --device cuda --resume

python -m training.feature_competition train --bank "$COMPETITION_BANK" \
  --kind ssl --device cuda --resume

python -m training.feature_competition evaluate --bank "$COMPETITION_BANK" \
  --kind supervised --device cuda

python -m training.feature_competition evaluate --bank "$COMPETITION_BANK" \
  --kind ssl --checkpoint step_000000.pt --device cuda

python -m training.feature_competition evaluate --bank "$COMPETITION_BANK" \
  --kind ssl --device cuda

python -m training.feature_competition summarize --bank "$COMPETITION_BANK"
```

Defaults are every arm in the bank (A/B/Bp/C/D for a full bank) and seeds 42/142/242: 15 SSL fits and 30 supervised fits,
run sequentially in the current terminal. SSL uses batch 32; supervised fits use
batch 8. Both use learning rate 1e-4 and 10,000 steps. The supervised network is a
small spatial heatmap localizer, not the SSL encoder plus supervision.

Use the same batch size in every arm of a comparison if memory requires a change.
For example append `--batch-size 16` to every SSL training command. Evaluation
batch size can be reduced independently. A resumed training run must retain its
original batch size and other settings.

## Read the outputs

- `manifest.json`: data settings, normalization, common centre set and hashes.
- `audit/report.json`: geometry, movement signal/noise and input matching gates.
- `runs/*/progress.json`: training losses and current step.
- `runs/*/step_*.pt`: untrained and 2,000-step interval checkpoints.
- `runs/*/eval_*/metrics.json`: per-view global/spatial probes or supervised image
  localization, on both common fixed-brain-size and own-distribution test images.
- `summary/recovery.csv` and `summary/recovery.png`: latest evaluated step per run;
  the CSV includes the untrained floor when evaluated.
- `summary/report.json`: seed-wise A−B and C−D recovery penalties, their visibility
  interaction, paired localization-error intervals and supervised equivalence.

R² is not clipped to zero. Primary comparisons use `regime: common`. The summary
compares matched training steps and refuses incompatible paired recipes/grids.
Subject intervals condition on the trained models; use the three seed estimates
to assess training variability. These reports do not automatically declare the
hypothesis true or false.

Spatial evaluation pools the native 16³ backbone map to 8³ by default, without
PCA. `evaluate --spatial-grid 16` preserves the native grid but costs more memory
and probe computation. Use the same grid for all models being compared.

For recovery curves, evaluate the intermediate saved checkpoints explicitly
using, for example, `--checkpoint step_002000.pt`. The summary uses the largest
evaluated step rather than choosing by test performance.

## Small software check

This checks execution only. It uses small cohorts, a narrower model and two
optimizer steps; do not interpret its recovery or confidence intervals as science.
The resolution stays at 64 to preserve the calibrated geometry.

```bash
export COMPETITION_SMOKE=results/feature_competition_smoke

python -m training.feature_competition prepare --bank "$COMPETITION_SMOKE" \
  --train-samples 8 --probe-samples 8 --test-samples 8 --pilot-samples 6
python -m training.feature_competition audit --bank "$COMPETITION_SMOKE" \
  --noise-subjects 6 --noise-repeats 4
python -m training.feature_competition train --bank "$COMPETITION_SMOKE" \
  --kind ssl --seeds 42 --steps 2 --batch-size 2 --hidden-channels 8 \
  --res-channels 4 --save-every 2 --log-every 1 --device cpu
python -m training.feature_competition train --bank "$COMPETITION_SMOKE" \
  --kind supervised --seeds 42 --steps 2 --batch-size 2 \
  --save-every 2 --log-every 1 --device cpu
python -m training.feature_competition evaluate --bank "$COMPETITION_SMOKE" \
  --kind ssl --seeds 42 --spatial-grid 2 --batch-size 2 --device cpu
python -m training.feature_competition evaluate --bank "$COMPETITION_SMOKE" \
  --kind supervised --seeds 42 --batch-size 2 --device cpu
python -m training.feature_competition summarize --bank "$COMPETITION_SMOKE"
```

Focused tests:

```bash
python -m unittest discover -s tests -p test_feature_competition.py -v
```
