# Global + local content identifiability: synthetic test

**Status (2026-11-03): E1 (2D, noise-free) run and reported here. E2–E9 not started; waiting for approval, as the
spec requires.**

<!-- RESULTS -->

## What was run

- **Code** (all new, under `experiments/gl_ident/`; no existing training code touched):
  `config.py` (YAML-loadable dataclasses), `generator.py`, `guards.py`, `models.py`, `losses.py`, `train.py`,
  `evaluate.py`, `summarize.py` (tables and figures; never re-scores), `configs/e1.yaml`, `tests/test_gl_ident.py`,
  `run_e1.sh` (local runner used for E1), `runai_submit.sh` (Run:ai submit for later experiments; copies the flags
  of `experiments/generated/encoder/*.runai.sh`; dry-run checked, never submitted).
- **E1**: 2D, 96×96 images, m = 4, 24×24 coarse grid, (d_γ, d_c, d_s) = (4, 4, 3), SNR = ∞, λ_ν = 0.
  Three models trained separately: local (n_g = 2, so R = 3), global window (w = 6, disjoint supports), pooled
  baseline. Seeds 0, 1, 2. 30,000 steps, batch 64 subjects, Adam lr 1e-3 with cosine decay to 0, checkpoint at
  the best validation loss (checked every 1,000 steps). InfoNCE, temperature 0.1, cosine similarity.
- **Added for context (not in the spec):**
  - the untrained (step-0) model of seed 0 for each model, scored with the same protocol, as a floor;
  - seed 0 of local and global with the spec's literal code size (see deviation 1);
  - a residual leakage check next to M2 (see deviation 9).
- **Hardware**: this Mac (Apple M4 Pro, MPS), torch 2.12.1, conda env `HCD_env`. The three models trained as three
  parallel streams.

## Deviations from the spec

<!-- DEVIATIONS -->

## Reproduce

```bash
cd <repo root>
PY=/opt/miniconda3/envs/HCD_env/bin/python
PYTHONPATH=. $PY -m experiments.gl_ident.guards                          # writes results/guards.md
PYTHONPATH=. $PY experiments/gl_ident/tests/test_gl_ident.py              # 13 tests (pytest-style, no pytest needed)
bash experiments/gl_ident/run_e1.sh                                      # E1: 11 runs, ~4.5 h on the M4 Pro
PYTHONPATH=. $PY -m experiments.gl_ident.summarize                       # summary CSVs, e1_tables.md, plots
```

`results/` is matched by the repo's `.gitignore` (`results/`), so run outputs, `guards.md`, the summary CSVs and
the plots stay local unless force-added.
