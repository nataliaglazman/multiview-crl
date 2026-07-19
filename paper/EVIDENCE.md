# Evidence ledger — `paper/draft.tex`

Maps every quantitative claim in the draft to its provenance and reportability status.
Update this alongside the draft; do not promote a number to a headline until its status is SAFE.

**Status key**
- **SAFE** — measured under a valid protocol, replicated or robust; reportable as-is.
- **CONFOUNDED** — a known protocol flaw makes the comparison unfair. Do not report.
- **SUSPECT** — valid protocol, but evaluated on a checkpoint/generator mismatch. Needs re-run.
- **ARTIFACT-RISK** — the measurement may reflect probe limitations rather than the model.
- **PENDING** — not yet run.

---

## Headline claims

| # | Claim (draft §) | Numbers | Source | Status |
|---|---|---|---|---|
| 1 | Content completeness is tied at patch pooling | 0.409 vs 0.409 mean lin-R²; block-MCC 0.540/0.535; best-format 0.419/0.416 | `content_rank_out6`, `--per-factor-mlp`, seeds 0–4 | **SAFE** — multi-seed, the strongest result in the paper |
| 2 | Contrastive advantage is GAP-only, shrinks with spatial pooling | +0.060 GAP → +0.015 stats → +0.010 patch | `content_rank_pca` | **SAFE** |
| 3 | GroupNorm deletes the global channel signal | GAP mean R² 0.34→0.18; brain_size 0.78→0.34; **(μ,σ) alone → 0.68**; patch 0.307≈0.309 | `eval/probe_prenorm_groupnorm.py`, no-retrain forward-hook | **SAFE** — the smoking gun; no retraining confound |
| 4 | Temporal-atrophy reversal proves re-gauging | lin_gap 0.486 vs 0.173; lin_patch 0.623 vs **0.682** | `content_rank_out6`, seeds 0–4 | **SAFE** — the single most persuasive per-factor result |
| 5 | No dimensional collapse | eff-rank GAP 29 vs 13; patch 115 vs 127 | `content_rank_out4`, seeds 0/1 | **SAFE** — rules out the obvious alternative |
| 6 | Not data-limited (encoders beat pixels) | raw-pixel 16³: lin −0.44, mlp −2.5 vs model 0.36 | `content_rank_out4` | **SAFE** |
| 7 | MLP ≤ linear everywhere → no hidden nonlinear content | −0.02 to −0.11 across cells | `cv_probe_r2`, 5-fold × multi-seed | **SAFE** (read as MLP ≈ linear; early-stopping data tax) |
| 8 | Contrastive degrades voxel localisation | tissue_GM AUC 0.74 vs 0.88; deformation R² 0.007 vs 0.061 | notebook §7e, 300 samples | **SAFE for tissue/deformation** (unchanged factors) but see #14 |
| 9 | Lesion localisation gap | AUC 0.78 vs 0.95 (nulls 0.46/0.57) | notebook §7e | **SUSPECT** — checkpoint predates generator fix `7ac56a3`; lesion is a CHANGED factor → scored on OOD images |
| 10 | Style isolation win | leak c→s 0.18 vs 0.49; separation 0.158 vs −0.087 | `run_dci_compare` | **CONFOUNDED** — baseline scored 48/0 all-content, style merged; `--baseline-per-block` did not take |
| 11 | Gain leaks into content at 0.481 | gain s→s 0.522; bias 0.03, noise 0.01; style→brain_size 0.60 | `run_dci_compare` per-latent, contrastive run only | **SAFE** — single-model per-latent, no baseline comparison involved |
| 12 | Lesions ≈ 0 in both models | ~0 at every pooling | multiple | **ARTIFACT-RISK** — at 8³ the flat probe overfits (feat 22528 vs ~2000 samples); PCA-trunc shows patch R²@24 = 0.325 > full 0.228. Lesion encodability is OPEN |
| 13 | Patch false-negative rates | 42% dead @ B=32; 23% @128; 16% @400; 8³: 53/43/40% | re-measured 2026-07-18 on current generator | **SAFE** |
| 14 | Broad morphometric factors don't replicate the clean localisation gap | in_mass 0.02–0.18 vs uniform 0.145; contrastive 0.07 vs baseline 0.16–0.18 | notebook §7f, old-renderer swap (images matched) | **SAFE** — and correctly reported in the draft as a *qualification*, not buried |
| 15 | Benchmark undercounts content | 9 probed vs ~585 true shared (z_deformation 4³=64 + z_fissure 8³=512) | `eval/synthetic_dataset.py` code read | **SAFE** — structural fact about the generator |

---

## Blocking issues before submission

Ordered by how much they threaten the paper.

1. **Fair isolation re-run** (`--baseline-per-block`) — claim #10.
   Isolation is the *one axis where contrastive genuinely wins*, so it carries the paper's positive
   content. Right now it is unreportable. Either the flag is missing or the baseline config exposes
   no content/style index split — diagnose first. Expected direction: the recon baseline leaks gain
   ≥ contrastive, since recon must encode gain with no pressure to exclude it.

2. **Lesion re-run under matched generator vintage** — claim #9.
   Cheapest meaningful fix (`7e-pre` old-renderer swap on the existing checkpoint). Gives one valid
   focal-factor localisation number, which is currently the draft's weakest load-bearing evidence.

3. **ADNI replication** — the whole scope question.
   As written this is a synthetic-benchmark paper. That is publishable, but the title and abstract
   should not imply otherwise until real-data numbers exist. Minimum viable: the completeness tie
   and the localisation gap on ADNI, even without ground-truth factors (use FreeSurfer-derived
   morphometrics as surrogate factors — `freesurfer.csv` is already in the repo).

4. **Seed replication on voxel probes** — claim #8.
   Single 70/30 split. Needs seeds for error bars before any localisation claim goes in an abstract.

5. **`clean_content` replication** — claim #15's consequence.
   The nuisance shortcut plausibly *causes* the InfoNCE saturation and part of the delocalisation.
   Running the main comparison with the fields zeroed separates "the objective delocalises" from
   "the benchmark's signal is delocalised". Without this, a reviewer can attribute the whole
   localisation result to benchmark design.

6. **Encoder-only arm** — the fidelity gap.
   The draft is honest that it tests "contrastive term added to reconstruction", not the actual
   multiview CRL method. A reviewer familiar with Yao et al. will ask. `main_conv_synthetic.py`
   exists and compiles; it has not been run.

7. **Projection-head ablation** — alternative explanation for the tie.
   `--contrastive-proj-dim` implemented, not run. Cheap A/B, removes a confound a reviewer will raise.

---

## Framing risks

- **The methodology claim in §5.3 is currently broader than the evidence.** The draft asserts pooled
  linear probes are "not a neutral instrument" generally. The evidence is one backbone with
  `SplitGroupNorm`. Either (a) check how common such normalisation is in comparable SSL pipelines,
  or (b) narrow the claim to "whenever the encoder normalises away per-channel scale" — which is
  still a real and useful class. A `\todo` marks this in the draft.

- **VCI is informal.** It is currently an evaluation-design argument, not a definition with a
  proposition. Decide whether to formalise (gauge group = permutation/invertible *within* location)
  or keep informal. Formalising raises the ceiling but invites theory scrutiny the paper can't
  currently meet.

- **Venue fit.** The argument is a measurement critique with mechanism. That reads better at a CRL /
  disentanglement venue or workshop than at MICCAI/MIDL, where a negative result on synthetic data
  with no clinical endpoint is a hard sell. If MIDL/MICCAI is the target, blocking issue #3 becomes
  mandatory rather than merely important.

- **Pre-registration is an asset — keep it.** §9 commits to the pass/fail criterion in advance.
  That is unusually good practice for this literature and worth stating explicitly in the abstract
  if the ablation grid makes it into the paper.
