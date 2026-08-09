#!/usr/bin/env python
"""Why does ``selection/mcc_by_pool/patch`` peak early and then decay on contrastive runs?

The recon-only arm rises monotonically; the Barlow Twins arms spike within ~1-2k steps and
then fall back toward their step-0 value.  Before testing anything on a checkpoint, this
script's ``--calibrate`` mode establishes what the metric can even DETECT, by perturbing a
synthetic block of the same shape (N=400, 60 foreground positions x 44 channels) in each
candidate way at the real operating point (mcc ~0.84, not at the ceiling where nothing
moves).  Measured, and reproducible with ``--calibrate``:

    perturbation                                    effect on block-MCC
    ------------------------------------------------------------------
    background noise magnitude x67                  0.0000  (exactly flat)
    channel spatial maps fully decorrelated        +0.0447  (HELPS)
    invertible anisotropic map, cond 1e2            -0.2566
    invertible anisotropic map, cond 1e4            -0.3812
    signal amplitude x0.5 vs fixed noise            -0.3869

Two candidate explanations die there, and they are the two that sound most plausible:

  (DILUTION, ruled out)  "Training drops background patches (``main_multimodal.py:338``)
      while the probe pools all of them (``dci.py:334``), so unconstrained background
      drifts in as noise."  It cannot produce a decay: ``block_mcc`` standardises every
      feature before the ridge, so background MAGNITUDE is invisible, and the sweep is
      flat to four decimals across a 67x range.  Background does cost a constant level
      offset (0.835 foreground-only vs 0.719 all-positions here) because it adds
      features, but the position count does not change during training, so the offset
      cannot trend.

  (DECORRELATION, ruled out)  "``bt_lambda 1`` puts 1892 off-diagonal terms against 44
      on-diagonal at d=44, orthogonalising the channels' spatial maps and destroying the
      multi-channel co-activation a local factor needs."  Measured the other way: a fully
      channel-private encoding scores HIGHER than a fully redundant one (0.873 vs 0.829).
      A linear probe does not need cross-channel redundancy.

  (CONDITIONING, survives)  An exactly INVERTIBLE map costs up to 0.48 of block-MCC.
      Block-identifiability is defined up to invertible maps, but that guarantee is
      vacuous at N~1500: once the map pushes a factor below the sampling noise, no
      estimator recovers it.  Verified: a PCA-whitened probe does not bring it back at
      any number of components (k=50..319 all fail), so this is not a probe-choice
      artifact and there is no probe-side control that separates "re-gauged into an
      ill-conditioned basis" from "destroyed".

So the live question is not which probe to use.  It is whether the content block's own
geometry is collapsing, which is measurable WITHOUT a probe and is actionable through the
only anti-collapse force in this config (``bt_std_coeff``'s variance hinge, plus the
GAP-BT term).  The three tests below follow from that.

MEASURED on the first real run (``synthetic-causal-clean-content-mse-real-lambda-01-normalized``,
88k steps): CONDITIONING did not happen either.  ``content_rank`` ROSE 18.7 -> 39.7 (+112%)
and its post-peak correlation with the MCC was +0.045 (p=0.77).  What that run showed
instead, and what to check first on any other:

  * ``on_diag`` 28.5 -> 0.006 (90%-saturated by step 450) and ``off_diag`` 4.76 -> 0.023
    with 94% of its descent done BY the MCC peak.  The correlation terms are converged
    before the decay starts, so they cannot be driving it.  Read a term's post-peak
    DESCENT, not its level -- ``Loss/Contrastive`` was still 3.1x ``Loss/Recon`` at the
    end while contributing almost no gradient.
  * The decay is NOT patch-specific: patch -0.049, gap -0.039, stats -0.036 over the same
    window.  Any explanation that turns on patch pooling or on local-vs-global content is
    excluded by that alone.
  * ``feat_std_mean`` 0.29 -> 66.9.  Nothing in this objective bounds the feature scale:
    on_diag/off_diag are correlations, sim is normalised by a detached denominator, and
    the variance hinge ``relu(1 - std)`` is ONE-SIDED.  Calibration case G confirms this
    cannot move the metric directly (StandardScaler), but it is an uncontrolled degree of
    freedom worth knowing about.
  * ``info_all`` fell 2.9% against the MCC's 5.4%, so roughly half the drop is matched by
    genuine loss on the all-channels capacity measure.

The whole phenomenon lived in ~0.04 of range: first 0.8606 -> peak 0.8994 -> final 0.8508,
i.e. the final checkpoint is 0.0098 BELOW an essentially untrained encoder.  Get error bars
(``block_mcc`` returns a per-seed std that the selection path does not log) before
engineering around a difference this size.

  1. CURVES (start here; no GPU, no checkpoint, answers in seconds).  Reads the run's
     TensorBoard scalars.  Is the MCC peak where ``on_diag`` saturates -- i.e. where
     alignment is achieved and nearly all remaining gradient is decorrelation?  Does
     ``selection/content_rank`` fall in lockstep with the patch MCC, which given the
     calibration above is the single most diagnostic pairing available?  Is
     ``feat_std_mean`` anywhere near the hinge's target of 1?  And what is the TRUE
     Recon:Contrastive magnitude ratio -- both ``scale_*`` weights are nominally 1, but
     BT's raw value at d=44 sums ~1936 terms, so replenishment may not be competing with
     erosion at all.

  2. STRATA.  Recomputes the identical block-MCC on position subsets: all patches
     (reproduces the logged number), the exact training ``_keep_pos`` rule, strictly
     covered, and background-only.  Given (DILUTION) is ruled out this no longer decides
     anything on its own, but it localises the decay -- if foreground falls as much as
     the whole volume, the objective is eroding content it does see -- and the
     background-only column says whether background positions carry factor information
     at all.  Reported as a GAP over a permutation null, because ``mcc_by_pool/*`` is
     logged RAW and a falling rank deflates the null too, so part of the drop is free.

  3. GEOMETRY.  Probe-free measurement of the content block: participation-ratio
     effective rank, the condition number at 95% energy, and the MCC-vs-PCA-rank curve
     (how many leading directions are needed to reach the block's own ceiling).  If the
     late checkpoint needs more components for the same score, or its effective rank has
     collapsed, conditioning is the mechanism and ``bt_std_coeff`` / ``bt_gap_weight``
     are the levers.

Usage
-----
    # what the metric can detect -- pure numpy, no run needed
    python -m eval.patch_mcc_decay --calibrate

    # timing + term balance from the logs alone
    python -m eval.patch_mcc_decay --run-dir results/synthetic/BT_RUN --tests curves

    # everything, best (near-peak) vs final checkpoint, against the recon arm
    python -m eval.patch_mcc_decay --run-dir results/synthetic/BT_RUN \
        --compare-run-dir results/synthetic/RECON_ONLY --num-samples 1500

Related: ``eval.bt_term_balance`` sizes the four BT coefficients at a fixed checkpoint
(use it to act on test 1's ratio); ``eval.patch_background_diagnostic`` stratifies the
LOSS and GRADIENT by patch coverage (this script stratifies the PROBE).

NOTE on checkpoint vintage: ``render_structure`` was fixed in commit 7ac56a3 for
temporal_atrophy / sulcal_widening / lesion.  A checkpoint trained before it saw
different rendering for those factors, so scoring it against the current generator is
invalid for them.  brain_size, lr_asymmetry and the style-leak numbers are unaffected.
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from eval.identifiability_metrics import block_mcc

# torch is imported lazily inside the extraction path only, so the scoring, geometry and
# verdict logic stay unit-testable on plain numpy (same convention as eval.run_dci_compare).

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Matches identifiability_metrics._make_regressor, so the view-alignment map is fitted with
# the same regularisation path as every other ridge in this project.
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


# --------------------------------------------------------------------------- #
# Test 1 — curves.  No checkpoint, no GPU.
# --------------------------------------------------------------------------- #


def read_tb_scalars(tb_dir):
    """All scalar tags from a run's tensorboard dir → ``{tag: (steps, values)}``."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:  # writing needs it; reading is a separate install on some envs
        raise SystemExit("The curves test needs `pip install tensorboard`. The other tests do not.") from exc

    acc = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        out[tag] = (
            np.array([e.step for e in events], dtype=np.float64),
            np.array([e.value for e in events], dtype=np.float64),
        )
    return out


def _saturation_step(steps, values, tol):
    """First step at which a decaying series has done ``1 - tol`` of its total descent.

    Defined on the descent itself rather than on a slope threshold, so it does not depend
    on the logging interval: ``v_start - v_min`` is the total work available and this
    returns where only ``tol`` of it is left.
    """
    if len(values) < 3:
        return None
    v_start, v_min = values[0], values.min()
    if v_start - v_min <= 0:
        return None
    hit = np.where(values <= v_min + tol * (v_start - v_min))[0]
    return float(steps[hit[0]]) if len(hit) else None


def _tail_median(steps, values, frac=0.2):
    if len(values) == 0:
        return float("nan")
    return float(np.median(values[steps >= steps[-1] - frac * (steps[-1] - steps[0])]))


def test_curves(run_dir, level=0, sat_tol=0.1):
    tb_dir = os.path.join(run_dir, "tensorboard")
    if not os.path.isdir(tb_dir):
        logger.error("No tensorboard dir at %s — skipping the curves test.", tb_dir)
        return
    scalars = read_tb_scalars(tb_dir)

    print(f"\n{'=' * 78}\nTEST 1 — CURVES  ({os.path.basename(run_dir.rstrip('/'))})\n{'=' * 78}")

    mcc_tag = "selection/mcc_by_pool/patch"
    if mcc_tag not in scalars:
        logger.error("%s not logged — was this a synthetic run with select_by_synthetic_dci?", mcc_tag)
        return
    m_steps, m_vals = scalars[mcc_tag]
    peak_i = int(np.argmax(m_vals))
    peak_step, peak_val, final_val = m_steps[peak_i], m_vals[peak_i], m_vals[-1]
    print(
        f"\n  patch MCC   first {m_vals[0]:.4f} @ {m_steps[0]:.0f}   peak {peak_val:.4f} @ {peak_step:.0f}   "
        f"final {final_val:.4f} @ {m_steps[-1]:.0f}"
    )
    print(f"              peak → final {final_val - peak_val:+.4f};  " f"first → final {final_val - m_vals[0]:+.4f}")

    # --- Rank collapse is the mechanism the calibration leaves standing. ---
    if "selection/content_rank" in scalars:
        r_steps, r_vals = scalars["selection/content_rank"]
        at_peak = np.interp(peak_step, r_steps, r_vals)
        print(
            f"\n  content_rank  @peak {at_peak:.3f} → final {r_vals[-1]:.3f} "
            f"({100 * (r_vals[-1] - at_peak) / max(abs(at_peak), 1e-9):+.1f}%)"
        )
        common = m_steps[m_steps >= peak_step]
        if len(common) >= 5:
            rho, p = spearmanr(m_vals[m_steps >= peak_step], np.interp(common, r_steps, r_vals))
            print(f"    post-peak Spearman(patch MCC, content_rank) = {rho:+.3f} (p={p:.3g}, n={len(common)})")
            if rho > 0.5:
                print("    → they fall together: CONDITIONING/rank collapse, the one mechanism that")
                print("      survived calibration. Levers: bt_std_coeff (the only anti-collapse term),")
                print("      bt_gap_weight, and the recon:contrastive ratio below.")

    # --- Does the peak sit where alignment saturates and decorrelation takes over? ---
    on_tag, off_tag = f"Contrastive/on_diag_loss_L{level}", f"Contrastive/off_diag_loss_L{level}"
    if on_tag in scalars and off_tag in scalars:
        o_steps, o_vals = scalars[on_tag]
        f_steps, f_vals = scalars[off_tag]
        sat = _saturation_step(o_steps, o_vals, sat_tol)
        print(
            f"\n  on_diag     {o_vals[0]:.3f} → {o_vals[-1]:.3f};  "
            f"{100 * (1 - sat_tol):.0f}%-saturated @ {'n/a' if sat is None else f'{sat:.0f}'}"
        )
        print(f"  off_diag    {f_vals[0]:.3f} → {f_vals[-1]:.3f}")
        if f_vals[0] - f_vals.min() > 0:
            done = f_vals[0] - np.interp(peak_step, f_steps, f_vals)
            print(
                f"              {100 * (1 - done / (f_vals[0] - f_vals.min())):.0f}% of its descent happens "
                "AFTER the MCC peak"
            )
        if sat is not None:
            ratio = peak_step / max(sat, 1.0)
            print(
                f"\n  → MCC peak @ {peak_step:.0f} vs on_diag saturation @ {sat:.0f} (ratio {ratio:.2f}): "
                + ("aligned" if 0.4 <= ratio <= 2.5 else "NOT aligned — the peak is not the saturation point")
            )
    else:
        logger.warning("BT diagnostics (%s) not in the logs — is this an InfoNCE run?", on_tag)

    # --- Which BT term is still exerting force after the peak? ---
    #
    # on_diag and off_diag are CORRELATIONS, so they bottom out and stop asking for
    # anything; sim is a DISTANCE (Lemma C.2 needs g_k(x_k) = g_k'(x_k') a.s., which a
    # correlation cannot see) and has no informativeness counterpart, so if it is still
    # descending past the peak it is the only one-way contraction left in the objective.
    # A term whose value is large but whose gradient has vanished is not eroding anything
    # — read the post-peak DESCENT, not the level.
    rows, shares = [], []
    for name in ("on_diag_loss", "off_diag_loss", "sim_loss", "var_loss"):
        tag = f"Contrastive/{name}_L{level}"
        if tag not in scalars:
            continue
        t_steps, t_vals = scalars[tag]
        total = t_vals[0] - t_vals.min()
        share = (np.interp(peak_step, t_steps, t_vals) - t_vals[-1]) / total if total > 0 else 0.0
        shares.append(share)
        rows.append(
            f"    {name:<14}{t_vals[0]:>12.4g} → {t_vals[-1]:<12.4g}"
            f"post-peak {100 * share:>5.1f}%" + ("   ← still working" if share > 0.25 else "")
        )
    if rows:
        print("\n  BT terms, post-peak descent (share of each term's own total, after the MCC peak):")
        print("\n".join(rows))
        if max(shares) <= 0.25:
            print("\n  → EVERY term converged BEFORE the peak. The contrastive objective is at a fixed")
            print("    point while the metric decays, so it is not the cause: what follows the peak is")
            print("    recon/VQ-phase drift. Changing the contrastive loss cannot fix this curve.")

    # --- Is the variance hinge, the only anti-collapse term, actually holding? ---
    std_tag = f"Contrastive/feat_std_mean_L{level}"
    if std_tag in scalars:
        s_steps, s_vals = scalars[std_tag]
        tail = _tail_median(s_steps, s_vals)
        print(
            f"\n  feat_std_mean (hinge target 1.0):  {s_vals[0]:.4f} → {tail:.4f}"
            + ("   ← ≪ 1: the hinge is NOT holding; raise bt_std_coeff" if tail < 0.2 else "")
        )

    # --- Are the two objectives actually balanced? ---
    if "Loss/Recon" in scalars and "Loss/Contrastive" in scalars:
        r_tail = _tail_median(*scalars["Loss/Recon"])
        c_tail = _tail_median(*scalars["Loss/Contrastive"])
        print(
            f"\n  Loss/Recon {r_tail:.5f}   Loss/Contrastive {c_tail:.5f}   "
            f"→ contrastive:recon = {c_tail / max(r_tail, 1e-12):.1f}:1"
        )
        print("    (both scale_* weights are nominally 1 — this is the ratio that acts. Recon is the")
        print("     only force replenishing what the contrastive term removes.)")

    # --- Migration or loss?  The reading is stated at main_multimodal.py:2556. ---
    if "selection/info_all" in scalars:
        ia_steps, ia_vals = scalars["selection/info_all"]
        at_peak = np.interp(peak_step, ia_steps, ia_vals)
        ia_drop = (at_peak - ia_vals[-1]) / max(abs(at_peak), 1e-9)
        mcc_drop = (peak_val - final_val) / max(abs(peak_val), 1e-9)
        print(f"\n  info_all    @peak {at_peak:.4f} → final {ia_vals[-1]:.4f}")
        print(
            f"  → info_all fell {100 * ia_drop:+.1f}% vs patch MCC {100 * mcc_drop:+.1f}%: "
            + ("REAL LOSS" if ia_drop > 0.5 * mcc_drop else "MIGRATION / re-gauging, not loss")
        )

    for pool in ("gap", "stats"):
        tag = f"selection/mcc_by_pool/{pool}"
        if tag in scalars:
            p_steps, p_vals = scalars[tag]
            at_peak = np.interp(peak_step, p_steps, p_vals)
            print(f"  mcc {pool:<6} @patch-peak {at_peak:.4f} → final {p_vals[-1]:.4f} ({p_vals[-1] - at_peak:+.4f})")

    # --- Is the STYLE BOTTLENECK forcing view-specific detail into content? ---
    #
    # The decoder has to produce BOTH views from shared content plus per-view style. If
    # style is too narrow to express the view difference, the only remaining way to lower
    # the reconstruction loss is to put view-specific information into the CONTENT
    # channels. That is not information leaving the model, it is capacity being
    # reallocated — which is why it can move the MCC much more than it moves info_all.
    #
    # content→view accuracy is the direct witness: it is 0.5 when content is view-
    # invariant and climbs toward 1.0 as content absorbs view-specific signal. A rise
    # here across the post-peak window, while the MCC falls, IS the mechanism.
    print("\n  Style bottleneck — is content absorbing view-specific detail?")
    _any = False
    for tag, label, ref in (
        ("selection/content_view_acc", "content→view acc", "0.5 = view-invariant"),
        ("selection/style_sufficiency", "style→style suff", "1.0 = style captures its factors"),
        ("selection/style_modality", "style_modality", "higher = better"),
        ("selection/style_to_content_leak", "style→content leak", "lower = better"),
        ("selection/content_to_style_leak", "content→style leak", "lower = better"),
    ):
        if tag not in scalars:
            continue
        _any = True
        t_steps, t_vals = scalars[tag]
        at_peak = np.interp(peak_step, t_steps, t_vals)
        print(
            f"    {label:<20}@peak {at_peak:>7.4f} → final {t_vals[-1]:>7.4f} " f"({t_vals[-1] - at_peak:+.4f})   {ref}"
        )
    if "selection/content_view_acc" in scalars:
        cv_steps, cv_vals = scalars["selection/content_view_acc"]
        cv_peak = float(np.interp(peak_step, cv_steps, cv_vals))
        rise, excess = cv_vals[-1] - cv_peak, cv_vals[-1] - 0.5
        # "High" and "rising" are different findings and only the second can explain a
        # decay that happens over the same window. Keep them apart.
        if rise > 0.02:
            print("\n    → content→view RISES across the post-peak window while the MCC falls. Content")
            print("      is progressively absorbing view-specific detail — a style CAPACITY problem.")
            print("      Widen style (content_size down at fixed hidden width, more style codebook")
            print("      entries, or quantize_style off) to remove the incentive.")
        elif excess > 0.1:
            print(f"\n    → content→view is PINNED at {cv_vals[-1]:.4f} and flat ({rise:+.4f}), so it does")
            print("      NOT explain the decay — but content is not view-invariant at ANY point, which")
            print("      breaks the premise of the identifiability claim (Yao Lemma C.2 needs")
            print("      g_k(x_k) = g_k'(x_k') a.s., not merely correlated across views).")
            print("      Mechanism is documented at losses.py:1176: the per-channel standardisation")
            print("      makes on_diag/off_diag BLIND to a per-view constant offset (measured there:")
            print("      on_diag unchanged to 4 dp while the view probe goes 0.503 → 1.000). Only the")
            print("      `sim` distance term can see it — check bt_sim_coeff, and note that")
            print("      bt_sim_normalize divides by the total across-sample-and-position variance,")
            print("      so a consistent offset can read ~0 in the loss and still be perfectly")
            print("      separable at the `stats` pooling content_view is scored on.")
        else:
            print("\n    → content→view is flat and near chance: content IS view-invariant, and the")
            print("      style bottleneck does not explain this decay.")
    elif not _any:
        logger.warning("No selection/style_* or content_view_acc tags — style-side scoring was off for this run.")


# --------------------------------------------------------------------------- #
# Extraction — one pass, everything tests 2 and 3 need
# --------------------------------------------------------------------------- #


def extract_patch_block(model, dataset, device, grid, level, batch_size=16, num_workers=0):
    """Level-``level`` content block at patch pooling, with per-position brain coverage.

    Mirrors ``eval.dci._extract_synthetic_representations`` — same ``pool_only=True,
    patch_grid=grid`` forward, same Gumbel-mask content split, view 1 only — but also
    returns mask coverage per (sample, position) so the block can be sliced by position.

    Returns ``(content, gt_content, coverage)`` shaped ``(N, P, C)``, ``(N, F)``,
    ``(N, P)``.  Position order is row-major over (D, H, W) in both, because the model
    pools with ``adaptive_avg_pool3d(...).flatten(2)`` (``vqvae.py:1356``) and so does the
    coverage below.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from eval.dci import _parse_content_indices

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats, gts, covs = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"]
            n_views = len(imgs)
            x = torch.cat(imgs, dim=0).to(device)

            enc_out = model(x, pool_only=True, n_views=n_views, patch_grid=tuple(grid))
            feat = enc_out[2][level]  # (n_views*B, C, P)
            soft_masks = enc_out[6] if len(enc_out) > 6 else {}
            b = feat.shape[0] // n_views

            if isinstance(soft_masks, dict) and level in soft_masks:
                mask = soft_masks[level]
                raw = torch.where((mask[0] if isinstance(mask, tuple) else mask).bool())[-1]
            else:
                raw = None
            content_idx = _parse_content_indices(raw) or list(range(feat.shape[1]))

            # (B, C_content, P) → (B, P, C_content): position-major, so a position subset
            # is a plain slice and the later flatten reproduces dci.py's patch-major layout.
            feats.append(feat[:b, content_idx, :].permute(0, 2, 1).cpu().numpy())

            m = batch["mask"][0].to(device).float()
            if m.dim() == 4:
                m = m.unsqueeze(1)
            covs.append(F.adaptive_avg_pool3d(m, tuple(grid)).flatten(1).cpu().numpy())

            gts.append(batch["gt_latents"]["z_content"].numpy())

    return np.concatenate(feats), np.concatenate(gts), np.concatenate(covs)


# --------------------------------------------------------------------------- #
# Test 4 — view gap.  How BIG is the content/view dependence, not whether it exists.
# --------------------------------------------------------------------------- #
#
# `selection/content_view_acc` is a multivariate L2-regularised logistic probe on ~176
# standardised features with ~2000 rows per view (identifiability_metrics.py:136). With a
# BINARY label that saturates at 1.0 the moment any consistent linear direction separates
# the views, at essentially any effect size, and then has no headroom left. So 1.0 proves
# that g_1(x) = g_2(x') a.s. fails and says nothing about magnitude — and with
# --separate-encoders, g_1 and g_2 are different networks, so exact agreement was never
# on offer.
#
# What matters is which KIND of failure it is:
#
#   OFFSET-ONLY   content_v2 ~ content_v1 + b for a constant b. The content INFORMATION is
#                 the same in both views. This is provably free for every metric in this
#                 project: block_mcc, DCI and mcc_by_pool are all computed on VIEW 1 ONLY
#                 and StandardScaler centres each feature, so a per-view constant cannot
#                 move them by construction. Nothing to fix.
#   STRUCTURAL    the views still separate after each view's mean is removed, so content
#                 carries view-specific information beyond a shift. This is the case that
#                 justifies acting on the invariance term.
#
# The per-view-mean-removed probe is what tells them apart, and per-channel AUC says how
# widely it is spread: losses.py:1180 records "every content channel had view-AUC 1.000
# under BT and 0/44 above 0.7 under VICReg", so 1-of-44 and 44-of-44 are both live
# outcomes on this project and they mean very different things.


def _auc(x, labels):
    """Direction-free AUC of a single scalar, via the Mann-Whitney rank statistic."""
    from scipy.stats import rankdata

    r = rankdata(x)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n0 == 0 or n1 == 0:
        return float("nan")
    a = (r[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1)
    return float(max(a, 1.0 - a))


def _grouped_view_acc(X, labels, groups, kind="linear", seeds=(0, 1), n_splits=5):
    """View-classification accuracy with a subject's TWO ROWS kept in the same fold.

    The shipped ``cv_probe_acc`` uses StratifiedKFold on the stacked pair, so subject i's
    view-1 row can train while its view-2 row tests. The classifier then learns subject
    identity off the shared content and predicts the paired test row's OPPOSITE label,
    which puts the floor BELOW chance: measured 0.3550 on statistically identical views
    where the true chance is 0.5 (grouped: 0.4975; unpaired controls: 0.4987). It does not
    inflate a genuine positive — with a real offset, shipped 0.9062 vs grouped 0.9019 —
    so a high ``content_view_acc`` is real, but its floor is not 0.5 and "how far above
    chance" cannot be read off it.

    ``kind="nonlinear"`` matters for a separate reason: logistic regression on standardised
    features can only respond to a MEAN SHIFT. A view difference in variance or covariance
    is invisible to it by construction, so a linear probe cannot distinguish "offset only"
    from "no structural difference" — only a nonlinear probe can.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    accs = []
    for seed in seeds:
        # GroupKFold is deterministic, so vary folds by permuting the group ids.
        g = np.random.RandomState(seed).permutation(int(groups.max()) + 1)[groups]
        fold = []
        for tr, te in GroupKFold(n_splits=n_splits).split(X, labels, g):
            sc = StandardScaler().fit(X[tr])
            clf = (
                LogisticRegression(max_iter=1000, random_state=seed)
                if kind == "linear"
                else MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, early_stopping=True, random_state=seed)
            )
            clf.fit(sc.transform(X[tr]), labels[tr])
            fold.append(accuracy_score(labels[te], clf.predict(sc.transform(X[te]))))
        accs.append(float(np.mean(fold)))
    return float(np.mean(accs))


def _grouped_linear_aligned_acc(c1, c2, labels, groups, seeds=(0, 1), n_splits=5):
    """View accuracy after aligning view 2 onto view 1 with a per-fold LINEAR map.

    The map is fitted on training subjects only and applied to everyone, so an alignment
    that only works by memorising the test rows cannot help. If the views stop being
    separable here, they differ by an invertible linear transform — information-preserving,
    and the exact equivalence class block-identifiability is stated up to.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    n = c1.shape[0]
    accs = []
    for seed in seeds:
        g = np.random.RandomState(seed).permutation(n)[np.arange(n)]
        fold = []
        for tr_s, te_s in GroupKFold(n_splits=n_splits).split(np.arange(n), groups=g):
            aligner = RidgeCV(alphas=RIDGE_ALPHAS).fit(c2[tr_s], c1[tr_s])
            c2a = np.asarray(aligner.predict(c2))
            X = np.vstack([c1, c2a])
            tr = np.concatenate([tr_s, tr_s + n])
            te = np.concatenate([te_s, te_s + n])
            sc = StandardScaler().fit(X[tr])
            clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, early_stopping=True, random_state=seed)
            clf.fit(sc.transform(X[tr]), labels[tr])
            fold.append(accuracy_score(labels[te], clf.predict(sc.transform(X[te]))))
        accs.append(float(np.mean(fold)))
    return float(np.mean(accs))


def view_gap_stats(c1, c2, seeds=(0, 1)):
    """Effect size of the content block's view dependence. Pure numpy/sklearn.

    ``c1``/``c2`` are the same content block for view 1 and view 2, ``(N, D)``.
    """
    from eval.identifiability_metrics import cv_probe_acc

    c1, c2 = np.asarray(c1, dtype=np.float64), np.asarray(c2, dtype=np.float64)
    n = c1.shape[0]
    labels = np.array([0] * n + [1] * n)
    groups = np.concatenate([np.arange(n), np.arange(n)])

    stacked = np.vstack([c1, c2])
    # Reported so this reconciles with the selection/content_view_acc curve, not because
    # it is the number to judge on — see _grouped_view_acc.
    acc_shipped = cv_probe_acc(stacked, labels, seeds=seeds)["mean"]
    acc_raw = _grouped_view_acc(stacked, labels, groups, "linear", seeds)
    # Remove EACH view's own mean, then ask again — linear first (must collapse if the raw
    # signal was a shift, since that is all a linear probe can see), then nonlinear, which
    # is the only one of the two that can detect a structural difference.
    centred = np.vstack([c1 - c1.mean(0), c2 - c2.mean(0)])
    acc_centred = _grouped_view_acc(centred, labels, groups, "linear", seeds)
    acc_centred_nl = _grouped_view_acc(centred, labels, groups, "nonlinear", seeds)

    # Between-view variance share and Cohen's d, per feature.
    m1, m2 = c1.mean(0), c2.mean(0)
    sd_pool = np.sqrt(0.5 * (c1.var(0, ddof=1) + c2.var(0, ddof=1)))
    cohen_d = np.abs(m1 - m2) / np.maximum(sd_pool, 1e-12)
    grand = stacked.mean(0)
    ss_between = n * ((m1 - grand) ** 2 + (m2 - grand) ** 2)
    ss_total = ((stacked - grand) ** 2).sum(0)
    eta2 = ss_between / np.maximum(ss_total, 1e-12)

    aucs = np.array([_auc(stacked[:, j], labels) for j in range(stacked.shape[1])])
    # Sample-wise tracking after the shift is removed: do the two views carry the same
    # per-subject signal in each feature?
    a, b = c1 - m1, c2 - m2
    denom = a.std(0) * b.std(0)
    track = np.where(denom > 1e-12, (a * b).mean(0) / np.maximum(denom, 1e-12), np.nan)

    # AFFINE control. With --separate-encoders the two views have different encoders and
    # therefore different output SCALES. A per-view scale is an affine nuisance carrying no
    # information, exactly like an offset — but mean-removal does not remove it, so the
    # centred probe above would report it as "structural". It also shows up specifically in
    # std/amax/amin while leaving a mean-only (gap) probe at the floor, which is precisely
    # the signature this project produces. Standardising each view removes scale as well as
    # shift, so whatever survives HERE is not affine and is the only thing an invariance
    # term needs to fix.
    z1 = (c1 - m1) / np.maximum(c1.std(0), 1e-12)
    z2 = (c2 - m2) / np.maximum(c2.std(0), 1e-12)
    acc_affine_nl = _grouped_view_acc(np.vstack([z1, z2]), labels, groups, "nonlinear", seeds)
    scale_ratio = c2.std(0) / np.maximum(c1.std(0), 1e-12)

    # FULL-LINEAR control. Standardising each view removes only a DIAGONAL affine map. A
    # per-view invertible MATRIX is precisely the nuisance block-identifiability permits
    # (the theory identifies the block up to an invertible map), and it would still show
    # up as "non-affine" above. So align view 2 onto view 1 with a linear map fitted on
    # the training rows only, and ask again: what survives THIS is not a linear nuisance
    # and is the only thing an invariance term is needed for.
    acc_linear_nl = _grouped_linear_aligned_acc(c1, c2, labels, groups, seeds)

    return {
        "acc_shipped": acc_shipped,
        "acc_raw": acc_raw,
        "acc_centred": acc_centred,
        "acc_centred_nl": acc_centred_nl,
        "acc_affine_nl": acc_affine_nl,
        "acc_linear_nl": acc_linear_nl,
        "scale_ratio": scale_ratio,
        "auc": aucs,
        "eta2": eta2,
        "cohen_d": cohen_d,
        "track": track,
        "n_feat": stacked.shape[1],
    }


def _print_view_gap(label, pooling, s):
    aucs, eta2 = s["auc"], s["eta2"]
    print(f"\n  {label} — view gap @ {pooling} pooling ({s['n_feat']} features)")
    print(f"    view probe   shipped {s['acc_shipped']:.4f}  (== selection/content_view_acc; floor is NOT 0.5)")
    print(f"                 subject-grouped, linear    raw {s['acc_raw']:.4f}   mean-removed {s['acc_centred']:.4f}")
    print(f"                 subject-grouped, NONLINEAR                        mean-removed {s['acc_centred_nl']:.4f}")
    print(f"                 subject-grouped, NONLINEAR         per-view STANDARDISED {s['acc_affine_nl']:.4f}")
    print(
        f"                 subject-grouped, NONLINEAR      LINEARLY ALIGNED v2->v1 {s['acc_linear_nl']:.4f}  <- decisive"
    )
    _sr = s["scale_ratio"]
    print(
        f"    per-view scale      std(v2)/std(v1) median {np.nanmedian(_sr):.3f}  "
        f"range [{np.nanmin(_sr):.3f}, {np.nanmax(_sr):.3f}]"
    )
    print(
        f"    per-feature AUC     median {np.nanmedian(aucs):.3f}   "
        f">0.7: {int((aucs > 0.7).sum())}/{len(aucs)}   "
        f">0.9: {int((aucs > 0.9).sum())}/{len(aucs)}   "
        f"=1.00: {int((aucs >= 0.999).sum())}/{len(aucs)}"
    )
    print(
        f"    between-view var    eta^2 mean {np.nanmean(eta2):.4f}  median {np.nanmedian(eta2):.4f}  max {np.nanmax(eta2):.4f}"
    )
    print(f"    Cohen's d           median {np.nanmedian(s['cohen_d']):.3f}  max {np.nanmax(s['cohen_d']):.3f}")
    print(f"    cross-view tracking median r {np.nanmedian(s['track']):+.3f} (after removing each view's mean)")
    if s.get("by_stat"):
        cells = "   ".join(f"{k} {v:.3f}" for k, v in s["by_stat"].items())
        print(f"    non-affine BY STAT  {cells}   (each per-view standardised; floor ~0.50)")
        _ex = {k: v for k, v in s["by_stat"].items() if k != "mean" and v >= 0.70}
        if _ex and set(_ex) <= {"amax", "amin"}:
            print("      ← confined to the ORDER STATISTICS. amax/amin are outlier-driven and scale-")
            print("        sensitive (this run reaches feat_std 66.9), so this is a much weaker")
            print("        finding than the same signal in std would be.")


def view_gap_verdict(s):
    """Report the OFFSET and STRUCTURAL components separately — they are independent.

    A run can have either, both, or neither, so collapsing them onto one axis mislabels
    the pure cases. The linear mean-removed probe cannot decide the structural question at
    all: logistic regression on standardised features responds only to a mean shift, so
    removing the means must collapse it whether or not structure remains. Only the
    nonlinear mean-removed probe can, and its empirical floor is ~0.50 (validated on
    matched synthetic controls).
    """
    # The raw probe must itself be above chance first: the centred LINEAR probe sits
    # BELOW 0.5 whenever the views share content (it learns subject identity and inverts
    # the paired row), so a raw-minus-centred "drop" appears even in a fully invariant
    # control. Requiring acc_raw > 0.60 removes that false positive.
    offset = s["acc_raw"] > 0.60 and ((s["acc_raw"] - s["acc_centred"]) > 0.10 or float(np.nanmax(s["cohen_d"])) > 0.3)
    nl = s["acc_centred_nl"]
    print("\n    → OFFSET component:     ", end="")
    if offset:
        print(
            f"PRESENT (linear probe {s['acc_raw']:.3f} → {s['acc_centred']:.3f} on mean removal, "
            f"max Cohen's d {np.nanmax(s['cohen_d']):.2f})"
        )
        print("        A per-view constant carries NO view-specific information, and every metric")
        print("        here is scored on VIEW 1 with per-feature standardisation — so it cannot move")
        print("        block_mcc / DCI / mcc_by_pool at all. Exactly zero, not approximately.")
    else:
        print("none detected")
    print("      BEYOND-SHIFT:         ", end="")
    print(f"{'none' if nl < 0.55 else 'yes'} (nonlinear, mean-removed {nl:.3f}; floor ~0.50)")
    aff, lin = s["acc_affine_nl"], s["acc_linear_nl"]
    print(f"      NON-DIAGONAL:         {'none' if aff < 0.55 else 'yes'} (per-view standardised {aff:.3f})")
    # The DECIDING number. Standardising each view removes only a DIAGONAL map; a per-view
    # invertible MATRIX still reads as "non-affine" there — validated, a random invertible
    # matrix scores 0.954 standardised and 0.497 linearly aligned. And an invertible matrix
    # is exactly the equivalence class block-identifiability is stated up to, so it is not
    # a violation at all.
    print("      NON-LINEAR component: ", end="")
    if lin < 0.55:
        print(f"none detected (linearly aligned {lin:.3f})")
        if aff >= 0.55:
            print("        The signal above is a per-view invertible LINEAR map — different bases, not")
            print("        different information. That is precisely the equivalence class")
            print("        block-identifiability is stated up to, and with --separate-encoders (two")
            print("        distinct networks) it is the expected outcome, not a defect.")
        print("        Nothing here needs an invariance term; judge the objective on the style side.")
        print("        (A null here is a LOWER bound: the MLP detector misses subtle nonlinear")
        print("         differences — validated, it fails to flag a tanh applied to one view.)")
    elif lin < 0.70:
        print(f"WEAK ({lin:.3f} vs a ~0.50 floor)")
        print("        A little view information survives even a fitted linear alignment. Weigh it")
        print("        against eta^2 and the AUC census before spending a run on it.")
    else:
        print(f"STRONG ({lin:.3f} vs a ~0.50 floor)")
        print("        Content carries view-specific information that NO linear map removes. This is")
        print("        the case that justifies raising bt_sim_coeff / disabling bt_sim_normalize —")
        print("        the sim distance is the only term that can see it (losses.py:1176) — and the")
        print("        case where widening style has a real incentive to remove.")
        print("        NOTE: check the gap-pooling block above first. gap/patch are pooled BEFORE")
        print("        SplitGroupNorm and stats is read AFTER it (vqvae.py:1346), so a result that")
        print("        appears only at stats is attributable to that per-sample normalisation rather")
        print("        than to the encoder.")


def stats_group_breakdown(c1, c2, seeds=(0, 1)):
    """Which of [mean | std | amax | amin] carries the non-affine view signal?

    ``_pool_and_split_view`` builds the stats block stat-major — 44 means, then 44 stds,
    then amax, then amin (dci.py:277) — so the block splits cleanly into four equal groups.
    The distinction matters: a difference in ``std`` is a real difference in the spatial
    DISTRIBUTION of a content channel, whereas ``amax``/``amin`` are order statistics over
    the map, dominated by outliers and highly sensitive to the unbounded feature scale
    (this project's runs reach feat_std 66.9). Non-affine signal confined to the extremes
    is a much weaker finding than the same signal in std.
    """
    c1, c2 = np.asarray(c1, dtype=np.float64), np.asarray(c2, dtype=np.float64)
    d = c1.shape[1]
    if d % 4:
        return {}
    n_ch = d // 4
    n = c1.shape[0]
    labels = np.array([0] * n + [1] * n)
    groups = np.concatenate([np.arange(n), np.arange(n)])
    out = {}
    for gi, name in enumerate(("mean", "std", "amax", "amin")):
        sl = slice(gi * n_ch, (gi + 1) * n_ch)
        a, b = c1[:, sl], c2[:, sl]
        za = (a - a.mean(0)) / np.maximum(a.std(0), 1e-12)
        zb = (b - b.mean(0)) / np.maximum(b.std(0), 1e-12)
        out[name] = _grouped_view_acc(np.vstack([za, zb]), labels, groups, "nonlinear", seeds)
    return out


def test_viewgap(label, model, dataset, device, level, batch_size, seeds):
    from eval.dci import _extract_synthetic_representations

    out = {}
    for pooling in ("stats", "gap"):
        # "stats" is the pooling selection/content_view_acc itself uses
        # (run_dci_compare.py:727), so the raw number here should reproduce that curve.
        ld, _, _, _ = _extract_synthetic_representations(model, dataset, device, batch_size, 0, pooling=pooling)
        if level not in ld:
            continue
        c1, _, c2, _, _ = ld[level]
        if c1 is None or c2 is None:
            logger.warning("No second view at %s pooling — view gap needs both.", pooling)
            continue
        s = view_gap_stats(c1, c2, seeds=seeds)
        if pooling == "stats":
            s["by_stat"] = stats_group_breakdown(c1, c2, seeds=seeds)
        _print_view_gap(label, pooling, s)
        out[pooling] = s
    if "stats" in out:
        view_gap_verdict(out["stats"])
    return out


# --------------------------------------------------------------------------- #
# Test 2 — strata
# --------------------------------------------------------------------------- #


def build_strata(coverage, thresh):
    """Position subsets, mirroring the training rule and ``patch_background_diagnostic``.

    ``fg_train_rule`` is the exact predicate from ``main_multimodal.py:335`` —
    ``(frac >= thresh).any()`` — evaluated over the whole eval set rather than per batch,
    which makes it slightly more permissive than any single training step.
    """
    mean_cov = coverage.mean(0)
    keep_any = (coverage >= thresh).any(0)
    return {
        "all": np.ones(coverage.shape[1], dtype=bool),
        "fg_train_rule": keep_any,
        "fg_strict": mean_cov >= thresh,
        "pure": mean_cov >= 0.95,
        "background": ~keep_any,
    }


def test_strata(label, content, gt, coverage, thresh, seeds, n_splits, n_null):
    strata = build_strata(coverage, thresh)
    rng = np.random.RandomState(0)
    n = content.shape[0]
    print(f"\n  {label}:  N={n}  positions={content.shape[1]}  content channels={content.shape[2]}")
    print(f"  {'stratum':<16}{'n_pos':>7}{'n_feat':>9}{'mcc':>9}{'+-sd':>9}{'null':>8}{'gap':>9}")
    print("  " + "-" * 67)

    out = {}
    for name, sel in strata.items():
        if not sel.any():
            print(f"  {name:<16}{0:>7}  (empty)")
            continue
        x = content[:, sel, :].reshape(n, -1)  # patch-major, as dci.py builds it
        # std is the spread ACROSS SEEDS, which the selection path computes and discards.
        # The whole patch-MCC phenomenon spans ~0.04, so a drop is only a finding if it
        # clears this band — print it next to the value rather than making it a re-run.
        _m = block_mcc(x, gt, seeds=seeds, n_splits=n_splits)
        mcc, mcc_sd = _m["mean"], _m["std"]
        # A permutation null, because mcc_by_pool/* is logged RAW and its floor moves with
        # the block's rank: a falling rank lowers the null and the raw number drops free.
        null = (
            float(
                np.mean(
                    [
                        block_mcc(x, gt[rng.permutation(n)], seeds=seeds[:1], n_splits=n_splits)["mean"]
                        for _ in range(n_null)
                    ]
                )
            )
            if n_null > 0
            else 0.0
        )
        out[name] = {"n_pos": int(sel.sum()), "mcc": mcc, "mcc_sd": mcc_sd, "null": null, "gap": mcc - null}
        print(
            f"  {name:<16}{int(sel.sum()):>7}{x.shape[1]:>9}{mcc:>9.4f}{mcc_sd:>9.4f}" f"{null:>8.4f}{mcc - null:>9.4f}"
        )

    fg, bg = strata["fg_train_rule"], strata["background"]
    if bg.any():
        e_fg = float((content[:, fg, :] ** 2).mean())
        e_bg = float((content[:, bg, :] ** 2).mean())
        print(f"\n    foreground:background energy = {e_fg / max(e_bg, 1e-12):.3f}  (fg {e_fg:.4g}, bg {e_bg:.4g})")
        out["_energy_ratio"] = e_fg / max(e_bg, 1e-12)
    return out


# --------------------------------------------------------------------------- #
# Test 3 — geometry (probe-free)
# --------------------------------------------------------------------------- #


def spectrum(X):
    """Covariance eigenvalues, descending.  Gram side: identical non-zero spectrum, and
    O(n^2 p) instead of O(p^3) at p = 22528."""
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(0, keepdims=True)
    return np.linalg.eigvalsh(Xc @ Xc.T / max(len(Xc) - 1, 1)).clip(min=0)[::-1]


def eff_rank(ev):
    """Participation ratio normalised to (0, 1] — same convention as
    ``eval.entropy_uniformity.eff_rank``."""
    s1, s2 = ev.sum(), (ev**2).sum()
    return float(s1**2 / s2 / len(ev)) if s2 > 0 else 0.0


def n_at_energy(ev, frac=0.95):
    if ev.sum() <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(ev) / ev.sum(), frac) + 1)


def test_geometry(label, content, gt, coverage, thresh, seeds, n_splits, ranks):
    """Spectrum of the content block, plus how many leading directions the factors need.

    No probe artifact can enter the first three columns — they are properties of the
    representation alone.  The rank curve is where the two meet: if the late checkpoint
    needs more components for the same score, the factors have moved down-spectrum.
    """
    strata = build_strata(coverage, thresh)
    n = content.shape[0]
    print(f"\n  {label} — geometry")
    print(f"  {'stratum':<16}{'eff_rank':>10}{'n@95%':>8}{'cond@95%':>11}")
    print("  " + "-" * 45)
    out = {}
    for name in ("all", "fg_train_rule"):
        x = content[:, strata[name], :].reshape(n, -1)
        ev = spectrum(x)
        k95 = n_at_energy(ev, 0.95)
        cond = float(ev[0] / max(ev[min(k95, len(ev)) - 1], 1e-300))
        out[name] = {"eff_rank": eff_rank(ev), "n95": k95, "cond95": cond}
        print(f"  {name:<16}{out[name]['eff_rank']:>10.4f}{k95:>8}{cond:>11.3g}")

    print(f"\n  block-MCC from the top-k principal directions (fg_train_rule):")
    print("  " + "".join(f"{f'k={k}':>10}" for k in ranks) + f"{'full':>10}")
    x = content[:, strata["fg_train_rule"], :].reshape(n, -1)
    curve = []
    for k in ranks:
        kk = int(min(k, x.shape[1], n * (n_splits - 1) // n_splits - 1))
        # PCA is fitted on ALL rows here, which leaks a little covariance into the folds.
        # It is the same leak at every checkpoint, so the early-vs-late COMPARISON is
        # sound; do not read a single absolute value off this row.
        xk = PCA(n_components=kk, svd_solver="randomized", random_state=0).fit_transform(x)
        curve.append(block_mcc(xk, gt, seeds=seeds[:1], n_splits=n_splits)["mean"])
    full = block_mcc(x, gt, seeds=seeds[:1], n_splits=n_splits)["mean"]
    print("  " + "".join(f"{v:>10.4f}" for v in curve) + f"{full:>10.4f}")
    out["rank_curve"] = dict(zip(ranks, curve))
    out["rank_full"] = full
    return out


# --------------------------------------------------------------------------- #


def report_delta(strata_res, geom_res, arms):
    if len(arms) < 2:
        return
    early, late = arms[0], arms[-1]
    print(f"\n{'=' * 78}\nVERDICT — {early} → {late}\n{'=' * 78}")

    if strata_res.get(early) and strata_res.get(late):
        print(f"\n  {'stratum':<16}{'gap ' + early:>20}{'gap ' + late:>20}{'Δ':>10}")
        print("  " + "-" * 66)
        d = {}
        for name in ("all", "fg_train_rule", "fg_strict", "pure", "background"):
            a, b = strata_res[early].get(name), strata_res[late].get(name)
            if a and b:
                d[name] = b["gap"] - a["gap"]
                print(f"  {name:<16}{a['gap']:>20.4f}{b['gap']:>20.4f}{d[name]:>+10.4f}")
        if "fg_train_rule" in d:
            print()
            if d["fg_train_rule"] < -0.005:
                print("  → The decay is present in FOREGROUND positions, which the objective does")
                print("    train on. It is not an artifact of unmasked background in the probe")
                print("    (that mechanism is ruled out by --calibrate anyway).")
            else:
                print("  → Foreground holds. Whatever moved is outside the trained positions —")
                print("    check the background row and the energy ratio.")

    if geom_res.get(early) and geom_res.get(late):
        a, b = geom_res[early]["fg_train_rule"], geom_res[late]["fg_train_rule"]
        print(f"\n  eff_rank (fg)  {a['eff_rank']:.4f} → {b['eff_rank']:.4f}  ({b['eff_rank'] - a['eff_rank']:+.4f})")
        print(f"  n@95%    (fg)  {a['n95']} → {b['n95']}")
        print(f"  cond@95% (fg)  {a['cond95']:.3g} → {b['cond95']:.3g}")
        if b["eff_rank"] < 0.8 * a["eff_rank"] or b["cond95"] > 3 * a["cond95"]:
            print("\n  → CONDITIONING: the block's geometry collapsed. This is the one mechanism")
            print("    that survived calibration, and an invertible map is enough to cost 0.26-0.48")
            print("    of block-MCC. Levers: bt_std_coeff (the only anti-collapse term in this")
            print("    config), bt_gap_weight, and the recon:contrastive ratio from test 1.")
        else:
            print("\n  → Geometry is stable, so conditioning does not explain the drop. With")
            print("    dilution and decorrelation ruled out by --calibrate, the remaining")
            print("    reading is genuine information loss — rebalance recon vs contrastive.")


def run_calibration():
    """Reproduce the sensitivity table in this module's docstring, on plain numpy."""
    print(f"\n{'=' * 78}\nCALIBRATION — what can block-MCC at patch pooling detect?\n{'=' * 78}")
    rng = np.random.RandomState(0)
    n, p, c, f = 400, 256, 44, 9
    n_fg = 60
    gt = rng.randn(n, f)
    w = rng.randn(c, f) / np.sqrt(f)
    fg = np.zeros(p, dtype=bool)
    fg[:n_fg] = True
    sig = gt @ w.T
    noise = 6.0  # puts the probe at ~0.84, the real operating point

    def score(block):
        return block_mcc(block[:, fg, :].reshape(n, -1), gt, seeds=(0,), n_splits=5)["mean"]

    def score_all(block):
        return block_mcc(block.reshape(n, -1), gt, seeds=(0,), n_splits=5)["mean"]

    def base(seed=1):
        r = np.random.RandomState(seed)
        b = np.zeros((n, p, c))
        b[:, fg, :] = sig[:, None, :] + noise * r.randn(n, n_fg, c)
        b[:, ~fg, :] = 0.3 * r.randn(n, p - n_fg, c)
        return b

    b0 = base()
    m_fg, m_all = score(b0), score_all(b0)
    print(f"\n  operating point:  mcc(foreground) {m_fg:.4f}   mcc(all positions) {m_all:.4f}")
    print(f"  background costs a constant {m_all - m_fg:+.4f} level offset (more features), not a trend.")

    print("\n  A. DILUTION — background noise magnitude")
    for bgn in (0.3, 3.0, 20.0):
        b = base()
        r = np.random.RandomState(5)
        b[:, ~fg, :] = bgn * r.randn(n, p - n_fg, c)
        print(f"     bg x{bgn / 0.3:>5.1f}   mcc(all) {score_all(b):.4f}")
    print("     → flat: block_mcc standardises every feature, so magnitude is invisible. RULED OUT.")

    print("\n  B. DECORRELATION — channel-private vs shared spatial maps")
    for mix in (0.0, 1.0):
        r = np.random.RandomState(7)
        b = np.zeros((n, p, c))
        priv = np.einsum("nf,pcf->npc", gt, r.randn(n_fg, c, f) / np.sqrt(f))
        b[:, fg, :] = (1 - mix) * sig[:, None, :] + mix * priv + noise * r.randn(n, n_fg, c)
        b[:, ~fg, :] = 0.3 * r.randn(n, p - n_fg, c)
        print(f"     mix={mix:.1f} ({'fully shared' if mix == 0 else 'fully private'})   mcc(fg) {score(b):.4f}")
    print("     → private scores HIGHER. A linear probe does not need cross-channel redundancy. RULED OUT.")

    print("\n  C. CONDITIONING — exactly invertible anisotropic map")
    q1, _ = np.linalg.qr(rng.randn(c, c))
    q2, _ = np.linalg.qr(rng.randn(c, c))
    for logc in (2, 4, 6):
        m_ = q1 @ np.diag(np.logspace(0, -logc, c)) @ q2
        print(f"     cond 1e{logc}   mcc(fg) {score(b0 @ m_):.4f}   ({score(b0 @ m_) - m_fg:+.4f})")
    print("     → information preserved by construction, yet up to -0.48. SURVIVES.")
    print("       (A PCA-whitened probe does not recover it at k=50..319, so there is no")
    print("        probe-side control separating 're-gauged' from 'destroyed' at this N.)")

    print("\n  D. REAL LOSS — signal amplitude against a fixed noise floor")
    for s in (1.0, 0.5, 0.0):
        r = np.random.RandomState(11)
        b = np.zeros((n, p, c))
        b[:, fg, :] = s * sig[:, None, :] + noise * r.randn(n, n_fg, c)
        b[:, ~fg, :] = 0.3 * r.randn(n, p - n_fg, c)
        print(f"     signal x{s:.1f}   mcc(fg) {score(b):.4f}")
    print("\n  → C and D are the only live mechanisms, and they are not separable by a probe.")
    print("    Measure the block's geometry directly instead (test 3).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true", help="Run the sensitivity calibration and exit (no run needed)")
    ap.add_argument("--run-dir", help="Training run directory (with settings.json)")
    ap.add_argument("--compare-run-dir", default=None, help="Second run scored identically (e.g. the recon-only arm)")
    ap.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Checkpoint filenames inside each run dir, EARLY FIRST. "
        "Default: vqvae_best.pt then vqvae_model.pt when both exist.",
    )
    ap.add_argument(
        "--tests",
        nargs="+",
        default=["curves", "strata", "geometry", "viewgap"],
        choices=["curves", "strata", "geometry", "viewgap"],
    )
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--grid", type=int, nargs=3, default=None, help="Patch grid (default: the run's --patch-grid)")
    ap.add_argument("--num-samples", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-null", type=int, default=2, help="Permutations for the null floor (0 to skip)")
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 16, 64, 256])
    ap.add_argument("--fg-thresh", type=float, default=None, help="Default: the run's --patch-foreground-thresh")
    ap.add_argument("--sat-tol", type=float, default=0.1, help="on_diag is 'saturated' with this fraction left")
    ap.add_argument(
        "--causal",
        default="match",
        choices=["match", "iid"],
        help="Eval-set factor distribution. 'match' (default) forwards the run's SCM; 'iid' forces "
        "independent factors. NOT interchangeable — under a random graph ventricle_size and "
        "brain_size correlate at ~0.8.",
    )
    ap.add_argument("--cache-dataset", action="store_true", help="Cache rendered volumes in RAM (~4 GB at N=1000)")
    args = ap.parse_args()

    if args.calibrate:
        run_calibration()
        return
    if not args.run_dir:
        ap.error("--run-dir is required unless --calibrate is given")

    run_dirs = [args.run_dir] + ([args.compare_run_dir] if args.compare_run_dir else [])

    if "curves" in args.tests:
        for rd in run_dirs:
            test_curves(rd, level=args.level, sat_tol=args.sat_tol)

    want = {"strata", "geometry", "viewgap"} & set(args.tests)
    if not want:
        return

    import torch

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for rd in run_dirs:
        names = args.checkpoints or [
            n for n in ("vqvae_best.pt", "vqvae_model.pt") if os.path.exists(os.path.join(rd, n))
        ]
        if not names:
            logger.error("No checkpoints found in %s — pass --checkpoints.", rd)
            continue

        print(f"\n{'=' * 78}\nTESTS 2/3 — {os.path.basename(rd.rstrip('/'))}\n{'=' * 78}")
        _, run_args, _ = load_model_from_run_dir(rd, checkpoint=os.path.join(rd, names[0]), device=device)
        grid = args.grid or list(getattr(run_args, "patch_grid", [8, 8, 8]))
        thresh = args.fg_thresh if args.fg_thresh is not None else getattr(run_args, "patch_foreground_thresh", 0.05)
        dataset = build_synthetic_test_set(
            run_args,
            num_samples=args.num_samples,
            cache=args.cache_dataset,
            causal=(True if args.causal == "match" else False),
        )
        logger.info("grid=%s  foreground thresh=%.3f  causal=%s", grid, thresh, args.causal)

        strata_res, geom_res, arms = {}, {}, []
        for name in names:
            model, _, _ = load_model_from_run_dir(rd, checkpoint=os.path.join(rd, name), device=device)
            model.to(device)
            label = name.replace(".pt", "")
            arms.append(label)

            if "viewgap" in want:
                test_viewgap(label, model, dataset, device, args.level, args.batch_size, tuple(args.seeds))

            content = None
            if {"strata", "geometry"} & want:
                content, gt, coverage = extract_patch_block(
                    model, dataset, device, grid, args.level, args.batch_size, args.num_workers
                )
                if "strata" in want:
                    strata_res[label] = test_strata(
                        label, content, gt, coverage, thresh, tuple(args.seeds), args.n_splits, args.n_null
                    )
                if "geometry" in want:
                    geom_res[label] = test_geometry(
                        label, content, gt, coverage, thresh, tuple(args.seeds), args.n_splits, args.ranks
                    )
            del model, content
            if device.type == "cuda":
                torch.cuda.empty_cache()

        report_delta(strata_res, geom_res, arms)


if __name__ == "__main__":
    main()
