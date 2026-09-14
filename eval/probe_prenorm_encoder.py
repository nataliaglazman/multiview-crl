#!/usr/bin/env python
"""Where INSIDE the encoder does a content factor die, and does it die in one view only?

`probe_prenorm_groupnorm.py` taps `content_norms`, the SplitGroupNorm between the encoder
and the codebook.  That norm is not on the probe path: `VQVAE.forward` pools
`enc_in_v{0,1}_pool`, captured BEFORE it (`models/vqvae.py:1331-1351`), so every probe in
the repo reads features that already left the encoder untouched by it.  This script taps
inside the encoder instead, separately per view, and reports the stage at which a factor's
recoverability drops:

    pre_norm      input to the encoder's final norm       (Encoder.layers[-2])
    post_norm     its output
    encoder_out   after the residual stack                (what every probe reads)
    codebook_in   after content_norms[level]              (what the codebook sees)

Both views are scored apart because a pooled both-views probe cannot see an asymmetric
loss, and a shared content code is capped by whichever view carries the factor least
well.  The motivating asymmetry is in the LUT: the ventricle/WM edge is the STRONGEST
internal edge in T1 (0.70, against WM/GM 0.30) and the WEAKEST in FLAIR (0.30, against
0.40), so the salience rank inverts between views.  That predicts T1 ahead of FLAIR.
MEASURED, it is not: on the first run scored here (12 content channels, step 40001)
FLAIR led T1 by 0.142 at the stage the factor is best carried.  The script reports the
direction rather than assuming it -- do not read the LUT argument as settled.

Reading it
----------
The verdict RANKS the per-stage drops rather than naming the first one over the floor,
because "which stage" is only meaningful if one clearly dominates.  When the top two are
within NOISE_FLOOR of each other the loss is gradual across the encoder's tail, and
changing one stage alone will not recover the factor.

    one stage dominates             that stage is the lever (norm -> --norm-type /
                                    --split-encoder-norm; residual stack -> its norms
                                    or its width).
    top two comparable              a gradual loss, not a culprit; treat it as capacity
                                    or as what the objective spends channels on.
    views asymmetric                the trailing view's encoder is the constraint; the
                                    objective is the lever (--contrastive-proj-dim).
    views symmetric                 no view-consistency story applies at all.
    all stages flat, low everywhere lost upstream -- check the generator's SNR in the
                                    WEAK view (`generator_defects` scores `render(...)[0]`,
                                    i.e. T1 only, so it has never measured FLAIR).

The CONTENT vs STYLE table separates the two ways a factor can leave content.  Content
falling while style rises is MIGRATION -- the factor is still in the latent, just on the
pathway with no invariance constraint on it, and the levers are the style-capacity flags
(--style-spatial-size, --detach-style-injection, --style-dropout-prob,
--scale-style-hsic-loss).  Both blocks falling is DESTRUCTION, where capping style changes
nothing and the question is content capacity.  The content ladder alone cannot tell these
apart, which is why the style block is read off the same tensor (under --mask-mode fixed
the first n_content channels are content and the rest are style).

The RMS table is the separate test for the codebook: the content codebook quantizes by
squared euclidean distance (`models/vqvae.py:491`) and is shared across views, so a
per-view scale gap would send the two views to different entries.  Ratios at ~1.00 rule
that out.

Nothing is retrained.  Only the tap point changes.

Factors are drawn i.i.d. (`causal=False`) by default, not from the run's SCM.  Under a
random graph ventricle_size and brain_size correlate ~0.8, so an SCM-matched probe reads
brain_size in disguise -- per-factor attribution requires i.i.d.

`--causal-eval` switches the eval distribution to the run's training SCM, for ONE purpose:
the i.i.d.-vs-matched A/B.  A factor that reads ~0 i.i.d. but high under the SCM was never
encoded as a separable direction -- the encoder holds it only through its training-time
correlates, and the gap between the two runs is the size of that shortcut.  This is the
signature to expect when Barlow Twins runs at a high redundancy weight (`bt_lambda`) on
SCM-coupled factors: the off-diagonal penalty rewards a DECORRELATED basis, which on
coupled data is not the factor basis, so the weaker member of a correlated pair is
actively suppressed rather than merely unlearned -- which is how a trained encoder ends
up BELOW its untrained floor.  Matched numbers are inflated and never reportable alone.

Absolute R^2 here is mostly a statement about the architecture: an untrained encoder reads
ventricle_size at 0.513 from a random projection.  `--floor` (default on) scores an
untrained twin of the same architecture through the identical path and reports the gap.

Usage:
  python -m eval.probe_prenorm_encoder --run-dir results/synthetic/<run>
  python -m eval.probe_prenorm_encoder --run-dir ... --factor ventricle_size --no-floor
  python -m eval.probe_prenorm_encoder --self-test          # torch-free
"""

from __future__ import annotations

import argparse
import csv
import logging

import numpy as np

from eval.identifiability_metrics import cv_probe_r2_multi
from eval.identifiability_report import NOISE_FLOOR
from eval.run_dci_compare import PROBE_DIM_AUTO, _auto_probe_dim

logger = logging.getLogger(__name__)

STAGES = ("pre_norm", "post_norm", "encoder_out", "codebook_in")
VIEWS = ("v0", "v1")
POOLINGS = ("gap", "patch")
BLOCKS = ("content", "style")
VIEW_LABEL = {"v0": "T1", "v1": "FLAIR"}


# --------------------------------------------------------------------------- #
# Scoring and rendering.  Pure numpy so --self-test needs no torch build.
# --------------------------------------------------------------------------- #
def reduce_block(X, probe_dim="auto", seed=0):
    """PCA-reduce a block. ``probe_dim`` is "auto" (the imported rule), 0 (off), or a width.

    "auto" is `run_dci_compare._auto_probe_dim`, imported rather than re-derived. Mind its
    calibration: it was set for the 8^3 x 44-channel patch block, 22528 features against
    N~2000, a genuine p>>n case. Its threshold is d > N/4, so at N=2000 it also fires on a
    768-feature block (4^3 x 12 channels) that ridge handles perfectly well, and cuts it to
    64 top-VARIANCE components. PCA keeps variance, not signal, so a small localised
    structure -- a ventricle -- can be deleted outright by that step while a global factor
    survives. If the patch column reads negative everywhere while gap does not, suspect
    this before concluding anything about the model: re-run with ``--probe-dim 0``.
    """
    from sklearn.decomposition import PCA

    if probe_dim == PROBE_DIM_AUTO:
        width = _auto_probe_dim(X.shape[0], X.shape[1])
    else:
        width = int(probe_dim)
    if not width or width >= X.shape[1]:
        return X
    return PCA(n_components=width, random_state=seed).fit_transform(X)


def score_ladder(feats, gt, seeds=(0, 1)):
    """{(stage, view, pooling): (N, D)} -> {(stage, view, pooling): (n_factors,) R^2}."""
    return {key: np.asarray(cv_probe_r2_multi(X, gt, seeds=seeds)["mean"]) for key, X in feats.items()}


def _fmt(v):
    return "     --" if v is None or not np.isfinite(v) else f"{v:>7.3f}"


def _present_stages(scores):
    return [s for s in STAGES if any((s, v, p) in scores for v in VIEWS for p in POOLINGS)]


def _present_views(scores):
    return [v for v in VIEWS if any((s, v, p) in scores for s in STAGES for p in POOLINGS)]


def render_focus(scores, names, factor, floor=None):
    """The ladder for one factor: every stage x view x pooling, floor-subtracted."""
    j = names.index(factor)
    stages, views = _present_stages(scores), _present_views(scores)

    def cell(stage, view, pooling):
        key = (stage, view, pooling)
        if key not in scores:
            return None
        val = float(scores[key][j])
        if floor is not None and key in floor:
            val -= float(floor[key][j])
        return val

    tag = "learned - untrained floor" if floor is not None else "absolute (NO floor subtracted)"
    print(f"\n{factor} R^2 by encoder stage and view  [{tag}]")
    head = f"  {'stage':<14}" + "".join(f"{VIEW_LABEL[v] + ' ' + p:>14}" for v in views for p in POOLINGS)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for stage in stages:
        row = "".join(f"{_fmt(cell(stage, v, p)):>14}" for v in views for p in POOLINGS)
        print(f"  {stage:<14}{row}")
    return {(s, v, p): cell(s, v, p) for s in stages for v in views for p in POOLINGS}


def render_all_factors(scores, names, stage="encoder_out", pooling="gap", floor=None):
    """Every factor at one stage, ABSOLUTE and floor side by side.

    The delta alone cannot support a cross-view claim.  The untrained floor is a random
    projection of THAT VIEW's images, so it differs per view by construction: a random
    channel mean of a T1 volume tracks brain_size well (T1's global mean is dominated by
    it) while the same projection of a FLAIR volume does not.  A view can therefore look
    "weaker" purely because its floor is higher.  Both columns are printed so that reading
    is available rather than hidden.
    """
    views = _present_views(scores)
    if not any((stage, v, pooling) in scores for v in views):
        return
    has_floor = floor is not None
    print(f"\nAll factors at {stage}, {pooling} pooling  (abs = learned, flr = untrained floor, d = gap over it)")
    cols = ("abs", "flr", "d") if has_floor else ("abs",)
    head = f"  {'factor':<18}" + "".join(f"{VIEW_LABEL[v] + ' ' + c:>10}" for v in views for c in cols)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for j, name in enumerate(names):
        row = ""
        for v in views:
            key = (stage, v, pooling)
            if key not in scores:
                row += "".join(f"{'--':>10}" for _ in cols)
                continue
            a = float(scores[key][j])
            f = float(floor[key][j]) if has_floor and key in floor else float("nan")
            vals = (a, f, a - f) if has_floor else (a,)
            row += "".join(f"{x:>10.3f}" for x in vals)
        print(f"  {name:<18}{row}")

    if has_floor and len(views) == 2:
        k0 = (stage, views[0], pooling)
        k1 = (stage, views[1], pooling)
        if k0 in floor and k1 in floor:
            f0, f1 = np.nanmean(floor[k0]), np.nanmean(floor[k1])
            a0, a1 = np.nanmean(scores[k0]), np.nanmean(scores[k1])
            print(
                f"\n  mean over factors: {VIEW_LABEL[views[0]]} abs {a0:+.3f} flr {f0:+.3f}   "
                f"{VIEW_LABEL[views[1]]} abs {a1:+.3f} flr {f1:+.3f}"
            )
            if abs(f0 - f1) > NOISE_FLOOR:
                hi = VIEW_LABEL[views[0] if f0 > f1 else views[1]]
                print(f"  The FLOORS differ by {abs(f0 - f1):.3f} ({hi} higher). A per-view difference in the")
                print("  DELTA column of that size says nothing about the learned code -- compare abs.")


def render_migration(c_scores, s_scores, names, factor, floor_c=None, floor_s=None, pooling="gap"):
    """Content vs style for one factor at every stage -- did it MOVE or was it destroyed?

    Content falling while style rises is migration: the factor is still in the latent, just
    on the unconstrained pathway.  Both falling is destruction.  The two call for completely
    different fixes, and the content ladder alone cannot tell them apart.
    """
    j = names.index(factor)
    stages = [s for s in STAGES if any((s, v, pooling) in c_scores for v in VIEWS)]
    views = _present_views(c_scores)
    if not stages or not s_scores:
        return None

    def cell(scores, floor, stage, view):
        key = (stage, view, pooling)
        if key not in scores:
            return None
        val = float(scores[key][j])
        if floor is not None and key in floor:
            val -= float(floor[key][j])
        return val

    tag = "learned - untrained floor" if floor_c is not None else "absolute (NO floor subtracted)"
    print(f"\n{factor}: CONTENT vs STYLE block, {pooling} pooling  [{tag}]")
    head = f"  {'stage':<14}" + "".join(f"{VIEW_LABEL[v] + ' ' + b:>16}" for v in views for b in BLOCKS)
    print(head)
    print("  " + "-" * (len(head) - 2))
    out = {}
    for stage in stages:
        cells = []
        for v in views:
            c = cell(c_scores, floor_c, stage, v)
            st = cell(s_scores, floor_s, stage, v)
            out[(stage, v, "content")], out[(stage, v, "style")] = c, st
            cells += [c, st]
        print(f"  {stage:<14}" + "".join(f"{_fmt(x):>16}" for x in cells))
    return out


def style_migration_note(migration, start="pre_norm", terminal="encoder_out", patch_present=()):
    """Per view: did the factor move to style, or was it destroyed?

    Scored PER VIEW, because one view can migrate while the other destroys, and a max
    across views reports neither.  ``terminal`` is encoder_out rather than codebook_in:
    codebook_in is a branch off the main path (content_norms feeds the codebook only),
    while encoder_out is what the probes and the contrastive loss actually read.

    Migration does NOT require style to RISE.  Style can already hold the factor from the
    first stage and simply keep it while content sheds it -- what marks migration is style
    ending up above the floor and above content, not a positive slope.
    """
    if not migration:
        return
    if not any((terminal, v, "content") in migration for v in VIEWS):
        return

    print(f"\n  content vs style per view ({start} -> {terminal}):")
    verdicts = {}
    for v in VIEWS:
        c0, c1 = migration.get((start, v, "content")), migration.get((terminal, v, "content"))
        s0, s1 = migration.get((start, v, "style")), migration.get((terminal, v, "style"))
        if c0 is None or c1 is None or not (np.isfinite(c0) and np.isfinite(c1)):
            continue
        line = f"    {VIEW_LABEL[v]:<6} content {c0:+.3f} -> {c1:+.3f}"
        has_style = s0 is not None and s1 is not None and np.isfinite(s0) and np.isfinite(s1)
        if has_style:
            line += f"    style {s0:+.3f} -> {s1:+.3f}"

        lost = c0 - c1
        if lost <= NOISE_FLOOR:
            verdicts[v] = "keeps"
            print(line + "   -> content keeps it")
            continue
        if has_style and s1 > NOISE_FLOOR and s1 > c1:
            verdicts[v] = "migration"
            print(line + "   -> MIGRATION (style ends up holding it, content does not)")
        elif v in patch_present:
            verdicts[v] = "reformat"
            print(line + "   -> NOT destroyed: the patch column still holds it (spatial format)")
        elif not has_style or s1 <= NOISE_FLOOR:
            verdicts[v] = "destruction"
            print(line + "   -> DESTRUCTION (neither block holds it, at gap OR patch)")
        else:
            verdicts[v] = "mixed"
            print(line + "   -> mixed")

        # Relocation accounting: style gaining less than content lost is destroyed, not moved.
        if has_style:
            gained = max(s1 - s0, 0.0)
            unaccounted = lost - gained
            if unaccounted > NOISE_FLOOR:
                print(
                    f"           content lost {lost:.3f}, style gained {s1 - s0:+.3f}"
                    f"  -> {unaccounted:.3f} is destroyed, not relocated"
                )

    vals = set(verdicts.values())
    if "migration" in vals:
        which = ", ".join(VIEW_LABEL[v] for v, k in verdicts.items() if k == "migration")
        print(f"\n  Style holds it in {which}. That part is relocation, and the levers are the")
        print("  style-capacity flags -- check settings.json for style_spatial_size first:")
        print("    --style-spatial-size 1|2   cap style's spatial grid so it cannot carry a cavity")
        print("    --detach-style-injection   stop recon backprop teaching style to hold anatomy")
        print("    --style-dropout-prob 0.25  force recon from content alone on some samples")
        print("    --scale-style-hsic-loss    explicit style-content independence (supervised)")
    if "destruction" in vals:
        which = ", ".join(VIEW_LABEL[v] for v, k in verdicts.items() if k == "destruction")
        print(f"\n  In {which} neither block holds it, so capping style recovers nothing there.")
        print("  That part is content capacity and what the objective spends channels on.")
    if vals == {"migration", "destruction"} or ("migration" in vals and "destruction" in vals):
        print("\n  The two views differ, so ONE fix will not cover both. Treat the relocation and")
        print("  the destruction as separate problems rather than looking for a single cause.")


def render_channel_map(feats, gt, names, factor, stage="encoder_out", pooling="gap", floor_feats=None, seeds=(0, 1)):
    """Which factor does each individual channel's pooled value track?

    The gap block's column j IS channel j's global mean, so this needs no new features --
    it just scores each column on its own instead of letting ridge combine all twelve.
    That distinguishes "the encoder has a ventricle channel" from "the ventricle is
    reconstructable from a combination", which the block-level R^2 cannot separate.

    Reported ABSOLUTE, not floor-subtracted: channel j of an untrained twin is a different
    random projection from channel j of the trained model, so a per-channel difference
    between them means nothing.  The untrained BEST-channel row is the reference instead --
    it says what one random projection already achieves.
    """
    views = [v for v in VIEWS if (stage, v, pooling) in feats]
    if not views:
        return {}
    short = [n[:6] for n in names]
    best = {}
    for v in views:
        X = feats[(stage, v, pooling)]
        per_ch = np.stack(
            [np.asarray(cv_probe_r2_multi(X[:, [j]], gt, seeds=seeds)["mean"]) for j in range(X.shape[1])]
        )
        print(f"\n{VIEW_LABEL[v]} - per-channel {pooling} R^2 at {stage}  (absolute; one channel at a time)")
        head = "  ch  " + "".join(f"{c:>8}" for c in short) + "   best"
        print(head)
        print("  " + "-" * (len(head) - 2))
        for j in range(per_ch.shape[0]):
            k = int(np.nanargmax(per_ch[j]))
            print(f"  {j:<4}" + "".join(f"{x:>8.3f}" for x in per_ch[j]) + f"   {names[k]} {per_ch[j, k]:.3f}")
        if floor_feats is not None and (stage, v, pooling) in floor_feats:
            F = floor_feats[(stage, v, pooling)]
            f_ch = np.stack(
                [np.asarray(cv_probe_r2_multi(F[:, [j]], gt, seeds=seeds)["mean"]) for j in range(F.shape[1])]
            )
            print("  " + "-" * (len(head) - 2))
            print("  untr" + "".join(f"{x:>8.3f}" for x in f_ch.max(0)) + "   best UNTRAINED channel")
        jf = names.index(factor)
        bj = int(np.nanargmax(per_ch[:, jf]))
        best[v] = (bj, float(per_ch[bj, jf]))

    print(f"\n  {factor}: best SINGLE channel per view")
    for v in views:
        bj, r = best[v]
        print(f"    {VIEW_LABEL[v]:<6} channel {bj:<3} R^2 {r:+.3f}")
    if len(views) == 2:
        (b0, r0), (b1, r1) = best[views[0]], best[views[1]]
        if max(r0, r1) <= NOISE_FLOOR:
            print("  No single channel in either view tracks it -- if the block-level R^2 is high,")
            print("  the factor is spread across channels rather than held by a detector.")
        elif abs(r0 - r1) > NOISE_FLOOR:
            hi = VIEW_LABEL[views[0] if r0 > r1 else views[1]]
            lo = VIEW_LABEL[views[1] if r0 > r1 else views[0]]
            print(f"  {hi} has a channel for it; {lo} does not. Compare each view's 'best' column")
            print("  above to see what that view spends its channels on instead.")
    return best


def render_rms(rms, block="content"):
    """Feature scale per stage per view.  A per-view gap here is what the shared,
    squared-euclidean codebook cannot absorb."""
    rms = {(s, v): x for (s, v, b), x in rms.items() if b == block}
    if not rms:
        return
    views = [v for v in VIEWS if any((s, v) in rms for s in STAGES)]
    print(f"\n{block.capitalize()}-block feature RMS  (a per-view gap is what the shared L2 codebook sees)")
    head = f"  {'stage':<14}" + "".join(f"{VIEW_LABEL[v]:>12}" for v in views) + f"{'v1/v0':>10}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for stage in STAGES:
        if not any((stage, v) in rms for v in views):
            continue
        vals = [rms.get((stage, v), float("nan")) for v in views]
        ratio = vals[1] / vals[0] if len(vals) == 2 and np.isfinite(vals[0]) and abs(vals[0]) > 1e-9 else float("nan")
        print(f"  {stage:<14}" + "".join(f"{v:>12.3f}" for v in vals) + f"{ratio:>10.2f}")


def format_note(ladder, factor, terminal="encoder_out"):
    """Is the factor in the CHANNEL MEANS or in the SPATIAL LAYOUT? They are different codes.

    gap averages every latent position, so it is structurally blind to a factor stored as
    "this location is cavity" -- a 1-3 voxel ventricle in a 16^3 latent is invisible to it.
    patch keeps position and can see that.

    Neither is a ceiling.  gap's 12 features are linear combinations of patch's 768, so
    patch ought to dominate; that it sometimes does not is a ridge-conditioning artefact of
    standardising 768 features, which dilutes a signal concentrated in the global mean.
    So each pooling is only a LOWER BOUND, and a factor counts as absent only when BOTH
    are at the floor.
    """
    print(f"\n  encoding FORMAT at {terminal} (each pooling is a lower bound, never a ceiling):")
    verdicts = {}
    for v in VIEWS:
        g = ladder.get((terminal, v, "gap"))
        pt = ladder.get((terminal, v, "patch"))
        if g is None and pt is None:
            continue
        gv = float("nan") if g is None else g
        pv = float("nan") if pt is None else pt
        hi_g, hi_p = gv > NOISE_FLOOR, pv > NOISE_FLOOR
        if hi_g and hi_p:
            kind = "both channel means AND spatial layout"
        elif hi_g:
            kind = "CHANNEL MEANS only (gap); not visible spatially"
        elif hi_p:
            kind = "SPATIAL LAYOUT only (patch); gap pooling is BLIND to it"
        else:
            kind = "neither pooling finds it"
        verdicts[v] = kind
        best = np.nanmax([gv, pv])
        print(f"    {VIEW_LABEL[v]:<6} gap {gv:+.3f}   patch {pv:+.3f}   -> {kind} (at least {best:+.3f})")

    kinds = set(verdicts.values())
    if any("SPATIAL" in k for k in kinds) and any("CHANNEL" in k for k in kinds):
        sp = ", ".join(VIEW_LABEL[v] for v, k in verdicts.items() if "SPATIAL" in k)
        ch = ", ".join(VIEW_LABEL[v] for v, k in verdicts.items() if "CHANNEL" in k)
        print(f"\n  The two views use DIFFERENT FORMATS: {ch} in channel means, {sp} spatially.")
        print("  Both encode the factor; a single pooling would have called one of them empty.")
        print("  A cross-view objective has to reconcile two codes, not recover a missing one.")
    return verdicts


def verdict(ladder, factor, has_floor=False):
    """Name the stage that loses the factor.  `ladder` is render_focus's return."""

    def g(stage, view):
        v = ladder.get((stage, view, "gap"))
        return float("nan") if v is None else v

    transitions = (
        ("pre_norm", "post_norm", "the encoder's FINAL NORM", "--norm-type / --split-encoder-norm"),
        ("post_norm", "encoder_out", "the RESIDUAL STACK", "its norms, or its width"),
        ("encoder_out", "codebook_in", "content_norms (codebook path only)", "--split-encoder-norm"),
    )
    ranked = []
    for a, b, label, lever in transitions:
        per_view = {v: g(a, v) - g(b, v) for v in VIEWS if np.isfinite(g(a, v)) and np.isfinite(g(b, v))}
        if per_view:
            # A transition where one view GAINS while the other loses is not "a stage that
            # loses the factor" -- max() alone would report the loss and hide the gain.
            disagree = max(per_view.values()) > NOISE_FLOOR and min(per_view.values()) < -NOISE_FLOOR
            ranked.append((max(per_view.values()), label, lever, per_view, disagree))
    ranked.sort(reverse=True, key=lambda r: r[0])

    print(f"\nverdict for {factor}  (gap pooling; a move under {NOISE_FLOOR} is not reportable)")
    if not ranked:
        print("  Not enough stages captured to rank anything.")
        return

    print("  where it is lost, largest first  (positive = the factor got WORSE across that stage):")
    for cost, label, _, per_view, disagree in ranked:
        detail = "  ".join(f"{VIEW_LABEL[v]} {d:+.3f}" for v, d in per_view.items())
        flag = "   <- views DISAGREE in sign" if disagree else ""
        print(f"    {cost:+.3f}  {label:<36} ({detail}){flag}")

    top, top_label, top_lever, _, top_disagree = ranked[0]
    if top_disagree:
        gainers = ", ".join(VIEW_LABEL[v] for v, d in ranked[0][3].items() if d < -NOISE_FLOOR)
        losers = ", ".join(VIEW_LABEL[v] for v, d in ranked[0][3].items() if d > NOISE_FLOOR)
        print(f"\n  {top_label} is NOT a culprit stage: it BUILDS the factor in {gainers} and")
        print(f"  loses it in {losers}. That is a per-view difference in what the encoder can")
        print("  extract, not a stage deleting information. Read the per-view columns, and")
        print("  compare against the recon-only baseline before blaming the architecture.")
        top = float("-inf")
    if top <= NOISE_FLOOR:
        print(f"\n  No single stage loses more than {NOISE_FLOOR}. The factor is either absent")
        print("  throughout or bleeds away gradually; read the ladder, not this line.")
    else:
        tied = [lbl for cost, lbl, _, _, _ in ranked[1:] if np.isfinite(cost) and (top - cost) <= NOISE_FLOOR]
        if tied:
            print(f"\n  {top_label} loses the most ({top:.3f}), but {' and '.join(tied)} is within")
            print(f"  {NOISE_FLOOR} of it -- treat them as one gradual loss across the encoder's tail,")
            print("  not as a single culprit stage. Changing one alone is unlikely to be enough.")
        else:
            print(f"\n  {top_label} loses the most ({top:.3f}). Lever: {top_lever}.")

    # View asymmetry, reported wherever the factor is best carried rather than at the end.
    best_stage = max(
        (s for s in ("pre_norm", "post_norm", "encoder_out") if np.isfinite(g(s, "v0")) or np.isfinite(g(s, "v1"))),
        key=lambda s: np.nanmax([g(s, "v0"), g(s, "v1")]),
        default=None,
    )
    if best_stage is not None:
        b0, b1 = g(best_stage, "v0"), g(best_stage, "v1")
        if np.isfinite(b0) and np.isfinite(b1):
            hi, lo = (VIEW_LABEL["v0"], VIEW_LABEL["v1"]) if b0 >= b1 else (VIEW_LABEL["v1"], VIEW_LABEL["v0"])
            gap = abs(b0 - b1)
            where = f"best carried at {best_stage} ({VIEW_LABEL['v0']} {b0:.3f} / {VIEW_LABEL['v1']} {b1:.3f})"
            if gap > NOISE_FLOOR:
                print(f"\n  Views are ASYMMETRIC: {where} -- {hi} ahead of {lo} by {gap:.3f}.")
            else:
                print(f"\n  Views are SYMMETRIC: {where}, gap {gap:.3f}.")
                print("  Both encoders treat it alike, so a view-consistency story does not apply;")
                print("  look at capacity and at what the objective spends channels on instead.")

    def gp(stage, view):
        v = ladder.get((stage, view, "patch"))
        return float("nan") if v is None else v

    out = [g("encoder_out", v) for v in VIEWS if np.isfinite(g("encoder_out", v))]
    patch_out = [gp("encoder_out", v) for v in VIEWS if np.isfinite(gp("encoder_out", v))]
    patch_has_it = bool(patch_out) and max(patch_out) > NOISE_FLOOR
    if has_floor and out and max(out) < 0 and not patch_has_it:
        print("\n  NOTE: at encoder_out the trained model is BELOW its untrained floor in every view.")
        print("  Training did not fail to learn this factor -- it removed information a random")
        print("  projection of the same architecture still had.")


# --------------------------------------------------------------------------- #
# Feature collection (torch).
# --------------------------------------------------------------------------- #
def _collect(model, inner, loader, device, level, patch_grid, n_content):
    import torch
    import torch.nn.functional as F

    from models.vqvae import ResidualStack

    if level >= len(inner.encoders):
        raise SystemExit(f"--level {level} but this model has {len(inner.encoders)} encoder level(s).")

    sep = bool(getattr(inner, "separate_encoders", False)) and getattr(inner, "encoders_v1", None) is not None
    encs = {"v0": inner.encoders[level]}
    if sep:
        encs["v1"] = inner.encoders_v1[level]

    raw = {}
    handles = []

    def _tap(enc, tag):
        # Encoder.build appends [.., Conv3d, final norm, ResidualStack], so layers[-2] is
        # the norm the content/style mask's tensor is normalised by. Assert it rather than
        # trust the index: a silent mis-tap here reads as a confident result.
        if not isinstance(enc.layers[-1], ResidualStack):
            raise SystemExit(
                f"Encoder layer order changed (layers[-1] is {type(enc.layers[-1]).__name__}, "
                "expected ResidualStack) — the pre/post-norm tap points are no longer layers[-2]."
            )
        norm = enc.layers[-2]
        raw[("pre_norm", tag)], raw[("post_norm", tag)], raw[("encoder_out", tag)] = [], [], []
        handles.append(norm.register_forward_pre_hook(lambda m, i, t=tag: raw[("pre_norm", t)].append(i[0].detach())))
        handles.append(norm.register_forward_hook(lambda m, i, o, t=tag: raw[("post_norm", t)].append(o.detach())))
        handles.append(enc.register_forward_hook(lambda m, i, o, t=tag: raw[("encoder_out", t)].append(o.detach())))

    for tag, enc in encs.items():
        _tap(enc, tag)

    cn_key = str(level)
    has_cn = hasattr(inner, "content_norms") and cn_key in inner.content_norms
    if has_cn:
        raw[("codebook_in", "cn")] = []
        handles.append(
            inner.content_norms[cn_key].register_forward_hook(
                lambda m, i, o: raw[("codebook_in", "cn")].append(o.detach())
            )
        )

    use_latent_mask = bool(getattr(inner, "latent_mask", False))
    feats = {(s, v, p): [] for s in STAGES for v in VIEWS for p in POOLINGS}
    style_feats = {(s, v, p): [] for s in STAGES for v in VIEWS for p in POOLINGS}
    sq_sum = {(s, v, b): [0.0, 0] for s in STAGES for v in VIEWS for b in BLOCKS}
    gts = []

    def _as_views(tensors, batch_half):
        """One capture holding both views, or one capture per view."""
        if len(tensors) >= 2:
            return tensors[0], tensors[1]
        t = tensors[0]
        if t.shape[0] == 2 * batch_half:
            return t[:batch_half], t[batch_half:]
        return t, None

    with torch.no_grad():
        for batch in loader:
            v1, v2 = batch["image"]
            half = v1.shape[0]
            x = torch.cat([v1, v2], dim=0).to(device)
            fwd_mask = None
            if use_latent_mask and batch.get("mask") is not None:
                fwd_mask = torch.cat(batch["mask"], 0).to(device)
            for key in raw:
                raw[key].clear()

            model(x, return_recon=False, pool_only=True, n_views=2, subsets=[(0, 1)], patch_grid=None, mask=fwd_mask)

            per_stage = {}
            for stage in ("pre_norm", "post_norm", "encoder_out"):
                if sep:
                    a = raw[(stage, "v0")][0] if raw[(stage, "v0")] else None
                    b = raw[(stage, "v1")][0] if raw.get((stage, "v1")) else None
                else:
                    caps = raw[(stage, "v0")]
                    a, b = _as_views(caps, half) if caps else (None, None)
                per_stage[stage] = (a, b)
            if has_cn and raw[("codebook_in", "cn")]:
                per_stage["codebook_in"] = _as_views(raw[("codebook_in", "cn")], half)

            for stage, (a, b) in per_stage.items():
                for view, full in (("v0", a), ("v1", b)):
                    if full is None:
                        continue
                    # Under --mask-mode fixed the first n_content channels ARE content and
                    # the rest ARE style, so the two blocks can be read off the same tensor.
                    blocks = {"content": full[:, :n_content].float()}
                    if full.shape[1] > n_content:
                        blocks["style"] = full[:, n_content:].float()
                    for block, t in blocks.items():
                        sq_sum[(stage, view, block)][0] += float(t.pow(2).sum().item())
                        sq_sum[(stage, view, block)][1] += int(t.numel())
                        sink = feats if block == "content" else style_feats
                        sink[(stage, view, "gap")].append(t.mean(dim=[2, 3, 4]).cpu().numpy())
                        sink[(stage, view, "patch")].append(
                            F.adaptive_avg_pool3d(t, patch_grid).flatten(1).cpu().numpy()
                        )
            gts.append(batch["gt_latents"]["z_content"].numpy())

    for h in handles:
        h.remove()

    feats = {k: np.concatenate(v, 0) for k, v in feats.items() if v}
    style_feats = {k: np.concatenate(v, 0) for k, v in style_feats.items() if v}
    rms = {k: float(np.sqrt(s / n)) for k, (s, n) in sq_sum.items() if n}
    return feats, style_feats, rms, np.concatenate(gts, 0)


def _load(run_dir, checkpoint, random_init, seed):
    from eval.run_dci_synthetic import load_model_from_run_dir

    model, run_args, device = load_model_from_run_dir(run_dir, checkpoint, None, random_init=random_init, seed=seed)
    inner = model.module if hasattr(model, "module") else model
    model.eval()
    return model, inner, run_args, device


def _content_width(run_args, all_channels):
    if getattr(run_args, "mask_mode", "onthefly") != "fixed":
        logger.warning(
            "mask_mode=%r: content is selected by logit order and is NOT the first k channels, "
            "so the content slice is approximate. Only 'fixed' guarantees it.",
            getattr(run_args, "mask_mode", "onthefly"),
        )
    hidden = int(getattr(run_args, "vqvae_hidden_channels", 0)) or None
    n_content = int(getattr(run_args, "content_size", 0) or 0)
    if all_channels or n_content <= 0 or (hidden and n_content > hidden):
        n_content = hidden or n_content
    return n_content


def _ladder(model, inner, loader, device, args, n_content):
    feats, style, rms, gt = _collect(model, inner, loader, device, args.level, args.patch_grid, n_content)
    pd = args.probe_dim
    for key, X in feats.items():
        if pd == PROBE_DIM_AUTO and _auto_probe_dim(X.shape[0], X.shape[1]) and X.shape[1] < X.shape[0] / 2:
            logger.warning(
                "--probe-dim auto is reducing %s from %d features to %d, but %d features against "
                "N=%d is not p>>n. PCA keeps variance, not signal, so a localised factor can be "
                "deleted by this step. Re-run with --probe-dim 0 before trusting that column.",
                key,
                X.shape[1],
                _auto_probe_dim(X.shape[0], X.shape[1]),
                X.shape[1],
                X.shape[0],
            )
            break
    return (
        {k: reduce_block(X, pd) for k, X in feats.items()},
        {k: reduce_block(X, pd) for k, X in style.items()},
        rms,
        gt,
    )


# --------------------------------------------------------------------------- #
def _self_test():
    """Plant a known drop at a known stage and check the ladder finds it. No torch."""
    rng = np.random.default_rng(0)
    n, d = 300, 24
    names = [f"f{i}" for i in range(4)]
    gt = rng.standard_normal((n, len(names)))
    focus = 1

    def block(carries, scale=1.0):
        X = rng.standard_normal((n, d)) * 0.5
        if carries:
            X[:, :3] += scale * gt[:, focus : focus + 1]
        X[:, 3:6] += gt[:, 0:1]  # a factor every stage keeps, as the positive control
        return X

    feats = {}
    for pooling in POOLINGS:
        feats[("pre_norm", "v0", pooling)] = block(True)
        feats[("pre_norm", "v1", pooling)] = block(True)
        feats[("post_norm", "v0", pooling)] = block(False)  # the planted drop
        feats[("post_norm", "v1", pooling)] = block(False)
        feats[("encoder_out", "v0", pooling)] = block(False)
        feats[("encoder_out", "v1", pooling)] = block(False)

    style_feats = {}
    for pooling in POOLINGS:
        style_feats[("pre_norm", "v0", pooling)] = block(False)
        style_feats[("pre_norm", "v1", pooling)] = block(False)
        style_feats[("post_norm", "v0", pooling)] = block(True)  # the planted migration
        style_feats[("post_norm", "v1", pooling)] = block(True)
        style_feats[("encoder_out", "v0", pooling)] = block(True)
        style_feats[("encoder_out", "v1", pooling)] = block(True)

    scores = score_ladder(feats, gt)
    style_scores = score_ladder(style_feats, gt)
    ladder = render_focus(scores, names, names[focus])
    migration = render_migration(scores, style_scores, names, names[focus])
    migration_patch = render_migration(scores, style_scores, names, names[focus], pooling="patch")
    render_all_factors(scores, names, floor=None)
    chan_pre = render_channel_map(feats, gt, names, names[focus], stage="pre_norm")
    chan_out = render_channel_map(feats, gt, names, names[focus], stage="encoder_out")
    render_rms({("pre_norm", "v0", "content"): 1.0, ("pre_norm", "v1", "content"): 0.43})
    verdict(ladder, names[focus], has_floor=False)
    fmt = format_note(ladder, names[focus])
    patch_present = {v for v, k in fmt.items() if "SPATIAL" in k or "both" in k}
    style_migration_note(migration, patch_present=patch_present)
    if migration_patch:
        print("\n  --- same split, PATCH pooling (where a spatially-held factor lives) ---")
        style_migration_note(migration_patch)

    # The planted factor sits on channels 0-2 at pre_norm only, so a per-channel scan must
    # find a detector there and none at encoder_out.
    assert chan_pre["v0"][0] < 3, f"channel map missed the planted detector: {chan_pre['v0']}"
    assert chan_pre["v0"][1] > 0.3, f"planted detector too weak: {chan_pre['v0']}"
    assert chan_out["v0"][1] < NOISE_FLOOR, f"detector should be gone at encoder_out: {chan_out['v0']}"

    pre = ladder[("pre_norm", "v0", "gap")]
    post = ladder[("post_norm", "v0", "gap")]
    assert pre - post > NOISE_FLOOR, f"planted drop not detected: {pre:.3f} -> {post:.3f}"
    assert ladder[("pre_norm", "v0", "gap")] > 0.3, "positive control did not recover the planted factor"
    print("\nself-test OK: the planted pre->post drop was detected and named.")


def main():
    ap = argparse.ArgumentParser(description="Per-view, per-stage factor recovery inside the encoder (no retraining).")
    ap.add_argument("--run-dir", help="Training run dir with settings.json.")
    ap.add_argument("--checkpoint", default=None, help="Checkpoint path (default: the run's best).")
    ap.add_argument("--level", type=int, default=0, help="Encoder level to tap.")
    ap.add_argument("--factor", default="ventricle_size", help="Factor the focus table and verdict are about.")
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patch-grid", type=int, default=4, help="Local pooling grid GxGxG.")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument(
        "--floor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score an untrained twin through the identical path and report the gap.",
    )
    ap.add_argument("--floor-seed", type=int, default=0, help="Seed for the untrained twin's weights.")
    ap.add_argument("--all-channels", action="store_true", help="Read the full hidden width, not just content.")
    ap.add_argument(
        "--channel-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score each content channel's pooled value on its own, to see which factor each "
        "channel tracks and whether a view has a detector for the focus factor at all.",
    )
    ap.add_argument(
        "--probe-dim",
        default=PROBE_DIM_AUTO,
        help="PCA width for each block: 'auto' (run_dci_compare's p>>n rule, the default), "
        "0 to disable reduction entirely, or an integer width. Use 0 when the patch block is "
        "already well-conditioned -- 'auto' fires at d > N/4 and will crush a 768-feature "
        "block to 64 top-variance components, which can delete a small localised factor.",
    )
    ap.add_argument(
        "--causal-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw eval factors from the run's TRAINING SCM instead of i.i.d. Default off. "
        "Turn it on only to run the i.i.d.-vs-matched A/B: a factor that reads ~0 i.i.d. but "
        "high under the SCM was never encoded separably -- the encoder holds it only through "
        "its training-time correlates. Matched per-factor numbers are inflated and are NOT "
        "reportable on their own.",
    )
    ap.add_argument("--csv", default=None, help="Write the per-stage table here.")
    ap.add_argument("--self-test", action="store_true", help="Run the torch-free self-test and exit.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.self_test:
        _self_test()
        return
    if not args.run_dir:
        ap.error("--run-dir is required (or pass --self-test)")

    seeds = tuple(int(s) for s in args.seeds.split(","))
    from torch.utils.data import DataLoader

    from eval.dci import CONTENT_FACTOR_NAMES
    from eval.run_dci_synthetic import build_synthetic_test_set

    model, inner, run_args, device = _load(args.run_dir, args.checkpoint, random_init=False, seed=None)
    n_content = _content_width(run_args, args.all_channels)

    # Default causal=False, deliberately: per-factor attribution needs i.i.d. factors, and
    # under the run's SCM ventricle_size correlates with brain_size at ~0.8. --causal-eval
    # switches it on for the A/B that tells "never encoded" from "encoded only via correlates".
    if args.causal_eval:
        logger.warning(
            "--causal-eval: factors follow the run's TRAINING SCM. Per-factor numbers are "
            "INFLATED by correlated factors and are only meaningful next to the i.i.d. run."
        )
    ds = build_synthetic_test_set(run_args, args.num_samples, causal=bool(args.causal_eval))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    feats, style_feats, rms, gt = _ladder(model, inner, loader, device, args, n_content)
    names = CONTENT_FACTOR_NAMES[: gt.shape[1]]
    if args.factor not in names:
        ap.error(f"--factor {args.factor!r} not in {names}")
    scores = score_ladder(feats, gt, seeds=seeds)
    style_scores = score_ladder(style_feats, gt, seeds=seeds) if style_feats else {}

    floor_scores = floor_style = None
    f_feats_keep = None
    if args.floor:
        logger.info("scoring the untrained twin (seed %d) on the SAME rows ...", args.floor_seed)
        f_model, f_inner, _, f_device = _load(args.run_dir, args.checkpoint, random_init=True, seed=args.floor_seed)
        f_feats, f_style, _, f_gt = _ladder(f_model, f_inner, loader, f_device, args, n_content)
        floor_scores = score_ladder(f_feats, f_gt, seeds=seeds)
        floor_style = score_ladder(f_style, f_gt, seeds=seeds) if f_style else None
        f_feats_keep = f_feats

    print(f"\nrun: {args.run_dir} | level {args.level} | content channels {n_content} | N={gt.shape[0]}")
    print(
        "factors drawn from the run's TRAINING SCM (--causal-eval): per-factor numbers are inflated"
        if args.causal_eval
        else "factors drawn i.i.d. (causal=False) so per-factor attribution is unambiguous"
    )
    widths = {k: X.shape[1] for k, X in feats.items()}
    for pooling in POOLINGS:
        ws = {w for (st, v, p), w in widths.items() if p == pooling}
        if ws:
            print(f"  {pooling} probe width: {sorted(ws)}  (PCA-reduced when p >> n; see reduce_block)")
    ladder = render_focus(scores, names, args.factor, floor=floor_scores)
    migration = render_migration(scores, style_scores, names, args.factor, floor_scores, floor_style)
    # FLAIR carries this factor spatially, so the content/style split has to be read at
    # patch as well -- a gap-only split calls a spatially-held factor absent from both.
    migration_patch = render_migration(
        scores, style_scores, names, args.factor, floor_scores, floor_style, pooling="patch"
    )
    render_all_factors(scores, names, floor=floor_scores)
    if args.channel_map:
        render_channel_map(feats, gt, names, args.factor, floor_feats=f_feats_keep, seeds=seeds)
    render_rms(rms, block="content")
    render_rms(rms, block="style")
    verdict(ladder, args.factor, has_floor=floor_scores is not None)
    fmt = format_note(ladder, args.factor)
    patch_present = {v for v, k in fmt.items() if "SPATIAL" in k or "both" in k}
    style_migration_note(migration, patch_present=patch_present)
    if migration_patch:
        print("\n  --- same split, PATCH pooling (where a spatially-held factor lives) ---")
        style_migration_note(migration_patch)

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["stage", "view", "pooling", "factor", "r2", "r2_floor", "r2_minus_floor"])
            for (stage, view, pooling), vals in sorted(scores.items()):
                for j, name in enumerate(names):
                    fl = float(floor_scores[(stage, view, pooling)][j]) if floor_scores else float("nan")
                    w.writerow([stage, view, pooling, name, f"{vals[j]:.6f}", f"{fl:.6f}", f"{vals[j] - fl:.6f}"])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
