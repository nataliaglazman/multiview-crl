#!/usr/bin/env python
"""Does the GAP term's uniform mean actually dilute anything? Answer from a checkpoint.

``--bt-gap-pool`` rests on one unverified claim: that the across-SUBJECT variance is
concentrated in a minority of patch positions, so averaging uniformly over all P dilutes
the subject signal by roughly the fraction of positions carrying it.

If it does not, both pooling modes are fitting noise and the lever is elsewhere. This
settles that on a checkpoint you already have, with no training run. (Do not read a NO as
"go turn on ``--bt-corr-ema``" — every synthetic experiment YAML already sets it to 0.99.) The weights come from the shipped ``GapPositionPool`` itself, so the profile measured
here cannot drift from the one training would use.

A NO IS NOT ONE ANSWER — the script separates four, because they call for different things:

    delocalised     every channel's profile is flat. A feature at position p is not about
                    the anatomy at p, so no position weighting can work, shared or not.
    disagree        channels ARE localised, in DIFFERENT places. A shared weight vector is
                    the wrong instrument by construction; per-channel weights could work.
    anti-aligned    weighting by variance is significantly WORSE than its own permutation,
                    which means the high-variance positions carry per-view style rather
                    than shared content. Variance is then actively the wrong statistic.
    wrong statistic variance fails but per-position FACTOR RECOVERY succeeds. Measurable
                    here because synthetic ground truth exists — but it is supervised and
                    cannot ship as a training signal, so it is evidence that a better
                    unsupervised surrogate exists, not a green light.

THE NULL IS THE POINT
---------------------
"Variance pooling beat the uniform mean" is not evidence on its own: ANY concentrated
weighting raises the across-subject std of the pooled feature, simply by averaging fewer
things. So the verdict is a PERMUTATION TEST — the same weight values reassigned to
positions at random, many times. That null has identical concentration and carries no
information about WHERE the signal is, so the honest effect is ``variance - shuffled``,
not ``variance - uniform``.

DO NOT GATE THIS ON PROFILE ENTROPY, measured: at P=256 with a planted factor in 4
positions — the exact regime this flag exists for — the variance profile has entropy 0.999
of 1.0 and STILL lifts cross-view correlation 0.031 -> 0.170 (+0.146 over its own
permutation). Entropy is hopelessly insensitive to a mild reweighting spread over many
positions, so an entropy threshold rejects the true positives. The concentration stats
below are descriptive only; ``top10pct_mass`` is the useful one. The permutation z decides.

READ THE FOREGROUND CAVEAT
--------------------------
This extracts features over the FULL patch grid. If the run set ``--patch-foreground-mask``
(the synthetic BT configs do), training never saw the always-background positions, and
those have near-zero across-subject variance — so the all-positions profile looks peaked
for a reason training has already handled. The ``foreground`` row below restricts to
positions carrying non-trivial variance and is the one to read for such a run. If the two
rows disagree, the concentration is background structure, not focal anatomy, and the answer
is NO.

Usage
-----
    python -m eval.gap_pool_profile --run-dir results/synthetic/RUN
    python -m eval.gap_pool_profile --run-dir results/synthetic/RUN --checkpoint-name vqvae_model.pt
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def profile_stats(a):
    """Shape of a position-weight profile that sums to 1."""
    a = np.asarray(a, dtype=np.float64)
    p = a.size
    h = float(-(a * np.log(a + 1e-12)).sum())
    order = np.sort(a)[::-1]
    top10 = int(max(1, round(0.10 * p)))
    return {
        "entropy": h / np.log(p) if p > 1 else 1.0,  # 1.0 = uniform, 0.0 = one position
        "eff_pos": float(np.exp(h)),
        "max_over_uniform": float(a.max() * p),
        "top10pct_mass": float(order[:top10].sum()),  # 0.10 under a flat profile
        "n_positions": p,
    }


def pooled_metrics(hz, weights, batch_size, draws, seed=0):
    """What this weighting buys, averaged over random batches of the TRAINING size.

    ``hz`` is (2, N, C, P). Returns the across-subject std of the pooled feature — the
    quantity ``gap_feat_std_mean`` reports — and the cross-view correlation of the pooled
    feature, which is how much of what survives pooling is the SHARED subject factor rather
    than each view's own within-subject interaction.
    """
    from training.losses import barlow_twins_loss

    w = torch.as_tensor(weights, dtype=torch.float32)
    rng = np.random.RandomState(seed)
    n, d = hz.shape[1], hz.shape[2]
    b = min(batch_size, n)
    stds, corrs, on_diags = [], [], []
    for _ in range(draws):
        idx = rng.choice(n, size=b, replace=False)
        pooled = torch.matmul(hz[:, idx], w)  # (2, b, C)
        stds.append(float(pooled.std(dim=1, unbiased=False).mean()))
        x, y = pooled[0], pooled[1]
        x = (x - x.mean(0)) / (x.std(0, unbiased=False) + 1e-8)
        y = (y - y.mean(0)) / (y.std(0, unbiased=False) + 1e-8)
        corrs.append(float((x * y).mean()))
        # The shipped loss, so on_diag matches what training would report at this pooling.
        loss = barlow_twins_loss(pooled, estimated_content_indices=[list(range(d))], subsets=[[0, 1]], lambd=1.0)
        diag = getattr(loss, "_contrastive_diag", None) or {}
        on_diags.append(float(diag.get("on_diag_loss", np.nan)))
    return {
        "feat_std": float(np.mean(stds)),
        "xview_corr": float(np.mean(corrs)),
        "on_diag": float(np.nanmean(on_diags)),
    }


def channel_profiles(hz):
    """Per-(view, channel) position profiles, and how much they AGREE about position.

    The pooled profile averages over channels, so it cannot tell "no channel is spatially
    organised" apart from "channels are organised in DIFFERENT places" — and the second is
    what you would expect from a representation that stores content in channel identity.
    Those two have opposite implications: the first says drop the idea, the second says a
    SHARED position weighting is the wrong instrument (it is one vector for all channels
    by construction) while per-channel weights could still work.

    ``agreement`` is the fraction of the average channel's concentration that survives
    averaging: 1.0 = every channel peaks in the same places, 0.0 = they cancel out. The
    cosine is computed on profiles CENTERED at uniform, because raw profiles are
    non-negative and sum to 1, which makes their cosine high whatever they do.
    """
    v = hz.detach().float().var(dim=1, unbiased=False)  # (2, C, P) across SUBJECTS
    v = (v / (v.sum(-1, keepdim=True) + 1e-12)).reshape(-1, v.shape[-1]).numpy()  # (2C, P)
    p = v.shape[1]
    per = np.array([profile_stats(row)["top10pct_mass"] for row in v])
    pooled = profile_stats(v.mean(0))["top10pct_mass"]
    # Excess over the 0.10 a flat profile gives, so "agreement" is not inflated by the floor.
    per_excess, pooled_excess = float(per.mean()) - 0.10, pooled - 0.10
    c = v - 1.0 / p  # centre at uniform before measuring alignment
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    g = c @ c.T
    off = g[~np.eye(len(g), dtype=bool)]
    return {
        "per_channel_top10_mean": float(per.mean()),
        # The MEAN is diluted by channels that carry nothing — and with a learned content
        # mask most of them may. The upper quartile is what "are ANY channels spatially
        # organised" actually asks, so the verdict gates on p75, not the mean.
        "per_channel_top10_p75": float(np.percentile(per, 75)),
        "per_channel_top10_max": float(per.max()),
        "pooled_top10": pooled,
        "agreement": float(pooled_excess / per_excess) if per_excess > 1e-6 else float("nan"),
        # Diluted the same way: dead channels contribute random profiles at cosine ~0, so
        # read this next to p75 rather than on its own.
        "mean_pairwise_cosine": float(off.mean()),
        "n_profiles": int(v.shape[0]),
    }


def feature_scale(hz):
    """Per-channel across-subject std of the GAP-pooled feature, and the hinge it implies.

    ``--bt-gap-std-coeff`` applies ``relu(1 - std)`` per channel, and it has no restoring
    force above 1 — so once a channel passes the target the term stops pulling and the scale
    parks wherever momentum left it. That makes ``gap_feat_std`` a readout of what the HINGE
    did, not of anything about the representation, and it is why it cannot be used to
    diagnose pooling (``on_diag``/``off_diag`` are scale-invariant anyway).

    What this separates is the state the hinge is actually in:

        below      most channels under 1     hinge pulling hard, scale still climbing
        pinned     tight band just above 1   hinge boundary-active, holding the scale there
        parked     broad, well above 1       hinge inert, inflation already applied

    ``implied_var_loss`` reproduces ``losses.barlow_twins_loss``'s formula exactly (mean over
    channels per view, summed over the two views), so compare it against the run's logged
    ``Contrastive/gap_var_loss_L*``. If they roughly agree, these features are the ones
    training sees; if they diverge badly, that discrepancy is itself the finding and nothing
    below should be trusted until it is explained.
    """
    pooled = hz.mean(-1)  # (2, N, C) — the plain GAP the loss consumes
    std = pooled.std(dim=1, unbiased=False)  # (2, C) across SUBJECTS
    flat = std.reshape(-1).numpy()
    return {
        "std_mean": float(flat.mean()),
        "std_median": float(np.median(flat)),
        "std_min": float(flat.min()),
        "std_max": float(flat.max()),
        "frac_below_1": float((flat < 1.0).mean()),
        "frac_pinned": float(((flat >= 1.0) & (flat < 1.3)).mean()),
        # Exactly losses.barlow_twins_loss's var_loss: relu(1-std).mean() per view, summed.
        "implied_var_loss": float(torch.relu(1.0 - std).mean(dim=1).sum()),
        "n_channels": int(std.shape[1]),
    }


def _ridge_r2(x_tr, y_tr, x_te, y_te, alpha=1.0):
    """Closed-form multi-output ridge. Returns per-target R^2 on the test split."""
    mx = x_tr.mean(0)
    sx = x_tr.std(0) + 1e-8
    my = y_tr.mean(0)
    a = (x_tr - mx) / sx
    w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ (y_tr - my))
    pred = ((x_te - mx) / sx) @ w + my
    ss_res = ((y_te - pred) ** 2).sum(0)
    ss_tot = ((y_te - y_te.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot


def per_factor_recovery(hz, gt, fit_idx, alpha=1.0, null_subtract=True, seed=0):
    """Per-position recovery R^2 for EACH factor separately: returns (K, P).

    The averaged profile is misleading on this project's factor set, and the reason is
    arithmetic rather than subtle. Measured per-factor R^2 on a contrastive run: brain_size
    0.92, lr_asymmetry 0.88, cortical_thickness 0.88 — all globally visible, since any patch
    of a bigger brain looks different — against lesion_x/y/z at 0.10-0.15, which are the
    spatially localised ones. A mean over that set is roughly "how well does this position
    predict brain size", so it reads flat across positions whatever the localised factors
    do, and averaging is what hid them.

    Same non-circular split as ``factor_recovery_profile``: scored on a held-out half of
    ``fit_idx``.
    """
    x = hz.mean(0).numpy()  # (N, C, P), views averaged
    n_fit = len(fit_idx)
    a_idx, b_idx = fit_idx[: n_fit // 2], fit_idx[n_fit // 2 :]
    out = np.zeros((gt.shape[1], x.shape[2]))
    for p in range(x.shape[2]):
        out[:, p] = _ridge_r2(x[a_idx, :, p], gt[a_idx], x[b_idx, :, p], gt[b_idx], alpha)
    if null_subtract:
        # Per-position permutation floor. A ridge on C features at every position returns a
        # positive R^2 on pure chance, and that floor is the same at every position, so it
        # ADDS A CONSTANT to the profile — which pulls the concentration toward flat and
        # hides exactly the mild localisation this is trying to resolve. Subtract it.
        g_null = gt[np.random.RandomState(seed + 991).permutation(len(gt))]
        for p in range(x.shape[2]):
            out[:, p] -= _ridge_r2(x[a_idx, :, p], g_null[a_idx], x[b_idx, :, p], g_null[b_idx], alpha)
    return np.maximum(0.0, out)


def profile_reliability(hz, gt, idx, alpha=1.0):
    """Split-half correlation of each factor's per-position R^2 profile: is its SHAPE real?

    Concentration measured on a noisy profile is inflated, so without this the WORST
    recovered factor reliably looks the MOST localised — backwards, and it happened on this
    project's baseline (lesion_z, weakest at peak R^2 0.39, was the only factor to clear a
    raw concentration threshold).

    A permutation null does NOT fix it: shuffling subject labels drives R^2 to zero at every
    position, and a profile of all zeros with one surviving positive entry normalises to a
    MAXIMALLY concentrated profile. The null then looks more localised than the data.

    Split-half does fix it, and is the standard tool. Estimate each factor's profile on two
    disjoint halves of the subjects and correlate them. Real spatial structure reproduces;
    estimation noise does not. Read alongside the concentration: high top10% with low
    reliability is noise, high top10% with high reliability is localisation.
    """
    h = len(idx) // 2
    a = per_factor_recovery(hz, gt, idx[:h], alpha)
    b = per_factor_recovery(hz, gt, idx[h:], alpha)
    rel = np.full(a.shape[0], np.nan)
    con = np.full((a.shape[0], 2), np.nan)
    for k in range(a.shape[0]):
        if a[k].std() > 1e-9 and b[k].std() > 1e-9:
            rel[k] = float(np.corrcoef(a[k], b[k])[0, 1])
        for j, arr in enumerate((a[k], b[k])):
            if arr.max() > 0:
                con[k, j] = profile_stats(arr / arr.sum())["top10pct_mass"]
    return rel, con


def profile_overlap(profiles):
    """Mean pairwise cosine between per-factor position profiles, centred at uniform.

    This is the question that decides whether a SHARED weighting could ever work: if the
    localised factors peak in the same places, one weight vector can serve them all; if they
    peak in different places, no single vector can, however well each factor is localised.
    Centred at uniform because non-negative profiles that sum to 1 have high raw cosine
    whatever they do.
    """
    if len(profiles) < 2:
        return float("nan")
    m = np.asarray(profiles, dtype=np.float64)
    m = m / (m.sum(axis=1, keepdims=True) + 1e-12)
    c = m - 1.0 / m.shape[1]
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    g = c @ c.T
    return float(g[~np.eye(len(g), dtype=bool)].mean())


def factor_recovery_profile(hz, gt, fit_idx, alpha=1.0):
    """Weight each position by how well ITS features predict the ground-truth factors.

    Variance is a proxy for "carries subject signal" and a poor one here: this project's own
    measurement is that the factor information sits in the LOW-variance tail, so a position
    can have large across-subject variance and carry no factor at all (background intensity
    is the obvious case). On synthetic data the factors are known, so this measures the
    thing directly instead of proxying it.

    Computed on ``fit_idx`` only — the pooled probe that consumes these weights is scored on
    the held-out half, or the weighting would be fitted and evaluated on the same subjects.
    """
    x = hz.mean(0).numpy()  # (N, C, P), views averaged: content is the shared part
    n_fit = len(fit_idx)
    inner = fit_idx[: n_fit // 2], fit_idx[n_fit // 2 :]  # split again, to score each position
    r2 = np.zeros(x.shape[2])
    for p in range(x.shape[2]):
        s = _ridge_r2(x[inner[0], :, p], gt[inner[0]], x[inner[1], :, p], gt[inner[1]], alpha)
        r2[p] = max(0.0, float(np.mean(s)))
    if r2.sum() <= 0:
        return np.full(x.shape[2], 1.0 / x.shape[2]), r2
    return r2 / r2.sum(), r2


def pooled_factor_r2(hz, gt, weights, fit_idx, eval_idx, alpha=1.0):
    """Mean factor R^2 of the POOLED feature: probe trained on fit_idx, scored on eval_idx."""
    w = torch.as_tensor(np.asarray(weights), dtype=torch.float32)
    pooled = torch.matmul(hz, w).numpy()  # (2, N, C)
    out = []
    for v in range(pooled.shape[0]):
        s = _ridge_r2(pooled[v][fit_idx], gt[fit_idx], pooled[v][eval_idx], gt[eval_idx], alpha)
        out.append(float(np.mean(s)))
    return float(np.mean(out))


def main():
    p = argparse.ArgumentParser(description="Is the GAP term's uniform pooling diluting focal subject signal?")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint-name", default="vqvae_model.pt")
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument(
        "--causal",
        choices=("match", "iid"),
        default="iid",
        help="Factor distribution for the test set. Defaults to 'iid' — NOT 'match', which is "
        "the default in the aggregate-ranking tools — because this script's headline output is "
        "PER-FACTOR, and METHODOLOGY_REPORT is explicit that 'match is for aggregate ranking, "
        "iid is for per-factor attribution; match inflates every factor with a recoverable "
        "parent'. Under a random SCM with edge_prob 0.5, brain_size is recovered at 0.92 and "
        "correlates ~0.8 with ventricle_size, so under 'match' a probe reads brain_size in "
        "disguise at every position and every factor inherits brain_size's flat, global "
        "profile. That compresses exactly the localisation this script exists to measure. Pass "
        "'match' only for the aggregate/variance sections, and do not read the per-factor table "
        "from a 'match' run.",
    )
    p.add_argument("--batch-size", type=int, default=0, help="Training batch size (0 = read from the run's settings).")
    p.add_argument("--draws", type=int, default=16, help="Random batches to average the pooled metrics over.")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--encode-batch", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--fg-floor",
        type=float,
        default=0.01,
        help="A position counts as foreground when its across-subject variance exceeds this "
        "fraction of the maximum. Background separation only — not a tuned parameter.",
    )
    p.add_argument(
        "--null-draws",
        type=int,
        default=20,
        help="Permutations of the variance profile used to build the null distribution.",
    )
    p.add_argument(
        "--min-effect",
        type=float,
        default=0.01,
        help="Smallest cross-view-correlation gain over the null worth a training run. A "
        "judgment call, not a statistic: a significant but tiny effect is still not worth "
        "the GPU hours.",
    )
    p.add_argument("--seed", type=int, default=0)
    cli = p.parse_args()

    from eval.bt_term_balance import _as_views
    from eval.dci import _extract_synthetic_representations
    from eval.run_dci_compare import _CONTENT, _CONTENT_V2
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args
    from models.vqvae import GapPositionPool

    args_ = load_run_args(cli.run_dir)
    grid = getattr(args_, "patch_grid", None)
    if not grid:
        raise SystemExit(
            "This run has no --patch-grid, so it has no patch positions and no GAP pooling "
            "question to answer. --bt-gap-pool only applies to --patch-contrastive runs."
        )
    B = cli.batch_size or int(getattr(args_, "batch_size", 128) or 128)
    fg_masked = bool(getattr(args_, "patch_foreground_mask", False))

    dataset = build_synthetic_test_set(args_, cli.num_samples, causal=cli.causal == "match")
    ckpt = os.path.join(cli.run_dir, cli.checkpoint_name)
    model, _a, device = load_model_from_run_dir(cli.run_dir, ckpt if os.path.exists(ckpt) else None, None)

    ld, _gt, _s1, _s2 = _extract_synthetic_representations(
        model, dataset, device, cli.encode_batch, cli.num_workers, pooling=tuple(grid)
    )
    if cli.level not in ld:
        raise SystemExit(f"Level {cli.level} not present in the extracted representations.")
    c1, c2 = ld[cli.level][_CONTENT], ld[cli.level][_CONTENT_V2]
    if c1 is None or c2 is None or c1.shape[1] == 0:
        raise SystemExit(f"No content features at level {cli.level}.")
    hz = _as_views(c1, c2, int(np.prod(grid)))  # (2, N, C, P)
    n_views, N, C, P = hz.shape

    # The profile the shipped module would compute, so these numbers cannot drift from
    # training's. ema=0 => this batch only, which is what we want for a static readout.
    pool = GapPositionPool(P, mode="variance", ema=0.0)
    a_var = pool.weights(hz, None).numpy()
    a_uniform = np.full(P, 1.0 / P)
    rng = np.random.RandomState(cli.seed)

    # Foreground restriction: drop the near-zero-variance positions training never saw.
    v = a_var / a_var.max()
    fg = v > cli.fg_floor
    a_var_fg = a_var[fg] / a_var[fg].sum()

    print(f"\nrun          {cli.run_dir}")
    print(f"level {cli.level}   grid {tuple(grid)} = {P} positions   C={C}   N={N}   batch={B}   draws={cli.draws}")
    print(f"run had --patch-foreground-mask: {fg_masked}")
    print(f"factor distribution: --causal {cli.causal}")
    if cli.causal == "match":
        print("  WARNING: the per-factor table below is NOT VALID under 'match'. Every factor")
        print("  with a recoverable parent reads partly as that parent, and brain_size is both")
        print("  recovered at 0.92 and globally visible, so all factors inherit its flat")
        print("  profile and real localisation is compressed away. Re-run with --causal iid")
        print("  before drawing any per-factor conclusion.")

    print("\n--- shape of the across-subject variance profile (DESCRIPTIVE ONLY) ---------")
    print(f"{'':12s} {'top10%mass':>11s} {'max/unif':>9s} {'eff_pos':>9s} {'entropy':>9s} {'P':>7s}")
    for name, prof in (("all", a_var), ("foreground", a_var_fg)):
        s = profile_stats(prof)
        print(
            f"{name:12s} {s['top10pct_mass']:11.4f} {s['max_over_uniform']:9.2f} "
            f"{s['eff_pos']:9.1f} {s['entropy']:9.4f} {s['n_positions']:7d}"
        )
    print(f"{'uniform ref':12s} {0.10:11.4f} {1.0:9.2f} {float(P):9.1f} {1.0:9.4f} {P:7d}")
    print(f"foreground = variance above {cli.fg_floor:g} x max ({int(fg.sum())} of {P} positions)")
    print("Entropy stays ~1.0 even for profiles that pool usefully — do not read it as the")
    print("answer. The permutation test below is the answer.")

    print("\n--- what the weighting buys, at the training batch size ---------------------")
    print(f"{'pooling':12s} {'feat_std':>10s} {'xview_corr':>11s} {'on_diag':>10s}")
    res = {}
    for name, w in (("uniform", a_uniform), ("variance", a_var)):
        res[name] = pooled_metrics(hz, w, B, cli.draws, seed=cli.seed)
        r = res[name]
        print(f"{name:12s} {r['feat_std']:10.5f} {r['xview_corr']:11.4f} {r['on_diag']:10.4f}")

    # ── permutation null ─────────────────────────────────────────────────────────────
    # Same weight VALUES, reassigned to positions at random. Identical concentration, no
    # information about where the signal sits — so whatever the variance profile beats
    # this by is what knowing the positions is worth.
    null = [
        pooled_metrics(hz, a_var[rng.permutation(P)], B, cli.draws, seed=cli.seed + 1 + k)["xview_corr"]
        for k in range(cli.null_draws)
    ]
    null_mean, null_std = float(np.mean(null)), float(np.std(null))
    print(
        f"{'shuffled':12s} {'':10s} {null_mean:11.4f}   (null mean over {cli.null_draws} permutations, sd {null_std:.4f})"
    )

    gain_vs_null = res["variance"]["xview_corr"] - null_mean
    gain_vs_unif = res["variance"]["xview_corr"] - res["uniform"]["xview_corr"]
    null_share = null_mean - res["uniform"]["xview_corr"]
    z = gain_vs_null / max(null_std, 1e-6)

    print("\nvariance-weighting result")
    print(
        f"  cross-view corr uniform {res['uniform']['xview_corr']:.4f} -> variance {res['variance']['xview_corr']:.4f}"
    )
    print(f"    concentration alone (shuffled - uniform):   {null_share:+.4f}")
    print(f"    POSITION information (variance - shuffled): {gain_vs_null:+.4f}   z = {z:.1f}")
    print(f"  feat_std        uniform {res['uniform']['feat_std']:.5f} -> variance {res['variance']['feat_std']:.5f}")

    # ── do the CHANNELS agree about position? ────────────────────────────────────────
    # The profile above averages over channels, so on its own it cannot separate "nothing
    # is spatially organised" from "channels are organised in different places". Those
    # need different responses, so measure it rather than assume.
    ch = channel_profiles(hz)
    print("\n--- is the flatness real, or channel disagreement? --------------------------")
    print(f"  mean per-channel top10% mass : {ch['per_channel_top10_mean']:.4f}   (0.10 = that channel is flat)")
    print(f"  p75  per-channel top10% mass : {ch['per_channel_top10_p75']:.4f}   <- the verdict gates on this")
    print(f"  max  per-channel top10% mass : {ch['per_channel_top10_max']:.4f}")
    print(f"  channel-averaged top10% mass : {ch['pooled_top10']:.4f}")
    print(f"  agreement (survives averaging): {ch['agreement']:.4f}   (1.0 = same places, 0.0 = cancel out)")
    print(f"  mean pairwise cosine          : {ch['mean_pairwise_cosine']:+.4f}   (profiles centred at uniform)")

    # ── what state is the variance hinge in? ─────────────────────────────────────────
    # Not a pooling question — but this is the number that motivated the pooling work, it
    # was misread, and every run inherits bt_gap_std_coeff from bt_std_coeff, so report it
    # where someone will see it.
    fs = feature_scale(hz)
    gap_std_c = getattr(args_, "bt_gap_std_coeff", None)
    gap_std_c = getattr(args_, "bt_std_coeff", 0.0) if gap_std_c is None else gap_std_c
    inherited = getattr(args_, "bt_gap_std_coeff", None) is None
    print("\n--- variance-hinge state (NOT a pooling readout — see --bt-gap-std-coeff) ---")
    print(f"  bt_gap_std_coeff  : {gap_std_c}{'  (INHERITED from --bt-std-coeff)' if inherited else ''}")
    print(f"  per-channel std   : mean {fs['std_mean']:.4f}  median {fs['std_median']:.4f}  "
          f"min {fs['std_min']:.4f}  max {fs['std_max']:.4f}")  # fmt: skip
    print(f"  below the target 1: {fs['frac_below_1']:.1%} of {fs['n_channels']} channels")
    print(f"  pinned in [1, 1.3): {fs['frac_pinned']:.1%}")
    print(f"  implied var_loss  : {fs['implied_var_loss']:.4f}   <- compare to logged Contrastive/gap_var_loss_L*")
    # The hinge only exists inside the Barlow Twins term. A recon-only baseline carries
    # bt_std_coeff in its settings.json without ever applying it, so classifying its "hinge
    # state" would be inventing a mechanism that never ran.
    _cl_type = getattr(args_, "contrastive_loss_type", "infonce")
    _cl_scale = float(getattr(args_, "scale_contrastive_loss", 1.0) or 0.0)
    if _cl_type != "barlow_twins" or _cl_scale == 0.0:
        print(f"  => NOT APPLICABLE: contrastive_loss_type={_cl_type}, scale_contrastive_loss={_cl_scale}.")
        print("     The variance hinge is inside the Barlow Twins term, so it never ran here —")
        print("     bt_std_coeff is an inherited config value with no effect. The std numbers")
        print("     above describe the representation this model happens to have, not a hinge.")
    elif float(gap_std_c) > 0:
        if fs["frac_below_1"] < 0.02:
            print("  => INERT: essentially every channel is past the target, so the term contributes")
            print("     no gradient while its per-channel inflation stands.")
        elif fs["frac_pinned"] > 0.5:
            print("  => BOUNDARY-ACTIVE: most channels sit just above the target, so the hinge")
            print("     crosses in and out of active every step (a gap_var_loss curve that")
            print("     alternates between 0 and small positive is this). It is spending gradient")
            print("     continuously on a quantity on_diag/off_diag cannot see — the state where")
            print("     lowering --bt-gap-std-coeff to 0.1-0.5 changes the most.")
        else:
            print(f"  => PULLING: {fs['frac_below_1']:.0%} of channels are still below the target, so the hinge")
            print("     is applying its full force and the scale has not converged. A gap_var_loss")
            print("     that is never exactly 0 is this state.")
    if fg_masked:
        print("  CAVEAT: this run used --patch-foreground-mask, and the features above are pooled")
        print("     over ALL positions while training pools over the kept subset only. That alone")
        print("     shifts the scale, so if implied_var_loss disagrees with the logged")
        print("     gap_var_loss (e.g. logged sometimes exactly 0 against a large implied value),")
        print("     believe the logged one and treat this whole section as indicative only. The")
        print("     scale-invariant sections above are unaffected either way.")

    # ── weight by FACTOR RECOVERY instead of variance ────────────────────────────────
    # Variance is a proxy, and this project measured the factor information to sit in the
    # low-variance tail. On synthetic the factors are known, so ask them directly.
    gt = np.asarray(_gt, dtype=np.float64)
    perm = np.random.RandomState(cli.seed).permutation(N)
    fit_idx, eval_idx = perm[: N // 2], perm[N // 2 :]
    a_fac, r2_map = factor_recovery_profile(hz, gt, fit_idx)
    s_fac = profile_stats(a_fac)
    r2_uniform = pooled_factor_r2(hz, gt, a_uniform, fit_idx, eval_idx)
    r2_factor = pooled_factor_r2(hz, gt, a_fac, fit_idx, eval_idx)
    rng2 = np.random.RandomState(cli.seed + 977)
    r2_null = [pooled_factor_r2(hz, gt, a_fac[rng2.permutation(P)], fit_idx, eval_idx) for _ in range(cli.null_draws)]
    r2_null_mean, r2_null_std = float(np.mean(r2_null)), float(np.std(r2_null))
    z_fac = (r2_factor - r2_null_mean) / max(r2_null_std, 1e-6)

    print("\n--- weighting by FACTOR RECOVERY rather than variance -----------------------")
    print(
        f"  per-position factor R^2: max {r2_map.max():.4f}  mean {r2_map.mean():.4f}  nonzero {int((r2_map>0).sum())}/{P}"
    )
    print(f"  profile top10% mass    : {s_fac['top10pct_mass']:.4f}   (0.10 = flat)")
    print(f"  pooled factor R^2  uniform {r2_uniform:.4f} -> factor-weighted {r2_factor:.4f}")
    print(f"    permutation null {r2_null_mean:.4f} (sd {r2_null_std:.4f})   z = {z_fac:.1f}")
    print("  Profile fitted on half the subjects, probe scored on the held-out half.")

    # ── PER-FACTOR spatial profiles ──────────────────────────────────────────────────
    # The averaged profile above is dominated by whichever factors are globally visible,
    # so it cannot see a localised factor at all. Split it.
    from eval.dci import CONTENT_FACTOR_NAMES

    pf = per_factor_recovery(hz, gt, fit_idx)
    # Concentration measured on a NOISY profile is inflated: a factor recovered at R^2 0.1
    # has a per-position profile that is mostly estimation noise, and noise concentrates.
    # Uncorrected, the WORST-recovered factor reliably looks the MOST localised — which is
    # backwards, and did exactly that on this project's baseline (lesion_z, the weakest
    # factor at peak R^2 0.39, was the only one to clear a raw 0.15 threshold). So give each
    # factor its own noise floor: the same measurement with subject labels shuffled, which
    # destroys the factor-feature relationship while preserving every other property.
    reliab, con = profile_reliability(hz, gt, fit_idx)
    names = (CONTENT_FACTOR_NAMES + [f"factor_{i}" for i in range(pf.shape[0])])[: pf.shape[0]]
    print("\n--- per-factor spatial profiles (the averaged one hides localised factors) ---")
    print(f"  {'factor':20s} {'peak R^2':>9s} {'top10%':>8s} {'+/-':>7s} {'sigma':>7s} {'split-half':>11s}")
    rows = []
    for k, nm in enumerate(names):
        prof = pf[k]
        if prof.max() <= 0:
            print(f"  {nm:20s} {0.0:9.4f} {'-':>8s} {'-':>7s} {'-':>7s} {'-':>11s}   (not recovered)")
            continue
        s = profile_stats(prof / prof.sum())
        # Error bar from the two disjoint subject halves. NOT compared against 0.10: top10%
        # is the mass of the largest 10% of entries, so it sits above 0.10 in any finite
        # sample even when the true profile is flat, and the bias is far larger than this
        # error bar. Measured: a genuinely global planted factor reads 0.115 +/- 0.001, i.e.
        # 22 "sigma" above 0.10 while being flat by construction. The reference has to be an
        # empirically flat profile, so it is the LEAST concentrated factor in this set —
        # filled in after the loop, once that minimum is known.
        err = abs(con[k, 0] - con[k, 1]) / 2.0 if np.all(np.isfinite(con[k])) else np.nan
        r = reliab[k]
        rows.append((nm, prof, s, r, err))
    # Reference = the least concentrated factor measured here: an empirically flat profile at
    # this data's own noise level, which 0.10 is not.
    ref_i = int(np.argmin([s["top10pct_mass"] for _, _, s, _, _ in rows])) if rows else -1
    ref = rows[ref_i][2]["top10pct_mass"] if rows else 0.10
    ref_err = rows[ref_i][4] if rows else np.nan
    scored = []
    for nm, prof, s, r, err in rows:
        den = np.sqrt(err**2 + ref_err**2) if np.isfinite(err) and np.isfinite(ref_err) else np.nan
        sigma = (s["top10pct_mass"] - ref) / den if np.isfinite(den) and den > 1e-9 else np.nan
        scored.append((nm, prof, s, r, err, sigma))
        _r = "        n/a" if not np.isfinite(r) else f"{r:11.3f}"
        _e = "    n/a" if not np.isfinite(err) else f"{err:7.3f}"
        _s = "    n/a" if not np.isfinite(sigma) else f"{sigma:7.1f}"
        print(f"  {nm:20s} {prof.max():9.4f} {s['top10pct_mass']:8.3f} {_e} {_s} {_r}")
    print("  R^2 is null-subtracted per position (a ridge on C features returns a positive R^2")
    print("  by chance, and that floor is constant across positions, so it flattens the profile).")
    print(f"  +/- is half the spread between two disjoint subject halves. sigma is measured")
    print(f"  against the LEAST concentrated factor here ({rows[ref_i][0] if rows else '-'}, top10% {ref:.3f}),")
    print("  NOT against 0.10 — top10% takes the largest 10% of entries, so it exceeds 0.10 in")
    print("  any finite sample even for a flat profile (a planted global factor reads 0.115).")
    print("  Localised = sigma above ~2 with a split-half correlation that holds up.")
    # Do the localised ones agree about WHERE? This is what decides whether one shared
    # weight vector could serve them, which is what --bt-gap-pool actually implements.
    loc = [(nm, prof) for nm, prof, s, r, err, sg in scored if np.isfinite(sg) and sg > 2.0 and np.isfinite(r) and r > 0.3]  # fmt: skip
    ov = profile_overlap([p for _, p in loc]) if len(loc) >= 2 else float("nan")
    if len(loc) >= 2:
        print(f"  localised factors ({', '.join(nm for nm, _ in loc)}):")
        print(f"    mean pairwise profile cosine {ov:+.3f}   (1.0 = same places, ~0 = different places)")
        if ov < 0.3:
            print("    => they peak in DIFFERENT places, so no single shared position weighting")
            print("       can serve them — which is exactly what --bt-gap-pool implements.")
        else:
            print("    => they peak in the SAME places; a shared weighting is at least coherent")
            print("       for them, even if the aggregate test above did not clear its bar.")
    elif len(loc) == 1:
        print(f"  only one factor is localised ({loc[0][0]}); a shared weighting would serve it alone.")
    else:
        print("  no factor is meaningfully localised — every one is readable everywhere.")

    # ── combined verdict ─────────────────────────────────────────────────────────────
    var_wins = z > 3.0 and gain_vs_null >= cli.min_effect
    fac_wins = z_fac > 3.0 and (r2_factor - r2_null_mean) >= cli.min_effect
    channels_localised = ch["per_channel_top10_p75"] > 0.13  # meaningfully above the 0.10 floor
    channels_disagree = not (ch["agreement"] > 0.5) if channels_localised else False
    # DRIVEN BY THE PER-FACTOR TABLE, not by the averaged profile. The averaged profile is
    # dominated by whichever factors are globally visible, and this project's are: measured
    # under --causal iid, brain_size and cortical_thickness sit at top10% ~0.12 while
    # ventricle_size reaches 0.66 and the lesions 0.48-0.80. Averaging those together reads
    # flat and says nothing. `loc` already applies both gates (concentration above the
    # least-concentrated factor, and split-half reliability), so use it.
    frac_carrying = float((r2_map >= 0.5 * r2_map.max()).mean()) if r2_map.max() > 0 else 0.0
    factors_localised = len(loc) > 0
    # No factor cleared the gates AND the averaged profile is flat: genuinely nothing to
    # concentrate on. Both conditions, because either alone has a known failure mode.
    factor_delocalised = (not factors_localised) and frac_carrying > 0.8 and s_fac["top10pct_mass"] < 0.15

    print("\n--- verdict ------------------------------------------------------------------")
    # A significantly NEGATIVE z is not "no signal" — it is a finding. It says the
    # high-across-subject-variance positions are anti-aligned with shared content, i.e.
    # the variance is tracking VIEW-SPECIFIC signal (gain/bias/noise, background
    # intensity) rather than subject anatomy. Report it separately or it reads as a null.
    if z < -3.0:
        print(
            f"VARIANCE IS ANTI-ALIGNED WITH CONTENT (z = {z:.1f}, significantly WORSE than its own\n"
            f"permutation: {gain_vs_null:+.4f}). Upweighting the highest-variance positions REMOVES\n"
            "cross-view agreement, so across-subject variance here is dominated by per-view\n"
            "style — the global intensity transforms this project already tracks with\n"
            "eval/style_leak_by_position.py and eval/background_leak_diagnostic.py.\n"
            "--bt-gap-pool variance would actively hurt this model. Read the factor-recovery\n"
            "section below, which does not use variance as a proxy.\n"
        )
    if var_wins or fac_wins:
        which = "variance" if var_wins else "factor-recovery"
        print(
            f"YES via {which} weighting. The concentration carries real positional information\n"
            "rather than just averaging fewer positions."
        )
        if fac_wins and not var_wins:
            print(
                "  NOTE: variance weighting FAILED and factor weighting passed, so variance was\n"
                "  simply the wrong statistic — the shipped --bt-gap-pool variance mode will not\n"
                "  reproduce this. Weighting by factor recovery is supervised and cannot ship as\n"
                "  an unsupervised training signal; treat this as evidence that a better\n"
                "  UNSUPERVISED surrogate is worth looking for, not as a green light."
            )
        else:
            print(f"  Run experiments/gap_pool_variance.yaml; effect to reproduce: {gain_vs_unif:+.4f} xview corr.")
    elif channels_disagree:
        print(
            f"NO for a SHARED position weighting, but not because the representation is flat.\n"
            f"Individual channels ARE spatially organised (mean top10% {ch['per_channel_top10_mean']:.3f} vs 0.10\n"
            f"flat) and they disagree about WHERE (agreement {ch['agreement']:.2f}). One weight vector\n"
            "shared across channels is the wrong instrument by construction — it can only\n"
            "express a position preference every channel shares, and there isn't one.\n"
            "Do NOT run --bt-gap-pool as built. Per-channel weights are the version of this\n"
            "idea that could work, at C x P parameters and a much larger collapse surface."
        )
    elif factors_localised and len(loc) >= 2 and np.isfinite(ov) and ov < 0.3:
        print(
            f"NO for a SHARED position weighting — and NOT because the factors are spread out.\n"
            f"{len(loc)} factors are genuinely localised ({', '.join(nm for nm, _ in loc)}),\n"
            f"but they peak in DIFFERENT places (mean pairwise profile cosine {ov:+.3f}).\n"
            "--bt-gap-pool implements ONE weight vector shared across every channel and every\n"
            "factor, so it can only express a position preference they all share, and there\n"
            "isn't one: mass moved toward the ventricles is mass taken from the lesions.\n"
            "That is a structural mismatch between the instrument and the data, not a\n"
            "tuning problem — do not run it, and do not expect the learned variant to fix it.\n"
            "The version of this idea that could work is PER-FACTOR or per-channel weighting,\n"
            "at C x P parameters and a much larger collapse surface, and it would need a\n"
            "target other than the aggregate BT loss to be worth the complexity."
        )
    elif factors_localised:
        print(
            f"NO on the aggregate test, though {len(loc)} factor(s) are localised\n"
            f"({', '.join(nm for nm, _ in loc)})"
            + (f" and they peak in SIMILAR places (cosine {ov:+.3f}).\n" if np.isfinite(ov) else ".\n")
            + "A shared weighting is at least coherent for them, so the aggregate result is\n"
            "worth re-checking on more subjects before dropping the idea — the permutation\n"
            "tests above may simply lack the power to see an effect this size."
        )
    elif channels_localised and factor_delocalised:
        # The case both of this project's step-60k checkpoints land in, and the one the
        # earlier version of this script mislabelled as "channels are flat" while printing
        # a per-channel concentration 3-5x the flat floor. Variance structure and factor
        # structure are DIFFERENT THINGS and only the second one matters here.
        print(
            f"NO — and the reason is precise. Spatial structure EXISTS: per-channel variance\n"
            f"profiles are concentrated (p75 top10% {ch['per_channel_top10_p75']:.3f} vs 0.10 flat) and the\n"
            f"channels agree about where (agreement {ch['agreement']:.2f}). But the FACTOR INFORMATION is\n"
            f"delocalised: {frac_carrying:.0%} of positions independently predict the factors at half the\n"
            f"best position's R^2 or better (per-position R^2 mean {r2_map.mean():.3f}, max {r2_map.max():.3f}), and\n"
            f"the factor-recovery profile is nearly flat (top10% {s_fac['top10pct_mass']:.3f}).\n"
            "So there is nothing to concentrate ON: essentially every position already\n"
            "carries the factors, and reweighting cannot add information that uniform\n"
            "averaging did not already collect. What the variance profile is peaked on is\n"
            "something else — anatomy and per-view intensity structure, not factor content.\n"
            "Consistent with eval/receptive_field_test.py's finding that far-background\n"
            "positions predict brain_size at R^2 0.77: a feature at position p is not about\n"
            "the anatomy at p. Cross-check with eval/plot_local_vs_global.py."
        )
    else:
        print(
            f"NO — no weighting beats its own permutation (variance z = {z:.1f}, factor z = {z_fac:.1f}),\n"
            f"and no individual FACTOR cleared the localisation gates either (per-channel\n"
            f"variance p75 top10% {ch['per_channel_top10_p75']:.3f}, but that is variance structure, not\n"
            "factor content). Nothing here for a position weighting to exploit.\n"
            "Before believing that: check the run used --causal iid. Under 'match' every\n"
            "factor reads partly as its recoverable parent, and a globally-visible parent\n"
            "flattens every child's profile — which produced a false 'delocalised' reading on\n"
            "this project once already. Cross-check with eval/receptive_field_test.py and\n"
            "eval/plot_local_vs_global.py."
        )
    if z_fac > 3.0 and not fac_wins:
        print(
            f"\n  (Factor weighting DID beat its permutation, z = {z_fac:.1f}, but by only\n"
            f"  {r2_factor - r2_null_mean:+.4f} against --min-effect {cli.min_effect:g}. Statistically real,\n"
            "  practically nothing — and it is supervised, so it could not ship anyway.)"
        )
    print()


if __name__ == "__main__":
    main()
