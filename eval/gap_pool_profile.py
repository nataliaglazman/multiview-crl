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
    p.add_argument("--causal", choices=("match", "iid"), default="match")
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

    # ── combined verdict ─────────────────────────────────────────────────────────────
    var_wins = z > 3.0 and gain_vs_null >= cli.min_effect
    fac_wins = z_fac > 3.0 and (r2_factor - r2_null_mean) >= cli.min_effect
    channels_localised = ch["per_channel_top10_p75"] > 0.13  # meaningfully above the 0.10 floor
    channels_disagree = not (ch["agreement"] > 0.5) if channels_localised else False

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
    else:
        print(
            f"NO — no weighting beats its own permutation (variance z = {z:.1f}, factor z = {z_fac:.1f})\n"
            f"and individual channels are flat too (mean top10% {ch['per_channel_top10_mean']:.3f} against a\n"
            "0.10 floor). The representation is spatially delocalised: a feature at position p\n"
            "is not about the anatomy at position p, so there is no positional structure for\n"
            "ANY position weighting to exploit — shared or per-channel.\n"
            "This is consistent with content being stored in channel identity rather than\n"
            "spatial layout. Cross-check with eval/receptive_field_test.py and\n"
            "eval/plot_local_vs_global.py. Note --bt-corr-ema is already 0.99 in every\n"
            "synthetic config, so the off-diagonal sampling floor is not the fallback."
        )
    print()


if __name__ == "__main__":
    main()
