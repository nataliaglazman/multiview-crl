#!/usr/bin/env python
"""Does the GAP term's uniform mean actually dilute anything? Answer from a checkpoint.

``--bt-gap-pool`` rests on one unverified claim: that the across-SUBJECT variance is
concentrated in a minority of patch positions, so averaging uniformly over all P dilutes
the subject signal by roughly the fraction of positions carrying it. That is the suspected
cause of ``gap_feat_std`` parking at ~0.004 against a hinge target of 1.

If the profile is flat, uniform pooling is not diluting anything, both pooling modes are
fitting noise, and the lever is elsewhere (most likely ``--bt-corr-ema``, which attacks the
off-diagonal's d(d-1)/B sampling floor — a different problem). This settles that on a
checkpoint you already have, with no training run.

The weights come from the shipped ``GapPositionPool`` itself, so the profile measured here
cannot drift from the one training would use.

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

    print("\n--- verdict ------------------------------------------------------------------")
    print(
        f"cross-view corr   uniform {res['uniform']['xview_corr']:.4f} -> variance {res['variance']['xview_corr']:.4f}"
    )
    print(f"  concentration alone (shuffled - uniform):   {null_share:+.4f}")
    print(f"  POSITION information (variance - shuffled): {gain_vs_null:+.4f}   z = {z:.1f}")
    print(f"feat_std          uniform {res['uniform']['feat_std']:.5f} -> variance {res['variance']['feat_std']:.5f}")

    if z <= 3.0:
        print(
            f"\nNO — variance weighting does not beat its own permutation (z = {z:.1f}).\n"
            "Either the profile is flat, or its concentration is not in positions that carry\n"
            "subject signal; either way any gain over uniform is just averaging fewer things,\n"
            "and a learned prior would be fitting noise. Spend the GPU time on the\n"
            "off-diagonal sampling floor (--bt-corr-ema) instead."
        )
    elif gain_vs_null < cli.min_effect:
        print(
            f"\nMARGINAL — the effect is real (z = {z:.1f}) but small ({gain_vs_null:+.4f} against a\n"
            f"--min-effect of {cli.min_effect:g}). Position information exists and is not worth a\n"
            "training run on its own. Revisit if something else raises the stakes."
        )
    else:
        print(
            f"\nYES — variance weighting beats its permutation by {gain_vs_null:+.4f} (z = {z:.1f}), so the\n"
            "concentration carries real positional information rather than just averaging\n"
            "fewer positions. Run experiments/gap_pool_variance.yaml.\n"
            f"Effect size to reproduce in training: {gain_vs_unif:+.4f} cross-view correlation,\n"
            f"feat_std {res['uniform']['feat_std']:.5f} -> {res['variance']['feat_std']:.5f}."
        )
    print()


if __name__ == "__main__":
    main()
