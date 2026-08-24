#!/usr/bin/env python
"""Would PER-CHANNEL position weighting beat uniform GAP? Supervised ceiling, from a checkpoint.

``--bt-gap-pool`` learns ONE position weighting shared by every channel, and it is rejected on
this project's data because the factors are localised in mutually orthogonal places (see
``eval/gap_pool_profile.py``): mass moved toward the ventricles is mass taken from the lesions.
The obvious next idea is to give each channel its own profile, so different channels can read
different anatomy.

Before building that (C x P parameters, a much larger collapse surface, and a second entropy
floor to tune), this measures whether it could pay off AT ALL — by handing the scheme ground
truth and asking how much it beats uniform GAP even then. If the SUPERVISED ceiling is close to
GAP, no unsupervised objective will do better and the idea is dead without a training run.

Three poolings, scored on identical held-out subjects:

    uniform         GAP: the plain mean over positions               (N, C)
    per-channel     each channel reads its single most               (N, C)
                    factor-informative position, chosen on the
                    fit split only
    full patch      every (channel, position), PCA-reduced           (N, C*P)

The first two have IDENTICAL dimensionality, so the comparison is not confounded by feature
count — the only difference is WHERE each channel reads from. The third is the absolute
ceiling: what is recoverable if the spatial layout is used without restriction. Read it as:

    per-channel ~= uniform          the idea is dead; channels have nothing distinct to read
    per-channel >> uniform          real headroom; the unsupervised version is worth designing
    full patch  >> per-channel      the information is in spatial PATTERN, not in one position
                                    per channel, and no per-channel POOLING recovers it --
                                    that argues for keeping the patch term, not for weighting

Hard selection (one position per channel) is a LOWER bound on soft per-channel weighting, and
deliberately so: it is the cheapest member of the family, so if it shows nothing the soft
version is not worth writing, while if it shows a lot the soft version can only do better.

The per-channel choice uses factor labels, so this is a ceiling and not a shippable method.
Defaults to --causal iid for the same reason gap_pool_profile does: under 'match' every factor
reads partly as its recoverable parent and per-factor structure is compressed away.

Usage
-----
    python -m eval.per_channel_pool_ceiling --run-dir results/synthetic/RUN
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _ridge_r2(x_tr, y_tr, x_te, y_te, alpha=1.0):
    """Closed-form multi-output ridge; per-target R^2 on the test split."""
    mx, sx, my = x_tr.mean(0), x_tr.std(0) + 1e-8, y_tr.mean(0)
    a = (x_tr - mx) / sx
    w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ (y_tr - my))
    pred = ((x_te - mx) / sx) @ w + my
    ss_res = ((y_te - pred) ** 2).sum(0)
    ss_tot = ((y_te - y_te.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot


def _pca_reduce(x_tr, x_te, dim):
    """PCA fitted on the TRAIN split only, applied to both."""
    mu = x_tr.mean(0)
    _, _, vt = np.linalg.svd(x_tr - mu, full_matrices=False)
    comp = vt[:dim]
    return (x_tr - mu) @ comp.T, (x_te - mu) @ comp.T


def best_position_per_channel_for_factor(x, gt, fit_idx, k):
    """Each channel's best position FOR ONE FACTOR — the most generous per-channel ceiling.

    ``best_position_per_channel`` picks one position per channel by summing signal over ALL
    factors, which is what any shippable scheme must do: there is one pooled representation,
    not one per factor. That single allocation is dominated by whichever factors correlate
    most strongly, and the rest starve — the same orthogonality problem one level down.

    This relaxes that by letting the allocation depend on the factor, which no unsupervised
    objective could ever do. It answers a narrower question: is a factor unreachable because
    the ALLOCATION went elsewhere, or because no single-position-per-channel read can express
    it at all? If a factor stays near zero even here, its information lives in the spatial
    PATTERN and no per-channel pooling recovers it.
    """
    xf, yf = x[fit_idx], gt[fit_idx, k]
    xz = (xf - xf.mean(0)) / (xf.std(0) + 1e-8)
    yz = (yf - yf.mean()) / (yf.std() + 1e-8)
    corr = np.einsum("ncp,n->cp", xz, yz) / len(fit_idx)
    return (corr**2).argmax(1)


def best_position_per_channel(x, gt, fit_idx):
    """For each channel, the position whose scalar feature carries the most factor signal.

    Chosen on ``fit_idx`` ONLY — the pooled features it defines are scored on held-out
    subjects, or the selection would be fitted and evaluated on the same data and every
    channel would appear informative.

    Univariate R^2 is the squared Pearson correlation, so the whole (C, P, K) score tensor is
    one vectorised operation rather than C*P regressions.
    """
    xf = x[fit_idx]  # (n, C, P)
    yf = gt[fit_idx]  # (n, K)
    xz = (xf - xf.mean(0)) / (xf.std(0) + 1e-8)
    yz = (yf - yf.mean(0)) / (yf.std(0) + 1e-8)
    n = len(fit_idx)
    # corr[c, p, k] = <xz[:, c, p], yz[:, k]> / n
    corr = np.einsum("ncp,nk->cpk", xz, yz) / n
    score = (corr**2).sum(-1)  # (C, P): total factor signal at this (channel, position)
    return score.argmax(1), score


def main():
    p = argparse.ArgumentParser(description="Supervised ceiling for per-channel position weighting.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint-name", default="vqvae_model.pt")
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument(
        "--causal",
        choices=("match", "iid"),
        default="iid",
        help="'iid' by default: per-factor structure is what this measures, and under 'match' "
        "every factor with a recoverable parent inherits that parent's profile.",
    )
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--encode-batch", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--patch-dim",
        type=int,
        default=0,
        help="PCA components for the full-patch ceiling (0 = min(128, n_fit/4), matching this "
        "project's --probe-dim auto convention: at C*P=22528 features against ~1000 fit "
        "subjects, unreduced ridge returns negative R^2 on weak targets).",
    )
    p.add_argument("--seed", type=int, default=0)
    cli = p.parse_args()

    from eval.bt_term_balance import _as_views
    from eval.dci import CONTENT_FACTOR_NAMES, _extract_synthetic_representations
    from eval.run_dci_compare import _CONTENT, _CONTENT_V2
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args

    args_ = load_run_args(cli.run_dir)
    grid = getattr(args_, "patch_grid", None)
    if not grid:
        raise SystemExit("This run has no --patch-grid: no positions, nothing to weight.")

    dataset = build_synthetic_test_set(args_, cli.num_samples, causal=cli.causal == "match")
    ckpt = os.path.join(cli.run_dir, cli.checkpoint_name)
    model, _a, device = load_model_from_run_dir(cli.run_dir, ckpt if os.path.exists(ckpt) else None, None)
    ld, gt, _s1, _s2 = _extract_synthetic_representations(
        model, dataset, device, cli.encode_batch, cli.num_workers, pooling=tuple(grid)
    )
    if cli.level not in ld:
        raise SystemExit(f"Level {cli.level} not present.")
    c1, c2 = ld[cli.level][_CONTENT], ld[cli.level][_CONTENT_V2]
    if c1 is None or c2 is None or c1.shape[1] == 0:
        raise SystemExit(f"No content features at level {cli.level}.")
    hz = _as_views(c1, c2, int(np.prod(grid)))
    x = hz.mean(0).numpy()  # (N, C, P), views averaged: content is the shared part
    gt = np.asarray(gt, dtype=np.float64)
    n_all, C, P = x.shape

    perm = np.random.RandomState(cli.seed).permutation(n_all)
    fit_idx, eval_idx = perm[: n_all // 2], perm[n_all // 2 :]
    names = (CONTENT_FACTOR_NAMES + [f"factor_{i}" for i in range(gt.shape[1])])[: gt.shape[1]]

    print(f"\nrun     {cli.run_dir}")
    print(f"level {cli.level}   grid {tuple(grid)} = {P} positions   C={C}   N={n_all}")
    print(f"factor distribution: --causal {cli.causal}")
    print(f"fit {len(fit_idx)} subjects / score {len(eval_idx)} held out\n")

    best_p, score = best_position_per_channel(x, gt, fit_idx)
    gap = x.mean(-1)  # (N, C)
    per_ch = np.stack([x[:, c, best_p[c]] for c in range(C)], axis=1)  # (N, C)
    flat = x.reshape(n_all, -1)  # (N, C*P)
    pdim = cli.patch_dim or min(128, max(8, len(fit_idx) // 4))
    f_tr, f_te = _pca_reduce(flat[fit_idx], flat[eval_idx], pdim)

    res = {}
    res["uniform GAP"] = _ridge_r2(gap[fit_idx], gt[fit_idx], gap[eval_idx], gt[eval_idx])
    res["per-channel best"] = _ridge_r2(per_ch[fit_idx], gt[fit_idx], per_ch[eval_idx], gt[eval_idx])
    # Most generous per-channel ceiling: allocation chosen per factor. Unshippable (no
    # unsupervised objective can allocate per factor), but it separates "this factor lost the
    # allocation" from "no single-position read can express this factor at all".
    per_fac = np.empty(gt.shape[1])
    for k in range(gt.shape[1]):
        bpk = best_position_per_channel_for_factor(x, gt, fit_idx, k)
        pk = np.stack([x[:, c, bpk[c]] for c in range(C)], axis=1)
        per_fac[k] = _ridge_r2(pk[fit_idx], gt[fit_idx, k : k + 1], pk[eval_idx], gt[eval_idx, k : k + 1])[0]
    res["per-channel PER-FACTOR"] = per_fac
    res[f"full patch (PCA {pdim})"] = _ridge_r2(f_tr, gt[fit_idx], f_te, gt[eval_idx])

    print("--- factor R^2 by pooling (held-out subjects) -------------------------------")
    hdr = "  {:24s}".format("pooling") + "".join(f"{n[:11]:>12s}" for n in names) + f"{'MEAN':>9s}"
    print(hdr)
    for k, v in res.items():
        print("  {:24s}".format(k) + "".join(f"{r:12.3f}" for r in v) + f"{float(np.mean(v)):9.3f}")
    print(f"  {'':24s}" + "".join(f"{'':12s}" for _ in names) + f"{'':9s}")
    print("  uniform GAP and per-channel best have the SAME dimensionality (C), so the only")
    print("  difference between them is where each channel reads from.")

    u_ = res["uniform GAP"]
    pc_ = res["per-channel best"]
    pf_ = res["per-channel PER-FACTOR"]
    fp_ = res[f"full patch (PCA {pdim})"]
    gain = float(np.mean(pc_)) - float(np.mean(u_))
    ceil = float(np.mean(fp_)) - float(np.mean(u_))

    # PER FACTOR, because the mean hides the whole result. On this project's contrastive
    # checkpoint the mean reads "+0.15, weak" while sulcal_widening goes -0.014 -> 0.745 and
    # ventricle_size goes 0.173 -> -0.021. Those need different responses and the average
    # describes neither.
    print("\n--- per factor: who does per-channel selection actually serve? --------------")
    print(f"  {'factor':20s} {'GAP':>8s} {'per-ch':>8s} {'per-fac':>8s} {'patch':>8s}   diagnosis")
    served, starved, pattern_only = [], [], []
    for k, nm in enumerate(names):
        d = ""
        if fp_[k] - max(u_[k], 0) < 0.05:
            d = "no spatial info to recover"
        elif pc_[k] - u_[k] > 0.05:
            d = "SERVED by the shared allocation"
            served.append(nm)
        elif pf_[k] - max(u_[k], 0) > 0.05:
            d = "STARVED — reachable, lost the allocation"
            starved.append(nm)
        else:
            d = "PATTERN-ONLY — no per-channel read works"
            pattern_only.append(nm)
        print(f"  {nm:20s} {u_[k]:8.3f} {pc_[k]:8.3f} {pf_[k]:8.3f} {fp_[k]:8.3f}   {d}")
    print("  per-fac = each channel picks its best position FOR THAT FACTOR: unshippable, and")
    print("  the most generous per-channel ceiling there is. A factor still flat there cannot")
    print("  be reached by any per-channel pooling, however the weights are learned.")

    print("\n--- how much do the channels actually differ? ------------------------------")
    uniq = len(np.unique(best_p))
    print(f"  distinct best positions across {C} channels: {uniq}")
    print(f"  most common position claimed by {int(np.bincount(best_p).max())} channels")
    print("  DESCRIPTIVE ONLY — do not read a high count as evidence of useful specialisation.")
    print("  It is the opposite: when the selection is uninformative it scatters across many")
    print("  positions, and when channels genuinely group onto shared anatomy it concentrates")
    print("  on few. Measured on planted data: channels specialised to 4 factors used 4 of 24")
    print("  distinct positions and WON, while a fully-distributed code used 22 of 24 and was")
    print("  dead. The R^2 comparison decides; this line only says what the selection did.")

    print("\n--- verdict ----------------------------------------------------------------")
    print(f"  per-channel - uniform : {gain:+.4f} mean factor R^2")
    print(f"  full patch  - uniform : {ceil:+.4f}  (absolute ceiling for using spatial layout)")
    if gain < 0.01:
        print(
            "\n  DEAD. Even choosing each channel's best position WITH GROUND TRUTH barely beats\n"
            "  the uniform mean, so no unsupervised per-channel weighting will do better.\n"
            "  Do not build it."
        )
        if ceil > 0.05:
            print(
                f"  Note the full-patch ceiling is much higher ({ceil:+.4f}): the spatial information\n"
                "  is real, but it lives in the PATTERN across positions, not in one position per\n"
                "  channel. That argues for keeping the patch term, not for weighting the GAP one."
            )
    elif gain < 0.5 * ceil:
        print(
            f"\n  WEAK. Per-channel selection recovers {gain / ceil:.0%} of what the full spatial layout\n"
            "  offers. Some headroom, but most of the information is in the spatial pattern that\n"
            "  a per-channel pooling cannot express, and this is the SUPERVISED ceiling — an\n"
            "  unsupervised objective would get less. Weigh that against C x P parameters."
        )
    else:
        print(
            f"\n  WORTH BUILDING. Per-channel selection captures {gain / ceil:.0%} of the full-patch\n"
            "  ceiling at the same dimensionality as GAP, so the information is in fact\n"
            "  reachable one-position-per-channel. Next question is whether an UNSUPERVISED\n"
            "  objective can find those positions — this used labels to pick them."
        )
    print()


if __name__ == "__main__":
    main()
