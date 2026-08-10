#!/usr/bin/env python
"""How much of the content block's SUBJECT variance do the 9 LABELED factors explain?

Measured on `synthetic-causal-clean-content-mse-real-lambda-01-normalized`, the pre-norm L0
content maps improve on every probe-free axis between the MCC peak (2k) and the end (89k),
decomposing `z_view = T_view + S(i) + E_view(i)`:

    component                          2k        89k
    ------------------------------------------------
    T  positional template          66.7%      61.3%
    S  subject signal, shared       30.8%      37.4%     +21%
    E  view-specific residual        2.42%      1.35%    -45%
    encoder gauge (T1-T2)           0.1246     0.0588    -53%

Over the same window block-MCC FALLS at every pooling (patch -0.049, gap -0.039, stats
-0.036).  More subject content, encoded more view-invariantly, read out worse.  So the decay
is not content destruction, not view leak, and not encoder drift.

The remaining account is CROWDING.  Synthetic content is ~585-dimensional (z_deformation and
z_fissure are unlabeled); only 9 dims carry labels.  Nothing puts the +6.5 points of new
subject variance in the labeled factors' directions, and packing unlabeled anatomy into a
fixed 44-channel block pushes the labeled factors into a smaller, worse-conditioned subspace
that ridge under-recovers.  That fits what nothing else does: the decay is brain-localised
(unlabeled anatomy is in tissue, background has none) and `info_all` fell 2.9% against the
MCC's 5.4% (capacity preserved, factor-decodability worse).

If crowding is right, **the decay is an artifact of scoring a 44-dim block against 9 of its
~585 true factors, not a pathology** — and the six-seed gate in the investigation doc would
be establishing that the artifact reproduces.

What this measures
------------------
The REVERSE direction of every existing probe.  `info_c2c` and `block_mcc` ask "can the
factors be read out of content?".  This asks "how much of content IS the labeled factors?".

    1. extract pre-SplitGroupNorm L0 content maps for both views (forward hooks, because
       the model's own `enc_out[2]` is post-norm and already pooled)
    2. standardise per view per channel over (samples, positions) — the BT `fold` convention,
       which quotients out the per-channel gain/offset the objective leaves free
    3. average the two views to cancel E, subtract the across-subject mean to cancel T,
       leaving the subject-varying block S(i)
    4. CV-regress every (channel, position) feature on the 9 GT content factors and report
       the VARIANCE-WEIGHTED R^2 — the fraction of subject variance that is labeled

Then split V_S into its labeled and unlabeled parts, in absolute units, at each checkpoint:

    V_S_labeled   = R2_w * V_S
    V_S_unlabeled = (1 - R2_w) * V_S

Reading the result
------------------
    V_S_unlabeled UP a lot, V_S_labeled flat or slightly down
        CROWDING.  The block gained anatomy the benchmark does not label.  The MCC decay is
        a scoring artifact; stop optimising against it and either label z_deformation /
        z_fissure or score with a completeness measure that does not assume 9 factors.

    V_S_labeled DOWN
        The labeled factors really are being degraded, inside their own subspace.  Crowding
        is not the mechanism and the conditioning question is still open — go to --per-factor
        to see which of the 9 are losing ground.

    both roughly flat
        The decay lives in the readout geometry rather than in this variance split; the next
        instrument is the MCC-vs-PCA-rank curve in `patch_mcc_decay --tests geometry`.

Guards this applies
-------------------
* `--probe both` runs a kernel arm alongside ridge.  CROWDING and "the labeled factors went
  nonlinear" make the SAME linear prediction; only the kernel arm separates them.  A linear
  drop with a flat kernel score means re-encoding, not crowding.
* Per-channel pre-standardisation std is printed at every checkpoint.  `_zc` floors the
  divisor at 1e-6, so a channel that went near-constant would be amplified into noise and
  would corrupt this AND the decomposition above.  `content_rank` rising 18.7 -> 39.7 argues
  against it, but check the printed minimum before trusting either.
* Foreground and background strata are sampled and reported separately, never pooled — the
  decay is brain-localised, so a pooled number averages the effect away.
* A permutation floor (factors shuffled across subjects) is reported alongside every score.
* Content channel indices are compared across checkpoints and a mismatch is reported loudly:
  a moved Gumbel mask makes the two columns different representations.

Usage
-----
    # the headline comparison: peak vs final on the same run
    python -m eval.labeled_variance_share --run-dir results/synthetic/BT_RUN \
        --checkpoints vqvae_model_step2000.pt vqvae_model.pt

    # separate crowding from nonlinear re-encoding
    python -m eval.labeled_variance_share --run-dir results/synthetic/BT_RUN \
        --checkpoints vqvae_model_step2000.pt vqvae_model.pt --probe both

    # which of the 9 factors is losing ground (leave-one-out, ridge only)
    python -m eval.labeled_variance_share --run-dir results/synthetic/BT_RUN \
        --checkpoints vqvae_model_step2000.pt vqvae_model.pt --per-factor

NOTE on checkpoint vintage: `render_structure` was fixed in commit 7ac56a3 for
temporal_atrophy / sulcal_widening / lesion.  A checkpoint trained before it saw different
rendering for those factors, so scoring it against the current generator is invalid for them.
brain_size and lr_asymmetry are unaffected.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold

from eval.dci import CONTENT_FACTOR_NAMES

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Extraction — pre-SplitGroupNorm spatial maps, via forward hooks
# --------------------------------------------------------------------------- #
#
# The model applies `content_norms[str(lvl)]` to each encoder's output before the
# contrastive pool (models/vqvae.py:1266), and `_extract_synthetic_representations` reads
# `enc_out[2]`, which is downstream of that.  Hooking the encoder modules themselves is the
# only way to get the pre-norm tensor, and pre-norm is what the decomposition above and the
# investigation doc's "pure offset" claim are both about.


def _install_encoder_hooks(model_inner):
    """Capture each encoder level's raw output, per view. Returns (captured, handles, sep)."""
    captured = {"v0": [None] * model_inner.nb_levels, "v1": [None] * model_inner.nb_levels}
    handles = []

    def make_hook(view, lvl):
        def fn(module, inputs, output):
            captured[view][lvl] = output.detach()

        return fn

    sep = getattr(model_inner, "separate_encoders", False) and (model_inner.encoders_v1 is not None)
    for i, enc in enumerate(model_inner.encoders):
        handles.append(enc.register_forward_hook(make_hook("v0" if sep else "shared", i)))
    if sep:
        for i, enc in enumerate(model_inner.encoders_v1):
            handles.append(enc.register_forward_hook(make_hook("v1", i)))
    if not sep:
        captured["shared"] = [None] * model_inner.nb_levels
    return captured, handles, sep


def _brain_mask(batch, x_v1):
    """Per-sample brain mask as ``(B, 1, D, H, W)`` float, whatever the dataset exposes.

    ``SyntheticBrainDataset._render`` POPS ``brain_mask`` out of ``gt_latents`` and moves it
    to ``batch["mask"]`` (data/datasets.py:709), so reading it from ``gt_latents`` raises
    KeyError on every synthetic run.  ADNI datasets also carry ``mask``.  The intensity
    fallback is last resort and matches the generator's own rule when it has no mask.
    """
    bm = None
    if "mask" in batch and isinstance(batch["mask"], (list, tuple)) and len(batch["mask"]):
        bm = batch["mask"][0]
    elif "brain_mask" in batch.get("gt_latents", {}):
        bm = batch["gt_latents"]["brain_mask"]
    if bm is None:
        bm = (x_v1 > 0.05).float()
    bm = torch.as_tensor(bm).float()
    if bm.dim() == 4:
        bm = bm[:, None]
    return bm


def _content_indices(soft_masks, level, n_channels):
    """Content channel indices for one level, or all channels when the level has no mask."""
    if isinstance(soft_masks, dict) and level in soft_masks:
        mask = soft_masks[level]
        if isinstance(mask, tuple):
            mask = mask[0]
        idx = torch.where(mask.bool().flatten())[0]
        return idx.cpu().numpy().astype(int)
    return np.arange(n_channels, dtype=int)


@torch.no_grad()
def extract_prenorm_content(model, dataset, device, level=0, batch_size=8, num_workers=0, max_samples=None):
    """Pre-norm L0 content maps for both views, plus GT factors and a foreground mask.

    Returns ``(Z1, Z2, gt_content, fg_latent, c_idx)`` with ``Z*`` shaped
    ``(N, Cc, D, H, W)`` on CPU, ``gt_content`` ``(N, 9)``, ``fg_latent`` a boolean
    ``(D, H, W)`` population brain mask resampled to the latent grid.
    """
    from torch.utils.data import DataLoader

    inner = model.module if hasattr(model, "module") else model
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    Z1, Z2, GT, MASK = [], [], [], []
    c_idx = None
    seen = 0

    for batch in loader:
        imgs = batch["image"]
        x = torch.cat(imgs, dim=0).to(device)

        captured, handles, sep = _install_encoder_hooks(inner)
        try:
            out = model(x, return_recon=False, pool_only=True, n_views=len(imgs), subsets=[(0, 1)], patch_grid=None)
            soft_masks = out[6] if isinstance(out, tuple) and len(out) > 6 else {}
        finally:
            for h in handles:
                h.remove()

        if sep:
            s1, s2 = captured["v0"][level], captured["v1"][level]
        else:
            s = captured["shared"][level]
            B = s.shape[0] // len(imgs)
            s1, s2 = s[:B], s[B : 2 * B]

        if c_idx is None:
            c_idx = _content_indices(soft_masks, level, s1.shape[1])
        Z1.append(s1[:, c_idx].cpu())
        Z2.append(s2[:, c_idx].cpu())
        GT.append(batch["gt_latents"]["z_content"].cpu().numpy())

        MASK.append(F.adaptive_avg_pool3d(_brain_mask(batch, imgs[0]), s1.shape[2:]).cpu())

        seen += s1.shape[0]
        if max_samples and seen >= max_samples:
            break

    Z1 = torch.cat(Z1)[:max_samples].double()
    Z2 = torch.cat(Z2)[:max_samples].double()
    gt = np.concatenate(GT, axis=0)[:max_samples].astype(np.float64)
    fg = (torch.cat(MASK)[:max_samples].mean(0)[0] > 0.5).numpy()
    return Z1, Z2, gt, fg, c_idx


# --------------------------------------------------------------------------- #
# The decomposition
# --------------------------------------------------------------------------- #


def zc(Z, eps=1e-6):
    """Per-view, per-channel standardisation over (samples, positions) — BT `fold`.

    Quotients out exactly the per-channel gain/offset that `on_diag` leaves free, so what
    survives is a statement about structure rather than about encoder parameterisation.
    Returns ``(standardised, per_channel_std)``; inspect the std to catch dead channels,
    which the ``eps`` floor would otherwise amplify into noise.
    """
    m = Z.mean(dim=(0, 2, 3, 4), keepdim=True)
    sd = Z.std(dim=(0, 2, 3, 4), keepdim=True)
    return (Z - m) / sd.clamp_min(eps), sd.flatten().numpy()


def subject_block(Z1, Z2):
    """Kill E by averaging views, kill T by removing the across-subject mean.

    Returns ``S`` of shape ``(N, Cc, D, H, W)``: the subject-varying content, with the
    positional template and (to first order) the view-specific residual removed.
    """
    Zb = 0.5 * (Z1 + Z2)
    return Zb - Zb.mean(dim=0, keepdim=True)


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #


def _fit_predict(kind, Xtr, Ytr, Xte, alpha, gamma):
    if kind == "ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=alpha)
    else:
        from sklearn.kernel_ridge import KernelRidge

        model = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
    model.fit(Xtr, Ytr)
    return np.asarray(model.predict(Xte))


def cv_variance_weighted_r2(X, Y, kind="ridge", n_splits=5, seed=0, alpha=1.0, gamma=None):
    """Variance-weighted CV R^2 of ``X -> Y`` over all output columns at once.

    Weighting by each feature's own variance (rather than averaging per-feature R^2) makes
    the number mean "fraction of the block's subject variance that the labeled factors
    explain", which is the quantity the crowding hypothesis is about.  An unweighted mean
    would let near-dead background positions count as much as high-variance brain ones.

    Also returns the unweighted mean for reference — a large gap between them means the
    explained variance is concentrated in a few features.
    """
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if gamma is None:
        gamma = 1.0 / max(X.shape[1], 1)

    sse = np.zeros(Y.shape[1])
    sst = np.zeros(Y.shape[1])
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X):
        sc = StandardScaler().fit(X[tr])
        Yhat = _fit_predict(kind, sc.transform(X[tr]), Y[tr], sc.transform(X[te]), alpha, gamma)
        # Centre the held-out fold on the TRAIN mean, so SST is the honest baseline a
        # constant predictor would achieve rather than an oracle fold mean.
        mu = Y[tr].mean(0)
        sse += ((Y[te] - Yhat) ** 2).sum(0)
        sst += ((Y[te] - mu) ** 2).sum(0)

    with np.errstate(divide="ignore", invalid="ignore"):
        per_feature = 1.0 - sse / sst
    per_feature = np.nan_to_num(per_feature, nan=0.0, posinf=0.0, neginf=0.0)
    weighted = float(1.0 - sse.sum() / max(sst.sum(), 1e-300))
    return {
        "weighted": weighted,
        "unweighted": float(np.mean(per_feature)),
        "per_feature": per_feature,
        "total_var": float(sst.sum() / len(Y)),
    }


# --------------------------------------------------------------------------- #
# Feature sampling
# --------------------------------------------------------------------------- #


def sample_features(S, fg, n_per_stratum, seed=0):
    """Flatten ``S`` to ``(N, F)`` per stratum, sampling positions inside/outside the brain.

    The strata are never pooled: the decay is brain-localised, so one number over both would
    average the effect away.  The returned index arrays are ``(channel, flat_position)``
    pairs, identical across checkpoints for a fixed seed, which is what makes the two
    columns comparable feature by feature.
    """
    rng = np.random.default_rng(seed)
    N, C = S.shape[0], S.shape[1]
    flat = S.reshape(N, C, -1).numpy()
    fg_flat = fg.reshape(-1)

    # "all" exists only as a consistency anchor against the notebook decomposition — its V_S
    # should land near V_S + V_E/2 from that run (~0.32 at 2k, ~0.38 at 89k), because view
    # averaging halves E rather than removing it.  Draw conclusions from the strata, not here.
    out = {}
    for name, keep in (("foreground", fg_flat), ("background", ~fg_flat), ("all", np.ones_like(fg_flat))):
        pos = np.flatnonzero(keep)
        if pos.size == 0:
            continue
        n = min(n_per_stratum, C * pos.size)
        ch = rng.integers(0, C, size=n)
        pp = pos[rng.integers(0, pos.size, size=n)]
        out[name] = (flat[:, ch, pp], np.stack([ch, pp], 1))
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def score_checkpoint(model, dataset, device, cli):
    Z1, Z2, gt, fg, c_idx = extract_prenorm_content(
        model, dataset, device, cli.level, cli.batch_size, cli.num_workers, cli.num_samples
    )
    Z1s, sd1 = zc(Z1)
    Z2s, sd2 = zc(Z2)
    S = subject_block(Z1s, Z2s)

    strata = sample_features(S, fg, cli.n_features, seed=cli.seed)
    rng = np.random.default_rng(cli.seed)
    perm = rng.permutation(len(gt))

    res = {
        "c_idx": c_idx,
        "min_std": float(min(sd1.min(), sd2.min())),
        "fg_positions": int(fg.sum()),
        "n_positions": int(fg.size),
        "n_samples": int(len(gt)),
        "strata": {},
    }
    kinds = ("ridge", "kernel") if cli.probe == "both" else (cli.probe,)
    for name, (Y, _idx) in strata.items():
        entry = {"V_S": float(Y.var(0).sum() / Y.shape[1])}
        for kind in kinds:
            r = cv_variance_weighted_r2(gt, Y, kind=kind, n_splits=cli.n_splits, seed=cli.seed, alpha=cli.alpha)
            null = cv_variance_weighted_r2(
                gt[perm], Y, kind=kind, n_splits=cli.n_splits, seed=cli.seed, alpha=cli.alpha
            )
            entry[kind] = {
                "r2_w": r["weighted"],
                "r2_u": r["unweighted"],
                "null": null["weighted"],
                "var_per_feature": r["total_var"] / max(Y.shape[1], 1),
            }
        res["strata"][name] = entry

    if cli.per_factor:
        res["per_factor"] = {}
        for name, (Y, _idx) in strata.items():
            full = cv_variance_weighted_r2(gt, Y, "ridge", cli.n_splits, cli.seed, cli.alpha)["weighted"]
            drops = {}
            for j, fname in enumerate(CONTENT_FACTOR_NAMES[: gt.shape[1]]):
                Xj = np.delete(gt, j, axis=1)
                r = cv_variance_weighted_r2(Xj, Y, "ridge", cli.n_splits, cli.seed, cli.alpha)["weighted"]
                drops[fname] = full - r
            res["per_factor"][name] = drops
    return res


def _fmt_block(label, res, kinds):
    lines = [
        f"\n  {label}",
        f"    samples {res['n_samples']}   content channels {len(res['c_idx'])}"
        f"   min per-channel std {res['min_std']:.3e}",
    ]
    frac = res["fg_positions"] / max(res["n_positions"], 1)
    lines.append(f"    latent foreground {res['fg_positions']}/{res['n_positions']} positions ({frac:.1%})")
    if res["min_std"] < 1e-4:
        lines.append("    WARNING: a channel is near-constant; zc's 1e-6 floor is amplifying it into noise")
    if not 0.02 < frac < 0.98:
        lines.append("    WARNING: degenerate brain mask — the foreground/background split is not meaningful")
    for sname, entry in res["strata"].items():
        lines.append(f"    {sname:<11} V_S/feature {entry['V_S']:.4f}")
        for kind in kinds:
            e = entry[kind]
            lab = e["r2_w"] * entry["V_S"]
            unlab = (1.0 - e["r2_w"]) * entry["V_S"]
            lines.append(
                f"      {kind:<7} R2_w {e['r2_w']:6.3f} (null {e['null']:+.3f}, unweighted {e['r2_u']:6.3f})"
                f"   labeled {lab:.4f}   UNlabeled {unlab:.4f}"
            )
    return "\n".join(lines)


def _verdict(a, b, kinds):
    """Compare two checkpoints on the foreground stratum and name the outcome."""
    if "foreground" not in a["strata"] or "foreground" not in b["strata"]:
        return "no foreground stratum — cannot call it"
    ea, eb = a["strata"]["foreground"], b["strata"]["foreground"]
    kind = "ridge" if "ridge" in kinds else kinds[0]
    lab_a, lab_b = ea[kind]["r2_w"] * ea["V_S"], eb[kind]["r2_w"] * eb["V_S"]
    unl_a, unl_b = (1 - ea[kind]["r2_w"]) * ea["V_S"], (1 - eb[kind]["r2_w"]) * eb["V_S"]
    d_lab = (lab_b - lab_a) / max(abs(lab_a), 1e-12)
    d_unl = (unl_b - unl_a) / max(abs(unl_a), 1e-12)

    out = [
        f"    labeled subject variance   {lab_a:.4f} -> {lab_b:.4f}  ({d_lab:+.1%})",
        f"    UNlabeled subject variance {unl_a:.4f} -> {unl_b:.4f}  ({d_unl:+.1%})",
    ]
    if d_unl > 0.15 and d_lab > -0.10:
        out.append("    => CROWDING. The block gained anatomy the benchmark does not label.")
        out.append("       The MCC decay is a scoring artifact; the seed gate would reproduce the artifact.")
    elif d_lab < -0.10:
        out.append("    => LABELED FACTORS DEGRADED. Crowding is not the mechanism.")
        out.append("       Run --per-factor, then patch_mcc_decay --tests geometry.")
    else:
        out.append("    => NEITHER moved much. The decay is in the readout geometry, not this split.")
        out.append("       Next instrument: patch_mcc_decay --tests geometry (MCC-vs-PCA-rank).")
    if "kernel" in kinds:
        ka, kb = ea["kernel"]["r2_w"], eb["kernel"]["r2_w"]
        ra, rb = ea["ridge"]["r2_w"], eb["ridge"]["r2_w"]
        if (rb - ra) < -0.02 and (kb - ka) > -0.01:
            out.append("    NOTE: ridge fell but kernel held — this is NONLINEAR RE-ENCODING, not crowding.")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description="Fraction of content's subject variance that the 9 labeled factors explain")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True, help="checkpoint filenames within --run-dir")
    p.add_argument("--num-samples", type=int, default=600)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--causal",
        choices=("match", "iid"),
        default="match",
        help="match reproduces the training SCM; iid drops it and shifts every number",
    )
    p.add_argument("--n-features", type=int, default=3000, help="sampled (channel, position) features per stratum")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe", choices=("ridge", "kernel", "both"), default="ridge")
    p.add_argument("--per-factor", action="store_true", help="leave-one-out per factor (ridge only)")
    p.add_argument("--out", default="")
    cli = p.parse_args()

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args

    ref_args = load_run_args(cli.run_dir)
    dataset = build_synthetic_test_set(ref_args, cli.num_samples, causal=cli.causal == "match")
    kinds = ("ridge", "kernel") if cli.probe == "both" else (cli.probe,)

    results = {}
    for name in cli.checkpoints:
        ckpt = os.path.join(cli.run_dir, name)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)
        model, _a, device = load_model_from_run_dir(cli.run_dir, ckpt, None)
        logger.info("scoring %s", name)
        results[name] = score_checkpoint(model, dataset, device, cli)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 96)
    print("  HOW MUCH OF CONTENT'S SUBJECT VARIANCE IS LABELED?")
    print("  pre-SplitGroupNorm L%d, views averaged (kills E), across-subject mean removed (kills T)" % cli.level)
    print("=" * 96)
    for name in cli.checkpoints:
        print(_fmt_block(name, results[name], kinds))

    names = list(cli.checkpoints)
    if len(names) >= 2:
        a, b = results[names[0]], results[names[-1]]
        if set(a["c_idx"].tolist()) != set(b["c_idx"].tolist()):
            print("\n  WARNING: content channel indices DIFFER between checkpoints — the Gumbel mask moved,")
            print("           so the two columns are different representations and are not comparable.")
        print(f"\n  {names[0]}  ->  {names[-1]}   (foreground stratum)")
        print(_verdict(a, b, kinds))

    if cli.per_factor:
        print("\n  Leave-one-out unique contribution, variance-weighted (ridge, foreground):")
        for name in names:
            pf = results[name].get("per_factor", {}).get("foreground", {})
            if pf:
                top = sorted(pf.items(), key=lambda kv: -kv[1])
                print(f"    {name}: " + "  ".join(f"{k} {v:+.4f}" for k, v in top))

    if cli.out:
        with open(cli.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "checkpoint",
                    "stratum",
                    "probe",
                    "V_S",
                    "r2_weighted",
                    "r2_unweighted",
                    "null",
                    "labeled",
                    "unlabeled",
                ]
            )
            for name in names:
                for sname, entry in results[name]["strata"].items():
                    for kind in kinds:
                        e = entry[kind]
                        w.writerow(
                            [
                                name,
                                sname,
                                kind,
                                entry["V_S"],
                                e["r2_w"],
                                e["r2_u"],
                                e["null"],
                                e["r2_w"] * entry["V_S"],
                                (1 - e["r2_w"]) * entry["V_S"],
                            ]
                        )
        print(f"\n  wrote {cli.out}")


if __name__ == "__main__":
    main()
