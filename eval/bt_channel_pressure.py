#!/usr/bin/env python
"""Does the BT patch term penalise exactly the channels that carry the ventricle?

No training. The question is not "what did the objective do" — the R2 ladders already
answer that — but "what does the objective WANT", and that is computable from a forward
pass on a checkpoint you already have.

The setup
---------
``ventricle_size`` is the one factor the contrastive run loses to its recon-only ablation
(gap -0.089, stats -0.114, patch -0.175), and at patch it reads BELOW the random-init
floor (learned -0.126) — active removal, not failure to acquire. ``view_consistency``
ruled out the obvious explanation: the ventricle's render Jacobian has cos(T1, FLAIR) =
0.992, the MOST view-consistent factor measured, so no view-inconsistency story survives.

The hypothesis under test here is about the SHAPE of the evidence rather than its
cross-view behaviour. ``bt_patch_stat`` defaults to "fold" (main_multimodal.py), so the
patch BT term folds patches into the batch: rows are (subject, patch) pairs and each
channel is standardised over that pooled population (losses.py). A channel that only
fires on the handful of patches containing the ventricle has its cross-view correlation
diluted by its occupancy fraction,

    corr  ~  f * A^2 / (f * A^2 + sigma^2)

with f ~ 0.01-0.1 for the ventricle against f ~ 0.6-0.8 for a cortical-shell channel. At
equal per-patch SNR the sparse channel's diagonal is diluted ~10-50x, so BT's on-diagonal
term — which drives diag(c) toward 1 — charges it far more loss. The cheapest way to pay
less is to make the channel respond densely, which destroys the localisation the patch
probe reads.

What is measured, per channel
-----------------------------
1. ``c_ii``       the folded-patch diagonal cross-view correlation, computed by the SAME
                  fold and standardisation the loss uses (``_center_patch_features``, then
                  permute/reshape to (2, B*P, C), then per-column z-scoring). This IS the
                  quantity ``on_diag`` optimises, read off a trained model.
2. ``occupancy``  participation ratio of the channel's across-batch variance over
                  foreground patches: ``(sum v)^2 / (P * sum v^2)``, in (1/P, 1]. 1.0 = the
                  channel varies equally everywhere; 1/P = all of its variation sits in one
                  patch. Computed on the CENTERED features, because across-batch variation
                  at each position is exactly what the loss sees.
3. ``carries``    per-channel CV R^2 of each content factor from that channel's P patch
                  values alone. "Is this a ventricle channel?" — a spatial-profile question,
                  so a scalar correlation with the channel mean will not answer it.

Run it on the ABLATION first. That model demonstrably carries the ventricle at patch
(R2 0.924), so if its ventricle-carrying channels are the low-occupancy, low-``c_ii`` ones,
the objective scores the working representation worst. That is motive and opportunity shown
on a representation that actually works — which a retrain cannot show, because a retrain
only reports the outcome.

Reading it
----------
    H1 SUPPORTED   ventricle_size's weighted c_ii and occupancy are well below the
                   all-channel baseline, and below cortical_thickness / lr_asymmetry
                   (the controls, whose channels should be dense and view-consistent).
                   The per-factor Spearman(carries, c_ii) is negative for the ventricle.
    H1 OUT         the ventricle's channels sit at or above the baseline on both. Then BT
                   has no structural reason to prefer removing them and the deficit needs
                   a different mechanism (style-rejection collateral, the off-diagonal term
                   against the SCM, VQ resolution).

The corollary (``--random-init``) adds an untrained arm built from the same settings. H1
predicts BT training shifts the occupancy distribution UP relative to both the untrained
encoder and the recon-only ablation: the objective should be selecting for dense channels.
If contrastive ~ ablation ~ untrained, that selection did not happen and H1 is weakened
independently of the per-factor table.

What this does NOT establish
----------------------------
That removing the penalty would rescue the ventricle. A channel can be sparse and
penalised for reasons unrelated to why it is absent in the trained model. This narrows the
candidate set cheaply; the counterfactual still needs one confirmatory run with
``bt_patch_weight: 0``.

Usage
-----
    python -m eval.bt_channel_pressure --run ablation=results/synthetic/<recon_only_run>
    python -m eval.bt_channel_pressure \
        --run contrastive=results/synthetic/<bt_run> \
        --run ablation=results/synthetic/<recon_only_run> \
        --random-init --num-samples 256

    # per-channel table for one factor, and the raw matrix
    python -m eval.bt_channel_pressure --run bt=<dir> --per-channel ventricle_size
    python -m eval.bt_channel_pressure --run bt=<dir> --csv channels.csv

A GPU is strongly recommended: this forwards ``2 * num_samples`` volumes at the run's own
resolution.
"""

from __future__ import annotations

import argparse
import csv
import logging
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from eval.identifiability_metrics import cv_probe_r2_multi
from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir
from training.losses import _center_patch_features

logger = logging.getLogger(__name__)

CONTENT_NAMES = [
    "brain_size",
    "ventricle_size",
    "lesion_x",
    "lesion_y",
    "lesion_z",
    "cortical_thickness",
    "temporal_atrophy",
    "lr_asymmetry",
    "sulcal_widening",
]

# Factors whose evidence is a large-support boundary displacement. They are the positive
# controls: whatever the ventricle does, these should come out dense and view-consistent,
# and if they do not the measurement itself is wrong.
CONTROL_FACTORS = ("cortical_thickness", "lr_asymmetry")


def _patch_grid_for(args, level):
    """Resolve the run's patch grid spec to a plain 3-tuple for ``level``."""
    pg = getattr(args, "patch_grid", None)
    if pg is None:
        raise SystemExit("This run has no patch_grid in settings.json — nothing to fold.")
    if len(pg) > 0 and isinstance(pg[0], (list, tuple)):
        pg = pg[level]
    return tuple(pg)


def collect(model, args, ds, num_samples, level, device, batch_size, channels="content"):
    """Forward the test set and return ``(hz, Z, info)``.

    ``hz`` is (2, N, C, P) on CPU — the same tensor the patch contrastive path builds,
    including the foreground-position mask, so downstream statistics are computed on
    exactly the features the loss saw.
    """
    pg = _patch_grid_for(args, level)
    subsets = getattr(args, "subsets", None)
    fg_thresh = float(getattr(args, "patch_foreground_thresh", 0.05))
    use_fg = bool(getattr(args, "patch_foreground_mask", False))

    hz_chunks, z_rows = [], []
    keep_pos = None
    n_total = None
    content_idx = None
    idx_source = "all channels"

    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        v1, v2, m1, zc = [], [], [], []
        for i in range(start, stop):
            x1, x2, lat = ds._inner[i]
            mask = lat["brain_mask"]
            x1, x2 = ds.normalize_views(x1, x2, mask, mask.clone())
            v1.append(x1)
            v2.append(x2)
            m1.append(mask)
            zc.append(lat["z_content"].numpy())

        # [all view-0 ; all view-1] on dim 0, matching train_step's concat convention.
        images = torch.cat([torch.stack(v1), torch.stack(v2)], 0).to(device)
        masks = torch.cat([torch.stack(m1), torch.stack(m1)], 0).to(device)

        with torch.no_grad():
            out = model(
                images,
                return_recon=False,
                pool_only=True,
                n_views=2,
                subsets=subsets,
                patch_grid=getattr(args, "patch_grid"),
                mask=masks,
            )
        if channels == "content" and idx_source == "all channels":
            # BT sees only the content subset, so that is the population to score. out[3] is
            # estimated_content_indices (per subset); fall back loudly rather than silently
            # scoring 48 channels as if they were the 44 the loss operates on.
            try:
                ci = out[3][0]
                content_idx = (ci.cpu() if torch.is_tensor(ci) else torch.as_tensor(list(ci))).long()
                idx_source = "estimated_content_indices"
            except (TypeError, IndexError, KeyError):
                content_idx = None
                idx_source = "all channels (no content indices returned)"
        enc_pooled = out[2][level]
        if enc_pooled.ndim != 3:
            raise SystemExit(
                f"Expected patch-pooled features (2B, C, P) at level {level}, got shape "
                f"{tuple(enc_pooled.shape)}. Is patch_contrastive set for this run?"
            )
        hz_level = enc_pooled.reshape(2, -1, *enc_pooled.shape[1:])

        if use_fg:
            with torch.no_grad():
                frac = F.adaptive_avg_pool3d(masks, pg).flatten(1)
                batch_keep = (frac >= fg_thresh).any(dim=0)
                if not bool(batch_keep.any()):
                    batch_keep = torch.ones_like(batch_keep)
            # The training mask is recomputed per batch; union it across batches so every
            # chunk contributes the same positions and the fold stays rectangular.
            keep_pos = batch_keep if keep_pos is None else (keep_pos | batch_keep)
            n_total = int(batch_keep.numel())

        hz_chunks.append(hz_level.cpu())
        z_rows.extend(zc)

    hz = torch.cat(hz_chunks, dim=1)
    if keep_pos is not None:
        hz = hz[..., keep_pos.cpu()]
        n_kept = int(keep_pos.sum())
    else:
        n_kept = n_total = hz.shape[-1]
    if content_idx is not None:
        hz = hz[:, :, content_idx, :]
    info = {"n_kept": n_kept, "n_total": n_total, "channels": idx_source, "channel_ids": content_idx}
    return hz.float(), np.stack(z_rows), info


def bt_diagonal(hz, center_mode):
    """Per-channel diag(c) under the loss's own fold and standardisation."""
    hz = _center_patch_features(hz, center_mode)
    z = hz.permute(0, 1, 3, 2).reshape(hz.shape[0], -1, hz.shape[2])
    zi, zj = z[0], z[1]
    zi = (zi - zi.mean(dim=0)) / (zi.std(dim=0, unbiased=False) + 1e-6)
    zj = (zj - zj.mean(dim=0)) / (zj.std(dim=0, unbiased=False) + 1e-6)
    c = (zi.T @ zj) / zi.shape[0]
    return c.diagonal().numpy()


def occupancy(hz, center_mode):
    """Participation ratio of each channel's across-batch variance over patches.

    A channel with no across-batch variance anywhere is DEAD, not maximally localised, so
    it returns NaN rather than 0 — otherwise dead channels drag the all-channel baseline
    down and manufacture the very effect this script is testing for.
    """
    hz = _center_patch_features(hz, center_mode)
    v = hz.var(dim=1, unbiased=False).mean(dim=0)  # (C, P), averaged over views
    total = v.sum(dim=1)
    pr = (total**2) / (v.shape[1] * (v**2).sum(dim=1) + 1e-20)
    pr = pr.numpy()
    pr[total.numpy() <= 1e-12 * max(float(total.max()), 1e-12)] = np.nan
    return pr


def channel_factor_r2(hz, Z, seeds, n_splits, probe_dim="auto"):
    """(C, F) CV R^2 of every factor from each channel's patch profile alone.

    ``probe_dim`` mirrors run_dci_compare's rule: at P > N/4 a ridge on the raw patch
    profile is in the p>>n regime where it returns negative R^2 on weak targets — the
    defect that inverted the lesion dims in the 20 Aug investigation. "auto" reduces only
    when that threshold is crossed; 0 disables reduction; an integer forces it.
    """
    from sklearn.decomposition import PCA

    x = hz.mean(dim=0).numpy()  # (N, C, P), view-averaged
    n_samp, n_ch, n_patch = x.shape
    if probe_dim == "auto":
        k = min(64, n_samp // 4) if n_patch > n_samp // 4 else 0
    else:
        k = int(probe_dim)
    k = min(k, n_patch, n_samp - 1) if k else 0

    r2 = np.empty((n_ch, Z.shape[1]))
    for c in range(n_ch):
        xc = x[:, c, :]
        if k:
            xc = PCA(n_components=k, random_state=0).fit_transform(xc)
        r2[c] = cv_probe_r2_multi(xc, Z, n_splits=n_splits, seeds=seeds)["mean"]
    return r2, k


def _spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")

    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        return r

    ra, rb = rank(a[ok]), rank(b[ok])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _weighted(values, weights):
    w = np.clip(weights, 0.0, None)
    return float((values * w).sum() / w.sum()) if w.sum() > 1e-9 else float("nan")


def analyse(label, hz, Z, center_mode, seeds, n_splits, info, probe_dim):
    c_ii = bt_diagonal(hz, center_mode)
    occ = occupancy(hz, center_mode)
    r2, k = channel_factor_r2(hz, Z, seeds, n_splits, probe_dim)

    n_view, n_samp, n_ch, n_patch = hz.shape
    print(f"\n{'=' * 88}\n  MODEL: {label}\n{'=' * 88}")
    print(
        f"  N={n_samp}  channels={n_ch} ({info['channels']})  patches={n_patch}"
        + (f" of {info['n_total']} ({info['n_kept']} foreground)" if info["n_total"] else "")
        + f"  centering={center_mode!r}"
        + (f"  probe PCA->{k}" if k else "  probe on raw patch profile")
    )
    print(
        f"  all-channel baseline:  mean c_ii {np.nanmean(c_ii):+.3f}   "
        f"mean occupancy {np.nanmean(occ):.3f}   dead channels {int(np.isnan(occ).sum())}"
    )

    print(f"\n  {'factor':<20}{'best r2':>9}{'w.c_ii':>9}{'w.occ':>8}{'rho(r2,c_ii)':>14}   top channels")
    print(f"  {'-' * 84}")
    for fi, name in enumerate(CONTENT_NAMES[: Z.shape[1]]):
        col = r2[:, fi]
        top = np.argsort(col)[::-1][:3]
        tops = " ".join(f"c{t:02d}({col[t]:.2f})" for t in top)
        print(
            f"  {name:<20}{col.max():>9.3f}{_weighted(c_ii, col):>9.3f}"
            f"{_weighted(occ, col):>8.3f}{_spearman(col, c_ii):>14.3f}   {tops}"
        )

    q = np.nanpercentile(occ, [10, 50, 90])
    print(f"\n  occupancy over channels:  p10 {q[0]:.3f}   median {q[1]:.3f}   p90 {q[2]:.3f}")
    return {"label": label, "c_ii": c_ii, "occ": occ, "r2": r2}


def _verdict(results):
    print(f"\n{'=' * 88}\n  H1 READOUT\n{'=' * 88}")
    vi = CONTENT_NAMES.index("ventricle_size")
    for res in results:
        col = res["r2"][:, vi]
        wc, wo = _weighted(res["c_ii"], col), _weighted(res["occ"], col)
        ctrl_c = [_weighted(res["c_ii"], res["r2"][:, CONTENT_NAMES.index(f)]) for f in CONTROL_FACTORS]
        ctrl_o = [_weighted(res["occ"], res["r2"][:, CONTENT_NAMES.index(f)]) for f in CONTROL_FACTORS]
        print(
            f"  {res['label']:<16} ventricle channels: c_ii {wc:+.3f} occ {wo:.3f}   |   "
            f"controls: c_ii {np.nanmean(ctrl_c):+.3f} occ {np.nanmean(ctrl_o):.3f}   |   "
            f"all-channel: c_ii {np.nanmean(res['c_ii']):+.3f} occ {np.nanmean(res['occ']):.3f}"
        )
    print(
        "\n  H1 supported if the ventricle row sits well below BOTH the controls and the\n"
        "  all-channel baseline on c_ii and occupancy — most importantly in the ABLATION,\n"
        "  which demonstrably carries the ventricle at patch."
    )
    if len(results) > 1:
        print(f"\n  OCCUPANCY SHIFT (corollary)   {'model':<16}{'mean':>8}{'median':>9}{'p90':>8}")
        for res in results:
            q = np.nanpercentile(res["occ"], [50, 90])
            print(f"  {'':<30}{res['label']:<16}{np.nanmean(res['occ']):>8.3f}{q[0]:>9.3f}{q[1]:>8.3f}")
        print("  H1 predicts the BT arm above the recon-only and untrained arms.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="LABEL=DIR", help="repeatable")
    p.add_argument("--checkpoint", default="vqvae_model.pt")
    p.add_argument("--random-init", action="store_true", help="add an untrained arm from the FIRST run's settings")
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--causal",
        default="match",
        choices=["match", "iid"],
        help="'match' reproduces the training SCM (aggregate ranking); 'iid' decorrelates "
        "the factors and is the honest setting for per-factor attribution. Default 'match' "
        "because c_ii and occupancy are properties of the features, not of the labels — but "
        "the per-channel r2 columns ARE label-dependent, so re-read those under 'iid'.",
    )
    p.add_argument(
        "--channels",
        default="content",
        choices=["content", "all"],
        help="'content' scores only the channels BT operates on (the default, and the "
        "faithful one); 'all' includes the style channels.",
    )
    p.add_argument("--probe-dim", default="auto", help="auto | 0 (off) | integer, for the per-channel probe")
    p.add_argument("--per-channel", default=None, help="also dump the full per-channel table for this factor")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arms = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run expects LABEL=DIR, got {spec!r}")
        label, run_dir = spec.split("=", 1)
        arms.append((label, run_dir, False))
    if args.random_init:
        arms.append(("untrained", arms[0][1], True))

    results = []
    for label, run_dir, rand in arms:
        model, run_args, device = load_model_from_run_dir(
            run_dir, checkpoint=None if rand else args.checkpoint, random_init=rand
        )
        model.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ds = build_synthetic_test_set(run_args, num_samples=args.num_samples, causal=args.causal == "match")
        hz, Z, info = collect(model, run_args, ds, args.num_samples, args.level, device, args.batch_size, args.channels)
        center_mode = getattr(run_args, "patch_center_mode", "none")
        results.append(analyse(label, hz, Z, center_mode, tuple(args.seeds), args.n_splits, info, args.probe_dim))
        del model, hz
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _verdict(results)

    if args.per_channel:
        fi = CONTENT_NAMES.index(args.per_channel)
        for res in results:
            print(f"\n  per-channel, {args.per_channel}, {res['label']}")
            print(f"    {'ch':>4}{'r2':>9}{'c_ii':>9}{'occ':>8}")
            for c in np.argsort(res["r2"][:, fi])[::-1]:
                print(f"    {c:>4}{res['r2'][c, fi]:>9.3f}{res['c_ii'][c]:>9.3f}{res['occ'][c]:>8.3f}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "channel", "c_ii", "occupancy"] + [f"r2_{n}" for n in CONTENT_NAMES])
            for res in results:
                for c in range(len(res["c_ii"])):
                    w.writerow([res["label"], c, res["c_ii"][c], res["occ"][c]] + list(res["r2"][c]))
        print(f"\n  wrote {args.csv}")


if __name__ == "__main__":
    main()
