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

Both views are scored apart, because the hypothesis under test is asymmetric.  On the
pseudo-MRI LUT the ventricle/WM edge is the STRONGEST internal edge in T1 (0.70, against
WM/GM 0.30) and the WEAKEST in FLAIR (0.30, against WM/GM 0.40) -- the salience rank
inverts.  A shared content code is then limited by whichever view carries the factor
least well, which a pooled both-views probe cannot see.

Reading it
----------
    drop pre_norm -> post_norm      the norm removes it.  Per-voxel channel norm deletes
                                    magnitude and contrast DEPTH is a magnitude, so the
                                    fix is architectural (--norm-type, --split-encoder-norm).
    drop post_norm -> encoder_out   same family, inside the residual stack.
    one view low at EVERY stage     that encoder never extracted it; the norms are
                                    innocent and the fix is the objective (projection
                                    head, weaker invariance).
    both views fine at encoder_out  it survives the encoder; whatever loses it is
                                    downstream -- the content codebook quantizes by
                                    squared euclidean distance (`models/vqvae.py:491`)
                                    and is shared across views, so a per-view scale gap
                                    sends the two views to different entries.

Nothing is retrained.  Only the tap point changes.

Factors are drawn i.i.d. (`causal=False`), never from the run's SCM.  Under a random graph
ventricle_size and brain_size correlate ~0.8, so an SCM-matched probe reads brain_size in
disguise -- per-factor attribution requires i.i.d.  See `build_synthetic_test_set`.

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
from eval.run_dci_compare import _auto_probe_dim

logger = logging.getLogger(__name__)

STAGES = ("pre_norm", "post_norm", "encoder_out", "codebook_in")
VIEWS = ("v0", "v1")
POOLINGS = ("gap", "patch")
VIEW_LABEL = {"v0": "T1", "v1": "FLAIR"}


# --------------------------------------------------------------------------- #
# Scoring and rendering.  Pure numpy so --self-test needs no torch build.
# --------------------------------------------------------------------------- #
def reduce_block(X, seed=0):
    """PCA-reduce a p>>n block, by `run_dci_compare._auto_probe_dim`'s rule.

    Unreduced, a ridge probe on the patch block returns a NEGATIVE R^2 on weak targets,
    which is exactly the regime a ventricle investigation lives in.  The rule is imported
    rather than re-derived so the two scripts cannot drift.
    """
    from sklearn.decomposition import PCA

    width = _auto_probe_dim(X.shape[0], X.shape[1])
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
    """Every factor at one stage -- the positive controls that say the probe works."""
    views = _present_views(scores)
    if not any((stage, v, pooling) in scores for v in views):
        return
    print(f"\nAll factors at {stage}, {pooling} pooling  (context: the survivors should NOT split by view)")
    head = f"  {'factor':<18}" + "".join(f"{VIEW_LABEL[v]:>10}" for v in views) + f"{'v1/v0':>10}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for j, name in enumerate(names):
        vals = []
        for v in views:
            key = (stage, v, pooling)
            if key not in scores:
                vals.append(float("nan"))
                continue
            val = float(scores[key][j])
            if floor is not None and key in floor:
                val -= float(floor[key][j])
            vals.append(val)
        ratio = vals[1] / vals[0] if len(vals) == 2 and np.isfinite(vals[0]) and abs(vals[0]) > 1e-6 else float("nan")
        print(f"  {name:<18}" + "".join(f"{v:>10.3f}" for v in vals) + f"{ratio:>10.2f}")


def render_rms(rms):
    """Feature scale per stage per view.  A per-view gap here is what the shared,
    squared-euclidean codebook cannot absorb."""
    if not rms:
        return
    views = [v for v in VIEWS if any((s, v) in rms for s in STAGES)]
    print("\nContent-block feature RMS  (a per-view gap is what the shared L2 codebook sees)")
    head = f"  {'stage':<14}" + "".join(f"{VIEW_LABEL[v]:>12}" for v in views) + f"{'v1/v0':>10}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for stage in STAGES:
        if not any((stage, v) in rms for v in views):
            continue
        vals = [rms.get((stage, v), float("nan")) for v in views]
        ratio = vals[1] / vals[0] if len(vals) == 2 and np.isfinite(vals[0]) and abs(vals[0]) > 1e-9 else float("nan")
        print(f"  {stage:<14}" + "".join(f"{v:>12.3f}" for v in vals) + f"{ratio:>10.2f}")


def verdict(ladder, factor):
    """Name the stage that loses the factor.  `ladder` is render_focus's return."""

    def g(stage, view):
        v = ladder.get((stage, view, "gap"))
        return float("nan") if v is None else v

    def worst_drop(a, b):
        drops = [g(a, v) - g(b, v) for v in VIEWS if np.isfinite(g(a, v)) and np.isfinite(g(b, v))]
        return max(drops) if drops else float("nan")

    norm_cost = worst_drop("pre_norm", "post_norm")
    res_cost = worst_drop("post_norm", "encoder_out")
    cb_cost = worst_drop("encoder_out", "codebook_in")
    out0, out1 = g("encoder_out", "v0"), g("encoder_out", "v1")
    pre0, pre1 = g("pre_norm", "v0"), g("pre_norm", "v1")

    print(f"\nverdict for {factor}  (gap pooling; a move under {NOISE_FLOOR} is not reportable)")
    if np.isfinite(norm_cost) and norm_cost > NOISE_FLOOR:
        print(f"  The encoder's FINAL NORM removes it: pre -> post drops {norm_cost:.3f}.")
        print("  Per-voxel channel norm deletes magnitude, and contrast depth IS a magnitude.")
        print("  Next: --norm-type group (or none) at the encoder, or --split-encoder-norm.")
    elif np.isfinite(res_cost) and res_cost > NOISE_FLOOR:
        print(f"  The RESIDUAL STACK removes it: post_norm -> encoder_out drops {res_cost:.3f}.")
        print("  Same family as above; the norms inside ResidualStack are the suspects.")
    elif np.isfinite(out0) and np.isfinite(out1) and max(out0, out1) > NOISE_FLOOR and min(out0, out1) < NOISE_FLOOR:
        lo = VIEW_LABEL["v1" if out1 < out0 else "v0"]
        print(
            f"  Only one view carries it out of the encoder ({VIEW_LABEL['v0']} {out0:.3f} vs {VIEW_LABEL['v1']} {out1:.3f})."
        )
        print(f"  The norms are innocent -- the {lo} encoder never extracted it, which is the")
        print("  salience-inversion prediction. Next: --contrastive-proj-dim, so the invariance")
        print("  constraint stops landing on the features the probes read.")
    elif np.isfinite(out0) and np.isfinite(out1) and min(out0, out1) > NOISE_FLOOR:
        extra = f" (encoder_out -> codebook_in drops {cb_cost:.3f})" if np.isfinite(cb_cost) else ""
        print(f"  BOTH views carry it out of the encoder ({out0:.3f} / {out1:.3f}){extra}.")
        print("  Nothing in the encoder loses it, so the loss is downstream: the shared content")
        print("  codebook quantizes by squared euclidean distance, so a per-view scale gap sends")
        print("  the views to different entries. Next: --separate-content-codebooks as the test.")
    elif np.isfinite(pre0) and np.isfinite(pre1) and max(pre0, pre1) < NOISE_FLOOR:
        print(f"  Neither view carries it even at pre_norm ({pre0:.3f} / {pre1:.3f}).")
        print("  It is lost upstream of everything measured here -- check the generator's SNR")
        print("  in the WEAK view (generator_defects scores render(...)[0], i.e. T1 only).")
    else:
        print("  No stage loses more than the noise floor and no view is clearly starved.")
        print("  Re-run with more --num-samples before reading anything into this.")


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
    sq_sum = {(s, v): [0.0, 0] for s in STAGES for v in VIEWS}
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
                for view, t in (("v0", a), ("v1", b)):
                    if t is None:
                        continue
                    t = t[:, :n_content].float()
                    sq_sum[(stage, view)][0] += float(t.pow(2).sum().item())
                    sq_sum[(stage, view)][1] += int(t.numel())
                    feats[(stage, view, "gap")].append(t.mean(dim=[2, 3, 4]).cpu().numpy())
                    feats[(stage, view, "patch")].append(F.adaptive_avg_pool3d(t, patch_grid).flatten(1).cpu().numpy())
            gts.append(batch["gt_latents"]["z_content"].numpy())

    for h in handles:
        h.remove()

    feats = {k: np.concatenate(v, 0) for k, v in feats.items() if v}
    rms = {k: float(np.sqrt(s / n)) for k, (s, n) in sq_sum.items() if n}
    return feats, rms, np.concatenate(gts, 0)


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
    feats, rms, gt = _collect(model, inner, loader, device, args.level, args.patch_grid, n_content)
    return {k: reduce_block(X) for k, X in feats.items()}, rms, gt


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

    scores = score_ladder(feats, gt)
    ladder = render_focus(scores, names, names[focus])
    render_all_factors(scores, names, floor=None)
    render_rms({("pre_norm", "v0"): 1.0, ("pre_norm", "v1"): 0.43})
    verdict(ladder, names[focus])

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

    # causal=False, deliberately and explicitly: per-factor attribution needs i.i.d.
    # factors, and under the run's SCM ventricle_size correlates with brain_size at ~0.8.
    ds = build_synthetic_test_set(run_args, args.num_samples, causal=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    feats, rms, gt = _ladder(model, inner, loader, device, args, n_content)
    names = CONTENT_FACTOR_NAMES[: gt.shape[1]]
    if args.factor not in names:
        ap.error(f"--factor {args.factor!r} not in {names}")
    scores = score_ladder(feats, gt, seeds=seeds)

    floor_scores = None
    if args.floor:
        logger.info("scoring the untrained twin (seed %d) on the SAME rows ...", args.floor_seed)
        f_model, f_inner, _, f_device = _load(args.run_dir, args.checkpoint, random_init=True, seed=args.floor_seed)
        f_feats, _, f_gt = _ladder(f_model, f_inner, loader, f_device, args, n_content)
        floor_scores = score_ladder(f_feats, f_gt, seeds=seeds)

    print(f"\nrun: {args.run_dir} | level {args.level} | content channels {n_content} | N={gt.shape[0]}")
    print("factors drawn i.i.d. (causal=False) so per-factor attribution is unambiguous")
    ladder = render_focus(scores, names, args.factor, floor=floor_scores)
    render_all_factors(scores, names, floor=floor_scores)
    render_rms(rms)
    verdict(ladder, args.factor)

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
