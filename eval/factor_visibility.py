"""Is a factor missing because of resolution, because of the readout, or because of the label?

Probes the ground-truth factors straight off the **raw volumes** — no encoder, no training —
through readouts that differ in exactly one property each. Because a linear probe on the raw
image is an upper bound on what any encoder reading the same image can deliver, a factor that
is unrecoverable here is unrecoverable for the model too, and the reason is whichever readout
it fails on:

    gap        4 global scalars (mean/std/max/min per view). Position-blind by construction:
               a feature at the front and the same feature at the back give the same mean.
               This is what ``--eval-pooling gap`` hands the probes.
    patch      the volume average-pooled to ``--patch-grid`` (default 4x5x4, ~80 positions),
               the pooling the patch objectives align on.
    patch_dbl  the same grid under ``--patch-center-mode double``: each sample's mean over
               positions removed, so a spatially constant per-subject code contributes
               nothing and only the subject x location interaction survives.
               ``--patch-center-mode position`` is deliberately absent: it subtracts the
               across-sample mean at each patch, i.e. one constant vector, which the probe's
               StandardScaler undoes exactly. It cannot change ANY linear-probe score, so it
               would only ever reprint the ``patch`` column. It still matters for the
               objectives themselves, whose cosine similarity is not affine-invariant.
    vox8/vox16 average-pooled to 8^3 / 16^3. Finer than the patch grid, and the pair
               separates "needs more spatial detail" from "needs any spatial detail".
    centroid   intensity-weighted centroid of the most extreme voxels in each view (the
               lesion is painted at a fixed intensity distinct from the white matter around
               it, so its own voxels are extremal). An explicit position oracle: if a
               positional factor is not recoverable from this, it is not recoverable at all.

Reading the table:

    high under patch/vox*, ~0 under gap      -> the READOUT is the problem. Global pooling
                                                discards it; more input resolution will not
                                                bring it back.
    high under patch, low under patch_dbl    -> the factor lives in a spatially constant
                                                per-subject code, which that centring mode
                                                deletes on purpose.
    rises from one resolution to the next    -> RESOLUTION is the problem for that factor.
    low everywhere, centroid included        -> the LABEL is the problem: the factor does not
                                                correspond to anything recoverable in the image
                                                at any resolution, through any readout.

Example:
    python -m eval.factor_visibility --res 32 64 128 --num-samples 350 --patch-grid 4 5 4
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.datasets import SyntheticBrainDataset
from eval.dci import CONTENT_FACTOR_NAMES
from eval.identifiability_metrics import cv_probe_r2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=int, nargs="+", default=[32, 64])
    p.add_argument("--num-samples", type=int, default=250)
    p.add_argument("--n-content", type=int, default=9)
    p.add_argument("--n-style", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--quantile", type=float, default=0.99, help="Extremal-voxel cut for the centroid oracle")
    p.add_argument(
        "--patch-grid",
        type=int,
        nargs=3,
        default=[4, 5, 4],
        help="Patch grid (D, H, W) for the patch oracles, matching main_multimodal's "
        "--patch-grid. Default 4 5 4 (~80 patches).",
    )
    p.add_argument(
        "--probe-dim",
        type=int,
        default=64,
        help="PCA-reduce readouts wider than this before probing. Without it a 8192-wide "
        "readout against a few hundred samples overfits and returns large negative R², "
        "which reads as 'absent' when it only means 'unidentifiable probe'. 0 disables.",
    )
    p.add_argument(
        "--synthetic-normalize",
        type=str,
        default="fixed_reference",
        choices=["per_sample", "shared", "fixed_reference"],
    )
    p.add_argument("--synthetic-clean-content", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--self-test", action="store_true", help="Run the torch-free unit checks and exit")
    return p.parse_args()


def extremal_centroid(vol, quantile, high=True):
    """Intensity-weighted centroid of the most extreme voxels, in normalized coordinates.

    ``vol`` is ``(B, D, H, W)``. Returns ``(B, 3)`` in [-1, 1] per axis. Weighting by the
    thresholded intensity rather than a hard mask keeps the estimate stable when the cut
    lands mid-blob, and the per-sample quantile makes it invariant to each view's overall
    intensity level (which style shifts).
    """
    B = vol.shape[0]
    flat = vol.reshape(B, -1)
    thr = torch.quantile(flat, quantile if high else 1.0 - quantile, dim=1, keepdim=True)
    w = (flat - thr).clamp(min=0) if high else (thr - flat).clamp(min=0)

    grids = torch.meshgrid(*[torch.linspace(-1, 1, s, device=vol.device) for s in vol.shape[1:]], indexing="ij")
    coords = torch.stack([g.reshape(-1) for g in grids], dim=1)  # (S, 3)

    total = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return (w @ coords) / total


def readouts(v1, v2, quantile, patch_grid):
    """Every readout for one batch, as {name: (B, D)}. Volumes are (B, 1, D, H, W)."""
    a, b = v1[:, 0], v2[:, 0]

    def stats(x):
        f = x.reshape(x.shape[0], -1)
        return torch.stack([f.mean(1), f.std(1), f.max(1).values, f.min(1).values], dim=1)

    out = {"gap": torch.cat([stats(a), stats(b)], dim=1)}
    for g in (8, 16):
        out[f"vox{g}"] = torch.cat(
            [F.adaptive_avg_pool3d(v, (g, g, g)).flatten(1) for v in (v1, v2)],
            dim=1,
        )
    # Both polarities: the lesion is dark against white matter in T1 and bright in FLAIR,
    # so neither view alone is guaranteed to put it at the extreme the other one does.
    out["centroid"] = torch.cat(
        [
            extremal_centroid(a, quantile, high=True),
            extremal_centroid(a, quantile, high=False),
            extremal_centroid(b, quantile, high=True),
            extremal_centroid(b, quantile, high=False),
        ],
        dim=1,
    )
    # Raw patch grid, kept in (B, views, P) so the centring can run over the whole split
    # rather than per batch -- ``position`` centring subtracts an across-SAMPLE mean, so a
    # per-batch estimate of it would be a different (and much noisier) operator.
    p1, p2 = (F.adaptive_avg_pool3d(v, tuple(patch_grid)).flatten(2)[:, 0] for v in (v1, v2))
    out["_patch_raw"] = torch.stack([p1, p2], dim=1)
    return out


def derive_patch(raw, modes=(("none", "patch"), ("double", "patch_dbl"))):
    """Patch readouts matching ``main_multimodal``'s --patch-grid / --patch-center-mode.

    ``raw`` is ``(N, views, P)``. Centring is delegated to the training code's own
    ``_center_patch_features`` rather than reimplemented, so what this measures is exactly
    what the patch objectives see: ``position`` removes the shared anatomy at each patch,
    ``double`` additionally removes each sample's spatially constant code.
    """
    from training.losses import _center_patch_features

    hz = torch.as_tensor(raw).permute(1, 0, 2).unsqueeze(2)  # (views, N, 1, P)
    out = {}
    for mode, name in modes:
        c = _center_patch_features(hz, mode)
        out[name] = c.permute(1, 0, 2, 3).reshape(hz.shape[1], -1).numpy()
    return out


def collect(ds, batch_size, quantile, patch_grid):
    """Readouts and GT factors for the whole split, accumulating only the readouts.

    The volumes are dropped batch by batch; at res 128 holding them all would cost more
    than the machine has, and nothing downstream needs them.
    """
    acc, gt = {}, []
    for batch in DataLoader(ds, batch_size=batch_size):
        v1, v2 = batch["image"]
        for k, v in readouts(v1, v2, quantile, patch_grid).items():
            acc.setdefault(k, []).append(v)
        gt.append(batch["gt_latents"]["z_content"].numpy())
    return {k: torch.cat(v).numpy() for k, v in acc.items()}, np.concatenate(gt)


def run_resolution(args, res):
    ds = SyntheticBrainDataset(
        mode="val",
        spatial_size=(res, res, res),
        cache=False,
        synthetic_mode="pseudo_mri",
        synthetic_seed=args.seed,
        synthetic_num_samples=args.num_samples,
        synthetic_n_content=args.n_content,
        synthetic_n_style=args.n_style,
        synthetic_normalize=args.synthetic_normalize,
        synthetic_clean_content=args.synthetic_clean_content,
    )
    reps, gt = collect(ds, args.batch_size, args.quantile, args.patch_grid)
    reps.update(derive_patch(reps.pop("_patch_raw")))
    names = CONTENT_FACTOR_NAMES[: gt.shape[1]]
    order = ["gap", "patch", "patch_dbl", "vox8", "vox16", "centroid"]

    raw_width = {k: v.shape[1] for k, v in reps.items()}
    if args.probe_dim > 0:
        from sklearn.decomposition import PCA

        for k, v in reps.items():
            n_comp = min(args.probe_dim, v.shape[0], v.shape[1])
            if v.shape[1] > n_comp:
                reps[k] = PCA(n_components=n_comp, random_state=0).fit_transform(v)

    # The lesion is a fixed-radius sphere in [-1,1]^3 coordinates, so its voxel count
    # scales with res^3 -- the whole reason a factor can be invisible at one resolution
    # and plain at another.
    lesion_r_vox = 0.1 / 2.0 * res
    print(f"\n=== res {res} (n={args.num_samples}, normalize={args.synthetic_normalize}) ===", flush=True)
    print(
        f"  lesion sphere: radius {lesion_r_vox:.1f} vox -> ~{4/3*np.pi*lesion_r_vox**3:.0f} voxels",
        flush=True,
    )
    print(f"  {'factor':<19s}" + "".join(f"{k:>10s}" for k in order), flush=True)
    scores = {}
    for j, nm in enumerate(names):
        row = {k: cv_probe_r2(reps[k], gt[:, j])["mean"] for k in order}
        scores[nm] = row
        print(f"  {nm:<19s}" + "".join(f"{row[k]:>10.3f}" for k in order), flush=True)
    print(f"  {'(raw width)':<19s}" + "".join(f"{raw_width[k]:>10d}" for k in order), flush=True)
    print(f"  {'(probe width)':<19s}" + "".join(f"{reps[k].shape[1]:>10d}" for k in order), flush=True)
    return scores


SPATIAL = ("patch", "patch_dbl", "vox8", "vox16", "centroid")


def verdict(per_res):
    """Name the binding constraint per factor from the pattern across readouts/resolutions."""
    resolutions = sorted(per_res)
    hi, lo = resolutions[-1], resolutions[0]
    print("\n=== verdict (R² > 0.15 counts as recovered) ===", flush=True)

    # "Not recoverable" is only meaningful once the probes are shown to recover SOMETHING.
    # Underpowered probes drive every factor to chance, which would otherwise be reported
    # as nine separate label problems.
    best = max(max(v[k] for k in SPATIAL) for v in per_res[hi].values())
    if best < 0.30:
        print(
            f"  INCONCLUSIVE — best spatial recovery over all factors is only {best:.2f}. "
            "The probes are underpowered, so nothing here distinguishes an absent factor "
            "from an unidentifiable probe. Raise --num-samples (want several x --probe-dim).",
            flush=True,
        )
        return

    for nm in per_res[hi]:
        top = per_res[hi][nm]
        spatial = max(top[k] for k in SPATIAL)
        gained = spatial - max(per_res[lo][nm][k] for k in SPATIAL)
        if spatial < 0.15:
            msg = "LABEL — not recoverable by any readout at any resolution tested"
        elif top["gap"] < 0.15:
            msg = f"READOUT — spatial {spatial:.2f} vs gap {top['gap']:.2f}; GAP discards it"
        elif gained > 0.10:
            msg = f"RESOLUTION — spatial recovery improved {gained:+.2f} from res {lo} to {hi}"
        else:
            msg = f"fine — recovered at gap {top['gap']:.2f}"
        print(f"  {nm:<20s} {msg}", flush=True)


def _self_test():
    """Centroid oracle must land on a single blob's true coordinate, and survive a flat field."""
    res, idx = 16, (3, 8, 13)
    axis = torch.linspace(-1, 1, res)
    vol = torch.zeros(len(idx), res, res, res)
    for i, c in enumerate(idx):
        vol[i, c, 8, 8] = 1.0
    got = extremal_centroid(vol, 0.99, high=True)[:, 0]
    want = axis[list(idx)]
    assert torch.allclose(got, want, atol=1e-4), f"got {got.tolist()}, want {want.tolist()}"

    # Dark blob in a bright field, via the low polarity.
    dark = torch.ones(1, res, res, res)
    dark[0, 13, 8, 8] = 0.0
    got_low = extremal_centroid(dark, 0.99, high=False)[0, 0]
    assert abs(got_low.item() - axis[13].item()) < 1e-4, got_low

    flat = torch.ones(2, 8, 8, 8)
    assert torch.isfinite(extremal_centroid(flat, 0.99, high=True)).all(), "uniform volume must not produce NaN"
    print(f"self-test OK (centroids {[round(v, 4) for v in got.tolist()]} == {[round(v, 4) for v in want.tolist()]})")


def main():
    args = parse_args()
    if args.self_test:
        _self_test()
        return
    per_res = {r: run_resolution(args, r) for r in sorted(args.res)}
    if len(per_res) >= 1:
        verdict(per_res)


if __name__ == "__main__":
    main()
