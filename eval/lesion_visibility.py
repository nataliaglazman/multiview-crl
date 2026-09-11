#!/usr/bin/env python
"""Is the lesion visible in the IMAGE, and how does that depend on resolution?

    python -m eval.lesion_visibility --resolutions 32 64 96 128 --figure lesion_res.png

Every lesion measurement in this project so far has gone through an encoder, so a low
score is ambiguous: the information may be absent from the image, or present and
discarded. This measures the image alone -- no checkpoint, no training, no fitted
probe for the primary numbers -- and sweeps resolution, because the lesion is a fixed
FRACTION of the volume (radius 0.1 in normalised coordinates) while the generator's
final blur is a fixed 3 VOXELS. Those two scale differently, so visibility is
resolution-dependent by construction and the sweep separates the two.

Reported per resolution and view:

* geometry -- lesion radius and voxel count, and the same at encoder level 0 (which
  downsamples by 2), where a radius near 1 cell means the lesion is a single feature.
* contrast -- peak and mean lesion-on minus lesion-off inside the lesion, in units of
  the view's own foreground sigma, so T1 and FLAIR and every resolution are comparable.
* detection with a FIXED detector -- ``lesion_reconstruction.locate``, polarity-aware
  (FLAIR bright, T1 dark), never sees the target. Median error in voxels and the
  fraction within one lesion radius. This is the number that says "visible" or not.
* an oracle bound -- the same detector on the lesion-on minus lesion-off difference,
  which is the best any image-space method could do.
* a linear ceiling -- ridge from voxels pooled to a common grid (the same grid at every
  resolution, so p is not confounded with res) to the lesion coordinates, against a
  shuffled-label null.

Detection is fitting-free, so it needs no train/test split and cannot overfit; read it
first. The ridge row is the only fitted number and carries its own null.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# The lesion sits in white matter and renders with opposite contrast in the two views
# (eval/synthetic_dataset.py:448-452): T1 lesion 0.4 against WM 0.8, FLAIR 1.0 against
# WM 0.4. So a bright-blob detector is right for FLAIR and wrong for T1; T1 is searched
# on the negated image. Getting this backwards reports T1 as invisible whatever the data.
VIEW_POLARITY = {"T1": -1.0, "FLAIR": +1.0}


def znorm(volume, mask):
    """The dataset's default per-view normalisation: z-score within the brain mask."""
    values = volume[mask]
    std = float(values.std())
    return (volume - float(values.mean())) / (std if std > 1e-6 else 1.0), std


def render_pair(dataset, idx):
    """Lesion-on and lesion-off for one subject, identical latents, seeds and affine."""
    import torch

    renderer = dataset.renderer
    _v1, _v2, lat = dataset[idx]
    tissue, lesion = renderer.render_structure(
        lat["z_content"], lat["z_deformation"], lat["z_fissure"], "cpu", clean=dataset.clean_content
    )
    mask = np.asarray(lat["brain_mask"]).squeeze() > 0
    seed = dataset.sample_seed_for(idx)
    out = {}
    with torch.inference_mode():
        for view, modality in enumerate(("T1", "FLAIR")):
            style = lat[f"z_style_v{view + 1}"]
            on = renderer.render_modality(tissue, lesion, style, modality, seed * 2 + view, "cpu")
            off = renderer.render_modality(tissue, torch.zeros_like(lesion), style, modality, seed * 2 + view, "cpu")
            on = np.asarray(on).squeeze()
            off = np.asarray(off).squeeze()
            # Freeze the ON normalisation and apply it to OFF too, so removing the lesion
            # cannot shift the whole image through the normaliser and fake a difference.
            on_z, sigma = znorm(on, mask)
            off_z = (off - float(on[mask].mean())) / (sigma if sigma > 1e-6 else 1.0)
            out[modality] = {"on": on_z, "off": off_z, "sigma": sigma}
    support = np.asarray(lesion).squeeze() > 0
    # Tissue labels are [bg, CSF, WM, GM, fissure]; lesions are confined to WM
    # (synthetic_dataset.py:430), so WM is the region a lesion could possibly be in.
    white_matter = np.asarray(tissue).squeeze() == 2
    return out, mask, support, white_matter


def locate_within(volume, radius_vox, search):
    """``locate``'s bright-blob score, with the argmax restricted to ``search``.

    The restriction must happen at the argmax, not in the filter input. Masking the
    volume first (setting outside-region voxels to a large negative) smears that edge
    through both Gaussians, and the difference-of-Gaussians then peaks on the region
    BOUNDARY rather than on any blob inside it -- which reports 0% hits even for a
    lesion the blind detector finds every time.
    """
    from scipy.ndimage import gaussian_filter

    a = np.asarray(volume, dtype=np.float64).squeeze()
    sigma = max(0.5, radius_vox / 2)
    score = gaussian_filter(a, sigma) - gaussian_filter(a, max(1.0, radius_vox * 1.5))
    if not search.any():
        return np.full(3, np.nan)
    inside = score[search]
    if float(inside.max()) - float(inside.min()) <= 1e-8:
        return np.full(3, np.nan)
    flat = np.where(search.reshape(-1), score.reshape(-1), -np.inf)
    return np.array(np.unravel_index(int(flat.argmax()), score.shape), dtype=float)


def measure_subject(views, mask, support, white_matter, radius_vox):
    """Contrast and a ladder of three localisers, from fully blind to fully oracular."""
    from eval.lesion_reconstruction import locate

    truth = np.argwhere(support).mean(0) if support.any() else np.full(3, np.nan)
    row = {"lesion_voxels": int(support.sum())}

    def error(found):
        return float(np.linalg.norm(found - truth)) if np.isfinite(found).all() else np.nan

    for modality, data in views.items():
        delta = data["on"] - data["off"]
        inside = np.abs(delta[support]) if support.any() else np.array([0.0])
        row[f"{modality}_peak_sigma"] = float(inside.max())
        row[f"{modality}_mean_sigma"] = float(inside.mean())
        polarity = VIEW_POLARITY[modality]
        # 1. Blind: search the whole brain. Fails if the lesion is not the most extreme
        #    blob in the image -- in T1 the ventricles and sulci are darker than it is.
        row[f"{modality}_image_error"] = error(locate(data["on"] * polarity * mask, radius_vox))
        # 2. Restricted to white matter, where a lesion could be. Semi-oracle: it uses the
        #    tissue map, so it answers "is the lesion distinctive AMONG WHITE MATTER", which
        #    is the question a trained encoder with anatomical context could answer.
        row[f"{modality}_wm_error"] = error(locate_within(data["on"] * polarity * mask, radius_vox, white_matter))
        # 3. Oracle: the counterfactual difference. Nothing in image space beats this, so
        #    it bounds the detector itself and proves whether the signal is present at all.
        row[f"{modality}_oracle_error"] = error(locate(delta * mask, radius_vox, response=True))
    return row, truth


def pooled_voxels(volume, grid):
    """Mean-pool a volume to (grid, grid, grid) so p is the same at every resolution."""
    import torch
    import torch.nn.functional as F

    tensor = torch.as_tensor(np.asarray(volume), dtype=torch.float32).reshape(1, 1, *volume.shape)
    with torch.inference_mode():
        return F.adaptive_avg_pool3d(tensor, (grid,) * 3).flatten().numpy()


def ridge_ceiling(features, targets, seed=0):
    """Held-out R^2 for lesion coordinates from pooled voxels, plus a shuffled null."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    n = len(features)
    if n < 40:
        return None
    order = np.random.default_rng(seed).permutation(n)
    train, test = order[: int(0.7 * n)], order[int(0.7 * n) :]
    scaler = StandardScaler().fit(features[train])
    x_train, x_test = scaler.transform(features[train]), scaler.transform(features[test])
    out = {}
    for label, y in (("observed", targets), ("shuffled", targets[np.random.default_rng(seed + 7).permutation(n)])):
        model = RidgeCV(alphas=np.logspace(-2, 6, 17)).fit(x_train, y[train])
        prediction = model.predict(x_test)
        truth = y[test]
        ss = ((truth - truth.mean(0)) ** 2).sum(0)
        r2 = 1 - np.divide(((truth - prediction) ** 2).sum(0), ss, out=np.full(3, np.nan), where=ss > 1e-12)
        out[label] = float(np.nanmean(r2))
    return out


def sweep(resolutions, num_samples, settings, pool_grid=16, seed=0):
    from eval.synthetic_dataset import Synthetic3DDisentanglementDataset

    results = []
    for res in resolutions:
        dataset = Synthetic3DDisentanglementDataset(
            num_samples=num_samples, res=res, mode="pseudo_mri", seed=seed, **settings
        )
        radius_vox = dataset.renderer.lesion_radius * (res - 1) / 2
        rows, truths, pooled = [], [], {"T1": [], "FLAIR": []}
        for idx in range(num_samples):
            views, mask, support, white_matter = render_pair(dataset, idx)
            row, truth = measure_subject(views, mask, support, white_matter, radius_vox)
            rows.append(row)
            truths.append(truth)
            for modality, data in views.items():
                pooled[modality].append(pooled_voxels(data["on"] * mask, pool_grid))
            if (idx + 1) % 16 == 0:
                logger.info("res %d: %d/%d subjects", res, idx + 1, num_samples)
        truths = np.asarray(truths)
        entry = {
            "resolution": res,
            "radius_vox": radius_vox,
            "radius_feature_cells_level0": radius_vox / 2.0,
            "lesion_voxels_median": float(np.median([r["lesion_voxels"] for r in rows])),
            "volume_fraction": float(np.median([r["lesion_voxels"] for r in rows]) / res**3),
            "views": {},
        }
        for modality in ("T1", "FLAIR"):
            errors = np.array([r[f"{modality}_image_error"] for r in rows], dtype=float)
            wm = np.array([r[f"{modality}_wm_error"] for r in rows], dtype=float)
            oracle = np.array([r[f"{modality}_oracle_error"] for r in rows], dtype=float)
            entry["views"][modality] = {
                "peak_sigma_median": float(np.median([r[f"{modality}_peak_sigma"] for r in rows])),
                "mean_sigma_median": float(np.median([r[f"{modality}_mean_sigma"] for r in rows])),
                "image_error_median": float(np.nanmedian(errors)),
                "image_hit_rate": float(np.nanmean(errors <= radius_vox)),
                "wm_error_median": float(np.nanmedian(wm)),
                "wm_hit_rate": float(np.nanmean(wm <= radius_vox)),
                "oracle_error_median": float(np.nanmedian(oracle)),
                "oracle_hit_rate": float(np.nanmean(oracle <= radius_vox)),
                "ridge": ridge_ceiling(np.asarray(pooled[modality]), truths, seed),
            }
        results.append(entry)
    return results


def print_report(results, pool_grid):
    print("\n" + "=" * 92)
    print("LESION VISIBILITY IN THE IMAGE   no encoder, no checkpoint")
    print("=" * 92)
    print("\n  GEOMETRY")
    print("   res   radius vox   voxels   % of volume   radius in level-0 feature cells")
    for e in results:
        print(
            f"  {e['resolution']:4d}   {e['radius_vox']:10.2f}   {e['lesion_voxels_median']:6.0f}"
            f"   {100 * e['volume_fraction']:10.3f}%   {e['radius_feature_cells_level0']:29.2f}"
        )

    print("\n  CONTRAST inside the lesion, in units of the view's own foreground sigma")
    print("   res        T1 peak    T1 mean     FLAIR peak   FLAIR mean")
    for e in results:
        t, f = e["views"]["T1"], e["views"]["FLAIR"]
        print(
            f"  {e['resolution']:4d}   {t['peak_sigma_median']:12.3f}{t['mean_sigma_median']:11.3f}"
            f"{f['peak_sigma_median']:15.3f}{f['mean_sigma_median']:13.3f}"
        )

    print("\n  LOCALISER LADDER   hit rate = found within one lesion radius, fitting-free")
    print("    blind = whole brain | in-WM = search restricted to white matter (uses tissue map)")
    print("    oracle = given the lesion-on minus lesion-off difference")
    print("                    T1                          FLAIR")
    print("   res    blind    in-WM   oracle      blind    in-WM   oracle")
    for e in results:
        t, f = e["views"]["T1"], e["views"]["FLAIR"]
        print(
            f"  {e['resolution']:4d}  {100 * t['image_hit_rate']:6.0f}%{100 * t['wm_hit_rate']:8.0f}%"
            f"{100 * t['oracle_hit_rate']:8.0f}%     {100 * f['image_hit_rate']:6.0f}%"
            f"{100 * f['wm_hit_rate']:8.0f}%{100 * f['oracle_hit_rate']:8.0f}%"
        )

    print(f"\n  LINEAR CEILING from voxels pooled to {pool_grid}^3, held-out R^2 for lesion x/y/z")
    print("   res        T1 R2   T1 null       FLAIR R2   FLAIR null")
    for e in results:
        t, f = e["views"]["T1"]["ridge"], e["views"]["FLAIR"]["ridge"]
        if t is None or f is None:
            print(f"  {e['resolution']:4d}        n/a (need >= 40 subjects)")
            continue
        print(
            f"  {e['resolution']:4d}   {t['observed']:+10.3f}{t['shuffled']:+10.3f}"
            f"{f['observed']:+15.3f}{f['shuffled']:+12.3f}"
        )

    print("\n  Read the fixed detector first: it fits nothing and cannot overfit. A high hit rate")
    print("  means the lesion IS in the image at that resolution, so any encoder that misses it")
    print("  is discarding available information. A low hit rate at 64 that rises with resolution")
    print("  means the lesion is genuinely too faint or too small at 64, and no objective or")
    print("  architecture change would recover it.")
    print("  The oracle column is the same detector given the counterfactual difference; a low")
    print("  image hit rate beside a high oracle one means the lesion is present but not")
    print("  separable from anatomy without knowing the lesion-free image.")


def figure(results, settings, path, resolutions, seed=0, subject=0):
    """Slices through the lesion at each resolution: T1 on, off, difference, and FLAIR."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from eval.synthetic_dataset import Synthetic3DDisentanglementDataset

    columns = ["T1 with lesion", "T1 lesion-free", "T1 difference", "FLAIR with lesion", "FLAIR difference"]
    fig, axes = plt.subplots(len(resolutions), 5, figsize=(16, 3.2 * len(resolutions)), squeeze=False)
    for row, res in enumerate(resolutions):
        dataset = Synthetic3DDisentanglementDataset(
            num_samples=max(subject + 1, 4), res=res, mode="pseudo_mri", seed=seed, **settings
        )
        views, mask, support, _wm = render_pair(dataset, subject)
        if not support.any():
            continue
        z = int(round(np.argwhere(support).mean(0)[2]))
        radius_vox = dataset.renderer.lesion_radius * (res - 1) / 2
        panels = [
            (views["T1"]["on"], "gray", None),
            (views["T1"]["off"], "gray", None),
            (views["T1"]["on"] - views["T1"]["off"], "coolwarm", 1.5),
            (views["FLAIR"]["on"], "gray", None),
            (views["FLAIR"]["on"] - views["FLAIR"]["off"], "coolwarm", 1.5),
        ]
        for col, (volume, cmap, limit) in enumerate(panels):
            ax = axes[row][col]
            slice_ = (volume * mask)[:, :, z].T
            kwargs = {"vmin": -limit, "vmax": limit} if limit else {"vmin": -2.5, "vmax": 2.5}
            ax.imshow(slice_, cmap=cmap, origin="lower", **kwargs)
            centre = np.argwhere(support).mean(0)
            ax.add_patch(plt.Circle((centre[0], centre[1]), radius_vox, fill=False, color="lime", lw=1.2))
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(columns[col], fontsize=11)
            if col == 0:
                entry = next(e for e in results if e["resolution"] == res) if results else None
                label = f"res {res}"
                if entry:
                    label += f"\nr={entry['radius_vox']:.1f} vox\n{entry['lesion_voxels_median']:.0f} voxels"
                ax.set_ylabel(label, fontsize=10)
    fig.suptitle(
        "Same subject, same lesion, rendered at each resolution. Green circle = true lesion extent.\n"
        "Difference panels are lesion-on minus lesion-off in sigma units (red positive, blue negative).",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    print(f"\nWrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resolutions", type=int, nargs="+", default=[32, 64, 96, 128])
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--pool-grid", type=int, default=16, help="Common grid for the ridge ceiling")
    p.add_argument("--lesion-radius", type=float, default=0.1)
    p.add_argument("--n-content", type=int, default=9)
    p.add_argument("--clean-content", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--identifiable-ventricle", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--figure", default=None, help="Write the resolution comparison figure here")
    p.add_argument("--figure-only", action="store_true", help="Skip the sweep; just draw the figure")
    p.add_argument("--out", default=None, help="Write the measurements as JSON")
    cli = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = {
        "n_content": cli.n_content,
        "clean_content": cli.clean_content,
        "lesion_radius": cli.lesion_radius,
        "identifiable_ventricle": cli.identifiable_ventricle,
        "causal": False,
        "hierarchical_content": False,
    }
    results = []
    if not cli.figure_only:
        results = sweep(cli.resolutions, cli.num_samples, settings, cli.pool_grid, cli.seed)
        print_report(results, cli.pool_grid)
        if cli.out:
            Path(cli.out).write_text(json.dumps({"settings": settings, "results": results}, indent=2) + "\n")
            print(f"Wrote {cli.out}")
    if cli.figure:
        figure(results, settings, cli.figure, cli.resolutions, cli.seed)


if __name__ == "__main__":
    main()
