"""Where a fixed lesion site sits relative to the ventricles and the cortex, on the causal bank's training subjects.

    python out/burden_feasibility_20261102/lesion_site.py [--subjects 200] [--sites 27,22,37 ...] [--search]

For each site (64^3 voxel indices) the lesion goes where bank.py's set_position puts it: the voxel nearest the site
whose full-size sphere fits inside the subject's white matter. Reported per site, over subjects: the lesion centre's
total SD and the share of subjects placed exactly at the site; the gap from the full-size lesion's edge to the nearest
ventricle voxel and to the nearest other non-white-matter voxel (cortex, sulcal clefts, fissure, background), in voxels
(median and 10th-90th percentiles); and the correlation of each gap and of the centre's displacement with every causal
factor, so a site's neighbours are named by the factors that move them. --search scans every voxel admissible in at
least --min-share of the subjects and prints the most stable site at each ventricle gap.
"""

import argparse
import warnings

import bank as B
import numpy as np
from scipy.ndimage import distance_transform_edt

VENTRICLE_RADIUS = 0.35  # CSF inside this undeformed radius is ventricle (largest ventricle .28); outside it, clefts


def structures(n, seed_offset=0):
    """``n`` training subjects of the causal bank: (tissue maps (n, 64, 64, 64), causal factors (n, 7))."""
    args = B.run_args(0.4, causal=True)
    ds = B.T.make_dataset(args, "train", n + seed_offset)
    ds._inner.scm = B.causal_scm(args.seed)
    real = ds._inner.renderer.render_structure
    kept = {}

    def structure(*a, **k):
        tissue, load = real(*a, **k)
        kept["tissue"] = tissue
        return tissue, load

    ds._inner.renderer.render_structure = structure
    tissues, factors = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(seed_offset, seed_offset + n):
            item = ds[i]
            tissues.append(kept["tissue"].cpu().numpy().astype(np.int8))
            z = np.asarray(item["gt_latents"]["z_content"], dtype=np.float64)
            factors.append(z[list(B.CAUSAL_FACTORS)])
    return np.stack(tissues), np.stack(factors), ds._inner.renderer


def subject_maps(tissue, renderer):
    """Per subject: the nearest admissible voxel for every voxel, and the distance (voxels) of every voxel to the
    nearest ventricle voxel and to the nearest other non-white-matter voxel."""
    spacing = 2.0 / (renderer.res - 1)
    reach = renderer.lesion_radius + spacing / 2
    white = tissue == 2
    admissible = distance_transform_edt(white, sampling=spacing) > reach + 1e-7
    _, nearest = distance_transform_edt(~admissible, return_indices=True)
    radius = np.linalg.norm(renderer.coords.cpu().numpy(), axis=-1)
    ventricle = (tissue == 1) & (radius < VENTRICLE_RADIUS)
    other = ~white & ~ventricle
    return nearest.astype(np.int16), distance_transform_edt(~ventricle), distance_transform_edt(~other), admissible


def site_table(site, maps, factors, lesion_vox):
    """The statistics of one site over subjects, as a dict."""
    centres = np.stack([nearest[:, site[0], site[1], site[2]] for nearest, *_ in maps]).astype(float)
    gaps_v = np.array([dv[tuple(c.astype(int))] for c, (_, dv, _, _) in zip(centres, maps)]) - lesion_vox
    gaps_o = np.array([do[tuple(c.astype(int))] for c, (_, _, do, _) in zip(centres, maps)]) - lesion_vox
    moved = np.linalg.norm(centres - np.asarray(site), axis=1)

    def corr(y):
        return {
            name: float(np.corrcoef(y, factors[:, j])[0, 1]) if y.std() > 0 else 0.0
            for j, name in enumerate(B.CAUSAL_FACTORS.values())
        }

    return dict(
        site=list(map(int, site)),
        spread=float(np.sqrt(centres.var(0).sum())),
        exact=float((moved == 0).mean()),
        ventricle_gap=[float(np.percentile(gaps_v, q)) for q in (10, 50, 90)],
        other_gap=[float(np.percentile(gaps_o, q)) for q in (10, 50, 90)],
        corr_ventricle_gap=corr(gaps_v),
        corr_other_gap=corr(gaps_o),
        corr_moved=corr(moved),
    )


def show(row):
    v, o = row["ventricle_gap"], row["other_gap"]
    print(
        f"  site {tuple(row['site'])}: centre SD {row['spread']:.1f} voxels, {row['exact']:.0%} exactly there; "
        f"edge to ventricle {v[1]:.1f} ({v[0]:.1f}-{v[2]:.1f}), to cortex/other {o[1]:.1f} ({o[0]:.1f}-{o[2]:.1f})"
    )
    for key, label in (
        ("corr_ventricle_gap", "ventricle gap"),
        ("corr_other_gap", "cortex gap"),
        ("corr_moved", "displacement"),
    ):
        named = ", ".join(f"{k} {c:+.2f}" for k, c in row[key].items() if abs(c) >= 0.2) or "none at |r| .2"
        print(f"    {label} tracks: {named}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", type=int, default=200)
    p.add_argument("--sites", nargs="*", default=[",".join(map(str, B.FIXED_CENTRE))])
    p.add_argument("--search", action="store_true", help="Scan every voxel admissible in --min-share of the subjects")
    p.add_argument("--min-share", type=float, default=0.5)
    a = p.parse_args()
    tissues, factors, renderer = structures(a.subjects)
    maps = [subject_maps(t, renderer) for t in tissues]
    lesion_vox = renderer.lesion_radius / (2.0 / (renderer.res - 1))
    print(f"{a.subjects} training subjects; full-size lesion radius {lesion_vox:.2f} voxels")
    for site in a.sites:
        show(site_table(tuple(int(v) for v in site.split(",")), maps, factors, lesion_vox))
    if a.search:
        share = np.mean([m[3] for m in maps], axis=0)
        candidates = np.argwhere(share >= a.min_share)
        print(
            f"\n{len(candidates)} voxels admissible in at least {a.min_share:.0%} of the subjects; most stable site per "
            "ventricle gap:"
        )
        rows = [site_table(tuple(c), maps, factors, lesion_vox) for c in candidates]
        best = {}
        for row in rows:
            band = int(np.floor(row["ventricle_gap"][0]))  # 10th-percentile gap, whole voxels
            if band not in best or row["spread"] < best[band]["spread"]:
                best[band] = row
        for band in sorted(best):
            print(f"  10th-percentile ventricle gap >= {band} voxels:")
            show(best[band])


if __name__ == "__main__":
    main()
