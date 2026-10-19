"""Do the lesions a reconstruction loses share a location? Reads a lesion_reconstruction audit.

    python -m eval.lesion.lesion_miss_location \\
        --audit-dir results/synthetic/RUN/lesion_reconstruction_iid

``eval.lesion.lesion_reconstruction`` records, per subject, the rendered lesion centre and where
an image-only FLAIR blob detector fires on the input and on the reconstruction. This rebuilds the
same subjects from the audit's saved settings, checks every lesion centre against the audit, and
re-renders their tissue maps. It then compares the lesions the detector still finds in the
reconstruction ("hits") with those it finds in the input but loses in the reconstruction ("new
misses"): depth inside white matter, distance to grey matter, to CSF and to the midline fissure,
distance from the brain's centre, and the tissue under the reconstruction detector's peak.
Lesions the detector already misses on the input say nothing about the model and are listed
separately. No model is loaded.
"""

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np

TISSUE = {0: "background", 1: "CSF", 2: "white matter", 3: "grey matter", 4: "midline fissure"}
FEATURES = {
    "wm_depth": "depth inside white matter (vox)",
    "gm_distance": "distance to grey matter (vox)",
    "csf_distance": "distance to CSF (vox)",
    "fissure_distance": "distance to midline fissure (vox)",
    "centre_distance": "distance from brain centre (vox)",
    "flair_response_gain": "FLAIR response gain",
}


def distance_to(tissue, labels):
    """Euclidean distance (voxels) from every voxel to the nearest voxel with one of ``labels``."""
    from scipy.ndimage import distance_transform_edt

    target = np.isin(tissue, labels)
    if not target.any():
        return np.full(tissue.shape, np.nan)
    return distance_transform_edt(~target)


def lesion_surroundings(ds, index):
    """Rendered lesion centre (as the audit computes it) and its distances to each tissue."""
    inner = ds._inner
    latents = inner[index][2]
    tissue, lesion = inner.renderer.render_structure(
        latents["z_content"], latents["z_deformation"], latents["z_fissure"], "cpu", clean=inner.clean_content
    )
    tissue, support = tissue.numpy(), lesion.numpy() > 0
    centre = np.argwhere(support).mean(0)
    voxel = tuple(int(round(c)) for c in centre)
    brain = np.argwhere(tissue > 0).mean(0)
    return (
        tissue,
        centre,
        {
            "wm_depth": float(distance_to(tissue, [0, 1, 3, 4])[voxel]),
            "gm_distance": float(distance_to(tissue, [3])[voxel]),
            "csf_distance": float(distance_to(tissue, [1])[voxel]),
            "fissure_distance": float(distance_to(tissue, [4])[voxel]),
            "centre_distance": float(np.linalg.norm(centre - brain)),
        },
    )


def located(row, key, truth, radius):
    estimate = np.array([float(row[f"{key}_{a}"]) for a in range(3)])
    return bool(np.linalg.norm(estimate - truth) <= radius), estimate


def summarise(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return "n/a"
    q1, q2, q3 = np.percentile(values, [25, 50, 75])
    return f"{q2:.2f} [{q1:.2f}, {q3:.2f}]"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit-dir", type=Path, required=True, help="A lesion_reconstruction output directory")
    parser.add_argument("--hit-radius", type=float, default=3.0, help="Detector error (voxels) that counts as found")
    args = parser.parse_args(argv)
    from scipy.stats import mannwhitneyu

    from eval.lesion.lesion_reconstruction import make_dataset

    summary = json.loads((args.audit_dir / "summary.json").read_text())
    with (args.audit_dir / "samples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    audit = summary["arguments"]
    ds = make_dataset(Namespace(**summary["run_settings"]), audit["num_samples"], audit["causal"], audit["split"])

    records, empty = [], 0
    for row in rows:
        index = int(row["index"])
        truth = np.array([float(row[f"truth_{a}"]) for a in range(3)])
        if not np.isfinite(truth).all():
            empty += 1
            continue
        tissue, centre, surroundings = lesion_surroundings(ds, index)
        if not np.allclose(centre, truth, atol=1e-6):
            raise ValueError(
                f"Subject {index}: re-rendered lesion centre {centre} differs from the audit's {truth}; "
                "the generator or settings changed since the audit"
            )
        input_hit, _ = located(row, "input_flair_blob", truth, args.hit_radius)
        recon_hit, peak = located(row, "recon_flair_blob", truth, args.hit_radius)
        group = "missed on input" if not input_hit else "hit" if recon_hit else "new miss"
        peak_tissue = (
            TISSUE.get(int(tissue[tuple(int(round(p)) for p in peak)]), "?") if np.isfinite(peak).all() else "none"
        )
        records.append(
            dict(
                index=index,
                group=group,
                centre_x=truth[0],
                centre_y=truth[1],
                centre_z=truth[2],
                **surroundings,
                flair_response_gain=float(row["flair_response_gain"]),
                recon_peak_error=float(np.linalg.norm(peak - truth)),
                recon_peak_tissue=peak_tissue,
            )
        )

    groups = {name: [r for r in records if r["group"] == name] for name in ("hit", "new miss", "missed on input")}
    print(
        f"{len(records)} subjects: {len(groups['hit'])} found in both, {len(groups['new miss'])} found in the input "
        f"but lost in the reconstruction, {len(groups['missed on input'])} missed on the input too "
        f"(found = detector within {args.hit_radius:g} voxels of the lesion centre)"
        + (f"; {empty} without a rendered lesion skipped" if empty else "")
        + "\n"
    )
    print(f"{'median [IQR]':<36}{'found in both':>24}{'lost in reconstruction':>26}{'p':>8}")
    for key, label in FEATURES.items():
        hit, miss = (np.array([r[key] for r in groups[g]], dtype=float) for g in ("hit", "new miss"))
        hit, miss = hit[np.isfinite(hit)], miss[np.isfinite(miss)]
        p = mannwhitneyu(hit, miss).pvalue if len(hit) > 1 and len(miss) > 1 else float("nan")
        print(f"{label:<36}{summarise(hit):>24}{summarise(miss):>26}{p:>8.3f}")
    print("\nWhere the reconstruction detector fired instead, for the lesions it lost:")
    tissues = [r["recon_peak_tissue"] for r in groups["new miss"]]
    for tissue in sorted(set(tissues), key=tissues.count, reverse=True):
        print(f"  {tissue}: {tissues.count(tissue)}")
    print(f"\n{'index':>6}  {'centre (x, y, z)':<22}{'WM depth':>9}{'to GM':>7}{'to CSF':>8}{'gain':>6}  peak")
    for r in groups["new miss"]:
        centre = f"({r['centre_x']:.0f}, {r['centre_y']:.0f}, {r['centre_z']:.0f})"
        print(
            f"{r['index']:>6}  {centre:<22}{r['wm_depth']:>9.1f}{r['gm_distance']:>7.1f}{r['csf_distance']:>8.1f}"
            f"{r['flair_response_gain']:>6.2f}  {r['recon_peak_tissue']}, {r['recon_peak_error']:.0f} vox away"
        )
    out = args.audit_dir / "miss_location.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
