"""Export paired ADNI T1/FLAIR scans into a bank that ``warm.py --real`` trains on. No label enters its arrays.

    python adni_bank.py BANK --data-dir DIR --labels CSV --bias-correction n4|none [--masks-dir DIR]
        [--val-frac 0.2] [--test-frac 0.1] [--split-seed 0] [--check-split SPLIT_JSON] [--spacing 3] [--res 64]
        [--qc-only]

Subjects resolve as both trainers resolve them (utils.utils.load_data on the labels CSV), and the split is
data.splits.subject_level_split at the encoder ADNI recipe's fractions and seed, so a kept subject sits in the split it
had in the October conv_mlp runs; --check-split compares with such a run's split.json. The exclusion rule, fixed on 24
Nov 2026 before this recipe saw ADNI, then drops pairs within their split: T1/FLAIR brain-mask Dice below BRAIN_DICE, or
either mask under BRAIN_ML mL, on the masks as stored.

Every scan and mask must share one grid (all were rigidly registered to MNI), so one box serves every subject. Each
scan, in RAS orientation:
  1. --bias-correction n4: SimpleITK's N4 inside the brain mask, fitted on a 3x shrunk copy, the field divided out.
  2. --spacing mm voxels: the mean over the brain voxels of each k^3 block, k = spacing / the scan's spacing (3 for the
     1 mm MNI grid). A block is brain when at least half of it is; averaging brain voxels only keeps the edge from
     darkening.
  3. The centre --res^3 box of that grid, zero-padded where the grid is smaller.
  4. z-scored over its brain voxels, 0 outside, stored as float16.
Writes {split}_t1.npy and {split}_flair.npy (n, res, res, res), {split}_ids.npy (subject IDs in row order),
manifest.csv (every resolved subject: split, row, group, QC, exclusion, the brain volume the box cut off, file names,
and the image ID, series ID and acquisition date where a file name carries them) and bank_meta.json, then prints
BANK DONE. --qc-only stops after manifest.csv and bank_meta.json (QC DONE), so exclusions and counts can be read first.
Next: residual_bank.py BANK t1 flair, then warm.py BANK RUN cold STEPS EVERY --real --t1key t1 ....
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.splits import split_summary, subject_level_split  # noqa: E402

SPLITS = ("train", "val", "test")
VIEWS = {"t1": ("image", "mask_image"), "flair": ("z_image", "mask_z_image")}
BRAIN_DICE = 0.9
BRAIN_ML = 300.0
WORKERS = int(os.environ.get("BANK_WORKERS", 8))
N4_SHRINK = 3
N4_ITERATIONS = (50, 50, 50, 50)
IMAGE_ID = re.compile(r"(?<![A-Za-z0-9])I(\d{4,9})(?!\d)")
SERIES_ID = re.compile(r"(?<![A-Za-z0-9])S(\d{3,9})(?!\d)")
DATE = re.compile(r"(?<!\d)((?:19|20)\d{2})(\d{2})(\d{2})\d{0,9}(?!\d)")
COLUMNS = (
    "subject",
    "group",
    "split",
    "row",
    "excluded",
    "reason",
    "mask_ml_t1",
    "mask_ml_flair",
    "mask_dice",
    "cut_ml_t1",
    "cut_ml_flair",
    "t1_file",
    "flair_file",
    "t1_image_id",
    "flair_image_id",
    "t1_series_id",
    "flair_series_id",
    "t1_date",
    "flair_date",
)


def identifiers(path):
    """The LONI image ID, series ID and acquisition date (YYYY-MM-DD) a file name carries; '' where it has none."""
    name = os.path.basename(str(path))
    image, series = IMAGE_ID.search(name), SERIES_ID.search(name)
    date = ""
    for match in DATE.finditer(name):
        year, month, day = (int(g) for g in match.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
            date = f"{year:04d}-{month:02d}-{day:02d}"
            break
    return (image.group(1) if image else "", series.group(1) if series else "", date)


def load(path):
    """A NIfTI volume as float32 in RAS orientation (any NaN read as 0), and its voxel size in mm."""
    image = nib.as_closest_canonical(nib.load(str(path)))
    volume = np.nan_to_num(np.asarray(image.dataobj, dtype=np.float32), copy=False)
    return volume, tuple(float(z) for z in image.header.get_zooms()[:3])


def brain_mask(item, view):
    """The view's brain mask from disk (> .5), or where its image is non-zero when the subject has no masks."""
    image_key, mask_key = VIEWS[view]
    if mask_key in item:
        return load(item[mask_key])[0] > 0.5
    return load(item[image_key])[0] != 0


def n4(volume, mask, spacing):
    """SimpleITK's N4 inside ``mask``, fitted on an N4_SHRINK-times smaller copy; the volume over the fitted field."""
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    fit = mask & (volume > 0)
    tiny = 1e-3 * float(volume[fit].mean())
    image = sitk.GetImageFromArray(np.ascontiguousarray(np.maximum(volume, tiny).transpose(2, 1, 0)))
    image.SetSpacing([float(s) for s in spacing])
    where = sitk.GetImageFromArray(np.ascontiguousarray(fit.astype(np.uint8).transpose(2, 1, 0)))
    where.CopyInformation(image)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(N4_ITERATIONS))
    corrector.Execute(sitk.Shrink(image, [N4_SHRINK] * 3), sitk.Shrink(where, [N4_SHRINK] * 3))
    log_field = sitk.GetArrayFromImage(corrector.GetLogBiasFieldAsImage(image)).transpose(2, 1, 0)
    return np.where(mask, volume / np.exp(log_field), 0).astype(np.float32)


def block_mean(volume, mask, k):
    """Each k^3 block's mean over its brain voxels, and whether at least half the block is brain (else 0). Each axis
    is first trimmed to a multiple of k, the remainder split between its two ends."""
    box = tuple(slice((n % k) // 2, (n % k) // 2 + n // k * k) for n in volume.shape)
    volume, mask = volume[box], mask[box].astype(np.float32)
    shape = [n // k for n in volume.shape]

    def summed(x):
        return x.reshape(shape[0], k, shape[1], k, shape[2], k).sum((1, 3, 5))

    count, total = summed(mask), summed(volume * mask)
    brain = count >= 0.5 * k**3
    out = np.zeros(shape, dtype=np.float32)
    out[brain] = total[brain] / count[brain]
    return out, brain


def centre_box(x, res):
    """The centre res^3 box of ``x``: cropped where ``x`` is larger, zero-padded where smaller (any odd voxel goes to
    the far end), the same box for every volume of one shape."""
    out = np.zeros((res,) * 3, dtype=x.dtype)
    source, target = [], []
    for n in x.shape:
        start = abs(n - res) // 2
        source.append(slice(start, start + res) if n >= res else slice(0, n))
        target.append(slice(0, res) if n >= res else slice(start, start + n))
    out[tuple(target)] = x[tuple(source)]
    return out


def standardize(volume, brain):
    """z-scored over the brain voxels, 0 elsewhere."""
    values = volume[brain]
    out = np.zeros(volume.shape, dtype=np.float32)
    out[brain] = (values - values.mean()) / max(float(values.std()), 1e-8)
    return out


def prepare(item, view, settings):
    """One scan through steps 1-4, and the brain volume (mL) the box cut off."""
    image_key, _ = VIEWS[view]
    volume, spacing = load(item[image_key])
    mask = brain_mask(item, view)
    if settings["bias_correction"] == "n4":
        volume = n4(volume, mask, spacing)
    volume, brain = block_mean(volume, mask, settings["k"])
    kept = centre_box(brain, settings["res"])
    cut = float(brain.sum() - kept.sum()) * settings["spacing"] ** 3 / 1000
    return standardize(centre_box(volume, settings["res"]), kept), cut


def qc(job):
    """Brain-mask volumes (mL) and the T1/FLAIR mask Dice of one subject, on its masks as stored."""
    position, item, voxel_ml = job
    masks = [brain_mask(item, view) for view in VIEWS]
    volumes = [float(m.sum()) * voxel_ml for m in masks]
    overlap = float((masks[0] & masks[1]).sum())
    dice = 2 * overlap / max(float(masks[0].sum() + masks[1].sum()), 1.0)
    return position, {"mask_ml_t1": volumes[0], "mask_ml_flair": volumes[1], "mask_dice": dice}


def export(job):
    """Steps 1-4 for a run of one split's rows, written into the bank's arrays; returns the volumes cut off."""
    root, split, rows, settings = job
    arrays = {view: np.load(f"{root}/{split}_{view}.npy", mmap_mode="r+") for view in VIEWS}
    cut = {}
    for row, item in rows:
        cut[row] = {}
        for view, array in arrays.items():
            array[row], cut[row][view] = prepare(item, view, settings)
    for array in arrays.values():
        array.flush()
    return split, cut


def resolve(data_dir, labels, masks_dir):
    """The subjects the trainers load, in their order, and MyCustomDataset's label map (group name -> integer)."""
    import pandas as pd

    from utils.utils import load_data

    table = pd.read_csv(labels)
    label_map = {group: i for i, group in enumerate(sorted(table["Group"].unique()))}
    items, missing = load_data(table, data_dir, label_map, masks_dir=masks_dir)
    if not items:
        raise SystemExit(f"no subject in {labels} has both a T1 and a FLAIR under {data_dir}")
    if not all("mask_image" in it and "mask_z_image" in it for it in items):
        # The trainers' rule: masks come from disk for every subject or for none.
        for it in items:
            it.pop("mask_image", None)
            it.pop("mask_z_image", None)
    return items, missing, label_map


def one_grid(items):
    """The (shape, affine) every scan and mask shares, read from the headers; refuses a second grid."""
    grids = {}
    for it in items:
        for key in ("image", "z_image", "mask_image", "mask_z_image"):
            if key in it:
                header = nib.load(it[key])
                grids.setdefault((header.shape[:3], tuple(np.round(header.affine, 3).ravel())), []).append(it[key])
    if len(grids) > 1:
        lines = [f"  {len(paths)} files on shape {shape}, e.g. {paths[0]}" for (shape, _), paths in grids.items()]
        raise SystemExit("scans and masks must share one grid (one rigid MNI box); found:\n" + "\n".join(lines))
    shape, affine = next(iter(grids))
    return shape, np.asarray(affine).reshape(4, 4)


def git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent)
        return out.stdout.strip() or None
    except OSError:
        return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("bank")
    p.add_argument("--data-dir", required=True, help="The folder of subject folders, <subject>/{t1,t2}")
    p.add_argument("--labels", required=True, help="Labels CSV with Subject and Group, as the trainers read it")
    p.add_argument("--masks-dir", default=None, help="Where <subject>/{t1,t2}/*_brain_mask.nii.gz live (data dir)")
    p.add_argument("--bias-correction", choices=("n4", "none"), required=True)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--check-split", default=None, help="A run's split.json the split must reproduce")
    p.add_argument("--spacing", type=float, default=3.0, help="Output voxel size in mm (default 3)")
    p.add_argument("--res", type=int, default=64, help="Output box edge in voxels, a multiple of 4 (default 64)")
    p.add_argument("--qc-only", action="store_true", help="Stop after manifest.csv and bank_meta.json")
    a = p.parse_args(argv)
    if a.res < 16 or a.res % 4:
        p.error("--res must be a multiple of 4, at least 16: warm.py pools 4^3 voxels into a map cell")
    if not a.spacing > 0:
        p.error("--spacing must be positive")
    return a


def main(argv=None):
    a = parse_args(argv)
    root = Path(a.bank)
    root.mkdir(parents=True, exist_ok=True)
    items, missing, label_map = resolve(a.data_dir, a.labels, a.masks_dir)
    groups = {i: group for group, i in label_map.items()}
    split = subject_level_split(items, val_frac=a.val_frac, test_frac=a.test_frac, seed=a.split_seed)
    subjects = {mode: [items[i]["subject"] for i in indices] for mode, indices in split.items()}
    if a.check_split:
        reference = json.loads(Path(a.check_split).read_text())
        differ = {m: len(set(subjects[m]) ^ set(reference["subjects"].get(m, []))) for m in SPLITS}
        if any(differ.values()) or reference.get("label_map", label_map) != label_map:
            raise SystemExit(f"the split differs from {a.check_split}: subjects differing per split {differ}")
        print("split matches", a.check_split, flush=True)
    for line in split_summary(items, split, label_names=groups):
        print(line, flush=True)

    shape, affine = one_grid(items)
    spacing = [float(z) for z in nib.as_closest_canonical(nib.load(items[0]["image"])).header.get_zooms()[:3]]
    if max(spacing) - min(spacing) > 1e-3 * max(spacing):
        raise SystemExit(f"scans must have isotropic voxels, got {spacing}")
    k = int(round(a.spacing / spacing[0]))
    if k < 1 or abs(k * spacing[0] - a.spacing) > 1e-3 * a.spacing:
        raise SystemExit(f"--spacing {a.spacing} must be a whole multiple of the scans' {spacing[0]} mm")
    print(f"grid {shape} at {spacing[0]} mm; {k}^3 blocks at {a.spacing} mm, centre {a.res}^3 box", flush=True)

    voxel_ml = float(np.prod(spacing)) / 1000
    jobs = [(i, it, voxel_ml) for i, it in enumerate(items)]
    with ProcessPoolExecutor(WORKERS) if WORKERS > 1 else _Serial() as pool:
        measures = dict(pool.map(qc, jobs, chunksize=8))
    rows, kept = [], {mode: [] for mode in SPLITS}
    for mode in SPLITS:
        for i in split[mode]:
            it, m = items[i], measures[i]
            reasons = []
            if m["mask_dice"] < BRAIN_DICE:
                reasons.append(f"mask Dice {m['mask_dice']:.3f} < {BRAIN_DICE}")
            for view in VIEWS:
                if m[f"mask_ml_{view}"] < BRAIN_ML:
                    reasons.append(f"{view} mask {m[f'mask_ml_{view}']:.0f} mL < {BRAIN_ML:.0f}")
            row = -1 if reasons else len(kept[mode])
            if not reasons:
                kept[mode].append(i)
            ids = {view: identifiers(it[VIEWS[view][0]]) for view in VIEWS}
            rows.append(
                {
                    "subject": it["subject"],
                    "group": groups[it["label"]],
                    "split": mode,
                    "row": row,
                    "excluded": int(bool(reasons)),
                    "reason": "; ".join(reasons),
                    **{key: round(value, 4) for key, value in m.items()},
                    "cut_ml_t1": "",
                    "cut_ml_flair": "",
                    "t1_file": os.path.basename(it["image"]),
                    "flair_file": os.path.basename(it["z_image"]),
                    **{f"{view}_image_id": ids[view][0] for view in VIEWS},
                    **{f"{view}_series_id": ids[view][1] for view in VIEWS},
                    **{f"{view}_date": ids[view][2] for view in VIEWS},
                }
            )
    counts = {
        mode: {"resolved": len(split[mode]), "excluded": len(split[mode]) - len(kept[mode]), "kept": len(kept[mode])}
        for mode in SPLITS
    }
    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "data_dir": str(a.data_dir),
        "labels": str(a.labels),
        "masks_dir": a.masks_dir,
        "masks_from_disk": "mask_image" in items[0],
        "bias_correction": a.bias_correction,
        "n4": {"shrink": N4_SHRINK, "iterations": list(N4_ITERATIONS)} if a.bias_correction == "n4" else None,
        "input_grid": {"shape": list(shape), "spacing_mm": spacing, "affine": affine.tolist()},
        "spacing_mm": a.spacing,
        "block": k,
        "res": a.res,
        "split": {"val_frac": a.val_frac, "test_frac": a.test_frac, "seed": a.split_seed, "check": a.check_split},
        "label_map": label_map,
        "exclusion": {"mask_dice_below": BRAIN_DICE, "mask_ml_below": BRAIN_ML, "fixed": "2026-11-24"},
        "counts": counts,
        "unresolved_subjects": [str(s) for s in missing],
    }

    def write():
        with open(root / "manifest.csv", "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        (root / "bank_meta.json").write_text(json.dumps(meta, indent=1))

    print("unresolved CSV subjects", len(missing), flush=True)
    for mode in SPLITS:
        print(mode, counts[mode], flush=True)
    for r in rows:
        if r["excluded"]:
            print(f"  excluded {r['split']:5s} {r['subject']}: {r['reason']}", flush=True)
    write()
    if a.qc_only:
        print("QC DONE", flush=True)
        return
    settings = {"bias_correction": a.bias_correction, "k": k, "res": a.res, "spacing": a.spacing}
    jobs = []
    for mode in SPLITS:
        n = len(kept[mode])
        np.save(root / f"{mode}_ids.npy", np.asarray([items[i]["subject"] for i in kept[mode]], dtype=str))
        for view in VIEWS:
            if not n:  # a zero-length memmap cannot be mapped
                np.save(root / f"{mode}_{view}.npy", np.zeros((0, a.res, a.res, a.res), dtype=np.float16))
                continue
            np.lib.format.open_memmap(
                root / f"{mode}_{view}.npy", mode="w+", dtype=np.float16, shape=(n, a.res, a.res, a.res)
            ).flush()
        ordered = list(enumerate(items[i] for i in kept[mode]))
        jobs += [(str(root), mode, ordered[s : s + 8], settings) for s in range(0, n, 8)]
    cuts = {mode: {} for mode in SPLITS}
    with ProcessPoolExecutor(WORKERS) if WORKERS > 1 else _Serial() as pool:
        for mode, cut in pool.map(export, jobs):
            cuts[mode].update(cut)
            print("done", mode, min(cut), "to", max(cut), flush=True)
    for r in rows:
        if not r["excluded"]:
            for view in VIEWS:
                r[f"cut_ml_{view}"] = round(cuts[r["split"]][r["row"]][view], 2)
    worst = max((max(c.values()) for mode in SPLITS for c in cuts[mode].values()), default=0.0)
    meta["largest_cut_ml"] = worst
    write()
    print(f"largest brain volume outside the box: {worst:.1f} mL", flush=True)
    print("BANK DONE", flush=True)


class _Serial:
    """ProcessPoolExecutor's map in this process (BANK_WORKERS=1): for tests and debugging."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @staticmethod
    def map(fn, jobs, chunksize=1):
        return map(fn, jobs)


if __name__ == "__main__":
    main()
