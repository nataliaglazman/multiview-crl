#!/usr/bin/env python
"""Generate a SynthSeg-simulated multiview dataset from ADNI segmentations.

Supports both FreeSurfer aseg and MUSE (ADNI T1 MUSE) segmentations.
MUSE labels are auto-detected and remapped to FreeSurfer conventions before
SynthSeg synthesis.

Pipeline:
  1. Discover segmentation files for ADNI subjects
  2. Simulate atrophy at configurable severity levels (alpha in [0, 1])
  3. (If MUSE) Remap labels to FreeSurfer space for SynthSeg compatibility
  4. Synthesize paired T1/T2 images via SynthSeg (or built-in fallback)
  5. Organize output in MyCustomDataset-compatible layout + labels CSV

Output structure (directly loadable by MyCustomDataset):
  <out_dir>/
    <subject>_alpha0.00/
      t1/synth_T1.nii.gz
      t2/synth_FLAIR.nii.gz
    <subject>_alpha0.40/
      t1/synth_T1.nii.gz
      t2/synth_FLAIR.nii.gz
    ...
    labels.csv          # Subject,Group[,alpha,original_subject]

Usage:
  # MUSE segmentations (auto-detected, remapped for SynthSeg):
  python scripts/generate_synthseg_dataset.py \\
      --seg-dir /data/natalia/ADNI_MUSE_segs \\
      --adni-labels /data/natalia/labels_cleaned_3class.csv \\
      --out-dir /data/natalia/ADNI_synthseg \\
      --alphas 0.0 0.2 0.4 0.6 0.8 1.0

  # Explicit label set (skip auto-detection):
  python scripts/generate_synthseg_dataset.py \\
      --seg-dir /data/natalia/ADNI_MUSE_segs \\
      --adni-labels /data/natalia/labels_cleaned_3class.csv \\
      --out-dir /data/natalia/ADNI_synthseg \\
      --label-set muse

  # FreeSurfer segmentations + SynthSeg synthesis:
  python scripts/generate_synthseg_dataset.py \\
      --seg-dir /data/natalia/ADNI_freesurfer \\
      --adni-labels /data/natalia/labels_cleaned_3class.csv \\
      --out-dir /data/natalia/ADNI_synthseg \\
      --label-set freesurfer \\
      --synthesizer synthseg \\
      --synthseg-script "python /path/to/SynthSeg/scripts/commands/generation.py"

  # Test with built-in synthesizer (no SynthSeg needed):
  python scripts/generate_synthseg_dataset.py \\
      --seg-dir /data/natalia/ADNI_MUSE_segs \\
      --adni-labels /data/natalia/labels_cleaned_3class.csv \\
      --out-dir /data/natalia/ADNI_synthseg \\
      --alphas 0.0 0.5 1.0 \\
      --synthesizer builtin

  # Train with generated data (same as any ADNI dataset):
  python scripts/launch.py experiments/your_exp.yaml --cluster local \\
      --set dataroot=/data/natalia/ADNI_synthseg \\
            labels_path=/data/natalia/ADNI_synthseg/labels.csv
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.atrophy_simulator import (
    detect_label_set,
    get_bilateral_map,
    get_tissue_classes,
    remap_muse_to_freesurfer,
    simulate_atrophy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Segmentation discovery ────────────────────────────────────────────────────


def _ensure_nifti_ext(path):
    """Add a NIfTI extension if the file lacks one (common in ADNI LONI downloads).

    Checks the first two bytes: gzip magic (\\x1f\\x8b) → .nii.gz, else → .nii.
    Creates a symlink next to the original so nibabel can identify the format.
    Returns the path with the extension (original path if it already has one).
    """
    KNOWN_EXTS = (".nii.gz", ".nii", ".mgz", ".mgh", ".mha", ".nrrd")
    if any(path.endswith(ext) for ext in KNOWN_EXTS):
        return path

    if not os.path.isfile(path):
        return path

    with open(path, "rb") as f:
        magic = f.read(2)

    ext = ".nii.gz" if magic == b"\x1f\x8b" else ".nii"
    linked = path + ext

    if not os.path.exists(linked):
        try:
            os.symlink(os.path.basename(path), linked)
        except OSError:
            # Fallback: hard link (works on NFS where symlinks may be restricted)
            try:
                os.link(path, linked)
            except OSError:
                return path  # give up, let nibabel try the raw path

    return linked


def find_seg(seg_dir, subject_id):
    """Locate segmentation for a subject.

    Checks (in order): ADNI MUSE download layout, FreeSurfer layout, flat layout.

    ADNI MUSE layout on LONI is:
      <seg_dir>/ADNI/<PTID>/T1_MUSE_segmentation/<date>/<IMAGEUID>[.nii.gz]
    We take the most recent date if multiple timepoints exist.
    Files without NIfTI extensions get a symlink so nibabel can load them.
    """
    import glob

    # ── ADNI MUSE download layout ─────────────────────────────────────────
    # <seg_dir>[/ADNI]/<subject>/T1_MUSE_segmentation/<date>/<file>
    for prefix in [os.path.join(seg_dir, "ADNI", subject_id), os.path.join(seg_dir, subject_id)]:
        muse_dir = os.path.join(prefix, "T1_MUSE_segmentation")
        if os.path.isdir(muse_dir):
            # Pick the most recent date subfolder
            date_dirs = sorted(
                [d for d in os.listdir(muse_dir) if os.path.isdir(os.path.join(muse_dir, d))],
                reverse=True,
            )
            for dd in date_dirs:
                session_dir = os.path.join(muse_dir, dd)
                # NIfTI directly in the date folder
                for pattern in ["*.nii.gz", "*.nii", "*.mgz"]:
                    hits = glob.glob(os.path.join(session_dir, pattern))
                    if hits:
                        return _ensure_nifti_ext(sorted(hits)[0])
                # ADNI LONI layout: date/<IMAGEUID_dir>/<file>.nii[.gz]
                for entry in sorted(os.listdir(session_dir)):
                    sub = os.path.join(session_dir, entry)
                    if os.path.isdir(sub):
                        for pattern in ["*.nii.gz", "*.nii", "*.mgz"]:
                            hits = glob.glob(os.path.join(sub, pattern))
                            if hits:
                                return _ensure_nifti_ext(sorted(hits)[0])
                        # Bare files without extension inside the I* dir
                        for f in sorted(os.listdir(sub)):
                            fpath = os.path.join(sub, f)
                            if os.path.isfile(fpath):
                                return _ensure_nifti_ext(fpath)

    # ── FreeSurfer layouts ────────────────────────────────────────────────
    candidates = [
        os.path.join(seg_dir, subject_id, "mri", "aseg.mgz"),
        os.path.join(seg_dir, subject_id, "mri", "aseg.nii.gz"),
        os.path.join(seg_dir, subject_id, "mri", "aparc+aseg.mgz"),
        os.path.join(seg_dir, subject_id, "mri", "aparc+aseg.nii.gz"),
        os.path.join(seg_dir, subject_id, "aseg.mgz"),
        os.path.join(seg_dir, subject_id, "aseg.nii.gz"),
    ]
    # ── Flat / MUSE naming ────────────────────────────────────────────────
    candidates += [
        os.path.join(seg_dir, subject_id, f"{subject_id}_seg.nii.gz"),
        os.path.join(seg_dir, subject_id, f"{subject_id}_muse.nii.gz"),
        os.path.join(seg_dir, f"{subject_id}_seg.nii.gz"),
        os.path.join(seg_dir, f"{subject_id}_muse.nii.gz"),
        os.path.join(seg_dir, f"{subject_id}_aseg.nii.gz"),
        os.path.join(seg_dir, f"{subject_id}.nii.gz"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return _ensure_nifti_ext(c)
    return None


def discover_subjects(seg_dir, adni_labels_csv):
    """Match ADNI label CSV subjects to segmentation files on disk."""
    import pandas as pd

    df = pd.read_csv(adni_labels_csv)
    assert (
        "Subject" in df.columns and "Group" in df.columns
    ), f"Labels CSV must have 'Subject' and 'Group' columns, got: {list(df.columns)}"

    found = []
    missing = []
    for _, row in df.iterrows():
        subj = str(row["Subject"])
        seg_path = find_seg(seg_dir, subj)
        if seg_path:
            found.append({"subject": subj, "group": row["Group"], "seg_path": seg_path})
        else:
            missing.append(subj)

    logger.info(f"Found segmentations for {len(found)}/{len(found) + len(missing)} subjects")
    if missing:
        logger.warning(f"Missing segmentations for {len(missing)} subjects (first 5): {missing[:5]}")
    return found


# ── MUSE → FreeSurfer remap step ─────────────────────────────────────────────


def remap_seg_for_synthseg(seg_path, out_path, label_set):
    """If MUSE, remap to FreeSurfer labels so SynthSeg intensity priors match.

    For FreeSurfer segmentations this is a no-op (returns the input path).
    """
    if label_set != "muse":
        return seg_path

    if os.path.exists(out_path):
        return out_path

    img = nib.load(seg_path)
    seg = np.asarray(img.dataobj).astype(np.int32)
    remapped = remap_muse_to_freesurfer(seg)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(nib.Nifti1Image(remapped, img.affine, img.header), out_path)
    return out_path


# ── Synthesizers ──────────────────────────────────────────────────────────────


def _builtin_synthesize_one(seg_path, out_path, seed=0):
    """Tissue-aware label-to-image synthesis with partial volume effects.

    Mimics the SynthSeg/lab2im approach:
      1. Sample per-label intensities (bilateral-consistent, tissue-grouped,
         guaranteed contrast between CSF/GM/WM)
      2. Render hard intensity image with per-voxel texture noise
      3. Detect tissue boundaries and blend with smoothed version (partial volume)
      4. Add smooth spatial intensity gradient
      5. Multiplicative bias field
      6. Simulate acquisition blur (anisotropic slice profile)
      7. Rician noise
    """
    from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, zoom

    img = nib.load(seg_path)
    seg = np.asarray(img.dataobj).astype(np.int32)
    shape = seg.shape
    rng = np.random.default_rng(seed)

    label_set = detect_label_set(seg)
    bilateral = get_bilateral_map(label_set)
    tissue_classes = get_tissue_classes(label_set)

    # Build label → tissue class lookup
    label_to_tissue = {}
    for tissue, lbls in tissue_classes.items():
        for lbl in lbls:
            label_to_tissue[lbl] = tissue

    # ── Step 1: Sample tissue-class base intensities with guaranteed contrast ─
    # Pick 3 anchor values with minimum separation, then jitter per-structure.
    anchors = sorted(rng.uniform(30, 230, size=3))
    # Enforce minimum 40-unit gaps between CSF / GM / WM
    anchors[1] = max(anchors[1], anchors[0] + 40)
    anchors[2] = max(anchors[2], anchors[1] + 40)
    # Randomly assign ordering: T1-like (WM>GM>CSF) or T2-like (CSF>GM>WM)
    # or intermediate. The random anchor order already gives variety.
    if rng.random() < 0.5:
        anchors = anchors[::-1]  # flip → T2-like contrast

    tissue_base = {
        "csf": anchors[0],
        "cortical_gm": anchors[1] + rng.normal(0, 8),
        "subcortical_gm": anchors[1] + rng.normal(0, 8),
        "cerebellum_gm": anchors[1] + rng.normal(0, 8),
        "wm": anchors[2] + rng.normal(0, 8),
        "brainstem": anchors[1] * 0.5 + anchors[2] * 0.5 + rng.normal(0, 5),
        "vessel": anchors[0] + rng.normal(0, 5),
        "other": anchors[1] + rng.normal(0, 10),
    }

    # ── Step 2: Per-label intensities (bilateral-consistent + small jitter) ───
    labels = [l for l in np.unique(seg) if l != 0]
    canonical_intensity = {}
    intensity_map = {}

    for lbl in labels:
        canon = bilateral.get(lbl, lbl)
        if canon not in canonical_intensity:
            tissue = label_to_tissue.get(lbl, "other")
            base = tissue_base.get(tissue, 120.0)
            canonical_intensity[canon] = base + rng.normal(0, 4)
        intensity_map[lbl] = canonical_intensity[canon]

    # ── Step 3: Render hard intensity image with per-voxel texture ────────────
    hard = np.zeros(shape, dtype=np.float32)
    for lbl in labels:
        mask = seg == lbl
        mean_i = intensity_map[lbl]
        hard[mask] = rng.normal(mean_i, max(abs(mean_i) * 0.03, 1.0), size=int(mask.sum()))

    # ── Step 4: Partial volume at tissue boundaries ───────────────────────────
    # Detect boundary voxels (where label changes in 3x3x3 neighbourhood)
    seg_max = maximum_filter(seg, size=3)
    seg_min = minimum_filter(seg, size=3)
    boundary = (seg_max != seg_min).astype(np.float32)
    # Expand boundary into a soft transition zone
    pv_sigma = rng.uniform(0.8, 1.3)
    boundary_soft = gaussian_filter(boundary, sigma=pv_sigma)
    boundary_soft = np.clip(boundary_soft * 2.5, 0, 1)

    # Smoothed version for blending at boundaries
    smooth = gaussian_filter(hard, sigma=pv_sigma * 1.5)

    # Interior stays sharp, boundaries get partial volume
    synth = hard * (1 - boundary_soft) + smooth * boundary_soft

    # ── Step 5: Smooth spatial intensity gradient (tissue inhomogeneity) ──────
    texture = rng.normal(0, 1, size=shape).astype(np.float32)
    texture = gaussian_filter(texture, sigma=4.0) * 6
    synth += texture

    # ── Step 6: Multiplicative bias field ─────────────────────────────────────
    n_control = 4
    bias_coeff = rng.normal(0, 0.04, size=(3, n_control, n_control, n_control))
    bias_low = np.zeros(shape, dtype=np.float32)
    for ax in range(3):
        zoomed = zoom(bias_coeff[ax], np.array(shape) / n_control, order=3)
        if zoomed.shape != tuple(shape):
            zoomed = zoomed[: shape[0], : shape[1], : shape[2]]
        bias_low += zoomed
    synth *= np.exp(bias_low)

    # ── Step 7: Acquisition blur (simulate finite slice thickness) ────────────
    acq_sigma = [rng.uniform(0.3, 0.7) for _ in range(3)]
    synth = gaussian_filter(synth, sigma=acq_sigma)

    # ── Step 8: Rician noise ──────────────────────────────────────────────────
    noise_std = rng.uniform(1.5, 4.0)
    noise_r = rng.normal(0, noise_std, size=shape)
    noise_i = rng.normal(0, noise_std, size=shape)
    synth = np.sqrt(np.maximum(synth + noise_r, 0) ** 2 + noise_i**2)

    # ── Step 9: Normalize and mask background ─────────────────────────────────
    brain_mask = seg > 0
    synth[~brain_mask] = 0
    if brain_mask.any():
        p99 = np.percentile(synth[brain_mask], 99)
        if p99 > 0:
            synth = synth / p99 * 255.0
    synth = np.clip(synth, 0, 255).astype(np.float32)

    out_img = nib.Nifti1Image(synth, img.affine, img.header)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(out_img, out_path)
    return out_path


def synthesize_builtin(seg_path, t1_path, t2_path, seed=42):
    _builtin_synthesize_one(seg_path, t1_path, seed=seed)
    _builtin_synthesize_one(seg_path, t2_path, seed=seed + 1_000_000)


def synthesize_synthseg(seg_path, t1_path, t2_path, synthseg_script, seed=42):
    for out_path, s in [(t1_path, seed), (t2_path, seed + 1_000_000)]:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cmd = f"{synthseg_script} --input_seg {seg_path} --output {out_path} --seed {s}"
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"SynthSeg failed for {seg_path} → {out_path}: {e.stderr}")
            raise


def synthesize_lab2im(seg_path, t1_path, t2_path, seed=42):
    try:
        from SynthSeg.brain_generator import BrainGenerator
    except ImportError:
        raise ImportError(
            "SynthSeg not found. Install via FreeSurfer or clone "
            "https://github.com/BBillot/SynthSeg and add to PYTHONPATH."
        )

    for out_path, s in [(t1_path, seed), (t2_path, seed + 1_000_000)]:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.random.seed(s)
        generator = BrainGenerator(seg_path)
        im, lab = generator.generate_brain()
        ref = nib.load(seg_path)
        out_img = nib.Nifti1Image(im.squeeze(), ref.affine, ref.header)
        nib.save(out_img, out_path)


# ── Per-subject worker ────────────────────────────────────────────────────────


def _process_subject(args_tuple):
    """Process one (subject, alpha) pair: atrophy → synthesis."""
    (
        subject_id,
        group,
        seg_path,
        alpha,
        out_dir,
        synthesizer,
        synthseg_script,
        wmh_fraction,
        smooth_sigma,
        base_seed,
        label_set,
        remap_to_fs,
    ) = args_tuple

    sample_id = f"{subject_id}_alpha{alpha:.2f}"
    sample_dir = os.path.join(out_dir, sample_id)
    t1_path = os.path.join(sample_dir, "t1", "synth_T1.nii.gz")
    t2_path = os.path.join(sample_dir, "t2", "synth_FLAIR.nii.gz")

    if os.path.exists(t1_path) and os.path.exists(t2_path):
        return {"sample_id": sample_id, "group": group, "alpha": alpha, "subject": subject_id, "status": "cached"}

    seg_out_dir = os.path.join(out_dir, "_segmentations")
    os.makedirs(seg_out_dir, exist_ok=True)
    atrophied_seg_path = os.path.join(seg_out_dir, f"{sample_id}_seg.nii.gz")

    if not os.path.exists(atrophied_seg_path):
        simulate_atrophy(
            seg_path,
            alpha=alpha,
            label_set=label_set,
            wmh_fraction=wmh_fraction,
            smooth_sigma=smooth_sigma,
            out_path=atrophied_seg_path,
        )

    # Optional: remap MUSE → FreeSurfer labels before synthesis.
    # Only needed if the SynthSeg generation model was trained on FS labels.
    # lab2im and the builtin synthesizer work with any label set — MUSE's
    # 152 ROIs actually produce more realistic cortical intensity variation
    # than collapsing to FS's ~30 labels.
    synth_seg_path = atrophied_seg_path
    if remap_to_fs and label_set == "muse":
        remapped_path = os.path.join(seg_out_dir, f"{sample_id}_seg_fs.nii.gz")
        synth_seg_path = remap_seg_for_synthseg(atrophied_seg_path, remapped_path, label_set)

    seed = base_seed + hash(sample_id) % (2**31)

    if synthesizer == "builtin":
        synthesize_builtin(synth_seg_path, t1_path, t2_path, seed=seed)
    elif synthesizer == "synthseg":
        synthesize_synthseg(synth_seg_path, t1_path, t2_path, synthseg_script, seed=seed)
    elif synthesizer == "lab2im":
        synthesize_lab2im(synth_seg_path, t1_path, t2_path, seed=seed)
    else:
        raise ValueError(f"Unknown synthesizer: {synthesizer}")

    return {"sample_id": sample_id, "group": group, "alpha": alpha, "subject": subject_id, "status": "generated"}


# ── Main pipeline ─────────────────────────────────────────────────────────────


def generate_dataset(
    seg_dir,
    adni_labels_csv,
    out_dir,
    alphas,
    synthesizer="builtin",
    synthseg_script=None,
    label_set=None,
    remap_to_fs=False,
    wmh_fraction=0.0,
    smooth_sigma=0.6,
    seed=42,
    num_workers=4,
    alpha_group_map=None,
):
    os.makedirs(out_dir, exist_ok=True)

    subjects = discover_subjects(seg_dir, adni_labels_csv)
    if not subjects:
        logger.error("No subjects found. Check --seg-dir and --adni-labels paths.")
        return None

    # Auto-detect label set from the first subject if not specified
    if label_set is None:
        first_seg = nib.load(subjects[0]["seg_path"])
        label_set = detect_label_set(np.asarray(first_seg.dataobj).astype(np.int32))
        logger.info(f"Auto-detected label set: {label_set}")
    else:
        logger.info(f"Using label set: {label_set}")

    work_items = []
    for s in subjects:
        for alpha in alphas:
            group = s["group"]
            if alpha_group_map:
                for (lo, hi), g in alpha_group_map.items():
                    if lo <= alpha < hi:
                        group = g
                        break
            work_items.append(
                (
                    s["subject"],
                    group,
                    s["seg_path"],
                    alpha,
                    out_dir,
                    synthesizer,
                    synthseg_script,
                    wmh_fraction,
                    smooth_sigma,
                    seed,
                    label_set,
                    remap_to_fs,
                )
            )

    logger.info(
        f"Generating {len(work_items)} samples "
        f"({len(subjects)} subjects × {len(alphas)} alpha values) "
        f"using '{synthesizer}' synthesizer, label_set='{label_set}', "
        f"{num_workers} workers"
    )

    results = []
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_process_subject, w): w for w in work_items}
            done = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    w = futures[future]
                    logger.error(f"Failed: {w[0]}_alpha{w[3]:.2f}: {e}")
                done += 1
                if done % 50 == 0 or done == len(work_items):
                    logger.info(f"  Progress: {done}/{len(work_items)}")
    else:
        for i, w in enumerate(work_items):
            try:
                result = _process_subject(w)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed: {w[0]}_alpha{w[3]:.2f}: {e}")
            if (i + 1) % 50 == 0 or i + 1 == len(work_items):
                logger.info(f"  Progress: {i + 1}/{len(work_items)}")

    cached = sum(1 for r in results if r["status"] == "cached")
    generated = sum(1 for r in results if r["status"] == "generated")
    logger.info(f"Done: {generated} generated, {cached} cached, {len(work_items) - len(results)} failed")

    labels_path = os.path.join(out_dir, "labels.csv")
    results.sort(key=lambda r: (r["subject"], r["alpha"]))
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Subject", "Group", "alpha", "original_subject"])
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "Subject": r["sample_id"],
                    "Group": r["group"],
                    "alpha": r["alpha"],
                    "original_subject": r["subject"],
                }
            )
    logger.info(f"Labels CSV: {labels_path} ({len(results)} entries)")

    manifest = {
        "seg_dir": seg_dir,
        "adni_labels_csv": adni_labels_csv,
        "alphas": alphas,
        "synthesizer": synthesizer,
        "label_set": label_set,
        "wmh_fraction": wmh_fraction,
        "smooth_sigma": smooth_sigma,
        "seed": seed,
        "n_subjects": len(subjects),
        "n_samples": len(results),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return labels_path


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate SynthSeg-simulated multiview dataset from ADNI segmentations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--seg-dir",
        required=True,
        help="Directory with segmentation files. Accepts FreeSurfer layout "
        "(subject/mri/aseg.mgz), MUSE layout (subject_seg.nii.gz), or flat.",
    )
    parser.add_argument(
        "--adni-labels",
        required=True,
        help="ADNI labels CSV with 'Subject' and 'Group' columns.",
    )
    parser.add_argument("--out-dir", required=True, help="Output dataset directory.")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Atrophy severity levels (default: 0.0 0.2 0.4 0.6 0.8 1.0).",
    )
    parser.add_argument(
        "--label-set",
        choices=["freesurfer", "muse"],
        default=None,
        help="Segmentation label convention. Auto-detected from the first file if omitted.",
    )
    parser.add_argument(
        "--synthesizer",
        choices=["builtin", "synthseg", "lab2im"],
        default="builtin",
        help="Image synthesis backend: 'builtin' (simple, no deps), "
        "'synthseg' (subprocess call), or 'lab2im' (Python API). Default: builtin.",
    )
    parser.add_argument(
        "--synthseg-script",
        default="python SynthSeg/scripts/commands/generation.py",
        help="SynthSeg generation command (only for --synthesizer synthseg).",
    )
    parser.add_argument("--wmh-fraction", type=float, default=0.0, help="WMH injection fraction (0 = none).")
    parser.add_argument("--smooth-sigma", type=float, default=0.6, help="Label boundary smoothing sigma.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel workers.")
    parser.add_argument(
        "--remap-to-fs",
        action="store_true",
        help="Remap MUSE labels to FreeSurfer before synthesis. Only needed if "
        "your SynthSeg generation model was trained on FS labels. Off by default — "
        "MUSE's 152 ROIs give better cortical intensity variation than FS's ~30.",
    )
    parser.add_argument(
        "--remap-groups-by-alpha",
        action="store_true",
        help="Override diagnostic group based on alpha: [0, 0.3)->CN, [0.3, 0.7)->MCI, [0.7, 1.01)->AD.",
    )

    args = parser.parse_args()

    alpha_group_map = None
    if args.remap_groups_by_alpha:
        alpha_group_map = {(0, 0.3): "CN", (0.3, 0.7): "MCI", (0.7, 1.01): "AD"}

    labels_path = generate_dataset(
        seg_dir=args.seg_dir,
        adni_labels_csv=args.adni_labels,
        out_dir=args.out_dir,
        alphas=args.alphas,
        synthesizer=args.synthesizer,
        synthseg_script=args.synthseg_script,
        label_set=args.label_set,
        remap_to_fs=args.remap_to_fs,
        wmh_fraction=args.wmh_fraction,
        smooth_sigma=args.smooth_sigma,
        seed=args.seed,
        num_workers=args.num_workers,
        alpha_group_map=alpha_group_map,
    )

    if labels_path:
        print(f"\nDataset ready at: {args.out_dir}")
        print(f"Labels CSV:       {labels_path}")
        print(f"\nTo train:")
        print(f"  python scripts/launch.py experiments/<exp>.yaml --cluster local \\")
        print(f"      --set dataroot={args.out_dir} labels_path={labels_path}")


if __name__ == "__main__":
    main()
