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
  python scripts/generate_synthsegataset.py \\
      --seg-dir /data/natalia/ADNI_MUSE_segs \\
      --adni-labels /data/natalia/labels_cleaned_3class.csv \\
      --out-dir /data/natalia/ADNI_synthseg \\
      --alphas 0.0 0.2 0.4 0.6 0.8 1.0

  # Explicit label set (skip auto-detection):
  python scripts/generate_synthsegataset.py \\
      --seg-dir /data/natalia/ADNI_MUSE_segs \\
      --adni-labels /data/natalia/labels_cleaned_3class.csv \\
      --out-dir /data/natalia/ADNI_synthseg \\
      --label-set muse

  # FreeSurfer segmentations + SynthSeg synthesis:
  python scripts/generate_synthsegataset.py \\
      --seg-dir /data/natalia/ADNI_freesurfer \\
      --adni-labels /data/natalia/labels_cleaned_3class.csv \\
      --out-dir /data/natalia/ADNI_synthseg \\
      --label-set freesurfer \\
      --synthesizer synthseg \\
      --synthseg-script "python /path/to/SynthSeg/scripts/commands/generation.py"

  # Test with built-in synthesizer (no SynthSeg needed):
  python scripts/generate_synthsegataset.py \\
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
    CONTRAST_PRIORS,
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


def discover_subjects(seg_dir, adni_labels_csv=None):
    """Find subjects with segmentation files on disk.

    If adni_labels_csv is provided, match CSV subjects to segmentation files.
    Otherwise, auto-discover all subjects from the segmentation directory
    (scans ADNI/<PTID>/ subdirectories and flat files).
    """
    if adni_labels_csv is not None:
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

    # Auto-discover from directory structure
    found = []
    # Check ADNI/<PTID>/ layout
    adni_dir = os.path.join(seg_dir, "ADNI")
    scan_dirs = [adni_dir] if os.path.isdir(adni_dir) else [seg_dir]
    for scan_dir in scan_dirs:
        if not os.path.isdir(scan_dir):
            continue
        for entry in sorted(os.listdir(scan_dir)):
            if not os.path.isdir(os.path.join(scan_dir, entry)):
                continue
            seg_path = find_seg(seg_dir, entry)
            if seg_path:
                found.append({"subject": entry, "group": "unknown", "seg_path": seg_path})

    logger.info(f"Auto-discovered {len(found)} subjects from {seg_dir}")
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


def _builtin_synthesize_one(seg_path, out_path, seed=0, contrast=None):
    """Label-to-image synthesis from a segmentation map.

    Pipeline:
      1. GMM sampling: per-voxel intensity = N(mean[label], std[label])
         with bilateral consistency and tissue-class contrast guarantees
      2. Gaussian blur (simulates acquisition PSF)
      3. Multiplicative bias field
      4. Rician noise
      5. Intensity normalisation

    No spatial deformation is applied — the generated data is intended for
    a pipeline where real images are already registered to a common template,
    so synthetic data should match that aligned geometry.

    Parameters
    ----------
    contrast : str or None
        None  — fully random contrast (SynthSeg-style domain randomization).
        't1'  — T1-weighted priors (WM bright, GM mid, CSF dark).
        't2'  — T2-weighted priors (CSF bright, GM mid, WM dark).
        'flair' — FLAIR priors (CSF suppressed, GM/WM bright).
    """
    from scipy.ndimage import gaussian_filter

    img = nib.load(seg_path)
    seg = np.asarray(img.dataobj).astype(np.int32)
    rng = np.random.default_rng(seed)

    label_set = detect_label_set(seg)
    bilateral = get_bilateral_map(label_set)
    tissue_classes = get_tissue_classes(label_set)

    # Build label → tissue class lookup
    label_to_tissue = {}
    for tissue, lbls in tissue_classes.items():
        for lbl in lbls:
            label_to_tissue[lbl] = tissue

    shape = seg.shape

    # ── 2. Sample GMM parameters ──────────────────────────────────────────
    if contrast is not None and contrast in CONTRAST_PRIORS:
        # Contrast-specific: sample from N(prior_mean, prior_std) per tissue
        priors = CONTRAST_PRIORS[contrast]
        tissue_mean = {}
        for tissue, (mu, sigma) in priors.items():
            tissue_mean[tissue] = rng.normal(mu, sigma)
    else:
        # Fully random: three anchors with guaranteed separation
        anchors = sorted(rng.uniform(40, 220, size=3))
        anchors[1] = max(anchors[1], anchors[0] + 35)
        anchors[2] = max(anchors[2], anchors[1] + 35)
        if rng.random() < 0.5:
            anchors = anchors[::-1]
        tissue_mean = {
            "csf": anchors[0],
            "cortical_gm": anchors[1] + rng.normal(0, 10),
            "subcortical_gm": anchors[1] + rng.normal(0, 10),
            "cerebellum_gm": anchors[1] + rng.normal(0, 10),
            "wm": anchors[2] + rng.normal(0, 10),
            "brainstem": 0.5 * anchors[1] + 0.5 * anchors[2] + rng.normal(0, 5),
            "vessel": anchors[0] + rng.normal(0, 5),
            "other": anchors[1] + rng.normal(0, 10),
        }

    # Per-label means: bilateral pairs share the same mean, small jitter
    # between structures within a tissue class.
    labels = [l for l in np.unique(seg) if l != 0]
    canonical_mean = {}
    mean_map = {}
    std_map = {}

    for lbl in labels:
        canon = bilateral.get(lbl, lbl)
        if canon not in canonical_mean:
            tissue = label_to_tissue.get(lbl, "other")
            base = tissue_mean.get(tissue, 120.0)
            canonical_mean[canon] = base + rng.normal(0, 5)
        mean_map[lbl] = canonical_mean[canon]
        # Per-label std: sampled from a prior range (lab2im uses U(5, 25))
        std_map[lbl] = rng.uniform(5, 25)

    # ── 3. GMM sampling: synth[v] = N(mean[label[v]], std[label[v]]) ─────
    synth = np.zeros(shape, dtype=np.float32)
    for lbl in labels:
        mask = seg == lbl
        n = int(mask.sum())
        if n > 0:
            synth[mask] = rng.normal(mean_map[lbl], std_map[lbl], size=n)

    # ── 4. Gaussian blur (acquisition PSF / slice profile) ───────────────
    blur_sigma = [rng.uniform(0.5, 1.15) for _ in range(3)]
    synth = gaussian_filter(synth, sigma=blur_sigma)

    # ── 5. Gamma augmentation (contrast non-linearity, lab2im std=0.2) ───
    synth = np.clip(synth, 0, None)
    fg = synth[seg > 0]
    if fg.size > 0 and fg.max() > 0:
        synth = synth / fg.max()
        gamma = np.exp(rng.normal(0, 0.2))
        synth = np.power(synth, gamma)
        synth *= 300.0  # lab2im clipping range

    # ── 7. Rician noise ──────────────────────────────────────────────────
    noise_std = rng.uniform(2, 8)
    noise_r = rng.normal(0, noise_std, size=shape)
    noise_i = rng.normal(0, noise_std, size=shape)
    synth = np.sqrt(np.maximum(synth + noise_r, 0) ** 2 + noise_i**2)

    # ── 8. Normalise to [0, 255], zero background ────────────────────────
    brain_mask = seg > 0
    synth[~brain_mask] = 0
    if brain_mask.any():
        p99 = np.percentile(synth[brain_mask], 99.5)
        if p99 > 0:
            synth = synth / p99 * 255.0
    synth = np.clip(synth, 0, 255).astype(np.float32)

    out_img = nib.Nifti1Image(synth, img.affine, img.header)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(out_img, out_path)
    return out_path


def synthesize_builtin(seg_path, t1_path, t2_path, seed=42, contrast_mode="paired"):
    """Generate a paired T1/T2 from the same segmentation.

    Parameters
    ----------
    contrast_mode : str
        'paired'  — view 1 uses T1 priors, view 2 uses FLAIR priors.
                    The style axis is meaningful: it captures real T1-vs-FLAIR
                    contrast differences. Best for content/style disentanglement.
        'random'  — both views use fully random contrast (SynthSeg-style).
                    Maximum style diversity, but the style axis is unstructured.
    """
    if contrast_mode == "paired":
        _builtin_synthesize_one(seg_path, t1_path, seed=seed, contrast="t1")
        _builtin_synthesize_one(seg_path, t2_path, seed=seed + 1_000_000, contrast="flair")
    else:
        _builtin_synthesize_one(seg_path, t1_path, seed=seed, contrast=None)
        _builtin_synthesize_one(seg_path, t2_path, seed=seed + 1_000_000, contrast=None)


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
        contrast_mode,
    ) = args_tuple

    sample_id = f"{subject_id}_alpha{alpha:.2f}"
    sample_dir = os.path.join(out_dir, sample_id)
    t1_path = os.path.join(sample_dir, "t1", "synth_T1.nii.gz")
    t2_path = os.path.join(sample_dir, "t2", "synth_FLAIR.nii.gz")

    if os.path.exists(t1_path) and os.path.exists(t2_path):
        return {"sample_id": sample_id, "group": group, "alpha": alpha, "subject": subject_id, "status": "cached"}

    # Skip subjects whose input segmentation is known-bad (recorded on previous failure)
    bad_inputs_file = os.path.join(out_dir, "_bad_inputs.txt")
    if os.path.exists(bad_inputs_file):
        with open(bad_inputs_file) as f:
            bad_paths = set(f.read().splitlines())
        if seg_path in bad_paths:
            return {"sample_id": sample_id, "group": group, "alpha": alpha, "subject": subject_id, "status": "skipped"}

    # Validate input file before expensive processing
    try:
        test_img = nib.load(seg_path)
        _ = np.asarray(test_img.dataobj).shape
    except Exception as e:
        # Record bad input so future runs skip it immediately
        with open(bad_inputs_file, "a") as f:
            f.write(seg_path + "\n")
        raise RuntimeError(f"Corrupted input segmentation {seg_path}: {e}") from e

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
        synthesize_builtin(synth_seg_path, t1_path, t2_path, seed=seed, contrast_mode=contrast_mode)
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
    out_dir,
    alphas,
    adni_labels_csv=None,
    synthesizer="builtin",
    synthseg_script=None,
    label_set=None,
    remap_to_fs=False,
    contrast_mode="paired",
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
                    contrast_mode,
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
        default=None,
        help="ADNI labels CSV with 'Subject' and 'Group' columns. "
        "If omitted, all subjects are auto-discovered from --seg-dir.",
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
        "--contrast-mode",
        choices=["paired", "random"],
        default="paired",
        help="Contrast mode for the builtin synthesizer. "
        "'paired' (default): view 1 = T1 priors, view 2 = FLAIR priors — "
        "the style axis captures real modality contrast differences. "
        "'random': both views get fully random SynthSeg-style contrast — "
        "maximum diversity but unstructured style.",
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
        out_dir=args.out_dir,
        alphas=args.alphas,
        adni_labels_csv=args.adni_labels,
        synthesizer=args.synthesizer,
        synthseg_script=args.synthseg_script,
        label_set=args.label_set,
        remap_to_fs=args.remap_to_fs,
        contrast_mode=args.contrast_mode,
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
