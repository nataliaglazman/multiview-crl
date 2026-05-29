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
from data.atrophy_simulator import detect_label_set, remap_muse_to_freesurfer, simulate_atrophy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Segmentation discovery ────────────────────────────────────────────────────


def find_seg(seg_dir, subject_id):
    """Locate segmentation for a subject.

    Checks (in order): ADNI MUSE download layout, FreeSurfer layout, flat layout.

    ADNI MUSE layout on LONI is:
      <seg_dir>/ADNI/<PTID>/T1_MUSE_segmentation/<date>/<IMAGEUID>[.nii.gz]
    We take the most recent date if multiple timepoints exist.
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
                # Try NIfTI files first, then bare image ID files
                for pattern in ["*.nii.gz", "*.nii", "*.mgz", "I*"]:
                    hits = glob.glob(os.path.join(session_dir, pattern))
                    if hits:
                        return sorted(hits)[0]

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
            return c
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
    """Simple label-to-image synthesis without SynthSeg.

    Assigns random Gaussian intensities per label, smooths, adds bias field
    and Rician noise. Good enough for pipeline testing; use SynthSeg for
    publication-quality synthesis.
    """
    img = nib.load(seg_path)
    seg = np.asarray(img.dataobj).astype(np.int32)
    rng = np.random.default_rng(seed)

    labels = np.unique(seg)
    intensity_map = {}
    for lbl in labels:
        if lbl == 0:
            intensity_map[lbl] = 0.0
        else:
            intensity_map[lbl] = rng.uniform(20, 255)

    synth = np.zeros_like(seg, dtype=np.float32)
    for lbl in labels:
        mask = seg == lbl
        if lbl == 0:
            continue
        mean_i = intensity_map[lbl]
        std_i = mean_i * 0.05
        synth[mask] = rng.normal(mean_i, max(std_i, 1.0), size=int(mask.sum()))

    from scipy.ndimage import gaussian_filter

    synth = gaussian_filter(synth, sigma=0.8)

    shape = synth.shape
    n_control = 4
    bias_coeff = rng.normal(0, 0.02, size=(3, n_control, n_control, n_control))
    from scipy.ndimage import zoom

    bias_field_low = np.zeros(shape, dtype=np.float32)
    for ax in range(3):
        zoomed = zoom(bias_coeff[ax], np.array(shape) / n_control, order=3)
        if zoomed.shape != tuple(shape):
            zoomed = zoomed[: shape[0], : shape[1], : shape[2]]
        bias_field_low += zoomed
    bias_field = np.exp(bias_field_low)
    synth *= bias_field

    noise_std = rng.uniform(1, 5)
    noise_r = rng.normal(0, noise_std, size=shape)
    noise_i = rng.normal(0, noise_std, size=shape)
    synth = np.sqrt((synth + noise_r) ** 2 + noise_i**2)

    synth = np.clip(synth, 0, None)
    brain_mask = seg > 0
    if brain_mask.any():
        fg = synth[brain_mask]
        p99 = np.percentile(fg, 99)
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
    """Process one (subject, alpha) pair: atrophy → remap → synthesis."""
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

    # Remap MUSE → FreeSurfer labels before synthesis so SynthSeg intensity
    # priors match the expected label conventions.
    synth_seg_path = atrophied_seg_path
    if label_set == "muse" and synthesizer != "builtin":
        remapped_path = os.path.join(seg_out_dir, f"{sample_id}_seg_fs.nii.gz")
        synth_seg_path = remap_seg_for_synthseg(atrophied_seg_path, remapped_path, label_set)

    seed = base_seed + hash(sample_id) % (2**31)

    if synthesizer == "builtin":
        synthesize_builtin(atrophied_seg_path, t1_path, t2_path, seed=seed)
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
