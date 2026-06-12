#!/usr/bin/env python
"""Visual test of the SynthSeg pipeline on real ADNI MUSE segmentations.

Processes a few subjects, generates atrophied segmentations and synthetic
T1/T2 images, and saves PNG slice montages for visual inspection.

Usage:
  # Process one subject (auto-finds seg in ADNI MUSE layout):
  python scripts/test_synthseg_on_adni.py \
      --seg-dir /data/natalia/ADNI_segmentations \
      --subject 067_S_6980 \
      --out-dir /data/natalia/synthseg_test

  # Process first N subjects from a labels CSV:
  python scripts/test_synthseg_on_adni.py \
      --seg-dir /data/natalia/ADNI_segmentations \
      --adni-labels /data/natalia/labels_cleaned_3class.csv \
      --max-subjects 3 \
      --out-dir /data/natalia/synthseg_test

  # Custom alpha values and WMH:
  python scripts/test_synthseg_on_adni.py \
      --seg-dir /data/natalia/ADNI_segmentations \
      --subject 067_S_6980 \
      --alphas 0.0 0.3 0.6 1.0 \
      --wmh-fraction 0.05 \
      --out-dir /data/natalia/synthseg_test

Output:
  <out_dir>/
    067_S_6980/
      montage_segmentations.png   # original + atrophied segs side by side
      montage_synthetic.png       # generated T1/T2 at each alpha
      original_seg.nii.gz         # for viewing in ITK-SNAP / freeview
      alpha0.00_seg.nii.gz
      alpha0.00_T1.nii.gz
      alpha0.00_T2.nii.gz
      ...
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.atrophy_simulator import MUSE, detect_label_set, simulate_atrophy
from scripts.generate_synthseg_dataset import _builtin_synthesize_one, _stable_seed, find_seg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def plot_slices(volumes, titles, out_path, suptitle=None, cmap_seg="nipy_spectral", cmap_img="gray"):
    """Save a montage of axial/coronal/sagittal mid-slices for each volume.

    Parameters
    ----------
    volumes : list of (np.ndarray, str)
        Each entry is (3D array, type) where type is 'seg' or 'img'.
    titles : list of str
        Column title for each volume.
    out_path : str
        PNG output path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(volumes)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    if n == 1:
        axes = axes[:, np.newaxis]

    for col, ((vol, vtype), title) in enumerate(zip(volumes, titles)):
        mid = [s // 2 for s in vol.shape]
        slices = [
            vol[mid[0], :, :],  # axial
            vol[:, mid[1], :],  # coronal
            vol[:, :, mid[2]],  # sagittal
        ]
        slice_names = ["Axial", "Coronal", "Sagittal"]

        for row, (sl, sname) in enumerate(zip(slices, slice_names)):
            ax = axes[row, col]
            sl_2d = np.rot90(sl)
            if vtype == "seg":
                ax.imshow(sl_2d, cmap=cmap_seg, interpolation="nearest")
            else:
                ax.imshow(sl_2d, cmap=cmap_img, interpolation="bilinear")
            if row == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(sname, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


def process_subject(subject_id, seg_path, out_dir, alphas, wmh_fraction, smooth_sigma, seed):
    """Run atrophy + synthesis for one subject, save volumes and visualizations."""
    subj_dir = os.path.join(out_dir, subject_id)
    os.makedirs(subj_dir, exist_ok=True)

    # Load original
    img = nib.load(seg_path)
    orig_seg = np.asarray(img.dataobj).astype(np.int32)
    label_set = detect_label_set(orig_seg)
    logger.info(f"  Label set: {label_set}")
    logger.info(f"  Shape: {orig_seg.shape}, unique labels: {len(np.unique(orig_seg))}")

    # Check key structures are present
    L = MUSE if label_set == "muse" else None
    if L:
        for name, lbl in [
            ("L Hippo", L["LH_HIPPO"]),
            ("R Hippo", L["RH_HIPPO"]),
            ("L Vent", L["LH_LATERAL_VENT"]),
            ("R Vent", L["RH_LATERAL_VENT"]),
            ("L Entorhinal", L["LH_ENTORHINAL"]),
            ("R Entorhinal", L["RH_ENTORHINAL"]),
        ]:
            vol = int((orig_seg == lbl).sum())
            logger.info(f"    {name} (label {lbl}): {vol} voxels")

    # Save original for reference
    orig_out = os.path.join(subj_dir, "original_seg.nii.gz")
    nib.save(nib.Nifti1Image(orig_seg, img.affine, img.header), orig_out)

    # Generate atrophied segs + synthetic images at each alpha
    seg_volumes = [("Original", orig_seg)]
    synth_volumes = []

    for alpha in alphas:
        tag = f"alpha{alpha:.2f}"
        logger.info(f"  Processing alpha={alpha:.2f}")

        # Atrophy simulation
        seg_out_path = os.path.join(subj_dir, f"{tag}_seg.nii.gz")
        result = simulate_atrophy(
            seg_path,
            alpha=alpha,
            label_set=label_set,
            wmh_fraction=wmh_fraction,
            wmh_seed=_stable_seed(subject_id),
            smooth_sigma=smooth_sigma,
            out_path=seg_out_path,
        )
        atrophied = np.asarray(result.dataobj).astype(np.int32)
        seg_volumes.append((f"α={alpha:.1f}", atrophied))

        # Log volume changes
        if L:
            hippo_orig = int((orig_seg == L["LH_HIPPO"]).sum() + (orig_seg == L["RH_HIPPO"]).sum())
            hippo_new = int((atrophied == L["LH_HIPPO"]).sum() + (atrophied == L["RH_HIPPO"]).sum())
            vent_orig = int((orig_seg == L["LH_LATERAL_VENT"]).sum() + (orig_seg == L["RH_LATERAL_VENT"]).sum())
            vent_new = int((atrophied == L["LH_LATERAL_VENT"]).sum() + (atrophied == L["RH_LATERAL_VENT"]).sum())
            logger.info(f"    Hippocampus: {hippo_orig} → {hippo_new} voxels ({hippo_new - hippo_orig:+d})")
            logger.info(f"    Ventricles:  {vent_orig} → {vent_new} voxels ({vent_new - vent_orig:+d})")

        # Synthesize T1 and T2 directly from MUSE labels — lab2im and the
        # builtin synthesizer sample random intensities per label, so MUSE's
        # 152 ROIs give better cortical variation than remapping to FS's ~30.
        t1_path = os.path.join(subj_dir, f"{tag}_T1.nii.gz")
        t2_path = os.path.join(subj_dir, f"{tag}_T2.nii.gz")
        # Noise seed varies per alpha; contrast seed depends only on the subject
        # so contrast stays fixed across the atrophy spectrum (see _builtin_synthesize_one).
        s = seed + _stable_seed(f"{subject_id}_{tag}")
        cs = seed + _stable_seed(subject_id)
        _builtin_synthesize_one(seg_out_path, t1_path, seed=s, contrast="t1", contrast_seed=cs)
        _builtin_synthesize_one(
            seg_out_path, t2_path, seed=s + 1_000_000, contrast="flair", contrast_seed=cs + 1_000_000
        )

        t1 = nib.load(t1_path).get_fdata()
        t2 = nib.load(t2_path).get_fdata()
        synth_volumes.append((f"T1 α={alpha:.1f}", t1))
        synth_volumes.append((f"T2 α={alpha:.1f}", t2))

    # ── Plot segmentation montage ─────────────────────────────────────────
    plot_slices(
        [(v, "seg") for _, v in seg_volumes],
        [t for t, _ in seg_volumes],
        os.path.join(subj_dir, "montage_segmentations.png"),
        suptitle=f"{subject_id} — Atrophy Progression ({label_set} labels)",
    )

    # ── Plot synthetic image montage ──────────────────────────────────────
    plot_slices(
        [(v, "img") for _, v in synth_volumes],
        [t for t, _ in synth_volumes],
        os.path.join(subj_dir, "montage_synthetic.png"),
        suptitle=f"{subject_id} — Synthetic T1/T2 (builtin synthesizer)",
    )

    logger.info(f"  Done: {subj_dir}")
    return subj_dir


def main():
    parser = argparse.ArgumentParser(
        description="Visual test of the SynthSeg pipeline on real ADNI MUSE segmentations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seg-dir",
        required=True,
        help="Root of ADNI segmentations (e.g. /data/natalia/ADNI_segmentations).",
    )
    parser.add_argument(
        "--subject",
        type=str,
        nargs="+",
        default=None,
        help="One or more subject PTIDs to process (e.g. 067_S_6980).",
    )
    parser.add_argument(
        "--adni-labels",
        type=str,
        default=None,
        help="Labels CSV to auto-discover subjects (used if --subject not given).",
    )
    parser.add_argument("--max-subjects", type=int, default=3, help="Max subjects when using --adni-labels.")
    parser.add_argument("--out-dir", required=True, help="Output directory for test results.")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.4, 0.8],
        help="Atrophy severity levels (default: 0.0 0.4 0.8).",
    )
    parser.add_argument("--wmh-fraction", type=float, default=0.0)
    parser.add_argument("--smooth-sigma", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Resolve subjects
    subjects = []
    if args.subject:
        for s in args.subject:
            seg_path = find_seg(args.seg_dir, s)
            if seg_path:
                subjects.append((s, seg_path))
                logger.info(f"Found: {s} → {seg_path}")
            else:
                logger.error(f"No segmentation found for {s} in {args.seg_dir}")
    elif args.adni_labels:
        import pandas as pd

        df = pd.read_csv(args.adni_labels)
        col = "Subject" if "Subject" in df.columns else "PTID"
        for subj in df[col].unique()[: args.max_subjects]:
            seg_path = find_seg(args.seg_dir, str(subj))
            if seg_path:
                subjects.append((str(subj), seg_path))
                logger.info(f"Found: {subj} → {seg_path}")
    else:
        parser.error("Provide --subject or --adni-labels.")

    if not subjects:
        logger.error("No subjects with segmentations found.")
        sys.exit(1)

    logger.info(f"\nProcessing {len(subjects)} subjects, alphas={args.alphas}\n")

    for subj_id, seg_path in subjects:
        logger.info(f"[{subj_id}] seg={seg_path}")
        process_subject(
            subj_id,
            seg_path,
            args.out_dir,
            alphas=args.alphas,
            wmh_fraction=args.wmh_fraction,
            smooth_sigma=args.smooth_sigma,
            seed=args.seed,
        )

    print(f"\n{'='*60}")
    print(f"Results in: {args.out_dir}")
    print(f"View PNGs:")
    for subj_id, _ in subjects:
        d = os.path.join(args.out_dir, subj_id)
        print(f"  {d}/montage_segmentations.png")
        print(f"  {d}/montage_synthetic.png")
    print(f"\nView NIfTIs in ITK-SNAP / freeview:")
    print(f"  freeview {args.out_dir}/<subject>/original_seg.nii.gz")
    print(f"  freeview {args.out_dir}/<subject>/alpha0.80_T1.nii.gz")


if __name__ == "__main__":
    main()
