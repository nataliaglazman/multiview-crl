#!/usr/bin/env python
"""Rigidly register MUSE segmentations into MNI space before SynthSeg synthesis.

The real ADNI data (`/data/natalia/ADNI_registered`) is rigidly registered to the
ICBM152 linear template, landing every subject on the same (181, 217, 181) 1mm grid.
The MUSE segmentations used to drive SynthSeg synthesis are in *native* scanner space
(anisotropic, per-subject grids), so images synthesised from them do not match the
real data. This pre-pass fixes that:

  1. Register each subject's original T1 -> MNI (NiftyReg reg_aladin, RIGID only),
     reusing the exact recipe from `utils/helpers.py` / `utils/registration2.py`.
  2. Apply the resulting transform to the segmentation with NEAREST-NEIGHBOUR
     interpolation (reg_resample -inter NN), resampling it onto the MNI grid.
  3. Write the MNI-space segmentations in a layout `generate_synthseg_dataset.py`
     can discover directly.

Because synthesis inherits the segmentation grid, feeding these MNI-space segs into
`generate_synthseg_dataset.py` produces T1/FLAIR in the same space as the real data.

The transform is computed from the T1 (intensity) but applied to the segmentation,
which is valid only if the seg and T1 occupy the same native world space — MUSE segs
are computed in T1 space, so they do. The script verifies this per subject and skips
mismatches unless --force.

Usage:
  python scripts/register_segs_to_mni.py \\
      --seg-dir   /data/natalia/ADNI_MUSE_segs \\
      --t1-root   /data/natalia/ADNI_synthseg_originals_nifti \\
      --mni       /path/to/icbm_avg_152_t1_tal_lin.nii \\
      --out-dir   /data/natalia/ADNI_synthseg_segs_mni \\
      --adni-labels /data/natalia/labels_cleaned_3class.csv

Then synthesise as usual, pointing --seg-dir at the registered segs:
  python scripts/generate_synthseg_dataset.py \\
      --seg-dir   /data/natalia/ADNI_synthseg_segs_mni \\
      --adni-labels /data/natalia/labels_cleaned_3class.csv \\
      --out-dir   /data/natalia/ADNI_synthseg \\
      --alphas 0.0 0.2 0.4 0.6 0.8 1.0
"""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_synthseg_dataset import discover_subjects

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_t1(t1_root, subject_id):
    """Locate the original T1 for a subject under <t1_root>/<subject>/t1/.

    Prefers an MPRAGE series when several NIfTIs are present.
    """
    t1_dir = os.path.join(t1_root, subject_id, "t1")
    candidates = sorted(glob.glob(os.path.join(t1_dir, "*.nii")) + glob.glob(os.path.join(t1_dir, "*.nii.gz")))
    if not candidates:
        return None
    mprage = [c for c in candidates if "MPRAGE" in os.path.basename(c).upper()]
    return (mprage or candidates)[0]


def _world_bbox(img):
    """World-space (min, max) corner of an image's voxel grid."""
    n = np.array(img.shape[:3]) - 1
    corners = np.array([[x, y, z] for x in (0, n[0]) for y in (0, n[1]) for z in (0, n[2])], dtype=float)
    world = corners @ img.affine[:3, :3].T + img.affine[:3, 3]
    return world.min(axis=0), world.max(axis=0)


def check_same_world_space(seg_path, t1_path, tol=2.0):
    """Check the seg and T1 occupy the same physical space.

    NiftyReg resamples in world coordinates, so a different array *orientation*
    (axis flips/permutations) is harmless as long as both images map to the same
    physical bounding box — the common MUSE case. We therefore compare world
    bounding boxes, not affine matrices. Opposite handedness (a left/right mirror)
    is not safe to ignore and is reported separately.

    Returns (same_fov, same_handedness).
    """
    seg = nib.load(seg_path)
    t1 = nib.load(t1_path)
    smin, smax = _world_bbox(seg)
    tmin, tmax = _world_bbox(t1)
    same_fov = bool(np.allclose(smin, tmin, atol=tol) and np.allclose(smax, tmax, atol=tol))
    same_handedness = (np.linalg.det(seg.affine[:3, :3]) > 0) == (np.linalg.det(t1.affine[:3, :3]) > 0)
    return same_fov, same_handedness


def register_subject(subject_id, seg_path, t1_path, mni_path, out_dir, platform, force):
    """Register one subject's T1->MNI (rigid) and resample its seg onto the MNI grid (NN)."""
    from nipype.interfaces import niftyreg

    seg_out = os.path.join(out_dir, subject_id, f"{subject_id}_seg.nii.gz")
    if os.path.exists(seg_out):
        return "cached"

    same_fov, same_handedness = check_same_world_space(seg_path, t1_path)
    problem = None
    if not same_handedness:
        problem = "opposite handedness (possible left/right mirror)"
    elif not same_fov:
        problem = "different physical FOV (likely a different scan/timepoint)"
    if problem:
        if not force:
            logger.warning(f"{subject_id}: seg and T1 have {problem} — skipping (use --force to override)")
            return "grid_mismatch"
        logger.warning(f"{subject_id}: seg and T1 have {problem} — proceeding anyway (--force)")

    aff_path = os.path.join(out_dir, "_transforms", f"{subject_id}_t1_to_mni_aff.txt")
    t1_mni_path = os.path.join(out_dir, "_t1_in_mni", f"{subject_id}_T1_mni.nii.gz")
    for p in (seg_out, aff_path, t1_mni_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    # ── T1 -> MNI, rigid only (mirrors utils/helpers.py register_images) ──────────
    reg = niftyreg.RegAladin()
    reg.inputs.ref_file = mni_path  # fixed = MNI template
    reg.inputs.flo_file = t1_path  # moving = subject T1
    reg.inputs.res_file = t1_mni_path  # resampled T1 (kept for QC)
    reg.inputs.aff_file = aff_path
    reg.inputs.rig_only_flag = True
    reg.inputs.nac_flag = True
    reg.inputs.platform_val = platform
    reg.run()

    # ── Apply transform to the segmentation, NN (mirrors resample_mask_with_transform) ──
    res = niftyreg.RegResample()
    res.inputs.ref_file = mni_path
    res.inputs.flo_file = seg_path
    res.inputs.trans_file = aff_path
    res.inputs.inter_val = "NN"
    res.inputs.out_file = seg_out
    res.run()

    # ── Cast back to integer labels and confirm the MNI grid ──────────────────────
    out_img = nib.load(seg_out)
    labels = np.rint(np.asarray(out_img.dataobj)).astype(np.int32)
    nib.save(nib.Nifti1Image(labels, out_img.affine, out_img.header), seg_out)

    mni = nib.load(mni_path)
    if tuple(out_img.shape[:3]) != tuple(mni.shape[:3]) or not np.allclose(out_img.affine, mni.affine, atol=1e-2):
        logger.warning(
            f"{subject_id}: registered seg grid {out_img.shape[:3]} does not match MNI {mni.shape[:3]} — check output"
        )
    return "registered"


def main():
    parser = argparse.ArgumentParser(
        description="Rigidly register MUSE segmentations into MNI space before synthesis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--seg-dir", required=True, help="Directory with MUSE segmentations (as for generate_synthseg_dataset)."
    )
    parser.add_argument("--t1-root", required=True, help="Root with <subject>/t1/<subject>_*.nii original T1s.")
    parser.add_argument("--mni", required=True, help="ICBM152 linear template (icbm_avg_152_t1_tal_lin.nii).")
    parser.add_argument("--out-dir", required=True, help="Output directory for MNI-space segmentations.")
    parser.add_argument(
        "--adni-labels",
        default=None,
        help="ADNI labels CSV (Subject, Group). If omitted, subjects are auto-discovered from --seg-dir.",
    )
    parser.add_argument(
        "--platform",
        type=int,
        default=0,
        help="NiftyReg -platf backend: 0=CPU (default), 1=CUDA, 2=OpenCL. "
        "The original ADNI registration used 1; the resulting transform is identical, only speed differs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Register even when the seg and T1 affines disagree (different world spaces).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.mni):
        logger.error(f"MNI template not found: {args.mni}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    subjects = discover_subjects(args.seg_dir, args.adni_labels)
    if not subjects:
        logger.error("No subjects found. Check --seg-dir and --adni-labels.")
        sys.exit(1)

    counts = {"registered": 0, "cached": 0, "grid_mismatch": 0, "no_t1": 0, "failed": 0}
    for i, s in enumerate(subjects):
        subject_id = s["subject"]
        seg_path = s["seg_path"]
        t1_path = find_t1(args.t1_root, subject_id)
        if not t1_path:
            logger.warning(f"{subject_id}: no T1 under {os.path.join(args.t1_root, subject_id, 't1')} — skipping")
            counts["no_t1"] += 1
            continue
        try:
            status = register_subject(subject_id, seg_path, t1_path, args.mni, args.out_dir, args.platform, args.force)
            counts[status] += 1
        except Exception as e:
            logger.error(f"{subject_id}: registration failed: {e}")
            counts["failed"] += 1
        if (i + 1) % 25 == 0 or i + 1 == len(subjects):
            logger.info(f"  Progress: {i + 1}/{len(subjects)}  {counts}")

    logger.info(f"Done: {counts}")
    print(f"\nMNI-space segmentations in: {args.out_dir}")
    print("Now synthesise from them:")
    print(f"  python scripts/generate_synthseg_dataset.py \\")
    print(f"      --seg-dir {args.out_dir} \\")
    if args.adni_labels:
        print(f"      --adni-labels {args.adni_labels} \\")
    print(f"      --out-dir /data/natalia/ADNI_synthseg --alphas 0.0 0.2 0.4 0.6 0.8 1.0")


if __name__ == "__main__":
    main()
