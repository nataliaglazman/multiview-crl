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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_synthseg_dataset import _builtin_synthesize_one, discover_subjects

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


def _recenter_to_origin(in_path, out_path):
    """Rewrite an image's affine so its volume centre sits at world (0, 0, 0).

    Keeps the orientation/spacing (the 3x3) but discards the (sometimes bogus) MUSE
    origin, so reg_aladin starts within capture range of the MNI template.
    """
    img = nib.load(in_path)
    rot = img.affine[:3, :3]
    vox_centre = (np.array(img.shape[:3]) - 1) / 2.0
    new_affine = img.affine.copy()
    new_affine[:3, 3] = -rot @ vox_centre
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), new_affine, img.header), out_path)
    return out_path


def register_subject(subject_id, seg_path, t1_path, mni_path, out_dir, platform, force, omp=0, register_via="t1"):
    """Register a subject's segmentation onto the MNI grid (rigid).

    register_via='t1'    — register the real T1 -> MNI and apply the transform to the
                           seg (requires seg and T1 to share world space).
    register_via='synth' — synthesise a pseudo-T1 from the seg itself, register that
                           -> MNI, and apply to the seg. Immune to seg/T1 header
                           mismatches (flips, bogus origins) and needs no external T1.
    """
    from nipype.interfaces import niftyreg

    seg_out = os.path.join(out_dir, subject_id, f"{subject_id}_seg.nii.gz")
    if os.path.exists(seg_out):
        return "cached"

    aff_path = os.path.join(out_dir, "_transforms", f"{subject_id}_aff.txt")
    qc_path = os.path.join(out_dir, "_qc_in_mni", f"{subject_id}_mni.nii.gz")
    for p in (seg_out, aff_path, qc_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    if register_via == "synth":
        # Pseudo-T1 built from the seg shares the seg's exact affine, so the transform
        # always applies cleanly. Recenter first so the (often bogus) origin can't blow
        # past reg_aladin's capture range.
        work = os.path.join(out_dir, "_work")
        seg_moving = _recenter_to_origin(seg_path, os.path.join(work, f"{subject_id}_seg_c.nii.gz"))
        moving = os.path.join(work, f"{subject_id}_pseudoT1.nii.gz")
        _builtin_synthesize_one(seg_moving, moving, seed=0, contrast="t1")
    else:
        same_fov, same_handedness = check_same_world_space(seg_path, t1_path)
        problem = None
        if not same_handedness:
            problem = "opposite handedness (possible left/right mirror)"
        elif not same_fov:
            problem = "different physical FOV (likely a different scan/timepoint)"
        if problem:
            if not force:
                logger.warning(f"{subject_id}: seg and T1 have {problem} — skipping (--force or --register-via synth)")
                return "grid_mismatch"
            logger.warning(f"{subject_id}: seg and T1 have {problem} — proceeding anyway (--force)")
        moving = t1_path
        seg_moving = seg_path

    # ── rigid registration: moving -> MNI (mirrors utils/helpers.py register_images) ──
    reg = niftyreg.RegAladin()
    reg.inputs.ref_file = mni_path
    reg.inputs.flo_file = moving
    reg.inputs.res_file = qc_path
    reg.inputs.aff_file = aff_path
    reg.inputs.rig_only_flag = True
    reg.inputs.nac_flag = True
    reg.inputs.platform_val = platform
    if omp > 0:
        reg.inputs.omp_core_val = omp
    reg.run()

    # ── QC: explicit LIN resample of the moving image into MNI ──
    # (reg_aladin's own res_file is unreliable across NiftyReg builds; this guarantees it.)
    res_qc = niftyreg.RegResample()
    res_qc.inputs.ref_file = mni_path
    res_qc.inputs.flo_file = moving
    res_qc.inputs.trans_file = aff_path
    res_qc.inputs.inter_val = "LIN"
    res_qc.inputs.out_file = qc_path
    if omp > 0:
        res_qc.inputs.omp_core_val = omp
    res_qc.run()

    # ── apply the transform to the segmentation, NN (mirrors resample_mask_with_transform) ──
    res = niftyreg.RegResample()
    res.inputs.ref_file = mni_path
    res.inputs.flo_file = seg_moving
    res.inputs.trans_file = aff_path
    res.inputs.inter_val = "NN"
    res.inputs.out_file = seg_out
    if omp > 0:
        res.inputs.omp_core_val = omp
    res.run()

    # ── cast back to integer labels and confirm the MNI grid ──
    out_img = nib.load(seg_out)
    labels = np.rint(np.asarray(out_img.dataobj)).astype(np.int32)
    nib.save(nib.Nifti1Image(labels, out_img.affine, out_img.header), seg_out)
    mni = nib.load(mni_path)
    if tuple(out_img.shape[:3]) != tuple(mni.shape[:3]) or not np.allclose(out_img.affine, mni.affine, atol=1e-2):
        logger.warning(f"{subject_id}: registered seg grid {out_img.shape[:3]} != MNI {mni.shape[:3]} — check output")
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
    parser.add_argument(
        "--t1-root",
        default=None,
        help="Root with <subject>/t1/<subject>_*.nii original T1s. Required for --register-via t1.",
    )
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
        help="Register even when the seg and T1 occupy different physical space "
        "(different FOV or opposite handedness).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel registration processes (CPU only; use with --platform 0). Each NiftyReg job "
        "is capped to (cpu_count // num_workers) OpenMP threads to avoid over-subscription.",
    )
    parser.add_argument(
        "--register-via",
        choices=["t1", "synth"],
        default="t1",
        help="'t1' (default): register the real T1 and apply its transform to the seg. "
        "'synth': register a pseudo-T1 synthesised from the seg — robust to seg/T1 header "
        "mismatches (flips, bogus origins) and needs no --t1-root.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.mni):
        logger.error(f"MNI template not found: {args.mni}")
        sys.exit(1)

    if args.register_via == "t1" and not args.t1_root:
        logger.error("--t1-root is required for --register-via t1 (or use --register-via synth)")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    subjects = discover_subjects(args.seg_dir, args.adni_labels)
    if not subjects:
        logger.error("No subjects found. Check --seg-dir and --adni-labels.")
        sys.exit(1)

    # Threads per NiftyReg process: cap when parallelising so workers don't over-subscribe.
    omp = max(1, (os.cpu_count() or 1) // args.num_workers) if args.num_workers > 1 else 0

    counts = {"registered": 0, "cached": 0, "grid_mismatch": 0, "no_t1": 0, "failed": 0}
    work = []
    for s in subjects:
        subject_id = s["subject"]
        t1_path = None
        if args.register_via == "t1":
            t1_path = find_t1(args.t1_root, subject_id)
            if not t1_path:
                logger.warning(f"{subject_id}: no T1 under {os.path.join(args.t1_root, subject_id, 't1')} — skipping")
                counts["no_t1"] += 1
                continue
        work.append(
            (
                subject_id,
                s["seg_path"],
                t1_path,
                args.mni,
                args.out_dir,
                args.platform,
                args.force,
                omp,
                args.register_via,
            )
        )

    logger.info(
        f"Registering {len(work)} subjects via {args.register_via} "
        f"({args.num_workers} worker(s), omp={omp or 'default'})"
    )

    done = 0
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {pool.submit(register_subject, *w): w[0] for w in work}
            for fut in as_completed(futures):
                subject_id = futures[fut]
                try:
                    counts[fut.result()] += 1
                except Exception as e:
                    logger.error(f"{subject_id}: registration failed: {e}")
                    counts["failed"] += 1
                done += 1
                if done % 25 == 0 or done == len(work):
                    logger.info(f"  Progress: {done}/{len(work)}  {counts}")
    else:
        for w in work:
            subject_id = w[0]
            try:
                counts[register_subject(*w)] += 1
            except Exception as e:
                logger.error(f"{subject_id}: registration failed: {e}")
                counts["failed"] += 1
            done += 1
            if done % 25 == 0 or done == len(work):
                logger.info(f"  Progress: {done}/{len(work)}  {counts}")

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
