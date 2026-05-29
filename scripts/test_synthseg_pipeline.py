#!/usr/bin/env python
"""Integration test for the SynthSeg generation pipeline.

Creates tiny synthetic MUSE segmentation volumes in a temp directory, runs the
full pipeline end-to-end with the built-in synthesizer, and validates outputs.

Run on the cluster (or anywhere nibabel + scipy are installed):
  python scripts/test_synthseg_pipeline.py

Exits 0 on success, 1 on failure.
"""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SHAPE = (16, 16, 16)
AFFINE = np.eye(4) * 2.0
AFFINE[3, 3] = 1.0


def make_muse_seg(path, seed=0):
    """Create a tiny volume with realistic MUSE label layout."""
    rng = np.random.default_rng(seed)
    seg = np.zeros(SHAPE, dtype=np.int32)

    # Background = 0 by default. Fill a brain-shaped core.
    seg[2:14, 2:14, 2:14] = 42  # right cerebral cortex (exterior)

    # Subcortical structures in the center
    seg[6:10, 6:10, 6:10] = 47  # right hippocampus (MUSE)
    seg[6:10, 6:10, 10:14] = 48  # left hippocampus
    seg[5:7, 5:7, 7:9] = 31  # right amygdala
    seg[5:7, 9:11, 7:9] = 32  # left amygdala
    seg[4:6, 7:9, 7:9] = 51  # right lateral ventricle
    seg[10:12, 7:9, 7:9] = 52  # left lateral ventricle
    seg[7:9, 4:6, 7:9] = 116  # right entorhinal
    seg[7:9, 10:12, 7:9] = 117  # left entorhinal
    seg[3:5, 3:5, 3:5] = 81  # frontal WM right
    seg[3:5, 11:13, 3:5] = 82  # frontal WM left
    seg[12:14, 7:9, 7:9] = 46  # CSF
    seg[7:9, 7:9, 2:4] = 35  # brain stem

    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(seg, AFFINE), path)
    return path


def make_labels_csv(path, subjects):
    """Write a minimal labels CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Subject", "Group"])
        writer.writeheader()
        for subj, group in subjects:
            writer.writerow({"Subject": subj, "Group": group})


class Check:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, condition, msg):
        if condition:
            self.passed += 1
            print(f"  ✓ {msg}")
        else:
            self.failed += 1
            self.errors.append(msg)
            print(f"  ✗ {msg}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed")
        if self.errors:
            print("Failures:")
            for e in self.errors:
                print(f"  - {e}")
        return self.failed == 0


def test_label_detection():
    """Test auto-detection of MUSE vs FreeSurfer label sets."""
    from data.atrophy_simulator import detect_label_set

    c = Check()
    print("\n[1] Label set detection")

    muse_seg = np.zeros((8, 8, 8), dtype=np.int32)
    muse_seg[2:6, 2:6, 2:6] = 116
    c.ok(detect_label_set(muse_seg) == "muse", "Detects MUSE from entorhinal label 116")

    fs_seg = np.zeros((8, 8, 8), dtype=np.int32)
    fs_seg[2:6, 2:6, 2:6] = 17
    c.ok(detect_label_set(fs_seg) == "freesurfer", "Detects FreeSurfer from hippocampus label 17")

    empty_seg = np.zeros((8, 8, 8), dtype=np.int32)
    c.ok(detect_label_set(empty_seg) == "freesurfer", "Falls back to FreeSurfer on empty seg")

    return c


def test_muse_configs():
    """Test that MUSE configs use correct label IDs."""
    from data.atrophy_simulator import get_configs, get_vent_labels, get_wm_labels

    c = Check()
    print("\n[2] MUSE config construction")

    configs = get_configs("muse")
    c.ok(len(configs) == 6, f"6 structure configs (got {len(configs)})")

    hippo_labels = {cfg.label for cfg in configs[:2]}
    c.ok(hippo_labels == {47, 48}, f"Hippocampus labels are 47/48 (got {hippo_labels})")

    amyg_labels = {cfg.label for cfg in configs[2:4]}
    c.ok(amyg_labels == {31, 32}, f"Amygdala labels are 31/32 (got {amyg_labels})")

    ent_labels = {cfg.label for cfg in configs[4:6]}
    c.ok(ent_labels == {116, 117}, f"Entorhinal labels are 116/117 (got {ent_labels})")

    wm = get_wm_labels("muse")
    c.ok(len(wm) == 8, f"MUSE WM has 8 lobar labels (got {len(wm)})")
    c.ok(set(wm) == {81, 82, 83, 84, 85, 86, 87, 88}, f"WM labels correct (got {wm})")

    vent = get_vent_labels("muse")
    c.ok(vent == (52, 51), f"Ventricle labels are (52, 51) (got {vent})")

    return c


def test_muse_to_fs_remap():
    """Test MUSE → FreeSurfer remapping."""
    from data.atrophy_simulator import remap_muse_to_freesurfer

    c = Check()
    print("\n[3] MUSE → FreeSurfer remapping")

    seg = np.zeros((4, 4, 4), dtype=np.int32)
    seg[0, 0, 0] = 47  # MUSE right hippocampus
    seg[1, 0, 0] = 48  # MUSE left hippocampus
    seg[0, 1, 0] = 51  # MUSE right lateral ventricle
    seg[0, 0, 1] = 82  # MUSE frontal WM left
    seg[1, 1, 0] = 116  # MUSE right entorhinal
    seg[2, 0, 0] = 46  # MUSE CSF

    remapped = remap_muse_to_freesurfer(seg)

    c.ok(remapped[0, 0, 0] == 53, f"R hippocampus: MUSE 47 → FS 53 (got {remapped[0, 0, 0]})")
    c.ok(remapped[1, 0, 0] == 17, f"L hippocampus: MUSE 48 → FS 17 (got {remapped[1, 0, 0]})")
    c.ok(remapped[0, 1, 0] == 43, f"R lat vent: MUSE 51 → FS 43 (got {remapped[0, 1, 0]})")
    c.ok(remapped[0, 0, 1] == 2, f"L frontal WM: MUSE 82 → FS 2 (got {remapped[0, 0, 1]})")
    c.ok(remapped[1, 1, 0] == 42, f"R entorhinal → FS R cortex 42 (got {remapped[1, 1, 0]})")
    c.ok(remapped[2, 0, 0] == 24, f"CSF: MUSE 46 → FS 24 (got {remapped[2, 0, 0]})")

    return c


def test_atrophy_simulation():
    """Test erosion on a tiny MUSE segmentation."""
    from data.atrophy_simulator import simulate_atrophy

    c = Check()
    print("\n[4] Atrophy simulation (MUSE)")

    with tempfile.TemporaryDirectory() as tmp:
        seg_path = make_muse_seg(os.path.join(tmp, "test_seg.nii.gz"))
        out_path = os.path.join(tmp, "atrophied.nii.gz")

        original = nib.load(seg_path).get_fdata().astype(np.int32)
        orig_hippo_vol = int((original == 47).sum() + (original == 48).sum())

        result = simulate_atrophy(
            seg_path,
            alpha=0.8,
            label_set="muse",
            smooth_sigma=0,
            out_path=out_path,
        )
        atrophied = np.asarray(result.dataobj).astype(np.int32)
        new_hippo_vol = int((atrophied == 47).sum() + (atrophied == 48).sum())

        c.ok(os.path.exists(out_path), "Output file created")
        c.ok(new_hippo_vol < orig_hippo_vol, f"Hippocampus volume decreased: {orig_hippo_vol} → {new_hippo_vol}")
        c.ok(atrophied.shape == original.shape, "Output shape matches input")

        # alpha=0 should be a no-op
        noop = simulate_atrophy(seg_path, alpha=0.0, label_set="muse", smooth_sigma=0)
        noop_arr = np.asarray(noop.dataobj).astype(np.int32)
        c.ok(np.array_equal(noop_arr, original), "alpha=0 is identity")

    return c


def test_full_pipeline():
    """End-to-end: fake subjects → generate_dataset → verify output layout."""
    from scripts.generate_synthseg_dataset import generate_dataset

    c = Check()
    print("\n[5] Full pipeline (MUSE + builtin synthesizer)")

    with tempfile.TemporaryDirectory() as tmp:
        seg_dir = os.path.join(tmp, "segs")
        out_dir = os.path.join(tmp, "dataset")

        subjects = [("SUBJ_001", "CN"), ("SUBJ_002", "AD")]
        for subj, _ in subjects:
            make_muse_seg(os.path.join(seg_dir, f"{subj}.nii.gz"), seed=hash(subj) % 2**31)

        labels_csv = os.path.join(tmp, "labels.csv")
        make_labels_csv(labels_csv, subjects)

        alphas = [0.0, 0.5, 1.0]
        labels_path = generate_dataset(
            seg_dir=seg_dir,
            adni_labels_csv=labels_csv,
            out_dir=out_dir,
            alphas=alphas,
            synthesizer="builtin",
            label_set="muse",
            smooth_sigma=0,
            seed=42,
            num_workers=1,
        )

        c.ok(labels_path is not None, "generate_dataset returned labels path")
        c.ok(os.path.exists(labels_path), "labels.csv exists")

        # Check labels CSV content
        with open(labels_path) as f:
            rows = list(csv.DictReader(f))
        expected_n = len(subjects) * len(alphas)
        c.ok(len(rows) == expected_n, f"labels.csv has {expected_n} rows (got {len(rows)})")
        c.ok(
            all(k in rows[0] for k in ("Subject", "Group", "alpha", "original_subject")),
            "labels.csv has all expected columns",
        )

        # Check directory structure is MyCustomDataset-compatible
        for subj, _ in subjects:
            for alpha in alphas:
                sample_id = f"{subj}_alpha{alpha:.2f}"
                t1 = os.path.join(out_dir, sample_id, "t1", "synth_T1.nii.gz")
                t2 = os.path.join(out_dir, sample_id, "t2", "synth_FLAIR.nii.gz")
                c.ok(os.path.exists(t1), f"{sample_id}/t1/synth_T1.nii.gz exists")
                c.ok(os.path.exists(t2), f"{sample_id}/t2/synth_FLAIR.nii.gz exists")

        # Check T1 and T2 have different intensities (different random seeds)
        s0 = f"SUBJ_001_alpha{alphas[0]:.2f}"
        t1_data = nib.load(os.path.join(out_dir, s0, "t1", "synth_T1.nii.gz")).get_fdata()
        t2_data = nib.load(os.path.join(out_dir, s0, "t2", "synth_FLAIR.nii.gz")).get_fdata()
        c.ok(not np.allclose(t1_data, t2_data), "T1 and T2 have different intensities")

        # Check manifest
        manifest_path = os.path.join(out_dir, "manifest.json")
        c.ok(os.path.exists(manifest_path), "manifest.json exists")
        with open(manifest_path) as f:
            manifest = json.load(f)
        c.ok(manifest["label_set"] == "muse", f"manifest records label_set=muse")
        c.ok(manifest["n_samples"] == expected_n, f"manifest sample count correct")

        # Check resume: running again should cache everything
        labels_path_2 = generate_dataset(
            seg_dir=seg_dir,
            adni_labels_csv=labels_csv,
            out_dir=out_dir,
            alphas=alphas,
            synthesizer="builtin",
            label_set="muse",
            smooth_sigma=0,
            seed=42,
            num_workers=1,
        )
        c.ok(labels_path_2 is not None, "Re-run completes (resume/cache works)")

    return c


def test_freesurfer_path():
    """Verify FreeSurfer label set still works."""
    from data.atrophy_simulator import simulate_atrophy

    c = Check()
    print("\n[6] FreeSurfer label set (regression)")

    with tempfile.TemporaryDirectory() as tmp:
        seg = np.zeros(SHAPE, dtype=np.int32)
        seg[2:14, 2:14, 2:14] = 3  # FS left cortex
        seg[6:10, 6:10, 6:10] = 17  # FS left hippocampus
        seg[4:6, 7:9, 7:9] = 4  # FS left lateral ventricle
        seg[12:14, 7:9, 7:9] = 24  # FS CSF

        seg_path = os.path.join(tmp, "fs_seg.nii.gz")
        nib.save(nib.Nifti1Image(seg, AFFINE), seg_path)

        result = simulate_atrophy(seg_path, alpha=0.5, label_set="freesurfer", smooth_sigma=0)
        arr = np.asarray(result.dataobj).astype(np.int32)

        orig_hippo = int((seg == 17).sum())
        new_hippo = int((arr == 17).sum())
        c.ok(new_hippo < orig_hippo, f"FS hippocampus eroded: {orig_hippo} → {new_hippo}")
        c.ok(arr.shape == seg.shape, "Output shape preserved")

    return c


def main():
    all_checks = []

    all_checks.append(test_label_detection())
    all_checks.append(test_muse_configs())
    all_checks.append(test_muse_to_fs_remap())
    all_checks.append(test_atrophy_simulation())
    all_checks.append(test_full_pipeline())
    all_checks.append(test_freesurfer_path())

    total_pass = sum(c.passed for c in all_checks)
    total_fail = sum(c.failed for c in all_checks)
    total = total_pass + total_fail

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_pass}/{total} passed, {total_fail} failed")

    if total_fail > 0:
        print("\nFailed checks:")
        for c in all_checks:
            for e in c.errors:
                print(f"  - {e}")
        sys.exit(1)
    else:
        print("\nAll tests passed ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
