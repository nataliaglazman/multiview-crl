"""adni_bank.py: the ADNI export that warm.py --real reads, on fake NIfTI trees. The split must be the trainers' own
(MyCustomDataset's), the exclusion rule must drop exactly the pairs it names, and each kept scan must be its brain's
block means, z-scored, inside one fixed box. The N4 test needs SimpleITK and is skipped without it."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "out/burden_feasibility_20261102"))
sys.path.insert(0, str(ROOT))
import adni_bank  # noqa: E402

GROUPS = ("CN", "MCI", "AD")
VOXEL = 4.0  # mm: the 30 x 36 x 30 grid spans 120 x 144 x 120 mm, so a normal fake brain clears the 300 mL rule
SHAPE = (30, 36, 30)
AFFINE = np.diag([VOXEL, VOXEL, VOXEL, 1.0])
SPLITS = ("train", "val", "test")


def ellipsoid(radii_mm, shift=(0, 0, 0)):
    axes = [(np.arange(n) - (n - 1) / 2 - s) * VOXEL for n, s in zip(SHAPE, shift)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return (x / radii_mm[0]) ** 2 + (y / radii_mm[1]) ** 2 + (z / radii_mm[2]) ** 2 <= 1


def fake_tree(root, subjects=12, bad=(), tiny=(), odd_grid=()):
    """<root>/ADNI_stripped_masks/<subject>/{t1,t2} with ADNI-style file names and masks alongside, plus labels.csv.
    ``bad`` subjects get a FLAIR mask moved 6 voxels (Dice < .9), ``tiny`` ones a 130 mL brain, ``odd_grid`` ones a
    FLAIR on another origin. Brains reach the box's y edge, so some volume falls outside a 16-voxel box at 8 mm."""
    rng = np.random.default_rng(0)
    data = Path(root) / "ADNI_stripped_masks"
    rows = []
    for i in range(subjects):
        subject = f"{100 + i:03d}_S_{1000 + i:04d}"
        radii = (30, 35, 30) if i in tiny else (50 + 2 * (i % 3), 70, 50)
        masks = {"t1": ellipsoid(radii), "t2": ellipsoid(radii, shift=(6, 0, 0) if i in bad else (0, 0, 0))}
        for view, name, series in (("t1", "MPRAGE", 1000), ("t2", "Sagittal_3D_FLAIR", 2000)):
            folder = data / subject / view
            folder.mkdir(parents=True)
            gradient = np.linspace(0.8, 1.2, SHAPE[0])[:, None, None]
            tissue = (100 if view == "t1" else 60) * gradient + 5 * rng.standard_normal(SHAPE)
            affine = AFFINE.copy()
            if view == "t2" and i in odd_grid:
                affine[0, 3] = 2.0
            stem = f"ADNI_{subject}_MR_{name}_br_raw_201901{10 + i % 20:02d}093000123_S{series + i}_I{20000 + i}"
            mask = masks[view].astype(np.float32)
            nib.save(nib.Nifti1Image((tissue * mask).astype(np.float32), affine), folder / f"{stem}.nii.gz")
            nib.save(nib.Nifti1Image(mask, affine), folder / f"{stem}_brain_mask.nii.gz")
        rows.append(f"{subject},{GROUPS[i % 3]}")
    labels = Path(root) / "labels.csv"
    labels.write_text("Subject,Group\n" + "\n".join(rows) + "\n")
    return data, labels


def export(data, labels, bank, *extra):
    with mock.patch.object(adni_bank, "WORKERS", 1):
        adni_bank.main(
            [str(bank), "--data-dir", str(data), "--labels", str(labels), "--bias-correction", "none"]
            + ["--spacing", "8", "--res", "16", *extra]
        )


def manifest(bank):
    with open(Path(bank) / "manifest.csv") as stream:
        return list(csv.DictReader(stream))


class PieceTests(unittest.TestCase):
    def test_identifiers_from_adni_and_dcm2niix_names(self):
        cases = {
            "ADNI_002_S_0413_MR_MPRAGE_br_raw_20060607123456789_S14781_I15799.nii.gz": ("15799", "14781", "2006-06-07"),
            "ADNI_941_S_7106_MR_Sagittal_3D_FLAIR_20230119_S1234567_I9876543_brain.nii.gz": (
                "9876543",
                "1234567",
                "2023-01-19",
            ),
            "_ADNI_12M4_TS_2_20060502122933_3.nii": ("", "", "2006-05-02"),
            "FLAIR.nii.gz": ("", "", ""),
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(adni_bank.identifiers(f"/data/{name}"), expected)

    def test_block_mean_averages_brain_voxels_only(self):
        volume = np.arange(6**3, dtype=np.float32).reshape(6, 6, 6)
        mask = np.zeros((6, 6, 6), dtype=bool)
        mask[:3, :3, :2] = True  # 18 of the first block's 27 voxels: brain
        mask[3:, 3:, 3] = True  # 9 of the last block's 27: not brain
        out, brain = adni_bank.block_mean(volume, mask, 3)
        self.assertEqual(out.shape, (2, 2, 2))
        self.assertTrue(brain[0, 0, 0])
        self.assertAlmostEqual(float(out[0, 0, 0]), float(volume[:3, :3, :2].mean()), places=4)
        self.assertFalse(brain[1, 1, 1])
        self.assertEqual(float(out[1, 1, 1]), 0.0)
        self.assertEqual(int(brain.sum()), 1)

    def test_block_mean_trims_the_remainder_from_both_ends(self):
        for n, first in ((8, 1.0), (7, (50 + 2) / 3)):  # 8 = 2 x 3 + 2: a voxel off each end; 7: the far end's
            volume = np.ones((n, 6, 6), dtype=np.float32)
            volume[0] = volume[-1] = 50.0
            out, _ = adni_bank.block_mean(volume, np.ones((n, 6, 6), dtype=bool), 3)
            with self.subTest(n=n):
                self.assertEqual(out.shape, (2, 2, 2))
                self.assertAlmostEqual(float(out[0, 0, 0]), first, places=4)
                self.assertAlmostEqual(float(out[1, 0, 0]), 1.0, places=4)

    def test_centre_box_crops_and_pads_about_the_centre(self):
        x = np.arange(5 * 10 * 4, dtype=np.float32).reshape(5, 10, 4) + 1
        box = adni_bank.centre_box(x, 8)
        self.assertEqual(box.shape, (8, 8, 8))
        np.testing.assert_array_equal(box[1:6, :, 2:6], x[:, 1:9, :])
        self.assertEqual(float(box[0].sum() + box[6:].sum() + box[:, :, :2].sum() + box[:, :, 6:].sum()), 0.0)


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data, cls.labels = fake_tree(cls.tmp.name, subjects=12, bad=(1,), tiny=(4,))
        cls.bank = Path(cls.tmp.name) / "bank"
        export(cls.data, cls.labels, cls.bank)
        cls.rows = manifest(cls.bank)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_split_is_the_trainers_own(self):
        from data.datasets import MyCustomDataset

        for mode in SPLITS:
            dataset = MyCustomDataset(
                data_dir=str(self.data),
                mode=mode,
                spatial_size=(96, 112, 96),
                labels_path=str(self.labels),
                masks_dir=str(self.data),
                val_frac=0.2,
                test_frac=0.1,
                split_seed=0,
            )
            with self.subTest(mode=mode):
                self.assertEqual(
                    sorted(it["subject"] for it in dataset.items),
                    sorted(r["subject"] for r in self.rows if r["split"] == mode),
                )

    def test_rule_drops_exactly_the_named_pairs(self):
        excluded = {r["subject"]: r["reason"] for r in self.rows if r["excluded"] == "1"}
        self.assertEqual(sorted(excluded), ["101_S_1001", "104_S_1004"])
        self.assertIn("mask Dice", excluded["101_S_1001"])
        self.assertIn("t1 mask", excluded["104_S_1004"])
        self.assertIn("flair mask", excluded["104_S_1004"])
        for r in self.rows:
            dice = float(r["mask_dice"])
            self.assertEqual(
                r["excluded"] == "1", dice < 0.9 or min(float(r["mask_ml_t1"]), float(r["mask_ml_flair"])) < 300
            )
        meta = json.loads((self.bank / "bank_meta.json").read_text())
        self.assertEqual(sum(c["excluded"] for c in meta["counts"].values()), 2)
        self.assertEqual(sum(c["kept"] for c in meta["counts"].values()), 10)

    def test_arrays_hold_z_scored_brains_in_row_order(self):
        for mode in SPLITS:
            kept = [r for r in self.rows if r["split"] == mode and r["excluded"] == "0"]
            ids = np.load(self.bank / f"{mode}_ids.npy")
            self.assertEqual(ids.tolist(), [r["subject"] for r in sorted(kept, key=lambda r: int(r["row"]))])
            for view in ("t1", "flair"):
                array = np.load(self.bank / f"{mode}_{view}.npy")
                self.assertEqual((array.shape, array.dtype), ((len(kept), 16, 16, 16), np.float16))
                for volume in array.astype(np.float32):
                    brain = volume != 0
                    self.assertGreater(int(brain.sum()), 500)
                    self.assertAlmostEqual(float(volume[brain].mean()), 0.0, places=2)
                    self.assertAlmostEqual(float(volume[brain].std()), 1.0, places=2)

    def test_one_scan_by_hand(self):
        row = next(r for r in self.rows if r["split"] == "train" and r["excluded"] == "0")
        folder = self.data / row["subject"] / "t1"
        image = nib.load(folder / row["t1_file"]).get_fdata(dtype=np.float32)
        mask = nib.load(next(folder.glob("*_brain_mask.nii.gz"))).get_fdata(dtype=np.float32) > 0.5
        means, brain = adni_bank.block_mean(image, mask, 2)
        box, kept = adni_bank.centre_box(means, 16), adni_bank.centre_box(brain, 16)
        expected = np.where(kept, (box - box[kept].mean()) / box[kept].std(), 0)
        stored = np.load(self.bank / "train_t1.npy")[int(row["row"])].astype(np.float32)
        np.testing.assert_allclose(stored, expected, atol=2e-2)
        self.assertGreater(float(row["cut_ml_t1"]), 0.0)  # the y extent overhangs the 16-voxel box

    def test_manifest_carries_scan_identifiers(self):
        row = next(r for r in self.rows if r["subject"] == "102_S_1002")
        self.assertEqual((row["t1_image_id"], row["flair_image_id"]), ("20002", "20002"))
        self.assertEqual((row["t1_series_id"], row["flair_series_id"]), ("1002", "2002"))
        self.assertEqual(row["t1_date"], "2019-01-12")

    def test_check_split_accepts_the_same_split_and_refuses_another(self):
        subjects = {mode: [r["subject"] for r in self.rows if r["split"] == mode] for mode in SPLITS}
        good = Path(self.tmp.name) / "good.json"
        good.write_text(json.dumps({"subjects": subjects, "label_map": {"AD": 0, "CN": 1, "MCI": 2}}))
        export(self.data, self.labels, Path(self.tmp.name) / "checked", "--check-split", str(good), "--qc-only")
        swapped = {mode: list(v) for mode, v in subjects.items()}
        swapped["train"][0], swapped["val"][0] = swapped["val"][0], swapped["train"][0]
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text(json.dumps({"subjects": swapped}))
        with self.assertRaises(SystemExit):
            export(self.data, self.labels, Path(self.tmp.name) / "refused", "--check-split", str(bad))

    def test_qc_only_writes_no_arrays(self):
        out = Path(self.tmp.name) / "qc"
        export(self.data, self.labels, out, "--qc-only")
        self.assertTrue((out / "manifest.csv").exists() and (out / "bank_meta.json").exists())
        self.assertFalse(list(out.glob("*.npy")))
        self.assertEqual(len(manifest(out)), 12)


class RefusalTests(unittest.TestCase):
    def test_a_second_grid_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, labels = fake_tree(tmp, subjects=6, odd_grid=(2,))
            with self.assertRaises(SystemExit) as raised:
                export(data, labels, Path(tmp) / "bank", "--qc-only")
            self.assertIn("one grid", str(raised.exception))

    def test_spacing_must_be_a_whole_multiple_of_the_scans(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, labels = fake_tree(tmp, subjects=6)
            with mock.patch.object(adni_bank, "WORKERS", 1), self.assertRaises(SystemExit):
                adni_bank.main(
                    [str(Path(tmp) / "bank"), "--data-dir", str(data), "--labels", str(labels)]
                    + ["--bias-correction", "none", "--spacing", "6", "--res", "16"]
                )


@unittest.skipUnless(__import__("importlib").util.find_spec("SimpleITK"), "SimpleITK is not installed")
class N4Tests(unittest.TestCase):
    def test_n4_flattens_a_smooth_field(self):
        grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, 48)] * 3, indexing="ij"))
        radius = np.sqrt((grid**2).sum(0))
        brain = radius < 0.9
        tissue = np.where(radius < 0.5, 100.0, 60.0) * brain
        field = 1 + 0.3 * grid[0] + 0.15 * grid[1] ** 2
        rng = np.random.default_rng(0)
        volume = ((tissue + rng.normal(0, 1, tissue.shape)) * field * brain).astype(np.float32)
        corrected = adni_bank.n4(volume, brain, (2.0, 2.0, 2.0))
        self.assertEqual(float(np.abs(corrected[~brain]).max()), 0.0)
        for inside in (radius < 0.45, (radius > 0.55) & brain):
            before = volume[inside].std() / volume[inside].mean()
            after = corrected[inside].std() / corrected[inside].mean()
            with self.subTest(before=round(float(before), 3), after=round(float(after), 3)):
                self.assertLess(after, 0.5 * before)


if __name__ == "__main__":
    unittest.main()
