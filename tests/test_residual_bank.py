"""Stage 37's residual options (out/burden_feasibility_20261102/residual_bank.py): --components and --source."""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import residual_bank as R

SIZES = {"train": 210, "val": 3, "test": 5}
KEYS = ("t1_04", "flair")


def tiny_bank(root, seed=0):
    """Scans with a few shared directions of variation plus noise, on an 8^3 grid, and the arrays the builder reads."""
    rng = np.random.default_rng(seed)
    modes = rng.standard_normal((6, 8, 8, 8))
    for split, n in SIZES.items():
        for key in KEYS:
            scans = 5 + np.tensordot(rng.standard_normal((n, 6)), modes, 1) + 0.3 * rng.standard_normal((n, 8, 8, 8))
            np.save(root / f"{split}_{key}.npy", scans.astype(np.float16))
        np.save(root / f"{split}_vox.npy", rng.uniform(10, 90, n).astype(np.float32))
        np.save(root / f"{split}_zc.npy", rng.standard_normal((n, 9)).astype(np.float32))


def build(*argv):
    with contextlib.redirect_stdout(io.StringIO()):
        R.main([str(a) for a in argv])


class ResidualBankTests(unittest.TestCase):
    def setUp(self):
        workers = R.WORKERS
        R.WORKERS = 1
        self.addCleanup(setattr, R, "WORKERS", workers)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.source = Path(folder.name) / "bank"
        self.source.mkdir()
        tiny_bank(self.source)
        build(self.source, *KEYS)

    def test_a_linked_bank_gets_its_own_residuals_and_leaves_the_source_alone(self):
        before = {key: np.load(self.source / f"test_{key}_res.npy").copy() for key in KEYS}
        view = self.source.parent / "bank_k3"
        build(view, *KEYS, "--components", 3, "--source", self.source)
        for key in KEYS:
            np.testing.assert_array_equal(np.load(self.source / f"test_{key}_res.npy"), before[key])
            residual = view / f"test_{key}_res.npy"
            self.assertFalse(residual.is_symlink())
            self.assertFalse(np.array_equal(np.load(residual), before[key]))
            for name in (f"test_{key}.npy", f"basis_{key}.npz"):
                self.assertTrue((view / name).is_symlink())
                self.assertFalse(os.path.isabs(os.readlink(view / name)))
            np.testing.assert_array_equal(np.load(view / f"train_{key}.npy"), np.load(self.source / f"train_{key}.npy"))
        np.testing.assert_array_equal(np.load(view / "test_vox.npy"), np.load(self.source / "test_vox.npy"))

    def test_the_default_removes_16_directions_and_more_directions_leave_less(self):
        scan = np.load(self.source / "test_flair.npy")[0]
        mask = np.ones(scan.shape, dtype=bool)
        data = np.load(self.source / "basis_flair.npz")
        left = {k: R.remove_anatomy(scan, mask, data["mean"], data["basis"][:, :k]) for k in (0, 3, 16)}
        self.assertLess(np.square(left[3]).sum(), np.square(left[0]).sum())
        self.assertLess(np.square(left[16]).sum(), np.square(left[3]).sum())
        scale = float(
            np.std(
                [
                    R.remove_anatomy(s, mask, data["mean"], data["basis"][:, :16])
                    for s in np.load(self.source / "train_flair.npy")[:200]
                ]
            )
        )
        stored = np.load(self.source / "test_flair_res.npy")[0].astype(np.float32)
        np.testing.assert_allclose(stored[mask], left[16] / scale, atol=2e-2)

    def test_more_directions_than_the_basis_holds_are_refused(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            build(self.source.parent / "bank_k65", *KEYS, "--components", 65, "--source", self.source)

    def test_the_source_must_be_another_bank(self):
        with self.assertRaises(SystemExit):
            build(self.source, *KEYS, "--source", self.source)


if __name__ == "__main__":
    unittest.main()
