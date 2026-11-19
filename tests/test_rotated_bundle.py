"""out/burden_feasibility_20261102/rotated_bundle.py: a bundle's codes rotated into their cross-view canonical
directions."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "out/burden_feasibility_20261102"))
import rotated_bundle

NOISE_SD = np.array([0.1, 0.5, 1.0])  # how noisily each view reads each shared factor


def views(seed=0, n=2000, width=10):
    """Two views sharing three factors read with different noise, so their canonical correlations differ, each mixed
    with view-specific noise units through its own random linear map."""
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal((n, 3))
    out = []
    for _ in range(2):
        read = shared + NOISE_SD * rng.standard_normal((n, 3))
        private = rng.standard_normal((n, width - 3))
        out.append(np.column_stack([read, private]) @ rng.standard_normal((width, width)))
    return shared, out[0], out[1]


class RotationTests(unittest.TestCase):
    def test_distinct_agreement_separates_the_factors(self):
        shared, view1, view2 = views()
        rotated1, rotated2, k, correlations = rotated_bundle.rotate(view1, view2)
        self.assertEqual(k, 3)
        self.assertTrue(np.all(np.diff(correlations) <= 1e-12))
        ceiling = 1 / np.sqrt(1 + NOISE_SD**2)  # the most a view's reading of factor j can correlate with it
        for j in range(3):  # strongest agreement first, so component j is factor j
            for rotated in (rotated1, rotated2):
                self.assertGreater(abs(np.corrcoef(rotated[:, j], shared[:, j])[0, 1]), ceiling[j] - 0.03)

    def rotate_file(self, *options):
        """Write a small bundle, rotate it with the script, and return the original factors and the rotated bundle."""
        shared, view1, view2 = views(seed=1, n=500)
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
            np.savez_compressed(
                source,
                z_content=shared.astype(np.float32),
                causal_adj=np.zeros((3, 3), dtype=np.float32),
                emb_view1=view1.astype(np.float32),
                emb_view2=view2.astype(np.float32),
                meta=json.dumps({"factor_digest": "abc", "n_rows": 500}),
            )
            script = ROOT / "out/burden_feasibility_20261102/rotated_bundle.py"
            subprocess.run(
                [sys.executable, str(script), str(source), str(target), *options],
                check=True,
                capture_output=True,
                env={"PYTHONPATH": str(ROOT)},
            )
            out = np.load(target)
            return shared, {key: out[key] for key in out.files}

    def test_the_rotated_bundle_keeps_its_rows(self):
        shared, out = self.rotate_file("--components", "4")
        meta = json.loads(str(out["meta"]))
        np.testing.assert_array_equal(out["z_content"], shared.astype(np.float32))
        self.assertEqual(out["emb_view1"].shape, (500, 4))
        self.assertEqual(out["emb_view2"].shape, (500, 4))
        self.assertEqual((meta["factor_digest"], meta["rotation"], meta["components"]), ("abc", "cca", 4))

    def test_all_pairs_is_a_pure_rotation(self):
        _, out = self.rotate_file("--all-pairs")
        self.assertEqual(out["emb_view1"].shape, (500, 10))
        self.assertEqual(json.loads(str(out["meta"]))["components"], 10)


if __name__ == "__main__":
    unittest.main()
