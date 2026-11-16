"""mlmv probes and config: the leakage probe recovers a known linear factor; config loading is strict."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from mlmv.config import load_config, parse_overrides, save_config
from mlmv.eval.metrics import cv_balanced_accuracy, cv_r2, effective_rank


class TestProbes(unittest.TestCase):
    def test_recovers_known_linear_factor(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((600, 16))
        y = x @ rng.standard_normal((16, 3)) + 0.05 * rng.standard_normal((600, 3))
        cfg = load_config().eval
        res = cv_r2(x, y, np.arange(600), cfg, probes=["ridge", "mlp"])
        self.assertGreater(res["ridge"], 0.95)
        self.assertGreater(res["best"], 0.95)
        absent = cv_r2(x, rng.standard_normal((600, 2)), np.arange(600), cfg, probes=["ridge"])
        self.assertLess(absent["best"], 0.05)

    def test_grouped_folds_and_balanced_accuracy(self):
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 3, 300)
        x = np.eye(3)[labels] + 0.1 * rng.standard_normal((300, 3))
        self.assertGreater(cv_balanced_accuracy(x, labels, np.arange(300)), 0.95)
        self.assertLess(cv_balanced_accuracy(rng.standard_normal((300, 3)), labels, np.arange(300)), 0.5)

    def test_effective_rank(self):
        rng = np.random.default_rng(2)
        self.assertGreater(effective_rank(rng.standard_normal((4000, 8))), 7.5)
        low = rng.standard_normal((4000, 1)) @ rng.standard_normal((1, 8))
        self.assertLess(effective_rank(low), 1.1)


class TestReadoutMasks(unittest.TestCase):
    def test_readouts_stay_inside_the_eroded_regions(self):
        from mlmv.data.geometry import build_geometry
        from mlmv.eval.leakage import readout_masks
        from mlmv.synth.three_level import SyntheticSource

        cfg = load_config(overrides={"synth": {"image": 32, "lattice": 8, "decoder_hidden": 32}})
        src = SyntheticSource(cfg.synth, stride=2)
        geo = build_geometry(src.feature_labels, 1.5, "chebyshev", 8, 64, log=lambda s: None, eligible_fraction=0.5)
        eroded = geo.eroded > 0
        m = readout_masks(src, geo)
        self.assertEqual(m["readout"], "eroded")
        # every candidate cell's 2x2x2 feature voxels are inside the eroded regions
        L, f = src.gen.L, 2
        for c in m["cell_candidates"].tolist():
            x, y, z = np.unravel_index(c, L)
            self.assertTrue(eroded[f * x : f * x + f, f * y : f * y + f, f * z : f * z + f].all())
        labels = src.feature_labels.reshape(-1)
        for j, r in enumerate(m["regions"]):
            w = m["region_weights"][j].numpy() > 0
            if r in m["fallback"]:
                self.assertTrue((w == (labels == r)).all())
            else:
                self.assertTrue((w <= ((labels == r) & eroded.reshape(-1))).all())
        self.assertTrue((m["subject_weights"].numpy() > 0).sum() == eroded.sum())
        # a region with nothing left after erosion falls back to its full extent
        geo.eroded = np.where(src.feature_labels == m["regions"][0], 0, geo.eroded)
        self.assertIn(m["regions"][0], readout_masks(src, geo)["fallback"])
        full = readout_masks(src, None)
        self.assertEqual(full["readout"], "full")
        self.assertEqual(len(full["cell_candidates"]), int(np.prod(L)))


class TestConfig(unittest.TestCase):
    def test_overrides_and_strictness(self):
        cfg = load_config(overrides=["optim.steps=10", "sampler.L.fraction=0.25", "loss.weights={G: 2.0}"])
        self.assertEqual(cfg.optim.steps, 10)
        self.assertEqual(cfg.sampler.L.fraction, 0.25)
        self.assertEqual(cfg.loss.weights, {"S": 1.0, "L": 1.0, "G": 2.0})
        with self.assertRaises(KeyError):
            load_config(overrides=["optim.stepz=1"])
        self.assertEqual(parse_overrides(["a.b=[1, 2]"]), {"a": {"b": [1, 2]}})

    def test_save_roundtrip(self):
        cfg = load_config(overrides=["seed=3"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            save_config(cfg, path)
            raw = yaml.safe_load(path.read_text())
            self.assertIn("_commit", raw)
            raw.pop("_commit")
            (Path(tmp) / "run.yaml").write_text(yaml.safe_dump(raw))
            self.assertEqual(load_config(Path(tmp) / "run.yaml").to_dict(), cfg.to_dict())

    def test_run_configs_load(self):
        paths = sorted(Path(__file__).resolve().parents[1].joinpath("mlmv/config/runs").rglob("*.yaml"))
        self.assertGreaterEqual(len(paths), 13)
        for path in paths:
            cfg = load_config(path)
            if path.parent.name.startswith("phase1"):
                self.assertEqual(cfg.run.name, path.stem)  # the job names its output directory after the file


if __name__ == "__main__":
    unittest.main()
