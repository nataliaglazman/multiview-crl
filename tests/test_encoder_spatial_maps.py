"""Final-stage map analysis: real CLI run, per-cell probes, survival and factor interventions."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from eval.encoder import encoder_spatial_maps as spatial_maps
from eval.encoder.encoder_target_protocol import dataset, digest
from eval.protocol.score_checkpoint import build_model
from tests.test_encoder_target_followups import config


class SpatialMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        cls.old_determinism = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.run_dir = cls.root / "run"
        cls.run_dir.mkdir()
        cfg = config()
        (cls.run_dir / "settings.json").write_text(json.dumps(cfg))
        model = build_model(cfg, "cpu")
        torch.save(model.state_dict(), cls.run_dir / "model_init.pt")
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
        torch.save(model.state_dict(), cls.run_dir / "model.pt")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.old_threads)
        torch.use_deterministic_algorithms(cls.old_determinism[0], warn_only=cls.old_determinism[1])

    def test_cli_writes_maps_tables_and_figures_without_touching_the_run(self):
        out = self.root / "maps"
        before = {p.name: digest(p) for p in self.run_dir.iterdir()}
        args = ["--run-dir", str(self.run_dir), "--out-dir", str(out), "--grid", "2", "--test-samples", "12"]
        args += ["--gallery-subjects", "2", "--intervention-subjects", "2", "--threads", "1", "--device", "cpu"]
        with contextlib.redirect_stdout(io.StringIO()):
            spatial_maps.main(args)
        self.assertEqual(before, {p.name: digest(p) for p in self.run_dir.iterdir()})
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["encoder_unchanged"])
        self.assertEqual(report["native_grid"], 4)  # res 16, stride 4
        for arm in ("trained", "initial"):
            self.assertEqual(
                report["cohorts"][arm]["test"]["input_sha256"], report["cohorts"]["trained"]["test"]["input_sha256"]
            )
        for name in ("gallery_trained.png", "gallery_initial.png", "responses_t1.png", "responses_flair.png"):
            self.assertTrue((out / name).exists(), name)
        for name in ("decodability_trained.png", "decodability_initial.png", "responses.csv", "decodability.csv"):
            self.assertTrue((out / name).exists(), name)
        self.assertFalse([p for p in out.iterdir() if p.name.startswith(".features-")])
        with (out / "responses.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2 * 2 * 9 * 4)  # arms x views x factors x stages
        for row in rows:
            if row["stage"] != "global" and row["gap_survival"]:
                self.assertLessEqual(0.0, float(row["gap_survival"]))
                self.assertLessEqual(float(row["gap_survival"]), 1.0 + 1e-9)
            if row["stage"] == "global":
                self.assertGreaterEqual(float(row["global_response"]), 0.0)
        with (out / "decodability.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2 * 2 * 2 * len(spatial_maps.TARGETS))
        with np.load(out / "decodability_maps.npz") as bank:
            self.assertEqual(bank["trained/t1/projected/ventricle_size"].shape, (2, 2, 2))
            self.assertEqual(len(bank.files), 2 * 2 * 2 * len(spatial_maps.TARGETS))
        with (out / "channel_mcc.csv").open() as stream:
            channels = list(csv.DictReader(stream))
        self.assertEqual(len(channels), 2 * 2 * 9)  # arms x views x content channels
        self.assertTrue(all(r["local_best_factor"] in spatial_maps.CONTENT_FACTOR_NAMES for r in channels))
        for arm in ("trained", "initial"):
            self.assertTrue((out / f"channel_maps_{arm}.png").exists())
            for view in ("t1", "flair"):
                matched = [r["global_factor"] for r in channels if (r["arm"], r["view"]) == (arm, view)]
                self.assertEqual(sorted(matched), sorted(spatial_maps.CONTENT_FACTOR_NAMES))  # one-to-one
        with np.load(out / "channel_maps.npz") as bank:
            self.assertEqual(len(bank.files), 2 * 2 * 9)
            self.assertLessEqual(np.abs(bank["trained/t1/channel1"]).max(), 1.0 + 1e-9)
        with self.assertRaises(FileExistsError):
            spatial_maps.main(args)

    def test_channel_matching_equals_channel_mcc_and_locates_local_encoding(self):
        rng = np.random.default_rng(3)
        factors = rng.normal(size=(300, 3))
        codes = np.column_stack((-factors[:, 2], factors[:, 0], factors[:, 1])) + 0.3 * rng.normal(size=(300, 3))
        assignment = spatial_maps.channel_assignment(codes, factors)
        self.assertEqual({c: f for c, (f, _) in assignment.items()}, {0: 2, 1: 0, 2: 1})
        official = spatial_maps.channel_mcc(codes, factors)["per_factor"]
        for f, r in assignment.values():
            self.assertAlmostEqual(official[f], r, places=12)
        grid = np.zeros((300, 4, 2))  # channel 0 tracks factor 1 only at cell 3; cell 0 is constant
        grid[:, 1:] = rng.normal(size=(300, 3, 2))
        grid[:, 3, 0] += 3 * factors[:, 1]
        r = spatial_maps.cell_correlations(grid, factors)
        self.assertEqual(r.shape, (4, 2, 3))
        self.assertEqual(int(np.abs(r[:, 0, 1]).argmax()), 3)
        self.assertGreater(abs(r[3, 0, 1]), 0.9)
        np.testing.assert_array_equal(r[0], 0.0)

    def test_cellwise_ridge_matches_sklearn_and_finds_the_informative_cell(self):
        rng = np.random.default_rng(0)
        fit, test = rng.normal(size=(200, 3, 4)), rng.normal(size=(100, 3, 4))
        weights = np.array([1.0, -2.0, 0.5, 0.0])
        y_fit = np.column_stack((fit[:, 1] @ weights + 0.1 * rng.normal(size=200), rng.normal(size=200)))
        y_test = np.column_stack((test[:, 1] @ weights + 0.1 * rng.normal(size=100), rng.normal(size=100)))
        scores = spatial_maps.cellwise_r2(fit, test, y_fit, y_test)
        self.assertGreater(scores[1, 0], 0.95)
        self.assertLess(max(scores[0, 0], scores[2, 0]), 0.1)
        self.assertLess(np.abs(scores[:, 1]).max(), 0.1)  # a pure-noise target stays near zero
        single = spatial_maps.cellwise_r2(fit, test, y_fit, y_test, alphas=(3.0,))
        mean, scale = fit[:, 1].mean(0), fit[:, 1].std(0)
        ridge = Ridge(alpha=3.0).fit((fit[:, 1] - mean) / scale, y_fit[:, 0])
        self.assertAlmostEqual(
            single[1, 0], r2_score(y_test[:, 0], ridge.predict((test[:, 1] - mean) / scale)), places=10
        )
        banks = np.arange(2 * 3 * 8).reshape(2, 3 * 8)  # (subjects, channel-major channels x 2³ cells)
        self.assertEqual(spatial_maps.cells(banks, 2)[1, 5, 2], banks[1, 2 * 8 + 5])

    def test_survival_separates_coherent_from_cancelling_changes(self):
        self.assertAlmostEqual(spatial_maps.survival(np.tile([[1.0, 2.0]], (64, 1))), 1.0)
        checker = np.where(np.indices((4, 4, 4)).sum(0).ravel() % 2 == 0, 1.0, -1.0)
        self.assertAlmostEqual(spatial_maps.survival(checker[:, None] * np.array([[1.0, 2.0]])), 0.0)
        self.assertTrue(np.isnan(spatial_maps.survival(np.zeros((8, 3)))))

    def test_factor_pairs_replay_the_input_and_change_only_their_factor(self):
        ds = dataset(config(res=32), 64, "test")
        inner = ds._inner
        own, _ = spatial_maps.render_factor_pair(ds, 0, 1, 0.0)
        for v in range(2):
            np.testing.assert_allclose(own[v].numpy(), ds[0]["image"][v].numpy(), atol=5e-6, rtol=0)
        lat = inner[0][2]
        for k in (0, 1, 5, 8):
            a, b = spatial_maps.render_factor_pair(ds, 0, k, 0.5)
            tissues = []
            for sign in (-1, 1):
                content = lat["z_content"].clone()
                content[k] += sign * 0.5
                tissues.append(
                    inner.renderer.render_structure(content, lat["z_deformation"], lat["z_fissure"], "cpu", clean=True)[
                        0
                    ]
                )
            changed = maximum_filter((tissues[0] != tissues[1]).numpy(), size=3)
            self.assertTrue(changed.any())
            for v in range(2):
                difference = (b[v] - a[v]).numpy()[0]
                self.assertLess(np.abs(difference[~changed]).max(), 2e-6)
                self.assertGreater(np.abs(difference[changed]).max(), 1e-3)
        a, b = spatial_maps.render_factor_pair(ds, 0, 3, 0.5)  # lesion y: delegated to render_pair
        self.assertEqual((len(a), len(b), tuple(a[0].shape)), (2, 2, (1, 32, 32, 32)))


if __name__ == "__main__":
    unittest.main()
