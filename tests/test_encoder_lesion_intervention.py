"""Controlled images, movement-aware scoring, probe split isolation and real checkpoints."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scipy.ndimage import maximum_filter

from eval.encoder import encoder_lesion_intervention as audit
from eval.encoder.encoder_target_protocol import dataset
from eval.lesion.checkpoint_lesion_analysis import state_digest
from eval.protocol.score_checkpoint import build_model
from tests.test_encoder_target_followups import config


def options(**changes):
    return SimpleNamespace(
        **{
            "num_samples": 3,
            "subject_offset": 0,
            "axes": ["x", "y", "z"],
            "eps": 0.5,
            "batch_size": 2,
            "grids": [1, 2],
            "include_native": True,
            "stages": ("backbone", "projected"),
            "seed": 1729,
            "bootstrap": 20,
            **changes,
        }
    )


class LesionInterventionTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_pair_only_changes_one_control_with_frozen_acquisition_and_normalization(self):
        cfg = config(res=32, synthetic_lesion_radius=0.14)
        ds = dataset(cfg, 20, "test")
        for axis in "xyz":
            with patch.object(ds._inner, "render_pseudo_mri", wraps=ds._inner.render_pseudo_mri) as render:
                pair = audit.render_pair(ds, 0, axis, 0.5)
                first, second = render.call_args_list[-2:]
            expected = torch.zeros(9)
            expected[2 + "xyz".index(axis)] = 1
            torch.testing.assert_close(second.args[0] - first.args[0], expected)
            for a, b in zip(first.args[1:], second.args[1:]):
                if isinstance(a, torch.Tensor):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                else:
                    self.assertEqual(a, b)
            again = audit.render_pair(ds, 0, axis, 0.5)
            self.assertTrue(all(lesion.any() for lesion in pair["lesions"]))
            for v in range(2):
                for endpoint in "ab":
                    torch.testing.assert_close(pair[endpoint][v], again[endpoint][v], rtol=0, atol=0)
                delta = (pair["b"][v] - pair["a"][v]).numpy()[0]
                self.assertLess(np.abs(delta[~maximum_filter(pair["support"], size=3)]).max(), 2e-6)

    def test_true_following_beats_anatomy_proxy_despite_good_endpoint_r2(self):
        rng = np.random.default_rng(42)
        anatomy = rng.normal(size=(20, 3))
        delta = np.tile([0.1, 0.03, -0.02], (20, 1))
        truth = np.stack((anatomy - delta / 2, anatomy + delta / 2), axis=1)
        constant = np.repeat(anatomy[:, None], 2, axis=1)
        ids = np.repeat(np.arange(10), 2)
        perfect = audit.movement_metrics(truth, truth, ids, 31.5)
        proxy = audit.movement_metrics(truth, constant, ids, 31.5)
        inverted = audit.movement_metrics(truth, truth[:, ::-1], ids, 31.5)
        np.testing.assert_allclose(
            (perfect["movement_skill"], perfect["movement_gain"], perfect["movement_rmse_vox"]), (1, 1, 0), atol=1e-12
        )
        self.assertGreater(proxy["endpoint_mean_r2"], 0.99)
        self.assertEqual((proxy["movement_skill"], proxy["movement_gain"]), (0, 0))
        np.testing.assert_allclose((inverted["movement_skill"], inverted["movement_gain"]), (-3, -1), atol=1e-12)
        self.assertEqual(perfect["movement_skill_ci95_low"], 1)
        unchanged = audit.movement_metrics(constant, constant, ids, 31.5)
        self.assertEqual(unchanged["n_moved"], 0)
        self.assertTrue(np.isnan(unchanged["movement_skill"]))
        self.assertEqual(unchanged["no_move_predicted_displacement_vox"], 0)

    def test_real_extraction_reads_active_spatial_head_and_preserves_state_and_gradients(self):
        cfg = config(hidden_channels=16, separate_spatial_readout=True)
        ds = dataset(cfg, 20, "test")
        model = build_model(cfg, "cpu")
        with torch.no_grad():
            model.spatial_readout[-1].weight.zero_()
            model.spatial_readout[-1].bias.fill_(7)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        before = state_digest(model)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            pairs = audit.extract_pairs(model, ds, options(), "cpu", Path(tmp) / "pairs")
            try:
                arrays, truth, rows, replay, metadata = pairs
                self.assertEqual(truth.shape, (18, 3))
                self.assertEqual(metadata["n_pairs"], 9)
                self.assertGreater(metadata["n_moved"], 0)
                self.assertEqual({key[0] for key in arrays}, {"t1", "flair"})
                for view in ("t1", "flair"):
                    for grid in (2, 4):
                        values = arrays[view, grid, "projected"]
                        self.assertEqual(values.shape, (18, 9 * grid**3))
                        self.assertTrue((values == 7).all())
                    backbone = arrays[view, 2, "backbone"]
                    self.assertGreater(np.abs(backbone[1::2] - backbone[::2]).sum(), 0)
                for values in replay.values():
                    np.testing.assert_array_equal(values, 0)
                self.assertEqual(state_digest(model), before)
                self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters()))
            finally:
                audit.close_arrays(pairs[0])

    def test_quantized_no_moves_are_retained_and_not_redrawn(self):
        cfg = config()
        ds = dataset(cfg, 20, "test")
        model = build_model(cfg, "cpu")
        args = options(eps=1e-12, axes=["x"], include_native=False)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            pairs = audit.extract_pairs(model, ds, args, "cpu", Path(tmp) / "pairs")
            try:
                self.assertEqual(pairs[4]["n_pairs"], args.num_samples)
                self.assertEqual(pairs[4]["n_moved"], 0)
                self.assertEqual([r["subject_id"] for r in pairs[2]], [0, 1, 2])
                for values in pairs[0].values():
                    np.testing.assert_array_equal(values[::2], values[1::2])
            finally:
                audit.close_arrays(pairs[0])

    def test_intervention_labels_never_fit_or_select_probes(self):
        rng = np.random.default_rng(6)
        val = rng.normal(size=(40, 3)).astype("float32")
        endpoints = rng.normal(size=(12, 3)).astype("float32")
        key = ("flair", 2, "projected")
        rows = [{"subject_id": i, "intervention_axis": "x", "moved": True} for i in range(6)]
        pairs = ({key: endpoints}, endpoints.copy(), rows, {key: np.zeros(6)}, {})
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            first = audit.score_features({key: val}, val.copy(), pairs, options(axes=["x"]), "before", Path(tmp))
            # Change labels only, preserving images/features. Fitting must not see them.
            pairs[1][:] += 100
            second = audit.score_features({key: val}, val.copy(), pairs, options(axes=["x"]), "after", Path(tmp))
            fields = ("probe", "condition", "target", "alpha", "gamma", "validation_mse_standardized")
            self.assertEqual([[r[k] for k in fields] for r in first[2]], [[r[k] for k in fields] for r in second[2]])
            with np.load(Path(tmp) / "before_predictions.npz") as a, np.load(Path(tmp) / "after_predictions.npz") as b:
                for name in a.files:
                    if name != "truth":
                        np.testing.assert_array_equal(a[name], b[name])
            observed = [
                r
                for r in first[0]
                if r["probe"] == "ridge" and r["condition"] == "observed" and r["intervention_axis"] == "all"
            ]
            self.assertGreater(observed[0]["movement_skill"], 0.999)

    def create_run(self, root):
        cfg = config(hidden_channels=16, separate_spatial_readout=True)
        run = root / "run"
        run.mkdir()
        (run / "settings.json").write_text(json.dumps(cfg))
        model = build_model(cfg, "cpu")
        for name in ("model.pt", "model_init.pt"):
            torch.save(model.state_dict(), run / name)
        return run

    def test_cli_runs_real_probes_pairs_initial_and_leaves_no_large_files_or_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            run = self.create_run(Path(tmp))
            before = {p.name: p.read_bytes() for p in run.iterdir()}
            out = Path(tmp) / "audit"
            argv = [
                "--run-dir",
                str(run),
                "--out-dir",
                str(out),
                "--num-samples",
                "3",
                "--subject-offset",
                "0",
                "--grids",
                "1",
                "2",
                "--include-native",
                "--bootstrap",
                "10",
                "--device",
                "cpu",
            ]
            with patch.object(torch.Tensor, "backward", side_effect=AssertionError("No encoder training")):
                audit.main(argv)
            self.assertEqual(before, {p.name: p.read_bytes() for p in run.iterdir()})
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["encoder_unchanged"])
            self.assertTrue(report["source_checkpoints_unchanged"])
            self.assertEqual(
                report["cohorts"]["trained"]["moves"]["input_sha256"],
                report["cohorts"]["initial"]["moves"]["input_sha256"],
            )
            self.assertEqual(len(report["probe_split"]["fit_validation_ids"]), 15)
            self.assertEqual(len(report["probe_split"]["tune_validation_ids"]), 5)
            for row in report["summary"]:
                if row["stage"] == "projected":
                    self.assertEqual(row["readout_head"], "global" if row["grid"] == 1 else "spatial")
            for name in ("pairs.csv", "summary.csv", "sensitivity.csv", "probe_parameters.csv"):
                self.assertTrue((out / name).is_file())
            with np.load(out / "trained_predictions.npz") as a, np.load(out / "initial_predictions.npz") as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
            self.assertFalse(list(out.rglob("*.npy")))
            self.assertFalse(list(out.glob(".features-*")))
            with self.assertRaises(FileExistsError):
                audit.main(argv)

    def test_failed_scoring_cleans_features_and_marks_report_failed(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            run = self.create_run(Path(tmp))
            out = Path(tmp) / "audit"
            with patch.object(audit, "score_features", side_effect=RuntimeError("probe failure")):
                with self.assertRaisesRegex(RuntimeError, "probe failure"):
                    audit.main(
                        [
                            "--run-dir",
                            str(run),
                            "--out-dir",
                            str(out),
                            "--num-samples",
                            "2",
                            "--subject-offset",
                            "0",
                            "--axes",
                            "x",
                            "--grids",
                            "1",
                            "2",
                            "--device",
                            "cpu",
                        ]
                    )
            self.assertEqual(json.loads((out / "report.json").read_text())["status"], "failed")
            self.assertFalse(list(out.rglob("*.npy")))
            self.assertFalse(list(out.glob(".features-*")))


if __name__ == "__main__":
    unittest.main()
