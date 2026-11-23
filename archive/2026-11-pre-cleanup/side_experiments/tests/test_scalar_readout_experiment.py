"""Scalar semantics, response cancellation, label isolation and actual CPU runs."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from eval.encoder import scalar_readout_audit as audit
from eval.encoder.encoder_target_protocol import digest
from eval.protocol.score_checkpoint import build_model
from eval.synthetic.synthetic_dataset import LesionPlacementError
from models.scalar_readout import DescriptorDecoder, ScalarReadout, descriptors, flip_volume, ssl_objective
from tests.test_encoder_target_followups import config
from training import scalar_readout_data as data
from training import scalar_readout_experiment as experiment


class ScalarReadoutTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.addCleanup(
            torch.use_deterministic_algorithms,
            torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        torch.set_num_threads(1)

    def test_geometric_outputs_follow_position_and_reflection(self):
        model = ScalarReadout(1, grid=4, width=1, resolution=16)
        model.stem = nn.Identity()
        with torch.no_grad():
            model.location.weight.fill_(40)
            model.location.bias.zero_()
            model.values.weight.zero_()
            model.values.bias.zero_()
        image = torch.zeros(1, 1, 4, 4, 4)
        image[0, 0, 1, 2, 3] = 1
        code = model(image)
        expected = 2 * (torch.tensor([1.0, 2.0, 3.0]) * 4 + 1.5) / 15 - 1
        torch.testing.assert_close(code[0, 2:5], expected, atol=1e-6, rtol=0)
        flipped = model(flip_volume(image, [-1, 1, -1]))
        torch.testing.assert_close(flipped[:, 2:5], code[:, 2:5] * torch.tensor([-1.0, 1.0, -1.0]))
        self.assertEqual(code.shape, (1, 9))

    def test_descriptors_commute_with_reflections_and_decoder_has_only_nine_inputs(self):
        image = torch.randn(3, 1, 16, 16, 16)
        signs = [-1, -1, 1]
        torch.testing.assert_close(
            descriptors(flip_volume(image, signs), 4), flip_volume(descriptors(image, 4), signs), atol=1e-6, rtol=1e-6
        )
        head, decoder = ScalarReadout(1, 4, 4, 16), DescriptorDecoder(4)
        features = torch.randn(3, 1, 4, 4, 4)
        batch = dict(
            features=features,
            flip_features=flip_volume(features, signs),
            photo_features=features * 0.95,
            descriptors=descriptors(image, 4),
            flip_descriptors=descriptors(flip_volume(image, signs), 4),
            flip_signs=torch.tensor(signs).float().expand(3, 3),
        )
        observed_shapes = []
        handle = decoder.register_forward_pre_hook(lambda _, inputs: observed_shapes.append(inputs[0].shape))
        loss, _ = ssl_objective(
            head,
            decoder,
            batch,
            torch.tensor(0.0),
            torch.tensor(1.0),
            dict(reconstruction=1.0, equivariance=1.0, photometric=1.0, variance=1.0),
        )
        loss.backward()
        handle.remove()
        self.assertEqual(observed_shapes, [torch.Size([3, 9])] * 3)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))
        with self.assertRaisesRegex(ValueError, "unlabelled"):
            experiment.train_ssl(head, {**batch, "truth": object()}, SimpleNamespace(), "cpu")

    def test_oracle_matches_actual_coupled_responses_instead_of_forcing_diagonal(self):
        class IdentityHead(nn.Module):
            def forward(self, x):
                return x[:, :, 0, 0, 0]

        truth = np.arange(36, dtype=np.float32).reshape(4, 9) / 10
        pairs = np.stack([truth, truth + np.arange(9, dtype=np.float32)[None] / 10], 1)
        obs = dict(features=truth[:, :, None, None, None], truth=truth)
        intervention = dict(features=pairs[:, :, :, None, None, None], truth=pairs)
        terms = experiment.oracle_objective(
            IdentityHead(), obs, intervention, np.arange(4), np.arange(4), torch.ones(9), "cpu"
        )
        self.assertEqual(float(sum(terms.values())), 0.0)
        intervention["truth"] = pairs.copy()
        intervention["truth"][:, 1, 3] += 1
        self.assertGreater(
            float(
                experiment.oracle_objective(
                    IdentityHead(), obs, intervention, np.arange(4), np.arange(4), torch.ones(9), "cpu"
                )["response"]
            ),
            0,
        )

    def test_scalar_matching_and_vector_probe_distinguish_rotation(self):
        rng = np.random.default_rng(5)
        y, test = rng.normal(size=(256, 9)), rng.normal(size=(256, 9))
        permutation = rng.permutation(9)
        x, xt = y[:, permutation] * 2 + 3, test[:, permutation] * 2 + 3
        fitted = audit.fit_maps(x, y, 42)
        self.assertTrue(np.all(audit.r2(test, audit.apply_map(xt, fitted["scalar"])) > 0.999))
        self.assertLess(float(audit.r2(test, audit.apply_map(xt, fitted["shuffled_scalar"])).mean()), 0.1)
        q, _ = np.linalg.qr(rng.normal(size=(9, 9)))
        rotated = np.einsum("ni,ij->nj", y, q, optimize=False)
        rotated_test = np.einsum("ni,ij->nj", test, q, optimize=False)
        fitted = audit.fit_maps(rotated, y, 42)
        self.assertGreater(float(audit.r2(test, audit.apply_map(rotated_test, fitted["vector_ridge"])).mean()), 0.99)
        self.assertLess(float(audit.r2(test, audit.apply_map(rotated_test, fitted["scalar"])).mean()), 0.7)

    def test_rms_matrix_retains_responses_that_cancel_and_zero_directions(self):
        reference = np.tile(np.array([-1.0, 1.0])[:, None], (1, 9))
        pairs = np.zeros((2, 2, 9))
        pairs[0, 1, 0], pairs[1, 1, 0] = 1.0, -1.0
        rows = [dict(valid=True, pair_index=i, eps=0.5, factor_index=0, zero_image=False) for i in range(2)]
        result = audit.summarize_responses(pairs, reference, rows)
        first = result[0]
        self.assertEqual((first["signed_mean"], first["rms"]), (0.0, 1.0))
        self.assertEqual(result[1]["rms"], 0.0)

    def test_failed_interventions_are_retained_without_redrawing(self):
        args = SimpleNamespace(view="t1", grid=4)
        actual = data.render

        def fail_changed_brain(ds, ctx, z):
            if z[0] != ctx["lat"]["z_content"][0]:
                raise LesionPlacementError("test: no admissible center")
            return actual(ds, ctx, z)

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            data, "render", side_effect=fail_changed_brain
        ), contextlib.redirect_stdout(io.StringIO()):
            banks = data.Banks(tmp)
            try:
                bank, metadata = data.intervention_bank(
                    config(), "test", 1, 0, [0.5], args, data.Extractor(None, args, "cpu"), banks
                )
                self.assertEqual(metadata["failed_pairs"], 1)
                self.assertEqual(len(bank["rows"]), 9)
                self.assertFalse(bank["rows"][0]["valid"])
            finally:
                banks.close()

    def test_full_comparison_preserves_encoder_isolates_splits_and_cleans_cache(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            run, out, cache = root / "run", root / "result", root / "cache"
            run.mkdir()
            cfg = config()
            (run / "settings.json").write_text(json.dumps(cfg))
            model = build_model(cfg, "cpu")
            torch.save(model.state_dict(), run / "model.pt")
            before = digest(run / "model.pt")
            experiment.main(
                [
                    "--run-dir",
                    str(run),
                    "--out-dir",
                    str(out),
                    "--view",
                    "t1",
                    "--device",
                    "cpu",
                    "--steps",
                    "2",
                    "--train-samples",
                    "8",
                    "--probe-samples",
                    "8",
                    "--test-samples",
                    "8",
                    "--train-pair-subjects",
                    "1",
                    "--test-pair-subjects",
                    "1",
                    "--eval-eps",
                    ".5",
                    "--batch-size",
                    "4",
                    "--grid",
                    "4",
                    "--descriptor-grid",
                    "4",
                    "--width",
                    "4",
                    "--cache-dir",
                    str(cache),
                ]
            )
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(before, digest(run / "model.pt"))
            self.assertEqual(report["encoder_state_before"], report["encoder_state_after"])
            self.assertEqual(
                report["arms"]["geometric_oracle"]["initial_state_sha256"],
                report["arms"]["geometric_ssl"]["initial_state_sha256"],
            )
            self.assertEqual(list(cache.iterdir()), [])
            cohorts = report["cohorts"]
            self.assertTrue(
                set(cohorts["test"]["subject_ids"]).isdisjoint(cohorts["test_interventions"]["subject_ids"])
            )
            self.assertNotEqual(cohorts["train"]["generator_seed"], cohorts["val"]["generator_seed"])
            self.assertNotEqual(cohorts["test"]["generator_seed"], cohorts["val"]["generator_seed"])
            for arm in experiment.ARMS:
                torch.load(out / f"{arm}.pt", weights_only=True, map_location="cpu")
            recovery = list(csv.DictReader((out / "recovery.csv").open()))
            self.assertTrue(any(r["method"] == "shuffled_scalar" for r in recovery))
            self.assertEqual(
                sum(r["method"] == "geometric_coordinates" and r["arm"] == "geometric_ssl" for r in recovery), 3
            )
            self.assertTrue((out / "response_matrix.png").exists())
            self.assertTrue((out / "scalar_recovery.png").exists())
            with self.assertRaises(FileExistsError):
                experiment.main(
                    [
                        "--run-dir",
                        str(run),
                        "--out-dir",
                        str(out),
                        "--view",
                        "t1",
                        "--train-samples",
                        "8",
                        "--train-pair-subjects",
                        "1",
                    ]
                )

    def test_image_source_ssl_only_never_renders_training_interventions(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            (root / "settings.json").write_text(json.dumps(config()))
            actual = experiment.intervention_bank
            splits = []

            def recording(*args, **kwargs):
                splits.append(args[1])
                return actual(*args, **kwargs)

            with patch.object(experiment, "intervention_bank", side_effect=recording):
                experiment.main(
                    [
                        "--run-dir",
                        str(root),
                        "--out-dir",
                        str(root / "result"),
                        "--view",
                        "flair",
                        "--source",
                        "image",
                        "--arms",
                        "geometric_ssl",
                        "--device",
                        "cpu",
                        "--steps",
                        "1",
                        "--train-samples",
                        "8",
                        "--probe-samples",
                        "8",
                        "--test-samples",
                        "8",
                        "--train-pair-subjects",
                        "1",
                        "--test-pair-subjects",
                        "1",
                        "--eval-eps",
                        ".5",
                        "--batch-size",
                        "4",
                        "--grid",
                        "4",
                        "--descriptor-grid",
                        "4",
                        "--width",
                        "4",
                    ]
                )
            self.assertEqual(splits, ["test"])


if __name__ == "__main__":
    unittest.main()
