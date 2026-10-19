"""Intervention evaluator for the spatial bottleneck: null responses, GAP-blind changes, the same subjects."""

import contextlib
import csv
import io
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from test_spatial_bottleneck import trainer_argv, write_source_run

from eval.protocol import spatial_bottleneck_eval as factor_eval
from eval.protocol import spatial_bottleneck_interventions as interventions
from models.spatial_bottleneck import SpatialBottleneck
from training import spatial_bottleneck_experiment as experiment


def toy_readouts(train, latent_dim, seed=0):
    torch.manual_seed(seed)
    head = SpatialBottleneck(tuple(train.shape[2:]), latent_dim=latent_dim, width=4, hidden=16)
    head.fit_normalization(train)
    head.eval()
    readouts = interventions.Readouts(
        train, head.template, head.scale, {"z:toy": head}, [latent_dim], (2, 2, 2), 4, "cpu"
    )
    return readouts, head


class InterventionUnitTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_each_intervention_moves_one_latent_and_holds_the_rest(self):
        torch.manual_seed(0)
        latents = dict(
            z_content=torch.randn(9),
            z_deformation=torch.randn(4, 4, 4),
            z_fissure=torch.randn(8, 8, 8),
            z_style_v1=torch.randn(3),
            z_style_v2=torch.randn(3),
            z_lesion=None,
        )
        before = {k: v.clone() for k, v in latents.items() if v is not None}
        targets = interventions.build_targets(["a", "b"], ["gain", "bias"], latents)
        self.assertEqual([t["name"] for t in targets], ["a", "b", "gain", "bias", "~z_deformation", "~z_fissure"])
        expected = {"b": {"z_content"}, "bias": {"z_style_v1", "z_style_v2"}, "~z_fissure": {"z_fissure"}}
        for target in targets:
            moved = interventions.shifted(latents, target, 0.5)
            changed = {k for k, v in before.items() if not torch.equal(moved[k], v)}
            self.assertEqual(changed, expected.get(target["name"], changed))
            self.assertIsNone(moved["z_lesion"])
            if target["kind"] == "nuisance":
                self.assertAlmostEqual((moved[target["block"]] - before[target["block"]]).norm().item(), 0.5, places=5)
            else:
                key = "z_content" if target["kind"] == "content" else "z_style_v1"
                change = moved[key] - before[key]
                self.assertEqual(torch.nonzero(change).flatten().tolist(), [target["index"]])
                self.assertAlmostEqual(change[target["index"]].item(), 0.5, places=6)
        for k, v in before.items():
            torch.testing.assert_close(latents[k], v, rtol=0, atol=0)

    def test_readouts_match_spatial_bottleneck_eval_features(self):
        torch.manual_seed(1)
        train = torch.randn(20, 2, 2, 4, 4, 4)
        readouts, head = toy_readouts(train, 3)
        self.assertEqual(readouts.names, ["map", "patch2x2x2", "gap", "pca3", "z:toy"])
        gram = factor_eval.map_gram([train[:, 0], train[:, 1]], head.template, head.scale)
        scores = factor_eval.pca_scores(gram, np.arange(40), 3)
        standardised = ((train - head.template) / head.scale).transpose(0, 1).reshape(40, -1).double().numpy()
        projected = (standardised - standardised.mean(0)) @ readouts.axes[3]
        np.testing.assert_allclose(projected, scores, rtol=1e-6, atol=1e-8)

        minus = torch.randn(5, 2, 2, 4, 4, 4)
        plus = minus + torch.randn_like(minus)
        out = readouts.respond(minus, plus, chunk=2)
        delta = (plus - minus).double()
        np.testing.assert_allclose(out["map"], (delta / head.scale).reshape(5, 2, -1).numpy(), rtol=1e-5, atol=1e-6)
        gap_std = train.mean((3, 4, 5)).double().numpy().std(0)
        np.testing.assert_allclose(out["gap"], delta.mean((3, 4, 5)).numpy() / gap_std, rtol=1e-5, atol=1e-6)
        change = factor_eval.encode_views(head, plus, 4, "cpu") - factor_eval.encode_views(head, minus, 4, "cpu")
        code_std = factor_eval.encode_views(head, train, 4, "cpu").std(0)
        np.testing.assert_allclose(out["z:toy"], change / code_std, rtol=1e-5, atol=1e-6)
        # spatial_bottleneck_eval scores new maps through one Gram over training and new rows.
        rows = [train[:, 0], train[:, 1], minus[:, 0], minus[:, 1], plus[:, 0], plus[:, 1]]
        kernel = factor_eval.pca_scores(factor_eval.map_gram(rows, head.template, head.scale), np.arange(40), 3)
        kernel_change = (kernel[50:60] - kernel[40:50]).reshape(2, 5, 3).transpose(1, 0, 2)
        expected = kernel_change / scores.reshape(2, 20, 3).std(1)
        signs = np.sign(np.einsum("nvk,nvk->k", expected, out["pca3"]))
        np.testing.assert_allclose(out["pca3"], expected * signs, rtol=1e-4, atol=1e-6)

    def test_gap_invisible_spatial_change_moves_map_and_code_but_not_gap(self):
        torch.manual_seed(2)
        train = torch.randn(24, 2, 1, 4, 4, 4)
        readouts, _ = toy_readouts(train, 2)
        # Multiples of 1/32, so adding the change below is exact in float32.
        base = torch.randint(-64, 64, (16, 2, 1, 4, 4, 4)) / 32.0
        # Mass moves between two voxels in different 2x2x2 cells: the channel mean is unchanged.
        spatial = torch.zeros(1, 4, 4, 4)
        spatial[0, 1, 1, 1], spatial[0, 2, 2, 2] = 1.0, -1.0
        conditions = {
            interventions.NULL: base + 1e-3 * torch.randn_like(base),
            "spatial": base + spatial,
            "global": base + 0.5,
            "~field": base + 0.3 * torch.randn(1, 4, 4, 4),
        }
        responses = {}
        for label, plus in conditions.items():
            for name, value in readouts.respond(base, plus).items():
                responses.setdefault(name, {})[label] = value
        np.testing.assert_array_equal(responses["gap"]["spatial"], 0.0)
        factors = [
            dict(name="spatial", kind="content"),
            dict(name="global", kind="content"),
            dict(name="~field", kind="nuisance"),
        ]
        measured = {f["name"]: np.ones(16, dtype=bool) for f in factors}
        draws = np.random.default_rng(0).integers(0, 16, size=(200, 16))
        results = {
            name: interventions.score_representation(r, factors, measured, draws) for name, r in responses.items()
        }

        gap = results["gap"]["per_factor"]
        self.assertEqual(gap["spatial"]["snr_render"], 0.0)
        self.assertEqual(gap["spatial"]["snr_render_ci"], [0.0, 0.0])
        self.assertEqual(gap["spatial"]["snr_nuisance"], 0.0)
        self.assertGreater(gap["global"]["snr_render_ci"][0], 10)
        for name in ("map", "patch2x2x2", "z:toy"):
            row = results[name]["per_factor"]["spatial"]
            self.assertGreater(row["snr_render_ci"][0], 5, name)
            self.assertGreater(row["snr_nuisance"], 0, name)
            self.assertGreater(row["recall"], 0.8, name)
        self.assertEqual(results["map"]["identification"]["factors"], ["spatial", "global"])
        self.assertEqual(results["map"]["nuisance_reference"], ["~field", "~field"])

    def test_snr_reads_measured_subjects_and_scales_by_lesion_displacement(self):
        rng = np.random.default_rng(3)
        n = 20
        null = rng.normal(size=(n, 5))
        lesion = 2.0 * rng.normal(size=(n, 5))
        # Half the nudges left the lesion centre on its voxel: no image change, no measurement.
        lesion[::2] = 0.0
        responses = {"lesion_x": lesion, "brain_size": rng.normal(size=(n, 5)), "ignored": np.zeros((n, 5))}
        measured = {"lesion_x": np.arange(n) % 2 == 1, "brain_size": np.ones(n, bool), "ignored": np.zeros(n, bool)}
        distance = np.where(measured["lesion_x"], rng.uniform(1.0, 4.0, n), 0.0)
        factors = [dict(name=name, kind="content") for name in responses]
        draws = rng.integers(0, n, size=(300, n))
        record = interventions.score_view(responses, null, factors, measured, draws, {"lesion_x": distance})

        row = record["factors"]["lesion_x"]
        null_mean = np.linalg.norm(null, axis=1).mean()
        self.assertAlmostEqual(row["snr_render"], np.linalg.norm(lesion[1::2], axis=1).mean() / null_mean)
        self.assertAlmostEqual(row["snr_render_per_voxel"], row["snr_render"] / distance[1::2].mean())
        low, high = np.percentile(record["boot"]["lesion_x"]["snr_render_per_voxel"], [2.5, 97.5])
        self.assertTrue(low < row["snr_render_per_voxel"] < high)
        self.assertTrue(np.isnan(record["factors"]["brain_size"]["snr_render_per_voxel"]))
        self.assertIsNone(record["boot"]["brain_size"]["snr_render_per_voxel"])
        self.assertEqual(record["factors"]["ignored"]["snr_render"], 0.0)
        self.assertEqual(record["identification"]["factors"], ["lesion_x", "brain_size"])


def assert_null_response(case, payload, factor):
    """Every representation's response to ``factor`` is exactly zero and it is never a class to identify."""
    record = next(f for f in payload["factors"] if f["name"] == factor)
    case.assertEqual((record["moved"], record["image_change"], record["unchanged_response"]), (False, 0, 0))
    for name in payload["representations"]:
        result = payload["results"][name]
        case.assertEqual(result["per_factor"][factor]["response"], 0.0, name)
        case.assertEqual(result["per_factor"][factor]["snr_render"], 0.0, name)
        case.assertGreater(result["per_factor"]["brain_size"]["response"], 0.0, name)
        case.assertNotIn(factor, result["identification"]["factors"])
        case.assertIn("lesion_x", result["identification"]["factors"])


class InterventionCliTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_renderer_ignored_content_dimension_gives_no_response(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            run = write_source_run(root, synthetic_n_content=10)  # dimension 9 is drawn but never rendered
            experiment.main(trainer_argv(run, root / "d2", 2))
            argv = ["--bottleneck-dirs", str(root / "d2"), "--out-dir", str(root / "out"), "--device", "cpu"]
            payload = interventions.main(
                argv + ["--bootstrap", "100", "--factors", "brain_size", "lesion_x", "z_content_9"]
            )
            self.assertEqual([f["name"] for f in payload["factors"]], ["brain_size", "lesion_x", "z_content_9"])
            assert_null_response(self, payload, "z_content_9")
            self.assertFalse(payload["lesion_centre"]["rendered"])  # sample_targets needs nine content factors

    def test_checkpoint_cli_null_response_and_the_observational_subjects(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            run = write_source_run(root, synthetic_shared_gain_bias=True, synthetic_lesion_placement="wm_interior")
            bottleneck = root / "d2"
            experiment.main(trainer_argv(run, bottleneck, 2))
            argv = ["--bottleneck-dirs", str(bottleneck), "--device", "cpu", "--bootstrap", "100"]
            payload = interventions.main(argv + ["--out-dir", str(root / "out"), "--num-subjects", "6"])

            names = {"map", "patch2x2x2", "gap", "pca2", "z_init:d2", "z:d2"}
            self.assertEqual(set(payload["representations"]), names)
            factors = {f["name"]: f for f in payload["factors"]}
            self.assertEqual(len(factors), 9 + 3 + 2)
            kinds = [factors[n]["kind"] for n in ("gain", "bias", "noise_sigma")]
            self.assertEqual(kinds, ["style_shared", "style_shared", "style"])
            # Clean content zeroes the deformation and fissure fields: the renderer ignores them, and
            # there is no nuisance null.
            assert_null_response(self, payload, "~z_deformation")
            assert_null_response(self, payload, "~z_fissure")
            self.assertFalse(payload["nulls"]["nuisance"]["available"])
            self.assertTrue(payload["lesion_centre"]["rendered"])
            self.assertGreater(max(factors["lesion_x"]["centre_shift"]), 0.0)
            self.assertGreater(factors["lesion_x"]["displacement_vox"], 0.0)
            self.assertIsInstance(factors["lesion_x"]["midline_crossings"], int)
            self.assertEqual(factors["~z_deformation"]["centre_shift"], [0.0, 0.0, 0.0])
            self.assertIsNone(factors["gain"]["centre_shift"])
            map_rows = payload["results"]["map"]["per_factor"]
            self.assertGreater(map_rows["lesion_x"]["snr_render_per_voxel"], 0.0)
            self.assertIsNone(map_rows["brain_size"]["snr_render_per_voxel"])
            with (root / "out" / "lesion_moves.csv").open() as handle:
                moves = list(csv.DictReader(handle))
            self.assertEqual(len(moves), (9 + 2) * 6)
            first = next(row for row in moves if row["factor"] == "lesion_x")
            ends = np.array([[float(first[f"centre_{end}_{axis}"]) for axis in "xyz"] for end in ("minus", "plus")])
            self.assertAlmostEqual(float(first["displacement_vox"]), np.linalg.norm(ends[1] - ends[0]) * 15 / 2)
            self.assertEqual(first["crossed_midline"], str(bool(ends[0, 0] * ends[1, 0] < 0)))

            # Same subjects as the bottleneck's saved test outputs and spatial_bottleneck_eval's split.
            record = payload["bottlenecks"][0]
            self.assertTrue(all(v < 1e-5 for v in record["parity"].values()), record["parity"])
            self.assertEqual(payload["subjects"]["ids"], [f"test:{i}" for i in range(6)])
            observed = factor_eval.main(argv + ["--out-dir", str(root / "eval"), "--test-samples", "6"])
            self.assertEqual(payload["splits"], observed["splits"])
            source, saved_args, _ = experiment.load_source(
                Namespace(run_dir=run, checkpoint="vqvae_model.pt", seed=42), "cpu"
            )
            dataset = experiment.build_split_dataset(saved_args, "test", 6)
            _, _, (content, _) = factor_eval.extract_split(source, dataset, 2, "cpu", 0, 2**30, None)
            np.testing.assert_array_equal(np.array(payload["subjects"]["z_content"]), content)
            self.assertEqual(payload["subjects"]["render_seeds"], [dataset._inner.sample_seed_for(i) for i in range(6)])

            with (root / "out" / "interventions.csv").open() as handle:
                self.assertEqual(len(handle.readlines()), 1 + len(names) * len(factors))
            for csv_name in ("alias.csv", "confusion.csv"):
                self.assertTrue((root / "out" / csv_name).is_file())

            real = interventions.render_pair

            def drifted(dataset, latents, seed):
                x1, x2, mask = real(dataset, latents, seed)
                return x1 + 1e-6, x2, mask

            with mock.patch.object(interventions, "render_pair", drifted):
                with self.assertRaisesRegex(ValueError, "does not reproduce the dataset's images"):
                    interventions.main(argv + ["--out-dir", str(root / "drifted")])
            outputs = torch.load(bottleneck / "test_outputs.pt", weights_only=True)
            outputs["codes"] = outputs["codes"].roll(1, 0)
            torch.save(outputs, bottleneck / "test_outputs.pt")
            with self.assertRaisesRegex(ValueError, "do not reproduce"):
                interventions.main(argv + ["--out-dir", str(root / "tampered")])


if __name__ == "__main__":
    unittest.main()
