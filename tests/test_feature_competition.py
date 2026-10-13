"""Controls that make the competition comparison interpretable."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from eval.synthetic.feature_competition_report import fit_probe, hemisphere_split, metrics
from training import feature_competition
from training.feature_competition import parse_args
from training.feature_competition_data import (
    ARMS,
    MATCHED_PAIRS,
    SPLITS,
    CompetitionRenderer,
    array_hash,
    load_bank,
    write_json,
)
from training.feature_competition_train import (
    GapRegressor,
    ImageLocalizer,
    batch_indices,
    build_ssl,
    heldout_check,
    infonce,
    run_path,
    ssl_loss,
    standardize_batch,
    state_hash,
)


class CompetitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.renderer = CompetitionRenderer()

    def test_common_mask_and_local_pixels_survive_competitor_change(self):
        r = self.renderer
        norm = dict(mean=0.5, scale=0.6)
        for i in range(3):
            subject = r.subject("pilot", i)
            outputs = {a: r.render(subject, a, norm) for a in ARMS}
            self.assertEqual(len({array_hash(x[2]) for x in outputs.values()}), 1)
            roi = binary_dilation(outputs["A"][2] > 0, structure=np.ones((3, 3, 3)))
            for a, b in MATCHED_PAIRS:
                core = outputs[a][2] > 0
                np.testing.assert_array_equal(outputs[a][0][:, 0, core], outputs[b][0][:, 0, core])
                off_a = r.render(subject, a, norm, lesion_on=False)[0]
                off_b = r.render(subject, b, norm, lesion_on=False)[0]
                np.testing.assert_allclose(
                    (outputs[a][0] - off_a)[:, 0, roi], (outputs[b][0] - off_b)[:, 0, roi], atol=5e-7, rtol=1e-5
                )
                self.assertFalse(np.array_equal(outputs[a][0], outputs[b][0]))
            self.assertFalse(np.array_equal(outputs["A"][0], outputs["C"][0]))

    def test_noise_redraw_preserves_geometry_and_subject_draws(self):
        r = self.renderer
        s = r.subject("train", 3)
        a = r.render(s, "A", noise_repeat=1)
        b = r.render(s, "A", noise_repeat=2)
        np.testing.assert_array_equal(a[2], b[2])
        self.assertFalse(np.array_equal(a[0], b[0]))
        np.testing.assert_array_equal(s["styles"], r.subject("train", 3)["styles"])
        self.assertEqual(
            len({r.subject(split, i)["namespace"] for split in ("train", "probe", "test", "pilot") for i in range(10)}),
            40,
        )

    def test_default_contrast_replays_existing_renderer_and_invalid_scale_rejected(self):
        r = self.renderer
        s = r.subject("train", 1)
        tissue, mask = r.tissue(0), r.lesion(s["centre"])
        kw = dict(
            tissue_map=tissue,
            lesion_load=mask,
            z_style=torch.from_numpy(s["styles"][0]),
            modality="T1",
            view_seed=5,
            device="cpu",
        )
        torch.testing.assert_close(
            r.renderer.render_modality(**kw), r.renderer.render_modality(**kw, lesion_contrast_scale=1), rtol=0, atol=0
        )
        with self.assertRaises(ValueError):
            r.renderer.render_modality(**kw, lesion_contrast_scale=float("nan"))
        with self.assertRaises(ValueError):
            r.renderer.render_modality(**kw, lesion_contrast_scale=100)

    def test_training_initialization_order_and_gradient(self):
        settings = dict(hidden_channels=8, res_channels=4)
        torch.manual_seed(42)
        a = build_ssl(settings)
        torch.manual_seed(42)
        b = build_ssl(settings)
        self.assertEqual(state_hash(a), state_hash(b))
        for step in (1, 5, 20):
            np.testing.assert_array_equal(batch_indices(20, 4, 42, step), batch_indices(20, 4, 42, step))
        pair = torch.randn(2, 2, 1, 16, 16, 16)
        loss = ssl_loss(a, pair)
        code = a(torch.cat([pair[:, 0], pair[:, 1]]), pool_only=True, n_views=2)[2][0][:, :9]
        unit = torch.nn.functional.normalize(code, dim=-1)
        scores = unit[:2] @ unit[2:].T / 0.1
        expected = torch.nn.functional.cross_entropy(scores, torch.arange(2)) + torch.nn.functional.cross_entropy(
            scores.T, torch.arange(2)
        )
        torch.testing.assert_close(loss, expected)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.encoder.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.encoder_v1.parameters()))

    def test_image_control_receives_image_only_at_inference(self):
        model = ImageLocalizer(16)
        x = torch.randn(2, 1, 16, 16, 16)
        centre = torch.tensor([[5.0, 7.0, 6.0], [8.0, 4.0, 9.0]])
        loss = model.loss(x, centre)
        loss.backward()
        prediction = model(x)
        self.assertEqual(tuple(prediction.shape), (2, 3))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_probe_and_metric_use_heldout_targets_only_for_scoring(self):
        rng = np.random.default_rng(22)
        train = rng.normal(size=(30, 3))
        test = rng.normal(size=(12, 3))
        truth = test * 2 + 1
        prediction, _ = fit_probe(train, train * 2 + 1, test, "ridge")
        self.assertGreater(metrics(truth, prediction)["mean_r2"], 0.99)
        self.assertEqual(metrics(truth, truth)["mean_error_vox"], 0)

    def test_hemisphere_metrics_separate_the_side_from_fine_position(self):
        self.assertEqual(hemisphere_split(self.renderer.centres[:, 0]), 31.5)
        self.assertIsNone(hemisphere_split(np.arange(20.0, 40.0)))
        rng = np.random.default_rng(3)
        truth = np.c_[rng.choice([22, 23, 24, 25, 26, 37, 38, 39, 40, 41], 200), rng.integers(24, 40, (200, 2))]
        side = truth[:, 0] > 31.5
        # Knows only the side: the hemisphere is perfect, the position within it is not known at all.
        side_only = np.c_[np.where(side, 39.0, 24.0), np.full((200, 2), 31.5)]
        result = metrics(truth, side_only, split=31.5)
        self.assertEqual((result["hemisphere_auc"], result["hemisphere_accuracy"]), (1.0, 1.0))
        self.assertGreater(result["mean_r2"], 0.25)
        self.assertLess(abs(result["fine_position_r2"]), 0.05)
        perfect = metrics(truth, truth.astype(float), split=31.5)
        self.assertAlmostEqual(perfect["fine_position_r2"], 1.0)
        self.assertNotIn("hemisphere_auc", metrics(truth, side_only))

    def test_cli_defaults_and_incomplete_bank(self):
        a = parse_args(["train", "--bank", "/tmp/example", "--kind", "ssl"])
        self.assertEqual(a.batch_size, 32)
        with tempfile.TemporaryDirectory() as path:
            write_json(Path(path) / "manifest.json", dict(version=1, ready=False))
            with self.assertRaises(ValueError):
                load_bank(path)

    def test_bprime_breaks_cross_view_sharing_only(self):
        r = self.renderer
        norm = dict(mean=0.5, scale=0.6)
        subjects = [s for s in (r.subject("pilot", i) for i in range(30)) if abs(s["brain"] - s["brain_other"]) > 0.1]
        for s in subjects[:3]:
            self.assertEqual(r.brain_sizes(s, "Bp"), (s["brain"], s["brain_other"]))
            b, bp = r.render(s, "B", norm), r.render(s, "Bp", norm)
            # T1 is B's T1. FLAIR keeps B's style, bias field, noise draw and lesion, with its own brain size.
            np.testing.assert_array_equal(b[0][0], bp[0][0])
            np.testing.assert_array_equal(b[1][0], bp[1][0])
            np.testing.assert_array_equal(b[2], bp[2])
            self.assertFalse(np.array_equal(b[0][1], bp[0][1]))
            np.testing.assert_array_equal(b[1][0], b[1][1])
            self.assertFalse(np.array_equal(bp[1][0], bp[1][1]))

    def test_extra_brain_draw_leaves_earlier_draws_unchanged(self):
        r = self.renderer
        for split, index in (("train", 0), ("test", 7), ("pilot", 3)):
            rng = np.random.default_rng(np.random.SeedSequence([r.seed, SPLITS[split], index]))
            raw_brain = float(rng.normal())
            centre = r.centres[rng.integers(len(r.centres))]
            styles = rng.normal(size=(2, 3)).astype(np.float32)
            s = r.subject(split, index)
            self.assertEqual(s["raw_brain"], raw_brain)
            np.testing.assert_array_equal(s["centre"], centre)
            np.testing.assert_array_equal(s["styles"], styles)
        draws = np.array([[r.subject("train", i)[k] for k in ("raw_brain", "raw_brain_other")] for i in range(300)])
        self.assertLess(abs(np.corrcoef(draws.T)[0, 1]), 0.15)

    def test_gap_regressor_is_the_ssl_architecture_plus_a_linear_head(self):
        settings = dict(hidden_channels=8, res_channels=4)
        torch.manual_seed(42)
        ssl = build_ssl(settings)
        torch.manual_seed(42)
        model = GapRegressor(settings, mean=(30.0, 31.0, 32.0), scale=(8.0, 3.5, 3.5))
        for part in ("encoder", "to_encoding"):
            mine, reference = getattr(model.encoder, part).state_dict(), getattr(ssl, part).state_dict()
            self.assertEqual(mine.keys(), reference.keys())
            for key, value in reference.items():
                torch.testing.assert_close(mine[key], value)
        x = torch.randn(2, 1, 16, 16, 16)
        model.loss(x, torch.tensor([[30.0, 31.0, 32.0], [38.0, 28.0, 35.0]])).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.encoder.parameters()))
        restored = GapRegressor(settings)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(x), model(x))
        self.assertEqual(tuple(model(x).shape), (2, 3))

    def test_standardized_infonce_centres_and_scales_each_view_over_the_batch(self):
        torch.manual_seed(1)
        code = 1 + 0.01 * torch.randn(2, 6, 12)
        z = (code - code.mean(1, keepdim=True)) / (code.std(1, keepdim=True) + 1e-6)
        torch.testing.assert_close(standardize_batch(code), z)
        unit = torch.nn.functional.normalize(z[..., :9], dim=-1)
        scores = unit[0] @ unit[1].T / 0.1
        target = torch.arange(6)
        expected = torch.nn.functional.cross_entropy(scores, target) + torch.nn.functional.cross_entropy(
            scores.T, target
        )
        torch.testing.assert_close(infonce(code, standardize=True), expected)
        # Raw codes this alike leave the loss at chance; standardized, the same codes are told apart.
        self.assertAlmostEqual(float(infonce(code)), 2 * np.log(6), places=2)
        self.assertGreater(abs(float(infonce(code, standardize=True)) - 2 * np.log(6)), 0.1)
        self.assertEqual(run_path("/b", "ssl", "A", 42, "paired", "std").name, "ssl-std_A_s42_paired")
        self.assertEqual(run_path("/b", "ssl", "A", 42).name, "ssl_A_s42_paired")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["train", "--bank", "/tmp/x", "--kind", "supervised", "--standardize-code"])

    def test_heldout_check_sits_at_chance_without_shared_content(self):
        torch.manual_seed(0)
        model = build_ssl(dict(hidden_channels=8, res_channels=4))
        bank = np.random.default_rng(0).normal(size=(9, 2, 1, 16, 16, 16)).astype(np.float32)
        row = heldout_check(model, "ssl", bank, None, 4, "cpu")
        self.assertAlmostEqual(row["chance_loss"], 2 * np.log(4))
        self.assertEqual(row["heldout_subjects"], 8)
        self.assertLess(abs(row["heldout_loss"] - row["chance_loss"]), 0.1)
        self.assertTrue(0 <= row["heldout_top1"] <= 1)
        self.assertTrue(model.training)

    def test_cli_arms_kinds_and_checkpoint_options(self):
        from training import feature_competition_data

        self.assertEqual(feature_competition.ARMS, feature_competition_data.ARMS)
        a = parse_args(
            ["train", "--bank", "/tmp/x", "--kind", "supervised_gap", "--arms", "A", "Bp", "--save-steps", "250", "500"]
        )
        self.assertEqual((a.arms, a.save_steps, a.batch_size), (["A", "Bp"], [250, 500], 8))
        self.assertIsNone(parse_args(["evaluate", "--bank", "/tmp/x", "--kind", "ssl"]).arms)
        self.assertEqual(parse_args(["prepare", "--bank", "/tmp/x"]).arms, list(ARMS))
        self.assertEqual(
            parse_args(["evaluate", "--bank", "/tmp/x", "--kind", "ssl", "--checkpoint", "all"]).checkpoint, "all"
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["train", "--bank", "/tmp/x", "--kind", "ssl", "--save-steps", "0"])

    def test_pilot_pipeline_runs_end_to_end_on_a_tiny_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bank"
            common = ["--seeds", "42", "--device", "cpu", "--batch-size", "2"]
            small = ["--hidden-channels", "8", "--res-channels", "4"]
            commands = [
                ["prepare", "--arms", "A", "Bp", "--train-samples", "8", "--probe-samples", "8", "--test-samples", "8"]
                + ["--pilot-samples", "6"],
                ["audit", "--noise-subjects", "3", "--noise-repeats", "2"],
                ["train", "--kind", "ssl", *common, *small, "--steps", "2", "--save-every", "2", "--save-steps", "1"],
                [
                    "train",
                    "--kind",
                    "supervised_gap",
                    "--arms",
                    "A",
                    "--views",
                    "flair",
                    *common,
                    *small,
                    "--steps",
                    "1",
                ],
                ["evaluate", "--kind", "ssl", *common, "--checkpoint", "all", "--spatial-grid", "2"],
                ["evaluate", "--kind", "supervised_gap", "--arms", "A", "--views", "flair", *common],
                ["train", "--kind", "ssl", "--standardize-code", "--arms", "A", *common, *small, "--steps", "1"],
                ["evaluate", "--kind", "ssl", "--standardize-code", "--arms", "A", *common, "--spatial-grid", "2"],
                ["summarize"],
                ["curves"],
            ]
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                for command in commands:
                    feature_competition.main([*command, "--bank", str(root)])
                feature_competition.main(
                    ["evaluate", "--kind", "ssl", *common, "--checkpoint", "all"]
                    + ["--spatial-grid", "2", "--bank", str(root)]
                )
            self.assertEqual(json.loads((root / "manifest.json").read_text())["arms"], ["A", "Bp"])
            audit = json.loads((root / "audit/report.json").read_text())
            self.assertEqual(audit["matched_pairs"], [["A", "Bp"]])
            self.assertTrue(audit["input_gate_passed"])
            self.assertEqual(set(audit["visibility"]), {"t1_A", "flair_A", "t1_Bp", "flair_Bp"})
            progress = json.loads((root / "runs/ssl_A_s42_paired/progress.json").read_text())
            self.assertEqual([row["step"] for row in progress["monitor"]], [0, 1, 2])
            self.assertAlmostEqual(progress["monitor"][0]["chance_loss"], 2 * np.log(2))
            for step in (0, 1, 2):
                self.assertTrue((root / f"runs/ssl_A_s42_paired/eval_{step:06d}/metrics.json").exists())
            bp = json.loads((root / "runs/ssl_Bp_s42_paired/eval_000002/metrics.json").read_text())
            self.assertEqual({row["view"] for row in bp["rows"] if row["regime"] == "competitor"}, {"t1", "flair"})
            ssl = json.loads((root / "runs/ssl_A_s42_paired/eval_000002/metrics.json").read_text())
            self.assertEqual({row["source"] for row in ssl["rows"]}, {"global", "pooled", "spatial"})
            gap = json.loads((root / "runs/supervised_gap_A_s42_flair/eval_000001/metrics.json").read_text())
            self.assertEqual(
                {(row["source"], row["probe"]) for row in gap["rows"] if row["regime"] == "common"},
                {
                    ("global", "supervised"),
                    ("global", "ridge"),
                    ("global", "rbf"),
                    ("pooled", "ridge"),
                    ("pooled", "rbf"),
                },
            )
            report = json.loads((root / "summary/report.json").read_text())
            self.assertIn("ABp", {row["pair"] for row in report["comparisons"]})
            standardized = json.loads((root / "runs/ssl-std_A_s42_paired/settings.json").read_text())
            self.assertTrue(standardized["standardize_code"])
            self.assertIn("batch-standardized", (root / "summary/curves.csv").read_text())
            for name in ("curves.csv", "curves_A.png", "curves_Bp.png", "curves_arms_plain.png"):
                self.assertTrue((root / "summary" / name).exists())
            self.assertEqual(log.getvalue().count("Evaluation exists"), 6)


if __name__ == "__main__":
    unittest.main()
