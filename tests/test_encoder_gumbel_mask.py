"""Learned (top-k Gumbel-softmax) content mask in the encoder-only model: flags, selection, losses, readers, runs."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import training.losses as losses
from eval.lesion.checkpoint_lesion_analysis import batch_features
from eval.metrics import cross_reconstruction, dci
from eval.protocol.score_checkpoint import build_model, encode_blocks
from models.multiview_encoder import MultiviewConvEncoder
from training import main_conv_synthetic as trainer

TINY = ["--res", "16", "--hidden-channels", "8", "--res-channels", "4", "--nb-res-layers", "1", "--latent-dim", "12"]
TINY += ["--conv-readout", "mlp", "--encoder-head-hidden", "8", "--tau", "0.1", "--best-metric", "none"]
LEARNED = TINY + ["--mask-mode", "learned"]
CRITERIA = (torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss())
MOVED = [0, 1, 2, 4, 5, 6, 7, 8, 10]  # unit 3 out of the content block, unit 10 in
STYLE = [3, 9, 11]


def config(*argv, **changes):
    cfg = vars(trainer.parse_args([*LEARNED, *argv]))
    cfg.update(changes)
    return cfg


def move_units(model):
    """Swap unit 3 out of the content block and unit 10 in, through the logits alone."""
    with torch.no_grad():
        model.mask_logits[3], model.mask_logits[10] = -3.0, 3.0
    return model


def images(n=4, seed=3):
    return torch.randn(n, 1, 16, 16, 16, generator=torch.Generator().manual_seed(seed))


class ParserTests(unittest.TestCase):
    def rejects(self, argv):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            trainer.parse_args(argv)

    def test_flags_and_their_rules(self):
        default = trainer.parse_args([])
        self.assertEqual((default.mask_mode, default.mask_lr_scale, default.mask_estimator), ("fixed", 1.0, "relaxed"))
        self.assertEqual(default.mask_init_gap, 4.0)
        for gap in (0.0, 1.0, 4.0):
            self.assertEqual(trainer.parse_args(LEARNED + ["--mask-init-gap", str(gap)]).mask_init_gap, gap)
        args = trainer.parse_args(LEARNED + ["--mask-lr-scale", "10", "--mask-estimator", "straight_through"])
        self.assertEqual(
            (args.mask_mode, args.mask_lr_scale, args.mask_estimator), ("learned", 10.0, "straight_through")
        )
        for argv in (
            LEARNED + ["--content-channels", "12"],
            LEARNED + ["--contrastive-loss-type", "barlow_twins"],
            LEARNED + ["--contrastive-proj-dim", "8"],
            LEARNED + ["--mask-lr-scale", "0"],
            LEARNED + ["--mask-lr-scale", "nan"],
            LEARNED + ["--mask-lr-scale", "inf"],
            TINY + ["--mask-lr-scale", "10"],
            TINY + ["--mask-estimator", "straight_through"],
            TINY + ["--mask-init-gap", "1"],
            LEARNED + ["--mask-init-gap", "-1"],
            LEARNED + ["--mask-init-gap", "nan"],
            LEARNED + ["--mask-init-gap", "inf"],
        ):
            with self.subTest(argv=argv[len(TINY) :]):
                self.rejects(argv)

    def test_real_data_accepts_the_learned_mask(self):
        real = ["--dataset-name", "ADNI_stripped_masks", "--dataroot", "/data", "--labels-path", "/labels.csv"]
        real += ["--spatial-size", "96", "112", "96", "--val-frac", "0.2", "--mask-mode", "learned"]
        self.assertEqual(trainer.parse_args(real).mask_mode, "learned")


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.x = images()

    def test_starts_at_the_fixed_split_with_the_fixed_models_weights(self):
        learned, fixed = build_model(config(), "cpu"), build_model(config(mask_mode="fixed"), "cpu")
        state, reference = learned.state_dict(), fixed.state_dict()
        self.assertEqual(set(state) - set(reference), {"mask_logits"})
        for key, value in reference.items():
            torch.testing.assert_close(state[key], value, rtol=0, atol=0)
        self.assertEqual(learned.content_units(), list(range(9)))
        for grid in (None, [2, 2, 2]):
            with self.subTest(grid=grid), torch.no_grad():
                a = learned(self.x, pool_only=True, n_views=2, patch_grid=grid)
                b = fixed(self.x, pool_only=True, n_views=2, patch_grid=grid)
                torch.testing.assert_close(a[2][0], b[2][0], rtol=0, atol=0)
                torch.testing.assert_close(a[6][0], b[6][0], rtol=0, atol=0)
                self.assertEqual(a[3], b[3])

    def test_smaller_initial_gap_preserves_weights_rng_and_evaluation_selection(self):
        legacy = config()
        legacy.pop("mask_init_gap")  # Settings from before this option existed.
        reference = build_model(legacy, "cpu")
        reference_rng = torch.get_rng_state().clone()
        self.assertEqual(trainer.mask_state(reference)["logits"], [2.0] * 9 + [-2.0] * 3)
        with torch.no_grad():
            expected = reference(self.x, pool_only=True, n_views=2)
        for gap in (1.0, 0.0):
            with self.subTest(gap=gap):
                model = build_model(config(mask_init_gap=gap), "cpu")
                torch.testing.assert_close(torch.get_rng_state(), reference_rng, rtol=0, atol=0)
                info = trainer.mask_state(model)
                self.assertEqual(info["margin"], gap)
                self.assertEqual(info["content_units"], list(range(9)))
                self.assertEqual(info["logits"], [gap / 2] * 9 + [-gap / 2] * 3)
                for key, value in reference.state_dict().items():
                    if key != "mask_logits":
                        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
                with torch.no_grad():
                    actual = model(self.x, pool_only=True, n_views=2)
                torch.testing.assert_close(actual[2][0], expected[2][0], rtol=0, atol=0)
                torch.testing.assert_close(actual[6][0], expected[6][0], rtol=0, atol=0)

    def test_loading_a_checkpoint_restores_logits_instead_of_reinitializing_them(self):
        trained = move_units(build_model(config(mask_init_gap=1.0), "cpu"))
        legacy = config()
        legacy.pop("mask_init_gap")
        for cfg in (legacy, config(mask_init_gap=0.0)):
            restored = build_model(cfg, "cpu", trained.state_dict())
            self.assertEqual(restored.content_units(), MOVED)
            torch.testing.assert_close(restored.mask_logits, trained.mask_logits, rtol=0, atol=0)

    def test_construction_rules(self):
        common = dict(hidden_channels=8, res_channels=4, nb_res_layers=1, latent_dim=12, content_channels=9)
        for changes, message in (
            (dict(mask_mode="onthefly"), "fixed or learned"),
            (dict(mask_mode="learned", content_channels=12), "style units"),
            (dict(mask_mode="learned", proj_dim=4), "proj_dim=0"),
            (dict(mask_mode="learned", mask_init_gap=-1), "finite and nonnegative"),
            (dict(mask_mode="learned", mask_init_gap=float("nan")), "finite and nonnegative"),
            (dict(mask_mode="learned", mask_init_gap=float("inf")), "finite and nonnegative"),
            (dict(mask_init_gap=1), "only applies"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                MultiviewConvEncoder(**{**common, **changes})

    def test_forward_and_every_reader_follow_the_selected_units(self):
        model = move_units(build_model(config(lesion_keypoints=1), "cpu"))
        content = MOVED + [12, 13, 14]  # the lesion branch's keypoint coordinates count as content
        self.assertEqual(model.content_units(), MOVED)
        self.assertEqual(model.unit_split(15), (content, STYLE))
        self.assertEqual(model.unit_split(12), (MOVED, STYLE))
        with torch.no_grad():
            out = model(self.x, pool_only=True, n_views=2)
            patches = model(self.x, pool_only=True, n_views=2, patch_grid=[2, 2, 2])
        code = out[2][0]
        self.assertEqual(out[3], [content])
        self.assertEqual(torch.nonzero(out[6][0][0]).flatten().tolist(), content)
        self.assertEqual(patches[3], [MOVED])

        subjects = [
            {
                "image": [self.x[i], self.x[2 + i]],
                "gt_latents": {"z_content": torch.zeros(9), "z_style_v1": torch.zeros(3), "z_style_v2": torch.zeros(3)},
            }
            for i in range(2)
        ]
        blocks = encode_blocks(model, subjects, "cpu", 2, 9)
        for view, rows in enumerate((slice(0, 2), slice(2, 4))):
            np.testing.assert_allclose(blocks["content"][view], code[rows][:, content].numpy(), atol=1e-6)
            np.testing.assert_allclose(blocks["style"][view], code[rows][:, STYLE].numpy(), atol=1e-6)
        with self.assertRaisesRegex(ValueError, "content_channels"):
            encode_blocks(model, subjects, "cpu", 2, 8)

        features, _, _ = batch_features(model, self.x, [1, 2])
        np.testing.assert_allclose(features[(1, "projected")], code[:, content].numpy(), atol=1e-6)
        np.testing.assert_allclose(features[(1, "style")], code[:, STYLE].numpy(), atol=1e-6)
        np.testing.assert_allclose(features[(2, "projected")], patches[2][0][:, MOVED].flatten(1).numpy(), atol=1e-6)

        level = dci._extract_synthetic_representations(model, subjects, "cpu", batch_size=2)[0][0]
        np.testing.assert_allclose(level[0], code[:2][:, content].numpy(), atol=1e-6)
        np.testing.assert_allclose(level[1], code[:2][:, STYLE].numpy(), atol=1e-6)
        np.testing.assert_allclose(level[2], code[2:][:, content].numpy(), atol=1e-6)

        loader = DataLoader([{"image": [self.x[i], self.x[2 + i]], "label": 0} for i in range(2)], batch_size=2)
        with patch.dict(cross_reconstruction._BATCH_CACHE, {"key": None, "batches": None}):
            reps = cross_reconstruction._collect_representations(model, loader, SimpleNamespace(), "cpu")[0]
        np.testing.assert_allclose(reps["content_v1"], code[2:][:, content].numpy(), atol=1e-6)
        np.testing.assert_allclose(reps["style_v0"], code[:2][:, STYLE].numpy(), atol=1e-6)

    def test_separate_spatial_head_stays_all_content(self):
        cfg = config("--separate-spatial-readout", "--patch-loss-weight", "1", "--train-patch-grid", "2", "2", "2")
        model = move_units(build_model(cfg, "cpu"))
        with torch.no_grad():
            patches = model(self.x, pool_only=True, n_views=2, patch_grid=[2, 2, 2])
        self.assertEqual(patches[2][0].shape[1], 9)
        torch.testing.assert_close(patches[6][0], torch.ones(1, 9))
        self.assertEqual(patches[3], [list(range(9))])
        self.assertEqual(model.unit_split(9), (list(range(9)), []))
        features, args = torch.randn(4, 9, 8), SimpleNamespace(**cfg)
        torch.testing.assert_close(
            trainer.patch_contrastive_loss(features, model, args, *CRITERIA, mask=model.sample_content_mask()),
            trainer.patch_contrastive_loss(features, model, args, *CRITERIA),
        )


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_training_draws_are_top_k_and_train_every_logit(self):
        cfg = config()
        model = build_model(cfg, "cpu").train()
        pooled = torch.randn(16, 12)
        for hard in (False, True):
            with self.subTest(hard=hard):
                model.zero_grad()
                mask = model.sample_content_mask(hard=hard)
                self.assertEqual(mask.shape, (1, 12))
                self.assertTrue(mask.requires_grad)
                self.assertAlmostEqual(float(mask.sum()), 9.0, places=4)
                if hard:
                    torch.testing.assert_close(mask, mask.round(), atol=1e-6, rtol=0)
                else:
                    self.assertTrue(((mask > 1e-4) & (mask < 1 - 1e-4)).any())
                trainer.contrastive_loss(pooled, model, SimpleNamespace(**cfg), *CRITERIA, mask=mask).backward()
                grad = model.mask_logits.grad
                self.assertTrue(torch.isfinite(grad).all() and (grad != 0).all())

    def test_binary_mask_losses_equal_the_selected_units_losses(self):
        cfg = config()
        model, args = move_units(build_model(cfg, "cpu")), SimpleNamespace(**cfg)
        mask = model.sample_content_mask()  # eval mode: the selection itself
        self.assertFalse(mask.requires_grad)
        self.assertEqual(torch.nonzero(mask[0]).flatten().tolist(), MOVED)
        pooled = torch.randn(16, 12)
        a, b = F.normalize(pooled[:8, MOVED], dim=1), F.normalize(pooled[8:, MOVED], dim=1)
        expected = F.cross_entropy(a @ b.T / 0.1, torch.arange(8)) + F.cross_entropy(b @ a.T / 0.1, torch.arange(8))
        torch.testing.assert_close(trainer.contrastive_loss(pooled, model, args, *CRITERIA, mask=mask), expected)

        features = torch.randn(6, 12, 5)
        a, b = F.normalize(features[:, MOVED], dim=1).chunk(2)
        expected = (
            sum(
                F.cross_entropy(a[:, :, p] @ b[:, :, p].T / 0.1, torch.arange(3))
                + F.cross_entropy(b[:, :, p] @ a[:, :, p].T / 0.1, torch.arange(3))
                for p in range(5)
            )
            / 5
        )
        torch.testing.assert_close(
            trainer.patch_contrastive_loss(features, model, args, *CRITERIA, mask=mask), expected
        )

        # Unreachable from the CLI (Barlow Twins passes the logits no gradient), but still exact.
        bt = SimpleNamespace(**{**cfg, "contrastive_loss_type": "barlow_twins"})
        sliced = losses.barlow_twins_loss(torch.stack([pooled[:8, MOVED], pooled[8:, MOVED]]), lambd=cfg["bt_lambda"])
        torch.testing.assert_close(trainer.contrastive_loss(pooled, model, bt, *CRITERIA, mask=mask), sliced.squeeze())

    def test_one_draw_per_step_and_held_out_losses_use_the_selection(self):
        cfg = config("--patch-loss-weight", "0.5", "--train-patch-grid", "2", "2", "2", "--lesion-keypoints", "1")
        cfg["lesion_decorrelation_weight"] = 1.0
        model, args, x = move_units(build_model(cfg, "cpu")), SimpleNamespace(**cfg), images()
        _, _, terms = trainer.training_objective(model, x, args, *CRITERIA)
        self.assertNotIn("mask_leak", terms)
        with torch.no_grad():
            pooled, patches = model.global_and_patch_features(x, [2, 2, 2])
        mask = torch.zeros(1, 12)
        mask[0, MOVED] = 1.0
        torch.testing.assert_close(terms["global"], trainer.contrastive_loss(pooled, model, args, *CRITERIA, mask=mask))
        torch.testing.assert_close(
            terms["patch"], trainer.patch_contrastive_loss(patches, model, args, *CRITERIA, mask=mask)
        )
        torch.testing.assert_close(terms["lesion_decorrelation"], trainer.lesion_decorrelation(pooled, args, MOVED))

        model.train()
        for estimator, hard in (("relaxed", False), ("straight_through", True)):
            with self.subTest(estimator=estimator):
                with patch.object(model, "sample_content_mask", wraps=model.sample_content_mask) as draw:
                    _, _, terms = trainer.training_objective(
                        model, x, SimpleNamespace(**{**cfg, "mask_estimator": estimator}), *CRITERIA
                    )
                self.assertEqual((draw.call_count, draw.call_args.kwargs), (1, {"hard": hard}))
                self.assertGreaterEqual(float(terms["mask_leak"]), 0.0)

    def test_relaxed_mask_swaps_a_noisy_content_unit_for_a_shared_one(self):
        """Fixed features, logits only: content unit 8 is per-view noise, style unit 9 is shared.

        Taking 9 instead of 8 cuts this InfoNCE 14- to 29-fold (seeds 0-9 of this setup). The relaxed
        estimator made the swap in 10/10 of those seeds, the straight-through one in 0/10: cosine
        similarity has no gradient for a unit zeroed in both views, so under binary draws unit 9's
        logit gets a signal only when a draw includes it.
        """
        torch.manual_seed(0)
        cfg = config()
        model, args = build_model(cfg, "cpu").train(), SimpleNamespace(**cfg)
        g = torch.Generator().manual_seed(100)
        shared = torch.randn(64, 12, generator=g)
        views = [shared + 0.1 * torch.randn(64, 12, generator=g) for _ in range(2)]
        for view in views:
            view[:, 8] = 3 * torch.randn(64, generator=g)
        pooled = torch.cat(views)
        pooled[:, 10:] = torch.randn(128, 2, generator=g)
        optimizer = torch.optim.Adam([model.mask_logits], lr=0.05)
        for _ in range(2000):
            loss = trainer.contrastive_loss(pooled, model, args, *CRITERIA, mask=model.sample_content_mask())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertEqual(model.content_units(), [0, 1, 2, 3, 4, 5, 6, 7, 9])


class RunTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_synthetic_run_trains_the_logits_and_records_the_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [*LEARNED, "--mask-lr-scale", "1000", "--out-dir", tmp, "--model-id", "gumbel"]
            argv += ["--mask-init-gap", "1"]
            argv += ["--require-new-run", "--device", "cpu", "--batch-size", "2", "--train-steps", "2"]
            argv += ["--eval-every", "2", "--log-every", "1", "--num-train-samples", "8", "--num-val-samples", "20"]
            argv += ["--no-cache", "--synthetic-clean-content", "--synthetic-normalize", "fixed_reference"]
            # The real loop and evaluate(); only the expensive factor probes are stubbed.
            with patch.object(sys, "argv", ["trainer", *argv]), patch.object(
                trainer.dci, "compute_dci_synthetic", return_value={}
            ), patch.object(trainer.dci, "flatten_dci_results", return_value={}), contextlib.redirect_stdout(
                io.StringIO()
            ) as output:
                trainer.main()
            run, text = Path(tmp) / "gumbel", output.getvalue()
            self.assertIn("content mask: learned top-9 of 12 units, Gumbel-softmax (tau 1, relaxed)", text)
            self.assertIn("NOTE: --lr x --mask-lr-scale x --train-steps = 0.2", text)
            self.assertIn("logits +0.5/-0.5, initial gap 1", text)
            self.assertIn("| mask leak ", text)
            floor = json.loads((run / "dci_step0.json").read_text())["content_mask"]
            trained = json.loads((run / "dci_step2.json").read_text())["content_mask"]
            self.assertEqual((floor["content_units"], floor["logits"]), (list(range(9)), [0.5] * 9 + [-0.5] * 3))
            self.assertNotEqual(trained["logits"], floor["logits"])
            progress = json.loads((run / "training_progress.json").read_text())
            self.assertAlmostEqual(progress["content_mask"]["lr"], 0.1)
            self.assertIn("mask_leak", progress["last_loss_terms"])
            cfg = json.loads((run / "settings.json").read_text())
            self.assertEqual(cfg["mask_init_gap"], 1.0)
            restored = build_model(cfg, "cpu", torch.load(run / "model.pt", weights_only=True))
        self.assertEqual(restored.content_units(), trained["content_units"])
        torch.testing.assert_close(restored.mask_logits.detach(), torch.tensor(trained["logits"]))

    def test_adni_run_records_the_selection(self):
        from tests.test_encoder_adni import SHAPE
        from tests.test_encoder_adni import TINY as REAL_TINY
        from tests.test_encoder_adni import fake_adni, run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = ["--dataset-name", "ADNI_stripped_masks", "--dataroot", tmp, "--labels-path", str(fake_adni(root))]
            argv += ["--spatial-size", *map(str, SHAPE), "--image-spacing", "1", "--val-frac", "0.4"]
            argv += ["--test-frac", "0.1", "--out-dir", str(root / "runs"), "--device", "cpu", *REAL_TINY]
            argv += ["--mask-mode", "learned", "--mask-lr-scale", "100", "--model-id", "gumbel", "--no-cache"]
            output = run(argv)
            floor = json.loads((root / "runs" / "gumbel" / "separation_step0.json").read_text())
            trained = json.loads((root / "runs" / "gumbel" / "separation_step2.json").read_text())
        self.assertIn("content units [0, 1, 2, 3, 4, 5, 6, 7, 8] (learned mask", output)
        self.assertEqual(floor["content_mask"]["content_units"], list(range(9)))
        self.assertNotEqual(trained["content_mask"]["logits"], floor["content_mask"]["logits"])
        self.assertIn("val/loss", trained)


class LauncherTests(unittest.TestCase):
    def test_learned_mask_experiments_resolve_on_every_cluster(self):
        from scripts import launch

        experiments = Path(__file__).resolve().parents[1] / "experiments" / "encoder"
        for name in ("adni_gumbel", "conv_mlp_gumbel"):
            for cluster in ("runai", "slurm", "slurm_bio"):
                with self.subTest(name=name, cluster=cluster):
                    config = launch.resolve_config(experiments / f"{name}.yaml", cluster, {})
                    args = trainer.parse_args(launch.encoder_cli_args(config))
                    self.assertEqual(
                        (args.mask_mode, args.mask_lr_scale, args.mask_estimator), ("learned", 10.0, "relaxed")
                    )
        override = {"mask_estimator": "straight_through", "mask_lr_scale": 3}
        config = launch.resolve_config(experiments / "adni_gumbel.yaml", "runai", override)
        args = trainer.parse_args(launch.encoder_cli_args(config))
        self.assertEqual((args.mask_lr_scale, args.mask_estimator), (3.0, "straight_through"))
        fixed = launch.encoder_cli_args(launch.resolve_config(experiments / "adni.yaml", "runai", {}))
        self.assertFalse(any(token.startswith("--mask-") for token in fixed))


if __name__ == "__main__":
    unittest.main()
