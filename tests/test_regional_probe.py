"""eval.encoder.regional_probe: the toy readouts, parity with the model's own forward, and two real runs."""

import contextlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from eval.encoder import regional_probe as probe

if importlib.util.find_spec("lpips") is None:
    # training.losses imports LPIPS at module scope; the encoder-only trainer never builds it.
    sys.modules["lpips"] = types.ModuleType("lpips")
    sys.modules["lpips"].LPIPS = None
from eval.diagnostics.style_path_audit import effective_style  # noqa: E402
from eval.protocol.score_checkpoint import build_model, load_settings, make_dataset  # noqa: E402
from training import main_conv_synthetic as trainer  # noqa: E402

TINY = ["--res", "32", "--hidden-channels", "8", "--res-channels", "4", "--nb-res-layers", "1", "--latent-dim", "12"]
TINY += ["--content-channels", "9", "--conv-readout", "mlp", "--encoder-head-hidden", "8", "--tau", "0.1"]
TINY += ["--best-metric", "none", "--batch-size", "2", "--lr", "0.05", "--log-every", "1", "--device", "cpu"]
TINY += ["--num-train-samples", "8", "--num-val-samples", "64", "--no-cache", "--require-new-run"]
TINY += ["--synthetic-clean-content", "--synthetic-normalize", "fixed_reference"]
TINY += ["--synthetic-lesion-placement", "wm_interior", "--train-steps", "1", "--eval-every", "1"]
PATCH = ["--patch-loss-weight", "1", "--train-patch-grid", "4", "4", "4", "--patch-foreground-mask"]


def train(out_dir, model_id, extra=()):
    argv = ["trainer", "--out-dir", str(out_dir), "--model-id", model_id, *TINY, *extra]
    with patch.object(sys, "argv", argv), patch.object(trainer, "evaluate", return_value=({}, {})):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.main()
    return Path(out_dir) / model_id


class RegionalProbeTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())

    def test_toy_readouts(self):
        with contextlib.redirect_stdout(io.StringIO()):
            probe.self_test()

    def test_adaptive_mean_matches_torch(self):
        cells = np.random.default_rng(0).normal(size=(3, 2, 5 * 4 * 3))
        ours = probe.adaptive_mean(cells, (5, 4, 3), (2, 3, 2))
        theirs = F.adaptive_avg_pool3d(torch.tensor(cells).reshape(3, 2, 5, 4, 3), (2, 3, 2)).flatten(1).numpy()
        np.testing.assert_allclose(ours, theirs, atol=1e-12)

    def test_runs_are_scored_on_shared_subjects_with_readouts_matching_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = train(tmp, "plain")
            regional = train(tmp, "regional", PATCH)
            args = probe.parse_args(
                [
                    "--run-dirs",
                    str(plain),
                    str(regional),
                    "--device",
                    "cpu",
                    "--batch-size",
                    "8",
                    "--probe-grid",
                    "2",
                    "2",
                    "2",
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()) as output:
                report = probe.run(args)
            # The regional readout repeats forward's pool_only + patch_grid path exactly.
            cfg = load_settings(regional)
            model = build_model(cfg, "cpu", torch.load(regional / "model.pt", weights_only=True)).eval()
            batch = next(iter(torch.utils.data.DataLoader(make_dataset(cfg, 4), batch_size=4)))
            x = torch.cat(batch["image"], dim=0)
            with torch.no_grad():
                code, cells, maps = probe.readouts(model, x, [4, 4, 4], [4, 4, 4])
                out = model(x, pool_only=True, n_views=2, patch_grid=[4, 4, 4])
                forward_cells = out[2][0][:, model.unit_split(out[2][0].shape[1])[0]].flatten(1)
                forward_code = model(x, pool_only=True, n_views=2)[2][0]
            torch.testing.assert_close(cells, forward_cells)
            torch.testing.assert_close(code, forward_code[:, model.unit_split(forward_code.shape[1])[0]])
            self.assertEqual(maps.shape[1], 8 * 64)
            # Each view is scored against its own rendered style: recompute B's FLAIR gain row directly.
            dataset = make_dataset(cfg, cfg["num_val_samples"])
            scale = dataset._inner.renderer.style_scale
            flair_codes, flair_gain = [], []
            with torch.no_grad():
                for batch in torch.utils.data.DataLoader(dataset, batch_size=8):
                    x = torch.cat(batch["image"], dim=0)
                    flair_codes.append(
                        probe.readouts(model, x, [4, 4, 4], [2, 2, 2])[0][x.shape[0] // 2 :].double().numpy()
                    )
                    flair_gain += [effective_style(z, scale)[0] for z in batch["gt_latents"]["z_style_v2"]]
            expected_flair_gain = probe.r2(np.concatenate(flair_codes), np.array(flair_gain))

        self.assertEqual(len(report["data_sets"]), 1)  # the two runs differ only in training, not in subjects
        self.assertEqual(
            [c["letter"] + " " + c["name"] for c in report["columns"]],
            ["A model_init.pt", "A model.pt", "B model_init.pt", "B model.pt"],
        )
        trained = report["columns"][3]["scores"]
        for view in probe.VIEWS:
            self.assertEqual(set(trained[view]), set(probe.READOUTS))
            names = set(trained[view]["code"])
            self.assertIn("brain_size", names)
            self.assertIn("lesion y within hemisphere", names)
            self.assertNotIn("lesion_y", names)  # the single lesion is scored by its rendered centre
            self.assertTrue(np.isfinite(trained[view]["regional"]["brain_size"]))
        anatomy = report["data_sets"][0]["anatomy_only"]
        self.assertEqual(set(anatomy), {n for n in names if n.startswith("lesion ")})
        style = ["style gain", "style bias", "style noise_sigma"]
        self.assertEqual(report["data_sets"][0]["style_rows"], {"t1": style, "flair": style})
        for column in report["columns"]:
            for view in probe.VIEWS:
                for readout in probe.READOUTS:
                    self.assertTrue(all(np.isfinite(column["scores"][view][readout][s]) for s in style))
        self.assertAlmostEqual(trained["flair"]["code"]["style gain"], expected_flair_gain, places=10)
        text = output.getvalue()
        self.assertIn("Style exclusion", text)
        self.assertIn("natural readout", text)
        self.assertIn("the trained spatial head", text)
        self.assertIn("the global head per cell", text)


if __name__ == "__main__":
    unittest.main()
