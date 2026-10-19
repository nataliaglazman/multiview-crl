"""--init-from in the encoder-only trainer: an exact warm start, its provenance, and refused architectures."""

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from training import encoder_args

if importlib.util.find_spec("lpips") is None:
    # training.losses imports LPIPS at module scope; the encoder-only trainer never builds it.
    sys.modules["lpips"] = types.ModuleType("lpips")
    sys.modules["lpips"].LPIPS = None
from training import main_conv_synthetic as trainer  # noqa: E402

TINY = ["--res", "16", "--hidden-channels", "8", "--res-channels", "4", "--nb-res-layers", "1", "--latent-dim", "12"]
TINY += ["--content-channels", "9", "--conv-readout", "mlp", "--encoder-head-hidden", "8", "--tau", "0.1"]
TINY += ["--best-metric", "none", "--batch-size", "2", "--lr", "0.05", "--log-every", "1", "--device", "cpu"]
TINY += ["--num-train-samples", "8", "--num-val-samples", "20", "--no-cache", "--require-new-run"]
TINY += ["--synthetic-clean-content", "--synthetic-normalize", "fixed_reference"]
GRIDS = (None, [2, 2, 2])  # the global code, and the patch code through the spatial readout


def record_codes(codes, x):
    """An ``evaluate`` stand-in that keeps the model's global and patch codes of ``x`` at each step."""

    def evaluate(model, val_dataset, device, args, save_dir, step, writer=None, floor=None):
        with torch.no_grad():
            codes[step] = [model(x, pool_only=True, n_views=2, patch_grid=grid)[2][0].clone() for grid in GRIDS]
        return {}, {}

    return evaluate


def train(out_dir, model_id, *extra, evaluate=None):
    """The real trainer loop on a tiny synthetic bank; only the evaluation is replaced."""
    argv = ["trainer", "--out-dir", str(out_dir), "--model-id", model_id, *TINY, *extra]
    stub = evaluate or (lambda *args, **kwargs: ({}, {}))
    with patch.object(sys, "argv", argv), patch.object(trainer, "evaluate", side_effect=stub):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            trainer.main()
    return Path(out_dir) / model_id, output.getvalue()


def refusal(test, out_dir, model_id, *extra):
    """The trainer's exit message for a refused warm start; the run directory must not exist after it."""
    with test.assertRaises(SystemExit) as raised:
        train(out_dir, model_id, *extra)
    test.assertFalse((Path(out_dir) / model_id).exists())
    return str(raised.exception.code)


class InitFromTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_warm_start_reproduces_the_source_code_exactly_and_records_its_provenance(self):
        x = torch.randn(4, 1, 16, 16, 16, generator=torch.Generator().manual_seed(0))
        source_codes, warm_codes = {}, {}
        with tempfile.TemporaryDirectory() as tmp:
            source, _ = train(
                tmp, "source", "--train-steps", "2", "--eval-every", "2", evaluate=record_codes(source_codes, x)
            )
            digest = hashlib.sha256((source / "model.pt").read_bytes()).hexdigest()
            warm, output = train(
                tmp,
                "warm",
                *("--train-steps", "1", "--eval-every", "1", "--init-from", str(source)),
                evaluate=record_codes(warm_codes, x),
            )
            source_state = torch.load(source / "model.pt", weights_only=True)
            initial = torch.load(warm / "model_init.pt", weights_only=True)
            trained = torch.load(warm / "model.pt", weights_only=True)
            settings = json.loads((warm / "settings.json").read_text())
            progress = json.loads((warm / "training_progress.json").read_text())
            fresh = json.loads((source / "training_progress.json").read_text())
            replay = encoder_args.parse_args(encoder_args.cli_arguments(settings))
            # The source retrained since: a relaunch with the recorded digest refuses its new weights.
            changed = {key: value.clone() for key, value in source_state.items()}
            changed[next(key for key, value in changed.items() if value.is_floating_point())] += 1
            torch.save(changed, source / "model.pt")
            message = refusal(
                self, tmp, "replay", "--init-from", str(source / "model.pt"), "--init-from-sha256", digest
            )

        # Same seeds, so a warm start that loaded nothing would read the source's step 0 instead.
        self.assertFalse(torch.equal(source_codes[0][0], source_codes[2][0]))
        for grid, a, b in zip(GRIDS, source_codes[2], warm_codes[0]):
            with self.subTest(grid=grid):
                self.assertTrue(torch.equal(a, b))
        self.assertEqual(initial.keys(), source_state.keys())
        self.assertTrue(all(torch.equal(initial[key], source_state[key]) for key in source_state))
        self.assertFalse(all(torch.equal(trained[key], initial[key]) for key in initial))
        self.assertIn(f"init: warm start from {source / 'model.pt'} (sha256 {digest}, recorded at its step 2)", output)

        self.assertEqual((settings["init_from"], settings["init_from_sha256"]), (str(source), digest))
        self.assertEqual(replay.init_from_sha256, digest)
        self.assertEqual((progress["step"], progress["status"]), (1, "complete"))
        self.assertEqual(
            progress["init_from"],
            {
                "path": str(source),
                "checkpoint": str(source / "model.pt"),
                "sha256": digest,
                "source_step": 2,
                "source_status": "complete",
            },
        )
        self.assertIsNone(fresh["init_from"])
        self.assertIn("checkpoint changed since that was recorded", message)

    def test_another_architecture_is_refused_before_the_run_directory_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            torch.save(trainer.build_encoder(encoder_args.parse_args(TINY)).state_dict(), source / "model.pt")
            for extra, expected in (
                (["--latent-dim", "16"], "strict load_state_dict refused it"),
                (["--norm-type", "layer"], "strict load_state_dict refused it"),
                (["--conv-readout", "linear"], "strict load_state_dict refused it"),
                # Every shape matches; only the content split moves.
                (["--content-channels", "6"], "its fixed content mask marks units [0, 1, 2, 3, 4, 5, 6, 7, 8]"),
            ):
                with self.subTest(extra=extra):
                    message = refusal(self, tmp, "warm", "--train-steps", "1", "--init-from", str(source), *extra)
                    self.assertIn("not this run's architecture", message)
                    self.assertIn(expected, message)
            message = refusal(self, tmp, "warm", "--train-steps", "1", "--init-from", str(Path(tmp) / "missing"))
            self.assertIn("need a checkpoint file, or a run directory with model.pt", message)

    def test_cli_checks_the_recorded_digest(self):
        args = encoder_args.parse_args([])
        self.assertEqual((args.init_from, args.init_from_sha256), (None, None))
        args = encoder_args.parse_args(["--init-from", "run", "--init-from-sha256", "AB" * 32])
        self.assertEqual((args.init_from, args.init_from_sha256), ("run", "ab" * 32))
        for argv in (
            ["--init-from-sha256", "ab" * 32],
            ["--init-from", "run", "--init-from-sha256", "ab" * 31],
            ["--init-from", "run", "--init-from-sha256", "xy" * 32],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                encoder_args.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
