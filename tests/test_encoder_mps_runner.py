"""Local GPU launchers preserve the recipe, changing only device and output location."""

import argparse
import ast
import contextlib
import io
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts import run_encoder_mps as runner

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/generated/encoder_resnet_stride8_s42.mps.sh"


def trainer_parse_args():
    # The real parser, without importing the trainer (training.losses needs LPIPS).
    path = ROOT / "training/main_conv_synthetic.py"
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_args"]
    namespace = {"argparse": argparse}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["parse_args"]


def runner_args(**overrides):
    values = dict(
        config=ROOT / "experiments/encoder_comparison.json",
        variant="resnet_stride8",
        seed=42,
        batch_size=None,
        train_steps=None,
        eval_every=None,
        model_id=None,
        results_dir=Path("/tmp/encoder_ablations_mps"),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class EncoderMpsRunnerTests(unittest.TestCase):
    def test_command_is_the_cluster_recipe_except_device_and_location(self):
        config = runner.comparison.read_json(ROOT / "experiments/encoder_comparison.json")
        variants = runner.ablations.launch.load_yaml(ROOT / "experiments/encoder_ablations/variants.yaml")
        cluster = runner.ablations.training_options(config, variants["resnet_stride8"], "resnet_stride8", 42, "/runs")
        local, _ = runner.make_options(runner_args())
        changed = {key for key in local.keys() | cluster.keys() if local.get(key) != cluster.get(key)}
        self.assertEqual(changed, {"device", "out_dir", "model_id"})
        self.assertEqual(local["model_id"], "resnet_stride8_s42_mps")
        parsed = trainer_parse_args()(runner.comparison.cli_arguments(local))
        for key, value in local.items():
            self.assertEqual(getattr(parsed, key), value)

    def test_overridden_runs_never_take_the_recipe_run_directory(self):
        options, recipe = runner.make_options(runner_args(batch_size=16, train_steps=200))
        self.assertEqual(options["model_id"], "resnet_stride8_s42_mps_b16_t200")
        self.assertEqual(recipe["batch_size"], 32)
        options, _ = runner.make_options(runner_args(eval_every=500))
        self.assertEqual(options["model_id"], "resnet_stride8_s42_mps")
        for bad in (dict(batch_size=1), dict(model_id="../elsewhere"), dict(seed=7), dict(variant="missing")):
            with self.assertRaises(ValueError):
                runner.make_options(runner_args(**bad))

    def test_shell_wrapper_previews_the_runner_command(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        env = {**os.environ, "ENCODER_PYTHON": sys.executable}
        preview = subprocess.check_output(["bash", str(SCRIPT), "--dry-run"], env=env, text=True).strip()
        with contextlib.redirect_stdout(io.StringIO()) as direct:
            runner.main(["--dry-run"])
        self.assertEqual(preview, direct.getvalue().strip())
        tokens = shlex.split(preview)
        self.assertEqual(tokens[:3], [sys.executable, "-m", "training.main_conv_synthetic"])
        parsed = trainer_parse_args()(tokens[3:])
        self.assertEqual((parsed.device, parsed.batch_size, parsed.resnet_output_stride), ("mps", 32, 8))

    def test_cuda_patch_launcher_matches_mps_and_has_separate_outputs(self):
        env = {**os.environ, "ENCODER_PYTHON": sys.executable}
        commands = {}
        for device in ("mps", "cuda"):
            script = ROOT / f"experiments/generated/encoder_conv_mlp_patch_s42.{device}.sh"
            subprocess.run(["bash", "-n", str(script)], check=True)
            preview = subprocess.check_output(["bash", str(script), "--dry-run"], env=env, text=True)
            commands[device] = vars(trainer_parse_args()(shlex.split(preview)[3:]))
        cuda, mps = commands["cuda"], commands["mps"]
        self.assertEqual({key for key in cuda if cuda[key] != mps[key]}, {"device", "out_dir", "model_id"})
        self.assertEqual(cuda["device"], "cuda")
        self.assertEqual(cuda["model_id"], "conv_mlp_s42_cuda_patch8x8x8_w1")
        self.assertEqual(cuda["out_dir"], str(ROOT / "results/encoder_patch_cuda/runs"))
        self.assertEqual((cuda["conv_readout"], cuda["batch_size"], cuda["train_steps"]), ("mlp", 32, 10000))
        self.assertEqual((cuda["patch_loss_weight"], cuda["train_patch_grid"]), (1, [8, 8, 8]))
        self.assertTrue(cuda["deterministic_warn_only"])
        small, _ = runner.make_options(
            runner_args(device="cuda", variant="conv_mlp", batch_size=8, patch_loss_weight=1)
        )
        self.assertEqual(small["model_id"], "conv_mlp_s42_cuda_b8_patch8x8x8_w1")

    @unittest.skipUnless(torch.cuda.is_available(), "NVIDIA GPU required")
    def test_cuda_backend_check_runs_patch_update_without_writing_a_run(self):
        with tempfile.TemporaryDirectory() as temp:
            env = {**os.environ, "ENCODER_PYTHON": sys.executable}
            script = ROOT / "experiments/generated/encoder_conv_mlp_patch_s42.cuda.sh"
            result = subprocess.run(
                ["bash", str(script), "--check", "--batch-size", "2", "--results-dir", temp],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Backend check passed: CUDA matches CPU", result.stdout)
            self.assertEqual(list(Path(temp).iterdir()), [], "--check must not write a run")

    @unittest.skipUnless(torch.backends.mps.is_available(), "Apple-silicon GPU required")
    def test_backend_check_runs_a_real_step_and_matches_cpu(self):
        # torch reads PYTORCH_ENABLE_MPS_FALLBACK once, at import, so the check needs a fresh process.
        with tempfile.TemporaryDirectory() as temp:
            env = {**os.environ, "PYTHONPATH": str(ROOT)}
            env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
            command = [sys.executable, str(ROOT / "scripts/run_encoder_mps.py"), "--check", "--batch-size", "2"]
            result = subprocess.run(
                [*command, "--results-dir", temp], env=env, cwd=temp, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Backend check passed: MPS matches CPU", result.stdout)
            self.assertEqual(list(Path(temp).iterdir()), [], "--check must not write a run")


if __name__ == "__main__":
    unittest.main()
