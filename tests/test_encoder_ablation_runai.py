"""Generated jobs preserve pairing, parse in the real trainer and never submit in preview."""

import contextlib
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts import generate_encoder_ablation_runai as generator
from training.main_conv_synthetic import parse_args


class EncoderAblationRunaiTests(unittest.TestCase):
    def setUp(self):
        self.config = generator.comparison.read_json(generator.ROOT / "experiments/encoder_comparison.json")
        self.variants = generator.launch.load_yaml(generator.ROOT / "experiments/encoder_ablations/variants.yaml")

    def test_each_variant_differs_from_control_in_one_architectural_option(self):
        for name, variant in self.variants.items():
            candidate = generator.training_options(self.config, variant, name, 42, "/runs")
            reference = generator.training_options(
                self.config, {"reference": variant["reference"], "overrides": {}}, "control", 42, "/runs"
            )
            self.assertEqual(
                {key for key in candidate if candidate[key] != reference[key]}, {"model_id", *variant["overrides"]}
            )
            parsed = parse_args(generator.comparison.cli_arguments(candidate))
            for key, value in candidate.items():
                self.assertEqual(getattr(parsed, key), value)
            self.assertEqual(parsed.loader_seed, 10042)
            self.assertEqual(parsed.data_seed, 42)
            self.assertTrue(parsed.deterministic_warn_only)

    def test_generated_shell_submission_preserves_arguments_and_quoting(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "scripts"
            with contextlib.redirect_stdout(io.StringIO()):
                generator.main(
                    [
                        "--output-dir",
                        str(output),
                        "--repo-path",
                        "/nfs/repo with spaces",
                        "--results-dir",
                        "results/ablation with spaces",
                        "--include-controls",
                    ]
                )
            scripts = sorted(output.glob("*.runai.sh"))
            self.assertEqual(len(scripts), 5)
            # Substitute a local argument recorder for runai; never contact a cluster.
            fake = Path(temp) / "runai"
            fake.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
            fake.chmod(0o755)
            env = {**os.environ, "PATH": str(Path(temp)) + os.pathsep + os.environ["PATH"]}
            for script in scripts:
                subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
                preview = subprocess.check_output(["bash", str(script), "--dry-run"], env=env, text=True).strip()
                captured = json.loads(subprocess.check_output(["bash", str(script)], env=env, text=True))
                self.assertEqual(captured[:3], ["training", "standard", "submit"])
                self.assertEqual(captured[-5:-1], ["--command", "--", "bash", "-c"])
                self.assertEqual(captured[-1].strip(), preview)
                self.assertEqual(captured[captured.index("--gpu-devices-request") + 1], "1")
                self.assertIn("path=/nfs,mount=/nfs,readwrite", captured)
                subprocess.run(["bash", "-n", "-c", preview], check=True, capture_output=True)
                tokens = shlex.split(preview)
                args = parse_args(tokens[tokens.index("training.main_conv_synthetic") + 1 :])
                self.assertEqual(args.out_dir, "/nfs/repo with spaces/results/ablation with spaces/runs")
                self.assertEqual(args.train_steps, 10000)
                self.assertIn("OMP_NUM_THREADS=1", tokens)
                self.assertIn("--no-cache", tokens)
                self.assertNotIn("--use-wandb", tokens)

    def test_invalid_architecture_options_are_not_silently_ignored(self):
        bad = [
            ["--resnet-norm", "group"],
            ["--resnet-output-stride", "8"],
            ["--encoder-architecture", "resnet18", "--conv-readout", "mlp"],
            ["--conv-readout", "mlp", "--encoder-head-hidden", "0"],
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            for args in bad:
                with self.subTest(args=args), self.assertRaises(SystemExit):
                    parse_args(args)
        self.assertEqual(
            parse_args(
                [
                    "--encoder-architecture",
                    "resnet18",
                    "--resnet-output-stride",
                    "8",
                    "--res",
                    "64",
                    "--eval-pooling",
                    "patch",
                    "--eval-patch-grid",
                    "8",
                    "8",
                    "8",
                ]
            ).resnet_output_stride,
            8,
        )


if __name__ == "__main__":
    unittest.main()
