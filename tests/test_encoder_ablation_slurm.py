"""SLURM resource resolution, paired CLI parity and batch execution without a cluster."""

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

from scripts import generate_encoder_ablation_slurm as generator
from training.main_conv_synthetic import parse_args


class EncoderAblationSlurmTests(unittest.TestCase):
    def generate(self, output, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            generator.main(["--output-dir", str(output), "--include-controls", *extra])
        return sorted(output.glob("*.slurm_bio.sh"))

    def test_scripts_use_bio_resources_and_same_training_options_as_runai(self):
        with tempfile.TemporaryDirectory() as temp:
            scripts = self.generate(Path(temp))
            self.assertEqual(len(scripts), 5)
            config = generator.ablations.comparison.read_json(generator.ROOT / "experiments/encoder_comparison.json")
            variants = generator.ablations.launch.load_yaml(
                generator.ROOT / "experiments/encoder_ablations/variants.yaml"
            )
            env = {**os.environ, "ENCODER_REPO": "/cluster/repo with spaces", "ENCODER_PYTHON": "/env/bin/python"}
            for script in scripts:
                subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
                text = script.read_text()
                for flag in (
                    "--partition=biomed_a100_gpu",
                    "--constraint=a100_80g",
                    "--time=48:00:00",
                    "--gres=gpu:1",
                    "--mem=64G",
                    "--cpus-per-task=8",
                    "--ntasks=1",
                ):
                    self.assertIn("#SBATCH " + flag, text)
                self.assertNotIn("conda env remove", text)
                self.assertNotIn("pip install", text)
                self.assertNotIn("#SBATCH", text.split("set -euo pipefail", 1)[1])
                preview = subprocess.check_output(["bash", str(script), "--dry-run"], env=env, text=True)
                tokens = shlex.split(preview)
                parsed = parse_args(tokens[3:])
                name = parsed.model_id.removesuffix("_s42")
                variant = variants.get(name, {"reference": name, "overrides": {}})
                expected = generator.ablations.training_options(
                    config, variant, name, 42, "/cluster/repo with spaces/results/encoder_ablations_slurm_bio"
                )
                self.assertEqual(tokens[:3], ["/env/bin/python", "-m", "training.main_conv_synthetic"])
                for key, value in expected.items():
                    self.assertEqual(getattr(parsed, key), value)

    def test_batch_execution_quotes_paths_and_stops_on_failed_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            scripts = self.generate(base / "scripts", "--results-dir", "results/with spaces")
            repo = base / "repo with spaces"
            (repo / "training").mkdir(parents=True)
            (repo / "training/main_conv_synthetic.py").touch()
            fake_bin = base / "bin"
            fake_bin.mkdir()
            module = fake_bin / "module"
            module.write_text("#!/bin/sh\nexit 0\n")
            module.chmod(0o755)
            fake_python = fake_bin / "python with spaces"
            fake_python.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "args = sys.argv[1:]\n"
                "with open(os.environ['ABLATION_TEST_LOG'], 'a') as stream:\n"
                "    stream.write(json.dumps({'args': args, 'cwd': os.getcwd(), "
                "'threads': os.environ.get('OMP_NUM_THREADS'), "
                "'cublas': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}) + '\\n')\n"
                "stage = 'cuda' if args[0] == '-c' else ('tests' if args[1] == 'unittest' else 'train')\n"
                "sys.exit(7 if stage == os.environ.get('ABLATION_FAIL_STAGE') else 0)\n"
            )
            fake_python.chmod(0o755)
            log = base / "invocations.jsonl"
            env = {key: value for key, value in os.environ.items() if key not in ("ENCODER_REPO", "BASH_ENV")}
            env.update(
                ENCODER_PYTHON=str(fake_python),
                SLURM_SUBMIT_DIR=str(repo),
                SLURM_JOB_ID="123",
                ABLATION_TEST_LOG=str(log),
                PATH=str(fake_bin) + os.pathsep + os.environ["PATH"],
            )
            script = next(p for p in scripts if "conv_mlp" in p.name)
            # Preview and accidental direct execution cannot launch Python or load modules.
            direct_env = {key: value for key, value in env.items() if key != "SLURM_JOB_ID"}
            subprocess.run(["bash", str(script), "--dry-run"], env=direct_env, check=True, capture_output=True)
            direct = subprocess.run(["bash", str(script)], env=direct_env, capture_output=True, text=True)
            self.assertEqual(direct.returncode, 2)
            self.assertIn("Submit with sbatch", direct.stderr)
            self.assertFalse(log.exists())
            for stage, count in (("cuda", 1), ("tests", 2), ("none", 3)):
                with self.subTest(stage=stage):
                    if log.exists():
                        log.unlink()
                    result = subprocess.run(
                        ["bash", str(script)], env={**env, "ABLATION_FAIL_STAGE": stage}, capture_output=True, text=True
                    )
                    self.assertEqual(result.returncode, 0 if stage == "none" else 7, result.stderr)
                    calls = [json.loads(line) for line in log.read_text().splitlines()]
                    self.assertEqual(len(calls), count)
                    for call in calls:
                        self.assertEqual(Path(call["cwd"]).resolve(), repo.resolve())
                        self.assertEqual(call["threads"], "1")
                        self.assertEqual(call["cublas"], ":4096:8")
                    if stage == "none":
                        args = parse_args(calls[-1]["args"][2:])
                        self.assertEqual(args.out_dir, str(repo / "results/with spaces/runs"))
                        self.assertEqual(args.conv_readout, "mlp")

    def test_resource_namespaces_and_three_seed_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            cluster = base / "cluster.yaml"
            cluster.write_text("_slurm:\n  partition: old\n_slurm_bio:\n  partition: bio\n")
            self.assertEqual(generator.load_resources(cluster), {"partition": "bio"})
            scripts = self.generate(base / "scripts", "--seeds", "42", "142", "242", "--repo-path", "/remote/repo")
            self.assertEqual(len(scripts), 15)
            env = {key: value for key, value in os.environ.items() if key != "ENCODER_REPO"}
            for script in scripts:
                preview = subprocess.check_output(["bash", str(script), "--dry-run"], env=env, text=True)
                args = parse_args(shlex.split(preview)[3:])
                self.assertEqual(args.data_seed, 42)
                self.assertEqual(args.model_seed, args.seed)
                self.assertEqual(args.loader_seed, args.seed + 10000)
                self.assertEqual(args.out_dir, "/remote/repo/results/encoder_ablations_slurm_bio/runs")


if __name__ == "__main__":
    unittest.main()
