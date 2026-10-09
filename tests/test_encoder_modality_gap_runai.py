"""Exercise the Run:ai wrapper through both shell boundaries without submitting a job."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_encoder_adni_modality_gap_runai.sh"


class ModalityGapLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo with spaces"
        self.repo.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.env = {
            **os.environ,
            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "REPO_PATH": str(self.repo),
            "MODEL_ID": "encoder_adni_conv_mlp_layernorm_s42",
            "RUN_DIR": str(self.repo / "runs/model"),
            "CHECKPOINT": "model_best.pt",
            "BATCH_SIZE": "4",
            "JOB_NAME": "adni-probe-test",
            "PROBE_TEST_SUBMIT": str(self.root / "submit.json"),
            "PROBE_TEST_PYTHON": str(self.root / "python.json"),
            "PROBE_TEST_EXIT": "0",
        }
        self.stub(
            "runai",
            "import json, os, subprocess, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['PROBE_TEST_SUBMIT']).write_text(json.dumps(sys.argv[1:]))\n"
            "command = sys.argv[sys.argv.index('--') + 1:]\n"
            "raise SystemExit(subprocess.run(command).returncode)\n",
        )
        self.stub(
            "python",
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['PROBE_TEST_PYTHON']).write_text(json.dumps({"
            "'argv': sys.argv[1:], 'cwd': os.getcwd(), 'pythonpath': os.environ['PYTHONPATH']}))\n"
            "raise SystemExit(int(os.environ['PROBE_TEST_EXIT']))\n",
        )

    def stub(self, name, body):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\n{body}")
        path.chmod(0o755)

    def launch(self, *args):
        return subprocess.run(["bash", str(SCRIPT), *args], env=self.env, text=True, capture_output=True)

    def test_dry_run_matches_actual_command_and_delivers_required_arguments(self):
        dry = self.launch("--dry-run")
        self.assertEqual(dry.returncode, 0, dry.stderr)
        tokens = shlex.split(dry.stdout.splitlines()[-1])
        self.assertFalse((self.root / "submit.json").exists())
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        submitted = json.loads((self.root / "submit.json").read_text())
        self.assertEqual(tokens[1:], submitted)
        self.assertNotIn("\n", submitted[-1])
        self.assertEqual(submitted[submitted.index("--") + 1 : -1], ["bash", "-c"])
        received = json.loads((self.root / "python.json").read_text())
        self.assertEqual(received["cwd"], str(self.repo.resolve()))
        self.assertEqual(received["pythonpath"], str(self.repo))
        self.assertEqual(
            received["argv"],
            [
                "-m",
                "eval.adni.encoder_modality_gap",
                "--run-dir",
                self.env["RUN_DIR"],
                "--checkpoint",
                "model_best.pt",
                "--device",
                "cuda",
                "--batch-size",
                "4",
            ],
        )

    def test_paths_and_forwarded_arguments_preserve_quotes_spaces_and_shell_literals(self):
        self.env["RUN_DIR"] = str(self.repo / "runs/a 'quoted' $value; `false` $(false)")
        self.env["CHECKPOINT"] = "model init.pt"
        extra = ["--out-dir", str(self.repo / "probe output"), "--seed", "3"]
        result = self.launch(*extra)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = json.loads((self.root / "python.json").read_text())["argv"]
        self.assertEqual(argv[argv.index("--run-dir") + 1], self.env["RUN_DIR"])
        self.assertEqual(argv[argv.index("--checkpoint") + 1], "model init.pt")
        self.assertEqual(argv[-len(extra) :], extra)

    def test_rejects_bad_job_names_and_preserves_exit_status(self):
        for bad in ("bad.name", "x" * 64):
            self.env["JOB_NAME"] = bad
            result = self.launch()
            self.assertEqual(result.returncode, 2)
            self.assertFalse((self.root / "submit.json").exists())
        self.env["JOB_NAME"] = "valid-probe"
        self.env["PROBE_TEST_EXIT"] = "23"
        self.assertEqual(self.launch().returncode, 23)


if __name__ == "__main__":
    unittest.main()
