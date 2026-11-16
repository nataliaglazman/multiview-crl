"""mlmv integration: a 32^3 synthetic run of 200 steps on CPU finishes in under 5 minutes and writes a leakage matrix."""

import csv
import json
import tempfile
import time
import unittest
from pathlib import Path

from mlmv.config import load_config
from mlmv.train.train_levels import run

ROOT = Path(__file__).resolve().parents[1]


class TestIntegration(unittest.TestCase):
    def test_cpu_run_writes_leakage_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(ROOT / "mlmv/config/runs/smoke_cpu.yaml", overrides={"run": {"out_dir": tmp}})
            self.assertEqual(cfg.optim.steps, 200)
            t0 = time.time()
            run_dir = run(cfg, log=lambda s: None)
            self.assertLess(time.time() - t0, 300)
            with open(run_dir / "eval" / "leakage.csv") as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(len(rows), 2 * 3 * 6 + 6)  # modalities x blocks x factors, plus the image reference
            rho = json.loads((run_dir / "geometry" / "rho.json").read_text())
            self.assertEqual(rho["theoretical"], 1.5)
            self.assertIsNotNone(rho["effective"])
            for name in (
                "config.yaml",
                "geometry/region_masks.npz",
                "geometry/eligibility.csv",
                "checkpoints/final.pt",
            ):
                self.assertTrue((run_dir / name).exists(), name)
            steps = [json.loads(line)["step"] for line in (run_dir / "scalars.jsonl").read_text().splitlines()]
            self.assertIn(199, steps)

    def test_fix_settings_run_and_keep_checkpoints(self):
        # batch standardisation, no expander, kept checkpoints and the scale logging, end to end
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(
                ROOT / "mlmv/config/runs/smoke_cpu.yaml",
                overrides={
                    "run": {"out_dir": tmp, "name": "fixes", "log_every": 10, "keep_every": 10},
                    "optim": {"steps": 30, "warmup_steps": 5},
                    "model": {"expander": []},
                    "loss": {"standardise": "batch"},
                },
            )
            run_dir = run(cfg, log=lambda s: None)
            kept = sorted(p.name for p in (run_dir / "checkpoints").glob("step_*.pt"))
            self.assertEqual(kept, ["step_000010.pt", "step_000020.pt", "step_000030.pt"])
            last = json.loads((run_dir / "scalars.jsonl").read_text().splitlines()[-1])
            for key in ("p1/S/deviation_rms_t1", "p1/G/template_rms_flair", "p1/L/min_var", "p1/G/mean_var"):
                self.assertIn(key, last)
            self.assertTrue((run_dir / "eval" / "leakage.csv").exists())


if __name__ == "__main__":
    unittest.main()
