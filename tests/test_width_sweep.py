"""Stage 74's content-size sweep on the fixed bank (out/burden_feasibility_20261102/width_sweep.py): the elbow rule,
the own-part saturation width, the pass rule and the figure, on fake runs of known width."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "out/burden_feasibility_20261102"))
import width_sweep as W


def fake_rows(true_width, widths=(4, 5, 6, 7, 8, 9, 10, 12, 16), seeds=(42, 3, 4), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seeds:
        for k in widths:
            loss = 0.4 + 2.0 * max(0, true_width - k) + 0.01 * rng.standard_normal()
            own = 0.85 * min(k, true_width) / true_width - 0.02 * rng.random(len(W.FACTORS))
            rows.append({"width": k, "seed": s, "heldout_loss": loss, "own_r2": dict(zip(W.FACTORS, own))})
    return rows


class WidthSweepTest(unittest.TestCase):
    def test_self_test(self):
        W.self_test()

    def test_elbow_finds_the_true_width(self):
        for true in (6, 7, 9):
            s = W.summarize(fake_rows(true), 0.05, 7)
            self.assertEqual(s["k_loss"], true)
            self.assertEqual(s["k_own"], true)
            self.assertEqual(s["passes"], abs(true - 8) <= 1)

    def test_missing_seed_cell_skips_per_seed_elbow(self):
        rows = [r for r in fake_rows(7) if not (r["seed"] == 3 and r["width"] == 5)]
        s = W.summarize(rows, 0.05, 7)
        self.assertNotIn(3, s["k_loss_per_seed"])
        self.assertEqual(s["k_loss_per_seed"][42], 7)
        self.assertEqual([t["seeds"] for t in s["table"] if t["width"] == 5][0], [4, 42])

    def test_score_writes_summary_and_figure(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "w.json"
            out.write_text(json.dumps({"runs": fake_rows(7)}))
            W.main(["score", str(out), "--figure", str(Path(tmp) / "w.png")])
            data = json.loads(out.read_text())
            self.assertEqual(data["summary"]["k_loss"], 7)
            self.assertTrue((Path(tmp) / "w.png").exists() and (Path(tmp) / "w.csv").exists())


if __name__ == "__main__":
    unittest.main()
