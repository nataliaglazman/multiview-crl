"""Stage 86's continuation scoring helpers (out/burden_feasibility_20261102/continue_check.py).

Run from the repo root with PYTHONPATH=out/burden_feasibility_20261102:out/regional_pilot_20261028/shim:.
"""

import tempfile
import unittest
from pathlib import Path

import continue_check as CC


class ContinueCheckTest(unittest.TestCase):
    def test_points_names_totals_and_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run = tmp / "runs" / "x_plus40k_plus60k_a100_s4"
            run.mkdir(parents=True)
            (run / "report.json").write_text("{}")
            for step in (10000, 20000):
                (run / f"model_{step:05d}.pt").write_bytes(b"w")
            pairs = CC.points([run], at=tmp / "at", starts={4: tmp / "start4"}, start=60000)
            self.assertEqual(pairs[0], f"cont_at60000_s4={tmp / 'start4'}")
            self.assertEqual([p.split("=")[0] for p in pairs[1:]], ["cont_at70000_s4", "cont_at80000_s4"])
            folder = tmp / "at" / "x_plus40k_plus60k_a100_s4_at80000"
            self.assertTrue((folder / "model.pt").is_symlink())
            self.assertEqual((folder / "model.pt").resolve(), (run / "model_20000.pt").resolve())

    def test_rules(self):
        flat = [(60000, False, (0.01, 0.01))] + [(t, False, (0.02, 0.05)) for t in range(70000, 130000, 10000)]
        late = [(60000, False, (0.01, 0.01))] + [(t, t >= 100000, (0.3 if t < 100000 else 0.45, 0.42))
                                                 for t in range(70000, 130000, 10000)]
        rises = [(60000, False, (0.01, 0.01))] + [(t, False, (0.15, 0.2)) for t in range(70000, 130000, 10000)]
        self.assertEqual(CC.rules({4: flat})[4], "stable trade")
        self.assertEqual(CC.rules({4: late})[4], "repaired")
        self.assertEqual(CC.rules({4: rises})[4], "unresolved")
        held = [(60000, True, (0.44, 0.43))] + [(t, True, (0.5, 0.5)) for t in range(70000, 130000, 10000)]
        dips = held[:-1] + [(120000, False, (0.35, 0.36))]
        wobbly = held[:3] + [(90000, False, (0.38, 0.39))] + held[4:]
        self.assertEqual(CC.rules({42: held})[42], "stable")
        self.assertEqual(CC.rules({42: dips})[42], "lost")
        self.assertEqual(CC.rules({42: wobbly})[42], "unstable")


if __name__ == "__main__":
    unittest.main()
