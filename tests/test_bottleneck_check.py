"""bottleneck_check.py: a narrow code drops a weak shared latent through a linear map and keeps it through the MLP map,
the account stage 36 tests; and stage 39's prediction targets, the reuse of finished bottlenecks and the table."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import bottleneck_check as C
import frozen_prediction as FP


def smooth_r2(x, y):
    """5-fold R2 of a cubic ridge readout: reads a latent a 2-unit code holds in any smooth form."""
    model = make_pipeline(StandardScaler(), PolynomialFeatures(3), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    pred = cross_val_predict(model, x, y, cv=5)
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


class BottleneckCheckTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_a_linear_map_drops_the_weak_latent_and_the_mlp_map_keeps_it(self):
        # Two shared latents. The other view's features spread a over two directions (a and a squared) and show b once,
        # with noise. Two units through a linear map do best holding a and a squared; through the MLP map, a and b.
        rng = np.random.default_rng(0)
        n = 1200
        a, b = rng.uniform(-2, 2, n), rng.standard_normal(n)
        x = np.stack([np.column_stack([a, b, rng.standard_normal((n, 4))]) for _ in range(2)])
        t = np.stack(
            [np.column_stack([a, a**2 - (a**2).mean(), 0.5 * b + 0.5 * rng.standard_normal(n)]) for _ in range(2)]
        )
        x, t = torch.tensor(x, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)
        t = FP.whiten(t, FP.fit_whitening(t[:, :800], C.RIDGE))
        held = {}
        b_read = {}
        for hidden in (0, 64):
            model, held[hidden] = C.fit(
                x[:, :800], t[:, :800], x[:, 800:1000], t[:, 800:1000], 2, hidden, seed=0, epochs=200, patience=40
            )
            with torch.no_grad():
                code = model.head(x[:, 1000:]).numpy()
            b_read[hidden] = smooth_r2(code[0].astype(np.float64), b[1000:])
        # Over 3 data and 2 fit seeds: b read .10-.55 through the linear map, .92-.96 through the MLP; held-out loss
        # .30-.42 against .16-.17 (the optima are 1/3 and 1/6).
        self.assertGreater(b_read[64], 0.85)
        self.assertGreater(b_read[64] - b_read[0], 0.35)
        self.assertLess(held[64], held[0] - 0.1)

    def test_fit_keeps_its_best_held_out_state(self):
        x = torch.randn(2, 64, 5)
        model, held = C.fit(x, x[..., :3], x, x[..., :3], 2, 0, seed=1, epochs=3, patience=1)
        with torch.no_grad():
            self.assertAlmostEqual(float(model.loss(x, x[..., :3])), held, places=5)


class TargetTests(unittest.TestCase):
    """Stage 39's --targets: the tail features and the whole pooled vector as what the bottlenecks predict."""

    def test_targets_slice_the_pooled_vector(self):
        split = {"gap": np.zeros((2, 3, 4)), "pooled": np.arange(60, dtype=float).reshape(2, 3, 10)}
        self.assertIs(C.target_of(split, "gap", 4), split["gap"])
        np.testing.assert_array_equal(C.target_of(split, "tails", 4), split["pooled"][..., 4:])
        self.assertIs(C.target_of(split, "pooled", 4), split["pooled"])

    def test_gap_keys_keep_stage_36_names(self):
        self.assertEqual(C.key_of("MLP", 10, "gap"), "MLP 10")
        self.assertEqual(C.key_of("linear", 24, "tails"), "linear 24 tails")
        for target in C.TARGETS:
            self.assertEqual(C.parse_key(C.key_of("MLP", 10, target)), ("MLP", 10, target))

    def test_spread_counts_the_directions_a_target_uses(self):
        rng = np.random.default_rng(0)
        independent = rng.standard_normal((2, 4000, 6))
        shared = np.repeat(rng.standard_normal((2, 4000, 1)), 6, axis=2) + 1e-3 * rng.standard_normal((2, 4000, 6))
        self.assertAlmostEqual(C.spread(independent), 6, delta=0.2)
        self.assertAlmostEqual(C.spread(shared), 1, delta=0.05)
        constant = np.concatenate([independent, np.zeros((2, 4000, 1))], 2)
        self.assertAlmostEqual(C.spread(constant), C.spread(independent), places=6)


def fake_features(step=7, tails=True):
    rng = np.random.default_rng(0)

    def split(n):
        return {
            "pooled": rng.standard_normal((2, n, 12)).astype(np.float32),
            "gap": rng.standard_normal((2, n, 4)).astype(np.float32),
            "code": rng.standard_normal((2, n, 3)).astype(np.float32),
            "vox": rng.uniform(10, 90, n),
            "zc": rng.standard_normal((n, 9)),
        }

    return {
        "step": step,
        "bank": "bank",
        "units": 3,
        "tail_start": 4,
        "tails": tails,
        **{name: split(n) for name, n in (("train", 40), ("val", 10), ("test", 20))},
    }


class ReuseTests(unittest.TestCase):
    """A rerun fits only the bottlenecks a run's row lacks, and saves after each new one."""

    def assess(self, targets, done, features=None):
        fitted = []

        def fit(x, t, held_x, held_t, units, hidden, seed):
            fitted.append((t.shape[-1], units, hidden, seed))
            return SimpleNamespace(head=lambda z: torch.zeros(*z.shape[:2], units)), 0.5

        labels = (np.full(20, 50.0), 0.0, np.zeros((20, 3)))
        saved = []
        with mock.patch.object(C, "run_features", return_value=features or fake_features()), mock.patch.object(
            C, "labels", return_value=labels
        ), mock.patch.object(C, "fit", side_effect=fit), mock.patch.object(C, "read", return_value={"t1": {}}):
            with contextlib.redirect_stdout(io.StringIO()):
                out = C.assess(
                    Path("runs/run"), "cpu", targets, done, save=lambda row: saved.append(len(row["bottlenecks"]))
                )
        return out, fitted, saved

    def test_a_row_at_the_same_step_keeps_its_bottlenecks(self):
        kept = {C.key_of(label, units, "gap"): ["kept"] for units in C.WIDTHS for label in C.MAPS}
        done = {"step": 7, "train_rows": 40, "own_code": "kept", "bottlenecks": dict(kept)}
        out, fitted, saved = self.assess(["gap", "tails"], done)
        self.assertEqual(out["own_code"], "kept")
        for key in kept:
            self.assertEqual(out["bottlenecks"][key], ["kept"])
        self.assertEqual({width for width, *_ in fitted}, {8})  # only the tails: 12 pooled features after 4 averages
        self.assertEqual(len(fitted), len(C.WIDTHS) * len(C.MAPS) * len(C.SEEDS))
        self.assertEqual(saved, [5, 6, 7, 8])
        self.assertEqual(out["targets"]["tails"]["width"], 8)
        self.assertEqual(out["targets"]["gap"]["width"], 4)

    def test_a_row_from_another_step_is_refitted(self):
        done = {"step": 3, "train_rows": 40, "own_code": "old", "bottlenecks": {"MLP 10": ["old"]}}
        out, fitted, _ = self.assess(["gap", "pooled"], done)
        self.assertEqual(out["bottlenecks"]["MLP 10"], [{"held_out": 0.5, "t1": {}}] * len(C.SEEDS))
        self.assertEqual(sorted({width for width, *_ in fitted}), [4, 12])
        self.assertEqual(len(fitted), 2 * len(C.WIDTHS) * len(C.MAPS) * len(C.SEEDS))

    def test_no_tails_without_tail_pooling(self):
        with self.assertRaises(SystemExit):
            self.assess(["tails"], None, fake_features(tails=False))


def cell(each):
    out = {"error": 9.0, "alone_r2": 0.5, "error_by_size": [8.0, 9.0, 10.0]}
    for name, count in (("anatomy", 6), ("position", 3)):
        for kind in ("ridge", "boosted"):
            out[f"{name}_{kind}"] = 0.5
            if each:
                out[f"{name}_{kind}_each"] = [0.5] * count
    return out


class ShowTests(unittest.TestCase):
    def shown(self, row):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            C.show("run", row)
        return out.getvalue()

    def test_stage_36_rows_still_print(self):
        fits = [{"held_out": 0.2, "t1": cell(False), "flair": cell(False)}]
        row = {"step": 1, "bank": "b", "rows": 20, "train_rows": 40, "own_sd": 20.0}
        row.update(own_code={"units": 10, "t1": cell(False), "flair": cell(False)}, bottlenecks={"MLP 10": fits})
        text = self.shown(row)
        self.assertIn("MLP map", text)
        self.assertNotIn("each factor", text)

    def test_new_rows_print_their_target_and_each_factor(self):
        fits = [{"held_out": 0.2, "t1": cell(True), "flair": cell(True)}]
        row = {"step": 1, "bank": "b", "rows": 20, "train_rows": 40, "own_sd": 20.0}
        row.update(
            own_code={"units": 10, "t1": cell(True), "flair": cell(True)},
            bottlenecks={"MLP 10": fits, "MLP 10 tails": fits},
            targets={"gap": {"width": 64, "spread": 2.5}, "tails": {"width": 384, "spread": 9.0}},
        )
        text = self.shown(row)
        self.assertIn("tails 384 features", text)
        self.assertIn("each factor", text)
        lines = [line for line in text.splitlines() if "MLP map" in line]
        self.assertEqual([line.split()[2] for line in lines], ["gap", "tails", "gap", "tails"])


if __name__ == "__main__":
    unittest.main()
