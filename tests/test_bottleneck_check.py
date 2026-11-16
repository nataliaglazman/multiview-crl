"""bottleneck_check.py's fit: a narrow code drops a weak shared latent through a linear map and keeps it through the
MLP map, the account stage 36 tests."""

import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
