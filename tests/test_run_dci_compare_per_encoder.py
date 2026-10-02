"""run_dci_compare --per-encoder: encoder 2's probes read a reduction fitted on encoder 2.

``score_reprs`` used to fit the probe PCA on view 1, reuse it for view 2, and score encoder
2 on those projections. That basis is right for the view-invariance probe and wrong for
encoder 2's own scores, because two encoders do not share a basis. On ident-vent-hsic it
depressed encoder 2's untrained patch floor by 0.13 and inflated its learned patch content
from +0.02 to +0.17. The planted data below puts encoder 2's signal in the columns encoder
1 leaves dead, so a shared basis projects it to a constant and the bug shows up as R^2 ~ 0
instead of a plausible number.
"""

import json
import unittest

import numpy as np

from eval.metrics.identifiability_metrics import view_invariance
from eval.protocol.run_dci_compare import (
    _CONTENT,
    _CONTENT_V2,
    _STYLE,
    _STYLE_V2,
    PROBE_DIM_AUTO,
    _reduce_reprs,
    _swap_encoders,
    score_reprs,
)

N, HALF = 240, 64  # 2*HALF = 128 > N/4 = 60, so `auto` reduces the stats and patch blocks to 60
NAMES, STYLE_NAMES = ["brain_size", "lesion_x"], ["bias", "noise_sigma"]
INFO = {
    "content_names": NAMES,
    "style_names": STYLE_NAMES,
    "n_content_channels": 8,
    "n_style_channels": 4,
    "has_split": True,
}


def _planted():
    rng = np.random.RandomState(1)
    gt, gs1, gs2 = rng.randn(N, 2), rng.randn(N, 2), rng.randn(N, 2)

    def lin(z, width):
        return z @ rng.randn(z.shape[1], width) + 0.05 * rng.randn(N, width)

    def wide(z, encoder):
        # Encoder 1 writes the left half and encoder 2 the right; the other half is dead.
        cols = [lin(z, HALF), np.zeros((N, HALF))]
        return np.hstack(cols if encoder == 1 else cols[::-1])

    def wide_level():
        return {0: (wide(gt, 1), wide(gs1, 1), wide(gt, 2), wide(gs2, 2), INFO)}

    reprs = {
        "gap": {0: (lin(gt, 8), lin(gs1, 4), lin(gt, 8), lin(gs2, 4), INFO)},  # `auto` leaves it alone
        "stats": wide_level(),
        "patch": wide_level(),
    }
    return reprs, gt, gs1, gs2


def _score(reprs, gt, gs1, gs2, **kw):
    kw = {"n_null": 2, "seeds": (0, 1), "probe_dim": PROBE_DIM_AUTO, "gt_style_v2": gs2, **kw}
    return score_reprs(reprs, gt, gs1, INFO, 0, **kw)


def _canon(row):
    return json.dumps(row, sort_keys=True, default=repr)


class PerEncoderBasisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reprs, cls.gt, cls.gs1, cls.gs2 = _planted()
        cls.both = _score(cls.reprs, cls.gt, cls.gs1, cls.gs2, per_encoder=True)
        cls.only1 = _score(cls.reprs, cls.gt, cls.gs1, cls.gs2, per_encoder=False)

    def test_planted_data_exercises_the_shared_basis_trap(self):
        shared = _reduce_reprs(self.reprs, 0, PROBE_DIM_AUTO)
        own = _reduce_reprs(_swap_encoders(self.reprs, 0), 0, PROBE_DIM_AUTO)
        for key in ("stats", "patch"):
            for shared_idx, own_idx in ((_CONTENT_V2, _CONTENT), (_STYLE_V2, _STYLE)):
                trapped, recovered = shared[key][0][shared_idx], own[key][0][own_idx]
                self.assertEqual(trapped.shape, (N, N // 4))
                self.assertLess(np.ptp(trapped, axis=0).max(), 1e-6, f"{key}: view 1's basis must erase encoder 2")
                self.assertGreater(np.ptp(recovered, axis=0).max(), 1.0, f"{key}: encoder 2's own basis keeps it")

    def test_encoder2_reduced_blocks_use_its_own_basis(self):
        detail = self.both["detail_v2"]
        for cell, names in (("content2content", NAMES), ("style2style", STYLE_NAMES)):
            for name in names:
                gap = detail[cell]["per_factor"][name]["by_pooling"]["patch"]["gap"]
                self.assertGreater(gap, 0.9, f"encoder 2 {cell} {name}@patch R2 gap {gap:+.3f}")
        # Block-MCC reads stats, which `auto` reduces here too.
        self.assertGreater(self.both["mcc_cc_v2"], 0.9)
        self.assertGreater(self.both["mcc_cc_v2"] - self.both["mcc_cc_null_v2"], 0.5)

    def test_encoder1_row_is_unchanged_by_per_encoder(self):
        self.assertFalse(any(k.endswith("_v2") for k in self.only1))
        enc1 = {k: v for k, v in self.both.items() if not k.endswith("_v2")}
        self.assertEqual(_canon(enc1), _canon(self.only1))

    def test_view_probe_keeps_view1_basis(self):
        stats = _reduce_reprs(self.reprs, 0, PROBE_DIM_AUTO)["stats"][0]
        c1, s1, c2, s2 = (stats[i] for i in (_CONTENT, _STYLE, _CONTENT_V2, _STYLE_V2))
        expected = view_invariance(c1, c2, s1, s2, seeds=(0, 1))
        self.assertEqual(self.both["content_view"], expected["content_acc"])
        self.assertEqual(self.both["style_view"], expected["style_acc"])
        # Guard that the assertion above can fail: per-encoder bases give a different answer.
        own2 = _reduce_reprs(_swap_encoders(self.reprs, 0), 0, PROBE_DIM_AUTO)["stats"][0]
        separate = view_invariance(c1, own2[_CONTENT], s1, own2[_STYLE], seeds=(0, 1))
        self.assertNotEqual(separate["content_acc"], expected["content_acc"])


if __name__ == "__main__":
    unittest.main()
