"""Stage 42's factoriser (out/burden_feasibility_20261102/factoriser.py): the label-free count of novel shared factors,
the deflation, the canonical pair, the naming, the graph truth among the anchors, PC's scoring, and the whole benchmark
on a code of the causal bank's SCM with a known answer."""

import argparse
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "out/burden_feasibility_20261102"))
import bank as B
import factoriser as F
import ica_simulation as I

SIZES = {"train": 2000, "val": 400, "test": 500}


def two_views(rng, concealed=3, units=12, view_error=0.3, weak_error=None, cube=False):
    """Two anchors and ``concealed`` hidden latents, each read by each view with its own error (the last hidden one with
    ``weak_error`` if given); each view's code mixes the readings, squares and products of them (and the first
    anchor's cube with ``cube``), and view-specific style, plus noise."""
    n = sum(SIZES.values())
    anchors = rng.standard_normal((n, 2))
    hidden = rng.uniform(-np.sqrt(3), np.sqrt(3), (n, concealed))
    errors = np.full(concealed, view_error)
    if weak_error is not None and concealed:
        errors[-1] = weak_error
    codes = []
    for _ in range(2):
        a = anchors + view_error * rng.standard_normal(anchors.shape)
        h = hidden + errors * rng.standard_normal(hidden.shape)
        harmonics = [a[:, 0] ** 2] + ([h[:, 0] ** 2, h[:, 0] * h[:, 1]] if concealed >= 2 else [])
        harmonics += [a[:, 0] ** 3] if cube else []
        features = np.column_stack([a, h, *harmonics, rng.standard_normal((n, 2))])
        features = (features - features.mean(0)) / features.std(0)
        code = features @ rng.standard_normal((features.shape[1], units))
        codes.append(code + 0.1 * code.std(0) * rng.standard_normal(code.shape))
    edges = np.cumsum([0, SIZES["train"], SIZES["val"], SIZES["test"]])
    data, slots = {}, {}
    for i, split in enumerate(F.SPLITS):
        rows = slice(edges[i], edges[i + 1])
        data[split] = {"code": np.stack([c[rows] for c in codes])}
        slots[split] = np.stack([anchors[rows] + 0.05 * rng.standard_normal(anchors[rows].shape) for _ in range(2)])
    return data, slots, hidden


def bank_like(seed, root=None, burden_error=0.25, units=24, sizes=(2000, 400, 1000)):
    """A code like a trained one on the causal bank, with a known answer: the bank's SCM (leaky ReLU, Gaussian own
    parts) and a uniform lesion centre, each latent read by each view with its own error (burden with
    ``burden_error``, the rest .1), plus the squares and product of the centre's x and y and three view-specific
    units, mixed into ``units`` units with noise. Beside the default anchors six latents are novel: burden's, lr's and
    sulcal's own parts and the centre's x, y and z. With ``root``, the SCM's graph is saved there as a bank's is."""
    rng = np.random.default_rng(seed)
    causal = B.causal_scm(B.run_args(0.4, causal=True).seed)
    total = sum(sizes)
    factors, own = I.sample(causal, total, "gaussian", "leaky_relu", rng)
    position = rng.uniform(-1, 1, (total, 3))
    latents = np.column_stack([factors, position])
    latents = (latents - latents.mean(0)) / latents.std(0)
    error = np.full(latents.shape[1], 0.1)
    error[F.FACTORS.index("burden")] = burden_error
    codes = []
    for _ in range(2):
        read = latents + error * rng.standard_normal(latents.shape)
        x, y = read[:, 7], read[:, 8]
        features = np.column_stack(
            [read, 0.5 * np.column_stack([x**2, y**2, x * y]), rng.standard_normal((total, 3))]
        )
        code = features @ rng.standard_normal((features.shape[1], units)) / np.sqrt(features.shape[1])
        codes.append(code + 0.05 * code.std(0) * rng.standard_normal(code.shape))
    if root is not None:
        np.save(f"{root}/causal_adj.npy", causal["adj"])
    cut = np.cumsum([0, *sizes])
    data = {"step": 0, "bank": root, "units": units}
    for i, split in enumerate(F.SPLITS):
        rows = slice(cut[i], cut[i + 1])
        data[split] = {"code": np.stack([c[rows] for c in codes]), "factors": factors[rows]}
    data["test"]["own"], data["test"]["position"] = own[cut[2] :], position[cut[2] :]
    return data


def scm():
    adj = np.zeros((len(F.FACTORS), len(F.FACTORS)), dtype=bool)
    for parent, child in B.CAUSAL_EDGES:
        adj[F.COLUMNS.index(parent), F.COLUMNS.index(child)] = True
    return adj


class CountTests(unittest.TestCase):
    def count(self, seed, **kwargs):
        data, slots, hidden = two_views(np.random.default_rng(seed), **kwargs)
        block, records = F.novel_block(data, slots, 8, 0.5, np.random.default_rng(seed + 100))
        return block, records, hidden

    def test_counts_the_hidden_latents_not_their_squares(self):
        for seed in range(3):
            block, records, hidden = self.count(seed)
            self.assertEqual(block["train"].shape[2], 3, records)
        test = block["test"][0]
        design = np.column_stack([test, np.ones(len(test))])
        truth = hidden[-SIZES["test"] :]
        fitted = design @ np.linalg.lstsq(design, truth, rcond=None)[0]
        r2 = 1 - ((truth - fitted) ** 2).sum(0) / ((truth - truth.mean(0)) ** 2).sum(0)
        # each view reads a hidden latent with error .3, so no direction of one view's code holds more than 1 / 1.09
        self.assertTrue(np.all(r2 > 0.7), r2)

    def test_no_shared_hidden_content_gives_no_components(self):
        block, records, _ = self.count(2, concealed=0)
        self.assertEqual(block["train"].shape[2], 0, records)

    def test_a_weaker_factor_still_counts(self):
        # The third hidden latent is read with error .8 (agreement about .6), below the others' .9.
        block, records, _ = self.count(7, weak_error=0.8)
        self.assertEqual(block["train"].shape[2], 3, records)

    def test_an_anchor_cube_is_flagged_not_counted(self):
        block, records, _ = self.count(9, cube=True)
        self.assertEqual(block["train"].shape[2], 3, records)
        self.assertTrue(any(r["shared"] and not r["kept"] for r in records), records)

    def test_a_taken_direction_leaves_no_trace(self):
        # CCA ignores scale, so any remnant of a subtracted direction (a ridge fit leaves one at every penalty) is the
        # only thing left along it and comes back exactly as shared as the direction itself.
        data, slots, _ = two_views(np.random.default_rng(0))
        left = F.residuals(data, slots)
        wa, wb, _ = F.canonical_pairs(left["train"][0], left["train"][1])
        taken = {s: np.stack([left[s][0] @ wa[:, :1], left[s][1] @ wb[:, :1]]) for s in F.SPLITS}
        again = F.residuals(data, {s: np.concatenate([slots[s], taken[s]], 2) for s in F.SPLITS})
        for split in F.SPLITS:
            self.assertLess(np.abs(again[split][0] @ wa[:, 0]).max(), 1e-8, split)
            self.assertLess(np.abs(again[split][1] @ wb[:, 0]).max(), 1e-8, split)

    def test_squares_are_not_extrapolated(self):
        reference = np.random.default_rng(1).standard_normal((500, 2))
        features = F.quadratic(np.array([[50.0, 0.0]]), reference)
        mean, sd = reference.mean(0), reference.std(0)
        self.assertAlmostEqual(features[0, 0], (50 - mean[0]) / sd[0])  # linear terms stay exact
        self.assertAlmostEqual(features[0, 2], ((reference[:, 0].max() - mean[0]) / sd[0]) ** 2)

    def test_canonical_pairs_find_the_shared_direction(self):
        rng = np.random.default_rng(4)
        shared = rng.standard_normal(3000)
        a = np.column_stack([shared + 0.2 * rng.standard_normal(3000), rng.standard_normal((3000, 3))])
        b = np.column_stack([rng.standard_normal((3000, 2)), shared + 0.2 * rng.standard_normal(3000)])
        u, w, rho = F.canonical_pairs(a, b)
        self.assertEqual(u.shape, (4, 3))
        self.assertGreater(rho[0], 0.9)
        self.assertLess(rho[1], 0.1)
        self.assertGreater(np.corrcoef(a @ u[:, 0], b @ w[:, 0])[0, 1], 0.9)
        self.assertGreater(abs(np.corrcoef(a @ u[:, 0], shared)[0, 1]), 0.95)


class TruthTests(unittest.TestCase):
    def test_default_anchors(self):
        truth, induced, hidden = F.anchor_truth(F.ANCHORS, scm())
        self.assertEqual(
            F.edge_names(truth),
            ["brain - thickness", "brain - ventricle", "temporal - thickness", "temporal - ventricle"],
        )
        self.assertEqual(F.edge_names(induced), ["temporal - ventricle"])  # through the concealed burden
        self.assertEqual(hidden, ["burden -> temporal", "temporal -> sulcal", "ventricle -> burden"])

    def test_burden_as_anchor(self):
        truth, induced, _ = F.anchor_truth(("brain", "ventricle", "burden", "temporal"), scm())
        self.assertEqual(
            F.edge_names(truth), ["brain - temporal", "brain - ventricle", "burden - temporal", "burden - ventricle"]
        )
        self.assertEqual(F.edge_names(induced), ["brain - temporal"])  # through the concealed thickness

    def test_d_separation(self):
        adj = np.zeros((5, 5), dtype=bool)
        for parent, child in ((0, 2), (1, 2), (2, 3), (4, 0), (4, 1)):
            adj[parent, child] = True
        self.assertFalse(F.d_separated(adj, 0, 1, []))  # the common cause 4
        self.assertTrue(F.d_separated(adj, 0, 1, [4]))
        self.assertFalse(F.d_separated(adj, 0, 1, [4, 3]))  # conditioning on the collider's child opens it
        self.assertTrue(F.d_separated(adj, 0, 3, [2]))

    def test_whole_skeleton_is_the_scm(self):
        whole = F.oracle_skeleton(scm(), list(range(len(F.FACTORS))))
        self.assertEqual(whole, {frozenset((B.CAUSAL_FACTORS[p], B.CAUSAL_FACTORS[c])) for p, c in B.CAUSAL_EDGES})


class ScoringTests(unittest.TestCase):
    def test_naming_undoes_a_permutation(self):
        rng = np.random.default_rng(5)
        targets = rng.standard_normal((400, 4))
        components = targets[:, [2, 0, 3]] * [2.0, -1.0, 0.5] + 0.1 * rng.standard_normal((400, 3))
        names, strength = F.name_components(components, targets)
        self.assertEqual(names.tolist(), [2, 0, 3])
        self.assertTrue(np.all(strength > 0.9))

    def test_pc_on_true_variables_scores_high(self):
        rng = np.random.default_rng(6)
        n = 3000
        brain = rng.standard_normal(n)
        thickness = 0.7 * brain + 0.7 * rng.standard_normal(n)
        temporal = 0.7 * thickness + 0.7 * rng.standard_normal(n)
        unrelated = rng.standard_normal((n, 2))
        truth = {frozenset(("brain", "thickness")), frozenset(("temporal", "thickness"))}
        g = F.graph_scores(
            np.column_stack([brain, thickness, temporal, unrelated]),
            ["brain", "thickness", "temporal", "c1", "c2"],
            truth,
            {"c1", "c2"},
            rows=1000,
            resamples=5,
            rng=rng,
        )
        self.assertGreater(g["f1_mean"], 0.9)
        self.assertEqual(g["edges_found"].get("brain - thickness"), 1.0)
        self.assertLess(g["block_edges_mean"], 0.5)


class BankLikeTests(unittest.TestCase):
    def test_counts_the_six_novel_latents_and_keeps_burden(self):
        # Seed 1 is where subtracting with a ridge fit lost burden's own part: a remnant of a taken direction came back,
        # mixed with burden, and was dropped as redundant.
        data = bank_like(1)
        slots, _ = F.anchored_slots(data, F.ANCHORS, 0.0, np.random.default_rng(0))
        block, records = F.novel_block(data, slots, 12, 0.5, np.random.default_rng(1))
        self.assertEqual(block["test"].shape[2], 6, records)
        targets, names = F.concealed_targets(data["test"], F.ANCHORS)
        for v in range(2):
            held = dict(zip(names, (F.r2(F.cv_predict(block["test"][v], y), y) for y in targets.T)))
            self.assertGreater(held.pop("burden own"), 0.5)
            self.assertGreater(min(held.values()), 0.8, held)

    def test_measured_anchors_count_the_same_six(self):
        data = bank_like(1)
        anchors = F.measured_anchors(data, F.ANCHORS, 0.0, np.random.default_rng(0))
        np.testing.assert_array_equal(anchors["test"][0], anchors["test"][1])
        block, records = F.novel_block(data, anchors, 12, 0.5, np.random.default_rng(1))
        self.assertEqual(block["test"].shape[2], 6, records)
        targets, names = F.concealed_targets(data["test"], F.ANCHORS)
        held = dict(zip(names, (F.r2(F.cv_predict(block["test"][0], y), y) for y in targets.T)))
        self.assertGreater(min(held.values()), 0.5, held)

    def test_the_trace_follows_every_examined_direction(self):
        data = bank_like(1)
        slots, _ = F.anchored_slots(data, F.ANCHORS, 0.0, np.random.default_rng(0))
        trace = []
        _, records = F.novel_block(data, slots, 12, 0.5, np.random.default_rng(1), 0.2, trace)
        self.assertEqual(len(trace), sum(1 for r in records if not r.get("empty")))
        self.assertFalse(records[-1]["shared"])  # the last direction examined is the one the count stopped at
        targets, names = F.concealed_targets(data["test"], F.ANCHORS)
        best = {
            names[int(np.argmax([abs(np.corrcoef(d[0], t)[0, 1]) for t in targets.T]))]
            for r, d in zip(records, trace)
            if r["kept"]
        }
        self.assertTrue({"lesion x", "lesion y", "lesion z", "lr own"} <= best, best)

    @unittest.skipUnless(importlib.util.find_spec("causallearn"), "needs causal-learn")
    def test_the_benchmark_runs_end_to_end(self):
        for source in F.SOURCES:
            a = argparse.Namespace(
                anchors=list(F.ANCHORS),
                anchor_source=source,
                anchor_noise=0.0,
                max_factors=12,
                min_agreement=0.5,
                max_redundancy=0.2,
                rows=300,
                resamples=2,
                train_subjects=None,
            )
            with tempfile.TemporaryDirectory() as root:
                data = bank_like(3, root)
                row = {"settings": F.settings(a), "step": 0, "bank": root, "units": 24}
                row["trained"] = row["floor"] = F.assess_codes(data, a)
                row["ceilings"] = F.ceilings(data, a)
            report = io.StringIO()
            with contextlib.redirect_stdout(report):
                F.show("bank-like", row)
            r = row["trained"]
            self.assertEqual(r["count"], 6, source)
            self.assertGreater(np.min(r["anchor_r2"]), 0.9)
            self.assertIn("novel block: 6 components", report.getvalue())
            self.assertIn(f"as {F.SOURCES[source]}", report.getvalue())
            self.assertIn("step 2 trace", report.getvalue())
            self.assertEqual(r["induced"], ["temporal - ventricle"])
            for view in r["views"]:
                # the lesion centre is uniform, so ICA can name its coordinates; the own parts are Gaussian and stay
                # mixed
                named = dict(zip(view["names"]["ica"]["names"], view["names"]["ica"]["matched_r"]))
                self.assertGreater(min(named.get(f"lesion {c}", 0) for c in "xyz"), 0.9, named)
                self.assertLess(view["graph"]["block_edges_mean"], 0.05 * (6 * 4 + 15) + 1)
                self.assertEqual(len(view["dropped_r2"]), len(r["target_names"]))


if __name__ == "__main__":
    unittest.main()
