"""The shared bundle path: row identity, VQ block export, and one-protocol comparison."""

import argparse
import csv
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from eval import bundle_identity as identity
from eval import compare_bundles as compare
from eval import dinov3_embed_synthetic as embed
from eval import dinov3_identifiability as scorer
from eval import export_vq_bundle as export


def planted(n=200, k=3, noise=0.1, seed=0, width=24):
    """A chain SCM, its factors, and features that mix them at the given noise level."""
    rng = np.random.RandomState(seed)
    z = rng.randn(n, k)
    z[:, 1] += 1.3 * z[:, 0]
    z[:, 2] += 1.3 * z[:, 1]
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
    features = z @ rng.randn(k, width) + noise * rng.randn(n, width)
    return z, adjacency, features


def write_bundle(path, features, z, adjacency, settings=None, n_style=2, seed=0):
    """Write a bundle exactly as the production writers do, identity fields included."""
    rng = np.random.RandomState(seed)
    latents = {
        "z_content": z.astype(np.float32),
        "z_style_v1": rng.randn(len(z), n_style).astype(np.float32),
        "z_style_v2": rng.randn(len(z), n_style).astype(np.float32),
        "causal_adj": np.repeat(adjacency.astype(np.float32)[None], len(z), axis=0),
    }
    settings = settings or {"synthetic_seed": 42, "synthetic_res": 16}
    meta = {
        "content_factor_names": [f"c{d}" for d in range(z.shape[1])],
        "style_factor_names": [f"s{d}" for d in range(n_style)],
        "generator": settings,
        **identity.identity_record(latents, settings, len(z)),
    }
    embed.save(Path(path), {1: features.astype(np.float32)}, latents, {}, [], meta)
    return meta


class RowIdentityTests(unittest.TestCase):
    def test_digest_survives_the_write_load_round_trip(self):
        # The writer stores one adjacency where the collate produced N, and float32 where
        # the caller may have had float64; a digest that did not survive both would make
        # every alignment check fail open.
        z, adjacency, features = planted()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b.npz"
            meta = write_bundle(path, features, z, adjacency)
            bundle = scorer.load_bundle(path)
            self.assertEqual(bundle["identity"]["factor_digest"], meta["factor_digest"])
            self.assertEqual(bundle["meta"]["n_rows"], len(z))

    def test_an_unstacked_adjacency_survives_the_writer(self):
        # The DINO extractor hands save() one adjacency per sample; export_vq_bundle reads
        # the single (K, K) matrix off the dataset's SCM. Reducing the second the way the
        # first needs would store its first ROW and quietly break every graph panel.
        z, adjacency, features = planted()
        latents = {"z_content": z.astype(np.float32), "causal_adj": adjacency.astype(np.float32)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flat.npz"
            embed.save(
                path, {1: features.astype(np.float32)}, latents, {}, [], {"content_factor_names": ["a", "b", "c"]}
            )
            loaded = scorer.load_bundle(path)
            self.assertEqual(loaded["adjacency"].shape, adjacency.shape)
            np.testing.assert_array_equal(loaded["adjacency"], adjacency)

    def test_stacked_and_flat_adjacencies_land_on_the_same_bundle(self):
        z, adjacency, features = planted()
        with tempfile.TemporaryDirectory() as tmp:
            stacked, flat = Path(tmp) / "s.npz", Path(tmp) / "f.npz"
            embed.save(
                stacked,
                {1: features.astype(np.float32)},
                {
                    "z_content": z.astype(np.float32),
                    "causal_adj": np.repeat(adjacency.astype(np.float32)[None], len(z), axis=0),
                },
                {},
                [],
                {},
            )
            embed.save(
                flat,
                {1: features.astype(np.float32)},
                {"z_content": z.astype(np.float32), "causal_adj": adjacency.astype(np.float32)},
                {},
                [],
                {},
            )
            a, b = scorer.load_bundle(stacked), scorer.load_bundle(flat)
            np.testing.assert_array_equal(a["adjacency"], b["adjacency"])
            self.assertEqual(a["identity"]["factor_digest"], b["identity"]["factor_digest"])

    def test_same_factors_with_different_features_stay_comparable(self):
        z, adjacency, strong = planted(noise=0.1)
        _z, _adj, weak = planted(noise=3.0, seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.npz", Path(tmp) / "b.npz"
            write_bundle(a, strong, z, adjacency)
            write_bundle(b, weak, z, adjacency)
            bundles = {"a": scorer.load_bundle(a), "b": scorer.load_bundle(b)}
            _records, problems = compare.check_alignment(bundles, strict=True)
            self.assertEqual(problems, [])

    def test_a_different_factor_draw_is_refused(self):
        z, adjacency, features = planted()
        other, _adj, _f = planted(seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.npz", Path(tmp) / "b.npz"
            write_bundle(a, features, z, adjacency)
            write_bundle(b, features, other, adjacency)
            bundles = {"a": scorer.load_bundle(a), "b": scorer.load_bundle(b)}
            with self.assertRaises(SystemExit) as raised:
                compare.check_alignment(bundles, strict=True)
            self.assertIn("not row-aligned", str(raised.exception))
            # Same generator settings, different draw: the message must say which.
            self.assertIn("different draw", str(raised.exception))

    def test_different_row_counts_report_as_a_length_problem(self):
        z, adjacency, features = planted(n=200)
        short_z, short_adj, short_f = planted(n=120, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.npz", Path(tmp) / "b.npz"
            write_bundle(a, features, z, adjacency)
            write_bundle(b, short_f, short_z, short_adj)
            records = {
                "a": identity.read_identity(scorer.load_bundle(a)["meta"]),
                "b": identity.read_identity(scorer.load_bundle(b)["meta"]),
            }
            problems = identity.compare(records)
            self.assertEqual(len(problems), 1)
            self.assertIn("different row counts", problems[0])

    def test_a_bundle_without_identity_fields_still_compares(self):
        # Bundles written before the identity fields existed must not become unusable:
        # the digest is a pure function of the arrays the file already carries.
        z, adjacency, features = planted()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.npz"
            write_bundle(path, features, z, adjacency)
            data = dict(np.load(path, allow_pickle=False))
            meta = json.loads(str(data["meta"]))
            for key in identity.IDENTITY_KEYS:
                meta.pop(key)
            data["meta"] = json.dumps(meta)
            np.savez_compressed(path, **data)
            bundle = scorer.load_bundle(path)
            self.assertEqual(compare.check_alignment({"x": bundle, "y": bundle}, strict=True)[1], [])


class VQBlockSelectionTests(unittest.TestCase):
    def level_tuple(self, n=40, content=6, style=4):
        rng = np.random.RandomState(0)
        arrays = [rng.randn(n, content), rng.randn(n, style), rng.randn(n, content), rng.randn(n, style)]
        return (*arrays, {"has_split": True, "n_content_channels": content, "n_style_channels": style})

    def test_each_block_selects_its_own_columns_for_each_view(self):
        level = self.level_tuple()
        for view, (c_idx, s_idx) in ((1, (0, 1)), (2, (2, 3))):
            with self.subTest(view=view):
                np.testing.assert_array_equal(export.select_block(level, "content", view), level[c_idx])
                np.testing.assert_array_equal(export.select_block(level, "style", view), level[s_idx])
                np.testing.assert_array_equal(
                    export.select_block(level, "all", view), np.concatenate([level[c_idx], level[s_idx]], axis=1)
                )

    def test_all_is_the_full_width_and_content_alone_is_not(self):
        level = self.level_tuple(content=6, style=4)
        self.assertEqual(export.select_block(level, "all", 1).shape[1], 10)
        self.assertEqual(export.select_block(level, "content", 1).shape[1], 6)

    def test_all_survives_a_level_with_no_style_split(self):
        rng = np.random.RandomState(0)
        level = (rng.randn(30, 5), None, rng.randn(30, 5), None, {"has_split": False})
        np.testing.assert_array_equal(export.select_block(level, "all", 1), level[0])
        self.assertIsNone(export.select_block(level, "style", 1))

    def test_pooling_strings_parse_to_what_the_extractor_takes(self):
        self.assertEqual(export.parse_pooling("gap"), ("gap", "gap"))
        self.assertEqual(export.parse_pooling("stats"), ("stats", "stats"))
        self.assertEqual(export.parse_pooling("4,4,4"), ((4, 4, 4), "4x4x4"))
        self.assertEqual(export.parse_pooling("2x2x2"), ((2, 2, 2), "2x2x2"))
        for bad in ("4,4", "0,4,4", "gap,4,4", "-1,2,3"):
            with self.subTest(bad=bad):
                with self.assertRaises(argparse.ArgumentTypeError):
                    export.parse_pooling(bad)


class ContentMaskStabilityTests(unittest.TestCase):
    """The Gumbel mask must name the same channels for every row of one feature array."""

    def helper(self):
        from eval.dci import _stabilise_content_indices

        return _stabilise_content_indices

    def test_a_stable_mask_is_passed_through_unchanged(self):
        stabilise = self.helper()
        seen = {}
        for _batch in range(3):
            self.assertEqual(stabilise(0, [0, 2, 5], seen, freeze=True), [0, 2, 5])
        self.assertNotIn("_warned", seen)

    def test_freezing_pins_the_first_batch_for_the_rest_of_the_pass(self):
        stabilise = self.helper()
        seen = {}
        self.assertEqual(stabilise(0, [0, 1, 2], seen, freeze=True), [0, 1, 2])
        with self.assertLogs("eval.dci", level="WARNING") as logs:
            self.assertEqual(stabilise(0, [3, 4, 5], seen, freeze=True), [0, 1, 2])
        self.assertIn("changed between batches", logs.output[0])

    def test_without_freezing_the_drift_is_reported_but_not_corrected(self):
        # The old behaviour is still reachable, because the reports already published were
        # produced with it; what it must not do any more is stay silent.
        stabilise = self.helper()
        seen = {}
        stabilise(0, [0, 1, 2], seen, freeze=False)
        with self.assertLogs("eval.dci", level="WARNING"):
            self.assertEqual(stabilise(0, [3, 4, 5], seen, freeze=False), [3, 4, 5])

    def test_each_level_warns_once_and_levels_are_tracked_separately(self):
        stabilise = self.helper()
        seen = {}
        stabilise(0, [0, 1], seen, freeze=True)
        stabilise(1, [7, 8], seen, freeze=True)
        with self.assertLogs("eval.dci", level="WARNING") as logs:
            stabilise(0, [2, 3], seen, freeze=True)
            stabilise(0, [4, 5], seen, freeze=True)
            stabilise(1, [9, 10], seen, freeze=True)
        self.assertEqual(len(logs.output), 2)
        self.assertEqual(stabilise(1, [9, 10], seen, freeze=True), [7, 8])

    def test_a_level_with_no_mask_stays_unsplit(self):
        stabilise = self.helper()
        seen = {}
        self.assertIsNone(stabilise(0, None, seen, freeze=True))
        self.assertEqual(seen, {})


class MatchedComparisonTests(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.options = argparse.Namespace(
            probe_kind="ridge", seeds=(0, 1), n_splits=3, n_null=2, null_seed=0, probe_dim=0, with_graph=False
        )

    def bundles(self):
        z, adjacency, strong = planted(noise=0.1)
        _z, _a, weak = planted(noise=3.0, seed=1)
        return z, adjacency, strong, weak

    def test_the_two_bundles_get_identical_folds_and_nulls(self):
        # The equalisation claim in one assertion: scoring a bundle twice, once alone and
        # once beside another, must not move a single number.
        z, adjacency, strong, weak = self.bundles()

        def make(X):
            return dict(
                path="<t>",
                X=X,
                raw=None,
                z_content=z,
                z_style=None,
                adjacency=adjacency,
                content_names=["a", "b", "c"],
                style_names=[],
                meta={},
                view="1",
            )

        alone = compare.score_all({"s": make(strong)}, {}, self.options)
        together = compare.score_all({"s": make(strong), "w": make(weak)}, {}, self.options)
        for name, row in alone["s"]["content"].items():
            if name == "_block":
                continue
            for key in ("real", "null", "gap"):
                self.assertAlmostEqual(row[key], together["s"]["content"][name][key], places=12)

    def test_a_cleaner_representation_scores_higher_on_every_factor(self):
        z, adjacency, strong, weak = self.bundles()

        def make(X):
            return dict(
                path="<t>",
                X=X,
                raw=None,
                z_content=z,
                z_style=None,
                adjacency=adjacency,
                content_names=["a", "b", "c"],
                style_names=[],
                meta={},
                view="1",
            )

        results = compare.score_all({"strong": make(strong), "weak": make(weak)}, {}, self.options)
        for name in ("a", "b", "c"):
            with self.subTest(factor=name):
                self.assertGreater(results["strong"]["content"][name]["gap"], results["weak"]["content"][name]["gap"])

    def test_equal_width_pins_every_bundle_to_the_narrowest_block(self):
        z, adjacency, wide = planted(width=40)
        _z, _a, narrow = planted(width=12, seed=1)

        def make(X):
            return dict(
                path="<t>",
                X=X,
                raw=None,
                z_content=z,
                z_style=None,
                adjacency=adjacency,
                content_names=["a", "b", "c"],
                style_names=[],
                meta={},
                view="1",
            )

        bundles = {"wide": make(wide), "narrow": make(narrow)}
        self.assertEqual(compare.common_width(bundles, {}), 12)
        # A floor narrower than either model pulls the shared width down to itself.
        self.assertEqual(compare.common_width(bundles, {"wide": make(planted(width=8, seed=2)[2])}), 8)

    def test_a_floor_is_only_subtracted_from_its_own_bundle(self):
        z, adjacency, strong, weak = self.bundles()

        def make(X):
            return dict(
                path="<t>",
                X=X,
                raw=None,
                z_content=z,
                z_style=None,
                adjacency=adjacency,
                content_names=["a", "b", "c"],
                style_names=[],
                meta={},
                view="1",
            )

        _zf, _af, floor_features = planted(noise=8.0, seed=3)
        results = compare.score_all({"s": make(strong), "w": make(weak)}, {"s": make(floor_features)}, self.options)
        self.assertIn("delta_floor", results["s"]["content"]["a"])
        self.assertNotIn("delta_floor", results["w"]["content"]["a"])

    def test_named_arguments_reject_malformed_and_duplicate_labels(self):
        self.assertEqual(
            compare.parse_named(["a=x.npz", "b=y.npz"], "--bundles"), {"a": Path("x.npz"), "b": Path("y.npz")}
        )
        for bad in (["nope.npz"], ["=x.npz"], ["a=x.npz", "a=y.npz"]):
            with self.subTest(bad=bad):
                with self.assertRaises(argparse.ArgumentTypeError):
                    compare.parse_named(bad, "--bundles")

    def test_report_flags_a_width_mismatch_it_cannot_fix(self):
        z, adjacency, wide = planted(width=40)
        _z, _a, narrow = planted(width=12, seed=1)

        def make(X):
            return dict(
                path="<t>",
                X=X,
                raw=None,
                z_content=z,
                z_style=None,
                adjacency=adjacency,
                content_names=["a", "b", "c"],
                style_names=[],
                meta={},
                view="1",
            )

        options = argparse.Namespace(**{**vars(self.options), "probe_dim": "auto"})
        results = compare.score_all({"wide": make(wide), "narrow": make(narrow)}, {}, options)
        results["wide"]["content"]["_block"]["probe_features"] = 40
        results["narrow"]["content"]["_block"]["probe_features"] = 12
        self.assertIn("different widths", compare.format_widths(results))


class GraphComparisonTests(unittest.TestCase):
    """PC on each representation's features, scored against the true SCM adjacency."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self.z, self.adjacency, self.strong = planted(n=240, noise=0.1, width=32)
        _z, _a, self.weak = planted(n=240, noise=3.0, seed=1, width=32)
        self.options = argparse.Namespace(
            probe_kind="ridge",
            seeds=(0,),
            n_splits=3,
            n_null=1,
            null_seed=0,
            probe_dim=0,
            with_graph=True,
            pc_ceiling=True,
            alphas=(0.05,),
            diagnostic_alpha=0.05,
            orientation=True,
            indep_test="fisherz",
            max_cond_set=None,
            readout_dim=8,
            holdout_readout=False,
        )

    def make(self, X):
        return dict(
            path="<t>",
            X=X,
            raw=None,
            z_content=self.z,
            z_style=None,
            adjacency=self.adjacency,
            content_names=["a", "b", "c"],
            style_names=[],
            meta={},
            view="1",
        )

    def scored(self, bundles, floors=None):
        truth = compare.truth_panel(bundles, self.options)
        options = argparse.Namespace(**{**vars(self.options), "pc_ceiling": False})
        return compare.score_all(bundles, floors or {}, options), truth, options

    def test_every_source_is_scored_against_the_same_true_skeleton(self):
        bundles = {"strong": self.make(self.strong), "weak": self.make(self.weak)}
        results, truth, _o = self.scored(bundles)
        skeletons = [tuple(map(tuple, panel["true_skeleton"])) for _l, panel in compare.graph_panels(results, truth)]
        self.assertEqual(len(set(skeletons)), 1, "the truth must not vary between sources")
        expected = self.adjacency | self.adjacency.T
        np.testing.assert_array_equal(np.array(skeletons[0], dtype=bool), expected)

    def test_a_degraded_representation_recovers_a_worse_graph(self):
        bundles = {"strong": self.make(self.strong), "weak": self.make(self.weak)}
        results, truth, options = self.scored(bundles)
        rows = dict(compare.graph_panels(results, truth))
        f1 = {label: compare._sweep_row(panel, options.diagnostic_alpha)["f1"] for label, panel in rows.items()}
        self.assertGreaterEqual(f1["strong"], f1["weak"])
        self.assertEqual(f1[compare.CEILING_LABEL], 1.0)

    def test_the_ceiling_is_scored_once_and_not_repeated_per_bundle(self):
        bundles = {"strong": self.make(self.strong), "weak": self.make(self.weak)}
        results, truth, _o = self.scored(bundles)
        self.assertIsNotNone(truth)
        for result in results.values():
            self.assertNotIn("truth", result["graph"])
        self.assertEqual(
            [label for label, _p in compare.graph_panels(results, truth)],
            [compare.CEILING_LABEL, "strong", "weak"],
        )

    def test_a_floor_appears_as_its_own_row(self):
        bundles = {"m": self.make(self.strong)}
        _zf, _af, floor_features = planted(n=240, noise=9.0, seed=4, width=32)
        results, truth, _o = self.scored(bundles, {"m": self.make(floor_features)})
        labels = [label for label, _p in compare.graph_panels(results, truth)]
        self.assertEqual(labels, [compare.CEILING_LABEL, "m", "m · floor"])

    def test_readout_width_is_pinned_to_the_narrowest_block(self):
        wide, narrow = self.make(self.strong), self.make(self.weak[:, :12])
        bundles = {"wide": wide, "narrow": narrow}
        options = argparse.Namespace(**{**vars(self.options), "readout_dim": None})
        # The default rule would give the 32-wide block 64-capped-to-32 and the 12-wide one 12.
        self.assertEqual(compare.common_readout(bundles, {}, options), 12)
        self.assertEqual(compare.common_readout(bundles, {"wide": self.make(self.strong[:, :5])}, options), 5)

    def test_the_ceiling_does_not_trigger_the_width_warning(self):
        # Its features ARE the factors, so it is always narrower than any model's readout;
        # letting that count as a mismatch would print the warning on every causal run.
        bundles = {"strong": self.make(self.strong), "weak": self.make(self.weak)}
        results, truth, options = self.scored(bundles)
        text = compare.format_graph(results, truth, options)
        self.assertIn("CAUSAL DISCOVERY", text)
        self.assertNotIn("different widths", text)
        self.assertIn("the ceiling reads out its 3 factors directly", text)

    def test_a_genuine_readout_mismatch_is_still_flagged(self):
        bundles = {"strong": self.make(self.strong), "weak": self.make(self.weak)}
        results, truth, options = self.scored(bundles)
        results["weak"]["graph"]["embeddings"]["graph_readout_dim"] = 4
        self.assertIn("different widths", compare.format_graph(results, truth, options))

    def test_both_alpha_selections_are_reported(self):
        bundles = {"strong": self.make(self.strong)}
        results, truth, options = self.scored(bundles)
        text = compare.format_graph(results, truth, options)
        self.assertIn("at the prespecified alpha=0.05", text)
        self.assertIn("best-F1 alpha, selected against the truth", text)
        self.assertIn("orientation vs the true CPDAG", text)

    def test_sweep_lookup_returns_none_when_pc_failed_at_that_alpha(self):
        panel = {"alpha_sweep": [{"alpha": 0.05, "error": "singular"}, {"alpha": 0.1, "f1": 0.5}]}
        self.assertIsNone(compare._sweep_row(panel, 0.05))
        self.assertIsNone(compare._sweep_row(panel, 0.2))
        self.assertEqual(compare._sweep_row(panel, 0.1)["f1"], 0.5)

    def test_graph_csv_has_one_row_per_source_and_alpha_with_selection_marked(self):
        bundles = {"strong": self.make(self.strong), "weak": self.make(self.weak)}
        results, truth, options = self.scored(bundles)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.csv"
            self.assertTrue(compare.write_graph_csv(path, results, truth, options))
            rows = list(csv.DictReader(path.open()))
        self.assertEqual({r["source"] for r in rows}, {compare.CEILING_LABEL, "strong", "weak"})
        for row in rows:
            self.assertIn("prespecified", row["selected"])
            self.assertEqual(float(row["alpha"]), 0.05)
            self.assertTrue(0.0 <= float(row["f1"]) <= 1.0)
            self.assertEqual(int(row["readout_dim"]), 3 if row["source"] == compare.CEILING_LABEL else 8)

    def test_no_graph_section_when_the_panel_was_not_run(self):
        bundles = {"strong": self.make(self.strong), "weak": self.make(self.weak)}
        options = argparse.Namespace(**{**vars(self.options), "with_graph": False})
        results = compare.score_all(bundles, {}, options)
        self.assertEqual(compare.format_graph(results, None, options), "")
        self.assertNotIn("CAUSAL DISCOVERY", compare.format_report(results, [], None, options))

    def test_a_non_causal_bundle_has_no_ceiling_to_score(self):
        bundle = self.make(self.strong)
        bundle["adjacency"] = None
        self.assertIsNone(compare.truth_panel({"m": bundle}, self.options))


if __name__ == "__main__":
    unittest.main()
