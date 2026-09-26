"""Figures from compare_bundles' JSON: what is drawn, and what is correctly not drawn."""

import json
import tempfile
import unittest
from pathlib import Path

from eval import plot_compare_bundles as plot
from eval.plot_identifiability import THEME


def factor_row(gap, floor=None, real=None):
    row = {"real": real if real is not None else gap - 0.02, "null": -0.02, "gap": gap, "std": 0.01, "mcc": 0.8}
    if floor is not None:
        row.update(floor_gap=floor, delta_floor=gap - floor, floor_mcc=0.6)
    return row


def panel(f1, shd, alpha=0.05, factors=("a", "b")):
    sweep = {
        "alpha": alpha,
        "f1": f1,
        "precision": f1,
        "recall": f1,
        "skeleton_shd": shd,
        "tp": 4,
        "fp": 1,
        "fn": 1,
        "exact_match": False,
    }
    return {
        "alpha_sweep": [sweep],
        "best": dict(sweep),
        "graph_readout_dim": 8,
        "graph_samples": 300,
        "readout_mode": "in_sample",
        "indep_test": "fisherz",
        "num_features": 16,
        "raw_r2_mean": 0.7,
        "partial_r2_mean": 0.5,
        "factors": [
            {
                "dim": i,
                "name": n,
                "parents": [] if i == 0 else [0],
                "raw_r2": 0.8 - 0.1 * i,
                "partial_r2": 0.6 - 0.1 * i,
                "gap": 0.2,
            }
            for i, n in enumerate(factors)
        ],
    }


def report(with_floor=True, with_graph=True, with_stability=True, with_style=True, labels=("vq", "dino")):
    results = {}
    for i, label in enumerate(labels):
        entry = {
            "content": {
                "a": factor_row(0.9 - 0.1 * i, 0.5 if with_floor else None),
                "b": factor_row(0.4 - 0.1 * i, 0.5 if with_floor else None),
                "_block": {"mean_gap": 0.65, "probe_features": 16},
            },
            "num_features": 16,
            "num_samples": 300,
            "has_voxels": False,
            "floor_path": "floor.npz" if with_floor else None,
            "graph": {"embeddings": panel(0.8 - 0.05 * i, 3 + i)} if with_graph else {},
        }
        if with_floor and with_graph:
            entry["graph"]["floor"] = panel(0.6, 6)
        if with_style:
            entry["style"] = {
                "gain": factor_row(0.5, 0.2 if with_floor else None),
                "_block": {"mean_gap": 0.5, "probe_features": 16},
            }
        results[label] = entry
    out = {"results": results, "options": {"diagnostic_alpha": 0.05}, "row_alignment": "verified"}
    if with_graph:
        out["truth_panel"] = panel(0.95, 1)
        if with_stability:
            out["graph_stability"] = {
                label: {
                    "repeats": 5,
                    "subsample": 240,
                    "requested": 5,
                    "f1_mean": 0.8,
                    "f1_std": 0.03,
                    "skeleton_shd_mean": 3.0,
                    "skeleton_shd_std": 0.5,
                }
                for label in list(labels) + [f"{l}{plot.FLOOR_SUFFIX}" for l in labels]
            }
    return out


class SeriesTests(unittest.TestCase):
    def test_hues_are_assigned_in_fixed_order(self):
        t = THEME["light"]
        colours = plot.series_colours(["vq", "vq_content", "dino"], t)
        self.assertEqual([colours[k] for k in ("vq", "vq_content", "dino")], t["series"][:3])

    def test_a_colour_follows_the_model_not_its_rank_in_a_subset(self):
        # The floor figure draws only the bundles that have a floor. If the hue were
        # assigned over that subset, dropping a model would repaint the survivors.
        t = THEME["light"]
        full = plot.series_colours(["vq", "vq_content", "dino"], t)
        data = report(labels=("vq", "vq_content", "dino"))
        for label in data["results"]:
            data["results"][label]["floor_path"] = None if label == "vq_content" else "f.npz"
        with tempfile.TemporaryDirectory() as tmp:
            plot.fig_delta_floor(data, "content", t, str(Path(tmp) / "d.png"))
        self.assertEqual(full["dino"], t["series"][2])

    def test_more_models_than_validated_slots_is_refused(self):
        with self.assertRaises(SystemExit) as raised:
            plot.series_colours([f"m{i}" for i in range(plot.MAX_SERIES + 1)], THEME["light"])
        self.assertIn("validated categorical slots", str(raised.exception))

    def test_both_themes_carry_a_hue_for_every_slot(self):
        for mode in ("light", "dark"):
            self.assertGreaterEqual(len(THEME[mode]["series"]), plot.MAX_SERIES)


class DataAccessTests(unittest.TestCase):
    def test_only_factors_every_bundle_scored_are_plotted(self):
        data = report()
        del data["results"]["dino"]["content"]["b"]
        self.assertEqual(plot.factor_order(data, "content"), ["a"])

    def test_graph_rows_put_each_floor_after_its_own_model(self):
        rows = plot.graph_rows(report())
        self.assertEqual(
            [label for label, _p, _b in rows], ["vq", f"vq{plot.FLOOR_SUFFIX}", "dino", f"dino{plot.FLOOR_SUFFIX}"]
        )
        self.assertEqual([base for _l, _p, base in rows], ["vq", "vq", "dino", "dino"])

    def test_the_sweep_row_is_matched_on_alpha_and_missing_is_none(self):
        p = panel(0.8, 3, alpha=0.05)
        self.assertAlmostEqual(plot._panel_row(p, 0.05)["f1"], 0.8)
        self.assertIsNone(plot._panel_row(p, 0.2))
        self.assertIsNone(plot._panel_row({"alpha_sweep": [{"alpha": 0.05, "error": "singular"}]}, 0.05))


class DrawingTests(unittest.TestCase):
    def draw(self, data, mode="light"):
        tmp = tempfile.mkdtemp()
        drawn = plot.draw_all(data, tmp, THEME[mode])
        return tmp, {Path(p).name for p in drawn}

    def test_a_full_report_draws_every_figure_with_a_csv_twin(self):
        tmp, names = self.draw(report())
        self.assertEqual(
            names,
            {
                "factor_recovery_content.png",
                "delta_floor_content.png",
                "factor_recovery_style.png",
                "delta_floor_style.png",
                "causal_discovery.png",
                "partial_r2.png",
            },
        )
        for name in names:
            twin = Path(tmp) / (name[:-4] + ".csv")
            self.assertTrue(twin.is_file(), twin)
            self.assertGreater(len(twin.read_text().splitlines()), 1, f"{twin} has no rows")

    def test_no_floor_means_no_floor_figure_but_the_rest_still_draws(self):
        _tmp, names = self.draw(report(with_floor=False))
        self.assertNotIn("delta_floor_content.png", names)
        self.assertIn("factor_recovery_content.png", names)
        self.assertIn("causal_discovery.png", names)

    def test_no_graph_means_no_causal_figures(self):
        _tmp, names = self.draw(report(with_graph=False))
        self.assertNotIn("causal_discovery.png", names)
        self.assertNotIn("partial_r2.png", names)
        self.assertIn("factor_recovery_content.png", names)

    def test_a_run_without_resampling_still_draws_the_causal_figure(self):
        _tmp, names = self.draw(report(with_stability=False))
        self.assertIn("causal_discovery.png", names)

    def test_dark_mode_renders_from_its_own_steps(self):
        _tmp, names = self.draw(report(), mode="dark")
        self.assertIn("factor_recovery_content.png", names)

    def test_the_factor_csv_carries_the_numbers_the_colours_encode(self):
        tmp, _names = self.draw(report())
        rows = (Path(tmp) / "factor_recovery_content.csv").read_text().splitlines()
        self.assertEqual(
            rows[0].split(","), ["block", "factor", "model", "gap", "real", "null", "floor_gap", "delta_floor", "mcc"]
        )
        self.assertEqual(len(rows), 1 + 2 * 2)  # two factors x two models

    def test_main_writes_into_the_requested_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "compare.json"
            src.write_text(json.dumps(report()))
            out = Path(tmp) / "figures"
            self.assertEqual(plot.main(["--json", str(src), "--out", str(out)]), 0)
            self.assertTrue((out / "causal_discovery.png").is_file())

    def test_an_empty_report_reports_failure_rather_than_writing_nothing_quietly(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "empty.json"
            src.write_text(json.dumps({"results": {}}))
            self.assertEqual(plot.main(["--json", str(src), "--out", str(Path(tmp) / "f")]), 1)


if __name__ == "__main__":
    unittest.main()
