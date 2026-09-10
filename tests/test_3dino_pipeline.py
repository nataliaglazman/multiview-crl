"""Pipeline tests: actual scoring of cached embeddings, matched floors, and stage wiring."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eval import run_3dino_identifiability as pipeline


def bundle(path, random_init=False):
    rng = np.random.RandomState(12)
    z = rng.randn(120, 3).astype(np.float32)
    z[:, 1] += z[:, 0]
    z[:, 2] += z[:, 1]
    style1, style2 = rng.randn(120, 2), rng.randn(120, 2)
    adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    meta = dict(
        backbone="3dino",
        model_id="AICONSlab/3DINO-ViT",
        architecture="ThreeDINOEncoder",
        volume_size=112,
        patch_size=16,
        token_pool="cls",
        grid_size=None,
        image_mean=[0.0],
        image_std=[1.0],
        window="per_volume",
        window_pct=[0.05, 99.95],
        window_values=None,
        synthetic_fixed_reference=None,
        generator={"synthetic_seed": 42},
        raw_grid=2,
        dtype="float32",
        num_samples=len(z),
        random_init=random_init,
        run_dir="fixture",
        content_factor_names=["a", "b", "c"],
        style_factor_names=["gain", "bias"],
        model_provenance={"revision": "fixture"},
    )
    features = rng.randn(120, 8) if random_init else np.column_stack([z, style1, rng.randn(120, 3)])
    np.savez(
        path,
        emb_view1=features,
        emb_view2=features + 0.01 * rng.randn(*features.shape),
        z_content=z,
        z_style_v1=style1,
        z_style_v2=style2,
        causal_adj=adj,
        raw_view1=z,
        raw_view2=z,
        meta=json.dumps(meta),
    )


class PipelineTests(unittest.TestCase):
    def test_old_pc_cannot_silently_ignore_the_conditioning_limit(self):
        def old_pc(data, **kwargs):
            pass

        with patch("causallearn.search.ConstraintBased.PC.pc", old_pc):
            pipeline.validate_graph_support(None)
            with self.assertRaisesRegex(ValueError, "causal-learn does not support"):
                pipeline.validate_graph_support(0)

    def test_real_scorer_produces_two_view_reports_summary_and_logs_from_cached_embeddings(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle(root / "trained.npz")
            bundle(root / "floor.npz", True)
            args = [
                "--embeddings",
                str(root / "trained.npz"),
                "--floor",
                str(root / "floor.npz"),
                "--output-dir",
                str(root / "eval"),
                "--n-splits",
                "2",
                "--n-null",
                "1",
                "--seeds",
                "0",
                "--alphas",
                ".05",
                "--no-orientation",
                "--readout-dim",
                "3",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.main(args), 0)
            manifest = json.loads((root / "eval/pipeline.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(
                [stage["name"] for stage in manifest["stages"]],
                ["score_view1", "score_view2"],
            )
            for view in (1, 2):
                for file in (
                    f"report_view{view}.json",
                    f"report_view{view}.txt",
                    f"factors_view{view}.csv",
                    f"score_view{view}.log",
                ):
                    self.assertTrue((root / "eval" / file).is_file())
                report = json.loads((root / f"eval/report_view{view}.json").read_text())
                self.assertIn("floor_mean_gap", report["content"]["_block"])
                self.assertTrue(report["graph"]["embeddings"]["best"])
            with (root / "eval/summary.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["view"] for r in rows], ["1", "2"])
            self.assertNotEqual(rows[0]["partial_r2"], "")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                pipeline.main(args)  # Must not overwrite an earlier evaluation.

    def test_floor_checks_all_labels_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "trained.npz", root / "floor.npz"
            bundle(first)
            bundle(second, True)
            pipeline.validate_floor(first, second, ["1", "2"])
            with np.load(second) as data:
                original = dict(data)
            for key in ("z_content", "z_style_v2", "causal_adj"):
                changed = dict(original)
                changed[key] = original[key] + 1
                np.savez(second, **changed)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    pipeline.validate_floor(first, second, ["1", "2"])
            for key, value in (
                ("token_pool", "grid"),
                ("window", "dataset"),
                ("random_init", False),
            ):
                changed = dict(original)
                meta = json.loads(str(changed["meta"]))
                meta[key] = value
                changed["meta"] = json.dumps(meta)
                np.savez(second, **changed)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    pipeline.validate_floor(first, second, ["1", "2"])

    def test_fresh_pipeline_uses_same_generator_preprocessing_and_auto_detects_finetuning(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "upstream"
            (repo / "dinov2/models").mkdir(parents=True)
            (repo / "dinov2/models/vision_transformer.py").touch()
            run = root / "run"
            (run / "encoder").mkdir(parents=True)
            (run / "settings.json").write_text("{}")
            (run / "preprocessing.json").write_text('{"backbone":"3dino"}')
            stages = []

            def stage(name, module, args, output_dir, manifest):
                stages.append((name, module, args))
                if module.endswith("embed_synthetic"):
                    bundle(Path(args[args.index("--out") + 1]), "--random-init" in args)

            with patch.object(pipeline, "run_stage", side_effect=stage), patch.object(
                pipeline, "write_summary", return_value=[]
            ), contextlib.redirect_stdout(io.StringIO()):
                pipeline.main(
                    [
                        "--three-dino-weights",
                        str(run / "encoder"),
                        "--three-dino-repo",
                        str(repo),
                        "--output-dir",
                        str(root / "eval"),
                        "--with-floor",
                        "--floor-seed",
                        "4",
                        "--eval-views",
                        "both",
                        "--holdout-readout",
                        "--indep-test",
                        "kci",
                        "--max-cond-set",
                        "2",
                    ]
                )
            first, floor, scoring = [stage[2] for stage in stages]
            for key in (
                "--run-dir",
                "--preprocessing",
                "--three-dino-weights",
                "--window",
                "--num-samples",
            ):
                self.assertEqual(first[first.index(key) + 1], floor[floor.index(key) + 1])
            self.assertEqual(
                first[first.index("--preprocessing") + 1],
                str((run / "preprocessing.json").resolve()),
            )
            self.assertEqual(floor[floor.index("--model-seed") + 1], "4")
            self.assertIn("--random-init", floor)
            self.assertIn("--holdout-readout", scoring)
            self.assertEqual(scoring[scoring.index("--indep-test") + 1], "kci")
            self.assertEqual(scoring[scoring.index("--max-cond-set") + 1], "2")

    def test_stage_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle(root / "trained.npz")
            with patch.object(
                pipeline, "run_stage", side_effect=RuntimeError("injected failure")
            ), self.assertRaisesRegex(RuntimeError, "injected failure"):
                pipeline.main(
                    [
                        "--embeddings",
                        str(root / "trained.npz"),
                        "--output-dir",
                        str(root / "eval"),
                    ]
                )
            manifest = json.loads((root / "eval/pipeline.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("injected failure", manifest["error"])


if __name__ == "__main__":
    unittest.main()
