"""Train-only centroid fitting, selective mutation and paired held-out evaluation."""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from eval.codebook_recalibration import (
    collect_points,
    main,
    make_dataset,
    paired_change,
    recalibrated_copy,
    refit_centers,
    response_comparison,
    share_training_reference,
    state_digest,
)


def real_model(**kwargs):
    source = Path(__file__).resolve().parents[1] / "models/vqvae.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        n for n in tree.body if not (isinstance(n, ast.Import) and any(a.name == "utils.utils" for a in n.names))
    ]
    namespace = {"__name__": "recalibration_test_model"}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["VQVAE"](
        hidden_channels=8,
        res_channels=4,
        nb_res_layers=1,
        nb_levels=1,
        embed_dim=8,
        nb_entries=8,
        scaling_rates=[2],
        use_checkpoint=False,
        content_size=6,
        style_size=2,
        content_style_levels=[0],
        mask_mode="fixed",
        inject_style_to_decoder=True,
        style_injection_mode="input",
        norm_type="layer",
        **kwargs,
    ).eval()


def dataset_args():
    return argparse.Namespace(
        synthetic_mode="pseudo_mri",
        synthetic_res=32,
        synthetic_clean_content=True,
        synthetic_normalize="fixed_reference",
        synthetic_n_content=9,
        synthetic_n_style=3,
        synthetic_identifiable_ventricle=True,
    )


class RecalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(5)

    def test_known_cluster_means_and_empty_entry_are_preserved(self):
        points = torch.tensor([[1.0, 1.0], [1.0, 3.0], [9.0, 1.0], [9.0, 3.0]])
        initial = torch.tensor([[0.0, 0.0], [10.0, 0.0], [100.0, 100.0]])
        before = initial.clone()
        centers, info = refit_centers(points, initial, 3, 2)
        torch.testing.assert_close(centers, torch.tensor([[1.0, 2.0], [9.0, 2.0], [100.0, 100.0]]))
        torch.testing.assert_close(initial, before, rtol=0, atol=0)
        self.assertEqual(info["entries_moved"], 2)
        self.assertEqual(info["trace"][1]["empty_entries_retained"], 1)
        self.assertLess(info["trace"][info["selected_iteration"]]["training_mse"], info["trace"][0]["training_mse"])
        other, _ = refit_centers(points, initial, 3, 4)
        torch.testing.assert_close(centers, other)

    def test_identity_fit_keeps_original_when_no_update_improves_objective(self):
        centers = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
        fitted, info = refit_centers(centers.repeat(3, 1), centers, 2, 2)
        torch.testing.assert_close(fitted, centers)
        self.assertEqual(info["selected_iteration"], 0)
        self.assertEqual(info["entries_moved"], 0)

    def test_only_content_embeddings_change_in_disposable_copy(self):
        for separate in (False, True):
            model = real_model(quantize_style=True, separate_style_codebooks=True, separate_content_codebooks=separate)
            before = state_digest(model)
            banks = {name: torch.randn(64, 8) for name in (("t1", "flair") if separate else ("shared",))}
            candidate, info, vectors, changed = recalibrated_copy(model, banks, "cpu", 2, 16)
            self.assertEqual(before, state_digest(model))
            expected = {"codebooks.0.embed", "codebooks_v1.0.embed"} if separate else {"codebooks.0.embed"}
            self.assertEqual(set(changed), expected)
            for key, value in candidate.state_dict().items():
                if key not in expected:
                    torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
            self.assertFalse(any(m.training for m in candidate.modules()))

    def test_train_reference_and_uniform_sampling_are_deterministic(self):
        train = make_dataset(dataset_args(), 4, "iid", "train")
        test = make_dataset(dataset_args(), 2, "iid", "test")
        with patch.object(test, "_compute_fixed_reference", side_effect=AssertionError("test data used for fit")):
            reference = share_training_reference(train, test)
            self.assertEqual(reference["source"], "train")
            self.assertEqual(train._fixed_mean, test._fixed_mean)
            model = real_model()
            before = state_digest(model)
            a = collect_points(model, train, "cpu", 2, 17, 4)
            b = collect_points(model, train, "cpu", 2, 17, 4)
            torch.testing.assert_close(a["shared"], b["shared"], rtol=0, atol=0)
            self.assertEqual(a["shared"].shape, (4 * 2 * 17, 8))
            self.assertEqual(before, state_digest(model))

    def test_paired_comparison_rejects_mismatches_and_uses_same_subjects(self):
        result = paired_change([1.0, 3.0, np.nan], [2.0, 4.0, 8.0])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["mean_delta"], 1.0)
        self.assertEqual(result["mean_delta_ci_low"], 1.0)
        row = dict(
            index=0,
            eps=0.25,
            view="t1",
            block="content",
            region="all",
            input_measurable=True,
            sites=8,
            input_delta_rms=0.2,
            pre_delta_rms=0.4,
        )
        with self.assertRaisesRegex(ValueError, "do not match"):
            response_comparison([row], [{**row, "index": 1}])
        with self.assertRaisesRegex(ValueError, "continuous features changed"):
            response_comparison([row], [{**row, "pre_delta_rms": 0.6}])

    def test_end_to_end_fits_before_heldout_access_and_never_writes_checkpoint(self):
        model = real_model(quantize_style=True, separate_style_codebooks=True)
        args = dataset_args()
        before_model = state_digest(model)
        fitting_finished = False
        real_refit = recalibrated_copy
        datasets = {}

        def tracked_dataset(args, n, causal, split):
            ds = make_dataset(args, n, causal, split)
            datasets[split] = ds
            if split == "test":
                inner = ds._inner

                class HeldOutGuard:
                    def __getattr__(self, key):
                        return getattr(inner, key)

                    def __len__(self):
                        return len(inner)

                    def __getitem__(self, index):
                        self_outer.assertTrue(fitting_finished, "Held-out images accessed during calibration")
                        return inner[index]

                self_outer = self
                ds._inner = HeldOutGuard()
            return ds

        def tracked_refit(*args, **kwargs):
            nonlocal fitting_finished
            result = real_refit(*args, **kwargs)
            fitting_finished = True
            return result

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "vqvae_model.pt"
            torch.save({"encoders": model.state_dict(), "step": 40}, checkpoint)
            before_file = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            output = Path(tmp) / "audit"
            with patch("eval.run_dci_synthetic.load_model_from_run_dir", return_value=(model, args, "cpu")), patch(
                "eval.codebook_recalibration.make_dataset", side_effect=tracked_dataset
            ), patch(
                "eval.codebook_recalibration.recalibrated_copy", side_effect=tracked_refit
            ), contextlib.redirect_stdout(
                io.StringIO()
            ):
                main(
                    [
                        "--run-dir",
                        tmp,
                        "--fit-samples",
                        "4",
                        "--num-samples",
                        "3",
                        "--sites-per-subject",
                        "32",
                        "--iterations",
                        "2",
                        "--chunk-size",
                        "32",
                        "--batch-size",
                        "2",
                        "--eps",
                        ".5",
                        "--threads",
                        "2",
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(before_file, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(before_model, state_digest(model))
            report = json.loads((output / "summary.json").read_text())
            self.assertTrue(report["original_unchanged"])
            self.assertTrue(report["candidate_unchanged_during_evaluation"])
            self.assertFalse(set(report["train_render_seeds"]) & set(report["test_render_seeds"]))
            self.assertEqual(report["fit"]["shared"]["training_points"], 4 * 2 * 32)
            self.assertEqual(report["changed_state_keys"], ["codebooks.0.embed"])
            self.assertEqual(len(report["reconstruction"]), 4)
            self.assertTrue(all(r["n"] == 3 for r in report["reconstruction"]))
            style = [r for r in report["response_comparison"] if r["block"] == "style" and r["n"]]
            self.assertTrue(all(abs(r["mean_delta"]) < 1e-7 for r in style))
            centers = np.load(output / "diagnostic_centers.npz")
            self.assertEqual(centers["shared_original"].shape, (8, 8))
            self.assertFalse(list(output.glob("*.pt")))


if __name__ == "__main__":
    unittest.main()
