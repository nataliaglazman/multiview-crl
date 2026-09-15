"""Location recovery, target isolation, masked spatial coordinates and cache reuse."""

import contextlib
import io
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from eval import lesion_pooling as lp
from eval import ventricle_pooling as vp


def moving_lesions():
    # Each position is seen in train, validation and test; global statistics are
    # exactly identical while the target position changes. No centroid-based ROI.
    coords = np.indices((4, 4, 4)).reshape(3, -1).T
    points = np.tile(coords, (4, 1))
    patches = torch.eye(64).repeat(4, 1)[None, :, None, :].repeat(2, 1, 1, 1)
    pooled, _ = vp.pool_descriptors(patches, torch.ones(64, dtype=torch.bool), (4, 4, 4), (2, 2, 2))
    pooled["patch_flat"] = patches.flatten(2)
    arrays = {f"{view}/{name}": tensor[v].numpy() for v, view in enumerate(vp.VIEWS) for name, tensor in pooled.items()}
    arrays.update(
        targets=np.concatenate((points, points / 3), axis=1).astype(float),
        subject_id=np.arange(256),
        mask_group=np.arange(256),
        is_fit=np.arange(256) < 192,
        train=np.arange(128),
        validation=np.arange(128, 192),
        test=np.arange(192, 256),
    )
    return arrays, {"format": "lesion_pooling_v1"}


class ToyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = torch.nn.ModuleList([torch.nn.Conv3d(1, 3, 1, bias=False)])
        self.content_channels_per_level = {0: 2}

    def forward(self, x, patch_grid=None, **kwargs):
        h = self.encoders[0](x)
        pooled = torch.nn.functional.adaptive_avg_pool3d(h, tuple(patch_grid)).flatten(2)
        return (
            None,
            None,
            [pooled],
            None,
            None,
            None,
            {0: torch.tensor([[1.0, 0.0, 1.0]])},
        )


class ToyRenderer:
    lesion_radius = 0.1

    def render_structure(self, zc, zd, zf, device, clean=False):
        lesion = torch.zeros(4, 4, 4)
        lesion[tuple(zc[2:5].long().tolist())] = 1
        return torch.zeros_like(lesion), lesion


class ToySubjects:
    res = 4

    def __init__(self, n):
        self.n = n
        self._inner = types.SimpleNamespace(renderer=ToyRenderer(), clean_content=True)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(i + 72)
        c = torch.tensor(rng.integers(0, 4, size=3), dtype=torch.float32)
        z = torch.cat((torch.zeros(2), c, torch.zeros(4)))
        _, lesion = self._inner.renderer.render_structure(z, None, None, "cpu")
        image = lesion[None]
        return {
            "image": [image, image * 2],
            "mask": [torch.ones_like(image)] * 2,
            "gt_latents": {
                "z_content": z,
                "z_deformation": torch.zeros(2, 2, 2),
                "z_fissure": torch.zeros(2, 2, 2),
            },
        }


class LesionPoolingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.threads = threadpool_limits(limits=2)
        self.threads.__enter__()

    def tearDown(self):
        self.threads.__exit__(None, None, None)

    def test_flattened_patches_recover_location_global_stats_cannot(self):
        arrays, metadata = moving_lesions()
        _, table, _ = lp.evaluate(arrays, metadata, draws=10)
        for row in table:
            if row["condition"] == "observed":
                if row["pooling"] in ("gap", "mean_std", "stats"):
                    self.assertLess(abs(row["mean_r2"]), 1e-8)
                elif row["pooling"] == "patch_flat":
                    self.assertGreater(row["mean_r2"], 0.99)
                    self.assertGreater(row["vs_gap_ci_low"], 0.95)
                    if row["target"] == "physical":
                        self.assertLess(row["median_error_vox"], 0.01)
            elif row["pooling"] == "patch_flat":
                self.assertLess(row["mean_r2"], 0.2)

    def test_physical_targets_follow_rendered_mask_and_do_not_change_images(self):
        base = ToySubjects(12)
        wrapped = lp.LesionSubjects(base, np.arange(12)[::-1])
        item = wrapped[0]
        original = base[11]
        for view in range(2):
            torch.testing.assert_close(item["image"][view], original["image"][view])
        expected = original["gt_latents"]["z_content"][2:5]
        torch.testing.assert_close(item["lesion_targets"][:3], expected)
        torch.testing.assert_close(item["lesion_targets"][3:], expected)
        base._inner.renderer.render_structure = lambda *a, **kw: (
            None,
            torch.zeros(4, 4, 4),
        )
        with self.assertRaisesRegex(ValueError, "no rendered lesion"):
            wrapped[0]

    def test_extra_targets_and_flattening_keep_partition_masks_separate(self):
        ds = lp.LesionSubjects(ToySubjects(18), np.arange(18))
        model = ToyEncoder().eval()
        args = types.SimpleNamespace(patch_foreground_mask=True)
        arrays, details = vp.extract_features(
            model,
            ds,
            args,
            "cpu",
            (4, 4, 4),
            (2, 2, 2),
            12,
            128,
            2,
            target_key="lesion_targets",
            include_patch_flat=True,
            partition_ends=[6, 12, 18],
        )
        self.assertEqual(arrays["targets"].shape, (18, 6))
        self.assertEqual(
            [(g["start"], g["stop"]) for g in details["mask_groups"]],
            [(0, 6), (6, 12), (12, 18)],
        )
        self.assertEqual(arrays["t1/patch_flat"].shape, (18, 2 * 64))
        image = ds[0]["image"][0][None]
        with torch.no_grad():
            expected = model.encoders[0](image)[:, [0, 2]].flatten(1).numpy()[0]
        np.testing.assert_allclose(arrays["t1/patch_flat"][0], expected)

    def test_cache_rejects_shared_masks_wrong_targets_and_missing_test_subjects(self):
        arrays, metadata = moving_lesions()
        arrays["mask_group"][arrays["test"]] = 0
        with self.assertRaisesRegex(ValueError, "cross subject"):
            lp.validate_cache(arrays, metadata)
        arrays, metadata = moving_lesions()
        with self.assertRaisesRegex(ValueError, "ventricular caches"):
            lp.validate_cache(arrays, {"format_version": 1})
        arrays["test"] = arrays["test"][:-1]
        with self.assertRaisesRegex(ValueError, "partition subjects"):
            lp.validate_cache(arrays, metadata)

    def test_test_coordinates_do_not_change_selection_or_predictions(self):
        arrays, metadata = moving_lesions()
        first, _, predictions = lp.evaluate(arrays, metadata, draws=5)
        arrays["targets"][arrays["test"]] += 100
        second, _, other = lp.evaluate(arrays, metadata, draws=5)
        observed = lambda rows: [(r["alpha"], r["gamma"]) for r in rows if r["condition"] == "observed"]
        self.assertEqual(observed(first), observed(second))
        for key in predictions:
            if key != "targets":
                np.testing.assert_array_equal(predictions[key], other[key])

    def test_full_run_and_cache_replay_leave_checkpoint_unchanged(self):
        args = types.SimpleNamespace(
            mask_mode="fixed",
            patch_grid=[4, 4, 4],
            batch_size=128,
            vqvae_nb_levels=1,
            patch_foreground_mask=True,
        )
        model = ToyEncoder().eval()
        fake = types.ModuleType("eval.run_dci_synthetic")
        fake.load_run_args = lambda *a: args
        fake.load_model_from_run_dir = lambda *a, **kw: (model, args, "cpu")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "vqvae_model.pt"
            torch.save({"encoders": model.state_dict(), "step": 42}, checkpoint)
            original = checkpoint.read_bytes()
            out = Path(directory) / "results"
            cli = lp.parser().parse_args(
                [
                    "--run-dir",
                    directory,
                    "--train-samples",
                    "16",
                    "--val-samples",
                    "8",
                    "--test-samples",
                    "8",
                    "--bootstrap",
                    "5",
                    "--out",
                    str(out),
                ]
            )
            with patch.dict("sys.modules", {"eval.run_dci_synthetic": fake}), patch.object(
                lp, "make_dataset", side_effect=lambda args, n, *a: ToySubjects(n)
            ), contextlib.redirect_stdout(io.StringIO()):
                first = lp.run(cli)
            self.assertEqual(checkpoint.read_bytes(), original)
            self.assertEqual(first["metadata"]["checkpoint_step"], 42)
            replay = lp.parser().parse_args(
                [
                    "--features",
                    str(out / "features.npz"),
                    "--bootstrap",
                    "5",
                    "--out",
                    str(Path(directory) / "replay"),
                ]
            )
            fake.load_model_from_run_dir = lambda *a, **kw: self.fail("Cache replay loaded a checkpoint")
            with patch.dict("sys.modules", {"eval.run_dci_synthetic": fake}), contextlib.redirect_stdout(io.StringIO()):
                second = lp.run(replay)
            self.assertEqual(first["comparisons"], second["comparisons"])
            self.assertTrue((out / "pooling_scores.csv").exists())
            with self.assertRaises(FileExistsError):
                lp.run(replay)


if __name__ == "__main__":
    unittest.main()
