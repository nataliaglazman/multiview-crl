"""Controls for the lesion reconstruction diagnostic, including real rendering."""

import argparse
import unittest

import numpy as np

from eval.lesion_reconstruction import audit, json_safe, locate, location_metrics, make_dataset, render_pair


class LocationTests(unittest.TestCase):
    def test_zero_response_does_not_invent_position(self):
        self.assertTrue(np.isnan(locate(np.zeros((16,) * 3), 2, response=True)).all())
        m = location_metrics([[1, 2, 3], [4, 5, 6]], [[np.nan] * 3] * 2)
        self.assertEqual(m["n_valid"], 0)

    def test_shuffled_subject_control_and_missing_denominator(self):
        truth = np.arange(30).reshape(10, 3)
        m = location_metrics(truth, truth)
        self.assertEqual(m["median_error_vox"], 0)
        self.assertGreater(m["shuffled_median_error_vox"], 0)
        pred = truth.astype(float)
        pred[0] = np.nan
        self.assertEqual(location_metrics(truth, pred)["within_2_vox_fraction_all"], 0.9)
        self.assertEqual(json_safe({"x": [np.nan, float("inf")]}), {"x": [None, None]})

    def test_detector_localizes_shift_and_polarity(self):
        grid = np.indices((32,) * 3)
        for center in ([9, 12, 20], [21, 18, 10]):
            blob = np.exp(-sum((grid[i] - center[i]) ** 2 for i in range(3)) / 8)
            np.testing.assert_array_equal(locate(-blob, 3, response=True), center)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        torch.set_num_threads(2)
        cls.args = argparse.Namespace(
            synthetic_mode="pseudo_mri",
            synthetic_res=32,
            synthetic_content_prior="uniform",
            synthetic_content_squash="none",
            synthetic_clean_content=True,
            synthetic_lesion_radius=0.14,
            synthetic_normalize="per_sample",
            synthetic_causal=True,
            synthetic_n_content=9,
        )

    def test_normalization_freeze_preserves_input_and_local_intervention(self):
        from scipy.ndimage import maximum_filter

        ds = make_dataset(self.args, 4, "iid", "test")
        self.assertFalse(ds._inner.causal)
        on, off, _, support, _, _ = render_pair(ds, 0)
        observed = ds[0]["image"]
        for v in range(2):
            np.testing.assert_allclose(on[v].numpy(), observed[v].numpy(), atol=1e-6)
            delta = (on[v] - off[v]).numpy().squeeze()
            self.assertLess(np.abs(delta[~maximum_filter(support, size=3)]).max(), 2e-6)

    def test_identity_and_constant_decoder_controls(self):
        class Identity:
            def __call__(self, x, **kwargs):
                return (x,)

        class Constant:
            def __call__(self, x, **kwargs):
                return (x * 0,)

        class Shifted:
            def __call__(self, x, **kwargs):
                return (x.roll(6, dims=2),)

        ds = make_dataset(self.args, 4, "iid", "test")
        rows, good, _ = audit(Identity(), ds, "cpu", batch_size=3, examples=0)
        self.assertEqual(len(rows), 4)
        for v in ("t1", "flair"):
            self.assertAlmostEqual(good["response"][v + "_response_gain"], 1.0, places=5)
            self.assertLess(good[f"recon_{v}_response"]["median_error_vox"], 1.5)
            self.assertEqual(good[f"input_{v}_response"], good[f"recon_{v}_response"])
        _, bad, _ = audit(Constant(), ds, "cpu", batch_size=3, examples=0)
        self.assertEqual(bad["recon_flair_response"]["n_valid"], 0)
        self.assertEqual(bad["response"]["flair_response_gain"], 0)
        _, shifted, _ = audit(Shifted(), ds, "cpu", batch_size=3, examples=0)
        self.assertGreater(shifted["recon_flair_response"]["median_error_vox"], 4)

    def test_field_mode_is_rejected(self):
        args = argparse.Namespace(**vars(self.args), synthetic_lesion_mode="field")
        with self.assertRaisesRegex(ValueError, "Field lesions"):
            make_dataset(args, 4, "iid", "test")


if __name__ == "__main__":
    unittest.main()
