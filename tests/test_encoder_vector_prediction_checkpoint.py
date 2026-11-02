"""Optional vector-prediction architecture and strict checkpoint reconstruction."""

import copy
import io
import unittest

import torch

from eval.protocol.score_checkpoint import build_model


def config(**changes):
    return {
        "hidden_channels": 8,
        "res_channels": 4,
        "nb_res_layers": 0,
        "downscale_factor": 4,
        "latent_dim": 7,
        "content_channels": 4,
        "conv_readout": "mlp",
        "encoder_head_hidden": 8,
        "norm_type": "layer",
        "res": 16,
        "seed": 42,
        **changes,
    }


class VectorPredictionCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_disabled_settings_preserve_legacy_state_and_features(self):
        legacy = build_model(config(), "cpu")
        disabled = build_model(
            config(vector_prediction_weight=0.0, vector_prediction_grid=[2, 2, 2], vector_prediction_hidden=11),
            "cpu",
            legacy.state_dict(),
        )
        self.assertIsNone(disabled.vector_prediction)
        self.assertEqual(set(legacy.state_dict()), set(disabled.state_dict()))
        self.assertFalse(any(key.startswith("vector_prediction.") for key in disabled.state_dict()))
        images = torch.randn(4, 1, 16, 16, 16)
        with torch.no_grad():
            expected = legacy(images, pool_only=True, n_views=2)[2][0]
            actual = disabled(images, pool_only=True, n_views=2)[2][0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_enabled_settings_restore_decoder_parameters_and_template_buffers(self):
        settings = config(
            vector_prediction_weight=0.25,
            vector_prediction_grid=[2, 3, 2],
            vector_prediction_hidden=11,
        )
        original = build_model(settings, "cpu")
        decoder = original.vector_prediction
        self.assertIsNotNone(decoder)
        self.assertEqual((decoder.grid, decoder.hidden), ((2, 3, 2), 11))
        self.assertTrue(list(decoder.parameters()))
        self.assertTrue(list(decoder.buffers()))
        with torch.no_grad():
            for parameter in decoder.parameters():
                parameter.copy_(torch.randn_like(parameter))
            for buffer in decoder.buffers():
                buffer.fill_(True if buffer.dtype == torch.bool else 3)

        stream = io.BytesIO()
        torch.save(original.state_dict(), stream)
        stream.seek(0)
        state = torch.load(stream, weights_only=True)
        restored = build_model(settings, "cpu", state)
        self.assertEqual(set(original.state_dict()), set(restored.state_dict()))
        for key, expected in original.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], expected, rtol=0, atol=0)
        images = torch.randn(4, 1, 16, 16, 16)
        with torch.no_grad():
            expected = original(images, pool_only=True, n_views=2)[2][0]
            actual = restored(images, pool_only=True, n_views=2)[2][0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key, expected in state.items():
            torch.testing.assert_close(restored.state_dict()[key], expected, rtol=0, atol=0)

    def test_quantile_settings_restore_its_outputs_and_buffers_and_refuse_a_spatial_state(self):
        settings = config(
            vector_prediction_weight=1.0,
            vector_prediction_grid=[2, 2, 2],
            vector_prediction_target="quantile",
            vector_prediction_components=3,
        )
        original = build_model(settings, "cpu")
        decoder = original.vector_prediction
        self.assertEqual(
            (decoder.target, decoder.components, decoder.decoders[0][-1].out_features), ("quantile", 3, 17)
        )
        with torch.no_grad():
            decoder.target_mean.fill_(2)
        restored = build_model(settings, "cpu", original.state_dict())
        torch.testing.assert_close(restored.vector_prediction.target_mean, decoder.target_mean, rtol=0, atol=0)
        spatial = config(vector_prediction_weight=1.0, vector_prediction_grid=[2, 2, 2])
        with self.assertRaises(RuntimeError):
            build_model(spatial, "cpu", original.state_dict())

    def test_enabled_default_settings_restore_default_grid_and_hidden_width(self):
        model = build_model(config(vector_prediction_weight=1.0), "cpu")
        self.assertEqual((model.vector_prediction.grid, model.vector_prediction.hidden), ((16, 16, 16), 128))

    def test_saved_decoder_requires_enabled_matching_settings_and_complete_state(self):
        settings = config(vector_prediction_weight=1.0, vector_prediction_grid=[2, 2, 2])
        original = build_model(settings, "cpu")
        state = original.state_dict()
        with self.assertRaises(RuntimeError):
            build_model(config(), "cpu", state)
        incomplete = copy.copy(state)
        missing = next(key for key, _ in original.named_buffers() if key.startswith("vector_prediction."))
        del incomplete[missing]
        with self.assertRaises(RuntimeError):
            build_model(settings, "cpu", incomplete)
        with self.assertRaises(RuntimeError):
            build_model({**settings, "vector_prediction_grid": [3, 2, 2]}, "cpu", state)


if __name__ == "__main__":
    unittest.main()
