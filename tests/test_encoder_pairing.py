"""Real optimization, independent RNGs and initialization/data replay across backbones."""

import contextlib
import gc
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from eval.protocol.score_checkpoint import build_model
from eval.protocol.score_checkpoint import make_dataset as evaluation_dataset
from training import main_conv_synthetic as trainer


class TinyImages(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index + 1200)
        image = torch.randn(1, 32, 32, 32, generator=generator)
        return {"index": index, "image": [image, image * 1.3 + 0.2]}


class EncoderPairingTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        benchmark = torch.backends.cudnn.benchmark
        cudnn_deterministic = torch.backends.cudnn.deterministic
        self.addCleanup(torch.set_num_threads, threads)
        self.addCleanup(torch.use_deterministic_algorithms, deterministic, warn_only=warn_only)
        self.addCleanup(setattr, torch.backends.cudnn, "benchmark", benchmark)
        self.addCleanup(setattr, torch.backends.cudnn, "deterministic", cudnn_deterministic)
        torch.set_num_threads(2)

    def test_training_pairs_batches_despite_architecture_and_evaluation_rng_consumption(self):
        def dataset(*_args, **_kwargs):
            torch.manual_seed(999)  # The real generator's constructor also resets the RNG.
            return TinyImages()

        def evaluate(model, *_args, **_kwargs):
            torch.rand(7 if model.encoder_architecture == "conv" else 123)
            return {}, {}

        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            progress = []
            for architecture in ("conv", "resnet18"):
                argv = [
                    "train",
                    "--out-dir",
                    temp,
                    "--model-id",
                    architecture,
                    "--encoder-architecture",
                    architecture,
                    "--res",
                    "32",
                    "--hidden-channels",
                    "8",
                    "--res-channels",
                    "4",
                    "--nb-res-layers",
                    "1",
                    "--encoder-head-hidden",
                    "4",
                    "--latent-dim",
                    "12",
                    "--content-channels",
                    "9",
                    "--batch-size",
                    "2",
                    "--num-train-samples",
                    "6",
                    "--train-steps",
                    "4",
                    "--eval-every",
                    "1",
                    "--no-floor-eval",
                    "--best-metric",
                    "none",
                    "--no-cuda",
                    "--no-cache",
                    "--require-new-run",
                    "--hash-training-inputs",
                    "--deterministic",
                    "--deterministic-warn-only",
                    "--cpu-threads",
                    "1",
                    "--no-separate-encoders",
                    "--seed",
                    "17",
                    "--model-seed",
                    "23",
                    "--data-seed",
                    "42",
                    "--loader-seed",
                    "1200",
                    "--tau",
                    "0.1",
                ]
                with patch("sys.argv", argv), patch.object(trainer, "make_dataset", side_effect=dataset), patch.object(
                    trainer, "evaluate", side_effect=evaluate
                ):
                    trainer.main()
                run = Path(temp) / architecture
                config = json.loads((run / "settings.json").read_text())
                initial = torch.load(run / "model_init.pt", map_location="cpu", weights_only=True)
                restored = build_model(config, "cpu")
                for key, value in restored.state_dict().items():
                    torch.testing.assert_close(value, initial[key], atol=0, rtol=0)
                trained = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
                weight = "encoder.conv1.weight" if architecture == "resnet18" else "encoder.layers.0.0.weight"
                self.assertFalse(torch.equal(trained[weight], initial[weight]))
                progress.append(json.loads((run / "training_progress.json").read_text()))
                self.assertEqual(progress[-1]["step"], 4)
                self.assertEqual(progress[-1]["status"], "complete")
                self.assertEqual(progress[-1]["subjects_seen_per_view"], 8)
                self.assertTrue(progress[-1]["deterministic_algorithms"])
                self.assertTrue(progress[-1]["deterministic_warn_only"])
                with patch("sys.argv", argv), self.assertRaises(FileExistsError):
                    trainer.main()
                del initial, restored, trained
                gc.collect()
            self.assertEqual(progress[0]["batch_order_sha256"], progress[1]["batch_order_sha256"])
            self.assertEqual(progress[0]["training_input_sha256"], progress[1]["training_input_sha256"])
            self.assertNotEqual(progress[0]["parameter_count"], progress[1]["parameter_count"])

    def test_warn_only_requires_determinism_flag(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            trainer.parse_args(["--deterministic-warn-only"])

    def test_generator_seed_round_trip_is_separate_from_model_seed(self):
        args = trainer.parse_args(
            [
                "--no-cache",
                "--res",
                "32",
                "--seed",
                "5",
                "--data-seed",
                "37",
                "--model-seed",
                "19",
                "--synthetic-clean-content",
                "--synthetic-normalize",
                "fixed_reference",
            ]
        )
        self.assertEqual((args.data_seed, args.model_seed, args.loader_seed), (37, 19, 10005))
        train_factory = trainer.make_dataset(args, "val", 2)
        eval_factory = evaluation_dataset(vars(args), 2, mode="val")
        self.assertEqual(train_factory._inner.seed, 38)
        for index in range(2):
            a, b = train_factory[index], eval_factory[index]
            for x, y in zip(a["image"], b["image"]):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
            torch.testing.assert_close(a["gt_latents"]["z_content"], b["gt_latents"]["z_content"], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
