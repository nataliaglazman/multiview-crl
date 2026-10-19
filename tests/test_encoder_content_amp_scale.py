"""--synthetic-content-amp-scale in the encoder-only trainer: parsing, the generator it builds, and eval replay."""

import contextlib
import importlib.util
import io
import sys
import types
import unittest

import torch

from eval.protocol.score_checkpoint import make_dataset as eval_dataset
from training import encoder_args

if importlib.util.find_spec("lpips") is None:
    # training.losses imports LPIPS at module scope; the encoder-only trainer never builds it.
    sys.modules["lpips"] = types.ModuleType("lpips")
    sys.modules["lpips"].LPIPS = None
from training.main_conv_synthetic import make_dataset  # noqa: E402

BASE = [
    *("--n-content", "9", "--res", "32", "--synthetic-clean-content"),
    *("--synthetic-normalize", "fixed_reference", "--synthetic-lesion-placement", "wm_interior"),
]


def parse(*extra):
    return encoder_args.parse_args([*BASE, *extra])


def structures(ds, count=4):
    """Tissue maps and lesion loads of the first subjects, exactly as the dataset renders them."""
    inner = ds._inner
    out = []
    for i in range(count):
        seed = inner.sample_seed_for(i)
        d = inner._draw_pseudo_mri(seed)
        out.append(
            inner.renderer.render_structure(
                d["z_content"], d["z_deformation"], d["z_fissure"], "cpu", clean=True, z_lesion=d["z_lesion"]
            )
        )
    return out


class ContentAmpScaleTests(unittest.TestCase):
    def test_cli_parses_nine_nonnegative_factors_and_refuses_the_rest(self):
        self.assertIsNone(parse().synthetic_content_amp_scale)
        args = parse("--synthetic-content-amp-scale", "1", "0", "1", "1", "1", "0", "0", "0", "0")
        self.assertEqual(args.synthetic_content_amp_scale, [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        for argv in (
            ["--synthetic-content-amp-scale", *["1"] * 8],
            ["--synthetic-content-amp-scale", "-1", *["1"] * 8],
            ["--synthetic-content-amp-scale", "inf", *["1"] * 8],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse(*argv)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            encoder_args.parse_args(["--n-content", "6", "--synthetic-content-amp-scale", *["1"] * 9])

    def test_zero_amplitudes_fix_the_anatomy_and_leave_the_lesion_moving(self):
        fixed = make_dataset(parse("--synthetic-content-amp-scale", "0", "0", *["1"] * 3, *["0"] * 4), "val", 8)
        rendered = structures(fixed)
        for tissue, _ in rendered[1:]:
            torch.testing.assert_close(tissue, rendered[0][0], atol=0, rtol=0)
        self.assertGreater(len({tuple(torch.nonzero(lesion).float().mean(0).tolist()) for _, lesion in rendered}), 1)
        brain = make_dataset(parse("--synthetic-content-amp-scale", "1", "0", *["1"] * 3, *["0"] * 4), "val", 8)
        volumes = {int((tissue > 0).sum()) for tissue, _ in structures(brain)}
        self.assertGreater(len(volumes), 1)

    def test_unit_amplitudes_reproduce_the_default_generator(self):
        default = make_dataset(parse(), "val", 4)
        ones = make_dataset(parse("--synthetic-content-amp-scale", *["1"] * 9), "val", 4)
        for i in range(2):
            for a, b in zip(default[i]["image"], ones[i]["image"]):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_checkpoint_scoring_rebuilds_the_training_generator(self):
        args = parse("--synthetic-content-amp-scale", "1", "0", *["1"] * 3, *["0"] * 4)
        trained = make_dataset(args, "val", 4)
        replayed = eval_dataset(vars(args), 4, mode="val")
        for i in range(2):
            for a, b in zip(trained[i]["image"], replayed[i]["image"]):
                torch.testing.assert_close(a, b, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
