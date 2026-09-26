"""--project-content: the post-head view must reproduce training's projection exactly.

The point of the flag is that a post-head score is comparable with the pre-head one, so the
eval-side projection has to be the SAME map the contrastive loss applied. Two things can
silently break that and neither shows up as an error: the patch layout (training holds
``(V, B, k, P)`` while extraction hands back a flattened patch-major ``(N, P*k)``), and the
floor twin quietly reusing trained weights. Both are asserted here against a verbatim copy
of ``training.main_multimodal._project_contrastive_content``.
"""

import os
import tempfile
import unittest

import numpy as np
import torch

from eval.run_dci_compare import (
    _CONTENT,
    _CONTENT_V2,
    check_projectable_poolings,
    load_contrastive_proj_heads,
    project_content_reprs,
)

K, HIDDEN, DIM, N, PATCHES = 5, 16, 8, 7, 4


class Args:
    contrastive_proj_dim = DIM
    contrastive_proj_hidden = HIDDEN
    contrastive_proj_mode = "head"
    content_style_levels = [0]


class FakeModel:
    """The helper only needs per-level content widths and a device."""

    def __init__(self, k=K):
        self.content_channels_per_level = {0: k}
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])


def training_reference(head, hz_c, is_patch):
    """Verbatim training.main_multimodal._project_contrastive_content."""
    x = hz_c.permute(0, 1, 3, 2) if is_patch else hz_c
    shape = x.shape
    out = head(x.reshape(-1, shape[-1])).reshape(*shape[:-1], -1)
    return out.permute(0, 1, 3, 2).contiguous() if is_patch else out


def write_checkpoint(path, prefix="module."):
    """A checkpoint shaped like training's: DataParallel prefix, unrelated encoder keys."""
    torch.manual_seed(1234)
    head = torch.nn.Sequential(
        torch.nn.Linear(K, HIDDEN),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(HIDDEN, DIM),
    )
    state = {f"{prefix}_contrastive_proj_heads.L0.{n}": v for n, v in head.state_dict().items()}
    state[f"{prefix}encoders.0.weight"] = torch.randn(3, 3)
    torch.save({"encoders": state, "step": 42}, path)
    return head


class ProjectionHeadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ckpt = os.path.join(self.tmp, "vqvae_model.pt")
        self.trained = write_checkpoint(self.ckpt)
        self.model = FakeModel()

    def test_loads_trained_weights_strictly_through_dataparallel_prefix(self):
        heads = load_contrastive_proj_heads(self.tmp, self.ckpt, Args(), self.model)
        self.assertEqual(set(heads), {0})
        for got, want in zip(heads[0].state_dict().values(), self.trained.state_dict().values()):
            torch.testing.assert_close(got, want)

    def test_gap_projection_matches_training(self):
        heads = load_contrastive_proj_heads(self.tmp, self.ckpt, Args(), self.model)
        v1 = np.random.RandomState(0).randn(N, K).astype(np.float32)
        v2 = np.random.RandomState(1).randn(N, K).astype(np.float32)
        style = np.zeros((N, 3), np.float32)
        level_data = {0: (v1.copy(), style, v2.copy(), None, {"level": 0})}

        out = project_content_reprs(level_data, heads, "gap")[0]
        with torch.no_grad():
            want = training_reference(heads[0], torch.stack([torch.from_numpy(v1), torch.from_numpy(v2)]), False)

        torch.testing.assert_close(torch.from_numpy(out[_CONTENT]), want[0])
        torch.testing.assert_close(torch.from_numpy(out[_CONTENT_V2]), want[1])
        self.assertEqual(out[_CONTENT].shape, (N, DIM))
        np.testing.assert_array_equal(out[1], style)  # style never passes through the head
        self.assertEqual(out[4]["n_content_channels"], DIM)
        self.assertTrue(out[4]["projected_content"])

    def test_patch_projection_matches_training_in_patch_major_order(self):
        heads = load_contrastive_proj_heads(self.tmp, self.ckpt, Args(), self.model)
        hz = torch.randn(2, N, K, PATCHES)  # (V, B, k, P), as the training loop holds it
        # eval.dci._pool_and_split_view flattens to (B, P*k) patch-major.
        flat = [hz[v].permute(0, 2, 1).flatten(1).numpy() for v in range(2)]
        level_data = {0: (flat[0].copy(), None, flat[1].copy(), None, {"level": 0})}

        out = project_content_reprs(level_data, heads, (2, 2, 1))[0]
        with torch.no_grad():
            want = training_reference(heads[0], hz, True)  # (V, B, d, P)

        for view, idx in ((0, _CONTENT), (1, _CONTENT_V2)):
            torch.testing.assert_close(torch.from_numpy(out[idx]), want[view].permute(0, 2, 1).flatten(1))
            self.assertEqual(out[idx].shape, (N, PATCHES * DIM))

    def test_floor_head_is_fresh_seeded_and_reproducible(self):
        kw = dict(random_init=True)
        a = load_contrastive_proj_heads(self.tmp, self.ckpt, Args(), self.model, init_seed=3, **kw)
        b = load_contrastive_proj_heads(self.tmp, self.ckpt, Args(), self.model, init_seed=3, **kw)
        c = load_contrastive_proj_heads(self.tmp, self.ckpt, Args(), self.model, init_seed=4, **kw)

        for got, want in zip(a[0].state_dict().values(), b[0].state_dict().values()):
            torch.testing.assert_close(got, want)
        self.assertFalse(torch.allclose(a[0][0].weight, c[0][0].weight), "distinct seeds must draw distinct heads")
        self.assertFalse(
            torch.allclose(a[0][0].weight, self.trained[0].weight),
            "the floor must NOT load trained weights, or `learned` subtracts a trained head from itself",
        )

    def test_stats_pooling_is_refused_before_extraction(self):
        with self.assertRaisesRegex(ValueError, "'stats' pooling"):
            check_projectable_poolings([("gap", "gap"), ("stats", "stats")])
        check_projectable_poolings([("gap", "gap"), ("patch", (2, 2, 2))])  # no raise

    def test_non_head_modes_are_refused_with_the_right_alternative(self):
        for mode, pointer in (("entropy", "entropy_uniformity"), ("bounded", "--squash-content")):
            args = Args()
            args.contrastive_proj_mode = mode
            with self.assertRaisesRegex(ValueError, pointer):
                load_contrastive_proj_heads(self.tmp, self.ckpt, args, self.model)

    def test_no_head_configured_returns_empty_and_missing_weights_raise(self):
        args = Args()
        args.contrastive_proj_dim = 0
        self.assertEqual(load_contrastive_proj_heads(self.tmp, self.ckpt, args, self.model), {})

        headless = os.path.join(self.tmp, "headless.pt")
        torch.save({"encoders": {"encoders.0.weight": torch.randn(2, 2)}}, headless)
        with self.assertRaisesRegex(ValueError, "carries no"):
            load_contrastive_proj_heads(self.tmp, headless, Args(), self.model)

    def test_width_mismatch_names_the_widths(self):
        heads = load_contrastive_proj_heads(self.tmp, self.ckpt, Args(), self.model)
        wrong = {0: (np.zeros((N, 4 * K), np.float32), None, None, None, {"level": 0})}
        with self.assertRaisesRegex(ValueError, f"{4 * K} wide per position"):
            project_content_reprs(wrong, heads, "gap")


if __name__ == "__main__":
    unittest.main()
