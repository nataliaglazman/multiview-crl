"""CPU tests of the cross-modal reconstruction term: loss, model path, config, train_step.

Like ``test_style_hsic``, modules are AST-loaded with only unrelated imports omitted
(``lpips``/``monai``/``data.datasets``), so the implementations under test run unchanged.
"""

import ast
import logging
import math
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def _strip_imports(tree, module_names):
    """Drop top-level ``import x`` / ``from x import y`` for the named modules."""
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] in module_names for a in node.names):
            continue
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in module_names:
            continue
        keep.append(node)
    tree.body = keep
    return tree


def load_losses():
    source = ROOT / "training/losses.py"
    tree = _strip_imports(ast.parse(source.read_text()), {"lpips", "utils"})
    namespace = {"__name__": "cross_recon_losses_test", "TBSummaryTypes": types.SimpleNamespace(SCALAR="scalar")}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["cross_reconstruction_loss"], namespace["swap_views"], namespace["cross_subject_donors"]


def load_model():
    source = ROOT / "models/vqvae.py"
    tree = _strip_imports(ast.parse(source.read_text()), {"utils"})
    namespace = {"__name__": "cross_recon_model_test", "utils": types.SimpleNamespace()}
    # HelperModule comes from utils.helper, which imports nothing heavy — re-add it by hand.
    helper_src = ROOT / "utils/helper.py"
    helper_ns = {"__name__": "cross_recon_helper_test"}
    exec(compile(ast.parse(helper_src.read_text()), str(helper_src), "exec"), helper_ns)
    namespace["HelperModule"] = helper_ns["HelperModule"]
    namespace["get_parameter_count"] = helper_ns["get_parameter_count"]
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["VQVAE"]


def load_config():
    source = ROOT / "utils/config.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.Import) and any(a.name == "data.datasets" for a in node.names))
    ]
    namespace = {"datasets": types.SimpleNamespace(SyntheticBrainDataset=object, MyCustomDataset=object)}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["parse_args"], namespace["update_args"]


cross_reconstruction_loss, swap_views, cross_subject_donors = load_losses()


class CrossSubjectDonorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def assert_other_subject(self, donors, subjects):
        for i, j in enumerate(donors.tolist()):
            self.assertNotEqual(subjects[i], subjects[j], f"row {i} borrowed from row {j}, the same subject")

    def test_distinct_subjects_get_a_derangement(self):
        for n in range(2, 9):
            for _ in range(20):
                donors = cross_subject_donors(range(n))
                self.assertEqual(sorted(donors.tolist()), list(range(n)))  # every style lent once
                self.assert_other_subject(donors, list(range(n)))

    def test_every_other_subject_is_an_equally_likely_donor(self):
        counts = torch.zeros(4, 4)
        for _ in range(3000):
            counts[torch.arange(4), cross_subject_donors(range(4))] += 1
        self.assertTrue(torch.equal(counts.diag(), torch.zeros(4)))
        off = counts[~torch.eye(4, dtype=torch.bool)] / 3000
        self.assertLess((off - 1 / 3).abs().max().item(), 0.05)

    def test_rows_of_a_repeated_subject_never_lend_to_each_other(self):
        subjects = ["a", "b", "a", "c", "b", "d"]
        for _ in range(50):
            donors = cross_subject_donors(subjects)
            self.assertEqual(sorted(donors.tolist()), list(range(6)))
            self.assert_other_subject(donors, subjects)

    def test_a_majority_subject_still_gets_other_subjects_style(self):
        # Three rows of 'a' cannot take three distinct non-'a' donors from one 'b' row, so
        # there is no permutation; each row must still borrow from another subject.
        subjects = ["a", "a", "b", "a"]
        for _ in range(20):
            donors = cross_subject_donors(subjects)
            self.assertEqual(donors.tolist()[:2] + donors.tolist()[3:], [2, 2, 2])
            self.assert_other_subject(donors, subjects)

    def test_tensor_ids_are_compared_by_value(self):
        # 0-d tensors hash by identity; read naively, every row would count as its own subject.
        subjects = torch.tensor([5, 5, 7, 7])
        for _ in range(20):
            self.assert_other_subject(cross_subject_donors(subjects), subjects.tolist())

    def test_a_batch_of_one_subject_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "two subjects"):
            cross_subject_donors(["a", "a"])
        with self.assertRaisesRegex(ValueError, "two subjects"):
            cross_subject_donors(range(1))


class CrossReconstructionLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)

    def test_target_is_the_other_view(self):
        # The one mistake this loss cannot survive: scoring against the view the content
        # came from. A decode that already equals the swapped batch must cost nothing.
        images = torch.randn(6, 1, 4, 4, 4)
        zero, _ = cross_reconstruction_loss(swap_views(images), images)
        self.assertAlmostEqual(zero.item(), 0.0, places=6)
        # ...and the unswapped batch — what a same-view target would reward — must not.
        same_view, _ = cross_reconstruction_loss(images.clone(), images)
        self.assertGreater(same_view.item(), 0.1)

    def test_halves_are_reported_against_opposite_views(self):
        v0 = torch.zeros(2, 1, 4, 4, 4)
        v1 = torch.ones(2, 1, 4, 4, 4)
        images = torch.cat([v0, v1])
        # First half decoded with style_v1 renders view 1 perfectly; second half is wrong by 1.
        decode = torch.cat([v1, v1])
        loss, diag = cross_reconstruction_loss(decode, images)
        self.assertAlmostEqual(diag["CrossRecon/mae_to_view1"], 0.0, places=6)
        self.assertAlmostEqual(diag["CrossRecon/mae_to_view0"], 1.0, places=6)
        self.assertAlmostEqual(diag["CrossRecon/mae"], 0.5, places=6)
        self.assertAlmostEqual(loss.item(), 0.5, places=6)

    def test_mask_is_swapped_with_the_target_and_excludes_background(self):
        images = torch.randn(4, 1, 4, 4, 4)
        decode = swap_views(images).clone()
        mask = torch.zeros(4, 1, 4, 4, 4)
        mask[:2, :, 0] = 1.0  # view 0's brain
        mask[2:, :, 1] = 1.0  # view 1's brain, a different slab
        # Corrupt only voxels that the SWAPPED mask excludes: the loss must not see them.
        decode[:2, :, 0] += 5.0  # row in half 0, but its target/mask is view 1's slab 1
        outside, _ = cross_reconstruction_loss(decode, images, mask=mask)
        self.assertAlmostEqual(outside.item(), 0.0, places=6)
        # Corrupting inside the swapped mask does register.
        decode[:2, :, 1] += 5.0
        inside, _ = cross_reconstruction_loss(decode, images, mask=mask)
        self.assertGreater(inside.item(), 1.0)

    def test_gradients_flow_and_nonfinite_decodes_are_survivable(self):
        decode = torch.randn(4, 1, 4, 4, 4, requires_grad=True)
        images = torch.randn(4, 1, 4, 4, 4)
        loss, _ = cross_reconstruction_loss(decode, images)
        loss.backward()
        self.assertGreater(decode.grad.norm().item(), 0)

        bad = torch.full((4, 1, 4, 4, 4), float("nan"), requires_grad=True)
        loss, diag = cross_reconstruction_loss(bad, images)
        self.assertTrue(math.isfinite(loss.item()))
        self.assertTrue(math.isfinite(diag["CrossRecon/mae"]))

    def test_shape_and_parity_are_enforced(self):
        images = torch.randn(4, 1, 4, 4, 4)
        with self.assertRaisesRegex(ValueError, "must match"):
            cross_reconstruction_loss(torch.randn(4, 1, 2, 2, 2), images)
        with self.assertRaisesRegex(ValueError, "even"):
            cross_reconstruction_loss(torch.randn(3, 1, 4, 4, 4), torch.randn(3, 1, 4, 4, 4))

    def test_autocast_returns_float32(self):
        decode = torch.randn(4, 1, 4, 4, 4, dtype=torch.bfloat16)
        images = torch.randn(4, 1, 4, 4, 4, dtype=torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, _ = cross_reconstruction_loss(decode, images)
        self.assertEqual(loss.dtype, torch.float32)


class CrossReconstructionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.Model = load_model()

    def setUp(self):
        torch.manual_seed(11)

    def model(self, **kwargs):
        defaults = dict(
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
            # A LayerNorm over the decoder's single output channel is identically zero, so a
            # layer-normed decoder would make every one of these assertions pass vacuously.
            decoder_norm_type="group",
        )
        defaults.update(kwargs)
        return self.Model(**defaults)

    def test_eight_tuple_is_unchanged_and_extras_follow_it_in_order(self):
        model = self.model().eval()
        x = torch.randn(4, 1, 8, 8, 8)
        plain = model(x, n_views=2, pool_only=True)
        self.assertEqual(len(plain), 8)

        outputs, cross = model(x, n_views=2, pool_only=True, cross_recon=True)
        self.assertEqual(len(outputs), 8)
        self.assertTrue(torch.equal(plain[0], outputs[0]))
        self.assertEqual(cross.shape, plain[0].shape)

        outputs, features, cross = model(x, n_views=2, pool_only=True, cross_recon=True, return_style_features=True)
        self.assertEqual(len(outputs), 8)
        self.assertIn(0, features)
        self.assertEqual(cross.shape, plain[0].shape)

    def test_identical_views_make_the_swap_a_no_op(self):
        # Style is the only thing exchanged, so when both halves carry the same volume the
        # cross decode must reproduce the normal one exactly. This is what pins down that
        # the swap is over the [v0; v1] batch axis and nothing else moved with it.
        model = self.model().eval()
        half = torch.randn(3, 1, 8, 8, 8)
        outputs, cross = model(torch.cat([half, half]), n_views=2, pool_only=True, cross_recon=True)
        self.assertTrue(torch.allclose(outputs[0], cross, atol=1e-6))

    def test_differing_views_change_the_decode_and_reach_the_encoder(self):
        model = self.model().eval()
        x = torch.cat([torch.randn(3, 1, 8, 8, 8), torch.randn(3, 1, 8, 8, 8) * 4 + 2])
        outputs, cross = model(x, n_views=2, pool_only=True, cross_recon=True)
        self.assertFalse(torch.allclose(outputs[0], cross, atol=1e-4))
        loss, _ = cross_reconstruction_loss(cross, x)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.norm() > 0 for p in model.encoders.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.norm() > 0 for p in model.decoders.parameters()))

    def test_quantized_style_is_swapped_after_quantization(self):
        model = self.model(quantize_style=True, style_spatial_size=2).eval()
        x = torch.cat([torch.randn(3, 1, 8, 8, 8), torch.randn(3, 1, 8, 8, 8) * 4 + 2])
        outputs, cross = model(x, n_views=2, pool_only=True, cross_recon=True)
        self.assertFalse(torch.allclose(outputs[0], cross, atol=1e-4))

    def test_style_dropout_does_not_touch_the_cross_decode(self):
        # Dropout zeroes style so the decoder learns to lean on content; applying it to the
        # cross decode would ask content alone to render the OTHER modality, i.e. teach the
        # exact leak this loss exists to punish. With p=1 the normal decode loses style
        # entirely while the cross decode must still differ between the two halves.
        model = self.model(style_dropout_prob=0.999).train()
        half_a, half_b = torch.randn(3, 1, 8, 8, 8), torch.randn(3, 1, 8, 8, 8) * 4 + 2
        outputs, cross = model(torch.cat([half_a, half_b]), n_views=2, pool_only=True, cross_recon=True)
        self.assertFalse(torch.allclose(outputs[0], cross, atol=1e-4))

    def test_style_index_decodes_each_row_with_its_donors_style(self):
        # Row i keeps its content and takes the other view's style from row index[i]: the same
        # decode as a same-subject swap on a batch whose other view was reordered to line the
        # donors up. Rows are independent here (per-sample norms, eval mode), so reordering one
        # view cannot move the other view's content.
        model = self.model().eval()
        v0, v1 = torch.randn(4, 1, 8, 8, 8), torch.randn(4, 1, 8, 8, 8) * 3 + 1
        index = torch.tensor([2, 0, 3, 1])
        _, cross = model(torch.cat([v0, v1]), n_views=2, pool_only=True, cross_recon=True, cross_style_index=index)
        _, first = model(torch.cat([v0, v1[index]]), n_views=2, pool_only=True, cross_recon=True)
        _, second = model(torch.cat([v0[index], v1]), n_views=2, pool_only=True, cross_recon=True)
        self.assertTrue(torch.allclose(cross[:4], first[:4], atol=1e-6))
        self.assertTrue(torch.allclose(cross[4:], second[4:], atol=1e-6))

        _, own = model(torch.cat([v0, v1]), n_views=2, pool_only=True, cross_recon=True)
        self.assertFalse(torch.allclose(cross, own, atol=1e-4))
        _, identity = model(
            torch.cat([v0, v1]), n_views=2, pool_only=True, cross_recon=True, cross_style_index=torch.arange(4)
        )
        self.assertTrue(torch.allclose(identity, own, atol=1e-6))

    def test_a_rows_own_other_view_does_not_reach_its_cross_decode(self):
        # What borrowing a stranger's style is for: the subject's own other view must have no
        # path into its cross decode, so it cannot smuggle that subject's anatomy in as style.
        model = self.model().eval()
        x = torch.cat([torch.randn(4, 1, 8, 8, 8), torch.randn(4, 1, 8, 8, 8) * 3 + 1]).requires_grad_()
        _, cross = model(x, n_views=2, pool_only=True, cross_recon=True, cross_style_index=torch.tensor([2, 0, 3, 1]))
        cross[0].abs().sum().backward()
        reach = x.grad.flatten(1).abs().sum(1)
        self.assertGreater(reach[0].item(), 0)  # row 0's content (subject 0, view 0)
        self.assertGreater(reach[4 + 2].item(), 0)  # the donor's style (subject 2, view 1)
        self.assertEqual(reach[4 + 0].item(), 0.0)  # subject 0's own view 1
        self.assertEqual(reach[[1, 2, 3, 5, 7]].sum().item(), 0.0)

    def test_preconditions_are_rejected_with_an_explanation(self):
        model = self.model().eval()
        x = torch.randn(4, 1, 8, 8, 8)
        with self.assertRaisesRegex(ValueError, "return_recon"):
            model(x, n_views=2, pool_only=True, return_recon=False, cross_recon=True)
        with self.assertRaisesRegex(ValueError, "n_views"):
            model(x, n_views=1, pool_only=True, cross_recon=True)
        with self.assertRaisesRegex(ValueError, "even batch"):
            model(torch.randn(3, 1, 8, 8, 8), n_views=2, pool_only=True, cross_recon=True)
        with self.assertRaisesRegex(ValueError, "one donor row per subject"):
            model(x, n_views=2, pool_only=True, cross_recon=True, cross_style_index=torch.tensor([1, 0, 0]))
        with self.assertRaisesRegex(ValueError, "only applies with cross_recon"):
            model(x, n_views=2, pool_only=True, cross_style_index=torch.tensor([1, 0]))

        no_style = self.model(inject_style_to_decoder=False).eval()
        with self.assertRaisesRegex(ValueError, "inject_style_to_decoder"):
            no_style(x, n_views=2, pool_only=True, cross_recon=True)

        deep = self.model(nb_levels=2, scaling_rates=[2, 2], content_style_levels=[1], content_ratios=[0.75]).eval()
        with self.assertRaisesRegex(ValueError, "content_style_levels"):
            deep(x, n_views=2, pool_only=True, cross_recon=True)

    def test_the_objective_is_reachable_on_a_two_modality_pair(self):
        # End-to-end: the term is only worth anything if a model can actually satisfy it.
        # Two 'modalities' of ONE anatomy, related by an affine intensity map (T1 vs T2 in
        # miniature). If the plumbing is right, optimising it leaves the model rendering
        # whichever modality the style says from a single shared content code — which is
        # exactly "cross decode close to the OTHER view, far from its own".
        torch.manual_seed(0)
        model = self.model(
            hidden_channels=16,
            res_channels=8,
            nb_entries=32,
            content_size=12,
            style_size=4,
            final_recon_norm=False,
        ).train()
        anatomy = torch.randn(4, 1, 8, 8, 8)
        x = torch.cat([anatomy, -0.7 * anatomy + 1.5])
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        first = last = None
        for _ in range(250):
            opt.zero_grad()
            (recon, diffs, *_), cross = model(x, n_views=2, pool_only=True, cross_recon=True)
            cross_loss, _ = cross_reconstruction_loss(cross, x)
            loss = (recon - x).abs().mean() + cross_loss + 0.25 * sum(d.mean() for d in diffs)
            loss.backward()
            opt.step()
            last = cross_loss.item()
            first = first if first is not None else last
        self.assertLess(last, first / 4)

        model.eval()
        with torch.no_grad():
            _, cross = model(x, n_views=2, pool_only=True, cross_recon=True)
        to_other = (cross - swap_views(x)).abs().mean().item()
        to_own = (cross - x).abs().mean().item()
        self.assertLess(to_other, to_own / 5)


class CrossReconstructionConfigTests(unittest.TestCase):
    def test_off_by_default_and_preconditions_checked_at_config_time(self):
        parse, update = load_config()
        args = parse().parse_args([])
        self.assertEqual(args.scale_cross_recon_loss, 0.0)
        self.assertEqual(args.cross_recon_start_step, 0)
        self.assertEqual(args.cross_recon_style_source, "other_subject")
        update(parse().parse_args(["--scale-cross-recon-loss", "0"]))  # disabled: no checks
        update(parse().parse_args(["--scale-cross-recon-loss", "0", "--batch-size", "1"]))
        on = ["--scale-cross-recon-loss", "1", "--inject-style-to-decoder", "--batch-size", "1"]
        update(parse().parse_args(on + ["--cross-recon-style-source", "same_subject"]))

        for flags, message in [
            (["--scale-cross-recon-loss", "1"], "inject-style-to-decoder"),
            (
                ["--scale-cross-recon-loss", "1", "--inject-style-to-decoder", "--content-style-levels", "1"],
                "level 0",
            ),
            (
                ["--scale-cross-recon-loss", "1", "--inject-style-to-decoder", "--contrastive-only"],
                "contrastive-only",
            ),
            (on, "batch-size >= 2"),
            (
                [
                    "--scale-cross-recon-loss",
                    "1",
                    "--inject-style-to-decoder",
                    "--content-dim",
                    "512",
                    "--total-dim",
                    "512",
                ],
                "content/style split",
            ),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                update(parse().parse_args(flags))


class CrossReconstructionTrainStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.Model = load_model()

    def setUp(self):
        torch.manual_seed(5)

    def train_step(self, **overrides):
        source = ROOT / "training/main_multimodal.py"
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_step")
        namespace = dict(
            torch=torch,
            F=F,
            math=math,
            autocast=torch.autocast,
            clip_grad_norm_=torch.nn.utils.clip_grad_norm_,
            logger=logging.getLogger(__name__),
            cross_reconstruction_loss=cross_reconstruction_loss,
            cross_subject_donors=cross_subject_donors,
            NAN_SKIPPED_STEPS=0,
        )
        namespace.update(overrides)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        return namespace["train_step"]

    def test_weighting_logging_and_backward(self):
        train_step = self.train_step()
        parse, update = load_config()
        args = update(
            parse().parse_args(
                [
                    "--dataset-name",
                    "synthetic",
                    "--inject-style-to-decoder",
                    "--batch-size",
                    "4",
                    "--scale-contrastive-loss",
                    "0",
                ]
            )
        )
        model = self.Model(
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
            decoder_norm_type="group",
        ).eval()
        data = {"image": [torch.randn(4, 1, 8, 8, 8), torch.randn(4, 1, 8, 8, 8) * 3 + 1]}

        def run(optimizer=None):
            return train_step(
                data,
                [model],
                [],
                lambda h, *a, **kw: h.sum() * 0,
                optimizer,
                list(model.parameters()),
                args,
                recon_loss_fn=lambda out, target: (out["reconstruction"][0] - target).square().mean(),
                force_compute_recon=True,
            )

        baseline = run()
        self.assertNotIn("CrossRecon/mae", baseline[-1])

        args.scale_cross_recon_loss = 2.0
        weighted = run()
        diag = weighted[-1]
        self.assertAlmostEqual(diag["CrossRecon/weighted"], 2 * diag["CrossRecon/mae"], places=5)
        self.assertAlmostEqual(weighted[0] - baseline[0], diag["CrossRecon/weighted"], places=4)
        # The other reported losses are untouched by the new term.
        self.assertEqual(weighted[1:4], baseline[1:4])

        before = [p.detach().clone() for p in model.encoders.parameters()]
        run(torch.optim.SGD(model.parameters(), lr=0.1))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.encoders.parameters())))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_start_step_gates_the_term_and_resume_skips_the_warmup(self):
        train_step = self.train_step()
        parse, update = load_config()
        args = update(
            parse().parse_args(
                [
                    "--dataset-name",
                    "synthetic",
                    "--inject-style-to-decoder",
                    "--batch-size",
                    "4",
                    "--scale-contrastive-loss",
                    "0",
                    "--scale-cross-recon-loss",
                    "1",
                    "--cross-recon-start-step",
                    "100",
                ]
            )
        )
        model = self.Model(
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
            decoder_norm_type="group",
        ).eval()
        data = {"image": [torch.randn(4, 1, 8, 8, 8), torch.randn(4, 1, 8, 8, 8) * 3 + 1]}

        def run(step, **kwargs):
            return train_step(
                data,
                [model],
                [],
                lambda h, *a, **kw: h.sum() * 0,
                None,
                list(model.parameters()),
                args,
                recon_loss_fn=lambda out, target: (out["reconstruction"][0] - target).square().mean(),
                force_compute_recon=True,
                step=step,
                **kwargs,
            )

        self.assertNotIn("CrossRecon/mae", run(99)[-1])
        self.assertIn("CrossRecon/mae", run(100)[-1])
        args._resumed_past_cross_recon_start = True
        self.assertIn("CrossRecon/mae", run(1)[-1])

    def test_skipped_reconstruction_skips_the_cross_term(self):
        # It rides on the decoder pass, so a skip_recon_ratio step must not try to swap a
        # style tensor that was never produced.
        train_step = self.train_step()
        parse, update = load_config()
        args = update(
            parse().parse_args(
                [
                    "--dataset-name",
                    "synthetic",
                    "--inject-style-to-decoder",
                    "--batch-size",
                    "4",
                    "--scale-contrastive-loss",
                    "0",
                    "--scale-cross-recon-loss",
                    "1",
                ]
            )
        )
        model = self.Model(
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
            decoder_norm_type="group",
        ).eval()
        data = {"image": [torch.randn(4, 1, 8, 8, 8), torch.randn(4, 1, 8, 8, 8)]}
        result = train_step(
            data,
            [model],
            [],
            lambda h, *a, **kw: h.sum() * 0,
            None,
            list(model.parameters()),
            args,
            recon_loss_fn=lambda out, target: (out["reconstruction"][0] - target).square().mean(),
            force_compute_recon=False,
        )
        self.assertNotIn("CrossRecon/mae", result[-1])

    def test_other_subject_donors_are_drawn_from_batch_subjects_and_reach_the_decode(self):
        calls, chosen = [], {}

        def donors(subjects):
            calls.append(list(subjects))
            return chosen["index"]

        train_step = self.train_step(cross_subject_donors=donors)
        parse, update = load_config()
        args = update(
            parse().parse_args(
                [
                    "--dataset-name",
                    "synthetic",
                    "--inject-style-to-decoder",
                    "--batch-size",
                    "4",
                    "--scale-contrastive-loss",
                    "0",
                    "--scale-cross-recon-loss",
                    "1",
                ]
            )
        )
        model = self.Model(
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
            decoder_norm_type="group",
        ).eval()
        subjects = ["s0", "s1", "s0", "s2"]
        data = {"image": [torch.randn(4, 1, 8, 8, 8), torch.randn(4, 1, 8, 8, 8) * 3 + 1], "subject": subjects}

        def cross_mae():
            return train_step(
                data,
                [model],
                [],
                lambda h, *a, **kw: h.sum() * 0,
                None,
                list(model.parameters()),
                args,
                recon_loss_fn=lambda out, target: (out["reconstruction"][0] - target).square().mean(),
                force_compute_recon=True,
            )[-1]["CrossRecon/mae"]

        chosen["index"] = torch.arange(4)
        identity = cross_mae()
        chosen["index"] = torch.tensor([1, 2, 3, 0])
        shifted = cross_mae()
        self.assertEqual(calls, [subjects, subjects])

        args.cross_recon_style_source = "same_subject"
        own = cross_mae()
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(identity, own, places=6)
        self.assertNotAlmostEqual(shifted, own, places=4)

        # No IDs in the batch (synthetic): every row is its own subject.
        args.cross_recon_style_source = "other_subject"
        del data["subject"]
        cross_mae()
        self.assertEqual(calls[-1], [0, 1, 2, 3])

        # settings.json from before the flag: those runs were trained on the same-subject swap.
        del args.cross_recon_style_source
        cross_mae()
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
