"""CPU tests of the full-objective gradient audit and the shared Barlow Twins arms.

``training/losses.py`` and ``train_step`` are AST-loaded with only the lpips/utils imports
omitted and a small differentiable stand-in for LPIPS, so the implementations under test run
unchanged. Run by path: ``PYTHONPATH=. python tests/test_loss_gradient_audit.py``.
"""

import ast
import contextlib
import importlib.util
import io
import logging
import math
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from eval import gradient_attribution as ga
from eval import loss_gradient_audit as audit
from eval.reconstruction_attribution import frozen_checkpoint
from eval.ventricle_routing import state_digest
from training.bt_objective import make_barlow_loss_functions
from training.style_alignment import within_modality_style_loss
from training.style_hsic import style_content_hsic_loss

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("loss_audit_fixtures", ROOT / "tests/test_style_hsic.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class StandInLPIPS(torch.nn.Module):
    """Differentiable stand-in with LPIPS's call signature and (N, 1, 1, 1) output."""

    def __init__(self, **_kwargs):
        super().__init__()

    def forward(self, x, y):
        return (x - y).abs().mean(dim=(1, 2, 3), keepdim=True)


def load_losses():
    source = ROOT / "training/losses.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in {"lpips", "utils"})
    ]
    namespace = {
        "__name__": "loss_audit_losses",
        "LPIPS": StandInLPIPS,
        "TBSummaryTypes": types.SimpleNamespace(SCALAR="scalar"),
    }
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace


def load_train_step(losses):
    source = ROOT / "training/main_multimodal.py"
    function = next(
        n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "train_step"
    )
    namespace = dict(
        torch=torch,
        F=F,
        math=math,
        autocast=torch.autocast,
        clip_grad_norm_=torch.nn.utils.clip_grad_norm_,
        logger=logging.getLogger(__name__),
        NAN_SKIPPED_STEPS=0,
        BaselineLoss=losses["BaselineLoss"],
        style_infonce_loss=losses["style_infonce_loss"],
        cross_reconstruction_loss=losses["cross_reconstruction_loss"],
        style_content_hsic_loss=style_content_hsic_loss,
        within_modality_style_loss=within_modality_style_loss,
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["train_step"]


LOSSES = load_losses()
BT, STATS, BASELINE = LOSSES["barlow_twins_loss"], LOSSES["stats_pool"], LOSSES["BaselineLoss"]
TRAIN_STEP = load_train_step(LOSSES)
MODEL = fixtures.load_model()


def legacy_losses(args):
    """main_multimodal's Barlow Twins closures before training/bt_objective.py (git f3fba27)."""
    _nce_center = getattr(args, "patch_center_mode", "none")
    _bt_lambda = getattr(args, "bt_lambda", 0.005)
    _bt_stat = getattr(args, "bt_patch_stat", "fold")
    _bt_gap_w = getattr(args, "bt_gap_weight", 0.0)
    _bt_corr_ema = float(getattr(args, "bt_corr_ema", 0.0) or 0.0)
    _bt_ema_patch, _bt_ema_gap, _bt_ema_plain = {}, {}, {}
    _bt_gap_lam = getattr(args, "bt_gap_lambda", None)
    _bt_gap_lam = _bt_lambda if _bt_gap_lam is None else _bt_gap_lam
    _bt_sim_c = getattr(args, "bt_sim_coeff", 0.0)
    _bt_std_c = getattr(args, "bt_std_coeff", 0.0)
    _bt_gap_std_c = getattr(args, "bt_gap_std_coeff", None)
    _bt_gap_std_c = _bt_std_c if _bt_gap_std_c is None else _bt_gap_std_c
    _bt_gap_sim_c = getattr(args, "bt_gap_sim_coeff", None)
    _bt_gap_sim_c = _bt_sim_c if _bt_gap_sim_c is None else _bt_gap_sim_c
    _bt_sim_norm = getattr(args, "bt_sim_normalize", False)
    _bt_gap_pool = getattr(args, "bt_gap_pooling", "gap")
    _bt_whiten = getattr(args, "bt_sim_whiten", False)
    _bt_whiten_eps = getattr(args, "bt_sim_whiten_eps", 1e-3)
    _bt_patch_w = getattr(args, "bt_patch_weight", 1.0)
    _bt_norm = bool(getattr(args, "bt_normalize_terms", False))

    def loss_func(z_rec_tuple, estimated_content_indices, subsets, soft_content_mask=None):
        return BT(
            z_rec_tuple,
            estimated_content_indices=estimated_content_indices,
            subsets=subsets,
            soft_content_mask=soft_content_mask,
            lambd=_bt_lambda,
            sim_coeff=_bt_sim_c,
            std_coeff=_bt_std_c,
            sim_normalize=_bt_sim_norm,
            sim_whiten=_bt_whiten,
            sim_whiten_eps=_bt_whiten_eps,
            corr_ema=_bt_ema_plain,
            corr_ema_decay=_bt_corr_ema,
            normalize_terms=_bt_norm,
        )

    def patch_loss_func(z_rec_tuple, estimated_content_indices, subsets, soft_content_mask=None):
        _l = BT(
            z_rec_tuple,
            estimated_content_indices=estimated_content_indices,
            subsets=subsets,
            soft_content_mask=soft_content_mask,
            lambd=_bt_lambda,
            center_mode=_nce_center,
            patch_stat=_bt_stat,
            sim_coeff=_bt_sim_c,
            std_coeff=_bt_std_c,
            sim_normalize=_bt_sim_norm,
            corr_ema=_bt_ema_patch,
            corr_ema_decay=_bt_corr_ema,
            normalize_terms=_bt_norm,
        )
        if _bt_gap_w > 0 and z_rec_tuple.ndim == 4:
            if _bt_gap_pool == "stats":
                _gz, _gidx, _gmask = STATS(z_rec_tuple, estimated_content_indices, soft_content_mask)
            else:
                _gz, _gidx, _gmask = z_rec_tuple.mean(-1), estimated_content_indices, soft_content_mask
            _lg = BT(
                _gz,
                estimated_content_indices=_gidx,
                subsets=subsets,
                soft_content_mask=_gmask,
                lambd=_bt_gap_lam,
                sim_coeff=_bt_gap_sim_c,
                std_coeff=_bt_gap_std_c,
                sim_normalize=_bt_sim_norm,
                sim_whiten=_bt_whiten,
                sim_whiten_eps=_bt_whiten_eps,
                corr_ema=_bt_ema_gap,
                corr_ema_decay=_bt_corr_ema,
                normalize_terms=_bt_norm,
            )
            _total = _bt_patch_w * _l + _bt_gap_w * _lg
            _d = dict(getattr(_l, "_contrastive_diag", None) or {})
            for _k, _v in (getattr(_lg, "_contrastive_diag", None) or {}).items():
                _d[f"gap_{_k}"] = _v
            _total._contrastive_diag = _d
            return _total
        if _bt_patch_w == 1.0:
            return _l
        _scaled = _bt_patch_w * _l
        _scaled._contrastive_diag = dict(getattr(_l, "_contrastive_diag", None) or {})
        return _scaled

    return loss_func, patch_loss_func


def config(*extra):
    parse, update = fixtures.load_config()
    base = (
        "--dataset-name synthetic --mask-mode fixed --inject-style-to-decoder --vqvae-nb-levels 1 "
        "--content-dim 6 --total-dim 8 --batch-size 6 --contrastive-loss-type barlow_twins "
        "--patch-contrastive --patch-grid 2 2 2 --bt-gap-weight 1.5 --bt-patch-weight 0.5 --bt-lambda 2 "
        "--bt-corr-ema 0.9 --bt-sim-coeff 0.1 --bt-std-coeff 0.2 --scale-recon-loss 2 "
        "--scale-contrastive-loss 3 --scale-style-hsic-loss 0.5"
    ).split()
    return update(parse().parse_args([*base, *extra]))


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return MODEL(
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
        norm_type="layer",
        # The legacy output norm makes an untrained decoder's output exactly constant, so
        # the reconstruction terms would carry no encoder gradient to attribute.
        final_recon_norm=False,
    ).eval()


def batch(seed, b=6):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(b, 1, 8, 8, 8, generator=g)
    x1 = 0.5 * x0 + torch.randn(b, 1, 8, 8, 8, generator=g)
    mask = (torch.rand(b, 1, 8, 8, 8, generator=g) > 0.2).float()
    return {"image": [x0, x1], "mask": [mask, mask], "gt_latents": {"z_content": torch.randn(b, 3, generator=g)}}


def objective(model, args, mode="reference", train_step_impl=TRAIN_STEP):
    return audit.TrainingObjective(
        model,
        args,
        10,
        ema_mode=mode,
        train_step_impl=train_step_impl,
        recon_loss=BASELINE(),
        loss_impl=BT,
        stats_impl=STATS,
    )


class BarlowArmTests(unittest.TestCase):
    def test_arms_forward_the_training_settings(self):
        args = types.SimpleNamespace(
            bt_lambda=2.0,
            bt_gap_lambda=None,
            bt_sim_coeff=0.1,
            bt_gap_sim_coeff=0.3,
            bt_std_coeff=0.2,
            bt_gap_std_coeff=None,
            bt_sim_normalize=True,
            bt_sim_whiten=True,
            bt_sim_whiten_eps=0.01,
            bt_corr_ema=0.9,
            bt_normalize_terms=True,
            patch_center_mode="position",
            bt_patch_stat="per_position",
            bt_patch_weight=0.5,
            bt_gap_weight=1.5,
            bt_gap_pooling="gap",
        )
        calls = []

        def spy(hz, **kwargs):
            calls.append((hz, kwargs))
            return hz.sum() * 0

        states = {}
        plain, patch_fn = make_barlow_loss_functions(args, states=states, loss_impl=spy, stats_impl=STATS)
        hz = torch.randn(2, 4, 3, 5)
        for _ in range(2):
            patch_fn(hz, [[0, 1, 2]], [(0, 1)])
        (p_hz, p), (g_hz, g) = calls[:2]
        self.assertIs(p_hz, hz)
        torch.testing.assert_close(g_hz, hz.mean(-1))
        self.assertEqual((p["center_mode"], p["patch_stat"], p["sim_whiten"]), ("position", "per_position", False))
        self.assertEqual((g["center_mode"], g["sim_whiten"], g["sim_whiten_eps"]), ("none", True, 0.01))
        self.assertEqual((p["lambd"], p["sim_coeff"], p["std_coeff"]), (2.0, 0.1, 0.2))
        # A None GAP override falls back to the patch value; a set one wins.
        self.assertEqual((g["lambd"], g["sim_coeff"], g["std_coeff"]), (2.0, 0.3, 0.2))
        for kw in (p, g):
            self.assertEqual((kw["sim_normalize"], kw["corr_ema_decay"], kw["normalize_terms"]), (True, 0.9, True))
        # One EMA dict per arm, persistent across calls, never shared.
        self.assertIs(p["corr_ema"], states["patch"])
        self.assertIs(g["corr_ema"], states["gap"])
        self.assertIs(calls[2][1]["corr_ema"], states["patch"])
        plain(hz.mean(-1), [[0, 1, 2]], [(0, 1)])
        self.assertIs(calls[-1][1]["corr_ema"], states["global"])
        self.assertTrue(calls[-1][1]["sim_whiten"])

    def test_matches_the_pre_refactor_training_closures_bit_for_bit(self):
        variants = [
            ("--patch-center-mode", "position", "--bt-gap-weight", "0", "--bt-patch-weight", "1"),
            ("--bt-gap-lambda", "5", "--bt-gap-std-coeff", "0.4", "--bt-normalize-terms"),
            ("--bt-gap-pooling", "stats", "--bt-sim-normalize", "--bt-patch-stat", "per_position"),
            ("--bt-gap-weight", "0", "--bt-patch-weight", "0.25"),
        ]
        for extra in variants:
            args = config(*extra)
            legacy = legacy_losses(args)
            shared = make_barlow_loss_functions(args, loss_impl=BT, stats_impl=STATS)
            for step in range(3):  # Several steps, so the correlation EMA state is exercised.
                g = torch.Generator().manual_seed(step)
                for arm, shape in ((1, (2, 16, 6, 8)), (0, (2, 16, 6))):
                    data = torch.randn(*shape, generator=g)
                    outputs = []
                    for functions in (legacy, shared):
                        hz = data.clone().requires_grad_()
                        loss = functions[arm](hz, [list(range(6))], [(0, 1)])
                        (grad,) = torch.autograd.grad(loss.sum(), hz)
                        outputs.append((loss.detach(), grad, loss._contrastive_diag))
                    (old, old_grad, old_diag), (new, new_grad, new_diag) = outputs
                    with self.subTest(extra=extra, step=step, arm=arm):
                        self.assertTrue(torch.equal(old, new))
                        self.assertTrue(torch.equal(old_grad, new_grad))
                        self.assertEqual(old_diag, new_diag)

    def test_captured_components_partition_the_combined_loss(self):
        args = config()
        correlations = {}
        _, patch_fn = make_barlow_loss_functions(
            args, correlations=correlations, capture_components=True, loss_impl=BT, stats_impl=STATS
        )
        hz = torch.randn(2, 16, 6, 8, requires_grad=True)
        loss = patch_fn(hz, [list(range(6))], [(0, 1)])
        parts = loss._loss_components
        self.assertEqual({k.split("/")[0] for k in parts}, {"patch", "gap"})
        self.assertEqual({k.split("/")[1] for k in parts}, {"on_diag", "off_diag", "sim", "variance"})
        torch.testing.assert_close(sum(parts.values()).reshape(1), loss)
        # Captured correlations are the instantaneous, pre-EMA matrices.
        c = correlations["gap"][(0, 0, 1)]["c"]
        inst = float(c.square().sum() - c.diagonal().square().sum())
        self.assertAlmostEqual(inst, loss._contrastive_diag["gap_off_diag_inst"], places=4)

        _, bare = make_barlow_loss_functions(
            config("--bt-sim-coeff", "0", "--bt-std-coeff", "0"),
            capture_components=True,
            loss_impl=BT,
            stats_impl=STATS,
        )
        keys = bare(hz, [list(range(6))], [(0, 1)])._loss_components
        self.assertEqual(set(keys), {"patch/on_diag", "patch/off_diag", "gap/on_diag", "gap/off_diag"})


class TrainStepObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def run_step(self, model, args, data, observer=None, optimizer=None):
        plain, patch_fn = make_barlow_loss_functions(
            args, loss_impl=BT, stats_impl=STATS, capture_components=observer is not None
        )
        return TRAIN_STEP(
            data,
            [model],
            [],
            plain,
            optimizer,
            list(model.parameters()),
            args,
            recon_loss_fn=BASELINE(),
            patch_loss_func=patch_fn,
            step=10,
            force_compute_recon=True,
            loss_observer=observer,
        )

    def test_observer_leaves_the_training_step_unchanged(self):
        args, data = config(), batch(1)
        results, weights = [], []
        for observe in (False, True):
            model = tiny_model()
            torch.manual_seed(0)
            out = self.run_step(
                model,
                args,
                data,
                observer=(lambda *a: None) if observe else None,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            )
            results.append(out[:4])
            weights.append([p.detach().clone() for p in model.parameters()])
        self.assertEqual(results[0], results[1])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(*weights)))

    def test_components_are_weighted_and_partition_value_and_gradient(self):
        args, model, captured = config(), tiny_model(), {}
        torch.manual_seed(0)
        recon = BASELINE()
        plain, patch_fn = make_barlow_loss_functions(args, loss_impl=BT, stats_impl=STATS, capture_components=True)
        out = TRAIN_STEP(
            batch(2),
            [model],
            [],
            plain,
            None,
            list(model.parameters()),
            args,
            recon_loss_fn=recon,
            patch_loss_func=patch_fn,
            step=10,
            force_compute_recon=True,
            loss_observer=lambda total, parts, diag: captured.update(total=total, parts=parts, diag=diag),
        )
        parts, diag = captured["parts"], captured["diag"]
        arms = [f"content/L0/{a}/{t}" for a in ("patch", "gap") for t in ("on_diag", "off_diag", "sim", "variance")]
        expected = {"reconstruction/pixel", "reconstruction/perceptual", "reconstruction/commitment_L0"}
        self.assertEqual(set(parts), expected | {"commitment/total", "hsic/total", *arms})
        total = captured["total"].sum()
        self.assertAlmostEqual(total.item(), out[0], places=6)
        torch.testing.assert_close(sum(v.sum() for v in parts.values()), total)
        # Each term carries every coefficient training applies to it.
        pixel = recon.summaries["scalar"]["Loss-MAE-Reconstruction"]
        torch.testing.assert_close(parts["reconstruction/pixel"], 2 * pixel)
        self.assertAlmostEqual(
            parts["content/L0/patch/off_diag"].item(), 3 * 0.5 * 2 * diag["Contrastive/off_diag_loss_L0"], places=4
        )
        self.assertAlmostEqual(
            parts["content/L0/gap/off_diag"].item(), 3 * 1.5 * 2 * diag["Contrastive/gap_off_diag_loss_L0"], places=4
        )
        self.assertAlmostEqual(parts["hsic/total"].item(), diag["Style/hsic_weighted"], places=6)
        params = list(model.parameters())

        def grad(value):
            gs = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
            return torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten() for p, g in zip(params, gs)])

        grads = [grad(v.sum()) for v in parts.values() if v.requires_grad]
        self.assertGreater(float(grad(parts["reconstruction/pixel"]).norm()), 0)
        # Norm-scale comparison: per-element float32 re-association noise (~1e-5 here) is not a
        # defect. A dropped term is orders of magnitude larger (see the tampered-observer test).
        error = (sum(grads) - grad(total)).norm() / sum(g.norm() for g in grads)
        self.assertLess(float(error), 1e-4)

    def test_single_count_commitment_drops_the_inner_count(self):
        captured = {}
        torch.manual_seed(0)
        self.run_step(
            tiny_model(),
            config("--single-count-commitment"),
            batch(3),
            observer=lambda total, parts, diag: captured.update(parts),
        )
        self.assertIn("commitment/total", captured)
        self.assertNotIn("reconstruction/commitment_L0", captured)


class TrainingObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_reference_mode_differentiates_the_frozen_ema_mixture(self):
        model = tiny_model()
        with frozen_checkpoint(model):
            obj = objective(model, config())
            with self.assertRaisesRegex(ValueError, "not been calibrated"):
                obj.evaluate(batch(0))
            report = obj.calibrate([batch(1), batch(2)])
            instantaneous = []
            for data in (batch(1), batch(2)):
                with torch.no_grad():
                    obj.evaluate(data, reference_only=True)
                instantaneous.append(obj.last_correlations["gap"][(0, 0, 1)]["c"])
            reference = obj.reference["gap"][(0, 0, 1)]["c"]
            torch.testing.assert_close(reference, (instantaneous[0] + instantaneous[1]) / 2)
            measured = obj.evaluate(batch(3))
            c = obj.last_correlations["gap"][(0, 0, 1)]["c"]
            mixed = 0.9 * reference + 0.1 * c
            off = mixed.square().sum() - mixed.diagonal().square().sum()
            # scale_contrastive x gap weight x gap lambda (unset, so bt_lambda).
            expected = 3 * 1.5 * 2 * off
            torch.testing.assert_close(measured["terms"]["content/L0/gap/off_diag"], expected)
            again = obj.evaluate(batch(3))  # measuring never advances the reference
            torch.testing.assert_close(again["terms"]["content/L0/gap/off_diag"], expected)
        self.assertEqual(set(report), {"patch", "gap"})
        self.assertEqual(report["gap"]["batches"], 2)
        self.assertGreater(report["gap"]["offdiag_noise_reference"], report["gap"]["offdiag_noise_training_ema"])

    def test_instantaneous_mode_is_the_objective_without_ema(self):
        model = tiny_model()
        with frozen_checkpoint(model):
            torch.manual_seed(0)
            a = objective(model, config(), "instantaneous").evaluate(batch(4))
            torch.manual_seed(0)
            b = objective(model, config("--bt-corr-ema", "0")).evaluate(batch(4))
        self.assertEqual(set(a["terms"]), set(b["terms"]))
        for key, value in b["terms"].items():
            torch.testing.assert_close(a["terms"][key], value, rtol=0, atol=0)

    def test_collect_reproduces_the_total_and_leaves_the_model_unchanged(self):
        model = tiny_model()
        before = state_digest(model)
        with frozen_checkpoint(model):
            obj = objective(model, config())
            obj.calibrate([batch(1)])
            stacks, values, checks, named, diagnostics = audit.collect(obj, [batch(5), batch(6)])
        self.assertEqual(state_digest(model), before)
        self.assertEqual(stacks["total"].shape, (2, sum(p.numel() for _, p in named)))
        # The stored per-component stacks re-sum to the stored total. Components partly cancel,
        # so float error scales with their summed norms, as in the audit's own bound.
        parts = [v.astype(np.float64) for k, v in stacks.items() if k != "total"]
        scale = sum(np.linalg.norm(p, axis=1) for p in parts)
        self.assertLess((np.linalg.norm(sum(parts) - stacks["total"], axis=1) / scale).max(), 1e-4)
        self.assertLess(max(c["gradient_error_vs_component_norms"] for c in checks), 1e-4)
        self.assertEqual(len(values["total"]), 2)
        self.assertIn("Contrastive/off_diag_inst_L0", diagnostics)

    def test_terms_missing_from_the_observer_fail_parity(self):
        def tampered(key, detach):
            def impl(*args, loss_observer, **kwargs):
                def observer(total, parts, diag):
                    parts = dict(parts)
                    if detach:
                        parts[key] = parts[key].detach()
                    else:
                        del parts[key]
                    loss_observer(total, parts, diag)

                return TRAIN_STEP(*args, loss_observer=observer, **kwargs)

            return impl

        model = tiny_model()
        with frozen_checkpoint(model):
            # A term absent from the observer fails the value check...
            obj = objective(model, config(), "instantaneous", tampered("content/L0/gap/off_diag", False))
            with self.assertRaises(AssertionError):
                obj.evaluate(batch(7))
            # ...and one whose value is observed without its gradient fails the gradient check.
            obj = objective(model, config(), "instantaneous", tampered("content/L0/gap/on_diag", True))
            with self.assertRaisesRegex(RuntimeError, "without passing through"):
                audit.collect(obj, [batch(7)])


class StepSweepTests(unittest.TestCase):
    def test_sweep_reports_excess_and_first_order_and_always_restores(self):
        model = torch.nn.Module()
        model.encoders = torch.nn.Linear(3, 2)
        named = ga._encoder_params(model)
        params = [p for _, p in named]
        saved = [p.detach().clone() for p in params]
        origin = torch.cat([p.detach().flatten() for p in params])

        def measure():
            theta = torch.cat([p.detach().flatten() for p in params])
            return {"linear": float((origin - theta).sum()), "flat": 1.0}

        groups = {"content": np.ones((2, origin.numel()), dtype=np.float32)}
        cli = types.SimpleNamespace(etas=[0.1, 0.2], random_controls=1, seed=0)
        # A unit step along -ones/|ones| raises "linear" by eta * sqrt(n).
        slope = math.sqrt(origin.numel())
        base, steps, summary = audit.step_sweep(cli, named, groups, measure, lambda u: {"linear": float(u.sum())})
        rows = {(r["metric"], r["eta"]): r for r in summary}
        self.assertEqual(base, {"linear": 0.0, "flat": 1.0})
        self.assertAlmostEqual(rows[("linear", 0.1)]["delta"], 0.1 * slope, places=5)
        self.assertAlmostEqual(rows[("linear", 0.2)]["first_order"], 0.2 * slope, places=5)
        self.assertAlmostEqual(rows[("linear", 0.1)]["linearity_r2"], 1.0, places=6)
        self.assertIsNone(rows[("flat", 0.1)]["first_order"])
        self.assertEqual(rows[("flat", 0.2)]["delta"], 0.0)
        row = rows[("linear", 0.2)]
        self.assertAlmostEqual(row["excess_over_random"], row["delta"] - row["random_delta"], places=9)
        self.assertEqual({s["direction"] for s in steps}, {"gradient", "random_1"})
        self.assertTrue(all(torch.equal(p, s) for p, s in zip(params, saved)))

        calls = []

        def failing():
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("probe failed")
            return measure()

        with self.assertRaisesRegex(RuntimeError, "probe failed"):
            audit.step_sweep(cli, named, groups, failing)
        self.assertTrue(all(torch.equal(p, s) for p, s in zip(params, saved)))

    def test_block_mcc_measure_caches_the_ground_truth(self):
        seen = []

        def fake_mcc(model, dataset, device, grid, level, batch_size, targets, seeds, n_splits):
            seen.append(targets)
            return 0.5, "targets"

        cli = types.SimpleNamespace(mcc_batch=2, seeds=[0], n_splits=2)
        with patch.object(ga, "_mcc_now", fake_mcc):
            measure = audit.block_mcc_measure(None, None, "cpu", [2, 2, 2], cli)
            self.assertEqual(measure(), {"block_mcc": 0.5})
            measure()
        self.assertEqual(seen, [None, "targets"])


def decodable_batch(seed, b=6):
    """Images whose global intensity carries z_content[:, 1], so a GAP probe has signal to read."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(b, 3, generator=g)
    shift = z[:, 1].view(b, 1, 1, 1, 1)
    x0 = 0.3 * torch.randn(b, 1, 8, 8, 8, generator=g) + shift
    x1 = 0.3 * torch.randn(b, 1, 8, 8, 8, generator=g) + 0.5 * shift
    return torch.cat([x0, x1]), torch.ones(2 * b, 1, 8, 8, 8, dtype=torch.bool), z


class DecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_envelope_gradient_is_the_derivative_of_the_refit_objective(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 5)) * [1.0, 2.0, 0.5, 3.0, 1.0] + [0.0, 1.0, -2.0, 0.5, 3.0]
        y = X @ [0.5, -0.2, 0.0, 0.1, 0.3] + 0.5 * rng.normal(size=80)
        fit = audit.ridge_feature_gradient(X, y)
        self.assertAlmostEqual(audit.penalised_fit_r2(X, y, fit["alpha"]), fit["fit_r2"], places=10)

        def refit_objective(features):
            return (1.0 - audit.penalised_fit_r2(features, y, fit["alpha"])) * fit["ss_tot"]

        E, eps = rng.normal(size=X.shape), 1e-5
        numeric = (refit_objective(X + eps * E) - refit_objective(X - eps * E)) / (2 * eps)
        self.assertAlmostEqual(numeric, float((fit["feature_gradient"] * E).sum()), delta=1e-6 * abs(numeric))
        # A per-channel rescaling or shift is absorbed by the standardised refit: zero derivative.
        G = fit["feature_gradient"]
        self.assertAlmostEqual(float((G[:, 0] * X[:, 0]).sum()), 0.0, places=8)
        self.assertAlmostEqual(float(G[:, 3].sum()), 0.0, places=8)

    def test_calibration_equals_one_full_backward_and_predicts_small_steps(self):
        model = tiny_model()
        batches = [decodable_batch(seed) for seed in range(4)]
        with frozen_checkpoint(model):
            params = [p for _, p in ga._encoder_params(model)]
            decoder = audit.IidDecoding(model, batches, ["ventricle_size"], [1], "cpu")
            baseline = decoder.calibrate(params, seeds=(0,), n_splits=2)
            self.assertEqual(set(baseline), {"r2/ventricle_size/T1", "r2/ventricle_size/FLAIR"})
            views = [decoder._views(x, m) for x, m, _ in batches]
            for v in (0, 1):
                features = torch.cat([pair[v] for pair in views])
                fit = audit.ridge_feature_gradient(features.detach().double().numpy(), decoder.y[:, 0])
                weights = torch.as_tensor(fit["feature_gradient"]).to(features)
                grads = torch.autograd.grad(
                    features, params, grad_outputs=weights, retain_graph=True, allow_unused=True
                )
                full = torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p, g in zip(params, grads)])
                chunked = torch.as_tensor(decoder.targets[("ventricle_size", v)]["gradient"])
                self.assertLess(float((chunked - full.double()).norm() / full.norm()), 1e-5)
            del views

            # Central finite difference of the fixed-alpha fit along the steepest direction. This
            # tiny model is strongly curved: the ratio is 1.022 at eta 1e-3 and 0.99998 at 1e-4.
            gradient = decoder.targets[("ventricle_size", 0)]["gradient"]
            u = gradient / np.linalg.norm(gradient)
            slope = decoder.slopes(u)["fit/ventricle_size/T1"]
            saved = [p.detach().clone() for p in params]
            fits, eta = [], 1e-4
            for sign in (1.0, -1.0):
                with torch.no_grad():
                    offset = 0
                    for p, s in zip(params, saved):
                        step = torch.as_tensor(u[offset : offset + p.numel()]).view_as(p).to(p)
                        p.copy_(s - sign * eta * step)
                        offset += p.numel()
                fits.append(decoder.scores((0,), 2)["fit/ventricle_size/T1"])
            with torch.no_grad():
                for p, s in zip(params, saved):
                    p.copy_(s)
        self.assertGreater(slope, 0)
        self.assertAlmostEqual((fits[0] - fits[1]) / (2 * eta) / slope, 1.0, delta=0.005)

    def test_attribution_is_additive_and_signed(self):
        rng = np.random.default_rng(1)
        d = rng.normal(size=6)
        parts = {
            "content/L0/gap/on_diag": np.stack([d, 2 * d]),
            "reconstruction/pixel": np.stack([-d, -d]),
            "hsic/total": rng.normal(size=(2, 6)),
        }
        stacks = dict(parts, total=sum(parts.values()))
        groups, _ = audit.group_stacks(stacks, {k: [0.0, 0.0] for k in stacks})
        targets = {("ventricle_size", 0): dict(gradient=d, ss_tot=2.0)}
        rows = {r["component"]: r for r in audit.decoding_attribution(stacks, groups, targets)}
        helps, hurts = rows["content/L0/gap/on_diag"], rows["reconstruction/pixel"]
        self.assertAlmostEqual(helps["rate"], 1.5 * float(d @ d) / 2.0)
        self.assertAlmostEqual(helps["cosine"], 1.0)
        self.assertEqual((helps["hurting_batches"], hurts["hurting_batches"]), (0.0, 1.0))
        self.assertAlmostEqual(sum(rows[k]["rate"] for k in parts), rows["total"]["rate"])
        self.assertAlmostEqual(sum(rows[k]["share_of_total"] for k in parts), 1.0)
        self.assertEqual((rows["content"]["kind"], rows["total"]["kind"], helps["view"]), ("group", "total", "T1"))

    def test_only_gap_assigned_factors_are_accepted_and_checked_first(self):
        self.assertEqual(audit.decode_columns(["ventricle_size", "brain_size"]), [1, 0])
        with self.assertRaisesRegex(ValueError, "patch"):
            audit.decode_columns(["lesion_x"])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            audit.decode_columns(["ventricle"])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            audit.decode_columns(["sulcal_widening"], n_content=5)
        cli = types.SimpleNamespace(threads=1, decode_factors=["lesion_x"], checkpoints=["x.pt"], run_dir="/missing")
        with self.assertRaisesRegex(ValueError, "patch"):  # before any checkpoint lookup
            audit.run(cli)


class SummaryTests(unittest.TestCase):
    def test_shares_partition_the_total_and_groups_collect_both_commitment_counts(self):
        rng = np.random.default_rng(0)
        keys = ("content/L0/patch/on_diag", "reconstruction/pixel", "reconstruction/commitment_L0", "commitment/total")
        parts = {k: rng.normal(size=(3, 5)) for k in keys}
        stacks = dict(parts, total=sum(parts.values()))
        values = {k: [1.0, 2.0, 3.0] for k in stacks}
        named = [("encoders.w", torch.zeros(3)), ("content_norms.b", torch.zeros(2))]
        rows, pairs, modules = audit.summarize(stacks, values, named)
        shares = {r["component"]: r["signed_total_projection"] for r in rows}
        self.assertAlmostEqual(sum(v for k, v in shares.items() if k != "total"), 1.0, places=9)
        self.assertEqual(len(pairs), 10)
        self.assertEqual({m["module"] for m in modules}, {"encoders", "content_norms"})
        groups, group_values = audit.group_stacks(stacks, values)
        self.assertEqual(list(groups), ["content", "reconstruction", "commitment"])
        np.testing.assert_allclose(
            groups["commitment"], parts["reconstruction/commitment_L0"] + parts["commitment/total"]
        )
        self.assertEqual(group_values["commitment"], [2.0, 4.0, 6.0])


class SettingsAndCliTests(unittest.TestCase):
    def test_default_adversarial_weight_without_a_discriminator_is_accepted(self):
        args = config()
        self.assertEqual(args.scale_adv_loss, 0.1)  # the parser default, with no GAN
        audit.validate_settings(args)
        args.use_gan = True
        with self.assertRaisesRegex(ValueError, "GAN"):
            audit.validate_settings(args)

    def test_caveats_and_ema_window(self):
        args = config()
        self.assertEqual(audit.objective_caveats(args), [])
        args.style_dropout_prob = 0.3
        self.assertIn("style_dropout_prob", audit.objective_caveats(args)[0])
        self.assertAlmostEqual(audit.ema_window(types.SimpleNamespace(bt_corr_ema=0.99)), 199.0)
        self.assertEqual(audit.ema_window(types.SimpleNamespace(bt_corr_ema=0.0)), 0.0)

    def test_cli_defaults_and_rejections(self):
        with patch.object(audit, "run") as run:
            ga.main(["--run-dir", "RUN"])
        cli = run.call_args.args[0]
        self.assertEqual(
            (cli.target, cli.grad_batches, cli.ema_mode, cli.ema_reference_batches), ("balance", 16, "reference", None)
        )
        self.assertEqual(cli.etas, [0.05, 0.2, 0.8])
        self.assertEqual((cli.decode_factors, cli.decode_samples, cli.mcc_batch), (["ventricle_size"], 512, 8))
        with patch.object(audit, "run") as run:
            ga.main(["--run-dir", "RUN", "--decode-factors", "--eval-batch", "4"])
        self.assertEqual((run.call_args.args[0].decode_factors, run.call_args.args[0].mcc_batch), ([], 4))
        for bad in (
            ["--ema-mode", "cold"],
            ["--precondition"],
            ["--level", "1"],
            ["--ema-reference-batches", "0"],
            ["--target", "decode", "--decode-factors"],
            ["--decode-samples", "10"],
        ):
            with self.subTest(bad=bad), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                ga.main(["--run-dir", "RUN", *bad])


if __name__ == "__main__":
    unittest.main()
