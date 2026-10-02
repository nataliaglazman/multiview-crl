"""Frozen, component-wise encoder gradients of the current training objective.

The objective is not re-implemented: ``train_step`` runs unchanged with no optimizer, and its
``loss_observer`` hands over every weighted term of ``total_loss`` from that one forward.
Every batch must pass two parity checks before it is used: the terms sum to the total, and
their encoder gradients sum to the total's. A term that enters training but not the audit
therefore fails loudly instead of vanishing from the tables. See GRADIENT_ATTRIBUTION.md.
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from eval.gradients.gradient_attribution import _encoder_params, _flat, cosine, linearity_check, snr_decomposition
from eval.gradients.reconstruction_attribution import frozen_checkpoint
from training.bt_objective import make_barlow_loss_functions

LOG = logging.getLogger(__name__)

VALUE_PARITY_RTOL = 2e-5
# Each component backpropagates through the encoder's large conv reductions separately, so
# float32 re-association leaves the summed component gradients ~1e-5 of their summed norms
# away from the total's (measured 9e-6 on ident-vent-hsic). The bound is 100x that noise;
# an unobserved force above 0.1% of the components' summed magnitude still fails it.
GRADIENT_PARITY_RTOL = 1e-3
GROUPS = ("content", "style", "reconstruction", "commitment", "hsic", "cross_reconstruction")
# Below this CV R^2 a factor has no linear direction for first-order rates to act on: R^2 is
# locally quadratic in any emerging signal, so its derivative vanishes whatever a step does.
UNDECODABLE_R2 = 0.05


def validate_settings(args):
    if getattr(args, "contrastive_loss_type", None) != "barlow_twins":
        raise ValueError("Full loss attribution currently supports Barlow Twins checkpoints only.")
    if getattr(args, "dataset_name", None) != "synthetic":
        raise ValueError("Full loss attribution requires the synthetic dataset.")
    if int(getattr(args, "vqvae_nb_levels", 1)) != 1:
        raise ValueError("Full loss attribution currently requires one VQ level.")
    if getattr(args, "mask_mode", None) != "fixed":
        raise ValueError("Full loss attribution currently requires a fixed content/style mask.")
    if getattr(args, "contrastive_proj_mode", "head") != "head":
        raise ValueError("Only the standard contrastive projection-head mode is supported.")
    if getattr(args, "recon_loss_fn", "BaselineLoss") != "BaselineLoss":
        raise ValueError("Full loss attribution currently requires BaselineLoss.")
    # scale_adv_loss defaults to 0.1 whether or not a discriminator exists; the generator
    # term only enters the objective under --use-gan.
    if getattr(args, "use_gan", False) and float(getattr(args, "scale_adv_loss", 0.0) or 0.0) > 0:
        raise ValueError(
            "Full loss attribution cannot reproduce the GAN generator term; refusing an incomplete objective."
        )
    for name in (
        "use_moco",
        "freeze_encoder",
        "scale_content_modality_adv",
        "scale_content_patch_modality_adv",
        "scale_style_modality_ce",
    ):
        if getattr(args, name, False):
            raise ValueError(f"Full loss attribution does not support active {name}; refusing an incomplete objective.")
    if [tuple(s) for s in args.subsets] != [(0, 1)]:
        raise ValueError("Full loss attribution requires exactly two views and subset (0, 1).")
    if float(getattr(args, "skip_recon_ratio", 0)) != 0:
        raise ValueError("Stochastic reconstruction skipping is unsupported; its frequency changes the objective.")


def objective_caveats(args):
    """Settings under which the eval-mode objective is not the one training optimised."""
    caveats = []
    if float(getattr(args, "style_dropout_prob", 0.0) or 0.0) > 0:
        caveats.append(
            f"style_dropout_prob={args.style_dropout_prob}: training zeroes injected style at that rate, "
            "eval mode never does, so reconstruction gradients describe the no-dropout objective."
        )
    return caveats


def ema_window(args):
    """Effective number of batches in the correlation EMA, (1+m)/(1-m); 0 when it is off."""
    m = float(getattr(args, "bt_corr_ema", 0.0) or 0.0)
    return (1.0 + m) / (1.0 - m) if 0.0 < m < 1.0 else 0.0


class TrainingObjective:
    """Expose weighted, disjoint components from one real training forward.

    ``ema_mode="reference"`` differentiates m*C_ref + (1-m)*C_batch against a frozen
    checkpoint estimate C_ref (see ``calibrate``). ``"instantaneous"`` drops the EMA.
    A cold EMA is not offered: at t=1 the bias correction makes it exactly instantaneous.
    """

    def __init__(
        self,
        model,
        args,
        step,
        *,
        ema_mode="reference",
        train_step_impl=None,
        recon_loss=None,
        loss_impl=None,
        stats_impl=None,
    ):
        validate_settings(args)
        if ema_mode not in ("reference", "instantaneous"):
            raise ValueError(f"Unknown EMA mode {ema_mode!r}.")
        if train_step_impl is None:
            from training.main_multimodal import train_step

            train_step_impl = train_step
        if recon_loss is None:
            from training.losses import BaselineLoss

            recon_loss = BaselineLoss().to(next(model.parameters()).device)
        self.model, self.args, self.step = model, copy.deepcopy(args), step
        self.train_step, self.recon_loss = train_step_impl, recon_loss
        self.loss_impl, self.stats_impl = loss_impl, stats_impl
        self.ema_mode = ema_mode
        self.reference = {}
        self.last_correlations = {}

    def evaluate(self, batch, *, reference_only=False):
        args = copy.deepcopy(self.args)
        if reference_only:
            # Reference collection needs only the alignment forward. No extra style pair,
            # reconstruction, commitment, or HSIC computation is required.
            args.scale_style_contrastive_loss = 0.0
            args.scale_style_hsic_loss = 0.0
        ema_on = float(getattr(args, "bt_corr_ema", 0.0) or 0.0) > 0
        if reference_only or self.ema_mode == "instantaneous":
            args.bt_corr_ema = 0.0
        elif ema_on and not self.reference:
            raise ValueError("EMA reference has not been calibrated.")
        # A copy per batch: the reference never advances on the batches being measured.
        states = {} if reference_only else copy.deepcopy(self.reference)
        self.last_correlations = {}
        plain, patch = make_barlow_loss_functions(
            args,
            states=states,
            correlations=self.last_correlations,
            capture_components=True,
            loss_impl=self.loss_impl,
            stats_impl=self.stats_impl,
        )
        observed = {}

        def capture(total, components, diagnostics):
            # total_loss is (1,)-shaped and the terms mostly 0-d, so compare scalars.
            total = total.sum()
            terms = {k: (v if torch.is_tensor(v) else total.new_tensor(float(v))).sum() for k, v in components.items()}
            if not all(bool(torch.isfinite(t)) for t in (total, *terms.values())):
                raise ValueError("Non-finite loss component: train_step would skip this batch.")
            summed = sum(terms.values(), total.new_zeros(()))
            torch.testing.assert_close(summed, total, rtol=VALUE_PARITY_RTOL, atol=1e-6)
            observed.update(
                total=total, terms=terms, diagnostics=diagnostics, value_error=float((summed - total).abs())
            )

        self.train_step(
            batch,
            [self.model],
            [],
            plain,
            None,
            list(self.model.parameters()),
            args,
            recon_loss_fn=self.recon_loss,
            patch_loss_func=patch,
            step=self.step,
            force_compute_recon=not reference_only,
            loss_observer=capture,
        )
        if not observed:
            raise RuntimeError("train_step returned without calling the loss observer.")
        return observed

    def calibrate(self, batches):
        """Average instantaneous correlations on separate batches, then freeze them.

        A large t makes bias correction converge to one. With this fixed reference,
        each measured batch differentiates m*C_ref + (1-m)*C_batch. It never advances
        the reference based on the preceding measured batch.

        Returns, per arm, how much of the reference's off-diagonal is sampling noise next
        to training's EMA: sum_{i!=j} Var(C_ij)/n against sum_{i!=j} Var(C_ij)(1-m)/(1+m).
        """
        sums, squares, counts, signatures, dtypes = {}, {}, {}, {}, {}
        with torch.no_grad():
            for n, batch in enumerate(batches, 1):
                self.evaluate(batch, reference_only=True)
                for arm, records in self.last_correlations.items():
                    for key, record in records.items():
                        index = (arm, key)
                        if index in signatures and signatures[index] != record["sig"]:
                            raise ValueError("Correlation dimensions changed during reference calibration.")
                        signatures[index], dtypes[index] = record["sig"], record["c"].dtype
                        c = record["c"].double()
                        sums[index] = sums.get(index, 0) + c
                        squares[index] = squares.get(index, 0) + c * c
                        counts[index] = counts.get(index, 0) + 1
                if n % 20 == 0:
                    LOG.info("EMA reference: %d batches", n)
        if not sums:
            raise ValueError("No correlation reference batches were collected.")
        m = float(getattr(self.args, "bt_corr_ema", 0.0) or 0.0)
        self.reference, report = {}, {}
        for (arm, key), total in sums.items():
            n = counts[(arm, key)]
            mean = total / n
            self.reference.setdefault(arm, {})[key] = {
                "c": mean.to(dtypes[(arm, key)]).detach(),
                "t": 10**9,
                "sig": signatures[(arm, key)],
            }
            off = ~torch.eye(mean.shape[0], dtype=torch.bool, device=mean.device)
            row = report.setdefault(
                arm,
                dict(
                    batches=n, offdiag_sq_reference=0.0, offdiag_noise_reference=None, offdiag_noise_training_ema=None
                ),
            )
            row["offdiag_sq_reference"] += float(mean[off].square().sum())
            if n > 1:
                variance = ((squares[(arm, key)] / n - mean.square()) * n / (n - 1)).clamp_min(0)[off].sum()
                row["offdiag_noise_reference"] = (row["offdiag_noise_reference"] or 0.0) + float(variance / n)
                row["offdiag_noise_training_ema"] = (row["offdiag_noise_training_ema"] or 0.0) + float(
                    variance * (1 - m) / (1 + m)
                )
        return report


def gradient(loss, params):
    if not loss.requires_grad:
        return _flat([torch.zeros_like(p) for p in params])
    gradients = torch.autograd.grad(loss.sum(), params, allow_unused=True, retain_graph=True)
    result = _flat([torch.zeros_like(p) if g is None else g for p, g in zip(params, gradients)])
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite encoder gradient.")
    return result


def collect(objective, batches):
    """Per-batch weighted gradients, values, parity checks and train_step diagnostics."""
    named = _encoder_params(objective.model)
    params = [p for _, p in named]
    stacks, values, checks, diagnostics = {}, {}, [], {}
    keys = None
    for index, batch in enumerate(batches):
        measured = objective.evaluate(batch)
        terms, total = measured["terms"], measured["total"]
        if keys is not None and set(terms) != keys:
            raise ValueError(f"Loss component set changed across batches: {sorted(set(terms) ^ keys)}")
        keys = set(terms)
        actual = gradient(total, params)
        summed = torch.zeros_like(actual)
        scale = 0.0
        for key, term in terms.items():
            g = gradient(term, params)
            summed.add_(g)
            scale += float(g.norm())
            stacks.setdefault(key, []).append(g.cpu().numpy())
            values.setdefault(key, []).append(float(term.detach()))
        error = float((summed - actual).norm())
        if error > GRADIENT_PARITY_RTOL * scale + 1e-12:
            raise RuntimeError(
                f"Batch {index}: component gradients miss the total encoder gradient by {error:.3g} "
                f"({error / max(scale, 1e-30):.2e} of their summed norms). A term enters total_loss "
                "without passing through the loss observer."
            )
        checks.append(
            dict(
                batch=index,
                value_error=measured["value_error"],
                value_relative_error=measured["value_error"] / max(abs(float(total.detach())), 1e-30),
                gradient_error=error,
                gradient_error_vs_component_norms=error / max(scale, 1e-30),
                gradient_error_vs_total_norm=error / max(float(actual.norm()), 1e-30),
            )
        )
        stacks.setdefault("total", []).append(actual.cpu().numpy())
        values.setdefault("total", []).append(float(total.detach()))
        for key, value in measured["diagnostics"].items():
            if isinstance(value, (int, float)) or (torch.is_tensor(value) and value.numel() == 1):
                diagnostics.setdefault(key, []).append(float(value))
        LOG.info("Attributed batch %d: %d components; weighted encoder norm %.5g", index + 1, len(terms), actual.norm())
        del measured, terms, total, actual, summed
    if not checks:
        raise ValueError("No gradient batches.")
    means = {k: float(np.mean(v)) for k, v in diagnostics.items()}
    return {k: np.stack(v) for k, v in stacks.items()}, values, checks, named, means


def component_group(key):
    """Commitment is counted wherever it enters: inside BaselineLoss AND as Loss/VQ."""
    if key == "total":
        return "total"
    return "commitment" if "commitment" in key else key.split("/")[0]


def group_stacks(stacks, values):
    groups, group_values = {}, {}
    for key, array in stacks.items():
        if key == "total":
            continue
        name = component_group(key)
        if name in groups:
            groups[name] = groups[name] + array
            group_values[name] = [a + b for a, b in zip(group_values[name], values[key])]
        else:
            groups[name], group_values[name] = array.copy(), list(values[key])
    order = sorted(groups, key=lambda g: GROUPS.index(g) if g in GROUPS else len(GROUPS))
    return {g: groups[g] for g in order}, {g: group_values[g] for g in order}


def summarize(stacks, values, named):
    total = stacks["total"].astype(np.float64)
    denominator = np.square(total).sum(1)
    rows, pairs, modules = [], [], []
    for key, raw in stacks.items():
        gs = raw.astype(np.float64)
        norms = np.linalg.norm(gs, axis=1)
        dots = (gs * total).sum(1)
        fractions = np.divide(dots, denominator, out=np.full(len(gs), np.nan), where=denominator > 0)
        rows.append(
            dict(
                component=key,
                weighted_loss_mean=float(np.mean(values[key])),
                mean_batch_grad_norm=float(norms.mean()),
                batch_grad_norm_std=float(norms.std()),
                mean_gradient_norm=float(np.linalg.norm(gs.mean(0))),
                rms_batch_grad_norm=float(np.sqrt(np.mean(norms**2))),
                signed_total_projection=float(np.nanmean(fractions)) if np.isfinite(fractions).any() else None,
                **{f"snr_{k}": v for k, v in snr_decomposition(gs).items()},
            )
        )
        offset = 0
        grouped = {}
        for name, parameter in named:
            group = name.split(".")[0]
            chunk = gs[:, offset : offset + parameter.numel()]
            grouped[group] = grouped.get(group, 0) + np.square(chunk).sum(1)
            offset += parameter.numel()
        for group, squared in grouped.items():
            modules.append(dict(component=key, module=group, rms_batch_grad_norm=float(np.sqrt(squared.mean()))))
    keys = list(stacks)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            cs = np.array([cosine(a, b) for a, b in zip(stacks[left], stacks[right])])
            valid = cs[np.isfinite(cs)]
            pairs.append(
                dict(
                    left=left,
                    right=right,
                    valid_batches=len(valid),
                    mean_batch_cosine=float(valid.mean()) if len(valid) else None,
                    std_batch_cosine=float(valid.std()) if len(valid) else None,
                    negative_fraction=float((valid < 0).mean()) if len(valid) else None,
                    mean_gradient_cosine=cosine(stacks[left].mean(0), stacks[right].mean(0)),
                )
            )
    return rows, pairs, modules


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def matched_random(direction, params, seed):
    """A random direction with the SAME per-tensor norms as ``direction`` (so unit-norm too).

    Block-MCC falls under almost any perturbation of sufficient size. Without this control a
    loss that degrades identifiability cannot be told from a step that is merely large;
    matching the per-layer energy leaves only the direction within each tensor to the loss.
    """
    generator = torch.Generator().manual_seed(seed)
    pieces, offset = [], 0
    for p in params:
        n = p.numel()
        noise = torch.randn(n, generator=generator).to(direction)
        pieces.append(noise / noise.norm().clamp_min(1e-12) * direction[offset : offset + n].norm())
        offset += n
    return torch.cat(pieces)


def decode_columns(factors, n_content=None):
    """Validate decoding targets; return their ``z_content`` columns.

    Only GAP-assigned factors are accepted. Those are the morphometry factors, and
    run_dci_compare scores them from pooled channel means, a d-wide block that ridge handles
    at a few hundred subjects. Patch-assigned factors (lesion_x/y/z) need ~C*P features
    against N, where an unreduced ridge overfits and the in-sample objective is degenerate.
    """
    from eval.metrics.dci import CONTENT_FACTOR_NAMES
    from eval.protocol.run_dci_compare import FACTOR_POOLING

    names = CONTENT_FACTOR_NAMES[: n_content or len(CONTENT_FACTOR_NAMES)]
    columns = []
    for factor in factors:
        if factor not in names:
            raise ValueError(f"Unknown content factor {factor!r}; choose from {names}.")
        pooling = FACTOR_POOLING.get(factor, "stats")
        if pooling != "gap":
            raise ValueError(
                f"{factor} is scored at {pooling} pooling by run_dci_compare; i.i.d. decoding supports "
                "GAP-assigned factors only."
            )
        columns.append(names.index(factor))
    return columns


def iid_batches(args, n, batch_size, workers):
    """Render ``n`` test subjects once, factors drawn i.i.d., keeping only what probes read."""
    from torch.utils.data import DataLoader

    from eval.ventricle.ventricle_routing import make_dataset

    iid_args = copy.deepcopy(args)
    iid_args.synthetic_style_alignment_pairs = False
    dataset = make_dataset(iid_args, n, "iid", "test")
    batches = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers):
        images = torch.cat(batch["image"], 0)
        masks = batch.get("mask")
        if masks is not None:
            masks = torch.cat(masks, 0) if isinstance(masks, (list, tuple)) else torch.cat([masks, masks], 0)
            masks = (masks.unsqueeze(1) if masks.ndim == images.ndim - 1 else masks) > 0
        batches.append((images, masks, batch["gt_latents"]["z_content"]))
    return batches


def ridge_feature_gradient(X, y):
    """Fit the repo's ridge probe and return dJ/dX of its refit objective.

    J(X) = min_{w,b} ||y - S(X) w - b||^2 + alpha ||w||^2, where S is the StandardScaler map
    and alpha the RidgeCV choice, then held fixed. The minimiser is unique, so by the
    envelope theorem dJ/dX is the partial derivative at the fitted (w, b): the probe stays
    fixed while the features and their standardisation move. A change that a refit probe
    absorbs, such as a per-channel rescaling, therefore has no effect. Also returns SS_tot
    and the penalised in-sample R^2 = 1 - J/SS_tot whose first-order change this attributes.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from eval.metrics.identifiability_metrics import _make_regressor

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    scaler = StandardScaler().fit(X)
    alpha = float(_make_regressor("ridge", 0).fit(scaler.transform(X), y).alpha_)
    # RidgeCV only chooses alpha; the refit at that fixed alpha is what penalised_fit_r2 re-measures.
    probe = Ridge(alpha=alpha).fit(scaler.transform(X), y)
    w = torch.as_tensor(np.asarray(probe.coef_, dtype=np.float64).ravel())
    features = torch.as_tensor(X).requires_grad_()
    scale = features.std(0, unbiased=False)
    # StandardScaler leaves a constant column unscaled; mirror that rather than divide by ~0.
    scale = torch.where(torch.as_tensor(scaler.scale_ == 1.0), torch.ones_like(scale), scale)
    residual = torch.as_tensor(y) - ((features - features.mean(0)) / scale) @ w - float(probe.intercept_)
    data_term = residual.square().sum()
    (feature_gradient,) = torch.autograd.grad(data_term, features)
    ss_tot = float(np.square(y - y.mean()).sum())
    penalty = alpha * float(w.square().sum())
    return dict(
        feature_gradient=feature_gradient.numpy(),
        ss_tot=ss_tot,
        alpha=alpha,
        fit_r2=1.0 - (float(data_term) + penalty) / ss_tot,
        in_sample_r2=1.0 - float(data_term) / ss_tot,
    )


def penalised_fit_r2(X, y, alpha):
    """1 - J/SS_tot of the standardised ridge refit at a FIXED alpha: the envelope objective."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64).ravel()
    standardised = StandardScaler().fit_transform(X)
    probe = Ridge(alpha=alpha).fit(standardised, y)
    residual = y - probe.predict(standardised)
    penalty = alpha * float(np.square(probe.coef_).sum())
    return 1.0 - (float(np.square(residual).sum()) + penalty) / float(np.square(y - y.mean()).sum())


VIEW_LABELS = ("T1", "FLAIR")


def decode_metric(factor, view):
    """CV R^2 of the repo's probe: the decoding number itself."""
    return f"r2/{factor}/{VIEW_LABELS[view]}"


def fit_metric(factor, view):
    """Fixed-alpha penalised in-sample R^2: the quantity the first-order rates differentiate."""
    return f"fit/{factor}/{VIEW_LABELS[view]}"


class IidDecoding:
    """Linear decodability of content factors from each view's GAP content channels.

    Each factor is read the way run_dci_compare reads a GAP-assigned factor: pooled pre-norm
    content channels of one view, StandardScaler + the repo's RidgeCV, k-fold CV. Views are
    scored apart because their encoders are separate. Factors are drawn i.i.d., never from
    the training SCM. Under the SCM ventricle_size correlates ~0.8 with brain_size, so a
    matched probe reads brain_size in disguise (see eval/gradients/probe_prenorm_encoder.py).
    """

    def __init__(self, model, batches, factors, columns, device):
        self.model, self.batches, self.factors, self.device = model, batches, list(factors), device
        self.y = torch.cat([z for _, _, z in batches]).double().numpy()[:, columns]
        self.targets = {}

    def _views(self, images, masks):
        # mask= matters only for latent_mask models, which must be evaluated with it.
        out = self.model(
            images.to(self.device),
            return_recon=False,
            pool_only=True,
            n_views=2,
            mask=None if masks is None else masks.to(self.device).float(),
        )
        features, content_masks = out[2][0], out[6]
        mask = content_masks.get(0) if isinstance(content_masks, dict) else None
        if mask is None:
            index = torch.arange(features.shape[1], device=features.device)
        else:
            index = torch.where((mask[0] if isinstance(mask, tuple) else mask).bool())[-1]
        b = features.shape[0] // 2
        return features[:b, index], features[b:, index]

    def features(self):
        views = ([], [])
        with torch.no_grad():
            for images, masks, _ in self.batches:
                for v, f in enumerate(self._views(images, masks)):
                    views[v].append(f.double().cpu())
        return [torch.cat(v).numpy() for v in views]

    def scores(self, seeds, n_splits, features=None, spread=False):
        """CV R^2 per (factor, view), plus each calibrated probe's fixed-alpha fit.

        With ``spread`` the CV entries are (mean, sd over seeds) pairs.
        """
        from eval.metrics.identifiability_metrics import cv_probe_r2_multi

        out = {}
        for v, X in enumerate(self.features() if features is None else features):
            result = cv_probe_r2_multi(X, self.y, n_splits=n_splits, seeds=seeds, kind="ridge")
            for j, factor in enumerate(self.factors):
                mean, sd = float(result["mean"][j]), float(result["std"][j])
                out[decode_metric(factor, v)] = (mean, sd) if spread else mean
                if (factor, v) in self.targets:
                    out[fit_metric(factor, v)] = penalised_fit_r2(X, self.y[:, j], self.targets[(factor, v)]["alpha"])
        return out

    def calibrate(self, params, seeds, n_splits):
        """Score the checkpoint, fit each probe, and backpropagate its envelope gradient.

        The (N, d) feature gradient is pushed through the encoder one cached batch at a
        time, so only one batch's graph is ever alive. Must run with encoder parameters
        requiring grad and the model in eval mode (inside ``frozen_checkpoint``).
        """
        self.targets = {}
        X = self.features()
        baseline = self.scores(seeds, n_splits, X, spread=True)
        fits = {(f, v): ridge_feature_gradient(X[v], self.y[:, j]) for v in (0, 1) for j, f in enumerate(self.factors)}
        sums = {key: torch.zeros(sum(p.numel() for p in params), dtype=torch.float64) for key in fits}
        offset = 0
        for images, masks, _ in self.batches:
            views = self._views(images, masks)
            n = views[0].shape[0]
            for (factor, v), fit in fits.items():
                if not views[v].requires_grad:
                    continue
                weights = torch.as_tensor(fit["feature_gradient"][offset : offset + n]).to(views[v])
                grads = torch.autograd.grad(
                    views[v], params, grad_outputs=weights, retain_graph=True, allow_unused=True
                )
                sums[(factor, v)] += _flat(
                    [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]
                ).cpu()
            offset += n
            del views
        self.targets = {
            key: dict({k: v for k, v in fit.items() if k != "feature_gradient"}, gradient=sums[key].numpy())
            for key, fit in fits.items()
        }
        return baseline

    def slopes(self, direction):
        """First-order d(fit R^2)/d(eta) of a step theta - eta * direction, per fit metric."""
        u = np.asarray(direction, dtype=np.float64)
        return {fit_metric(f, v): float(t["gradient"] @ u) / t["ss_tot"] for (f, v), t in self.targets.items()}


def decoding_attribution(stacks, groups, targets):
    """Each term's first-order effect on each factor's i.i.d. decodability.

    rate = g_k . dJ/dtheta / SS_tot is the change in penalised in-sample R^2 per unit of a raw
    descent step along -g_k. Positive helps decoding. Rates are additive: over the components
    they sum to the total's, batch by batch.
    """
    entries = [(k, "component", s) for k, s in stacks.items() if k != "total"]
    entries += [(k, "group", s) for k, s in groups.items()] + [("total", "total", stacks["total"])]
    rows = []
    for (factor, view), target in targets.items():
        d = target["gradient"]
        total = float((stacks["total"].astype(np.float64) @ d).mean()) / target["ss_tot"]
        for name, kind, stack in entries:
            gs = stack.astype(np.float64)
            rates = gs @ d / target["ss_tot"]
            cosines = np.array([cosine(g, d) for g in gs])
            rows.append(
                dict(
                    factor=factor,
                    view=VIEW_LABELS[view],
                    component=name,
                    kind=kind,
                    rate=float(rates.mean()),
                    rate_sd=float(rates.std()),
                    cosine=float(np.nanmean(cosines)) if np.isfinite(cosines).any() else None,
                    hurting_batches=float((rates < 0).mean()),
                    share_of_total=float(rates.mean()) / total if total else None,
                )
            )
    return rows


def block_mcc_measure(model, dataset, device, grid, cli):
    """Patch block-MCC on ``dataset`` as a sweep metric; ground truth cached after one call."""
    cache = {}

    def measure():
        from eval.gradients.gradient_attribution import _mcc_now

        score, cache["targets"] = _mcc_now(
            model, dataset, device, grid, 0, cli.mcc_batch, cache.get("targets"), tuple(cli.seeds), cli.n_splits
        )
        return {"block_mcc": score}

    return measure


def step_sweep(cli, named, groups, measure, predict=None):
    """Temporary unit steps along each group's mean gradient, re-measuring every metric.

    ``measure()`` returns ``{metric: value}``; ``predict(u)`` returns first-order slopes
    ``{metric: d value / d eta}`` along -u where one exists. A NEGATIVE delta means stepping
    along that loss's descent direction lowers the metric. The excess over matched-random
    directions is the attribution; the raw delta also carries generic perturbation
    sensitivity. Parameters are restored after every direction, and on error.
    """
    params = [p for _, p in named]
    backups = [p.detach().clone() for p in params]

    def restore():
        with torch.no_grad():
            for p, saved in zip(params, backups):
                p.copy_(saved)

    base = measure()
    steps, summary = [], []
    try:
        for key, stack in groups.items():
            mean = torch.as_tensor(stack.mean(0), device=backups[0].device, dtype=backups[0].dtype)
            if float(mean.norm()) == 0:
                continue
            direction = mean / mean.norm()
            curves = {}
            for trial in range(1 + cli.random_controls):
                label = "gradient" if trial == 0 else f"random_{trial}"
                current = direction if trial == 0 else matched_random(direction, params, 1000 + cli.seed + trial)
                curves[label] = []
                for eta in cli.etas:
                    with torch.no_grad():
                        offset = 0
                        for p, saved in zip(params, backups):
                            n = p.numel()
                            p.copy_(saved - eta * current[offset : offset + n].view_as(p))
                            offset += n
                    values = measure()
                    curves[label].append(values)
                    steps.extend(
                        dict(group=key, direction=label, eta=eta, metric=m, base=base[m], value=v, delta=v - base[m])
                        for m, v in values.items()
                    )
                restore()
            slopes = predict(direction.detach().double().cpu().numpy()) if predict is not None else {}
            for metric in base:
                deltas = [values[metric] - base[metric] for values in curves["gradient"]]
                controls = [
                    [c[metric] - base[metric] for c in curve] for label, curve in curves.items() if label != "gradient"
                ]
                control = np.mean(controls, axis=0) if controls else np.full(len(cli.etas), np.nan)
                r2 = linearity_check(cli.etas, deltas)
                for i, eta in enumerate(cli.etas):
                    summary.append(
                        dict(
                            group=key,
                            metric=metric,
                            eta=eta,
                            delta=deltas[i],
                            random_delta=float(control[i]),
                            excess_over_random=deltas[i] - float(control[i]),
                            linearity_r2=r2,
                            first_order=eta * slopes[metric] if metric in slopes else None,
                        )
                    )
                LOG.info("Steps %s on %s: delta %s, matched random %s", key, metric, deltas, control.tolist())
    finally:
        restore()
    return base, steps, summary


def _fmt(value, spec):
    """Format a table cell; placeholders for missing or infinite values keep the column width."""
    width = int("".join(ch for ch in spec.split(".")[0] if ch.isdigit()) or 0)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—".rjust(width)
    if isinstance(value, float) and math.isinf(value):
        return "inf".rjust(width)
    return format(value, spec)


def print_decoding(decoding):
    print(
        f"\n  i.i.d. decoding: {decoding['subjects']} test subjects, factors drawn i.i.d.; GAP content channels "
        "per view,\n  ridge CV R^2 as in run_dci_compare. Values at the checkpoint:"
    )
    for factor in decoding["factors"]:
        cells = ""
        for v, label in enumerate(VIEW_LABELS):
            mean, sd = decoding["baseline"][decode_metric(factor, v)]
            fit = decoding["probes"][decode_metric(factor, v)]["fit_r2"]
            cells += f"   {label} {_fmt(mean, '.4f')} (sd {_fmt(sd, '.4f')}; in-sample fit {_fmt(fit, '.4f')})"
        print(f"    {factor:<22s}{cells}")
        blind = [
            label
            for v, label in enumerate(VIEW_LABELS)
            if decoding["baseline"][decode_metric(factor, v)][0] < UNDECODABLE_R2
        ]
        if blind:
            print(
                f"    CAVEAT: {factor} is not linearly decodable in {'/'.join(blind)} (CV R^2 < {UNDECODABLE_R2}). "
                "Near R^2 = 0 a new\n    signal enters quadratically, so first-order rates are blind there; "
                "read the finite steps (--target decode)."
            )
    lookup = {(r["factor"], r["view"], r["component"]): r for r in decoding["rows"]}
    for factor in decoding["factors"]:
        print(
            f"\n  {factor}: first-order change of in-sample R^2 per unit raw descent step on each term"
            "\n  (dR2/deta > 0 helps decoding; hurts = fraction of batches whose step lowers it)"
        )
        print(f"  {'':38s}" + "".join(f"{label + ' dR2/deta':>17s}{'cos':>8s}{'hurts':>7s}" for label in VIEW_LABELS))
        for kind in ("component", "group", "total"):
            names = dict.fromkeys(
                r["component"] for r in decoding["rows"] if r["factor"] == factor and r["kind"] == kind
            )
            if kind == "group":
                print(f"  {'group':38s}")
            for name in names:
                cells = ""
                for label in VIEW_LABELS:
                    row = lookup[(factor, label, name)]
                    cells += f"{_fmt(row['rate'], '+17.3e')}{_fmt(row['cosine'], '+8.3f')}"
                    cells += f"{_fmt(row['hurting_batches'], '7.0%')}"
                print(f"  {name:38s}{cells}")


def print_sweep(sweep):
    base, summary = sweep
    etas = list(dict.fromkeys(r["eta"] for r in summary))
    print(f"\n  Unit steps along -g/|g| per group, eta {etas}; delta per step (negative = lowers the metric):")
    metrics = list(dict.fromkeys(r["metric"] for r in summary))

    def rows_for(metric, group):
        return [r for r in summary if r["metric"] == metric and r["group"] == group]

    for metric in (m for m in metrics if not m.startswith("fit/")):
        print(f"  {metric} (at the checkpoint {base[metric]:.4f})")
        companion = "fit/" + metric[len("r2/") :] if metric.startswith("r2/") else None
        for group in dict.fromkeys(r["group"] for r in summary if r["metric"] == metric):
            rs = rows_for(metric, group)
            print(
                f"    {group:<24s}"
                + "".join(_fmt(r["delta"], "+11.2e") for r in rs)
                + f"   linearity R2 {_fmt(rs[0]['linearity_r2'], '.3f')}"
            )
            print(f"    {'  matched random':<24s}" + "".join(_fmt(r["random_delta"], "+11.2e") for r in rs))
            print(f"    {'  EXCESS (attribution)':<24s}" + "".join(_fmt(r["excess_over_random"], "+11.2e") for r in rs))
            if companion in metrics:
                # The fixed-alpha fit is what the first-order rates differentiate; agreement here
                # validates the rate table, while CV R^2 above is the decoding number itself.
                fs = rows_for(companion, group)
                print(
                    f"    {'  fixed-alpha fit':<24s}"
                    + "".join(_fmt(r["delta"], "+11.2e") for r in fs)
                    + f"   linearity R2 {_fmt(fs[0]['linearity_r2'], '.3f')}"
                )
                print(f"    {'  first-order':<24s}" + "".join(_fmt(r["first_order"], "+11.2e") for r in fs))


def print_report(title, rows, group_rows, pairs, group_pairs, checks, reference, caveats, decoding=None, sweep=None):
    print(f"\n{title}")
    head = f"  {'':38s}{'loss':>11s}{'rms |g|':>11s}{'|E g|':>11s}{'share':>9s}{'b->SNR1':>9s}"
    for label, table in (("component", rows), ("group", group_rows)):
        print(f"\n  {label}{head[len(label) + 2:]}")
        for row in table:
            # An identically zero gradient (an inactive hinge) has no SNR to report.
            needed = row.get("snr_batches_for_snr1") if row["rms_batch_grad_norm"] > 0 else None
            print(
                f"  {row['component']:38s}{_fmt(row['weighted_loss_mean'], '11.4g')}"
                f"{_fmt(row['rms_batch_grad_norm'], '11.4g')}{_fmt(row.get('snr_signal_norm'), '11.4g')}"
                f"{_fmt(row['signed_total_projection'], '9.3f')}{_fmt(needed, '9.3g')}"
            )
    print("\n  |E g| is bias-corrected (negative = not resolved from zero at this many batches).")
    print("  share = mean over batches of g_k.g/|g|^2; components sum to 1 and may be negative.")

    print("\n  group cosines (mean over batches; fraction of batches negative):")
    for p in group_pairs:
        if "total" in (p["left"], p["right"]):
            continue
        print(
            f"    {p['left']:>20s} vs {p['right']:<20s}{_fmt(p['mean_batch_cosine'], '+8.3f')}"
            f"   ({_fmt(p['negative_fraction'], '.0%')} negative)"
        )
    conflicts = sorted(
        (
            p
            for p in pairs
            if "total" not in (p["left"], p["right"])
            and p["mean_batch_cosine"] is not None
            and p["mean_batch_cosine"] < 0
        ),
        key=lambda p: p["mean_batch_cosine"],
    )[:8]
    if conflicts:
        print("\n  strongest component conflicts:")
        for p in conflicts:
            print(
                f"    {p['left']:>32s} vs {p['right']:<32s}{p['mean_batch_cosine']:+8.3f}"
                f"   ({p['negative_fraction']:.0%} negative)"
            )

    if reference:
        print("\n  EMA reference, off-diagonal sum of C_ij^2 (correlation^2 units, before lambda):")
        for arm, row in reference.items():
            print(
                f"    {arm:>6s}: reference {row['offdiag_sq_reference']:.4g} over {row['batches']} batches; "
                f"its sampling noise {_fmt(row['offdiag_noise_reference'], '.3g')} vs training EMA "
                f"{_fmt(row['offdiag_noise_training_ema'], '.3g')}"
            )

    if decoding:
        print_decoding(decoding)
    if sweep is not None:
        print_sweep(sweep)

    worst_value = max(c["value_relative_error"] for c in checks)
    worst_gradient = max(c["gradient_error_vs_component_norms"] for c in checks)
    print(f"\n  Parity on every batch: terms sum to train_step's total (worst relative error {worst_value:.1e});")
    print(
        f"  their encoder gradients sum to the total's (worst {worst_gradient:.1e} of summed norms, "
        f"bound {GRADIENT_PARITY_RTOL:g})."
    )
    for caveat in caveats:
        print(f"  CAVEAT: {caveat}")


def run(cli):
    from torch.utils.data import DataLoader, Subset

    from eval.lesion.lesion_reconstruction import json_safe
    from eval.protocol.run_dci_compare import load_contrastive_proj_heads
    from eval.protocol.run_dci_synthetic import load_model_from_run_dir
    from eval.ventricle.ventricle_routing import make_dataset, state_digest

    torch.set_num_threads(cli.threads)
    decode_factors = list(getattr(cli, "decode_factors", None) or [])
    if decode_factors:
        decode_columns(decode_factors)  # fail on a bad name before anything is loaded or rendered
    checkpoints = []
    for name in cli.checkpoints:
        checkpoint = Path(name)
        if not checkpoint.is_absolute():
            checkpoint = Path(cli.run_dir) / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoints.append(checkpoint)
    base_output = (
        Path(cli.out) if cli.out else Path(cli.run_dir) / f"gradient_attribution_{datetime.now():%Y%m%d_%H%M%S}"
    )

    for checkpoint in checkpoints:
        model, args, device = load_model_from_run_dir(cli.run_dir, str(checkpoint), device=cli.device, seed=cli.seed)
        validate_settings(args)
        caveats = objective_caveats(args)
        for caveat in caveats:
            LOG.warning(caveat)
        heads = load_contrastive_proj_heads(cli.run_dir, str(checkpoint), args, model)
        if heads:
            model._contrastive_proj_heads = torch.nn.ModuleDict({f"L{k}": v for k, v in heads.items()})
        # The shared loader is permissive; a strict reload guarantees the audited weights are
        # exactly the checkpoint's, heads included.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(
            {k.removeprefix("module."): v for k, v in state.get("encoders", state).items()}, strict=True
        )
        step = state.get("step")
        if step is None:
            raise ValueError("Checkpoint step is required to reproduce reconstruction/cross-reconstruction schedules.")
        del state
        batch_size = cli.grad_batch_size or int(args.batch_size)
        if batch_size < 2:
            raise ValueError("Need at least two subjects per batch.")
        per_level = getattr(args, "patch_grid_per_level", None)
        grid = list(per_level[0] if per_level else getattr(args, "patch_grid", [8, 8, 8]))
        if cli.grid is not None and list(cli.grid) != grid:
            raise ValueError(f"--grid {cli.grid} must match the saved training patch grid {grid}.")
        if batch_size != int(args.batch_size):
            LOG.warning(
                "Batch size %d differs from training (%d); correlation/variance gradients change.",
                batch_size,
                args.batch_size,
            )

        window = ema_window(args)
        n_reference = 0
        if window and cli.ema_mode == "reference":
            n_reference = cli.ema_reference_batches or math.ceil(window)
            ratio = window / n_reference
            LOG.info("EMA reference: %d batches of %d (window (1+m)/(1-m) = %.0f)", n_reference, batch_size, window)
            if ratio > 2:
                caveats.append(
                    f"EMA reference uses {n_reference} batches against a training window of {window:.0f}: its "
                    f"correlation estimate carries {ratio:.1f}x the EMA's sampling variance, which inflates the "
                    "off-diagonal gradient. See the reference noise lines."
                )
                LOG.warning(caveats[-1])
        n = (cli.grad_batches + n_reference) * batch_size
        ds_args = copy.deepcopy(args)
        ds_args.synthetic_style_alignment_pairs = (
            getattr(args, "style_contrastive_mode", "cosine") == "within_modality"
            and getattr(args, "scale_style_contrastive_loss", 0) > 0
        )
        # Deterministic validation subjects; no training subject or probe label is used.
        dataset = make_dataset(ds_args, n, cli.causal, "val")
        loader_options = dict(batch_size=batch_size, shuffle=False, num_workers=cli.workers)
        reference_loader = DataLoader(Subset(dataset, range(n_reference * batch_size)), **loader_options)
        loader = DataLoader(Subset(dataset, range(n_reference * batch_size, n)), **loader_options)
        decoder = None
        if decode_factors:
            batches = iid_batches(ds_args, cli.decode_samples, cli.mcc_batch, cli.workers)
            columns = decode_columns(decode_factors, batches[0][2].shape[1])
            decoder = IidDecoding(model, batches, decode_factors, columns, device)
            LOG.info(
                "Cached %d i.i.d. subjects for decoding (%.0f MB)",
                len(decoder.y),
                sum(x.numel() * x.element_size() for x, _, _ in batches) / 2**20,
            )
        before = state_digest(model)
        output = base_output / checkpoint.stem if len(checkpoints) > 1 else base_output
        output.mkdir(parents=True, exist_ok=False)
        decoding = sweep = None
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))), frozen_checkpoint(model):
            torch.manual_seed(cli.seed)
            objective = TrainingObjective(model, args, step, ema_mode=cli.ema_mode)
            reference = objective.calibrate(reference_loader) if n_reference else {}
            stacks, values, checks, named, diagnostics = collect(objective, loader)
            groups, group_values = group_stacks(stacks, values)
            rows, pairs, modules = summarize(stacks, values, named)
            group_rows, group_pairs, group_modules = summarize(
                {**groups, "total": stacks["total"]}, {**group_values, "total": values["total"]}, named
            )
            if decoder is not None:
                baseline = decoder.calibrate([p for _, p in named], tuple(cli.seeds), cli.n_splits)
                decoding = dict(
                    subjects=len(decoder.y),
                    factors=decode_factors,
                    baseline=baseline,
                    probes={
                        decode_metric(f, v): {k: t[k] for k in ("alpha", "fit_r2", "in_sample_r2", "ss_tot")}
                        for (f, v), t in decoder.targets.items()
                    },
                    rows=decoding_attribution(stacks, groups, decoder.targets),
                )
            if cli.target in ("mcc", "decode") and not cli.snr:
                measures = []
                if cli.target == "mcc":
                    ds_args.synthetic_style_alignment_pairs = False
                    test = make_dataset(ds_args, cli.mcc_samples, cli.causal, "test")
                    measures.append(block_mcc_measure(model, test, device, grid, cli))
                if decoder is not None:
                    measures.append(lambda: decoder.scores(tuple(cli.seeds), cli.n_splits))

                def measure():
                    return {k: v for m in measures for k, v in m().items()}

                base, steps, summary = step_sweep(
                    cli, named, groups, measure, decoder.slopes if decoder is not None else None
                )
                write_csv(output / "steps.csv", steps)
                write_csv(output / "steps_summary.csv", summary)
                sweep = (base, summary)
        if state_digest(model) != before:
            raise RuntimeError("Registered model state changed during attribution.")
        write_csv(output / "components.csv", rows)
        write_csv(output / "groups.csv", group_rows)
        write_csv(output / "cosines.csv", pairs)
        write_csv(output / "group_cosines.csv", group_pairs)
        write_csv(output / "modules.csv", modules + group_modules)
        write_csv(output / "parity.csv", checks)
        batch_rows = [
            dict(batch=i, component=k, weighted_loss=values[k][i], gradient_norm=float(np.linalg.norm(g)))
            for k, gradients in stacks.items()
            for i, g in enumerate(gradients)
        ]
        write_csv(output / "batches.csv", batch_rows)
        if decoding is not None:
            write_csv(output / "decoding.csv", decoding["rows"])
        metadata = dict(
            checkpoint=str(checkpoint.resolve()),
            checkpoint_step=step,
            settings=vars(args),
            arguments=vars(cli),
            actual_batch_size=batch_size,
            gradient_batches=cli.grad_batches,
            ema_window_batches=window,
            ema_reference=reference,
            model_state_unchanged=True,
            state_sha256=before,
            components=rows,
            groups=group_rows,
            cosines=pairs,
            group_cosines=group_pairs,
            parity=checks,
            train_step_diagnostics=diagnostics,
            decoding=decoding,
            steps=None if sweep is None else dict(base=sweep[0], summary=sweep[1]),
            parameter_names=[name for name, _ in named],
            caveats=caveats,
            notes=[
                "Eval mode: BN/dropout and codebook EMA are frozen; FP32, no optimizer step.",
                "EMA reference is estimated at this checkpoint, not historical training state.",
                "Instantaneous mode drops the EMA and so changes the correlation derivative.",
                "Components are disjoint; total is their sum, not an additional objective.",
                "Norms/cosines are on encoder parameters only; no full-model clipping factor is inferred.",
                "Signed projections can be negative or exceed one; they are not importance percentages.",
                "SNR estimates describe these batches; a near-zero mean does not establish harmless noise.",
                "No AdamW momentum/preconditioning, AMP, decoder updates or historical causation is reproduced.",
                "Decoding uses i.i.d. factors whatever --causal says. Its rates are first-order changes of the "
                "penalised in-sample ridge R^2 at the fitted alpha; finite steps re-measure the CV R^2 itself.",
            ],
        )
        (output / "summary.json").write_text(json.dumps(json_safe(metadata), indent=2, allow_nan=False) + "\n")
        ema = f"reference x{n_reference}" if n_reference else ("off" if not window else cli.ema_mode)
        print_report(
            f"Weighted encoder gradients: {checkpoint.name}, step {step}, B={batch_size}, "
            f"{cli.grad_batches} batches, EMA {ema}",
            rows,
            group_rows,
            pairs,
            group_pairs,
            checks,
            reference,
            caveats,
            decoding,
            sweep,
        )
        print("  Frozen eval-mode diagnostic; no checkpoint changed.")
        print(f"Saved {output}")
