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

from eval.gradient_attribution import _encoder_params, _flat, cosine, linearity_check, snr_decomposition
from eval.reconstruction_attribution import frozen_checkpoint
from training.bt_objective import make_barlow_loss_functions

LOG = logging.getLogger(__name__)

VALUE_PARITY_RTOL = 2e-5
# Each component backpropagates through the encoder's large conv reductions separately, so
# float32 re-association leaves the summed component gradients ~1e-5 of their summed norms
# away from the total's (measured 9e-6 on ident-vent-hsic). The bound is 100x that noise;
# an unobserved force above 0.1% of the components' summed magnitude still fails it.
GRADIENT_PARITY_RTOL = 1e-3
GROUPS = ("content", "style", "reconstruction", "commitment", "hsic", "cross_reconstruction")


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


def mcc_sweep(cli, model, dataset, device, grid, named, groups):
    """Temporary unit steps along each group's mean gradient; state restored even on error.

    A NEGATIVE delta means stepping along that loss's descent direction lowers block-MCC.
    The excess over matched-random directions is the attribution; the raw delta also
    carries generic perturbation sensitivity.
    """
    from eval.gradient_attribution import _mcc_now

    params = [p for _, p in named]
    seeds = tuple(cli.seeds)
    base, targets = _mcc_now(model, dataset, device, grid, 0, cli.mcc_batch, None, seeds, cli.n_splits)
    backups = [p.detach().clone() for p in params]

    def restore():
        with torch.no_grad():
            for p, saved in zip(params, backups):
                p.copy_(saved)

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
                    score, _ = _mcc_now(model, dataset, device, grid, 0, cli.mcc_batch, targets, seeds, cli.n_splits)
                    curves[label].append(score - base)
                    steps.append(
                        dict(group=key, direction=label, eta=eta, base_mcc=base, mcc=score, delta_mcc=score - base)
                    )
                restore()
            controls = [v for k, v in curves.items() if k != "gradient"]
            control = np.mean(controls, axis=0) if controls else np.full(len(cli.etas), np.nan)
            r2 = linearity_check(cli.etas, curves["gradient"])
            for i, eta in enumerate(cli.etas):
                summary.append(
                    dict(
                        group=key,
                        eta=eta,
                        delta_mcc=curves["gradient"][i],
                        random_delta_mcc=float(control[i]),
                        excess_over_random=curves["gradient"][i] - float(control[i]),
                        linearity_r2=r2,
                    )
                )
            LOG.info(
                "MCC %s: delta %s, matched random %s, linearity R2 %.3f", key, curves["gradient"], control.tolist(), r2
            )
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


def print_report(title, rows, group_rows, pairs, group_pairs, checks, reference, caveats, mcc=None):
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

    if mcc is not None:
        base, summary = mcc
        print(f"\n  block-MCC {base:.4f}; delta after a unit step along -g/|g| (negative = degrades):")
        for group in dict.fromkeys(r["group"] for r in summary):
            rs = [r for r in summary if r["group"] == group]
            print(
                f"    {group:<22s}"
                + "".join(f"{r['delta_mcc']:+10.4f}" for r in rs)
                + f"   linearity R2 {_fmt(rs[0]['linearity_r2'], '.3f')}"
            )
            print(f"    {'  matched random':<22s}" + "".join(f"{_fmt(r['random_delta_mcc'], '+10.4f')}" for r in rs))
            print(
                f"    {'  EXCESS (attribution)':<22s}"
                + "".join(f"{_fmt(r['excess_over_random'], '+10.4f')}" for r in rs)
            )

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

    from eval.lesion_reconstruction import json_safe
    from eval.run_dci_compare import load_contrastive_proj_heads
    from eval.run_dci_synthetic import load_model_from_run_dir
    from eval.ventricle_routing import make_dataset, state_digest

    torch.set_num_threads(cli.threads)
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
        before = state_digest(model)
        output = base_output / checkpoint.stem if len(checkpoints) > 1 else base_output
        output.mkdir(parents=True, exist_ok=False)
        mcc = None
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
            if cli.target == "mcc" and not cli.snr:
                ds_args.synthetic_style_alignment_pairs = False
                test = make_dataset(ds_args, cli.mcc_samples, cli.causal, "test")
                base, steps, sweep = mcc_sweep(cli, model, test, device, grid, named, groups)
                write_csv(output / "mcc_steps.csv", steps)
                write_csv(output / "mcc_summary.csv", sweep)
                mcc = (base, sweep)
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
            mcc=None if mcc is None else dict(base=mcc[0], sweep=mcc[1]),
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
            mcc,
        )
        print("  Frozen eval-mode diagnostic; no checkpoint changed.")
        print(f"Saved {output}")
