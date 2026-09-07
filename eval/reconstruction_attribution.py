"""Implementation of ``eval.gradient_attribution --target reconstruction``.

The metric is masked pixel MAE, with no perceptual or VQ commitment contribution.
By default predictions are RAW, matching the current BaselineLoss pixel path;
--recon-clamp explicitly selects the older clamped metric. Per-view errors are also
reported. Reconstruction gradients are gradients of this metric, not of the full loss.

Similarity uses the run's raw features, channel masks, foreground rule, per-channel
detached variance denominator, level weight, arm weights and similarity coefficients.
Correlation centering, normalization and EMA do not enter this isolated term. The
formula is checked against barlow_twins_loss in tests/test_reconstruction_attribution.py.

Each trial is theta' = theta - eta * g, where g is a WEIGHTED similarity gradient averaged
over fixed batches. Unlike the MCC mode, directions are NOT normalized: coefficients
retain their effect. These are local raw-gradient interventions, not AdamW updates.
Gradients are compared against a disjoint, fixed evaluation set's pixel-MAE gradient.
Quantization uses a straight-through gradient, so measured MAE can disagree with its
first-order prediction. Zero finite-step changes can mean unchanged quantizer assignments.

The model runs in eval mode (deterministic masks; no codebook EMA or dropout), with only
encoder parameters differentiable. Decoder/codebook parameters and all registered buffers
are restored before every trial and on exit. No checkpoint is written. Mask-logit updates,
optimizer momentum, decoder adaptation and historical training causality are outside scope.
"""

from __future__ import annotations

import contextlib
import csv
import logging
from pathlib import Path

import numpy as np

from eval.gradient_attribution import _encoder_params, _flat, cosine

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def frozen_checkpoint(model):
    """Protect parameters, buffers, requires_grad flags and mixed module modes, even on error."""
    import torch

    tensors = list(model.parameters()) + list(model.buffers())
    saved = [t.detach().clone() for t in tensors]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    modes = [(m, m.training) for m in model.modules()]

    def restore():
        with torch.no_grad():
            for t, original in zip(tensors, saved):
                t.copy_(original)

    try:
        model.eval()
        for p, _ in flags:
            p.requires_grad_(False)
        for _, p in _encoder_params(model):
            p.requires_grad_(True)
        yield restore
    finally:
        restore()
        for p, requires_grad in flags:
            p.requires_grad_(requires_grad)
        for module, training in modes:
            module.training = training


def _inputs(batch, device):
    import torch

    images = batch["image"]
    if not isinstance(images, (list, tuple)) or len(images) != 2:
        raise ValueError("Reconstruction attribution currently requires exactly two paired views.")
    x = torch.cat(images, dim=0).to(device).float()
    masks = batch.get("mask")
    if masks is not None:
        if isinstance(masks, (list, tuple)):
            masks = torch.cat(masks, dim=0)
        else:
            masks = torch.cat([masks, masks], dim=0)
        masks = masks.to(device).float()
        if masks.ndim == x.ndim - 1:
            masks = masks.unsqueeze(1)
        if masks.shape != x.shape:
            masks = masks.expand_as(x)
    return x, masks


def similarity_loss(hz, normalize=False, patch_stat="fold"):
    """Isolated sim from barlow_twins_loss, on selected UNcentered content (2,B,C[,P])."""
    if hz.ndim == 4 and patch_stat != "per_position":
        hz = hz.permute(0, 1, 3, 2).reshape(2, -1, hz.shape[2])
    a, b = hz[0].float(), hz[1].float()
    diff = (a - b).square()
    if normalize:
        denominator = (a.var(dim=0, unbiased=False) + b.var(dim=0, unbiased=False)).detach()
        diff = diff / (denominator + 1e-8)
    return diff.mean()


def _content_features(model, batch, args_, device, grid, level):
    import torch
    import torch.nn.functional as F

    x, masks = _inputs(batch, device)
    out = model(
        x,
        return_recon=False,
        pool_only=True,
        n_views=2,
        subsets=[(0, 1)],
        patch_grid=grid,
        mask=masks,
    )
    features = out[2][level]
    hz = features.reshape(2, -1, *features.shape[1:])
    if hz.ndim != 4:
        raise ValueError("Expected patch features (2,B,C,P); this mode tests the patch + GAP companion setup.")
    if getattr(args_, "patch_foreground_mask", False) and masks is not None:
        level_grid = grid[level] if isinstance(grid[0], (tuple, list)) else grid
        frac = F.adaptive_avg_pool3d(masks, tuple(level_grid)).flatten(1)
        keep = (frac >= float(getattr(args_, "patch_foreground_thresh", 0.05))).any(0)
        if bool(keep.any()):
            hz = hz[..., keep]
    mask = out[6].get(level)
    if mask is not None:
        if isinstance(mask, tuple):
            selected = [hz[v][:, m.reshape(-1).bool(), :] for v, m in enumerate(mask)]
            if selected[0].shape != selected[1].shape:
                raise ValueError("Per-view masks must select the same number of content channels.")
            hz = torch.stack(selected)
        else:
            hz = hz[:, :, mask.reshape(-1).bool(), :]
    else:
        # No mask is safe only for an all-content level. Do not silently assume the first
        # k channels when training selects content by an activation/Gumbel mask.
        counts = getattr(model, "content_channels_per_level", {})
        if counts.get(level, hz.shape[2]) != hz.shape[2]:
            raise ValueError("No forward content mask for this partial-content level; cannot reproduce its selection.")
    if hz.shape[2] == 0:
        raise ValueError("The checkpoint selects no content channels at this level.")
    return hz


def sim_weights(args_, level):
    scale = float(getattr(args_, "scale_contrastive_loss", 1.0))
    levels = getattr(args_, "contrastive_level_weights", None)
    if levels is not None and level < len(levels):
        scale *= float(levels[level])
    patch = float(getattr(args_, "bt_sim_coeff", 0.0))
    gap = getattr(args_, "bt_gap_sim_coeff", None)
    gap = patch if gap is None else float(gap)
    return {
        "patch_sim": scale * float(getattr(args_, "bt_patch_weight", 1.0)) * patch,
        "gap_sim": scale * float(getattr(args_, "bt_gap_weight", 0.0)) * gap,
    }


def _sim_values(model, batch, args_, device, grid, level):
    hz = _content_features(model, batch, args_, device, grid, level)
    normalize = bool(getattr(args_, "bt_sim_normalize", False))
    return {
        "patch_sim": similarity_loss(hz, normalize, getattr(args_, "bt_patch_stat", "fold")),
        "gap_sim": similarity_loss(hz.mean(-1), normalize),
    }


def _gradient(loss, params):
    import torch

    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return _flat([torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)])


def collect_sim_gradients(model, loader, args_, device, grid, level, params):
    """Keep per-batch gradients so conflict frequency and batch variation remain visible."""
    weights = sim_weights(args_, level)
    grads = {k: [] for k in weights}
    values = {k: [] for k in weights}
    for i, batch in enumerate(loader):
        losses = _sim_values(model, batch, args_, device, grid, level)
        for key, loss in losses.items():
            values[key].append(float(loss.detach()))
            grads[key].append(_gradient(loss * weights[key], params).cpu())
        logger.info("Similarity gradients: batch %d/%d", i + 1, len(loader))
        del losses, loss
    import torch

    per_batch = {k: torch.stack(v) for k, v in grads.items()}
    per_batch["both_sim"] = per_batch["patch_sim"] + per_batch["gap_sim"]
    return {k: v.mean(0).to(device) for k, v in per_batch.items()}, per_batch, values


def pixel_sums(x, reconstruction, mask=None, clamp=False):
    """Differentiable error sums and voxel counts for each view; aggregate before dividing."""
    import torch

    y = reconstruction.float()
    if not bool(torch.isfinite(y).all()):
        raise ValueError("Non-finite decoder output: cannot attribute reconstruction changes reliably.")
    if clamp:
        y = y.clamp(-1.0, 1.0)
    if mask is not None:
        # BaselineLoss masks predictions before calling its masked pixel loss.
        y = y * mask
    b = x.shape[0] // 2
    sums, counts = [], []
    for v in range(2):
        sl = slice(v * b, (v + 1) * b)
        error = (x[sl] - y[sl]).abs()
        if mask is None:
            counts.append(error.new_tensor(error.numel()))
        else:
            error = error * mask[sl]
            counts.append(mask[sl].sum())
        sums.append(error.sum())
    return torch.stack(sums), torch.stack(counts)


def reconstruction_metrics(model, batches, device, params=None, clamp=False):
    """Whole-set MAE and optional gradient, accumulating sums to handle unequal masks/batches."""
    import torch
    import torch.nn.functional as F

    totals = np.zeros(2, dtype=np.float64)
    counts = np.zeros(2, dtype=np.float64)
    grad_sum = None
    context = torch.enable_grad() if params is not None else torch.no_grad()
    with context:
        for batch in batches:
            x, mask = _inputs(batch, device)
            out = model(x, return_recon=True, pool_only=True, n_views=2, subsets=[(0, 1)], mask=mask)
            y = out[0]
            if y is None:
                raise ValueError("Checkpoint has no reconstruction output.")
            if y.shape[2:] != x.shape[2:]:
                y = F.interpolate(y, size=x.shape[2:], mode="trilinear", align_corners=False)
            sums, ns = pixel_sums(x, y, mask, clamp)
            totals += sums.detach().cpu().numpy()
            counts += ns.detach().cpu().numpy()
            if params is not None:
                g = _gradient(sums.sum(), params)
                grad_sum = g if grad_sum is None else grad_sum + g
            del out, y, sums
    if np.any(counts <= 0):
        raise ValueError("Each evaluation view needs at least one foreground voxel.")
    metrics = {"mae": float(totals.sum() / counts.sum())}
    metrics.update({f"view{v}_mae": float(totals[v] / counts[v]) for v in range(2)})
    return metrics, None if grad_sum is None else grad_sum / counts.sum()


def temporary_step(params, gradient, eta):
    import torch

    if not bool(torch.isfinite(gradient).all()):
        raise ValueError("Non-finite gradient; refusing a temporary step.")
    with torch.no_grad():
        offset = 0
        for p in params:
            n = p.numel()
            p.add_(gradient[offset : offset + n].view_as(p), alpha=-eta)
            offset += n


def _validate_run(args_, level):
    if getattr(args_, "contrastive_loss_type", None) != "barlow_twins":
        raise ValueError("--target reconstruction currently isolates Barlow Twins similarity terms only.")
    if getattr(args_, "contrastive_only", False):
        raise ValueError(
            "This contrastive-only run did not train its decoder; reconstruction attribution is undefined."
        )
    if not getattr(args_, "patch_contrastive", False):
        raise ValueError("This diagnostic requires a patch-contrastive run with an optional GAP companion.")
    if (
        int(getattr(args_, "contrastive_proj_dim", 0) or 0) > 0
        or getattr(args_, "contrastive_proj_mode", "head") != "head"
    ):
        raise ValueError("Loss-facing projection/bounded/entropy transforms are not supported by this diagnostic.")
    if getattr(args_, "use_moco", False):
        raise ValueError("MoCo runs are not supported by this Barlow Twins diagnostic.")
    subsets = [tuple(s) for s in getattr(args_, "subsets", [(0, 1)])]
    if subsets != [(0, 1)]:
        raise ValueError("This diagnostic requires exactly one subset: (0, 1).")
    if not 0 <= level < int(getattr(args_, "vqvae_nb_levels", 1)):
        raise ValueError("--level is outside the model's encoder levels.")
    if getattr(args_, "freeze_encoder", False):
        logger.warning("This run freezes its encoder. Results describe hypothetical unfrozen updates only.")


def run(cli):
    import torch
    from torch.utils.data import DataLoader, Subset

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    found = False
    for name in cli.checkpoints:
        checkpoint = Path(cli.run_dir) / name
        if not checkpoint.is_file():
            logger.warning("Missing checkpoint %s; skipping", checkpoint)
            continue
        found = True
        model, args_, device = load_model_from_run_dir(cli.run_dir, str(checkpoint), seed=0)
        _validate_run(args_, cli.level)
        training_b = int(getattr(args_, "batch_size", 128))
        batch_size = cli.grad_batch_size or training_b
        if batch_size < 2:
            raise ValueError("Similarity gradient batches must contain at least two subjects.")
        if batch_size != training_b:
            logger.warning(
                "Gradient batch size %d differs from training (%d); normalized GAP gradients depend on it.",
                batch_size,
                training_b,
            )
        grid = cli.grid or getattr(args_, "patch_grid_per_level", None) or getattr(args_, "patch_grid", None)
        if not grid:
            raise ValueError("No patch grid in settings; provide --grid.")
        n_grad = cli.grad_batches * batch_size
        dataset = build_synthetic_test_set(args_, n_grad + cli.recon_samples, cache=False, causal=cli.causal == "match")
        grad_loader = DataLoader(Subset(dataset, range(n_grad)), batch_size=batch_size, shuffle=False)
        # Cache only the small held-out set, keeping identical pixels across every trial.
        eval_batches = list(
            DataLoader(
                Subset(dataset, range(n_grad, n_grad + cli.recon_samples)),
                batch_size=cli.recon_batch_size,
                shuffle=False,
            )
        )
        weights = sim_weights(args_, cli.level)
        records = []
        with frozen_checkpoint(model) as restore:
            params = [p for _, p in _encoder_params(model)]
            theta_norm = float(_flat(params).norm())
            base, g_recon = reconstruction_metrics(model, eval_batches, device, params, cli.recon_clamp)
            grads, per_batch, values = collect_sim_gradients(
                model,
                grad_loader,
                args_,
                device,
                grid,
                cli.level,
                params,
            )
            # Pixel-only control; excludes commitment/perceptual and optional view-gradient balancing.
            grads["recon_pixel_control"] = g_recon * float(getattr(args_, "scale_recon_loss", 1.0))
            print(f"\nRECONSTRUCTION ATTRIBUTION — {name}")
            print(f"  {n_grad} gradient subjects; {cli.recon_samples} disjoint evaluation subjects; eval mode.")
            print(
                f"  Pixel MAE ({'clamped' if cli.recon_clamp else 'raw'}): {base['mae']:.7f}"
                f"   view0 {base['view0_mae']:.7f}   view1 {base['view1_mae']:.7f}"
            )
            for key, weight in weights.items():
                print(f"  {key}: unweighted {np.mean(values[key]):.6g} x total coefficient {weight:g}")
            print("  cosine < 0 = conflict; -dot > 0 = predicted MAE increase per unit eta.")
            print(
                f"  {'direction':<22}{'||weighted grad||':>19}{'cos(recon)':>13}"
                f"{'-dot(recon,g)':>16}{'conflict batches':>19}"
            )
            for key, g in grads.items():
                cos = cosine(g_recon.cpu().numpy(), g.cpu().numpy())
                slope = -float(torch.dot(g_recon, g))
                if not bool(torch.isfinite(g).all()) or not np.isfinite(slope):
                    raise ValueError(f"Non-finite gradient for {key}.")
                conflict = float((per_batch[key] @ g_recon.cpu() < 0).float().mean()) if key in per_batch else None
                fraction = f"{conflict:.0%}" if conflict is not None and float(g.norm()) > 0 else "—"
                print(f"  {key:<22}{float(g.norm()):>19.5e}{cos:>13.4f}{slope:>+16.5e}{fraction:>19}")
                if float(g.norm()) == 0:
                    print("    Zero weighted gradient: no active force; skipping temporary steps.")
                    continue
                for eta in cli.etas:
                    restore()
                    temporary_step(params, g, eta)
                    logger.info("Scoring %s at eta=%g on %d held-out subjects", key, eta, cli.recon_samples)
                    measured, _ = reconstruction_metrics(model, eval_batches, device, clamp=cli.recon_clamp)
                    record = {
                        "checkpoint": name,
                        "direction": key,
                        "eta": eta,
                        "gradient_norm": float(g.norm()),
                        "cos_recon": cos,
                        "conflict_fraction": conflict,
                        "predicted_delta_mae": eta * slope,
                        "relative_parameter_step": eta * float(g.norm()) / max(theta_norm, 1e-12),
                    }
                    for metric in base:
                        record[f"baseline_{metric}"] = base[metric]
                        record[f"delta_{metric}"] = measured[metric] - base[metric]
                    records.append(record)
            restore()
            restored, _ = reconstruction_metrics(model, eval_batches, device, clamp=cli.recon_clamp)
            if any(not np.isclose(restored[k], base[k], rtol=1e-6, atol=1e-8) for k in base):
                raise RuntimeError("Restored baseline changed; the finite-step attribution is not reproducible.")
        print("\n  Temporary raw-gradient steps: POSITIVE delta MAE = worse reconstruction.")
        print(
            f"  {'direction':<22}{'eta':>10}{'|step|/|theta|':>16}{'predicted':>14}"
            f"{'delta MAE':>14}{'delta view0':>14}{'delta view1':>14}"
        )
        for row in records:
            print(
                f"  {row['direction']:<22}{row['eta']:>10.2g}{row['relative_parameter_step']:>16.3e}"
                f"{row['predicted_delta_mae']:>+14.5e}{row['delta_mae']:>+14.5e}"
                f"{row['delta_view0_mae']:>+14.5e}{row['delta_view1_mae']:>+14.5e}"
            )
        print(
            "\n  Check signs across batches and step sizes. Straight-through VQ gradients can disagree"
            "\n  with decoded MAE; zero changes may mean no quantizer boundary was crossed. Large or"
            "\n  inconsistent steps are inconclusive. This is local conflict, not a retraining counterfactual."
            "\n  Decoder/codebook state was frozen; original parameters and buffers were restored."
        )
        if cli.out and records:
            directory = Path(cli.out)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{checkpoint.stem}_reconstruction_attribution.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            logger.info("Wrote %s", path)
        del model, grads, per_batch, eval_batches, dataset
    if not found:
        raise FileNotFoundError("None of the requested checkpoints exist.")
