"""Checkpoint-free anatomy/pooling and Barlow Twins sensitivity experiment.

Real renderer interventions measure input pooling loss. Controlled, already-aligned
feature tensors then test the shipped BT objective. These are optimistic oracle
features, NOT encoder outputs or evidence about a trained representation.
See PATCH_SIGNAL_AUDIT.md. No model, optimizer, or checkpoint is loaded.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval.lesion_reconstruction import json_safe, make_dataset
from eval.lesion_reconstruction import render_pair as lesion_pair
from eval.ventricle_routing import render_pair as ventricle_pair

LOG = logging.getLogger(__name__)
DEFAULTS = dict(
    synthetic_mode="pseudo_mri",
    synthetic_res=64,
    synthetic_n_content=9,
    synthetic_clean_content=True,
    synthetic_identifiable_ventricle=True,
    synthetic_normalize="fixed_reference",
    synthetic_causal=False,
    patch_foreground_mask=True,
    patch_foreground_thresh=0.05,
    patch_center_mode="position",
    bt_patch_stat="fold",
    bt_lambda=6.0,
    bt_sim_coeff=0.0114,
    bt_std_coeff=0.227,
    bt_sim_normalize=False,
    bt_normalize_terms=True,
    bt_corr_ema=0.99,
    bt_patch_weight=1.0,
    bt_gap_weight=1.0,
    bt_gap_pooling="gap",
    scale_contrastive_loss=100.0,
)


def pool(volume, grid):
    return F.adaptive_avg_pool3d(volume, (grid,) * 3).flatten(1)


def collect_batch(ds, indices, factor, eps, grids, threshold, foreground_mask):
    """Input-space energy comparisons use block-volume weighting, not vector norms.

    For integer nonoverlapping bins, lift(pool(delta)) is an orthogonal projection:
    retained squared energy is in [0,1]. Masks are fixed from the original subject.
    """
    delta, masks, scales = [], [], []
    for index in indices:
        if factor == "ventricle":
            pair = ventricle_pair(ds, index, eps)
            a, b, mask = pair["a"], pair["b"], pair["mask"]
            support = pair["support"]
        else:
            b, a, mask, support, _, _ = lesion_pair(ds, index)
        # Exclude floating-point normalization residuals outside the intervention's
        # true support, dilated for the renderer's 3³ blur.
        roi = F.max_pool3d(torch.as_tensor(support).float()[None, None], 3, 1, 1)[0] * mask
        delta.append(torch.stack([(y - x) * roi for x, y in zip(a, b)]))
        masks.append(mask)
        scales.append([float(y[mask > 0].std(unbiased=False)) for y in b])
    # V,B,1,X,Y,Z. Original foreground contrast sets the oracle's dimensionless scale.
    delta = torch.stack(delta).transpose(0, 1)
    masks = torch.stack(masks)
    scales = torch.tensor(scales).T.clamp_min(1e-8)
    rows, features = [], {}
    for grid in grids:
        fractions = pool(masks, grid)
        keep = (fractions >= threshold).any(0) if foreground_mask else torch.ones(grid**3, dtype=torch.bool)
        if not keep.any():  # Same fallback as training/main_multimodal.py.
            keep[:] = True
        bin_volume = (ds.res // grid) ** 3
        for v, modality in enumerate(("t1", "flair")):
            signed = pool(delta[v], grid)
            abs_mean = pool(delta[v].abs(), grid)
            energy = delta[v].square().flatten(1).sum(1)
            pooled_energy = signed.square().sum(1) * bin_volume
            kept_energy = signed[:, keep].square().sum(1) * bin_volume
            abs_energy = abs_mean.square().sum(1) * bin_volume
            for j, index in enumerate(indices):
                valid = float(energy[j]) > 1e-12
                rows.append(
                    dict(
                        subject=index,
                        factor=factor,
                        view=modality,
                        grid=grid,
                        valid_input=valid,
                        kept_positions=int(keep.sum()),
                        total_positions=grid**3,
                        affected_kept_fraction=float((abs_mean[j, keep] > 1e-8).float().mean()),
                        input_delta_rms=float(delta[v, j].square().mean().sqrt()),
                        pooled_energy_retained=float(pooled_energy[j] / energy[j]) if valid else np.nan,
                        masked_energy_retained=float(kept_energy[j] / energy[j]) if valid else np.nan,
                        mask_keeps_pooled_energy=(
                            float(kept_energy[j] / pooled_energy[j]) if pooled_energy[j] > 1e-16 else np.nan
                        ),
                        signed_to_absolute_pool_energy=(
                            float(pooled_energy[j] / abs_energy[j]) if abs_energy[j] > 1e-16 else np.nan
                        ),
                    )
                )
            features[grid, modality] = (signed / scales[v, :, None], abs_mean > 1e-8, keep)
    return rows, features


def planted_features(signal, channels, layout, strength, seed):
    """Ideal shared target in channel 0; unrelated shared features fill other channels.

    Mixed: nuisance also occupies channel 0. Dedicated: channel 0 contains only
    target. A subject-level sign provides controlled variation independently of
    anatomy. Both views use the SAME modality-derived template, not raw T1/FLAIR.
    """
    generator = torch.Generator().manual_seed(seed)
    b, p = signal.shape
    common = 0.5 * torch.randn(b, channels, 1, generator=generator)
    common = common + np.sqrt(0.75) * torch.randn(b, channels, p, generator=generator)
    if layout == "dedicated":
        common[:, 0] = 0
    elif layout != "mixed":
        raise ValueError("Unknown feature layout")
    signs = torch.randint(0, 2, (b, 1), generator=generator) * 2 - 1
    target = torch.zeros_like(common)
    target[:, 0] = signal * signs * strength
    # Nonzero cyclic shift guarantees that no subject keeps its own target donor.
    shift = int(torch.randint(1, b, (), generator=generator))
    return common, target, target.roll(shift, dims=0)


def loss_settings(settings, arm):
    gap = arm == "gap"

    def coefficient(name):
        override = settings.get("bt_gap_" + name) if gap else None
        return settings["bt_" + name] if override is None else override

    return dict(
        lambd=coefficient("lambda"),
        sim_coeff=coefficient("sim_coeff"),
        std_coeff=coefficient("std_coeff"),
        sim_normalize=settings["bt_sim_normalize"],
        normalize_terms=settings["bt_normalize_terms"],
        center_mode="none" if gap else settings["patch_center_mode"],
        patch_stat=settings["bt_patch_stat"],
        sim_whiten=bool(settings.get("bt_sim_whiten", False)) if gap else False,
        sim_whiten_eps=settings.get("bt_sim_whiten_eps", 1e-3),
    )


def measure_loss(common, target, shuffled, keep, settings, arm):
    """Paired counterfactuals at fixed nuisance, with an explicit EMA reference.

    Instantaneous deltas are fully recomputed losses. EMA measurements clone a
    converged matched-reference correlation for EACH condition; they represent
    one hypothetical step after that history, not separate trained trajectories.
    """
    from training.losses import barlow_twins_loss, stats_pool

    kwargs = loss_settings(settings, arm)
    c, t, s = [x[..., keep] for x in (common, target, shuffled)]

    def evaluate(x, y, state=None, decay=0.0):
        hz = torch.stack((x, y))
        if arm == "gap":
            hz = stats_pool(hz)[0] if settings["bt_gap_pooling"] == "stats" else hz.mean(-1)
        # Explicit indices are essential for the per_position C,P layout.
        return barlow_twins_loss(
            hz,
            estimated_content_indices=[list(range(hz.shape[2]))],
            subsets=[(0, 1)],
            corr_ema=state,
            corr_ema_decay=decay,
            **kwargs,
        )

    matched = evaluate(c + t, c + t)
    mismatch = evaluate(c + t, c + s)
    removed = evaluate(c, c)
    base_diag = matched._contrastive_diag
    out = dict(
        matched_loss=float(matched),
        mismatch_loss=float(mismatch),
        both_removed_loss=float(removed),
        mismatch_delta=float(mismatch - matched),
        both_removed_delta=float(removed - matched),
        target_energy_share=float(t.square().sum() / (t.square().sum() + c.square().sum()).clamp_min(1e-20)),
        selected_positions=int(keep.sum()),
        signal_nonzero_subjects=int((t.square().flatten(1).sum(1) > 1e-16).sum()),
    )
    for key in ("on_diag_loss", "off_diag_loss", "sim_loss", "var_loss"):
        out["matched_" + key] = base_diag[key]
        out["mismatch_delta_" + key] = mismatch._contrastive_diag[key] - base_diag[key]
        out["removed_delta_" + key] = removed._contrastive_diag[key] - base_diag[key]

    def derivative(state=None, decay=0.0):
        r = torch.tensor(1.0, requires_grad=True)
        loss = evaluate(c + t, c + t + r * (s - t), copy.deepcopy(state), decay)
        return float(torch.autograd.grad(loss, r)[0])

    out["mismatch_directional_gradient"] = derivative()
    decay = float(settings["bt_corr_ema"] or 0)
    if decay:
        state = {}
        with torch.no_grad():
            evaluate(c + t, c + t, state, decay)
        # Install an explicit asymptotic matched history, including bias correction.
        for value in state.values():
            value["c"] = value["c"] / (1 - decay)
            value["t"] = max(1000, int(np.ceil(40 / (1 - decay))))
        ema_matched = evaluate(c + t, c + t, copy.deepcopy(state), decay)
        ema_mismatch = evaluate(c + t, c + s, copy.deepcopy(state), decay)
        out["ema_one_step_mismatch_delta"] = float(ema_mismatch - ema_matched)
        out["ema_one_step_directional_gradient"] = derivative(state, decay)
    weight = settings["bt_gap_weight" if arm == "gap" else "bt_patch_weight"]
    out["training_arm_multiplier"] = weight * settings["scale_contrastive_loss"]
    out["weighted_mismatch_delta"] = out["mismatch_delta"] * out["training_arm_multiplier"]
    return out


def run_experiment(ds, cli, settings):
    geometry, losses = [], []
    for start in range(0, len(ds), cli.batch_size):
        indices = list(range(start, min(start + cli.batch_size, len(ds))))
        for factor in ("ventricle", "lesion"):
            rows, features = collect_batch(
                ds,
                indices,
                factor,
                cli.eps,
                cli.grids,
                settings["patch_foreground_thresh"],
                settings["patch_foreground_mask"],
            )
            geometry.extend(rows)
            for (grid, view), (signal, affected, foreground) in features.items():
                # ROI is batch-union touched positions; it never changes subject rows.
                scopes = {"training_mask": foreground, "affected_union_oracle": foreground & affected.any(0)}
                for strength in cli.strengths:
                    for layout in ("mixed", "dedicated"):
                        common, target, shuffled = planted_features(
                            signal, cli.channels, layout, strength, cli.seed + start
                        )
                        for scope, keep in scopes.items():
                            if int(keep.sum()) < 2:
                                continue
                            for arm in ("patch", "gap"):
                                row = dict(
                                    batch_start=start,
                                    subjects=len(indices),
                                    factor=factor,
                                    view=view,
                                    grid=grid,
                                    strength=strength,
                                    layout=layout,
                                    scope=scope,
                                    arm=arm,
                                )
                                row.update(measure_loss(common, target, shuffled, keep, settings, arm))
                                losses.append(row)
            LOG.info("Scored %s, subjects %d:%d", factor, start, start + len(indices))
    return geometry, losses


def aggregate(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for key, group in groups.items():
        record = dict(zip(keys, key))
        record["n_rows"] = len(group)
        for field in group[0]:
            if field in keys or field in ("subject", "batch_start"):
                continue
            values = np.asarray([r[field] for r in group], dtype=float)
            finite = values[np.isfinite(values)]
            record[field] = dict(
                mean=float(finite.mean()) if len(finite) else None,
                min=float(finite.min()) if len(finite) else None,
                max=float(finite.max()) if len(finite) else None,
                n_valid=len(finite),
            )
        result.append(record)
    return result


def density_controls(cli, settings):
    """Vary occupied positions at fixed active-patch amplitude and feature width."""
    rows = []
    for occupied in (1, 8, 64, 512):
        signal = torch.zeros(min(cli.num_samples, cli.batch_size), 512)
        signal[:, :occupied] = 1
        for layout in ("mixed", "dedicated"):
            common, target, shuffled = planted_features(signal, cli.channels, layout, 1, cli.seed)
            for arm in ("patch", "gap"):
                row = dict(occupied_positions=occupied, total_positions=512, layout=layout, arm=arm)
                row.update(measure_loss(common, target, shuffled, torch.ones(512, dtype=torch.bool), settings, arm))
                rows.append(row)
    return rows


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--settings", type=Path, help="Optional settings.json; no checkpoint is opened.")
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grids", type=int, nargs="+", default=[8], help="Cubic grids, e.g. --grids 4 8 16.")
    p.add_argument("--strengths", type=float, nargs="+", default=[1.0], help="Oracle target amplitude multipliers.")
    p.add_argument("--channels", type=int, default=12, help="Controlled feature width, not an encoder width claim.")
    p.add_argument("--eps", type=float, default=0.25, help="Ventricle z_content[1] low/high half step.")
    p.add_argument(
        "--seed", type=int, default=0, help="Oracle nuisance/sign/shuffle seed; renderer uses saved split seeds."
    )
    p.add_argument("--causal", choices=["iid", "match"], default="iid")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--center-mode", choices=["none", "position", "double"])
    p.add_argument("--patch-stat", choices=["fold", "per_position"])
    p.add_argument("--corr-ema", type=float)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--out-dir", type=Path)
    return p


def main(cli=None):
    cli = parser().parse_args() if cli is None else cli
    settings = DEFAULTS.copy()
    if cli.settings:
        settings.update(json.loads(cli.settings.read_text()))
    for name, key in (
        ("center_mode", "patch_center_mode"),
        ("patch_stat", "bt_patch_stat"),
        ("corr_ema", "bt_corr_ema"),
    ):
        if getattr(cli, name) is not None:
            settings[key] = getattr(cli, name)
    if cli.num_samples < 2 or cli.batch_size < 2 or cli.num_samples % cli.batch_size == 1:
        raise ValueError("Every batch needs at least two subjects, including the last batch.")
    if cli.channels < 2 or cli.cpu_threads < 1 or not np.isfinite(cli.eps) or cli.eps <= 0:
        raise ValueError("Need channels >= 2, positive threads and finite positive eps.")
    if any(not np.isfinite(s) or s < 0 for s in cli.strengths):
        raise ValueError("Strengths must be finite and nonnegative.")
    if not 0 <= float(settings["bt_corr_ema"] or 0) < 1:
        raise ValueError("Correlation EMA must be in [0,1).")
    if settings["bt_gap_pooling"] not in ("gap", "stats"):
        raise ValueError("Supported global pooling: gap or stats.")
    if settings["patch_center_mode"] not in ("none", "position", "double") or settings["bt_patch_stat"] not in (
        "fold",
        "per_position",
    ):
        raise ValueError("Unsupported patch centering/statistic.")
    directory = cli.out_dir or Path("results") / datetime.now().strftime("patch_signal_audit_%Y%m%d_%H%M%S_%f")
    if directory.exists():
        raise FileExistsError(directory)
    torch.set_num_threads(cli.cpu_threads)
    ds = make_dataset(argparse.Namespace(**settings), cli.num_samples, cli.causal, cli.split)
    if any(g < 1 or g > ds.res or ds.res % g for g in cli.grids):
        raise ValueError("Each grid must divide rendered resolution exactly (nonoverlapping pooling bins).")
    LOG.info("No checkpoint: %d³ renderer; grids=%s, feature channels=%d", ds.res, cli.grids, cli.channels)
    geometry, losses = run_experiment(ds, cli, settings)
    controls = density_controls(cli, settings)
    if not losses:
        raise ValueError("No eligible patch sets: need at least two selected positions.")
    report = dict(
        protocol=__doc__,
        settings=settings,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()},
        render_resolution=ds.res,
        geometry=aggregate(geometry, ["factor", "view", "grid"]),
        loss=aggregate(losses, ["factor", "view", "grid", "strength", "layout", "scope", "arm"]),
        density_control=controls,
        limitations=[
            "No learned encoder, projector, quantizer, reconstruction or style pathway is modelled.",
            "Each modality supplies a separate spatial template copied into BOTH ideal aligned views.",
            "Target magnitude is input change divided by original-on foreground contrast, then scaled by --strengths.",
            "Nuisance has unit expected variance: 1/4 subject-global, 3/4 subject-position; it is shared perfectly across views.",
            "Affected-union selection uses intervention knowledge for an oracle diagnostic, never a proposed unsupervised training mask.",
            "Loss contrasts are nonlinear and coupled across patches; they are not additive per-patch loss attribution.",
            "Positive mismatch gradient means decreasing the explicit mismatch locally decreases this oracle loss.",
            "Zero penalty for deleting a factor can occur when other factors keep channels noncollapsed.",
            "EMA is a single hypothetical step from a converged matched history, not a fitted training trajectory.",
            "Batch min/max are descriptive; they are not confidence intervals or independent patch samples.",
        ],
    )
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    for name, rows in (("geometry", geometry), ("loss", losses), ("density_control", controls)):
        with (directory / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("\nINPUT POOLING: mean retained squared energy; 1=all, 0=none")
    for r in report["geometry"]:

        def show(key):
            value = r[key]["mean"]
            return "undefined" if value is None else f"{value:.4g}"

        print(
            f"{r['factor']:9s} {r['view']:5s} grid={r['grid']:2d} "
            f"pooled={show('pooled_energy_retained')} masked={show('masked_energy_retained')} "
            f"affected/kept={show('affected_kept_fraction')}"
        )
    print("\nBT ORACLE: Δshuffle=L(mismatched)-L(matched); Δdrop=L(both removed)-L(matched)")
    for r in report["loss"]:
        print(
            f"{r['factor']:9s} {r['view']:5s} g={r['grid']:2d} strength={r['strength']:g} "
            f"{r['layout']:9s} {r['scope']:21s} {r['arm']:5s} "
            f"Δshuffle={r['mismatch_delta']['mean']:+.5g} Δdrop={r['both_removed_delta']['mean']:+.5g} "
            f"dL/dr={r['mismatch_directional_gradient']['mean']:+.5g}"
        )
    print("\nDENSITY CONTROL: fixed active-patch amplitude, varying occupied positions out of 512")
    for r in controls:
        print(
            f"K={r['occupied_positions']:3d} {r['layout']:9s} {r['arm']:5s} "
            f"Δshuffle={r['mismatch_delta']:+.5g} Δsim={r['mismatch_delta_sim_loss']:+.5g} "
            f"Δdrop={r['both_removed_delta']:+.5g}"
        )
    print(f"\nControlled loss sensitivity only; no claim about a learned encoder. Saved {directory}")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
