"""Held-out scalar semantics and actual finite-intervention tracking."""

import json

import numpy as np
import torch

from eval.encoder.encoder_lesion_intervention import movement_metrics
from eval.encoder.encoder_target_protocol import save_csv
from eval.encoder.scalar_readout_audit import affine_matrix, apply_map, r2
from eval.lesion.checkpoint_lesion_analysis import json_safe
from models.local_scalar_readout import frame
from models.scalar_readout import coordinates
from training.scalar_readout_data import RAW_NAMES

NAMES = (
    *RAW_NAMES,
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "sulcal_amplitude",
    "brain_centroid_x",
    "brain_centroid_y",
    "brain_centroid_z",
)


def targets(bank, view, grid, resolution):
    support = torch.from_numpy(np.array(bank["support"][..., view, :, :, :, :])).float()
    leading = support.shape[:-4]
    centre, spread = frame(support.reshape(-1, *support.shape[-4:]), coordinates(grid, resolution))
    centroid = bank["truth"][..., 2:5]
    brain = (centroid - centre.numpy().reshape(*leading, 3)) / spread.numpy().reshape(*leading, 3)
    return np.concatenate((bank["raw"], centroid, bank["truth"][..., 8:9], brain), -1)


def ridge_map(x, y, seed):
    order = np.random.default_rng(seed).permutation(len(x))
    cut = max(2, min(len(x) - 2, int(len(x) * 0.75)))
    candidates = []
    for alpha in (0.01, 0.1, 1, 10, 100):
        mapping = affine_matrix(x[order[:cut]], y[order[:cut]], alpha=alpha)
        score = r2(y[order[cut:]], apply_map(x[order[cut:]], mapping))
        candidates.append((float(np.nanmean(score)), alpha))
    return affine_matrix(x, y, alpha=max(candidates)[1])


def single_map(x, y):
    a, b = x - x.mean(0), y - y.mean(0)
    cross = np.einsum("ni,nj->ij", a, b, optimize=False)
    denominator = np.outer((a * a).sum(0), (b * b).sum(0))
    score = np.divide(cross**2, denominator, out=np.zeros_like(cross), where=denominator > 1e-12)
    return affine_matrix(x, y, units=score.argmax(0))


def evaluate_arm(arm, codes, banks, args, directory):
    recovery, interventions, movements, response, localization = [], [], [], [], []
    mappings, all_predictions = {}, {}
    pair_rows = banks["pairs"]["rows"]
    valid_rows = sorted((r for r in pair_rows if r["valid"]), key=lambda r: r["pair_index"])
    if [r["pair_index"] for r in valid_rows] != list(range(len(codes["pairs"]["code"]))):
        raise ValueError("Intervention code and metadata order differ")
    for v, view in enumerate(args.views):
        y = {s: targets(b, v, args.grid, args.resolution) for s, b in banks.items()}
        x = {s: c["code"][..., v, :] for s, c in codes.items()}
        perm = np.random.default_rng(args.seed).permutation(len(y["val"]))
        maps = {
            "joint_ridge": (
                list(range(len(NAMES))),
                ridge_map(x["val"], y["val"], args.seed),
            ),
            "best_single_scalar": (
                list(range(len(NAMES))),
                single_map(x["val"], y["val"]),
            ),
            "shuffled_joint": (
                list(range(len(NAMES))),
                ridge_map(x["val"], y["val"][perm], args.seed),
            ),
            "shuffled_scalar": (
                list(range(len(NAMES))),
                single_map(x["val"], y["val"][perm]),
            ),
        }
        predictions = {
            name: (indices, {s: apply_map(x[s], mapping) for s in x}) for name, (indices, mapping) in maps.items()
        }
        selected = None
        if "physical" in codes["val"]:
            val_centres = codes["val"]["physical"][:, v]
            errors = ((val_centres - y["val"][:, None, 9:12]) ** 2).mean((0, 2))
            selected = int(errors.argmin())
            for head in range(val_centres.shape[1]):
                native = {
                    s: np.concatenate(
                        (
                            c["physical"][..., v, head, :],
                            c["relative"][..., v, head, :],
                        ),
                        -1,
                    )
                    for s, c in codes.items()
                }
                predictions[f"native_head_{head}"] = ([9, 10, 11, 13, 14, 15], native)
                error = np.linalg.norm(native["test"][:, :3] - y["test"][:, 9:12], axis=-1) * (args.resolution - 1) / 2
                localization.append(
                    dict(
                        arm=arm,
                        view=view,
                        head=head,
                        selected_on_validation=head == selected,
                        mean_error_vox=float(error.mean()),
                        median_error_vox=float(np.median(error)),
                    )
                )
            predictions["selected_native_head"] = predictions[f"native_head_{selected}"]
            mapping = affine_matrix(
                x["val"],
                y["val"][:, [8, 12]],
                units=np.array([x["val"].shape[-1] - 1] * 2),
            )
            maps["amplitude_scalar"] = ([8, 12], mapping)
            predictions["amplitude_scalar"] = (
                [8, 12],
                {s: apply_map(x[s], mapping) for s in x},
            )
            if arm.endswith("oracle"):
                predictions["direct_amplitude"] = (
                    [12],
                    {s: c["amplitude"][..., v, :] for s, c in codes.items()},
                )
        thresholds = np.quantile(banks["val"]["contrast"][:, v], [0.25, 0.75])
        contrast = banks["test"]["contrast"][:, v]
        groups = {
            "all": np.ones(len(contrast), bool),
            "low_contrast": contrast <= thresholds[0],
            "middle_contrast": (contrast > thresholds[0]) & (contrast < thresholds[1]),
            "high_contrast": contrast >= thresholds[1],
        }
        for method, (indices, predicted) in predictions.items():
            for group, mask in groups.items():
                if mask.sum() < 2:
                    continue
                actual, estimate = y["test"][mask][:, indices], predicted["test"][mask]
                scores = r2(actual, estimate)
                for j, target in enumerate(indices):
                    rmse = float(np.sqrt(np.mean((actual[:, j] - estimate[:, j]) ** 2)))
                    recovery.append(
                        dict(
                            arm=arm,
                            view=view,
                            method=method,
                            group=group,
                            target=NAMES[target],
                            n=int(mask.sum()),
                            r2=scores[j],
                            rmse=rmse,
                            rmse_vox=(rmse * (args.resolution - 1) / 2 if target in (9, 10, 11) else None),
                        )
                    )
            for eps in sorted({r["eps"] for r in pair_rows}):
                for k, factor in enumerate(RAW_NAMES):
                    chosen = [r for r in valid_rows if r["eps"] == eps and r["factor_index"] == k]
                    ids = [r["pair_index"] for r in chosen]
                    if not ids:
                        continue
                    actual = y["pairs"][ids, 1][:, indices] - y["pairs"][ids, 0][:, indices]
                    estimate = predicted["pairs"][ids, 1] - predicted["pairs"][ids, 0]
                    for j, target in enumerate(indices):
                        energy = float((actual[:, j] ** 2).sum())
                        interventions.append(
                            dict(
                                arm=arm,
                                view=view,
                                method=method,
                                eps=eps,
                                intervention=factor,
                                target=NAMES[target],
                                n=len(ids),
                                n_zero_image=sum(r["zero_image"] for r in chosen),
                                true_delta_rms=float(np.sqrt(np.mean(actual[:, j] ** 2))),
                                predicted_delta_rms=float(np.sqrt(np.mean(estimate[:, j] ** 2))),
                                response_skill=(
                                    1 - float(((estimate[:, j] - actual[:, j]) ** 2).sum()) / energy
                                    if energy > 1e-12
                                    else None
                                ),
                            )
                        )
                if all(k in indices for k in (9, 10, 11)) and (
                    method in ("joint_ridge", "shuffled_joint", "selected_native_head")
                ):
                    chosen = [r for r in valid_rows if r["eps"] == eps and r["factor_index"] in (2, 3, 4)]
                    ids = [r["pair_index"] for r in chosen]
                    if ids:
                        columns = [indices.index(k) for k in (9, 10, 11)]
                        metrics = movement_metrics(
                            y["pairs"][ids, :, 9:12],
                            predicted["pairs"][ids][:, :, columns],
                            [r["subject_id"] for r in chosen],
                            (args.resolution - 1) / 2,
                            args.bootstrap,
                            args.seed,
                        )
                        movements.append(dict(arm=arm, view=view, method=method, eps=eps, **metrics))
            all_predictions[f"{view}_{method}"] = predicted["pairs"]
        std = x["val"].std(0)
        for eps in sorted({r["eps"] for r in pair_rows}):
            for k, factor in enumerate(RAW_NAMES):
                requested = [r for r in pair_rows if r["eps"] == eps and r["factor_index"] == k]
                chosen = [r for r in requested if r["valid"]]
                ids = [r["pair_index"] for r in chosen]
                delta = (x["pairs"][ids, 1] - x["pairs"][ids, 0]) / (2 * eps) / np.maximum(std, 1e-6)
                for unit in range(x["val"].shape[-1]):
                    response.append(
                        dict(
                            arm=arm,
                            view=view,
                            eps=eps,
                            intervention=factor,
                            unit=unit,
                            validation_std=std[unit],
                            constant_unit=bool(std[unit] < 1e-6),
                            signed_mean=float(delta[:, unit].mean()) if ids else None,
                            rms=(float(np.sqrt(np.mean(delta[:, unit] ** 2))) if ids else None),
                            n_valid=len(ids),
                            n_failed=len(requested) - len(ids),
                            n_zero_image=sum(r["zero_image"] for r in chosen),
                        )
                    )
        mappings[view] = dict(
            selected_head=selected,
            contrast_quartiles=thresholds.tolist(),
            maps={
                name: dict(
                    indices=indices,
                    **{k: value.tolist() if isinstance(value, np.ndarray) else value for k, value in mapping.items()},
                )
                for name, (indices, mapping) in maps.items()
            },
        )
    (directory / f"{arm}_calibration.json").write_text(
        json.dumps(json_safe(mappings), indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        directory / f"{arm}_codes.npz",
        **{f"{s}_{k}": value for s, c in codes.items() for k, value in c.items()},
        **all_predictions,
    )
    return dict(
        recovery=recovery,
        intervention_recovery=interventions,
        movement=movements,
        response_matrix=response,
        localization=localization,
    )


def write_results(directory, results, banks, args):
    for key, rows in results.items():
        save_csv(directory / f"{key}.csv", rows)
    save_csv(directory / "intervention_pairs.csv", banks["pairs"]["rows"])
    np.savez_compressed(
        directory / "truth.npz",
        target_names=np.array(NAMES),
        **{
            f"{split}_targets_{view}": targets(bank, v, args.grid, args.resolution)
            for split, bank in banks.items()
            for v, view in enumerate(args.views)
        },
        **{
            f"{split}_{key}": bank[key]
            for split, bank in banks.items()
            for key in ("truth", "raw", "contrast")
            if key in bank
        },
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = results["recovery"]
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    views = list(dict.fromkeys(r["view"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    for v, view in enumerate(views):
        for ax, target, method in (
            (axes[0], "centroid_x", "joint_ridge"),
            (axes[1], "sulcal_amplitude", "amplitude_scalar"),
        ):
            values = []
            for arm in arms:
                selected = [
                    r["r2"]
                    for r in rows
                    if r["arm"] == arm
                    and r["view"] == view
                    and r["group"] == "all"
                    and r["method"]
                    == ("best_single_scalar" if arm == "original_global" and method == "amplitude_scalar" else method)
                    and (r["target"].startswith("centroid_") if target == "centroid_x" else r["target"] == target)
                ]
                values.append(float(np.mean(selected)) if selected else np.nan)
            ax.bar(
                np.arange(len(arms)) + (v - (len(views) - 1) / 2) * 0.35,
                values,
                width=0.35,
                label=view,
            )
    for ax, title in zip(
        axes,
        (
            "Physical centroid: joint ridge mean R²",
            "Signed amplitude: one scalar + affine R²",
        ),
    ):
        ax.set_xticks(range(len(arms)), arms, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(title)
        ax.legend()
    fig.savefig(directory / "recovery.png", dpi=140)
    plt.close(fig)
