"""Held-out scalar recovery and finite intervention matrices.

Calibration uses validation observations only. Test interventions never choose
units, calibrations, hyperparameters, or checkpoints. Matrices include RMS as well
as signed responses so opposing effects across subjects cannot cancel silently.
"""

import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from eval.encoder.encoder_target_protocol import save_csv
from eval.lesion.checkpoint_lesion_analysis import json_safe
from models.scalar_readout import SEMANTIC_NAMES
from training.scalar_readout_data import RAW_NAMES


def r2(y, prediction):
    y, prediction = np.asarray(y, np.float64), np.asarray(prediction, np.float64)
    total = np.sum((y - y.mean(0)) ** 2, axis=0)
    return np.divide(
        total - np.sum((y - prediction) ** 2, axis=0), total, out=np.full_like(total, np.nan), where=total > 1e-12
    )


def affine_matrix(x, y, units=None, alpha=0.0):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    mean, std = x.mean(0), x.std(0)
    std = np.maximum(std, 1e-6)
    a, b = (x - mean) / std, y - y.mean(0)
    if units is None:
        gram = np.einsum("ni,nj->ij", a, a, optimize=False)
        cross = np.einsum("ni,nj->ij", a, b, optimize=False)
        coefficient = np.linalg.solve(gram + np.eye(x.shape[1]) * max(alpha, 1e-8), cross)
    else:
        coefficient = np.zeros((x.shape[1], y.shape[1]))
        for j, unit in enumerate(units):
            denominator = float(a[:, unit] @ a[:, unit])
            coefficient[unit, j] = float(a[:, unit] @ b[:, j]) / max(denominator, 1e-12)
    weight = coefficient / std[:, None]
    bias = y.mean(0) - np.einsum("i,ij->j", mean, weight, optimize=False)
    return dict(weight=weight, bias=bias, units=units, alpha=alpha)


def apply_map(x, mapping):
    return np.einsum("...i,ij->...j", x, mapping["weight"], optimize=False) + mapping["bias"]


def scalar_mapping(x, y):
    a, b = x - x.mean(0), y - y.mean(0)
    numerator = np.einsum("ni,nj->ij", a, b, optimize=False)
    denominator = np.sqrt(np.outer((a * a).sum(0), (b * b).sum(0)))
    correlation = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
    rows, cols = linear_sum_assignment(-(correlation**2))
    units = np.zeros(y.shape[1], int)
    units[cols] = rows
    return affine_matrix(x, y, units=units)


def fit_maps(x, y, seed):
    """One-to-one scalar matching; ridge tuning on an internal validation split."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    cut = max(2, min(len(x) - 2, int(0.75 * len(x))))
    fit, tune = order[:cut], order[cut:]
    candidates = []
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        fitted = affine_matrix(x[fit], y[fit], alpha=alpha)
        score = float(np.nanmean(r2(y[tune], apply_map(x[tune], fitted))))
        candidates.append((score if np.isfinite(score) else -np.inf, alpha))
    alpha = max(candidates)[1]
    return dict(
        scalar=scalar_mapping(x, y),
        vector_ridge=affine_matrix(x, y, alpha=alpha),
        shuffled_scalar=scalar_mapping(x, y[rng.permutation(len(y))]),
    )


def summarize_responses(code_pairs, reference_codes, rows):
    std = np.asarray(reference_codes, np.float64).std(0)
    normalizer = np.maximum(std, 1e-6)
    responses = []
    for row in rows:
        if not row["valid"]:
            continue
        i = row["pair_index"]
        delta = (code_pairs[i, 1] - code_pairs[i, 0]) / (2 * row["eps"]) / normalizer
        responses.append((row, delta))
    output = []
    for eps in sorted({r["eps"] for r in rows}):
        for k, factor in enumerate(RAW_NAMES):
            selected = [(r, d) for r, d in responses if r["eps"] == eps and r["factor_index"] == k]
            requested = [r for r in rows if r["eps"] == eps and r["factor_index"] == k]
            for j in range(9):
                values = np.array([d[j] for _, d in selected])
                output.append(
                    dict(
                        eps=eps,
                        factor=factor,
                        unit=j,
                        n_valid=len(values),
                        n_failed=len(requested) - len(values),
                        n_zero_image=sum(r["zero_image"] for r, _ in selected),
                        validation_unit_std=std[j],
                        constant_unit=bool(std[j] < 1e-6),
                        signed_mean=float(values.mean()) if len(values) else np.nan,
                        rms=float(np.sqrt(np.mean(values**2))) if len(values) else np.nan,
                    )
                )
    return output


def audit_arm(name, codes, observations, pair_codes, pairs, out, seed, direct=False):
    """All label access is confined to post-training evaluation / calibration."""
    recovery, movement, calibrated, mappings = [], [], {}, {}
    for family, names, key in (("semantic", SEMANTIC_NAMES, "truth"), ("raw", RAW_NAMES, "raw")):
        maps = fit_maps(codes["val"], observations["val"][key], seed)
        if direct and family == "semantic":
            maps["direct"] = dict(weight=np.eye(9), bias=np.zeros(9), units=np.arange(9), alpha=0.0)
        if family == "semantic" and "geometric" in name:
            maps["geometric_coordinates"] = dict(weight=np.eye(9), bias=np.zeros(9), units=np.arange(9), alpha=0.0)
        mappings[family] = maps
        for method, fitted in maps.items():
            pred = apply_map(codes["test"], fitted)
            truth = observations["test"][key]
            score = r2(truth, pred)
            for j, target in enumerate(names):
                if method == "geometric_coordinates" and j not in (2, 3, 4):
                    continue
                recovery.append(
                    dict(
                        arm=name,
                        family=family,
                        method=method,
                        target=target,
                        unit=int(fitted["units"][j]) if fitted["units"] is not None else None,
                        r2=score[j],
                        rmse=float(np.sqrt(np.mean((truth[:, j] - pred[:, j]) ** 2))),
                    )
                )
            prediction = apply_map(pair_codes, fitted)
            actual = pairs[key]
            calibrated[f"{family}_{method}"] = prediction
            for eps in sorted({r["eps"] for r in pairs["rows"]}):
                for k, factor in enumerate(RAW_NAMES):
                    chosen = [r for r in pairs["rows"] if r["valid"] and r["eps"] == eps and r["factor_index"] == k]
                    ids = [r["pair_index"] for r in chosen]
                    for j, target in enumerate(names):
                        if method == "geometric_coordinates" and j not in (2, 3, 4):
                            continue
                        delta = actual[ids, 1, j] - actual[ids, 0, j]
                        estimate = prediction[ids, 1, j] - prediction[ids, 0, j]
                        energy = float(np.sum(delta**2))
                        movement.append(
                            dict(
                                arm=name,
                                family=family,
                                method=method,
                                eps=eps,
                                intervention=factor,
                                target=target,
                                n=len(ids),
                                n_zero_image=sum(r["zero_image"] for r in chosen),
                                true_delta_rms=float(np.sqrt(np.mean(delta**2))) if ids else np.nan,
                                predicted_delta_rms=float(np.sqrt(np.mean(estimate**2))) if ids else np.nan,
                                delta_rmse=float(np.sqrt(np.mean((estimate - delta) ** 2))) if ids else np.nan,
                                response_skill=(
                                    1 - float(np.sum((estimate - delta) ** 2)) / energy if energy > 1e-12 else np.nan
                                ),
                            )
                        )
    responses = [dict(arm=name, **r) for r in summarize_responses(pair_codes, codes["val"], pairs["rows"])]
    serializable = {
        family: {
            method: {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in fitted.items()}
            for method, fitted in maps.items()
        }
        for family, maps in mappings.items()
    }
    (out / f"{name}_calibration.json").write_text(json.dumps(json_safe(serializable), indent=2, allow_nan=False) + "\n")
    np.savez_compressed(
        out / f"{name}_codes.npz", validation=codes["val"], test=codes["test"], intervention=pair_codes, **calibrated
    )
    return recovery, movement, responses


def write_results(out, recovery, movement, responses, pairs):
    save_csv(out / "recovery.csv", recovery)
    save_csv(out / "intervention_recovery.csv", movement)
    save_csv(out / "response_matrix.csv", responses)
    save_csv(out / "intervention_pairs.csv", pairs["rows"])
    np.savez_compressed(out / "intervention_truth.npz", raw=pairs["raw"], semantic=pairs["truth"])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = list(dict.fromkeys(r["arm"] for r in responses))
    epsilons = sorted({r["eps"] for r in responses})
    fig, axes = plt.subplots(
        len(arms), len(epsilons), squeeze=False, figsize=(5 * len(epsilons), 3.8 * len(arms)), layout="constrained"
    )
    max_value = max((r["rms"] for r in responses if np.isfinite(r["rms"])), default=1)
    for i, arm in enumerate(arms):
        for j, eps in enumerate(epsilons):
            matrix = np.full((9, 9), np.nan)
            for r in responses:
                if r["arm"] == arm and r["eps"] == eps:
                    matrix[r["unit"], RAW_NAMES.index(r["factor"])] = r["rms"]
            ax = axes[i, j]
            im = ax.imshow(matrix, vmin=0, vmax=max(max_value, 1e-6), cmap="viridis", aspect="auto")
            ax.set_xticks(range(9), RAW_NAMES, rotation=60, ha="right", fontsize=7)
            ax.set_yticks(range(9), [f"z{k}" for k in range(9)], fontsize=8)
            ax.set_title(f"{arm}, ±{eps:g}", fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="RMS finite response / validation code SD")
    fig.suptitle("Raw scalar responses: RMS avoids cancellation; zero-image and failed-pair counts are in CSV")
    fig.savefig(out / "response_matrix.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        values = [
            next(
                r["r2"]
                for r in recovery
                if r["arm"] == arm and r["family"] == "semantic" and r["method"] == "scalar" and r["target"] == target
            )
            for target in SEMANTIC_NAMES
        ]
        ax.bar(np.arange(9) - 0.4 + (i + 0.5) * width, values, width, label=arm)
    ax.set_xticks(range(9), SEMANTIC_NAMES, rotation=35, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("Held-out R² (one scalar + affine calibration)")
    ax.set_title("Units matched on validation only; no scalar mixing in this probe")
    ax.legend(fontsize=8)
    fig.savefig(out / "scalar_recovery.png", dpi=140)
    plt.close(fig)
