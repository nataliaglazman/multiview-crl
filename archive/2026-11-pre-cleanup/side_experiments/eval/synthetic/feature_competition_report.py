"""Held-out readouts, spatial image controls, and paired competition summaries."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from training.feature_competition_data import (
    ARMS,
    BRAIN_RULE,
    COMMON_ARM,
    VIEWS,
    bootstrap_interval,
    file_hash,
    images,
    load_bank,
    targets,
    write_json,
)
from training.feature_competition_train import build_model, run_path, torch_load
from utils.encoder_runtime import select_encoder_device

# Reference arm -> arms compared with it. A-B is the competition penalty; A-Bp separates input
# interference (competitor variation in the images) from competition through the loss (B vs Bp).
COMPARISONS = {"A": ("B", "Bp"), "C": ("D",)}
# Frozen-feature readouts: the 9 content units the loss acts on, the backbone map averaged over all positions
# (the vector the global head reads), and the map pooled to a grid. Probing the pooled vector shows what the
# backbone has learned before the head passes it on (in the arm-A pilot it led the head by ~1,000 steps).
SOURCES = ("global", "pooled", "spatial")


def hemisphere_split(centres_x):
    """Midpoint of the widest gap between admissible lesion-centre x values when they form two groups, else None.

    At the default geometry the centres sit at x 22-26 or 37-41, so x is mostly which hemisphere the lesion is in.
    """
    values = np.unique(np.asarray(centres_x, dtype=float))
    if len(values) < 2:
        return None
    gaps = np.diff(values)
    widest = int(gaps.argmax())
    return float(values[widest] + gaps[widest] / 2) if gaps[widest] > 2 else None


def metrics(truth, prediction, split=None):
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 3:
        raise ValueError("Expected matching N by 3 coordinate arrays")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Non-finite coordinates")
    variance = ((truth - truth.mean(0)) ** 2).sum(0)
    r2 = np.full(3, np.nan)
    active = variance > 1e-12
    r2[active] = 1 - ((prediction - truth) ** 2).sum(0)[active] / variance[active]
    error = np.linalg.norm(truth - prediction, axis=1)
    result = dict(
        r2=r2.tolist(),
        mean_r2=float(np.nanmean(r2)) if active.any() else None,
        mean_error_vox=float(error.mean()),
        median_error_vox=float(np.median(error)),
        error_mean_ci=bootstrap_interval(error),
        subjects=len(truth),
    )
    side = truth[:, 0] > split if split is not None else None
    if side is not None and 0 < side.sum() < len(side):
        # Mean R² mixes the hemisphere (x is mostly which side) with fine position, so score them apart.
        # y and z vary within both hemispheres; their mean R² is the fine-position score. Within-hemisphere
        # x is left out: probes fitted to raw x mostly learn the side, and their residual within a
        # hemisphere is noise against a spread of about 1.3 voxels.
        result.update(
            hemisphere_split=split,
            hemisphere_auc=float(roc_auc_score(side, prediction[:, 0])),
            hemisphere_accuracy=float(((prediction[:, 0] > split) == side).mean()),
            fine_position_r2=float(np.nanmean(r2[1:])) if active[1:].any() else None,
        )
    return result


def fit_probe(train_x, train_y, test_x, kind):
    if len(train_x) < 8:
        raise ValueError("Need at least eight calibration subjects")
    if kind == "ridge":
        model = Ridge(solver="lsqr")
        # Up to 1e6: the spatial descriptor has tens of thousands of standardized features for a few
        # hundred subjects, where a penalty of 100 or less barely regularizes.
        grid = {"regressor__model__alpha": np.logspace(-2, 6, 9).tolist()}
    else:
        model = KernelRidge(kernel="rbf")
        grid = {
            "regressor__model__alpha": [0.1, 1.0],
            "regressor__model__gamma": [0.1 / train_x.shape[1], 1.0 / train_x.shape[1]],
        }
    pipeline = Pipeline([("scale", StandardScaler()), ("model", model)])
    estimator = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
    search = GridSearchCV(
        estimator, grid, scoring="neg_mean_squared_error", cv=KFold(3, shuffle=True, random_state=12), n_jobs=1
    )
    with threadpool_limits(limits=1):
        search.fit(train_x, train_y)
        prediction = search.predict(test_x)
    return prediction, search.best_params_


@torch.no_grad()
def extract(model, bank, device, batch_size, spatial_grid, view=None):
    """Features of every subject per view and source. SSL models read both views; a GAP regressor its own view."""
    encoder, views = (model, VIEWS) if view is None else (model.encoder, (view,))
    out = {v: {source: [] for source in SOURCES} for v in views}
    model.eval()
    for start in range(0, len(bank), batch_size):
        batch = torch.from_numpy(np.array(bank[start : start + batch_size], copy=True)).to(device)
        if view is None:
            x, n_views = torch.cat([batch[:, 0], batch[:, 1]]), 2
        else:
            x, n_views = batch[:, VIEWS.index(view)], 1
        features = encoder._encode(x, n_views=n_views, view_idx=None)
        if spatial_grid > min(features.shape[2:]):
            raise ValueError("Spatial grid exceeds the native backbone grid")
        readouts = {
            "global": encoder._global_code(features)[:, :9],
            "pooled": features.mean(dim=(2, 3, 4)),
            "spatial": F.adaptive_avg_pool3d(features, spatial_grid).flatten(1),
        }
        for v, name in enumerate(views):
            part = slice(v * len(batch), (v + 1) * len(batch))
            for source, value in readouts.items():
                out[name][source].append(value[part].cpu().numpy())
    return {v: {s: np.concatenate(parts) for s, parts in sources.items()} for v, sources in out.items()}


@torch.no_grad()
def predict_localizer(model, bank, view, device, batch_size):
    model.eval()
    parts = []
    for start in range(0, len(bank), batch_size):
        x = torch.from_numpy(np.array(bank[start : start + batch_size, VIEWS.index(view)], copy=True)).to(device)
        parts.append(model(x).cpu().numpy())
    return np.concatenate(parts)


def checkpoint_files(folder, name):
    files = sorted(folder.glob("step_*.pt")) if name == "all" else [folder / name]
    if not files:
        raise FileNotFoundError(f"No checkpoints in {folder}")
    return files


def evaluate(args):
    root = Path(args.bank)
    meta, device = load_bank(root), select_encoder_device(args.device)
    missing = [arm for arm in args.arms if arm not in meta["arms"]]
    if missing:
        raise ValueError(f"Arms not in this bank: {missing}")
    train_y, test_y = targets(root, "probe"), targets(root, "test")
    evaluator = file_hash(Path(__file__))
    split = hemisphere_split(np.load(root / "centres.npy")[:, 0])
    for arm in args.arms:
        common = COMMON_ARM[arm]
        for seed in args.seeds:
            for run_view in args.views if args.kind != "ssl" else ["paired"]:
                folder = run_path(root, args.kind, arm, seed, run_view, "std" if args.standardize_code else "")
                for path in checkpoint_files(folder, args.checkpoint):
                    evaluate_checkpoint(args, root, meta, device, path, arm, common, train_y, test_y, evaluator, split)


def evaluate_checkpoint(args, root, meta, device, path, arm, common, train_y, test_y, evaluator, split):
    ckpt = torch_load(path, device)
    settings = ckpt["settings"]
    if settings["bank_id"] != meta["bank_id"]:
        raise ValueError("Checkpoint and bank differ")
    out = path.parent / f"eval_{ckpt['step']:06d}"
    done = out / "metrics.json"
    if done.exists() and not args.overwrite:
        previous = json.loads(done.read_text())
        if (previous.get("evaluator_sha256"), previous.get("spatial_grid"), previous.get("views")) == (
            evaluator,
            args.spatial_grid,
            list(args.views),
        ):
            print(f"Evaluation exists: {done}", flush=True)
            return
    model = build_model(settings)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    out.mkdir(exist_ok=True)
    rows, feature_cache = [], {}
    for regime, bank_arm in (("common", common), ("own", arm)):
        if regime == "own" and bank_arm == common:
            # The same test images as the common regime: reuse those rows rather than refit.
            rows += [dict(row, regime="own") for row in rows if row["regime"] == "common"]
            continue
        if args.kind != "supervised" and bank_arm not in feature_cache:
            own = None if args.kind == "ssl" else settings["view"]
            feature_cache[bank_arm] = tuple(
                extract(model, images(root, split_name, bank_arm), device, args.batch_size, args.spatial_grid, own)
                for split_name in ("probe", "test")
            )
        if args.kind == "ssl":
            train_f, test_f = feature_cache[bank_arm]
            for view in args.views:
                for source in SOURCES:
                    for kind in ("ridge", "rbf"):
                        prediction, params = fit_probe(
                            train_f[view][source], train_y[:, :3], test_f[view][source], kind
                        )
                        filename = f"{regime}_{view}_{source}_{kind}.npz"
                        np.savez(out / filename, truth=test_y[:, :3], prediction=prediction)
                        rows.append(
                            dict(
                                regime=regime,
                                view=view,
                                source=source,
                                probe=kind,
                                probe_params=params,
                                predictions=filename,
                                **metrics(test_y[:, :3], prediction, split),
                            )
                        )
                if regime == "own" and BRAIN_RULE[arm] != "fixed":
                    # Manipulation check: is the competitor in the code? Bp's FLAIR view has its own draw.
                    column = 4 if BRAIN_RULE[arm] == "per_view" and view == "flair" else 3
                    prediction, _ = fit_probe(
                        train_f[view]["global"], train_y[:, column], test_f[view]["global"], "ridge"
                    )
                    variance = np.sum((test_y[:, column] - test_y[:, column].mean()) ** 2)
                    rows.append(
                        dict(
                            regime="competitor",
                            view=view,
                            source="global",
                            probe="ridge",
                            brain_size_r2=float(1 - np.sum((prediction - test_y[:, column]) ** 2) / variance),
                        )
                    )
        else:
            # Both supervised controls predict coordinates directly from one view's image.
            source = "global" if args.kind == "supervised_gap" else "image"
            run_view = settings["view"]
            prediction = predict_localizer(model, images(root, "test", bank_arm), run_view, device, args.batch_size)
            filename = f"{regime}_{run_view}_{source}_supervised.npz"
            np.savez(out / filename, truth=test_y[:, :3], prediction=prediction)
            rows.append(
                dict(
                    regime=regime,
                    view=run_view,
                    source=source,
                    probe="supervised",
                    predictions=filename,
                    **metrics(test_y[:, :3], prediction, split),
                )
            )
            if args.kind == "supervised_gap":
                # The same frozen-feature probes as SSL, on this view's code and pooled vector.
                train_f, test_f = feature_cache[bank_arm]
                for source in ("global", "pooled"):
                    for kind in ("ridge", "rbf"):
                        prediction, params = fit_probe(
                            train_f[run_view][source], train_y[:, :3], test_f[run_view][source], kind
                        )
                        filename = f"{regime}_{run_view}_{source}_{kind}.npz"
                        np.savez(out / filename, truth=test_y[:, :3], prediction=prediction)
                        rows.append(
                            dict(
                                regime=regime,
                                view=run_view,
                                source=source,
                                probe=kind,
                                probe_params=params,
                                predictions=filename,
                                **metrics(test_y[:, :3], prediction, split),
                            )
                        )
    constant = np.broadcast_to(train_y[:, :3].mean(0), test_y[:, :3].shape)
    write_json(
        done,
        dict(
            bank_id=meta["bank_id"],
            kind=args.kind,
            arm=arm,
            seed=settings["seed"],
            step=ckpt["step"],
            settings=settings,
            evaluator_sha256=evaluator,
            views=list(args.views),
            spatial_grid=args.spatial_grid,
            spatial_description="Average-pooled backbone map, flattened without PCA",
            constant_baseline=metrics(test_y[:, :3], constant, split),
            rows=rows,
        ),
    )
    print(f"Evaluation saved: {done}", flush=True)


def summarize(args):
    root = Path(args.bank)
    meta = load_bank(root)
    records = []
    for folder in sorted((root / "runs").glob("*")):
        reports = sorted(folder.glob("eval_*/metrics.json"))
        if not reports:
            continue
        # Highest training step, never chosen by probe accuracy.
        path = reports[-1]
        result = json.loads(path.read_text())
        if result["bank_id"] != meta["bank_id"]:
            raise ValueError("Evaluation bank mismatch")
        floor_path = folder / "eval_000000/metrics.json"
        floor_rows = json.loads(floor_path.read_text())["rows"] if floor_path.exists() else []
        signature = dict(result["settings"])
        for key in ("arm", "seed", "view"):
            signature.pop(key, None)
        for row in result["rows"]:
            if "mean_r2" in row:
                floor = next(
                    (f for f in floor_rows if all(f[k] == row[k] for k in ("regime", "view", "source", "probe"))), None
                )
                initial = floor.get("mean_r2") if floor else None
                records.append(
                    dict(
                        kind=result["kind"] + ("-std" if result["settings"].get("standardize_code") else ""),
                        arm=result["arm"],
                        seed=result["seed"],
                        step=result["step"],
                        **row,
                        directory=str(path.parent),
                        training_signature=signature,
                        spatial_grid=result["spatial_grid"],
                        floor_mean_r2=initial,
                        gain_over_floor=(
                            row["mean_r2"] - initial if initial is not None and row["mean_r2"] is not None else None
                        ),
                    )
                )
    if not records:
        raise ValueError("No completed evaluations; run evaluate first")
    output = root / "summary"
    output.mkdir(exist_ok=True)
    fields = [
        "kind",
        "arm",
        "seed",
        "step",
        "regime",
        "view",
        "source",
        "probe",
        "mean_r2",
        "floor_mean_r2",
        "gain_over_floor",
        "mean_error_vox",
        "median_error_vox",
    ]
    with (output / "recovery.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    lookup = {(r["kind"], r["arm"], r["seed"], r["regime"], r["view"], r["source"], r["probe"]): r for r in records}
    comparisons = []
    for key, a in lookup.items():
        kind, arm, seed, regime, view, source, probe = key
        for other in COMPARISONS.get(arm, ()):
            b = lookup.get((kind, other, seed, regime, view, source, probe))
            if b is None or a["step"] != b["step"]:
                continue
            if a["training_signature"] != b["training_signature"] or a["spatial_grid"] != b["spatial_grid"]:
                raise ValueError("Paired runs have different training recipes or evaluation grids")
            pa = np.load(Path(a["directory"]) / a["predictions"])
            pb = np.load(Path(b["directory"]) / b["predictions"])
            if not np.array_equal(pa["truth"], pb["truth"]):
                raise ValueError("Comparison target identities differ")
            error_a = np.linalg.norm(pa["prediction"] - pa["truth"], axis=1)
            error_b = np.linalg.norm(pb["prediction"] - pb["truth"], axis=1)
            ci = bootstrap_interval(error_b - error_a)
            comparisons.append(
                dict(
                    kind=kind,
                    pair=arm + other,
                    seed=seed,
                    step=a["step"],
                    regime=regime,
                    view=view,
                    source=source,
                    probe=probe,
                    r2_penalty=(
                        a["mean_r2"] - b["mean_r2"] if a["mean_r2"] is not None and b["mean_r2"] is not None else None
                    ),
                    error_penalty=float((error_b - error_a).mean()),
                    error_penalty_ci=ci,
                    supervised_equivalent_within_one_voxel=((ci[0] >= -1 and ci[1] <= 1) if kind != "ssl" else None),
                )
            )
    shared_keys = ("kind", "seed", "step", "regime", "view", "source", "probe")
    # R_A - R_B = (R_A - R_Bp) + (R_Bp - R_B): input interference plus competition through the loss.
    decomposition = []
    for a in comparisons:
        if a["pair"] != "AB" or a["r2_penalty"] is None:
            continue
        match = next((b for b in comparisons if b["pair"] == "ABp" and all(a[k] == b[k] for k in shared_keys)), None)
        if match and match["r2_penalty"] is not None:
            decomposition.append(
                {
                    **{k: a[k] for k in shared_keys},
                    "total_penalty": a["r2_penalty"],
                    "input_interference": match["r2_penalty"],
                    "through_loss": a["r2_penalty"] - match["r2_penalty"],
                }
            )
    interactions = []
    for a in comparisons:
        if a["pair"] != "AB" or a["r2_penalty"] is None:
            continue
        match = next(
            (
                b
                for b in comparisons
                if b["pair"] == "CD"
                and all(a[k] == b[k] for k in ("kind", "seed", "step", "regime", "view", "source", "probe"))
            ),
            None,
        )
        if match and match["r2_penalty"] is not None:
            interactions.append(
                {
                    **{k: a[k] for k in ("kind", "seed", "step", "regime", "view", "source", "probe")},
                    "visibility_interaction": a["r2_penalty"] - match["r2_penalty"],
                }
            )
    audit = json.loads((root / "audit/report.json").read_text())
    write_json(
        output / "report.json",
        dict(
            bank_id=meta["bank_id"],
            records=records,
            comparisons=comparisons,
            decomposition=decomposition,
            interactions=interactions,
            visibility=audit["comparisons"],
            scope="Paired seed estimates; subject intervals condition on trained models. Inspect controls before claiming suppression.",
        ),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [arm for arm in ARMS if any(r["arm"] == arm for r in records)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, view in zip(axes, VIEWS):
        for j, arm in enumerate(arms):
            selected = [
                r
                for r in records
                if r["kind"] == "ssl"
                and r["regime"] == "common"
                and r["view"] == view
                and r["source"] == "global"
                and r["probe"] == "ridge"
                and r["arm"] == arm
                and r["mean_r2"] is not None
            ]
            if selected:
                values = [r["mean_r2"] for r in selected]
                ax.scatter(np.full(len(values), j), values)
                ax.plot([j - 0.2, j + 0.2], [np.mean(values)] * 2, color="black")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xticks(range(len(arms)), arms)
        ax.set_title(view)
        ax.set_xlabel("Arm (dots: training seeds)")
    axes[0].set_ylabel("Lesion coordinate R²: global ridge probe")
    fig.tight_layout()
    fig.savefig(output / "recovery.png", dpi=160)
    plt.close(fig)
    print(f"Summary: {output / 'report.json'}", flush=True)


# (kind, source, probe, panel title) for the learning-curve figure; every other readout is in the CSV.
CURVE_PANELS = (
    ("ssl", "global", "ridge", "Self-supervised: 9-unit code"),
    ("ssl", "pooled", "ridge", "Self-supervised: pooled features"),
    ("ssl", "spatial", "ridge", "Self-supervised: spatial map"),
    ("supervised_gap", "global", "supervised", "Labels, same encoder: prediction"),
    ("supervised_gap", "pooled", "ridge", "Labels, same encoder: pooled features"),
    ("supervised", "image", "supervised", "Labels: heatmap localizer"),
)
# (source, probe, label) columns of the printed table, per run kind; ridge probes only, the rest is in the CSV.
TABLE_COLUMNS = {
    "ssl": (("global", "ridge", "code"), ("pooled", "ridge", "pooled"), ("spatial", "ridge", "spatial")),
    "supervised_gap": (("global", "supervised", "prediction"), ("pooled", "ridge", "pooled")),
    "supervised": (("image", "supervised", "prediction"),),
}
VIEW_COLORS = {"t1": "#2a78d6", "flair": "#eb6834"}
# Categorical slots in fixed order for arm identity, and the readouts the arm comparison shows.
ARM_COLORS = {"A": "#2a78d6", "B": "#eb6834", "Bp": "#1baf7a", "C": "#eda100", "D": "#e87ba4"}
ARM_LABELS = {
    "A": "A: brain size fixed",
    "B": "B: brain size shared by both views",
    "Bp": "B′: brain size drawn per view",
    "C": "C: fixed, lesion contrast ×1.5",
    "D": "D: shared, lesion contrast ×1.5",
}
COMPARISON_PANELS = (
    ("global", "rbf", "9-unit code, nonlinear probe"),
    ("global", "ridge", "9-unit code, linear probe"),
    ("pooled", "ridge", "pooled features, linear probe"),
)
INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def curves(args):
    """Learning curves: held-out loss at every checkpoint, and test-set lesion R² at every evaluated step.

    Reads the common (fixed brain size) test regime. Writes summary/curves.csv with every number,
    one summary/curves_<arm>.png per arm, and prints a table per run.
    """
    root = Path(args.bank)
    meta = load_bank(root)
    records = []
    for folder in sorted((root / "runs").glob("*")):
        if not (folder / "settings.json").exists():
            continue
        settings = json.loads((folder / "settings.json").read_text())
        if settings["arm"] not in args.arms or settings["kind"] not in args.kinds:
            continue
        progress = folder / "progress.json"
        monitor = json.loads(progress.read_text()).get("monitor", []) if progress.exists() else []
        monitor = {row["step"]: row for row in monitor}
        evaluations = {}
        for path in folder.glob("eval_*/metrics.json"):
            result = json.loads(path.read_text())
            if result["bank_id"] != meta["bank_id"]:
                raise ValueError("Evaluation bank mismatch")
            evaluations[result["step"]] = result
        for step in sorted(set(monitor) | set(evaluations)):
            check = monitor.get(step, {})
            base = dict(run=folder.name, kind=settings["kind"], arm=settings["arm"], seed=settings["seed"], step=step)
            base.update(variant="batch-standardized" if settings.get("standardize_code") else "plain")
            base.update({k: check.get(k) for k in ("heldout_loss", "heldout_top1", "chance_loss", "chance_top1")})
            rows = evaluations.get(step, {}).get("rows", [])
            readouts = [r for r in rows if r["regime"] == "common" and "mean_r2" in r]
            # Manipulation check, per view: brain size read from the code, in arms where it varies.
            competitor = {r["view"]: r["brain_size_r2"] for r in rows if r["regime"] == "competitor"}
            for r in readouts or [{}]:
                records.append(
                    dict(
                        base,
                        view=r.get("view"),
                        source=r.get("source"),
                        probe=r.get("probe"),
                        mean_r2=r.get("mean_r2"),
                        hemisphere_auc=r.get("hemisphere_auc"),
                        fine_r2=r.get("fine_position_r2"),
                        median_error_vox=r.get("median_error_vox"),
                        brain_size_r2=competitor.get(r.get("view")),
                    )
                )
    if not records:
        raise ValueError("No matching runs; train first")
    output = root / "summary"
    output.mkdir(exist_ok=True)
    fields = list(records[0])
    with (output / "curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print_curves(records)
    for arm in [a for a in ARMS if any(r["arm"] == a for r in records)]:
        plot_curves([r for r in records if r["arm"] == arm], arm, output / f"curves_{arm}.png")
    for variant in sorted({r["variant"] for r in records if r["kind"] == "ssl"}):
        if len({r["arm"] for r in records if r["kind"] == "ssl" and r["variant"] == variant}) > 1:
            plot_arm_comparison(records, variant, output / f"curves_arms_{variant}.png")
    print(f"Curves: {output / 'curves.csv'}", flush=True)


def print_curves(records):
    """One line per step and view: held-out loss, then R² / hemisphere AUC / y-z R² for each readout."""

    def number(value, width):
        return f"{value:{width}.2f}" if value is not None else " " * (width - 1) + "-"

    print(
        "\nLesion on the common test set. R²: mean over x, y, z. AUC: which hemisphere, from predicted x (0.5 chance)."
        " y/z: mean R² of y and z, the position within a hemisphere. brain: brain size read from the code (R²).",
        flush=True,
    )
    for run in sorted({r["run"] for r in records}):
        rows = [r for r in records if r["run"] == run]
        columns = TABLE_COLUMNS[rows[0]["kind"]]
        chance = next((r for r in rows if r["chance_loss"] is not None), None)
        title = f"\n{run}: held-out loss"
        if chance:
            title += f" (InfoNCE chance {chance['chance_loss']:.3f}, top-1 chance {chance['chance_top1']:.3f})"
        print(title, flush=True)
        head = "   step  heldout" + ("   top1" if chance else "") + "  view "
        competitor = any(r.get("brain_size_r2") is not None for r in rows)
        print(
            head
            + "".join(f" | {label + ': R²   AUC   y/z':>22}" for _, _, label in columns)
            + (" |  brain" if competitor else ""),
            flush=True,
        )
        for step in sorted({r["step"] for r in rows}):
            at = [r for r in rows if r["step"] == step]
            lead = f"{step:7d}  {number(at[0]['heldout_loss'], 7)}" + (
                f" {number(at[0]['heldout_top1'], 6)}" if chance else ""
            )
            views = [v for v in VIEWS if any(r["view"] == v for r in at)] or [None]
            for i, view in enumerate(views):
                cells = []
                for source, probe, _ in columns:
                    row = next((r for r in at if (r["view"], r["source"], r["probe"]) == (view, source, probe)), {})
                    values = (row.get("mean_r2"), row.get("hemisphere_auc"), row.get("fine_r2"))
                    cells.append(" | " + " ".join(number(v, 6) for v in values).rjust(22))
                prefix = lead if i == 0 else " " * len(lead)
                brain = next((r.get("brain_size_r2") for r in at if r["view"] == view), None)
                tail = f" | {number(brain, 6)}" if competitor else ""
                print(f"{prefix}  {view or '-':5s}" + "".join(cells) + tail, flush=True)


def plot_curves(records, arm, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    hairline, line, marker = 0.45, 0.9, 4  # points at 160 dpi: 1 px, 2 px and ~9 px
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), facecolor=SURFACE)
    axes = axes.ravel()
    r2_values = [r["mean_r2"] for r in records if r["mean_r2"] is not None]
    low = min([-0.1] + [v - 0.05 for v in r2_values])
    for ax in axes[:7]:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=hairline)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(AXIS)
            spine.set_linewidth(hairline)
        ax.tick_params(colors=SECONDARY, labelsize=8, width=hairline)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_xlabel("training step", color=SECONDARY, fontsize=9)
    ssl = [r for r in records if r["kind"] == "ssl" and r["heldout_loss"] is not None]
    style = {"plain": "-", "batch-standardized": "--"}
    variant = {r["run"]: r["variant"] for r in records}
    ax = axes[0]
    ax.set_title("Self-supervised: held-out InfoNCE", color=INK, fontsize=10, loc="left")
    for run in sorted({r["run"] for r in ssl}):
        steps = sorted({r["step"]: r for r in ssl if r["run"] == run}.values(), key=lambda r: r["step"])
        ax.plot(
            [r["step"] for r in steps],
            [r["heldout_loss"] for r in steps],
            color=SECONDARY,
            linestyle=style[variant[run]],
            linewidth=line,
            marker="o",
            markersize=marker,
            markeredgecolor=SURFACE,
            markeredgewidth=line,
        )
    if ssl:
        chance = ssl[0]["chance_loss"]
        losses = [r["heldout_loss"] for r in ssl]
        # At least half a nat below chance, so noise around chance can't pass for learning.
        ax.set_ylim(min(min(losses), chance - 0.5) - 0.05, max(max(losses), chance) + 0.1)
        ax.axhline(chance, color=MUTED, linewidth=hairline)
        ax.annotate(
            "chance (2 ln B)",
            (1, chance),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            color=SECONDARY,
            fontsize=8,
        )
    else:
        ax.text(0.5, 0.5, "no runs", ha="center", va="center", color=MUTED, transform=ax.transAxes)
    for ax, (kind, source, probe, title) in zip(axes[1:7], CURVE_PANELS):
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_ylim(low, 1.0)
        ax.axhline(0, color=AXIS, linewidth=hairline)
        selected = [
            r
            for r in records
            if r["kind"] == kind and r["source"] == source and r["probe"] == probe and r["mean_r2"] is not None
        ]
        for run, view in sorted({(r["run"], r["view"]) for r in selected}):
            points = sorted((r["step"], r["mean_r2"]) for r in selected if r["run"] == run and r["view"] == view)
            ax.plot(
                *zip(*points),
                color=VIEW_COLORS[view],
                linestyle=style[variant[run]],
                linewidth=line,
                marker="o",
                markersize=marker,
                markeredgecolor=SURFACE,
                markeredgewidth=line,
            )
        if not selected:
            ax.text(0.5, 0.5, "no evaluations", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        ax.set_ylabel("lesion centre R²", color=SECONDARY, fontsize=9)
    legend = axes[7]
    legend.axis("off")
    for view, label in (("t1", "T1"), ("flair", "FLAIR")):
        legend.plot([], [], color=VIEW_COLORS[view], linewidth=line, marker="o", markersize=marker, label=label)
    if len(set(variant.values())) > 1:
        for name in style:
            legend.plot([], [], color=SECONDARY, linestyle=style[name], linewidth=line, label=f"SSL loss: {name}")
    legend.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK, title="view", title_fontsize=9)
    legend.text(
        0.0,
        0.42,
        "R²: common test set (brain size fixed),\nmean over x, y, z; step 0 is the untrained floor.\n"
        "Pooled: the backbone map averaged over all\npositions, read by a ridge probe.\n"
        "Seeds are overlaid. Every value, including\nhemisphere AUC and y/z R², is in curves.csv.",
        color=SECONDARY,
        fontsize=8,
        va="top",
        transform=legend.transAxes,
    )
    fig.suptitle(f"Arm {arm}: lesion-location learning curves", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_arm_comparison(records, variant, path):
    """The arms' SSL learning curves on shared axes: held-out InfoNCE and top-1, then lesion R² per view."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    ssl = [r for r in records if r["kind"] == "ssl" and r["variant"] == variant]
    arms = [a for a in ARMS if any(r["arm"] == a for r in ssl)]
    hairline, line, marker = 0.45, 0.9, 4  # points at 160 dpi: 1 px, 2 px and ~9 px
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), facecolor=SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=hairline)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(AXIS)
            spine.set_linewidth(hairline)
        ax.tick_params(colors=SECONDARY, labelsize=8, width=hairline)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_xlabel("training step", color=SECONDARY, fontsize=9)

    def draw(ax, points, arm):
        ax.plot(
            *zip(*sorted(points)),
            color=ARM_COLORS[arm],
            linewidth=line,
            marker="o",
            markersize=marker,
            markeredgecolor=SURFACE,
            markeredgewidth=line,
        )

    monitored = {(r["run"], r["step"]): r for r in ssl if r["heldout_loss"] is not None}
    arm_of = {r["run"]: r["arm"] for r in ssl}
    for row, (key, chance_key, title) in enumerate(
        (("heldout_loss", "chance_loss", "Held-out InfoNCE"), ("heldout_top1", "chance_top1", "Held-out top-1"))
    ):
        ax = axes[row, 0]
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        for run in sorted({run for run, _ in monitored}):
            draw(ax, [(step, r[key]) for (name, step), r in monitored.items() if name == run], arm_of[run])
        if monitored:
            chance = next(iter(monitored.values()))[chance_key]
            ax.axhline(chance, color=MUTED, linewidth=hairline)
            ax.annotate(
                "chance",
                (1, chance),
                xycoords=("axes fraction", "data"),
                ha="right",
                va="bottom",
                color=SECONDARY,
                fontsize=8,
            )
            if key == "heldout_loss":
                values = [r[key] for r in monitored.values()]
                top = min(max(values), chance + 5)
                ax.set_ylim(min(min(values), chance - 0.5) - 0.2, top + 0.2)
                if max(values) > top:
                    ax.text(
                        0.02,
                        0.97,
                        "points above the axis are in curves.csv",
                        color=MUTED,
                        fontsize=7,
                        va="top",
                        transform=ax.transAxes,
                    )
    low = min([-0.1] + [r["mean_r2"] - 0.05 for r in ssl if r["mean_r2"] is not None])
    for row, view in enumerate(VIEWS):
        for column, (source, probe, title) in enumerate(COMPARISON_PANELS, start=1):
            ax = axes[row, column]
            ax.set_title(f"{'T1' if view == 't1' else 'FLAIR'}: {title}", color=INK, fontsize=10, loc="left")
            ax.set_ylim(low, 1.0)
            ax.axhline(0, color=AXIS, linewidth=hairline)
            ax.set_ylabel("lesion centre R²", color=SECONDARY, fontsize=9)
            chosen = [
                r
                for r in ssl
                if r["view"] == view and r["source"] == source and r["probe"] == probe and r["mean_r2"] is not None
            ]
            for run in sorted({r["run"] for r in chosen}):
                draw(ax, [(r["step"], r["mean_r2"]) for r in chosen if r["run"] == run], arm_of[run])
            if not chosen:
                ax.text(0.5, 0.5, "no evaluations", ha="center", va="center", color=MUTED, transform=ax.transAxes)
    for arm in arms:
        axes[0, 3].plot(
            [], [], color=ARM_COLORS[arm], linewidth=line, marker="o", markersize=marker, label=ARM_LABELS[arm]
        )
    axes[0, 3].legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle(
        f"Lesion location by arm, {variant} InfoNCE (common test set with brain size fixed; seeds overlaid)",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
