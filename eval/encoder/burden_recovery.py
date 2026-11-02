"""Where a run keeps the lesion burden: the final vector, the regional codes or the backbone map.

A burden lesion can sit anywhere, so the three representations need different probes, and one number per view
cannot say which of them gained or lost it. This compares runs at matched checkpoints on one set of subjects, per
view and readout:

- **code**: the exported global content vector. **regional**: the spatial head's content units on the run's
  ``--train-patch-grid`` cells. **map**: the backbone features on those cells.
- **ridge**: ``regional_probe``'s linear probe, on the regional and map cells averaged to ``--probe-grid``.
- **kernel**: RBF kernel ridge on the same features (first PCA to ``KERNEL_PCA`` components when wider), on a
  centred target because the kernel has no intercept.
- **quantile**: regional and map only. Each channel's quantiles over the brain's interior cells, after removing each
  position's training mean and SD, then ridge. This is the position-free read an amount calls for; a probe over
  concatenated cells sees a lesion at any one cell in about 2 of 400 subjects, so it can read ~0 from a map that does
  hold the burden. Interior means at least ``INTERIOR`` mean brain occupancy, because a wider set's tails belong to
  brain-edge anatomy: on the raw FLAIR image, cells with any brain read Spearman 0.24 and interior cells 0.63.
- **image**: the quantile probe on each view's own image cells, with no network. It is the positive control: when it
  reads nothing, this probe says nothing about any representation.

Each probe predicts z_content's ``lesion_burden`` on held-out folds. Scores are computed on those predictions over
all subjects, and over bins of the rendered lesion's voxels, where tiny lesions stay in and the probe is not
refitted. The bins restrict the range, so read **spearman** there; **r2** is relative to the bin's own mean.

The first run is the control. Every run is scored at ``--checkpoint`` against its own ``model_init.pt``, and each
later run minus the control gets a paired bootstrap interval over subjects. A pair is refused unless it is matched:
the same generator and subjects, the same settings outside the prediction options, the same step, batch order and
input hashes, and the same encoder at initialization. ``--allow-unmatched`` prints the differences instead.

Example:
    python -m eval.encoder.burden_recovery --run-dirs results/encoder/runs/conv_mlp_burden_nativegrid_s42 \\
        results/encoder/runs/conv_mlp_burden_vector_prediction_s42 --split test --num-samples 1000 --out burden.json
    python -m eval.encoder.burden_recovery --self-test
"""

import argparse
import json
import os

import numpy as np

from eval.encoder.regional_lesion_detection import lesion_fractions
from eval.encoder.regional_probe import (
    ALPHAS,
    READOUTS,
    SEEDS,
    VIEWS,
    adaptive_mean,
    check_run,
    data_key,
    held_out,
    readouts,
)

KERNEL_PCA = 256
INTERIOR = 0.9
QUANTILES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999)
BIN_EDGES = (10, 20, 30)
# Settings a matched pair may differ in: names and the intervention itself.
FREE_SETTINGS = ("model_id", "out_dir")
FREE_PREFIX = "vector_prediction_"
MATCHED_PROGRESS = ("step", "status", "batch_order_sha256", "training_input_sha256")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dirs", nargs="+", help="Run directories; the first is the control")
    p.add_argument("--checkpoint", default="model.pt", help="Checkpoint inside each run; model_init.pt is the floor")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--num-samples", type=int, default=None, help="Subjects (default: the control's val size)")
    p.add_argument("--probe-grid", type=int, nargs=3, default=[4, 4, 4], help="Cells of the ridge and kernel probes")
    p.add_argument("--bootstrap", type=int, default=1000, help="Paired resamples of a run minus the control")
    p.add_argument("--allow-unmatched", action="store_true", help="Print a pair's differences instead of refusing it")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--out", default=None, help="Also write the scores to this JSON file")
    p.add_argument("--self-test", action="store_true", help="Check the probes on toy cells")
    args = p.parse_args(argv)
    if not args.self_test and not args.run_dirs:
        p.error("--run-dirs is required unless --self-test")
    if min(args.probe_grid) < 1:
        p.error("--probe-grid needs positive cell counts")
    return args


def match_problems(control, other):
    """Why ``other`` is not matched with ``control``: each is a run's {settings, progress, init}."""
    problems = []
    keys = {*control["settings"], *other["settings"]}
    for key in sorted(k for k in keys if k not in FREE_SETTINGS and not k.startswith(FREE_PREFIX)):
        first, second = control["settings"].get(key), other["settings"].get(key)
        if first != second:
            problems.append(f"setting {key}: {first!r} vs {second!r}")
    for key in MATCHED_PROGRESS:
        first, second = control["progress"].get(key), other["progress"].get(key)
        if first is None or first != second:
            problems.append(f"progress {key}: {first!r} vs {second!r}")
    if control["init"] is not None and other["init"] is not None:
        shared = {k for k in control["init"] if not k.startswith("vector_prediction.")}
        changed = [
            k for k in sorted(shared) if k not in other["init"] or not _equal(control["init"][k], other["init"][k])
        ]
        if changed:
            problems.append(f"model_init.pt: {len(changed)} encoder tensors differ, first {changed[0]}")
    return problems


def _equal(first, second):
    return np.array_equal(np.asarray(first), np.asarray(second))


def lesion_bins(voxels):
    """{name: subject mask} over the rendered lesion's voxels."""
    bins = {"all": np.ones(len(voxels), dtype=bool)}
    bins.update({f"<{edge}": voxels < edge for edge in BIN_EDGES})
    bins[f">={BIN_EDGES[-1]}"] = voxels >= BIN_EDGES[-1]
    return bins


def bin_scores(prediction, target, bins):
    """Held-out R² and Spearman of one prediction per bin; NaN below 8 subjects."""
    from scipy.stats import spearmanr

    out = {}
    for name, rows in bins.items():
        n, y, p = int(rows.sum()), target[rows], prediction[rows]
        if n < 8 or not np.ptp(y) > 0:
            out[name] = {"n": n, "r2": float("nan"), "spearman": float("nan")}
            continue
        rho = spearmanr(p, y).statistic if np.ptp(p) > 0 else 0.0
        out[name] = {
            "n": n,
            "r2": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "spearman": float(rho),
        }
    return out


def kernel_predictions(X, y, seed):
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    from eval.metrics.identifiability_metrics import _make_regressor

    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        scaler = StandardScaler().fit(X[tr])
        train, test = scaler.transform(X[tr]), scaler.transform(X[te])
        size = min(KERNEL_PCA, len(tr) - 1)
        if train.shape[1] > size:
            pca = PCA(size, random_state=seed).fit(train)
            train, test = pca.transform(train), pca.transform(test)
        centre = y[tr].mean()
        pred[te] = _make_regressor("kernel", seed).fit(train, y[tr] - centre).predict(test) + centre
    return pred


def quantile_predictions(cells, y, seed):
    """Held-out ridge on each channel's quantiles over cells: (subjects, channels, positions)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(cells):
        z = (cells - cells[tr].mean(0, keepdims=True)) / (cells[tr].std(0, keepdims=True) + 1e-6)
        features = np.quantile(z, QUANTILES, axis=2).transpose(1, 2, 0).reshape(len(z), -1)
        scaler = StandardScaler().fit(features[tr])
        model = RidgeCV(alphas=ALPHAS).fit(scaler.transform(features[tr]), y[tr])
        pred[te] = model.predict(scaler.transform(features[te]))
    return pred


def predictions(flat, cells, y):
    """{probe: (seeds, subjects) held-out predictions} for one representation; ``cells`` is None for the vector."""
    out = {
        "ridge": np.stack([held_out(flat, y, seed) for seed in SEEDS]),
        "kernel": np.stack([kernel_predictions(flat, y, seed) for seed in SEEDS]),
    }
    if cells is not None:
        out["quantile"] = np.stack([quantile_predictions(cells, y, seed) for seed in SEEDS])
    return out


def seed_mean_scores(seeds, y, bins):
    """``bin_scores`` averaged over the seeds' predictions."""
    per_seed = [bin_scores(p, y, bins) for p in seeds]
    return {
        name: {key: float(np.mean([s[name][key] for s in per_seed])) for key in ("r2", "spearman")}
        | {"n": per_seed[0][name]["n"]}
        for name in bins
    }


def score_representations(rep, y, bins, probe_grid, grid, foreground, interior):
    """Scores and mean predictions per view, readout and probe for one checkpoint."""
    scores, means = {}, {}
    for view in VIEWS:
        for readout in READOUTS:
            cells = rep[view][readout]
            flat = cells if readout == "code" else _flat(cells, grid, foreground, probe_grid)
            held = predictions(flat, None if readout == "code" else cells[:, :, interior], y)
            for probe, seeds in held.items():
                scores[f"{view}/{readout}/{probe}"] = seed_mean_scores(seeds, y, bins)
                means[f"{view}/{readout}/{probe}"] = seeds.mean(0)
    return scores, means


def _flat(cells, grid, kept, probe_grid):
    """(subjects, channels, grid cells) averaged to ``probe_grid`` cells that hold foreground."""
    n, channels = cells.shape[:2]
    occupied = adaptive_mean(kept.reshape(1, 1, -1).astype(np.float64), grid, probe_grid)[0] > 0
    averaged = adaptive_mean(cells, grid, probe_grid).reshape(n, channels, -1)
    return averaged[:, :, occupied].reshape(n, -1)


def paired_delta(first, second, y, rows, draws, seed=0):
    """Mean and 95% bootstrap interval of Spearman(second) - Spearman(first) over the same subjects."""
    from scipy.stats import rankdata

    picks = np.flatnonzero(rows)
    if len(picks) < 8 or draws < 1:
        return None
    rng = np.random.default_rng(seed)

    def spearman(pred, chosen):
        a, b = rankdata(pred[chosen]), rankdata(y[chosen])
        return np.corrcoef(a, b)[0, 1] if np.ptp(a) > 0 and np.ptp(b) > 0 else 0.0

    samples = [spearman(second, c) - spearman(first, c) for c in (rng.choice(picks, len(picks)) for _ in range(draws))]
    return {
        "delta": float(spearman(second, picks) - spearman(first, picks)),
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
    }


def encode(model, grid, dataset, device, batch_size):
    """Per view: the vector (subjects, units), and the regional and map cells (subjects, channels, grid cells)."""
    import torch
    from torch.utils.data import DataLoader

    parts = {view: {r: [] for r in READOUTS} for view in VIEWS}
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size):
            x = torch.cat(batch["image"], dim=0).as_subclass(torch.Tensor).to(device)
            b = x.shape[0] // 2
            for name, value in zip(READOUTS, readouts(model, x, grid, grid)):
                value = value.float().cpu().numpy()
                if name != "code":
                    value = value.reshape(len(value), -1, int(np.prod(grid)))
                parts["t1"][name].append(value[:b])
                parts["flair"][name].append(value[b:])
    return {view: {r: np.concatenate(v) for r, v in rep.items()} for view, rep in parts.items()}


def subject_table(dataset, grid, batch_size, threshold=0.05):
    """One pass over ``dataset``: z_content, the foreground and interior cells, each view's z-scored image cells."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from models.vector_prediction import prepare_vector_targets

    factors, foreground, occupancy, count = [], None, 0, 0
    images = {view: [] for view in VIEWS}
    for batch in DataLoader(dataset, batch_size=batch_size):
        factors.append(batch["gt_latents"]["z_content"].double().numpy())
        x = torch.cat(batch["image"], dim=0).as_subclass(torch.Tensor).float()
        masks = torch.cat(batch["mask"], dim=0).as_subclass(torch.Tensor).float()
        cells = F.adaptive_avg_pool3d(masks, tuple(grid))
        present = (cells >= threshold).any(0).flatten()
        foreground = present if foreground is None else foreground | present
        occupancy, count = occupancy + cells.sum(0).flatten(), count + len(cells)
        z = prepare_vector_targets(x, masks, tuple(grid))[0].flatten(2).numpy()
        images["t1"].append(z[: len(z) // 2])
        images["flair"].append(z[len(z) // 2 :])
    interior = (occupancy / count >= INTERIOR).numpy()
    return np.concatenate(factors), foreground.numpy(), interior, {v: np.concatenate(z) for v, z in images.items()}


def load_run(run_dir, checkpoint):
    import torch

    from eval.protocol.score_checkpoint import load_settings

    settings = load_settings(run_dir)
    check_run(run_dir, settings)
    if settings.get("synthetic_lesion_target") != "burden":
        raise SystemExit(f"{run_dir}: not a burden run (--synthetic-lesion-target burden)")
    path = os.path.join(run_dir, "training_progress.json")
    progress = {}
    if os.path.isfile(path):
        with open(path) as fp:
            progress = json.load(fp)
    init = os.path.join(run_dir, "model_init.pt")
    if not os.path.isfile(init):
        raise SystemExit(f"{run_dir}: no model_init.pt, so no untrained floor")
    trained = os.path.join(run_dir, checkpoint)
    if not os.path.isfile(trained):
        raise SystemExit(f"{run_dir}: no {checkpoint}")
    return {
        "dir": run_dir,
        "settings": settings,
        "progress": progress,
        "init": {k: v.numpy() for k, v in torch.load(init, map_location="cpu").items()},
        "trained": torch.load(trained, map_location="cpu"),
    }


def run(args):
    import torch

    from eval.metrics.dci import content_factor_names
    from eval.protocol.score_checkpoint import build_model, make_dataset
    from utils.encoder_runtime import select_encoder_device

    device = select_encoder_device(args.device)
    runs = [load_run(run_dir, args.checkpoint) for run_dir in args.run_dirs]
    control = runs[0]
    num_samples = args.num_samples or control["settings"]["num_val_samples"]
    key = data_key(control["settings"], args.split, num_samples)
    unmatched = {}
    for other in runs[1:]:
        problems = match_problems(control, other)
        if data_key(other["settings"], args.split, num_samples) != key:
            problems.insert(0, "generator: these runs do not score the same subjects")
        if problems:
            unmatched[other["dir"]] = problems
            print(f"NOT MATCHED with the control: {other['dir']}", flush=True)
            for problem in problems:
                print(f"  {problem}", flush=True)
    if unmatched and not args.allow_unmatched:
        raise SystemExit("Refusing unmatched runs; --allow-unmatched scores them anyway")
    grid = [list(r["settings"].get("train_patch_grid") or [8, 8, 8]) for r in runs]
    if any(g != grid[0] for g in grid):
        raise SystemExit(f"Runs read different cell grids: {grid}")
    grid = grid[0]
    if any(p > g for p, g in zip(args.probe_grid, grid)):
        raise SystemExit(f"--probe-grid {args.probe_grid} exceeds the cell grid {grid}")

    cfg = control["settings"]
    dataset = make_dataset(cfg, num_samples, mode=args.split)
    threshold = cfg.get("patch_foreground_thresh", 0.05)
    factors, foreground, interior, images = subject_table(dataset, grid, args.batch_size, threshold)
    if not interior.any():
        raise SystemExit(f"No cell of grid {grid} is at least {INTERIOR:g} brain on average; use a finer grid")
    target = factors[:, content_factor_names(factors.shape[1], "burden").index("lesion_burden")]
    voxels = lesion_fractions(dataset, (1, 1, 1))[:, 0] * cfg["res"] ** 3
    bins = lesion_bins(voxels)
    print(
        f"{args.split} split, {len(target)} subjects; rendered lesion voxels median {np.median(voxels):.0f}, "
        + ", ".join(f"{name} n={int(rows.sum())}" for name, rows in bins.items())
        + f"; {int(foreground.sum())}/{foreground.size} cells hold foreground, {int(interior.sum())} are interior",
        flush=True,
    )

    report = {
        "split": args.split,
        "subjects": len(target),
        "checkpoint": args.checkpoint,
        "grid": grid,
        "probe_grid": args.probe_grid,
        "bins": {name: int(rows.sum()) for name, rows in bins.items()},
        "unmatched": unmatched,
        "image": {
            view: seed_mean_scores(
                np.stack([quantile_predictions(images[view][:, :, interior], target, seed) for seed in SEEDS]),
                target,
                bins,
            )
            for view in VIEWS
        },
        "runs": [],
    }
    for current in runs:
        scored = {}
        for stage, state in (("floor", current["init"]), ("trained", current["trained"])):
            state = {k: torch.as_tensor(v) for k, v in state.items()}
            model = build_model(current["settings"], device, state).eval()
            rep = encode(model, grid, dataset, device, args.batch_size)
            del model
            scored[stage] = score_representations(rep, target, bins, args.probe_grid, grid, foreground, interior)
            print(f"  scored {current['dir']} at {stage}", flush=True)
        current["means"] = scored["trained"][1]
        report["runs"].append(
            {
                "run_dir": current["dir"],
                "step": current["progress"].get("step"),
                "trained": scored["trained"][0],
                "floor": scored["floor"][0],
            }
        )

    for result, other in zip(report["runs"][1:], runs[1:]):
        result["minus_control"] = {
            rep: {
                name: paired_delta(control["means"][rep], other["means"][rep], target, rows, args.bootstrap)
                for name, rows in bins.items()
            }
            for rep in other["means"]
        }
    print_report(report)
    if args.out:
        with open(args.out, "w") as fp:
            json.dump(report, fp, indent=2)
        print(f"wrote {args.out}", flush=True)
    return report


def print_report(report):
    runs = report["runs"]
    print("\nHeld-out burden recovery: Spearman (R²), trained then the run's own untrained floor", flush=True)
    for label, result in zip("ABCDEFG", runs):
        print(f"  {label}: {result['run_dir']} (step {result['step']})", flush=True)
    names = list(report["bins"])
    print("\nimage/quantile, no network: the probe's positive control", flush=True)
    for view, scores in report["image"].items():
        print(f"  {view:>5s} " + "  ".join(f"{n} {scores[n]['spearman']:+.3f}" for n in names), flush=True)
    for rep in runs[0]["trained"]:
        print(f"\n{rep}", flush=True)
        for name in names:
            line = f"  {name:>5s} n={report['bins'][name]:<5d}"
            for label, result in zip("ABCDEFG", runs):
                now, floor = result["trained"][rep][name], result["floor"][rep][name]
                line += f"  {label} {now['spearman']:+.3f} ({now['r2']:+.3f}) floor {floor['spearman']:+.3f}"
                delta = result.get("minus_control", {}).get(rep, {}).get(name)
                if delta is not None:
                    line += f"  {label}-A {delta['delta']:+.3f} [{delta['low']:+.3f}, {delta['high']:+.3f}]"
            print(line, flush=True)


def self_test():
    """Burden in randomly placed cells: only the quantile probe reads it, and tails at the edge can hide it."""
    rng = np.random.default_rng(0)
    n, channels, grid = 300, 4, (4, 4, 4)
    positions = int(np.prod(grid))
    burden = rng.normal(size=n)
    amount = (np.tanh(burden) + 1) / 2
    cells = 0.2 * rng.normal(size=(n, channels, positions))
    place = rng.integers(positions, size=n)
    cells[np.arange(n), 0, place] += 6 * amount
    kept = np.ones(positions, dtype=bool)
    flat = _flat(cells, grid, kept, grid)
    voxels = 150 * amount
    bins = lesion_bins(voxels)
    assert bins["<10"].sum() + (voxels >= 10).sum() == n and bins["all"].all()

    linear = bin_scores(held_out(flat, burden, 0), burden, bins)["all"]["spearman"]
    read = bin_scores(quantile_predictions(cells, burden, 0), burden, bins)["all"]["spearman"]
    assert linear < 0.5 < 0.8 < read, (linear, read)
    edge = np.concatenate([cells, 3 * rng.standard_cauchy(size=(n, channels, positions))], axis=2)
    buried = bin_scores(quantile_predictions(edge, burden, 0), burden, bins)["all"]["spearman"]
    assert buried < read - 0.2, (buried, read)
    vector = np.column_stack([np.tanh(burden), rng.normal(size=(n, 3))])
    assert bin_scores(kernel_predictions(vector, burden, 0), burden, bins)["all"]["spearman"] > 0.9
    centred = burden + 100
    offset = bin_scores(kernel_predictions(vector, centred, 0), centred, bins)["all"]["r2"]
    assert offset > 0.8, "the kernel has no intercept, so the target must be centred"

    # Occupied cells only: an all-background probe cell must not enter the flat features.
    half = kept.copy()
    half[: positions // 2] = False
    assert _flat(cells, grid, half, (2, 1, 1)).shape == (n, channels)

    tiny = bin_scores(burden, burden, {"few": np.arange(n) < 3})["few"]
    assert np.isnan(tiny["spearman"]) and tiny["n"] == 3
    good = burden + 0.1 * rng.normal(size=n)
    delta = paired_delta(rng.normal(size=n), good, burden, bins["all"], 200)
    assert delta["low"] > 0.5
    assert paired_delta(good, good, burden, bins["all"], 200) == {"delta": 0.0, "low": 0.0, "high": 0.0}

    base = {
        "settings": {"lr": 1e-4, "model_id": "a", "vector_prediction_weight": 0.0},
        "progress": {key: 1 for key in MATCHED_PROGRESS},
        "init": {"encoder.w": np.ones(2)},
    }
    same = {
        "settings": {"lr": 1e-4, "model_id": "b", "vector_prediction_weight": 1.0},
        "progress": dict(base["progress"]),
        "init": {"encoder.w": np.ones(2), "vector_prediction.t": np.zeros(2)},
    }
    assert match_problems(base, same) == []
    changed = {
        "settings": {**same["settings"], "lr": 1e-3},
        "progress": {**same["progress"], "training_input_sha256": 2},
        "init": {"encoder.w": np.zeros(2)},
    }
    assert len(match_problems(base, changed)) == 3
    assert any("step" in p for p in match_problems({**base, "progress": {}}, same))
    print("self-test passed")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
