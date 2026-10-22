"""What a run holds about each content factor in its global code and in its regional representation.

A code read through the volume average (GAP) keeps what survives averaging: whole-brain amounts such as
brain size or ventricle volume do; focal and positional factors largely do not (the backbone map holds the
lesion's y and z, the average erases them). This probe reads three representations of the same val subjects,
per view and checkpoint:

- **code**: the global content code, the volume average through the global head; what ``dci_step*.json`` scores.
- **regional**: the spatial head on the run's ``--train-patch-grid`` cells of the backbone map (the head that
  ``--patch-loss-weight`` trains; without it the global head applied to each cell), content units only,
  averaged down to ``--probe-grid`` cells.
- **map**: the backbone map itself averaged to ``--probe-grid`` cells, before any head.

Targets: every named z_content factor, and with one position lesion its rendered centre: the side (AUC), x
within a hemisphere, and y and z over all subjects and within a hemisphere. Every target gets one probe at every
readout: ridge regression on standardized features, alphas 1e-2..1e6 chosen inside each training fold, 5 folds,
seeds 0-2, held-out R²; the side's AUC comes from the same ridge's held-out scores. Floors: the same probes at
model_init.pt and, for the lesion rows, an anatomy-only column (the true anatomy latents), since wm_interior
placement ties the side to anatomy a little.

Each target also has a natural readout, the one the summary reports: global factors (amounts, whole-brain
shape) from the code, regional ones (focal temporal atrophy, the corrugation-mode sulcal pattern, the lesion's
position) from the regional representation. The full tables show every readout, so a target read better
elsewhere is visible.

Style rows test the other half of identifiability, that content excludes style. Each view's representation is
scored against that view's own gain, bias and noise sigma as the renderer applied them
(``eval.diagnostics.style_path_audit.effective_style``: the raw draws are clipped, and the noise level uses
only their magnitude), at every readout. Untrained features carry much of the gain, so the floor is the same
readout at model_init.pt: excluded style sits near 0 and below it. A style summary reports every readout.

Example:
    python -m eval.encoder.regional_probe --run-dirs results/encoder/runs/<plain run> results/encoder/runs/<patch run>
    python -m eval.encoder.regional_probe --self-test
"""

import argparse
import json
import os

import numpy as np

VIEWS = ("t1", "flair")
READOUTS = ("code", "regional", "map")
LESION_COLUMNS = [2, 3, 4]  # lesion_x / y / z in z_content
ALPHAS = np.logspace(-2, 6, 17)
SEEDS = (0, 1, 2)
# Generator settings two runs must share to be scored on the same subjects.
DATA_KEYS = ("res", "n_content", "n_style", "data_seed", "dataset_name")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dirs", nargs="+", help="Encoder-only run directories (settings.json + checkpoints)")
    p.add_argument(
        "--checkpoints",
        nargs="+",
        default=["model_init.pt", "model.pt"],
        help="Checkpoint filenames inside each run directory; missing ones are skipped",
    )
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Subjects to score (default: each run's --num-val-samples, the set the training log scores)",
    )
    p.add_argument(
        "--probe-grid",
        type=int,
        nargs=3,
        default=[4, 4, 4],
        help="Cells the regional and map readouts are averaged to before probing (each <= the run's grid)",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--out", default=None, help="Also write the scores to this JSON file")
    p.add_argument("--self-test", action="store_true", help="Check the probes on toy representations; no torch")
    args = p.parse_args(argv)
    if not args.self_test and not args.run_dirs:
        p.error("--run-dirs is required unless --self-test")
    if min(args.probe_grid) < 1:
        p.error("--probe-grid needs positive cell counts")
    return args


def held_out(X, y, seed):
    """Held-out ridge predictions of ``y``: standardized features, RidgeCV over ALPHAS, 5 shuffled folds."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        scaler = StandardScaler().fit(X[tr])
        pred[te] = RidgeCV(alphas=ALPHAS).fit(scaler.transform(X[tr]), y[tr]).predict(scaler.transform(X[te]))
    return pred


def r2(X, y):
    """Held-out R², averaged over SEEDS; NaN for fewer than 20 subjects or a constant target."""
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(y) < 20 or not np.var(y) > 0:
        return float("nan")
    scores = [1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum() for p in (held_out(X, y, s) for s in SEEDS)]
    return float(np.mean(scores))


def auc(X, label):
    """ROC AUC of the held-out ridge scores for a 0/1 label, averaged over SEEDS."""
    from sklearn.metrics import roc_auc_score

    label = np.asarray(label).astype(int)
    if len(label) < 20 or np.bincount(label, minlength=2).min() < 5:
        return float("nan")
    X = np.asarray(X, dtype=np.float64)
    return float(np.mean([roc_auc_score(label, held_out(X, 2.0 * label - 1, s)) for s in SEEDS]))


def within(X, y, side):
    """R² fitted and scored inside each hemisphere separately, averaged by subject count."""
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    scores, weights = [], []
    for value in (0, 1):
        rows = side == value
        score = r2(X[rows], y[rows])
        if np.isfinite(score):
            scores.append(score)
            weights.append(rows.sum())
    return float(np.average(scores, weights=weights)) if scores else float("nan")


def adaptive_mean(cells, grid, out_grid):
    """(N, C, prod(grid)) cell values, x-major as ``flatten`` orders them, averaged to ``out_grid`` cells.

    The bins are torch's ``adaptive_avg_pool3d`` bins. Returns (N, C * prod(out_grid)).
    """
    cells = np.asarray(cells, dtype=np.float64)
    n, c = cells.shape[:2]
    cube = cells.reshape(n, c, *grid)
    bins = [[(i * g // p, -(-(i + 1) * g // p)) for i in range(p)] for g, p in zip(grid, out_grid)]
    out = np.empty((n, c, *out_grid))
    for i, (a0, a1) in enumerate(bins[0]):
        for j, (b0, b1) in enumerate(bins[1]):
            for k, (c0, c1) in enumerate(bins[2]):
                out[:, :, i, j, k] = cube[:, :, a0:a1, b0:b1, c0:c1].mean((2, 3, 4))
    return out.reshape(n, -1)


def natural_readout(target, cfg):
    """The readout a target is summarised at: regional for focal or positional targets, the code otherwise.

    None for a style row: content should exclude style at every readout, so it has no natural one.
    """
    if target.startswith("style "):
        return None
    if target.startswith("lesion ") or target == "temporal_atrophy":
        return "regional"
    if target == "sulcal_widening" and cfg.get("synthetic_sulcal_mode", "corrugation") == "corrugation":
        return "regional"  # a fixed zero-mean corrugation: its sign cancels under the average
    return "code"


def build_targets(factors, names, centres=None, side=None):
    """(name, kind, values) per target; kind is ``r2``, ``within`` or ``auc``.

    The single lesion's latents are replaced by its rendered centre (voxels, x first) when ``centres`` is
    given; the burden target's never-rendered slots are left out.
    """
    lesion_latents = ("lesion_x", "lesion_y", "lesion_z")
    targets = [
        (name, "r2", factors[:, j])
        for j, name in enumerate(names)
        if not name.startswith("unused_") and not (centres is not None and name in lesion_latents)
    ]
    if centres is not None:
        targets += [
            ("lesion side", "auc", side),
            ("lesion x within hemisphere", "within", centres[:, 0]),
            ("lesion y", "r2", centres[:, 1]),
            ("lesion y within hemisphere", "within", centres[:, 1]),
            ("lesion z", "r2", centres[:, 2]),
            ("lesion z within hemisphere", "within", centres[:, 2]),
        ]
    return targets


def style_targets(effective, names):
    """(name, kind, values) per style factor of one view: ``effective`` holds its rendered gain, bias, sigma."""
    return [(f"style {name}", "r2", effective[:, j]) for j, name in enumerate(names[: effective.shape[1]])]


def score(rep, targets, side=None):
    """Every target's probe score on one representation (subjects x features)."""
    out = {}
    for name, kind, values in targets:
        if kind == "auc":
            out[name] = auc(rep, values)
        elif kind == "within":
            out[name] = within(rep, values, side)
        else:
            out[name] = r2(rep, values)
    return out


def readouts(model, x, grid, probe_grid):
    """(code, regional, map) of a view-major batch ``x`` from one backbone pass, each (2B, features).

    ``regional`` repeats ``forward``'s pool_only + patch_grid path (a test checks the parity), keeps the
    content units and averages the cells down to ``probe_grid``.
    """
    import torch.nn.functional as F

    h = model._encode(x, 2, None)
    code = model._global_code(h, x, 2, None)
    code = code[:, model.unit_split(code.shape[1])[0]]
    if model.readout_type == "mlp":
        cells = F.adaptive_avg_pool3d(h, tuple(grid)).flatten(2)
        cells = model.patch_readout(cells.transpose(1, 2)).transpose(1, 2)
    else:
        cells = model._patch_pool(model.to_encoding(h), tuple(grid))
    cells = cells[:, model.unit_split(cells.shape[1])[0]]
    regional = F.adaptive_avg_pool3d(cells.reshape(*cells.shape[:2], *grid), tuple(probe_grid)).flatten(1)
    return code, regional, F.adaptive_avg_pool3d(h, tuple(probe_grid)).flatten(1)


def data_key(cfg, split, num_samples):
    """Runs with the same key score the same rendered subjects."""
    keys = sorted({*DATA_KEYS, *(k for k in cfg if k.startswith("synthetic_"))})
    return json.dumps([split, num_samples, *[(k, cfg.get(k)) for k in keys]], sort_keys=True)


def check_run(run_dir, cfg):
    if cfg.get("dataset_name", "synthetic") != "synthetic":
        raise SystemExit(f"{run_dir}: trained on real data ({cfg['dataset_name']}); there are no true factors to score")
    if cfg.get("synthetic_mode", "pseudo_mri") != "pseudo_mri":
        raise SystemExit(f"{run_dir}: synthetic mode {cfg['synthetic_mode']!r}, not pseudo_mri")


def encode(models, grids, dataset, device, batch_size, probe_grid):
    """One pass over ``dataset``: every model's readouts per view, the subjects' z_content, each view's style.

    Returns ``(reps, factors, styles)``. ``reps[m][view][readout]`` is (subjects, features) for model ``m``;
    ``styles[view]`` is the (subjects, n_style) gain / bias / noise sigma the renderer applied, absent when the
    samples carry no style draws or the dataset no renderer.
    """
    import torch
    from torch.utils.data import DataLoader

    from eval.diagnostics.style_path_audit import effective_style

    parts = [{view: {r: [] for r in READOUTS} for view in VIEWS} for _ in models]
    factors, styles = [], {view: [] for view in VIEWS}
    style_scale = getattr(getattr(getattr(dataset, "_inner", None), "renderer", None), "style_scale", None)
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size):
            x = torch.cat(batch["image"], dim=0).to(device)
            b = x.shape[0] // 2
            for rep, model, grid in zip(parts, models, grids):
                for name, value in zip(READOUTS, readouts(model, x, grid, probe_grid)):
                    value = value.double().cpu().numpy()
                    rep["t1"][name].append(value[:b])
                    rep["flair"][name].append(value[b:])
            latents = batch["gt_latents"]
            factors.append(latents["z_content"].double().numpy())
            if style_scale is not None and "z_style_v1" in latents:
                for v, view in enumerate(VIEWS):
                    draws = latents[f"z_style_v{v + 1}"]
                    styles[view].append(np.stack([effective_style(z, style_scale)[: draws.shape[1]] for z in draws]))
    reps = [{view: {r: np.concatenate(rep[view][r]) for r in READOUTS} for view in VIEWS} for rep in parts]
    styles = {view: np.concatenate(s) for view, s in styles.items() if s and s[0].shape[1] > 0}
    return reps, np.concatenate(factors), styles


def subject_targets(cfg, dataset, factors, styles):
    """``(content targets, per-view style targets, side, centres)`` for the scored subjects.

    Content targets are shared by both views; each view is scored against its own style. ``side`` and
    ``centres`` (rendered lesion centres, voxels, x first) are None unless the run has one position lesion.
    """
    from eval.encoder.lesion_side_probe import hemisphere, lesion_centres
    from eval.metrics.dci import STYLE_FACTOR_NAMES, content_factor_names

    names = content_factor_names(factors.shape[1], cfg.get("synthetic_lesion_target", "position"))
    centres = side = None
    if cfg.get("synthetic_lesion_target", "position") == "position" and factors.shape[1] >= 5:
        centres = lesion_centres(dataset)
        if not np.isfinite(centres).all():
            raise SystemExit("Some subjects have no rendered lesion; the centre rows need one per subject")
        side = hemisphere(centres, cfg["res"])
    style = {view: style_targets(values, STYLE_FACTOR_NAMES) for view, values in styles.items()}
    return build_targets(factors, names, centres, side), style, side, centres


def run(args):
    import torch

    from eval.protocol.score_checkpoint import build_model, load_settings, make_dataset
    from utils.encoder_runtime import select_encoder_device

    device = select_encoder_device(args.device)
    datasets, columns = {}, []
    letters = [chr(ord("A") + i) for i in range(len(args.run_dirs))]
    for letter, run_dir in zip(letters, args.run_dirs):
        cfg = load_settings(run_dir)
        check_run(run_dir, cfg)
        num_samples = args.num_samples or cfg["num_val_samples"]
        key = data_key(cfg, args.split, num_samples)
        datasets.setdefault(key, {"cfg": cfg, "num_samples": num_samples, "letters": []})["letters"].append(letter)
        grid = list(cfg.get("train_patch_grid") or [8, 8, 8])
        if any(p > g for p, g in zip(args.probe_grid, grid)):
            raise SystemExit(f"{run_dir}: --probe-grid {args.probe_grid} exceeds the run's cell grid {grid}")
        trained = cfg.get("patch_loss_weight", 0.0) > 0
        print(
            f"run {letter}: {run_dir} (data set {list(datasets).index(key) + 1}; regional = "
            f"{'the trained spatial head' if trained else 'the global head per cell (no --patch-loss-weight)'} "
            f"on {grid} cells)",
            flush=True,
        )
        for name in args.checkpoints:
            path = os.path.join(run_dir, name)
            if not os.path.isfile(path):
                print(f"  {name}: missing, skipped", flush=True)
                continue
            columns.append({"letter": letter, "name": name, "checkpoint": path, "key": key, "cfg": cfg, "grid": grid})
    if not columns:
        raise SystemExit("No checkpoint was found in any run directory")

    for key, data in datasets.items():
        mine = [c for c in columns if c["key"] == key]
        cfg = data["cfg"]
        dataset = make_dataset(cfg, data["num_samples"], mode=args.split)
        models = [build_model(c["cfg"], device, torch.load(c["checkpoint"], map_location="cpu")).eval() for c in mine]
        reps, factors, styles = encode(
            models, [c["grid"] for c in mine], dataset, device, args.batch_size, args.probe_grid
        )
        del models
        data["targets"], data["style"], side, _ = subject_targets(cfg, dataset, factors, styles)
        data["side"] = side
        lesion_rows = [t for t in data["targets"] if t[0].startswith("lesion ")]
        anatomy = np.delete(factors, LESION_COLUMNS, axis=1)
        data["anatomy_only"] = score(anatomy, lesion_rows, side) if lesion_rows else {}
        for column, rep in zip(mine, reps):
            column["scores"] = {
                view: {r: score(rep[view][r], data["targets"] + data["style"].get(view, []), side) for r in READOUTS}
                for view in VIEWS
            }

    report = {
        "split": args.split,
        "probe_grid": args.probe_grid,
        "probe": "ridge, alphas 1e-2..1e6, 5 folds, seeds 0-2",
    }
    report["data_sets"] = [
        {
            "runs": d["letters"],
            "subjects": d["num_samples"],
            "anatomy_only": d["anatomy_only"],
            "style_rows": {view: [t[0] for t in rows] for view, rows in d["style"].items()},
        }
        for d in datasets.values()
    ]
    report["columns"] = [
        {k: c[k] for k in ("letter", "name", "checkpoint", "grid")}
        | {"run_dir": args.run_dirs[ord(c["letter"]) - ord("A")], "scores": c["scores"]}
        for c in columns
    ]
    for view in VIEWS:
        for letter in dict.fromkeys(c["letter"] for c in columns):
            mine = [c for c in columns if c["letter"] == letter]
            data = datasets[mine[0]["key"]]
            labels = [f"{'init ' if c['name'] == 'model_init.pt' else ''}{r}" for c in mine for r in READOUTS]
            values = [c["scores"][view][r] for c in mine for r in READOUTS]
            if data["anatomy_only"]:
                labels.append("anatomy")
                values.append(data["anatomy_only"])
            print(
                f"\nrun {letter}, {view}: held-out R² (AUC for the side); * = the target's natural readout", flush=True
            )
            print(f"  {'':30s}" + "".join(f"{label:>15s}" for label in labels), flush=True)
            for name, _, _ in data["targets"] + data["style"].get(view, []):
                natural = natural_readout(name, data["cfg"])
                cells = []
                for label, v in zip(labels, values):
                    mark = "*" if label.split()[-1] == natural else " "
                    cells.append(f"{v.get(name, float('nan')):>14.3f}{mark}" if name in v else f"{'':>15s}")
                print(f"  {name:30s}" + "".join(cells), flush=True)
    print("\nSummary at each target's natural readout: trained (Δ vs the same readout at init)", flush=True)
    for view in VIEWS:
        print(f"  {view}", flush=True)
        for data in datasets.values():
            for name, _, _ in data["targets"]:
                natural = natural_readout(name, data["cfg"])
                line = f"    {name:30s}{natural:>10s}"
                for letter in data["letters"]:
                    by_name = {c["name"]: c["scores"][view][natural][name] for c in columns if c["letter"] == letter}
                    trained, init = by_name.get("model.pt"), by_name.get("model_init.pt")
                    if trained is None:
                        line += f"   {letter} {'n/a':>6s}"
                    elif init is None:
                        line += f"   {letter} {trained:6.3f}"
                    else:
                        line += f"   {letter} {trained:6.3f} ({trained - init:+.3f})"
                print(line, flush=True)
    if any(d["style"] for d in datasets.values()):
        print(
            "\nStyle exclusion: each view's own rendered gain, bias and noise sigma at every readout, trained "
            "(Δ vs init). Content that excludes style reads near 0 here, and below its untrained floor.",
            flush=True,
        )
        for view in VIEWS:
            print(f"  {view}", flush=True)
            for data in datasets.values():
                for name, _, _ in data["style"].get(view, []):
                    for letter in data["letters"]:
                        line = f"    {name:20s} {letter}"
                        for r in READOUTS:
                            by_name = {c["name"]: c["scores"][view][r][name] for c in columns if c["letter"] == letter}
                            trained, init = by_name.get("model.pt"), by_name.get("model_init.pt")
                            if trained is None:
                                line += f"   {r} {'n/a':>6s}"
                            elif init is None:
                                line += f"   {r} {trained:6.3f}"
                            else:
                                line += f"   {r} {trained:6.3f} ({trained - init:+.3f})"
                        print(line, flush=True)
    if args.out:
        with open(args.out, "w") as fp:
            json.dump(report, fp, indent=2)
        print(f"wrote {args.out}", flush=True)
    return report


def self_test():
    """A regional factor that cancels under the average reads from the cells only; a global one from both."""
    rng = np.random.default_rng(0)
    n, c, grid = 300, 3, (4, 4, 4)
    global_factor, regional_factor = rng.normal(size=n), rng.normal(size=n)
    cells = 0.3 * rng.normal(size=(n, c, int(np.prod(grid))))
    cells += global_factor[:, None, None]  # every cell moves with the global factor
    cells[:, 0, 0] += 2 * regional_factor  # one cell up and another down: the average cannot see it
    cells[:, 0, 1] -= 2 * regional_factor
    code = cells.mean(2)
    regional = adaptive_mean(cells, grid, grid)
    assert r2(code, global_factor) > 0.9 and r2(regional, global_factor) > 0.9
    assert r2(code, regional_factor) < 0.05 < 0.8 < r2(regional, regional_factor)
    # Averaging a 4x4x4 grid to 2x2x2 takes the mean of each 2x2x2 block.
    coarse = adaptive_mean(cells, grid, (2, 2, 2))
    assert coarse.shape == (n, c * 8)
    np.testing.assert_allclose(
        coarse.reshape(n, c, 8)[:, :, 0], cells.reshape(n, c, 4, 4, 4)[:, :, :2, :2, :2].mean((2, 3, 4))
    )
    # Uneven bins follow adaptive pooling: 5 cells to 2 split as [0, 3) and [2, 5).
    one = np.arange(5.0).reshape(1, 1, 5)
    np.testing.assert_allclose(adaptive_mean(one, (5, 1, 1), (2, 1, 1)), [[1.0, 3.0]])

    side = (rng.normal(size=n) > 0).astype(int)
    position = rng.normal(size=n) + 4 * side  # a hemisphere jump plus a within-hemisphere position
    rep = np.column_stack([side + 0.05 * rng.normal(size=n), rng.normal(size=(n, 2))])
    assert auc(rep, side) > 0.99 and within(rep, position, side) < 0.05
    targets = build_targets(np.column_stack([global_factor] * 5), ["brain_size", "x", "unused_3", "lesion_y", "y"])
    assert [t[0] for t in targets] == ["brain_size", "x", "lesion_y", "y"]
    centres = np.column_stack([position, global_factor, regional_factor])
    names = [
        t[0]
        for t in build_targets(np.zeros((n, 5)), ["brain_size", "v", "lesion_x", "lesion_y", "lesion_z"], centres, side)
    ]
    assert names[:2] == ["brain_size", "v"] and "lesion_x" not in names and names[2] == "lesion side", names
    assert natural_readout("lesion y", {}) == "regional" and natural_readout("brain_size", {}) == "code"
    assert natural_readout("sulcal_widening", {"synthetic_sulcal_mode": "atrophy"}) == "code"
    assert natural_readout("sulcal_widening", {}) == natural_readout("temporal_atrophy", {}) == "regional"

    # Style rows: one per rendered style factor of the view, scored like a content factor, no natural readout.
    effective = np.column_stack(
        [1 + 0.3 * np.tanh(global_factor), 0.1 * np.tanh(regional_factor), 0.01 + rng.random(n)]
    )
    rows = style_targets(effective, ["gain", "bias", "noise_sigma"])
    assert [r[0] for r in rows] == ["style gain", "style bias", "style noise_sigma"] and {r[1] for r in rows} == {"r2"}
    assert style_targets(effective[:, :2], ["gain", "bias", "noise_sigma"])[-1][0] == "style bias"
    assert natural_readout("style gain", {}) is None
    # A code that excludes style reads ~0 on it; one that carries the gain reads it.
    assert r2(regional, rows[0][2]) > 0.8 and r2(rng.normal(size=(n, 3)), rows[0][2]) < 0.05
    print("self-test passed")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
