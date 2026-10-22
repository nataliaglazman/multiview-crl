"""Do a run's regional codes find the lesion? Cell-by-cell lesion detection, against the image itself.

``regional_probe`` asks whether one linear map over all cells' codes recovers the lesion's coordinates. That mixes
two things: whether the code of the cell holding the lesion responds to it (detection), and whether a linear map
can turn "which cell responded" into a coordinate (readout, which is not linear). This probe asks the first
directly, on the run's own val subjects and the run's ``--train-patch-grid`` cells. For every subject and every
candidate cell (a cell that holds lesion in at least one subject), the features are the cell's

- **regional** code: the spatial head's content units for that cell (the global head applied per cell when the
  run has no ``--patch-loss-weight``);
- **map**: the backbone features averaged over the cell;
- **image**: the cell's own intensities (mean, SD, min, max), a model-free reference for how visible the lesion
  is in that view.

Labels: a cell is positive when it holds at least half of the subject's largest per-cell lesion load, negative when
it holds none; cells in between are left out. Each cell position's mean over the training subjects is subtracted
first, so a cell is judged against the same cell in other subjects, not against cells elsewhere in the brain.
One logistic probe per representation, fitted on 4/5 of the subjects and scored on the rest (5 folds, seeds 0-2):

- **AUC**: positive vs negative cells of the held-out subjects;
- **hit**: the share of held-out subjects whose top-scoring cell holds lesion, against **chance** (the share of
  candidate cells that do), and **exact**: the top cell is the subject's largest-load cell.

Scored at model_init.pt and model.pt, per view.

Example:
    python -m eval.encoder.regional_lesion_detection --run-dirs results/encoder/runs/<plain> results/encoder/runs/patch8
    python -m eval.encoder.regional_lesion_detection --self-test
"""

import argparse
import json
import os

import numpy as np

from eval.encoder.regional_probe import SEEDS, VIEWS, check_run, data_key

IMAGE_STATS = ("mean", "sd", "min", "max")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dirs", nargs="+", help="Encoder-only run directories (settings.json + checkpoints)")
    p.add_argument("--checkpoints", nargs="+", default=["model_init.pt", "model.pt"])
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--num-samples", type=int, default=None, help="Subjects (default: the run's val size)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--out", default=None, help="Also write the scores to this JSON file")
    p.add_argument("--self-test", action="store_true", help="Check the detector on toy cells; no torch")
    args = p.parse_args(argv)
    if not args.self_test and not args.run_dirs:
        p.error("--run-dirs is required unless --self-test")
    return args


def labels_from_fractions(fractions):
    """Per subject and cell: 1 (at least half the subject's largest cell load), 0 (no lesion), -1 (in between)."""
    fractions = np.asarray(fractions, dtype=np.float64)
    peak = fractions.max(1, keepdims=True)
    labels = np.full(fractions.shape, -1, dtype=int)
    labels[fractions == 0] = 0
    labels[(fractions >= 0.5 * peak) & (peak > 0)] = 1
    return labels


def detect(features, fractions, seeds=SEEDS, folds=5):
    """Held-out cell-level lesion detection from (subjects, cells, features) and per-cell lesion loads.

    Only candidate cells (lesion in at least one subject) are scored. Returns AUC, hit, exact and their chance
    levels, averaged over ``seeds``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    fractions = np.asarray(fractions, dtype=np.float64)
    candidates = np.flatnonzero(fractions.max(0) > 0)
    X = np.asarray(features, dtype=np.float64)[:, candidates]
    load = fractions[:, candidates]
    labels = labels_from_fractions(fractions)[:, candidates]
    largest = load.argmax(1)
    n, p, f = X.shape
    out = {"auc": [], "hit": [], "exact": []}
    for seed in seeds:
        scores = np.zeros((n, p))
        for train, test in KFold(folds, shuffle=True, random_state=seed).split(np.arange(n)):
            mean = X[train].mean(0)  # each cell against the same cell in the training subjects
            rows = labels[train].ravel() >= 0
            scaler = StandardScaler().fit((X[train] - mean).reshape(-1, f)[rows])
            probe = LogisticRegression(class_weight="balanced", max_iter=2000)
            probe.fit(scaler.transform((X[train] - mean).reshape(-1, f)[rows]), labels[train].ravel()[rows])
            scores[test] = probe.decision_function(scaler.transform((X[test] - mean).reshape(-1, f))).reshape(-1, p)
        scored = labels >= 0
        out["auc"].append(roc_auc_score(labels[scored], scores[scored]))
        top = scores.argmax(1)
        out["hit"].append(np.mean(load[np.arange(n), top] > 0))
        out["exact"].append(np.mean(top == largest))
    result = {k: float(np.mean(v)) for k, v in out.items()}
    result["chance_hit"] = float(np.mean((load > 0).sum(1) / p))
    result["chance_exact"] = 1.0 / p
    result["candidate_cells"] = int(p)
    return result


def lesion_fractions(dataset, grid):
    """Every subject's lesion load averaged over each ``grid`` cell: (subjects, cells), x-major."""
    import torch
    import torch.nn.functional as F

    inner = dataset._inner
    rows = []
    for idx in range(len(dataset)):
        draw = inner._draw_pseudo_mri(inner.sample_seed_for(idx))
        _, lesion = inner.renderer.render_structure(
            draw["z_content"],
            draw["z_deformation"],
            draw["z_fissure"],
            "cpu",
            clean=inner.clean_content,
            z_lesion=draw["z_lesion"],
        )
        cells = F.adaptive_avg_pool3d(torch.as_tensor(lesion, dtype=torch.float32)[None, None], tuple(grid))
        rows.append(cells.flatten().numpy())
    return np.stack(rows)


def image_cells(x, grid):
    """Per cell of a (B, 1, D, H, W) batch: its intensities' mean, SD, min and max, as (B, 4, cells)."""
    import torch
    import torch.nn.functional as F

    mean = F.adaptive_avg_pool3d(x, tuple(grid))
    sd = (F.adaptive_avg_pool3d(x * x, tuple(grid)) - mean * mean).clamp_min(0).sqrt()
    high = F.adaptive_max_pool3d(x, tuple(grid))
    low = -F.adaptive_max_pool3d(-x, tuple(grid))
    return torch.cat([mean, sd, low, high], dim=1).flatten(2)


def run(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from eval.encoder.regional_probe import regional_cells
    from eval.protocol.score_checkpoint import build_model, load_settings, make_dataset
    from utils.encoder_runtime import select_encoder_device

    device = select_encoder_device(args.device)
    datasets, columns = {}, []
    for letter, run_dir in zip((chr(ord("A") + i) for i in range(len(args.run_dirs))), args.run_dirs):
        cfg = load_settings(run_dir)
        check_run(run_dir, cfg)
        if cfg.get("synthetic_lesion_target", "position") != "position":
            raise SystemExit(f"{run_dir}: lesion target {cfg['synthetic_lesion_target']!r}; this needs one lesion")
        grid = list(cfg.get("train_patch_grid") or [8, 8, 8])
        num_samples = args.num_samples or cfg["num_val_samples"]
        key = json.dumps([data_key(cfg, args.split, num_samples), grid])
        datasets.setdefault(key, {"cfg": cfg, "num_samples": num_samples, "grid": grid, "letters": []})
        datasets[key]["letters"].append(letter)
        print(f"run {letter}: {run_dir} (data set {list(datasets).index(key) + 1}; {grid} cells)", flush=True)
        for name in args.checkpoints:
            path = os.path.join(run_dir, name)
            if os.path.isfile(path):
                columns.append({"letter": letter, "name": name, "checkpoint": path, "key": key, "cfg": cfg})
            else:
                print(f"  {name}: missing, skipped", flush=True)
    if not columns:
        raise SystemExit("No checkpoint was found in any run directory")

    report = {"split": args.split, "data_sets": []}
    for key, data in datasets.items():
        mine = [c for c in columns if c["key"] == key]
        dataset = make_dataset(data["cfg"], data["num_samples"], mode=args.split)
        fractions = lesion_fractions(dataset, data["grid"])
        keep = np.flatnonzero(fractions.max(0) > 0)  # candidate cells; the rest never hold lesion
        models = [build_model(c["cfg"], device, torch.load(c["checkpoint"], map_location="cpu")).eval() for c in mine]
        feats = {(c["letter"], c["name"], r): {v: [] for v in VIEWS} for c in mine for r in ("regional", "map")}
        image = {v: [] for v in VIEWS}
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=args.batch_size):
                x = torch.cat(batch["image"], dim=0).to(device)
                b = x.shape[0] // 2
                stats = image_cells(x, data["grid"])[:, :, keep].transpose(1, 2).cpu().numpy()
                image["t1"].append(stats[:b])
                image["flair"].append(stats[b:])
                for column, model in zip(mine, models):
                    h = model._encode(x, 2, None)
                    for name, cells in (
                        ("regional", regional_cells(model, h, data["grid"])),
                        ("map", F.adaptive_avg_pool3d(h, tuple(data["grid"])).flatten(2)),
                    ):
                        cells = cells[:, :, keep].transpose(1, 2).double().cpu().numpy()
                        feats[(column["letter"], column["name"], name)]["t1"].append(cells[:b])
                        feats[(column["letter"], column["name"], name)]["flair"].append(cells[b:])
        del models
        loads = fractions[:, keep]
        entry = {"runs": data["letters"], "subjects": data["num_samples"], "grid": data["grid"], "scores": {}}
        for view in VIEWS:
            rows = [("image cells", detect(np.concatenate(image[view]), loads))]
            for (letter, name, readout), parts in feats.items():
                label = f"{letter} {'init ' if name == 'model_init.pt' else ''}{readout}"
                rows.append((label, detect(np.concatenate(parts[view]), loads)))
            entry["scores"][view] = dict(rows)
            first = rows[0][1]
            print(
                f"\n{view}: lesion-cell detection over {first['candidate_cells']} candidate cells "
                f"(chance: hit {first['chance_hit']:.2f}, exact {first['chance_exact']:.3f})",
                flush=True,
            )
            print(f"  {'':22s}{'AUC':>8s}{'hit':>8s}{'exact':>8s}", flush=True)
            for label, s in rows:
                print(f"  {label:22s}{s['auc']:8.3f}{s['hit']:8.3f}{s['exact']:8.3f}", flush=True)
        report["data_sets"].append(entry)
    print(
        "\nReading: 'image cells' is how visible the lesion is in that view's own intensities. A representation "
        "that finds the lesion approaches it; one near AUC 0.5 and chance hit does not respond to it.",
        flush=True,
    )
    if args.out:
        with open(args.out, "w") as fp:
            json.dump(report, fp, indent=2)
        print(f"wrote {args.out}", flush=True)
    return report


def self_test():
    """A lesion signal on top of a strong per-cell baseline is found once each cell is compared with itself."""
    rng = np.random.default_rng(0)
    n, p, f = 200, 30, 4
    baseline = 3.0 * rng.normal(size=(1, p, f))  # every cell position looks different
    features = baseline + rng.normal(size=(n, p, f))
    fractions = np.zeros((n, p))
    lesion = rng.integers(0, p, size=n)
    fractions[np.arange(n), lesion] = 1.0
    fractions[np.arange(n), (lesion + 1) % p] = 0.2  # partly covered neighbours are left out of the AUC
    assert (labels_from_fractions(fractions)[np.arange(n), (lesion + 1) % p] == -1).all()
    blind = detect(features, fractions)
    assert abs(blind["auc"] - 0.5) < 0.08 and blind["exact"] < 0.15, blind
    features[np.arange(n), lesion, 0] += 4.0
    found = detect(features, fractions)
    assert found["auc"] > 0.95 and found["exact"] > 0.8 and found["hit"] >= found["exact"], found
    assert abs(found["chance_exact"] - 1 / p) < 1e-12 and found["candidate_cells"] == p
    print("self-test passed")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
