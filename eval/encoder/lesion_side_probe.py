"""Is a run's lesion_x the lesion's hemisphere or its position? A side probe and within-hemisphere probes.

Under ``--synthetic-lesion-placement wm_interior`` the lesion_x latent picks the lesion's hemisphere first:
the midline splits the admissible white matter in two, so the lower half of the latent's range lands in
one hemisphere and the upper half in the other. A code that knows only the side therefore scores lesion_x
up to 2/pi = 0.64 with the training log's ridge probe, about what a perfect localiser scores, while
lesion_y and lesion_z stay at their floor. The per-factor table cannot tell the two apart; this can.
Under ``canonical`` placement the latent sets the centre directly, so the side is its sign and the same
ceiling applies.

- **side**: a logistic probe for the hemisphere of the rendered lesion centre (left or right of the
  midline) from the content code. Accuracy and AUC, stratified 5-fold CV, seeds 0-2.
- **within hemisphere**: ridge R² of lesion_x, and of the rendered centre's x, fitted and scored inside
  each hemisphere separately, so the side cannot help; the two hemispheres are averaged by subject count.
  Near 0 with a high side accuracy means the code holds the side only.
- **all subjects**: ridge R² of lesion_x / y / z, as ``dci_step*.json`` reports them (``cv_probe_r2``,
  5-fold, seeds 0-2). On the run's own val subjects the T1 rows are the log's content→content numbers.
- **references**, no model involved: lesion_x from the true side alone (the most a side-only code can
  score), lesion_x / y / z from the true centre (a perfect localiser), and lesion_x within each
  hemisphere from the true centre.

Each run is scored at model_init.pt (step 0; after --init-from, the loaded weights) and model.pt.

Example:
    python -m eval.encoder.lesion_side_probe --run-dirs \
        results/encoder/runs/conv_mlp_lesion_anat6_c12_s42 results/encoder/runs/conv_mlp_lesion_anat6_c12_s142
    python -m eval.encoder.lesion_side_probe --self-test
"""

import argparse
import json
import os

import numpy as np

from eval.metrics.identifiability_metrics import cv_probe_acc, cv_probe_r2

VIEWS = ("t1", "flair")
LESION = ("lesion_x", "lesion_y", "lesion_z")
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
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--out", default=None, help="Also write the scores to this JSON file")
    p.add_argument("--self-test", action="store_true", help="Check the probes on toy codes; no torch")
    args = p.parse_args(argv)
    if not args.self_test and not args.run_dirs:
        p.error("--run-dirs is required unless --self-test")
    return args


def hemisphere(centres, res):
    """1 where the lesion centre (voxel indices, x first) lies right of the midline, x > 0."""
    return (np.asarray(centres)[:, 0] > (res - 1) / 2).astype(int)


def cv_probe_auc(X, y, n_splits=5, seeds=(0, 1, 2)):
    """``cv_probe_acc``'s logistic probe and folds, scored by held-out ROC AUC instead of accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    if X.shape[1] == 0 or X.shape[0] < n_splits * 4 or np.bincount(y, minlength=2).min() < n_splits:
        return float("nan")
    per_seed = []
    for seed in seeds:
        scores = np.zeros(len(y))
        for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000, random_state=seed).fit(sc.transform(X[tr]), y[tr])
            scores[te] = clf.decision_function(sc.transform(X[te]))
        per_seed.append(roc_auc_score(y, scores))
    return float(np.mean(per_seed))


def within_hemisphere_r2(features, target, side):
    """Ridge R² fitted and scored inside each hemisphere; the mean is weighted by subject count."""
    features = np.asarray(features, dtype=np.float64).reshape(len(side), -1)
    per_side = {}
    for name, value in (("left", 0), ("right", 1)):
        rows = side == value
        per_side[name] = cv_probe_r2(features[rows], target[rows])["mean"]
    weights = np.array([(side == 0).sum(), (side == 1).sum()], dtype=float)
    scores = np.array([per_side["left"], per_side["right"]])
    finite = np.isfinite(scores)
    per_side["mean"] = (
        float((scores[finite] * weights[finite]).sum() / weights[finite].sum()) if finite.any() else np.nan
    )
    return per_side


def score_code(code, lesion, centres, side):
    """Every probe on one view's content code.

    ``lesion`` is the (N, 3) lesion_x/y/z latents, ``centres`` the rendered centres in voxels (x first)
    and ``side`` their hemisphere, 0 or 1.
    """
    code = np.asarray(code, dtype=np.float64)
    return {
        "side_accuracy": cv_probe_acc(code, side)["mean"],
        "side_auc": cv_probe_auc(code, side),
        "all_subjects": {name: cv_probe_r2(code, lesion[:, j])["mean"] for j, name in enumerate(LESION)},
        "centre_all_subjects": {axis: cv_probe_r2(code, centres[:, j])["mean"] for j, axis in enumerate("xyz")},
        "lesion_x_within_hemisphere": within_hemisphere_r2(code, lesion[:, 0], side),
        "centre_x_within_hemisphere": within_hemisphere_r2(code, centres[:, 0], side),
    }


def references(lesion, centres, side):
    """What no-model codes score: the true side alone, and the true centre (a perfect localiser)."""
    return {
        "lesion_x_from_side": cv_probe_r2(side[:, None].astype(float), lesion[:, 0])["mean"],
        "from_centre": {name: cv_probe_r2(centres, lesion[:, j])["mean"] for j, name in enumerate(LESION)},
        "lesion_x_within_hemisphere_from_centre": within_hemisphere_r2(centres, lesion[:, 0], side),
        "right_share": float(side.mean()),
        "majority_accuracy": float(max(side.mean(), 1 - side.mean())),
    }


def lesion_centres(dataset):
    """Rendered lesion centre of every subject in voxel indices, x first; NaN where no lesion was drawn.

    Structure only, from the accepted draw the dataset renders (``sample_seed_for``), so no view is
    rendered twice.
    """
    inner = dataset._inner
    centres = []
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
        mask = lesion.numpy() > 0
        centres.append(np.argwhere(mask).mean(0) if mask.any() else np.full(3, np.nan))
    return np.array(centres)


def data_key(cfg, split, num_samples):
    """Runs with the same key score the same rendered subjects."""
    keys = sorted({*DATA_KEYS, *(k for k in cfg if k.startswith("synthetic_"))})
    return json.dumps([split, num_samples, *[(k, cfg.get(k)) for k in keys]], sort_keys=True)


def check_run(run_dir, cfg):
    if cfg.get("dataset_name", "synthetic") != "synthetic":
        raise SystemExit(f"{run_dir}: trained on real data ({cfg['dataset_name']}); there is no lesion to locate")
    if cfg.get("synthetic_lesion_target", "position") != "position":
        raise SystemExit(f"{run_dir}: lesion target is {cfg['synthetic_lesion_target']!r}, not one lesion's position")
    if cfg.get("n_content", 0) < 5:
        raise SystemExit(f"{run_dir}: n_content {cfg.get('n_content')} has no lesion_x/y/z latents (z_content[2:5])")


def run(args):
    import torch

    from eval.encoder.within_cluster_probe import encode_runs
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
        if key not in datasets:
            datasets[key] = {"cfg": cfg, "num_samples": num_samples, "res": cfg["res"], "letters": []}
        datasets[key]["letters"].append(letter)
        progress = os.path.join(run_dir, "training_progress.json")
        step = None
        if os.path.isfile(progress):
            with open(progress) as fp:
                step = json.load(fp).get("step")
        print(f"run {letter}: {run_dir} (step {step}; data set {list(datasets).index(key) + 1})", flush=True)
        if cfg.get("init_from"):
            print(f"  warm start: model_init.pt holds {cfg['init_from']}'s weights, not a random init", flush=True)
        for name in args.checkpoints:
            path = os.path.join(run_dir, name)
            if not os.path.isfile(path):
                print(f"  {name}: missing, skipped", flush=True)
                continue
            label = f"{letter} {name}"
            columns.append({"label": label, "run_dir": run_dir, "checkpoint": path, "key": key, "cfg": cfg})
            columns[-1]["step"] = 0 if name == "model_init.pt" else step
    if not columns:
        raise SystemExit("No checkpoint was found in any run directory")

    # One rendering pass per data set: every checkpoint that scores it encodes each batch in turn.
    for key, data in datasets.items():
        mine = [c for c in columns if c["key"] == key]
        if not mine:
            continue
        dataset = make_dataset(data["cfg"], data["num_samples"], mode=args.split)
        models = [build_model(c["cfg"], device, torch.load(c["checkpoint"], map_location="cpu")) for c in mine]
        codes, factors = encode_runs(models, dataset, device, args.batch_size)
        del models
        for column, code in zip(mine, codes):
            column["codes"] = code
        # After the encoding pass, so every subject's accepted draw is already known.
        data["centres"] = lesion_centres(dataset)
        data["lesion"] = factors[:, 2:5]
    datasets = {key: data for key, data in datasets.items() if "centres" in data}

    report = {
        "split": args.split,
        "probe": "ridge and logistic probes of eval.metrics.identifiability_metrics (5-fold, seeds 0-2)",
        "data_sets": [],
        "columns": [],
    }
    for number, data in enumerate(datasets.values(), start=1):
        valid = np.isfinite(data["centres"]).all(1)
        data["valid"] = valid
        data["side"] = hemisphere(data["centres"][valid], data["res"])
        refs = references(data["lesion"][valid], data["centres"][valid], data["side"])
        report["data_sets"].append(
            {"runs": data["letters"], "subjects": int(valid.sum()), "without_lesion": int((~valid).sum()), **refs}
        )
        within = refs["lesion_x_within_hemisphere_from_centre"]
        print(
            f"\ndata set {number} (runs {', '.join(data['letters'])}): {int(valid.sum())} subjects, "
            f"{refs['right_share']:.0%} right of the midline (a majority guess scores side accuracy "
            f"{refs['majority_accuracy']:.2f})"
            + (f", {int((~valid).sum())} without a lesion dropped" if (~valid).any() else ""),
            flush=True,
        )
        print(
            f"  references, no model: lesion_x from the true side alone {refs['lesion_x_from_side']:.3f}; "
            "from the true centre "
            + "/".join(f"{refs['from_centre'][name]:.3f}" for name in LESION)
            + f" (x/y/z); lesion_x within hemisphere from the true centre {within['mean']:.3f}",
            flush=True,
        )

    rows = (
        ("side accuracy", lambda s: s["side_accuracy"]),
        ("side AUC", lambda s: s["side_auc"]),
        ("lesion_x R², all subjects", lambda s: s["all_subjects"]["lesion_x"]),
        ("lesion_x R², within hemisphere", lambda s: s["lesion_x_within_hemisphere"]["mean"]),
        ("centre x R², within hemisphere", lambda s: s["centre_x_within_hemisphere"]["mean"]),
        ("lesion_y R², all subjects", lambda s: s["all_subjects"]["lesion_y"]),
        ("lesion_z R², all subjects", lambda s: s["all_subjects"]["lesion_z"]),
    )
    scores = {view: [] for view in VIEWS}
    for column in columns:
        data = datasets[column["key"]]
        lesion, centres = data["lesion"][data["valid"]], data["centres"][data["valid"]]
        for view in VIEWS:
            scores[view].append(score_code(column["codes"][view][data["valid"]], lesion, centres, data["side"]))
        report["columns"].append(
            {k: column[k] for k in ("label", "run_dir", "checkpoint", "step")}
            | {"scores": {view: scores[view][-1] for view in VIEWS}}
        )
    width = max(14, *(len(c["label"]) + 2 for c in columns))
    for view in VIEWS:
        print(f"\n{view} content code", flush=True)
        print(f"  {'':34s}" + "".join(f"{c['label']:>{width}s}" for c in columns), flush=True)
        for label, value in rows:
            print(f"  {label:34s}" + "".join(f"{value(s):>{width}.3f}" for s in scores[view]), flush=True)
    print(
        "\nReading: a side-only code has side accuracy near 1, lesion_x near the 'from the true side alone' "
        "reference, and within-hemisphere R² near 0. Position shows as within-hemisphere R² and lesion_y/z "
        "above their step-0 values; even the true centre leaves part of lesion_x unexplained within a "
        "hemisphere, so compare with that reference, not with 1.",
        flush=True,
    )
    if args.out:
        with open(args.out, "w") as fp:
            json.dump(report, fp, indent=2)
        print(f"wrote {args.out}", flush=True)
    return report


def self_test():
    """A side-only code and a position code separate on the within-hemisphere score, not on lesion_x."""
    rng = np.random.default_rng(0)
    n = 600
    lesion = rng.normal(size=(n, 3))
    side = (lesion[:, 0] > 0).astype(int)
    # A rendered centre that jumps across a midline gap, then moves with the latent inside each hemisphere.
    centres = np.column_stack(
        [31.5 + np.where(side == 1, 4.0, -4.0) + 3.0 * np.tanh(lesion[:, 0]), 30 + 3 * lesion[:, 1], 30 + lesion[:, 2]]
    )
    assert (hemisphere(centres, 64) == side).all()
    noise = 0.05 * rng.normal(size=(n, 4))

    side_only = np.column_stack([side, rng.normal(size=(n, 3))]) + noise
    s = score_code(side_only, lesion, centres, side)
    assert s["side_accuracy"] > 0.97 and s["side_auc"] > 0.99, s
    assert abs(s["all_subjects"]["lesion_x"] - 2 / np.pi) < 0.06, s["all_subjects"]
    assert s["lesion_x_within_hemisphere"]["mean"] < 0.05, s["lesion_x_within_hemisphere"]
    assert s["all_subjects"]["lesion_y"] < 0.05, s["all_subjects"]

    position = np.column_stack([lesion, rng.normal(size=n)]) + noise
    s = score_code(position, lesion, centres, side)
    assert s["lesion_x_within_hemisphere"]["mean"] > 0.9, s["lesion_x_within_hemisphere"]
    assert s["all_subjects"]["lesion_y"] > 0.9 and s["side_auc"] > 0.99, s

    refs = references(lesion, centres, side)
    assert abs(refs["lesion_x_from_side"] - 2 / np.pi) < 0.06, refs
    assert refs["from_centre"]["lesion_y"] > 0.95, refs

    # Each hemisphere is scored on its own subjects; the mean is weighted by their counts.
    target = np.where(side == 1, 10.0, -10.0) + rng.normal(size=n)
    w = within_hemisphere_r2(rng.normal(size=(n, 2)), target, side)
    assert w["left"] < 0.05 and w["right"] < 0.05, w
    print("self-test passed")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
