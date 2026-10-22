"""The val subjects a run's probe gets most wrong, written as NIfTI to open in a viewer.

Encodes the run's own val subjects once with ``regional_probe``'s readouts, probes one target from one readout
and view with the same ridge (out-of-fold predictions, averaged over seeds 0-2), ranks the subjects by error,
and writes the worst ones (and, with ``--best``, the best ones for contrast) as:

- ``t1`` and ``flair``: the views as the encoder sees them, after ``--synthetic-normalize``;
- ``brainmask`` (the T1 view's), ``lesion`` (the rendered lesion load) and ``tissue`` (the renderer's label map:
  0 background, 1 CSF, 2 WM, 3 GM, then the generator's extra labels such as the fissure);

plus ``subjects.csv`` (every subject: rank, truth, prediction, error, the content factors, each view's rendered
style and, for one position lesion, its centre and side) and ``manifest.json``. The error is prediction minus
truth; for ``lesion side`` it is how far the held-out score falls on the wrong side (positive = misclassified).
Targets are named as in ``regional_probe``'s tables, e.g. "lesion y", "lesion y within hemisphere",
"temporal_atrophy", "sulcal_widening", "style gain".

Example:
    python -m eval.encoder.probe_worst_subjects --run-dir <run> --target "lesion y" --readout regional \
        --view flair --worst 10 --best 3 --out worst_flair_lesion_y
    python -m eval.encoder.probe_worst_subjects --self-test
"""

import argparse
import csv
import json
import os

import numpy as np

from eval.encoder.regional_probe import READOUTS, SEEDS, VIEWS, check_run, held_out

VOLUMES = ("t1", "flair", "brainmask", "lesion", "tissue")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", help="Encoder-only run directory (settings.json + checkpoint)")
    p.add_argument("--checkpoint", default="model.pt", help="Checkpoint filename inside the run directory")
    p.add_argument("--target", default="lesion y", help="Target to rank by, named as in regional_probe's tables")
    p.add_argument("--readout", choices=READOUTS, default="regional")
    p.add_argument("--view", choices=VIEWS, default="flair", help="Whose representation is probed")
    p.add_argument("--worst", type=int, default=10, help="Worst subjects to write")
    p.add_argument("--best", type=int, default=0, help="Best subjects to write too, for contrast")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--num-samples", type=int, default=None, help="Subjects to score (default: the run's val size)")
    p.add_argument("--probe-grid", type=int, nargs=3, default=[4, 4, 4])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument(
        "--voxel-size",
        type=float,
        default=1.0,
        help="Diagonal of the NIfTI affine; 1.0 is the repo's identity convention, ~2.8 makes a 64^3 volume "
        "brain-sized in a viewer",
    )
    p.add_argument("--out", default="probe_worst_subjects", help="Output directory (created; must be empty)")
    p.add_argument("--self-test", action="store_true", help="Check the ranking on toy data; no torch")
    args = p.parse_args(argv)
    if not args.self_test and not args.run_dir:
        p.error("--run-dir is required unless --self-test")
    if args.worst < 1 or args.best < 0:
        p.error("--worst must be positive and --best non-negative")
    return args


def subject_errors(X, kind, y, side=None):
    """Out-of-fold ``(prediction, error, badness)`` per subject; the ranking sorts by badness, worst first.

    ``r2`` and ``within`` (fitted inside each hemisphere): the prediction averaged over SEEDS, error =
    prediction - truth, badness = |error|. ``auc``: the held-out ridge score of the ±1 label; error =
    badness = how far the score falls on the wrong side, positive when the subject is misclassified.
    """
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if kind == "auc":
        sign = 2.0 * y - 1
        prediction = np.mean([held_out(X, sign, s) for s in SEEDS], axis=0)
        error = -sign * prediction
        return prediction, error, error
    prediction = np.empty(len(y))
    groups = [side == value for value in (0, 1)] if kind == "within" else [np.ones(len(y), dtype=bool)]
    for rows in groups:
        prediction[rows] = np.mean([held_out(X[rows], y[rows], s) for s in SEEDS], axis=0)
    error = prediction - y
    return prediction, error, np.abs(error)


def subject_volumes(dataset, index):
    """The five volumes written for one subject, as numpy arrays on the dataset's grid."""
    inner = dataset._inner
    item = dataset[index]
    draw = inner._draw_pseudo_mri(inner.sample_seed_for(index))
    tissue, lesion = inner.renderer.render_structure(
        draw["z_content"],
        draw["z_deformation"],
        draw["z_fissure"],
        "cpu",
        clean=inner.clean_content,
        z_lesion=draw["z_lesion"],
    )

    def array(volume, dtype=np.float32):
        return np.asarray(volume.detach().cpu().squeeze().numpy(), dtype=dtype)

    return {
        "t1": array(item["image"][0]),
        "flair": array(item["image"][1]),
        "brainmask": array(item["mask"][0], np.uint8),
        "lesion": array(lesion),
        "tissue": array(tissue, np.int16),
    }


def run(args):
    import nibabel as nib
    import torch

    from eval.encoder.regional_probe import encode, score, subject_targets
    from eval.metrics.dci import STYLE_FACTOR_NAMES, content_factor_names
    from eval.protocol.score_checkpoint import build_model, load_settings, make_dataset
    from utils.encoder_runtime import select_encoder_device

    if os.path.isdir(args.out) and os.listdir(args.out):
        raise SystemExit(f"{args.out} is not empty; choose a new --out")
    cfg = load_settings(args.run_dir)
    check_run(args.run_dir, cfg)
    checkpoint = os.path.join(args.run_dir, args.checkpoint)
    if not os.path.isfile(checkpoint):
        raise SystemExit(f"No checkpoint {checkpoint}")
    grid = list(cfg.get("train_patch_grid") or [8, 8, 8])
    if any(p > g for p, g in zip(args.probe_grid, grid)):
        raise SystemExit(f"--probe-grid {args.probe_grid} exceeds the run's cell grid {grid}")
    device = select_encoder_device(args.device)
    dataset = make_dataset(cfg, args.num_samples or cfg["num_val_samples"], mode=args.split)
    model = build_model(cfg, device, torch.load(checkpoint, map_location="cpu")).eval()
    reps, factors, styles = encode([model], [grid], dataset, device, args.batch_size, args.probe_grid)
    targets, style, side, centres = subject_targets(cfg, dataset, factors, styles)
    available = {t[0]: t for t in targets + style.get(args.view, [])}
    if args.target not in available:
        raise SystemExit(f"--target {args.target!r} is not scored for this run; choose from {sorted(available)}")
    name, kind, truth = available[args.target]
    X = reps[0][args.view][args.readout]
    overall = score(X, [available[name]], side)[name]
    prediction, error, badness = subject_errors(X, kind, truth, side)
    order = np.argsort(-badness, kind="stable")
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(1, len(order) + 1)
    metric = "AUC" if kind == "auc" else "R²"
    print(
        f"{args.run_dir} {args.checkpoint}: {name!r} from the {args.view} {args.readout} readout, "
        f"held-out {metric} {overall:.3f} over {len(truth)} subjects",
        flush=True,
    )

    os.makedirs(args.out, exist_ok=True)
    names = content_factor_names(factors.shape[1], cfg.get("synthetic_lesion_target", "position"))
    columns = ["index", "rank", "truth", "prediction", "error", *names]
    columns += [f"{view}_{s}" for view in styles for s in STYLE_FACTOR_NAMES[: styles[view].shape[1]]]
    if centres is not None:
        columns += ["centre_x", "centre_y", "centre_z", "side"]
    with open(os.path.join(args.out, "subjects.csv"), "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(columns)
        for i in order:
            row = [int(i), int(rank[i]), float(truth[i]), float(prediction[i]), float(error[i]), *factors[i]]
            row += [float(v) for view in styles for v in styles[view][i]]
            if centres is not None:
                row += [*map(float, centres[i]), "right" if side[i] else "left"]
            writer.writerow(row)

    affine = np.eye(4) * args.voxel_size
    affine[3, 3] = 1.0
    picks = [("worst", i) for i in order[: args.worst]] + [("best", i) for i in order[::-1][: args.best]]
    manifest = {
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "target": name,
        "readout": args.readout,
        "view": args.view,
        "probe_grid": args.probe_grid,
        "held_out_score": overall,
        "score_kind": metric,
        "voxel_size": args.voxel_size,
        "subjects": [],
    }
    print(f"\n{'':6s}{'rank':>5s}{'subject':>9s}{'truth':>10s}{'prediction':>12s}{'error':>9s}", flush=True)
    for group, i in picks:
        stem = f"{group}_rank{int(rank[i]):03d}_subject{int(i):04d}"
        files = {}
        for volume, values in subject_volumes(dataset, int(i)).items():
            path = os.path.join(args.out, f"{stem}_{volume}.nii.gz")
            nib.save(nib.Nifti1Image(values, affine), path)
            files[volume] = os.path.basename(path)
        manifest["subjects"].append(
            {
                "group": group,
                "index": int(i),
                "rank": int(rank[i]),
                "truth": float(truth[i]),
                "prediction": float(prediction[i]),
                "error": float(error[i]),
                "files": files,
            }
        )
        print(f"{group:6s}{int(rank[i]):5d}{int(i):9d}{truth[i]:10.3f}{prediction[i]:12.3f}{error[i]:9.3f}", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"\nwrote {len(picks)} subjects ({len(VOLUMES)} volumes each), subjects.csv and manifest.json to {args.out}")
    return manifest


def self_test():
    """Corrupted subjects rank worst; misclassified ones get a positive side error."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.normal(size=n)
    X = np.column_stack([y + 0.05 * rng.normal(size=n), rng.normal(size=n)])
    X[:3, 0] += 5.0  # three subjects whose feature lies about their target
    prediction, error, badness = subject_errors(X, "r2", y)
    assert set(np.argsort(-badness)[:3]) == {0, 1, 2}, np.argsort(-badness)[:5]
    np.testing.assert_allclose(error, prediction - y)

    side = (rng.normal(size=n) > 0).astype(int)
    position = rng.normal(size=n) + 10 * side
    # The feature knows the position within a hemisphere but not the 10-voxel jump between them; fitted
    # inside each hemisphere, the jump never needs explaining.
    Xw = np.column_stack([position - 10 * side + 0.05 * rng.normal(size=n), rng.normal(size=n)])
    _, error_w, badness_w = subject_errors(Xw, "within", position, side)
    assert np.median(badness_w) < 0.2, np.median(badness_w)

    label = (rng.normal(size=n) > 0).astype(int)
    feature = np.column_stack([2.0 * label - 1 + 0.1 * rng.normal(size=n), rng.normal(size=n)])
    feature[:4, 0] *= -1  # four subjects that look like the other class
    _, error_s, badness_s = subject_errors(feature, "auc", label)
    assert (error_s[:4] > 0).all() and (error_s[4:] < 0).mean() > 0.95
    assert set(np.argsort(-badness_s)[:4]) == {0, 1, 2, 3}
    print("self-test passed")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
