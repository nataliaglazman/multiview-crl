"""Diagnose modality leakage in a frozen encoder-only ADNI checkpoint.

Measures geometry in L2-normalized content, then compares linear and RBF modality
probes before/after subtracting per-modality means fitted on CV training subjects.
See ENCODER_MODALITY_GAP.md for the protocol and Run:ai command.
"""

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

CONDITIONS = ("normalized", "mean_centered")
PROBES = ("linear", "rbf")


def normalize_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def geometry(a, b):
    """Descriptive geometry; no fitted transform from here is used by a probe."""
    ma, mb = a.mean(0), b.mean(0)
    gap = np.linalg.norm(ma - mb)
    spread = np.sqrt(0.5 * (np.square(a - ma).sum(1).mean() + np.square(b - mb).sum(1).mean()))
    pair_squared = np.square(a - b).sum(1).mean()
    valid = (np.linalg.norm(a, axis=1) > 1e-12) & (np.linalg.norm(b, axis=1) > 1e-12)
    cosine = np.clip((normalize_rows(a[valid]) * normalize_rows(b[valid])).sum(1), -1, 1)
    return {
        "centroid_distance": float(gap),
        "within_modality_rms": float(spread),
        "centroid_distance_over_within_rms": float(gap / spread) if spread > 1e-12 else None,
        "paired_squared_distance_mean": float(pair_squared),
        "mean_offset_fraction_of_pair_squared_distance": (
            float(np.clip(gap**2 / pair_squared, 0, 1)) if pair_squared > 1e-12 else None
        ),
        "paired_cosine_mean": float(cosine.mean()) if len(cosine) else None,
        "paired_cosine_std": float(cosine.std()) if len(cosine) else None,
        "paired_cosine_valid_pairs": int(valid.sum()),
        "zero_vectors": [int((np.linalg.norm(x, axis=1) <= 1e-12).sum()) for x in (a, b)],
    }


def subject_folds(subjects, n_splits, seed):
    """All scans and both views of a subject stay in the same fold."""
    subjects = np.asarray(subjects, dtype=str)
    unique = np.unique(subjects)
    if not 2 <= n_splits <= len(unique):
        raise ValueError("Need 2 <= folds <= number of unique subjects")
    for train, test in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(unique):
        yield np.flatnonzero(np.isin(subjects, unique[train])), np.flatnonzero(np.isin(subjects, unique[test]))


def fold_features(a, b, train, test, center):
    """Normalize before entry; estimate each modality mean on this fold's training rows only."""
    means = np.stack([a[train].mean(0), b[train].mean(0)]) if center else np.zeros((2, a.shape[1]))
    x_train = np.concatenate([a[train] - means[0], b[train] - means[1]])
    x_test = np.concatenate([a[test] - means[0], b[test] - means[1]])
    return x_train, x_test, means


def diagnose(a, b, subjects, n_splits=3, seed=1729):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    subjects = np.asarray(subjects, dtype=str)
    if a.ndim != 2 or b.shape != a.shape or a.shape[1] < 1:
        raise ValueError("Expected matching (pairs, content_channels) arrays with at least one channel")
    if subjects.shape != (len(a),) or np.any(subjects == ""):
        raise ValueError("Need one nonempty subject ID per paired row")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite content features")
    folds = list(subject_folds(subjects, n_splits, seed))
    a, b = normalize_rows(a), normalize_rows(b)
    n = len(a)
    labels = np.repeat([0, 1], n)
    fold_ids = np.full(n, -1, dtype=int)
    predictions = {f"{c}_{p}": np.full(2 * n, -1, dtype=int) for c in CONDITIONS for p in PROBES}
    centered = (np.empty_like(a), np.empty_like(b))
    fold_reports = []
    with threadpool_limits(limits=1):
        for fold, (train, test) in enumerate(folds):
            fold_ids[test] = fold
            y_train = np.repeat([0, 1], len(train))
            y_test = np.repeat([0, 1], len(test))
            result = {"fold": fold, "train_pairs": len(train), "test_pairs": len(test), "accuracy": {}}
            for condition in CONDITIONS:
                x_train, x_test, means = fold_features(a, b, train, test, condition == "mean_centered")
                if condition == "mean_centered":
                    result["training_modality_means"] = means.tolist()
                    centered[0][test], centered[1][test] = x_test[: len(test)], x_test[len(test) :]
                # Class-constant coordinates can leave ~1e-17 signed residuals after
                # subtraction. StandardScaler would amplify those into a perfect
                # modality label. Decide numerical constancy on training rows only.
                constant = x_train.std(axis=0) <= 1e-12
                x_train[:, constant] = 0
                x_test[:, constant] = 0
                result.setdefault("numerically_constant_channels", {})[condition] = np.flatnonzero(constant).tolist()
                for probe in PROBES:
                    estimator = (
                        LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
                        if probe == "linear"
                        else SVC(C=1.0, kernel="rbf", gamma="scale")
                    )
                    clf = make_pipeline(StandardScaler(), estimator)
                    clf.fit(x_train, y_train)
                    pred = clf.predict(x_test)
                    key = f"{condition}_{probe}"
                    predictions[key][np.r_[test, n + test]] = pred
                    result["accuracy"][key] = float(np.mean(pred == y_test))
            fold_reports.append(result)
    scores = {
        key: {
            "accuracy": float(np.mean(pred == labels)),
            "fold_accuracy": [f["accuracy"][key] for f in fold_reports],
        }
        for key, pred in predictions.items()
    }
    report = {
        "pairs": n,
        "subjects": len(np.unique(subjects)),
        "content_channels": a.shape[1],
        "chance_accuracy": 0.5,
        "seed": seed,
        "n_splits": n_splits,
        "geometry_normalized": geometry(a, b),
        "paired_cosine_after_fold_centering": geometry(*centered)["paired_cosine_mean"],
        "probes": scores,
        "folds": fold_reports,
        "protocol": {
            "input": "L2-normalized content block, before any contrastive projector",
            "grouping": "All rows and both modalities of a subject share a fold",
            "centering": "Subtract modality means fitted on CV training subjects; no subsequent L2 normalization",
            "scaling": "One pooled StandardScaler fitted on each condition's training rows only",
            "numerical_tolerance": "Channels with training-fold std <= 1e-12 are zeroed in both fold partitions",
            "probes": "Fixed C=1 logistic regression and C=1, gamma=scale RBF SVM; no hyperparameter tuning",
            "accuracy": "Pooled out-of-fold row accuracy; both modalities have equal counts",
        },
        "interpretation": [
            "Mean centering uses known modality labels: this is a diagnostic intervention, not learned invariance.",
            "A centered linear probe near chance is expected after removing class means; read the RBF probe too.",
            "A high centered RBF score detects residual modality differences beyond a constant normalized-space offset.",
            "Both probes near chance mean no leakage detected by these probes, not proof of equal distributions.",
            "Centering is in normalized space; a constant raw-space offset need not remain constant after normalization.",
        ],
    }
    rows = []
    for view, name in enumerate(("T1", "FLAIR")):
        for i, subject in enumerate(subjects):
            rows.append(
                {
                    "pair_index": i,
                    "subject": subject,
                    "view": name,
                    "modality": view,
                    "fold": int(fold_ids[i]),
                    **{key: int(pred[view * n + i]) for key, pred in predictions.items()},
                }
            )
    return report, rows


def extract_checkpoint(args):
    """Restore the saved model and validation preprocessing, checking saved subject identity."""
    import torch
    from torch.utils.data import DataLoader

    from eval.protocol.score_checkpoint import build_model, load_settings
    from training.main_conv_synthetic import make_real_dataset
    from utils.encoder_runtime import select_encoder_device

    run_dir = Path(args.run_dir).resolve()
    cfg = load_settings(run_dir)
    if cfg.get("dataset_name", "synthetic") == "synthetic":
        raise ValueError("This diagnostic requires an encoder-only real-data run")
    split = json.loads((run_dir / "split.json").read_text())
    for key in ("dataroot", "labels_path", "masks_dir", "cache_dir"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    checkpoint = (run_dir / args.checkpoint).resolve()
    checkpoint_bytes = checkpoint.read_bytes()
    device = select_encoder_device(args.device)
    state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    model = build_model(cfg, device, state)
    dataset = make_real_dataset(SimpleNamespace(**cfg), "val")
    subjects = np.asarray([item["subject"] for item in dataset.items], dtype=str)
    if subjects.tolist() != split["subjects"]["val"]:
        raise ValueError("Resolved validation subjects differ from split.json; check data paths and labels")
    list(subject_folds(subjects, args.folds, args.seed))
    chunks = ([], [])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            images = torch.cat(batch["image"], dim=0).as_subclass(torch.Tensor).to(device)
            b = len(batch["index"])
            content = model(images, pool_only=True, n_views=2)[2][0][:, : cfg["content_channels"]]
            chunks[0].append(content[:b].float().cpu().numpy())
            chunks[1].append(content[b:].float().cpu().numpy())
            if step == 1 or step == len(loader) or step % 10 == 0:
                print(f"Encoded validation batch {step}/{len(loader)}", flush=True)
    metadata = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "split": "val",
        "device": device,
        "batch_size": args.batch_size,
        "resolved_settings": cfg,
    }
    return np.concatenate(chunks[0]), np.concatenate(chunks[1]), subjects, metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="Encoder-only ADNI run with settings.json and split.json")
    source.add_argument("--features", help="Re-score this tool's saved features.npz without loading MRI or PyTorch")
    parser.add_argument("--checkpoint", default="model_best.pt", help="Filename within run-dir, or absolute path")
    parser.add_argument("--batch-size", type=int, default=4, help="Encoding subjects per view")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--out-dir", help="New report directory; default: timestamped directory beside the input")
    for flag in ("dataroot", "labels-path", "masks-dir", "cache-dir"):
        parser.add_argument(f"--{flag}", help="Override the saved cluster path")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.folds < 2:
        parser.error("Need positive batch-size and at least two folds")
    if args.features and any(
        getattr(args, k) is not None for k in ("dataroot", "labels_path", "masks_dir", "cache_dir")
    ):
        parser.error("Data path overrides only apply with --run-dir")
    return args


def main(argv=None):
    args = parse_args(argv)
    base = Path(args.run_dir) if args.run_dir else Path(args.features).parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = Path(args.out_dir) if args.out_dir else base / f"modality_gap_{stamp}"
    if out.exists():
        raise FileExistsError(f"Report directory already exists: {out}")
    if args.features:
        with np.load(args.features, allow_pickle=False) as features:
            a, b, subjects = features["content_t1"], features["content_flair"], features["subjects"]
            metadata = json.loads(str(features["metadata_json"]))
        metadata = {**metadata, "features_source": str(Path(args.features).resolve())}
    else:
        a, b, subjects, metadata = extract_checkpoint(args)
    print("Fitting modality probes with subjects held out ...", flush=True)
    report, rows = diagnose(a, b, subjects, args.folds, args.seed)
    import sklearn

    report.update(schema_version=1, created_at_utc=stamp, source=metadata, sklearn_version=sklearn.__version__)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        out / "features.npz",
        content_t1=a,
        content_flair=b,
        subjects=subjects,
        metadata_json=np.array(json.dumps(metadata, allow_nan=False)),
    )
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (out / "predictions.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    g = report["geometry_normalized"]
    print(
        f"\n{report['pairs']} validation pairs, {report['subjects']} subjects, {report['content_channels']} content channels"
    )
    print(f"Normalized centroid distance: {g['centroid_distance']:.6f}")
    print(
        f"Paired cosine: {g['paired_cosine_mean']}; after fold-fitted centering: {report['paired_cosine_after_fold_centering']}"
    )
    print("Modality accuracy (chance 0.5):              linear      RBF")
    for condition in CONDITIONS:
        scores = [report["probes"][f"{condition}_{probe}"]["accuracy"] for probe in PROBES]
        print(f"  {condition:<38s}{scores[0]:.3f}      {scores[1]:.3f}")
    print("Centering uses known modality. A centered linear score near 0.5 alone is not evidence of invariance.")
    print(f"Report: {out / 'report.json'}", flush=True)
    return report


if __name__ == "__main__":
    main()
