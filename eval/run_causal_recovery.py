#!/usr/bin/env python
"""Batch the causal panel from analyze_synthetic_recovery.ipynb.

    python -m eval.run_causal_recovery --run-dirs results/run1 results/run2
    python -m eval.run_causal_recovery --runs-file runs.txt --output-dir results/causal_recovery

Each run needs settings.json and a VQVAE checkpoint. See CAUSAL_EVALUATION.md
for metric interpretation. The default reproduces the notebook's supervised,
in-sample graph readout and truth-selected alpha sweep, not directed recovery.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
DEFAULT_ALPHAS = (0.01, 0.05, 0.1, 0.2)


def skeleton_metrics(estimated, truth):
    """Score each unordered pair once; exact_match also handles empty graphs."""
    import numpy as np

    estimated = np.asarray(estimated, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if estimated.shape != truth.shape or truth.ndim != 2 or truth.shape[0] != truth.shape[1]:
        raise ValueError("Estimated and true adjacency must be equally sized square matrices")
    estimated = estimated | estimated.T
    truth = truth | truth.T
    iu = np.triu_indices(len(truth), k=1)
    tp = int((estimated[iu] & truth[iu]).sum())
    fp = int((estimated[iu] & ~truth[iu]).sum())
    fn = int((~estimated[iu] & truth[iu]).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(
        precision=p,
        recall=r,
        f1=2 * p * r / (p + r) if p + r else 0.0,
        tp=tp,
        fp=fp,
        fn=fn,
        skeleton_shd=fp + fn,
        exact_match=(fp + fn == 0),
    )


def fit_probe(X, y):
    """Notebook linear probe: 70/30 split, train-only scaling, Ridge(alpha=1)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    scaler = StandardScaler().fit(Xtr)
    model = Ridge(alpha=1.0).fit(scaler.transform(Xtr), ytr)
    return float(r2_score(yte, model.predict(scaler.transform(Xte)), multioutput="variance_weighted"))


def evaluate_arrays(X_content, z_content, adjacency, alphas=DEFAULT_ALPHAS):
    """Evaluate already aligned features and factors with the supplied panel's protocol."""
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.preprocessing import StandardScaler

    try:
        from causallearn.search.ConstraintBased.PC import pc
    except ImportError as exc:
        raise ImportError("Graph recovery requires causal-learn: python -m pip install causal-learn") from exc

    X = np.asarray(X_content, dtype=np.float64)
    z = np.asarray(z_content, dtype=np.float64)
    adj = np.asarray(adjacency, dtype=bool)
    if X.ndim != 2 or z.ndim != 2 or X.shape[0] != z.shape[0]:
        raise ValueError("Features and ground truth must be aligned 2D arrays")
    if len(X) < 20 or X.shape[1] == 0 or z.shape[1] < 2:
        raise ValueError("Need at least 20 samples, one content feature, and two content factors")
    if not np.isfinite(X).all() or not np.isfinite(z).all():
        raise ValueError("Features and ground truth must be finite")
    n_content = z.shape[1]
    if adj.shape != (n_content, n_content):
        raise ValueError("SCM adjacency dimension does not match z_content")
    if not alphas or any(not 0 < a < 1 for a in alphas):
        raise ValueError("PC alpha values must be between 0 and 1")

    parents = [np.flatnonzero(adj[:, d]).tolist() for d in range(n_content)]
    residuals = z.copy()
    for d, pa in enumerate(parents):
        if pa:
            residuals[:, d] -= LinearRegression().fit(z[:, pa], z[:, d]).predict(z[:, pa])
    raw = [fit_probe(X, z[:, d : d + 1]) for d in range(n_content)]
    partial = [fit_probe(X, residuals[:, d : d + 1]) for d in range(n_content)]

    Xsc = StandardScaler().fit_transform(X)
    # Extra sample-count cap avoids invalid PCA for small datasets / many factors.
    n_pca = min(64, max(n_content, len(Xsc) // 5), Xsc.shape[1], len(Xsc))
    if Xsc.shape[1] > n_pca:
        Xsc = PCA(n_components=n_pca, random_state=0).fit_transform(Xsc)
    z_hat = np.column_stack(
        [RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(Xsc, z[:, d]).predict(Xsc) for d in range(n_content)]
    )
    if np.any(np.std(z_hat, axis=0) <= np.finfo(float).eps):
        raise ValueError("A decoded factor is constant; Fisher-Z graph recovery is undefined")

    truth = adj | adj.T
    np.fill_diagonal(truth, False)
    sweep = []
    best = None
    for alpha in alphas:
        cg = pc(z_hat, alpha=alpha, indep_test="fisherz", show_progress=False)
        graph = cg.G.graph
        estimated = (graph != 0) | (graph.T != 0)
        np.fill_diagonal(estimated, False)
        row = dict(alpha=alpha, **skeleton_metrics(estimated, truth), adjacency=estimated.astype(int).tolist())
        sweep.append(row)
        # Preserve the notebook's tie-breaking: last alpha wins equal F1.
        if best is None or row["f1"] >= best["f1"]:
            best = row

    return dict(
        num_samples=len(X),
        num_features=X.shape[1],
        n_content=n_content,
        graph_readout_dim=Xsc.shape[1],
        raw_r2_mean=float(np.mean(raw)),
        partial_r2_mean=float(np.mean(partial)),
        factors=[
            dict(dim=d, parents=parents[d], raw_r2=raw[d], partial_r2=partial[d], gap=raw[d] - partial[d])
            for d in range(n_content)
        ],
        true_dag=adj.astype(int).tolist(),
        true_skeleton=truth.astype(int).tolist(),
        best=best,
        alpha_sweep=sweep,
    )


def extract_content(model, dataset, device, level, pooling, batch_size, num_workers):
    """Capture raw view-1 encoder maps, as in notebook section 3, at one level."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    inner = model.module if hasattr(model, "module") else model
    features, targets = [], []
    captured = []
    separate = getattr(inner, "separate_encoders", False) and inner.encoders_v1 is not None

    def hook(module, inputs, output):
        captured.append(output.detach())

    handle = inner.encoders[level].register_forward_hook(hook)
    model.eval()
    try:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        with torch.no_grad():
            # Notebook section 4 freezes the channel split using sample 0.
            # In on-the-fly mask mode it can otherwise vary with each batch.
            sample_images = dataset[0]["image"]
            out = model(
                torch.cat([image[None] for image in sample_images], dim=0).to(device),
                return_recon=False,
                pool_only=True,
                n_views=len(sample_images),
                subsets=[(0, 1)],
                patch_grid=None,
            )
            if len(captured) != 1:
                raise ValueError(f"Expected one encoder output at level {level}, got {len(captured)}")
            masks = out[6]
            if level in masks:
                mask = masks[level]
                mask = mask[0] if isinstance(mask, tuple) else mask
                indices = torch.where(mask.detach().flatten() > 0.5)[0]
            else:
                indices = torch.arange(captured[0].shape[1], device=captured[0].device)
            if len(indices) == 0:
                raise ValueError(f"Level {level} has no content channels")
            for batch in loader:
                captured.clear()
                images = batch["image"]
                model(
                    torch.cat(images, dim=0).to(device),
                    return_recon=False,
                    pool_only=True,
                    n_views=len(images),
                    subsets=[(0, 1)],
                    patch_grid=None,
                )
                if len(captured) != 1:
                    raise ValueError(f"Expected one encoder output at level {level}, got {len(captured)}")
                maps = captured[0] if separate else captured[0][: len(images[0])]
                maps = maps[:, indices]
                pooled = (
                    maps.mean(dim=(2, 3, 4)) if pooling == "gap" else F.adaptive_avg_pool3d(maps, pooling).flatten(1)
                )
                features.append(pooled.cpu().numpy())
                targets.append(batch["gt_latents"]["z_content"].numpy())
    finally:
        handle.remove()
    return np.concatenate(features), np.concatenate(targets)


def evaluate_run(run_dir, cli):
    """Load each run independently so SCM, renderer and normalization match it."""
    with (run_dir / "settings.json").open() as f:
        settings = json.load(f)
    if not settings.get("synthetic_causal", False):
        return dict(status="skipped", reason="settings['synthetic_causal'] is False")

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    checkpoint = run_dir / cli.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model, args, device = load_model_from_run_dir(str(run_dir), str(checkpoint), device=cli.device, seed=0)
    levels = settings.get("content_style_levels") or [0]
    level = cli.level if cli.level is not None else levels[0]
    if not 0 <= level < model.nb_levels:
        raise ValueError(f"Invalid encoder level {level}; model has {model.nb_levels} levels")
    dataset = build_synthetic_test_set(args, cli.num_samples, cache=False, causal=True)
    # Use the instantiated SCM: pseudo-MRI may determine its own factor count.
    scm = getattr(getattr(dataset, "_inner", dataset), "scm", None)
    if scm is None:
        raise ValueError("Matched synthetic dataset did not expose a causal SCM")
    X, z = extract_content(model, dataset, device, level, cli.pooling, cli.batch_size, cli.num_workers)
    result = evaluate_arrays(X, z, scm["adj"], cli.alphas)
    result.update(
        status="ok",
        checkpoint=str(checkpoint),
        level=level,
        pooling=cli.pooling,
        causal_settings={key: value for key, value in settings.items() if key.startswith("synthetic_")},
    )
    return result


def collect_runs(patterns, runs_file=None):
    """Expand shell-style globs and deduplicate while preserving input order.

    Paths in a runs file are relative to that file; command-line paths are
    relative to the working directory. Unmatched inputs become error rows.
    """
    inputs = [(p, Path.cwd()) for p in patterns]
    if runs_file:
        file = Path(runs_file).expanduser().resolve()
        inputs.extend(
            (line.strip(), file.parent)
            for line in file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    runs = []
    for pattern, base in inputs:
        path = Path(pattern).expanduser()
        pattern = str(path if path.is_absolute() else base / path)
        for match in sorted(glob.glob(pattern)) or [pattern]:
            path = Path(match).resolve()
            if path not in runs:
                runs.append(path)
    return runs


def write_reports(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(
        protocol=dict(
            graph_target="undirected_skeleton",
            alpha_selection="best_f1_against_truth_last_tie",
            graph_readout="supervised_in_sample_ridgecv",
            parent_adjustment="linear_all_samples",
            probe="ridge_alpha1_70_30_split_seed0",
            content_mask="sample0_fixed",
            empty_graph_f1=0.0,
        ),
        runs=results,
    )
    (output_dir / "causal_recovery.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    columns = [
        "run_dir",
        "status",
        "reason",
        "level",
        "pooling",
        "num_samples",
        "num_features",
        "raw_r2_mean",
        "partial_r2_mean",
        "alpha",
        "f1",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
        "skeleton_shd",
        "exact_match",
    ]
    with (output_dir / "causal_recovery.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow({**result, **result.get("best", {})})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs", nargs="+", default=[], help="Run directories or quoted glob patterns")
    parser.add_argument("--runs-file", help="Text file with one directory/glob per line, relative to that file")
    parser.add_argument("--output-dir", type=Path, default=Path("results/causal_recovery"))
    parser.add_argument("--checkpoint", default="vqvae_model.pt", help="Checkpoint filename inside each run")
    parser.add_argument("--level", type=int, help="Default: first content_style_levels entry, otherwise 0")
    parser.add_argument("--pooling", default="4,4,4", help="gap or a 3D patch grid (default: 4,4,4, as in notebook)")
    parser.add_argument("--num-samples", type=int, help="Default: each run's synthetic_num_test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", help="cpu, cuda, cuda:0, etc.; default: CUDA when available, else CPU")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    cli = parser.parse_args(argv)
    if cli.pooling != "gap":
        try:
            cli.pooling = tuple(int(x) for x in cli.pooling.split(","))
            if len(cli.pooling) != 3 or min(cli.pooling) < 1:
                raise ValueError
        except ValueError:
            parser.error("--pooling must be gap or three positive integers, e.g. 4,4,4")
    if any(not 0 < a < 1 for a in cli.alphas):
        parser.error("--alphas must be strictly between 0 and 1")
    if cli.batch_size < 1 or cli.num_workers < 0 or (cli.num_samples is not None and cli.num_samples < 20):
        parser.error("Require batch-size >= 1, num-workers >= 0, num-samples >= 20")
    if Path(cli.checkpoint).name != cli.checkpoint:
        parser.error("--checkpoint must be a filename inside each run directory")
    runs = collect_runs(cli.run_dirs, cli.runs_file)
    if not runs:
        parser.error("Supply --run-dirs and/or a non-empty --runs-file")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    results = []
    for i, run in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] {run}", flush=True)
        try:
            result = evaluate_run(run, cli)
        except Exception as exc:
            logger.exception("Failed to evaluate %s", run)
            result = dict(status="error", reason=f"{type(exc).__name__}: {exc}")
        result["run_dir"] = str(run)
        results.append(result)
        if result["status"] == "ok":
            best = result["best"]
            print(
                f"  L{result['level']} partial R²={result['partial_r2_mean']:.3f} " f"(raw={result['raw_r2_mean']:.3f})"
            )
            print(
                f"  Best skeleton F1={best['f1']:.3f} P={best['precision']:.3f} R={best['recall']:.3f} "
                f"alpha={best['alpha']:g} SHD={best['skeleton_shd']} exact_match={best['exact_match']}"
            )
            for factor in result["factors"]:
                print(
                    f"    d{factor['dim']} pa={factor['parents']}: raw={factor['raw_r2']:.3f} "
                    f"partial={factor['partial_r2']:.3f} gap={factor['gap']:+.3f}"
                )
        else:
            print(f"  {result['status']}: {result['reason']}")
        # Persist after every run so a later failure doesn't discard completed work.
        write_reports(results, cli.output_dir)
    print(f"\nSaved CSV and JSON to {cli.output_dir.resolve()}")
    print("F1 measures the skeleton only; alpha is selected against truth and the graph readout is in-sample.")
    return int(any(result["status"] == "error" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
