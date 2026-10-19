#!/usr/bin/env python
"""Which anatomical factors survive a spatial bottleneck? Per-factor probes against baselines.

    python -m eval.protocol.spatial_bottleneck_eval \\
        --bottleneck-dirs results/bottleneck_d16 [results/bottleneck_d16_s1 results/bottleneck_d32 ...] \\
        --out-dir results/bottleneck_eval

``training.spatial_bottleneck_experiment`` reports map compression and cross-view agreement,
and never reads a factor label. This script asks what those numbers cannot: which factors
can still be read from ``z``, compared with what the uncompressed map and the simple
readouts give on the same subjects.

Representations, all computed from the same frozen VQ-VAE content maps ``H``:

    map         full map, standardised by the bottleneck's template and per-channel scale (reference)
    patchDxHxW  channel means on a coarse grid (the existing local readout)
    gap         channel means over the whole map (the existing global readout)
    pca{d}      top-d principal components of the map, fitted on training subjects (linear compression, same size)
    z_init:RUN  the bottleneck readout at its initialisation: same seed, same normalisation (floor)
    z:RUN       the trained, validation-selected bottleneck code (proposed)

Protocol
--------
* **The bottleneck's own subjects.** Splits are regenerated from the source run's saved
  generator settings with the trainer's own split builder. Before scoring, the regenerated
  test maps must reproduce the saved test codes and GAP vectors, and the source checkpoint
  and settings must match their recorded SHA-256. A mismatch stops the run.
* **More held-out subjects.** ``--val-samples`` / ``--test-samples`` extend those splits
  with further subjects from the same split seeds (the first ones are unchanged, which the
  parity check confirms). The readout is frozen, so this only costs rendering, and 64 test
  subjects leave per-factor intervals wide.
* **Fitting uses training subjects only.** Feature standardisation, PCA and probe weights
  are fitted on training subjects. Each factor's hyperparameters (ridge penalty; RBF width)
  are chosen by validation R². Test labels are used once, for scoring.
* **Two probes.** Linear (ridge) and RBF kernel ridge. Block identifiability allows an
  invertible nonlinear change of coordinates, so a low linear score alone does not show that
  the information is gone. Both probes go through the Gram matrix, so the full map
  (hundreds of thousands of features) is probed exactly without forming a design matrix.
* **Per view.** Content factors are shared, so either view's code should predict them. Each
  view is probed separately; tables show the mean over views.
* **Uncertainty.** Paired bootstrap over test subjects with the probe fits held fixed. Every
  representation is resampled with the same draws, so ``z - gap`` and the other differences
  compare the same subjects. Training-sample and seed variability are not included; pass
  several bottleneck seeds for that.
* **Style targets** are the gain, bias and noise sigma the renderer applied
  (``eval.diagnostics.style_path_audit.effective_style``); the raw noise draw's sign never
  reaches the image. For a content code these measure leakage, so lower is better.

What this does not show
-----------------------
Under ``--synthetic-causal`` the factors are correlated, so a factor can be predicted through
another that co-varies with it (for example a lesion coordinate through anatomy). An
intervention test, which moves one factor and holds the rest, is the check for that; it is
not part of this script. Validation R² is the maximum over the hyperparameter grid, so it is
optimistic; use it only to compare settings (such as latent sizes), never as a result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
PROBES = ("linear", "rbf")
#: Ridge penalties as multiples of the mean squared (centred) feature norm, so one grid
#: serves a 16-d code and a million-feature map alike.
LINEAR_PENALTIES = tuple(float(p) for p in np.logspace(-6, 3, 19))
#: RBF widths as multiples of 1 / median squared distance between training subjects.
RBF_WIDTHS = (0.1, 0.3, 1.0, 3.0, 10.0)
RBF_PENALTIES = tuple(float(p) for p in np.logspace(-5, 1, 13))
#: Relative error above which a regenerated array is not the one the bottleneck saw. A
#: different subject gives O(1); CPU vs CUDA re-encoding stays well below 1e-3.
PARITY_TOLERANCE = 1e-2
#: Baselines each trained code is compared with (``pca`` and ``z_init`` resolve per run).
BASELINES = ("map", "patch", "gap", "pca", "z_init")


# --------------------------------------------------------------------------- #
# Probes (numpy only)
# --------------------------------------------------------------------------- #


def r2_columns(y, pred):
    """Per-column R², sklearn's ``r2_score`` convention; NaN where a column is constant."""
    resid = np.square(y - pred).sum(0)
    total = np.square(y - y.mean(0)).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, 1.0 - resid / total, np.nan)


def center_kernel(K, rows):
    """Linear kernel of the same features re-centred on the mean of ``rows``."""
    a = K[:, rows].mean(1)
    return K - a[:, None] - a[None, :] + K[np.ix_(rows, rows)].mean()


def standardized_kernel(X, train):
    """Linear kernel of features z-scored with training statistics; constant columns dropped."""
    X = np.asarray(X, dtype=np.float64)
    mean, std = X[train].mean(0), X[train].std(0)
    keep = std > 1e-10 * (1.0 + np.abs(mean))
    Z = (X[:, keep] - mean[keep]) / std[keep]
    return Z @ Z.T


def pca_scores(gram, fit_rows, n_components):
    """Principal-component scores of every row, components fitted on ``fit_rows`` only.

    This is PCA of the explicit features computed from their Gram matrix, so the full map
    never has to be held as a design matrix.
    """
    centred = center_kernel(gram, fit_rows)
    w, U = np.linalg.eigh(centred[np.ix_(fit_rows, fit_rows)])
    order = np.argsort(w)[::-1][:n_components]
    w, U = w[order], U[:, order]
    if len(w) < n_components or w[-1] <= 1e-10 * max(w[0], 1e-300):
        raise ValueError(f"Training maps span fewer than {n_components} components; PCA size is too large")
    return centred[:, fit_rows] @ U / np.sqrt(w)


def probe_kernels(K, train, kind):
    """Candidate ``(setting, kernel, penalties)`` triples for one probe family."""
    if kind == "linear":
        centred = center_kernel(K, train)
        size = np.trace(centred[np.ix_(train, train)]) / len(train)
        if size > 0:
            yield {"kernel": "linear"}, centred, [p * size for p in LINEAR_PENALTIES]
    elif kind == "rbf":
        sq = np.diag(K)
        dist = np.clip(sq[:, None] + sq[None, :] - 2.0 * K, 0.0, None)
        within = dist[np.ix_(train, train)][~np.eye(len(train), dtype=bool)]
        median = float(np.median(within)) if within.size else 0.0
        if median > 0:
            for width in RBF_WIDTHS:
                yield {"kernel": "rbf", "width": width}, np.exp(-width / median * dist), RBF_PENALTIES
    else:
        raise ValueError(f"Unknown probe {kind!r}")


def fit_select(candidates, Y, train, val, test):
    """Kernel ridge fitted on ``train``; each target's setting chosen by validation R².

    Predicting the training mean is always a candidate, so a target that no setting predicts
    on validation subjects scores about 0 on test instead of an overfitted negative. Test
    rows only receive the chosen setting's predictions.
    """
    mean = Y[train].mean(0)
    best_val = np.nan_to_num(r2_columns(Y[val], np.broadcast_to(mean, Y[val].shape)), nan=-np.inf)
    test_pred = np.tile(mean, (len(test), 1))
    choice = [{"kernel": "mean"} for _ in range(Y.shape[1])]
    centred_y = Y[train] - mean
    for setting, K, penalties in candidates:
        w, U = np.linalg.eigh(K[np.ix_(train, train)])
        w = np.clip(w, 0.0, None)
        coords = U.T @ centred_y
        to_val, to_test = K[np.ix_(val, train)] @ U, K[np.ix_(test, train)] @ U
        for penalty in penalties:
            coef = coords / (w + penalty)[:, None]
            score = np.nan_to_num(r2_columns(Y[val], to_val @ coef + mean), nan=-np.inf)
            better = score > best_val
            if better.any():
                best_val[better] = score[better]
                test_pred[:, better] = (to_test @ coef + mean)[:, better]
                for j in np.flatnonzero(better):
                    choice[j] = dict(setting, penalty=float(penalty))
    return dict(val_r2=np.where(np.isfinite(best_val), best_val, np.nan), test_pred=test_pred, choice=choice)


def bootstrap_r2(y, pred, draws):
    """R² for each resample (row of ``draws``) and target: shape ``(B, T)``."""
    yb, pb = y[draws], pred[draws]
    resid = np.square(yb - pb).sum(1)
    total = np.square(yb - yb.mean(1, keepdims=True)).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, 1.0 - resid / total, np.nan)


def score_representations(kernels, targets, rows, probes, draws):
    """Probe every representation; ``kernels[name]`` holds one (n, n) kernel per view.

    ``targets[view]`` is ``(n, T)`` and ``rows`` the train/val/test row indices. Returns
    ``{name: {probe: record}}``; ``boot`` is the ``(B, T)`` view-averaged bootstrap R²
    that the paired comparisons subtract.
    """
    train, val, test = rows
    scored = {}
    for name, per_view in kernels.items():
        scored[name] = {}
        for kind in probes:
            fits = [
                fit_select(probe_kernels(K, train, kind), targets[view], train, val, test)
                for view, K in enumerate(per_view)
            ]
            point = np.stack([r2_columns(targets[view][test], fit["test_pred"]) for view, fit in enumerate(fits)])
            boot = np.mean(
                [bootstrap_r2(targets[view][test], fit["test_pred"], draws) for view, fit in enumerate(fits)], axis=0
            )
            scored[name][kind] = dict(
                test_r2=point.mean(0),
                test_r2_by_view=point,
                val_r2=np.mean([fit["val_r2"] for fit in fits], axis=0),
                choice=[fit["choice"] for fit in fits],
                boot=boot,
            )
    return scored


def interval(boot):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(boot, [2.5, 97.5], axis=0)


def content_mean(values, content_columns):
    """Mean over the content factors along the last axis (NaN-aware)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.asarray(values)[..., content_columns], axis=-1)


def paired_difference(a, b, content_columns):
    """``a - b`` on the same test subjects: point, 95% interval and share of draws <= 0."""
    delta_boot = a["boot"] - b["boot"]
    mean_boot = content_mean(a["boot"], content_columns) - content_mean(b["boot"], content_columns)
    return dict(
        delta=a["test_r2"] - b["test_r2"],
        ci=interval(delta_boot),
        p_le_0=np.mean(delta_boot <= 0, axis=0),
        content_mean=dict(
            delta=float(content_mean(a["test_r2"], content_columns) - content_mean(b["test_r2"], content_columns)),
            ci=[float(v) for v in interval(mean_boot)],
            p_le_0=float(np.mean(mean_boot <= 0)),
        ),
    )


# --------------------------------------------------------------------------- #
# Extraction (torch)
# --------------------------------------------------------------------------- #


class LabelTap:
    """Passes batches to the label-free map extractor, keeping their labels here.

    ``extract_content_maps`` never sees ``gt_latents``. Labels are recorded in loader order,
    which is the row order of the maps it returns; ``index`` is checked to be sequential.
    """

    def __init__(self, loader, style_scale):
        self.loader = loader
        self.dataset = loader.dataset
        self.style_scale = style_scale
        self.content, self.style, self.seen = [], ([], []), 0

    def __iter__(self):
        from eval.diagnostics.style_path_audit import effective_style

        for batch in self.loader:
            index = batch["index"].tolist()
            if index != list(range(self.seen, self.seen + len(index))):
                raise ValueError("Loader must visit subjects in order without shuffling")
            self.seen += len(index)
            latents = batch.pop("gt_latents")
            self.content.append(latents["z_content"].double().numpy())
            for view in range(2):
                draws = latents.get(f"z_style_v{view + 1}")
                if self.style_scale is not None and draws is not None and draws.shape[1]:
                    width = min(draws.shape[1], 3)
                    self.style[view].append(np.stack([effective_style(z, self.style_scale)[:width] for z in draws]))
            yield batch

    def labels(self):
        content = np.concatenate(self.content)
        style = tuple(np.concatenate(parts) if parts else np.zeros((len(content), 0)) for parts in self.style)
        return content, style


def extract_split(source, dataset, batch_size, device, level, max_bytes, indices):
    from torch.utils.data import DataLoader

    from training import spatial_bottleneck_experiment as experiment

    renderer = getattr(getattr(dataset, "_inner", None), "renderer", None)
    tap = LabelTap(
        DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        getattr(renderer, "style_scale", None),
    )
    bank, indices = experiment.extract_content_maps(source, tap, device, level, max_bytes, indices)
    return bank, indices, tap.labels()


def encode_views(head, bank, batch_size, device):
    """``(n, 2, d)`` codes for paired maps ``(n, 2, C, D, H, W)``."""
    import torch

    codes = []
    with torch.no_grad():
        for start in range(0, len(bank), batch_size):
            maps = bank[start : start + batch_size].to(device)
            codes.append(head.encode(maps.flatten(0, 1)).reshape(len(maps), 2, -1).cpu().double())
    return torch.cat(codes).numpy()


def initial_readout(saved, seed):
    """The readout ``fit_readout`` starts from: same seed, config and training normalisation."""
    import torch

    from models.spatial_bottleneck import SpatialBottleneck

    torch.manual_seed(seed)
    model = SpatialBottleneck(**saved["model_config"])
    with torch.no_grad():
        model.template.copy_(saved["state_dict"]["template"])
        model.scale.copy_(saved["state_dict"]["scale"])
    return model.eval()


def map_gram(rows, template, scale, device="cpu", max_elements=2**24):
    """Gram matrix of standardised, flattened maps for ``rows`` (a list of ``(n_i, C, D, H, W)``).

    float64, accumulated one channel and voxel block at a time, so no standardised copy of
    the map bank is ever held. MPS has no float64, so only CUDA leaves the CPU.
    """
    import torch

    device = device if str(device).startswith("cuda") else "cpu"
    n = sum(len(r) for r in rows)
    voxels = math.prod(template.shape[1:])
    step = max(1, max_elements // n)
    gram = torch.zeros(n, n, dtype=torch.float64, device=device)
    for channel in range(template.shape[0]):
        centre = template[channel].reshape(-1)
        flat = [r[:, channel].reshape(len(r), -1) for r in rows]
        for start in range(0, voxels, step):
            stop = start + step
            block = torch.cat([(f[:, start:stop] - centre[start:stop]) / scale[channel].reshape(()) for f in flat])
            block = block.to(device, torch.float64)
            gram += block @ block.T
    return gram.cpu().numpy()


def relative_error(actual, expected):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        return math.inf
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12))


# --------------------------------------------------------------------------- #
# Runs and provenance
# --------------------------------------------------------------------------- #


def load_runs(directories):
    runs, names = [], set()
    for directory in directories:
        directory = Path(directory)
        for required in ("report.json", "bottleneck.pt", "test_outputs.pt"):
            if not (directory / required).is_file():
                raise FileNotFoundError(f"{directory / required} is missing; is this a bottleneck output directory?")
        name = directory.name
        while name in names:
            name += "+"
        names.add(name)
        runs.append(dict(name=name, dir=directory, report=json.loads((directory / "report.json").read_text())))
    return runs


def shared_source(runs):
    """The source, level and splits all runs must share for one evaluation to cover them."""

    def key(report):
        source, arguments = report["source"], report["arguments"]
        return dict(
            checkpoint_sha256=source["checkpoint_sha256"],
            settings_sha256=source["settings_sha256"],
            level=source["level"],
            content_indices_per_view=source["content_indices_per_view"],
            counts={split: arguments[f"{split}_samples"] for split in SPLITS},
            sample_seeds={split: report["splits"][split]["sample_seed"] for split in SPLITS},
        )

    first = key(runs[0]["report"])
    for run in runs[1:]:
        if key(run["report"]) != first:
            raise ValueError(
                f"{run['dir']} was trained on a different source, level or split than {runs[0]['dir']}; "
                "evaluate them separately"
            )
    return first


def resolve_source(report, run_dir_override):
    """Source run directory and checkpoint path, optionally relocated to ``--run-dir``."""
    from training.spatial_bottleneck_experiment import file_digest

    recorded = Path(report["source"]["checkpoint"])
    run_dir = Path(run_dir_override or report["arguments"]["run_dir"])
    checkpoint = run_dir / recorded.name if run_dir_override else recorded
    for path, field in ((checkpoint, "checkpoint_sha256"), (run_dir / "settings.json", "settings_sha256")):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; pass --run-dir if the source run moved")
        if file_digest(path) != report["source"][field]:
            raise ValueError(f"{path} does not match the {field} recorded by the bottleneck run")
    return run_dir, checkpoint


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _cell(value, width=0, signed=False):
    if value is None or not np.isfinite(value):
        return f"{'n/a':>{width}}"
    return f"{value:>+{width}.3f}" if signed else f"{value:>{width}.3f}"


def _interval_cell(entry):
    lo, hi = entry["ci"]
    return f"{_cell(entry['delta'], 7, True)} [{_cell(lo, 6, True)},{_cell(hi, 6, True)}]"


def print_report(payload):
    factors = payload["factors"]
    content = [f["name"] for f in factors if f["block"] == "content"]
    style = [f["name"] for f in factors if f["block"] == "style"]
    names = list(payload["results"])
    width = max(9, *(len(n) + 1 for n in names))
    print(f"\nSource {payload['source']['checkpoint']}  level {payload['source']['level']}")
    print("Subjects: " + ", ".join(f"{s} {payload['splits'][s]['n']}" for s in SPLITS))
    if any(payload["splits"][s]["n"] != payload["splits"][s]["n_bottleneck"] for s in SPLITS):
        print(
            "  (extended beyond the bottleneck's "
            + ", ".join(f"{s} {payload['splits'][s]['n_bottleneck']}" for s in SPLITS)
            + "; parity checked on those)"
        )
    for run in payload["bottlenecks"]:
        test = run["compression"]
        print(
            f"  {run['name']}: d={run['latent_dim']}, selected step {run['best_step']}, "
            f"recon/template {_cell(test.get('reconstruction_over_template'))}, "
            f"alignment {_cell(test.get('alignment'))} vs mismatched {_cell(test.get('mismatched_alignment'))}"
        )
    for kind in payload["probes"]:
        print(f"\n== {kind} probe: test R², mean over views ==")
        print(f"{'factor':<20}" + "".join(f"{n:>{width}}" for n in names))
        for block, rows in (("content", content), ("style (leakage; lower is better)", style)):
            if not rows:
                continue
            print(f"-- {block}")
            for factor in rows:
                cells = [payload["results"][n][kind]["test_r2"][factor] for n in names]
                print(f"{factor:<20}" + "".join(_cell(v, width) for v in cells))
        means = [payload["results"][n][kind]["content_mean"]["test"] for n in names]
        print(f"{'content mean':<20}" + "".join(_cell(v, width) for v in means))
        for run_name, by_probe in payload["comparisons"].items():
            block = by_probe[kind]
            column = max(25, *(len(b) + 4 for b in block))
            print(f"\n   z:{run_name} minus each baseline ({kind}), [95% bootstrap interval over test subjects]")
            print(f"{'factor':<20}" + "".join(f"{'- ' + b:>{column}}" for b in block))
            for factor in content + style:
                cells = [_interval_cell(record["factors"][factor]) for record in block.values()]
                print(f"{factor:<20}" + "".join(f"{c:>{column}}" for c in cells))
            cells = [_interval_cell(record["content_mean"]) for record in block.values()]
            print(f"{'content mean':<20}" + "".join(f"{c:>{column}}" for c in cells))
    if payload.get("latent_dim_selection"):
        print("\nLatent size chosen by validation content-mean R² (mean over runs with that size):")
        for kind, choice in payload["latent_dim_selection"].items():
            print(
                f"  {kind}: d={choice['latent_dim']} (validation {choice['val_content_mean']:.3f}); "
                f"its test content mean {choice['test_content_mean']:.3f}"
            )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_outputs(out_dir, payload):
    """``payload`` must already be JSON-safe; missing values become empty CSV cells."""
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "factor_recovery.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    blocks = {f["name"]: f["block"] for f in payload["factors"]}
    with (out_dir / "factor_recovery.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "representation",
                "probe",
                "factor",
                "block",
                "test_r2",
                "ci_low",
                "ci_high",
                "test_r2_view1",
                "test_r2_view2",
                "val_r2_selected",
            ]
        )
        for name, by_probe in payload["results"].items():
            for kind, record in by_probe.items():
                for factor, block in blocks.items():
                    writer.writerow(
                        [name, kind, factor, block, record["test_r2"][factor], *record["test_ci"][factor]]
                        + [record["test_r2_view1"][factor], record["test_r2_view2"][factor], record["val_r2"][factor]]
                    )
    with (out_dir / "comparisons.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bottleneck", "probe", "baseline", "factor", "delta", "ci_low", "ci_high", "p_le_0"])
        for run_name, by_probe in payload["comparisons"].items():
            for kind, by_baseline in by_probe.items():
                for baseline, record in by_baseline.items():
                    for factor, entry in record["factors"].items():
                        writer.writerow(
                            [run_name, kind, baseline, factor, entry["delta"], *entry["ci"], entry["p_le_0"]]
                        )
                    entry = record["content_mean"]
                    writer.writerow(
                        [run_name, kind, baseline, "content_mean", entry["delta"], *entry["ci"], entry["p_le_0"]]
                    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_grid(text):
    if text.lower() == "none":
        return None
    grid = tuple(int(part) for part in text.replace("x", ",").split(","))
    if len(grid) != 3 or min(grid) < 1:
        raise argparse.ArgumentTypeError("--patch-grid must be three positive integers or 'none'")
    return grid


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bottleneck-dirs", type=Path, nargs="+", required=True, help="Bottleneck output dirs")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory for the results")
    parser.add_argument("--run-dir", type=Path, default=None, help="Source VQ run, if it moved since training")
    parser.add_argument("--probes", nargs="+", choices=PROBES, default=list(PROBES))
    parser.add_argument("--patch-grid", type=parse_grid, default=(2, 2, 2), help="D,H,W or 'none'")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples of the test subjects")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed")
    parser.add_argument("--val-samples", type=int, default=None, help="Extend validation (>= the run's count)")
    parser.add_argument("--test-samples", type=int, default=None, help="Extend test (>= the run's count)")
    parser.add_argument("--max-map-gib", type=float, default=None, help="Map bank cap (default: the run's)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=0, help="0 keeps torch's default")
    args = parser.parse_args(argv)
    if args.bootstrap < 100:
        parser.error("--bootstrap needs at least 100 resamples")
    if args.out_dir.exists():
        parser.error("--out-dir must be a new directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch

    from eval.metrics.dci import STYLE_FACTOR_NAMES, content_factor_names, dataset_lesion_target
    from training import spatial_bottleneck_experiment as experiment
    from utils.encoder_runtime import select_encoder_device

    started = time.time()
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    device = select_encoder_device(args.device)
    runs = load_runs(args.bottleneck_dirs)
    shared = shared_source(runs)
    first = runs[0]["report"]
    run_dir, checkpoint = resolve_source(first, args.run_dir)
    source, saved_args, _ = experiment.load_source(
        Namespace(run_dir=run_dir, checkpoint=str(checkpoint), seed=first["arguments"]["seed"]), device
    )

    trained = shared["counts"]
    wanted = dict(trained, val=args.val_samples or trained["val"], test=args.test_samples or trained["test"])
    for split in ("val", "test"):
        if wanted[split] < trained[split]:
            raise ValueError(f"--{split}-samples must be at least the bottleneck's {trained[split]}")
    batch_size = first["arguments"]["batch_size"]
    budget = int((args.max_map_gib or first["arguments"]["max_map_gib"]) * 2**30)
    banks, labels, indices, datasets = {}, {}, shared["content_indices_per_view"], {}
    for split in SPLITS:
        dataset = experiment.build_split_dataset(saved_args, split, wanted[split])
        if int(dataset._inner.seed) != shared["sample_seeds"][split]:
            raise ValueError(f"Regenerated {split} split has seed {dataset._inner.seed}, not the recorded one")
        banks[split], indices, labels[split] = extract_split(
            source, dataset, batch_size, device, shared["level"], budget, indices
        )
        budget -= banks[split].numel() * banks[split].element_size()
        datasets[split] = dataset
        print(f"Extracted {split}: {tuple(banks[split].shape)}", flush=True)
    del source

    counts = [len(banks[split]) for split in SPLITS]
    edges = np.cumsum([0, *counts])
    rows = tuple(np.arange(edges[i], edges[i + 1]) for i in range(3))
    n_all = int(edges[-1])
    n_content = labels["train"][0].shape[1]
    n_style = labels["train"][1][0].shape[1]
    content_names = content_factor_names(n_content, dataset_lesion_target(datasets["test"]))
    content_names += [f"z_content_{j}" for j in range(len(content_names), n_content)]
    factors = [dict(name=n, block="content") for n in content_names]
    factors += [dict(name=n, block="style") for n in STYLE_FACTOR_NAMES[:n_style]]
    names = [f["name"] for f in factors]
    content_columns = [
        j for j, f in enumerate(factors) if f["block"] == "content" and not f["name"].startswith("unused")
    ]
    z_content = np.concatenate([labels[s][0] for s in SPLITS])
    targets = [np.concatenate([z_content, np.concatenate([labels[s][1][view] for s in SPLITS])], 1) for view in (0, 1)]

    # Representations: per view, an (n_all, n_all) kernel over train|val|test subjects.
    kernels = {}
    heads, bottlenecks = [], []
    for run in runs:
        head = experiment.load_bottleneck(run["dir"] / "bottleneck.pt", device)
        if tuple(head.map_shape) != tuple(banks["train"].shape[2:]):
            raise ValueError(f"{run['dir']}: map shape {head.map_shape} differs from the regenerated maps")
        if heads and not (
            torch.equal(head.template.cpu(), heads[0].template.cpu())
            and torch.equal(head.scale.cpu(), heads[0].scale.cpu())
        ):
            raise ValueError(f"{run['dir']}: normalisation differs from {runs[0]['dir']}, so its training maps differ")
        heads.append(head)
    template, scale = heads[0].template.cpu(), heads[0].scale.cpu()

    gram = map_gram([banks[s][:, view] for view in (0, 1) for s in SPLITS], template, scale, device)
    view_rows = [np.arange(view * n_all, (view + 1) * n_all) for view in (0, 1)]
    kernels["map"] = [gram[np.ix_(r, r)] for r in view_rows]
    if args.patch_grid is not None and all(g <= m for g, m in zip(args.patch_grid, template.shape[1:])):
        pooled = torch.cat(
            [
                torch.nn.functional.adaptive_avg_pool3d(banks[s].flatten(0, 1), args.patch_grid).reshape(
                    len(banks[s]), 2, -1
                )
                for s in SPLITS
            ]
        ).numpy()
        patch_name = "patch" + "x".join(map(str, args.patch_grid))
        kernels[patch_name] = [standardized_kernel(pooled[:, view], rows[0]) for view in (0, 1)]
    else:
        patch_name = None
    gap = torch.cat([banks[s].mean((3, 4, 5)) for s in SPLITS]).numpy()
    kernels["gap"] = [standardized_kernel(gap[:, view], rows[0]) for view in (0, 1)]
    pca_fit_rows = np.concatenate([view * n_all + rows[0] for view in (0, 1)])
    for d in sorted({head.latent_dim for head in heads}):
        scores = pca_scores(gram, pca_fit_rows, d)
        kernels[f"pca{d}"] = [standardized_kernel(scores[r], rows[0]) for r in view_rows]

    for run, head in zip(runs, heads):
        report, saved = run["report"], torch.load(run["dir"] / "bottleneck.pt", map_location="cpu", weights_only=True)
        outputs = torch.load(run["dir"] / "test_outputs.pt", map_location="cpu", weights_only=True)
        codes = {s: encode_views(head, banks[s], batch_size, device) for s in SPLITS}
        init = initial_readout(saved, report["arguments"]["seed"]).to(device)
        init_codes = {s: encode_views(init, banks[s], batch_size, device) for s in SPLITS}
        seen_val, seen_test = banks["val"][: trained["val"]], trained["test"]
        parity = dict(
            test_codes=relative_error(codes["test"][:seen_test], outputs["codes"]),
            test_gap=relative_error(gap[rows[2][:seen_test]], outputs["gap"]),
            val_reconstruction=relative_error(
                experiment.evaluate(head, seen_val, batch_size, device)[0]["reconstruction"],
                report["validation"]["reconstruction"],
            ),
            initial_val_reconstruction=relative_error(
                experiment.evaluate(init, seen_val, batch_size, device)[0]["reconstruction"],
                report["training"]["initial_validation"]["reconstruction"],
            ),
        )
        failed = {k: v for k, v in parity.items() if k != "initial_val_reconstruction" and v > PARITY_TOLERANCE}
        if failed:
            extended = wanted != trained
            raise ValueError(
                f"{run['dir']}: regenerated subjects do not reproduce the saved outputs (relative error {failed}); "
                "the generator, source checkpoint or code changed since training"
                + (
                    ". Under fixed_reference normalisation a split under 64 subjects is normalised on its own "
                    "subjects, so it cannot be extended: drop --val-samples/--test-samples"
                    if extended
                    else ""
                )
            )
        z_init_reproduced = parity["initial_val_reconstruction"] <= PARITY_TOLERANCE
        if not z_init_reproduced:
            print(
                f"WARNING {run['name']}: the rebuilt initial readout does not match the recorded step-0 validation "
                "loss (different torch version?). z_init is a random readout at the same seed, not the exact one.",
                flush=True,
            )
        for label, values in (("z", codes), ("z_init", init_codes)):
            stacked = np.concatenate([values[s] for s in SPLITS])
            kernels[f"{label}:{run['name']}"] = [standardized_kernel(stacked[:, view], rows[0]) for view in (0, 1)]
        bottlenecks.append(
            dict(
                name=run["name"],
                dir=run["dir"],
                latent_dim=head.latent_dim,
                seed=report["arguments"]["seed"],
                best_step=report["training"]["best_step"],
                compression={
                    k: report["test"].get(k)
                    for k in ("reconstruction_over_template", "alignment", "mismatched_alignment")
                },
                parity=parity,
                z_init_reproduced=z_init_reproduced,
            )
        )
        del init

    print(f"Probing {len(kernels)} representations x {len(args.probes)} probes", flush=True)
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, counts[2], size=(args.bootstrap, counts[2]))
    scored = score_representations(kernels, targets, rows, args.probes, draws)

    results = {}
    for name, by_probe in scored.items():
        results[name] = {}
        for kind, record in by_probe.items():
            ci = interval(record["boot"])
            mean_boot = content_mean(record["boot"], content_columns)
            mean_ci = interval(mean_boot)
            results[name][kind] = dict(
                test_r2=dict(zip(names, record["test_r2"])),
                test_ci={n: [ci[0, j], ci[1, j]] for j, n in enumerate(names)},
                test_r2_view1=dict(zip(names, record["test_r2_by_view"][0])),
                test_r2_view2=dict(zip(names, record["test_r2_by_view"][1])),
                val_r2=dict(zip(names, record["val_r2"])),
                content_mean=dict(
                    test=float(content_mean(record["test_r2"], content_columns)),
                    ci=list(mean_ci),
                    val=float(content_mean(record["val_r2"], content_columns)),
                ),
                settings=[dict(zip(names, view)) for view in record["choice"]],
            )

    comparisons = {}
    for run in bottlenecks:
        baselines = {
            "map": "map",
            "patch": patch_name,
            "gap": "gap",
            "pca": f"pca{run['latent_dim']}",
            "z_init": f"z_init:{run['name']}",
        }
        comparisons[run["name"]] = {}
        for kind in args.probes:
            mine = scored[f"z:{run['name']}"][kind]
            comparisons[run["name"]][kind] = {}
            for label in BASELINES:
                other = baselines[label]
                if other is None:
                    continue
                diff = paired_difference(mine, scored[other][kind], content_columns)
                comparisons[run["name"]][kind][other] = dict(
                    factors={
                        n: dict(
                            delta=diff["delta"][j], ci=[diff["ci"][0, j], diff["ci"][1, j]], p_le_0=diff["p_le_0"][j]
                        )
                        for j, n in enumerate(names)
                    },
                    content_mean=diff["content_mean"],
                )

    selection = {}
    if len({run["latent_dim"] for run in bottlenecks}) > 1:
        for kind in args.probes:
            by_dim = {}
            for run in bottlenecks:
                summary = results[f"z:{run['name']}"][kind]["content_mean"]
                by_dim.setdefault(run["latent_dim"], []).append((summary["val"], summary["test"]))
            best = max(by_dim, key=lambda d: np.mean([v for v, _ in by_dim[d]]))
            selection[kind] = dict(
                latent_dim=best,
                val_content_mean=float(np.mean([v for v, _ in by_dim[best]])),
                test_content_mean=float(np.mean([t for _, t in by_dim[best]])),
                runs=len(by_dim[best]),
            )

    payload = dict(
        source=dict(run_dir=run_dir, checkpoint=checkpoint, **{k: v for k, v in shared.items() if k != "counts"}),
        splits={
            s: dict(n=counts[i], n_bottleneck=trained[s], sample_seed=shared["sample_seeds"][s])
            for i, s in enumerate(SPLITS)
        },
        factors=factors,
        content_mean_factors=[names[j] for j in content_columns],
        probes=list(args.probes),
        bottlenecks=bottlenecks,
        results=results,
        comparisons=comparisons,
        latent_dim_selection=selection,
        protocol=dict(
            fitting="standardisation, PCA and probe weights fitted on training subjects only",
            selection="per factor, by validation R² over the probe's grid (validation R² is optimistic)",
            map_features="(H - template) / per-channel scale from the bottleneck; probed through the Gram matrix",
            pca="fitted on both views' training maps; components standardised before probing",
            views="each view probed separately; tables average the two",
            uncertainty=f"paired bootstrap over test subjects, {args.bootstrap} draws, probe fits held fixed",
            style_targets="effective gain/bias/noise sigma applied by the renderer",
            linear_penalties=list(LINEAR_PENALTIES),
            rbf_widths=list(RBF_WIDTHS),
            rbf_penalties=list(RBF_PENALTIES),
            parity_tolerance=PARITY_TOLERANCE,
            bootstrap_seed=args.seed,
        ),
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        elapsed_seconds=round(time.time() - started, 1),
    )
    payload = _json_safe(payload)
    write_outputs(args.out_dir, payload)
    print_report(payload)
    print(f"\nSaved {args.out_dir / 'factor_recovery.json'} and CSVs", flush=True)
    return payload


if __name__ == "__main__":
    main()
