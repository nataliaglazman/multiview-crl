"""Compare frozen image bundles on matched synthetic discovery and treatment tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from eval.causal.effect_estimation import LEARNERS, estimate_ate, make_folds, unadjusted_ate
from eval.causal.tabular_discovery import score_discovery, validate_discovery_options
from eval.causal.tabular_scm import simulate_tabular
from eval.dino.dinov3_identifiability import load_bundle, usable_factors
from eval.metrics.bundle_identity import require_same_rows
from eval.protocol.compare_bundles import parse_named

DEFAULT_SEEDS = tuple(range(10))
VIEW_NAMES = {"1": "T1", "2": "FLAIR", "both": "paired"}


def load_arms(paths, floors, view, discovery):
    if not paths:
        raise ValueError("Supply at least one bundle")
    if floors.keys() - paths.keys():
        raise ValueError("Each untrained floor must name a trained bundle")
    views = ("1", "2") if view == "each" else (view,)
    arms, records = {}, {}
    reference = None
    for label, path in paths.items():
        for v in views:
            trained_width = None
            for kind, source in (("trained", path), ("untrained", floors.get(label))):
                if source is None:
                    continue
                bundle = load_bundle(source, view=v)
                key = f"{label} / {VIEW_NAMES[v]}" + (" (untrained)" if kind == "untrained" else "")
                if key in arms:
                    raise ValueError(f"Duplicate arm name: {key}")
                records[key] = bundle["identity"]
                if reference is None:
                    reference = bundle
                z, x = bundle["z_content"], bundle["X"]
                if not np.array_equal(z, reference["z_content"]) or not np.array_equal(
                    bundle["adjacency"], reference["adjacency"]
                ):
                    raise ValueError(f"Bundles are not row-aligned: {source} has different factors or graph")
                if bundle["content_names"] != reference["content_names"]:
                    raise ValueError(f"{source} has different content factor names or order")
                if x.ndim != 2 or len(x) != len(z) or not x.shape[1] or not np.isfinite(x).all():
                    raise ValueError(f"{source} needs one finite, nonempty feature vector per subject")
                if kind == "trained":
                    trained_width = x.shape[1]
                elif x.shape[1] != trained_width:
                    raise ValueError(f"{source} and {path} have different feature widths")
                arms[key] = {
                    "X": x,
                    "kind": kind,
                    "label": label,
                    "view": v,
                    "path": str(source),
                    "width": x.shape[1],
                    "meta": bundle["meta"],
                }
    require_same_rows(records)
    z, adjacency, names = reference["z_content"], reference["adjacency"], reference["content_names"]
    if not np.isfinite(z).all() or len(names) != z.shape[1] or len(set(names)) != len(names):
        raise ValueError("Bundle needs finite image factors with unique names")
    if adjacency is not None and adjacency.shape != (z.shape[1], z.shape[1]):
        raise ValueError("Bundle image graph must match the image factor columns")
    if discovery and adjacency is None:
        raise ValueError("Discovery needs a known image graph in the bundle. Use --effects-only for an effect task.")
    z, names, keep = usable_factors(z, names)
    if not names:
        raise ValueError("Bundle needs at least one varying image factor")
    if adjacency is not None:
        adjacency = adjacency[np.ix_(keep, keep)]
    return z, names, adjacency, arms, reference["identity"]


def decode_factors(x, z, fit, held):
    """Fit one shared ridge readout, with one penalty chosen across all image factors."""
    x_scale, z_scale = StandardScaler().fit(x[fit]), StandardScaler().fit(z[fit])
    model = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
    model.fit(x_scale.transform(x[fit]), z_scale.transform(z[fit]))
    return z_scale.inverse_transform(model.predict(x_scale.transform(x[held])))


def write_tables(root, seed, scm, z, names, identity):
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"observed_{seed}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", *scm.tabular_names, "treatment", "outcome"])
        writer.writerows(
            (i, *w, int(a), y) for i, (w, a, y) in enumerate(zip(scm.covariates, scm.treatment, scm.outcome))
        )
    np.savez_compressed(
        root / f"truth_{seed}.npz",
        row=np.arange(len(z)),
        z_content=z,
        content_factor_names=names,
        covariate_names=scm.tabular_names,
        propensity=scm.propensity,
        y0=scm.y0,
        y1=scm.y1,
        mu0=scm.mu0,
        mu1=scm.mu1,
        effect_adjacency=scm.effect_adjacency,
        discovery_adjacency=scm.adjacency,
        discovery_treatment=scm.discovery_treatment,
        discovery_outcome=scm.discovery_outcome,
        identity=json.dumps(identity),
    )


def summarize(runs, arms):
    summary = {}
    tabular_error = [abs(run["effects"]["tabular"]["error"]) for run in runs]
    for key in runs[0]["effects"]:
        scores = [run["effects"][key] for run in runs]
        error = np.array([score["error"] for score in scores])
        result = dict(
            mean_ate=float(np.mean([score["ate"] for score in scores])),
            bias=float(error.mean()),
            mae=float(np.abs(error).mean()),
            rmse=float(np.sqrt(np.square(error).mean())),
            coverage=float(np.mean([score["covers_truth"] for score in scores])),
            mae_delta_vs_tabular=float(np.mean(np.abs(error) - tabular_error)),
        )
        floor = key + " (untrained)"
        if floor in arms:
            floor_error = [abs(run["effects"][floor]["error"]) for run in runs]
            result["mae_delta_vs_untrained"] = float(np.mean(np.abs(error) - floor_error))
        graphs = [run["discovery"].get(key, {}) for run in runs]
        scored = [graph for graph in graphs if "f1" in graph]
        if any(graphs):
            result["graphs_scored"] = len(scored)
        if scored:
            result["skeleton_f1"] = float(np.mean([graph["f1"] for graph in scored]))
            result["image_tabular_f1"] = float(
                np.mean([graph["edge_groups"]["image_tabular"]["f1"] for graph in scored])
            )
        summary[key] = result
    return summary


def benchmark(z, names, adjacency, arms, identity, args):
    discovery = not args.effects_only
    readout_r2, decoded, held = {}, {}, []
    if discovery:
        validate_discovery_options(args.alpha, "fisherz", None)
        fit, held = train_test_split(np.arange(len(z)), test_size=args.graph_fraction, random_state=args.split_seed)
        if len(held) < z.shape[1] + 7:
            raise ValueError("Too few held-out rows for discovery at this graph width. Increase --graph-fraction.")
        decoded["oracle factors"] = z[held]
        for key, arm in arms.items():
            decoded[key] = decode_factors(arm["X"], z, fit, held)
            readout_r2[key] = dict(zip(names, r2_score(z[held], decoded[key], multioutput="raw_values").tolist()))

    runs = []
    for seed in args.seeds:
        scm = simulate_tabular(
            z,
            seed=seed,
            effect=args.effect,
            heterogeneity=args.heterogeneity,
            confounding=args.confounding,
            overlap=args.overlap,
            image_adjacency=adjacency,
        )
        w, a, y = scm.covariates, scm.treatment, scm.outcome
        folds = make_folds(a, args.folds, args.split_seed)
        effects = {"unadjusted": unadjusted_ate(a, y)}
        blocks = {"tabular": None, "oracle factors": z, **{key: arm["X"] for key, arm in arms.items()}}
        for key, block in blocks.items():
            x = w if block is None else np.column_stack((w, block))
            effects[key] = estimate_ate(
                a, y, x, folds, learner=args.learner, clip=args.clip, seed=args.split_seed, tabular_width=w.shape[1]
            )
        for score in effects.values():
            score.update(error=score["ate"] - scm.ate, covers_truth=score["ci_low"] <= scm.ate <= score["ci_high"])

        graphs = {}
        if discovery:
            continuous = np.column_stack((w, scm.discovery_treatment, scm.discovery_outcome))[held]
            for key, factors in decoded.items():
                try:
                    graphs[key] = score_discovery(
                        np.column_stack((factors, continuous)),
                        scm.adjacency,
                        alpha=args.alpha,
                        n_image_factors=z.shape[1],
                    )
                except np.linalg.LinAlgError as error:
                    graphs[key] = {"error": str(error)}

        if args.tables_dir is not None:
            write_tables(args.tables_dir, seed, scm, z, names, identity)
        runs.append(
            dict(
                seed=seed,
                ate_truth=scm.ate,
                treated=int(a.sum()),
                propensity_range=[float(scm.propensity.min()), float(scm.propensity.max())],
                effects=effects,
                discovery=graphs,
            )
        )
    return dict(
        settings={
            key: value for key, value in vars(args).items() if key not in ("bundles", "floors", "out", "tables_dir")
        },
        identity=identity,
        n_subjects=len(z),
        image_factors=names,
        discovery_nodes=[*names, *scm.tabular_names, "treatment", "outcome"] if discovery else [],
        discovery_truth=scm.adjacency.astype(int).tolist() if discovery else None,
        discovery_held_out_rows=list(map(int, held)),
        arms={key: {k: v for k, v in arm.items() if k != "X"} for key, arm in arms.items()},
        readout_r2=readout_r2,
        runs=runs,
        summary=summarize(runs, arms),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--floors", nargs="+", metavar="NAME=PATH", help="Matched untrained bundle for each named arm")
    parser.add_argument(
        "--view", choices=("1", "2", "both", "each"), default="1", help="each scores both views separately"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Synthetic task seeds (default: 0 to 9)"
    )
    parser.add_argument("--effect", type=float, default=1.0)
    parser.add_argument("--heterogeneity", type=float, default=0.0)
    parser.add_argument("--confounding", type=float, default=1.0)
    parser.add_argument(
        "--overlap", type=float, default=0.1, help="Lower bound on true treatment and control probabilities"
    )
    parser.add_argument("--learner", choices=LEARNERS, default="linear")
    parser.add_argument(
        "--clip", type=float, default=0.025, help="Lower bound on fitted treatment and control probabilities"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--effects-only", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05, help="Fixed significance threshold for the graph test")
    parser.add_argument(
        "--graph-fraction", type=float, default=0.5, help="Fraction held out from the discovery readout fit"
    )
    parser.add_argument(
        "--tables-dir", type=Path, help="Write each observed table and separate synthetic truth archive"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be distinct nonnegative integers")
    try:
        paths = parse_named(args.bundles, "--bundles")
        floors = parse_named(args.floors, "--floors")
        inputs = load_arms(paths, floors, args.view, not args.effects_only)
        report = benchmark(*inputs, args)
    except (argparse.ArgumentTypeError, ValueError, KeyError, ImportError, OSError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"{report['n_subjects']} subjects, {len(report['runs'])} tasks. Average treatment effect target: {args.effect:g}"
    )
    print(f"{'Arm':36} {'ATE':>8} {'Bias':>8} {'MAE':>8} {'PC F1':>8} {'MRI–tab F1':>10}")
    for key, row in report["summary"].items():
        graph = f"{row['skeleton_f1']:.3f}" if "skeleton_f1" in row else "-"
        cross = f"{row['image_tabular_f1']:.3f}" if "image_tabular_f1" in row else "-"
        print(f"{key:36} {row['mean_ate']:8.3f} {row['bias']:8.3f} {row['mae']:8.3f} {graph:>8} {cross:>10}")
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
