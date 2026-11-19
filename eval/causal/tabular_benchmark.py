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

from eval.causal.effect_estimation import (
    LEARNERS,
    estimate_ate,
    estimate_cate,
    make_folds,
    proximal_ate,
    shared_directions,
    unadjusted_ate,
)
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
    reference, own, own_source = None, None, None
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
                with np.load(source, allow_pickle=False) as data:
                    bundle_own = data["z_own"].astype(np.float64) if "z_own" in data.files else None
                if bundle_own is not None:
                    if bundle_own.shape != z.shape or not np.isfinite(bundle_own).all():
                        raise ValueError(f"{source} needs finite own noise matching its image factors")
                    if own is None:
                        own, own_source = bundle_own, source
                    elif not np.array_equal(bundle_own, own):
                        raise ValueError(f"{source} and {own_source} disagree on the image factors' own noise")
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
    if own is not None:
        own = own[:, keep]
    return z, names, adjacency, arms, reference["identity"], own


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
        tabular_causes=scm.causes,
        tabular_cause_names=scm.cause_names,
        tabular_cause_targets=np.asarray(scm.cause_targets, dtype=int),
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
        if "components" in scores[0]:
            result["components"] = scores[0]["components"]
        floor = key + " (untrained)"
        if floor in runs[0]["effects"]:
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
            for group, label in (("out_of_image", "image_out"), ("into_image", "image_in")):
                if group in scored[0]["edge_groups"]:
                    result[f"{label}_f1"] = float(np.mean([graph["edge_groups"][group]["f1"] for graph in scored]))
                    counts = [graph["orientation"][group] for graph in scored]
                    true = sum(count["true"] for count in counts)
                    result[f"{label}_oriented"] = sum(count["oriented"] for count in counts) / true if true else 0.0
        summary[key] = result
    return summary


def counterfactual_modifiers(names, own, kinds):
    """What the counterfactual tasks' effect varies with: (label, values or None for the hidden image score)."""
    modifiers = []
    if "score" in kinds:
        modifiers.append(("image score", None))
    if "own" in kinds:
        if own is None:
            raise ValueError("--counterfactual-modifiers own needs bundles that store each factor's own noise (z_own)")
        modifiers += [(f"{name} own part", own[:, j]) for j, name in enumerate(names)]
    return modifiers


def counterfactual_scores(estimate, treatment, outcome, y0, y1, labels):
    """Per modifier, each learner's per-subject effect error (PEHE), the share of the effect's variation it recovers
    (R²), its policy regret (the effect lost by treating exactly those with a positive estimate, against treating
    those with a positive effect), and the error of the outcome under the treatment not given. The abducted
    counterfactual keeps the subject's own noise: the observed outcome minus or plus the DR-learner's effect. With
    this simulator's shared outcome noise its error is the DR-learner's effect error. The mean prediction is the
    other arm's held-out outcome model, which leaves the subject's noise in the error."""
    scores = {}
    treated = treatment == 1
    for j, label in enumerate(labels):
        effect = y1[:, j] - y0[:, j]
        best = np.maximum(effect, 0).mean()
        row = {}
        for learner in ("dr_learner", "t_learner"):
            estimated = estimate[learner][:, j]
            squared = float(np.mean((estimated - effect) ** 2))
            row[learner] = dict(
                pehe=float(np.sqrt(squared)),
                r2=float(1 - squared / effect.var()) if effect.var() > 0 else None,
                regret=float(best - np.mean(effect * (estimated > 0))),
            )
        missing = np.where(treated, y0[:, j], y1[:, j])
        abducted = outcome[:, j] - np.where(treated, 1.0, -1.0) * estimate["dr_learner"][:, j]
        predicted = np.where(treated, estimate["mu0"][:, j], estimate["mu1"][:, j])
        row["counterfactual_rmse"] = dict(
            abducted=float(np.sqrt(np.mean((abducted - missing) ** 2))),
            mean_prediction=float(np.sqrt(np.mean((predicted - missing) ** 2))),
        )
        scores[label] = row
    return scores


def counterfactual_panel(z, adjacency, own, arms, seed, modifiers, args):
    """One task seed's counterfactual tasks, one per modifier, sharing treatment and the outcome under control, and
    each arm's per-subject effect estimates scored against the simulated potential outcomes."""
    tasks = [
        simulate_tabular(
            z,
            seed=seed,
            effect=args.effect,
            heterogeneity=args.counterfactual_heterogeneity,
            confounding=args.confounding,
            overlap=args.overlap,
            image_adjacency=adjacency,
            modifier=values,
        )
        for _, values in modifiers
    ]
    a, w = tasks[0].treatment, tasks[0].covariates
    if any(not np.array_equal(task.treatment, a) or not np.array_equal(task.y0, tasks[0].y0) for task in tasks):
        raise ValueError("counterfactual tasks of one seed must share treatment and the outcome under control")
    y, y0, y1 = (np.column_stack([getattr(task, key) for task in tasks]) for key in ("outcome", "y0", "y1"))
    folds = make_folds(a, args.folds, args.split_seed)
    blocks = {"tabular": None, "oracle factors": z}
    if own is not None:
        blocks["oracle own parts"] = own
    blocks.update({key: arm["X"] for key, arm in arms.items()})
    labels = [label for label, _ in modifiers]
    panel = {}
    for key, block in blocks.items():
        x = w if block is None else np.column_stack((w, block))
        estimate = estimate_cate(
            a, y, x, folds, learner=args.learner, clip=args.clip, seed=args.split_seed, tabular_width=w.shape[1]
        )
        panel[key] = counterfactual_scores(estimate, a, y, y0, y1, labels)
    return panel


def summarize_counterfactual(runs, labels, heterogeneity):
    """Each arm's counterfactual scores averaged over the task seeds; for a trained arm, its DR-learner R² minus its
    untrained floor's, paired by seed."""
    summary = {}
    for key in runs[0]["counterfactual"]:
        rows = [run["counterfactual"][key] for run in runs]
        floor = key + " (untrained)"
        summary[key] = {}
        for label in labels:
            result = {}
            for learner in ("dr_learner", "t_learner"):
                for metric in ("pehe", "r2", "regret"):
                    values = [row[label][learner][metric] for row in rows]
                    if all(v is not None for v in values):
                        result[f"{learner}_{metric}"] = float(np.mean(values))
            for kind in ("abducted", "mean_prediction"):
                result[f"counterfactual_rmse_{kind}"] = float(
                    np.mean([row[label]["counterfactual_rmse"][kind] for row in rows])
                )
            if floor in runs[0]["counterfactual"] and "dr_learner_r2" in result:
                paired = [
                    row[label]["dr_learner"]["r2"] - run["counterfactual"][floor][label]["dr_learner"]["r2"]
                    for row, run in zip(rows, runs)
                ]
                result["dr_learner_r2_delta_vs_untrained"] = float(np.mean(paired))
            summary[key][label] = result
    return dict(heterogeneity=heterogeneity, modifiers=labels, summary=summary)


def proxy_pairs(arms):
    """Each bundle's T1 and FLAIR vectors, trained and untrained: (label, name suffix, T1, FLAIR)."""
    pairs = []
    for label in dict.fromkeys(arm["label"] for arm in arms.values()):
        for suffix in ("", " (untrained)"):
            t1, flair = (arms.get(f"{label} / {VIEW_NAMES[v]}{suffix}") for v in ("1", "2"))
            if t1 is not None and flair is not None:
                pairs.append((label, suffix, t1["X"], flair["X"]))
    return pairs


def benchmark(z, names, adjacency, arms, identity, own, args):
    discovery = not args.effects_only
    own_parts = own
    modifiers = counterfactual_modifiers(names, own_parts, args.counterfactual_modifiers) if args.counterfactual else []
    own = None if args.no_tabular_causes else own
    readout_r2, decoded, held = {}, {}, []
    if discovery:
        validate_discovery_options(args.alpha, "fisherz", None)
        fit, held = train_test_split(np.arange(len(z)), test_size=args.graph_fraction, random_state=args.split_seed)
        if len(held) < z.shape[1] + 7 + (0 if own is None else 2):
            raise ValueError("Too few held-out rows for discovery at this graph width. Increase --graph-fraction.")
        decoded["oracle factors"] = z[held]
        for key, arm in arms.items():
            decoded[key] = decode_factors(arm["X"], z, fit, held)
            readout_r2[key] = dict(zip(names, r2_score(z[held], decoded[key], multioutput="raw_values").tolist()))

    # The proxies' shared directions depend on the vectors alone, not on the task, so they are counted once.
    proxies = [(*pair, shared_directions(pair[2], pair[3])) for pair in proxy_pairs(arms)] if args.proximal else []
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
            own_noise=own,
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
        for label, suffix, t1, flair, shared in proxies:
            effects[f"{label} / paired{suffix}"] = estimate_ate(
                a,
                y,
                np.column_stack((w, t1, flair)),
                folds,
                learner=args.learner,
                clip=args.clip,
                seed=args.split_seed,
                tabular_width=w.shape[1],
            )
            for name, z_proxy, w_proxy in (("T1→FLAIR", t1, flair), ("FLAIR→T1", flair, t1)):
                effects[f"{label} / proximal {name}{suffix}"] = proximal_ate(
                    a, y, w, z_proxy, w_proxy, interactions=args.heterogeneity != 0, components=shared
                )
        for score in effects.values():
            score.update(error=score["ate"] - scm.ate, covers_truth=score["ci_low"] <= scm.ate <= score["ci_high"])
        counterfactual = (
            counterfactual_panel(z, adjacency, own_parts, arms, seed, modifiers, args) if modifiers else None
        )

        graphs = {}
        if discovery:
            continuous = np.column_stack((w, scm.causes, scm.discovery_treatment, scm.discovery_outcome))[held]
            first_cause = z.shape[1] + w.shape[1]
            for key, factors in decoded.items():
                try:
                    graphs[key] = score_discovery(
                        np.column_stack((factors, continuous)),
                        scm.adjacency,
                        alpha=args.alpha,
                        n_image_factors=z.shape[1],
                        tabular_causes=range(first_cause, first_cause + scm.causes.shape[1]),
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
                discovery_truth=scm.adjacency.astype(int).tolist() if discovery else None,
                tabular_causes=[
                    dict(name=name, factor=names[target], strength=strength)
                    for name, target, strength in zip(scm.cause_names, scm.cause_targets, scm.cause_strengths)
                ],
                **({} if counterfactual is None else {"counterfactual": counterfactual}),
            )
        )
    return dict(
        settings={
            key: value for key, value in vars(args).items() if key not in ("bundles", "floors", "out", "tables_dir")
        },
        identity=identity,
        n_subjects=len(z),
        image_factors=names,
        discovery_nodes=[*names, *scm.tabular_names, *scm.cause_names, "treatment", "outcome"] if discovery else [],
        discovery_held_out_rows=list(map(int, held)),
        arms={key: {k: v for k, v in arm.items() if k != "X"} for key, arm in arms.items()},
        readout_r2=readout_r2,
        runs=runs,
        summary=summarize(runs, arms),
        counterfactual=(
            summarize_counterfactual(runs, [label for label, _ in modifiers], args.counterfactual_heterogeneity)
            if modifiers
            else None
        ),
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
    parser.add_argument(
        "--proximal",
        action="store_true",
        help="Also adjust for both views at once, and estimate the effect by proximal two-stage least squares with one"
        " view's vector instrumenting the other's (needs --view each)",
    )
    parser.add_argument(
        "--counterfactual",
        action="store_true",
        help="Also estimate each subject's effect and the outcome under the treatment not given, on tasks whose effect"
        " varies (--counterfactual-heterogeneity, --counterfactual-modifiers), scored against the simulated truth",
    )
    parser.add_argument(
        "--counterfactual-heterogeneity",
        type=float,
        default=1.0,
        help="The counterfactual tasks' effect is --effect plus this times the standardized modifier",
    )
    parser.add_argument(
        "--counterfactual-modifiers",
        nargs="+",
        choices=("score", "own"),
        default=["score"],
        help="What the effect varies with, one task each: the hidden image score, and/or each image factor's own part"
        " (needs bundles with z_own)",
    )
    parser.add_argument(
        "--no-tabular-causes",
        action="store_true",
        help="Leave out the tabular causes of image factors that bundles with own noise (z_own) otherwise add",
    )
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
    if args.proximal and args.view != "each":
        parser.error("--proximal needs --view each, which loads both views")
    if args.counterfactual and args.counterfactual_heterogeneity == 0:
        parser.error("--counterfactual needs an effect that varies: a nonzero --counterfactual-heterogeneity")
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
    causes = any("image_in_f1" in row for row in report["summary"].values())
    header = f"{'Arm':42} {'ATE':>8} {'Bias':>8} {'MAE':>8} {'PC F1':>8} {'MRI–tab F1':>10}"
    print(header + (f" {'MRI→tab F1':>10} {'tab→MRI F1':>10} {'oriented':>8}" if causes else ""))

    def cell(row, key, width):
        return f"{row[key]:{width}.3f}" if key in row else f"{'-':>{width}}"

    for key, row in report["summary"].items():
        line = f"{key:42} {row['mean_ate']:8.3f} {row['bias']:8.3f} {row['mae']:8.3f} "
        line += f"{cell(row, 'skeleton_f1', 8)} {cell(row, 'image_tabular_f1', 10)}"
        if causes:
            line += (
                f" {cell(row, 'image_out_f1', 10)} {cell(row, 'image_in_f1', 10)} {cell(row, 'image_in_oriented', 8)}"
            )
        print(line)
    shared = {key: row["components"] for key, row in report["summary"].items() if "components" in row}
    if shared:
        print(
            "Proximal arms, canonical pairs the two views share beyond chance: "
            + ", ".join(f"{k} {v}" for k, v in shared.items())
        )
    if report["counterfactual"]:
        show_counterfactual(report["counterfactual"], args.effect)
    print(f"Report: {args.out}")


def show_counterfactual(counterfactual, effect):
    """The DR-learner's scores per arm: on the image-score task its R², PEHE, regret and the counterfactual outcome's
    error from the other arm's model alone; on each own-part task its R²."""
    labels = counterfactual["modifiers"]
    own = [label for label in labels if label != "image score"]
    score = "image score" in labels
    print(
        f"\nCounterfactual tasks: effect {effect:g} + {counterfactual['heterogeneity']:g} x modifier, DR-learner unless"
        " marked T (T-learner). R² is the share of the effect's variation recovered; regret is the effect lost by"
        " treating those with a positive estimate; cf mean is the counterfactual outcome's RMSE from the other arm's"
        " model alone (the abducted counterfactual's RMSE is the PEHE)."
    )
    prefix = "  own-part R²:"
    head = f"{'Arm':42}"
    if score:
        head += f" {'R² score':>8} {'T R²':>6} {'PEHE':>6} {'regret':>6} {'cf mean':>7}"
    if own:
        head += prefix + "".join(f" {label.replace(' own part', '')[:9]:>9}" for label in own)
    print(head)

    def r2(row, learner="dr_learner", width=8):
        key = f"{learner}_r2"
        return f"{row[key]:{width}.3f}" if key in row else f"{'-':>{width}}"

    for key, rows in counterfactual["summary"].items():
        line = f"{key:42}"
        if score:
            row = rows["image score"]
            line += (
                f" {r2(row)} {r2(row, 't_learner', 6)} {row['dr_learner_pehe']:6.3f} {row['dr_learner_regret']:6.3f}"
            )
            line += f" {row['counterfactual_rmse_mean_prediction']:7.3f}"
        if own:
            line += " " * len(prefix) + "".join(f" {r2(rows[label]):>9}" for label in own)
        print(line)


if __name__ == "__main__":
    main()
