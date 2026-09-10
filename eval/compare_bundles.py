#!/usr/bin/env python
"""Score several feature bundles through one protocol and put them in one table.

    python -m eval.compare_bundles \\
        --bundles vq_all=results/bundles/vq_all_gap.npz \\
                  vq_content=results/bundles/vq_content_gap.npz \\
                  dino=results/3dino/pretrained.npz \\
        --floors  vq_all=results/bundles/vq_all_gap_floor.npz \\
                  dino=results/3dino/random_init.npz \\
        --equal-width --with-graph --out results/matched/compare.json

Every bundle is scored by ``eval.dinov3_identifiability.score`` with ONE options object,
so the CV splits, the permutation nulls, the PCA rule, the probe and the floor handling
are not merely specified the same -- they are the same objects doing the same work.  The
splits depend on row count alone and the nulls are redrawn from a fixed seed per bundle,
so bundle *k*'s fold *i* holds the same rows as bundle *j*'s fold *i* and both are
differenced against the same permutation.

Three things this checks that a pair of separate runs cannot:

1. **Row identity.**  Bundles are compared only when their factor digests agree
   (``eval/bundle_identity.py``).  Equal ``--num-samples`` is not evidence of that: two
   draws of one generator at different seeds give 2000 rows each and share none.
2. **Feature width.**  ``auto`` PCA picks a width per block, so two bundles can arrive at
   the table with different probe capacity and part of the difference between them is that.
   The effective width is printed per bundle, and ``--equal-width`` pins every bundle to
   the narrowest block's width so the comparison is at matched capacity.
3. **Floor pairing.**  A floor belongs to its own architecture.  Each bundle's Δfloor is
   computed against the floor given for THAT bundle; one architecture's floor is never
   subtracted from another's, which would not be a baseline correction at all.

Causal discovery (``--with-graph``)
-----------------------------------
Each representation additionally gets the PC panel from
``eval.run_causal_recovery.evaluate_arrays``: a supervised readout decodes the factors from
that representation, PC recovers a skeleton from the decoded columns, and the skeleton is
scored **against the true SCM adjacency** the generator used.  One row per source, plus a
``truth (ceiling)`` row running the identical panel on the ground-truth factors themselves
and a row per untrained floor.

Two alpha selections are reported and they answer different questions.  The **prespecified**
``--diagnostic-alpha`` row is the one a head-to-head can be read off, because every source
is tested at the same threshold.  The **best-F1** row is each source at its own most
flattering alpha, chosen by looking at the answer; the alpha column is part of that result,
not a footnote to it.  With ``--orientation`` (on by default) edge directions are also
scored against the true DAG's CPDAG -- the CPDAG, because PC identifies an equivalence
class and the default ``chain`` SCM's is entirely undirected.

``--equal-width`` matters more here than for the probes.  The readout has its own width
rule that caps at each block's feature count, so a 48-channel VQ block and an 18k-dim
embedding otherwise get readouts of 48 and 64 and part of the graph difference is that gap.
The ceiling is exempt: its features ARE the factors, so it is ``n_content`` columns wide by
construction.

This scores how well the SCM survives a representation, not causal discovery from raw
features, and both the in-sample readout and the truth-selected alpha make it optimistic.
See ``eval/CAUSAL_EVALUATION.md``.

Read ``gap`` (real minus permutation null) across bundles, and ``Δfloor`` only where both
bundles have their own floor.  The columns are defined in
``eval/COMPARING_3DINO_VQVAE.md``; ``--self-test`` runs the whole path -- probes and PC --
on planted numpy arrays with no bundle files.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from eval import dinov3_identifiability as scorer
from eval.bundle_identity import compare as compare_identity
from eval.bundle_identity import read_identity
from eval.run_causal_recovery import INDEP_TESTS
from eval.run_dci_compare import PROBE_DIM_AUTO

logger = logging.getLogger(__name__)


def parse_named(values, flag):
    """``label=path`` pairs -> an ordered dict, with the duplicate-label case rejected."""
    out = {}
    for item in values or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"{flag} takes label=path, got {item!r}")
        label, path = item.split("=", 1)
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError(f"{flag} entry {item!r} has an empty label")
        if label in out:
            raise argparse.ArgumentTypeError(f"{flag} label {label!r} appears twice")
        out[label] = Path(path)
    return out


def check_alignment(bundles, strict=True):
    """Verify every bundle describes the same evaluation rows. Returns the problems found."""
    # A bundle loaded from disk carries the identity taken from the file's own arrays.
    # One assembled in memory (the self-test, callers holding arrays) has no file to read,
    # so fall back to digesting the content factors it does have -- a weaker check, since
    # it cannot see a style-factor or adjacency difference, but not a silent one.
    records = {
        label: bundle.get("identity") or read_identity({}, {"z_content": bundle["z_content"]}, len(bundle["X"]))
        for label, bundle in bundles.items()
    }
    problems = compare_identity(records)
    if problems and strict:
        raise SystemExit(
            "Refusing to compare bundles that are not row-aligned:\n  - "
            + "\n  - ".join(problems)
            + "\n\nRebuild them from one --run-dir with one --num-samples, or pass "
            "--allow-row-mismatch to score them anyway (the table will not be a comparison)."
        )
    for line in problems:
        logger.warning("%s", line)
    return records, problems


def common_width(bundles, floors):
    """The narrowest feature count across everything that will be probed.

    A floor is included because it is probed too: pinning the models to a width its own
    floor cannot reach would compare a reduced model against an unreduced baseline.
    """
    widths = [b["X"].shape[1] for b in bundles.values()]
    widths += [f["X"].shape[1] for f in floors.values() if f is not None]
    return int(min(widths))


def common_readout(bundles, floors, options):
    """The graph readout width every bundle can actually reach.

    ``--equal-width`` alone does not make the graph panel comparable: ``--probe-dim``
    reduces the block tables 1/2 probe, while PC's supervised readout has its own width
    rule (``run_causal_recovery.readout_width``) that caps at the block's own feature
    count.  A 48-channel VQ block and an 18,432-dim embedding therefore get readouts of 48
    and 64, and part of any difference in the recovered graph is that gap rather than the
    representation.  Pinning both to what the narrowest block can reach removes it.
    """
    from eval.run_causal_recovery import readout_width

    rows = min(len(b["X"]) for b in bundles.values())
    n_content = min(b["z_content"].shape[1] for b in bundles.values())
    return int(readout_width(rows, common_width(bundles, floors), n_content, options.readout_dim))


def truth_panel(bundles, options):
    """PC on the ground-truth factors themselves — the finite-sample reference row.

    Computed once here rather than once per bundle. It depends only on the factors and the
    adjacency, which the row-identity check has already established are shared, so scoring
    it per bundle would repeat the same PC search and invite the reader to compare rows
    that are the same computation. It is a reference, not an upper bound: a representation
    can beat it by decoding factors into columns PC happens to find easier to separate.
    """
    reference = next(iter(bundles.values()))
    if reference["adjacency"] is None:
        return None
    Z, names, keep = scorer.usable_factors(reference["z_content"], reference["content_names"])
    panel = scorer.graph_panel(Z, Z, reference["adjacency"][np.ix_(keep, keep)], options)
    panel["factor_names"] = names
    return panel


def score_all(bundles, floors, options):
    """``{label: report}``, each produced by the identical scoring call."""
    results = {}
    for label, bundle in bundles.items():
        floor = floors.get(label)
        if floor is not None and floor["X"].shape[0] != bundle["X"].shape[0]:
            raise SystemExit(f"floor for {label!r} has {len(floor['X'])} rows, bundle has {len(bundle['X'])}")
        logger.info("Scoring %s (%d x %d)", label, *bundle["X"].shape)
        results[label] = scorer.score(bundle, floor, options)
    return results


def factor_rows(results, block):
    """``[(factor, {label: row})]`` over the factors every bundle scored, in bundle order."""
    per_label = {label: result.get(block, {}) for label, result in results.items()}
    names = [n for n in next(iter(per_label.values()), {}) if n != "_block"]
    shared = [n for n in names if all(n in rows for rows in per_label.values())]
    for name in names:
        if name not in shared:
            logger.warning("Factor %r is missing from some bundles and is left out of the table", name)
    return [(name, {label: rows[name] for label, rows in per_label.items()}) for name in shared]


def format_block(results, block, column):
    """One side-by-side table: factors down, bundles across, ``column`` in the cells."""
    rows = factor_rows(results, block)
    if not rows:
        return ""
    labels = list(results)
    if not any(column in cells[label] for _n, cells in rows for label in labels):
        return ""
    headers = ["factor", *labels]
    if len(labels) == 2:
        headers.append(f"{labels[1]}−{labels[0]}")
    body = []
    for name, cells in rows:
        line = [name] + [scorer._fmt(cells[label].get(column)) for label in labels]
        if len(labels) == 2:
            a, b = cells[labels[0]].get(column), cells[labels[1]].get(column)
            line.append(scorer._fmt(None if a is None or b is None else b - a))
        body.append(line)
    means = []
    for label in labels:
        vals = [cells[label].get(column) for _n, cells in rows]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        means.append(scorer._fmt(float(np.mean(vals)) if vals else None))
    footer = ["mean", *means]
    if len(labels) == 2:
        footer.append("")
    title = {
        "gap": "real − permutation null",
        "delta_floor": "gap − own untrained floor",
        "delta_voxels": "gap − downsampled-voxel baseline",
    }[column]
    return f"{block.upper()} · {title}\n{scorer._table(headers, [*body, footer])}\n"


def format_widths(results):
    lines = []
    for label, result in results.items():
        block = result.get("content", {}).get("_block", {})
        lines.append(
            f"  {label:<14} {result['num_features']:>7} features"
            f" -> {block.get('probe_features', '?'):>5} probed"
            f"   floor: {'yes' if result.get('floor_path') else 'no':<3}"
            f"   voxels: {'yes' if result.get('has_voxels') else 'no'}"
        )
    widths = {r.get("content", {}).get("_block", {}).get("probe_features") for r in results.values()}
    note = ""
    if len({w for w in widths if w is not None}) > 1:
        note = (
            "\n  NOTE: the bundles were probed at different widths, so part of any difference\n"
            "        below is probe capacity rather than representation. Pass --equal-width.\n"
        )
    return "FEATURE WIDTH\n" + "\n".join(lines) + "\n" + note


#: Label of the reference row: PC on the ground-truth factors rather than on a
#: representation. It is filtered out of the readout-width check, because its "features"
#: ARE the factors, so it is n_content columns wide by construction and can never match a
#: model's readout width.
CEILING_LABEL = "truth (ceiling)"


def graph_panels(results, truth):
    """``[(label, panel)]`` in reading order: the ceiling, then each bundle and its floor."""
    rows = []
    if truth:
        rows.append((CEILING_LABEL, truth))
    for label, result in results.items():
        panels = result.get("graph") or {}
        if panels.get("embeddings"):
            rows.append((label, panels["embeddings"]))
        if panels.get("floor"):
            rows.append((f"{label} · floor", panels["floor"]))
    return rows


def _sweep_row(panel, alpha):
    """The panel's scored row at exactly ``alpha``, or None if PC failed there."""
    for row in panel.get("alpha_sweep", []):
        if abs(row.get("alpha", float("nan")) - alpha) < 1e-12 and "f1" in row:
            return row
    return None


def format_graph_scores(rows, selector, diagnostic_alpha):
    """Skeleton recovery against the TRUE adjacency, one line per source.

    ``selector`` is either the prespecified ``--diagnostic-alpha`` row or each panel's own
    best-F1 alpha.  Both are reported because they answer different questions: the fixed
    alpha is the one a head-to-head can be read off, since every source is scored at the
    same test threshold; the best-alpha row is each source at its own most flattering
    setting, chosen by looking at the answer, and the alpha column is part of the result.
    """
    headers = ["source", "alpha", "F1", "prec", "rec", "SHD", "TP", "FP", "FN", "exact"]
    body = []
    for label, panel in rows:
        row = selector(panel)
        if not row:
            body.append([label, *["—"] * 9])
            continue
        body.append(
            [
                label,
                f"{row['alpha']:g}",
                f"{row['f1']:.3f}",
                f"{row['precision']:.3f}",
                f"{row['recall']:.3f}",
                str(row["skeleton_shd"]),
                str(row["tp"]),
                str(row["fp"]),
                str(row["fn"]),
                "yes" if row["exact_match"] else "no",
            ]
        )
    return scorer._table(headers, body)


def format_graph_factors(rows):
    """Partial R² per factor, across sources — where a graph difference comes from.

    Partial R² is the factor's own variation recovered after its SCM parents are regressed
    out, so it is the quantity a recovered edge actually rests on: a source that reads a
    factor only through its parents scores high raw and near zero here, and PC then sees a
    column that is mostly the parent.
    """
    names = [factor["name"] for _label, panel in rows for factor in panel.get("factors", [])]
    ordered = list(dict.fromkeys(names))
    if not ordered:
        return ""
    labels = [label for label, _p in rows]
    body = []
    for name in ordered:
        found = [next((f for f in panel.get("factors", []) if f["name"] == name), None) for _label, panel in rows]
        parents = next((str(f["parents"]) for f in found if f), "")
        body.append([name, parents, *(scorer._fmt(f["partial_r2"] if f else None) for f in found)])
    return scorer._table(["factor", "parents", *labels], body)


def format_graph_orientation(rows, diagnostic_alpha):
    """Edge DIRECTIONS against the true DAG's CPDAG, at the prespecified alpha."""
    body = []
    for label, panel in rows:
        row = _sweep_row(panel, diagnostic_alpha)
        orientation = (row or {}).get("orientation")
        if not orientation:
            continue
        body.append(
            [
                label,
                str(orientation["correct_directed"]),
                str(orientation["reversed"]),
                str(orientation["undirected_in_estimate"]),
                str(orientation["directed_in_estimate"]),
                str(orientation["bidirected_in_estimate"]),
                str(orientation["both_undirected"]),
                str(orientation["cpdag_shd"]),
                "yes" if orientation["cpdag_exact_match"] else "no",
            ]
        )
    if not body:
        return ""
    headers = ["source", "correct", "rev", "undir_est", "dir_est", "bidir", "both_undir", "CPDAG SHD", "exact"]
    return scorer._table(headers, body)


def graph_sources(bundles, floors, options):
    """``(sources, Z, adjacency)`` — every feature block the causal table scores.

    Same labels and order as :func:`graph_panels`, and the same per-source reduction the
    headline panel uses, so a stability band describes the rows above it rather than a
    slightly different pipeline.
    """
    reference = next(iter(bundles.values()))
    if reference["adjacency"] is None:
        return [], None, None
    Z, _names, keep = scorer.usable_factors(reference["z_content"], reference["content_names"])
    adjacency = reference["adjacency"][np.ix_(keep, keep)]
    sources = [(CEILING_LABEL, Z)]
    for label, bundle in bundles.items():
        sources.append((label, scorer.graph_features(bundle["X"], options)))
        if floors.get(label) is not None:
            sources.append((f"{label} · floor", scorer.graph_features(floors[label]["X"], options)))
    return sources, Z, adjacency


def graph_stability(sources, Z, adjacency, options, repeats, fraction):
    """Re-run PC on repeated row subsamples, to say whether a graph difference is real.

    The panel is deterministic given its input, so the headline table has no error bar and
    a two-edge difference between two sources reads exactly like a twenty-edge one.  This
    resamples the thing that actually varies — which subjects are in the evaluation set —
    and reports the spread.

    Subsampling WITHOUT replacement rather than a bootstrap: duplicated rows would inflate
    the dependence Fisher-Z is testing for, so a bootstrap would bias the skeleton towards
    extra edges.  Every source sees the SAME rows at every repeat, which makes the
    per-repeat differences paired — that is what lets the paired column have a much tighter
    spread than the two marginal columns it is built from.

    The feature reduction is fitted once, outside the loop, so the band reflects the graph
    step rather than PCA being refitted; it is the same reduction the headline row used.
    """
    if not sources or repeats < 2:
        return {}
    n = len(Z)
    size = max(int(round(n * fraction)), 20)
    if size >= n:
        raise ValueError("--graph-subsample must be below 1.0 or every repeat is the same rows")
    rng = np.random.RandomState(options.null_seed)
    per_source = {label: [] for label, _X in sources}
    for repeat in range(repeats):
        rows = rng.choice(n, size=size, replace=False)
        logger.info("Graph stability repeat %d/%d on %d rows", repeat + 1, repeats, size)
        for label, X in sources:
            panel = scorer.graph_panel(X[rows], Z[rows], adjacency, options)
            per_source[label].append(_sweep_row(panel, options.diagnostic_alpha))
    return summarise_stability(per_source, sources, size, repeats)


def summarise_stability(per_source, sources, size, repeats):
    """``{label: {...}}`` with each metric's mean/std and the paired delta vs the first model."""
    metrics = ("f1", "precision", "recall", "skeleton_shd")
    reference = next((label for label, _X in sources if label != CEILING_LABEL), None)
    out = {}
    for label, rows in per_source.items():
        scored = [row for row in rows if row]
        entry = {"repeats": len(scored), "subsample": size, "requested": repeats}
        for metric in metrics:
            values = [row[metric] for row in scored]
            entry[f"{metric}_mean"] = float(np.mean(values)) if values else float("nan")
            entry[f"{metric}_std"] = float(np.std(values)) if values else float("nan")
        if reference and label != reference:
            paired = [row["f1"] - other["f1"] for row, other in zip(rows, per_source[reference]) if row and other]
            entry["f1_delta_mean"] = float(np.mean(paired)) if paired else float("nan")
            entry["f1_delta_std"] = float(np.std(paired)) if paired else float("nan")
            entry["f1_delta_vs"] = reference
        out[label] = entry
    return out


def format_graph_stability(stability):
    """The band table: is the difference in the rows above bigger than the resampling noise?"""
    if not stability:
        return ""
    headers = ["source", "F1 mean", "±", "SHD mean", "±", "ΔF1 vs ref", "±", "resolved"]
    body = []
    for label, entry in stability.items():
        delta, spread = entry.get("f1_delta_mean"), entry.get("f1_delta_std")
        # "resolved" only when the paired difference clears twice its own spread. Two SDs of
        # a paired difference over this many repeats is a rough screen, not a test -- it has
        # no multiplicity correction and the repeats share rows.
        resolved = "—"
        if delta is not None and np.isfinite(delta) and np.isfinite(spread):
            resolved = "yes" if abs(delta) > 2 * spread else "no"
        body.append(
            [
                label,
                f"{entry['f1_mean']:.3f}",
                f"{entry['f1_std']:.3f}",
                f"{entry['skeleton_shd_mean']:.1f}",
                f"{entry['skeleton_shd_std']:.1f}",
                scorer._fmt(delta) if delta is not None else "ref",
                f"{spread:.3f}" if spread is not None and np.isfinite(spread) else "",
                resolved,
            ]
        )
    first = next(iter(stability.values()))
    caption = (
        f"  {first['repeats']}/{first['requested']} repeats on {first['subsample']} rows each, "
        "the same rows for every source"
    )
    return f"{scorer._table(headers, body)}\n{caption}"


def format_graph(results, truth, options, stability=None):
    """The whole causal-discovery section: PC on each representation, scored against truth."""
    rows = graph_panels(results, truth)
    if not rows:
        return ""
    alpha = options.diagnostic_alpha
    out = [
        "CAUSAL DISCOVERY · PC skeleton vs the TRUE SCM adjacency",
        "",
        f"  at the prespecified alpha={alpha:g} — the row to read a head-to-head off",
        format_graph_scores(rows, lambda panel: _sweep_row(panel, alpha), alpha),
    ]
    scored = [(label, panel) for label, panel in rows if label != CEILING_LABEL]
    features = {panel.get("num_features") for _l, panel in scored}
    widths = {panel.get("graph_readout_dim") for _l, panel in scored}
    samples = {panel.get("graph_samples") for _l, panel in rows}
    modes = {panel.get("readout_mode") for _l, panel in rows}

    def _join(values):
        return "/".join(str(v) for v in sorted(values, key=lambda v: (v is None, v)))

    out.append(
        f"  features: {_join(features)} in, read out {'/'.join(sorted(str(m) for m in modes))} at "
        f"{_join(widths)} dims, PC on {_join(samples)} rows, "
        f"{next(iter(rows))[1].get('indep_test')} independence test"
        + (f"; the ceiling uses its {truth.get('num_features')} factors directly" if truth else "")
    )
    if len(widths) > 1 or len(features) > 1:
        which = " and ".join(
            part for part, differs in (("feature", len(features) > 1), ("readout", len(widths) > 1)) if differs
        )
        out.append(
            f"  NOTE: the sources differ in {which} width, so part of the difference above is\n"
            "        capacity rather than representation. Pass --equal-width."
        )
    out += [
        "",
        "  at each source's own best-F1 alpha, selected against the truth (optimistic)",
        format_graph_scores(rows, lambda panel: panel.get("best"), alpha),
    ]
    factors = format_graph_factors(rows)
    if factors:
        out += ["", "  partial R² per factor — the recovered signal each edge rests on", factors]
    band = format_graph_stability(stability)
    if band:
        out += [
            "",
            "  stability under row resampling — is the difference above bigger than the noise?",
            band,
        ]
    orientation = format_graph_orientation(rows, alpha)
    if orientation:
        out += [
            "",
            f"  orientation vs the true CPDAG at alpha={alpha:g}"
            " (PC identifies an equivalence class, so undirected can be correct)",
            orientation,
        ]
    out.append(
        "\n  PC runs on a supervised readout of each representation's decoded factors, so this\n"
        "  scores how well the SCM survives that representation, not causal discovery from raw\n"
        "  features. The default readout is in-sample (--holdout-readout makes it held-out at\n"
        "  70% of the rows) and the best-alpha row selects against the answer, so both are\n"
        "  optimistic diagnostics. See eval/CAUSAL_EVALUATION.md.\n"
    )
    return "\n".join(out)


def format_report(results, problems, truth=None, options=None, stability=None):
    parts = [
        "=" * 92,
        "MATCHED BUNDLE COMPARISON",
        "=" * 92,
        "",
        format_widths(results),
    ]
    if problems:
        parts += ["ROW ALIGNMENT", *(f"  !! {p}" for p in problems), ""]
    for block in ("content", "style"):
        # delta_voxels last: it is the weakest claim of the three (a feature that does not
        # beat average-pooled voxels has not earned its forward pass) but it is the only one
        # that needs no second model, so it is the column a single bundle still gets.
        for column in ("gap", "delta_floor", "delta_voxels"):
            table = format_block(results, block, column)
            if table:
                parts.append(table)
    parts.append(
        "Read across a row for the same factor under different representations. 'gap' is\n"
        "comparable everywhere; 'Δfloor' only where each bundle has its own floor, and\n"
        "'Δvoxels' only where the bundle stored one (export with --raw-grid).\n"
    )
    if options is not None:
        graph = format_graph(results, truth, options, stability)
        if graph:
            parts.append(graph)
    return "\n".join(parts)


def write_csv(path, results):
    columns = [
        "bundle",
        "block",
        "factor",
        "real",
        "null",
        "gap",
        "std",
        "mcc",
        "floor_gap",
        "delta_floor",
        "voxels_gap",
        "delta_voxels",
        "probe_features",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for label, result in results.items():
            for block in ("content", "style"):
                rows = result.get(block, {})
                width = rows.get("_block", {}).get("probe_features")
                for name, row in rows.items():
                    if name == "_block":
                        continue
                    writer.writerow({**row, "bundle": label, "block": block, "factor": name, "probe_features": width})


def write_graph_csv(path, results, truth, options):
    """One row per (source, alpha): the skeleton scored against the true adjacency.

    Every alpha is written, not just the headline, so the alpha sweep can be replotted
    without re-running PC — and so a reader can see whether a source's advantage at the
    prespecified alpha survives the rest of the sweep.
    """
    columns = [
        "source",
        "alpha",
        "selected",
        "f1",
        "precision",
        "recall",
        "skeleton_shd",
        "tp",
        "fp",
        "fn",
        "exact_match",
        "cpdag_shd",
        "raw_r2_mean",
        "partial_r2_mean",
        "readout_dim",
        "graph_samples",
        "readout_mode",
        "indep_test",
    ]
    rows = graph_panels(results, truth)
    if not rows:
        return False
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for label, panel in rows:
            best = panel.get("best") or {}
            for sweep in panel.get("alpha_sweep", []):
                if "f1" not in sweep:
                    continue
                selected = []
                if abs(sweep["alpha"] - options.diagnostic_alpha) < 1e-12:
                    selected.append("prespecified")
                if best and abs(sweep["alpha"] - best["alpha"]) < 1e-12:
                    selected.append("best_f1")
                writer.writerow(
                    {
                        **sweep,
                        "source": label,
                        "selected": "+".join(selected),
                        "cpdag_shd": (sweep.get("orientation") or {}).get("cpdag_shd"),
                        "raw_r2_mean": panel.get("raw_r2_mean"),
                        "partial_r2_mean": panel.get("partial_r2_mean"),
                        "readout_dim": panel.get("graph_readout_dim"),
                        "graph_samples": panel.get("graph_samples"),
                        "readout_mode": panel.get("readout_mode"),
                        "indep_test": panel.get("indep_test"),
                    }
                )
    return True


def _self_test():
    """Two representations of one planted SCM, scored through the real path — probes and PC."""
    rng = np.random.RandomState(0)
    n, k = 240, 3
    z = rng.randn(n, k)
    z[:, 1] += 1.3 * z[:, 0]
    z[:, 2] += 1.3 * z[:, 1]
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
    names = ["a", "b", "c"]
    strong = z @ rng.randn(k, 32) + 0.1 * rng.randn(n, 32)
    weak = z @ rng.randn(k, 32) + 3.0 * rng.randn(n, 32)

    def make(X):
        return dict(
            path="<self-test>",
            X=X,
            raw=None,
            z_content=z,
            z_style=None,
            adjacency=adjacency,
            content_names=names,
            style_names=[],
            meta={},
            view="1",
        )

    bundles = {"strong": make(strong), "weak": make(weak)}
    options = argparse.Namespace(
        probe_kind="ridge",
        seeds=(0, 1),
        n_splits=3,
        n_null=2,
        null_seed=0,
        probe_dim=0,
        with_graph=True,
        pc_ceiling=True,
        alphas=(0.05,),
        diagnostic_alpha=0.05,
        orientation=True,
        indep_test="fisherz",
        max_cond_set=None,
        readout_dim=8,
        holdout_readout=False,
        graph_probe_dim=16,
        graph_repeats=4,
        graph_subsample=0.8,
    )
    records, problems = check_alignment(bundles, strict=True)
    assert not problems, problems
    assert len({r["factor_digest"] for r in records.values()}) == 1, "identical factors must digest alike"

    assert common_readout(bundles, {}, options) == 8, "an explicit readout width must be honoured"
    truth = truth_panel(bundles, options)
    assert truth and truth["best"], "the ceiling panel must run PC on the factors themselves"

    scored = argparse.Namespace(**{**vars(options), "pc_ceiling": False})
    results = score_all(bundles, {}, scored)

    sources, gt, adjacency = graph_sources(bundles, {}, scored)
    assert [label for label, _X in sources] == [CEILING_LABEL, "strong", "weak"], sources
    # --graph-probe-dim must reach the panel's own features, which --probe-dim does not.
    assert dict(sources)["strong"].shape[1] == 16, dict(sources)["strong"].shape
    assert dict(sources)[CEILING_LABEL].shape[1] == 3, "the ceiling is never reduced"
    stability = graph_stability(sources, gt, adjacency, scored, scored.graph_repeats, scored.graph_subsample)
    print(format_report(results, problems, truth, scored, stability))

    gaps = {label: result["content"]["_block"]["mean_gap"] for label, result in results.items()}
    assert gaps["strong"] > gaps["weak"], gaps
    assert gaps["strong"] > 0.5, gaps

    # The ceiling is scored once and reused; a per-bundle copy would repeat the same search.
    assert all("truth" not in (r["graph"] or {}) for r in results.values()), "ceiling must not be repeated"
    rows = dict(graph_panels(results, truth))
    assert set(rows) == {"truth (ceiling)", "strong", "weak"}, sorted(rows)
    for label, panel in rows.items():
        row = _sweep_row(panel, options.diagnostic_alpha)
        assert row, f"{label} has no row at the prespecified alpha"
        assert 0.0 <= row["f1"] <= 1.0 and row["skeleton_shd"] >= 0, row
        assert row["orientation"]["cpdag_shd"] >= 0, row["orientation"]
        # The ceiling's features ARE the 3 factors, so it reads out at 3 whatever is asked.
        expected = 3 if label == CEILING_LABEL else 8
        assert panel["graph_readout_dim"] == expected, (label, panel["graph_readout_dim"])
    assert rows["strong"]["best"]["f1"] >= rows["weak"]["best"]["f1"], "cleaner features, no worse graph"

    assert set(stability) == {CEILING_LABEL, "strong", "weak"}, sorted(stability)
    for label, entry in stability.items():
        assert entry["repeats"] == scored.graph_repeats, entry
        assert entry["subsample"] == int(round(len(gt) * scored.graph_subsample)), entry
        assert 0.0 <= entry["f1_mean"] <= 1.0 and entry["f1_std"] >= 0.0, entry
    assert "f1_delta_mean" not in stability["strong"], "the reference source has no paired delta"
    assert stability["weak"]["f1_delta_vs"] == "strong", stability["weak"]
    assert stability["weak"]["f1_delta_mean"] <= 0.0, stability["weak"]

    mismatched = {"strong": bundles["strong"], "shifted": make(weak)}
    mismatched["shifted"]["z_content"] = z + 1.0
    assert check_alignment(mismatched, strict=False)[1], "a different factor draw must be reported"
    print(
        "SELF-TEST OK: matched scoring separates the two, PC scores against the true "
        "adjacency with a resampling band, and a factor mismatch is caught."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundles", nargs="+", metavar="LABEL=PATH", help="Two or more bundles to compare")
    parser.add_argument(
        "--floors",
        nargs="+",
        metavar="LABEL=PATH",
        default=[],
        help="Untrained twin per bundle label; each is only ever subtracted from its own bundle",
    )
    parser.add_argument("--view", default="1", choices=["1", "2", "both"])
    parser.add_argument("--out", type=Path, help="Write the full report as JSON (and .txt alongside)")
    parser.add_argument("--csv", type=Path, help="Write the per-factor rows as CSV")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Run on planted numpy data and exit")
    parser.add_argument(
        "--allow-row-mismatch",
        dest="strict_rows",
        action="store_false",
        help="Score bundles whose rows differ. The output is then not a comparison.",
    )

    probe = parser.add_argument_group("probes (one setting, applied to every bundle)")
    probe.add_argument("--probe-kind", default="ridge", choices=["ridge", "kernel", "mlp"])
    probe.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    probe.add_argument("--n-splits", type=int, default=5)
    probe.add_argument("--n-null", type=int, default=3)
    probe.add_argument("--null-seed", type=int, default=0)
    probe.add_argument(
        "--probe-dim", default=PROBE_DIM_AUTO, help="PCA width for every bundle: 'auto', an integer, or 0 for none"
    )
    probe.add_argument(
        "--equal-width",
        action="store_true",
        help="Override --probe-dim with the narrowest block's width, so every bundle is probed at "
        "the same capacity. With --with-graph it pins --readout-dim to what the narrowest block can "
        "reach as well, since the graph readout has its own width rule that caps at each block's "
        "feature count.",
    )

    graph = parser.add_argument_group("graph")
    graph.add_argument(
        "--with-graph",
        action="store_true",
        help="Also run PC on each representation and score the recovered skeleton against the true "
        "SCM adjacency. Slow: one PC search per source per alpha.",
    )
    graph.add_argument("--alphas", type=float, nargs="+", default=list(scorer.DEFAULT_ALPHAS))
    graph.add_argument("--diagnostic-alpha", type=float, default=0.05)
    graph.add_argument("--no-orientation", dest="orientation", action="store_false")
    graph.add_argument("--no-pc-ceiling", dest="pc_ceiling", action="store_false")
    graph.add_argument("--indep-test", default="fisherz", choices=list(INDEP_TESTS))
    graph.add_argument("--max-cond-set", type=int)
    graph.add_argument("--readout-dim", type=int)
    graph.add_argument(
        "--graph-probe-dim",
        default=0,
        help="PCA the features handed to the graph panel to this width (0 = off, 'auto' for "
        "run_dci_compare's rule). --probe-dim does not reach that panel. --equal-width sets this "
        "to the narrowest block's width for you.",
    )
    graph.add_argument(
        "--graph-repeats",
        type=int,
        default=0,
        help="Re-run PC on this many row subsamples to put an error bar on the graph scores. "
        "0 (default) skips it. The panel is deterministic, so without this a two-edge difference "
        "between two sources is indistinguishable from a real one. Costs one PC search per "
        "source per repeat.",
    )
    graph.add_argument(
        "--graph-subsample",
        type=float,
        default=0.8,
        help="Fraction of rows each --graph-repeats draw keeps, without replacement (default 0.8). "
        "Not a bootstrap: duplicated rows would inflate the dependence Fisher-Z tests for.",
    )
    graph.add_argument("--holdout-readout", action="store_true")

    cli = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    if cli.self_test:
        _self_test()
        return 0
    if not cli.bundles or len(cli.bundles) < 2:
        parser.error("--bundles needs at least two label=path entries (or pass --self-test)")
    if cli.probe_dim != PROBE_DIM_AUTO:
        try:
            cli.probe_dim = int(cli.probe_dim)
        except ValueError:
            parser.error(f"--probe-dim must be an integer or {PROBE_DIM_AUTO!r}")
    if cli.n_null < 0 or cli.n_splits < 2 or not cli.seeds:
        parser.error("Require --n-null >= 0, --n-splits >= 2 and at least one seed")
    if cli.graph_probe_dim != PROBE_DIM_AUTO:
        try:
            cli.graph_probe_dim = int(cli.graph_probe_dim)
        except ValueError:
            parser.error(f"--graph-probe-dim must be an integer or {PROBE_DIM_AUTO!r}")
    if cli.graph_repeats and cli.graph_repeats < 2:
        parser.error("--graph-repeats needs at least 2 draws to have a spread")
    if not 0 < cli.graph_subsample < 1:
        parser.error("--graph-subsample must be between 0 and 1, exclusive")
    cli.seeds = tuple(cli.seeds)

    bundle_paths = parse_named(cli.bundles, "--bundles")
    floor_paths = parse_named(cli.floors, "--floors")
    unknown = set(floor_paths) - set(bundle_paths)
    if unknown:
        parser.error(f"--floors labels not in --bundles: {', '.join(sorted(unknown))}")

    bundles = {label: scorer.load_bundle(path, cli.view) for label, path in bundle_paths.items()}
    floors = {label: scorer.load_bundle(path, cli.view) for label, path in floor_paths.items()}
    _records, problems = check_alignment(bundles, strict=cli.strict_rows)
    if cli.equal_width:
        cli.probe_dim = common_width(bundles, floors)
        logger.info("--equal-width: probing every bundle at %d features", cli.probe_dim)

    truth = None
    if cli.with_graph:
        if cli.equal_width:
            cli.readout_dim = common_readout(bundles, floors, cli)
            # --probe-dim does not reach the graph panel, so pin its features separately:
            # without this the panel's raw/partial R2 columns compare a narrow block against
            # a wide one and part of the difference is the extra features.
            cli.graph_probe_dim = common_width(bundles, floors)
            logger.info(
                "--equal-width: graph features at %d, readout at %d dims",
                cli.graph_probe_dim,
                cli.readout_dim,
            )
        # The ceiling depends only on the factors, which every bundle shares, so it is
        # scored once here and the per-bundle panels are told not to repeat it.
        if cli.pc_ceiling:
            logger.info("Scoring the ground-truth ceiling panel")
            truth = truth_panel(bundles, cli)
        cli = argparse.Namespace(**{**vars(cli), "pc_ceiling": False})

    results = score_all(bundles, floors, cli)

    stability = {}
    if cli.with_graph and cli.graph_repeats:
        sources, Z, adjacency = graph_sources(bundles, floors, cli)
        logger.info("Graph stability: %d repeats x %d sources", cli.graph_repeats, len(sources))
        stability = graph_stability(sources, Z, adjacency, cli, cli.graph_repeats, cli.graph_subsample)

    report = format_report(results, problems, truth, cli, stability)
    payload = {
        "bundles": {label: str(path) for label, path in bundle_paths.items()},
        "floors": {label: str(path) for label, path in floor_paths.items()},
        "row_alignment": problems or "verified",
        "options": scorer.options_record(cli),
        "truth_panel": truth,
        "graph_stability": stability,
        "results": results,
    }
    saved = []
    if cli.out:
        cli.out.parent.mkdir(parents=True, exist_ok=True)
        cli.out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
        cli.out.with_suffix(".txt").write_text(report)
        saved += [cli.out.resolve(), cli.out.with_suffix(".txt").resolve()]
    if cli.csv:
        cli.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(cli.csv, results)
        saved.append(cli.csv.resolve())
        # The graph rows are one per (source, alpha) rather than one per factor, so they get
        # their own file instead of being padded into the factor table's columns.
        graph_csv = cli.csv.with_name(f"{cli.csv.stem}_graph{cli.csv.suffix}")
        if write_graph_csv(graph_csv, results, truth, cli):
            saved.append(graph_csv.resolve())
    if not cli.quiet:
        print(report)
    for path in saved:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
