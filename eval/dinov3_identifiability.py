#!/usr/bin/env python
"""Identifiability of DINOv3 embeddings: do they recover the factors, and the SCM graph?

    python -m eval.dinov3_identifiability --embeddings emb.npz
    python -m eval.dinov3_identifiability --embeddings emb.npz --floor emb_random_init.npz \
        --out results/dinov3/report.json --csv results/dinov3/factors.csv

Input is what ``eval/dinov3_embed_synthetic.py`` writes.  Two questions, two protocols,
both taken from the scripts that already own them so a DINOv3 number and a VQ-VAE number
mean the same thing:

1. **Factor recovery** (table 1/2) — per-factor cross-validated probe R² with a
   permutation null, and block-MCC, from ``eval.identifiability_metrics``, batched the way
   ``eval.run_dci_compare._score_block`` batches them.  ``gap = real − null`` is the
   reportable column: at these feature widths a probe on a signal-free target does not
   score 0, it scores slightly negative, and the null is what removes that bias.  With
   ``--floor`` (embeddings from ``--random-init``) the ``Δfloor`` column removes what the
   *architecture* recovers before any pretraining, which on this generator is most of the
   global morphometry.  The voxel baseline column is the same probe on downsampled voxels:
   a feature that does not beat it has not earned its forward pass.

2. **Graph recovery** (table 3) — ``eval.run_causal_recovery.evaluate_arrays``, unchanged:
   parent-residualised partial R², then PC (causal-learn, Fisher-Z) on the supervised
   readout of the decoded factors, scored against the true skeleton over an alpha sweep.
   ``--orientation`` additionally scores edge directions against the true DAG's CPDAG —
   the CPDAG, because PC identifies a Markov equivalence class and the default ``chain``
   SCM's is entirely undirected.  That panel does its own scaling/PCA internally, so it is
   handed the unreduced features while tables 1/2 use ``--probe-dim``.

   The ``truth`` row runs the identical panel on the ground-truth factors themselves.  It
   is the ceiling: PC at this sample size cannot do better than that row, so a low F1
   above it is the embedding's fault and a low F1 *at* it is not.

Caveats that are not optional
-----------------------------
The graph panel selects alpha against the truth and reads the graph out of an in-sample
supervised fit, so it is an optimistic diagnostic, not held-out causal discovery — see
``eval/CAUSAL_EVALUATION.md``.  Both R² protocols differ (table 1 is 5-fold CV over seeds,
table 3 is the panel's single 70/30 Ridge split), so read each against its own column, not
across tables.  The scoring layer is torch-free: ``--self-test`` runs it on planted numpy
data with no embeddings file.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from eval.identifiability_metrics import block_mcc, cv_probe_r2_multi
from eval.run_dci_compare import PROBE_DIM_AUTO, _auto_probe_dim

logger = logging.getLogger(__name__)
DEFAULT_ALPHAS = (0.01, 0.05, 0.1, 0.2)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_bundle(path, view="1"):
    """One embeddings ``.npz`` -> the arrays the scoring functions take.

    ``view`` picks T1 (``1``), FLAIR (``2``) or ``both`` (their concatenation, which is
    what a downstream model with access to the pair would see).
    """
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"])) if "meta" in data else {}

    def _views(prefix):
        keys = [f"{prefix}{v}" for v in ("1", "2")] if view == "both" else [f"{prefix}{view}"]
        arrays = [data[key] for key in keys if key in data]
        if len(arrays) != len(keys):
            raise KeyError(f"{path} has no {', '.join(keys)}; it was written with --views {meta.get('views')}")
        return np.concatenate(arrays, axis=1).astype(np.float64)

    z_content = data["z_content"].astype(np.float64)
    n_content, n_style = z_content.shape[1], 0
    style = None
    if "z_style_v1" in data:
        # Style is per view; score view 1's, which is the view whose embeddings are
        # scored unless --view says otherwise.
        style = data["z_style_v1" if view != "2" else "z_style_v2"].astype(np.float64)
        n_style = style.shape[1]
    return dict(
        path=str(path),
        X=_views("emb_view"),
        raw=_views("raw_view") if any(f"raw_view{v}" in data for v in ("1", "2")) else None,
        z_content=z_content,
        z_style=style,
        adjacency=data["causal_adj"].astype(bool) if "causal_adj" in data else None,
        content_names=meta.get("content_factor_names") or [f"d{d}" for d in range(n_content)],
        style_names=meta.get("style_factor_names") or [f"s{d}" for d in range(n_style)],
        meta=meta,
        view=view,
    )


def usable_factors(Z, names):
    """Drop constant columns: they carry no signal and make Fisher-Z undefined."""
    keep = [j for j in range(Z.shape[1]) if np.std(Z[:, j]) > np.finfo(float).eps]
    dropped = [names[j] for j in range(Z.shape[1]) if j not in keep]
    if dropped:
        logger.warning("Dropping constant ground-truth factor(s): %s", ", ".join(dropped))
    return Z[:, keep], [names[j] for j in keep], keep


def reduce_features(X, probe_dim, seed=0):
    """PCA the feature block when it is in the p>>n regime, by run_dci_compare's rule.

    ``auto`` touches only blocks wider than N/4 — the regime where a ridge probe on a
    signal-free target returns a systematically negative R², which penalises exactly the
    weak factors under investigation.  Concatenating slices puts these embeddings there:
    3 axes x 3 slices x 2048 dims is 18k features against a few hundred samples.
    """
    from sklearn.decomposition import PCA

    width = _auto_probe_dim(*X.shape) if probe_dim == PROBE_DIM_AUTO else int(probe_dim)
    if width <= 0 or X.shape[1] <= width:
        return X, X.shape[1]
    return PCA(n_components=min(width, X.shape[0]), random_state=seed).fit_transform(X), width


# --------------------------------------------------------------------------- #
# Table 1/2 — factor recovery
# --------------------------------------------------------------------------- #


def factor_recovery(X, Z, names, options, rng):
    """Per-factor CV probe R² with a permutation null, plus matched |corr| from block-MCC.

    Real targets and their nulls share one batched probe per block, as in
    ``run_dci_compare._score_block``: one StandardScaler + ridge decomposition of ``X`` is
    reused across every target instead of being recomputed per factor and per permutation.
    """
    n_factors, n = Z.shape[1], len(Z)
    if n_factors == 0:
        return {}
    columns = [Z[:, j] for j in range(n_factors)]
    for j in range(n_factors):
        columns.extend(Z[rng.permutation(n), j] for _ in range(options.n_null))
    scores = cv_probe_r2_multi(
        X, np.column_stack(columns), n_splits=options.n_splits, seeds=options.seeds, kind=options.probe_kind
    )
    mcc = block_mcc(X, Z, kind=options.probe_kind, seeds=options.seeds, n_splits=options.n_splits)

    out = {}
    for j, name in enumerate(names):
        start = n_factors + j * options.n_null
        null = float(np.mean(scores["mean"][start : start + options.n_null])) if options.n_null else float("nan")
        out[name] = dict(
            real=float(scores["mean"][j]),
            std=float(scores["std"][j]),
            null=null,
            gap=float(scores["mean"][j]) - null,
            mcc=float(mcc["per_factor"][j]),
            mcc_std=float(mcc["per_factor_std"][j]),
        )
    out["_block"] = dict(
        mcc_mean=mcc["mean"],
        mcc_std=mcc["std"],
        mcc_assignment_identity=mcc["assignment_identity"],
        mean_gap=float(np.mean([v["gap"] for v in out.values()])),
    )
    return out


def score_block(bundle, floor, block, options, seed=0):
    """Table 1 (``content``) or table 2 (``style``) for one embedding set and its floor."""
    Z = bundle["z_content"] if block == "content" else bundle["z_style"]
    names = bundle["content_names"] if block == "content" else bundle["style_names"]
    if Z is None or Z.shape[1] == 0:
        return {}
    Z, names, _ = usable_factors(Z, names)
    X, width = reduce_features(bundle["X"], options.probe_dim, seed)
    rows = factor_recovery(X, Z, names, options, np.random.RandomState(options.null_seed))
    rows["_block"]["probe_features"] = width

    for label, source in (("floor", floor), ("voxels", bundle if bundle["raw"] is not None else None)):
        if source is None:
            continue
        features = source["X"] if label == "floor" else source["raw"]
        reference = factor_recovery(
            reduce_features(features, options.probe_dim, seed)[0],
            Z,
            names,
            options,
            np.random.RandomState(options.null_seed),
        )
        for name in names:
            rows[name][f"{label}_gap"] = reference[name]["gap"]
            rows[name][f"delta_{label}"] = rows[name]["gap"] - reference[name]["gap"]
            rows[name][f"{label}_mcc"] = reference[name]["mcc"]
        rows["_block"][f"{label}_mean_gap"] = reference["_block"]["mean_gap"]
        rows["_block"][f"{label}_mcc_mean"] = reference["_block"]["mcc_mean"]
    return rows


# --------------------------------------------------------------------------- #
# Table 3 — graph recovery
# --------------------------------------------------------------------------- #


def graph_panel(X, z_content, adjacency, options):
    """The shared causal panel, unchanged, on one feature set."""
    from eval.run_causal_recovery import evaluate_arrays

    try:
        return evaluate_arrays(
            X,
            z_content,
            adjacency,
            alphas=options.alphas,
            diagnostic_alpha=options.diagnostic_alpha,
            orientation=options.orientation,
        )
    except Exception as exc:  # noqa: BLE001 - one panel failing must not lose the tables
        logger.exception("Graph panel failed")
        return dict(status="error", reason=f"{type(exc).__name__}: {exc}")


def score_graphs(bundle, floor, options):
    """The panel for the embeddings, the ground-truth ceiling, and the untrained floor."""
    if bundle["adjacency"] is None:
        logger.warning("No causal_adj in the embeddings file; the generator was not causal, so no graph to recover")
        return {}
    Z, names, keep = usable_factors(bundle["z_content"], bundle["content_names"])
    adjacency = bundle["adjacency"][np.ix_(keep, keep)]
    panels = {"embeddings": graph_panel(bundle["X"], Z, adjacency, options)}
    if options.pc_ceiling:
        panels["truth"] = graph_panel(Z, Z, adjacency, options)
    if floor is not None:
        panels["floor"] = graph_panel(floor["X"], Z, adjacency, options)
    for panel in panels.values():
        panel["factor_names"] = names
    return panels


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _table(headers, rows):
    widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
    line = "  ".join("-" * w for w in widths)
    body = [
        "  ".join(str(v).ljust(w) if i == 0 else str(v).rjust(w) for i, (v, w) in enumerate(zip(row, widths))).rstrip()
        for row in [headers, *rows]
    ]
    return "\n".join([body[0], line, *body[1:]])


def _fmt(value, spec="+.3f"):
    return "—" if value is None or not np.isfinite(value) else format(value, spec)


def format_factor_table(rows, title, has_floor, has_voxels):
    if not rows:
        return f"{title}\n  (no factors)\n"
    headers = ["factor", "R²", "null", "gap", "±", "MCC"]
    if has_voxels:
        headers += ["voxels", "Δvox"]
    if has_floor:
        headers += ["floor", "Δfloor"]
    body = []
    for name, row in rows.items():
        if name == "_block":
            continue
        line = [
            name,
            _fmt(row["real"]),
            _fmt(row["null"]),
            _fmt(row["gap"]),
            _fmt(row["std"], ".3f"),
            _fmt(row["mcc"], ".3f"),
        ]
        if has_voxels:
            line += [_fmt(row.get("voxels_gap")), _fmt(row.get("delta_voxels"))]
        if has_floor:
            line += [_fmt(row.get("floor_gap")), _fmt(row.get("delta_floor"))]
        body.append(line)
    block = rows["_block"]
    footer = (
        f"  mean gap {_fmt(block['mean_gap'])}   block-MCC {_fmt(block['mcc_mean'], '.3f')}"
        f" ±{_fmt(block['mcc_std'], '.3f')}   probe features {block['probe_features']}"
    )
    if "floor_mean_gap" in block:
        footer += f"   floor mean gap {_fmt(block['floor_mean_gap'])}"
    return f"{title}\n{_table(headers, body)}\n{footer}\n"


def format_graph_table(panels, diagnostic_alpha):
    if not panels:
        return "3. GRAPH RECOVERY\n  unavailable — the generator was not causal.\n"
    headers = ["source", "alpha", "F1", "prec", "rec", "SHD", "exact", "raw R²", "partial R²"]
    body = []
    for label, panel in panels.items():
        best = panel.get("best")
        if not best:
            body.append(
                [
                    label,
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    _fmt(panel.get("raw_r2_mean")),
                    _fmt(panel.get("partial_r2_mean")),
                ]
            )
            continue
        body.append(
            [
                label,
                f"{best['alpha']:g}",
                f"{best['f1']:.3f}",
                f"{best['precision']:.3f}",
                f"{best['recall']:.3f}",
                str(best["skeleton_shd"]),
                "yes" if best["exact_match"] else "no",
                _fmt(panel["raw_r2_mean"]),
                _fmt(panel["partial_r2_mean"]),
            ]
        )
    out = ["3. GRAPH RECOVERY — undirected skeleton, best-F1 alpha selected against truth", _table(headers, body)]

    per_factor = panels["embeddings"].get("factors")
    if per_factor:
        rows = [
            [
                factor["name"],
                str(factor["parents"]),
                _fmt(factor["raw_r2"]),
                _fmt(factor["partial_r2"]),
                _fmt(factor["gap"]),
                _fmt(factor.get("readout_test_r2")),
            ]
            for factor in per_factor
        ]
        out += [
            "",
            "  per factor (embeddings panel)",
            _table(["factor", "parents", "raw", "partial", "gap", "PC R²"], rows),
        ]

    oriented = [
        (label, row)
        for label, panel in panels.items()
        for row in panel.get("alpha_sweep", [])
        if "orientation" in row and abs(row["alpha"] - diagnostic_alpha) < 1e-12
    ]
    if oriented:
        rows = [
            [
                label,
                str(o["correct_directed"]),
                str(o["reversed"]),
                str(o["undirected_in_estimate"]),
                str(o["directed_in_estimate"]),
                str(o["bidirected_in_estimate"]),
                str(o["both_undirected"]),
                str(o["cpdag_shd"]),
                "yes" if o["cpdag_exact_match"] else "no",
            ]
            for label, row in oriented
            for o in [row["orientation"]]
        ]
        out += [
            "",
            f"  orientation vs the true CPDAG at alpha={diagnostic_alpha:g}"
            " (PC identifies an equivalence class, so undirected can be correct)",
            _table(
                ["source", "correct", "rev", "undir_est", "dir_est", "bidir", "both_undir", "CPDAG SHD", "exact"], rows
            ),
        ]
    out.append(
        "\n  Table 3's raw/partial R² use the panel's full-width Ridge probe on a single 70/30 split."
        "\n  Where the features are wider than the sample count that probe is biased low; table 1's"
        "\n  null-corrected gap is the reading to quote for factor recovery."
    )
    return "\n".join(out) + "\n"


def format_verdict(content, panels, has_floor):
    lines = ["4. VERDICT"]
    block = content.get("_block", {})
    learned = [name for name, row in content.items() if name != "_block" and row["gap"] > 0.1]
    lines.append(f"  {len(learned)}/{max(len(content) - 1, 0)} content factors decode above their permutation null.")
    if "voxels_mean_gap" in block:
        delta = block["mean_gap"] - block["voxels_mean_gap"]
        verb = "above" if delta > 0 else "BELOW"
        lines.append(f"  Mean gap is {abs(delta):.3f} {verb} the downsampled-voxel baseline.")
    if has_floor:
        delta = block["mean_gap"] - block.get("floor_mean_gap", float("nan"))
        lines.append(
            f"  Pretraining is worth {delta:+.3f} mean R² over the same architecture at random init"
            + ("." if np.isfinite(delta) else " (floor unavailable).")
        )
    else:
        lines.append(
            "  No --floor given. An absolute R² here is mostly a statement about the architecture:"
            " on this generator an UNTRAINED encoder already reads six of nine content factors above 0.8."
        )
    embeddings, truth = panels.get("embeddings", {}), panels.get("truth")
    if embeddings.get("best") and truth and truth.get("best"):
        gap = embeddings["best"]["f1"] - truth["best"]["f1"]
        lines.append(
            f"  Skeleton F1 {embeddings['best']['f1']:.3f} against a ground-truth ceiling of "
            f"{truth['best']['f1']:.3f} ({gap:+.3f}). PC on the true factors is the most any"
            " representation could score at this sample size."
        )
    elif embeddings.get("best"):
        lines.append(f"  Skeleton F1 {embeddings['best']['f1']:.3f}; no ground-truth ceiling was computed.")
    lines.append(
        "  Alpha is selected against the truth and the graph readout is in-sample, so table 3 is an"
        " optimistic diagnostic, not held-out causal discovery."
    )
    return "\n".join(lines) + "\n"


def format_report(result):
    meta = result["embeddings_meta"]
    header = [
        "=" * 100,
        f"{'3DINO' if meta.get('backbone') == '3dino' else 'DINOv3'} IDENTIFIABILITY — {meta.get('model_id', '?')}"
        + ("  [RANDOM INIT]" if meta.get("random_init") else ""),
        "=" * 100,
        f"  embeddings   {result['embeddings_path']}  (view {result['view']}, {result['num_samples']} samples,"
        f" {result['num_features']} features)",
        (
            f"  volume       {meta.get('volume_size')}³ · token pool {meta.get('token_pool')} · window {meta.get('window')}"
            if meta.get("backbone") == "3dino"
            else f"  slices       {meta.get('slices')} per axis on {','.join(meta.get('axes', []))},"
            f" {meta.get('slice_agg')} · token pool {meta.get('token_pool')} · window {meta.get('window')}"
        ),
        f"  generator    {json.dumps(meta.get('generator', {}), sort_keys=True)[:200]}",
        f"  floor        {result.get('floor_path') or 'none'}",
        "",
    ]
    parts = [
        "\n".join(header),
        format_factor_table(
            result["content"], "1. CONTENT FACTORS", result.get("floor_path") is not None, result["has_voxels"]
        ),
        format_factor_table(
            result["style"], "2. STYLE / NUISANCE FACTORS", result.get("floor_path") is not None, result["has_voxels"]
        ),
        format_graph_table(result["graph"], result["options"]["diagnostic_alpha"]),
        format_verdict(result["content"], result["graph"], result.get("floor_path") is not None),
    ]
    return "\n".join(parts)


def write_csv(path, result):
    columns = [
        "block",
        "factor",
        "real",
        "null",
        "gap",
        "std",
        "mcc",
        "voxels_gap",
        "delta_voxels",
        "floor_gap",
        "delta_floor",
        "raw_r2",
        "partial_r2",
    ]
    panel = {factor["name"]: factor for factor in result["graph"].get("embeddings", {}).get("factors", [])}
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for block in ("content", "style"):
            for name, row in result[block].items():
                if name == "_block":
                    continue
                causal = panel.get(name, {})
                writer.writerow(
                    {
                        **row,
                        "block": block,
                        "factor": name,
                        "raw_r2": causal.get("raw_r2"),
                        "partial_r2": causal.get("partial_r2"),
                    }
                )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def options_record(options):
    """JSON-safe copy of the scoring knobs — Paths and tuples do not survive json.dumps."""
    raw = vars(options) if isinstance(options, argparse.Namespace) else dict(options)
    safe = {}
    for key, value in raw.items():
        if isinstance(value, tuple):
            value = list(value)
        safe[key] = value if isinstance(value, (bool, int, float, str, list, type(None))) else str(value)
    return safe


def score(bundle, floor, options):
    content = score_block(bundle, floor, "content", options)
    style = score_block(bundle, floor, "style", options)
    graph = score_graphs(bundle, floor, options) if options.with_graph else {}
    return dict(
        embeddings_path=bundle["path"],
        floor_path=floor["path"] if floor else None,
        view=bundle["view"],
        num_samples=int(len(bundle["X"])),
        num_features=int(bundle["X"].shape[1]),
        has_voxels=bundle["raw"] is not None,
        embeddings_meta=bundle["meta"],
        floor_meta=floor["meta"] if floor else None,
        options=options_record(options),
        content=content,
        style=style,
        graph=graph,
    )


def _self_test():
    """Plant a chain SCM, mix it into features, and check every table reads it."""
    rng = np.random.RandomState(0)
    n, n_factors = 400, 3
    z = rng.randn(n, n_factors)
    z[:, 1] += 1.3 * z[:, 0]
    z[:, 2] += 1.3 * z[:, 1]
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
    mixing = rng.randn(n_factors, 24)
    names = ["a", "b", "c"]

    def bundle(X, raw=None, path="planted"):
        return dict(
            path=path,
            X=X,
            raw=raw,
            z_content=z,
            z_style=None,
            adjacency=adjacency,
            content_names=names,
            style_names=[],
            meta={},
            view="1",
        )

    options = argparse.Namespace(
        probe_kind="ridge",
        seeds=(0, 1),
        n_splits=5,
        n_null=2,
        null_seed=0,
        probe_dim=PROBE_DIM_AUTO,
        alphas=[0.05],
        diagnostic_alpha=0.05,
        orientation=True,
        pc_ceiling=True,
        with_graph=True,
    )
    signal = bundle(z @ mixing + 0.1 * rng.randn(n, 24), raw=rng.randn(n, 8))
    noise = bundle(rng.randn(n, 24), path="floor")

    rows = score_block(signal, noise, "content", options)
    for name in names:
        assert rows[name]["gap"] > 0.8, (name, rows[name])
        assert rows[name]["delta_floor"] > 0.7, (name, rows[name])
        assert rows[name]["delta_voxels"] > 0.7, (name, rows[name])
    assert rows["_block"]["mcc_mean"] > 0.9
    print(f"  self-test: planted factors  gap {rows['a']['gap']:+.3f}  MCC {rows['_block']['mcc_mean']:.3f}")

    empty = score_block(noise, None, "content", options)
    assert all(abs(empty[name]["gap"]) < 0.15 for name in names), empty
    print(f"  self-test: signal-free features  gap {empty['a']['gap']:+.3f} (null-corrected, so ~0)")

    panels = score_graphs(signal, noise, options)
    assert panels["embeddings"]["best"]["exact_match"], panels["embeddings"]["best"]
    assert panels["truth"]["best"]["exact_match"], panels["truth"]["best"]
    # A chain's CPDAG is fully undirected: an exact CPDAG match must NOT claim orientations.
    oriented = panels["embeddings"]["alpha_sweep"][0]["orientation"]
    assert oriented["cpdag_exact_match"] and oriented["both_undirected"] == 2 and oriented["correct_directed"] == 0
    assert panels["embeddings"]["factors"][1]["partial_r2"] > 0.5
    print(
        f"  self-test: PC skeleton F1 {panels['embeddings']['best']['f1']:.3f}"
        f"  ceiling {panels['truth']['best']['f1']:.3f}  CPDAG SHD {oriented['cpdag_shd']}"
    )

    # A representation that keeps only the root factor: the child's partial R² must drop
    # even though its raw R² stays high through the parent.
    parent_only = bundle(np.column_stack([z[:, 0], rng.randn(n, 10)]))
    child = score_graphs(parent_only, None, options)["embeddings"]["factors"][1]
    assert child["raw_r2"] > 0.4 and child["partial_r2"] < 0.15, child
    print(f"  self-test: parent-only features  child raw {child['raw_r2']:+.3f}  partial {child['partial_r2']:+.3f}")

    result = score(signal, noise, options)
    assert "1. CONTENT FACTORS" in format_report(result) and "3. GRAPH RECOVERY" in format_report(result)
    json.dumps(result, allow_nan=False, default=float)
    print("  self-test: report renders and serialises")
    print("  self-test PASSED")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", type=Path, help="npz from eval.dinov3_embed_synthetic")
    parser.add_argument("--floor", type=Path, help="npz from the same script with --random-init")
    parser.add_argument("--view", default="1", choices=["1", "2", "both"], help="1=T1, 2=FLAIR, both=concatenated")
    parser.add_argument("--out", type=Path, help="Write the full report as JSON")
    parser.add_argument("--csv", type=Path, help="Write the per-factor table as CSV")
    parser.add_argument("--quiet", action="store_true", help="Write the outputs without printing the report")
    parser.add_argument("--self-test", action="store_true", help="Run the numpy self-test and exit")

    probe = parser.add_argument_group("probes")
    probe.add_argument("--probe-kind", default="ridge", choices=["ridge", "kernel", "mlp"])
    probe.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="CV seeds")
    probe.add_argument("--n-splits", type=int, default=5)
    probe.add_argument("--n-null", type=int, default=3, help="Permutation-null repeats per factor")
    probe.add_argument("--null-seed", type=int, default=0)
    probe.add_argument(
        "--probe-dim",
        default=PROBE_DIM_AUTO,
        help="PCA width for tables 1/2: 'auto' (default) reduces only blocks wider than N/4, "
        "an integer reduces every block, 0 leaves them alone. Table 3 is unaffected.",
    )

    graph = parser.add_argument_group("graph")
    graph.add_argument("--no-graph", dest="with_graph", action="store_false", help="Skip table 3")
    graph.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    graph.add_argument("--diagnostic-alpha", type=float, default=0.05, help="Fixed alpha for the orientation table")
    graph.add_argument("--no-orientation", dest="orientation", action="store_false", help="Skeleton scores only")
    graph.add_argument("--no-pc-ceiling", dest="pc_ceiling", action="store_false", help="Skip PC on the true factors")

    cli = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    if cli.self_test:
        _self_test()
        return 0
    if cli.embeddings is None:
        parser.error("--embeddings is required (or pass --self-test)")
    if cli.probe_dim != PROBE_DIM_AUTO:
        try:
            cli.probe_dim = int(cli.probe_dim)
        except ValueError:
            parser.error(f"--probe-dim must be an integer or {PROBE_DIM_AUTO!r}")
    if any(not 0 < a < 1 for a in cli.alphas) or not 0 < cli.diagnostic_alpha < 1:
        parser.error("PC alphas must be strictly between 0 and 1")
    if cli.n_null < 0 or cli.n_splits < 2 or not cli.seeds:
        parser.error("Require --n-null >= 0, --n-splits >= 2 and at least one seed")
    cli.seeds = tuple(cli.seeds)

    bundle = load_bundle(cli.embeddings, cli.view)
    floor = load_bundle(cli.floor, cli.view) if cli.floor else None
    if floor is not None and floor["X"].shape[0] != bundle["X"].shape[0]:
        parser.error("The floor embeddings must come from the same number of samples as the embeddings")
    if floor is not None and not np.allclose(floor["z_content"], bundle["z_content"]):
        parser.error("The floor embeddings were built from different factor draws; rerun it with the same generator")

    result = score(bundle, floor, cli)
    report = format_report(result)
    # Persist before printing, so a closed stdout cannot cost the scored run.
    saved = []
    if cli.out:
        cli.out.parent.mkdir(parents=True, exist_ok=True)
        cli.out.write_text(json.dumps(result, indent=2, default=float) + "\n")
        cli.out.with_suffix(".txt").write_text(report)
        saved += [cli.out.resolve(), cli.out.with_suffix(".txt").resolve()]
    if cli.csv:
        cli.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(cli.csv, result)
        saved.append(cli.csv.resolve())
    if not cli.quiet:
        print(report)
    for path in saved:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
