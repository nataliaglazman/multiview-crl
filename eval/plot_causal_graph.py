#!/usr/bin/env python
"""The content SCM itself, drawn — the true DAG a synthetic run samples z_content from.

``plot_causal_recovery`` draws what PC *recovered*; this draws what the generator actually
used, with no run and no scoring involved.  The graph is whatever
``synthetic_dataset.build_content_scm`` returns for a config's
``(synthetic_n_content, synthetic_causal_graph, synthetic_causal_edge_prob, synthetic_seed)``,
so the figure is reproducible from the YAML alone.

    python -m eval.plot_causal_graph --graph random --n-content 9 --edge-prob 0.5 --seed 42 \
        --out figures/

Note on the seed: ``SyntheticBrainDataset`` passes the *un-split-adjusted* ``synthetic_seed``
as ``scm_seed`` (``data/datasets.py``), precisely so train/val/test share one graph — so
``--seed`` here is the run's ``synthetic_seed``, not a split seed.

What it draws
-------------
``causal_graph.png``    Node-link, laid out in longest-path layers so every arrow points
                        right and the topological order is the reading order. Exogenous
                        factors (no parents — pure noise sources) are filled, endogenous
                        ones outlined, so "what is upstream of everything" is visible
                        without tracing arrows.
``causal_matrix.png``   The same graph as a parent-row x child-column grid. Strictly
                        upper-triangular by construction; a filled cell that is not is a
                        bug, and this is where it would show.

Each ships a ``.csv`` twin (edge list, and the 0/1 matrix), since a figure should never be
the only way to read an edge.

Colour
------
Imported from ``plot_identifiability`` rather than restated, so a hue means the same thing
across the eval figures. Fill-vs-outline carries the exogenous/endogenous split as well as
hue, which keeps it readable without colour.
"""

from __future__ import annotations

import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, Patch, Rectangle  # noqa: E402

from eval.plot_identifiability import THEME, _save, _style, _write_csv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Names as every other eval script spells them (eval/dci.py CONTENT_FACTOR_NAMES); dims past
# the renderer's nine have no anatomical meaning, so they stay bare indices.
FACTOR_NAMES = [
    "brain_size",
    "ventricle_size",
    "lesion_x",
    "lesion_y",
    "lesion_z",
    "cortical_thickness",
    "temporal_atrophy",
    "lr_asymmetry",
    "sulcal_widening",
]


def factor_names(n):
    return [FACTOR_NAMES[i] if i < len(FACTOR_NAMES) else f"z{i}" for i in range(n)]


# --------------------------------------------------------------------------- #
# The graph — from the generator when it can be imported, else re-derived
# --------------------------------------------------------------------------- #
def _adjacency_mirror(n_dims, graph_type, edge_prob, seed):
    """Torch-free mirror of ``build_content_scm``'s adjacency branch, byte-for-byte.

    ``synthetic_dataset`` imports torch and nibabel at module scope, which a plotting box
    need not have. Only the *adjacency* is mirrored (the mechanism weights need torch), the
    draw order is identical, and ``--self-test`` asserts the two agree wherever torch IS
    importable — so this cannot drift silently.
    """
    rng = np.random.RandomState(seed)
    adj = np.zeros((n_dims, n_dims), dtype=bool)
    if graph_type == "chain":
        for i in range(n_dims - 1):
            adj[i, i + 1] = True
    elif graph_type == "full":
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                adj[i, j] = True
    elif graph_type == "random":
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                if rng.rand() < edge_prob:
                    adj[i, j] = True
    else:
        raise ValueError(f"Unknown causal graph type: {graph_type}")
    return adj


def load_scm(n_dims, graph_type, edge_prob, seed):
    """Return (adj, weights) — weights is {child: [w per parent]} or None without torch."""
    try:
        from eval.synthetic_dataset import build_content_scm
    except ImportError as exc:
        logger.info("synthetic_dataset unavailable (%s) — adjacency from the torch-free mirror", exc)
        return _adjacency_mirror(n_dims, graph_type, edge_prob, seed), None
    scm = build_content_scm(n_dims, graph_type, edge_prob, seed)
    weights = {c: [float(v) for v in w] for c, w in scm["weights"].items()}
    return np.asarray(scm["adj"], dtype=bool), weights


def edge_list(adj):
    return [(i, j) for i in range(adj.shape[0]) for j in range(adj.shape[0]) if adj[i, j]]


def layers(adj):
    """Longest-path depth per node. Well defined: ``build_content_scm`` is upper-triangular."""
    n = adj.shape[0]
    depth = [0] * n
    for j in range(n):
        pa = np.where(adj[:, j])[0]
        depth[j] = 0 if len(pa) == 0 else 1 + max(depth[int(i)] for i in pa)
    out = {}
    for j, d in enumerate(depth):
        out.setdefault(d, []).append(j)
    return [out[d] for d in sorted(out)]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
NODE_W, NODE_H = 0.86, 0.42  # data units; layer pitch is 1.0 in x, 0.95 in y


def _positions(adj):
    pos = {}
    for x, members in enumerate(layers(adj)):
        for k, node in enumerate(members):
            pos[node] = (float(x), (k - (len(members) - 1) / 2.0) * 0.95)
    return pos


def _arc(pos, i, j):
    """(rad, extreme point) for the edge i -> j.

    Adjacent layers go straight; longer hops bow away from the crowded centre line so they
    do not pass under an intermediate node. matplotlib's ``arc3`` bows to the RIGHT of the
    direction of travel for positive rad, i.e. downward when travel is +x, and the arc's
    farthest point from the chord sits at ``mid + rad/2 * (dy, -dx)`` — which is what keeps
    the axes limits below from clipping a deep swing.
    """
    (xi, yi), (xj, yj) = pos[i], pos[j]
    dx, dy = xj - xi, yj - yi
    if dx <= 1:
        rad = 0.0
    else:
        rad = (1.0 if (yi + yj) / 2.0 <= 0 else -1.0) * min(0.08 + 0.05 * (dx - 1), 0.20)
    return rad, ((xi + xj) / 2.0 + rad * dy / 2.0, (yi + yj) / 2.0 - rad * dx / 2.0)


def fig_graph(adj, names, t, subtitle, path):
    pos = _positions(adj)
    n_layers = len(layers(adj))
    arcs = {(i, j): _arc(pos, i, j) for i, j in edge_list(adj)}

    # Limits from what is actually drawn — node bodies and every arc's deepest point.
    xs = [x for x, _ in pos.values()] + [p[0] for _, p in arcs.values()]
    ys = [y + s * NODE_H / 2 for _, y in pos.values() for s in (-1, 1)] + [p[1] for _, p in arcs.values()]
    x0, x1 = min(xs) - NODE_W / 2 - 0.2, max(xs) + NODE_W / 2 + 0.2
    y0, y1 = min(ys) - 0.22, max(ys) + 0.22

    fig, ax = plt.subplots(figsize=(1.9 * max(n_layers, 3) + 1.4, 1.9 * max(n_layers, 3) * (y1 - y0) / (x1 - x0) + 1.0))
    _style(ax, t, title="Content SCM (ground-truth causal graph)", subtitle=subtitle)
    ax.xaxis.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    roots = {j for j in range(adj.shape[0]) if not adj[:, j].any()}
    patches = {}
    for j, (x, y) in pos.items():
        exo = j in roots
        e = Ellipse(
            (x, y),
            NODE_W,
            NODE_H,
            facecolor=t["series"][1] if exo else t["surface"],
            edgecolor=t["series"][1] if exo else t["series"][0],
            linewidth=1.4,
            zorder=3,
        )
        ax.add_patch(e)
        patches[j] = e
        ax.text(
            x,
            y,
            names[j],
            ha="center",
            va="center",
            fontsize=7.2,
            color=t["surface"] if exo else t["ink"],
            zorder=4,
        )

    for (i, j), (rad, _) in arcs.items():
        ax.add_patch(
            FancyArrowPatch(
                pos[i],
                pos[j],
                patchA=patches[i],
                patchB=patches[j],
                shrinkA=0,
                shrinkB=0,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.9,
                color=t["ink2"],
                alpha=0.7,
                zorder=2,
            )
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.legend(
        handles=[
            Patch(facecolor=t["series"][1], edgecolor=t["series"][1], label="exogenous (no parents)"),
            Patch(facecolor=t["surface"], edgecolor=t["series"][0], label="endogenous"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=7.5,
        labelcolor=t["ink2"],
    )
    _save(fig, path, t)


def fig_matrix(adj, names, t, subtitle, path):
    n = adj.shape[0]
    fig, ax = plt.subplots(figsize=(0.52 * n + 3.0, 0.52 * n + 2.4))
    _style(ax, t, title="Adjacency: parent (row) -> child (column)", subtitle=subtitle)
    ax.xaxis.grid(False)

    for i in range(n):
        for j in range(n):
            ax.add_patch(
                Rectangle(
                    (j - 0.42, i - 0.42),
                    0.84,
                    0.84,
                    facecolor=t["series"][0] if adj[i, j] else t["floor_fill"],
                    edgecolor="none",
                    zorder=2,
                )
            )
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(n - 0.4, -0.6)  # row 0 on top, matching how the matrix is indexed
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # Column labels stay at the bottom: rotated labels along the top run into the subtitle.
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.5, color=t["ink2"])
    ax.set_yticklabels(names, fontsize=7.5, color=t["ink2"])
    ax.set_aspect("equal")
    _save(fig, path, t)


# --------------------------------------------------------------------------- #
def _self_test():
    adj = _adjacency_mirror(9, "chain", 0.5, 0)
    assert edge_list(adj) == [(i, i + 1) for i in range(8)], "chain is i -> i+1"
    assert [len(lyr) for lyr in layers(adj)] == [1] * 9, "a chain is nine single-node layers"

    full = _adjacency_mirror(5, "full", 0.5, 0)
    assert full.sum() == 10 and not np.tril(full).any(), "full = every i<j pair, upper triangular"

    rnd = _adjacency_mirror(9, "random", 0.5, 42)
    assert not np.tril(rnd).any(), "random DAG must stay upper triangular (acyclic)"
    assert rnd.sum() == 19, f"seed 42 / p=0.5 / n=9 draws 19 edges, got {rnd.sum()}"
    assert _adjacency_mirror(9, "random", 0.5, 42).tolist() == rnd.tolist(), "seeded draw is reproducible"
    assert {0, 2} == {j for j in range(9) if not rnd[:, j].any()}, "brain_size and lesion_x are the roots"
    assert [len(lyr) for lyr in layers(rnd)] == [2, 2, 2, 1, 2], "five longest-path layers"

    assert _adjacency_mirror(6, "random", 0.0, 1).sum() == 0, "edge_prob 0 gives an empty graph"
    assert _adjacency_mirror(6, "random", 1.0, 1).sum() == 15, "edge_prob 1 gives the full DAG"
    try:
        _adjacency_mirror(4, "nope", 0.5, 0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown graph type must raise, as build_content_scm does")

    try:
        from eval.synthetic_dataset import build_content_scm
    except ImportError:
        logger.info("self-test OK (torch absent — mirror-vs-generator check skipped)")
        return 0
    for args in ((9, "chain", 0.5, 42), (9, "random", 0.5, 42), (7, "random", 0.3, 0), (5, "full", 0.5, 3)):
        want = np.asarray(build_content_scm(*args)["adj"], dtype=bool)
        assert _adjacency_mirror(*args).tolist() == want.tolist(), f"mirror drifted from the generator at {args}"
    logger.info("self-test OK (mirror matches build_content_scm)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", default="random", choices=["chain", "full", "random"], help="synthetic_causal_graph.")
    p.add_argument("--n-content", type=int, default=9, help="synthetic_n_content.")
    p.add_argument("--edge-prob", type=float, default=0.5, help="synthetic_causal_edge_prob (random only).")
    p.add_argument("--seed", type=int, default=42, help="synthetic_seed — the run's SCM seed.")
    p.add_argument("--out", default="figures", help="Output directory.")
    p.add_argument("--dark", action="store_true", help="Re-step to the dark surface.")
    p.add_argument("--self-test", action="store_true", help="Run the invariant checks and exit.")
    cli = p.parse_args(argv)

    if cli.self_test:
        return _self_test()

    adj, weights = load_scm(cli.n_content, cli.graph, cli.edge_prob, cli.seed)
    names = factor_names(cli.n_content)
    edges = edge_list(adj)
    detail = f", edge_prob {cli.edge_prob:g}" if cli.graph == "random" else ""
    subtitle = f"{cli.graph} DAG, n_content {cli.n_content}{detail}, seed {cli.seed} — {len(edges)} edges"
    logger.info("%s", subtitle)

    t = THEME["dark" if cli.dark else "light"]
    os.makedirs(cli.out, exist_ok=True)

    graph_png = os.path.join(cli.out, "causal_graph.png")
    fig_graph(adj, names, t, subtitle, graph_png)
    rows = []
    for i, j in edges:
        pa = np.where(adj[:, j])[0].tolist()
        w = "" if weights is None else f"{weights[j][pa.index(i)]:.4f}"
        rows.append([i, names[i], j, names[j], w])
    _write_csv(
        graph_png.replace(".png", ".csv"),
        ["parent_index", "parent", "child_index", "child", "weight"],
        rows,
    )

    matrix_png = os.path.join(cli.out, "causal_matrix.png")
    fig_matrix(adj, names, t, subtitle, matrix_png)
    _write_csv(
        matrix_png.replace(".png", ".csv"),
        ["parent", *names],
        [[names[i], *[int(adj[i, j]) for j in range(len(names))]] for i in range(len(names))],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
