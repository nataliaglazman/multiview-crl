#!/usr/bin/env python
"""Causal discovery on the CHANNELS themselves, with no supervised readout.

    python -m eval.latent_causal_discovery --run-dirs RUN --reduce gap --floor --ceiling

``run_causal_recovery`` never hands PC a representation.  It hands PC ``n_content``
supervised ``RidgeCV`` reconstructions of the true factors, so the labels build the
variables before any independence test runs.  Measured consequence on this project: a
trained model scored SHD 8 against a ceiling -- PC on the TRUE factors -- of SHD 9, i.e.
it beat the truth, because a ridge readout of PCA components is smoother and closer to
linear-Gaussian than a ``leaky_relu`` factor and its dependencies are therefore EASIER to
detect than the real ones.  That is the readout doing the causal work.

This script removes the readout.  Each content channel is reduced to one scalar per
subject and PC runs on those ``d`` variables directly, so **no label touches the graph**.

What that costs, and how it is paid
-----------------------------------
Channels are not factors.  Under block identifiability the content block equals the true
factors up to an arbitrary invertible map, so channel 3 is not ``ventricle_size`` and a
graph over channels is not comparable to a graph over factors.  Scoring therefore needs a
correspondence, and this is the one place labels enter -- AFTER the graph is fixed, to name
nodes, never to build them.  Two guards keep that honest:

* the correspondence is ONE Hungarian assignment on ``|corr|``, not a per-factor supervised
  regression from a 128-dim PCA basis;
* every score is reported against a null over RANDOM assignments.  If the matched score
  sits inside that null, the recovered graph is not aligned to the truth at all, and the
  matching was doing the work instead of the representation.

An honest reading needs all three rows: ``--ceiling`` (PC on the true factors) bounds what
the estimator can do at this row count, ``--floor`` (an untrained twin) bounds what a random
projection already gives, and the assignment null bounds what the matching gives.

Reductions
----------
``--reduce`` picks the single scalar each channel contributes.  ``gap`` is the spatial mean,
which is what the GAP companion term optimises and what erases a localised factor (it
contributes ~1/P of the mean).  ``pc1`` is the channel's dominant spatial mode, which is NOT
constrained to uniform spatial weighting and so can carry a localised factor that ``gap``
drops.  ``std`` and ``max`` are the order statistics the stats pooling adds.  One scalar per
channel either way -- a multi-statistic pooling would put several nodes per channel and
break the one-node-one-channel premise the matching rests on.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from eval.run_causal_recovery import DEFAULT_ALPHAS, INDEP_TESTS, skeleton_metrics

logger = logging.getLogger(__name__)
REDUCTIONS = ("gap", "pc1", "std", "max")


def reduce_channels(maps, mode="gap"):
    """``(N, C, P)`` spatial maps -> ``(N, C)``, one scalar per channel per subject."""
    import numpy as np

    maps = np.asarray(maps, dtype=np.float64)
    if maps.ndim != 3:
        raise ValueError(f"expected (N, C, P) maps; got {maps.shape}")
    if mode == "gap":
        return maps.mean(axis=2)
    if mode == "std":
        return maps.std(axis=2)
    if mode == "max":
        return maps.max(axis=2)
    if mode == "pc1":
        from sklearn.decomposition import PCA

        # Per channel, the score on its dominant spatial mode. Sign is arbitrary and stays
        # so: the skeleton is estimated from correlations, which are sign-symmetric.
        out = np.empty((maps.shape[0], maps.shape[1]))
        for c in range(maps.shape[1]):
            block = maps[:, c, :]
            if np.allclose(block.std(axis=0), 0):
                out[:, c] = 0.0
                continue
            out[:, c] = PCA(n_components=1, random_state=0).fit_transform(block)[:, 0]
        return out
    raise ValueError(f"reduce must be one of {REDUCTIONS}, got {mode!r}")


def discover(values, alphas=DEFAULT_ALPHAS, indep_test="fisherz", max_cond_set=None):
    """``{alpha: boolean skeleton}`` over the columns of ``values``. No labels involved."""
    import numpy as np

    try:
        from causallearn.search.ConstraintBased.PC import pc
    except ImportError as exc:
        raise ImportError("Graph recovery requires causal-learn: python -m pip install causal-learn") from exc

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Need a 2D array with at least two columns")
    if not np.isfinite(values).all():
        raise ValueError("Reduced channels must be finite")
    keep = values.std(axis=0) > np.finfo(float).eps
    if not keep.all():
        # A dead channel has no conditional-independence structure to test; carrying it
        # would abort PC for the whole graph rather than for that one node.
        logger.warning("dropping %d constant channel(s) before PC", int((~keep).sum()))
    out = {}
    for alpha in alphas:
        graph = pc(values[:, keep], alpha=alpha, indep_test=indep_test, show_progress=False, max_k=max_cond_set).G.graph
        adjacency = (np.asarray(graph) != 0) | (np.asarray(graph).T != 0)
        np.fill_diagonal(adjacency, False)
        full = np.zeros((values.shape[1], values.shape[1]), dtype=bool)
        idx = np.flatnonzero(keep)
        full[np.ix_(idx, idx)] = adjacency
        out[alpha] = full
    return out


def match_channels(reduced, factors):
    """Hungarian assignment ``factor -> channel`` on ``|corr|``. The only use of labels.

    Runs AFTER discovery, so it can rename nodes but cannot create or remove an edge.
    Returns ``(assignment, quality)`` where ``quality[f]`` is the matched ``|corr|`` -- a
    weak match means the node this factor was scored at barely tracks it, which is the
    caveat that belongs beside the score.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    reduced, factors = np.asarray(reduced, dtype=float), np.asarray(factors, dtype=float)
    if len(reduced) != len(factors):
        raise ValueError("Reduced channels and factors must have the same number of rows")
    n_f, n_c = factors.shape[1], reduced.shape[1]
    if n_c < n_f:
        raise ValueError(f"Need at least as many channels as factors; got {n_c} < {n_f}")
    sim = np.zeros((n_f, n_c))
    for f in range(n_f):
        for c in range(n_c):
            if factors[:, f].std() > 0 and reduced[:, c].std() > 0:
                sim[f, c] = abs(np.corrcoef(factors[:, f], reduced[:, c])[0, 1])
    rows, cols = linear_sum_assignment(-sim)
    assignment = np.empty(n_f, dtype=int)
    quality = np.empty(n_f)
    for f, c in zip(rows, cols):
        assignment[f], quality[f] = c, sim[f, c]
    return assignment, quality


def score_assignment(adjacency, truth, assignment):
    """Skeleton metrics for the factor-level graph induced by one channel assignment."""
    import numpy as np

    adjacency = np.asarray(adjacency, dtype=bool)
    induced = adjacency[np.ix_(assignment, assignment)]
    return skeleton_metrics(induced, truth)


def assignment_null(adjacency, truth, n_channels, n_draws=200, seed=0):
    """F1/SHD over RANDOM factor->channel assignments of the SAME recovered graph.

    This is the null the matched score has to beat. It holds the graph fixed and varies only
    the naming, so it isolates exactly what the Hungarian step contributed: a matched score
    inside this null means the recovered structure is not aligned to the truth, however good
    the absolute number looks.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n_f = len(np.asarray(truth))
    f1, shd = [], []
    for _ in range(n_draws):
        draw = rng.permutation(n_channels)[:n_f]
        got = score_assignment(adjacency, truth, draw)
        f1.append(got["f1"])
        shd.append(got["skeleton_shd"])
    return dict(
        n_draws=n_draws,
        f1_mean=float(np.mean(f1)),
        f1_std=float(np.std(f1)),
        f1_max=float(np.max(f1)),
        shd_mean=float(np.mean(shd)),
        shd_min=int(np.min(shd)),
    )


def evaluate_reduced(reduced, factors, truth_dag, cli, label=""):
    """Discover on the channels, name the nodes, score against truth and against the null."""
    import numpy as np

    truth = np.asarray(truth_dag, dtype=bool)
    truth = truth | truth.T
    np.fill_diagonal(truth, False)
    assignment, quality = match_channels(reduced, factors)
    graphs = discover(reduced, cli.alphas, cli.indep_test, cli.max_cond_set)
    rows, best = [], None
    for alpha, adjacency in graphs.items():
        got = score_assignment(adjacency, truth, assignment)
        null = assignment_null(adjacency, truth, reduced.shape[1], cli.null_draws, seed=0)
        # An edge touching a channel no factor matched cannot be scored either way; saying
        # how many there are keeps "SHD 7" from reading as a statement about the whole graph.
        unmatched = sorted(set(range(reduced.shape[1])) - set(assignment.tolist()))
        pairs = np.triu_indices(reduced.shape[1], k=1)
        unscorable = int(sum(adjacency[i, j] and (i in unmatched or j in unmatched) for i, j in zip(*pairs)))
        row = dict(
            alpha=alpha,
            **got,
            null=null,
            f1_over_null=got["f1"] - null["f1_mean"],
            edges_total=int(adjacency[pairs].sum()),
            edges_unscorable=unscorable,
        )
        rows.append(row)
        if best is None or row["f1"] >= best["f1"]:
            best = row
    return dict(
        label=label,
        reduce=cli.reduce,
        n_channels=int(reduced.shape[1]),
        n_factors=int(factors.shape[1]),
        n_samples=int(len(reduced)),
        indep_test=cli.indep_test,
        max_cond_set=cli.max_cond_set,
        assignment=assignment.tolist(),
        match_quality=[float(q) for q in quality],
        alpha_sweep=rows,
        best=best,
    )


def format_report(results):
    """One block per row, with the null beside every score it has to be read against."""
    from eval.causal_factor_diagnostics import factor_name

    out = []
    head = ["Row", "reduce", "chan", "F1", "null F1", "learned", "SHD", "null SHD", "match |r|"]
    table = [head]
    for r in results:
        b = r.get("best")
        if not b:
            table.append([r["label"], r["reduce"], str(r["n_channels"]), *(["—"] * 6)])
            continue
        q = r["match_quality"]
        table.append(
            [
                r["label"],
                r["reduce"],
                str(r["n_channels"]),
                f"{b['f1']:.3f}",
                f"{b['null']['f1_mean']:.3f}",
                f"{b['f1_over_null']:+.3f}",
                str(b["skeleton_shd"]),
                f"{b['null']['shd_mean']:.1f}",
                f"{min(q):.2f}-{max(q):.2f}",
            ]
        )
    widths = [max(len(row[i]) for row in table) for i in range(len(head))]
    line = lambda row: "  ".join(  # noqa: E731
        v.ljust(w) if i < 2 else v.rjust(w) for i, (v, w) in enumerate(zip(row, widths))
    ).rstrip()
    out.append(line(table[0]))
    out.append("  ".join("-" * w for w in widths))
    out.extend(line(row) for row in table[1:])
    out.append("")
    out.append("No label built the graph: PC ran on the channels. Labels only NAMED the nodes afterwards,")
    out.append("by one Hungarian match on |corr|. 'null' holds that graph fixed and randomises the naming,")
    out.append("so a 'learned' at or below 0 means the recovered structure is not aligned to the truth.")
    out.append("'match |r|' is the range of matched correlations: a low one means the node a factor was")
    out.append("scored at barely tracks it, whatever the F1 says.")
    for r in results:
        b = r.get("best")
        if b and b["edges_unscorable"]:
            out.append(
                f"  {r['label']}: {b['edges_unscorable']} of {b['edges_total']} recovered edges touch an "
                f"unmatched channel and are outside the score."
            )
    if results and results[0].get("match_quality"):
        out.append("\nPer-factor match quality (best channel for each factor, |corr|):")
        for r in results:
            q = r.get("match_quality") or []
            named = ", ".join(f"{factor_name(i)} {v:.2f}" for i, v in enumerate(q))
            out.append(f"  {r['label']}: {named}")
    return "\n".join(out) + "\n"


def _self_test():
    """Checks for the label-free paths: reduction, matching, scoring and the null.

    The failure mode these guard against is a WRONG NUMBER rather than an error -- a
    matching that silently permutes, or a null that is not actually a null.
    """
    import numpy as np

    failures = []

    def check(label, condition):
        if not condition:
            failures.append(label)
        print(f"  {'ok  ' if condition else 'FAIL'}  {label}")

    rng = np.random.default_rng(0)
    N, C, P = 400, 5, 64
    maps = rng.standard_normal((N, C, P))
    localised = rng.standard_normal(N)
    # GAP attenuates a one-position signal to A/P while the noise in the mean falls only as
    # 1/sqrt(P), so the amplitude sets exactly how much survives: measured here, |r| runs
    # 0.16 -> 0.57 as A goes 1.5 -> 6. A=2.5 puts GAP in the regime the flag exists for
    # (|r| 0.27) while leaving the signal obvious at its own position (SNR 2.5).
    maps[:, 2, 17] += 2.5 * localised  # one channel, one position

    gap, pc1 = reduce_channels(maps, "gap"), reduce_channels(maps, "pc1")
    check("reductions keep the shape (N, C)", gap.shape == (N, C) and pc1.shape == (N, C))
    r_gap = abs(np.corrcoef(gap[:, 2], localised)[0, 1])
    r_pc1 = abs(np.corrcoef(pc1[:, 2], localised)[0, 1])
    check(f"gap loses a localised factor (|r|={r_gap:.2f})", r_gap < 0.40)
    check(f"pc1 keeps it (|r|={r_pc1:.2f})", r_pc1 > 0.80)
    check(f"pc1 beats gap by a wide margin ({r_pc1 / r_gap:.1f}x)", r_pc1 > 2.0 * r_gap)
    check("std/max run", reduce_channels(maps, "std").shape == (N, C))
    try:
        reduce_channels(maps, "stats")
        check("an unknown reduction is rejected", False)
    except ValueError:
        check("an unknown reduction is rejected", True)

    # Channels are a known permutation of the factors, plus two spare channels.
    z = rng.standard_normal((N, 4))
    perm = [3, 0, 2, 1]
    reduced = np.column_stack([z[:, perm], rng.standard_normal((N, 2))])
    assignment, quality = match_channels(reduced, z)
    check("matching inverts a known permutation", list(assignment) == [1, 3, 2, 0])
    check("matched quality is ~1 on exact copies", float(min(quality)) > 0.99)

    truth = np.zeros((4, 4), dtype=bool)
    truth[0, 1] = truth[1, 2] = truth[2, 3] = True
    channel_graph = np.zeros((6, 6), dtype=bool)
    for f1, f2 in ((0, 1), (1, 2), (2, 3)):
        channel_graph[assignment[f1], assignment[f2]] = True
    channel_graph |= channel_graph.T
    exact = score_assignment(channel_graph, truth | truth.T, assignment)
    check("the right assignment recovers the planted chain exactly", exact["skeleton_shd"] == 0)
    null = assignment_null(channel_graph, truth | truth.T, 6, n_draws=300, seed=1)
    check(f"a random naming scores worse (null F1 {null['f1_mean']:.2f})", null["f1_mean"] < exact["f1"])
    check("the null never beats an exact match here", null["shd_min"] >= exact["skeleton_shd"])

    wrong = np.asarray([1, 0, 3, 2])
    check(
        "a wrong assignment scores worse than the right one",
        score_assignment(channel_graph, truth | truth.T, wrong)["f1"] < exact["f1"],
    )
    try:
        match_channels(rng.standard_normal((N, 2)), z)
        check("fewer channels than factors is rejected", False)
    except ValueError:
        check("fewer channels than factors is rejected", True)

    print("\n" + ("FAILED: " + ", ".join(failures) if failures else "All checks passed."))
    return 1 if failures else 0


def evaluate_run(run_dir, cli, random_init=False, init_seed=0):
    """Encode a run, reduce its channels, and score the label-free graph."""
    import numpy as np

    from eval.run_causal_recovery import extract_content
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    with (run_dir / "settings.json").open() as fh:
        settings = json.load(fh)
    if not settings.get("synthetic_causal", False):
        return None, None, dict(status="skipped", reason="settings['synthetic_causal'] is False")
    checkpoint = run_dir / cli.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    logger.info("Encoding %s%s", run_dir, f" [UNTRAINED FLOOR, seed {init_seed}]" if random_init else "")
    model, args, device = load_model_from_run_dir(
        str(run_dir), str(checkpoint), device=cli.device, random_init=random_init, seed=init_seed
    )
    levels = settings.get("content_style_levels") or [0]
    level = cli.level if cli.level is not None else levels[0]
    dataset = build_synthetic_test_set(args, cli.num_samples, cache=False, causal=True)
    scm = getattr(getattr(dataset, "_inner", dataset), "scm", None)
    if scm is None:
        raise ValueError("Matched synthetic dataset did not expose a causal SCM")
    # patch_grid, not gap: the reduction is this script's own choice and must be applied to
    # the spatial map, not to a map GAP already collapsed.
    flat, z = extract_content(model, dataset, device, level, cli.pooling, cli.batch_size, cli.num_workers)
    maps = flat.reshape(len(flat), -1, int(np.prod(cli.pooling)))
    return maps, z, scm["adj"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs", nargs="+", default=[], help="Run directories")
    parser.add_argument("--checkpoint", default="vqvae_model.pt")
    parser.add_argument("--level", type=int, help="Default: first content_style_levels entry")
    parser.add_argument("--pooling", default="4,4,4", help="3D patch grid the channels are reduced over")
    parser.add_argument("--num-samples", type=int, help="Default: each run's synthetic_num_test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--reduce", default="gap", choices=list(REDUCTIONS), help="Scalar per channel")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--indep-test", default="fisherz", choices=list(INDEP_TESTS))
    parser.add_argument("--max-cond-set", type=int, help="Cap PC's conditioning-set size")
    parser.add_argument("--null-draws", type=int, default=200, help="Random assignments in the naming null")
    parser.add_argument("--floor", action="store_true", help="Also score an untrained twin of every run")
    parser.add_argument("--floor-seeds", type=int, default=1)
    parser.add_argument("--ceiling", action="store_true", help="Also score PC on the TRUE factors")
    parser.add_argument("--out", type=Path, help="Write the full JSON here")
    parser.add_argument("--self-test", action="store_true", help="Run the torch-free checks and exit")
    cli = parser.parse_args(argv)
    if cli.self_test:
        return _self_test()
    if not cli.run_dirs:
        parser.error("Supply --run-dirs")
    try:
        cli.pooling = tuple(int(x) for x in cli.pooling.split(","))
        if len(cli.pooling) != 3 or min(cli.pooling) < 1:
            raise ValueError
    except ValueError:
        parser.error("--pooling must be three positive integers, e.g. 4,4,4")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    import numpy as np

    results, ceiling_done = [], False
    plan = []
    for run in cli.run_dirs:
        plan.append((Path(run).resolve(), False, 0))
        if cli.floor:
            plan.extend((Path(run).resolve(), True, s) for s in range(cli.floor_seeds))
    for run_dir, random_init, seed in plan:
        label = run_dir.name + (f"-floor-s{seed}" if random_init else "")
        try:
            maps, z, adj = evaluate_run(run_dir, cli, random_init, seed)
        except Exception as exc:
            logger.exception("Failed on %s", label)
            results.append(dict(label=label, status="error", reason=f"{type(exc).__name__}: {exc}"))
            continue
        if maps is None:
            results.append(dict(label=label, **adj))
            continue
        reduced = reduce_channels(maps, cli.reduce)
        results.append(dict(evaluate_run_status="ok", **evaluate_reduced(reduced, z, adj, cli, label)))
        if cli.ceiling and not ceiling_done:
            ceiling_done = True
            # PC on the true factors, scored through the identity naming: the bound on what
            # this estimator reaches at this row count, before any representation.
            truth_rows = dict(evaluate_reduced(np.asarray(z), np.asarray(z), adj, cli, "truth-ceiling"))
            results.append(dict(evaluate_run_status="ok", **truth_rows))
    print("\n" + format_report([r for r in results if r.get("best") or r.get("status")]))
    if cli.out:
        cli.out.parent.mkdir(parents=True, exist_ok=True)
        cli.out.write_text(
            json.dumps(dict(protocol=dict(readout="none", naming="hungarian_abs_corr"), runs=results), indent=2) + "\n"
        )
        print(f"Saved JSON to {cli.out.resolve()}")
    return int(any(r.get("status") == "error" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
