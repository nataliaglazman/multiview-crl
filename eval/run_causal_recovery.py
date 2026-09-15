#!/usr/bin/env python
"""Batch the causal panel from analyze_synthetic_recovery.ipynb.

    python -m eval.run_causal_recovery --run-dirs results/run1 results/run2
    python -m eval.run_causal_recovery --runs-file runs.txt --output-dir results/causal_recovery

Each run needs settings.json and a VQVAE checkpoint. See CAUSAL_EVALUATION.md
for metric interpretation. The default reproduces the notebook's supervised,
in-sample graph readout and truth-selected alpha sweep, not directed recovery.

An F1 on its own says nothing, because both ends of the scale are already taken:
PC on a RANDOM projection of the same architecture recovers edges (--floor), and
PC on the TRUE factors does not recover all of them (--ceiling). Pass both and
read where a run sits between them, not its absolute score.
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
# PC's conditional-independence tests. fisherz is partial correlation, so it sees only the
# LINEAR part of a dependence; this generator's mechanisms are leaky_relu of a weighted
# parent sum, which fisherz is therefore misspecified for. kci is nonparametric and sees
# the nonlinear part, at roughly two orders of magnitude more compute.
INDEP_TESTS = ("fisherz", "kci")
# Suffixes appended to a run's directory name for its two reference rows. They are not real
# paths -- they exist so every downstream consumer (the summary table, the factor CSV,
# --reference-run matching) keeps the reference rows distinguishable from the run itself,
# which keys off Path(run_dir).name. Matching the -floor suffix run_dci_compare.py uses.
_FLOOR_SUFFIX = "-floor"
_CEILING_SUFFIX = "-ceiling"


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


def edge_type(graph, i, j):
    """The causal-learn edge between ``i`` and ``j``, as a name.

    causal-learn stores the endpoint at ``a`` of the edge ``a—b`` in ``graph[a, b]``:
    ``-1`` is a tail, ``1`` an arrowhead, ``0`` no edge.  So ``i -> j`` is
    ``graph[i, j] == -1`` with ``graph[j, i] == 1``, ``i — j`` is ``-1`` both ways and
    ``i <-> j`` is ``1`` both ways.
    """
    tail, head = int(graph[i][j]), int(graph[j][i])
    return {
        (0, 0): "none",
        (-1, -1): "undirected",
        (-1, 1): "forward",
        (1, -1): "backward",
        (1, 1): "bidirected",
    }.get((tail, head), "other")


def true_cpdag(adjacency):
    """The true DAG's CPDAG, in causal-learn's matrix encoding.

    PC identifies a Markov equivalence class, not a DAG, so this — not the DAG — is what
    an oriented estimate can be scored against.  On the default ``chain`` SCM the CPDAG is
    entirely undirected, and scoring orientations against the DAG instead would charge the
    estimate for two edges that no observational method can orient.
    """
    import numpy as np
    from causallearn.graph.Dag import Dag
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.DAG2CPDAG import dag2cpdag

    adjacency = np.asarray(adjacency, dtype=bool)
    nodes = [GraphNode(f"X{d + 1}") for d in range(len(adjacency))]
    dag = Dag(nodes)
    for i, j in zip(*np.where(adjacency)):
        dag.add_directed_edge(nodes[i], nodes[j])
    return np.asarray(dag2cpdag(dag).graph, dtype=int)


def orientation_metrics(estimated, truth_cpdag):
    """Score an estimated CPDAG against the true one, edge type by edge type.

    ``cpdag_shd`` counts node pairs whose edge type differs at all — a missing edge, an
    extra edge, and an edge oriented the wrong way each cost 1, so it is comparable to
    ``skeleton_shd`` but strictly harder.  The breakdown covers only pairs adjacent in
    BOTH graphs, which is where orientation is the question rather than detection:
    ``undirected_in_estimate`` is PC declining to orient an edge the truth's equivalence
    class does orient, which is a weaker failure than ``reversed``.
    """
    import numpy as np

    estimated = np.asarray(estimated, dtype=int)
    truth_cpdag = np.asarray(truth_cpdag, dtype=int)
    if estimated.shape != truth_cpdag.shape:
        raise ValueError("Estimated and true CPDAG must be equally sized square matrices")
    counts = dict(
        correct_directed=0,
        reversed=0,
        undirected_in_estimate=0,
        directed_in_estimate=0,
        bidirected_in_estimate=0,
        both_undirected=0,
    )
    mismatched = 0
    for i in range(len(truth_cpdag)):
        for j in range(i + 1, len(truth_cpdag)):
            est, true = edge_type(estimated, i, j), edge_type(truth_cpdag, i, j)
            mismatched += est != true
            if est == "none" or true == "none":
                continue
            if est == true:
                counts["correct_directed" if true in ("forward", "backward") else "both_undirected"] += 1
            elif {est, true} == {"forward", "backward"}:
                counts["reversed"] += 1
            elif est == "undirected":
                counts["undirected_in_estimate"] += 1
            elif est == "bidirected":
                counts["bidirected_in_estimate"] += 1
            else:
                counts["directed_in_estimate"] += 1
    return dict(counts, cpdag_shd=mismatched, cpdag_exact_match=(mismatched == 0))


def readout_width(n_samples, n_features, n_content, readout_dim=None):
    """PCA width for the supervised readout PC's decoded factors come out of.

    ``readout_dim=None`` reproduces the original rule: 64, floored at the factor count so
    the readout is never narrower than the graph it has to express, and capped at N/5 and
    at the block's own width so a small sample is not handed a near-singular basis.

    An explicit width pins it instead, which is what makes two models with different
    feature counts comparable — a 48-channel encoder block and an 18432-dim embedding
    otherwise get readouts of 48 and 64, and part of any difference in the recovered graph
    is that gap rather than the representation.  A width can only be lowered to what a
    block actually has, so match on the narrower of the two.
    """
    want = int(readout_dim) if readout_dim is not None else min(64, max(n_content, n_samples // 5))
    if want < 1:
        raise ValueError("Readout width must be at least 1")
    return min(want, n_features, n_samples)


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


def evaluate_arrays(
    X_content,
    z_content,
    adjacency,
    alphas=DEFAULT_ALPHAS,
    factor_rescue=False,
    diagnostic_alpha=0.05,
    orientation=False,
    readout_dim=None,
    holdout_readout=False,
    indep_test="fisherz",
    max_cond_set=None,
):
    """Evaluate already aligned features and factors with the supplied panel's protocol.

    ``orientation=True`` additionally scores each recovered graph's edge DIRECTIONS
    against the true DAG's CPDAG (see :func:`orientation_metrics`).  Off by default so
    existing reports are unchanged; the headline metrics stay skeleton-only either way.

    ``readout_dim`` pins the readout's PCA width (see :func:`readout_width`), so two models
    with different feature counts can be compared at equal readout capacity.

    ``holdout_readout=True`` fits the readout on a 70/30 train split and runs PC on the
    held-out rows only.  The default in-sample readout decodes the same rows it was fit on
    with the true labels, which is why the panel is documented as an optimistic diagnostic;
    this makes the graph a held-out result at the cost of 70% of the rows.

    ``indep_test`` selects PC's conditional-independence test: ``"fisherz"`` (partial
    correlation, linear-Gaussian) or ``"kci"`` (kernel-based, nonparametric).  This
    generator's mechanisms are ``leaky_relu`` of a weighted parent sum, so Fisher-Z is
    misspecified — it can only see the linear part of a dependence — while KCI can see the
    nonlinear part.  KCI's cost is the catch, and it is worse than a constant factor:
    measured here, 9 factors at 500 rows did not finish one alpha in 30 minutes, where
    Fisher-Z is instant.

    ``max_cond_set`` caps the size of PC's conditioning sets (its ``max_k``), which is what
    makes KCI tractable at that width — 9 factors at 300 rows went from not finishing to
    9.7 s at ``max_cond_set=2``, recovering the same edge count as ``1``.  It is an
    approximation: pairs that only separate on a larger conditioning set keep their edge,
    so the skeleton can only gain edges, never lose them.  All five arguments default to
    the original behaviour.
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from eval.causal_factor_diagnostics import factor_name

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
    if indep_test not in INDEP_TESTS:
        raise ValueError(f"indep_test must be one of {sorted(INDEP_TESTS)}, got {indep_test!r}")

    parents = [np.flatnonzero(adj[:, d]).tolist() for d in range(n_content)]
    residuals = z.copy()
    for d, pa in enumerate(parents):
        if pa:
            residuals[:, d] -= LinearRegression().fit(z[:, pa], z[:, d]).predict(z[:, pa])
    raw = [fit_probe(X, z[:, d : d + 1]) for d in range(n_content)]
    partial = [fit_probe(X, residuals[:, d : d + 1]) for d in range(n_content)]

    Xsc = StandardScaler().fit_transform(X)
    # Extra sample-count cap avoids invalid PCA for small datasets / many factors.
    n_pca = readout_width(len(Xsc), Xsc.shape[1], n_content, readout_dim)
    if Xsc.shape[1] > n_pca:
        Xsc = PCA(n_components=n_pca, random_state=0).fit_transform(Xsc)

    # One 70/30 split, shared by the held-out readout and by the PC R2 diagnostic that
    # tests whether the readout's compression loses a factor the full probe can decode.
    # Sharing it is what makes that diagnostic exactly the decoding quality of the columns
    # PC is handed under holdout_readout. Fit its scaler/PCA on TRAIN rows only.
    train, test = train_test_split(np.arange(len(X)), test_size=0.3, random_state=0)
    scaler = StandardScaler().fit(X[train])
    Xtr, Xte = scaler.transform(X[train]), scaler.transform(X[test])
    if X.shape[1] > n_pca:
        pca = PCA(n_components=min(n_pca, len(train)), random_state=0).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
    readout_r2, holdout_columns = [], []
    for d in range(n_content):
        model = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(Xtr, z[train, d])
        prediction = model.predict(Xte)
        readout_r2.append(float(r2_score(z[test, d], prediction)))
        holdout_columns.append(prediction)

    if holdout_readout:
        # PC now sees rows whose decoding never saw their own labels. A single split rather
        # than cross-fitting on purpose: a cross-fitted row's decoding depends on every
        # other row's label, and PC's Fisher-Z tests assume the rows are independent draws.
        z_hat, z_graph, readout_features = np.column_stack(holdout_columns), z[test], Xte.shape[1]
    else:
        z_hat = np.column_stack(
            [RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(Xsc, z[:, d]).predict(Xsc) for d in range(n_content)]
        )
        z_graph, readout_features = z, Xsc.shape[1]
    if len(z_hat) < 20:
        raise ValueError(f"Graph recovery needs at least 20 rows; the readout left {len(z_hat)}")
    if len(z_hat) < 20 * n_content:
        logger.warning(
            "PC is running on %d rows for %d factors; Fisher-Z is thin at that ratio. Raise the sample "
            "count -- with holdout_readout only 30%% of the rows reach PC.",
            len(z_hat),
            n_content,
        )

    truth = adj | adj.T
    np.fill_diagonal(truth, False)
    truth_cpdag = true_cpdag(adj) if orientation else None
    sweep = []
    best = None

    def recover(values, alpha):
        if np.any(np.std(values, axis=0) <= np.finfo(float).eps):
            raise ValueError("A decoded factor is constant; conditional-independence testing is undefined")
        cg = pc(values, alpha=alpha, indep_test=indep_test, show_progress=False, max_k=max_cond_set)
        graph = cg.G.graph
        estimated = (graph != 0) | (graph.T != 0)
        np.fill_diagonal(estimated, False)
        row = dict(alpha=alpha, **skeleton_metrics(estimated, truth), adjacency=estimated.astype(int).tolist())
        if truth_cpdag is not None:
            row["orientation"] = orientation_metrics(graph, truth_cpdag)
            row["cpdag"] = np.asarray(graph, dtype=int).tolist()
        return row

    passes = dict.fromkeys([*alphas, diagnostic_alpha])
    # n_content >= 5 so the warning fires on real factor counts rather than on small tests:
    # the blow-up is in the number of conditioning sets, which is what the width drives.
    if indep_test == "kci" and max_cond_set is None and n_content >= 5:
        # PC re-runs the whole search per alpha, and KCI's cost grows steeply in both the
        # row count and the conditioning-set size, so they multiply. Say so before spending
        # it rather than after: 9 factors at 500 rows did not finish one alpha in 30 min.
        logger.warning(
            "KCI on %d rows x %d factors, %d alpha(s), with no conditioning-set cap. Measured: 9 factors "
            "at 500 rows did not finish one alpha in 30 minutes. Pass max_cond_set=2 and a single alpha.",
            len(z_hat),
            n_content,
            len(passes),
        )
    # Always include the fixed diagnostic alpha, but select the headline best
    # only from the user's original sweep to preserve the existing metric.
    for alpha in passes:
        try:
            row = recover(z_hat, alpha)
        except (ValueError, np.linalg.LinAlgError) as exc:
            sweep.append(dict(alpha=alpha, error=str(exc)))
            continue
        sweep.append(row)
        # Preserve the notebook's tie-breaking: last alpha wins equal F1.
        if alpha in alphas and (best is None or row["f1"] >= best["f1"]):
            best = row

    factors = []
    for d in range(n_content):
        # z_graph, not z: under holdout_readout the decoded columns cover the test rows only.
        var = float(np.var(z_graph[:, d]))
        corr = float(np.corrcoef(z_graph[:, d], z_hat[:, d])[0, 1]) if var > 0 and np.std(z_hat[:, d]) > 0 else None
        factors.append(
            dict(
                dim=d,
                name=factor_name(d),
                parents=parents[d],
                raw_r2=raw[d],
                partial_r2=partial[d],
                gap=raw[d] - partial[d],
                readout_test_r2=readout_r2[d],
                decoded_variance_ratio=float(np.var(z_hat[:, d])) / var if var > 0 else None,
                decoded_gt_correlation=corr,
            )
        )
    result = dict(
        num_samples=len(X),
        num_features=X.shape[1],
        n_content=n_content,
        graph_readout_dim=readout_features,
        graph_samples=len(z_hat),
        readout_mode="holdout" if holdout_readout else "in_sample",
        indep_test=indep_test,
        max_cond_set=max_cond_set,
        raw_r2_mean=float(np.mean(raw)),
        partial_r2_mean=float(np.mean(partial)),
        factors=factors,
        true_dag=adj.astype(int).tolist(),
        true_skeleton=truth.astype(int).tolist(),
        best=best,
        alpha_sweep=sweep,
        graph_status="ok" if best else "unavailable",
    )
    if truth_cpdag is not None:
        result["true_cpdag"] = truth_cpdag.tolist()
    if factor_rescue:
        baseline = next((r for r in sweep if r["alpha"] == diagnostic_alpha and "f1" in r), None)
        repairs = []
        for d in range(n_content):
            repaired = z_hat.copy()
            repaired[:, d] = z_graph[:, d]
            try:
                metrics = recover(repaired, diagnostic_alpha)
                repairs.append(
                    dict(
                        dim=d,
                        **metrics,
                        f1_gain=metrics["f1"] - baseline["f1"] if baseline else None,
                        shd_reduction=baseline["skeleton_shd"] - metrics["skeleton_shd"] if baseline else None,
                    )
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                repairs.append(dict(dim=d, error=str(exc)))
        result["factor_rescue"] = dict(alpha=diagnostic_alpha, baseline=baseline, factors=repairs)
    return result


def truth_ceiling(z, adjacency, cli, cache):
    """PC run on the TRUE factors: the best skeleton this protocol can reach.

    An F1 of 0.77 means nothing on its own, because PC does not recover the whole skeleton
    even when the readout is perfect.  Fisher-Z sees only the linear part of a leaky_relu
    mechanism, a finite row count costs power, and ``--max-cond-set`` is an approximation.
    All three cost edges before any representation is involved.  Feeding ``z`` in as its own
    features prices them: whatever the ceiling misses is the protocol's, not the model's.

    Everything else is held at the settings the runs were scored under -- same alphas, same
    test, same conditioning cap, same ``holdout_readout`` split -- so the row count PC sees
    matches.  ``readout_dim`` is deliberately NOT passed: with 9 features and 9 factors the
    readout is already the identity, and pinning it wider would only re-add PCA.

    Returns ``(row, key)``.  ``row`` is ``None`` when this exact draw was already scored --
    two arms on one SCM share a ceiling, and one row for it is enough.
    """
    import hashlib

    import numpy as np

    z = np.ascontiguousarray(np.asarray(z, dtype=float))
    adjacency = np.ascontiguousarray(np.asarray(adjacency))
    digest = hashlib.sha256(z.tobytes() + adjacency.tobytes()).hexdigest()[:12]
    if digest in cache:
        return None, digest
    row = evaluate_arrays(
        z,
        z,
        adjacency,
        cli.alphas,
        False,  # factor_rescue is meaningless here: every factor is already truth
        cli.diagnostic_alpha,
        getattr(cli, "orientation", False),
        None,
        getattr(cli, "holdout_readout", False),
        getattr(cli, "indep_test", "fisherz"),
        getattr(cli, "max_cond_set", None),
    )
    # Marked only on success, so a run whose ceiling failed does not poison the digest and
    # suppress the retry on the next run that shares the draw.
    cache[digest] = True
    return row, digest


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


def evaluate_run(run_dir, cli, random_init=False, init_seed=0, ceiling_cache=None):
    """Load each run independently so SCM, renderer and normalization match it.

    ``random_init=True`` skips the checkpoint and scores this run's exact architecture
    untrained -- the floor every trained number has to be read as a gap over.  ``init_seed``
    fixes the draw: those weights ARE the measurement, and unseeded they would be a
    different random projection on every invocation, with the noise landing straight in the
    reported gap.  The checkpoint is still required to exist, so a floor can only be
    produced for a run that actually has a trained twin to be compared against.
    """
    with (run_dir / "settings.json").open() as f:
        settings = json.load(f)
    if not settings.get("synthetic_causal", False):
        return dict(status="skipped", reason="settings['synthetic_causal'] is False")

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    checkpoint = run_dir / cli.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    logger.info("Evaluating %s%s", run_dir, f" [UNTRAINED FLOOR, seed {init_seed}]" if random_init else "")
    model, args, device = load_model_from_run_dir(
        str(run_dir), str(checkpoint), device=cli.device, random_init=random_init, seed=init_seed
    )
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
    result = evaluate_arrays(
        X,
        z,
        scm["adj"],
        cli.alphas,
        cli.factor_rescue,
        cli.diagnostic_alpha,
        getattr(cli, "orientation", False),
        getattr(cli, "readout_dim", None),
        getattr(cli, "holdout_readout", False),
        getattr(cli, "indep_test", "fisherz"),
        getattr(cli, "max_cond_set", None),
    )
    result.update(
        status="ok" if result["best"] else "partial",
        checkpoint=None if random_init else str(checkpoint),
        level=level,
        pooling=cli.pooling,
        role="floor" if random_init else "trained",
        random_init=random_init,
        init_seed=init_seed,
        causal_settings={key: value for key, value in settings.items() if key.startswith("synthetic_")},
    )
    # The ceiling depends only on the factor draw and the SCM, so it is the same for a run
    # and its floor twin; scoring it once off the trained pass is enough.
    if getattr(cli, "ceiling", False) and not random_init:
        # Guarded: the ceiling is a reference row, so losing it must not also lose the run's
        # own scores, which are the expensive half and are already complete by here.
        try:
            ceiling, key = truth_ceiling(z, scm["adj"], cli, ceiling_cache if ceiling_cache is not None else {})
        except Exception as exc:
            logger.exception("Truth ceiling failed for %s; keeping the run's own scores", run_dir)
            result["ceiling_error"] = f"{type(exc).__name__}: {exc}"
            return result
        result["ceiling_key"] = key
        if ceiling is not None:
            ceiling.update(
                status="ok" if ceiling["best"] else "partial",
                checkpoint=None,
                level=level,
                pooling=cli.pooling,
                role="ceiling",
                ceiling_key=key,
                causal_settings=result["causal_settings"],
            )
            result["ceiling"] = ceiling
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


def build_plan(runs, floor=False, floor_seeds=1):
    """``(label, run_dir, random_init, init_seed)`` per evaluation, in execution order.

    Each run is followed immediately by its own untrained twins rather than all runs first,
    so an evaluation that dies partway still leaves every scored run next to the floor it
    has to be read against -- a run with no floor is uninterpretable, and the rows are
    written out after every evaluation.
    """
    plan = []
    for run in runs:
        plan.append((str(run), run, False, 0))
        if floor:
            plan.extend((f"{run}{_FLOOR_SUFFIX}-s{seed}", run, True, seed) for seed in range(floor_seeds))
    return plan


def print_result(result):
    """Console block for one scored row. Shared so a floor or ceiling row prints alike."""
    if not result.get("factors"):
        print(f"  {result['status']}: {result['reason']}")
        return
    best = result["best"]
    print(f"  L{result['level']} partial R²={result['partial_r2_mean']:.3f} (raw={result['raw_r2_mean']:.3f})")
    print(
        f"  Readout: {result['readout_mode']} at {result['graph_readout_dim']} dims, "
        f"PC on {result['graph_samples']} rows with {result['indep_test']}"
    )
    if best:
        print(
            f"  Best skeleton F1={best['f1']:.3f} P={best['precision']:.3f} R={best['recall']:.3f} "
            f"alpha={best['alpha']:g} SHD={best['skeleton_shd']} exact_match={best['exact_match']}"
        )
        if "orientation" in best:
            o = best["orientation"]
            print(
                f"  Orientation vs the true CPDAG: correct={o['correct_directed']} "
                f"reversed={o['reversed']} unoriented={o['undirected_in_estimate']} "
                f"CPDAG SHD={o['cpdag_shd']} exact_match={o['cpdag_exact_match']}"
            )
    else:
        print("  PC unavailable; factor scores retained. See alpha_sweep errors in JSON.")
    for factor in result["factors"]:
        print(
            f"    d{factor['dim']} pa={factor['parents']}: raw={factor['raw_r2']:.3f} "
            f"partial={factor['partial_r2']:.3f} gap={factor['gap']:+.3f}"
        )


def _reference_rows(results):
    """``{trained run_dir: (its floor rows, its ceiling row)}`` for rows that have them.

    Floors are matched by the suffix their run_dir carries, ceilings by the draw digest both
    sides recorded -- not by position, so a partial evaluation still pairs correctly.
    """
    scored = [r for r in results if r.get("best")]
    ceilings = {r["ceiling_key"]: r for r in scored if r.get("role") == "ceiling" and r.get("ceiling_key")}
    pairs = {}
    for run in scored:
        if run.get("role", "trained") != "trained":
            continue
        prefix = run["run_dir"] + _FLOOR_SUFFIX
        floors = [r for r in scored if r.get("role") == "floor" and r["run_dir"].startswith(prefix)]
        ceiling = ceilings.get(run.get("ceiling_key"))
        if floors or ceiling:
            pairs[run["run_dir"]] = (floors, ceiling)
    return pairs


def format_floor_block(results):
    """Trained minus its untrained twin, with the truth ceiling beside it.

    The absolute F1 is not a statement about the model. PC on a RANDOM projection of the
    same architecture already recovers edges, and PC on the true factors does not recover
    all of them, so the interpretable quantity is where a run sits between those two. This
    block is empty unless --floor or --ceiling was passed.
    """
    import numpy as np

    pairs = _reference_rows(results)
    if not pairs:
        return ""
    by_dir = {r["run_dir"]: r for r in results}
    seeds = max((len(floors) for floors, _ in pairs.values()), default=0)
    head = ["Run", "F1", "floor", "learned"]
    if seeds > 1:
        head.append("fl.rng")
    head += ["ceiling", "Partial R²", "floor", "learned"]
    rows = [head]
    for run_dir, (floors, ceiling) in pairs.items():
        run = by_dir[run_dir]
        f1 = run["best"]["f1"]
        pr = run["partial_r2_mean"]
        f1_floors = [f["best"]["f1"] for f in floors]
        pr_floors = [f["partial_r2_mean"] for f in floors]
        cells = [Path(run_dir).name, f"{f1:.3f}"]
        cells += [f"{np.mean(f1_floors):.3f}", f"{f1 - np.mean(f1_floors):+.3f}"] if f1_floors else ["—", "—"]
        if seeds > 1:
            cells.append(f"{max(f1_floors) - min(f1_floors):.3f}" if len(f1_floors) > 1 else "—")
        cells.append(f"{ceiling['best']['f1']:.3f}" if ceiling else "—")
        cells.append(f"{pr:.3f}")
        cells += [f"{np.mean(pr_floors):.3f}", f"{pr - np.mean(pr_floors):+.3f}"] if pr_floors else ["—", "—"]
        rows.append(cells)
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

    def line(row):
        return "  ".join(
            value.ljust(width) if i == 0 else value.rjust(width) for i, (value, width) in enumerate(zip(row, widths))
        ).rstrip()

    out = ["\nFloor-subtracted, at the best-F1 alpha:", line(rows[0]), "  ".join("-" * w for w in widths)]
    out.extend(line(row) for row in rows[1:])
    if seeds:
        out.append(
            f"\nfloor = an UNTRAINED twin of the same architecture ({seeds} init seed"
            f"{'s' if seeds > 1 else ''}), everything else held fixed."
        )
        if seeds == 1:
            out.append("One seed is one draw of a random projection; --floor-seeds 3 bounds how much it moves.")
    if any(ceiling for _, ceiling in pairs.values()):
        out.append("ceiling = PC on the TRUE factors, same row count, test and alphas. What IT misses is what")
        out.append("the protocol costs (a linear test, finite rows, --max-cond-set), before any model.")
    return "\n".join(out) + "\n"


def format_summary(results):
    """One row per run, with failed/skipped metrics clearly marked unavailable."""
    names = [Path(result["run_dir"]).name for result in results]
    rows = [["Directory", "F1", "Precision", "Recall", "SHD", "Partial R²", "Status"]]
    for result, name in zip(results, names):
        # Keep runs with identical basenames distinguishable in the table.
        label = result["run_dir"] if names.count(name) > 1 else name
        if result.get("best"):
            best = result["best"]
            metrics = [f"{best[key]:.3f}" for key in ("f1", "precision", "recall")]
            metrics.extend([str(best["skeleton_shd"]), f"{result['partial_r2_mean']:.3f}"])
        else:
            metrics = ["—"] * 5
            if "partial_r2_mean" in result:
                metrics[-1] = f"{result['partial_r2_mean']:.3f}"
        rows.append([label, *metrics, result["status"]])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

    def line(row):
        return "  ".join(
            value.ljust(width) if i in (0, 6) else value.rjust(width)
            for i, (value, width) in enumerate(zip(row, widths))
        ).rstrip()

    table = [line(rows[0]), "  ".join("-" * width for width in widths)]
    table.extend(line(row) for row in rows[1:])
    table.append("\nF1/precision/recall/SHD use the best-F1 alpha; SHD counts missing + extra skeleton edges.")
    table.append("Partial R² is the mean across content factors. — = unavailable (see JSON/CSV for reasons).")
    return "\n".join(table) + "\n" + format_floor_block(results)


def write_reports(results, output_dir, reference_run=None, diagnostic_alpha=0.05, protocol=None):
    from eval.causal_factor_diagnostics import write_factor_reports

    output_dir.mkdir(parents=True, exist_ok=True)
    # The readout mode is a property of how the runs were scored, so read it off them
    # rather than restating a default that --holdout-readout would silently contradict.
    mode = next((r.get("readout_mode") for r in results if r.get("readout_mode")), "in_sample")
    test = next((r.get("indep_test") for r in results if r.get("indep_test")), "fisherz")
    payload = dict(
        protocol=protocol
        or dict(
            graph_target="undirected_skeleton",
            alpha_selection="best_f1_against_truth_last_tie",
            indep_test=test,
            max_cond_set=next((r.get("max_cond_set") for r in results if r.get("max_cond_set")), None),
            graph_readout=f"supervised_{mode}_ridgecv",
            parent_adjustment="linear_all_samples",
            probe="ridge_alpha1_70_30_split_seed0",
            content_mask="sample0_fixed",
            empty_graph_f1=0.0,
            # Which reference rows this payload carries, so a reader of the JSON alone can
            # tell a missing floor from a floor of zero.
            floor_seeds=sorted({r["init_seed"] for r in results if r.get("role") == "floor"}),
            has_truth_ceiling=any(r.get("role") == "ceiling" for r in results),
        ),
        runs=results,
    )
    (output_dir / "causal_recovery.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (output_dir / "causal_recovery_summary.txt").write_text(format_summary(results))
    columns = [
        "directory_name",
        "run_dir",
        "role",
        "init_seed",
        "status",
        "reason",
        "level",
        "pooling",
        "num_samples",
        "num_features",
        "indep_test",
        "max_cond_set",
        "readout_mode",
        "graph_readout_dim",
        "graph_samples",
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
            writer.writerow({**result, **(result.get("best") or {}), "directory_name": Path(result["run_dir"]).name})
    return write_factor_reports(results, output_dir, reference_run, diagnostic_alpha)


def _self_test():
    """Checks for the reference-row plumbing, which needs neither torch nor causal-learn.

    The scoring itself cannot run here (PC needs causal-learn, the encoder needs torch), so
    what is checked is the part that silently produces a WRONG NUMBER rather than an error:
    which floor row gets subtracted from which run.
    """

    def row(name, role="trained", f1=0.5, partial=0.3, key=None, seed=0):
        return dict(
            run_dir=name,
            role=role,
            init_seed=seed,
            status="ok",
            partial_r2_mean=partial,
            best=dict(f1=f1, precision=f1, recall=f1, skeleton_shd=4, exact_match=False, alpha=0.05),
            **({"ceiling_key": key} if key else {}),
        )

    failures = []

    def check(label, condition):
        if not condition:
            failures.append(label)
        print(f"  {'ok  ' if condition else 'FAIL'}  {label}")

    runs = [Path("/r/arm_a"), Path("/r/arm_b")]
    check("without --floor the plan is one pass per run", build_plan(runs) == [(str(r), r, False, 0) for r in runs])
    planned = build_plan(runs, floor=True, floor_seeds=3)
    check("--floor adds one untrained pass per seed", len(planned) == 2 * (1 + 3))
    check("each run is followed by its own floors", [p[1] for p in planned[:4]] == [runs[0]] * 4)
    check("floor passes carry distinct seeds", [p[3] for p in planned[1:4]] == [0, 1, 2])
    check("only floor passes set random_init", [p[2] for p in planned[:4]] == [False, True, True, True])
    check(
        "labels stay unique so the rows do not collide",
        len({p[0] for p in planned}) == len(planned) and planned[1][0] == f"/r/arm_a{_FLOOR_SUFFIX}-s0",
    )

    a = row("/r/arm_a", f1=0.688, partial=0.384, key="d0")
    b = row("/r/arm_b", f1=0.765, partial=0.005, key="d0")
    floors_a = [row(f"/r/arm_a{_FLOOR_SUFFIX}-s{s}", "floor", f1=0.40 + 0.05 * s, partial=0.01, seed=s) for s in (0, 1)]
    ceiling = row(f"/r/arm_a{_CEILING_SUFFIX}", "ceiling", f1=0.812, partial=1.0, key="d0")

    pairs = _reference_rows([a, *floors_a, b, ceiling])
    check(
        "floors attach to their own run only",
        [r["run_dir"] for r in pairs["/r/arm_a"][0]] == [f["run_dir"] for f in floors_a],
    )
    check("a run without floors gets none", pairs["/r/arm_b"][0] == [])
    check("both arms share one ceiling via the draw digest", pairs["/r/arm_a"][1] is pairs["/r/arm_b"][1] is ceiling)

    # A ceiling whose digest no arm recorded must not be silently attached to an arm.
    stray = _reference_rows([a, row("/r/other-ceiling", "ceiling", key="d9")])
    check("a ceiling from another SCM draw is not borrowed", stray == {})

    # A directory that genuinely ends in the floor suffix is a run, not arm_a's twin.
    impostor = row(f"/r/arm_a{_FLOOR_SUFFIX}-s0", f1=0.9)
    check("a real run named like a floor is not absorbed", _reference_rows([a, impostor]) == {})

    block = format_floor_block([a, *floors_a, b, ceiling])
    check("learned F1 is trained minus the floor MEAN", "+0.263" in block)  # 0.688 - 0.425
    check("learned partial R² is reported too", "+0.374" in block)  # 0.384 - 0.010
    check("the across-seed spread shows once there is more than one", "fl.rng" in block and "0.050" in block)
    check("the ceiling appears for both arms", block.count("0.812") == 2)
    # arm_b has a ceiling but no floor: the five floor-derived cells (floor/learned/fl.rng
    # for F1, floor/learned for partial R²) must read as missing, never as a floor of zero,
    # which would silently turn its absolute F1 into its "learned" F1.
    no_floor = next(line for line in block.splitlines() if line.startswith("arm_b "))
    check("an arm with no floor prints dashes, not zeros", no_floor.count("—") == 5)
    check("its own and its ceiling's scores still print", all(v in no_floor for v in ("0.765", "0.812", "0.005")))

    one = format_floor_block([a, floors_a[0], ceiling])
    check("one seed hides the spread column", "fl.rng" not in one)
    check("one seed says so", "--floor-seeds 3" in one)
    check("ceiling alone still renders", "ceiling" in format_floor_block([a, ceiling]))
    check("no reference rows means no block", format_floor_block([a, b]) == "")

    scored = dict(
        level=0,
        raw_r2_mean=0.4,
        readout_mode="holdout",
        graph_readout_dim=128,
        graph_samples=600,
        indep_test="fisherz",
        factors=[dict(dim=0, parents=[], raw_r2=0.1, partial_r2=0.1, gap=0.0)],
    )
    print_result(dict(status="skipped", reason="settings['synthetic_causal'] is False"))
    print_result({**a, **scored, "best": None})
    check("print_result survives rows with no graph", True)

    print("\n" + ("FAILED: " + ", ".join(failures) if failures else "All checks passed."))
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs", nargs="+", default=[], help="Run directories or quoted glob patterns")
    parser.add_argument("--runs-file", help="Text file with one directory/glob per line, relative to that file")
    parser.add_argument(
        "--from-json", type=Path, help="Regenerate outputs from saved causal_recovery.json without evaluation"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Default: results/causal_recovery, or the input JSON directory with --from-json"
    )
    parser.add_argument("--checkpoint", default="vqvae_model.pt", help="Checkpoint filename inside each run")
    parser.add_argument("--level", type=int, help="Default: first content_style_levels entry, otherwise 0")
    parser.add_argument("--pooling", default="4,4,4", help="gap or a 3D patch grid (default: 4,4,4, as in notebook)")
    parser.add_argument("--num-samples", type=int, help="Default: each run's synthetic_num_test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", help="cpu, cuda, cuda:0, etc.; default: CUDA when available, else CPU")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument(
        "--reference-run", help="Reference directory path or unique basename; default: first scored run"
    )
    parser.add_argument("--diagnostic-alpha", type=float, default=0.05, help="Fixed PC alpha for factor comparisons")
    parser.add_argument(
        "--factor-rescue", action="store_true", help="Rerun PC replacing each decoded factor with truth"
    )
    parser.add_argument(
        "--orientation",
        action="store_true",
        help="Also score edge DIRECTIONS against the true DAG's CPDAG. The headline metrics stay "
        "skeleton-only; this adds a per-alpha breakdown to the JSON and one console line.",
    )
    parser.add_argument(
        "--indep-test",
        default="fisherz",
        choices=list(INDEP_TESTS),
        help="PC's conditional-independence test. 'fisherz' (default) is partial correlation and sees "
        "only the LINEAR part of a dependence, which is misspecified for this generator's leaky_relu "
        "mechanisms. 'kci' is nonparametric and sees the nonlinear part, at ~100x the cost and growing "
        "steeply with the sample count -- pair it with a single --alphas value.",
    )
    parser.add_argument(
        "--max-cond-set",
        type=int,
        help="Cap PC's conditioning-set size (its max_k). Unbounded by default. This is what makes "
        "--indep-test kci tractable: measured, 9 factors at 300 rows went from not finishing to 9.7s at "
        "2, recovering the same edges as 1. An approximation -- pairs that only separate on a larger "
        "conditioning set keep their edge, so the skeleton can gain edges but never lose them.",
    )
    parser.add_argument(
        "--readout-dim",
        type=int,
        help="PCA width for the supervised readout PC's decoded factors come from. Default: the "
        "min(64, max(n_content, N/5)) rule. Pin it to compare models whose blocks differ in width; "
        "a block narrower than the request keeps its own width, and the effective value is reported "
        "as graph_readout_dim.",
    )
    parser.add_argument(
        "--holdout-readout",
        action="store_true",
        help="Fit the readout on a 70/30 train split and run PC on the held-out rows only, instead of "
        "decoding the same rows the readout was fit on. Removes the panel's in-sample optimism at the "
        "cost of 70%% of the rows, so pair it with --num-samples 2000 or more.",
    )
    parser.add_argument(
        "--floor",
        action="store_true",
        help="Also score an UNTRAINED twin of every run (row '<name>-floor-s<seed>'), same "
        "architecture, pooling, readout width and alphas. PC on a random projection of this "
        "architecture already recovers edges, so an absolute F1 without this is not a statement "
        "about what the model learned. Roughly doubles runtime per floor seed.",
    )
    parser.add_argument(
        "--floor-seeds",
        type=int,
        default=1,
        help="Init seeds to draw the untrained twin with (default 1). One seed is ONE draw of a "
        "random projection and carries sampling noise straight into every floor-subtracted number; "
        "3 makes the spread visible as its own column. Ignored without --floor.",
    )
    parser.add_argument(
        "--ceiling",
        action="store_true",
        help="Also score PC on the TRUE factors (row '<name>-ceiling'), at the same row count, test, "
        "alphas and conditioning cap. This is the upper bound of the protocol, not of a model: what it "
        "misses is what Fisher-Z's linear test, the finite row count and --max-cond-set cost before any "
        "representation is involved. Cheap under fisherz; one extra PC sweep per distinct SCM draw.",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Run the built-in checks of the torch-free paths and exit"
    )
    cli = parser.parse_args(argv)
    if cli.self_test:
        return _self_test()
    if cli.floor_seeds < 1:
        parser.error("--floor-seeds must be at least 1")
    if not 0 < cli.diagnostic_alpha < 1:
        parser.error("--diagnostic-alpha must be strictly between 0 and 1")
    if cli.from_json:
        if cli.run_dirs or cli.runs_file:
            parser.error("Use --from-json on its own, without --run-dirs or --runs-file")
        if cli.factor_rescue:
            parser.error("--factor-rescue needs a fresh evaluation; decoded samples are not stored in JSON")
        with cli.from_json.open() as f:
            payload = json.load(f)
            results = payload["runs"]
        output_dir = cli.output_dir or cli.from_json.parent
        factor_report = write_reports(
            results, output_dir, cli.reference_run, cli.diagnostic_alpha, payload.get("protocol")
        )
        print(format_summary(results))
        print(factor_report)
        print(f"Saved CSV, JSON, summary and factor tables to {output_dir.resolve()}")
        return int(any(result["status"] in ("error", "partial") for result in results))
    cli.output_dir = cli.output_dir or Path("results/causal_recovery")
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
    plan = build_plan(runs, cli.floor, cli.floor_seeds)
    results, ceiling_cache = [], {}
    for i, (label, run, random_init, init_seed) in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] {label}", flush=True)
        try:
            result = evaluate_run(run, cli, random_init=random_init, init_seed=init_seed, ceiling_cache=ceiling_cache)
        except Exception as exc:
            logger.exception("Failed to evaluate %s", label)
            result = dict(status="error", reason=f"{type(exc).__name__}: {exc}")
        result["run_dir"] = label
        # A skipped or errored run never reached evaluate_run's update, so it carries no role.
        result.setdefault("role", "floor" if random_init else "trained")
        result.setdefault("init_seed", init_seed)
        ceiling = result.pop("ceiling", None)
        results.append(result)
        print_result(result)
        if ceiling is not None:
            ceiling["run_dir"] = f"{run}{_CEILING_SUFFIX}"
            results.append(ceiling)
            print(f"\n[{i}/{len(plan)}] {ceiling['run_dir']}  (PC on the true factors)", flush=True)
            print_result(ceiling)
        # Persist after every run so a later failure doesn't discard completed work.
        # The requested reference may be a later run; use the default until it arrives.
        reference_ready = any(
            r.get("factors") and cli.reference_run in (r["run_dir"], Path(r["run_dir"]).name) for r in results
        )
        write_reports(results, cli.output_dir, cli.reference_run if reference_ready else None, cli.diagnostic_alpha)
    print("\n" + format_summary(results))
    print(write_reports(results, cli.output_dir, cli.reference_run, cli.diagnostic_alpha))
    print(f"Saved CSV, JSON, summary and factor tables to {cli.output_dir.resolve()}")
    print("F1 measures the skeleton only, and alpha is selected against truth.")
    if not cli.holdout_readout:
        print("The graph readout is in-sample; --holdout-readout removes that optimism.")
    if not cli.floor:
        print(
            "NO FLOOR MEASURED. PC on an untrained twin of this architecture already recovers edges, "
            "so the F1 above is not on its own a statement about what the model learned. Re-run with "
            "--floor (and --ceiling for the other end) before reporting it."
        )
    return int(any(result["status"] in ("error", "partial") for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
