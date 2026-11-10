"""Skeleton recovery on held-out image factors and continuous tabular variables."""

from __future__ import annotations

import inspect

import numpy as np

from eval.causal.latent_causal_discovery import discover
from eval.causal.run_causal_recovery import INDEP_TESTS, skeleton_metrics


def validate_discovery_options(alpha, indep_test, max_cond_set):
    if not 0 < alpha < 1:
        raise ValueError("Discovery alpha must be between 0 and 1")
    if indep_test not in INDEP_TESTS:
        raise ValueError(f"Discovery independence test must be one of {INDEP_TESTS}")
    if max_cond_set is not None:
        if not isinstance(max_cond_set, (int, np.integer)) or max_cond_set < 0:
            raise ValueError("Discovery conditioning-set limit must be a nonnegative integer")
        from causallearn.search.ConstraintBased.PC import pc

        if "max_k" not in inspect.signature(pc).parameters:
            raise ValueError("Installed causal-learn does not support a conditioning-set limit. Upgrade causal-learn.")


def score_discovery(
    values,
    adjacency,
    *,
    alpha=0.05,
    indep_test="fisherz",
    max_cond_set=None,
    n_image_factors=None,
):
    """Score stable PC's undirected edges against the known graph.

    Use the same held-out subjects in each arm and the same node order as
    adjacency. When provided, the image factors must occupy the first columns.
    Fisher-Z tests partial correlation, which is approximate for non-Gaussian factors.
    """
    validate_discovery_options(alpha, indep_test, max_cond_set)
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Discovery requires a finite matrix with at least two columns")
    n_samples, n_nodes = values.shape
    truth = np.asarray(adjacency)
    if truth.shape != (n_nodes, n_nodes) or not np.isin(truth, [0, 1]).all() or np.any(np.diag(truth)):
        raise ValueError("Discovery adjacency must match the columns and contain binary edges without self-loops")
    truth = truth.astype(bool)
    if n_image_factors is not None and (
        not isinstance(n_image_factors, (int, np.integer)) or not 0 < n_image_factors < n_nodes
    ):
        raise ValueError("Discovery image-factor count must be between zero and the number of columns")
    if n_samples < 4:
        raise ValueError("Discovery requires at least four rows")

    spread = values.std(axis=0)
    keep = spread > np.finfo(float).eps
    n_variable = int(keep.sum())
    if n_variable >= 2 and indep_test == "fisherz":
        depth = n_variable - 2 if max_cond_set is None else min(max_cond_set, n_variable - 2)
        if n_samples < depth + 4:
            raise ValueError(f"Fisher-Z needs at least {depth + 4} rows at conditioning depth {depth}")
        variable = values[:, keep]
        standardized = (variable - variable.mean(axis=0)) / spread[keep]
        if n_samples > n_variable and np.linalg.matrix_rank(standardized) < n_variable:
            raise np.linalg.LinAlgError("Fisher-Z cannot score collinear discovery columns")

    estimated = np.zeros_like(truth, dtype=bool)
    if n_variable >= 2:
        try:
            estimated = discover(values, (alpha,), indep_test, max_cond_set)[alpha]
        except ValueError as error:
            if "correlation matrix is singular" in str(error):
                raise np.linalg.LinAlgError(str(error)) from error
            raise
    result = dict(
        **skeleton_metrics(estimated, truth),
        adjacency=estimated.astype(int).tolist(),
        dropped_constant_columns=np.flatnonzero(~keep).tolist(),
        alpha=float(alpha),
        indep_test=indep_test,
        max_cond_set=int(max_cond_set) if max_cond_set is not None else None,
        n_samples=n_samples,
    )
    if n_image_factors is not None:
        image = np.arange(n_nodes) < n_image_factors
        groups = {
            "within_image": image[:, None] & image,
            "image_tabular": image[:, None] ^ image,
            "within_tabular": (~image)[:, None] & ~image,
        }
        result["edge_groups"] = {
            group: skeleton_metrics(estimated & mask, truth & mask) for group, mask in groups.items()
        }
    return result
