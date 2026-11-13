"""Small shared metrics: effective rank, CV probes (ridge / MLP R^2, logistic balanced accuracy), CKA.

The probes reuse ``eval/maps/phase0_probe.py`` (ridge and torch-MLP R^2, linear CKA) under subject-grouped
cross-validation, with inputs and targets standardised on each training fold.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from eval.maps.phase0_probe import linear_cka, mlp_r2_torch, ridge_r2

__all__ = ["effective_rank", "cv_r2", "cv_balanced_accuracy", "linear_cka"]


def effective_rank(x) -> float:
    """exp(entropy of the normalised singular values) of ``x [N, d]`` after centring."""
    x = torch.as_tensor(x, dtype=torch.float64)
    if x.shape[0] < 2:
        return float("nan")
    s = torch.linalg.svdvals(x - x.mean(0))
    p = s / s.sum().clamp_min(1e-12)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def _mlp_cfg(eval_cfg, device):
    return SimpleNamespace(
        mlp_hidden=tuple(eval_cfg.mlp_hidden),
        mlp_lr=1e-3,
        mlp_weight_decay=1e-4,
        mlp_epochs=eval_cfg.mlp_epochs,
        mlp_batch=4096,
        device=device,
    )


def cv_r2(x, y, groups, eval_cfg, probes=None, device="cpu", seed: int = 0) -> dict:
    """Mean R^2 over subject-grouped folds for each probe, and ``best`` = the larger of them.

    ``x [N, d]``, ``y [N]`` or ``[N, t]`` (uniform average over target columns), ``groups [N]`` subject ids.
    """
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    y = y[:, None] if y.ndim == 1 else y
    keep = y.std(0) > 1e-12
    if not keep.any():
        return dict(ridge=float("nan"), mlp=float("nan"), best=float("nan"))
    y = y[:, keep]
    probes = list(probes or eval_cfg.probes)
    folds = min(eval_cfg.cv_folds, len(np.unique(groups)))
    scores = {p: [] for p in probes}
    torch.manual_seed(seed)
    for tr, te in GroupKFold(n_splits=folds).split(x, y, groups):
        sx, sy = StandardScaler().fit(x[tr]), StandardScaler().fit(y[tr])
        xtr, xte, ytr, yte = sx.transform(x[tr]), sx.transform(x[te]), sy.transform(y[tr]), sy.transform(y[te])
        if "ridge" in probes:
            scores["ridge"].append(ridge_r2(xtr, ytr, xte, yte, eval_cfg.ridge_alpha))
        if "mlp" in probes:
            scores["mlp"].append(mlp_r2_torch(xtr, ytr, xte, yte, _mlp_cfg(eval_cfg, device)))
    out = {p: float(np.mean(v)) for p, v in scores.items()}
    for p in ("ridge", "mlp"):
        out.setdefault(p, float("nan"))
    out["best"] = float(np.nanmax([out["ridge"], out["mlp"]]))
    return out


def cv_balanced_accuracy(x, labels, groups, folds: int = 5) -> float:
    """Logistic regression, subject-grouped CV, balanced accuracy on the pooled held-out predictions."""
    x, labels = np.asarray(x, np.float64), np.asarray(labels)
    folds = min(folds, len(np.unique(groups)))
    pred = np.empty_like(labels)
    for tr, te in GroupKFold(n_splits=folds).split(x, labels, groups):
        sx = StandardScaler().fit(x[tr])
        if len(np.unique(labels[tr])) < 2:
            pred[te] = labels[tr][0]
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(max_iter=2000).fit(sx.transform(x[tr]), labels[tr])
        pred[te] = clf.predict(sx.transform(x[te]))
    return float(balanced_accuracy_score(labels, pred))
