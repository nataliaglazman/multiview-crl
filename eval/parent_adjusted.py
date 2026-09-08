"""Fold-local nonlinear parent adjustment for representation probes (numpy only)."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from eval.identifiability_metrics import _make_regressor
from eval.run_dci_compare import _auto_probe_dim


def _parent_model(seed):
    return HistGradientBoostingRegressor(
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


@threadpool_limits.wrap(limits=1)
def prepare_parent_folds(y, parents, seeds=(0, 1, 2), n_splits=5, inner_splits=3):
    """Prepare residual targets once, reusable across poolings and untrained twins.

    An outer test label is never used to fit its parent predictor. Training targets
    use inner out-of-fold parent predictions rather than in-sample residuals. The
    test residual uses a parent model fitted on the entire outer training fold.
    With no parents, residuals equal the original targets exactly.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    parents = np.asarray(parents, dtype=np.float64)
    if parents.ndim != 2 or len(parents) != len(y):
        raise ValueError("parents must have shape (samples, number_of_parents)")
    if not np.isfinite(y).all() or not np.isfinite(parents).all():
        raise ValueError("Parent adjustment requires finite factors.")
    if len(y) < n_splits * 4:
        raise ValueError(f"Parent adjustment requires at least {n_splits * 4} samples.")
    plans = []
    for seed in seeds:
        folds = []
        for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(y):
            if parents.shape[1]:
                parent = _parent_model(seed).fit(parents[tr], y[tr])
                pred_test = parent.predict(parents[te])
                pred_train = np.empty(len(tr))
                for fit, hold in KFold(inner_splits, shuffle=True, random_state=seed).split(tr):
                    inner = _parent_model(seed).fit(parents[tr[fit]], y[tr[fit]])
                    pred_train[hold] = inner.predict(parents[tr[hold]])
                parent_r2 = float(r2_score(y[te], pred_test))
            else:
                pred_train, pred_test, parent_r2 = 0.0, 0.0, float("nan")
            folds.append(
                dict(
                    train=tr,
                    test=te,
                    raw_train=y[tr],
                    raw_test=y[te],
                    residual_train=y[tr] - pred_train,
                    residual_test=y[te] - pred_test,
                    parent_r2=parent_r2,
                )
            )
        plans.append((seed, folds))
    return plans


@threadpool_limits.wrap(limits=1)
def prepare_probe_folds(X, plans, probe_dim="auto"):
    """Fit PCA (if requested) and scaling exclusively on each outer training fold."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not X.shape[1] or not np.isfinite(X).all():
        raise ValueError("Parent-adjusted probes require a nonempty finite representation matrix.")
    prepared = []
    for seed, folds in plans:
        features = []
        for fold in folds:
            tr, te = fold["train"], fold["test"]
            train, test = X[tr], X[te]
            width = _auto_probe_dim(len(tr), X.shape[1]) if probe_dim == "auto" else int(probe_dim or 0)
            if width > 0 and X.shape[1] > width:
                pca = PCA(n_components=min(width, len(tr)), random_state=seed).fit(train)
                train, test = pca.transform(train), pca.transform(test)
            scaler = StandardScaler().fit(train)
            features.append((scaler.transform(train), scaler.transform(test)))
        prepared.append(features)
    return prepared


@threadpool_limits.wrap(limits=1)
def score_parent_folds(features, plans, kind="ridge", n_null=3, null_seed=0):
    """Full and residual R² with identical folds/features; shuffle residuals within splits.

    Null permutations never move test targets into the probe's training targets.
    They are repeated deterministically across pooling rungs and floor draws. These
    are finite-sample decoding baselines, not a conditional-independence test.
    """
    full_seeds, residual_seeds, null_seeds, parent_seeds = [], [], [], []
    for (seed, folds), seed_features in zip(plans, features):
        full, residual, nulls, parent = [], [], [], []
        for fi, (fold, (train, test)) in enumerate(zip(folds, seed_features)):

            def predict_score(y_train, y_test):
                model = _make_regressor(kind, seed).fit(train, y_train)
                return float(r2_score(y_test, np.asarray(model.predict(test)).reshape(-1)))

            full.append(predict_score(fold["raw_train"], fold["raw_test"]))
            residual.append(predict_score(fold["residual_train"], fold["residual_test"]))
            parent.append(fold["parent_r2"])
            rng = np.random.RandomState((int(null_seed) + 1009 * int(seed) + 9176 * fi) % (2**32))
            nulls.extend(
                predict_score(rng.permutation(fold["residual_train"]), rng.permutation(fold["residual_test"]))
                for _ in range(n_null)
            )
        full_seeds.append(float(np.mean(full)))
        residual_seeds.append(float(np.mean(residual)))
        null_seeds.append(float(np.mean(nulls)) if nulls else float("nan"))
        parent_seeds.append(float(np.mean(parent)))
    real, null = float(np.mean(residual_seeds)), float(np.mean(null_seeds))
    return dict(
        full_r2_raw=float(np.mean(full_seeds)),
        r2_raw=real,
        r2_null=null,
        r2=real - null,
        r2_std=float(np.std(residual_seeds)),
        parent_r2=float(np.mean(parent_seeds)),
    )
