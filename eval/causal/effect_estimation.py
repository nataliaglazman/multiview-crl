"""Cross-fitted estimation of a binary treatment's average effect."""

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

LEARNERS = ("linear", "rbf")
_NORMAL_975 = 1.959963984540054


def _binary_treatment(treatment):
    treatment = np.asarray(treatment, dtype=float)
    if treatment.ndim != 1 or not np.isin(treatment, (0, 1)).all():
        raise ValueError("treatment must be a vector of zeros and ones")
    treatment = treatment.astype(int)
    if np.bincount(treatment, minlength=2).min() < 2:
        raise ValueError("each treatment arm needs at least two subjects")
    return treatment


def _observations(treatment, outcome):
    treatment = _binary_treatment(treatment)
    outcome = np.asarray(outcome, dtype=float)
    if outcome.shape != treatment.shape or not np.isfinite(outcome).all():
        raise ValueError("outcome must contain one finite value per subject")
    return treatment, outcome


def make_folds(treatment, n_splits=5, seed=0):
    """Create reusable, treatment-stratified train and evaluation row indices."""
    treatment = _binary_treatment(treatment)
    if not 2 <= n_splits <= np.bincount(treatment).min():
        raise ValueError("n_splits must be between two and the smaller treatment arm's size")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return tuple(splitter.split(treatment, treatment))


def _fold_indices(treatment, folds):
    rows = np.arange(len(treatment))
    eval_count = np.zeros(len(treatment), dtype=int)
    checked = []
    for train, test in folds:
        train, test = np.asarray(train), np.asarray(test)
        if any(index.ndim != 1 or index.dtype.kind not in "iu" or not index.size for index in (train, test)):
            raise ValueError("fold indices must be nonempty integer vectors")
        if not np.array_equal(np.sort(np.concatenate((train, test))), rows):
            raise ValueError("each fold must partition the subject rows")
        if np.unique(treatment[train]).size != 2:
            raise ValueError("each training fold must include both treatment arms")
        eval_count[test] += 1
        checked.append((train, test))
    if not np.all(eval_count == 1):
        raise ValueError("the folds must evaluate each subject exactly once")
    return checked


def _fold_features(covariates, train, test, tabular_width, learner, seed):
    x_train = covariates[train]
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    if tabular_width < covariates.shape[1]:
        # A common scale preserves orthogonal rotations of the image code.
        scale[tabular_width:] = np.sqrt(np.mean(scale[tabular_width:] ** 2))
    scale[scale == 0] = 1.0
    x_train = (x_train - center) / scale
    x_test = (covariates[test] - center) / scale
    if learner == "rbf":
        basis = Nystroem(
            gamma=1.0 / covariates.shape[1],
            n_components=min(128, len(train)),
            random_state=seed,
        )
        x_train, x_test = (
            np.column_stack((x_train, basis.fit_transform(x_train))),
            np.column_stack((x_test, basis.transform(x_test))),
        )
    return x_train, x_test


def _effect_summary(estimate, se):
    return {
        "ate": float(estimate),
        "se": float(se),
        "ci_low": float(estimate - _NORMAL_975 * se),
        "ci_high": float(estimate + _NORMAL_975 * se),
    }


def unadjusted_ate(treatment, outcome):
    """Return the raw treated-minus-control mean and its standard error."""
    treatment, outcome = _observations(treatment, outcome)
    control, treated = outcome[treatment == 0], outcome[treatment == 1]
    se = np.sqrt(treated.var(ddof=1) / treated.size + control.var(ddof=1) / control.size)
    return _effect_summary(treated.mean() - control.mean(), se)


def _effective_n(weights):
    return float(weights.sum() ** 2 / np.square(weights).sum())


@threadpool_limits.wrap(limits=1)
def estimate_ate(treatment, outcome, covariates, folds, *, learner="linear", clip=0.025, seed=0, tabular_width=0):
    """Estimate the average treatment effect using held-out nuisance predictions.

    The leading `tabular_width` covariates get individual scales. The rest
    share one scale. Each fold fits preprocessing and the models for treatment
    probability and arm-specific outcome means on its training subjects.
    """
    treatment, outcome = _observations(treatment, outcome)
    covariates = np.asarray(covariates, dtype=float)
    if (
        covariates.ndim != 2
        or covariates.shape[0] != len(outcome)
        or not covariates.shape[1]
        or not np.isfinite(covariates).all()
    ):
        raise ValueError("covariates must be a finite subject-by-feature matrix")
    if not isinstance(tabular_width, (int, np.integer)) or not 0 <= tabular_width <= covariates.shape[1]:
        raise ValueError("tabular_width must be an integer between zero and the number of features")
    if learner not in LEARNERS:
        raise ValueError(f"learner must be one of {LEARNERS}")
    if not 0 < clip < 0.5:
        raise ValueError("clip must lie between zero and one half")

    propensity = np.empty_like(outcome)
    means = np.empty((2, len(outcome)))
    for fold, (train, test) in enumerate(_fold_indices(treatment, folds)):
        x_train, x_test = _fold_features(covariates, train, test, tabular_width, learner, seed + fold)
        a_train, y_train = treatment[train], outcome[train]
        propensity[test] = LogisticRegression(max_iter=1000).fit(x_train, a_train).predict_proba(x_test)[:, 1]
        for arm in (0, 1):
            mask = a_train == arm
            means[arm, test] = Ridge(alpha=1.0).fit(x_train[mask], y_train[mask]).predict(x_test)

    clipped = np.clip(propensity, clip, 1.0 - clip)
    treated, control = treatment == 1, treatment == 0
    scores = means[1] - means[0]
    scores[treated] += (outcome[treated] - means[1, treated]) / clipped[treated]
    scores[control] -= (outcome[control] - means[0, control]) / (1.0 - clipped[control])
    factual = means[treatment, np.arange(len(outcome))]
    return {
        **_effect_summary(scores.mean(), scores.std(ddof=1) / np.sqrt(scores.size)),
        "propensity_min": float(propensity.min()),
        "propensity_max": float(propensity.max()),
        "clipped_fraction": float(np.mean(clipped != propensity)),
        "ess_treated": _effective_n(1.0 / clipped[treated]),
        "ess_control": _effective_n(1.0 / (1.0 - clipped[control])),
        "outcome_rmse": float(np.sqrt(np.mean((outcome - factual) ** 2))),
        "propensity_brier": float(np.mean((treatment - propensity) ** 2)),
    }
