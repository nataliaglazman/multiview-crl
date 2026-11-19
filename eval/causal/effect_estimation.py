"""Cross-fitted estimation of a binary treatment's average effect."""

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

LEARNERS = ("linear", "rbf")
_NORMAL_975 = 1.959963984540054
_SECOND_STAGE_ALPHAS = np.logspace(-2, 4, 13)


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


def _checked_covariates(covariates, n, tabular_width, learner, clip):
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim != 2 or covariates.shape[0] != n or not covariates.shape[1] or not np.isfinite(covariates).all():
        raise ValueError("covariates must be a finite subject-by-feature matrix")
    if not isinstance(tabular_width, (int, np.integer)) or not 0 <= tabular_width <= covariates.shape[1]:
        raise ValueError("tabular_width must be an integer between zero and the number of features")
    if learner not in LEARNERS:
        raise ValueError(f"learner must be one of {LEARNERS}")
    if not 0 < clip < 0.5:
        raise ValueError("clip must lie between zero and one half")
    return covariates


def _cross_fit(treatment, outcome, covariates, folds, tabular_width, learner, seed):
    """Held-out treatment probabilities, (n,), and each arm's outcome predictions, (2,) + outcome's shape, each fold's
    models fitted on the other folds; and each fold's rows and features, for a second stage."""
    propensity = np.empty(len(treatment))
    means = np.empty((2,) + outcome.shape)
    fitted = []
    for fold, (train, test) in enumerate(_fold_indices(treatment, folds)):
        x_train, x_test = _fold_features(covariates, train, test, tabular_width, learner, seed + fold)
        a_train, y_train = treatment[train], outcome[train]
        propensity[test] = LogisticRegression(max_iter=1000).fit(x_train, a_train).predict_proba(x_test)[:, 1]
        for arm in (0, 1):
            mask = a_train == arm
            predicted = Ridge(alpha=1.0).fit(x_train[mask], y_train[mask]).predict(x_test)
            means[arm, test] = predicted.reshape((len(test),) + outcome.shape[1:])
        fitted.append((train, test, x_train, x_test))
    return propensity, means, fitted


@threadpool_limits.wrap(limits=1)
def estimate_ate(treatment, outcome, covariates, folds, *, learner="linear", clip=0.025, seed=0, tabular_width=0):
    """Estimate the average treatment effect using held-out nuisance predictions.

    The leading `tabular_width` covariates get individual scales. The rest
    share one scale. Each fold fits preprocessing and the models for treatment
    probability and arm-specific outcome means on its training subjects.
    """
    treatment, outcome = _observations(treatment, outcome)
    covariates = _checked_covariates(covariates, len(outcome), tabular_width, learner, clip)
    propensity, means, _ = _cross_fit(treatment, outcome, covariates, folds, tabular_width, learner, seed)

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


@threadpool_limits.wrap(limits=1)
def estimate_cate(treatment, outcomes, covariates, folds, *, learner="linear", clip=0.025, seed=0, tabular_width=0):
    """Each subject's treatment effect, from models that never saw that subject.

    ``outcomes`` is (n,) or (n, m): outcomes that share one treatment, estimated together. The T-learner is the
    difference of the two arms' held-out outcome models, the same models ``estimate_ate`` uses. The DR-learner
    (Kennedy 2023) regresses each subject's AIPW score, the doubly robust pseudo-outcome, on the covariates,
    cross-fitted over the same folds with the outcome models' features; it stays consistent when either the treatment
    or the outcome model is right. Inverse-probability weights make those scores noisy, so the second stage's ridge
    penalty is chosen per outcome by leave-one-out on each fold's training subjects. Returns both effects and both
    arms' predicted outcomes, each shaped like ``outcomes``, and the treatment probabilities.
    """
    treatment = _binary_treatment(treatment)
    outcomes = np.asarray(outcomes, dtype=float)
    if outcomes.ndim not in (1, 2) or outcomes.shape[0] != len(treatment) or not np.isfinite(outcomes).all():
        raise ValueError("outcomes must hold one finite value per subject in each column")
    covariates = _checked_covariates(covariates, len(treatment), tabular_width, learner, clip)
    columns = outcomes.reshape(len(treatment), -1)
    propensity, means, fitted = _cross_fit(treatment, columns, covariates, folds, tabular_width, learner, seed)
    clipped = np.clip(propensity, clip, 1.0 - clip)[:, None]
    treated = (treatment == 1)[:, None]
    factual = np.where(treated, means[1], means[0])
    pseudo = means[1] - means[0] + np.where(treated, 1.0 / clipped, -1.0 / (1.0 - clipped)) * (columns - factual)
    dr = np.empty_like(pseudo)
    for train, test, x_train, x_test in fitted:
        second = RidgeCV(alphas=_SECOND_STAGE_ALPHAS, alpha_per_target=True).fit(x_train, pseudo[train])
        dr[test] = second.predict(x_test).reshape(len(test), -1)
    shape = outcomes.shape
    return {
        "t_learner": (means[1] - means[0]).reshape(shape),
        "dr_learner": dr.reshape(shape),
        "mu0": means[0].reshape(shape),
        "mu1": means[1].reshape(shape),
        "propensity": propensity,
    }


def _matrix(values, n, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] != n or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite subject-by-feature matrix")
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    return (values - values.mean(axis=0)) / scale


def canonical_pairs(z, w, ridge=1e-6):
    """Canonical correlation analysis of two standardized subject-by-feature matrices: each side's weights (one
    column per pair) and the canonical correlations, largest first."""

    def whitener(x):
        covariance = x.T @ x / len(x)
        values, vectors = np.linalg.eigh(covariance + ridge * np.eye(len(covariance)))
        return (vectors / np.sqrt(np.clip(values, ridge, None))) @ vectors.T

    left, right = whitener(z), whitener(w)
    u, s, vt = np.linalg.svd(left @ (z.T @ w / len(z)) @ right)
    k = len(s)
    return left @ u[:, :k], right @ vt[:k].T, s


def shared_directions(z, w, draws=50, seed=0):
    """How many canonical pairs of the proxies agree beyond chance: those whose correlation exceeds the 95th
    percentile of the largest canonical correlation with the subjects' pairing shuffled."""
    z, w = _matrix(z, len(z), "treatment_proxy"), _matrix(w, len(z), "outcome_proxy")
    _, _, observed = canonical_pairs(z, w)
    rng = np.random.default_rng(seed)
    null = [canonical_pairs(z, w[rng.permutation(len(w))])[2][0] for _ in range(draws)]
    return int((observed > np.percentile(null, 95)).sum())


@threadpool_limits.wrap(limits=1)
def proximal_ate(
    treatment, outcome, covariates, treatment_proxy, outcome_proxy, *, interactions=False, components=None
):
    """Estimate the average treatment effect when the confounder is seen only through two proxies.

    Proximal two-stage least squares (Miao et al. 2018; Tchetgen Tchetgen et al. 2020). Both proxies measure the
    unmeasured confounder and are independent of each other given it, so their measurement errors are independent.
    The outcome bridge is linear in the outcome-side proxy W and the covariates, and the treatment-side proxy Z
    instruments W: stage 1 fits W on the treatment, the covariates and Z; stage 2 fits the outcome on the treatment,
    the covariates and the fitted W. Adjusting for W directly would leave the confounding that W's measurement error
    hides.

    Proxies wider than the confounder leave the bridge underidentified, so both are first reduced to their shared
    directions: the leading `components` canonical pairs, or by default the pairs that agree beyond a shuffled-pairing
    null. Neither the treatment nor the outcome enters that choice. `interactions` adds treatment-by-W terms,
    instrumented by treatment-by-Z, for an effect that varies with the confounder. Standard errors are
    heteroskedasticity-robust (HC0), from the structural residuals, and ignore the choice of directions.
    """
    treatment, outcome = _observations(treatment, outcome)
    n = len(outcome)
    covariates = _matrix(covariates, n, "covariates")
    z = _matrix(treatment_proxy, n, "treatment_proxy")
    w = _matrix(outcome_proxy, n, "outcome_proxy")
    if not z.shape[1] or not w.shape[1]:
        raise ValueError("each proxy needs at least one feature")
    left, right, correlations = canonical_pairs(z, w)
    k = shared_directions(z, w) if components is None else components
    if not isinstance(k, (int, np.integer)) or not 0 <= k <= len(correlations):
        raise ValueError(f"components must be an integer between zero and {len(correlations)}")
    z, w = z @ left[:, :k], w @ right[:, :k]
    a = treatment[:, None].astype(float)
    exogenous = np.column_stack((np.ones(n), a, covariates))
    endogenous, instruments = (np.column_stack((w, a * w)), np.column_stack((z, a * z))) if interactions else (w, z)
    first = np.column_stack((exogenous, instruments))
    regressors = np.column_stack((exogenous, endogenous))
    fitted = first @ np.linalg.lstsq(first, regressors, rcond=None)[0]
    beta = np.linalg.lstsq(fitted, outcome, rcond=None)[0]
    residual = outcome - regressors @ beta
    bread = np.linalg.pinv(fitted.T @ fitted)
    covariance = bread @ ((fitted * residual[:, None] ** 2).T @ fitted) @ bread
    # W is centred, so the treatment coefficient is the effect averaged over subjects, with or without interactions.
    return {
        **_effect_summary(beta[1], np.sqrt(covariance[1, 1])),
        "components": int(k),
        "canonical_r": [float(r) for r in correlations[:k]],
    }
