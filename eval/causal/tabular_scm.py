"""Known tabular causal models on the rows of a fixed image cohort."""

from dataclasses import dataclass

import numpy as np

TABULAR_NAMES = ("tabular_1", "tabular_2", "tabular_3")
CAUSE_NAMES = ("tabular_cause_1", "tabular_cause_2")


@dataclass(frozen=True)
class TabularSCM:
    """Observed data, counterfactual truth and a continuous discovery companion.

    The adjacency matrices are parent by child. The effect graph's node order is
    image factors, tabular covariates, treatment, outcome. The discovery graph
    adds any tabular causes of image factors after the covariates.
    """

    covariates: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    propensity: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    effect_adjacency: np.ndarray
    discovery_treatment: np.ndarray
    discovery_outcome: np.ndarray
    adjacency: np.ndarray
    causes: np.ndarray
    cause_targets: tuple
    cause_strengths: tuple
    tabular_names = TABULAR_NAMES

    @property
    def cause_names(self):
        return CAUSE_NAMES[: self.causes.shape[1]]

    @property
    def ate(self):
        """Average treatment effect across these subjects."""
        return float(np.mean(self.y1 - self.y0))


def _base_adjacency(width, image_adjacency, n_causes=0):
    adjacency = np.zeros((width + len(TABULAR_NAMES) + n_causes + 2,) * 2, dtype=bool)
    if image_adjacency is not None:
        image_adjacency = np.asarray(image_adjacency)
        if image_adjacency.shape != (width, width) or not np.isin(image_adjacency, (0, 1)).all():
            raise ValueError("image_adjacency must be a binary square matrix matching the image factor columns")
        reachable = image_adjacency.astype(bool)
        for node in range(width):
            reachable |= reachable[:, [node]] & reachable[[node], :]
        if np.diag(reachable).any():
            raise ValueError("image_adjacency must be acyclic")
        adjacency[:width, :width] = image_adjacency
    adjacency[width, width + 1] = True
    adjacency[width + 2, -1] = True
    return adjacency


def simulate_tabular(
    image_factors,
    *,
    seed=0,
    effect=1.0,
    heterogeneity=0.0,
    confounding=1.0,
    overlap=0.1,
    image_adjacency=None,
    own_noise=None,
):
    """Attach synthetic variables without changing the order or number of subjects.

    Each standardized image factor contributes an equal-magnitude seeded weight
    to a unit-variance score. A second score combines the first two tabular
    covariates, which have a direct edge between them. The scores have the same
    weight in treatment assignment and baseline outcome. The third tabular
    covariate affects outcome alone. Assignment probabilities lie between
    ``overlap`` and ``1 - overlap``. The image score can modify the effect.

    The discovery companion uses continuous treatment, independent Gaussian
    errors and a constant effect. One seeded image factor and the first tabular
    covariate drive exposure. A different image factor, the second tabular
    covariate and the third drive outcome. Each edge has a seeded weight. No
    variable is a direct parent of both exposure and outcome. A shared parent's
    edge into exposure nearly cancels once PC conditions on outcome, and with
    the effect task's equations it cancels exactly. Supply the known image
    graph on exactly the input columns to score graph recovery.

    ``own_noise`` holds each image factor's exogenous noise in the image SCM,
    independent of that factor's parents. With it, the discovery companion adds
    two tabular causes of distinct seeded image factors. Each is a noisy copy of
    its factor's standardized own noise, ``rho * u + sqrt(1 - rho**2) * e``.
    Under Gaussian noise this is exactly the joint law of a cause that enters
    the factor's equation linearly. It is independent of every factor that does
    not descend from its target. The images stay fixed, so a tabular cause can
    reach an image only through its target's own variation. The effect task
    never uses these columns, and they leave every other output unchanged.
    """
    factors = np.asarray(image_factors, dtype=float)
    if factors.ndim != 2 or factors.shape[0] < 2 or factors.shape[1] == 0 or not np.isfinite(factors).all():
        raise ValueError("image_factors must be a finite matrix with at least two rows and one column")
    if not np.isfinite([effect, heterogeneity, confounding, overlap]).all() or confounding < 0:
        raise ValueError("SCM parameters must be finite and confounding must be nonnegative")
    if not 0 < overlap < 0.5:
        raise ValueError("overlap must lie strictly between 0 and 0.5")

    n, width = factors.shape
    if own_noise is not None:
        own_noise = np.asarray(own_noise, dtype=float)
        if own_noise.shape != factors.shape or not np.isfinite(own_noise).all():
            raise ValueError("own_noise must be a finite matrix matching the image factors")
        if np.any(own_noise.std(axis=0) == 0):
            raise ValueError("own_noise must vary in each column")
    scale = factors.std(axis=0)
    if np.any(scale == 0):
        raise ValueError("image_factors must vary in each column")
    sign_rng, tabular_rng, effect_rng, discovery_rng, cause_rng = [
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(5)
    ]
    signs = sign_rng.choice((-1.0, 1.0), size=width)
    standardized = (factors - factors.mean(axis=0)) / scale
    image_score = standardized @ signs
    score_scale = image_score.std()
    if score_scale <= np.finfo(float).eps:
        raise ValueError("image factors cancel under the seeded weights")
    image_score = (image_score - image_score.mean()) / score_scale

    tabular_1, tabular_noise, tabular_3 = tabular_rng.standard_normal((len(TABULAR_NAMES), n))
    tabular_2 = 0.6 * tabular_1 + 0.8 * tabular_noise
    covariates = np.column_stack((tabular_1, tabular_2, tabular_3))
    tabular_score = (tabular_1 + tabular_2) / np.sqrt(3.2)
    index = confounding * (image_score + tabular_score)
    propensity = overlap + (1 - 2 * overlap) / (1 + np.exp(-np.clip(index, -700, 700)))
    treatment = (effect_rng.random(n) < propensity).astype(int)
    mu0 = index + 0.5 * tabular_3
    mu1 = mu0 + effect + heterogeneity * image_score
    outcome_noise = effect_rng.standard_normal(n)
    y0, y1 = mu0 + outcome_noise, mu1 + outcome_noise

    exposure_parent, *outcome_parent = discovery_rng.permutation(width)[:2]
    weights = confounding * discovery_rng.choice((-1.0, 1.0), size=(2, 2)) * discovery_rng.uniform(0.5, 1.0, (2, 2))
    discovery_treatment = (
        weights[0, 0] * standardized[:, exposure_parent] + weights[0, 1] * tabular_1 + discovery_rng.standard_normal(n)
    )
    discovery_outcome = (
        weights[1, 0] * standardized[:, outcome_parent].sum(axis=1)
        + weights[1, 1] * tabular_2
        + 0.5 * tabular_3
        + effect * discovery_treatment
        + discovery_rng.standard_normal(n)
    )
    causes, cause_targets, cause_strengths = np.empty((n, 0)), (), ()
    if own_noise is not None:
        if width < len(CAUSE_NAMES):
            raise ValueError("Tabular causes need at least as many image factors as causes")
        cause_targets = tuple(int(d) for d in cause_rng.permutation(width)[: len(CAUSE_NAMES)])
        strengths = (confounding != 0) * cause_rng.choice((-1.0, 1.0), len(CAUSE_NAMES))
        strengths = strengths * cause_rng.uniform(0.5, 0.8, len(CAUSE_NAMES))
        own = (own_noise - own_noise.mean(axis=0)) / own_noise.std(axis=0)
        noise = cause_rng.standard_normal((n, len(CAUSE_NAMES)))
        causes = own[:, cause_targets] * strengths + noise * np.sqrt(1 - strengths**2)
        cause_strengths = tuple(map(float, strengths))

    base = _base_adjacency(width, image_adjacency)
    adjacency = _base_adjacency(width, image_adjacency, causes.shape[1])
    first_cause = width + len(TABULAR_NAMES)
    for i, (target, strength) in enumerate(zip(cause_targets, cause_strengths)):
        adjacency[first_cause + i, target] = strength != 0
    adjacency[[exposure_parent, width], -2] = confounding != 0
    adjacency[[*outcome_parent, width + 1], -1] = confounding != 0
    adjacency[-2, -1] = effect != 0
    effect_adjacency = base
    effect_adjacency[: width + 2, -2:] = confounding != 0
    effect_adjacency[-2, -1] = effect != 0
    if heterogeneity != 0:
        effect_adjacency[:width, -1] = True
        effect_adjacency[-2, -1] = True

    return TabularSCM(
        covariates=covariates,
        treatment=treatment,
        outcome=np.where(treatment, y1, y0),
        propensity=propensity,
        mu0=mu0,
        mu1=mu1,
        y0=y0,
        y1=y1,
        effect_adjacency=effect_adjacency,
        discovery_treatment=discovery_treatment,
        discovery_outcome=discovery_outcome,
        adjacency=adjacency,
        causes=causes,
        cause_targets=cause_targets,
        cause_strengths=cause_strengths,
    )
