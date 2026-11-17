"""Can ICA turn a code that mixes the causal bank's factors into the factors' own parts, and can LiNGAM then find their
causal order? No images and no training: the bank's SCM is sampled directly and mixed into a 24-unit code.

Sources: the seven SCM factors' exogenous noises (their own parts) and the lesion centre's three coordinates (uniform
and independent). The SCM is bank.causal_scm's graph and weights, children's noise scale 0.4, with the bank's link
(leaky ReLU, slope 0.2) or a linear one. The noise is Gaussian, as the bank draws it, or non-Gaussian: Laplace,
uniform, or skewed (a centred gamma with shape 2), all with unit variance.

Code: the ten latents, each standardized, mixed by a random 24 x 10 matrix, plus Gaussian reading noise on every unit
(--reading, relative to the unit's SD). --harmonics W also mixes in six squares and products of the sharp latents
(brain size, lr and the lesion centre) at weight W relative to the latents, as the trained codes hold such directions
(HANDOFF item 29). The default sweep is 0, 0.3 and 1.

Part A, own parts: FastICA to ten components, fitted on 2,000 training draws and applied to 1,000 test draws, then
matched to the ten true sources by one Hungarian match on |corr| (labels only score). Reported per source: the matched
R2. Beside it: PCA alone; a null that rotates the ten whitened components at random, so no rule picks the coordinates
(mean and 95th percentile over 30 rotations); and the supervised ceiling, a 5-fold ridge readout of each source from
the whole code.

Part B, causal order: DirectLiNGAM on the seven factors themselves (the positive control, no mixing) and on a random
orthogonal mix of them (a code identified only up to its block), each scored against the SCM's edges: F1 of the
skeleton and of the directed edges.

    python out/burden_feasibility_20261102/ica_simulation.py [--seeds 5] [--reading 0.3] [--harmonics 0 0.3 1]
"""

import argparse
import os
import sys
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import ortho_group
from sklearn.decomposition import PCA, FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank as B  # noqa: E402

FACTORS = list(B.CAUSAL_FACTORS)
SHORT = ["brain", "vent", "burden", "thick", "temp", "lr", "sulc"]
NOISES = ("gaussian", "laplace", "uniform", "skewed")
LINKS = ("leaky_relu", "linear")
TRAIN, TEST, UNITS, COMPONENTS, ROTATIONS = 2000, 1000, 24, 10, 30


def draw_noise(kind, rng, shape):
    """Unit-variance exogenous noise of the given kind."""
    if kind == "gaussian":
        return rng.standard_normal(shape)
    if kind == "laplace":
        return rng.laplace(0.0, 1 / np.sqrt(2), shape)
    if kind == "uniform":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), shape)
    return (rng.gamma(2.0, 1.0, shape) - 2.0) / np.sqrt(2.0)


def sample(scm, n, kind, link, rng, scale=0.4):
    """n draws of the seven factors and their exogenous noises, as eval.synthetic's sample_content_from_scm draws them,
    with noise of the given kind and link."""
    eps = draw_noise(kind, rng, (n, scm["n_dims"]))
    z = np.zeros((n, scm["n_dims"]))
    for d in range(scm["n_dims"]):
        parents = scm["parents"][d]
        if not parents:
            z[:, d] = eps[:, d]
            continue
        signal = z[:, parents] @ scm["weights"][d].numpy().astype(np.float64)
        if link == "leaky_relu":
            signal = np.where(signal > 0, signal, 0.2 * signal)
        z[:, d] = signal + scale * eps[:, d]
    return z[:, FACTORS], eps[:, FACTORS]


def standardize(x, reference=None):
    reference = x if reference is None else reference
    return (x - reference.mean(0)) / reference.std(0)


def make_code(latents, rng, reading, harmonics):
    """The 24-unit code: standardized latents (and six squares and products of the sharp ones at weight
    ``harmonics``) mixed by a random matrix, plus Gaussian reading noise on every unit."""
    features = standardize(latents)
    if harmonics > 0:
        brain, lr, x, y, zc = (features[:, i] for i in (0, 5, 7, 8, 9))
        features = np.column_stack(
            [
                features,
                harmonics * standardize(np.column_stack([brain**2, x**2, y**2, zc**2, brain * lr, x * y])),
            ]
        )
    mixing = rng.standard_normal((features.shape[1], UNITS)) / np.sqrt(features.shape[1])
    code = features @ mixing
    return code + reading * code.std(0) * rng.standard_normal(code.shape)


def matched_r2(components, sources):
    """Each source's squared correlation with the component one Hungarian match on |corr| gives it."""
    k = sources.shape[1]
    corr = np.abs(np.corrcoef(components.T, sources.T)[: components.shape[1], components.shape[1] :])
    rows, cols = linear_sum_assignment(-corr)
    out = np.zeros(k)
    out[cols] = corr[rows, cols] ** 2
    return out


def ceiling(code, sources):
    model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    out = []
    for y in sources.T:
        pred = cross_val_predict(model, code, y, cv=KFold(5, shuffle=True, random_state=0))
        out.append(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return np.array(out)


def part_a(scm, kind, link, harmonics, reading, seed):
    rng = np.random.default_rng(seed)
    factors, eps = sample(scm, TRAIN + TEST, kind, link, rng)
    position = rng.uniform(-1, 1, (TRAIN + TEST, 3))
    sources = np.column_stack([eps, position])
    code = make_code(np.column_stack([factors, position]), rng, reading, harmonics)
    train, test = standardize(code[:TRAIN]), standardize(code[TRAIN:], code[:TRAIN])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ica = FastICA(COMPONENTS, whiten="unit-variance", max_iter=2000, tol=1e-5, random_state=seed).fit(train)
    pca = PCA(COMPONENTS, whiten=True).fit(train)
    whitened = pca.transform(test)
    truth = sources[TRAIN:]
    null = np.array(
        [
            matched_r2(whitened @ ortho_group.rvs(COMPONENTS, random_state=seed * 100 + r), truth)
            for r in range(ROTATIONS)
        ]
    )
    return {
        "ica": matched_r2(ica.transform(test), truth),
        "pca": matched_r2(whitened, truth),
        "null_mean": null.mean(0),
        "null_95": np.percentile(null, 95, axis=0),
        "ceiling": ceiling(test, truth),
    }


def graph_f1(estimate, truth):
    """F1 of the skeleton and of the directed edges; both matrices are parent x child."""

    def f1(est, tru):
        hit = (est & tru).sum()
        return 0.0 if hit == 0 else 2 * hit / (est.sum() + tru.sum())

    skeleton = lambda m: np.triu(m | m.T, 1)  # noqa: E731
    return f1(skeleton(estimate), skeleton(truth)), f1(estimate, truth)


def lingam_graph(x, seed):
    from causallearn.search.FCMBased import lingam

    model = lingam.DirectLiNGAM(random_state=seed)
    model.fit(standardize(x))
    return (np.abs(model.adjacency_matrix_) > 1e-3).T  # adjacency_matrix_[i, j]: j -> i, so transpose to parent x child


def part_b(scm, kind, link, seed):
    rng = np.random.default_rng(1000 + seed)
    factors, _ = sample(scm, TRAIN, kind, link, rng)
    truth = scm["adj"][np.ix_(FACTORS, FACTORS)].astype(bool)
    mixed = factors @ ortho_group.rvs(len(FACTORS), random_state=seed)
    return graph_f1(lingam_graph(factors, seed), truth), graph_f1(lingam_graph(mixed, seed), truth)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--reading", type=float, default=0.3, help="reading noise SD on each code unit, relative to its SD")
    p.add_argument("--harmonics", type=float, nargs="+", default=[0.0, 0.3, 1.0], help="weights of the harmonic terms")
    a = p.parse_args()
    scm = B.causal_scm(B.run_args(0.4, causal=True).seed)
    seeds = range(a.seeds)
    print(
        f"Part A: own parts recovered by ICA from a {UNITS}-unit code of the ten latents (reading noise"
        f" {a.reading}); matched R2 on {TEST} test draws, means over {a.seeds} seeds."
    )
    header = " ".join(f"{n:>6s}" for n in SHORT)
    print(
        f"  {'noise':9s} {'link':10s} {'harm':4s} | ICA: {header}  {'pos':>5s} | own-part means: ICA   PCA  null null95 ceil"
    )
    for harmonics in a.harmonics:
        for link in LINKS:
            for kind in NOISES:
                rows = [part_a(scm, kind, link, harmonics, a.reading, s) for s in seeds]
                mean = {k: np.mean([r[k] for r in rows], axis=0) for k in rows[0]}
                own = slice(0, len(FACTORS))
                print(
                    f"  {kind:9s} {link:10s} {harmonics:4.1f} | ICA: "
                    + " ".join(f"{v:6.2f}" for v in mean["ica"][own])
                    + f"  {mean['ica'][len(FACTORS):].mean():5.2f} | {'':15s}{mean['ica'][own].mean():4.2f}"
                    f"  {mean['pca'][own].mean():4.2f}  {mean['null_mean'][own].mean():4.2f}"
                    f"   {mean['null_95'][own].mean():4.2f}  {mean['ceiling'][own].mean():4.2f}",
                    flush=True,
                )
    print(
        "  ICA: each own part's R2 with the ICA component matched to it; pos: the same for the lesion centre (mean of x, y,"
        " z);\n  PCA: the whitened principal components matched the same way; null: random rotations of them (mean and"
        " 95th percentile);\n  ceil: a supervised ridge readout of each own part from the whole code."
    )
    print(
        f"\nPart B: DirectLiNGAM's graph against the SCM's edges, skeleton F1 / directed F1, means over {a.seeds} seeds."
    )
    print(f"  {'noise':9s} {'link':10s} | {'the factors themselves':>22s} | {'a random orthogonal mix of them':>31s}")
    for link in LINKS:
        for kind in NOISES:
            rows = [part_b(scm, kind, link, s) for s in seeds]
            plain = np.mean([r[0] for r in rows], axis=0)
            mixed = np.mean([r[1] for r in rows], axis=0)
            print(
                f"  {kind:9s} {link:10s} | {plain[0]:10.2f} / {plain[1]:4.2f}      | {mixed[0]:18.2f} / {mixed[1]:4.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
