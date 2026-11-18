"""Can the own parts be separated from a code that mixes the causal bank's factors, when they are Gaussian (where ICA
fails, item 33) but their variances change across the levels of an observed variable such as age, site or phase? And
can the anchors then place a concealed factor in the graph? No images and no training.

Data: ica_simulation.py's set-up (the bank's SCM, its link and noise scale; the lesion centre uniform; the ten latents
standardized and mixed into a 24-unit code with reading noise, optionally with six squares and products of the sharp
latents), plus an observed variable with --levels equal-sized levels drawn at random per subject. Each of the seven
exogenous noises is Gaussian with an SD that changes across levels: exp(strength * g), one standard normal g per noise
and level, rescaled to unit mean variance. The lesion centre does not change. --modulated F leaves only that share of
the noises to change; the rest form a block with the lesion centre.

Separation: whiten the code's ten leading principal components (each level centred on its own mean), then jointly
diagonalise the levels' covariances by Jacobi rotations (Cardoso & Souloumiac 1996); the rotated components are the
estimates. Second-order nonstationary source separation (Pham & Cardoso 2001); the linear case of time-contrastive
learning and iVAE (Hyvarinen & Morioka 2016; Khemakhem et al. 2020). Uses the level labels, never the factors.
Fitted on 2,000 training draws, scored on 1,000 test draws. Each own part's R2 with the component one Hungarian
match on |corr| assigns it (labels only score), beside FastICA, PCA, random rotations of the whitened components
(mean over 30) and a supervised ridge readout from the whole code (the ceiling).

Placement: the four FreeSurfer anchors (brain, ventricle, thickness, temporal) regressed on the ten components; a
component loads on an anchor when its standardized coefficient exceeds --threshold. A noise's true anchor set is the
anchors it is an ancestor of (or is). For the concealed factors: burden -> temporal; lr and sulcal -> none. Scored as
precision and recall over the seven noises and four anchors, with the true noises in place of the components as the
ceiling.

    python out/burden_feasibility_20261102/heterogeneity_ica.py [--seeds 5] [--levels 5]
        [--strengths 0 0.25 0.5 1] [--harmonics 0 0.3] [--modulated 1] [--threshold 0.05]
"""

import argparse
import os
import sys
import warnings

import numpy as np
from scipy.stats import ortho_group
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank as B  # noqa: E402
import ica_simulation as I  # noqa: E402

ANCHORS = ("brain", "ventricle", "thickness", "temporal")
TRAIN, TEST, COMPONENTS, ROTATIONS = I.TRAIN, I.TEST, I.COMPONENTS, 30


def joint_diagonalize(matrices, tol=1e-10, sweeps=500):
    """Orthogonal V that jointly diagonalises symmetric matrices (K, m, m) by Jacobi rotations (Cardoso & Souloumiac
    1996): V.T @ M @ V is as diagonal as one rotation can make every M."""
    stack = np.concatenate(list(matrices), axis=1).astype(float)  # (m, m * K)
    m = stack.shape[0]
    v = np.eye(m)
    for _ in range(sweeps):
        moved = False
        for p in range(m - 1):
            for q in range(p + 1, m):
                ip, iq = np.arange(p, stack.shape[1], m), np.arange(q, stack.shape[1], m)
                g = np.array([stack[p, ip] - stack[q, iq], stack[p, iq] + stack[q, ip]])
                gg = g @ g.T
                ton, toff = gg[0, 0] - gg[1, 1], gg[0, 1] + gg[1, 0]
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton * ton + toff * toff))
                c, s = np.cos(theta), np.sin(theta)
                if abs(s) <= tol:
                    continue
                moved = True
                stack[:, ip], stack[:, iq] = c * stack[:, ip] + s * stack[:, iq], c * stack[:, iq] - s * stack[:, ip]
                stack[p], stack[q] = c * stack[p] + s * stack[q], c * stack[q] - s * stack[p]
                v[:, p], v[:, q] = c * v[:, p] + s * v[:, q], c * v[:, q] - s * v[:, p]
        if not moved:
            break
    return v


def sample(scm, n, levels, strength, modulated, rng):
    """The seven factors, their exogenous noises and each subject's level, with each noise's SD changing by level."""
    level = rng.integers(levels, size=n)
    scales = np.exp(strength * rng.standard_normal((levels, scm["n_dims"])))
    scales /= np.sqrt(np.mean(scales**2, axis=0))
    fixed = rng.permutation(scm["n_dims"])[: int(round((1 - modulated) * scm["n_dims"]))]
    scales[:, fixed] = 1.0
    eps = rng.standard_normal((n, scm["n_dims"])) * scales[level]
    z = np.zeros((n, scm["n_dims"]))
    for d in range(scm["n_dims"]):
        parents = scm["parents"][d]
        if not parents:
            z[:, d] = eps[:, d]
            continue
        signal = z[:, parents] @ scm["weights"][d].numpy().astype(np.float64)
        z[:, d] = np.where(signal > 0, signal, 0.2 * signal) + 0.4 * eps[:, d]
    return z[:, I.FACTORS], eps[:, I.FACTORS], level


def separate(train, test, level_train, level_test, levels):
    """Whitened leading components (pooled within-level covariance) and the joint diagonaliser of the levels'
    covariances, fitted on training draws; returns the test components and the whitened test data."""

    def centred(x, level):
        return x - np.stack([x[level == k].mean(0) for k in range(levels)])[level]

    within = centred(train, level_train)
    values, vectors = np.linalg.eigh(np.cov(within.T))
    top = np.argsort(values)[::-1][:COMPONENTS]
    whiten = vectors[:, top] / np.sqrt(values[top])
    white = within @ whiten
    covariances = [np.cov(white[level_train == k].T) for k in range(levels)]
    rotation = joint_diagonalize(covariances)
    test_white = centred(test, level_test) @ whiten
    return test_white @ rotation, test_white


def loadings(components, anchors):
    """Standardized OLS coefficients of each anchor (columns) on the components (rows)."""
    x = I.standardize(components)
    y = I.standardize(anchors)
    return np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0][1:]


def ancestry(scm):
    """Noise j reaches factor i: j is i or one of its ancestors (rows: the seven noises, columns: the four anchors)."""
    adj = scm["adj"][np.ix_(I.FACTORS, I.FACTORS)].astype(bool)
    reach = np.eye(len(I.FACTORS), dtype=bool) | adj
    for _ in range(len(I.FACTORS)):
        reach = reach | (reach.astype(int) @ reach.astype(int) > 0)
    names = [B.CAUSAL_FACTORS[d] for d in I.FACTORS]
    return reach[:, [names.index(a) for a in ANCHORS]]


def placement(load, truth, threshold):
    found = np.abs(load) > threshold
    hit = (found & truth).sum()
    return hit / max(found.sum(), 1), hit / truth.sum(), found


def run(scm, levels, strength, modulated, harmonics, reading, threshold, seed):
    rng = np.random.default_rng(seed)
    factors, eps, level = sample(scm, TRAIN + TEST, levels, strength, modulated, rng)
    position = rng.uniform(-1, 1, (TRAIN + TEST, 3))
    sources = np.column_stack([eps, position])
    code = I.make_code(np.column_stack([factors, position]), rng, reading, harmonics)
    train, test = I.standardize(code[:TRAIN]), I.standardize(code[TRAIN:], code[:TRAIN])
    components, white = separate(train, test, level[:TRAIN], level[TRAIN:], levels)
    truth = sources[TRAIN:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ica = FastICA(COMPONENTS, whiten="unit-variance", max_iter=2000, tol=1e-5, random_state=seed).fit(train)
    null = np.array(
        [
            I.matched_r2(white @ ortho_group.rvs(COMPONENTS, random_state=seed * 100 + r), truth)
            for r in range(ROTATIONS)
        ]
    )
    names = [B.CAUSAL_FACTORS[d] for d in I.FACTORS]
    anchors = factors[TRAIN:, [names.index(a) for a in ANCHORS]]
    corr = np.abs(np.corrcoef(components.T, truth.T)[:COMPONENTS, COMPONENTS:])
    rows, cols = I.linear_sum_assignment(-corr)
    match = dict(zip(cols, rows))  # source -> component
    load = loadings(components, anchors)[[match[j] for j in range(len(I.FACTORS))]]
    oracle = loadings(truth[:, : len(I.FACTORS)], anchors)
    reach = ancestry(scm)
    return {
        "jd": I.matched_r2(components, truth),
        "ica": I.matched_r2(ica.transform(test), truth),
        "pca": I.matched_r2(white, truth),
        "null": null.mean(0),
        "ceiling": I.ceiling(test, truth),
        "placement": placement(load, reach, threshold)[:2],
        "oracle_placement": placement(oracle, reach, threshold)[:2],
        "found": placement(load, reach, threshold)[2],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    p.add_argument("--harmonics", type=float, nargs="+", default=[0.0, 0.3])
    p.add_argument("--modulated", type=float, default=1.0, help="share of the seven noises whose SD changes by level")
    p.add_argument("--reading", type=float, default=0.3)
    p.add_argument("--threshold", type=float, default=0.05, help="standardized coefficient that counts as a loading")
    a = p.parse_args()
    scm = B.causal_scm(B.run_args(0.4, causal=True).seed)
    seeds = range(a.seeds)
    reach = ancestry(scm)
    print(
        f"Own parts from a {I.UNITS}-unit code by joint diagonalisation across {a.levels} levels (reading noise"
        f" {a.reading}, {a.modulated:.0%} of the noises modulated); matched R2 on {TEST} test draws, means over"
        f" {a.seeds} seeds. Strength: the SD of the log noise SD across levels."
    )
    print(
        f"  {'harm':4s} {'str':4s} | JD: " + " ".join(f"{n:>6s}" for n in I.SHORT) + f"  {'pos':>5s}"
        " | own-part means: JD   ICA   PCA  null  ceil | placement prec / rec (true noises)"
    )
    concealed = {"burden": 2, "lr": 5, "sulcal": 6}
    for harmonics in a.harmonics:
        for strength in a.strengths:
            rows = [run(scm, a.levels, strength, a.modulated, harmonics, a.reading, a.threshold, s) for s in seeds]
            mean = {k: np.mean([r[k] for r in rows], axis=0) for k in ("jd", "ica", "pca", "null", "ceiling")}
            own = slice(0, len(I.FACTORS))
            place = np.mean([r["placement"] for r in rows], axis=0)
            oracle = np.mean([r["oracle_placement"] for r in rows], axis=0)
            print(
                f"  {harmonics:4.1f} {strength:4.2f} | JD: "
                + " ".join(f"{v:6.2f}" for v in mean["jd"][own])
                + f"  {mean['jd'][len(I.FACTORS):].mean():5.2f} |{'':16s}{mean['jd'][own].mean():4.2f}"
                f"  {mean['ica'][own].mean():4.2f}  {mean['pca'][own].mean():4.2f}  {mean['null'][own].mean():4.2f}"
                f"  {mean['ceiling'][own].mean():4.2f} | {place[0]:.2f} / {place[1]:.2f} ({oracle[0]:.2f} / {oracle[1]:.2f})",
                flush=True,
            )
            if strength == max(a.strengths):
                found = np.mean([r["found"] for r in rows], axis=0)
                for name, j in concealed.items():
                    shares = ", ".join(f"{anchor} {found[j, i]:.1f}" for i, anchor in enumerate(ANCHORS))
                    true = [anchor for i, anchor in enumerate(ANCHORS) if reach[j, i]] or ["none"]
                    print(f"      {name}'s component loads on (share of seeds): {shares}; truth: {', '.join(true)}")
    print(
        "  JD: each own part's R2 with its matched component; pos: the lesion centre (mean of x, y, z; unmodulated);"
        "\n  ICA, PCA, null (random rotations), ceil (supervised ridge from the whole code): own-part means."
        "\n  placement: whether a matched component's coefficient on an anchor exceeds the threshold, against the noise"
        " reaching that anchor in the SCM; in brackets the same with the true noises."
    )


if __name__ == "__main__":
    main()
