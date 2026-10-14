"""Test candidate causes of content modality leakage in an encoder-only ADNI checkpoint.

Follow-up to eval/adni/encoder_modality_gap.py, which measures that T1 and FLAIR content
separate. Each check targets one candidate cause; see ENCODER_MODALITY_GAP_CAUSES.md.

  offset    where the mean T1-FLAIR offset lies: a sideways shift or a difference in how
            concentrated each modality is, and whether it runs along directions subjects barely vary in
  shape     FLAIR/T1 variance ratio along each direction, against a within-pair swap null
  ladder    modality probes after removing each modality's mean (in normalized or raw space) and
            then its covariance, every statistic fitted on the training folds
  loss      the run's own InfoNCE on these codes: in-batch top-1 and what the offset costs it
  subjects  per-subject residuals against mask and blur descriptors, ADNI site and optional covariates
  splits    validation against unaugmented training subjects, size-matched where n matters
"""

import argparse
import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import linalg, special, stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from eval.adni.encoder_modality_gap import PROBES, normalize_rows, subject_folds

CONDITIONS = ("normalized", "mean_centered", "raw_offset_removed", "whitened")
SUBJECT_MEASURES = ("residual", "offset_along_gap", "modality_evidence")
SPLITS = ("val", "train")
ADNI_SITE = re.compile(r"^(\d{3})_S_\d{4}$")
MISSING = ("", "na", "nan", "none", "null")
TINY = 1e-12
# What the loss check assumes when the features carry no run settings (the encoder ADNI recipe).
RECIPE = {
    "contrastive_loss_type": "infonce",
    "tau": 0.1,
    "cross_view_negs_only": True,
    "standardize_code": False,
    "contrastive_proj_dim": 0,
    "batch_size": 32,
}


def _num(x):
    """JSON-safe float: None for missing or non-finite values."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _cosine(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > TINY and nv > TINY else None


def _gaussian_accuracy(distance):
    """Best accuracy for two equal-covariance Gaussians whose means are this Mahalanobis distance apart."""
    return float(stats.norm.cdf(distance / 2))


def _jitter(cov):
    """A relative ridge, so channels that never vary leave a covariance invertible."""
    return 1e-8 * max(np.trace(cov) / len(cov), TINY) * np.eye(len(cov))


def remove_raw_offset(raw_a, raw_b, rows=None):
    """Give both modalities' raw codes the pooled raw mean, using ``rows`` (default all) for the means.

    That is the most a constant per-modality bias in the encoders could change. The shared mean
    stays, so the cosine geometry the loss sees is otherwise untouched.
    """
    rows = slice(None) if rows is None else rows
    ma, mb = raw_a[rows].mean(0), raw_b[rows].mean(0)
    pooled = (ma + mb) / 2
    return raw_a - ma + pooled, raw_b - mb + pooled


def swap_pairs(a, b, rng):
    """Exchange the two views of a random half of the pairs: the null in which the modality label is arbitrary."""
    flip = rng.random(len(a)) < 0.5
    a2, b2 = a.copy(), b.copy()
    a2[flip], b2[flip] = b[flip], a[flip]
    return a2, b2


def validate(a, b, subjects, n_splits):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    subjects = np.asarray(subjects, dtype=str)
    if a.ndim != 2 or b.shape != a.shape or a.shape[1] < 1:
        raise ValueError("Expected matching (pairs, content_channels) arrays with at least one channel")
    if subjects.shape != (len(a),) or np.any(subjects == ""):
        raise ValueError("Need one nonempty subject ID per paired row")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite content features")
    if len(np.unique(subjects)) < n_splits:
        raise ValueError(f"Need at least {n_splits} subjects for {n_splits} folds")
    return a, b, subjects


# ---------------------------------------------------------------------------
# offset: where the mean T1-FLAIR offset lies
# ---------------------------------------------------------------------------


def within_covariance(a, b):
    """Covariance of both modalities about their own means, pooled."""
    ca, cb = a - a.mean(0), b - b.mean(0)
    return (ca.T @ ca + cb.T @ cb) / (len(a) + len(b))


def offset_geometry(a, b, rng, directions=4000):
    """Where the offset between the mean L2-normalized T1 row and the mean FLAIR row lies.

    ``radial_share`` is the fraction of the offset's squared length along the two modalities' average
    mean: near 1 the means are parallel and differ only in length (one modality's codes are more
    concentrated), near 0 one cloud is shifted sideways relative to the other.

    ``lda_accuracy`` is the best linear accuracy the offset alone would give if each modality were a
    Gaussian with the pooled within-modality covariance. The same offset pointed in a random direction
    gives ``random_direction_accuracy``; an offset that beats most random directions runs along
    directions in which subjects barely vary.
    """
    ma, mb = a.mean(0), b.mean(0)
    gap, centre = ma - mb, (ma + mb) / 2
    length = float(np.linalg.norm(gap))
    out = {
        "centroid_distance": length,
        "mean_length_t1": float(np.linalg.norm(ma)),
        "mean_length_flair": float(np.linalg.norm(mb)),
        "cosine_between_means": _cosine(ma, mb),
        "radial_share": None,
        "lda_accuracy": None,
        "isotropic_accuracy": None,
        "random_direction_accuracy": None,
        "random_direction_percentile": None,
        "low_variance_half_share": None,
        "random_low_variance_half_share": None,
    }
    if length > TINY and np.linalg.norm(centre) > TINY:
        out["radial_share"] = float((gap @ centre) ** 2 / (length**2 * (centre @ centre)))
    cov = within_covariance(a, b)
    d = len(cov)
    if length <= TINY or np.trace(cov) <= TINY or d < 2:
        return out
    values, vectors = np.linalg.eigh(cov + _jitter(cov))  # ascending: lowest-variance directions first
    share = (vectors.T @ gap) ** 2 / length**2
    distance = length * np.sqrt(share @ (1 / values))
    u = rng.standard_normal((directions, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    random_share = (u @ vectors) ** 2
    random_distance = length * np.sqrt(random_share @ (1 / values))
    half = d // 2
    low = random_share[:, :half].sum(1)
    out.update(
        lda_accuracy=_gaussian_accuracy(distance),
        isotropic_accuracy=_gaussian_accuracy(length / np.sqrt(np.trace(cov) / d)),
        random_direction_accuracy={
            "median": _gaussian_accuracy(np.median(random_distance)),
            "p95": _gaussian_accuracy(np.quantile(random_distance, 0.95)),
        },
        random_direction_percentile=float(np.mean(random_distance < distance)),
        low_variance_half_share=float(share[:half].sum()),
        random_low_variance_half_share={
            "p05": float(np.quantile(low, 0.05)),
            "median": float(np.median(low)),
            "p95": float(np.quantile(low, 0.95)),
        },
    )
    return out


def check_offset(a, b, rng, permutations):
    out = offset_geometry(a, b, rng)
    if permutations:
        null = [np.linalg.norm(np.subtract(*[x.mean(0) for x in swap_pairs(a, b, rng)])) for _ in range(permutations)]
        out["centroid_distance_null_p95"] = float(np.quantile(null, 0.95))
    return out


# ---------------------------------------------------------------------------
# shape: FLAIR/T1 variance ratio along each direction
# ---------------------------------------------------------------------------


def variance_ratios(a, b):
    """FLAIR/T1 variance along each direction that decorrelates both modalities at once, ascending.

    These generalized eigenvalues do not change under any invertible linear map applied to both
    modalities, so they compare the two clouds' shapes whatever the channels' scales.
    """
    ca, cb = a - a.mean(0), b - b.mean(0)
    sa, sb = ca.T @ ca / len(ca), cb.T @ cb / len(cb)
    jitter = _jitter((sa + sb) / 2)
    return linalg.eigh(sb + jitter, sa + jitter, eigvals_only=True)


def effective_rank(x):
    """exp(entropy) of the normalized covariance eigenvalues: how many directions the codes use."""
    centred = x - x.mean(0)
    values = np.clip(np.linalg.eigvalsh(centred.T @ centred / len(x)), 0, None)
    if values.sum() <= TINY:
        return None
    p = values[values > 0] / values.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def relative_ratios(a, b):
    """Variance ratios over their median: each direction's FLAIR/T1 gain relative to a typical direction.

    On L2-normalized rows each modality's total spread is close to 1, so a modality that varies less
    along a few directions varies more along all the others. Dividing by the median keeps that
    bookkeeping from flagging every remaining direction.
    """
    ratios = variance_ratios(a, b)
    return ratios / np.median(ratios)


def check_shape(a, b, rng, permutations):
    """Each modality's spread about its own mean, and the relative variance ratios against a swap null.

    Pass rows L2-normalized after ``remove_raw_offset``: normalizing two clouds that sit in different
    places compresses each along its own mean direction, so a pure raw offset would otherwise read
    as a shape difference. Both views are centred before swapping, so the null asks whether the
    clouds differ in shape once each sits on its own mean. The null is taken per rank, because
    sample generalized eigenvalues spread away from 1 even when the two covariances are equal.
    """
    ca, cb = a - a.mean(0), b - b.mean(0)
    out = {
        "spread_t1": float(np.sqrt(np.square(ca).sum(1).mean())),
        "spread_flair": float(np.sqrt(np.square(cb).sum(1).mean())),
        "effective_rank_t1": effective_rank(ca),
        "effective_rank_flair": effective_rank(cb),
    }
    if np.square(ca).sum() + np.square(cb).sum() <= TINY or a.shape[1] < 2:
        return out
    relative = relative_ratios(ca, cb)
    distance = float(np.sqrt(np.square(np.log(relative)).sum()))
    out.update(
        variance_ratios=variance_ratios(ca, cb).tolist(), relative_ratios=relative.tolist(), shape_distance=distance
    )
    if permutations:
        null = np.array([relative_ratios(*swap_pairs(ca, cb, rng)) for _ in range(permutations)])
        null_distance = np.sqrt(np.square(np.log(null)).sum(1))
        lower, upper = np.quantile(null, [0.025, 0.975], axis=0)
        out.update(
            null_lower=lower.tolist(),
            null_upper=upper.tolist(),
            directions_flair_higher=int((relative > upper).sum()),
            directions_flair_lower=int((relative < lower).sum()),
            shape_distance_null_p95=float(np.quantile(null_distance, 0.95)),
            shape_distance_p=float((1 + (null_distance >= distance).sum()) / (1 + permutations)),
        )
    return out


# ---------------------------------------------------------------------------
# ladder: modality probes after removing the mean, then the covariance
# ---------------------------------------------------------------------------


def _estimator(probe):
    if probe == "linear":
        return LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    return SVC(C=1.0, kernel="rbf", gamma="scale")


def _whiten(x, train):
    """Remove one modality's training-fold mean and sample covariance.

    The plain sample covariance whitens the training rows exactly. A shrinkage estimate would leave
    part of a real covariance difference in place, and the probe would then read it as structure
    beyond second order. Eigenvalues are floored only so a direction the codes never use stays finite.
    """
    rows = x[train]
    centre = rows.mean(0)
    cov = (rows - centre).T @ (rows - centre) / len(rows)
    if np.trace(cov) <= TINY:
        return x - centre
    values, vectors = np.linalg.eigh(cov)
    values = np.clip(values, 1e-6 * np.trace(cov) / len(cov), None)
    return (x - centre) @ (vectors / np.sqrt(values)) @ vectors.T


def ladder_features(raw_a, raw_b, a, b, train, condition):
    """One condition's T1 and FLAIR rows; every fitted statistic comes from the training rows."""
    if condition == "normalized":
        return a, b
    if condition == "mean_centered":
        return a - a[train].mean(0), b - b[train].mean(0)
    if condition == "raw_offset_removed":
        return tuple(normalize_rows(x) for x in remove_raw_offset(raw_a, raw_b, train))
    if condition == "whitened":
        return _whiten(a, train), _whiten(b, train)
    raise ValueError(f"Unknown condition {condition!r}")


def probe_ladder(raw_a, raw_b, folds, decisions=False):
    """Held-out modality accuracy per condition and probe, with the diagnostic's estimators and folds.

    ``normalized`` and ``mean_centered`` reproduce encoder_modality_gap.diagnose exactly. With
    ``decisions`` it also returns the mean-centred RBF probe's held-out decision values (T1 rows
    first, then FLAIR; positive means FLAIR).
    """
    a, b = normalize_rows(raw_a), normalize_rows(raw_b)
    n = len(a)
    labels = np.repeat([0, 1], n)
    predictions = {f"{c}_{p}": np.full(2 * n, -1) for c in CONDITIONS for p in PROBES}
    margin = np.full(2 * n, np.nan)
    for train, test in folds:
        y_train = np.repeat([0, 1], len(train))
        rows = np.r_[test, n + test]
        for condition in CONDITIONS:
            xa, xb = ladder_features(raw_a, raw_b, a, b, train, condition)
            x_train = np.concatenate([xa[train], xb[train]])
            x_test = np.concatenate([xa[test], xb[test]])
            # As in the diagnostic: zero channels that are constant on the training rows, so
            # StandardScaler cannot amplify floating-point residue into a modality label.
            constant = x_train.std(axis=0) <= TINY
            x_train[:, constant] = 0
            x_test[:, constant] = 0
            for probe in PROBES:
                clf = make_pipeline(StandardScaler(), _estimator(probe)).fit(x_train, y_train)
                predictions[f"{condition}_{probe}"][rows] = clf.predict(x_test)
                if decisions and condition == "mean_centered" and probe == "rbf":
                    margin[rows] = clf.decision_function(x_test)
    accuracy = {key: float(np.mean(pred == labels)) for key, pred in predictions.items()}
    return accuracy, (margin if decisions else None)


def check_ladder(raw_a, raw_b, subjects, n_splits, seed, rng, permutations):
    """The ladder on every pair, plus its within-pair swap null at this split's size."""
    folds = list(subject_folds(subjects, n_splits, seed))
    with threadpool_limits(limits=1):
        accuracy, margin = probe_ladder(raw_a, raw_b, folds, decisions=True)
        null = [probe_ladder(*swap_pairs(raw_a, raw_b, rng), folds)[0] for _ in range(permutations)]
    out = {"pairs": len(raw_a), "accuracy": accuracy, "permutations": permutations}
    if permutations:
        out["null_p95"] = {key: float(np.quantile([n[key] for n in null], 0.95)) for key in accuracy}
        out["p_value"] = {
            key: float((1 + sum(n[key] >= value for n in null)) / (1 + permutations)) for key, value in accuracy.items()
        }
    return out, margin


def retrieval_top1(a, b):
    """T1->FLAIR and FLAIR->T1 subject retrieval top-1 among these pairs, averaged (chance 1/n)."""
    similarity = a @ b.T
    target = np.arange(len(a))
    return float(((similarity.argmax(1) == target).mean() + (similarity.argmax(0) == target).mean()) / 2)


def check_matched(raw_a, raw_b, subjects, size, draws, n_splits, seed, rng):
    """Ladder and retrieval on random subsets of ``size`` subjects: comparable with a split that small."""
    unique = np.unique(subjects)
    ladders, retrievals = [], []
    with threadpool_limits(limits=1):
        for _ in range(draws):
            keep = np.isin(subjects, rng.choice(unique, size, replace=False))
            folds = list(subject_folds(subjects[keep], n_splits, seed))
            ladders.append(probe_ladder(raw_a[keep], raw_b[keep], folds)[0])
            retrievals.append(retrieval_top1(normalize_rows(raw_a[keep]), normalize_rows(raw_b[keep])))
    return {
        "subjects": int(size),
        "draws": draws,
        "accuracy_mean": {key: float(np.mean([d[key] for d in ladders])) for key in ladders[0]},
        "accuracy_sd": {key: float(np.std([d[key] for d in ladders])) for key in ladders[0]},
        "retrieval_top1_mean": float(np.mean(retrievals)),
        "retrieval_top1_sd": float(np.std(retrievals)),
    }


# ---------------------------------------------------------------------------
# loss: the run's InfoNCE on these codes
# ---------------------------------------------------------------------------


def _cross_entropy(scores):
    """CrossEntropyLoss with targets arange(rows): mean over rows of -log softmax at the diagonal."""
    return float(np.mean(special.logsumexp(scores, axis=1) - np.diag(scores)))


def infonce(z1, z2, tau, cross_view_negs_only=True, standardize=False):
    """training.losses.infonce_base_loss for one batch of two views, with its in-batch top-1.

    Cross-view-only negatives sum two cross-entropies (T1 anchors, then FLAIR anchors); with
    same-view negatives there is one cross-entropy over all 2B anchors. ``standardize`` is the
    trainer's --standardize-code: each unit centred and scaled over the batch, per view.
    """
    if standardize:
        z1, z2 = [(z - z.mean(0)) / (z.std(0, ddof=1) + 1e-6) for z in (z1, z2)]
    u, v = normalize_rows(z1), normalize_rows(z2)
    cross = u @ v.T / tau
    d = len(u)
    if cross_view_negs_only:
        target = np.arange(d)
        hits = (cross.argmax(1) == target).sum() + (cross.argmax(0) == target).sum()
        return _cross_entropy(cross) + _cross_entropy(cross.T), hits / (2 * d)
    within_u, within_v = u @ u.T / tau, v @ v.T / tau
    np.fill_diagonal(within_u, -np.inf)
    np.fill_diagonal(within_v, -np.inf)
    scores = np.block([[cross, within_u], [within_v, cross.T]])
    return _cross_entropy(scores), (scores.argmax(1) == np.arange(2 * d)).sum() / (2 * d)


def check_loss(raw_a, raw_b, settings, rng, batches, batch_sizes):
    """Mean loss over random batches of these pairs: as they are, with the offset removed, fully aligned.

    ``offset_removed`` gives both modalities the pooled raw mean, which is all a per-modality bias in
    the encoders could change. ``aligned`` feeds the pair average as both views, the loss's floor for
    this spread of subjects. ``offset_share`` is the fraction of the gap to that floor the offset accounts for.
    """
    kind = settings["contrastive_loss_type"]
    if kind != "infonce":
        return {
            "skipped": f"{kind} standardizes each view over the batch, so a per-modality offset costs it nothing "
            "by construction"
        }
    if settings["contrastive_proj_dim"]:
        return {"skipped": "the loss reads a projection head's output, which these features do not include"}
    average = (raw_a + raw_b) / 2
    variants = {
        "as_is": (raw_a, raw_b),
        "offset_removed": remove_raw_offset(raw_a, raw_b),
        "aligned": (average, average),
    }
    loss_args = (settings["tau"], settings["cross_view_negs_only"], settings["standardize_code"])
    out = {
        "tau": settings["tau"],
        "cross_view_negs_only": settings["cross_view_negs_only"],
        "standardize_code": settings["standardize_code"],
        "batches": batches,
        "by_batch_size": {},
    }
    for size in batch_sizes:
        if size > len(raw_a):
            out["by_batch_size"][str(size)] = {"skipped": f"only {len(raw_a)} pairs"}
            continue
        draws = [rng.choice(len(raw_a), size, replace=False) for _ in range(batches)]
        result = {}
        for name, (x, y) in variants.items():
            values = np.array([infonce(x[i], y[i], *loss_args) for i in draws])
            result[name] = {"loss": float(values[:, 0].mean()), "top1": float(values[:, 1].mean())}
        offset = result["as_is"]["loss"] - result["offset_removed"]["loss"]
        misalignment = result["as_is"]["loss"] - result["aligned"]["loss"]
        result["offset_cost"] = offset
        result["misalignment_cost"] = misalignment
        result["offset_share"] = offset / misalignment if misalignment > TINY else None
        out["by_batch_size"][str(size)] = result
    return out


# ---------------------------------------------------------------------------
# subjects: per-subject residuals against descriptors and covariates
# ---------------------------------------------------------------------------


def subject_measures(a, b, margin):
    """Per subject: ``residual``, the T1-FLAIR difference left after removing the mean offset;
    ``offset_along_gap``, this subject's T1-FLAIR difference along the mean offset (its mean is the
    centroid distance); ``modality_evidence``, how far the held-out mean-centred RBF probe puts this
    subject's two scans on their correct sides (positive = correct)."""
    n = len(a)
    ma, mb = a.mean(0), b.mean(0)
    gap = ma - mb
    length = np.linalg.norm(gap)
    direction = gap / length if length > TINY else np.zeros_like(gap)
    return {
        "residual": np.linalg.norm((b - mb) - (a - ma), axis=1),
        "offset_along_gap": (a - b) @ direction,
        "modality_evidence": 0.5 * (margin[n:] - margin[:n]),
    }


def site_codes(subjects):
    """ADNI PTIDs are <site>_S_<RID>; None unless most subject IDs follow that pattern."""
    matches = [ADNI_SITE.match(str(s)) for s in subjects]
    if sum(m is not None for m in matches) < 0.5 * len(matches):
        return None
    return np.array([m.group(1) if m else None for m in matches], dtype=object)


def classify_column(values):
    """Numeric when every present value parses and more than ten differ; otherwise categorical."""
    present = [v for v in values if v.lower() not in MISSING]
    try:
        numbers = [float(v) for v in present]
    except ValueError:
        numbers = None
    if numbers is not None and len(set(numbers)) > 10:
        return "numeric", np.array([float(v) if v.lower() not in MISSING else np.nan for v in values])
    return "categorical", np.array([v if v.lower() not in MISSING else None for v in values], dtype=object)


def read_covariates(path, subject_column, subjects):
    """Columns of a CSV keyed by subject ID, aligned to ``subjects``; the first row wins for repeats."""
    with open(path, newline="") as fp:
        reader = csv.DictReader(fp)
        fields = reader.fieldnames or []
        if subject_column not in fields:
            raise ValueError(f"{path} has no {subject_column!r} column (columns: {', '.join(fields)})")
        rows = {}
        for row in reader:
            rows.setdefault(str(row[subject_column]).strip(), row)
    matched = sum(str(s) in rows for s in subjects)
    columns = {}
    for name in fields:
        if name != subject_column:
            columns[name] = classify_column([(rows.get(str(s), {}).get(name) or "").strip() for s in subjects])
    return columns, matched


def association(target, kind, values, min_group):
    """Spearman correlation for a numeric covariate; Kruskal-Wallis across groups for a categorical one."""
    target = np.asarray(target, dtype=float)
    if kind == "numeric":
        values = np.asarray(values, dtype=float)
        keep = np.isfinite(values) & np.isfinite(target)
        if keep.sum() < 10 or np.ptp(values[keep]) == 0 or np.ptp(target[keep]) == 0:
            return None
        rho, p = stats.spearmanr(values[keep], target[keep])
        return {"test": "spearman", "n": int(keep.sum()), "effect": float(rho), "p": float(p)}
    keep = np.array([v is not None for v in values]) & np.isfinite(target)
    labels, kept = np.asarray(values, dtype=object)[keep].astype(str), target[keep]
    groups = {g: kept[labels == g] for g in np.unique(labels) if (labels == g).sum() >= min_group}
    if len(groups) < 2:
        return None
    try:
        h, p = stats.kruskal(*groups.values())
    except ValueError:  # every value identical
        return None
    n = sum(len(m) for m in groups.values())
    return {
        "test": "kruskal",
        "n": int(n),
        "effect": float(h / (n - 1)),  # epsilon-squared
        "p": float(p),
        "groups": {g: {"n": int(len(m)), "median": float(np.median(m))} for g, m in groups.items()},
    }


def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    if not len(p):
        return p
    order = np.argsort(p)
    scaled = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.empty_like(p)
    q[order] = np.clip(np.minimum.accumulate(scaled[::-1])[::-1], 0, 1)
    return q


def check_subjects(measures, covariates, min_group):
    tests = []
    for measure, target in measures.items():
        for name, (kind, values) in covariates.items():
            result = association(target, kind, values, min_group)
            if result is not None:
                tests.append({"measure": measure, "covariate": name, **result})
    for test, q in zip(tests, benjamini_hochberg([t["p"] for t in tests])):
        test["q"] = float(q)
    return sorted(tests, key=lambda t: (t["q"], t["p"]))


def descriptor_covariates(descriptors):
    """Extraction descriptors as covariates: the blurriest axis is a category, the rest are numeric."""
    covariates = {}
    for name, values in descriptors.items():
        if name.startswith("blur_axis"):
            covariates[name] = ("categorical", np.array([f"axis{int(v)}" for v in values], dtype=object))
        else:
            covariates[name] = ("numeric", np.asarray(values, dtype=float))
    return covariates


# ---------------------------------------------------------------------------
# One split, then the comparison
# ---------------------------------------------------------------------------


def analyse_split(data, settings, args, matched_size=None, probe_permutations=0, covariates_path=None):
    raw_a, raw_b, subjects = validate(data["content_t1"], data["content_flair"], data["subjects"], args.folds)
    a, b = normalize_rows(raw_a), normalize_rows(raw_b)
    ac, bc = a - a.mean(0), b - b.mean(0)
    # One generator per check, so turning one check's size up or down leaves the others' draws alone.
    rngs = {
        name: np.random.default_rng([args.seed, k])
        for k, name in enumerate(("offset", "shape", "ladder", "matched", "loss"))
    }
    report = {
        "pairs": len(a),
        "subjects": int(len(np.unique(subjects))),
        "content_channels": int(a.shape[1]),
        "repeated_subjects": bool(len(np.unique(subjects)) < len(subjects)),
        "geometry": {
            "paired_cosine": _num(np.mean(np.sum(a * b, 1))),
            "paired_cosine_after_centering": _num(np.mean(np.sum(normalize_rows(ac) * normalize_rows(bc), 1))),
            "retrieval_top1": retrieval_top1(a, b),
            "retrieval_chance": 1 / len(a),
        },
        "offset": check_offset(a, b, rngs["offset"], args.permutations),
        "shape": check_shape(
            *(normalize_rows(x) for x in remove_raw_offset(raw_a, raw_b)), rngs["shape"], args.permutations
        ),
    }
    report["ladder"], margin = check_ladder(
        raw_a, raw_b, subjects, args.folds, args.seed, rngs["ladder"], probe_permutations
    )
    if matched_size is not None and matched_size < report["subjects"]:
        report["matched"] = check_matched(
            raw_a, raw_b, subjects, matched_size, args.matched_draws, args.folds, args.seed, rngs["matched"]
        )
    sizes = args.loss_batch_sizes or [settings["batch_size"], 4 * settings["batch_size"]]
    report["loss"] = check_loss(raw_a, raw_b, settings, rngs["loss"], args.loss_batches, sizes)
    measures = subject_measures(a, b, margin)
    covariates = descriptor_covariates(data.get("descriptors", {}))
    sites = site_codes(subjects)
    if sites is not None:
        covariates["site"] = ("categorical", sites)
    if covariates_path:
        extra, matched = read_covariates(covariates_path, args.subject_column, subjects)
        report["covariates_matched_subjects"] = matched
        covariates.update({name if name not in covariates else f"csv_{name}": column for name, column in extra.items()})
    report["covariates"] = sorted(covariates)
    report["associations"] = check_subjects(measures, covariates, args.min_group)
    rows = []
    for i, subject in enumerate(subjects):
        row = {"subject": subject, **{m: _num(v[i]) for m, v in measures.items()}}
        for name, (_, values) in covariates.items():
            value = values[i]
            row[name] = _num(value) if isinstance(value, (float, np.floating)) else value
        rows.append(row)
    return report, rows


def resolved_settings(metadata, args):
    """The run's loss settings, from the features' metadata where present, else the recipe defaults."""
    saved = (metadata or {}).get("resolved_settings") or {}
    settings, assumed = {}, []
    for key, default in RECIPE.items():
        if saved.get(key) is not None:
            settings[key] = saved[key]
        else:
            settings[key] = default
            assumed.append(key)
    if args.tau is not None:
        settings["tau"] = args.tau
    settings["assumed_from_recipe"] = assumed
    return settings


COMPARISON = (
    ("pairs", ("pairs",)),
    ("centroid distance", ("offset", "centroid_distance")),
    ("  swap-null p95", ("offset", "centroid_distance_null_p95")),
    ("paired cosine", ("geometry", "paired_cosine")),
    ("paired cosine after centring", ("geometry", "paired_cosine_after_centering")),
    ("radial share", ("offset", "radial_share")),
    ("LDA accuracy from the offset", ("offset", "lda_accuracy")),
    ("offset share, low-variance half", ("offset", "low_variance_half_share")),
    ("spread T1", ("shape", "spread_t1")),
    ("spread FLAIR", ("shape", "spread_flair")),
    ("shape distance (relative ratios)", ("shape", "shape_distance")),
)


def comparison(reports):
    """Side by side; n-dependent probe and retrieval numbers come from the size-matched draws."""
    names = list(reports)
    table = {label: {s: _lookup(reports[s], path) for s in names} for label, path in COMPARISON}
    for key in (f"{c}_{p}" for c in CONDITIONS for p in PROBES):
        table[f"probe {key}"] = {
            s: (
                reports[s]["matched"]["accuracy_mean"][key]
                if "matched" in reports[s]
                else reports[s]["ladder"]["accuracy"][key]
            )
            for s in names
        }
    table["retrieval top-1 (size-matched)"] = {
        s: (
            reports[s]["matched"]["retrieval_top1_mean"]
            if "matched" in reports[s]
            else reports[s]["geometry"]["retrieval_top1"]
        )
        for s in names
    }
    for size in sorted({k for s in names for k in reports[s]["loss"].get("by_batch_size", {})}, key=int):
        table[f"offset cost to the loss, batch {size}"] = {
            s: reports[s]["loss"]["by_batch_size"].get(size, {}).get("offset_cost") for s in names
        }
    return table


def _lookup(report, path):
    value = report
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def analyse(splits, metadata, args):
    settings = resolved_settings(metadata, args)
    smallest = min(len(np.unique(np.asarray(d["subjects"], dtype=str))) for d in splits.values())
    reports, rows = {}, {}
    for name, data in splits.items():
        n_subjects = len(np.unique(np.asarray(data["subjects"], dtype=str)))
        # The swap null of the probes is costly, so it runs once, on a split of the smallest size,
        # and the larger splits' size-matched draws are read against it.
        primary = n_subjects == smallest and not any(r["ladder"]["permutations"] for r in reports.values())
        print(f"Analysing {name}: {len(data['subjects'])} pairs ...", flush=True)
        reports[name], rows[name] = analyse_split(
            data,
            settings,
            args,
            matched_size=smallest if n_subjects > smallest else None,
            probe_permutations=args.probe_permutations if primary else 0,
            covariates_path=args.covariates,
        )
    report = {
        "settings": settings,
        "splits": reports,
        "comparison": comparison(reports) if len(reports) > 1 else None,
        "protocol": {
            "space": "Checks read L2-normalized content rows. Shape, the raw_offset_removed rung and the loss "
            "first give both modalities the pooled raw mean (remove_raw_offset); the loss normalizes itself",
            "swap_null": "Exchange T1 and FLAIR within a random half of the pairs; the shape null swaps "
            "after centring each modality on its own mean",
            "ladder": "encoder_modality_gap's folds (subjects grouped), pooled StandardScaler, C=1 logistic "
            "regression and RBF SVM; raw_offset_removed removes the training-fold raw offset before L2 "
            "normalization; whitened removes each modality's training-fold mean and sample covariance",
            "probe_null": "One swap null for the ladder, on the first split of the smallest size; larger splits "
            "report size-matched subject draws",
            "loss": "The trainer's global InfoNCE on random batches of these (held-out, unaugmented) pairs",
            "associations": "Spearman (numeric) or Kruskal-Wallis (categorical, groups >= min_group); "
            "Benjamini-Hochberg q over all tests of a split",
        },
        "interpretation": [
            "offset: radial share near 1 = one modality's codes are more concentrated; near 0 = a sideways "
            "shift. LDA accuracy above the random-direction p95 = the offset runs along low-variance directions.",
            "shape: ratios outside the swap envelope = directions one modality reads at a different gain.",
            "ladder: mean-centred RBF high but whitened RBF near its null = a covariance difference; whitened "
            "still high = structure beyond mean and covariance (e.g. subgroups). raw_offset_removed well below "
            "mean_centered = part of the residual is a raw offset that L2 normalization made subject-dependent.",
            "loss: top-1 near 1 = saturated batches; offset share = how much of the loss above its aligned "
            "floor the offset accounts for.",
            "splits: a much smaller gap on training subjects = the encoders generalize differently.",
            "Centring and whitening use the known modality label: diagnostic interventions, not invariance.",
        ],
    }
    return report, rows


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _f(x, digits=3):
    return "-" if x is None else f"{x:.{digits}f}"


def print_split(name, r):
    o, s, g, lad = r["offset"], r["shape"], r["geometry"], r["ladder"]
    print(f"\n== {name}: {r['pairs']} pairs, {r['subjects']} subjects, {r['content_channels']} content channels ==")
    print(
        f"geometry  paired cosine {_f(g['paired_cosine'])}, after centring {_f(g['paired_cosine_after_centering'])}; "
        f"retrieval top-1 {_f(g['retrieval_top1'])} (chance {_f(g['retrieval_chance'])})"
    )
    print(
        f"offset    centroid distance {_f(o['centroid_distance'])} (swap-null p95 {_f(o.get('centroid_distance_null_p95'))}); "
        f"mean length T1 {_f(o['mean_length_t1'])}, FLAIR {_f(o['mean_length_flair'])}, cosine between means "
        f"{_f(o['cosine_between_means'])}"
    )
    print(f"          radial share {_f(o['radial_share'])}  (1 = concentration difference, 0 = sideways shift)")
    if o["lda_accuracy"] is not None:
        rd, lo = o["random_direction_accuracy"], o["random_low_variance_half_share"]
        print(
            f"          accuracy the offset alone gives: {_f(o['lda_accuracy'])}; same length, random direction: "
            f"median {_f(rd['median'])}, p95 {_f(rd['p95'])} (beats {100 * o['random_direction_percentile']:.0f}% of "
            f"directions); isotropic {_f(o['isotropic_accuracy'])}"
        )
        print(
            f"          offset share in the low-variance half of directions {_f(o['low_variance_half_share'])} "
            f"(random direction: median {_f(lo['median'])}, p95 {_f(lo['p95'])})"
        )
    print(
        f"shape     spread T1 {_f(s['spread_t1'])}, FLAIR {_f(s['spread_flair'])}; effective rank T1 "
        f"{_f(s['effective_rank_t1'], 1)}, FLAIR {_f(s['effective_rank_flair'], 1)}"
    )
    if "relative_ratios" in s:
        ratios = s["relative_ratios"]
        line = f"          FLAIR/T1 variance ratio over its median, lowest {', '.join(_f(x, 2) for x in ratios[:3])}"
        line += f"; highest {', '.join(_f(x, 2) for x in ratios[-3:][::-1])}"
        print(line)
        if "null_upper" in s:
            print(
                f"          swap envelope: lowest >= {', '.join(_f(x, 2) for x in s['null_lower'][:3])}; highest <= "
                f"{', '.join(_f(x, 2) for x in s['null_upper'][-3:][::-1])}"
            )
            print(
                f"          directions outside it: FLAIR higher {s['directions_flair_higher']}, lower "
                f"{s['directions_flair_lower']}; shape distance {_f(s['shape_distance'])} "
                f"(null p95 {_f(s['shape_distance_null_p95'])}, p {_f(s['shape_distance_p'])})"
            )
    head = "ladder    held-out modality accuracy, read against each rung's null"
    null = lad.get("null_p95")
    print(head + ("   linear [null p95]   rbf [null p95]" if null else "   linear   rbf"))
    for c in CONDITIONS:
        cells = []
        for p in PROBES:
            key = f"{c}_{p}"
            cells.append(f"{_f(lad['accuracy'][key])}" + (f" [{_f(null[key])}]" if null else ""))
        print(f"          {c:<20s}" + "   ".join(f"{cell:>15s}" for cell in cells))
    if "matched" in r:
        m = r["matched"]
        print(
            f"          size-matched ({m['subjects']} subjects x {m['draws']} draws): "
            + ", ".join(f"{k} {_f(m['accuracy_mean'][k])}" for k in ("mean_centered_rbf", "whitened_rbf"))
            + f"; retrieval top-1 {_f(m['retrieval_top1_mean'])}"
        )
    loss = r["loss"]
    if "skipped" in loss:
        print(f"loss      skipped: {loss['skipped']}")
    else:
        negs = "cross-view negatives only" if loss["cross_view_negs_only"] else "with same-view negatives"
        std = ", standardized code" if loss["standardize_code"] else ""
        print(f"loss      tau {loss['tau']}, {negs}{std}, {loss['batches']} random batches")
        for size, res in loss["by_batch_size"].items():
            if "skipped" in res:
                print(f"          batch {size}: skipped ({res['skipped']})")
                continue
            print(
                f"          batch {size}: in-batch top-1 {_f(res['as_is']['top1'])}, loss {_f(res['as_is']['loss'])}; "
                f"change with the offset removed {-res['offset_cost']:+.3f}, fully aligned "
                f"{-res['misalignment_cost']:+.3f} (offset share {_f(res['offset_share'], 2)})"
            )
    hits = [t for t in r["associations"] if t["q"] < 0.1]
    print(f"subjects  {len(r['associations'])} tests over covariates: {', '.join(r['covariates']) or 'none'}")
    for t in hits[:8]:
        effect = "rho" if t["test"] == "spearman" else "eps2"
        print(f"          {t['measure']} ~ {t['covariate']}: {effect} {_f(t['effect'], 2)}, n {t['n']}, q {t['q']:.2g}")
    if not hits:
        print("          none at q < 0.1")


def print_comparison(table):
    names = list(next(iter(table.values())))
    print("\n== splits side by side (probe and retrieval rows size-matched) ==")
    print(f"{'':<42s}" + "".join(f"{n:>10s}" for n in names))
    for label, values in table.items():
        print(
            f"{label:<42s}"
            + "".join(f"{_f(values[n]) if not isinstance(values[n], int) else values[n]:>10}" for n in names)
        )


# ---------------------------------------------------------------------------
# Extraction from a checkpoint (torch) and feature files
# ---------------------------------------------------------------------------


def disable_augmentation(dataset):
    """Serve a training split with the validation preprocessing: no random affine or intensity draws."""
    if dataset._cache is not None:
        dataset._aug_transform = None
        return dataset
    from utils.utils import transforms

    _, deterministic = transforms(
        spacing=dataset.spacing,
        crop_margin=dataset.crop_margin,
        spatial_size=dataset.spatial_size,
        masks_from_disk=dataset.masks_from_disk,
        asymmetric_aug=dataset.asymmetric_aug,
        shared_brain_mask=dataset.shared_brain_mask,
    )
    dataset.monai_transform = deterministic
    return dataset


def scan_descriptors(images, masks, voxel_ml):
    """Per scan: brain-mask volume (mL) and blur along the blurriest axis.

    ``blur`` is that axis's mean absolute intensity step between neighbouring voxels, one voxel in
    from the mask edge, over the mean of the other two axes' steps: about 1 for an isotropic scan,
    lower along the slice axis of a thick-slice acquisition.
    """
    import torch
    import torch.nn.functional as F

    if masks.dim() == images.dim() - 1:
        masks = masks.unsqueeze(1)
    inside = (masks > 0.5).float()
    volume = inside.sum(dim=(1, 2, 3, 4)) * voxel_ml
    core = 1 - F.max_pool3d(1 - inside, 3, stride=1, padding=1)
    steps = []
    for axis in (2, 3, 4):
        size = images.shape[axis] - 1
        diff = (images.narrow(axis, 1, size) - images.narrow(axis, 0, size)).abs()
        both = core.narrow(axis, 1, size) * core.narrow(axis, 0, size)
        steps.append((diff * both).sum(dim=(1, 2, 3, 4)) / both.sum(dim=(1, 2, 3, 4)).clamp_min(1))
    steps = torch.stack(steps, 1)
    ratio = steps / ((steps.sum(1, keepdim=True) - steps) / 2).clamp_min(TINY)
    blur, axis = ratio.min(1)
    return volume, blur, axis, inside


def extract_checkpoint(args):
    """Encode each requested split's subjects through the saved model, with no augmentation."""
    import torch
    from torch.utils.data import DataLoader

    from eval.protocol.score_checkpoint import build_model, load_settings
    from training.main_conv_synthetic import make_real_dataset
    from utils.encoder_runtime import select_encoder_device

    run_dir = Path(args.run_dir).resolve()
    cfg = load_settings(run_dir)
    if cfg.get("dataset_name", "synthetic") == "synthetic":
        raise ValueError("This diagnostic requires an encoder-only real-data run")
    split = json.loads((run_dir / "split.json").read_text())
    for key in ("dataroot", "labels_path", "masks_dir", "cache_dir"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    checkpoint = (run_dir / args.checkpoint).resolve()
    checkpoint_bytes = checkpoint.read_bytes()
    device = select_encoder_device(args.device)
    state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    model = build_model(cfg, device, state)
    units = model.content_units()
    spacing = cfg.get("image_spacing", 2.0)
    voxel_ml = float(np.prod(spacing) if np.ndim(spacing) else spacing ** 3) / 1000
    metadata = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "device": device,
        "batch_size": args.batch_size,
        "augmentation": "off for every split",
        "resolved_settings": cfg,
    }
    splits = {}
    for name in args.splits:
        dataset = make_real_dataset(SimpleNamespace(**cfg), name)
        if name == "train":
            disable_augmentation(dataset)
        subjects = np.asarray([item["subject"] for item in dataset.items], dtype=str)
        if subjects.tolist() != split["subjects"][name]:
            raise ValueError(f"Resolved {name} subjects differ from split.json; check data paths and labels")
        content, descriptors = ([], []), {}
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        with torch.inference_mode():
            for step, batch in enumerate(loader, 1):
                images = torch.cat(batch["image"], dim=0).as_subclass(torch.Tensor).to(device)
                masks = torch.cat(batch["mask"], dim=0).as_subclass(torch.Tensor).to(device)
                b = len(batch["index"])
                codes = model(images, pool_only=True, n_views=2)[2][0][:, units].float().cpu().numpy()
                content[0].append(codes[:b])
                content[1].append(codes[b:])
                volume, blur, axis, inside = scan_descriptors(images.float(), masks.float(), voxel_ml)
                overlap = (inside[:b] * inside[b:]).sum(dim=(1, 2, 3, 4))
                dice = 2 * overlap / (inside[:b].sum(dim=(1, 2, 3, 4)) + inside[b:].sum(dim=(1, 2, 3, 4))).clamp_min(1)
                batch_desc = {
                    "mask_ml_t1": volume[:b],
                    "mask_ml_flair": volume[b:],
                    "mask_volume_ratio": volume[b:] / volume[:b].clamp_min(TINY),
                    "mask_dice": dice,
                    "blur_t1": blur[:b],
                    "blur_flair": blur[b:],
                    "blur_axis_t1": axis[:b],
                    "blur_axis_flair": axis[b:],
                }
                for key, value in batch_desc.items():
                    descriptors.setdefault(key, []).append(value.float().cpu().numpy())
                if step == 1 or step == len(loader) or step % 25 == 0:
                    print(f"Encoded {name} batch {step}/{len(loader)}", flush=True)
        splits[name] = {
            "content_t1": np.concatenate(content[0]),
            "content_flair": np.concatenate(content[1]),
            "subjects": subjects,
            "descriptors": {key: np.concatenate(values) for key, values in descriptors.items()},
        }
    return splits, metadata


def _split_spec(spec):
    """``val=path/to/file.npz`` -> ("val", path); a bare path (even one containing '=') -> (None, path)."""
    name, sep, path = spec.partition("=")
    return (name, path) if sep and re.fullmatch(r"\w+", name) else (None, spec)


def load_features(specs):
    """``split=path`` (or a bare path, whose split comes from its metadata, else val) per feature file.

    Reads this tool's files and encoder_modality_gap's features.npz alike.
    """
    splits, metadata, sources = {}, None, {}
    for spec in specs:
        name, path = _split_spec(spec)
        with np.load(path, allow_pickle=False) as features:
            meta = json.loads(str(features["metadata_json"]))
            name = name or meta.get("split", "val")
            if name in splits:
                raise ValueError(f"Two feature files for split {name!r}")
            splits[name] = {
                "content_t1": features["content_t1"],
                "content_flair": features["content_flair"],
                "subjects": features["subjects"],
                "descriptors": {k[len("desc_") :]: features[k] for k in features.files if k.startswith("desc_")},
            }
        metadata = metadata or meta
        sources[name] = str(Path(path).resolve())
    return splits, {**(metadata or {}), "features_source": sources}


def save_features(path, name, data, metadata):
    descriptors = {f"desc_{k}": v for k, v in data.get("descriptors", {}).items()}
    np.savez_compressed(
        path,
        content_t1=data["content_t1"],
        content_flair=data["content_flair"],
        subjects=np.asarray(data["subjects"], dtype=str),
        metadata_json=np.array(json.dumps({**metadata, "split": name}, allow_nan=False)),
        **descriptors,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="Encoder-only ADNI run with settings.json and split.json")
    source.add_argument(
        "--features",
        nargs="+",
        help="Re-analyse saved features without PyTorch: split=path pairs (this tool's features_<split>.npz "
        "or encoder_modality_gap's features.npz; a bare path is read as its saved split, else val)",
    )
    parser.add_argument("--checkpoint", default="model_best.pt", help="Filename within run-dir, or absolute path")
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=list(SPLITS), help="Subjects to encode (test stays unread)"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Encoding subjects per view")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729, help="Probe folds (as encoder_modality_gap) and all draws")
    parser.add_argument("--permutations", type=int, default=1000, help="Swap-null draws for offset and shape")
    parser.add_argument(
        "--probe-permutations", type=int, default=100, help="Swap-null draws for the probe ladder (0 skips it)"
    )
    parser.add_argument("--matched-draws", type=int, default=3, help="Size-matched subject draws for larger splits")
    parser.add_argument("--loss-batches", type=int, default=500, help="Random batches per loss estimate")
    parser.add_argument(
        "--loss-batch-sizes", type=int, nargs="+", help="Default: the run's batch size and four times it"
    )
    parser.add_argument("--tau", type=float, help="Override the run's InfoNCE temperature")
    parser.add_argument("--covariates", help="CSV with a subject column plus covariates (e.g. WMH volume, protocol)")
    parser.add_argument("--subject-column", default="Subject", help="Subject ID column of --covariates")
    parser.add_argument("--min-group", type=int, default=5, help="Smallest category kept in a group test")
    parser.add_argument("--out-dir", help="New report directory; default: timestamped directory beside the input")
    for flag in ("dataroot", "labels-path", "masks-dir", "cache-dir"):
        parser.add_argument(f"--{flag}", help="Override the saved cluster path")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.folds < 2 or args.matched_draws < 1 or args.loss_batches < 1:
        parser.error("Need positive batch-size, matched-draws and loss-batches, and at least two folds")
    if args.permutations < 0 or args.probe_permutations < 0:
        parser.error("Permutation counts cannot be negative")
    if args.loss_batch_sizes and min(args.loss_batch_sizes) < 2:
        parser.error("Loss batches need at least two pairs")
    if args.features and any(
        getattr(args, k) is not None for k in ("dataroot", "labels_path", "masks_dir", "cache_dir")
    ):
        parser.error("Data path overrides only apply with --run-dir")
    return args


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if args.features:
        base = Path(_split_spec(args.features[0])[1]).parent
    else:
        base = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else base / f"modality_gap_causes_{stamp}"
    if out.exists():
        raise FileExistsError(f"Report directory already exists: {out}")
    splits, metadata = load_features(args.features) if args.features else extract_checkpoint(args)
    report, rows = analyse(splits, metadata, args)
    import sklearn

    report.update(schema_version=1, created_at_utc=stamp, source=metadata, sklearn_version=sklearn.__version__)
    out.mkdir(parents=True, exist_ok=False)
    for name, data in splits.items():
        save_features(out / f"features_{name}.npz", name, data, metadata)
        fields = list(dict.fromkeys(key for row in rows[name] for key in row))
        with (out / f"subjects_{name}.csv").open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[name])
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, split_report in report["splits"].items():
        print_split(name, split_report)
    if report["comparison"]:
        print_comparison(report["comparison"])
    assumed = report["settings"]["assumed_from_recipe"]
    if assumed:
        print(f"\nLoss settings not in the features' metadata, recipe values assumed: {', '.join(assumed)}")
    print("Centring and whitening use the known modality label: they are diagnostics, not learned invariance.")
    print(f"Report: {out / 'report.json'}", flush=True)
    return report


if __name__ == "__main__":
    main()
