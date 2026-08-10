"""Trustworthy identifiability metrics for the synthetic multi-view CRL model.

Organised by the evaluation tiers described in ``identifiability_report.ipynb``:

  Tier 1 (theory-aligned, robust — match the Yao-2024 *block* identifiability
    claim): content informativeness, content/style separation, block-MCC.
  Tier 2 (useful, interpret with care): DCI informativeness, probe-free
    cross-view consistency, causal partial-R^2.
  Tier 3 (component-wise — NOT claimed by the theory, kept as a diagnostic):
    DCI disentanglement/completeness, per-channel MCC, factor-correlation ceiling.

Every function takes plain numpy arrays, so the metrics are unit-testable
without a trained model (see ``eval/test_identifiability_metrics.py`` / the
``__main__`` smoke test at the bottom).

Conventions
-----------
* ``repr``/``X``  : (N, D) representation array (already content- or style-sliced).
* ``factors``/``Z``: (N, K) ground-truth factor array, one column per scalar factor.
* ``groups``     : (D,) int array mapping each repr column to its channel id, so
  importance can be aggregated to the channel (the unit the Gumbel mask splits).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _spstats
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Core cross-validated probes
# --------------------------------------------------------------------------- #


def _make_regressor(kind, seed):
    if kind == "ridge":
        return RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0))
    if kind == "kernel":
        # RBF kernel ridge — the probe Yao et al. (ICLR 2024) evaluate with. alpha is
        # searched rather than fixed: on a pure-interaction target alpha=1.0 scores
        # R²=0.77 where alpha=0.01 scores 0.90, so a fixed default understates the
        # nonlinear ceiling — the one direction of error that would wrongly read as
        # "no nonlinear gain". gamma keeps its 1/n_features default (inputs are already
        # standardized); it mattered far less than alpha in testing.
        return GridSearchCV(KernelRidge(kernel="rbf"), {"alpha": [0.01, 0.1, 1.0]}, cv=3)
    if kind == "mlp":
        return MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=500, early_stopping=True, random_state=seed)
    raise ValueError(f"unknown regressor kind: {kind}")


def cv_probe_r2(X, y, n_splits=5, seeds=(0, 1, 2), kind="ridge", clip=False):
    """K-fold CV test-R^2 of predicting ``y`` from ``X``, averaged over seeds.

    Returns ``{"mean", "std", "per_seed"}`` where ``std`` is the spread of the
    per-seed mean R^2 (an honest stability band). ``clip`` floors negatives at 0.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if X.shape[1] == 0 or X.shape[0] < n_splits * 4:
        return {"mean": float("nan"), "std": float("nan"), "per_seed": []}

    per_seed = []
    for seed in seeds:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_scores = []
        for tr, te in kf.split(X):
            sc = StandardScaler().fit(X[tr])
            model = _make_regressor(kind, seed)
            model.fit(sc.transform(X[tr]), y[tr])
            pred = np.asarray(model.predict(sc.transform(X[te]))).reshape(len(te), -1)
            fold_scores.append(r2_score(y[te], pred, multioutput="variance_weighted"))
        per_seed.append(float(np.mean(fold_scores)))

    arr = np.array(per_seed)
    if clip:
        arr = np.clip(arr, 0.0, 1.0)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "per_seed": per_seed}


def _make_multi_regressor(kind, seed):
    """Multi-output twin of ``_make_regressor``.

    ``RidgeCV(alpha_per_target=True)`` fits a single GCV decomposition of ``X`` and
    picks a per-column alpha from it, so a batched fit reproduces the single-target
    probe column-by-column while decomposing ``X`` only once.  Other kinds fall back
    to ``_make_regressor`` (MLPRegressor is natively multi-output).
    """
    if kind == "ridge":
        return RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0), alpha_per_target=True)
    return _make_regressor(kind, seed)


def cv_probe_r2_multi(X, Y, n_splits=5, seeds=(0, 1, 2), kind="ridge"):
    """Per-column K-fold CV test-R^2 of predicting each column of ``Y`` from ``X``.

    Equivalent to calling :func:`cv_probe_r2` on every column of ``Y`` separately,
    but fits one multi-output regressor per (seed, fold) so the ``StandardScaler`` +
    ridge decomposition of ``X[train]`` is computed once and reused across all
    targets.  When many factors (and their permutation nulls) share the same ``X`` —
    the case in ``eval.run_dci_compare._score_block`` — this avoids re-decomposing
    ``X`` once per target, which is where that script's score phase spends its time.

    Returns ``{"mean": (T,), "std": (T,)}``: the per-seed mean R^2 averaged over
    seeds and its spread across seeds, matching ``cv_probe_r2`` column-wise.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    T = Y.shape[1]
    if X.shape[1] == 0 or X.shape[0] < n_splits * 4:
        return {"mean": np.full(T, float("nan")), "std": np.full(T, float("nan"))}

    per_seed = np.empty((len(seeds), T))
    for si, seed in enumerate(seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold = np.empty((n_splits, T))
        for fi, (tr, te) in enumerate(kf.split(X)):
            sc = StandardScaler().fit(X[tr])
            model = _make_multi_regressor(kind, seed)
            model.fit(sc.transform(X[tr]), Y[tr])
            pred = np.asarray(model.predict(sc.transform(X[te]))).reshape(len(te), T)
            fold[fi] = r2_score(Y[te], pred, multioutput="raw_values")
        per_seed[si] = fold.mean(axis=0)
    return {"mean": per_seed.mean(axis=0), "std": per_seed.std(axis=0)}


def cv_probe_acc(X, y, n_splits=5, seeds=(0, 1, 2)):
    """K-fold CV accuracy of a logistic probe (e.g. predicting view identity)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    if X.shape[1] == 0 or X.shape[0] < n_splits * 4 or len(np.unique(y)) < 2:
        return {"mean": float("nan"), "std": float("nan")}
    per_seed = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(sc.transform(X[tr]), y[tr])
            fold.append(accuracy_score(y[te], clf.predict(sc.transform(X[te]))))
        per_seed.append(float(np.mean(fold)))
    return {"mean": float(np.mean(per_seed)), "std": float(np.std(per_seed))}


# --------------------------------------------------------------------------- #
# Tier 1 — theory-aligned (block identifiability)
# --------------------------------------------------------------------------- #


def content_informativeness(content_repr, factors, factor_names, raw_repr=None, **kw):
    """Per-factor CV test-R^2 from the content block, with optional raw baseline.

    Returns a list of dicts: ``{factor, enc_mean, enc_std, raw_mean, delta}``.
    ``delta = enc - raw`` is the encoder's credit above trivial patch intensity.
    """
    rows = []
    for j, name in enumerate(factor_names):
        enc = cv_probe_r2(content_repr, factors[:, j], **kw)
        raw = cv_probe_r2(raw_repr, factors[:, j], **kw) if raw_repr is not None else {"mean": float("nan")}
        rows.append(
            {
                "factor": name,
                "enc_mean": enc["mean"],
                "enc_std": enc["std"],
                "raw_mean": raw["mean"],
                "delta": enc["mean"] - raw["mean"] if raw_repr is not None else float("nan"),
            }
        )
    return rows


def parents_from_adjacency(adjacency):
    """adjacency[i, j] True  =>  i is a parent of j. Returns {j: [parents]}."""
    adjacency = np.asarray(adjacency).astype(bool)
    return {j: np.where(adjacency[:, j])[0].tolist() for j in range(adjacency.shape[1])}


def partial_r2_vs_parents(content_repr, z_content, adjacency, factor_names, **kw):
    """How much of each factor's *own* variation the content captures, beyond parents.

    For factor ``j`` with SCM parents ``Pa(j)``, residualise ``z_j`` on its
    parents (linear), then probe that residual from the content block. A high
    partial-R^2 means the encoder isn't just reading off ``z_j``'s parents.
    """
    parents = parents_from_adjacency(adjacency)
    rows = []
    for j, name in enumerate(factor_names):
        pa = parents.get(j, [])
        yj = z_content[:, j]
        if pa:
            resid = yj - LinearRegression().fit(z_content[:, pa], yj).predict(z_content[:, pa])
        else:
            resid = yj
        full = cv_probe_r2(content_repr, yj, **kw)
        part = cv_probe_r2(content_repr, resid, **kw)
        rows.append(
            {
                "factor": name,
                "n_parents": len(pa),
                "full_r2": full["mean"],
                "partial_r2": part["mean"],
            }
        )
    return rows


def content_sufficiency(content_repr, all_repr, shared_factors, factor_names, **kw):
    """Does adding style channels improve shared-factor prediction? Δ≈0 ⇒ sufficient."""
    rows = []
    for j, name in enumerate(factor_names):
        c = cv_probe_r2(content_repr, shared_factors[:, j], **kw)
        a = cv_probe_r2(all_repr, shared_factors[:, j], **kw)
        rows.append(
            {"factor": name, "content_r2": c["mean"], "content_style_r2": a["mean"], "delta": a["mean"] - c["mean"]}
        )
    return rows


def view_invariance(content_v1, content_v2, style_v1, style_v2, all_v1=None, all_v2=None, **kw):
    """Predict view identity (v1=0 / v2=1) from content / style / all.

    content→view ≈ chance (0.5) ⇒ content is view-invariant; style→view high ⇒
    style carries the view-specific signal. Returns dict of accuracies (+chance).
    """
    n = content_v1.shape[0]
    labels = np.array([0] * n + [1] * n)
    out = {"chance": 0.5}
    out["content_acc"] = cv_probe_acc(np.vstack([content_v1, content_v2]), labels, **kw)["mean"]
    out["style_acc"] = cv_probe_acc(np.vstack([style_v1, style_v2]), labels, **kw)["mean"]
    if all_v1 is not None and all_v2 is not None:
        out["all_acc"] = cv_probe_acc(np.vstack([all_v1, all_v2]), labels, **kw)["mean"]
    return out


def block_mcc(X, Z, kind="ridge", seeds=(0, 1, 2), n_splits=5):
    """Block-/affine-identifiability MCC: fit a readout X→Z, Hungarian-match
    predicted↔true dims on |Pearson|, average matched |corr| (CV, over seeds).

    This is the *correct* MCC for block identifiability — it allows the
    invertible transform the theory permits (unlike per-channel MCC).

    ``mean`` is an unweighted average over factors, so it hides sign splits: a
    representation that concentrates onto one dominant factor while losing several
    mid-scale ones reads as a uniform decline.  The per-factor breakdown that makes such a
    split visible is computed here anyway (``corr[row, col]`` is one matched |corr| per
    factor) and used to be discarded by the ``.mean()``, so these extra keys are free:

        per_factor      matched |corr| per TRUE factor, averaged over folds and seeds
        per_factor_std  across-seed spread of the above — per-factor curves are noisier
                        than the mean over 9, so never read one without it
        per_factor_diag UNMATCHED |corr| between factor j's own predictor and factor j.
                        Immune to assignment permutation; read it when per_factor jumps.
        assignment_identity
                        fraction of factors matched to their own predictor (1.0 = clean).
                        Below 1.0 the Hungarian assignment has permuted — likely between
                        correlated factors under an SCM — and per_factor entries have
                        swapped for reasons that are not about the representation.
    """
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    if X.shape[1] == 0 or Z.shape[1] == 0 or X.shape[0] < n_splits * 4:
        nanvec = np.full(max(Z.shape[1], 1), float("nan"))
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "per_factor": nanvec,
            "per_factor_std": nanvec.copy(),
            "per_factor_diag": nanvec.copy(),
            "assignment_identity": float("nan"),
        }

    n_factors = Z.shape[1]
    per_seed, per_seed_factor, per_seed_diag, per_seed_ident = [], [], [], []
    for seed in seeds:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_mcc, fold_factor, fold_diag, fold_ident = [], [], [], []
        for tr, te in kf.split(X):
            sc = StandardScaler().fit(X[tr])
            model = _make_regressor(kind, seed)
            model.fit(sc.transform(X[tr]), Z[tr])
            Zhat = np.asarray(model.predict(sc.transform(X[te]))).reshape(len(te), -1)
            corr = _abs_corr_matrix(Zhat, Z[te])
            row, col = linear_sum_assignment(-corr)
            fold_mcc.append(float(corr[row, col].mean()))

            # Index by TRUE factor (``col``), not by predicted dim, so a permuted
            # assignment still lands each score under the factor it describes.
            pf = np.full(n_factors, np.nan)
            pf[col] = corr[row, col]
            fold_factor.append(pf)
            dg = np.full(n_factors, np.nan)
            diag = np.diag(corr)
            dg[: len(diag)] = diag
            fold_diag.append(dg)
            fold_ident.append(float(np.mean(row == col)))
        per_seed.append(float(np.mean(fold_mcc)))
        per_seed_factor.append(np.nanmean(np.stack(fold_factor), axis=0))
        per_seed_diag.append(np.nanmean(np.stack(fold_diag), axis=0))
        per_seed_ident.append(float(np.mean(fold_ident)))

    pf_arr = np.stack(per_seed_factor)
    return {
        "mean": float(np.mean(per_seed)),
        "std": float(np.std(per_seed)),
        "per_factor": np.nanmean(pf_arr, axis=0),
        "per_factor_std": np.nanstd(pf_arr, axis=0),
        "per_factor_diag": np.nanmean(np.stack(per_seed_diag), axis=0),
        "assignment_identity": float(np.mean(per_seed_ident)),
    }


def _abs_corr_matrix(A, B):
    """|Pearson| between every column of A (rows) and B (cols)."""
    dA, dB = A.shape[1], B.shape[1]
    out = np.zeros((dA, dB))
    for i in range(dA):
        ai = A[:, i]
        for j in range(dB):
            bj = B[:, j]
            if ai.std() < 1e-8 or bj.std() < 1e-8:
                out[i, j] = 0.0
            else:
                out[i, j] = abs(np.corrcoef(ai, bj)[0, 1])
    return np.nan_to_num(out)


# --------------------------------------------------------------------------- #
# Tier 2 — useful, interpret with care
# --------------------------------------------------------------------------- #


def dci_informativeness_sections(sections, **kw):
    """Mean per-factor CV R^2 for each repr→factor section (DCI informativeness).

    ``sections``: dict ``label -> (repr_arr, factor_arr, factor_names)``. Returns
    ``label -> {"mean_r2", "per_factor"}`` with negatives clipped (informativeness
    is a [0,1] readout quality, matched pairs should be high, cross pairs low).
    """
    out = {}
    for label, (repr_arr, factor_arr, names) in sections.items():
        if repr_arr is None or repr_arr.shape[1] == 0 or factor_arr.shape[1] == 0:
            out[label] = {"mean_r2": float("nan"), "per_factor": {}}
            continue
        per = {}
        for j, name in enumerate(names):
            per[name] = cv_probe_r2(repr_arr, factor_arr[:, j], clip=True, **kw)["mean"]
        vals = [v for v in per.values() if not np.isnan(v)]
        out[label] = {"mean_r2": float(np.mean(vals)) if vals else float("nan"), "per_factor": per}
    return out


def crossview_consistency(content_v1, content_v2, style_v1, style_v2):
    """Probe-free disentanglement signal from paired views (no decoder needed).

    Content of the *same* subject across views should be ~identical (high cosine);
    style should differ more. Reports within-subject cosine for content vs style,
    plus a cross-subject baseline for content (should be much lower than within).
    """

    def _cos(a, b):
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return (a * b).sum(1)

    n = content_v1.shape[0]
    perm = np.roll(np.arange(n), 1)  # mismatched pairing for the baseline
    return {
        "content_within_subject": float(np.mean(_cos(content_v1, content_v2))),
        "content_cross_subject": float(np.mean(_cos(content_v1, content_v2[perm]))),
        "style_within_subject": float(np.mean(_cos(style_v1, style_v2))) if style_v1.shape[1] else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Tier 3 — component-wise (aspirational; NOT guaranteed by the theory)
# --------------------------------------------------------------------------- #


def _grouped_perm_importance(model, X, y, groups, n_repeats, rng):
    base = r2_score(y, model.predict(X))
    uniq = np.unique(groups)
    imp = np.zeros(len(uniq))
    for gi, g in enumerate(uniq):
        cols = np.where(groups == g)[0]
        drop = 0.0
        for _ in range(n_repeats):
            Xp = X.copy()
            order = rng.permutation(X.shape[0])
            Xp[:, cols] = X[np.ix_(order, cols)]
            drop += base - r2_score(y, model.predict(Xp))
        imp[gi] = max(0.0, drop / n_repeats)
    return imp


def importance_matrix(repr_arr, factors, groups, method="rf_perm", train_ratio=0.8, n_repeats=3, seed=0):
    """Channel×factor importance matrix from a regularized probe.

    method: ``"rf_perm"`` (grouped permutation, correlation-robust), ``"l1"``
    (|LassoCV coef| summed per channel). Importance is always at the channel
    granularity (``groups``), so D/C are computed on the unit the mask splits on.
    """
    rng = np.random.RandomState(seed)
    N = repr_arr.shape[0]
    split = int(N * train_ratio)
    sc = StandardScaler().fit(repr_arr[:split])
    Xtr, Xte = sc.transform(repr_arr[:split]), sc.transform(repr_arr[split:])
    Ytr, Yte = factors[:split], factors[split:]
    n_groups = len(np.unique(groups))
    R = np.zeros((n_groups, factors.shape[1]))
    for j in range(factors.shape[1]):
        if method == "l1":
            m = LassoCV(n_alphas=20, max_iter=20000, tol=1e-3, random_state=seed).fit(Xtr, Ytr[:, j])
            fi = np.abs(m.coef_)
            R[:, j] = np.array([fi[groups == g].sum() for g in range(n_groups)])
        else:  # rf_perm
            m = RandomForestRegressor(n_estimators=200, max_features="sqrt", random_state=seed).fit(Xtr, Ytr[:, j])
            R[:, j] = _grouped_perm_importance(m, Xte, Yte[:, j], groups, n_repeats, rng)
    return R


def dci_DC(importance):
    """Disentanglement + completeness from a (n_codes, n_factors) importance matrix.

    NB: component-wise metrics — only meaningful for *independent* factors and an
    axis-aligned model. Read alongside ``factor_correlation_ceiling``.
    """
    R = np.asarray(importance, dtype=np.float64)
    n_codes, n_factors = R.shape
    comp = 1.0 - _spstats.entropy(R + 1e-11, base=max(n_codes, 2), axis=0)
    dis_pc = 1.0 - _spstats.entropy(R.T + 1e-11, base=max(n_factors, 2), axis=0)
    w = R.sum(axis=1)
    tot = w.sum()
    disent = float(np.sum(dis_pc * w / tot)) if tot > 0 else 0.0
    return {"disentanglement": disent, "completeness": float(np.mean(comp))}


def per_channel_mcc(channel_repr, Z):
    """Strict axis-aligned MCC: |corr| between raw channels and sources, Hungarian.

    No readout is fitted — this asks whether *one channel ↔ one factor*, which the
    theory does NOT guarantee. Diagnostic only.
    """
    channel_repr = np.asarray(channel_repr, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    if channel_repr.shape[1] == 0 or Z.shape[1] == 0:
        return float("nan")
    corr = _abs_corr_matrix(channel_repr, Z)
    row, col = linear_sum_assignment(-corr)
    return float(corr[row, col].mean())


def factor_correlation_ceiling(factors):
    """Mean |Pearson| among GT factor columns — the dependence DCI-D can't exceed.

    High value ⇒ factors are correlated (e.g. a causal SCM) ⇒ DCI disentanglement
    is bounded below 1 *regardless of the model*, so don't read D as a failure.
    """
    factors = np.asarray(factors, dtype=np.float64)
    k = factors.shape[1]
    if k < 2:
        return 0.0
    C = np.abs(np.corrcoef(factors, rowvar=False))
    off = C[~np.eye(k, dtype=bool)]
    return float(np.nanmean(off))


# --------------------------------------------------------------------------- #
# Smoke test (no model required): run `python -m eval.identifiability_metrics`
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    N, C, K = 200, 12, 5
    # factor 0 driven by channel 0; factor 1 by channels 1-2; rest noise.
    Z = rng.randn(N, K)
    X = rng.randn(N, C) * 0.3
    X[:, 0] += Z[:, 0]
    X[:, 1] += 0.7 * Z[:, 1]
    X[:, 2] += 0.7 * Z[:, 1]
    names = [f"f{i}" for i in range(K)]
    raw = rng.randn(N, 8)
    adj = np.triu(np.ones((K, K), dtype=bool), 1)  # chain-ish DAG

    print("content_informativeness:", content_informativeness(X, Z, names, raw_repr=raw)[:2])
    print("partial_r2_vs_parents  :", partial_r2_vs_parents(X, Z, adj, names)[1])
    print("content_sufficiency    :", content_sufficiency(X, X, Z, names)[0])
    print(
        "view_invariance        :",
        view_invariance(X[:100], X[:100] + 0.01 * rng.randn(100, C), X[:100, :3], X[:100, :3] + 1.0),
    )
    print("block_mcc              :", block_mcc(X, Z))
    groups = np.repeat(np.arange(C), 1)
    R = importance_matrix(X, Z, groups, method="l1")
    print("importance_matrix shape:", R.shape, "| dci_DC:", dci_DC(R))
    print("per_channel_mcc        :", round(per_channel_mcc(X, Z), 3))
    print("factor_corr_ceiling    :", round(factor_correlation_ceiling(Z), 3))
    print("dci_informativeness    :", dci_informativeness_sections({"c->c": (X, Z, names)})["c->c"]["mean_r2"])
    print(
        "crossview_consistency  :",
        crossview_consistency(X[:100], X[:100] + 0.01 * rng.randn(100, C), X[:100, :3], X[:100, :3] + 1.0),
    )
    print("\nALL SMOKE TESTS RAN OK")
