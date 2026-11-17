"""Stage 42: the concealed-factor benchmark of paper/factor-discovery-framework.tex on a frozen code. No training.

Some causal factors are supplied as anchors, observed variables with labels on training subjects only (by default the
ones FreeSurfer measures: brain, ventricle, thickness, temporal). The rest are concealed (burden, lr, sulcal), and so
is the lesion centre, shared content outside the causal graph. Per run, on its trained code and on its untrained twin
(calibrated as causal_bundle.py does):

  1. Anchored slots: a ridge readout of each anchor from each view's code (--anchor-noise adds measurement noise to
     the training labels, as a fraction of each anchor's SD).
  2. The novel block, label-free: each view's code minus its least-squares fit on the anchored slots and their squares
     and products, and the two views' residuals compared by CCA, both on training subjects. Factors are counted by
     greedy deflation: take the top canonical pair while its agreement on validation subjects beats both
     --min-agreement and the 95th percentile of a null that shuffles the pairing of validation subjects between the
     views; count it unless it is redundant (below); subtract from the code its fit on the slots and every direction
     taken so far, with their squares and products; repeat.
     - The fit is exact. It leaves the code no variance along a taken direction, where a ridge fit, at any penalty,
       leaves a remnant that is all that is left there. CCA ignores scale, so it finds the remnant again as shared as
       the direction itself, and dropping it as redundant drags real factors out with it.
     - Squares and products are taken of the inputs clipped to their training range. Unclipped, a held-out subject
       beyond that range is extrapolated quadratically and the error compounds over the steps.
     - Higher powers (a cube of brain size, say) and the SCM's kinks survive a degree-2 subtraction and are shared
       too. So a direction is counted only if a degree-3 polynomial ridge of the OTHER view's slots and counted factors
       predicts it with held-out R2 below --max-redundancy; otherwise it is subtracted uncounted. Dropped directions
       are left out of that prediction: one that carried part of a real factor would make the factor look redundant.
     - Unpredictable leftovers still agree up to .50 on the controls (tests/test_factoriser.py), where real factors
       agree at .56 and above. Hence --min-agreement .5.
     - Known limit: canonical pairs of near-equal agreement can mix a factor with a leftover. Then a leftover is
       counted as one more factor (2 of 30 toy controls), or a factor loses part of itself to a dropped mixture (a
       code like the bank's with burden read at error .35: burden's own part .31 and .40 in the block on 2 of 15
       seeds, .56-.61 from the whole code). The report prints both readings for every concealed target.
  3. Coordinates inside the block: the CCA order, and FastICA (item 33: separates non-Gaussian sources only). Each
     component is named after one concealed target (the concealed factors' own parts, their exogenous noise inverted
     from the SCM, and the lesion centre) by one Hungarian match on |corr|, against random rotations of the block.
     Labels are used for scoring only.
  4. The graph: PC (Fisher-Z) on the anchored slots and the block's components, --rows test subjects per resampling,
     --resamples resamplings. The components are residuals of the anchors, so they are uncorrelated with the anchors
     and with each other by construction, and PC cannot connect them to anything: the graph is the anchors' graph
     with the concealed factors latent. Its truth is what PC finds among the anchors with infinite data, which has
     edges the SCM lacks where a concealed factor mediates or confounds two anchors (ventricle - temporal through
     burden with the default anchors). Edges touching a component are false; their chance rate is printed beside them.

Ceilings on the test subjects' true variables (the same for the trained code and its twin): PC on the true anchors,
scored the same way, and PC on all seven true factors against the SCM's skeleton. The second is the graph that
recovering the concealed factors whole, not as residuals of the anchors, would make reachable.

    python out/burden_feasibility_20261102/factoriser.py OUT_JSON RUN [RUN ...]
        [--anchors brain ventricle thickness temporal] [--anchor-noise 0] [--train-subjects N]

Runs already in OUT_JSON at the same step and with the same settings are reused; use one OUT_JSON per anchor set.
"""

import argparse
import itertools
import json
import warnings
from pathlib import Path

import bank as B
import batch_price
import causal_bundle
import code_audit
import numpy as np
import torch
import warm
from scipy.optimize import linear_sum_assignment
from scipy.stats import ortho_group
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

COLUMNS = list(B.CAUSAL_FACTORS)
FACTORS = [B.CAUSAL_FACTORS[d] for d in COLUMNS]
ANCHORS = ("brain", "ventricle", "thickness", "temporal")
SPLITS = ("train", "val", "test")
ALPHAS = np.logspace(-3, 3, 13)
NULL_DRAWS, ROTATIONS, PC_ALPHA = 200, 10, 0.05
SETTINGS = ("anchors", "anchor_noise", "max_factors", "min_agreement", "max_redundancy", "rows", "resamples")


def load(run, device, trained, train_subjects=None):
    """Each split's (2, N, units) code and seven causal factors; for test subjects also the factors' own parts and the
    lesion centre (scoring only)."""
    report, model, pool, residual, step = code_audit.load_run(run, device, trained=trained)
    a = report["args"]
    if pool is None:
        raise SystemExit(f"{run.name}: the code comes from TailCode's head, so the run needs --pool tail or gapnorm")
    root = a["bank"]
    if not Path(f"{root}/causal_adj.npy").exists():
        raise SystemExit(f"{root} has no causal_adj.npy: the benchmark needs a causal bank")
    if not trained:
        causal_bundle.calibrate(model, pool, warm.Bank(root, "train", a["t1key"], residual=residual), device)
    out = {"step": step, "bank": root, "units": pool.main}
    for split in SPLITS:
        n = {"train": train_subjects, "val": None, "test": int(a.get("test_subjects", 1000))}[split]
        bank = warm.Bank(root, split, a["t1key"], n, residual)
        code = np.stack(warm.features(model, bank, device, pool=pool)["code"])[..., : pool.main]
        out[split] = {"code": code.astype(np.float64), "factors": bank.zc[:, COLUMNS]}
        if split == "test":
            cells = np.load(f"{root}/test_cells.npy", mmap_mode="r")[: len(bank.vox)]
            out[split]["own"] = causal_bundle.own_noise(bank.zc, root, COLUMNS)
            out[split]["position"] = batch_price.lesion_centroid(cells)
    return out


def load_adj(root):
    return np.load(f"{root}/causal_adj.npy")[np.ix_(COLUMNS, COLUMNS)].astype(bool)


class Clip(BaseEstimator, TransformerMixin):
    """Clips each column to its training range, so that polynomial terms are not extrapolated."""

    def fit(self, x, y=None):
        self.low_, self.high_ = x.min(0), x.max(0)
        return self

    def transform(self, x):
        return np.clip(x, self.low_, self.high_)


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def poly_ridge(degree=2):
    return make_pipeline(
        Clip(), StandardScaler(), PolynomialFeatures(degree, include_bias=False), StandardScaler(), RidgeCV(ALPHAS)
    )


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def cv_predict(x, y):
    return cross_val_predict(ridge(), x, y, cv=KFold(5, shuffle=True, random_state=0))


def anchored_slots(data, anchors, noise, rng):
    """Step 1: per view, a ridge readout of each anchor fitted on training subjects; returns the slots per split
    (2, N, L) and each anchor's test R2 per view."""
    index = [FACTORS.index(name) for name in anchors]
    truth = data["train"]["factors"][:, index]
    labels = truth + noise * truth.std(0) * rng.standard_normal(truth.shape)
    slots = {split: np.zeros((2, len(data[split]["code"][0]), len(index))) for split in SPLITS}
    scores = np.zeros((2, len(index)))
    for v in range(2):
        for j in range(len(index)):
            model = ridge().fit(data["train"]["code"][v], labels[:, j])
            for split in SPLITS:
                slots[split][v][:, j] = model.predict(data[split]["code"][v])
            scores[v, j] = r2(slots["test"][v][:, j], data["test"]["factors"][:, index[j]])
    return slots, scores


def quadratic(x, reference):
    """``x`` standardized by ``reference`` (the training inputs), then the squares and products of its columns clipped
    to the reference's range."""
    mean, sd = reference.mean(0), reference.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    z = (x - mean) / sd
    clipped = np.clip(z, (reference.min(0) - mean) / sd, (reference.max(0) - mean) / sd)
    i, j = np.triu_indices(x.shape[1])
    return np.column_stack([z, clipped[:, i] * clipped[:, j]])


def residuals(data, inputs):
    """Each view's code minus its least-squares fit on ``inputs`` (per split, (2, N, k)) and their squares and
    products, fitted on training subjects; per split (2, N, units)."""
    out = {split: np.zeros_like(data[split]["code"]) for split in SPLITS}
    for v in range(2):
        reference = inputs["train"][v]
        model = LinearRegression().fit(quadratic(reference, reference), data["train"]["code"][v])
        for split in SPLITS:
            out[split][v] = data[split]["code"][v] - model.predict(quadratic(inputs[split][v], reference))
    return out


def canonical_pairs(a, b, ridge_fraction=1e-3):
    """All canonical pairs of training matrices ``a`` and ``b``: the weights for each (one column per pair) and the
    canonical correlations, largest first."""
    a, b = a - a.mean(0), b - b.mean(0)

    def inverse_root(x):
        cov = x.T @ x / len(x)
        cov += ridge_fraction * np.trace(cov) / len(cov) * np.eye(len(cov))
        values, vectors = np.linalg.eigh(cov)
        return (vectors / np.sqrt(values)) @ vectors.T

    wa, wb = inverse_root(a), inverse_root(b)
    u, s, vt = np.linalg.svd(wa @ (a.T @ b / len(a)) @ wb)
    return wa @ u[:, : len(s)], wb @ vt[: len(s)].T, s


def held_out_agreement(x, y, rng):
    """The correlation of two validation projections, and the 95th percentile of |corr| with the pairing shuffled."""
    observed = float(np.corrcoef(x, y)[0, 1])
    null = [abs(np.corrcoef(x, y[rng.permutation(len(y))])[0, 1]) for _ in range(NULL_DRAWS)]
    return observed, float(np.percentile(null, 95))


def redundancy(inputs, projections):
    """Held-out R2 of each view's candidate direction from a degree-3 polynomial ridge of the OTHER view's ``inputs``
    (fitted on training subjects, scored on validation subjects), averaged over views."""
    scores = []
    for v in range(2):
        model = poly_ridge(3).fit(inputs["train"][1 - v], projections["train"][v])
        scores.append(r2(model.predict(inputs["val"][1 - v]), projections["val"][v]))
    return float(np.mean(scores))


def novel_block(data, slots, max_factors, min_agreement, rng, max_redundancy=0.2):
    """Step 2: greedy deflation. Each step takes the top canonical pair of the two views' residuals while it agrees on
    validation subjects. It is counted unless the other view's anchored slots and counted factors predict it at
    ``max_redundancy`` or above; either way it is subtracted. Returns the factors' components per split (2, N, K) and
    every step's record."""
    taken = {split: np.zeros((2, len(slots[split][0]), 0)) for split in SPLITS}
    factors = {split: np.zeros((2, len(slots[split][0]), 0)) for split in SPLITS}
    total = min(data["train"]["code"][v].var(0).sum() for v in range(2))
    records = []
    for _ in range(3 * max_factors):
        left = residuals(data, {split: np.concatenate([slots[split], taken[split]], 2) for split in SPLITS})
        if min(left["train"][v].var(0).sum() for v in range(2)) < 1e-9 * total:
            records.append({"shared": False, "kept": False, "empty": True})
            break
        wa, wb, rho = canonical_pairs(left["train"][0], left["train"][1])
        top = {split: np.stack([left[split][0] @ wa[:, 0], left[split][1] @ wb[:, 0]]) for split in SPLITS}
        agreement, null = held_out_agreement(top["val"][0], top["val"][1], rng)
        record = {"train_rho": float(rho[0]), "val_agreement": agreement, "val_null95": null, "kept": False}
        record["shared"] = agreement > max(null, min_agreement)
        records.append(record)
        if not record["shared"]:
            break
        counted = {split: np.concatenate([slots[split], factors[split]], 2) for split in SPLITS}
        record["redundancy"] = redundancy(counted, top)
        record["kept"] = record["redundancy"] < max_redundancy
        for split in SPLITS:
            taken[split] = np.concatenate([taken[split], top[split][..., None]], 2)
            if record["kept"]:
                factors[split] = np.concatenate([factors[split], top[split][..., None]], 2)
        if factors["train"].shape[2] == max_factors:
            break
    return factors, records


def ica_block(block, seed=0):
    """Step 3: FastICA per view, fitted on training subjects, on the block's components."""
    k = block["train"].shape[2]
    out = {split: np.zeros_like(block[split]) for split in SPLITS}
    if k == 0:
        return out
    for v in range(2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = FastICA(k, whiten="unit-variance", max_iter=2000, tol=1e-5, random_state=seed)
            model.fit(block["train"][v])
        for split in SPLITS:
            out[split][v] = model.transform(block[split][v])
    return out


def concealed_targets(split_data, anchors):
    """The concealed factors' own parts and the lesion centre, with their names."""
    concealed = [name for name in FACTORS if name not in anchors]
    own = split_data["own"][:, [FACTORS.index(name) for name in concealed]]
    names = [f"{name} own" for name in concealed] + ["lesion x", "lesion y", "lesion z"]
    return np.column_stack([own, split_data["position"]]), names


def name_components(components, targets):
    """One Hungarian match on |corr|: for each component, the target index it is named after (-1 if none) and |r|."""
    k = components.shape[1]
    names, strength = -np.ones(k, dtype=int), np.zeros(k)
    if k == 0:
        return names, strength
    corr = np.abs(np.corrcoef(components.T, targets.T)[:k, k:])
    rows, cols = linear_sum_assignment(-corr)
    names[rows], strength[rows] = cols, corr[rows, cols]
    return names, strength


def d_separated(adj, x, y, given):
    """Whether ``x`` and ``y`` are d-separated by ``given`` in the DAG ``adj`` (parent x child): disconnected in the
    moral graph of the ancestors of {x, y} and ``given`` once ``given`` is removed."""
    given = {int(g) for g in given}
    keep = {int(x), int(y)} | given
    stack = list(keep)
    while stack:
        for parent in np.flatnonzero(adj[:, stack.pop()]).tolist():
            if parent not in keep:
                keep.add(parent)
                stack.append(parent)
    links = {node: set() for node in keep}
    for child in keep:
        parents = [p for p in np.flatnonzero(adj[:, child]).tolist() if p in keep]
        for p in parents:
            links[p].add(child)
            links[child].add(p)
        for p, q in itertools.combinations(parents, 2):
            links[p].add(q)
            links[q].add(p)
    seen, stack = {int(x)}, [int(x)]
    while stack:
        for node in links[stack.pop()] - seen - given:
            if node == y:
                return False
            seen.add(node)
            stack.append(node)
    return True


def oracle_skeleton(adj, observed):
    """The adjacencies PC finds among ``observed`` (indices into FACTORS) with infinite data and the rest of the DAG
    latent: the pairs that no subset of the other observed variables d-separates."""
    edges = set()
    for x, y in itertools.combinations(observed, 2):
        others = [o for o in observed if o not in (x, y)]
        subsets = itertools.chain.from_iterable(itertools.combinations(others, k) for k in range(len(others) + 1))
        if not any(d_separated(adj, x, y, s) for s in subsets):
            edges.add(frozenset((FACTORS[x], FACTORS[y])))
    return edges


def anchor_truth(anchors, adj):
    """What PC can find among the anchors with the concealed factors latent: the skeleton, the edges in it the SCM
    lacks (made by a concealed mediator or common cause), and the SCM's edges that touch a concealed factor."""
    skeleton = oracle_skeleton(adj, [FACTORS.index(name) for name in anchors])
    pairs = list(zip(*np.nonzero(adj)))
    direct = {frozenset((FACTORS[i], FACTORS[j])) for i, j in pairs}
    hidden = sorted(f"{FACTORS[i]} -> {FACTORS[j]}" for i, j in pairs if not {FACTORS[i], FACTORS[j]} <= set(anchors))
    return skeleton, skeleton - direct, hidden


def edge_names(edges):
    return sorted(" - ".join(sorted(edge)) for edge in edges)


def pc_edges(data, labels):
    from causallearn.search.ConstraintBased.PC import pc

    graph = pc(data, PC_ALPHA, "fisherz", show_progress=False).G.graph
    k = len(labels)
    return {frozenset((labels[i], labels[j])) for i in range(k) for j in range(i + 1, k) if graph[i, j] != 0}


def graph_scores(data, labels, truth, block, rows, resamples, rng):
    """PC over resamplings of ``rows`` subjects: skeleton F1 against ``truth``, the share of graphs holding each edge
    found, and per graph the false edges that touch a node in ``block`` and the other false edges."""
    f1s, found, touching, false = [], {}, [], []
    for _ in range(resamples):
        chosen = rng.choice(len(data), min(rows, len(data)), replace=False)
        edges = pc_edges(data[chosen], labels)
        hit = len(edges & truth)
        f1s.append(0.0 if hit == 0 else 2 * hit / (len(edges) + len(truth)))
        for key in edge_names(edges):
            found[key] = found.get(key, 0) + 1
        touching.append(sum(1 for edge in edges if edge & block))
        false.append(sum(1 for edge in edges - truth if not edge & block))
    return {
        "f1_mean": float(np.mean(f1s)),
        "f1_sd": float(np.std(f1s)),
        "edges_found": {key: count / resamples for key, count in sorted(found.items())},
        "block_edges_mean": float(np.mean(touching)),
        "false_edges_mean": float(np.mean(false)),
    }


def assess_codes(data, a, seed=0):
    rng = np.random.default_rng(seed)
    truth, induced, hidden = anchor_truth(a.anchors, load_adj(data["bank"]))
    slots, anchor_r2 = anchored_slots(data, a.anchors, a.anchor_noise, rng)
    block, records = novel_block(data, slots, a.max_factors, a.min_agreement, rng, a.max_redundancy)
    targets, target_names = concealed_targets(data["test"], a.anchors)
    k = block["test"].shape[2]
    labels = [f"c{i + 1}" for i in range(k)]
    versions = {"cca": block, "ica": ica_block(block, seed)}
    out = {
        "anchor_r2": anchor_r2.tolist(),
        "count": k,
        "deflation": records,
        "target_names": target_names,
        "truth": edge_names(truth),
        "induced": edge_names(induced),
        "hidden": hidden,
        "views": [],
    }
    for v in range(2):
        code, comps = data["test"]["code"][v], block["test"][v]
        view = {
            "full_code_r2": [r2(cv_predict(code, y), y) for y in targets.T],
            "block_r2": [r2(cv_predict(comps, y), y) if k else 0.0 for y in targets.T],
            "names": {},
        }
        for version, x in versions.items():
            names, strength = name_components(x["test"][v], targets)
            view["names"][version] = {
                "names": [target_names[i] if i >= 0 else "unmatched" for i in names],
                "matched_r": strength.tolist(),
            }
        standard = (comps - comps.mean(0)) / comps.std(0) if k else comps
        null = [
            name_components(standard @ ortho_group.rvs(k, random_state=seed * 1000 + r), targets)[1].mean()
            for r in range(ROTATIONS if k > 1 else 0)
        ]
        view["rotation_null_r"] = float(np.mean(null)) if null else None
        view["graph"] = graph_scores(
            np.column_stack([slots["test"][v], comps]),
            list(a.anchors) + labels,
            truth,
            set(labels),
            a.rows,
            a.resamples,
            rng,
        )
        out["views"].append(view)
    return out


def ceilings(data, a, seed=1):
    """PC on the test subjects' true anchors (against the anchors' truth) and on all seven true factors (against the
    SCM's skeleton)."""
    rng = np.random.default_rng(seed)
    adj = load_adj(data["bank"])
    truth, _, _ = anchor_truth(a.anchors, adj)
    whole = oracle_skeleton(adj, list(range(len(FACTORS))))
    factors = data["test"]["factors"]
    anchors = factors[:, [FACTORS.index(name) for name in a.anchors]]
    return {
        "anchors": graph_scores(anchors, list(a.anchors), truth, set(), a.rows, a.resamples, rng),
        "whole": graph_scores(factors, FACTORS, whole, set(), a.rows, a.resamples, rng),
        "whole_truth": edge_names(whole),
    }


def settings(a):
    return {key: getattr(a, key) for key in SETTINGS} | {"train_subjects": a.train_subjects}


def assess(run, device, a):
    row = {"settings": settings(a)}
    for trained in (True, False):
        data = load(run, device, trained, a.train_subjects)
        row.setdefault("step", data["step"])
        row.update(bank=data["bank"], units=data["units"])
        row["trained" if trained else "floor"] = assess_codes(data, a)
    row["ceilings"] = ceilings(data, a)
    return row


def held(graphs, edges, induced=()):
    """Each edge's share of graphs, for one graph record or a T1 / FLAIR pair."""
    graphs = graphs if isinstance(graphs, list) else [graphs]
    return ", ".join(
        f"{edge}{' (induced)' if edge in induced else ''} "
        + " / ".join(f"{g['edges_found'].get(edge, 0):.2f}" for g in graphs)
        for edge in edges
    )


def show(name, row):
    s = row["settings"]
    anchors, induced = s["anchors"], row["trained"]["induced"]
    print(
        f"{name}  step {row['step']}, {row['units']} units, bank {Path(row['bank']).name}; anchors {', '.join(anchors)}"
        f" (label noise {s['anchor_noise']})"
    )
    truth = row["trained"]["truth"]
    print(
        "  truth among the anchors, concealed factors latent: "
        + ", ".join(e + (" (induced)" if e in induced else "") for e in truth)
        + f"; the SCM's edges touching a concealed factor: {', '.join(row['trained']['hidden']) or 'none'}"
    )
    c = row["ceilings"]
    print(f"  ceilings, PC on the test subjects' true variables ({s['rows']} rows x {s['resamples']}):")
    print(
        f"    the anchors: F1 {c['anchors']['f1_mean']:.2f} ± {c['anchors']['f1_sd']:.2f}; truth edges held"
        f" {held(c['anchors'], truth, induced)}; false edges per graph {c['anchors']['false_edges_mean']:.1f}"
    )
    print(
        f"    all seven factors, against the SCM: F1 {c['whole']['f1_mean']:.2f} ± {c['whole']['f1_sd']:.2f}; edges held"
        f" {held(c['whole'], c['whole_truth'])}"
        + (f"; {held(c['whole'], induced, induced)}" if induced else "")
        + f"; false edges per graph {c['whole']['false_edges_mean']:.1f}"
    )
    for kind in ("trained", "floor"):
        r = row[kind]
        k = r["count"]
        print(f"  {'trained code' if kind == 'trained' else 'untrained twin (floor)'}:")
        print(
            "    step 1, anchored slots, test R2 T1 / FLAIR: "
            + ", ".join(f"{n} {r['anchor_r2'][0][j]:.2f} / {r['anchor_r2'][1][j]:.2f}" for j, n in enumerate(anchors))
        )
        steps = [
            f"{d['val_agreement']:.2f} ({d['val_null95']:.2f}) R2 {d['redundancy']:.2f}"
            + ("" if d["kept"] else " dropped")
            for d in r["deflation"]
            if d["shared"]
        ]
        stop = r["deflation"][-1]
        if stop.get("empty"):
            end = "stopped: the code has no variance left"
        elif not stop["shared"]:
            end = f"stopped at {stop['val_agreement']:.2f} ({stop['val_null95']:.2f})"
        else:
            end = "reached --max-factors" if k == s["max_factors"] else "reached the step limit"
        print(
            f"    step 2, novel block: {k} components. Each shared direction taken: held-out agreement (null95), and R2"
            f" from the other view's slots and counted factors (dropped at {s['max_redundancy']} or above): "
            + (", ".join(steps) or "none")
            + f"; {end}"
        )
        print("    block content, R2 of supervised readouts (scoring only), T1 / FLAIR:")
        for i, target in enumerate(r["target_names"]):
            full = " / ".join(f"{view['full_code_r2'][i]:.2f}" for view in r["views"])
            part = " / ".join(f"{view['block_r2'][i]:.2f}" for view in r["views"])
            print(f"      {target:12s} full code {full}, block {part}")
        nulls = [view["rotation_null_r"] for view in r["views"]]
        print(
            "    step 3, each component named after a concealed target, |r| (random rotations of the block: mean |r| "
            + (" / ".join(f"{x:.2f}" for x in nulls) if nulls[0] is not None else "-")
            + ", T1 / FLAIR):"
        )
        for v, view in enumerate(r["views"]):
            for version, x in view["names"].items():
                named = ", ".join(f"{n} {m:.2f}" for n, m in zip(x["names"], x["matched_r"])) or "none"
                print(f"      {('T1', 'FLAIR')[v]:5s} {version}: {named}")
        g = [view["graph"] for view in r["views"]]
        pairs = k * len(anchors) + k * (k - 1) // 2
        print(
            "    step 4, PC on the slots and the components, T1 / FLAIR: F1 against the anchors' truth "
            + " / ".join(f"{x['f1_mean']:.2f} ± {x['f1_sd']:.2f}" for x in g)
        )
        print(f"      truth edges held {held(g, truth, induced)}")
        print(
            f"      false edges among the slots per graph {g[0]['false_edges_mean']:.1f} / {g[1]['false_edges_mean']:.1f};"
            f" edges touching a component per graph {g[0]['block_edges_mean']:.1f} / {g[1]['block_edges_mean']:.1f}"
            f" (all false; chance at most about {PC_ALPHA * pairs:.1f})"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_json", type=Path)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--anchors", nargs="+", default=list(ANCHORS), choices=FACTORS)
    p.add_argument("--anchor-noise", type=float, default=0.0, help="training-label noise, a fraction of each SD")
    p.add_argument("--max-factors", type=int, default=12)
    p.add_argument("--min-agreement", type=float, default=0.5, help="lowest held-out agreement a component may have")
    p.add_argument("--max-redundancy", type=float, default=0.2, help="highest cross-view redundancy a factor may have")
    p.add_argument("--rows", type=int, default=300, help="test subjects per PC resampling")
    p.add_argument("--resamples", type=int, default=20)
    p.add_argument("--train-subjects", type=int, default=None, help="cap on the training subjects (smoke tests)")
    a = p.parse_args()
    if len(set(a.anchors)) != len(a.anchors) or len(a.anchors) >= len(FACTORS):
        p.error("--anchors must be distinct and leave at least one factor concealed")
    device = warm.pick_device()
    results = json.loads(a.out_json.read_text()) if a.out_json.exists() else {}
    for run in a.runs:
        try:
            step = int(torch.load(run / "model.pt", map_location="cpu")["step"])
            last = int(json.loads((run / "report.json").read_text())["args"]["steps"])
        except Exception as error:  # no checkpoint yet, or one being written right now
            print("skipping", run.name, "for now:", error, flush=True)
            continue
        cached = results.get(run.name, {})
        if step != last or (cached.get("step") == step and cached.get("settings") == settings(a)):
            continue
        print("assessing", run.name, flush=True)
        results[run.name] = assess(run, device, a)
        a.out_json.write_text(json.dumps(results, indent=1))
    for name in sorted(results):
        show(name, results[name])


if __name__ == "__main__":
    main()
