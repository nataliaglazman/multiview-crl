"""The plan's block node in causal discovery with tabular variables (section 10, step 3). No training.

The tabular benchmark's discovery task (eval.causal.tabular_benchmark) links image factors to tabular variables: one
factor drives an exposure, another the outcome, and two tabular causes each drive one factor through its own part.
Here only some factors are measured (the anchors, as FreeSurfer gives them). The rest are concealed, and the code
stands in for them as one block node: each view's code minus its fit on the measured anchors, cut to the directions
the two views agree on by factoriser.novel_block (the label-free count, stage 44's rule). PC (Fisher-Z, alpha .05) then
runs on the anchors, the block's components and the tabular variables.

Truth: the discovery DAG with each factor's own part as an explicit root (own part -> factor; a tabular cause is a
child of its target's own part), concealed factors latent, the concealed factors' own parts observed as the block.
The skeleton PC would find among the observed nodes with infinite data comes from d-separation. Scored per task seed:
  - block-tabular: a tabular node linked to the block counts once (any component), against whether it links to any
    concealed own part in the truth;
  - anchor-tabular and tabular-tabular edges as they are.
Arms per view: the trained code's block, its untrained twin's, the true concealed own parts residualised on the
anchors the same way (the oracle block), and the anchors alone (no block).

Rows: the block is fitted on the benchmark's 500 fitting rows (400 to fit, 100 to count), and PC runs on its 500
held-out discovery rows. Bundles are test subjects only, so the fit uses fewer subjects than factoriser.py (2,000).

    PYTHONPATH=out/burden_feasibility_20261102:out/regional_pilot_20261028/shim:. python \
        out/burden_feasibility_20261102/block_tabular.py OUT_JSON STAGE_DIR LABEL=BUNDLE [LABEL=BUNDLE ...]
"""

import argparse
import itertools
import json
from pathlib import Path

import factoriser as F
import numpy as np
from causallearn.search.ConstraintBased.PC import pc

ALPHA = 0.05


def oracle(adj, observed):
    """Pairs of ``observed`` that no subset of the other observed nodes d-separates."""
    edges = set()
    for x, y in itertools.combinations(observed, 2):
        others = [o for o in observed if o not in (x, y)]
        subsets = itertools.chain.from_iterable(itertools.combinations(others, k) for k in range(len(others) + 1))
        if not any(F.d_separated(adj, x, y, s) for s in subsets):
            edges.add(frozenset((x, y)))
    return edges


def augmented(discovery_adj, n_factors, cause_cols, cause_targets):
    """The discovery DAG with one own-part root per factor appended (indices n .. n + n_factors - 1)."""
    n = len(discovery_adj)
    adj = np.zeros((n + n_factors,) * 2, dtype=bool)
    adj[:n, :n] = discovery_adj
    for i in range(n_factors):
        adj[n + i, i] = True
    for col, target in zip(cause_cols, cause_targets):
        adj[col, target] = False
        adj[n + target, col] = True
    return adj


def pc_skeleton(values):
    z = (values - values.mean(0)) / values.std(0)
    graph = pc(z, ALPHA, "fisherz", show_progress=False).G.graph
    k = len(z[0])
    return {frozenset((i, j)) for i in range(k) for j in range(i + 1, k) if graph[i, j] != 0}


def counts(found, truth):
    return {"tp": len(found & truth), "fp": len(found - truth), "fn": len(truth - found)}


def block_components(code, anchors, fit, held, rng):
    """factoriser.novel_block on two views' codes, with the fitting rows split 400 / 100 and the held rows as test."""
    train, val = fit[:400], fit[400:]
    splits = {"train": train, "val": val, "test": held}
    data = {s: {"code": code[:, r].astype(np.float64)} for s, r in splits.items()}
    anc = {s: np.stack([anchors[r], anchors[r]]) for s, r in splits.items()}
    block, records = F.novel_block(data, anc, 12, 0.5, rng, 0.2)
    return block["test"], records


def oracle_block(own, anchors, fit, held):
    """The true concealed own parts minus their fit on the anchors (with squares and products), as residuals() does."""
    reference = anchors[fit]
    model = F.LinearRegression().fit(F.quadratic(reference, reference), own[fit])
    return own[held] - model.predict(F.quadratic(anchors[held], reference))


def score_arm(anchors, block, tab, truth_sets, n_anchor, n_tab):
    """PC on [anchors | block | tabular]; returns counts per edge group with the block collapsed to one node."""
    k = block.shape[1]
    values = np.column_stack([anchors, block, tab]) if k else np.column_stack([anchors, tab])
    edges = pc_skeleton(values)
    a_idx = list(range(n_anchor))
    b_idx = list(range(n_anchor, n_anchor + k))
    t_idx = list(range(n_anchor + k, n_anchor + k + n_tab))
    found_bt = {t for t in range(n_tab) if any(frozenset((b, t_idx[t])) in edges for b in b_idx)}
    found_at = {(a, t) for a in range(n_anchor) for t in range(n_tab) if frozenset((a_idx[a], t_idx[t])) in edges}
    found_tt = {frozenset((s, t)) for s, t in itertools.combinations(range(n_tab), 2)
                if frozenset((t_idx[s], t_idx[t])) in edges}
    return {
        "block_tab": counts(found_bt, truth_sets["block_tab"]),
        "anchor_tab": counts(found_at, truth_sets["anchor_tab"]),
        "tab_tab": counts(found_tt, truth_sets["tab_tab"]),
        "block_tab_found": sorted(found_bt),
        "components": k,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("stage", type=Path, help="stage directory holding tabular.json and tables/")
    parser.add_argument("bundles", nargs="+", metavar="LABEL=BUNDLE")
    parser.add_argument("--anchors", nargs="+", default=list(F.ANCHORS))
    a = parser.parse_args()

    report = json.loads((a.stage / "tabular.json").read_text())
    names = report["discovery_nodes"]
    factors = report["image_factors"]
    held = np.array(report["discovery_held_out_rows"])
    fit = np.setdiff1d(np.arange(report["n_subjects"]), held)
    fit = np.random.default_rng(0).permutation(fit)
    seeds = report["settings"]["seeds"]
    anchor_idx = [factors.index(x) for x in a.anchors]
    concealed_idx = [i for i in range(len(factors)) if i not in anchor_idx]
    nf = len(factors)

    bundles = {}
    for pair in a.bundles:
        label, path = pair.split("=", 1)
        bundles[label] = {"trained": np.load(path)}
        floor = Path(path.replace(".npz", ".floor.npz"))
        if floor.exists():  # a code with no untrained twin (a chart, stage 82) is scored without one
            bundles[label]["untrained"] = np.load(floor)
    first = next(iter(bundles.values()))["trained"]
    z, own = first["z_content"].astype(np.float64), first["z_own"].astype(np.float64)
    for b in bundles.values():
        for kind in b.values():
            if not np.allclose(kind["z_content"], z):
                raise SystemExit("bundles do not share rows")
    anchors = z[:, anchor_idx]

    out = {"anchors": a.anchors, "concealed": [factors[i] for i in concealed_idx], "seeds": [], "stage": str(a.stage)}
    for seed in seeds:
        truth = np.load(a.stage / "tables" / f"truth_{seed}.npz")
        if not np.allclose(truth["z_content"], z):
            raise SystemExit(f"truth_{seed} rows differ from the bundles")
        obs = np.genfromtxt(a.stage / "tables" / f"observed_{seed}.csv", delimiter=",", names=True)
        tab_names = names[nf:]
        cause_cols = [names.index(c) for c in truth["tabular_cause_names"]]
        targets = [int(t) for t in truth["tabular_cause_targets"]]
        adj = augmented(truth["discovery_adjacency"].astype(bool), nf, cause_cols, targets)
        tab_cols = list(range(nf, len(names)))
        own_concealed = [len(names) + i for i in concealed_idx]
        observed = anchor_idx + own_concealed + tab_cols
        skeleton = oracle(adj, observed)
        truth_sets = {
            "block_tab": {t for t, col in enumerate(tab_cols) if any(frozenset((u, col)) in skeleton
                                                                      for u in own_concealed)},
            "anchor_tab": {(i, t) for i, ai in enumerate(anchor_idx) for t, col in enumerate(tab_cols)
                           if frozenset((ai, col)) in skeleton},
            "tab_tab": {frozenset((s, t)) for (s, cs), (t, ct) in itertools.combinations(enumerate(tab_cols), 2)
                        if frozenset((cs, ct)) in skeleton},
        }
        tab = np.column_stack(
            [obs["tabular_1"], obs["tabular_2"], obs["tabular_3"], truth["tabular_causes"],
             truth["discovery_treatment"], truth["discovery_outcome"]]
        )
        tab = tab[held]
        entry = {
            "seed": seed,
            "exposure_parent": [factors[i] for i in np.flatnonzero(truth["discovery_adjacency"][:nf, -2])],
            "outcome_parent": [factors[i] for i in np.flatnonzero(truth["discovery_adjacency"][:nf, -1])],
            "cause_targets": [factors[t] for t in targets],
            "truth_block_tab": [tab_names[t] for t in sorted(truth_sets["block_tab"])],
            "truth_anchor_tab": sorted(f"{a.anchors[i]} - {tab_names[t]}" for i, t in truth_sets["anchor_tab"]),
            "arms": {},
        }
        ob = oracle_block(own[:, concealed_idx], anchors, fit, held)
        entry["arms"]["oracle block"] = score_arm(anchors[held], ob, tab, truth_sets, len(anchor_idx), len(tab_cols))
        entry["arms"]["anchors only"] = score_arm(anchors[held], np.zeros((len(held), 0)), tab, truth_sets,
                                                  len(anchor_idx), len(tab_cols))
        for label, b in bundles.items():
            for kind, bundle in b.items():
                code = np.stack([bundle["emb_view1"], bundle["emb_view2"]]).astype(np.float64)
                comps, records = block_components(code, anchors, fit, held, np.random.default_rng(seed))
                for v, view in enumerate(("T1", "FLAIR")):
                    arm = f"{label} / {view}" + (" (untrained)" if kind == "untrained" else "")
                    entry["arms"][arm] = score_arm(anchors[held], comps[v], tab, truth_sets, len(anchor_idx),
                                                   len(tab_cols))
                    entry["arms"][arm]["count_records"] = [
                        {k: r.get(k) for k in ("val_agreement", "kept", "redundancy")} for r in records
                    ]
        out["seeds"].append(entry)
        print(f"seed {seed}: block-tab truth {entry['truth_block_tab']}", flush=True)

    summary = {}
    for arm in out["seeds"][0]["arms"]:
        row = {}
        for group in ("block_tab", "anchor_tab", "tab_tab"):
            c = {k: sum(s["arms"][arm][group][k] for s in out["seeds"]) for k in ("tp", "fp", "fn")}
            p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else float("nan")
            r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
            f1 = 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]) if c["tp"] + c["fp"] + c["fn"] else float("nan")
            row[group] = {**c, "precision": p, "recall": r, "f1": f1}
        row["components_mean"] = float(np.mean([s["arms"][arm]["components"] for s in out["seeds"]]))
        summary[arm] = row
    out["summary"] = summary
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"{'arm':45s} {'comps':>5s}  block-tab P/R/F1      anchor-tab P/R/F1     tab-tab F1")
    for arm, row in summary.items():
        bt, at, tt = row["block_tab"], row["anchor_tab"], row["tab_tab"]
        print(f"{arm:45s} {row['components_mean']:5.1f}  {bt['precision']:.2f} {bt['recall']:.2f} {bt['f1']:.2f}"
              f"      {at['precision']:.2f} {at['recall']:.2f} {at['f1']:.2f}      {tt['f1']:.2f}")


if __name__ == "__main__":
    main()
