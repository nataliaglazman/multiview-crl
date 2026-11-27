"""Stage 84 (no training; rules in HANDOFF section 9, written before this code and any result): label-free completeness
of a run's code from the other runs' codes (A), and the chart's width from its rebuild error (B).

A. Every content-only code is a function of the content, so a code that holds all of it determines every other run's
   code, and a code missing a part cannot determine a code that holds it. Per ordered pair (candidate A, reference B)
   on one bank's subjects, every code entering as its shared canonical variates (chart_check.shared_variates): A's T1
   variates predict B's FLAIR variates and A's FLAIR variates B's T1 ones (least squares, then RBF kernel ridge with
   its setting chosen by leave-one-out on the training subjects; scored on the test subjects), against B's own
   variates from the same view as the ceiling. q(A | B) is the smallest share, over the whitened target's directions, of what B's
   own view explains that A's view also explains, the lower of the two view directions. A code's score in a panel is
   its minimum q over the panel's other codes.
B. The five 10-unit parents charted at 6 and 5 units by chart_check.py (run_stage84.sh); `width` reads those with
   stage 82b's 7- and 8-unit certificates and applies the width rule.

    python panel_check.py completeness OUT.json > panel.txt
    python panel_check.py width OUT.json
Stage 85 (rules in HANDOFF section 9): the same score within each cell of a pool_check.py detect report, then the
method per triple of seeds (the top score among three, charted at stage 84 B's rule width) and the rules E1-E3.
    python panel_check.py panel OUT.json --detect DETECT.json
    python panel_check.py pipeline OUT.json --detect DETECT.json --panel PANEL.json --charts CHART.json...
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

import chart_check
import pool_check

HERE = Path(__file__).resolve().parent
CAUSAL = HERE / "causal"
STAGE = CAUSAL / "stage84"
GAMMAS = pool_check.GAMMAS  # times 1 / the code's width, on standardized codes
ALPHAS = pool_check.ALPHAS
EXPLAINED_FLOOR = 0.05  # the reference's explained share below which a direction is not divided by
VIEWS = ("T1", "FLAIR")
WIDTH_BOUND = 0.10  # rule B: the smallest width whose worst-direction 95% bound is at most this
R77 = "ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_"


def standardized(train, test):
    mean, sd = train.mean(0), train.std(0) + 1e-12
    return (train - mean) / sd, (test - mean) / sd


def rbf(a, b, gamma):
    sq = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


class LooKernel:
    """Least squares on the standardized code, then RBF kernel ridge on its residual (sklearn's KernelRidge
    parametrization), the (gamma, alpha) setting chosen per target by leave-one-out error on the training rows. The
    kernel's eigendecomposition is computed once per code and reused for every target."""

    def __init__(self, x_train, x_test, gammas=GAMMAS, alphas=ALPHAS):
        x_train, x_test = standardized(x_train, x_test)
        self.design_train = np.column_stack([np.ones(len(x_train)), x_train])
        self.design_test = np.column_stack([np.ones(len(x_test)), x_test])
        self.alphas = alphas
        self.kernels = {}
        for g in gammas:
            lam, vec = np.linalg.eigh(rbf(x_train, x_train, g / x_train.shape[1]))
            self.kernels[g] = (np.maximum(lam, 0.0), vec, rbf(x_test, x_train, g / x_train.shape[1]))

    def predict(self, y_train):
        coef = np.linalg.lstsq(self.design_train, y_train, rcond=None)[0]
        resid = y_train - self.design_train @ coef
        best = None
        for g, (lam, vec, _) in self.kernels.items():
            proj, sq = vec.T @ resid, vec**2
            for a in self.alphas:
                shrink = lam / (lam + a)
                loo = (resid - vec @ (shrink[:, None] * proj)) / (1 - sq @ shrink)[:, None]
                err = float((loo**2).sum())
                if best is None or err < best[0]:
                    best = (err, g, a)
        _, g, a = best
        lam, vec, k_test = self.kernels[g]
        beta = vec @ ((vec.T @ resid) / (lam + a)[:, None])
        return self.design_test @ coef + k_test @ beta, (g, a)


def inverse_root(m, floor=1e-9):
    w, v = np.linalg.eigh((m + m.T) / 2)
    return (v / np.sqrt(np.maximum(w, floor))) @ v.T


def explained(target_test, prediction):
    """I minus the whitened residual second moment: the share of each direction of the test target a map explains."""
    centred = target_test - target_test.mean(0)
    whiten = inverse_root(centred.T @ centred / len(centred))
    resid = target_test - prediction
    return np.eye(target_test.shape[1]) - whiten @ (resid.T @ resid / len(resid)) @ whiten


def completeness(e_candidate, e_reference, floor=EXPLAINED_FLOOR):
    """The smallest generalized eigenvalue of the candidate's explained share against the reference's own."""
    root = inverse_root(e_reference, floor)
    m = root @ e_candidate @ root
    return float(np.linalg.eigvalsh((m + m.T) / 2).min())


def targets_of(code_train, code_test):
    """Per view, the code's shared canonical variates (chart_check.shared_variates, its fixed seed 0)."""
    u_train, u_test, _, _, info = chart_check.shared_variates(
        code_train.astype(np.float64), code_test.astype(np.float64), np.random.default_rng(0)
    )
    return u_train, u_test, info["kept"]


def pair_matrix(codes, pairs):
    """codes: label -> (code_train (2, n, d), code_test (2, m, d)); pairs: (candidate, reference) labels, each
    reference also paired with itself (the ceiling). Returns q[(candidate, reference)], per view direction too."""
    needed = {}
    for a, b in pairs:
        needed.setdefault(a, set()).add(b)
        needed.setdefault(b, set()).add(b)
    # every code enters as its own shared variates, as target and as input: view-private coordinates carry no content
    # and, as kernel inputs, blur the map (a complete code with two private coordinates read q .30 in the unit test)
    targets = {x: targets_of(*codes[x]) for x in needed}
    explained_of, settings = {}, {}
    for x in sorted(needed):
        for v in range(2):  # view v of the candidate predicts view 1 - v of each reference's variates
            model = LooKernel(targets[x][0][v], targets[x][1][v])
            for b in sorted(needed[x]):
                u_train, u_test, _ = targets[b]
                prediction, setting = model.predict(u_train[1 - v])
                explained_of[(x, b, v)] = explained(u_test[1 - v], prediction)
                settings[(x, b, v)] = setting
        print(f"  {x}: {len(needed[x])} targets", file=sys.stderr, flush=True)
    q, by_view = {}, {}
    for a, b in pairs:
        per_view = [completeness(explained_of[(a, b, v)], explained_of[(b, b, v)]) for v in range(2)]
        by_view[(a, b)] = per_view
        q[(a, b)] = min(per_view)
    references = {b for _, b in pairs}
    ceiling = {b: [float(np.linalg.eigvalsh(explained_of[(b, b, v)]).min()) for v in range(2)] for b in references}
    kept = {x: targets[x][2] for x in targets}
    return q, by_view, ceiling, kept, settings


def panel_scores(q, members, how=min):
    return {x: how([q[(x, y)] for y in members if y != x]) for x in members}


def pairs_right(scores, held):
    holders = [x for x in scores if held[x]]
    failures = [x for x in scores if not held[x]]
    pairs = [(h, f) for h in holders for f in failures]
    return sum(scores[h] > scores[f] for h, f in pairs), len(pairs)


def selection(q, members, held, k):
    """Among every k of a cell's codes, each scored against the other k - 1: does the top score hold?"""
    with_holder = right = 0
    random_pick = 0.0
    for subset in itertools.combinations(members, k):
        if not any(held[x] for x in subset):
            continue
        with_holder += 1
        s = panel_scores(q, subset)
        right += held[max(subset, key=lambda x: s[x])]
        random_pick += sum(held[x] for x in subset) / k
    return with_holder, right, random_pick


def cells_and_verdicts():
    """cell -> [(label, run)], label -> holds, label -> rho1 (where scored), as ρ1 was judged (stages 75 and 76),
    plus stage 74's 12- and 16-unit cells (stage 82b) and stage 77's 20 codes (all fail)."""
    cells, held, rho1 = {}, {}, {}
    s75 = json.loads((CAUSAL / "stage75" / "pool_check.json").read_text())
    for label, holds in s75["verdicts"].items():
        _, arm, seed = label.split("_")
        cells.setdefault(f"c8_{arm}", []).append((label, pool_check.run_name(arm, seed[1:])))
        held[label] = bool(holds)
        if label in s75.get("leftover", {}):
            rho1[label] = float(s75["leftover"][label]["rho"][0])
    for name in ("existing", "new_seeds"):
        for label, row in json.loads((CAUSAL / "stage76" / f"{name}.json").read_text())["rows"].items():
            cells.setdefault(row["cell"], []).append((label, row["run"]))
            held[label], rho1[label] = bool(row["holds"]), float(row["rho1"])
    for width in (12, 16):
        for label, row in json.loads((CAUSAL / "stage82" / f"b_c{width}.json").read_text()).items():
            cells.setdefault(f"c{width}_a100", []).append((label, row["run"]))
            held[label] = bool(row["parent"]["holds"])
            if "rho1" in row["parent"]:
                rho1[label] = float(row["parent"]["rho1"])
    for label, row in json.loads((CAUSAL / "stage77" / "runs.json").read_text())["rows"].items():
        cells.setdefault("s77", []).append((label, row["run"]))
        held[label], rho1[label] = bool(row["holds"]), float(row["rho1"])
    return cells, held, rho1


def load(run):
    d = np.load(pool_check.POOL / f"{run}.npz")
    return d["code_train"], d["code_test"]


def verdict(right, total, works, fails):
    return "works" if right >= works else "fails" if right <= fails else "unresolved"


def report_cell(cell, labels, q, held, rho1, ceiling, kept):
    """One cell's scores (minimum and median over the rest of the cell), pairs and picks, printed and returned."""
    s, s_med = panel_scores(q, labels), panel_scores(q, labels, np.median)
    right, n = pairs_right(s, held)
    right_med, _ = pairs_right(s_med, held)
    sel = {k: selection(q, labels, held, k) for k in (2, 3, 4) if k < len(labels)}
    print(f"\n  {cell}: pairs right {right}/{n} (median variant {right_med}/{n})")
    for x in sorted(labels, key=lambda x: -s[x]):
        r = f"  rho1 {rho1[x]:.2f}" if x in rho1 else ""
        print(f"    {x:16s} {'holds' if held[x] else 'fails'}  score {s[x]:+.2f}  median {s_med[x]:+.2f}  "
              f"ceiling {min(ceiling[x]):.2f}  shared k {kept[x]}{r}")
    for k, (with_holder, ok, rand) in sel.items():
        print(f"    K = {k}: {with_holder} subsets have a holder; the top score holds in {ok} of them, a random "
              f"pick in {rand:.1f}")
    return {"scores": s, "median_scores": s_med, "pairs_right": [right, n], "median_pairs_right": [right_med, n],
            "selection": sel}


def q_report(q, by_view, ceiling, kept, settings):
    return {"cells": {}, "q": {f"{a}|{b}": v for (a, b), v in q.items()},
            "q_by_view": {f"{a}|{b}": v for (a, b), v in by_view.items()}, "ceiling": ceiling, "kept": kept,
            "settings": {f"{x}|{b}|{VIEWS[v]}": list(s) for (x, b, v), s in settings.items()}}


def write_json(out, report):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def completeness_stage(out):
    cells, held, rho1 = cells_and_verdicts()
    reference_cell = "c8_ae500"
    codes = {label: load(run) for members in cells.values() for label, run in members}
    pairs = [(a, b) for cell, members in cells.items() if cell != "s77"
             for a, _ in members for b, _ in members if a != b]
    pairs += [(a, b) for a, _ in cells["s77"] for b, _ in cells[reference_cell]]
    q, by_view, ceiling, kept, settings = pair_matrix(codes, pairs)
    report = q_report(q, by_view, ceiling, kept, settings)
    print("== A. Cross-seed completeness: each code's score is the minimum, over the rest of its cell, of q(A | B), the")
    print("   share of B's shared content (as B's other view reads it) that A's code explains in its worst direction.")
    totals = {"wide": [0, 0], "eight": [0, 0], "wide_median": [0, 0], "eight_median": [0, 0]}
    for cell, members in cells.items():
        if cell == "s77":
            continue
        entry = report_cell(cell, [x for x, _ in members], q, held, rho1, ceiling, kept)
        report["cells"][cell] = entry
        units = int(cell.split("_")[0][1:])
        group = None if units >= 12 else "eight" if units == 8 else "wide"
        if group:
            for key, value in (("", entry["pairs_right"]), ("_median", entry["median_pairs_right"])):
                totals[group + key][0] += value[0]
                totals[group + key][1] += value[1]
    s77 = {x: min(q[(x, b)] for b, _ in cells[reference_cell]) for x, _ in cells["s77"]}
    ref_scores = report["cells"][reference_cell]["scores"]
    ref_holders = [ref_scores[x] for x, _ in cells[reference_cell] if held[x]]
    report["s77"] = s77
    print(f"\n  stage 77's 20 failing codes scored against the {reference_cell} ten: "
          f"{min(s77.values()):+.2f} to {max(s77.values()):+.2f}; the ten's holders in their own cell "
          f"{min(ref_holders):+.2f} to {max(ref_holders):+.2f}; below every holder: "
          f"{sum(v < min(ref_holders) for v in s77.values())} of 20")
    wide, eight = totals["wide"], totals["eight"]
    a1 = verdict(wide[0], wide[1], 9, 7)
    a2 = "works" if eight[0] >= 0.8 * eight[1] else "fails"
    report["rules"] = {"A1": [*wide, a1], "A2": [*eight, a2], "totals": totals}
    print(f"\n  A1 (9 and 10 units; rho1 7 of 10): holder above failure in {wide[0]} of {wide[1]} within-cell pairs: {a1}")
    print(f"  A2 (8 units; rho1 69 of 69): {eight[0]} of {eight[1]} ({eight[0] / eight[1]:.0%}): {a2}")
    print(f"  described: the median variant, {totals['wide_median'][0]}/{wide[1]} at 9-10 units and "
          f"{totals['eight_median'][0]}/{eight[1]} at 8")
    write_json(out, report)


def cells_from_detect(path):
    """cell -> [(label, run)], label -> holds, label -> rho1, from a pool_check.py detect report (stage 76's rows)."""
    cells, held, rho1 = {}, {}, {}
    for label, row in json.loads(Path(path).read_text())["rows"].items():
        cells.setdefault(row["cell"], []).append((label, row["run"]))
        held[label], rho1[label] = bool(row["holds"]), float(row["rho1"])
    return cells, held, rho1


def panel_stage(out, detect):
    """Stage 85: each code of a detect report scored against the rest of its cell."""
    cells, held, rho1 = cells_from_detect(detect)
    codes = {label: load(run) for members in cells.values() for label, run in members}
    pairs = [(a, b) for members in cells.values() for a, _ in members for b, _ in members if a != b]
    q, by_view, ceiling, kept, settings = pair_matrix(codes, pairs)
    report = q_report(q, by_view, ceiling, kept, settings)
    print("== Cross-seed completeness within each cell (stage 84 A's procedure)")
    for cell, members in sorted(cells.items()):
        report["cells"][cell] = report_cell(cell, [x for x, _ in members], q, held, rho1, ceiling, kept)
    write_json(out, report)


def rule_width(bounds):
    """Stage 84 B's rule on one wide code: the smallest charted width whose worst-direction 95% bound is at most
    WIDTH_BOUND, else the widest."""
    ok = [u for u in sorted(bounds) if bounds[u] <= WIDTH_BOUND]
    return min(ok) if ok else max(bounds)


def pipeline_cell(labels, q, held, charts, k=3):
    """Per k seeds of a cell: keep the top cross-seed score among them (each scored against the other k - 1), chart it
    at its rule width, and succeed if that chart holds. charts: label -> {units: (chart holds, worst bound)}."""
    width = {x: rule_width({u: b for u, (_, b) in charts[x].items()}) for x in labels}
    rows = []
    for subset in itertools.combinations(labels, k):
        s = panel_scores(q, subset)
        pick = max(subset, key=lambda x: s[x])
        rows.append({"seeds": list(subset), "pick": pick, "pick_holds": held[pick], "width": width[pick],
                     "success": bool(charts[pick][width[pick]][0]), "with_holder": any(held[x] for x in subset)})
    return rows, width


STAGE85 = {"c10_acb": 7, "c10_auni": 7, "c13_dcb": 10}  # each cell's |C|


def pipeline_stage(out, detect, panel, chart_paths):
    """Stage 85's rules E1-E3 from the detect report, the cross-seed report and the charts at three widths."""
    cells, held, rho1 = cells_from_detect(detect)
    label_of = {run: label for members in cells.values() for label, run in members}
    q = {tuple(key.split("|")): v for key, v in json.loads(Path(panel).read_text())["q"].items()}
    charts = {}
    for path in chart_paths:
        for row in json.loads(Path(path).read_text()).values():
            charts.setdefault(label_of[row["run"]], {})[int(row["units"])] = (
                bool(row["chart"]["holds"]), float(row["certificate"]["worst_upper95"]))
    report = {"cells": {}}
    print("== Stage 85: the method per triple of seeds (the top cross-seed score among three, charted at its rule width)")
    for cell, n_c in STAGE85.items():
        if cell not in cells:
            print(f"\n  {cell}: missing")
            continue
        labels = sorted((x for x, _ in cells[cell]), key=lambda x: int(x.rsplit("_s", 1)[1]))
        rows, width = pipeline_cell(labels, q, held, charts)
        scores = panel_scores(q, labels)
        success = sum(r["success"] for r in rows)
        picks = [r for r in rows if r["with_holder"]]
        at_n = [x for x in labels if held[x]]
        report["cells"][cell] = {"holds": sum(held[x] for x in labels), "seeds": len(labels), "success": success,
                                 "triples": len(rows), "pick_right": sum(r["pick_holds"] for r in picks),
                                 "with_holder": len(picks), "width": width, "rows": rows,
                                 "chart_at_c_where_parent_holds": sum(charts[x][n_c][0] for x in at_n)}
        print(f"\n  {cell} (|C| {n_c}): {sum(held[x] for x in labels)} of {len(labels)} wide codes hold; the method succeeds "
              f"in {success} of {len(rows)} triples; the pick holds in {sum(r['pick_holds'] for r in picks)} of the "
              f"{len(picks)} triples with a holder; the {n_c}-unit chart holds in "
              f"{report['cells'][cell]['chart_at_c_where_parent_holds']} of the {len(at_n)} holders")
        for x in labels:
            marks = "  ".join(f"{u}: {'holds' if charts[x][u][0] else 'fails'} {charts[x][u][1]:.3f}"
                              for u in sorted(charts[x]))
            print(f"    {x:14s} {'holds' if held[x] else 'fails'}  score {scores[x]:+.2f}  rho1 {rho1[x]:.2f}  "
                  f"width {width[x]}  | {marks}")
    c = report["cells"]
    rules = {}
    if "c10_acb" in c:
        rules["E1"] = verdict(c["c10_acb"]["success"], c["c10_acb"]["triples"], 8, 5)
    if "c13_dcb" in c:
        share = c["c13_dcb"]["success"] / c["c13_dcb"]["triples"]
        rules["E2"] = "works" if share >= 0.8 else "fails" if share <= 0.5 else "unresolved"
    if "c10_acb" in c and "c10_auni" in c:
        gap = c["c10_acb"]["holds"] - c["c10_auni"]["holds"]
        rules["E3"] = "rests on code batches" if gap >= 2 else "does not rest on code batches"
    report["rules"] = rules
    print("\n  rules: " + "; ".join(f"{k} {v}" for k, v in rules.items()))
    write_json(out, report)


def width_stage(out):
    """Rule B: per 10-unit parent, the worst-direction 95% bound at 5-8 units; the width is the smallest at or under
    WIDTH_BOUND."""
    sources = {5: [STAGE / "chart_u5.json"], 6: [STAGE / "chart_u6.json"],
               7: [CAUSAL / "stage82" / "cert_u7a.json", CAUSAL / "stage82" / "cert_u7b.json"],
               8: [CAUSAL / "stage82" / "cert_u8a.json", CAUSAL / "stage82" / "cert_u8b.json"]}
    bounds, means = {}, {}
    for units, paths in sources.items():
        for path in paths:
            for row in json.loads(path.read_text()).values():
                seed = 42 if not row["run"].rsplit("_", 1)[1].startswith("s") else int(row["run"].rsplit("_s", 1)[1])
                c = row["certificate"]
                bounds.setdefault(seed, {})[units] = c["worst_upper95"]
                means.setdefault(seed, {})[units] = c["mean"]
    rows, chosen = {}, {}
    print("== B. Worst-direction rebuild error (95% bound) of the chart of each 10-unit parent, by chart width")
    for seed in sorted(bounds):
        ok = [u for u in sorted(bounds[seed]) if bounds[seed][u] <= WIDTH_BOUND]
        chosen[seed] = min(ok) if ok else None
        holds = chart_check.PARENT_HOLDS[seed]
        rows[seed] = {"holds": holds, "worst_upper95": bounds[seed], "mean": means[seed], "width": chosen[seed]}
        print(f"  s{seed:<3d} parent {'holds' if holds else 'fails'}  "
              + "  ".join(f"{u}: {bounds[seed][u]:.4f} (mean {means[seed][u]:.4f})" for u in sorted(bounds[seed]))
              + f"  -> width {chosen[seed]}")
    holders = [s for s in chosen if chart_check.PARENT_HOLDS[s]]
    at7 = sum(chosen[s] == 7 for s in holders)
    low = sum(chosen[s] is not None and chosen[s] <= 6 for s in holders)
    result = "works" if at7 == len(holders) else "fails" if low >= 2 else "unresolved"
    print(f"  rule B: width 7 in {at7} of {len(holders)} holding parents, 6 or fewer in {low}: {result}")
    out.write_text(json.dumps({"rows": rows, "rule": [at7, len(holders), low, result]}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("completeness", "width", "panel", "pipeline"))
    p.add_argument("out", type=Path)
    p.add_argument("--detect", type=Path, help="panel, pipeline: a pool_check.py detect report")
    p.add_argument("--panel", type=Path, help="pipeline: the panel step's report")
    p.add_argument("--charts", nargs="*", type=Path, default=[], help="pipeline: chart_check.py reports")
    a = p.parse_args()
    if a.what == "completeness":
        completeness_stage(a.out)
    elif a.what == "width":
        width_stage(a.out)
    elif a.what == "panel":
        panel_stage(a.out, a.detect)
    else:
        pipeline_stage(a.out, a.detect, a.panel, a.charts)


if __name__ == "__main__":
    main()
