from __future__ import annotations

import csv
import logging
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DROP_KS = (0, 1, 2, 4, 8, 16, 24)
FACTORS = [
    "brain_size",
    "ventricle_size",
    "lesion_x",
    "lesion_y",
    "lesion_z",
    "cortical_thickness",
    "temporal_atrophy",
    "lr_asymmetry",
    "sulcal_widening",
]
MODEL_REF = {"lesion_x": (0.775, 0.695), "lesion_y": (0.804, 0.727), "lesion_z": (0.522, 0.286)}


def per_view_center(a, b):
    return a - a.mean(0, keepdims=True), b - b.mean(0, keepdims=True)


def per_view_standardise(a, b):
    a, b = per_view_center(a, b)
    return a / (a.std(0, keepdims=True) + 1e-8), b / (b.std(0, keepdims=True) + 1e-8)


def view_acc(a, b, kind="lin", n_splits=3, seed=0):
    X = np.vstack([a, b]).astype(np.float64)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    groups = np.r_[np.arange(len(a)), np.arange(len(b))]
    accs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=groups):
        sc = StandardScaler().fit(X[tr])
        clf = (
            LogisticRegression(max_iter=2000)
            if kind == "lin"
            else MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=seed)
        )
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append(accuracy_score(y[te], clf.predict(sc.transform(X[te]))))
    return float(np.mean(accs))


def offset_size(a, b):
    pooled = np.sqrt(0.5 * (a.var(0) + b.var(0))) + 1e-12
    dmean = np.abs(a.mean(0) - b.mean(0)) / pooled
    ratio = (a.std(0) + 1e-12) / (b.std(0) + 1e-12)
    return {
        "dmean_mean": float(dmean.mean()),
        "dmean_max": float(dmean.max()),
        "std_ratio_med": float(np.median(ratio)),
    }


def extract_views(dataset, grid, batch_size=32, num_workers=0):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    f1, f2, gtc = [], [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"]
            if not isinstance(img, (list, tuple)) or len(img) < 2:
                raise RuntimeError("Expected two views per sample; got a single-view batch.")
            for src, dst in ((img[0], f1), (img[1], f2)):
                x = src.float()
                p = F.adaptive_avg_pool3d(x, (grid, grid, grid)) if x.dim() == 5 else x
                dst.append(p.reshape(p.shape[0], -1).cpu().numpy())
            gtc.append(batch["gt_latents"]["z_content"].numpy())
    return np.concatenate(f1), np.concatenate(f2), np.concatenate(gtc)


def probe_all(X, Z, seeds):
    from eval.identifiability_metrics import cv_probe_r2

    return [cv_probe_r2(X, Z[:, j], seeds=seeds)["mean"] for j in range(Z.shape[1])]


def per_channel_view_auc(a, b):
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    out = np.empty(a.shape[1])
    for c in range(a.shape[1]):
        s = np.r_[a[:, c], b[:, c]]
        if np.allclose(s, s[0]):
            out[c] = 0.5
            continue
        auc = roc_auc_score(y, s)
        out[c] = max(auc, 1.0 - auc)
    return out


def run_view_offset_probe(cli):
    from eval.dci import _extract_synthetic_representations
    from eval.run_dci_compare import _CONTENT, _CONTENT_V2, _STYLE, _STYLE_V2, parse_poolings
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args

    ref_args = load_run_args(cli.run_dir)
    dataset = build_synthetic_test_set(ref_args, cli.num_samples, causal=cli.causal == "match")
    ckpt = os.path.join(cli.run_dir, cli.checkpoint_name)
    model, _a, device = load_model_from_run_dir(cli.run_dir, ckpt if os.path.exists(ckpt) else None, None)

    rows = []
    for key, value in parse_poolings(cli.poolings):
        ld, _gc, _s1, _s2 = _extract_synthetic_representations(
            model, dataset, device, cli.batch_size, cli.num_workers, pooling=value
        )
        if cli.level not in ld:
            continue
        for bname, idx in (("content", (_CONTENT, _CONTENT_V2)), ("style", (_STYLE, _STYLE_V2))):
            b1, b2 = ld[cli.level][idx[0]], ld[cli.level][idx[1]]
            if b1 is None or b2 is None or b1.shape[1] == 0:
                continue
            s1, s2 = per_view_standardise(b1, b2)
            c1, c2 = per_view_center(b1, b2)
            rows.append(
                {
                    "pooling": key,
                    "block": bname,
                    "raw_lin": view_acc(b1, b2, "lin"),
                    "cent_lin": view_acc(c1, c2, "lin"),
                    "std_lin": view_acc(s1, s2, "lin"),
                    "std_mlp": view_acc(s1, s2, "mlp"),
                    **offset_size(b1, b2),
                }
            )
    del model

    print("\n" + "=" * 96)
    print("  CAN A PROBE TELL WHICH VIEW A FEATURE VECTOR CAME FROM?   0.5 = view-invariant")
    print("  grouped CV (a sample's two rows share a fold); 'raw_lin' at stats ~ content_view")
    print("=" * 96)
    hdr = f"  {'pooling':<9}{'block':<9}{'raw_lin':>9}{'cent_lin':>10}{'std_lin':>9}{'std_MLP':>9}{'|dmean|/sd':>12}{'max':>8}{'sd ratio':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(
            f"  {r['pooling']:<9}{r['block']:<9}{r['raw_lin']:>9.3f}{r['cent_lin']:>10.3f}{r['std_lin']:>9.3f}{r['std_mlp']:>9.3f}{r['dmean_mean']:>12.3f}{r['dmean_max']:>8.2f}{r['std_ratio_med']:>10.2f}"
        )
    print("\n  std_lin is uninformative BY CONSTRUCTION (standardising removes exactly what a")
    print("  linear probe reads). std_MLP is the number that decides it.")

    con = [r for r in rows if r["block"] == "content"]
    if con:
        c = con[0]
        print("\n" + "=" * 96)
        print(f"  VERDICT   content  raw {c['raw_lin']:.3f} -> standardised+MLP {c['std_mlp']:.3f}")
        if c["raw_lin"] - c["std_mlp"] > 0.25 and c["std_mlp"] < 0.65:
            print("  => NO DETECTABLE NON-AFFINE DIFFERENCE REMAINS. The leak is the per-view constant")
            print("     component — invisible to BT's alignment term at any bt_lambda, and penalised")
            print("     directly by an MSE term. Prediction: --bt-sim-coeff drives content_view -> 0.5.")
            print("     (Caveat: the probe reads marginals, so sign flips / symmetric folds escape it.)")
        elif c["std_mlp"] > 0.8:
            print("  => NOT AFFINE. The views stay separable after removing per-view means and scales,")
            print("     so BT's blind spot does not explain this leak and an MSE term will not fix it.")
        else:
            print("  => PARTIAL. Some of the leak is affine and some is not; an MSE term should help")
            print("     but will not close it.")

    os.makedirs(cli.out, exist_ok=True)
    path = os.path.join(cli.out, "view_offset.csv")
    cols = ["pooling", "block", "raw_lin", "cent_lin", "std_lin", "std_mlp", "dmean_mean", "dmean_max", "std_ratio_med"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    logger.info("Wrote %s", path)


def run_view_leak_channels(cli):
    from eval.dci import _extract_synthetic_representations
    from eval.identifiability_metrics import cv_probe_r2
    from eval.run_dci_compare import _CONTENT, _CONTENT_V2, parse_poolings
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args

    seeds = tuple(int(s) for s in cli.seeds.split(","))
    ref_args = load_run_args(cli.run_dir)
    dataset = build_synthetic_test_set(ref_args, cli.num_samples, causal=cli.causal == "match")
    ckpt = os.path.join(cli.run_dir, cli.checkpoint_name)
    model, _a, device = load_model_from_run_dir(cli.run_dir, ckpt if os.path.exists(ckpt) else None, None)

    rows = []
    for key, value in parse_poolings(cli.poolings):
        ld, gt_content, _s1, _s2 = _extract_synthetic_representations(
            model, dataset, device, cli.batch_size, cli.num_workers, pooling=value
        )
        if cli.level not in ld:
            continue
        b1, b2 = ld[cli.level][_CONTENT], ld[cli.level][_CONTENT_V2]
        if b1 is None or b2 is None or b1.shape[1] == 0:
            continue

        auc = per_channel_view_auc(b1, b2)
        order = np.argsort(-auc)
        d = b1.shape[1]

        print("\n" + "=" * 84)
        print(f"  PER-CHANNEL VIEW LEAK — pooling '{key}', {d} content features")
        print("=" * 84)
        print(f"  channels with AUC > 0.9 : {int((auc > 0.9).sum()):d} / {d}")
        print(f"  channels with AUC > 0.7 : {int((auc > 0.7).sum()):d} / {d}")
        print(f"  channels with AUC < 0.6 : {int((auc < 0.6).sum()):d} / {d}")
        print(f"  median AUC              : {np.median(auc):.3f}      max {auc.max():.3f}")
        top = ", ".join(f"{auc[i]:.2f}" for i in order[:10])
        print(f"  top-10 channel AUCs     : {top}")

        print(f"\n  {'drop top-k':>11}{'kept':>7}{'view acc (MLP)':>17}{'mean factor R²':>17}")
        print("  " + "-" * 52)
        for k in DROP_KS:
            if k >= d:
                continue
            keep = order[k:]
            va = view_acc(b1[:, keep], b2[:, keep], "mlp")
            r2 = cv_probe_r2(b1[:, keep], gt_content, seeds=seeds)["mean"]
            rows.append({"pooling": key, "k": k, "kept": len(keep), "view_acc": va, "factor_r2": r2})
            print(f"  {k:>11}{len(keep):>7}{va:>17.3f}{r2:>17.3f}")

        sub = [r for r in rows if r["pooling"] == key]
        first, last = sub[0], sub[-1]
        print()
        if last["view_acc"] > 0.8:
            print("  => DISTRIBUTED. Dropping the worst leakers does not restore view-invariance;")
            print("     every channel carries modality. No channel-level intervention will help —")
            print("     the objective is simply not producing a view-invariant block.")
        elif last["factor_r2"] > 0.8 * first["factor_r2"]:
            print("  => LOCALISED AND SEPARABLE. The leak sits in a few channels that carry little")
            print("     content: excluding them buys view-invariance almost for free.")
        else:
            print("  => ENTANGLED. View info and content live in the same channels — the leak falls")
            print("     only as fast as the content does, so channel selection cannot separate them.")

    del model
    os.makedirs(cli.out, exist_ok=True)
    path = os.path.join(cli.out, "view_leak_channels.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pooling", "k", "kept", "view_acc", "factor_r2"])
        for r in rows:
            w.writerow([r["pooling"], r["k"], r["kept"], r["view_acc"], r["factor_r2"]])
    logger.info("Wrote %s", path)


def run_view_asymmetry_probe(cli):
    from eval.run_dci_synthetic import build_synthetic_test_set, load_run_args

    ref_args = load_run_args(cli.run_dir)
    dataset = build_synthetic_test_set(ref_args, cli.num_samples, causal=cli.causal == "match")
    seeds = tuple(int(s) for s in cli.seeds.split(","))
    grids = tuple(int(g) for g in cli.grids.split(","))
    rows = []
    for grid in grids:
        logger.info("grid %d^3 ...", grid)
        X1, X2, Z = extract_views(dataset, grid, cli.batch_size, cli.num_workers)
        names = FACTORS[: Z.shape[1]]
        res = {
            "T1": probe_all(X1, Z, seeds),
            "FLAIR": probe_all(X2, Z, seeds),
            "both": probe_all(np.hstack([X1, X2]), Z, seeds),
        }

        print("\n" + "=" * 84)
        print(f"  PER-VIEW FACTOR CEILING — raw voxels pooled to {grid}^3 ({X1.shape[1]} features per view)")
        print("  a view-invariant code is capped near the WEAKER view; reconstruction is not")
        print("=" * 84)
        hdr = f"  {'factor':<20}{'T1':>9}{'FLAIR':>9}{'both':>9}{'FLAIR-T1':>10}   reference (base / contrastive)"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for j, nm in enumerate(names):
            t1, fl, bo = res["T1"][j], res["FLAIR"][j], res["both"][j]
            ref = MODEL_REF.get(nm)
            ref_s = f"{ref[0]:.3f} / {ref[1]:.3f}" if ref else ""
            flag = "  <-" if ref else ""
            print(f"  {nm:<20}{t1:>9.3f}{fl:>9.3f}{bo:>9.3f}{fl - t1:>10.3f}   {ref_s}{flag}")
            rows.append({"grid": grid, "factor": nm, "t1": t1, "flair": fl, "both": bo})

        les = [j for j, nm in enumerate(names) if nm.startswith("lesion")]
        if les:
            mt1 = float(np.mean([res["T1"][j] for j in les]))
            mfl = float(np.mean([res["FLAIR"][j] for j in les]))
            mbs = float(np.mean([MODEL_REF[names[j]][0] for j in les if names[j] in MODEL_REF]))
            mct = float(np.mean([MODEL_REF[names[j]][1] for j in les if names[j] in MODEL_REF]))
            print(f"\n  lesion mean   T1 {mt1:.3f}   FLAIR {mfl:.3f}   |   baseline {mbs:.3f}   contrastive {mct:.3f}")
            if mfl - mt1 < 0.05:
                print("  => VIEWS ARE EQUALLY INFORMATIVE about lesions. The weaker-view ceiling cannot")
                print("     explain the contrastive deficit; it is a genuine failure to use what is there.")
            elif abs(mct - mt1) < abs(mct - mfl):
                print("  => CONTRASTIVE SITS AT THE T1 (WEAKER) CEILING. Consistent with view-invariance")
                print("     costing the stronger view's extra lesion information — not a locality bug.")
            else:
                print("  => Views are unequal, but the contrastive model is NOT at the weaker ceiling.")
                print("     The asymmetry is real yet does not by itself account for the deficit.")

        if "brain_size" in names:
            b = names.index("brain_size")
            gap = abs(res["FLAIR"][b] - res["T1"][b])
            if gap > 0.15:
                print(f"\n  [!] CONTROL FAILED: brain_size differs by {gap:.3f} between views. It is geometric,")
                print("      not contrast-dependent, so this points at the extraction rather than the data.")

    os.makedirs(cli.out, exist_ok=True)
    path = os.path.join(cli.out, "view_asymmetry.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grid", "factor", "t1", "flair", "both"])
        for r in rows:
            w.writerow([r["grid"], r["factor"], r["t1"], r["flair"], r["both"]])
    logger.info("Wrote %s", path)
