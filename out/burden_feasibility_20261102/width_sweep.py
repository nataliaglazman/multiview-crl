"""Does the content-size sweep find the fixed bank's width? Stage 74.

Daunhawer et al. (ICLR 2023, App. B.2) set the encoding size at the elbow of the held-out contrastive loss over a
width sweep. training/content_size_numerical.py reproduces their numerical check (5 content latents: elbow k* = 6, own
content saturating at 5, spare units taking style). This repeats it on the images: stage 56's recipe on the
fixed-position causal bank (7 shared latents, so |C| + 1 = 8 under the cosine) at a range of widths.

measure  (GPU) per run: its test codes (anchor_check.py's export into CODES/<run>.npz, reused while newer than
         model.pt), the held-out loss (the run's InfoNCE term, as training computes it, on consecutive batches of the
         test subjects, as warm.py --real does on validation subjects; the prediction term is left out), and each
         factor's own part read from each view's code (anchor_check.readout_r2, averaged over the views).
score    (no torch) means over seeds per width, then two widths:
  k_loss  the elbow: the smallest width whose mean held-out loss is within FRAC of the total drop above the minimum
          (content_size_numerical.elbow, FRAC .05)
  k_own   the smallest width whose mean own-part R2 (over the seven factors) is within FRAC of its total rise
Pass rule, fixed before any result (E-W in paper/experiment-plan-2026-11-24.md): the elbow sets the width if k_loss
is within one unit of |C| + 1. k_own is the labelled reference it is read against.

    python width_sweep.py measure OUT_JSON CODES LABEL=RUN [LABEL=RUN ...] [--device cuda]
    python width_sweep.py score OUT_JSON [--frac .05] [--latents 7] [--figure OUT_PNG]
    python width_sweep.py --self-test
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

FACTORS = ("brain", "ventricle", "burden", "thickness", "temporal", "lr", "sulcal")  # anchor_check.FACTORS


def elbow(ks, values, frac, rising=False):
    """Smallest k whose value is within frac of the total change from the first k to the best (training/
    content_size_numerical.elbow for a falling loss; mirrored for a rising score)."""
    v = -np.asarray(values, dtype=float) if rising else np.asarray(values, dtype=float)
    lo, hi = v.min(), v[0]
    return next(k for k, x in zip(ks, v) if x <= lo + frac * (hi - lo))


def heldout_loss(t1, flair, model, args, device, size=32):
    import torch

    import training.main_conv_synthetic as T

    sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
    out = []
    with torch.no_grad():
        for s in range(0, len(t1) - size + 1, size):
            pair = torch.from_numpy(np.concatenate([t1[s : s + size], flair[s : s + size]]).astype(np.float32))
            out.append(float(T.contrastive_loss(pair.to(device), model, args, sim, crit, None)))
    return float(np.mean(out))


def measure(out_json, codes, pairs, device):
    import anchor_check as A
    import code_audit
    import warm

    rows = []
    for pair in pairs:
        label, run = pair.split("=", 1)
        run = Path(run)
        npz = Path(codes) / f"{run.name}.npz"
        if not npz.exists() or npz.stat().st_mtime < (run / "model.pt").stat().st_mtime:
            print(f"export {run.name}", flush=True)
            A.export(run, device, npz)
        d = np.load(npz)
        meta = json.loads(str(d["meta"]))
        if meta["loss_type"] != "infonce":
            raise ValueError(f"{run.name}: the held-out loss is the InfoNCE term, not {meta['loss_type']}")
        report, model, _, _, step = code_audit.load_run(run, device, trained=True)
        a = report["args"]
        args = warm.run_args()
        args.content_channels = meta["width"]
        if a.get("tau", "None") != "None":
            args.tau = float(a["tau"])
        t1, flair = d["code_t1"], d["code_flair"]
        own = (A.readout_r2(t1, d["own"]) + A.readout_r2(flair, d["own"])) / 2
        row = {
            "label": label,
            "run": run.name,
            "bank": meta["bank"],
            "step": int(step),
            "width": int(meta["width"]),
            "seed": 42 if a.get("seed", "None") == "None" else int(a["seed"]),
            "heldout_loss": heldout_loss(t1, flair, model.to(device), args, device),
            "own_r2": dict(zip(FACTORS, map(float, own))),
        }
        print(
            f"{label}: width {row['width']} seed {row['seed']} held-out loss {row['heldout_loss']:.3f} own "
            + " ".join(f"{f} {v:.2f}" for f, v in row["own_r2"].items()),
            flush=True,
        )
        rows.append(row)
    if len({r["bank"] for r in rows}) > 1:
        raise ValueError("one bank per sweep: " + ", ".join(sorted({r["bank"] for r in rows})))
    Path(out_json).write_text(json.dumps({"runs": rows}, indent=1))


def summarize(rows, frac, latents):
    widths = sorted({r["width"] for r in rows})
    table = []
    for k in widths:
        rs = [r for r in rows if r["width"] == k]
        own = np.array([[r["own_r2"][f] for f in FACTORS] for r in rs])
        loss = np.array([r["heldout_loss"] for r in rs])
        table.append(
            {
                "width": k,
                "seeds": sorted(r["seed"] for r in rs),
                "heldout_loss": float(loss.mean()),
                "heldout_loss_sd": float(loss.std()),
                "own_mean": float(own.mean()),
                "own_mean_sd": float(own.mean(1).std()),
                "own_min": float(own.min(1).mean()),
                **{f"own_{f}": float(own[:, j].mean()) for j, f in enumerate(FACTORS)},
            }
        )
    k_loss = elbow(widths, [t["heldout_loss"] for t in table], frac)
    k_own = elbow(widths, [t["own_mean"] for t in table], frac, rising=True)
    per_seed = {}
    for seed in sorted({r["seed"] for r in rows}):
        rs = sorted((r for r in rows if r["seed"] == seed), key=lambda r: r["width"])
        if len(rs) == len(widths):
            per_seed[seed] = elbow([r["width"] for r in rs], [r["heldout_loss"] for r in rs], frac)
    target = latents + 1
    return {
        "latents": latents,
        "target": target,
        "frac": frac,
        "k_loss": k_loss,
        "k_own": k_own,
        "k_loss_per_seed": per_seed,
        "passes": abs(k_loss - target) <= 1,
        "table": table,
    }


def figure(summary, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = summary["table"]
    k = [r["width"] for r in t]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    top.errorbar(k, [r["heldout_loss"] for r in t], yerr=[r["heldout_loss_sd"] for r in t], marker="o", color="#2a78d6")
    top.set_ylabel("held-out InfoNCE")
    bottom.plot(k, [r["own_mean"] for r in t], marker="o", color="#2a78d6", lw=2, label="mean of seven")
    bottom.plot(k, [r["own_min"] for r in t], marker="v", color="#eb6834", lw=2, label="weakest factor")
    for f in FACTORS:
        bottom.plot(k, [r[f"own_{f}"] for r in t], color="#898781", lw=0.8, alpha=0.6)
    bottom.set_ylabel("own-part R² (grey: each factor)")
    bottom.set_xlabel("content units")
    bottom.legend(frameon=False, fontsize=9)
    for ax in (top, bottom):
        ax.axvline(summary["target"], color="#898781", ls="--", lw=1)
        ax.axvline(summary["k_loss"], color="#eb6834", ls=":", lw=1.5)
        ax.spines[["top", "right"]].set_visible(False)
    top.set_title(
        f"|C| = {summary['latents']}: target {summary['target']} (dashed), elbow {summary['k_loss']} (dotted), "
        f"own parts saturate at {summary['k_own']}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    with open(Path(out).with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[c for c in t[0] if c != "seeds"])
        w.writeheader()
        for r in t:
            w.writerow({c: v for c, v in r.items() if c != "seeds"})


def score(out_json, frac, latents, figure_path):
    data = json.loads(Path(out_json).read_text())
    summary = summarize(data["runs"], frac, latents)
    data["summary"] = summary
    Path(out_json).write_text(json.dumps(data, indent=1))
    print(
        f"{'width':>5} {'seeds':>5} {'loss':>13} {'own mean':>8} {'weakest':>7}  "
        + " ".join(f"{f[:6]:>6}" for f in FACTORS)
    )
    for r in summary["table"]:
        print(
            f"{r['width']:5d} {len(r['seeds']):5d} {r['heldout_loss']:7.3f} ± {r['heldout_loss_sd']:.3f} "
            f"{r['own_mean']:8.2f} {r['own_min']:7.2f}  " + " ".join(f"{r[f'own_{f}']:6.2f}" for f in FACTORS)
        )
    print(
        f"elbow k_loss {summary['k_loss']} (per seed {summary['k_loss_per_seed']}), own parts saturate at "
        f"k_own {summary['k_own']}; target |C| + 1 = {summary['target']}: " + ("PASS" if summary["passes"] else "FAIL")
    )
    if figure_path:
        figure(summary, figure_path)
        print(f"figure {figure_path} and its .csv")


def self_test():
    rng = np.random.default_rng(0)
    rows = []
    for seed in (42, 3, 4):
        for k in (4, 5, 6, 7, 8, 9, 10, 12, 16):
            loss = 0.5 + 3.0 * max(0, 7 - k) + 0.02 * rng.standard_normal()
            own = np.clip(np.minimum(k, 7) / 7 - 0.1 * rng.random(7), 0, 1) * 0.9
            rows.append(
                {"width": k, "seed": seed, "heldout_loss": loss, "own_r2": dict(zip(FACTORS, own)), "bank": "b"}
            )
    s = summarize(rows, 0.05, 7)
    assert s["k_loss"] == 7 and s["passes"], s
    assert s["k_own"] in (7, 8), s["k_own"]
    assert set(s["k_loss_per_seed"].values()) == {7}
    assert elbow([1, 2, 3], [0.1, 0.4, 0.52], 0.05, rising=True) == 3
    assert elbow([1, 2, 3], [0.1, 0.5, 0.52], 0.05, rising=True) == 2
    assert elbow([1, 2, 3], [5.0, 1.5, 0.9], 0.05) == 3
    assert elbow([1, 2, 3], [5.0, 0.95, 0.9], 0.05) == 2
    with __import__("tempfile").TemporaryDirectory() as tmp:
        figure(s, Path(tmp) / "w.png")
        assert (Path(tmp) / "w.csv").exists()
    print("self-test passed")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv == ["--self-test"]:
        return self_test()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("out_json")
    m.add_argument("codes")
    m.add_argument("runs", nargs="+", help="LABEL=RUN_DIR")
    m.add_argument("--device", default="cuda")
    s = sub.add_parser("score")
    s.add_argument("out_json")
    s.add_argument("--frac", type=float, default=0.05)
    s.add_argument("--latents", type=int, default=7)
    s.add_argument("--figure", default=None)
    a = p.parse_args(argv)
    if a.cmd == "measure":
        measure(a.out_json, a.codes, a.runs, a.device)
    else:
        score(a.out_json, a.frac, a.latents, a.figure)


if __name__ == "__main__":
    main()
