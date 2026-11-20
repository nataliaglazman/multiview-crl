"""Elbow of the held-out InfoNCE over content size for an encoder-only sweep (conv_mlp_content_size.yaml).

Daunhawer et al. (ICLR 2023, App. B.2) choose the encoding size at the elbow of the held-out
contrastive loss. Reads each run's last dci_step*.json (val/loss, per-factor ridge R², content->view
accuracy), averages over seeds per content size, and applies training.content_size_numerical.elbow.

    python scripts/analyze_content_size.py 'results/encoder/runs/conv_mlp_csize_k*_s*' --out results/content_size_brain
"""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

from training.content_size_numerical import elbow


def last_eval(run):
    steps = [(int(re.search(r"dci_step(\d+)", p).group(1)), p) for p in glob.glob(f"{run}/dci_step*.json")]
    steps = [s for s in steps if s[0] > 0]
    if not steps:
        return None
    step, path = max(steps)
    return step, json.loads(Path(path).read_text())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("runs", help="glob of run directories")
    p.add_argument("--out", default=None)
    p.add_argument("--elbow-frac", type=float, default=0.05)
    a = p.parse_args(argv)

    rows = {}
    for run in sorted(glob.glob(a.runs)):
        settings = json.loads(Path(run, "settings.json").read_text())
        ev = last_eval(run)
        if ev is None or "val/loss" not in ev[1]:
            print(f"skip {run}: no finished eval with val/loss")
            continue
        step, m = ev
        per = m.get("per_factor", {}).get("content→content", {})
        rows.setdefault(settings["content_channels"], []).append(
            {
                "seed": settings["seed"],
                "step": step,
                "val_loss": m["val/loss"],
                "view_acc": m.get("content->view/acc", float("nan")),
                "r2": {f: v["ridge"] for f, v in per.items()},
            }
        )
    ks = sorted(rows)
    factors = sorted({f for rs in rows.values() for r in rs for f in r["r2"]})
    table = []
    for k in ks:
        rs = rows[k]
        table.append(
            {
                "k": k,
                "n_seeds": len(rs),
                "val_loss": float(np.mean([r["val_loss"] for r in rs])),
                "val_loss_sd": float(np.std([r["val_loss"] for r in rs])),
                "content_view_acc": float(np.nanmean([r["view_acc"] for r in rs])),
                **{f"r2_{f}": float(np.nanmean([r["r2"].get(f, np.nan) for r in rs])) for f in factors},
            }
        )
    k_star = elbow(ks, [r["val_loss"] for r in table], a.elbow_frac)
    head = f"{'k':>3} {'n':>2} {'val loss':>15} {'view acc':>8} " + " ".join(f"{f[:10]:>10}" for f in factors)
    print(head)
    for r in table:
        print(
            f"{r['k']:>3} {r['n_seeds']:>2} {r['val_loss']:>8.4f} ± {r['val_loss_sd']:.3f} {r['content_view_acc']:>8.3f} "
            + " ".join(f"{r['r2_' + f]:>10.3f}" for f in factors)
        )
    print(f"elbow (within {a.elbow_frac:.0%} of the drop): k* = {k_star}")
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps({"k_star": k_star, "table": table, "runs": rows}, indent=1))


if __name__ == "__main__":
    main()
