"""Does a 10-unit code lose the lesion's volume because its map to the other view is linear? No training of the network.

Stage 35 (HANDOFF section 7, item 25): with --content-units 10 the prediction code reads lesion volume ~5 voxels worse
than with 24. The account tested here: each view's code reaches the other view's whitened GAP through a linear map, so
N units predict at most N of its directions, and the GAP spreads the shared latents over more than 10. Through a
non-linear map, a code that holds the latents in any invertible form can predict all of them, so 10 units would do.

For each finished warm.py run, on its frozen TailCode features (the head's input, per view, from the unjittered scans):
fresh bottlenecks shaped like TailCode's head (features -> 100 -> K, LeakyReLU, shared by the views) are fitted on the
training subjects to predict the other view's GAP, whitened as warm.py --predict-weight whitens it (ridge 0.1), through
the linear map or through the linear map plus a one-hidden-layer MLP of width 128 (warm.py --predict-hidden), at K = 10
and 24, from three seeds each, stopped early on the validation subjects. There is no InfoNCE, so only the map differs.
Each bottleneck is read on the test subjects as precision_audit.py reads a code: e, the lesion-volume reading error in
voxels with the anatomy part handed to the readout (ridge or boosting, the better); "alone", own R2 from the code only;
e in the largest third of lesions; and the mean R2 of the six anatomy factors and of the lesion centre, by ridge and by
boosting. The run's own code is read the same way.

Reading: if the MLP bottleneck at 10 units reads volume about as well as the linear one at 24, and the linear one at 10
does not, the linear map is what limits the 10-unit code, and warm.py --predict-hidden should lift it. The linear one
at 24 should read about as the run's own 24-unit code does; if not, the fit, not the map, sets these numbers.

    python out/burden_feasibility_20261102/bottleneck_check.py OUT_JSON RUN [RUN ...] [--train-subjects N]

Runs already in OUT_JSON at the same step are reused. --train-subjects caps the training subjects (local smoke tests).
"""

import argparse
import copy
import json
from pathlib import Path

import batch_price
import code_audit
import frozen_prediction as FP
import numpy as np
import precision_audit as PA
import torch
import warm
from torch import nn

WIDTHS = (10, 24)
MAPS = {"linear": 0, "MLP": 128}
SEEDS = (0, 1, 2)
RIDGE = 0.1  # warm.py --predict-ridge's default
CLIP = 5.0  # standardized features are clipped at this many SD, as in own_variation.py
EPOCHS, BATCH, LR, DECAY, PATIENCE = 400, 128, 1e-3, 1e-4, 40


class Bottleneck(nn.Module):
    """TailCode's head shape, shared by the views, and warm.py's two prediction maps."""

    def __init__(self, width, units, channels, hidden):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(width, 100), nn.LeakyReLU(), nn.Linear(100, units))
        self.maps = warm.prediction_decoders(units, channels, hidden)

    def loss(self, x, targets):
        return warm.cross_view_prediction(self.head(x), targets, self.maps)


def fit(x, t, held_x, held_t, units, hidden, seed, epochs=EPOCHS, patience=PATIENCE):
    """A bottleneck fitted on (2, N, width) features ``x`` to predict the other view's (2, N, channels) targets ``t``,
    stopped at its best loss on the held-out pair; returns it and that loss."""
    torch.manual_seed(seed)
    model = Bottleneck(x.shape[-1], units, t.shape[-1], hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=DECAY)
    order = torch.Generator().manual_seed(seed)
    best, best_state, waited = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ in range(epochs):
        model.train()
        for batch in torch.randperm(x.shape[1], generator=order).split(BATCH):
            if len(batch) < 2:
                continue
            optimizer.zero_grad()
            model.loss(x[:, batch], t[:, batch]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            held = float(model.loss(held_x, held_t))
        if held < best - 1e-5:
            best, best_state, waited = held, copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), best


def run_features(run, device, train_subjects=None):
    """The run's pooled features (the head's input), GAP and code per split, each (2, N, width), and the labels."""
    report, model, pool, residual, step = code_audit.load_run(run, device, trained=True)
    a = report["args"]
    if pool is None:
        raise SystemExit(f"{run.name}: the code comes from TailCode's head, so the run needs --pool tail or gapnorm")
    out = {"step": step, "bank": a["bank"], "units": pool.main}
    for split in ("train", "val", "test"):
        n = {"train": train_subjects, "val": None, "test": int(a.get("test_subjects", 1000))}[split]
        bank = warm.Bank(a["bank"], split, a["t1key"], n, residual)
        f = warm.features(model, bank, device, pool=pool)
        out[split] = {k: np.stack(f[k]).astype(np.float32) for k in ("pooled", "gap", "code")}
        out[split].update(vox=bank.vox, zc=bank.zc)
    out["test"]["code"] = out["test"]["code"][..., : pool.main]
    return out


def read(code, vox, expected, position, anatomy):
    """precision_audit.py's reading of a (2, N, units) code in each view, plus the mean ridge and boosted R2 of the
    lesion centre and of the anatomy factors."""
    out = {}
    for v, view in enumerate(PA.VIEWS):
        cell, _ = PA.assess_stage(code[v].astype(np.float64), vox, expected)
        for name, target in (("position", position), ("anatomy", anatomy)):
            for kind in ("ridge", "boosted"):
                cell[f"{name}_{kind}"] = float(np.mean([PA.r2(PA.predictions(code[v], y, kind), y) for y in target.T]))
        out[view] = cell
    return out


def assess(run, device, train_subjects=None):
    f = run_features(run, device, train_subjects)
    root, test = f["bank"], f["test"]
    vox = test["vox"]
    expected, explained, _ = PA.own_volume(root, vox)
    position = batch_price.lesion_centroid(np.load(f"{root}/test_cells.npy", mmap_mode="r")[: len(vox)])
    anatomy = test["zc"][:, warm.ANATOMY]
    mean = f["train"]["pooled"].mean(1, keepdims=True)
    sd = f["train"]["pooled"].std(1, keepdims=True) + 1e-6

    def inputs(split):
        return torch.from_numpy(np.clip((f[split]["pooled"] - mean) / sd, -CLIP, CLIP))

    whitening = FP.fit_whitening(torch.from_numpy(f["train"]["gap"]), RIDGE)

    def targets(split):
        return FP.whiten(torch.from_numpy(f[split]["gap"]), whitening)

    x, t, held_x, held_t, test_x = inputs("train"), targets("train"), inputs("val"), targets("val"), inputs("test")
    out = {
        "step": f["step"],
        "bank": root,
        "rows": len(vox),
        "train_rows": int(x.shape[1]),
        "own_sd": float((vox - expected).std()),
        "explained": explained,
        "own_code": {"units": f["units"], **read(test["code"], vox, expected, position, anatomy)},
        "bottlenecks": {},
    }
    for units in WIDTHS:
        for label, hidden in MAPS.items():
            fits = []
            for seed in SEEDS:
                model, held = fit(x, t, held_x, held_t, units, hidden, seed)
                with torch.no_grad():
                    code = model.head(test_x).numpy()
                fits.append({"held_out": held, **read(code, vox, expected, position, anatomy)})
            out["bottlenecks"][f"{label} {units}"] = fits
            print(f"  {label} map, {units} units done", flush=True)
    return out


def mean_over(fits, view, get):
    values = [get(f[view]) for f in fits]
    return None if any(v is None for v in values) else float(np.mean(values))


def voxels(value):
    return PA.voxels(value, 4).strip()


def show(name, row):
    print(
        f"{name}  step {row['step']}, bank {Path(row['bank']).name}, {row['rows']} test subjects, bottlenecks fitted on"
        f" {row['train_rows']} training subjects; burden's own part SD {row['own_sd']:.1f} vox"
    )
    print(
        f"    {'code':10s} {'units':>5s} {'held-out':>8s} | {'e T1 / FLAIR':>13s} | {'spread':>13s} | {'alone':>13s} |"
        f" {'largest':>11s} | {'anatomy r/b, T1  FLAIR':>25s} | {'position r/b, T1  FLAIR':>25s}"
    )
    entries = [("run's own", row["own_code"]["units"], "-", [row["own_code"]])]
    for key, fits in row["bottlenecks"].items():
        label, units = key.split()
        entries.append((f"{label} map", int(units), f"{np.mean([f['held_out'] for f in fits]):.3f}", fits))
    for label, units, held, fits in entries:

        def both(get, show_value):
            return " / ".join(show_value(mean_over(fits, v, get)) for v in PA.VIEWS)

        spread = "-"
        if len(fits) > 1:
            errors = [[f[v]["error"] for f in fits] for v in PA.VIEWS]
            spread = " / ".join("-" if None in e else f"{max(e) - min(e):.1f}" for e in errors)
        factors = {
            name: "  ".join(
                f"{mean_over(fits, v, lambda c: c[f'{name}_ridge']):.2f}/"
                f"{mean_over(fits, v, lambda c: c[f'{name}_boosted']):.2f}"
                for v in PA.VIEWS
            )
            for name in ("anatomy", "position")
        }
        print(
            f"    {label:10s} {units:5d} {held:>8s} | {both(lambda c: c['error'], voxels):>13s} | {spread:>13s} |"
            f" {both(lambda c: c['alone_r2'], lambda x: f'{x:.2f}'):>13s} |"
            f" {both(lambda c: c['error_by_size'][2], voxels):>11s} | {factors['anatomy']:>25s} |"
            f" {factors['position']:>25s}"
        )
    print(
        "    e: lesion-volume reading error in voxels, the anatomy part handed to the readout; alone: own R2 from the"
        " code only;\n    largest: e in the largest third of lesions; anatomy and position: mean R2 of the six anatomy"
        " factors and of the lesion centre,\n    ridge / boosted; held-out: the prediction loss on validation"
        f" subjects. Bottleneck rows: means over {len(SEEDS)} seeds; spread: max - min of e over them."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_json", type=Path)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--train-subjects", type=int, default=None, help="cap on the training subjects (smoke tests)")
    a = p.parse_args()
    device = warm.pick_device()
    results = json.loads(a.out_json.read_text()) if a.out_json.exists() else {}
    for run in a.runs:
        try:
            step = int(torch.load(run / "model.pt", map_location="cpu")["step"])
            last = int(json.loads((run / "report.json").read_text())["args"]["steps"])
        except Exception as error:  # no checkpoint yet, or one being written right now
            print("skipping", run.name, "for now:", error, flush=True)
            continue
        if step != last or results.get(run.name, {}).get("step") == step:
            continue
        print("assessing", run.name, flush=True)
        results[run.name] = assess(run, device, a.train_subjects)
        a.out_json.write_text(json.dumps(results, indent=1))
    for name in sorted(results):
        show(name, results[name])


if __name__ == "__main__":
    main()
