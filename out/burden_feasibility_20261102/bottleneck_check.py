"""Does a 10-unit code lose the lesion's volume because of how it predicts the other view? No training of the network.

Stage 36 (HANDOFF section 7, item 26) asked whether the map was the reason: each view's code reaches the other view's
whitened GAP through a linear map, so N units predict at most N of its directions. A non-linear map did not help.
Stage 39 asks whether the target is the reason. The 10-unit codes spend about half their directions on the lesion's
position (item 29), and the GAP they predict is full of position structure. TailCode's tail features are means over
the interior cells, so they carry the lesion's volume and focal anatomy and little of its position. Stage 40 asks
whether the head is the reason: every check so far kept TailCode's head (features -> 100 -> K). --heads tries wider or
deeper ones ("512": one hidden layer of 512 units; "512x2": two), with the same maps.

For each finished warm.py run, on its frozen TailCode features (the head's input, per view, from the unjittered scans):
fresh bottlenecks shaped like TailCode's head (features -> 100 -> K, LeakyReLU, shared by the views; --heads changes
its hidden layers) are fitted on the training subjects to predict the other view's target, whitened as warm.py
--predict-weight whitens the GAP (ridge 0.1), through the linear map or through the linear map plus a one-hidden-layer
MLP of width 128 (warm.py --predict-hidden), at K = 10 and 24, from three seeds each, stopped early on the validation
subjects. There is no InfoNCE. --targets:

  gap     the other view's backbone GAP (64): what warm.py --predict-weight predicts (the default)
  tails   the other view's tail features (384): the pooled vector without its GAP
  pooled  the other view's whole pooled vector, the head's input (448): the GAP and the tails together

Each bottleneck is read on the test subjects as precision_audit.py reads a code: e, the lesion-volume reading error in
voxels with the anatomy part handed to the readout (ridge or boosting, the better); "alone", own R2 from the code only;
e in the largest third of lesions; and the R2 of each anatomy factor and lesion-centre coordinate by ridge and by
boosting. The run's own code is read the same way.

Reading: a target that lets the 10-unit bottleneck read volume about as well as the 24-unit one, while keeping the
lesion position and the anatomy, is what a 10-unit training arm should predict. A head that does so means the narrow
code is limited by the head's capacity, not by what the objective ranks first. The GAP rows of TailCode's head use
stage 36's seeds, so they repeat its check.

    python out/burden_feasibility_20261102/bottleneck_check.py OUT_JSON RUN [RUN ...] [--targets gap tails pooled]
        [--heads 100 512 512x2] [--train-subjects N]

Bottlenecks already in OUT_JSON for a run at the same step are reused, so a rerun fits only what is missing. The GAP
target's entries with TailCode's head keep stage 36's names ("MLP 10"); the others carry their target ("MLP 10 tails")
and their head ("MLP 10 head512x2").
--train-subjects caps the training subjects (local smoke tests).
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
TARGETS = ("gap", "tails", "pooled")
HEAD = "100"  # TailCode's head: one hidden layer of 100 units
RIDGE = 0.1  # warm.py --predict-ridge's default
CLIP = 5.0  # standardized features are clipped at this many SD, as in own_variation.py
EPOCHS, BATCH, LR, DECAY, PATIENCE = 400, 128, 1e-3, 1e-4, 40
FACTOR_NAMES = [warm.FACTORS[i] for i in warm.ANATOMY] + ["lesion x", "lesion y", "lesion z"]


def head_layers(name):
    """A head's hidden widths from its name: "512" is one hidden layer of 512 units, "512x2" two of them."""
    width, cross, depth = name.partition("x")
    if not width.isdigit() or int(width) < 1 or (cross and (not depth.isdigit() or int(depth) < 1)):
        raise ValueError(f"a head is WIDTH or WIDTHxDEPTH, got {name!r}")
    return (int(width),) * (int(depth) if cross else 1)


class Bottleneck(nn.Module):
    """A head shaped like TailCode's, with hidden widths ``layers`` (TailCode's: one of 100), shared by the views, and
    warm.py's two prediction maps."""

    def __init__(self, width, units, channels, hidden, layers=(100,)):
        super().__init__()
        parts, last = [], width
        for size in layers:
            parts += [nn.Linear(last, size), nn.LeakyReLU()]
            last = size
        self.head = nn.Sequential(*parts, nn.Linear(last, units))
        self.maps = warm.prediction_decoders(units, channels, hidden)

    def loss(self, x, targets):
        return warm.cross_view_prediction(self.head(x), targets, self.maps)


def fit(x, t, held_x, held_t, units, hidden, seed, epochs=EPOCHS, patience=PATIENCE, layers=(100,)):
    """A bottleneck fitted on (2, N, width) features ``x`` to predict the other view's (2, N, channels) targets ``t``,
    stopped at its best loss on the held-out pair; returns it and that loss."""
    torch.manual_seed(seed)
    model = Bottleneck(x.shape[-1], units, t.shape[-1], hidden, layers)
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
    out = {
        "step": step,
        "bank": a["bank"],
        "units": pool.main,
        "tail_start": pool.average_width,
        "tails": bool(pool.thresholds),
    }
    for split in ("train", "val", "test"):
        n = {"train": train_subjects, "val": None, "test": int(a.get("test_subjects", 1000))}[split]
        bank = warm.Bank(a["bank"], split, a["t1key"], n, residual)
        f = warm.features(model, bank, device, pool=pool)
        out[split] = {k: np.stack(f[k]).astype(np.float32) for k in ("pooled", "gap", "code")}
        out[split].update(vox=bank.vox, zc=bank.zc)
    out["test"]["code"] = out["test"]["code"][..., : pool.main]
    return out


def target_of(split, target, tail_start):
    """One split's (2, N, width) target: the GAP, the tail features (the pooled vector after its averages) or both."""
    if target == "gap":
        return split["gap"]
    return split["pooled"][..., tail_start:] if target == "tails" else split["pooled"]


def key_of(label, units, target, head=HEAD):
    """A bottleneck's name in OUT_JSON: stage 36's ("MLP 10") for the GAP target and TailCode's head, with the target
    ("MLP 10 tails") and the head ("MLP 10 head512x2") added otherwise."""
    return " ".join([label, str(units)] + [target] * (target != "gap") + [f"head{head}"] * (head != HEAD))


def parse_key(key):
    label, units, *rest = key.split()
    target = next((part for part in rest if part in TARGETS), "gap")
    head = next((part[4:] for part in rest if part.startswith("head")), HEAD)
    return label, int(units), target, head


def spread(target):
    """About how many directions a (2, N, width) target spreads over: the participation ratio of each view's
    correlation eigenvalues (warm.py's target_spread), averaged over the views."""
    ratios = []
    for view in np.asarray(target, dtype=np.float64):
        sd = view.std(0)
        z = (view[:, sd > 0] - view[:, sd > 0].mean(0)) / sd[sd > 0]
        values = np.clip(np.linalg.eigvalsh(z.T @ z / len(z)), 0, None)
        ratios.append(values.sum() ** 2 / np.square(values).sum())
    return float(np.mean(ratios))


def read(code, vox, expected, position, anatomy):
    """precision_audit.py's reading of a (2, N, units) code in each view, plus the ridge and boosted R2 of each anatomy
    factor and lesion-centre coordinate, and their means."""
    out = {}
    for v, view in enumerate(PA.VIEWS):
        cell, _ = PA.assess_stage(code[v].astype(np.float64), vox, expected)
        for name, target in (("position", position), ("anatomy", anatomy)):
            for kind in ("ridge", "boosted"):
                each = [PA.r2(PA.predictions(code[v], y, kind), y) for y in target.T]
                cell[f"{name}_{kind}"] = float(np.mean(each))
                cell[f"{name}_{kind}_each"] = [float(r) for r in each]
        out[view] = cell
    return out


def labels(root, vox):
    """For the test subjects: the volume the other true latents predict, its share of the variance, and the lesion
    centre."""
    expected, explained, _ = PA.own_volume(root, vox)
    position = batch_price.lesion_centroid(np.load(f"{root}/test_cells.npy", mmap_mode="r")[: len(vox)])
    return expected, explained, position


def assess(run, device, targets, done=None, train_subjects=None, save=None, heads=(HEAD,)):
    """The run's own code and its bottlenecks for each of ``targets`` and ``heads``. ``done`` (the run's row at the same
    step) keeps the bottlenecks it already has; ``save`` is called with the row after each new one."""
    f = run_features(run, device, train_subjects)
    if not f["tails"] and set(targets) & {"tails", "pooled"}:
        raise SystemExit(f"{run.name}: --pool gapnorm has no tail features")
    test = f["test"]
    vox = test["vox"]
    expected, explained, position = labels(f["bank"], vox)
    anatomy = test["zc"][:, warm.ANATOMY]
    mean = f["train"]["pooled"].mean(1, keepdims=True)
    sd = f["train"]["pooled"].std(1, keepdims=True) + 1e-6

    def inputs(split):
        return torch.from_numpy(np.clip((f[split]["pooled"] - mean) / sd, -CLIP, CLIP))

    x, held_x, test_x = inputs("train"), inputs("val"), inputs("test")
    if done is not None and done.get("step") == f["step"] and done.get("train_rows") == int(x.shape[1]):
        out = done
    else:
        out = {
            "step": f["step"],
            "bank": f["bank"],
            "rows": len(vox),
            "train_rows": int(x.shape[1]),
            "own_sd": float((vox - expected).std()),
            "explained": explained,
            "own_code": {"units": f["units"], **read(test["code"], vox, expected, position, anatomy)},
            "bottlenecks": {},
        }
    out.setdefault("targets", {})
    for target in targets:
        train_target = target_of(f["train"], target, f["tail_start"])
        whitening = FP.fit_whitening(torch.from_numpy(train_target), RIDGE)
        t, held_t = (
            FP.whiten(torch.from_numpy(target_of(f[split], target, f["tail_start"])), whitening)
            for split in ("train", "val")
        )
        out["targets"][target] = {"width": int(train_target.shape[-1]), "spread": spread(train_target)}
        for head in heads:
            for units in WIDTHS:
                for label, hidden in MAPS.items():
                    key = key_of(label, units, target, head)
                    if key in out["bottlenecks"]:
                        continue
                    fits = []
                    for seed in SEEDS:
                        model, held = fit(x, t, held_x, held_t, units, hidden, seed, layers=head_layers(head))
                        with torch.no_grad():
                            code = model.head(test_x).numpy()
                        fits.append({"held_out": held, **read(code, vox, expected, position, anatomy)})
                    out["bottlenecks"][key] = fits
                    if save is not None:
                        save(out)
                    print(f"  {target}, head {head}: {label} map, {units} units done", flush=True)
    return out


def mean_over(fits, view, get):
    values = [get(f[view]) for f in fits]
    return None if any(v is None for v in values) else float(np.mean(values))


def voxels(value):
    return PA.voxels(value, 4).strip()


def order(key):
    label, units, target, head = parse_key(key)
    layers = head_layers(head)
    return (
        TARGETS.index(target) if target in TARGETS else len(TARGETS),
        units,
        label != "linear",
        label,
        len(layers),
        layers,
    )


def show(name, row):
    print(
        f"{name}  step {row['step']}, bank {Path(row['bank']).name}, {row['rows']} test subjects, bottlenecks fitted on"
        f" {row['train_rows']} training subjects; burden's own part SD {row['own_sd']:.1f} vox"
    )
    targets = row.get("targets", {})
    if targets:
        print(
            "    targets: "
            + "; ".join(
                f"{t} {targets[t]['width']} features over ~{targets[t]['spread']:.1f} directions"
                for t in TARGETS
                if t in targets
            )
        )
    print(
        f"    {'code':10s} {'target':>6s} {'head':>6s} {'units':>5s} {'held-out':>8s} | {'e T1 / FLAIR':>13s} |"
        f" {'seeds':>11s} | {'alone':>11s} | {'largest':>11s} | {'anatomy r/b, T1  FLAIR':>25s} |"
        f" {'position r/b, T1  FLAIR':>25s}"
    )
    entries = [("run's own", "-", "-", row["own_code"]["units"], "-", [row["own_code"]])]
    for key in sorted(row["bottlenecks"], key=order):
        label, units, target, head = parse_key(key)
        fits = row["bottlenecks"][key]
        entries.append((f"{label} map", target, head, units, f"{np.mean([f['held_out'] for f in fits]):.3f}", fits))
    for label, target, head, units, held, fits in entries:

        def both(get, show_value):
            return " / ".join(show_value(mean_over(fits, v, get)) for v in PA.VIEWS)

        seeds = "-"
        if len(fits) > 1:
            errors = [[f[v]["error"] for f in fits] for v in PA.VIEWS]
            seeds = " / ".join("-" if None in e else f"{max(e) - min(e):.1f}" for e in errors)
        factors = {
            name: "  ".join(
                f"{mean_over(fits, v, lambda c: c[f'{name}_ridge']):.2f}/"
                f"{mean_over(fits, v, lambda c: c[f'{name}_boosted']):.2f}"
                for v in PA.VIEWS
            )
            for name in ("anatomy", "position")
        }
        print(
            f"    {label:10s} {target:>6s} {head:>6s} {units:5d} {held:>8s} |"
            f" {both(lambda c: c['error'], voxels):>13s} |"
            f" {seeds:>11s} | {both(lambda c: c['alone_r2'], lambda x: f'{x:.2f}'):>11s} |"
            f" {both(lambda c: c['error_by_size'][2], voxels):>11s} | {factors['anatomy']:>25s} |"
            f" {factors['position']:>25s}"
        )
    each = [entry for entry in entries if "anatomy_boosted_each" in entry[5][0][PA.VIEWS[0]]]
    if each:
        left = "    boosted R2 of each factor, mean of the views and seeds:"
        print(f"{left:<64s} " + " ".join(f"{n.split()[-1][:5]:>5s}" for n in FACTOR_NAMES))
        for label, target, head, units, _, fits in each:
            values = [
                np.mean([f[v][f"{name}_boosted_each"][i] for f in fits for v in PA.VIEWS])
                for name, count in (("anatomy", len(warm.ANATOMY)), ("position", 3))
                for i in range(count)
            ]
            left = f"    {label:10s} {target:>6s} {head:>6s} {units:5d}"
            print(f"{left:<64s} " + " ".join(f"{x:5.2f}" for x in values))
    print(
        "    e: lesion-volume reading error in voxels, the anatomy part handed to the readout; alone: own R2 from the"
        " code only;\n    largest: e in the largest third of lesions; anatomy and position: mean R2 of the six anatomy"
        " factors and of the lesion centre,\n    ridge / boosted; held-out: the prediction loss on validation"
        f" subjects. Bottleneck rows: means over {len(SEEDS)} seeds; seeds: max - min of e over them."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_json", type=Path)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--targets", nargs="+", choices=TARGETS, default=["gap"], help="what the bottlenecks predict (gap)")
    p.add_argument("--heads", nargs="+", default=[HEAD], help="the heads' hidden layers, WIDTH or WIDTHxDEPTH (100)")
    p.add_argument("--train-subjects", type=int, default=None, help="cap on the training subjects (smoke tests)")
    a = p.parse_args()
    for head in a.heads:
        try:
            head_layers(head)
        except ValueError as error:
            p.error(str(error))
    device = warm.pick_device()
    results = json.loads(a.out_json.read_text()) if a.out_json.exists() else {}

    def saver(name):
        def save(row):
            results[name] = row
            a.out_json.write_text(json.dumps(results, indent=1))

        return save

    wanted = [
        key_of(label, units, target, head)
        for target in a.targets
        for head in a.heads
        for units in WIDTHS
        for label in MAPS
    ]
    for run in a.runs:
        try:
            step = int(torch.load(run / "model.pt", map_location="cpu")["step"])
            last = int(json.loads((run / "report.json").read_text())["args"]["steps"])
        except Exception as error:  # no checkpoint yet, or one being written right now
            print("skipping", run.name, "for now:", error, flush=True)
            continue
        cached = results.get(run.name, {})
        if step != last or (cached.get("step") == step and all(k in cached["bottlenecks"] for k in wanted)):
            continue
        print("assessing", run.name, flush=True)
        results[run.name] = assess(run, device, a.targets, cached, a.train_subjects, saver(run.name), a.heads)
        a.out_json.write_text(json.dumps(results, indent=1))
    for name in sorted(results):
        show(name, results[name])


if __name__ == "__main__":
    main()
