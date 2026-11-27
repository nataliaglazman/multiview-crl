"""Stage 77, step 1 (no images): does a sufficiency term make a matched-width code keep a faint shared latent?

    python sufficiency_toy.py calibrate OUT.json [--workers 8] [--steps 8000] > calibration.txt
    python sufficiency_toy.py arms OUT.json --share S --noise N [--workers 8] [--steps 8000] > arms.txt
    python sufficiency_toy.py --self-test

A small analogue of the 8-unit problem (HANDOFF stage 77). Seven shared latents, named as on the bank: six strong and
independent, and a faint one (burden) whose parent (ventricle) explains a share of it and whose own part each view
reads with noise of its own. Each view is a fixed random non-linear map of the latents, three style latents of its
own and noise. Per view an MLP backbone (32 wide) and a head shared by the views, as TailCode's, give the code.
Objectives (InfoNCE: cosine on per-unit batch-standardized codes, tau .1, batch 32, the other view's batch as
negatives; teacher: a momentum copy whose backbone features, whitened by running statistics, are the targets):
  baseline  InfoNCE + linear cross-view prediction of the other view's targets (weight 1)
  suff      baseline + sufficiency (weight 1): per view an explainer (MLP, code -> its own view's targets) trained by
            squared error with the code detached; the residuals, explainers frozen, whitened by running statistics;
            the loss pushes down their running cross-view correlation matrix (the batch's part carries the gradient,
            the running matrix the direction), so the gradient reaches the encoder only through the code
  lean      InfoNCE + sufficiency, no prediction
Scored on 1,000 test subjects with own_directions' analysis and holds.py's rule (pool_check.score), plus the leftover's
rho1 on the trained backbone (pool_check.shared_leftover, stage 75's measure). The calibration and the rule are in
HANDOFF stage 77: the setting is chosen on the baseline alone before any sufficiency run.
"""

import argparse
import copy
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from torch import nn

NAMES = ["brain", "ventricle", "burden", "thickness", "temporal", "lr", "sulcal"]
FAINT, PARENT = 2, 1
STYLE, DIM, MIX, HIDDEN = 3, 24, 48, 32
SHARES, NOISES = (0.5, 0.7, 0.85), (0.3, 0.6, 1.0)
BANK_SHARE = 0.7
WORLD_SEED = 1234


def world(share, noise, n_train=2000, n_test=1000, seed=WORLD_SEED):
    """The fixed generator: training and test subjects, each a dict with x (2, n, DIM) and own (n, 7)."""
    rng = np.random.default_rng(seed)
    width = 7 + STYLE
    maps = [
        (rng.standard_normal((width, MIX)) / np.sqrt(width), rng.standard_normal(MIX), rng.standard_normal((MIX, DIM)))
        for _ in range(2)
    ]
    out = []
    for n in (n_train, n_test):
        own = rng.standard_normal((n, 7))
        latents = own.copy()
        latents[:, FAINT] = np.sqrt(share) * own[:, PARENT] + np.sqrt(1 - share) * own[:, FAINT]
        views = []
        for a, b, m in maps:
            seen = latents.copy()
            seen[:, FAINT] += noise * rng.standard_normal(n)
            u = np.column_stack([seen, rng.standard_normal((n, STYLE))])
            views.append(np.tanh(u @ a + b) @ m / np.sqrt(MIX) + 0.05 * rng.standard_normal((n, DIM)))
        out.append({"x": np.stack(views).astype(np.float32), "own": own})
    return out


class Encoder(nn.Module):
    """A backbone per view and a head shared by the views."""

    def __init__(self, units):
        super().__init__()
        self.backbones = nn.ModuleList(
            nn.Sequential(nn.Linear(DIM, 64), nn.SiLU(), nn.Linear(64, HIDDEN), nn.SiLU()) for _ in range(2)
        )
        self.head = nn.Sequential(nn.Linear(HIDDEN, 64), nn.SiLU(), nn.Linear(64, units))

    def forward(self, x):
        h = [b(xv) for b, xv in zip(self.backbones, x)]
        return h, [self.head(hv) for hv in h]


class Running:
    """Running mean and covariance of (n, d) batches, and whitening by them (ridge times the mean variance)."""

    def __init__(self, momentum=0.01, ridge=0.1):
        self.momentum, self.ridge, self.mean, self.cov = momentum, ridge, None, None

    @torch.no_grad()
    def update(self, x):
        mean, cov = x.mean(0), torch.cov(x.T)
        if self.mean is None:
            self.mean, self.cov = mean.clone(), cov.clone()
        else:
            self.mean.lerp_(mean, self.momentum)
            self.cov.lerp_(cov, self.momentum)

    def whiten(self, x):
        c = self.cov + self.ridge * self.cov.diagonal().mean() * torch.eye(len(self.cov))
        w, v = torch.linalg.eigh(c)
        return (x - self.mean) @ (v / w.clamp_min(1e-8).sqrt()) @ v.T


def explainer(units, width):
    return nn.Sequential(nn.Linear(units, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, width))


class Sufficiency(nn.Module):
    """The views' residual cross-correlation given the code: explainers fitted with the code detached, the term's
    gradient reaching the code only (explainers frozen for it)."""

    def __init__(self, units, width, momentum=0.01):
        super().__init__()
        self.explainers = nn.ModuleList(explainer(units, width) for _ in range(2))
        self.stats = [Running(momentum), Running(momentum)]
        self.momentum, self.cross = momentum, None

    def fit_loss(self, z, targets):
        return sum(F.mse_loss(e(zv.detach()), t) for e, zv, t in zip(self.explainers, z, targets)) / 2

    def loss(self, z, targets):
        whitened = []
        for e, s, zv, t in zip(self.explainers, self.stats, z, targets):
            frozen = {k: p.detach() for k, p in e.named_parameters()}
            r = t - torch.func.functional_call(e, frozen, (zv,))
            s.update(r.detach())
            whitened.append(s.whiten(r))
        c = whitened[0].T @ whitened[1] / len(z[0])
        if self.cross is None:
            self.cross = c.detach().clone()
        else:
            self.cross.lerp_(c.detach(), self.momentum)
        return (self.cross * c).sum()

    def value(self):
        """The running matrix's squared singular values, largest first (squared canonical correlations)."""
        return torch.linalg.svdvals(self.cross).square().tolist()


def infonce(z, tau=0.1):
    s = [F.normalize((zv - zv.mean(0)) / (zv.std(0) + 1e-6), dim=1) for zv in z]
    logits = s[0] @ s[1].T / tau
    labels = torch.arange(len(logits))
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def train(data, units, arm, seed, steps, batch=32, lr=1e-3, momentum=0.99):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(data["x"])
    n = x.shape[1]
    model = Encoder(units)
    teacher = copy.deepcopy(model).requires_grad_(False)
    predict = nn.ModuleList(nn.Linear(units, HIDDEN) for _ in range(2))
    suff = Sufficiency(units, HIDDEN) if arm in ("suff", "lean") else None
    params = list(model.parameters())
    if arm != "lean":
        params += list(predict.parameters())
    if suff is not None:
        params += list(suff.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    stats = [Running(), Running()]
    for _ in range(steps):
        xb = x[:, rng.choice(n, batch, replace=False)]
        _, z = model(xb)
        with torch.no_grad():
            th, _ = teacher(xb)
            for s, hv in zip(stats, th):
                s.update(hv)
            targets = [s.whiten(hv) for s, hv in zip(stats, th)]
        loss = infonce(z)
        if arm != "lean":
            loss = loss + (F.mse_loss(predict[0](z[0]), targets[1]) + F.mse_loss(predict[1](z[1]), targets[0])) / 2
        if suff is not None:
            loss = loss + suff.loss(z, targets) + suff.fit_loss(z, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for pt, ps in zip(teacher.parameters(), model.parameters()):
                pt.lerp_(ps, 1 - momentum)
    return model, (None if suff is None else suff.value()[:3])


def r2(features, target):
    predicted = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), features, target, cv=KFold(5))
    return float(1 - ((target - predicted) ** 2).mean() / target.var())


@torch.no_grad()
def encode(model, data):
    h, z = model(torch.as_tensor(data["x"]))
    return np.stack([v.numpy() for v in z]).astype(np.float64), np.stack([v.numpy() for v in h]).astype(np.float64)


def job(spec):
    """One run: spec = (share, noise, units, arm, seed, steps). Returns its scores."""
    import pool_check

    torch.set_num_threads(1)
    share, noise, units, arm, seed, steps = spec
    train_data, test_data = world(share, noise)
    model, running = train(train_data, units, arm, seed, steps)
    code_tr, hid_tr = encode(model, train_data)
    code_te, hid_te = encode(model, test_data)
    s = pool_check.score(code_te, test_data["own"], NAMES)
    left = pool_check.shared_leftover(code_tr, hid_tr, code_te, hid_te)
    return {
        "share": share,
        "noise": noise,
        "units": units,
        "arm": arm,
        "seed": seed,
        "holds": s["holds"],
        "last": s["last"],
        "worst": s["worst"],
        "own": {n: min(s[v]["own_r2"][n] for v in ("T1", "FLAIR")) for n in NAMES},
        "faint_available": float(np.mean([r2(hid_te[v], test_data["own"][:, FAINT]) for v in range(2)])),
        "rho1": float(left["rho"][0]),
        "running_sq_canonical": running,
    }


def run_all(specs, workers):
    with Pool(workers) as pool:
        return pool.map(job, specs)


def strong_mean(row):
    return float(np.mean([row["own"][n] for n in NAMES if n != "burden"]))


def calibrate(out, workers, steps):
    specs = [(s, n, u, "baseline", seed, steps) for s, n in itertools.product(SHARES, NOISES) for u in (8, 12)
             for seed in range(5)]
    rows = run_all(specs, workers)
    print("== Calibration on the baseline alone (HANDOFF stage 77): (a) the 8-unit backbone holds the faint own part")
    print("   (ridge R2 >= .5), (b) 12 units hold in >= 4 of 5 seeds, (c) 8 units fail in >= 3 of 5")
    table, passing = [], []
    for s, n in itertools.product(SHARES, NOISES):
        eight = [r for r in rows if (r["share"], r["noise"], r["units"]) == (s, n, 8)]
        twelve = [r for r in rows if (r["share"], r["noise"], r["units"]) == (s, n, 12)]
        avail = float(np.mean([r["faint_available"] for r in eight]))
        held12, fail8 = sum(r["holds"] for r in twelve), sum(not r["holds"] for r in eight)
        ok = avail >= 0.5 and held12 >= 4 and fail8 >= 3
        faint8 = np.mean([r["own"]["burden"] for r in eight])
        faint12 = np.mean([r["own"]["burden"] for r in twelve])
        print(f"  share {s:.2f} noise {n:.1f}: available {avail:.2f}; 12 units hold {held12}/5 (burden own {faint12:.2f});"
              f" 8 units fail {fail8}/5 (burden own {faint8:.2f}){'  MEETS (a)-(c)' if ok else ''}")
        table.append({"share": s, "noise": n, "available": avail, "held12": held12, "fail8": fail8, "meets": ok})
        if ok:
            passing.append((-fail8, abs(s - BANK_SHARE), n, s))
    chosen = None
    if passing:
        _, _, n, s = min(passing)
        chosen = {"share": s, "noise": n}
        print(f"  CHOSEN: share {s:.2f}, noise {n:.1f}")
    else:
        print("  No setting meets (a)-(c): the toy is uninformative and step 1 stops (HANDOFF stage 77).")
    Path(out).write_text(json.dumps({"rows": rows, "table": table, "chosen": chosen}, indent=1, default=float) + "\n")
    print(f"Saved {out}")


def arms(out, share, noise, workers, steps):
    specs = [(share, noise, 8, arm, seed, steps) for arm in ("baseline", "suff", "lean") for seed in range(10)]
    rows = run_all(specs, workers)
    print(f"== Arms at share {share:.2f}, noise {noise:.1f}, 8 units, seeds 0-9 (rule: HANDOFF stage 77)")
    by = {arm: [r for r in rows if r["arm"] == arm] for arm in ("baseline", "suff", "lean")}
    for arm, rs in by.items():
        for r in rs:
            extra = "" if r["running_sq_canonical"] is None else "  running rho^2 " + " ".join(
                f"{v:.2f}" for v in r["running_sq_canonical"])
            print(f"  {arm:8s} seed {r['seed']}: {'holds' if r['holds'] else 'fails'}  last {r['last'][0]:.2f}/"
                  f"{r['last'][1]:.2f}  burden own {r['own']['burden']:.2f}  strong mean {strong_mean(r):.2f}"
                  f"  rho1 {r['rho1']:.2f}{extra}")
    held = {arm: sum(r["holds"] for r in rs) for arm, rs in by.items()}
    cost = np.mean([strong_mean(r) for r in by["suff"]]) - np.mean([strong_mean(r) for r in by["baseline"]])
    verdict = ("passes" if held["suff"] >= 8 and held["suff"] >= held["baseline"] + 3
               else "fails" if held["suff"] <= held["baseline"] + 1 else "unresolved")
    print(f"\n  holds: baseline {held['baseline']}/10, suff {held['suff']}/10, lean {held['lean']}/10")
    print(f"  strong own parts, suff minus baseline: {cost:+.3f} ({'a cost' if cost < -0.05 else 'no cost'})")
    print(f"  RULE: {verdict}")
    for arm, rs in by.items():
        fresh = [r for r in rs if r["seed"] >= 5]
        print(f"  described, seeds 5-9 only: {arm} holds {sum(r['holds'] for r in fresh)}/5;"
              f" mean rho1 {np.mean([r['rho1'] for r in rs]):.2f}")
    Path(out).write_text(json.dumps({"rows": rows, "held": held, "cost": cost, "verdict": verdict}, indent=1,
                                    default=float) + "\n")
    print(f"Saved {out}")


def self_test():
    """The term's estimate on fixed codes: high when the code lacks a part both views' targets share, low when it
    holds it, after the explainers have been fitted."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n = 4000
    shared, other = rng.standard_normal((n, 1)), rng.standard_normal((n, 3))
    targets = [torch.as_tensor(np.column_stack([shared, other, rng.standard_normal((n, 4))]), dtype=torch.float32)
               for _ in range(2)]
    values = {}
    for name, code in (("lacks", other), ("holds", np.column_stack([shared, other]))):
        z = torch.as_tensor(code, dtype=torch.float32)
        suff = Sufficiency(z.shape[1], 8)
        opt = torch.optim.Adam(suff.parameters(), lr=3e-3)
        for _ in range(1500):
            idx = rng.choice(n, 64, replace=False)
            zb, tb = [z[idx], z[idx]], [t[idx] for t in targets]
            suff.loss(zb, tb)
            opt.zero_grad()
            suff.fit_loss(zb, tb).backward()
            opt.step()
        values[name] = suff.value()[0]
    assert values["lacks"] > 0.6 and values["holds"] < 0.15, values
    print("sufficiency_toy self-test passed", {k: round(v, 3) for k, v in values.items()})


def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("out", type=Path)
    a = sub.add_parser("arms")
    a.add_argument("out", type=Path)
    a.add_argument("--share", type=float, required=True)
    a.add_argument("--noise", type=float, required=True)
    for s in (c, a):
        s.add_argument("--workers", type=int, default=8)
        s.add_argument("--steps", type=int, default=8000)
    args = p.parse_args()
    if args.command == "calibrate":
        calibrate(args.out, args.workers, args.steps)
    else:
        arms(args.out, args.share, args.noise, args.workers, args.steps)


if __name__ == "__main__":
    main()
