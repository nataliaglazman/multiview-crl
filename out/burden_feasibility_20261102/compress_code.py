"""Compress a trained code to fewer units without labels, as the start of a narrower run (stage 65).

    python compress_code.py RUN OUT.pt --units 8 [--method redundancy|pca] [--seed 0]

The code of RUN's training subjects (both views stacked) is standardized per unit, as the loss does, and whitened, so
no direction is linearly predictable from the others. ``redundancy`` then removes, one at a time, the direction a
small MLP predicts best from everything orthogonal to it (R2 on held-out subjects): a curve, a bent copy of what
the code already holds, is predictable this way; a factor's own part, independent of the rest, is not. ``pca`` is
the naive control: it drops the lowest-variance principal directions of the standardized code. Either way the kept
directions, orthonormal in the whitened space, become the new head's units: OUT.pt holds RUN's network and TailCode
with the last head layer rewritten to the new width, for ``warm.py --init-run``. No label is read.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import code_audit
import warm


def whiten(codes):
    """Pooled standardization and PCA whitening of (N, d) codes: returns mean, SD, and W with x = W (c - mean) / SD."""
    mean, sd = codes.mean(0), codes.std(0) + 1e-8
    s = (codes - mean) / sd
    values, vectors = np.linalg.eigh(np.cov(s.T))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    return mean, sd, (vectors / np.sqrt(np.maximum(values, 1e-10))).T, values


def predictable(x_fit, x_val, seed, starts=None, restarts=6, steps=1500, hidden=64):
    """The unit direction w of whitened x best predicted, out of sample, from x's part orthogonal to w by an MLP.

    Returns (w, held-out R2). Starts from each given direction and ``restarts`` random ones; keeps the best on x_val.
    """
    d = x_fit.shape[1]
    torch.manual_seed(seed)
    fit, val = torch.as_tensor(x_fit, dtype=torch.float32), torch.as_tensor(x_val, dtype=torch.float32)
    inits = [] if starts is None else [torch.as_tensor(s, dtype=torch.float32) for s in starts]
    inits += [torch.randn(d) for _ in range(restarts)]
    best = (None, -np.inf)
    for init in inits:
        w = nn.Parameter(init / init.norm())
        mlp = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        opt = torch.optim.Adam([{"params": [w], "lr": 1e-2}, {"params": mlp.parameters(), "lr": 3e-3}])

        def error(x):
            u = w / w.norm()
            target = x @ u
            return ((mlp(x - target[:, None] * u).squeeze(1) - target) ** 2).mean() / target.var()

        for _ in range(steps):
            opt.zero_grad()
            error(fit).backward()
            opt.step()
        with torch.no_grad():
            r2 = 1 - float(error(val))
            u = (w / w.norm()).numpy().astype(np.float64)
        if r2 > best[1]:
            best = (u, r2)
    return best


def complement(u):
    """An orthonormal (d, d-1) basis of the directions orthogonal to unit vector u."""
    q, _ = np.linalg.qr(np.column_stack([u, np.eye(len(u))]))
    return q[:, 1:]


def redundancy(x, units, subjects, seed):
    """Remove, one at a time, the most predictable direction; return the kept (d, units) basis and the steps' log."""
    rng = np.random.default_rng(seed)
    held = rng.permutation(subjects)[: subjects // 2]
    val_rows = np.isin(np.arange(len(x)) % subjects, held)
    basis = np.eye(x.shape[1])
    log = []
    while basis.shape[1] > units:
        y = x @ basis
        # every remaining direction's own predictability, for the record, from the current axes
        u, r2 = predictable(y[~val_rows], y[val_rows], seed + len(log), starts=list(np.eye(y.shape[1])))
        log.append({"held_out_r2": round(r2, 4), "direction": (basis @ u).round(4).tolist()})
        basis = basis @ complement(u)
    return basis, log


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--units", type=int, required=True)
    p.add_argument("--method", choices=("redundancy", "pca"), default="redundancy")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    device = warm.pick_device()
    report, model, pool, residual, step = code_audit.load_run(a.run, device)
    args = report["args"]
    if pool is None:
        raise SystemExit("The code comes from TailCode's head: the run needs --pool tail or gapnorm")
    if pool.units != pool.main:
        raise SystemExit("Remainder units are not supported")
    width = pool.units
    if not 1 <= a.units < width:
        raise SystemExit(f"--units must be below the run's {width}")
    train = warm.Bank(args["bank"], "train", args["t1key"], residual=residual)
    codes = np.concatenate(warm.features(model, train, device, pool=pool)["code"])  # view-major, (2N, width)
    subjects = len(train.vox)
    mean, sd, white, variances = whiten(codes)
    x = (codes - mean) / sd @ white.T
    if a.method == "pca":
        basis, log = np.eye(width)[:, : a.units], [{"dropped_variances": variances[a.units :].round(4).tolist()}]
    else:
        basis, log = redundancy(x, a.units, subjects, a.seed)
    mix = basis.T @ white / sd  # new code = mix @ (c - mean)
    head = pool.head[2]
    weight, bias = head.weight.detach().cpu().double().numpy(), head.bias.detach().cpu().double().numpy()
    new_weight = np.concatenate([mix @ weight[:width], weight[width:]])
    new_bias = np.concatenate([mix @ (bias[:width] - mean), bias[width:]])
    state = {k: v.detach().cpu().clone() for k, v in pool.state_dict().items()}
    state["head.2.weight"] = torch.as_tensor(new_weight, dtype=head.weight.dtype)
    state["head.2.bias"] = torch.as_tensor(new_bias, dtype=head.bias.dtype)
    new = (codes - mean) @ mix.T
    check = float(np.abs(np.cov(((new - new.mean(0)) / new.std(0)).T) - np.eye(a.units)).max())
    record = {
        "run": str(a.run),
        "step": step,
        "method": a.method,
        "units": [width, a.units],
        "seed": a.seed,
        "train_rows": len(codes),
        "variances": variances.round(4).tolist(),
        "removed": log,
        "kept_correlation_max_offdiag": round(check, 6),
    }
    torch.save({"model": model.state_dict(), "pool": state, "step": 0, "compression": record}, a.out)
    a.out.with_suffix(".json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "removed"}))
    for row in log:
        print({k: v for k, v in row.items() if k != "direction"})
    print("SAVED", a.out)


if __name__ == "__main__":
    main()
