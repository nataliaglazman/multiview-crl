"""Stress test of the pay result: exact weighted loss, fair position code, hard ceiling, position-free targets."""
import os

import numpy as np
import torch

torch.set_num_threads(3)
here = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(here, "targets_cache.npz"))
G = 16
tr_t, tr_k, tr_z, tr_l = d["train_target"], d["train_keep"], d["train_z"], d["train_lesion"].reshape(-1, G, G, G)
va_t, va_k, va_z, va_l = d["val_target"], d["val_keep"], d["val_z"], d["val_lesion"].reshape(-1, G, G, G)
tmpl = (tr_t * tr_k).sum(0) / np.maximum(tr_k.sum(0), 1)
tr_r, va_r = (tr_t - tmpl) * tr_k, (va_t - tmpl) * va_k
axes = np.stack(np.meshgrid(*[np.arange(G)] * 3, indexing="ij"))


def centroid(l):
    w = l / np.maximum(l.sum((1, 2, 3), keepdims=True), 1e-9)
    return np.stack([(w * a).sum((1, 2, 3)) for a in axes], 1) / (G - 1) * 2 - 1


tr_c, va_c = centroid(tr_l), centroid(va_l)
anat = [i for i in range(tr_z.shape[1]) if i not in (2, 3, 4)]
rng = np.random.RandomState(0)
B = rng.normal(size=(32, 3)) * 4.0
fourier = lambda c: np.concatenate([np.sin(c @ B.T), np.cos(c @ B.T)], 1)


def inputs(kind, z, c):
    parts = {"anatomy": z[:, anat], "burden": z[:, [2]], "position": c, "fourier": fourier(c)}
    return np.concatenate([parts[k] for k in kind], 1)


def weights(y, k):
    w = np.minimum(1 + np.abs(y), 4.0)
    return w / ((w * k).sum((1, 2, 3), keepdims=True) / k.sum((1, 2, 3), keepdims=True)) * k


def fit(kind, view, hidden=128, steps=8000, lr=1e-3, wd=1e-2, bs=64):
    torch.manual_seed(0)
    Y, K = tr_r[:, view], tr_k[:, view].astype("f4")
    W = weights(Y, K)
    X, Y, W, K = [torch.tensor(a, dtype=torch.float32) for a in (inputs(kind, tr_z, tr_c), Y, W, K)]
    Xv = torch.tensor(inputs(kind, va_z, va_c), dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, G**3))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        i = torch.randint(len(X), (bs,))
        e = (net(X[i]).reshape(bs, G, G, G) - Y[i]) ** 2 * W[i]
        loss = (e.sum((1, 2, 3)) / K[i].sum((1, 2, 3))).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        return net(X).reshape(-1, G, G, G).numpy(), net(Xv).reshape(-1, G, G, G).numpy()


def wmse(P, y, k):
    return float(((P - y) ** 2 * weights(y, k)).sum((1, 2, 3)).mean() / k.sum((1, 2, 3)).mean())


res = {}
for view, vname in ((0, "T1"), (1, "FLAIR")):
    yt, yv, kt, kv = tr_r[:, view], va_r[:, view], tr_k[:, view].astype("f4"), va_k[:, view].astype("f4")
    print(f"\n=== predicting {vname}: template-only weighted loss, val {wmse(0 * yv, yv, kv):.5f}")
    base = None
    for label, kind, h in (
        ("anatomy", ("anatomy",), 128),
        ("anatomy+burden", ("anatomy", "burden"), 128),
        ("anat+burden+xyz", ("anatomy", "burden", "position"), 128),
        ("anat+burden+fourier64", ("anatomy", "burden", "fourier"), 128),
        ("anat+burden+fourier64 h1024", ("anatomy", "burden", "fourier"), 1024),
    ):
        Pt, Pv = fit(kind, view, h)
        tr, va = wmse(Pt, yt, kt), wmse(Pv, yv, kv)
        if base is None:
            base = (tr, va, Pt, Pv)
        print(
            f"{label:<30s} train {tr:.5f} (pay {100 * (base[0] - tr) / base[0]:+6.2f}%)   val {va:.5f} (pay {100 * (base[1] - va) / base[1]:+6.2f}%)",
            flush=True,
        )
    # ceiling: hand the anatomy-only decoder the TRUE lesion map, one fitted amplitude per view
    x_t, x_v = tr_l * kt, va_l * kv
    e_t = (yt - base[2]) * kt
    a = float((x_t * e_t).sum() / (x_t * x_t).sum())
    Pc = base[3] + a * x_v
    print(
        f"{'CEILING: anatomy + true lesion map':<30s} amplitude {a:+.2f}   val {wmse(Pc, yv, kv):.5f} (pay {100 * (base[1] - wmse(Pc, yv, kv)) / base[1]:+6.2f}%)",
        flush=True,
    )
