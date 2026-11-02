"""What the vector-prediction loss pays for each kind of information: the recipe's decoder fed true latents."""
import os
import time

import numpy as np
import torch

torch.set_num_threads(4)
here = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(here, "targets_cache.npz"))
G = 16


def split(s):
    t, k, z, l = d[f"{s}_target"], d[f"{s}_keep"], d[f"{s}_z"], d[f"{s}_lesion"]
    return t, k, z, l.reshape(-1, G, G, G)


tr_t, tr_k, tr_z, tr_l = split("train")
va_t, va_k, va_z, va_l = split("val")
# template per view, occupancy-free mean over kept cells
tmpl = (tr_t * tr_k).sum(0) / np.maximum(tr_k.sum(0), 1)
tr_r, va_r = (tr_t - tmpl) * tr_k, (va_t - tmpl) * va_k
axes = np.stack(np.meshgrid(*[np.arange(G)] * 3, indexing="ij"))


def centroid(l):
    w = l / np.maximum(l.sum((1, 2, 3), keepdims=True), 1e-9)
    return np.stack([(w * a).sum((1, 2, 3)) for a in axes], 1) / (G - 1) * 2 - 1


tr_c, va_c = centroid(tr_l), centroid(va_l)
vol = lambda l: l.sum((1, 2, 3)) * 64  # voxels at res 64
print(
    "rendered lesion voxels, val: median",
    np.median(vol(va_l)),
    "quartiles",
    np.percentile(vol(va_l), [25, 75]),
    "max",
    vol(va_l).max(),
)
print("z shape", tr_z.shape)
names = ["brain_size", "ventricle_size", "lesion_burden", "unused_3", "unused_4", "gyrification?", "?", "?", "?"]
anat = [i for i in range(tr_z.shape[1]) if i not in (2, 3, 4)]


def inputs(kind, z, c):
    parts = {"anatomy": z[:, anat], "burden": z[:, [2]], "position": c}
    return np.concatenate([parts[k] for k in kind], 1) if kind else np.zeros((len(z), 1))


def fit(kind, hidden=128, view=1, steps=6000, lr=1e-3, scale=1.0, offset=0.0, bs=64, seed=0):
    torch.manual_seed(seed)
    X, Y, K = [
        torch.tensor(a, dtype=torch.float32)
        for a in (inputs(kind, tr_z, tr_c), tr_r[:, view], tr_k[:, view].astype("f4"))
    ]
    Xv, Yv, Kv = [
        torch.tensor(a, dtype=torch.float32)
        for a in (inputs(kind, va_z, va_c), va_r[:, view], va_k[:, view].astype("f4"))
    ]
    X, Xv = X * scale + offset, Xv * scale + offset
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, G**3))
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if lr > 2e-4 else None
    for s in range(steps):
        i = torch.randint(len(X), (bs,))
        e = (net(X[i]).reshape(bs, G, G, G) - Y[i]) ** 2 * K[i]
        loss = (e.sum((1, 2, 3)) / K[i].sum((1, 2, 3))).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched:
            sched.step()
    with torch.no_grad():
        P = net(Xv).reshape(-1, G, G, G)
    return P.numpy()


def report(P, view):
    k, y = va_k[:, view].astype(bool), va_r[:, view]
    mse = ((P - y) ** 2 * k).sum((1, 2, 3)) / k.sum((1, 2, 3))
    les = va_l > 0.5 * va_l.reshape(len(va_l), -1).max(1)[:, None, None, None]
    les = les & k
    le = ((P - y) ** 2)[les].mean() if les.any() else float("nan")
    # how much of the true deviation at lesion cells the prediction recovers
    slope = float((P[les] * y[les]).sum() / (y[les] ** 2).sum())
    return float(mse.mean()), float(le), slope


rows = {}
views = {0: "T1 target", 1: "FLAIR target"}
for view, vname in views.items():
    base = float((((va_r[:, view]) ** 2 * va_k[:, view]).sum((1, 2, 3)) / va_k[:, view].sum((1, 2, 3))).mean())
    print(f"\n=== {vname}: template-only MSE {base:.5f}")
    for label, kind, hidden in (
        ("anatomy", ("anatomy",), 128),
        ("anatomy+burden", ("anatomy", "burden"), 128),
        ("anatomy+position", ("anatomy", "position"), 128),
        ("anatomy+burden+position", ("anatomy", "burden", "position"), 128),
        ("anatomy+burden+position h1024", ("anatomy", "burden", "position"), 1024),
    ):
        t = time.time()
        P = fit(kind, hidden, view)
        m, le, sl = report(P, view)
        rows[(view, label)] = m
        print(
            f"{label:<34s} MSE {m:.5f}  explained {1 - m / base:6.3f}  lesion-cell MSE {le:.3f}  lesion slope {sl:6.3f}  ({time.time() - t:.0f}s)",
            flush=True,
        )
    a = rows[(view, "anatomy")]
    for lbl in ("anatomy+burden", "anatomy+burden+position", "anatomy+burden+position h1024"):
        print(
            f"  pay over anatomy, {lbl:<32s} {a - rows[(view, lbl)]:+.5f}  ({100 * (a - rows[(view, lbl)]) / a:.2f}% of the anatomy-only loss)"
        )
