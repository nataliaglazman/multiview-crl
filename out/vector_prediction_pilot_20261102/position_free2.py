"""The exact third-arm target, oracle pay: interior cells, training-only PCs, per view, z-scored quantiles."""
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

torch.set_num_threads(4)
S = sys.argv[1]
d = np.load(os.path.join(S, "targets_cache2.npz"))
G = 16
tr_t, tr_k, tr_o, tr_z, tr_l = d["train_target"], d["train_keep"], d["train_occ"], d["train_z"], d["train_lesion"]
va_t, va_k, va_z, va_l = d["val_target"], d["val_keep"], d["val_z"], d["val_lesion"]
anat = [i for i in range(tr_z.shape[1]) if i not in (2, 3, 4)]
vol_v = va_l.sum(1) * 64
QS = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999])


def design(view, subjects, threshold, K):
    t, k, o = tr_t[:subjects, view], tr_k[:subjects, view], tr_o[:subjects, view]
    w = o * k
    template = (t * w).sum(0) / np.maximum(w.sum(0), 1e-12)
    cells = (w.mean(0) >= threshold).ravel()

    def residual(x, keep):
        return ((x - template) * keep).reshape(len(x), -1)[:, cells]

    fit = residual(t, k)
    V = np.linalg.svd(fit, full_matrices=False)[2][:K] if K else np.zeros((0, fit.shape[1]))

    def target(x, keep):
        r = residual(x, keep)
        r = r - r @ V.T @ V
        return np.quantile(r, QS, axis=1).T

    return target, int(cells.sum()), target(t, k)


def mlp(Xt, Yt, Xv, Yv, steps=6000):
    torch.manual_seed(0)
    mx, sx = Xt.mean(0), Xt.std(0) + 1e-9
    X, Y = torch.tensor((Xt - mx) / sx, dtype=torch.float32), torch.tensor(Yt, dtype=torch.float32)
    V = torch.tensor((Xv - mx) / sx, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.GELU(), torch.nn.Linear(128, Y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.randint(len(X), (64,))
        loss = (net(X[i]) - Y[i]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    with torch.no_grad():
        return float((net(X) - Y).square().mean()), float(
            (net(V) - torch.tensor(Yv, dtype=torch.float32)).square().mean()
        )


def ridge(Xt, yt, Xv):
    mx, sx = Xt.mean(0), Xt.std(0) + 1e-9
    return RidgeCV(alphas=np.logspace(-2, 4, 13)).fit((Xt - mx) / sx, yt).predict((Xv - mx) / sx)


small = vol_v < np.percentile(vol_v, 33.3)
for view, name in ((0, "T1"), (1, "FLAIR")):
    print(f"=== {name} target", flush=True)
    for label, subjects, threshold, K in (
        ("interior .9, 16 PCs from 300", 300, 0.9, 16),
        ("interior .9, 16 PCs from 2000", 2000, 0.9, 16),
        ("interior .9, no PCs", 300, 0.9, 0),
        ("interior .9, 8 PCs from 300", 300, 0.9, 8),
        ("half brain .5, 16 PCs from 300", 300, 0.5, 16),
    ):
        target, cells, fitted = design(view, subjects, threshold, K)
        Qt, Qv = target(tr_t[:, view], tr_k[:, view]), target(va_t[:, view], va_k[:, view])
        m, s = fitted.mean(0), fitted.std(0) + 1e-9
        Qt, Qv = (Qt - m) / s, (Qv - m) / s
        p = ridge(Qt, tr_z[:, 2], Qv)
        r2 = 1 - ((va_z[:, 2] - p) ** 2).sum() / ((va_z[:, 2] - va_z[:, 2].mean()) ** 2).sum()
        at, av = mlp(tr_z[:, anat], Qt, va_z[:, anat], Qv)
        bt, bv = mlp(tr_z[:, anat + [2]], Qt, va_z[:, anat + [2]], Qv)
        print(
            f"{label:<31s} {cells:4d} cells | burden R2 {r2:.3f} rho {spearmanr(p, va_z[:, 2]).statistic:.3f} small {spearmanr(p[small], va_z[small, 2]).statistic:.3f} | "
            f"MSE {at:.3f}/{av:.3f} -> {bt:.3f}/{bv:.3f} pay {100 * (at - bt) / at:5.1f}% / {100 * (av - bv) / av:5.1f}%",
            flush=True,
        )
