"""Does a position-free summary of the other view's local deviations carry burden, and does the loss then pay for it?"""
import os

import numpy as np
import torch

torch.set_num_threads(3)
from scipy.stats import spearmanr

here = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(here, "targets_cache.npz"))
G = 16
tr_t, tr_k, tr_z, tr_l = d["train_target"], d["train_keep"], d["train_z"], d["train_lesion"].reshape(-1, G, G, G)
va_t, va_k, va_z, va_l = d["val_target"], d["val_keep"], d["val_z"], d["val_lesion"].reshape(-1, G, G, G)
tmpl = (tr_t * tr_k).sum(0) / np.maximum(tr_k.sum(0), 1)
tr_r, va_r = (tr_t - tmpl) * tr_k, (va_t - tmpl) * va_k
anat = [i for i in range(tr_z.shape[1]) if i not in (2, 3, 4)]
vol_t, vol_v = tr_l.sum((1, 2, 3)) * 64, va_l.sum((1, 2, 3)) * 64
QS = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999])


def quantiles(r, k):
    return np.stack([np.quantile(r[i][k[i].astype(bool)], QS) for i in range(len(r))])


def pca_residual(tr, va, K):
    mu = tr.reshape(len(tr), -1).mean(0)
    A = tr.reshape(len(tr), -1) - mu
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt[:K]
    f = lambda x: (x.reshape(len(x), -1) - mu - (x.reshape(len(x), -1) - mu) @ V.T @ V).reshape(x.shape)
    return f(tr), f(va)


def mlp(Xt, Yt, Xv, Yv, steps=6000):
    torch.manual_seed(0)
    mx, sx = Xt.mean(0), Xt.std(0) + 1e-9
    my, sy = Yt.mean(0), Yt.std(0) + 1e-9
    X, Y = torch.tensor((Xt - mx) / sx, dtype=torch.float32), torch.tensor((Yt - my) / sy, dtype=torch.float32)
    Xv_ = torch.tensor((Xv - mx) / sx, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.GELU(), torch.nn.Linear(128, Y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        i = torch.randint(len(X), (64,))
        loss = (net(X[i]) - Y[i]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    with torch.no_grad():
        return float((net(X) - Y).square().mean()), float((net(Xv_).numpy() - (Yv - my) / sy).__pow__(2).mean())


def ridge_r2(Xt, yt, Xv, yv):
    from sklearn.linear_model import RidgeCV

    mx, sx = Xt.mean(0), Xt.std(0) + 1e-9
    m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit((Xt - mx) / sx, yt)
    p = m.predict((Xv - mx) / sx)
    return 1 - ((yv - p) ** 2).sum() / ((yv - yv.mean()) ** 2).sum(), p


for view, name in ((0, "T1"), (1, "FLAIR")):
    print(f"\n=== {name} target")
    yt, yv, kt, kv = tr_r[:, view], va_r[:, view], tr_k[:, view], va_k[:, view]
    for label, (rt, rv) in (
        ("residual from template", (yt, yv)),
        ("after 16 population PCs", pca_residual(yt * kt, yv * kv, 16)),
    ):
        Qt, Qv = quantiles(rt, kt), quantiles(rv, kv)
        r2, p = ridge_r2(Qt, tr_z[:, 2], Qv, va_z[:, 2])
        small = vol_v < np.percentile(vol_v, 33.3)
        rho = spearmanr(p[small], vol_v[small]).statistic
        at, av = mlp(tr_z[:, anat], Qt, va_z[:, anat], Qv)
        bt, bv = mlp(tr_z[:, anat + [2]], Qt, va_z[:, anat + [2]], Qv)
        print(
            f"{label:<26s} burden R2 from the 17 quantiles {r2:6.3f}  smallest-third spearman {rho:5.2f} | "
            f"MSE anatomy {at:.4f}/{av:.4f} -> +burden {bt:.4f}/{bv:.4f} (train/val) pay {100 * (at - bt) / at:5.1f}% / {100 * (av - bv) / av:5.1f}%",
            flush=True,
        )
