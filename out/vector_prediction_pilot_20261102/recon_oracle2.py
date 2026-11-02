"""What full-image reconstruction with a style input pays for burden: oracle inputs, MLP and shared coordinate decoders."""
import os
import sys

import numpy as np
import torch

torch.set_num_threads(3)
S = sys.argv[1]
d = np.load(os.path.join(S, "recon_cache.npz"))
keep = d["keep"]
c16 = np.load(os.path.join(S, "targets_cache2.npz"))


def centroid(l):
    G = 16
    l = l.reshape(-1, G, G, G)
    ax = np.stack(np.meshgrid(*[np.arange(G)] * 3, indexing="ij"))
    w = l / np.maximum(l.sum((1, 2, 3), keepdims=True), 1e-9)
    return np.stack([(w * a).sum((1, 2, 3)) for a in ax], 1) / (G - 1) * 2 - 1


tr_c, va_c = centroid(c16["train_lesion"]), centroid(c16["val_lesion"])
tr_x, va_x, tr_o, va_o = d["train_x"], d["val_x"], d["train_o"], d["val_o"]
tr_z, va_z, tr_s, va_s = d["train_z"], d["val_z"], d["train_style"], d["val_style"]
anat = [i for i in range(tr_z.shape[1]) if i not in (2, 3, 4)]
grid = np.stack(np.meshgrid(*[np.arange(32)] * 3, indexing="ij"), -1).reshape(-1, 3)[keep] / 31 * 2 - 1
coords = torch.tensor(grid, dtype=torch.float32)
freq = 2.0 ** torch.arange(4) * np.pi
pe = torch.cat([coords] + [f(coords * w) for w in freq for f in (torch.sin, torch.cos)], 1)


def inputs(kind, view, z, s, c):
    st = s[:, view]
    st = (st - tr_s[:, view].mean(0)) / (tr_s[:, view].std(0) + 1e-9)
    parts = {"anatomy": z[:, anat], "style": st, "burden": z[:, [2]], "position": c}
    return torch.tensor(np.concatenate([parts[k] for k in kind], 1), dtype=torch.float32)


class Coord(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n + pe.shape[1], 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x, cells):
        return self.net(torch.cat([x[:, None].expand(-1, len(cells), -1), pe[cells][None].expand(len(x), -1, -1)], -1))[
            ..., 0
        ]


def fit(kind, view, shared, steps):
    torch.manual_seed(0)
    X, Xv = inputs(kind, view, tr_z, tr_s, tr_c), inputs(kind, view, va_z, va_s, va_c)
    Y, W = torch.tensor(tr_x[:, view]), torch.tensor((tr_o[:, view] >= 0.5).astype("f4"))
    Yv, Wv = torch.tensor(va_x[:, view]), torch.tensor((va_o[:, view] >= 0.5).astype("f4"))
    m = Y.mean(0)
    Y, Yv = Y - m, Yv - m
    n = Y.shape[1]
    all_cells = torch.arange(n)
    net = (
        Coord(X.shape[1])
        if shared
        else torch.nn.Sequential(torch.nn.Linear(X.shape[1], 512), torch.nn.GELU(), torch.nn.Linear(512, n))
    )
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        i = torch.randint(len(X), (8,))
        if shared:
            cells = torch.randint(n, (768,))
            e = (net(X[i], cells) - Y[i][:, cells]) ** 2 * W[i][:, cells]
        else:
            e = (net(X[i]) - Y[i]) ** 2 * W[i]
        loss = e.sum() / W[i].sum() if not shared else e.sum() / W[i][:, cells].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()

    def score(X_, Y_, W_):
        with torch.no_grad():
            total = sum(
                ((net(X_[a : a + 32], all_cells) if shared else net(X_[a : a + 32])) - Y_[a : a + 32])
                .pow(2)
                .mul(W_[a : a + 32])
                .sum()
                for a in range(0, len(X_), 32)
            )
        return float(total / W_.sum())

    return score(X[:400], Y[:400], W[:400]), score(Xv, Yv, Wv)


for shared, label, steps in ((True, "shared coordinate decoder", 20000),):
    for view, name in ((1, "FLAIR"), (0, "T1")):
        base = None
        for kind in (("anatomy", "style"), ("anatomy", "style", "burden"), ("anatomy", "style", "burden", "position")):
            a, b = fit(kind, view, shared, steps)
            base = base or (a, b)
            print(
                f"{label:<27s} {name:<6s} {'+'.join(kind):<36s} train {a:.5f} (pay {100 * (base[0] - a) / base[0]:+6.2f}%)  val {b:.5f} (pay {100 * (base[1] - b) / base[1]:+6.2f}%)",
                flush=True,
            )
