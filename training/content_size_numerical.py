"""Encoding-size sweep on the numerical simulation of Daunhawer et al. (ICLR 2023, arXiv 2303.09166).

The paper selects the number of content units by training one model per encoding size k and
looking for the elbow of the held-out SymInfoNCE loss as a function of k (App. B.2, Fig. 7a; the
elbow sits at the true n_c = 5). Per-group kernel-ridge R² of content c, style s and
modality-specific m1 from the view-1 code (Sec. 5.1, Fig. 3/10) says what each extra unit holds.

Generative process (Sec. 5.1): c ~ N(0, Σc), m_i ~ N(0, I), s ~ N(a + Bc, I),
s̃_j = s_j + ε_j with probability p_change per dimension, x1 = f1(c, s, m1), x2 = f2(c, s̃, m2),
f1 ≠ f2 invertible 3-layer leaky-ReLU MLPs. Two 7-layer MLP encoders, SymInfoNCE with
sim(a, b) = -||a - b||², τ = 1 (Table 2a).

    python -m training.content_size_numerical --k 5 --seed 0                # one run
    python -m training.content_size_numerical --summarize results/content_size_numerical/independent
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import models.encoders as encoders
import utils.invertible_network_utils as invertible_network_utils


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5, help="encoding size (output units of each encoder)")
    p.add_argument("--n-c", type=int, default=5)
    p.add_argument("--n-s", type=int, default=5)
    p.add_argument("--n-m", type=int, default=5)
    p.add_argument("--regime", choices=["independent", "stat", "causal", "stat_causal"], default="independent")
    p.add_argument("--p-change", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0, help="encoder init, batches and (unless --data-seed) mixing")
    p.add_argument("--data-seed", type=int, default=None, help="mixing functions and latent SCM; default = seed")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--val-size", type=int, default=10000)
    p.add_argument("--val-batch", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--probe-train", type=int, default=5000)
    p.add_argument("--probe-test", type=int, default=2000)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out-dir", default="results/content_size_numerical")
    p.add_argument("--summarize", default=None, help="regime directory to summarise instead of training")
    p.add_argument("--elbow-frac", type=float, default=0.05)
    return p.parse_args(argv)


class LatentSCM:
    """Latents (c, s, m1, m2) and the style perturbation, fixed by data_seed."""

    def __init__(self, args):
        rng = np.random.default_rng(args.data_seed + 7919)
        self.n_c, self.n_s, self.n_m = args.n_c, args.n_s, args.n_m
        self.p_change = args.p_change
        stat = args.regime in ("stat", "stat_causal")
        causal = args.regime in ("causal", "stat_causal")
        if stat:
            A = rng.standard_normal((self.n_c, self.n_c))
            cov = A @ A.T + 0.1 * np.eye(self.n_c)
            d = np.sqrt(np.diag(cov))
            cov = cov / np.outer(d, d)
        else:
            cov = np.eye(self.n_c)
        self.chol_c = torch.tensor(np.linalg.cholesky(cov), dtype=torch.float32)
        self.a = torch.tensor(rng.standard_normal(self.n_s) if causal else np.zeros(self.n_s), dtype=torch.float32)
        self.B = torch.tensor(
            rng.standard_normal((self.n_s, self.n_c)) if causal else np.zeros((self.n_s, self.n_c)),
            dtype=torch.float32,
        )

    def sample(self, n, gen):
        c = torch.randn(n, self.n_c, generator=gen) @ self.chol_c.T
        s = self.a + c @ self.B.T + torch.randn(n, self.n_s, generator=gen)
        change = torch.rand(n, self.n_s, generator=gen) < self.p_change
        s_tilde = s + change * torch.randn(n, self.n_s, generator=gen)
        m1 = torch.randn(n, self.n_m, generator=gen)
        m2 = torch.randn(n, self.n_m, generator=gen)
        return {"c": c, "s": s, "s_tilde": s_tilde, "m1": m1, "m2": m2}


def build_mixing(n, seed):
    np.random.seed(seed)
    return invertible_network_utils.construct_invertible_mlp(
        n=n, n_layers=3, n_iter_cond_thresh=25000, cond_thresh_ratio=1e-3, act_fct="leaky_relu"
    )


def build_encoder(d, k):
    return encoders.get_mlp(n_in=d, n_out=k, layers=[d * 10, d * 50, d * 50, d * 50, d * 50, d * 10])


def sym_infonce(h1, h2, tau):
    """SimCLR-style symmetric InfoNCE (within- and cross-view negatives), sim = -||a - b||²."""
    n = h1.shape[0]
    s12 = -torch.cdist(h1, h2).pow(2) / tau
    s11 = -torch.cdist(h1, h1).pow(2) / tau
    s22 = -torch.cdist(h2, h2).pow(2) / tau
    eye = torch.eye(n, dtype=torch.bool, device=h1.device)
    s11 = s11.masked_fill(eye, float("-inf"))
    s22 = s22.masked_fill(eye, float("-inf"))
    scores = torch.cat([torch.cat([s12, s11], 1), torch.cat([s22, s12.T], 1)], 0)
    # Row i < n: positive s12[i, i] at column i. Row n + i: positive s12.T[i, i] at column n + i.
    return F.cross_entropy(scores, torch.arange(2 * n, device=h1.device))


def views(lat, f1, f2, device):
    x1 = f1(torch.cat([lat["c"], lat["s"], lat["m1"]], 1).to(device))
    x2 = f2(torch.cat([lat["c"], lat["s_tilde"], lat["m2"]], 1).to(device))
    return x1, x2


@torch.no_grad()
def encode(g, x, chunk=8192):
    return torch.cat([g(x[i : i + chunk]) for i in range(0, len(x), chunk)])


@torch.no_grad()
def val_loss(g1, g2, x1, x2, args):
    g1.eval(), g2.eval()
    h1, h2 = encode(g1, x1), encode(g2, x2)
    losses = [
        sym_infonce(h1[i : i + args.val_batch], h2[i : i + args.val_batch], args.tau).item()
        for i in range(0, len(h1) - args.val_batch + 1, args.val_batch)
    ]
    g1.train(), g2.train()
    return float(np.mean(losses)), h1


def kernel_r2(h, lat, args):
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler

    ntr, nte = args.probe_train, args.probe_test
    X = StandardScaler().fit(h[:ntr]).transform(h[: ntr + nte])
    out = {}
    for name in ("c", "s", "m1"):
        Y = lat[name].numpy()[: ntr + nte]
        Y = (Y - Y[:ntr].mean(0)) / Y[:ntr].std(0)
        gs = GridSearchCV(
            KernelRidge(kernel="rbf"),
            {"alpha": [1.0, 0.1, 0.01, 0.001], "gamma": [0.01, 0.1, 1.0]},
            cv=3,
            n_jobs=-1,
        ).fit(X[:ntr], Y[:ntr])
        pred = gs.predict(X[ntr:])
        out[name] = float(r2_score(Y[ntr:], pred))
        out[f"{name}_per_dim"] = [float(r2_score(Y[ntr:, j], pred[:, j])) for j in range(Y.shape[1])]
    return out


def train(args):
    args.data_seed = args.seed if args.data_seed is None else args.data_seed
    run_dir = Path(args.out_dir) / args.regime / f"k{args.k}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    d = args.n_c + args.n_s + args.n_m

    scm = LatentSCM(args)
    f1 = build_mixing(d, args.data_seed * 2 + 1).to(device)
    f2 = build_mixing(d, args.data_seed * 2 + 2).to(device)
    torch.manual_seed(args.seed)
    g1, g2 = build_encoder(d, args.k).to(device), build_encoder(d, args.k).to(device)
    params = list(g1.parameters()) + list(g2.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    val_lat = scm.sample(args.val_size, torch.Generator().manual_seed(10**6 + args.data_seed))
    vx1, vx2 = views(val_lat, f1, f2, device)
    gen = torch.Generator().manual_seed(args.seed)

    history, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        x1, x2 = views(scm.sample(args.batch_size, gen), f1, f2, device)
        loss = sym_infonce(g1(x1), g2(x2), args.tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            vl, _ = val_loss(g1, g2, vx1, vx2, args)
            history.append({"step": step, "train_loss": loss.item(), "val_loss": vl})
            print(
                f"k={args.k} s={args.seed} step {step} train {loss.item():.4f} val {vl:.4f} ({time.time()-t0:.0f}s)",
                flush=True,
            )

    vl, h1 = val_loss(g1, g2, vx1, vx2, args)
    r2 = kernel_r2(h1.cpu().numpy(), val_lat, args)
    metrics = {"args": vars(args), "val_loss": vl, "r2": r2, "history": history, "seconds": time.time() - t0}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"k={args.k} s={args.seed} val {vl:.4f} R² c {r2['c']:.3f} s {r2['s']:.3f} m1 {r2['m1']:.3f}", flush=True)
    return metrics


def elbow(ks, losses, frac):
    """Smallest k whose loss is within frac of the total drop above the minimum."""
    lo, hi = min(losses), losses[0]
    return next(k for k, v in zip(ks, losses) if v <= lo + frac * (hi - lo))


def summarize(regime_dir, frac):
    rows = {}
    for f in sorted(Path(regime_dir).glob("k*_s*/metrics.json")):
        m = json.loads(f.read_text())
        rows.setdefault(m["args"]["k"], []).append(m)
    ks = sorted(rows)
    table = []
    for k in ks:
        ms = rows[k]
        table.append(
            {
                "k": k,
                "n_seeds": len(ms),
                "val_loss": float(np.mean([m["val_loss"] for m in ms])),
                "val_loss_sd": float(np.std([m["val_loss"] for m in ms])),
                **{f"r2_{g}": float(np.mean([m["r2"][g] for m in ms])) for g in ("c", "s", "m1")},
            }
        )
    k_star = elbow(ks, [r["val_loss"] for r in table], frac)
    seeds = sorted({m["args"]["seed"] for ms in rows.values() for m in ms})
    per_seed = {}
    for s in seeds:
        pts = [(k, m["val_loss"]) for k in ks for m in rows[k] if m["args"]["seed"] == s]
        if len(pts) == len(ks):
            per_seed[s] = elbow([p[0] for p in pts], [p[1] for p in pts], frac)
    n_c = next(iter(rows.values()))[0]["args"]["n_c"]
    print(f"{regime_dir}: true n_c = {n_c}")
    print(f"{'k':>3} {'seeds':>5} {'val loss':>16} {'R² c':>6} {'R² s':>6} {'R² m1':>6}")
    for r in table:
        print(
            f"{r['k']:>3} {r['n_seeds']:>5} {r['val_loss']:>9.3f} ± {r['val_loss_sd']:.3f}"
            f" {r['r2_c']:>6.3f} {r['r2_s']:>6.3f} {r['r2_m1']:>6.3f}"
        )
    print(f"elbow (mean curve, within {frac:.0%} of the drop): k* = {k_star}; per seed: {per_seed}")
    out = {"n_c": n_c, "k_star": k_star, "k_star_per_seed": per_seed, "table": table, "elbow_frac": frac}
    Path(regime_dir, "summary.json").write_text(json.dumps(out, indent=1))
    plot(regime_dir, table, n_c, k_star)
    return out


def plot(regime_dir, table, n_c, k_star):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r["k"] for r in table]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].errorbar(ks, [r["val_loss"] for r in table], yerr=[r["val_loss_sd"] for r in table], marker="o")
    ax[0].axvline(n_c, color="grey", ls="--", lw=1, label=f"true n_c = {n_c}")
    ax[0].axvline(k_star, color="C3", ls=":", lw=1.5, label=f"elbow k* = {k_star}")
    ax[0].set(xlabel="encoding size k", ylabel="held-out SymInfoNCE", title="Validation loss")
    ax[0].legend(frameon=False)
    for g, lab in (("c", "content c"), ("s", "style s"), ("m1", "modality m1")):
        ax[1].plot(ks, [r[f"r2_{g}"] for r in table], marker="o", label=lab)
    ax[1].axvline(n_c, color="grey", ls="--", lw=1)
    ax[1].set(xlabel="encoding size k", ylabel="kernel-ridge R²", title="What the code holds", ylim=(-0.05, 1.05))
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path(regime_dir, "elbow.png"), dpi=150)
    import csv

    with open(Path(regime_dir, "elbow.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


if __name__ == "__main__":
    a = parse_args()
    if a.summarize:
        summarize(a.summarize, a.elbow_frac)
    else:
        train(a)
