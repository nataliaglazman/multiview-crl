"""Term gradients at a trained state: pilot gap_region16 backbone + a decoder trained on cached codes."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch

from models.vector_prediction import _prepared_targets
from training import main_conv_synthetic as T

here = os.path.dirname(os.path.abspath(__file__))
ckpt = sys.argv[1]
args = T.parse_args(open(os.path.join(here, "flags.txt")).read().split() + ["--device", "mps", "--model-id", "x"])
device = torch.device("mps")
torch.manual_seed(args.seed)
train = T.make_dataset(args, "train", args.num_train_samples)
val = T.make_dataset(args, "val", args.num_val_samples)
torch.manual_seed(args.model_seed)
model = T.build_encoder(args).to(device)
T.fit_vector_prediction_template(model, train, args, device)
raw = torch.load(ckpt, map_location="cpu", weights_only=False)
state = raw.get("model", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
missing, unexpected = model.load_state_dict(state, strict=False)
print(
    "missing outside decoder:",
    [k for k in missing if not k.startswith("vector_prediction")][:5],
    "unexpected:",
    list(unexpected)[:5],
    flush=True,
)
P = model.vector_prediction
grid = P.grid


def cached(data, n):
    Z, R, K = [], [], []
    model.eval()
    with torch.no_grad():
        for b in torch.utils.data.DataLoader(torch.utils.data.Subset(data, range(n)), batch_size=32):
            x = torch.cat(b["image"], 0).as_subclass(torch.Tensor).to(device)
            m = torch.cat(b["mask"], 0).as_subclass(torch.Tensor).to(device)
            Z.append(model(x, pool_only=True, n_views=2)[2][0][:, : model.content_channels].cpu())
            t, k, _ = _prepared_targets(x, m, grid, 0.5, 1e-6)
            nb = len(x) // 2
            r = t - P.template.repeat_interleave(nb, 0)
            R.append(r.cpu())
            K.append(k.cpu())
    Z, R, K = torch.cat(Z), torch.cat(R), torch.cat(K)
    return Z, R.float(), K.float()


def view_major(Z, R, K, bs=32):
    # batches were view-major per batch: reorder into (n, 2, ...)
    n = len(Z) // 2
    idx = []
    for s in range(0, n, bs):
        c = min(bs, n - s)
    return Z, R, K


Zs, Rs, Ks = [], [], []
Z, R, K = cached(train, 1200)
Zv, Rv, Kv = cached(val, 400)


def split(A, n, bs=32):
    parts = [A[i : i + 2 * bs] for i in range(0, len(A), 2 * bs)]
    return torch.stack([torch.cat([p[: len(p) // 2] for p in parts]), torch.cat([p[len(p) // 2 :] for p in parts])])


Z2, R2, K2 = split(Z, 1200), split(R, 1200), split(K, 1200)
Zv2, Rv2, Kv2 = split(Zv, 400), split(Rv, 400), split(Kv, 400)
print(
    "raw SD by view, median unit:",
    Z2.std(1).median(1).values.tolist(),
    "|mean| median",
    Z2.mean(1).abs().median(1).values.tolist(),
    flush=True,
)
mean, var = Z2.mean(1), Z2.var(1, unbiased=False)
P.input_mean.copy_(mean.to(device))
P.input_variance.copy_(var.to(device))
P.input_gauge_fitted.fill_(True)
Xt = (Z2 - mean[:, None]) / var[:, None].sqrt()
Xv = (Zv2 - mean[:, None]) / var[:, None].sqrt()
# decoders trained alone on the fixed codes, on the CPU, for the recipe's loss shape
dec = P.decoders.to("cpu")
opt = torch.optim.AdamW(dec.parameters(), lr=1e-3)
steps = 4000
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
torch.manual_seed(0)


def loss_of(d, X, Y, K):
    w = (1 + Y.abs()).clamp(max=4)
    c = K.sum((1, 2, 3, 4), True)
    w = w / ((w * K).sum((1, 2, 3, 4), True) / c) * K
    e = (d(X).reshape(Y.shape) - Y).square() * w
    return (e.sum((1, 2, 3, 4)) / c.flatten()).mean()


for s in range(steps):
    i = torch.randint(1200, (32,))
    # decoder v reads view v and predicts the OTHER view
    l = loss_of(dec[0], Xt[0, i], R2[1, i], K2[1, i]) + loss_of(dec[1], Xt[1, i], R2[0, i], K2[0, i])
    opt.zero_grad()
    l.backward()
    opt.step()
    sched.step()
with torch.no_grad():
    for name, a, b in ((("T1->FLAIR"), 0, 1), ("FLAIR->T1", 1, 0)):
        base = loss_of(lambda x: torch.zeros(len(x), np.prod(grid)), Xv[a], Rv2[b], Kv2[b])
        print(
            name,
            "held-out loss",
            float(loss_of(dec[a], Xv[a], Rv2[b], Kv2[b])),
            "template only",
            float(base),
            flush=True,
        )
P.decoders.to(device)
model.train()
sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
enc = [p for n, p in model.named_parameters() if not n.startswith("vector_prediction.")]
flat = (
    lambda: torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in enc]).detach().cpu()
)
rng = np.random.RandomState(0)
rows = {"global": [], "patch": [], "prediction": []}
store = {k: [] for k in rows}
P.gauge_momentum = 1e-9
for b in range(6):
    idx = rng.choice(len(train), args.batch_size, replace=False)
    items = [train[int(i)] for i in idx]
    x = (
        torch.cat([torch.stack([it["image"][v] for it in items]) for v in range(2)])
        .as_subclass(torch.Tensor)
        .to(device)
    )
    m = torch.cat([torch.stack([it["mask"][v] for it in items]) for v in range(2)]).as_subclass(torch.Tensor).to(device)
    _, _, terms = T.training_objective(model, x, args, sim, crit, m)
    for name, loss in (
        ("global", terms["global"]),
        ("patch", terms["patch_weighted"]),
        ("prediction", terms["vector_prediction_weighted"]),
    ):
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        g = flat()
        rows[name].append(float(g.pow(2).mean().sqrt()))
        store[name].append(g)
    model.zero_grad(set_to_none=True)
    print(
        b,
        {k: f"{v[-1]:.3g}" for k, v in rows.items()},
        "losses",
        round(float(terms["global"]), 3),
        round(float(terms["patch"]), 3),
        round(float(terms["vector_prediction"]), 3),
        flush=True,
    )
m = {k: float(np.mean(v)) for k, v in rows.items()}
print("RMS", m, "global/pred", m["global"] / m["prediction"], "patch/pred", m["patch"] / m["prediction"])
E = {k: torch.stack(v).mean(0) for k, v in store.items()}
print("RMS of E g", {k: float(v.pow(2).mean().sqrt()) for k, v in E.items()})
