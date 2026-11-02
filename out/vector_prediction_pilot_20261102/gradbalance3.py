"""The quantile term's encoder gradient on real data: at init, and on the pilot encoder with a fitted decoder."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch

from models.vector_prediction import _prepared_targets, deviation_quantiles
from training import main_conv_synthetic as T

S = os.path.dirname(os.path.abspath(__file__))
ckpt = sys.argv[1]
args = T.parse_args(
    open(os.path.join(S, "flags.txt")).read().split()
    + [
        "--device",
        "mps",
        "--model-id",
        "x",
        "--vector-prediction-target",
        "quantile",
        "--vector-prediction-components",
        "8",
    ]
)
device = torch.device("mps")
torch.manual_seed(args.seed)
train = T.make_dataset(args, "train", args.num_train_samples)
val = T.make_dataset(args, "val", args.num_val_samples)
torch.manual_seed(args.model_seed)
model = T.build_encoder(args).to(device)
meta = T.fit_vector_prediction_template(model, train, args, device)
print({k: meta[k] for k in ("target", "components", "interior_cells")}, flush=True)
P = model.vector_prediction
sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
enc = [p for n, p in model.named_parameters() if not n.startswith("vector_prediction.")]
flat = (
    lambda: torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in enc]).detach().cpu()
)


def gradients(label):
    model.train()
    P.gauge_momentum = 1e-9
    rng = np.random.RandomState(0)
    rows = {"global": [], "patch": [], "prediction": []}
    losses = []
    for b in range(5):
        idx = rng.choice(len(train), args.batch_size, replace=False)
        items = [train[int(i)] for i in idx]
        x = (
            torch.cat([torch.stack([it["image"][v] for it in items]) for v in range(2)])
            .as_subclass(torch.Tensor)
            .to(device)
        )
        m = (
            torch.cat([torch.stack([it["mask"][v] for it in items]) for v in range(2)])
            .as_subclass(torch.Tensor)
            .to(device)
        )
        _, _, terms = T.training_objective(model, x, args, sim, crit, m)
        losses.append(float(terms["vector_prediction"]))
        for name, loss in (
            ("global", terms["global"]),
            ("patch", terms["patch_weighted"]),
            ("prediction", terms["vector_prediction_weighted"]),
        ):
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            rows[name].append(float(flat().pow(2).mean().sqrt()))
        model.zero_grad(set_to_none=True)
    r = {k: float(np.mean(v)) for k, v in rows.items()}
    print(
        label,
        "RMS",
        {k: f"{v:.3g}" for k, v in r.items()},
        f"global/pred {r['global'] / r['prediction']:.1f}  patch/pred {r['patch'] / r['prediction']:.1f}  term {np.mean(losses):.3f}",
        flush=True,
    )


gradients("init")
raw = torch.load(ckpt, map_location="cpu", weights_only=False)
state = raw.get("model", raw) if isinstance(raw, dict) else raw
print(model.load_state_dict(state, strict=False).unexpected_keys[:3], flush=True)


def cached(data, n):
    Z, Q = [], []
    model.eval()
    with torch.no_grad():
        for b in torch.utils.data.DataLoader(torch.utils.data.Subset(data, range(n)), batch_size=32):
            x = torch.cat(b["image"], 0).as_subclass(torch.Tensor).to(device)
            m = torch.cat(b["mask"], 0).as_subclass(torch.Tensor).to(device)
            Z.append(model(x, pool_only=True, n_views=2)[2][0][:, : model.content_channels].cpu())
            t, k, _ = _prepared_targets(x, m, P.grid, 0.5, 1e-6)
            k_ = len(x) // 2
            q = deviation_quantiles(P, ((t - P.template.repeat_interleave(k_, 0)) * k).flatten(1))
            Q.append(((q - P.target_mean.repeat_interleave(k_, 0)) / P.target_sd.repeat_interleave(k_, 0)).cpu())
    return torch.cat(Z), torch.cat(Q)


def split(A, bs=32):
    parts = [A[i : i + 2 * bs] for i in range(0, len(A), 2 * bs)]
    return torch.stack([torch.cat([p[: len(p) // 2] for p in parts]), torch.cat([p[len(p) // 2 :] for p in parts])])


Z, Q = cached(train, 1200)
Zv, Qv = cached(val, 400)
Z2, Q2, Zv2, Qv2 = split(Z), split(Q), split(Zv), split(Qv)
mean, var = Z2.mean(1), Z2.var(1, unbiased=False)
P.input_mean.copy_(mean.to(device))
P.input_variance.copy_(var.to(device))
P.input_gauge_fitted.fill_(True)
Xt, Xv = (Z2 - mean[:, None]) / var[:, None].sqrt(), (Zv2 - mean[:, None]) / var[:, None].sqrt()
dec = P.decoders.to("cpu")
opt = torch.optim.AdamW(dec.parameters(), lr=1e-3)
steps = 4000
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
torch.manual_seed(0)
for s in range(steps):
    i = torch.randint(1200, (32,))
    l = (dec[0](Xt[0, i]) - Q2[1, i]).square().mean() + (dec[1](Xt[1, i]) - Q2[0, i]).square().mean()
    opt.zero_grad()
    l.backward()
    opt.step()
    sched.step()
with torch.no_grad():
    print(
        "held out MSE T1->FLAIR",
        float((dec[0](Xv[0]) - Qv2[1]).square().mean()),
        "FLAIR->T1",
        float((dec[1](Xv[1]) - Qv2[0]).square().mean()),
        "predict-the-mean",
        float(Qv2.square().mean()),
        flush=True,
    )
P.decoders.to(device)
gradients("trained, decoder fitted")
