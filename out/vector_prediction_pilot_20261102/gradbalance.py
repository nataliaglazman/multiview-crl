"""Per-term encoder gradients on the real vector-prediction recipe, at a given state."""
import json
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch

from training import main_conv_synthetic as T

here = os.path.dirname(os.path.abspath(__file__))
ckpt = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "init" else None
batches = int(sys.argv[2]) if len(sys.argv) > 2 else 6
args = T.parse_args(open(os.path.join(here, "flags.txt")).read().split() + ["--device", "mps", "--model-id", "x"])
device = torch.device("mps")
torch.manual_seed(args.seed)
train = T.make_dataset(args, "train", args.num_train_samples)
torch.manual_seed(args.model_seed)
model = T.build_encoder(args).to(device)
T.fit_vector_prediction_template(model, train, args, device)
if ckpt:
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.train()
sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
enc = [(n, p) for n, p in model.named_parameters() if not n.startswith("vector_prediction.")]
dec = [(n, p) for n, p in model.named_parameters() if n.startswith("vector_prediction.")]


def flat(params):
    return (
        torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for _, p in params])
        .detach()
        .cpu()
    )


rng = np.random.RandomState(0)
rows = []
grads = {"global": [], "patch": [], "prediction": []}
for b in range(batches):
    idx = rng.choice(len(train), args.batch_size, replace=False)
    items = [train[int(i)] for i in idx]
    x = (
        torch.cat([torch.stack([it["image"][v] for it in items]) for v in range(2)])
        .as_subclass(torch.Tensor)
        .to(device)
    )
    m = torch.cat([torch.stack([it["mask"][v] for it in items]) for v in range(2)]).as_subclass(torch.Tensor).to(device)
    _, _, terms = T.training_objective(model, x, args, sim, crit, m)
    row = {k: float(terms[k]) for k in ("global", "patch", "vector_prediction", "vector_prediction_unweighted")}
    parts = {
        "global": terms["global"],
        "patch": terms["patch_weighted"],
        "prediction": terms["vector_prediction_weighted"],
    }
    gs = {}
    for name, loss in parts.items():
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        gs[name] = (flat(enc), flat(dec))
        grads[name].append(gs[name][0])
    model.zero_grad(set_to_none=True)
    total_enc = sum(g[0] for g in gs.values())
    total_dec = gs["prediction"][1]
    e, d = float(total_enc.norm()), float(total_dec.norm())
    row.update(
        enc_norm=e,
        dec_norm=d,
        joint_norm=float(np.hypot(e, d)),
        clip_alone=min(1, args.grad_clip / e),
        clip_joint=min(1, args.grad_clip / float(np.hypot(e, d))),
    )
    for k, g in gs.items():
        row[f"rms_{k}"] = float(g[0].pow(2).mean().sqrt())
    rows.append(row)
    print(b, {k: round(v, 4) for k, v in row.items()}, flush=True)
mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
print("\nmean", json.dumps({k: round(v, 5) for k, v in mean.items()}, indent=0))
# E g over batches: cosines between mean term gradients, and per-layer share of the prediction term.
E = {k: torch.stack(v).mean(0) for k, v in grads.items()}
print("cos(E g_global, E g_pred)", float(torch.nn.functional.cosine_similarity(E["global"], E["prediction"], dim=0)))
print("cos(E g_patch , E g_pred)", float(torch.nn.functional.cosine_similarity(E["patch"], E["prediction"], dim=0)))
off, names = 0, {}
for n, p in enc:
    size = p.numel()
    key = n.split(".")[0]
    for term in E:
        names.setdefault(key, {}).setdefault(term, 0.0)
        names[key][term] += float(E[term][off : off + size].pow(2).sum())
    off += size
for key, v in names.items():
    print(f"{key:<24s}" + "  ".join(f"{t} {np.sqrt(x):.4g}" for t, x in v.items()))
