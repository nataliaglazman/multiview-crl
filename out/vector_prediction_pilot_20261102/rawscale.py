"""How much do raw content units vary between subjects, relative to their mean, and what do term gradients become
if the decoder reads the standardized code?"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch

from models.vector_prediction import vector_prediction_loss
from training import main_conv_synthetic as T

here = os.path.dirname(os.path.abspath(__file__))
args = T.parse_args(open(os.path.join(here, "flags.txt")).read().split() + ["--device", "mps", "--model-id", "x"])
device = torch.device("mps")
torch.manual_seed(args.seed)
train = T.make_dataset(args, "train", args.num_train_samples)
torch.manual_seed(args.model_seed)
model = T.build_encoder(args).to(device)
T.fit_vector_prediction_template(model, train, args, device)
model.train()
sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
enc = [p for n, p in model.named_parameters() if not n.startswith("vector_prediction.")]


def flat():
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in enc]).detach().cpu()


rng = np.random.RandomState(0)
out = {"global": [], "raw": [], "standardized": []}
for b in range(4):
    idx = rng.choice(len(train), args.batch_size, replace=False)
    items = [train[int(i)] for i in idx]
    x = (
        torch.cat([torch.stack([it["image"][v] for it in items]) for v in range(2)])
        .as_subclass(torch.Tensor)
        .to(device)
    )
    m = torch.cat([torch.stack([it["mask"][v] for it in items]) for v in range(2)]).as_subclass(torch.Tensor).to(device)
    pooled, _, terms = T.training_objective(model, x, args, sim, crit, m)
    c = pooled[:, : model.content_channels]
    n = len(c) // 2
    sd = c[:n].std(0)
    if b == 0:
        print("raw content: per-unit between-subject SD", np.round(sd.detach().cpu().numpy(), 5))
        print("raw content: per-unit |mean|           ", np.round(c[:n].mean(0).abs().detach().cpu().numpy(), 4))
        print("SD/|mean| median", float((sd / c[:n].mean(0).abs()).median()))
    z = torch.cat([(v - v.mean(0)) / v.std(0).clamp_min(1e-6) for v in c.chunk(2)])
    std_loss, _ = vector_prediction_loss(model.vector_prediction, z, x, m, 1.0, 4.0)
    for name, loss in (("global", terms["global"]), ("raw", terms["vector_prediction"]), ("standardized", std_loss)):
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        out[name].append(float(flat().pow(2).mean().sqrt()))
    print(
        b,
        {k: f"{v[-1]:.3g}" for k, v in out.items()},
        "raw loss",
        round(float(terms["vector_prediction"]), 3),
        "std loss",
        round(float(std_loss), 3),
        flush=True,
    )
print({k: float(np.mean(v)) for k, v in out.items()})
print(
    "standardized / raw gradient:",
    np.mean(out["standardized"]) / np.mean(out["raw"]),
    " global / standardized:",
    np.mean(out["global"]) / np.mean(out["standardized"]),
)
