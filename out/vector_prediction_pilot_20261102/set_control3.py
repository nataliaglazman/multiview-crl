"""Does the choice of cells decide whether a position-free probe reads burden? Raw image and trained map."""
import importlib.util
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.set_num_threads(4)
from eval.encoder.burden_recovery import encode
from eval.encoder.regional_lesion_detection import lesion_fractions
from eval.protocol.score_checkpoint import build_model, load_settings, make_dataset

S = sys.argv[1]
spec = importlib.util.spec_from_file_location("sc", os.path.join(S, "set_control_fns.py"))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
d = os.path.join(S, "pilot_eval", "predcode")
cfg = load_settings(d)
grid = [16, 16, 16]
data = make_dataset(cfg, 400, mode="val")
factors, occ, raw = [], [], {"t1": [], "flair": []}
for b in DataLoader(data, batch_size=16):
    factors.append(b["gt_latents"]["z_content"].numpy())
    x = torch.cat(b["image"], 0).as_subclass(torch.Tensor).float()
    m = torch.cat(b["mask"], 0).as_subclass(torch.Tensor).float()
    o = F.adaptive_avg_pool3d(m, grid)
    c = F.adaptive_avg_pool3d(x * m, grid) / o.clamp_min(1e-6)
    keep = (o >= 0.5).float()
    n = keep.sum((1, 2, 3, 4), True)
    mean = (c * keep).sum((1, 2, 3, 4), True) / n
    sd = (((c - mean) ** 2 * keep).sum((1, 2, 3, 4), True) / n).sqrt()
    z = ((c - mean) / sd * keep).flatten(1).numpy()
    h = len(z) // 2
    raw["t1"].append(z[:h])
    raw["flair"].append(z[h:])
    occ.append(o.flatten(1).numpy())
y = np.concatenate(factors)[:, 2].astype(float)
occ = np.concatenate(occ)
vol = lesion_fractions(data, (1, 1, 1))[:, 0] * 64**3
small = vol < np.percentile(vol, 33.3)
sets = {"any>=.05": (occ >= 0.05).any(0), "mean>=.5": occ.mean(0) >= 0.5, "mean>=.9": occ.mean(0) >= 0.9}
print({k: int(v.sum()) for k, v in sets.items()}, flush=True)
for view in ("t1", "flair"):
    cells = np.concatenate(raw[view])
    for name, keep in sets.items():
        print(f"RAW {view} cells {name} ({keep.sum()})", flush=True)
        sc.report("channel quantiles", sc.fixed_probe(cells[:, keep][:, None], y, sc.channel_quantiles), y, small)
for stage, ckpt in (("trained", "model.pt"), ("floor", "model_init.pt")):
    model = build_model(cfg, "cpu", torch.load(os.path.join(d, ckpt), map_location="cpu")).eval()
    rep = encode(model, grid, data, "cpu", 16)
    del model
    for view in ("t1", "flair"):
        for name in ("mean>=.5", "mean>=.9"):
            cells = rep[view]["map"][:, :, sets[name]]
            print(f"{stage} {view} map {name}", flush=True)
            sc.report("channel quantiles", sc.fixed_probe(cells, y, sc.channel_quantiles), y, small)
            sc.report("whitened quantiles", sc.fixed_probe(cells, y, sc.white_quantiles), y, small)
            sc.report("cell distance quantiles", sc.fixed_probe(cells, y, sc.distance_quantiles), y, small)
