"""Cache the exact prediction targets + latents + rendered lesion for the recipe's train and val subjects."""
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from eval.encoder.regional_lesion_detection import lesion_fractions
from models.vector_prediction import _prepared_targets
from training import main_conv_synthetic as T

here = os.path.dirname(os.path.abspath(__file__))
args = T.parse_args(open(os.path.join(here, "flags.txt")).read().split() + ["--device", "cpu", "--model-id", "x"])
torch.set_num_threads(4)
grid = (16, 16, 16)
out = {}
for split, n in (("train", 2000), ("val", 400)):
    data = T.make_dataset(args, split, n)
    loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False, num_workers=0)
    tgt, kp, occ, z = [], [], [], []
    for b in loader:
        x = torch.cat(b["image"], 0).as_subclass(torch.Tensor)
        m = torch.cat(b["mask"], 0).as_subclass(torch.Tensor)
        t, k, o = _prepared_targets(x, m, grid, 0.5, 1e-6)
        n_b = len(x) // 2
        tgt.append(t.reshape(2, n_b, *grid).permute(1, 0, 2, 3, 4).numpy())
        kp.append(k.reshape(2, n_b, *grid).permute(1, 0, 2, 3, 4).numpy())
        z.append(b["gt_latents"]["z_content"].numpy())
    frac = lesion_fractions(data, grid)  # (n, 4096) lesion load per cell, x-major
    out[split] = dict(
        target=np.concatenate(tgt).astype("f4"), keep=np.concatenate(kp), z=np.concatenate(z), lesion=frac.astype("f4")
    )
    print(split, out[split]["target"].shape, flush=True)
np.savez(os.path.join(here, "targets_cache.npz"), **{f"{s}_{k}": v for s, d in out.items() for k, v in d.items()})
