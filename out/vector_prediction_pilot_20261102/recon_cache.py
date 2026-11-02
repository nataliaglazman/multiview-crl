"""Raw images at 32³ with each view's rendered style, for the full-image reconstruction oracle."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F

from eval.diagnostics.style_path_audit import effective_style
from training import main_conv_synthetic as T

S = sys.argv[1]
args = T.parse_args(open(os.path.join(S, "flags.txt")).read().split() + ["--device", "cpu", "--model-id", "x"])
torch.set_num_threads(3)
G = (32, 32, 32)
out = {}
for split, n in (("train", 2000), ("val", 400)):
    data = T.make_dataset(args, split, n)
    scale = data._inner.renderer.style_scale
    X, O, Z, ST = [], [], [], []
    for b in torch.utils.data.DataLoader(data, batch_size=32):
        x = torch.cat(b["image"], 0).as_subclass(torch.Tensor)
        m = torch.cat(b["mask"], 0).as_subclass(torch.Tensor).float()
        o = F.adaptive_avg_pool3d(m, G)
        c = F.adaptive_avg_pool3d(x * m, G) / o.clamp_min(1e-6)
        k = len(x) // 2
        X.append(c.reshape(2, k, -1).permute(1, 0, 2).numpy().astype("f4"))
        O.append(o.reshape(2, k, -1).permute(1, 0, 2).numpy().astype("f4"))
        Z.append(b["gt_latents"]["z_content"].numpy())
        ST.append(
            np.stack(
                [[effective_style(z, scale)[:3] for z in b["gt_latents"][f"z_style_v{v + 1}"]] for v in range(2)], 1
            )
        )
    out[split] = dict(
        x=np.concatenate(X), o=np.concatenate(O), z=np.concatenate(Z), style=np.concatenate(ST).astype("f4")
    )
    print(split, out[split]["x"].shape, out[split]["style"].shape, flush=True)
keep = out["train"]["o"][:300].mean((0, 1)) >= 0.5
np.savez(
    os.path.join(S, "recon_cache.npz"),
    keep=keep,
    **{f"{s}_{k}": (v[..., keep] if k in ("x", "o") else v) for s, d in out.items() for k, v in d.items()},
)
print("kept cells", keep.sum(), flush=True)
