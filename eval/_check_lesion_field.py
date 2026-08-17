#!/usr/bin/env python
"""Verify lesion_mode='sphere' is bit-identical to the legacy path, and smoke-test 'field'.

python -m eval._check_lesion_field
"""

import numpy as np
import torch

from data.datasets import SyntheticBrainDataset

BASE = dict(
    mode="test",
    spatial_size=(64, 64, 64),
    cache=False,
    synthetic_mode="pseudo_mri",
    synthetic_seed=42,
    synthetic_num_samples=4,
    synthetic_n_content=9,
    synthetic_n_style=3,
)


def render(**kw):
    ds = SyntheticBrainDataset(**BASE, **kw)
    out = []
    for i in range(4):
        d = ds[i]
        out.append((d["image"][0].numpy(), d["image"][1].numpy(), d["gt_latents"]))
    return out


print("sphere (default) vs sphere (explicit) — must be bit-identical")
a, b = render(), render(synthetic_lesion_mode="sphere")
same = all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))
print(f"  identical: {same}")
assert same, "sphere mode is NOT bit-identical — the legacy path changed"

print("\nfield mode, gp prior, distinct kernels")
f = render(
    synthetic_lesion_mode="field",
    synthetic_field_prior="gp",
    synthetic_field_kernels="distinct",
    synthetic_wm_softness=0.02,
)
v1, v2, lat = f[0]
print(f"  z_lesion present: {'z_lesion' in lat}   shape {tuple(lat['z_lesion'].shape) if 'z_lesion' in lat else None}")
print(f"  field_lengthscales: {lat.get('field_lengthscales')}")
print(f"  view1 range [{v1.min():.3f}, {v1.max():.3f}]   view2 range [{v2.min():.3f}, {v2.max():.3f}]")
print(f"  differs from sphere: {not np.array_equal(v1, a[0][0])}")

print("\nlesion load is smooth in the latent (injectivity check)")
ds = SyntheticBrainDataset(
    **BASE, synthetic_lesion_mode="field", synthetic_field_prior="gp", synthetic_wm_softness=0.02
)
r = ds._inner.renderer if hasattr(ds, "_inner") else None
if r is not None:
    zc = torch.randn(9)
    zd = torch.randn(8, 8, 8)
    zf = torch.randn(8, 8, 8)
    outs = []
    for eps in (0.0, 0.05, 0.10):
        zl = torch.randn(8, 8, 8) if eps == 0.0 else zl_base + eps
        if eps == 0.0:
            zl_base = zl
        _, load = r.render_structure(zc, zd, zf, torch.device("cpu"), z_lesion=zl)
        outs.append(float(load.sum()))
    print(f"  total lesion load vs +eps shift: {[round(o, 2) for o in outs]}")
    print("  strictly increasing (monotone sigmoid):", outs[0] < outs[1] < outs[2])
