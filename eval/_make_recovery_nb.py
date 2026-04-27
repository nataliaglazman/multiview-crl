"""Builds eval/analyze_synthetic_recovery.ipynb. Run once: `python eval/_make_recovery_nb.py`."""
import json
import os

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    CELLS.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": text.splitlines(keepends=True),
        }
    )


md(
    """# Synthetic recovery analysis

How well does the trained VQ-VAE recover the **ground-truth generative factors** of the synthetic
dataset? For each (factor, encoder code-group) pair we fit a probe (linear + MLP) and report R².

A well-disentangled model should show:
- **Content code → shared factors** (z_content, z_deformation, z_fissure): high R²
- **Style code → per-view factors** (z_style_v1 / v2): high R² for matching view
- **Cross terms** (content code → style factor, style code → content factor): low R²
- **Content code is view-invariant**: encoding view 1 vs view 2 of the same sample yields nearly
  identical content vectors

Set `CHECKPOINT_PATH` below to point at your `vqvae_model.pt` from a synthetic run.
"""
)

code(
    """\
import os, sys, json
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ---- USER CONFIG ---------------------------------------------------------
CHECKPOINT_PATH = "/path/to/results/synthetic/<TAG>/vqvae_model.pt"   # <-- edit
NUM_SAMPLES     = 500          # samples to encode for probing
PROBE_TYPES     = ("linear", "mlp")  # set to ("linear",) to skip MLP
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------------------------------------

settings_path = os.path.join(os.path.dirname(CHECKPOINT_PATH), "settings.json")
with open(settings_path) as f:
    settings = json.load(f)

print(f"Loaded settings from {settings_path}")
print(f"  dataset_name      : {settings.get('dataset_name')}")
print(f"  synthetic_mode    : {settings.get('synthetic_mode')}")
print(f"  vqvae_nb_levels   : {settings.get('vqvae_nb_levels')}")
print(f"  content_style_lvls: {settings.get('content_style_levels')}")
"""
)

md("## 1. Build & load the VQ-VAE\n")

code(
    """\
import models.vqvae as vqvae

# Match the construction used in main_multimodal.py for VQ-VAE.
# Pull the per-level content_channels from the checkpoint when possible
# (so we don't need to recompute them from ratios).
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
state_dict = checkpoint["encoders"]
prefix = "online." if any(k.startswith("online.") for k in state_dict) else ""

hidden_channels = settings["vqvae_hidden_channels"]
nb_levels       = settings["vqvae_nb_levels"]
embed_dim       = settings["vqvae_embed_dim"]
content_style_levels = settings.get("content_style_levels", [0])

# Per-level content_channels detection from codebook conv_in widths.
content_ch_per_level = {}
for lvl in content_style_levels:
    cb_key = f"{prefix}module.codebooks.{lvl}.conv_in.weight"
    if cb_key in state_dict:
        cb_in = state_dict[cb_key].shape[1]
        if lvl == nb_levels - 1:
            content_ch_per_level[lvl] = cb_in
        else:
            content_ch_per_level[lvl] = cb_in - embed_dim

print("Detected content_channels per level:", content_ch_per_level)

# Build the model. Most kwargs come straight from settings.
vqvae_model = vqvae.VQVAE(
    in_channels=1,
    hidden_channels=hidden_channels,
    res_channels=settings.get("vqvae_res_channels", 32),
    nb_res_layers=settings.get("vqvae_nb_res_layers", 2),
    nb_levels=nb_levels,
    embed_dim=embed_dim,
    nb_entries=settings.get("vqvae_nb_entries", 256),
    scaling_rates=settings.get("vqvae_scaling_rates", [2, 2, 2]),
    content_size=max(content_ch_per_level.values()) if content_ch_per_level else hidden_channels,
    style_size=hidden_channels - (max(content_ch_per_level.values()) if content_ch_per_level else hidden_channels),
    content_style_levels=content_style_levels,
    mask_mode=settings.get("mask_mode", "fixed"),
    separate_encoders=settings.get("separate_encoders", False),
    pass_full_to_next_level=settings.get("pass_full_to_next_level", False),
).to(DEVICE)

# Wrap in DataParallel to match training-time key prefixes ("module.").
vqvae_model = torch.nn.DataParallel(vqvae_model)

# Strip the "online." prefix if needed (MoCo wrapper).
clean = {k[len("online."):] if k.startswith("online.") else k: v for k, v in state_dict.items()}
missing, unexpected = vqvae_model.load_state_dict(clean, strict=False)
print(f"  missing keys   : {len(missing)} (showing first 3) {missing[:3]}")
print(f"  unexpected keys: {len(unexpected)} (showing first 3) {unexpected[:3]}")
vqvae_model.eval();
"""
)

md("## 2. Build the synthetic dataset\n")

code(
    """\
from data.datasets import SyntheticBrainDataset

# Use the val split so we don't probe on training data.
res = settings.get("spatial_size", [32, 32, 32])[0]
ds = SyntheticBrainDataset(
    mode="val",
    spatial_size=tuple(settings.get("spatial_size", [res, res, res])),
    synthetic_mode=settings.get("synthetic_mode", "pseudo_mri"),
    synthetic_seed=settings.get("synthetic_seed", 42),
    synthetic_n_content=settings.get("synthetic_n_content", 5),
    synthetic_n_style=settings.get("synthetic_n_style", 3),
    synthetic_num_samples=NUM_SAMPLES,
)
print(f"Dataset: {len(ds)} samples at res={res}")

# Sanity: pull one item, inspect shapes.
item = ds[0]
print("image shapes :", [x.shape for x in item["image"]])
print("mask shapes  :", [x.shape for x in item["mask"]])
print("gt latent keys:", list(item["gt_latents"].keys()))
"""
)

md(
    """## 3. Encode the validation set

For each sample we run both views through the encoder and capture the per-level pooled features
plus the model-applied content/style mask split.
"""
)

code(
    """\
@torch.no_grad()
def encode_batch(model, x, n_views=2):
    \"\"\"Returns encoder_outputs (list of pooled features per level) and the
    soft content masks (dict: level -> mask tensor or (mask_v0, mask_v1) tuple).\"\"\"
    out = model(x, return_recon=False, pool_only=True, n_views=n_views,
                subsets=[(0, 1)], patch_grid=None)
    # forward returns: (recon, diffs, encoder_outputs, est_idx, _, _, fwd_soft_masks, _)
    encoder_outputs = out[2]
    soft_masks = out[6]
    return encoder_outputs, soft_masks


# Pre-allocate storage
features_per_level = {lvl: {"v1": [], "v2": []} for lvl in range(nb_levels)}
gt = {k: [] for k in ds[0]["gt_latents"].keys()}

batch_size = 16
loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

for batch in loader:
    v1, v2 = batch["image"]
    x = torch.cat([v1, v2], dim=0).to(DEVICE)
    enc_outs, soft_masks = encode_batch(vqvae_model, x, n_views=2)

    for lvl, enc in enumerate(enc_outs):
        # enc shape: (n_views * B, C) after pool_only=True
        n_views_x_b = enc.shape[0]
        B = n_views_x_b // 2
        feats = enc.reshape(2, B, -1).cpu().numpy()
        features_per_level[lvl]["v1"].append(feats[0])
        features_per_level[lvl]["v2"].append(feats[1])

    for k in gt:
        v = batch["gt_latents"][k]
        gt[k].append(v.cpu().numpy())

# Concatenate
for lvl in features_per_level:
    for view in ("v1", "v2"):
        features_per_level[lvl][view] = np.concatenate(features_per_level[lvl][view], 0)
gt = {k: np.concatenate(v, 0) for k, v in gt.items()}

# Flatten ground-truth tensors so each sample becomes a vector.
gt_flat = {k: v.reshape(v.shape[0], -1).astype(np.float32) for k, v in gt.items()}

print("Per-level encoder feature shapes (view 1):")
for lvl, d in features_per_level.items():
    print(f"  level {lvl}: {d['v1'].shape}")

print("\\nGround-truth factor shapes:")
for k, v in gt_flat.items():
    print(f"  {k}: {v.shape}")
"""
)

md(
    """## 4. Split each level's features into content / style channels

We use the model's actual mask (so the split matches what the contrastive loss saw).
Levels not in `content_style_levels` are treated as "all content".
"""
)

code(
    """\
def get_mask_indices(soft_masks, lvl, n_channels):
    \"\"\"Return (content_idx, style_idx) for a given level.\"\"\"
    if lvl not in soft_masks:
        return np.arange(n_channels), np.array([], dtype=int)
    m = soft_masks[lvl]
    if isinstance(m, tuple):
        m = m[0]   # use view-0 mask; view-1 mask is identical when masks are symmetric
    m = m.detach().cpu().numpy().flatten()
    content_idx = np.where(m > 0.5)[0]
    style_idx   = np.where(m <= 0.5)[0]
    return content_idx, style_idx


# Re-grab one batch's masks (they don't change between batches)
v1, v2 = ds[0]["image"]
x = torch.cat([v1[None], v2[None]], 0).to(DEVICE)
_, masks_for_split = encode_batch(vqvae_model, x, n_views=2)

splits = {}
for lvl in range(nb_levels):
    n_ch = features_per_level[lvl]["v1"].shape[-1]
    c_idx, s_idx = get_mask_indices(masks_for_split, lvl, n_ch)
    splits[lvl] = {"content": c_idx, "style": s_idx}
    print(f"level {lvl}: {n_ch} channels → content={len(c_idx)}, style={len(s_idx)}")
"""
)

md("## 5. Probe utilities\n")

code(
    """\
def fit_probe(X, y, kind="linear"):
    \"\"\"Train a probe and return test R² (averaged across output dims if y is multi-dim).\"\"\"
    if X.shape[1] == 0 or X.shape[0] < 20:
        return float("nan")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    if kind == "linear":
        model = Ridge(alpha=1.0)
    else:
        model = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=500,
                              random_state=0, early_stopping=True, tol=1e-4)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return float(r2_score(yte, pred, multioutput="variance_weighted"))


def build_code_groups(features_per_level, splits):
    \"\"\"Return dict: name -> (n_samples, n_features) ndarray.\"\"\"
    groups = {}
    for lvl in features_per_level:
        c_idx = splits[lvl]["content"]
        s_idx = splits[lvl]["style"]
        for view in ("v1", "v2"):
            f = features_per_level[lvl][view]
            groups[f"L{lvl}_content_{view}"] = f[:, c_idx] if len(c_idx) else np.zeros((len(f), 0))
            groups[f"L{lvl}_style_{view}"]   = f[:, s_idx] if len(s_idx) else np.zeros((len(f), 0))
        groups[f"L{lvl}_all_v1"] = features_per_level[lvl]["v1"]
    return groups


code_groups = build_code_groups(features_per_level, splits)
print("Code groups:", list(code_groups.keys()))
"""
)

md("## 6. Compute the R² matrix (factor × code group)\n")

code(
    """\
factor_names = list(gt_flat.keys())
group_names  = list(code_groups.keys())

results = {kind: pd.DataFrame(index=factor_names, columns=group_names, dtype=float)
           for kind in PROBE_TYPES}

for kind in PROBE_TYPES:
    for fname in factor_names:
        y = gt_flat[fname]
        if y.shape[1] > 64:
            # For very high-dim ground-truth (e.g. brain_mask flattened), reduce
            # to the top-32 PCA components so the MLP probe doesn't blow up.
            from sklearn.decomposition import PCA
            y = PCA(n_components=32, random_state=0).fit_transform(y)
        for gname in group_names:
            X = code_groups[gname]
            results[kind].loc[fname, gname] = fit_probe(X, y, kind=kind)
    print(f"=== Probe: {kind} ===")
    print(results[kind].round(3))
    print()
"""
)

md("## 7. Heatmap visualisation\n")

code(
    """\
def plot_r2_heatmap(df, title):
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(df.columns)),
                                     max(4, 0.45 * len(df.index))))
    data = df.astype(float).values
    im = ax.imshow(data, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df.index)));   ax.set_yticklabels(df.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=8)
    ax.set_title(title); plt.colorbar(im, ax=ax, label="R²")
    plt.tight_layout(); plt.show()


for kind in PROBE_TYPES:
    plot_r2_heatmap(results[kind], f"Factor recovery R² ({kind} probe)")
"""
)

md(
    """## 8. Cross-view content consistency

If content is truly view-invariant, encoding view 1 vs view 2 of the same sample should yield
nearly identical content vectors. We measure this with cosine similarity per level.
"""
)

code(
    """\
from numpy.linalg import norm

def cos_sim_paired(A, B):
    a = A / (norm(A, axis=1, keepdims=True) + 1e-8)
    b = B / (norm(B, axis=1, keepdims=True) + 1e-8)
    return (a * b).sum(axis=1)


print("Cross-view cosine similarity (mean ± std):")
for lvl in range(nb_levels):
    c_idx = splits[lvl]["content"]
    s_idx = splits[lvl]["style"]
    v1, v2 = features_per_level[lvl]["v1"], features_per_level[lvl]["v2"]
    if len(c_idx):
        sim_c = cos_sim_paired(v1[:, c_idx], v2[:, c_idx])
        print(f"  level {lvl} CONTENT: {sim_c.mean():.3f} ± {sim_c.std():.3f}")
    if len(s_idx):
        sim_s = cos_sim_paired(v1[:, s_idx], v2[:, s_idx])
        print(f"  level {lvl} STYLE  : {sim_s.mean():.3f} ± {sim_s.std():.3f}  (lower = better)")
"""
)

md(
    """## 9. Summary metrics

Three numbers that summarise disentanglement quality:

- **Content purity**: avg R² for `content code → shared factors` minus avg R² for `content code → per-view factors`. Higher is better.
- **Style purity**: avg R² for `style code → per-view factors` minus avg R² for `style code → shared factors`. Higher is better.
- **View invariance**: avg cross-view cosine similarity of the content code. Closer to 1 is better.
"""
)

code(
    """\
SHARED_FACTORS  = {"z_content", "z_deformation", "z_fissure", "brain_mask"}
PER_VIEW_FACTORS = {"z_style_v1", "z_style_v2"}

primary_kind = PROBE_TYPES[0]
df = results[primary_kind]

def avg_r2(df, factor_set, group_substr):
    cols = [c for c in df.columns if group_substr in c]
    rows = [f for f in df.index   if f in factor_set]
    if not cols or not rows: return float("nan")
    return df.loc[rows, cols].astype(float).values.mean()


content_on_shared   = avg_r2(df, SHARED_FACTORS, "content_v1")
content_on_perview  = avg_r2(df, PER_VIEW_FACTORS, "content_v1")
style_on_perview    = avg_r2(df, PER_VIEW_FACTORS, "style_v1")
style_on_shared     = avg_r2(df, SHARED_FACTORS, "style_v1")

print(f"Content purity = {content_on_shared:.3f} - {content_on_perview:.3f} = "
      f"{content_on_shared - content_on_perview:+.3f}")
print(f"Style   purity = {style_on_perview:.3f} - {style_on_shared:.3f} = "
      f"{style_on_perview - style_on_shared:+.3f}")

# View invariance averaged across levels with content channels
sims = []
for lvl in range(nb_levels):
    c_idx = splits[lvl]["content"]
    if len(c_idx):
        v1, v2 = features_per_level[lvl]["v1"], features_per_level[lvl]["v2"]
        sims.append(cos_sim_paired(v1[:, c_idx], v2[:, c_idx]).mean())
print(f"View invariance (mean cos-sim) = {np.mean(sims):.3f}")
"""
)

md(
    """## What the numbers should look like

| Metric | Healthy run | Collapsed encoder | Style-leaks-into-content |
|---|---|---|---|
| Content purity | > 0.4 | ~0 | ≤ 0 |
| Style purity   | > 0.4 | ~0 | small positive |
| View invariance | > 0.85 | varies | low |

If content purity is high but style purity is near 0, the model collapsed style channels — they
might be unused, in which case lower `--total-dim` next time. If both are near 0, the encoder isn't
learning anything useful (check loss curves).
"""
)

# Build the notebook JSON
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = os.path.join(os.path.dirname(__file__), "analyze_synthetic_recovery.ipynb")
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {OUT} ({len(CELLS)} cells)")
