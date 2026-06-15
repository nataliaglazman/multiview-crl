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
# Conv_in shape is (lvl_entries, content_ch [+ embed_dim if has-upper-conditioning], 1, 1, 1).
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

# Per-level content ratios — required when the trained pyramid is non-uniform
# (e.g. L0=10/22, L1=6/10, L2=4/6). VQVAE.build computes
# `content_channels_per_level[lvl] = round(ratio * hidden_channels)`, so we
# back-derive each ratio from the detected content channel counts. Without
# this the constructor falls back to `default_ratio = content_size / total`
# and all levels collapse to the same width, producing the load_state_dict
# size-mismatch errors.
levels_sorted = sorted(content_ch_per_level.keys())
if levels_sorted:
    detected_ratios = [content_ch_per_level[lvl] / hidden_channels for lvl in levels_sorted]
    # Prefer settings.json if present (cleaner provenance), else fall back to detected.
    content_ratios_arg = settings.get("content_ratios") or detected_ratios
    _first_content = content_ch_per_level[levels_sorted[0]]
    content_size_arg = max(1, _first_content)
    style_size_arg = max(1, hidden_channels - _first_content)
else:
    content_ratios_arg = None
    content_size_arg = hidden_channels
    style_size_arg = 0

# Forward every shape-sensitive kwarg we know about. Anything missing from
# settings.json defaults to the constructor's own default.
def _opt(name, default=None):
    return settings.get(name, default)

vqvae_model = vqvae.VQVAE(
    in_channels=1,
    hidden_channels=hidden_channels,
    res_channels=_opt("vqvae_res_channels", 32),
    nb_res_layers=_opt("vqvae_nb_res_layers", 2),
    nb_levels=nb_levels,
    embed_dim=embed_dim,
    nb_entries=_opt("vqvae_nb_entries", 256),
    scaling_rates=_opt("vqvae_scaling_rates", [2, 2, 2]),
    content_size=content_size_arg,
    style_size=style_size_arg,
    content_ratios=content_ratios_arg,
    content_style_levels=content_style_levels,
    mask_mode=_opt("mask_mode", "fixed"),
    separate_encoders=_opt("separate_encoders", False),
    pass_full_to_next_level=_opt("pass_full_to_next_level", False),
    narrow_encoder_input=_opt("narrow_encoder_input", False),
    use_content_projection=_opt("use_content_projection", False),
    inject_style_to_decoder=_opt("inject_style_to_decoder", False),
    quantize_style=_opt("quantize_style", False),
    separate_content_codebooks=_opt("separate_content_codebooks", False),
    separate_style_codebooks=_opt("separate_style_codebooks", False),
    style_embed_dim=_opt("style_embed_dim", None),
    style_nb_entries=_opt("style_nb_entries", None),
    style_injection_mode=_opt("style_injection_mode", "concat"),
    top_level_recon_only=_opt("top_level_recon_only", False),
    skip_decoder_concat_levels=_opt("skip_decoder_concat_levels", None),
    final_recon_norm=not _opt("no_final_recon_norm", False),
).to(DEVICE)
print(f"Built VQVAE with content_ratios={content_ratios_arg}")

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
_inner_for_hooks = vqvae_model.module if hasattr(vqvae_model, "module") else vqvae_model


def install_encoder_hooks(model_inner):
    \"\"\"Install forward hooks on each encoder level so we can capture spatial outputs.

    The model's `forward` nulls `encoder_outputs[l]` after decoding to free memory,
    so we have to grab the spatial maps in-flight. This works regardless of
    `pool_only` / `return_recon` settings.
    \"\"\"
    captured = {"v0": [None] * model_inner.nb_levels, "v1": [None] * model_inner.nb_levels}
    handles = []

    def make_hook(view, lvl):
        def fn(module, inputs, output):
            captured[view][lvl] = output.detach()
        return fn

    sep = getattr(model_inner, "separate_encoders", False) and (model_inner.encoders_v1 is not None)
    for i, enc in enumerate(model_inner.encoders):
        # When NOT separated, both views go through the same encoder via a concat
        # batch — we capture and split below.
        handles.append(enc.register_forward_hook(make_hook("v0" if sep else "shared", i)))
    if sep:
        for i, enc in enumerate(model_inner.encoders_v1):
            handles.append(enc.register_forward_hook(make_hook("v1", i)))

    return captured, handles, sep


@torch.no_grad()
def encode_batch(model, x, n_views=2):
    \"\"\"Run forward, return spatial encoder maps per level + soft masks.

    We use forward hooks to capture encoder outputs because the model nulls
    its internal `encoder_outputs` list after the decoder consumes them.
    Spatial maps are then patch-pooled downstream (see ``pool_spatial``)
    rather than globally pooled — the trainer's std-only global pool throws
    away the layout that ``z_deformation`` / ``z_fissure`` / ``brain_mask``
    live in, so probing on it would under-report a working model.
    \"\"\"
    captured, handles, sep = install_encoder_hooks(_inner_for_hooks)
    try:
        # pool_only=True + return_recon=False keeps memory low (skips decoder)
        # but the value returned at out[2] (encoder_pools) is the constant
        # mean-pool we don't want. We use the hooks instead.
        out = model(x, return_recon=False, pool_only=True, n_views=n_views,
                    subsets=[(0, 1)], patch_grid=None)
        soft_masks = out[6]
    finally:
        for h in handles:
            h.remove()

    # Build per-level spatial maps in (n_views*B, C, D, H, W) layout.
    spatial_maps = []
    if sep:
        for lvl in range(_inner_for_hooks.nb_levels):
            v0 = captured["v0"][lvl]
            v1 = captured["v1"][lvl]
            spatial_maps.append(torch.cat([v0, v1], dim=0))
    else:
        # Shared encoder was called once with concat batch; capture is already
        # the (n_views*B, C, ...) tensor.
        for lvl in range(_inner_for_hooks.nb_levels):
            spatial_maps.append(captured["shared"][lvl])
    return spatial_maps, soft_masks


PATCH_GRID = (4, 4, 4)  # adaptive-pool every level to 4³=64 patches → preserves spatial layout


def pool_spatial(s, grid=PATCH_GRID):
    \"\"\"Adaptive-pool (B, C, D, H, W) → (B, C, P), P = prod(grid).

    Replaces global mean/std pooling. Spatial GT factors (z_deformation,
    z_fissure, brain_mask) only live in the spatial dimensions — collapsing
    those to per-channel scalars guarantees the probe can't recover them
    regardless of what the encoder learned.
    \"\"\"
    return F.adaptive_avg_pool3d(s, grid).flatten(2)  # (B, C, P)


# Pre-allocate storage
features_per_level = {lvl: {"v1": [], "v2": []} for lvl in range(nb_levels)}
n_channels_per_level = {}
gt = {k: [] for k in ds[0]["gt_latents"].keys()}

batch_size = 16
loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

for batch in loader:
    v1, v2 = batch["image"]
    x = torch.cat([v1, v2], dim=0).to(DEVICE)
    spatial_maps, soft_masks = encode_batch(vqvae_model, x, n_views=2)

    for lvl, s in enumerate(spatial_maps):
        n_channels_per_level[lvl] = s.shape[1]
        pooled = pool_spatial(s)                                # (2*B, C, P)
        n2B, C, P = pooled.shape; B = n2B // 2
        feats = pooled.reshape(2, B, C * P).cpu().numpy()       # flatten (C, P) → C*P
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

P = int(np.prod(PATCH_GRID))
print(f"Per-level pooled feature shapes (view 1) — patch-pooled, reshape order (C × {P}):")
for lvl, d in features_per_level.items():
    C = n_channels_per_level[lvl]
    print(f"  level {lvl}: {d['v1'].shape}  (C={C}, P={P}, flat={C*P})")

print("\\nGround-truth factor shapes:")
for k, v in gt_flat.items():
    print(f"  {k}: {v.shape}")

# Sanity: patch features should vary across samples even where the global mean was constant.
print("\\nSanity check (per-feature std across samples — should be > 0):")
for lvl in range(nb_levels):
    f = features_per_level[lvl]["v1"]
    print(f"  L{lvl}: mean across-sample std={f.std(0).mean():.4f}")
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
    \"\"\"Return channel indices (in the original C-wide spatial map) for content / style.\"\"\"
    if lvl not in soft_masks:
        return np.arange(n_channels), np.array([], dtype=int)
    m = soft_masks[lvl]
    if isinstance(m, tuple):
        m = m[0]   # use view-0 mask; view-1 mask is identical when masks are symmetric
    m = m.detach().cpu().numpy().flatten()
    content_idx = np.where(m > 0.5)[0]
    style_idx   = np.where(m <= 0.5)[0]
    return content_idx, style_idx


def to_flat_indices(c_idx, s_idx, C, P):
    \"\"\"Map channel indices into the C*P-wide flat (channel × patch) layout.

    With reshape order (B, C, P) → (B, C*P), channel ``c``'s ``P`` patches
    occupy positions ``[c*P, (c+1)*P)`` in the flat vector.
    \"\"\"
    def expand(idx):
        if len(idx) == 0:
            return np.array([], dtype=int)
        return np.concatenate([np.arange(c * P, (c + 1) * P) for c in idx])
    return expand(c_idx), expand(s_idx)


# Re-grab one batch's masks (they don't change between batches)
v1, v2 = ds[0]["image"]
x = torch.cat([v1[None], v2[None]], 0).to(DEVICE)
_, masks_for_split = encode_batch(vqvae_model, x, n_views=2)

P = int(np.prod(PATCH_GRID))
splits = {}
for lvl in range(nb_levels):
    C = n_channels_per_level[lvl]
    c_idx, s_idx = get_mask_indices(masks_for_split, lvl, C)
    c_flat, s_flat = to_flat_indices(c_idx, s_idx, C, P)
    splits[lvl] = {"content": c_flat, "style": s_flat}
    print(f"level {lvl}: C={C} channels (P={P} patches each) → "
          f"content={len(c_idx)}, style={len(s_idx)} "
          f"(flat indices: content={len(c_flat)}, style={len(s_flat)})")
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

md(
    """## 10. TB sanity check — did the contrastive loss actually engage during training?

Open the run's TensorBoard log dir (`<save_dir>/tb/` or whatever your trainer uses) and look
for these scalars:

- **`contrastive/level_*/top1_acc`** (or similarly named): for batch size B, *chance* is `1/B`.
  If this stays near `1/B` for the whole run → the contrastive loss never engaged.
  Healthy runs climb above 0.5 within a few thousand steps.
- **`contrastive/level_*/loss`**: should drop from `ln(B)` toward 0 (e.g. ln(16) ≈ 2.77 → < 1.0).

Quick programmatic check below — pulls scalars from the TB event file using `tbparse` if
installed, otherwise prints the path you can open manually.
"""
)

code(
    """\
import glob
tb_dir = os.path.join(os.path.dirname(CHECKPOINT_PATH), "tb")
event_files = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))
print(f"Looking for TB events in: {tb_dir}")
print(f"Found {len(event_files)} event file(s).")

if not event_files:
    print("\\nNo TB events found — open your run's log dir manually and check that "
          "contrastive accuracy climbed above 1/batch_size.")
else:
    try:
        from tbparse import SummaryReader
        reader = SummaryReader(tb_dir, pivot=False)
        scalars = reader.scalars
        contrastive_tags = [t for t in scalars["tag"].unique()
                             if "contrast" in t.lower() and ("acc" in t.lower() or "loss" in t.lower())]
        print(f"\\nFound {len(contrastive_tags)} contrastive scalar(s).")
        for tag in sorted(contrastive_tags)[:12]:
            sub = scalars[scalars["tag"] == tag].sort_values("step")
            if len(sub) == 0: continue
            first, last = sub.iloc[0]["value"], sub.iloc[-1]["value"]
            print(f"  {tag:60s}  first={first:.4f}  last={last:.4f}  Δ={last-first:+.4f}")

        # Hard call: did top-1 accuracy actually move?
        acc_tags = [t for t in contrastive_tags if "acc" in t.lower()]
        if acc_tags:
            print("\\nVerdict on contrastive engagement:")
            for tag in acc_tags:
                sub = scalars[scalars["tag"] == tag].sort_values("step")
                last = sub.iloc[-1]["value"]
                B = settings.get("batch_size", 16)
                chance = 1.0 / B
                if last > chance * 3:
                    verdict = "✓ engaged"
                elif last > chance * 1.5:
                    verdict = "~ marginal"
                else:
                    verdict = "✗ STUCK AT CHANCE — contrastive loss never engaged"
                print(f"  {tag}: last={last:.3f}, chance≈{chance:.3f}  → {verdict}")
    except ImportError:
        print("\\n`tbparse` not installed (pip install tbparse). Open the TB dir manually:")
        print(f"  tensorboard --logdir {tb_dir}")
"""
)

md(
    """### If contrastive accuracy is at chance

Then the bug is upstream of this notebook — the contrastive loss never learned to align cross-view
positives. The most likely cause given the GroupNorm+mean-pool issue we just diagnosed:

> The trainer also computes its contrastive loss on a mean-pooled feature whose per-channel
> spatial mean is constant across samples (since GroupNorm + affine bias). With constant features,
> all positives and negatives have the same similarity, so the InfoNCE loss is at `ln(B)` forever.

Two possible fixes for the trainer (not applied — flagging for discussion):

1. **Pool with std instead of mean** in the contrastive feature path
   (`models/vqvae.py` around line 1103: replace `enc_in_v0_pool.mean(dim=[2,3,4])` with
   `enc_in_v0_pool.std(dim=[2,3,4])`, or concat both).
2. **Pool BEFORE the encoder's final GroupNorm**, by exposing an intermediate-feature hook.

Option 1 is one line and matches what this notebook does to recover signal.
"""
)

md(
    """## 11. Discrete codebook recovery

Everything above probes the **continuous** patch-pooled encoder features. But the decoder only
ever sees the **quantized codebook indices** — so the questions that matter for identifiability are
really about the discrete codes. This block extracts the actual `id_outputs` (content codes) and
`style_id_outputs` (style codes) and relates them to the ground-truth factors.

Two things to know:

- The extraction loop above used `pool_only=True`, which sets `skip_codebook=True` inside
  `VQVAE.forward` — so every `id_output` comes back `None`. We **must** re-encode with
  `return_recon=True, pool_only=False` (the training-style forward) to get real codes.
- Content uses a **shared** codebook across views (unless `separate_content_codebooks`), which makes
  an *exact* cross-view identifiability test possible (Section 13): the same subject's content codes
  should be identical in T1 and T2.
"""
)

code(
    """\
# --- Re-encode the val set and capture discrete codes -----------------------
# return_recon=True is REQUIRED: with pool_only the model skips the codebook and
# id_outputs come back as None.
DISCRETE_NUM_SAMPLES = min(NUM_SAMPLES, 300)   # runs the decoder, so cap for memory
DISCRETE_BATCH       = 8

_m = vqvae_model.module if hasattr(vqvae_model, "module") else vqvae_model
separate_content_cb = bool(settings.get("separate_content_codebooks", False))
separate_style_cb   = bool(settings.get("separate_style_codebooks", False))
quantize_style      = bool(settings.get("quantize_style", False))

# Codebook sizes (K) per level — read from the model so bincount lengths are exact.
content_K = {}
for lvl in range(nb_levels):
    cb = _m.codebooks[lvl]
    content_K[lvl] = int(getattr(cb, "n_embed", _opt("vqvae_nb_entries", 256)))
style_K = {}
if quantize_style and hasattr(_m, "style_codebooks"):
    for k_lvl, cb in _m.style_codebooks.items():
        style_K[int(k_lvl)] = int(getattr(cb, "n_embed", _opt("vqvae_nb_entries", 256)))
print("content codebook sizes:", content_K)
print("style codebook sizes  :", style_K if style_K else "(no style quantization)")


@torch.no_grad()
def extract_codes(model, x, n_views=2):
    out = model(x, return_recon=True, pool_only=False, n_views=n_views, subsets=[(0, 1)])
    id_raw    = out[5]        # list, appended coarse->fine (loop runs level nb-1 .. 0)
    style_raw = out[7]        # dict keyed by actual level index
    id_lvl    = id_raw[::-1]  # reverse so id_lvl[lvl] is level `lvl` (0 = finest)
    return id_lvl, style_raw


content_codes    = {lvl: {"v0": [], "v1": []} for lvl in range(nb_levels)}
style_codes      = {lvl: {"v0": [], "v1": []} for lvl in style_K}
code_grid_shape  = {}    # lvl -> (d, h, w) of the code map, needed for spatial analysis

# Collect whatever GT factors this synthetic mode actually exposes. NOTE: the
# SyntheticBrainDataset wrapper pops "brain_mask" out of gt_latents (it becomes
# the separate `mask` field), so it is NOT a gt_latents key — we capture it from
# batch["mask"] below for the Section 14 fallback.
_WANT_GT = ("z_content", "z_style_v1", "z_style_v2", "z_deformation", "z_fissure")
_avail_gt = set(ds[0]["gt_latents"].keys())
gtc = {k: [] for k in _WANT_GT if k in _avail_gt}
fg_masks = []    # foreground (brain) mask, view 0 — spatial fallback if the renderer is absent
print("GT factors found:", list(gtc.keys()))

loader_d = torch.utils.data.DataLoader(ds, batch_size=DISCRETE_BATCH, shuffle=False, num_workers=0)
seen = 0
for batch in loader_d:
    if seen >= DISCRETE_NUM_SAMPLES:
        break
    v1, v2 = batch["image"]
    B = v1.shape[0]
    x = torch.cat([v1, v2], dim=0).to(DEVICE)
    id_lvl, style_raw = extract_codes(vqvae_model, x, n_views=2)

    for lvl in range(nb_levels):
        ids = id_lvl[lvl]
        if ids is None:
            continue
        code_grid_shape[lvl] = tuple(ids.shape[1:])           # (d, h, w)
        ids = ids.reshape(2 * B, -1).cpu().to(torch.int32).numpy()   # (2B, n_voxels)
        content_codes[lvl]["v0"].append(ids[:B])
        content_codes[lvl]["v1"].append(ids[B:])
    for lvl, sid in style_raw.items():
        sid = sid.reshape(2 * B, -1).cpu().to(torch.int32).numpy()
        style_codes[lvl]["v0"].append(sid[:B])
        style_codes[lvl]["v1"].append(sid[B:])

    for k in gtc:
        gtc[k].append(batch["gt_latents"][k].cpu().numpy())
    if "mask" in batch:
        fg_masks.append(np.asarray(batch["mask"][0].cpu().numpy()))
    seen += B

for lvl in list(content_codes):
    if content_codes[lvl]["v0"]:
        for vw in ("v0", "v1"):
            content_codes[lvl][vw] = np.concatenate(content_codes[lvl][vw], 0)
    else:
        del content_codes[lvl]
for lvl in list(style_codes):
    for vw in ("v0", "v1"):
        style_codes[lvl][vw] = np.concatenate(style_codes[lvl][vw], 0)
gtc = {k: np.concatenate(v, 0) for k, v in gtc.items()}
fg_masks = np.concatenate(fg_masks, 0) if fg_masks else None

N_d = gtc["z_content"].shape[0]
print()
print(f"Extracted discrete codes for {N_d} samples.")
for lvl in content_codes:
    print(f"  L{lvl} content grid {code_grid_shape[lvl]} -> {content_codes[lvl]['v0'].shape}  K={content_K[lvl]}")
for lvl in style_codes:
    print(f"  L{lvl} style   grid -> {style_codes[lvl]['v0'].shape}  K={style_K[lvl]}")
"""
)

md(
    """### Codebook utilisation

Effective vocabulary (perplexity) and number of used entries per level. The synthetic generator has
**9 content + 3 style** scalar factors (plus two spatial fields): if perplexity collapses toward 1,
or only a handful of entries are used, the codebook is bottlenecking the factors rather than encoding
them.
"""
)

code(
    """\
def perplexity(codes_flat, K):
    counts = np.bincount(codes_flat.reshape(-1), minlength=K).astype(np.float64)
    p = counts / counts.sum()
    nz = p[p > 0]
    ppl = float(np.exp(-(nz * np.log(nz)).sum()))
    return ppl, int((counts > 0).sum())


print("Codebook utilisation (perplexity = effective vocabulary size):")
for lvl in content_codes:
    allc = np.concatenate([content_codes[lvl]["v0"], content_codes[lvl]["v1"]], 0)
    ppl, used = perplexity(allc, content_K[lvl])
    print(f"  L{lvl} content: perplexity={ppl:6.1f}  used={used:4d}/{content_K[lvl]}  ({100*used/content_K[lvl]:.0f}%)")
for lvl in style_codes:
    allc = np.concatenate([style_codes[lvl]["v0"], style_codes[lvl]["v1"]], 0)
    ppl, used = perplexity(allc, style_K[lvl])
    print(f"  L{lvl} style  : perplexity={ppl:6.1f}  used={used:4d}/{style_K[lvl]}")
"""
)

md(
    """## 12. Code -> ground-truth factor mutual information (probe-free)

Every R² number in Section 6 came from a *fitted probe*, so it conflates "the information is there"
with "my probe found it" (the capacity confound that makes block-MCC read high at small content
sizes). Discrete codes let us sidestep the probe: we summarise each sample as a **bag-of-codes
histogram** and estimate the mutual information between each codebook entry's usage and each scalar
GT factor directly. The result is a DCI-shaped `codes × factors` importance matrix.

What a clean run looks like:
- **content codes** carry high MI with `z_content`, near-zero MI with `z_style` (the separation claim,
  measured on the representation the decoder actually consumes);
- **style codes** (if `quantize_style`) carry high MI with `z_style`, near-zero with `z_content`.
"""
)

code(
    """\
from sklearn.feature_selection import mutual_info_regression
from eval.dci import disentanglement as _disent, completeness as _complete

CONTENT_FACTOR_LABELS = ["brain_size", "ventricle_size", "lesion_x", "lesion_y", "lesion_z",
                         "cortical_thickness", "temporal_atrophy", "lr_asymmetry", "sulcal_widening"]


def scalar_factor_table():
    # returns list of (label, values (N,), group) for the scalar GT factors
    out = []
    nc = gtc["z_content"].shape[1]
    for j in range(nc):
        name = CONTENT_FACTOR_LABELS[j] if nc == len(CONTENT_FACTOR_LABELS) else f"z_content[{j}]"
        out.append((name, gtc["z_content"][:, j], "content"))
    for v in ("z_style_v1", "z_style_v2"):
        if v in gtc:
            for j in range(gtc[v].shape[1]):
                out.append((f"{v}[{j}]", gtc[v][:, j], "style"))
    return out


def code_histograms(code_grid, K):
    # (N, n_voxels) int -> (N, K) row-normalised usage fractions
    N = code_grid.shape[0]
    H = np.zeros((N, K), dtype=np.float64)
    for i in range(N):
        H[i] = np.bincount(code_grid[i], minlength=K)
    return H / (H.sum(1, keepdims=True) + 1e-12)


def mi_matrix(H, factors):
    M = np.zeros((H.shape[1], len(factors)))
    for j, (_name, y, _grp) in enumerate(factors):
        M[:, j] = mutual_info_regression(H, y, discrete_features=False, random_state=0)
    return M


factors = scalar_factor_table()
content_cols = [j for j, f in enumerate(factors) if f[2] == "content"]
style_cols   = [j for j, f in enumerate(factors) if f[2] == "style"]

mi_by_level = {}
print("Discrete code -> factor analysis (per masked content level):")
for lvl in [l for l in content_style_levels if l in content_codes]:
    H = code_histograms(content_codes[lvl]["v0"], content_K[lvl])
    M = mi_matrix(H, factors)
    mi_by_level[lvl] = (H, M)
    c2c, c2s = M[:, content_cols].mean(), M[:, style_cols].mean()
    print(f"  L{lvl} CONTENT codes: mean MI to content={c2c:.4f}  to style={c2s:.4f}"
          f"  separation={c2c - c2s:+.4f}  | discrete DCI: D={_disent(M):.3f} C={_complete(M):.3f}")

# Same for style codes, if the model quantizes style.
for lvl in style_codes:
    H = code_histograms(style_codes[lvl]["v0"], style_K[lvl])
    M = mi_matrix(H, factors)
    s2c, s2s = M[:, content_cols].mean(), M[:, style_cols].mean()
    print(f"  L{lvl} STYLE   codes: mean MI to content={s2c:.4f}  to style={s2s:.4f}"
          f"  separation={s2s - s2c:+.4f}")
"""
)

code(
    """\
# Heatmap + "what does each code mean" for the finest masked content level.
def plot_mi_heatmap(M, factors, title, top=30):
    used = np.where(M.sum(1) > 0)[0]
    order = used[np.argsort(-M[used].sum(1))][:top]
    data = M[order]
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(factors)), max(4, 0.28 * len(order))))
    im = ax.imshow(data, vmin=0, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(factors))); ax.set_xticklabels([f[0] for f in factors], rotation=45, ha="right")
    ax.set_yticks(range(len(order)));   ax.set_yticklabels([f"code {c}" for c in order], fontsize=7)
    ax.set_title(title); plt.colorbar(im, ax=ax, label="mutual information")
    plt.tight_layout(); plt.show()


if mi_by_level:
    lvl = min(mi_by_level)
    H, M = mi_by_level[lvl]
    plot_mi_heatmap(M, factors, f"L{lvl} content code -> GT factor MI (top codes by total MI)")

    print(f"Code dictionary (L{lvl}, top used codes -> strongest-MI factor):")
    counts = np.bincount(content_codes[lvl]["v0"].reshape(-1), minlength=content_K[lvl])
    for c in np.argsort(-counts)[:8]:
        if counts[c] == 0:
            continue
        j = int(np.argmax(M[c]))
        print(f"  code {c:4d} (used {counts[c]:6d}x): {factors[j][0]:18s}  MI={M[c, j]:.3f}")
"""
)

md(
    """## 13. Cross-view code agreement (discrete identifiability)

Because content uses a **shared** codebook, the von Kügelgen / Yao content-invariance claim has an
*exact* discrete test: the same subject's content code map should be **identical** across T1 and T2.
We report per-voxel agreement against two baselines:

- **chance** = collision probability of the marginal code distribution (`sum_k p_k^2`) — high if the
  codebook collapsed to a few entries, so agreement alone can mislead;
- **shuffled** = pair subject *i*'s T1 codes with subject *i+1*'s T2 codes — agreement above this is
  the *subject-specific* invariance, not just a shared background pattern.

Content should beat both; style codes should sit near chance (style is modality-specific).
"""
)

code(
    """\
def chance_agreement(v0, v1, K):
    counts = np.bincount(np.concatenate([v0.reshape(-1), v1.reshape(-1)]), minlength=K).astype(np.float64)
    p = counts / counts.sum()
    return float((p ** 2).sum())


print("Per-voxel code agreement across paired views (v0=T1, v1=T2):")
if separate_content_cb:
    print("  WARNING: separate_content_codebooks=True -> views use different vocabularies; agreement is meaningless.")
for lvl in content_codes:
    v0, v1 = content_codes[lvl]["v0"], content_codes[lvl]["v1"]
    agree = float((v0 == v1).mean())
    ch    = chance_agreement(v0, v1, content_K[lvl])
    roll  = np.roll(np.arange(v0.shape[0]), 1)
    shuf  = float((v0 == v1[roll]).mean())
    print(f"  L{lvl} CONTENT: agree={agree:.3f}  chance={ch:.3f}  shuffled={shuf:.3f}"
          f"  -> subject-specific gain={agree - shuf:+.3f}")
for lvl in style_codes:
    if separate_style_cb:
        print(f"  L{lvl} STYLE  : (separate style codebooks -> skipping)")
        continue
    v0, v1 = style_codes[lvl]["v0"], style_codes[lvl]["v1"]
    agree = float((v0 == v1).mean())
    ch    = chance_agreement(v0, v1, style_K[lvl])
    print(f"  L{lvl} STYLE  : agree={agree:.3f}  chance={ch:.3f}  (expect LOW: style is modality-specific)")
"""
)

md(
    """## 14. Spatial code -> anatomy correspondence

Bag-of-codes throws away *where* each code fires, but the code map is a 3D grid and the strongest GT
signals are spatial. We recompute the tissue segmentation and lesion mask from the GT latents (via
the renderer), downsample them to each level's code-grid resolution, and ask how the content codes
partition anatomy:

- **NMI / AMI** between content code and tissue class (bg / csf / wm / gm) — a probe-free measure of
  whether the discrete partition aligns with anatomy;
- **per-code tissue purity** — does code *k* mean "white matter"?
- **lesion localisation** — does a single code pick out the focal WM lesion?
"""
)

code(
    """\
import torch.nn.functional as F
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

SPATIAL_N = min(N_d, 150)
TISSUE_NAMES = {0: "bg", 1: "csf/fissure", 2: "wm", 3: "gm"}
have_renderer = hasattr(ds, "_inner") and hasattr(getattr(ds, "_inner"), "renderer")


def resample_label(vol_np, grid):
    # nearest-neighbour downsample a (D,H,W) label/mask volume to `grid`
    t = torch.from_numpy(vol_np.astype(np.float32))[None, None]
    return F.interpolate(t, size=grid, mode="nearest")[0, 0]


if not content_codes:
    print("No content codes extracted; skipping spatial analysis.")
else:
    lvl = min(content_codes.keys())
    grid = code_grid_shape[lvl]
    codes_vox, tissue_vox, lesion_vox = [], [], []
    for i in range(SPATIAL_N):
        codes_vox.append(content_codes[lvl]["v0"][i])
        if have_renderer:
            try:
                tis, les = ds._inner.renderer.render_structure(
                    torch.from_numpy(gtc["z_content"][i]).float(),
                    torch.from_numpy(gtc["z_deformation"][i]).float(),
                    torch.from_numpy(gtc["z_fissure"][i]).float(),
                    torch.device("cpu"),
                )
                tissue_vox.append(resample_label(tis.cpu().numpy(), grid).round().long().numpy().reshape(-1))
                lesion_vox.append((resample_label(les.cpu().numpy(), grid) > 0.5).numpy().reshape(-1))
            except Exception as e:
                have_renderer, tissue_vox, lesion_vox = False, [], []
                print(f"  renderer recompute failed ({e}); trying brain_mask fallback.")
    codes_vox = np.concatenate(codes_vox)

    if tissue_vox:
        tissue_vox = np.concatenate(tissue_vox)
        lesion_vox = np.concatenate(lesion_vox)
        print(f"L{lvl} content code vs TISSUE class: "
              f"NMI={normalized_mutual_info_score(tissue_vox, codes_vox):.3f}  "
              f"AMI={adjusted_mutual_info_score(tissue_vox, codes_vox):.3f}")
        print("Per-code tissue purity (top used codes):")
        ids, cnts = np.unique(codes_vox, return_counts=True)
        for c in ids[np.argsort(-cnts)][:10]:
            sel = codes_vox == c
            cls, cc = np.unique(tissue_vox[sel], return_counts=True)
            print(f"  code {c:4d}: n={sel.sum():7d}  dominant={TISSUE_NAMES.get(int(cls[cc.argmax()])):11s}"
                  f"  purity={cc.max() / cc.sum():.2f}")
        if lesion_vox.sum() > 0:
            best = None
            for c in ids:
                pred = codes_vox == c
                tp = int((pred & lesion_vox).sum())
                if tp == 0:
                    continue
                prec, rec = tp / pred.sum(), tp / lesion_vox.sum()
                f1 = 2 * prec * rec / (prec + rec)
                if best is None or f1 > best[-1]:
                    best = (c, prec, rec, f1)
            if best:
                print(f"Best lesion-aligned code: {best[0]}  precision={best[1]:.2f}"
                      f"  recall={best[2]:.2f}  F1={best[3]:.2f}")
        else:
            print("No lesion voxels survive downsampling at this level (lesions may be sub-voxel).")
    elif fg_masks is not None:
        fg_vox = np.concatenate([
            (resample_label(np.squeeze(fg_masks[i]), grid) > 0.5).numpy().reshape(-1)
            for i in range(SPATIAL_N)
        ])
        print(f"L{lvl} content code vs FOREGROUND (brain_mask): "
              f"NMI={normalized_mutual_info_score(fg_vox, codes_vox):.3f}")
    else:
        print("No anatomy GT available (no renderer, no brain_mask); skipping spatial analysis.")
"""
)

md(
    """## 15. Discrete summary

The discrete counterparts of the Section 9 headline numbers — all read off the codes the decoder
actually consumes, most of them probe-free.
"""
)

code(
    """\
print("=== Discrete codebook summary ===")
if mi_by_level:
    lvl = min(mi_by_level)
    _H, M = mi_by_level[lvl]
    c2c, c2s = M[:, content_cols].mean(), M[:, style_cols].mean()
    print(f"Content/style MI separation (L{lvl}, probe-free) : {c2c - c2s:+.4f}  (content={c2c:.4f}, style={c2s:.4f})")
if content_codes:
    lvl = min(content_codes)
    v0, v1 = content_codes[lvl]["v0"], content_codes[lvl]["v1"]
    roll = np.roll(np.arange(v0.shape[0]), 1)
    print(f"Cross-view content-code agreement (L{lvl})        : {float((v0 == v1).mean()):.3f}"
          f"  (subject-specific gain {float((v0 == v1).mean()) - float((v0 == v1[roll]).mean()):+.3f})")
    ppl, used = perplexity(np.concatenate([v0, v1], 0), content_K[lvl])
    print(f"Content codebook utilisation (L{lvl})             : {used}/{content_K[lvl]} used, perplexity {ppl:.1f}")
print()
print("Reading guide:")
print("  separation > 0           -> content codes specialise on anatomy, not modality")
print("  agreement >> shuffled    -> content codes are subject-specific and view-invariant")
print("  agreement ~= chance with low perplexity -> codebook collapse, not invariance")
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
