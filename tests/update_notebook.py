import json
import re

with open('/home/ng24/projects/multiview-crl/eval/view_latents.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Update initialization
        source = source.replace(
            "all_content_indices = None  # view-0 (T1) content indices (deterministic in eval mode)\nall_content_indices_v1 = None",
            "all_content_indices = {}  # dict mapping level -> view-0 (T1) content indices\nall_content_indices_v1 = {}"
        )
        
        # 2. Update model call
        source = source.replace(
            "_, _, enc_features, est_content_idx, _, _, _ = vqvae_model(",
            "_, _, enc_features, est_content_idx, _, _, soft_masks, *_ = vqvae_model("
        )
        source = source.replace(
            "_, _, enc_features, est_content_idx, _, _, soft_masks, *extra = vqvae_model(",
            "_, _, enc_features, est_content_idx, _, _, soft_masks, *_ = vqvae_model("
        )

        
        # 3. Update indices extraction
        ext_old = """        if est_content_idx is not None:
            if modality == "T1" and all_content_indices is None:
                all_content_indices = est_content_idx[0]
            elif modality == "T2" and all_content_indices_v1 is None:
                if _has_per_view:
                    all_content_indices_v1 = est_content_idx[0]  # T2's own indices
                # else: shared mask, v1 indices = v0 indices (no need to store)"""
        ext_new = """        for lvl, mask in soft_masks.items():
            if isinstance(mask, tuple):
                mask_v0, mask_v1 = mask
                if modality == "T1" and lvl not in all_content_indices:
                    all_content_indices[lvl] = torch.where(mask_v0.bool())[-1].tolist()
                elif modality == "T2" and lvl not in all_content_indices_v1:
                    all_content_indices_v1[lvl] = torch.where(mask_v1.bool())[-1].tolist()
            else:
                if modality == "T1" and lvl not in all_content_indices:
                    all_content_indices[lvl] = torch.where(mask.bool())[-1].tolist()"""
        source = source.replace(ext_old, ext_new)
        
        # 4. Update printing indices
        p_old = """all_style_indices = [i for i in range(hidden_channels) if i not in set(all_content_indices or [])]
if _has_per_view and all_content_indices_v1 is not None:
    all_style_indices_v1 = [i for i in range(hidden_channels) if i not in set(all_content_indices_v1)]
else:
    all_style_indices_v1 = all_style_indices

if all_content_indices is not None:
    print(f"\\n  content indices v0 ({len(all_content_indices)} ch): {all_content_indices[:8]}...")
    print(f"  style indices   v0 ({len(all_style_indices)} ch):   {all_style_indices[:8]}...")
    if _has_per_view and all_content_indices_v1 is not None:
        print(f"  content indices v1 ({len(all_content_indices_v1)} ch): {all_content_indices_v1[:8]}...")
        print(f"  style indices   v1 ({len(all_style_indices_v1)} ch):   {all_style_indices_v1[:8]}...")"""
        p_new = """all_style_indices = {}
all_style_indices_v1 = {}
for lvl in all_content_indices.keys():
    all_style_indices[lvl] = [i for i in range(hidden_channels) if i not in set(all_content_indices.get(lvl, []))]
    if _has_per_view and lvl in all_content_indices_v1:
        all_style_indices_v1[lvl] = [i for i in range(hidden_channels) if i not in set(all_content_indices_v1[lvl])]
    else:
        all_style_indices_v1[lvl] = all_style_indices[lvl]

if len(all_content_indices) > 0:
    for lvl in all_content_indices.keys():
        print(f"\\n--- Level {lvl} ---")
        print(f"  content indices v0 ({len(all_content_indices[lvl])} ch): {all_content_indices[lvl][:8]}...")
        print(f"  style indices   v0 ({len(all_style_indices[lvl])} ch):   {all_style_indices[lvl][:8]}...")
        if _has_per_view and lvl in all_content_indices_v1:
            print(f"  content indices v1 ({len(all_content_indices_v1[lvl])} ch): {all_content_indices_v1[lvl][:8]}...")
            print(f"  style indices   v1 ({len(all_style_indices_v1[lvl])} ch):   {all_style_indices_v1[lvl][:8]}...")"""
        source = source.replace(p_old, p_new)

        # 5. Feature stats section
        feat_stat_old = """# Extra breakdown for level 0: content vs style channels
if all_content_indices is not None:
    level0_feats = all_features["level_0"]
    print(f"\\n--- Level 0 channel breakdown (hidden_channels={hidden_channels}) ---")
    content_f = level0_feats[:, all_content_indices]
    style_f = level0_feats[:, all_style_indices] if all_style_indices else None
    ct1 = content_f[t1_mask]
    ct2 = content_f[t2_mask]
    print(
        f"  Content  ({len(all_content_indices)} ch)  — T1-T2 paired dist: {np.linalg.norm(ct1-ct2,axis=1).mean():.4f}"
    )
    if style_f is not None and len(all_style_indices):
        st1 = style_f[t1_mask]
        st2 = style_f[t2_mask]
        print(
            f"  Style    ({len(all_style_indices)} ch)  — T1-T2 paired dist: {np.linalg.norm(st1-st2,axis=1).mean():.4f}"
        )
    print("  (Content dist should be smaller than style dist if disentanglement is working)")"""
        feat_stat_new = """# Extra breakdown for content vs style channels
if len(all_content_indices) > 0:
    for lvl in all_content_indices.keys():
        lvl_feats = all_features[f"level_{lvl}"]
        print(f"\\n--- Level {lvl} channel breakdown (hidden_channels={hidden_channels}) ---")
        content_f = lvl_feats[:, all_content_indices[lvl]]
        style_f = lvl_feats[:, all_style_indices[lvl]] if len(all_style_indices[lvl]) > 0 else None
        ct1 = content_f[t1_mask]
        ct2 = content_f[t2_mask]
        print(
            f"  Content  ({len(all_content_indices[lvl])} ch)  — T1-T2 paired dist: {np.linalg.norm(ct1-ct2,axis=1).mean():.4f}"
        )
        if style_f is not None:
            st1 = style_f[t1_mask]
            st2 = style_f[t2_mask]
            print(
                f"  Style    ({len(all_style_indices[lvl])} ch)  — T1-T2 paired dist: {np.linalg.norm(st1-st2,axis=1).mean():.4f}"
            )
        print("  (Content dist should be smaller than style dist if disentanglement is working)")"""
        source = source.replace(feat_stat_old, feat_stat_new)
        
        # 6. PCA
        pca_old = """    # For level 0, show only the content dims — these are what the contrastive
    # loss acts on.  Plotting all embed_dim at level 0 mixes in style dims,
    # which encode T1/T2 contrast differences and will dominate the first PC.
    # The full content+style breakdown for level 0 is in Section 7.
    if level_idx == 0 and all_content_indices is not None:
        level0_feats = all_features["level_0"]
        # With per-view masks, T1 and T2 have different content channels.
        # Select the correct content channels for each modality, then
        # concatenate back (both now have k content dims).
        if _has_per_view:
            f_t1 = level0_feats[t1_mask][:, all_content_indices]
            f_t2 = level0_feats[t2_mask][:, all_content_indices_v1]
            features = np.empty((len(level0_feats), f_t1.shape[1]))
            features[t1_mask] = f_t1
            features[t2_mask] = f_t2
            level_label = f"Level 0 — content ({f_t1.shape[1]} dims, per-view mask)"
        else:
            features = level0_feats[:, all_content_indices]
            level_label = f"Level 0 — content ({len(all_content_indices)} dims)"
    else:
        features = all_features[f"level_{level_idx}"]
        level_label = f"Level {level_idx}\""""
        pca_new = """    if level_idx in all_content_indices:
        lvl_feats = all_features[f"level_{level_idx}"]
        if _has_per_view and level_idx in all_content_indices_v1:
            f_t1 = lvl_feats[t1_mask][:, all_content_indices[level_idx]]
            f_t2 = lvl_feats[t2_mask][:, all_content_indices_v1[level_idx]]
            features = np.empty((len(lvl_feats), f_t1.shape[1]))
            features[t1_mask] = f_t1
            features[t2_mask] = f_t2
            level_label = f"Level {level_idx} — content ({f_t1.shape[1]} dims, per-view mask)"
        else:
            features = lvl_feats[:, all_content_indices[level_idx]]
            level_label = f"Level {level_idx} — content ({len(all_content_indices[level_idx])} dims)"
    else:
        features = all_features[f"level_{level_idx}"]
        level_label = f"Level {level_idx}\""""
        source = source.replace(pca_old, pca_new)

        # 7. t-SNE
        tsne_old = """    # Apply the same content-only filtering as the PCA section for level 0,
    # including per-view mask handling so T2 uses its own content indices.
    if level_idx == 0 and all_content_indices is not None:
        level0_feats = all_features["level_0"]
        if _has_per_view and all_content_indices_v1 is not None:
            f_t1 = level0_feats[t1_mask][:, all_content_indices]
            f_t2 = level0_feats[t2_mask][:, all_content_indices_v1]
            features = np.empty((len(level0_feats), f_t1.shape[1]))
            features[t1_mask] = f_t1
            features[t2_mask] = f_t2
            level_label = f"Level 0 — content ({f_t1.shape[1]} dims, per-view mask)"
        else:
            features = level0_feats[:, all_content_indices]
            level_label = f"Level 0 — content ({len(all_content_indices)} dims)"
    else:
        features = all_features[f"level_{level_idx}"]
        level_label = f"Level {level_idx}\""""
        source = source.replace(tsne_old, pca_new.replace("Level 0 ", f"Level {level_idx} ").replace("Level 0", f"Level {level_idx}"))

        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source'] and cell['source'][-1].endswith('\n') and not source.endswith('\n'):
            cell['source'][-1] = cell['source'][-1][:-1]

with open('/home/ng24/projects/multiview-crl/eval/view_latents.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
