import json

with open('/home/ng24/projects/multiview-crl/eval/view_latents.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Output changes
        # UMAP Visualization (around line 1059)
        source = source.replace(
            "if all_content_indices is not None and len(all_style_indices) > 0:",
            "if 0 in all_content_indices and len(all_style_indices.get(0, [])) > 0:"
        ).replace(
            "level0_feats[:, all_content_indices]",
            "level0_feats[:, all_content_indices[0]]"
        ).replace(
            "level0_feats[:, all_style_indices]",
            "level0_feats[:, all_style_indices[0]]"
        ).replace(
            "len(all_content_indices)",
            "len(all_content_indices.get(0, []))"
        )
        
        # Codebook Analysis (around line 1203 / 1306)
        source = source.replace(
            "if all_content_indices is not None:",
            "if 0 in all_content_indices:"
        ).replace(
            "all_content_indices[:8]",
            "all_content_indices[0][:8]" if "all_content_indices[:8]" in source else "all_content_indices[:8]" # handled earlier, but just in case
        )
        
        # content_idx assignments
        source = source.replace(
            "content_idx = np.array(all_content_indices) if all_content_indices is not None else np.arange(hidden_channels)",
            "content_idx = np.array(all_content_indices[0]) if 0 in all_content_indices else np.arange(hidden_channels)"
        ).replace(
            "content_idx = list(all_content_indices) if all_content_indices is not None else []",
            "content_idx = list(all_content_indices[0]) if 0 in all_content_indices else []"
        ).replace(
            "content_idx_v1 = list(all_content_indices_v1) if all_content_indices_v1 is not None else content_idx",
            "content_idx_v1 = list(all_content_indices_v1[0]) if 0 in all_content_indices_v1 else content_idx"
        ).replace(
            "content_idx = list(all_content_indices) if all_content_indices is not None else list(range(hidden_channels))",
            "content_idx = list(all_content_indices[0]) if 0 in all_content_indices else list(range(hidden_channels))"
        )
        
        # Style index logic
        source = source.replace(
            "style_idx = np.array(all_style_indices) if len(all_style_indices) > 0 else None",
            "style_idx = np.array(all_style_indices.get(0, [])) if len(all_style_indices.get(0, [])) > 0 else None"
        ).replace(
            "len(all_style_indices)",
            "len(all_style_indices.get(0, []))"
        )

        # Dictionary saves at the end
        source = source.replace(
            "np.array(all_content_indices) if all_content_indices else",
            "np.array(all_content_indices.get(0, [])) if 0 in all_content_indices else"
        ).replace(
            "np.array(all_style_indices) if all_style_indices else",
            "np.array(all_style_indices.get(0, [])) if 0 in all_style_indices else"
        )

        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source'] and cell['source'][-1].endswith('\n') and not source.endswith('\n'):
            cell['source'][-1] = cell['source'][-1][:-1]

with open('/home/ng24/projects/multiview-crl/eval/view_latents.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
