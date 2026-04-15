"""Cross-reconstruction evaluation for content/style separation quality.

Measures how well content and style are disentangled by:
1. Content invariance: content representations of the same subject across T1/T2
   should be similar (style should not leak into content).
2. Style invariance: style representations of different subjects within the same
   modality should be similar (content should not leak into style).
3. Linear probe leakage: a linear probe on content should NOT predict modality,
   and a linear probe on style should NOT predict subject identity.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def _collect_representations(vqvae_model, dataloader, args, device, max_batches=200):
    """Encode all samples and collect content/style representations per view."""
    was_training = vqvae_model.training
    vqvae_model.eval()
    try:
        # Unwrap DataParallel / MoCo
        raw = vqvae_model
        if hasattr(raw, "online"):
            raw = raw.online
        if hasattr(raw, "module"):
            raw = raw.module

        content_v0_list, content_v1_list = [], []
        style_v0_list, style_v1_list = [], []

        for batch_idx, data in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            samples = data["image"]
            n_views = len(samples)
            if n_views < 2:
                continue
            images = torch.cat(samples, 0).to(device, non_blocking=True)
            B = samples[0].shape[0]

            (
                _recon,
                _diffs,
                encoder_features,
                estimated_content_indices,
                _decoder_outputs,
                _id_outputs,
                soft_content_masks,
                _style_id_outputs,
            ) = vqvae_model(images, return_recon=False, pool_only=True, n_views=2)

            # Use level 0 (finest) features for evaluation
            hz = encoder_features[0]  # (2B, hidden_channels)
            hz_v0, hz_v1 = hz[:B], hz[B:]

            # Determine content/style channel split
            if estimated_content_indices is not None:
                if isinstance(estimated_content_indices[0], list):
                    c_idx_v0 = estimated_content_indices[0]
                    c_idx_v1 = estimated_content_indices[1] if len(estimated_content_indices) > 1 else c_idx_v0
                else:
                    c_idx_v0 = c_idx_v1 = estimated_content_indices[0]
            else:
                c_idx_v0 = c_idx_v1 = list(range(len(args.content_indices[0])))

            all_channels = set(range(hz.shape[1]))
            s_idx_v0 = sorted(all_channels - set(c_idx_v0))
            s_idx_v1 = sorted(all_channels - set(c_idx_v1))

            content_v0_list.append(hz_v0[:, c_idx_v0].cpu().numpy())
            content_v1_list.append(hz_v1[:, c_idx_v1].cpu().numpy())
            style_v0_list.append(hz_v0[:, s_idx_v0].cpu().numpy())
            style_v1_list.append(hz_v1[:, s_idx_v1].cpu().numpy())

        return {
            "content_v0": np.concatenate(content_v0_list, axis=0),
            "content_v1": np.concatenate(content_v1_list, axis=0),
            "style_v0": np.concatenate(style_v0_list, axis=0),
            "style_v1": np.concatenate(style_v1_list, axis=0),
        }
    finally:
        vqvae_model.train(was_training)


def _cosine_sim(a, b):
    """Row-wise cosine similarity between two (N, D) arrays."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a_norm * b_norm).sum(axis=1)


def evaluate_content_style_separation(vqvae_model, dataloader, args, device, max_batches=200):
    """Run all content/style separation metrics.

    Args:
        vqvae_model: The VQ-VAE model (possibly wrapped in DataParallel/MoCo).
        dataloader: DataLoader yielding paired views.
        args: Parsed argument namespace.
        device: Torch device.
        max_batches: Maximum number of batches to evaluate.

    Returns:
        dict of metric_name → float.
    """
    reps = _collect_representations(vqvae_model, dataloader, args, device, max_batches)
    metrics = {}
    N = reps["content_v0"].shape[0]

    # --- 1. Content invariance across views ---
    # Same subject's content should be similar across T1 (v0) and T2 (v1).
    content_sim = _cosine_sim(reps["content_v0"], reps["content_v1"])
    metrics["content/cross_view_cosine_mean"] = float(np.mean(content_sim))
    metrics["content/cross_view_cosine_std"] = float(np.std(content_sim))

    # --- 2. Style invariance within modality ---
    # Different subjects' style within T1 should be similar (style = modality info, not subject).
    # Compare each sample to a random other sample's style within same view.
    perm = np.random.permutation(N)
    style_intra_sim_v0 = _cosine_sim(reps["style_v0"], reps["style_v0"][perm])
    metrics["style/intra_view_cosine_mean_v0"] = float(np.mean(style_intra_sim_v0))

    # Style should differ across modalities (T1 vs T2).
    style_cross_sim = _cosine_sim(reps["style_v0"], reps["style_v1"])
    metrics["style/cross_view_cosine_mean"] = float(np.mean(style_cross_sim))

    # --- 3. Linear probe: modality from content (should be chance = 0.5) ---
    # If content is modality-invariant, a classifier should not be able to tell
    # whether content came from T1 or T2.
    content_all = np.concatenate([reps["content_v0"], reps["content_v1"]], axis=0)
    modality_labels = np.array([0] * N + [1] * N)

    scaler = StandardScaler()
    content_scaled = scaler.fit_transform(content_all)
    try:
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        scores = cross_val_score(clf, content_scaled, modality_labels, cv=5, scoring="accuracy")
        content_modality_acc = float(np.mean(scores))
    except Exception:
        content_modality_acc = 0.5
    metrics["content/modality_probe_acc"] = content_modality_acc
    # Higher = better separation (1.0 means content carries no modality info)
    metrics["content/modality_invariance"] = 1.0 - abs(content_modality_acc - 0.5) * 2

    # --- 4. Linear probe: subject identity from style (should be chance) ---
    # If style only encodes modality, a regressor should not predict which subject
    # the style came from.
    subject_ids = np.arange(N)
    style_scaled_v0 = StandardScaler().fit_transform(reps["style_v0"])
    try:
        ridge = Ridge(alpha=1.0)
        # Use subject ordering as a proxy for identity (regress index).
        # With random subject order, R2 should be ~0 if style carries no subject info.
        scores = cross_val_score(ridge, style_scaled_v0, subject_ids, cv=5, scoring="r2")
        style_subject_r2 = float(np.mean(scores))
    except Exception:
        style_subject_r2 = 0.0
    metrics["style/subject_probe_r2"] = style_subject_r2
    # Higher = better separation (1.0 means style carries no subject info)
    metrics["style/subject_invariance"] = 1.0 - max(0.0, style_subject_r2)

    # --- 5. Linear probe: modality from style (should be high ~1.0) ---
    # Style SHOULD encode modality. A classifier on style should easily tell T1 from T2.
    style_all = np.concatenate([reps["style_v0"], reps["style_v1"]], axis=0)
    style_all_scaled = StandardScaler().fit_transform(style_all)
    try:
        clf2 = LogisticRegression(max_iter=1000, solver="lbfgs")
        scores2 = cross_val_score(clf2, style_all_scaled, modality_labels, cv=5, scoring="accuracy")
        style_modality_acc = float(np.mean(scores2))
    except Exception:
        style_modality_acc = 0.5
    metrics["style/modality_probe_acc"] = style_modality_acc

    # --- 6. Composite separation score (optimization target for sweep) ---
    # Combines: content should not predict modality + style should not predict subject
    metrics["separation_score"] = (metrics["content/modality_invariance"] + metrics["style/subject_invariance"]) / 2.0

    return metrics
