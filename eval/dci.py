# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Disentanglement, Completeness and Informativeness.

Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""
from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import scipy
import torch
from six.moves import range
from sklearn import ensemble
from torch.utils.data import DataLoader

import utils.utils as utils

logger = logging.getLogger(__name__)


def compute_dci(
    ground_truth_data,
    representation_function,
    random_state,
    num_train=10000,
    num_test=100,
    batch_size=16,
    factor_types=None,
):
    """Computes the DCI scores according to Sec 2.

    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      random_state: Numpy random state used for randomness.
      artifact_dir: Optional path to directory where artifacts can be saved.
      num_train: Number of points used for training.
      num_test: Number of points used for testing.
      batch_size: Batch size for sampling.

    Returns:
      Dictionary with average disentanglement score, completeness and
        informativeness (train and test).
    """
    # mus_train are of shape [num_codes, num_train], while ys_train are of shape
    # [num_factors, num_train].
    mus_train, ys_train = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, num_train, random_state, batch_size
    )
    if factor_types is None:
        factor_types = ["discrete"] * mus_train.shape[0]  # take discrete factors by default
    assert mus_train.shape[1] == num_train
    assert ys_train.shape[1] == num_train
    mus_test, ys_test = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, num_test, random_state, batch_size
    )
    scores, _ = _compute_dci(mus_train, ys_train, mus_test, ys_test, factor_types)
    return scores


def _compute_dci(mus_train, ys_train, mus_test, ys_test, factor_types):
    """Computes score based on both training and testing codes and factors.

    Returns:
        scores: dict with scalar summary metrics.
        detail: dict with per-factor / per-code arrays and the importance matrix.
    """
    scores = {}
    importance_matrix, train_err, test_err, pf_train, pf_test = compute_importance_gbt(
        mus_train, ys_train, mus_test, ys_test, factor_types
    )
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    scores["informativeness_train"] = train_err
    scores["informativeness_test"] = test_err
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    detail = {
        "importance_matrix": importance_matrix,
        "per_factor_train": pf_train,
        "per_factor_test": pf_test,
        "per_code_disentanglement": disentanglement_per_code(importance_matrix),
        "per_factor_completeness": completeness_per_factor(importance_matrix),
    }
    return scores, detail


def compute_dci_on_fixed_data(observations, labels, representation_function, train_percentage=0.8, batch_size=100):
    """Computes the DCI scores on the fixed set of observations and labels.

    Args:
      observations: Observations on which to compute the score. Observations have
        shape (num_observations, 64, 64, num_channels).
      labels: Observed factors of variations.
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      train_percentage: Percentage of observations used for training.
      batch_size: Batch size used to compute the representation.

    Returns:
      DCI score.
    """
    mus = utils.obtain_representation(observations, representation_function, batch_size)
    assert labels.shape[1] == observations.shape[0], "Wrong labels shape."
    assert mus.shape[1] == observations.shape[0], "Wrong representation shape."
    mus_train, mus_test = utils.split_train_test(mus, train_percentage)
    ys_train, ys_test = utils.split_train_test(labels, train_percentage)
    return _compute_dci(mus_train, ys_train, mus_test, ys_test)


def compute_importance_gbt(x_train, y_train, x_test, y_test, factor_types):
    """Compute importance based on gradient boosted trees."""
    num_factors = y_train.shape[0]
    assert num_factors == len(factor_types)
    num_codes = x_train.shape[0]
    # sklearn's default max_features=None rescans every code at every split, which
    # is the dominant cost once patch pooling pushes num_codes into the tens of
    # thousands. Subsample features only in that regime, so gap/stats pooling (tens
    # of codes) keeps its historical numbers bit-for-bit.
    _max_features = "sqrt" if num_codes > 2000 else None
    importance_matrix = np.zeros(shape=[num_codes, num_factors], dtype=np.float64)
    train_loss = []
    test_loss = []
    for i, ftype in zip(range(num_factors), factor_types):
        if ftype == "discrete":
            model = ensemble.GradientBoostingClassifier(max_features=_max_features)
            model.fit(x_train.T, y_train[i, :])
            importance_matrix[:, i] = np.abs(model.feature_importances_)
            train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
            test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
        else:
            model = ensemble.GradientBoostingRegressor(max_features=_max_features)
            model.fit(x_train.T, y_train[i, :])
            importance_matrix[:, i] = np.abs(model.feature_importances_)
            train_loss.append(model.score(x_train.T, y_train[i, :]))
            test_loss.append(model.score(x_test.T, y_test[i, :]))
    return (
        importance_matrix,
        np.mean(train_loss),
        np.mean(test_loss),
        np.array(train_loss),
        np.array(test_loss),
    )


def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1.0 - scipy.stats.entropy(importance_matrix.T + 1e-11, base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.0:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code * code_importance)


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1.0 - scipy.stats.entropy(importance_matrix + 1e-11, base=importance_matrix.shape[0])


def completeness(importance_matrix):
    """ "Compute completeness of the representation."""
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.0:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)


# ---------------------------------------------------------------------------
# Synthetic-data DCI
# ---------------------------------------------------------------------------

CONTENT_FACTOR_NAMES = [
    "brain_size",
    "ventricle_size",
    "lesion_x",
    "lesion_y",
    "lesion_z",
    "cortical_thickness",
    "temporal_atrophy",
    "lr_asymmetry",
    "sulcal_widening",
]

STYLE_FACTOR_NAMES = ["gain", "bias", "noise_sigma"]


def _parse_content_indices(content_idx_raw):
    """Normalise the various formats content_idx can arrive in to a plain list[int]."""
    if content_idx_raw is None:
        return None
    if isinstance(content_idx_raw, (list, tuple)) and len(content_idx_raw) > 0:
        first = content_idx_raw[0]
        if isinstance(first, (list, tuple)):
            return list(first)
        if isinstance(first, torch.Tensor):
            return first.cpu().tolist()
        return list(content_idx_raw)
    if isinstance(content_idx_raw, torch.Tensor):
        return content_idx_raw.cpu().tolist()
    return None


def _split_content_style(features_np, content_indices):
    """Split a (B, D) array into content and style columns.

    Returns (content, style).  style is None when there is no split.
    """
    if content_indices is not None and len(content_indices) > 0:
        all_channels = set(range(features_np.shape[1]))
        style_indices = sorted(all_channels - set(content_indices))
        content = features_np[:, content_indices]
        style = features_np[:, style_indices] if style_indices else None
        return content, style
    return features_np, None


def _pool_and_split_view(feat_view, content_indices, pooling, use_patch_grid, use_stats):
    """Pool one view's spatial map to a feature vector and split content/style.

    Factored out of the extraction loop so view 1 and view 2 are processed
    identically (view 2 is needed for the content->view leakage probe).
    Returns ``(content, style)`` numpy arrays; ``style`` is ``None`` when there
    is no content/style split at this level.
    """
    if use_stats and feat_view.dim() == 5:
        B, C = feat_view.shape[:2]
        flat = feat_view.view(B, C, -1)
        feat_view = torch.cat(
            [flat.mean(dim=2), flat.std(dim=2), flat.amax(dim=2), flat.amin(dim=2)],
            dim=1,
        )
    elif use_patch_grid and feat_view.dim() == 3:
        # (B, C, P) → (B, P*C) patch-major so the expansion loop below works.
        feat_view = feat_view.permute(0, 2, 1).flatten(1)
    elif feat_view.dim() == 5:
        feat_view = feat_view.mean(dim=[2, 3, 4])

    feat_np = feat_view.cpu().numpy()
    if use_patch_grid and content_indices is not None:
        n_patches = int(np.prod(pooling))
        n_channels = feat_np.shape[1] // n_patches
        expanded_content, expanded_style = [], []
        style_ch = sorted(set(range(n_channels)) - set(content_indices))
        for p in range(n_patches):
            offset = p * n_channels
            expanded_content.extend(offset + c for c in content_indices)
            expanded_style.extend(offset + c for c in style_ch)
        content = feat_np[:, expanded_content]
        style = feat_np[:, expanded_style] if expanded_style else None
    elif use_stats and content_indices is not None:
        n_stats = 4
        n_channels = feat_np.shape[1] // n_stats
        expanded_content, expanded_style = [], []
        style_ch = sorted(set(range(n_channels)) - set(content_indices))
        for s in range(n_stats):
            offset = s * n_channels
            expanded_content.extend(offset + c for c in content_indices)
            expanded_style.extend(offset + c for c in style_ch)
        content = feat_np[:, expanded_content]
        style = feat_np[:, expanded_style] if expanded_style else None
    else:
        content, style = _split_content_style(feat_np, content_indices)
    return content, style


def _extract_synthetic_representations(encoder, dataset, device, batch_size=32, num_workers=0, pooling="gap"):
    """Run the encoder over a synthetic dataset and collect per-level representations + GT factors.

    A single forward pass extracts features from every encoder level.  Each level's
    content/style split uses that level's own ``soft_content_masks`` entry (from the
    Gumbel mask), so levels without a mask have no split (all channels → content).

    Args:
        encoder: VQVAE (or compatible) model.
        dataset: SyntheticBrainDataset.
        device: torch device.
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.
        pooling: How to reduce spatial maps to feature vectors.
            ``"gap"``   — global average pool → (B, C).
            ``(D,H,W)`` — patch-grid pool → (B, C*D*H*W).  E.g. ``(2,2,2)``
                          gives 8 patches per channel.
            ``"stats"`` — per-channel (mean, std, max, min) → (B, C*4).

    Returns:
        level_data: ``{level_idx: (content_repr, style_repr, content_v2, style_v2, factor_info)}``
            where the ``_v2`` arrays are the second view's features (``None`` when
            only one view is present), used for the content->view leakage probe.
        gt_content, gt_style_v1, gt_style_v2: shared GT arrays ``(N, n_factors)``
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    use_patch_grid = isinstance(pooling, (list, tuple))
    use_stats = pooling == "stats"

    per_level = {}
    all_gt_content = []
    all_gt_style_v1 = []
    all_gt_style_v2 = []

    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"]
            x = torch.cat(imgs, dim=0).to(device)

            if use_patch_grid:
                enc_out = encoder(x, pool_only=True, n_views=len(imgs), patch_grid=pooling)
            elif use_stats:
                enc_out = encoder(x, pool_only=False, return_recon=False, n_views=len(imgs))
            else:
                enc_out = encoder(x, pool_only=True, n_views=len(imgs))

            if isinstance(enc_out, tuple):
                encoder_features = enc_out[2]
                soft_masks = enc_out[6] if len(enc_out) > 6 else {}
            else:
                encoder_features = [enc_out]
                soft_masks = {}

            n_views = len(imgs)

            for lvl_idx, feat in enumerate(encoder_features):
                B_per_view = feat.shape[0] // n_views

                # Content/style channel split from this level's Gumbel mask
                # (shared across views — symmetric masks use the view-0 indices).
                if isinstance(soft_masks, dict) and lvl_idx in soft_masks:
                    mask = soft_masks[lvl_idx]
                    if isinstance(mask, tuple):
                        content_idx_raw = torch.where(mask[0].bool())[-1]
                    else:
                        content_idx_raw = torch.where(mask.bool())[-1]
                else:
                    content_idx_raw = None
                content_indices = _parse_content_indices(content_idx_raw)

                if lvl_idx not in per_level:
                    per_level[lvl_idx] = {"content": [], "style": [], "content_v2": [], "style_v2": []}

                # View 1 (primary) — unchanged behaviour.
                content, style = _pool_and_split_view(
                    feat[:B_per_view], content_indices, pooling, use_patch_grid, use_stats
                )
                per_level[lvl_idx]["content"].append(content)
                if style is not None:
                    per_level[lvl_idx]["style"].append(style)

                # View 2 — needed for the content->view leakage probe. Same
                # content_indices (symmetric mask), processed identically.
                if n_views >= 2:
                    content2, style2 = _pool_and_split_view(
                        feat[B_per_view : 2 * B_per_view], content_indices, pooling, use_patch_grid, use_stats
                    )
                    per_level[lvl_idx]["content_v2"].append(content2)
                    if style2 is not None:
                        per_level[lvl_idx]["style_v2"].append(style2)

            gt = batch["gt_latents"]
            all_gt_content.append(gt["z_content"].numpy())
            all_gt_style_v1.append(gt["z_style_v1"].numpy())
            all_gt_style_v2.append(gt["z_style_v2"].numpy())

    gt_content = np.concatenate(all_gt_content, axis=0)
    gt_style_v1 = np.concatenate(all_gt_style_v1, axis=0)
    gt_style_v2 = np.concatenate(all_gt_style_v2, axis=0)

    n_c = gt_content.shape[1]
    n_s = gt_style_v1.shape[1]
    pooling_label = f"patch{list(pooling)}" if use_patch_grid else str(pooling)

    level_data = {}
    for lvl, data in per_level.items():
        content_repr = np.concatenate(data["content"], axis=0)
        style_repr = np.concatenate(data["style"], axis=0) if data["style"] else None
        content_v2 = np.concatenate(data["content_v2"], axis=0) if data["content_v2"] else None
        style_v2 = np.concatenate(data["style_v2"], axis=0) if data["style_v2"] else None
        level_data[lvl] = (
            content_repr,
            style_repr,
            content_v2,
            style_v2,
            {
                "content_names": CONTENT_FACTOR_NAMES[:n_c],
                "style_names": STYLE_FACTOR_NAMES[:n_s],
                "n_content": n_c,
                "n_style": n_s,
                "n_content_channels": content_repr.shape[1],
                "n_style_channels": style_repr.shape[1] if style_repr is not None else 0,
                "has_split": style_repr is not None,
                "pooling": pooling_label,
                "level": lvl,
            },
        )
    return level_data, gt_content, gt_style_v1, gt_style_v2


def _null_permuted_dci(repr_arr, factor_arr, split, factor_types, n_null, rng):
    """DCI scores under row-permuted factors — the structural floor for a block.

    Permuting the factor rows destroys the representation↔factor correspondence,
    so whatever D/C/I survives is what the block's *shape* (n_codes × n_factors)
    yields from noise alone.  This is the quantity to subtract before comparing a
    wide block (e.g. 46 content channels) against a narrow one (2 style channels):
    real minus null is the part that is not a shape artifact.

    Averaged over ``n_null`` permutations.  Returns means for
    ``{disentanglement, completeness, informativeness_train,
    informativeness_test}`` plus matching ``*_std`` keys.
    """
    keys = ["disentanglement", "completeness", "informativeness_train", "informativeness_test"]
    acc = {k: [] for k in keys}
    n = repr_arr.shape[0]
    mus_train = repr_arr[:split].T
    mus_test = repr_arr[split:].T
    for _ in range(n_null):
        f = factor_arr[rng.permutation(n)]
        try:
            scores, _ = _compute_dci(mus_train, f[:split].T, mus_test, f[split:].T, factor_types)
            for k in keys:
                acc[k].append(scores[k])
        except Exception as e:
            logger.warning("Null DCI permutation failed: %s", e)
    out = {}
    for k in keys:
        out[k] = float(np.mean(acc[k])) if acc[k] else float("nan")
        out[f"{k}_std"] = float(np.std(acc[k])) if acc[k] else float("nan")
    return out


def _ridge_informativeness(repr_arr, factor_arr, seeds=(0, 1, 2)):
    """Linear-probe informativeness for one repr→factor block.

    Cross-validated, standardized RidgeCV test-R² per factor — the same probe as
    ``eval.run_dci_compare`` (via ``cv_probe_r2``) — so this in-training number is
    directly comparable to the cross-model comparison table, unlike the GBT
    ``informativeness_test`` which a nonlinear learner inflates.  Returns
    ``(mean_over_factors, per_factor_array)``; non-finite factors are dropped from
    the mean.
    """
    from eval.identifiability_metrics import cv_probe_r2

    per_factor = np.array(
        [cv_probe_r2(repr_arr, factor_arr[:, j], seeds=seeds)["mean"] for j in range(factor_arr.shape[1])]
    )
    finite = per_factor[np.isfinite(per_factor)]
    return (float(finite.mean()) if finite.size else float("nan")), per_factor


def compute_dci_synthetic(
    encoder,
    dataset,
    device,
    train_ratio=0.8,
    batch_size=32,
    num_workers=0,
    pooling="gap",
    levels=None,
    per_encoder=False,
    null_permute=False,
    n_null=5,
    null_seed=0,
):
    """Compute DCI metrics for a synthetic dataset with known GT content/style factors.

    Evaluates four combinations per requested encoder level:
      - content_repr → content_factors  (should be high D, C, I)
      - content_repr → style_factors    (should be low I — content shouldn't encode style)
      - style_repr   → style_factors    (should be high)
      - style_repr   → content_factors  (should be low — style shouldn't encode content)

    With ``per_encoder=True`` and two views present, the same four combinations are
    additionally scored for the **second** encoder (the ``_v2`` representations),
    emitted as ``content_v2→…`` / ``style_v2→…`` blocks so per-view informativeness
    on the GT factors is visible under ``--separate-encoders``.  The primary view's
    blocks stay unsuffixed.  Second-view style is read against the second view's
    style factors (``z_style_v2``); content factors are shared across views.

    Args:
        encoder: VQVAE encoder (or any nn.Module with compatible forward signature).
        dataset: SyntheticBrainDataset instance.
        device: torch device.
        train_ratio: fraction of samples for train split.
        batch_size: batch size for representation extraction.
        num_workers: DataLoader workers.
        pooling: How to reduce spatial maps — ``"gap"``, ``"stats"``, or ``(D,H,W)`` tuple.
        levels: List of encoder levels to evaluate.  ``None`` → ``[0]``.
        per_encoder: If True (and a second view is present), also score the second
            encoder's representations against the GT factors, emitting ``content_v2→…``
            / ``style_v2→…`` blocks alongside the primary view's. Doubles the per-level
            probe cost, so it is intended for ``--separate-encoders`` runs.
        null_permute: If True, also compute a row-permuted null floor per block
            (``"{label}/null/..."`` keys) — the score the block's shape yields
            from noise alone.  Real minus null is the non-structural signal.
        n_null: Number of permutations to average for the null floor.
        null_seed: Seed for the permutation RNG.

    Returns:
        dict with keys like ``"L0/content→content/disentanglement"``.
        When a single level is requested the ``L{i}/`` prefix is omitted.
        Each block also gets ``"{label}/informativeness_ridge"`` — the linear
        RidgeCV-probe analogue of the GBT ``informativeness_test``, computed the
        same way as ``eval.run_dci_compare`` so the two are directly comparable.
        With ``null_permute`` each block also gets ``"{label}/null/{metric}"``
        (and ``"..._std"``) entries.
    """
    level_data, gt_content, gt_style_v1, gt_style_v2 = _extract_synthetic_representations(
        encoder, dataset, device, batch_size, num_workers, pooling=pooling
    )

    if levels is None:
        levels = [0]
    available = sorted(level_data.keys())
    use_prefix = len(levels) > 1

    n = gt_content.shape[0]
    split = int(n * train_ratio)
    gt_style = gt_style_v1

    results = {}
    null_rng = np.random.RandomState(null_seed)
    for lvl in levels:
        if lvl not in level_data:
            logger.warning("Level %d not found in encoder outputs (available: %s)", lvl, available)
            continue

        content_repr, style_repr, content_v2, style_v2, info = level_data[lvl]
        prefix = f"L{lvl}/" if use_prefix else ""
        results[f"{prefix}factor_info"] = info

        pairs = [(f"{prefix}content→content", content_repr, gt_content, info["content_names"])]
        pairs.append((f"{prefix}content→style", content_repr, gt_style, info["style_names"]))
        if style_repr is not None:
            pairs.append((f"{prefix}style→style", style_repr, gt_style, info["style_names"]))
            pairs.append((f"{prefix}style→content", style_repr, gt_content, info["content_names"]))

        # Second encoder (view 2): score its representations against the GT factors
        # so per-view informativeness is visible when --separate-encoders gives each
        # modality its own encoder. View-2 style is read against gt_style_v2 (style
        # factors are view-specific); content factors are shared across views.
        if per_encoder and content_v2 is not None:
            pairs.append((f"{prefix}content_v2→content", content_v2, gt_content, info["content_names"]))
            pairs.append((f"{prefix}content_v2→style", content_v2, gt_style_v2, info["style_names"]))
            if style_v2 is not None:
                pairs.append((f"{prefix}style_v2→style", style_v2, gt_style_v2, info["style_names"]))
                pairs.append((f"{prefix}style_v2→content", style_v2, gt_content, info["content_names"]))

        for label, repr_arr, factor_arr, factor_names in pairs:
            mus_train = repr_arr[:split].T
            mus_test = repr_arr[split:].T
            ys_train = factor_arr[:split].T
            ys_test = factor_arr[split:].T

            n_factors = ys_train.shape[0]
            factor_types = ["continuous"] * n_factors

            try:
                scores, detail = _compute_dci(mus_train, ys_train, mus_test, ys_test, factor_types)
                for k, v in scores.items():
                    results[f"{label}/{k}"] = v
                detail["factor_names"] = factor_names
                results[f"{label}/detail"] = detail
            except Exception as e:
                logger.warning("DCI computation failed for %s: %s", label, e)
                for k in [
                    "disentanglement",
                    "completeness",
                    "informativeness_train",
                    "informativeness_test",
                ]:
                    results[f"{label}/{k}"] = float("nan")

            try:
                ridge_mean, ridge_pf = _ridge_informativeness(repr_arr, factor_arr)
                results[f"{label}/informativeness_ridge"] = ridge_mean
                _detail = results.get(f"{label}/detail")
                if isinstance(_detail, dict):
                    _detail["per_factor_ridge"] = ridge_pf
            except Exception as e:
                logger.warning("Ridge informativeness failed for %s: %s", label, e)
                results[f"{label}/informativeness_ridge"] = float("nan")

            if null_permute:
                null = _null_permuted_dci(repr_arr, factor_arr, split, factor_types, n_null, null_rng)
                for k, v in null.items():
                    results[f"{label}/null/{k}"] = v

        # ── Canonical block-identifiability metrics (Yao-style) ──────────────
        # These are the trustworthy headline numbers (see eval/identifiability_metrics):
        #   block_mcc      — identifiability up to the allowed invertible map
        #                    (content->content high; content->style low = no leakage).
        #   content->view  — over-allocation leakage signal; ~0.5 means content is
        #                    view-invariant (clean of style), the Yao/von Kügelgen claim.
        try:
            from eval.identifiability_metrics import block_mcc as _block_mcc
            from eval.identifiability_metrics import view_invariance as _view_invariance

            results[f"{prefix}content->content/block_mcc"] = _block_mcc(content_repr, gt_content)["mean"]
            if style_repr is not None:
                results[f"{prefix}style->style/block_mcc"] = _block_mcc(style_repr, gt_style)["mean"]
                results[f"{prefix}content->style/block_mcc"] = _block_mcc(content_repr, gt_style)["mean"]
            if per_encoder and content_v2 is not None:
                results[f"{prefix}content_v2->content/block_mcc"] = _block_mcc(content_v2, gt_content)["mean"]
                if style_v2 is not None:
                    results[f"{prefix}style_v2->style/block_mcc"] = _block_mcc(style_v2, gt_style_v2)["mean"]
                    results[f"{prefix}content_v2->style/block_mcc"] = _block_mcc(content_v2, gt_style_v2)["mean"]
            if content_v2 is not None:
                _s1 = style_repr if style_repr is not None else content_repr[:, :0]
                _s2 = style_v2 if style_v2 is not None else content_v2[:, :0]
                _vi = _view_invariance(content_repr, content_v2, _s1, _s2)
                results[f"{prefix}content->view/acc"] = _vi["content_acc"]
                if style_repr is not None:
                    results[f"{prefix}style->view/acc"] = _vi["style_acc"]
        except Exception as e:
            logger.warning("Block-identifiability metrics failed at L%s: %s", lvl, e)

    logger.info("DCI synthetic results:")
    for k, v in results.items():
        if not isinstance(v, dict) and not np.isnan(v):
            logger.info("  %-45s %.4f", k, v)

    return results


def flatten_dci_results(results):
    """Flatten DCI results dict into a single-row dict for TB / W&B logging.

    Keeps the per-block summary scalars (disentanglement, completeness,
    informativeness train/test, block-MCC, view accuracy) and expands only the
    per-factor *test* R² into named columns. The per-factor train/completeness
    arrays and the per-code disentanglement are intentionally dropped here — they
    are too many scalars to log per step and remain available in the ``detail``
    dict (and the CSV from ``dci_results_to_rows``).

    - Replaces Unicode arrows with ASCII ``->``
    - Skips importance matrices and factor_info dicts
    """
    flat = {}
    for k, v in results.items():
        key = k.replace("→", "->")
        if isinstance(v, (int, float, np.floating)):
            flat[key] = float(v)
        elif isinstance(v, dict) and "factor_names" in v and "per_factor_test" in v:
            section = key.replace("/detail", "")
            names = v["factor_names"]
            arr = v["per_factor_test"]
            for j, name in enumerate(names):
                flat[f"{section}/per_factor_test/{name}"] = float(arr[j])
    return flat


DCI_CSV_COLUMNS = ["level", "section", "factor", "train_r2", "test_r2", "ridge_r2", "completeness", "disentanglement"]


def dci_results_to_rows(results):
    """Convert DCI results dict into a table that mirrors the terminal printout.

    Each row is one factor within one section.  A ``(mean)`` row per section
    carries the summary scores and the section-level disentanglement.

    Columns: level, section, factor, train_r2, test_r2, ridge_r2, completeness, disentanglement.
    ``ridge_r2`` is the linear RidgeCV-probe informativeness; ``train_r2``/``test_r2``
    are the GBT informativeness.
    """
    levels_found = set()
    for k in results:
        parts = k.split("/", 1)
        if parts[0].startswith("L") and parts[0][1:].isdigit():
            levels_found.add(int(parts[0][1:]))
    has_prefix = bool(levels_found)
    if not has_prefix:
        levels_found = {0}

    section_names = [
        "content->content",
        "content->style",
        "style->style",
        "style->content",
        # Second encoder (only present when compute_dci_synthetic ran with per_encoder).
        "content_v2->content",
        "content_v2->style",
        "style_v2->style",
        "style_v2->content",
    ]
    section_keys = [
        "content→content",
        "content→style",
        "style→style",
        "style→content",
        "content_v2→content",
        "content_v2→style",
        "style_v2→style",
        "style_v2→content",
    ]

    rows = []
    for lvl in sorted(levels_found):
        pfx = f"L{lvl}/" if has_prefix else ""
        for sec_name, sec_key in zip(section_names, section_keys):
            full = f"{pfx}{sec_key}"

            detail = results.get(f"{full}/detail")
            if detail is not None and "factor_names" in detail:
                names = detail["factor_names"]
                pf_train = detail.get("per_factor_train")
                pf_test = detail.get("per_factor_test")
                pf_comp = detail.get("per_factor_completeness")
                pf_ridge = detail.get("per_factor_ridge")
                for j, name in enumerate(names):
                    rows.append(
                        {
                            "level": lvl,
                            "section": sec_name,
                            "factor": name,
                            "train_r2": round(float(pf_train[j]), 4) if pf_train is not None else "",
                            "test_r2": round(float(pf_test[j]), 4) if pf_test is not None else "",
                            "ridge_r2": round(float(pf_ridge[j]), 4) if pf_ridge is not None else "",
                            "completeness": round(float(pf_comp[j]), 4) if pf_comp is not None else "",
                            "disentanglement": "",
                        }
                    )

            i_train = results.get(f"{full}/informativeness_train")
            i_test = results.get(f"{full}/informativeness_test")
            i_ridge = results.get(f"{full}/informativeness_ridge")
            c_score = results.get(f"{full}/completeness")
            d_score = results.get(f"{full}/disentanglement")
            if any(v is not None for v in (i_train, i_test, i_ridge, c_score, d_score)):
                rows.append(
                    {
                        "level": lvl,
                        "section": sec_name,
                        "factor": "(mean)",
                        "train_r2": round(float(i_train), 4) if i_train is not None else "",
                        "test_r2": round(float(i_test), 4) if i_test is not None else "",
                        "ridge_r2": round(float(i_ridge), 4) if i_ridge is not None else "",
                        "completeness": round(float(c_score), 4) if c_score is not None else "",
                        "disentanglement": round(float(d_score), 4) if d_score is not None else "",
                    }
                )

    return rows
