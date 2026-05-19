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
    scores = _compute_dci(mus_train, ys_train, mus_test, ys_test, factor_types)
    return scores


def _compute_dci(mus_train, ys_train, mus_test, ys_test, factor_types):
    """Computes score based on both training and testing codes and factors."""
    scores = {}
    importance_matrix, train_err, test_err = compute_importance_gbt(
        mus_train, ys_train, mus_test, ys_test, factor_types
    )
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    scores["informativeness_train"] = train_err
    scores["informativeness_test"] = test_err
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    return scores


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
    importance_matrix = np.zeros(shape=[num_codes, num_factors], dtype=np.float64)
    train_loss = []
    test_loss = []
    for i, ftype in zip(range(num_factors), factor_types):
        if ftype == "discrete":
            model = ensemble.GradientBoostingClassifier()
            model.fit(x_train.T, y_train[i, :])
            importance_matrix[:, i] = np.abs(model.feature_importances_)
            train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
            test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
        else:
            model = ensemble.GradientBoostingRegressor()
            model.fit(x_train.T, y_train[i, :])
            importance_matrix[:, i] = np.abs(model.feature_importances_)
            train_loss.append(model.score(x_train.T, y_train[i, :]))
            test_loss.append(model.score(x_test.T, y_test[i, :]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


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


def _extract_synthetic_representations(encoder, dataset, device, batch_size=32, num_workers=0):
    """Run the encoder over a synthetic dataset, returning content/style representations and GT factors.

    Returns:
        content_repr: (N, n_content_channels) numpy array
        style_repr:   (N, n_style_channels)  numpy array  — None if no content/style split
        gt_content:   (N, n_content_factors) numpy array   (the scalar z_content)
        gt_style_v1:  (N, n_style_factors) numpy array
        gt_style_v2:  (N, n_style_factors) numpy array
        factor_info:  dict with metadata (factor names, counts)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_content_repr = []
    all_style_repr = []
    all_gt_content = []
    all_gt_style_v1 = []
    all_gt_style_v2 = []

    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"]
            x = torch.cat(imgs, dim=0).to(device)

            enc_out = encoder(x, pool_only=True, n_views=len(imgs))

            if isinstance(enc_out, tuple):
                encoder_features = enc_out[2]
                content_idx_raw = enc_out[3]
                if len(encoder_features) > 0 and encoder_features[0].dim() == 5:
                    pooled = encoder_features[0].mean(dim=[2, 3, 4])
                elif len(encoder_features) > 0:
                    pooled = encoder_features[0]
                else:
                    pooled = x.view(x.size(0), -1)
            else:
                pooled = enc_out
                content_idx_raw = None

            B_per_view = pooled.shape[0] // len(imgs)
            pooled_v1 = pooled[:B_per_view]

            content_indices = None
            if content_idx_raw is not None:
                if isinstance(content_idx_raw, (list, tuple)) and len(content_idx_raw) > 0:
                    first = content_idx_raw[0]
                    if isinstance(first, (list, tuple)):
                        content_indices = list(first)
                    elif isinstance(first, torch.Tensor):
                        content_indices = first.cpu().tolist()
                    else:
                        content_indices = list(content_idx_raw)
                elif isinstance(content_idx_raw, torch.Tensor):
                    content_indices = content_idx_raw.cpu().tolist()

            pooled_np = pooled_v1.cpu().numpy()

            if content_indices is not None and len(content_indices) > 0:
                all_channels = set(range(pooled_np.shape[1]))
                style_indices = sorted(all_channels - set(content_indices))
                all_content_repr.append(pooled_np[:, content_indices])
                if style_indices:
                    all_style_repr.append(pooled_np[:, style_indices])
            else:
                all_content_repr.append(pooled_np)

            gt = batch["gt_latents"]
            all_gt_content.append(gt["z_content"].numpy())
            all_gt_style_v1.append(gt["z_style_v1"].numpy())
            all_gt_style_v2.append(gt["z_style_v2"].numpy())

    content_repr = np.concatenate(all_content_repr, axis=0)
    style_repr = np.concatenate(all_style_repr, axis=0) if all_style_repr else None
    gt_content = np.concatenate(all_gt_content, axis=0)
    gt_style_v1 = np.concatenate(all_gt_style_v1, axis=0)
    gt_style_v2 = np.concatenate(all_gt_style_v2, axis=0)

    n_c = gt_content.shape[1]
    n_s = gt_style_v1.shape[1]

    factor_info = {
        "content_names": CONTENT_FACTOR_NAMES[:n_c],
        "style_names": STYLE_FACTOR_NAMES[:n_s],
        "n_content": n_c,
        "n_style": n_s,
        "n_content_channels": content_repr.shape[1],
        "n_style_channels": style_repr.shape[1] if style_repr is not None else 0,
        "has_split": style_repr is not None,
    }
    return content_repr, style_repr, gt_content, gt_style_v1, gt_style_v2, factor_info


def compute_dci_synthetic(encoder, dataset, device, train_ratio=0.8, batch_size=32, num_workers=0):
    """Compute DCI metrics for a synthetic dataset with known GT content/style factors.

    Evaluates four combinations:
      - content_repr → content_factors  (should be high D, C, I)
      - content_repr → style_factors    (should be low I — content shouldn't encode style)
      - style_repr   → style_factors    (should be high)
      - style_repr   → content_factors  (should be low — style shouldn't encode content)

    Args:
        encoder: VQVAE encoder (or any nn.Module with compatible forward signature).
        dataset: SyntheticBrainDataset instance.
        device: torch device.
        train_ratio: fraction of samples for train split.
        batch_size: batch size for representation extraction.
        num_workers: DataLoader workers.

    Returns:
        dict with keys like "content→content/disentanglement", etc.
    """
    content_repr, style_repr, gt_content, gt_style_v1, gt_style_v2, info = _extract_synthetic_representations(
        encoder, dataset, device, batch_size, num_workers
    )

    n = content_repr.shape[0]
    split = int(n * train_ratio)
    gt_style = gt_style_v1

    pairs = [("content→content", content_repr, gt_content)]
    pairs.append(("content→style", content_repr, gt_style))
    if style_repr is not None:
        pairs.append(("style→style", style_repr, gt_style))
        pairs.append(("style→content", style_repr, gt_content))

    results = {"factor_info": info}
    for label, repr_arr, factor_arr in pairs:
        mus_train = repr_arr[:split].T
        mus_test = repr_arr[split:].T
        ys_train = factor_arr[:split].T
        ys_test = factor_arr[split:].T

        n_factors = ys_train.shape[0]
        factor_types = ["continuous"] * n_factors

        try:
            scores = _compute_dci(mus_train, ys_train, mus_test, ys_test, factor_types)
            for k, v in scores.items():
                results[f"{label}/{k}"] = v
        except Exception as e:
            logger.warning("DCI computation failed for %s: %s", label, e)
            for k in ["disentanglement", "completeness", "informativeness_train", "informativeness_test"]:
                results[f"{label}/{k}"] = float("nan")

    logger.info("DCI synthetic results:")
    for k, v in results.items():
        if k != "factor_info" and not np.isnan(v):
            logger.info("  %-45s %.4f", k, v)

    return results
