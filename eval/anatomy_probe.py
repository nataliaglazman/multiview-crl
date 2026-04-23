"""Anatomy probe — measures modality-invariant anatomical info in content vs style.

Mirrors the demographic-probe patterns in ``eval/view_latents.ipynb`` (cells 10,
27, 56, 69):

  * Loads demographics from ``merged_data.csv`` sitting next to the labels CSV.
  * Decodes ``PTGENDER`` (1.0/2.0 → Male/Female), ``PTRACCAT`` (multi-race
    codes split on ``|``, first code wins), ``Group`` (CN/MCI/AD), ``AGE``.
  * Extracts per-level pooled encoder features using the same
    ``vqvae(img, return_recon=False, pool_only=True, view_idx=...)`` contract
    the notebook uses, and reads per-level content/style indices from the
    returned ``soft_masks`` dict (supports per-view masks).
  * Probes with a ``Normalizer(l2) → StandardScaler → {LogisticRegression, MLP}``
    pipeline under 5-fold StratifiedKFold, reports balanced accuracy for
    classification and R²/MAE for age regression.

Returns a flat dict suitable for W&B / TB logging.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler

logger = logging.getLogger("multiview_crl")


# ---------------------------------------------------------------------------
# Demographic lookup (follows view_latents.ipynb cell 56 / 69)
# ---------------------------------------------------------------------------

_GENDER_MAP = {1.0: "Male", 2.0: "Female"}
_RACE_MAP = {
    1: "Am Indian/Alaskan",
    2: "Asian",
    3: "Native Hawaiian/Pacific Isl",
    4: "Black",
    5: "White",
    6: "More than one",
    7: "Unknown",
}


def _load_demo_lookup(labels_path: str) -> pd.DataFrame:
    demo_path = os.path.join(os.path.dirname(labels_path), "merged_data.csv")
    if not os.path.exists(demo_path):
        raise FileNotFoundError(
            f"Expected demographic metadata at {demo_path} (next to labels CSV). " "See view_latents.ipynb cell 56."
        )
    demo = pd.read_csv(demo_path)
    return demo.drop_duplicates(subset="Subject").set_index("Subject")


def _lookup_gender(lookup: pd.DataFrame, subj: str):
    if subj not in lookup.index or "PTGENDER" not in lookup.columns:
        return None
    v = lookup.loc[subj, "PTGENDER"]
    return _GENDER_MAP.get(float(v)) if pd.notna(v) else None


def _lookup_race(lookup: pd.DataFrame, subj: str):
    if subj not in lookup.index or "PTRACCAT" not in lookup.columns:
        return None
    v = lookup.loc[subj, "PTRACCAT"]
    if pd.isna(v):
        return None
    first = str(v).split("|")[0]
    try:
        return _RACE_MAP.get(int(first))
    except ValueError:
        return None


def _lookup_group(lookup: pd.DataFrame, subj: str):
    if subj not in lookup.index or "Group" not in lookup.columns:
        return None
    v = lookup.loc[subj, "Group"]
    return None if pd.isna(v) else str(v)


def _lookup_age(lookup: pd.DataFrame, subj: str):
    if subj not in lookup.index:
        return None
    for col in ("AGE", "Age", "age"):
        if col in lookup.columns:
            v = lookup.loc[subj, col]
            return float(v) if pd.notna(v) else None
    return None


# ---------------------------------------------------------------------------
# Feature extraction (mirrors notebook cell 10)
# ---------------------------------------------------------------------------

_VIEW_IDX = {"T1": 0, "T2": 1}


@torch.no_grad()
def extract_features(vqvae_model, items, val_transforms, device, nb_levels: int):
    """Run the model over ``items`` (T1+T2 per subject) and collect pooled
    per-level features plus per-level content/style indices.

    Follows cell 10 of view_latents.ipynb.  Returns all rows in T1/T2 alternating
    order: row 2k = T1 of subject k, row 2k+1 = T2 of subject k.
    """
    vqvae_module = vqvae_model.module if hasattr(vqvae_model, "module") else vqvae_model
    has_per_view = (
        getattr(vqvae_module, "separate_encoders", False)
        and getattr(vqvae_module, "channel_logits_v1", None) is not None
    )

    all_features: Dict[str, List[np.ndarray]] = {f"level_{i}": [] for i in range(nb_levels)}
    all_modalities: List[str] = []
    all_subjects: List[str] = []
    content_indices: Dict[int, List[int]] = {}
    content_indices_v1: Dict[int, List[int]] = {}
    style_indices: Dict[int, List[int]] = {}
    style_indices_v1: Dict[int, List[int]] = {}

    for idx, item in enumerate(items):
        if idx % 20 == 0:
            logger.info(f"[anatomy_probe] extracting {idx}/{len(items)}")

        data_dict = {"image_t1": item["image"], "image_t2": item["z_image"]}
        transformed = val_transforms(data_dict)

        for modality, key in (("T1", "image_t1"), ("T2", "image_t2")):
            img = transformed[key].unsqueeze(0).to(device)
            out = vqvae_model(
                img,
                return_recon=False,
                pool_only=True,
                view_idx=_VIEW_IDX[modality],
            )
            _, _, enc_features, _, _, _, soft_masks, *_ = out

            for L, pooled in enumerate(enc_features):
                all_features[f"level_{L}"].append(pooled.squeeze(0).cpu().float().numpy())

            for L, mask in soft_masks.items():
                if isinstance(mask, tuple):
                    mask_v0, mask_v1 = mask
                    if modality == "T1" and L not in content_indices:
                        content_indices[L] = torch.where(mask_v0.bool())[-1].tolist()
                        style_indices[L] = torch.where(~mask_v0.bool())[-1].tolist()
                    if modality == "T2" and L not in content_indices_v1:
                        content_indices_v1[L] = torch.where(mask_v1.bool())[-1].tolist()
                        style_indices_v1[L] = torch.where(~mask_v1.bool())[-1].tolist()
                else:
                    indices = torch.where(mask.bool())[-1].tolist()
                    style = torch.where(~mask.bool())[-1].tolist()
                    if modality == "T1" and L not in content_indices:
                        content_indices[L] = indices
                        style_indices[L] = style
                    if modality == "T2" and has_per_view and L not in content_indices_v1:
                        content_indices_v1[L] = indices
                        style_indices_v1[L] = style

            all_modalities.append(modality)
            all_subjects.append(item["subject"])

    for k in all_features:
        all_features[k] = np.stack(all_features[k], axis=0)
    all_modalities = np.array(all_modalities)
    all_subjects = np.array(all_subjects)

    if not has_per_view:
        content_indices_v1 = dict(content_indices)
        style_indices_v1 = dict(style_indices)

    return {
        "features": all_features,
        "modalities": all_modalities,
        "subjects": all_subjects,
        "content_indices": content_indices,
        "content_indices_v1": content_indices_v1,
        "style_indices": style_indices,
        "style_indices_v1": style_indices_v1,
        "has_per_view": has_per_view,
        "nb_levels": nb_levels,
    }


def _per_view_select(feats: np.ndarray, idx_v0, idx_v1):
    """Mirror notebook's `_per_view_select`: T1 rows use idx_v0, T2 rows idx_v1,
    producing a k-dim column-aligned matrix."""
    n = feats.shape[0]
    t1_mask = np.arange(n) % 2 == 0
    out = np.empty((n, len(idx_v0)), dtype=feats.dtype)
    out[t1_mask] = feats[t1_mask][:, idx_v0]
    out[~t1_mask] = feats[~t1_mask][:, idx_v1]
    return out


def _build_feature_sets(extracted) -> Dict[str, np.ndarray]:
    feats_by_level = extracted["features"]
    nb_levels = extracted["nb_levels"]
    ci = extracted["content_indices"]
    ci_v1 = extracted["content_indices_v1"]
    si = extracted["style_indices"]
    si_v1 = extracted["style_indices_v1"]

    sets: Dict[str, np.ndarray] = {}
    content_parts, style_parts = [], []
    for L in range(nb_levels):
        f = feats_by_level[f"level_{L}"]
        c0 = list(ci.get(L, []))
        c1 = list(ci_v1.get(L, c0))
        s0 = list(si.get(L, []))
        s1 = list(si_v1.get(L, s0))
        if c0:
            X_c = _per_view_select(f, c0, c1)
            sets[f"Content L{L}"] = X_c
            content_parts.append(X_c)
        if s0:
            X_s = _per_view_select(f, s0, s1)
            sets[f"Style L{L}"] = X_s
            style_parts.append(X_s)
        sets[f"All L{L}"] = f
    if content_parts:
        sets["Content all"] = np.concatenate(content_parts, axis=1)
    if style_parts:
        sets["Style all"] = np.concatenate(style_parts, axis=1)
    return sets


# ---------------------------------------------------------------------------
# Probes (Normalizer → StandardScaler → LR / MLP, 5-fold StratifiedKFold)
# ---------------------------------------------------------------------------


def _lr_probe():
    return make_pipeline(
        Normalizer(norm="l2"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=42),
    )


def _mlp_probe():
    return make_pipeline(
        Normalizer(norm="l2"),
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            random_state=42,
        ),
    )


def _ridge_probe():
    return make_pipeline(Normalizer(norm="l2"), StandardScaler(), Ridge(alpha=1.0))


def _mlp_regressor():
    return make_pipeline(
        Normalizer(norm="l2"),
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=42),
    )


def _cv_classification(X, y, key: str) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr = cross_val_score(_lr_probe(), X, y, cv=cv, scoring="balanced_accuracy")
    mlp = cross_val_score(_mlp_probe(), X, y, cv=cv, scoring="balanced_accuracy")
    chance = cross_val_score(
        DummyClassifier(strategy="most_frequent"),
        np.zeros((len(y), 1)),
        y,
        cv=cv,
        scoring="balanced_accuracy",
    ).mean()
    return {
        f"{key}/lr_balAcc": float(lr.mean()),
        f"{key}/lr_balAcc_std": float(lr.std()),
        f"{key}/mlp_balAcc": float(mlp.mean()),
        f"{key}/mlp_balAcc_std": float(mlp.std()),
        f"{key}/chance": float(chance),
    }


def _cv_regression(X, y, key: str) -> Dict[str, float]:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_lr = cross_val_score(_ridge_probe(), X, y, cv=cv, scoring="r2")
    r2_mlp = cross_val_score(_mlp_regressor(), X, y, cv=cv, scoring="r2")
    mae_lr = -cross_val_score(_ridge_probe(), X, y, cv=cv, scoring="neg_mean_absolute_error")
    mae_mlp = -cross_val_score(_mlp_regressor(), X, y, cv=cv, scoring="neg_mean_absolute_error")
    dummy_mae = -cross_val_score(
        DummyRegressor(strategy="mean"),
        np.zeros((len(y), 1)),
        y,
        cv=cv,
        scoring="neg_mean_absolute_error",
    ).mean()
    return {
        f"{key}/lr_r2": float(r2_lr.mean()),
        f"{key}/lr_mae": float(mae_lr.mean()),
        f"{key}/mlp_r2": float(r2_mlp.mean()),
        f"{key}/mlp_mae": float(mae_mlp.mean()),
        f"{key}/chance_mae": float(dummy_mae),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

TARGETS_CLS = ("sex", "race", "group")
TARGETS_REG = ("age",)


def anatomy_eval(vqvae_model, items, val_transforms, device, labels_path: str, nb_levels: int) -> Dict[str, float]:
    """Run the anatomy probe suite.  Returns a flat metrics dict.

    Parameters
    ----------
    vqvae_model : the VQ-VAE encoder (same object used in val_step).
    items : list from ``utils.load_data`` — each dict has 'image', 'z_image',
        'subject' (+ optional masks).  Patient-level split is implicit: each
        subject contributes one T1 row and one T2 row.
    val_transforms : MONAI transform used by the notebook.
    device : torch device.
    labels_path : path to labels CSV; ``merged_data.csv`` is expected next to
        it (same directory), matching the notebook convention.
    nb_levels : number of VQ-VAE levels (``args.vqvae_nb_levels``).
    """
    lookup = _load_demo_lookup(labels_path)
    extracted = extract_features(vqvae_model, items, val_transforms, device, nb_levels)
    subjects = extracted["subjects"]
    modalities = extracted["modalities"]

    # Probes are T1-only — mirrors notebook cell 69, avoids modality confound.
    t1_mask = modalities == "T1"
    subjects_t1 = subjects[t1_mask]

    targets = {
        "sex": np.array([_lookup_gender(lookup, s) for s in subjects_t1], dtype=object),
        "race": np.array([_lookup_race(lookup, s) for s in subjects_t1], dtype=object),
        "group": np.array([_lookup_group(lookup, s) for s in subjects_t1], dtype=object),
        "age": np.array([_lookup_age(lookup, s) for s in subjects_t1], dtype=object),
    }

    feature_sets = _build_feature_sets(extracted)
    # Slice to T1 rows only (classification/regression probes).
    feature_sets_t1 = {name: X[t1_mask] for name, X in feature_sets.items()}

    results: Dict[str, float] = {}

    # Classification targets
    for t in TARGETS_CLS:
        y_raw = targets[t]
        if y_raw is None or all(v is None for v in y_raw):
            logger.warning(f"[anatomy_probe] no values for target '{t}' — skipping")
            continue
        valid = np.array([v is not None for v in y_raw])
        y = y_raw[valid].astype(str)
        classes, counts = np.unique(y, return_counts=True)
        keep = classes[counts >= 10]  # ≥ 2 × n_splits (StratifiedKFold)
        mask = np.isin(y, keep)
        if mask.sum() < 10:
            logger.warning(f"[anatomy_probe] insufficient data for '{t}' after rare-class drop — skipping")
            continue
        y = LabelEncoder().fit_transform(y[mask])
        for fs_name, X_full in feature_sets_t1.items():
            X = X_full[valid][mask]
            key = f"anatomy/{t}/{_slug(fs_name)}"
            results.update(_cv_classification(X, y, key))

    # Age regression
    if "age" in targets:
        y_raw = targets["age"]
        valid = np.array([v is not None for v in y_raw])
        if valid.sum() >= 10:
            y = y_raw[valid].astype(float)
            for fs_name, X_full in feature_sets_t1.items():
                X = X_full[valid]
                key = f"anatomy/age/{_slug(fs_name)}"
                results.update(_cv_regression(X, y, key))
        else:
            logger.warning("[anatomy_probe] not enough age labels — skipping")

    # Modality probe — uses BOTH T1 and T2 rows, labelled 0/1.
    y_mod = (modalities == "T2").astype(int)
    for fs_name, X in feature_sets.items():
        key = f"modality/{_slug(fs_name)}"
        results.update(_cv_classification(X, y_mod, key))

    # Separation score per level (classification-averaged).
    for L in range(nb_levels):
        content_key = f"anatomy/{{t}}/content_l{L}"
        mod_key = f"modality/content_l{L}/lr_balAcc"
        if mod_key not in results:
            continue
        anat_scores = [
            results[content_key.format(t=t) + "/lr_balAcc"]
            for t in TARGETS_CLS
            if content_key.format(t=t) + "/lr_balAcc" in results
        ]
        if anat_scores:
            results[f"sep/L{L}"] = float(np.mean(anat_scores)) - results[mod_key]

    return results


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("⊕", "plus")
