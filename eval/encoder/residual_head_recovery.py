"""Evaluation-only held-out factor probes for independent image/residual codes.

The encoder and its fitted preprocessing remain fixed. Only the supervised ridge
readouts are fitted, within subject folds of the supplied diagnostic cohort.
Repeated validation monitoring does not make that cohort an untouched final test.
"""

import random
from contextlib import contextmanager

import numpy as np
import torch

from eval.metrics.dci import content_factor_names, dataset_lesion_target

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
FOLD_SEED = 42


@contextmanager
def _preserve_evaluation_state(model, device):
    """Keep inference from changing training modes, gradients or random streams."""
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    mps_state = torch.mps.get_rng_state() if torch.device(device).type == "mps" else None
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        # Assign individual flags; model.train() could overwrite mixed modes or
        # invoke a custom freezing policy different from the caller's state.
        for module, training in modes:
            module.training = training


def _out_of_fold_r2(features, targets, names, folds):
    """Fit the scaler and RidgeCV only on each outer training fold."""
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    predictions = np.empty_like(targets, dtype=np.float64)
    for train, test in KFold(folds, shuffle=True, random_state=FOLD_SEED).split(features):
        scaler = StandardScaler().fit(features[train])
        # RidgeCV's built-in generalized leave-one-out selection runs entirely
        # inside the outer training fold, including when only two rows remain.
        probe = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(scaler.transform(features[train]), targets[train])
        predictions[test] = probe.predict(scaler.transform(features[test]))
    scores = {}
    for column, name in enumerate(names):
        target = targets[:, column]
        # No recovery number exists for a target that never varies. In
        # particular the burden generator's unused factors are not successes.
        score = None if np.ptp(target) <= 1e-12 else float(r2_score(target, predictions[:, column]))
        scores[name] = score if score is None or np.isfinite(score) else None
    return scores


def _rendered_burden(model, dataset):
    """Return actual rendered lesion voxels when the dataset exposes that renderer."""
    inner = getattr(dataset, "_inner", None)
    renderer = getattr(inner, "renderer", None)
    supported = (
        callable(getattr(inner, "_draw_pseudo_mri", None))
        and callable(getattr(inner, "sample_seed_for", None))
        and callable(getattr(renderer, "render_structure", None))
        and hasattr(inner, "clean_content")
    )
    input_size = getattr(getattr(model, "residual_heads", None), "input_size", None)
    if not supported or input_size is None:
        return None
    from eval.encoder.regional_lesion_detection import lesion_fractions

    fractions = lesion_fractions(dataset, (1, 1, 1))
    return np.asarray(fractions, dtype=np.float64).reshape(len(dataset)) * int(input_size) ** 3


def residual_head_recovery(model, dataset, device, batch_size):
    """Held-out RidgeCV R² for each view's image, residual and combined content.

    Returns ``None`` for an image-only model. The supplied cohort supplies only
    diagnostic readout labels; neither PCA nor model statistics are fitted here.
    Constant target scores are ``None``, and fewer than four subjects produce an
    explicit insufficient-subjects report rather than an invalid tiny-fold fit.
    RNG and all module training flags are restored even if evaluation fails.
    """
    residual_units = int(getattr(model, "residual_units", 0))
    if residual_units <= 0:
        return None
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    subjects = len(dataset)
    report = {
        "status": "complete" if subjects >= 4 else "insufficient_subjects",
        "num_subjects": subjects,
        "folds": min(5, subjects) if subjects >= 4 else 0,
        "probe": {
            "name": "StandardScaler+RidgeCV",
            "evaluation_only": True,
            "fold_seed": FOLD_SEED,
            "alphas": list(ALPHAS),
            "scaler_fit": "outer training fold only",
            "alpha_selection": "generalized leave-one-out within outer training fold",
            "score": "R2 over all out-of-fold subject predictions",
        },
        "factor_names": [],
        "rendered_burden_available": False,
        "views": {},
    }
    if subjects < 4:
        return report

    # Lazy import avoids a cycle when score_checkpoint invokes this helper.
    from eval.protocol.score_checkpoint import encode_blocks

    with _preserve_evaluation_state(model, device):
        blocks = encode_blocks(model, dataset, device, batch_size, model.content_channels)
        targets = np.asarray(blocks["z_content"], dtype=np.float64)
        if targets.ndim != 2 or targets.shape[0] != subjects:
            raise ValueError("content targets must have one row per diagnostic subject")
        names = content_factor_names(targets.shape[1], dataset_lesion_target(dataset))
        if len(names) != targets.shape[1]:
            raise ValueError("content target width exceeds the supported synthetic factor names")
        burden = _rendered_burden(model, dataset)
        if burden is not None:
            names = [*names, "rendered_burden_voxels"]
            targets = np.column_stack((targets, burden))
            report["rendered_burden_available"] = True
        if not np.isfinite(targets).all():
            raise ValueError("diagnostic content targets must be finite")
        report["factor_names"] = names
        for view, codes in zip(("t1", "flair"), blocks["content"]):
            codes = np.asarray(codes, dtype=np.float64)
            expected_width = model.content_channels + residual_units
            if codes.shape != (subjects, expected_width):
                raise ValueError(f"expected combined content shape {(subjects, expected_width)}, got {codes.shape}")
            if not np.isfinite(codes).all():
                raise ValueError("diagnostic codes must be finite")
            readouts = {
                "image": codes[:, : model.content_channels],
                "residual": codes[:, model.content_channels :],
                "combined": codes,
            }
            report["views"][view] = {
                name: {
                    "width": features.shape[1],
                    "r2": _out_of_fold_r2(features, targets, names, report["folds"]),
                }
                for name, features in readouts.items()
            }
    return report
