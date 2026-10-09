"""Subject-disjoint linear diagnosis and demographic probes of frozen content."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


def _subject_folds(labels, subjects, seed=0):
    """Exclude missing/conflicting labels and singleton classes, then split subjects.

    Stratifying unique subjects guarantees every retained class occurs in every
    fold. Repeat scans and both modalities always stay on the same side.
    """
    labels, subjects = np.asarray(labels), np.asarray(subjects)
    if labels.ndim != 1 or subjects.shape != labels.shape:
        raise ValueError("Probe labels and subject IDs must be aligned one-dimensional arrays")
    keep = labels >= 0
    conflicting = np.zeros(len(labels), dtype=bool)
    for subject in np.unique(subjects[keep]):
        indices = (subjects == subject) & keep
        if len(np.unique(labels[indices])) > 1:
            conflicting |= indices
    keep &= ~conflicting
    class_subjects = {
        int(label): len(np.unique(subjects[(labels == label) & keep])) for label in np.unique(labels[keep])
    }
    rare = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels[keep]):
        indices = (labels == label) & keep
        if len(np.unique(subjects[indices])) < 2:
            rare |= indices
    keep &= ~rare
    y, groups = labels[keep], subjects[keep]
    unique_subjects, first = np.unique(groups, return_index=True)
    subject_labels = y[first]
    classes, counts = np.unique(subject_labels, return_counts=True)
    folds = []
    if len(classes) >= 2:
        splitter = StratifiedKFold(n_splits=min(3, int(counts.min())), shuffle=True, random_state=seed)
        for train, test in splitter.split(unique_subjects, subject_labels):
            folds.append(
                (
                    np.flatnonzero(np.isin(groups, unique_subjects[train])),
                    np.flatnonzero(np.isin(groups, unique_subjects[test])),
                )
            )
    stats = {
        "n": int(keep.sum()),
        "subjects": len(unique_subjects),
        "missing": int((labels < 0).sum()),
        "conflicting": int(conflicting.sum()),
        "rare": int(rare.sum()),
        "classes": len(classes),
        "folds": len(folds),
        "chance": (float(np.unique(y, return_counts=True)[1].max() / len(y)) if len(y) else float("nan")),
        "balanced_chance": 1.0 / len(classes) if len(classes) else float("nan"),
    }
    stats.update({f"class_{label}_subjects": count for label, count in class_subjects.items()})
    return keep, folds, stats


def _score(features, labels, folds):
    predictions = np.empty_like(labels)
    for train, test in folds:
        probe = make_pipeline(
            Normalizer(norm="l2"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver="lbfgs"),
        )
        probe.fit(features[train], labels[train])
        predictions[test] = probe.predict(features[test])
    return float(accuracy_score(labels, predictions)), float(balanced_accuracy_score(labels, predictions))


def evaluate_subject_probes(representations):
    """Probe each view and pooled views using identical subject folds per target.

    These classifiers are fitted/evaluated within the held-out validation cohort;
    their labels never enter encoder training. Unscorable targets have NaN scores
    and explicit coverage counts, rather than invented chance-level results.
    """
    targets = dict(representations.get("demographics", {}))
    if "labels" in representations:
        targets["diagnosis"] = representations["labels"]
    if not targets:
        return {}
    subjects = representations["subjects"]
    metrics = {}
    for target, labels in targets.items():
        labels = np.asarray(labels)
        keep, folds, stats = _subject_folds(labels, subjects)
        prefix = f"content/{target}_probe"
        metrics.update({f"{prefix}_{key}": float(value) for key, value in stats.items()})
        y = labels[keep]
        views = [representations[f"content_v{view}"][keep] for view in (0, 1)]
        pooled_folds = [
            (
                np.concatenate([train, train + len(y)]),
                np.concatenate([test, test + len(y)]),
            )
            for train, test in folds
        ]
        for suffix, features, outcomes, split in (
            ("_v0", views[0], y, folds),
            ("_v1", views[1], y, folds),
            ("", np.concatenate(views), np.tile(y, 2), pooled_folds),
        ):
            acc, balanced = _score(features, outcomes, split) if split else (float("nan"), float("nan"))
            metrics[f"{prefix}_acc{suffix}"] = acc
            metrics[f"{prefix}_balanced_acc{suffix}"] = balanced
            if target == "diagnosis":
                metrics[f"content/diagnosis_info{suffix}"] = (
                    max(0.0, (acc - stats["chance"]) / max(1e-6, 1.0 - stats["chance"]))
                    if np.isfinite(acc)
                    else float("nan")
                )
    return metrics
