"""Site balanced accuracy per block: logistic regression on each block pooled over the volume, subject-grouped CV.

This is the training-time version (one pooled code per subject). Phase 4 adds the per-region probe,
permutations and FDR.
"""

from __future__ import annotations

import numpy as np

from mlmv.eval.metrics import cv_balanced_accuracy


def site_accuracy_per_block(codes, sites, folds: int = 5) -> dict:
    """``codes[modality][level]["subject"] [n, d]``, ``sites [n]`` -> ``{modality: {level: balanced acc}}``."""
    groups = np.arange(len(sites))
    return {
        m: {lvl: cv_balanced_accuracy(c["subject"], sites, groups, folds) for lvl, c in md.items()}
        for m, md in codes.items()
    }
