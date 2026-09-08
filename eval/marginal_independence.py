"""Evaluation-only product of empirical content marginals, with images re-rendered."""

import hashlib
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def permute_content_marginals(content, seed=0):
    content = np.asarray(content)
    if content.ndim != 2 or min(content.shape) < 1 or len(content) < 2:
        raise ValueError("Marginal shuffling requires at least two samples and one factor.")
    if not np.isfinite(content).all():
        raise ValueError("Marginal shuffling requires finite content factors.")
    rng = np.random.RandomState(seed)
    permutations = np.column_stack([rng.permutation(len(content)) for _ in range(content.shape[1])])
    return content[permutations, np.arange(content.shape[1])], permutations


def _mean_abs_correlation(content):
    centered = content - content.mean(0)
    norm = np.linalg.norm(centered, axis=0)
    valid = norm > 0
    if valid.sum() < 2:
        return float("nan")
    standardized = centered[:, valid] / norm[valid]
    corr = standardized.T @ standardized
    return float(np.abs(corr[np.triu_indices(len(corr), 1)]).mean())


class MarginalShuffledDataset(Dataset):
    """Shuffle named content, keeping each subject's style, fields and noise seeds.

    ``source`` must be the training-matched SyntheticBrainDataset in pseudo_mri mode.
    The source's inner renderer is called once per subject to obtain its exact latent
    draw; only the small latent dictionaries are kept, not those source images/masks.
    This deliberately reuses the existing sampler rather than duplicating its SCM,
    uniform-prior transform, or nuisance draws. Re-rendered images may then be cached.
    Source normalization is reused, including its unshuffled fixed-reference constants.
    No SCM is exposed: the shuffled labels no longer satisfy the source causal graph.
    """

    def __init__(self, source, seed=0, cache=True):
        if len(source) < 2:
            raise ValueError("Marginal shuffling requires at least two samples.")
        np.random.RandomState(seed)  # Validate the seed before collecting latent draws.
        inner = getattr(source, "_inner", None)
        if inner is None or getattr(inner, "mode", None) != "pseudo_mri":
            raise ValueError("Marginal-preserving independence currently supports synthetic pseudo_mri only.")
        self.source = source
        self.latents = []
        for idx in range(len(source)):
            latent = dict(inner[idx][2])
            for key in ("brain_mask", "causal_adj", "z_global_atrophy", "z_content_residuals"):
                latent.pop(key, None)
            self.latents.append(latent)
            if (idx + 1) % 250 == 0:
                logger.info("Collecting matched latent draws for marginal shuffle: %d/%d", idx + 1, len(source))
        original = torch.stack([latent["z_content"] for latent in self.latents]).numpy()
        shuffled, permutations = permute_content_marginals(original, seed)
        self.content = torch.from_numpy(shuffled.copy())
        self._cache = [None] * len(source) if cache else None
        self.evaluation_distribution = dict(
            mode="shuffled",
            shuffle_seed=int(seed),
            n_samples=len(source),
            source_causal=bool(getattr(inner, "causal", False)),
            source_hierarchical=bool(getattr(inner, "hierarchical_content", False)),
            mean_abs_correlation_before=_mean_abs_correlation(original),
            mean_abs_correlation_after=_mean_abs_correlation(shuffled),
            source_content_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
            shuffled_content_sha256=hashlib.sha256(shuffled.tobytes()).hexdigest(),
            permutation_sha256=hashlib.sha256(permutations.astype("<i8").tobytes()).hexdigest(),
        )

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        if self._cache is not None and self._cache[idx] is not None:
            return self._cache[idx]
        latent = dict(self.latents[idx])
        latent["z_content"] = self.content[idx].clone()
        inner = self.source._inner
        x1, x2, mask = inner.render_pseudo_mri(
            latent["z_content"],
            latent["z_deformation"],
            latent["z_fissure"],
            latent["z_style_v1"],
            latent["z_style_v2"],
            inner.sample_seed_for(idx),
            z_lesion=latent.get("z_lesion"),
        )
        mask2 = mask.clone()
        x1, x2 = self.source.normalize_views(x1, x2, mask, mask2)
        result = dict(image=[x1, x2], mask=[mask, mask2], gt_latents=latent, z_image=[{}, {}], index=idx, label=0)
        if self._cache is not None:
            self._cache[idx] = result
        return result
