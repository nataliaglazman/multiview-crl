#!/usr/bin/env python
"""Export a label-free content block from a PCA normative model, in the shared bundle format.

    python -m eval.protocol.export_normative_bundle --run-dir results/synthetic/RUN \\
        --num-samples 2000 --out results/bundles/normative.npz

    # label-free PC on its channels, against the truth and against random mixings of the block
    python -m eval.causal.latent_causal_discovery --bundles normative=results/bundles/normative.npz \\
        --ceiling --mixing-null 30

No encoder weights are trained. The block comes from the population model the residual lesion branch
already reads (``models.normative_residual``), used here as a representation. The bundle is
written by ``eval.dino.dinov3_embed_synthetic.save``, so ``compare_bundles`` and
``dinov3_identifiability`` score it exactly as they score a VQ or DINO bundle of the same rows.

How the block is built
----------------------
1. **Per-view normative model.** For each view, the mean image and the first ``--components``
   principal modes, fitted on the run's TRAINING split (never the rows being scored): the
   first ``--fit-samples`` subjects, 2/3 for the modes and 1/3 for the residual SD, as
   ``main_conv_synthetic`` fits its lesion branch's. A subject's mode scores are its
   coordinates on those modes.
2. **Shared modes, not loud ones.** Variance is the wrong ranking: on the synthetic recipe
   the top mode of either view is the scanner gain (r ~ 0.8, about half the voxel variance).
   CCA between the two views' mode scores ranks directions by how strongly T1 and FLAIR
   agree on them instead. ``--shared-dims auto`` keeps every canonical pair whose
   correlation beats a permutation null (FLAIR rows shuffled against T1; the 95th
   percentile of the largest null correlation). The count is a block width, not a factor
   count: a boundary shift is nonlinear in voxels and needs several linear modes, so it
   overshoots the number of factors (15 for 6 anatomy factors on the causal recipe).
3. **A rotation that keeps the factors' correlations.** Canonical directions are
   uncorrelated by construction, so PC deletes nearly every edge between them at its first
   test and the graph over them says nothing about the truth. ``--rotation spatial_ica``
   rotates the block non-orthogonally so that each channel's loading map (the voxelwise
   covariance between the image and that channel) is as independent as possible across
   voxels: "each factor moves its own voxels". The subject scores stay free to correlate.
   This is an assumption, not a measurement. Factors that move the same voxels cannot be
   separated by it: cortical thickness rides brain size's shells.
4. **The lesion, from the residual.** A lesion sits somewhere different in every subject,
   so no population mode holds it. Three more channels are the brain-frame coordinates of
   the peak of the smoothed, unsigned residual z (``--lesion none`` drops them).

Views
-----
``--block per_view`` (default) writes T1's channels to ``emb_view1`` and FLAIR's to
``emb_view2``. Each comes from its own view's image alone, through one shared rotation, so
channel j means the same thing in both: comparable with an encoder's per-view content.
``--block unified`` writes the average of the two views' channels, and the residual peak of
both views summed, to ``emb_view1`` only. It reads both images per subject, so it is not
comparable with a single-view encoder; it is the stronger label-free reference.

``meta["rotated_channels"]`` names the columns the rotation chose, so
``latent_causal_discovery --mixing-null`` redraws exactly that choice: random
non-orthogonal mixings of those columns, the lesion columns left alone. That is the null
the rotation has to beat. The naming null alone is too lenient for this block, because
every candidate shares the same, correct lesion channels.

There is no untrained twin to use as a floor: read the probe tables against their
permutation nulls, and the graph against the mixing null.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ROTATIONS = ("spatial_ica", "promax", "none")
BLOCKS = ("per_view", "unified")
LESION_MODES = ("peak", "none")
#: Gaussian width, in voxels, of the smoothing applied to the residual before its peak is read.
LESION_SIGMA = 1.0
#: The null threshold: this quantile of the largest canonical correlation under shuffled pairs.
NULL_QUANTILE = 0.95


def parse_shared_dims(text):
    if text == "auto":
        return text
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--shared-dims must be 'auto' or a positive integer, got {text!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("--shared-dims must be positive")
    return value


# --------------------------------------------------------------------------- #
# The block, on plain arrays (torch-free; --self-test runs these)
# --------------------------------------------------------------------------- #


def standardizer(scores):
    """Column mean and SD of the fitting rows, as a function applied to any rows."""
    import numpy as np

    scores = np.asarray(scores, dtype=np.float64)
    mean, sd = scores.mean(0), scores.std(0)
    sd = np.where(sd > np.finfo(float).eps, sd, 1.0)
    return lambda rows: (np.asarray(rows, dtype=np.float64) - mean) / sd


def cca(a, b, reg=1e-6):
    """Canonical directions ``(wa, wb)`` and correlations of two standardized score blocks.

    ``a @ wa[:, j]`` and ``b @ wb[:, j]`` are the j-th canonical pair: unit variance on these
    rows, correlated at ``rho[j]`` with each other and uncorrelated with every other pair.
    """
    import numpy as np

    n = len(a)
    caa = a.T @ a / n + reg * np.eye(a.shape[1])
    cbb = b.T @ b / n + reg * np.eye(b.shape[1])
    ia = np.linalg.inv(np.linalg.cholesky(caa)).T
    ib = np.linalg.inv(np.linalg.cholesky(cbb)).T
    u, rho, vt = np.linalg.svd(ia.T @ (a.T @ b / n) @ ib)
    return ia @ u, ib @ vt.T, rho


def shared_dims(a, b, permutations=100, seed=0, quantile=NULL_QUANTILE):
    """Canonical correlations, the null threshold, and how many pairs beat it.

    The null breaks the subject pairing and nothing else: FLAIR rows are shuffled against
    T1 and the CCA refitted, so a null correlation is what this many modes reach by chance.
    Each rank is compared with the null's LARGEST correlation, which keeps the count from
    growing with the number of ranks tested.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    rho = cca(a, b)[2]
    null = [cca(a, b[rng.permutation(len(b))])[2][0] for _ in range(permutations)]
    threshold = float(np.quantile(null, quantile))
    return rho, threshold, int((rho > threshold).sum())


def varimax(loadings, iterations=500, tol=1e-10):
    """Orthogonal rotation maximising the variance of the squared loadings (raw, no Kaiser)."""
    import numpy as np

    p, k = loadings.shape
    rotation = np.eye(k)
    total = 0.0
    for _ in range(iterations):
        lam = loadings @ rotation
        u, s, vt = np.linalg.svd(loadings.T @ (lam**3 - lam @ np.diag(np.diag(lam.T @ lam)) / p))
        rotation = u @ vt
        previous, total = total, s.sum()
        if previous and total / previous < 1 + tol:
            break
    return rotation


def promax(loadings, power=4):
    """Oblique rotation ``T`` (pattern ``loadings @ T``) from a sharpened varimax target, as R's ``promax``."""
    import numpy as np

    rotation = varimax(loadings)
    lam = loadings @ rotation
    target = lam * np.abs(lam) ** (power - 1)
    oblique = np.linalg.lstsq(lam, target, rcond=None)[0]
    oblique = oblique @ np.diag(np.sqrt(np.diag(np.linalg.inv(oblique.T @ oblique))))
    return rotation @ oblique


def rotation_matrix(loadings, kind, seed=0):
    """``R`` with ``channels = u @ R`` for whitened variates ``u`` whose loading maps are ``loadings``.

    With ``u`` whitened, the image is ``x ~ u @ loadings.T``, so a rotation ``T`` of the
    loadings gives the channels ``u @ inv(T).T``; spatial ICA's mixing matrix is already
    the map from ``u`` to the channels.
    """
    import numpy as np

    m = loadings.shape[1]
    if kind == "none" or m == 1:
        return np.eye(m)
    if kind == "promax":
        return np.linalg.inv(promax(loadings)).T
    if kind == "spatial_ica":
        from sklearn.decomposition import FastICA

        # Voxels are the samples: the maps, not the subjects, are made independent.
        ica = FastICA(n_components=m, whiten="unit-variance", random_state=seed, max_iter=2000)
        ica.fit(loadings)
        return np.asarray(ica.mixing_, dtype=np.float64)
    raise ValueError(f"rotation must be one of {ROTATIONS}, got {kind!r}")


def finish_rotation(rotation, u_fit, loadings):
    """Unit-variance channels, loudest first, each signed so its loading map peaks positive.

    Label-free and deterministic, so two exports agree on channel order and sign. Loudness
    is the norm of the channel's pattern, the voxel variance it accounts for.
    """
    import numpy as np

    rotation = rotation / (u_fit @ rotation).std(0)
    pattern = loadings @ np.linalg.inv(rotation).T
    order = np.argsort(-np.linalg.norm(pattern, axis=0), kind="stable")
    rotation, pattern = rotation[:, order], pattern[:, order]
    signs = np.sign(pattern[np.abs(pattern).argmax(0), np.arange(pattern.shape[1])])
    return rotation * np.where(signs == 0, 1.0, signs)


def brain_frame_peak(magnitude, support, sigma=LESION_SIGMA):
    """Brain-frame coordinates of the peak of a smoothed 3-D map, searched inside ``support``.

    The frame is the subject's own: the peak voxel minus the support's centroid, divided by
    its spread per axis, so a coordinate does not move with brain size alone.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    support = np.asarray(support, dtype=bool)
    if not support.any():
        return np.zeros(3)
    smooth = gaussian_filter(np.asarray(magnitude, dtype=np.float64), sigma)
    smooth[~support] = -np.inf
    peak = np.asarray(np.unravel_index(int(np.argmax(smooth)), smooth.shape), dtype=np.float64)
    coords = np.argwhere(support).astype(np.float64)
    spread = coords.std(0)
    return (peak - coords.mean(0)) / np.where(spread > 0, spread, 1.0)


# --------------------------------------------------------------------------- #
# Generator and normative model
# --------------------------------------------------------------------------- #


def generator_split(run_dir, split, num_samples, causal):
    """``(dataset, settings)`` for one split of the run's generator, VQ-VAE or encoder-only.

    Built by the same factory each run type is scored with, so the rows are the rows that
    run's own exports hold. ``settings`` is what the bundle's generator digest covers.
    """
    from eval.causal.latent_causal_discovery import is_encoder_only

    settings = json.loads((Path(run_dir) / "settings.json").read_text())
    if is_encoder_only(settings):
        from eval.protocol.score_checkpoint import make_dataset

        if not causal:
            settings = dict(settings, synthetic_causal=False)
        dataset = make_dataset(settings, num_samples, mode=split)
        generator = {key: value for key, value in settings.items() if key.startswith("synthetic_")}
        # The four rows-defining settings the encoder trainer stores under its own names.
        seed = settings.get("data_seed")
        generator.update(
            synthetic_seed=seed if seed is not None else settings.get("seed", 42),
            synthetic_res=settings["res"],
            synthetic_n_content=settings["n_content"],
            synthetic_n_style=settings["n_style"],
        )
    else:
        from eval.protocol.run_dci_synthetic import build_synthetic_test_set, load_run_args

        args = load_run_args(run_dir)
        dataset = build_synthetic_test_set(args, num_samples, cache=False, causal=causal, split=split)
        generator = {
            key: value
            for key, value in vars(args).items()
            if key.startswith("synthetic_") and not key.startswith("synthetic_num_")
        }
    generator = {key: value for key, value in generator.items() if not key.startswith("synthetic_num_")}
    generator["synthetic_num_samples"] = len(dataset)
    return dataset, generator


def _batches(dataset, batch_size, num_workers):
    import torch

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


def fit_images(dataset, batch_size, num_workers):
    """The fitting subjects' brain voxels, ``(N, 2, V)``, and where those voxels sit in the grid.

    Two passes, because the support is the union over every fitting subject and both views,
    and holding full grids to find it would cost ~3x the memory of the compact array.
    """
    import numpy as np

    support = None
    for batch in _batches(dataset, batch_size, num_workers):
        inside = ((batch["image"][0] != 0) | (batch["image"][1] != 0)).flatten(1).any(0).numpy()
        support = inside if support is None else support | inside
    index = np.flatnonzero(support)
    images = np.empty((len(dataset), 2, len(index)), dtype=np.float32)
    row = 0
    for batch in _batches(dataset, batch_size, num_workers):
        n = len(batch["image"][0])
        for view in range(2):
            images[row : row + n, view] = batch["image"][view].flatten(1).numpy()[:, index]
        row += n
    return images, index, tuple(batch["image"][0].shape[2:])


def normative_model(images, components):
    """``models.normative_residual.NormativeResidual`` itself, fitted on the compact voxels.

    The brain voxels are laid out as a (V, 1, 1) volume. The class only masks, projects and
    divides, so it is indifferent to the layout, and this block reads the same population
    model the lesion branch reads: first 2/3 for the modes, last 1/3 for the residual SD.
    """
    import torch

    from models.normative_residual import NormativeResidual

    n, _, v = images.shape
    split = (2 * n) // 3
    if split <= components:
        raise SystemExit(f"--fit-samples {n} leaves {split} subjects for {components} modes; need more than that")
    model = NormativeResidual(2, components, (v, 1, 1))
    data = torch.from_numpy(images).view(n, 2, v, 1, 1)
    model.fit(data[:split], data[split:])
    return model, split


def mode_scores(model, compact, view, chunk=256):
    """``(B, k)`` coordinates of compact ``(B, V)`` images on one view's modes.

    In row chunks, so scoring the fitting set never holds a float64 copy of all of it.
    """
    import numpy as np

    mean = model.mean[view, 0].flatten().double().numpy()
    basis = model.basis[view].flatten(1).double().numpy()
    return np.concatenate(
        [(np.asarray(compact[i : i + chunk], dtype=np.float64) - mean) @ basis.T for i in range(0, len(compact), chunk)]
    )


def variance_fractions(model, compact, view, scores, chunks=16):
    """Each mode's share of the total voxel variance of these images."""
    import numpy as np

    mean = model.mean[view, 0].flatten().numpy()
    total = sum(
        float(((compact[:, part] - mean[part]).astype(np.float64) ** 2).sum())
        for part in np.array_split(np.arange(compact.shape[1]), chunks)
    )
    return (scores**2).sum(0) / total


def residual_z(model, compact, view):
    """``(B, V)`` residual z of compact images, through the class's own ``forward``."""
    import torch

    x = torch.from_numpy(compact.astype("float32")).view(len(compact), 1, -1, 1, 1)
    with torch.no_grad():
        return model(x, view).flatten(1).double().numpy()


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def build(cli):
    """Fit everything on the training split, then encode the scored split.

    Returns ``(embeddings, latents, meta)``.
    """
    import numpy as np

    from eval.dino.dinov3_embed_synthetic import git_sha
    from eval.metrics.bundle_identity import identity_record
    from eval.metrics.dci import STYLE_FACTOR_NAMES, content_factor_names, dataset_lesion_target
    from eval.protocol.identifiability_report import _causal_adjacency

    started = time.time()
    # The training subjects as the run trained on them, whatever --causal asks of the scored rows.
    fit_set, _ = generator_split(cli.run_dir, "train", cli.fit_samples, True)
    dataset, generator = generator_split(cli.run_dir, cli.split, cli.num_samples, cli.causal)

    logger.info("Fitting the normative model on %d training subjects", len(fit_set))
    images, index, grid = fit_images(fit_set, cli.batch_size, cli.num_workers)
    model, n_modes = normative_model(images, cli.components)
    scores = [mode_scores(model, images[:, view], view) for view in range(2)]
    explained = {
        f"view{view + 1}": variance_fractions(model, images[:n_modes, view], view, scores[view][:n_modes])
        .round(4)
        .tolist()
        for view in range(2)
    }

    scale_a, scale_b = standardizer(scores[0]), standardizer(scores[1])
    a, b = scale_a(scores[0]), scale_b(scores[1])
    wa, wb, _ = cca(a, b)
    rho, threshold, m_auto = shared_dims(a, b, cli.null_permutations, cli.seed)
    m = m_auto if cli.shared_dims == "auto" else cli.shared_dims
    if m < 1:
        raise SystemExit(
            f"No canonical pair beats the permutation null ({threshold:.3f}): the views share no linear "
            "structure at this --components; pass --shared-dims to force a width"
        )
    if m > cli.components:
        raise SystemExit(f"--shared-dims {m} exceeds --components {cli.components}")
    logger.info(
        "Canonical correlations %s; null %.3f -> %d shared dims (using %d)",
        np.round(rho, 3).tolist(),
        threshold,
        m_auto,
        m,
    )

    # The unified variates of the fitting subjects, whitened, and their voxel loading maps over
    # both views. u_fit is centred on these rows, so the images need no centring here.
    pair = (a @ wa[:, :m] + b @ wb[:, :m]) / 2
    whiten = 1.0 / pair.std(0)
    u_fit = pair * whiten
    u32 = u_fit.astype(np.float32)
    loadings = np.concatenate([images[:, view].T @ u32 for view in range(2)]).astype(np.float64) / len(u_fit)
    rotation = finish_rotation(rotation_matrix(loadings, cli.rotation, cli.seed), u_fit, loadings)
    to_channels = np.diag(whiten) @ rotation
    del images

    logger.info("Encoding %d %s subjects", len(dataset), cli.split)
    views = (1,) if cli.block == "unified" else (1, 2)
    parts = {view: [] for view in views}
    held_out = {"P": [], "Q": []}
    gt = {"z_content": [], "z_style_v1": [], "z_style_v2": []}
    for batch in _batches(dataset, cli.batch_size, cli.num_workers):
        full = [batch["image"][view][:, 0].numpy() for view in range(2)]
        compact = [volume.reshape(len(volume), -1)[:, index] for volume in full]
        P = scale_a(mode_scores(model, compact[0], 0)) @ wa[:, :m]
        Q = scale_b(mode_scores(model, compact[1], 1)) @ wb[:, :m]
        held_out["P"].append(P)
        held_out["Q"].append(Q)
        channels = (
            {1: (P + Q) / 2 @ to_channels} if cli.block == "unified" else {1: P @ to_channels, 2: Q @ to_channels}
        )
        if cli.lesion == "peak":
            magnitude = []
            for view in range(2):
                z = np.zeros((len(full[view]), int(np.prod(grid))))
                z[:, index] = np.abs(residual_z(model, compact[view], view))
                magnitude.append(z.reshape(len(z), *grid))
            if cli.block == "unified":
                peaks = {
                    1: [
                        brain_frame_peak(t1_mag + fl_mag, (t1 != 0) | (fl != 0))
                        for t1_mag, fl_mag, t1, fl in zip(*magnitude, *full)
                    ]
                }
            else:
                peaks = {
                    view + 1: [brain_frame_peak(mag, vol != 0) for mag, vol in zip(magnitude[view], full[view])]
                    for view in range(2)
                }
            channels = {view: np.column_stack([channels[view], np.asarray(peaks[view])]) for view in views}
        for view in views:
            parts[view].append(channels[view])
        for key in gt:
            gt[key].append(batch["gt_latents"][key].numpy())

    embeddings = {view: np.concatenate(parts[view]) for view in views}
    latents = {key: np.concatenate(value) for key, value in gt.items()}
    adjacency = _causal_adjacency(dataset)
    if adjacency is not None:
        latents["causal_adj"] = np.asarray(adjacency)
    P, Q = np.concatenate(held_out["P"]), np.concatenate(held_out["Q"])
    rho_scored = [float(np.corrcoef(P[:, j], Q[:, j])[0, 1]) for j in range(m)]
    n_lesion = 3 if cli.lesion == "peak" else 0
    n_content = latents["z_content"].shape[1]
    if m + n_lesion < n_content:
        logger.warning(
            "%d channels for %d factors: latent_causal_discovery needs at least one channel per factor",
            m + n_lesion,
            n_content,
        )

    meta = dict(
        model_id=f"normative:{Path(cli.run_dir).name}",
        backbone="normative_pca",
        source="eval.protocol.export_normative_bundle",
        run_dir=str(cli.run_dir),
        block=cli.block,
        rotation=cli.rotation,
        lesion=cli.lesion,
        lesion_sigma=LESION_SIGMA if n_lesion else None,
        components=cli.components,
        fit_split="train",
        fit_samples=len(fit_set),
        mode_subjects=n_modes,
        calibration_subjects=len(fit_set) - n_modes,
        support_voxels=int(len(index)),
        mode_variance_fraction=explained,
        shared_dims=m,
        shared_dims_auto=m_auto,
        canonical_correlations=[round(float(r), 4) for r in rho],
        canonical_correlations_scored=[round(r, 4) for r in rho_scored],
        null_threshold=round(threshold, 4),
        null_permutations=cli.null_permutations,
        rotated_channels=list(range(m)),
        lesion_channels=list(range(m, m + n_lesion)),
        seed=cli.seed,
        split=cli.split,
        pooling="none",
        n_content_channels=m + n_lesion,
        n_style_channels=0,
        has_split=False,
        embedding_dim={f"view{view}": int(features.shape[1]) for view, features in embeddings.items()},
        views=list(embeddings),
        num_samples=int(len(dataset)),
        causal=adjacency is not None,
        content_factor_names=content_factor_names(n_content, dataset_lesion_target(dataset)),
        style_factor_names=STYLE_FACTOR_NAMES[: latents["z_style_v1"].shape[1]],
        generator=generator,
        git_sha=git_sha(),
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        elapsed_seconds=round(time.time() - started, 1),
        **identity_record(latents, generator, len(dataset), cli.distribution),
    )
    return embeddings, latents, meta


# --------------------------------------------------------------------------- #
# Self-test and CLI
# --------------------------------------------------------------------------- #


def _abs_corr(x, z):
    import numpy as np

    return np.abs(np.corrcoef(x, z, rowvar=False)[: x.shape[1], x.shape[1] :])


def _matched(x, z):
    """Hungarian-matched |corr| per factor, as ``channel_mcc`` scores it, and each factor's channel."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    corr = _abs_corr(x, z)
    rows, cols = linear_sum_assignment(-corr.T)
    channel = np.empty(z.shape[1], dtype=int)
    channel[rows] = cols
    return corr.T[rows, cols], channel


def _self_test():
    """The block's pieces on planted arrays: shared width, rotation, lesion peak. No torch."""
    import numpy as np

    failures = []

    def check(label, condition):
        if not condition:
            failures.append(label)
        print(f"  {'ok  ' if condition else 'FAIL'}  {label}")

    rng = np.random.default_rng(0)
    n, d = 3000, 3

    # Three correlated factors (a chain), each moving its own block of 150 "voxels" with a
    # sparse, heavy-tailed map. Two views render them with different maps, plus private style.
    z = np.zeros((n, d))
    for j in range(d):
        z[:, j] = rng.standard_normal(n) + (0.8 * z[:, j - 1] if j else 0.0)
    voxels = 150 * d

    def view_images(seed):
        r = np.random.default_rng(seed)
        maps = np.zeros((d, voxels))
        for j in range(d):
            maps[j, 150 * j : 150 * (j + 1)] = r.laplace(size=150) * (r.random(150) < 0.3)
        style = r.standard_normal((n, 2)) @ r.standard_normal((2, voxels))
        return z @ maps + style + 0.3 * r.standard_normal((n, voxels))

    x1, x2 = view_images(1), view_images(2)

    def pca_scores(x, k=8):
        centred = x - x.mean(0)
        return centred @ np.linalg.svd(centred, full_matrices=False)[2][:k].T

    a, b = (standardizer(s)(s) for s in (pca_scores(x1), pca_scores(x2)))
    rho, threshold, m = shared_dims(a, b, permutations=40)
    check(f"the null counts the three shared factors (m={m}, rho={np.round(rho[:4], 2).tolist()})", m == d)
    check(f"shared pairs clear the private ones by a wide margin (null {threshold:.2f})", rho[2] - rho[3] > 0.5)

    wa, wb, _ = cca(a, b)
    pair = (a @ wa[:, :m] + b @ wb[:, :m]) / 2
    u = pair / pair.std(0)
    centred = np.concatenate([x1 - x1.mean(0), x2 - x2.mean(0)], axis=1)
    loadings = centred.T @ u / n
    upper = np.triu_indices(d, 1)
    corr_true = np.abs(np.corrcoef(z, rowvar=False))[upper]
    for kind in ("none", "spatial_ica", "promax"):
        channels = u @ finish_rotation(rotation_matrix(loadings, kind), u, loadings)
        matched, channel = _matched(channels, z)
        corr = np.abs(np.corrcoef(channels[:, channel], rowvar=False))[upper]
        listed = np.round(matched, 2).tolist()
        error = np.abs(corr - corr_true).max()
        if kind == "none":
            check(f"unrotated canonical channels are uncorrelated (max |r| {corr.max():.2f})", corr.max() < 0.05)
            check(f"... so they are not the factors (matched {listed})", matched.min() < 0.9)
        elif kind == "spatial_ica":
            check(f"spatial_ica recovers each factor as one channel (matched {listed})", matched.min() > 0.95)
            check(f"spatial_ica keeps the factors' correlations (max |r| error {error:.2f})", error < 0.05)
        else:
            check(f"promax recovers each factor as one channel (matched {listed})", matched.min() > 0.95)
            # Not a guarantee, so not a check: the sharpened target bends the correlations,
            # which is why spatial_ica is the default for an export a graph is read from.
            print(f"  info  promax bends the factors' correlations (max |r| error {error:.2f})")
    rotation = finish_rotation(rotation_matrix(loadings, "spatial_ica"), u, loadings)
    channels = u @ rotation
    check("channels come out at unit variance", np.allclose(channels.std(0), 1.0))
    again = finish_rotation(rotation_matrix(loadings, "spatial_ica", seed=5), u, loadings)
    check("order and sign do not depend on the ICA seed", np.abs(rotation - again).max() < 0.05)
    to_channels = np.diag(1 / pair.std(0)) @ rotation
    t1, fl = (a @ wa[:, :m]) @ to_channels, (b @ wb[:, :m]) @ to_channels
    check("the unified block is the mean of the per-view blocks", np.allclose((t1 + fl) / 2, channels))

    # A blob in a noisy volume, inside an off-centre support.
    grid = (24, 24, 24)
    support = np.zeros(grid, dtype=bool)
    support[4:20, 6:22, 2:18] = True
    volume = np.abs(rng.standard_normal(grid))
    centre = np.array([15, 9, 12])
    volume[tuple(centre)] += 30.0
    peak = brain_frame_peak(volume, support)
    coords = np.argwhere(support).astype(float)
    check(
        "the residual peak lands on the blob, in brain-frame units",
        np.allclose(peak, (centre - coords.mean(0)) / coords.std(0)),
    )
    outside = volume.copy()
    outside[1, 1, 1] += 100.0
    check("a peak outside the support is ignored", np.allclose(brain_frame_peak(outside, support), peak))

    print("\n" + ("FAILED: " + ", ".join(failures) if failures else "All checks passed."))
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", help="Synthetic training run (VQ-VAE or encoder-only) whose generator to use")
    parser.add_argument("--out", type=Path, help="Destination .npz")
    parser.add_argument("--num-samples", type=int, default=2000, help="Scored rows")
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "val"],
        help="Scored split. 'test' matches export_vq_bundle; the fit always uses the training split",
    )
    parser.add_argument(
        "--causal",
        default="match",
        choices=["match", "iid"],
        help="'match' reproduces the run's training SCM; 'iid' forces independent factors (no graph to score)",
    )
    parser.add_argument("--fit-samples", type=int, default=1200, help="Training subjects: 2/3 modes, 1/3 residual SD")
    parser.add_argument("--components", type=int, default=20, help="Principal modes per view")
    parser.add_argument(
        "--shared-dims", type=parse_shared_dims, default="auto", help="'auto' (permutation null) or a fixed width"
    )
    parser.add_argument("--null-permutations", type=int, default=100)
    parser.add_argument("--rotation", default="spatial_ica", choices=ROTATIONS)
    parser.add_argument("--block", default="per_view", choices=BLOCKS)
    parser.add_argument("--lesion", default="peak", choices=LESION_MODES)
    parser.add_argument("--seed", type=int, default=0, help="Permutation null and ICA initialisation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--self-test", action="store_true", help="Run the torch-free checks and exit")
    cli = parser.parse_args(argv)
    if cli.self_test:
        return _self_test()
    if not cli.run_dir or not cli.out:
        parser.error("--run-dir and --out are required")
    if cli.components < 1 or cli.null_permutations < 1:
        parser.error("--components and --null-permutations must be positive")
    if cli.num_samples < 2 or cli.batch_size < 1 or cli.num_workers < 0:
        parser.error("--num-samples must be at least 2, --batch-size positive, and --num-workers nonnegative")
    mode_subjects = (2 * cli.fit_samples) // 3
    if mode_subjects <= cli.components or cli.fit_samples - mode_subjects < 2:
        parser.error(
            "--fit-samples must leave more subjects than --components for modes and at least 2 for residual SD"
        )
    if cli.shared_dims != "auto" and cli.shared_dims > cli.components:
        parser.error("--shared-dims cannot exceed --components")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    cli.distribution = cli.causal
    cli.causal = cli.causal == "match"

    from eval.dino.dinov3_embed_synthetic import save

    embeddings, latents, meta = build(cli)
    arrays = save(cli.out, embeddings, latents, {}, [], meta)
    print(f"\nSaved {cli.out.resolve()}")
    for key, value in sorted(arrays.items()):
        if key != "meta":
            print(f"  {key:<14} {value.shape}")
    print(
        f"  shared dims    {meta['shared_dims']} (null {meta['null_threshold']}); held-out canonical r "
        f"{meta['canonical_correlations_scored'][:6]}"
    )
    print(f"  factor digest  {meta['factor_digest'][:16]}  ({meta['n_rows']} rows)")
    print(
        f"\nGraph, label-free:\n  python -m eval.causal.latent_causal_discovery --bundles normative={cli.out} "
        "--ceiling --mixing-null 30"
    )
    print(f"Probes:\n  python -m eval.protocol.compare_bundles --bundles normative={cli.out} vq=<vq.npz>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
