#!/usr/bin/env python
"""Does the bottleneck code move when only one factor moves? An intervention test for ``z``.

    python -m eval.protocol.spatial_bottleneck_interventions \\
        --bottleneck-dirs results/bottleneck_d16 [results/bottleneck_d16_s1 results/bottleneck_d32 ...] \\
        --out-dir results/bottleneck_interventions --num-subjects 128 --eps 0.5

``eval.protocol.spatial_bottleneck_eval`` probes each factor from ``z`` on held-out subjects, so a
factor can score through another one that varies with it: under ``--synthetic-causal`` the factors
are correlated, and ``wm_interior`` places the lesion relative to the white matter around it. Here
each test subject is rendered twice, with one factor at ``z_j - eps`` and at ``z_j + eps`` and every
other latent held, so the anatomy cannot change with the lesion. If ``z`` responds when only the
lesion moves, and that response cannot be mistaken for its response to brain size, ventricle size
or any other factor, then ``z`` encodes the lesion itself.

The representations are those of ``spatial_bottleneck_eval`` (``map``, ``patch2x2x2``, ``gap``,
``pca{d}``, ``z_init:RUN``, ``z:RUN``), computed from the same frozen VQ-VAE content maps.

Protocol
--------
* **The bottleneck's own test subjects.** The first ``--num-subjects`` of the regenerated test split
  (default: the bottleneck's test count; a larger number extends the split from the same seed, as
  ``spatial_bottleneck_eval --test-samples`` does). The regenerated maps must reproduce the saved
  test codes, GAP vectors and validation loss, exactly as in that script, and re-rendering each
  subject from its latents must reproduce the dataset's images bit for bit. A mismatch stops the run.
* **Interventions.** Every latent except the moved one is held: the other content dimensions, the
  deformation and fissure fields, the lesion field or placement draws, both style vectors and the
  render seed, so both conditions share their bias field and noise. Under ``--synthetic-causal`` the
  moved factor's descendants are held as well (an intervention, not conditioning), which can leave
  the SCM's support. A style target moves its coordinate in both views; each view's code sees only
  its own scan, so a view's response is that view's sensitivity. That is leakage for per-view style,
  but a style factor identical in both views (gain and bias under ``synthetic_shared_gain_bias``) is
  shared information, which a cross-view code is expected to carry. The two nuisance targets move
  the deformation and fissure fields along a fixed random unit direction.
* **Where the lesion went.** With a single sphere lesion and nine content factors, each content and
  nuisance intervention also records the rendered lesion centre at both endpoints
  (``eval.encoder.encoder_target_protocol.sample_targets``, generator coordinates), written per
  subject to ``lesion_moves.csv`` with the displacement in voxels and whether the centre crossed the
  midline. Under ``wm_interior`` the lesion latents are not coordinates but conditional quantiles
  (x, then y given x, then z given x and y) through each subject's own white matter. So a nudge can
  leave the centre on its voxel (an unchanged image, not a blind code), a lesion near the median x
  quantile can jump hemispheres, moving ``lesion_x`` shifts the centre in y and z too, and an anatomy
  intervention can move the lesion; confusions among them start there. ``snr_render_per_voxel``
  divides a lesion coordinate's response by how far the centre actually moved.
* **Responses** ``f(x(z_j + eps)) - f(x(z_j - eps))`` per subject and view, in the units each probe of
  ``spatial_bottleneck_eval`` sees: the map divided by the bottleneck's per-channel scale; every other
  representation divided by its training-subject standard deviation in that view, with constant
  columns dropped. PCA components are fitted on both views' training maps, as there. The linear
  readouts (map, patch, GAP, PCA) are applied to the change in the map, so a change one cannot see
  reads exactly zero. A subject whose images an intervention leaves bit-identical (a quantised
  lesion centre that did not move, a clamped style value) is not measured for that factor: its
  response is set to 0, which the metrics skip, per-sample SNRs average over the measured subjects
  only, and ``active_frac`` reports the share that moved.
* **Nulls.** The render null re-renders the same latents with another render seed (bias field and
  noise redrawn). The nuisance null is the largest deformation/fissure response. They are biased in
  opposite directions. The two views of a subject differ in modality, style, bias field and noise,
  so a code trained for cross-view agreement learns to ignore the render null; that shrinks its
  denominator and flatters ``z``. The nuisance fields are shared by both views, which is what such a
  code keeps; that inflates the denominator and flatters the frozen readouts. Under
  ``--synthetic-clean-content`` the renderer zeroes the nuisance fields, so the nuisance null does
  not exist; rely on the scale-free metrics (recall, aliasing, consistency) there.
* **Uncertainty.** SNR intervals come from a paired bootstrap over subjects, with the same draws for
  every representation, factor and null. Each view is scored separately; tables average the two.

Reported per representation and factor
--------------------------------------
snr_render, snr_nuisance  mean response norm over the null's. Both conditions share the render seed,
                          so a representation that ignores the factor reads 0, and 1 means the
                          factor moves it as much as redrawing the noise does.
snr_render_per_voxel      lesion coordinates only: snr_render per voxel of actual centre displacement.
pop_snr                   norm of the mean response over the sign-flip render-null floor: whether
                          subjects share a response direction. A code with a lesion coordinate does;
                          a local map change at a subject-specific site does not.
consistency               leave-one-out cosine of each response with the mean direction.
recall, id_acc            cross-validated nearest-centroid accuracy at naming which content factor
                          moved from one subject's response (chance 1/J); ``mistaken for`` is the
                          factor its moves are most often taken to be.
alias                     |cos| between the mean responses of two content factors, against the floor
                          that two unrelated directions reach at this N.
rank                      how many content factors give separable mean directions.

The metrics are the functions of ``eval.protocol.interventional_identifiability``. Interventions that
change no image (``unused_*`` under the burden target, dimensions the renderer never reads, or the
nuisance fields under clean content) are reported but left out of identification and aliasing.

What this does not show
-----------------------
A response shows that the code changes when the lesion moves, not that a probe reads the right
coordinate from it; ``spatial_bottleneck_eval`` measures that. The mean-direction metrics (pop_snr,
alias, rank) assume subjects share a response direction. In the full map a lesion move is a local
change at a subject-specific site, so read snr_render and recall there.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np

from eval.protocol import interventional_identifiability as interventional
from eval.protocol import spatial_bottleneck_eval as observational

NULL = "__null__"
BOOTSTRAPPED = ("snr_render", "snr_nuisance", "snr_render_per_voxel")
VIEW_MEANS = ("response", *BOOTSTRAPPED, "pop_snr", "pop_snr_nuisance", "consistency")


def _ratio(numerator, denominator):
    numerator, denominator = np.asarray(numerator, dtype=np.float64), np.asarray(denominator, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0), np.nan)


def _mean(values, axis=None):
    stacked = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stacked, axis=axis)


# --------------------------------------------------------------------------- #
# Interventions (torch)
# --------------------------------------------------------------------------- #


def build_targets(content_names, style_names, latents, seed=0):
    """Every intervention: content and style coordinates, then the two nuisance fields.

    Nuisance directions are fixed unit-norm draws shaped like this generator's own fields, so the
    GP priors (whose fields live on ``field_grid``) are covered as well as the i.i.d. grids.
    """
    import torch

    targets = [dict(name=n, kind="content", block="content", index=j) for j, n in enumerate(content_names)]
    targets += [dict(name=n, kind="style", block="style", index=k) for k, n in enumerate(style_names)]
    generator = torch.Generator().manual_seed(seed)
    for block in ("z_deformation", "z_fissure"):
        direction = torch.randn(latents[block].shape, generator=generator, dtype=latents[block].dtype)
        targets.append(
            dict(
                name=interventional.NUISANCE_PREFIX + block,
                kind="nuisance",
                block=block,
                direction=direction / direction.norm(),
            )
        )
    return targets


def shifted(latents, target, delta):
    """``latents`` with one target moved by ``delta``; a style target moves that coordinate in both views."""
    moved = interventional._perturb(latents, target["block"], target.get("index"), delta, target.get("direction"))
    if target["block"] == "style":
        moved["z_style_v2"][target["index"]] = moved["z_style_v2"][target["index"]] + delta
    return moved


def render_pair(dataset, latents, seed):
    """A normalised view pair and its mask, exactly as ``SyntheticBrainDataset`` renders these latents."""
    x1, x2, mask = dataset._inner.render_pseudo_mri(
        latents["z_content"],
        latents["z_deformation"],
        latents["z_fissure"],
        latents["z_style_v1"],
        latents["z_style_v2"],
        seed,
        z_lesion=latents.get("z_lesion"),
    )
    x1, x2 = dataset.normalize_views(x1, x2, mask, mask.clone())
    return x1, x2, mask


def lesion_centre(generator, latents):
    """Rendered lesion centre ``(3,)`` from ``sample_targets``; NaN when the lesion is empty."""
    import torch

    from eval.encoder.encoder_target_protocol import sample_targets

    try:
        return torch.from_numpy(sample_targets(generator, latents)[0][9:12]).double()
    except ValueError:
        return torch.full((3,), math.nan, dtype=torch.float64)


class ConditionRenders:
    """Subject ``i`` at ``-eps`` (item ``2i``) and at ``+eps`` (item ``2i + 1``), as a map-style dataset.

    ``target=None`` is the render null: the same latents at the original and at a shifted render
    seed. A subject whose moved anatomy leaves no room for its lesion (``wm_interior``) comes back
    blank with ``ok=False`` and is dropped from every condition afterwards. With ``generator`` set,
    content and nuisance interventions also return the rendered lesion centre.
    """

    def __init__(self, dataset, base, target, eps, null_offset, generator=None):
        self.dataset, self.base, self.target, self.eps, self.null_offset = dataset, base, target, eps, null_offset
        moves_structure = target is not None and target["kind"] in ("content", "nuisance")
        self.generator = generator if moves_structure else None

    def __len__(self):
        return 2 * len(self.base)

    def __getitem__(self, item):
        import torch

        from eval.synthetic.synthetic_dataset import LesionPlacementError

        subject, side = divmod(item, 2)
        latents, seed = self.base[subject]
        if self.target is None:
            seed += side * self.null_offset
        else:
            latents = shifted(latents, self.target, (2 * side - 1) * self.eps)
        try:
            x1, x2, mask = render_pair(self.dataset, latents, seed)
            ok = True
        except LesionPlacementError:
            x1 = x2 = mask = torch.zeros(1, *(self.dataset.res,) * 3)
            ok = False
        out = {"image": [x1, x2], "mask": [mask, mask.clone()], "ok": ok}
        if self.generator is not None:
            out["centre"] = lesion_centre(self.generator, latents) if ok else torch.full((3,), math.nan).double()
        return out


class PairTap:
    """Passes interleaved ``(-eps, +eps)`` batches to the label-free map extractor.

    Keeps, per subject, whether both renders succeeded, how far the image moved (L2 over both
    views) and, when rendered, the lesion centre at both endpoints. An intervention that leaves an
    image bit-identical was not an intervention for that subject.
    """

    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset
        self.ok, self.image_change, self.centres = [], [], []

    def __iter__(self):
        import torch

        for batch in self.loader:
            ok = batch.pop("ok")
            if len(ok) % 2:
                raise ValueError("Each batch must hold whole (-eps, +eps) pairs; use an even batch size")
            images = torch.cat([view.flatten(1) for view in batch["image"]], 1).double()
            self.ok.append((ok[0::2] & ok[1::2]).numpy())
            self.image_change.append((images[1::2] - images[0::2]).norm(dim=1).numpy())
            centre = batch.pop("centre", None)
            if centre is not None:
                self.centres.append(torch.stack((centre[0::2], centre[1::2]), 1).numpy())
            yield batch

    def result(self):
        centres = np.concatenate(self.centres) if self.centres else None
        return np.concatenate(self.ok), np.concatenate(self.image_change), centres


def extract_condition(
    source, dataset, base, target, eps, null_offset, batch_size, workers, device, level, budget, indices, generator
):
    """Paired maps at ``-eps`` and ``+eps``, with each subject's render status, image change and centres.

    The maps are ``(n, 2, C, D, H, W)``; the lesion centres are ``(n, 2, 3)`` at ``-eps`` and
    ``+eps`` in generator coordinates, or None unless ``generator`` renders them.
    """
    from torch.utils.data import DataLoader

    from training import spatial_bottleneck_experiment as experiment

    renders = ConditionRenders(dataset, base, target, eps, null_offset, generator)
    tap = PairTap(DataLoader(renders, batch_size=batch_size, shuffle=False, num_workers=workers))
    bank, _ = experiment.extract_content_maps(source, tap, device, level, budget, indices)
    ok, change, centres = tap.result()
    return bank[0::2], bank[1::2], ok, change, centres


# --------------------------------------------------------------------------- #
# Representations
# --------------------------------------------------------------------------- #


def training_units(features):
    """Per-view multipliers that put feature changes in ``standardized_kernel``'s units.

    ``features`` is ``(n, 2, k)`` on training subjects; a column that kernel drops gets weight 0.
    """
    features = np.asarray(features, dtype=np.float64)
    mean, std = features.mean(0), features.std(0)
    keep = std > 1e-10 * (1.0 + np.abs(mean))
    return np.where(keep, 1.0 / np.where(keep, std, 1.0), 0.0)


def pca_axes(train, template, scale, coefficients):
    """Map-space axes ``A``: a standardised map ``s`` scores ``(s - mean) @ A`` on the fitted components.

    ``coefficients`` is ``(2n, d)``, one row per PCA fit row (view 0's training maps, then view 1's).
    With ``coefficients = scores / w`` these are the axes ``pca_scores`` projects onto, so every
    condition is scored with one fixed basis instead of a refit (and possibly sign-flipped) one.
    """
    import torch

    coefficients = torch.as_tensor(coefficients, dtype=torch.float64)
    axes = []
    for channel in range(template.shape[0]):
        rows = torch.cat([train[:, 0, channel], train[:, 1, channel]]).reshape(len(coefficients), -1).double()
        rows = (rows - template[channel].reshape(-1).double()) / scale[channel].double().reshape(())
        axes.append((rows - rows.mean(0)).T @ coefficients)
    return torch.cat(axes).numpy()


class Readouts:
    """Every representation ``spatial_bottleneck_eval`` probes, as a response to a paired map change.

    ``codes`` maps names such as ``z:RUN`` to readouts with an ``encode`` method; ``train`` is the
    bottleneck's training map bank ``(n, 2, C, D, H, W)``, used only for units and PCA.
    """

    def __init__(self, train, template, scale, codes, latent_dims, patch_grid, batch_size, device):
        import torch.nn.functional as F

        self.template, self.scale = template.cpu(), scale.cpu()
        self.codes, self.batch_size, self.device = dict(codes), batch_size, device
        grid = tuple(template.shape[1:])
        fits = patch_grid is not None and all(g <= m for g, m in zip(patch_grid, grid))
        self.patch_grid = tuple(patch_grid) if fits else None
        self.patch_name = "patch" + "x".join(map(str, patch_grid)) if fits else None
        n = len(train)
        units = {}
        if self.patch_grid is not None:
            pooled = F.adaptive_avg_pool3d(train.flatten(0, 1), self.patch_grid).reshape(n, 2, -1)
            units[self.patch_name] = training_units(pooled.numpy())
        units["gap"] = training_units(train.mean((3, 4, 5)).numpy())
        gram = observational.map_gram([train[:, 0], train[:, 1]], self.template, self.scale, device)
        self.axes = {}
        for d in sorted(set(latent_dims)):
            scores = observational.pca_scores(gram, np.arange(2 * n), d)
            self.axes[d] = pca_axes(train, self.template, self.scale, scores / np.square(scores).sum(0))
            units[f"pca{d}"] = training_units(scores.reshape(2, n, d).transpose(1, 0, 2))
        for name, head in self.codes.items():
            units[name] = training_units(observational.encode_views(head, train, batch_size, device))
        self.units = units
        self.names = ["map", *([self.patch_name] if self.patch_name else []), "gap"]
        self.names += [f"pca{d}" for d in self.axes] + list(self.codes)

    def respond(self, minus, plus, chunk=16):
        """``{name: (n, 2, k)}`` float32 responses ``f(plus) - f(minus)`` for paired map banks."""
        import torch
        import torch.nn.functional as F

        n, out = len(minus), {}
        scale = self.scale.double()
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            delta = plus[start:stop].double() - minus[start:stop].double()
            flat = (delta / scale).reshape(stop - start, 2, -1)
            parts = {"map": flat}
            if self.patch_grid is not None:
                pooled = F.adaptive_avg_pool3d(delta.flatten(0, 1), self.patch_grid)
                parts[self.patch_name] = pooled.reshape(stop - start, 2, -1)
            parts["gap"] = delta.mean((3, 4, 5))
            for d, axes in self.axes.items():
                parts[f"pca{d}"] = flat @ torch.from_numpy(axes)
            for name, part in parts.items():
                block = (part.numpy() * self.units.get(name, 1.0)).astype(np.float32)
                out.setdefault(name, np.empty((n, 2, block.shape[-1]), dtype=np.float32))[start:stop] = block
        for name, head in self.codes.items():
            change = observational.encode_views(head, plus, self.batch_size, self.device)
            change -= observational.encode_views(head, minus, self.batch_size, self.device)
            out[name] = (change * self.units[name]).astype(np.float32)
        return out


# --------------------------------------------------------------------------- #
# Scoring (numpy only)
# --------------------------------------------------------------------------- #


def _subset_mean(values, subset, draws=None):
    """Mean over ``subset`` rows; with ``draws``, one mean per bootstrap row (NaN if none drawn)."""
    weights = subset.astype(np.float64)
    if draws is None:
        return float(_ratio((values * weights).sum(), weights.sum()))
    return _ratio((values * weights)[draws].sum(1), weights[draws].sum(1))


def score_view(responses, null, factors, measured, draws, displacement=None):
    """Probe-free intervention metrics of one representation in one view.

    ``responses[name]`` is ``(n, k)``, ``f(+eps) - f(-eps)`` per subject; ``null`` is the render-null
    response of the same subjects. ``measured[name]`` marks the subjects whose images the
    intervention changed: per-sample SNRs average over those, so a centre that stayed on its voxel
    is not read as blindness (a factor that changed no image keeps every subject and reads 0).
    ``displacement[name]``, in voxels, adds ``snr_render_per_voxel``. ``draws`` is ``(B, n)``
    bootstrap rows, shared by every representation.
    """
    displacement = displacement or {}
    kinds = {f["name"]: f["kind"] for f in factors}
    moved = {name: bool(mask.any()) for name, mask in measured.items()}
    subsets = {name: mask if mask.any() else np.ones_like(mask) for name, mask in measured.items()}
    wrapped = {name: {"content": value} for name, value in responses.items()}
    norms = {name: interventional._fro(value) for name, value in responses.items()}
    null_norm = interventional._fro(null)
    nuisance = [name for name in responses if kinds[name] == "nuisance" and moved[name]]
    nuisance_mean = {name: _subset_mean(norms[name], subsets[name]) for name in nuisance}
    reference = max(nuisance, key=nuisance_mean.get, default=None)
    pop, pop_floor = interventional.population_snr(wrapped, list(responses), null)
    nuisance_pop = max((pop[name] for name in nuisance), default=np.nan)
    null_boot = null_norm[draws].mean(1)
    nuisance_boot = (
        np.max([_subset_mean(norms[name], subsets[name], draws) for name in nuisance], axis=0) if nuisance else None
    )
    record = dict(
        null_response=float(null_norm.mean()),
        pop_floor=pop_floor,
        nuisance_reference=reference,
        factors={},
        boot={},
    )
    for name, value in responses.items():
        mean, boot = _subset_mean(norms[name], subsets[name]), _subset_mean(norms[name], subsets[name], draws)
        per_voxel, per_voxel_boot = np.nan, None
        if name in displacement:
            distance = np.nan_to_num(displacement[name], nan=0.0)
            moving = subsets[name] & (distance > 0)
            if moving.any():
                travel = _subset_mean(distance, moving)
                per_voxel = float(_ratio(_subset_mean(norms[name], moving), travel * null_norm.mean()))
                per_voxel_boot = _ratio(
                    _subset_mean(norms[name], moving, draws), _subset_mean(distance, moving, draws) * null_boot
                )
        record["factors"][name] = dict(
            response=mean,
            snr_render=float(_ratio(mean, null_norm.mean())),
            snr_nuisance=float(_ratio(mean, nuisance_mean[reference])) if reference else np.nan,
            snr_render_per_voxel=per_voxel,
            pop_snr=pop[name],
            pop_snr_nuisance=float(_ratio(pop[name], nuisance_pop)),
            consistency=interventional.loo_consistency(value),
        )
        record["boot"][name] = dict(
            snr_render=_ratio(boot, null_boot),
            snr_nuisance=_ratio(boot, nuisance_boot) if nuisance_boot is not None else None,
            snr_render_per_voxel=per_voxel_boot,
        )
    present = [f["name"] for f in factors if f["kind"] == "content" and moved[f["name"]]]
    if len(present) >= 2:
        accuracy, chance, recall, confusion = interventional.identification_accuracy(wrapped, present)
        matrix = interventional.response_matrix(wrapped, present)
        rank, singular, threshold = interventional.signal_rank(matrix, null)
        gap_at, gap_ratio = interventional.spectral_gap(singular)
        record.update(
            identification=dict(factors=present, accuracy=accuracy, chance=chance, recall=recall, confusion=confusion),
            alias=dict(
                factors=present,
                matrix=interventional.alias_matrix(matrix),
                floor=interventional.alias_floor(null, len(present)),
            ),
            rank=dict(
                signal=rank,
                target=len(present),
                singular_values=singular,
                threshold=threshold,
                spectral_gap_at=gap_at,
                spectral_gap_ratio=gap_ratio,
            ),
        )
    return record


def combine_views(views, factors):
    """View means, with bootstrap intervals from the per-draw mean; confusions summed; aliasing averaged."""
    combined = dict(
        null_response=float(_mean([v["null_response"] for v in views])),
        nuisance_reference=[v["nuisance_reference"] for v in views],
        per_factor={},
    )
    for name in (f["name"] for f in factors):
        records = [v["factors"][name] for v in views]
        row = {key: float(_mean([r[key] for r in records])) for key in VIEW_MEANS}
        for key in BOOTSTRAPPED:
            boots = [v["boot"][name][key] for v in views]
            bounds = observational.interval(np.mean(boots, axis=0)) if all(b is not None for b in boots) else None
            row[f"{key}_ci"] = [math.nan, math.nan] if bounds is None else [float(b) for b in bounds]
            row[f"{key}_by_view"] = [r[key] for r in records]
        combined["per_factor"][name] = row
    if not all("identification" in v for v in views):
        return combined
    present = views[0]["identification"]["factors"]
    chance = views[0]["identification"]["chance"]
    recall = _mean([v["identification"]["recall"] for v in views], axis=0)
    confusion = np.sum([v["identification"]["confusion"] for v in views], axis=0)
    alias = np.mean([v["alias"]["matrix"] for v in views], axis=0)
    others = alias.copy()
    np.fill_diagonal(others, -np.inf)
    for k, name in enumerate(present):
        row = combined["per_factor"][name]
        mistakes = confusion[k].astype(np.float64)
        mistakes[k] = 0.0
        total = confusion[k].sum()
        row.update(
            recall=float(recall[k]),
            confused_with=present[int(mistakes.argmax())] if mistakes.max() > 0 else None,
            confused_share=float(mistakes.max() / total) if total else math.nan,
            max_alias=float(others[k].max()),
            alias_with=present[int(others[k].argmax())],
            direction_estimable=bool(
                not (row["pop_snr"] < 1.5 and np.isfinite(recall[k]) and recall[k] > 2.0 * chance)
            ),
        )
    combined["identification"] = dict(
        factors=present,
        accuracy=float(_mean([v["identification"]["accuracy"] for v in views])),
        accuracy_by_view=[v["identification"]["accuracy"] for v in views],
        chance=chance,
        confusion=confusion,
    )
    combined["alias"] = dict(
        factors=present,
        matrix=alias,
        matrix_by_view=[v["alias"]["matrix"] for v in views],
        floor=float(_mean([v["alias"]["floor"] for v in views])),
        floor_by_view=[v["alias"]["floor"] for v in views],
    )
    combined["rank"] = [v["rank"] for v in views]
    return combined


def score_representation(by_target, factors, measured, draws, displacement=None):
    """``by_target[name]`` is ``(n, 2, k)`` for every factor, ``by_target[NULL]`` the render null."""
    views = [
        score_view(
            {f["name"]: by_target[f["name"]][:, view] for f in factors},
            by_target[NULL][:, view],
            factors,
            measured,
            draws,
            displacement,
        )
        for view in (0, 1)
    ]
    return combine_views(views, factors)


# --------------------------------------------------------------------------- #
# Source parity
# --------------------------------------------------------------------------- #


def saved_output_parity(head, init, banks, outputs, report, trained, batch_size, device):
    """``spatial_bottleneck_eval``'s check that the regenerated subjects are the ones the bottleneck saw."""
    from training import spatial_bottleneck_experiment as experiment

    test, val = banks["test"][: trained["test"]], banks["val"][: trained["val"]]
    return dict(
        test_codes=observational.relative_error(
            observational.encode_views(head, test, batch_size, device), outputs["codes"]
        ),
        test_gap=observational.relative_error(test.mean((3, 4, 5)), outputs["gap"]),
        val_reconstruction=observational.relative_error(
            experiment.evaluate(head, val, batch_size, device)[0]["reconstruction"],
            report["validation"]["reconstruction"],
        ),
        initial_val_reconstruction=observational.relative_error(
            experiment.evaluate(init, val, batch_size, device)[0]["reconstruction"],
            report["training"]["initial_validation"]["reconstruction"],
        ),
    )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _cell(value, width=0, digits=2):
    if value is None or not math.isfinite(value):
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.{digits}f}"


def print_report(payload):
    reps, results, factors = payload["representations"], payload["results"], payload["factors"]
    meta = {f["name"]: f for f in factors}
    kinds = ("content", "style_shared", "style", "nuisance")
    names = {kind: [f["name"] for f in factors if f["kind"] == kind] for kind in kinds}
    moved = [n for n in names["content"] if meta[n]["moved"]]
    width = max(12, *(len(rep) + 2 for rep in reps))
    subjects, nulls = payload["subjects"], payload["nulls"]
    print(f"\nSource {payload['source']['checkpoint']}  level {payload['source']['level']}  eps ±{payload['eps']}")
    dropped = f"; {len(subjects['dropped'])} dropped (no room for the lesion)" if subjects["dropped"] else ""
    print(f"Subjects: {subjects['n']} from the test split (seed {subjects['sample_seed']}){dropped}")
    print(f"Render null: {nulls['render']}")
    nuisance = nulls["nuisance"]
    print(
        "Nuisance null: "
        + (f"largest of {', '.join(nuisance['targets'])}" if nuisance["available"] else nuisance["reason"])
    )
    print("  The render null flatters a code trained for cross-view agreement; the nuisance null the frozen readouts.")
    ignored = [f["name"] for f in factors if not f["moved"]]
    if ignored:
        print(f"Changed no image (left out of identification and aliasing): {', '.join(ignored)}")
    for name, reason in payload["void_representations"].items():
        print(f"Not scored: {name} ({reason})")
    shifted_centre = [f for f in factors if f["centre_shift"] is not None]
    if shifted_centre:
        print("Rendered lesion centre per intervention: mean |delta| in x / y / z (generator coordinates),")
        print("mean displacement in voxels, share of subjects whose centre moved, and midline crossings:")
        for f in shifted_centre:
            print(
                f"  {f['name']:<20}"
                + " ".join(_cell(v, 6, 3) for v in f["centre_shift"])
                + f"{_cell(f['displacement_vox'], 8)} vox{f['centre_moved_frac']:>6.0%} moved"
                + f"{f['midline_crossings']:>4} crossed"
            )
    elif payload["lesion_centre"]["note"]:
        print(f"Rendered lesion centres not tracked: {payload['lesion_centre']['note']}")
    print(
        "snr averages over subjects whose images changed: 0 = no response (both renders share the render seed); "
        "1 = as large as redrawing the noise."
    )

    for factor in (n for n in moved if n.startswith("lesion")):
        print(
            f"\n== {factor} moved by ±{payload['eps']}, everything else held "
            f"(image changed for {meta[factor]['active_frac']:.0%} of subjects) =="
        )
        if meta[factor]["displacement_vox"] is not None:
            print(
                f"   rendered centre moved {meta[factor]['displacement_vox']:.1f} voxels on average; "
                f"{meta[factor]['midline_crossings']} subject(s) crossed the midline. "
                "snr/vox divides snr_render by that displacement."
            )
        print(
            f"{'representation':<{width}}{'snr_render [95% CI]':>23}{'snr/vox':>9}{'snr_nuis':>10}{'pop_snr':>9}"
            f"{'consist':>9}{'recall':>8}   {'mistaken for':<26}max |cos| (with)"
        )
        for rep in reps:
            row = results[rep]["per_factor"][factor]
            low, high = row["snr_render_ci"]
            mistaken = f"{row['confused_with']} ({row['confused_share']:.0%})" if row.get("confused_with") else "-"
            alias = f"{_cell(row.get('max_alias'))} ({row['alias_with']})" if row.get("alias_with") else "n/a"
            print(
                f"{rep:<{width}}{_cell(row['snr_render'], 8)} [{_cell(low, 5)},{_cell(high, 6)}]"
                f"{_cell(row['snr_render_per_voxel'], 9, 3)}{_cell(row['snr_nuisance'], 10)}"
                f"{_cell(row['pop_snr'], 9, 1)}{_cell(row['consistency'], 9)}"
                f"{_cell(row.get('recall'), 8)}   {mistaken:<26}{alias}"
            )
        others = [n for n in moved if n != factor]
        print(f"   |cos| of the mean {factor} response with each other factor's, and the unrelated-direction floor:")
        print(f"{'':<{width}}" + "".join(f"{n[:13]:>14}" for n in others) + f"{'floor':>8}")
        for rep in reps:
            alias = results[rep].get("alias")
            if alias:
                order = alias["factors"]
                row = alias["matrix"][order.index(factor)]
                cells = "".join(_cell(row[order.index(n)], 14) for n in others)
                print(f"{rep:<{width}}{cells}{_cell(alias['floor'], 8)}")

    for key, title in (("snr_render", "the render null"), ("snr_nuisance", "the nuisance null")):
        if key == "snr_nuisance" and not nuisance["available"]:
            continue
        print(f"\n== {key}: response over {title}, every factor (mean over views) ==")
        print(f"{'factor':<20}" + "".join(f"{rep:>{width}}" for rep in reps))
        sections = [("content", names["content"])]
        sections += [(observational.BLOCK_TITLES[kind], names[kind]) for kind in ("style_shared", "style")]
        if key == "snr_render":
            sections.append(("nuisance", names["nuisance"]))
        for label, rows in sections:
            if rows:
                print(f"-- {label}")
            for factor in rows:
                print(f"{factor:<20}" + "".join(_cell(results[rep]["per_factor"][factor][key], width) for rep in reps))

    identified = [rep for rep in reps if results[rep].get("identification")]
    if identified:
        chance = results[identified[0]]["identification"]["chance"]
        print(f"\n== Naming the moved content factor from one response: recall (chance {chance:.2f}) ==")
        print(f"{'representation':<{width}}{'id_acc':>8}{'rank':>10}" + "".join(f"{n[:11]:>12}" for n in moved))
        for rep in identified:
            ident = results[rep]["identification"]
            rank = ",".join(str(r["signal"]) for r in results[rep]["rank"]) + f"/{len(moved)}"
            cells = "".join(_cell(results[rep]["per_factor"][n].get("recall"), 12) for n in moved)
            print(f"{rep:<{width}}{_cell(ident['accuracy'], 8)}{rank:>10}{cells}")


def write_outputs(out_dir, payload, lesion_moves=()):
    """``payload`` must already be JSON-safe; missing values become empty CSV cells.

    ``lesion_moves`` holds one row per tracked intervention and subject (both centre endpoints).
    """
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "interventions.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    meta = {f["name"]: f for f in payload["factors"]}
    with (out_dir / "interventions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "representation",
                "factor",
                "kind",
                "active_frac",
                "image_change",
                "centre_shift_x",
                "centre_shift_y",
                "centre_shift_z",
                "displacement_vox",
                "midline_crossings",
                "response",
                "snr_render",
                "snr_render_low",
                "snr_render_high",
                "snr_nuisance",
                "snr_nuisance_low",
                "snr_nuisance_high",
                "snr_render_per_voxel",
                "snr_render_per_voxel_low",
                "snr_render_per_voxel_high",
                "pop_snr",
                "pop_snr_nuisance",
                "consistency",
                "recall",
                "confused_with",
                "confused_share",
                "max_alias",
                "alias_with",
                "direction_estimable",
                "snr_render_view1",
                "snr_render_view2",
                "id_acc",
                "id_chance",
            ]
        )
        for rep in payload["representations"]:
            result = payload["results"][rep]
            ident = result.get("identification") or {}
            for name, row in result["per_factor"].items():
                factor = meta[name]
                shift = factor["centre_shift"] or [None] * 3
                writer.writerow(
                    [rep, name, factor["kind"], factor["active_frac"], factor["image_change"], *shift]
                    + [factor["displacement_vox"], factor["midline_crossings"]]
                    + [row["response"], row["snr_render"], *row["snr_render_ci"]]
                    + [row["snr_nuisance"], *row["snr_nuisance_ci"]]
                    + [row["snr_render_per_voxel"], *row["snr_render_per_voxel_ci"]]
                    + [row["pop_snr"], row["pop_snr_nuisance"]]
                    + [row["consistency"], row.get("recall"), row.get("confused_with"), row.get("confused_share")]
                    + [row.get("max_alias"), row.get("alias_with"), row.get("direction_estimable")]
                    + [*row["snr_render_by_view"], ident.get("accuracy"), ident.get("chance")]
                )
    with (out_dir / "alias.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["representation", "factor", "other", "abs_cos", "abs_cos_view1", "abs_cos_view2", "floor"])
        for rep in payload["representations"]:
            alias = payload["results"][rep].get("alias")
            if not alias:
                continue
            for i, factor in enumerate(alias["factors"]):
                for j, other in enumerate(alias["factors"]):
                    if i != j:
                        by_view = [matrix[i][j] for matrix in alias["matrix_by_view"]]
                        writer.writerow([rep, factor, other, alias["matrix"][i][j], *by_view, alias["floor"]])
    with (out_dir / "confusion.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["representation", "moved", "identified_as", "count_both_views"])
        for rep in payload["representations"]:
            ident = payload["results"][rep].get("identification")
            if not ident:
                continue
            for i, factor in enumerate(ident["factors"]):
                for j, other in enumerate(ident["factors"]):
                    writer.writerow([rep, factor, other, ident["confusion"][i][j]])
    if lesion_moves:
        with (out_dir / "lesion_moves.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(lesion_moves[0]))
            writer.writeheader()
            for row in lesion_moves:
                writer.writerow({k: "" if isinstance(v, float) and not math.isfinite(v) else v for k, v in row.items()})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bottleneck-dirs", type=Path, nargs="+", required=True, help="Bottleneck output dirs")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory for the results")
    parser.add_argument("--run-dir", type=Path, default=None, help="Source VQ run, if it moved since training")
    parser.add_argument(
        "--num-subjects",
        type=int,
        default=None,
        help="Test subjects (default: the bottleneck's; more extends the split)",
    )
    parser.add_argument("--eps", type=float, default=0.5, help="Each factor is rendered at z - eps and z + eps")
    parser.add_argument("--factors", nargs="+", default=None, help="Only these targets (default: all of them)")
    parser.add_argument("--patch-grid", type=observational.parse_grid, default=(2, 2, 2), help="D,H,W or 'none'")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples of the subjects")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed")
    parser.add_argument("--null-seed-offset", type=int, default=7919, help="Render-seed shift of the render null")
    parser.add_argument("--batch-size", type=int, default=None, help="Even extraction batch (default: the run's)")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers that render the conditions")
    parser.add_argument("--max-map-gib", type=float, default=None, help="Map bank cap (default: the run's)")
    parser.add_argument("--max-response-gib", type=float, default=4.0, help="Cap on the held full-map responses")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=0, help="0 keeps torch's default")
    args = parser.parse_args(argv)
    if not math.isfinite(args.eps) or args.eps <= 0:
        parser.error("--eps must be positive and finite")
    if args.num_subjects is not None and args.num_subjects < 3:
        parser.error("--num-subjects needs at least three subjects")
    if args.bootstrap < 100:
        parser.error("--bootstrap needs at least 100 resamples")
    if args.batch_size is not None and (args.batch_size < 2 or args.batch_size % 2):
        parser.error("--batch-size must be even: each batch holds whole (-eps, +eps) pairs")
    if args.workers < 0 or args.max_response_gib <= 0:
        parser.error("--workers must be nonnegative and --max-response-gib positive")
    if args.out_dir.exists():
        parser.error("--out-dir must be a new directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch

    from eval.metrics.dci import STYLE_FACTOR_NAMES, content_factor_names, dataset_lesion_target
    from training import spatial_bottleneck_experiment as experiment
    from utils.encoder_runtime import select_encoder_device

    started = time.time()
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    device = select_encoder_device(args.device)
    runs = observational.load_runs(args.bottleneck_dirs)
    shared = observational.shared_source(runs)
    first = runs[0]["report"]
    run_dir, checkpoint = observational.resolve_source(first, args.run_dir)
    source, saved_args, _ = experiment.load_source(
        Namespace(run_dir=run_dir, checkpoint=str(checkpoint), seed=first["arguments"]["seed"]), device
    )
    if getattr(saved_args, "synthetic_mode", "pseudo_mri") != "pseudo_mri":
        raise ValueError("Interventions need --synthetic-mode pseudo_mri, whose latents can be re-rendered")

    trained = shared["counts"]
    n = args.num_subjects or trained["test"]
    if n < 3:
        raise ValueError(f"The bottleneck has {n} test subjects; pass --num-subjects 3 or more")
    counts = dict(trained, test=max(n, trained["test"]))
    batch_size = args.batch_size or 2 * math.ceil(first["arguments"]["batch_size"] / 2)
    budget = int((args.max_map_gib or first["arguments"]["max_map_gib"]) * 2**30)
    banks, labels, datasets, indices, remaining = {}, {}, {}, shared["content_indices_per_view"], budget
    for split in observational.SPLITS:
        dataset = experiment.build_split_dataset(saved_args, split, counts[split])
        if int(dataset._inner.seed) != shared["sample_seeds"][split]:
            raise ValueError(f"Regenerated {split} split has seed {dataset._inner.seed}, not the recorded one")
        banks[split], indices, labels[split] = observational.extract_split(
            source, dataset, batch_size, device, shared["level"], remaining, indices
        )
        remaining -= banks[split].numel() * banks[split].element_size()
        datasets[split] = dataset
        print(f"Extracted {split}: {tuple(banks[split].shape)}", flush=True)

    heads = []
    for run in runs:
        head = experiment.load_bottleneck(run["dir"] / "bottleneck.pt", device)
        if tuple(head.map_shape) != tuple(banks["train"].shape[2:]):
            raise ValueError(f"{run['dir']}: map shape {head.map_shape} differs from the regenerated maps")
        if heads and not (
            torch.equal(head.template.cpu(), heads[0].template.cpu())
            and torch.equal(head.scale.cpu(), heads[0].scale.cpu())
        ):
            raise ValueError(f"{run['dir']}: normalisation differs from {runs[0]['dir']}, so its training maps differ")
        heads.append(head)
    codes, bottlenecks = {}, []
    for run, head in zip(runs, heads):
        report = run["report"]
        saved = torch.load(run["dir"] / "bottleneck.pt", map_location="cpu", weights_only=True)
        outputs = torch.load(run["dir"] / "test_outputs.pt", map_location="cpu", weights_only=True)
        init = observational.initial_readout(saved, report["arguments"]["seed"]).to(device)
        parity = saved_output_parity(head, init, banks, outputs, report, trained, batch_size, device)
        failed = {
            k: v for k, v in parity.items() if k != "initial_val_reconstruction" and v > observational.PARITY_TOLERANCE
        }
        if failed:
            raise ValueError(
                f"{run['dir']}: regenerated subjects do not reproduce the saved outputs (relative error {failed}); "
                "the generator, source checkpoint or code changed since training"
                + (
                    ". Under fixed_reference normalisation a split under 64 subjects is normalised on its own "
                    "subjects, so it cannot be extended: drop --num-subjects"
                    if counts["test"] != trained["test"]
                    else ""
                )
            )
        z_init_reproduced = parity["initial_val_reconstruction"] <= observational.PARITY_TOLERANCE
        if not z_init_reproduced:
            print(
                f"WARNING {run['name']}: the rebuilt initial readout does not match the recorded step-0 validation "
                "loss (different torch version?). z_init is a random readout at the same seed, not the exact one.",
                flush=True,
            )
        codes[f"z_init:{run['name']}"] = init
        codes[f"z:{run['name']}"] = head
        bottlenecks.append(
            dict(
                name=run["name"],
                dir=run["dir"],
                latent_dim=head.latent_dim,
                seed=report["arguments"]["seed"],
                best_step=report["training"]["best_step"],
                parity=parity,
                z_init_reproduced=z_init_reproduced,
            )
        )
    template, scale = heads[0].template.cpu(), heads[0].scale.cpu()
    readouts = Readouts(
        banks["train"], template, scale, codes, [h.latent_dim for h in heads], args.patch_grid, batch_size, device
    )
    del banks

    test = datasets["test"]
    base = []
    for i in range(n):
        item = test[i]
        latents, seed = item["gt_latents"], int(test._inner.sample_seed_for(i))
        x1, x2, _ = render_pair(test, latents, seed)
        if not (torch.equal(x1, item["image"][0]) and torch.equal(x2, item["image"][1])):
            raise ValueError(
                f"Re-rendering test subject {i} from its latents does not reproduce the dataset's images, "
                "so an intervention would not start from that subject"
            )
        base.append((latents, seed))

    n_content = base[0][0]["z_content"].numel()
    content_names = content_factor_names(n_content, dataset_lesion_target(test))
    content_names += [f"z_content_{j}" for j in range(len(content_names), n_content)]
    style_names = STYLE_FACTOR_NAMES[: base[0][0]["z_style_v1"].numel()]
    targets = build_targets(content_names, style_names, base[0][0])
    for target in targets:
        index = target.get("index")
        if target["kind"] == "style" and all(
            torch.equal(z["z_style_v1"][index], z["z_style_v2"][index]) for z, _ in base
        ):
            target["kind"] = "style_shared"
    if args.factors:
        unknown = sorted(set(args.factors) - {t["name"] for t in targets})
        if unknown:
            raise ValueError(f"Unknown --factors {unknown}; choose from {[t['name'] for t in targets]}")
        targets = [t for t in targets if t["name"] in args.factors]
    held = (len(targets) + 1) * n * 2 * math.prod(template.shape) * 4
    if held > args.max_response_gib * 2**30:
        raise ValueError(
            f"Full-map responses need {held / 2**30:.2f} GiB; lower --num-subjects, restrict --factors "
            "or raise --max-response-gib"
        )

    generator, centre_note = observational.centre_generator(test)
    print(f"Rendering {len(targets)} interventions and the render null for {n} subjects", flush=True)
    responses, status = {name: {} for name in readouts.names}, {}
    for target in [None, *targets]:
        label = NULL if target is None else target["name"]
        tic = time.time()
        minus, plus, ok, change, centres = extract_condition(
            source,
            test,
            base,
            target,
            args.eps,
            args.null_seed_offset,
            batch_size,
            args.workers,
            device,
            shared["level"],
            budget,
            indices,
            generator,
        )
        for name, value in readouts.respond(minus, plus).items():
            responses[name][label] = value
        status[label] = (ok, change, centres)
        print(f"  {label:<20} rendered and encoded in {time.time() - tic:.1f}s", flush=True)
    del source, minus, plus

    keep = np.logical_and.reduce([ok for ok, _, _ in status.values()])
    if keep.sum() < 3:
        raise ValueError("Fewer than three subjects rendered under every intervention; lower --eps")
    if not keep.all():
        for by_target in responses.values():
            for label in by_target:
                by_target[label] = by_target[label][keep]
    # The encoder is deterministic, so a subject whose images the intervention left bit-identical
    # was not measured: its response is set to exactly 0, which the metrics skip. Anything nonzero
    # there is device noise, recorded as unchanged_response (exactly 0 on CPU).
    ids = np.array([f"test:{i}" for i in range(n)])
    voxel = (test.res - 1) / 2
    factors, measured, displacement, lesion_moves = [], {}, {}, []
    for target in targets:
        name = target["name"]
        _, change, centres = status[name]
        change = change[keep]
        measured[name] = change > 0
        unchanged, noise = change == 0, 0.0
        if unchanged.any():
            for by_target in responses.values():
                value = by_target[name]
                noise = max(noise, float(np.abs(value[unchanged]).max()))
                value[unchanged] = 0.0
        record = dict(
            name=name,
            kind=target["kind"],
            moved=bool(measured[name].any()),
            active_frac=float(measured[name].mean()),
            image_change=float(change.mean()),
            unchanged_response=noise,
            centre_shift=None,
            centre_moved_frac=None,
            displacement_vox=None,
            midline_crossings=None,
        )
        if centres is not None:
            centres = centres[keep]
            shift = centres[:, 1] - centres[:, 0]
            distance = np.linalg.norm(shift, axis=1) * voxel
            crossed = np.sign(centres[:, 0, 0]) * np.sign(centres[:, 1, 0]) < 0
            record.update(
                centre_shift=_mean(np.abs(shift), axis=0),
                centre_moved_frac=float((distance > 0).mean()),
                displacement_vox=float(_mean(distance)),
                midline_crossings=int(crossed.sum()),
            )
            if name.startswith("lesion_"):
                displacement[name] = distance
            for subject, ends, travel, across, changed in zip(ids[keep], centres, distance, crossed, measured[name]):
                lesion_moves.append(
                    dict(
                        factor=name,
                        subject_id=str(subject),
                        **{f"centre_minus_{axis}": float(ends[0, j]) for j, axis in enumerate("xyz")},
                        **{f"centre_plus_{axis}": float(ends[1, j]) for j, axis in enumerate("xyz")},
                        displacement_vox=float(travel),
                        crossed_midline=bool(across),
                        image_changed=bool(changed),
                    )
                )
        factors.append(record)
    moved = {f["name"]: f["moved"] for f in factors}
    void = {
        name: "every feature is constant on training subjects in at least one view"
        for name, units in readouts.units.items()
        if not np.any(units, axis=-1).all()
    }
    n_keep = int(keep.sum())
    draws = np.random.default_rng(args.seed).integers(0, n_keep, size=(args.bootstrap, n_keep))
    results = {}
    for name in readouts.names:
        by_target = responses.pop(name)
        if name not in void:
            results[name] = score_representation(by_target, factors, measured, draws, displacement)

    nuisance = [f["name"] for f in factors if f["kind"] == "nuisance"]
    if any(moved[name] for name in nuisance):
        nuisance_null = dict(available=True, targets=[name for name in nuisance if moved[name]])
    else:
        reason = (
            "unavailable: no deformation/fissure intervention was run"
            if not nuisance
            else (
                "unavailable: no deformation/fissure intervention changed an image"
                + (
                    " (the renderer zeroes these fields under --synthetic-clean-content)"
                    if getattr(saved_args, "synthetic_clean_content", False)
                    else ""
                )
            )
        )
        nuisance_null = dict(available=False, reason=reason)
    payload = dict(
        source=dict(run_dir=run_dir, checkpoint=checkpoint, **{k: v for k, v in shared.items() if k != "counts"}),
        splits={
            s: dict(n=counts[s], n_bottleneck=trained[s], sample_seed=shared["sample_seeds"][s])
            for s in observational.SPLITS
        },
        subjects=dict(
            split="test",
            requested=n,
            n=n_keep,
            sample_seed=shared["sample_seeds"]["test"],
            ids=ids[keep].tolist(),
            dropped=ids[~keep].tolist(),
            render_seeds=[seed for (_, seed), kept in zip(base, keep) if kept],
            z_content=labels["test"][0][:n][keep],
            rerender="each subject's images reproduced bit for bit from its latents and render seed",
        ),
        eps=args.eps,
        factors=factors,
        representations=[name for name in readouts.names if name not in void],
        void_representations=void,
        bottlenecks=bottlenecks,
        nulls=dict(
            render=f"same latents, render seed + {args.null_seed_offset} (bias field and noise redrawn)",
            render_image_change=float(status[NULL][1][keep].mean()),
            nuisance=nuisance_null,
        ),
        lesion_centre=dict(rendered=generator is not None, note=centre_note),
        results=results,
        protocol=dict(
            interventions="z_j - eps and z_j + eps; every other latent and the render seed held",
            style_targets="the coordinate moves in both views; each view's code is read from its own scan; "
            "a coordinate identical in both views for every subject is style_shared",
            centre_shift="mean |change| of the rendered lesion centre (sample_targets), generator coordinates",
            displacement="Euclidean centre displacement in voxels, (res - 1) / 2 per generator unit; "
            "midline crossing = the x-centre changes sign",
            measured_subjects="per-sample SNRs average over subjects whose images the intervention changed",
            snr_render_per_voxel="lesion coordinates: snr_render over the mean displacement of the measured subjects",
            nuisance_targets="deformation/fissure field moved by -eps/+eps along a fixed random unit direction",
            responses="f(+eps) - f(-eps) per subject and view, in the probe units of spatial_bottleneck_eval",
            map_units="divided by the bottleneck's per-channel scale",
            other_units="divided by the training-subject standard deviation in each view; constant columns dropped",
            pca="components fitted on both views' training maps through the Gram matrix; one fixed basis",
            identification="5-fold nearest-centroid (cosine) over subjects, content factors that changed an image",
            alias_floor="95th percentile of the largest |cos| between sign-flipped render-null means",
            uncertainty=f"paired bootstrap over subjects, {args.bootstrap} draws",
            views="each view scored separately; tables average the two",
            bootstrap_seed=args.seed,
            null_seed_offset=args.null_seed_offset,
            batch_size=batch_size,
        ),
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        elapsed_seconds=round(time.time() - started, 1),
    )
    payload = observational._json_safe(payload)
    write_outputs(args.out_dir, payload, lesion_moves)
    print_report(payload)
    print(f"\nSaved {args.out_dir / 'interventions.json'} and CSVs", flush=True)
    return payload


if __name__ == "__main__":
    main()
