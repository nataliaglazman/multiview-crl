"""Temporary, replayable banks for scalar readout experiments."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from eval.encoder.encoder_target_protocol import VIEWS
from eval.metrics.dci import CONTENT_FACTOR_NAMES
from eval.synthetic.factor_structure_audit import context, render, restore_dataset
from eval.synthetic.synthetic_dataset import LesionPlacementError
from models.scalar_readout import descriptors, flip_volume, pool_grid

RAW_NAMES = tuple(CONTENT_FACTOR_NAMES)
UNLABELLED_KEYS = ("features", "flip_features", "photo_features", "descriptors", "flip_descriptors", "flip_signs")


def semantic_target(ds, z, metadata):
    target = z.numpy().copy()
    target[2:5] = metadata["centroid"]
    renderer = ds._inner.renderer
    squash = renderer.content_squash
    if squash == "auto":
        squash = "tanh" if ds._inner.clean_content else "clamp"
    a = z[8].tanh() if squash == "tanh" else z[8].clamp(-1, 1) if squash == "clamp" else z[8]
    multiplier = 1.0 if renderer.content_amp_scale is None else renderer.content_amp_scale[8]
    target[8] = float(a) * 0.06 * renderer.content_scale * multiplier
    if not np.isfinite(target).all():
        raise ValueError("Non-finite target (empty lesion is not silently replaced)")
    return target


class Banks:
    """Own memmaps so they can all be closed before the temporary directory is removed."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.arrays = []

    def allocate(self, name, count, example):
        path = self.directory / f"{name}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(count, *example.shape))
        self.arrays.append(array)
        return array

    def close(self):
        for a in self.arrays:
            a.flush()
            a._mmap.close()
        self.arrays.clear()


class Extractor:
    def __init__(self, model, args, device):
        self.model, self.args, self.device = model, args, device

    @torch.inference_mode()
    def __call__(self, images):
        if self.model is None:
            return images.numpy(), None
        h = self.model._encode(images.to(self.device), n_views=1, view_idx=VIEWS.index(self.args.view))
        pooled = pool_grid(h, self.args.grid).cpu().numpy()
        reference = self.model._global_code(h, images.to(self.device))[:, :9].cpu().numpy()
        if not np.isfinite(pooled).all() or not np.isfinite(reference).all():
            raise ValueError("Non-finite frozen encoder features")
        return pooled, reference


def observational_bank(cfg, split, count, args, extract, banks, augment=False):
    ds, _ = restore_dataset(cfg, max(64, count), split)
    output, raw, truth, hashes, seeds = {}, [], [], hashlib.sha256(), []
    for idx in range(count):
        ctx = context(ds, idx)
        image, meta = render(ds, ctx, ctx["lat"]["z_content"])
        x = torch.from_numpy(image[VIEWS.index(args.view)])[None, None]
        hashes.update(x.numpy().tobytes())
        seeds.append(ctx["seed"])
        requests = [x]
        extra = {}
        if augment:
            rng = np.random.default_rng(args.seed + idx * 997 + 101)
            bits = int(rng.integers(1, 8))
            signs = np.array([-1 if bits & (1 << k) else 1 for k in range(3)], np.float32)
            flipped = flip_volume(x, signs)
            noise = torch.from_numpy(rng.normal(0, 0.01, size=x.shape).astype(np.float32))
            photo = (x * float(rng.uniform(0.9, 1.1)) + float(rng.uniform(-0.03, 0.03)) + noise) * (x != 0)
            requests += [flipped, photo]
            desc = descriptors(x, args.descriptor_grid)[0]
            extra = dict(descriptors=desc.numpy(), flip_descriptors=flip_volume(desc, signs).numpy(), flip_signs=signs)
        features, reference = extract(torch.cat(requests))
        items = dict(features=features[0], **extra)
        if augment:
            items.update(flip_features=features[1], photo_features=features[2])
        if reference is not None:
            items["reference"] = reference[0]
        for name, value in items.items():
            if name not in output:
                output[name] = banks.allocate(f"{split}/{name}", count, value)
            output[name][idx] = value
        raw.append(ctx["lat"]["z_content"].numpy())
        truth.append(semantic_target(ds, ctx["lat"]["z_content"], meta))
        if (idx + 1) % 32 == 0 or idx + 1 == count:
            print(f"  {split} observations: {idx + 1}/{count}", flush=True)
    output.update(raw=np.array(raw), truth=np.array(truth))
    metadata = dict(
        split=split,
        count=count,
        subject_ids=list(range(count)),
        accepted_seeds=seeds,
        generator_seed=int(ds._inner.seed),
        image_sha256=hashes.hexdigest(),
    )
    return output, metadata


def intervention_bank(cfg, split, count, offset, epsilons, args, extract, banks):
    ds, _ = restore_dataset(cfg, max(64, offset + count), split)
    capacity = count * len(epsilons) * 9
    output, rows, truth, raw, references = {}, [], [], [], []
    hashes, accepted, seeds = hashlib.sha256(), 0, []
    for idx in range(offset, offset + count):
        ctx = context(ds, idx)
        seeds.append(ctx["seed"])
        for eps in epsilons:
            for k, factor in enumerate(RAW_NAMES):
                row = dict(
                    subject_id=idx,
                    factor=factor,
                    factor_index=k,
                    eps=eps,
                    valid=False,
                    pair_index=-1,
                    input_delta_rms=None,
                    feature_delta_rms=None,
                    zero_image=False,
                    error="",
                )
                try:
                    controls = [ctx["lat"]["z_content"].clone() for _ in range(2)]
                    controls[0][k] -= eps
                    controls[1][k] += eps
                    endpoints = [render(ds, ctx, z) for z in controls]
                    images = torch.from_numpy(np.stack([x[VIEWS.index(args.view)] for x, _ in endpoints]))[:, None]
                    target = np.stack([semantic_target(ds, z, m) for z, (_, m) in zip(controls, endpoints)])
                    features, reference = extract(images)
                    if "features" not in output:
                        output["features"] = banks.allocate(f"{split}_pairs/features", capacity, features)
                    output["features"][accepted] = features
                    if reference is not None:
                        references.append(reference)
                    truth.append(target)
                    raw.append(np.stack([z.numpy() for z in controls]))
                    hashes.update(images.numpy().tobytes())
                    delta = (images[1] - images[0]).double()
                    row.update(
                        valid=True,
                        pair_index=accepted,
                        input_delta_rms=float(delta.square().mean().sqrt()),
                        feature_delta_rms=float(np.sqrt(np.mean((features[1].astype(np.float64) - features[0]) ** 2))),
                        zero_image=bool(delta.abs().max() <= 1e-6),
                    )
                    accepted += 1
                except LesionPlacementError as error:
                    row["error"] = str(error)
                rows.append(row)
        print(f"  {split} interventions: subject {idx-offset+1}/{count}; {accepted} valid pairs", flush=True)
    if not accepted:
        raise ValueError("No valid intervention pairs; subjects were not redrawn")
    output["features"] = output["features"][:accepted]
    output.update(truth=np.array(truth), raw=np.array(raw), rows=rows)
    if references:
        output["reference"] = np.array(references)
    return output, dict(
        split=split,
        subject_ids=list(range(offset, offset + count)),
        accepted_seeds=seeds,
        generator_seed=int(ds._inner.seed),
        image_sha256=hashes.hexdigest(),
        requested_pairs=capacity,
        valid_pairs=accepted,
        failed_pairs=capacity - accepted,
        zero_image_pairs=sum(r["zero_image"] for r in rows),
    )
