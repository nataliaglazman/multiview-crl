"""Label-separated banks for the frozen local-scalar experiment."""

import hashlib

import numpy as np
import torch

from eval.encoder.encoder_lesion_contrast import contrast_metrics, render_reference
from eval.encoder.encoder_target_protocol import VIEWS
from eval.synthetic.factor_structure_audit import context, render, restore_dataset
from eval.synthetic.synthetic_dataset import LesionPlacementError
from models.scalar_readout import flip_volume, pool_grid
from training.scalar_readout_data import RAW_NAMES, semantic_target

UNLABELLED = (
    "features",
    "support",
    "global",
    "flip_features",
    "flip_support",
    "photo_features",
    "photo_support",
    "signs",
)


class FrozenExtractor:
    def __init__(self, model, grid, views, device):
        self.model, self.grid, self.views, self.device = model, grid, views, device

    @torch.inference_mode()
    def __call__(self, images):
        # images: (N, V, 1, D,H,W); routing never confuses endpoint with modality.
        output, support, global_code = [], [], []
        for v, view in enumerate(self.views):
            x = images[:, v].to(self.device)
            h = self.model._encode(x, n_views=1, view_idx=VIEWS.index(view))
            output.append(pool_grid(h, self.grid).cpu().numpy())
            support.append(pool_grid((x != 0).float(), self.grid).cpu().numpy())
            global_code.append(self.model._global_code(h, x)[:, : self.model.content_channels].cpu().numpy())
        result = dict(
            features=np.stack(output, 1),
            support=np.stack(support, 1),
            global_=np.stack(global_code, 1),
        )
        result["global"] = result.pop("global_")
        if not all(np.isfinite(x).all() for x in result.values()):
            raise ValueError("Non-finite frozen features")
        return result


def selected_images(images, views):
    return torch.from_numpy(images[[VIEWS.index(v) for v in views]])[:, None]


def observations(
    cfg,
    split,
    count,
    args,
    extract,
    banks,
    augment=False,
    namespace=None,
    contrast=False,
):
    ds, _ = restore_dataset(cfg, max(64, count), split)
    result, truth, raw, contrasts = {}, [], [], []
    image_hash, seeds = hashlib.sha256(), []
    for idx in range(count):
        ctx = context(ds, idx)
        images, meta = render(ds, ctx, ctx["lat"]["z_content"])
        x = selected_images(images, args.views)
        requests = [x]
        if augment:
            rng = np.random.default_rng(args.seed + 101 + idx * 997)
            bits = int(rng.integers(1, 8))
            signs = np.array([-1 if bits & (1 << k) else 1 for k in range(3)], np.float32)
            photo = (x * float(rng.uniform(0.95, 1.05)) + float(rng.uniform(-0.01, 0.01))) * (x != 0)
            requests += [flip_volume(x, signs), photo]
        encoded = extract(torch.stack(requests))
        items = {k: a[0] for k, a in encoded.items()}
        if augment:
            items.update(
                {
                    f"{prefix}_{key}": encoded[key][n]
                    for n, prefix in ((1, "flip"), (2, "photo"))
                    for key in ("features", "support")
                }
            )
            items["signs"] = signs
        for key, value in items.items():
            if key not in result:
                result[key] = banks.allocate(f"{namespace or split}/{key}", count, value)
            result[key][idx] = value
        truth.append(semantic_target(ds, ctx["lat"]["z_content"], meta))
        raw.append(ctx["lat"]["z_content"].numpy())
        image_hash.update(x.numpy().tobytes())
        seeds.append(ctx["seed"])
        if contrast:
            reference = render_reference(ds, idx)
            _, lesion = ds._inner.renderer.render_structure(
                ctx["lat"]["z_content"],
                ctx["lat"]["z_deformation"],
                ctx["lat"]["z_fissure"],
                "cpu",
                clean=ds._inner.clean_content,
            )
            contrasts.append(
                [
                    contrast_metrics(
                        images[VIEWS.index(v)],
                        reference["images"][VIEWS.index(v)].numpy()[0],
                        lesion.numpy().astype(bool),
                        reference["tissue"],
                    )["matched_contrast"]
                    for v in args.views
                ]
            )
        if (idx + 1) % 32 == 0 or idx + 1 == count:
            print(f"  {namespace or split}: {idx+1}/{count} subjects", flush=True)
    result.update(truth=np.array(truth), raw=np.array(raw))
    if contrast:
        result["contrast"] = np.array(contrasts)
    return result, dict(
        split=split,
        subject_ids=list(range(count)),
        accepted_seeds=seeds,
        generator_seed=int(ds._inner.seed),
        input_sha256=image_hash.hexdigest(),
        intensity=cfg.get("synthetic_lesion_intensity", "fixed"),
    )


def interventions(cfg, split, count, offset, epsilons, args, extract, banks, namespace):
    ds, _ = restore_dataset(cfg, max(64, offset + count), split)
    capacity, accepted = count * len(epsilons) * 9, 0
    result, truth, raw, rows, seeds = {}, [], [], [], []
    image_hash = hashlib.sha256()
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
                    zero_image=False,
                    error="",
                )
                try:
                    controls = [ctx["lat"]["z_content"].clone() for _ in range(2)]
                    controls[0][k] -= eps
                    controls[1][k] += eps
                    endpoints = [render(ds, ctx, z) for z in controls]
                    images = torch.stack([selected_images(im, args.views) for im, _ in endpoints])
                    encoded = extract(images)
                    for key, value in encoded.items():
                        if key not in result:
                            result[key] = banks.allocate(f"{namespace}/{key}", capacity, value)
                        result[key][accepted] = value
                    truth.append(np.stack([semantic_target(ds, z, m) for z, (_, m) in zip(controls, endpoints)]))
                    raw.append(np.stack([z.numpy() for z in controls]))
                    image_hash.update(images.numpy().tobytes())
                    row.update(
                        valid=True,
                        pair_index=accepted,
                        zero_image=bool((images[1] - images[0]).abs().max() <= 1e-6),
                    )
                    accepted += 1
                except LesionPlacementError as error:
                    row["error"] = str(error)
                rows.append(row)
        print(
            f"  {namespace}: {idx-offset+1}/{count} subjects, {accepted} valid pairs",
            flush=True,
        )
    if not accepted:
        raise ValueError("No valid interventions; failed subjects were not redrawn")
    result = {k: a[:accepted] for k, a in result.items()}
    result.update(truth=np.array(truth), raw=np.array(raw), rows=rows)
    return result, dict(
        split=split,
        subject_ids=list(range(offset, offset + count)),
        accepted_seeds=seeds,
        generator_seed=int(ds._inner.seed),
        input_sha256=image_hash.hexdigest(),
        requested_pairs=capacity,
        valid_pairs=accepted,
        failed_pairs=capacity - accepted,
    )
