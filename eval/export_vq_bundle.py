#!/usr/bin/env python
"""Export a VQ-VAE run's features in the shared bundle format, so DINO and VQ meet.

    python -m eval.export_vq_bundle --run-dir results/synthetic/RUN \\
        --level 0 --pooling gap --block content --num-samples 2000 \\
        --out results/bundles/vq_content_gap.npz

The output is byte-for-byte the contract ``eval.dinov3_embed_synthetic`` writes -- it
calls that module's own ``save`` -- so ``eval.dinov3_identifiability`` scores a VQ-VAE
bundle through the identical code path it scores a DINO bundle with.  That is the point.
Two scripts that each implement "cross-validated ridge R² minus a permutation null" will
agree on the definition and still disagree on the number, and this project has already
paid for that twice; one scorer over two feature files cannot.

What this fixes relative to running the two panels separately
------------------------------------------------------------
* **One probe protocol.**  Same CV splits, same permutation nulls, same PCA rule, same
  floor handling, because it is the same function.
* **One evaluation set, verifiably.**  Both bundles carry a factor digest
  (``eval/bundle_identity.py``); ``eval.compare_bundles`` refuses to put two bundles in
  one table unless they describe the same rows.  Matching ``--num-samples`` is not
  evidence of that, and directory names are not either.
* **A frozen content mask.**  ``--freeze-content-mask`` (on by default) pins the Gumbel
  mask's channel selection to the first batch.  With ``mask_mode=onthefly`` the mask is
  redrawn per forward, so without this a stacked feature column does not describe one
  physical channel.  The extractor warns whenever the selection actually moves.
* **The choices recorded, not assumed.**  ``normalization_location`` and
  ``foreground_mask`` go into ``meta`` because they change what is being compared and are
  invisible in the arrays.

What it does not fix
--------------------
The representations are still different objects and no flag makes them the same one.  VQ
features here are **pre-codebook encoder outputs**, not quantized codes -- calling a score
over them "VQ code identifiability" would overstate it.  ``--block content`` is a learned
partition that DINO has no counterpart for, so a DINO embedding is only ever comparable to
``--block all``; export both and read the content block as the extra question it is.  The
input preprocessing differs by construction (DINO clips, rescales and resizes; the VQ
encoder reads generator-normalized volumes), which is a real difference between the
pipelines and not something the scorer can subtract.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

BLOCKS = ("content", "style", "all")
#: Index into the tuple ``eval.dci._extract_synthetic_representations`` returns per level.
_VIEW1 = {"content": 0, "style": 1}
_VIEW2 = {"content": 2, "style": 3}


def parse_pooling(text):
    """``gap`` / ``stats`` / ``D,H,W`` -> what the extractor takes, plus a label."""
    if text in ("gap", "stats"):
        return text, text
    try:
        grid = tuple(int(part) for part in text.replace("x", ",").split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--pooling {text!r} is not 'gap', 'stats' or 'D,H,W'") from exc
    if len(grid) != 3 or any(d < 1 for d in grid):
        raise argparse.ArgumentTypeError(f"--pooling {text!r} must be three positive integers")
    return grid, "x".join(str(d) for d in grid)


def parse_grid(text):
    if text is None:
        return None
    grid, _label = parse_pooling(text)
    if isinstance(grid, str):
        raise argparse.ArgumentTypeError("--raw-grid must be 'D,H,W'")
    return grid


def select_block(level_tuple, block, view):
    """The requested feature block for one view, or ``None`` when the level has no split.

    ``all`` concatenates content and style.  That reorders columns relative to the
    encoder's own channel order, which is immaterial to a standardized ridge probe and to
    block-MCC (both are invariant to a permutation of input columns) but would matter to a
    per-channel readout, so it is stated rather than left to be discovered.
    """
    import numpy as np

    if block == "all":
        parts = [level_tuple[_VIEW1[name] if view == 1 else _VIEW2[name]] for name in ("content", "style")]
        parts = [p for p in parts if p is not None]
        return np.concatenate(parts, axis=1) if parts else None
    return level_tuple[_VIEW1[block] if view == 1 else _VIEW2[block]]


def raw_voxel_baseline(dataset, views, grid, batch_size, num_workers):
    """Downsampled voxel intensities per view -- the trivial baseline a model must beat.

    Uses ``eval.dinov3_embed_synthetic.raw_voxel_features`` so the VQ bundle's voxel column
    is the same baseline the DINO bundle's is, rather than a second implementation of
    "average-pool the volume".
    """
    import numpy as np
    import torch

    from eval.dinov3_embed_synthetic import raw_voxel_features

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    per_view = {}
    for batch in loader:
        for view in views:
            per_view.setdefault(view, []).append(raw_voxel_features(batch["image"][view - 1], grid))
    return {view: np.concatenate(parts) for view, parts in per_view.items()}


def build(cli):
    """Extract the run's features and return ``(embeddings, latents, raw, meta)``."""
    import numpy as np

    from eval.bundle_identity import identity_record
    from eval.dci import _extract_synthetic_representations
    from eval.dinov3_embed_synthetic import git_sha
    from eval.identifiability_report import _causal_adjacency
    from eval.run_dci_compare import _resolve_checkpoint
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir, load_run_args

    started = time.time()
    args = load_run_args(cli.run_dir)
    dataset = build_synthetic_test_set(args, cli.num_samples, cache=not cli.no_cache, causal=cli.causal)
    settings = {
        key: value
        for key, value in vars(args).items()
        if key.startswith("synthetic_") and not key.startswith("synthetic_num_")
    }
    settings["synthetic_num_samples"] = len(dataset)

    checkpoint = None if cli.random_init else _resolve_checkpoint(cli.run_dir, cli.checkpoint)
    model, _args, device = load_model_from_run_dir(
        cli.run_dir, checkpoint, cli.device, random_init=cli.random_init, seed=cli.model_seed
    )
    pooling, pooling_label = cli.pooling
    level_data, z_content, z_style_v1, z_style_v2 = _extract_synthetic_representations(
        model,
        dataset,
        device,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        pooling=pooling,
        freeze_content_mask=cli.freeze_content_mask,
    )
    del model
    if cli.level not in level_data:
        raise SystemExit(f"level {cli.level} not in encoder output; available: {sorted(level_data)}")
    level_tuple = level_data[cli.level]
    info = level_tuple[4]

    embeddings = {}
    for view in cli.views:
        block = select_block(level_tuple, cli.block, view)
        if block is None:
            if view == 1:
                raise SystemExit(
                    f"level {cli.level} has no {cli.block!r} block "
                    f"(has_split={info['has_split']}); export --block all or pick another --level"
                )
            logger.warning("View %d has no features at this level; writing view 1 only", view)
            continue
        embeddings[view] = np.asarray(block)

    raw = {}
    if cli.raw_grid:
        raw = raw_voxel_baseline(dataset, list(embeddings), cli.raw_grid, cli.batch_size, cli.num_workers)

    latents = {"z_content": z_content, "z_style_v1": z_style_v1, "z_style_v2": z_style_v2}
    adjacency = _causal_adjacency(dataset)
    if adjacency is not None:
        latents["causal_adj"] = np.asarray(adjacency)

    meta = dict(
        model_id=f"vqvae:{Path(cli.run_dir).name}",
        backbone="vqvae",
        source="eval.export_vq_bundle",
        run_dir=str(cli.run_dir),
        checkpoint=str(checkpoint) if checkpoint else None,
        random_init=cli.random_init,
        model_seed=cli.model_seed,
        architecture="VQVAE",
        level=cli.level,
        block=cli.block,
        pooling=pooling_label,
        # The three settings that make two VQ extractions of one checkpoint disagree.
        # None is inferable from the arrays, so a bundle that omitted them could not be
        # told apart from one that chose differently.
        freeze_content_mask=cli.freeze_content_mask,
        normalization_location="pre_content_norms" if pooling != "stats" else "post_content_norms",
        foreground_mask=False,
        n_content_channels=info["n_content_channels"],
        n_style_channels=info["n_style_channels"],
        has_split=info["has_split"],
        embedding_dim={f"view{view}": int(features.shape[1]) for view, features in embeddings.items()},
        views=list(embeddings),
        num_samples=int(len(dataset)),
        causal=adjacency is not None,
        content_factor_names=list(info["content_names"]),
        style_factor_names=list(info["style_names"]),
        generator=settings,
        raw_grid=list(cli.raw_grid) if cli.raw_grid else None,
        git_sha=git_sha(),
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        elapsed_seconds=round(time.time() - started, 1),
        **identity_record(latents, settings, len(dataset)),
    )
    return embeddings, latents, raw, meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="Training run directory with settings.json")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint filename inside --run-dir")
    parser.add_argument("--out", type=Path, required=True, help="Destination .npz")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--level", type=int, default=0, help="Encoder level to export")
    parser.add_argument(
        "--block",
        default="content",
        choices=BLOCKS,
        help="Which VQ columns become the features. 'all' is the block comparable to a "
        "DINO embedding; 'content' is the learned partition DINO has no counterpart for.",
    )
    parser.add_argument("--pooling", type=parse_pooling, default=parse_pooling("gap"), help="gap | stats | D,H,W")
    parser.add_argument("--views", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    parser.add_argument(
        "--raw-grid",
        type=parse_grid,
        default=None,
        help="Also store a downsampled-voxel baseline at this grid, e.g. 4,4,4",
    )
    parser.add_argument(
        "--causal",
        default="match",
        choices=["match", "iid"],
        help="'match' reproduces the run's training SCM (compare like with like); 'iid' "
        "forces decorrelated factors, which is a different experiment, not a control.",
    )
    parser.add_argument("--random-init", action="store_true", help="Export the untrained twin (the floor)")
    parser.add_argument("--model-seed", type=int, default=0, help="Seed for --random-init weights")
    parser.add_argument(
        "--no-freeze-content-mask",
        dest="freeze_content_mask",
        action="store_false",
        help="Read each batch's Gumbel mask instead of pinning the first batch's channels. "
        "Reproduces the older extraction; only meaningful for mask_mode=onthefly checkpoints.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true", help="Do not keep rendered volumes in RAM")
    parser.add_argument("--device", default=None)

    cli = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    cli.causal = cli.causal == "match"

    from eval.dinov3_embed_synthetic import save

    embeddings, latents, raw, meta = build(cli)
    arrays = save(cli.out, embeddings, latents, raw, [], meta)
    print(f"\nSaved {cli.out.resolve()}")
    for key, value in sorted(arrays.items()):
        if key != "meta":
            print(f"  {key:<14} {value.shape}")
    print(f"  factor digest  {meta['factor_digest'][:16]}  ({meta['n_rows']} rows)")
    print(f"\nScore it:\n  python -m eval.dinov3_identifiability --embeddings {cli.out}")
    print(f"Compare it:\n  python -m eval.compare_bundles --bundles vq={cli.out} dino=<dino.npz>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
