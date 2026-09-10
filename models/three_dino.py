"""Full-volume adapter for the external, unmodified AICONSlab/3DINO model.

Upstream: https://github.com/AICONSlab/3DINO (see eval/3DINO.md).
This module does not vendor the upstream implementation or pretrained weights.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


def import_upstream(repo):
    root = Path(repo).expanduser().resolve()
    if not (root / "dinov2/models/vision_transformer.py").is_file():
        raise FileNotFoundError(f"--three-dino-repo must point to an AICONSlab/3DINO checkout: {root}")
    loaded = sys.modules.get("dinov2")
    if loaded is not None and Path(loaded.__file__).resolve().parent != root / "dinov2":
        raise RuntimeError("A different dinov2 package is already imported; run 3DINO in a fresh process")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("dinov2.models.vision_transformer")
    if not hasattr(module, "DinoVisionTransformer3d"):
        raise RuntimeError("The supplied checkout contains 2D DINOv2 rather than AICONSlab/3DINO")
    return root


def build_backbone(spec):
    from dinov2.models import build_model

    if spec["student"]["arch"] != "vit_large_3d":
        raise ValueError("The 3DINO-ViT adapter expects the published vit_large_3d architecture")
    model, _ = build_model(argparse.Namespace(**spec["student"]), only_teacher=True, img_size=spec["img_size"])
    return model


def load_backbone_weights(backbone, path):
    """Accept official teacher checkpoints or an exported bare backbone, strictly."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("teacher", "state_dict"):
        if key in state:
            state = state[key]
            break
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if any(key.startswith("backbone.") for key in state):
        # Official teacher checkpoints also contain the pretraining DINO/iBOT heads.
        state = {key.removeprefix("backbone."): value for key, value in state.items() if key.startswith("backbone.")}
    backbone.load_state_dict(state, strict=True)


class ThreeDINOEncoder(nn.Module):
    """Expose the official normalized CLS/patch tokens and checkpoint export."""

    def __init__(self, backbone, spec, provenance):
        super().__init__()
        self.backbone = backbone
        self.spec = spec
        self.provenance = provenance
        self.use_gradient_checkpointing = False

    def forward(self, pixel_values):
        if pixel_values.ndim != 5 or pixel_values.shape[1] != 1:
            raise ValueError("3DINO input must have shape (B, 1, X, Y, Z)")
        if self.use_gradient_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            features = checkpoint(self.backbone.forward_features, pixel_values, use_reentrant=False)
        else:
            features = self.backbone.forward_features(pixel_values)
        hidden = torch.cat([features["x_norm_clstoken"][:, None], features["x_norm_patchtokens"]], dim=1)
        return SimpleNamespace(last_hidden_state=hidden)

    def gradient_checkpointing_enable(self, **kwargs):
        self.use_gradient_checkpointing = True

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.backbone.state_dict(), directory / "model.pt")
        (directory / "config.json").write_text(
            json.dumps(dict(backend="3dino", spec=self.spec, provenance=self.provenance), indent=2) + "\n"
        )


def load_encoder(cli):
    root = import_upstream(cli.three_dino_repo)
    path = Path(cli.three_dino_weights).expanduser() if cli.three_dino_weights else None
    if not cli.random_init and (path is None or not path.exists()):
        raise FileNotFoundError(
            "3DINO needs --three-dino-weights pointing to the downloaded pretrained .pth file "
            "or a fine-tuned encoder directory. Request weights at https://huggingface.co/AICONSlab/3DINO-ViT. "
            "Use --random-init only for an explicitly untrained baseline."
        )
    if path is not None and path.is_dir():
        saved = json.loads((path / "config.json").read_text())
        if saved.get("backend") != "3dino":
            raise ValueError("The encoder directory is not a 3DINO export")
        spec = saved["spec"]
        path = path / "model.pt"
    else:
        from dinov2.configs import load_and_merge_config_3d
        from omegaconf import OmegaConf

        config = load_and_merge_config_3d("train/vit3d_highres")
        spec = dict(
            student=OmegaConf.to_container(config.student, resolve=True), img_size=int(config.crops.global_crops_size)
        )
    if cli.patch_size is not None and cli.patch_size != spec["student"]["patch_size"]:
        raise ValueError("--patch-size must match the pretrained 3DINO architecture")
    if cli.random_init:
        torch.manual_seed(cli.model_seed)
    backbone = build_backbone(spec)
    if not cli.random_init:
        load_backbone_weights(backbone, path)
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    provenance = dict(
        repository="https://github.com/AICONSlab/3DINO",
        revision=revision,
        checkpoint=str(path.resolve()) if path and not cli.random_init else None,
    )
    model = ThreeDINOEncoder(backbone, spec, provenance)
    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = getattr(torch, cli.dtype)
    config = SimpleNamespace(
        patch_size=spec["student"]["patch_size"], hidden_size=backbone.embed_dim, num_register_tokens=0, backend="3dino"
    )
    return model.to(device=device, dtype=dtype).eval(), device, dtype, config


def prepare_volumes(volumes, cli, window):
    """Single-channel volumes -> resized float input in [-1,1], without RGB or slicing."""
    if volumes.ndim != 5 or volumes.shape[1] != 1 or min(volumes.shape[-3:]) < 2:
        raise ValueError("3DINO requires genuine 3D volumes shaped (B, 1, X, Y, Z), not 2D images")
    x = volumes.float()
    if not torch.isfinite(x).all():
        raise ValueError("3DINO input contains non-finite voxels")
    if cli.window == "per_volume":
        bounds = torch.quantile(x.flatten(1), x.new_tensor(cli.window_pct) / 100, dim=1)
        lo, hi = (b.reshape(-1, 1, 1, 1, 1) for b in bounds)
    elif cli.window == "dataset":
        lo, hi = (x.new_tensor(float(v)) for v in window)
        if not torch.isfinite(lo) or not torch.isfinite(hi) or hi < lo:
            raise ValueError("Invalid saved dataset intensity window")
    else:
        raise ValueError("3DINO supports --window per_volume or dataset")
    span = hi - lo
    x = torch.where(span > 0, ((x - lo) / span.clamp_min(torch.finfo(x.dtype).eps) * 2 - 1).clamp(-1, 1), 0.0)
    size = (cli.volume_size,) * 3
    if tuple(x.shape[-3:]) != size:
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
    return x


def encode_volumes(volumes, encoder, config, cli, device, window, mean=None, std=None):
    """One embedding per full volume; the same differentiable path is used for training/eval."""
    size, patch = cli.volume_size, config.patch_size
    if size % patch:
        raise ValueError(f"--volume-size {size} must be a multiple of the 3DINO patch size {patch}")
    pixels = prepare_volumes(volumes, cli, window)
    dtype = next(encoder.parameters()).dtype
    hidden = encoder(pixel_values=pixels.to(device=device, dtype=dtype)).last_hidden_state.float()
    patches = hidden[:, 1:]
    grid = (size // patch,) * 3
    if patches.shape[1] != grid[0] ** 3:
        raise ValueError("3DINO token count disagrees with the volume's 3D patch grid")
    if cli.token_pool == "cls":
        return hidden[:, 0]
    if cli.token_pool == "mean":
        return patches.mean(1)
    if cli.token_pool == "cls_mean":
        return torch.cat([hidden[:, 0], patches.mean(1)], dim=-1)
    if cli.token_pool == "grid":
        maps = patches.transpose(1, 2).reshape(len(hidden), hidden.shape[-1], *grid)
        return F.adaptive_avg_pool3d(maps, cli.grid_size).flatten(1)
    raise ValueError(f"Unknown token pooling: {cli.token_pool}")


def extract(dataset, encoder, device, dtype, config, cli, window, mean, std):
    import numpy as np

    from eval.dinov3_embed_synthetic import raw_voxel_features

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cli.volume_batch, num_workers=cli.num_workers, shuffle=False
    )
    embeddings, latents, raw = {}, {}, {}
    with torch.no_grad():
        for batch in loader:
            for key in ("z_content", "z_style_v1", "z_style_v2", "causal_adj"):
                if key in batch["gt_latents"]:
                    latents.setdefault(key, []).append(np.asarray(batch["gt_latents"][key]))
            for view in cli.views:
                volumes = batch["image"][view - 1]
                features = encode_volumes(volumes, encoder, config, cli, device, window)
                embeddings.setdefault(view, []).append(features.cpu().numpy()[:, None])
                if cli.raw_grid:
                    raw.setdefault(view, []).append(raw_voxel_features(volumes, cli.raw_grid))
    return (
        {k: np.concatenate(v) for k, v in embeddings.items()},
        {k: np.concatenate(v) for k, v in latents.items()},
        {k: np.concatenate(v) for k, v in raw.items()},
        ["volume"],
    )
