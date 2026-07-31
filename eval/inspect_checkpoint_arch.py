"""Derive the architecture a VQ-VAE checkpoint was trained with, straight from its keys.

Rebuilding a model from ``settings.json`` is a guess: the JSON can drift from the weights
(mixed-vintage checkout, resumed run, checkpoint copied from another run), and
``load_state_dict(strict=False)`` then reports the damage only as a list of key names.
This reads the checkpoint itself, prints the kwargs that make the load clean, and flags
every ``settings.json`` entry that disagrees.

Usage:
    python -m eval.inspect_checkpoint_arch /path/to/vqvae_model.pt
    python -m eval.inspect_checkpoint_arch /path/to/vqvae_model.pt --rebuild
"""

import argparse
import json
import os
import re
from collections import defaultdict

import torch


def load_encoder_state(path):
    """Return the VQ-VAE state dict with the 'online.'/'module.' prefixes stripped."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["encoders"] if isinstance(ckpt, dict) and "encoders" in ckpt else ckpt
    clean = {}
    for k, v in sd.items():
        for p in ("online.", "module."):
            if k.startswith(p):
                k = k[len(p) :]
        clean[k] = v
    return clean


def _indices(keys, pattern):
    """Sorted unique integer group-1 matches of `pattern` over `keys`."""
    out = set()
    for k in keys:
        m = re.match(pattern, k)
        if m:
            out.add(int(m.group(1)))
    return sorted(out)


def _decoder_tail_has_norm(sd, lvl=0):
    """True when decoder `lvl` ends in a normalization layer (final_recon_norm)."""
    # Style-injecting decoders keep the output conv outside `.layers`.
    for tail in ("final_conv", "output_conv"):
        if f"decoders.{lvl}.{tail}.1.weight" in sd:
            return True
        if f"decoders.{lvl}.{tail}.0.weight" in sd:
            return False
    idxs = _indices(sd, rf"decoders\.{lvl}\.layers\.(\d+)\.")
    if not idxs:
        return None
    last = idxs[-1]
    w = sd.get(f"decoders.{lvl}.layers.{last}.weight")
    if w is None:
        return None
    return w.dim() == 1  # 1-D affine -> norm; 5-D kernel -> conv


def _norm_type(sd, prefix):
    """ChannelLayerNorm3d nests its affine under '.norm.'; GroupNorm stores it flat."""
    for k in sd:
        if k.startswith(prefix) and k.endswith(".norm.weight"):
            return "layer"
    return "group"


def infer_arch(sd):
    keys = list(sd)
    a = {}

    nb_levels = len(_indices(keys, r"decoders\.(\d+)\."))
    a["vqvae_nb_levels"] = nb_levels
    a["vqvae_embed_dim"] = sd["codebooks.0.conv_in.weight"].shape[0]
    a["vqvae_hidden_channels"] = sd["decoders.0.layers.0.weight"].shape[0]
    a["vqvae_nb_entries"] = [sd[f"codebooks.{i}.embed"].shape[1] for i in range(nb_levels)]
    a["vqvae_nb_res_layers"] = len(_indices(keys, r"decoders\.0\.layers\.1\.stack\.(\d+)\."))
    a["vqvae_res_channels"] = sd["decoders.0.layers.1.stack.0.layers.0.weight"].shape[0]

    # Content width per level, from each codebook's input conv. Levels below the
    # coarsest also carry embed_dim of conditioning from the level above.
    embed_dim = a["vqvae_embed_dim"]
    content_ch = {}
    for lvl in range(nb_levels):
        w = sd.get(f"codebooks.{lvl}.conv_in.weight")
        if w is None:
            continue
        content_ch[lvl] = w.shape[1] if lvl == nb_levels - 1 else w.shape[1] - embed_dim
    a["_content_channels_per_level"] = content_ch

    # A content/style split leaves parameters behind: the mask and the split norm.
    mask_levels = sorted(
        set(_indices(keys, r"fixed_mask_(\d+)"))
        | set(_indices(keys, r"channel_logits\.(\d+)"))
        | set(_indices(keys, r"split_gate_logits\.(\d+)"))
        | set(_indices(keys, r"content_norms\.(\d+)\."))
    )
    a["content_style_levels"] = mask_levels or None
    if any(k.startswith("fixed_mask_") for k in keys):
        a["mask_mode"] = "fixed"
    elif any(k.startswith("split_gate_logits.") for k in keys):
        a["mask_mode"] = "learned_split"
    elif any(k.startswith("channel_logits.") for k in keys):
        a["mask_mode"] = "learned"
    else:
        a["mask_mode"] = None  # "onthefly" or no split at all — leaves no parameters

    hidden = a["vqvae_hidden_channels"]
    if mask_levels:
        first = mask_levels[0]
        a["content_size"] = content_ch.get(first, hidden)
        a["style_size"] = hidden - a["content_size"]
        a["content_ratios"] = [content_ch[lvl] / hidden for lvl in mask_levels if lvl in content_ch]
    else:
        # No split: content fills the whole width. style_size MUST be 0 — clamping it
        # to 1 fabricates a mask buffer and a zero-width style codebook per level.
        a["content_size"] = hidden
        a["style_size"] = 0
        a["content_ratios"] = None

    a["quantize_style"] = any(k.startswith("style_codebooks.") for k in keys)
    a["separate_style_codebooks"] = any(k.startswith("style_codebooks_v1.") for k in keys)
    a["separate_encoders"] = any(k.startswith("encoders_v1.") for k in keys)
    a["separate_content_codebooks"] = any(k.startswith("codebooks_v1.") for k in keys)
    a["use_content_projection"] = any(k.startswith("content_projections.") for k in keys)
    if a["quantize_style"]:
        a["style_embed_dim"] = sd["style_codebooks.0.conv_in.weight"].shape[0]
        a["style_nb_entries"] = sd["style_codebooks.0.embed"].shape[1]

    # Style reaches the decoder only in shapes: a separate output conv (concat/film)
    # or a widened input conv (input mode).
    if any(re.match(r"decoders\.\d+\.film_layers\.", k) for k in keys):
        a["style_injection_mode"] = "film"
        a["inject_style_to_decoder"] = True
    elif any(re.match(r"decoders\.\d+\.final_conv\.", k) for k in keys):
        a["style_injection_mode"] = "concat"
        a["inject_style_to_decoder"] = True
    else:
        dec_in = sd["decoders.0.layers.0.weight"].shape[1]
        widened = dec_in > embed_dim * nb_levels
        a["style_injection_mode"] = "input" if widened else None
        a["inject_style_to_decoder"] = bool(widened)

    tail = _decoder_tail_has_norm(sd, 0)
    a["no_final_recon_norm"] = (not tail) if tail is not None else None
    a["norm_type"] = _norm_type(sd, "encoders.")
    a["decoder_norm_type"] = _norm_type(sd, "decoders.")
    return a


# settings.json key -> how to read the inferred value for comparison.
_COMPARED = [
    "vqvae_nb_levels",
    "vqvae_hidden_channels",
    "vqvae_embed_dim",
    "vqvae_res_channels",
    "vqvae_nb_res_layers",
    "content_size",
    "style_size",
    "content_ratios",
    "content_style_levels",
    "mask_mode",
    "quantize_style",
    "separate_style_codebooks",
    "separate_encoders",
    "separate_content_codebooks",
    "use_content_projection",
    "inject_style_to_decoder",
    "no_final_recon_norm",
    "norm_type",
    "decoder_norm_type",
]


def _equalish(inferred, stored):
    if inferred is None:
        return True  # no evidence in the weights — settings.json wins
    if isinstance(inferred, list) and isinstance(stored, list) and len(inferred) == len(stored):
        return all(abs(a - b) < 1e-6 if isinstance(a, float) else a == b for a, b in zip(inferred, stored))
    if isinstance(inferred, bool) or isinstance(stored, bool):
        return bool(inferred) == bool(stored)
    return inferred == stored


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--settings", default=None, help="Default: settings.json next to the checkpoint.")
    ap.add_argument(
        "--rebuild", action="store_true", help="Build a VQVAE from the inferred kwargs and load it strictly."
    )
    args = ap.parse_args()

    sd = load_encoder_state(args.checkpoint)
    arch = infer_arch(sd)

    print(f"\n=== inferred from {args.checkpoint} ({len(sd)} tensors) ===")
    for k, v in arch.items():
        print(f"  {k:28s} = {v}")

    settings_path = args.settings or os.path.join(os.path.dirname(args.checkpoint), "settings.json")
    settings = None
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            settings = json.load(f)
        print(f"\n=== vs {settings_path} ===")
        disagree = []
        for k in _COMPARED:
            if k not in settings:
                continue
            if not _equalish(arch.get(k), settings[k]):
                disagree.append((k, arch.get(k), settings[k]))
        if disagree:
            print("  MISMATCH (checkpoint wins — settings.json does not describe these weights):")
            for k, ckpt_v, set_v in disagree:
                print(f"    {k:26s} checkpoint={ckpt_v!r}   settings.json={set_v!r}")
        else:
            print("  settings.json agrees with the weights on every shape-sensitive flag.")
    else:
        print(f"\n(no settings.json at {settings_path})")

    if args.rebuild:
        import models.vqvae as vqvae

        s = settings or {}
        model = vqvae.VQVAE(
            in_channels=1,
            hidden_channels=arch["vqvae_hidden_channels"],
            res_channels=arch["vqvae_res_channels"],
            nb_res_layers=arch["vqvae_nb_res_layers"],
            nb_levels=arch["vqvae_nb_levels"],
            embed_dim=arch["vqvae_embed_dim"],
            nb_entries=arch["vqvae_nb_entries"],
            scaling_rates=s.get("vqvae_scaling_rates", [2, 2, 2]),
            content_size=arch["content_size"],
            style_size=arch["style_size"],
            content_ratios=arch["content_ratios"],
            content_style_levels=arch["content_style_levels"] or [0],
            mask_mode=arch["mask_mode"] or s.get("mask_mode", "fixed"),
            separate_encoders=arch["separate_encoders"],
            pass_full_to_next_level=s.get("pass_full_to_next_level", False),
            narrow_encoder_input=s.get("narrow_encoder_input", False),
            use_content_projection=arch["use_content_projection"],
            inject_style_to_decoder=arch["inject_style_to_decoder"],
            quantize_style=arch["quantize_style"],
            separate_content_codebooks=arch["separate_content_codebooks"],
            separate_style_codebooks=arch["separate_style_codebooks"],
            style_embed_dim=arch.get("style_embed_dim"),
            style_nb_entries=arch.get("style_nb_entries"),
            style_injection_mode=arch["style_injection_mode"] or "concat",
            top_level_recon_only=s.get("top_level_recon_only", False),
            skip_decoder_concat_levels=s.get("skip_decoder_concat_levels", None),
            final_recon_norm=not arch["no_final_recon_norm"],
            norm_type=arch["norm_type"],
            decoder_norm_type=arch["decoder_norm_type"],
            latent_mask=s.get("latent_mask", False),
            latent_mask_thresh=s.get("latent_mask_thresh", 0.0),
        )
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("\n=== rebuild check ===")
        if not missing and not unexpected:
            print("  CLEAN: every trained tensor loaded, nothing left at fresh init.")
        else:
            print(f"  still missing   ({len(missing)}): {missing[:8]}")
            print(f"  still unexpected({len(unexpected)}): {unexpected[:8]}")
            groups = defaultdict(int)
            for k in list(missing) + list(unexpected):
                groups[k.split(".")[0]] += 1
            print(f"  by top-level module: {dict(groups)}")


if __name__ == "__main__":
    main()
