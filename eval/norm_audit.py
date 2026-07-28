"""Is `--norm-type layer` actually applied everywhere it needs to be?

The background-leak argument assumes ONE thing about the architecture: that no operation
between the input and the probed features mixes information across spatial positions.
`--norm-type layer` is supposed to guarantee that. This script checks the guarantee on the
REAL model rather than on a standalone `Encoder`, because the production path has extras a
bare encoder does not: `content_norms` (SplitGroupNorm), the dual-view branch, latent
masking, and the pooling that produces `encoder_pools`.

Two independent checks, and they can disagree:

  1. STATIC INVENTORY. Walk every module and classify each normalization as spatially
     coupling (any nn.GroupNorm / nn.BatchNorm, whose statistics pool over positions) or
     not (ChannelLayerNorm3d, which normalizes across channels within one voxel). Reported
     split by encoder path vs decoder path, since only the encoder feeds the probes.

  2. DYNAMIC REACH. Differentiate an actual probed feature -- taken from
     `model(x, pool_only=True, patch_grid=...)`, exactly as the eval scripts read it -- with
     respect to the input, for a position whose receptive field contains no brain. Any
     nonzero gradient on brain voxels means information reaches that position. This catches
     coupling the inventory would miss (a stray mean, an attention op, a global pool),
     because it tests the composed function instead of a list of layer types.

A clean result is: zero coupling norms on the encoder path AND gradient confined to a
neighbourhood that excludes the brain. Anything else localises the problem.

Usage:
  python -m eval.norm_audit --norm-type layer --res 128
  python -m eval.norm_audit --run-dir results/synthetic/<run>     # audit a trained run's config
"""

from __future__ import annotations

import argparse
import logging
import types

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

COUPLING = (nn.GroupNorm, nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d, nn.InstanceNorm3d)


def _stub_optional_deps():
    """Import `training.main_multimodal` on a machine without MONAI/TensorBoard.

    The point is to build the model through the SAME `build_vqvae` the training script
    uses -- its docstring calls itself the single source of truth for the args->model
    mapping -- rather than re-deriving the mapping here and auditing a model that is not
    the one that gets trained. Neither stubbed package affects module construction.
    """
    import sys

    from eval.background_leak_mechanism import _stub_utils_if_needed

    _stub_utils_if_needed()
    for name, attrs in (("tensorboard", {}), ("lpips", {"LPIPS": object}), ("wandb", {})):
        try:
            __import__(name)
        except ImportError:
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m
    try:
        import torch.utils.tensorboard  # noqa: F401
    except Exception:
        w = types.ModuleType("torch.utils.tensorboard")
        w.SummaryWriter = object
        sys.modules["torch.utils.tensorboard"] = w


def _get_builder():
    """Prefer training.main_multimodal.build_vqvae (the single source of truth for the
    args->model mapping). Fall back to constructing VQVAE directly when the training-only
    dependencies (MONAI/LPIPS/TensorBoard) are absent, mirroring the same kwargs.
    """
    _stub_optional_deps()
    try:
        from training.main_multimodal import build_vqvae

        logger.info("using training.main_multimodal.build_vqvae")
        return build_vqvae
    except Exception as e:  # training-only deps missing
        logger.info("build_vqvae unavailable (%s); constructing VQVAE directly", type(e).__name__)
        from models.vqvae import VQVAE

        def build(a):
            return VQVAE(
                in_channels=1,
                hidden_channels=a.vqvae_hidden_channels,
                res_channels=a.vqvae_res_channels,
                nb_res_layers=2,
                nb_levels=a.vqvae_nb_levels,
                embed_dim=a.vqvae_embed_dim,
                nb_entries=a.vqvae_nb_entries,
                scaling_rates=a.vqvae_scaling_rates,
                use_checkpoint=False,
                content_size=len(a.content_indices[0]),
                style_size=len(a.style_indices),
                inject_style_to_decoder=a.inject_style_to_decoder,
                content_style_levels=a.content_style_levels,
                content_ratios=a.content_ratios,
                separate_encoders=a.separate_encoders,
                separate_content_codebooks=a.separate_content_codebooks,
                mask_mode=a.mask_mode,
                quantize_style=a.quantize_style,
                separate_style_codebooks=a.separate_style_codebooks,
                style_injection_mode=a.style_injection_mode,
                pass_full_to_next_level=a.pass_full_to_next_level,
                final_recon_norm=not a.no_final_recon_norm,
                norm_type=a.norm_type,
                decoder_norm_type=a.decoder_norm_type,
            )

        return build


def _default_args(**over):
    """Minimal args namespace matching experiments/synthetic_defaults.yaml + overrides."""
    a = types.SimpleNamespace(
        vqvae_hidden_channels=48,
        vqvae_res_channels=32,
        vqvae_nb_levels=1,
        vqvae_embed_dim=48,
        vqvae_nb_entries=[256],
        vqvae_scaling_rates=[4],
        gradient_checkpointing=False,
        content_indices=[list(range(48))],
        style_indices=[],
        inject_style_to_decoder=True,
        content_style_levels=[0],
        content_ratios=[0.95],
        separate_encoders=True,
        separate_content_codebooks=True,
        separate_style_codebooks=True,
        mask_mode="fixed",
        quantize_style=True,
        style_injection_mode="film",
        pass_full_to_next_level=True,
        no_final_recon_norm=True,
        norm_type="layer",
        decoder_norm_type="group",
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


def classify(model):
    """-> (coupling, safe) lists of (qualified_name, type, path_tag)."""
    coupling, safe = [], []
    for name, mod in model.named_modules():
        tag = "DECODER" if (".decoders" in name or ".upscalers" in name or name.startswith("decoders")) else "encoder"
        if isinstance(mod, COUPLING):
            coupling.append((name, type(mod).__name__, tag))
        elif type(mod).__name__ == "ChannelLayerNorm3d" or isinstance(mod, nn.LayerNorm):
            safe.append((name, type(mod).__name__, tag))
    return coupling, safe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None, help="Take the architecture from a run's settings.json.")
    ap.add_argument("--norm-type", default="layer", choices=["group", "layer"])
    ap.add_argument("--decoder-norm-type", default="group", choices=["group", "layer"])
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--grid", type=int, default=None)
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    build = _get_builder()

    if cli.run_dir:
        from eval.run_dci_synthetic import load_model_from_run_dir

        model, args, _ = load_model_from_run_dir(cli.run_dir, None, torch.device("cpu"))
        res = int(getattr(args, "synthetic_res", cli.res))
        rates = getattr(args, "vqvae_scaling_rates", [4])
        nt = getattr(args, "norm_type", "?")
        dnt = getattr(args, "decoder_norm_type", None) or nt
    else:
        args = _default_args(norm_type=cli.norm_type, decoder_norm_type=cli.decoder_norm_type)
        model = build(args)
        res, rates = cli.res, args.vqvae_scaling_rates
        nt, dnt = cli.norm_type, cli.decoder_norm_type
    model.eval()

    print(f"\nencoder norm: {nt} | decoder norm: {dnt} | res {res}^3 | scaling_rates {list(rates)}")

    # ---- 1. static inventory --------------------------------------------------
    coupling, safe = classify(model)
    enc_coupling = [c for c in coupling if c[2] == "encoder"]
    print(f"\n{'=' * 74}\n1) STATIC INVENTORY of normalization layers\n{'=' * 74}")
    print(f"  spatially-coupling norms on the ENCODER path : {len(enc_coupling)}")
    print(f"  spatially-coupling norms on the DECODER path : {len(coupling) - len(enc_coupling)}")
    print(f"  per-voxel (safe) norms, all paths            : {len(safe)}")
    if enc_coupling:
        print("\n  !! these sit on the encoder path and pool over spatial positions:")
        for n, t, _ in enc_coupling[:20]:
            print(f"       {t:<14} {n}")
        if len(enc_coupling) > 20:
            print(f"       ... and {len(enc_coupling) - 20} more")
    else:
        print("\n  OK: no GroupNorm/BatchNorm anywhere on the encoder path.")

    # ---- 2. dynamic reach through the REAL pool_only path ---------------------
    g = cli.grid or max(1, res // int(rates[0]))
    grid = (g, g, g)
    print(f"\n{'=' * 74}\n2) DYNAMIC REACH through model(pool_only=True, patch_grid={list(grid)})\n{'=' * 74}")

    x = torch.zeros(2, 1, res, res, res, requires_grad=True)
    out = model(x, pool_only=True, n_views=2, patch_grid=grid)
    feats = (out[2] if isinstance(out, tuple) else out)[0]  # (n_views*B, C, P)
    P = feats.shape[-1]
    gg = round(P ** (1 / 3))
    corner = 0  # flat index of grid position (0,0,0)
    feats[0, :, corner].sum().backward()
    gr = x.grad.detach().abs()[0, 0].numpy()

    tot = float(gr.sum())
    if tot <= 0:
        # Not a failure. With a zero background the corner's pre-activations are pure conv
        # biases; if a ReLU clamps them the position is COMPLETELY blind, which is a
        # stronger clean result than a small-but-nonzero reach. Only report it as a problem
        # if the inventory found coupling norms, in which case something is inconsistent.
        print(f"  corner feature gradient is identically zero (max |grad| = {float(gr.max()):.3e}).")
        print("  This position is fully blind: no input voxel can change it. That is the")
        print("  cleanest possible outcome, not a failed measurement.")
        print(f"\n{'=' * 74}\nVERDICT\n{'=' * 74}")
        if enc_coupling:
            print("  INCONSISTENT: inventory lists coupling norms on the encoder path, yet the")
            print("  corner is blind. Re-check with a real (non-zero) input volume.")
        else:
            print("  CLEAN (strongest form). No coupling norms on the encoder path, and the corner")
            print("  feature is a constant — it cannot carry any factor, so background R^2 there")
            print("  must come from the probe or the data, not from the encoder.")
        return
    nz = np.argwhere(gr > gr.max() * 1e-3)
    reach = float(nz.max()) + 1
    # 90% gradient mass radius from the corner's input footprint
    idx = np.argwhere(np.ones_like(gr, bool))
    d = np.sqrt((idx**2).sum(1))
    w = gr.reshape(-1)
    o = np.argsort(d)
    r90 = float(d[o][np.searchsorted(np.cumsum(w[o]), 0.9 * tot)])

    brain_r = 0.65 * (res / 2)
    ctr = (res - 1) / 2.0
    d_ctr = np.sqrt(((idx - ctr) ** 2).sum(1)).reshape(gr.shape)
    brain_mass = float(gr[d_ctr <= brain_r].sum() / tot)

    print(f"  probed feature grid                : {gg}^3 ({P} positions)")
    print(f"  1e-3 contour reach from the corner : {reach:.0f} input voxels")
    print(f"  90% of gradient mass within radius : {r90:.0f} input voxels")
    print(f"  nominal brain surface at radius    : {brain_r:.1f} from centre")
    print(f"  FRACTION of the corner's gradient mass landing INSIDE the brain: {brain_mass:.4f}")
    print("\n  ~0      -> the corner cannot see the brain; no spatial coupling survives.")
    print("  >>0     -> information reaches it. If the inventory above is clean, the route is")
    print("             not a normalization layer and the leak is somewhere else entirely.")

    print(f"\n{'=' * 74}\nVERDICT\n{'=' * 74}")
    if not enc_coupling and brain_mass < 1e-4:
        print("  CLEAN. No coupling norms on the encoder path, and a corner feature's gradient")
        print("  does not land on the brain. Any background factor R^2 measured on this model")
        print("  cannot be arriving through the encoder — look at the probe or the data next.")
    elif enc_coupling:
        print("  NOT CLEAN: the encoder still contains spatially-coupling norms (listed above).")
        print("  --norm-type did not reach them. That is the leak, and it is a code path to fix.")
    else:
        print(f"  UNEXPLAINED: inventory is clean, yet {brain_mass:.2%} of the corner's gradient")
        print("  mass lands inside the brain. The route is not a normalization layer. Suspects:")
        print("  a global pool in the probe path, latent masking, or a receptive field that")
        print("  genuinely spans this volume at this resolution.")


if __name__ == "__main__":
    main()
