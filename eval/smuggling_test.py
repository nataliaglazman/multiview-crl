"""Do BACKGROUND latent positions carry information the decoder needs for the BRAIN?

"Codebook smuggling": empty background is trivially cheap to reconstruct, so its latent
positions have spare capacity. If the encoder stashes brain information there, ablating
those positions should damage BRAIN reconstruction.

Confound to separate: decoder receptive field. Latents just OUTSIDE the brain legitimately
help reconstruct the brain BOUNDARY -- ordinary upsampling, not smuggling. So background
ablation is stratified by distance to the brain:

    near band only matters   -> decoder receptive field (expected, benign)
    far band matters too     -> genuine smuggling (information placed where it is not needed)

Readouts per condition (latents replaced by their across-batch mean, removing
sample-specific information while keeping the typical value):

    brain MSE   the thing smuggled information would help
    bg MSE      if these latents were FOR the background, ablating them should hurt HERE

The decisive signature is the ASYMMETRY: background ablation that raises brain error while
leaving (or improving) background error means those positions were not serving the
background at all.

NOTE: a "random positions" control is NOT size-matchable here -- 3066 random positions out
of 4096 include most of the brain, so it is a strictly harsher ablation and cannot be
compared against the background condition. It is reported for reference only.

Usage:
  python -m eval.smuggling_test --run-dir results/synthetic/<run> --num-samples 128
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as Fn

logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--centre-radius", type=float, default=6.0, help="Centre region radius, input voxels.")
    ap.add_argument("--device", default=None)
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from torch.utils.data import DataLoader

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, args, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint, device)
    model.eval()
    inner = model.module if hasattr(model, "module") else model
    inner = inner.online if hasattr(inner, "online") else inner
    enc = inner.encoders[cli.level]

    ds = build_synthetic_test_set(args, cli.num_samples)
    loader = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=0)

    mode = {"which": None}

    def hook(mod, inp, out):
        if mode["which"] is None:
            return None
        o = out.clone()
        flat = o.reshape(o.shape[0], o.shape[1], -1)
        flat[:, :, mode["which"]] = flat[:, :, mode["which"]].mean(dim=0, keepdim=True)
        return flat.reshape(o.shape)

    enc.register_forward_hook(hook)

    # ---- geometry from the first batch -------------------------------------------
    b0 = next(iter(loader))
    x0 = b0["image"][0].to(device)
    m0 = b0["mask"][0].to(device).float()
    with torch.no_grad():
        lat = enc(x0).shape[2:]
    gz, gy, gx = lat
    P = gz * gy * gx
    cov = Fn.adaptive_avg_pool3d(m0, lat).reshape(m0.shape[0], -1).mean(0).cpu().numpy()
    brain_lat, bg_lat = cov >= 0.5, cov <= 0.1

    coords = np.argwhere(np.ones(tuple(lat), bool))
    bpos = coords[brain_lat]
    if len(bpos) == 0:
        raise SystemExit("no brain latent positions found")
    d2 = ((coords[:, None, :] - bpos[None, :, :]) ** 2).sum(-1)
    dist = np.sqrt(d2.min(1))  # latent-space distance to nearest brain position

    bd = dist[bg_lat]
    edges = np.quantile(bd, [0.0, 1 / 3, 2 / 3, 1.0])
    bands = {
        f"bg near [{edges[0]:.1f},{edges[1]:.1f})": bg_lat & (dist < edges[1]),
        f"bg mid  [{edges[1]:.1f},{edges[2]:.1f})": bg_lat & (dist >= edges[1]) & (dist < edges[2]),
        f"bg FAR  [{edges[2]:.1f},{edges[3]:.1f}]": bg_lat & (dist >= edges[2]),
    }
    cz, cy, cx = (gz - 1) / 2.0, (gy - 1) / 2.0, (gx - 1) / 2.0
    dctr = np.sqrt(((coords - np.array([cz, cy, cx])) ** 2).sum(-1))
    rng = np.random.default_rng(0)
    rnd = np.zeros(P, bool)
    rnd[rng.permutation(P)[: int(bg_lat.sum())]] = True

    conds = {
        "baseline": None,
        "bg ALL": bg_lat,
        **bands,
        "centre only": dctr <= cli.centre_radius / (x0.shape[-1] / gx),
        "random (unmatched)": rnd,
    }

    tot = {}
    for batch in loader:
        x = batch["image"][0].to(device)
        m = batch["mask"][0].to(device).float()
        D = x.shape[-1]
        ZZ, YY, XX = torch.meshgrid(*[torch.arange(D)] * 3, indexing="ij")
        dout = torch.sqrt(((ZZ - (D - 1) / 2.0) ** 2 + (YY - (D - 1) / 2.0) ** 2 + (XX - (D - 1) / 2.0) ** 2)).to(
            device
        )
        centre_out = (dout <= cli.centre_radius).unsqueeze(0).unsqueeze(0)
        brain_out, bg_out = m > 0.5, ~(m > 0.5)

        def err(r, region):
            reg = region.expand_as(r)
            return float((((r - x) ** 2) * reg).sum() / reg.sum().clamp_min(1))

        with torch.no_grad():
            for name, sel in conds.items():
                mode["which"] = None if sel is None else torch.as_tensor(np.where(sel)[0], device=device)
                r = model(x, return_recon=True, n_views=1)[0]
                if r.shape[2:] != x.shape[2:]:
                    r = Fn.interpolate(r, size=x.shape[2:], mode="trilinear", align_corners=False)
                for reg, mask in (("centre", centre_out), ("brain", brain_out), ("bg", bg_out)):
                    tot.setdefault((name, reg), []).append(err(r, mask))
            mode["which"] = None

    mean = lambda k: float(np.mean(tot[k]))
    print(f"\nrun: {cli.run_dir}")
    print(
        f"latent grid {tuple(lat)} ({P} positions) | brain latents={int(brain_lat.sum())} bg latents={int(bg_lat.sum())}"
    )
    print("distances are in LATENT voxels to the nearest brain latent\n")

    hdr = f"{'condition':<26}{'n':>6}{'brain MSE':>12}{'Δbrain':>10}{'bg MSE':>11}{'Δbg':>9}"
    print(hdr + "\n" + "-" * len(hdr))
    b_brain, b_bg = mean(("baseline", "brain")), mean(("baseline", "bg"))
    print(f"{'baseline':<26}{'-':>6}{b_brain:>12.5f}{'-':>10}{b_bg:>11.5f}{'-':>9}")
    for name, sel in conds.items():
        if sel is None:
            continue
        db = (mean((name, "brain")) - b_brain) / max(b_brain, 1e-12)
        dg = (mean((name, "bg")) - b_bg) / max(b_bg, 1e-12)
        print(
            f"{name:<26}{int(sel.sum()):>6}{mean((name,'brain')):>12.5f}{db:>+9.0%}{mean((name,'bg')):>11.5f}{dg:>+8.0%}"
        )

    far_key = [k for k in bands if k.startswith("bg FAR")][0]
    far_db = (mean((far_key, "brain")) - b_brain) / max(b_brain, 1e-12)
    far_dg = (mean((far_key, "bg")) - b_bg) / max(b_bg, 1e-12)
    near_key = [k for k in bands if k.startswith("bg near")][0]
    near_db = (mean((near_key, "brain")) - b_brain) / max(b_brain, 1e-12)

    print("\nverdict:")
    if b_brain > 0.05:
        print(f"  WARNING: baseline brain MSE is {b_brain:.4f} -- this model reconstructs poorly.")
        print("  Ablation results are not interpretable until reconstruction works.")
    elif far_db > 0.10 and far_db > far_dg:
        print(f"  SMUGGLING: ablating the FAR background band raises brain error {far_db:+.0%} while")
        print(f"  background error moves {far_dg:+.0%}. Those latents sit beyond any plausible decoder")
        print("  reach yet carry information the brain reconstruction depends on.")
    elif near_db > 0.10 and far_db < 0.5 * near_db:
        print(f"  DECODER RECEPTIVE FIELD: near band {near_db:+.0%} vs far band {far_db:+.0%}. Only")
        print("  boundary-adjacent latents matter -- ordinary upsampling reach, not smuggling.")
    else:
        print(f"  No background-specific effect: near {near_db:+.0%}, far {far_db:+.0%}.")


if __name__ == "__main__":
    main()
