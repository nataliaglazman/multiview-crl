"""Do BACKGROUND latent positions carry information the decoder needs for the CENTRE?

"Codebook smuggling" hypothesis: empty background is trivially cheap to reconstruct, so
its latent positions have spare capacity. If the decoder's receptive field lets those
positions influence central output voxels, the encoder can profitably stash central
information there -- which would show up as background positions predicting central
factors, with no normalization artefact involved.

This is a DECODER-SIDE INCENTIVE, structurally different from an encoder-side artefact
(GroupNorm broadcast, receptive-field reach), and it makes a different prediction: it
should survive a change of normalization, because it is driven by reconstruction utility
rather than by how activations are normalised.

Direct test (forward hook, no retraining):
  1. normal forward                                  -> recon_base
  2. forward with BACKGROUND latent positions replaced by their across-batch mean
     (removes sample-specific information there, keeps the typical value) -> recon_ablate
  3. compare reconstruction error inside the CENTRE region

  centre error rises  -> background latents were carrying centre information: SMUGGLING
  centre error flat   -> background latents contribute nothing to the centre: no smuggling

Controls: ablating the CENTRE and measuring BACKGROUND error (should barely move), and
ablating a random position set of the same size (isolates "any ablation hurts").

Usage:
  python -m eval.smuggling_test --run-dir results/synthetic/<run> --num-samples 128
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

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
        sel = mode["which"]
        o = out.clone()
        flat = o.reshape(o.shape[0], o.shape[1], -1)
        flat[:, :, sel] = flat[:, :, sel].mean(dim=0, keepdim=True)
        return flat.reshape(o.shape)

    enc.register_forward_hook(hook)

    tot = {}
    n_batches = 0
    for batch in loader:
        imgs = batch["image"]
        x = imgs[0].to(device)
        m = batch["mask"][0].to(device).float()
        with torch.no_grad():
            mode["which"] = None
            out = model(x, return_recon=True, n_views=1)
            rec0 = out[0]
            if rec0.shape[2:] != x.shape[2:]:
                rec0 = torch.nn.functional.interpolate(rec0, size=x.shape[2:], mode="trilinear", align_corners=False)
            _, _, D, H, W = rec0.shape
            lat = inner.encoders[cli.level](x).shape[2:]
            gz, gy, gx = lat
            zz, yy, xx = torch.meshgrid(torch.arange(gz), torch.arange(gy), torch.arange(gx), indexing="ij")
            scale = D / gz
            cz = (gz - 1) / 2.0
            dl = torch.sqrt((zz - cz) ** 2 + (yy - (gy - 1) / 2.0) ** 2 + (xx - (gx - 1) / 2.0) ** 2).reshape(-1)
            cov = torch.nn.functional.adaptive_avg_pool3d(m, lat).reshape(m.shape[0], -1).mean(0).cpu()
            bg_lat = (cov <= 0.1).numpy()
            ctr_lat = (dl <= cli.centre_radius / scale).numpy()
            g = torch.Generator().manual_seed(0)
            rnd_lat = np.zeros_like(bg_lat)
            rnd_lat[torch.randperm(len(bg_lat), generator=g)[: int(bg_lat.sum())].numpy()] = True

            # output-space regions
            ZZ, YY, XX = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W), indexing="ij")
            dout = torch.sqrt((ZZ - (D - 1) / 2.0) ** 2 + (YY - (H - 1) / 2.0) ** 2 + (XX - (W - 1) / 2.0) ** 2).to(
                device
            )
            centre_out = (dout <= cli.centre_radius).unsqueeze(0).unsqueeze(0)
            brain_out = m > 0.5
            bg_out = ~brain_out

            def err(r, region):
                d = (r - x) ** 2
                reg = region.expand_as(d)
                return float((d * reg).sum() / reg.sum().clamp_min(1))

            base = {"centre": err(rec0, centre_out), "brain": err(rec0, brain_out), "bg": err(rec0, bg_out)}

            for name, sel in (("ablate_bg", bg_lat), ("ablate_centre", ctr_lat), ("ablate_random", rnd_lat)):
                mode["which"] = torch.as_tensor(np.where(sel)[0], device=device)
                o = model(x, return_recon=True, n_views=1)[0]
                if o.shape[2:] != x.shape[2:]:
                    o = torch.nn.functional.interpolate(o, size=x.shape[2:], mode="trilinear", align_corners=False)
                for reg, mask in (("centre", centre_out), ("brain", brain_out), ("bg", bg_out)):
                    tot.setdefault((name, reg), []).append(err(o, mask))
                tot.setdefault(("n_pos", name), []).append(int(sel.sum()))
            mode["which"] = None
            for reg in ("centre", "brain", "bg"):
                tot.setdefault(("baseline", reg), []).append(base[reg])
        n_batches += 1

    def mean(k):
        return float(np.mean(tot[k])) if k in tot else float("nan")

    P = int(np.prod(lat))
    print(f"\nrun: {cli.run_dir}")
    print(f"latent grid {tuple(lat)} ({P} positions) | centre radius {cli.centre_radius:g} input voxels")
    print(
        f"latent positions ablated: bg={int(np.mean(tot[('n_pos','ablate_bg')]))}  "
        f"centre={int(np.mean(tot[('n_pos','ablate_centre')]))}  random={int(np.mean(tot[('n_pos','ablate_random')]))}\n"
    )

    hdr = f"{'condition':<18}{'centre MSE':>12}{'brain MSE':>12}{'background MSE':>16}"
    print(hdr + "\n" + "-" * len(hdr))
    for name in ("baseline", "ablate_bg", "ablate_centre", "ablate_random"):
        print(f"{name:<18}{mean((name,'centre')):>12.5f}{mean((name,'brain')):>12.5f}{mean((name,'bg')):>16.5f}")

    b, ab, ar = mean(("baseline", "centre")), mean(("ablate_bg", "centre")), mean(("ablate_random", "centre"))
    rise_bg = (ab - b) / max(b, 1e-12)
    rise_rnd = (ar - b) / max(b, 1e-12)
    print("\nverdict:")
    print(f"  ablating BACKGROUND latents changes CENTRE reconstruction error by {rise_bg:+.1%}")
    print(f"  ablating RANDOM   latents changes CENTRE reconstruction error by {rise_rnd:+.1%}  (control)")
    if rise_bg > 0.10 and rise_bg > 1.5 * max(rise_rnd, 0):
        print("  -> SMUGGLING: background latents carry information the decoder uses for the centre.")
    elif rise_bg > 0.10:
        print("  -> centre degrades, but the random control degrades comparably: this is generic")
        print("     ablation sensitivity, not background-specific smuggling.")
    else:
        print("  -> NO smuggling: background latents contribute nothing to central reconstruction.")


if __name__ == "__main__":
    main()
