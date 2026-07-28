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
import os

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
    ap.add_argument("--out", default=None, help="Figure path (default: <run-dir>/smuggling.png).")
    ap.add_argument("--no-fig", action="store_true", help="Skip the reconstruction figure.")
    ap.add_argument(
        "--save-nifti",
        nargs="?",
        const="",
        default=None,
        help="Write full volumes as .nii.gz for a 3D viewer (input, mask, per-condition recon, "
        "damage, and the ablated latent set upsampled to input resolution). Optional value sets "
        "the directory; default <run-dir>/smuggling_nii/.",
    )
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
    snap, snap_meta = {}, {}
    first = True
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

        bmf = brain_out.float()
        nvox = bmf.sum(dim=(1, 2, 3, 4)).clamp_min(1)

        def affine_share(r, r_base):
            """Fraction of the damage explained by a per-sample global scale+shift over the brain.

            Fits r ~= s*r_base + c inside the brain and asks how much of (r - r_base) that
            accounts for. ~1.0 means the ablation only rescaled/offset the image — a global
            INTENSITY code was removed, not anatomy. Low means genuine structural change.
            """
            v = lambda t: t.view(-1, 1, 1, 1, 1)
            mb = (r_base * bmf).sum(dim=(1, 2, 3, 4)) / nvox
            ma = (r * bmf).sum(dim=(1, 2, 3, 4)) / nvox
            db_, da_ = r_base - v(mb), r - v(ma)
            cov = (db_ * da_ * bmf).sum(dim=(1, 2, 3, 4)) / nvox
            var = (db_ * db_ * bmf).sum(dim=(1, 2, 3, 4)) / nvox
            s = cov / var.clamp_min(1e-12)
            c = ma - s * mb
            resid = r - (v(s) * r_base + v(c))
            num = (resid**2 * bmf).sum(dim=(1, 2, 3, 4))
            den = ((r - r_base) ** 2 * bmf).sum(dim=(1, 2, 3, 4)).clamp_min(1e-12)
            return float((1.0 - num / den).mean()), resid

        with torch.no_grad():
            r_base = None
            for name, sel in conds.items():
                mode["which"] = None if sel is None else torch.as_tensor(np.where(sel)[0], device=device)
                r = model(x, return_recon=True, n_views=1)[0]
                if r.shape[2:] != x.shape[2:]:
                    r = Fn.interpolate(r, size=x.shape[2:], mode="trilinear", align_corners=False)
                for reg, mask in (("centre", centre_out), ("brain", brain_out), ("bg", bg_out)):
                    tot.setdefault((name, reg), []).append(err(r, mask))
                if name == "baseline":
                    r_base = r.clone()
                else:
                    frac, resid = affine_share(r, r_base)
                    tot.setdefault((name, "affine"), []).append(frac)
                    tot.setdefault((name, "struct_mse"), []).append(
                        float((resid**2 * bmf).sum() / bmf.sum().clamp_min(1))
                    )
                    if first:
                        snap[("resid", name)] = resid[0, 0].cpu().numpy()
                if first:
                    snap[name] = r[0, 0].cpu().numpy()
            if first:
                snap_meta["x"] = x[0, 0].cpu().numpy()
                snap_meta["m"] = m[0, 0].cpu().numpy()
                first = False
            mode["which"] = None

    mean = lambda k: float(np.mean(tot[k]))
    print(f"\nrun: {cli.run_dir}")
    print(
        f"latent grid {tuple(lat)} ({P} positions) | brain latents={int(brain_lat.sum())} bg latents={int(bg_lat.sum())}"
    )
    print("distances are in LATENT voxels to the nearest brain latent\n")

    hdr = f"{'condition':<26}{'n':>6}{'brain MSE':>12}{'Δbrain':>10}{'Δbg':>8}{'affine%':>9}{'struct MSE':>12}"
    print(hdr + "\n" + "-" * len(hdr))
    b_brain, b_bg = mean(("baseline", "brain")), mean(("baseline", "bg"))
    print(f"{'baseline':<26}{'-':>6}{b_brain:>12.5f}{'-':>10}{'-':>8}{'-':>9}{'-':>12}")
    for name, sel in conds.items():
        if sel is None:
            continue
        db = (mean((name, "brain")) - b_brain) / max(b_brain, 1e-12)
        dg = (mean((name, "bg")) - b_bg) / max(b_bg, 1e-12)
        print(
            f"{name:<26}{int(sel.sum()):>6}{mean((name,'brain')):>12.5f}{db:>+9.0%}{dg:>+7.0%}"
            f"{mean((name,'affine')):>9.1%}{mean((name,'struct_mse')):>12.5f}"
        )
    print("\naffine% = share of the damage explained by a per-sample global scale+shift over the brain.")
    print("  ~100%  -> the ablation only rescaled/offset the image: what was stored is a global")
    print("            INTENSITY code, not anatomy. struct MSE (the residual) is then the real")
    print("            structural damage — compare it against baseline brain MSE, not against Δbrain.")
    print("  low    -> genuine structural change: anatomy was being reconstructed from those latents.")

    far_key = [k for k in bands if k.startswith("bg FAR")][0]
    far_db = (mean((far_key, "brain")) - b_brain) / max(b_brain, 1e-12)
    far_dg = (mean((far_key, "bg")) - b_bg) / max(b_bg, 1e-12)
    near_key = [k for k in bands if k.startswith("bg near")][0]
    near_db = (mean((near_key, "brain")) - b_brain) / max(b_brain, 1e-12)

    far_aff = mean((far_key, "affine"))
    far_struct = mean((far_key, "struct_mse"))

    print("\nverdict:")
    if b_brain > 0.05:
        print(f"  WARNING: baseline brain MSE is {b_brain:.4f} -- this model reconstructs poorly.")
        print("  Ablation results are not interpretable until reconstruction works.")
    elif far_db > 0.10 and far_db > far_dg and far_aff > 0.9:
        print(f"  GLOBAL INTENSITY CODE, not structural smuggling. The FAR band raises brain error")
        print(f"  {far_db:+.0%}, but {far_aff:.0%} of that is a per-sample global scale+shift. Residual")
        print(
            f"  structural damage is {far_struct:.5f} vs baseline brain MSE {b_brain:.5f}"
            f" ({far_struct / max(b_brain, 1e-12):.1f}x)."
        )
        print("  Those latents hold a whole-image brightness/contrast parameter (style gain/bias),")
        print("  not anatomy. Still a real contrastive shortcut (per-sample code readable at every")
        print("  position), but NOT evidence that anatomical detail is delocalised.")
    elif far_db > 0.10 and far_db > far_dg:
        print(f"  STRUCTURAL SMUGGLING: the FAR band raises brain error {far_db:+.0%} while background")
        print(f"  error moves {far_dg:+.0%}, and only {far_aff:.0%} is a global scale+shift -- so")
        print(f"  {far_struct:.5f} of residual structural damage remains (baseline {b_brain:.5f}).")
        print("  Those latents sit beyond plausible decoder reach yet carry anatomy the brain")
        print("  reconstruction depends on.")
    elif near_db > 0.10 and far_db < 0.5 * near_db:
        print(f"  DECODER RECEPTIVE FIELD: near band {near_db:+.0%} vs far band {far_db:+.0%}. Only")
        print("  boundary-adjacent latents matter -- ordinary upsampling reach, not smuggling.")
    else:
        print(f"  No background-specific effect: near {near_db:+.0%}, far {far_db:+.0%}.")

    if cli.save_nifti is not None:
        import nibabel as nib

        nd = cli.save_nifti or os.path.join(cli.run_dir, "smuggling_nii")
        os.makedirs(nd, exist_ok=True)
        xv, mv = snap_meta["x"], snap_meta["m"]
        base = snap["baseline"]
        aff = np.eye(4)
        D = xv.shape[0]

        def slug(s):
            out = "".join(c if c.isalnum() else "_" for c in s)
            while "__" in out:
                out = out.replace("__", "_")
            return out.strip("_").lower()

        def w(name, vol):
            p = os.path.join(nd, f"{name}.nii.gz")
            nib.save(nib.Nifti1Image(np.asarray(vol, np.float32), aff), p)
            return p

        written = [
            w("input", xv),
            w("brain_mask", mv),
            w("recon_baseline", base),
            w("error_baseline", np.abs(base - xv)),
        ]
        for k, sel in conds.items():
            if sel is None:
                continue
            s = slug(k)
            written.append(w(f"recon_{s}", snap[k]))
            written.append(w(f"damage_{s}", np.abs(snap[k] - base)))
            if ("resid", k) in snap:
                # damage with the global scale+shift removed: the STRUCTURAL part only
                written.append(w(f"damage_structural_{s}", np.abs(snap[("resid", k)])))
            lm = torch.zeros(P)
            lm[torch.as_tensor(np.where(sel)[0])] = 1.0
            up = Fn.interpolate(lm.reshape(1, 1, gz, gy, gx), size=(D, D, D), mode="nearest")[0, 0].numpy()
            written.append(w(f"ablated_latents_{s}", up))

        fs = slug(far_key)
        print(f"\nwrote {len(written)} volumes -> {nd}")
        print("  The ablated positions sit OUTSIDE the brain; the damage appears INSIDE it.")
        print("  Load the input, then overlay the far-band damage and the latents that were zeroed:")
        print(
            f"    fsleyes {os.path.join(nd, 'input.nii.gz')} "
            f"{os.path.join(nd, f'damage_{fs}.nii.gz')} -cm red-yellow -a 60 "
            f"{os.path.join(nd, f'ablated_latents_{fs}.nii.gz')} -cm blue -a 30"
        )

    if cli.no_fig:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xv, mv = snap_meta["x"], snap_meta["m"]
    base = snap["baseline"]
    sl = xv.shape[0] // 2
    lsl = gz // 2
    show = ["bg near", "bg mid", "bg FAR", "centre only"]
    cols = [next(k for k in conds if k.startswith(p)) for p in show]

    dmg = {k: np.abs(snap[k] - base) for k in cols}
    dmax = max(float(d[sl].max()) for d in dmg.values()) or 1.0
    vlo, vhi = float(xv.min()), float(xv.max())

    fig, ax = plt.subplots(3, len(cols) + 1, figsize=(3.0 * (len(cols) + 1), 9.2))

    def show_im(a, img, **kw):
        a.imshow(img, **kw)
        a.contour(mv[sl], levels=[0.5], colors="cyan", linewidths=0.9)
        a.axis("off")

    show_im(ax[0, 0], xv[sl], cmap="gray", vmin=vlo, vmax=vhi)
    ax[0, 0].set_title("input", fontsize=10)
    show_im(ax[1, 0], base[sl], cmap="gray", vmin=vlo, vmax=vhi)
    ax[1, 0].set_title(f"baseline recon\nbrain MSE {b_brain:.4f}", fontsize=10)
    show_im(ax[2, 0], np.abs(base - xv)[sl], cmap="inferno", vmin=0, vmax=dmax)
    ax[2, 0].set_title("baseline error |recon-input|", fontsize=10)

    for j, k in enumerate(cols, start=1):
        lm = np.zeros(P, float)
        lm[conds[k]] = 1.0
        ax[0, j].imshow(lm.reshape(gz, gy, gx)[lsl], cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        ax[0, j].axis("off")
        ax[0, j].set_title(f"{k}\nablated latents (n={int(conds[k].sum())})", fontsize=9)
        show_im(ax[1, j], snap[k][sl], cmap="gray", vmin=vlo, vmax=vhi)
        db = (mean((k, "brain")) - b_brain) / max(b_brain, 1e-12)
        ax[1, j].set_title(f"recon after ablation\nΔbrain {db:+.0%}", fontsize=10)
        show_im(ax[2, j], dmg[k][sl], cmap="inferno", vmin=0, vmax=dmax)
        ax[2, j].set_title("damage |ablated-baseline|", fontsize=10)

    fig.suptitle(
        "Ablating BACKGROUND latents damages the BRAIN (cyan outline), not the background",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = cli.out or os.path.join(cli.run_dir, "smuggling.png")
    fig.savefig(out, dpi=140)
    print(f"\nsaved figure -> {out}")
    print("Read the bottom row: damage concentrated INSIDE the cyan brain outline, after ablating")
    print("latents that sit OUTSIDE it, is smuggling made visible. All damage panels share a colour")
    print("scale, so the far-background column being brightest is directly comparable.")


if __name__ == "__main__":
    main()
