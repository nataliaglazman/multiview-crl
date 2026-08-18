"""Post-hoc training-dynamics forensics from checkpoints alone — no forward pass, no data, no GPU.

Motivation: a step change in training dynamics (recon jumping to a higher plateau, metrics
freezing) can take many hours to reach, so re-running with extra logging is expensive. Almost
everything needed to characterise such a jump is ALREADY inside every checkpoint:

  * ``optimizer_state_dict`` holds Adam's ``exp_avg_sq`` — an EMA of SQUARED GRADIENTS over a
    ~1/(1-beta2) step window. Crucially, ``clip_grad_norm_`` modifies ``.grad`` IN PLACE before
    ``optimizer.step()``, so Adam only ever sees POST-clip gradients. With ``max_norm=2.0``
    (main_multimodal.py) that means:

        sqrt(sum_i exp_avg_sq_i)  ~=  RMS post-clip gradient norm  <=  2.0

    A value pinned near 2.0 means the clip was BINDING over that window — every parameter in
    the model was being rescaled by min(1, 2/||g||), i.e. the optimizer was throttled. A value
    that falls away from 2.0 means the clip stopped binding and the EFFECTIVE learning rate for
    the whole model rose, with nothing in any loss curve to show for it.

  * The codebook's ``cluster_size`` / ``embed`` are registered buffers, so codebook utilization
    and perplexity are recoverable per checkpoint.

  * Norm ``weight`` (gamma) tensors show whether feature scale was inflated, e.g. by a variance
    hinge driving std toward its target.

Usage:
    python -m eval.checkpoint_forensics /path/to/run_dir
    python -m eval.checkpoint_forensics ckpt_1000.pt ckpt_9000.pt ckpt_12000.pt
    python -m eval.checkpoint_forensics /path/to/run_dir --csv out.csv
"""

import argparse
import glob
import math
import os
import re

import torch

CLIP_MAX_NORM = 2.0  # must match clip_grad_norm_ in training/main_multimodal.py


def _step_of(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _find_checkpoints(targets):
    out = []
    for t in targets:
        if os.path.isdir(t):
            for pat in ("*.pt", "*.pth"):
                out.extend(glob.glob(os.path.join(t, "**", pat), recursive=True))
        else:
            out.append(t)
    # De-duplicate, keep the ones that look step-tagged first, then sort by step.
    return sorted(set(out), key=lambda p: (_step_of(p), p))


def _grad_stats(ckpt):
    """Recover the post-clip gradient norm from Adam's second moment."""
    opt = ckpt.get("optimizer_state_dict")
    if not opt or "state" not in opt:
        return None
    state = opt["state"]
    if not state:
        return None
    sq_total, m_total, n_tensors, steps = 0.0, 0.0, 0, []
    for _, st in state.items():
        v = st.get("exp_avg_sq")
        m = st.get("exp_avg")
        if v is None:
            continue
        sq_total += float(v.double().sum())
        if m is not None:
            m_total += float(m.double().pow(2).sum())
        n_tensors += 1
        s = st.get("step")
        if s is not None:
            steps.append(float(s.item() if torch.is_tensor(s) else s))
    if n_tensors == 0:
        return None
    t = max(steps) if steps else 0.0
    # Adam stores v un-bias-corrected; correct it so early checkpoints are comparable.
    b2_corr = 1.0 - 0.999**t if t > 0 else 1.0
    rms_norm = math.sqrt(sq_total / max(b2_corr, 1e-12))
    return {
        "adam_step": t,
        "grad_norm_rms": rms_norm,
        "grad_norm_mean": math.sqrt(m_total) if m_total else float("nan"),
        "clip_binding": rms_norm / CLIP_MAX_NORM,
        "n_param_tensors": n_tensors,
    }


def _encoder_sd(ckpt):
    sd = ckpt.get("encoders", ckpt)
    clean = {}
    for k, v in sd.items():
        for p in ("online.", "module."):
            if k.startswith(p):
                k = k[len(p) :]
        clean[k] = v
    return clean


def _codebook_stats(sd):
    """Utilization + perplexity per codebook, from the cluster_size buffer."""
    out = {}
    for k, v in sd.items():
        if not k.endswith("cluster_size"):
            continue
        cs = v.double()
        tot = cs.sum()
        if tot <= 0:
            continue
        p = cs / tot
        nz = p[p > 0]
        ppl = float(torch.exp(-(nz * nz.log()).sum()))
        lvl = k.replace(".cluster_size", "")
        out[lvl] = {
            "entries": cs.numel(),
            "active": int((cs > 1.0).sum()),
            "perplexity": ppl,
            "ppl_ratio": ppl / cs.numel(),
        }
    return out


def _norm_gamma_stats(sd):
    """Scale of the normalization gammas — catches feature-scale inflation."""
    gammas = {}
    for k, v in sd.items():
        # LayerNorm/GroupNorm weight tensors are 1-D and named '...norm...weight'
        if v.ndim == 1 and k.endswith(".weight") and ("norm" in k.lower() or "Norm" in k):
            gammas[k] = v.float()
    if not gammas:
        return None
    allv = torch.cat([g.flatten() for g in gammas.values()])
    # The content/style split norms are the ones the contrastive terms act through.
    split = {k: g for k, g in gammas.items() if "content_norms" in k}
    res = {
        "n_norm_tensors": len(gammas),
        "gamma_mean": float(allv.mean()),
        "gamma_max": float(allv.abs().max()),
        "gamma_p99": float(allv.abs().quantile(0.99)),
    }
    # Content and style blocks separately: the BT variance hinge acts on the CONTENT
    # channels only (losses.py selects content_indices before relu(1-std)), and it is the
    # only term in the objective that is sensitive to PER-CHANNEL scale — on_diag/off_diag
    # standardise each channel, and sim_loss has a detached per-channel denominator under
    # bt_sim_normalize. So if the gamma heterogeneity is concentrated in the content block
    # and absent from style, the hinge is the only thing in the loss that could have put it
    # there. If both blocks are equally spread, something scale-blind is responsible and the
    # hinge is exonerated.
    for _tag, _sub in (
        ("content", {k: g for k, g in gammas.items() if "norm_content" in k}),
        ("style", {k: g for k, g in gammas.items() if "norm_style" in k}),
    ):
        if not _sub:
            continue
        _v = torch.cat([g.flatten() for g in _sub.values()])
        _g2 = _v.double().pow(2)
        res[f"{_tag}_gamma_spread"] = float(_v.abs().max() / _v.abs().clamp_min(1e-9).min())
        res[f"{_tag}_eff_dim"] = float(_g2.sum().pow(2) / _g2.pow(2).sum())
        res[f"{_tag}_n"] = int(_v.numel())

    if split:
        sv = torch.cat([g.flatten() for g in split.values()])
        res["split_gamma_mean"] = float(sv.mean())
        res["split_gamma_max"] = float(sv.abs().max())
        # Heterogeneity across channels is what survives a channel-LayerNorm downstream.
        res["split_gamma_spread"] = float(sv.abs().max() / sv.abs().clamp_min(1e-9).min())
        # EFFECTIVE CODEBOOK DIMENSION. The quantizer assigns by squared Euclidean distance
        # (models/vqvae.py: flatten^2 - 2*flatten@embed + embed^2), so a channel's influence on
        # which code is chosen scales as gamma_c^2. The participation ratio of gamma^2,
        # (sum g^2)^2 / sum g^4, is the number of channels that actually drive that decision:
        # it equals C when all gammas are equal and collapses toward 1 as they spread. Channels
        # far below it are effectively NOT quantized — the codebook spends no resolution on
        # them, so whatever content they carry does not reach the decoder. That failure is
        # invisible in the commitment loss, which is dominated by the high-gamma channels the
        # codebook does fit well.
        g2 = sv.double().pow(2)
        res["split_eff_dim"] = float(g2.sum().pow(2) / g2.pow(2).sum())
        res["split_n_channels"] = int(sv.numel())
        res["split_eff_dim_frac"] = res["split_eff_dim"] / max(1, sv.numel())
    return res


# ---------------------------------------------------------------------------
# Tier 2: reconstruction decomposition, PER VIEW. Needs torch + the run's
# settings.json + the synthetic generator; one batch per checkpoint.
# ---------------------------------------------------------------------------
def _recon_stats(run_dir, ckpt_path, device, n_samples, batch_size, causal):
    """Per-view pixel error, saturation and commitment cost from one forward pass.

    Split out per view because the two encoders are separate under --separate-encoders
    and keep their own style codebooks (vqvae.py: view 0 -> style_codebooks, view 1 ->
    style_codebooks_v1). A capacity change confined to one of them shows up here as a
    per-view asymmetry in pixel error and NOWHERE in the aggregate Loss/Recon curve.

    Mirrors BaselineLoss._calculate_pixel_loss: masked L1 averaged over brain voxels
    only, so the numbers are comparable to Recon/Loss-MAE-Reconstruction.
    """
    import torch as _t
    from torch.utils.data import DataLoader

    from eval.phase0_extract import load_args, load_model
    from eval.run_dci_synthetic import build_synthetic_test_set

    args = load_args(run_dir, None)
    model = load_model(args, ckpt_path, device)
    model.eval()

    ds = build_synthetic_test_set(args, num_samples=n_samples, cache=True, causal=causal)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    _pg = (
        tuple(args.patch_grid)
        if getattr(args, "patch_contrastive", False) and getattr(args, "patch_grid", None)
        else None
    )
    acc = {}
    with _t.no_grad():
        for data in loader:
            imgs = data["image"]
            n_views = len(imgs)
            x = _t.cat(imgs, 0).to(device)
            m = data.get("mask")
            m = _t.cat(m, 0).to(device).float() if m is not None else None
            out = model(
                x,
                return_recon=True,
                pool_only=True,
                n_views=n_views,
                subsets=getattr(args, "subsets", [(0, 1)]),
                patch_grid=_pg,
                mask=m,
            )
            recon, diffs = out[0], out[1]
            if recon is None:
                raise SystemExit("Model returned no reconstruction (contrastive_only run?).")
            if recon.shape[2:] != x.shape[2:]:
                recon = _t.nn.functional.interpolate(recon, size=x.shape[2:], mode="trilinear", align_corners=False)
            B = x.shape[0] // n_views
            for v in range(n_views):
                xv, yv = x[v * B : (v + 1) * B].float(), recon[v * B : (v + 1) * B].float()
                mv = m[v * B : (v + 1) * B] if m is not None else None
                sat = (yv.abs() > 1.0).float().mean().item()
                raw_std = yv.std().item()
                if mv is not None:
                    mae = ((xv - yv).abs() * mv).sum().item() / mv.sum().clamp_min(1.0).item()
                else:
                    mae = (xv - yv).abs().mean().item()
                d = acc.setdefault(v, {"mae": 0.0, "sat": 0.0, "std": 0.0, "n": 0})
                d["mae"] += mae
                d["sat"] += sat
                d["std"] += raw_std
                d["n"] += 1
            # ── View-difference fidelity ────────────────────────────────────────
            # The two views are the SAME subjects, so (x_v0 - x_v1) is exactly the
            # view-specific signal the model is supposed to reproduce, and
            # (y_v0 - y_v1) is how much of it actually survives to the output.
            #
            # This is the number that says whether a low per-view MAE means the model
            # RECONSTRUCTS each view, or merely that the two targets are so similar that
            # emitting one image scores well on both. A ratio near 0 with a good MAE is
            # the latter, and such a run is not a control for anything about view
            # identity. The correlation is the stronger of the two: magnitude can be
            # right while the difference is put in the wrong places.
            if n_views == 2:
                x0, x1 = x[:B].float(), x[B : 2 * B].float()
                y0, y1 = recon[:B].float(), recon[B : 2 * B].float()
                dx, dy = x0 - x1, y0 - y1
                if m is not None:
                    mm = (m[:B] * m[B : 2 * B]).float()
                    den = mm.sum().clamp_min(1.0)
                    t_l1 = float((dx.abs() * mm).sum() / den)
                    r_l1 = float((dy.abs() * mm).sum() / den)
                    sel = mm.flatten() > 0
                    a, b = dx.flatten()[sel], dy.flatten()[sel]
                else:
                    t_l1, r_l1 = float(dx.abs().mean()), float(dy.abs().mean())
                    a, b = dx.flatten(), dy.flatten()
                a = a - a.mean()
                b = b - b.mean()
                denom = (a.norm() * b.norm()).clamp_min(1e-12)
                acc["tgt_view_l1"] = acc.get("tgt_view_l1", 0.0) + t_l1
                acc["rec_view_l1"] = acc.get("rec_view_l1", 0.0) + r_l1
                acc["view_diff_r"] = acc.get("view_diff_r", 0.0) + float((a @ b) / denom)

            acc.setdefault("commit", 0.0)
            acc["commit"] += float(sum(dd.float().mean() for dd in diffs))
            acc["nb"] = acc.get("nb", 0) + 1
            break  # one batch is enough; raise --batch-size for a tighter estimate
    _nb = max(1, acc.get("nb", 1))
    res = {"commit": acc.get("commit", 0.0) / _nb}
    if "tgt_view_l1" in acc:
        res["tgt_view_l1"] = acc["tgt_view_l1"] / _nb
        res["rec_view_l1"] = acc["rec_view_l1"] / _nb
        res["view_diff_ratio"] = res["rec_view_l1"] / max(res["tgt_view_l1"], 1e-12)
        res["view_diff_r"] = acc["view_diff_r"] / _nb
    for v, d in acc.items():
        if not isinstance(v, int):
            continue
        res[f"view{v}_mae"] = d["mae"] / d["n"]
        res[f"view{v}_sat"] = d["sat"] / d["n"]
        res[f"view{v}_raw_std"] = d["std"] / d["n"]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="Run directory (searched recursively) or explicit .pt paths")
    ap.add_argument("--csv", default=None, help="Also write the table to this CSV")
    ap.add_argument(
        "--recon",
        action="store_true",
        help="Tier 2: also run ONE forward pass per checkpoint and report PER-VIEW pixel error, "
        "decoder saturation and raw output std. Needs the run's settings.json and the synthetic "
        "generator. This is the readout that separates a per-view capacity loss from a uniform one.",
    )
    ap.add_argument("--run-dir", default=None, help="Run dir with settings.json (defaults to the checkpoint's dir)")
    ap.add_argument("--recon-samples", type=int, default=64, help="Synthetic samples to render for --recon")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size for --recon")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--iid",
        action="store_true",
        help="Force i.i.d. eval factors. Default forwards the run's SCM so the reconstruction is "
        "measured on the distribution the model was trained on.",
    )
    args = ap.parse_args()

    paths = _find_checkpoints(args.targets)
    if not paths:
        raise SystemExit("No .pt/.pth checkpoints found.")

    rows = []
    for p in paths:
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [skip] {os.path.basename(p)}: {e}")
            continue
        if not isinstance(ckpt, dict):
            continue
        row = {"file": os.path.basename(p), "step": ckpt.get("step", _step_of(p))}
        g = _grad_stats(ckpt)
        if g:
            row.update(g)
        sd = _encoder_sd(ckpt)
        for lvl, cb in _codebook_stats(sd).items():
            row[f"cb[{lvl}].active"] = cb["active"]
            row[f"cb[{lvl}].ppl"] = cb["perplexity"]
        ng = _norm_gamma_stats(sd)
        if ng:
            row.update(ng)
        rows.append(row)

    if not rows:
        raise SystemExit("No readable checkpoints.")
    rows.sort(key=lambda r: (r.get("step") if isinstance(r.get("step"), int) else -1))

    if len(rows) < 2:
        print("\n" + "!" * 78)
        print("ONLY ONE CHECKPOINT FOUND — every reading below is COMPARATIVE and cannot be")
        print("interpreted alone. VQ-VAE mode overwrites a single vqvae_model.pt (see")
        print("utils/checkpointing.py), so a run keeps no history. Comparison points that")
        print("usually exist anyway:")
        print("  * vqvae_best.pt in the same directory — full optimizer state, and usually a")
        print("    PRE-jump snapshot, since a jump degrades whatever metric drives selection.")
        print("  * the matching recon-only / baseline run — the control for the gamma numbers.")
        print("  * emergency_*.pt from any OOM or crash, which also carry optimizer state.")
        print("Point this script at several at once:  python -m eval.checkpoint_forensics a.pt b.pt")
        print("For future runs, --save-all-checkpoints keeps versioned encoders (weights only,")
        print("no optimizer state, so they give gammas and codebook but no gradient/clip row).")
        print("!" * 78)

    print("\n" + "=" * 78)
    print("GRADIENT / CLIP REGIME  (from Adam exp_avg_sq; clip max_norm = %.1f)" % CLIP_MAX_NORM)
    print("=" * 78)
    print(f"  {'step':>8}  {'grad_norm':>10}  {'vs clip':>8}  {'clip binding?':>14}  {'file':<28}")
    for r in rows:
        gn = r.get("grad_norm_rms")
        if gn is None:
            continue
        frac = r["clip_binding"]
        verdict = "YES (throttled)" if frac > 0.95 else ("partial" if frac > 0.5 else "no")
        print(f"  {r['step']:>8}  {gn:>10.4f}  {frac:>7.2f}x  {verdict:>14}  {r['file']:<28}")
    print("\n  Reading: grad_norm is the POST-clip norm, so it cannot exceed the clip. Pinned at")
    print("  ~%.1f => the clip was binding and every parameter was rescaled by 2/||g||." % CLIP_MAX_NORM)
    print("  A DROP between two checkpoints means the clip stopped binding and the effective")
    print("  learning rate for the whole model rose at that point.")

    cb_keys = sorted({k for r in rows for k in r if k.startswith("cb[")})
    if cb_keys:
        print("\n" + "=" * 78)
        print("CODEBOOK  (from the cluster_size buffer)")
        print("=" * 78)
        print("  " + f"{'step':>8}  " + "  ".join(f"{k:>22}" for k in cb_keys))
        for r in rows:
            print("  " + f"{r['step']:>8}  " + "  ".join(f"{r.get(k, float('nan')):>22.3f}" for k in cb_keys))

    if any("gamma_mean" in r for r in rows):
        print("\n" + "=" * 78)
        print("NORM GAMMAS  (feature-scale inflation)")
        print("=" * 78)
        hdr = [
            "gamma_p99",
            "split_gamma_spread",
            "split_eff_dim",
            "content_gamma_spread",
            "content_eff_dim",
            "content_n",
            "style_gamma_spread",
            "style_eff_dim",
        ]
        print("  " + f"{'step':>8}  " + "  ".join(f"{h:>13}" for h in hdr))
        for r in rows:
            print("  " + f"{r['step']:>8}  " + "  ".join(f"{r.get(h, float('nan')):>13.3f}" for h in hdr))
        print("\n  split_eff_dim = participation ratio of gamma^2 over the SplitGroupNorm channels:")
        print("  how many of split_n_channels actually influence which codebook entry is chosen.")
        print("  Well below the channel count means the codebook resolves only a few directions")
        print("  and the rest of the content block is effectively discarded at the quantizer —")
        print("  which degrades reconstruction while leaving the commitment loss LOW.")

    if args.recon:
        print("\n" + "=" * 78)
        print("RECONSTRUCTION, PER VIEW  (one forward pass per checkpoint)")
        print("=" * 78)
        hdr = [
            "view0_mae",
            "view1_mae",
            "tgt_view_l1",
            "rec_view_l1",
            "view_diff_ratio",
            "view_diff_r",
            "view0_sat",
            "view1_sat",
            "commit",
        ]
        print("  " + f"{'step':>8}  " + "  ".join(f"{h:>13}" for h in hdr))
        for r in rows:
            ck = os.path.join(os.path.dirname(paths[0]), r["file"])
            ck = ck if os.path.exists(ck) else next((p for p in paths if os.path.basename(p) == r["file"]), None)
            rd = args.run_dir or os.path.dirname(ck)
            try:
                st = _recon_stats(rd, ck, args.device, args.recon_samples, args.batch_size, None if args.iid else True)
            except Exception as e:
                print(f"  {r['step']:>8}  [failed] {type(e).__name__}: {e}")
                continue
            r.update(st)
            print("  " + f"{r['step']:>8}  " + "  ".join(f"{st.get(h, float('nan')):>13.5f}" for h in hdr))
        print("\n  view0_mae vs view1_mae is the discriminator: a gap that OPENS between two")
        print("  checkpoints localises the degradation to one encoder / style codebook. Both")
        print("  rising together means the cause is shared and the per-view split is a bystander.")
        print("  view*_sat > 0 with raw_std growing is the unbounded-decoder runaway instead.")
        print("\n  tgt_view_l1  how much the two views' TARGETS differ (the signal to reproduce)")
        print("  rec_view_l1  how much the two RECONSTRUCTIONS differ (what survives to output)")
        print("  view_diff_ratio = rec/tgt: 1.0 = the view difference is fully reproduced,")
        print("                   ~0 = the model emits essentially the same image for both views")
        print("                   and a low per-view MAE only means the targets are similar.")
        print("  view_diff_r  correlation of (x_v0-x_v1) with (y_v0-y_v1) — whether the")
        print("               difference is placed CORRECTLY, not just scaled right. This is the")
        print("               stronger of the two; ratio near 1 with r near 0 is a model")
        print("               manufacturing view difference in the wrong places.")
        print("  A run with view_diff_ratio ~0 is NOT a control for anything about view identity.")

    if args.csv:
        import csv as _csv

        cols = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
