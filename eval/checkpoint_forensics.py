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
    if split:
        sv = torch.cat([g.flatten() for g in split.values()])
        res["split_gamma_mean"] = float(sv.mean())
        res["split_gamma_max"] = float(sv.abs().max())
        # Heterogeneity across channels is what survives a channel-LayerNorm downstream.
        res["split_gamma_spread"] = float(sv.abs().max() / sv.abs().clamp_min(1e-9).min())
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="Run directory (searched recursively) or explicit .pt paths")
    ap.add_argument("--csv", default=None, help="Also write the table to this CSV")
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

    print("\n" + "=" * 78)
    print("GRADIENT / CLIP REGIME  (from Adam exp_avg_sq; clip max_norm = %.1f)" % CLIP_MAX_NORM)
    print("=" * 78)
    print(f"  {'step':>8}  {'grad_norm':>10}  {'vs clip':>8}  {'clip binding?':>14}")
    for r in rows:
        gn = r.get("grad_norm_rms")
        if gn is None:
            continue
        frac = r["clip_binding"]
        verdict = "YES (throttled)" if frac > 0.95 else ("partial" if frac > 0.5 else "no")
        print(f"  {r['step']:>8}  {gn:>10.4f}  {frac:>7.2f}x  {verdict:>14}")
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
        hdr = ["gamma_mean", "gamma_p99", "split_gamma_mean", "split_gamma_max", "split_gamma_spread"]
        print("  " + f"{'step':>8}  " + "  ".join(f"{h:>18}" for h in hdr))
        for r in rows:
            print("  " + f"{r['step']:>8}  " + "  ".join(f"{r.get(h, float('nan')):>18.4f}" for h in hdr))

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
