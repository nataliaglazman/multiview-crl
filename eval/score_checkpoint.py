"""Score one trained checkpoint: per-factor R², block-MCC, and PC graph recovery.

Takes a ``training.main_conv_synthetic`` run directory (``settings.json`` + ``model.pt``),
rebuilds the model from the settings it was trained with, encodes a fresh validation split,
and reports three things about the resulting vectors:

1. **Recovery** — per-factor cross-validated ridge R² and block-MCC of the content block
   against the ground-truth factors, plus the content->view leakage probe.
2. **Floor** — the same numbers from an untrained twin: same architecture, same seed, no
   training. Every headline is printed as a delta against it. On this generator an
   untrained encoder already scores ~0.38 block-MCC and ~0.16 R² at GAP pooling, so a raw
   number on its own says nothing about what was learned.
3. **Causal graph** — the PC algorithm run on the decoded factors, scored against the
   generator's true SCM adjacency, via ``eval.run_causal_recovery.evaluate_arrays`` (the
   same protocol the rest of the repo uses; not re-derived here). Alongside it, PC run on
   the ground-truth factors themselves, which bounds what any encoder could reach on this
   sample size.

The graph section needs a run whose dataset was built with ``--synthetic-causal``; without
an SCM there is no true graph to score against and the section says so rather than
inventing one.

Example:
    python -m eval.score_checkpoint --run-dir results/dummy_infonce_4
    python -m eval.score_checkpoint --run-dir results/x --pooling patch --patch-grid 8 8 8
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import SyntheticBrainDataset
from eval.dci import CONTENT_FACTOR_NAMES
from eval.identifiability_metrics import block_mcc, cv_probe_acc, cv_probe_r2
from models.multiview_encoder import MultiviewConvEncoder


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", help="Run directory holding settings.json and model.pt")
    p.add_argument("--checkpoint", default="model.pt", help="Checkpoint filename inside --run-dir")
    p.add_argument("--num-samples", type=int, default=None, help="Validation samples (default: the run's)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--pooling", choices=["gap", "patch"], default=None, help="Default: the run's --eval-pooling")
    p.add_argument("--patch-grid", type=int, nargs=3, default=None, help="Default: the run's --eval-patch-grid")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--no-floor", action="store_true", help="Skip the untrained twin (faster, and unreportable)")
    p.add_argument("--no-graph", action="store_true", help="Skip PC graph recovery")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2], help="PC significance sweep")
    p.add_argument("--indep-test", choices=["fisherz", "kci"], default="fisherz")
    p.add_argument("--max-cond-set", type=int, default=None, help="Cap PC's conditioning-set size (needed for kci)")
    p.add_argument(
        "--holdout-readout", action="store_true", help="Fit the graph readout on 70%% and run PC on the rest"
    )
    p.add_argument("--out", default=None, help="Report JSON path (default: <run-dir>/score_report.json)")
    p.add_argument("--plot", action="store_true", help="Also render figures into <run-dir>/figures")
    p.add_argument("--plot-dir", default=None, help="Render figures here instead (implies --plot)")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def load_settings(run_dir):
    with open(os.path.join(run_dir, "settings.json")) as fp:
        return json.load(fp)


def build_model(cfg, device, state_dict=None):
    """Rebuild the trained architecture from the run's settings; random init when no state."""
    torch.manual_seed(cfg.get("seed", 42))
    model = MultiviewConvEncoder(
        in_channels=1,
        hidden_channels=cfg["hidden_channels"],
        res_channels=cfg["res_channels"],
        nb_res_layers=cfg["nb_res_layers"],
        downscale_factor=cfg["downscale_factor"],
        latent_dim=cfg["latent_dim"],
        content_channels=cfg["content_channels"],
        separate_encoders=not cfg.get("no_separate_encoders", False),
        proj_dim=cfg.get("contrastive_proj_dim", 0),
        proj_hidden=cfg.get("contrastive_proj_hidden", 256),
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model.to(device).eval()


def make_val_dataset(cfg, num_samples):
    """The run's own validation distribution — every synthetic knob read back from settings."""
    res = cfg["res"]
    return SyntheticBrainDataset(
        mode="val",
        spatial_size=(res, res, res),
        cache=False,
        synthetic_mode=cfg.get("synthetic_mode", "pseudo_mri"),
        synthetic_seed=cfg.get("seed", 42),
        synthetic_num_samples=num_samples,
        synthetic_n_content=cfg["n_content"],
        synthetic_n_style=cfg["n_style"],
        synthetic_style_scale=cfg.get("synthetic_style_scale", 1.0),
        synthetic_content_scale=cfg.get("synthetic_content_scale", 1.0),
        synthetic_n_deformation_grid=cfg.get("synthetic_n_deformation_grid", 4),
        synthetic_n_fissure_grid=cfg.get("synthetic_n_fissure_grid", 8),
        synthetic_hierarchical_content=cfg.get("synthetic_hierarchical_content", False),
        synthetic_normalize=cfg.get("synthetic_normalize", "per_sample"),
        synthetic_causal=cfg.get("synthetic_causal", False),
        synthetic_clean_content=cfg.get("synthetic_clean_content", False),
    )


@torch.no_grad()
def encode(model, ds, device, batch_size, content_channels, patch_grid=None):
    """Content block of both views, plus GT factors and the SCM adjacency when present.

    Returns ``(content_v1, content_v2, z_content, adjacency_or_None)``. The content block is
    the first ``content_channels`` units, matching the split the loss and the training-time
    eval both use.
    """
    c1, c2, gt, adj = [], [], [], None
    for batch in DataLoader(ds, batch_size=batch_size):
        x = torch.cat(batch["image"], dim=0).to(device)
        feats = model(x, pool_only=True, n_views=2, patch_grid=patch_grid)[2][0]
        feats = feats.reshape(feats.shape[0], feats.shape[1], -1) if feats.dim() > 2 else feats.unsqueeze(-1)
        feats = feats[:, :content_channels].flatten(1).cpu()
        n = feats.shape[0] // 2
        c1.append(feats[:n])
        c2.append(feats[n:])
        gt.append(batch["gt_latents"]["z_content"].numpy())
        if adj is None and "causal_adj" in batch["gt_latents"]:
            # One copy per sample comes out of the collate; they are all the same SCM.
            adj = batch["gt_latents"]["causal_adj"][0].numpy().astype(bool)
    return (torch.cat(c1).numpy(), torch.cat(c2).numpy(), np.concatenate(gt), adj)


def recovery(X, X_v2, z, names):
    """Per-factor ridge R², block-MCC, and the content->view leakage probe."""
    mcc = block_mcc(X, z)
    per_factor = {
        nm: {
            "ridge_r2": float(cv_probe_r2(X, z[:, j])["mean"]),
            "mcc": float(mcc["per_factor"][j]),
            "mcc_std": float(mcc["per_factor_std"][j]),
        }
        for j, nm in enumerate(names)
    }
    labels = np.array([0] * len(X) + [1] * len(X_v2))
    return {
        "block_mcc": float(mcc["mean"]),
        "ridge_r2_mean": float(np.mean([v["ridge_r2"] for v in per_factor.values()])),
        "content_to_view_acc": float(cv_probe_acc(np.vstack([X, X_v2]), labels)["mean"]),
        "assignment_identity": float(mcc["assignment_identity"]),
        "per_factor": per_factor,
    }


def print_recovery(res, floor, names):
    head = "" if floor is None else f"{'floor':>10s}{'Δ':>10s}"
    print("\n=== recovery (content block) ===", flush=True)
    print(f"  {'metric':<24s}{'value':>10s}" + head, flush=True)
    for key, label in (
        ("block_mcc", "block MCC"),
        ("ridge_r2_mean", "ridge R² (mean)"),
        ("content_to_view_acc", "content->view acc"),
    ):
        line = f"  {label:<24s}{res[key]:>10.3f}"
        if floor is not None:
            line += f"{floor[key]:>10.3f}{res[key] - floor[key]:>+10.3f}"
        print(line, flush=True)
    print(f"  {'(MCC assignment id.)':<24s}{res['assignment_identity']:>10.3f}", flush=True)

    print("\n=== per factor ===", flush=True)
    print(
        f"  {'factor':<20s}{'ridge R²':>10s}{'MCC':>8s}{'±':>7s}"
        + ("" if floor is None else f"{'R² floor':>10s}{'Δ':>10s}"),
        flush=True,
    )
    for nm in names:
        v = res["per_factor"][nm]
        line = f"  {nm:<20s}{v['ridge_r2']:>10.3f}{v['mcc']:>8.3f}{v['mcc_std']:>7.3f}"
        if floor is not None:
            f = floor["per_factor"][nm]["ridge_r2"]
            line += f"{f:>10.3f}{v['ridge_r2'] - f:>+10.3f}"
        print(line, flush=True)


def print_graph(panel, title):
    if panel is None:
        return
    best = panel.get("best")
    print(f"\n=== causal graph: {title} ===", flush=True)
    if panel.get("graph_status") != "ok" or best is None:
        print("  PC produced no scorable graph (every alpha errored); see the JSON sweep.", flush=True)
        return
    print(
        f"  best alpha {best['alpha']}: F1 {best['f1']:.3f}  precision {best['precision']:.3f}  "
        f"recall {best['recall']:.3f}  SHD {best['skeleton_shd']}  (tp {best['tp']} fp {best['fp']} fn {best['fn']})",
        flush=True,
    )
    print(f"  raw R² {panel['raw_r2_mean']:.3f} | partial R² {panel['partial_r2_mean']:.3f}", flush=True)
    print(
        "  alpha sweep: "
        + "  ".join(f"{r['alpha']}:{r['f1']:.2f}" if "f1" in r else f"{r['alpha']}:err" for r in panel["alpha_sweep"]),
        flush=True,
    )


def graph_panel(X, z, adjacency, args):
    """PC recovery for one feature matrix, or None when causal-learn is not installed.

    Returning None rather than propagating keeps a missing optional dependency from
    discarding the recovery report that has already been computed and printed.
    """
    from eval.run_causal_recovery import evaluate_arrays

    return evaluate_arrays(
        X,
        z,
        adjacency,
        alphas=tuple(args.alphas),
        indep_test=args.indep_test,
        max_cond_set=args.max_cond_set,
        holdout_readout=args.holdout_readout,
    )


def _self_test():
    """Shape and wiring checks that need no checkpoint and no generator."""
    cfg = dict(
        hidden_channels=64,
        res_channels=8,
        nb_res_layers=1,
        downscale_factor=4,
        latent_dim=6,
        content_channels=4,
        seed=0,
    )
    m = build_model(cfg, "cpu")
    x = torch.randn(4, 1, 16, 16, 16)
    feats = m(x, pool_only=True, n_views=2)[2][0]
    assert feats.shape == (4, 6), feats.shape

    z = np.random.RandomState(0).randn(80, 3)
    X = np.concatenate([z + 0.01 * np.random.RandomState(1).randn(80, 3)], axis=1)
    out = recovery(X, X + 0.01, z, ["a", "b", "c"])
    assert out["block_mcc"] > 0.9, out["block_mcc"]
    assert set(out["per_factor"]) == {"a", "b", "c"}
    assert 0.0 <= out["content_to_view_acc"] <= 1.0
    print("self-test OK (model forward shape, recovery on a near-identity readout)")


def main():
    args = parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.run_dir:
        raise SystemExit("--run-dir is required (or pass --self-test)")

    cfg = load_settings(args.run_dir)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    pooling = args.pooling or cfg.get("eval_pooling", "gap")
    patch_grid = None
    if pooling == "patch":
        patch_grid = tuple(args.patch_grid or cfg.get("eval_patch_grid", [4, 5, 4]))
    n = args.num_samples or cfg.get("num_val_samples", 400)

    print(f"run     : {args.run_dir}", flush=True)
    print(f"device  : {device}   pooling: {pooling}{'' if patch_grid is None else f' {list(patch_grid)}'}", flush=True)
    print(f"encoding: res {cfg['res']}, {n} val samples, content block = {cfg['content_channels']}", flush=True)

    ds = make_val_dataset(cfg, n)
    state = torch.load(os.path.join(args.run_dir, args.checkpoint), map_location=device)
    model = build_model(cfg, device, state)
    X, X2, z, adj = encode(model, ds, device, args.batch_size, cfg["content_channels"], patch_grid)
    names = CONTENT_FACTOR_NAMES[: z.shape[1]]

    report = {"run_dir": args.run_dir, "pooling": pooling, "num_samples": int(len(X)), "settings": cfg}
    report["trained"] = recovery(X, X2, z, names)

    floor = None
    if not args.no_floor:
        fX, fX2, _, _ = encode(
            build_model(cfg, device), ds, device, args.batch_size, cfg["content_channels"], patch_grid
        )
        floor = recovery(fX, fX2, z, names)
        report["floor"] = floor
    print_recovery(report["trained"], floor, names)

    if not args.no_graph:
        if adj is None:
            print(
                "\n=== causal graph ===\n  SKIPPED: this run's dataset has no SCM, so there is no true graph to "
                "score against. Re-run training with --synthetic-causal to enable it.",
                flush=True,
            )
            report["graph_status"] = "no_scm"
        else:
            try:
                report["graph"] = graph_panel(X, z, adj, args)
            except ImportError as exc:
                print(f"\n=== causal graph ===\n  SKIPPED: {exc}", flush=True)
                report["graph_status"] = "no_causallearn"
                adj = None
        if adj is not None:
            print_graph(report["graph"], "decoded from embeddings")
            report["graph_truth"] = graph_panel(z, z, adj, args)
            print_graph(report["graph_truth"], "PC on ground-truth factors (upper bound)")
            if floor is not None:
                report["graph_floor"] = graph_panel(fX, z, adj, args)
                print_graph(report["graph_floor"], "untrained floor")

    # Always write the report: the plotting script reads it, and a figure that can
    # disagree with the numbers it came from is worse than no figure.
    out = args.out or os.path.join(args.run_dir, "score_report.json")
    with open(out, "w") as fp:
        json.dump(report, fp, indent=2, default=float)
    print(f"\nwrote {out}", flush=True)

    if args.plot or args.plot_dir:
        from eval.plot_score_checkpoint import render

        for path in render(report, args.plot_dir or os.path.join(args.run_dir, "figures")):
            print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
