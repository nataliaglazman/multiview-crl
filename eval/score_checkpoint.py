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
2b. **Per encoder** — the same metrics for each view's encoder separately, each against
   its own floor. Under ``--separate-encoders`` (the default) the two are tied only by a
   loss on their OUTPUTS, so nothing makes them equally good at carrying a factor;
   reporting view 1 as "the model" hides a lopsided pair.
2c. **Style block** — the units after ``content_channels``, per view: style->style should
   be high, content->style and style->content should stay near the floor. Style targets
   are the gain / bias / noise sigma the renderer applied to that view, not the raw draws.
3. **Causal graph** — the PC algorithm run on the decoded factors, scored against the
   generator's true SCM adjacency, via ``eval.run_causal_recovery.evaluate_arrays`` (the
   same protocol the rest of the repo uses; not re-derived here). Alongside it, PC run on
   the ground-truth factors themselves, which bounds what any encoder could reach on this
   sample size.

Optionally, --lesion-analysis compares backbone, projected content and style features
at GAP and spatial grids, predicting both lesion controls and rendered centroids.
It includes matched untrained and shuffled-label controls. --lesion-analysis-only
skips the usual recovery/graph work and runs only this frozen diagnostic.

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
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import SyntheticBrainDataset
from eval.dci import CONTENT_FACTOR_NAMES, STYLE_FACTOR_NAMES
from eval.identifiability_metrics import block_mcc, channel_mcc, cv_probe_acc, cv_probe_r2
from models.multiview_encoder import MultiviewConvEncoder


def parse_args(argv=None):
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
    p.add_argument(
        "--no-per-encoder",
        action="store_true",
        help="Score view 1 only. By default both views' encoders are scored separately, "
        "since under --separate-encoders nothing forces them to be equally good.",
    )
    p.add_argument(
        "--no-dci",
        action="store_true",
        help="Skip the GBT DCI scores. ~1s at gap pooling; the cost grows with the feature "
        "count, so it is worth skipping at wide patch poolings.",
    )
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
    p.add_argument(
        "--lesion-analysis", action="store_true", help="Append backbone/content/style lesion and centroid probes"
    )
    p.add_argument(
        "--lesion-analysis-only", action="store_true", help="Run only lesion analysis, skipping recovery and graph"
    )
    p.add_argument(
        "--lesion-grids",
        type=int,
        nargs="+",
        default=None,
        help="Cubic grids for lesion analysis; GAP (1) always included. Default: 1 and 4, or native size if smaller",
    )
    p.add_argument(
        "--lesion-shuffles", type=int, default=3, help="Number of matched shuffled-target controls (default: 3)"
    )
    p.add_argument("--lesion-seed", type=int, default=1729, help="Seed for lesion target permutations")
    args = p.parse_args(argv)
    args.lesion_analysis = args.lesion_analysis or args.lesion_analysis_only
    if args.lesion_analysis:
        if args.lesion_shuffles < 1:
            p.error("--lesion-shuffles must be at least 1")
        if args.lesion_grids is not None and any(grid < 1 for grid in args.lesion_grids):
            p.error("--lesion-grids must be positive")
        if args.batch_size < 1:
            p.error("--batch-size must be positive")
    return args


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
        encoder_architecture=cfg.get("encoder_architecture", "conv"),
        encoder_head_hidden=cfg.get("encoder_head_hidden", 100),
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model.to(device).eval()


def make_dataset(cfg, num_samples, mode="val"):
    """Restore the encoder-only run's generator on a named subject split."""
    if mode not in ("train", "val", "test"):
        raise ValueError(f"Unknown dataset split: {mode!r}")
    res = cfg["res"]
    return SyntheticBrainDataset(
        mode=mode,
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
        # Read back, not defaulted: rebuilding the split with the default chain graph for a
        # run trained on a random one would score PC against an SCM the model never saw,
        # and nothing downstream would flag it.
        synthetic_causal_graph=cfg.get("synthetic_causal_graph", "chain"),
        synthetic_causal_edge_prob=cfg.get("synthetic_causal_edge_prob", 0.5),
        synthetic_causal_noise_scale=cfg.get("synthetic_causal_noise_scale", 0.4),
        synthetic_causal_nonlinearity=cfg.get("synthetic_causal_nonlinearity", "leaky_relu"),
        synthetic_clean_content=cfg.get("synthetic_clean_content", False),
        synthetic_lesion_placement=cfg.get("synthetic_lesion_placement", "legacy"),
    )


def make_val_dataset(cfg, num_samples):
    """Backwards-compatible validation factory used by existing checkpoint probes."""
    return make_dataset(cfg, num_samples, mode="val")


@torch.no_grad()
def encode_blocks(model, ds, device, batch_size, content_channels, patch_grid=None):
    """Content and style blocks of both views from one forward pass, plus their GT factors.

    Returns a dict with ``content`` and ``style`` as ``(view_1, view_2)`` arrays (style has
    zero columns when ``latent_dim == content_channels``), ``z_content``, ``adjacency`` (or
    None) and ``style_targets``: per view, the gain / bias / noise sigma the renderer applied
    (``eval.style_path_audit.effective_style``). The raw draws are not the target: the
    renderer clips gain and bias to [-1, 1] and uses only |z| for noise, so the noise draw's
    sign never reaches the image and a raw target caps even a perfect block near R² 0.
    ``style_targets`` is None when the samples carry no style draws or no renderer.
    """
    blocks = {"content": ([], []), "style": ([], [])}
    gt, adj = [], None
    renderer = getattr(getattr(ds, "_inner", None), "renderer", None)
    style_scale = getattr(renderer, "style_scale", None)
    if style_scale is not None:
        from eval.style_path_audit import effective_style
    style_gt = ([], [])
    for batch in DataLoader(ds, batch_size=batch_size):
        x = torch.cat(batch["image"], dim=0).to(device)
        feats = model(x, pool_only=True, n_views=2, patch_grid=patch_grid)[2][0]
        feats = feats.reshape(feats.shape[0], feats.shape[1], -1) if feats.dim() > 2 else feats.unsqueeze(-1)
        n = feats.shape[0] // 2
        for name, block in (("content", feats[:, :content_channels]), ("style", feats[:, content_channels:])):
            block = block.flatten(1).cpu()
            blocks[name][0].append(block[:n])
            blocks[name][1].append(block[n:])
        latents = batch["gt_latents"]
        gt.append(latents["z_content"].numpy())
        if style_scale is not None and "z_style_v1" in latents:
            for view in range(2):
                draws = latents[f"z_style_v{view + 1}"]
                style_gt[view].append(np.stack([effective_style(z, style_scale)[: draws.shape[1]] for z in draws]))
        if adj is None and "causal_adj" in latents:
            # One copy per sample comes out of the collate; they are all the same SCM.
            adj = latents["causal_adj"][0].numpy().astype(bool)
    style_targets = None
    if style_gt[0] and style_gt[0][0].shape[1] > 0:
        style_targets = tuple(np.concatenate(parts) for parts in style_gt)
    return {
        "content": tuple(torch.cat(parts).numpy() for parts in blocks["content"]),
        "style": tuple(torch.cat(parts).numpy() for parts in blocks["style"]),
        "z_content": np.concatenate(gt),
        "style_targets": style_targets,
        "adjacency": adj,
    }


def encode(model, ds, device, batch_size, content_channels, patch_grid=None):
    """Content block of both views, plus GT factors and the SCM adjacency when present.

    Returns ``(content_v1, content_v2, z_content, adjacency_or_None)``. The content block is
    the first ``content_channels`` units, matching the split the loss and the training-time
    eval both use.
    """
    blocks = encode_blocks(model, ds, device, batch_size, content_channels, patch_grid)
    return (*blocks["content"], blocks["z_content"], blocks["adjacency"])


def dci_scores(X, z, train_ratio=0.8):
    """D/C/I from ``eval.dci._compute_dci`` — the repo's GBT implementation, not re-derived.

    Split positionally at ``train_ratio`` and transposed to (features, samples), matching
    how ``compute_dci_synthetic`` calls it, so these numbers line up with the ones the
    training-time eval writes rather than being a second opinion computed differently.
    """
    from eval.dci import _compute_dci

    split = int(len(X) * train_ratio)
    scores, _ = _compute_dci(X[:split].T, z[:split].T, X[split:].T, z[split:].T, ["continuous"] * z.shape[1])
    return {k: float(v) for k, v in scores.items()}


def recovery(X, X_v2, z, names, with_dci=True):
    """Per-factor ridge R², both MCC flavours, DCI, and the content->view leakage probe."""
    mcc = block_mcc(X, z)
    chan = channel_mcc(X, z)
    chan_s = channel_mcc(X, z, method="spearman")
    per_factor = {
        nm: {
            "ridge_r2": float(cv_probe_r2(X, z[:, j])["mean"]),
            "mcc": float(mcc["per_factor"][j]),
            "mcc_std": float(mcc["per_factor_std"][j]),
            "channel_mcc": float(chan["per_factor"][j]),
            "channel_mcc_spearman": float(chan_s["per_factor"][j]),
        }
        for j, nm in enumerate(names)
    }
    labels = np.array([0] * len(X) + [1] * len(X_v2))
    out = {
        "block_mcc": float(mcc["mean"]),
        "channel_mcc": float(chan["mean"]),
        "channel_mcc_spearman": float(chan_s["mean"]),
        "ridge_r2_mean": float(np.mean([v["ridge_r2"] for v in per_factor.values()])),
        "content_to_view_acc": float(cv_probe_acc(np.vstack([X, X_v2]), labels)["mean"]),
        "assignment_identity": float(mcc["assignment_identity"]),
        "channel_assignment_identity": float(chan["assignment_identity"]),
        "n_matched_channels": int(chan["n_matched"]),
        "per_factor": per_factor,
    }
    if with_dci:
        out["dci"] = dci_scores(X, z)
    return out


def style_recovery(style, content, style_targets, z, content_names, style_names):
    """One view's style block against its own style factors, plus both leakage directions.

    The same cross-validated ridge probe as ``recovery``. A target that is constant across
    subjects has no defined R² and is reported as NaN rather than scored.
    """

    def probe(X, Y, names):
        per_factor = {
            nm: float(cv_probe_r2(X, Y[:, j])["mean"]) if np.ptp(Y[:, j]) > 1e-12 else float("nan")
            for j, nm in enumerate(names)
        }
        finite = [v for v in per_factor.values() if np.isfinite(v)]
        return {"mean": float(np.mean(finite)) if finite else float("nan"), "per_factor": per_factor}

    return {
        "n_style_channels": int(style.shape[1]),
        "style_to_style": probe(style, style_targets, style_names),
        "content_to_style": probe(content, style_targets, style_names),
        "style_to_content": probe(style, z, content_names),
    }


def print_style(views, floors, style_names):
    """Both views' style blocks side by side, each against its own untrained floor."""
    print(f"\n=== style block ({views[0]['n_style_channels']} features per view) ===", flush=True)
    has_floor = floors is not None
    head = f"  {'ridge R²':<26s}"
    for view in ("view 1", "view 2"):
        head += f"{view:>9s}" + (f"{'floor':>9s}{'Δ':>9s}" if has_floor else "")
    print(head, flush=True)
    rows = [(f"style -> {nm}", "style_to_style", nm) for nm in style_names]
    rows += [(f"content -> {nm}", "content_to_style", nm) for nm in style_names]
    rows += [("style -> content (mean)", "style_to_content", None)]
    for label, block, nm in rows:
        line = f"  {label:<26s}"
        for v, res in enumerate(views):
            value = res[block]["mean"] if nm is None else res[block]["per_factor"][nm]
            line += f"{value:>9.3f}"
            if has_floor:
                f = floors[v][block]["mean"] if nm is None else floors[v][block]["per_factor"][nm]
                line += f"{f:>9.3f}{value - f:>+9.3f}"
        print(line, flush=True)
    print("  style -> style should be high; content -> style and style -> content near the floor.", flush=True)


def print_recovery(res, floor, names):
    head = "" if floor is None else f"{'floor':>10s}{'Δ':>10s}"
    print("\n=== recovery (content block) ===", flush=True)
    print(f"  {'metric':<24s}{'value':>10s}" + head, flush=True)
    rows = [
        ("block_mcc", "block MCC"),
        ("channel_mcc", "channel MCC"),
        ("channel_mcc_spearman", "channel MCC (spearman)"),
        ("ridge_r2_mean", "ridge R² (mean)"),
        ("content_to_view_acc", "content->view acc"),
    ]
    for key, label in rows:
        line = f"  {label:<24s}{res[key]:>10.3f}"
        if floor is not None:
            line += f"{floor[key]:>10.3f}{res[key] - floor[key]:>+10.3f}"
        print(line, flush=True)
    print(f"  {'(MCC assignment id.)':<24s}{res['assignment_identity']:>10.3f}", flush=True)

    if res.get("dci"):
        print("\n=== DCI (GBT importances) ===", flush=True)
        print(f"  {'metric':<24s}{'value':>10s}" + head, flush=True)
        for key, label in (
            ("disentanglement", "disentanglement"),
            ("completeness", "completeness"),
            ("informativeness_test", "informativeness (test)"),
            ("informativeness_train", "informativeness (train)"),
        ):
            line = f"  {label:<24s}{res['dci'][key]:>10.3f}"
            if floor is not None and floor.get("dci"):
                line += f"{floor['dci'][key]:>10.3f}{res['dci'][key] - floor['dci'][key]:>+10.3f}"
            print(line, flush=True)

    print("\n=== per factor ===", flush=True)
    print(
        f"  {'factor':<20s}{'ridge R²':>10s}{'blockMCC':>10s}{'±':>7s}{'chanMCC':>9s}"
        + ("" if floor is None else f"{'R² floor':>10s}{'Δ':>10s}"),
        flush=True,
    )
    for nm in names:
        v = res["per_factor"][nm]
        line = (
            f"  {nm:<20s}{v['ridge_r2']:>10.3f}{v['mcc']:>10.3f}{v['mcc_std']:>7.3f}"
            f"{v.get('channel_mcc', float('nan')):>9.3f}"
        )
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


ENCODER_ROWS = (
    ("block_mcc", "block MCC"),
    ("channel_mcc", "channel MCC"),
    ("ridge_r2_mean", "ridge R² (mean)"),
)


def print_encoder_comparison(v1, v2, floor_v1, floor_v2, shared_encoder):
    """The two views' encoders side by side, each against its own floor.

    Each view gets its own floor column because the two untrained encoders are separate
    random draws (``encoder_v1`` is a deep copy at init but diverges immediately), so a
    view-1 floor is not the right reference for view 2.
    """
    print("\n=== per encoder ===", flush=True)
    if shared_encoder:
        print("  (--no-separate-encoders: one encoder, two view inputs)", flush=True)
    has_floor = floor_v1 is not None and floor_v2 is not None
    head = f"  {'metric':<20s}{'view 1':>9s}{'view 2':>9s}{'gap':>9s}"
    print(head + (f"{'Δ v1':>9s}{'Δ v2':>9s}" if has_floor else ""), flush=True)
    for key, label in ENCODER_ROWS:
        a, b = v1[key], v2[key]
        line = f"  {label:<20s}{a:>9.3f}{b:>9.3f}{a - b:>+9.3f}"
        if has_floor:
            line += f"{a - floor_v1[key]:>+9.3f}{b - floor_v2[key]:>+9.3f}"
        print(line, flush=True)
    if v1.get("dci") and v2.get("dci"):
        for key, label in (("disentanglement", "DCI disentangle."), ("completeness", "DCI completeness")):
            a, b = v1["dci"][key], v2["dci"][key]
            line = f"  {label:<20s}{a:>9.3f}{b:>9.3f}{a - b:>+9.3f}"
            if has_floor and floor_v1.get("dci") and floor_v2.get("dci"):
                line += f"{a - floor_v1['dci'][key]:>+9.3f}{b - floor_v2['dci'][key]:>+9.3f}"
            print(line, flush=True)
    # content->view accuracy is one probe over both views' rows, so it is a property of the
    # pair and identical whichever view is passed first. Printed once, not per encoder.
    print(f"  {'content->view acc':<20s}{v1['content_to_view_acc']:>9.3f}   (shared: one probe over both)", flush=True)

    # Per factor, because a matching pair of MEANS can still hide the two encoders having
    # split the factors between them — each carrying what the other dropped.
    pf1, pf2 = v1.get("per_factor") or {}, v2.get("per_factor") or {}
    if not pf1 or not pf2:
        return
    print("\n  --- per factor, both encoders ---", flush=True)
    print(
        f"  {'factor':<20s}{'v1 R²':>9s}{'v2 R²':>9s}{'gap':>9s}"
        f"{'v1 bMCC':>10s}{'v2 bMCC':>10s}{'gap':>9s}"
        f"{'v1 cMCC':>10s}{'v2 cMCC':>10s}{'gap':>9s}",
        flush=True,
    )
    for nm in pf1:
        a, b = pf1[nm], pf2.get(nm, {})
        cells = ""
        for key in ("ridge_r2", "mcc", "channel_mcc"):
            x, y = a.get(key, float("nan")), b.get(key, float("nan"))
            width = 9 if key == "ridge_r2" else 10
            cells += f"{x:>{width}.3f}{y:>10.3f}{x - y:>+9.3f}"
        print(f"  {nm:<20s}{cells}", flush=True)


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


def append_style(report, blocks, floor_blocks, names):
    """Score and print both views' style blocks, or record why there is nothing to score."""
    targets = blocks["style_targets"]
    if blocks["style"][0].shape[1] == 0:
        report["style_status"] = "no_style_units"
        print(
            "\n=== style block ===\n  SKIPPED: latent_dim == content_channels, so there are no style units.", flush=True
        )
        return
    if targets is None:
        report["style_status"] = "no_style_factors"
        print("\n=== style block ===\n  SKIPPED: the samples carry no rendered style factors.", flush=True)
        return
    style_names = STYLE_FACTOR_NAMES[: targets[0].shape[1]]

    def score(b):
        return {
            f"view{v + 1}": style_recovery(
                b["style"][v], b["content"][v], targets[v], b["z_content"], names, style_names
            )
            for v in range(2)
        }

    report["style_status"] = "ok"
    report["style"] = score(blocks)
    floors = None
    if floor_blocks is not None:
        report["style_floor"] = score(floor_blocks)
        floors = list(report["style_floor"].values())
    print_style(list(report["style"].values()), floors, style_names)


def append_lesion_analysis(report, model, ds, cfg, device, args):
    from eval.checkpoint_lesion_analysis import print_analysis, run_analysis

    grids = args.lesion_grids
    if grids is None:
        if cfg.get("encoder_architecture", "conv") == "resnet18":
            native = (cfg["res"] + 31) // 32
        else:
            native = cfg["res"] // cfg["downscale_factor"]
        grids = [1, min(4, max(1, native))]
    floor_factory = None if args.no_floor else lambda: build_model(cfg, device)
    report["lesion_analysis"] = run_analysis(
        model, floor_factory, ds, device, args.batch_size, grids, args.lesion_shuffles, args.lesion_seed
    )
    print_analysis(report["lesion_analysis"])


def write_report(report, args):
    output = args.out
    if output is None and args.lesion_analysis:
        output = str(Path(args.run_dir) / f"score_lesion_{datetime.now():%Y%m%d_%H%M%S_%f}.json")
    if output is None:
        # Default rather than skip: plot_score_checkpoint reads this file, and a figure that
        # can disagree with the numbers it came from is worse than no figure.
        output = str(Path(args.run_dir) / "score_report.json")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if args.lesion_analysis:
        from eval.checkpoint_lesion_analysis import json_safe, save_tables

        with open(output, "w") as fp:
            json.dump(json_safe(report), fp, indent=2, default=float, allow_nan=False)
        for table in save_tables(report["lesion_analysis"], output):
            print(f"wrote {table}", flush=True)
    else:
        with open(output, "w") as fp:
            json.dump(report, fp, indent=2, default=float)
    print(f"\nwrote {output}", flush=True)


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
    state = torch.load(os.path.join(args.run_dir, args.checkpoint), map_location="cpu")
    model = build_model(cfg, device, state)
    del state
    report = {
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "pooling": pooling,
        "patch_grid": patch_grid,
        "num_samples": int(len(ds)),
        "settings": cfg,
    }
    if args.lesion_analysis_only:
        report["mode"] = "lesion_analysis_only"
        append_lesion_analysis(report, model, ds, cfg, device, args)
        write_report(report, args)
        return
    blocks = encode_blocks(model, ds, device, args.batch_size, cfg["content_channels"], patch_grid)
    X, X2 = blocks["content"]
    z, adj = blocks["z_content"], blocks["adjacency"]
    names = CONTENT_FACTOR_NAMES[: z.shape[1]]

    report = {"run_dir": args.run_dir, "pooling": pooling, "num_samples": int(len(X)), "settings": cfg}
    report["trained"] = recovery(X, X2, z, names, with_dci=not args.no_dci)

    floor = None
    fX = fX2 = None
    floor_blocks = None
    if not args.no_floor:
        floor_blocks = encode_blocks(
            build_model(cfg, device), ds, device, args.batch_size, cfg["content_channels"], patch_grid
        )
        fX, fX2 = floor_blocks["content"]
        floor = recovery(fX, fX2, z, names, with_dci=not args.no_dci)
        report["floor"] = floor
    print_recovery(report["trained"], floor, names)

    # View 2 goes through its own encoder under --separate-encoders (the default), and the
    # two are trained only by a loss that ties their OUTPUTS together -- nothing makes them
    # equally good at carrying a factor. Scoring one and reporting it as "the model" hides
    # that, so score both.
    if not args.no_per_encoder:
        shared = cfg.get("no_separate_encoders", False)
        report["trained_v2"] = recovery(X2, X, z, names, with_dci=not args.no_dci)
        floor_v2 = None
        if floor is not None:
            floor_v2 = recovery(fX2, fX, z, names, with_dci=not args.no_dci)
            report["floor_v2"] = floor_v2
        print_encoder_comparison(report["trained"], report["trained_v2"], floor, floor_v2, shared)

    append_style(report, blocks, floor_blocks, names)

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

    if args.lesion_analysis:
        append_lesion_analysis(report, model, ds, cfg, device, args)
    write_report(report, args)

    if args.plot or args.plot_dir:
        from eval.plot_score_checkpoint import render

        for path in render(report, args.plot_dir or os.path.join(args.run_dir, "figures")):
            print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
