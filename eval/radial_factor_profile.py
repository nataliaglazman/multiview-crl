"""Is the background hotspot part of a MONOTONE falloff, or a separate second peak?

`background_leak_mechanism.py` profiles background positions by distance to the brain and
compares norms. It never looks INSIDE the brain, so it cannot answer the sharpest objection
to a receptive-field explanation:

    if the edge signal were ordinary conv reach, tissue BETWEEN the ventricle and the
    background is strictly closer to the ventricle, so it would have to carry at least as
    much signal. A high-centre / low-middle / high-edge profile is impossible under reach.

So this script profiles EVERY position by distance from the volume centre — through the
ventricle, out through tissue, across the boundary, into the background — and asks whether
the profile is monotone (reach) or bimodal (something else).

Three mechanisms predict three different shapes, and the script separates them:

  REACH        monotone decay from the centre, hitting zero past the RF radius.
  GLOBAL       flat everywhere, with a DIP in mid-brain: every position carries the same
  SUMMARY      whole-volume statistic, but in-brain positions additionally carry local
               anatomy, which for a factor that is not that anatomy acts as probe noise.
               Bimodal *because the middle is noisier*, not because the ends know more.
  BOUNDARY     bimodal for a different reason: the brain/background edge is where brain
               SIZE is legible, and under a random SCM ventricle_size is correlated with
               brain_size. Mid-brain sees homogeneous tissue and no edge.

GLOBAL vs BOUNDARY is settled by the `|bsize` columns, which probe the part of the factor
that brain_size does NOT explain. If the second peak is boundary-mediated it collapses
there; if it is a genuine global summary it survives.

The norm arms separate GLOBAL from the other two: GroupNorm couples every position to the
whole volume, per-voxel LayerNorm does not, and both arms share conv weights (same seed).

Runs on an UNTRAINED encoder, like background_leak_mechanism -- the question is what the
architecture crossed with the data statistics forces, before any training.

Usage:
  python -m eval.radial_factor_profile --num-samples 400 --downscale 4
  python -m eval.radial_factor_profile --num-samples 120 --res 128 --downscale 4 \
      --normalize fixed_reference --causal
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from eval.background_leak_mechanism import _stub_utils_if_needed, build_encoder, generate, probe

logger = logging.getLogger(__name__)

FACTORS = ("ventricle_size", "brain_size")


def resid_on(y, b):
    """Part of y that a linear function of b does not explain."""
    b = np.asarray(b, np.float64).reshape(len(b), -1)
    y = np.asarray(y, np.float64)
    return y - LinearRegression().fit(b, y).predict(b)


def shell_bins(g, n_shells):
    """Radial bins from the volume centre, in feature voxels. Returns [((lo,hi), mask), ...]."""
    ii, jj, kk = np.meshgrid(*[np.arange(g)] * 3, indexing="ij")
    c = (g - 1) / 2.0
    d = np.sqrt((ii - c) ** 2 + (jj - c) ** 2 + (kk - c) ** 2).reshape(-1)
    out = []
    for lo, hi in zip(np.linspace(0, d.max(), n_shells + 1)[:-1], np.linspace(0, d.max(), n_shells + 1)[1:]):
        sel = (d >= lo) & (d < hi)
        if int(sel.sum()) >= 20:
            out.append(((lo, hi), sel))
    return out


def run_checkpoint(cli):
    """Radial profile of a TRAINED model, using the run's own data settings."""
    import torch.nn.functional as Fn
    from torch.utils.data import DataLoader

    _stub_utils_if_needed()
    from eval.dci import CONTENT_FACTOR_NAMES
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, args, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint, device)
    model.eval()

    res = int(getattr(args, "synthetic_res", 64))
    rates = getattr(args, "vqvae_scaling_rates", [4]) or [4]
    g = cli.grid or max(1, res // int(rates[0]))
    grid = (g, g, g)

    ds = build_synthetic_test_set(args, cli.num_samples)
    if cli.causal_eval:
        # build_synthetic_test_set drops the SCM; rebuild with it forwarded.
        from data.datasets import SyntheticBrainDataset

        ds = SyntheticBrainDataset(
            mode="test",
            spatial_size=getattr(args, "spatial_size", None) or (res, res, res),
            cache=True,
            synthetic_mode=getattr(args, "synthetic_mode", "pseudo_mri"),
            synthetic_seed=getattr(args, "synthetic_seed", 42),
            synthetic_num_samples=cli.num_samples,
            synthetic_n_content=getattr(args, "synthetic_n_content", 9),
            synthetic_n_style=getattr(args, "synthetic_n_style", 3),
            synthetic_normalize=getattr(args, "synthetic_normalize", "per_sample"),
            synthetic_clean_content=getattr(args, "synthetic_clean_content", False),
            synthetic_causal=True,
            synthetic_causal_graph=getattr(args, "synthetic_causal_graph", "random"),
            synthetic_causal_edge_prob=getattr(args, "synthetic_causal_edge_prob", 0.5),
        )

    shells = shell_bins(g, cli.n_shells)
    per_shell = [[] for _ in shells]
    cov_acc, Z = [], []
    for batch in DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=0):
        imgs = batch["image"]
        nv = len(imgs)
        x = torch.cat(imgs, 0).to(device)
        with torch.no_grad():
            out = model(x, pool_only=True, n_views=nv, patch_grid=grid)
            f = (out[2] if isinstance(out, tuple) else out)[0]
            f = f.reshape(nv, -1, *f.shape[1:])[0]  # view 0: (B, C, P)
        for si, (_, sel) in enumerate(shells):
            per_shell[si].append(f[:, :, torch.from_numpy(sel)].mean(2).cpu().numpy())
        m = torch.cat(batch["mask"], 0).to(device).float()
        cov_acc.append(Fn.adaptive_avg_pool3d(m, grid).flatten(1)[: f.shape[0]].cpu().numpy())
        v = batch["gt_latents"]["z_content"]
        Z.append(np.asarray(v).reshape(v.shape[0], -1))

    Z = np.concatenate(Z, 0)
    cov = np.concatenate(cov_acc, 0).mean(0)
    cols = [np.concatenate(a, 0) for a in per_shell]

    y_v = Z[:, CONTENT_FACTOR_NAMES.index("ventricle_size")]
    y_b = Z[:, CONTENT_FACTOR_NAMES.index("brain_size")]
    corr = float(np.corrcoef(y_v, y_b)[0, 1])
    v_res = resid_on(y_v, y_b)

    norm = getattr(args, "norm_type", "?")
    dec = getattr(args, "decoder_norm_type", None) or norm
    print(f"\nrun: {cli.run_dir}")
    print(f"encoder norm: {norm} | decoder norm: {dec} | res {res}^3 stride {rates[0]} -> {g}^3 | N={Z.shape[0]}")
    print(f"eval factors: {'SCM-coupled (--causal-eval)' if cli.causal_eval else 'i.i.d. (default test set)'}")
    print(f"corr(ventricle_size, brain_size) = {corr:+.3f}")
    print("\nR^2 by distance from the VOLUME CENTRE. '|bsize' = part of ventricle brain_size can't explain.\n")
    hdr = f"{'shell (feat vox)':<18}{'n':>6}{'cov':>7}{'vent':>9}{'vent|bsize':>12}{'bsize':>9}"
    print(hdr + "\n" + "-" * len(hdr))
    vent = []
    for si, ((lo, hi), sel) in enumerate(shells):
        F_ = cols[si]
        rv, rvb, rb = probe(F_, y_v), probe(F_, v_res), probe(F_, y_b)
        vent.append(rv)
        print(
            f"{f'[{lo:.1f},{hi:.1f})':<18}{int(sel.sum()):>6}{float(cov[sel].mean()):>7.2f}"
            f"{rv:>9.3f}{rvb:>12.3f}{rb:>9.3f}"
        )

    v = np.array(vent)
    k = max(1, len(v) // 3)
    inner, middle, outer = v[:k].max(), v[k:-k].min(), v[-k:].max()
    lift = float(min(inner, outer) - middle)
    print("\nverdict:")
    if lift > 0.05 and outer > 0.05:
        print(f"  BIMODAL: outer peak {outer:.3f} sits {lift:+.3f} above the mid-brain minimum {middle:.3f}.")
        print("  A receptive-field route cannot produce a second peak at the far edge. If this run uses")
        print("  norm_type=layer, the leak was NOT removed -- check that the checkpoint really is the")
        print("  LayerNorm run (resume_training can silently continue an older GroupNorm checkpoint).")
    else:
        print(f"  MONOTONE: no second peak (outer {outer:.3f} vs mid-brain min {middle:.3f}, lift {lift:+.3f}).")
        print("  The background hotspot is gone; what remains decays from the structure outward.")
    if cli.causal_eval or abs(corr) > 0.3:
        print(
            f"\n  NOTE: corr(ventricle, brain_size) = {corr:+.2f}. Read the 'vent|bsize' column, not 'vent' --"
            "\n  at this correlation the raw ventricle column is largely brain_size."
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-samples", type=int, default=400)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--normalize", default="per_sample", choices=["per_sample", "fixed_reference"])
    ap.add_argument("--clean-content", action="store_true")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--n-shells", type=int, default=10)
    ap.add_argument(
        "--run-dir",
        default=None,
        help="Profile a TRAINED checkpoint instead of the two untrained norm arms. res, stride, "
        "normalization and clean_content are taken from the run's own settings.",
    )
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--grid", type=int, default=None, help="Feature grid per axis (default: run's native).")
    ap.add_argument(
        "--causal-eval",
        action="store_true",
        help="Build the eval set WITH the causal SCM. build_synthetic_test_set never forwards "
        "synthetic_causal, so the default eval set has i.i.d. factors even for an SCM-trained run "
        "-- and under the SCM ventricle_size and brain_size are ~0.8 correlated, which changes "
        "what the ventricle columns mean. Run both.",
    )
    ap.add_argument("--device", default=None)
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if cli.run_dir:
        return run_checkpoint(cli)

    _stub_utils_if_needed()  # eval.dci imports utils.utils, which imports MONAI at module scope
    from eval.dci import CONTENT_FACTOR_NAMES

    res = cli.res
    logger.info(
        "rendering %d volumes at %d^3 (causal=%s, normalize=%s)...",
        cli.num_samples,
        res,
        cli.causal,
        cli.normalize,
    )
    X, M, Z = generate(cli.num_samples, res, cli.causal, cli.seed, cli.normalize, cli.clean_content)
    N = X.shape[0]
    g = res // cli.downscale

    y_v = Z[:, CONTENT_FACTOR_NAMES.index("ventricle_size")]
    y_b = Z[:, CONTENT_FACTOR_NAMES.index("brain_size")]
    corr = float(np.corrcoef(y_v, y_b)[0, 1])

    # mean brain coverage per feature position, to label where the brain actually is
    cov = torch.nn.functional.adaptive_avg_pool3d(M.float(), (g, g, g)).flatten(1).mean(0).numpy()

    # distance of each feature position from the volume centre, in FEATURE voxels
    ii, jj, kk = np.meshgrid(*[np.arange(g)] * 3, indexing="ij")
    c = (g - 1) / 2.0
    d_ctr = np.sqrt((ii - c) ** 2 + (jj - c) ** 2 + (kk - c) ** 2).reshape(-1)

    edges = np.linspace(0, d_ctr.max(), cli.n_shells + 1)
    shells = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (d_ctr >= lo) & (d_ctr < hi)
        if int(sel.sum()) >= 20:
            shells.append(((lo, hi), torch.from_numpy(sel)))

    # ── extract per-shell readouts for both norm arms ─────────────────────────────
    acc = {}
    for norm_type in ("group", "layer"):
        enc = build_encoder(norm_type, cli.seed, cli.hidden, downscale=cli.downscale)
        per_shell = [[] for _ in shells]
        for s in range(0, N, cli.batch_size):
            with torch.no_grad():
                f = enc(X[s : s + cli.batch_size]).flatten(2)  # (b, C, P)
            for si, (_, sel_t) in enumerate(shells):
                per_shell[si].append(f[:, :, sel_t].mean(2).numpy())
        acc[norm_type] = [np.concatenate(a, 0) for a in per_shell]

    print(f"\nres {res}^3, stride {cli.downscale} -> {g}^3 features | N={N} | causal={cli.causal}")
    print(f"corr(ventricle_size, brain_size) = {corr:+.3f}")
    print("\nR^2 by distance from the VOLUME CENTRE (all positions, not just background).")
    print("'|bsize' = the part of the factor brain_size does NOT explain.\n")

    hdr = (
        f"{'shell (feat vox)':<18}{'n':>6}{'cov':>7}"
        f"{'GN vent':>9}{'LN vent':>9}{'GN v|bs':>9}{'LN v|bs':>9}{'GN bsize':>10}{'LN bsize':>10}"
    )
    print(hdr + "\n" + "-" * len(hdr))

    v_res = resid_on(y_v, y_b)
    rows = []
    for si, ((lo, hi), sel_t) in enumerate(shells):
        G, L = acc["group"][si], acc["layer"][si]
        r = dict(
            gv=probe(G, y_v),
            lv=probe(L, y_v),
            gvb=probe(G, v_res),
            lvb=probe(L, v_res),
            gb=probe(G, y_b),
            lb=probe(L, y_b),
        )
        rows.append(r)
        mc = float(cov[sel_t.numpy()].mean())
        print(
            f"{f'[{lo:.1f},{hi:.1f})':<18}{int(sel_t.sum()):>6}{mc:>7.2f}"
            f"{r['gv']:>9.3f}{r['lv']:>9.3f}{r['gvb']:>9.3f}{r['lvb']:>9.3f}{r['gb']:>10.3f}{r['lb']:>10.3f}"
        )

    # ── shape verdict ─────────────────────────────────────────────────────────────
    def bimodal(key):
        """Second peak at the outer end that a monotone falloff cannot produce."""
        v = np.array([r[key] for r in rows])
        if len(v) < 4 or not np.isfinite(v).all():
            return False, 0.0
        k = max(1, len(v) // 3)
        inner, middle, outer = v[:k].max(), v[k:-k].min() if len(v) > 2 * k else v[k], v[-k:].max()
        lift = min(inner, outer) - middle
        return (lift > 0.05 and outer > 0.05), float(lift)

    print("\nverdict:")
    for arm, key, kb in (("GroupNorm", "gv", "gvb"), ("LayerNorm", "lv", "lvb")):
        bi, lift = bimodal(key)
        bi_b, lift_b = bimodal(kb)
        shape = f"BIMODAL (outer peak {lift:+.2f} above the middle)" if bi else "monotone / no second peak"
        print(f"  {arm}: {shape}")
        if bi:
            if not bi_b:
                print(
                    f"    -> and it DISAPPEARS once brain_size is removed (lift {lift:+.2f} -> {lift_b:+.2f}):"
                    " BOUNDARY-mediated."
                )
                print("       The outer peak is the brain/background edge encoding brain SIZE, inherited by")
                print(f"       ventricle through the causal graph (corr {corr:+.2f}). Not a normalizer leak.")
            else:
                print(f"    -> and it SURVIVES brain_size removal (lift {lift_b:+.2f}): a genuine global summary,")
                print("       not the boundary. For LayerNorm that would mean a non-normalizer global route.")
    print("\n  A monotone LayerNorm profile alongside a bimodal GroupNorm one is the expected result:")
    print("  reach explains the inside, the normalizer explains the outside, and removing the")
    print("  normalizer removes the second peak.")


if __name__ == "__main__":
    main()
