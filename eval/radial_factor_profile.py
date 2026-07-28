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
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

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
