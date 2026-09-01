#!/usr/bin/env python
"""Self-tests for the pure-numpy core of ``probe.site_sharing``.

Every case builds synthetic block Jacobians whose answer is known in advance, so a failure
here is a bug in the statistic rather than a finding about a model.  The three that matter:

  * a site-shared mechanism must read as homogeneous *and* bound (the null must pass);
  * a per-position channel permutation must be caught by binding and be invisible to a
    flattened readout (this is the whole argument for the measurement);
  * conditioning drift must be caught by the spectrum while leaving binding intact.

Run:  python -m tests.test_site_sharing
"""

from __future__ import annotations

import numpy as np

from probe.site_sharing import (
    block_slices,
    build_verdict,
    live_channels,
    match_columns,
    reduce_arm,
    scale_shape,
    shape_deviation,
)

M, D = 96, 12  # block voxels, latent channels


def make_arm(n_subj, sites, rng, *, noise=0.02, perm_after=None, cond_after=None, dead=()):
    """Synthetic Jacobians from one shared mechanism, with optional per-site corruption.

    ``perm_after`` / ``cond_after``: site index from which the corruption starts, so the
    reference site (highest energy, site 0 by construction) stays clean.
    """
    base = rng.normal(size=(M, D))
    base[:, 0] *= 6.0  # make site 0 the highest-energy reference deterministically

    per_subject = []
    for _ in range(n_subj):
        out = {}
        for ui, u in enumerate(sites):
            J = base + noise * rng.normal(size=(M, D))
            # A dead channel is one the decoder does not respond to at all, so it is
            # silenced AFTER the noise: a column carrying signal-scale noise is not dead,
            # and must not be excluded (that would bias binding upward).
            J[:, list(dead)] = 0.0
            J *= 10.0 ** rng.uniform(-2, 2)  # anatomy changes SCALE at every site
            if perm_after is not None and ui >= perm_after:
                p = rng.permutation(D)
                J = J[:, p]
            if cond_after is not None and ui >= cond_after:
                J = J * np.linspace(1.0, cond_after_strength(cond_after), D)[None, :]
            out[u] = {"J": J, "block_energy_fraction": 0.9}
        per_subject.append(out)
    return per_subject


def cond_after_strength(_):
    return 40.0


def sites_list(n):
    return [(i, 0, 0) for i in range(n)]


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    return cond


def main():
    rng = np.random.default_rng(0)
    ok = True

    print("\nscale/shape decomposition")
    J = rng.normal(size=(M, D))
    s1, sh1 = scale_shape(J)
    s2, sh2 = scale_shape(J * 1000.0)
    ok &= check("shape is scale-invariant", np.allclose(sh1, sh2, atol=1e-9), f"max|d|={np.abs(sh1 - sh2).max():.2e}")
    ok &= check("scale tracks magnitude", np.isclose(s2 - s1, np.log(1000.0)), f"d={s2 - s1:.4f}")
    ok &= check("shape is zero-mean", abs(sh1.mean()) < 1e-9)
    ok &= check("deviation from self is 0", shape_deviation(np.stack([sh1, sh1]))[0] == 0.0)

    print("\nlive-channel detection")
    norms = np.abs(rng.normal(size=(40, D))) + 1.0
    norms[:, [3, 7]] = 1e-9
    live = live_channels(norms)
    ok &= check("dead channels found", set(np.nonzero(~live)[0]) == {3, 7}, f"dead={np.nonzero(~live)[0]}")

    print("\ncolumn matching")
    A = rng.normal(size=(M, D))
    perm = rng.permutation(D)
    a, c = match_columns(A, A[:, perm])
    ok &= check("identity when unpermuted", (match_columns(A, A)[0] == np.arange(D)).all())
    ok &= check("recovers a known permutation", (a == np.argsort(perm)).all())
    ok &= check("sign flips are absorbed", (match_columns(A, A * -1.0)[0] == np.arange(D)).all())
    ok &= check("matched cos ~ 1", c.min() > 0.999, f"min={c.min():.6f}")

    print("\nnull: one shared mechanism, scale varying 4 orders of magnitude across sites")
    sites = sites_list(24)
    red = reduce_arm(make_arm(4, sites, np.random.default_rng(1)), sites, 1e-3, 2)
    h, b = red["homogeneity"], red["binding"]
    ok &= check("homogeneity ratio ~ 1", h["ratio"] < 3.0, f"ratio={h['ratio']:.2f}")
    ok &= check(
        "binding ~ 1.0", b["primary"]["identity_frac_median"] > 0.95, f"bind={b['primary']['identity_frac_median']:.3f}"
    )
    ok &= check("same-site null ~ 1.0", b["same_site_null"] > 0.95, f"null={b['same_site_null']:.3f}")
    print(
        f"        scale spans {h['scale_log_range'][1] - h['scale_log_range'][0]:.1f} nats and does not move the ratio"
    )

    print("\nviolation: channel labelling permutes from site 12 on")
    red = reduce_arm(make_arm(4, sites, np.random.default_rng(2), perm_after=12), sites, 1e-3, 2)
    bind = red["binding"]["primary"]["identity_frac_median"]
    per_site = red["binding"]["primary"]["per_site_identity"]
    ok &= check("binding collapses", bind < 0.75, f"bind={bind:.3f}")
    ok &= check("clean sites still bind", per_site[:12].mean() > 0.95, f"clean={per_site[:12].mean():.3f}")
    ok &= check("permuted sites do not", per_site[12:].mean() < 0.35, f"permuted={per_site[12:].mean():.3f}")
    ok &= check(
        "spectrum is blind to a permutation",
        red["homogeneity"]["ratio"] < 3.0,
        f"ratio={red['homogeneity']['ratio']:.2f} — this is why binding is a separate test",
    )

    print("\nviolation: conditioning drifts from site 12 on")
    red = reduce_arm(make_arm(4, sites, np.random.default_rng(3), cond_after=12), sites, 1e-3, 2)
    ok &= check(
        "homogeneity ratio blows up", red["homogeneity"]["ratio"] > 10.0, f"ratio={red['homogeneity']['ratio']:.1f}"
    )
    ok &= check(
        "binding survives a conditioning change",
        red["binding"]["primary"]["identity_frac_median"] > 0.9,
        f"bind={red['binding']['primary']['identity_frac_median']:.3f}",
    )

    print("\ndead channels are excluded from both statistics")
    red = reduce_arm(make_arm(4, sites, np.random.default_rng(4), dead=(2, 5, 9)), sites, 1e-3, 2)
    ok &= check("live count correct", red["n_live_channels"] == D - 3, f"live={red['n_live_channels']}")
    ok &= check("dead indices reported", red["dead_channels"] == [2, 5, 9], f"dead={red['dead_channels']}")
    ok &= check("binding unaffected", red["binding"]["primary"]["identity_frac_median"] > 0.95)

    # The default threshold is deliberately permissive: dropping a weak-but-real channel
    # would remove a hard matching case and inflate binding, so only genuine silence is cut.
    weak = np.abs(rng.normal(size=(40, D))) + 1.0
    weak[:, 4] *= 0.01
    ok &= check("weak-but-live channel retained at default", bool(live_channels(weak)[4]))

    print("\nblock geometry")
    for latent, out in (((16, 16, 16), (64, 64, 64)), ((5, 7, 5), (91, 109, 91)), ((3, 3, 3), (8, 8, 8))):
        covered = np.zeros(out, dtype=int)
        empty = False
        for i in range(latent[0]):
            for j in range(latent[1]):
                for k in range(latent[2]):
                    sl = block_slices((i, j, k), latent, out)
                    covered[sl] += 1
                    empty |= any(s.stop <= s.start for s in sl)
        ok &= check(
            f"blocks tile {latent} -> {out} exactly",
            not empty and covered.min() == 1 and covered.max() == 1,
            f"min={covered.min()} max={covered.max()} empty={empty}",
        )

    print("\nverdict logic")
    good = {
        "homogeneity": {"ratio": 1.4},
        "binding": {"primary": {"identity_frac_median": 0.98}},
    }
    bad_ctrl = {
        "homogeneity": {"ratio": 1.0},
        "binding": {"primary": {"identity_frac_median": 0.4}},
    }
    v = build_verdict({"arms": {"full": good, "frozen_norm": good}}, 3.0, 0.9)
    ok &= check("clean run passes", v["ok"])
    v = build_verdict({"arms": {"full": good, "frozen_norm": bad_ctrl}}, 3.0, 0.9)
    ok &= check("broken control fails the run", not v["ok"] and "CONTROL FAILED" in v["lines"][0])
    v = build_verdict({"arms": {"full": {**good, "binding": {"primary": {"identity_frac_median": 0.5}}}}}, 3.0, 0.9)
    ok &= check("low binding fails", not v["ok"])
    v = build_verdict({"arms": {"full": {**good, "homogeneity": {"ratio": 12.0}}}}, 3.0, 0.9)
    ok &= check("inhomogeneous spectra fail", not v["ok"])

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
