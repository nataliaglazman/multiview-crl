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
    effective_rank,
    live_channels,
    match_columns,
    reduce_arm,
    scale_shape,
    shape_deviation,
    window_fits,
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
            out[u] = {"J": J, "energy_profile": {0: 0.9, 1: 0.95}, "rank_profile": {0: float(D), 1: float(D)}}
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
    a, c, m = match_columns(A, A[:, perm])
    ok &= check("identity when unpermuted", (match_columns(A, A)[0] == np.arange(D)).all())
    ok &= check("recovers a known permutation", (a == np.argsort(perm)).all())
    ok &= check("sign flips are absorbed", (match_columns(A, A * -1.0)[0] == np.arange(D)).all())
    ok &= check("matched cos ~ 1", c.min() > 0.999, f"min={c.min():.6f}")
    ok &= check("margin is large for a clean match", np.median(m) > 0.5, f"median margin={np.median(m):.3f}")

    print("\ndegenerate matching is flagged, not silently reported as drift")
    # d columns forced into a 3-dimensional space: every permutation scores alike, so the
    # identity fraction is meaningless and only the margin reveals that.
    low = rng.normal(size=(M, 3)) @ rng.normal(size=(3, D))
    _, _, m_deg = match_columns(low, low + 1e-3 * rng.normal(size=(M, D)))
    ok &= check("degenerate margin collapses", np.median(m_deg) < 0.05, f"median margin={np.median(m_deg):.4f}")
    ok &= check(
        "effective rank sees the true dimension", effective_rank(low) < 4.0, f"eff_rank={effective_rank(low):.2f}"
    )
    ok &= check("effective rank ~ d for full-rank J", effective_rank(A) > D * 0.6, f"eff_rank={effective_rank(A):.2f}")

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

    print("\nrank-deficient J: the spectrum must be truncated or it measures the noise floor")
    # One shared rank-2 mechanism at every site. The trailing 10 singular values are pure
    # float32-scale noise, independent per site — no site-sharing failure anywhere.
    # Build the spectrum explicitly: two dominant directions that are IDENTICAL everywhere,
    # then a decaying tail that is numerically resolved but physically irrelevant and whose
    # decay jitters per site. Equal weighting over all D log-values lets the 10 near-null
    # directions dominate the RMS, so the tail's jitter is reported as site heterogeneity.
    r_sites = sites_list(16)
    U = np.linalg.qr(rng.normal(size=(M, D)))[0]
    V = np.linalg.qr(rng.normal(size=(D, D)))[0]
    head = np.array([1.0, 0.7])
    # The tail shape is drawn per SITE and reused across subjects, which is what registered
    # anatomy produces: the same location looks alike in every subject. Site-consistent tail
    # structure lands in dev_across but not in the within-site floor, so it survives the
    # self-calibration and is reported as heterogeneity of a mechanism that never varied.
    site_tail = {u: np.exp(rng.normal(0, 0.6, D - 2)) for u in r_sites}
    deficient = []
    for _ in range(4):
        out = {}
        for u in r_sites:
            tail = 1e-3 * np.exp(-np.arange(D - 2)) * site_tail[u] * np.exp(rng.normal(0, 0.02, D - 2))
            J = (U * np.concatenate([head, tail])) @ V.T
            J *= 10.0 ** rng.uniform(-1, 1)
            out[u] = {"J": J, "energy_profile": {0: 0.9}, "rank_profile": {0: 2.0}}
        deficient.append(out)
    untrunc = reduce_arm(deficient, r_sites, 1e-3, 2, spectrum_energy=1.0)["homogeneity"]["ratio"]
    trunc = reduce_arm(deficient, r_sites, 1e-3, 2, spectrum_energy=0.99)
    ok &= check(
        "untruncated spectrum invents heterogeneity",
        untrunc > 5.0,
        f"ratio={untrunc:.1f} on data with ONE shared mechanism",
    )
    ok &= check(
        "truncation removes it",
        trunc["homogeneity"]["ratio"] < 3.0,
        f"ratio={trunc['homogeneity']['ratio']:.2f} keeping {trunc['spectrum_keep']}/{D} values",
    )

    print("\nblock geometry")
    grids = (((16, 16, 16), (64, 64, 64)), ((5, 7, 5), (91, 109, 91)), ((3, 3, 3), (8, 8, 8)))
    # Checked per axis, since block_slices treats the axes independently and a 3D count
    # multiplies the per-axis overlaps together (2 per axis reads as 8).
    for latent, out in grids:
        for a in range(3):
            cov = np.zeros(out[a], dtype=int)
            spans = []
            for u in range(latent[a]):
                site = [0, 0, 0]
                site[a] = u
                sl = block_slices(tuple(site), latent, out)[a]
                if sl.start < 0 or sl.stop > out[a]:
                    continue  # dropped by window_fits; not measured, so not required to cover
                cov[sl] += 1
                spans.append(sl)
            exact = out[a] % latent[a] == 0
            lo, hi = spans[0].start, spans[-1].stop
            inner = cov[lo:hi]
            # A fixed window size is what binding needs; at fractional stride ceil(stride)
            # exceeds the stride, so adjacent windows share a voxel at the seam. GAPS would
            # be a real defect — part of the volume belonging to no site — so those are what
            # is asserted against; an overlapping seam is the accepted cost of uniformity.
            good = inner.min() >= 1 and inner.max() <= (1 if exact else 2)
            ok &= check(
                f"{latent}->{out} axis {a}: {'tiles exactly' if exact else 'no gaps, seams overlap <= 1'}",
                good,
                f"min={inner.min()} max={inner.max()} over [{lo},{hi})",
            )

    # THE invariant the binding comparison depends on: every measured window is the same
    # size, so J_u and J_u0 can be matched column-by-column. A run crashed here when an
    # edge-clipped window came back 6400 rows against another site's 8000.
    print("\nevery fitting window has identical size (binding compares them directly)")
    for latent, out in grids:
        for dil in (0, 1, 2):
            sizes = set()
            for i in range(latent[0]):
                for j in range(latent[1]):
                    for k in range(latent[2]):
                        if window_fits((i, j, k), latent, out, dil):
                            sl = block_slices((i, j, k), latent, out, dil)
                            sizes.add(tuple(s.stop - s.start for s in sl))
            ok &= check(f"{latent}->{out} at +{dil}: one window size", len(sizes) <= 1, f"sizes={sorted(sizes)}")

    print("\nblock dilation and edge handling")
    vol = lambda sl: int(np.prod([s.stop - s.start for s in sl]))  # noqa: E731
    tight = block_slices((2, 2, 2), (8, 8, 8), (64, 64, 64), 0)
    wide = block_slices((2, 2, 2), (8, 8, 8), (64, 64, 64), 2)
    ok &= check("dilation grows the window", vol(wide) > vol(tight), f"{vol(tight)} -> {vol(wide)} voxels")
    ok &= check("a corner site does not fit when dilated", not window_fits((0, 0, 0), (8, 8, 8), (64, 64, 64), 2))
    ok &= check("an interior site does", window_fits((4, 4, 4), (8, 8, 8), (64, 64, 64), 2))
    clipped = block_slices((0, 0, 0), (8, 8, 8), (64, 64, 64), 4, clip=True)
    ok &= check("clip=True stays inside the volume", all(s.start >= 0 and s.stop <= 64 for s in clipped))

    print("\nverdict gating")

    def arm(ratio=1.4, bind=0.98, energy=0.9, rank_ratio=0.9, null=0.98, conf=0.9):
        return {
            "homogeneity": {"ratio": ratio, "true_site_sd": 0.01},
            "binding": {
                "primary": {
                    "identity_frac_median": bind,
                    "margin_median": 0.4,
                    "confident_frac_median": conf,
                    "identity_frac_confident": bind,
                },
                "same_site_null": null,
                "chance": 0.02,
            },
            "energy_profile": {"0": energy, "1": 0.95},
            "rank_profile": {"0": 48.0 * rank_ratio, "1": 48.0 * rank_ratio},
            "effective_rank": 40.0,
            "effective_rank_ratio": rank_ratio,
            "n_live_channels": 48,
        }

    v = build_verdict({"cli": {"block_dilation": 0}, "arms": {"full": arm(), "linear": arm()}}, 3.0, 0.9)
    ok &= check("clean run passes", v["ok"] and not v["blocked"])

    # The energy gate must read the dilation actually measured at. A run measured at +2 with
    # 68% captured is fine even though B(u) itself holds 3% — reading the "0" entry made the
    # gate fire on a window that was in fact wide enough.
    wide = arm()
    wide["energy_profile"] = {"0": 0.034, "2": 0.675}
    v = build_verdict({"cli": {"block_dilation": 2}, "arms": {"full": wide, "linear": arm()}}, 3.0, 0.9)
    ok &= check("energy gate reads the measured dilation", not v["blocked"])
    ok &= check("  ...and still reports poor localisation", any("B(u) itself holds" in ln for ln in v["lines"]))
    v = build_verdict({"cli": {"block_dilation": 0}, "arms": {"full": wide, "linear": arm()}}, 3.0, 0.9)
    ok &= check("same profile measured at +0 does block", v["blocked"])

    # Each gate must BLOCK interpretation rather than let a headline number through.
    for name, bad in (
        ("tight window", arm(energy=0.04)),
        ("degenerate matching", arm(rank_ratio=0.1)),
        ("noisy instrument", arm(null=0.35)),
    ):
        v = build_verdict({"cli": {"block_dilation": 0}, "arms": {"full": bad, "linear": arm()}}, 3.0, 0.9)
        ok &= check(f"{name} blocks the verdict", v["blocked"] and not v["ok"])
        ok &= check(f"  ...and reports no A1 conclusion", any("No conclusion about A1" in ln for ln in v["lines"]))

    v = build_verdict({"cli": {"block_dilation": 0}, "arms": {"full": arm(), "linear": arm(bind=0.4)}}, 3.0, 0.9)
    ok &= check("broken linear control blocks", v["blocked"] and any("CONTROL FAILED" in ln for ln in v["lines"]))

    # A rank that stays flat as the window grows is architectural; one that recovers is a
    # windowing artefact. The verdict must distinguish them — they need different responses.
    flat = arm(rank_ratio=0.05)
    flat["rank_profile"] = {"0": 2.2, "1": 2.3, "2": 2.4, "4": 2.4}
    v = build_verdict({"cli": {"block_dilation": 0}, "arms": {"full": flat, "linear": arm()}}, 3.0, 0.9)
    ok &= check("flat rank profile reads as architectural", any("LOCAL DECODING MAP" in ln for ln in v["lines"]))

    recovers = arm(rank_ratio=0.05)
    recovers["rank_profile"] = {"0": 2.2, "1": 12.0, "2": 31.0, "4": 40.0}
    v = build_verdict({"cli": {"block_dilation": 0}, "arms": {"full": recovers, "linear": arm()}}, 3.0, 0.9)
    ok &= check("recovering rank profile reads as windowing", any("windowing artefact" in ln for ln in v["lines"]))
    ok &= check("  ...and does not claim architecture", not any("LOCAL DECODING MAP" in ln for ln in v["lines"]))

    v = build_verdict({"cli": {"block_dilation": 0}, "arms": {"full": arm(bind=0.5), "linear": arm()}}, 3.0, 0.9)
    ok &= check("low binding fails once gates pass", not v["ok"] and not v["blocked"])
    v = build_verdict({"cli": {"block_dilation": 0}, "arms": {"full": arm(ratio=12.0), "linear": arm()}}, 3.0, 0.9)
    ok &= check("inhomogeneous spectra fail once gates pass", not v["ok"] and not v["blocked"])

    v = build_verdict(
        {"cli": {"block_dilation": 0}, "arms": {"full": arm(), "frozen_norm": arm()}, "freeze_was_noop": True}, 3.0, 0.9
    )
    ok &= check("freeze no-op is called out", any("NO-OP" in ln for ln in v["lines"]))

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
