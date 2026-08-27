#!/usr/bin/env python
"""Is a logged scalar RINGING, or just drifting noisily? Eyeballing cannot tell them apart.

Motivation: ``selection/*`` metrics are sampled every 2000 steps (``main_multimodal.py``
``step % 2000 == 1``, ``--dci-every``), so they can only resolve periods >= 4000 steps.
Anything faster aliases into what looks like random up-and-down jumping, with every factor
moving together because they share the sample grid. Meanwhile ``Contrastive/*`` is logged
every ``--log-steps`` (default 50) -- a 40x finer grid -- so a fast oscillation IS in the
logs, just not in the curves anyone reads. This script asks the 50-step series directly
whether a periodic component exists, and at what period.

The discriminator is the AUTOCORRELATION SIGN, not the periodogram. A periodogram always
has a peak somewhere, and on any drifting series that peak lands at the longest resolvable
period -- so "there is a spectral peak" is not evidence of anything. What separates the
two hypotheses is the shape of the ACF after detrending:

    ringing (damped oscillator)  ACF dips NEGATIVE at lag ~ period/2, then recovers.
                                 A negative lobe means "high now => low half a period
                                 later", which is what an overshooting feedback loop does
                                 and what a drift cannot do.
    drift / random walk          ACF decays monotonically toward 0 and stays >= 0.
    white noise                  ACF ~ 0 at every lag > 0.

So the statistic is ``min(ACF)`` over lags up to N/4, and the lag where it occurs gives
period ~ 2 x lag. Significance is against an AR(1) (red-noise) null fitted to the series
itself -- NOT against white noise, which every autocorrelated series beats trivially and
which would report ringing on a plain random walk.

Detrending matters and is not cosmetic. ``Contrastive/feat_std_mean_L0`` climbs
monotonically (nothing in the BT objective bounds feature scale: the variance hinge
``relu(1 - std)`` is one-sided), and an untreated trend dominates every low frequency and
drags the ACF positive at all lags -- masking a real negative lobe. Default is a rolling
median high-pass wide enough to leave the periods of interest untouched.

Usage:
    python -m eval.detect_ringing <run_dir_or_tb_logdir> --tag Contrastive/off_diag_inst_L0
    python -m eval.detect_ringing <logdir> --tag Contrastive/feat_std_mean_L0 --from 25000 --to 40000
    python -m eval.detect_ringing <logdir> --all-contrastive --from 25000
    python -m eval.detect_ringing <logdir> --tag X --detrend linear --max-period 4000
"""

import argparse

import numpy as np

from eval.find_jump_cause import _load_scalars


def _clean(steps, vals, lo=None, hi=None):
    """Dedupe by step (resumes replay steps), sort, window, drop non-finite."""
    ok = np.isfinite(vals)
    steps, vals = steps[ok], vals[ok]
    if lo is not None:
        m = steps >= lo
        steps, vals = steps[m], vals[m]
    if hi is not None:
        m = steps <= hi
        steps, vals = steps[m], vals[m]
    if len(steps) == 0:
        return steps, vals
    # Keep the LAST value at each step: a resume replays steps the checkpoint already
    # covered, and the later write is the one the continued run actually produced.
    order = np.argsort(steps, kind="stable")
    steps, vals = steps[order], vals[order]
    keep = np.ones(len(steps), dtype=bool)
    keep[:-1] = steps[:-1] != steps[1:]
    return steps[keep], vals[keep]


def _resample(steps, vals):
    """Put the series on a uniform step grid so lag == a fixed number of steps.

    Returns (grid_vals, dt). Gaps (a resume that lost some logging) are linearly
    interpolated; a series with a huge gap relative to its span is reported by the caller.
    """
    d = np.diff(steps)
    dt = float(np.median(d)) if len(d) else 1.0
    if dt <= 0:
        dt = 1.0
    grid = np.arange(steps[0], steps[-1] + dt * 0.5, dt)
    return np.interp(grid, steps, vals), dt


def _detrend(y, mode, win_pts):
    if mode == "none":
        return y
    if mode == "linear":
        x = np.arange(len(y), dtype=np.float64)
        return y - np.polyval(np.polyfit(x, y, 1), x)
    # rolling median high-pass: removes any slow trend (incl. exponential growth)
    # without assuming its shape, and leaves periods << win_pts untouched.
    w = max(3, int(win_pts) | 1)
    if w >= len(y):
        x = np.arange(len(y), dtype=np.float64)
        return y - np.polyval(np.polyfit(x, y, 1), x)
    pad = w // 2
    yp = np.pad(y, pad, mode="edge")
    med = np.array([np.median(yp[i : i + w]) for i in range(len(y))])
    return y - med


def _acf(y, max_lag):
    y = y - y.mean()
    denom = float(np.dot(y, y))
    if denom <= 0:
        return np.zeros(max_lag + 1)
    out = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        out[k] = float(np.dot(y[: len(y) - k], y[k:])) / denom if k < len(y) else 0.0
    return out


def _ar1_surrogates(y, n_surr, rng):
    """Red-noise null: AR(1) matched to the series' own lag-1 autocorrelation + variance."""
    y = y - y.mean()
    if len(y) < 3 or np.dot(y, y) <= 0:
        return np.zeros((n_surr, len(y)))
    phi = float(np.dot(y[:-1], y[1:]) / np.dot(y[:-1], y[:-1]))
    phi = float(np.clip(phi, -0.995, 0.995))
    sd = float(np.std(y) * np.sqrt(max(1.0 - phi**2, 1e-6)))
    out = np.empty((n_surr, len(y)))
    e = rng.standard_normal((n_surr, len(y))) * sd
    out[:, 0] = e[:, 0]
    for t in range(1, len(y)):
        out[:, t] = phi * out[:, t - 1] + e[:, t]
    return out


def analyse(steps, vals, detrend="rolling", max_period=None, n_surr=400, seed=0, min_lag_steps=None):
    steps = np.asarray(steps, float)
    vals = np.asarray(vals, float)
    if len(steps) < 32:
        return {"ok": False, "why": f"only {len(steps)} points after cleaning (need >= 32)"}

    y0, dt = _resample(steps, vals)
    span = steps[-1] - steps[0]
    # Longest period the ACF can support with a stable estimate.
    max_lag = max(4, len(y0) // 4)
    if max_period is not None:
        max_lag = min(max_lag, max(4, int(max_period / dt / 2)))

    # Detrend window: 4x the longest period we are willing to call an oscillation, so the
    # high-pass cannot itself create or erase a lobe inside the search range.
    win_pts = max(5, 4 * max_lag)
    y = _detrend(y0, detrend, win_pts)

    min_lag = 1
    if min_lag_steps:
        min_lag = max(1, int(min_lag_steps / dt))

    acf = _acf(y, max_lag)
    lags = np.arange(max_lag + 1)
    search = (lags >= min_lag) & (lags <= max_lag)
    if not search.any():
        return {"ok": False, "why": "search range empty; widen --max-period"}
    k_min = int(lags[search][np.argmin(acf[search])])
    acf_min = float(acf[k_min])

    rng = np.random.default_rng(seed)
    surr = _ar1_surrogates(y, n_surr, rng)
    null = np.array([float(np.min(_acf(s, max_lag)[search])) for s in surr])
    # One-sided: how often does red noise dip at least this far negative?
    p = float((null <= acf_min).mean())

    # Detection floor, so a NULL RESULT IS INFORMATIVE rather than just "nothing found".
    # Taking min over many lags carries a multiple-comparison penalty: even white noise
    # dips somewhere. The 5th percentile of the null IS that penalty, in ACF units. A pure
    # sinusoid carrying variance fraction f of the residual contributes ACF ~= -f at its
    # half-period, so an oscillation below this fraction cannot be separated from noise
    # here however real it is — widen the window (more points) or narrow --min/--max-period
    # (fewer lags searched) to lower it.
    floor = float(-np.percentile(null, 5))

    # Periodogram as a secondary read (reported, never used to decide).
    yz = y - y.mean()
    if len(yz) % 2:
        yz = yz[:-1]
    freqs = np.fft.rfftfreq(len(yz), d=dt)
    power = np.abs(np.fft.rfft(yz * np.hanning(len(yz)))) ** 2
    ok_f = freqs > 0
    peak_period = float(1.0 / freqs[ok_f][np.argmax(power[ok_f])]) if ok_f.any() else float("nan")

    return {
        "ok": True,
        "n": len(steps),
        "dt": dt,
        "span": span,
        "gap_max": float(np.max(np.diff(steps))) if len(steps) > 1 else 0.0,
        "acf_min": acf_min,
        "acf_min_lag_steps": k_min * dt,
        "period_steps": 2.0 * k_min * dt,
        "p_value": p,
        "detect_floor": floor,
        "acf1": float(acf[1]) if max_lag >= 1 else float("nan"),
        "peak_period_fft": peak_period,
        "resid_std": float(np.std(y)),
        "series_std": float(np.std(y0)),
        "trend_frac": float(1.0 - np.var(y) / max(np.var(y0), 1e-30)),
        "max_period_searched": 2.0 * max_lag * dt,
    }


def _verdict(r, alpha=0.05):
    if not r["ok"]:
        return "NO DATA", r["why"]
    if r["p_value"] <= alpha and r["acf_min"] < -0.1:
        return (
            "RINGING",
            f"ACF dips to {r['acf_min']:+.3f} at lag {r['acf_min_lag_steps']:.0f} steps "
            f"(p={r['p_value']:.3f} vs AR(1)) -> period ~{r['period_steps']:.0f} steps",
        )
    # DRIFT is read from the PRE-detrend series, not the residual: the high-pass removes
    # exactly the low-frequency content that makes a series look like drift, so a residual
    # test would call every trended series "white" and throw away the finding.
    if r["trend_frac"] > 0.5 or r["acf1"] > 0.5:
        return (
            "DRIFT",
            f"{r['trend_frac'] * 100:.0f}% of variance is trend and the residual ACF stays >= "
            f"{r['acf_min']:+.3f} (lag-1 {r['acf1']:+.2f}) — moves, but never overshoots. "
            f"Would have caught a ring carrying >{r['detect_floor'] * 100:.0f}% of residual variance.",
        )
    return (
        "NO RINGING",
        f"min ACF {r['acf_min']:+.3f} at lag {r['acf_min_lag_steps']:.0f} steps is within "
        f"AR(1) noise (p={r['p_value']:.3f}). Sensitive down to a ring carrying "
        f"{r['detect_floor'] * 100:.0f}% of residual variance — below that this test cannot see it.",
    )


def _report(tag, r, nyquist_note=True):
    verdict, why = _verdict(r)
    print(f"\n  {tag}")
    print(f"    VERDICT: {verdict} — {why}")
    if not r["ok"]:
        return
    print(
        f"    n={r['n']} points, {r['dt']:.0f}-step grid, span {r['span']:.0f} steps "
        f"(largest gap {r['gap_max']:.0f})"
    )
    print(
        f"    trend removed {r['trend_frac'] * 100:.0f}% of variance | "
        f"residual std {r['resid_std']:.4g} of series std {r['series_std']:.4g}"
    )
    print(
        f"    ACF: lag-1 {r['acf1']:+.3f}, min {r['acf_min']:+.3f} @ {r['acf_min_lag_steps']:.0f} steps | "
        f"FFT peak period {r['peak_period_fft']:.0f} steps | searched up to {r['max_period_searched']:.0f}"
    )
    print(
        f"    detection floor: a ring must carry >{r['detect_floor'] * 100:.0f}% of residual "
        f"variance to be separable from AR(1) noise at n={r['n']}"
    )
    if nyquist_note and r["ok"] and r["period_steps"] < 4000 and verdict == "RINGING":
        print(
            f"    NOTE: period {r['period_steps']:.0f} < 4000 steps, so a metric sampled every "
            f"2000 steps ALIASES this into apparent noise."
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logdir", help="Run directory or TensorBoard log directory (searched recursively)")
    ap.add_argument("--tag", action="append", default=[], help="Scalar tag to test (repeatable)")
    ap.add_argument(
        "--all-contrastive",
        action="store_true",
        help="Test every Contrastive/* and Codebook/* tag — the fine-grained (log_steps) series",
    )
    ap.add_argument("--from", dest="lo", type=float, default=None, help="First step to include")
    ap.add_argument("--to", dest="hi", type=float, default=None, help="Last step to include")
    ap.add_argument(
        "--detrend",
        choices=("rolling", "linear", "none"),
        default="rolling",
        help="Trend removal before the ACF. 'rolling' (default) is a median high-pass that "
        "handles the monotone climb in feat_std_mean without assuming a shape.",
    )
    ap.add_argument(
        "--max-period",
        type=float,
        default=None,
        help="Longest period (in steps) to consider an oscillation. Default: span/2.",
    )
    ap.add_argument(
        "--min-period",
        type=float,
        default=None,
        help="Shortest period to consider, to skip single-sample sawtooth from the logger itself.",
    )
    ap.add_argument("--n-surrogates", type=int, default=400, help="AR(1) surrogates for the null (default 400)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    series = _load_scalars(args.logdir)
    print(f"Loaded {len(series)} scalars from {args.logdir}")

    tags = list(args.tag)
    if args.all_contrastive:
        tags += sorted(t for t in series if t.startswith(("Contrastive/", "Codebook/")))
    if not tags:
        tags = [t for t in ("Contrastive/off_diag_inst_L0", "Contrastive/feat_std_mean_L0") if t in series]
        if not tags:
            cand = [t for t in series if t.startswith("Contrastive/")]
            raise SystemExit(f"No --tag given and no defaults present. Contrastive tags: {cand[:12]}")

    seen = set()
    tags = [t for t in tags if not (t in seen or seen.add(t))]
    missing = [t for t in tags if t not in series]
    for t in missing:
        print(f"  [skip] {t!r} not in this run")
    tags = [t for t in tags if t in series]

    print(
        "\nTesting for a periodic component. The statistic is the most NEGATIVE autocorrelation\n"
        "over the searched lags, against an AR(1) null fitted to the series itself — a drifting\n"
        "or random-walk series has an ACF that decays but never overshoots, an oscillator's does."
    )

    rows = []
    for tag in tags:
        s, v = series[tag]
        s, v = _clean(np.asarray(s, float), np.asarray(v, float), args.lo, args.hi)
        r = analyse(
            s,
            v,
            detrend=args.detrend,
            max_period=args.max_period,
            n_surr=args.n_surrogates,
            seed=args.seed,
            min_lag_steps=(args.min_period / 2) if args.min_period else None,
        )
        _report(tag, r)
        rows.append((tag, r))

    ring = [(t, r) for t, r in rows if r["ok"] and _verdict(r)[0] == "RINGING"]
    if ring:
        print("\n  RINGING, by period:")
        for t, r in sorted(ring, key=lambda tr: tr[1]["period_steps"]):
            print(f"    {r['period_steps']:>8.0f} steps  (p={r['p_value']:.3f})  {t}")
    else:
        print("\n  No tag showed a periodic component beyond red noise.")
        print("  That rules out a limit cycle at these tags — the movement is drift, not ringing.")


if __name__ == "__main__":
    main()
