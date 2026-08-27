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

``--msd`` answers the OTHER question, and it is the one that matters once ringing is ruled
out: a metric that jumps around is either sitting in a bounded noise ball (the model is
stable, the swings are its jitter and go nowhere) or DIFFUSING (the model is wandering, and
each swing is a step of a random walk that does not come back). Those look identical on a
plot and call for opposite responses. Mean-squared displacement separates them:

    MSD(delta) = mean over pairs of (y(t+delta) - y(t))^2,  classified by its GROWTH exponent

    slope ~ 0   BOUNDED JITTER — stationary. MSD plateaus at 2*sigma^2, so the swings have a
                fixed amplitude and the representation is not going anywhere. The reported
                implied per-sample std is directly comparable to the logged
                selection/mcc_by_factor_std/* — if they match, the swings are probe noise;
                if the implied std is much larger, the MODEL is jittering.
    slope ~ 1   DIFFUSION — random walk. The encoder is wandering in parameter space, driven
                by gradient noise. This is what a noise-dominated gradient produces once the
                deterministic signal is exhausted, and the swings do NOT return.
    slope > 1   DRIFT — faster than a random walk, i.e. a systematic direction.

Two things this gets right that a naive version does not, both found by testing it against
series whose answer was known:

  * The verdict comes from the growth exponent, NOT from the a/b/c decomposition of
    a*delta^2 + b*delta + c. Those shares depend entirely on where you evaluate them: at
    delta = one sampling interval the constant term dominates for every series by
    construction, which reads a random walk as bounded. The shares are reported as detail;
    the exponent decides.

  * Thresholds are calibrated by SIMULATION at the observed series length, not from a
    regression standard error. MSD values at different lags come from heavily overlapping
    windows on one realization, so they are strongly dependent and an analytic SE understates
    the spread badly — at n=18 that made a genuine random walk report BOUNDED JITTER with
    apparent confidence. With simulated bands the same case correctly reports INCONCLUSIVE.

That last point is a practical limit worth knowing before running this: the 2000-step
selection cadence gives ~18 points over a 35k-step window, and at that length the bounded
and diffusion bands OVERLAP — the test cannot separate them however the run behaves. Point
it at a fine-grained (log_steps) proxy for the representation instead, e.g.
Contrastive/pos_sim_mean_L0 or Contrastive/feat_std_mean_L0, which have ~40x the points.

Usage:
    python -m eval.detect_ringing <run_dir_or_tb_logdir> --tag Contrastive/off_diag_inst_L0
    python -m eval.detect_ringing <logdir> --tag Contrastive/feat_std_mean_L0 --from 25000 --to 40000
    python -m eval.detect_ringing <logdir> --all-contrastive --from 25000
    python -m eval.detect_ringing <logdir> --tag X --detrend linear --max-period 4000
    python -m eval.detect_ringing <logdir> --msd --tag selection/mcc_by_pool/patch --from 31000
    python -m eval.detect_ringing <logdir> --msd --all-selection --from 31000
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


def _msd_slope(y, max_lag):
    """Pair-count-weighted log-log slope of MSD(delta) for one series. NaN if undefined."""
    n = len(y)
    lags = np.arange(1, min(max_lag, n - 2) + 1)
    if len(lags) < 4:
        return float("nan")
    msd = np.array([float(np.mean((y[k:] - y[:-k]) ** 2)) for k in lags])
    pos = msd > 0
    if pos.sum() < 4:
        return float("nan")
    x = np.log(lags[pos].astype(float))
    yv = np.log(msd[pos])
    w = (n - lags[pos]).astype(float)
    w = w / w.mean()
    xb = np.sum(w * x) / np.sum(w)
    yb = np.sum(w * yv) / np.sum(w)
    sxx = float(np.sum(w * (x - xb) ** 2))
    if sxx <= 0:
        return float("nan")
    return float(np.sum(w * (x - xb) * (yv - yb)) / sxx)


def _simulate_slopes(kind, n, max_lag, n_sim, rng, y=None):
    """Distribution of the fitted MSD slope for a known regime at THIS series length.

    ``bounded`` is matched to the observed series' own lag-1 autocorrelation, so a smooth
    stationary process is not mistaken for diffusion purely because it is autocorrelated.
    """
    out = np.empty(n_sim)
    if kind == "bounded":
        phi = 0.0
        if y is not None and len(y) > 2:
            yc = y - y.mean()
            den = float(np.dot(yc[:-1], yc[:-1]))
            if den > 0:
                phi = float(np.clip(np.dot(yc[:-1], yc[1:]) / den, -0.95, 0.95))
        e = rng.standard_normal((n_sim, n))
        s = np.empty((n_sim, n))
        s[:, 0] = e[:, 0]
        for t in range(1, n):
            s[:, t] = phi * s[:, t - 1] + e[:, t] * np.sqrt(max(1 - phi**2, 1e-6))
    elif kind == "diffusion":
        s = np.cumsum(rng.standard_normal((n_sim, n)), axis=1)
    else:
        raise ValueError(kind)
    for i in range(n_sim):
        out[i] = _msd_slope(s[i], max_lag)
    return out[np.isfinite(out)]


def analyse_msd(steps, vals, max_lag_frac=0.34, at_lag=None, n_sim=300, seed=0):
    """Mean-squared displacement: is the movement bounded, diffusive, or ballistic?

    Fits MSD(delta) = a*delta^2 + b*delta + c under a >= 0, b >= 0, c >= 0 (the three terms
    are variances and cannot be negative; an unconstrained fit happily returns a negative
    diffusion coefficient and reads as nonsense). Those shares are reported at ``at_lag``
    (default: the largest fitted lag) as supporting detail only — the VERDICT comes from the
    log-log growth exponent, which does not depend on where you choose to evaluate it.
    """
    from scipy.optimize import nnls

    steps = np.asarray(steps, float)
    vals = np.asarray(vals, float)
    if len(steps) < 8:
        return {"ok": False, "why": f"only {len(steps)} points after cleaning (need >= 8)"}

    y, dt = _resample(steps, vals)
    n = len(y)
    max_lag = max(2, int(n * max_lag_frac))
    lags = np.arange(1, max_lag + 1)
    msd = np.array([float(np.mean((y[k:] - y[:-k]) ** 2)) for k in lags])
    pairs = n - lags
    d = lags * dt

    # Weight by pair count: MSD at large lag is estimated from few pairs and is far noisier.
    w = np.sqrt(pairs.astype(float))
    A = np.column_stack([d**2, d, np.ones_like(d)]) * w[:, None]
    coef, _ = nnls(A, msd * w)
    a, b, c = (float(x) for x in coef)

    # Split the fitted MSD at the LARGEST lag, not the smallest: at delta = one sampling
    # interval the constant term dominates by construction for every series (a*delta^2 and
    # b*delta are smallest there), which reads every curve as bounded whatever it is.
    at = float(at_lag) if at_lag else float(max_lag * dt)
    parts = np.array([a * at**2, b * at, c])
    tot = float(parts.sum())
    shares = parts / tot if tot > 0 else np.zeros(3)

    # The actual discriminator: how fast MSD GROWS. slope 0 = bounded, 1 = diffusion,
    # 2 = ballistic drift.
    slope = _msd_slope(y, max_lag)

    # Thresholds are CALIBRATED BY SIMULATION at this n, not taken from an analytic standard
    # error. MSD values at different lags come from heavily OVERLAPPING windows on a single
    # realization, so they are strongly dependent and a regression SE understates the spread
    # by a wide margin — at n=18 that made a genuine random walk report BOUNDED JITTER with
    # apparent confidence, which is the worst failure this tool could have. Simulating each
    # regime at the observed length gives the real overlap, so a length that cannot separate
    # them says so instead.
    rng = np.random.default_rng(int(seed))
    nsim = int(n_sim)
    null_bounded = _simulate_slopes("bounded", n, max_lag, nsim, rng, y=y)
    null_diffusion = _simulate_slopes("diffusion", n, max_lag, nsim, rng)

    return {
        "ok": True,
        "n": len(steps),
        "dt": dt,
        "span": steps[-1] - steps[0],
        "max_lag_steps": float(max_lag * dt),
        "min_pairs": int(pairs.min()),
        "a": a,
        "b": b,
        "c": c,
        "at_lag": at,
        "share_drift": float(shares[0]),
        "share_diffusion": float(shares[1]),
        "share_bounded": float(shares[2]),
        "msd_at": tot,
        "implied_step_std": float(np.sqrt(tot)) if tot > 0 else 0.0,
        "implied_bounded_std": float(np.sqrt(c / 2.0)) if c > 0 else 0.0,
        "loglog_slope": slope,
        "null_bounded": null_bounded,
        "null_diffusion": null_diffusion,
        "step_std": float(np.sqrt(np.mean((y[1:] - y[:-1]) ** 2))),
        "series_std": float(np.std(y)),
    }


def _verdict_msd(r):
    """Classify from the log-log MSD slope and its CI, not from the a/b/c shares.

    The shares depend on where you evaluate them; the growth exponent does not. A CI that
    spans two categories is reported as INCONCLUSIVE rather than resolved to the nearest
    one — at the 2000-step selection cadence there are often too few points to tell a
    random walk from bounded jitter, and saying so is the useful answer.
    """
    if not r["ok"]:
        return "NO DATA", r["why"]
    s = r["loglog_slope"]
    if not np.isfinite(s):
        return "INCONCLUSIVE", "not enough usable lags to fit a growth exponent"

    # Only two regimes are simulated. The question that matters is BOUNDED vs UNBOUNDED —
    # does the representation come back, or is it going somewhere — and a drift null would
    # need an arbitrary choice of noise level that decides the answer by itself. Anything
    # growing faster than a random walk is unambiguously going somewhere, so super-diffusive
    # is read off the diffusion band's upper edge rather than simulated separately.
    nb, nd = r["null_bounded"], r["null_diffusion"]
    if len(nb) < 20 or len(nd) < 20:
        return "INCONCLUSIVE", f"slope {s:+.2f}; too few usable simulations to calibrate at n={r['n']}"
    b_lo, b_hi = (float(x) for x in np.percentile(nb, [5, 95]))
    d_lo, d_hi = (float(x) for x in np.percentile(nd, [5, 95]))
    head = f"slope {s:+.2f} vs simulated bands at n={r['n']}: bounded[{b_lo:+.2f},{b_hi:+.2f}] diffusion[{d_lo:+.2f},{d_hi:+.2f}]"

    in_b, in_d = b_lo <= s <= b_hi, d_lo <= s <= d_hi
    if s > d_hi:
        return "DRIFT", f"{head}. Faster than a random walk — a systematic direction, not noise."
    if in_b and not in_d:
        return "BOUNDED JITTER", f"{head}. The swings have a fixed amplitude and go nowhere."
    if in_d and not in_b:
        return "DIFFUSION", f"{head}. A random walk: these swings do NOT return."
    if in_b and in_d:
        return (
            "INCONCLUSIVE",
            f"{head}. The bands OVERLAP at n={r['n']} — this many points cannot tell bounded "
            f"jitter from a random walk. Widen --from/--to, or run this on a fine-grained "
            f"(log_steps) proxy such as Contrastive/pos_sim_mean_L0.",
        )
    if s < b_lo:
        # Growing SLOWER than a stationary process means the MSD turns over: the series
        # comes back toward where it was. That is mean reversion, and an oscillation is the
        # common cause — which is the other mode's question, not this one's.
        return (
            "MEAN-REVERTING",
            f"{head}. MSD grows SLOWER than stationary noise, i.e. it turns over — the series "
            f"returns toward earlier values. Run without --msd to test for a periodic component.",
        )
    return "INCONCLUSIVE", f"{head}. Slope falls outside both bands — inspect the curve directly."


def _report_msd(tag, r):
    verdict, why = _verdict_msd(r)
    print(f"\n  {tag}")
    print(f"    VERDICT: {verdict} — {why}")
    if not r["ok"]:
        return
    print(
        f"    n={r['n']} points, {r['dt']:.0f}-step grid, span {r['span']:.0f} steps | "
        f"lags to {r['max_lag_steps']:.0f} steps (min {r['min_pairs']} pairs)"
    )
    print(f"    RMS move between consecutive samples: {r['step_std']:.4g}  " f"(series std {r['series_std']:.4g})")
    print(
        f"    at delta={r['at_lag']:.0f} steps the fitted MSD splits "
        f"{r['share_bounded'] * 100:.0f}% bounded / {r['share_diffusion'] * 100:.0f}% diffusion / "
        f"{r['share_drift'] * 100:.0f}% drift"
    )
    print(
        f"    bounded component implies a per-sample std of {r['implied_bounded_std']:.4g} — "
        f"compare against the logged selection/mcc_by_factor_std/* to tell model jitter from probe noise"
    )
    if verdict == "DIFFUSION":
        print(
            "    NOTE: linear MSD growth means these swings do NOT return — the representation "
            "is wandering, not vibrating around a fixed point."
        )
    elif verdict == "BOUNDED JITTER":
        print(
            "    NOTE: the MSD plateaus, so the swings have a fixed amplitude and the model is "
            "NOT drifting away. Shrink the noise ball (lower LR / lower loss weight / more "
            "subjects per step) rather than hunting a trend."
        )


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
    ap.add_argument(
        "--all-selection",
        action="store_true",
        help="Test every selection/* and dci_synthetic/* tag — the coarse (2000-step) series. "
        "Most useful with --msd, which needs no resolution beyond the sampling interval.",
    )
    ap.add_argument(
        "--msd",
        action="store_true",
        help="Mean-squared-displacement mode: classify the movement as BOUNDED JITTER (stationary, "
        "swings go nowhere), DIFFUSION (random walk, swings do not return) or DRIFT (systematic). "
        "Use this once --detect-ringing has come back negative.",
    )
    ap.add_argument(
        "--at-lag",
        type=float,
        default=None,
        help="Step gap at which to split the MSD into its three components. Default: the series' "
        "own sampling interval, i.e. the gap between the consecutive evals you are eyeballing.",
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
    if args.all_selection:
        tags += sorted(t for t in series if t.startswith(("selection/", "dci_synthetic/")))
    if not tags:
        _defaults = (
            ("selection/mcc_by_pool/patch", "selection/mcc_cc_gap", "selection/encoder_l2")
            if args.msd
            else ("Contrastive/off_diag_inst_L0", "Contrastive/feat_std_mean_L0")
        )
        tags = [t for t in _defaults if t in series]
        if not tags:
            _pre = "selection/" if args.msd else "Contrastive/"
            cand = [t for t in series if t.startswith(_pre)]
            raise SystemExit(f"No --tag given and no defaults present. {_pre}* tags: {cand[:12]}")

    seen = set()
    tags = [t for t in tags if not (t in seen or seen.add(t))]
    missing = [t for t in tags if t not in series]
    for t in missing:
        print(f"  [skip] {t!r} not in this run")
    tags = [t for t in tags if t in series]

    if args.msd:
        print(
            "\nMean-squared displacement, classified by how fast it GROWS: flat = bounded jitter\n"
            "that goes nowhere, linear = diffusion (a random walk whose swings do NOT return),\n"
            "faster = systematic drift. Ruling out ringing does not distinguish these, and they\n"
            "call for opposite responses. Bands are simulated at each series' own length, so a\n"
            "series with too few points reports INCONCLUSIVE instead of guessing."
        )
        rows = []
        for tag in tags:
            s, v = series[tag]
            s, v = _clean(np.asarray(s, float), np.asarray(v, float), args.lo, args.hi)
            r = analyse_msd(s, v, at_lag=args.at_lag)
            _report_msd(tag, r)
            rows.append((tag, r))
        by_verdict = {}
        for t, r in rows:
            by_verdict.setdefault(_verdict_msd(r)[0], []).append(t)
        print("\n  Summary:")
        for k in ("DIFFUSION", "DRIFT", "MIXED", "BOUNDED JITTER", "NO DATA"):
            if k in by_verdict:
                print(f"    {k:>15}: {len(by_verdict[k])}  {', '.join(by_verdict[k][:6])}")
        return

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
        print("  Not a limit cycle at these tags. Re-run with --msd to tell bounded jitter")
        print("  (swings that go nowhere) from diffusion (swings that do not return).")


if __name__ == "__main__":
    main()
