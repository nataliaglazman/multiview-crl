#!/usr/bin/env python
"""Smoke test for --bt-gap-pool: the weighted position pooling behind the BT GAP term.

Checks the properties the pooling has to have for the GAP term to keep meaning what it
meant, and then measures whether weighting actually buys anything on planted data:

  1. 'learned' at init IS the uniform mean (zero logits -> softmax uniform), so enabling
     the flag does not perturb a resumed run at step 0.
  2. Weights sum to 1, are shared across views, and do not depend on the sample.
  3. The foreground keep-mask is honoured: weights are indexed by FULL-grid position and
     sliced afterwards, and a mask/feature mismatch raises rather than silently
     misaligning (the failure mode --patch-foreground-mask invites, since it recomputes
     the kept set every batch).
  4. The variance EMA is per-position and bias-corrected, so a position dropped for some
     batches is not treated as if it had been observed every step.
  5. The entropy hinge is zero while the weights are diffuse and positive once they
     concentrate past the floor.
  6. THE POINT: on a planted signal where a focal subject factor lives in a few positions
     and the rest carry within-subject noise, variance pooling raises the across-subject
     std of the pooled feature over the uniform mean. This is the dilution hypothesis
     that motivates the whole flag — if it fails here it will fail on real data.

Usage:
    python -m scripts.smoke_bt_gap_pool
"""

from __future__ import annotations

import math

import numpy as np
import torch

from models.vqvae import GapPositionPool

torch.manual_seed(0)

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def planted(n_subj=128, n_ch=8, grid=64, n_focal=4, focal_gain=1.0, noise=1.0):
    """(2, B, C, P) where a subject factor lives in n_focal positions only.

    Both views see the same subject factor (that is what content means here) plus
    independent within-subject spatial noise, which is the interaction term r that
    dominates the patch fold on registered volumes.
    """
    s = torch.randn(n_subj, n_ch) * focal_gain  # per-subject factor
    z = torch.randn(2, n_subj, n_ch, grid) * noise  # interaction, independent per view
    z[..., :n_focal] += s.unsqueeze(0).unsqueeze(-1)  # focal positions carry the subject
    return z


print("\n1. 'learned' at init is exactly the uniform mean")
p = GapPositionPool(64, mode="learned")
z = torch.randn(2, 32, 8, 64)
out, a = p(z, None)
check("pooled == z.mean(-1)", torch.allclose(out, z.mean(-1), atol=1e-6))
check("weights uniform", torch.allclose(a, torch.full((64,), 1 / 64), atol=1e-7))
check("weights sum to 1", abs(a.sum().item() - 1.0) < 1e-6)

print("\n2. weights are shared across views and independent of the sample")
pv = GapPositionPool(64, mode="variance", ema=0.0)
z = planted()
a_full = pv.weights(z, None)
a_view0 = pv.weights(z[:1], None)
check("one weight vector, not one per view", a_full.shape == (64,))
# Halving the batch changes the ESTIMATE but the weights still apply identically to
# every sample and view — that is what "independent of the sample" means here.
out, a = pv(z, None)
manual = (z * a).sum(-1)
check("pooling is a plain weighted sum with the same a for all n, v", torch.allclose(out, manual, atol=1e-6))
check("view-0-only estimate is still a single (P,) vector", a_view0.shape == (64,))

print("\n3. foreground keep-mask is honoured and mismatches raise")
p = GapPositionPool(64, mode="learned")
with torch.no_grad():
    p.logits[:8] = 5.0  # concentrate on the first 8 FULL-grid positions
keep = torch.zeros(64, dtype=torch.bool)
keep[4:12] = True  # keeps 4..11, i.e. half of the heavy ones
z_kept = torch.randn(2, 16, 8, int(keep.sum()))
a_kept = p.weights(z_kept, keep)
check("weights sliced to kept count", a_kept.numel() == 8)
check("weights renormalised over survivors", abs(a_kept.sum().item() - 1.0) < 1e-6)
# Positions 4..7 had logit 5, positions 8..11 had 0 — the heavy mass must land on the
# first four of the kept set, which is what "indexed by full-grid position" buys.
check("heavy mass follows FULL-grid position", bool((a_kept[:4] > a_kept[4:]).all()))
try:
    p.weights(torch.randn(2, 16, 8, 7), keep)  # 7 features vs 8 kept
    check("mismatch raises", False, "no exception")
except ValueError as e:
    check("mismatch raises", "disagree about the patch grid" in str(e))
try:
    p.weights(torch.randn(2, 16, 8, 64), torch.zeros(32, dtype=torch.bool))
    check("wrong-size mask raises", False, "no exception")
except ValueError as e:
    check("wrong-size mask raises", "FULL patch grid" in str(e))
try:
    p.weights(torch.randn(2, 16, 8, 64), torch.arange(64))
    check("non-bool mask raises", False, "no exception")
except TypeError as e:
    check("non-bool mask raises", "boolean keep mask" in str(e))

print("\n4. variance EMA is per-position and bias-corrected")
pv = GapPositionPool(16, mode="variance", ema=0.9)
z = planted(n_subj=64, n_ch=4, grid=16, n_focal=2)
keep_a = torch.ones(16, dtype=torch.bool)
keep_b = torch.zeros(16, dtype=torch.bool)
keep_b[:8] = True
pv.weights(z, keep_a)
pv.weights(z[..., :8], keep_b)
pv.weights(z[..., :8], keep_b)
t = pv.var_ema_t
check("kept-every-step positions counted 3x", bool((t[:8] == 3).all()), f"t[:8]={t[:8].tolist()}")
check("positions dropped twice counted 1x", bool((t[8:] == 1).all()), f"t[8:]={t[8:].tolist()}")
# Bias correction: after ONE observation the corrected value must equal that observation,
# not (1-m) times it — otherwise a freshly-seen position is silently down-weighted ~10x.
pv1 = GapPositionPool(8, mode="variance", ema=0.9)
zz = planted(n_subj=64, n_ch=4, grid=8, n_focal=2)
a1 = pv1.weights(zz, None)
pv0 = GapPositionPool(8, mode="variance", ema=0.0)
a0 = pv0.weights(zz, None)
check("1 step of EMA == no EMA (bias corrected)", torch.allclose(a1, a0, atol=1e-5))

print("\n5. entropy hinge: silent while diffuse, active once concentrated")
p = GapPositionPool(64, mode="learned", entropy_floor=0.5)
a_uniform = p.weights(torch.randn(2, 8, 4, 64), None)
pen_uniform = p.entropy_penalty(a_uniform)
check("no penalty at uniform", pen_uniform.item() == 0.0, f"{pen_uniform.item():.4f}")
with torch.no_grad():
    p.logits.zero_()
    p.logits[0] = 20.0  # collapse onto one position
a_collapsed = p.weights(torch.randn(2, 8, 4, 64), None)
pen_collapsed = p.entropy_penalty(a_collapsed)
check("penalty once past the floor", pen_collapsed.item() > 0, f"{pen_collapsed.item():.4f}")
check(
    "penalty is the hinge value",
    abs(pen_collapsed.item() - 0.5 * math.log(64)) < 0.05,
    f"hinge={pen_collapsed.item():.4f} vs floor={0.5 * math.log(64):.4f}",
)
# Deliberately NOT asserting a gradient here: this state is fully saturated, where the
# entropy gradient vanishes by construction. Section 9 tests the gradient in the band
# where the hinge actually operates.
d = GapPositionPool.diagnostics(a_uniform)
check("diagnostics: uniform reads entropy 1.0", abs(d["gap_pool_entropy"] - 1.0) < 1e-5, str(d))
d = GapPositionPool.diagnostics(a_collapsed)
check("diagnostics: collapsed reads eff_pos ~1", d["gap_pool_eff_pos"] < 1.1, str(d))

print("\n6. does weighting actually beat the uniform mean on planted focal signal?")
# 4 focal positions out of 256: the regime the flag exists for.
z = planted(n_subj=256, n_ch=16, grid=256, n_focal=4, focal_gain=1.0, noise=1.0)
uniform = z.mean(-1)  # (2, B, C)
pv = GapPositionPool(256, mode="variance", ema=0.0)
weighted, a = pv(z, None)
# Across-SUBJECT std of the pooled feature: the quantity gap_feat_std reports and the
# one the variance hinge is trying to lift off ~0.004 on real runs.
std_u = uniform.std(dim=1, unbiased=False).mean().item()
std_w = weighted.std(dim=1, unbiased=False).mean().item()


# Cross-view correlation of the pooled feature per channel: how much of the pooled
# signal is the shared subject factor rather than each view's own interaction noise.
def xcorr(x):
    a_, b_ = x[0], x[1]
    a_ = (a_ - a_.mean(0)) / (a_.std(0, unbiased=False) + 1e-8)
    b_ = (b_ - b_.mean(0)) / (b_.std(0, unbiased=False) + 1e-8)
    return (a_ * b_).mean().item()


corr_u, corr_w = xcorr(uniform), xcorr(weighted)
print(f"     uniform : across-subject std {std_u:.4f}   cross-view corr {corr_u:.4f}")
print(f"     variance: across-subject std {std_w:.4f}   cross-view corr {corr_w:.4f}")
print(f"     weight mass on the 4 focal positions: {a[:4].sum().item():.4f} (uniform would be {4/256:.4f})")
check("variance pooling concentrates on the focal positions", a[:4].sum().item() > 4 / 256)
check("subject signal survives pooling better", corr_w > corr_u, f"{corr_w:.4f} vs {corr_u:.4f}")

print("\n7. the flags parse and survive update_args")
from utils.config import parse_args, update_args  # noqa: E402

_base = [
    "--dataset-name", "synthetic", "--contrastive-loss-type", "barlow_twins",
    "--patch-contrastive", "--patch-grid", "4", "4", "4", "--vqvae-nb-levels", "1",
    "--content-style-levels", "0", "--bt-gap-weight", "1",
]  # fmt: skip
a = update_args(parse_args().parse_args(_base + ["--bt-gap-pool", "variance", "--bt-gap-pool-ema", "0.95"]))
check("variance flags parse", a.bt_gap_pool == "variance" and a.bt_gap_pool_ema == 0.95)
a = update_args(parse_args().parse_args(_base + ["--bt-gap-pool", "learned", "--bt-gap-pool-entropy", "0.3"]))
check("learned flags parse", a.bt_gap_pool == "learned" and a.bt_gap_pool_entropy == 0.3)
a = update_args(parse_args().parse_args(_base))
check("default is 'mean' (existing runs untouched)", a.bt_gap_pool == "mean")

print("\n8. Barlow Twins consumes the pooled tensor and reports the GAP diagnostics")
from training.losses import barlow_twins_loss  # noqa: E402

z = planted(n_subj=64, n_ch=12, grid=64, n_focal=4)
pool = GapPositionPool(64, mode="learned", entropy_floor=0.5)
gap_in, a = pool(z, None)
check("pooled shape is (n_views, B, C)", gap_in.shape == (2, 64, 12))
ci = [list(range(12))]
loss_pooled = barlow_twins_loss(gap_in, estimated_content_indices=ci, subsets=[[0, 1]], lambd=0.013, std_coeff=0.5)
loss_mean = barlow_twins_loss(z.mean(-1), estimated_content_indices=ci, subsets=[[0, 1]], lambd=0.013, std_coeff=0.5)
check("uniform-init pooling reproduces the mean-pooled loss", torch.allclose(loss_pooled, loss_mean, atol=1e-5))
d = loss_pooled._contrastive_diag
check("GAP diag carries the terms the weighted accounting reads",
      all(k in d for k in ("on_diag_loss", "off_diag_loss", "sim_loss", "var_loss", "feat_std_mean")),
      str(sorted(d))[:110])  # fmt: skip

print("\n9. the entropy hinge reaches the encoder-facing graph, not just the logits")


def hinge_grad_at(peak, floor=0.5, grid=64):
    """Gradient the entropy hinge puts on the heavy logit at a given concentration."""
    p = GapPositionPool(grid, mode="learned", entropy_floor=floor)
    with torch.no_grad():
        p.logits.zero_()
        p.logits[0] = peak
    a_ = p.weights(torch.randn(2, 8, 4, grid), None)
    pen_ = p.entropy_penalty(a_)
    h_ = -(a_ * torch.log(a_ + 1e-12)).sum().item() / math.log(grid)
    if pen_.requires_grad:
        pen_.backward()
    g_ = p.logits.grad[0].item() if p.logits.grad is not None else 0.0
    return h_, pen_.item(), g_


# What patch_loss_func assembles: gap term + coeff * hinge, with the hinge gradient
# landing on the pooling logits and the BT gradient on the features. Tested at a
# MODERATE concentration — the band the hinge exists to police — because a fully
# saturated softmax has a vanishing entropy gradient (see entropy_penalty's docstring),
# so asserting "grad > 0" there passes on a value of 3e-6 and proves nothing.
z = planted(n_subj=64, n_ch=12, grid=64, n_focal=4).requires_grad_(True)
pool = GapPositionPool(64, mode="learned", entropy_floor=0.5)
with torch.no_grad():
    pool.logits[0] = 6.0  # H/logP ~ 0.23: past the floor, softmax not yet saturated
gap_in, a = pool(z, None)
total = barlow_twins_loss(gap_in, estimated_content_indices=ci, subsets=[[0, 1]], lambd=0.013)
pen = pool.entropy_penalty(a)
total = total + 1.0 * pen
total.backward()
check("features receive gradient", z.grad is not None and torch.isfinite(z.grad).all())
check("logits receive gradient", pool.logits.grad is not None and pool.logits.grad.abs().sum() > 0)
check("hinge pushes the concentrated logit DOWN with real force", pool.logits.grad[0].item() > 0.1,
      f"grad[0]={pool.logits.grad[0].item():.4f}")  # fmt: skip

# Pin the saturation behaviour rather than let it pass silently: it is a real limit of
# any smooth penalty on a softmax, and the point of pinning it is that a future change
# claiming to fix collapse has to move THIS number.
h_lo, pen_lo, g_lo = hinge_grad_at(4.0)
h_mid, pen_mid, g_mid = hinge_grad_at(6.0)
h_hi, pen_hi, g_hi = hinge_grad_at(20.0)
print(f"     peak  4: H/logP {h_lo:.3f}  hinge {pen_lo:.3f}  grad {g_lo:.6f}")
print(f"     peak  6: H/logP {h_mid:.3f}  hinge {pen_mid:.3f}  grad {g_mid:.6f}")
print(f"     peak 20: H/logP {h_hi:.3f}  hinge {pen_hi:.3f}  grad {g_hi:.6f}")
check("silent above the floor", pen_lo == 0.0 and g_lo == 0.0)
check("strong in the active band", g_mid > 0.1)
check("KNOWN: vanishes once the softmax saturates", g_hi < 1e-4,
      "documented absorbing state — watch gap_pool_entropy, it cannot self-recover")  # fmt: skip

print("\n10. variance mode is detached; learned mode is not")
pv = GapPositionPool(32, mode="variance", ema=0.0)
zz = planted(n_subj=32, n_ch=4, grid=32, n_focal=2)
check("variance weights carry no grad_fn", pv.weights(zz, None).grad_fn is None)
pl = GapPositionPool(32, mode="learned")
check("learned weights carry grad_fn", pl.weights(zz, None).grad_fn is not None)
check("variance mode exposes no parameters", len(list(pv.parameters())) == 0)
check("learned mode exposes exactly one parameter tensor", len(list(pl.parameters())) == 1)
# Buffers must checkpoint, so they have to be in the state_dict.
check("variance EMA state is checkpointed", {"var_ema", "var_ema_t"} <= set(pv.state_dict()))

print("\n11. eval.gap_pool_profile's verdict discriminates (calibration regression)")
# The probe decides whether --bt-gap-pool is worth a training run at all, so a
# miscalibrated verdict costs either GPU hours or a real effect. Pinned here because the
# first version gated on profile ENTROPY and returned NO on the 4-of-256 focal case — the
# exact regime the flag exists for — since entropy barely moves under a mild reweighting
# spread over many positions. The permutation test replaced it; these four scenarios are
# what "replaced it correctly" means.
from eval.gap_pool_profile import pooled_metrics  # noqa: E402


def verdict_for(focal, gain, grid=256, n_subj=512, n_ch=16, null_draws=12, min_effect=0.01):
    if focal == 0:  # same total signal spread over EVERY position -> flat profile
        s = torch.randn(n_subj, n_ch) * gain
        zz_ = torch.randn(2, n_subj, n_ch, grid) + s.unsqueeze(0).unsqueeze(-1) / math.sqrt(grid)
    else:
        zz_ = planted(n_subj=n_subj, n_ch=n_ch, grid=grid, n_focal=focal, focal_gain=gain)
    w_ = GapPositionPool(grid, mode="variance", ema=0.0).weights(zz_, None).numpy()
    rng_ = np.random.RandomState(0)
    var_ = pooled_metrics(zz_, w_, 128, 8)["xview_corr"]
    null_ = [
        pooled_metrics(zz_, w_[rng_.permutation(grid)], 128, 8, seed=1 + k)["xview_corr"] for k in range(null_draws)
    ]
    gain_ = var_ - float(np.mean(null_))
    z_ = gain_ / max(float(np.std(null_)), 1e-6)
    return ("NO" if z_ <= 3.0 else ("MARGINAL" if gain_ < min_effect else "YES")), z_


for _name, _focal, _gain, _want in (
    ("focal 4/256 (the target regime)", 4, 1.0, "YES"),
    ("focal 32/256", 32, 1.0, "YES"),
    ("signal spread evenly (flat)", 0, 1.0, "NO"),
    ("pure noise, no subject signal", 0, 0.0, "NO"),
):
    _got, _z = verdict_for(_focal, _gain)
    check(f"{_name} -> {_want}", _got == _want, f"got {_got} (z={_z:.1f})")

print("\n12. channel-agreement diagnostic separates the three ways a profile goes flat")
# A flat channel-AVERAGED profile has three causes with three different responses, and the
# probe has to tell them apart or its NO is uninterpretable:
#   delocalised     every channel flat            -> drop the idea entirely
#   disagree        channels peaked, different places -> a SHARED weight vector cannot help,
#                                                     per-channel weights could
#   style_variance  variance peaked on view-specific signal -> variance is the wrong statistic
from eval.gap_pool_profile import channel_profiles  # noqa: E402


def _regime(kind, grid=128, n_subj=800, n_ch=16, n_fac=4):
    gt_ = np.random.RandomState(0).randn(n_subj, n_fac)
    zz_ = torch.randn(2, n_subj, n_ch, grid) * 0.5

    def f(k):
        return torch.tensor(gt_[:, k], dtype=torch.float32)[None, :, None]

    if kind == "agree":  # all channels encode factors in the SAME positions
        for k in range(n_fac):
            zz_[:, :, k, :8] += f(k)
    elif kind == "disagree":  # each channel localises somewhere different
        for k in range(n_fac):
            zz_[:, :, k, 8 * k : 8 * k + 8] += f(k)
    elif kind == "style_variance":  # content spread thin, huge PER-VIEW variance in a few positions
        for k in range(n_fac):
            zz_[:, :, k, :] += f(k) / math.sqrt(grid)
        zz_[..., 120:] += torch.randn(2, n_subj, 1, 1) * 6.0
    elif kind == "delocalised":  # factors equally visible at every position
        for k in range(n_fac):
            zz_[:, :, k, :] += f(k)
    c_ = channel_profiles(zz_)
    loc_ = c_["per_channel_top10_p75"] > 0.13
    return loc_, ((not (c_["agreement"] > 0.5)) if loc_ else False), c_


torch.manual_seed(0)
for _kind, _want_loc, _want_dis in (
    ("agree", True, False),
    ("disagree", True, True),
    ("style_variance", True, False),
    ("delocalised", False, False),
):
    _loc, _dis, _c = _regime(_kind)
    check(
        f"{_kind}: localised={_want_loc} disagree={_want_dis}",
        (_loc, _dis) == (_want_loc, _want_dis),
        f"p75={_c['per_channel_top10_p75']:.3f} agreement={_c['agreement']:+.2f}",
    )

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
raise SystemExit(1 if FAILS else 0)
