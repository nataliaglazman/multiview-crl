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
# one the variance hinge acts on. (On real runs that hinge, not the pooling, is what sets
# feat_std — 0.004 pre-hinge, ~1.1 parked after. It cannot diagnose pooling; see
# --bt-gap-std-coeff. The scale-invariant cross-view correlation below is the real test.)
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

print("\n13. feature_scale reproduces the shipped var_loss and classifies the hinge state")
# The probe reports the variance hinge's state so a run can tell "inert" from
# "boundary-active" without guessing from a noisy TensorBoard curve. Its implied_var_loss
# has to equal what barlow_twins_loss actually computes, or the comparison against the
# logged Contrastive/gap_var_loss_L* is meaningless.
from eval.gap_pool_profile import feature_scale  # noqa: E402

for _name, _scale, _want_below in (("below target", 0.3, True), ("pinned", 1.05, False), ("parked", 3.0, False)):
    _n, _c, _p = 400, 20, 64
    _b = torch.randn(2, _n, _c, _p)
    _b = _b / _b.mean(-1).std(dim=1, unbiased=False).mean() * _scale
    _fs = feature_scale(_b)
    _l = barlow_twins_loss(
        _b.mean(-1), estimated_content_indices=[list(range(_c))], subsets=[[0, 1]], lambd=1.0, std_coeff=1.0
    )
    _shipped = _l._contrastive_diag["var_loss"]
    check(
        f"{_name}: implied_var_loss == shipped var_loss",
        abs(_fs["implied_var_loss"] - _shipped) < 1e-4,
        f"implied={_fs['implied_var_loss']:.4f} shipped={_shipped:.4f}",
    )
    check(
        f"{_name}: below-target fraction reads {'high' if _want_below else 'low'}",
        (_fs["frac_below_1"] > 0.5) == _want_below,
        f"below1={_fs['frac_below_1']:.0%} pinned={_fs['frac_pinned']:.0%}",
    )

print("\n14. per-factor profiles see localisation the AVERAGED profile cannot")
# Why this exists: the averaged factor-recovery profile was used to conclude "the factor
# information is delocalised" on two real checkpoints (top10% 0.115 / 0.117 against a 0.10
# flat floor). This plants three PERFECTLY localised factors plus one global one and shows
# the averaged profile still reads 0.10 — indistinguishable from the real data. The
# averaged measurement cannot support that conclusion, and this pins the fact.
from eval.gap_pool_profile import per_factor_recovery, profile_overlap, profile_stats  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
_p, _n, _c = 128, 1000, 24
_gt = np.random.randn(_n, 4)
_z = torch.randn(2, _n, _c, _p) * 0.5


def _fac(k):
    return torch.tensor(_gt[:, k], dtype=torch.float32)[None, :, None, None]


_z += _fac(0) * 0.5  # global: readable everywhere
_z[:, :, :, 0:8] += _fac(1) * 3.0  # localised at 0-7
_z[:, :, :, 60:68] += _fac(2) * 3.0  # localised at 60-67, a DIFFERENT place
_z[:, :, :, 0:8] += _fac(3) * 3.0  # localised at 0-7, the SAME place as factor 1

_pf = per_factor_recovery(_z, _gt, np.random.RandomState(0).permutation(_n)[: _n // 2])
_stats = [profile_stats(_pf[k] / _pf[k].sum()) for k in range(4)]
check("global factor reads flat", _stats[0]["top10pct_mass"] < 0.15, f"top10%={_stats[0]['top10pct_mass']:.3f}")
check(
    "all three localised factors read concentrated",
    all(_stats[k]["top10pct_mass"] > 0.5 for k in (1, 2, 3)),
    f"top10%={[round(_stats[k]['top10pct_mass'], 2) for k in (1, 2, 3)]}",
)
check(
    "each localised factor peaks where it was planted",
    int(_pf[1].argmax()) < 8 and 60 <= int(_pf[2].argmax()) < 68 and int(_pf[3].argmax()) < 8,
    f"peaks={[int(_pf[k].argmax()) for k in (1, 2, 3)]}",
)
_avg = _pf.mean(0)
_avg_top10 = profile_stats(_avg / _avg.sum())["top10pct_mass"]
check(
    "THE POINT: averaging hides all of it (reads flat despite 3 localised factors)",
    _avg_top10 < 0.15,
    f"averaged top10%={_avg_top10:.3f} vs real-data 0.115/0.117 — indistinguishable",
)
check(
    "same-place factors have cosine ~1",
    profile_overlap([_pf[1], _pf[3]]) > 0.9,
    f"{profile_overlap([_pf[1], _pf[3]]):+.3f}",
)
check(
    "different-place factors have cosine ~0",
    abs(profile_overlap([_pf[1], _pf[2]])) < 0.3,
    f"{profile_overlap([_pf[1], _pf[2]]):+.3f}",
)

print("\n15. split-half reliability separates real localisation from noise concentration")
# Concentration on a NOISY profile is inflated, so raw top10% alone makes the WORST
# recovered factor look the MOST localised. A permutation null does not fix this: shuffling
# labels drives R^2 to zero everywhere, and an all-zeros profile with one surviving entry
# normalises to MAXIMAL concentration, so the null reads more localised than the data.
# Split-half does fix it. The case that matters is f2: genuinely localised but recovered at
# only peak R^2 ~0.12, the same level as this project's lesion factors — it must still be
# detected, or a negative result on those factors would be a sensitivity failure, not a
# finding.
from eval.gap_pool_profile import profile_reliability  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
_p2, _n2, _c2 = 128, 2000, 24
_gt2 = np.random.randn(_n2, 4)
_z2 = torch.randn(2, _n2, _c2, _p2) * 0.5


def _fac2(k):
    return torch.tensor(_gt2[:, k], dtype=torch.float32)[None, :, None, None]


_z2 += _fac2(0) * 2.0  # f0 strong, global
_z2[:, :, :, 0:8] += _fac2(1) * 3.0  # f1 localised, strongly recovered
_z2[:, :, :, 60:68] += _fac2(2) * 0.9  # f2 localised, WEAKLY recovered (the sensitivity case)
_z2 += _fac2(3) * 0.25  # f3 global and near-unrecoverable (the noise trap)


def _localised(z_, gt_, k_count):
    """Replicates main()'s decision: sigma against the LEAST concentrated factor, plus
    split-half reliability. Returns [(is_localised, top10, sigma, r), ...]."""
    _ix = np.random.RandomState(0).permutation(len(gt_))[: len(gt_) // 2]
    _pf_ = per_factor_recovery(z_, gt_, _ix)
    _rl_, _cn_ = profile_reliability(z_, gt_, _ix)
    _rows = []
    for k in range(k_count):
        if _pf_[k].max() <= 0:
            _rows.append(None)
            continue
        _t = profile_stats(_pf_[k] / _pf_[k].sum())["top10pct_mass"]
        _e = abs(_cn_[k, 0] - _cn_[k, 1]) / 2.0 if np.all(np.isfinite(_cn_[k])) else np.nan
        _rows.append((_t, _e, _rl_[k]))
    _ok = [r for r in _rows if r is not None]
    _ri = int(np.argmin([r[0] for r in _ok]))
    _ref, _rerr = _ok[_ri][0], _ok[_ri][1]
    _out = []
    for r in _rows:
        if r is None:
            _out.append((False, float("nan"), float("nan"), float("nan")))
            continue
        _t, _e, _r = r
        _den = math.sqrt(_e**2 + _rerr**2) if np.isfinite(_e) and np.isfinite(_rerr) else float("nan")
        _sg = (_t - _ref) / _den if np.isfinite(_den) and _den > 1e-9 else float("nan")
        _out.append((np.isfinite(_sg) and _sg > 2.0 and np.isfinite(_r) and _r > 0.3, _t, _sg, _r))
    return _out


_res2 = _localised(_z2, _gt2, 4)
for _k, _nm, _want in (
    (0, "strong global -> not localised", False),
    (1, "strongly-recovered localised -> localised", True),
    (2, "WEAKLY-recovered localised -> still localised", True),
    (3, "near-unrecoverable global -> not localised (noise trap)", False),
):
    check(_nm, _res2[_k][0] == _want, f"top10%={_res2[_k][1]:.3f} sigma={_res2[_k][2]:.1f} r={_res2[_k][3]:.2f}")

print("\n16. the decision is calibrated for MILD localisation, and 0.10 is the wrong reference")
# Two failures this pins. (a) A fixed 0.15 cut was calibrated on planted extremes (top10%
# ~1.0, all mass in 8 of 128 positions); real localisation seen through a wide receptive
# field is far milder and that cut calls it flat. (b) top10% takes the largest 10% of
# entries, so it exceeds 0.10 in ANY finite sample -- a planted GLOBAL factor reads 0.115,
# and against a 0.10 reference with a split-half error bar that is 22 "sigma" of nothing.
# The reference has to be an empirically flat profile: the least concentrated factor here.
torch.manual_seed(0)
np.random.seed(0)
_n3, _p3, _c3 = 2000, 128, 24
_gt3 = np.random.randn(_n3, 3)
_z3 = torch.randn(2, _n3, _c3, _p3) * 0.5
_f3 = lambda k: torch.tensor(_gt3[:, k], dtype=torch.float32)[None, :, None, None]  # noqa: E731
_z3 += _f3(0) * 2.0  # global
_z3 += _f3(1) * 0.8
_z3[:, :, :, 0:16] += _f3(1) * 0.8  # MILD: visible everywhere, stronger in a region
_z3 += _f3(2) * 0.8
_z3[:, :, :, 96:112] += _f3(2) * 0.8  # MILD, elsewhere
_mild = _localised(_z3, _gt3, 3)
check("mild: global factor not localised", not _mild[0][0], f"top10%={_mild[0][1]:.3f} sigma={_mild[0][2]:.1f}")
for _k in (1, 2):
    check(
        f"mild: localised factor {_k} detected despite top10% only {_mild[_k][1]:.2f}",
        _mild[_k][0],
        f"sigma={_mild[_k][2]:.1f} r={_mild[_k][3]:.2f}",
    )
check("mild: the old fixed 0.15 cut would have been needed", _mild[1][1] > 0.15 or _mild[2][1] > 0.15)

torch.manual_seed(0)
np.random.seed(0)
_gt4 = np.random.randn(_n3, 3)
_z4 = torch.randn(2, _n3, _c3, _p3) * 0.5
_f4 = lambda k: torch.tensor(_gt4[:, k], dtype=torch.float32)[None, :, None, None]  # noqa: E731
for _k, _g in ((0, 2.0), (1, 1.5), (2, 1.0)):
    _z4 += _f4(_k) * _g  # ALL global, none localised
_allg = _localised(_z4, _gt4, 3)
check(
    "all-global: none flagged as localised", not any(r[0] for r in _allg), f"top10%={[round(r[1], 3) for r in _allg]}"
)
check(
    "all-global: raw top10% still reaches 0.12+ on pure noise (why eyeballing fails)",
    max(r[1] for r in _allg) > 0.115,
    f"max raw top10%={max(r[1] for r in _allg):.3f} — comparable to real lesion_y/cortical values",
)

print("\n17. the per-channel ceiling test discriminates the three regimes it must")
# eval/per_channel_pool_ceiling.py answers "would giving each channel its own position
# weighting help?" by handing the scheme ground truth and asking how much it beats uniform
# GAP even then. It has to separate three cases that call for different decisions, and the
# third is the one that would otherwise be mistaken for a win.
from eval.per_channel_pool_ceiling import _pca_reduce, _ridge_r2, best_position_per_channel  # noqa: E402

_rng = np.random.RandomState(0)


def _ceiling(x_, gt_):
    _nn = len(gt_)
    _pm = _rng.permutation(_nn)
    _ft, _ev = _pm[: _nn // 2], _pm[_nn // 2 :]
    _bp, _ = best_position_per_channel(x_, gt_, _ft)
    _gap = x_.mean(-1)
    _pc = np.stack([x_[:, c, _bp[c]] for c in range(x_.shape[1])], 1)
    _fl = x_.reshape(_nn, -1)
    _tr, _te = _pca_reduce(_fl[_ft], _fl[_ev], min(128, max(8, len(_ft) // 4)))
    _u = float(np.mean(_ridge_r2(_gap[_ft], gt_[_ft], _gap[_ev], gt_[_ev])))
    _p = float(np.mean(_ridge_r2(_pc[_ft], gt_[_ft], _pc[_ev], gt_[_ev])))
    _f = float(np.mean(_ridge_r2(_tr, gt_[_ft], _te, gt_[_ev])))
    return _p - _u, _f - _u, len(np.unique(_bp))


_N, _C, _P, _K = 2000, 24, 64, 4
_g = _rng.randn(_N, _K)

_xa = _rng.randn(_N, _C, _P) * 0.5  # channels specialise: channel c reads factor c%K at 8*(c%K)
for _c in range(_C):
    _xa[:, _c, 8 * (_c % _K)] += _g[:, _c % _K] * 3.0
_ga, _ca, _ua = _ceiling(_xa, _g)
check("specialised channels -> large gain", _ga > 0.05, f"gain={_ga:+.3f} ceil={_ca:+.3f}")
check("specialised channels -> gain captures most of the ceiling", _ga >= 0.5 * _ca, f"{_ga / _ca:.0%}")

_xb = _rng.randn(_N, _C, _P) * 0.5  # every channel carries every factor everywhere
for _k in range(_K):
    _xb += _g[:, _k][:, None, None] * 0.8
_gb, _cb, _ub = _ceiling(_xb, _g)
check("fully distributed -> no gain (idea is dead)", _gb < 0.01, f"gain={_gb:+.3f}")

_xc = _rng.randn(_N, _C, _P) * 0.5  # factor is a CONTRAST between two positions
for _k in range(_K):
    _xc[:, :, 4 * _k] += _g[:, _k][:, None] * 2.0
    _xc[:, :, 4 * _k + 1] -= _g[:, _k][:, None] * 2.0
_gc, _cc, _uc = _ceiling(_xc, _g)
check("spatial pattern -> full patch beats per-channel", _cc > _gc * 1.5, f"gain={_gc:+.3f} ceil={_cc:+.3f}")

# The distinct-position count is descriptive, and reading it as evidence of specialisation is
# BACKWARDS — pinned because the first version of the script did exactly that.
check(
    "distinct-position count is NOT evidence of specialisation",
    _ua < _ub,
    f"specialised used {_ua}/{_C} distinct, distributed(dead) used {_ub}/{_C}",
)

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
raise SystemExit(1 if FAILS else 0)
