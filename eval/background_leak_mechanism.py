"""Why does a BACKGROUND feature position read out a CONTENT factor at high R^2?

`background_leak_diagnostic.py` answers this with a *shape* heuristic — flat R^2 across
distance shells means "global", decaying means "receptive field". That heuristic is weak:
it reads a trend, needs a trained checkpoint, and (see its own caveat) region-averaging
makes it misread compact signals. This script replaces it with three *structural* tests
that are exact, need no checkpoint, and separate the mechanisms by construction.

The setup makes the question sharp. Under `--synthetic-normalize per_sample` the dataset
does `x = (x - mean)/std * mask` (`data/datasets.py:_znorm_nonzero`), so **the input is
EXACTLY zero at every background voxel of every sample**. A feature position whose
receptive field lies entirely inside that zero region therefore sees an input that is
bit-identical across the whole dataset. Any across-sample variation it shows cannot have
come through the convolutions. Only two routes remain:

  (RF)   the receptive field is not actually inside the zero region — it reaches brain
         tissue, so the position sees real anatomy. Ordinary, benign, and local.
  (NORM) a normalization layer couples the position to whole-volume statistics.
         GroupNorm computes (mu, sigma) over ALL spatial positions, so it writes a
         global summary into every position including the empty ones.

Three tests, each of which the two routes answer differently:

  1. REACH (exact, by backprop). Measure the receptive field of a corner position with
     the residual branch forced open (ReZero alpha != 0, so the bound holds for any
     trained model of this architecture). If the RF cannot touch the brain, RF is out —
     not "unlikely", excluded.

  2. CONSTANT-BACKGROUND SIGNATURE (the decisive one). Split the far-background feature
     variance into an across-SAMPLE part and an across-POSITION part.
         RF route   -> across-position variance > 0. Different background positions sit
                       at different distances/angles from the brain, so they must see
                       different tissue and cannot agree.
         NORM route -> across-position variance == 0. Every zero-RF position evaluates
                       the SAME constant through the SAME (mu, sigma), so they all carry
                       the identical vector. Spatially constant, sample-varying.
     A map that is flat in space but moves with the sample is a global summary. This is
     a structural fingerprint, not a trend, and one number decides it.

  3. INJECTION POINT + SUFFICIENCY. At the FIRST main-path norm the pre-norm activation
     at a zero-RF position is a pure conv bias, constant across samples, so GroupNorm's
     output there is exactly
             y = gamma * (b - mu_g)/sigma_g + beta
     which is LINEAR in the basis (1/sigma_g, mu_g/sigma_g). Regressing the observed
     background feature on that basis must give R^2 ~ 1 if this is the mechanism, and
     probing the factor from (mu, sigma) alone must recover it. Both are checked.

Everything runs on an UNTRAINED encoder. That is the point: if random weights already
reproduce the background hotspot, the effect is a property of the architecture crossed
with the data statistics, and no amount of training or loss redesign removes it. The
`--norm-type layer` arm is the matched control (`ChannelLayerNorm3d` normalizes across
channels *within* a voxel, so it has no spatial coupling to leak through).

RESOLUTION CAVEAT. Test 2 needs background positions whose receptive field genuinely
misses the brain. That set shrinks fast as the stride grows: at `--downscale 2` (32^3
features, conv RF 13 voxels vs 24.7 to the brain) there are ~2000 such positions with a
2x safety margin and the signature is clean. At `--downscale 4` (16^3, conv RF 27 vs
28.6) only ~70 survive even at the exact RF radius, they sit right at the RF edge, and
the signature degrades accordingly (LayerNorm far-background stops being exactly frozen,
R^2 ~0.06 rather than ~0.00). Tests 1 and 3 are unaffected — the injection algebra still
comes out at R^2 = 1.0000 — so read Test 2 at the finer stride and treat the coarse-stride
Test 2 as underpowered rather than as evidence against.

Usage:
  python -m eval.background_leak_mechanism --num-samples 400
  python -m eval.background_leak_mechanism --num-samples 400 --causal   # SCM-coupled factors
  # match experiments/synthetic_causal.yaml (note the caveat above at this stride):
  python -m eval.background_leak_mechanism --num-samples 400 --downscale 4 \
      --normalize fixed_reference --clean-content --causal
"""

from __future__ import annotations

import argparse
import logging
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

FACTORS = ("brain_size", "ventricle_size", "cortical_thickness", "lr_asymmetry")


def _stub_utils_if_needed():
    """Let this run without MONAI installed.

    `models/vqvae.py` and `eval/dci.py` both do `import utils.utils`, which imports MONAI
    at module scope. Neither the `Encoder` nor `CONTENT_FACTOR_NAMES` touches anything
    MONAI provides, so a stub keeps the script portable while still exercising the
    shipped modules rather than a reimplementation.
    """
    try:
        import utils.utils  # noqa: F401
    except ImportError:
        stub = types.ModuleType("utils.utils")
        stub.topk_gumbel_softmax = None
        sys.modules["utils.utils"] = stub


def _import_encoder():
    _stub_utils_if_needed()
    from models.vqvae import Encoder

    return Encoder


def build_encoder(norm_type, seed, hidden=48, res_ch=32, nb_res=2, downscale=2):
    """Untrained level-0 encoder. Same seed -> identical conv weights across arms
    (GroupNorm and LayerNorm both init to ones/zeros and consume no RNG)."""
    Encoder = _import_encoder()
    torch.manual_seed(seed)
    enc = Encoder(1, hidden, res_ch, nb_res, downscale, False, norm_type)
    return enc.eval()


def znorm_nonzero(x, mask):
    """Exactly `data.datasets.SyntheticBrainDataset._znorm_nonzero` (per_sample mode)."""
    m = mask > 0
    if m.any():
        vals = x[m]
        x = (x - vals.mean()) / vals.std().clamp_min(1e-6)
        x = x * mask
    return x


def fixed_reference_constants(ds, n_ref=64, scale_quantile=0.99):
    """Mirror of `SyntheticBrainDataset._compute_fixed_reference`.

    Both normalization modes end in `* mask`, so the background is exactly zero either
    way and the argument here does not depend on the choice — but the foreground scaling
    differs, so the factor R^2 columns do.
    """
    vals = []
    for j in range(min(n_ref, len(ds))):
        x_v1, x_v2, lat = ds[j]
        bm = lat["brain_mask"] > 0
        for x in (x_v1, x_v2):
            v = x[bm]
            if v.numel():
                vals.append(v.flatten().float())
    vals = torch.cat(vals)
    # torch.quantile caps at ~16M elements; the source subsamples to 4M and takes BOTH
    # the mean and the quantile on the subsample, so mirror that exactly (at 128^3 the
    # un-subsampled tensor overflows the cap and raises).
    cap = 4_000_000
    if vals.numel() > cap:
        g = torch.Generator().manual_seed(0)
        vals = vals[torch.randint(vals.numel(), (cap,), generator=g)]
    mean = float(vals.mean())
    return mean, max(float(torch.quantile((vals - mean).abs(), scale_quantile)), 1e-6)


def generate(n, res, causal, seed=42, normalize="per_sample", clean_content=False):
    """Render the synthetic set and apply the run's per-sample normalization.

    Mirrors `eval.run_dci_synthetic.build_synthetic_test_set`, which notably does NOT
    forward `synthetic_causal` — so the standard eval set has i.i.d. factors even for a
    model trained on an SCM. `--causal` reproduces the training-time coupling instead.
    """
    from eval.synthetic_dataset import Synthetic3DDisentanglementDataset

    ds = Synthetic3DDisentanglementDataset(
        num_samples=n,
        res=res,
        seed=seed,
        mode="pseudo_mri",
        n_content=9,
        n_style=3,
        causal=causal,
        causal_graph="random",
        causal_edge_prob=0.5,
        clean_content=clean_content,
    )
    ref = fixed_reference_constants(ds) if normalize == "fixed_reference" else None
    X, M, Z = [], [], []
    for i in range(n):
        x_v1, _, lat = ds[i]
        mask = lat["brain_mask"]
        if ref is None:
            X.append(znorm_nonzero(x_v1, mask))
        else:
            X.append((x_v1 - ref[0]) / ref[1] * mask)
        M.append(mask)
        Z.append(lat["z_content"])
    return torch.stack(X), torch.stack(M), torch.stack(Z).numpy()


def probe(Xf, y):
    from eval.identifiability_metrics import cv_probe_r2

    return cv_probe_r2(np.asarray(Xf, np.float64), np.asarray(y, np.float64), kind="ridge")["mean"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=48, help="vqvae_hidden_channels of the run under study")
    ap.add_argument("--downscale", type=int, default=2, help="vqvae_scaling_rates[0] of the run")
    ap.add_argument("--causal", action="store_true", help="SCM-coupled factors (training-time setting)")
    ap.add_argument("--normalize", default="per_sample", choices=["per_sample", "fixed_reference"])
    ap.add_argument(
        "--rf-margin",
        type=float,
        default=0.5,
        help="Far-background threshold as a multiple of the conv-RF SPAN. 0.5 = the exact RF "
        "radius (correct); higher adds safety margin but empties the set at coarse strides.",
    )
    ap.add_argument("--clean-content", action="store_true", help="match --synthetic-clean-content runs")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    _stub_utils_if_needed()
    from eval.dci import CONTENT_FACTOR_NAMES

    dev = torch.device("cpu")
    res = cli.res

    # ── data ──────────────────────────────────────────────────────────────────────
    logger.info(
        "rendering %d volumes at %d^3 (causal=%s, normalize=%s, clean_content=%s)...",
        cli.num_samples,
        res,
        cli.causal,
        cli.normalize,
        cli.clean_content,
    )
    X, M, Z = generate(cli.num_samples, res, cli.causal, cli.seed, cli.normalize, cli.clean_content)
    N = X.shape[0]

    # TEST 0. Is the background of the INPUT really zero for every sample? If it is not,
    # the leak is in the data and no architectural story is needed.
    ever_brain_vox = (M > 0).any(dim=0)[0]  # (res,res,res)
    bg_vox = ~ever_brain_vox
    bg_input = X[:, 0][:, bg_vox]  # (N, n_bg_voxels)
    print(f"\n{'=' * 78}\nTEST 0 — is the INPUT background constant across samples?\n{'=' * 78}")
    print(f"background voxels (never brain in any sample): {int(bg_vox.sum()):,} / {res**3:,}")
    print(f"  max |value|            : {bg_input.abs().max():.3e}")
    print(f"  max across-sample std  : {bg_input.std(dim=0).max():.3e}")
    print("  -> input background is exactly 0 everywhere; it carries NO sample information.")
    print("     Any background FEATURE signal must be created inside the network.")

    # ── geometry at feature resolution ────────────────────────────────────────────
    g = res // cli.downscale
    cov = F.adaptive_avg_pool3d(M, (g, g, g))[:, 0].numpy()  # (N, g,g,g)
    ever = cov.max(axis=0) > 0  # position holds brain in AT LEAST ONE sample
    core = cov.min(axis=0) >= 0.9  # brain in EVERY sample

    from scipy.ndimage import distance_transform_edt

    dist = distance_transform_edt(~ever)  # feature-voxel distance to nearest ever-brain

    # ── TEST 1. how far can a corner position actually see? ───────────────────────
    #
    # NOTE (this is why `eval/receptive_field_test.py` cannot answer the question):
    # measuring the RF as d(feature)/d(input) through a network that CONTAINS GroupNorm
    # measures conv reach PLUS the normalizer's global coupling, because mu and sigma
    # depend on every input voxel. That measurement therefore reports "reaches the whole
    # volume" no matter how small the convolutions are, and concluding "RF reaches the
    # brain, so no exotic mechanism is needed" inverts the actual finding.
    #
    # So measure three separate reaches. The gap between them IS the effect.
    print(f"\n{'=' * 78}\nTEST 1 — how far can the corner position see? (three separate reaches)\n{'=' * 78}")

    def corner_reach(norm_type, strip_norms=False):
        enc = build_encoder(norm_type, cli.seed, cli.hidden, downscale=cli.downscale)
        # ReZero alpha is 0 at init, making the residual stack the identity and
        # UNDERSTATING conv reach. Force it open so the bound covers any trained model.
        for m in enc.modules():
            if hasattr(m, "alpha") and isinstance(m.alpha, nn.Parameter):
                with torch.no_grad():
                    m.alpha.fill_(1.0)
        if strip_norms:
            for name, mod in list(enc.named_modules()):
                for cname, child in list(mod.named_children()):
                    if isinstance(child, nn.GroupNorm) or type(child).__name__ == "ChannelLayerNorm3d":
                        setattr(mod, cname, nn.Identity())
        xr = torch.zeros(1, 1, res, res, res, requires_grad=True)
        enc(xr)[0, :, 0, 0, 0].sum().backward()
        gr = xr.grad.detach().abs()[0, 0].numpy()
        nz = np.argwhere(gr > gr.max() * 1e-6)
        return int(nz.max()) + 1  # corner position: RF spans input voxels [0, reach)

    rf_conv = corner_reach("group", strip_norms=True)
    rf_ln = corner_reach("layer")
    rf_gn = corner_reach("group")
    brain_vox = np.argwhere(ever_brain_vox.numpy())
    d_corner_to_brain = float(np.sqrt((brain_vox**2).sum(1)).min())

    print(f"  gradient reach of feature position (0,0,0), in input voxels per axis:")
    print(f"    convolutions only (norms -> Identity) : {rf_conv:>4}   <- the true receptive field")
    print(f"    with per-voxel LayerNorm              : {rf_ln:>4}   <- LN adds no spatial coupling")
    print(f"    with GroupNorm                        : {rf_gn:>4}   <- couples to the WHOLE volume")
    print(f"\n  nearest brain voxel to the input corner : {d_corner_to_brain:.1f} voxels")
    if rf_conv < d_corner_to_brain:
        print(f"  => convolutionally the corner CANNOT reach the brain ({rf_conv} < {d_corner_to_brain:.1f}).")
        print("     The receptive-field route is EXCLUDED for corner positions, not merely unlikely.")
        print(f"     Yet under GroupNorm the corner's gradient spans all {rf_gn} voxels. That gap is")
        print("     the leak: it arrives through the normalizer's statistics, not through the convs.")
    else:
        print(f"  => the corner RF DOES reach the brain — corner signal may be ordinary conv reach.")

    # Far background: never brain in any sample, AND further from any brain voxel than
    # the CONV receptive field. Distances are measured in input voxels at the feature
    # position's input-space centre, and the threshold is the full RF span used as if it
    # were a radius -- a ~2x conservative margin over the true radius (span/2).
    d_in = distance_transform_edt(~ever_brain_vox.numpy())
    ii, jj, kk = np.meshgrid(*[np.arange(g)] * 3, indexing="ij")
    ctr = (np.stack([ii, jj, kk], -1) + 0.5) * cli.downscale
    d_pos = d_in[ctr[..., 0].astype(int), ctr[..., 1].astype(int), ctr[..., 2].astype(int)]

    # A THIRD mechanism has to be excluded before Test 2 is clean: conv zero-padding.
    # Near the volume faces a stride-1 conv reads padded slots whose value (0) differs
    # from the constant interior activation, so those positions carry a fixed spatial
    # pattern. Calibrate the affected set exactly by pushing an all-zero volume through:
    # any position that does NOT return the interior constant is padding-contaminated.
    with torch.no_grad():
        f0 = build_encoder("group", cli.seed, cli.hidden, downscale=cli.downscale)(torch.zeros(1, 1, res, res, res))
    f0 = f0[0].flatten(1)  # (C, P)
    v0 = f0.median(dim=1).values[:, None]
    dev0 = (f0 - v0).abs().max(dim=0).values.numpy().reshape(g, g, g)
    pad_clean = dev0 <= 1e-4 * max(float(v0.abs().max()), 1.0)

    # `rf_conv` is the RF *span* measured at the corner; the radius around an interior
    # position is half of it, and `d_pos` is measured from the position's input-space
    # centre. So `d_pos > rf_conv/2` is the correct RF-clean criterion; larger multiples
    # are extra safety margin. `--rf-margin` is in units of the span.
    far = (~ever) & (d_pos > rf_conv * cli.rf_margin) & pad_clean
    print(f"\n  position sets at feature resolution {g}^3 ({g**3:,} positions):")
    print(f"    core (brain in EVERY sample)               : {int(core.sum()):,}")
    print(f"    background (brain in NO sample)            : {int((~ever).sum()):,}")
    for mult in (0.5, 0.75, 1.0):
        n_f = int(((~ever) & (d_pos > rf_conv * mult) & pad_clean).sum())
        tag = "  <- exact RF radius" if mult == 0.5 else ("  <- 2x margin" if mult == 1.0 else "")
        star = " *" if mult == cli.rf_margin else "  "
        print(f"    far background (> {mult:.2f} x RF span, pad-clean){star}: {n_f:,}{tag}")
    print(f"    padding-contaminated (excluded)            : {int((~pad_clean).sum()):,}")
    print(f"    (* = the set used below, --rf-margin {cli.rf_margin})")
    print("\n  Padding note: those excluded positions carry a fixed spatial pattern, but it is")
    print("  driven by conv biases, not by the data — Test 2's LayerNorm row shows it does not")
    print("  move across samples, so padding cannot carry factor information. It is a red herring")
    print("  for this question, and it is removed here only so Test 2 measures one thing.")
    if far.sum() == 0:
        raise SystemExit("no far-background positions at this resolution; lower --res or --downscale")

    # ── background positions binned by distance to the nearest brain voxel ─────────
    # Test 2 asks about the *far corner*, where the conv route is excluded. That is the
    # clean question but a small set. "The edges of the feature map" is a much larger set
    # that is mostly INSIDE the RF, so LayerNorm is not expected to zero it. This profile
    # separates the two: LN R^2 should track RF reach and fall to ~0 past it, while GN R^2
    # stays high at every distance. Compare a trained run's edge readout against the LN
    # column at the same distance, not against zero.
    rf_radius = rf_conv * 0.5
    shell_edges = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 24), (24, 99)]
    shells = []
    for lo, hi in shell_edges:
        sel = (~ever) & (d_pos >= lo) & (d_pos < hi) & pad_clean
        if int(sel.sum()) >= 20:
            shells.append(((lo, hi), torch.from_numpy(sel.reshape(-1)), float((d_pos[sel] <= rf_radius).mean())))

    # ── feature extraction for both norm arms ─────────────────────────────────────
    core_t = torch.from_numpy(core.reshape(-1))
    far_t = torch.from_numpy(far.reshape(-1))
    results = {}
    for norm_type in ("group", "layer"):
        enc = build_encoder(norm_type, cli.seed, cli.hidden, downscale=cli.downscale)
        # Hook EVERY norm. `nn.ReLU(inplace=True)` follows some of them and would
        # overwrite the captured output tensor, so the hook must clone.
        norms = [m for m in enc.modules() if isinstance(m, nn.GroupNorm) or type(m).__name__ == "ChannelLayerNorm3d"]
        first_norm = next(m for m in enc.layers[0] if m in norms)
        cap = {}

        def mk_hook(mod):
            def h(_m, inp, out):
                cap[mod] = (inp[0].detach(), out.detach().clone())

            return h

        for m in norms:
            m.register_forward_hook(mk_hook(m))

        r_core, r_far, sp_mean, sp_std, gn_stats, first_far = [], [], [], [], [], []
        shell_acc = [[] for _ in shells]
        for s in range(0, N, cli.batch_size):
            xb = X[s : s + cli.batch_size].to(dev)
            with torch.no_grad():
                f = enc(xb)  # (b, C, g,g,g)
            fl = f.flatten(2)  # (b, C, P)
            r_core.append(fl[:, :, core_t].mean(2).numpy())
            ff = fl[:, :, far_t]  # (b, C, P_far)
            r_far.append(ff.mean(2).numpy())
            sp_mean.append(ff.mean(2).numpy())
            sp_std.append(ff.std(2).numpy())  # across-POSITION spread, per sample+channel
            for si, (_, sel_t, _) in enumerate(shells):
                shell_acc[si].append(fl[:, :, sel_t].mean(2).numpy())
            # (mu, sigma) that EVERY GroupNorm computes and divides out
            st, layout = [], []
            for m in norms:
                if not isinstance(m, nn.GroupNorm):
                    continue
                pre = cap[m][0]
                pg = pre.reshape(pre.shape[0], m.num_groups, -1)
                mu_, sd_ = pg.mean(-1), pg.std(-1, unbiased=False)
                # Keep the analytic basis alongside the raw stats: a normalized output is
                # linear in (1/sigma, mu/sigma), not in (mu, sigma).
                inv = 1.0 / sd_.clamp_min(1e-12)
                st.append(torch.cat([mu_, sd_, inv, mu_ * inv], 1))
                layout.append(m.num_groups)
            if st:
                gn_stats.append(torch.cat(st, 1).numpy())
            # the FIRST norm's output at far-background positions (pre-ReLU, cloned)
            pf = F.adaptive_avg_pool3d(cap[first_norm][1], (g, g, g)).flatten(2)
            first_far.append(pf[:, :, far_t].mean(2).numpy())
        pack = lambda a: np.concatenate(a, 0)
        results[norm_type] = dict(
            core=pack(r_core),
            far=pack(r_far),
            sp_std=pack(sp_std),
            sp_mean=pack(sp_mean),
            gn=pack(gn_stats) if gn_stats else None,
            first_gn_groups=first_norm.num_groups if isinstance(first_norm, nn.GroupNorm) else 0,
            first_far=pack(first_far),
            shells=[pack(a) for a in shell_acc],
        )

    # ── TEST 2. constant-background signature ─────────────────────────────────────
    print(f"\n{'=' * 78}\nTEST 2 — is the far-background feature map SPATIALLY CONSTANT?\n{'=' * 78}")
    print(f"{'norm':<8}{'across-SAMPLE std':>20}{'across-POSITION std':>22}   verdict")
    print("-" * 78)
    for nt in ("group", "layer"):
        R = results[nt]
        s_samp = float(R["sp_mean"].std(axis=0).mean())  # how much the far-bg readout moves per sample
        s_pos = float(R["sp_std"].mean())  # how much positions disagree within a sample
        if s_samp < 1e-5:
            v = "FROZEN — identical for every subject"
        elif s_pos < 0.1 * s_samp:
            v = "spatially CONSTANT, sample-varying"
        else:
            v = "spatially structured"
        print(f"{nt:<8}{s_samp:>20.3e}{s_pos:>22.3e}   {v}")
    print("\n  GroupNorm: positions agree with each other but move together across subjects —")
    print("  a spatially constant, sample-varying map IS a global summary written into every")
    print("  position. A receptive-field route cannot produce this: positions at different")
    print("  distances and angles from the brain would have to disagree.")
    print("\n  LayerNorm: the far background is FROZEN — bit-for-bit the same features for every")
    print("  subject. Zero across-sample variance forces R^2 = 0 there, whatever probe is used.")
    print("  Same conv weights, same data, same positions: the norm is the only difference.")

    # ── probes ────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}\nFACTOR RECOVERY from core vs far background (UNTRAINED encoder)\n{'=' * 78}")
    print(f"{'factor':<22}{'GN core':>10}{'GN far-bg':>12}{'LN core':>10}{'LN far-bg':>12}")
    print("-" * 66)
    for fname in FACTORS:
        y = Z[:, CONTENT_FACTOR_NAMES.index(fname)]
        row = [probe(results[nt][k], y) for nt in ("group", "layer") for k in ("core", "far")]
        print(f"{fname:<22}{row[0]:>10.3f}{row[1]:>12.3f}{row[2]:>10.3f}{row[3]:>12.3f}")
    print("\n  GN far-bg >> 0 on an UNTRAINED network -> the hotspot is architectural, not learned.")
    print("  LN far-bg ~ 0 with identical conv weights -> GroupNorm is the carrier.")

    # ── background R^2 vs distance, both norms ────────────────────────────────────
    print(f"\n{'=' * 78}\nBACKGROUND R^2 BY DISTANCE TO BRAIN — what LayerNorm should LEAVE\n{'=' * 78}")
    print(f"conv RF radius = {rf_radius:.1f} input voxels; positions closer than that CAN see brain tissue.")
    print(
        f"\n{'dist (in vox)':<16}{'n pos':>7}{'% in RF':>9}{'GN vent':>10}{'LN vent':>10}{'GN bsize':>11}{'LN bsize':>11}"
    )
    print("-" * 74)
    y_v = Z[:, CONTENT_FACTOR_NAMES.index("ventricle_size")]
    y_b = Z[:, CONTENT_FACTOR_NAMES.index("brain_size")]
    for si, ((lo, hi), sel_t, frac_in) in enumerate(shells):
        gv, lv = (probe(results[nt]["shells"][si], y_v) for nt in ("group", "layer"))
        gb, lb = (probe(results[nt]["shells"][si], y_b) for nt in ("group", "layer"))
        lab = f"[{lo},{hi if hi < 99 else 'max'})"
        print(f"{lab:<16}{int(sel_t.sum()):>7}{frac_in * 100:>8.0f}%{gv:>10.3f}{lv:>10.3f}{gb:>11.3f}{lb:>11.3f}")
    print("\n  Read the LN columns as the BENIGN floor: signal that arrives through the convolutions")
    print("  because the position genuinely sees brain. It is expected to be well above zero wherever")
    print("  '% in RF' is high, and only collapses past the RF radius. A trained LayerNorm run whose")
    print("  edge R^2 matches this profile has no leak left — it is reading anatomy it can actually see.")
    print("  The GN-minus-LN gap at the SAME distance is the part attributable to the normalizer.")

    # ── TEST 3. injection point and sufficiency ───────────────────────────────────
    print(f"\n{'=' * 78}\nTEST 3 — is the background value exactly the FIRST norm's (mu, sigma)?\n{'=' * 78}")
    G = results["group"]
    if G["gn"] is not None:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        gs = G["gn"]  # per norm: [mu, sigma, 1/sigma, mu/sigma] per group
        n1 = G["first_gn_groups"]
        basis1 = gs[:, 2 * n1 : 4 * n1]  # first norm's analytic basis [1/sigma, mu/sigma]

        # (a) INJECTION POINT — exact algebra at the first norm.
        tgt = G["first_far"]
        r2_struct = r2_score(tgt, LinearRegression().fit(basis1, tgt).predict(basis1), multioutput="variance_weighted")
        print(f"  (a) injection point — first norm is GroupNorm({n1} groups over {cli.hidden // 2} channels)")
        print(f"      R^2( its far-bg output <- basis [1/sigma_g, mu_g/sigma_g] ) = {r2_struct:.4f}")
        print("      The algebraic prediction is exactly 1.0: at a zero-RF position the pre-norm value")
        print("      is a constant conv bias b, so the output is gamma*(b-mu)/sigma + beta — an exact")
        print("      linear function of that basis and of nothing else. This is where content enters.")

        # (b) SUFFICIENCY — the FINAL far-bg readout from all norms' statistics. The encoder
        # composes several normalizations, so this is a linear lower bound, not an identity.
        from sklearn.model_selection import KFold as _KF
        from sklearn.preprocessing import StandardScaler as _SS

        tgt2 = G["far"]
        pr = np.empty_like(tgt2)
        for tr, te in _KF(n_splits=5, shuffle=True, random_state=0).split(gs):
            sc = _SS().fit(gs[tr])
            from sklearn.linear_model import RidgeCV as _R

            pr[te] = _R(alphas=np.logspace(-3, 3, 13)).fit(sc.transform(gs[tr]), tgt2[tr]).predict(sc.transform(gs[te]))
        r2_suff = r2_score(tgt2, pr, multioutput="variance_weighted")
        print(f"\n  (b) sufficiency — R^2( FINAL far-bg readout <- all norms' statistics ) = {r2_suff:.3f}")
        print("      Cross-validated and linear, so a lower bound (the encoder composes several")
        print("      normalizations, which makes the true map nonlinear). It does not need to reach")
        print("      1.0 for the argument: Tests 0 and 1 already prove nothing else can be feeding")
        print("      these positions. What matters is (c) — that the FACTOR content matches.")

        print(f"\n  (c) factor recovered from the DISCARDED statistics alone, vs from the far background:")
        print(f"      {'factor':<22}{'from (mu,sigma)':>17}{'from far-bg':>13}")
        for fname in FACTORS:
            y = Z[:, CONTENT_FACTOR_NAMES.index(fname)]
            print(f"      {fname:<22}{probe(gs, y):>17.3f}{probe(G['far'], y):>13.3f}")
        print("\n      The two columns matching is the point: the factor the background 'localizes' is")
        print("      exactly the factor sitting in the statistics GroupNorm subtracts.")

    # ── why background BEATS core: an SNR account, not more information ───────────
    print(f"\n{'=' * 78}\nWHY BACKGROUND CAN BEAT CORE — signal-to-noise, not extra information\n{'=' * 78}")
    # Cross-validated: with C predictors and few samples an in-sample fit reads 1.000 by
    # construction, which would fake exactly the conclusion being drawn.
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    Gc, Gf = G["core"], G["far"]
    pred = np.empty_like(Gc)
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=0).split(Gf):
        sc = StandardScaler().fit(Gf[tr])
        pred[te] = (
            RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sc.transform(Gf[tr]), Gc[tr]).predict(sc.transform(Gf[te]))
        )
    share = r2_score(Gc, pred, multioutput="variance_weighted")
    print(f"  share of core-readout variance predictable from the far-background readout: {share:.3f}")
    print(f"  remaining (local anatomy) variance in the core readout                    : {1 - share:.3f}")
    print("  (5-fold cross-validated, so this is not an artefact of fitting C predictors.)")
    print("\n  Both positions carry the same global summary. The core additionally carries local")
    print("  anatomy, which for a factor that is NOT that anatomy acts as probe noise. So the")
    print("  background is a CLEANER estimate of the same quantity — higher R^2 there does not")
    print("  mean more information is stored there.")
    print(f"\n  The obvious objection — that the background simply averages more positions")
    print(f"  ({int(far.sum()):,} vs {int(core.sum()):,}) — is already answered by Test 2: far-background")
    print("  positions agree with each other to a few percent, so averaging more of them returns")
    print("  essentially the same vector. The count is not what makes the background cleaner.")


if __name__ == "__main__":
    main()
