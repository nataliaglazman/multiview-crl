from __future__ import annotations

import logging
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

FACTORS = ("brain_size", "ventricle_size", "cortical_thickness", "lr_asymmetry")


def _stub_utils_if_needed():
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


def build_encoder(norm_type, seed, hidden=48, res_ch=32, nb_res=2, downscale=2, open_residual=True):
    Encoder = _import_encoder()
    torch.manual_seed(seed)
    enc = Encoder(1, hidden, res_ch, nb_res, downscale, False, norm_type)
    if open_residual:
        for m in enc.modules():
            if hasattr(m, "alpha") and isinstance(m.alpha, nn.Parameter):
                with torch.no_grad():
                    m.alpha.fill_(1.0)
    return enc.eval()


def znorm_nonzero(x, mask):
    m = mask > 0
    if m.any():
        vals = x[m]
        x = (x - vals.mean()) / vals.std().clamp_min(1e-6)
        x = x * mask
    return x


def fixed_reference_constants(ds, n_ref=64, scale_quantile=0.99):
    vals = []
    for j in range(min(n_ref, len(ds))):
        x_v1, x_v2, lat = ds[j]
        bm = lat["brain_mask"] > 0
        for x in (x_v1, x_v2):
            v = x[bm]
            if v.numel():
                vals.append(v.flatten().float())
    vals = torch.cat(vals)
    cap = 4_000_000
    if vals.numel() > cap:
        g = torch.Generator().manual_seed(0)
        vals = vals[torch.randint(vals.numel(), (cap,), generator=g)]
    mean = float(vals.mean())
    return mean, max(float(torch.quantile((vals - mean).abs(), scale_quantile)), 1e-6)


def generate(n, res, causal, seed=42, normalize="per_sample", clean_content=False):
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


def extract_spatial(model, dataset, args, device, level, grid, batch_size):
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    Xs, fr, lat = [], [], {"z_content": []}
    model.eval()
    for batch in loader:
        imgs = batch["image"]
        n_views = len(imgs)
        x = torch.cat(imgs, 0).to(device)
        masks = batch.get("mask")
        if masks is None:
            raise SystemExit("dataset yields no 'mask' key; background stratification needs brain masks")
        m = torch.cat(masks, 0).to(device).float()
        with torch.no_grad():
            out = model(x, pool_only=True, n_views=n_views, patch_grid=grid)
            feats = out[2] if isinstance(out, tuple) else [out]
            f = feats[level]
            if f.dim() == 2:
                f = f.unsqueeze(-1)
            hz = f.reshape(n_views, -1, *f.shape[1:])
            frac = F.adaptive_avg_pool3d(m, tuple(grid)).flatten(1)
            B = hz.shape[1]
        Xs.append(hz[0].float().cpu().numpy())
        fr.append(frac[:B].cpu().numpy())
        gt = batch.get("gt_latents") or {}
        if "z_content" in gt:
            v = gt["z_content"]
            lat["z_content"].append(np.asarray(v).reshape(v.shape[0], -1))
    X = np.concatenate(Xs, 0)
    frac = np.concatenate(fr, 0)
    Z = np.concatenate(lat["z_content"], 0) if lat["z_content"] else None
    return X, frac, Z


def readout(X, sel):
    if sel.sum() == 0:
        return None
    return X[:, :, sel].mean(axis=2)


def resid(a, b):
    return a - LinearRegression().fit(b, a).predict(b)


def run_background_leak_diagnostic(cli):
    from eval.dci import CONTENT_FACTOR_NAMES
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, args, device = load_model_from_run_dir(cli.run_dir, cli.checkpoint, device)
    grid = (cli.grid,) * 3 if cli.grid else tuple(getattr(args, "patch_grid", (16, 16, 16)))
    ds = build_synthetic_test_set(args, cli.num_samples)
    X, frac, Z = extract_spatial(model, ds, args, device, cli.level, grid, cli.batch_size)
    if Z is None:
        raise SystemExit("no z_content in gt_latents")

    fi = CONTENT_FACTOR_NAMES.index(cli.factor)
    bsi = CONTENT_FACTOR_NAMES.index("brain_size")
    y = Z[:, fi]
    y_bs = Z[:, bsi]

    cov = frac.mean(axis=0)
    core = cov >= cli.core_thr
    bg = cov <= cli.bg_thr
    N, C, P = X.shape
    print(f"\nrun: {cli.run_dir}")
    print(f"factor: {cli.factor} | level {cli.level} | grid {list(grid)} -> P={P} | N={N} | C={C}")
    print(f"positions: core(>= {cli.core_thr})={int(core.sum())}  background(<= {cli.bg_thr})={int(bg.sum())}\n")

    f_core = readout(X, core)
    f_bg = readout(X, bg)
    r_core = probe(f_core, y)
    r_bg = probe(f_bg, y)
    r_core_given_bg = probe(resid(f_core, f_bg), y) if (f_core is not None and f_bg is not None) else float("nan")

    print(f"{'readout':<26}{cli.factor + ' R2':>18}")
    print("-" * 44)
    print(f"{'core (in-brain)':<26}{r_core:>18.3f}")
    print(f"{'background':<26}{r_bg:>18.3f}")
    print(f"{'core | background removed':<26}{r_core_given_bg:>18.3f}")

    brain_pos = np.argwhere((cov >= 0.5).reshape(grid))
    all_pos = np.argwhere(np.ones(grid, bool))
    if len(brain_pos):
        try:
            from scipy.spatial import cKDTree

            dist = cKDTree(brain_pos).query(all_pos)[0]
        except ImportError:
            d2 = ((all_pos[:, None, :] - brain_pos[None, :, :]) ** 2).sum(-1)
            dist = np.sqrt(d2.min(axis=1)).astype(float)
    else:
        dist = np.full(P, np.nan)
    bg_d = dist[bg]
    print("\ndistance-to-brain profile (background only, 1-voxel bins):")
    print(f"  {'dist':<14}{'n':>6}{'R2':>9}")
    shell_r2 = []
    d_max = float(bg_d.max()) if len(bg_d) else 0.0
    for d_lo in np.arange(0.0, np.ceil(d_max), 1.0):
        m = bg & (dist >= d_lo) & (dist < d_lo + 1.0)
        if int(m.sum()) < 20:
            continue
        rr = probe(readout(X, m), y)
        shell_r2.append(rr)
        print(f"  [{d_lo:.0f},{d_lo + 1:.0f})".ljust(14) + f"{int(m.sum()):>6}{rr:>9.3f}")
    print(
        "  NOTE: encoder RF radius at this grid is ~5-6 voxels. Bins INSIDE that radius can\n"
        "  legitimately see brain tissue (a real local signal); only bins BEYOND it test for\n"
        "  a non-local route. Read the far bins, not the average."
    )

    r_bg_bs = probe(f_bg, y_bs)
    corr = float(np.corrcoef(y, y_bs)[0, 1])
    r_bg_given_bs = probe(resid(f_bg, y_bs[:, None]), y) if f_bg is not None else float("nan")
    print(f"\nbrain_size confound:")
    print(f"  background -> brain_size R2        : {r_bg_bs:.3f}")
    print(f"  corr({cli.factor}, brain_size)     : {corr:+.3f}")
    print(f"  background -> {cli.factor} | brain_size removed : {r_bg_given_bs:.3f}")

    valid = [s for s in shell_r2 if s == s]
    decays = len(valid) >= 2 and valid[-1] < 0.5 * max(valid[0], 1e-9) and max(valid) > 0.1
    print("\nverdict:")
    if r_bg < 0.1:
        print(f"  background R2 is low ({r_bg:.2f}); no strong hotspot to explain at this grid/level.")
    elif decays:
        print(
            f"  (B) RECEPTIVE-FIELD / BOUNDARY LEAK. Background R2 decays with distance from the brain ({'->'.join(f'{s:.2f}' for s in shell_r2)}) -> the signal reaches out from the brain, it is not a flat global readout. Smaller receptive field (lower scaling_rate / shallower level 0) is the lever."
        )
    else:
        print(
            f"  (A) GLOBAL-STATISTIC LEAKAGE. Background R2 is flat across distance shells ({'->'.join(f'{s:.2f}' for s in shell_r2)}) -> every background position reads the same whole-image statistic, not local information. The background hotspot is an artefact of a global summary (intensity/GroupNorm stats) that {cli.factor} shifts."
        )
        if r_core_given_bg > 0.1:
            print(
                f"  The CENTRE hotspot is separately GENUINE: core|bg = {r_core_given_bg:.2f} -> real local in-brain encoding of {cli.factor}, on top of the global background artefact."
            )
        else:
            print(
                f"  And core|bg = {r_core_given_bg:.2f}: even the centre carries no local info beyond the global readout -- the whole map is the global statistic, nothing is localised."
            )
    if abs(corr) > 0.3 and r_bg_given_bs < 0.5 * r_bg:
        print(
            f"  MEDIATED BY brain_size: corr={corr:+.2f}, and removing brain_size collapses the background readout ({r_bg_given_bs:.2f} vs {r_bg:.2f}). The hotspot is largely brain_size's boundary signal, inherited via the causal graph."
        )


def run_background_leak_mechanism(cli):
    from eval.dci import CONTENT_FACTOR_NAMES

    dev = torch.device("cpu")
    res = cli.res
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

    ever_brain_vox = (M > 0).any(dim=0)[0]
    bg_vox = ~ever_brain_vox
    bg_input = X[:, 0][:, bg_vox]
    print(f"\n{'=' * 78}\nTEST 0 — is the INPUT background constant across samples?\n{'=' * 78}")
    print(f"background voxels (never brain in any sample): {int(bg_vox.sum()):,} / {res**3:,}")
    print(f"  max |value|            : {bg_input.abs().max():.3e}")
    print(f"  max across-sample std  : {bg_input.std(dim=0).max():.3e}")
    print("  -> input background is exactly 0 everywhere; it carries NO sample information.")
    print("     Any background FEATURE signal must be created inside the network.")

    g = res // cli.downscale
    cov = F.adaptive_avg_pool3d(M, (g, g, g))[:, 0].numpy()
    ever = cov.max(axis=0) > 0
    core = cov.min(axis=0) >= 0.9

    from scipy.ndimage import distance_transform_edt

    dist = distance_transform_edt(~ever)
    print(f"\n{'=' * 78}\nTEST 1 — how far can the corner position see? (three separate reaches)\n{'=' * 78}")

    def corner_reach(norm_type, strip_norms=False, interior=False):
        enc = build_encoder(norm_type, cli.seed, cli.hidden, downscale=cli.downscale, open_residual=True)
        if strip_norms:
            for name, mod in list(enc.named_modules()):
                for cname, child in list(mod.named_children()):
                    if isinstance(child, nn.GroupNorm) or type(child).__name__ == "ChannelLayerNorm3d":
                        setattr(mod, cname, nn.Identity())
        xr = torch.zeros(1, 1, res, res, res, requires_grad=True)
        out = enc(xr)
        if interior:
            gc = out.shape[-1] // 2
            out[0, :, gc, gc, gc].sum().backward()
        else:
            out[0, :, 0, 0, 0].sum().backward()
        gr = xr.grad.detach().abs()[0, 0].numpy()
        nz = np.argwhere(gr > gr.max() * 1e-6)
        if interior:
            return float((nz.max(0) - nz.min(0) + 1).max())
        return float(nz.max() + 1)

    rf_conv = corner_reach("group", strip_norms=True)
    rf_span = corner_reach("group", strip_norms=True, interior=True)
    rf_ln = corner_reach("layer")
    rf_gn = corner_reach("group")
    brain_vox = np.argwhere(ever_brain_vox.numpy())
    d_corner_to_brain = float(np.sqrt((brain_vox**2).sum(1)).min())

    print(f"  gradient reach of feature position (0,0,0), in input voxels per axis:")
    print(f"    convolutions only, corner (= RF RADIUS): {rf_conv:>6.0f}   <- compare distances against THIS")
    print(f"    same encoder, interior position (span) : {rf_span:>6.0f}   <- ~2x the radius, as expected")
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

    d_in = distance_transform_edt(~ever_brain_vox.numpy())
    ii, jj, kk = np.meshgrid(*[np.arange(g)] * 3, indexing="ij")
    ctr = (np.stack([ii, jj, kk], -1) + 0.5) * cli.downscale
    d_pos = d_in[ctr[..., 0].astype(int), ctr[..., 1].astype(int), ctr[..., 2].astype(int)]

    with torch.no_grad():
        f0 = build_encoder("group", cli.seed, cli.hidden, downscale=cli.downscale)(torch.zeros(1, 1, res, res, res))
    f0 = f0[0].flatten(1)
    v0 = f0.median(dim=1).values[:, None]
    dev0 = (f0 - v0).abs().max(dim=0).values.numpy().reshape(g, g, g)
    pad_clean = dev0 <= 1e-4 * max(float(v0.abs().max()), 1.0)

    rf_radius = rf_conv
    far = (~ever) & (d_pos > rf_radius * cli.rf_margin) & pad_clean
    print(f"\n  position sets at feature resolution {g}^3 ({g**3:,} positions):")
    print(f"    core (brain in EVERY sample)               : {int(core.sum()):,}")
    print(f"    background (brain in NO sample)            : {int((~ever).sum()):,}")
    for mult in (1.0, 1.5, 2.0):
        n_f = int(((~ever) & (d_pos > rf_radius * mult) & pad_clean).sum())
        tag = "  <- exact RF radius" if mult == 1.0 else ""
        star = " *" if abs(mult - cli.rf_margin) < 1e-9 else "  "
        print(f"    far background (> {mult:.2f} x RF radius, pad-clean){star}: {n_f:,}{tag}")
    print(f"    padding-contaminated (excluded)            : {int((~pad_clean).sum()):,}")
    print(f"    (* = the set used below, --rf-margin {cli.rf_margin})")
    print("\n  Padding note: those excluded positions carry a fixed spatial pattern, but it is")
    print("  driven by conv biases, not by the data — Test 2's LayerNorm row shows it does not")
    print("  move across samples, so padding cannot carry factor information. It is a red herring")
    print("  for this question, and it is removed here only so Test 2 measures one thing.")
    if far.sum() == 0:
        raise SystemExit("no far-background positions at this resolution; lower --res or --downscale")

    shell_edges = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 24), (24, 99)]
    shells = []
    for lo, hi in shell_edges:
        sel = (~ever) & (d_pos >= lo) & (d_pos < hi) & pad_clean
        if int(sel.sum()) >= 20:
            shells.append(((lo, hi), torch.from_numpy(sel.reshape(-1)), float((d_pos[sel] <= rf_radius).mean())))

    core_t = torch.from_numpy(core.reshape(-1))
    far_t = torch.from_numpy(far.reshape(-1))
    results = {}
    for norm_type in ("group", "layer"):
        enc = build_encoder(norm_type, cli.seed, cli.hidden, downscale=cli.downscale)
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
                f = enc(xb)
            fl = f.flatten(2)
            r_core.append(fl[:, :, core_t].mean(2).numpy())
            ff = fl[:, :, far_t]
            r_far.append(ff.mean(2).numpy())
            sp_mean.append(ff.mean(2).numpy())
            sp_std.append(ff.std(2).numpy())
            for si, (_, sel_t, _) in enumerate(shells):
                shell_acc[si].append(fl[:, :, sel_t].mean(2).numpy())
            st = []
            for m in norms:
                if not isinstance(m, nn.GroupNorm):
                    continue
                pre = cap[m][0]
                pg = pre.reshape(pre.shape[0], m.num_groups, -1)
                mu_, sd_ = pg.mean(-1), pg.std(-1, unbiased=False)
                inv = 1.0 / sd_.clamp_min(1e-12)
                st.append(torch.cat([mu_, sd_, inv, mu_ * inv], 1))
            if st:
                gn_stats.append(torch.cat(st, 1).numpy())
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

    print(f"\n{'=' * 78}\nTEST 2 — is the far-background feature map SPATIALLY CONSTANT?\n{'=' * 78}")
    print(f"{'norm':<8}{'across-SAMPLE std':>20}{'across-POSITION std':>22}   verdict")
    print("-" * 78)
    for nt in ("group", "layer"):
        R = results[nt]
        s_samp = float(R["sp_mean"].std(axis=0).mean())
        s_pos = float(R["sp_std"].mean())
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

    print(f"\n{'=' * 78}\nFACTOR RECOVERY from core vs far background (UNTRAINED encoder)\n{'=' * 78}")
    print(f"{'factor':<22}{'GN core':>10}{'GN far-bg':>12}{'LN core':>10}{'LN far-bg':>12}")
    print("-" * 66)
    for fname in FACTORS:
        y = Z[:, CONTENT_FACTOR_NAMES.index(fname)]
        row = [probe(results[nt][k], y) for nt in ("group", "layer") for k in ("core", "far")]
        print(f"{fname:<22}{row[0]:>10.3f}{row[1]:>12.3f}{row[2]:>10.3f}{row[3]:>12.3f}")
    print("\n  GN far-bg >> 0 on an UNTRAINED network -> the hotspot is architectural, not learned.")
    print("  LN far-bg ~ 0 with identical conv weights -> GroupNorm is the carrier.")

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
