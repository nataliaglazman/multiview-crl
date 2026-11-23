"""Counterfactual images from tabular data. A tabular variable causes one image factor; change it, move the frozen code
by the change's estimated effect, decode the code, and score the decoded change against the generator's own
counterfactual image of the same subject.

    python out/burden_feasibility_20261102/counterfactual_images.py RUN OUT_DIR [--targets ventricle ...]
        [--delta 3] [--rho 0.7] [--subjects 200] [--views t1 flair] [--epochs 150] [--seed 0]

The tabular cause is the tabular benchmark's (eval/causal/TABULAR_BENCHMARK.md): T = rho u + sqrt(1 - rho^2) e, with u
the target factor's standardized own noise and e fresh noise. Read as the SCM T -> u, raising T by delta (in T's SDs)
raises u by rho delta for every subject, with each subject's e kept (abduction). The true counterfactual latents are the
bank's SCM run forward from the subject's own noise with the target's moved, so the target and its descendants change.
The true counterfactual images are the generator's render of those latents with every other draw (anatomy fields,
style, noise, the lesion placement rule) the subject's own, normalized and given a normative residual exactly as the
bank's were. The factual subject is rendered the same way and must reproduce the bank's stored images.

Learned on the training subjects:
  - which factors T reaches directly: PC (Fisher-Z, alpha .05) on T beside the seven factors read from the code by
    ridge, the ridge fitted on one half of the training subjects and PC run on the other. T's neighbours are its direct
    effects; labels name the factors, they do not build the code;
  - T's total effect on the code. T is exogenous, so the slope of the code on T is its causal effect, and each test
    subject's counterfactual code is its code plus slope x delta (the change is the same for every subject);
  - per view, a decoder from the code to both channels of the view's input (recon_check.py's decoder and schedule), and
    the same decoder from the seven true factors, the oracle that knows the true counterfactual latents. Decoders are
    saved in OUT_DIR and reused.

Scores on the chosen test subjects whose counterfactual anatomy still holds the lesion, per target, view and channel
(scan, residual), over the brain: the share of the true change reproduced, 1 - sum((predicted - true)^2) / sum(true^2),
the mean cosine between each subject's predicted and true change, and the predicted change's size over the true one.
  code, estimated effect   decode(counterfactual code) - decode(code)
  code, true codes         decode(code of the true counterfactual) - decode(code of the true factual): the decoder alone
  truth (oracle)           the true-factor decoder's change between the true factual and counterfactual latents
Also the code's own change (estimated against the code of the true counterfactual image), and each factor's mean change
read by the ridge readout from the counterfactual codes, against the factor's true mean change in its own SDs.
"""

import argparse
import json
import warnings
from pathlib import Path

import bank as B
import code_audit
import numpy as np
import own_variation
import recon_check as R
import residual_bank
import torch
import warm
from sklearn.linear_model import LinearRegression, RidgeCV

from eval.causal.latent_causal_discovery import discover
from eval.synthetic.synthetic_dataset import LesionPlacementError

NAMES = list(B.CAUSAL_FACTORS.values())
COLUMNS = list(B.CAUSAL_FACTORS)  # z_content indices of the seven factors
ARMS = ("code, estimated effect", "code, true codes", "truth (oracle)")


def mechanism(signal, nonlinearity):
    if nonlinearity == "leaky_relu":
        return np.where(signal > 0, signal, 0.2 * signal)
    if nonlinearity == "tanh":
        return np.tanh(signal)
    return signal


def exogenous(z, scm, scale, nonlinearity):
    """Every dimension's exogenous noise: sample_content_from_scm inverted (a root is its own noise)."""
    noise = np.empty_like(z)
    for d in range(scm["n_dims"]):
        parents = scm["parents"][d]
        if parents:
            signal = z[:, parents] @ np.asarray(scm["weights"][d], dtype=np.float64)
            noise[:, d] = (z[:, d] - mechanism(signal, nonlinearity)) / scale
        else:
            noise[:, d] = z[:, d]
    return noise


def structural(noise, scm, scale, nonlinearity):
    """sample_content_from_scm run forward from given noise; its edges run from lower to higher index."""
    z = np.empty_like(noise)
    for d in range(scm["n_dims"]):
        parents = scm["parents"][d]
        if parents:
            signal = z[:, parents] @ np.asarray(scm["weights"][d], dtype=np.float64)
            z[:, d] = mechanism(signal, nonlinearity) + scale * noise[:, d]
        else:
            z[:, d] = noise[:, d]
    return z


def bank_setting(root):
    """bank.py's options for ``root``, read from its folder name; the factual re-render checks them."""
    tokens = Path(root).name.split("_")
    components = [int(t[1:]) for t in tokens if t[:1] == "k" and t[1:].isdigit()]
    return dict(
        causal="causal" in tokens,
        field="off" if "nofield" in tokens else "smooth" if "smoothfield" in tokens else "lesion",
        gain="off" if "nogain" in tokens else "on",
        position="fixed" if "fixedpos" in tokens else "random",
        components=components[0] if components else residual_bank.K,
    )


class Generator:
    """The bank's generator for its test subjects, set as bank.py set it: re-renders a subject with its content latents
    replaced, normalized by the training split's fixed reference and given the bank's normative residual."""

    def __init__(self, root, t1key, setting):
        if not setting["causal"]:
            raise SystemExit(f"{root} is not a causal bank: there is no SCM to run a counterfactual through")
        args = B.run_args({key: value for value, key in B.T1_KEYS.items()}[t1key], causal=True)
        sizes = {s: len(np.load(f"{root}/{s}_zc.npy", mmap_mode="r")) for s in ("train", "test")}
        datasets = []
        for split in ("train", "test"):
            ds = B.T.make_dataset(args, split, sizes[split])
            ds._inner.scm = B.causal_scm(args.seed)
            B.set_field(ds._inner.renderer, setting["field"])
            B.set_gain(ds._inner.renderer, setting["gain"])
            B.set_position(ds._inner.renderer, setting["position"])
            ds[0]
            datasets.append(ds)
        self.ds = datasets[1]
        self.ds._fixed_mean, self.ds._fixed_scale = datasets[0]._fixed_mean, datasets[0]._fixed_scale
        self.accepted = {}
        self.keys = (t1key, "flair")
        sample = np.asarray(np.load(f"{root}/train_flair.npy", mmap_mode="r")[:200], dtype=np.float32)
        self.mask = (sample != 0).mean(0) >= 0.5
        self.normative = {}
        for key in self.keys:
            data = np.load(f"{root}/basis_{key}.npz")
            mean, basis = data["mean"], data["basis"][:, : setting["components"]]
            train = np.load(f"{root}/train_{key}.npy", mmap_mode="r")
            scale = float(np.std([residual_bank.remove_anatomy(train[i], self.mask, mean, basis) for i in range(200)]))
            self.normative[key] = (mean, basis, scale)

    def residual(self, scan, key):
        mean, basis, scale = self.normative[key]
        out = np.zeros(self.mask.shape, dtype=np.float32)
        out[self.mask] = residual_bank.remove_anatomy(scan, self.mask, mean, basis) / scale
        return out.astype(np.float16)

    def _pair(self, seed, draw, content):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.ds._inner.render_pseudo_mri(
                content,
                draw["z_deformation"],
                draw["z_fissure"],
                draw["z_style_v1"],
                draw["z_style_v2"],
                seed,
                z_lesion=draw["z_lesion"],
            )

    def render(self, index, shift=None):
        """Subject ``index``'s (scan, residual) per view as float16, with ``shift`` added to its content latents; None
        when the shifted anatomy cannot hold the lesion. The factual render goes first: it finds the candidate the bank
        accepted (a subject whose first draw had no room for the lesion was redrawn), which every shift then reuses."""
        if index not in self.accepted:
            seed, draw, out = self.ds._inner._first_fitting(index, lambda s, d: self._pair(s, d, d["z_content"]))
            self.accepted[index] = (seed, draw)
        else:
            seed, draw = self.accepted[index]
            out = None
        if shift is not None:
            try:
                out = self._pair(seed, draw, draw["z_content"] + torch.as_tensor(shift, dtype=draw["z_content"].dtype))
            except LesionPlacementError:
                return None
        elif out is None:
            out = self._pair(seed, draw, draw["z_content"])
        x1, x2, mask = out
        x1, x2 = self.ds.normalize_views(x1, x2, mask, mask.clone())
        out = {}
        for key, x in zip(self.keys, (x1, x2)):
            scan = x[0].cpu().numpy().astype(np.float16)
            out[key] = np.stack([scan, self.residual(scan, key)])
        return out


def coded_bank(images, keys, uses_residual):
    """A warm.Bank over rendered arrays, so warm.features encodes them as it encodes a bank."""
    bank = object.__new__(warm.Bank)
    bank.t1, bank.flair = images[keys[0]][:, 0], images[keys[1]][:, 0]
    bank.residuals = (images[keys[0]][:, 1], images[keys[1]][:, 1]) if uses_residual else (None, None)
    bank.vox, bank.zc = np.zeros(len(bank.t1)), np.zeros((len(bank.t1), 9))
    return bank


def decoder_for(out, view, label, inputs, images, stats, device, epochs, log):
    """A decoder from ``inputs`` (train, val) to the view's two channels, trained once and saved in ``out``."""
    path = out / f"decoder_{view}_{label}.pt"
    mean, scale, mask = stats
    mu, sd = inputs["train"].mean(0), inputs["train"].std(0) + 1e-6
    decoder = R.Decoder(inputs["train"].shape[1]).to(device)
    if path.exists():
        decoder.load_state_dict(torch.load(path, map_location=device))
        log(f"  {view} decoder from {label}: reused {path.name}")
    else:
        log(f"  {view} decoder from {label}:")
        z = {**inputs, "test": inputs["val"][:1]}
        decoder, _, _, epochs_run = R.fit_decoder(z, images, mean, scale, mask, device, epochs, log)
        torch.save(decoder.state_dict(), path)
        log(f"    {epochs_run} epochs, saved {path.name}")
    decoder.eval()

    @torch.no_grad()
    def decode(x, batch=16):
        rows = []
        for start in range(0, len(x), batch):
            z = torch.from_numpy(((x[start : start + batch] - mu) / sd).astype(np.float32)).to(device)
            rows.append((decoder(z).cpu().numpy() * scale[None, :, None, None, None]).astype(np.float32))
        return np.concatenate(rows) + mean[None]

    return decode


def change_scores(predicted, true, mask):
    """Share of the true change reproduced, mean per-subject cosine, and size over true size, over ``mask``."""
    p, t = predicted[:, mask].astype(np.float64), true[:, mask].astype(np.float64)
    norms = np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1)
    cosine = np.where(norms > 0, (p * t).sum(1) / np.where(norms > 0, norms, 1), 0)
    return dict(
        share=float(1 - ((p - t) ** 2).sum() / max((t**2).sum(), 1e-12)),
        cosine=float(cosine.mean()),
        size=float(np.sqrt((p**2).sum() / max((t**2).sum(), 1e-12))),
    )


def figure(path, title, factual, counterfactual, changes, subjects, ids):
    """Rows: subjects (the axial slice where the true scan change is largest), labelled by their test index ``ids``;
    columns: factual and counterfactual scans, then the true and each arm's predicted scan change."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grey = plt.get_cmap("gray").copy()
    grey.set_bad("black")
    labels = ["factual", "counterfactual", "true change", *changes]
    fig, axes = plt.subplots(
        len(subjects), len(labels), figsize=(1.9 * len(labels), 2.0 * len(subjects)), squeeze=False
    )
    for r, (i, subject) in enumerate(zip(subjects, ids)):
        true_change = counterfactual[i, 0].astype(np.float32) - factual[i, 0].astype(np.float32)
        z = int(np.abs(true_change).sum((0, 1)).argmax())
        head = factual[i, 0] != 0
        lo, hi = np.percentile(factual[i, 0][head].astype(np.float32), (1, 99))
        lim = float(np.percentile(np.abs(true_change[head]), 99.5)) or 1.0
        panels = [factual[i, 0], counterfactual[i, 0], true_change, *[c[i, 0] for c in changes.values()]]
        for c, (label, vol) in enumerate(zip(labels, panels)):
            ax = axes[r, c]
            if c < 2:
                img = np.where(head, vol.astype(np.float32), np.nan)[:, :, z].T
                ax.imshow(img, origin="lower", cmap=grey, vmin=lo, vmax=hi)
            else:
                ax.imshow(vol[:, :, z].T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(label, fontsize=7)
            if c == 0:
                ax.set_ylabel(f"#{subject}, z={z}", fontsize=7)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--targets", nargs="+", default=["ventricle"], choices=NAMES)
    p.add_argument("--delta", type=float, default=3.0, help="The tabular cause's change, in its SDs")
    p.add_argument("--rho", type=float, default=0.7, help="The cause's correlation with its target's own noise")
    p.add_argument("--subjects", type=int, default=200, help="Test subjects given counterfactual images")
    p.add_argument("--views", nargs="+", default=list(R.VIEWS), choices=R.VIEWS)
    p.add_argument("--epochs", type=int, default=R.EPOCHS)
    p.add_argument("--train-subjects", type=int, default=None, help="Cap the training subjects (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="Default: warm.pick_device()")
    a = p.parse_args()
    if not 0 < abs(a.rho) < 1:
        p.error("--rho must lie strictly between -1 and 1, and not be 0")
    a.out.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)  # noqa: E731
    device = a.device or warm.pick_device()
    report, model, pool, uses_residual, step = code_audit.load_run(a.run, device)
    if pool is None:
        raise SystemExit(f"{a.run.name}: the code comes from TailCode's head, so the run needs --pool tail or gapnorm")
    root, t1key = report["args"]["bank"], report["args"]["t1key"]
    n_test = int(report["args"].get("test_subjects", 1000))
    scm, scale, nonlinearity = own_variation.scm_settings()
    if not np.array_equal(np.load(f"{root}/causal_adj.npy"), scm["adj"]):
        raise SystemExit(f"{root}'s causal_adj.npy is not bank.causal_scm's graph")
    keys = (t1key, "flair")
    views = {"t1": 0, "flair": 1}

    # Latents and every dimension's own noise, per split.
    zc = {s: np.load(f"{root}/{s}_zc.npy").astype(np.float64) for s in ("train", "val", "test")}
    zc["train"], zc["test"] = zc["train"][: a.train_subjects], zc["test"][:n_test]
    noise = {s: exogenous(z, scm, scale, nonlinearity) for s, z in zc.items()}
    if not np.allclose(structural(noise["test"], scm, scale, nonlinearity), zc["test"], atol=1e-4):
        raise SystemExit("the SCM does not run forward to the stored latents")
    factors = {s: z[:, COLUMNS] for s, z in zc.items()}
    sd = factors["train"].std(0)

    # Each view's code on every split.
    def encode(bank):
        return [c[:, : pool.main].astype(np.float32) for c in warm.features(model, bank, device, pool=pool)["code"]]

    sizes = {"train": a.train_subjects, "val": None, "test": n_test}
    codes = {s: encode(warm.Bank(root, s, t1key, n, uses_residual)) for s, n in sizes.items()}
    log(f"{a.run.name} (step {step}): codes of {', '.join(f'{len(c[0])} {s}' for s, c in codes.items())} subjects")

    # Factual re-renders of the chosen test subjects, checked against the bank.
    gen = Generator(root, t1key, bank_setting(root))
    chosen = np.arange(min(a.subjects, n_test))
    factual = [gen.render(int(i)) for i in chosen]
    stored = {key: R.view_images(root, "test", key, len(chosen)) for key in keys}
    worst = max(
        float(np.abs(np.stack([f[key] for f in factual]).astype(np.float32) - stored[key].astype(np.float32)).max())
        for key in keys
    )
    log(f"factual re-render of {len(chosen)} test subjects: largest difference from the bank {worst:.4f}")
    if worst > 0.05:
        raise SystemExit("the re-rendered factual images do not reproduce the bank's: the generator settings differ")
    factual = {key: np.stack([f[key] for f in factual]) for key in keys}
    factual_codes = encode(coded_bank(factual, keys, uses_residual))

    # Decoders, one per view and input, shared by every target.
    decoders = {}
    for view in a.views:
        key = keys[views[view]]
        images = {s: R.view_images(root, s, key, a.train_subjects if s == "train" else None) for s in ("train", "val")}
        mask = (images["train"][:, 0] != 0).any(0)
        stats = (*R.image_stats(images["train"], mask), mask)
        inputs = {
            "code": {s: codes[s][views[view]] for s in ("train", "val")},
            "truth": {s: (factors[s] / sd).astype(np.float32) for s in ("train", "val")},
        }
        decoders[view] = {
            label: decoder_for(a.out, view, label, source, images, stats, device, a.epochs, log)
            for label, source in inputs.items()
        }
        decoders[view]["mask"] = mask
        del images

    results = {"run": str(a.run), "step": step, "delta": a.delta, "rho": a.rho, "targets": {}}
    rng = np.random.default_rng(a.seed)
    half = len(zc["train"]) // 2
    for target in a.targets:
        d = COLUMNS[NAMES.index(target)]
        ref = noise["train"][:, d].mean(), noise["train"][:, d].std()
        cause = {
            s: a.rho * (noise[s][:, d] - ref[0]) / ref[1] + np.sqrt(1 - a.rho**2) * rng.standard_normal(len(noise[s]))
            for s in ("train", "test")
        }
        moved = {s: noise[s].copy() for s in ("train", "test")}
        for s in moved:
            moved[s][:, d] += a.rho * a.delta * ref[1]
        shift = structural(moved["test"], scm, scale, nonlinearity) - zc["test"]

        # Counterfactual renders; a subject whose shifted anatomy cannot hold the lesion is left out.
        counter = [gen.render(int(i), shift[i]) for i in chosen]
        ok = np.array([c is not None for c in counter])
        counter = {key: np.stack([c[key] for c in counter if c is not None]) for key in keys}
        kept = chosen[ok]
        true_effect = shift[kept][:, COLUMNS].mean(0) / sd
        entry = {
            "true_change_sd": dict(zip(NAMES, true_effect.tolist())),
            "subjects": int(ok.sum()),
            "lesion_misfit": int((~ok).sum()),
        }
        true_codes = encode(coded_bank(counter, keys, uses_residual))
        log(
            f"\n{target}: tabular cause raised by {a.delta:g} SD; {ok.sum()} counterfactual subjects ({(~ok).sum()} left out)"
        )

        for view in a.views:
            v = views[view]
            readout = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(codes["train"][v][:half], factors["train"][:half] / sd)
            decoded = readout.predict(codes["train"][v][half:])
            graph = discover(np.column_stack([decoded, cause["train"][half:]]), (0.05,))[0.05]
            direct = [NAMES[j] for j in np.flatnonzero(graph[-1, :-1])]
            slope = LinearRegression().fit(cause["train"][:, None], codes["train"][v]).coef_[:, 0]
            estimated = codes["test"][v][kept] + slope * a.delta
            code_change = change_scores(
                np.repeat((slope * a.delta)[None], len(kept), 0),
                true_codes[v] - factual_codes[v][ok],
                np.ones(slope.shape, dtype=bool),
            )
            read_estimated = (readout.predict(estimated) - readout.predict(codes["test"][v][kept])).mean(0)
            read_true = (readout.predict(true_codes[v]) - readout.predict(factual_codes[v][ok])).mean(0)
            factor_change = {"counterfactual code": read_estimated, "true counterfactual code": read_true}
            dec, mask = decoders[view], decoders[view]["mask"]
            key = keys[v]
            true_change = counter[key].astype(np.float32) - factual[key][ok].astype(np.float32)
            truth_inputs = [(factors["test"][kept] / sd), ((zc["test"] + shift)[kept][:, COLUMNS] / sd)]
            changes = {
                ARMS[0]: dec["code"](estimated) - dec["code"](codes["test"][v][kept]),
                ARMS[1]: dec["code"](true_codes[v]) - dec["code"](factual_codes[v][ok]),
                ARMS[2]: dec["truth"](truth_inputs[1]) - dec["truth"](truth_inputs[0]),
            }
            scores = {
                arm: {ch: change_scores(c[:, k], true_change[:, k], mask) for k, ch in enumerate(R.CHANNELS)}
                for arm, c in changes.items()
            }
            entry[view] = dict(
                direct_effects_found=direct,
                code_change=code_change,
                factor_change={k: dict(zip(NAMES, v_.tolist())) for k, v_ in factor_change.items()},
                images=scores,
            )
            log(f"  {view}: PC finds the cause's direct effects on {direct or 'nothing'} (true: {target})")
            log(
                f"    the code's change, estimated against the true counterfactual's code: share {code_change['share']:.2f}, "
                f"cosine {code_change['cosine']:.2f}, size {code_change['size']:.2f}"
            )
            log(
                "    mean change of each factor, in its SDs: true / read from the counterfactual code / from the true "
                "counterfactual's code"
            )
            for j, name in enumerate(NAMES):
                if abs(true_effect[j]) > 0.01 or abs(read_estimated[j]) > 0.05 or abs(read_true[j]) > 0.05:
                    log(f"      {name:10s} {true_effect[j]:+.2f} / {read_estimated[j]:+.2f} / {read_true[j]:+.2f}")
            for arm in ARMS:
                cells = "  ".join(
                    f"{ch} {scores[arm][ch]['share']:+.2f} (cos {scores[arm][ch]['cosine']:.2f}, size "
                    f"{scores[arm][ch]['size']:.2f})"
                    for ch in R.CHANNELS
                )
                log(f"    {arm:24s} {cells}")
            order = np.argsort(-np.abs(true_change[:, 0][:, mask]).sum(1))[:4]
            figure(
                a.out / f"counterfactual_{target}_{view}.png",
                f"{view}: tabular cause of {target} raised by {a.delta:g} SD (rows: the four largest true changes)",
                factual[key][ok],
                counter[key],
                changes,
                order,
                kept[order],
            )
        results["targets"][target] = entry
    (a.out / "counterfactual.json").write_text(json.dumps(results, indent=1))
    log(f"\nSaved {a.out / 'counterfactual.json'} and the counterfactual_*.png figures")


if __name__ == "__main__":
    main()
