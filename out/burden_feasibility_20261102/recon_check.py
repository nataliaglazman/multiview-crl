"""How much of each view's image a frozen code holds: a decoder learned from the code to the image. No network training.

For each finished warm.py run, per view, the run's exported code (frozen, from the unjittered scans) is the only input
to a fresh 3D decoder (code -> 128 x 4^3 -> four upsampling conv blocks -> 64^3, plus a linear path from the code to
every voxel) trained on the bank's training subjects to rebuild that view's input, both channels: the scan and its
normative residual. Each channel is predicted as its deviation from the training mean image, scaled by its spread over the brain; MSE over the brain; early stopping on
the validation subjects. The same decoder, seed and schedule for every row, so rows differ only in their input:

  code      the run's code
  twin      the run's untrained twin's code (its architecture at its own init, TailCode's statistics filled from
            training subjects as causal_bundle --untrained does): the floor
  truth     the bank's rendered shared latents (the 7 SCM factors and the lesion centre, 10 numbers): what a decoder of
            this size rebuilds from the shared content alone, from 2,000 subjects
  truth + style  the same plus the view's own style (gain, bias, noise level; redrawn from the generator's seeds as
            style_audit.py does, 13 numbers): the ceiling. Trained codes carry no style (stage 34) but untrained twins
            carry gain, which scales the whole scan, so a twin can beat the truth row on the scan, not this one.
            Per-voxel noise stays unpredictable, so the ceiling is below 1.

Scores on the test subjects, per channel:

  R2        1 - SSE / SST over brain voxels, SST around the training mean image: the share of the between-subject
            image variation the code rebuilds
  lesion    the same over each subject's lesion cells (16^3 cells with any lesion load, at full resolution)
  burden    detector.py's fixed blob filter read on the rebuilt residual, then precision_audit.py's readout with the
            anatomy part handed in: e, the lesion-volume reading error in voxels, and own R2. The "image" row reads
            the true residual the same way (precision_audit's image stage).

Figures beside OUT_JSON: per bank and view, test subjects at four lesion sizes (axial slice through the lesion), true
image and each row's rebuild, scan and residual; and each row's per-voxel R2 map of the residual.

    python recon_check.py OUT_JSON [LABEL=]RUN [[LABEL=]RUN ...] [--no-twin] [--train-subjects N] [--epochs E]

Rows for other banks already in OUT_JSON are kept; rows for the given runs are recomputed. --train-subjects caps the
training subjects and --epochs the decoder's epochs (local smoke tests).
"""

import argparse
import json
from pathlib import Path

import batch_price
import causal_bundle
import code_audit
import detector
import numpy as np
import precision_audit as PA
import style_audit
import torch
import torch.nn.functional as F
import warm
from scipy.ndimage import binary_erosion
from torch import nn

VIEWS = ("t1", "flair")
CHANNELS = ("scan", "residual")
TRUTH = [0, 1, 2, 5, 6, 7, 8]  # z_content indices of the 7 rendered SCM factors (bank.CAUSAL_FACTORS)
BATCH, LR, DECAY, EPOCHS, PATIENCE, SEED = 16, 1e-3, 1e-4, 150, 15, 0
FIGURE_QUANTILES = (0.2, 0.5, 0.8, 0.95)


class Decoder(nn.Module):
    def __init__(self, units, out=2, width=(128, 96, 64, 32, 16)):
        super().__init__()
        self.inp = nn.Linear(units, width[0] * 4**3)
        self.width = width[0]
        blocks = []
        for a, b in zip(width[:-1], width[1:]):
            blocks += [
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(a, b, 3, padding=1),
                nn.GroupNorm(8, b),
                nn.SiLU(),
            ]
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Conv3d(width[-1], out, 3, padding=1)
        self.linear = nn.Linear(units, out * 64**3)  # a linear path, so no row does worse than a linear map
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, z):
        conv = self.out(self.blocks(F.silu(self.inp(z)).view(len(z), self.width, 4, 4, 4)))
        return conv + self.linear(z).view(conv.shape)


def view_images(root, split, key, n=None):
    """(N, 2, 64, 64, 64) float16: the view's scan and normative residual."""
    scan = np.load(f"{root}/{split}_{key}.npy", mmap_mode="r")[:n]
    res = np.load(f"{root}/{split}_{key}_res.npy", mmap_mode="r")[:n]
    return np.stack([np.asarray(scan, dtype=np.float16), np.asarray(res, dtype=np.float16)], 1)


def lesion_masks(root, split, n):
    """(N, 64, 64, 64) bool: each subject's 16^3 cells with any lesion load, at full resolution."""
    cells = np.asarray(np.load(f"{root}/{split}_cells.npy", mmap_mode="r")[:n]) > 0
    return cells.repeat(4, 1).repeat(4, 2).repeat(4, 3)


def truth_inputs(root, n_train):
    """Per split, the rendered shared latents: the 7 SCM factors and the lesion centre (16^3 cell units)."""
    out = {}
    for split in ("train", "val", "test"):
        n = n_train if split == "train" else None
        zc = np.load(f"{root}/{split}_zc.npy")[:n][:, TRUTH]
        centre = batch_price.lesion_centroid(np.load(f"{root}/{split}_cells.npy", mmap_mode="r")[: len(zc)])
        out[split] = np.column_stack([zc, centre]).astype(np.float32)
    return out


def style_inputs(root, n_train, cache):
    """Per split, each view's style targets (gain, bias, noise level), cached as ``cache``."""
    if cache.exists():
        saved = np.load(cache)
        out = {s: [saved[f"{s}_{v}"] for v in VIEWS] for s in ("train", "val", "test")}
        if len(out["train"][0]) == (n_train or len(np.load(f"{root}/train_zc.npy", mmap_mode="r"))):
            return out
    out = {}
    for split in ("train", "val", "test"):
        n = n_train if split == "train" and n_train else len(np.load(f"{root}/{split}_zc.npy", mmap_mode="r"))
        _, latents = style_audit.redraw(root, n, split=split)
        out[split] = [style_audit.style_targets(z).astype(np.float32) for z in latents]
    np.savez(cache, **{f"{s}_{v}": out[s][i] for s in out for i, v in enumerate(VIEWS)})
    return out


def code_inputs(run, device, trained, n_train):
    """Per split and view, the run's exported code (trained) or its untrained twin's."""
    report, model, pool, residual, step = code_audit.load_run(run, device, trained=trained)
    a = report["args"]
    if pool is None:
        raise SystemExit(f"{run.name}: the code comes from TailCode's head, so the run needs --pool tail or gapnorm")
    if not trained:
        causal_bundle.calibrate(model, pool, warm.Bank(a["bank"], "train", a["t1key"], residual=residual), device)
    out = {}
    for split in ("train", "val", "test"):
        n = {"train": n_train, "val": None, "test": int(a.get("test_subjects", 1000))}[split]
        codes = warm.features(model, warm.Bank(a["bank"], split, a["t1key"], n, residual), device, pool=pool)["code"]
        out[split] = [c[:, : pool.main].astype(np.float32) for c in codes]
    return out, step


def fit_decoder(z, images, mean, scale, mask, device, epochs, log):
    """Train a decoder from z["train"] to the training images, early-stopped on validation; returns it at its best."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    mu, sd = z["train"].mean(0), z["train"].std(0) + 1e-6
    norm = {s: torch.from_numpy((v - mu) / sd).to(device) for s, v in z.items()}
    decoder = Decoder(norm["train"].shape[1]).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    m = torch.from_numpy(mask).to(device)[None, None]
    mean_t, scale_t = (
        torch.from_numpy(mean).to(device)[None],
        torch.from_numpy(scale).to(device)[None, :, None, None, None],
    )

    def target(split, idx):
        x = torch.from_numpy(images[split][np.sort(idx)].astype(np.float32)).to(device)
        return (x - mean_t) / scale_t, np.sort(idx)

    def loss_of(pred, y):
        return (((pred - y) ** 2) * m).sum() / (m.sum() * len(y) * y.shape[1])

    best, best_loss, wait = None, np.inf, 0
    for epoch in range(epochs):
        decoder.train()
        order = rng.permutation(len(norm["train"]))
        for start in range(0, len(order), BATCH):
            y, idx = target("train", order[start : start + BATCH])
            loss = loss_of(decoder(norm["train"][idx]), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        decoder.eval()
        with torch.no_grad():
            val = []
            for start in range(0, len(norm["val"]), BATCH):
                y, idx = target("val", np.arange(start, min(start + BATCH, len(norm["val"]))))
                val.append(float(loss_of(decoder(norm["val"][idx]), y)) * len(idx))
            val = sum(val) / len(norm["val"])
        sched.step(val)
        if val < best_loss - 1e-5:
            best_loss, wait = val, 0
            best = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
        else:
            wait += 1
        if epoch % 10 == 0 or wait == 0:
            log(f"    epoch {epoch:3d} val {val:.4f}{'  best' if wait == 0 else ''}")
        if wait >= PATIENCE:
            break
    decoder.load_state_dict(best)
    decoder.eval()
    return decoder, norm["test"], float(best_loss), epoch + 1


@torch.no_grad()
def evaluate(decoder, z, test, mean, scale, mask, lesion, interior, vox, expected, chosen, device):
    """Rebuild the test images batch by batch and score them; returns the scores, the chosen subjects' rebuilds and the
    residual's per-voxel R2 map."""
    sums = {k: np.zeros(2) for k in ("err", "tot", "lerr", "ltot")}
    verr, vtot = np.zeros(test.shape[2:]), np.zeros(test.shape[2:])
    feats, shown = [], {}
    for start in range(0, len(z), BATCH):
        idx = np.arange(start, min(start + BATCH, len(z)))
        pred = decoder(z[idx]).float().cpu().numpy() * scale[None, :, None, None, None] + mean[None]
        true = test[idx].astype(np.float32)
        err, tot = (pred - true) ** 2, (true - mean[None]) ** 2
        les = lesion[idx][:, None].repeat(2, 1)
        sums["err"] += err[:, :, mask].sum((0, 2))
        sums["tot"] += tot[:, :, mask].sum((0, 2))
        sums["lerr"] += (err * les).sum((0, 2, 3, 4))
        sums["ltot"] += (tot * les).sum((0, 2, 3, 4))
        verr += err[:, 1].sum(0)
        vtot += tot[:, 1].sum(0)
        feats += [detector.features_of(img, interior) for img in pred[:, 1]]
        for i in set(chosen) & set(idx.tolist()):
            shown[i] = pred[i - start]
    out = {
        name: {
            "r2": float(1 - sums["err"][c] / sums["tot"][c]),
            "lesion_r2": float(1 - sums["lerr"][c] / sums["ltot"][c]),
        }
        for c, name in enumerate(CHANNELS)
    }
    burden, _ = PA.assess_stage(np.array(feats, dtype=np.float32), vox, expected, full=False)
    out["burden"] = {"error": burden["error"], "own_r2": burden["own_r2"]}
    return out, np.stack([shown[i] for i in chosen]), 1 - verr / np.maximum(vtot, 1e-12)


def image_stats(images, mask, chunk=100):
    """The training mean image and each channel's spread around it over the brain, in chunks (float16 in memory)."""
    total = np.zeros(images.shape[1:], dtype=np.float64)
    for start in range(0, len(images), chunk):
        total += images[start : start + chunk].astype(np.float64).sum(0)
    mean = (total / len(images)).astype(np.float32)
    sq = np.zeros(2)
    for start in range(0, len(images), chunk):
        dev = images[start : start + chunk].astype(np.float32) - mean[None]
        sq += (dev[:, :, mask] ** 2).sum((0, 2))
    return mean, np.sqrt(sq / (len(images) * mask.sum())).astype(np.float32)


def parse_runs(items):
    runs = []
    for item in items:
        label, _, path = item.rpartition("=")
        path = Path(path)
        runs.append((label or path.name, path))
    return runs


def figure(path, bank_name, view, true, rows, chosen, centres, maps, mask):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["true"] + list(rows)
    fig, axes = plt.subplots(2 * len(chosen), len(labels), figsize=(1.9 * len(labels), 3.8 * len(chosen)))
    axes = np.atleast_2d(axes)
    for r, (i, z) in enumerate(zip(chosen, centres)):
        for c, label in enumerate(labels):
            vol = true[r].astype(np.float32) if label == "true" else rows[label][r]
            for ch in range(2):
                ax = axes[2 * r + ch, c]
                img = (vol[ch] * mask)[:, :, z].T
                lim = (0, float(true[r, 0].max())) if ch == 0 else (-4, 4)
                ax.imshow(img, origin="lower", cmap="gray" if ch == 0 else "RdBu_r", vmin=lim[0], vmax=lim[1])
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0 and ch == 0:
                    ax.set_title(label, fontsize=8)
                if c == 0:
                    ax.set_ylabel(f"#{i} {CHANNELS[ch]}\nz={z}", fontsize=7)
    fig.suptitle(f"{bank_name} {view}: test subjects at lesion-volume quantiles {FIGURE_QUANTILES}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(maps), figsize=(2.2 * len(maps), 4.6), squeeze=False)
    for c, (label, m) in enumerate(maps.items()):
        cuts = ((m[:, :, 32].T, mask[:, :, 32].T, "axial z=32"), (m[:, 32, :].T, mask[:, 32, :].T, "coronal y=32"))
        for r, (cut, mcut, name) in enumerate(cuts):
            ax = axes[r, c]
            im = ax.imshow(np.where(mcut, cut, np.nan), origin="lower", cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(label, fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.6, label="per-voxel R2, residual")
    fig.suptitle(f"{bank_name} {view}: where each input rebuilds the residual", fontsize=9)
    fig.savefig(str(path).replace(".png", "_r2map.png"), dpi=110)
    plt.close(fig)


def show(results):
    print("Rebuilding each view's image from a frozen code. R2: share of between-subject variation rebuilt over the")
    print("brain (scan, residual) and over the lesion cells (residual). burden: lesion-volume error e (voxels) and own")
    print("R2 from the fixed blob filter on the rebuilt residual; 'image' reads the true residual.")
    for bank_name, bank in results.items():
        print(f"\n{bank_name}  (burden own part SD {bank['own_sd']:.1f} voxels)")
        print(
            f"{'row':42s} {'view':5s} {'units':>5s} {'scan R2':>8s} {'res R2':>7s} {'lesion':>7s} {'e vox':>6s} "
            f"{'own':>5s} {'epochs':>6s}"
        )
        for view in VIEWS:
            img = bank["image"][view]
            print(
                f"{'image (true residual)':42s} {view:5s} {'':>5s} {'':>8s} {'':>7s} {'':>7s} "
                f"{PA.voxels(img['error'])} {img['own_r2']:5.2f}"
            )
            for label, row in bank["rows"].items():
                if view not in row:
                    continue
                s = row[view]
                print(
                    f"{label[:42]:42s} {view:5s} {row['units']:5d} {s['scan']['r2']:8.3f} "
                    f"{s['residual']['r2']:7.3f} {s['residual']['lesion_r2']:7.3f} {PA.voxels(s['burden']['error'])} "
                    f"{s['burden']['own_r2']:5.2f} {s['epochs']:6d}"
                )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("runs", nargs="+")
    p.add_argument("--no-twin", action="store_true", help="Skip the untrained twins' rows")
    p.add_argument("--train-subjects", type=int, default=None)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    a = p.parse_args()
    device = warm.pick_device()
    results = json.loads(a.out.read_text()) if a.out.exists() else {}
    log = lambda s: print(s, flush=True)  # noqa: E731

    by_bank = {}
    for label, run in parse_runs(a.runs):
        report = json.loads((run / "report.json").read_text())
        by_bank.setdefault(report["args"]["bank"], []).append((label, run, report))

    for root, runs in by_bank.items():
        bank_name = Path(root).name
        key = runs[0][2]["args"]["t1key"]
        n_test = int(runs[0][2]["args"].get("test_subjects", 1000))
        entry = results.setdefault(bank_name, {"rows": {}, "image": {}})
        vox = np.array(np.load(f"{root}/test_vox.npy"), dtype=np.float64)[:n_test]
        expected, _, _ = PA.own_volume(root, vox)
        entry["own_sd"] = float((vox - expected).std())
        sample = np.asarray(np.load(f"{root}/train_flair.npy", mmap_mode="r")[:200], dtype=np.float32)
        interior = binary_erosion((sample != 0).mean(0) >= 0.9, iterations=2)
        lesion = lesion_masks(root, "test", n_test)
        quantiles = np.quantile(vox, FIGURE_QUANTILES)
        chosen = [int(np.argmin(np.abs(vox - q))) for q in quantiles]
        centres = batch_price.lesion_centroid(np.load(f"{root}/test_cells.npy", mmap_mode="r")[:n_test])
        slices = [int(np.clip(round(centres[i, 2] * 4 + 1.5), 0, 63)) for i in chosen]

        truth = truth_inputs(root, a.train_subjects)
        styles = style_inputs(root, a.train_subjects, a.out.parent / f"{a.out.stem}_{bank_name}_styles.npz")
        inputs = {
            "truth (10 latents)": (0, truth),
            "truth + style (13)": (0, {s: [np.column_stack([truth[s], st]) for st in styles[s]] for s in truth}),
        }
        for label, run, report in runs:
            for trained in (True,) if a.no_twin else (True, False):
                codes, step = code_inputs(run, device, trained, a.train_subjects)
                inputs[label if trained else f"{label} twin"] = (step, codes)

        for v, view in enumerate(VIEWS):
            vkey = key if view == "t1" else "flair"
            images = {
                s: view_images(root, s, vkey, a.train_subjects if s == "train" else None) for s in ("train", "val")
            }
            test = view_images(root, "test", vkey, n_test)
            mask = (images["train"][:, 0] != 0).any(0)
            mean, scale = image_stats(images["train"], mask)
            feats = np.array(
                [detector.features_of(img.astype(np.float32), interior) for img in test[:, 1]], dtype=np.float32
            )
            img, _ = PA.assess_stage(feats, vox, expected, full=False)
            entry["image"][view] = {"error": img["error"], "own_r2": img["own_r2"]}
            shown, maps = {}, {}
            for label, (step, source) in inputs.items():
                z = {s: (source[s][v] if isinstance(source[s], list) else source[s]) for s in source}
                z["test"] = z["test"][:n_test]
                row = entry["rows"].setdefault(label, {})
                kind = "truth" if label.startswith("truth") else "twin" if label.endswith(" twin") else "code"
                row.update(kind=kind, units=int(z["train"].shape[1]), step=int(step))
                log(f"{bank_name} {view}: {label} ({row['units']} units)")
                decoder, z_test, val, epochs = fit_decoder(z, images, mean, scale, mask, device, a.epochs, log)
                s, shown[label], maps[label] = evaluate(
                    decoder, z_test, test, mean, scale, mask, lesion, interior, vox, expected, chosen, device
                )
                s.update(val_loss=val, epochs=epochs)
                row[view] = s
                a.out.write_text(json.dumps(results, indent=1))
                log(
                    f"    scan R2 {s['scan']['r2']:.3f}  residual R2 {s['residual']['r2']:.3f}  lesion "
                    f"{s['residual']['lesion_r2']:.3f}  burden e {PA.voxels(s['burden']['error']).strip()}"
                )
            fig = a.out.parent / f"{a.out.stem}_{bank_name}_{view}.png"
            figure(fig, bank_name, view, test[chosen], shown, chosen, slices, maps, mask)
            np.savez_compressed(
                str(fig).replace(".png", ".npz"),
                true=test[chosen],
                chosen=chosen,
                slices=slices,
                labels=np.array(list(shown)),
                rebuilt=np.stack(list(shown.values())),
                r2map=np.stack(list(maps.values())),
            )
    show(results)


if __name__ == "__main__":
    main()
