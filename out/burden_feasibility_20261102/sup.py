"""Supervised controls for one-lesion burden on the bank, through the pilot recipe's own encoder.

Arms (labels are the rendered lesion volume, never used by any self-supervised objective):
  vector    the exact exported path: backbone -> GAP -> MLP head -> 14 content units -> one linear scalar per view.
            This is training.encoder_burden_control.BurdenControl, on bank images and uniform batches.
  cellsum   same backbone, a per-cell MLP head, then the MEAN over cells (no GAP before the head) -> linear scalar.
            Tests whether the head's place relative to the pool matters.
  dense     same backbone -> 1x1x1 conv -> per-cell lesion load, weighted MSE against the true 16^3 load, burden = sum
            of cells. Dense labels: the most this backbone can tell about the lesion. Not a vector.

    python sup.py BANK OUT ARM T1KEY STEPS EVAL_EVERY [--lr 3e-4 --batch 16 --train-subjects 2000 --seed 42 --device mps]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F

from eval.protocol.score_checkpoint import build_model, load_settings

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference"
BINS = ((0, 10), (10, 30), (30, 60), (60, 100), (100, 1e9))


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class BurdenControl(nn.Module):
    """The exported path, as in training.encoder_burden_control: GAP -> head -> content units -> per-view scalar."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        with torch.random.fork_rng(devices=[]):
            self.heads = nn.ModuleList(nn.Linear(encoder.content_channels, 1) for _ in range(2))

    def forward(self, images):
        content = self.encoder(images, pool_only=True, n_views=2)[2][0][:, : self.encoder.content_channels]
        return torch.cat([head(rows) for head, rows in zip(self.heads, content.chunk(2))], dim=0)[:, 0]


class CellSum(nn.Module):
    """Backbone map -> per-cell MLP (the recipe's head, applied before the pool) -> mean -> per-view linear scalar."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList(nn.Linear(encoder.content_channels, 1) for _ in range(2))

    def forward(self, images):
        h = self.encoder._encode(images, 2, None)
        cells = self.encoder.to_encoding(h.flatten(2).transpose(1, 2))[:, :, : self.encoder.content_channels]
        rows = cells.mean(dim=1)
        return torch.cat([head(r) for head, r in zip(self.heads, rows.chunk(2))], dim=0)[:, 0]


class TailPool(nn.Module):
    """Backbone map -> per-channel GAP plus tail means -> the recipe's head -> 14 units -> per-view linear scalar.

    Tails are the mean over all cells of ReLU(|z| - t) on each side, where z standardizes each channel within the
    subject, at t = 2, 3, 4. Nothing here knows about lesions: it reports how much of each channel lies far from
    its own bulk, which a plain mean cannot see. Each view gets a fixed-gauge BatchNorm before the shared head.
    """

    THRESHOLDS = (2.0, 3.0, 4.0)

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        width = 64 * (1 + 2 * len(self.THRESHOLDS))
        self.gauges = nn.ModuleList(nn.BatchNorm1d(width, affine=False) for _ in range(2))
        self.first = nn.Linear(width, 100)
        self.rest = nn.Sequential(nn.LeakyReLU(), nn.Linear(100, 14))
        self.heads = nn.ModuleList(nn.Linear(14, 1) for _ in range(2))

    def forward(self, images):
        h = self.encoder._encode(images, 2, None)
        mean = h.mean((2, 3, 4), keepdim=True)
        z = (h - mean) / (h.std((2, 3, 4), keepdim=True) + 1e-6)
        parts = [mean.flatten(1)]
        for t in self.THRESHOLDS:
            parts += [F.relu(z - t).mean((2, 3, 4)), F.relu(-z - t).mean((2, 3, 4))]
        rows = torch.cat(parts, 1)
        out = []
        for gauge, head, part in zip(self.gauges, self.heads, rows.chunk(2)):
            out.append(head(self.rest(self.first(gauge(part)))))
        return torch.cat(out, 0)[:, 0]


class NormTail(nn.Module):
    """warm.TailCode's pool with labels: normative tails over interior cells -> 14 units -> per-view linear scalar.

    With ``tails=False`` it keeps only TailCode's GAP, gauge and head (the gapnorm arm)."""

    def __init__(self, encoder, keep, tails=True):
        super().__init__()
        from warm import TailCode

        self.encoder = encoder
        self.pool = TailCode(keep, latent=14, thresholds=TailCode.THRESHOLDS if tails else ())
        self.heads = nn.ModuleList(nn.Linear(14, 1) for _ in range(2))

    def forward(self, images):
        rows = self.pool(self.encoder._encode(images, 2, None))
        return torch.cat([head(part) for head, part in zip(self.heads, rows.chunk(2))], 0)[:, 0]


class Dense(nn.Module):
    """Backbone map -> per-view 1x1x1 conv -> per-cell load."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList(nn.Conv3d(64, 1, 1) for _ in range(2))

    def forward(self, images):
        h = self.encoder._encode(images, 2, None)
        return torch.cat([head(m) for head, m in zip(self.heads, h.chunk(2))], dim=0)[:, 0]


class Bank:
    def __init__(self, root, split, t1key, n=None, target="volume", residual=False):
        self.t1 = np.load(f"{root}/{split}_{t1key}.npy", mmap_mode="r")
        self.flair = np.load(f"{root}/{split}_flair.npy", mmap_mode="r")
        self.residuals = (None, None)
        if residual:
            self.residuals = tuple(np.load(f"{root}/{split}_{k}_res.npy", mmap_mode="r") for k in (t1key, "flair"))
        self.cells = np.load(f"{root}/{split}_cells.npy", mmap_mode="r")
        self.vox = np.array(np.load(f"{root}/{split}_vox.npy"))
        self.latent = np.array(np.load(f"{root}/{split}_zc.npy")[:, 2])
        if target == "temporal":
            # The positive control's target: the temporal atrophy latent, in the same slot as volume.
            self.vox = np.array(np.load(f"{root}/{split}_zc.npy")[:, 6])
            self.latent = self.vox
        self.n = len(self.vox) if n is None else n

    def images(self, indices, device):
        from warm import stacked_views

        order = np.argsort(indices)
        return stacked_views(self.t1, self.flair, self.residuals, np.asarray(indices)[order], np.argsort(order)).to(
            device
        )

    def cell_targets(self, indices, device):
        order = np.argsort(indices)
        back = np.argsort(order)
        rows = np.asarray(self.cells[np.asarray(indices)[order]], dtype=np.float32)[back]
        return torch.from_numpy(np.concatenate((rows, rows))).to(device)


def score(prediction, bank, n):
    truth = bank.vox[:n]
    # Calibrated on the same rows only for the dense arm's sum, by the caller; here the prediction is in voxels.
    out = {
        "r2": float(1 - np.square(prediction - truth).sum() / np.square(truth - truth.mean()).sum()),
        "spearman": float(spearmanr(prediction, truth).statistic) if np.ptp(prediction) > 0 else 0.0,
        "latent_spearman": float(spearmanr(prediction, bank.latent[:n]).statistic) if np.ptp(prediction) > 0 else 0.0,
        "pearson2": float(np.corrcoef(prediction, truth)[0, 1] ** 2) if np.ptp(prediction) > 0 else 0.0,
    }
    for low, high in BINS:
        keep = (truth >= low) & (truth < high)
        if keep.sum() > 10 and np.ptp(prediction[keep]) > 0:
            out[f"spearman_{int(low)}"] = float(spearmanr(prediction[keep], truth[keep]).statistic)
            out[f"n_{int(low)}"] = int(keep.sum())
    return out


@torch.no_grad()
def predict(model, arm, bank, n, mean, sd, device, batch=16):
    rows = [[], []]
    for start in range(0, n, batch):
        images = bank.images(np.arange(start, min(start + batch, n)), device)
        output = model(images)
        if arm == "dense":
            output = output.flatten(1).sum(1) * 64  # cell means -> voxels
        else:
            output = output * sd + mean
        for view, part in enumerate(output.cpu().numpy().reshape(2, -1)):
            rows[view].append(part)
    return [np.concatenate(r) for r in rows]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bank")
    p.add_argument("out", type=Path)
    p.add_argument("arm", choices=("vector", "cellsum", "dense", "tail", "normtail", "gapnorm"))
    p.add_argument("t1key", choices=("t1_00", "t1_04"))
    p.add_argument("steps", type=int)
    p.add_argument("every", type=int)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--train-subjects", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=pick_device())
    p.add_argument("--lesion-weight", type=float, default=100.0)
    p.add_argument("--init", default=None, help="Start both backbones from this sup.py checkpoint")
    p.add_argument("--target", choices=("volume", "temporal"), default="volume")
    p.add_argument("--residual", action="store_true", help="Add each view's normative residual as a second channel")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    device = a.device
    cfg = load_settings(REFERENCE)
    cfg.update(model_seed=a.seed, deterministic=False, deterministic_warn_only=False)
    torch.manual_seed(a.seed)
    encoder = build_model(cfg, device)
    if a.residual:
        from warm import widen_input

        widen_input(encoder)
        encoder = encoder.to(device)
    if a.arm in ("normtail", "gapnorm"):
        from warm import interior_cells

        model = NormTail(encoder, interior_cells(a.bank), tails=a.arm == "normtail").to(device)
    else:
        model = {"vector": BurdenControl, "cellsum": CellSum, "dense": Dense, "tail": TailPool}[a.arm](encoder).to(
            device
        )
    if a.init is not None:
        state = torch.load(a.init, map_location="cpu")["model"]
        keep = {k: v for k, v in state.items() if k.startswith(("encoder.encoder.", "encoder.encoder_v1."))}
        print("backbone keys loaded", len(keep), model.load_state_dict(keep, strict=False).unexpected_keys, flush=True)
    train = Bank(a.bank, "train", a.t1key, a.train_subjects, a.target, a.residual)
    val = Bank(a.bank, "val", a.t1key, target=a.target, residual=a.residual)
    test = Bank(a.bank, "test", a.t1key, target=a.target, residual=a.residual)
    mean, sd = float(train.vox[: train.n].mean()), float(train.vox[: train.n].std())
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(a.seed)
    report = {"args": {k: str(v) for k, v in vars(a).items()}, "volume_mean_sd": [mean, sd], "history": []}

    def evaluate(step):
        model.eval()
        rows = {"step": step}
        for name, bank, n in (("val", val, val.n), ("train400", train, min(400, train.n))):
            pred = predict(model, a.arm, bank, n, mean, sd, device)
            rows[name] = {view: score(pred[i], bank, n) for i, view in enumerate(("t1", "flair"))}
        model.train()
        report["history"].append(rows)
        (a.out / "report.json").write_text(json.dumps(report, indent=1))
        v = rows["val"]
        t = rows["train400"]
        print(
            f"  step {step:5d} val R2 T1 {v['t1']['r2']:+.3f} FLAIR {v['flair']['r2']:+.3f} | rho {v['t1']['spearman']:+.3f} "
            f"{v['flair']['spearman']:+.3f} | train R2 {t['t1']['r2']:+.3f} {t['flair']['r2']:+.3f}",
            flush=True,
        )

    evaluate(0)
    start = time.time()
    running = 0.0
    for step in range(1, a.steps + 1):
        indices = rng.integers(0, train.n, a.batch)
        images = train.images(indices, device)
        if a.arm == "dense":
            target = train.cell_targets(indices, device)
            weight = 1 + a.lesion_weight * (target > 0).float()
            loss = (weight * (model(images) - target).square()).sum() / weight.sum()
        else:
            target = torch.as_tensor((train.vox[indices] - mean) / sd, device=device, dtype=torch.float32).repeat(2)
            loss = F.mse_loss(model(images), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        running += loss.item()
        if step % 50 == 0:
            print(f"step {step:5d} loss {running / 50:.4f} {(time.time() - start) / step:.2f}s/step", flush=True)
            running = 0.0
        if step % a.every == 0 or step == a.steps:
            evaluate(step)
            torch.save({"model": model.state_dict(), "step": step}, a.out / "model.pt")
    model.eval()
    pred = predict(model, a.arm, test, test.n, mean, sd, device)
    report["test"] = {view: score(pred[i], test, test.n) for i, view in enumerate(("t1", "flair"))}
    np.savez(a.out / "test_predictions.npz", t1=pred[0], flair=pred[1], vox=test.vox, latent=test.latent)
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    print("TEST", json.dumps(report["test"]), flush=True)


if __name__ == "__main__":
    sys.exit(main())
