"""The real self-supervised objective, end to end, with no labels in training.

The pilot control's recipe and objective (global + patch InfoNCE, neighbourhood batches, recipe lr and clip), with:

  --pool gap    the recipe's global code: GAP -> head. Through T.training_objective, unchanged. The matched control.
  --pool tail   the global code reads GAP plus normative tail means (TailCode). The patch term is unchanged.
  INIT          cold, or start both backbones from a sup.py checkpoint.

Each eval scores, on fresh test subjects, 5-fold ridge of volume and the anatomy latents from the 14-unit code, from
the backbone GAP, and for --pool tail from the pooled vector before the head. Labels only ever score.

    python warm.py BANK OUT INIT|cold STEPS EVERY [--pool gap|tail] [--t1key t1_00] [--uniform] [--frozen-backbone]
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

import data.cluster_batches as cb
import training.encoder_args as encoder_args
import training.main_conv_synthetic as T

HERE = Path(__file__).resolve().parent
ANATOMY = [0, 1, 5, 6, 7, 8]
FACTORS = {0: "brain", 1: "ventricle", 5: "thickness", 6: "temporal", 7: "lr", 8: "sulcal"}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def run_args():
    flags = (HERE / "reference" / "flags.txt").read_text().split()
    args = encoder_args.parse_args(flags + ["--vector-prediction-weight", "0"])
    args.cache = False
    return args


def ridge(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        pred = cross_val_predict(pipe, x, y, cv=KFold(5, shuffle=True, random_state=0))
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def interior_cells(root, occupancy=0.9):
    """Cells of the 16^3 map whose mean brain occupancy over training subjects reaches ``occupancy``."""
    sample = np.asarray(np.load(f"{root}/train_flair.npy", mmap_mode="r")[:300]) != 0
    cells = F.avg_pool3d(torch.from_numpy(sample.astype(np.float32))[:, None], 4)[:, 0].mean(0)
    return (cells >= occupancy).flatten().numpy()


class TailCode(nn.Module):
    """The global code from GAP plus normative tail means, through the recipe's head shape.

    Each channel at each interior cell is z-scored by running statistics of that position and view, which need no
    labels; the spread is about the running mean, because a neighbourhood batch shares its anatomy. Per channel the
    pool adds the mean over interior cells of ReLU(z - t) and ReLU(-z - t) at t = 2, 3, 4, log-compressed, so a cell
    in its normal range contributes exactly nothing and the loss's gradient reaches rare cells directly. Nothing here
    knows about lesions, and every factor goes through this one route.
    """

    THRESHOLDS = (2.0, 3.0, 4.0)

    def __init__(self, keep, channels=64, latent=18, momentum=0.01):
        super().__init__()
        self.register_buffer("keep", torch.from_numpy(keep))
        cells = int(keep.sum())
        self.register_buffer("mean", torch.zeros(2, channels, cells))
        self.register_buffer("var", torch.ones(2, channels, cells))
        self.register_buffer("started", torch.zeros(2, dtype=torch.bool))
        self.momentum = momentum
        width = channels * (1 + 2 * len(self.THRESHOLDS))
        self.gauges = nn.ModuleList(nn.BatchNorm1d(width, affine=False) for _ in range(2))
        self.head = nn.Sequential(nn.Linear(width, 100), nn.LeakyReLU(), nn.Linear(100, latent))

    def pooled(self, h):
        """Per view, the gauged pooled vector of (2B, C, D, H, W) view-major maps."""
        out = []
        for view, part in enumerate(h.chunk(2)):
            rows = part.flatten(2)[:, :, self.keep]
            if self.training:
                with torch.no_grad():
                    if not self.started[view]:
                        self.mean[view] = rows.mean(0)
                        self.var[view] = rows.var(0, unbiased=False)
                        self.started[view] = True
                    else:
                        self.mean[view] += self.momentum * (rows.mean(0) - self.mean[view])
                        self.var[view] += self.momentum * ((rows - self.mean[view]).square().mean(0) - self.var[view])
            z = (rows - self.mean[view]) / (self.var[view].sqrt() + 1e-6)
            parts = [part.mean((2, 3, 4))]
            for t in self.THRESHOLDS:
                parts += [torch.log1p(100 * F.relu(z - t).mean(2)), torch.log1p(100 * F.relu(-z - t).mean(2))]
            out.append(self.gauges[view](torch.cat(parts, 1)))
        return out

    def forward(self, h):
        return torch.cat([self.head(part) for part in self.pooled(h)], 0)


class Bank:
    def __init__(self, root, split, t1key, n=None):
        self.t1 = np.load(f"{root}/{split}_{t1key}.npy", mmap_mode="r")
        self.flair = np.load(f"{root}/{split}_flair.npy", mmap_mode="r")
        self.vox = np.array(np.load(f"{root}/{split}_vox.npy"), dtype=np.float64)[:n]
        self.zc = np.array(np.load(f"{root}/{split}_zc.npy"), dtype=np.float64)[:n]

    def images(self, indices, device):
        order = np.argsort(indices)
        back = np.argsort(order)
        chosen = np.asarray(indices)[order]
        t1 = np.asarray(self.t1[chosen], dtype=np.float32)[back]
        flair = np.asarray(self.flair[chosen], dtype=np.float32)[back]
        return torch.from_numpy(np.concatenate((t1, flair)))[:, None].to(device)


@torch.no_grad()
def features(model, bank, device, batch=32, pool=None):
    states = [(m, m.training) for m in (model, pool) if m is not None]
    for m, _ in states:
        m.eval()
    rows = {"code": [[], []], "gap": [[], []], "pooled": [[], []]}
    for start in range(0, len(bank.vox), batch):
        x = bank.images(np.arange(start, min(start + batch, len(bank.vox))), device)
        h = model._encode(x, 2, None)
        gap = h.mean((2, 3, 4)).cpu().numpy()
        if pool is None:
            code = model._global_code(h, x, 2)[:, :14].cpu().numpy()
        else:
            parts = pool.pooled(h)
            code = torch.cat([pool.head(p) for p in parts], 0)[:, :14].cpu().numpy()
            for view in range(2):
                rows["pooled"][view].append(parts[view].cpu().numpy())
        for view in range(2):
            rows["code"][view].append(np.split(code, 2)[view])
            rows["gap"][view].append(np.split(gap, 2)[view])
    for m, flag in states:
        m.train(flag)
    return {k: [np.concatenate(v[i]) for i in range(2)] for k, v in rows.items() if v[0]}


def evaluate(model, bank, device, pool=None):
    feats = features(model, bank, device, pool=pool)
    row = {}
    for view, name in enumerate(("t1", "flair")):
        cell = {
            "code_volume_r2": ridge(feats["code"][view], bank.vox),
            "gap_volume_r2": ridge(feats["gap"][view], bank.vox),
            "code_latent_r2": ridge(feats["code"][view], bank.zc[:, 2]),
        }
        if "pooled" in feats:
            cell["pooled_volume_r2"] = ridge(feats["pooled"][view], bank.vox)
        for column, factor in FACTORS.items():
            cell[f"code_{factor}"] = ridge(feats["code"][view], bank.zc[:, column])
        row[name] = cell
    if "pooled" in feats:
        row["cross_view_code_r"] = float(
            np.mean(np.abs(np.diag(np.corrcoef(feats["code"][0].T, feats["code"][1].T)[:14, 14:])))
        )
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bank")
    p.add_argument("out", type=Path)
    p.add_argument("init")
    p.add_argument("steps", type=int)
    p.add_argument("every", type=int)
    p.add_argument("--pool", choices=("gap", "tail"), default="gap")
    p.add_argument("--t1key", default="t1_00")
    p.add_argument("--uniform", action="store_true")
    p.add_argument("--frozen-backbone", action="store_true")
    p.add_argument("--device", default=pick_device())
    p.add_argument("--test-subjects", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    args = run_args()
    if a.t1key == "t1_04":
        args.synthetic_lesion_t1_value = 0.4
    seed = args.model_seed if a.seed is None else a.seed
    torch.manual_seed(seed)
    model = T.build_encoder(args)
    if a.init != "cold":
        state = torch.load(a.init, map_location="cpu")["model"]
        keep = {
            k[len("encoder.") :]: v
            for k, v in state.items()
            if k.startswith("encoder.encoder.") or k.startswith("encoder.encoder_v1.")
        }
        result = model.load_state_dict(keep, strict=False)
        print("loaded", len(keep), "unexpected", result.unexpected_keys, flush=True)
    model = model.to(a.device)
    pool = None
    if a.pool == "tail":
        pool = TailCode(interior_cells(a.bank)).to(a.device)
        print("interior cells", int(pool.keep.sum()), flush=True)
    if a.frozen_backbone:
        for name, parameter in model.named_parameters():
            if name.startswith("encoder."):
                parameter.requires_grad_(False)
    train = Bank(a.bank, "train", a.t1key)
    test = Bank(a.bank, "test", a.t1key, a.test_subjects)
    neighbours = cb.nearest_neighbours(train.zc[:, ANATOMY], args.stratify_neighbours)
    sampler = cb.NeighbourhoodBatchSampler(neighbours, args.batch_size, args.loader_seed, 0.0)
    rng = np.random.default_rng(args.seed)

    def batches():
        while True:
            if a.uniform:
                yield rng.integers(0, len(train.vox), args.batch_size)
            else:
                yield from sampler

    source = batches()
    parameters = [q for q in model.parameters() if q.requires_grad]
    if pool is not None:
        parameters += list(pool.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
    report = {"args": {k: str(v) for k, v in vars(a).items()}, "history": []}

    def record(step, terms):
        row = {"step": step, "loss": terms, "test": evaluate(model, test, a.device, pool)}
        report["history"].append(row)
        (a.out / "report.json").write_text(json.dumps(report, indent=1))
        t = row["test"]
        print(
            f"step {step:5d} code volume R2 T1 {t['t1']['code_volume_r2']:+.3f} FLAIR {t['flair']['code_volume_r2']:+.3f} | "
            f"GAP {t['t1']['gap_volume_r2']:+.3f} {t['flair']['gap_volume_r2']:+.3f} | "
            f"T1 brain {t['t1']['code_brain']:.2f} vent {t['t1']['code_ventricle']:.2f} thick {t['t1']['code_thickness']:.2f} "
            f"temp {t['t1']['code_temporal']:.2f} | {terms}",
            flush=True,
        )

    record(0, {})
    model.train()
    if pool is not None:
        pool.train()
    start = time.time()
    keep = {"global": 0.0, "patch": 0.0}
    last = {}
    for step in range(1, a.steps + 1):
        indices = np.asarray(next(source))
        x = train.images(indices, a.device)
        if pool is None:
            _, loss, terms = T.training_objective(model, x, args, sim, crit, None)
        else:
            h = model._encode(x, 2, None)
            patches = model.patch_readout(model._patch_pool(h, args.train_patch_grid).transpose(1, 2)).transpose(1, 2)
            terms = {
                "global": T.contrastive_loss(pool(h), model, args, sim, crit, None),
                "patch": T.patch_contrastive_loss(patches, model, args, sim, crit, x != 0, None, None),
            }
            loss = terms["global"] + args.patch_loss_weight * terms["patch"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        keep["global"] += float(terms["global"]) / 50
        keep["patch"] += float(terms["patch"]) / 50
        if step % 50 == 0:
            print(
                f"step {step:5d} global {keep['global']:.3f} patch {keep['patch']:.3f} "
                f"{(time.time() - start) / step:.2f}s/step",
                flush=True,
            )
            last = dict(keep)
            keep = {"global": 0.0, "patch": 0.0}
        if step % a.every == 0 or step == a.steps:
            record(step, last)
            state = {"model": model.state_dict(), "step": step}
            if pool is not None:
                state["pool"] = pool.state_dict()
            torch.save(state, a.out / "model.pt")
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
