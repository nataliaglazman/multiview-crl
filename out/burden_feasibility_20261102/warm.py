"""The real self-supervised objective, end to end, with no labels in training.

The pilot control's recipe and objective (global + patch InfoNCE, neighbourhood batches, recipe lr and clip), with:

  --pool gap    the recipe's global code: GAP -> head. Through T.training_objective, unchanged. The matched control.
  --pool tail   the global code reads GAP plus normative tail means (TailCode). The patch term is unchanged.
  --pool gapnorm  GAP through TailCode's BatchNorm gauge and head, no tails: separates the tails from the gauge.
  --residual    each view's input gains a second channel, its normative residual (residual_bank.py, no labels).
  --queue N     the global InfoNCE also takes N earlier subjects as negatives, from a momentum encoder (MoCo-style),
                so the loss keeps paying after the batch of 32 is told apart. The patch term is unchanged.
  --gauge running  TailCode's gauge standardizes with running statistics in training as at test. The default BatchNorm
                uses each batch's own statistics in training and running ones at test, which differ when batches are
                neighbourhoods; the momentum encoder then also differs from the online one.
  --decorrelate W  adds W times a Barlow Twins term on the 14-unit code: each unit should correlate fully between the
                two views and not with the other units, so the loss keeps paying until every unit carries its own shared
                signal, where InfoNCE stops once the batch is told apart. Needs --pool tail or gapnorm.
  --group-dropout P  in a fraction P of steps the head sees TailCode's GAP features with the tails zeroed, in another P
                the tails with the GAP zeroed (after the gauge, both views alike), so the code has to tell subjects apart
                from either group alone. Needs --pool tail and no queue.
  --jitter V, --rotate D  each training image, every subject and view separately, is shifted up to V voxels and
                rotated up to D degrees about each axis, its residual channel with it, so the views cannot be matched
                voxel for voxel.
  INIT          cold, or start both backbones from a sup.py checkpoint.

Batches: by default an anchor and 31 of its 50 nearest training subjects on the six anatomy factors (the recipe's
oracle neighbourhoods, which use anatomy labels). --uniform draws 32 distinct subjects uniformly. --code-batches K
rebuilds the neighbourhoods every K steps on the current model's own code, both views, with no labels; each rebuild
logs how far its neighbourhoods match anatomy, lesion position and burden (labels there only score).

Each eval scores, on fresh test subjects, 5-fold ridge of volume and the anatomy latents from the 14-unit code, from
the backbone GAP, and for --pool tail from the pooled vector before the head. Labels only ever score.

    python warm.py BANK OUT INIT|cold STEPS EVERY [--pool gap|tail|gapnorm] [--gauge batch|running] [--residual]
        [--t1key t1_00] [--uniform | --code-batches K] [--queue N] [--decorrelate W] [--group-dropout P]
        [--jitter V] [--rotate D]
"""

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path

import batch_price
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
OFF_DIAGONAL = 1 / 13  # 1 / (units - 1): the off-diagonal sum weighs about as much as the diagonal


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


class RunningGauge(nn.Module):
    """Per-feature standardization by a running mean and SD, updated only in training and applied the same way in
    training and at test. The spread is about the running mean, as in TailCode, so it is the population's spread even
    when a batch is a neighbourhood."""

    def __init__(self, width, momentum=0.01):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("var", torch.ones(width))
        self.register_buffer("started", torch.zeros((), dtype=torch.bool))
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                if not self.started:
                    self.mean.copy_(x.mean(0))
                    self.var.copy_(x.var(0, unbiased=False))
                    self.started.fill_(True)
                else:
                    self.mean += self.momentum * (x.mean(0) - self.mean)
                    self.var += self.momentum * ((x - self.mean).square().mean(0) - self.var)
        return (x - self.mean) / (self.var + 1e-5).sqrt()


class TailCode(nn.Module):
    """The global code from GAP plus normative tail means, through the recipe's head shape.

    Each channel at each interior cell is z-scored by running statistics of that position and view, which need no
    labels; the spread is about the running mean, because a neighbourhood batch shares its anatomy. Per channel the
    pool adds the mean over interior cells of ReLU(z - t) and ReLU(-z - t) at t = 2, 3, 4, log-compressed, so a cell
    in its normal range contributes exactly nothing and the loss's gradient reaches rare cells directly. Nothing here
    knows about lesions, and every factor goes through this one route.

    ``thresholds=()`` keeps only the GAP and the BatchNorm gauge: the control that separates the tails from the gauge.
    ``gauge="running"`` replaces the BatchNorm gauge with RunningGauge, so the head sees one normalization in training
    and at test. ``drop`` ("gap" or "tails"), set per step by the training loop, zeroes that group after the gauge, in
    training only.
    """

    THRESHOLDS = (2.0, 3.0, 4.0)

    def __init__(self, keep, channels=64, latent=18, momentum=0.01, thresholds=THRESHOLDS, gauge="batch"):
        super().__init__()
        if gauge not in ("batch", "running"):
            raise ValueError(f"gauge must be batch or running, got {gauge!r}")
        self.register_buffer("keep", torch.from_numpy(keep))
        cells = int(keep.sum())
        self.register_buffer("mean", torch.zeros(2, channels, cells))
        self.register_buffer("var", torch.ones(2, channels, cells))
        self.register_buffer("started", torch.zeros(2, dtype=torch.bool))
        self.momentum = momentum
        self.thresholds = tuple(thresholds)
        self.channels = channels
        self.drop = None
        width = channels * (1 + 2 * len(self.thresholds))
        if gauge == "batch":
            self.gauges = nn.ModuleList(nn.BatchNorm1d(width, affine=False) for _ in range(2))
        else:
            self.gauges = nn.ModuleList(RunningGauge(width, momentum) for _ in range(2))
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
            for t in self.thresholds:
                parts += [torch.log1p(100 * F.relu(z - t).mean(2)), torch.log1p(100 * F.relu(-z - t).mean(2))]
            gauged = self.gauges[view](torch.cat(parts, 1))
            if self.training and self.drop:
                kept = torch.ones(gauged.shape[1], device=gauged.device)
                kept[: self.channels] = float(self.drop != "gap")
                kept[self.channels :] = float(self.drop != "tails")
                gauged = gauged * kept
            out.append(gauged)
        return out

    def forward(self, h):
        return torch.cat([self.head(part) for part in self.pooled(h)], 0)


class CodeGauge:
    """Running per-view mean and SD of the online content code: the queue's stand-in for per-batch standardization,
    which cannot reach codes from earlier batches. Same spread-about-the-running-mean rule as TailCode."""

    def __init__(self, units, device, momentum=0.01):
        self.mean = torch.zeros(2, units, device=device)
        self.var = torch.ones(2, units, device=device)
        self.started = False
        self.momentum = momentum

    @torch.no_grad()
    def update(self, codes):
        if not self.started:
            self.mean.copy_(codes.mean(1))
            self.var.copy_(codes.var(1, unbiased=False))
            self.started = True
            return
        self.mean += self.momentum * (codes.mean(1) - self.mean)
        self.var += self.momentum * ((codes - self.mean[:, None]).square().mean(1) - self.var)

    def __call__(self, codes, view):
        return F.normalize((codes - self.mean[view]) / (self.var[view].sqrt() + 1e-6), dim=-1)


class NegativeQueue:
    """Earlier batches' momentum-encoder codes per view, with their bank indices, so no subject is its own negative."""

    def __init__(self, size, units, device):
        self.codes = torch.zeros(2, size, units, device=device)
        self.ids = torch.full((size,), -1, dtype=torch.long, device=device)
        self.pointer = 0

    @torch.no_grad()
    def push(self, keys, ids):
        rows = (self.pointer + torch.arange(keys.shape[1], device=keys.device)) % self.codes.shape[1]
        self.codes[:, rows] = keys
        self.ids[rows] = ids
        self.pointer = int((self.pointer + keys.shape[1]) % self.codes.shape[1])


def queue_infonce(queries, keys, queue, ids, gauge, tau):
    """Cross-view InfoNCE with many negatives, as MoCo: each online code against the other view's momentum key of the
    same subject; negatives are the batch's other keys and every queued key of a different subject. (2, B, units) in,
    mean of both directions out."""
    filled = queue.ids >= 0
    stored_ids = queue.ids[filled]
    total = 0.0
    for a, b in ((0, 1), (1, 0)):
        q = gauge(queries[a], a)
        batch = q @ gauge(keys[b], b).T / tau
        past = q @ gauge(queue.codes[b][filled], b).T / tau
        past = past.masked_fill(stored_ids[None, :] == ids[:, None], float("-inf"))
        logits = torch.cat([batch, past], 1)
        total = total + F.cross_entropy(logits, torch.arange(len(q), device=q.device))
    return total / 2


@torch.no_grad()
def momentum_update(online, target, momentum):
    for kept, live in zip(target.parameters(), online.parameters()):
        kept.mul_(momentum).add_(live.detach(), alpha=1 - momentum)
    for kept, live in zip(target.buffers(), online.buffers()):
        kept.copy_(live)


def global_code(model, pool, h, x):
    return model._global_code(h, x, 2) if pool is None else pool(h)


def cross_view_decorrelation(z1, z2):
    """Barlow Twins on (B, units) codes: each unit standardized over the batch per view and C their cross-view
    correlation matrix; (1 - C_ii)^2 summed over units, plus OFF_DIAGONAL times C_ij^2 summed over i != j."""
    z1, z2 = ((z - z.mean(0)) / (z.std(0, unbiased=False) + 1e-6) for z in (z1, z2))
    c = z1.T @ z2 / len(z1)
    diagonal = torch.diagonal(c)
    return (1 - diagonal).square().sum() + OFF_DIAGONAL * (c.square().sum() - diagonal.square().sum())


def jitter(x, shift, degrees, rng):
    """Each row of (N, C, D, H, W) cubic volumes moved by its own random rigid transform, every channel alike: shifts
    up to ``shift`` voxels and rotations up to ``degrees`` about each axis, trilinear, zero outside."""
    n, size = len(x), x.shape[-1]
    theta = np.zeros((n, 3, 4))
    for i, (a, b, c) in enumerate(np.deg2rad(rng.uniform(-degrees, degrees, (n, 3)))):
        rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        theta[i, :, :3] = rz @ ry @ rx
    theta[:, :, 3] = rng.uniform(-shift, shift, (n, 3)) * 2 / size
    grid = F.affine_grid(torch.as_tensor(theta, dtype=x.dtype, device=x.device), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def widen_input(model):
    """Two input channels, image then residual. The image weights are kept and both backbones stay identical."""
    first = model.encoder.layers[0][0]
    wide = nn.Conv3d(2, first.out_channels, first.kernel_size, first.stride, first.padding)
    with torch.no_grad():
        wide.weight[:, 0] = first.weight[:, 0]
        wide.bias.copy_(first.bias)
    model.encoder.layers[0][0] = wide
    model.encoder_v1.layers[0][0] = copy.deepcopy(wide)


def stacked_views(t1, flair, residuals, chosen, back):
    """(2B, channels, D, H, W) view-major images, with each view's residual as a second channel when it has one."""
    views = []
    for view, extra in zip((t1, flair), residuals):
        rows = [np.asarray(view[chosen], dtype=np.float32)[back]]
        if extra is not None:
            rows.append(np.asarray(extra[chosen], dtype=np.float32)[back])
        views.append(np.stack(rows, 1))
    return torch.from_numpy(np.concatenate(views))


class Bank:
    def __init__(self, root, split, t1key, n=None, residual=False):
        self.t1 = np.load(f"{root}/{split}_{t1key}.npy", mmap_mode="r")
        self.flair = np.load(f"{root}/{split}_flair.npy", mmap_mode="r")
        self.residuals = (None, None)
        if residual:
            self.residuals = tuple(np.load(f"{root}/{split}_{k}_res.npy", mmap_mode="r") for k in (t1key, "flair"))
        self.vox = np.array(np.load(f"{root}/{split}_vox.npy"), dtype=np.float64)[:n]
        self.zc = np.array(np.load(f"{root}/{split}_zc.npy"), dtype=np.float64)[:n]

    def images(self, indices, device):
        order = np.argsort(indices)
        return stacked_views(self.t1, self.flair, self.residuals, np.asarray(indices)[order], np.argsort(order)).to(
            device
        )


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
    p.add_argument("--pool", choices=("gap", "tail", "gapnorm"), default="gap")
    p.add_argument(
        "--gauge",
        choices=("batch", "running"),
        default="batch",
        help="TailCode's gauge: BatchNorm (batch statistics in training, running at test) or running statistics in both",
    )
    p.add_argument("--residual", action="store_true", help="Add each view's normative residual as a second channel")
    p.add_argument("--decorrelate", type=float, default=0.0, help="Weight of a Barlow Twins term on the 14-unit code")
    p.add_argument(
        "--group-dropout",
        type=float,
        default=0.0,
        help="Fraction of steps in which the head sees only TailCode's GAP, and again only its tails",
    )
    p.add_argument("--jitter", type=float, default=0.0, help="Random shift of each training image, in voxels")
    p.add_argument("--rotate", type=float, default=0.0, help="Random rotation of each training image, degrees per axis")
    p.add_argument("--queue", type=int, default=0, help="Negatives from a momentum-encoder queue of this many subjects")
    p.add_argument("--queue-momentum", type=float, default=0.99)
    p.add_argument("--t1key", default="t1_00")
    p.add_argument("--uniform", action="store_true", help="32 distinct subjects drawn uniformly")
    p.add_argument(
        "--code-batches",
        type=int,
        default=0,
        help="Rebuild the neighbourhood batches every this many steps on the model's own code (no labels)",
    )
    p.add_argument("--frozen-backbone", action="store_true")
    p.add_argument("--device", default=pick_device())
    p.add_argument("--test-subjects", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    if a.uniform and a.code_batches:
        raise ValueError("--uniform and --code-batches are different batch rules")
    if a.gauge != "batch" and a.pool == "gap":
        raise ValueError("--gauge applies to TailCode, so to --pool tail or gapnorm")
    if a.decorrelate and a.pool == "gap":
        raise ValueError("--decorrelate needs --pool tail or gapnorm")
    if a.group_dropout and (a.pool != "tail" or a.queue):
        raise ValueError("--group-dropout needs --pool tail and no --queue")
    if not 0 <= a.group_dropout <= 0.5:
        raise ValueError("--group-dropout is a fraction of steps per group, at most 0.5")
    a.out.mkdir(parents=True, exist_ok=False)
    args = run_args()
    if a.t1key == "t1_04":
        args.synthetic_lesion_t1_value = 0.4
    seed = args.model_seed if a.seed is None else a.seed
    torch.manual_seed(seed)
    model = T.build_encoder(args)
    if a.residual:
        widen_input(model)
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
    if a.pool != "gap":
        thresholds = () if a.pool == "gapnorm" else TailCode.THRESHOLDS
        pool = TailCode(interior_cells(a.bank), thresholds=thresholds, gauge=a.gauge).to(a.device)
        print("interior cells", int(pool.keep.sum()), "tail thresholds", pool.thresholds, "gauge", a.gauge, flush=True)
    if a.frozen_backbone:
        for name, parameter in model.named_parameters():
            if name.startswith("encoder."):
                parameter.requires_grad_(False)
    train = Bank(a.bank, "train", a.t1key, residual=a.residual)
    test = Bank(a.bank, "test", a.t1key, a.test_subjects, a.residual)
    scored = {
        "anatomy": train.zc[:, ANATOMY],
        "lesion_position": batch_price.lesion_centroid(np.load(f"{a.bank}/train_cells.npy", mmap_mode="r")),
        "burden": train.vox[:, None],
    }
    batching = {"spread": None}

    def rebuild(neighbours, seed):
        """Batches from these neighbourhoods; the spread is each block's mean SD among a subject's neighbours over its
        population SD (1 = no matching), for the log only."""
        batching["sampler"] = cb.NeighbourhoodBatchSampler(neighbours, args.batch_size, seed, 0.0)
        batching["spread"] = {k: round(float((v[neighbours].std(1) / v.std(0)).mean()), 3) for k, v in scored.items()}

    if not a.uniform and not a.code_batches:
        rebuild(cb.nearest_neighbours(train.zc[:, ANATOMY], args.stratify_neighbours), args.loader_seed)
    rng = np.random.default_rng(args.seed)
    aug = np.random.default_rng(None if args.seed is None else args.seed + 1)

    def batches():
        while True:
            if a.uniform:
                yield rng.choice(len(train.vox), args.batch_size, replace=False)
                continue
            current = batching["sampler"]
            for batch in current:
                yield batch
                if batching["sampler"] is not current:
                    break

    source = batches()
    parameters = [q for q in model.parameters() if q.requires_grad]
    if pool is not None:
        parameters += list(pool.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
    queue = None
    if a.queue > 0:
        queue = NegativeQueue(a.queue, 14, a.device)
        gauge = CodeGauge(14, a.device)
        momentum_model = copy.deepcopy(model).eval()
        momentum_pool = None if pool is None else copy.deepcopy(pool).eval()
        for frozen in (momentum_model, momentum_pool):
            if frozen is not None:
                frozen.requires_grad_(False)
    report = {"args": {k: str(v) for k, v in vars(a).items()}, "history": []}

    def record(step, terms):
        row = {"step": step, "loss": terms, "test": evaluate(model, test, a.device, pool), "spread": batching["spread"]}
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
    keep = {}
    last = {}
    for step in range(1, a.steps + 1):
        if a.code_batches and (step - 1) % a.code_batches == 0:
            codes = features(model, train, a.device, pool=pool)["code"]
            rebuild(cb.nearest_neighbours(np.concatenate(codes, 1), args.stratify_neighbours), args.loader_seed + step)
            print(f"step {step:5d} code neighbourhoods rebuilt, spread {batching['spread']}", flush=True)
        indices = np.asarray(next(source))
        x = train.images(indices, a.device)
        if a.jitter or a.rotate:
            x = jitter(x, a.jitter, a.rotate, aug)
        if a.group_dropout:
            u = aug.random()
            pool.drop = "tails" if u < a.group_dropout else "gap" if u < 2 * a.group_dropout else None
        if queue is not None:
            h = model._encode(x, 2, None)
            queries = torch.stack(global_code(model, pool, h, x)[:, :14].chunk(2))
            pair = queries
            with torch.no_grad():
                keys = torch.stack(
                    global_code(momentum_model, momentum_pool, momentum_model._encode(x, 2, None), x)[:, :14].chunk(2)
                )
            gauge.update(queries.detach())
            ids = torch.as_tensor(indices, device=a.device)
            patches = model.patch_readout(model._patch_pool(h, args.train_patch_grid).transpose(1, 2)).transpose(1, 2)
            terms = {
                "global": queue_infonce(queries, keys, queue, ids, gauge, args.tau),
                "patch": T.patch_contrastive_loss(patches, model, args, sim, crit, x[:, :1] != 0, None, None),
            }
            loss = terms["global"] + args.patch_loss_weight * terms["patch"]
        elif pool is None:
            _, loss, terms = T.training_objective(model, x, args, sim, crit, x[:, :1] != 0)
        else:
            h = model._encode(x, 2, None)
            patches = model.patch_readout(model._patch_pool(h, args.train_patch_grid).transpose(1, 2)).transpose(1, 2)
            code = pool(h)
            pair = torch.stack(code[:, :14].chunk(2))
            terms = {
                "global": T.contrastive_loss(code, model, args, sim, crit, None),
                "patch": T.patch_contrastive_loss(patches, model, args, sim, crit, x[:, :1] != 0, None, None),
            }
            loss = terms["global"] + args.patch_loss_weight * terms["patch"]
        if a.decorrelate:
            terms["decor"] = cross_view_decorrelation(*pair)
            loss = loss + a.decorrelate * terms["decor"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        if queue is not None:
            momentum_update(model, momentum_model, a.queue_momentum)
            if pool is not None:
                momentum_update(pool, momentum_pool, a.queue_momentum)
            queue.push(keys, ids)
        for name, value in terms.items():
            keep[name] = keep.get(name, 0.0) + float(value) / 50
        if step % 50 == 0:
            print(
                f"step {step:5d} " + " ".join(f"{k} {v:.3f}" for k, v in keep.items()) + " "
                f"{(time.time() - start) / step:.2f}s/step",
                flush=True,
            )
            last = dict(keep)
            keep = {}
        if step % a.every == 0 or step == a.steps:
            record(step, last)
            state = {"model": model.state_dict(), "step": step}
            if pool is not None:
                state["pool"] = pool.state_dict()
            torch.save(state, a.out / "model.pt")
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
