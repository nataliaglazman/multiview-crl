"""The real self-supervised objective, end to end, with no labels in training (except the --supervise diagnostic).

The pilot control's recipe and objective (global + patch InfoNCE, neighbourhood batches, recipe lr and clip), with:

  --pool gap    the recipe's global code: GAP -> head. Through T.training_objective, unchanged. The matched control.
  --pool tail   the global code reads GAP plus normative tail means (TailCode). The patch term is unchanged.
  --pool regional_tail  TailCode with averages over a 2x2x2 grid and the same normative tail means.
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
  --shared-jitter  with --jitter or --rotate, one transform per subject that moves both views alike: a training subject
                still never looks the same twice, but its two views stay matched voxel for voxel. Separates the two
                things independent jitter does.
  --patch-weight W  the patch InfoNCE's weight in place of the recipe's 1.0; at 0 the patch term is not computed. The
                patch term pairs the two views cell by cell on the 16^3 grid, so --jitter also misaligns it; this
                separates the two.
  --contrastive-loss-type barlow_twins  replace global InfoNCE with Barlow Twins on the exported code. Needs
                --patch-weight 0.
  --content-units N  the code is the first N outputs of TailCode's head in place of the recipe's 14, the head widened
                to N + 4 and the backbone and its init unchanged, so the code has room for more factors. Needs
                TailCode pooling and --patch-weight 0, since the backbone's own head is not widened.
  --remainder-units K  K more code units after the main ones (the recipe's 14, or --content-units), trained only by a
                second InfoNCE each step, on a batch of an anchor and its nearest training subjects on the main units'
                current code (no labels). They gain only by telling apart subjects the main units cannot, so they are
                paid for what the views share beyond the main code, which InfoNCE stops paying for once the batch is
                told apart (a conditional InfoNCE, as in Sordoni et al. 2021's decomposed estimator). The main units
                keep the uniform batches. --remainder-neighbours M (default 31, the tightest a batch of 32 allows) sizes
                the neighbourhoods, --remainder-every N (default 500) rebuilds them, --remainder-weight W (default 1)
                weighs the term. That batch has its own sampler and augmentation draws and leaves the running statistics
                alone, so the main batches, their augmentation and the statistics follow a run without the option.
                --remainder-space gap builds the neighbourhoods on both views' backbone GAP in place of the main
                units: everything the backbone holds is then matched, not only what the main units kept (stage 23's
                main units dropped brain size, so its remainder units learned that). Needs --pool tail or gapnorm,
                --patch-weight 0, --uniform and InfoNCE.
  --critic-weights main|remainder|both  one learnable weight per code unit inside the cosine of the main InfoNCE, of the
                remainder term or of both, shared by the two views and starting at 1, so step 0 is the recipe's loss.
                The equal-weight cosine makes a unit whose views agree poorly a cost; with weights it can count for
                less. The weights' overall size cancels in the cosine, so they only move importance between units.
                --critic-lr sets their learning rate (default 1e-3, their own AdamW group without weight decay). The
                weights are logged at each eval. Needs TailCode pooling and InfoNCE, and none of --queue,
                --hard-negatives, --half-views or --supervise; remainder and both need --remainder-units.
  --supervise volume  a diagnostic, never the method: in place of the global InfoNCE, each view's code units are read
                by their own linear map and regressed (MSE) on the rendered lesion volume, standardized on the training
                subjects, so the recipe's backbone, pool and head are trained with labels along the code's own path,
                with the same input, jitter and batches. precision_audit.py then gives the most precise volume reading
                this architecture reaches: the ceiling for any label-free objective. The InfoNCE is logged, not
                trained. Needs TailCode pooling, --patch-weight 0 and InfoNCE, and none of --queue, --hard-negatives,
                --decorrelate, --half-views or --remainder-units.
  --seed S      replicate seed, by the recipe's rule: model init S, batch order S + 10000, uniform batches S and
                augmentation S + 1. The default is the recipe's 42, so --seed 42 repeats the default run.
  INIT          cold, or start both backbones from a sup.py checkpoint.

Batches: by default an anchor and 31 of its 50 nearest training subjects on the six anatomy factors (the recipe's
oracle neighbourhoods, which use anatomy labels). --uniform draws 32 distinct subjects uniformly. --code-batches K
rebuilds the neighbourhoods every K steps on the current model's own code, both views, with no labels; each rebuild
logs how far its neighbourhoods match anatomy, lesion position and burden (labels there only score).

Each eval scores, on fresh test subjects, 5-fold ridge of volume and the anatomy latents from the 14-unit code, from
the backbone GAP, and for --pool tail from the pooled vector before the head. Labels only ever score.

    python warm.py BANK OUT INIT|cold STEPS EVERY [--pool gap|tail|regional_tail|gapnorm] [--gauge batch|running]
        [--residual]
        [--t1key t1_00] [--uniform | --code-batches K] [--queue N] [--decorrelate W] [--group-dropout P]
        [--jitter V] [--rotate D] [--shared-jitter] [--patch-weight W] [--content-units N] [--seed S]
        [--remainder-units K [--remainder-neighbours M] [--remainder-every N] [--remainder-weight W]
         [--remainder-space main|gap]] [--critic-weights main|remainder|both [--critic-lr LR]] [--supervise volume]
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
    """The global code from whole-map or regional averages plus normative tail means.

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
    REGIONAL_GRID = (2, 2, 2)

    def __init__(
        self,
        keep,
        channels=64,
        latent=18,
        momentum=0.01,
        thresholds=THRESHOLDS,
        gauge="batch",
        units=14,
        regional=False,
    ):
        super().__init__()
        if gauge not in ("batch", "running"):
            raise ValueError(f"gauge must be batch or running, got {gauge!r}")
        self.units = units  # the code is the head's first ``units`` outputs; the rest are unused style units
        self.register_buffer("keep", torch.from_numpy(keep))
        cells = int(keep.sum())
        self.register_buffer("mean", torch.zeros(2, channels, cells))
        self.register_buffer("var", torch.ones(2, channels, cells))
        self.register_buffer("started", torch.zeros(2, dtype=torch.bool))
        self.momentum = momentum
        self.thresholds = tuple(thresholds)
        self.regional = regional
        self.average_width = channels * (int(np.prod(self.REGIONAL_GRID)) if regional else 1)
        self.drop = None
        width = self.average_width + channels * 2 * len(self.thresholds)
        if gauge == "batch":
            self.gauges = nn.ModuleList(nn.BatchNorm1d(width, affine=False) for _ in range(2))
        else:
            self.gauges = nn.ModuleList(RunningGauge(width, momentum) for _ in range(2))
        self.head = nn.Sequential(nn.Linear(width, 100), nn.LeakyReLU(), nn.Linear(100, latent))

    def pooled(self, h):
        """Per view, the gauged pooled vector of (2B, C, D, H, W) view-major maps."""
        return [self.pooled_view(part, view) for view, part in enumerate(h.chunk(2))]

    def pooled_view(self, part, view, visible=None, update=True):
        """The gauged pooled vector of one view's (B, C, D, H, W) maps, with that view's statistics and gauge.

        ``visible`` (B, D, H, W) bool, for an image that shows part of the volume, pools over those cells only (the
        tails over those that are also interior). ``update=False`` leaves the view's running statistics alone and, in
        training, gauges with the batch's own statistics without storing them."""
        if visible is not None and self.regional:
            raise ValueError("Half-view pooling requires --pool tail or gapnorm")
        rows = part.flatten(2)[:, :, self.keep]
        if self.training and update:
            with torch.no_grad():
                if not self.started[view]:
                    self.mean[view] = rows.mean(0)
                    self.var[view] = rows.var(0, unbiased=False)
                    self.started[view] = True
                else:
                    self.mean[view] += self.momentum * (rows.mean(0) - self.mean[view])
                    self.var[view] += self.momentum * ((rows - self.mean[view]).square().mean(0) - self.var[view])
        z = (rows - self.mean[view]) / (self.var[view].sqrt() + 1e-6)
        if visible is None:
            average = (
                F.adaptive_avg_pool3d(part, self.REGIONAL_GRID).flatten(1) if self.regional else part.mean((2, 3, 4))
            )
            parts = [average]
            for t in self.thresholds:
                parts += [torch.log1p(100 * F.relu(z - t).mean(2)), torch.log1p(100 * F.relu(-z - t).mean(2))]
        else:
            seen = visible.flatten(1).to(part.dtype)[:, None]
            inside = seen[:, :, self.keep]
            parts = [(part.flatten(2) * seen).sum(2) / seen.sum(2).clamp_min(1)]
            for t in self.thresholds:
                for tail in (F.relu(z - t), F.relu(-z - t)):
                    parts.append(torch.log1p(100 * (tail * inside).sum(2) / inside.sum(2).clamp_min(1)))
        features = torch.cat(parts, 1)
        if self.training and not update:
            gauged = F.batch_norm(features, None, None, training=True, eps=self.gauges[view].eps)
        else:
            gauged = self.gauges[view](features)
        if self.training and self.drop:
            kept = torch.ones(gauged.shape[1], device=gauged.device)
            kept[: self.average_width] = float(self.drop != "gap")
            kept[self.average_width :] = float(self.drop != "tails")
            gauged = gauged * kept
        return gauged

    def forward(self, h):
        return torch.cat([self.head(part) for part in self.pooled(h)], 0)


class CriticWeights(nn.Module):
    """--critic-weights: one positive weight per code unit, the softplus of a parameter that starts where it is 1."""

    def __init__(self, units):
        super().__init__()
        self.theta = nn.Parameter(torch.full((units,), float(np.log(np.expm1(1.0)))))

    def forward(self):
        return F.softplus(self.theta)


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


def hard_negative_infonce(code, units, tau, concentration, sim, weight=None):
    """The recipe's global InfoNCE (each view standardized over the batch, cosine, cross-view negatives only, both
    directions summed) with each anchor's negatives weighted by exp(concentration * cosine), rescaled to mean 1, so the
    negatives most like the anchor carry most of the loss: Robinson et al. 2021's hard-negative InfoNCE without
    debiasing, their beta being concentration * tau. Concentration 0 is the recipe's loss. ``weight`` (units,), when
    given, scales each standardized unit by its square root before the cosine: --critic-weights' weighted critic."""
    b = code.shape[0] // 2
    hz = T.standardize_contrastive_code(torch.stack([code[:b, :units], code[b:, :units]]))
    if weight is not None:
        hz = hz * weight.sqrt()
    off = ~torch.eye(b, dtype=torch.bool, device=code.device)
    total = 0.0
    for i, j in ((0, 1), (1, 0)):
        cos = sim(hz[i].unsqueeze(-2), hz[j].unsqueeze(-3))
        pos, neg = cos.diag() / tau, cos[off].view(b, b - 1)
        weights = torch.softmax(concentration * neg, 1) * (b - 1)
        total = total + (torch.logsumexp(torch.cat([pos[:, None], neg / tau + weights.log()], 1), 1) - pos).mean()
    return total


def jitter(x, shift, degrees, rng, shared=False):
    """Each row of (N, C, D, H, W) cubic volumes moved by its own random rigid transform, every channel alike: shifts
    up to ``shift`` voxels and rotations up to ``degrees`` about each axis, trilinear, zero outside. ``shared`` reads
    the rows as view-major pairs (row i and row i + N/2 are one subject) and gives each pair one transform."""
    n, size = len(x), x.shape[-1]
    draws = n // 2 if shared else n
    theta = np.zeros((draws, 3, 4))
    for i, (a, b, c) in enumerate(np.deg2rad(rng.uniform(-degrees, degrees, (draws, 3)))):
        rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        theta[i, :, :3] = rz @ ry @ rx
    theta[:, :, 3] = rng.uniform(-shift, shift, (draws, 3)) * 2 / size
    if shared:
        theta = np.concatenate([theta, theta])
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


def half_masks(root, axes, size, device):
    """(2 * len(axes), 1, D, H, W) masks, the two halves of the volume along each axis (x, y, z = array dims 0, 1, 2),
    split at the centre of the training subjects' mean brain mask on that axis."""
    occupancy = (np.asarray(np.load(f"{root}/train_flair.npy", mmap_mode="r")[:300]) != 0).mean(0)
    coords = torch.arange(size, dtype=torch.float32)
    masks = []
    for axis in axes:
        dim = "xyz".index(axis)
        centre = float((np.indices(occupancy.shape)[dim] * occupancy).sum() / occupancy.sum())
        shape = [1, 1, 1, 1, 1]
        shape[2 + dim] = size
        lower = (coords < centre).view(shape).expand(1, 1, size, size, size).float()
        masks += [lower, 1 - lower]
    return torch.cat(masks).to(device)


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
            code = torch.cat([pool.head(p) for p in parts], 0)[:, : pool.units].cpu().numpy()
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
        n = feats["code"][0].shape[1]
        row["cross_view_code_r"] = float(
            np.mean(np.abs(np.diag(np.corrcoef(feats["code"][0].T, feats["code"][1].T)[:n, n:])))
        )
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bank")
    p.add_argument("out", type=Path)
    p.add_argument("init")
    p.add_argument("steps", type=int)
    p.add_argument("every", type=int)
    p.add_argument("--pool", choices=("gap", "tail", "regional_tail", "gapnorm"), default="gap")
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
    p.add_argument("--shared-jitter", action="store_true", help="One jitter transform per subject, both views alike")
    p.add_argument(
        "--patch-weight",
        type=float,
        default=None,
        help="Weight of the patch InfoNCE in place of the recipe's; 0 drops it",
    )
    p.add_argument("--contrastive-loss-type", choices=("infonce", "barlow_twins"), default="infonce")
    p.add_argument(
        "--content-units",
        type=int,
        default=None,
        help="Code width: the first N of TailCode's outputs, its head widened to match (default the recipe's 14)",
    )
    p.add_argument(
        "--remainder-units",
        type=int,
        default=None,
        help="Code units after the main ones, trained only to tell apart nearest neighbours on the main units' code",
    )
    p.add_argument(
        "--remainder-neighbours",
        type=int,
        default=None,
        help="Neighbourhood size on the main units for --remainder-units (default batch size - 1, the tightest)",
    )
    p.add_argument(
        "--remainder-every", type=int, default=None, help="Rebuild the remainder neighbourhoods every N steps (500)"
    )
    p.add_argument("--remainder-weight", type=float, default=None, help="Weight of the remainder InfoNCE (default 1)")
    p.add_argument(
        "--remainder-space",
        choices=("main", "gap"),
        default=None,
        help="What the remainder neighbourhoods are built on: the main units' code (default) or both views' GAP",
    )
    p.add_argument(
        "--critic-weights",
        choices=("main", "remainder", "both"),
        default=None,
        help="A learnable weight per code unit inside the cosine of the main InfoNCE, the remainder term or both",
    )
    p.add_argument("--critic-lr", type=float, default=None, help="Learning rate of --critic-weights (default 1e-3)")
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
    p.add_argument(
        "--code-neighbours",
        type=int,
        default=None,
        help="Neighbourhood size for --code-batches in place of the recipe's 50; smaller makes a batch more alike",
    )
    p.add_argument(
        "--hard-negatives",
        type=float,
        default=0.0,
        help="Weight each negative by exp(this x cosine), mean 1 per anchor: hard-negative InfoNCE (0 = the recipe's)",
    )
    p.add_argument(
        "--half-views",
        default=None,
        help="Axes (any of x, y, z) for a third view: each subject's T1 restricted to a random half of the volume "
        "along a random one of them, aligned with FLAIR on the first --half-units code units",
    )
    p.add_argument(
        "--half-units",
        type=int,
        default=None,
        help="Code units FLAIR shares with the half view: they can hold only what every half shows, and the rest, "
        "aligned only between the whole T1 and FLAIR, hold what some half lacks",
    )
    p.add_argument("--frozen-backbone", action="store_true")
    p.add_argument("--device", default=pick_device())
    p.add_argument("--test-subjects", type=int, default=1000)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Replicate seed for init, batch order and augmentation, by the recipe's rule (default its 42)",
    )
    p.add_argument(
        "--supervise",
        choices=("volume",),
        default=None,
        help="Diagnostic only: regress the code on the rendered lesion volume (labels) in place of InfoNCE",
    )
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
    if a.patch_weight is not None and a.patch_weight < 0:
        raise ValueError("--patch-weight cannot be negative")
    if a.contrastive_loss_type == "barlow_twins" and (
        a.patch_weight != 0 or a.queue or a.hard_negatives or a.half_views is not None or a.decorrelate
    ):
        raise ValueError(
            "Barlow Twins needs --patch-weight 0 and cannot be combined with "
            "--queue, --hard-negatives, --half-views or --decorrelate"
        )
    if a.shared_jitter and not (a.jitter or a.rotate):
        raise ValueError("--shared-jitter needs --jitter or --rotate")
    if a.content_units is not None and (a.pool == "gap" or a.patch_weight != 0):
        raise ValueError("--content-units needs tail, regional_tail or gapnorm pooling and --patch-weight 0")
    if a.content_units is not None and (a.content_units < 1 or a.decorrelate):
        raise ValueError("--content-units must be positive, and --decorrelate's weights assume the recipe's 14 units")
    if a.hard_negatives < 0 or (a.hard_negatives and (a.pool == "gap" or a.queue)):
        raise ValueError("--hard-negatives must be non-negative and needs --pool tail or gapnorm without --queue")
    if a.remainder_units is None and any(
        v is not None for v in (a.remainder_neighbours, a.remainder_every, a.remainder_weight, a.remainder_space)
    ):
        raise ValueError(
            "--remainder-neighbours, --remainder-every, --remainder-weight and --remainder-space need --remainder-units"
        )
    if a.remainder_units is not None and (
        a.remainder_units < 1
        or a.pool == "gap"
        or a.patch_weight != 0
        or not a.uniform
        or a.contrastive_loss_type != "infonce"
        or a.queue
        or a.hard_negatives
        or a.half_views is not None
        or a.decorrelate
        or a.group_dropout
    ):
        raise ValueError(
            "--remainder-units must be positive and needs --pool tail or gapnorm, --patch-weight 0, --uniform and "
            "InfoNCE, without --queue, --hard-negatives, --half-views, --decorrelate or --group-dropout"
        )
    args = run_args()
    if a.remainder_units is not None:
        a.remainder_neighbours = args.batch_size - 1 if a.remainder_neighbours is None else a.remainder_neighbours
        a.remainder_every = 500 if a.remainder_every is None else a.remainder_every
        a.remainder_weight = 1.0 if a.remainder_weight is None else a.remainder_weight
        a.remainder_space = a.remainder_space or "main"
        if a.remainder_neighbours < args.batch_size - 1 or a.remainder_every < 1 or a.remainder_weight <= 0:
            raise ValueError(
                f"--remainder-neighbours must be at least {args.batch_size - 1}, --remainder-every and "
                "--remainder-weight positive"
            )
    if a.code_neighbours is not None and (not a.code_batches or a.code_neighbours < args.batch_size - 1):
        raise ValueError(f"--code-neighbours sizes --code-batches' neighbourhoods, at least {args.batch_size - 1}")
    if (a.half_views is None) != (a.half_units is None):
        raise ValueError("--half-views and --half-units go together")
    width = args.content_channels if a.content_units is None else a.content_units
    if a.half_views is not None and (
        not a.half_views
        or set(a.half_views) - set("xyz")
        or len(set(a.half_views)) != len(a.half_views)
        or a.pool not in ("tail", "gapnorm")
        or a.gauge != "batch"
        or a.queue
        or not 1 <= a.half_units < width
    ):
        raise ValueError(
            "--half-views takes distinct axes among x, y, z and needs --pool tail or gapnorm with the BatchNorm gauge,"
            f" no --queue, and 1 <= --half-units < {width}"
        )
    if a.supervise and (
        a.pool == "gap"
        or a.patch_weight != 0
        or a.contrastive_loss_type != "infonce"
        or a.queue
        or a.hard_negatives
        or a.decorrelate
        or a.half_views is not None
        or a.remainder_units is not None
    ):
        raise ValueError(
            "--supervise needs TailCode pooling, --patch-weight 0 and InfoNCE, and none of --queue, --hard-negatives,"
            " --decorrelate, --half-views or --remainder-units"
        )
    if a.critic_weights is None and a.critic_lr is not None:
        raise ValueError("--critic-lr needs --critic-weights")
    if a.critic_weights is not None:
        a.critic_lr = 1e-3 if a.critic_lr is None else a.critic_lr
        if (
            a.pool == "gap"
            or a.contrastive_loss_type != "infonce"
            or a.queue
            or a.hard_negatives
            or a.half_views is not None
            or a.supervise
            or (a.critic_weights != "main" and a.remainder_units is None)
            or not a.critic_lr > 0
        ):
            raise ValueError(
                "--critic-weights needs TailCode pooling and InfoNCE, none of --queue, --hard-negatives, --half-views or"
                " --supervise, and a positive --critic-lr; remainder and both need --remainder-units"
            )
    a.out.mkdir(parents=True, exist_ok=False)
    if a.t1key == "t1_04":
        args.synthetic_lesion_t1_value = 0.4
    if a.seed is not None:
        args.seed, args.model_seed, args.loader_seed = a.seed, a.seed, a.seed + 10000
    if a.patch_weight is not None:
        args.patch_loss_weight = a.patch_weight
    args.contrastive_loss_type = a.contrastive_loss_type
    if a.contrastive_loss_type == "barlow_twins":
        args.standardize_code = False  # Barlow Twins standardizes the code internally.
    print(
        "seed", args.seed, "global loss", args.contrastive_loss_type, "patch weight", args.patch_loss_weight, flush=True
    )
    if a.code_neighbours or a.hard_negatives:
        print("code neighbours", a.code_neighbours, "hard negatives", a.hard_negatives, flush=True)
    if a.half_views:
        print("half views along", a.half_views, "units shared with FLAIR", a.half_units, flush=True)
    torch.manual_seed(args.model_seed)
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
    units = args.content_channels if a.content_units is None else a.content_units
    rest = a.remainder_units or 0
    pool = None
    if a.pool != "gap":
        thresholds = () if a.pool == "gapnorm" else TailCode.THRESHOLDS
        latent = units + rest + args.latent_dim - args.content_channels
        pool = TailCode(
            interior_cells(a.bank),
            latent=latent,
            thresholds=thresholds,
            gauge=a.gauge,
            units=units + rest,
            regional=a.pool == "regional_tail",
        )
        pool.main = units  # the main loss's slice; remainder units follow it
        pool = pool.to(a.device)
        print("interior cells", int(pool.keep.sum()), "tail thresholds", pool.thresholds, "gauge", a.gauge, flush=True)
        print("code units", units, flush=True)
        if rest:
            print(
                "remainder units",
                rest,
                "neighbours",
                a.remainder_neighbours,
                "rebuilt every",
                a.remainder_every,
                "weight",
                a.remainder_weight,
                "built on",
                "the main units" if a.remainder_space == "main" else "both views' GAP",
                flush=True,
            )
    args.content_channels = units  # the loss's slice; --content-units widens TailCode's head, not the backbone's
    critic = nn.ModuleDict()
    for term, width in (("main", units), ("remainder", rest)):
        if a.critic_weights in (term, "both"):
            critic[term] = CriticWeights(width).to(a.device)
    if len(critic):
        print("critic weights on", ", ".join(critic), "learning rate", a.critic_lr, flush=True)
    if a.frozen_backbone:
        for name, parameter in model.named_parameters():
            if name.startswith("encoder."):
                parameter.requires_grad_(False)
    train = Bank(a.bank, "train", a.t1key, residual=a.residual)
    readout, volume = None, None
    if a.supervise:
        with torch.random.fork_rng(devices=[]):  # the readout's init leaves every later draw as in a run without it
            readout = nn.ModuleList(nn.Linear(units, 1) for _ in range(2)).to(a.device)
        volume = (float(train.vox.mean()), float(train.vox.std()))
        print(
            "supervised on the rendered lesion volume (a diagnostic, with labels): mean",
            round(volume[0], 1),
            "SD",
            round(volume[1], 1),
            flush=True,
        )
    halves = None if a.half_views is None else half_masks(a.bank, a.half_views, train.t1.shape[-1], a.device)
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
    remainder = {"sampler": None, "spread": None}
    aug_rest = np.random.default_rng(None if args.seed is None else args.seed + 2)

    def rebuild_remainder(step):
        """Neighbourhoods for the remainder units' batches, on both views' current main-unit code or backbone GAP."""
        feats = features(model, train, a.device, pool=pool)
        if a.remainder_space == "gap":
            space = np.concatenate(feats["gap"], 1)
        else:
            space = np.concatenate([c[:, :units] for c in feats["code"]], 1)
        neighbours = cb.nearest_neighbours(space, a.remainder_neighbours)
        remainder["sampler"] = cb.NeighbourhoodBatchSampler(neighbours, args.batch_size, args.loader_seed + step, 0.0)
        remainder["spread"] = {k: round(float((v[neighbours].std(1) / v.std(0)).mean()), 3) for k, v in scored.items()}

    def rest_batches():
        while True:
            current = remainder["sampler"]
            for batch in current:
                yield batch
                if remainder["sampler"] is not current:
                    break

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
    rest_source = rest_batches() if rest else None
    parameters = [q for q in model.parameters() if q.requires_grad]
    if pool is not None:
        parameters += list(pool.parameters())
    if readout is not None:
        parameters += list(readout.parameters())
    if len(critic):
        groups = [{"params": parameters}, {"params": list(critic.parameters()), "lr": a.critic_lr, "weight_decay": 0.0}]
        optimizer = torch.optim.AdamW(groups, lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
    queue = None
    if a.queue > 0:
        queue = NegativeQueue(a.queue, units, a.device)
        gauge = CodeGauge(units, a.device)
        momentum_model = copy.deepcopy(model).eval()
        momentum_pool = None if pool is None else copy.deepcopy(pool).eval()
        for frozen in (momentum_model, momentum_pool):
            if frozen is not None:
                frozen.requires_grad_(False)
    report = {"args": {k: str(v) for k, v in vars(a).items()}, "history": []}
    if volume is not None:
        report["volume_mean_sd"] = list(volume)

    def record(step, terms):
        row = {"step": step, "loss": terms, "test": evaluate(model, test, a.device, pool), "spread": batching["spread"]}
        if rest:
            row["remainder_spread"] = remainder["spread"]
        if len(critic):
            row["critic_weights"] = {k: [round(float(w), 4) for w in v()] for k, v in critic.items()}
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
        if len(critic):
            print(
                f"step {step:5d} critic weights "
                + " ".join(f"{k} min {min(w):.2f} max {max(w):.2f}" for k, w in row["critic_weights"].items()),
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
            size = args.stratify_neighbours if a.code_neighbours is None else a.code_neighbours
            rebuild(cb.nearest_neighbours(np.concatenate(codes, 1), size), args.loader_seed + step)
            print(f"step {step:5d} code neighbourhoods rebuilt, spread {batching['spread']}", flush=True)
        if rest and (step - 1) % a.remainder_every == 0:
            rebuild_remainder(step)
            print(f"step {step:5d} remainder neighbourhoods rebuilt, spread {remainder['spread']}", flush=True)
        indices = np.asarray(next(source))
        x = train.images(indices, a.device)
        whole_t1 = x[: len(indices)]
        if a.jitter or a.rotate:
            x = jitter(x, a.jitter, a.rotate, aug, a.shared_jitter)
        if a.group_dropout:
            u = aug.random()
            pool.drop = "tails" if u < a.group_dropout else "gap" if u < 2 * a.group_dropout else None
        if queue is not None:
            h = model._encode(x, 2, None)
            queries = torch.stack(global_code(model, pool, h, x)[:, :units].chunk(2))
            pair = queries
            with torch.no_grad():
                momentum_code = global_code(momentum_model, momentum_pool, momentum_model._encode(x, 2, None), x)
                keys = torch.stack(momentum_code[:, :units].chunk(2))
            gauge.update(queries.detach())
            ids = torch.as_tensor(indices, device=a.device)
            terms = {"global": queue_infonce(queries, keys, queue, ids, gauge, args.tau)}
        elif pool is None:
            _, loss, terms = T.training_objective(model, x, args, sim, crit, x[:, :1] != 0)
        else:
            h = model._encode(x, 2, None)
            code = pool(h)
            pair = torch.stack(code[:, :units].chunk(2))
            if a.supervise:
                guess = torch.cat([head(part) for head, part in zip(readout, code[:, :units].chunk(2))])[:, 0]
                target = (train.vox[indices] - volume[0]) / volume[1]
                target = torch.as_tensor(target, dtype=guess.dtype, device=guess.device).repeat(2)
                terms = {"supervised": F.mse_loss(guess, target)}
                with torch.no_grad():  # the recipe's loss on this code, logged for comparison and not trained
                    terms["global"] = T.contrastive_loss(code, model, args, sim, crit, None)
            elif a.hard_negatives:
                terms = {"hard": hard_negative_infonce(code, units, args.tau, a.hard_negatives, sim)}
                with torch.no_grad():  # the recipe's loss, for comparison with every other arm
                    terms["global"] = T.contrastive_loss(code, model, args, sim, crit, None)
            elif "main" in critic:
                terms = {"global": hard_negative_infonce(code, units, args.tau, 0.0, sim, critic["main"]())}
                with torch.no_grad():  # the equal-weight loss, for comparison with every other arm
                    terms["unweighted"] = T.contrastive_loss(code, model, args, sim, crit, None)
            else:
                terms = {"global": T.contrastive_loss(code, model, args, sim, crit, None)}
            if halves is not None:
                b = len(indices)
                mask = halves[torch.as_tensor(aug.integers(0, len(halves), b), device=a.device)]
                x3 = torch.cat([whole_t1 * mask, mask], 1)  # the mask rides along so it moves with the image
                if a.jitter or a.rotate:
                    x3 = jitter(x3, a.jitter, a.rotate, aug)
                seen = F.avg_pool3d(x3[:, -1:], 4)[:, 0] > 0.5
                h3 = model._encode(x3[:, :-1], 1, 0)
                half = pool.head(pool.pooled_view(h3, 0, visible=seen, update=False))
                terms["half"] = hard_negative_infonce(torch.cat([code[b:], half]), a.half_units, args.tau, 0.0, sim)
            if rest:
                xr = train.images(np.asarray(next(rest_source)), a.device)
                if a.jitter or a.rotate:
                    xr = jitter(xr, a.jitter, a.rotate, aug_rest, a.shared_jitter)
                hr = model._encode(xr, 2, None)
                coder = torch.cat(
                    [pool.head(pool.pooled_view(part, view, update=False)) for view, part in enumerate(hr.chunk(2))]
                )
                weight = critic["remainder"]() if "remainder" in critic else None
                terms["remainder"] = hard_negative_infonce(coder[:, units:], rest, args.tau, 0.0, sim, weight)
        if queue is not None or pool is not None:
            loss = terms["supervised"] if a.supervise else terms["hard"] if a.hard_negatives else terms["global"]
            if halves is not None:
                loss = loss + terms["half"]
            if rest:
                loss = loss + a.remainder_weight * terms["remainder"]
            if args.patch_loss_weight > 0:
                pooled = model._patch_pool(h, args.train_patch_grid).transpose(1, 2)
                patches = model.patch_readout(pooled).transpose(1, 2)
                terms["patch"] = T.patch_contrastive_loss(patches, model, args, sim, crit, x[:, :1] != 0, None, None)
                loss = loss + args.patch_loss_weight * terms["patch"]
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
            if readout is not None:
                state["readout"] = readout.state_dict()
            if len(critic):
                state["critic"] = critic.state_dict()
            torch.save(state, a.out / "model.pt")
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
