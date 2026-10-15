"""Where does the real training loss push each global attention head, and how consistently?

A ``--global-pool attention`` encoder (``models/attention_pool.py``) pools its backbone map with
one softmax per head m over the grid positions p:

    l_m(p) = q_m . LayerNorm(x_p),   x_p = h_p + W phi(p),   a_m(p) = softmax_p l_m(p)

Here h_p is the backbone feature at p, phi(p) fixed Fourier features of p, W the learned position
weights, q_m the head's learned query, l_m(p) its logits and a_m(p) its attention weights. A head
moves its weight only by changing its logits, so the gradient of the training loss L with respect
to them says where L wants the head to look. With g = dL/da (autograd's gradient on the weights),
the softmax gives

    dL/dl_m(p) = a_m(p) (g_m(p) - sum_p' a_m(p') g_m(p')).

This script rebuilds a trained encoder-only run's options from its settings.json with the
trainer's own parser (``training.encoder_args``) and draws the first ``--batches`` K batches of
its training split in the trainer's own order (its loader seed and batching, full batches only).
For each it runs the exact training loss (``training.main_conv_synthetic.training_objective``)
forward and backward, with no optimizer step, and sums dL/dl over each view's subjects (the first
half of a batch is T1, the second FLAIR). Minus that sum is the batch's *descent map*: the change
to one logit map shared by every subject (the kind a positional encoding makes) that lowers L
fastest. Gradients are raw, before the trainer's ``--grad-clip`` and AdamW's per-parameter scaling.

The question is whether L rewards the synthetic temporal-atrophy site. That factor sits under one
bilateral Gaussian template t at a fixed site (see ``attention_template_overlap``; sigma is the
run's ``--synthetic-temporal-sigma``). If L rewards the site but the heads never get there, the
pool cannot express what L asks for; if L does not reward it, the objective is the blocker.

Per head and view, from the mean descent map D over the K batches:

split-half r    correlation of D over the first K/2 batches with D over the rest: near 1 when the
                push replicates, near 0 when it is batch noise.
r brain         correlation of D with the per-cell brain fraction (the brain mask average-pooled
                onto the grid), and r site with t (the template average-pooled onto the grid,
                summing to 1).
pull            D . t, the first-order fall in L when t is added to the head's logits. Given as a
                z-score and a rank (1 = no null template is pulled as hard) against same-shape
                templates centred farther than ``--null-min-distance`` sigma from the site: at
                random in-brain voxels (``null``), and on the sphere through the site (``shell``),
                which lies on the WM/GM boundary, so a push toward that boundary everywhere pulls
                them as hard as the site.
adjusted        the same pull on D minus its least-squares fit on the brain fraction. A loss that
                moves weight between brain and background as a whole pulls every in-brain template
                alike and ranks the others by how much of them lies in the brain, so the raw rank
                partly measures each template's brain coverage. The adjusted one asks whether the
                site stands out within that push.
realisable r    the logit change one step along the head's mean query gradient makes:
                Delta l(p) = u . k(p), where u is the unit descent direction (minus the mean of the
                K batch gradients on q_m, normalised) and k(p) the view's mean LayerNorm key at p.
                Its correlations with the brain fraction and with t.

``reading`` sorts each row: ``inconsistent`` when split-half r < 0.5, ``rewards site`` when the
adjusted pull is positive and ranks 1 against both nulls (above every null template), else ``no
site reward``. A location no different from the null centres ranks first among 1 + 20 templates
with probability 1/21, so across 8 rows a single ``rewards site`` can be chance; compare the run's
step-0 report (below) before reading much into one row.

Per head:

agreement       |mean_k g_k| / mean_k |g_k| over the K batch gradients g_k on q_m: 1 when every
                batch pushes the query the same way, about 1/sqrt(K) for pure noise (printed beside it).
dL/dW via keys  the largest |dL/dW| entry of any batch through the head's logits. Exactly zero at
                q = 0: the logits depend on W only through q. The pool line gives the total and the
                part through the pooled values, which at q = 0 only shifts every subject's pooled
                vector by the same sum_p a(p) W phi(p), whatever the head looks at.

``parity`` checks the capture: autograd's query gradient must equal sum_p dL/dl(p) k(p) rebuilt
from the captured weights; a mismatch means the wrapper caught the wrong call.

query by view   the query gradient split by the view whose logits it comes through,
                g = g_t1 + g_flair, where g_v sums dL/dl(p) k(p) over view v's subjects (k(p) is a
                subject's LayerNorm key at p). Per head: each part's |mean_k g_v,k|, its agreement
                as above, and the cosine between the two views' mean parts, near -1 when the views
                pull the one shared query in opposite directions. Its parity checks that the parts
                sum to autograd's query gradient.

Finite moves. The pull is the slope of L along an offset toward the site, and a zero slope does
not show that a head sitting on the site would not lower L. A head that starts to pool a factor
the code lacks adds a new feature to the code; the feature's cross terms with the existing code
average out over the batch's pairs, and only its squared term favours the true pairs, so InfoNCE
falls at second order. So the script also evaluates L itself with the logits offset,

    l_m(p) + s o(p),   o = the template average-pooled onto the grid and scaled to peak 1,

in both views at once (q and W are shared by the views), for all heads together and for each head
alone, at each ``--offset-scales`` s; and the same with every null template, each scaled to peak 1,
in place of the site's. Started from uniform weights, the offset gives the template's peak cells
e^s times the weight of a cell outside it; ``overlap`` per s is sum_p min(softmax(s o), t), the
share of a uniform head that the offset moves onto the site. Per head choice and s:

change          L(s) - L(0) with the site's offset, averaged over the K batches: negative when the
                offset lowers L.
first order     -s sum_views D . o, the change the descent maps predict; the change approaches it
                as s -> 0. ``beyond`` is the change minus it: what the slope misses.
z (rank)        of the site's fall (minus its change) among the null templates' falls, and of its
                fall beyond first order among theirs: rank 1 = the offset toward the site lowers L
                more than the offset toward any null template.

``reading``: ``rewards site`` when the site's change is negative and ranks 1 against both nulls,
``rewards site beyond 1st order`` when only its part beyond first order does, else ``no site
reward``. If offsets toward the site lower L more than offsets elsewhere while the pull does not
favour the site, the objective does reward the site and the uniform start is the blocker: it has
no first-order path there. If they do not, the objective is the blocker.

The sweep draws the same K batches a second time (their indices are checked). Per batch it runs
the plain step loss once, keeping the backbone map it encodes, then re-evaluates the step loss under
no_grad with ``model._encode`` returning that map and ``pool.attention`` adding the offset, so an
evaluation costs only the pool, the readout and the loss: (1 + both nulls' templates) x (1 + heads)
x scales evaluations per batch, about 60 ms each on one CPU thread at the run's size. Every
evaluation replays the random draws of the batch's training step (only a learned content mask
draws), and without an offset it reproduces the step loss (``baseline``).

Run it on the trained checkpoint and on ``--checkpoint model_init.pt`` (step 0, saved by the
trainer) of the same run: the seeds are the run's, so both see the same batches. Writes the JSON
report and, beside it, ``*_maps.npz``: every batch's descent maps, their mean and split halves,
the realisable changes, the template and the brain fraction, every batch's per-view query
gradients, and the finite moves' changes and first-order predictions per batch, offset profile
(the site's, then each null's), head choice and scale. Only encoder-only ``--global-pool
attention`` runs on the synthetic generator are accepted, as for ``attention_template_overlap``.

Example:
    python -m eval.diagnostics.attention_pool_gradients \
        --run-dir results/encoder/runs/conv_mlp_temporal009_attnfull_s42 --device cuda
    python -m eval.diagnostics.attention_pool_gradients --self-test
"""

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from eval.diagnostics.attention_template_overlap import (
    TEMPORAL_CENTRE,
    VIEWS,
    cell_coordinates,
    generator_axis,
    grid_template,
    in_brain_centres,
    overlap,
    pool_parameters,
    pool_to_grid,
    refusal,
    shell_centres,
)

# A mean descent map that replicates less than this across batch halves is mostly batch noise.
SPLIT_HALF_MIN = 0.5
# Autograd's query gradient and the one rebuilt from the captured weights agree to float32 rounding.
PARITY_TOLERANCE = 1e-3
# The offset sweep's evaluation without an offset reproduces the step loss to this share of it.
BASELINE_TOLERANCE = 1e-5


def offset_scales(text):
    """``--offset-scales``: comma-separated positive logit-offset scales."""
    try:
        scales = [float(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}")
    if not scales or not all(0 < scale < float("inf") for scale in scales):
        raise argparse.ArgumentTypeError(f"expected positive finite scales, got {text!r}")
    return scales


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", help="Encoder-only run directory holding settings.json and the checkpoint")
    p.add_argument(
        "--checkpoint", default="model.pt", help="Checkpoint filename inside --run-dir; model_init.pt is step 0"
    )
    p.add_argument("--batches", type=int, default=16, help="K: the first K training batches, in the trainer's order")
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Subjects per view in a batch (default: the run's). InfoNCE's negatives, and so its gradient, change with it",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="mps needs PYTORCH_ENABLE_MPS_FALLBACK=1 exported before Python starts",
    )
    p.add_argument("--null-templates", type=int, default=20, help="Same-shape templates drawn for each null")
    p.add_argument(
        "--null-min-distance",
        type=float,
        default=3.0,
        help="Null centres lie farther than this many sigma from the true centre",
    )
    p.add_argument("--null-seed", type=int, default=0)
    p.add_argument(
        "--offset-scales",
        type=offset_scales,
        default=[0.5, 1.0, 2.0, 4.0, 8.0],
        help="Finite moves: the logit offsets s * o to evaluate, o a template scaled to peak 1 (comma-separated)",
    )
    p.add_argument("--out", default=None, help="Report JSON (default: <run-dir>/attention_pool_gradients.json)")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)
    if args.batches < 2:
        p.error("--batches must be at least 2: the split-half correlation needs two halves")
    if args.batch_size is not None and args.batch_size < 2:
        p.error("--batch-size must be at least 2: InfoNCE needs negatives")
    if args.null_templates < 2:
        p.error("--null-templates must be at least 2 for a z-score")
    if not 0 <= args.null_min_distance < float("inf"):
        p.error("--null-min-distance must be finite and nonnegative")
    return args


def training_options(cfg, batch_size=None):
    """The run's trainer options, parsed by the trainer's own parser from its settings.json.

    Returns them with the settings keys today's trainer no longer has, which are dropped: the loss
    is then today's objective under every setting it still knows. ``batch_size`` overrides the run's.
    """
    import training.encoder_args as encoder_args

    known = {action.dest for action in encoder_args.build_parser()._actions if action.option_strings}
    dropped = sorted(key for key in cfg if key not in known)
    options = encoder_args.parse_args(encoder_args.cli_arguments({k: v for k, v in cfg.items() if k in known}))
    if batch_size is not None:
        options.batch_size = batch_size
    return options, dropped


def training_loader(options, dataset, device):
    """The trainer's training DataLoader over ``dataset``: its batching, shuffled by a generator seeded with
    ``--loader-seed``. Single-process: the sampler runs in the main process either way, so the order is the same.
    """
    import tempfile

    import torch
    from torch.utils.data import DataLoader

    import data.cluster_batches as cluster_batches
    import training.main_conv_synthetic as trainer

    with tempfile.TemporaryDirectory() as scratch:
        # A --stratify-batches run's sampler writes stratification.json beside the run; here into scratch.
        batching = cluster_batches.loader_batching(
            options, lambda: trainer.stratified_batch_sampler(options, dataset, device, scratch)
        )
    return DataLoader(dataset, num_workers=0, generator=torch.Generator().manual_seed(options.loader_seed), **batching)


def training_batches(loader, count):
    """The first ``count`` batches the trainer's loop draws: ``loader`` epoch after epoch."""
    if len(loader) == 0:
        raise SystemExit("The training split holds no full batch at this batch size")
    drawn = 0
    while True:
        for batch in loader:
            yield batch
            drawn += 1
            if drawn == count:
                return


def training_loss(model, options, device):
    """One batch -> the trainer's step loss: ``training_objective`` on the batch's images, as its loop calls it."""
    import torch

    import training.main_conv_synthetic as trainer

    sim_metric = torch.nn.CosineSimilarity(dim=-1)
    criterion = torch.nn.CrossEntropyLoss()

    def loss(batch):
        images = torch.cat(batch["image"], dim=0).as_subclass(torch.Tensor).to(device)
        masks = None
        if options.patch_foreground_mask:
            masks = torch.cat(batch["mask"], dim=0).as_subclass(torch.Tensor).to(device)
        return trainer.training_objective(model, images, options, sim_metric, criterion, masks)[1]

    return loss


@contextlib.contextmanager
def captured_attention(pool):
    """Wrap ``pool.tokens`` and ``pool.attention``; each attention call appends a record to the yielded list.

    A record holds the call's tokens x (B, N, C), detached, its weights a (B, heads, N), which keep
    their gradient (``retain_grad``), and the backbone grid (D, H, W) the tokens came from.
    """
    calls, grid = [], {}
    tokens, attention = pool.tokens, pool.attention

    def tokens_hook(h):
        grid["last"] = tuple(h.shape[2:])
        return tokens(h)

    def attention_hook(x):
        weights = attention(x)
        if weights.requires_grad:
            weights.retain_grad()
        calls.append({"tokens": x.detach(), "weights": weights, "grid": grid.get("last")})
        return weights

    pool.tokens, pool.attention = tokens_hook, attention_hook
    try:
        yield calls
    finally:
        del pool.tokens, pool.attention


def logit_gradient(weights):
    """dL/dl of a softmax over the last axis from its weights a and their gradient g: a (g - sum a g), in float64."""
    a, g = weights.detach().cpu().double(), weights.grad.detach().cpu().double()
    return a * (g - (a * g).sum(-1, keepdim=True))


def batch_gradients(model, batch, objective, calls):
    """One forward and backward of ``objective`` on ``batch``, with no optimizer step.

    ``calls`` is the list ``captured_attention`` fills. Returns numpy arrays: dL/dl summed over
    each view's subjects (2, heads, D, H, W), the query gradient (heads, C) and its part through
    each view's logits (2, heads, C), the LayerNorm keys summed over each view's subjects (2, N, C),
    and with a positional encoding the position gradient (C, 6F) in total and through each head's
    logits (heads, C, 6F); plus the loss and the parity error of the query gradient rebuilt from
    dL/dl and the keys.
    """
    import torch
    import torch.nn.functional as F

    from models.attention_pool import fourier_positions

    pool = model.attention_pool
    model.zero_grad(set_to_none=True)
    calls.clear()
    loss = objective(batch)
    loss.backward()
    if len(calls) != 1:
        raise RuntimeError(f"The attention pool ran {len(calls)} times in one training step; expected once")
    call = calls[0]
    tokens, weights, grid = call["tokens"], call["weights"], call["grid"]
    if weights.grad is None:
        raise RuntimeError("The loss does not reach the attention weights")
    b = tokens.shape[0] // 2
    dl = logit_gradient(weights)
    keys = F.layer_norm(tokens, (pool.channels,)).cpu().double()
    query_grad = pool.query.grad.detach().cpu().double()
    rebuilt = torch.einsum("bmn,bnc->mc", dl, keys)
    out = {
        "loss": float(loss.detach()),
        "logit_grad": torch.stack([dl[:b].sum(0), dl[b:].sum(0)]).reshape(2, pool.num_heads, *grid).numpy(),
        "query_grad": query_grad.numpy(),
        "query_grad_views": torch.stack(
            [torch.einsum("bmn,bnc->mc", dl[:b], keys[:b]), torch.einsum("bmn,bnc->mc", dl[b:], keys[b:])]
        ).numpy(),
        "key_sum": torch.stack([keys[:b].sum(0), keys[b:].sum(0)]).numpy(),
        "parity": float((rebuilt - query_grad).norm() / query_grad.norm().clamp_min(1e-30)),
        "grid": grid,
        "subjects": b,
    }
    if pool.position is not None:
        position = pool.position
        phi = fourier_positions(grid, pool.num_frequencies, device=tokens.device, dtype=tokens.dtype)
        # The tokens again, as a function of W alone (h held fixed), scored by the fixed queries:
        # the gradient of sum dL/dl * l through them is dL/dW through the logits, head by head.
        moved = tokens + phi @ (position - position.detach()).T
        logits = torch.einsum("bnc,mc->bmn", F.layer_norm(moved, (pool.channels,)), pool.query.detach())
        upstream = dl.to(device=logits.device, dtype=logits.dtype)
        via_keys = [
            torch.autograd.grad(logits[:, m], position, upstream[:, m], retain_graph=m + 1 < pool.num_heads)[0]
            for m in range(pool.num_heads)
        ]
        total = position.grad if position.grad is not None else torch.zeros_like(position)
        out["position_grad"] = total.detach().cpu().double().numpy()
        out["position_grad_keys"] = torch.stack(via_keys).detach().cpu().double().numpy()
    return out


def collect(model, batches, objective):
    """The attention pool's gradients under ``objective`` for each of ``batches`` (the trainer's batch dicts).

    Returns per-batch arrays (``logit_grad``, ``query_grad``, ``query_grad_views``,
    ``position_grad``, ``position_grad_keys``, ``loss``, ``parity``, ``indices``) and, over every
    subject of every batch, each view's mean LayerNorm key (``keys``, (2, N, C)) and per-cell brain
    fraction (``cell_brain``, (2, D, H, W)), and view 1's mean brain mask at full resolution
    (``voxel_brain``), from which the in-brain null centres are drawn.
    """
    import torch
    import torch.nn.functional as F

    names = ("logit_grad", "query_grad", "query_grad_views", "position_grad", "position_grad_keys", "loss", "parity")
    per_batch = {name: [] for name in names}
    key_sum = brain_sum = voxel_sum = 0.0
    subjects, indices = 0, []
    started = time.perf_counter()
    with captured_attention(model.attention_pool) as calls:
        for k, batch in enumerate(batches):
            out = batch_gradients(model, batch, objective, calls)
            for name in names:
                per_batch[name].append(out.get(name))
            key_sum = key_sum + out["key_sum"]
            masks = [m.as_subclass(torch.Tensor).double() for m in batch["mask"]]
            brain_sum = (
                brain_sum + torch.stack([F.adaptive_avg_pool3d(m, out["grid"])[:, 0].sum(0) for m in masks]).numpy()
            )
            voxel_sum = voxel_sum + masks[0][:, 0].sum(0).numpy()
            subjects += out["subjects"]
            if "index" in batch:
                indices.append([int(i) for i in batch["index"]])
            print(
                f"  batch {k + 1}: loss {out['loss']:.4f}, parity {out['parity']:.1e} "
                f"({time.perf_counter() - started:.0f} s)",
                flush=True,
            )
    collected = {name: np.stack(values) if values[0] is not None else None for name, values in per_batch.items()}
    collected.update(
        keys=key_sum / subjects, cell_brain=brain_sum / subjects, voxel_brain=voxel_sum / subjects, indices=indices
    )
    return collected


def generator_states(device):
    """The global random states a step on ``device`` draws from: the CPU generator's and the device's own."""
    import torch

    states = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        states["cuda"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        states["mps"] = torch.mps.get_rng_state()
    return states


def restore_generators(states, device):
    """Set the global random states ``generator_states`` returned."""
    import torch

    torch.set_rng_state(states["cpu"])
    if "cuda" in states:
        torch.cuda.set_rng_state(states["cuda"], device)
    if "mps" in states:
        torch.mps.set_rng_state(states["mps"])


def offset_losses(model, batches, objective, profiles, scales):
    """``objective`` on each of ``batches`` with the global heads' logits offset to l_m(p) + s o(p).

    For every profile o in ``profiles`` (T, D, H, W), scale s in ``scales`` and head choice: all
    heads at once, then each head alone; an offset applies to both views. Per batch, the first
    evaluation is the plain step and keeps each backbone map it encodes. Every later one runs with
    ``model._encode`` returning those maps in call order and ``pool.attention`` returning
    softmax(q . LayerNorm(x) + offset). All run under no_grad, and each starts from the random
    state the first did, which the batch then leaves as one step does: a pass seeded as
    ``collect``'s was replays its draws.

    Returns ``baseline``, the plain step loss (K,), ``losses`` with each offset (K, T, 1 + heads,
    S), and the batches' ``indices``.
    """
    import torch
    import torch.nn.functional as F

    pool = model.attention_pool
    encode, attention = model._encode, pool.attention
    parameter = next(model.parameters())
    device = parameter.device
    profiles = torch.as_tensor(np.reshape(profiles, (len(profiles), 1, -1)), dtype=parameter.dtype, device=device)
    heads = torch.eye(pool.num_heads, dtype=parameter.dtype, device=device)
    # (1 + heads, heads, 1): which heads' logits an offset moves.
    choices = torch.cat([heads.sum(0, keepdim=True), heads])[..., None]
    plan = [(profile, choice, scale) for profile in profiles for choice in choices for scale in scales]
    state = {"maps": [], "next": None, "offset": None, "attention": 0}

    def cached(x, n_views, view_idx=None):
        if state["next"] is None:
            state["maps"].append(encode(x, n_views, view_idx))
            return state["maps"][-1]
        if state["next"] == len(state["maps"]):
            raise RuntimeError("An offset evaluation encoded more often than the plain step")
        state["next"] += 1
        return state["maps"][state["next"] - 1]

    def shifted(x):
        state["attention"] += 1
        if state["offset"] is None:
            return attention(x)
        logits = torch.einsum("bnc,mc->bmn", F.layer_norm(x, (pool.channels,)), pool.query)
        return (logits + state["offset"]).softmax(dim=-1)

    def evaluate(batch, offset):
        state.update(offset=offset, attention=0)
        loss = objective(batch)
        if state["attention"] != 1:
            raise RuntimeError(f"The attention pool ran {state['attention']} times in one evaluation; expected once")
        if state["next"] not in (None, len(state["maps"])):
            raise RuntimeError("An offset evaluation encoded less often than the plain step")
        return loss.detach()

    baseline, losses, indices = [], [], []
    started = time.perf_counter()
    model._encode, pool.attention = cached, shifted
    try:
        with torch.no_grad():
            for k, batch in enumerate(batches):
                # On the device once, so the objective's own transfer of the inputs is a no-op.
                staged = dict(batch)
                for key in ("image", "mask"):
                    if key in batch:
                        staged[key] = [part.as_subclass(torch.Tensor).to(device) for part in batch[key]]
                start = generator_states(device)
                state.update(maps=[], next=None)
                values = [evaluate(staged, None)]
                for profile, choice, scale in plan:
                    restore_generators(start, device)
                    state["next"] = 0
                    values.append(evaluate(staged, scale * choice * profile))
                values = torch.stack(values).cpu().double().numpy()
                baseline.append(values[0])
                losses.append(values[1:].reshape(len(profiles), len(choices), len(scales)))
                if "index" in batch:
                    indices.append([int(i) for i in batch["index"]])
                print(
                    f"  offsets, batch {k + 1}: loss {values[0]:.4f} without offset, {len(plan)} offsets "
                    f"({time.perf_counter() - started:.0f} s)",
                    flush=True,
                )
    finally:
        del model._encode, pool.attention
    return {"baseline": np.array(baseline), "losses": np.stack(losses), "indices": indices}


def correlation(a, b):
    """Pearson r over every cell; None when either map is constant."""
    a = np.ravel(a) - np.mean(a)
    b = np.ravel(b) - np.mean(b)
    scale = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / scale) if scale > 0 else None


def brain_residual(descent, brain):
    """``descent`` minus its least-squares fit c0 + c1 * brain fraction over the cells."""
    x = np.ravel(brain) - np.mean(brain)
    y = np.ravel(descent) - np.mean(descent)
    slope = float(x @ y) / float(x @ x) if x @ x > 0 else 0.0
    return (y - slope * x).reshape(np.shape(descent))


def z_rank(value, scores):
    """``value``'s z-score in units of the null ``scores``' SD (None when they are constant), and its rank.

    Rank 1 means no null score reaches ``value``.
    """
    sd = float(scores.std(ddof=1))
    return ((value - float(scores.mean())) / sd if sd > 0 else None), int(1 + (scores >= value).sum())


def site_pull(descent, template, nulls, prefix=""):
    """``descent . template``, and its z-score and rank among the same products with each null's templates.

    Rank 1 means no null template is pulled at least as hard as the site; z is in units of the
    null pulls' SD.
    """
    pull = float((descent * template).sum())
    out = {f"{prefix}pull": pull}
    for name, stack in nulls.items():
        scores = (stack * descent).sum(axis=(-3, -2, -1))
        out[f"{prefix}{name}_median"] = float(np.median(scores))
        out[f"{prefix}{name}_p90"] = float(np.percentile(scores, 90))
        out[f"{prefix}{name}_z"], out[f"{prefix}{name}_rank"] = z_rank(pull, scores)
    return out


def reading(row):
    if row["split_half_r"] is None or row["split_half_r"] < SPLIT_HALF_MIN:
        return "inconsistent"
    if row["adjusted_pull"] > 0 and row["adjusted_null_rank"] == 1 and row["adjusted_shell_null_rank"] == 1:
        return "rewards site"
    return "no site reward"


def query_views(parts, total):
    """Each head's query gradient split by view, from the per-batch parts (K, 2, heads, C) and autograd's total.

    Returns a row per head (each part's |mean_k g_k|, mean_k |g_k| and agreement, and the cosine
    between the two views' mean parts) and the largest relative error of the parts' sum against
    ``total`` (K, heads, C) over the batches.
    """
    mean = parts.mean(0)
    mean_norm = np.linalg.norm(mean, axis=-1)
    batch_norm = np.linalg.norm(parts, axis=-1).mean(0)
    rows = []
    for head in range(parts.shape[2]):
        row = {"head": head}
        for v, view in enumerate(VIEWS):
            row[f"{view}_norm"] = float(mean_norm[v, head])
            row[f"{view}_batch_norm"] = float(batch_norm[v, head])
            row[f"{view}_agreement"] = (
                float(mean_norm[v, head] / batch_norm[v, head]) if batch_norm[v, head] > 0 else None
            )
        scale = float(mean_norm[0, head] * mean_norm[1, head])
        row["cosine"] = float(mean[0, head] @ mean[1, head]) / scale if scale > 0 else None
        rows.append(row)
    flat = total.reshape(len(total), -1)
    residual = np.linalg.norm(parts.sum(1).reshape(len(total), -1) - flat, axis=1)
    return rows, float((residual / np.maximum(np.linalg.norm(flat, axis=1), 1e-30)).max())


def offset_profiles(template, nulls):
    """The site's template and then every null template, each scaled to peak 1, (T, D, H, W); and each null's rows."""
    profiles = np.concatenate([template[None], *nulls.values()])
    profiles = profiles / profiles.max(axis=(1, 2, 3), keepdims=True)
    groups, start = {}, 1
    for name, stack in nulls.items():
        groups[name] = slice(start, start + len(stack))
        start += len(stack)
    return profiles, groups


def move_reading(row):
    if row["change"] < 0 and row["null_rank"] == 1 and row["shell_null_rank"] == 1:
        return "rewards site"
    if row["beyond_first_order"] < 0 and row["beyond_null_rank"] == 1 and row["beyond_shell_null_rank"] == 1:
        return "rewards site beyond 1st order"
    return "no site reward"


def score_moves(moves, descent, profiles, groups, scales, step_loss):
    """Per head choice and scale: the site's loss change, its first-order prediction, and both against the nulls'.

    ``moves`` is ``offset_losses``' output for ``profiles`` (the site's first; ``groups`` gives each
    null's rows), ``descent`` the batches' descent maps (K, 2, heads, D, H, W) and ``step_loss``
    their step losses, which the evaluations without an offset must reproduce. Returns the report
    section and the per-batch arrays.
    """
    baseline = np.asarray(moves["baseline"], dtype=np.float64)
    change = np.asarray(moves["losses"], dtype=np.float64) - baseline[:, None, None, None]
    # dL/ds at s = 0 is minus each profile's pull, summed over both views and the chosen heads.
    pulls = np.einsum("kvmdhw,tdhw->ktm", descent, profiles)
    pulls = np.concatenate([pulls.sum(-1, keepdims=True), pulls], axis=-1)
    first = -pulls[..., None] * np.asarray(scales, dtype=np.float64)
    mean_change, mean_first = change.mean(0), first.mean(0)
    beyond = mean_change - mean_first
    choices = ["all", *range(descent.shape[2])]
    rows = []
    for c, heads in enumerate(choices):
        for s, scale in enumerate(scales):
            row = {
                "heads": heads,
                "scale": float(scale),
                "change": float(mean_change[0, c, s]),
                "first_order": float(mean_first[0, c, s]),
                "beyond_first_order": float(beyond[0, c, s]),
            }
            for name, part in groups.items():
                row[f"{name}_median_change"] = float(np.median(mean_change[part, c, s]))
                row[f"{name}_z"], row[f"{name}_rank"] = z_rank(-row["change"], -mean_change[part, c, s])
            for name, part in groups.items():
                z, rank = z_rank(-row["beyond_first_order"], -beyond[part, c, s])
                row[f"beyond_{name}_z"], row[f"beyond_{name}_rank"] = z, rank
            row["reading"] = move_reading(row)
            rows.append(row)
    site = profiles[0]
    overlaps = []
    for scale in scales:
        weights = np.exp(scale * (site - site.max()))
        overlaps.append(float(overlap(weights / weights.sum(), site / site.sum())))
    report = {
        "scales": [float(scale) for scale in scales],
        "head_choices": choices,
        "profiles": {"site": 1, **{name: part.stop - part.start for name, part in groups.items()}},
        "batches": int(len(baseline)),
        "baseline_loss": float(baseline.mean()),
        "baseline_vs_step_max_abs": float(np.abs(baseline - np.asarray(step_loss, dtype=np.float64)).max()),
        "overlap_from_uniform": overlaps,
        "rows": rows,
    }
    arrays = {
        "move_change": change,
        "move_first_order": first,
        "move_profiles": profiles,
        "move_scales": np.asarray(scales, dtype=np.float64),
    }
    return report, arrays


def analyse(collected, res, sigma, null_count=20, min_distance=3.0, seed=0):
    """Score every head and view's descent map against the template and both nulls; returns the report and maps.

    ``collected`` is ``collect``'s output. The null centres use ``attention_template_overlap``'s
    seeds, so with the same mean brain mask the two scripts compare against the same templates;
    ``arrays["nulls"]`` holds those templates for the offset sweep. Also splits each head's query
    gradient by view.
    """
    descent = -np.asarray(collected["logit_grad"], dtype=np.float64)
    count, n_heads = descent.shape[0], descent.shape[2]
    grid = tuple(descent.shape[3:])
    template = grid_template(res, sigma, grid)
    centres = {
        "null": in_brain_centres(collected["voxel_brain"], sigma, null_count, min_distance, (seed, 0)),
        "shell_null": shell_centres(collected["voxel_brain"], sigma, null_count, min_distance, (seed, 1)),
    }
    for name, points in centres.items():
        if len(points) < 2:
            raise ValueError(f"Fewer than two {name} centres lie in the brain farther than {min_distance:g} sigma")
    nulls = {
        name: np.stack([grid_template(res, sigma, grid, tuple(point)) for point in points])
        for name, points in centres.items()
    }
    brain = np.asarray(collected["cell_brain"], dtype=np.float64)
    half = count // 2
    mean = descent.mean(0)
    halves = np.stack([descent[:half].mean(0), descent[half:].mean(0)])

    query = np.asarray(collected["query_grad"], dtype=np.float64)
    mean_query = query.mean(0)
    mean_norm = np.linalg.norm(mean_query, axis=1)
    batch_norm = np.linalg.norm(query, axis=2).mean(0)
    direction = -mean_query / np.where(mean_norm > 0, mean_norm, 1.0)[:, None]
    realisable = np.einsum("vnc,mc->vmn", collected["keys"], direction).reshape(2, n_heads, *grid)

    rows = []
    for v, view in enumerate(VIEWS):
        for head in range(n_heads):
            d = mean[v, head]
            row = {
                "view": view,
                "head": head,
                "split_half_r": correlation(halves[0, v, head], halves[1, v, head]),
                "r_brain": correlation(d, brain[v]),
                "r_template": correlation(d, template),
                **site_pull(d, template, nulls),
                **site_pull(brain_residual(d, brain[v]), template, nulls, "adjusted_"),
                "realisable_r_brain": correlation(realisable[v, head], brain[v]) if mean_norm[head] > 0 else None,
                "realisable_r_template": correlation(realisable[v, head], template) if mean_norm[head] > 0 else None,
            }
            row["reading"] = reading(row)
            rows.append(row)

    keys_path = collected.get("position_grad_keys")
    heads = []
    for head in range(n_heads):
        row = {
            "head": head,
            "agreement": float(mean_norm[head] / batch_norm[head]) if batch_norm[head] > 0 else None,
            "mean_gradient_norm": float(mean_norm[head]),
            "batch_gradient_norm": float(batch_norm[head]),
            "position_grad_keys_max_abs": None if keys_path is None else float(np.abs(keys_path[:, head]).max()),
        }
        heads.append(row)
    position = None
    if collected.get("position_grad") is not None:
        total, through_keys = np.asarray(collected["position_grad"]), np.asarray(keys_path).sum(1)
        position = {
            "max_abs": float(np.abs(total).max()),
            "via_keys_max_abs": float(np.abs(through_keys).max()),
            "via_values_max_abs": float(np.abs(total - through_keys).max()),
        }
    parts = np.asarray(collected["query_grad_views"], dtype=np.float64)
    views, views_parity = query_views(parts, query)
    loss = np.asarray(collected["loss"], dtype=np.float64)
    report = {
        "sigma": float(sigma),
        "res": int(res),
        "grid": list(grid),
        "batches": int(count),
        "split_halves": [[0, half], [half, int(count)]],
        "noise_agreement": 1.0 / math.sqrt(count),
        "loss": {"mean": float(loss.mean()), "per_batch": loss.tolist()},
        "parity_max": float(np.max(collected["parity"])),
        "nulls": {
            name: {
                "requested": int(null_count),
                "count": int(len(points)),
                "min_distance_sigma": float(min_distance),
                "seed": int(seed),
                "centres_folded": points.tolist(),
            }
            for name, points in centres.items()
        },
        "rows": rows,
        "heads": heads,
        "position": position,
        "query_views": views,
        "query_views_parity_max": views_parity,
    }
    arrays = {
        "descent": descent,
        "mean_descent": mean,
        "half_descent": halves,
        "realisable": realisable,
        "template": template,
        "cell_brain": brain,
        "query_grad_views": parts,
        "nulls": nulls,
    }
    return report, arrays


def print_report(report):
    grid = "x".join(str(g) for g in report["grid"])
    loss = report["loss"]["per_batch"]
    print(
        f"\n=== where the training loss pushes each attention head ({report['batches']} training batches of "
        f"{report.get('batch_size', '?')} subjects per view; sigma {report['sigma']:g}, {grid} grid) ===",
        flush=True,
    )
    print(
        f"  loss {report['loss']['mean']:.4f} (batches {min(loss):.4f} to {max(loss):.4f}); "
        f"query-gradient parity {report['parity_max']:.1e}",
        flush=True,
    )
    if report["parity_max"] > PARITY_TOLERANCE:
        print("  WARNING: the rebuilt query gradient disagrees with autograd's; the capture is suspect", flush=True)
    for name, null in report["nulls"].items():
        if null["count"] < null["requested"]:
            print(f"  WARNING: only {null['count']} of {null['requested']} {name} centres could be drawn", flush=True)

    def r(value):
        return "n/a" if value is None else f"{value:+.2f}"

    def zr(row, key):
        z = row[f"{key}_z"]
        return f"{'n/a' if z is None else format(z, '+.2f')} ({row[f'{key}_rank']})"

    print(
        f"\n  {'view':<7s}{'head':>4s}{'split r':>9s}{'r brain':>9s}{'r site':>8s}   {'pull z (rank): null':>20s}"
        f"{'shell':>12s}   {'adjusted: null':>14s}{'shell':>12s}   {'realisable r: brain':>19s}{'site':>7s}  reading",
        flush=True,
    )
    for row in report["rows"]:
        print(
            f"  {row['view']:<7s}{row['head']:>4d}{r(row['split_half_r']):>9s}{r(row['r_brain']):>9s}"
            f"{r(row['r_template']):>8s}   {zr(row, 'null'):>20s}{zr(row, 'shell_null'):>12s}   "
            f"{zr(row, 'adjusted_null'):>14s}{zr(row, 'adjusted_shell_null'):>12s}   "
            f"{r(row['realisable_r_brain']):>19s}{r(row['realisable_r_template']):>7s}  {row['reading']}",
            flush=True,
        )
    rewarded = sum(row["reading"] == "rewards site" for row in report["rows"])
    print(
        f"  {rewarded} of {len(report['rows'])} rows read 'rewards site' (adjusted pull above every null template; "
        "a location like the null centres does that with probability 1/21 per null)",
        flush=True,
    )
    pool = report.get("pool") or {}
    norms = pool.get("query_norm") or [None] * len(report["heads"])
    print(
        f"\n  {'head':>6s}{'|q|':>7s}{'agreement':>11s}{'|mean g|':>11s}{'mean |g_k|':>12s}"
        f"{'max |dL/dW| via keys':>22s}",
        flush=True,
    )
    for row, norm in zip(report["heads"], norms):
        agreement = "n/a" if row["agreement"] is None else f"{row['agreement']:.2f}"
        keys = row["position_grad_keys_max_abs"]
        print(
            f"  {row['head']:>6d}{'' if norm is None else format(norm, '.3f'):>7s}{agreement:>11s}"
            f"{row['mean_gradient_norm']:>11.2e}{row['batch_gradient_norm']:>12.2e}"
            f"{'n/a' if keys is None else format(keys, '.2e'):>22s}",
            flush=True,
        )
    position = report["position"]
    if position is not None:
        print(
            f"  position weights W: max |dL/dW| {position['max_abs']:.2e} in total, {position['via_keys_max_abs']:.2e} "
            f"through the logits (where heads look), {position['via_values_max_abs']:.2e} through the pooled values",
            flush=True,
        )
    first, second = report["split_halves"]
    print(
        f"  split r: the mean descent map over batches 1-{first[1]} vs {second[0] + 1}-{second[1]}; "
        "r brain / r site: its correlation with the per-cell\n"
        "  brain fraction and the template. pull = descent . template, as a z-score and rank (1 = above every null)\n"
        "  against same-shape templates at random in-brain centres (null) and on the site's sphere (shell); adjusted\n"
        "  first removes the descent map's fit on the brain fraction. realisable: the logit change of one step along\n"
        f"  the mean query gradient. agreement = |mean_k g_k| / mean_k |g_k|; pure noise gives about 1/sqrt(K) = "
        f"{report['noise_agreement']:.2f}.",
        flush=True,
    )
    views = report.get("query_views")
    if views is not None:
        print_query_views(views, report["query_views_parity_max"])
    if report.get("finite_move") is not None:
        print_moves(report["finite_move"])


def print_query_views(views, parity):
    print(
        "\n  query gradient by view: g = g_t1 + g_flair, g_v = sum over view v's subjects of dL/dl times the "
        f"LayerNorm keys\n  (the parts' sum against autograd's query gradient: parity {parity:.1e})",
        flush=True,
    )
    if parity > PARITY_TOLERANCE:
        print("  WARNING: the per-view parts do not sum to autograd's query gradient", flush=True)

    def agreement(value):
        return "n/a" if value is None else f"{value:.2f}"

    print(
        f"  {'head':>6s}{'|mean g_t1|':>13s}{'|mean g_flair|':>16s}{'agreement: t1':>15s}{'flair':>7s}"
        f"{'cos(t1, flair)':>16s}",
        flush=True,
    )
    for row in views:
        cosine = "n/a" if row["cosine"] is None else f"{row['cosine']:+.2f}"
        print(
            f"  {row['head']:>6d}{row['t1_norm']:>13.2e}{row['flair_norm']:>16.2e}"
            f"{agreement(row['t1_agreement']):>15s}{agreement(row['flair_agreement']):>7s}{cosine:>16s}",
            flush=True,
        )
    print(
        "  |mean g_v|: the view's part averaged over the batches; agreement as above, per view; cos near -1: the views\n"
        "  pull the one shared query in opposite directions.",
        flush=True,
    )


def print_moves(moves):
    others = ", ".join(
        f"{count} {name.replace('_', ' ')}" for name, count in moves["profiles"].items() if name != "site"
    )
    print(
        f"\n=== finite moves: the step loss with logits offset by s * o, o a template scaled to peak 1, in both views "
        f"({moves['batches']} batches) ===",
        flush=True,
    )
    print(
        f"  the site's template, and {others} templates in its place; without offset the loss is "
        f"{moves['baseline_loss']:.4f}, the step loss to {moves['baseline_vs_step_max_abs']:.1e}",
        flush=True,
    )
    if moves["baseline_vs_step_max_abs"] > BASELINE_TOLERANCE * max(1.0, abs(moves["baseline_loss"])):
        print(
            "  WARNING: without offset the sweep does not reproduce the step loss; the cached map is suspect",
            flush=True,
        )
    print(
        "  overlap of softmax(s o) with the template, from uniform weights: "
        + ", ".join(f"s {s:g}: {o:.2f}" for s, o in zip(moves["scales"], moves["overlap_from_uniform"])),
        flush=True,
    )

    def zr(row, key):
        z = row[f"{key}_z"]
        return f"{'n/a' if z is None else format(z, '+.2f')} ({row[f'{key}_rank']})"

    print(
        f"\n  {'heads':<6s}{'s':>5s}{'change':>11s}{'1st order':>11s}{'beyond':>11s}   {'fall z (rank): null':>20s}"
        f"{'shell':>12s}   {'beyond: null':>12s}{'shell':>12s}  reading",
        flush=True,
    )
    for row in moves["rows"]:
        print(
            f"  {str(row['heads']):<6s}{row['scale']:>5g}{row['change']:>11.2e}{row['first_order']:>11.2e}"
            f"{row['beyond_first_order']:>11.2e}   {zr(row, 'null'):>20s}{zr(row, 'shell_null'):>12s}   "
            f"{zr(row, 'beyond_null'):>12s}{zr(row, 'beyond_shell_null'):>12s}  {row['reading']}",
            flush=True,
        )
    rewarded = sum(row["reading"] == "rewards site" for row in moves["rows"])
    beyond = sum(row["reading"] != "no site reward" for row in moves["rows"])
    print(
        f"  {rewarded} of {len(moves['rows'])} rows read 'rewards site' and {beyond} reward it at least beyond first "
        "order (rows share batches and templates, so they are not independent)",
        flush=True,
    )
    print(
        "  change: L(s) - L(0) with the site's offset, mean over batches (negative = the loss falls). 1st order:\n"
        "  -s sum_views D . o from the descent maps above; beyond = change - 1st order. fall z (rank): the site's fall\n"
        "  (minus its change) against the null templates' falls (rank 1 = falls more than with any null template);\n"
        "  beyond: the same for the fall beyond first order. heads: all at once, or one head alone.",
        flush=True,
    )


def write_outputs(report, arrays, out):
    """The JSON report, and the descent and realisable maps, the per-view query gradients and the finite moves beside it."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fp:
        json.dump(report, fp, indent=2, default=float)
    maps_path = os.path.splitext(out)[0] + "_maps.npz"
    added = ("query_grad_views", "move_change", "move_first_order", "move_profiles", "move_scales")
    np.savez_compressed(
        maps_path,
        views=np.array(VIEWS),
        descent=arrays["descent"].astype(np.float32),
        mean_descent=arrays["mean_descent"].astype(np.float32),
        half_descent=arrays["half_descent"].astype(np.float32),
        realisable=arrays["realisable"].astype(np.float32),
        template=arrays["template"].astype(np.float32),
        cell_brain=arrays["cell_brain"].astype(np.float32),
        **{name: np.asarray(arrays[name], dtype=np.float32) for name in added if name in arrays},
    )
    print(f"\nwrote {out}\nwrote {maps_path}", flush=True)
    return out, maps_path


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        _self_test()
        return
    if not args.run_dir:
        raise SystemExit("--run-dir is required (or pass --self-test)")

    import torch

    import training.main_conv_synthetic as trainer
    from eval.protocol.score_checkpoint import build_model, load_settings
    from utils.encoder_runtime import select_encoder_device

    cfg = load_settings(args.run_dir)
    reason = refusal(cfg)
    if reason is not None:
        raise SystemExit(reason)
    options, dropped = training_options(cfg, args.batch_size)
    if options.batch_size > options.num_train_samples:
        raise SystemExit(f"--batch-size {options.batch_size} exceeds the {options.num_train_samples} training subjects")
    device = select_encoder_device(args.device, False)
    sigma = float(options.synthetic_temporal_sigma)
    print(f"run       : {args.run_dir} ({args.checkpoint})", flush=True)
    print(
        f"objective : {options.contrastive_loss_type}, tau {options.tau:g}, batch {options.batch_size}"
        + (f" (the run's is {cfg['batch_size']})" if options.batch_size != cfg["batch_size"] else "")
        + f"; first {args.batches} training batches, loader seed {options.loader_seed}",
        flush=True,
    )
    print(
        f"device    : {device}   pool: attention, width {options.attention_pool_width}, "
        f"{options.attention_pool_heads} heads; template sigma {sigma:g}, res {options.res}",
        flush=True,
    )
    if dropped:
        print(f"  WARNING: today's trainer has no option for these settings, dropped: {', '.join(dropped)}", flush=True)

    state = torch.load(os.path.join(args.run_dir, args.checkpoint), map_location="cpu")
    # build_model applies the run's runtime settings (threads, determinism), as the trainer did.
    model = build_model(cfg, device, state).train()
    del state
    # Each subject is read once, so the trainer's in-memory cache would only hold memory.
    options.cache = False
    dataset = trainer.make_dataset(options, "train", options.num_train_samples)
    loader = training_loader(options, dataset, device)
    objective = training_loss(model, options, device)
    # Only a learned content mask draws from the global RNG within a step; seeded so a rerun repeats it.
    torch.manual_seed(options.seed)
    collected = collect(model, training_batches(loader, args.batches), objective)
    report, arrays = analyse(collected, options.res, sigma, args.null_templates, args.null_min_distance, args.null_seed)
    report = {
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "split": "train",
        "batch_size": int(options.batch_size),
        "loader_seed": int(options.loader_seed),
        **report,
        "pool": pool_parameters(model.attention_pool),
        "batch_indices": collected["indices"],
        "dropped_settings": dropped,
        "settings": cfg,
    }
    print_report(report)

    profiles, groups = offset_profiles(arrays["template"], arrays["nulls"])
    print(
        f"\nfinite moves: {len(profiles)} templates x {1 + model.attention_pool.num_heads} head choices x "
        f"{len(args.offset_scales)} scales on each of the {args.batches} batches, drawn again",
        flush=True,
    )
    loader = training_loader(options, dataset, device)
    # Seeded as for the gradients, so each batch's evaluations replay its step's draws.
    torch.manual_seed(options.seed)
    moves = offset_losses(model, training_batches(loader, args.batches), objective, profiles, args.offset_scales)
    if moves["indices"] != collected["indices"]:
        raise SystemExit(
            "The offset sweep drew other batches than the gradients; the trainer's order is not reproducible"
        )
    report["finite_move"], move_arrays = score_moves(
        moves, arrays["descent"], profiles, groups, args.offset_scales, collected["loss"]
    )
    arrays.update(move_arrays)
    print_moves(report["finite_move"])
    write_outputs(report, arrays, args.out or os.path.join(args.run_dir, "attention_pool_gradients.json"))


def _self_test():
    """Planted descent maps through the analysis (numpy), then the capture, the gradients and the CLI (torch)."""
    res, sigma, grid = 64, 0.09, (16, 16, 16)
    count, channels = 8, 32
    coords = cell_coordinates(res, grid)
    template = grid_template(res, sigma, grid)
    g = generator_axis(res)
    voxel_brain = (np.linalg.norm(np.stack(np.meshgrid(g, g, g, indexing="ij")), axis=0) < 0.65).astype(float)
    cell_brain = pool_to_grid(voxel_brain, grid)
    shell = np.exp(-((np.linalg.norm(coords, axis=0) - np.linalg.norm(TEMPORAL_CENTRE)) ** 2) / (2 * sigma**2))
    elsewhere = grid_template(res, sigma, grid, (0.0, 0.3, 0.2))
    # Five heads with known answers: the site, a push off the brain, noise alone, a bump elsewhere, the site's sphere.
    signal = np.stack([template, -cell_brain, np.zeros(grid), elsewhere, shell / shell.sum()])
    signal -= signal.mean(axis=(1, 2, 3), keepdims=True)
    n_heads = len(signal)
    rng = np.random.default_rng(0)
    descent = signal[None, None] + rng.normal(scale=1e-4, size=(count, 2, n_heads, *grid))
    # Key channel 0 is the site (peak 1), channel 1 the brain fraction, the rest noise.
    keys = np.concatenate(
        [
            (template / template.max()).reshape(1, -1, 1),
            cell_brain.reshape(1, -1, 1),
            rng.normal(size=(1, template.size, channels - 2)),
        ],
        axis=2,
    ).repeat(2, axis=0)
    unit = np.eye(channels)
    query = np.stack([-unit[0], unit[1], np.zeros(channels), -unit[0], -unit[0]])[None].repeat(count, axis=0)
    query = query + rng.normal(scale=1e-4, size=query.shape)
    query[:, 2] = rng.normal(size=(count, channels))
    # Per-view parts that sum to the query gradient: head 0's views pull it to -e0/2 + 3 e5 and -e0/2 - 3 e5.
    split = np.zeros((count, n_heads, channels))
    split[:, 0, 5] = 3.0
    position_keys = np.zeros((count, n_heads, 4, 6))
    position = np.zeros((count, 4, 6))
    position[3, 1, 2] = -0.04
    collected = {
        "logit_grad": -descent,
        "query_grad": query,
        "query_grad_views": np.stack([query / 2 + split, query / 2 - split], axis=1),
        "position_grad": position,
        "position_grad_keys": position_keys,
        "keys": keys,
        "cell_brain": np.stack([cell_brain] * 2),
        "voxel_brain": voxel_brain,
        "loss": np.full(count, 2.0),
        "parity": np.zeros(count),
    }
    report, arrays = analyse(collected, res, sigma)
    rows = {(row["view"], row["head"]): row for row in report["rows"]}
    site, off_brain, noise, other, sphere = (rows[("t1", m)] for m in range(n_heads))
    assert site["reading"] == "rewards site" and site["null_rank"] == 1 and site["shell_null_rank"] == 1, site
    assert site["adjusted_null_rank"] == 1 and site["null_z"] > 3 and site["adjusted_shell_null_z"] > 3, site
    assert site["split_half_r"] > 0.99 and site["r_template"] > 0.99, site
    # The raw pull ranks templates by brain coverage under a brain-wide push; the adjusted one is flat.
    assert off_brain["r_brain"] < -0.99 and off_brain["null_z"] < 0, off_brain
    assert abs(off_brain["adjusted_pull"]) < 1e-3 * abs(off_brain["pull"]), off_brain
    assert noise["reading"] == "inconsistent" and abs(noise["split_half_r"]) < 0.2, noise
    assert other["reading"] == "no site reward" and other["adjusted_pull"] < 0, other
    # A push onto the site's sphere beats the in-brain null but not the shell null, which shares its radius.
    assert sphere["pull"] > sphere["null_p90"] and abs(sphere["pull"] / sphere["shell_null_median"] - 1) < 0.02, sphere
    heads = report["heads"]
    assert heads[0]["agreement"] > 0.99 and abs(heads[2]["agreement"] * math.sqrt(count) - 1) < 0.4, heads
    assert np.isclose(report["noise_agreement"], 1 / math.sqrt(count))
    assert site["realisable_r_template"] > 0.99 and off_brain["realisable_r_brain"] < -0.99, (site, off_brain)
    assert report["position"] == {"max_abs": 0.04, "via_keys_max_abs": 0.0, "via_values_max_abs": 0.04}
    assert np.allclose(arrays["mean_descent"], descent.mean(0))
    assert arrays["half_descent"].shape == (2, 2, n_heads, *grid)

    # The query split: head 0's views nearly oppose (cosine -8.75 / 9.25); the noise head's views share its noise.
    views = report["query_views"]
    assert report["query_views_parity_max"] < 1e-12 and len(views) == n_heads, report["query_views_parity_max"]
    assert (
        abs(views[0]["cosine"] + 8.75 / 9.25) < 1e-3
        and min(views[0]["t1_agreement"], views[0]["flair_agreement"]) > 0.99
    )
    assert views[2]["cosine"] > 0.999 and abs(views[2]["flair_agreement"] - heads[2]["agreement"]) < 1e-9, views[2]

    # Finite moves, scored: a planted sweep whose change is exactly the first-order prediction, plus a
    # curvature that lowers the loss only for the site's offset on head 2 (whose descent map is noise).
    profiles, groups = offset_profiles(arrays["template"], arrays["nulls"])
    assert len(profiles) == 1 + sum(null["count"] for null in report["nulls"].values())
    assert np.allclose(profiles.max(axis=(1, 2, 3)), 1) and np.allclose(profiles[0], template / template.max())
    scales = [1.0, 4.0]
    pulls = np.einsum("kvmdhw,tdhw->ktm", descent, profiles)
    bent = np.zeros((count, len(profiles), 1 + n_heads, len(scales)))
    bent[:, 0, 3] = -0.01 * np.square(scales)
    planted = 2.0 - np.concatenate([pulls.sum(-1, keepdims=True), pulls], axis=-1)[..., None] * np.array(scales) + bent
    moves, move_arrays = score_moves(
        {"baseline": np.full(count, 2.0), "losses": planted}, descent, profiles, groups, scales, np.full(count, 2.0)
    )
    rows = {(row["heads"], row["scale"]): row for row in moves["rows"]}
    assert len(rows) == (1 + n_heads) * len(scales) and moves["baseline_vs_step_max_abs"] == 0
    assert np.allclose(move_arrays["move_change"] - move_arrays["move_first_order"], bent, rtol=0, atol=1e-12)
    # Head 0's descent map is the template minus its mean, in both views.
    expected = -2 * ((template - template.mean()) * profiles[0]).sum()
    assert np.isclose(rows[(0, 4.0)]["first_order"], 4.0 * expected, rtol=1e-3), (rows[(0, 4.0)], expected)
    assert rows[(0, 1.0)]["reading"] == "rewards site" and rows[(0, 1.0)]["null_z"] > 3, rows[(0, 1.0)]
    for heads in ("all", 0, 1, 3, 4):
        assert all(abs(rows[(heads, s)]["beyond_first_order"]) < 1e-12 for s in scales)
    for scale in scales:
        row = rows[(2, scale)]
        assert np.isclose(row["beyond_first_order"], -0.01 * scale**2) and row["reading"] == "rewards site", row
        assert row["beyond_null_rank"] == 1 and row["beyond_shell_null_rank"] == 1, row
    assert 0 < moves["overlap_from_uniform"][0] < moves["overlap_from_uniform"][1] < 1, moves["overlap_from_uniform"]

    assert parse_args(["--offset-scales", "2,0.5"]).offset_scales == [2.0, 0.5]
    for bad in ("0", "1,-2", "x", "inf"):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                parse_args(["--offset-scales", bad])
            except SystemExit:
                continue
        raise AssertionError(f"--offset-scales {bad} was accepted")
    print("self-test: planted descent maps, query split and finite moves read back (numpy)", flush=True)

    if importlib.util.find_spec("torch") is None:
        print("self-test: torch is not installed; the gradient, capture and CLI checks were skipped", flush=True)
        return
    _self_test_torch()
    print(
        "self-test OK (dL/dl matches autograd, the query gradient rebuilds from the capture, dL/dW splits into "
        "logits and values and is zero through the logits at q = 0, a planted site reward reads back, the CLI "
        "draws the trainer's batches, the per-view query parts sum to autograd's and each is autograd's with the "
        "other view detached, small offsets change the loss by their first-order prediction, finite offsets find "
        "a site the first-order pull reads as zero, a learned mask's draws replay)",
        flush=True,
    )


def _self_test_torch():
    import tempfile

    import torch
    from torch.utils.data import DataLoader

    import training.encoder_args as encoder_args
    import training.main_conv_synthetic as trainer
    from eval.protocol.score_checkpoint import build_model

    # dL/dl rebuilt from the weights' gradient is the logits' own gradient.
    torch.manual_seed(0)
    logits = torch.randn(3, 4, 10, dtype=torch.float64, requires_grad=True)
    weights = logits.softmax(-1)
    weights.retain_grad()
    (weights.square() * torch.randn(3, 4, 10, dtype=torch.float64)).sum().backward()
    assert torch.allclose(logit_gradient(weights), logits.grad)

    # A small full-width run; its settings.json round-trips through the trainer's parser.
    argv = (
        "--res 32 --hidden-channels 16 --res-channels 8 --nb-res-layers 1 --downscale-factor 4 --latent-dim 12 "
        "--content-channels 9 --n-content 9 --n-style 3 --conv-readout mlp --global-pool attention "
        "--attention-pool-width full --synthetic-normalize fixed_reference --synthetic-clean-content "
        "--synthetic-temporal-sigma 0.09 --num-train-samples 8 --num-val-samples 4 --batch-size 4 --no-cache "
        "--cross-view-negs-only --tau 0.1 --train-steps 10 --seed 3 --model-id selftest"
    ).split()
    cfg = json.loads(json.dumps(vars(encoder_args.parse_args(argv))))
    options, dropped = training_options(cfg)
    assert json.loads(json.dumps(vars(options))) == cfg and dropped == []
    assert training_options(dict(cfg, retired_option=1))[1] == ["retired_option"]
    assert training_options(cfg, batch_size=2)[0].batch_size == 2

    # The real training loss at q = 0, over three batches: two per epoch, so the order crosses an epoch.
    model = build_model(cfg, "cpu").train()
    pool = model.attention_pool
    dataset = trainer.make_dataset(options, "train", options.num_train_samples)
    collected = collect(
        model, training_batches(training_loader(options, dataset, "cpu"), 3), training_loss(model, options, "cpu")
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(options.loader_seed),
    )
    expected = [batch["index"].tolist() for _ in range(2) for batch in loader][:3]
    assert collected["indices"] == expected, (collected["indices"], expected)
    assert collected["logit_grad"].shape == (3, 2, 4, 8, 8, 8) and collected["keys"].shape == (2, 512, 16)
    assert np.max(collected["parity"]) < 1e-4, collected["parity"]
    scale = np.abs(collected["logit_grad"]).max()
    assert np.abs(collected["logit_grad"].sum(axis=(3, 4, 5))).max() < 1e-5 * scale * 512
    assert np.all(collected["position_grad_keys"] == 0) and np.abs(collected["position_grad"]).max() > 0
    assert 0.1 < collected["voxel_brain"].mean() < 0.9 and collected["cell_brain"].shape == (2, 8, 8, 8)
    # The two views' parts of the query gradient sum to autograd's.
    parts_sum = collected["query_grad_views"].sum(1) - collected["query_grad"]
    assert np.abs(parts_sum).max() < 1e-4 * np.abs(collected["query_grad"]).max(), np.abs(parts_sum).max()

    # Off q = 0, dL/dW through the logits (rebuilt here) plus dL/dW with the weights detached is the total.
    with torch.no_grad():
        pool.query.normal_(0.0, 0.5)
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    loss = training_loss(model, options, "cpu")
    with captured_attention(pool) as calls:
        out = batch_gradients(model, batch, loss, calls)
    attention = pool.attention
    pool.attention = lambda x: attention(x).detach()
    model.zero_grad(set_to_none=True)
    loss(batch).backward()
    del pool.attention
    values_only = pool.position.grad.double().numpy()
    assert np.abs(out["position_grad_keys"]).max() > 0 and out["parity"] < 1e-4, out["parity"]
    residual = out["position_grad"] - out["position_grad_keys"].sum(0) - values_only
    assert np.abs(residual).max() < 1e-5 * np.abs(out["position_grad"]).max(), np.abs(residual).max()

    def drawn(seed, count=2):
        """``count`` batches of 4 subjects per view, shuffled by ``seed``: the same ones on every call."""
        return training_batches(
            DataLoader(dataset, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(seed)), count
        )

    # In float64, with a random query and position weights: each view's part of the query gradient is
    # autograd's with the other view's weights detached, and a small offset changes the real training
    # loss by its first-order prediction, -s sum_views D . o, for every template and head choice.
    double = build_model(cfg, "cpu").double().train()
    with torch.no_grad():
        double.attention_pool.query.normal_(0.0, 0.5)
        double.attention_pool.position.normal_(0.0, 0.1)
    step = training_loss(double, options, "cpu")

    def step64(batch):
        return step(dict(batch, image=[image.double() for image in batch["image"]]))

    exact = collect(double, drawn(1), step64)
    assert (
        np.abs(exact["query_grad_views"].sum(1) - exact["query_grad"]).max() < 1e-12 * np.abs(exact["query_grad"]).max()
    )
    attention = double.attention_pool.attention
    for v in range(2):

        def one_view(x, v=v):
            weights = attention(x)
            b = len(weights) // 2
            return torch.cat([weights[:b], weights[b:].detach()] if v == 0 else [weights[:b].detach(), weights[b:]])

        double.attention_pool.attention = one_view
        double.zero_grad(set_to_none=True)
        step64(next(drawn(1))).backward()
        del double.attention_pool.attention
        part = exact["query_grad_views"][0, v]
        assert np.allclose(double.attention_pool.query.grad.numpy(), part, rtol=1e-9, atol=1e-12 * np.abs(part).max())
    exact_report, exact_arrays = analyse(exact, 32, 0.09)
    profiles, groups = offset_profiles(exact_arrays["template"], exact_arrays["nulls"])
    small = offset_losses(double, drawn(1), step64, profiles, [1e-4])
    moved, moved_arrays = score_moves(small, exact_arrays["descent"], profiles, groups, [1e-4], exact["loss"])
    assert moved["baseline_vs_step_max_abs"] < 1e-12 and small["indices"] == exact["indices"], moved
    change, predicted = moved_arrays["move_change"], moved_arrays["move_first_order"]
    assert np.abs(predicted).max() > 0 and np.abs(change - predicted).max() < 1e-3 * np.abs(predicted).max()

    # A planted loss that rewards head 0's weight on the site, and nothing else: at q = 0 its descent map
    # is the template minus its mean, so the analysis must read the site, exactly, and no push elsewhere.
    with torch.no_grad():
        pool.query.zero_()
    target = torch.zeros(4, 512)
    target[0] = torch.from_numpy(grid_template(32, 0.09, (8, 8, 8)).reshape(-1)).float()

    def planted(batch):
        h = model._encode(torch.cat(batch["image"], dim=0), 2, None)
        return -(pool.attention(pool.tokens(h)) * target).sum(-1).mean()

    seeded = DataLoader(dataset, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(0))
    planted_report, _ = analyse(collect(model, training_batches(seeded, 2), planted), 32, 0.09)
    first, *rest = (row for row in planted_report["rows"] if row["view"] == "t1")
    assert first["reading"] == "rewards site" and first["r_template"] > 0.9999 and first["null_rank"] == 1, first
    assert first["realisable_r_template"] is not None and planted_report["heads"][0]["agreement"] > 0.5
    assert all(row["reading"] == "inconsistent" and row["split_half_r"] is None for row in rest), rest
    assert all(row["agreement"] is None for row in planted_report["heads"][1:])

    # A planted loss flat to first order at uniform attention that falls when head 0 concentrates on the
    # site: minus the square of head 0's weight on the site minus its uniform value. Its pull is zero, so
    # the first-order reading cannot see the site; the finite moves must.
    site = target[0]

    def flat(batch):
        h = model._encode(torch.cat(batch["image"], dim=0), 2, None)
        on_site = (pool.attention(pool.tokens(h))[:, 0] * site).sum(-1)
        return -(on_site - site.mean()).square().mean()

    flat_gradients = collect(model, drawn(0), flat)
    _, flat_arrays = analyse(flat_gradients, 32, 0.09)
    profiles, groups = offset_profiles(flat_arrays["template"], flat_arrays["nulls"])
    flat_moves, flat_arrays = score_moves(
        offset_losses(model, drawn(0), flat, profiles, [1.0, 4.0]),
        flat_arrays["descent"],
        profiles,
        groups,
        [1.0, 4.0],
        flat_gradients["loss"],
    )
    rows = {(row["heads"], row["scale"]): row for row in flat_moves["rows"]}
    for heads in ("all", 0):
        for scale in (1.0, 4.0):
            row = rows[(heads, scale)]
            assert row["reading"] == "rewards site" and row["null_z"] > 3 and row["shell_null_z"] > 3, row
            assert abs(row["first_order"]) < 1e-3 * abs(row["change"]) and row["beyond_null_rank"] == 1, row
    assert np.all(flat_arrays["move_change"][:, :, 2:] == 0), "heads 1-3 alone moved a loss that only reads head 0"

    # A learned content mask draws a sample every step; each batch's evaluations replay its step's draws,
    # and the sweep leaves the generator where the steps did.
    learned_cfg = json.loads(json.dumps(vars(encoder_args.parse_args(argv + ["--mask-mode", "learned"]))))
    learned = build_model(learned_cfg, "cpu").train()
    learned_step = training_loss(learned, training_options(learned_cfg)[0], "cpu")
    batch = next(drawn(0))
    assert float(learned_step(batch)) != float(learned_step(batch)), "the learned mask drew the same sample twice"
    torch.manual_seed(5)
    steps = collect(learned, drawn(0), learned_step)
    after_steps = torch.rand(3)
    torch.manual_seed(5)
    replayed = offset_losses(learned, drawn(0), learned_step, profiles[:2], [1.0, 1.0])
    assert np.array_equal(replayed["baseline"], steps["loss"]) and torch.equal(torch.rand(3), after_steps)
    assert np.array_equal(replayed["losses"][..., 0], replayed["losses"][..., 1])

    # The CLI on a saved step-0 run reads the same batches and gradients as above.
    with tempfile.TemporaryDirectory() as tmp:
        run = os.path.join(tmp, "run")
        os.makedirs(run)
        with open(os.path.join(run, "settings.json"), "w") as fp:
            json.dump(cfg, fp)
        torch.save(build_model(cfg, "cpu").state_dict(), os.path.join(run, "model.pt"))
        main(["--run-dir", run, "--batches", "3", "--device", "cpu"])
        with open(os.path.join(run, "attention_pool_gradients.json")) as fp:
            saved = json.load(fp)
        assert saved["batch_indices"] == expected and saved["batch_size"] == 4 and saved["split"] == "train"
        assert saved["batches"] == 3 and np.isclose(saved["noise_agreement"], 1 / math.sqrt(3))
        assert len(saved["rows"]) == 8 and len(saved["heads"]) == 4 and saved["parity_max"] < 1e-4
        assert saved["position"]["via_keys_max_abs"] == 0 and saved["position"]["via_values_max_abs"] > 0
        assert saved["pool"]["width"] == "full" and saved["settings"] == cfg
        maps = np.load(os.path.join(run, "attention_pool_gradients_maps.npz"))
        assert maps["descent"].shape == (3, 2, 4, 8, 8, 8)
        assert np.allclose(maps["descent"], -collected["logit_grad"], rtol=1e-5, atol=1e-12)
        assert len(saved["query_views"]) == 4 and saved["query_views_parity_max"] < 1e-4
        assert np.allclose(maps["query_grad_views"], collected["query_grad_views"], rtol=1e-5, atol=1e-12)
        moves = saved["finite_move"]
        templates = 1 + sum(null["count"] for null in saved["nulls"].values())
        assert moves["scales"] == [0.5, 1.0, 2.0, 4.0, 8.0] and moves["head_choices"] == ["all", 0, 1, 2, 3]
        assert len(moves["rows"]) == 25 and moves["batches"] == 3 and moves["baseline_vs_step_max_abs"] < 1e-6
        assert maps["move_change"].shape == maps["move_first_order"].shape == (3, templates, 5, 5)

        main(
            ["--run-dir", run, "--batches", "2", "--batch-size", "2", "--device", "cpu", "--out", f"{tmp}/b2.json"]
            + ["--offset-scales", "1,4"]
        )
        with open(f"{tmp}/b2.json") as fp:
            b2 = json.load(fp)
            assert b2["batch_size"] == 2 and b2["finite_move"]["scales"] == [1.0, 4.0], b2["finite_move"]["scales"]

        for change, message in (({"global_pool": "gap"}, "GAP"), ({"n_content": 6}, "z_content[6]")):
            with open(os.path.join(run, "settings.json"), "w") as fp:
                json.dump(dict(cfg, **change), fp)
            try:
                main(["--run-dir", run])
            except SystemExit as exc:
                assert message in str(exc), exc
            else:
                raise AssertionError(f"{change} was not refused")


if __name__ == "__main__":
    main()
