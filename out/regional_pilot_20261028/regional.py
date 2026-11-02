"""Pilot: contrast each region of the image only with the same region of the other view, instead of the whole image.

The encoder trainer's patch InfoNCE on a coarse grid is this scheme already. The backbone map is averaged over each of
g^3 regions, the shared MLP head turns each region's average into a dense code, and InfoNCE runs separately per
region, with the other subjects' same region as the negatives (training.losses.patch_infonce_loss). So no trainer
change is needed to test it.

Same bank, recipe, init and eval as out/xpred_pilot_20261028 (its USER_FLAGS, minus that run's patch Barlow Twins
term), so its base arm (global InfoNCE + patch BT at 16^3) is a reference row. Arms:
  gap          global InfoNCE on the GAP code only (the paper's scheme)
  region2      per-region InfoNCE on a 2^3 grid (8 regions), no global term
  region3      per-region InfoNCE on a 3^3 grid (27 regions), no global term
  gap_region2  both, weight 1 each
  region4, region8, region16
               per-region InfoNCE on finer grids, no global term, all-background regions dropped
               (--patch-foreground-mask: about half the 4^3 regions and 60-70% of the 8^3/16^3 ones hold no brain in
               any batch subject; region3 keeps its 4-8 empty regions)
  region16dot  region16 with --patch-similarity dot, which keeps each cell's size, so cells alike in every subject drop
               out instead of being renormalised to unit length
  gap_region16 global InfoNCE + region16, weight 1 each (does a global term keep the global factors while 16^3 regions
               still find the lesion?)
  gap_region16_double
               gap_region16 with --patch-center-mode double: the region term sees each subject's region codes minus
               that subject's own mean over the kept regions, so only what differs between locations can win it,
               and the global term carries what is common to them
  gap01_region16_double, gap03_region16_double
               gap_region16_double with the global term at weight 0.1 or 0.3 (region term 1): at weight 1 the global
               term cost most of region16's lesion gain, and without it the global code lost its factors
  region16_double
               gap_region16_double without the global term: its matched control, which region16 (position-centred)
               only approximates
  lin_gap_region16_double, lin_region16_double
               gap_region16_double with a 1x1-conv readout (--conv-readout linear) instead of the MLP head, global term
               at weight 1 or 0 (its control). The global code is then exactly the mean of the 16^3 cell codes, so the
               global term's gradient is the same at every cell of a subject and the double-centred region term's sums
               to zero over its kept cells: the two are orthogonal all the way into the backbone map, where under the
               MLP head (global code = head of the mean map) they were not. In the MLP arms the region loss rose with
               the global weight (0.29 at 0, 0.80 at 1); does it still, and does the lesion still fall?
  gap_region16_double_sepadam
               gap_region16_double with each term on its own AdamW (own moments, own clipping; weight decay applied
               once per step, as before). gradcheck.py found that the global term slows the region term by setting
               Adam's per-weight step sizes (its gradients are ~19x the region term's at init), not by pulling
               against it (cosine +0.1 to +0.4). Here neither term's scale sets the other's steps, and the global
               weight stops mattering, since Adam ignores a term's scale. Does the region loss now fall at the
               regions-only pace, and does the lesion follow?
  wide64_sepadam
               gap_region16_double_sepadam with the head's output widened from 18 to 64 units. The global term still
               reads the first 14, the region term now reads all 64, so a cell's code is as wide as the backbone map
               under it. cellcode.py found the 14-unit cell codes agree across views on the lesion but keep little of
               it, though one direction of the map holds all the cell probe reads. Is that because 14 units are too
               few, or because the loss does not value a feature this rare? Read the transfer rows: a wide code can
               hold the lesion in one view without any training, so only agreement across views counts.
  region16_oracle
               region16 with an oracle filter on its negatives: where an anchor cell and another subject's same cell
               both hold a lesion, that subject leaves the anchor's softmax. Lesions cluster, so at B=32 about two
               thirds of lesion cells have such a negative, and the loss pays to tell the two lesions apart, which
               works against a plain "lesion here" feature. Does removing every one of them help the lesion? It reads
               the generator's lesion maps (train_lesion_cells.npy, from oracle_check.py), so it is a diagnostic only.
  predcode_sepadam
               gap_region16_double_sepadam with a third term on a third AdamW: each view's 14 cell units go through a
               1x1x1 conv that predicts the OTHER view's z-scored cell image (the xpred pilot's target, mask and MSE, with
               the cell code as input in place of the 64-channel map). That pilot's term put the lesion in the map and in
               its own one-number output but not in the 14 units, which is where the region term holds agreement across
               views. Here one output has to do both.
Region codes are centred per region over the batch and scaled per unit (--standardize-patch-code), the region analogue
of the global --standardize-code; cosine similarity (unless dot) and tau as the global term.

Eval (val, 400 subjects): 5-fold ridge R^2 of each content factor, and of the view's own gain (style leak), from each
view's global 14-unit code and from the 14 content units of every region's code concatenated at 2^3, 3^3, 4^3 and 8^3
(112, 378, 896 and 7168 features). Every arm starts from the same init, so step 0 is each readout's untrained floor.
Arms started before the 4^3/8^3 readouts existed are re-scored from their checkpoints with the eval command, into
ARM_post.jsonl.

    PYTHONPATH=SHIM:. python out/regional_pilot_20261028/regional.py BANK_DIR OUT_DIR ARM STEPS EVAL_EVERY
    PYTHONPATH=SHIM:. python out/regional_pilot_20261028/regional.py eval BANK_DIR OUT_DIR ARM STEP
"""

import copy
import importlib.util
import json
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

import training.encoder_args as encoder_args
import training.main_conv_synthetic as T
from eval.metrics.dci import content_factor_names

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "xpred_pilot", os.path.join(HERE, "..", "xpred_pilot_20261028", "pilot.py")
)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

# The xpred pilot's patch Barlow Twins options and how many values each takes; dropped from every arm.
PATCH_FLAGS = {
    "--patch-foreground-mask": 0,
    "--patch-foreground-thresh": 1,
    "--patch-loss-type": 1,
    "--patch-loss-weight": 1,
    "--train-patch-grid": 3,
}
REGION = ["--patch-loss-type", "infonce", "--standardize-patch-code", "--patch-loss-weight", "1.0"]
FOREGROUND = ["--patch-foreground-mask"]
LINEAR = ["--conv-readout", "linear"]
# arm -> (extra flags, weight of the global term)
ARMS = {
    "gap": ([], 1.0),
    "region2": (REGION + ["--train-patch-grid", "2", "2", "2"], 0.0),
    "region3": (REGION + ["--train-patch-grid", "3", "3", "3"], 0.0),
    "gap_region2": (REGION + ["--train-patch-grid", "2", "2", "2"], 1.0),
    "region4": (REGION + FOREGROUND + ["--train-patch-grid", "4", "4", "4"], 0.0),
    "region8": (REGION + FOREGROUND + ["--train-patch-grid", "8", "8", "8"], 0.0),
    "region16": (REGION + FOREGROUND + ["--train-patch-grid", "16", "16", "16"], 0.0),
    "region16dot": (REGION + FOREGROUND + ["--patch-similarity", "dot", "--train-patch-grid", "16", "16", "16"], 0.0),
    "gap_region16": (REGION + FOREGROUND + ["--train-patch-grid", "16", "16", "16"], 1.0),
    "gap_region16_double": (
        REGION + FOREGROUND + ["--patch-center-mode", "double", "--train-patch-grid", "16", "16", "16"],
        1.0,
    ),
    "gap01_region16_double": (
        REGION + FOREGROUND + ["--patch-center-mode", "double", "--train-patch-grid", "16", "16", "16"],
        0.1,
    ),
    "gap03_region16_double": (
        REGION + FOREGROUND + ["--patch-center-mode", "double", "--train-patch-grid", "16", "16", "16"],
        0.3,
    ),
    "region16_double": (
        REGION + FOREGROUND + ["--patch-center-mode", "double", "--train-patch-grid", "16", "16", "16"],
        0.0,
    ),
    "lin_gap_region16_double": (
        REGION + FOREGROUND + LINEAR + ["--patch-center-mode", "double", "--train-patch-grid", "16", "16", "16"],
        1.0,
    ),
    "lin_region16_double": (
        REGION + FOREGROUND + LINEAR + ["--patch-center-mode", "double", "--train-patch-grid", "16", "16", "16"],
        0.0,
    ),
    "gap_region16_double_sepadam": (
        REGION + FOREGROUND + ["--patch-center-mode", "double", "--train-patch-grid", "16", "16", "16"],
        1.0,
    ),
    "wide64_sepadam": (
        REGION
        + FOREGROUND
        + ["--patch-center-mode", "double", "--train-patch-grid", "16", "16", "16", "--latent-dim", "64"],
        1.0,
    ),
    "region16_oracle": (REGION + FOREGROUND + ["--train-patch-grid", "16", "16", "16"], 0.0),
    "predcode_sepadam": (
        REGION + FOREGROUND + ["--patch-center-mode", "double", "--train-patch-grid", "16", "16", "16"],
        1.0,
    ),
}
# Arms whose region and global terms each get their own AdamW (see update) instead of one AdamW on the weighted sum.
SEPARATE_ADAM = ("gap_region16_double_sepadam", "wide64_sepadam", "predcode_sepadam")
# Arms whose region term reads every head output (see objective); the global term and the evals read the content units.
WIDE_REGION = ("wide64_sepadam",)
# Arms whose region term leaves same-cell lesions out of each other's negatives (see filtered_region_loss).
ORACLE_FILTER = ("region16_oracle",)
# Arms with a third term and AdamW: the content units of every cell predict the other view (see code_prediction).
PREDICT_CODE = ("predcode_sepadam",)
DROPPED = -1e4  # a filtered candidate's logit: its exp() is 0 in float32, and masked_fill leaves it no gradient
READOUT_GRIDS = (2, 3, 4, 8)
COLUMNS = (
    "brain_size",
    "ventricle_size",
    "lesion_burden",
    "cortical_thickness",
    "temporal_atrophy",
    "lr_asymmetry",
    "sulcal_widening",
    "gain (own style)",
)


def arm_args(arm):
    flags, i = [], 0
    while i < len(P.USER_FLAGS):
        n_values = PATCH_FLAGS.get(P.USER_FLAGS[i])
        if n_values is None:
            flags.append(P.USER_FLAGS[i])
            i += 1
        else:
            i += 1 + n_values
    args = encoder_args.parse_args(flags + ARMS[arm][0])
    args.cache = False
    return args


def region_codes(model, h, g):
    """(N, latent_dim, g^3): the patch path of MultiviewConvEncoder.global_and_patch_features at grid g^3."""
    if model.readout_type == "mlp":
        return model.patch_readout(model._patch_pool(h, (g,) * 3).transpose(1, 2)).transpose(1, 2)
    return model._patch_pool(model.to_encoding(h), (g,) * 3)


def features(model, bank, device, batch=32):
    """Per view: the global code's content units, and every region's content units concatenated per grid."""
    model.eval()
    units = model.content_units()
    out = {}
    with torch.no_grad():
        for start in range(0, len(bank), batch):
            idx = list(range(start, min(start + batch, len(bank))))
            x = bank.batch(idx, device)[0]
            h = model._encode(x, 2, None)
            reads = {"code": model._global_code(h, x, 2)[:, units]}
            for g in READOUT_GRIDS:
                reads[f"r{g}"] = region_codes(model, h, g)[:, units].flatten(1)
            b = len(idx)
            for name, value in reads.items():
                out.setdefault(f"{name}_t1", []).append(value[:b].cpu())
                out.setdefault(f"{name}_flair", []).append(value[b:].cpu())
    model.train()
    return {key: torch.cat(value).double().numpy() for key, value in out.items()}


def evaluate(model, bank, device):
    names = content_factor_names(9, "burden")
    scores = {}
    for key, X in features(model, bank, device).items():
        view = 0 if key.endswith("t1") else 1
        row = {name: P.ridge_r2(X, bank.zc[:, j]) for j, name in enumerate(names) if not name.startswith("unused")}
        row["gain (own style)"] = P.ridge_r2(X, bank.zs[:, view, 0])
        scores[key] = row
    return scores


def show(step, scores, loss_terms):
    print(f"\nstep {step}  " + "  ".join(f"{k} {v:.4f}" for k, v in loss_terms.items()), flush=True)
    print(f"{'':>10}" + " ".join(f"{c[:10]:>10}" for c in COLUMNS), flush=True)
    for key, row in scores.items():
        print(f"{key:>10}" + " ".join(f"{row[c]:>10.3f}" for c in COLUMNS), flush=True)


def filtered_region_loss(patches, args, crit, masks, lesion):
    """T.patch_contrastive_loss as region16 runs it (standardised cosine InfoNCE, cross-view negatives only), with a
    negative left out of the anchor's softmax where both subjects hold a lesion in that cell. lesion: (B, g^3) bool, on
    the CPU: the same mask built on MPS marked positives too (see oracle_check.py)."""
    keep = T.foreground_positions(masks, args.train_patch_grid, args.patch_foreground_thresh).to(patches.device)
    a, b = patches[..., keep][:, : args.content_channels].chunk(2, dim=0)
    hz = F.normalize(T.standardize_patch_code(torch.stack((a, b))), dim=2, eps=1e-6)
    scores = torch.einsum("bcp,dcp->pbd", hz[0], hz[1]) / args.tau  # (positions, subjects, subjects)
    scores = scores.masked_fill(same_cell_negatives(lesion, keep.cpu()).to(scores.device), DROPPED)
    P, B = scores.shape[:2]
    targets = torch.arange(B, device=scores.device).repeat(P)
    return crit(scores.reshape(P * B, B), targets) + crit(scores.transpose(1, 2).reshape(P * B, B), targets)


def same_cell_negatives(lesion, keep):
    """(positions, B, B) bool: subject j is a negative of anchor i and both hold a lesion at that kept position."""
    on = lesion.flatten(1)[:, keep].T
    return on.unsqueeze(1) & on.unsqueeze(2) & ~torch.eye(lesion.shape[0], dtype=torch.bool)


def code_prediction(patches, args, predictors, targets, keep):
    """P.cross_prediction with each view's content units at every cell as the predictors' input, in place of the map."""
    cells = patches[:, : args.content_channels].reshape(len(patches), args.content_channels, *args.train_patch_grid)
    return P.cross_prediction(types.SimpleNamespace(_last_map=cells), predictors, targets, keep)


def objective(arm, model, x, args, sim, crit, masks, lesion=None, prediction=None):
    """T.training_objective's terms; in a WIDE_REGION arm the region term reads all args.latent_dim head outputs, in an
    ORACLE_FILTER arm it uses filtered_region_loss, and a PREDICT_CODE arm adds code_prediction on prediction =
    (predictors, targets, keep)."""
    if arm not in WIDE_REGION + ORACLE_FILTER + PREDICT_CODE:
        return T.training_objective(model, x, args, sim, crit, masks)[2]
    pooled, patches = model.global_and_patch_features(x, args.train_patch_grid, n_views=2)
    if arm in PREDICT_CODE:
        patch = T.patch_contrastive_loss(patches, model, args, sim, crit, masks)
    elif arm in ORACLE_FILTER:
        patch = filtered_region_loss(patches, args, crit, masks, lesion)
    else:
        region_args = copy.copy(args)
        region_args.content_channels = args.latent_dim
        patch = T.patch_contrastive_loss(patches, model, region_args, sim, crit, masks)
    terms = {
        "global": T.contrastive_loss(pooled, model, args, sim, crit),
        "patch": patch,
        "patch_weighted": args.patch_loss_weight * patch,
    }
    if arm in PREDICT_CODE:
        terms["predict"] = code_prediction(patches, args, *prediction)
    return terms


def optimizers_for(arm, model, args, predictors=None):
    """One AdamW, or for SEPARATE_ADAM arms one per term (region first; a PREDICT_CODE arm's third also moves the
    predictors). Weight decay sits on the first only, so it is applied once per step either way."""
    optimizers = (torch.optim.AdamW(model.parameters(), lr=args.lr),)
    if arm in SEPARATE_ADAM:
        optimizers += (torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0),)
    if predictors is not None:
        params = [*model.parameters(), *predictors.parameters()]
        optimizers += (torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0),)
    return optimizers


def update(model, terms, global_weight, optimizers, grad_clip, predictors=None):
    """One step. With one AdamW: the weighted sum of both terms, clipped as a whole. With more: the region and the
    global term, and a PREDICT_CODE arm's prediction term, each clipped and stepped by its own AdamW, all gradients
    taken at the same weights before any optimizer moves them, so no term's gradient scale enters another's step sizes.
    """
    if len(optimizers) == 1:
        optimizers[0].zero_grad(set_to_none=True)
        (global_weight * terms["global"] + terms["patch_weighted"]).backward()
        T.clip_encoder_gradients(model, grad_clip)
        optimizers[0].step()
        return
    head = [] if predictors is None else list(predictors.parameters())
    params = list(model.parameters()) + head
    losses = [terms["patch_weighted"], global_weight * terms["global"]] + (
        [] if predictors is None else [terms["predict"]]
    )
    grads = [
        torch.autograd.grad(loss, params, retain_graph=i + 1 < len(losses), allow_unused=True)
        for i, loss in enumerate(losses)
    ]
    for optimizer, term_grads in zip(optimizers, grads):
        for p, g in zip(params, term_grads):
            p.grad = torch.zeros_like(p) if g is None else g
        T.clip_encoder_gradients(model, grad_clip)
        if head:
            torch.nn.utils.clip_grad_norm_(head, grad_clip)
        optimizer.step()


def train(bank_dir, out_dir, arm, steps, eval_every):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    args = arm_args(arm)
    global_weight = ARMS[arm][1]
    print(
        f"arm {arm}: global weight {global_weight:g}, region weight {args.patch_loss_weight:g}"
        + (f" on a {args.train_patch_grid} grid" if args.patch_loss_weight > 0 else ""),
        flush=True,
    )
    train_bank, val_bank = P.Bank(bank_dir, "train"), P.Bank(bank_dir, "val")
    lesion = torch.from_numpy(np.load(f"{HERE}/train_lesion_cells.npy") > 0) if arm in ORACLE_FILTER else None
    torch.manual_seed(args.model_seed)
    model = T.build_encoder(args).to(device).train()
    predictors = None
    if arm in PREDICT_CODE:
        predictors = (
            torch.nn.ModuleList(torch.nn.Conv3d(args.content_channels, 1, 1) for _ in range(2)).to(device).train()
        )
    optimizers = optimizers_for(arm, model, args, predictors)
    sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
    order = np.random.default_rng(args.loader_seed)
    log = open(f"{out_dir}/{arm}.jsonl", "a")
    scores = evaluate(model, val_bank, device)
    show(0, scores, {})
    log.write(json.dumps({"step": 0, "scores": scores}) + "\n")
    log.flush()
    queue, running, started = [], {}, time.time()
    for step in range(1, steps + 1):
        if len(queue) < args.batch_size:
            queue += order.permutation(len(train_bank)).tolist()
        idx, queue = queue[: args.batch_size], queue[args.batch_size :]
        x, masks, targets, keep = train_bank.batch(idx, device)
        cells = None if lesion is None else lesion[idx]
        terms = objective(arm, model, x, args, sim, crit, masks, cells, (predictors, targets, keep))
        parts = {"global": terms["global"].item(), "region": terms["patch"].item()}
        if predictors is not None:
            parts["predict"] = terms["predict"].item()
        update(model, terms, global_weight, optimizers, args.grad_clip, predictors)
        for key, value in parts.items():
            running[key] = running.get(key, 0.0) + value
        if step % 50 == 0:
            print(
                f"  step {step} ({(time.time() - started) / step:.2f} s/step) "
                + " ".join(f"{k} {v / 50:.4f}" for k, v in running.items()),
                flush=True,
            )
            running = {}
        if step % eval_every == 0 or step == steps:
            scores = evaluate(model, val_bank, device)
            show(step, scores, parts)
            log.write(json.dumps({"step": step, "scores": scores, "loss": parts}) + "\n")
            log.flush()
            state = {"model": model.state_dict(), "optimizer": optimizers[0].state_dict()}
            if len(optimizers) > 1:
                state["optimizer_global"] = optimizers[1].state_dict()
            if predictors is not None:
                state["predictors"] = predictors.state_dict()
                state["optimizer_predict"] = optimizers[2].state_dict()
            torch.save(state, f"{out_dir}/{arm}_s{step}.pt")


def rescore(bank_dir, out_dir, arm, step):
    """Score a saved checkpoint with the current readouts, into ARM_post.jsonl."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    args = arm_args(arm)
    model = T.build_encoder(args)
    model.load_state_dict(torch.load(f"{out_dir}/{arm}_s{step}.pt", map_location="cpu")["model"])
    model.to(device).train()
    scores = evaluate(model, P.Bank(bank_dir, "val"), device)
    show(step, scores, {})
    with open(f"{out_dir}/{arm}_post.jsonl", "a") as log:
        log.write(json.dumps({"step": step, "scores": scores}) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        rescore(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    else:
        train(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
