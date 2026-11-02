"""Pilot: does a cross-view prediction term make the encoder pick up ONE burden lesion at T1 lesion value 0.4?

Starts from the user's run patch8_sulcal_ventricle_strat_burden_bt, with its CLI (USER_FLAGS) parsed by the trainer's
own parser, model builder and loss functions. Differences from that run, identical in every arm:
  - device mps, and a pre-rendered bank of the same 2000 train / 400 val subjects;
  - uniform shuffled batches instead of the oracle neighbourhood sampler.
Arms:
  base    the run's objective: global InfoNCE on the standardized 14-unit code + 1.0 x patch Barlow Twins at 16^3
  xpred   base + w x cross-view prediction: each view's backbone map (64 channels at 16^3) goes through a 1x1x1 conv
          that predicts the OTHER view's image averaged over the same 4^3-voxel cells. Targets are z-scored within
          each subject's brain cells, which removes that view's gain and bias. The loss is MSE over cells that are at
          least half brain. No labels: the target is the other image.
Eval (val, 400 subjects): 5-fold ridge R^2 of each content factor from each view's 14 content units (what the
trainer's eval reads) and from its 64-channel backbone GAP.

    python out/xpred_pilot_20261028/pilot.py bank BANK_DIR
    python out/xpred_pilot_20261028/pilot.py time BANK_DIR
    python out/xpred_pilot_20261028/pilot.py train BANK_DIR OUT_DIR ARM STEPS EVAL_EVERY [WEIGHT [INIT START TAG]]
INIT continues from a saved checkpoint (model and predictors; the optimizer restarts unless the checkpoint holds
its state), numbering steps from START; TAG names the log and checkpoints (default ARM).
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import training.encoder_args as encoder_args
import training.main_conv_synthetic as T
from eval.metrics.dci import content_factor_names

USER_FLAGS = """
--attention-pool-frequencies 4 --attention-pool-heads 4 --attention-pool-width slice --batch-size 32 --best-metric none
--bt-lambda 0.005 --no-cache --no-causal-eval --content-channels 14 --contrastive-loss-type infonce
--contrastive-proj-dim 0 --contrastive-proj-hidden 256 --conv-readout mlp --cpu-threads 1 --cross-view-negs-only
--data-seed 42 --dataset-name synthetic --downscale-factor 4 --encoder-architecture conv --encoder-head-hidden 100
--eval-every 1000 --eval-patch-grid 4 5 4 --eval-pooling gap --floor-eval --global-pool gap --grad-clip 2.0
--hidden-channels 64 --image-spacing 2.0 --latent-dim 18 --lesion-keypoints 0 --loader-seed 10042 --log-every 100
--lr 0.0001 --mask-mode fixed --model-seed 42 --n-content 9 --n-style 3 --nb-res-layers 2 --norm-type group
--num-train-samples 2000 --num-val-samples 400 --num-workers 0 --patch-foreground-mask --patch-foreground-thresh 0.05
--patch-loss-type barlow_twins --patch-loss-weight 1.0 --res 64 --res-channels 32 --seed 42
--no-separate-spatial-readout --standardize-code --synthetic-clean-content --synthetic-content-scale 1.0
--synthetic-identifiable-ventricle --synthetic-lesion-count 1 --synthetic-lesion-intensity styled
--synthetic-lesion-placement wm_interior --synthetic-lesion-radius 0.1 --synthetic-lesion-t1-value 0.4
--synthetic-lesion-target burden --synthetic-mode pseudo_mri --synthetic-n-deformation-grid 4
--synthetic-n-fissure-grid 8 --synthetic-normalize fixed_reference --synthetic-style-scale 1.0
--synthetic-sulcal-mode atrophy --synthetic-temporal-sigma 0.18 --tau 0.1 --train-patch-grid 16 16 16
--train-steps 20000
""".split()
SPLITS = {"train": 2000, "val": 400}
WORKERS = 8
CELL = 4
KEEP = 0.5


def user_args():
    args = encoder_args.parse_args(USER_FLAGS)
    args.cache = False
    return args


def render_chunk(job):
    """Render subjects [start, stop) of one split straight into the bank's memmaps."""
    bank, mode, start, stop = job
    torch.set_num_threads(1)
    args = user_args()
    ds = T.make_dataset(args, mode, SPLITS[mode])
    t1 = np.load(f"{bank}/{mode}_t1.npy", mmap_mode="r+")
    flair = np.load(f"{bank}/{mode}_flair.npy", mmap_mode="r+")
    mask = np.load(f"{bank}/{mode}_mask.npy", mmap_mode="r+")
    zc = np.load(f"{bank}/{mode}_zc.npy", mmap_mode="r+")
    zs = np.load(f"{bank}/{mode}_zs.npy", mmap_mode="r+")
    for i in range(start, stop):
        item = ds[i]
        t1[i] = item["image"][0][0].numpy().astype(np.float16)
        flair[i] = item["image"][1][0].numpy().astype(np.float16)
        mask[i] = item["mask"][0][0].numpy() > 0
        lat = item["gt_latents"]
        zc[i] = np.asarray(lat["z_content"], dtype=np.float32)[:9]
        zs[i] = np.stack(
            [np.asarray(lat["z_style_v1"], dtype=np.float32)[:3], np.asarray(lat["z_style_v2"], dtype=np.float32)[:3]]
        )
    for array in (t1, flair, mask, zc, zs):
        array.flush()
    return mode, start, stop


def make_bank(bank):
    os.makedirs(bank, exist_ok=True)
    res = user_args().res
    jobs = []
    for mode, n in SPLITS.items():
        np.lib.format.open_memmap(f"{bank}/{mode}_t1.npy", "w+", np.float16, (n, res, res, res))
        np.lib.format.open_memmap(f"{bank}/{mode}_flair.npy", "w+", np.float16, (n, res, res, res))
        np.lib.format.open_memmap(f"{bank}/{mode}_mask.npy", "w+", np.bool_, (n, res, res, res))
        np.lib.format.open_memmap(f"{bank}/{mode}_zc.npy", "w+", np.float32, (n, 9))
        np.lib.format.open_memmap(f"{bank}/{mode}_zs.npy", "w+", np.float32, (n, 2, 3))
        bounds = np.linspace(0, n, WORKERS + 1).astype(int)
        jobs += [(bank, mode, int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    started = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for mode, start, stop in pool.map(render_chunk, jobs):
            print(f"  {mode} {start}-{stop} rendered ({time.time() - started:.0f} s)", flush=True)


class Bank:
    """One split in memory: float16 images, brain masks, 16^3 cell targets and the cells they keep."""

    def __init__(self, bank, mode):
        self.t1 = torch.from_numpy(np.load(f"{bank}/{mode}_t1.npy"))
        self.flair = torch.from_numpy(np.load(f"{bank}/{mode}_flair.npy"))
        self.mask = torch.from_numpy(np.load(f"{bank}/{mode}_mask.npy"))
        self.zc = np.load(f"{bank}/{mode}_zc.npy")
        self.zs = np.load(f"{bank}/{mode}_zs.npy")
        frac = F.avg_pool3d(self.mask[:, None].float(), CELL)
        self.keep = frac >= KEEP
        self.targets = [
            standardize(F.avg_pool3d(view[:, None].float(), CELL), self.keep) for view in (self.t1, self.flair)
        ]

    def __len__(self):
        return len(self.zc)

    def batch(self, idx, device):
        idx = torch.as_tensor(idx)
        x = torch.cat([self.t1[idx], self.flair[idx]])[:, None].float().to(device)
        masks = torch.cat([self.mask[idx], self.mask[idx]])[:, None].to(device)
        targets = torch.cat([self.targets[1][idx], self.targets[0][idx]]).to(device)  # T1 map -> FLAIR, FLAIR map -> T1
        keep = torch.cat([self.keep[idx], self.keep[idx]]).to(device)
        return x, masks, targets, keep


def standardize(cells, keep):
    """z-score each subject's cell values over its kept cells (removes a view's gain and bias)."""
    n = keep.flatten(1).sum(1).clamp_min(1).view(-1, 1, 1, 1, 1)
    mean = (cells * keep).flatten(1).sum(1).view(-1, 1, 1, 1, 1) / n
    var = ((cells - mean).square() * keep).flatten(1).sum(1).view(-1, 1, 1, 1, 1) / n
    return (cells - mean) / (var.sqrt() + 1e-6) * keep


def build(args, arm, device):
    torch.manual_seed(args.model_seed)
    model = T.build_encoder(args)
    encode = model._encode

    def encode_and_keep(x, n_views, view_idx):
        h = encode(x, n_views, view_idx)
        model._last_map = h
        return h

    model._encode = encode_and_keep
    predictors = None
    if arm == "xpred":
        predictors = torch.nn.ModuleList([torch.nn.Conv3d(args.hidden_channels, 1, 1) for _ in range(2)])
    model.to(device).train()
    if predictors is not None:
        predictors.to(device).train()
    return model, predictors


def cross_prediction(model, predictors, targets, keep):
    """MSE of each view's map predicting the other view's z-scored cell image, over kept cells."""
    h = model._last_map
    b = h.shape[0] // 2
    pred = torch.cat([predictors[0](h[:b]), predictors[1](h[b:])])
    err = (pred - targets).square() * keep
    return err.flatten(1).sum(1).div(keep.flatten(1).sum(1).clamp_min(1)).mean()


def features(model, bank, device, batch=32):
    """Per view: the 14 content units of the global code and the 64-channel backbone GAP, for every subject."""
    model.eval()
    out = {"code_t1": [], "code_flair": [], "gap_t1": [], "gap_flair": []}
    with torch.no_grad():
        for start in range(0, len(bank), batch):
            idx = list(range(start, min(start + batch, len(bank))))
            x, _, _, _ = bank.batch(idx, device)
            h = model._encode(x, 2, None)
            code = model._global_code(h, x, 2)
            b = len(idx)
            units = model.content_units()
            out["code_t1"].append(code[:b, units].cpu())
            out["code_flair"].append(code[b:, units].cpu())
            out["gap_t1"].append(h[:b].mean((2, 3, 4)).cpu())
            out["gap_flair"].append(h[b:].mean((2, 3, 4)).cpu())
    model.train()
    return {key: torch.cat(value).double().numpy() for key, value in out.items()}


def ridge_r2(X, y):
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 5, 17)))
    pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=0))
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def evaluate(model, bank, device):
    names = content_factor_names(9, "burden")
    feats = features(model, bank, device)
    scores = {}
    for key, X in feats.items():
        view = 0 if key.endswith("t1") else 1
        scores[key] = {
            name: ridge_r2(X, bank.zc[:, j]) for j, name in enumerate(names) if not name.startswith("unused")
        }
        scores[key]["gain (own style)"] = ridge_r2(X, bank.zs[:, view, 0])
    return scores


def show(step, scores, loss_terms):
    cols = (
        "lesion_burden",
        "brain_size",
        "ventricle_size",
        "cortical_thickness",
        "temporal_atrophy",
        "gain (own style)",
    )
    head = " ".join(f"{c[:12]:>12}" for c in cols)
    print(f"\nstep {step}  " + "  ".join(f"{k} {v:.4f}" for k, v in loss_terms.items()), flush=True)
    print(f"{'':>12}{head}", flush=True)
    for key, row in scores.items():
        print(f"{key:>12}" + " ".join(f"{row[c]:>12.3f}" for c in cols), flush=True)


def train(bank_dir, out_dir, arm, steps, eval_every, weight=1.0, init=None, start_step=0, tag=None):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args = user_args()
    train_bank, val_bank = Bank(bank_dir, "train"), Bank(bank_dir, "val")
    model, predictors = build(args, arm, device)
    params = list(model.parameters()) + (list(predictors.parameters()) if predictors is not None else [])
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    tag = tag or arm
    if init is not None:
        saved = torch.load(init, map_location="cpu")
        model.load_state_dict(saved["model"])
        if predictors is not None:
            predictors.load_state_dict(saved["predictors"])
        if saved.get("optimizer") is not None:
            optimizer.load_state_dict(saved["optimizer"])
        print(
            f"continuing from {init} at step {start_step}, weight {weight:g}, "
            f"optimizer {'restored' if saved.get('optimizer') is not None else 'restarted'}",
            flush=True,
        )
    sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
    order = np.random.default_rng(args.loader_seed + start_step)
    log = open(f"{out_dir}/{tag}.jsonl", "a")
    scores = evaluate(model, val_bank, device)
    show(start_step, scores, {})
    log.write(json.dumps({"step": start_step, "scores": scores}) + "\n")
    log.flush()
    queue, running, started = [], {}, time.time()
    for step in range(1, steps + 1):
        if len(queue) < args.batch_size:
            queue += order.permutation(len(train_bank)).tolist()
        idx, queue = queue[: args.batch_size], queue[args.batch_size :]
        x, masks, targets, keep = train_bank.batch(idx, device)
        _, loss, terms = T.training_objective(model, x, args, sim, crit, masks)
        parts = {"global": terms["global"].item(), "patch": terms["patch"].item()}
        if predictors is not None:
            xpred = cross_prediction(model, predictors, targets, keep)
            loss = loss + weight * xpred
            parts["xpred"] = xpred.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        T.clip_encoder_gradients(model, args.grad_clip)
        if predictors is not None:
            torch.nn.utils.clip_grad_norm_(predictors.parameters(), args.grad_clip)
        optimizer.step()
        for key, value in parts.items():
            running[key] = running.get(key, 0.0) + value
        if step % 50 == 0:
            print(
                f"  step {start_step + step} ({(time.time() - started) / step:.2f} s/step) "
                + " ".join(f"{k} {v / 50:.4f}" for k, v in running.items()),
                flush=True,
            )
            running = {}
        if step % eval_every == 0 or step == steps:
            scores = evaluate(model, val_bank, device)
            show(start_step + step, scores, parts)
            log.write(json.dumps({"step": start_step + step, "scores": scores, "loss": parts}) + "\n")
            log.flush()
            state = {
                "model": model.state_dict(),
                "predictors": predictors.state_dict() if predictors is not None else None,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(state, f"{out_dir}/{tag}_s{start_step + step}.pt")
    torch.save(state, f"{out_dir}/{tag}.pt")


def time_steps(bank_dir):
    """Seconds per step for each arm and the backbone-gradient norm of each term at init (weight 1)."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args = user_args()
    bank = Bank(bank_dir, "val")
    sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
    model, predictors = build(args, "xpred", device)
    x, masks, targets, keep = bank.batch(list(range(32)), device)
    _, _, terms = T.training_objective(model, x, args, sim, crit, masks)
    xpred = cross_prediction(model, predictors, targets, keep)
    backbone = [p for p in model.encoder.parameters()] + [p for p in model.encoder_v1.parameters()]
    for name, term in (("global", terms["global"]), ("patch", terms["patch"]), ("xpred", xpred)):
        grads = torch.autograd.grad(term, backbone, retain_graph=True, allow_unused=True)
        norm = torch.sqrt(sum((g.square().sum() for g in grads if g is not None), torch.tensor(0.0, device=device)))
        print(f"  init {name}: value {term.item():.4f}, backbone grad norm {norm.item():.4e}", flush=True)
    for arm in ("base", "xpred"):
        model, predictors = build(args, arm, device)
        params = list(model.parameters()) + (list(predictors.parameters()) if predictors is not None else [])
        optimizer = torch.optim.AdamW(params, lr=args.lr)
        for step in range(6):
            if step == 2:
                if device.type == "mps":
                    torch.mps.synchronize()
                started = time.time()
            _, loss, _ = T.training_objective(model, x, args, sim, crit, masks)
            if predictors is not None:
                loss = loss + cross_prediction(model, predictors, targets, keep)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        print(f"  {arm}: {(time.time() - started) / 4:.2f} s/step", flush=True)


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "bank":
        make_bank(sys.argv[2])
    elif command == "time":
        time_steps(sys.argv[2])
    else:
        weight = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
        init, start, tag = (sys.argv[8], int(sys.argv[9]), sys.argv[10]) if len(sys.argv) > 10 else (None, 0, None)
        train(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), weight, init, start, tag)
