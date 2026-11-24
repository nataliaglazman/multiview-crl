"""Training on immutable competition banks; labels are used only by the supervised control."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.multiview_encoder import MultiviewConvEncoder
from training.feature_competition_data import VIEWS, array_hash, file_hash, images, load_bank, targets, write_json
from training.losses import infonce_base_loss
from utils.encoder_runtime import select_encoder_device


def build_ssl(settings, separate_encoders=True):
    return MultiviewConvEncoder(
        hidden_channels=settings["hidden_channels"],
        res_channels=settings["res_channels"],
        nb_res_layers=2,
        downscale_factor=4,
        latent_dim=12,
        content_channels=9,
        separate_encoders=separate_encoders,
        conv_readout="mlp",
        encoder_head_hidden=100,
    )


class GapRegressor(nn.Module):
    """One view through the SSL encoder's architecture (backbone, GAP, MLP head, 9 content units), then a
    linear map to the lesion centre, trained with labels: can that global code carry location at all?

    Seeded like the SSL model, it starts from the same backbone and head weights (encoder_v1 is a copy).
    """

    def __init__(self, settings, mean=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
        super().__init__()
        self.encoder = build_ssl(settings, separate_encoders=False)
        self.head = nn.Linear(9, 3)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def code(self, x):
        return self.encoder(x, pool_only=True)[2][0][:, :9]

    def forward(self, x):
        return self.head(self.code(x)) * self.scale + self.mean

    def loss(self, x, centre):
        return F.mse_loss(self.head(self.code(x)), (centre - self.mean) / self.scale)


class ImageLocalizer(nn.Module):
    def __init__(self, res):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, 1),
        )
        axis = torch.arange(res // 2, dtype=torch.float32) * 2 + 0.5
        self.register_buffer(
            "coordinates", torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
        )

    def logits(self, x):
        return self.network(x).flatten(1)

    def forward(self, x):
        return self.logits(x).softmax(1) @ self.coordinates

    def loss(self, x, centre):
        # A supervised target heatmap, expressed in input voxel coordinates.
        d2 = ((self.coordinates[None] - centre[:, None]) ** 2).sum(-1)
        target = (-d2 / (2 * 1.5**2)).softmax(1)
        return -(target * self.logits(x).log_softmax(1)).sum(1).mean()


def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run_path(root, kind, arm, seed, view="paired", variant=""):
    # A loss variant gets its own folder, e.g. ssl-std_A_s42_paired for the batch-standardized code.
    return Path(root) / "runs" / f"{kind}{'-' + variant if variant else ''}_{arm}_s{seed}_{view}"


def batch_indices(count, batch_size, seed, step):
    return np.random.default_rng([seed, step]).choice(count, batch_size, replace=False)


def build_model(settings):
    if settings["kind"] == "ssl":
        return build_ssl(settings)
    if settings["kind"] == "supervised_gap":
        return GapRegressor(settings, settings["target_mean"], settings["target_scale"])
    return ImageLocalizer(settings["res"])


def ssl_code(model, paired_images):
    b = len(paired_images)
    x = torch.cat([paired_images[:, 0], paired_images[:, 1]])
    return model(x, pool_only=True, n_views=2)[2][0].reshape(2, b, -1)


def standardize_batch(code):
    """Centre and scale every unit over the batch, separately for each view.

    In arm A every subject has the same anatomy, so raw codes are nearly identical (cosine ~0.996 between
    subjects) and the cosine softmax is almost uniform. Standardized, the small differences reach the loss.
    """
    return (code - code.mean(1, keepdim=True)) / (code.std(1, keepdim=True) + 1e-6)


def infonce(code, standardize=False):
    return infonce_base_loss(
        standardize_batch(code) if standardize else code,
        list(range(9)),
        nn.CosineSimilarity(dim=-1),
        nn.CrossEntropyLoss(),
        tau=0.1,
        cross_view_negs_only=True,
    ).sum()


def ssl_loss(model, paired_images, standardize=False):
    return infonce(ssl_code(model, paired_images), standardize)


@torch.no_grad()
def heldout_check(model, kind, bank, labels, batch_size, device, view=None, standardize=False):
    """The training loss on held-out subjects, in fixed consecutive batches of the training size.

    For SSL also the cross-view top-1 retrieval rate. Chance is 2 ln B and 1/B. In arm A the lesion is
    the only thing a subject's two views share, so any drop below chance there is lesion information.
    """
    model.eval()
    losses, top1 = [], []
    for start in range(0, len(bank) - batch_size + 1, batch_size):
        x = torch.from_numpy(np.array(bank[start : start + batch_size], copy=True)).to(device)
        if kind == "ssl":
            code = ssl_code(model, x)
            losses.append(float(infonce(code, standardize)))
            unit = F.normalize((standardize_batch(code) if standardize else code)[..., :9], dim=-1)
            scores = unit[0] @ unit[1].T
            target = torch.arange(batch_size, device=device)
            hits = (scores.argmax(1) == target).float().mean() + (scores.argmax(0) == target).float().mean()
            top1.append(float(hits) / 2)
        else:
            y = torch.from_numpy(labels[start : start + batch_size].copy()).to(device)
            losses.append(float(model.loss(x[:, VIEWS.index(view)], y)))
    model.train()
    if not losses:
        raise ValueError("Need at least one full held-out batch")
    row = dict(heldout_loss=float(np.mean(losses)), heldout_subjects=len(losses) * batch_size)
    if kind == "ssl":
        row.update(
            heldout_top1=float(np.mean(top1)), chance_loss=2 * float(np.log(batch_size)), chance_top1=1 / batch_size
        )
    return row


def torch_load(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def save_checkpoint(path, payload):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = [
        "training/feature_competition.py",
        "training/feature_competition_data.py",
        "training/feature_competition_train.py",
        "eval/synthetic/feature_competition_report.py",
        "eval/synthetic/synthetic_dataset.py",
        "models/multiview_encoder.py",
        "models/vqvae.py",
        "training/losses.py",
    ]
    return {path: file_hash(root / path) for path in files}


def train(args):
    meta = load_bank(args.bank)
    device = select_encoder_device(args.device)
    if args.batch_size > meta["counts"]["train"] or args.batch_size < 2:
        raise ValueError("Need 2 <= batch size <= training subjects")
    if not (Path(args.bank) / "audit/report.json").exists():
        raise ValueError("Run the input audit before training")
    audit_report = json.loads((Path(args.bank) / "audit/report.json").read_text())
    if audit_report["bank_id"] != meta["bank_id"]:
        raise ValueError("Audit belongs to another bank")
    if not audit_report.get("input_gate_passed"):
        raise ValueError("Input visibility/axis-coverage gate did not pass; inspect audit/report.json before training")
    missing = [arm for arm in args.arms if arm not in meta["arms"] or arm not in audit_report.get("arms", [])]
    if missing:
        raise ValueError(f"Arms not in this bank or its audit: {missing}")
    supervised = args.kind != "ssl"
    standardize = args.kind == "ssl" and args.standardize_code
    train_targets = targets(args.bank, "train")[:, :3]
    probe_targets = targets(args.bank, "probe")[:, :3]
    if args.batch_size > meta["counts"]["probe"]:
        raise ValueError("The held-out check needs batch size <= probe subjects")
    for arm in args.arms:
        x_bank = images(args.bank, "train", arm)
        heldout_bank = images(args.bank, "probe", arm)
        y_bank = train_targets if supervised else None
        views = args.views if supervised else ["paired"]
        checkpoints = {0, args.steps, *range(args.save_every, args.steps + 1, args.save_every)}
        checkpoints |= {s for s in args.save_steps if s <= args.steps}
        for seed in args.seeds:
            for view in views:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                folder = run_path(args.bank, args.kind, arm, seed, view, "std" if standardize else "")
                settings = dict(
                    kind=args.kind,
                    arm=arm,
                    seed=seed,
                    view=view,
                    bank_id=meta["bank_id"],
                    hidden_channels=args.hidden_channels,
                    res_channels=args.res_channels,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    res=meta["res"],
                    tau=0.1,
                    grad_clip=2.0,
                    optimizer="AdamW",
                    weight_decay=0.01,
                    device=device,
                    source_hashes=source_hashes(),
                    minibatches="independent without-replacement draw per step; paired across arms",
                )
                if standardize:
                    settings.update(standardize_code=True)
                if args.kind == "supervised_gap":
                    # Standardize the regression target with training-split statistics only.
                    settings.update(
                        target_mean=train_targets.mean(0).tolist(), target_scale=train_targets.std(0).tolist()
                    )
                model = build_model(settings)
                init_hash = state_hash(model)
                model.to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
                start, history, monitor = 0, [], []
                if folder.exists():
                    if not args.resume:
                        raise FileExistsError(f"{folder} exists; use --resume with identical settings")
                    saved = torch_load(folder / "last.pt", device)
                    if saved["settings"] != settings:
                        raise ValueError("Resume settings/source/bank mismatch")
                    model.load_state_dict(saved["model"])
                    optimizer.load_state_dict(saved["optimizer"])
                    start, history, monitor = saved["step"], saved["history"], saved.get("monitor", [])
                    if start > args.steps:
                        raise ValueError("Requested endpoint precedes saved checkpoint")
                else:
                    folder.mkdir(parents=True)
                    write_json(folder / "settings.json", dict(settings, initial_state_sha256=init_hash))

                def save(step):
                    check = heldout_check(
                        model,
                        args.kind,
                        heldout_bank,
                        probe_targets,
                        args.batch_size,
                        device,
                        view=view,
                        standardize=standardize,
                    )
                    monitor[:] = [row for row in monitor if row["step"] != step] + [dict(step=step, **check)]
                    chance = f" (chance {check['chance_loss']:.3f})" if "chance_loss" in check else ""
                    top1 = f" top1={check['heldout_top1']:.3f}" if "heldout_top1" in check else ""
                    print(
                        f"{folder.name} step={step} heldout_loss={check['heldout_loss']:.4f}{chance}{top1}", flush=True
                    )
                    payload = dict(
                        settings=settings,
                        step=step,
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        history=history,
                        monitor=monitor,
                        initial_state_sha256=init_hash,
                    )
                    save_checkpoint(folder / f"step_{step:06d}.pt", payload)
                    save_checkpoint(folder / "last.pt", payload)
                    write_json(
                        folder / "progress.json",
                        dict(step=step, requested_steps=args.steps, device=device, history=history, monitor=monitor),
                    )

                if start == 0:
                    save(0)
                model.train()
                for step in range(start + 1, args.steps + 1):
                    ids = batch_indices(len(x_bank), args.batch_size, seed, step)
                    x = torch.from_numpy(np.array(x_bank[ids], copy=True)).to(device)
                    if args.kind == "ssl":
                        loss = ssl_loss(model, x, standardize)
                    else:
                        y = torch.from_numpy(y_bank[ids].copy()).to(device)
                        loss = model.loss(x[:, VIEWS.index(view)], y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at {folder} step {step}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                    if step % args.log_every == 0 or step == args.steps:
                        row = dict(step=step, loss=float(loss.detach().cpu()), batch_sha256=array_hash(ids))
                        history.append(row)
                        print(f"{folder.name} step={step} loss={row['loss']:.6f}", flush=True)
                    if step in checkpoints:
                        save(step)
                print(f"Saved {folder / 'last.pt'}", flush=True)
