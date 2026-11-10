"""Train one model (local | global | pooled) on on-the-fly synthetic pairs, then evaluate it.

    PYTHONPATH=. python -m experiments.gl_ident.train --config experiments/gl_ident/configs/e1.yaml \
        --set model.kind=local train.seed=0

Writes results/runs/<name>_<kind>_s<seed>/: config.yaml, train.csv, val.csv, best.pt, last.pt, metrics.json,
tb/ (TensorBoard). Seeds: the generator weights and the val/test sets are fixed for every run; train.seed sets
the model init, the training latent stream, the noise stream and the site/window draws. The training latent
stream depends only on (train.seed, step), so it is common to every condition.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from experiments.gl_ident.config import load_config, save_config
from experiments.gl_ident.evaluate import evaluate, headline, test_data
from experiments.gl_ident.generator import SyntheticGenerator, cpu_generator, derive_seed
from experiments.gl_ident.losses import compute_loss
from experiments.gl_ident.models import TwoView, local_receptive_field

RESULTS = Path(__file__).parent / "results"


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def val_loss(model, cfg, x1, x2):
    """Mean loss over full batches of the fixed val set, with fixed site / window draws."""
    model.eval()
    B = cfg.train.batch_size
    total, count = 0.0, 0
    for b, i in enumerate(range(0, x1.shape[0] - B + 1, B)):
        loss, _ = compute_loss(model, cfg, x1[i : i + B], x2[i : i + B], cpu_generator("val-sample", b))
        total, count = total + float(loss), count + 1
    model.train()
    return total / count


def build_model(cfg, device):
    torch.manual_seed(derive_seed("init", cfg.train.seed))
    return TwoView(cfg).to(device)


def train(cfg, out_dir: Path, eval_untrained=False):
    device = resolve_device(cfg.train.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    tc, seed = cfg.train, cfg.train.seed
    gen = SyntheticGenerator(cfg.generator, device=device)
    model = build_model(cfg, device)
    info = dict(device=device, params=sum(p.numel() for p in model.parameters()))
    if cfg.model.kind == "local":
        info["receptive_field_sites"] = local_receptive_field(model.enc[0], cfg.generator.n_sites, cfg.generator.dim)
        info["receptive_field_expected"] = cfg.model.n_g
    test = test_data(gen)
    if eval_untrained:
        untrained = evaluate(model, cfg, gen, data=test, device=device)
        (out_dir / "metrics_untrained.json").write_text(json.dumps(untrained, indent=1))

    val_lat = gen.split("val")
    vx1, vx2 = gen.split_images("val", val_lat)
    opt = torch.optim.Adam(model.parameters(), lr=tc.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc.steps)
    writer = SummaryWriter(out_dir / "tb")
    writer.add_text("config", "```\n" + (out_dir / "config.yaml").read_text() + "\n```")
    writer.add_text("info", json.dumps(info))
    train_log = csv.writer(open(out_dir / "train.csv", "w", newline="", buffering=1))
    train_log.writerow(["step", "loss", "top1", "pos_sim", "lr"])
    val_log = csv.writer(open(out_dir / "val.csv", "w", newline="", buffering=1))
    val_log.writerow(["step", "val_loss", "best"])
    best, best_step, t0 = math.inf, 0, time.time()
    model.train()
    for step in range(1, tc.steps + 1):
        lat = gen.sample_latents(tc.batch_size, cpu_generator("train-latents", seed, step))
        x1, x2 = gen.observe(lat, cpu_generator("train-noise", seed, step))
        loss, diag = compute_loss(model, cfg, x1, x2, cpu_generator("train-sample", seed, step))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % tc.log_every == 0 or step == 1:
            lr = sched.get_last_lr()[0]
            train_log.writerow([step, loss.item(), diag.get("top1"), diag.get("pos"), lr])
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", lr, step)
            for key, val in diag.items():
                writer.add_scalar(f"train/{key}", val, step)
        if step % tc.eval_every == 0 or step == tc.steps:
            vl = val_loss(model, cfg, vx1, vx2)
            improved = vl < best
            if improved:
                best, best_step = vl, step
                torch.save(dict(model=model.state_dict(), step=step, val_loss=vl), out_dir / "best.pt")
            val_log.writerow([step, vl, int(improved)])
            writer.add_scalar("val/loss", vl, step)
            print(
                f"[{out_dir.name}] step {step} loss {loss.item():.4f} val {vl:.4f} best {best:.4f}@{best_step} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    torch.save(dict(model=model.state_dict(), step=tc.steps), out_dir / "last.pt")
    info.update(best_step=best_step, best_val_loss=best, train_seconds=time.time() - t0)

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["model"])
    t1 = time.time()
    metrics = evaluate(model, cfg, gen, data=test, device=device)
    info["eval_seconds"] = time.time() - t1
    metrics["info"] = info
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    for key, val in headline(metrics, cfg.model.kind).items():
        writer.add_scalar(f"test/{key}", val, best_step)
    writer.close()
    return metrics


def parse_set(items):
    out = {}
    for item in items or []:
        key, val = item.split("=", 1)
        try:
            out[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            out[key] = val
    return out


def run_name(cfg):
    return f"{cfg.name}_{cfg.model.kind}_s{cfg.train.seed}"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument(
        "--set", nargs="+", action="extend", default=[], help="dotted overrides, e.g. model.kind=global train.seed=1"
    )
    p.add_argument("--out-root", default=str(RESULTS / "runs"))
    p.add_argument("--eval-untrained", action="store_true", help="also score the step-0 model (floor)")
    args = p.parse_args(argv)
    cfg = load_config(*args.config, overrides=parse_set(args.set))
    out_dir = Path(args.out_root) / run_name(cfg)
    metrics = train(cfg, out_dir, eval_untrained=args.eval_untrained)
    print(json.dumps({**headline(metrics, cfg.model.kind), **metrics["info"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
