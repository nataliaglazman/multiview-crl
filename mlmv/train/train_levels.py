"""Train the multi-level encoder (Phases 1 and 2; Phase 3a later behind ``phase3a.enabled``).

    python -m mlmv.train.train_levels --config mlmv/config/runs/phase1_exact.yaml [--set optim.steps=200 ...]

Per step: B subjects; both modalities through the encoder; the template is updated and subtracted; the sampler
draws S positions, L half-splits and G region sets; the three level losses are weighted and summed.

Outputs under ``run.out_dir / run.name``: ``config.yaml`` (resolved, with the commit), ``geometry/``
(``rho.json``, ``region_masks.npz``, ``eligibility.csv``), ``tb/`` (TensorBoard, if installed) and
``scalars.jsonl`` (every logged scalar), ``checkpoints/``, and for synthetic runs ``eval/`` (leakage matrix).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mlmv.config import load_config, save_config
from mlmv.data.geometry import build_geometry
from mlmv.data.sampler import ViewPairSampler
from mlmv.eval.consistency import g_cross_split_cka, g_region_sums
from mlmv.eval.leakage import collect_codes, evaluate_leakage, format_matrix
from mlmv.eval.metrics import effective_rank
from mlmv.eval.site_probe import site_accuracy_per_block
from mlmv.losses.levels import LEVELS, LevelObjective
from mlmv.models.encoder import MODALITIES, MultiLevelModel, Template, build_encoder
from mlmv.models.receptive_field import interior_positions, measure_rho, write_rho


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_source(cfg, stride: int, device):
    if cfg.run.source == "synthetic":
        from mlmv.synth.three_level import SyntheticSource

        return SyntheticSource(cfg.synth, stride, device)
    raise NotImplementedError(f"run.source={cfg.run.source!r}: the ADNI loader is Phase 0, not built yet")


class Logger:
    """TensorBoard when available, and always a JSON-lines copy (``scalars.jsonl``)."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / "scalars.jsonl"
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(run_dir / "tb"))
        except Exception:  # tensorboard missing: keep the JSON log only
            self.tb = None

    def scalars(self, step: int, values: dict) -> None:
        values = {k: float(v) for k, v in values.items() if v is not None and np.isfinite(float(v))}
        with open(self.path, "a") as fp:
            fp.write(json.dumps(dict(step=step, **values)) + "\n")
        if self.tb is not None:
            for k, v in values.items():
                self.tb.add_scalar(k, v, step)

    def close(self):
        if self.tb is not None:
            self.tb.close()


def peak_memory_mb(device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KiB on Linux


def lr_lambda(warmup: int, total: int):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return f


class Fifo:
    """Last ``size`` rows of each level's view-a block samples, for the effective-rank log."""

    def __init__(self, size: int):
        self.size, self.rows = size, {}

    def push(self, level, x):
        x = x.detach().float().cpu()
        prev = self.rows.get(level)
        self.rows[level] = (x if prev is None else torch.cat([prev, x]))[-self.size :]

    def get(self, level):
        return self.rows.get(level)


def level_weights(cfg, losses, encoder):
    """Fixed weights, or (``balance: gradnorm``) weights that equalise each level's gradient norm on the last
    trunk block: ``w_l = lambda_l * mean_j(g_j) / g_l`` (detached)."""
    base = {lvl: float(cfg.loss.weights[lvl]) for lvl in LEVELS}
    if cfg.loss.balance == "none":
        return base, {}
    if cfg.loss.balance != "gradnorm":
        raise ValueError(f"loss.balance must be none or gradnorm, got {cfg.loss.balance!r}")
    params = [p for p in encoder.blocks[-1].parameters() if p.requires_grad]
    norms = {}
    for lvl in LEVELS:
        grads = torch.autograd.grad(losses[lvl], params, retain_graph=True, allow_unused=True)
        norms[lvl] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads if g is not None)))
    mean = np.mean(list(norms.values()))
    return {lvl: base[lvl] * mean / max(norms[lvl], 1e-12) for lvl in LEVELS}, norms


def build_run_geometry(cfg, source, rho, log):
    return build_geometry(
        source.feature_labels,
        rho,
        cfg.geometry.metric,
        cfg.geometry.min_positions_per_half,
        cfg.geometry.eligibility_normals,
        seed=cfg.seed,
        log=log,
    )


def monitor(model, source, geometry, cfg, device, P):
    """Validation probes: site balanced accuracy per block, and G cross-split CKA."""
    n = min(source.sizes["val"], cfg.eval.max_subjects)
    codes, truth, _ = collect_codes(model, source, "val", range(n), positions_per_subject=1, seed=cfg.seed)
    acc = site_accuracy_per_block(codes, truth["site"], cfg.eval.cv_folds)
    sums, counts = g_region_sums(model, source, "val", range(n), geometry)
    cka = g_cross_split_cka(sums, counts, geometry, cfg.sampler.G.volume_ratio, n_splits=5, seed=cfg.seed)
    out = {}
    for m in MODALITIES:
        for lvl in LEVELS:
            out[f"{P}/{lvl}/site_bal_acc_{m}"] = acc[m][lvl]
        out[f"{P}/G/cross_split_cka_{m}"] = cka[m]["median"]
    return out


def save_checkpoint(path, step, model, objective, optim, sched, gen, rho_used):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        dict(
            step=step,
            model=model.state_dict(),
            objective=objective.state_dict(),
            optim=optim.state_dict(),
            sched=sched.state_dict(),
            gen=gen.get_state(),
            rho_used=rho_used,
        ),
        tmp,
    )
    tmp.replace(path)


def run(cfg, log=print) -> Path:
    seed_everything(cfg.seed)
    device = pick_device(cfg.run.device)
    P = "p1" if cfg.run.source == "synthetic" else "p2"  # scalar prefix: phase / level / name
    run_dir = Path(cfg.run.out_dir) / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    log(f"run dir {run_dir}  device {device}")

    encoder = build_encoder(cfg.model)
    theoretical = encoder.theoretical_radius()
    log(
        f"theoretical receptive-field radius {theoretical} feature voxels (footprint {encoder.theoretical_footprint()})"
    )
    if theoretical > cfg.model.max_rf_radius:
        raise SystemExit(
            f"refusing to train: receptive-field radius {theoretical} > model.max_rf_radius {cfg.model.max_rf_radius}"
        )
    rho = theoretical if cfg.geometry.rho == "auto" else float(cfg.geometry.rho)
    effective = None

    source = make_source(cfg, encoder.stride, device)
    geometry = build_run_geometry(cfg, source, rho, log)
    geometry.save(run_dir / "geometry")
    write_rho(run_dir / "geometry" / "rho.json", theoretical, effective, rho)
    log(f"geometry: rho {rho}, metric {geometry.metric}, eligible regions {geometry.eligible}")
    sampler = ViewPairSampler(geometry, cfg.sampler, mask=source.feature_mask)

    dims = encoder.dims
    template = Template(sum(dims.values()), geometry.shape, cfg.model.template_momentum, cfg.model.template_subtraction)
    model = MultiLevelModel(encoder, template).to(device)
    objective = LevelObjective(cfg, dims, sampler.membership(), rho, geometry.metric).to(device)
    params = [p for p in list(model.parameters()) + list(objective.parameters()) if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda(cfg.optim.warmup_steps, cfg.optim.steps))
    gen = torch.Generator().manual_seed(cfg.seed)
    logger = Logger(run_dir)
    fifo = Fifo(cfg.loss.buffer.size)
    amp = device.type == "cuda" and cfg.run.amp == "bf16"

    start = 0
    last = run_dir / "checkpoints" / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if state["rho_used"] != rho:
            rho = state["rho_used"]
            geometry = build_run_geometry(cfg, source, rho, log)
            sampler = ViewPairSampler(geometry, cfg.sampler, mask=source.feature_mask)
            objective.set_membership(sampler.membership())
            objective.rho = rho
        model.load_state_dict(state["model"])
        objective.load_state_dict(state["objective"])
        optim.load_state_dict(state["optim"])
        sched.load_state_dict(state["sched"])
        gen.set_state(state["gen"])
        start = state["step"]
        log(f"resumed from step {start}")

    t0 = time.time()
    batch_size = cfg.sampler.batch_subjects
    for step in range(start, cfg.optim.steps):
        model.train()
        objective.train()
        idx = torch.randperm(source.sizes["train"], generator=gen)[:batch_size]
        data = source.batch("train", idx.tolist())
        sample = sampler.sample(batch_size, gen)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            feats = {
                m: {lvl: f.flatten(2) for lvl, f in model.blocks(data[m].to(device), m, update=True).items()}
                for m in MODALITIES
            }
            losses, info, pairs = objective(feats, sample, data["subject"])
        weights, norms = level_weights(cfg, losses, encoder)
        total = sum(weights[lvl] * losses[lvl] for lvl in LEVELS)
        optim.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
        optim.step()
        sched.step()
        for lvl in LEVELS:
            fifo.push(lvl, pairs[lvl][0])

        if step % cfg.run.log_every == 0 or step == cfg.optim.steps - 1:
            scalars = {
                f"{P}/total/loss": float(total.detach()),
                f"{P}/total/grad_norm": float(grad_norm),
                f"{P}/total/lr": sched.get_last_lr()[0],
                f"{P}/total/peak_memory_mb": peak_memory_mb(device),
                f"{P}/total/sec_per_step": (time.time() - t0) / max(1, step - start + 1),
            }
            for lvl in LEVELS:
                scalars[f"{P}/{lvl}/loss"] = float(losses[lvl].detach())
                scalars[f"{P}/{lvl}/weight"] = weights[lvl]
                for k, v in info[lvl].items():
                    scalars[f"{P}/{lvl}/{k}"] = v
                if lvl in norms:
                    scalars[f"{P}/{lvl}/trunk_grad_norm"] = norms[lvl]
                rows = fifo.get(lvl)
                scalars[f"{P}/{lvl}/effective_rank"] = effective_rank(rows)
                scalars[f"{P}/{lvl}/effective_rank_frac"] = effective_rank(rows) / dims[lvl]
            logger.scalars(step, scalars)
            log(
                f"step {step}: total {scalars[P + '/total/loss']:.4f} "
                + " ".join(f"{lvl} {scalars[P + '/' + lvl + '/loss']:.3f}" for lvl in LEVELS)
                + f"  peak {scalars[P + '/total/peak_memory_mb']:.0f} MB"
            )

        if step + 1 == cfg.optim.rho_measure_step:
            positions = interior_positions(
                geometry.shape, math.ceil(theoretical), cfg.optim.rho_positions, gen, mask=source.feature_mask
            )
            probe = {m: data[m][:2].to(device) for m in MODALITIES}
            measured = measure_rho(encoder, probe, positions, cfg.optim.rho_mass, gen)
            effective = measured["effective"]
            logger.scalars(step, {f"{P}/total/rho_effective": effective})
            if cfg.geometry.rho == "auto" and effective != rho:
                log(f"effective rho {effective} differs from {rho}: recomputing masks and eligibility")
                rho = effective
                geometry = build_run_geometry(cfg, source, rho, log)
                geometry.save(run_dir / "geometry")
                sampler = ViewPairSampler(geometry, cfg.sampler, mask=source.feature_mask)
                objective.set_membership(sampler.membership())
                objective.rho = rho
            write_rho(
                run_dir / "geometry" / "rho.json", theoretical, effective, rho, dict(measured=measured, step=step + 1)
            )

        if (step + 1) % cfg.run.eval_every == 0:
            logger.scalars(step, monitor(model, source, geometry, cfg, device, P))
        if (step + 1) % cfg.run.ckpt_every == 0:
            save_checkpoint(last, step + 1, model, objective, optim, sched, gen, rho)

    save_checkpoint(last, cfg.optim.steps, model, objective, optim, sched, gen, rho)
    torch.save(dict(model=model.state_dict(), rho_used=rho, config=cfg.to_dict()), run_dir / "checkpoints" / "final.pt")
    if cfg.run.source == "synthetic":
        result = evaluate_leakage(model, source, cfg, run_dir / "eval", split="test", device=str(device))
        for m in MODALITIES:
            log(f"leakage matrix ({m}, best R^2):\n{format_matrix(result['rows'], m)}")
        failed = [c for c in result["checks"] if not c["passed"]]
        log(f"pass checks: {len(result['checks']) - len(failed)}/{len(result['checks'])} passed")
    logger.close()
    return run_dir


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", nargs="*", default=[], help="run YAML(s), applied in order over defaults.yaml")
    p.add_argument("--set", nargs="*", default=[], help="overrides, e.g. optim.steps=200 run.name=test")
    args = p.parse_args(argv)
    cfg = load_config(*args.config, overrides=args.set)
    run(cfg)


if __name__ == "__main__":
    main()
