"""Stage 80 (no backbone training; rules written 26 Nov before any of its results; the user asked for it): on a frozen
backbone, does a fresh head trained with the sufficiency term keep all seven own parts where the run's own head lost
one?

Why. Stage 77's sufficiency term (warm.py --sufficiency-weight) took its targets from a momentum copy of the run's own
backbone, so a run could lower it by carrying less shared content as well as by explaining more. Both of its arms read
ventricle and the lesion volume worse than the recipe, from the code and from the backbone's own GAP (training logs,
26 Nov). Here the backbone and the pooling, with its gauge's stored statistics, are frozen: the targets cannot move
and nothing can be forgotten, so the term can only be lowered by the head explaining more. In the failing 8-unit codes
the pooled features still read ventricle's and burden's own parts better than the code (stage 64's own_variation,
item 59), so a head that keeps them exists on these backbones.

Codes: the frozen recipe at 8 units on the fixed bank (code batches every 500, 31 neighbours, residual, jitter 2 / 3
degrees, prediction weight 1, tails): stage 64's five (seeds 42, 2, 3, 4, 5) and stage 76 part B's ten A100 runs
(seeds 6-15). Their own verdicts (stages 75 and 76): 7 fail (42, 4, 7, 8, 9, 14, 15), 8 hold. The scoring repeats
them from the sources' bundles and exports.

Per code, two fresh heads from one initialisation (head seed 0; TailCode's head, 448 -> 100 -> 12 at 8 units, the code
its first 8 outputs), on the frozen pooled features: the unaugmented ones and 8 jittered draws per training subject, as
frozen_prediction.py caches them (the run's own jitter; each view draws its own).
  control  the recipe's objective on the head: InfoNCE (tau .1, cross-view negatives, batch-standardized cosine)
           plus the cross-view prediction term (weight 1, linear maps to the other view's whitened GAP, ridge .1);
  suff     the same plus warm.Sufficiency (weight 1, explainers 8 -> 128 -> 128 -> 64, fitted alongside), its targets
           each view's own whitened GAP, here frozen.
Both: code batches (an anchor and its 31 nearest in both views' current code, rebuilt every 500 steps from the
unaugmented training features), batch 32, AdamW lr 1e-4, gradients clipped at 2 over every trained parameter, 20,000
steps. Scored as stage 76 (causal_bundle.py, pool_check.py detect: holds and rho1).

Rules (fixed before any result):
  R1 repair: suff holds in at least 4 of the 7 failing codes.
  R2 no harm: suff holds in at least 7 of the 8 holding codes.
  R3 the term, not the refit: suff holds in at least 2 more of the 15 codes than control.
  Works if R1, R2 and R3 all pass. Fails if suff holds in 2 or fewer of the 7 failing codes. Otherwise unresolved.
  Described only: control's holds (whether a fresh head alone repairs, i.e. where the lottery lives), rho1 of every
  head, the own parts against the source code's.
Expectation, written before: control repairs 1-3 of the 7 failures and keeps most holders; suff repairs 4-6 and keeps
all holders.

    python head_refit.py train SOURCE_RUN OUT_RUN control|suff [--steps 20000] [--cache CACHE.pt]
    python head_refit.py summary DETECT.json
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import code_audit
import frozen_prediction as FP
import numpy as np
import torch
import warm
from torch import nn

import data.cluster_batches as cb

HERE = Path(__file__).resolve().parent
STAGE = HERE / "causal" / "stage80"
TAU, PREDICT_WEIGHT, SUFFICIENCY_WEIGHT, RIDGE = 0.1, 1.0, 1.0, 0.1
BATCH, NEIGHBOURS, REBUILD, DRAWS, LR, CLIP, EVERY = 32, 31, 500, 8, 1e-4, 2.0, 2000
SOURCE_HOLDS = {42: False, 2: True, 3: True, 4: False, 5: True, 6: True, 7: False, 8: False, 9: False, 10: True,
                11: True, 12: True, 13: True, 14: False, 15: False}


def fresh_head(pool, seed):
    """TailCode's head re-initialised by PyTorch's default rule under ``seed``; nothing else drawn changes."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for layer in pool.head:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()


def head_codes(pool, pooled, units):
    with torch.no_grad():
        return torch.stack([pool.head(p)[:, :units] for p in pooled])


def train(source, out, arm, steps, cache_path, head_seed=0, device=None):
    device = device or warm.pick_device()
    report, model, pool, residual, source_step = code_audit.load_run(source, device)
    a = report["args"]
    if pool is None or source_step != int(a["steps"]) or a.get("supervise", "None") != "None":
        raise SystemExit(f"{source.name}: needs a finished, unsupervised TailCode run")
    units = pool.units
    FP.freeze_features(model, pool)
    frozen = (FP.state_digest(model), FP.state_digest(pool, exclude_head=True))
    test_n = int(a.get("test_subjects", 1000))
    cache = FP.prepare_cache(
        source, a, model, pool, residual,
        SimpleNamespace(cache=cache_path, draws=DRAWS, seed=42, batch=BATCH, test_subjects=test_n, device=device),
    )
    fresh_head(pool, head_seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(head_seed + 1)
        decoders = warm.prediction_decoders(units, cache["train"]["gap"].shape[-1]).to(device)
    trained = list(pool.head.parameters()) + list(decoders.parameters())
    sufficiency = None
    if arm == "suff":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed + 2)
            sufficiency = warm.Sufficiency(units, cache["train"]["gap"].shape[-1], 128).to(device)
        trained += list(sufficiency.parameters())
    optimizer = torch.optim.AdamW(trained, lr=LR)
    stats = FP.fit_whitening(cache["train"]["gap"], RIDGE)
    targets = FP.whiten(cache["train"]["gap"], stats).to(device)
    jittered = cache["train_pooled"].to(device)
    clean = cache["train"]["pooled"].to(device)
    test_inputs = cache["test"]["pooled"].to(device)
    test_bank = warm.Bank(a["bank"], "test", a["t1key"], test_n, residual)
    sim = nn.CosineSimilarity(dim=-1)
    rng = np.random.default_rng(head_seed)
    history, totals, count = [], {}, 0
    sampler, batches = None, None

    def record(step):
        evaluation = FP.score_codes(head_codes(pool, test_inputs, units), cache["test"], test_bank)
        row = {"step": step, "loss": {k: v / max(count, 1) for k, v in totals.items()}, "test": evaluation}
        if sufficiency is not None:
            row["sufficiency_sq_canonical"] = sufficiency.value()
        history.append(row)
        t1 = evaluation["t1"]
        print(f"step {step:5d} code volume R2 {t1['code_volume_r2']:+.3f} / {evaluation['flair']['code_volume_r2']:+.3f}"
              f" | T1 " + " ".join(f"{k[5:]} {v:.2f}" for k, v in t1.items() if k.startswith("code_") and
                                    k not in ("code_volume_r2", "code_latent_r2"))
              + f" | {row['loss']}", flush=True)

    record(0)
    started = time.time()
    for step in range(1, steps + 1):
        if (step - 1) % REBUILD == 0:
            code = head_codes(pool, clean, units).cpu().numpy()
            neighbours = cb.nearest_neighbours(np.concatenate(list(code), 1), NEIGHBOURS)
            sampler = cb.NeighbourhoodBatchSampler(neighbours, BATCH, head_seed * 100_000 + step, 0.0)
            batches = iter(sampler)
        try:
            indices = np.asarray(next(batches))
        except StopIteration:
            batches = iter(sampler)
            indices = np.asarray(next(batches))
        draws = rng.integers(DRAWS, size=(2, len(indices)))
        subjects = torch.as_tensor(indices, device=device)
        inputs = torch.stack([jittered[torch.as_tensor(draws[v], device=device), v, subjects] for v in range(2)])
        code = torch.stack([pool.head(inputs[v]) for v in range(2)])
        pair = code[..., :units]
        batch_targets = targets[:, subjects]
        terms = {"global": warm.hard_negative_infonce(torch.cat(list(code)), units, TAU, 0.0, sim),
                 "predict": warm.cross_view_prediction(pair, batch_targets, decoders)}
        loss = terms["global"] + PREDICT_WEIGHT * terms["predict"]
        if sufficiency is not None:
            terms["sufficiency"] = sufficiency(pair, batch_targets)
            terms["explain"] = sufficiency.fit(pair, batch_targets)
            loss = loss + SUFFICIENCY_WEIGHT * terms["sufficiency"] + terms["explain"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trained, CLIP)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise ValueError(f"nonfinite loss or gradient at step {step}")
        optimizer.step()
        count += 1
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
        if step % EVERY == 0 or step == steps:
            record(step)
            totals, count = {}, 0
        if step % 2000 == 0:
            print(f"step {step:5d} {(time.time() - started) / step * 1000:.1f} ms/step", flush=True)
    if frozen != (FP.state_digest(model), FP.state_digest(pool, exclude_head=True)):
        raise RuntimeError("the frozen backbone or pooling changed")
    out.mkdir(parents=True, exist_ok=False)
    args = dict(a)
    args.update(steps=str(steps), init=str(source), frozen_backbone="True", head_refit=arm, head_seed=str(head_seed),
                sufficiency_weight=str(SUFFICIENCY_WEIGHT if arm == "suff" else 0.0), uniform="False",
                code_batches=str(REBUILD), code_neighbours=str(NEIGHBOURS))
    state = {"model": model.state_dict(), "pool": pool.state_dict(), "step": steps,
             "prediction_decoders": decoders.state_dict(), "target_whitening": stats}
    if sufficiency is not None:
        state["sufficiency"] = {"explainers": sufficiency.explainers.state_dict(), "running": sufficiency.running.cpu()}
    torch.save(state, out / "model.pt")
    (out / "report.json").write_text(json.dumps({"args": args, "history": history, "source": str(source),
                                                  "source_step": source_step}, indent=1, default=float))
    print("DONE", flush=True)


def summary(detect_json):
    rows = json.loads(Path(detect_json).read_text())["rows"]
    by = {}
    for label, row in rows.items():
        arm, seed = label.split("_")[1], int(label.rsplit("_s", 1)[1])
        by.setdefault(arm, {})[seed] = row
    seeds = sorted(SOURCE_HOLDS)
    fails = [s for s in seeds if not SOURCE_HOLDS[s]]
    holds = [s for s in seeds if SOURCE_HOLDS[s]]
    src = by.get("src", {})
    print("seed  source verdict (stage 75/76 | rescored)   control              suff")
    for s in seeds:
        cells = []
        for arm in ("src", "ctrl", "suff"):
            r = by.get(arm, {}).get(s)
            cells.append("-" if r is None else f"{'holds' if r['holds'] else 'fails'} rho1 {r['rho1']:.2f}")
        print(f"{s:4d}  {'holds' if SOURCE_HOLDS[s] else 'fails':5s} | {cells[0]:22s} {cells[1]:20s} {cells[2]}")
    mismatch = [s for s in seeds if s in src and src[s]["holds"] != SOURCE_HOLDS[s]]
    if mismatch:
        print(f"note: the rescored source verdict differs from stage 75/76's at seeds {mismatch}")
    count = {arm: {"fail": sum(by.get(arm, {}).get(s, {}).get("holds", False) for s in fails),
                   "hold": sum(by.get(arm, {}).get(s, {}).get("holds", False) for s in holds)}
             for arm in ("ctrl", "suff")}
    missing = [f"{arm}_s{s}" for arm in ("ctrl", "suff") for s in seeds if s not in by.get(arm, {})]
    for arm in ("ctrl", "suff"):
        c = count[arm]
        print(f"{arm}: holds in {c['fail']} of {len(fails)} failing codes and {c['hold']} of {len(holds)} holding codes")
    r1 = count["suff"]["fail"] >= 4
    r2 = count["suff"]["hold"] >= 7
    r3 = sum(count["suff"].values()) - sum(count["ctrl"].values()) >= 2
    print(f"R1 repair (suff >= 4 of 7 failures): {'pass' if r1 else 'fail'}")
    print(f"R2 no harm (suff >= 7 of 8 holders): {'pass' if r2 else 'fail'}")
    print(f"R3 the term (suff - control >= 2 of 15): {'pass' if r3 else 'fail'}")
    verdict = ("incomplete: missing " + ", ".join(missing) if missing else
               "works" if r1 and r2 and r3 else "fails" if count["suff"]["fail"] <= 2 else "unresolved")
    print(f"STAGE 80: {verdict}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("source", type=Path)
    t.add_argument("out", type=Path)
    t.add_argument("arm", choices=("control", "suff"))
    t.add_argument("--steps", type=int, default=20_000)
    t.add_argument("--cache", type=Path, default=None)
    t.add_argument("--head-seed", type=int, default=0)
    s = sub.add_parser("summary")
    s.add_argument("detect_json", type=Path)
    a = p.parse_args()
    if a.command == "summary":
        summary(a.detect_json)
        return
    cache = a.cache or STAGE / "cache" / f"{a.source.name}.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    train(a.source, a.out, a.arm, a.steps, cache, a.head_seed)


if __name__ == "__main__":
    main()
