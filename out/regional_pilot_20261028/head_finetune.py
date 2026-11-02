"""Does the 14-unit cell code take up the lesion when the prediction term asks for what the population does not explain?

The backbone stays frozen at a checkpoint, so its map cannot change. The head and fresh predictors are fine-tuned on cached
train maps under the run's three terms, each on its own AdamW, with the prediction target raw or pcaK
(residual_targets.py). Lesion labels only score the result, through cellcode.score, at EVALS.

    PYTHONPATH=SHIM:. python out/regional_pilot_20261028/head_finetune.py BANK_DIR OUT_JSON CHECKPOINT ARM ...
"""

import copy
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "out/regional_pilot_20261028")
import residual_targets as RT  # noqa: E402

C, L, R, pilot, T = RT.C, RT.L, RT.R, RT.pilot, RT.C.T
STEPS = 2000
EVALS = (500, 2000)
VIEWS = RT.VIEWS


def cached_maps(model, bank, device):
    """Per view: (n, 64, 16, 16, 16) half-precision backbone maps on the CPU."""
    out = [[], []]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(bank), RT.BATCH):
            idx = list(range(start, min(start + RT.BATCH, len(bank))))
            h = model._encode(bank.batch(idx, device)[0], 2, None).cpu().half()
            out[0].append(h[: len(idx)])
            out[1].append(h[len(idx) :])
    return [torch.cat(part) for part in out]


def readouts(model, h, args):
    patches = model.patch_readout(model._patch_pool(h, args.train_patch_grid).transpose(1, 2)).transpose(1, 2)
    return model._global_code(h, None, 2), patches


def parity(model, bank, maps, args, device):
    """The cached path must give the training path's readouts at the checkpoint."""
    idx = list(range(RT.BATCH))
    x, masks, _, _ = bank.batch(idx, device)
    with torch.no_grad():
        pooled, patches = model.global_and_patch_features(x, args.train_patch_grid, n_views=2)
        h = torch.cat([maps[0][idx], maps[1][idx]]).float().to(device)
        cached_pooled, cached_patches = readouts(model, h, args)
    print(
        "parity: largest difference, global",
        round((pooled - cached_pooled).abs().max().item(), 4),
        "patch",
        round((patches - cached_patches).abs().max().item(), 4),
        "of a code scale",
        round(patches.abs().mean().item(), 3),
        flush=True,
    )


def scored(model, predictors, val, val_target, centre, keep, brain, lesion, device):
    out = C.cell_maps(model, val, device)
    codes = {key: out[key][:, : C.CONTENT] for key in ("code_t1", "code_flair")}
    row = C.score({**out, **codes}, keep, brain, lesion)
    with torch.no_grad():
        for v, view in enumerate(VIEWS):
            guess = predictors[v](codes[f"code_{view}"].to(device))[:, 0].cpu().numpy()
            deviation = (guess - centre[v])[:, None]
            row[f"{view} guess AUROC"], row[f"{view} guess top cell"] = C.probe_between(
                deviation, deviation, brain, lesion
            )
    return row, R.evaluate(model, val, device)


def report(arm, step, row, factors, losses):
    pick = lambda key: f"{row[key + ' AUROC']:.3f} / {100 * row[key + ' top cell']:4.1f}"  # noqa: E731
    print(
        f"{arm:<7}{step:>5}  T1 own {pick('t1 double')}  into T1 {pick('t1 double transfer')}  "
        f"FLAIR own {pick('flair double')}  into FLAIR {pick('flair double transfer')}  "
        f"T1 guess {pick('t1 guess')}  FLAIR guess {pick('flair guess')}  losses {losses}",
        flush=True,
    )
    for key in ("code_t1", "code_flair"):
        print(f"       {key:<11}" + " ".join(f"{k[:10]} {v:.2f}" for k, v in factors[key].items()), flush=True)


def main():
    bank_dir, out_json, state, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args = R.arm_args("predcode_sepadam")
    train, val = pilot.Bank(bank_dir, "train"), pilot.Bank(bank_dir, "val")
    brain = val.keep[:, 0].numpy()
    lesion = np.load(L.CACHE)
    keep = T.foreground_positions(val.mask[:, None], C.CELLS, C.FOREGROUND)
    models = [RT.population_basis(train, v) for v in range(2)]
    fit_targets, val_targets = RT.targets_for(train, models), RT.targets_for(val, models)
    model = C.build(state, pilot.user_args(), device)
    started = time.time()
    maps = cached_maps(model, train, device)
    print(f"cached maps ({time.time() - started:.0f} s)", flush=True)
    parity(model, train, maps, args, device)
    initial = copy.deepcopy(model.to_encoding.state_dict())
    backbone = copy.deepcopy(model.encoder.state_dict())
    sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
    results = {}
    for arm in arms:
        model.to_encoding.load_state_dict(initial)
        torch.manual_seed(args.model_seed)
        predictors = torch.nn.ModuleList(torch.nn.Conv3d(C.CONTENT, 1, 1) for _ in range(2)).to(device)
        head = list(model.to_encoding.parameters())
        tail = list(predictors.parameters())
        optimizers = (
            torch.optim.AdamW(head, lr=args.lr),
            torch.optim.AdamW(head, lr=args.lr, weight_decay=0.0),
            torch.optim.AdamW(head + tail, lr=args.lr, weight_decay=0.0),
        )
        centre = [RT.population(fit_targets[arm][v], train.keep[:, 0]) for v in range(2)]
        order = np.random.default_rng(args.loader_seed)
        queue, sums, started = [], np.zeros(3), time.time()
        model.train()
        for step in range(1, STEPS + 1):
            if len(queue) < args.batch_size:
                queue += order.permutation(len(train)).tolist()
            idx, queue = torch.as_tensor(queue[: args.batch_size]), queue[args.batch_size :]
            h = torch.cat([maps[0][idx], maps[1][idx]]).float().to(device)
            masks = torch.cat([train.mask[idx], train.mask[idx]])[:, None].to(device)
            kept = torch.cat([train.keep[idx], train.keep[idx]]).to(device)
            target = torch.cat([fit_targets[arm][0][idx], fit_targets[arm][1][idx]])[:, None].to(device)
            pooled, patches = readouts(model, h, args)
            terms = (
                T.patch_contrastive_loss(patches, model, args, sim, crit, masks),
                T.contrastive_loss(pooled, model, args, sim, crit),
                R.code_prediction(patches, args, predictors, target, kept),
            )
            sums += [term.item() for term in terms]
            params = head + tail
            grads = [
                torch.autograd.grad(loss, params, retain_graph=i < 2, allow_unused=True) for i, loss in enumerate(terms)
            ]
            for optimizer, term_grads in zip(optimizers, grads):
                for p, g in zip(params, term_grads):
                    p.grad = torch.zeros_like(p) if g is None else g
                torch.nn.utils.clip_grad_norm_(head, args.grad_clip)
                torch.nn.utils.clip_grad_norm_(tail, args.grad_clip)
                optimizer.step()
            if step % 500 == 0:
                print(
                    f"  {arm} step {step} ({(time.time() - started) / step:.3f} s/step) region {sums[0] / 500:.4f} global {sums[1] / 500:.4f} predict {sums[2] / 500:.4f}",
                    flush=True,
                )
                means = [round(float(v / 500), 4) for v in sums]
                sums = np.zeros(3)
            if step in EVALS:
                row, factors = scored(model, predictors, val, val_targets[arm], centre, keep, brain, lesion, device)
                report(arm, step, row, factors, means)
                results[f"{arm} s{step}"] = {"probes": row, "factors": factors, "losses": means}
                json.dump(results, open(out_json, "w"), indent=1)
                model.train()
    same = all(torch.equal(v, backbone[k]) for k, v in model.encoder.state_dict().items())
    print("backbone unchanged:", same, flush=True)


if __name__ == "__main__":
    main()
