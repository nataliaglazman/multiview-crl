"""Does a harder-negative batch rule pay more for burden's remainder? The run's loss on codes built from the true
latents of the causal bank, where burden = f(ventricle) + its own noise, so a code that holds ventricle gains only that
remainder from the lesion.

Each view's code holds five anatomy factors (brain, ventricle, thickness, lr, sulcal; temporal left out) at cross-view
r 0.9, every unit with independent noise (batch_price.py's model). One candidate unit at r 0.75 is added, and its reward
is the drop in loss:

  burden               the bank's burden: only its remainder beyond ventricle is new to the code
  burden, independent  the same values shuffled across subjects, so all of it is new, as on the independent bank
  temporal             another child whose parents the code holds: the factor that displaced burden by 20,000 steps

Rules, 32 subjects a batch, the neighbourhoods built on the five-unit code (both views) as --code-batches builds them:

  uniform              32 subjects drawn uniformly
  code50               the recipe: an anchor and 31 of its 50 nearest
  code40, code31       tighter neighbourhoods (warm.py --code-neighbours)
  ... hard G           negatives weighted by exp(G cosine) (warm.py --hard-negatives G, warm.hard_negative_infonce)

A rule helps if it raises burden's reward against temporal's, the competitor. A weighting changes the loss's scale, so
compare ratios across rules and rewards only within one.

    python out/burden_feasibility_20261102/remainder_price.py OUT_JSON (--bank CAUSAL_BANK | --simulate N)

--simulate draws N subjects from the causal bank's SCM (bank.causal_scm) instead of reading train_zc.npy.
"""

import argparse
import json
from pathlib import Path

import batch_price
import numpy as np
import torch
import warm
from own_variation import scm_settings

import data.cluster_batches as cb
from eval.synthetic.synthetic_dataset import sample_content_from_scm

BASE = [0, 1, 5, 7, 8]  # brain, ventricle, thickness, lr, sulcal
CANDIDATES = ("burden", "burden_independent", "temporal")
RULES = [("uniform", 0.0), ("code50", 0.0), ("code40", 0.0), ("code31", 0.0)]
RULES += [("code50", g) for g in (1.0, 2.0, 5.0, 10.0)] + [("code31", g) for g in (2.0, 5.0)]
BATCHES = 300
R_BASE, R_CANDIDATE = 0.9, 0.75
SIM = torch.nn.CosineSimilarity(dim=-1)


def latents(a):
    if a.bank:
        return np.array(np.load(f"{a.bank}/train_zc.npy"), dtype=np.float64)
    scm, scale, nonlinearity = scm_settings()
    gen = torch.Generator().manual_seed(0)
    draws = [sample_content_from_scm(scm, gen, scale, nonlinearity) for _ in range(a.simulate)]
    return torch.stack(draws).double().numpy()


def mean_loss(views, batches, concentration):
    z1, z2 = views
    return float(
        np.mean(
            [
                warm.hard_negative_infonce(torch.cat([z1[b], z2[b]]), z1.shape[1], batch_price.TAU, concentration, SIM)
                for b in batches
            ]
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_json", type=Path)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--bank", help="A causal bank (bank.py --causal); its training subjects' latents")
    source.add_argument("--simulate", type=int, help="Draw this many subjects from the causal bank's SCM instead")
    p.add_argument("--r-burden", type=float, default=R_CANDIDATE, help="Cross-view r of both burden units")
    a = p.parse_args()
    z = latents(a)
    n = len(z)
    rng = np.random.default_rng(0)
    base = batch_price.views(batch_price.standard(z[:, BASE]), R_BASE, rng)
    columns = {"burden": z[:, 2], "burden_independent": rng.permutation(z[:, 2]), "temporal": z[:, 6]}
    added = {}
    for name, column in columns.items():
        r = R_CANDIDATE if name == "temporal" else a.r_burden
        extra = batch_price.views(batch_price.standard(column[:, None]), r, rng)
        added[name] = [torch.cat([b, e], 1) for b, e in zip(base, extra)]
    batches = {"uniform": [rng.choice(n, 32, replace=False) for _ in range(BATCHES)]}
    code = torch.cat(base, 1).numpy()
    for size in (50, 40, 31):
        sampler = cb.NeighbourhoodBatchSampler(cb.nearest_neighbours(code, size), 32, 0, 0.0)
        drawn = []
        while len(drawn) < BATCHES:
            drawn.extend(np.asarray(b) for b in sampler)
        batches[f"code{size}"] = drawn[:BATCHES]
    corr = float(np.corrcoef(z[:, 1], z[:, 2])[0, 1])
    print(
        f"{n} subjects; corr(ventricle, burden) {corr:.2f}; base r {R_BASE}, burden r {a.r_burden}, temporal r"
        f" {R_CANDIDATE}",
        flush=True,
    )
    print(
        f"  {'rule':16s} {'base loss':>9s} | reward: {'burden':>8s} {'indep.':>8s} {'temporal':>8s} |"
        f" burden/temporal  burden/independent",
        flush=True,
    )
    report = {
        "subjects": n,
        "corr_ventricle_burden": corr,
        "r_base": R_BASE,
        "r_burden": a.r_burden,
        "r_temporal": R_CANDIDATE,
        "rules": {},
    }
    for rule, concentration in RULES:
        label = rule + (f" hard {concentration:g}" if concentration else "")
        start = mean_loss(base, batches[rule], concentration)
        reward = {name: start - mean_loss(added[name], batches[rule], concentration) for name in CANDIDATES}
        row = {
            "base_loss": start,
            "reward": reward,
            "burden_over_temporal": reward["burden"] / reward["temporal"],
            "burden_over_independent": reward["burden"] / reward["burden_independent"],
        }
        report["rules"][label] = row
        print(
            f"  {label:16s} {start:9.3f} | {'':7s} {reward['burden']:8.3f} {reward['burden_independent']:8.3f}"
            f" {reward['temporal']:8.3f} | {row['burden_over_temporal']:15.2f} {row['burden_over_independent']:19.2f}",
            flush=True,
        )
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
