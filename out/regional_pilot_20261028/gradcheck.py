"""Per-layer gradient check at saved checkpoints, no training: do the global and region terms pull against each other,
and how much does Adam shrink the region term's steps once the global term shares its step sizes?

N fixed training batches, the same for every checkpoint. Per batch: the gradient of the global term g_G (trained at the
arm's weight w) and of the region term g_R over every weight. Training clips each batch's summed gradient to norm 2,
and AdamW divides each weight's step by the root of a running mean square v of what it was fed. Both are replayed
here, once for the trained gradient w*g_G + g_R and once for g_R alone. Per layer, and for the whole network:
  share         the layer's part of the region term's own progress per step, without the global term
  rms G/R       root-mean-square of w*g_G over that of g_R: which term sets the layer's step sizes
  cos           angle between the expected gradients E[g_G] and E[g_R], from independent batch halves (one half's mean
                of one term against the other half's mean of the other, so batch noise does not pull it toward 0);
                rel G and rel R are each term's split-half cosine, how reproducible its direction is
  dilution      the region term's own progress per step with the global term in v, over that without it
                (1 = Adam's step sizes untouched; 0.33 = region steps shrunk 3x)
  interf.       the global term's part of the step, projected on the region gradient, over the region term's own part,
                in Adam's metric (negative = the global steps undo region progress)
  slowdown      region-loss progress per step without the global term over with it, to first order:
                1 / (dilution * (1 + interf.)); compare with the steps-to-same-loss ratios in the logs
  v/v_adam      median over the layer's weights of the replayed v over the checkpoint's own (bias-corrected) Adam v

    PYTHONPATH=out/regional_pilot_20261028/shim:. python out/regional_pilot_20261028/gradcheck.py BANK_DIR OUT_JSON N \
        init:ARM|CHECKPOINT ...
init:ARM is that arm's untrained model, scored at global weights 0.1, 0.3 and 1; a checkpoint ARM_sSTEP.pt is scored at
its arm's weight (1 if the arm trained without the global term).
"""

import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regional as R  # noqa: E402

LAYERS = (
    ("layers.0.", "conv1"),
    ("layers.1.", "conv2"),
    ("layers.2.", "conv3"),
    ("layers.3.", "conv3"),
    ("layers.4.stack.0.", "res1"),
    ("layers.4.stack.1.", "res2"),
)
EPS = 1e-8


def layer_of(name):
    if name.startswith("to_encoding"):
        return "head"
    view, rest = name.split(".", 1)
    for prefix, label in LAYERS:
        if rest.startswith(prefix):
            return f"{'T1' if view == 'encoder' else 'FLAIR'} {label}"
    raise ValueError(name)


def load(spec):
    if spec.startswith("init:"):
        arm, state, name = spec[5:], None, f"init ({spec[5:]})"
    else:
        name = os.path.splitext(os.path.basename(spec))[0]
        arm, state = re.sub(r"_s\d+$", "", name), torch.load(spec, map_location="cpu")
    args = R.arm_args(arm)
    torch.manual_seed(args.model_seed)
    model = R.T.build_encoder(args)
    if state is not None:
        model.load_state_dict(state["model"])
    return name, arm, args, model, state


def gradients(model, args, bank, batches, device):
    """(N, P) float64 gradients of the global and of the region term, one row per batch."""
    sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
    params = list(model.parameters())
    rows = {"global": [], "patch_weighted": []}
    for b, idx in enumerate(batches):
        torch.manual_seed(b)
        x, masks = bank.batch(list(idx), device)[:2]
        _, _, terms = R.T.training_objective(model, x, args, sim, crit, masks)
        for key, retain in (("global", True), ("patch_weighted", False)):
            grads = torch.autograd.grad(terms[key], params, retain_graph=retain, allow_unused=True)
            flat = [(torch.zeros_like(p) if g is None else g).flatten() for p, g in zip(params, grads)]
            rows[key].append(torch.cat(flat).cpu())
    return torch.stack(rows["global"]).double(), torch.stack(rows["patch_weighted"]).double()


def adam_v(state, params):
    """The checkpoint's bias-corrected AdamW second moment, flattened in parameter order."""
    beta2 = state["optimizer"]["param_groups"][0]["betas"][1]
    out = []
    for i, p in enumerate(params):
        s = state["optimizer"]["state"][i]
        out.append((s["exp_avg_sq"] / (1 - beta2 ** float(s["step"]))).flatten())
    return torch.cat(out).double()


def check(G, Rg, w, clip, groups, v_adam=None):
    n = len(G)
    halves = (slice(0, n // 2), slice(n // 2, n))
    c_w = torch.clamp(clip / (w * w * (G * G).sum(1) + 2 * w * (G * Rg).sum(1) + (Rg * Rg).sum(1)).sqrt(), max=1.0)
    c_0 = torch.clamp(clip / (Rg * Rg).sum(1).sqrt(), max=1.0)
    v_w = ((c_w[:, None] * (w * G + Rg)) ** 2).mean(0)
    s_w = v_w.sqrt() + EPS
    s_0 = ((c_0[:, None] * Rg) ** 2).mean(0).sqrt() + EPS
    g_R = [Rg[h].mean(0) for h in halves]
    g_G = [G[h].mean(0) for h in halves]
    m_R = [(c_w[h, None] * Rg[h]).mean(0) for h in halves]
    m_G = [(c_w[h, None] * w * G[h]).mean(0) for h in halves]
    m_0 = [(c_0[h, None] * Rg[h]).mean(0) for h in halves]
    rows = {}
    for label, index in groups.items():

        def cross(a, b, s=1.0):
            s = s[index] if torch.is_tensor(s) else s
            return 0.5 * float((a[0][index] * b[1][index] / s).sum() + (a[1][index] * b[0][index] / s).sum())

        def rel(a):
            return float((a[0][index] * a[1][index]).sum() / (a[0][index].norm() * a[1][index].norm()))

        own_w, push_w, own_0 = cross(g_R, m_R, s_w), cross(g_R, m_G, s_w), cross(g_R, m_0, s_0)
        gg, rr, gr = cross(g_G, g_G), cross(g_R, g_R), cross(g_G, g_R)
        trained = v_adam is not None and bool((v_adam[index] > 0).any())
        if trained:
            used = v_adam[index] > 0
        rows[label] = {
            "own0": own_0,
            "rms G/R": float(w * G[:, index].square().sum().sqrt() / Rg[:, index].square().sum().sqrt()),
            "cos": gr / np.sqrt(gg * rr) if gg > 0 and rr > 0 else float("nan"),
            "rel G": rel(g_G),
            "rel R": rel(g_R),
            "dilution": own_w / own_0,
            "interf.": push_w / own_w,
            "slowdown": own_0 / (own_w + push_w),
            "v/v_adam": float((v_w[index][used] / v_adam[index][used]).median()) if trained else float("nan"),
        }
    total = rows["all"]["own0"]
    for row in rows.values():
        row["share"] = row.pop("own0") / total
    rows["all"]["clipped"] = float((c_w < 1).double().mean())
    rows["all"]["clipped without"] = float((c_0 < 1).double().mean())
    return rows


def show(name, w, rows):
    a = rows["all"]
    print(
        f"\n== {name}, global weight {w:g}: slowdown {a['slowdown']:.2f}x = 1 / (dilution {a['dilution']:.3f}"
        f" x (1 + interf. {a['interf.']:+.3f})); cos {a['cos']:+.3f}; batches clipped {a['clipped']:.0%}"
        f" (region term alone {a['clipped without']:.0%})",
        flush=True,
    )
    columns = ("share", "rms G/R", "cos", "rel G", "rel R", "dilution", "interf.", "v/v_adam")
    print(f"{'layer':>12}" + "".join(f"{c:>10}" for c in columns), flush=True)
    for label, row in rows.items():
        print(f"{label:>12}" + "".join(f"{row[c]:>10.3f}" for c in columns), flush=True)


def main():
    bank_dir, out_json, n_batches, specs = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    bank = R.P.Bank(bank_dir, "train")
    order = np.random.default_rng(0).permutation(len(bank))
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for spec in specs:
        name, arm, args, model, state = load(spec)
        batches = [order[i * args.batch_size : (i + 1) * args.batch_size] for i in range(n_batches)]
        params = list(model.parameters())
        labels = [layer_of(p_name) for p_name, _ in model.named_parameters()]
        sizes = [p.numel() for p in params]
        groups = {"all": torch.ones(sum(sizes), dtype=torch.bool)}
        position = np.repeat(np.arange(len(sizes)), sizes)
        for label in dict.fromkeys(labels):
            groups[label] = torch.from_numpy(np.isin(position, [i for i, lb in enumerate(labels) if lb == label]))
        G, Rg = gradients(model.to(device).train(), args, bank, batches, device)
        v_adam = adam_v(state, params) if state is not None and "optimizer" in state else None
        weight = R.ARMS[arm][1]
        weights = (0.1, 0.3, 1.0) if state is None else (weight if weight > 0 else 1.0,)
        for w in weights:
            rows = check(G, Rg, w, args.grad_clip, groups, v_adam)
            show(name, w, rows)
            results[f"{name} w{w:g}"] = rows
        json.dump(results, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
