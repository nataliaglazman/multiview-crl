"""What would InfoNCE pay for burden if the code already held it?

The twin test showed the lesion is worth ~0.001 nats to the run's global loss, but that code barely holds the lesion.
This asks about the destination instead. On a trained code, on the run's own loss function and batches:

  nats     Gaussian cross-view information in the code (canonical correlations), raw and inside a neighbourhood,
           against log B, the most InfoNCE can pay at batch size B.
  prize    the loss with one extra unit that holds the true burden, at several cross-view reliabilities, against the
           same unit holding view-independent noise (what an idle unit costs), and against the plain code.
  ramp     the loss as burden is mixed into the code at amplitude w, to show whether the reward starts first or
           second order in w, i.e. whether a small lesion signal gets any pull.
  ladder   the same loss on codes built from the true latents, adding one factor at a time, then burden last.

    python prize.py BANK OUT_JSON NAME=CHECKPOINT [...]
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "out/burden_twin_20261102")
import twin_common  # noqa: E402

import data.cluster_batches as cb  # noqa: E402
import training.losses as losses  # noqa: E402
import training.main_conv_synthetic as T  # noqa: E402

ANATOMY = [0, 1, 5, 6, 7, 8]
NAMES = {0: "brain", 1: "ventricle", 5: "thickness", 6: "temporal", 7: "lr", 8: "sulcal"}
BATCHES = 300
TAU = 0.1
SIM, CRIT = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()


def loss_of(z1, z2):
    """The run's loss on (B, d) per-view codes: standardized per view over the batch, then cross-view InfoNCE."""
    hz = T.standardize_contrastive_code(torch.stack((z1, z2)))
    value = losses.infonce_loss(
        hz,
        sim_metric=SIM,
        criterion=CRIT,
        tau=TAU,
        estimated_content_indices=[list(range(hz.shape[-1]))],
        subsets=[(0, 1)],
        cross_view_negs_only=True,
    )
    return float(value.squeeze())


def mean_loss(z1, z2, batches):
    return float(np.mean([loss_of(z1[b], z2[b]) for b in batches]))


def encode(checkpoint, bank, cache):
    if os.path.exists(cache):
        data = np.load(cache)
        return data["z1"], data["z2"]
    args = twin_common.run_args()
    torch.manual_seed(args.model_seed)
    device = "mps"
    model = T.build_encoder(args)
    weights = torch.load(checkpoint, map_location="cpu")["model"]
    print(" load", model.load_state_dict(weights, strict=False), flush=True)
    model = model.to(device).eval()
    t1 = np.load(f"{bank}/train_t1_04.npy", mmap_mode="r")
    flair = np.load(f"{bank}/train_flair.npy", mmap_mode="r")
    rows = [[], []]
    with torch.no_grad():
        for s in range(0, len(t1), 32):
            a = torch.from_numpy(np.asarray(t1[s : s + 32], dtype=np.float32))[:, None]
            b = torch.from_numpy(np.asarray(flair[s : s + 32], dtype=np.float32))[:, None]
            x = torch.cat([a, b]).to(device)
            h = model._encode(x, 2, None)
            code = model._global_code(h, x, 2)[:, :14].cpu()
            u, v = code.chunk(2)
            rows[0].append(u)
            rows[1].append(v)
    z1, z2 = torch.cat(rows[0]).numpy(), torch.cat(rows[1]).numpy()
    np.savez(cache, z1=z1, z2=z2)
    return z1, z2


def canonical(a, b):
    a, b = a - a.mean(0), b - b.mean(0)

    def whiten(x):
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        keep = s > s[0] * 1e-5
        return u[:, keep]

    rho = np.clip(np.linalg.svd(whiten(a).T @ whiten(b), compute_uv=False), 0, 0.999999)
    return rho, float(-0.5 * np.log(1 - rho**2).sum())


def noisy_unit(value, rho, rng):
    """Two views of ``value``, standard normal in rank, with cross-view correlation ``rho``."""
    keep = np.sqrt(rho)
    return [keep * value + np.sqrt(1 - rho) * rng.standard_normal(len(value)) for _ in range(2)]


def main():
    bank, out_json, states = sys.argv[1], sys.argv[2], sys.argv[3:]
    args = twin_common.run_args()
    zc = np.array(np.load(f"{bank}/train_zc.npy"), dtype=np.float64)
    vox = np.array(np.load(f"{bank}/train_vox.npy"), dtype=np.float64)
    burden = (vox - vox.mean()) / vox.std()
    n = len(zc)
    rng = np.random.default_rng(0)
    neighbours = cb.nearest_neighbours(zc[:, ANATOMY], args.stratify_neighbours)
    sampler = cb.NeighbourhoodBatchSampler(neighbours, 32, args.loader_seed, 0.0)
    near = []
    while len(near) < BATCHES:
        near.extend(sampler)
    near = [torch.as_tensor(b) for b in near[:BATCHES]]
    uniform = [torch.as_tensor(rng.choice(n, 32, replace=False)) for _ in range(BATCHES)]
    wide = {B: [torch.as_tensor(rng.choice(n, B, replace=False)) for _ in range(40)] for B in (128, 512, 2000)}
    batches = {"uniform32": uniform, "neighbour32": near, **{f"uniform{B}": v for B, v in wide.items()}}
    report = {
        "log_B": {k: float(np.log(len(v[0]))) for k, v in batches.items()},
        "anatomy_sd_within_neighbourhood_batches": float(
            np.mean([zc[b.numpy()][:, ANATOMY].std(0, ddof=1).mean() for b in near]) / zc[:, ANATOMY].std(0).mean()
        ),
        "burden_sd_within_neighbourhood_batches": float(
            np.mean([burden[b.numpy()].std(ddof=1) for b in near]) / burden.std()
        ),
    }

    for item in states:
        name, checkpoint = item.split("=", 1)
        z1, z2 = encode(checkpoint, bank, f"{os.path.dirname(out_json)}/codes_{name}.npz")
        z1 = (z1 - z1.mean(0)) / z1.std(0)
        z2 = (z2 - z2.mean(0)) / z2.std(0)
        t1, t2 = torch.from_numpy(z1).float(), torch.from_numpy(z2).float()
        row = {}
        rho, nats = canonical(z1, z2)
        res1 = z1 - z1[neighbours].mean(1)
        res2 = z2 - z2[neighbours].mean(1)
        rho_n, nats_n = canonical(res1, res2)
        row["info"] = {
            "canonical_r": rho.round(3).tolist(),
            "nats_total": nats,
            "nats_inside_neighbourhood": nats_n,
            "canonical_r_inside_neighbourhood": rho_n.round(3).tolist(),
        }
        print(name, row["info"], flush=True)

        row["prize"] = {}
        for label, sets in batches.items():
            base = mean_loss(t1, t2, sets)
            cell = {"base": base}
            noise = rng.standard_normal((n, 2))
            cell["idle_unit_noise"] = mean_loss(
                torch.cat([t1, torch.from_numpy(noise[:, :1]).float()], 1),
                torch.cat([t2, torch.from_numpy(noise[:, 1:]).float()], 1),
                sets,
            )
            for reliability in (1.0, 0.9, 0.7, 0.5):
                u, v = noisy_unit(burden, reliability, rng)
                cell[f"burden_unit_rho{reliability}"] = mean_loss(
                    torch.cat([t1, torch.from_numpy(u[:, None]).float()], 1),
                    torch.cat([t2, torch.from_numpy(v[:, None]).float()], 1),
                    sets,
                )
            row["prize"][label] = cell
            print(" ", label, {k: round(v, 4) for k, v in cell.items()}, flush=True)

        # Amplitude ramp: burden mixed into every unit along a random direction, as a weak signal would arrive.
        direction = rng.standard_normal(14)
        direction /= np.linalg.norm(direction)
        ramp = {}
        for label in ("uniform32", "neighbour32"):
            ramp[label] = {}
            for w in (0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5):
                u = t1 + float(w) * torch.from_numpy(np.outer(burden, direction) * np.sqrt(14)).float()
                v = t2 + float(w) * torch.from_numpy(np.outer(burden, direction) * np.sqrt(14)).float()
                ramp[label][str(w)] = mean_loss(u, v, batches[label])
            print("  ramp", label, {k: round(x, 4) for k, x in ramp[label].items()}, flush=True)
        row["ramp"] = ramp
        report[name] = row
        with open(out_json, "w") as stream:
            json.dump(report, stream, indent=1)

    # True-latent ladder: each factor seen by both views with independent noise at cross-view r = 0.96.
    ladder = {}
    order = [1, 0, 5, 6, 7, 8]
    for label in ("uniform32", "neighbour32", "uniform512", "uniform2000"):
        rows = []
        columns = [[], []]
        for factor in [*order, "burden"]:
            value = burden if factor == "burden" else (zc[:, factor] - zc[:, factor].mean()) / zc[:, factor].std()
            u, v = noisy_unit(value, 0.96, rng)
            columns[0].append(u)
            columns[1].append(v)
            a = torch.from_numpy(np.stack(columns[0], 1)).float()
            b = torch.from_numpy(np.stack(columns[1], 1)).float()
            rows.append(("burden" if factor == "burden" else NAMES[factor], mean_loss(a, b, batches[label])))
        ladder[label] = rows
        print(" ladder", label, [(k, round(x, 3)) for k, x in rows], flush=True)
    report["ladder_r096"] = ladder
    with open(out_json, "w") as stream:
        json.dump(report, stream, indent=1)


if __name__ == "__main__":
    main()
