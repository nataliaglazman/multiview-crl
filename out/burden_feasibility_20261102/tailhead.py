"""Label-free route to the vector: normative tail pooling over a self-supervised map, then the run's own InfoNCE.

From a frozen backbone, each channel at each interior cell (mean brain occupancy >= 0.9) is z-scored by that
position's training mean and SD. No labels are used. Each subject then gets, per channel, the mean of ReLU(z - t) and
ReLU(-z - t) over interior cells at t = 2, 3, 4, next to the channel's GAP: a position-free summary of how far the
map strays from the population, for any factor. A random head is trained on these with the run's global InfoNCE.
Labels only score, by ridge on the test split.

    python tailhead.py BANK T1KEY OUT_JSON NAME=CHECKPOINT [...]
"""

import json
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import headonly  # noqa: E402
import warm  # noqa: E402

import data.cluster_batches as cb  # noqa: E402
import training.main_conv_synthetic as T  # noqa: E402

THRESHOLDS = (2.0, 3.0, 4.0)
OCCUPANCY = 0.9


def interior_cells(root):
    sample = torch.from_numpy(
        (np.asarray(np.load(f"{root}/train_flair.npy", mmap_mode="r")[:300]) != 0).astype(np.float32)
    )
    occupancy = torch.nn.functional.avg_pool3d(sample[:, None], 4)[:, 0].mean(0)
    return (occupancy >= OCCUPANCY).flatten().numpy()


def extract(checkpoint, root, t1key, keep, cache):
    if os.path.exists(cache):
        data = np.load(cache)
        return {s: [data[f"{s}_0"], data[f"{s}_1"]] for s in ("train", "test")}
    args = warm.run_args()
    torch.manual_seed(args.model_seed)
    model = T.build_encoder(args)
    state = torch.load(checkpoint, map_location="cpu")["model"]
    if any(k.startswith("encoder.encoder.") for k in state):
        state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.encoder")}
    print(" load", model.load_state_dict(state, strict=False).unexpected_keys[:3], flush=True)
    device = warm.pick_device()
    model = model.to(device).eval()
    index = torch.from_numpy(np.flatnonzero(keep)).to(device)
    out = {}
    with torch.no_grad():
        for split in ("train", "test"):
            bank = warm.Bank(root, split, t1key)
            rows = [[], []]
            for start in range(0, len(bank.vox), 32):
                x = bank.images(np.arange(start, min(start + 32, len(bank.vox))), device)
                h = model._encode(x, 2, None).flatten(2)[:, :, index].cpu().numpy().astype(np.float32)
                for view, part in enumerate(np.split(h, 2)):
                    rows[view].append(part)
            out[split] = [np.concatenate(r) for r in rows]
    np.savez(cache, **{f"{s}_{v}": out[s][v] for s in out for v in range(2)})
    return out


def compress(rows, reference):
    """Tail means are heavy-tailed: log-compress on the training scale, standardize, clip to +/-6."""
    scale = reference.mean(0, keepdims=True) + 1e-9
    logged = np.log1p(rows / scale)
    base = np.log1p(reference / scale)
    return np.clip((logged - base.mean(0, keepdims=True)) / np.maximum(base.std(0, keepdims=True), 1e-3), -6, 6)


def tails(maps):
    result = {"train": [], "test": []}
    for view in range(2):
        mean = maps["train"][view].mean(0, keepdims=True)
        sd = maps["train"][view].std(0, keepdims=True) + 1e-6

        def raw(split):
            z = (maps[split][view] - mean) / sd
            parts = []
            for t in THRESHOLDS:
                parts += [np.maximum(z - t, 0).mean(2), np.maximum(-z - t, 0).mean(2)]
            return np.concatenate(parts, 1)

        reference = raw("train")
        for split in ("train", "test"):
            result[split].append(np.concatenate([maps[split][view].mean(2), compress(raw(split), reference)], 1))
    return result


def main():
    root, t1key, out_json, specs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    folder = os.path.dirname(out_json)
    keep = interior_cells(root)
    print("interior cells", int(keep.sum()), flush=True)
    train = warm.Bank(root, "train", t1key)
    test = warm.Bank(root, "test", t1key)
    neighbours = cb.nearest_neighbours(train.zc[:, warm.ANATOMY], 50)
    report = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for item in specs:
        name, checkpoint = item.split("=", 1)
        maps = extract(checkpoint, root, t1key, keep, f"{folder}/maps_{t1key}_{name}.npz")
        pooled = tails(maps)
        row = {
            "input_t1": headonly.probes(pooled["test"][0], test),
            "input_flair": headonly.probes(pooled["test"][1], test),
            "cross_view_linear_r": None,
        }
        print(name, "tail ridge t1", row["input_t1"], flush=True)
        print(name, "tail ridge flair", row["input_flair"], flush=True)
        for label, shared, uniform in (
            ("neighbour_sep", False, False),
            ("uniform_sep", False, True),
            ("neighbour_shared", True, False),
        ):
            loss, c1, c2 = headonly.train_head(pooled, neighbours, shared, uniform)
            row[label] = {
                "loss_last200": round(loss, 4),
                "code_t1": headonly.probes(c1, test),
                "code_flair": headonly.probes(c2, test),
            }
            print(name, label, row[label], flush=True)
        report[name] = row
        with open(out_json, "w") as stream:
            json.dump(report, stream, indent=1)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())
