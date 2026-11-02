"""What does the 14-unit cell code do with the lesion? Read from a checkpoint, with no training.

Cells are the cell probe's own: positives at least LESION_FRACTION lesion, negatives lesion-free, in the placement
region. Every number is on validation subjects, for each view.

  signal    how far a lesion moves a cell. size is the distance between the positive and negative means; spread is the
            negatives' sd along that line; size / spread is the separation along it; dprime is the separation a linear
            read can reach by also cancelling the background (Mahalanobis). Rows: the centred map, 14 random directions
            of it, and the code with the arm's centring.
  gain      how strongly the head passes a direction of the map: the 14 content rows of its Jacobian at the lesion cells,
            on the lesion's direction, on random directions, and on the map's largest directions of variation.
  predictor the probe on the code with the checkpoint's predictor direction removed, and the predictor's share of the
            code's lesion displacement. Skipped when the checkpoint has no predictor.
  push      what each term asks of the cell code: its gradient at lesion cells along the lesion's displacement, as a
            cosine (positive grows the lesion response, negative shrinks it), and the gradient's size at lesion cells
            relative to all kept cells. error is the prediction term's squared error at lesion cells against other cells.

    PYTHONPATH=SHIM:. python out/regional_pilot_20261028/code_lesion_audit.py BANK_DIR OUT_JSON CHECKPOINT ...
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "out/regional_pilot_20261028")
import cellcode as C  # noqa: E402

L, R, pilot, posthoc, T = C.L, C.R, C.pilot, C.posthoc, C.T
VIEWS = ("t1", "flair")
BATCH = 32
PUSH_BATCHES = 12
RANDOM_GAINS = 200


def probe_cells(brain, lesion):
    """(subject, cell) of the probe's positives and negatives, flattened cell index."""
    n = len(lesion)
    lesion = lesion.reshape(n, -1)
    brain = brain.reshape(n, -1) & (lesion >= posthoc.LESION_FRACTION).any(0)
    rng = np.random.default_rng(0)
    pos, neg = [], []
    for i in range(n):
        pos += [(i, c) for c in np.flatnonzero(brain[i] & (lesion[i] >= posthoc.LESION_FRACTION))]
        free = np.flatnonzero(brain[i] & (lesion[i] == 0))
        neg += [(i, c) for c in rng.choice(free, size=min(len(free), 40), replace=False)]
    return np.array(pos), np.array(neg)


def at(values, cells):
    """(cells, channels) from (n, channels, 16, 16, 16)."""
    flat = values.reshape(*values.shape[:2], -1)
    return flat[cells[:, 0], :, cells[:, 1]]


def signal(values, pos, neg):
    on, off = at(values, pos), at(values, neg)
    line = on.mean(0) - off.mean(0)
    cov = np.cov(off, rowvar=False)
    size = float(np.linalg.norm(line))
    spread = float(np.sqrt(line @ cov @ line) / size)
    ridge = 1e-6 * np.trace(cov) / len(cov) * np.eye(len(cov))
    return {
        "size / spread": size / spread,
        "dprime": float(np.sqrt(line @ np.linalg.solve(cov + ridge, line))),
    }


def jacobians(model, h_cells):
    """(cells, 14, 64) Jacobian of the head's content units at raw map vectors."""
    first, act, last = model.to_encoding[0], model.to_encoding[1], model.to_encoding[2]
    with torch.no_grad():
        slopes = (first(h_cells) > 0).float() + (first(h_cells) <= 0).float() * act.negative_slope
    return (
        last.weight[: C.CONTENT].detach().cpu().numpy()[None]
        * slopes.cpu().numpy()[:, None]
        @ first.weight.detach().cpu().numpy()
    )


def gains(model, h, centred, pos, neg):
    device = next(model.parameters()).device
    j = jacobians(model, torch.from_numpy(at(h, pos)).to(device))
    line = at(centred, pos).mean(0) - at(centred, neg).mean(0)
    line = line / np.linalg.norm(line)
    eigenvalues, vectors = np.linalg.eigh(np.cov(at(centred, neg), rowvar=False))
    top = vectors[:, ::-1]
    gain = lambda d: float(np.linalg.norm(j @ d, axis=-1).mean())  # noqa: E731
    draws = np.random.default_rng(0).standard_normal((64, RANDOM_GAINS))
    draws = draws / np.linalg.norm(draws, axis=0)
    random = float(np.mean([gain(draws[:, k]) for k in range(RANDOM_GAINS)]))
    return {
        "lesion direction": gain(line) / random,
        "largest direction": gain(top[:, 0]) / random,
        "largest 5": float(np.mean([gain(top[:, k]) for k in range(5)])) / random,
        "largest 14": float(np.mean([gain(top[:, k]) for k in range(14)])) / random,
        "smallest 14": float(np.mean([gain(top[:, -1 - k]) for k in range(14)])) / random,
        "lesion share of top 14": float(np.sum((top[:, :14].T @ line) ** 2)),
    }


def without_predictor(out, weights, keep, double, brain, lesion):
    """The probe on each view's code with its own predictor direction removed from the raw units."""
    codes = []
    for v, view in enumerate(VIEWS):
        unit = weights[v] / np.linalg.norm(weights[v])
        code = out[f"code_{view}"][:, : C.CONTENT].numpy()
        along = np.moveaxis(np.moveaxis(code, 1, -1) @ unit, -1, 1)[:, None]
        codes.append(torch.from_numpy(code - along * unit[None, :, None, None, None]))
    units = C.standardised(*codes, keep, double)[0]
    row = {}
    for v, view in enumerate(VIEWS):
        row[f"{view} own"] = C.probe_between(units[v], units[v], brain, lesion)
        row[f"{view} transfer"] = C.probe_between(units[1 - v], units[v], brain, lesion)
    return row


def push(model, predictors, args, bank, lesion, displacement, device):
    """Per view: the cosine of each term's descent direction with the lesion's displacement at lesion cells."""
    sim, crit = torch.nn.CosineSimilarity(dim=-1), torch.nn.CrossEntropyLoss()
    flat = torch.from_numpy(lesion.reshape(len(lesion), -1) >= posthoc.LESION_FRACTION)
    terms = ["region"] + ([] if predictors is None else ["predict"])
    totals = {(view, term): [0.0, 0.0, 0.0, 0] for view in VIEWS for term in terms}
    errors = [0.0, 0.0, 0.0, 0.0]
    for b in range(PUSH_BATCHES):
        idx = list(range(b * BATCH, (b + 1) * BATCH))
        x, masks, targets, keep = bank.batch(idx, device)
        patches = model.global_and_patch_features(x, args.train_patch_grid, n_views=2)[1]
        losses = {"region": T.patch_contrastive_loss(patches, model, args, sim, crit, masks)}
        if predictors is not None:
            losses["predict"] = R.code_prediction(patches, args, predictors, targets, keep)
            with torch.no_grad():
                cells = patches[:, : C.CONTENT].reshape(len(patches), C.CONTENT, *args.train_patch_grid)
                guess = torch.cat([predictors[0](cells[:BATCH]), predictors[1](cells[BATCH:])])
                err = ((guess - targets).square() * keep).flatten(1).cpu()
            inside = keep.flatten(1).cpu()
            hit = torch.cat([flat[idx], flat[idx]]) & inside
            errors[0] += err[hit].sum().item()
            errors[1] += hit.sum().item()
            errors[2] += err[inside & ~hit].sum().item()
            errors[3] += (inside & ~hit).sum().item()
        for term, loss in losses.items():
            grad = torch.autograd.grad(loss, patches, retain_graph=True)[0][:, : C.CONTENT].detach().cpu()
            inside = keep.flatten(1).cpu()
            for v, view in enumerate(VIEWS):
                part = grad[v * BATCH : (v + 1) * BATCH]
                on = flat[idx] & inside[v * BATCH : (v + 1) * BATCH]
                vectors = part.movedim(1, -1)[on]
                unit = torch.from_numpy(displacement[view]).float()
                unit = unit / unit.norm()
                row = totals[view, term]
                row[0] += (-(vectors @ unit) / vectors.norm(dim=1).clamp_min(1e-12)).sum().item()
                row[1] += vectors.norm(dim=1).sum().item()
                row[2] += part.movedim(1, -1)[inside[v * BATCH : (v + 1) * BATCH]].norm(dim=1).mean().item()
                row[3] += len(vectors)
    out = {}
    for (view, term), (cosines, norms, background, count) in totals.items():
        out[f"{view} {term} cosine"] = cosines / count
        out[f"{view} {term} size at lesion cells"] = (norms / count) / (background / PUSH_BATCHES)
    if predictors is not None:
        out["error at lesion cells / other cells"] = (errors[0] / errors[1]) / (errors[2] / errors[3])
        out["lesion cells' share of the error"] = errors[0] / (errors[0] + errors[2])
    return out


def main():
    bank_dir, out_json, states = sys.argv[1], sys.argv[2], sys.argv[3:]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args = R.arm_args("predcode_sepadam")
    bank = pilot.Bank(bank_dir, "val")
    brain = bank.keep[:, 0].numpy()
    lesion = np.load(L.CACHE)
    keep = T.foreground_positions(bank.mask[:, None], C.CELLS, C.FOREGROUND)
    pos, neg = probe_cells(brain, lesion)
    print(f"{len(pos)} positive and {len(neg)} negative cells", flush=True)
    results = {}
    for state in states:
        name = os.path.splitext(os.path.basename(state))[0]
        started = time.time()
        model = C.build(state, pilot.user_args(), device)
        saved = torch.load(state, map_location="cpu").get("predictors")
        predictors = None
        if saved is not None:
            predictors = torch.nn.ModuleList(torch.nn.Conv3d(C.CONTENT, 1, 1) for _ in range(2)).to(device)
            predictors.load_state_dict(saved)
        out = C.cell_maps(model, bank, device)
        double = C.standardised(out["code_t1"][:, : C.CONTENT], out["code_flair"][:, : C.CONTENT], keep, True)[0]
        row, displacement = {}, {}
        for v, view in enumerate(VIEWS):
            h = out[view].numpy()
            centred = h - h.mean(0, keepdims=True)
            row[f"{view} map"] = signal(centred, pos, neg)
            draws = [signal(C.random_directions(centred, d), pos, neg) for d in range(C.DRAWS)]
            row[f"{view} random14"] = {key: float(np.mean([d[key] for d in draws])) for key in draws[0]}
            row[f"{view} code"] = signal(double[v], pos, neg)
            row[f"{view} gain"] = gains(model, h, centred, pos, neg)
            code = out[f"code_{view}"][:, : C.CONTENT].numpy()
            code = code - code.mean(0, keepdims=True)
            displacement[view] = at(code, pos).mean(0) - at(code, neg).mean(0)
        if saved is not None:
            weights = [saved[f"{v}.weight"].flatten().numpy() for v in range(2)]
            removed = without_predictor(out, weights, keep, True, brain, lesion)
            row["without predictor direction"] = {key: list(value) for key, value in removed.items()}
            for v, view in enumerate(VIEWS):
                unit = weights[v] / np.linalg.norm(weights[v])
                row[f"{view} predictor share of displacement"] = float(
                    (displacement[view] @ unit) ** 2 / (displacement[view] @ displacement[view])
                )
        row["push"] = push(model, predictors, args, bank, lesion, displacement, device)
        results[name] = row
        print(name, f"({time.time() - started:.0f} s)", flush=True)
        print(json.dumps(row, indent=1, default=lambda v: round(float(v), 3)), flush=True)
        json.dump(results, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
