"""Stage 54: where in the image each direction of a frozen code reads, and the rotation of the code whose directions
read from the fewest places. No training.

Two views identify the shared anatomy only as a block (HANDOFF section 8): any invertible remix of the code holds it as
well. Locality is structure that can pick one remix. If each factor is computed from its own region of the image, the
directions that read single factors read from few places, and their mixtures from the union of those places (the
structural sparsity of Zheng et al. 2022, here in a linear form). This is step 3b of the factoriser's plan (items
35-36), never built until now.

Per run, on its trained code and on its untrained twin (calibrated as causal_bundle.py does), per view, in eval mode:

  1. Footprints, label-free. On --subjects training subjects (no jitter), each code unit's gradient with respect to
     both input channels, each voxel's gradient weighted by how far that subject's voxel lies from the population's
     mean image (--weight variation; --weight sensitivity keeps the plain gradient). Per cell of the 16^3 map (a 4^3
     block of voxels) and channel, the units x units Gram G_c = sum over the cell's voxels of J_v J_v', averaged over
     subjects. A direction w of the code then has the footprint f_w(c) = w' G_c w: how much of its subject-to-subject
     variation comes from cell c, to first order. With the weighting, voxels that never vary count for nothing; the
     plain gradient of a code pooled over the whole brain is nearly the same map for every direction. A footprint is
     read as a share of its total.
     - Effective cells: 1 / sum_c f(c)^2, the number of cells a flat footprint of the same spread would cover; per
       subject, the same for each subject's own footprint.
     - The code's mean footprint: the average over any orthonormal set of directions in the whitened frame (below),
       the same for every rotation. A direction's specific footprint is its share minus the mean's (where it reads
       more than the code as a whole), and KL is the divergence of its footprint from the mean's, in nats (0 for a
       direction that reads like the whole code).
  2. The footprint-sparse rotation. M = sum_c G_c is the code's total gradient energy. In the frame whitened by M every
     unit-length direction's footprint sums to 1, and for an orthonormal set of directions the summed sparsity
     sum_k sum_c f_k(c)^2 equals the diagonal energy of the rotated cell Grams. Maximising it is their approximate joint
     diagonalisation, solved by Jacobi angles (Cardoso & Souloumiac 1996) from the identity and from --restarts random
     starts. A footprint every direction shares becomes the identity in this frame and does not move the optimum. If
     the factors are read from disjoint voxels and the code is linear in them, the whitened Grams are exactly jointly
     diagonal and the rotation returns the factors. The whitening is by gradient energy, not by the code's covariance,
     so the components stay correlated across subjects and PC can find edges among them (stage 51's canonical
     directions were uncorrelated by construction).
     - pooled: the cell Grams averaged over subjects, so where a direction reads on average.
     - subject: each subject's own cell Grams, through their fourth moment summed over subjects, so a direction that
       reads one place in each subject but a different place in each (the drawn lesion) also counts as local.
  3. Scoring; labels only score. The targets are the seven causal factors and the lesion centre (x, y, z).
     - Names: each component is named after one target by one Hungarian match on |corr| over test subjects. MCC is the
       mean matched |r| of the seven factors. It is compared with the code's own units and with --rotations random
       rotations in the same whitened frame.
     - Readout footprints: a ridge readout of each target, fitted on training subjects, is a direction of the code.
       Its footprint is where the code reads that target.
     - Image maps, model-free: per cell, the mean over its voxels of the squared correlation, across training
       subjects, between the scan's intensity and each factor's own part (its SCM noise, so its parents' regions drop
       out) or each lesion coordinate. These show where the image shows the target.
     - A component's cosine with its target's readout footprint and each footprint's correlation over cells with its
       target's image map, both on specific footprints. Each footprint's share inside the brain interior (TailCode's
       cells, at least 90% brain), on its boundary, and outside (under 10% brain); and, with the residual input, the
       share that comes through the residual channel.

Limits. GroupNorm statistics reach every voxel, so every direction also takes a diffuse gradient from the whole
volume: read every footprint against the untrained twin's. Footprints cannot tell apart factors that are read from the
same place (burden and the lesion centre on the fixed bank). A code that holds a factor non-linearly mixes it
differently in each subject, which the joint diagonalisation can only approximate.

Writes OUT_JSON (runs already in it at the same step and settings are reused), the footprint maps (MAPS/<run>.npz,
MAPS defaulting to OUT_JSON's folder /footprints), figures (MAPS/<run>.<trained|twin>.view<1|2>.<components|targets>.png)
and, with --bundles DIR, each view's code and both rotations of it as bundles for label-free discovery
(DIR/<run>.fp_<raw|pooled|subject>[.floor].npz).

    python out/burden_feasibility_20261102/footprints.py OUT_JSON RUN [RUN ...] [--subjects 300] [--batch 8]
        [--weight variation|sensitivity] [--restarts 5] [--rotations 30] [--image-subjects 1000] [--maps DIR]
        [--bundles DIR] [--no-figures]
"""

import argparse
import json
from pathlib import Path

import bank as B
import batch_price
import causal_bundle
import code_audit
import factoriser
import numpy as np
import torch
import torch.nn.functional as F
import warm
from scipy.stats import ortho_group
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

COLUMNS = list(B.CAUSAL_FACTORS)
FACTOR_NAMES = [B.CAUSAL_FACTORS[d] for d in COLUMNS]
TARGETS = FACTOR_NAMES + ["lesion x", "lesion y", "lesion z"]
CELL = 4  # voxels per side of one cell of the 16^3 map
KINDS = ("pooled", "subject")
SETS = ("units",) + KINDS
ALPHAS = np.logspace(-3, 3, 13)
INTERIOR, OUTSIDE = 0.9, 0.1  # brain occupancy of a cell: interior at or above, outside below


def jacobian(code_fn, x, units):
    """(B, units, channels, D, H, W): each code unit's gradient with respect to the input, per subject. In eval mode
    each subject's code depends on its own input only, so the gradient of a unit's batch sum is per subject."""
    x = x.detach().requires_grad_(True)
    code = code_fn(x)[:, :units]
    rows = [torch.autograd.grad(code[:, k].sum(), x, retain_graph=k + 1 < units)[0] for k in range(units)]
    return torch.stack(rows, 1)


class Grams:
    """Running sums of the cell Grams of the code's input gradient: per channel and cell, averaged over subjects, and
    the fourth moment of each subject's own cell Grams (channels summed), averaged over subjects."""

    def __init__(self, units, channels, cells):
        self.pooled = np.zeros((channels, cells, units, units))
        self.moment = np.zeros((units * units, units * units))
        self.n = 0

    def add(self, jac):
        b, units, channels, size = jac.shape[0], jac.shape[1], jac.shape[2], jac.shape[-1]
        g = size // CELL
        blocks = jac.reshape(b, units, channels, g, CELL, g, CELL, g, CELL)
        blocks = blocks.permute(0, 2, 3, 5, 7, 1, 4, 6, 8).reshape(b, channels, g**3, units, CELL**3)
        gram = blocks @ blocks.transpose(-1, -2)
        self.pooled += gram.sum(0).cpu().double().numpy()
        own = gram.sum(1).reshape(b * g**3, units * units)
        self.moment += (own.T @ own).cpu().double().numpy()
        self.n += b

    def result(self):
        return self.pooled / self.n, self.moment / self.n


def whitener(total, floor=1e-8):
    """(k, K) rows that whiten the code by its total gradient energy; directions with none are dropped."""
    values, vectors = np.linalg.eigh((total + total.T) / 2)
    keep = values > floor * values.max()
    return (vectors[:, keep] / np.sqrt(values[keep])).T


def cell_tensor(grams, white):
    """(k, k, k, k): sum over cells of A_c (x) A_c, A_c the cells' (cells, K, K) Grams in the whitened frame."""
    a = np.einsum("id,cde,je->cij", white, grams, white)
    flat = a.reshape(len(a), -1)
    k = white.shape[0]
    return (flat.T @ flat).reshape(k, k, k, k)


def whitened_moment(moment, white):
    """The fourth moment of the per-subject cell Grams, (K^2, K^2), taken into the whitened frame as (k, k, k, k)."""
    both = np.kron(white, white)
    k = white.shape[0]
    return (both @ moment @ both.T).reshape(k, k, k, k)


def rotate_tensor(tensor, rotation):
    """The tensor in the basis given by ``rotation``'s columns, every mode alike."""
    for mode in range(4):
        tensor = np.moveaxis(np.tensordot(rotation, tensor, axes=([0], [mode])), 0, mode)
    return tensor


def diagonal_energy(tensor):
    return float(np.einsum("iiii->", tensor))


def jacobi(tensor, sweeps=100, tol=1e-10, gain=1e-7):
    """Jacobi angles on a (k, k, k, k) tensor sum_i A_i (x) A_i of symmetric A_i: the orthogonal basis (columns) that
    maximises sum_k T[k, k, k, k] = sum_i ||diag(R' A_i R)||^2, from the basis the tensor is given in. Each plane
    rotation is the exact optimum for its pair, so the objective never falls. Stops when no plane rotates by more than
    ``tol`` or a sweep raises the objective by less than ``gain`` of it."""
    t = np.array(tensor, dtype=np.float64)
    k = t.shape[0]
    basis = np.eye(k)
    energy = diagonal_energy(t)
    for _ in range(sweeps):
        largest = 0.0
        for p in range(k - 1):
            for q in range(p + 1, k):
                # (cos 2a, sin 2a) is the leading eigenvector of G = sum_i h_i h_i', h_i = (A_pp - A_qq, 2 A_pq)
                g00 = t[p, p, p, p] - 2 * t[p, p, q, q] + t[q, q, q, q]
                g01 = 2 * (t[p, p, p, q] - t[q, q, p, q])
                g11 = 4 * t[p, q, p, q]
                angle = 0.25 * np.arctan2(2 * g01, g00 - g11)
                c, s = np.cos(angle), np.sin(angle)
                if abs(s) < tol:
                    continue
                largest = max(largest, abs(s))
                for mode in range(4):
                    view = np.moveaxis(t, mode, 0)
                    tp, tq = view[p].copy(), view[q].copy()
                    view[p] = c * tp + s * tq
                    view[q] = c * tq - s * tp
                bp, bq = basis[:, p].copy(), basis[:, q].copy()
                basis[:, p], basis[:, q] = c * bp + s * bq, c * bq - s * bp
        previous, energy = energy, diagonal_energy(t)
        if largest < tol or energy - previous < gain * abs(energy):
            break
    return basis, energy


def joint_diagonalise(tensor, restarts=5, seed=0):
    """The best of Jacobi angles from the identity and from ``restarts`` random orthogonal starts: (basis, energy)."""
    k = tensor.shape[0]
    best = jacobi(tensor)
    for r in range(restarts if k > 1 else 0):
        start = ortho_group.rvs(k, random_state=seed * 1000 + r)
        basis, energy = jacobi(rotate_tensor(tensor, start))
        if energy > best[1]:
            best = (start @ basis, energy)
    return best


def footprints(directions, grams):
    """(n, cells): each direction's footprint w' G_c w, from (cells, K, K) Grams."""
    return np.einsum("nd,cde,ne->nc", directions, grams, directions)


def shares(footprint):
    return footprint / footprint.sum(-1, keepdims=True)


def effective_cells(footprint):
    return 1.0 / np.square(shares(footprint)).sum(-1)


def subject_cells(directions, total, moment):
    """Each direction's effective cells per subject: (w' M w)^2 / sum over subjects and cells of (w' G_sc w)^2 per
    subject. Equal to the pooled count when the footprint is the same in every subject, below it when it moves."""
    k = directions.shape[1]
    outer = np.einsum("nd,ne->nde", directions, directions).reshape(len(directions), k * k)
    energy = np.einsum("nd,de,ne->n", directions, total, directions)
    return np.square(energy) / np.einsum("ni,ij,nj->n", outer, moment, outer)


def mean_share(grams, white):
    """The code's mean footprint as shares: the average footprint over an orthonormal set of whitened directions,
    the same for every rotation."""
    return shares(np.einsum("kd,cde,ke->c", white, grams, white))


def divergence(footprint, mean):
    """KL of each footprint (as shares) from the code's mean footprint, in nats."""
    p = shares(footprint)
    ratio = np.divide(p, mean, out=np.ones_like(p), where=(p > 0) & (mean > 0))
    return (p * np.log(ratio)).sum(-1)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def correlation(a, b):
    if np.ptp(a) <= 0 or np.ptp(b) <= 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def oriented(directions):
    """Each row's sign set so that its largest weight is positive, a label-free convention."""
    rows = np.arange(len(directions))
    return directions * np.sign(directions[rows, np.abs(directions).argmax(1)])[:, None]


def rotations(grams, moment, restarts, seed=0):
    """The pooled and subject rotations of one view's code as (n, K) directions in raw code units, most local first,
    with the whitening and the energies."""
    summed = grams.sum(0)
    total = summed.sum(0)
    white = whitener(total)
    out = {"kept": int(white.shape[0]), "units": int(total.shape[0]), "white": white}
    for kind, tensor in (("pooled", cell_tensor(summed, white)), ("subject", whitened_moment(moment, white))):
        basis, energy = joint_diagonalise(tensor, restarts, seed)
        directions = oriented(basis.T @ white)
        local = (
            effective_cells(footprints(directions, summed))
            if kind == "pooled"
            else subject_cells(directions, total, moment)
        )
        out[kind] = directions[np.argsort(local)]
        out[f"{kind}_energy"] = energy
        out[f"{kind}_identity_energy"] = diagonal_energy(tensor)
    return out


def occupancy(root, subjects=300):
    """(cells,) mean brain occupancy of each cell over training subjects (FLAIR non-zero, as warm.interior_cells)."""
    sample = np.asarray(np.load(f"{root}/train_flair.npy", mmap_mode="r")[:subjects]) != 0
    cells = F.avg_pool3d(torch.from_numpy(sample.astype(np.float32))[:, None], CELL)[:, 0].mean(0)
    return cells.flatten().numpy().astype(np.float64)


def region_shares(footprint, occ):
    """(n, 3): each footprint's share inside the brain interior, on its boundary, and outside."""
    share = shares(footprint)
    inside, outside = occ >= INTERIOR, occ < OUTSIDE
    return np.stack([share[:, inside].sum(1), share[:, ~inside & ~outside].sum(1), share[:, outside].sum(1)], 1)


def lesion_centre(root, split, n):
    return batch_price.lesion_centroid(np.load(f"{root}/{split}_cells.npy", mmap_mode="r")[:n])


def image_maps(root, t1key, subjects, batch=32):
    """(2 views, 10 targets, cells): per cell, the mean over its voxels of the squared correlation across training
    subjects between the scan's intensity and each factor's own part, then each lesion coordinate."""
    bank = warm.Bank(root, "train", t1key, subjects)
    n = len(bank.vox)
    own = causal_bundle.own_noise(bank.zc, root, COLUMNS)
    y = np.column_stack([own, lesion_centre(root, "train", n)])
    sd = y.std(0)
    y = np.divide(y - y.mean(0), sd, out=np.zeros_like(y), where=sd > 0)
    maps = []
    for scans in (bank.t1, bank.flair):
        size = scans.shape[-1]
        total = np.zeros(size**3)
        square = np.zeros(size**3)
        cross = np.zeros((size**3, y.shape[1]))
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            x = np.asarray(scans[start:stop], dtype=np.float64).reshape(-1, size**3)
            total += x.sum(0)
            square += np.square(x).sum(0)
            cross += x.T @ y[start:stop]
        mean = total / n
        spread = np.sqrt(np.maximum(square / n - np.square(mean), 0))
        corr = np.divide(cross / n, spread[:, None], out=np.zeros_like(cross), where=spread[:, None] > 1e-12)
        volume = torch.from_numpy(np.square(corr).T.reshape(y.shape[1], 1, size, size, size))
        maps.append(F.avg_pool3d(volume, CELL).reshape(y.shape[1], -1).numpy())
    return np.stack(maps)


def mean_images(bank, subjects, batch=32):
    """(2 views, channels, D, H, W): the training subjects' mean input, both views, every channel the run reads."""
    n = min(subjects, len(bank.vox))
    total = None
    for start in range(0, n, batch):
        x = bank.images(np.arange(start, min(start + batch, n)), "cpu").double()
        b = len(x) // 2
        part = torch.stack([x[:b].sum(0), x[b:].sum(0)])
        total = part if total is None else total + part
    return (total / n).float()


def readouts(train_code, train_targets, test_code, test_targets):
    """Each target's ridge readout fitted on training subjects: (targets, K) directions in raw code units, and each
    readout's R2 on test subjects (None for a constant target)."""
    scaler = StandardScaler().fit(train_code)
    directions, r2 = [], []
    for j in range(train_targets.shape[1]):
        if np.ptp(train_targets[:, j]) <= 1e-12:
            directions.append(np.zeros(train_code.shape[1]))
            r2.append(None)
            continue
        model = RidgeCV(alphas=ALPHAS).fit(scaler.transform(train_code), train_targets[:, j])
        y = test_targets[:, j]
        pred = model.predict(scaler.transform(test_code))
        r2.append(float(1 - np.square(y - pred).sum() / max(np.square(y - y.mean()).sum(), 1e-300)))
        directions.append(model.coef_ / scaler.scale_)
    return np.array(directions), r2


def view_code(model, pool, view):
    """One view's exported content code as a function of that view's (B, channels, D, H, W) images."""

    def code(x):
        return pool.head(pool.pooled_view(model._encode(x, 1, view), view))[:, : pool.main]

    return code


def all_targets(bank, root, split):
    return np.column_stack([bank.zc[:, COLUMNS], lesion_centre(root, split, len(bank.vox))])


def score_set(directions, test_code, mean, targets, grams, total, moment, occ, maps, mean_fp):
    """One set of directions: names, MCC, locality, specificity, regions, channel shares and image-map agreement."""
    summed = grams.sum(0)
    components = (test_code - mean) @ directions.T
    names, strength = factoriser.name_components(components, targets)
    fp = footprints(directions, summed)
    specific = shares(fp) - mean_fp
    matched = {int(t): (int(c), float(strength[c])) for c, t in enumerate(names) if t >= 0}
    factor_r = [matched[j][1] if j in matched else 0.0 for j in range(len(FACTOR_NAMES))]
    out = {
        "names": [TARGETS[t] if t >= 0 else None for t in names],
        "matched_r": strength.tolist(),
        "mcc": float(np.mean(factor_r)),
        "kl": divergence(fp, mean_fp).tolist(),
        "pooled_cells": effective_cells(fp).tolist(),
        "subject_cells": subject_cells(directions, total, moment).tolist(),
        "regions": region_shares(fp, occ).tolist(),
        "image_r": [correlation(specific[c], maps[t]) if t >= 0 else None for c, t in enumerate(names)],
        "mean_abs_corr": (
            float(np.abs(np.corrcoef(components.T)[np.triu_indices(len(directions), 1)]).mean())
            if len(directions) > 1
            else 0.0
        ),
    }
    if grams.shape[0] > 1:
        out["residual_share"] = (footprints(directions, grams[1]).sum(1) / fp.sum(1)).tolist()
    return out, fp, specific, matched


def assess_view(grams, moment, train_code, test_code, train_targets, test_targets, occ, maps, a, seed=0):
    """Rotations, footprints and scores for one view of one model."""
    summed, total = grams.sum(0), grams.sum((0, 1))
    rot = rotations(grams, moment, a.restarts, seed)
    white = rot["white"]
    mean_fp = mean_share(summed, white)
    mean = train_code.mean(0)
    readout_dirs, readout_r2 = readouts(train_code, train_targets, test_code, test_targets)
    readout_fp = footprints(readout_dirs, summed)
    active = [j for j in range(len(TARGETS)) if readout_r2[j] is not None]
    readout_specific = np.zeros_like(readout_fp)
    readout_specific[active] = shares(readout_fp[active]) - mean_fp
    out = {
        "kept": rot["kept"],
        "units": rot["units"],
        "readout_r2": readout_r2,
        "mean_cells": float(1 / np.square(mean_fp).sum()),
        "image_cells": effective_cells(maps).tolist(),
        "sets": {},
        "readouts": {
            "kl": [
                float(divergence(readout_fp[[j]], mean_fp)[0]) if j in active else None for j in range(len(TARGETS))
            ],
            "regions": [
                region_shares(readout_fp[[j]], occ)[0].tolist() if j in active else None for j in range(len(TARGETS))
            ],
            "image_r": [
                correlation(readout_specific[j], maps[j]) if j in active else None for j in range(len(TARGETS))
            ],
        },
    }
    for kind in KINDS:
        out[f"{kind}_energy"] = rot[f"{kind}_energy"]
        out[f"{kind}_identity_energy"] = rot[f"{kind}_identity_energy"]
    arrays = {"readouts": readout_fp, "readout_specific": readout_specific, "mean_share": mean_fp}
    directions = {"units": np.eye(train_code.shape[1]), "pooled": rot["pooled"], "subject": rot["subject"]}
    for kind, dirs in directions.items():
        row, fp, specific, matched = score_set(
            dirs, test_code, mean, test_targets, grams, total, moment, occ, maps, mean_fp
        )
        row["readout_cos"] = [
            cosine(specific[matched[j][0]], readout_specific[j]) if j in matched and j in active else None
            for j in range(len(TARGETS))
        ]
        row["by_target"] = [matched[j][0] if j in matched else None for j in range(len(TARGETS))]
        out["sets"][kind] = row
        arrays[kind] = fp
        arrays[f"{kind}_specific"] = specific
        arrays[f"{kind}_directions"] = dirs
    k = white.shape[0]
    null = {"mcc": [], "kl": [], "pooled_cells": [], "subject_cells": []}
    for r in range(a.rotations if k > 1 else 0):
        dirs = ortho_group.rvs(k, random_state=seed * 1000 + 500 + r).T @ white
        names, strength = factoriser.name_components((test_code - mean) @ dirs.T, test_targets)
        factor_r = [0.0] * len(FACTOR_NAMES)
        for c, t in enumerate(names):
            if 0 <= t < len(FACTOR_NAMES):
                factor_r[t] = float(strength[c])
        fp = footprints(dirs, summed)
        null["mcc"].append(float(np.mean(factor_r)))
        null["kl"].append(float(divergence(fp, mean_fp).mean()))
        null["pooled_cells"].append(float(effective_cells(fp).mean()))
        null["subject_cells"].append(float(subject_cells(dirs, total, moment).mean()))
    out["random"] = {key: [float(np.mean(v)), float(np.std(v))] if v else None for key, v in null.items()}
    return out, arrays


def assess_model(run, device, a, trained, maps, occ, means_cache):
    report, model, pool, residual, step = code_audit.load_run(run, device, trained=trained)
    args = report["args"]
    if pool is None:
        raise SystemExit(f"{run.name}: the code comes from TailCode's head, so the run needs --pool tail or gapnorm")
    root, t1key = args["bank"], args["t1key"]
    train = warm.Bank(root, "train", t1key, None, residual)
    if not trained:
        causal_bundle.calibrate(model, pool, train, device)
    test = warm.Bank(root, "test", t1key, int(args.get("test_subjects", 1000)), residual)
    units = pool.main
    centre = None
    if a.weight == "variation":
        key = (root, t1key, residual)
        if key not in means_cache:
            means_cache[key] = mean_images(train, a.image_subjects)
        centre = means_cache[key].to(device)
    chosen = np.arange(min(a.subjects, len(train.vox)))
    grams = None
    for start in range(0, len(chosen), a.batch):
        x = train.images(chosen[start : start + a.batch], device)
        b = len(x) // 2
        if grams is None:
            grams = [Grams(units, x.shape[1], (x.shape[-1] // CELL) ** 3) for _ in range(2)]
        for view in range(2):
            part = x[view * b : (view + 1) * b]
            jac = jacobian(view_code(model, pool, view), part, units).detach()
            if centre is not None:
                jac = jac * (part - centre[view])[:, None]
            grams[view].add(jac)
    codes = {
        split: [c[:, :units].astype(np.float64) for c in warm.features(model, bank, device, pool=pool)["code"]]
        for split, bank in (("train", train), ("test", test))
    }
    train_targets, test_targets = all_targets(train, root, "train"), all_targets(test, root, "test")
    views, arrays = [], {}
    for view in range(2):
        g, m = grams[view].result()
        row, view_arrays = assess_view(
            g, m, codes["train"][view], codes["test"][view], train_targets, test_targets, occ, maps[view], a
        )
        views.append(row)
        for key, value in view_arrays.items():
            arrays[f"view{view + 1}_{key}"] = value
        arrays[f"view{view + 1}_test_components"] = np.stack(
            [
                (codes["test"][view] - codes["train"][view].mean(0)) @ view_arrays[f"{kind}_directions"].T
                for kind in KINDS
            ]
        )
        arrays[f"view{view + 1}_test_code"] = codes["test"][view]
    out = {"step": step, "subjects": int(len(chosen)), "residual": residual, "views": views}
    return out, arrays, test, root, t1key


def settings(a):
    return {
        "subjects": a.subjects,
        "weight": a.weight,
        "restarts": a.restarts,
        "rotations": a.rotations,
        "image_subjects": a.image_subjects,
    }


def write_bundles(out_dir, name, arrays, test, root, t1key, floor, step):
    suffix = ".floor" if floor else ""
    for index, kind in enumerate(("raw",) + KINDS):
        embeddings = [
            arrays[f"view{v}_test_code"] if kind == "raw" else arrays[f"view{v}_test_components"][index - 1]
            for v in (1, 2)
        ]
        extra = {"source_run": name, "step": step, "untrained": floor, "features": f"code, footprint {kind}"}
        causal_bundle.save(out_dir / f"{name}.fp_{kind}{suffix}.npz", test, root, t1key, embeddings, extra)


def figures(path, name, row, arrays, maps, occ, model):
    """Two PNGs per view. Components: the code's mean footprint, then every component of the pooled rotation as its
    specific footprint (red where it reads more than the code as a whole, blue where less). Targets: per target, its
    image map, its readout's specific footprint and the specific footprint of the pooled component named after it.
    Image maps and the mean footprint are maxima over one axis; specific footprints, signed, are means over it. The
    cyan line is the brain's outline (cells that are half brain)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = round(len(occ) ** (1 / 3))
    outline = [occ.reshape((grid,) * 3).max(axis) for axis in range(3)]

    def panel(ax, values, signed, label=None):
        volume = values.reshape((grid,) * 3)
        images = [volume.mean(axis) if signed else volume.max(axis) for axis in range(3)]
        limit = max(float(np.abs(np.stack(images)).max()), 1e-30)
        for axis, (image, shape) in enumerate(zip(images, outline)):
            if signed:
                ax[axis].imshow(image.T, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
            else:
                ax[axis].imshow(image.T, origin="lower", cmap="magma")
            ax[axis].contour(shape.T, levels=[0.5], colors="cyan", linewidths=0.6)
            ax[axis].set_xticks([])
            ax[axis].set_yticks([])
        if label:
            ax[0].set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")

    for view in (1, 2):
        stats = row["views"][view - 1]["sets"]["pooled"]
        specific = arrays[f"view{view}_pooled_specific"]
        fig, axes = plt.subplots(len(specific) + 1, 3, figsize=(4.8, 1.4 * (len(specific) + 1)), squeeze=False)
        panel(axes[0], arrays[f"view{view}_mean_share"], False, "code's mean\nfootprint")
        for c in range(len(specific)):
            label = stats["names"][c] or "unmatched"
            panel(
                axes[c + 1],
                specific[c],
                True,
                f"c{c + 1} {label} |r| {stats['matched_r'][c]:.2f}\nKL {stats['kl'][c]:.2f}, "
                f"{stats['pooled_cells'][c]:.0f} cells",
            )
        for axis, title in enumerate(("over x", "over y", "over z")):
            axes[0][axis].set_title(title, fontsize=7)
        fig.suptitle(f"{name} ({model}), view {view}: pooled rotation", fontsize=8)
        fig.tight_layout()
        fig.savefig(path / f"{name}.{model}.view{view}.components.png", dpi=110)
        plt.close(fig)

        readout = arrays[f"view{view}_readout_specific"]
        fig, axes = plt.subplots(len(TARGETS), 9, figsize=(12, 1.4 * len(TARGETS)), squeeze=False)
        for j, target in enumerate(TARGETS):
            panel(axes[j][0:3], maps[view - 1][j], False, target)
            panel(axes[j][3:6], readout[j], True)
            c = stats["by_target"][j]
            if c is not None:
                panel(axes[j][6:9], specific[c], True)
            else:
                for ax in axes[j][6:9]:
                    ax.axis("off")
        for block, title in enumerate(("image map", "readout, specific footprint", "named pooled component")):
            axes[0][3 * block + 1].set_title(title, fontsize=8)
        fig.suptitle(f"{name} ({model}), view {view}", fontsize=8)
        fig.tight_layout()
        fig.savefig(path / f"{name}.{model}.view{view}.targets.png", dpi=110)
        plt.close(fig)


def f2(v):
    if v is None:
        return "   -"
    text = f"{v:.2f}"
    return text[1:] if text.startswith("0.") else text.replace("-0.", "-.")


def mean_of(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def show(name, row):
    t0 = row["trained"]
    print(
        f"{name}  step {t0['step']}, Jacobians on {t0['subjects']} training subjects, footprints weighted by "
        f"{row['settings']['weight']}"
    )
    for v, view in enumerate(("T1", "FLAIR")):
        t, u = row["trained"]["views"][v], row["twin"]["views"][v]
        print(
            f"  {view}: {t['kept']} of {t['units']} directions carry gradient (twin {u['kept']}); the code's mean "
            f"footprint covers {t['mean_cells']:.0f} cells (twin {u['mean_cells']:.0f})"
        )
        print(
            f"    {'':16s} {'MCC':>5s} {'KL':>5s} {'cells':>6s} {'per subj':>8s} {'interior':>8s} {'outside':>7s} "
            f"{'resid':>5s} {'|corr|':>6s}   twin: {'MCC':>5s} {'KL':>5s} {'cells':>6s}"
        )
        for kind in SETS:
            a, b = t["sets"][kind], u["sets"][kind]
            resid = f2(mean_of(a["residual_share"])) if "residual_share" in a else "   -"
            print(
                f"    {kind + (' rotation' if kind != 'units' else ''):16s} {f2(a['mcc']):>5s} "
                f"{f2(mean_of(a['kl'])):>5s} {mean_of(a['pooled_cells']):6.0f} {mean_of(a['subject_cells']):8.0f} "
                f"{f2(mean_of([r[0] for r in a['regions']])):>8s} {f2(mean_of([r[2] for r in a['regions']])):>7s} "
                f"{resid:>5s} {f2(a['mean_abs_corr']):>6s}         {f2(b['mcc']):>5s} {f2(mean_of(b['kl'])):>5s} "
                f"{mean_of(b['pooled_cells']):6.0f}"
            )
        rt, ru = t["random"], u["random"]
        if rt["mcc"] is not None:
            twin = (
                f"         {f2(ru['mcc'][0]):>5s} {f2(ru['kl'][0]):>5s} {ru['pooled_cells'][0]:6.0f}"
                if ru["mcc"]
                else ""
            )
            print(
                f"    {'random rotations':16s} {f2(rt['mcc'][0]):>5s} {f2(rt['kl'][0]):>5s} "
                f"{rt['pooled_cells'][0]:6.0f} {rt['subject_cells'][0]:8.0f}   MCC sd {rt['mcc'][1]:.3f}{twin}"
            )
        print(
            "    per target: R2 | image map cells | readout: KL, r(image) | pooled: |r|, KL, cos(readout), r(image) | "
            "subject: |r|, cos(readout) | twin pooled |r|"
        )
        for j, target in enumerate(TARGETS):
            cells = []
            for kind in KINDS:
                s = t["sets"][kind]
                c = s["by_target"][j]
                cells.append(
                    (
                        s["matched_r"][c] if c is not None else None,
                        s["kl"][c] if c is not None else None,
                        s["readout_cos"][j],
                        s["image_r"][c] if c is not None else None,
                    )
                )
            tw = u["sets"]["pooled"]
            c = tw["by_target"][j]
            print(
                f"    {target:10s} {f2(t['readout_r2'][j]):>5s} | {t['image_cells'][j]:5.0f} | "
                f"{f2(t['readouts']['kl'][j]):>5s} {f2(t['readouts']['image_r'][j]):>5s} | "
                f"{f2(cells[0][0]):>5s} {f2(cells[0][1]):>5s} {f2(cells[0][2]):>5s} {f2(cells[0][3]):>5s} | "
                f"{f2(cells[1][0]):>5s} {f2(cells[1][2]):>5s} | {f2(tw['matched_r'][c] if c is not None else None):>5s}"
            )
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--subjects", type=int, default=300, help="Training subjects whose Jacobians are taken")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument(
        "--weight",
        choices=("variation", "sensitivity"),
        default="variation",
        help="Weight each voxel's gradient by the subject's deviation from the mean image, or not",
    )
    p.add_argument("--restarts", type=int, default=5, help="Random starts of the joint diagonalisation")
    p.add_argument("--rotations", type=int, default=30, help="Random rotations in the null")
    p.add_argument("--image-subjects", type=int, default=1000, help="Training subjects for image maps and mean images")
    p.add_argument("--maps", type=Path, default=None, help="Footprint maps and figures (default OUT's folder)")
    p.add_argument("--bundles", type=Path, default=None, help="Also write bundles of the codes and rotations here")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--device", default=warm.pick_device())
    a = p.parse_args()
    maps_dir = a.maps or a.out.parent / "footprints"
    maps_dir.mkdir(parents=True, exist_ok=True)
    if a.bundles:
        a.bundles.mkdir(parents=True, exist_ok=True)
    results = json.loads(a.out.read_text()) if a.out.exists() else {}
    image_cache, occ_cache, means_cache = {}, {}, {}
    for run in a.runs:
        name = run.name
        step = int(torch.load(run / "model.pt", map_location="cpu")["step"])
        report = json.loads((run / "report.json").read_text())["args"]
        root, t1key = report["bank"], report["t1key"]
        npz = maps_dir / f"{name}.npz"
        wanted = []
        if a.bundles:
            wanted = [a.bundles / f"{name}.fp_{k}{f}.npz" for k in ("raw",) + KINDS for f in ("", ".floor")]
        old = results.get(name)
        if old and old["trained"]["step"] == step and old["settings"] == settings(a) and npz.exists():
            if all(path.exists() for path in wanted):
                print(f"reuse {name} (step {step})", flush=True)
                continue
        key = (root, t1key)
        if key not in image_cache:
            print(f"image maps for {root}, {t1key}", flush=True)
            image_cache[key] = image_maps(root, t1key, a.image_subjects)
            occ_cache[root] = occupancy(root)
        maps, occ = image_cache[key], occ_cache[root]
        row, arrays = {"settings": settings(a)}, {}
        for trained, model in ((True, "trained"), (False, "twin")):
            print(f"{name}: {model}", flush=True)
            out, model_arrays, test, root, t1key = assess_model(run, a.device, a, trained, maps, occ, means_cache)
            row[model] = out
            arrays.update({f"{model}_{k}": v for k, v in model_arrays.items()})
            if a.bundles:
                write_bundles(a.bundles, name, model_arrays, test, root, t1key, not trained, out["step"])
            if not a.no_figures:
                figures(maps_dir, name, out, model_arrays, maps, occ, model)
        np.savez_compressed(npz, image_maps=maps, occupancy=occ, targets=np.array(TARGETS), **arrays)
        results[name] = row
        a.out.write_text(json.dumps(results, indent=1))
    for name, row in results.items():
        show(name, row)


if __name__ == "__main__":
    main()
