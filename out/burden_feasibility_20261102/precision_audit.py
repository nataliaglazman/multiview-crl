"""Where a warm.py run loses the lesion's volume, in voxels, and whether its loss would pay for a sharper reading.

The account being tested: burden's own part (the volume the other factors do not set) sits near .3 in the backbone
because a small target is read with a large error. On the causal bank the other factors explain ~.58 of the volume's
variance, leaving an own part of ~20 voxels (SD), and the network's volume reading has ~20-30 voxels of error, about its
precision on the independent bank, while the image allows far better (the volume is an exact function of the burden
latent).

Target: each test subject's lesion volume minus the volume expected from every other true latent (an additive spline
ridge fitted on the bank's training subjects). On the causal bank that is what burden's own variation does to the
volume; on the independent bank it is nearly the whole volume.

Stages:

  image    detector.py's fixed blob filter on the bank's stored normative residual: what the image allows
  map      the backbone's 16^3 map, position-free: per channel, the mean of the 1, 3 and 10 most extreme interior cells
           of TailCode's normative z, in each direction
  gap      the backbone's global average
  pooled   TailCode's gauged GAP + tails, the head's input
  code     the exported code

Each stage is read by label-trained readouts (ridge and gradient boosting, 5-fold on the test subjects; the labels are
this diagnostic's only), two ways:

  own R2   from the stage's features plus the expected volume, so the readout is handed the anatomy part and only the
           lesion reading is tested. The reading error e = SD_own * sqrt(1/R2 - 1) is the error of a volume reading
           that leaves that R2, so own R2 = SD^2 / (SD^2 + e^2). Down the stages this is the precision ladder.
  alone    from the stage's features only: what the representation itself holds (own_variation.py's question, in
           voxels). It falls below own R2 when the stage lacks the anatomy that sets the expected volume.

Also: the raw-volume ridge R2 (the older notes' number); e within each third of the volume the other latents predict,
a lesion-size split that does not select on the own part; the untrained twin's e (the floor); and how the two views'
readings agree: the correlation of their readings, and of their errors once the true own part is removed. A shared
error looks like content to a cross-view loss.

Price: the drop in the run's loss (warm.hard_negative_infonce at concentration 0, the recipe's InfoNCE) when one more
unit, the own part read with error e independently in each view, joins the test subjects' code. e = 0, 5, 10, 20, 30
voxels, in uniform 32-subject sets and in sets of an anchor and its 31 nearest on the code (both views), as code
batches build them. If going from the run's own e to ~10 voxels buys little in both, the loss does not pay for
precision. One unit is diluted by the code's width, so compare rewards within a run.

The price is taken on four codes: as trained, and with what a smooth function of the lesion's position, of the six
anatomy factors, or of both predicts of each unit removed (out of fold; the labels are used only here). InfoNCE stops
paying once a batch is told apart, after about log2(batch) bits (Chen et al. 2021, "Intriguing properties of
contrastive losses", appendix B.2). So if sharpening earns much more once position is removed, the lesion's position is
what leaves burden's precision unpaid; if removing anatomy does as much, any easy content does. The R2 of position and
anatomy left in each code checks the removal.

    python out/burden_feasibility_20261102/precision_audit.py OUT_JSON RUN [RUN ...] [--jitter-draws K] [--split train]

Two checks of the network's reading, each into its own OUT_JSON, with no floor and no price (precision_compare.py puts
them beside the default audit):

  --jitter-draws K  the run's own training jitter moves the test scans before the network, and each feature is the mean
                    over K independent moves. With K = 1, a reading sharper than on raw scans means the network reads
                    the raw scans it never trained on worse. A larger K averages over where the lesion falls on the
                    backbone's 4-voxel grid, so a reading that sharpens with K depends on that pose. The image stage
                    stays on raw scans.
  --split train     the first training subjects (as many as the run's test subjects) in place of test subjects: a
                    reading much sharper there than on test subjects is memorised.

Runs already in OUT_JSON at the same step and condition are reused; image features are cached beside OUT_JSON per bank.
"""

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import batch_price
import causal_bundle
import code_audit
import detector
import numpy as np
import torch
import warm
from scipy.ndimage import binary_erosion
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, SplineTransformer, StandardScaler

import data.cluster_batches as cb

BURDEN = 2  # z_content index of the lesion's burden latent (--synthetic-lesion-target burden)
STAGES = ("image", "map", "gap", "pooled", "code")
VIEWS = ("t1", "flair")
TOPK = (1, 3, 10)
CLIP = 5.0  # standardized features are clipped at this many SD, as in own_variation.py
ERRORS = (0.0, 5.0, 10.0, 20.0, 30.0)
VARIANTS = ("as trained", "position removed", "anatomy removed", "both removed")
PRICE_SETS = 300
SIM = torch.nn.CosineSimilarity(dim=-1)
WORKERS = 6


def clipped(x):
    return np.clip(x, -CLIP, CLIP)


def predictions(x, y, kind):
    """Out-of-fold predictions of y from x, 5-fold on these rows."""
    if kind == "ridge":
        model = make_pipeline(StandardScaler(), FunctionTransformer(clipped), RidgeCV(alphas=np.logspace(-2, 5, 15)))
    else:
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_val_predict(model, x, y, cv=KFold(5, shuffle=True, random_state=0))


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def reading_error(value, sd):
    """The volume-reading error that leaves own R2 ``value`` on a target of SD ``sd``; None when nothing is read."""
    return float(sd * np.sqrt(1 / value - 1)) if value > 0 else None


def partial_r(a, b, given):
    """Correlation of ``a`` and ``b`` once a linear fit on ``given`` is removed from each."""
    design = np.column_stack([np.ones_like(given), given])
    ra, rb = (v - design @ np.linalg.lstsq(design, v, rcond=None)[0] for v in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1])


def own_volume(root, vox, split="test"):
    """For the ``split`` subjects' volume ``vox``: the volume every other true latent predicts (an additive spline ridge
    fitted on the training subjects), the share of the variance it explains, and the burden latent's rank correlation
    with the volume."""
    train = np.load(f"{root}/train_zc.npy").astype(np.float64)
    test = np.load(f"{root}/{split}_zc.npy").astype(np.float64)[: len(vox)]
    others = [j for j in range(train.shape[1]) if j != BURDEN]
    model = make_pipeline(
        StandardScaler(), SplineTransformer(n_knots=8, degree=3), RidgeCV(alphas=np.logspace(-4, 3, 15))
    )
    model.fit(train[:, others], np.load(f"{root}/train_vox.npy").astype(np.float64))
    expected = model.predict(test[:, others])
    return expected, r2(expected, vox), float(spearmanr(test[:, BURDEN], vox).statistic)


def image_job(task):
    source, start, stop, interior = task
    images = np.load(source, mmap_mode="r")
    rows = [detector.features_of(np.asarray(images[i], dtype=np.float32), interior) for i in range(start, stop)]
    return start, np.array(rows, dtype=np.float32)


def image_features(root, key, n, cache, split="test"):
    """detector.py's fixed blob features of the first ``n`` ``split`` subjects' stored normative residual, read over the
    interior as residual.py reads them; None when the bank has no residual for ``key``."""
    source = Path(f"{root}/{split}_{key}_res.npy")
    if not source.exists():
        return None
    name = Path(root).name if split == "test" else f"{Path(root).name}_{split}"
    path = cache / f"image_{name}_{key}_{n}.npy"
    if path.exists():
        return np.load(path)
    sample = np.asarray(np.load(f"{root}/train_flair.npy", mmap_mode="r")[:200], dtype=np.float32)
    interior = binary_erosion((sample != 0).mean(0) >= 0.9, iterations=2)
    tasks = [(str(source), s, min(s + 50, n), interior) for s in range(0, n, 50)]
    with ProcessPoolExecutor(WORKERS) as pool:
        rows = sorted(pool.map(image_job, tasks), key=lambda r: r[0])
    feats = np.concatenate([r[1] for r in rows])
    np.save(path, feats)
    return feats


def view_features(part, view, pool):
    """One view's position-free map summary, GAP, pooled features and code from its backbone map ``part``."""
    z = (part.flatten(2)[:, :, pool.keep] - pool.mean[view]) / (pool.var[view].sqrt() + 1e-6)
    summary = []
    for sign in (1.0, -1.0):
        top = (sign * z).topk(max(TOPK), dim=2).values
        summary += [top[:, :, :k].mean(2) for k in TOPK]
    pooled = pool.pooled_view(part, view)
    return {
        "map": torch.cat(summary, 1),
        "gap": part.mean((2, 3, 4)),
        "pooled": pooled,
        "code": pool.head(pooled)[:, : pool.units],
    }


@torch.no_grad()
def network_features(model, pool, bank, device, batch=32, jitter=None, draws=1, seed=0):
    """Per view: the position-free map summary, GAP, pooled features and code of every subject in ``bank``. With
    ``jitter`` = (shift, degrees, shared), the run's training jitter moves the scans first, and each feature is the mean
    over ``draws`` independent moves."""
    rng = np.random.default_rng(seed)
    moves = draws if jitter else 1
    rows = {k: [[], []] for k in STAGES[1:]}
    for start in range(0, len(bank.vox), batch):
        x = bank.images(np.arange(start, min(start + batch, len(bank.vox))), device)
        total = [None, None]
        for _ in range(moves):
            h = model._encode(warm.jitter(x, jitter[0], jitter[1], rng, jitter[2]) if jitter else x, 2, None)
            for view, part in enumerate(h.chunk(2)):
                out = view_features(part, view, pool)
                total[view] = out if total[view] is None else {k: total[view][k] + v for k, v in out.items()}
        for view in range(2):
            for k, v in total[view].items():
                rows[k][view].append((v / moves).float().cpu().numpy())
    return {k: [np.concatenate(v[i]) for i in range(2)] for k, v in rows.items()}


def assess_stage(x, vox, expected, full=True):
    """One stage in one view: the precision readout (features plus the expected volume) and, when ``full``, the readout
    from the features alone, the raw volume and the error by size third. Returns the scores and the precision readout's
    out-of-fold predictions."""
    own = vox - expected
    given = np.column_stack([x, expected])
    best, best_kind = None, None
    for kind in ("ridge", "boosted"):
        pred = predictions(given, own, kind)
        if best is None or r2(pred, own) > r2(best, own):
            best, best_kind = pred, kind
    out = {"own_r2": r2(best, own), "readout": best_kind}
    out["error"] = reading_error(out["own_r2"], float(own.std()))
    if full:
        out["alone_r2"] = r2(predictions(x, own, best_kind), own)
        out["raw_r2"] = r2(predictions(x, vox, "ridge"), vox)
        # Thirds of the volume the other latents predict: choosing subjects by their volume itself would tie own to the
        # expected volume inside a third. Scored by squared correlation, as the readout's shrinkage is global.
        thirds = np.array_split(np.argsort(expected, kind="stable"), 3)
        out["error_by_size"] = [
            reading_error(float(np.corrcoef(best[t], own[t])[0, 1] ** 2), float(own[t].std())) for t in thirds
        ]
        out["size_bounds"] = [float(expected[t].min()) for t in thirds[1:]]
        out["size_means"] = [float(vox[t].mean()) for t in thirds]
    return out, best


def mean_loss(z, sets):
    units = z[0].shape[1]
    return float(
        np.mean(
            [
                float(warm.hard_negative_infonce(torch.cat([z[0][s], z[1][s]]), units, batch_price.TAU, 0.0, SIM))
                for s in sets
            ]
        )
    )


def without(code, covariates):
    """``code`` minus what a smooth function of ``covariates`` (additive splines plus a cubic polynomial, ridge)
    predicts of each unit, out of fold, so that information goes and the rest stays."""
    model = make_pipeline(
        StandardScaler(),
        FeatureUnion(
            [("spline", SplineTransformer(n_knots=5, degree=3)), ("poly", PolynomialFeatures(3, include_bias=False))]
        ),
        RidgeCV(alphas=np.logspace(-3, 3, 13)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return code - cross_val_predict(model, covariates, code, cv=KFold(5, shuffle=True, random_state=0))


def price(codes, own, position, anatomy, seed=0):
    """The drop in the run's loss when the own part, read with error e independently in each view, joins the code: as
    trained, and with the lesion's position, the anatomy or both removed. Also the mean ridge R2 of position and of
    anatomy each code still holds, per view. Every code sees the same uniform sets, anchors and reading noise."""
    rng = np.random.default_rng(seed)
    n = len(own)
    noise = {e: [rng.standard_normal(n) for _ in range(2)] for e in ERRORS}
    uniform = [rng.choice(n, 32, replace=False) for _ in range(PRICE_SETS)]
    anchors = rng.choice(n, PRICE_SETS)
    removed = {
        "position removed": position,
        "anatomy removed": anatomy,
        "both removed": np.column_stack([position, anatomy]),
    }
    out = {}
    for variant in VARIANTS:
        code = codes if variant == "as trained" else [without(c, removed[variant]) for c in codes]
        row = {
            "left": {
                kind: [float(np.mean([r2(predictions(c, y, "ridge"), y) for y in target.T])) for c in code]
                for kind, target in (("position", position), ("anatomy", anatomy))
            }
        }
        neighbours = cb.nearest_neighbours(np.concatenate(code, 1), 31)
        z = [torch.from_numpy(c).float() for c in code]
        for rule, chosen in (("uniform", uniform), ("neighbour", [np.r_[i, neighbours[i]] for i in anchors])):
            base = mean_loss(z, chosen)
            rewards = []
            for e in ERRORS:
                extra = [torch.from_numpy(own + e * noise[e][v]).float()[:, None] for v in range(2)]
                rewards.append(base - mean_loss([torch.cat([z[v], extra[v]], 1) for v in range(2)], chosen))
            row[rule] = {"loss": base, "reward": rewards}
        out[variant] = row
    return out


def reprice(run, device):
    """The price table alone, for a run assessed before the price had its position- and anatomy-removed codes."""
    report, model, pool, residual, _ = code_audit.load_run(run, device, trained=True)
    a = report["args"]
    test = warm.Bank(a["bank"], "test", a["t1key"], int(a.get("test_subjects", 1000)), residual)
    expected = own_volume(a["bank"], test.vox)[0]
    position = batch_price.lesion_centroid(np.load(f"{a['bank']}/test_cells.npy", mmap_mode="r")[: len(test.vox)])
    codes = network_features(model, pool, test, device)["code"]
    return price(codes, test.vox - expected, position, test.zc[:, warm.ANATOMY])


def assess(run, device, cache, split="test", draws=0):
    checks = split != "test" or draws > 0
    out = {"errors": list(ERRORS), "split": split, "jitter_draws": draws}
    for trained in (True,) if checks else (True, False):
        report, model, pool, residual, step = code_audit.load_run(run, device, trained=trained)
        a = report["args"]
        root = a["bank"]
        if pool is None:
            raise ValueError("the map statistics come from TailCode, so the run needs --pool tail or gapnorm")
        jitter = None
        if draws:
            jitter = (float(a.get("jitter", 0)), float(a.get("rotate", 0)), a.get("shared_jitter") == "True")
            if not (jitter[0] or jitter[1]):
                raise ValueError("it trained without jitter, so --jitter-draws has nothing to match")
        if not trained:
            causal_bundle.calibrate(model, pool, warm.Bank(root, "train", a["t1key"], residual=residual), device)
        subjects = warm.Bank(root, split, a["t1key"], int(a.get("test_subjects", 1000)), residual)
        vox = subjects.vox
        if trained:
            expected, explained, spearman = own_volume(root, vox, split)
            out.update(
                step=step,
                bank=root,
                rows=len(vox),
                vox_sd=float(vox.std()),
                own_sd=float((vox - expected).std()),
                explained=explained,
                spearman=spearman,
            )
        feats = network_features(model, pool, subjects, device, jitter=jitter, draws=draws)
        if trained:
            feats["image"] = [image_features(root, key, len(vox), cache, split) for key in (a["t1key"], "flair")]
        scores, best = {}, {}
        for v, view in enumerate(VIEWS):
            scores[view] = {}
            for stage in STAGES:
                if stage in feats and feats[stage][v] is not None:
                    scores[view][stage], best[(stage, view)] = assess_stage(feats[stage][v], vox, expected, trained)
        out["trained" if trained else "untrained"] = scores
        if trained:
            own = vox - expected
            out["cross_view"] = {
                stage: {
                    "reading_r": float(np.corrcoef(best[(stage, "t1")], best[(stage, "flair")])[0, 1]),
                    "shared_error_r": partial_r(best[(stage, "t1")], best[(stage, "flair")], own),
                }
                for stage in STAGES
                if (stage, "t1") in best and (stage, "flair") in best
            }
            if not checks:
                position = batch_price.lesion_centroid(np.load(f"{root}/test_cells.npy", mmap_mode="r")[: len(vox)])
                out["price"] = price(feats["code"], own, position, subjects.zc[:, warm.ANATOMY])
    return out


def voxels(value, width=6):
    if value is None:
        return f"{'-':>{width}s}"
    return f"{'>99':>{width}s}" if value > 99 else f"{value:{width}.1f}"


def show(name, row):
    print(
        f"{name}  step {row['step']}, {row['rows']} {row.get('split', 'test')} subjects, bank {Path(row['bank']).name}"
    )
    if row.get("jitter_draws"):
        print(
            f"  scans moved by the run's training jitter, each feature the mean over {row['jitter_draws']} moves"
            " (the image stage reads raw scans)"
        )
    print(
        f"  lesion volume SD {row['vox_sd']:.1f} vox; the other true latents explain {row['explained']:.2f} of its"
        f" variance, leaving an own part of SD {row['own_sd']:.1f} vox (burden latent vs volume: Spearman"
        f" {row['spearman']:.3f})"
    )
    print(
        "  e: volume-reading error in voxels, from own R2 (features + the volume the other latents predict), own R2 ="
        " SD^2 / (SD^2 + e^2)\n  alone: own R2 from the features only; floor e: the untrained twin's; by size: e in the"
        " smallest / middle / largest third of lesions"
    )
    print(
        f"    {'stage':7s} {'view':5s} {'e':>6s} {'own R2':>6s} {'alone':>6s} | {'floor e':>7s} | {'raw R2':>6s} |"
        f" {'e by size':>20s} | views: reading r, shared error r"
    )
    for stage in STAGES:
        agree = row["cross_view"].get(stage)
        for view in VIEWS:
            cell = row["trained"][view].get(stage)
            if cell is None:
                continue
            floor = row.get("untrained", {}).get(view, {}).get(stage)
            sizes = " /".join(voxels(e, 5) for e in cell["error_by_size"])
            tail = f" | {agree['reading_r']:+.2f}, {agree['shared_error_r']:+.2f}" if agree and view == "t1" else ""
            print(
                f"    {stage:7s} {view:5s} {voxels(cell['error'])} {cell['own_r2']:6.2f} {cell['alone_r2']:6.2f} |"
                f" {voxels(floor['error'] if floor else None, 7)} | {cell['raw_r2']:6.2f} | {sizes:>20s}{tail}"
            )
    means = row["trained"]["t1"]["code"]["size_means"]
    print(
        "    size thirds are of the volume the other latents predict (not the volume itself, which would tie own to"
        f" it); mean lesion volume {' / '.join(f'{m:.0f}' for m in means)} voxels"
    )
    if "price" not in row:
        return
    errors = row["errors"]
    sharpen = errors.index(10.0), errors.index(20.0)
    print(
        "  price: drop in the run's loss (nats) when one unit holding the own part, read with error e, joins the code."
        "\n  20->10: what sharpening the reading from 20 to 10 voxels earns. left: mean R2 of lesion position / of"
        " anatomy still in the code, T1 then FLAIR"
    )
    print(
        f"    {'code':16s} {'sets':9s} {'loss':>6s} "
        + " ".join(f"{'e ' + format(e, 'g'):>7s}" for e in errors)
        + f" {'20->10':>7s} | left: position, anatomy"
    )
    for variant, cell in row["price"].items():
        for i, rule in enumerate(("uniform", "neighbour")):
            reward = cell[rule]["reward"]
            values = " ".join(f"{r:+7.4f}" for r in reward) + f" {reward[sharpen[0]] - reward[sharpen[1]]:+7.4f}"
            left = ""
            if i == 0:
                pos, ana = cell["left"]["position"], cell["left"]["anatomy"]
                left = f" | {pos[0]:.2f}/{pos[1]:.2f}, {ana[0]:.2f}/{ana[1]:.2f}"
            print(f"    {variant if i == 0 else '':16s} {rule:9s} {cell[rule]['loss']:6.3f} {values}{left}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_json", type=Path)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--jitter-draws", type=int, default=0, help="Move the test scans by the run's training jitter first")
    p.add_argument("--split", choices=("test", "train"), default="test", help="train: score training subjects instead")
    a = p.parse_args()
    if a.jitter_draws < 0:
        p.error("--jitter-draws must be 0 or more")
    checks = a.split != "test" or a.jitter_draws > 0
    device = warm.pick_device()
    results = json.loads(a.out_json.read_text()) if a.out_json.exists() else {}
    for run in a.runs:
        try:
            step = int(torch.load(run / "model.pt", map_location="cpu")["step"])
            last = int(json.loads((run / "report.json").read_text())["args"]["steps"])
        except Exception as error:  # no checkpoint yet, or one being written right now
            print("skipping", run.name, "for now:", error, flush=True)
            continue
        if step != last:
            continue
        cached = results.get(run.name, {})
        condition = (cached.get("split", "test"), cached.get("jitter_draws", 0)) == (a.split, a.jitter_draws)
        if cached.get("step") == step and condition:
            if not checks and VARIANTS[0] not in cached.get("price", {}):
                print("repricing", run.name, flush=True)
                with torch.no_grad():
                    cached["price"] = reprice(run, device)
                a.out_json.write_text(json.dumps(results, indent=1))
            continue
        print("assessing", run.name, flush=True)
        try:
            with torch.no_grad():
                results[run.name] = assess(run, device, a.out_json.parent, a.split, a.jitter_draws)
        except ValueError as error:
            print("skipping", run.name + ":", error, flush=True)
            continue
        a.out_json.write_text(json.dumps(results, indent=1))
    for name in sorted(results):
        show(name, results[name])


if __name__ == "__main__":
    main()
