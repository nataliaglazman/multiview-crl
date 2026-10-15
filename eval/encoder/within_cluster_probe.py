"""Anatomy in a stratified run's code, read within the oracle's clusters as well as across them.

A run trained with ``--stratify-batches`` only ever compares subjects of one k-means cluster, and
``--standardize-code`` subtracts each batch's mean from every code unit first. The loss therefore
never sees where a cluster sits in code space, and nothing keeps the code on one scale from
cluster to cluster. The usual per-factor probe fits one readout across every test subject, so it
can score such a code low on a factor the code still resolves finely inside each cluster.

This script scores both. Each test subject goes to the nearest centre of a run's oracle clusters
(the centres of its stratification.json labels, in the standardized anatomy space k-means used).
The **within** score subtracts every cluster's mean from both the code and the factor before the
probe. The **global** score is the training-time probe on the raw code. Both are
``cv_probe_r2``'s ridge R² (5-fold CV, seeds 0-2), as ``dci_step*.json`` reports per factor.
Clusters with a single test subject drop out of the within score. ``within share`` is the part of
a factor's test variance left after the cluster means are removed: the variance the within score
explains a share of. Lesion factors are not clustered, so their two scores should agree.

Example:
    python -m eval.encoder.within_cluster_probe \
        --run-dirs results/encoder/runs/<k1 run> results/encoder/runs/<k60 run>
    python -m eval.encoder.within_cluster_probe --self-test
"""

import argparse
import json
import os

import numpy as np

import data.cluster_batches as cluster_batches
from eval.metrics.identifiability_metrics import cv_probe_r2

VIEWS = ("t1", "flair")
# Generator settings two runs must share for their test subjects (and clusters) to be the same.
DATA_KEYS = ("res", "n_content", "n_style", "data_seed", "dataset_name")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dirs", nargs="+", help="Encoder-only run directories to score (settings.json + checkpoint)")
    p.add_argument(
        "--clusters-from",
        default=None,
        help="Run whose stratification.json defines the clusters. Default: the --run-dirs run with the largest K",
    )
    p.add_argument("--checkpoint", default="model.pt", help="Checkpoint filename inside each run directory")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Test subjects (default: the runs' --num-val-samples, the set the training log scores). From 64 up, "
        "the first N subjects of a split and their normalization never change with N, so a larger N (e.g. 1200, "
        "~20 a cluster at K = 60) only adds subjects; its global scores then differ from the log's",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--out", default=None, help="Also write the scores to this JSON file")
    p.add_argument(
        "--self-test", action="store_true", help="Check the within-cluster arithmetic on toy codes; no torch"
    )
    args = p.parse_args(argv)
    if not args.self_test and not args.run_dirs:
        p.error("--run-dirs is required unless --self-test")
    return args


def load_json(path):
    with open(path) as fp:
        return json.load(fp)


def cluster_centres(features, labels):
    """Standardization and per-cluster means of the features k-means saw, so new rows can be assigned.

    ``cluster_batches.kmeans_labels`` clusters standardized features (population SD, a constant
    column only centred); the centres here live in that same space.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    mean, sd = features.mean(axis=0), features.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    clusters = np.unique(labels)
    scaled = (features - mean) / sd
    centres = np.stack([scaled[labels == c].mean(axis=0) for c in clusters])
    return mean, sd, clusters, centres


def assign(features, mean, sd, clusters, centres):
    """Nearest centre of every row, as a cluster label."""
    scaled = (np.asarray(features, dtype=np.float64) - mean) / sd
    distance = ((scaled[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    return clusters[distance.argmin(axis=1)]


def demean_within(groups, *arrays):
    """Rows of groups with at least two members, each array minus its group's mean.

    Returns the kept-row mask and the demeaned arrays. A lone row would be demeaned to exactly zero
    in every column, so it carries no within-group information and is dropped.
    """
    groups = np.asarray(groups)
    names, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    keep = counts[inverse] >= 2
    out = []
    for array in arrays:
        array = np.asarray(array, dtype=np.float64)
        array = array.reshape(len(array), -1)[keep]
        index = inverse[keep]
        sums = np.zeros((len(names), array.shape[1]))
        np.add.at(sums, index, array)
        out.append(array - (sums / np.maximum(counts, 1)[:, None])[index])
    return keep, out


def within_share(factors, groups):
    """Per column: the share of its variance (over the kept rows) left after removing group means."""
    keep, (residual,) = demean_within(groups, factors)
    kept = np.asarray(factors, dtype=np.float64).reshape(len(groups), -1)[keep]
    total = ((kept - kept.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (residual**2).sum(axis=0) / total


def probe_scores(code, factors, groups):
    """Global and within-cluster ridge R² of every factor column from ``code``."""
    code = np.asarray(code, dtype=np.float64)
    factors = np.asarray(factors, dtype=np.float64)
    _, (code_within, factors_within) = demean_within(groups, code, factors)
    return [
        {
            "global": cv_probe_r2(code, factors[:, j])["mean"],
            "within": cv_probe_r2(code_within, factors_within[:, j])["mean"],
        }
        for j in range(factors.shape[1])
    ]


def encode_runs(models, dataset, device, batch_size):
    """Content code of both views under every model, from one pass over the rendered subjects."""
    import torch
    from torch.utils.data import DataLoader

    codes = [{view: [] for view in VIEWS} for _ in models]
    factors = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size):
            x = torch.cat(batch["image"], dim=0).to(device)
            n = x.shape[0] // 2
            for model, code in zip(models, codes):
                feats = model(x, pool_only=True, n_views=2)[2][0]
                content_units, _ = model.unit_split(feats.shape[1])
                content = feats[:, content_units].flatten(1).cpu().double().numpy()
                code["t1"].append(content[:n])
                code["flair"].append(content[n:])
            factors.append(batch["gt_latents"]["z_content"].double().numpy())
    return [{view: np.concatenate(parts) for view, parts in code.items()} for code in codes], np.concatenate(factors)


def run(args):
    import torch

    from eval.metrics.dci import content_factor_names
    from eval.protocol.score_checkpoint import build_model, load_settings, make_dataset
    from utils.encoder_runtime import select_encoder_device

    runs = []
    for run_dir in args.run_dirs:
        path = os.path.join(run_dir, "stratification.json")
        progress = os.path.join(run_dir, "training_progress.json")
        runs.append(
            {
                "run_dir": run_dir,
                "cfg": load_settings(run_dir),
                "stratification": load_json(path) if os.path.isfile(path) else None,
                "step": load_json(progress).get("step") if os.path.isfile(progress) else None,
            }
        )
    source = args.clusters_from
    if source is None:
        stratified = [r for r in runs if r["stratification"] is not None and r["stratification"]["k"] > 1]
        if not stratified:
            raise SystemExit("No --run-dirs run has a stratification.json with K > 1; pass --clusters-from")
        source = max(stratified, key=lambda r: r["stratification"]["k"])["run_dir"]
    strat = load_json(os.path.join(source, "stratification.json"))
    if strat["mode"] != "oracle":
        raise SystemExit(f"{source} clustered a stage-0 code ({strat['mode']}); its centres are not in anatomy space")
    cfg = load_settings(source)
    for r in runs:
        differ = [k for k in DATA_KEYS if r["cfg"].get(k) != cfg.get(k)]
        differ += [k for k in set(r["cfg"]) | set(cfg) if k.startswith("synthetic_") and r["cfg"].get(k) != cfg.get(k)]
        if differ:
            raise SystemExit(f"{r['run_dir']} and {source} use different data: {', '.join(sorted(differ))}")

    names = content_factor_names(cfg["n_content"], cfg.get("synthetic_lesion_target", "position"))
    anatomy = [names.index(name) for name in strat["anatomy_factors"]]
    print(f"clusters: {source}, K = {strat['k']} oracle clusters of {', '.join(strat['anatomy_factors'])}", flush=True)
    bank = cluster_batches.bank_latents(make_dataset(cfg, cfg["num_train_samples"], mode="train"))
    labels = np.asarray(strat["labels"])
    if len(labels) != len(bank):
        raise SystemExit(f"stratification.json has {len(labels)} labels for {len(bank)} training subjects")
    centre_fit = cluster_centres(bank[:, anatomy], labels)
    agreement = float((assign(bank[:, anatomy], *centre_fit) == labels).mean())
    print(f"  training subjects back in their own cluster after recomputing the centres: {agreement:.1%}", flush=True)

    device = select_encoder_device(args.device)
    models = []
    for r in runs:
        r["checkpoint"] = os.path.join(r["run_dir"], args.checkpoint)
        models.append(build_model(r["cfg"], device, torch.load(r["checkpoint"], map_location="cpu")))
    num_samples = args.num_samples or cfg["num_val_samples"]
    dataset = make_dataset(cfg, num_samples, mode=args.split)
    codes, factors = encode_runs(models, dataset, device, args.batch_size)
    groups = assign(factors[:, anatomy], *centre_fit)
    keep, _ = demean_within(groups, factors)
    present = len(np.unique(groups))
    print(
        f"  {num_samples} {args.split} subjects in {present} clusters; {int(keep.sum())} in the "
        f"{len(np.unique(groups[keep]))} clusters with >= 2 enter the within score",
        flush=True,
    )
    share = within_share(factors, groups)

    letters = [chr(ord("A") + i) for i in range(len(runs))]
    for letter, r in zip(letters, runs):
        k = r["stratification"]["k"] if r["stratification"] else "unstratified"
        print(f"  run {letter}: {r['run_dir']} (K = {k}, step {r['step']}, {args.checkpoint})", flush=True)
    report = {
        "clusters_from": source,
        "k": strat["k"],
        "anatomy_factors": strat["anatomy_factors"],
        "split": args.split,
        "num_samples": num_samples,
        "clusters_present": present,
        "subjects_within": int(keep.sum()),
        "training_reassignment_agreement": agreement,
        "probe": "eval.metrics.identifiability_metrics.cv_probe_r2 (ridge, 5-fold, seeds 0-2)",
        "runs": [
            {"label": letter, "run_dir": r["run_dir"], "checkpoint": r["checkpoint"], "step": r["step"]}
            for letter, r in zip(letters, runs)
        ],
        "views": {},
    }
    for view in VIEWS:
        scores = [probe_scores(code[view], factors, groups) for code in codes]
        print(f"\n{view} content code: ridge R², global vs within the K = {strat['k']} clusters", flush=True)
        head = f"  {'factor':<20s}{'within share':>13s}"
        head += "".join(f"{letter + ' global':>11s}{letter + ' within':>11s}" for letter in letters)
        print(head, flush=True)
        report["views"][view] = {}
        for j, name in enumerate(names):
            row = f"  {name:<20s}{share[j]:>13.3f}"
            row += "".join(f"{s[j]['global']:>11.3f}{s[j]['within']:>11.3f}" for s in scores)
            print(row + ("   clustered" if j in anatomy else ""), flush=True)
            report["views"][view][name] = {
                "clustered": j in anatomy,
                "within_share": float(share[j]),
                "runs": [{"label": letter, **s[j]} for letter, s in zip(letters, scores)],
            }
    if args.out:
        with open(args.out, "w") as fp:
            json.dump(report, fp, indent=2)
        print(f"\nwrote {args.out}", flush=True)
    return report


def self_test():
    """The within score finds anatomy kept only inside clusters, and drops it when only cluster means are kept."""
    rng = np.random.default_rng(0)
    k, per = 20, 30
    groups = np.repeat(np.arange(k), per)
    between = rng.normal(size=k)[groups]
    within = 0.6 * rng.normal(size=k * per)
    factor = (between + within)[:, None]
    noise = 0.05 * rng.normal(size=(k * per, 3))

    # A code that resolves the factor inside each cluster but puts every cluster at a random offset.
    scrambled = np.column_stack([within + 3.0 * rng.normal(size=k)[groups], rng.normal(size=k * per), within]) + noise
    s = probe_scores(scrambled, factor, groups)[0]
    assert s["within"] > 0.95 and s["global"] < 0.45, s
    # A code that carries only which cluster a subject is in.
    coarse = np.column_stack([between, between**2, rng.normal(size=k * per)]) + noise
    s = probe_scores(coarse, factor, groups)[0]
    assert s["within"] < 0.05 and s["global"] > 0.6, s

    share = within_share(factor, groups)[0]
    expected = (within - np.bincount(groups, within)[groups] / per).var() / factor[:, 0].var()
    assert abs(share - expected) < 1e-9, (share, expected)

    # Singletons drop out; group means are removed exactly.
    keep, (demeaned,) = demean_within(np.array([0, 0, 1, 2, 2, 2]), np.array([1.0, 3.0, 9.0, 1.0, 2.0, 6.0]))
    assert keep.tolist() == [True, True, False, True, True, True]
    assert np.allclose(demeaned[:, 0], [-1.0, 1.0, -2.0, -1.0, 3.0])

    # Centres rebuilt from k-means labels put the clustered rows back in their own cluster.
    features = rng.normal(size=(600, 6)) * [1, 2, 3, 1, 1, 5] + 7
    labels = cluster_batches.kmeans_labels(features, 12, seed=0)
    fit = cluster_centres(features, labels)
    agreement = (assign(features, *fit) == labels).mean()
    assert agreement > 0.99, agreement
    print("self-test passed")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
