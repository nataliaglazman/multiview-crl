"""Training batches drawn from one cluster of subjects at a time: the encoder-only trainer's --stratify-batches.

Every batch holds members of a single k-means cluster, so InfoNCE's negatives share whatever was
clustered and a subject can only be told apart from them by the rest. This is the batch rule of
MCL's Algorithm 1 (Zhang et al., "Learning the Unlearned", arXiv 2402.11816): cycle through the
clusters, each batch taken from the untraversed members of the next cluster that has any left, until
the epoch has used them all. Here every batch is full, because InfoNCE depends on the number of
negatives: a cluster smaller than the batch size is dropped, and each cluster's remainder sits out
the epoch (it is reshuffled into the next one).

Torch-free: DataLoader accepts any iterable of index lists as ``batch_sampler``.
"""

import numpy as np

# Content factors left out of the oracle's clustering: every lesion latent, and the slots the burden
# target draws but never renders (eval.metrics.dci.content_factor_names).
LESION_PREFIXES = ("lesion_", "unused_")


def anatomy_columns(names):
    """Indices of the content factors that are anatomy rather than lesion."""
    return [j for j, name in enumerate(names) if not name.startswith(LESION_PREFIXES)]


def bank_latents(dataset):
    """Every subject's z_content in a SyntheticBrainDataset, read from the generator without rendering a view.

    ``sample_seed_for`` settles which candidate draw a subject keeps (it checks that the lesion fits),
    so these are the latents of the images the dataset renders. About 10 ms a subject at res 64.
    """
    inner = dataset._inner
    return np.stack(
        [
            np.asarray(inner._draw_pseudo_mri(inner.sample_seed_for(i))["z_content"], dtype=np.float64)
            for i in range(len(dataset))
        ]
    )


def standardize(features):
    """Each column centred and scaled to unit SD over the rows; a constant column is only centred."""
    features = np.asarray(features, dtype=np.float64)
    sd = features.std(axis=0)
    return (features - features.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def kmeans_labels(features, k, seed):
    """k-means cluster of every row of the standardized ``features``. K = 1 is one cluster, without a fit."""
    features = standardize(features)
    if k == 1:
        return np.zeros(len(features), dtype=np.int64)
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(features).astype(np.int64)


class ClusterBatchSampler:
    """Full batches of ``batch_size`` subject indices, each from a single cluster of ``labels``.

    Each epoch shuffles every kept cluster, cuts it into full batches and emits them cycling through
    the clusters (in a fresh random order each epoch), skipping a cluster once it has none left. A
    subject appears at most once per epoch. The order replays from ``seed``.
    """

    def __init__(self, labels, batch_size, seed):
        labels = np.asarray(labels)
        if labels.ndim != 1 or not len(labels):
            raise ValueError("labels must be a non-empty vector, one cluster per subject")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        names, counts = np.unique(labels, return_counts=True)
        self.sizes = {int(c): int(n) for c, n in zip(names, counts)}
        self.kept = [c for c, n in self.sizes.items() if n >= self.batch_size]
        if not self.kept:
            raise ValueError(
                f"No cluster has {self.batch_size} subjects (largest {max(self.sizes.values())}); "
                "use fewer clusters or a smaller batch"
            )
        self.members = [np.flatnonzero(labels == c) for c in self.kept]
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return sum(len(m) // self.batch_size for m in self.members)

    def __iter__(self):
        b = self.batch_size
        queues = []
        for members in self.members:
            order = self._rng.permutation(members)
            queues.append([order[i * b : (i + 1) * b].tolist() for i in range(len(order) // b)])
        cycle = self._rng.permutation(len(queues))
        position = [0] * len(queues)
        remaining = len(self)
        while remaining:
            for c in cycle:
                if position[c] < len(queues[c]):
                    yield queues[c][position[c]]
                    position[c] += 1
                    remaining -= 1


def loader_batching(args, build_sampler):
    """The training DataLoader's batching options: shuffled full batches unless --stratify-batches is set.

    ``build_sampler`` is called only for a stratified run, so a default run builds its loader exactly
    as before the flag existed. Hand-built option namespaces may omit the flag; they read as none.
    """
    if getattr(args, "stratify_batches", "none") == "none":
        return {"batch_size": args.batch_size, "shuffle": True, "drop_last": True}
    return {"batch_sampler": build_sampler()}


def within_cluster_sd_ratio(values, labels, batch_size):
    """Per column: the SD inside a batch's cluster over the SD across every subject.

    Each cluster's variance (ddof 1) is weighted by the full batches it gives per epoch, so this is
    the spread of a factor among an average batch's negatives, relative to an unstratified batch's.
    Dropped clusters give no batches and so no weight. One cluster gives exactly 1.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    variance = np.zeros(values.shape[1])
    weight = 0
    for c in np.unique(labels):
        rows = values[labels == c]
        batches = len(rows) // batch_size
        if batches and len(rows) > 1:
            variance += batches * rows.var(axis=0, ddof=1)
            weight += batches
    bank = values.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(variance / max(weight, 1)) / bank


def summarize(labels, batch_size, factors, names, anatomy, k):
    """Clusters kept, subjects and batches per epoch, and each factor's within-cluster SD ratio.

    ``anatomy`` are the columns of ``factors`` the clustering should condition. Their mean ratio is
    set beside K^(-1/d), d = len(anatomy): the share of each factor's spread left inside one of K
    equal cells of a d-dimensional cube.
    """
    sizes = np.unique(np.asarray(labels), return_counts=True)[1]
    kept = sizes[sizes >= batch_size]
    batches = int((sizes // batch_size).sum())
    ratio = within_cluster_sd_ratio(factors, labels, batch_size)
    return {
        "clusters": int(len(sizes)),
        "cluster_sizes": sorted(int(n) for n in sizes),
        "clusters_kept": int(len(kept)),
        "subjects": int(sizes.sum()),
        "subjects_in_kept_clusters": int(kept.sum()),
        "batches_per_epoch": batches,
        "subjects_per_epoch": batches * int(batch_size),
        "anatomy_factors": [names[j] for j in anatomy],
        "sd_ratio": {name: float(r) for name, r in zip(names, ratio)},
        "sd_ratio_anatomy_mean": float(np.mean(ratio[anatomy])),
        "theory_k_pow_minus_1_over_d": float(k ** (-1.0 / len(anatomy))),
    }
