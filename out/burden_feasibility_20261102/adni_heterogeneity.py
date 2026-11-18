"""Does ADNI's anatomy carry the heterogeneity heterogeneity_ica.py needs? On the FreeSurfer volumes in ADNIMERGE
(baseline visits, one FreeSurfer version at a time), with age bins (or sex) as the observed variable: separate the
volumes by joint diagonalisation of the bins' covariances, as heterogeneity_ica.py separates a code, and report

  - each component's strength: the SD across bins of its log SD (heterogeneity_ica.py's strength axis);
  - each component's stability: the mean |corr| between it and its match in --boot bootstrap refits (subjects resampled
    within bins), where 1 means the bins identify that component;
  - the same with the bins shuffled across subjects, the null.

Diagnosis is not used: it descends from the anatomy, and splitting on it would couple the sources. Volumes are logged,
then standardized. No images and no labels beyond the table.

    python out/burden_feasibility_20261102/adni_heterogeneity.py ADNIMERGE.csv [--bins 5] [--by age|sex] [--boot 50]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heterogeneity_ica import joint_diagonalize  # noqa: E402

VOLUMES = ("ICV", "WholeBrain", "Ventricles", "Hippocampus", "Entorhinal", "Fusiform", "MidTemp")


def separate(x, level, levels):
    """Unmixing matrix (features x components) and each component's SD per level."""
    centred = x - np.stack([x[level == k].mean(0) for k in range(levels)])[level]
    values, vectors = np.linalg.eigh(np.cov(centred.T))
    whiten = vectors / np.sqrt(values)
    white = centred @ whiten
    rotation = joint_diagonalize([np.cov(white[level == k].T) for k in range(levels)])
    unmix = whiten @ rotation
    sds = np.stack([(centred[level == k] @ unmix).std(0) for k in range(levels)])
    return unmix, sds


def strength(sds):
    return np.log(sds).std(0)


def stability(x, level, levels, unmix, boot, rng):
    """Mean |corr| of each component with its Hungarian match among bootstrap refits' components."""
    reference = x @ unmix
    out = []
    for _ in range(boot):
        rows = np.concatenate([rng.choice(np.flatnonzero(level == k), (level == k).sum()) for k in range(levels)])
        refit, _ = separate(x[rows], level[rows], levels)
        corr = np.abs(np.corrcoef(reference.T, (x @ refit).T)[: unmix.shape[1], unmix.shape[1] :])
        r, c = linear_sum_assignment(-corr)
        matched = np.zeros(unmix.shape[1])
        matched[r] = corr[r, c]
        out.append(matched)
    return np.mean(out, axis=0)


def report(name, x, level, levels, boot, rng):
    unmix, sds = separate(x, level, levels)
    stable = stability(x, level, levels, unmix, boot, rng)
    order = np.argsort(-strength(sds))
    loads = np.corrcoef((x @ unmix).T, x.T)[: x.shape[1], x.shape[1] :]
    print(f"  {name}: component strength / stability, and the volume it correlates with most")
    for j in order:
        top = int(np.argmax(np.abs(loads[j])))
        print(f"    {strength(sds)[j]:5.2f} / {stable[j]:4.2f}   {VOLUMES[top]} ({loads[j, top]:+.2f})")
    return strength(sds), stable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("adnimerge")
    p.add_argument("--bins", type=int, default=5, help="age bins (equal counts)")
    p.add_argument("--by", choices=("age", "sex"), default="age")
    p.add_argument("--boot", type=int, default=50)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    table = pd.read_csv(a.adnimerge, low_memory=False)
    base = table[table.VISCODE == "bl"].dropna(subset=[*VOLUMES, "AGE", "PTGENDER", "FSVERSION"])
    print(f"ADNIMERGE baseline visits with all of {', '.join(VOLUMES)}, age and sex: {len(base)}")
    for version, rows in base.groupby("FSVERSION"):
        x = np.log(rows[list(VOLUMES)].to_numpy(float))
        x = (x - x.mean(0)) / x.std(0)
        if a.by == "age":
            level = pd.qcut(rows.AGE.to_numpy(), a.bins, labels=False)
            levels = a.bins
        else:
            level = (rows.PTGENDER.to_numpy() == "Female").astype(int)
            levels = 2
        print(f"\n{version}: {len(rows)} subjects, {levels} {a.by} levels")
        real = report(a.by, x, level, levels, a.boot, rng)
        null = report("shuffled " + a.by, x, rng.permutation(level), levels, a.boot, rng)
        print(
            f"  mean strength {real[0].mean():.2f} (shuffled {null[0].mean():.2f}); components with stability >= .9:"
            f" {(real[1] >= 0.9).sum()} of {len(real[1])} (shuffled {(null[1] >= 0.9).sum()})"
        )


if __name__ == "__main__":
    main()
