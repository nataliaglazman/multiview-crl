"""Positive control for burden probes on raw image cells, then candidates on a trained map."""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

torch.set_num_threads(3)
QS = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999])
ALPHAS = np.logspace(-2, 6, 17)


def fixed_probe(cells, y, features, seed=0):
    """Held-out ridge on position-free features fitted inside each fold. cells: (n, C, P)."""
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(cells):
        F = features(cells, tr)
        m, s = F[tr].mean(0), F[tr].std(0) + 1e-9
        pred[te] = RidgeCV(alphas=ALPHAS).fit((F[tr] - m) / s, y[tr]).predict((F[te] - m) / s)
    return pred


def centred(cells, tr):
    return (cells - cells[tr].mean(0, keepdims=True)) / (cells[tr].std(0, keepdims=True) + 1e-6)


def channel_quantiles(cells, tr):
    z = centred(cells, tr)
    return np.quantile(z, QS, axis=2).transpose(1, 2, 0).reshape(len(z), -1)


def whiten(cells, tr):
    z = centred(cells, tr)
    C = z[tr].transpose(0, 2, 1).reshape(-1, z.shape[1])
    cov = C.T @ C / len(C) + 1e-3 * np.eye(C.shape[1])
    w, V = np.linalg.eigh(cov)
    return np.einsum("nct,cd->ndt", z, V / np.sqrt(w)[None])


def white_quantiles(cells, tr):
    z = whiten(cells, tr)
    return np.quantile(z, QS, axis=2).transpose(1, 2, 0).reshape(len(z), -1)


def distance_quantiles(cells, tr):
    z = whiten(cells, tr)
    d = np.sqrt((z**2).sum(1))
    return np.quantile(d, QS, axis=1).T


def report(name, pred, y, small):
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(
        f"  {name:<30s} R2 {r2:+.3f}  spearman {spearmanr(pred, y).statistic:+.3f}  smallest third {spearmanr(pred[small], y[small]).statistic:+.3f}",
        flush=True,
    )


def suite(cells, y, small, set_probe=True):
    if set_probe:
        report("learned set (current)", set_predictions(cells, y, 0, 300), y, small)
    report("channel quantiles", fixed_probe(cells, y, channel_quantiles), y, small)
    if cells.shape[1] > 1:
        report("whitened channel quantiles", fixed_probe(cells, y, white_quantiles), y, small)
        report("cell distance quantiles", fixed_probe(cells, y, distance_quantiles), y, small)
