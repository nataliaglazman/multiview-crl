#!/usr/bin/env python
"""Did the factors LEAVE the content block, or just stop being linearly readable?

`eval.gradient_attribution` established that reconstruction, not the contrastive term,
drives the post-peak decay of block-MCC. It did not establish what recon does. Two
possibilities behave identically under a ridge probe:

  REFORMAT  the factors are still there, encoded more nonlinearly (or in a different
            basis) as the encoder specialises for reconstruction. Under
            ``--synthetic-clean-content`` this is what you should EXPECT: the image is a
            deterministic function of 9 content + 3 style factors, so reconstruction at
            its optimum REQUIRES the factors to be present -- but only requires them to be
            present, not to be LINEARLY present. Block-MCC uses a ridge readout, so a
            perfectly reconstructing encoder can score poorly on it.

  LOSS      the factors are genuinely less recoverable by any means.

The distinction decides whether the decay matters. A reformat is a statement about the
probe; a loss is a statement about the representation.

Two tests, run on the same extracted blocks:

  1. PROBE LADDER. block-MCC under ridge / kernel / MLP readouts at each checkpoint. If the
     nonlinear gain (kernel - ridge) is much larger at the LATE checkpoint, the factors
     moved into a form the linear probe cannot see. If the ladder is flat at both, they did
     not.

  2. CROSS-CHECKPOINT LINEAR RECOVERABILITY. Fit a linear map late -> early on one half of
     the subjects, apply it to the other half, and score block-MCC on the mapped block. If
     the mapped late block scores like the true early block, then everything the early
     representation had is still linearly present in the late one, up to a change of basis
     -- a re-gauging, not a loss. The reverse direction (early -> late) is reported as a
     control: it should be easier if the late block is strictly richer.

Default pooling is GAP, not patch. The decay is present there too (0.8286 -> 0.7896 on the
reference run) and 44 features make the kernel and MLP probes cheap, where 22528 patch
features would make them impractical. Pass --pooling patch to pay for the fuller picture.

Usage
-----
    python -m eval.reformat_vs_loss --run-dir results/synthetic/RUN \\
        --checkpoints vqvae_best.pt vqvae_model.pt
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

# torch only inside the extraction path, so the scoring below stays numpy-testable.

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def probe_ladder(x, z, kinds=("ridge", "kernel", "mlp"), seeds=(0, 1), n_splits=5):
    """block-MCC under several readout families on the same block."""
    from eval.identifiability_metrics import block_mcc

    out = {}
    for kind in kinds:
        try:
            out[kind] = block_mcc(np.asarray(x), np.asarray(z), kind=kind, seeds=seeds, n_splits=n_splits)["mean"]
        except Exception as exc:  # a probe family failing should not lose the others
            logger.warning("probe %s failed: %s", kind, exc)
            out[kind] = float("nan")
    return out


def linear_recoverability(x_from, x_to, z, seeds=(0, 1), n_splits=5, seed=0):
    """Can ``x_to`` be linearly reconstructed from ``x_from``, well enough to score like it?

    The map is fitted on one half of the SUBJECTS and everything is scored on the other
    half, so a map that merely memorises cannot help. Returns the mapped block's block-MCC
    alongside the true target's on the same held-out subjects, which is the only fair
    reference -- comparing against the full-sample score would confound the split.
    """
    from sklearn.linear_model import RidgeCV

    from eval.identifiability_metrics import block_mcc

    x_from, x_to, z = np.asarray(x_from), np.asarray(x_to), np.asarray(z)
    n = x_from.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    fit_i, ev_i = perm[: n // 2], perm[n // 2 :]

    mapper = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)).fit(x_from[fit_i], x_to[fit_i])
    mapped = np.asarray(mapper.predict(x_from[ev_i]))

    return {
        "mapped": block_mcc(mapped, z[ev_i], seeds=seeds, n_splits=n_splits)["mean"],
        "true_target": block_mcc(x_to[ev_i], z[ev_i], seeds=seeds, n_splits=n_splits)["mean"],
        "source": block_mcc(x_from[ev_i], z[ev_i], seeds=seeds, n_splits=n_splits)["mean"],
        "map_r2": float(mapper.score(x_from[ev_i], x_to[ev_i])),
    }


def verdict(ladder_a, ladder_b, rec_late_to_early, label_a, label_b, sd=0.002):
    # The two tests are not alternatives and the real answer is usually a MIXTURE, so lead
    # with the split rather than with whichever branch fires. What matters is how much of
    # the linear decay a nonlinear readout buys back, and how much survives every probe.
    lin_decay = ladder_a.get("ridge", np.nan) - ladder_b.get("ridge", np.nan)
    best_a = max((v for v in ladder_a.values() if np.isfinite(v)), default=np.nan)
    best_b = max((v for v in ladder_b.values() if np.isfinite(v)), default=np.nan)
    irreducible = best_a - best_b
    print("\n  Verdict")
    if np.isfinite(lin_decay) and np.isfinite(irreducible) and lin_decay > 0:
        share = 100 * max(lin_decay - irreducible, 0.0) / lin_decay
        print(f"    linear (ridge) decay        {lin_decay:+.4f}")
        print(f"    best-probe decay            {irreducible:+.4f}   <- survives EVERY readout tried")
        print(f"    recovered by going nonlinear {share:.0f}% of the linear decay")
        if irreducible > 3 * sd:
            print("      So this is a MIXTURE: some reformatting, but most of the drop is factor")
            print("      information no probe here can recover. Report the best-probe decay as the")
            print("      loss; the ridge decay overstates it.")
        else:
            print("      Essentially all of the linear decay is recovered nonlinearly — the factors")
            print("      are still present and the ridge probe is what is failing.")

    gain_a = ladder_a.get("kernel", np.nan) - ladder_a.get("ridge", np.nan)
    gain_b = ladder_b.get("kernel", np.nan) - ladder_b.get("ridge", np.nan)
    print(f"\n    nonlinear gain (kernel - ridge):  {label_a} {gain_a:+.4f}   {label_b} {gain_b:+.4f}")
    if np.isfinite(gain_a) and np.isfinite(gain_b) and (gain_b - gain_a) > 3 * sd:
        print("      The LATE block gains much more from a nonlinear readout, so the factors")
        print("      became less LINEARLY readable rather than less present. Under clean_content")
        print("      that is the expected consequence of optimising reconstruction: recon needs")
        print("      the factors present, not linearly present.")
    else:
        print("      No differential nonlinear gain — the late block is not simply hiding the")
        print("      factors behind a nonlinearity.")

    m, t, s = rec_late_to_early["mapped"], rec_late_to_early["true_target"], rec_late_to_early["source"]
    print(
        f"\n    linear map {label_b} -> {label_a}:  mapped {m:.4f}   true {label_a} {t:.4f}   "
        f"(map R^2 {rec_late_to_early['map_r2']:.3f})"
    )
    if np.isfinite(m) and np.isfinite(t) and (t - m) < 3 * sd:
        print("      The late block linearly CONTAINS what the early one had. The decay is a")
        print("      change of basis that the ridge probe pays for, not information leaving —")
        print("      i.e. a measurement artifact of scoring a re-gauged representation.")
    else:
        print(f"      The map falls {t - m:+.4f} short of the true early block, so the early")
        print("      representation held something the late one cannot linearly supply. That is")
        print("      genuine loss, and the amount is the size of the gap.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--checkpoints", nargs="+", default=["vqvae_best.pt", "vqvae_model.pt"], help="EARLY first, LATE second."
    )
    ap.add_argument("--pooling", choices=("gap", "patch", "stats"), default="gap")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--kinds", nargs="+", default=["ridge", "kernel", "mlp"])
    ap.add_argument("--causal", choices=("match", "iid"), default="match")
    cli = ap.parse_args()

    import os

    import torch

    from eval.dci import _extract_synthetic_representations
    from eval.run_dci_compare import _CONTENT
    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    if len(cli.checkpoints) != 2:
        raise SystemExit("--checkpoints takes exactly two, EARLY first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, args_, _ = load_model_from_run_dir(cli.run_dir, os.path.join(cli.run_dir, cli.checkpoints[0]), device)
    dataset = build_synthetic_test_set(args_, cli.num_samples, cache=False, causal=cli.causal == "match")
    pooling = cli.pooling if cli.pooling != "patch" else tuple(getattr(args_, "patch_grid", [8, 8, 8]))

    blocks, gt = {}, None
    for name in cli.checkpoints:
        path = os.path.join(cli.run_dir, name)
        if not os.path.exists(path):
            raise SystemExit(f"missing checkpoint: {path}")
        model, _, _ = load_model_from_run_dir(cli.run_dir, path, device)
        model.to(device)
        ld, gc, _s1, _s2 = _extract_synthetic_representations(
            model, dataset, device, cli.batch_size, 0, pooling=pooling
        )
        blocks[name] = ld[cli.level][_CONTENT]
        gt = gc if gt is None else gt
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    early, late = cli.checkpoints
    la, lb = early.replace(".pt", ""), late.replace(".pt", "")
    print(f"\n{'=' * 78}\nREFORMAT vs LOSS — {cli.pooling} pooling, {blocks[early].shape}\n{'=' * 78}")

    print(f"\n  {'probe':<10}{la:>16}{lb:>16}{'delta':>12}")
    print("  " + "-" * 54)
    lad = {}
    for name in (early, late):
        lad[name] = probe_ladder(blocks[name], gt, tuple(cli.kinds), tuple(cli.seeds), cli.n_splits)
    for kind in cli.kinds:
        a, b = lad[early].get(kind, np.nan), lad[late].get(kind, np.nan)
        print(f"  {kind:<10}{a:>16.4f}{b:>16.4f}{b - a:>+12.4f}")

    print(f"\n  cross-checkpoint linear recoverability")
    rec = linear_recoverability(blocks[late], blocks[early], gt, tuple(cli.seeds), cli.n_splits)
    ctrl = linear_recoverability(blocks[early], blocks[late], gt, tuple(cli.seeds), cli.n_splits)
    print(
        f"    {lb} -> {la}   mapped {rec['mapped']:.4f}  vs true {rec['true_target']:.4f}  "
        f"(map R^2 {rec['map_r2']:.3f})"
    )
    print(
        f"    {la} -> {lb}   mapped {ctrl['mapped']:.4f}  vs true {ctrl['true_target']:.4f}  "
        f"(map R^2 {ctrl['map_r2']:.3f})   [control]"
    )

    verdict(lad[early], lad[late], rec, la, lb)
    print()


if __name__ == "__main__":
    main()
