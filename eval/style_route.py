"""Is the trained STYLE block's factor knowledge just the input's intensity histogram?

`lut = base*gain + bias` writes the style parameters into the intensity histogram and
nowhere else, so the style pathway has to be a histogram reader. But a histogram also
counts tissue — a bigger ventricle means more CSF-dark voxels — so a channel built to
recover `gain`/`bias` picks up every VOLUME factor for free, without learning anything
spatial. `generator_defects --tests global` shows the data-generating process offers that
route (measured on the flagship preset: `ventricle_size` reaches R^2 0.44 from 13
permutation-invariant numbers once `--synthetic-identifiable-ventricle` is on).

What that test CANNOT show is whether a trained model actually takes it. Style here is a
stack of full-resolution conv channels; it could equally be storing a spatial blob of the
ventricle and never touching the histogram. The two have opposite fixes — a capacity
bottleneck on style versus a spatial one — so the distinction has to be measured, and it
needs a checkpoint. That is this script.

Four readouts per content factor, all on the same parent-residualised target:

  content      content block -> factor.  Where the factor is supposed to live.
  style        style block   -> factor.  THE LEAK.
  hist         input's shuffle-proof statistics -> factor.  The route's capacity.
  style|hist   R^2(factor | style, hist) - R^2(factor | hist).  What style knows about
               the factor BEYOND anything the histogram could have told it.

Read `style|hist`:

  ~0 with a large `style`   style's factor knowledge IS the histogram. The leak follows
                            the generator, and the lever is style CAPACITY (--style-embed-dim,
                            --style-nb-entries, the style channel count = hidden - content):
                            a style block with room to spare fills it with the cheapest
                            thing available, and volume is the cheapest thing available.
  clearly > 0               style holds spatial factor evidence the histogram cannot supply.
                            The lever is then style's spatial extent (--style-spatial-size),
                            not its capacity, and the histogram framing is not the story.

The volume-preserving factors are the control. Position, asymmetry and corrugation do not
move any tissue count, so the histogram is blind to them (measured: |R^2| <= 0.07 for all
six). If style scores well on THOSE too, it is simply capacious rather than reading the
histogram, and neither reading above applies.

Usage:
  python -m eval.style_route --run-dir results/synthetic/<run>
  python -m eval.style_route --run-dir <run> --pooling gap --num-samples 800
  python -m eval.style_route --run-dir <run> --random-init   # untrained floor
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from eval.dci import CONTENT_FACTOR_NAMES, _extract_synthetic_representations
from eval.generator_defects import _global_stats
from eval.identifiability_metrics import cv_probe_r2_multi, residualise_on_parents

# The measured split from `generator_defects --tests global` (flagship preset, res 32,
# n=384, parent-residualised): the first three change a tissue volume fraction and read
# 0.115-0.604 from the histogram; the rest are volume-preserving and read -0.014 to 0.067.
# Listed rather than inferred at runtime so the control group cannot drift silently.
VOLUME_FACTORS = ("brain_size", "ventricle_size", "cortical_thickness")
SHAPE_FACTORS = ("lesion_x", "lesion_y", "lesion_z", "temporal_atrophy", "lr_asymmetry", "sulcal_widening")

# `eval.identifiability_report.NOISE_FLOOR`, calibrated from `patch_mcc_decay --calibrate`:
# the smallest perturbation that reliably moved a block metric was +0.0447.
NOISE_FLOOR = 0.05


def _parse_pooling(spec):
    if spec in ("gap", "stats"):
        return spec
    parts = [int(v) for v in spec.replace(",", " ").split()]
    if len(parts) != 3:
        raise ValueError(f"--pooling wants 'gap', 'stats', or three ints; got {spec!r}")
    return tuple(parts)


def _input_histograms(ds):
    """Shuffle-proof statistics of view 1, in dataset order.

    `_extract_synthetic_representations` builds its loader with `shuffle=False`, so
    iterating the dataset here lands on the same rows as the encoder blocks. Returns
    `(H, adjacency)`; the adjacency is read off the first item that carries one.
    """
    rows, adj = [], None
    for i in range(len(ds)):
        item = ds[i]
        got = _global_stats(item["image"][0], item["mask"][0])
        if got is None:
            raise RuntimeError(f"sample {i} has an empty foreground mask; cannot summarise its histogram")
        rows.append(got[1])
        if adj is None and "causal_adj" in item["gt_latents"]:
            adj = item["gt_latents"]["causal_adj"].numpy()
    return torch.stack(rows).numpy(), adj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None, help="Defaults to the run's vqvae_model.pt")
    ap.add_argument("--num-samples", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument(
        "--pooling",
        default="gap",
        help="'gap' (default), 'stats', or a patch grid like '8 8 8'. gap is the right rung "
        "here: style is a global quantity, and it puts the style block (n_style_channels) on "
        "the same footing as the 13-number histogram it is being compared against.",
    )
    ap.add_argument("--probe", default="ridge", choices=["ridge", "kernel"])
    ap.add_argument(
        "--causal",
        default="match",
        choices=["match", "iid"],
        help="'match' reproduces the run's SCM (targets are then parent-residualised); 'iid' "
        "decorrelates the factors instead. Per-factor attribution needs one or the other — "
        "raw factors under a matched SCM read their parents.",
    )
    ap.add_argument("--random-init", action="store_true", help="Untrained floor: same path, unlearned weights.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    cli = ap.parse_args()

    from eval.run_dci_synthetic import build_synthetic_test_set, load_model_from_run_dir

    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, args, device = load_model_from_run_dir(
        cli.run_dir, cli.checkpoint, device, random_init=cli.random_init, seed=cli.seed
    )
    ds = build_synthetic_test_set(args, cli.num_samples, causal=(cli.causal == "match"))
    pooling = _parse_pooling(cli.pooling)

    level_data, gt_content, _gs1, _gs2 = _extract_synthetic_representations(
        model, ds, device, cli.batch_size, 0, pooling=pooling
    )
    if cli.level not in level_data:
        raise SystemExit(f"level {cli.level} not in encoder outputs (have {sorted(level_data)})")
    content, style, _c2, _s2, info = level_data[cli.level]
    if style is None or not style.shape[1]:
        raise SystemExit(
            "this run has no style block at that level (content_size == hidden_channels?), "
            "so there is no leak to attribute"
        )

    hist, adj = _input_histograms(ds)
    if len(hist) != len(content):
        raise SystemExit(f"row mismatch: {len(hist)} histograms vs {len(content)} encoder rows")

    names = CONTENT_FACTOR_NAMES[: gt_content.shape[1]]
    if adj is not None:
        targets, basis = residualise_on_parents(gt_content, adj), "parent-residualised"
    else:
        targets, basis = gt_content, "raw (factors are i.i.d.)"

    blocks = {
        "content": content,
        "style": style,
        "hist": hist,
        "style+hist": np.concatenate([style, hist], axis=1),
    }
    scores = {k: cv_probe_r2_multi(X, targets, kind=cli.probe)["mean"] for k, X in blocks.items()}
    partial = scores["style+hist"] - scores["hist"]

    print(f"\nrun: {cli.run_dir}{'  [RANDOM INIT]' if cli.random_init else ''}")
    print(
        f"level {cli.level} | pooling {cli.pooling} | N={len(content)} | "
        f"content ch={content.shape[1]} style ch={style.shape[1]} hist dims={hist.shape[1]}"
    )
    print(f"targets: {basis} | probe: {cli.probe}\n")

    print(f"    {'factor':<20} {'content':>9} {'style':>8} {'hist':>8} {'style|hist':>11}")
    for j, name in enumerate(names):
        print(
            f"    {name:<20} {scores['content'][j]:>9.3f} {scores['style'][j]:>8.3f} "
            f"{scores['hist'][j]:>8.3f} {partial[j]:>+11.3f}"
        )

    if not cli.random_init:
        # Measured on this architecture's untrained twin (gap, N=300): style reads brain_size
        # at 0.725 and hist at 0.685 with nothing learned. Every column here has a floor and
        # it is not zero, so an absolute value means little on its own.
        print("\n    Re-run with --random-init and read the GAP: a random projection already scores")
        print("    ~0.7 on brain_size in every block. Only ventricle_size's `style|hist` is safe to")
        print("    read absolutely — it is a difference between two blocks on the same rows.")

    idx = {n: j for j, n in enumerate(names)}
    ctrl = [scores["style"][idx[n]] for n in SHAPE_FACTORS if n in idx]
    ctrl_max = max(ctrl) if ctrl else float("nan")
    print(f"\n    control — best style R² over the {len(ctrl)} volume-preserving factors: {ctrl_max:+.3f}")

    if "ventricle_size" not in idx:
        return
    v = idx["ventricle_size"]
    s_v, p_v = scores["style"][v], partial[v]
    print(f"    ventricle_size: style {s_v:+.3f}, of which {p_v:+.3f} is beyond the histogram")

    if s_v < NOISE_FLOOR:
        print("    => no ventricle leak into style at this checkpoint. Nothing to attribute.")
    elif np.isfinite(ctrl_max) and ctrl_max > s_v - NOISE_FLOOR:
        print("    => style scores about as well on the volume-preserving factors, which the")
        print("       histogram cannot supply. Style is broadly capacious rather than reading the")
        print("       histogram; treat the ventricle result as part of that, not as its own effect.")
    elif p_v < NOISE_FLOOR:
        print("    => style's ventricle knowledge IS the histogram: it adds nothing a shuffle-proof")
        print("       statistic could not already provide. The leak follows the generator, and the")
        print("       lever is style CAPACITY (--style-embed-dim, --style-nb-entries, and the style")
        print("       channel count = vqvae_hidden_channels - content_size), not spatial extent.")
    else:
        print("    => style holds ventricle evidence the histogram cannot supply, so it is carrying")
        print("       spatial structure. The lever is --style-spatial-size; the histogram framing")
        print("       does not explain this checkpoint.")


if __name__ == "__main__":
    main()
