"""Score a saved checkpoint of a finished (or running) run with the leakage evaluation.

    python -m mlmv.eval.evaluate_run --run-dir results/mlmv/p1_exact [--checkpoint final.pt] [--set eval.max_subjects=500]

Rebuilds the run from its own files: ``config.yaml`` (the resolved config it trained with), ``geometry/``
(the eroded regions in use at the end of training) and the checkpoint's model weights (``final.pt``,
``last.pt`` or a kept ``step_<N>.pt``). Readouts stay inside the eroded regions L and G are trained on
(``mlmv.eval.leakage.readout_masks``). Writes ``leakage.csv``, ``checks.csv``, ``counterfactual.csv`` and
``leakage.json`` to ``<run-dir>/eval/<checkpoint name>/``, and prints the matrices.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from mlmv.config import load_config
from mlmv.data.geometry import Geometry
from mlmv.eval.leakage import MODALITIES, evaluate_leakage, format_matrix
from mlmv.models.encoder import MultiLevelModel, Template, build_encoder
from mlmv.train.train_levels import make_source, pick_device


def load_run_config(run_dir, overrides=None):
    """The run's resolved config (keys added since it ran keep their current defaults), plus ``overrides``."""
    raw = yaml.safe_load((Path(run_dir) / "config.yaml").read_text())
    raw.pop("_commit", None)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run.yaml"
        path.write_text(yaml.safe_dump(raw))
        return load_config(path, overrides=overrides)


def load_geometry(run_dir) -> Geometry:
    d = Path(run_dir) / "geometry"
    masks = np.load(d / "region_masks.npz")
    meta = json.loads((d / "geometry.json").read_text())
    return Geometry(
        labels=masks["labels"],
        eroded=masks["eroded"],
        rho=meta["rho"],
        metric=meta["metric"],
        min_positions_per_half=meta["min_positions_per_half"],
        eligible=meta["eligible"],
        merges=meta.get("merges", []),
    )


def evaluate_run(run_dir, checkpoint: str = "final.pt", overrides=None, log=print) -> dict:
    run_dir = Path(run_dir)
    cfg = load_run_config(run_dir, overrides)
    device = pick_device(cfg.run.device)
    ckpt = Path(checkpoint) if Path(checkpoint).exists() else run_dir / "checkpoints" / checkpoint
    geometry = load_geometry(run_dir)
    encoder = build_encoder(cfg.model)
    template = Template(
        sum(encoder.dims.values()), geometry.shape, cfg.model.template_momentum, cfg.model.template_subtraction
    )
    model = MultiLevelModel(encoder, template).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    source = make_source(cfg, encoder.stride, device)
    out_dir = run_dir / "eval" / ckpt.stem
    log(f"scoring {ckpt} (step {state.get('step', 'final')}, rho {geometry.rho}) into {out_dir}")
    result = evaluate_leakage(model, source, cfg, out_dir, split="test", device=str(device), geometry=geometry)
    for m in MODALITIES:
        log(f"leakage matrix ({m}, best R^2):\n{format_matrix(result['rows'], m)}")
    failed = [c for c in result["checks"] if not c["passed"]]
    log(f"pass checks: {len(result['checks']) - len(failed)}/{len(result['checks'])} passed")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", required=True, help="e.g. results/mlmv/p1_exact")
    p.add_argument("--checkpoint", default="final.pt", help="file in <run-dir>/checkpoints/, or a path")
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. eval.max_subjects=500")
    args = p.parse_args(argv)
    evaluate_run(args.run_dir, args.checkpoint, args.set)


if __name__ == "__main__":
    main()
