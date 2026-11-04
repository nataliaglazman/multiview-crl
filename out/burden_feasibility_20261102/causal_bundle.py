"""A warm.py run's code on a causal bank's test subjects, as a bundle in the shared format, for the repo's causal
discovery: eval.causal.latent_causal_discovery (PC on the code's units, so no label builds the graph) and
eval.protocol.compare_bundles --with-graph (PC on factors decoded from the code).

    python out/burden_feasibility_20261102/causal_bundle.py RUN OUT.npz [--untrained] [--features code|gap|pooled]

The factors are the seven the renderer draws (bank.CAUSAL_FACTORS; z_content[3:5] are unused with the burden target)
and causal_adj is the bank's SCM restricted to them. emb_view1 is the T1 code, emb_view2 the FLAIR code.
--untrained keeps the run's architecture and its init but none of its training: the floor. Its TailCode statistics and
gauges are first filled from training subjects, so the floor is a calibrated random code rather than an unscaled one.
--features gap or pooled exports the backbone GAP (64) or TailCode's pooled features (the head's input) in place of
the code, for decoded discovery past the head; label-free discovery is meant for the code's units only.
"""

import argparse
import json
from pathlib import Path

import bank as B
import code_audit
import numpy as np
import torch
import warm

from eval.metrics.bundle_identity import identity_record


@torch.no_grad()
def calibrate(model, pool, train, device, subjects=960, size=32):
    """Fill an untrained TailCode's per-cell statistics and its gauges from training subjects, as cumulative averages,
    without touching a weight."""
    for gauge in pool.gauges:
        if isinstance(gauge, torch.nn.BatchNorm1d):
            gauge.reset_running_stats()
            gauge.momentum = None
    pool.train()
    for i, start in enumerate(range(0, min(subjects, len(train.vox)), size)):
        pool.momentum = 1 / (i + 1)
        for gauge in pool.gauges:
            if not isinstance(gauge, torch.nn.BatchNorm1d):
                gauge.momentum = 1 / (i + 1)
        x = train.images(np.arange(start, min(start + size, len(train.vox))), device)
        pool.pooled(model._encode(x, 2, None))
    pool.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--untrained", action="store_true", help="The run's architecture at its own init: the floor")
    p.add_argument("--features", choices=("code", "gap", "pooled"), default="code", help="What each view exports")
    a = p.parse_args()
    device = warm.pick_device()
    report, model, pool, residual, step = code_audit.load_run(a.run, device, trained=not a.untrained)
    args = report["args"]
    root, t1key = args["bank"], args["t1key"]
    if not Path(f"{root}/causal_adj.npy").exists():
        raise SystemExit(f"{root} has no causal_adj.npy: render it with bank.py --causal")
    if pool is None:
        raise SystemExit("The code comes from TailCode's head: the run needs --pool tail or gapnorm")
    if a.untrained:
        calibrate(model, pool, warm.Bank(root, "train", t1key, residual=residual), device)
    test = warm.Bank(root, "test", t1key, int(args.get("test_subjects", 1000)), residual)
    codes = warm.features(model, test, device, pool=pool)[a.features]
    columns = list(B.CAUSAL_FACTORS)
    z = np.asarray(test.zc[:, columns], dtype=np.float32)
    adj = np.load(f"{root}/causal_adj.npy")[np.ix_(columns, columns)].astype(np.float32)
    settings = {"bank": root, "t1key": t1key, "causal_edges": B.CAUSAL_EDGES, "test_subjects": len(z)}
    meta = {
        "content_factor_names": list(B.CAUSAL_FACTORS.values()),
        "views": ["1", "2"],
        "source_run": str(a.run),
        "step": step,
        "untrained": a.untrained,
        "features": a.features,
        **identity_record({"z_content": z, "causal_adj": adj}, settings, len(z), "match"),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        z_content=z,
        causal_adj=adj,
        emb_view1=codes[0].astype(np.float32),
        emb_view2=codes[1].astype(np.float32),
        meta=json.dumps(meta, indent=2),
    )
    print("BUNDLE", a.out, "rows", len(z), a.features, codes[0].shape[1], "step", step, "untrained", a.untrained)


if __name__ == "__main__":
    main()
