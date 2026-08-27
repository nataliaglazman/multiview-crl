#!/usr/bin/env python
"""View-asymmetry probe CLI entrypoint.

The actual analysis lives in [eval/view_leak_shared.py](eval/view_leak_shared.py).
"""

from __future__ import annotations

import argparse
import logging

from eval.view_leak_shared import run_view_asymmetry_probe

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Per-view factor ceilings from raw voxels.")
    p.add_argument("--run-dir", required=True, help="Run directory (its settings define the dataset).")
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--causal", choices=("match", "iid"), default="match")
    p.add_argument("--grids", default="16,8,4", help="Pooling grids to sweep (a lesion is local: 8-16 is the range).")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", default="view_asymmetry_out")
    cli = p.parse_args()
    run_view_asymmetry_probe(cli)


if __name__ == "__main__":
    main()
