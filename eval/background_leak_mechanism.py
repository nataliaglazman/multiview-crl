#!/usr/bin/env python
"""Background-leak mechanism CLI entrypoint.

The actual analysis lives in [eval/background_leak_shared.py](eval/background_leak_shared.py).
"""

from __future__ import annotations

import argparse
import logging

from eval.background_leak_shared import run_background_leak_mechanism

logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--normalize", default="per_sample", choices=["per_sample", "fixed_reference"])
    ap.add_argument("--rf-margin", type=float, default=1.0)
    ap.add_argument("--clean-content", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_background_leak_mechanism(cli)


if __name__ == "__main__":
    main()
