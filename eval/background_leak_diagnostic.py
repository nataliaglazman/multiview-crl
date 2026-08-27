#!/usr/bin/env python
"""Background-leak diagnostic CLI entrypoint.

The actual analysis lives in [eval/background_leak_shared.py](eval/background_leak_shared.py).
"""

from __future__ import annotations

import argparse
import logging

from eval.background_leak_shared import run_background_leak_diagnostic

logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--factor", default="ventricle_size")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--grid", type=int, default=None)
    ap.add_argument("--num-samples", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--core-thr", type=float, default=0.9)
    ap.add_argument("--bg-thr", type=float, default=0.1)
    ap.add_argument("--device", default=None)
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_background_leak_diagnostic(cli)


if __name__ == "__main__":
    main()
