#!/usr/bin/env python
"""View-leak channel-localization CLI entrypoint.

The actual analysis lives in [eval/view_leak_shared.py](eval/view_leak_shared.py).
"""

from __future__ import annotations

import argparse
import logging

from eval.view_leak_shared import run_view_leak_channels

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Is the view leak localised to a few content channels?")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint-name", default="vqvae_model.pt")
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--causal", choices=("match", "iid"), default="match")
    p.add_argument("--poolings", default="gap", help="gap is where the objective acts; stats is where content_view is.")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", default="view_leak_out")
    cli = p.parse_args()
    run_view_leak_channels(cli)


if __name__ == "__main__":
    main()
