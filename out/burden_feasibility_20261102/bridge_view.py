"""The bridge's first N code units as a run of their own, so every audit reads them unchanged. No training.

A run trained with warm.py --entropy-units N (the bridge: alignment on all its code units, InfoNCE's entropy on the
first N) should hold, in theory, a bijection of the shared latents in those N units. This writes RUN_firstN beside RUN:
the same checkpoint with TailCode's head cut to the first N code outputs (and the spare outputs every head carries
after the code), and RUN's report with --content-units N, so an audit rebuilds a head of that width and exports
exactly those units. The prediction maps, which read all units, are left out.

    python out/burden_feasibility_20261102/bridge_view.py RUN [--units N]
"""

import argparse
import json
from pathlib import Path

import torch


def first_units(state, units, width):
    """TailCode's state with its head's last layer cut to the first ``units`` code outputs and the outputs after the
    code's ``width``."""
    bias = state["head.2.bias"]
    keep = torch.cat([torch.arange(units), torch.arange(width, len(bias))])
    out = dict(state)
    out["head.2.weight"] = state["head.2.weight"][keep].clone()
    out["head.2.bias"] = bias[keep].clone()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--units", type=int, default=None, help="how many leading units (default the run's --entropy-units)")
    a = p.parse_args()
    report = json.loads((a.run / "report.json").read_text())
    args = report["args"]
    if args.get("content_units", "None") == "None" or args.get("remainder_units", "None") != "None":
        raise SystemExit(f"{a.run.name}: needs --content-units and no --remainder-units")
    width = int(args["content_units"])
    units = a.units if a.units is not None else int(args.get("entropy_units", "None").replace("None", "0"))
    if not 1 <= units < width:
        raise SystemExit(f"{a.run.name}: --units must be in 1..{width - 1} (or the run trained with --entropy-units)")
    state = torch.load(a.run / "model.pt", map_location="cpu")
    state["pool"] = first_units(state["pool"], units, width)
    state.pop("predict", None)
    out = a.run.parent / f"{a.run.name}_first{units}"
    out.mkdir(exist_ok=True)
    torch.save(state, out / "model.pt")
    report["args"] = dict(args, content_units=str(units), entropy_units="None")
    report["view"] = f"the first {units} code units of {a.run.name}"
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{out}: the first {units} of {width} code units, step {state['step']}")


if __name__ == "__main__":
    main()
