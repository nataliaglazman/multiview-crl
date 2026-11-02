"""A run directory for the T1-0.4 cell-code pilot's checkpoint, so burden_recovery can read real trained features."""
import json
import os
import sys

import torch

from eval.protocol.score_checkpoint import build_model
from training import main_conv_synthetic as T

S = sys.argv[1]
flags = (
    open(os.path.join(S, "flags.txt"))
    .read()
    .replace("--synthetic-lesion-t1-value 0.0", "--synthetic-lesion-t1-value 0.4")
    .split()
)
args = T.parse_args(flags + ["--vector-prediction-weight", "0", "--model-id", "predcode", "--device", "cpu"])
cfg = vars(args)
cfg["cpu_threads"] = 4
d = os.path.join(S, "pilot_eval", "predcode")
os.makedirs(d, exist_ok=True)
json.dump(cfg, open(os.path.join(d, "settings.json"), "w"), indent=2)
raw = torch.load("out/regional_pilot_20261028/predcode_sepadam_s4000.pt", map_location="cpu", weights_only=False)
print(type(raw), list(raw)[:6] if isinstance(raw, dict) else "")
state = raw.get("model", raw.get("state_dict", raw))
init = build_model(cfg, "cpu")
own = init.state_dict()
kept = {k: v for k, v in state.items() if k in own}
print("kept", len(kept), "of", len(state), "needed", len(own), "dropped", [k for k in state if k not in own][:6])
assert set(kept) == set(own)
torch.save(kept, os.path.join(d, "model.pt"))
torch.save(own, os.path.join(d, "model_init.pt"))
json.dump({"step": 4000, "status": "complete"}, open(os.path.join(d, "training_progress.json"), "w"))
