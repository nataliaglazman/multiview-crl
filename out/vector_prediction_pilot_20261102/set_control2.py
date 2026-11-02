"""Candidates on the pilot's real trained map and regional cells, against its untrained floor."""
import importlib.util
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

torch.set_num_threads(4)
from eval.encoder.burden_recovery import encode, subject_table
from eval.encoder.regional_lesion_detection import lesion_fractions
from eval.protocol.score_checkpoint import build_model, load_settings, make_dataset

S = sys.argv[1]
sys.argv = ["x", S]
spec = importlib.util.spec_from_file_location("sc", os.path.join(S, "set_control_fns.py"))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
d = os.path.join(S, "pilot_eval", "predcode")
cfg = load_settings(d)
cache = os.path.join(S, "pilot_eval", "cells.npz")
grid = [16, 16, 16]
data = make_dataset(cfg, 400, mode="val")
factors, kept = subject_table(data, grid, 16, 0.05)
y = factors[:, 2]
vol = lesion_fractions(data, (1, 1, 1))[:, 0] * 64**3
small = vol < np.percentile(vol, 33.3)
for stage, name in (("trained", "model.pt"), ("floor", "model_init.pt")):
    model = build_model(cfg, "cpu", torch.load(os.path.join(d, name), map_location="cpu")).eval()
    rep = encode(model, grid, data, "cpu", 16)
    del model
    for view in ("t1", "flair"):
        for readout in ("map", "regional"):
            cells = rep[view][readout][:, :, kept]
            print(f"{stage} {view} {readout} {cells.shape}", flush=True)
            sc.suite(cells, y, small, set_probe=False)
