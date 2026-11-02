import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin_common

import training.main_conv_synthetic as T

out, ckpt = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
args = twin_common.run_args()
json.dump(vars(args), open(f"{out}/settings.json", "w"), default=str)
torch.manual_seed(args.model_seed)
torch.save(T.build_encoder(args).state_dict(), f"{out}/model_init.pt")
torch.save(torch.load(ckpt, map_location="cpu")["model"], f"{out}/model.pt")
print("ok", sorted(os.listdir(out)))
