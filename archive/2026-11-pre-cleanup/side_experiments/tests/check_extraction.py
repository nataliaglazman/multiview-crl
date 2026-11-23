import sys
import os
import torch
import numpy as np
import pandas as pd
import json

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd
from monai.transforms import ResizeWithPadOrCropd, NormalizeIntensityd, ToTensord
import models.vqvae as vqvae
from utils.utils import load_data, CreateBrainMaskd, ApplyBrainMaskd

CHECKPOINT_PATH = "/home/ng24/projects/multiview-crl/results/ADNI_registered/multiview-05-content-lr-002-all-levels/vqvae_best.pt"
DATA_DIR = "/data/natalia/ADNI_stripped"
CSV_PATH = "/home/ng24/projects/nmpevqvae/labels_cleaned_3class.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(os.path.join(os.path.dirname(CHECKPOINT_PATH), "settings.json"), "r") as f:
    settings = json.load(f)

df = pd.read_csv(CSV_PATH)
label_values = sorted(df["Group"].unique())
label_map = {v: i for i, v in enumerate(label_values)}

items, missing = load_data(df, DATA_DIR, label_map)

# Model setup from the notebook
vqvae_model = vqvae.VQVAE(
    in_channels=1,
    hidden_channels=settings["vqvae_hidden_channels"],
    res_channels=settings.get("vqvae_res_channels", 32),
    nb_res_layers=settings.get("vqvae_nb_res_layers", 2),
    nb_levels=settings["vqvae_nb_levels"],
    embed_dim=settings["vqvae_embed_dim"],
    nb_entries=settings["vqvae_nb_entries"],
    scaling_rates=settings["vqvae_scaling_rates"],
    content_size=256,
    style_size=256,
    inject_style_to_decoder=settings.get("inject_style_to_decoder", False),
    content_style_levels=settings.get("content_style_levels", [0]),
    separate_encoders=False, 
    mask_mode="learned",
)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
sd = {k.replace("module.", ""): v for k, v in checkpoint["encoders"].items() if "momentum" not in k}
vqvae_model.load_state_dict(sd, strict=False)
vqvae_model.eval()
vqvae_model.to(DEVICE)

base_keys = ["image_t1", "image_t2"]
mask_keys = ["mask_t1", "mask_t2"]
all_keys = base_keys + mask_keys

val_transforms = Compose([
    LoadImaged(keys=base_keys),
    EnsureChannelFirstd(keys=base_keys, channel_dim="no_channel"),
    CreateBrainMaskd(keys=base_keys, mask_keys=mask_keys),
    Orientationd(keys=all_keys, axcodes="RAS"),
    ResizeWithPadOrCropd(keys=all_keys, spatial_size=(182, 218, 182)),
    NormalizeIntensityd(keys=base_keys, nonzero=True, channel_wise=True),
    ApplyBrainMaskd(keys=base_keys, mask_keys=mask_keys, threshold=0.5),
    ToTensord(keys=base_keys),
])

all_features = {"level_0": []}
for idx, item in enumerate(items[:2]):
    data_dict = {"image_t1": item["image"], "image_t2": item["z_image"]}
    transformed = val_transforms(data_dict)

    for modality, key in [("T1", "image_t1"), ("T2", "image_t2")]:
        img = transformed[key].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _, _, enc_features, _, _, _, soft_masks, *_ = vqvae_model(
                img,
                return_recon=False,
                pool_only=True,
                view_idx=0,
            )
            all_features["level_0"].append(enc_features[0].squeeze(0).cpu().float().numpy())

feats = np.array(all_features["level_0"])
t1_f = feats[[0, 2]]
t2_f = feats[[1, 3]]
print("T1 subj 0 vs T1 subj 1:", np.linalg.norm(t1_f[0] - t1_f[1]))
print("T2 subj 0 vs T2 subj 1:", np.linalg.norm(t2_f[0] - t2_f[1]))
print("Paired distances:", paired_distances)
print("Features T1 avg:", t1_f.mean(axis=1))
print("Features T2 avg:", t2_f.mean(axis=1))
