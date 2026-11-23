import json
import pandas as pd
import numpy as np
import torch
import sys
import os
from utils.utils import load_data

CSV_PATH = "/home/ng24/projects/nmpevqvae/labels_cleaned_3class.csv"
DATA_DIR = "/data/natalia/ADNI_stripped"

df = pd.read_csv(CSV_PATH)
label_values = sorted(df["Group"].unique())
label_map = {v: i for i, v in enumerate(label_values)}

items, missing = load_data(df, DATA_DIR, label_map)

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd
from monai.transforms import ResizeWithPadOrCropd, NormalizeIntensityd, ToTensord
from utils.utils import CreateBrainMaskd, ApplyBrainMaskd

base_keys = ["image_t1", "image_t2"]
mask_keys = ["mask_t1", "mask_t2"]
all_keys = base_keys + mask_keys

pre = [
    LoadImaged(keys=base_keys),
    EnsureChannelFirstd(keys=base_keys, channel_dim="no_channel"),
    CreateBrainMaskd(keys=base_keys, mask_keys=mask_keys),
    Orientationd(keys=all_keys, axcodes="RAS"),
]

val_transforms = Compose(
    pre
    + [
        ResizeWithPadOrCropd(keys=all_keys, spatial_size=(182, 218, 182)),
        NormalizeIntensityd(keys=base_keys, nonzero=True, channel_wise=True),
        ApplyBrainMaskd(keys=base_keys, mask_keys=mask_keys, threshold=0.5),
        ToTensord(keys=base_keys),
    ]
)

for i in range(3):
    item = items[i]
    data_dict = {"image_t1": item["image"], "image_t2": item["z_image"]}
    transformed = val_transforms(data_dict)
    img_t1 = transformed["image_t1"]
    img_t2 = transformed["image_t2"]
    
    diff = torch.norm(img_t1 - img_t2).item()
    print(f"Item {i}: T1 max {img_t1.max().item():.3f}, T2 max {img_t2.max().item():.3f}, L2 diff = {diff:.4f}")
    print(f"  T1 path: {item['image']}")
    print(f"  T2 path: {item['z_image']}")

