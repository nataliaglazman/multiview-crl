import json
import pandas as pd
import numpy as np
from utils.utils import load_data

CSV_PATH = "/home/ng24/projects/nmpevqvae/labels_cleaned_3class.csv"
DATA_DIR = "/data/natalia/ADNI_stripped"

df = pd.read_csv(CSV_PATH)
label_values = sorted(df["Group"].unique())
label_map = {v: i for i, v in enumerate(label_values)}

items, missing = load_data(df, DATA_DIR, label_map)

for i in range(3):
    print("Item", i)
    print("  image:", items[i].get("image"))
    print("  z_image:", items[i].get("z_image"))

