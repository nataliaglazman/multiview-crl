import sys, os, torch, numpy as np, pandas as pd, json
import models.vqvae as vqvae
from utils.utils import load_data
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd
from monai.transforms import ResizeWithPadOrCropd, NormalizeIntensityd, ToTensord
from utils.utils import CreateBrainMaskd, ApplyBrainMaskd

CHECKPOINT_PATH = "/home/ng24/projects/multiview-crl/results/ADNI_registered/multiview-05-content-lr-002-all-levels/vqvae_best.pt"
DATA_DIR = "/data/natalia/ADNI_stripped"
CSV_PATH = "/home/ng24/projects/nmpevqvae/labels_cleaned_3class.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open(os.path.join(os.path.dirname(CHECKPOINT_PATH), "settings.json"), "r") as f: settings = json.load(f)
df = pd.read_csv(CSV_PATH)
label_map = {v: i for i, v in enumerate(sorted(df["Group"].unique()))}
items, _ = load_data(df, DATA_DIR, label_map)

val_transforms = Compose([
    LoadImaged(keys=["image_t1"]), EnsureChannelFirstd(keys=["image_t1"], channel_dim="no_channel"),
    CreateBrainMaskd(keys=["image_t1"], mask_keys=["mask_t1"]), Orientationd(keys=["image_t1", "mask_t1"], axcodes="RAS"),
    ResizeWithPadOrCropd(keys=["image_t1", "mask_t1"], spatial_size=(182, 218, 182)),
    NormalizeIntensityd(keys=["image_t1"], nonzero=True, channel_wise=True),
    ApplyBrainMaskd(keys=["image_t1"], mask_keys=["mask_t1"], threshold=0.5), ToTensord(keys=["image_t1"])
])

vqvae_model = vqvae.VQVAE(
    in_channels=1, hidden_channels=settings["vqvae_hidden_channels"], res_channels=settings.get("vqvae_res_channels", 32),
    nb_res_layers=settings.get("vqvae_nb_res_layers", 2), nb_levels=settings["vqvae_nb_levels"],
    embed_dim=settings["vqvae_embed_dim"], nb_entries=settings["vqvae_nb_entries"], scaling_rates=settings["vqvae_scaling_rates"],
    content_size=256, style_size=256, inject_style_to_decoder=settings.get("inject_style_to_decoder", False),
    content_style_levels=settings.get("content_style_levels", [0]), separate_encoders=False, mask_mode="learned"
)
sd = {k.replace("module.", ""): v for k, v in torch.load(CHECKPOINT_PATH, map_location=DEVICE)["encoders"].items() if "momentum" not in k}
vqvae_model.load_state_dict(sd, strict=False)
vqvae_model.eval()
vqvae_model.to(DEVICE)

img = val_transforms({"image_t1": items[0]["image"]})["image_t1"].unsqueeze(0).to(DEVICE)

# 1. pool_only = True
with torch.no_grad(): _, _, enc_features, _, _, _, _, _ = vqvae_model(img, return_recon=False, pool_only=True, view_idx=0)
pool_val_1 = enc_features[0].squeeze(0).cpu().numpy()

# 2. direct from encoder
with torch.no_grad():
    spatial = vqvae_model.encoders[0](img)
    pool_val_2 = spatial.mean(dim=[2,3,4]).squeeze(0).cpu().numpy()

# 3. pool_only = False
with torch.no_grad(): _, _, enc_features_f, _, _, _, _, _ = vqvae_model(img, return_recon=False, pool_only=False, view_idx=0)
pool_val_3 = 0

print("pool_only=True mean:", pool_val_1.mean(), "std:", pool_val_1.std())
print("direct encoder mean:", pool_val_2.mean(), "std:", pool_val_2.std())
print("pool_only=False mean:", pool_val_3.mean(), "std:", pool_val_3.std())
print("Diff 1 vs 2:", np.linalg.norm(pool_val_1 - pool_val_2))
