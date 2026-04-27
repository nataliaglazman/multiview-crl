import torch
import numpy as np
import monai
import nibabel as nib
vol = torch.randn(1, 1, 91, 109, 91).cpu().float().numpy()
while vol.ndim > 3: vol = vol[0]
aff = np.eye(4)
img = nib.Nifti1Image(vol, aff)
print("Success")
