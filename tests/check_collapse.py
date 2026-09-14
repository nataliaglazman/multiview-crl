import numpy as np

feats = np.load('all_features.npy', allow_pickle=True) if 0 else None
# Wait, I didn't save all_features. Let's just modify check_extraction to print T1 vs T1 distance.
