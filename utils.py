import string
from itertools import chain, combinations
from typing import List

import numpy as np
import torch

import logging
import os
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Orientationd, ResizeWithPadOrCropd,
                              RandRotate90d, ToTensord, RandFlipd, RandAffined, Spacingd, RandZoomd, RandGaussianSmoothd, RandShiftIntensityd,
                              RandSimulateLowResolutiond, NormalizeIntensityd, ThresholdIntensityd, MaskIntensityd, Lambda, Lambdad,
                              CopyItemsd, MapTransform)
import torch
import torch.nn.functional as F
from enum import Enum


class CreateBrainMaskd(MapTransform):
    """Create a binary brain mask from the image (nonzero voxels) before resampling."""
    def __init__(self, keys, mask_keys, threshold=50):
        super().__init__(keys)
        self.mask_keys = mask_keys
        self.threshold = threshold  # Threshold for brain vs background
        
    def __call__(self, data):
        d = dict(data)
        for key, mask_key in zip(self.keys, self.mask_keys):
            # Create mask where image > threshold - handle both numpy and MetaTensor
            img = d[key]
            if hasattr(img, 'cpu'):  # PyTorch tensor
                mask = (img > self.threshold).float()
            else:  # numpy array
                mask = (np.array(img) > self.threshold).astype(np.float32)
            d[mask_key] = mask
        return d


class ApplyBrainMaskd(MapTransform):
    """Apply brain mask to zero out background after normalization."""
    def __init__(self, keys, mask_keys, threshold=0.5):
        super().__init__(keys)
        self.mask_keys = mask_keys
        self.threshold = threshold
        
    def __call__(self, data):
        d = dict(data)
        for key, mask_key in zip(self.keys, self.mask_keys):
            mask = d[mask_key] > self.threshold  # After resampling, threshold at 0.5
            d[key] = d[key] * mask
        return d



EPSILON = np.finfo(np.float32).tiny


class ConfigDict(object):
    def __init__(self, dict) -> None:
        self.dict = dict
        for k, v in dict.items():
            setattr(self, k, v)

    def get(self, key):
        return self.dict.get(key)


# ---- ground truth content style retrievel for numerical simulation -----------
# ------------------------------------------------------------------------------
def powerset(iterable, only_consider_whole_set=False):
    """
    Generate all subsets of views with at least two elements.

    Args:
        iterable: An iterable object containing the views.
        only_consider_whole_set: A boolean indicating whether to consider only the whole subset.

    Returns:
        A tuple containing two lists:
        - The first list contains all subsets of views with at least two elements.
        - The second list contains binary indicators showing whether a specific view is included in each subset.
    """
    s = list(iterable)
    sets = list(chain.from_iterable(combinations(s, r) for r in range(0, len(s) + 1)))
    if only_consider_whole_set:
        ps_leq_2 = [s for s in sets if len(s) == len(list(iterable))]  # if consider the whole subset
    else:
        ps_leq_2 = [s for s in sets if len(s) > 1]
    binary_indicator = [[int(view in s) for view in iterable] for s in ps_leq_2]
    return ps_leq_2, binary_indicator


def retrieve_content_style(zs):
    """
    Retrieve the content and style components from a list of zs.

    Parameters:
    zs (list): List of zs where each zs represents the latent space of a view.

    Returns:
    tuple: A tuple containing the content and style components.
           The content component is a set of common elements across all views.
           The style component is a list of sets, where each set represents the unique elements for each view.
    """
    zs = zs.tolist()
    # zs: shape: [n_views * nz]
    content = set(zs[0])
    for i in range(1, len(zs)):
        content.intersection_update(set(zs[i]))
    style = [set(z_Sk).difference(content) for z_Sk in zs]
    return content, style


def content_style_from_subsets(subsets, zs):
    """
    Retrieve content and style from subsets of zs.

    Args:
        subsets (list): List of subsets.
        zs (numpy.ndarray): Array of zs.

    Returns:
        tuple: A tuple containing two dictionaries - content_dict and style_dict.
            - content_dict: A dictionary mapping each subset to its corresponding content.
            - style_dict: A dictionary mapping each subset to its corresponding style.

    """
    content_dict, style_dict = {}, {}
    for subset in subsets:
        content, style = retrieve_content_style(zs[subset, :])
        if len(content) == 0:
            continue
        else:
            content_dict[subset] = content
            style_dict[subset] = {k: style[i] for i, k in enumerate(subset)}
    return content_dict, style_dict


def unpack_item_list(lst):
    if isinstance(lst, tuple):
        lst = list(lst)
    result_list = []
    for it in lst:
        if isinstance(it, (tuple, list)):
            result_list.append(unpack_item_list(it))
        else:
            result_list.append(it.item())
    return result_list


# ----------- exp-name generator --------------
# ---------------------------------------------


def valid_str(v):
    if hasattr(v, "__name__"):
        return valid_str(v.__name__)
    if isinstance(v, tuple) or isinstance(v, list):
        return "-".join([valid_str(x) for x in v])
    str_v = str(v).lower()
    valid_chars = "-_%s%s" % (string.ascii_letters, string.digits)
    str_v = "".join(c if c in valid_chars else "-" for c in str_v)
    return str_v


def get_exp_name(
    args,
    parser,
    blacklist=["evaluate", "num_train_batches", "num_eval_batches", "evaluate_iter"],
):
    exp_name = ""
    for x in vars(args):
        if getattr(args, x) != parser.get_default(x) and x not in blacklist:
            if isinstance(getattr(args, x), bool):
                exp_name += ("_" + x) if getattr(args, x) else ""
            else:
                exp_name += "_" + x + valid_str(getattr(args, x))
    return exp_name.lstrip("_")


# ----------- content mask-related utils  ----
# ---------------------------------------------
def topk_gumbel_softmax(k, logits, tau, hard=True):
    """
    Applies the top-k Gumbel-Softmax operation to the input logits.

    Args:
        k (int): The number of elements to select from the logits.
        logits (torch.Tensor): The input logits.
        tau (float): The temperature parameter for the Gumbel-Softmax operation.
        hard (bool, optional): Whether to use the straight-through approximation.
            If True, the output will be a one-hot vector. If False, the output will be a
            continuous approximation of the top-k elements. Default is True.

    Returns:
        torch.Tensor: The output tensor after applying the top-k Gumbel-Softmax operation.
    """
    m = torch.distributions.gumbel.Gumbel(torch.zeros_like(logits), torch.ones_like(logits))
    g = m.sample()
    logits = logits + g

    # continuous top k
    khot = torch.zeros_like(logits).type_as(logits)
    onehot_approx = torch.zeros_like(logits).type_as(logits)
    for i in range(k):
        khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).type_as(logits))
        logits = logits + torch.log(khot_mask)
        onehot_approx = torch.nn.functional.softmax(logits / tau, dim=1)
        khot = khot + onehot_approx

    if hard:
        # straight through
        khot_hard = torch.zeros_like(khot)
        val, ind = torch.topk(khot, k, dim=1)
        khot_hard = khot_hard.scatter_(1, ind, 1)
        res = khot_hard - khot.detach() + khot
    else:
        res = khot
    return res


def mask2indices(masks, keys):
    """
    Convert binary masks to indices of non-zero elements for each key.

    Args:
        masks (list): List of binary masks.
        keys (list): List of keys corresponding to the masks.

    Returns:
        dict: Dictionary mapping each key to a list of indices of non-zero elements in the corresponding mask.
    """
    estimated_content_indices = {}
    assert len(keys) == len(masks)
    for k, c_mask in zip(keys, masks):
        c_ind = torch.where(c_mask)[-1].tolist()
        estimated_content_indices[k] = c_ind
    return estimated_content_indices


def gumbel_softmax_mask(avg_logits: torch.Tensor, subsets: List, content_sizes: List):
    """
    Applies the Gumbel-Softmax function to generate masks for each subset.

    Args:
        avg_logits (torch.Tensor): The average logits for each subset.
        subsets (List): The list of subsets.
        conten_sizes (List): The list of content sizes for each subset.

    Returns:
        List: The list of masks generated using Gumbel-Softmax for each subset.
    """
    masks = []
    for i, subset in enumerate(subsets):
        m = topk_gumbel_softmax(k=content_sizes[i], logits=avg_logits, tau=1.0, hard=True)
        masks += [m]
    return masks


def smart_gumbel_softmax_mask(avg_logits: torch.Tensor, subsets: List, content_sizes: List):
    """
    Generates masks using smart Gumbel softmax for each subset.

    Args:
        avg_logits (torch.Tensor): Average logits.
        subsets (List): List of subsets.
        conten_sizes (List): List of content sizes.

    Returns:
        List: List of masks for each subset.
    """
    masks = []
    joint_content_size = content_sizes[-1]
    joint_content_mask = torch.eye(avg_logits.shape[-1])[:2].type_as(avg_logits)

    for i, subset in enumerate(subsets[:-1]):
        m = topk_gumbel_softmax(
            k=content_sizes[i] - joint_content_size,
            logits=avg_logits,
            tau=1.0,
            hard=True,
        )
        m = torch.concat([joint_content_mask, m], 0)
        masks += [m]
    return masks


# ----------- Evaluation-related utils ------------
# -------------------------------------------------
def evaluate_prediction(model, metric, X_train, y_train, X_test, y_test):
    """
    Evaluates the performance of a model by fitting it on training data, predicting on test data,
    and calculating a specified metric between the predicted and true labels.

    Parameters:
        model (object): The machine learning model to be evaluated.
        metric (function): The evaluation metric to be used.
        X_train (array-like): The training input samples.
        y_train (array-like): The training target values.
        X_test (array-like): The test input samples.
        y_test (array-like): The test target values.

    Returns:
        float: The evaluation score calculated using the specified metric.
    """
    # handle edge cases when inputs or labels are zero-dimensional
    if any([0 in x.shape for x in [X_train, y_train, X_test, y_test]]):
        return np.nan
    assert X_train.shape[1] == X_test.shape[1]
    if y_train.ndim > 1:
        assert y_train.shape[1] == y_test.shape[1]
        if y_train.shape[1] == 1:
            y_train = y_train.ravel()
            y_test = y_test.ravel()
    # handle edge cases when the inputs are one-dimensional
    if X_train.shape[1] == 1:
        X_train = X_train.reshape(-1, 1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metric(y_test, y_pred)


def generate_batch_factor_code(ground_truth_data, representation_function, num_points, random_state, batch_size):
    """Sample a single training sample based on a mini-batch of ground-truth data.

    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_function: Function that takes observation as input and
        outputs a representation.
      num_points: Number of points to sample.
      random_state: Numpy random state used for randomness.
      batch_size: Batchsize to sample points.

    Returns:
      representations: Codes (num_codes, num_points)-np array.
      factors: Factors generating the codes (num_factors, num_points)-np array.
    """
    representations = None
    factors = None
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_factors, current_observations = ground_truth_data.sample(num_points_iter, random_state)
        if i == 0:
            factors = current_factors
            representations = representation_function(current_observations)
        else:
            factors = np.vstack((factors, current_factors))
            representations = np.vstack((representations, representation_function(current_observations)))
        i += num_points_iter
    return np.transpose(representations), np.transpose(factors)

def load_data(df_filtered, data_dir, label_map):
    exts = ['.nii.gz', '.nii', '.mha', '.mhd', '.nrrd', '.npy']
    missing = []
    items = []
    for _, row in df_filtered.iterrows():
        subj = str(row['Subject'])
        found_t1 = None
        found_t2 = None
        
        # Find T1 image
        for ext in exts:
            candidate = os.path.join(data_dir, subj, 't1')
            if os.path.exists(candidate):
                candidate_files = os.listdir(candidate)
                for file in candidate_files:
                    if file.endswith(ext):
                        found_t1 = os.path.join(candidate, file)
                        break
            if found_t1:
                break
        
        # Find T2 image
        for ext in exts:
            candidate = os.path.join(data_dir, subj, 't2')
            if os.path.exists(candidate):
                candidate_files = os.listdir(candidate)
                for file in candidate_files:
                    if file.endswith(ext) and 'FLAIR' in file:
                        found_t2 = os.path.join(candidate, file)
                        break
            if found_t2:
                break
        
        # Only include subjects that have BOTH T1 and T2
        if found_t1 and found_t2:
            items.append({
                'image': found_t1,
                'z_image': found_t2,
                'label': label_map[row['Group']]
            })
        else:
            missing.append(subj)
            if not found_t1:
                logging.warning(f"Missing T1 for subject {subj}")
            if not found_t2:
                logging.warning(f"Missing T2 for subject {subj}")
    
    logging.info(f"Loaded {len(items)} subjects with both T1 and T2. Missing: {len(missing)}")
    return items, missing


def transforms(spacing=2.0):
    """
    Create training and validation transforms for brain MRI images.
    
    Args:
        spacing: Isotropic voxel spacing in mm.
                 - 1.0: Original resolution (~182x218x182)
                 - 2.0: Downsampled (~91x109x91)
    
    Returns:
        train_transforms, val_transforms
    """
    # Calculate spatial size based on spacing
    # Original 1mm images are approximately 182x218x182
    if spacing == 1.0:
        spatial_size = (182, 218, 182)
    elif spacing == 2.0:
        spatial_size = (91, 109, 91)
    else:
        # Calculate proportionally from 1mm reference
        spatial_size = tuple(int(s / spacing) for s in (182, 218, 182))
    
    logging.info(f"Using voxel spacing: {spacing}mm, spatial size: {spatial_size}")
    
    # Common transforms list builder
    def build_transforms(is_training=False):
        transforms_list = [
            LoadImaged(keys=['image_t1', 'image_t2']),
            EnsureChannelFirstd(keys=['image_t1', 'image_t2'], channel_dim="no_channel"),
            # Create brain mask BEFORE resampling (where original > 0)
            CreateBrainMaskd(keys=['image_t1', 'image_t2'], mask_keys=['mask_t1', 'mask_t2']),
        ]
        
        # Only add spacing transforms if not using original 1mm
        if spacing != 1.0:
            transforms_list.extend([
                Spacingd(keys=['image_t1', 'image_t2'], pixdim=(spacing, spacing, spacing), mode="bilinear"),
                Spacingd(keys=['mask_t1', 'mask_t2'], pixdim=(spacing, spacing, spacing), mode="nearest"),
            ])
        
        transforms_list.extend([
            Orientationd(keys=['image_t1', 'image_t2', 'mask_t1', 'mask_t2'], axcodes="RAS"),
            ResizeWithPadOrCropd(keys=['image_t1', 'image_t2', 'mask_t1', 'mask_t2'], spatial_size=spatial_size),
            NormalizeIntensityd(keys=['image_t1', 'image_t2'], nonzero=True, channel_wise=True),
            ApplyBrainMaskd(keys=['image_t1', 'image_t2'], mask_keys=['mask_t1', 'mask_t2'], threshold=0.5),
        ])
        
        # Add augmentations for training only
        if is_training:
            transforms_list.extend([
                RandAffined(keys=['image_t1', 'image_t2'], 
                            rotate_range=[-0.05, 0.05],
                            shear_range=[0.001, 0.05], 
                            scale_range=[0, 0.05],
                            mode='bilinear',
                            padding_mode='zeros',
                            prob=0.5),
                RandShiftIntensityd(keys=['image_t1', 'image_t2'], offsets=(-0.1, 0.1), prob=0.2),
            ])
        
        transforms_list.append(ToTensord(keys=['image_t1', 'image_t2', 'label']))
        return Compose(transforms_list)
    
    train_transforms = build_transforms(is_training=True)
    val_transforms = build_transforms(is_training=False)
    
    return train_transforms, val_transforms

class TBSummaryTypes(Enum):
    SCALAR = "scalar"
    SCALARS = "scalars"
    HISTOGRAM = "histogram"
    IMAGE = "image"
    IMAGE_AXIAL = "image_axial"
    IMAGE_CORONAL = "image_coronal"
    IMAGE_SAGITTAL = "image_sagittal"
    IMAGE3_AXIAL = "image3_axial"
    IMAGE3_CORONAL = "image3_coronal"
    IMAGE3_SAGITTAL = "image3_sagittal"
    IMAGE3 = "image3"
    IMAGES = "images"
    IMAGE_WITH_BOXES = "image_with_boxes"
    FIGURE = "figure"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
