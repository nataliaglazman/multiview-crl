"""Evaluation helpers: val_step, get_data, and eval_step."""

import numpy as np
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from torch.utils.data import DataLoader

import utils.utils as utils
from data.infinite_iterator import InfiniteIterator


def val_step(data, encoders, decoders, loss_func, args, recon_loss_fn=None, moco_loss_func=None):
    """
    Perform a validation step (no gradient updates).

    Thin wrapper around ``train_step`` with ``optimizer=None``.

    Args:
        data: Batch dictionary.
        encoders: List of encoder models.
        decoders: List of decoder models.
        loss_func: InfoNCE loss callable.
        args: Parsed argument namespace.
        recon_loss_fn: Reconstruction loss (optional).
        moco_loss_func: MoCo loss callable (optional).

    Returns:
        Same 5-tuple as ``train_step``.
    """
    # Import here to avoid a circular dependency — train_step lives in main_multimodal.
    from main_multimodal import train_step

    with torch.no_grad():
        return train_step(
            data,
            encoders,
            decoders,
            loss_func,
            optimizer=None,
            params=None,
            args=args,
            recon_loss_fn=recon_loss_fn,
            moco_loss_func=moco_loss_func,
        )


def get_data(
    dataset,
    encoders,
    decoders,
    loss_func,
    dataloader_kwargs,
    num_samples=None,
    args=None,
    recon_loss_fn=None,
    moco_loss_func=None,
):
    """
    Collect encodings and loss values from ``dataset`` for evaluation.

    Args:
        dataset: PyTorch dataset.
        encoders: List of encoder models.
        decoders: List of decoder models.
        loss_func: InfoNCE loss callable.
        dataloader_kwargs: Keyword args forwarded to ``DataLoader``.
        num_samples: Number of samples to collect (``None`` → full dataset).
        args: Parsed argument namespace.
        recon_loss_fn: Reconstruction loss (optional).
        moco_loss_func: MoCo loss callable (optional).

    Returns:
        dict: Keys ``'loss_values'``, ``'content_indices'``, ``'hz_<m>'``,
              ``'labels_<m>'``, ``'hz_<m>_subsets'`` for each modality ``m``.
    """
    loader = DataLoader(dataset, **dataloader_kwargs)
    iterator = InfiniteIterator(loader)

    rdict = {"loss_values": [], "content_indices": []}
    for m in args.modalities:
        rdict[f"hz_{m}"] = []
        rdict[f"labels_{m}"] = {v: [] for v in args.DATASETCLASS.FACTORS[m].values()}
        rdict[f"hz_{m}_subsets"] = {s: [] for s in args.subsets}

    i = 0
    num_samples = num_samples or len(dataset)
    with torch.no_grad():
        while i < num_samples:
            i += loader.batch_size
            data = next(iterator)

            loss_value, _, _, _, estimated_content_indices = val_step(
                data,
                encoders,
                decoders,
                loss_func,
                args=args,
                recon_loss_fn=recon_loss_fn,
                moco_loss_func=moco_loss_func,
            )

            rdict["loss_values"].append([loss_value])

            for m_midx, m in enumerate(args.modalities):
                samples = data[m]
                hz_m = encoders[m_midx](torch.concat(samples, 0)).detach().cpu().numpy()
                rdict[f"hz_{m}"].append(hz_m)

                for k in rdict[f"labels_{m}"]:
                    labels_k = torch.concat([data[f"z_{m}"][i][k] for i in range(len(samples))], 0)
                    rdict[f"labels_{m}"][k].append(labels_k)

                for s_id, s in enumerate(args.subsets):
                    if len(args.subsets) == 1:
                        rdict[f"hz_{m}_subsets"][s].append(hz_m)
                    else:
                        rdict[f"hz_{m}_subsets"][s].append(hz_m[..., estimated_content_indices[s_id]])

            del data
            rdict["content_indices"] += [estimated_content_indices]

    # Concatenate collected batches along the sample axis
    for k, v in rdict.items():
        if isinstance(v, list) and k != "content_indices":
            rdict[k] = np.concatenate(v, axis=0)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                rdict[k][k2] = np.concatenate(v2, axis=0)

    return rdict


def eval_step(ix, subset, modality, factor_name, discrete_factors_m, data, args):
    """
    Fit regression / classification models and return a result row.

    Args:
        ix (int): Factor index.
        subset: View subset identifier.
        modality (str): Modality name (e.g. ``'image'``).
        factor_name (str): Human-readable factor name.
        discrete_factors_m (list[int]): Indices of discrete factors for this modality.
        data (list): ``[train_inputs, train_labels, test_inputs, test_labels]``.
        args: Parsed argument namespace (needs ``args.grid_search_eval``).

    Returns:
        list: ``[subset, ix, modality, factor_name, factor_type,
                 r2_linreg, r2_krreg, acc_logreg, acc_mlp]``
    """
    r2_linreg = r2_krreg = acc_logreg = acc_mlp = np.nan

    factor_type = "discrete" if ix in discrete_factors_m else "continuous"

    if factor_type == "continuous":
        linreg = LinearRegression(n_jobs=-1)
        r2_linreg = utils.evaluate_prediction(linreg, r2_score, *data)
        if args.grid_search_eval:
            gskrreg = GridSearchCV(
                KernelRidge(kernel="rbf", gamma=0.1),
                param_grid={
                    "alpha": [1e0, 0.1, 1e-2, 1e-3],
                    "gamma": np.logspace(-2, 2, 4),
                },
                cv=3,
                n_jobs=-1,
            )
            r2_krreg = utils.evaluate_prediction(gskrreg, r2_score, *data)
        r2_krreg = utils.evaluate_prediction(MLPRegressor(max_iter=1000), r2_score, *data)

    if factor_type == "discrete" and factor_name != "object_zpos":
        logreg = LogisticRegression(n_jobs=-1, max_iter=1000)
        acc_logreg = utils.evaluate_prediction(logreg, accuracy_score, *data)
        mlpreg = MLPClassifier(max_iter=1000)
        acc_mlp = utils.evaluate_prediction(mlpreg, accuracy_score, *data)

    return [
        subset,
        ix,
        modality,
        factor_name,
        factor_type,
        r2_linreg,
        r2_krreg,
        acc_logreg,
        acc_mlp,
    ]
