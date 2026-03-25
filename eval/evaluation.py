"""Evaluation helpers: val_step, get_data, and eval_step."""

import numpy as np
import torch
import torch.nn as nn
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

    Models are temporarily switched to ``eval()`` mode so that:
    - BatchNorm uses its running statistics (not per-batch stats).
    - ``CodeLayer`` skips the EMA codebook update (``if self.training:``),
      preventing codebook corruption from validation batches.
    - ``VQVAE.forward()`` uses the deterministic top-k mask instead of the
      stochastic Gumbel-Softmax mask, giving consistent content indices across
      all evaluation batches.

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

    all_models = list(encoders) + list(decoders)

    # Record which models were in training mode so we can restore them afterwards.
    was_training = [m.training for m in all_models]

    try:
        for m in all_models:
            m.eval()

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
    finally:
        # Always restore the original training/eval state, even on exception.
        for m, was_train in zip(all_models, was_training):
            m.train(was_train)


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
                enc_input = torch.concat(samples, 0)
                # For VQVAE / MoCoEncoder, pass n_views so that separate
                # encoders (if enabled) route each view through its own stack.
                _is_vqvae = isinstance(enc_out_raw := encoders[m_midx], nn.Module) and (
                    hasattr(enc_out_raw, "online")
                    or hasattr(enc_out_raw, "encoders")
                    or (hasattr(enc_out_raw, "module") and hasattr(enc_out_raw.module, "encoders"))
                )
                if _is_vqvae:
                    enc_out = encoders[m_midx](enc_input, pool_only=True, n_views=len(samples))
                else:
                    enc_out = encoders[m_midx](enc_input)
                # VQ-VAE / MoCoEncoder returns a tuple; extract pooled features
                if isinstance(enc_out, tuple):
                    # forward returns (recon, diffs, encoder_features, content_idx, dec_out, id_out, ...)
                    encoder_features = enc_out[2]
                    if len(encoder_features) > 0 and encoder_features[0].dim() == 5:
                        hz_m = encoder_features[0].mean(dim=[2, 3, 4]).detach().cpu().numpy()
                    elif len(encoder_features) > 0:
                        hz_m = encoder_features[0].detach().cpu().numpy()
                    else:
                        hz_m = enc_input.view(enc_input.size(0), -1).detach().cpu().numpy()
                else:
                    hz_m = enc_out.detach().cpu().numpy()
                del enc_input, enc_out
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
