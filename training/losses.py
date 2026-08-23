"""Definition of loss functions."""

import contextlib
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from lpips import LPIPS
from torch import cat, reshape, tensor
from torch.fft import rfftn
from torch.nn import PairwiseDistance

from utils.utils import TBSummaryTypes


def _merge_diags(diags: list[dict]) -> dict:
    """Average contrastive diagnostics across subsets."""
    merged = {}
    for key in diags[0]:
        vals = [d[key] for d in diags if key in d]
        merged[key] = sum(vals) / len(vals) if vals else 0.0
    return merged


# for numerical experiment
class CLLoss(ABC):
    """Abstract class to define losses in the CL framework that use one
    positive pair and one negative pair"""

    @abstractmethod
    def loss(self, z_rec, z3_rec, l):
        """
        z1_t = h(z1)
        z2_t = h(z2)
        z3_t = h(z3)
        and z1 ~ p(z1), z3 ~ p(z3)
        and z2 ~ p(z2 | z1)

        returns the total loss and componentwise contributions
        """

    def __call__(self, z_rec, z3_rec, l):
        return self.loss(z_rec, z3_rec, l)


class LpSimCLRLoss(CLLoss):
    """Extended InfoNCE objective for non-normalized representations based on an Lp norm.

    Args:
        p: Exponent of the norm to use.
        tau: Rescaling parameter of exponent.
        alpha: Weighting factor between the two summands.
        simclr_compatibility_mode: Use logsumexp (as used in SimCLR loss) instead of logmeanexp
        pow: Use p-th power of Lp norm instead of Lp norm.
    """

    def __init__(
        self,
        p: int = 2,
        tau: float = 1.0,
        alpha: float = 0.5,
        simclr_compatibility_mode: bool = False,
        simclr_denominator: bool = True,
        pow: bool = False,
    ):
        self.p = p
        self.tau = tau
        self.alpha = alpha
        self.simclr_compatibility_mode = simclr_compatibility_mode
        self.simclr_denominator = simclr_denominator
        self.pow = pow

    def loss(self, z_rec, z3_rec, l):
        """
        Calculates the loss function for the given inputs.

        Args:
            z_rec (list): List of reconstructed z values.
            z3_rec (list): List of reconstructed z3 values.
            l (int): Length of the input lists.

        Returns:
            tuple: A tuple containing the mean loss, the loss array, and a list of mean positive and negative losses.
        """
        # del z1, z2_con_z1, z3
        neg = 0
        pos = 0
        if self.p < 1.0:
            # add small epsilon to make calculation of norm numerically more stable
            for i in range(l):
                neg = neg + torch.norm(
                    torch.abs(z_rec[i].unsqueeze(0) - z3_rec[i].unsqueeze(1) + 1e-12),
                    p=self.p,
                    dim=-1,
                )
            for i in range(l - 1):
                pos = torch.norm(torch.abs(z_rec[i] - z_rec[i + 1]) + 1e-12, p=self.p, dim=-1)
        else:
            for i in range(l):
                neg = neg + torch.pow(z_rec[i].unsqueeze(1) - z3_rec[i].unsqueeze(0), float(self.p)).sum(dim=-1)
            for i in range(l - 1):
                pos = pos + torch.pow(z_rec[i] - z_rec[i + 1], float(self.p)).sum(dim=-1)

        if not self.pow:
            neg = neg.pow(1.0 / self.p)
            pos = pos.pow(1.0 / self.p)

        if self.simclr_compatibility_mode:
            neg_and_pos = torch.cat((neg, pos.unsqueeze(1)), dim=1)

            loss_pos = pos / self.tau
            loss_neg = torch.logsumexp(-neg_and_pos / self.tau, dim=1)
        else:
            if self.simclr_denominator:
                neg_and_pos = torch.cat((neg, pos.unsqueeze(1)), dim=1)
            else:
                neg_and_pos = neg

            loss_pos = pos / self.tau
            loss_neg = _logmeanexp(-neg_and_pos / self.tau, dim=1)

        loss = 2 * (self.alpha * loss_pos + (1.0 - self.alpha) * loss_neg)

        loss_mean = torch.mean(loss)
        # loss_std = torch.std(loss)

        loss_pos_mean = torch.mean(loss_pos)
        loss_neg_mean = torch.mean(loss_neg)

        return loss_mean, loss, [loss_pos_mean, loss_neg_mean]


class CosineInfoNCELoss(CLLoss):
    """Standard InfoNCE: cosine similarity matrix + cross-entropy over all view pairs.

    Expects L2-normalized representations (UnifiedCLLoss normalizes before calling).
    """

    def __init__(self, tau: float = 1.0):
        self.tau = tau

    def loss(self, z_rec, z3_rec, l):
        # z_rec: [n_views, B, C] — already L2-normalized
        B = z_rec.shape[1]
        targets = torch.arange(B, device=z_rec.device)
        total = z_rec.new_zeros(())
        n_terms = 0
        for i in range(l):
            for j in range(i + 1, l):
                sim = (z_rec[i] @ z_rec[j].T) / self.tau  # (B, B)
                total = total + F.cross_entropy(sim, targets) + F.cross_entropy(sim.T, targets)
                n_terms += 2
        loss_mean = total / max(n_terms, 1)
        with torch.no_grad():
            pos_sim = (z_rec[0] * z_rec[1 % l]).sum(-1).mean() if l > 1 else z_rec.new_zeros(())
            neg_sim = z_rec.new_zeros(())
        return loss_mean, total, [pos_sim, neg_sim]


def _logmeanexp(x, dim):
    """
    Compute the log-mean-exponential of a tensor along a specified dimension.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int): The dimension along which to compute the log-mean-exponential.

    Returns:
        torch.Tensor: The log-mean-exponential of the input tensor along the specified dimension.
    """
    N = torch.tensor(x.shape[dim], dtype=x.dtype, device=x.device)
    return torch.logsumexp(x, dim=dim) - torch.log(N)


class UnifiedCLLoss(CLLoss):
    """Loss for view-specific encoders"""

    def __init__(
        self,
        base_loss: CLLoss,
    ):
        """
        Initializes the UnifiedCLLoss class.

        Args:
            base_loss (CLLoss): The base loss function.

        """
        self.base_loss = base_loss

    def loss(self, est_content_dict: dict, z_rec, z3_rec):
        """
        Computes the loss for all subsets of views.

        Args:
            est_content_dict (dict): A dictionary containing the estimated content indices for each subset.
            z_rec: The reconstructed z values.
            z3_rec: The reconstructed z3 values.

        Returns:
            tuple: A tuple containing the total loss mean, total loss,
            and a list of total loss means for positive and negative samples.

        """
        z_rec = torch.stack(z_rec, dim=0)  # [n_views, batch-size, nSk]
        z3_rec = torch.stack(z3_rec, dim=0)  # [n_views, batch-size, nSk]
        z_rec = F.normalize(z_rec, dim=-1, eps=1e-6)
        z3_rec = F.normalize(z3_rec, dim=-1, eps=1e-6)

        total_loss_mean, total_loss, total_loss_pos_mean, total_loss_neg_mean = (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        for subset, subset_c_ind in est_content_dict.items():
            l = len(subset)
            c_ind = (
                torch.from_numpy(np.stack(list(subset_c_ind.values()))).type_as(z_rec).long()
            )  # n_views_in_this_subset, batch_size, n_Sk
            c_ind = c_ind[:, None, :].repeat(1, z_rec.shape[1], 1)  # (n_views_in_this_subset, batch_size, content_size)

            hz = torch.gather(z_rec[list(subset), :], -1, c_ind)
            hz3 = torch.gather(z3_rec[list(subset), :], -1, c_ind)
            loss_mean, loss, loss_mean_list = self.base_loss.loss(z_rec=hz, z3_rec=hz3, l=l)
            total_loss_mean += loss_mean
            total_loss += loss
            total_loss_pos_mean += loss_mean_list[0]
            total_loss_neg_mean += loss_mean_list[1]
        return total_loss_mean, total_loss, [total_loss_pos_mean, total_loss_neg_mean]


# for multimodal experiment
def infonce_loss(
    hz,
    sim_metric,
    criterion,
    projector=None,
    tau=1.0,
    estimated_content_indices=None,
    subsets=None,
    soft_content_mask=None,
    cross_view_negs_only=False,
):
    """
    Calculates the sum of InfoNCE loss for a given input tensor `hz`, over all subsets.

    Args:
        hz (torch.Tensor): The input tensor of shape (n_views, batch_size, num_features).
        sim_metric: The similarity metric used for calculating the loss.
        criterion: The loss criterion used for calculating the loss.
        projector: The projector used for projecting the input tensor (optional).
        tau (float): The temperature parameter for the loss calculation (default: 1.0).
        estimated_content_indices: The estimated content indices (optional).
        subsets: The subsets of indices used for calculating the loss (optional).
        soft_content_mask: Differentiable (0/1) mask, shape (1, C), from Gumbel
            straight-through.  When provided, features are masked via
            ``hz * mask`` so gradients flow back to channel_logits.
        cross_view_negs_only (bool): When True, negatives come only from the
            other view (no same-view negatives).

    Returns:
        torch.Tensor: The calculated InfoNCE loss.

    """
    if estimated_content_indices is None:
        # Use all feature dimensions as content when no content indices are provided
        content_indices = list(range(hz.shape[-1]))
        return infonce_base_loss(
            hz, content_indices, sim_metric, criterion, projector, tau, cross_view_negs_only=cross_view_negs_only
        )
    else:
        total_loss = torch.zeros(1).type_as(hz)
        sub_diags = []
        for est_content_indices, subset in zip(estimated_content_indices, subsets):
            sub_loss = infonce_base_loss(
                hz[list(subset), ...],
                est_content_indices,
                sim_metric,
                criterion,
                projector,
                tau,
                soft_content_mask=soft_content_mask,
                cross_view_negs_only=cross_view_negs_only,
            )
            total_loss = total_loss + sub_loss
            if hasattr(sub_loss, "_contrastive_diag"):
                sub_diags.append(sub_loss._contrastive_diag)
        if sub_diags:
            total_loss._contrastive_diag = _merge_diags(sub_diags)
        return total_loss


def infonce_base_loss(
    hz_subset,
    content_indices,
    sim_metric,
    criterion,
    projector=None,
    tau=1.0,
    soft_content_mask=None,
    cross_view_negs_only=False,
):
    """
    Computes the InfoNCE (Normalized Cross Entropy) loss for multi-view data.

    When ``soft_content_mask`` is provided (a differentiable 0/1 tensor from
    Gumbel straight-through), the content selection is done via element-wise
    multiplication so that gradients flow back to ``channel_logits``.
    Otherwise falls back to integer-index selection (non-differentiable w.r.t. the mask).

    Args:
        hz_subset: Latent features, shape (n_views, batch_size, C).
        content_indices (list): Integer indices of content dimensions (used when
            soft_content_mask is None).
        sim_metric: Pairwise similarity function.
        criterion: Loss criterion (CrossEntropyLoss).
        projector: Optional projection function.
        tau (float): Temperature.
        soft_content_mask: Optional differentiable mask, shape (1, C).
        cross_view_negs_only (bool): When True, negatives come only from the
            other view (no same-view negatives).  This prevents the loss from
            decreasing via within-view instance discrimination alone, which is
            important when separate encoders produce features in different
            subspaces.

    Returns:
        torch.Tensor: Total loss value.
    """

    n_view = len(hz_subset)
    d = hz_subset.shape[1]  # batch size

    projector = projector or (lambda x: x)

    if cross_view_negs_only:
        # Only cross-view similarities — forces the model to align views.
        SIM_cross = {}
        for i in range(n_view):
            for j in range(n_view):
                if i == j:
                    continue
                if (j, i) in SIM_cross:
                    SIM_cross[(i, j)] = SIM_cross[(j, i)].transpose(-1, -2)
                    continue
                if soft_content_mask is not None:
                    hz_i = hz_subset[i] * soft_content_mask
                    hz_j = hz_subset[j] * soft_content_mask
                else:
                    hz_i = hz_subset[i][..., content_indices]
                    hz_j = hz_subset[j][..., content_indices]
                SIM_cross[(i, j)] = (sim_metric(hz_i.unsqueeze(-2), hz_j.unsqueeze(-3)) / tau).type_as(hz_subset)

        total_loss_value = torch.zeros(1, device=hz_subset.device, dtype=hz_subset.dtype)
        n_correct = 0
        n_total = 0
        pos_sims = []
        neg_sims = []
        for i in range(n_view):
            for j in range(n_view):
                if i >= j:
                    continue
                scores_ij = SIM_cross[(i, j)]  # (d, d)
                scores_ji = SIM_cross[(j, i)]  # (d, d)
                targets = torch.arange(d, dtype=torch.long, device=hz_subset.device)
                total_loss_value += criterion(scores_ij, targets)
                total_loss_value += criterion(scores_ji, targets)
                with torch.no_grad():
                    n_correct += (scores_ij.argmax(dim=1) == targets).sum().item()
                    n_correct += (scores_ji.argmax(dim=1) == targets).sum().item()
                    n_total += 2 * d
                    # Positive sims are the diagonals (in tau-scaled space, undo /tau)
                    pos_sims.append(scores_ij.diag() * tau)
                    pos_sims.append(scores_ji.diag() * tau)
                    # Negative sims: off-diagonal entries
                    mask_offdiag = ~torch.eye(d, dtype=torch.bool, device=hz_subset.device)
                    neg_sims.append(scores_ij[mask_offdiag] * tau)
                    neg_sims.append(scores_ji[mask_offdiag] * tau)
        with torch.no_grad():
            total_loss_value._contrastive_diag = {
                "top1_acc": n_correct / max(n_total, 1),
                "pos_sim_mean": torch.cat(pos_sims).mean().item(),
                "pos_sim_std": torch.cat(pos_sims).std().item(),
                "neg_sim_mean": torch.cat(neg_sims).mean().item(),
                "neg_sim_std": torch.cat(neg_sims).std().item(),
            }
        return total_loss_value

    # Default: include both cross-view and same-view negatives.
    SIM = [[None] * n_view for _ in range(n_view)]

    for i in range(n_view):
        for j in range(n_view):
            if j >= i:
                if soft_content_mask is not None:
                    hz_i = hz_subset[i] * soft_content_mask
                    hz_j = hz_subset[j] * soft_content_mask
                else:
                    hz_i = hz_subset[i][..., content_indices]
                    hz_j = hz_subset[j][..., content_indices]
                sim_ij = (sim_metric(hz_i.unsqueeze(-2), hz_j.unsqueeze(-3)) / tau).type_as(hz_subset)
                if i == j:
                    mask = torch.zeros_like(sim_ij, dtype=torch.bool)
                    mask[..., range(d), range(d)] = True
                    sim_ij = sim_ij.masked_fill(mask, float("-inf"))
                SIM[i][j] = sim_ij
            else:
                SIM[i][j] = SIM[j][i].transpose(-1, -2)

    total_loss_value = torch.zeros(1, device=hz_subset.device, dtype=hz_subset.dtype)
    n_correct = 0
    n_total = 0
    pos_sims = []
    neg_sims = []
    for i in range(n_view):
        for j in range(n_view):
            if i < j:
                raw_scores1 = torch.cat([SIM[i][j], SIM[i][i]], dim=-1)
                raw_scores2 = torch.cat([SIM[j][j], SIM[j][i]], dim=-1)
                raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)  # (2d, 2d)
                targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)
                total_loss_value += criterion(raw_scores, targets)
                with torch.no_grad():
                    n_correct += (raw_scores.argmax(dim=1) == targets).sum().item()
                    n_total += 2 * d
                    # Positive sims: diagonal of cross-view blocks SIM[i][j] and SIM[j][i]
                    pos_sims.append(SIM[i][j].diag() * tau)
                    pos_sims.append(SIM[j][i].diag() * tau)
                    # Negative sims: off-diagonal of cross-view block
                    mask_offdiag = ~torch.eye(d, dtype=torch.bool, device=hz_subset.device)
                    neg_sims.append(SIM[i][j][mask_offdiag] * tau)
    with torch.no_grad():
        _diag = {
            "top1_acc": n_correct / max(n_total, 1),
            "pos_sim_mean": torch.cat(pos_sims).mean().item(),
            "pos_sim_std": torch.cat(pos_sims).std().item(),
        }
        if neg_sims:
            _neg = torch.cat(neg_sims)
            _diag["neg_sim_mean"] = _neg.mean().item()
            _diag["neg_sim_std"] = _neg.std().item()
        total_loss_value._contrastive_diag = _diag
    return total_loss_value


def _center_patch_features(hz_c, mode):
    """Two-way centering of patch features, shape ``(n_views, B, C, P)``.

    ``"position"`` removes the across-batch mean at each patch position — on
    registered volumes that mean is the anatomy every subject shares there,
    which is what makes same-position negatives largely false. ``"double"``
    additionally removes each sample's mean over positions, i.e. any spatially
    constant per-sample code (the cheapest solution to same-position
    discrimination, and the one that flattens the spatial maps). What survives
    is the sample x position interaction.
    """
    if mode == "none":
        return hz_c
    if mode not in ("position", "double"):
        raise ValueError(f"Unknown patch centering mode: {mode!r}")

    B, P = hz_c.shape[1], hz_c.shape[3]
    if B < 2:
        # Subtracting the mean over a single sample would zero the tensor outright.
        return hz_c
    hz_c = hz_c - hz_c.mean(dim=1, keepdim=True)
    if mode == "double" and P > 1:
        hz_c = hz_c - hz_c.mean(dim=3, keepdim=True)
    return hz_c


def patch_infonce_loss(
    hz,
    sim_metric,
    criterion,
    projector=None,
    tau=1.0,
    estimated_content_indices=None,
    subsets=None,
    soft_content_mask=None,
    cross_view_negs_only=False,
    center_mode="none",
    center_weight=False,
):
    """
    Patch-level InfoNCE: aligns corresponding spatial patches across views.

    Same interface as ``infonce_loss`` but expects ``hz`` with shape
    ``(n_views, B, C, P)`` where P is the number of spatial patches.
    Computes InfoNCE independently per patch position and averages.

    This preserves spatial correspondence between views — patches at the
    same anatomical location should align, providing a much richer training
    signal than global average pooling.

    ``center_mode``/``center_weight`` are forwarded to ``_patch_infonce_base``.
    """
    if estimated_content_indices is None:
        content_indices = list(range(hz.shape[2]))
        return _patch_infonce_base(
            hz,
            content_indices,
            criterion,
            tau,
            soft_content_mask=soft_content_mask,
            cross_view_negs_only=cross_view_negs_only,
            center_mode=center_mode,
            center_weight=center_weight,
        )
    else:
        total_loss = torch.zeros(1).type_as(hz)
        sub_diags = []
        for est_content_indices, subset in zip(estimated_content_indices, subsets):
            sub_loss = _patch_infonce_base(
                hz[list(subset), ...],
                est_content_indices,
                criterion,
                tau,
                soft_content_mask=soft_content_mask,
                cross_view_negs_only=cross_view_negs_only,
                center_mode=center_mode,
                center_weight=center_weight,
            )
            total_loss = total_loss + sub_loss
            if hasattr(sub_loss, "_contrastive_diag"):
                sub_diags.append(sub_loss._contrastive_diag)
        if sub_diags:
            total_loss._contrastive_diag = _merge_diags(sub_diags)
        return total_loss


def _patch_infonce_base(
    hz_subset,
    content_indices,
    criterion,
    tau=1.0,
    soft_content_mask=None,
    cross_view_negs_only=False,
    center_mode="none",
    center_weight=False,
):
    """
    Core patch-level InfoNCE computation.

    Args:
        hz_subset: (n_views, B, C, P) patch-pooled features.
        content_indices: Channel indices for content (fallback when mask is None).
        criterion: CrossEntropyLoss instance.
        tau: Temperature.
        soft_content_mask: Optional (1, C) differentiable mask.
        cross_view_negs_only: Only use cross-view negatives.
        center_mode: "none", "position" or "double" — see _center_patch_features.
        center_weight: Weight each patch position's loss by its mean residual
            magnitude, measured before L2 normalisation. Without it, positions
            where every sample looks alike are renormalised back to unit vectors
            and contribute full-magnitude random directions.
    """
    n_view, B, C, P = hz_subset.shape

    # Apply content selection
    if soft_content_mask is not None:
        # (1, C) → (1, 1, C, 1) for broadcasting with (n_views, B, C, P)
        hz_c = hz_subset * soft_content_mask.view(1, 1, -1, 1)
    else:
        hz_c = hz_subset[:, :, content_indices, :]

    hz_c = _center_patch_features(hz_c, center_mode)

    if center_weight and center_mode != "none":
        with torch.no_grad():
            pos_w = hz_c.norm(dim=2).mean(dim=(0, 1))  # (P,)
            pos_w = pos_w / pos_w.mean().clamp_min(1e-12)
    else:
        pos_w = None

    def _ce(flat_scores, flat_targets, n_rows):
        """Cross-entropy over (P * n_rows, ...) logits, optionally weighted per position."""
        if pos_w is None:
            return criterion(flat_scores, flat_targets)
        per_row = F.cross_entropy(flat_scores, flat_targets, reduction="none")
        return (per_row.view(P, n_rows).mean(dim=1) * pos_w).mean()

    # L2-normalize along channel dimension
    hz_n = F.normalize(hz_c, dim=2, eps=1e-6)  # (n_views, B, C', P)

    if cross_view_negs_only:
        total_loss = torch.zeros(1, device=hz_subset.device, dtype=hz_subset.dtype)
        all_pos_sims = []
        all_neg_sims = []
        n_correct = 0
        n_total = 0

        for i in range(n_view):
            for j in range(n_view):
                if i >= j:
                    continue
                # Per-patch cosine similarity: (P, B, B)
                scores = torch.einsum("bcp,dcp->pbd", hz_n[i], hz_n[j]) / tau

                targets = torch.arange(B, device=hz_subset.device)
                # Reshape to (P*B, B) for batched cross-entropy
                loss_ij = _ce(scores.reshape(P * B, B), targets.repeat(P), B)

                scores_ji = scores.transpose(-1, -2)
                loss_ji = _ce(scores_ji.reshape(P * B, B), targets.repeat(P), B)

                total_loss = total_loss + loss_ij + loss_ji

                with torch.no_grad():
                    # Diagnostics (aggregated across patches)
                    preds_ij = scores.reshape(P * B, B).argmax(dim=1)
                    preds_ji = scores_ji.reshape(P * B, B).argmax(dim=1)
                    tgt_rep = targets.repeat(P)
                    n_correct += (preds_ij == tgt_rep).sum().item()
                    n_correct += (preds_ji == tgt_rep).sum().item()
                    n_total += 2 * P * B
                    # Positive sims: diagonals across all patches
                    diag_sims = torch.diagonal(scores, dim1=-2, dim2=-1)  # (P, B)
                    all_pos_sims.append(diag_sims.reshape(-1) * tau)
                    # Negative sims: off-diagonals
                    mask_offdiag = ~torch.eye(B, dtype=torch.bool, device=hz_subset.device)
                    all_neg_sims.append(scores[:, mask_offdiag].reshape(-1) * tau)

        with torch.no_grad():
            total_loss._contrastive_diag = {
                "top1_acc": n_correct / max(n_total, 1),
                "pos_sim_mean": torch.cat(all_pos_sims).mean().item(),
                "pos_sim_std": torch.cat(all_pos_sims).std().item(),
                "neg_sim_mean": torch.cat(all_neg_sims).mean().item(),
                "neg_sim_std": torch.cat(all_neg_sims).std().item(),
            }
        return total_loss

    # Default: cross-view + same-view negatives
    # Build per-patch similarity matrices across all view pairs
    SIM = [[None] * n_view for _ in range(n_view)]
    for i in range(n_view):
        for j in range(n_view):
            if j >= i:
                # (P, B, B) cosine similarity per patch
                sim_ij = torch.einsum("bcp,dcp->pbd", hz_n[i], hz_n[j]) / tau
                if i == j:
                    # Mask self-similarity diagonal
                    eye_mask = torch.eye(B, dtype=torch.bool, device=hz_subset.device)
                    sim_ij[:, eye_mask] = float("-inf")
                SIM[i][j] = sim_ij
            else:
                SIM[i][j] = SIM[j][i].transpose(-1, -2)

    total_loss = torch.zeros(1, device=hz_subset.device, dtype=hz_subset.dtype)
    n_correct = 0
    n_total = 0
    pos_sims = []
    neg_sims = []
    targets = torch.arange(2 * B, dtype=torch.long, device=hz_subset.device)
    for i in range(n_view):
        for j in range(n_view):
            if i < j:
                # (P, B, B) blocks → (P, 2B, 2B) combined matrix
                raw1 = torch.cat([SIM[i][j], SIM[i][i]], dim=-1)  # (P, B, 2B)
                raw2 = torch.cat([SIM[j][j], SIM[j][i]], dim=-1)  # (P, B, 2B)
                raw_scores = torch.cat([raw1, raw2], dim=-2)  # (P, 2B, 2B)

                loss = _ce(raw_scores.reshape(P * 2 * B, 2 * B), targets.repeat(P), 2 * B)
                total_loss = total_loss + loss

                with torch.no_grad():
                    preds = raw_scores.reshape(P * 2 * B, 2 * B).argmax(dim=1)
                    n_correct += (preds == targets.repeat(P)).sum().item()
                    n_total += P * 2 * B
                    diag_ij = torch.diagonal(SIM[i][j], dim1=-2, dim2=-1)  # (P, B)
                    pos_sims.append(diag_ij.reshape(-1) * tau)
                    mask_offdiag = ~torch.eye(B, dtype=torch.bool, device=hz_subset.device)
                    neg_sims.append(SIM[i][j][:, mask_offdiag].reshape(-1) * tau)

    with torch.no_grad():
        _diag = {
            "top1_acc": n_correct / max(n_total, 1),
            "pos_sim_mean": torch.cat(pos_sims).mean().item(),
            "pos_sim_std": torch.cat(pos_sims).std().item(),
        }
        if neg_sims:
            _neg = torch.cat(neg_sims)
            _diag["neg_sim_mean"] = _neg.mean().item()
            _diag["neg_sim_std"] = _neg.std().item()
        total_loss._contrastive_diag = _diag
    return total_loss


def split_infonce_loss(
    hz_align,
    hz_entropy,
    tau=1.0,
    tau_entropy=None,
    cross_view_negs_only=False,
):
    """Yao et al. (ICLR 2024) eq. (3.3): alignment on the raw block, entropy on the projection.

    Standard InfoNCE fuses two terms inside one cross-entropy over a single
    similarity matrix::

        L_i = -s_ii + log Sum_j exp(s_ij)
              ^^^^^   ^^^^^^^^^^^^^^^^^^^
              align    entropy/uniformity

    Their §5 maps the numerator to *content alignment* and the denominator to
    *entropy regularization*, and Defn 3.6 applies the projection ``t_k`` to the
    entropy term **only** — the alignment term keeps acting on the raw
    representation. That placement is the point: uniformity implies independent
    components, which fights disentanglement when the latents are causally
    dependent, so ``t_k`` shields the representation from the uniformity pressure
    while leaving the useful view-invariance signal direct. A SimCLR-style head
    (``_project_contrastive_content``) instead sits in front of *both* terms.

    Because the two terms are computed in different spaces this is no longer a
    normalized log-likelihood — it is a sum of two separately-defined terms, exactly
    as eq. (3.3) is written. Consequences: ``top1_acc`` is a projected-space proxy
    (the diagonal no longer comes from the matrix the loss normalizes over), and the
    two terms are reported separately since their balance is no longer automatic.

    The cross-view POSITIVE is excluded from the entropy sum. In standard InfoNCE the
    positive sits in the denominator but is balanced by a same-space numerator (net
    attractive); once the numerator moves to the raw space, a positive left in the
    denominator is purely repulsive and fights alignment outright.

    **Spatial alignment, global entropy.** ``hz_align`` may be patch-shaped
    ``(n_views, B, k, P)``: alignment is a per-sample *distance*, so it is computed per
    position and averaged, giving view-invariance at spatial resolution. ``hz_entropy``
    is always reduced to a single global vector per sample, because entropy is a
    property of the *distribution over samples* — maximizing it per position
    independently does NOT imply the joint map is invertible (and neighbouring
    positions are strongly correlated, so it would only add many weak, non-independent
    constraints). This asymmetry is why the two terms are pooled differently.

    Args:
        hz_align: Content block driving alignment — ``(n_views, B, k)`` pooled, or
            ``(n_views, B, k, P)`` patch-shaped (cosine per position, then averaged).
        hz_entropy: Projected block ``t(rep)``, ``(n_views, B, d)``. Drives entropy.
            A patch-shaped input is mean-pooled over ``P`` first — entropy stays global.
        tau: Temperature for the alignment term.
        tau_entropy: Temperature for the entropy term (defaults to ``tau``). Separate
            because the two terms now live in different geometries.
        cross_view_negs_only: When True the entropy sum runs over cross-view pairs
            only; otherwise same-view pairs join it with the self-similarity masked.

    Returns:
        Scalar loss with ``._contrastive_diag`` attached.
    """
    n_view, B = hz_align.shape[0], hz_align.shape[1]
    tau_u = tau if tau_entropy is None else tau_entropy
    eps = 1e-6

    # Alignment may arrive patch-shaped (V, B, k, P); the channel axis is then 2.
    # Entropy must be GLOBAL, so a patch-shaped entropy input is pooled over P first
    # (see the docstring: per-position entropy does not imply invertibility).
    _align_is_patch = hz_align.dim() == 4
    if hz_entropy.dim() == 4:
        hz_entropy = hz_entropy.mean(-1)
    a = F.normalize(hz_align, dim=2 if _align_is_patch else -1, eps=eps)  # raw
    p = F.normalize(hz_entropy, dim=-1, eps=eps)  # (V, B, d) — projected, global

    total_loss = torch.zeros(1, device=hz_align.device, dtype=hz_align.dtype)
    align_terms, entropy_terms, pos_sims, neg_sims = [], [], [], []
    n_correct = 0
    n_total = 0

    for i in range(n_view):
        for j in range(n_view):
            if i >= j:
                continue
            # --- Alignment: positives only, in the RAW space ---
            # Symmetric in (i, j), so the same vector serves both directions.
            # Patch input: cosine per spatial position, averaged over positions —
            # alignment is a per-sample DISTANCE, so it extends over space cleanly.
            if _align_is_patch:
                s_pos = (a[i] * a[j]).sum(dim=1).mean(dim=-1) / tau  # (B,)
            else:
                s_pos = (a[i] * a[j]).sum(-1) / tau  # (B,)

            # --- Entropy: negatives only, in the PROJECTED space ---
            s_cross = (p[i] @ p[j].transpose(-1, -2)) / tau_u  # (B, B)
            eye = torch.eye(B, dtype=torch.bool, device=hz_align.device)
            # Mask the cross-view DIAGONAL — the positive pair. In standard InfoNCE the
            # positive sits in the denominator but is balanced by the same-space
            # numerator (net attractive); here the numerator (alignment) is on the RAW
            # space, so a projected positive left in the denominator is PURELY
            # repulsive. That makes t reward the two views' raw content being
            # DIFFERENT, directly opposing alignment — the views never align and
            # pos_sim pins at 0. Excluding it (a same-sample pair is not a negative
            # anyway) lets entropy repel only genuine negatives.
            s_cross_neg = s_cross.masked_fill(eye, float("-inf"))
            if cross_view_negs_only:
                den_i, den_j = s_cross_neg, s_cross_neg.transpose(-1, -2)
            else:
                # Same-view pairs join the sum; mask each sample against itself so a
                # sample is never its own negative.
                s_ii = (p[i] @ p[i].transpose(-1, -2)) / tau_u
                s_jj = (p[j] @ p[j].transpose(-1, -2)) / tau_u
                s_ii = s_ii.masked_fill(eye, float("-inf"))
                s_jj = s_jj.masked_fill(eye, float("-inf"))
                den_i = torch.cat([s_cross_neg, s_ii], dim=1)  # (B, 2B)
                den_j = torch.cat([s_cross_neg.transpose(-1, -2), s_jj], dim=1)

            lse_i = torch.logsumexp(den_i, dim=1)  # (B,)
            lse_j = torch.logsumexp(den_j, dim=1)  # (B,)

            # Scale matches whichever branch of infonce_base_loss this replaces, so
            # switching modes never silently rescales contrastive against recon: the
            # cross-view branch there sums two per-view means (CE(s) + CE(s.T)), while
            # the default branch takes ONE mean over the concatenated 2B rows.
            term_i, term_j = (-s_pos + lse_i).mean(), (-s_pos + lse_j).mean()
            total_loss = total_loss + (term_i + term_j if cross_view_negs_only else 0.5 * (term_i + term_j))

            with torch.no_grad():
                # Both terms in LOSS units so they are directly comparable and
                # 2*(align_term + entropy_term) reconstructs this pair's contribution.
                # (Raw cosine is reported separately as pos_sim_mean.)
                align_terms.append(-s_pos)
                entropy_terms.append((lse_i + lse_j) / 2)
                pos_sims.append(s_pos * tau)
                offdiag = ~torch.eye(B, dtype=torch.bool, device=hz_align.device)
                neg_sims.append(s_cross[offdiag] * tau_u)
                # Projected-space proxy: would the projection alone pair the views?
                tgt = torch.arange(B, device=hz_align.device)
                n_correct += (s_cross.argmax(dim=1) == tgt).sum().item()
                n_correct += (s_cross.argmax(dim=0) == tgt).sum().item()
                n_total += 2 * B

    with torch.no_grad():
        _pos = torch.cat(pos_sims)
        _neg = torch.cat(neg_sims)
        total_loss._contrastive_diag = {
            "top1_acc": n_correct / max(n_total, 1),  # projected-space proxy
            "pos_sim_mean": _pos.mean().item(),
            "pos_sim_std": _pos.std().item(),
            "neg_sim_mean": _neg.mean().item(),
            "neg_sim_std": _neg.std().item(),
            # The two eq. (3.3) terms, reported apart and on the same scale: their
            # balance is no longer enforced by a shared softmax, so drift in either is
            # worth watching.
            "align_term": torch.cat(align_terms).mean().item(),
            "entropy_term": torch.cat(entropy_terms).mean().item(),
            # Health of t itself. Its codomain is the centered unit cube (-1,1)^k, so a
            # collapse to a constant (std -> 0, mean drifting off 0) or to the corners
            # (saturated -> 1) makes the entropy estimate degenerate while the loss can
            # still look healthy — the failure mode with no counterpart in the
            # SimCLR-head arm.
            "t_out_mean": hz_entropy.mean().item(),
            "t_out_std": hz_entropy.std().item(),
            "t_out_saturated": (hz_entropy.abs() > 0.99).float().mean().item(),
        }
    return total_loss


def moco_infonce_loss(
    q, k, queue, content_indices, tau=1.0, soft_content_mask=None, queue_v1=None, cross_view_negs_only=False
):
    """
    MoCo-style InfoNCE loss for a single view pair using a queue of negatives.

    For each ordered pair (i, j) with i < j we treat q[i] as query matched to k[j]
    as the positive key, and all queue columns as negatives (and vice versa for j→i).

    Args:
        q  (torch.Tensor): Online (query) embeddings,   shape (n_views, B, C).
        k  (torch.Tensor): Momentum (key) embeddings,   shape (n_views, B, C).
                           Must already be detached from the computation graph.
        queue (torch.Tensor): Negative key queue for view 0, shape (C, queue_size).
        content_indices (list[int]): Channel indices to use (fallback when
            soft_content_mask is None).
        tau (float): Temperature scaling factor.
        soft_content_mask: Optional differentiable (0/1) mask, shape (1, C).
            When provided, content selection uses ``features * mask`` so
            gradients flow back to channel_logits via Gumbel straight-through.
        queue_v1 (torch.Tensor | None): Separate negative queue for view 1,
            shape (C, queue_size).  When provided (typically with separate
            encoders), view-1 queries use this queue for negatives instead of
            ``queue``.  This prevents trivially easy negatives when the two
            encoders produce features in different subspaces.
        cross_view_negs_only (bool): When True, each view's queries use the
            OTHER view's queue for negatives instead of their own.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    n_view = q.shape[0]
    total_loss = torch.zeros(1, device=q.device, dtype=q.dtype)

    # Project to content dimensions and L2-normalise.
    # Use eps=1e-6 to avoid NaN when the Gumbel mask zeros out all channels
    # for a sample (norm → 0 → division by zero in F.normalize).
    _norm_eps = 1e-6
    if soft_content_mask is not None:
        # Differentiable masking — gradients flow to channel_logits
        q_c = F.normalize(q * soft_content_mask, dim=-1, eps=_norm_eps)  # (n_views, B, C)
        k_c = F.normalize(k * soft_content_mask, dim=-1, eps=_norm_eps)  # (n_views, B, C)
        mask_col = soft_content_mask.squeeze(0).unsqueeze(-1)  # (C, 1)
        queue_c = F.normalize(queue * mask_col, dim=0, eps=_norm_eps)
        queue_v1_c = F.normalize(queue_v1 * mask_col, dim=0, eps=_norm_eps) if queue_v1 is not None else queue_c
    else:
        q_c = F.normalize(q[..., content_indices], dim=-1, eps=_norm_eps)  # (n_views, B, d)
        k_c = F.normalize(k[..., content_indices], dim=-1, eps=_norm_eps)  # (n_views, B, d)
        queue_c = F.normalize(queue[content_indices, :], dim=0, eps=_norm_eps)  # (d, Q)
        queue_v1_c = (
            F.normalize(queue_v1[content_indices, :], dim=0, eps=_norm_eps) if queue_v1 is not None else queue_c
        )

    # Per-view queue lookup.
    # cross_view_negs_only: view-i queries use view-j's queue (other view).
    # Default: view-i queries use view-i's queue (same view).
    if cross_view_negs_only and n_view == 2:
        _queue_for_view = [queue_v1_c, queue_c]  # view-0 uses view-1's queue, vice versa
    else:
        _queue_for_view = [queue_c, queue_v1_c] if n_view == 2 else [queue_c] * n_view

    all_pos_sims = []
    all_neg_sims = []
    n_correct = 0
    n_total = 0

    for i in range(n_view):
        for j in range(n_view):
            if i >= j:
                continue
            # q[i] → positive k[j], negatives from queue
            pos_ij = (q_c[i] * k_c[j]).sum(dim=-1, keepdim=True)  # (B, 1)
            neg_ij = q_c[i] @ _queue_for_view[i]  # (B, Q)
            logits_ij = torch.cat([pos_ij, neg_ij], dim=1) / tau  # (B, Q+1)
            targets = torch.zeros(logits_ij.shape[0], dtype=torch.long, device=q.device)
            total_loss = total_loss + F.cross_entropy(logits_ij, targets)

            # Symmetric: q[j] → positive k[i], negatives from queue
            pos_ji = (q_c[j] * k_c[i]).sum(dim=-1, keepdim=True)  # (B, 1)
            neg_ji = q_c[j] @ _queue_for_view[j]  # (B, Q)
            logits_ji = torch.cat([pos_ji, neg_ji], dim=1) / tau  # (B, Q+1)
            total_loss = total_loss + F.cross_entropy(logits_ji, targets)

            # --- Diagnostics (detached) ---
            with torch.no_grad():
                all_pos_sims.append(pos_ij.squeeze(-1))  # raw cosine sim (before /tau)
                all_pos_sims.append(pos_ji.squeeze(-1))
                all_neg_sims.append(neg_ij)
                all_neg_sims.append(neg_ji)
                n_correct += (logits_ij.argmax(dim=1) == 0).sum().item()
                n_correct += (logits_ji.argmax(dim=1) == 0).sum().item()
                n_total += logits_ij.shape[0] + logits_ji.shape[0]

    # Attach diagnostics dict to the loss tensor (non-persistent, won't affect .backward())
    with torch.no_grad():
        total_loss._contrastive_diag = {
            "top1_acc": n_correct / max(n_total, 1),
            "pos_sim_mean": torch.cat(all_pos_sims).mean().item(),
            "pos_sim_std": torch.cat(all_pos_sims).std().item(),
            "neg_sim_mean": torch.cat(all_neg_sims).mean().item(),
            "neg_sim_std": torch.cat(all_neg_sims).std().item(),
        }

    return total_loss


def moco_loss(
    q,
    k,
    queue,
    sim_metric,
    tau=1.0,
    estimated_content_indices=None,
    subsets=None,
    soft_content_mask=None,
    queue_v1=None,
    cross_view_negs_only=False,
):
    """
    Top-level MoCo loss that mirrors the ``infonce_loss`` signature.

    Iterates over all (subset, content_indices) pairs and sums the per-subset
    ``moco_infonce_loss`` values.

    Args:
        q  (torch.Tensor): Online embeddings,    shape (n_views, B, C).
        k  (torch.Tensor): Momentum embeddings,  shape (n_views, B, C).
        queue (torch.Tensor): Negative queue for view 0, shape (C, queue_size).
        sim_metric: Unused — kept for API compatibility with ``infonce_loss``.
        tau (float): Temperature.
        estimated_content_indices (list[list[int]]): Content channel indices per subset.
        subsets (list[tuple]): View subsets (same length as estimated_content_indices).
        soft_content_mask: Optional differentiable mask from Gumbel straight-through.
        queue_v1 (torch.Tensor | None): Separate negative queue for view 1.
            When provided, view-1 queries use this queue for negatives.
        cross_view_negs_only (bool): When True, each view's queries use the
            OTHER view's queue for negatives instead of their own.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    if estimated_content_indices is None or subsets is None:
        # Fall back to using all channels
        return moco_infonce_loss(
            q, k, queue, list(range(q.shape[-1])), tau, queue_v1=queue_v1, cross_view_negs_only=cross_view_negs_only
        )

    total_loss = torch.zeros(1, device=q.device, dtype=q.dtype)
    sub_diags = []
    for content_indices, subset in zip(estimated_content_indices, subsets):
        q_sub = q[list(subset), ...]
        k_sub = k[list(subset), ...]
        sub_loss = moco_infonce_loss(
            q_sub,
            k_sub,
            queue,
            content_indices,
            tau,
            soft_content_mask=soft_content_mask,
            queue_v1=queue_v1,
            cross_view_negs_only=cross_view_negs_only,
        )
        total_loss = total_loss + sub_loss
        if hasattr(sub_loss, "_contrastive_diag"):
            sub_diags.append(sub_loss._contrastive_diag)
    if sub_diags:
        total_loss._contrastive_diag = _merge_diags(sub_diags)
    return total_loss


# ---------------------------------------------------------------------------
# Barlow Twins & VICReg — negative-free contrastive objectives
# ---------------------------------------------------------------------------


def barlow_twins_loss(
    hz,
    estimated_content_indices=None,
    subsets=None,
    soft_content_mask=None,
    lambd=0.005,
    center_mode="none",
    patch_stat="fold",
    sim_coeff=0.0,
    std_coeff=0.0,
    sim_normalize=False,
    corr_ema=None,
    corr_ema_decay=0.0,
    **_kwargs,
):
    """Barlow Twins loss over content channels of paired views.

    Computes the cross-correlation matrix between the two views' content
    embeddings (batch-normalised) and pushes it toward the identity matrix.

    Args:
        hz: Encoder features, shape ``(n_views, B, C)`` or
            ``(n_views, B, C, P)`` for patch mode (P patches are folded
            into the batch dimension).
        estimated_content_indices: Per-subset content channel indices.
        subsets: View subsets (same length as *estimated_content_indices*).
        soft_content_mask: Optional differentiable ``(1, C)`` mask.
        lambd: Weight on the off-diagonal (redundancy-reduction) term.
        center_mode: Patch centering, applied only in patch mode — see
            ``_center_patch_features``. Without it the folded batch mixes samples and
            positions, so the per-channel standardisation below removes only the grand
            mean and the *positional* pattern survives. On registered volumes both views
            agree on that pattern almost exactly, so the on-diagonal term reaches ~1 on
            shared anatomy alone, carrying no subject information. ``"position"`` removes
            it and makes the diagonal a statement about subjects.
        patch_stat: ``"fold"`` or ``"per_position"`` — see the note below. Patch mode only.
        sim_coeff: Weight on an MSE alignment term over the RAW (unstandardised) features.
            BT's ``on_diag`` is a correlation, which is invariant to a per-view constant
            offset and so cannot require the two views to coincide — only to co-vary. This
            is the term that can. 0 disables (default), leaving the loss bit-identical.
        corr_ema: Mutable dict owned by the CALLER, holding the running cross-correlation
            per (subset, view-i, view-j). Pass the same dict every step. Not checkpointed:
            bias correction makes a zero init unbiased, so a resume costs a noisier estimate
            for ~1/(1-decay) steps, not a transient.
        corr_ema_decay: EMA momentum for that matrix. 0 disables (default) and leaves the
            loss bit-identical. Cuts off_diag's d(d-1)/B sampling floor by (1-m)/(1+m):
            at d=44, B=128 the floor is 14.78, and m=0.99 takes it to 0.075. Raise ``lambd``
            when enabling — the gradient on this term scales with (1-m).
        sim_normalize: Divide the MSE by a detached per-channel ``2*sigma^2`` so it reads
            ``1 - rho`` rather than an absolute distance. Makes the term scale-free (it
            stops fighting the variance hinge, which is simultaneously rescaling sigma) and
            direction-equalised (it stops being magnitude-weighted toward the top PCs).
        std_coeff: Weight on VICReg's variance hinge ``relu(1 - std)`` over the same raw
            features. Mandatory whenever ``sim_coeff > 0``: MSE alone is minimised by
            collapsing both views to zero, and every other term here is scale-invariant.

    Returns:
        Scalar loss with ``._contrastive_diag`` attached.
    """
    # Patch mode. Two ways to turn (n_views, B, C, P) into a cross-correlation:
    #
    #   "fold"         — flatten to (n_views, B*P, C) and standardise each channel over all
    #                    B*P rows (the original behaviour).
    #   "per_position" — standardise each (channel, position) across the B samples and
    #                    average the per-position cross-correlations.
    #
    # MEASURED, so nobody re-derives it: with centering already applied these are very nearly
    # the SAME objective. Once each (c, p) has zero mean across samples, the folded
    # cross-correlation is *already* the average of the per-position cross-covariances; the
    # only thing "per_position" adds is per-position variance normalisation (a
    # heteroscedasticity correction). On matched synthetic data the two agree to ~0.1%:
    # subject-collapsed input gives RMS off-diagonal 0.5225 (fold) vs 0.5236 (per_position),
    # healthy input 0.0642 vs 0.0643.
    #
    # It also does NOT rescue the subject dimension, and the reason is worth recording: the
    # same test shows BT's off-diagonal DOES respond to across-subject collapse (0.52 against
    # 0.064 healthy) under both formulations. So "the fold hides subject collapse from the
    # off-diagonal" is false. If a run collapses across subjects with a healthy-looking loss,
    # look at the variance SHARE instead — on a real run the across-subject component was 0.6%
    # of what BT normalises by, so a fully collapsed subject dimension perturbs the correlation
    # matrix by <1% and lambda cannot reach it. The lever there is a GAP-pooled BT term, whose
    # rows are subjects, not a different patch statistic.
    #
    # Centering happens first either way. "per_position" removes the per-position mean and
    # scale, so center_mode "position" is redundant with it; "double" still contributes.
    _per_pos = False
    # UNCENTERED parallel copy, for the sim/var terms only.
    #
    # `_center_patch_features` subtracts each view's OWN across-batch mean per (channel,
    # position). That is exactly the per-view constant offset the MSE term exists to
    # penalise — computing the MSE downstream of it would make the distance term blind for
    # precisely the reason the correlation term is. The correlation terms still use the
    # centered tensor: removing the shared anatomy at each position is what makes their
    # diagonal a statement about subjects rather than about anatomy.
    hz_raw = hz
    if hz.ndim == 4:
        hz = _center_patch_features(hz, center_mode)
        if patch_stat == "per_position":
            _per_pos = True
        else:
            hz = hz.permute(0, 1, 3, 2).reshape(hz.shape[0], -1, hz.shape[2])
            hz_raw = hz_raw.permute(0, 1, 3, 2).reshape(hz_raw.shape[0], -1, hz_raw.shape[2])

    if subsets is None or estimated_content_indices is None:
        subsets = [list(range(hz.shape[0]))]
        estimated_content_indices = [list(range(hz.shape[-1]))]

    total_loss = torch.zeros(1, device=hz.device, dtype=hz.dtype)
    sub_diags = []

    for si, (content_indices, subset) in enumerate(zip(estimated_content_indices, subsets)):
        hz_sub = hz[list(subset)]  # (n_views_sub, B, C)
        hz_raw_sub = hz_raw[list(subset)]  # same slice, uncentered — sim/var only
        n_view = hz_sub.shape[0]

        for i in range(n_view):
            for j in range(i + 1, n_view):
                # Select content features.  In the mask path we index the
                # active (content) channels instead of multiplying by the mask:
                # the per-channel standardisation below divides the mask scale
                # back out, and multiplying would leave the zeroed style channels
                # in the correlation matrix, each adding a spurious (0-1)^2 to
                # on_diag (a constant offset plus a straight-through gradient
                # that pulls style channels toward content).
                if soft_content_mask is not None:
                    active = soft_content_mask.reshape(-1).bool()  # (C,)
                    z_i = hz_sub[i][:, active]  # (B, k)
                    z_j = hz_sub[j][:, active]
                    r_i = hz_raw_sub[i][:, active]
                    r_j = hz_raw_sub[j][:, active]
                else:
                    z_i = hz_sub[i][:, content_indices]  # (B, d)
                    z_j = hz_sub[j][:, content_indices]
                    r_i = hz_raw_sub[i][:, content_indices]
                    r_j = hz_raw_sub[j][:, content_indices]

                # Standardise (zero mean, unit std per dimension).  Biased std (÷B) matches
                # the /B in the cross-correlation, so a perfectly correlated dimension
                # reaches C_ii = 1 exactly.  Under "per_position" the statistics are taken
                # across SAMPLES at each (dimension, position), so the correlation below is
                # an across-subject quantity averaged over positions.
                # Forced to fp32 with autocast OFF. The contraction runs over B*P rows in
                # patch mode and the /B lands AFTER it, so under AMP the raw Gram is emitted
                # in fp16 and overflows (65504) once the diagonal correlation passes
                # 65504/(B*P) — only 0.50 at B=256, P=512, which BT reaches by construction.
                # off_diag below then evaluates inf - inf = NaN. Casting the operands alone
                # does NOT fix it: autocast re-casts matmul inputs to fp16 whatever dtype
                # they arrive in, so the region has to be disabled explicitly.
                with torch.amp.autocast("cuda", enabled=False):
                    z_i = z_i.float()
                    z_j = z_j.float()
                    # Distance + variance terms, on the RAW features, BEFORE the per-channel
                    # standardisation below.
                    #
                    # That standardisation is precisely what makes on_diag blind to a per-view
                    # CONSTANT offset. Measured: adding one leaves on_diag at 0.3084 unchanged
                    # to four decimals while the two views become perfectly separable (view
                    # probe 0.503 -> 1.000) and MSE rises 105x. On this project every content
                    # channel had view-AUC 1.000 under BT and 0/44 above 0.7 under VICReg,
                    # whose invariance term is exactly this MSE. Correlation measures whether
                    # the two views move together; only a distance measures whether they are
                    # in the same place, and block-identifiability needs the latter
                    # (Yao et al. Lemma C.2 requires g_k(x_k) = g_k'(x_k') almost surely).
                    #
                    # The variance hinge is NOT optional once sim_coeff > 0: MSE alone is
                    # minimised by z_i = z_j = 0, and on_diag/off_diag are both scale-invariant,
                    # so no other term here can see that collapse. They ship together.
                    # Computed UNCONDITIONALLY so the calibration workflow works — run with
                    # both coefficients at 0, read these off TensorBoard, pick weights from the
                    # measured ratio. But under no_grad when neither coefficient is set, so a
                    # run that does not use the terms pays no autograd graph for them. That
                    # matters: the uncentered copy of the patch fold is (2, B*P, d), which at
                    # patch_grid 8^3 and B=128 is ~25 MB plus its graph, and cuDNN reports
                    # workspace exhaustion as "unable to find an engine" rather than as OOM.
                    _weighted = sim_coeff > 0 or std_coeff > 0
                    _ctx = contextlib.nullcontext() if _weighted else torch.no_grad()
                    with _ctx:
                        r_i = r_i.float()
                        r_j = r_j.float()
                        if sim_normalize:
                            # Per-channel, scale-free MSE. For zero-mean equal-variance views with
                            # per-channel correlation rho_c, MSE_c = 2*sigma_c^2*(1 - rho_c), so this
                            # reads (1 - mean rho) when the means match and EXCESS over that is the
                            # per-view offset. Two reasons to prefer it here:
                            #
                            # 1. Raw MSE scales as sigma^2, and the variance hinge is simultaneously
                            #    driving sigma up (measured: GAP feat_std 0.004 PRE-hinge against a
                            #    hinge target of 1, a 250x rescale => up to 62500x on the raw term;
                            #    post-hinge it parks past the target, ~1.1 on 23 Aug 2026). The two terms
                            #    fight, and sim_loss rises while alignment IMPROVES — which is exactly
                            #    what was observed at both coeff 1 and 0.1.
                            # 2. Per-channel normalisation equalises directions, so the term stops
                            #    being magnitude-weighted toward the top PCs. That matters because the
                            #    factor information sits in the low-variance tail (n@95% = 44 of 44).
                            #
                            # The denominator is DETACHED: without it, shrinking sigma is a free way to
                            # cut the ratio, which is the collapse the hinge exists to prevent.
                            _den = (0.5 * (r_i.var(dim=0, unbiased=False) + r_j.var(dim=0, unbiased=False))).detach()
                            sim_loss = (((r_i - r_j) ** 2) / (2.0 * _den + 1e-8)).mean()
                        else:
                            sim_loss = F.mse_loss(r_i, r_j)
                        _raw_std_i = r_i.std(dim=0, unbiased=False)
                        var_loss = F.relu(1.0 - _raw_std_i).mean() + F.relu(1.0 - r_j.std(dim=0, unbiased=False)).mean()
                    # Captured HERE: z_i is overwritten by its standardised self below, after
                    # which its std is 1.0 by construction and tells you nothing.
                    _raw_std_mean = _raw_std_i.mean()
                    if _per_pos:
                        B, d, P = z_i.shape
                        z_i = (z_i - z_i.mean(0, keepdim=True)) / (z_i.std(0, unbiased=False, keepdim=True) + 1e-6)
                        z_j = (z_j - z_j.mean(0, keepdim=True)) / (z_j.std(0, unbiased=False, keepdim=True) + 1e-6)
                        c = torch.einsum("bip,bjp->ij", z_i, z_j) / (B * P)
                    else:
                        B, d = z_i.shape
                        z_i = (z_i - z_i.mean(dim=0)) / (z_i.std(dim=0, unbiased=False) + 1e-6)
                        z_j = (z_j - z_j.mean(dim=0)) / (z_j.std(dim=0, unbiased=False) + 1e-6)
                        c = (z_i.T @ z_j) / B

                # ── Optional EMA of the cross-correlation across STEPS ─────────────────
                #
                # off_diag sums d(d-1) squared correlations, each estimated from B rows, so
                # even at the optimum it carries a sampling floor of d(d-1)/B: 1892/128 =
                # 14.78 at d=44, against a measured 17.999 at GAP. 82% of that term is noise,
                # and it is not a harmless constant — the gradient is 2*c_ij*dc_ij/dtheta, so
                # at the optimum the multiplier is pure sampling error, and because it shares
                # rows with dc/dtheta the encoder can lower it by fitting THIS batch's noise.
                #
                # Averaging c over steps cuts the floor by (1-m)/(1+m) — an effective row
                # count of B*(1+m)/(1-m), i.e. 25472 at B=128, m=0.99. The gradient direction
                # still comes from the current batch (only the CURRENT term carries grad); the
                # multiplier becomes a low-variance estimate of the true correlation, so at the
                # optimum it goes to zero instead of chasing noise. Verified on planted data:
                # floor 14.78 -> 0.075 while a true correlation of 0.15 reads back as 0.1505.
                #
                # This is the only fix that scales: reaching the same floor by batch size alone
                # needs ~5900 subjects per step. GAP benefits most — it is the one term whose
                # rows are subjects, and subjects are the scarce resource.
                #
                # Bias-corrected (Adam-style), so a zero init is unbiased from step 1 and a
                # resume costs only a noisier estimate for ~1/(1-m) steps rather than a
                # transient. The state is deliberately NOT checkpointed for that reason.
                #
                # NOTE: gradient magnitude on this term scales with (1-m), so lambd needs
                # raising when the EMA is enabled. `off_diag_inst` below is the un-EMA'd value,
                # logged so runs stay comparable with every pre-EMA number.
                c_inst = c
                if corr_ema is not None and corr_ema_decay > 0:
                    _key = (si, i, j)
                    # The Gumbel mask is learned, so the ACTIVE CHANNEL SET can change between
                    # steps. Averaging across such a change would mix different channels into
                    # the same matrix entry, so reset whenever the signature moves.
                    _sig = (tuple(c.shape), int(z_i.shape[-1]))
                    _st = corr_ema.get(_key)
                    if _st is None or _st["sig"] != _sig:
                        _st = {"c": torch.zeros_like(c), "t": 0, "sig": _sig}
                        corr_ema[_key] = _st
                    _m = float(corr_ema_decay)
                    _c_ema = _m * _st["c"].to(c.device, c.dtype) + (1.0 - _m) * c
                    _st["c"] = _c_ema.detach()
                    _st["t"] += 1
                    c = _c_ema / (1.0 - _m ** _st["t"])  # bias correction

                # Loss: push diagonal toward 1, off-diagonal toward 0
                on_diag = (c.diagonal() - 1).pow(2).sum()
                off_diag = c.pow(2).sum() - c.diagonal().pow(2).sum()
                loss = on_diag + lambd * off_diag + sim_coeff * sim_loss + std_coeff * var_loss
                total_loss = total_loss + loss

                with torch.no_grad():
                    sub_diags.append(
                        {
                            "top1_acc": 0.0,  # not applicable for BT
                            "pos_sim_mean": c.diagonal().mean().item(),
                            "pos_sim_std": c.diagonal().std().item(),
                            "neg_sim_mean": (c.sum() - c.diagonal().sum()).item() / max(d * d - d, 1),
                            "neg_sim_std": 0.0,
                            "on_diag_loss": on_diag.item(),
                            "off_diag_loss": off_diag.item(),
                            # Un-EMA'd single-batch value: what every pre-EMA run logged.
                            # Read THIS to compare against existing numbers; read
                            # off_diag_loss to see what is actually being optimised.
                            "off_diag_inst": (c_inst.pow(2).sum() - c_inst.diagonal().pow(2).sum()).item(),
                            # Logged unweighted so the four terms can be balanced from what
                            # they actually are, rather than by transplanting VICReg's 25/25/1
                            # (the mistake that made bt_lambda 200x too weak at d=44).
                            "sim_loss": sim_loss.item(),
                            "var_loss": var_loss.item(),
                            "feat_std_mean": _raw_std_mean.item(),
                        }
                    )

    if sub_diags:
        total_loss._contrastive_diag = _merge_diags(sub_diags)
    return total_loss


def vicreg_loss(
    hz,
    estimated_content_indices=None,
    subsets=None,
    soft_content_mask=None,
    sim_coeff=25.0,
    std_coeff=25.0,
    cov_coeff=1.0,
    center_mode="none",
    **_kwargs,
):
    """VICReg (Variance-Invariance-Covariance) loss over content channels.

    More stable than Barlow Twins at very small batch sizes because it
    explicitly enforces per-dimension variance via a hinge loss rather
    than relying on correlation normalisation.

    Args:
        hz: Encoder features, shape ``(n_views, B, C)`` or
            ``(n_views, B, C, P)`` for patch mode.
        estimated_content_indices: Per-subset content channel indices.
        subsets: View subsets.
        soft_content_mask: Optional differentiable mask.
        sim_coeff: Weight on the invariance (MSE) term.
        std_coeff: Weight on the variance (hinge) term.
        cov_coeff: Weight on the covariance (decorrelation) term.
        center_mode: Patch centering, applied only in patch mode — see
            ``_center_patch_features``. Without it the folded batch mixes samples and
            positions, so most of the per-channel variance the hinge sees is positional
            rather than across-subject: the invariance term is satisfied by the shared
            anatomy every subject has at that voxel, and the variance term can be met by
            spatial variation alone. ``"position"`` removes the shared pattern so both
            terms are about subjects.

    Returns:
        Scalar loss with ``._contrastive_diag`` attached.
    """
    # Patch mode: fold patches into batch  (n_views, B, C, P) → (n_views, B*P, C).
    # Centering must happen BEFORE the fold, while the position axis still exists.
    if hz.ndim == 4:
        hz = _center_patch_features(hz, center_mode)
        hz = hz.permute(0, 1, 3, 2).reshape(hz.shape[0], -1, hz.shape[2])

    if subsets is None or estimated_content_indices is None:
        subsets = [list(range(hz.shape[0]))]
        estimated_content_indices = [list(range(hz.shape[-1]))]

    total_loss = torch.zeros(1, device=hz.device, dtype=hz.dtype)
    sub_diags = []

    for content_indices, subset in zip(estimated_content_indices, subsets):
        hz_sub = hz[list(subset)]
        n_view = hz_sub.shape[0]

        for i in range(n_view):
            for j in range(i + 1, n_view):
                if soft_content_mask is not None:
                    z_i = hz_sub[i] * soft_content_mask
                    z_j = hz_sub[j] * soft_content_mask
                else:
                    z_i = hz_sub[i][:, content_indices]
                    z_j = hz_sub[j][:, content_indices]

                B, d = z_i.shape

                # --- Invariance: MSE between paired views ---
                sim_loss = F.mse_loss(z_i, z_j)

                # --- Variance: hinge loss to keep std above 1 ---
                std_i = z_i.std(dim=0)
                std_j = z_j.std(dim=0)
                var_loss = F.relu(1.0 - std_i).mean() + F.relu(1.0 - std_j).mean()

                # --- Covariance: decorrelate dimensions ---
                # fp32 with autocast off, for the same reason as barlow_twins_loss: in patch
                # mode these Gram matrices contract over B*P rows and the /(B-1) lands after
                # the matmul, so fp16 output overflows to inf. fill_diagonal_ below happens to
                # delete the entries that overflow first, but any strongly correlated PAIR
                # overflows too and survives into cov_loss.
                with torch.amp.autocast("cuda", enabled=False):
                    z_i_f, z_j_f = z_i.float(), z_j.float()
                    z_i_c = z_i_f - z_i_f.mean(dim=0)
                    z_j_c = z_j_f - z_j_f.mean(dim=0)
                    cov_i = (z_i_c.T @ z_i_c) / max(B - 1, 1)
                    cov_j = (z_j_c.T @ z_j_c) / max(B - 1, 1)
                # Zero out diagonal (we only penalise off-diagonal)
                cov_i.fill_diagonal_(0)
                cov_j.fill_diagonal_(0)
                cov_loss = (cov_i.pow(2).sum() + cov_j.pow(2).sum()) / d

                loss = sim_coeff * sim_loss + std_coeff * var_loss + cov_coeff * cov_loss
                total_loss = total_loss + loss

                with torch.no_grad():
                    sub_diags.append(
                        {
                            "top1_acc": 0.0,
                            "pos_sim_mean": F.cosine_similarity(z_i, z_j, dim=-1).mean().item(),
                            "pos_sim_std": F.cosine_similarity(z_i, z_j, dim=-1).std().item(),
                            "neg_sim_mean": 0.0,
                            "neg_sim_std": 0.0,
                            "sim_loss": sim_loss.item(),
                            "var_loss": var_loss.item(),
                            "cov_loss": cov_loss.item(),
                        }
                    )

    if sub_diags:
        total_loss._contrastive_diag = _merge_diags(sub_diags)
    return total_loss


def style_infonce_loss(hz_v0_style, hz_v1_style, tau=0.1):
    """Cross-view repulsive loss on style channels.

    Replaces the all-same-modality-as-positive formulation, which collapsed
    style to a binary modality signal.  Instead:

    1. **Cross-view repulsion**: same-subject style across views should
       differ (they have independent z_style), preventing content leakage.
    2. **Within-view diversity**: per-dimension variance regularization
       prevents all style features from collapsing to a single point.
    """
    B = hz_v0_style.shape[0]
    if B < 2:
        return hz_v0_style.sum() * 0.0

    z0 = F.normalize(hz_v0_style, dim=-1)
    z1 = F.normalize(hz_v1_style, dim=-1)

    cross_sim = (z0 * z1).sum(dim=-1).mean()

    std0 = hz_v0_style.std(dim=0)
    std1 = hz_v1_style.std(dim=0)
    var_reg = F.relu(1.0 - std0).mean() + F.relu(1.0 - std1).mean()

    return cross_sim + var_reg


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


def content_modality_adv_loss(hz_v0_content, hz_v1_content, head, lambd=1.0):
    """Gradient-reversal modality classifier on content.

    Encoder sees -lambd * grad, head sees +grad. At equilibrium content carries
    no modality-linear info, independent of the style bottleneck size.

    head: nn.Linear(C_content, 2) or a small MLP.
    Returns (loss, accuracy) — accuracy near 0.5 means invariance achieved.
    """
    B = hz_v0_content.shape[0]
    z = torch.cat([hz_v0_content, hz_v1_content], dim=0)  # (2B, C)
    y = torch.cat([torch.zeros(B, dtype=torch.long), torch.ones(B, dtype=torch.long)]).to(z.device)
    logits = head(grad_reverse(z, lambd))
    loss = F.cross_entropy(logits, y)
    with torch.no_grad():
        acc = (logits.argmax(-1) == y).float().mean().item()
    return loss, acc


def content_patch_modality_adv_loss(hz_v0_content, hz_v1_content, head, lambd=1.0):
    """Patch-level gradient-reversal modality classifier on content.

    hz_v0_content, hz_v1_content: (B, k, P) patch-level content features.
    Reshapes to (B*P, k) per view so the adversary sees each patch independently,
    penalising position-specific modality encoding that the pooled variant misses.
    """
    B, k, P = hz_v0_content.shape
    z0 = hz_v0_content.permute(0, 2, 1).reshape(B * P, k)
    z1 = hz_v1_content.permute(0, 2, 1).reshape(B * P, k)
    z = torch.cat([z0, z1], dim=0)
    y = torch.cat([torch.zeros(B * P, dtype=torch.long), torch.ones(B * P, dtype=torch.long)]).to(z.device)
    logits = head(grad_reverse(z, lambd))
    loss = F.cross_entropy(logits, y)
    with torch.no_grad():
        acc = (logits.argmax(-1) == y).float().mean().item()
    return loss, acc


def style_modality_ce_loss(hz_v0_style, hz_v1_style, head):
    """Sufficiency: style must be linearly modality-predictive.

    Normal gradient flow — pushes style to *carry* modality info (and, via the
    capacity it frees in style, demographic info correlated with modality).
    Returns (loss, accuracy). Accuracy near 1.0 means style is sufficient.
    """
    B = hz_v0_style.shape[0]
    z = torch.cat([hz_v0_style, hz_v1_style], dim=0)
    y = torch.cat([torch.zeros(B, dtype=torch.long), torch.ones(B, dtype=torch.long)]).to(z.device)
    logits = head(z)
    loss = F.cross_entropy(logits, y)
    with torch.no_grad():
        acc = (logits.argmax(-1) == y).float().mean().item()
    return loss, acc


class BaurLoss(object):
    def __init__(self, lambda_reconstruction=1):
        super().__init__()

        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_gdl = 0

        # Use mean instead of sum for proper scaling with 3D images
        self.l1_loss = lambda x, y: PairwiseDistance(p=1)(x.view(x.shape[0], -1), y.view(y.shape[0], -1)).mean()
        self.l2_loss = lambda x, y: PairwiseDistance(p=2)(x.view(x.shape[0], -1), y.view(y.shape[0], -1)).mean()

    def __call__(self, originals, reconstructions):
        summaries = {}

        l1_reconstruction = self.l1_loss(originals, reconstructions) * self.lambda_reconstruction
        l2_reconstruction = self.l2_loss(originals, reconstructions) * self.lambda_reconstruction

        summaries[("summaries", "scalar", "L1-Reconstruction-Loss")] = l1_reconstruction.item()
        summaries[("summaries", "scalar", "L2-Reconstruction-Loss")] = l2_reconstruction.item()
        summaries[("summaries", "scalar", "Lambda-Reconstruction")] = self.lambda_reconstruction

        originals_gradients = self.__image_gradients(originals)
        reconstructions_gradients = self.__image_gradients(reconstructions)

        l1_gdl = (
            self.l1_loss(originals_gradients[0], reconstructions_gradients[0])
            + self.l1_loss(originals_gradients[1], reconstructions_gradients[1])
            + self.l1_loss(originals_gradients[2], reconstructions_gradients[2])
        ) * self.lambda_gdl

        l2_gdl = (
            self.l2_loss(originals_gradients[0], reconstructions_gradients[0])
            + self.l2_loss(originals_gradients[1], reconstructions_gradients[1])
            + self.l2_loss(originals_gradients[2], reconstructions_gradients[2])
        ) * self.lambda_gdl

        summaries[("summaries", "scalar", "L1-Image_Gradient-Loss")] = l1_gdl.item()
        summaries[("summaries", "scalar", "L2-Image_Gradient-Loss")] = l2_gdl.item()
        summaries[("summaries", "scalar", "Lambda-Image_Gradient")] = self.lambda_gdl

        loss_total = l1_reconstruction + l2_reconstruction + l1_gdl + l2_gdl

        summaries[("summaries", "scalar", "Total_Loss")] = loss_total.item()

        return loss_total, summaries

    def set_lambda_reconstruction(self, lambda_reconstruction):
        self.lambda_reconstruction = lambda_reconstruction
        return self.lambda_reconstruction

    def set_lambda_gdl(self, lambda_gdl):
        self.lambda_gdl = lambda_gdl
        return self.lambda_gdl

    @staticmethod
    def __image_gradients(image):
        input_shape = image.shape
        batch_size, features, depth, height, width = input_shape

        dz = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
        dy = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
        dx = image[:, :, :, :, 1:] - image[:, :, :, :, :-1]

        dzz = tensor(()).new_zeros(
            (batch_size, features, 1, height, width),
            device=image.device,
            dtype=dz.dtype,
        )
        dz = cat([dz, dzz], 2)
        dz = reshape(dz, input_shape)

        dyz = tensor(()).new_zeros((batch_size, features, depth, 1, width), device=image.device, dtype=dy.dtype)
        dy = cat([dy, dyz], 3)
        dy = reshape(dy, input_shape)

        dxz = tensor(()).new_zeros(
            (batch_size, features, depth, height, 1),
            device=image.device,
            dtype=dx.dtype,
        )
        dx = cat([dx, dxz], 4)
        dx = reshape(dx, input_shape)

        return dx, dy, dz


class BaselineLoss(torch.nn.Module):
    def __init__(self):
        super(BaselineLoss, self).__init__()

        self.pixel_factor = 1.0

        self.perceptual_factor = 0.002
        self.n_slices = 16
        self.perceptual_function = LPIPS(net="alex")

        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor) -> torch.Tensor:
        # Unpacking elements — compute entirely in float32 to avoid
        # float16 overflow (FFT magnitudes and perceptual scaling can
        # exceed the float16 range after enough training steps).
        x = y.float()
        y = network_output["reconstruction"][0].float()

        # The clamp below bounds large FINITE values but does NOT stop NaN — torch.clamp
        # propagates it (clamp(nan, -1, 1) == nan). The decoder runs in fp16 under autocast
        # with no output normalisation (--no-final-recon-norm), so once its activations pass
        # 65504 they become inf, and the decoder's own GroupNorm turns inf - inf into NaN.
        # Without this the whole run dies: the step-level guard skips every subsequent batch,
        # so one bad forward pass paralyses training permanently rather than costing one batch.
        if not torch.isfinite(y).all():
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

        # The decoder has no output activation, so y can have arbitrarily large values, and
        # LPIPS amplifies outliers into NaN.  Bound it for the PERCEPTUAL path only.
        #
        # It used to be clamped for BOTH terms, which made saturation an ABSORBING STATE:
        # torch.clamp has zero gradient outside its bounds, so a voxel whose raw output left
        # [-1, 1] stopped contributing any gradient and the pixel loss could not pull it back
        # (the previous comment here claimed it could — it cannot, it was reading the clamped
        # tensor).  Because each saturated voxel removes its own corrective signal, the
        # failure is self-reinforcing: output scale drifts up, saturation spreads, the
        # remaining gradient shrinks, and past a threshold it runs away and Loss/Recon jumps
        # to a permanent plateau at the cost of predicting +-1.  Observed at ~10k steps across
        # nearly every run, VQ unaffected.  The same dead gradient is why the fixed_reference
        # cross-style swap failed.
        #
        # Feeding the RAW y to the pixel term is safe: L1's gradient w.r.t. y is sign(x - y),
        # bounded at +-1 per voxel however large y gets, so there is no gradient explosion to
        # protect against here — only a loss VALUE that grows, which is the signal we want.
        y_percep = y.clamp(-1.0, 1.0)
        with torch.no_grad():
            _sat = (y.abs() > 1.0).float().mean()
            self.summaries[TBSummaryTypes.SCALAR]["Recon-Saturated_Fraction"] = _sat
            self.summaries[TBSummaryTypes.SCALAR]["Recon-Raw_Output_Std"] = y.std()

        q_losses = network_output["quantization_losses"]

        mask = network_output.get("mask")
        if mask is not None:
            mask = mask.float()
            # Zero reconstruction outside the brain so the perceptual term sees the same
            # support as the pixel term.
            y = y * mask
            y_percep = y_percep * mask

        loss = self._calculate_pixel_loss(
            x,
            y,
            mask=mask,
            n_views=int(network_output.get("n_views", 1) or 1),
            view_balance=float(network_output.get("view_balance", 0.0) or 0.0),
        ) + self._calculate_perceptual_loss(x, y_percep)

        # The VQ commitment cost is added HERE and again in main_multimodal as Loss/VQ
        # (`vq_loss = sum(diffs) * args.vq_commitment_weight`), so the effective weight has
        # always been 1.0 + vq_commitment_weight — five times what --vq-commitment-weight 0.25
        # advertises. That matters beyond bookkeeping: `diff` is (quantize.detach() - x)^2, so
        # this term is minimised by SHRINKING the representation entering the codebook, and
        # under a decorrelating contrastive objective a 256-entry codebook cannot cover the
        # code at full rank, making the cost irreducible except by collapsing rank. Over-
        # weighting that pressure by 5x is not a neutral accounting error.
        #
        # single_count keeps the per-level summaries (they are the only place the commitment
        # cost is broken out per level) but drops the addition, so --vq-commitment-weight
        # becomes the single source of truth. Default False = bit-identical to every prior run.
        _single = bool(network_output.get("single_count_commitment", False))
        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()

            self.summaries[TBSummaryTypes.SCALAR][f"Loss-MSE-VQ{idx}_Commitment_Cost"] = q_loss.detach()

            if not _single:
                loss = loss + q_loss

        return loss

    def _masked_mae(self, x, y, mask):
        if mask is None:
            return F.l1_loss(x, y)
        # Mean over brain voxels only — avoids diluting the loss with
        # background zeros whose target and prediction are both ~0.
        return ((x - y).abs() * mask).sum() / mask.sum().clamp_min(1.0)

    def _calculate_pixel_loss(self, x, y, mask=None, n_views=1, view_balance=0.0) -> torch.Tensor:
        """Masked L1, optionally re-weighted so each view's gradient scales with its own error.

        The default path is a single mean over the CONCATENATED [v0; v1] batch. That has two
        consequences worth stating, both measured on this project:

        1. It cannot show which view is failing. A run was found with view0 MAE 0.073 and view1
           MAE 0.576 whose aggregate looked like a modest global rise.
        2. L1's gradient is sign(x - y) — constant magnitude however wrong the output is. A view
           at 0.576 therefore generates exactly the same per-voxel pull as one at 0.073, so a
           collapsed or overshooting view is a STABLE equilibrium and training never leaves it.

        ``view_balance`` p > 0 fixes (2) by weighting view v by its own DETACHED error^p,
        renormalised to mean 1 so the overall loss scale (and hence its balance against the
        contrastive and commitment terms) is unchanged. At p=1 the example above gives view 1
        about 7.9x the gradient of view 0 — the pull the plain mean is missing — while leaving
        the per-voxel L1 metric alone, so it does not introduce the blur an MSE term would.
        The weight is detached deliberately: it is a gradient re-scaling, not a new objective,
        so it cannot be reduced by making a view look worse.

        p=0 (default) falls through to the original computation and is bit-identical.
        """
        per_view = None
        if n_views >= 2 and x.shape[0] % n_views == 0:
            B = x.shape[0] // n_views
            per_view = [
                self._masked_mae(
                    x[v * B : (v + 1) * B],
                    y[v * B : (v + 1) * B],
                    None if mask is None else mask[v * B : (v + 1) * B],
                )
                for v in range(n_views)
            ]
            # Logged unconditionally: the per-view split is the readout this loss was missing,
            # independent of whether the re-weighting is switched on.
            for v, pv in enumerate(per_view):
                self.summaries[TBSummaryTypes.SCALAR][f"Loss-MAE-Reconstruction_view{v}"] = pv.detach()

        if per_view is not None and view_balance > 0:
            stacked = torch.stack(per_view)
            w = stacked.detach().clamp_min(1e-8) ** float(view_balance)
            # Two-stage normalisation. The first line only sets the mean weight to 1; that is
            # NOT enough to leave the loss VALUE alone, because a weighted mean with weights
            # proportional to the values is the contraharmonic mean, which exceeds the plain
            # mean whenever the views disagree (measured: +71% at a 12x view imbalance). Left
            # uncorrected the recon term would silently gain weight against the contrastive and
            # commitment terms exactly when a view breaks, confounding the very comparison this
            # flag exists to make. The second line rescales so the weighted mean equals the
            # plain mean again. Both factors are detached and uniform across views, so the
            # RATIO between the views' gradients — the entire point — is untouched.
            w = w * (n_views / w.sum().clamp_min(1e-12))
            _plain = stacked.detach().sum()
            _weighted = (w * stacked.detach()).sum()
            w = w * (_plain / _weighted.clamp_min(1e-12))
            loss = (w * stacked).sum() / n_views
            self.summaries[TBSummaryTypes.SCALAR]["Recon-View_Weight_Max"] = w.max()
            self.summaries[TBSummaryTypes.SCALAR]["Recon-View_Weight_Ratio"] = w.max() / w.min().clamp_min(1e-12)
        elif mask is None:
            loss = F.l1_loss(x, y)
        else:
            diff = (x - y).abs() * mask
            denom = mask.sum().clamp_min(1.0)
            loss = diff.sum() / denom
        loss = loss * self.pixel_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-MAE-Reconstruction"] = loss.detach()

        return loss

    def _calculate_perceptual_loss(self, x, y) -> torch.Tensor:
        def _lpips_on_slices(x_vol, y_vol, perm_dims):
            x_p = x_vol.permute(*perm_dims)
            n_slices_total = x_p.shape[1]
            indices = torch.randperm(n_slices_total, device=x_vol.device)[: self.n_slices]
            sel_x = x_p[:, indices].contiguous().flatten(0, 1).detach()
            del x_p
            sel_y = y_vol.permute(*perm_dims)[:, indices].contiguous().flatten(0, 1)
            if sel_x.shape[-1] > 96 or sel_x.shape[-2] > 96:
                _target = (min(sel_x.shape[-2], 96), min(sel_x.shape[-1], 96))
                sel_x = F.adaptive_avg_pool2d(sel_x, _target)
                sel_y = F.adaptive_avg_pool2d(sel_y, _target)
            if sel_x.shape[1] == 1:
                sel_x = sel_x.expand(-1, 3, -1, -1)
                sel_y = sel_y.expand(-1, 3, -1, -1)
            p_loss = torch.mean(self.perceptual_function.forward(sel_x.float(), sel_y.float()))
            if not torch.isfinite(p_loss):
                return torch.zeros(1, device=x_vol.device, dtype=torch.float32, requires_grad=True).squeeze()
            return p_loss

        orientations = [
            ("Sagittal", (0, 2, 1, 3, 4)),
            ("Axial", (0, 4, 1, 2, 3)),
            ("Coronal", (0, 3, 1, 2, 4)),
        ]

        total_p_loss = torch.zeros(1, device=x.device, dtype=torch.float32)
        for name, perm_dims in orientations:
            p_loss = _lpips_on_slices(x, y, perm_dims=perm_dims)
            self.summaries[TBSummaryTypes.SCALAR][f"Loss-Perceptual_{name}-Reconstruction"] = p_loss.detach()
            total_p_loss = total_p_loss + p_loss

        loss = total_p_loss * self.perceptual_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual-Reconstruction"] = loss.detach()

        return loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries


class JukeboxPerceptualLoss(torch.nn.Module):
    """
    Loss made of three components:
        * Perceptual - based on the lpips library, with 2.5D approach for 3D volumes
        * Spectral - based on the magnitude of FFT
        * Pixel - MSE
    """

    def __init__(
        self,
        dimensions: int,
        include_pixel_loss: bool = True,
        is_fake_3d: bool = True,
        drop_ratio: float = 0.0,
        fake_3d_axis: Tuple[int, ...] = (2, 3, 4),
        lpips_kwargs: Dict = None,
        lpips_normalize: bool = False,
        fft_kwargs: Dict = None,
        pixel_loss_type: str = "mse",
    ):
        super(JukeboxPerceptualLoss, self).__init__()

        if not (dimensions in [2, 3]):
            raise NotImplementedError("Perceptual loss is implemented only in 2D and 3D.")

        if dimensions == 3 and is_fake_3d is False:
            raise NotImplementedError("True 3D perceptual loss is not implemented yet.")

        if pixel_loss_type not in ("mse", "l1"):
            raise ValueError(f"pixel_loss_type must be 'mse' or 'l1', got {pixel_loss_type!r}")

        self.dimensions = dimensions
        self.include_pixel_loss = include_pixel_loss
        self.pixel_loss_type = pixel_loss_type

        self.lpips_kwargs = (
            {
                "pretrained": True,
                "net": "alex",
                "version": "0.1",
                "lpips": True,
                "spatial": False,
                "pnet_rand": False,
                "pnet_tune": False,
                "use_dropout": True,
                "model_path": None,
                "eval_mode": True,
                "verbose": False,
            }
            if lpips_kwargs is None
            else lpips_kwargs
        )
        self.fake_3D_views = (
            (
                []
                + ([((0, 2, 1, 3, 4), (1, 3, 4))] if 2 in fake_3d_axis else [])
                + ([((0, 3, 1, 2, 4), (1, 2, 4))] if 3 in fake_3d_axis else [])
                + ([((0, 4, 1, 2, 3), (1, 2, 3))] if 4 in fake_3d_axis else [])
            )
            if is_fake_3d
            else None
        )
        self.keep_ratio = 1 - drop_ratio
        self.lpips_normalize = lpips_normalize
        self.perceptual_function = LPIPS(**self.lpips_kwargs) if self.dimensions == 2 or is_fake_3d else None
        self.perceptual_factor = 0.001
        self.fft_factor: float = 1.0
        self.fft_kwargs = (
            {"s": None, "dim": tuple(range(2, self.dimensions + 2)), "norm": "ortho"}
            if fft_kwargs is None
            else fft_kwargs
        )
        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor) -> torch.Tensor:
        y = y.float()
        y_pred = network_output["reconstruction"][0].float()
        # Decoder has no output activation; clamp to target range so FFT
        # magnitudes and LPIPS don't amplify early-training outliers into NaN.
        y_pred = y_pred.clamp(-1.0, 1.0)
        q_losses = network_output["quantization_losses"]

        mask = network_output.get("mask")
        if mask is not None:
            mask = mask.float()
            y_pred = y_pred * mask

        y_amplitude = self._get_fft_amplitude(y)
        y_pred_amplitude = self._get_fft_amplitude(y_pred)

        loss = F.mse_loss(y_pred_amplitude, y_amplitude) * self.fft_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Spectral-Reconstruction"] = loss.detach()
        self.summaries[TBSummaryTypes.SCALAR]["Auxiliary-FFT_Factor"] = self.fft_factor

        if self.dimensions == 3 and self.fake_3D_views:
            for idx, fake_views in enumerate(self.fake_3D_views):
                p_loss = (
                    self._calculate_fake_3d_loss(
                        y=y,
                        y_pred=y_pred,
                        permute_dims=fake_views[0],
                        view_dims=fake_views[1],
                    )
                    * self.perceptual_factor
                )
                p_loss = p_loss.mean()
                self.summaries[TBSummaryTypes.SCALAR][f"Loss-Perceptual_{idx}-Reconstruction"] = p_loss.detach()
                loss = loss + p_loss
        else:
            y_in, y_pred_in = y, y_pred
            if y_in.shape[1] == 1:
                y_in = y_in.expand(-1, 3, -1, -1)
                y_pred_in = y_pred_in.expand(-1, 3, -1, -1)
            p_loss_raw = self.perceptual_function.forward(y_in, y_pred_in, normalize=self.lpips_normalize).mean()
            if not torch.isfinite(p_loss_raw):
                p_loss_raw = torch.zeros(1, device=y.device, dtype=torch.float32, requires_grad=True).squeeze()
            p_loss = p_loss_raw * self.perceptual_factor
            self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual-Reconstruction"] = p_loss.detach()
            loss = loss + p_loss

        self.summaries[TBSummaryTypes.SCALAR]["Auxiliary-Perceptual_Factor"] = self.perceptual_factor

        if self.include_pixel_loss:
            if mask is not None:
                diff = (y_pred - y).abs() if self.pixel_loss_type == "l1" else (y_pred - y).pow(2)
                pixel_loss = (diff * mask).sum() / mask.sum().clamp_min(1.0)
            elif self.pixel_loss_type == "l1":
                pixel_loss = F.l1_loss(y_pred, y)
            else:
                pixel_loss = F.mse_loss(y_pred, y)
            key = "Loss-L1-Reconstruction" if self.pixel_loss_type == "l1" else "Loss-MSE-Reconstruction"
            self.summaries[TBSummaryTypes.SCALAR][key] = pixel_loss.detach()
            loss = loss + pixel_loss

        # Same double-count as BaselineLoss — see the note there.
        _single = bool(network_output.get("single_count_commitment", False))
        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()
            self.summaries[TBSummaryTypes.SCALAR][f"Loss-MSE-VQ{idx}_Commitment_Cost"] = q_loss.detach()
            if not _single:
                loss = loss + q_loss

        return loss

    def _calculate_fake_3d_loss(
        self,
        y: torch.Tensor,
        y_pred: torch.Tensor,
        permute_dims: Tuple[int, int, int, int, int],
        view_dims: Tuple[int, int, int],
    ):
        y_slices = (
            y.permute(*permute_dims)
            .contiguous()
            .view(-1, y.shape[view_dims[0]], y.shape[view_dims[1]], y.shape[view_dims[2]])
        )
        y_pred_slices = (
            y_pred.permute(*permute_dims)
            .contiguous()
            .view(-1, y_pred.shape[view_dims[0]], y_pred.shape[view_dims[1]], y_pred.shape[view_dims[2]])
        )
        indices = torch.randperm(y_pred_slices.shape[0], device=y_pred_slices.device)[
            : int(y_pred_slices.shape[0] * self.keep_ratio)
        ]
        y_pred_slices = y_pred_slices[indices]
        y_slices = y_slices[indices]
        # LPIPS backbones are pretrained on 3-channel RGB; MRI is 1-channel.
        if y_slices.shape[1] == 1:
            y_slices = y_slices.expand(-1, 3, -1, -1)
            y_pred_slices = y_pred_slices.expand(-1, 3, -1, -1)
        p_loss = torch.mean(self.perceptual_function.forward(y_slices, y_pred_slices, normalize=self.lpips_normalize))
        # LPIPS can return NaN on near-constant slices (internal feature
        # normalization divides by near-zero norm). Fall back to zero.
        if not torch.isfinite(p_loss):
            return torch.zeros(1, device=y.device, dtype=torch.float32, requires_grad=True).squeeze()
        return p_loss

    def _get_fft_amplitude(self, images: torch.Tensor) -> torch.Tensor:
        # Squared magnitude (|z|² = real² + imag²): avoids the NaN gradient of
        # sqrt at zero (common for brain-masked background voxels). rfftn
        # exploits conjugate symmetry to roughly halve peak memory.
        img_fft = rfftn(input=images, **self.fft_kwargs)
        return img_fft.real.pow(2) + img_fft.imag.pow(2)

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries

    def get_perceptual_factor(self) -> float:
        return self.perceptual_factor

    def set_perceptual_factor(self, perceptual_factor: float) -> float:
        self.perceptual_factor = perceptual_factor
        return self.get_perceptual_factor()

    def get_fft_factor(self) -> float:
        return self.fft_factor

    def set_fft_factor(self, fft_factor: float) -> float:
        self.fft_factor = fft_factor
        return self.get_fft_factor()
