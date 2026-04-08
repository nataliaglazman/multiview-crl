"""Definition of loss functions."""

from abc import ABC, abstractmethod
from typing import Dict, List

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
        pow: bool = True,
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
):
    """
    Patch-level InfoNCE: aligns corresponding spatial patches across views.

    Same interface as ``infonce_loss`` but expects ``hz`` with shape
    ``(n_views, B, C, P)`` where P is the number of spatial patches.
    Computes InfoNCE independently per patch position and averages.

    This preserves spatial correspondence between views — patches at the
    same anatomical location should align, providing a much richer training
    signal than global average pooling.
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
    """
    n_view, B, C, P = hz_subset.shape

    # Apply content selection
    if soft_content_mask is not None:
        # (1, C) → (1, 1, C, 1) for broadcasting with (n_views, B, C, P)
        hz_c = hz_subset * soft_content_mask.view(1, 1, -1, 1)
    else:
        hz_c = hz_subset[:, :, content_indices, :]

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
                loss_ij = criterion(scores.reshape(P * B, B), targets.repeat(P))

                scores_ji = scores.transpose(-1, -2)
                loss_ji = criterion(scores_ji.reshape(P * B, B), targets.repeat(P))

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

                loss = criterion(raw_scores.reshape(P * 2 * B, 2 * B), targets.repeat(P))
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

    Returns:
        Scalar loss with ``._contrastive_diag`` attached.
    """
    # Patch mode: fold patches into batch  (n_views, B, C, P) → (n_views, B*P, C)
    if hz.ndim == 4:
        hz = hz.permute(0, 1, 3, 2).reshape(hz.shape[0], -1, hz.shape[2])

    if subsets is None or estimated_content_indices is None:
        subsets = [list(range(hz.shape[0]))]
        estimated_content_indices = [list(range(hz.shape[-1]))]

    total_loss = torch.zeros(1, device=hz.device, dtype=hz.dtype)
    sub_diags = []

    for content_indices, subset in zip(estimated_content_indices, subsets):
        hz_sub = hz[list(subset)]  # (n_views_sub, B, C)
        n_view = hz_sub.shape[0]

        for i in range(n_view):
            for j in range(i + 1, n_view):
                # Select content features
                if soft_content_mask is not None:
                    z_i = hz_sub[i] * soft_content_mask  # (B, C)
                    z_j = hz_sub[j] * soft_content_mask
                else:
                    z_i = hz_sub[i][:, content_indices]  # (B, d)
                    z_j = hz_sub[j][:, content_indices]

                B, d = z_i.shape

                # Batch-normalise (zero mean, unit std per dimension)
                z_i = (z_i - z_i.mean(dim=0)) / (z_i.std(dim=0) + 1e-6)
                z_j = (z_j - z_j.mean(dim=0)) / (z_j.std(dim=0) + 1e-6)

                # Cross-correlation matrix  (d, d)
                c = (z_i.T @ z_j) / B

                # Loss: push diagonal toward 1, off-diagonal toward 0
                on_diag = (c.diagonal() - 1).pow(2).sum()
                off_diag = c.pow(2).sum() - c.diagonal().pow(2).sum()
                loss = on_diag + lambd * off_diag
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

    Returns:
        Scalar loss with ``._contrastive_diag`` attached.
    """
    # Patch mode: fold patches into batch  (n_views, B, C, P) → (n_views, B*P, C)
    if hz.ndim == 4:
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
                z_i_c = z_i - z_i.mean(dim=0)
                z_j_c = z_j - z_j.mean(dim=0)
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
        self.n_slices = 4  # Reduced from 32 → 8 → 4; compensated by 6× scale factor
        self.perceptual_function = LPIPS(net="squeeze")

        self.fft_factor = 1.0

        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor) -> torch.Tensor:
        # Unpacking elements — compute entirely in float32 to avoid
        # float16 overflow (FFT magnitudes and perceptual scaling can
        # exceed the float16 range after enough training steps).
        x = y.float()
        y = network_output["reconstruction"][0].float()

        # The decoder has no output activation, so y can have arbitrarily
        # large values.  Clamp to the expected input range [-1, 1] to
        # prevent the FFT magnitude and LPIPS from amplifying outliers
        # into NaN.  Gradients still flow through non-clamped voxels;
        # the clamp only kills gradient for values already far outside
        # the valid range (desired — push them back via the pixel loss,
        # not via an exploding FFT gradient).
        y = y.clamp(-1.0, 1.0)

        q_losses = network_output["quantization_losses"]

        loss = (
            self._calculate_pixel_loss(x, y)
            + self._calculate_frequency_loss(x, y)
            + self._calculate_perceptual_loss(x, y)
        )

        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()

            self.summaries[TBSummaryTypes.SCALAR][f"Loss-MSE-VQ{idx}_Commitment_Cost"] = q_loss.detach()

            loss = loss + q_loss

        return loss

    def _calculate_frequency_loss(self, x, y) -> torch.Tensor:
        # rfftn exploits conjugate symmetry of real inputs → output is ~half
        # the size of fftn along the last dim, cutting peak memory ~50%.
        # Per-sample loop keeps only one sample's FFT pair alive at a time,
        # further reducing peak VRAM for large 3D volumes.
        with torch.amp.autocast("cuda", enabled=False):
            loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)
            B = x.shape[0]
            for i in range(B):
                xi = (x[i : i + 1].float() + 1.0) / 2.0
                yi = (y[i : i + 1].float() + 1.0) / 2.0
                x_mag = torch.abs(rfftn(xi, norm="ortho"))
                del xi
                y_mag = torch.abs(rfftn(yi, norm="ortho"))
                del yi
                loss = loss + F.mse_loss(x_mag, y_mag)
                del x_mag, y_mag
            loss = loss / B

        loss = loss * self.fft_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Jukebox-Reconstruction"] = loss.detach()

        return loss

    def _calculate_pixel_loss(self, x, y) -> torch.Tensor:
        loss = F.l1_loss(x, y)
        loss = loss * self.pixel_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-MAE-Reconstruction"] = loss.detach()

        return loss

    def _calculate_perceptual_loss(self, x, y) -> torch.Tensor:
        # LPIPS backbone weights are frozen (requires_grad=False), so autograd
        # will not accumulate gradients into the backbone parameters.  However,
        # gradients DO need to flow back through the LPIPS computation to the
        # reconstruction tensor `y` so that the decoder is trained by the
        # perceptual objective.
        #
        # Speed optimisation: instead of computing LPIPS on all 3 orientations
        # every step (3× SqueezeNet passes), we randomly sample ONE orientation
        # per step and multiply by 3.  The expected gradient is identical, but
        # each step is ~3× cheaper for the perceptual component.

        def _lpips_on_slices(x_vol, y_vol, perm_dims):
            """Extract 2D slices along one orientation and compute LPIPS."""
            x_p = x_vol.permute(*perm_dims)  # (B, n_slices_total, C, H, W)
            n_slices_total = x_p.shape[1]
            indices = torch.randperm(n_slices_total, device=x_vol.device)[: self.n_slices]
            sel_x = x_p[:, indices].contiguous().flatten(0, 1).detach()
            del x_p
            sel_y = y_vol.permute(*perm_dims)[:, indices].contiguous().flatten(0, 1)
            # Cap spatial size to 96×96 to normalize memory across orientations
            if sel_x.shape[-1] > 96 or sel_x.shape[-2] > 96:
                _target = (min(sel_x.shape[-2], 96), min(sel_x.shape[-1], 96))
                sel_x = F.adaptive_avg_pool2d(sel_x, _target)
                sel_y = F.adaptive_avg_pool2d(sel_y, _target)
            p_loss = torch.mean(self.perceptual_function.forward(sel_x.float(), sel_y.float()))
            return p_loss

        orientations = [
            ("Sagittal", (0, 2, 1, 3, 4)),
            ("Axial", (0, 4, 1, 2, 3)),
            ("Coronal", (0, 3, 1, 2, 4)),
        ]

        # Randomly pick one orientation; multiply by 3 to keep expected value
        chosen_idx = torch.randint(len(orientations), (1,)).item()
        name, perm_dims = orientations[chosen_idx]
        p_loss = _lpips_on_slices(x, y, perm_dims=perm_dims)
        self.summaries[TBSummaryTypes.SCALAR][f"Loss-Perceptual_{name}-Reconstruction"] = p_loss.detach()

        # 3× for random orientation (1 of 3), 2× for halved slices (4 vs original 8)
        loss = p_loss * 6.0 * self.perceptual_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual-Reconstruction"] = loss.detach()

        return loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries
