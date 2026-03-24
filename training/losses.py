"""Definition of loss functions."""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from lpips import LPIPS
from torch import cat, reshape, tensor
from torch.fft import fftn
from torch.nn import PairwiseDistance

from utils.utils import TBSummaryTypes


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

    Returns:
        torch.Tensor: The calculated InfoNCE loss.

    """
    if estimated_content_indices is None:
        # Use all feature dimensions as content when no content indices are provided
        content_indices = list(range(hz.shape[-1]))
        return infonce_base_loss(hz, content_indices, sim_metric, criterion, projector, tau)
    else:
        total_loss = torch.zeros(1).type_as(hz)
        for est_content_indices, subset in zip(estimated_content_indices, subsets):
            total_loss += infonce_base_loss(
                hz[list(subset), ...],
                est_content_indices,
                sim_metric,
                criterion,
                projector,
                tau,
                soft_content_mask=soft_content_mask,
            )
        return total_loss


def infonce_base_loss(
    hz_subset, content_indices, sim_metric, criterion, projector=None, tau=1.0, soft_content_mask=None
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

    Returns:
        torch.Tensor: Total loss value.
    """

    n_view = len(hz_subset)
    d = hz_subset.shape[1]  # batch size
    SIM = [[None] * n_view for _ in range(n_view)]

    projector = projector or (lambda x: x)

    for i in range(n_view):
        for j in range(n_view):
            if j >= i:
                if soft_content_mask is not None:
                    # Differentiable masking: gradients flow to channel_logits
                    hz_i = hz_subset[i] * soft_content_mask  # (batch, C)
                    hz_j = hz_subset[j] * soft_content_mask
                else:
                    hz_i = hz_subset[i][..., content_indices]  # (batch, content_dims)
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
    for i in range(n_view):
        for j in range(n_view):
            if i < j:
                raw_scores1 = torch.cat([SIM[i][j], SIM[i][i]], dim=-1)
                raw_scores2 = torch.cat([SIM[j][j], SIM[j][i]], dim=-1)
                raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)  # (2d, 2d)
                targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)
                total_loss_value += criterion(raw_scores, targets)
    return total_loss_value


def moco_infonce_loss(q, k, queue, content_indices, tau=1.0, soft_content_mask=None):
    """
    MoCo-style InfoNCE loss for a single view pair using a queue of negatives.

    For each ordered pair (i, j) with i < j we treat q[i] as query matched to k[j]
    as the positive key, and all queue columns as negatives (and vice versa for j→i).

    Args:
        q  (torch.Tensor): Online (query) embeddings,   shape (n_views, B, C).
        k  (torch.Tensor): Momentum (key) embeddings,   shape (n_views, B, C).
                           Must already be detached from the computation graph.
        queue (torch.Tensor): Negative key queue,        shape (C, queue_size).
        content_indices (list[int]): Channel indices to use (fallback when
            soft_content_mask is None).
        tau (float): Temperature scaling factor.
        soft_content_mask: Optional differentiable (0/1) mask, shape (1, C).
            When provided, content selection uses ``features * mask`` so
            gradients flow back to channel_logits via Gumbel straight-through.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    n_view = q.shape[0]
    total_loss = torch.zeros(1, device=q.device, dtype=q.dtype)

    # Project to content dimensions and L2-normalise
    if soft_content_mask is not None:
        # Differentiable masking — gradients flow to channel_logits
        q_c = F.normalize(q * soft_content_mask, dim=-1)  # (n_views, B, C)
        k_c = F.normalize(k * soft_content_mask, dim=-1)  # (n_views, B, C)
        # queue shape is (C, Q) — mask along dim 0
        queue_c = F.normalize(queue * soft_content_mask.squeeze(0).unsqueeze(-1), dim=0)
    else:
        q_c = F.normalize(q[..., content_indices], dim=-1)  # (n_views, B, d)
        k_c = F.normalize(k[..., content_indices], dim=-1)  # (n_views, B, d)
        queue_c = F.normalize(queue[content_indices, :], dim=0)  # (d, Q)

    for i in range(n_view):
        for j in range(n_view):
            if i >= j:
                continue
            # q[i] → positive k[j], negatives from queue
            pos_ij = (q_c[i] * k_c[j]).sum(dim=-1, keepdim=True)  # (B, 1)
            neg_ij = q_c[i] @ queue_c  # (B, Q)
            logits_ij = torch.cat([pos_ij, neg_ij], dim=1) / tau  # (B, Q+1)
            targets = torch.zeros(logits_ij.shape[0], dtype=torch.long, device=q.device)
            total_loss = total_loss + F.cross_entropy(logits_ij, targets)

            # Symmetric: q[j] → positive k[i]
            pos_ji = (q_c[j] * k_c[i]).sum(dim=-1, keepdim=True)  # (B, 1)
            neg_ji = q_c[j] @ queue_c  # (B, Q)
            logits_ji = torch.cat([pos_ji, neg_ji], dim=1) / tau  # (B, Q+1)
            total_loss = total_loss + F.cross_entropy(logits_ji, targets)

    return total_loss


def moco_loss(q, k, queue, sim_metric, tau=1.0, estimated_content_indices=None, subsets=None, soft_content_mask=None):
    """
    Top-level MoCo loss that mirrors the ``infonce_loss`` signature.

    Iterates over all (subset, content_indices) pairs and sums the per-subset
    ``moco_infonce_loss`` values.

    Args:
        q  (torch.Tensor): Online embeddings,    shape (n_views, B, C).
        k  (torch.Tensor): Momentum embeddings,  shape (n_views, B, C).
        queue (torch.Tensor): Negative queue,    shape (C, queue_size).
        sim_metric: Unused — kept for API compatibility with ``infonce_loss``.
        tau (float): Temperature.
        estimated_content_indices (list[list[int]]): Content channel indices per subset.
        subsets (list[tuple]): View subsets (same length as estimated_content_indices).
        soft_content_mask: Optional differentiable mask from Gumbel straight-through.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    if estimated_content_indices is None or subsets is None:
        # Fall back to using all channels
        return moco_infonce_loss(q, k, queue, list(range(q.shape[-1])), tau)

    total_loss = torch.zeros(1, device=q.device, dtype=q.dtype)
    for content_indices, subset in zip(estimated_content_indices, subsets):
        q_sub = q[list(subset), ...]
        k_sub = k[list(subset), ...]
        total_loss = total_loss + moco_infonce_loss(
            q_sub,
            k_sub,
            queue,
            content_indices,
            tau,
            soft_content_mask=soft_content_mask,
        )
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
        self.n_slices = 8  # Reduced from 32 — 4× fewer SqueezeNet passes per orientation
        self.perceptual_function = LPIPS(net="squeeze")

        self.fft_factor = 1.0

        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    def forward(self, network_output: Dict[str, List[torch.Tensor]], y: torch.Tensor) -> torch.Tensor:
        # Unpacking elements
        x = y.float()
        y = network_output["reconstruction"][0].float()
        q_losses = network_output["quantization_losses"]

        loss = (
            self._calculate_pixel_loss(x, y)
            + self._calculate_frequency_loss(x, y)
            + self._calculate_perceptual_loss(x, y)
        )

        for idx, q_loss in enumerate(q_losses):
            q_loss = q_loss.float()

            self.summaries[TBSummaryTypes.SCALAR][f"Loss-MSE-VQ{idx}_Commitment_Cost"] = q_loss

            loss = loss + q_loss

        return loss

    def _calculate_frequency_loss(self, x, y) -> torch.Tensor:
        # Compute FFT on the full batch at once — cheaper than a per-sample loop
        # that builds a long autograd chain. Complex tensors are freed immediately
        # after mse_loss since we don't store them.
        with torch.amp.autocast("cuda", enabled=False):
            # fftn requires float32; x/y may be float16 under AMP
            x_f = (x.float() + 1.0) / 2.0
            y_f = (y.float() + 1.0) / 2.0
            loss = F.mse_loss(torch.abs(fftn(x_f, norm="ortho")), torch.abs(fftn(y_f, norm="ortho"))).to(x.dtype)

        loss = loss * self.fft_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Jukebox-Reconstruction"] = loss

        return loss

    def _calculate_pixel_loss(self, x, y) -> torch.Tensor:
        loss = F.l1_loss(x, y)
        loss = loss * self.pixel_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-MAE-Reconstruction"] = loss

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
        self.summaries[TBSummaryTypes.SCALAR][f"Loss-Perceptual_{name}-Reconstruction"] = p_loss

        loss = p_loss * 3.0 * self.perceptual_factor
        self.summaries[TBSummaryTypes.SCALAR]["Loss-Perceptual-Reconstruction"] = loss

        return loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries
