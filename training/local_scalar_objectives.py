"""Training-only statistics, frozen anatomy predictor and label-free objectives."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.local_scalar_readout import bands, barlow_twins, infonce, reconstruction_loss, within_view_decorrelation
from training.local_scalar_data import UNLABELLED


def tensors(bank, ids, device, keys=None):
    return {k: torch.from_numpy(np.array(bank[k][ids], dtype=np.float32)).to(device) for k in (keys or UNLABELLED)}


class ResidualTarget(nn.Module):
    """Frozen quadratic ridge prediction of signed feature maps from global code.

    Fitted on training images only, separately per view and output location.
    The random channel projection is fixed and saved; it contains no factor template.
    """

    @classmethod
    def from_state_dict(cls, state):
        """Restore saved training statistics without accessing the training bank."""
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        for name, value in state.items():
            model.register_buffer(name, value.detach().clone())
        model.channels = state["projection"].shape[0]
        model.grid_size = round((state["beta"].shape[-1] / model.channels) ** (1 / 3))
        model.requires_grad_(False)
        return model.eval()

    def __init__(self, train, channels=64, ridge=10.0, seed=42, batch_size=16):
        super().__init__()
        n, views, source_channels, grid, _, _ = train["features"].shape
        channels = min(channels, source_channels)
        total, squares, count = 0, 0, 0
        for start in range(0, n, batch_size):
            h = torch.from_numpy(np.array(train["features"][start : start + batch_size])).double()
            total = total + h.sum((0, 3, 4, 5))
            squares = squares + h.square().sum((0, 3, 4, 5))
            count += len(h) * grid**3
        mean = total / count
        sd = (squares / count - mean.square()).clamp_min(1e-8).sqrt()
        self.register_buffer("feature_mean", mean.float()[None, :, :, None, None, None])
        self.register_buffer("feature_sd", sd.float()[None, :, :, None, None, None])
        generator = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(source_channels, channels, generator=generator))
        self.register_buffer("projection", q.T.contiguous())
        g = torch.from_numpy(np.array(train["global"])).double()
        self.register_buffer("global_mean", g.mean(0).float()[None])
        self.register_buffer("global_sd", g.std(0, unbiased=False).clamp_min(1e-4).float()[None])
        self.register_buffer("quadratic_indices", torch.triu_indices(g.shape[-1], g.shape[-1]))
        design = self.design(g.float()).double()
        width = design.shape[-1]
        cross = torch.zeros(views, width, channels * grid**3, dtype=torch.float64)
        gram = torch.einsum("bvi,bvj->vij", design, design)
        for start in range(0, n, batch_size):
            h = torch.from_numpy(np.array(train["features"][start : start + batch_size]))
            y = self.target(h).flatten(2).double()
            cross += torch.einsum("bvi,bvj->vij", design[start : start + len(h)], y)
        penalty = torch.eye(width, dtype=torch.float64) * ridge
        penalty[0, 0] = 1e-8  # Leave the intercept effectively unpenalized.
        beta = torch.linalg.solve(gram + penalty, cross).float()
        self.register_buffer("beta", beta)
        self.grid_size, self.channels = grid, channels
        sums, counts = {}, {}
        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = tensors(
                    train,
                    slice(start, start + batch_size),
                    "cpu",
                    ("features", "global", "support"),
                )
                for name, value in bands(self.residual(batch)).items():
                    mask = F.adaptive_avg_pool3d(batch["support"].flatten(0, 1), value.shape[-3:]).reshape(
                        len(value), views, 1, *value.shape[-3:]
                    )
                    sums[name] = sums.get(name, 0) + (value.square() * mask).sum((0, 3, 4, 5))
                    counts[name] = counts.get(name, 0) + mask.sum((0, 3, 4, 5))
        for name in sums:
            scale = (sums[name] / counts[name].clamp_min(1)).sqrt().clamp_min(0.05)
            self.register_buffer(f"scale_{name}", scale[None, :, :, None, None, None])
        self.requires_grad_(False)
        self.eval()

    def normalize(self, h):
        return (h - self.feature_mean) / self.feature_sd

    def design(self, g):
        g = ((g - self.global_mean) / self.global_sd).clamp(-5, 5)
        quadratic = g[..., self.quadratic_indices[0]] * g[..., self.quadratic_indices[1]]
        return torch.cat((torch.ones_like(g[..., :1]), g, quadratic), -1)

    def target(self, h):
        return torch.einsum("bvcijk,tc->bvtijk", self.normalize(h), self.projection)

    def residual(self, batch):
        target = self.target(batch["features"])
        predicted = torch.einsum("bvi,vij->bvj", self.design(batch["global"]), self.beta).reshape_as(target)
        return target - predicted

    def scales(self):
        return {k: getattr(self, f"scale_{k}") for k in ("fine", "highpass", "coarse")}


def read(head, batch, normalizer, prefix=""):
    return head(normalizer.normalize(batch[prefix + "features"]), batch[prefix + "support"])


def unsupervised_objective(head, decoder, normalizer, batch, arm, args):
    # Reject accidental truth/intervention leakage at the optimizer boundary.
    if set(batch) != set(UNLABELLED):
        raise ValueError("Unsupervised objective accepts only the unlabelled bank")
    original, photo = read(head, batch, normalizer), read(head, batch, normalizer, "photo_")
    if arm == "barlow":
        loss = barlow_twins(head, original["code"], photo["code"], args.barlow_lambda)
        return loss, {"barlow": loss}
    if arm in ("infonce", "decorrelated"):
        terms = {"infonce": infonce(head, original["code"], photo["code"], args.temperature)}
        loss = terms["infonce"]
        if arm == "decorrelated":
            terms["decorrelation"] = within_view_decorrelation(original["code"], batch["global"])
            loss = loss + args.decorrelation_weight * terms["decorrelation"]
        return loss, terms
    if arm != "residual":
        raise ValueError(f"Unknown unsupervised arm: {arm}")
    residual = normalizer.residual(batch).detach()
    losses, terms = [], {}
    for name, code in (("original", original), ("photo", photo)):
        prediction = decoder(code["physical"], code["amplitude"])
        value, components, _ = reconstruction_loss(prediction, residual, batch["support"], normalizer.scales())
        losses.append(value)
        terms.update({f"{name}_{k}": v for k, v in components.items()})
    flipped = read(head, batch, normalizer, "flip_")
    terms["equivariance"] = F.mse_loss(flipped["physical"], original["physical"] * batch["signs"][:, None, None])
    terms["photometric"] = F.mse_loss(photo["code"], original["code"])
    floor = torch.full_like(original["code"][0], 0.02)
    floor[..., -1] = 0.1
    terms["variance"] = (floor - original["code"].var(0, unbiased=False).add(1e-6).sqrt()).clamp_min(0).square().mean()
    loss = (
        sum(losses) / 2
        + args.equivariance_weight * terms["equivariance"]
        + args.photometric_weight * terms["photometric"]
        + args.variance_weight * terms["variance"]
    )
    return loss, terms


def oracle_terms(head, normalizer, obs, pairs, scale):
    original = read(head, obs, normalizer)
    n, endpoints = pairs["features"].shape[:2]
    batch = {k: a.flatten(0, 1) for k, a in pairs.items() if k in ("features", "support")}
    endpoint = read(head, batch, normalizer)

    def four(code):
        return torch.cat(
            (
                code["physical"],
                code["amplitude"][..., None].expand(*code["physical"].shape[:-1], 1),
            ),
            -1,
        )

    predicted = four(original)
    actual = obs["truth"][:, None, None, [2, 3, 4, 8]]
    pair_predicted = four(endpoint).reshape(n, endpoints, *predicted.shape[1:])
    pair_actual = pairs["truth"][:, :, None, None, [2, 3, 4, 8]]
    return {
        "level": ((predicted - actual) / scale).square().mean(),
        "endpoint": ((pair_predicted - pair_actual) / scale).square().mean(),
        "response": (((pair_predicted[:, 1] - pair_predicted[:, 0]) - (pair_actual[:, 1] - pair_actual[:, 0])) / scale)
        .square()
        .mean(),
    }
