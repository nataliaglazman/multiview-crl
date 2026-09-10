"""Fixed content/style channels, shared by DINO fine-tuning and NPZ evaluation."""

import math


def make_partition(feature_dim, hidden_size, cli):
    """Keep each channel's role across spatial positions, CLS/mean blocks and slices."""
    fraction = cli.style_fraction
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("style_fraction must be in [0, 1)")
    style = int(hidden_size * fraction)
    if fraction > 0 and style == 0:
        raise ValueError("The requested style fraction rounds to zero channels")
    spatial = cli.grid_size ** (3 if cli.backbone == "3dino" else 2) if cli.token_pool == "grid" else 1
    repeats = 2 if cli.token_pool == "cls_mean" else 1
    if cli.backbone != "3dino" and cli.slice_agg == "concat":
        repeats *= len(cli.axes) * cli.slices
    if feature_dim != repeats * hidden_size * spatial:
        raise ValueError("Embedding shape does not match its saved pooling/channel layout")
    return dict(
        version=1,
        scheme="fixed_backbone_channels",
        feature_dim=feature_dim,
        hidden_size=hidden_size,
        content_channels=hidden_size - style,
        style_channels=style,
        content_dim=repeats * (hidden_size - style) * spatial,
        style_dim=repeats * style * spatial,
        spatial_positions=spatial,
        repeats=repeats,
        requested_style_fraction=fraction,
        effective_style_fraction=style / hidden_size,
    )


def split_embeddings(features, partition):
    """Works with NumPy arrays and Torch tensors; retains Torch autograd."""
    if partition.get("version") != 1 or partition.get("scheme") != "fixed_backbone_channels":
        raise ValueError("Unsupported embedding partition")
    c, s = partition["content_channels"], partition["style_channels"]
    repeats, spatial = partition["repeats"], partition["spatial_positions"]
    if c < 1 or s < 0 or repeats < 1 or spatial < 1 or c + s != partition["hidden_size"]:
        raise ValueError("Invalid content/style partition dimensions")
    width = repeats * (c + s) * spatial
    if (partition["feature_dim"], partition["content_dim"], partition["style_dim"]) != (
        width,
        repeats * c * spatial,
        repeats * s * spatial,
    ):
        raise ValueError("Inconsistent embedding partition widths")
    if features.ndim != 2 or features.shape[1] != width:
        raise ValueError("Embedding width differs from the saved content/style partition")
    blocks = features.reshape(len(features), repeats, c + s, spatial)
    return (
        blocks[:, :, :c, :].reshape(len(features), partition["content_dim"]),
        blocks[:, :, c:, :].reshape(len(features), partition["style_dim"]),
    )
