"""Which input voxels each cell of the conv backbone reads, without torch.

``training.encoder_args`` imports this to check a run on a host with no torch, and
``models.multiview_encoder`` uses it for ``--interior-only``. ``tests/test_encoder_pointwise.py`` holds it to the
built ``models.vqvae.Encoder`` by injecting a different padding value and seeing which cells move.
"""


def conv_footprint(downscale_factor, nb_res_layers):
    """(stride, low, high): map cell j reads input voxels stride * j + low to stride * j + high along each axis.

    Each layer's output i reads step * i - padding to step * i - padding + kernel - 1 of the layer before, composed
    down models.vqvae.Encoder: a 4^3 stride-2 conv per halving, one 3^3 conv, two 3^3 convs per residual block. A
    residual block's weight alpha starts at 0, so at step 0 its convs reach nothing, but the first step moves alpha.
    """
    steps = 0
    while 2**steps < downscale_factor:
        steps += 1
    stride, low, high = 1, 0, 0
    for kernel, step, padding in [(4, 2, 1)] * steps + [(3, 1, 1)] * (1 + 2 * nb_res_layers):
        low -= stride * padding
        high += stride * (kernel - 1 - padding)
        stride *= step
    return stride, low, high


def receptive_field_voxels(downscale_factor, nb_res_layers):
    """Width in voxels of a cell's receptive field along one axis."""
    _, low, high = conv_footprint(downscale_factor, nb_res_layers)
    return high - low + 1


def interior_bins(size, bins, downscale_factor, nb_res_layers):
    """Bins of a patch-grid axis whose cells all read inside an axis of ``size`` voxels, so none reads padding.

    The backbone map has size // stride cells along the axis. ``adaptive_avg_pool3d`` gives bin i the cells
    floor(i n / g) to ceil((i + 1) n / g) - 1, so a coarser bin is kept only if every cell it averages is.
    """
    stride, low, high = conv_footprint(downscale_factor, nb_res_layers)
    cells = size // stride
    first, last = -(low // stride), (size - 1 - high) // stride
    return [i for i in range(bins) if (i * cells) // bins >= first and -(-(i + 1) * cells // bins) - 1 <= last]
