import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator3D(nn.Module):
    """Lightweight 3D PatchGAN discriminator with spectral normalisation.

    Five strided Conv3d layers reduce a (1, D, H, W) volume to a small
    spatial grid of real/fake logits (one scalar per receptive-field patch).
    Spectral norm on every layer except the final one stabilises training
    without batch statistics (safe for small medical-image batch sizes).
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()

        def _block(in_ch, out_ch, stride):
            return nn.Sequential(
                spectral_norm(nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = nn.Sequential(
            _block(in_channels, base_channels, stride=2),  # /2
            _block(base_channels, base_channels * 2, stride=2),  # /4
            _block(base_channels * 2, base_channels * 4, stride=2),  # /8
            _block(base_channels * 4, base_channels * 8, stride=2),  # /16
            # Final layer: no spectral norm, produces patch logits
            nn.Conv3d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)
