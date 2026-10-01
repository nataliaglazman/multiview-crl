"""Volumetric counterpart of the torchvision ResNet-18 image backbone.

Uses the same stem, BasicBlock layout, widths, strides, BatchNorm defaults and
initialization, replacing 2D operators with 3D operators. GAP and the upstream
Linear -> LeakyReLU -> Linear readout live in MultiviewConvEncoder.
Reference: https://github.com/pytorch/vision/blob/v0.15.2/torchvision/models/resnet.py
"""

from torch import nn


def normalization(channels, kind):
    if kind == "batch":
        return nn.BatchNorm3d(channels)
    if kind == "group":
        return nn.GroupNorm(32, channels)
    raise ValueError(f"Unknown ResNet normalization: {kind!r}")


class BasicBlock3d(nn.Module):
    def __init__(self, in_channels, channels, stride=1, norm="batch"):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = normalization(channels, norm)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = normalization(channels, norm)
        self.downsample = None
        if stride != 1 or in_channels != channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, channels, 1, stride=stride, bias=False),
                normalization(channels, norm),
            )

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.relu(h + residual)


class ResNet18Features3d(nn.Module):
    """512-channel features; optional GN and reduced late-stage downsampling.

    Stride 16 removes layer4 downsampling; stride 8 removes layer3 and layer4
    downsampling. The original stem stays intact, and no dilation is introduced.
    """

    def __init__(self, in_channels=1, norm="batch", output_stride=32):
        super().__init__()
        if output_stride not in (8, 16, 32):
            raise ValueError("ResNet output_stride must be 8, 16 or 32")
        self.output_stride = output_stride
        self.norm = norm
        self.conv1 = nn.Conv3d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = normalization(64, norm)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.layer1 = self._stage(64, 64, stride=1, norm=norm)
        self.layer2 = self._stage(64, 128, stride=2, norm=norm)
        self.layer3 = self._stage(128, 256, stride=2 if output_stride >= 16 else 1, norm=norm)
        self.layer4 = self._stage(256, 512, stride=2 if output_stride == 32 else 1, norm=norm)
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _stage(in_channels, channels, stride, norm="batch"):
        return nn.Sequential(
            BasicBlock3d(in_channels, channels, stride, norm), BasicBlock3d(channels, channels, norm=norm)
        )

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)
