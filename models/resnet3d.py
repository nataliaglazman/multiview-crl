"""Volumetric counterpart of the torchvision ResNet-18 image backbone.

Uses the same stem, BasicBlock layout, widths, strides, BatchNorm defaults and
initialization, replacing 2D operators with 3D operators. GAP and the upstream
Linear -> LeakyReLU -> Linear readout live in MultiviewConvEncoder.
Reference: https://github.com/pytorch/vision/blob/v0.15.2/torchvision/models/resnet.py
"""

from torch import nn


class BasicBlock3d(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.downsample = None
        if stride != 1 or in_channels != channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(channels),
            )

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.relu(h + residual)


class ResNet18Features3d(nn.Module):
    """ResNet-18 spatial features: 512 channels at approximately 1/32 resolution."""

    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.layer1 = self._stage(64, 64, stride=1)
        self.layer2 = self._stage(64, 128, stride=2)
        self.layer3 = self._stage(128, 256, stride=2)
        self.layer4 = self._stage(256, 512, stride=2)
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _stage(in_channels, channels, stride):
        return nn.Sequential(BasicBlock3d(in_channels, channels, stride), BasicBlock3d(channels, channels))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)
