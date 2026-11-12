"""
DriverPoseNet7KP — Lightweight seven-keypoint pose estimation CNN.

Input : [B, 3, 256, 192]
Output: [B, 7, 64, 48]  (7 heatmaps, one per keypoint)

Architecture: residual CNN stem + downsampling residual blocks +
              upsampling path + 1x1 heatmap head.
Designed to be FPGA-friendly (INT8, stride-2 downsampling, 1x1 projections).
"""

from typing import Tuple

import torch
import torch.nn as nn

from .keypoints import NUM_KEYPOINTS


class ResidualBlock(nn.Module):
    """Basic residual block with optional 1x1 projection for channel/stride change."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class DriverPoseNet7KP(nn.Module):
    def __init__(self, num_keypoints: int = NUM_KEYPOINTS):
        super().__init__()
        if num_keypoints != NUM_KEYPOINTS:
            raise ValueError(
                f"DriverPoseNet7KP requires {NUM_KEYPOINTS} keypoints, got {num_keypoints}"
            )

        # Stem: 256x192 -> 64x48
        self.stem_conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(32)
        self.stem_relu = nn.ReLU(inplace=True)
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Downsampling residual path: 64x48 -> 8x6
        self.layer1 = ResidualBlock(32, 64, stride=2)   # 32x24
        self.layer2 = ResidualBlock(64, 128, stride=2)  # 16x12
        self.layer3 = ResidualBlock(128, 256, stride=2) # 8x6

        # Upsampling path: 8x6 -> 64x48
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )  # 16x12
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # 32x24
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # 64x48

        # Heatmap head: 64x48 -> 7x64x48
        self.head_conv = nn.Conv2d(32, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        x = self.stem_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        x = self.head_conv(x)
        return x

    def output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Return expected output shape for a given input shape (without batch)."""
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            return tuple(self.forward(dummy).shape[1:])


__all__ = ["DriverPoseNet7KP", "ResidualBlock"]
