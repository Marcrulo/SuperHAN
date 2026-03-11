"""
discriminator.py

WGAN-GP discriminator for face/hand super-resolution.

Follows the paper (Section 4.2):
    - DCGAN architecture
    - NO batch normalisation (required for WGAN-GP gradient penalty)
    - Takes 128x128 RGB image, outputs scalar critic score
    - Used with Wasserstein loss + gradient penalty (Gulrajani et al. 2017)

At test time the discriminator is not used (paper Section 4).
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    DCGAN-style discriminator without batch normalisation.

    Progressively downsamples 64x64 → 1x1 via strided convolutions,
    outputting a scalar critic value per image (not a probability).

    Args:
        in_channels:   Input image channels (3 for RGB).
        base_channels: Feature channels at first conv layer (64).
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        def conv_block(in_ch, out_ch, stride=2):
            # No BN — required for WGAN-GP (BN breaks the gradient penalty)
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
            )

        c = base_channels
        self.net = nn.Sequential(
            # 128x128 → 64x64
            conv_block(in_channels, c,     stride=2),
            # 64x64  → 32x32
            conv_block(c,           c * 2, stride=2),
            # 32x32  → 16x16
            conv_block(c * 2,       c * 4, stride=2),
            # 16x16  → 8x8
            conv_block(c * 4,       c * 8, stride=2),
            # 8x8    → 4x4
            conv_block(c * 8,       c * 8, stride=2),
            # 4x4    → 1x1
            nn.Conv2d(c * 8, 1, kernel_size=4, stride=1, padding=0, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 128, 128) image in [-1, 1]

        Returns:
            score: (B,) critic scores (unbounded; NOT a probability)
        """
        return self.net(x).view(-1)
