"""
generator.py

Super-resolution generator with the 12-3-2 residual block distribution
proposed in the Super-FAN paper (Section 4.1).

Architecture:
    Input:  (B, 3, 32, 32)   LR image in [-1, 1]
    Output: (B, 3, 128, 128) SR image in [-1, 1]

Block distribution (N1-N2-N3 = 12-3-2):
    12 residual blocks at 32x32  (input resolution)
     3 residual blocks at 64x64  (after first deconv upsample)
     2 residual blocks at 128x128 (after second deconv upsample)

Each residual block: Conv(3x3) → BN → ReLU → Conv(3x3) → BN + skip.
ReLU used throughout (paper found no improvement from PReLU).
No "long" skip connection over the full 12-block group (paper found
only marginal gains; omitted for simplicity).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building block ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Standard residual block as used in the Super-FAN SR network.
    Conv(3x3) → BN → ReLU → Conv(3x3) → BN, with identity skip.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ── Upsample block ─────────────────────────────────────────────────────────────

class UpsampleBlock(nn.Module):
    """
    Deconvolution-based 2x upsample followed by BN + ReLU.
    Uses ConvTranspose2d (deconv) as in the paper rather than
    pixel shuffle (which is used in SR-ResNet [19]).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Generator ──────────────────────────────────────────────────────────────────

class SRGenerator(nn.Module):
    """
    Super-resolution generator with 12-3-2 block distribution.

    Args:
        in_channels:  Input image channels (3 for RGB).
        out_channels: Output image channels (3 for RGB).
        base_channels: Feature channels throughout the network (64).
        n1, n2, n3:   Residual blocks at each of the 3 resolutions.
    """

    def __init__(
        self,
        in_channels:   int = 3,
        out_channels:  int = 3,
        base_channels: int = 64,
        n1: int = 12,
        n2: int = 3,
        n3: int = 2,
    ):
        super().__init__()

        # ── Entry conv: 3 → base_channels at 16x16 ───────────────────────────
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # ── Stage 1: N1=12 residual blocks at 16x16 ──────────────────────────
        self.stage1 = nn.Sequential(*[ResBlock(base_channels) for _ in range(n1)])

        # ── Upsample 16x16 → 32x32 ───────────────────────────────────────────
        self.up1 = UpsampleBlock(base_channels, base_channels)

        # ── Stage 2: N2=3 residual blocks at 32x32 ───────────────────────────
        self.stage2 = nn.Sequential(*[ResBlock(base_channels) for _ in range(n2)])

        # ── Upsample 32x32 → 64x64 ───────────────────────────────────────────
        self.up2 = UpsampleBlock(base_channels, base_channels)

        # ── Stage 3: N3=2 residual blocks at 64x64 ───────────────────────────
        self.stage3 = nn.Sequential(*[ResBlock(base_channels) for _ in range(n3)])

        # ── Exit conv: base_channels → 3, tanh to keep output in [-1, 1] ─────
        self.exit = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 16, 16) LR image in [-1, 1]

        Returns:
            sr: (B, 3, 64, 64) SR image in [-1, 1]
        """
        x = self.entry(x)     # (B, 64, 16, 16)
        x = self.stage1(x)    # (B, 64, 16, 16)
        x = self.up1(x)       # (B, 64, 32, 32)
        x = self.stage2(x)    # (B, 64, 32, 32)
        x = self.up2(x)       # (B, 64, 64, 64)
        x = self.stage3(x)    # (B, 64, 64, 64)
        return self.exit(x)   # (B,  3, 64, 64)
