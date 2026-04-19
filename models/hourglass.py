"""
hourglass.py

2-stack Hourglass network for hand keypoint heatmap regression.

Follows the architecture used in the Super-FAN paper (Bulat & Tzimiropoulos,
ICCV 2017), adapted for 21 hand keypoints instead of 68 facial landmarks.

Architecture overview:
    Input: (B, 3, 128, 128) HR or SR image
    Per stack:
        - Encoder path: series of residual blocks with max-pooling
        - Decoder path: nearest-neighbour upsampling + residual blocks
        - Skip connections between encoder and decoder at each resolution
        - Intermediate heatmap prediction head after each stack
    Output: list of (B, 21, 64, 64) heatmap tensors, one per stack
            (only the final stack's output is used at test time)

The intermediate supervision from stack 1 is used during training to
improve gradient flow, following the original hourglass paper
(Newell et al., ECCV 2016).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.constants import NUM_KEYPOINTS, HR_SIZE


# ── Building blocks ────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Pre-activation residual block as used in the original FAN paper.
    Architecture: BN → ReLU → Conv(1x1) → BN → ReLU → Conv(3x3) → BN → ReLU → Conv(1x1)
    with a learned projection shortcut if in_channels != out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = out_channels // 2

        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)

        self.bn2   = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False)

        self.bn3   = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

        # Projection shortcut (used when channel dims differ)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        out = F.relu(self.bn3(out), inplace=True)
        out = self.conv3(out)

        if self.shortcut is not None:
            residual = self.shortcut(x)

        return out + residual


class HourglassModule(nn.Module):
    """
    Single hourglass module with recursive structure.

    Args:
        depth:    Number of pooling levels (4 is standard, giving 64→4 spatial).
        channels: Number of feature channels throughout.
    """

    def __init__(self, depth: int, channels: int):
        super().__init__()
        self.depth = depth

        # Encoder (upper branch before pooling)
        self.encoder_res = ResidualBlock(channels, channels)

        # Recursive inner hourglass or bottom residual
        if depth > 1:
            self.inner = HourglassModule(depth - 1, channels)
        else:
            self.inner = ResidualBlock(channels, channels)

        # Lower branch (runs in parallel with inner hourglass)
        self.lower_res = ResidualBlock(channels, channels)

        # Decoder (after upsampling)
        self.decoder_res = ResidualBlock(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upper branch: residual at current resolution
        upper = self.encoder_res(x)

        # Lower branch: downsample → inner hourglass → upsample
        lower = F.max_pool2d(x, kernel_size=2, stride=2)
        lower = self.lower_res(lower)
        lower = self.inner(lower)
        lower = self.decoder_res(lower)
        lower = F.interpolate(lower, scale_factor=2, mode='nearest')

        return upper + lower


# ── Heatmap prediction head ────────────────────────────────────────────────────

class HeatmapHead(nn.Module):
    """
    Lightweight head that maps hourglass features to per-keypoint heatmaps.
    Used after each stack for intermediate supervision.
    """

    def __init__(self, in_channels: int, num_keypoints: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.bn    = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, num_keypoints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv1(x)), inplace=True)
        return self.conv2(x)   # raw logits; sigmoid applied in loss


# ── Full stacked hourglass (Hand-FAN) ─────────────────────────────────────────

class StackedHourglass(nn.Module):
    """
    2-stack Hourglass network for hand keypoint heatmap regression.

    This is the Hand-FAN: it replaces the pretrained facial FAN from the
    original Super-FAN paper.  It is trained in two stages:

        Stage 1 (standalone): trained directly on HR images with GT heatmaps
                              from FreiHAND's 3D annotations, giving it the
                              warm start that the paper got from pretraining.

        Stage 2 (joint):      fine-tuned jointly with the SR generator,
                              receiving SR output images as input and
                              contributing the heatmap loss to the SR network.

    Args:
        num_stacks:    Number of hourglass stacks (2 as in the paper).
        num_keypoints: Number of output heatmap channels (21 for FreiHAND).
        channels:      Internal feature dimension (256 is standard).
        depth:         Hourglass recursion depth (4 gives 64→4 spatial).
    """

    def __init__(
        self,
        num_stacks:    int = 2,
        num_keypoints: int = NUM_KEYPOINTS,
        channels:      int = 256,
        depth:         int = 4,
    ):
        super().__init__()
        self.num_stacks = num_stacks

        # ── Stem: bring 3-channel input to `channels` feature maps ───────────
        # Downsample 128→16 (3 stride-2 ops) so hourglass depth=4 gives
        # 16→8→4→2→1 encoder path — same working resolution as before.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 128→64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, stride=2),                                          # 64→32
            ResidualBlock(128, 128),
            nn.MaxPool2d(2, stride=2),                                          # 32→16
            ResidualBlock(128, channels),
        )
        # After stem: (B, channels, 16, 16)
        # Hourglass depth=4 then gives: 16→8→4→2→1 encoder, mirror decoder

        # ── Stacked hourglass modules ─────────────────────────────────────────
        self.hourglasses = nn.ModuleList(
            [HourglassModule(depth, channels) for _ in range(num_stacks)]
        )

        # ── Per-stack residual and 1x1 feature refinement ─────────────────────
        self.stack_res   = nn.ModuleList(
            [ResidualBlock(channels, channels) for _ in range(num_stacks)]
        )
        self.stack_conv  = nn.ModuleList(
            [nn.Conv2d(channels, channels, 1) for _ in range(num_stacks)]
        )

        # ── Heatmap prediction heads (one per stack) ──────────────────────────
        self.heads = nn.ModuleList(
            [HeatmapHead(channels, num_keypoints) for _ in range(num_stacks)]
        )

        # ── Remap intermediate heatmaps back to feature space for next stack ──
        # Used between stacks to pass landmark information forward
        self.hm_to_feat = nn.ModuleList(
            [nn.Conv2d(num_keypoints, channels, 1)
             for _ in range(num_stacks - 1)]
        )
        self.feat_to_feat = nn.ModuleList(
            [nn.Conv2d(channels, channels, 1)
             for _ in range(num_stacks - 1)]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: (B, 3, H, W) image tensor, normalised to [-1, 1]

        Returns:
            heatmaps: list of (B, num_keypoints, H//8, W//8) tensors,
                      one per stack.  Length == num_stacks.
                      The LAST element is the final prediction.
                      (H//8 because stem does 3× stride-2 downsampling)
        """
        features = self.stem(x)   # (B, channels, H//8, W//8)
        all_heatmaps = []

        for i in range(self.num_stacks):
            # Run hourglass
            hg_out = self.hourglasses[i](features)
            hg_out = self.stack_res[i](hg_out)
            hg_out = F.relu(self.stack_conv[i](hg_out), inplace=True)

            # Predict heatmaps for this stack
            hm = self.heads[i](hg_out)
            all_heatmaps.append(hm)

            # Pass information to next stack (skip for last stack)
            if i < self.num_stacks - 1:
                features = (features
                            + self.feat_to_feat[i](hg_out)
                            + self.hm_to_feat[i](hm))

        return all_heatmaps   # list of (B, 21, 16, 16) tensors

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method returning only the final stack's heatmaps,
        upsampled to match the input resolution.

        Args:
            x: (B, 3, H, W)

        Returns:
            heatmaps: (B, num_keypoints, H, W) — upsampled to input size
        """
        heatmaps = self.forward(x)[-1]   # final stack only
        return F.interpolate(heatmaps, size=x.shape[2:],
                             mode='bilinear', align_corners=False)
