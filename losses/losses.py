"""
losses.py

All loss functions for Super-FAN hands training (Section 4.4 of the paper).

Overall loss (Eq. 5):
    L = α·L_pixel + β·L_feature + γ·L_heatmap + ζ·L_wgan

Where:
    L_pixel    — MSE between SR and HR images (Eq. 1)
    L_feature  — MSE between ResNet-50 feature maps (Eq. 2)
    L_heatmap  — MSE between predicted and GT heatmaps (Eq. 4, adapted)
    L_wgan     — Wasserstein loss + gradient penalty (Eq. 3)

Key adaptation from paper:
    L_heatmap uses FreiHAND GT heatmaps as targets instead of a frozen
    teacher FAN, since we have ground truth annotations available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ── Pixel loss ─────────────────────────────────────────────────────────────────

class PixelLoss(nn.Module):
    """MSE loss between SR and HR images (Eq. 1 of the paper)."""

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(sr, hr)


# ── Perceptual loss ────────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    Perceptual loss over ResNet-50 feature maps (Eq. 2 of the paper).

    Uses features after B1, B2, B3 blocks of ResNet-50, rather than
    VGG-19 layer 5_4 (which SRGAN uses). The network is frozen.
    Input images expected in [-1, 1]; renormalised to ImageNet stats internally.
    """

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.b1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                resnet.maxpool, resnet.layer1)
        self.b2 = nn.Sequential(resnet.layer2)
        self.b3 = nn.Sequential(resnet.layer3)
        for p in self.parameters():
            p.requires_grad = False

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0
        mean = self.MEAN.to(x.device)
        std  = self.STD.to(x.device)
        return (x - mean) / std

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_n = self._normalise(sr)
        hr_n = self._normalise(hr)
        loss = 0.0
        for block in [self.b1, self.b2, self.b3]:
            sr_n = block(sr_n)
            hr_n = block(hr_n)
            loss = loss + F.mse_loss(sr_n, hr_n.detach())
        return loss


# ── Heatmap loss ───────────────────────────────────────────────────────────────

class HeatmapLoss(nn.Module):
    """
    Heatmap MSE loss (Eq. 4 of the paper, adapted for FreiHAND).

    Paper:   L_heatmap = ||fan_sr(SR) - fan_hr(HR)||²
    Ours:    L_heatmap = ||fan_sr(SR) - GT_heatmap||²

    Includes a γ warmup schedule: linearly ramps from 0→1 over warmup_steps
    to avoid noisy early FAN gradients corrupting the SR network.

    Intermediate stack outputs are also supervised (equal weight),
    improving gradient flow through the hourglass.

    Args:
        warmup_steps: Steps to ramp from 0→1. Set 0 to disable.
    """

    def __init__(self, warmup_steps: int = 5000):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.register_buffer('step', torch.tensor(0))

    @property
    def warmup_weight(self) -> float:
        if self.warmup_steps == 0:
            return 1.0
        return min(1.0, self.step.item() / self.warmup_steps)

    def forward(
        self,
        pred_heatmaps: list,
        gt_heatmaps:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_heatmaps: list of (B, 21, H, W) per HG stack.
            gt_heatmaps:   (B, 21, 64, 64) GT heatmaps.
        """
        if self.training:
            self.step += 1

        loss = 0.0
        for hm in pred_heatmaps:
            if hm.shape[-1] != gt_heatmaps.shape[-1]:
                hm = F.interpolate(hm, size=gt_heatmaps.shape[2:],
                                   mode='bilinear', align_corners=False)
            loss = loss + F.mse_loss(torch.sigmoid(hm), gt_heatmaps)

        return (loss / len(pred_heatmaps)) * self.warmup_weight


# ── WGAN-GP loss ───────────────────────────────────────────────────────────────

class WGANLoss(nn.Module):
    """
    Wasserstein GAN loss with gradient penalty (Eq. 3 of the paper).
    Implements improved WGAN training (Gulrajani et al. 2017).

    Args:
        lambda_gp: Gradient penalty coefficient (default 10).
    """

    def __init__(self, lambda_gp: float = 10.0):
        super().__init__()
        self.lambda_gp = lambda_gp

    def generator_loss(self, fake_scores: torch.Tensor) -> torch.Tensor:
        return -fake_scores.mean()

    def discriminator_loss(
        self,
        real_scores: torch.Tensor,
        fake_scores: torch.Tensor,
    ) -> torch.Tensor:
        return fake_scores.mean() - real_scores.mean()

    def gradient_penalty(
        self,
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        B = real.size(0)
        alpha = torch.rand(B, 1, 1, 1, device=real.device)
        interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
        d_interp = discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interp,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(B, -1)
        return self.lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def full_discriminator_loss(
        self,
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        real_scores = discriminator(real)
        fake_scores = discriminator(fake.detach())
        return (self.discriminator_loss(real_scores, fake_scores)
                + self.gradient_penalty(discriminator, real, fake))


# ── Combined SR loss ───────────────────────────────────────────────────────────

class SuperFANLoss(nn.Module):
    """
    Combined loss for Super-FAN training (Eq. 5 of the paper).

    L = α·L_pixel + β·L_feature + γ·L_heatmap + ζ·L_wgan

    Args:
        alpha:           Weight for pixel loss (default 1.0).
        beta:            Weight for perceptual loss (default 0.006).
        gamma:           Weight for heatmap loss (default 0.1).
        zeta:            Weight for WGAN generator loss (default 0.01).
        hm_warmup:       Warmup steps for heatmap loss.
        use_adversarial: Include WGAN loss (False in Stage 1, True in Stage 2).
    """

    def __init__(
        self,
        alpha:           float = 1.0,
        beta:            float = 0.006,
        gamma:           float = 0.1,
        zeta:            float = 0.01,
        hm_warmup:       int   = 5000,
        use_adversarial: bool  = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.zeta  = zeta
        self.use_adversarial = use_adversarial

        self.pixel_loss      = PixelLoss()
        self.perceptual_loss = PerceptualLoss()
        self.heatmap_loss    = HeatmapLoss(warmup_steps=hm_warmup)
        self.wgan_loss       = WGANLoss()

    def forward(
        self,
        sr:            torch.Tensor,
        hr:            torch.Tensor,
        pred_heatmaps: list,
        gt_heatmaps:   torch.Tensor,
        fake_scores:   torch.Tensor = None,
    ) -> dict:
        """
        Returns dict with keys: 'total', 'pixel', 'perceptual', 'heatmap', 'wgan'
        """
        l_pixel      = self.pixel_loss(sr, hr)
        l_perceptual = self.perceptual_loss(sr, hr)
        l_heatmap    = self.heatmap_loss(pred_heatmaps, gt_heatmaps)

        total = (self.alpha * l_pixel
                 + self.beta  * l_perceptual
                 + self.gamma * l_heatmap)

        l_wgan = torch.tensor(0.0, device=sr.device)
        if self.use_adversarial and fake_scores is not None:
            l_wgan = self.wgan_loss.generator_loss(fake_scores)
            total  = total + self.zeta * l_wgan

        return {
            'total':      total,
            'pixel':      l_pixel,
            'perceptual': l_perceptual,
            'heatmap':    l_heatmap,
            'adversarial': l_wgan,
        }
