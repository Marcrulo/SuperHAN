"""
inference.py

Minimal inference interface for SuperHAN models.

Usage:
    from inference import SuperResolution, SuperHAN
    from PIL import Image

    img = Image.open("hand_crop.jpg")

    sr = SuperResolution("checkpoints/sr/best.pt")
    sr_img = sr(img)                        # PIL.Image 128×128

    model = SuperHAN("checkpoints/superfan/best.pt")
    sr_img, keypoints = model(img)          # PIL.Image 128×128, ndarray (21, 2)
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.generator import SRGenerator
from models.hourglass import StackedHourglass
from data.constants import LR_SIZE, HEATMAP_SIZE, HR_SIZE


def _to_tensor(image) -> torch.Tensor:
    """PIL or HWC uint8/float ndarray → (1, 3, LR_SIZE, LR_SIZE) in [-1, 1]."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8) if image.dtype != np.uint8 else image)
    image = image.convert("RGB").resize((LR_SIZE, LR_SIZE), Image.BICUBIC)
    t = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0  # (3, 32, 32)
    t = t * 2.0 - 1.0
    return t.unsqueeze(0)  # (1, 3, 32, 32)


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    """(1, 3, H, W) in [-1, 1] → PIL.Image uint8."""
    t = (tensor.squeeze(0).clamp(-1, 1) + 1.0) / 2.0  # [0, 1]
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _keypoints_from_heatmaps(heatmaps: torch.Tensor, scale: float) -> np.ndarray:
    """(1, 21, H, W) logits → (21, 2) float32 pixel coords scaled to SR space."""
    B, K, H, W = heatmaps.shape
    hm_flat = torch.sigmoid(heatmaps).view(B, K, -1)
    idx = hm_flat.argmax(dim=-1)  # (1, 21)
    y = (idx // W).float()
    x = (idx %  W).float()
    coords = torch.stack([x, y], dim=-1).squeeze(0)  # (21, 2)
    return (coords.cpu().numpy() * scale).astype(np.float32)


class SuperResolution:
    """SR-only inference: pre-cropped hand image → 128×128 super-resolved image."""

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.generator = SRGenerator().to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.generator.load_state_dict(ckpt["generator_state"])
        self.generator.eval()

    def __call__(self, image) -> Image.Image:
        """
        Args:
            image: PIL.Image or np.ndarray (H, W, 3) RGB, any size.
        Returns:
            128×128 PIL.Image, RGB.
        """
        x = _to_tensor(image).to(self.device)
        with torch.inference_mode():
            sr = self.generator(x)
        return _to_pil(sr)


class SuperHAN:
    """Full SuperHAN inference: pre-cropped hand image → SR image + hand keypoints."""

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.generator = SRGenerator().to(self.device)
        self.fan = StackedHourglass().to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.generator.load_state_dict(ckpt["generator_state"])
        self.fan.load_state_dict(ckpt["fan_state"])
        self.generator.eval()
        self.fan.eval()
        self._kpt_scale = HR_SIZE / HEATMAP_SIZE  # 128 / 16 = 8

    def __call__(self, image) -> tuple:
        """
        Args:
            image: PIL.Image or np.ndarray (H, W, 3) RGB, any size.
        Returns:
            sr_image:  128×128 PIL.Image, RGB.
            keypoints: np.ndarray (21, 2) float32, (x, y) coords in SR image space (0–127).
        """
        x = _to_tensor(image).to(self.device)
        with torch.inference_mode():
            sr = self.generator(x)
            heatmaps_list = self.fan(sr)
        sr_image = _to_pil(sr)
        keypoints = _keypoints_from_heatmaps(heatmaps_list[-1], self._kpt_scale)
        return sr_image, keypoints
