"""
augmentations.py

Geometric and colour augmentations applied consistently to both the
hand crop image and its 2D keypoints, matching the augmentation strategy
described in Section 4.5 of the Super-FAN paper:
    - Random horizontal flip
    - Random scale  (0.85 – 1.15)
    - Random rotation (-30° – +30°)
    - Colour / brightness / contrast jitter

All transforms keep keypoints in sync with the image so that GT heatmaps
remain valid after augmentation.
"""

import random
import math
import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF

from data.constants import HR_SIZE


class HandAugmentor:
    """
    Applies a randomised augmentation pipeline to a (PIL image, uv, visibility)
    triplet.  All spatial transforms are applied to both the image and the 2D
    keypoints so they stay aligned.

    Args:
        flip_prob:      Probability of horizontal flip.
        scale_range:    (min, max) multiplicative scale factor.
        rotation_range: (min_deg, max_deg) rotation range.
        colour_prob:    Probability of applying each colour jitter transform.
    """

    # FreiHAND horizontal-flip keypoint permutation.
    # When we flip the image left-right the hand chirality swaps, so we must
    # also permute the keypoint indices to match the new anatomy.
    # FreiHAND doesn't contain mirrored pairs, so we disable flip by default
    # to avoid introducing anatomically incorrect samples.  Set flip_prob > 0
    # only if you explicitly want to include flipped hands as data augmentation
    # and understand the chirality implications.
    FLIP_PAIRS = []  # populated below if flip is enabled

    def __init__(
        self,
        flip_prob: float = 0.0,
        scale_range: tuple = (0.85, 1.15),
        rotation_range: tuple = (-30, 30),
        colour_prob: float = 0.5,
    ):
        self.flip_prob      = flip_prob
        self.scale_range    = scale_range
        self.rotation_range = rotation_range
        self.colour_prob    = colour_prob

    def __call__(
        self,
        image: Image.Image,
        uv: np.ndarray,
        visibility: np.ndarray,
    ) -> tuple:
        """
        Args:
            image:      PIL Image (hand crop, arbitrary size before HR resize)
            uv:         (21, 2) 2D keypoints in [0, HR_SIZE) space
            visibility: (21,)  boolean mask

        Returns:
            (augmented_image, augmented_uv, augmented_visibility)
        """
        w, h = image.size

        # ── 1. Random scale ───────────────────────────────────────────────────
        scale = random.uniform(*self.scale_range)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        image = image.resize((new_w, new_h), Image.BICUBIC)
        uv = uv * scale
        w, h = new_w, new_h

        # Crop or pad back to original size (centre crop)
        image, uv = _centre_crop(image, uv, w, h)

        # ── 2. Random rotation ────────────────────────────────────────────────
        angle = random.uniform(*self.rotation_range)
        image = TF.rotate(image, angle, interpolation=Image.BICUBIC,
                          expand=False)
        uv = _rotate_keypoints(uv, angle, cx=HR_SIZE / 2, cy=HR_SIZE / 2)

        # Update visibility after rotation (keypoints may leave the frame)
        visibility = (
            (uv[:, 0] >= 0) & (uv[:, 0] < HR_SIZE) &
            (uv[:, 1] >= 0) & (uv[:, 1] < HR_SIZE)
        ) & visibility

        # ── 3. Random horizontal flip ─────────────────────────────────────────
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            uv[:, 0] = HR_SIZE - 1 - uv[:, 0]
            # Permute keypoints if flip pairs are defined
            if self.FLIP_PAIRS:
                uv = uv[self._flip_permutation()]
                visibility = visibility[self._flip_permutation()]

        # ── 4. Colour jitter ──────────────────────────────────────────────────
        image = _colour_jitter(image, prob=self.colour_prob)

        return image, uv, visibility

    def _flip_permutation(self):
        """Build index array for keypoint permutation under horizontal flip."""
        perm = list(range(21))
        for a, b in self.FLIP_PAIRS:
            perm[a], perm[b] = b, a
        return perm


# ── Spatial helpers ────────────────────────────────────────────────────────────

def _centre_crop(
    image: Image.Image,
    uv: np.ndarray,
    w: int,
    h: int,
    target: int = HR_SIZE,
) -> tuple:
    """
    Centre-crop (or pad) the image to (target × target) and adjust keypoints.
    """
    # Compute crop/pad offsets
    offset_x = (w - target) // 2
    offset_y = (h - target) // 2

    if w >= target and h >= target:
        # Crop
        image = image.crop((offset_x, offset_y,
                             offset_x + target, offset_y + target))
    else:
        # Pad with black to target size then centre
        padded = Image.new("RGB", (target, target), (0, 0, 0))
        paste_x = max(0, -offset_x)
        paste_y = max(0, -offset_y)
        padded.paste(image, (paste_x, paste_y))
        image = padded
        offset_x = -paste_x
        offset_y = -paste_y

    uv = uv - np.array([offset_x, offset_y], dtype=np.float32)
    return image, uv


def _rotate_keypoints(
    uv: np.ndarray,
    angle_deg: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Rotate 2D keypoints around (cx, cy) by angle_deg (counter-clockwise).
    PIL's TF.rotate is also counter-clockwise.
    """
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # Translate to origin
    uv_c = uv - np.array([cx, cy])

    # Rotation matrix (CCW)
    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]], dtype=np.float32)
    uv_rot = (R @ uv_c.T).T

    # Translate back
    return uv_rot + np.array([cx, cy])


# ── Colour jitter ──────────────────────────────────────────────────────────────

def _colour_jitter(image: Image.Image, prob: float = 0.5) -> Image.Image:
    """Apply random brightness, contrast, and saturation jitter."""
    if random.random() < prob:
        image = ImageEnhance.Brightness(image).enhance(
            random.uniform(0.7, 1.3))
    if random.random() < prob:
        image = ImageEnhance.Contrast(image).enhance(
            random.uniform(0.7, 1.3))
    if random.random() < prob:
        image = ImageEnhance.Color(image).enhance(
            random.uniform(0.7, 1.3))
    return image
