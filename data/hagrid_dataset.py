"""
hagrid_dataset.py

HaGRID dataset loader for Super-FAN hands adaptation.
Drop-in replacement for freihand_dataset.py — produces identical
(lr, hr, heatmaps, uv, visible) batches consumed by trainer.py.

HaGRID directory layout expected:
    hagrid/
        <gesture>/              # e.g. fist/, like/, ok/, ...
            <user_id>.jpg       # one image per annotation entry
        annotations/
            <gesture>.json      # one JSON file per gesture class

Annotation format (per JSON file):
    {
        "<image_id>": {
            "bboxes": [[cx_norm, cy_norm, w_norm, h_norm], ...],
            "labels": ["<gesture>", ...],
            "hand_landmarks": [
                [                       # one list per hand / bbox
                    [x_norm, y_norm],   # 21 landmarks in MediaPipe order
                    ...
                ],
                ...
            ],
            "user_id": "<hex string>"
        },
        ...
    }

All coordinates (bboxes and landmarks) are normalised to [0, 1] relative
to the full image dimensions (width for x, height for y).

MediaPipe / HaGRID keypoint ordering (21 joints):
    0:  Wrist
    1:  Thumb CMC   2: Thumb MCP   3: Thumb IP    4: Thumb TIP
    5:  Index MCP   6: Index PIP   7: Index DIP   8: Index TIP
    9:  Middle MCP  10: Middle PIP 11: Middle DIP 12: Middle TIP
    13: Ring MCP    14: Ring PIP   15: Ring DIP   16: Ring TIP
    17: Pinky MCP   18: Pinky PIP  19: Pinky DIP  20: Pinky TIP
"""

import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from data.constants import (
    NUM_KEYPOINTS, HR_SIZE, LR_SIZE,
    HEATMAP_SIZE, HEATMAP_SIGMA, HAND_BONES,
)
from data.augmentations import HandAugmentor


# ── Heatmap utilities ──────────────────────────────────────────────────────────

def make_heatmaps(
    uv: np.ndarray,
    size: int = HEATMAP_SIZE,
    sigma: float = HEATMAP_SIGMA,
    visibility: np.ndarray = None,
) -> np.ndarray:
    """
    Generate a stack of 2D Gaussian heatmaps, one per keypoint.

    Args:
        uv:         (21, 2) keypoint coordinates in [0, size) space
        size:       spatial size of each heatmap (height == width)
        sigma:      Gaussian standard deviation in pixels
        visibility: (21,) boolean mask; if None all keypoints are visible

    Returns:
        heatmaps: (21, size, size) float32 array, values in [0, 1]
    """
    if visibility is None:
        visibility = np.ones(NUM_KEYPOINTS, dtype=bool)

    heatmaps = np.zeros((NUM_KEYPOINTS, size, size), dtype=np.float32)
    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)  # both (size, size)

    for i in range(NUM_KEYPOINTS):
        if not visibility[i]:
            continue
        cx, cy = uv[i]
        if cx < 0 or cy < 0 or cx >= size or cy >= size:
            continue
        gauss = np.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
                       / (2 * sigma ** 2))
        heatmaps[i] = gauss

    return heatmaps


# ── Bounding box helper ────────────────────────────────────────────────────────

def hagrid_bbox_to_pixels(
    bbox_norm: list,
    img_w: int,
    img_h: int,
) -> tuple:
    """
    Convert a HaGRID normalised bounding box to pixel coordinates.

    HaGRID stores bboxes as [cx_norm, cy_norm, w_norm, h_norm].

    Returns:
        (x_min, y_min, x_max, y_max) as floats in pixel space
    """
    cx, cy, bw, bh = bbox_norm
    x_min = (cx - bw / 2) * img_w
    y_min = (cy - bh / 2) * img_h
    x_max = (cx + bw / 2) * img_w
    y_max = (cy + bh / 2) * img_h
    return x_min, y_min, x_max, y_max


def pad_to_square(image: Image.Image) -> tuple:
    """
    Pad the shorter dimension of an image with black bars so the result is
    square. The longer dimension is kept entirely — no pixels are discarded.

    Args:
        image: PIL image of any aspect ratio.

    Returns:
        (square_pil, paste_x, paste_y)
            square_pil: square PIL image, same size on both axes.
            paste_x:    x pixel offset where the original image was pasted.
            paste_y:    y pixel offset where the original image was pasted.
    """
    w, h = image.size
    side = max(w, h)
    paste_x = (side - w) // 2
    paste_y = (side - h) // 2
    square = Image.new("RGB", (side, side), (0, 0, 0))
    square.paste(image, (paste_x, paste_y))
    return square, paste_x, paste_y


def remap_keypoints(
    uv: np.ndarray,
    crop_x_min: float,
    crop_y_min: float,
    paste_x: int,
    paste_y: int,
    square_side: int,
    target_size: int,
) -> np.ndarray:
    """
    Remap keypoints from original image space into the padded-square space,
    then scale to target_size.

    Steps:
        1. Subtract crop origin  ->  coords relative to the raw crop
        2. Add paste offset      ->  coords in the square canvas
        3. Scale by target_size / square_side

    Args:
        uv:          (21, 2) keypoints in original image pixel coords
        crop_x_min:  x_min of the PIL crop in original image space
        crop_y_min:  y_min of the PIL crop in original image space
        paste_x:     x offset where crop was pasted inside the square
        paste_y:     y offset likewise
        square_side: side length of the square canvas in pixels
        target_size: desired output size (HR_SIZE)

    Returns:
        uv_scaled: (21, 2) keypoints in [0, target_size) space
    """
    uv_in_crop   = uv - np.array([crop_x_min, crop_y_min])
    uv_in_square = uv_in_crop + np.array([paste_x, paste_y])
    scale        = target_size / square_side
    return uv_in_square * scale


# ── Dataset ────────────────────────────────────────────────────────────────────

class HaGRIDDataset(Dataset):
    """
    PyTorch Dataset for HaGRID, producing (LR, HR, heatmap) triplets.

    Each item:
        lr_image:  (3, LR_SIZE, LR_SIZE)   float32 tensor in [-1, 1]
        hr_image:  (3, HR_SIZE, HR_SIZE)   float32 tensor in [-1, 1]
        heatmaps:  (21, HEATMAP_SIZE, HEATMAP_SIZE) float32 in [0, 1]
        uv_scaled: (21, 2)                 float32 tensor in HR_SIZE space
        visible:   (21,)                   bool tensor

    Samples without landmark annotations are silently skipped during
    index construction so every __getitem__ call is guaranteed to succeed.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        train_frac: float = 0.9,
        augment: bool = True,
        simulate_real_world: bool = True,
        gestures: list = None,
        max_samples: int = None,
    ):
        """
        Args:
            root:               Path to the hagrid/ root directory.
            split:              "train" or "val".
            train_frac:         Fraction of samples used for training.
            augment:            Whether to apply geometric + colour augmentation.
            simulate_real_world: Add blur / JPEG / colour distortion to LR images.
            gestures:           Optional list of gesture names to include.
                                If None, all gestures found under annotations/ are used.
            max_samples:        Cap the dataset at this many samples (useful for
                                quick debugging runs). Applied after the train/val
                                split, so the ratio is preserved.
        """
        super().__init__()
        self.root = root
        self.augment = augment
        self.simulate_real_world = simulate_real_world
        self.augmentor = HandAugmentor() if augment else None

        # ── Discover gesture/annotation files ─────────────────────────────────
        ann_dir = os.path.join(root, "annotations")
        if not os.path.isdir(ann_dir):
            raise FileNotFoundError(
                f"Annotations directory not found: {ann_dir}\n"
                f"Expected layout: {root}/annotations/<gesture>.json"
            )

        available = [
            f[:-5] for f in os.listdir(ann_dir) if f.endswith(".json")
        ]
        if gestures is not None:
            gesture_list = [g for g in gestures if g in available]
            missing = set(gestures) - set(available)
            if missing:
                print(f"[HaGRIDDataset] Warning: gestures not found: {missing}")
        else:
            gesture_list = sorted(available)

        if not gesture_list:
            raise ValueError(
                f"No gesture annotation files found in {ann_dir}"
            )

        # ── Build flat sample list ─────────────────────────────────────────────
        # Each entry: (image_path, landmarks_array (21, 2) normalised, gesture)
        all_samples = []

        for gesture in gesture_list:
            ann_path = os.path.join(ann_dir, f"{gesture}.json")
            img_dir  = os.path.join(root, gesture)

            with open(ann_path) as f:
                annotations = json.load(f)

            for image_id, ann in annotations.items():
                landmarks_list = ann.get("hand_landmarks")
                if not landmarks_list:
                    continue  # skip samples without landmark annotations

                user_id = ann.get("user_id", image_id)
                img_path = os.path.join(img_dir, f"{user_id}.jpg")
                if not os.path.isfile(img_path):
                    # fallback: try image_id as filename
                    img_path = os.path.join(img_dir, f"{image_id}.jpg")
                if not os.path.isfile(img_path):
                    continue  # image not present on disk — skip

                # There may be multiple hands per image; emit one sample per hand
                bboxes = ann.get("bboxes", [None] * len(landmarks_list))
                for hand_idx, lm in enumerate(landmarks_list):
                    if len(lm) != NUM_KEYPOINTS:
                        continue  # incomplete annotation
                    lm_array = np.array(lm, dtype=np.float32)  # (21, 2) normalised

                    bbox_norm = bboxes[hand_idx] if hand_idx < len(bboxes) else None
                    all_samples.append((img_path, lm_array, bbox_norm, gesture))

        # ── Train / val split (deterministic) ─────────────────────────────────
        n_total = len(all_samples)
        n_train = int(n_total * train_frac)
        if split == "train":
            self.samples = all_samples[:n_train]
        else:
            self.samples = all_samples[n_train:]

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(
            f"[HaGRIDDataset] {split}: {len(self.samples)} samples "
            f"across {len(gesture_list)} gestures"
            + (f" (capped at {max_samples})" if max_samples is not None else "")
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, lm_norm, bbox_norm, gesture = self.samples[idx]

        # ── Load image ────────────────────────────────────────────────────────
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self))
        img_w, img_h = image.size

        # ── Denormalise landmarks to pixel coordinates ─────────────────────────
        # lm_norm: (21, 2) where column 0 is x (width) and column 1 is y (height)
        uv = lm_norm * np.array([img_w, img_h], dtype=np.float32)  # (21, 2)

        # ── Crop to keypoint extents + padding, then pad to square ──────────
        # Use the keypoint extents as the crop region — they are guaranteed
        # to contain the hand exactly, unlike the annotation bbox which can
        # be loosely fitted or misaligned. A padding margin is added so the
        # outermost keypoints are not sitting right at the crop edge.
        # The shorter axis is then padded with black bars to form a square.
        kp_x_min, kp_y_min = uv.min(axis=0)
        kp_x_max, kp_y_max = uv.max(axis=0)

        kp_w = kp_x_max - kp_x_min
        kp_h = kp_y_max - kp_y_min
        margin = max(kp_w, kp_h) * 0.2   # 20% of the hand extent

        x_min_c = max(0,         int(kp_x_min - margin))
        y_min_c = max(0,         int(kp_y_min - margin))
        x_max_c = min(img_w - 1, int(kp_x_max + margin))
        y_max_c = min(img_h - 1, int(kp_y_max + margin))

        if x_max_c <= x_min_c or y_max_c <= y_min_c:
            return self.__getitem__((idx + 1) % len(self))

        raw_crop = image.crop((x_min_c, y_min_c, x_max_c, y_max_c))
        image_crop, paste_x, paste_y = pad_to_square(raw_crop)
        square_side = image_crop.size[0]

        # ── Remap keypoints into the padded-square space ──────────────────────
        uv_scaled = remap_keypoints(
            uv, x_min_c, y_min_c, paste_x, paste_y, square_side, HR_SIZE
        )

        visibility = (
            (uv_scaled[:, 0] >= 0) & (uv_scaled[:, 0] < HR_SIZE) &
            (uv_scaled[:, 1] >= 0) & (uv_scaled[:, 1] < HR_SIZE)
        )

        # ── Augmentation ──────────────────────────────────────────────────────
        if self.augment and self.augmentor is not None:
            image_crop, uv_scaled, visibility = self.augmentor(
                image_crop, uv_scaled, visibility
            )

        # Recompute visibility after augmentation
        visibility = (
            (uv_scaled[:, 0] >= 0) & (uv_scaled[:, 0] < HR_SIZE) &
            (uv_scaled[:, 1] >= 0) & (uv_scaled[:, 1] < HR_SIZE)
        )

        # ── Build HR and LR images ─────────────────────────────────────────────
        hr_pil = image_crop.resize((HR_SIZE, HR_SIZE), Image.BICUBIC)
        lr_pil = image_crop.resize((LR_SIZE, LR_SIZE), Image.BICUBIC)

        if self.simulate_real_world:
            lr_pil = _simulate_degradation(lr_pil)

        hr_tensor = _pil_to_tensor(hr_pil)   # (3, HR_SIZE, HR_SIZE)
        lr_tensor = _pil_to_tensor(lr_pil)   # (3, LR_SIZE, LR_SIZE)

        # ── Build ground-truth heatmaps ───────────────────────────────────────
        uv_for_hm = uv_scaled * (HEATMAP_SIZE / HR_SIZE)
        heatmaps = make_heatmaps(
            uv_for_hm, size=HEATMAP_SIZE,
            sigma=HEATMAP_SIGMA, visibility=visibility
        )
        heatmaps_tensor = torch.from_numpy(heatmaps)                         # (21, H, H)
        uv_tensor       = torch.from_numpy(uv_scaled.astype(np.float32))    # (21, 2)

        return {
            "lr":       lr_tensor,                          # (3,  LR_SIZE, LR_SIZE)
            "hr":       hr_tensor,                          # (3,  HR_SIZE, HR_SIZE)
            "heatmaps": heatmaps_tensor,                    # (21, HEATMAP_SIZE, HEATMAP_SIZE)
            "uv":       uv_tensor,                          # (21, 2)  — for PCK eval
            "visible":  torch.from_numpy(visibility),       # (21,) bool
        }


# ── Image helpers ──────────────────────────────────────────────────────────────

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to float32 tensor in [-1, 1]."""
    t = TF.to_tensor(img)   # [0, 1]
    t = t * 2.0 - 1.0       # [-1, 1]
    return t


def _simulate_degradation(lr_pil: Image.Image) -> Image.Image:
    """
    Simulate real-world LR image degradations: random Gaussian blur,
    JPEG compression artefacts, and colour/brightness distortion.
    """
    import random
    from PIL import ImageFilter, ImageEnhance
    import io

    if random.random() > 0.3:
        radius = random.uniform(0.3, 1.5)
        lr_pil = lr_pil.filter(ImageFilter.GaussianBlur(radius=radius))

    if random.random() > 0.4:
        quality = random.randint(40, 85)
        buf = io.BytesIO()
        lr_pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        lr_pil = Image.open(buf).convert("RGB")

    if random.random() > 0.3:
        lr_pil = ImageEnhance.Brightness(lr_pil).enhance(random.uniform(0.7, 1.3))
    if random.random() > 0.3:
        lr_pil = ImageEnhance.Contrast(lr_pil).enhance(random.uniform(0.7, 1.3))
    if random.random() > 0.3:
        lr_pil = ImageEnhance.Color(lr_pil).enhance(random.uniform(0.7, 1.3))

    return lr_pil


# ── DataLoader factory ─────────────────────────────────────────────────────────

def build_dataloaders(
    root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    simulate_real_world: bool = True,
    gestures: list = None,
    max_samples: int = None,
) -> tuple:
    """
    Convenience function to build train and val DataLoaders.

    Args:
        root:                Path to hagrid/ root directory.
        batch_size:          Samples per batch.
        num_workers:         DataLoader worker processes.
        simulate_real_world: Pass-through to HaGRIDDataset.
        gestures:            Optional list of gesture names to include.
        max_samples:         Cap each split at this many samples. The val cap
                             is scaled by (1 - train_frac) / train_frac so the
                             train/val ratio stays consistent.

    Returns:
        (train_loader, val_loader)
    """
    train_ds = HaGRIDDataset(
        root, split="train", augment=True,
        simulate_real_world=simulate_real_world,
        gestures=gestures,
        max_samples=max_samples,
    )
    val_ds = HaGRIDDataset(
        root, split="val", augment=False,
        simulate_real_world=False,
        gestures=gestures,
        max_samples=max_samples // 9 if max_samples is not None else None,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader


# ── Quick sanity-check ─────────────────────────────────────────────────────────

def sanity_check(root: str, n: int = 8, out: str = "dataset_sanity_check.png",
                 gestures: list = None):
    """
    Save a grid of n samples showing LR | HR | HR+skeleton for each.
    Useful to verify crop, padding, and keypoint alignment before training.

    Args:
        root:     Path to hagrid/ root directory.
        n:        Number of samples to visualise (default 8).
        out:      Output PNG path.
        gestures: Optional gesture filter passed to HaGRIDDataset.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    FINGER_COLOURS = {
        "thumb":  "#CE93D8",
        "index":  "#4FC3F7",
        "middle": "#81C784",
        "ring":   "#FFB74D",
        "pinky":  "#F06292",
    }
    BONE_COLOURS = (
        [FINGER_COLOURS["thumb"]]  * 4 +
        [FINGER_COLOURS["index"]]  * 4 +
        [FINGER_COLOURS["middle"]] * 4 +
        [FINGER_COLOURS["ring"]]   * 4 +
        [FINGER_COLOURS["pinky"]]  * 4
    )

    def _t(tensor):
        return ((tensor.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)

    def _draw(ax, uv, vis):
        for bone_idx, (p, c) in enumerate(HAND_BONES):
            if vis[p] and vis[c]:
                col = BONE_COLOURS[bone_idx] if bone_idx < len(BONE_COLOURS) else "w"
                ax.plot([uv[p, 0], uv[c, 0]], [uv[p, 1], uv[c, 1]],
                        color=col, linewidth=1.5)
        ax.scatter(uv[vis, 0], uv[vis, 1], c="white", s=12,
                   zorder=5, edgecolors="black", linewidths=0.4)

    ds = HaGRIDDataset(root, split="train", augment=False,
                       simulate_real_world=False, gestures=gestures,
                       max_samples=n)

    actual_n = min(n, len(ds))
    ncols = 3   # LR | HR | HR + skeleton
    fig, axes = plt.subplots(actual_n, ncols,
                             figsize=(ncols * 3, actual_n * 3))
    fig.patch.set_facecolor("#111111")

    if actual_n == 1:
        axes = axes[np.newaxis, :]

    col_titles = [f"LR ({LR_SIZE}×{LR_SIZE})",
                  f"HR ({HR_SIZE}×{HR_SIZE})",
                  "HR + skeleton"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, color="#cccccc", fontsize=9, pad=4)

    for row in range(actual_n):
        sample = ds[row]
        lr_np  = _t(sample["lr"])
        hr_np  = _t(sample["hr"])
        uv     = sample["uv"].numpy()
        vis    = sample["visible"].numpy()

        axes[row, 0].imshow(lr_np)
        axes[row, 1].imshow(hr_np)
        # Upsample heatmap from HEATMAP_SIZE to HR_SIZE before overlaying
        hm_sum = sample["heatmaps"].sum(0).numpy()
        hm_up  = np.array(Image.fromarray(hm_sum).resize(
            (HR_SIZE, HR_SIZE), Image.BILINEAR))
        axes[row, 2].imshow(hr_np)
        axes[row, 2].imshow(hm_up, alpha=0.4, cmap="hot")
        _draw(axes[row, 2], uv, vis)

        visible_count = vis.sum()
        axes[row, 0].set_ylabel(f"{visible_count}/21 kp",
                                color="#aaaaaa", fontsize=7)

        for col in range(ncols):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].set_facecolor("#111111")
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor("#333333")

    plt.tight_layout(pad=0.5)
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {actual_n}-sample sanity check → {out}")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="HaGRID dataset sanity check")
    parser.add_argument("root", help="Path to hagrid/ root directory")
    parser.add_argument("--n",        type=int,   default=8,
                        help="Number of samples to visualise (default: 8)")
    parser.add_argument("--out",      default="dataset_sanity_check.png",
                        help="Output PNG path")
    parser.add_argument("--gestures", nargs="*",  default=None,
                        help="Gesture filter, e.g. --gestures fist like")
    args = parser.parse_args()

    sanity_check(args.root, n=args.n, out=args.out, gestures=args.gestures)