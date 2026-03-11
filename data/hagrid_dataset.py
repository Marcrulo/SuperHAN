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


def make_square_bbox(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    padding: float = 0.2,
) -> tuple:
    """
    Expand a bbox with padding and make it square (keeping centre fixed).

    Returns:
        (x_min, y_min, x_max, y_max) unclamped square bbox
    """
    w = x_max - x_min
    h = y_max - y_min
    pad = max(w, h) * padding

    x_min -= pad
    y_min -= pad
    x_max += pad
    y_max += pad

    side = max(x_max - x_min, y_max - y_min)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    x_min = cx - side / 2
    y_min = cy - side / 2
    x_max = cx + side / 2
    y_max = cy + side / 2

    return x_min, y_min, x_max, y_max


def crop_and_scale_keypoints(
    uv: np.ndarray,
    bbox: tuple,
    target_size: int,
) -> np.ndarray:
    """
    Remap 2D keypoints from the original image space into the cropped,
    resized image space.

    Args:
        uv:          (21, 2) keypoints in original image pixel coords
        bbox:        (x_min, y_min, x_max, y_max) square crop box
        target_size: side length after resizing (e.g. HR_SIZE=128)

    Returns:
        uv_scaled: (21, 2) keypoints in [0, target_size) space
    """
    x_min, y_min, x_max, y_max = bbox
    box_w = x_max - x_min
    box_h = y_max - y_min

    uv_cropped = uv - np.array([x_min, y_min])
    scale_x = target_size / box_w
    scale_y = target_size / box_h
    return uv_cropped * np.array([scale_x, scale_y])


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

        print(
            f"[HaGRIDDataset] {split}: {len(self.samples)} samples "
            f"across {len(gesture_list)} gestures"
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

        # ── Determine crop bounding box ───────────────────────────────────────
        if bbox_norm is not None:
            # Use the provided annotation bbox as the crop centre, then square it
            x_min_px, y_min_px, x_max_px, y_max_px = hagrid_bbox_to_pixels(
                bbox_norm, img_w, img_h
            )
        else:
            # Fall back to tight bbox from keypoints themselves
            x_min_px, y_min_px = uv.min(axis=0)
            x_max_px, y_max_px = uv.max(axis=0)

        bbox_square = make_square_bbox(x_min_px, y_min_px, x_max_px, y_max_px)

        # Clamp for PIL crop (cannot read outside image)
        x_min_c = max(0,         int(bbox_square[0]))
        y_min_c = max(0,         int(bbox_square[1]))
        x_max_c = min(img_w - 1, int(bbox_square[2]))
        y_max_c = min(img_h - 1, int(bbox_square[3]))

        if x_max_c <= x_min_c or y_max_c <= y_min_c:
            return self.__getitem__((idx + 1) % len(self))

        image_crop = image.crop((x_min_c, y_min_c, x_max_c, y_max_c))

        # ── Remap keypoints using the unclamped square bbox ───────────────────
        uv_scaled = crop_and_scale_keypoints(uv, bbox_square, HR_SIZE)

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
) -> tuple:
    """
    Convenience function to build train and val DataLoaders.

    Args:
        root:                Path to hagrid/ root directory.
        batch_size:          Samples per batch.
        num_workers:         DataLoader worker processes.
        simulate_real_world: Pass-through to HaGRIDDataset.
        gestures:            Optional list of gesture names to include.

    Returns:
        (train_loader, val_loader)
    """
    train_ds = HaGRIDDataset(
        root, split="train", augment=True,
        simulate_real_world=simulate_real_world,
        gestures=gestures,
    )
    val_ds = HaGRIDDataset(
        root, split="val", augment=False,
        simulate_real_world=False,
        gestures=gestures,
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

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    root = sys.argv[1] if len(sys.argv) > 1 else "./hagrid"

    ds = HaGRIDDataset(root, split="train", augment=False,
                       simulate_real_world=False)
    print(f"Dataset size: {len(ds)}")

    sample = ds[0]
    print("lr shape:      ", sample["lr"].shape)
    print("hr shape:      ", sample["hr"].shape)
    print("heatmaps shape:", sample["heatmaps"].shape)
    print("uv shape:      ", sample["uv"].shape)
    print("visible:       ", sample["visible"].sum().item(), "/ 21 keypoints")

    JOINT_NAMES = [
        "Wrist",
        "Thumb-CMC", "Thumb-MCP", "Thumb-IP",  "Thumb-TIP",
        "Index-MCP", "Index-PIP", "Index-DIP",  "Index-TIP",
        "Mid-MCP",   "Mid-PIP",   "Mid-DIP",    "Mid-TIP",
        "Ring-MCP",  "Ring-PIP",  "Ring-DIP",   "Ring-TIP",
        "Pinky-MCP", "Pinky-PIP", "Pinky-DIP",  "Pinky-TIP",
    ]
    vis_np = sample["visible"].numpy()
    uv_np  = sample["uv"].numpy()
    print("\nPer-joint visibility:")
    for i, (name, v) in enumerate(zip(JOINT_NAMES, vis_np)):
        coord  = f"({uv_np[i,0]:.1f}, {uv_np[i,1]:.1f})"
        status = "OK" if v else "OUTSIDE"
        print(f"  {i:2d}  {name:<12s}  {coord:<16s}  {status}")

    hr_np  = ((sample["hr"].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
    lr_np  = ((sample["lr"].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
    hm_sum = sample["heatmaps"].sum(0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(lr_np);  axes[0].set_title(f"LR input ({LR_SIZE}×{LR_SIZE})")
    axes[1].imshow(hr_np);  axes[1].set_title(f"HR target ({HR_SIZE}×{HR_SIZE})")
    axes[2].imshow(hr_np)
    axes[2].imshow(hm_sum, alpha=0.5, cmap="hot")
    axes[2].set_title("HR + GT heatmaps")

    uv  = sample["uv"].numpy()
    vis = sample["visible"].numpy()
    for (p, c) in HAND_BONES:
        if vis[p] and vis[c]:
            axes[2].plot([uv[p, 0], uv[c, 0]], [uv[p, 1], uv[c, 1]],
                         "c-", linewidth=1)
    axes[2].scatter(uv[vis, 0], uv[vis, 1], c="lime", s=10, zorder=5)

    plt.tight_layout()
    plt.savefig("dataset_sanity_check.png", dpi=150)
    print("Saved dataset_sanity_check.png")
