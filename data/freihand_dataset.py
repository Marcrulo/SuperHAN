"""
freihand_dataset.py

FreiHAND dataset loader for Super-FAN hands adaptation.

FreiHAND directory layout expected:
    freihand/
        training/
            rgb/          # 00000000.jpg ... 00032559.jpg
            mask/         # 00000000.jpg ... (optional, for visibility)
        training_K.json   # camera intrinsics, shape (32560, 3, 3)
        training_xyz.json # 3D keypoints in camera space, shape (32560, 21, 3)
        training_scale.json # scale factor per sample

FreiHAND keypoint ordering (21 joints):
    0: Wrist
    1-4:   Index  MCP, PIP, DIP, TIP
    5-8:   Middle MCP, PIP, DIP, TIP
    9-12:  Ring   MCP, PIP, DIP, TIP
    13-16: Pinky  MCP, PIP, DIP, TIP
    17-20: Thumb  MCP, IP,  DIP, TIP
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

def keypoints_3d_to_2d(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Project 3D keypoints in camera space to 2D image coordinates.

    Args:
        xyz: (21, 3) array of 3D keypoints in camera space (metres)
        K:   (3, 3) camera intrinsic matrix

    Returns:
        uv: (21, 2) array of 2D pixel coordinates (x, y)
    """
    # FreiHAND's own projection formula (from their GitHub repo):
    # uv = K @ xyz.T  then divide by homogeneous coordinate
    uv = np.matmul(K, xyz.T).T    # (21, 3)
    return uv[:, :2] / uv[:, 2:3] # (21, 2) pixel coords


def crop_and_scale_keypoints(
    uv: np.ndarray,
    bbox: tuple,
    target_size: int
) -> np.ndarray:
    """
    Remap 2D keypoints from the original image space into the cropped,
    resized image space used as our HR target.

    Args:
        uv:          (21, 2) keypoints in original image pixel coords
        bbox:        (x_min, y_min, x_max, y_max) crop box in original image
        target_size: side length of the square crop after resizing (e.g. 64)

    Returns:
        uv_scaled: (21, 2) keypoints in [0, target_size) space
    """
    x_min, y_min, x_max, y_max = bbox
    box_w = x_max - x_min
    box_h = y_max - y_min

    uv_cropped = uv - np.array([x_min, y_min])
    scale_x = target_size / box_w
    scale_y = target_size / box_h
    uv_scaled = uv_cropped * np.array([scale_x, scale_y])
    return uv_scaled


def make_heatmaps(
    uv: np.ndarray,
    size: int = HEATMAP_SIZE,
    sigma: float = HEATMAP_SIGMA,
    visibility: np.ndarray = None
) -> np.ndarray:
    """
    Generate a stack of 2D Gaussian heatmaps, one per keypoint.

    This directly replaces the teacher FAN's role from the paper: rather than
    running a frozen pretrained network on HR images to obtain heatmap targets,
    we construct ground-truth heatmaps from FreiHAND's 3D annotations.

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
            continue  # leave channel all-zero for occluded joints
        cx, cy = uv[i]
        # Skip keypoints projected outside the crop
        if cx < 0 or cy < 0 or cx >= size or cy >= size:
            continue
        gauss = np.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
                       / (2 * sigma ** 2))
        heatmaps[i] = gauss

    return heatmaps


# ── Bounding box helper ────────────────────────────────────────────────────────

def keypoints_to_bbox(uv: np.ndarray, padding: float = 0.35) -> tuple:
    """
    Compute a square bounding box around the visible hand keypoints.

    Returns the bbox BEFORE clamping to image bounds — the caller is
    responsible for clamping.  Keeping the unclamped square here means
    crop_and_scale_keypoints always gets a consistent square side-length
    to scale against, even when part of the box falls outside the image.

    Args:
        uv:      (21, 2) 2D keypoints in original image space
        padding: fractional padding around the tight bbox (default 35%)

    Returns:
        (x_min, y_min, x_max, y_max) as floats (may be outside image bounds)
    """
    x_min, y_min = uv.min(axis=0)
    x_max, y_max = uv.max(axis=0)

    w = x_max - x_min
    h = y_max - y_min
    pad = max(w, h) * padding   # uniform padding based on longer side

    x_min = x_min - pad
    y_min = y_min - pad
    x_max = x_max + pad
    y_max = y_max + pad

    # Make square by extending the shorter side, keeping centre fixed
    side = max(x_max - x_min, y_max - y_min)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    x_min = cx - side / 2
    y_min = cy - side / 2
    x_max = cx + side / 2
    y_max = cy + side / 2

    return x_min, y_min, x_max, y_max


# ── Dataset ────────────────────────────────────────────────────────────────────

class FreiHANDDataset(Dataset):
    """
    PyTorch Dataset for FreiHAND, producing (LR, HR, heatmap) triplets.

    Each item:
        lr_image:  (3, 16, 16)  float32 tensor in [-1, 1]  — SR network input
        hr_image:  (3, 64, 64)  float32 tensor in [-1, 1]  — SR target / discriminator
        heatmaps:  (21, 64, 64) float32 tensor in [0, 1]   — hand-FAN supervision target
        uv_scaled: (21, 2)      float32 tensor             — 2D keypoints in HR space
                                                             (useful for PCK evaluation)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        train_frac: float = 0.9,
        augment: bool = True,
        simulate_real_world: bool = True,
    ):
        """
        Args:
            root:               Path to the freihand/ directory.
            split:              "train" or "val".
            train_frac:         Fraction of samples used for training.
            augment:            Whether to apply geometric + colour augmentation.
            simulate_real_world: Add Gaussian blur / JPEG / colour distortion to
                                 LR images, as described in Section 5.3 of the paper,
                                 to improve robustness on real-world low-res images.
        """
        super().__init__()
        self.root = root
        self.augment = augment
        self.simulate_real_world = simulate_real_world
        self.augmentor = HandAugmentor() if augment else None

        # ── Load annotations ──────────────────────────────────────────────────
        with open(os.path.join(root, "training_K.json")) as f:
            all_K = np.array(json.load(f), dtype=np.float32)      # (32560, 3, 3)

        with open(os.path.join(root, "training_xyz.json")) as f:
            all_xyz = np.array(json.load(f), dtype=np.float32)    # (32560, 21, 3)

        # train/val split (deterministic — no shuffle so val is reproducible)
        n_total = len(all_K)
        n_train = int(n_total * train_frac)
        if split == "train":
            indices = np.arange(n_train)
        else:
            indices = np.arange(n_train, n_total)

        self.K_list   = all_K[indices]    # (N, 3, 3)
        self.xyz_list = all_xyz[indices]  # (N, 21, 3)
        self.indices  = indices
        self.img_dir  = os.path.join(root, "training", "rgb")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        sample_id = self.indices[idx]
        img_path  = os.path.join(self.img_dir, f"{sample_id:08d}.jpg")

        # ── Load image ────────────────────────────────────────────────────────
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # ── Project 3D → 2D keypoints ─────────────────────────────────────────
        xyz = self.xyz_list[idx]   # (21, 3)
        K   = self.K_list[idx]     # (3, 3)
        uv  = keypoints_3d_to_2d(xyz, K)  # (21, 2) in original image space

        # Sanity check: if the projected keypoints are wildly outside the image
        # (e.g. due to a near-zero Z value) skip this sample by returning the
        # next one.  This is rare in FreiHAND but guards against corrupt entries.
        MAX_ALLOWED = max(img_w, img_h) * 5
        if np.any(np.abs(uv) > MAX_ALLOWED) or np.any(xyz[:, 2] < 1e-3):
            return self.__getitem__((idx + 1) % len(self))

        # Clamp to valid image bounds for bbox computation
        uv_clamped = np.clip(uv, 0, [img_w - 1, img_h - 1])

        # ── Crop to hand bounding box ─────────────────────────────────────────
        # Compute bbox, then clamp to image bounds.  Critically, we pass the
        # SAME clamped bbox to both the PIL crop and keypoint remapping so
        # that scale factors stay consistent.
        # keypoints_to_bbox returns an unclamped square box — we use this
        # for keypoint scaling so the scale factor is always (HR_SIZE / square_side).
        # We clamp separately only for the PIL crop so we don't read outside the image.
        bbox_square = keypoints_to_bbox(uv_clamped)   # unclamped float square
        x_min_c = max(0,          int(bbox_square[0]))
        y_min_c = max(0,          int(bbox_square[1]))
        x_max_c = min(img_w - 1,  int(bbox_square[2]))
        y_max_c = min(img_h - 1,  int(bbox_square[3]))

        image_crop = image.crop((x_min_c, y_min_c, x_max_c, y_max_c))

        # ── Remap keypoints using the UNCLAMPED square bbox ───────────────────
        # This keeps the scale factor consistent: HR_SIZE / square_side_length.
        # Keypoints that fall in the padded region outside the image will land
        # near 0 or HR_SIZE and get correctly marked as outside by the visibility check.
        uv_scaled = crop_and_scale_keypoints(uv_clamped, bbox_square, HR_SIZE)

        # Determine visibility: keypoints that project inside the HR crop
        visibility = (
            (uv_scaled[:, 0] >= 0) & (uv_scaled[:, 0] < HR_SIZE) &
            (uv_scaled[:, 1] >= 0) & (uv_scaled[:, 1] < HR_SIZE)
        )

        # ── Augmentation (applied consistently to image + keypoints) ──────────
        if self.augment and self.augmentor is not None:
            image_crop, uv_scaled, visibility = self.augmentor(
                image_crop, uv_scaled, visibility
            )

        # Recompute visibility from scratch after augmentation — rotation and
        # scale can move keypoints outside [0, HR_SIZE] even if they were valid
        # before, and the augmentor's internal visibility update is only partial.
        visibility = (
            (uv_scaled[:, 0] >= 0) & (uv_scaled[:, 0] < HR_SIZE) &
            (uv_scaled[:, 1] >= 0) & (uv_scaled[:, 1] < HR_SIZE)
        )

        # ── Build HR and LR images ────────────────────────────────────────────
        hr_pil = image_crop.resize((HR_SIZE, HR_SIZE), Image.BICUBIC)
        lr_pil = image_crop.resize((LR_SIZE, LR_SIZE), Image.BICUBIC)

        # ── Optionally simulate real-world degradation on LR ──────────────────
        if self.simulate_real_world:
            lr_pil = _simulate_degradation(lr_pil)

        # ── Convert to tensors in [-1, 1] ─────────────────────────────────────
        hr_tensor = _pil_to_tensor(hr_pil)   # (3, 64, 64)
        lr_tensor = _pil_to_tensor(lr_pil)   # (3, 16, 16)

        # ── Build ground-truth heatmaps ───────────────────────────────────────
        # This is the key replacement for the paper's frozen teacher FAN:
        # we use FreiHAND's ground truth annotations directly.
        # uv_scaled is in [0, HR_SIZE] space; rescale to [0, HEATMAP_SIZE]
        # so keypoints land at correct positions in the heatmap grid.
        from data.constants import HEATMAP_SIZE as _HM_SIZE
        uv_for_hm = uv_scaled * (_HM_SIZE / HR_SIZE)
        heatmaps = make_heatmaps(uv_for_hm, size=HEATMAP_SIZE,
                                 sigma=HEATMAP_SIGMA, visibility=visibility)
        heatmaps_tensor = torch.from_numpy(heatmaps)   # (21, 64, 64)
        uv_tensor = torch.from_numpy(uv_scaled.astype(np.float32))  # (21, 2)

        return {
            "lr":       lr_tensor,       # (3,  16, 16)
            "hr":       hr_tensor,       # (3,  64, 64)
            "heatmaps": heatmaps_tensor, # (21, 64, 64)
            "uv":       uv_tensor,       # (21, 2)   — for PCK eval
            "visible":  torch.from_numpy(visibility),  # (21,) bool
        }


# ── Image helpers ──────────────────────────────────────────────────────────────

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to float32 tensor in [-1, 1]."""
    t = TF.to_tensor(img)          # [0, 1], shape (3, H, W)
    t = t * 2.0 - 1.0             # [-1, 1]
    return t


def _simulate_degradation(lr_pil: Image.Image) -> Image.Image:
    """
    Simulate real-world LR image degradations as described in Section 5.3
    of the paper: random Gaussian blur, JPEG compression artefacts,
    and colour/brightness distortion.

    Operates on a PIL image at LR_SIZE (16x16).
    """
    import random
    from PIL import ImageFilter, ImageEnhance
    import io

    # 1. Random Gaussian blur (kernel 3–7 px, mapped to radius for PIL)
    if random.random() > 0.3:
        # PIL GaussianBlur radius ≈ sigma; kernel 3px ≈ radius 0.5, 7px ≈ radius 2
        radius = random.uniform(0.3, 1.5)
        lr_pil = lr_pil.filter(ImageFilter.GaussianBlur(radius=radius))

    # 2. JPEG compression artefacts (quality 40–85)
    if random.random() > 0.4:
        quality = random.randint(40, 85)
        buf = io.BytesIO()
        lr_pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        lr_pil = Image.open(buf).convert("RGB")

    # 3. Colour / brightness / contrast jitter
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
) -> tuple:
    """
    Convenience function to build train and val DataLoaders.

    Args:
        root:                Path to freihand/ directory.
        batch_size:          Samples per batch.
        num_workers:         DataLoader worker processes.
        simulate_real_world: Pass-through to FreiHANDDataset.

    Returns:
        (train_loader, val_loader)
    """
    train_ds = FreiHANDDataset(
        root, split="train", augment=True,
        simulate_real_world=simulate_real_world
    )
    val_ds = FreiHANDDataset(
        root, split="val", augment=False,
        simulate_real_world=False   # evaluate on clean LR during validation
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


# ── Quick sanity-check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    root = sys.argv[1] if len(sys.argv) > 1 else "./freihand"

    # ── Raw projection diagnostic (before any cropping) ──────────────────────
    print("=== RAW PROJECTION DIAGNOSTIC ===")
    with open(os.path.join(root, "training_K.json")) as f:
        K = np.array(json.load(f)[0], dtype=np.float32)
    with open(os.path.join(root, "training_xyz.json")) as f:
        xyz = np.array(json.load(f)[0], dtype=np.float32)

    img0 = Image.open(os.path.join(root, "training", "rgb", "00000000.jpg"))
    img_w, img_h = img0.size

    print(f"Image size: {img_w} x {img_h}")
    print(f"K:\n{K}")
    print(f"xyz sample (first 3 joints):\n{xyz[:3]}")
    print(f"xyz Z range: {xyz[:,2].min():.4f} – {xyz[:,2].max():.4f}")

    uv_raw = np.matmul(K, xyz.T).T
    uv = uv_raw[:, :2] / uv_raw[:, 2:3]
    print(f"\nProjected UV (first 5 joints):\n{uv[:5]}")
    print(f"UV x range: {uv[:,0].min():.1f} – {uv[:,0].max():.1f}")
    print(f"UV y range: {uv[:,1].min():.1f} – {uv[:,1].max():.1f}")
    in_frame = ((uv[:,0] >= 0) & (uv[:,0] < img_w) &
                (uv[:,1] >= 0) & (uv[:,1] < img_h))
    print(f"Joints in frame: {in_frame.sum()} / 21")
    print("=== END DIAGNOSTIC ===")
    print()
    # ─────────────────────────────────────────────────────────────────────────

    ds = FreiHANDDataset(root, split="train", augment=False,
                          simulate_real_world=False)  # augment=False for clean diagnostic
    print(f"Dataset size: {len(ds)}")

    sample = ds[0]
    print("lr shape:      ", sample["lr"].shape)       # (3, 16, 16)
    print("hr shape:      ", sample["hr"].shape)       # (3, 64, 64)
    print("heatmaps shape:", sample["heatmaps"].shape) # (21, 64, 64)
    print("uv shape:      ", sample["uv"].shape)       # (21, 2)
    print("visible:       ", sample["visible"].sum().item(), "/ 21 keypoints")

    # Per-joint breakdown so we can see exactly which joints are dropping
    JOINT_NAMES = [
        "Wrist",
        "Index-MCP",  "Index-PIP",  "Index-DIP",  "Index-TIP",
        "Middle-MCP", "Middle-PIP", "Middle-DIP", "Middle-TIP",
        "Ring-MCP",   "Ring-PIP",   "Ring-DIP",   "Ring-TIP",
        "Pinky-MCP",  "Pinky-PIP",  "Pinky-DIP",  "Pinky-TIP",
        "Thumb-MCP",  "Thumb-IP",   "Thumb-DIP",  "Thumb-TIP",
    ]
    vis_np = sample["visible"].numpy()
    uv_np  = sample["uv"].numpy()
    print("\nPer-joint visibility:")
    for i, (name, v) in enumerate(zip(JOINT_NAMES, vis_np)):
        coord = f"({uv_np[i,0]:.1f}, {uv_np[i,1]:.1f})"
        status = "OK " if v else "OUTSIDE"
        print(f"  {i:2d}  {name:<12s}  {coord:<16s}  {status}")

    # Visualise first sample
    hr_np  = ((sample["hr"].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
    lr_np  = ((sample["lr"].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
    hm_sum = sample["heatmaps"].sum(0).numpy()  # collapsed heatmap

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(lr_np);  axes[0].set_title("LR input (16×16)")
    axes[1].imshow(hr_np);  axes[1].set_title("HR target (64×64)")
    axes[2].imshow(hr_np);
    axes[2].imshow(hm_sum, alpha=0.5, cmap="hot")
    axes[2].set_title("HR + GT heatmaps")

    uv = sample["uv"].numpy()
    vis = sample["visible"].numpy()
    for (p, c) in HAND_BONES:
        if vis[p] and vis[c]:
            axes[2].plot([uv[p, 0], uv[c, 0]], [uv[p, 1], uv[c, 1]],
                         "c-", linewidth=1)
    axes[2].scatter(uv[vis, 0], uv[vis, 1], c="lime", s=10, zorder=5)

    plt.tight_layout()
    plt.savefig("dataset_sanity_check.png", dpi=150)
    print("Saved dataset_sanity_check.png")
