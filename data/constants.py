"""
constants.py
Shared constants for the Super-FAN hands data pipeline.
Centralised here to avoid circular imports between freihand_dataset and augmentations.
"""

NUM_KEYPOINTS  = 21
HR_SIZE        = 128   # high-resolution target side length (pixels)
LR_SIZE        = 32    # low-resolution input side length (pixels)
HEATMAP_SIZE   = 16    # heatmap spatial resolution — matches FAN native output
                       # (stem does 3× stride-2: 128 → 64 → 32 → 16)
                       # GT heatmaps must be at this size so HeatmapLoss
                       # compares tensors at the same resolution without
                       # a destructive 128→16 downsample blurring the targets.
HEATMAP_SIGMA  = 1.5   # Gaussian sigma in pixels at HEATMAP_SIZE=16
                       # sigma=1.5 gives ~5x5 visible peak — enough gradient
                       # signal without being too spatially loose.

# FreiHAND bone connectivity: (parent, child) index pairs for 20 bones
# Joint ordering: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
HAND_BONES = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),    # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),    # index finger
    (0, 9),  (9, 10), (10, 11),(11, 12),  # middle finger
    (0, 13),(13, 14),(14, 15),(15, 16),   # ring finger
    (0, 17),(17, 18),(18, 19),(19, 20),   # pinky finger
]
