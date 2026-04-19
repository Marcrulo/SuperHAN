# SuperHAN — Project Summary for Claude Code

## What this project is

A PyTorch reimplementation of **Super-FAN** adapted for hand gesture images from the **HaGRID** dataset. The system jointly trains a super-resolution generator and a hand landmark detector (FAN) using a GAN-based pipeline.

The pipeline takes a low-resolution (32×32) hand image as input and produces:
- A super-resolved (128×128) image
- 21 hand keypoint predictions (MediaPipe ordering)

---

## Directory structure

```
SuperHAN/
├── data/
│   ├── hagrid_dataset.py      # ← Main dataset loader (start here)
│   ├── augmentations.py       # Geometric + colour augmentations
│   └── constants.py           # NUM_KEYPOINTS, HR_SIZE, LR_SIZE, HEATMAP_SIZE, etc.
├── models/
│   ├── generator.py           # SRGenerator — LR→HR super-resolution network
│   ├── discriminator.py       # WGAN-GP discriminator
│   └── hourglass.py           # StackedHourglass — hand landmark FAN
├── losses/
│   └── losses.py              # SuperFANLoss, HeatmapLoss, WGANLoss
├── train/
│   └── trainer.py             # train_fan_standalone, train_sr_pretrain, train_super_fan
├── eval/
│   ├── metrics.py             # PCK, AUC, PSNR, SSIM, evaluate_all()
│   ├── mediapipe_eval.py      # MediaPipe baseline evaluation
│   └── visualize.py           # Visual comparison grid + PCK curve plots
└── train.py                   # Entry point — CLI for all three training stages
```

---

## Dataset (HaGRID)

Expected layout:
```
hagrid/
├── annotations/
│   ├── fist.json
│   ├── like.json
│   └── ...
├── fist/
│   └── <user_id>.jpg
├── like/
│   └── <user_id>.jpg
└── ...
```

Annotation format: `{ "<image_id>": { "hand_landmarks": [[[x,y], ...]], "bboxes": [...], "user_id": "..." } }`

Landmarks are normalised `[0,1]`, in MediaPipe ordering (21 joints, wrist=0).

### How a sample is loaded (`hagrid_dataset.py.__getitem__`)

1. Load full image from disk
2. Denormalise landmarks to pixel coords
3. Compute crop from **keypoint extents** + 20% margin (clamped to image bounds)
4. Crop that region, then `pad_to_square` — pads the shorter axis with black bars
5. Remap keypoints into the padded-square space, scale to `HR_SIZE=128`
6. Optionally augment (flip, rotate, colour jitter)
7. Resize to HR (128×128) and LR (32×32)
8. Generate 21-channel Gaussian heatmaps at `HEATMAP_SIZE=16`
9. Return `{ lr, hr, heatmaps, uv, visible }`

---

## Training stages

Run from the project root. Data path is typically `../hagrid`.

```bash
# Stage 1 — train FAN standalone on HR images
python train.py --stage fan --data ../hagrid --save_dir checkpoints

# Stage 2a — pretrain SR generator + FAN jointly (no GAN)
python train.py --stage sr --data ../hagrid --save_dir checkpoints \
                --fan_ckpt checkpoints/fan_standalone.pt

# Stage 2b — full Super-FAN with GAN loss
python train.py --stage superfan --data ../hagrid --save_dir checkpoints \
                --sr_ckpt checkpoints/sr/best.pt
```

Useful flags:
- `--max_samples 1000` — cap dataset size for quick debug runs
- `--gestures fist like ok` — restrict to specific gesture classes
- `--batch_size`, `--epochs_fan`, `--epochs_sr`, `--epochs_gan`

---

## Evaluation

```bash
python -m eval.visualize \
    --data ../hagrid \
    --ckpt checkpoints/superfan/best.pt \
    --fan_ckpt checkpoints/fan_standalone.pt \
    --out eval_output/ \
    --max_samples 200
```

Produces:
- `eval_output/pck_curve.png` — PCK curves for all 4 conditions
- `eval_output/visual_comparison.png` — side-by-side grid

---

## Sanity check

Always run this after dataset changes to verify crops and keypoints look correct:

```bash
python -m data.hagrid_dataset ../hagrid --n 16 --gestures fist like --out sanity.png
```

Each row shows: `LR | HR | HR + skeleton + heatmap overlay`

---

## Key constants (`data/constants.py`)

| Constant | Value | Meaning |
|---|---|---|
| `NUM_KEYPOINTS` | 21 | MediaPipe hand joints |
| `HR_SIZE` | 128 | HR image / FAN input size |
| `LR_SIZE` | 32 | LR image size (4× upscale) |
| `HEATMAP_SIZE` | 16 | FAN output heatmap size |
| `HEATMAP_SIGMA` | 1.5 | Gaussian sigma for GT heatmaps |

---

## Key design decisions

- **Crop strategy**: keypoint extents + 20% margin → pad to square. Avoids annotation bbox which can be misaligned.
- **No 3D projection**: HaGRID landmarks are already 2D and normalised, unlike FreiHAND which needed camera projection.
- **Joint ordering**: HaGRID uses MediaPipe ordering natively — no remapping needed for MediaPipe baseline.
- **GT heatmaps**: built directly from annotations, replacing the frozen teacher FAN from the original paper.
- **Progress bars**: all training and evaluation loops use `tqdm.write` for clean output alongside progress bars.