"""
metrics.py

Evaluation metrics for Super-FAN hands.

Metrics:
    PCK  — Percentage of Correct Keypoints at a given normalised threshold.
           Standard FreiHAND evaluation metric.
    AUC  — Area Under the PCK Curve, summarising performance across all
           thresholds. Matches the AUC metric used in the Super-FAN paper.
    PSNR — Peak Signal-to-Noise Ratio for SR quality (paper Table 1).
    SSIM — Structural Similarity Index for SR quality (paper Table 1).

PCK normalisation:
    FreiHAND normalises the threshold by the hand scale, defined as the
    distance from the wrist (joint 0) to the middle finger MCP (joint 9).
    This is the standard normalisation for hand pose evaluation, analogous
    to the inter-ocular distance used for face alignment.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict


# ── Constants ──────────────────────────────────────────────────────────────────

# Joint indices for scale normalisation
WRIST_IDX      = 0
MIDDLE_MCP_IDX = 9   # wrist → middle MCP = hand scale reference

# PCK thresholds to evaluate at (as fractions of hand scale)
PCK_THRESHOLDS = np.linspace(0, 1, 100)


# ── Keypoint metrics ───────────────────────────────────────────────────────────

def hand_scale(uv: np.ndarray) -> float:
    """
    Compute hand scale as wrist → middle-MCP distance.
    Used to normalise PCK thresholds.

    Args:
        uv: (21, 2) predicted or GT keypoints in pixel space.

    Returns:
        scale: scalar distance in pixels.
    """
    diff = uv[MIDDLE_MCP_IDX] - uv[WRIST_IDX]
    return float(np.linalg.norm(diff)) + 1e-6   # avoid div-by-zero


def pck_sample(
    pred: np.ndarray,
    gt:   np.ndarray,
    threshold: float,
    visibility: np.ndarray = None,
) -> float:
    """
    PCK for a single sample at a given normalised threshold.

    Args:
        pred:       (21, 2) predicted keypoints in pixel space.
        gt:         (21, 2) ground truth keypoints in pixel space.
        threshold:  Normalised distance threshold (fraction of hand scale).
        visibility: (21,) boolean mask. Only visible joints are counted.

    Returns:
        pck: fraction of correct keypoints in [0, 1].
    """
    scale = hand_scale(gt)
    dists = np.linalg.norm(pred - gt, axis=1)   # (21,)
    correct = dists < (threshold * scale)

    if visibility is not None:
        correct = correct[visibility]
        if visibility.sum() == 0:
            return 0.0

    return float(correct.mean())


def pck_curve(
    all_pred:       np.ndarray,
    all_gt:         np.ndarray,
    all_visibility: np.ndarray = None,
    thresholds:     np.ndarray = PCK_THRESHOLDS,
) -> np.ndarray:
    """
    Compute PCK at every threshold in `thresholds`.

    Args:
        all_pred:       (N, 21, 2) predicted keypoints.
        all_gt:         (N, 21, 2) ground truth keypoints.
        all_visibility: (N, 21) boolean visibility masks, or None.
        thresholds:     Array of normalised thresholds.

    Returns:
        pck_values: (len(thresholds),) array of PCK values.
    """
    N = len(all_pred)
    pck_values = np.zeros(len(thresholds))

    for t_idx, thresh in enumerate(thresholds):
        sample_pcks = []
        for i in range(N):
            vis = all_visibility[i] if all_visibility is not None else None
            sample_pcks.append(pck_sample(all_pred[i], all_gt[i], thresh, vis))
        pck_values[t_idx] = np.mean(sample_pcks)

    return pck_values


def auc(pck_values: np.ndarray) -> float:
    """
    Area Under the PCK Curve, normalised to [0, 1].
    Matches the AUC metric reported in the Super-FAN paper (Table 2).

    Args:
        pck_values: (T,) PCK values at evenly-spaced thresholds.

    Returns:
        auc_score: scalar in [0, 1].
    """
    return float(np.trapezoid(pck_values) / (len(pck_values) - 1))


def keypoints_from_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Extract 2D keypoint coordinates from predicted heatmaps via
    soft-argmax (differentiable) or hard argmax (for evaluation).

    Args:
        heatmaps: (B, 21, H, W) raw logits from the hourglass.

    Returns:
        uv: (B, 21, 2) keypoint coordinates in heatmap pixel space (x, y).
    """
    B, K, H, W = heatmaps.shape
    hm = torch.sigmoid(heatmaps)

    # Flatten spatial dims and find argmax per keypoint
    hm_flat = hm.view(B, K, -1)                      # (B, 21, H*W)
    idx = hm_flat.argmax(dim=-1)                      # (B, 21)

    # Convert flat index to (x, y) coordinates
    y = (idx // W).float()                            # (B, 21)
    x = (idx %  W).float()                            # (B, 21)

    return torch.stack([x, y], dim=-1)                # (B, 21, 2)


# ── Image quality metrics ──────────────────────────────────────────────────────

def psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio between SR and HR images.
    Higher is better. Typical SR values: 20–24 dB (matches paper Table 1).

    Args:
        sr: (B, 3, H, W) SR images in [-1, 1].
        hr: (B, 3, H, W) HR images in [-1, 1].

    Returns:
        psnr_db: mean PSNR in dB across the batch.
    """
    # Convert to [0, 1] for standard PSNR computation
    sr_01 = (sr.clamp(-1, 1) + 1) / 2
    hr_01 = (hr.clamp(-1, 1) + 1) / 2

    mse = F.mse_loss(sr_01, hr_01, reduction='none')
    mse = mse.view(sr.size(0), -1).mean(dim=1)        # (B,)
    psnr_vals = 10 * torch.log10(1.0 / (mse + 1e-8))
    return float(psnr_vals.mean().item())


def ssim(
    sr: torch.Tensor,
    hr: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """
    Structural Similarity Index between SR and HR images.
    Higher is better, max = 1.0 (matches paper Table 1).

    Implements the standard SSIM formula (Wang et al. 2004) used in the paper.

    Args:
        sr:          (B, 3, H, W) SR images in [-1, 1].
        hr:          (B, 3, H, W) HR images in [-1, 1].
        window_size: Gaussian window size (11 is standard).
        sigma:       Gaussian sigma (1.5 is standard).

    Returns:
        ssim_score: mean SSIM across the batch.
    """
    sr_01 = (sr.clamp(-1, 1) + 1) / 2
    hr_01 = (hr.clamp(-1, 1) + 1) / 2

    # Build Gaussian window
    window = _gaussian_window(window_size, sigma).to(sr.device)

    B, C = sr_01.shape[:2]
    ssim_vals = []

    for c in range(C):
        s = sr_01[:, c:c+1]
        h = hr_01[:, c:c+1]
        ssim_vals.append(_ssim_channel(s, h, window, window_size))

    return float(torch.stack(ssim_vals).mean().item())


def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)          # (size, size)
    return window.unsqueeze(0).unsqueeze(0)            # (1, 1, size, size)


def _ssim_channel(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window_size // 2

    mu_x = F.conv2d(x, window, padding=pad)
    mu_y = F.conv2d(y, window, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sigma_x2  = F.conv2d(x * x, window, padding=pad) - mu_x2
    sigma_y2  = F.conv2d(y * y, window, padding=pad) - mu_y2
    sigma_xy  = F.conv2d(x * y, window, padding=pad) - mu_xy

    num   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denom = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (num / denom).mean()


# ── Helper: accumulate UV predictions over a loader ──────────────────────────

@torch.no_grad()
def _collect_uvs(
    input_fn,       # callable: batch -> (B, 3, H, W) image to run FAN on
    fan,
    val_loader: DataLoader,
    device:     str,
    hm_size:    int,
) -> tuple:
    """
    Run FAN on images produced by input_fn and collect predicted + GT UVs.
    Returns (all_pred_uv, all_gt_uv, all_visibility) as numpy arrays.
    """
    all_pred, all_gt, all_vis = [], [], []
    for batch in val_loader:
        img    = input_fn(batch).to(device)
        gt_uv  = batch['uv'].numpy()
        vis    = batch['visible'].numpy()

        hm_list = fan(img)
        hm_up   = F.interpolate(hm_list[-1], size=(hm_size, hm_size),
                                mode='bilinear', align_corners=False)
        pred_uv = keypoints_from_heatmaps(hm_up).cpu().numpy()

        all_pred.append(pred_uv)
        all_gt.append(gt_uv)
        all_vis.append(vis)

    return (np.concatenate(all_pred, axis=0),
            np.concatenate(all_gt,   axis=0),
            np.concatenate(all_vis,  axis=0))


def _summarise(pred_uv, gt_uv, visibility) -> dict:
    """Compute PCK curve, AUC and spot values from collected UVs."""
    pck_vals  = pck_curve(pred_uv, gt_uv, visibility, PCK_THRESHOLDS)
    auc_score = auc(pck_vals)
    def pck_at(t):
        return float(pck_vals[np.argmin(np.abs(PCK_THRESHOLDS - t))])
    return {
        'pck_curve': pck_vals,
        'auc':       auc_score,
        'pck_at_02': pck_at(0.2),
        'pck_at_05': pck_at(0.5),
        'pck_at_10': pck_at(1.0),
    }


# ── Full four-condition evaluation ────────────────────────────────────────────

@torch.no_grad()
def evaluate_all(
    generator,
    fan,
    fan_standalone,
    val_loader:  DataLoader,
    device:      str = "cuda",
    hm_size:     int = 128,
) -> dict:
    """
    Run all four evaluation conditions and return results for each.

    Conditions:
        1. FAN on LR       — FAN applied directly to 32x32 LR input
                             (upsampled to 128x128 with bilinear before FAN).
                             Baseline: shows performance without any SR.

        2. FAN on HR       — FAN applied to GT 128x128 HR images.
                             Performance ceiling: best possible localisation.

        3. SR then FAN     — Generator super-resolves LR first, then a
                             separately-trained FAN (fan_standalone, trained
                             only on HR images) runs on the SR output.
                             Sequential pipeline — no joint training benefit.

        4. Super-FAN (ours)— Generator and FAN trained jointly.
                             The system's full output.

    Also computes PSNR/SSIM for the SR output (conditions 3 and 4 share
    the same generator so SR quality is reported once).

    Args:
        generator:      SRGenerator trained jointly (Stage 2).
        fan:            StackedHourglass trained jointly (Stage 2).
        fan_standalone: StackedHourglass trained only on HR images (Stage 1).
                        Used for condition 3 (sequential pipeline).
        val_loader:     DataLoader (simulate_real_world=False).
        device:         'cuda' or 'cpu'.
        hm_size:        Heatmap upsample size (128).

    Returns:
        dict with keys: 'lr', 'hr', 'sr_then_fan', 'super_fan', 'sr_quality'
        Each condition sub-dict has: 'pck_curve', 'auc', 'pck_at_02/05/10'
        'sr_quality' has: 'psnr', 'ssim'
    """
    from data.constants import LR_SIZE

    generator.eval()
    fan.eval()
    fan_standalone.eval()

    print("  Condition 1/4: FAN on LR images...")
    lr_upsample = lambda b: F.interpolate(
        b['lr'], size=(hm_size, hm_size), mode='bilinear', align_corners=False
    )
    pred_lr, gt_lr, vis_lr = _collect_uvs(
        lr_upsample, fan_standalone, val_loader, device, hm_size
    )

    print("  Condition 2/4: FAN on GT HR images...")
    pred_hr, gt_hr, vis_hr = _collect_uvs(
        lambda b: b['hr'], fan_standalone, val_loader, device, hm_size
    )

    print("  Condition 3/4: SR then FAN (sequential)...")
    def sr_image(b):
        return generator(b['lr'].to(device))
    pred_seq, gt_seq, vis_seq = _collect_uvs(
        sr_image, fan_standalone, val_loader, device, hm_size
    )

    print("  Condition 4/4: Super-FAN joint (ours)...")
    pred_joint, gt_joint, vis_joint = _collect_uvs(
        sr_image, fan, val_loader, device, hm_size
    )

    # SR image quality (same generator used in conditions 3 & 4)
    print("  Computing SR image quality metrics...")
    total_psnr, total_ssim, n = 0.0, 0.0, 0
    for batch in val_loader:
        sr = generator(batch['lr'].to(device))
        hr = batch['hr'].to(device)
        total_psnr += psnr(sr, hr)
        total_ssim += ssim(sr, hr)
        n += 1

    return {
        'lr':          _summarise(pred_lr,    gt_lr,    vis_lr),
        'hr':          _summarise(pred_hr,    gt_hr,    vis_hr),
        'sr_then_fan': _summarise(pred_seq,   gt_seq,   vis_seq),
        'super_fan':   _summarise(pred_joint, gt_joint, vis_joint),
        'sr_quality':  {'psnr': total_psnr / n, 'ssim': total_ssim / n},
    }


def print_results(results: dict):
    """Pretty-print all four evaluation conditions side by side."""
    conditions = [
        ('lr',          'FAN on LR          (baseline)'),
        ('sr_then_fan', 'SR then FAN        (sequential)'),
        ('super_fan',   'Super-FAN joint    (ours)'),
        ('hr',          'FAN on HR          (ceiling)'),
    ]
    w = 46
    print("\n" + "=" * w)
    print("  Super-FAN Hands — Evaluation Results")
    print("=" * w)
    print(f"  {'Condition':<28} {'AUC':>5}  {'@0.2':>5}  {'@0.5':>5}  {'@1.0':>5}")
    print("-" * w)
    for key, label in conditions:
        r = results[key]
        print(f"  {label:<28} "
              f"{r['auc']:.3f}  "
              f"{r['pck_at_02']:.3f}  "
              f"{r['pck_at_05']:.3f}  "
              f"{r['pck_at_10']:.3f}")
    print("-" * w)
    sr = results['sr_quality']
    print(f"  SR quality:  PSNR={sr['psnr']:.2f} dB   SSIM={sr['ssim']:.4f}")
    print("=" * w + "\n")
