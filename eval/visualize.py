"""
visualize.py

Visual evaluation for Super-FAN hands.

Produces a grid of side-by-side comparisons:
    LR input (16x16) | SR output (64x64) | HR ground truth (64x64)

With skeleton overlaid on both SR and HR panels using predicted and GT
keypoints respectively, so misalignments are immediately visible.

Usage:
    python eval/visualize.py \
        --data freihand \
        --ckpt checkpoints/super_fan_epoch5.pt \
        --n_samples 16 \
        --out eval_output/
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from data.constants        import HAND_BONES, HR_SIZE, LR_SIZE
from eval.metrics          import keypoints_from_heatmaps


# ── Skeleton drawing ───────────────────────────────────────────────────────────

# Colour each finger differently for easy reading
FINGER_COLOURS = {
    "index":  "#4FC3F7",   # light blue
    "middle": "#81C784",   # green
    "ring":   "#FFB74D",   # orange
    "pinky":  "#F06292",   # pink
    "thumb":  "#CE93D8",   # purple
    "wrist":  "#FFFFFF",   # white
}

# Map bone index → finger name based on HAND_BONES ordering
# HAND_BONES: 0-3 thumb, 4-7 index, 8-11 middle, 12-15 ring, 16-19 pinky
BONE_COLOURS = (
    [FINGER_COLOURS["thumb"]]  * 4 +
    [FINGER_COLOURS["index"]]  * 4 +
    [FINGER_COLOURS["middle"]] * 4 +
    [FINGER_COLOURS["ring"]]   * 4 +
    [FINGER_COLOURS["pinky"]]  * 4
)


def draw_skeleton(
    ax,
    uv:         np.ndarray,
    visibility: np.ndarray = None,
    alpha:      float = 0.9,
    kp_size:    float = 8.0,
):
    """
    Draw hand skeleton on a matplotlib axis.

    Args:
        ax:         Matplotlib axis.
        uv:         (21, 2) keypoint coordinates (x, y) in pixel space.
        visibility: (21,) boolean mask. Hidden joints are skipped.
        alpha:      Drawing opacity.
        kp_size:    Keypoint scatter size.
    """
    if visibility is None:
        visibility = np.ones(21, dtype=bool)

    for bone_idx, (p, c) in enumerate(HAND_BONES):
        if visibility[p] and visibility[c]:
            colour = BONE_COLOURS[bone_idx] if bone_idx < len(BONE_COLOURS) else "white"
            ax.plot(
                [uv[p, 0], uv[c, 0]],
                [uv[p, 1], uv[c, 1]],
                color=colour, linewidth=1.5, alpha=alpha,
            )

    # Draw visible keypoints on top
    ax.scatter(
        uv[visibility, 0], uv[visibility, 1],
        c="white", s=kp_size, zorder=5, alpha=alpha, linewidths=0.5,
        edgecolors="black",
    )



def _tensor_to_np(t):
    t = (t.cpu().clamp(-1, 1) + 1) / 2
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _predict_uvs(fan, images, hm_size):
    hm_list = fan(images)
    hm_final = hm_list[-1]                              # (B, 21, 16, 16)
    # Confidence: max activation per keypoint before upsampling
    conf = torch.sigmoid(hm_final).flatten(2).max(dim=-1).values  # (B, 21)
    hm_up = F.interpolate(hm_final, size=(hm_size, hm_size),
                          mode='bilinear', align_corners=False)
    uv = keypoints_from_heatmaps(hm_up).cpu().numpy()
    return uv, conf.cpu().numpy()


@torch.no_grad()
def visualize_all_conditions(
    generator, fan_joint, fan_standalone, batch,
    device="cuda", save_path=None, mp_predictor=None,
):
    """
    5- or 6-column figure per sample row:
        1. FAN on LR       (baseline)
        2. SR then FAN     (sequential)
        3. Super-FAN joint (ours)
        4. FAN on HR       (ceiling)
        5. MediaPipe on HR (optional — passed via mp_predictor)
        6. Ground Truth

    Predicted skeletons drawn in finger colours throughout.
    """
    generator.eval(); fan_joint.eval(); fan_standalone.eval()

    lr_img = batch['lr'].to(device)
    hr_img = batch['hr'].to(device)
    gt_uv  = batch['uv'].numpy()
    vis    = batch['visible'].numpy()
    B      = lr_img.size(0)
    HM     = HR_SIZE

    sr    = generator(lr_img)
    lr_up = F.interpolate(lr_img, size=(HM, HM), mode='bilinear', align_corners=False)

    pred_lr,    conf_lr    = _predict_uvs(fan_standalone, lr_up,  HM)
    pred_seq,   conf_seq   = _predict_uvs(fan_standalone, sr,     HM)
    pred_joint, conf_joint = _predict_uvs(fan_joint,      sr,     HM)
    pred_hr,    conf_hr    = _predict_uvs(fan_standalone, hr_img, HM)

    # Visibility from predicted confidence (threshold 0.1) — not GT mask.
    # This shows which joints each method actually detected, making differences
    # between conditions visible rather than forcing all to use GT visibility.
    CONF_THRESH = 0.1
    vis_lr    = conf_lr    > CONF_THRESH
    vis_seq   = conf_seq   > CONF_THRESH
    vis_joint = conf_joint > CONF_THRESH
    vis_hr    = conf_hr    > CONF_THRESH

    # Optional MediaPipe columns (LR, SR, HR)
    pred_mp_lr = pred_mp_sr = pred_mp_hr = None
    if mp_predictor is not None:
        from eval.mediapipe_eval import _tensor_to_uint8
        lr_up_np = np.stack([_tensor_to_uint8(lr_up[i]) for i in range(B)])
        sr_np    = np.stack([_tensor_to_uint8(sr[i])    for i in range(B)])
        hr_np    = np.stack([_tensor_to_uint8(hr_img[i]) for i in range(B)])
        pred_mp_lr, _ = mp_predictor.predict_batch(lr_up_np)
        pred_mp_sr, _ = mp_predictor.predict_batch(sr_np)
        pred_mp_hr, _ = mp_predictor.predict_batch(hr_np)

    col_labels = [
        "FAN on LR\n(baseline)",
        "SR then FAN\n(sequential)",
        "Super-FAN joint\n(ours)",
        "FAN on HR\n(ceiling)",
    ]
    if pred_mp_lr is not None:
        col_labels += ["MediaPipe on LR", "MediaPipe on SR", "MediaPipe on HR"]
    col_labels.append("Ground Truth")

    NCOLS = len(col_labels)
    fig, axes = plt.subplots(B, NCOLS, figsize=(NCOLS * 2.8, B * 3.0))
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle("Per-condition visualisation", color="white", fontsize=13, y=1.01)
    if B == 1:
        axes = axes[np.newaxis, :]

    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, color="#aaaaaa", fontsize=8, pad=4)

    for i in range(B):
        # Build image and UV lists dynamically
        imgs_np  = [
            _tensor_to_np(lr_up[i]),   # FAN on LR
            _tensor_to_np(sr[i]),      # SR then FAN
            _tensor_to_np(sr[i]),      # Super-FAN joint
            _tensor_to_np(hr_img[i]),  # FAN on HR
        ]
        pred_uvs  = [pred_lr[i], pred_seq[i], pred_joint[i], pred_hr[i]]
        pred_viss = [vis_lr[i],  vis_seq[i],  vis_joint[i],  vis_hr[i]]

        if pred_mp_lr is not None:
            imgs_np   += [_tensor_to_np(lr_up[i]),
                          _tensor_to_np(sr[i]),
                          _tensor_to_np(hr_img[i])]
            pred_uvs  += [pred_mp_lr[i], pred_mp_sr[i], pred_mp_hr[i]]
            # MediaPipe always returns all 21 joints; use all-true mask
            pred_viss += [np.ones(21, dtype=bool)] * 3

        # GT column always last — use actual GT visibility
        imgs_np.append(_tensor_to_np(hr_img[i]))
        pred_uvs.append(None)
        pred_viss.append(vis[i])   # GT visibility for GT column

        for col in range(NCOLS):
            ax = axes[i, col]
            ax.imshow(imgs_np[col])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")

            if pred_uvs[col] is not None:
                draw_skeleton(ax, pred_uvs[col], visibility=pred_viss[col])
            else:
                draw_skeleton(ax, gt_uv[i], visibility=pred_viss[col],
                              alpha=1.0, kp_size=10.0)

    plt.tight_layout(pad=0.4)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved -> {save_path}")
    return fig


@torch.no_grad()
def visualize_batch(generator, fan, batch, device="cuda", save_path=None,
                    title="Super-FAN Hands"):
    """Quick 3-column check: LR | SR+pred | HR+GT."""
    generator.eval(); fan.eval()
    lr_img = batch['lr'].to(device); hr_img = batch['hr'].to(device)
    gt_uv  = batch['uv'].numpy();    vis    = batch['visible'].numpy()
    B      = lr_img.size(0)
    sr     = generator(lr_img)
    hm_up  = F.interpolate(fan(sr)[-1], size=(HR_SIZE, HR_SIZE),
                            mode='bilinear', align_corners=False)
    pred_uv = keypoints_from_heatmaps(hm_up).cpu().numpy()

    fig, axes = plt.subplots(B, 3, figsize=(9, B * 3.2))
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle(title, color="white", fontsize=13, y=1.01)
    if B == 1:
        axes = axes[np.newaxis, :]
    for col, ct in enumerate(["LR input", "SR output + pred", "HR target + GT"]):
        axes[0, col].set_title(ct, color="#aaaaaa", fontsize=9, pad=4)
    for i in range(B):
        lr_np = _tensor_to_np(F.interpolate(
            lr_img[i:i+1], size=(HR_SIZE, HR_SIZE), mode='nearest')[0])
        for col, img_np in enumerate([lr_np, _tensor_to_np(sr[i]),
                                      _tensor_to_np(hr_img[i])]):
            axes[i, col].imshow(img_np)
            axes[i, col].set_xticks([]); axes[i, col].set_yticks([])
            for spine in axes[i, col].spines.values():
                spine.set_edgecolor("#444444")
        draw_skeleton(axes[i, 1], pred_uv[i], visibility=vis[i])
        draw_skeleton(axes[i, 2], gt_uv[i],   visibility=vis[i])
    plt.tight_layout(pad=0.5)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved -> {save_path}")
    return fig



# ── PCK curve plot ─────────────────────────────────────────────────────────────

def plot_pck_curve(
    all_results:  dict,
    thresholds:   np.ndarray,
    mp_results:   dict = None,
    save_path:    str  = None,
) -> plt.Figure:
    """
    Plot all PCK curves on one axes for direct comparison.

    Args:
        all_results: dict from evaluate_all() with keys
                     'lr', 'sr_then_fan', 'super_fan', 'hr'.
        thresholds:  (T,) normalised threshold values.
        mp_results:  Optional dict from evaluate_mediapipe() with keys
                     'mp_on_lr', 'mp_on_hr'. If provided, two extra curves
                     are added for the MediaPipe baselines.
        save_path:   If given, save figure to this path.
    """
    conditions = [
        ('lr',          'FAN on LR (baseline)',     '#EF5350', '--'),
        ('sr_then_fan', 'SR then FAN (sequential)', '#FFB74D', '-.'),
        ('super_fan',   'Super-FAN joint (ours)',    '#4FC3F7', '-'),
        ('hr',          'FAN on HR (ceiling)',       '#81C784', ':'),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    for key, label, colour, ls in conditions:
        r = all_results[key]
        ax.plot(thresholds, r['pck_curve'] * 100,
                color=colour, linewidth=2, linestyle=ls,
                label=f"{label}  (AUC={r['auc']:.3f})")

    # MediaPipe baselines (optional)
    if mp_results is not None:
        mp_conditions = [
            ('mp_on_lr', 'MediaPipe on LR', '#546E7A'),
            ('mp_on_sr', 'MediaPipe on SR', '#90A4AE'),
            ('mp_on_hr', 'MediaPipe on HR', '#CFD8DC'),
        ]
        for key, label, colour in mp_conditions:
            if key not in mp_results:
                continue
            r   = mp_results[key]
            det = r.get('detection_rate', 1.0)
            ax.plot(thresholds, r['pck_curve'] * 100,
                    color=colour, linewidth=1.5, dashes=(4, 2, 1, 2),
                    label=f"{label}  (AUC={r['auc']:.3f}, det={det*100:.0f}%)")

    # Shade gain from baseline to our method
    ax.fill_between(
        thresholds,
        all_results['lr']['pck_curve'] * 100,
        all_results['super_fan']['pck_curve'] * 100,
        alpha=0.08, color="#4FC3F7",
    )

    ax.set_xlabel("Normalised threshold (fraction of hand scale)",
                  color="#aaaaaa", fontsize=10)
    ax.set_ylabel("PCK (%)", color="#aaaaaa", fontsize=10)
    ax.set_title("PCK Curve — Hand Keypoint Localisation",
                 color="white", fontsize=13)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.legend(facecolor="#2a2a2a", labelcolor="white", fontsize=9,
              loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.grid(True, color="#333333", linewidth=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved -> {save_path}")

    return fig


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from data.freihand_dataset import build_dataloaders
    from models.generator      import SRGenerator
    from models.hourglass      import StackedHourglass
    from eval.metrics          import evaluate_all, print_results, PCK_THRESHOLDS
    from eval.visualize        import visualize_all_conditions, plot_pck_curve

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True,
                        help="FreiHAND root dir")
    parser.add_argument("--ckpt",       required=True,
                        help="Super-FAN checkpoint (super_fan_epochN.pt)")
    parser.add_argument("--fan_ckpt",   required=True,
                        help="Standalone FAN checkpoint (fan_standalone.pt) "
                             "used for the sequential SR-then-FAN condition")
    parser.add_argument("--n_samples",  type=int, default=16,
                        help="Samples to visualise")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out",        default="eval_output",
                        help="Output directory for saved figures")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available()
                                                       else "cpu")
    parser.add_argument("--real_world", action="store_true",
                        help="Evaluate on simulated real-world LR images "
                             "(Gaussian blur + JPEG artefacts + colour distortion) "
                             "instead of clean bicubic downsampling")
    parser.add_argument("--no-mediapipe", dest="mediapipe",
                        action="store_false",
                        help="Skip MediaPipe baseline evaluation")
    parser.set_defaults(mediapipe=True)
    args = parser.parse_args()
    device = args.device

    # ── Load joint Super-FAN (generator + jointly-trained FAN) ───────────────
    generator = SRGenerator().to(device)
    fan_joint = StackedHourglass().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    generator.load_state_dict(ckpt['generator_state'])
    fan_joint.load_state_dict(ckpt['fan_state'])
    print(f"Loaded Super-FAN checkpoint from {args.ckpt}")

    # ── Load standalone FAN (used for conditions 1, 2, and 3) ────────────────
    fan_standalone = StackedHourglass().to(device)
    fan_ckpt = torch.load(args.fan_ckpt, map_location=device)
    fan_standalone.load_state_dict(fan_ckpt['model_state'])
    print(f"Loaded standalone FAN from {args.fan_ckpt}")

    # ── Data ──────────────────────────────────────────────────────────────────
    _, val_loader = build_dataloaders(
        args.data, batch_size=args.batch_size,
        simulate_real_world=False
    )

    # ── Quantitative evaluation: all 4 conditions ─────────────────────────────
    print("\nRunning evaluation across all 4 conditions...")
    results = evaluate_all(
        generator, fan_joint, fan_standalone,
        val_loader, device=device
    )
    print_results(results)

    # ── MediaPipe baseline (optional) ────────────────────────────────────────
    mp_results   = None
    mp_predictor = None
    if args.mediapipe:
        from eval.mediapipe_eval import (
            evaluate_mediapipe, print_mediapipe_results, MediaPipePredictor
        )
        print("\nRunning MediaPipe baseline evaluation...")
        mp_results = evaluate_mediapipe(
            generator, val_loader, device=device,
        )
        print_mediapipe_results(mp_results)
        # Keep predictor alive for the visual comparison column
        mp_predictor = MediaPipePredictor()

    # ── PCK curve: all conditions on one plot ─────────────────────────────────
    plot_pck_curve(
        results, PCK_THRESHOLDS,
        mp_results=mp_results,
        save_path=os.path.join(args.out, "pck_curve.png"),
    )

    # ── Visual comparison grid ────────────────────────────────────────────────
    print("Generating visual comparisons...")
    batch = next(iter(val_loader))
    batch = {k: v[:args.n_samples] for k, v in batch.items()}
    visualize_all_conditions(
        generator, fan_joint, fan_standalone, batch, device=device,
        mp_predictor=mp_predictor,
        save_path=os.path.join(args.out, "visual_comparison.png"),
    )
    if mp_predictor is not None:
        mp_predictor.close()
    print(f"\nAll outputs saved to {args.out}/")
