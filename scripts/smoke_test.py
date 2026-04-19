"""
scripts/smoke_test.py

Standalone shape/forward/backward sanity check. No dataset required.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --device cuda
"""

import sys
import argparse
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from data.constants   import LR_SIZE, HR_SIZE, HEATMAP_SIZE, NUM_KEYPOINTS
from models.generator     import SRGenerator
from models.discriminator import Discriminator
from models.hourglass     import StackedHourglass
from losses.losses        import SuperFANLoss, HeatmapLoss, WGANLoss


def _check(name: str, got, expected):
    ok = got == expected
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: got {got}, expected {expected}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch",  type=int, default=2)
    args = p.parse_args()

    device = args.device
    B = args.batch
    print(f"\nSmoke test  device={device}  batch={B}")
    print(f"Constants:  LR={LR_SIZE}  HR={HR_SIZE}  HEATMAP={HEATMAP_SIZE}  KP={NUM_KEYPOINTS}\n")

    passed = []

    # ── Synthetic inputs ──────────────────────────────────────────────────────
    lr_img  = torch.randn(B, 3, LR_SIZE,      LR_SIZE,      device=device)
    hr_img  = torch.randn(B, 3, HR_SIZE,       HR_SIZE,      device=device)
    gt_hm   = torch.randn(B, NUM_KEYPOINTS, HEATMAP_SIZE, HEATMAP_SIZE, device=device)

    # ── Generator ─────────────────────────────────────────────────────────────
    print("── Generator ──────────────────────────────────────────")
    gen = SRGenerator().to(device)
    sr  = gen(lr_img)
    passed.append(_check("output shape", tuple(sr.shape),
                          (B, 3, HR_SIZE, HR_SIZE)))

    # ── Discriminator ─────────────────────────────────────────────────────────
    print("── Discriminator ──────────────────────────────────────")
    disc      = Discriminator().to(device)
    real_out  = disc(hr_img)
    fake_out  = disc(sr.detach())
    passed.append(_check("real output ndim", real_out.ndim, 1))
    passed.append(_check("fake output ndim", fake_out.ndim, 1))
    passed.append(_check("real output len",  real_out.shape[0], B))

    # ── Hourglass FAN ─────────────────────────────────────────────────────────
    print("── StackedHourglass (FAN) ─────────────────────────────")
    fan      = StackedHourglass().to(device)
    hm_preds = fan(hr_img)
    passed.append(_check("output is list", isinstance(hm_preds, list), True))
    passed.append(_check("final heatmap shape", tuple(hm_preds[-1].shape),
                          (B, NUM_KEYPOINTS, HEATMAP_SIZE, HEATMAP_SIZE)))

    # ── HeatmapLoss ───────────────────────────────────────────────────────────
    print("── HeatmapLoss ────────────────────────────────────────")
    hm_loss_fn = HeatmapLoss(warmup_steps=0).to(device)
    hm_loss    = hm_loss_fn(hm_preds, gt_hm)
    passed.append(_check("heatmap loss scalar", hm_loss.shape, torch.Size([])))
    hm_loss.backward()
    passed.append(_check("heatmap loss backward", True, True))

    # ── SuperFANLoss (no GAN) ─────────────────────────────────────────────────
    print("── SuperFANLoss (no adversarial) ──────────────────────")
    sr2      = gen(lr_img)
    hm_preds2 = fan(sr2)
    loss_fn  = SuperFANLoss(use_adversarial=False, hm_warmup=0).to(device)
    losses   = loss_fn(sr2, hr_img, hm_preds2, gt_hm)
    passed.append(_check("total loss present", "total" in losses, True))
    passed.append(_check("total loss scalar",  losses["total"].shape, torch.Size([])))
    losses["total"].backward()
    passed.append(_check("total loss backward", True, True))

    # ── SuperFANLoss (with GAN) ───────────────────────────────────────────────
    print("── SuperFANLoss (with adversarial) ────────────────────")
    sr3       = gen(lr_img)
    hm_preds3 = fan(sr3)
    fake_sc   = disc(sr3)
    loss_fn2  = SuperFANLoss(use_adversarial=True, hm_warmup=0).to(device)
    losses2   = loss_fn2(sr3, hr_img, hm_preds3, gt_hm, fake_sc)
    passed.append(_check("adversarial loss present", "adversarial" in losses2, True))
    losses2["total"].backward()
    passed.append(_check("adversarial backward", True, True))

    # ── WGANLoss discriminator step ───────────────────────────────────────────
    print("── WGANLoss (discriminator step) ──────────────────────")
    wgan     = WGANLoss()
    sr_d     = gen(lr_img).detach()
    d_loss   = wgan.full_discriminator_loss(disc, hr_img, sr_d)
    passed.append(_check("d_loss scalar", d_loss.shape, torch.Size([])))
    d_loss.backward()
    passed.append(_check("d_loss backward", True, True))

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass = sum(passed)
    n_fail = len(passed) - n_pass
    print(f"\n{'='*54}")
    print(f"  {n_pass}/{len(passed)} checks passed", end="")
    if n_fail:
        print(f"  ← {n_fail} FAILED")
        sys.exit(1)
    else:
        print("  — all good")


if __name__ == "__main__":
    main()
