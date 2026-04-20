"""
train.py

Entry point for Super-FAN hands training (HaGRID version).

Usage:
    # Stage 1: train Hand-FAN standalone (required before anything else)
    python train.py --stage fan --data hagrid --save_dir checkpoints

    # Stage 2a: pre-train SR generator + FAN jointly (no GAN)
    python train.py --stage sr --data hagrid --save_dir checkpoints \
                    --fan_ckpt checkpoints/fan_standalone.pt

    # Stage 2b: full Super-FAN fine-tuning with GAN
    python train.py --stage superfan --data hagrid --save_dir checkpoints \
                    --sr_ckpt checkpoints/sr/best.pt

    # Optional: restrict to specific gestures
    python train.py --stage fan --data hagrid --gestures fist like ok
"""

import argparse
import torch

from data.hagrid_dataset import build_dataloaders
from models.generator      import SRGenerator
from models.discriminator  import Discriminator
from models.hourglass      import StackedHourglass
from train.trainer         import (
    train_fan_standalone,
    train_sr_pretrain,
    train_super_fan,
)


def parse_args():
    p = argparse.ArgumentParser(description="Super-FAN Hands Training (HaGRID)")
    p.add_argument("--stage",    required=True,
                   choices=["fan", "sr", "superfan"],
                   help="Training stage to run")
    p.add_argument("--data",     default=None,
                   help="Path to HaGRID root directory (required unless --dry_run)")
    p.add_argument("--save_dir", default="checkpoints",
                   help="Directory for saving checkpoints")
    p.add_argument("--fan_ckpt", default=None,
                   help="Path to pretrained FAN checkpoint (for sr/superfan stages)")
    p.add_argument("--sr_ckpt",  default=None,
                   help="Path to pretrained SR checkpoint (for superfan stage)")
    p.add_argument("--gestures", nargs="*", default=None,
                   help="Gesture subset to use (default: all). E.g. --gestures fist like ok")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap dataset size per split (useful for quick debug runs)")
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--epochs_fan",   type=int, default=30)
    p.add_argument("--epochs_sr",    type=int, default=60)
    p.add_argument("--epochs_gan",   type=int,   default=5)
    p.add_argument("--lr_gan",       type=float, default=1e-4)
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_every",    type=int, default=50)
    p.add_argument("--dry_run", action="store_true",
                   help="Skip real data; run 2 synthetic batches per stage to verify shapes/losses")
    args = p.parse_args()
    if not args.dry_run and not args.data:
        p.error("--data is required unless --dry_run is set")
    return args


def main():
    args = parse_args()
    device = args.device
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.dry_run:
        print("[dry_run] Skipping dataset load — using synthetic tensors")
        train_loader, val_loader = None, None
    else:
        train_loader, val_loader = build_dataloaders(
            root=args.data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            simulate_real_world=(args.stage != "fan"),  # clean images for FAN warmup
            gestures=args.gestures,
            max_samples=args.max_samples,
        )
        print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ── Models ─────────────────────────────────────────────────────────────────
    generator     = SRGenerator()
    discriminator = Discriminator()
    fan           = StackedHourglass()

    def _load_fan(path):
        ckpt = torch.load(path, map_location=device)
        fan.load_state_dict(ckpt['model_state'])
        print(f"Loaded FAN from {path}")

    def _load_sr(path):
        ckpt = torch.load(path, map_location=device)
        generator.load_state_dict(ckpt['generator_state'])
        fan.load_state_dict(ckpt['fan_state'])
        print(f"Loaded SR+FAN from {path}")

    # ── Stage dispatch ─────────────────────────────────────────────────────────
    if args.stage == "fan":
        print("\n=== Stage 1: Hand-FAN standalone warmup ===")
        train_fan_standalone(
            fan=fan,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs_fan,
            device=device,
            save_dir=args.save_dir,
            log_every=args.log_every,
            dry_run=args.dry_run,
        )

    elif args.stage == "sr":
        print("\n=== Stage 2a: SR pre-training (pixel + perceptual + heatmap) ===")
        if not args.dry_run:
            assert args.fan_ckpt, "Must provide --fan_ckpt for sr stage"
            _load_fan(args.fan_ckpt)
        train_sr_pretrain(
            generator=generator,
            fan=fan,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs_sr,
            device=device,
            save_dir=args.save_dir,
            log_every=args.log_every,
            dry_run=args.dry_run,
        )

    elif args.stage == "superfan":
        print("\n=== Stage 2b: Full Super-FAN (GAN fine-tuning) ===")
        if not args.dry_run:
            assert args.sr_ckpt, "Must provide --sr_ckpt for superfan stage"
            _load_sr(args.sr_ckpt)
        train_super_fan(
            generator=generator,
            discriminator=discriminator,
            fan=fan,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs_gan,
            lr=args.lr_gan,
            device=device,
            save_dir=args.save_dir,
            log_every=args.log_every,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()