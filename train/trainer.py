"""
trainer.py

Two-stage training procedure for Super-FAN hands (Section 4.5 of the paper).

Stage 1 — FAN warmup (standalone, no SR):
    Train Hand-FAN alone on HR images with GT heatmaps until convergence.
    This gives the FAN the warm start that the paper obtained for free
    from a pretrained facial FAN.

Stage 2a — SR pre-training (no GAN):
    Train generator with pixel + perceptual + heatmap losses.
    FAN is loaded from Stage 1 and trained jointly with the generator.
    Run for 60 epochs with LR decayed from 2.5e-4 → 1e-5.

Stage 2b — GAN fine-tuning (full Super-FAN):
    Introduce WGAN-GP discriminator.  Fine-tune all three networks jointly
    for 5 epochs with LR=2.5e-4, generator:discriminator ratio 1:1.
"""

import os
import csv
import time
import torch
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')  # non-interactive backend — safe for training loops
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader

from models.generator     import SRGenerator
from models.discriminator import Discriminator
from models.hourglass     import StackedHourglass
from losses.losses        import SuperFANLoss, HeatmapLoss, WGANLoss


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_dry_run_batch(batch_size: int = 2, device: str = "cpu") -> dict:
    """Synthetic batch matching dataset output shapes, for --dry_run mode."""
    return {
        'lr':       torch.randn(batch_size, 3,  32,  32,  device=device),
        'hr':       torch.randn(batch_size, 3,  128, 128, device=device),
        'heatmaps': torch.randn(batch_size, 21, 16,  16,  device=device),
        'uv':       torch.rand( batch_size, 21, 2,        device=device),
        'visible':  torch.ones( batch_size, 21,           device=device),
    }

def _cosine_lr(optimizer, base_lr: float, min_lr: float, total_steps: int):
    """Cosine annealing from base_lr to min_lr over total_steps."""
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=min_lr
    )


def _log(step: int, losses: dict, prefix: str = ""):
    parts = [f"{prefix}step={step}"]
    parts += [f"{k}={v.item():.5f}" for k, v in losses.items() if torch.is_tensor(v)]
    tqdm.write("  ".join(parts))


def _save(path: str, **state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    tqdm.write(f"Saved checkpoint → {path}")


def _latest_checkpoint(ckpt_dir: str, prefix: str = "epoch_") -> str | None:
    """Return the path to the highest-numbered epoch_NNNN.pt in ckpt_dir, or None."""
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = sorted(
        f for f in os.listdir(ckpt_dir)
        if f.startswith(prefix) and f.endswith(".pt")
    )
    return os.path.join(ckpt_dir, candidates[-1]) if candidates else None


class _CSVLogger:
    """
    Appends one row per epoch to <save_dir>/training_log.csv.
    Columns: timestamp, stage, epoch, train_loss, val_loss, [extra cols...]
    Safe to resume — appends to existing file.
    """
    def __init__(self, save_dir: str, stage: str, extra_cols: list = None):
        os.makedirs(save_dir, exist_ok=True)
        self.path  = os.path.join(save_dir, "training_log.csv")
        self.stage = stage
        self.cols  = ["timestamp", "stage", "epoch",
                      "train_loss", "val_loss"] + (extra_cols or [])
        # Write header only if file is new
        write_header = not os.path.exists(self.path)
        self._f = open(self.path, "a", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.cols,
                                 extrasaction="ignore")
        if write_header:
            self._w.writeheader()

    def log(self, epoch: int, train_loss: float, val_loss: float, **extra):
        row = dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                   stage=self.stage, epoch=epoch,
                   train_loss=round(train_loss, 6),
                   val_loss=round(val_loss, 6),
                   **{k: round(v, 6) if isinstance(v, float) else v
                      for k, v in extra.items()})
        self._w.writerow(row)
        self._f.flush()

    def close(self):
        self._f.close()


# ── Plot helper ───────────────────────────────────────────────────────────────

def _plot_logs(save_dir: str):
    """
    Read training_log.csv and save one PNG per stage showing train/val loss,
    plus a combined overview PNG covering all stages.
    Called at the end of each stage so plots are always up to date.
    """
    csv_path = os.path.join(save_dir, "training_log.csv")
    if not os.path.exists(csv_path):
        return

    # Parse CSV
    import csv as _csv

    def _is_float(s):
        try: float(s); return True
        except: return False

    rows = []
    with open(csv_path) as f:
        reader = _csv.DictReader(f)
        for row in reader:
            rows.append({k: (float(v) if v not in ('', 'nan') and _is_float(v)
                             else v)
                         for k, v in row.items()})

    if not rows:
        return

    stages     = ["fan", "sr", "superfan"]
    stage_cols = {
        "fan":      ["train_loss", "val_loss"],
        "sr":       ["train_loss", "val_loss", "pixel", "perceptual", "heatmap"],
        "superfan": ["train_loss", "val_loss", "pixel", "perceptual",
                     "heatmap", "adversarial", "d_loss"],
    }
    colors = {
        "train_loss":  "#ffffff",
        "val_loss":    "#a8ff78",
        "pixel":       "#60efff",
        "perceptual":  "#ff6b6b",
        "heatmap":     "#ffd166",
        "adversarial": "#c77dff",
        "d_loss":      "#ff9a3c",
    }
    stage_labels = {
        "fan":      "Stage 1 — FAN warmup",
        "sr":       "Stage 2a — SR pre-training",
        "superfan": "Stage 2b — Super-FAN (GAN)",
    }

    BG    = "#0a0a14"
    GRID  = "#1a1a2e"
    TICK  = "#555555"

    def _style_ax(ax):
        ax.set_facecolor(BG)
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.tick_params(colors=TICK, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    # ── Per-stage PNGs ──────────────────────────────────────────────────────
    for stage in stages:
        stage_rows = [r for r in rows if r.get("stage") == stage]
        if not stage_rows:
            continue

        cols = [c for c in stage_cols[stage]
                if any(isinstance(r.get(c), float) for r in stage_rows)]
        epochs = [r["epoch"] for r in stage_rows]

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor(BG)
        _style_ax(ax)

        for col in cols:
            vals = [r.get(col) for r in stage_rows]
            if any(isinstance(v, float) for v in vals):
                ax.plot(epochs, vals, label=col, color=colors.get(col, "#aaa"),
                        linewidth=1.5 if "loss" in col else 1.0,
                        linestyle="-" if col == "train_loss" else
                                  "--" if col == "val_loss" else ":",
                        alpha=0.9)

        ax.set_title(stage_labels[stage], color="#fff", fontsize=11, pad=10)
        ax.set_xlabel("Epoch", color=TICK, fontsize=9)
        ax.set_ylabel("Loss", color=TICK, fontsize=9)
        leg = ax.legend(fontsize=8, facecolor="#111", labelcolor="#ccc",
                        edgecolor=GRID)

        plt.tight_layout()
        out = os.path.join(save_dir, f"loss_{stage}.png")
        plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        tqdm.write(f"  Saved plot → {out}")

    # ── Combined overview PNG ───────────────────────────────────────────────
    present_stages = [s for s in stages
                      if any(r.get("stage") == s for r in rows)]
    if len(present_stages) < 2:
        return   # not worth a combined plot yet

    fig, axes = plt.subplots(1, len(present_stages),
                             figsize=(6 * len(present_stages), 4),
                             sharey=False)
    fig.patch.set_facecolor(BG)
    if len(present_stages) == 1:
        axes = [axes]

    for ax, stage in zip(axes, present_stages):
        stage_rows = [r for r in rows if r.get("stage") == stage]
        epochs = [r["epoch"] for r in stage_rows]
        _style_ax(ax)
        for col in ["train_loss", "val_loss"]:
            vals = [r.get(col) for r in stage_rows]
            if any(isinstance(v, float) for v in vals):
                ax.plot(epochs, vals, label=col,
                        color=colors[col], linewidth=1.5,
                        linestyle="-" if col == "train_loss" else "--")
        ax.set_title(stage_labels[stage], color="#fff", fontsize=10, pad=8)
        ax.set_xlabel("Epoch", color=TICK, fontsize=8)
        ax.set_ylabel("Loss", color=TICK, fontsize=8)
        ax.legend(fontsize=8, facecolor="#111", labelcolor="#ccc", edgecolor=GRID)

    fig.suptitle("Super-FAN Training Overview", color="#fff", fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(save_dir, "loss_overview.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    tqdm.write(f"  Saved plot → {out}")


# ── Stage 1: standalone FAN warmup ────────────────────────────────────────────

def train_fan_standalone(
    fan:         StackedHourglass,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:       int   = 30,
    lr:           float = 2.5e-4,
    device:       str   = "cuda",
    save_dir:     str   = "checkpoints",
    log_every:    int   = 50,
    dry_run:      bool  = False,
) -> StackedHourglass:
    """
    Train the Hand-FAN in isolation on HR images with GT heatmaps.

    This stage has no equivalent in the original paper (they used a
    pretrained facial FAN). It is necessary here to give the FAN a
    meaningful initialisation before it is coupled to the SR network.

    Args:
        fan:          StackedHourglass model.
        train_loader: DataLoader yielding dicts with 'hr' and 'heatmaps'.
        val_loader:   Validation DataLoader.
        epochs:       Training epochs (30 is usually sufficient).
        lr:           Initial learning rate.
        device:       'cuda' or 'cpu'.
        save_dir:     Directory for checkpoints.
        log_every:    Log interval in steps.

    Returns:
        Trained FAN (also saved to save_dir/fan_standalone.pt).
    """
    fan = fan.to(device)
    fan.train()

    heatmap_loss_fn = HeatmapLoss(warmup_steps=0)  # no warmup in standalone
    heatmap_loss_fn = heatmap_loss_fn.to(device)

    optimizer = optim.RMSprop(fan.parameters(), lr=lr)
    total_steps = 2 if dry_run else epochs * len(train_loader)
    scheduler = _cosine_lr(optimizer, lr, 1e-5, total_steps)

    global_step = 0
    best_val_loss = float('inf')
    logger = _CSVLogger(save_dir, stage="fan")

    epoch_range = range(1) if dry_run else range(epochs)
    epoch_bar = tqdm(epoch_range, desc="[FAN warmup]", unit="epoch")
    for epoch in epoch_bar:
        fan.train()
        train_total = 0.0
        batches = [_make_dry_run_batch(device=device)] * 2 if dry_run else train_loader
        batch_bar = tqdm(batches, desc=f"  epoch {epoch+1}/{epochs}",
                         unit="batch", leave=False)
        for batch in batch_bar:
            hr  = batch['hr'].to(device)
            gt  = batch['heatmaps'].to(device)

            if dry_run:
                tqdm.write(f"[dry_run] hr={tuple(hr.shape)}  gt_hm={tuple(gt.shape)}")

            pred = fan(hr)

            if dry_run:
                tqdm.write(f"[dry_run] pred[-1]={tuple(pred[-1].shape)}")

            loss = heatmap_loss_fn(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_total += loss.item()
            global_step += 1
            batch_bar.set_postfix(loss=f"{loss.item():.5f}")
            if global_step % log_every == 0:
                _log(global_step, {'heatmap': loss}, prefix="[FAN warmup] ")

        # ── Validation ────────────────────────────────────────────────────────
        n = max(len(batches), 1)
        train_loss = train_total / n
        val_batches = [_make_dry_run_batch(device=device)] if dry_run else val_loader
        val_loss   = _validate_fan(fan, val_batches, heatmap_loss_fn, device)
        epoch_bar.set_postfix(train=f"{train_loss:.5f}", val=f"{val_loss:.5f}")
        tqdm.write(f"[FAN warmup] epoch={epoch+1}/{epochs}  "
              f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")
        logger.log(epoch+1, train_loss, val_loss)

        if not dry_run and val_loss < best_val_loss:
            best_val_loss = val_loss
            _save(os.path.join(save_dir, "fan_standalone.pt"),
                  model_state=fan.state_dict(),
                  epoch=epoch, val_loss=val_loss)

    logger.close()
    if not dry_run:
        _plot_logs(save_dir)
    return fan


def _validate_fan(fan, loader, loss_fn, device):
    fan.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            hr  = batch['hr'].to(device)
            gt  = batch['heatmaps'].to(device)
            pred = fan(hr)
            total += loss_fn(pred, gt).item()
    fan.train()
    return total / len(loader)


# ── Stage 2a: SR pre-training (pixel + perceptual + heatmap) ──────────────────

def train_sr_pretrain(
    generator:    SRGenerator,
    fan:          StackedHourglass,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:       int   = 60,
    lr:           float = 2.5e-4,
    device:       str   = "cuda",
    save_dir:     str   = "checkpoints",
    log_every:    int   = 50,
    hm_warmup:    int   = 5000,
    dry_run:      bool  = False,
) -> tuple:
    """
    Train SR generator jointly with FAN using pixel + perceptual + heatmap
    losses (no GAN yet).  Corresponds to the paper's 60-epoch pre-training.

    Args:
        generator:    SRGenerator.
        fan:          StackedHourglass (loaded from Stage 1).
        train_loader: DataLoader.
        val_loader:   Validation DataLoader.
        epochs:       Training epochs (60 as in the paper).
        lr:           Initial learning rate (2.5e-4 → 1e-5 via cosine decay).
        device:       'cuda' or 'cpu'.
        save_dir:     Checkpoint directory.
        log_every:    Log interval in steps.
        hm_warmup:    Steps to ramp heatmap loss weight 0→1.

    Returns:
        (generator, fan) trained models.
    """
    ckpt_dir = os.path.join(save_dir, "sr")
    os.makedirs(ckpt_dir, exist_ok=True)

    generator = generator.to(device)
    fan       = fan.to(device)

    loss_fn = SuperFANLoss(
        use_adversarial=False,
        hm_warmup=hm_warmup,
    ).to(device)

    params = list(generator.parameters()) + list(fan.parameters())
    optimizer = optim.RMSprop(params, lr=lr)
    total_steps = 2 if dry_run else epochs * len(train_loader)
    scheduler = _cosine_lr(optimizer, lr, 1e-5, total_steps)

    # ── Resume from latest per-epoch checkpoint if available ──────────────────
    start_epoch   = 0
    global_step   = 0
    best_val_loss = float('inf')
    latest = _latest_checkpoint(ckpt_dir, prefix="epoch_")
    if latest:
        tqdm.write(f"[SR pretrain] Resuming from {latest}")
        ckpt = torch.load(latest, map_location=device)
        generator.load_state_dict(ckpt['generator_state'])
        fan.load_state_dict(ckpt['fan_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch   = ckpt['epoch'] + 1
        global_step   = ckpt['global_step']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        tqdm.write(f"[SR pretrain] Resumed at epoch {start_epoch}, step {global_step}")

    logger = _CSVLogger(save_dir, stage="sr",
                        extra_cols=["pixel", "perceptual", "heatmap"])

    epoch_range = range(1) if dry_run else range(start_epoch, epochs)
    epoch_bar = tqdm(epoch_range, desc="[SR pretrain]", unit="epoch")
    for epoch in epoch_bar:
        generator.train(); fan.train(); loss_fn.train()

        train_totals = {"total": 0.0, "pixel": 0.0,
                        "perceptual": 0.0, "heatmap": 0.0}
        batches = [_make_dry_run_batch(device=device)] * 2 if dry_run else train_loader
        batch_bar = tqdm(batches, desc=f"  epoch {epoch+1}/{epochs}",
                         unit="batch", leave=False)
        for batch in batch_bar:
            lr_img = batch['lr'].to(device)
            hr_img = batch['hr'].to(device)
            gt_hm  = batch['heatmaps'].to(device)

            sr      = generator(lr_img)
            pred_hm = fan(sr)

            if dry_run:
                tqdm.write(f"[dry_run] lr={tuple(lr_img.shape)}  sr={tuple(sr.shape)}  "
                           f"hr={tuple(hr_img.shape)}  pred_hm[-1]={tuple(pred_hm[-1].shape)}")

            losses  = loss_fn(sr, hr_img, pred_hm, gt_hm)

            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            scheduler.step()

            for k in train_totals:
                if k in losses:
                    train_totals[k] += losses[k].item()
            global_step += 1
            batch_bar.set_postfix(loss=f"{losses['total'].item():.5f}")
            if global_step % log_every == 0:
                _log(global_step, losses, prefix="[SR pretrain] ")

        # ── Validation ────────────────────────────────────────────────────────
        n = max(len(batches), 1)
        train_loss = train_totals["total"] / n
        val_batches = [_make_dry_run_batch(device=device)] if dry_run else val_loader
        val_loss   = _validate_sr(generator, fan, val_batches, loss_fn, device)
        epoch_bar.set_postfix(train=f"{train_loss:.5f}", val=f"{val_loss:.5f}")
        tqdm.write(f"[SR pretrain] epoch={epoch+1}/{epochs}  "
              f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")
        logger.log(epoch+1, train_loss, val_loss,
                   pixel=train_totals["pixel"]/n,
                   perceptual=train_totals["perceptual"]/n,
                   heatmap=train_totals["heatmap"]/n)

        if not dry_run:
            # Per-epoch checkpoint (always)
            _save(os.path.join(ckpt_dir, f"epoch_{epoch+1:04d}.pt"),
                  generator_state=generator.state_dict(),
                  fan_state=fan.state_dict(),
                  optimizer_state=optimizer.state_dict(),
                  scheduler_state=scheduler.state_dict(),
                  epoch=epoch, global_step=global_step,
                  best_val_loss=best_val_loss, val_loss=val_loss)

            # Best checkpoint (separate file)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save(os.path.join(ckpt_dir, "best.pt"),
                      generator_state=generator.state_dict(),
                      fan_state=fan.state_dict(),
                      optimizer_state=optimizer.state_dict(),
                      scheduler_state=scheduler.state_dict(),
                      epoch=epoch, global_step=global_step,
                      best_val_loss=best_val_loss, val_loss=val_loss)

    logger.close()
    if not dry_run:
        _plot_logs(save_dir)
    return generator, fan


def _validate_sr(generator, fan, loader, loss_fn, device):
    generator.eval(); fan.eval(); loss_fn.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            lr_img = batch['lr'].to(device)
            hr_img = batch['hr'].to(device)
            gt_hm  = batch['heatmaps'].to(device)
            sr     = generator(lr_img)
            pred   = fan(sr)
            total += loss_fn(sr, hr_img, pred, gt_hm)['total'].item()
    generator.train(); fan.train(); loss_fn.train()
    return total / len(loader)


# ── Stage 2b: GAN fine-tuning (full Super-FAN) ────────────────────────────────

def train_super_fan(
    generator:     SRGenerator,
    discriminator: Discriminator,
    fan:           StackedHourglass,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    epochs:        int   = 5,
    lr:            float = 2.5e-4,
    device:        str   = "cuda",
    save_dir:      str   = "checkpoints",
    log_every:     int   = 50,
    d_warmup_steps: int  = 100,
    dry_run:       bool  = False,
) -> tuple:
    """
    Full Super-FAN joint training with GAN loss (Stage 2b).

    Loads pre-trained generator and FAN from Stage 2a, introduces the
    WGAN-GP discriminator, and fine-tunes all three networks jointly
    for 5 epochs.  Generator:discriminator update ratio is 1:1.

    Args:
        generator:      SRGenerator (loaded from Stage 2a).
        discriminator:  Discriminator (randomly initialised).
        fan:            StackedHourglass (loaded from Stage 2a).
        train_loader:   DataLoader.
        val_loader:     Validation DataLoader.
        epochs:         Fine-tuning epochs (5 as in the paper).
        lr:             Learning rate (2.5e-4 as in the paper, no decay).
        device:         'cuda' or 'cpu'.
        save_dir:       Checkpoint directory.
        log_every:      Log interval in steps.
        d_warmup_steps: Train discriminator alone for this many steps before
                        allowing adversarial gradients to reach the generator.
                        Gives the critic time to become meaningful before the
                        generator tries to fool it. Default 100.

    Returns:
        (generator, discriminator, fan) trained models.
    """
    ckpt_dir = os.path.join(save_dir, "superfan")
    os.makedirs(ckpt_dir, exist_ok=True)

    generator     = generator.to(device)
    discriminator = discriminator.to(device)
    fan           = fan.to(device)

    loss_fn  = SuperFANLoss(use_adversarial=True, hm_warmup=0).to(device)
    wgan_fn  = WGANLoss()

    g_optimizer = optim.RMSprop(
        list(generator.parameters()) + list(fan.parameters()), lr=lr
    )
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=lr)

    # ── Resume from latest per-epoch checkpoint if available ──────────────────
    start_epoch   = 0
    global_step   = 0
    best_val_loss = float('inf')
    latest = _latest_checkpoint(ckpt_dir, prefix="epoch_")
    if latest:
        tqdm.write(f"[Super-FAN] Resuming from {latest}")
        ckpt = torch.load(latest, map_location=device)
        generator.load_state_dict(ckpt['generator_state'])
        discriminator.load_state_dict(ckpt['discriminator_state'])
        fan.load_state_dict(ckpt['fan_state'])
        g_optimizer.load_state_dict(ckpt['g_optimizer_state'])
        d_optimizer.load_state_dict(ckpt['d_optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        tqdm.write(f"[Super-FAN] Resumed at epoch {start_epoch}, step {global_step}")

    logger = _CSVLogger(save_dir, stage="superfan",
                        extra_cols=["pixel", "perceptual", "heatmap",
                                    "adversarial", "d_loss"])

    epoch_range = range(1) if dry_run else range(start_epoch, epochs)
    epoch_bar = tqdm(epoch_range, desc="[Super-FAN]", unit="epoch")
    for epoch in epoch_bar:
        generator.train(); discriminator.train(); fan.train()

        train_totals = {"total": 0.0, "pixel": 0.0, "perceptual": 0.0,
                        "heatmap": 0.0, "adversarial": 0.0, "d_loss": 0.0}
        batches = [_make_dry_run_batch(device=device)] * 2 if dry_run else train_loader
        batch_bar = tqdm(batches, desc=f"  epoch {epoch+1}/{epochs}",
                         unit="batch", leave=False)
        for batch in batch_bar:
            lr_img = batch['lr'].to(device)
            hr_img = batch['hr'].to(device)
            gt_hm  = batch['heatmaps'].to(device)

            # ── Discriminator update ──────────────────────────────────────────
            sr = generator(lr_img).detach()
            d_loss = wgan_fn.full_discriminator_loss(discriminator, hr_img, sr)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ── Generator + FAN update ────────────────────────────────────────
            # Skip adversarial gradients until discriminator is warmed up
            sr          = generator(lr_img)
            pred_hm     = fan(sr)
            fake_scores = discriminator(sr) if global_step >= d_warmup_steps else None

            if dry_run:
                tqdm.write(f"[dry_run] lr={tuple(lr_img.shape)}  sr={tuple(sr.shape)}  "
                           f"disc_out={tuple(discriminator(sr.detach()).shape)}")

            losses = loss_fn(sr, hr_img, pred_hm, gt_hm, fake_scores)
            g_optimizer.zero_grad()
            losses['total'].backward()
            g_optimizer.step()

            for k in train_totals:
                if k in losses:
                    train_totals[k] += losses[k].item()
            train_totals["d_loss"] += d_loss.item()
            global_step += 1
            batch_bar.set_postfix(g=f"{losses['total'].item():.5f}",
                                  d=f"{d_loss.item():.5f}")
            if global_step % log_every == 0:
                _log(global_step,
                     {**losses, 'd_loss': d_loss},
                     prefix="[Super-FAN] ")

        # ── Validation ────────────────────────────────────────────────────────
        n = max(len(batches), 1)
        train_loss = train_totals["total"] / n
        val_batches = [_make_dry_run_batch(device=device)] if dry_run else val_loader
        val_loss   = _validate_sr(generator, fan, val_batches, loss_fn, device)
        epoch_bar.set_postfix(train=f"{train_loss:.5f}", val=f"{val_loss:.5f}")
        tqdm.write(f"[Super-FAN] epoch={epoch+1}/{epochs}  "
              f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")
        logger.log(epoch+1, train_loss, val_loss,
                   pixel=train_totals["pixel"]/n,
                   perceptual=train_totals["perceptual"]/n,
                   heatmap=train_totals["heatmap"]/n,
                   adversarial=train_totals["adversarial"]/n,
                   d_loss=train_totals["d_loss"]/n)

        if not dry_run:
            # Per-epoch checkpoint
            _save(os.path.join(ckpt_dir, f"epoch_{epoch+1:04d}.pt"),
                  generator_state=generator.state_dict(),
                  discriminator_state=discriminator.state_dict(),
                  fan_state=fan.state_dict(),
                  g_optimizer_state=g_optimizer.state_dict(),
                  d_optimizer_state=d_optimizer.state_dict(),
                  epoch=epoch, global_step=global_step,
                  best_val_loss=best_val_loss, val_loss=val_loss)

            # Best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save(os.path.join(ckpt_dir, "best.pt"),
                      generator_state=generator.state_dict(),
                      discriminator_state=discriminator.state_dict(),
                      fan_state=fan.state_dict(),
                      g_optimizer_state=g_optimizer.state_dict(),
                      d_optimizer_state=d_optimizer.state_dict(),
                      epoch=epoch, global_step=global_step, val_loss=val_loss)

    logger.close()
    if not dry_run:
        _plot_logs(save_dir)
    return generator, discriminator, fan