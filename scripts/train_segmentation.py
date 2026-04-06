"""
Training script for the Building Segmentation model (Project 3).

Usage:
    python scripts/train_segmentation.py --data-dir data/sample --epochs 50
    python scripts/train_segmentation.py --data-dir data/raw/xbd --epochs 100 --batch-size 8
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.ingestion import XBDDataset
from src.pipeline.preprocessing import XBDTransform
from src.models.segmentation import (
    BuildingSegmentationModel,
    BceDiceLoss,
    build_segmentation_optimizer,
    build_lr_scheduler,
)
from src.utils.metrics import SegmentationMetrics
from src.utils.viz_utils import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train building segmentation model")
    parser.add_argument("--data-dir",      default="data/sample",           help="Root dir of xBD-format dataset")
    parser.add_argument("--output-dir",    default="checkpoints/segmentation")
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch-size",    type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--weight-decay",  type=float, default=1e-4)
    parser.add_argument("--image-size",    type=int,   default=512)
    parser.add_argument("--encoder",       default="resnet34")
    parser.add_argument("--val-split",     type=float, default=0.2)
    parser.add_argument("--num-workers",   type=int,   default=4)
    parser.add_argument("--resume",        default=None,   help="Path to checkpoint to resume from")
    parser.add_argument("--device",        default=None,   help="cuda / cpu (auto-detect if not set)")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device, metrics):
    model.train()
    metrics.reset()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  Train", leave=False):
        pre_img  = batch["pre_image"].to(device)
        bld_mask = batch["building_mask"].float().to(device)
        if bld_mask.dim() == 3:
            bld_mask = bld_mask.unsqueeze(1)   # (B, 1, H, W)

        optimizer.zero_grad()
        logits = model(pre_img)
        loss   = criterion(logits, bld_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        metrics.update(logits.detach().cpu(), bld_mask.detach().cpu())

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, metrics):
    model.eval()
    metrics.reset()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        pre_img  = batch["pre_image"].to(device)
        bld_mask = batch["building_mask"].float().to(device)
        if bld_mask.dim() == 3:
            bld_mask = bld_mask.unsqueeze(1)

        logits = model(pre_img)
        loss   = criterion(logits, bld_mask)

        total_loss += loss.item()
        metrics.update(logits.detach().cpu(), bld_mask.detach().cpu())

    return total_loss / len(loader)


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------ Data
    train_transform = XBDTransform(phase="train", image_size=args.image_size)
    val_transform   = XBDTransform(phase="val",   image_size=args.image_size)

    full_dataset = XBDDataset(
        root_dir=args.data_dir,
        split="train",
        task="segmentation",
        transform=train_transform,
    )

    val_size   = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # On macOS with MPS, num_workers > 0 can cause issues with forked processes
    use_workers = 0 if device.type == "mps" else args.num_workers
    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=use_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=use_workers, pin_memory=pin
    )

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ---------------------------------------------------------------- Model
    model = BuildingSegmentationModel(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
    ).to(device)
    logger.info(f"Model parameters: {model.n_parameters:,}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        logger.info(f"Resumed from: {args.resume}")

    # ---------------------------------------------------------------- Train
    criterion = BceDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = build_segmentation_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_lr_scheduler(optimizer, epochs=args.epochs, warmup_epochs=5)

    train_metrics = SegmentationMetrics()
    val_metrics   = SegmentationMetrics()

    best_val_iou   = 0.0
    train_loss_log = []
    val_loss_log   = []
    train_met_log  = []
    val_met_log    = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}  LR={scheduler.get_last_lr()[0]:.2e}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, train_metrics)
        val_loss   = validate(model, val_loader, criterion, device, val_metrics)

        scheduler.step()

        t_met = train_metrics.log(prefix=f"E{epoch} Train")
        v_met = val_metrics.log(  prefix=f"E{epoch} Val  ")

        train_loss_log.append(train_loss)
        val_loss_log.append(val_loss)
        train_met_log.append(t_met)
        val_met_log.append(v_met)

        # Save best checkpoint (move to CPU to avoid MPS save issues)
        if v_met["iou"] > best_val_iou:
            best_val_iou = v_met["iou"]
            ckpt_path = str(Path(args.output_dir) / "best.pth")
            cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            opt_state = optimizer.state_dict()
            torch.save({
                "epoch":            epoch,
                "model_state_dict": cpu_state,
                "optimizer_state_dict": opt_state,
                "val_iou":          best_val_iou,
                "args":             vars(args),
            }, ckpt_path)
            logger.info(f"  *** Saved best checkpoint: IoU={best_val_iou:.4f} → {ckpt_path}")

        # Periodic checkpoint
        if epoch % 10 == 0:
            periodic = str(Path(args.output_dir) / f"epoch_{epoch:03d}.pth")
            cpu_state_p = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({"epoch": epoch, "model_state_dict": cpu_state_p}, periodic)

    # Save training curves
    plot_training_curves(
        train_loss_log, val_loss_log,
        train_met_log,  val_met_log,
        metric_name="iou",
        save_path=str(Path(args.output_dir) / "training_curves.png"),
    )

    logger.info(f"\nTraining complete. Best Val IoU: {best_val_iou:.4f}")
    logger.info(f"Best checkpoint: {Path(args.output_dir) / 'best.pth'}")


if __name__ == "__main__":
    main()
