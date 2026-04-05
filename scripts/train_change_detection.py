"""
Training script for the Damage Classification model (Project 5).

Takes the 6-channel (pre+post) input and classifies damage level per pixel.
Best run after the segmentation model is trained (the seg checkpoint provides
building masks that constrain and improve damage predictions).

Usage:
    python scripts/train_change_detection.py --data-dir data/sample --epochs 60
    python scripts/train_change_detection.py --data-dir data/raw/xbd --batch-size 4 \
        --seg-checkpoint checkpoints/segmentation/best.pth
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.ingestion import XBDDataset
from src.pipeline.preprocessing import XBDTransform
from src.models.change_detection import (
    DamageClassificationModel,
    CombinedDamageLoss,
    build_damage_optimizer,
    NUM_DAMAGE_CLASSES,
)
from src.models.segmentation import BuildingSegmentationModel, build_lr_scheduler
from src.utils.metrics import DamageMetrics
from src.utils.viz_utils import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train damage classification model")
    parser.add_argument("--data-dir",         default="data/sample")
    parser.add_argument("--output-dir",        default="checkpoints/damage")
    parser.add_argument("--epochs",            type=int,   default=60)
    parser.add_argument("--batch-size",        type=int,   default=4)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--weight-decay",      type=float, default=1e-4)
    parser.add_argument("--image-size",        type=int,   default=512)
    parser.add_argument("--encoder",           default="resnet34")
    parser.add_argument("--val-split",         type=float, default=0.2)
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--resume",            default=None)
    parser.add_argument("--seg-checkpoint",    default=None,
                        help="Segmentation checkpoint to use for building-aware training")
    parser.add_argument("--device",            default=None)
    # Class weights to handle imbalance (background, no-dmg, minor, major, destroyed)
    parser.add_argument("--class-weights",     type=float, nargs=5,
                        default=[0.1, 1.0, 2.0, 3.0, 4.0])
    return parser.parse_args()


def train_one_epoch(
    model, seg_model, loader, optimizer, criterion, device, metrics
):
    model.train()
    metrics.reset()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  Train", leave=False):
        pre_img  = batch["pre_image"].to(device)
        post_img = batch["post_image"].to(device)
        dmg_mask = batch["damage_mask"].long().to(device)

        # If building-aware: zero out non-building pixels in target
        if seg_model is not None:
            with torch.no_grad():
                seg_logits    = seg_model(pre_img)
                building_mask = (torch.sigmoid(seg_logits).squeeze(1) > 0.5).long()
            dmg_mask = dmg_mask * building_mask

        pair_input = torch.cat([pre_img, post_img], dim=1)   # (B, 6, H, W)

        optimizer.zero_grad()
        logits = model(pair_input)
        loss   = criterion(logits, dmg_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        metrics.update(logits, dmg_mask)

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, seg_model, loader, criterion, device, metrics):
    model.eval()
    metrics.reset()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        pre_img  = batch["pre_image"].to(device)
        post_img = batch["post_image"].to(device)
        dmg_mask = batch["damage_mask"].long().to(device)

        if seg_model is not None:
            seg_logits    = seg_model(pre_img)
            building_mask = (torch.sigmoid(seg_logits).squeeze(1) > 0.5).long()
            dmg_mask      = dmg_mask * building_mask

        pair_input = torch.cat([pre_img, post_img], dim=1)
        logits     = model(pair_input)
        loss       = criterion(logits, dmg_mask)

        total_loss += loss.item()
        metrics.update(logits, dmg_mask)

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
        task="damage",
        transform=train_transform,
    )

    val_size   = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    use_workers = 0 if device.type == "mps" else args.num_workers
    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True,  num_workers=use_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds,   batch_size=args.batch_size,
        shuffle=False, num_workers=use_workers, pin_memory=pin
    )
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # --------------------------------------------------------- Optional seg model
    seg_model = None
    if args.seg_checkpoint:
        seg_model = BuildingSegmentationModel().to(device).eval()
        ckpt = torch.load(args.seg_checkpoint, map_location=device)
        seg_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        logger.info(f"Loaded seg model from: {args.seg_checkpoint}")
        for p in seg_model.parameters():
            p.requires_grad = False

    # --------------------------------------------------------- Damage model
    model = DamageClassificationModel(
        encoder_name=args.encoder,
        in_channels=6,
        num_classes=NUM_DAMAGE_CLASSES,
    ).to(device)
    logger.info(f"Model parameters: {model.n_parameters:,}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        logger.info(f"Resumed from: {args.resume}")

    # ----------------------------------------------------------------- Train
    criterion = CombinedDamageLoss(class_weights=args.class_weights)
    optimizer = build_damage_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_lr_scheduler(optimizer, epochs=args.epochs, warmup_epochs=5)

    train_metrics = DamageMetrics(num_classes=NUM_DAMAGE_CLASSES)
    val_metrics   = DamageMetrics(num_classes=NUM_DAMAGE_CLASSES)

    best_val_score = 0.0
    train_loss_log = []
    val_loss_log   = []
    train_met_log  = []
    val_met_log    = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}  LR={scheduler.get_last_lr()[0]:.2e}")

        train_loss = train_one_epoch(
            model, seg_model, train_loader, optimizer, criterion, device, train_metrics
        )
        val_loss = validate(
            model, seg_model, val_loader, criterion, device, val_metrics
        )
        scheduler.step()

        t_met = train_metrics.log(prefix=f"E{epoch} Train")
        v_met = val_metrics.log(  prefix=f"E{epoch} Val  ")

        train_loss_log.append(train_loss)
        val_loss_log.append(val_loss)
        train_met_log.append({"xview2_score": t_met.get("xview2_score", 0)})
        val_met_log.append(  {"xview2_score": v_met.get("xview2_score", 0)})

        score = v_met.get("xview2_score", 0)
        if score > best_val_score:
            best_val_score = score
            ckpt_path = str(Path(args.output_dir) / "best.pth")
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "xview2_score":     best_val_score,
                "args":             vars(args),
            }, ckpt_path)
            logger.info(f"  *** Saved best: xView2={best_val_score:.4f} → {ckpt_path}")

        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                str(Path(args.output_dir) / f"epoch_{epoch:03d}.pth"),
            )

    plot_training_curves(
        train_loss_log, val_loss_log,
        train_met_log,  val_met_log,
        metric_name="xview2_score",
        save_path=str(Path(args.output_dir) / "training_curves.png"),
    )

    logger.info(f"\nTraining complete. Best xView2 score: {best_val_score:.4f}")


if __name__ == "__main__":
    main()
