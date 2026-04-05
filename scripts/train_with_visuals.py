"""
Visual Training Script — see what the model learns in real-time.

This script saves prediction images every N epochs so you can
open the output folder and WATCH the model improve.

Run from VSCode terminal:
    python3 scripts/train_with_visuals.py --data-dir data/raw/xbd --epochs 30

Then open data/outputs/training_progress/ in VSCode or Finder
to see images updating as training progresses.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from loguru import logger
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.ingestion import XBDDataset
from src.pipeline.preprocessing import XBDTransform
from src.models.segmentation import (
    BuildingSegmentationModel,
    BceDiceLoss,
    build_segmentation_optimizer,
    build_lr_scheduler,
)
from src.models.change_detection import (
    DamageClassificationModel,
    CombinedDamageLoss,
    build_damage_optimizer,
    NUM_DAMAGE_CLASSES,
)
from src.utils.metrics import SegmentationMetrics, DamageMetrics


# Damage class colors (RGB 0-1)
DAMAGE_COLORS = {
    0: (0.0, 0.0, 0.0),      # background — black
    1: (0.0, 0.78, 0.0),     # no-damage  — green
    2: (1.0, 1.0, 0.0),      # minor      — yellow
    3: (1.0, 0.55, 0.0),     # major      — orange
    4: (0.86, 0.0, 0.0),     # destroyed  — red
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    """Convert normalized tensor back to viewable image."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def colorize_mask(mask, n_classes=5):
    """Convert class index mask to RGB image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3))
    for cls_idx, color in DAMAGE_COLORS.items():
        if cls_idx < n_classes:
            rgb[mask == cls_idx] = color
    return rgb


def save_segmentation_preview(
    model, dataset, device, epoch, output_dir, n_samples=4
):
    """
    Save a visual grid showing:
      [Satellite Image] [Ground Truth Mask] [Model Prediction] [Overlay]

    This is what you look at to understand what the model is learning.
    """
    model.eval()
    fig = plt.figure(figsize=(16, 4 * n_samples))
    fig.suptitle(
        f"Building Segmentation — Epoch {epoch}",
        fontsize=16, fontweight="bold", y=1.02
    )

    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)

    for row, idx in enumerate(indices):
        sample = dataset[int(idx)]
        pre_img   = sample["pre_image"]
        bld_mask  = sample["building_mask"]

        # Model prediction
        with torch.no_grad():
            x = pre_img.unsqueeze(0).to(device)
            logits = model(x)
            pred = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Denormalize image for display
        img_display = denormalize(pre_img)
        gt_mask     = bld_mask.numpy().astype(float)
        pred_binary = (pred > 0.5).astype(float)

        # Overlay: green = correct, red = false positive, blue = missed
        overlay = img_display.copy()
        tp = (pred_binary > 0.5) & (gt_mask > 0.5)
        fp = (pred_binary > 0.5) & (gt_mask < 0.5)
        fn = (pred_binary < 0.5) & (gt_mask > 0.5)
        overlay[tp] = overlay[tp] * 0.5 + np.array([0, 0.8, 0]) * 0.5   # green = correct
        overlay[fp] = overlay[fp] * 0.5 + np.array([0.9, 0, 0]) * 0.5   # red = false positive
        overlay[fn] = overlay[fn] * 0.5 + np.array([0, 0, 0.9]) * 0.5   # blue = missed

        # Plot
        ax1 = fig.add_subplot(n_samples, 4, row * 4 + 1)
        ax1.imshow(img_display)
        ax1.set_title("Satellite Image" if row == 0 else "")
        ax1.axis("off")

        ax2 = fig.add_subplot(n_samples, 4, row * 4 + 2)
        ax2.imshow(gt_mask, cmap="Greens", vmin=0, vmax=1)
        ax2.set_title("Ground Truth" if row == 0 else "")
        ax2.axis("off")

        ax3 = fig.add_subplot(n_samples, 4, row * 4 + 3)
        ax3.imshow(pred, cmap="Greens", vmin=0, vmax=1)
        ax3.set_title("Model Prediction" if row == 0 else "")
        ax3.axis("off")

        ax4 = fig.add_subplot(n_samples, 4, row * 4 + 4)
        ax4.imshow(overlay)
        ax4.set_title("Overlay (green=✓ red=FP blue=miss)" if row == 0 else "")
        ax4.axis("off")

    plt.tight_layout()
    path = Path(output_dir) / f"seg_epoch_{epoch:03d}.png"
    fig.savefig(str(path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved preview: {path}")


def save_damage_preview(
    seg_model, dmg_model, dataset, device, epoch, output_dir, n_samples=4
):
    """
    Save a visual grid showing:
      [Pre Image] [Post Image] [GT Damage] [Predicted Damage] [Buildings]
    """
    seg_model.eval()
    dmg_model.eval()
    fig = plt.figure(figsize=(20, 4 * n_samples))
    fig.suptitle(
        f"Damage Classification — Epoch {epoch}",
        fontsize=16, fontweight="bold", y=1.02
    )

    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)

    for row, idx in enumerate(indices):
        sample = dataset[int(idx)]
        pre_img  = sample["pre_image"]
        post_img = sample["post_image"]
        gt_dmg   = sample["damage_mask"].numpy()
        gt_bld   = sample["building_mask"].numpy()

        with torch.no_grad():
            pre_t  = pre_img.unsqueeze(0).to(device)
            post_t = post_img.unsqueeze(0).to(device)

            # Building prediction
            seg_logits = seg_model(pre_t)
            seg_pred   = (torch.sigmoid(seg_logits).squeeze().cpu().numpy() > 0.5)

            # Damage prediction
            pair_t = torch.cat([pre_t, post_t], dim=1)
            dmg_logits = dmg_model(pair_t)
            dmg_pred   = torch.argmax(dmg_logits, dim=1).squeeze().cpu().numpy()
            dmg_pred[~seg_pred] = 0  # mask with buildings

        pre_display  = denormalize(pre_img)
        post_display = denormalize(post_img)
        gt_rgb       = colorize_mask(gt_dmg)
        pred_rgb     = colorize_mask(dmg_pred)

        panels = [
            (pre_display,  "Pre-Disaster"),
            (post_display, "Post-Disaster"),
            (gt_rgb,       "Ground Truth Damage"),
            (pred_rgb,     "Predicted Damage"),
        ]

        for col, (img, title) in enumerate(panels):
            ax = fig.add_subplot(n_samples, 4, row * 4 + col + 1)
            ax.imshow(img)
            ax.set_title(title if row == 0 else "")
            ax.axis("off")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0, 0.78, 0), label="No Damage"),
        Patch(facecolor=(1, 1, 0),    label="Minor"),
        Patch(facecolor=(1, 0.55, 0), label="Major"),
        Patch(facecolor=(0.86, 0, 0), label="Destroyed"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = Path(output_dir) / f"dmg_epoch_{epoch:03d}.png"
    fig.savefig(str(path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved preview: {path}")


def save_training_dashboard(
    train_losses, val_losses, train_metrics, val_metrics,
    metric_key, output_dir
):
    """Save live-updating loss + metric curves."""
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Progress", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, val_losses,   "r-o", label="Val Loss",   markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Metric
    t_vals = [m.get(metric_key, 0) for m in train_metrics]
    v_vals = [m.get(metric_key, 0) for m in val_metrics]
    axes[1].plot(epochs, t_vals, "b-o", label=f"Train {metric_key}", markersize=4)
    axes[1].plot(epochs, v_vals, "r-o", label=f"Val {metric_key}",   markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric_key)
    axes[1].set_title(metric_key.upper())
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = Path(output_dir) / "training_dashboard.png"
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Training loops
# ===========================================================================

def train_segmentation_epoch(model, loader, optimizer, criterion, device, metrics):
    model.train()
    metrics.reset()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  Train", leave=False):
        pre_img  = batch["pre_image"].to(device)
        bld_mask = batch["building_mask"].float().to(device)
        if bld_mask.dim() == 3:
            bld_mask = bld_mask.unsqueeze(1)

        optimizer.zero_grad()
        logits = model(pre_img)
        loss   = criterion(logits, bld_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        metrics.update(logits, bld_mask)
    return total_loss / len(loader)


@torch.no_grad()
def validate_segmentation(model, loader, criterion, device, metrics):
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
        metrics.update(logits, bld_mask)
    return total_loss / len(loader)


def train_damage_epoch(model, seg_model, loader, optimizer, criterion, device, metrics):
    model.train()
    metrics.reset()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  Train", leave=False):
        pre_img  = batch["pre_image"].to(device)
        post_img = batch["post_image"].to(device)
        dmg_mask = batch["damage_mask"].long().to(device)

        if seg_model is not None:
            with torch.no_grad():
                seg_logits = seg_model(pre_img)
                bld_binary = (torch.sigmoid(seg_logits).squeeze(1) > 0.5).long()
            dmg_mask = dmg_mask * bld_binary

        pair = torch.cat([pre_img, post_img], dim=1)
        optimizer.zero_grad()
        logits = model(pair)
        loss = criterion(logits, dmg_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        metrics.update(logits, dmg_mask)
    return total_loss / len(loader)


@torch.no_grad()
def validate_damage(model, seg_model, loader, criterion, device, metrics):
    model.eval()
    metrics.reset()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  Val  ", leave=False):
        pre_img  = batch["pre_image"].to(device)
        post_img = batch["post_image"].to(device)
        dmg_mask = batch["damage_mask"].long().to(device)

        if seg_model is not None:
            seg_logits = seg_model(pre_img)
            bld_binary = (torch.sigmoid(seg_logits).squeeze(1) > 0.5).long()
            dmg_mask = dmg_mask * bld_binary

        pair   = torch.cat([pre_img, post_img], dim=1)
        logits = model(pair)
        loss   = criterion(logits, dmg_mask)
        total_loss += loss.item()
        metrics.update(logits, dmg_mask)
    return total_loss / len(loader)


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Visual training for GeoSight")
    p.add_argument("--data-dir",     default="data/raw/xbd")
    p.add_argument("--output-dir",   default="data/outputs/training_progress")
    p.add_argument("--phase",        choices=["segmentation", "damage", "both"], default="both")
    p.add_argument("--epochs",       type=int, default=30)
    p.add_argument("--batch-size",   type=int, default=2)
    p.add_argument("--image-size",   type=int, default=512)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--val-split",    type=float, default=0.1)
    p.add_argument("--save-every",   type=int, default=2,
                   help="Save visual previews every N epochs")
    p.add_argument("--seg-checkpoint", default=None,
                   help="Resume segmentation from checkpoint")
    p.add_argument("--dmg-checkpoint", default=None,
                   help="Resume damage model from checkpoint")
    p.add_argument("--device",       default=None)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    )

    output_dir = Path(args.output_dir)
    seg_vis_dir = output_dir / "segmentation"
    dmg_vis_dir = output_dir / "damage"
    seg_vis_dir.mkdir(parents=True, exist_ok=True)
    dmg_vis_dir.mkdir(parents=True, exist_ok=True)
    Path("checkpoints/segmentation").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/damage").mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Visual output: {output_dir}")

    # ------------------------------------------------------------------ Data
    transform = XBDTransform(phase="train", image_size=args.image_size)
    full_dataset = XBDDataset(
        root_dir=args.data_dir, split="train", transform=transform,
    )

    val_size   = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    use_workers = 0 if device.type == "mps" else 4
    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=use_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=use_workers, pin_memory=pin,
    )
    logger.info(f"Data: {len(train_ds)} train / {len(val_ds)} val")

    # =========================================================================
    # PHASE 1: Building Segmentation
    # =========================================================================

    if args.phase in ("segmentation", "both"):
        logger.info("=" * 60)
        logger.info("PHASE 1: BUILDING SEGMENTATION")
        logger.info("=" * 60)

        seg_model = BuildingSegmentationModel().to(device)
        if args.seg_checkpoint:
            ckpt = torch.load(args.seg_checkpoint, map_location=device, weights_only=False)
            seg_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            logger.info(f"Resumed from: {args.seg_checkpoint}")

        seg_criterion = BceDiceLoss()
        seg_optimizer = build_segmentation_optimizer(seg_model, lr=args.lr)
        seg_scheduler = build_lr_scheduler(seg_optimizer, epochs=args.epochs)
        seg_train_metrics = SegmentationMetrics()
        seg_val_metrics   = SegmentationMetrics()

        best_iou = 0.0
        t_losses, v_losses, t_mets, v_mets = [], [], [], []

        for epoch in range(1, args.epochs + 1):
            lr = seg_scheduler.get_last_lr()[0]
            logger.info(f"\nEpoch {epoch}/{args.epochs}  LR={lr:.2e}")

            t_loss = train_segmentation_epoch(
                seg_model, train_loader, seg_optimizer, seg_criterion,
                device, seg_train_metrics,
            )
            v_loss = validate_segmentation(
                seg_model, val_loader, seg_criterion, device, seg_val_metrics,
            )
            seg_scheduler.step()

            t_met = seg_train_metrics.log(prefix=f"E{epoch} Train")
            v_met = seg_val_metrics.log(prefix=f"E{epoch} Val  ")

            t_losses.append(t_loss)
            v_losses.append(v_loss)
            t_mets.append(t_met)
            v_mets.append(v_met)

            # Save dashboard (updates every epoch)
            save_training_dashboard(
                t_losses, v_losses, t_mets, v_mets,
                "iou", str(seg_vis_dir),
            )

            # Save visual previews every N epochs
            if epoch % args.save_every == 0 or epoch == 1:
                save_segmentation_preview(
                    seg_model, val_ds, device, epoch, str(seg_vis_dir),
                )

            # Save best checkpoint
            if v_met["iou"] > best_iou:
                best_iou = v_met["iou"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": seg_model.state_dict(),
                    "val_iou": best_iou,
                }, "checkpoints/segmentation/best.pth")
                logger.info(f"  *** Best IoU: {best_iou:.4f} → saved checkpoint")

        logger.info(f"\nSegmentation complete. Best IoU: {best_iou:.4f}")
        logger.info(f"Previews saved in: {seg_vis_dir}/")

    # =========================================================================
    # PHASE 2: Damage Classification
    # =========================================================================

    if args.phase in ("damage", "both"):
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: DAMAGE CLASSIFICATION")
        logger.info("=" * 60)

        # Load trained segmentation model
        seg_model_frozen = BuildingSegmentationModel().to(device).eval()
        seg_ckpt_path = args.seg_checkpoint or "checkpoints/segmentation/best.pth"
        if Path(seg_ckpt_path).exists():
            ckpt = torch.load(seg_ckpt_path, map_location=device, weights_only=False)
            seg_model_frozen.load_state_dict(ckpt.get("model_state_dict", ckpt))
            for p in seg_model_frozen.parameters():
                p.requires_grad = False
            logger.info(f"Loaded frozen seg model: {seg_ckpt_path}")
        else:
            logger.warning("No seg checkpoint found — training damage without building mask")
            seg_model_frozen = None

        dmg_model = DamageClassificationModel(in_channels=6).to(device)
        if args.dmg_checkpoint:
            ckpt = torch.load(args.dmg_checkpoint, map_location=device, weights_only=False)
            dmg_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            logger.info(f"Resumed from: {args.dmg_checkpoint}")

        class_weights = [0.1, 1.0, 2.0, 3.0, 4.0]
        dmg_criterion = CombinedDamageLoss(class_weights=class_weights)
        dmg_optimizer = build_damage_optimizer(dmg_model, lr=args.lr)
        dmg_scheduler = build_lr_scheduler(dmg_optimizer, epochs=args.epochs)
        dmg_train_metrics = DamageMetrics()
        dmg_val_metrics   = DamageMetrics()

        best_score = 0.0
        t_losses, v_losses, t_mets, v_mets = [], [], [], []

        for epoch in range(1, args.epochs + 1):
            lr = dmg_scheduler.get_last_lr()[0]
            logger.info(f"\nEpoch {epoch}/{args.epochs}  LR={lr:.2e}")

            t_loss = train_damage_epoch(
                dmg_model, seg_model_frozen, train_loader,
                dmg_optimizer, dmg_criterion, device, dmg_train_metrics,
            )
            v_loss = validate_damage(
                dmg_model, seg_model_frozen, val_loader,
                dmg_criterion, device, dmg_val_metrics,
            )
            dmg_scheduler.step()

            t_met = dmg_train_metrics.log(prefix=f"E{epoch} Train")
            v_met = dmg_val_metrics.log(prefix=f"E{epoch} Val  ")

            t_losses.append(t_loss)
            v_losses.append(v_loss)
            t_mets.append({"iou": t_met.get("xview2_score", 0)})
            v_mets.append({"iou": v_met.get("xview2_score", 0)})

            save_training_dashboard(
                t_losses, v_losses, t_mets, v_mets,
                "iou", str(dmg_vis_dir),
            )

            if epoch % args.save_every == 0 or epoch == 1:
                save_damage_preview(
                    seg_model_frozen or BuildingSegmentationModel().to(device),
                    dmg_model, val_ds, device, epoch, str(dmg_vis_dir),
                )

            score = v_met.get("xview2_score", 0)
            if score > best_score:
                best_score = score
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": dmg_model.state_dict(),
                    "xview2_score": best_score,
                }, "checkpoints/damage/best.pth")
                logger.info(f"  *** Best xView2: {best_score:.4f} → saved checkpoint")

        logger.info(f"\nDamage training complete. Best xView2: {best_score:.4f}")
        logger.info(f"Previews saved in: {dmg_vis_dir}/")

    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("ALL TRAINING COMPLETE")
    logger.info(f"Visual progress:  {output_dir}/")
    logger.info(f"Checkpoints:      checkpoints/")
    logger.info("=" * 60)
    logger.info("\nNext: run inference with:")
    logger.info("  python3 scripts/run_assessment.py \\")
    logger.info("    --pre  data/raw/xbd/train/images/<event>_pre_disaster.png \\")
    logger.info("    --post data/raw/xbd/train/images/<event>_post_disaster.png \\")
    logger.info("    --seg-checkpoint checkpoints/segmentation/best.pth \\")
    logger.info("    --dmg-checkpoint checkpoints/damage/best.pth \\")
    logger.info("    --output-dir data/outputs/my_assessment")


if __name__ == "__main__":
    main()
