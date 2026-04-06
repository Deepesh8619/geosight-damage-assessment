"""
GeoSight Training on Kaggle (Free P100 GPU)
=============================================
This is a self-contained training script for Kaggle notebooks.

STEPS:
1. Go to kaggle.com → New Notebook
2. Settings → Accelerator → GPU P100
3. Upload your xBD dataset:
   - Go to kaggle.com/datasets → New Dataset
   - Upload the train/ folder (images/ + labels/) as a zip
   - Name it "xbd-train"
4. In the notebook, add the dataset: + Add Data → Your Datasets → xbd-train
5. Copy-paste this ENTIRE file into a notebook cell and run it

Your dataset will be at: /kaggle/input/xbd-train/train/
Training output goes to: /kaggle/working/

After training, download the .pth files from the Output tab.
"""

# %% Cell 1: Install dependencies
!pip install -q segmentation-models-pytorch albumentations loguru

# %% Cell 2: Setup paths
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm
from loguru import logger

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# === CHANGE THIS PATH if your dataset is named differently ===
# On Kaggle, uploaded datasets appear at /kaggle/input/<dataset-name>/
DATA_DIR = "/kaggle/input/xbd-train"

# Auto-detect: some uploads have an extra nested folder
if Path(f"{DATA_DIR}/train/images").exists():
    DATA_ROOT = DATA_DIR
elif Path(f"{DATA_DIR}/train/train/images").exists():
    DATA_ROOT = f"{DATA_DIR}/train"
else:
    # List what's actually there so you can fix the path
    print("Could not find images. Contents of input:")
    for p in Path("/kaggle/input").rglob("*"):
        if p.is_dir():
            print(f"  {p}")
    raise FileNotFoundError("Fix DATA_DIR path above")

print(f"Data root: {DATA_ROOT}")
print(f"Images: {len(list(Path(DATA_ROOT, 'train/images').glob('*_pre_*.png')))} pre/post pairs")

OUTPUT_DIR = "/kaggle/working"
Path(f"{OUTPUT_DIR}/checkpoints/segmentation").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DIR}/checkpoints/damage").mkdir(parents=True, exist_ok=True)

# %% Cell 3: Dataset (self-contained, no imports from src/)

XBD_LABEL_MAP = {
    "background": 0, "no-damage": 1, "minor-damage": 2,
    "major-damage": 3, "destroyed": 4, "un-classified": 1,
}

class XBDDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.image_dir = Path(root_dir) / split / "images"
        self.label_dir = Path(root_dir) / split / "labels"
        self.transform = transform
        self.samples = self._build_samples()
        print(f"Loaded {len(self.samples)} pairs from {self.image_dir}")

    def _build_samples(self):
        samples = []
        for pre_path in sorted(self.image_dir.glob("*_pre_*.png")):
            post_path = self.image_dir / pre_path.name.replace("_pre_", "_post_")
            pre_label = self.label_dir / f"{pre_path.stem}.json"
            post_label = self.label_dir / f"{pre_path.stem.replace('_pre_', '_post_')}.json"
            if post_path.exists() and pre_label.exists() and post_label.exists():
                samples.append({
                    "pre": str(pre_path), "post": str(post_path),
                    "pre_label": str(pre_label), "post_label": str(post_label),
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        from shapely import wkt
        from rasterio.features import rasterize
        from affine import Affine

        s = self.samples[idx]
        pre = np.array(Image.open(s["pre"]).convert("RGB"))
        post = np.array(Image.open(s["post"]).convert("RGB"))
        h, w = pre.shape[:2]

        with open(s["pre_label"]) as f:
            pre_lbl = json.load(f)
        with open(s["post_label"]) as f:
            post_lbl = json.load(f)

        transform = Affine(1, 0, 0, 0, 1, 0)

        # Building mask
        bld_mask = np.zeros((h, w), dtype=np.uint8)
        shapes = []
        for feat in pre_lbl.get("features", {}).get("xy", []):
            wkt_str = feat.get("wkt")
            if wkt_str:
                try:
                    geom = wkt.loads(wkt_str)
                    if geom and not geom.is_empty:
                        shapes.append((geom, 1))
                except:
                    pass
        if shapes:
            bld_mask = rasterize(shapes, out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8)

        # Damage mask
        dmg_mask = np.zeros((h, w), dtype=np.uint8)
        for class_label, class_idx in sorted(XBD_LABEL_MAP.items(), key=lambda x: x[1]):
            shapes = []
            for feat in post_lbl.get("features", {}).get("xy", []):
                if feat.get("properties", {}).get("subtype", "").lower() != class_label:
                    continue
                wkt_str = feat.get("wkt")
                if wkt_str:
                    try:
                        geom = wkt.loads(wkt_str)
                        if geom and not geom.is_empty:
                            shapes.append((geom, class_idx))
                    except:
                        pass
            if shapes:
                cm = rasterize(shapes, out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8)
                dmg_mask = np.where(cm > 0, cm, dmg_mask)

        if self.transform:
            result = self.transform(image=pre, post_image=post, building_mask=bld_mask, damage_mask=dmg_mask)
            return {
                "pre_image": result["image"],
                "post_image": result["post_image"],
                "building_mask": torch.from_numpy(result["building_mask"]).long() if isinstance(result["building_mask"], np.ndarray) else result["building_mask"].long(),
                "damage_mask": torch.from_numpy(result["damage_mask"]).long() if isinstance(result["damage_mask"], np.ndarray) else result["damage_mask"].long(),
            }

        return {"pre_image": pre, "post_image": post, "building_mask": bld_mask, "damage_mask": dmg_mask}


# %% Cell 4: Augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(phase="train", size=512):
    if phase == "train":
        t = [
            A.RandomCrop(size, size), A.HorizontalFlip(0.5),
            A.VerticalFlip(0.5), A.RandomRotate90(0.5),
            A.ColorJitter(0.2, 0.2, 0.1, 0.05, p=0.4),
        ]
    else:
        t = [A.PadIfNeeded(size, size, border_mode=0)]

    t += [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    return A.Compose(t, additional_targets={"post_image": "image", "building_mask": "mask", "damage_mask": "mask"})


# %% Cell 5: Models
import segmentation_models_pytorch as smp

def create_seg_model():
    return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)

def create_dmg_model():
    return smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=6, classes=5)


# %% Cell 6: Loss functions (CUDA-safe)
class BceDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        targets = targets.float()
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits).view(-1)
        tgt = targets.view(-1)
        inter = (probs * tgt).sum()
        dice = 1 - (2 * inter + 1) / (probs.sum() + tgt.sum() + 1)
        return 0.5 * bce + 0.5 * dice

class FocalCELoss(nn.Module):
    def __init__(self, weights, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weights = weights  # list, not tensor — created on device in forward

    def forward(self, logits, targets):
        w = torch.tensor(self.weights, device=logits.device, dtype=torch.float32)
        ce = F.cross_entropy(logits, targets, weight=w, reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


# %% Cell 7: Training functions
def train_epoch(model, loader, optimizer, criterion, device, task="seg"):
    model.train()
    total_loss, tp, fp, fn = 0, 0, 0, 0
    for batch in tqdm(loader, leave=False):
        pre = batch["pre_image"].to(device)
        if task == "seg":
            target = batch["building_mask"].float().to(device).unsqueeze(1)
            logits = model(pre)
        else:
            post = batch["post_image"].to(device)
            target = batch["damage_mask"].long().to(device)
            logits = model(torch.cat([pre, post], dim=1))

        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()

        # Quick IoU for segmentation
        if task == "seg":
            pred = (torch.sigmoid(logits) > 0.5).float()
            tp += (pred * target).sum().item()
            fp += (pred * (1 - target)).sum().item()
            fn += ((1 - pred) * target).sum().item()

    avg_loss = total_loss / len(loader)
    iou = tp / (tp + fp + fn + 1e-8) if task == "seg" else 0
    return avg_loss, iou

@torch.no_grad()
def val_epoch(model, loader, criterion, device, task="seg"):
    model.eval()
    total_loss, tp, fp, fn = 0, 0, 0, 0
    all_preds, all_targets = [], []
    for batch in tqdm(loader, leave=False):
        pre = batch["pre_image"].to(device)
        if task == "seg":
            target = batch["building_mask"].float().to(device).unsqueeze(1)
            logits = model(pre)
        else:
            post = batch["post_image"].to(device)
            target = batch["damage_mask"].long().to(device)
            logits = model(torch.cat([pre, post], dim=1))

        loss = criterion(logits, target)
        total_loss += loss.item()

        if task == "seg":
            pred = (torch.sigmoid(logits) > 0.5).float()
            tp += (pred * target).sum().item()
            fp += (pred * (1 - target)).sum().item()
            fn += ((1 - pred) * target).sum().item()
        else:
            pred = logits.argmax(dim=1).cpu().numpy().ravel()
            tgt = target.cpu().numpy().ravel()
            valid = tgt != 0
            all_preds.extend(pred[valid].tolist())
            all_targets.extend(tgt[valid].tolist())

    avg_loss = total_loss / len(loader)
    if task == "seg":
        iou = tp / (tp + fp + fn + 1e-8)
    else:
        from sklearn.metrics import f1_score
        iou = f1_score(all_targets, all_preds, average="macro", zero_division=0) if all_preds else 0
    return avg_loss, iou


# %% Cell 8: TRAIN SEGMENTATION
print("=" * 60)
print("PHASE 1: BUILDING SEGMENTATION")
print("=" * 60)

SEG_EPOCHS = 50
SEG_BATCH = 8
SEG_LR = 0.0001
IMAGE_SIZE = 512

dataset = XBDDataset(DATA_ROOT, split="train", transform=get_transforms("train", IMAGE_SIZE))
val_size = int(len(dataset) * 0.1)
train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=SEG_BATCH, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=SEG_BATCH, shuffle=False, num_workers=2, pin_memory=True)

seg_model = create_seg_model().to(device)
seg_criterion = BceDiceLoss()
seg_optimizer = torch.optim.AdamW([
    {"params": seg_model.encoder.parameters(), "lr": SEG_LR * 0.1},
    {"params": seg_model.decoder.parameters(), "lr": SEG_LR},
    {"params": seg_model.segmentation_head.parameters(), "lr": SEG_LR},
], weight_decay=1e-4)
seg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(seg_optimizer, T_max=SEG_EPOCHS, eta_min=1e-7)

best_seg_iou = 0
for epoch in range(1, SEG_EPOCHS + 1):
    t_loss, t_iou = train_epoch(seg_model, train_loader, seg_optimizer, seg_criterion, device, "seg")
    v_loss, v_iou = val_epoch(seg_model, val_loader, seg_criterion, device, "seg")
    seg_scheduler.step()

    print(f"E{epoch:02d} | Train Loss={t_loss:.4f} IoU={t_iou:.4f} | Val Loss={v_loss:.4f} IoU={v_iou:.4f}", end="")

    if v_iou > best_seg_iou:
        best_seg_iou = v_iou
        torch.save({"epoch": epoch, "model_state_dict": seg_model.state_dict(), "val_iou": v_iou},
                    f"{OUTPUT_DIR}/checkpoints/segmentation/best.pth")
        print(f" *** BEST", end="")
    print()

print(f"\nSegmentation done. Best IoU: {best_seg_iou:.4f}")


# %% Cell 9: TRAIN DAMAGE CLASSIFICATION
print("\n" + "=" * 60)
print("PHASE 2: DAMAGE CLASSIFICATION")
print("=" * 60)

DMG_EPOCHS = 50
DMG_BATCH = 4
DMG_LR = 0.0001

# Load frozen segmentation model
seg_frozen = create_seg_model().to(device).eval()
ckpt = torch.load(f"{OUTPUT_DIR}/checkpoints/segmentation/best.pth")
seg_frozen.load_state_dict(ckpt["model_state_dict"])
for p in seg_frozen.parameters():
    p.requires_grad = False

dmg_model = create_dmg_model().to(device)
dmg_criterion = FocalCELoss(weights=[0.1, 1.0, 2.0, 3.0, 4.0])
dmg_optimizer = torch.optim.AdamW(dmg_model.parameters(), lr=DMG_LR, weight_decay=1e-4)
dmg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dmg_optimizer, T_max=DMG_EPOCHS, eta_min=1e-7)

# Re-create loaders (damage needs both pre and post)
train_loader = DataLoader(train_ds, batch_size=DMG_BATCH, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=DMG_BATCH, shuffle=False, num_workers=2, pin_memory=True)

best_dmg_f1 = 0
for epoch in range(1, DMG_EPOCHS + 1):
    t_loss, _ = train_epoch(dmg_model, train_loader, dmg_optimizer, dmg_criterion, device, "dmg")
    v_loss, v_f1 = val_epoch(dmg_model, val_loader, dmg_criterion, device, "dmg")
    dmg_scheduler.step()

    print(f"E{epoch:02d} | Train Loss={t_loss:.4f} | Val Loss={v_loss:.4f} F1={v_f1:.4f}", end="")

    if v_f1 > best_dmg_f1:
        best_dmg_f1 = v_f1
        torch.save({"epoch": epoch, "model_state_dict": dmg_model.state_dict(), "xview2_score": v_f1},
                    f"{OUTPUT_DIR}/checkpoints/damage/best.pth")
        print(f" *** BEST", end="")
    print()

print(f"\nDamage done. Best F1: {best_dmg_f1:.4f}")


# %% Cell 10: Download checkpoints
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Segmentation: IoU = {best_seg_iou:.4f}")
print(f"Damage:       F1  = {best_dmg_f1:.4f}")
print(f"\nCheckpoints saved to:")
print(f"  {OUTPUT_DIR}/checkpoints/segmentation/best.pth")
print(f"  {OUTPUT_DIR}/checkpoints/damage/best.pth")
print(f"\nDownload from the 'Output' tab on the right →")
print(f"Then copy to your local project: checkpoints/ folder")
