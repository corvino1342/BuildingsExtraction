import os
import time
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# ---- your imports ----
from dataset import MyDataset
from unet import UNet, UNetL, UNetLL


#python training.py --dataset_path /mnt/nas151/sar/Footprint/datasets --dataset_name WHUBuildingDataset --mode tiles --tile_size 256 --batch_size 4 --epochs 30 --lr 0.001 --arch unetLL --loss wbce+dice --output_dir /home/antoniocorvino/Projects/BuildingsExtraction/runs
def iou_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean()

def precision_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    true_positive = (preds * targets).sum()
    predicted_positive = preds.sum()
    return (true_positive + eps) / (predicted_positive + eps)

def recall_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    true_positive = (preds * targets).sum()
    actual_positive = targets.sum()
    return (true_positive + eps) / (actual_positive + eps)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice

def parse_args():
    parser = argparse.ArgumentParser("Building Footprint Training")

    # Dataset
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, choices=["MassachusettsBuildingDataset", "InriaBuildingDataset", "WHUBuildingDataset"], default="WHUBuildingDataset")
    parser.add_argument("--mode", type=str, choices=["tiles", "instances"], default="tiles")

    # Training
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Model
    parser.add_argument("--arch", type=str, choices=["unet", "unetL", "unetLL"], default="unet")

    # Loss
    parser.add_argument("--loss", type=str, choices=["bce", "wbce", "wbce+dice"], default="wbce")

    # Misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./runs")

    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def build_model(arch):
    if arch == "unet":
        return UNet(3, 1)
    if arch == "unetL":
        return UNetL(3, 1)
    if arch == "unetLL":
        return UNetLL(3, 1)

def build_loss(loss_name, train_dataset):
    dice = None

    if "wbce" in loss_name:
        pos = 0
        total = 0
        for _, mask in train_dataset:
            pos += mask.sum()
            total += mask.numel()

        neg = total - pos
        weight = torch.tensor(neg / (pos + 1e-6))
        bce = nn.BCEWithLogitsLoss(pos_weight=weight)
    else:
        bce = nn.BCEWithLogitsLoss()

    if "dice" in loss_name:
        dice = DiceLoss()

    return bce, dice

def train_one_epoch(model, loader, optimizer, bce, dice, device):
    model.train()

    loss_sum, iou_sum, p_sum, r_sum = 0, 0, 0, 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device).float()

        optimizer.zero_grad()
        out = model(imgs)

        loss = bce(out, masks)
        if dice:
            loss += dice(out, masks)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            loss_sum += loss.item()
            iou_sum += iou_score(out, masks)
            p_sum += precision_score(out, masks)
            r_sum += recall_score(out, masks)

    n = len(loader)
    return loss_sum/n, iou_sum/n, p_sum/n, r_sum/n

@torch.no_grad()
def validate(model, loader, bce, dice, device):
    model.eval()

    loss_sum, iou_sum, p_sum, r_sum = 0, 0, 0, 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device).float()

        out = model(imgs)
        loss = bce(out, masks)
        if dice:
            loss += dice(out, masks)

        loss_sum += loss.item()
        iou_sum += iou_score(out, masks)
        p_sum += precision_score(out, masks)
        r_sum += recall_score(out, masks)

    n = len(loader)
    return loss_sum/n, iou_sum/n, p_sum/n, r_sum/n

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()

    train_ds = MyDataset(
        image_dir=f"{args.dataset_path}/{args.dataset_name}/{args.mode}/train/images",
        mask_dir=f"{args.dataset_path}/{args.dataset_name}/{args.mode}/train/gt",
        transform=transform
    )

    val_ds = MyDataset(
        image_dir=f"{args.dataset_path}/{args.dataset_name}/{args.mode}/val/images",
        mask_dir=f"{args.dataset_path}/{args.dataset_name}/{args.mode}/val/gt",
        transform=transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = build_model(args.arch).to(device)
    bce, dice = build_loss(args.loss, train_ds)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    run_name = f"{args.arch}_{args.mode}_{args.loss}_dim{args.tile_size}_bs{args.batch_size}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    best_val = float("inf")

    for epoch in range(args.epochs):
        tr = train_one_epoch(model, train_loader, optimizer, bce, dice, device)
        va = validate(model, val_loader, bce, dice, device)

        print(f"[{epoch+1}/{args.epochs}] "
              f"TRAIN loss {tr[0]:.4f} IoU {tr[1]:.3f} | "
              f"VAL loss {va[0]:.4f} IoU {va[1]:.3f}")

        if va[0] < best_val:
            best_val = va[0]
            torch.save(model.state_dict(), f"{out_dir}/best_model.pth")

if __name__ == "__main__":
    main()