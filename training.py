import os
import time
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import MyDataset
from unet import UNet, UNetL, UNetLL



#python training.py --dataset_path /mnt/nas151/sar/Footprint/datasets --dataset_name WHUBuildingDataset --mode tiles --fixed_size True --tile_size 256 --batch_size 32 --epochs 30 --lr 0.001 --arch unetLL --loss wbce --output_dir /home/antoniocorvino/Projects/BuildingsExtraction/runs/



# --------------------------------------------------
# Metrics logger
# --------------------------------------------------
class MetricsLogger:
    def __init__(self, output_dir, filename="metrics.csv"):
        os.makedirs(output_dir, exist_ok=True)
        self.csv_path = os.path.join(output_dir, filename)

        self.header = [
            "epoch",
            "train_loss", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_iou", "val_precision", "val_recall",
            "epoch_time_sec"
        ]

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self.header)

    def log(self, epoch, tr, va, epoch_time):
        row = [
            epoch,
            tr["loss"], tr["iou"], tr["precision"], tr["recall"],
            va["loss"], va["iou"], va["precision"], va["recall"],
            round(epoch_time, 2)
        ]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)


# --------------------------------------------------
# Metrics (correct per-image averaging)
# --------------------------------------------------
def iou_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean()


def precision_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(preds) > threshold).float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    pp = preds.sum(dim=(1, 2, 3))
    return ((tp + eps) / (pp + eps)).mean()


def recall_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(preds) > threshold).float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    ap = targets.sum(dim=(1, 2, 3))
    return ((tp + eps) / (ap + eps)).mean()


# --------------------------------------------------
# Dice loss
# --------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


# --------------------------------------------------
# CLI
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("Building Footprint Training")

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="WHUBuildingDataset")
    parser.add_argument("--mode", type=str, choices=["tiles", "instances"], default="tiles") #if instances set batch_size to 1. UNet cannot handle batch with different tile size

    parser.add_argument("--fixed_size", type=bool,  help="Enforce fixed tile size (recommended for semantic tiles)")
    
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--arch", type=str, choices=["unet", "unetL", "unetLL"], default="unet")
    parser.add_argument("--loss", type=str, choices=["bce", "wbce", "wbce+dice"], default="wbce")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./runs")

    return parser.parse_args()


# --------------------------------------------------
# Utilities
# --------------------------------------------------
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


def build_loss(loss_name, train_dataset, device):
    dice = None

    if "wbce" in loss_name:
        pos, total = 0, 0
        for _, mask in tqdm(train_dataset, desc="Computing Weights"):
            pos += mask.sum()
            total += mask.numel()
        neg = total - pos
        weight = torch.tensor(neg / (pos + 1e-6), device=device)
        print(f"Weight: {weight:.2f}")
        bce = nn.BCEWithLogitsLoss(pos_weight=weight)
    else:
        bce = nn.BCEWithLogitsLoss()

    if "dice" in loss_name:
        dice = DiceLoss()

    return bce, dice


# --------------------------------------------------
# Training / validation
# --------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, bce, dice, device, tile_size):
    model.train()
    loss_sum = iou_sum = p_sum = r_sum = 0.0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device).float()



        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(imgs)
            loss = bce(out, masks)
            if dice:
                loss += dice(out, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            loss_sum += loss.item()
            iou_sum += iou_score(out, masks)
            p_sum += precision_score(out, masks)
            r_sum += recall_score(out, masks)

    n = len(loader)
    return {
        "loss": loss_sum / n,
        "iou": iou_sum / n,
        "precision": p_sum / n,
        "recall": r_sum / n,
    }


@torch.no_grad()
def validate(model, loader, bce, dice, device, tile_size):
    model.eval()
    loss_sum = iou_sum = p_sum = r_sum = 0.0

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
    return {
        "loss": loss_sum / n,
        "iou": iou_sum / n,
        "precision": p_sum / n,
        "recall": r_sum / n,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()

    train_ds = MyDataset(
        f"{args.dataset_path}/{args.dataset_name}/{args.mode}/train/images",
        f"{args.dataset_path}/{args.dataset_name}/{args.mode}/train/gt",
        transform=transform
    )

    val_ds = MyDataset(
        f"{args.dataset_path}/{args.dataset_name}/{args.mode}/val/images",
        f"{args.dataset_path}/{args.dataset_name}/{args.mode}/val/gt",
        transform=transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    model = build_model(args.arch).to(device)
    bce, dice = build_loss(args.loss, train_ds, device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    run_name = f"{args.arch}_{args.mode}_{args.loss}_dim{args.tile_size}_n{len(train_loader)}_bs{args.batch_size}"
    out_dir = os.path.join(args.output_dir, args.dataset_name)
    os.path.join(out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    logger = MetricsLogger(out_dir)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        start = time.time()


        
        tr = train_one_epoch(model, train_loader, optimizer, scaler, bce, dice, device, args.tile_size)
        va = validate(model, val_loader, bce, dice, device, args.tile_size)

        epoch_time = time.time() - start
        logger.log(epoch, tr, va, epoch_time)

        print(
            f"[{epoch}/{args.epochs}]\t"
            f"TRAIN loss {tr['loss']:.4f} IoU {tr['iou']:.3f} | "
            f"VAL loss {va['loss']:.4f} IoU {va['iou']:.3f}"
        )

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(model.state_dict(), f"{out_dir}/best_model.pth")

    torch.save(model.state_dict(), f"{out_dir}/last_model.pth")


if __name__ == "__main__":
    main()