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
from src.data.dataset import MyDataset
from unet import UNet, UNetL, UNetLL
from src.evaluation.metrics import iou_score, precision_score, recall_score, MetricsLogger
from src.training.losses import build_loss
from src.training.trainer import Trainer

#python training.py --dataset_path /mnt/nas151/sar/Footprint/data --dataset_name WHUBuildingDataset --mode tiles --fixed_size True --tile_size 256 --batch_size 32 --epochs 30 --lr 0.001 --arch unetLL --loss wbce --output_dir /home/antoniocorvino/Projects/BuildingsExtraction/runs/



# --------------------------------------------------
# CLI
# --------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser("Building Footprint Training")

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="WHUBuildingDataset")
    parser.add_argument("--mode", type=str, choices=["tiles", "instances"], default="tiles") #if instances set batch_size to 1. UNet cannot handle batch with different tile size

    parser.add_argument("--fixed_size", action="store_true",  help="Enforce fixed tile size (recommended for semantic tiles)")

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


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


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
        persistent_workers=args.num_workers > 0    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0    )

    from src.models.factory import build_model
    model = build_model(args.arch).to(device)

    bce, dice = build_loss(args.loss, train_ds, device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    run_name = (
        f"{args.arch}_{args.mode}_{args.loss}"
        f"_dim{args.tile_size}"
        f"_n{len(train_loader.dataset)}"
        f"_bs{args.batch_size}"
    )

    out_dir_ = os.path.join(args.output_dir, args.dataset_name)
    out_dir = os.path.join(out_dir_, run_name)
    os.makedirs(out_dir, exist_ok=True)

    logger = MetricsLogger(out_dir)

    best_val = float("inf")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        bce=bce,
        dice=dice,
        device=device,
        use_amp=(device.type == "cuda")
    )

    for epoch in range(1, args.epochs + 1):
        start = time.time()


        tr = trainer.train_one_epoch(train_loader)
        va = trainer.validate(val_loader)


        epoch_time = time.time() - start
        logger.log(epoch, tr, va, epoch_time)

        print(
            f"[{epoch}/{args.epochs}]\t"
            f"TRAIN loss {tr['loss']:.4f} IoU {tr['iou']:.3f} | "
            f"VAL loss {va['loss']:.4f} IoU {va['iou']:.3f} | "
            f"Time {epoch_time:.1f}s"
        )

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(model.state_dict(), f"{out_dir}/best_model.pth")

    torch.save(model.state_dict(), f"{out_dir}/last_model.pth")


if __name__ == "__main__":
    main()