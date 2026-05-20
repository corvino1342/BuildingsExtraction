# buildings_dataset_creation.py
from html import parser

from html import parser

from PIL import Image
import os
import shutil
import argparse
import csv
from tqdm import tqdm
import numpy as np
import albumentations as A 

# python -m src.data.tiling --dataset_name  MassachusettsBuildingDataset --dataset_path /home/unet/datasets --tile_size 256 --stride 256 --maps_to_use -1 --splits train val test --overwrite --save_stats --augment --augment_factor 3
# --------------------------------------------------
# Argument parser
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Create tiles for building footprint data"
    )

    parser.add_argument("--augment", action="store_true",
                    help="Apply data augmentation to tiles")
    parser.add_argument("--augment_factor", type=int, default=10,
                    help="Target dataset size increase factor (default: 10)")

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)

    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride for sliding window (default: tile_size)")

    parser.add_argument("--maps_to_use", type=int, default=-1)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])

    parser.add_argument("--skip_empty", action="store_true",
                        help="Skip tiles with little/no foreground")

    parser.add_argument("--fg_threshold", type=float, default=0.01,
                        help="Minimum foreground ratio to keep tile while --skip_empty is called")

    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--save_stats", action="store_true",
                        help="Save tile statistics to CSV")

    return parser.parse_args()


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def clear_tiles_directory(dataset_path, dataset_name, tile_size):
    out_dir = f"{dataset_path}/{dataset_name}/tiles_{tile_size}"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


# --------------------------------------------------
# Main tiling logic
# --------------------------------------------------
def tiles_creation(args):

    tile_size = args.tile_size
    stride = args.stride if args.stride else tile_size

    base_tiles_dir = f"{args.dataset_path}/{args.dataset_name}/tiles"
    out_root = f"{args.dataset_path}/{args.dataset_name}/tiles_{tile_size}"

    stats = []

    for split in args.splits:

        print(f"\nProcessing split: {split}")

        image_dir = f"{base_tiles_dir}/{split}/images"
        gt_dir = f"{base_tiles_dir}/{split}/gt"
        has_gt = os.path.exists(gt_dir)

        out_img_dir = f"{out_root}/{split}/images"
        out_gt_dir = f"{out_root}/{split}/gt"
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_gt_dir, exist_ok=True)

        image_names = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(image_dir)
            if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".TIF", ".TIFF"))
        )

        if args.maps_to_use > 0:
            image_names = image_names[:args.maps_to_use]

        for name in tqdm(image_names, desc=f"Tiling {split}"):

            img = Image.open(f"{image_dir}/{name}.tiff")
            if has_gt:
                mask = Image.open(f"{gt_dir}/{name}.tif")

            W, H = img.size
            tile_id = 0

            for y in range(0, H - tile_size + 1, stride):
                for x in range(0, W - tile_size + 1, stride):

                    box = (x, y, x + tile_size, y + tile_size)
                    img_tile = img.crop(box)

                    if has_gt:
                        mask_tile = mask.crop(box)
                        mask_np = np.array(mask_tile) > 0
                        fg_ratio = mask_np.mean()
                    else:
                        fg_ratio = 0.0

                    if args.skip_empty and has_gt:
                        if fg_ratio < args.fg_threshold:
                            continue

                    #  aug00 is the original tile without augmentation, aug01, aug02, ... are the augmented versions
                    img_name = f"{name}_{tile_id:06d}_aug00.tif"
                    img_tile.save(f"{out_img_dir}/{img_name}")

                    if has_gt:
                        mask_tile.save(f"{out_gt_dir}/{img_name}")
                    if args.save_stats:
                        stats.append({
                            "split": split,
                            "image": name,
                            "tile_id": tile_id,
                            "fg_ratio": fg_ratio
                        })
                    # Augmentation
                    if args.augment:
                        augmented_pairs = apply_augmentations(
                            img_tile,
                            mask_tile if has_gt else None,
                            args.augment_factor
                        )
                        for aug_id, (aug_img, aug_mask) in enumerate(augmented_pairs):
                            aug_name = f"{name}_{tile_id:06d}_aug{aug_id}.tif"
                            aug_img.save(f"{out_img_dir}/{aug_name}")
                            if has_gt:
                                aug_mask.save(f"{out_gt_dir}/{aug_name}")
                            if args.save_stats:
                                stats.append({
                                    "split": split,
                                    "image": name,
                                    "tile_id": tile_id,
                                    "fg_ratio": fg_ratio,
                                    "augmented": True,
                                    "aug_id": aug_id
                                })
                    

                    tile_id += 1

    # --------------------------------------------------
    # Save statistics
    # --------------------------------------------------
    if args.save_stats and stats:
        stats_path = f"{out_root}/tile_statistics.csv"
        with open(stats_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["split", "image", "tile_id", "fg_ratio", "augmented", "aug_id"]
            )
            writer.writeheader()
            writer.writerows(stats)

        print(f"\nTile statistics saved to: {stats_path}")

# --------------------------------------------------
# Data Augmentation
# --------------------------------------------------
def apply_augmentations(img, mask=None, augment_factor=10):
    """
    Applies random augmentations to an image and its mask.
    Returns a list of augmented (image, mask) pairs.
    """
    # Define augmentation pipeline
    transform = A.Compose([
        A.Rotate(limit=90, p=0.5),  # Rotate by -90 to +90 degrees (randomly)
        A.HorizontalFlip(p=0.5),   # 50% chance to flip horizontally
        A.VerticalFlip(p=0.5),     # 50% chance to flip vertically
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
    ])

    augmented_pairs = []
    for _ in range(augment_factor - 1):  # -1 because original is already in the dataset
        augmented = transform(image=np.array(img), mask=np.array(mask) if mask is not None else None)
        aug_img = Image.fromarray(augmented["image"])
        aug_mask = Image.fromarray(augmented["mask"]) if mask is not None else None
        augmented_pairs.append((aug_img, aug_mask))

    return augmented_pairs

# --------------------------------------------------
# Entry point
# --------------------------------------------------
def main():
    args = parse_args()

    if args.overwrite:
        clear_tiles_directory(
            args.dataset_path,
            args.dataset_name,
            args.tile_size
        )

    tiles_creation(args)


if __name__ == "__main__":
    main()