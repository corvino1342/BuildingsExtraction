# buildings_dataset_creation.py
from PIL import Image
import os
import shutil
import argparse
import csv
from tqdm import tqdm
import numpy as np

#python buildings_dataset_creation.py --dataset_name  WHUBuildingDataset --dataset_path /mnt/nas151/sar/Footprint/data
# --tile_size 128 --stride --maps_to_use -1 --splits train --skip_empty --fg_threshold --overwrite --save_stats
# --------------------------------------------------
# Argument parser
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Create tiles for building footprint data"
    )

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
            if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))
        )

        if args.maps_to_use > 0:
            image_names = image_names[:args.maps_to_use]

        for name in tqdm(image_names, desc=f"Tiling {split}"):

            img = Image.open(f"{image_dir}/{name}.TIF")
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

                    img_name = f"{name}_{tile_id:06d}.tif"
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

                    tile_id += 1

    # --------------------------------------------------
    # Save statistics
    # --------------------------------------------------
    if args.save_stats and stats:
        stats_path = f"{out_root}/tile_statistics.csv"
        with open(stats_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["split", "image", "tile_id", "fg_ratio"]
            )
            writer.writeheader()
            writer.writerows(stats)

        print(f"\nTile statistics saved to: {stats_path}")


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