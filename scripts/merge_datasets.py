"""
Merge several already-tiled datasets into ONE dataset folder that train.py /
MyDataset can read without any code change.

Result layout (same as every other dataset):
    <dataset_path>/<out_name>/tiles_<N>/<split>/images/<SRC>__<tile>.tif
    <dataset_path>/<out_name>/tiles_<N>/<split>/gt/<SRC>__<tile>.tif
    <dataset_path>/<out_name>/tiles_<N>/manifest.csv   (source of every merged tile)

By default files are SYMLINKS (no extra disk space on the NAS). The source
prefix avoids name collisions, and each gt link gets exactly the same name as
its image link, so MyDataset's sorted() pairing stays aligned.

Balancing (train split only, val is always kept whole so you can evaluate
each source on all of its data):
  --ratios 0.4 0.4 0.2   target share of each source in the merged train set.
                         Only UNDERSAMPLING is done: the total is the largest
                         one reachable without duplicating any tile, so the
                         smallest source (relative to its ratio) is kept whole.
  --originals_only       drop offline-augmented copies made by tiling.py
                         (keeps *_aug00 and tiles with no _aug suffix), so
                         augmentation does not inflate one source's share.

Example:
    python -m scripts.merge_datasets \
        --dataset_path /mnt/nas151/sar/Footprint/datasets \
        --sources WHUBuildingDataset InriaBuildingDataset MassachusettsBuildingDataset \
        --prefixes WHU INRIA MASS --ratios 0.4 0.4 0.2 \
        --tile_size 256 --out_name Merged_WHU_Inria_Mass --originals_only

Then:
    python -m scripts.train --dataset_name Merged_WHU_Inria_Mass --tile_size 256 ...
"""

import argparse
import csv
import os
import random
import re
import shutil
from pathlib import Path

EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
AUG_RE = re.compile(r"_aug(\d+)$")


def parse_args():
    p = argparse.ArgumentParser("Merge tiled building datasets")
    p.add_argument("--dataset_path", type=str, default="/mnt/nas151/sar/Footprint/datasets")
    p.add_argument("--sources", nargs="+", required=True,
                   help="Dataset names to merge (each must have tiles_<N>/<split>/{images,gt})")
    p.add_argument("--prefixes", nargs="*", default=None,
                   help="Optional short prefix per source (default: the source name)")
    p.add_argument("--ratios", nargs="*", type=float, default=None,
                   help="Target share of each source in the TRAIN split (e.g. 0.4 0.4 0.2). "
                        "Default: keep everything.")
    p.add_argument("--originals_only", action="store_true",
                   help="Skip tiling.py augmented copies (_aug01, _aug1, ...); keep _aug00")
    p.add_argument("--tile_size", type=int, default=256)
    p.add_argument("--out_name", type=str, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def index_by_stem(folder):
    return {f.stem: f for f in sorted(Path(folder).iterdir()) if f.suffix.lower() in EXTS}


def is_original(stem):
    m = AUG_RE.search(stem)
    return m is None or m.group(1) == "00"


def place(src, dst, mode):
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def collect(root, source, tiles_dir, split, originals_only):
    """Return (images_by_stem, masks_by_stem, sorted list of usable stems) or None."""
    base = root / source / tiles_dir / split
    img_dir, gt_dir = base / "images", base / "gt"
    if not img_dir.exists() or not gt_dir.exists():
        print(f"[skip] {source}/{split}: missing {img_dir} or {gt_dir}")
        return None

    images, masks = index_by_stem(img_dir), index_by_stem(gt_dir)
    stems = sorted(s for s in images if s in masks)
    n_orphans = len(images) - len(stems)
    if n_orphans:
        print(f"[warn] {source}/{split}: {n_orphans} image(s) without mask, skipped")
    if originals_only:
        stems = [s for s in stems if is_original(s)]
    return images, masks, stems


def target_counts(available, ratios):
    """Largest per-source counts matching `ratios` without oversampling any source."""
    total_ratio = sum(ratios)
    ratios = [r / total_ratio for r in ratios]
    total = min(n / r for n, r in zip(available, ratios) if r > 0)
    return [min(n, int(r * total)) for n, r in zip(available, ratios)]


def main():
    args = parse_args()
    random.seed(args.seed)

    prefixes = args.prefixes or args.sources
    if len(prefixes) != len(args.sources):
        raise ValueError("--prefixes must have the same length as --sources")
    if args.ratios is not None and len(args.ratios) != len(args.sources):
        raise ValueError("--ratios must have the same length as --sources")

    root = Path(args.dataset_path)
    tiles_dir = f"tiles_{args.tile_size}"
    out_root = root / args.out_name / tiles_dir

    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_root} exists, pass --overwrite to rebuild it")
        shutil.rmtree(out_root)

    manifest = []

    for split in args.splits:
        out_img = out_root / split / "images"
        out_gt = out_root / split / "gt"
        out_img.mkdir(parents=True, exist_ok=True)
        out_gt.mkdir(parents=True, exist_ok=True)

        collected = [collect(root, s, tiles_dir, split, args.originals_only) for s in args.sources]

        # Decide how many tiles to take from each source
        available = [len(c[2]) if c else 0 for c in collected]
        if split == "train" and args.ratios is not None:
            usable = [(n, r) for n, r in zip(available, args.ratios) if n > 0]
            counts = target_counts([n for n, _ in usable], [r for _, r in usable])
            it = iter(counts)
            counts = [next(it) if n > 0 else 0 for n in available]
        else:
            counts = available

        print(f"\n[{split}]")
        for source, prefix, c, n_keep, n_avail in zip(args.sources, prefixes, collected, counts, available):
            if c is None:
                continue
            images, masks, stems = c
            if n_keep < len(stems):
                stems = sorted(random.sample(stems, n_keep))

            for stem in stems:
                img_src = images[stem]
                # Same name for image and mask -> sorted() keeps them aligned.
                # PIL detects format from content, so the image's extension is fine for both.
                new_name = f"{prefix}__{stem}{img_src.suffix}"
                place(img_src, out_img / new_name, args.mode)
                place(masks[stem], out_gt / new_name, args.mode)
                
                manifest.append({"split": split, "source": source, "name": new_name,
                                 "image": str(img_src), "gt": str(masks[stem])})

            print(f"  {source:>30s}: {len(stems):>7d} / {n_avail:<7d} tiles used")

        kept = sum(counts)
        if kept:
            shares = ", ".join(f"{p} {n / kept:.0%}" for p, n in zip(prefixes, counts) if n)
            print(f"  total {kept} tiles  ({shares})")

    with open(out_root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "source", "name", "image", "gt"])
        w.writeheader()
        w.writerows(manifest)

    print(f"\nMerged dataset ready at {out_root}")
    print(f"Train with: --dataset_name {args.out_name} --tile_size {args.tile_size}")


if __name__ == "__main__":
    main()