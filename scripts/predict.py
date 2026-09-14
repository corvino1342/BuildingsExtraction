"""
Unified inference / building-extraction pipeline.

This project now has two separate, independent pipelines:

  1. train.py    -> trains a model and writes runs/<dataset>/<run_name>/best_model.pth
  2. predict.py  -> loads a trained model and runs inference (THIS FILE)

predict.py replaces test.py, infer_probability_maps.py and building_extraction.py
with a single script that has two subcommands:

  * `tiles`  - evaluate/run a trained model on a folder of already-cut tiles
               (a dataset split such as .../tiles_256/test/images), optionally
               scoring against ground truth. This covers what test.py and
               infer_probability_maps.py did.

  * `raster` - run true "building extraction" on one arbitrarily large,
               georeferenced raster (an orthophoto, a satellite scene, ...).
               The raster is tiled internally with a sliding window, each tile
               is passed through the model, and the per-tile probabilities are
               blended back into a single full-resolution, georeferenced
               probability map and binary mask. Optionally the mask is
               vectorized into building-footprint polygons (GeoJSON). This
               replaces the broken building_extraction.py (which mixed up
               lat/lon-to-pixel math with an undefined `transforms` object and
               never actually ran end to end).

IMPORTANT FIX vs. the previous scripts:
  train.py builds tiles with PIL + torchvision.transforms.ToTensor(), which
  scales uint8 images from [0, 255] to [0, 1] before they ever reach the
  model. infer_probability_maps.py and building_extraction.py instead read
  tiles with rasterio and fed the model raw pixel values (no /255 scaling) -
  a silent train/inference mismatch for any 8-bit imagery. This script
  normalizes tiles and raster windows the same way training does by default
  (`--normalize auto`); override with `--normalize none` only if your model
  was actually trained on raw, unscaled pixel values.

Usage
-----
Evaluate a trained model on a pre-tiled dataset split (mirrors test.py):

    python -m scripts.predict tiles \\
        --dataset_root /mnt/nas151/sar/Footprint/datasets \\
        --dataset_name WHUBuildingDataset --tile_size tiles_256 --split test \\
        --model_path runs/WHUBuildingDataset/unetLL_bce_dim256_.../best_model.pth \\
        --arch unetLL --output_dir experiments/whu_test --save_overlay

Run building extraction on one large georeferenced raster:

    python -m scripts.predict raster \\
        --input /data/orthophoto.tif \\
        --model_path runs/.../best_model.pth --arch unetLL \\
        --output_dir experiments/ortho_extraction \\
        --tile_size_px 256 --overlap 32 --vectorize
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.windows import Window
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches

from src.models.unet import UNet, UNetL, UNetLL

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
ARCHS = {"unet": UNet, "unetL": UNetL, "unetLL": UNetLL}


# ==================================================
# Model
# ==================================================

def build_model(arch, in_channels=3, n_classes=1):
    if arch not in ARCHS:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {list(ARCHS)}")
    return ARCHS[arch](n_channels=in_channels, n_classes=n_classes)


def resolve_device(device_str):
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        device = torch.device("cpu")
    return device


def load_model(model_path, arch, in_channels, device):
    model = build_model(arch, in_channels=in_channels)

    # load on CPU first to avoid CUDA OOM while loading, mirroring infer_probability_maps.py
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    if device.type == "cuda":
        try:
            model.to(device)
        except RuntimeError as e:
            print(f"CUDA OOM while moving model to GPU, falling back to CPU: {e}")
            device = torch.device("cpu")
            model.to(device)
    else:
        model.to(device)

    model.eval()
    print(f"Loaded {arch} ({in_channels} input channel(s)) from {model_path} on {device}")
    return model, device


@torch.no_grad()
def predict_batch(model, tensor_batch, device):
    """tensor_batch: (B, C, H, W) float32. Returns probabilities (B, H, W) float32 numpy."""
    tensor_batch = tensor_batch.to(device)
    probs = torch.sigmoid(model(tensor_batch))
    if probs.dim() == 4:
        probs = probs.squeeze(1)
    return probs.cpu().numpy().astype(np.float32)


def normalize_array(arr, src_dtype, mode):
    """
    Reproduce the normalization used at training time (see module docstring).
    arr: float32 array already read from disk (still in its native pixel range).
    """
    if mode == "none":
        return arr
    if mode == "divide255":
        return arr / 255.0
    # "auto": scale integer imagery (typically uint8) the same way ToTensor() would
    if np.issubdtype(np.dtype(src_dtype), np.integer):
        return arr / 255.0
    return arr


# ==================================================
# Shared I/O helpers
# ==================================================

def find_matching_gt(gt_dir, name):
    if gt_dir is None or not gt_dir.exists():
        return None
    for ext in SUPPORTED_EXTENSIONS:
        for candidate in (gt_dir / f"{name}{ext}", gt_dir / f"{name}{ext.upper()}"):
            if candidate.exists():
                return candidate
    return None


def save_prob_geotiff(out_path, array, profile):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)


def save_binary_mask(out_path, mask_bool, profile=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = mask_bool.astype(np.uint8) * 255
    if profile is not None:
        mask_profile = profile.copy()
        mask_profile.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(out_path, "w", **mask_profile) as dst:
            dst.write(data, 1)
    else:
        Image.fromarray(data, mode="L").save(out_path)


def save_overlay(out_path, image, pred_mask, true_mask=None, title=""):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_np = np.array(image)
    pred = pred_mask.astype(bool)
    overlay = np.zeros((*img_np.shape[:2], 4), dtype=np.float32)
    legend = []

    if true_mask is not None:
        gt = true_mask.astype(bool)
        tp, fp, fn = pred & gt, pred & ~gt, ~pred & gt
        overlay[tp] = [0, 1, 0, 0.4]
        overlay[fp] = [0, 0, 1, 0.4]
        overlay[fn] = [1, 0, 0, 0.4]
        legend = [
            patches.Patch(facecolor="green", label="True Positive"),
            patches.Patch(facecolor="blue", label="False Positive"),
            patches.Patch(facecolor="red", label="False Negative"),
        ]
    else:
        overlay[pred] = [1, 0, 0, 0.4]
        legend = [patches.Patch(facecolor="red", label="Prediction")]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_np)
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    ax.legend(handles=legend, loc="upper left", fontsize=10)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def confusion_counts(pred_bin, gt_bin):
    tp = int(np.logical_and(pred_bin, gt_bin).sum())
    fp = int(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = int(np.logical_and(~pred_bin, gt_bin).sum())
    tn = int(np.logical_and(~pred_bin, ~gt_bin).sum())
    return tp, fp, fn, tn


# ==================================================
# Pipeline 1: pre-tiled dataset folder (replaces test.py / infer_probability_maps.py)
# ==================================================

class TileFolderDataset(Dataset):
    """Reads pre-cut tiles from a dataset-style images/ folder, with optional
    extra derived-channel folders aligned by filename (same convention as
    train.py's --extra_channels)."""

    def __init__(self, images_dir, extra_dirs=None, names=None, normalize_mode="auto"):
        self.images_dir = Path(images_dir)
        self.extra_dirs = [Path(d) for d in (extra_dirs or [])]
        self.normalize_mode = normalize_mode

        all_paths = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        self._path_by_name = {p.stem: p for p in all_paths}

        if names:
            missing = [n for n in names if n not in self._path_by_name]
            if missing:
                raise FileNotFoundError(f"Tiles not found in {self.images_dir}: {missing}")
            self.names = list(names)
        else:
            self.names = sorted(self._path_by_name.keys())

    def __len__(self):
        return len(self.names)

    def image_path(self, name):
        return self._path_by_name[name]

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = self._path_by_name[name]

        if img_path.suffix.lower() in {".tif", ".tiff"}:
            with rasterio.open(img_path) as src:
                arr = src.read().astype(np.float32)
                src_dtype = src.dtypes[0]
                profile = src.profile
        else:
            pil_arr = np.array(Image.open(img_path).convert("RGB"))
            arr = pil_arr.transpose(2, 0, 1).astype(np.float32)
            src_dtype = pil_arr.dtype
            profile = None

        arr = normalize_array(arr, src_dtype, self.normalize_mode)
        tensor = torch.from_numpy(arr)

        extras = []
        for extra_dir in self.extra_dirs:
            extra_path = extra_dir / img_path.name
            with rasterio.open(extra_path) as src:
                extras.append(src.read(1).astype(np.float32))
        if extras:
            tensor = torch.cat([tensor, torch.from_numpy(np.stack(extras))], dim=0)

        return name, tensor, profile


def collate_tiles(batch):
    names, tensors, profiles = zip(*batch)
    return list(names), torch.stack(tensors, dim=0), list(profiles)


def run_tiles_pipeline(args, model, device):
    dataset_root = Path(args.dataset_root)
    base = dataset_root / args.dataset_name / args.tile_size / args.split
    image_dir = base / "images"
    gt_dir = base / "gt"

    if not image_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {image_dir}")

    extra_dirs = [
        dataset_root / args.dataset_name / args.tile_size / "derived" / args.split / name
        for name in args.extra_channels
    ]

    dataset = TileFolderDataset(
        image_dir, extra_dirs=extra_dirs, names=args.images, normalize_mode=args.normalize
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_tiles,
    )

    out_dir = Path(args.output_dir)
    prob_dir, mask_dir, overlay_dir = out_dir / "probability", out_dir / "mask", out_dir / "overlay"

    has_gt = gt_dir.exists()
    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_tile_rows = []

    for names, tensors, profiles in tqdm(loader, desc=f"Inference [{args.split}]"):
        probs = predict_batch(model, tensors, device)

        for name, prob, profile in zip(names, probs, profiles):
            pred_bin = prob > args.threshold

            if profile is not None:
                save_prob_geotiff(prob_dir / f"{name}.tif", prob, profile)
                save_binary_mask(mask_dir / f"{name}.tif", pred_bin, profile)
            else:
                np.save(prob_dir / f"{name}.npy", prob)
                save_binary_mask(mask_dir / f"{name}.png", pred_bin)

            if has_gt:
                gt_path = find_matching_gt(gt_dir, name)
                if gt_path is not None:
                    gt_bin = np.array(Image.open(gt_path).convert("L")) > 0
                    tp, fp, fn, _ = confusion_counts(pred_bin, gt_bin)
                    totals["tp"] += tp
                    totals["fp"] += fp
                    totals["fn"] += fn
                    per_tile_rows.append({
                        "name": name,
                        "iou": tp / max(tp + fp + fn, 1),
                        "precision": tp / max(tp + fp, 1),
                        "recall": tp / max(tp + fn, 1),
                    })

                    if args.save_overlay:
                        image = Image.open(dataset.image_path(name)).convert("RGB")
                        save_overlay(
                            overlay_dir / f"{name}.png", image, pred_bin, gt_bin,
                            title=f"{args.arch} | {name}",
                        )

    if has_gt and per_tile_rows:
        tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        overall = {
            "iou": tp / max(tp + fp + fn, 1),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "metrics_per_tile.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "iou", "precision", "recall"])
            writer.writeheader()
            writer.writerows(per_tile_rows)
        with open(out_dir / "metrics_summary.json", "w") as f:
            json.dump(overall, f, indent=2)

        print(
            f"\nDataset-level metrics -> IoU: {overall['iou']:.4f}  "
            f"Precision: {overall['precision']:.4f}  Recall: {overall['recall']:.4f}  "
            f"F1: {overall['f1']:.4f}"
        )

    print(f"\nDone. Outputs written to {out_dir}")


# ==================================================
# Pipeline 2: large georeferenced raster (replaces building_extraction.py)
# ==================================================

def compute_starts(dim, tile_size, stride):
    """Sliding-window start offsets along one axis, guaranteed to cover [0, dim)."""
    if dim <= tile_size:
        return [0]
    last_start = dim - tile_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def make_tile_weight(tile_size, overlap):
    """Feathering weight for blending overlapping tiles back into the mosaic.
    Ramps up across the overlap band at each edge (never reaching exactly 0,
    so true image-border pixels - covered by only one tile - still get
    nonzero weight); uniform (no seams to hide) when overlap is 0."""
    if overlap <= 0:
        return np.ones((tile_size, tile_size), dtype=np.float32)
    ramp = np.ones(tile_size, dtype=np.float32)
    # Strictly positive ramp (never touches 0): a pixel at the true image
    # border only ever gets a contribution from one tile, so if the ramp hit
    # zero there the mosaic would divide 0/0 at every image edge.
    taper = (np.arange(overlap, dtype=np.float32) + 1.0) / overlap
    ramp[:overlap] = taper
    ramp[tile_size - overlap:] = taper[::-1]
    return np.outer(ramp, ramp)


def accumulate(mosaic_sum, mosaic_weight, prob_tile, weight_tile, window, img_h, img_w):
    row_off, col_off = int(window.row_off), int(window.col_off)
    h, w = weight_tile.shape

    row_start, col_start = max(row_off, 0), max(col_off, 0)
    row_end, col_end = min(row_off + h, img_h), min(col_off + w, img_w)
    if row_end <= row_start or col_end <= col_start:
        return

    tr0, tc0 = row_start - row_off, col_start - col_off
    tr1, tc1 = tr0 + (row_end - row_start), tc0 + (col_end - col_start)

    mosaic_sum[row_start:row_end, col_start:col_end] += (
        prob_tile[tr0:tr1, tc0:tc1] * weight_tile[tr0:tr1, tc0:tc1]
    )
    mosaic_weight[row_start:row_end, col_start:col_end] += weight_tile[tr0:tr1, tc0:tc1]


class RasterWindowDataset(Dataset):
    """Slides a fixed-size window (with overlap) over one arbitrarily large
    georeferenced raster. Opens the file per-read so it's safe with
    num_workers > 0."""

    def __init__(self, raster_path, tile_size, overlap, in_channels, normalize_mode="auto"):
        self.raster_path = str(raster_path)
        self.tile_size = tile_size
        self.in_channels = in_channels
        self.normalize_mode = normalize_mode

        stride = tile_size - overlap
        if stride <= 0:
            raise ValueError("--overlap must be smaller than --tile_size_px")

        with rasterio.open(self.raster_path) as src:
            self.height, self.width = src.height, src.width
            self.src_dtype = src.dtypes[0]
            if src.count < in_channels:
                raise ValueError(
                    f"Raster has {src.count} band(s) but the model expects "
                    f"{in_channels} input channel(s) (--in_channels)"
                )

        row_starts = compute_starts(self.height, tile_size, stride)
        col_starts = compute_starts(self.width, tile_size, stride)
        self.windows = [Window(c, r, tile_size, tile_size) for r in row_starts for c in col_starts]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        with rasterio.open(self.raster_path) as src:
            arr = src.read(
                indexes=list(range(1, self.in_channels + 1)),
                window=window,
                boundless=True,
                fill_value=0,
            ).astype(np.float32)
        arr = normalize_array(arr, self.src_dtype, self.normalize_mode)
        return idx, torch.from_numpy(arr)


def vectorize_mask(binary_mask_255, transform, crs, min_area, out_path):
    try:
        from rasterio.features import shapes as rio_shapes
        from shapely.geometry import shape, mapping
    except ImportError as e:
        raise ImportError(
            "Vectorization needs shapely (pip install shapely)."
        ) from e

    mask_bool = binary_mask_255 > 0
    features = []
    for geom, value in rio_shapes(binary_mask_255, mask=mask_bool, transform=transform):
        if value == 0:
            continue
        poly = shape(geom)
        if poly.area < min_area:
            continue
        features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {"area": poly.area},
        })

    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": str(crs)}} if crs else None,
        "features": features,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(geojson, f)

    return len(features)


def run_raster_pipeline(args, model, device):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    with rasterio.open(args.input) as src:
        profile = src.profile
        height, width = src.height, src.width

    tile_size, overlap = args.tile_size_px, args.overlap

    dataset = RasterWindowDataset(
        args.input, tile_size, overlap, args.in_channels, normalize_mode=args.normalize
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    mosaic_sum = np.zeros((height, width), dtype=np.float32)
    mosaic_weight = np.zeros((height, width), dtype=np.float32)
    weight_tile = make_tile_weight(tile_size, overlap)

    for idxs, tensors in tqdm(loader, desc=f"Tiling {stem}", total=len(loader)):
        probs = predict_batch(model, tensors, device)
        for i, idx in enumerate(idxs.tolist()):
            accumulate(mosaic_sum, mosaic_weight, probs[i], weight_tile, dataset.windows[idx], height, width)

    prob_mosaic = mosaic_sum / np.maximum(mosaic_weight, 1e-6)

    prob_path = out_dir / f"{stem}_probability.tif"
    save_prob_geotiff(prob_path, prob_mosaic, profile)
    print(f"\nSaved probability map -> {prob_path}")

    binary_mask = prob_mosaic > args.threshold
    mask_path = out_dir / f"{stem}_mask.tif"
    save_binary_mask(mask_path, binary_mask, profile)
    print(f"Saved binary mask -> {mask_path}")

    if args.vectorize:
        vector_path = out_dir / f"{stem}_buildings.geojson"
        n_polygons = vectorize_mask(
            (binary_mask.astype(np.uint8) * 255), profile["transform"], profile["crs"],
            args.min_polygon_area, vector_path,
        )
        print(f"Saved {n_polygons} building polygon(s) -> {vector_path}")


# ==================================================
# CLI
# ==================================================

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Building extraction inference pipeline (see module docstring for examples).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model_path", type=str, required=True, help="Path to a *.pth state dict.")
    common.add_argument("--arch", type=str, choices=list(ARCHS), default="unetLL")
    common.add_argument("--in_channels", type=int, default=3)
    common.add_argument("--threshold", type=float, default=0.5)
    common.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                         choices=["cuda", "cpu"])
    common.add_argument("--output_dir", type=str, required=True)
    common.add_argument("--batch_size", type=int, default=8)
    common.add_argument("--num_workers", type=int, default=4)
    common.add_argument("--normalize", type=str, choices=["auto", "divide255", "none"], default="auto",
                         help="Pixel scaling applied before the model, see module docstring.")

    tiles_p = subparsers.add_parser(
        "tiles", parents=[common],
        help="Run inference on a folder of pre-cut dataset tiles (optionally scored against GT).",
    )
    tiles_p.add_argument("--dataset_root", type=str, required=True)
    tiles_p.add_argument("--dataset_name", type=str, required=True)
    tiles_p.add_argument("--tile_size", type=str, required=True, help="Subfolder name, e.g. tiles_256")
    tiles_p.add_argument("--split", type=str, default="test")
    tiles_p.add_argument("--extra_channels", type=str, nargs="*", default=[],
                          help="Derived extra channels to append, same convention as train.py")
    tiles_p.add_argument("--images", type=str, nargs="+", default=None,
                          help="Optional subset of tile names (without extension). Default: all tiles.")
    tiles_p.add_argument("--save_overlay", action="store_true",
                          help="Save TP/FP/FN overlay PNGs for tiles that have ground truth (slower).")

    raster_p = subparsers.add_parser(
        "raster", parents=[common],
        help="Run building extraction end-to-end on one large georeferenced raster.",
    )
    raster_p.add_argument("--input", type=str, required=True, help="Path to a georeferenced raster (e.g. GeoTIFF).")
    raster_p.add_argument("--tile_size_px", type=int, default=256)
    raster_p.add_argument("--overlap", type=int, default=32,
                           help="Overlap in pixels between adjacent tiles; blended to avoid seams.")
    raster_p.add_argument("--vectorize", action="store_true",
                           help="Also export building footprints as a GeoJSON (requires shapely).")
    raster_p.add_argument("--min_polygon_area", type=float, default=0.0,
                           help="Drop vectorized polygons smaller than this, in CRS units^2 "
                                "(e.g. m^2 for a projected CRS).")

    return parser


def main():
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)

    in_channels = args.in_channels
    if args.command == "tiles" and args.extra_channels:
        auto_in_channels = 3 + len(args.extra_channels)
        if in_channels != auto_in_channels:
            print(
                f"Setting in_channels={auto_in_channels} "
                f"(RGB + {len(args.extra_channels)} extra channel(s): {', '.join(args.extra_channels)})"
            )
            in_channels = auto_in_channels

    model, device = load_model(args.model_path, args.arch, in_channels, device)

    if args.command == "tiles":
        run_tiles_pipeline(args, model, device)
    else:
        run_raster_pipeline(args, model, device)


if __name__ == "__main__":
    main()
