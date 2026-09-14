# Building Extraction (Comesvil)

Semantic segmentation pipeline for extracting building footprints from aerial/satellite
imagery. A UNet-family model is trained on tiled RGB imagery with binary building masks,
then used to run inference either on pre-tiled evaluation data or directly on a large
georeferenced raster.

The project is organized around **two independent pipelines**:

| Pipeline | Script | Purpose |
|---|---|---|
| Training | `train.py` | Train a UNet variant on a tiled dataset, save `best_model.pth` / `last_model.pth` |
| Inference | `predict.py` | Load a trained model and run it — on dataset tiles or on a full raster — producing probability maps, masks, metrics and (optionally) building-footprint vectors |

Everything else in the repo either prepares data for training (`tiling.py`, `coco_utils.py`, `grid_tiling.py`, `check_balance.py`) or supplies building blocks the two pipelines import (`unet.py`, `dataset.py`, `losses.py`, `metrics.py`, `trainer.py`).

## Repository layout

```
src/
  models/
    unet.py            # UNet, UNetL, UNetLL architectures (same skip-connection design, 3 sizes)
  data/
    dataset.py          # MyDataset: paired image/mask tiles + optional extra channels
    tiling.py            # Slices full-size images into fixed-size tiles (+ optional augmentation)
    coco_utils.py         # Builds semantic/instance masks from COCO annotations
  training/
    losses.py            # BCE, Dice, Tversky, Focal, Focal-Tversky, weighted-BCE, combinations
    trainer.py            # Trainer: one train_one_epoch()/validate() loop, used by train.py
  evaluation/
    metrics.py            # iou_score / precision_score / recall_score + MetricsLogger (CSV)

scripts/  (suggested location — see "Where these files live" below)
  train.py               # Pipeline 1: training
  predict.py              # Pipeline 2: inference (tiles + raster subcommands)

check_balance.py          # Ad-hoc: plots the building-pixel-fraction histogram for a split
grid_tiling.py             # Ad-hoc: lat/lon grid + Sentinel Hub / Earth Engine download helpers
```

> **Where these files live:** the project docs are currently a flat list of scripts; the
> import paths used throughout (`from src.models.unet import ...`,
> `from src.data.dataset import MyDataset`, etc.) assume the `src/` package layout above,
> with `train.py` and `predict.py` run as `python -m scripts.train` / `python -m scripts.predict`
> from the repo root. Arrange the files into that layout (or adjust the imports) before running.

## Setup

```bash
pip install torch torchvision rasterio numpy pillow tqdm matplotlib albumentations shapely pycocotools opencv-python tifffile pandas pyproj
```

`shapely` is only needed for `predict.py raster --vectorize`. `pyproj` is only needed by
`grid_tiling.py`. A CUDA-capable GPU is optional but strongly recommended for training.

## 1. Data preparation

Expected on-disk layout for one dataset (e.g. `WHUBuildingDataset`):

```
<dataset_path>/<dataset_name>/
  tiles/<split>/images/*.tif        # source tiles, one image per file
  tiles/<split>/gt/*.tif            # matching binary masks (255 = building, 0 = background)
  tiles_<N>/<split>/images/*.tif    # fixed-size N×N tiles, produced by tiling.py
  tiles_<N>/<split>/gt/*.tif
  tiles_<N>/derived/<split>/<layer>/*.tif   # optional extra channels (see below)
```

- **`coco_utils.py`** — converts COCO-format polygon/RLE annotations into per-image binary
  masks (`--mode semantic`), or crops individual building instances with padding
  (`--mode instance` / `--mode stats`, the latter also plotting width/height/aspect-ratio
  distributions).
- **`tiling.py`** — slides a fixed-size window (`--tile_size`, `--stride`) over
  `tiles/<split>` to produce `tiles_<tile_size>/<split>`, with optional foreground-based
  filtering (`--skip_empty`, `--fg_threshold`) and optional augmentation
  (`--augment`, `--augment_factor`, via `albumentations`: rotation, flips, brightness/contrast).
- **`check_balance.py`** — quick histogram of the building-pixel fraction across a split's
  masks, to gauge class imbalance before picking a loss (edit the hardcoded paths at the
  top before running).
- **`grid_tiling.py`** — standalone helpers for building a lat/lon tile grid around a point
  or bounding box (UTM-based) and downloading imagery from Sentinel Hub / Earth Engine. It's
  exploratory/ad-hoc code (module-level script logic, hardcoded paths and API credentials) —
  **treat it as a reference, not something to run as-is**, and rotate any credentials
  currently hardcoded in it before sharing the repo.

## 2. Model

`src/models/unet.py` defines three sizes of the same encoder-decoder architecture (4
down/up stages, bilinear upsampling by default, batch norm + dropout in every conv block):

| Class | Base channels | Relative size |
|---|---|---|
| `UNet` | 64 | full |
| `UNetL` | 32 | ~1/4 params |
| `UNetLL` | 16 | ~1/16 params |

All three take `n_channels` (input channels — 3 for RGB, more if you append derived
channels) and `n_classes` (1 for binary building/background) and return raw logits.

## 3. Training — `train.py`

```bash
python -m scripts.train \
    --dataset_path /mnt/nas151/sar/Footprint/datasets \
    --dataset_name WHUBuildingDataset \
    --tile_size 256 --fixed_size \
    --batch_size 32 --epochs 50 --lr 1e-4 \
    --arch unetLL --loss bce \
    --extra_channels prob_mean   # optional, see below
```

Key points:

- Reads `<dataset_path>/<dataset_name>/tiles_<tile_size>/{train,val}/{images,gt}` via
  `MyDataset` (`src/data/dataset.py`), which pairs images and masks by sorted filename order
  and binarizes masks at `> 0`.
- `--extra_channels` appends additional single-band derived layers (e.g. `prob_mean`,
  `entropy` from an ensemble — see `predict.py` below) as extra input channels; `in_channels`
  is computed automatically from `3 + len(extra_channels)`.
- `--loss` selects from `bce`, `dice`, `focal`, `wbce`, `wbce_dice`, `tversky`,
  `focal_tversky` (all implemented in `src/training/losses.py`); `wbce` computes a
  positive-class weight from the full training set's foreground ratio before training starts.
- Training/validation loops live in `src/training/trainer.py` (mixed precision on CUDA);
  per-epoch loss/IoU/precision/recall are logged to `<output_dir>/<dataset_name>/<run_name>/metrics.csv`
  via `MetricsLogger`.
- Saves `best_model.pth` (lowest validation loss) and `last_model.pth`, plus a `config.json`
  snapshot of the run's hyperparameters, under
  `<output_dir>/<dataset_name>/<arch>_<loss>_dim<tile_size>_n<n_train>_bs<batch_size>/`.

## 4. Inference — `predict.py`

Single script, two subcommands. Both load a trained checkpoint the same way
(`--model_path`, `--arch`, `--in_channels`) and share `--threshold`, `--device`,
`--batch_size`, `--num_workers`, `--output_dir`, and `--normalize`.

**Normalization note:** `train.py` normalizes tiles to `[0, 1]` (via `ToTensor()`).
`predict.py` reproduces that by default (`--normalize auto`); only pass `--normalize none`
if a given checkpoint was actually trained on raw, unscaled pixel values.

### `tiles` — evaluate on a pre-tiled dataset split

```bash
python -m scripts.predict tiles \
    --dataset_root /mnt/nas151/sar/Footprint/datasets \
    --dataset_name WHUBuildingDataset --tile_size tiles_256 --split test \
    --model_path runs/WHUBuildingDataset/unetLL_bce_dim256_.../best_model.pth \
    --arch unetLL --output_dir experiments/whu_test --save_overlay
```

Runs the model over every tile in `<dataset_root>/<dataset_name>/<tile_size>/<split>/images`
(or a `--images` subset), writing per-tile probability maps and binary masks. If a matching
`gt/` folder exists, also computes dataset-level IoU/precision/recall/F1
(`metrics_summary.json`), a per-tile breakdown (`metrics_per_tile.csv`), and, with
`--save_overlay`, TP/FP/FN overlay images. `--extra_channels` mirrors `train.py`'s flag so a model trained with derived channels can be evaluated consistently.

### `raster` — building extraction on one large raster

```bash
python -m scripts.predict raster \
    --input /data/orthophoto.tif \
    --model_path runs/.../best_model.pth --arch unetLL \
    --output_dir experiments/ortho_extraction \
    --tile_size_px 256 --overlap 32 --vectorize
```

Tiles an arbitrarily large georeferenced raster internally with a sliding window, runs the
model per tile, and blends the overlapping predictions back into one full-resolution,
georeferenced probability map (`<stem>_probability.tif`) and binary mask
(`<stem>_mask.tif`) using edge-feathered weights (`--overlap` controls the blend band; `0`
disables blending). With `--vectorize`, the mask is polygonized into building footprints
(`<stem>_buildings.geojson`), optionally filtered by `--min_polygon_area`.

### Superseded scripts

`predict.py` replaces `test.py`, `infer_probability_maps.py` and `building_extraction.py`
(the latter never ran end-to-end — it referenced an undefined `transforms` object and an
incorrect `args.bbox`, and approximated pixel↔lat/lon conversion instead of using the
raster's actual affine transform). Keep them only for reference; new evaluation/inference
work should go through `predict.py`.

## Metrics

`src/evaluation/metrics.py` implements IoU, precision and recall as per-batch means over
sigmoid-thresholded predictions (`threshold=0.5` by default), used identically during
training (`Trainer`) and dataset-tile inference (`predict.py tiles`, which instead
aggregates confusion counts across the whole split for a single dataset-level score).
