import argparse
from pathlib import Path
import numpy as np
import torch
import rasterio
from tqdm import tqdm
from src.models.unet import UNet, UNetL, UNetLL

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# Example usage from the shell:
#
# python scripts/infer_probability_maps.py \
#   --dataset_root /mnt/nas151/sar/Footprint/datasets/WHUBuildingDataset/tiles_256 \
#   --runs_root /home/antoniocorvino/Projects/BuildingsExtraction/runs/WHUBuildingDataset \
#   --models unetLL_bce_dim256_n47088_bs32 unetLL_tversky_dim256_n47088_bs32 unetLL_focaltversky_dim256_n47088_bs32
#
# This will load each model from:
#   runs_root/<model_name>/best_model.pth
# and compute probability maps for all tiles in train/val/test.

def load_model(model_path, device):

    model = UNetLL(n_channels=3, n_classes=1)   # create architecture

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def predict(model, image_tensor):
    with torch.no_grad():
        out = model(image_tensor)
        out = torch.sigmoid(out)
    return out.squeeze().cpu().numpy()


def compute_entropy(p):
    eps = 1e-8
    return -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))


def compute_agreement(preds, threshold=0.5):
    binary = [(p > threshold).astype(np.float32) for p in preds]
    return np.mean(binary, axis=0)


def ensure_dirs(root, splits, layers):
    for s in splits:
        for l in layers:
            (root / s / l).mkdir(parents=True, exist_ok=True)


def save_tif(path, array, ref_profile):
    profile = ref_profile.copy()
    profile.update(dtype="float32", count=1)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype("float32"), 1)


def process_tile(tile_path, models, device):
    with rasterio.open(tile_path) as src:
        img = src.read().astype(np.float32)
        profile = src.profile

    img = torch.from_numpy(img).unsqueeze(0).to(device)

    preds = [predict(m, img) for m in models]

    mean_map = np.mean(preds, axis=0)
    std_map = np.std(preds, axis=0)
    entropy_map = compute_entropy(mean_map)
    agreement_map = compute_agreement(preds)

    return preds, mean_map, std_map, entropy_map, agreement_map, profile

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = Path(args.dataset_root)
    tiles_root = dataset_root
    derived_root = dataset_root / "derived"

    runs_root = Path(args.runs_root)

    model_paths = [runs_root / m / "best_model.pth" for m in args.models]

    splits = ["train", "val", "test"]
    layers = ["prob_mean", "prob_std", "entropy", "agreement"]

    ensure_dirs(derived_root, splits, layers)

    models = [load_model(p, device) for p in model_paths]
    # create directories for individual model predictions
    model_names = [Path(m).stem for m in args.models]

    for split in splits:

        for m in model_names:
            (derived_root / split / m).mkdir(parents=True, exist_ok=True)

        images_dir = tiles_root / split / "images"

        tiles = sorted(images_dir.glob("*.tif"))

        for tile in tqdm(tiles, desc=f"Processing {split}"):
            name = tile.name

            out_check = derived_root / split / "prob_mean" / name
            if out_check.exists():
                continue

            preds, mean_map, std_map, entropy_map, agreement_map, profile = process_tile(tile, models, device)

            # save individual model predictions
            for pred, model_name in zip(preds, model_names):
                save_tif(
                    derived_root / split / model_name / name,
                    pred,
                    profile
                )

            # save ensemble statistics
            save_tif(derived_root / split / "prob_mean" / name, mean_map, profile)
            save_tif(derived_root / split / "prob_std" / name, std_map, profile)
            save_tif(derived_root / split / "entropy" / name, entropy_map, profile)
            save_tif(derived_root / split / "agreement" / name, agreement_map, profile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Root folder containing tiles"
    )

    parser.add_argument(
        "--runs_root",
        required=True,
        help="Root folder containing training runs"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model run directories (best_model.pth will be loaded automatically)"
    )

    args = parser.parse_args()

    main(args)
