import argparse
from pathlib import Path
import numpy as np
import torch
import rasterio
from src.models.unet import UNet, UNetL, UNetLL

# python infer_probability_maps.py --dataset_root /mnt/nas151/sar/Footprint/datasets/WHUBuildingDataset --models model1.pth model2.pth model3.pth
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

    return mean_map, std_map, entropy_map, agreement_map, profile


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = Path(args.dataset_root)
    tiles_root = dataset_root
    derived_root = dataset_root / "derived"

    splits = ["train", "val", "test"]
    layers = ["prob_mean", "prob_std", "entropy", "agreement"]

    ensure_dirs(derived_root, splits, layers)

    models = [load_model(p, device) for p in args.models]
    
    for split in splits:
        images_dir = tiles_root / split / "images"

        tiles = sorted(images_dir.glob("*.tif"))

        for tile in tiles:
            mean_map, std_map, entropy_map, agreement_map, profile = process_tile(tile, models, device)

            name = tile.name

            save_tif(derived_root / split / "prob_mean" / name, mean_map, profile)
            save_tif(derived_root / split / "prob_std" / name, std_map, profile)
            save_tif(derived_root / split / "entropy" / name, entropy_map, profile)
            save_tif(derived_root / split / "agreement" / name, agreement_map, profile)

            print(f"Processed {tile}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Root folder containing tiles"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Paths to trained models"
    )

    args = parser.parse_args()

    main(args)
