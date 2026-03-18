import argparse
from pathlib import Path
import warnings

import numpy as np
import torch
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

from src.models.unet import UNet, UNetL, UNetLL

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def load_model(model_path, device):
    """Load UNetLL model weights and move to device."""
    model = UNetLL(n_channels=3, n_classes=1)

    # load weights on CPU first to avoid CUDA OOM during load
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    if device.type == "cuda":
        try:
            model.to(device)
            print(f"Loaded model {model_path} on CUDA")
        except RuntimeError as e:
            print(f"CUDA OOM while moving model {model_path} to GPU. Falling back to CPU: {e}")
            model.to("cpu")
    else:
        model.to("cpu")

    model.eval()
    return model


def predict(model, image_tensor):
    """
    Run forward pass and return probability map as float32 numpy array (H, W).
    image_tensor: torch.Tensor of shape (1, C, H, W)
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    # Optional: add normalization here if used during training
    # mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, -1, 1, 1)
    # std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, -1, 1, 1)
    # image_tensor = (image_tensor - mean) / std

    with torch.no_grad():
        logits = model(image_tensor)

    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return probs.astype(np.float32)


def save_tif(path, array, profile):
    """Save a single-band float32 GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    profile.update(dtype=rasterio.float32, count=1)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)


def ensure_dirs(derived_root, splits, layers, model_names):
    """Create output directories for per-model and ensemble layers."""
    for split in splits:
        for mn in model_names:
            (derived_root / split / mn).mkdir(parents=True, exist_ok=True)
        for layer in layers:
            (derived_root / split / layer).mkdir(parents=True, exist_ok=True)


def compute_entropy(p):
    """
    Binary entropy for probabilities p in [0,1].
    Returns float32 array with same shape as p.
    """
    eps = 1e-6
    return -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps)).astype(np.float32)


def compute_agreement(preds):
    """
    Compute model agreement as fraction of models predicting > 0.5.
    preds: list of arrays (H, W)
    Returns float32 array (H, W) in [0,1].
    """
    preds = np.stack(preds, axis=0)  # (M, H, W)
    binary = preds > 0.5
    return binary.mean(axis=0).astype(np.float32)


def infer_model_split(model, model_name, split, dataset_root, derived_root, max_tiles=None):
    """Run inference for a single model on a given split."""
    in_dir = dataset_root / split / "images"
    out_dir = derived_root / split / model_name

    tiles = sorted(in_dir.glob("*.tif"))
    if max_tiles is not None:
        tiles = tiles[:max_tiles]

    for tile in tqdm(tiles, desc=f"{model_name} {split}"):
        with rasterio.open(tile) as src:
            img = src.read().astype(np.float32)  # (C, H, W)
            profile = src.profile

        # Ensure 3 channels if model expects 3
        if img.shape[0] != 3:
            raise ValueError(f"Expected 3 channels, got {img.shape[0]} in {tile}")

        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, C, H, W)
        pred = predict(model, img_tensor)  # (H, W)
        save_tif(out_dir / tile.name, pred, profile)


def compute_ensemble_split(model_names, split, dataset_root, derived_root, max_tiles=None):
    """Compute ensemble statistics (mean, std, entropy, agreement) for a given split."""
    in_dir = dataset_root / split / "images"
    tiles = sorted(in_dir.glob("*.tif"))
    if max_tiles is not None:
        tiles = tiles[:max_tiles]

    for tile in tqdm(tiles, desc=f"ensemble {split}"):
        preds = []
        profile = None

        for mn in model_names:
            pred_path = derived_root / split / mn / tile.name
            with rasterio.open(pred_path) as src:
                if profile is None:
                    profile = src.profile
                preds.append(src.read(1).astype(np.float32))

        mean_map = np.mean(preds, axis=0).astype(np.float32)
        std_map = np.std(preds, axis=0).astype(np.float32)
        entropy_map = compute_entropy(mean_map)
        agreement_map = compute_agreement(preds)

        save_tif(derived_root / split / "prob_mean" / tile.name, mean_map, profile)
        save_tif(derived_root / split / "prob_std" / tile.name, std_map, profile)
        save_tif(derived_root / split / "entropy" / tile.name, entropy_map, profile)
        save_tif(derived_root / split / "agreement" / tile.name, agreement_map, profile)


def main(args):
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    dataset_root = Path(args.dataset_root)
    derived_root = dataset_root / "derived"
    runs_root = Path(args.runs_root)

    # args.models are run directories; model_names are their stems
    model_names = [Path(m).stem for m in args.models]

    splits = ["train", "val", "test"]
    layers = ["prob_mean", "prob_std", "entropy", "agreement"]

    ensure_dirs(derived_root, splits, layers, model_names)

    # Per-model inference
    for model_name, model_dir in zip(model_names, args.models):
        model_path = runs_root / model_dir / "best_model.pth"
        model = load_model(model_path, device)

        for split in splits:
            infer_model_split(model, model_name, split, dataset_root, derived_root, args.max_tiles)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Ensemble statistics
    for split in splits:
        compute_ensemble_split(model_names, split, dataset_root, derived_root, args.max_tiles)


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

    parser.add_argument(
        "--max_tiles",
        type=int,
        default=None,
        help="Process only the first N tiles of each split (useful for quick tests)"
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )

    args = parser.parse_args()
    main(args)
