import argparse
import time
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import patches

from src.models.unet import UNet, UNetL, UNetLL
# from src.models.deeplabv3 import DeepLabV3

# Supported image extensions
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Test a single model on images with ground truth.")

    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root directory containing the dataset.")
    parser.add_argument("--eval_dataset", type=str, required=True,
                        help="Name of the dataset to evaluate on.")
    parser.add_argument("--tile_size", type=str, default="tiles",
                        help="Subdirectory under eval_dataset containing the tiles.")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate on (e.g., test, val).")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model's best_model.pth weights.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model architecture (e.g., unetLL, deeplabv3).")

    parser.add_argument("--images", nargs="+",
                        help="Optional specific image names (without extension) to process.")
    parser.add_argument("--all_images", action="store_true",
                        help="Process all images in the dataset.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for converting probabilities to binary masks.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"],
                        help="Device to use for inference.")

    return parser.parse_args()

# --------------------------------------------------
# Model factory
# --------------------------------------------------

def build_model(model_name):
    if model_name.startswith("unetLL"):
        return UNetLL(3, 1)
    if model_name.startswith("unetL"):
        return UNetL(3, 1)
    if model_name.startswith("unet"):
        return UNet(3, 1)
    if model_name.startswith("deeplabv3"):
        return DeepLabV3(num_classes=1)
    raise ValueError(f"Unknown model type: {model_name}")

# --------------------------------------------------
# Utilities
# --------------------------------------------------

def save_mask(out_dir, image_name, pred_mask):
    pred_uint8 = (pred_mask > 0).astype(np.uint8) * 255
    pred_img = Image.fromarray(pred_uint8, mode="L")
    pred_img.save(out_dir / f"{image_name}.tif")

def save_overlay(out_path, image, pred_mask, true_mask, title):
    img_np = np.array(image)
    pred = pred_mask.astype(bool)
    gt = true_mask.astype(bool) if true_mask is not None else None

    overlay = np.zeros((img_np.shape[0], img_np.shape[1], 4), dtype=np.float32)

    legend_elements = []

    if gt is not None:
        tp = np.logical_and(pred, gt)
        fp = np.logical_and(pred, ~gt)
        fn = np.logical_and(~pred, gt)

        overlay[tp] = [0, 1, 0, 0.4]   # Green
        overlay[fp] = [0, 0, 1, 0.4]   # Blue
        overlay[fn] = [1, 0, 0, 0.4]   # Red

        legend_elements = [
            patches.Patch(facecolor='green', label='True Positive'),
            patches.Patch(facecolor='blue', label='False Positive'),
            patches.Patch(facecolor='red', label='False Negative'),
        ]
    else:
        overlay[pred] = [1, 0, 0, 0.4]
        legend_elements = [
            patches.Patch(facecolor='red', label='Prediction'),
        ]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_np)
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    plt.savefig(out_path)
    plt.close()

def save_three_plot(out_path, model_title, image, pred_mask, true_mask):
    if true_mask is None:
        ncols = 2
    else:
        ncols = 3

    fig, axes = plt.subplots(1, ncols, figsize=(18, 6), constrained_layout=True)

    # 1. Input image
    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontsize=14)
    axes[0].axis("off")

    # 2. Input + predicted mask
    axes[1].imshow(image)
    axes[1].imshow(pred_mask, cmap="jet", alpha=0.5)
    axes[1].contour(pred_mask, colors="white", linewidths=0.5)
    axes[1].set_title(f"{model_title}\nPrediction", fontsize=14)
    axes[1].axis("off")

    if true_mask is not None:
        axes[2].imshow(image, alpha=0.8)
        axes[2].imshow(true_mask, cmap="Blues", alpha=0.7)
        axes[2].imshow(pred_mask, cmap="Reds", alpha=0.5)
        axes[2].set_title("Prediction vs Ground Truth", fontsize=14)
        axes[2].axis("off")

        legend_elements = [
            patches.Patch(color='darkblue', label='Ground Truth'),
            patches.Patch(color='darkred', label='Prediction'),
        ]

        axes[2].legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0,
            fontsize=10
        )

    plt.savefig(out_path)
    plt.close()

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")

    dataset_root = Path(args.dataset_root)
    model_path = Path(args.model_path)

    image_dir = dataset_root / args.eval_dataset / args.tile_size / args.split / "images"
    gt_dir = dataset_root / args.eval_dataset / args.tile_size / args.split / "gt"

    if not image_dir.exists():
        raise ValueError(f"Images directory not found: {image_dir}")

    if args.all_images:
        image_paths = sorted(
            [p for p in image_dir.iterdir()
             if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    else:
        if not args.images:
            raise ValueError("Use --all_images or provide --images")
        image_paths = []
        for name in args.images:
            found = False
            for ext in SUPPORTED_EXTENSIONS:
                candidate = image_dir / f"{name}{ext}"
                if candidate.exists():
                    image_paths.append(candidate)
                    found = True
                    break
                candidate_upper = image_dir / f"{name}{ext.upper()}"
                if candidate_upper.exists():
                    image_paths.append(candidate_upper)
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"No file found for {name} with supported extensions")

    print(f"\nEvaluating on: {args.eval_dataset} ({args.split})")
    print(f"Images: {len(image_paths)}")
    print(f"Device: {device}\n")

    transform = transforms.ToTensor()

    # Load model
    model = build_model(args.model_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    result_dir = (
        Path("experiments")
        / f"eval_on_{args.eval_dataset}"
        / f"split_{args.split}"
        / f"{args.model_name}_trained_on_{model_path.parent.name}"
    )
    (result_dir / "predict").mkdir(parents=True, exist_ok=True)
    (result_dir / "overlay").mkdir(parents=True, exist_ok=True)
    (result_dir / "threeplot").mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # Inference loop
    for img_path in image_paths:
        image_name = img_path.stem
        image = Image.open(img_path).convert("RGB")

        gt_path = None
        for ext in SUPPORTED_EXTENSIONS:
            candidate = gt_dir / f"{image_name}{ext}"
            if candidate.exists():
                gt_path = candidate
                break
            candidate_upper = gt_dir / f"{image_name}{ext.upper()}"
            if candidate_upper.exists():
                gt_path = candidate_upper
                break

        true_mask = Image.open(gt_path).convert("L") if gt_path else None

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            prob = torch.sigmoid(logits)
            pred = (prob > args.threshold).float()

        pred_np = pred.squeeze().cpu().numpy()
        gt_np = np.array(true_mask) > 0 if true_mask else None

        save_mask(result_dir / "predict", image_name, pred_np)

        if gt_np is not None:
            save_overlay(
                result_dir / "overlay" / f"{image_name}.tif",
                image,
                pred_np,
                gt_np,
                f"{args.model_name} | trained on {args.eval_dataset}"
            )

            save_three_plot(
                result_dir / "threeplot" / f"{image_name}.tif",
                f"{args.model_name} | trained on {args.eval_dataset}",
                image,
                pred_np,
                gt_np
            )

        print(f"Processed {image_name}")

    print(f"\nTotal inference time: {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    main()