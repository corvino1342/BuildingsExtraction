import argparse
import time
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from src.models.unet import UNet, UNetL, UNetLL
from src.models.deeplabv3 import DeepLabV3


# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser("Flexible cross-dataset inference")

    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--tile_size", type=str, default="tiles")
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--runs_root", type=str, required=True)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Format: TRAIN_DATASET/MODEL_NAME")

    parser.add_argument("--all_images", action="store_true")
    parser.add_argument("--images", nargs="+",
                        help="Optional specific image names (without extension)")

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps"])

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
    pred_img.save(out_dir / "predict" / f"{image_name}.tif")


def save_overlay(out_path, image, pred_mask, true_mask, title):
    img_np = np.array(image)
    pred = pred_mask.astype(bool)
    gt = true_mask.astype(bool) if true_mask is not None else None

    overlay = np.zeros((img_np.shape[0], img_np.shape[1], 4), dtype=np.float32)

    if gt is not None:
        tp = np.logical_and(pred, gt)
        fp = np.logical_and(pred, ~gt)
        fn = np.logical_and(~pred, gt)

        overlay[tp] = [0, 1, 0, 0.4]
        overlay[fp] = [0, 0, 1, 0.4]
        overlay[fn] = [1, 0, 0, 0.4]
    else:
        overlay[pred] = [1, 0, 0, 0.4]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_np)
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
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

    plt.savefig(out_path)
    plt.close()

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    args = parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu"
    )

    dataset_root = Path(args.dataset_root)
    runs_root = Path(args.runs_root)

    image_dir = dataset_root / args.eval_dataset / args.tile_size / args.split / "images"
    gt_dir = dataset_root / args.eval_dataset / args.tile_size / args.split / "gt"

    if not image_dir.exists():
        raise ValueError(f"Images directory not found: {image_dir}")

    if args.all_images:
        image_paths = sorted(image_dir.glob("*"))
    else:
        if not args.images:
            raise ValueError("Use --all_images or provide --images")
        image_paths = [image_dir / f"{name}.tif" for name in args.images]

    print(f"\nEvaluating on: {args.eval_dataset} ({args.split})")
    print(f"Images: {len(image_paths)}")
    print(f"Device: {device}\n")

    transform = transforms.ToTensor()

    models = {}

    # Load models
    for run_id in args.runs:
        train_dataset, model_name = run_id.split("/")

        model = build_model(model_name).to(device)

        ckpt_path = runs_root / train_dataset / model_name / "best_model.pth"

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        model.eval()

        models[run_id] = model
        print(f"Loaded {run_id}")

    total_start = time.time()

    # Inference loop
    for img_path in image_paths:

        image = Image.open(img_path).convert("RGB")
        image_name = img_path.stem

        gt_path = gt_dir / img_path.name
        true_mask = Image.open(gt_path).convert("L") if gt_path.exists() else None

        input_tensor = transform(image).unsqueeze(0).to(device)

        for run_id, model in models.items():

            train_dataset, model_name = run_id.split("/")

            result_dir = (
                Path("experiments")
                / f"eval_on_{args.eval_dataset}"
                / f"split_{args.split}"
                / f"{model_name}_trained_on_{train_dataset}"
            )

            (result_dir / "predict").mkdir(parents=True, exist_ok=True)
            (result_dir / "overlay").mkdir(parents=True, exist_ok=True)
            (result_dir / "threeplot").mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                logits = model(input_tensor)
                prob = torch.sigmoid(logits)
                pred = (prob > args.threshold).float()

            pred_np = pred.squeeze().cpu().numpy()
            gt_np = np.array(true_mask) > 0 if true_mask else None

            save_mask(result_dir, image_name, pred_np)

            save_overlay(
                result_dir / "overlay" / f"{image_name}.tif",
                image,
                pred_np,
                gt_np,
                f"{model_name} | trained on {train_dataset}"
            )
            
            save_three_plot(
                result_dir / "threeplot" / f"{image_name}.tif",
                f"{model_name} | trained on {train_dataset}",
                image,
                pred_np,
                gt_np
            )

        print(f"Processed {image_name}")

    print(f"\nTotal inference time: {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()