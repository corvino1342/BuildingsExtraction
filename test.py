from matplotlib.pyplot import title
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import patches

import os
import time
import argparse
import torch
import numpy as np

from unet import UNet, UNetL, UNetLL

def MaskPredict(dataset_name, model_name, image_name, pred_mask, ref_mask):
    """
    Save predicted mask with same size, format, and aspect as reference mask.
    """
    # Ensure binary mask → uint8 (0 or 255)
    pred_mask_uint8 = (pred_mask > 0).astype(np.uint8) * 255

    # Create PIL image
    pred_img = Image.fromarray(pred_mask_uint8, mode="L")

    # Safety check: force same size as reference
    if pred_img.size != ref_mask.size:
        pred_img = pred_img.resize(ref_mask.size, resample=Image.NEAREST)

    os.makedirs(f'/home/antoniocorvino/Projects/BuildingsExtraction/runs/{dataset_name}/{model_name}/predict/', exist_ok=True)
    
    save_path = (
        f"/home/antoniocorvino/Projects/BuildingsExtraction/runs/{dataset_name}/{model_name}/predict/{image_name}.tif"
    )

    pred_img.save(save_path)

def ShortModelName(model_name):
    """
    Create a compact, human-readable model identifier for plots.
    Example:
    unet_IAD_WBCE_lr0p0001_n28000_dim256x256_bs32
    -> UNet | IAD | WBCE | 1e-04 | 256 | bs32
    """
    parts = model_name.split('_')

    arch = parts[0].upper() if parts else "MODEL"

    dt = "IAD" if "IAD" in parts else "MBD" if "MBD" in parts else "DATASET"

    loss = "WBCE" if "WBCE" in parts else "BCE" if "BCE" in parts else "WBCE+Dice" if "WBCEplusDL" in parts else "BCE+Dice" if "BCEplusDL" in parts else "LOSS"

    lr_part = next((p for p in parts if p.startswith("lr")), None)

    if lr_part is None:
        lr = "lr_var"
    else:
        lr_raw = lr_part[2:]  # strip 'lr'

        if 'e' in lr_raw:
            # already in scientific notation
            lr = lr_raw
        elif 'p' in lr_raw:
            # convert 0p0001 -> 1e-04
            value = float(lr_raw.replace('p', '.'))
            lr = f"{value:.0e}"

    dim = next((p.replace("dim", "") for p in parts if p.startswith("dim")), "dim?")
    dim = dim.split('x')[0]  # keep only one spatial dimension

    bs = next((p for p in parts if p.startswith("bs")), "bs?")

    return f"{arch} | {dt} | {loss} | {lr} | {dim} | {bs}"
# === Utility: Visualize predicted building mask overlay with confusion colors ===
def Overlay(dataset_name, model_name, image, image_name, pred_mask, true_mask, alpha=0.4):
    """
    Overlay predicted building mask and ground truth on the original image.
    Highlights:
        - True Positive (TP): Green (correctly predicted building pixels)
        - False Positive (FP): Blue (predicted building, but not in ground truth)
        - False Negative (FN): Red (missed building pixels)
        - True Negative (TN): Transparent (optional, background)
    Args:
        image: PIL.Image (RGB)
        pred_mask: numpy array, shape (H,W), binary (0/1 or bool)
        true_mask: numpy array, shape (H,W), binary (0/1 or bool)
        alpha: float, overlay transparency
    """
    if true_mask is not None:
        img_np = np.array(image)
        pred = pred_mask.astype(bool)
        gt = true_mask.astype(bool)

        # Confusion masks
        tp = np.logical_and(pred, gt)
        fp = np.logical_and(pred, np.logical_not(gt))
        fn = np.logical_and(np.logical_not(pred), gt)
        tn = np.logical_and(np.logical_not(pred), np.logical_not(gt))

        # RGBA overlay
        overlay = np.zeros((img_np.shape[0], img_np.shape[1], 4), dtype=np.float32)
        overlay[tp] = [0.0, 1.0, 0.0, alpha]   # Green
        overlay[fp] = [0.0, 0.0, 1.0, alpha]   # Blue
        overlay[fn] = [1.0, 0.0, 0.0, alpha]   # Red
        overlay[tn] = [0.0, 0.0, 0.0, 0.0]     # Transparent

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img_np)
        ax.imshow(overlay)
        ax.set_title(f"{ShortModelName(model_name)}", fontsize=16)
        ax.axis("off")

        legend_elements = [
            patches.Patch(facecolor='green', label='True Positive'),
            patches.Patch(facecolor='blue', label='False Positive'),
            patches.Patch(facecolor='red', label='False Negative'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
        os.makedirs(f'/home/antoniocorvino/Projects/BuildingsExtraction/runs/{dataset_name}/{model_name}/overlay/', exist_ok=True)

        plt.savefig(f'/home/antoniocorvino/Projects/BuildingsExtraction/runs/{dataset_name}/{model_name}/overlay/{image_name}.tif')
        #plt.show()
        plt.close()

def ThreePlot(dataset_name, model_name, image, image_name, pred_mask, true_mask):
    if true_mask is None:
        ncols = 2
    else:
        ncols = 3
    fig, axes = plt.subplots(1, ncols, figsize=(18, 6), constrained_layout=True)

    # --- 1. Input Image ---
    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontsize=16)
    axes[0].axis("off")

    # --- 2. Input + Predicted Mask Overlay ---
    axes[1].imshow(image)
    axes[1].imshow(pred_mask, cmap='jet', alpha=0.5)
    axes[1].contour(pred_mask, colors='white', linewidths=0.5)
    axes[1].set_title(f"{ShortModelName(model_name)}\nInput + Predicted Mask", fontsize=16)
    axes[1].axis("off")

    if true_mask is not None:
        # --- 3. Predicted vs True Mask Overlay ---
        axes[2].imshow(image, alpha=0.8)  # base image

        # Overlay "true" mask in green, semi-transparent
        axes[2].imshow(true_mask, cmap='Blues', alpha=0.7)

        # Overlay predicted mask in red, semi-transparent
        axes[2].imshow(pred_mask, cmap='Reds', alpha=0.5)

        axes[2].set_title("Predicted vs True Mask Overlay", fontsize=16)
        axes[2].axis("off")

        # Optional legend outside the plot for reference
        legend_elements = [
            patches.Patch(color='darkblue', label='Mask (Reference)'),
            patches.Patch(color='darkred', label='Prediction'),
        ]
        axes[2].legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0,
            fontsize=12
        )
    os.makedirs(f'/home/antoniocorvino/Projects/BuildingsExtraction/runs/{dataset_name}/{model_name}/threeplot/', exist_ok=True)

    plt.savefig(f'/home/antoniocorvino/Projects/BuildingsExtraction/runs/{dataset_name}/{model_name}/threeplot/{image_name}.tif')
    #plt.show()
    plt.close()




# --------------------------------------------------
# CLI
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("Building footprint inference")

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--models", nargs="+", required=True,
                        help="List of model run names")

    parser.add_argument("--images", nargs="+", required=True,
                        help="Image base names without extension")

    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps"])

    parser.add_argument("--threshold", type=float, default=0.5)

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
    raise ValueError(f"Unknown model type: {model_name}")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu"
    )

    print(f"\nDevice: {device}")
    print(f"Models: {args.models}")
    print(f"Images: {args.images}\n")

    transform = transforms.ToTensor()

    # --------------------------------------------------
    # Load models ONCE
    # --------------------------------------------------
    models = {}
    for name in args.models:
        start = time.time()

        model = build_model(name).to(device)
        ckpt = f"/home/antoniocorvino/Projects/BuildingsExtraction/runs/{args.dataset_name}/{name}/best_model.pth"

        model.load_state_dict(
            torch.load(ckpt, map_location=device),
            strict=False
        )
        model.eval()

        models[name] = model
        print(f"Loaded {name} in {(time.time()-start):.2f}s")

    # --------------------------------------------------
    # Inference loop
    # --------------------------------------------------
    total_start = time.time()

    for image_name in args.images:

        img_path = (
            f"{args.dataset_path}/{args.dataset_name}/tiles_{args.tile_size}/"
            f"{args.split}/images/{image_name}.tif"
        )
        mask_path = (
            f"{args.dataset_path}/{args.dataset_name}/tiles_{args.tile_size}/"
            f"{args.split}/gt/{image_name}.tif"
        )

        image = Image.open(img_path).convert("RGB")
        true_mask = Image.open(mask_path).convert("L") if os.path.exists(mask_path) else None

        input_tensor = transform(image).unsqueeze(0).to(device)

        print(f"\nImage: {image_name}")
        print(f"Input shape: {input_tensor.shape}")

        for model_name, model in models.items():

            start = time.time()
            with torch.no_grad():
                logits = model(input_tensor)
                prob = torch.sigmoid(logits)
                pred = (prob > args.threshold).float()

            infer_time = time.time() - start

            pred_np = pred.squeeze().cpu().numpy()
            gt_np = np.array(true_mask) > 0 if true_mask else None


            MaskPredict(args.dataset_name, model_name, image_name, pred_np, true_mask)
            Overlay(args.dataset_name, model_name, image, image_name, pred_np, gt_np)
            ThreePlot(args.dataset_name, model_name, image, image_name, pred_np, gt_np)

            print(
                f"  {model_name:<40} "
                f"Inference: {infer_time:.3f}s | "
                f"Foreground ratio: {pred_np.mean():.3f}"
            )

    print(f"\nTotal inference time: {(time.time()-total_start):.2f}s")


if __name__ == "__main__":
    main()




