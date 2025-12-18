import torch
from matplotlib.pyplot import title
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from unet import UNet1
import time
import numpy as np
from matplotlib import colors
from matplotlib import patches
import os

# === Utility: Visualize predicted building mask overlay with confusion colors ===
def Overlay(model_name, image, image_name, pred_mask, true_mask, alpha=0.4):
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
    ax.set_title(f"{model_name}", fontsize=16)
    ax.axis("off")

    legend_elements = [
        patches.Patch(facecolor='green', label='True Positive'),
        patches.Patch(facecolor='blue', label='False Positive'),
        patches.Patch(facecolor='red', label='False Negative'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
    plt.savefig(f'/Users/corvino/PycharmProjects/BuildingsExtraction/predictions/test/{model_name}/{image_name}.tif')
    plt.show()

def ThreePlot(model_name, image, pred_mask, true_mask):
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
    axes[1].set_title("Input + Predicted Mask", fontsize=16)
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

    plt.show()


# === Initial Configuration ===

model_loaded = UNet1

model_names = ['unet_IAD_BCE_lr0p0001_n44800_dim256x256_bs32',
               'unet_IAD_WBCE_lr0p0001_n44800_dim256x256_bs32',
               'unet_MBD_BCE_lr0p0001_n3945_dim256x256_bs32',
               'unet_MBD_WBCE_lr0p0001_n3945_dim256x256_bs32']

# === Prepare the image ===
start = time.time()

image_name = 'austin34_7'

img = Image.open(f'/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/InriaAerialDataset/tiles_512/test/images/{image_name}.tif').convert("RGB")
mask = Image.open(f'/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/InriaAerialDataset/tiles_512/test/gt/{image_name}.tif').convert("L")

print(f'\n{image_name}\n')

transform = transforms.Compose([transforms.ToTensor(), ])

input_tensor = transform(img).unsqueeze(0)
print(f'Time for Image Preparation: {(time.time() - start):.3f} s')

for model_name in model_names:

    model_path = f"/Users/corvino/PycharmProjects/BuildingsExtraction/runs/{model_name}/checkpoint_50.pth"

    # === Load the model ===
    start = time.time()

    model = model_loaded(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("mps")))
    model.eval()
    print(f'Time for Loading the Model: {(time.time()-start):.3f} s')



    # === Inference ===
    start = time.time()

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output)
        binary_mask = (prediction > 0.5).float()
    print(f'Time for Inference: {(time.time()-start):.3f} s')

    # === Visualization ===

    # Convert masks to numpy
    true_mask = np.array(mask) > 0
    binary_mask_np = binary_mask.squeeze().numpy()

    os.makedirs(f'/Users/corvino/PycharmProjects/BuildingsExtraction/predictions/test/{model_name}', exist_ok=True)

    Overlay(model_name, img, image_name, binary_mask_np, true_mask)
    #ThreePlot(model, img, binary_mask_np, true_mask=None)


