import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from unet import UNet1
import time
import numpy as np
from matplotlib import colors
from matplotlib import patches


# === Utility: Visualize predicted building mask overlay with confusion colors ===
def visualize_building_overlay(image, pred_mask, true_mask, alpha=0.4):
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
    ax.set_title("Building Prediction Overlay", fontsize=16)
    ax.axis("off")

    legend_elements = [
        patches.Patch(facecolor='green', label='True Positive'),
        patches.Patch(facecolor='blue', label='False Positive'),
        patches.Patch(facecolor='red', label='False Negative'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
    plt.show()


def three_plot(image, pred_mask, true_mask):
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

start = time.time()
model_loaded = UNet1

model = 'unet_AID_WBCE_lr0p0001_n28000_dim256x256_bs32'

model_path = f"/Users/corvino/PycharmProjects/BuildingsExtraction/runs/{model}/checkpoint_30.pth"

C_image_path = '/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/fotoxtest/foto3.png'
MBD_image_path = '/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/MassachusettsBuildingsDataset/tiles/test/images/22828930_15_2.tif'
AID_image_path = '/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/AerialImageDataset/tiles_512/test/images/chicago13_1.tif'
H_image_path = '/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/handmade/test/images/tile-0_1.png'

print(f'Time for Initial Configuration: {(time.time()-start):.3f} s')

# === Load the model ===
start = time.time()

model = model_loaded(n_channels=3, n_classes=1)
model.load_state_dict(torch.load(model_path, map_location=torch.device("mps")))
model.eval()
print(f'Time for Loading the Model: {(time.time()-start):.3f} s')


file_name = 'austin34_10'

img = Image.open(
    f'/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/AerialImageDataset/tiles_512/test/images/{file_name}.tif').convert("RGB")
mask = Image.open(
    f'/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/AerialImageDataset/tiles_512/test/gt/{file_name}.tif').convert("L")
print(f'\n{file_name}\n')

# === Prepare the image ===
start = time.time()

transform = transforms.Compose([transforms.ToTensor(),])

input_tensor = transform(img).unsqueeze(0)
print(f'Time for Image Preparation: {(time.time()-start):.3f} s')

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

#visualize_building_overlay(img, binary_mask_np, true_mask)
three_plot(img, binary_mask_np, true_mask=None)


