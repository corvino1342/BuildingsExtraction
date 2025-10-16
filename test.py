import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from unet import UNet1
import time
import numpy as np
from matplotlib import colors
from matplotlib import patches


# === Initial Configuration ===

start = time.time()
model_loaded = UNet1
model_path = "runs/unet_Massachusetts.pth"


#image_path = '../BuildingsHeight/datasets/tiles/test/22828930_15_1.tiff'
#mask_path = '../BuildingsHeight/datasets/tiles/test_labels/22828930_15_1.tif'

#image_path = f"dataset/test/images/tile-2_3.png"
#mask_path = f"dataset/test/masks/tile-2_3.png"

image_path = '../BuildingsHeight/datasets/massachusetts-buildings-dataset/tiff/test/22828990_15.tiff'
mask_path = '../BuildingsHeight/datasets/massachusetts-buildings-dataset/tiff/test_labels/22828990_15.tif'

print(f'Time for Initial Configuration: {(time.time()-start):.3f} s')

# === Load the model ===
start = time.time()

model = model_loaded(n_channels=3, n_classes=1)
model.load_state_dict(torch.load(model_path, map_location=torch.device("mps")))
model.eval()
print(f'Time for Loading the Model: {(time.time()-start):.3f} s')

# === Prepare the image ===
start = time.time()

transform = transforms.Compose([
    transforms.ToTensor(),
])

img = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")
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
start = time.time()


# Convert masks to numpy
binary_mask_np = binary_mask.squeeze().numpy()
mask_np = np.array(mask) / 255.0  # normalize if mask is 0-255


fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# --- 1. Input Image ---
axes[0].imshow(img)
axes[0].set_title("Input Image", fontsize=16)
axes[0].axis("off")

# --- 2. Input + Predicted Mask Overlay ---
axes[1].imshow(img)
axes[1].imshow(binary_mask_np, cmap='jet', alpha=0.5)
axes[1].contour(binary_mask_np, colors='white', linewidths=1)
axes[1].set_title("Input + Predicted Mask", fontsize=16)
axes[1].axis("off")

# --- 3. Predicted vs True Mask Overlay ---
axes[2].imshow(img, alpha=0.8)  # base image

# Overlay "true" mask in green, semi-transparent
axes[2].imshow(mask_np, cmap='Blues', alpha=0.7)

# Overlay predicted mask in red, semi-transparent
axes[2].imshow(binary_mask_np, cmap='Reds', alpha=0.5)

axes[2].set_title("Predicted vs True Mask Overlay", fontsize=16)
axes[2].axis("off")

# Optional legend outside the plot for reference
import matplotlib.patches as mpatches
legend_elements = [
    mpatches.Patch(color='darkblue', label='Mask (Reference)'),
    mpatches.Patch(color='darkred', label='Prediction'),
]
axes[2].legend(
    handles=legend_elements,
    loc='upper left',
    bbox_to_anchor=(1.05, 1),
    borderaxespad=0,
    fontsize=12
)

print(f'Time for Visualization: {(time.time()-start):.3f} s')

plt.show()
