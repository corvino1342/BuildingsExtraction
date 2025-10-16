import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from unet import UNet1

model_loaded = UNet1

# === Initial Configuration ===

tile_number = ('3_2')

model_path = "runs/unet_Massachusetts.pth"


#image_path = f"dataset/test/images/tile-{tile_number}.png"
#mask_path = f"dataset/test/masks/tile-{tile_number}.png"

image_path = '../BuildingsHeight/datasets/tiles/test/22828930_15_1.tiff'
mask_path = '../BuildingsHeight/datasets/tiles/test_labels/22828930_15_1.tif'

# === Load the model ===
model = model_loaded(n_channels=3, n_classes=1)
model.load_state_dict(torch.load(model_path, map_location=torch.device("mps")))
model.eval()

# === Prepare the image ===
transform = transforms.Compose([
    transforms.ToTensor(),
])

img = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")
input_tensor = transform(img).unsqueeze(0)

# === Inference ===
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.sigmoid(output)
    binary_mask = (prediction > 0.5).float()

# === Visualization ===
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib import patches

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

# --- 3. Difference Map ---
# Positive: False Negative (missed), Negative: False Positive (extra)
diff = mask_np - binary_mask_np
im = axes[2].imshow(diff, cmap='bwr', vmin=-1, vmax=1)
axes[2].set_title("Difference (Real - Predicted)", fontsize=16)
axes[2].axis("off")
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label='Error')

plt.show()