import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from unet import UNet2

model_loaded = UNet2

# === Initial Configuration ===

tile_number = ('0_0')

model_path = "runs/unet_Massachusetts.pth"


image_path = f"dataset/test/images/tile-{tile_number}.png"
mask_path = f"dataset/test/masks/tile-{tile_number}.png"

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
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Real Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Predicted Mask")
plt.imshow(binary_mask.squeeze().numpy(), cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

# === Save the predicted mask ===
from torchvision.utils import save_image

save_path = f"predictions/predicted_mask_{tile_number}.png"
save_image(binary_mask, save_path)
print(f"Predicted mask saved at: {save_path}")
